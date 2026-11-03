#!/usr/bin/env python3
"""Price the locked window against the pool on a REAL routing trace (campaign P1).

The window is the pool's *storage*, not a new policy (DESIGN.md §1: a second LRU over the window was
measured at 118 hits in 11 776 promotes).  So this tool does not simulate a policy at all: it runs
`scripts/moe-trace-replay.py`'s existing two-tier model (arena + pool, the delivered admission/eviction
policy) and then prices the SAME event stream two ways:

    pool    a grant is a host `memcpy` page-cache -> pinned slot, then a copy-engine H2D into the arena
    window  a grant is a device mapping over the range (which faults it in and holds it -- see MEASUREMENTS
            "registration is an eager hold"), and the bytes are then read IN PLACE: no host copy at all

The per-event costs come from `probe-locked-window.cpp` (see MEASUREMENTS.md) and the prior campaign's
probes; every constant is a command-line default so a re-measurement is one flag away.  The output is a
PER-DECODE-TOKEN cost, which is the number that decides whether the mode is worth building:

    * `l3` misses are the disk tier, the one both modes share (and the one the maintainer's charter names
      as the pain).  The window's advantage is that its cold grant is the disk read itself, batched through
      GUP, instead of a page-by-page fault inside a `memcpy`.

Caveat, do not quote without it: the trace carries routing, not timing.  The arena hit rate in this model is
the delivered policy's, and the constant that dominates the comparison (`memcpy`/CE vs grant) is measured on
ONE box (1x R9700, PCIe 5.0 x16).  A different link moves both arms, and a slow host moves the `memcpy` up.

usage:
  ./sim-locked-window.py TRACE [--arena-mib 4096] [--window-mib 1024 2048 4096] [--devices 1]
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_replayer():
    path = os.path.join(REPO, "scripts", "moe-trace-replay.py")
    spec = importlib.util.spec_from_file_location("moe_trace_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- measured constants (µs per MiB of expert bytes) -------------------------------------------------------
# probe-locked-window, 2026-11-03, 1x R9700, PCIe 5.0 x16; see MEASUREMENTS.md for the raw arms.
COSTS = {
    # a pool grant: host memcpy out of the page cache, then a CE copy pool -> arena
    "pool_memcpy_hot":   30.2,    # 50.1 us / 1.66 MiB
    "pool_memcpy_cold":  2125.0,  # 17.0 ms / 8 MiB (prior campaign, cold, 0/2048 pages resident)
    "ce_h2d":            40.0,    # 66.4 us / 1.66 MiB
    # a window grant: hipHostRegister over the range, which faults it in and holds it (no mlock needed)
    "grant_hot":          9.85,   # 16.35 us / 1.66 MiB, resident
    "grant_cold":        702.0,   # 1166 us / 1.66 MiB, provably non-resident (the disk read)
    "unregister":         0.5,    # per grant (0.49 us), one per eviction
    # the device side, in place (same for both arms once the bytes are readable)
    "sm_read":           34.0,    # 68.8 us for a 1.66 MiB read+write: ~half the link time is the read
}


def price(events, arm, c, cold_pagecache=0.5):
    """`cold_pagecache` is the fraction of L3 misses that are served from the PAGE CACHE rather than the
    disk.  It is the single biggest modelling assumption: on a box where the host set fits in RAM the
    buffered fill is a memcpy (page-cache hit) and the disk constant never applies; where it does not fit
    (`host-pool-speed`'s 122B on a 64 GiB class host) it does.  Default 0.5 = the campaign's reference arm
    (73 GiB model, 64 GiB emulated), and the number is printed so a reader can re-run it."""
    """Return (host_us, device_us, link_mib) for one arm, from the replayer's event bytes."""
    fills_hot  = events["promote_mib"] + events["handover_mib"]
    fills_cold = events["l3_mib"]
    read_hot   = events["p_cold_mib"]          # pool-resident, read on a miss (MOE_HOST_POOL_COLD)
    consumes   = fills_hot + fills_cold + read_hot + events["pool_fill_from_arena_mib"]

    cold_disk = fills_cold * (1.0 - cold_pagecache)
    cold_cache = fills_cold * cold_pagecache
    if arm == "pool":
        host = (fills_hot * (c["pool_memcpy_hot"] + c["ce_h2d"]) +
                cold_cache * (c["pool_memcpy_hot"] + c["ce_h2d"]) +
                cold_disk * (c["pool_memcpy_cold"] + c["ce_h2d"]) +
                read_hot * c["ce_h2d"])
        device = consumes * c["ce_h2d"] + consumes * c["sm_read"]
        # The pool's `memcpy` is host RAM -> host RAM (not link traffic); only the slot -> VRAM CE copy
        # crosses the link.  Corrected 2026-11-03: the earlier `* 2` mislabelled the host memcpy as link.
        link_mib = consumes
    else:
        host = (fills_hot * c["grant_hot"] +
                cold_cache * c["grant_hot"] +
                cold_disk * c["grant_cold"] +
                reads_unregistered_grant_evictions(events, c))
        device = consumes * c["sm_read"]
        link_mib = consumes             # the GPU reads the mapping in place, once
    return host, device, link_mib


def reads_unregistered_grant_evictions(events, c):
    return events["evictions"] * c["unregister"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--arena-mib", type=float, default=4096.0, help="per-device VRAM arena budget")
    ap.add_argument("--window-mib", type=float, nargs="+", default=[1024, 2048, 4096, 8192],
                    help="held-byte budgets to sweep (the pool's MOE_HOST_POOL_MIB, same shape)")
    ap.add_argument("--policy", default="shipped")
    ap.add_argument("--pool-init", default="heat", choices=("heat", "none", "id", "sidecar"),
                    help="'heat' is the ORACLE prior: an upper bound, not the achievable arm")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--costs", metavar="JSON", help="override the measured constants (a dict of the COSTS keys)")
    ap.add_argument("--cold-pagecache", type=float, default=0.5,
                    help="fraction of L3 misses that are page-cache hits, not disk (default 0.5)")
    args = ap.parse_args(argv)

    costs = dict(COSTS)
    if args.costs:
        with open(args.costs) as f:
            costs.update(json.load(f))

    R = load_replayer()
    trace = args.trace
    tmpdir = None
    if trace.endswith(".gz"):
        import gzip, tempfile
        tmpdir = tempfile.mkdtemp(prefix="lw-sim-")
        trace = os.path.join(tmpdir, os.path.basename(args.trace)[:-3])
        with gzip.open(args.trace, "rb") as fi, open(trace, "wb") as fo:
            while True:
                b = fi.read(1 << 22)
                if not b:
                    break
                fo.write(b)
        print("trace        (decompressed %s -> %s)" % (args.trace, trace))
    tables, recs = R.parse(trace, mass_band="decode")
    passes = R.segment_passes(recs)
    arena_slots = R.uniform_slots(tables, args.arena_mib * 1048576.0)
    n_tables = len(tables)
    if n_tables == 0:
        sys.exit("the trace has no tables (a fully-resident run traces no decode band; see the replayer header)")

    print("trace        %s" % args.trace)
    print("tables       %d   passes %d   arena %.0f MiB/device" % (n_tables, len(passes), args.arena_mib))
    print("costs        " + "  ".join("%s=%.1f" % (k, costs[k]) for k in
                                      ("pool_memcpy_hot", "grant_hot", "grant_cold", "ce_h2d")) + " us/MiB")
    print()
    hdr = ("window", "slots/tbl", "held MiB", "pool host/tok", "window host/tok", "delta", "l3 MiB/tok",
           "window grant/tok", "window cold/tok")
    print("%9s %9s %9s %14s %16s %8s %11s %16s %16s" % hdr)
    out = []
    for wmib in args.window_mib:
        per_table_mib = wmib / n_tables
        slots_p = {}
        for key, t in tables.items():
            emib = max(1e-9, t.expert_bytes / 1048576.0)
            slots_p[key] = max(0, min(t.n_experts, int(per_table_mib // emib)))
        res = R.score_pool(tables, passes, arena_slots, slots_p, args.policy, pool_init=args.pool_init)
        tokens = max(1, res["tokens"])
        # `score_pool` reports the promotions (MiB) and pool-resident cold reads (a count); price the count
        # with the trace's average expert blob (per-table blobs are within 1 % on a real MoE trace).
        avg_mib = (sum(t.expert_bytes for t in tables.values()) / max(1, len(tables))) / 1048576.0
        events = {
            "promote_mib": res["promote_mib"],
            "handover_mib": res["handover_mib"],
            "l3_mib": res["l3_mib"],
            "p_cold_mib": res["pool_cold"] * avg_mib,
            "pool_fill_from_arena_mib": 0.0,   # already inside promote_mib (the replayer's accounting)
            "evictions": res["evictions"],
        }
        pool_h, pool_d, pool_link = price(events, "pool", costs, args.cold_pagecache)
        win_h, win_d, win_link = price(events, "window", costs, args.cold_pagecache)
        row = {
            "window_mib": wmib,
            "slots_per_table": int(sum(slots_p.values()) / max(1, len(slots_p))),
            "held_mib": round(sum(v * tables[k].expert_bytes for k, v in slots_p.items()) / 1048576.0, 1),
            "tokens": tokens,
            "pool_host_us_per_tok": round(pool_h / tokens, 1),
            "window_host_us_per_tok": round(win_h / tokens, 1),
            "delta_us_per_tok": round((win_h - pool_h) / tokens, 1),
            "pool_link_mib_per_tok": round(pool_link / tokens, 2),
            "window_link_mib_per_tok": round(win_link / tokens, 2),
            "l3_mib_per_tok": round(res["l3_mib"] / tokens, 2),
            "grants_per_tok": round((res["promote_mib"] + res["handover_mib"] + res["l3_mib"]) /
                                    max(1e-9, avg_mib) / tokens, 1),
            "cold_grants_per_tok": round(res["l3_mib"] / max(1e-9, avg_mib) / tokens, 2),
            "h": round(res["h"], 4),
            "cold_pagecache": args.cold_pagecache,
        }
        out.append(row)
        print("%9.0f %9d %9.1f %14.1f %16.1f %+8.1f %11.2f %16.1f %16.2f" % (
            row["window_mib"], row["slots_per_table"], row["held_mib"], row["pool_host_us_per_tok"],
            row["window_host_us_per_tok"], row["delta_us_per_tok"], row["l3_mib_per_tok"],
            row["grants_per_tok"], row["cold_grants_per_tok"]))
    print()
    print("host µs/decode-token is HOST CPU time on the token's critical path; the device/link column is")
    print("the one CE transfer either arm pays (the pool's memcpy is host RAM -> host RAM, not link).")
    print("A negative delta means the window is cheaper.  `l3 MiB/tok` is the disk tier both modes share.")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("json: %s" % args.json)
    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
