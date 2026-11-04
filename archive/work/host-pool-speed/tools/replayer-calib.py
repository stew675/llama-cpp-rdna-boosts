#!/usr/bin/env python3
"""wip/host-pool-speed F2: calibrate `scripts/moe-trace-replay.py` against the delivered cache.

The delivery's replayer underpredicts the run's logged `h=` (HANDOVER.md §4/F2: 0.19 vs 0.349 at
3,932 MiB, 0.642 vs 0.754 at 22,518 MiB).  This tool sweeps the candidate causes as *variants* of one
trace, so the cause is identified by data rather than argument:

  band     the runtime's `h=` is produced by `moe_cache_update_host`'s access path, and the PREFILL band
           returns before it (whole-table staging, or a decline).  The replayer's default (`--score-band
           all`) therefore scores a band the runtime does not score.
  ghost    `access_locked` resets `ghost[expert] = 0` when the expert is ADMITTED; the replayer's
           `ghost` is a lifetime counter, which turns `admit=touch` into "admit everything after the 2nd
           miss ever" instead of "2nd miss since it was last resident".
  churn    the delivery's `--churn off` arm (a miss may take an empty slot but never evicts).
  seed     the arena pre-fill: the persisted sidecar (what the delivery uses) vs none vs an oracle.

Usage: python3 replayer-calib.py <trace> --measured-h 0.349 --arena-mib 3932 --sidecar <path>
Nothing here is part of the delivery; it imports the delivery's replayer.
"""
import argparse
import collections
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPLAYER = os.path.join(REPO, "scripts", "moe-trace-replay.py")


def load_replayer(path=REPLAYER):
    spec = importlib.util.spec_from_file_location("moe_trace_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def access_ghost_reset(replay, orig):
    """`access_locked` with the ghost reset on admission (the real code: moe-expert-cache.cu ~2119)."""
    def access(st, expert, period, admit, touch, allow_cold=True, protect=None, no_evict=False):
        if no_evict:
            return orig(st, expert, period, admit, touch, allow_cold, protect, no_evict)
        st.clock += 1
        replay.decay(st, period)
        if expert in st.expert_slot:
            st.hits += 1
            st.count[expert] += 1
            st.last[expert] = st.clock
            return True
        st.misses += 1
        st.ghost[expert] += 1
        if st.slots <= 0:
            if allow_cold:
                st.cold += 1
            return False
        slot = -1
        for s in range(st.slots):
            if st.slot_expert[s] < 0:
                slot = s
                break
        victim = None
        if slot < 0:
            for s in range(st.slots):
                e = st.slot_expert[s]
                if e < 0 or (protect is not None and e in protect):
                    continue
                if victim is None or st.count[e] < st.count[victim] or \
                        (st.count[e] == st.count[victim] and st.last[e] < st.last[victim]):
                    victim = e
                    slot = s
            if slot < 0:
                if allow_cold:
                    st.cold += 1
                return False
            if allow_cold:
                if admit == "value" and st.ghost[expert] <= st.count[victim]:
                    st.cold += 1
                    return False
                if admit == "touch" and st.ghost[expert] < touch:
                    st.cold += 1
                    return False
        if victim is not None:
            del st.expert_slot[victim]
            st.slot_expert[slot] = -1
            st.evictions += 1
        st.slot_expert[slot] = expert
        st.expert_slot[expert] = slot
        st.fills += 1
        st.count[expert] += 1
        st.last[expert] = st.clock
        st.ghost[expert] = 0            # <-- the real code's "outstanding demand is served" reset
        return True
    return access


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--arena-mib", type=float, default=3932.0)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--measured-h", type=float, default=None)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--admit", default="touch")
    ap.add_argument("--touch", type=int, default=2)
    ap.add_argument("--trace-replayer", default=REPLAYER)
    ap.add_argument("--calib", action="store_true",
                    help="only the two calibration variants (band all vs decode, sidecar seed, churn on)")
    args = ap.parse_args(argv)

    replay = load_replayer(args.trace_replayer)
    orig_access = replay.access

    tables, recs = replay.parse(args.trace, "all")
    if args.devices > 1:
        for t in tables.values():
            t.expert_bytes = t.expert_bytes // args.devices
    passes = segment = replay.segment_passes(recs)
    bands = collections.Counter(r.band for r in recs)
    one = sum(t.expert_bytes for t in tables.values())
    if args.slots is not None:
        uni = args.slots
    else:
        uni = replay.uniform_slots(tables, args.arena_mib * 1048576.0)
    for t in tables.values():
        t.slots = min(uni, t.n_experts)
    arena_bytes = args.arena_mib * 1048576.0

    sidecar = None
    if args.sidecar:
        sidecar = replay.load_heat_sidecar(args.sidecar, band="decode")
        if not sidecar:
            sys.exit("could not read sidecar %s" % args.sidecar)

    print("trace   : %s" % args.trace)
    print("records : %d (%s), %d passes" % (len(recs), ", ".join("%s=%d" % kv for kv in sorted(bands.items())),
                                            len(passes)))
    print("tables  : %d, slot-set %.1f MiB -> %d uniform slots/table at %.0f MiB"
          % (len(tables), one / 1048576.0, uni, args.arena_mib))
    if args.measured_h is not None:
        print("measured: h=%.4f (the run's own number; the model's error is measured-h)" % args.measured_h)
    print()

    rows = []
    bands = ("all", "decode") if args.calib else ("all", "decode")
    churns = (True,) if args.calib else (True, False)
    seed_names = ("sidecar",) if args.calib else (("sidecar", "heat", "none") if sidecar else ("heat", "none"))
    ghosts = (False,) if args.calib else (False, True)
    for band in bands:
        for churn in churns:
            for seed_name in seed_names:
                for ghost in ghosts:
                    replay.access = access_ghost_reset(replay, orig_access) if ghost else orig_access
                    prior = replay.seed_prior(tables, passes, seed_name, sidecar=sidecar,
                                              mass_band="decode")
                    r = replay.score(tables, passes, arena_bytes=arena_bytes, period=args.period,
                                     admit=args.admit, touch=args.touch, policy="shipped",
                                     prior=prior, churn=churn, score_band=band)
                    rows.append((band, "on" if churn else "off", seed_name, "reset" if ghost else "cumul", r))
    replay.access = orig_access

    print("%-6s %-5s %-8s %-6s %8s %8s %10s %10s" %
          ("band", "churn", "seed", "ghost", "h", "err", "MiB/tok", "evict"))
    for band, churn, seed_name, ghost, r in rows:
        err = "" if args.measured_h is None else "%+.3f" % (r["h"] - args.measured_h)
        print("%-6s %-5s %-8s %-6s %8.4f %8s %10.1f %10d" %
              (band, churn, seed_name, ghost, r["h"], err, r["mib_per_token"], r["evictions"]))
    if args.measured_h is not None:
        best = min(rows, key=lambda x: abs(x[4]["h"] - args.measured_h))
        print("\nclosest: band=%s churn=%s seed=%s ghost=%s -> h=%.4f (err %+.4f)" %
              (best[0], best[1], best[2], best[3], best[4]["h"], best[4]["h"] - args.measured_h))


if __name__ == "__main__":
    main()
