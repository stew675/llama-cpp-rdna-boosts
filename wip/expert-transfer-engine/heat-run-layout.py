#!/usr/bin/env python3
"""Offline: does a RUN-CONTIGUOUS resident set cost mass against the uniform heat set?

Reads the delivery's persisted heat map (`<base>.moe-heatmap`, format MHT2 -- see
`moe_cache_heat_map_save_locked` in `ggml/src/ggml-cuda/moe-expert-cache.cu`) and answers the question
`wip/expert-transfer-engine/README.md` S4 (Direction B) / S6.6 leaves open, without a GPU:

  for a fixed slot budget K per table,
    * today's seed  = the K hottest experts (heat-ranked, adaptive) -> the complement is SCATTERED
    * the candidate = R contiguous RUNS totalling K slots          -> the complement is R+1 runs
  what does the run-contiguous layout cost in covered routing mass, and how many DMA blits does a
  pruned fill need in each case?

The blit count is the whole point: the staged fill issues one copy per expert today (512/table on
Flash-Next), so a fill that could be expressed as a handful of RUN blits would keep the copy engine
(single-engine, no SM occupancy) while pruning the arena-resident bytes.

Measures only what the prior supports: MASS COVERAGE (a proxy for hit rate -- judge by the delta, as
`MOE_EXPERT_TRACE` / `scripts/moe-trace-replay.py` warn) and run/blit counts (exact).  It does not
model eviction or the decode band's churn.

Usage:
    heat-run-layout.py MODEL.moe-heatmap [--frac 0.434] [--runs 2,4,8,16,32] [--band dec|all]
"""
import argparse
import struct
import sys

HDR = 32          # magic[4] u32 version u32 n_tables u32 r0 u64 model_bytes u32 n_tensors u32 r1
REC = 80          # int32 layer; char role[64]; u32 n_experts; u64 host_bytes  (align 8)


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HDR:
        sys.exit(f"{path}: too short to be a heat map")
    magic, version, n_tables, _r0, model_bytes, n_tensors, _r1 = struct.unpack_from("<4sIIIQII", data, 0)
    if magic not in (b"MHT2", b"MHT1"):
        sys.exit(f"{path}: bad magic {magic!r} (expected MHT2/MHT1)")
    off = HDR
    tables = []
    for _ in range(n_tables):
        layer, role, n_exp, host_bytes = struct.unpack_from("<i64sIQ", data, off)
        off += REC
        role = role.split(b"\0", 1)[0].decode("ascii", "replace")
        allv = struct.unpack_from("<%dI" % n_exp, data, off)
        off += 4 * n_exp
        decv = ()
        if version >= 2 and off + 4 * n_exp <= len(data):
            decv = struct.unpack_from("<%dI" % n_exp, data, off)
            off += 4 * n_exp
        tables.append(dict(layer=layer, role=role, n=n_exp, host_bytes=host_bytes, all=allv, dec=decv))
    return version, model_bytes, n_tensors, tables


def rank_vec(t, band):
    """The arena ranks by the DECODE counter when it has mass (heat_rank_vec in the delivery)."""
    if band == "dec" and t["dec"] and any(t["dec"]):
        return t["dec"]
    return t["all"]


def runs_of(mask):
    """Number of maximal runs of True (a run = one DMA blit over adjacent experts)."""
    n = 0
    prev = 0
    for v in mask:
        if v and not prev:
            n += 1
        prev = v
    return n


def top_k_mask(v, k):
    order = sorted(range(len(v)), key=lambda e: (-v[e], e))
    m = [0] * len(v)
    for e in order[:k]:
        m[e] = 1
    return m


LENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def run_windows(v, k, r):
    """<= R disjoint contiguous runs, <= k slots, greedily maximum mass.

    Selection is by mass DENSITY (the fill's op count is the number of runs, not their length, so a
    policy forced to use R equal windows under-uses where mass concentrates), then each run is GROWN
    into its neighbours until the slot budget is spent -- otherwise the arm covers almost nothing and the
    comparison is not at equal slots.
    """
    n = len(v)
    if r <= 0 or k <= 0:
        return [0] * n
    pre = [0] * (n + 1)
    for i, x in enumerate(v):
        pre[i + 1] = pre[i] + x
    cand = []
    for L in LENS:
        if L > k:
            continue
        for s0 in range(0, n - L + 1):
            m = pre[s0 + L] - pre[s0]
            cand.append((m / L, m, s0, L))
    cand.sort(key=lambda c: (-c[0], -c[1], c[2]))
    taken = [0] * n
    runs = []          # [lo, hi) per selected run
    used = 0
    for dens, m, s0, L in cand:
        if len(runs) >= r or used + L > k:
            continue
        if any(taken[s0:s0 + L]):
            continue
        for e in range(s0, s0 + L):
            taken[e] = 1
        runs.append([s0, s0 + L])
        used += L
        if used == k:
            break
    while used < k and runs:                       # spend the slack by growing runs
        best = None
        for i, (lo, hi) in enumerate(runs):
            if lo > 0 and not taken[lo - 1] and (best is None or v[lo - 1] > best[0]):
                best = (v[lo - 1], i, lo - 1, lo, hi)
            if hi < n and not taken[hi] and (best is None or v[hi] > best[0]):
                best = (v[hi], i, hi, lo, hi + 1)
        if best is None:
            break
        _, i, e, lo, hi = best
        taken[e] = 1
        runs[i] = [lo, hi]
        used += 1
    return taken


def run_window_seq(v, k, r):
    """R EQUAL-length windows -- the pessimistic bound (printed for reference)."""
    n = len(v)
    L = max(1, k // r) if r > 0 else 0
    if L == 0:
        return [0] * n
    cand = sorted(range(0, n - L + 1), key=lambda s: (-sum(v[s:s + L]), s))
    taken = [0] * n
    used = 0
    for s in cand:
        if used + L > k:
            break
        if any(taken[s:s + L]):
            continue
        for e in range(s, s + L):
            taken[e] = 1
        used += L
    return taken


def mass_frac(v, mask):
    tot = sum(v)
    if tot == 0:
        return 0.0
    return sum(x for x, m in zip(v, mask) if m) / tot


def analyse(tables, frac, n_runs_list, band):
    # aggregate over tables, weighted by mass (every table gets the same slot budget)
    rows = []
    for r in n_runs_list:
        agg = dict(num=0.0, den=0.0, comp=0, res=0, ntab=0)
        rows.append((r, agg))
    uni = dict(num=0.0, den=0.0, comp=0, res=0, ntab=0)
    for r, agg in rows:
        agg.update(den=0.0, dnum=0.0, dcomp=0, dres=0, enum=0.0, ecomp=0, eres=0)
    for t in tables:
        v = rank_vec(t, band)
        n = t["n"]
        if len(v) != n or sum(v) == 0:
            continue
        k = max(1, int(round(frac * n)))
        tot = sum(v)
        mu = top_k_mask(v, k)
        uni["num"] += sum(x for x, m in zip(v, mu) if m)
        uni["den"] += tot
        uni["comp"] += runs_of([0 if m else 1 for m in mu])
        uni["res"] += runs_of(mu)
        uni["ntab"] += 1
        for r, agg in rows:
            for pol, key in ((run_windows, "d"), (run_window_seq, "e")):
                mr = pol(v, k, r)
                agg[key + "num"] += sum(x for x, m in zip(v, mr) if m)
                agg[key + "comp"] += runs_of([0 if m else 1 for m in mr])
                agg[key + "res"] += runs_of(mr)
            agg["den"] += tot
            agg["ntab"] += 1
    return uni, rows


def pct(a, k="num"):
    return 100.0 * a[k] / a["den"] if a["den"] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("heatmap")
    ap.add_argument("--frac", type=float, default=0.434,
                    help="slots per table as a fraction of its expert count (field census: 222/512)")
    ap.add_argument("--runs", default="2,4,8,16,32")
    ap.add_argument("--band", default="dec", choices=["dec", "all"])
    ap.add_argument("--sweep", action="store_true", help="sweep the fraction instead of one value")
    args = ap.parse_args()

    version, model_bytes, n_tensors, tables = parse(args.heatmap)
    n_runs = [int(x) for x in args.runs.split(",") if x.strip()]
    live = [t for t in tables if len(rank_vec(t, args.band)) == t["n"] and sum(rank_vec(t, args.band)) > 0]
    print(f"{args.heatmap}")
    print(f"  format v{version}, {len(tables)} heat tables, {len(live)} with mass")
    if live:
        e = live[0]["n"]
        print(f"  geometry: {e} experts/table (layers x roles = {len(live)})")
    print()

    fracs = [0.25, args.frac, 0.50, 0.75, 0.90, 0.99] if args.sweep else [args.frac]
    for frac in fracs:
        uni, rows = analyse(tables, frac, n_runs, args.band)
        if not uni["ntab"]:
            continue
        k = max(1, int(round(frac * live[0]["n"])))
        print(f"  frac {frac:.3f}  ({k} slots/table, band={args.band})")
        print(f"    {'policy':<20} {'mass%':>7} {'d-mass':>7} {'fill blits/tbl':>15} {'D2D blits/tbl':>14} {'ops/tbl':>8}")
        u_ops = (uni["comp"] + uni["res"]) / uni["ntab"]
        print(f"    {'uniform heat top-K':<20} {pct(uni):7.2f} {'--':>7} {uni['comp']/uni['ntab']:15.1f} "
              f"{uni['res']/uni['ntab']:14.1f} {u_ops:8.1f}")
        for r, agg in rows:
            n_t = agg["ntab"]
            for tag, key in (("runs R=%-2d dens" % r, "d"), ("runs R=%-2d equal" % r, "e")):
                ops = (agg[key + "comp"] + agg[key + "res"]) / n_t
                print(f"    {tag:<20} {pct(agg, key + 'num'):7.2f} "
                      f"{pct(agg, key + 'num') - pct(uni):+7.2f} "
                      f"{agg[key + 'comp']/n_t:15.1f} {agg[key + 'res']/n_t:14.1f} {ops:8.1f}")
        print()


if __name__ == "__main__":
    main()
