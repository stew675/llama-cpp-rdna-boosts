#!/usr/bin/env python3
"""wip/host-pool-speed P1-b screen: does per-table pool SIZING from decode heat mass beat the shipped
uniform rule?  (README.md §4.2, and §9.6's "simulate before implementing".)

The shipped rule gives every host table the same share: `slots = n_experts * FRAC/100`, so the *total*
pool bytes are fixed but the split is uniform, and a table with a long cold tail starves a table with a
flat hot set.  The candidate sizes each table by its **decode heat mass** (`heat_dec` in the delivery; here
the trace's decode-band routed mass) with a one-slot floor, under the same byte budget.

Both arms run through the delivery's own two-tier model (`score_pool`) with the same arena, the same
`--pool-init heat-complement` (the real layout: the pool is the arena's complement, ranked by the prior) and
the same pool policy, so the ONLY difference is the slot split.  The number to read is `L3 MiB/tok` (the
bytes that come from neither tier -- the disk reads the pool exists to remove) and `h`.

Usage: python3 pool-sizing-sim.py <trace> [--arena-mib 22517] [--fracs 25,50,80] [--policy pool-shipped]
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


def uniform_slots_p(tables, frac):
    """Exactly the shipped rule (`table_pool_locked`): n_experts * FRAC/100, floor 1."""
    return {k: max(1, min(int(t.n_experts * frac / 100), t.n_experts)) for k, t in tables.items()}


def budget_of(tables, slots_p):
    return sum(slots_p[k] * t.expert_bytes for k, t in tables.items())


def heat_slots_p(tables, frac, min_slots=1):
    """The candidate: each table's slots proportional to its decode heat mass, same total byte budget.

    `slots_i = clamp(alpha * mass_i, min_slots, n_experts_i)`, alpha solved so the byte total matches the
    shipped rule's.  `mass_i` is the table's decode routed mass (sum of its experts' trace counts).
    """
    budget = budget_of(tables, uniform_slots_p(tables, frac))
    mass = {k: float(sum(t.freq.values())) for k, t in tables.items()}
    if not any(mass.values()):
        return uniform_slots_p(tables, frac)

    def alloc(alpha):
        out = {}
        for k, t in tables.items():
            s = int(alpha * mass[k])
            out[k] = min(t.n_experts, max(min_slots, s))
        return out

    def total(a):
        return sum(a[k] * tables[k].expert_bytes for k in a)

    lo, hi = 0.0, 1.0
    while total(alloc(hi)) <= budget and hi < 1e12:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(alloc(mid)) <= budget:
            lo = mid
        else:
            hi = mid
    a = alloc(lo)
    # Hand the rounding leftovers to the hottest tables while they fit (largest mass first), then, if the
    # floors overshot the budget, take slots back from the coldest until it fits.
    for k in sorted(tables, key=lambda k: -mass[k]):
        while a[k] < tables[k].n_experts and total(a) + tables[k].expert_bytes <= budget:
            a[k] += 1
    for k in sorted(tables, key=lambda k: mass[k]):
        while total(a) > budget and a[k] > min_slots:
            a[k] -= 1
    return a


def equal_bytes_slots_p(tables, frac):
    """Candidate: every table gets the same BYTE budget rather than the same slot count.

    The shipped rule gives table i `n_experts * FRAC/100` slots, so a table whose experts are 2.5 MiB
    (ffn_down_exps) gets 2.5x the pinned bytes of a 1.7 MiB one (ffn_gate/up_exps) -- the byte budget is
    uniform per EXPERT, not per TABLE.  This one equalises the bytes and therefore cuts the big tables'
    slot count.
    """
    budget = budget_of(tables, uniform_slots_p(tables, frac))
    per_table = budget / float(len(tables))
    return {k: min(t.n_experts, max(1, int(round(per_table / float(t.expert_bytes)))))
            for k, t in tables.items()}


def coverage_slots_p(tables, frac, target=0.90):
    """Candidate: size each table by its own CONCENTRATION -- the slots needed to hold `target` of its mass.

    This is the only per-table signal that is not degenerate: a top-k router hands every table exactly
    `n_used` experts per token, so a table's TOTAL routed mass is a constant (`mass_i = n_used * steps`)
    and "allocate by heat mass" collapses to the uniform rule.  What differs between tables is the shape
    of the distribution, not its total.
    """
    budget = budget_of(tables, uniform_slots_p(tables, frac))
    want = {}
    for k, t in tables.items():
        tot = float(sum(t.freq.values()))
        if tot <= 0:
            want[k] = 1
            continue
        cum, k_i = 0.0, 0
        for e, c in sorted(t.freq.items(), key=lambda kv: -kv[1]):
            cum += c
            k_i += 1
            if cum / tot >= target:
                break
        want[k] = min(t.n_experts, max(1, k_i))

    def alloc(alpha):
        return {k: min(t.n_experts, max(1, int(alpha * want[k]))) for k, t in tables.items()}

    def total(a):
        return sum(a[k] * tables[k].expert_bytes for k in a)

    lo, hi = 0.0, 1.0
    while total(alloc(hi)) <= budget and hi < 1e12:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if total(alloc(mid)) <= budget:
            lo = mid
        else:
            hi = mid
    return alloc(lo)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--arena-mib", type=float, default=22517.0)
    ap.add_argument("--fracs", default="25,50,80")
    ap.add_argument("--policy", default="pool-shipped")
    ap.add_argument("--pool-init", default="heat-complement")
    ap.add_argument("--trace-replayer", default=REPLAYER)
    args = ap.parse_args(argv)

    replay = load_replayer(args.trace_replayer)
    tables, recs = replay.parse(args.trace, "decode")     # t.freq = the decode band (what the arena serves)
    passes = replay.segment_passes(recs)
    one = sum(t.expert_bytes for t in tables.values())
    slots_a = max(0, int(args.arena_mib * 1048576.0 / one)) if one else 0
    for t in tables.values():
        t.slots = min(slots_a, t.n_experts)

    print("trace  : %s" % args.trace)
    print("tables : %d, slot-set %.1f MiB, arena %.0f MiB -> %d slots/table"
          % (len(tables), one / 1048576.0, args.arena_mib, slots_a))
    print("policy : %s, pool init %s, decode band only" % (args.policy, args.pool_init))
    print()

    hdr = "%-6s %-9s %6s %9s %9s %11s %9s %10s" % ("frac", "sizing", "slots", "pool MiB", "h", "pool_h",
                                                   "L3 MiB/tk", "handover")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for frac in [int(x) for x in args.fracs.split(",") if x.strip()]:
        cands = (("uniform", uniform_slots_p(tables, frac)),
                 ("heat-mass", heat_slots_p(tables, frac)),
                 ("eq-bytes", equal_bytes_slots_p(tables, frac)),
                 ("cov90", coverage_slots_p(tables, frac, 0.90)))
        for name, sp in cands:
            r = replay.score_pool(tables, passes, slots_a, sp, args.policy, pool_init=args.pool_init,
                                  decode_only=True)
            out[(frac, name)] = r
            print("%-6d %-9s %6.1f %9.1f %9.4f %11.4f %9.2f %10.1f"
                  % (frac, name, sum(sp.values()) / float(len(sp)), r["pool_resident_mib"], r["h"],
                     r["pool_h"], r["l3_mib_per_tok"], r["handover_mib"]))
        u = out[(frac, "uniform")]
        for name, _ in cands[1:]:
            h = out[(frac, name)]
            dl3 = (h["l3_mib_per_tok"] / u["l3_mib_per_tok"] - 1.0) * 100.0 if u["l3_mib_per_tok"] else 0.0
            print("       -> %-9s vs uniform: L3 %+.1f %%, h %+.4f, pool_h %+.4f %s"
                  % (name, dl3, h["h"] - u["h"], h["pool_h"] - u["pool_h"],
                     "*** SEPARATES ***" if abs(dl3) >= 5.0 else "(not separated)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
