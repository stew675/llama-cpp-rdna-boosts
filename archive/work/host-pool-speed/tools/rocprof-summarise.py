#!/usr/bin/env python3
"""Summarise two `rocprofv3 --kernel-trace` CSVs into the per-kernel comparison used by
`wip/host-pool-speed/README.md` §15.1.

  rocprofv3 --kernel-trace -f csv -o k-pool -- <llama-cli pool arm>
  rocprofv3 --kernel-trace -f csv -o k-pinned -- <llama-cli pinned arm (GGML_MOE_DEVREMAP=0)>
  ./rocprof-summarise.py k-pool_kernel_trace.csv k-pinned_kernel_trace.csv

Why this exists: the residual decode gap on the pool arm is GPU work, not host work, so the instrument that
localises it is a per-kernel duration compare at *identical dispatch counts*.  Two traps, both measured:
  * a traced run's achieved t/s is meaningless (13.3 vs 27.2 t/s on the same two arms untraced 26 / 32) --
    do not quote a t/s from a traced run, only durations;
  * the CSVs are 40-75 MB each.  Do NOT commit them; this script regenerates the table from a fresh pair.
Use a short prompt and `-n 128` so the trace is pure decode (a prefill in the trace floods it with
`moe_cache_stage_kernel` and hides the decode-band kernels).
"""
import csv
import collections
import sys


def load(path):
    agg = collections.Counter()
    cnt = collections.Counter()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                d = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if d < 0:
                continue
            agg[r["Kernel_Name"]] += d
            cnt[r["Kernel_Name"]] += 1
    return agg, cnt


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    a, ca = load(argv[1])
    b, cb = load(argv[2])
    ta, tb = sum(a.values()), sum(b.values())
    print(f"pool   total {ta / 1e9:7.3f} s  {sum(ca.values()):7d} dispatches   ({argv[1]})")
    print(f"pinned total {tb / 1e9:7.3f} s  {sum(cb.values()):7d} dispatches   ({argv[2]})")
    print(f"GPU-time ratio {ta / max(tb, 1):.3f}")
    print()
    hdr = f"{'kernel':58s} {'pool s':>8s} {'pin s':>8s} {'n pool':>7s} {'n pin':>7s}  pool/pin us"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))[:16]:
        pa = a[k] / max(ca[k], 1) / 1000.0
        pb = b[k] / max(cb[k], 1) / 1000.0
        flag = "  <-- SAME COUNT, different duration" if ca[k] == cb[k] and pb > 0 and pa > 1.3 * pb else ""
        print(f"{k[:58]:58s} {a[k] / 1e9:8.3f} {b[k] / 1e9:8.3f} {ca[k]:7d} {cb[k]:7d}  "
              f"{pa:7.1f} / {pb:6.1f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
