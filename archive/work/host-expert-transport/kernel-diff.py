#!/usr/bin/env python3
"""Three-way per-kernel diff of rocprofv3 `--kernel-trace` CSVs, for the two-transport comparison in
`../SPIKE.md` / `../LEAD-pool-staging-gap.md`.

    ./kernel-diff.py DIR...          (each DIR holds trace_kernel_trace.csv + run.log)

Why this exists: a traced run's t/s is meaningless (the archive's rule), and a total-kernel-time compare
hides the two things that matter here:
  * a kernel with the SAME dispatch count but a different duration in two arms -- same work, different
    speed (contention or a different source), and
  * a kernel that only exists (or is only called N times more) in one arm -- extra work.
So: print the per-kernel total, the count, the us/call, and flag both cases against the FIRST arm given.
"""
import csv
import collections
import re
import sys


def load(d):
    agg, cnt = collections.Counter(), collections.Counter()
    with open(d + "/trace_kernel_trace.csv", newline="") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if t < 0:
                continue
            agg[r["Kernel_Name"]] += t
            cnt[r["Kernel_Name"]] += 1
    log = open(d + "/run.log", errors="replace").read()
    m = re.search(r"prompt eval time =\s*([0-9.]+) ms", log)
    wall = float(m.group(1)) / 1000.0 if m else 0.0
    return agg, cnt, wall


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    arms = [(d, *load(d)) for d in argv[1:]]
    names = sorted({k for _, a, _, _ in arms for k in a},
                   key=lambda k: -sum(a.get(k, 0) for _, a, _, _ in arms))
    hdr = f"{'kernel':50s}"
    for d, _, _, _ in arms:
        hdr += f" {d.split('/')[-1][:14]:>14s}"
    print(hdr)
    print("-" * len(hdr))
    for d, a, c, w in arms:
        print(f"{'TOTAL kernel s':50s} {a and sum(a.values()) / 1e9:14.2f}  ", end="")
    print()
    for d, a, c, w in arms:
        tot = sum(a.values()) / 1e9
        print(f"{'wall (prompt eval) s / kernel-over-wall':50s} {w:14.2f}  ({tot / w:.2f}x)", end="")
    print("\n")
    for k in names[:22]:
        line = f"{k[:50]:50s}"
        for i, (_, a, c, _) in enumerate(arms):
            line += f" {a[k] / 1e9:9.3f}/{c[k]:<4d}" if a[k] else f" {'-':>14s}"
        base_a, base_c = arms[0][1], arms[0][2]
        flags = []
        for (d, a, c, _), name in zip(arms[1:], [x[0].split('/')[-1] for x in arms[1:]]):
            if c[k] == base_c[k] and base_c[k] and a[k] > 1.25 * base_a[k]:
                flags.append(f"{name}: same count, {a[k] / max(base_a[k], 1):.2f}x slower")
            elif a[k] and not base_a[k]:
                flags.append(f"{name}: only here")
            elif a[k] > 1.5 * base_a[k] and c[k] > 1.5 * base_c[k]:
                flags.append(f"{name}: {c[k] / max(base_c[k], 1):.1f}x the calls")
        print(line + ("   <-- " + "; ".join(flags) if flags else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
