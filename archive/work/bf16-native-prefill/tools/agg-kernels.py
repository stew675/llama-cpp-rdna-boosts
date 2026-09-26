#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel_trace.csv by kernel name (total ns, count, mean)."""
import csv, sys, collections

def load(path):
    agg = collections.defaultdict(lambda: [0, 0])  # name -> [total_ns, count]
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                d = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            except (ValueError, KeyError):
                continue
            n = row["Kernel_Name"]
            agg[n][0] += d
            agg[n][1] += 1
    return agg

def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    names = sorted(set(a) | set(b), key=lambda n: -(a.get(n,[0])[0] + b.get(n,[0])[0]))
    ta = sum(v[0] for v in a.values()); tb = sum(v[0] for v in b.values())
    print(f"{'kernel':60s} {'A ms':>10s} {'A n':>6s} {'B ms':>10s} {'B n':>6s} {'d ms':>9s} {'d%':>7s}")
    print(f"{'TOTAL':60s} {ta/1e6:10.1f} {'':6s} {tb/1e6:10.1f} {'':6s} {(tb-ta)/1e6:9.1f} {100*(tb-ta)/ta:6.2f}%")
    for n in names[:35]:
        va, vb = a.get(n, [0,0]), b.get(n, [0,0])
        d = vb[0]-va[0]
        pct = (100*d/va[0]) if va[0] else float('nan')
        print(f"{n[:60]:60s} {va[0]/1e6:10.2f} {va[1]:6d} {vb[0]/1e6:10.2f} {vb[1]:6d} {d/1e6:9.2f} {pct:6.1f}%")

if __name__ == "__main__":
    main()
