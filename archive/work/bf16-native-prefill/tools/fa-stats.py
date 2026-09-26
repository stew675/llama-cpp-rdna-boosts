#!/usr/bin/env python3
"""Per-dispatch stats for kernels matching a regex from a rocprofv3 kernel_trace.csv."""
import csv, sys, re, collections
path, pat = sys.argv[1], sys.argv[2]
rx = re.compile(pat)
rows = collections.defaultdict(list)
meta = {}
with open(path, newline='') as f:
    for r in csv.DictReader(f):
        if not rx.search(r["Kernel_Name"]):
            continue
        d = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        key = (r["Kernel_Name"][:58], r["VGPR_Count"], r["SGPR_Count"], r["Grid_Size_X"], r["Workgroup_Size_X"], r["LDS_Block_Size"])
        rows[r["Kernel_Name"][:58]].append(d)
        meta[(r["Kernel_Name"][:58], r["VGPR_Count"], r["SGPR_Count"], r["Grid_Size_X"], r["Workgroup_Size_X"], r["LDS_Block_Size"])] = 1
for name, ds in sorted(rows.items(), key=lambda kv: -sum(kv[1])):
    ds.sort()
    n = len(ds)
    print(f"{name}\n  n={n} total={sum(ds)/1e6:.1f}ms mean={sum(ds)/n/1e3:.1f}us "
          f"min={ds[0]/1e3:.1f} med={ds[n//2]/1e3:.1f} max={ds[-1]/1e3:.1f}")
for k in sorted(meta): print("  meta:", k)
