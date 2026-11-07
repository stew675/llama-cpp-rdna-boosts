#!/usr/bin/env python3
import sys, os
import numpy as np
from analyze import load, N_ROWS, ROW_BYTES, PAGE, RPP, N_PAGES, N_HEADS

names = ['code80k','prose','python','json','threejs','zh']
d = {n: load(f'/tmp/ple/{n}.bin') for n in names}

def flat(n):
    return d[n]['rows'].reshape(-1)

def heat_map(vecs):
    return np.unique(np.concatenate(vecs), return_counts=True)

def cov_at(heat_u, heat_c, target, mib):
    order = np.argsort(-heat_c)
    k = min(int(mib*2**20//ROW_BYTES), order.size)
    hot = np.sort(heat_u[order[:k]])
    # count how many of target's accesses land in hot
    idx = np.searchsorted(hot, target)
    idx = np.clip(idx, 0, hot.size-1)
    return 100.0 * (hot[idx] == target).mean(), k

print("### Budget -> % of row accesses covered, heat map trained on ALL 6 runs, evaluated per run")
print(f"{'budget':>10} " + ' '.join(f'{n:>9}' for n in names) + f" {'union':>9}")
all_rows = np.concatenate([flat(n) for n in names])
hu, hc = heat_map([all_rows])
# overall coverage on the training set itself
mask = np.isin(all_rows, hu)
for mib in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
    row = []
    order = np.argsort(-hc)
    k = min(int(mib*2**20//ROW_BYTES), order.size)
    hot = np.sort(hu[order[:k]])
    for n in names:
        t = flat(n)
        idx = np.clip(np.searchsorted(hot, t), 0, max(hot.size-1,0))
        row.append(100.0*(hot[idx]==t).mean() if hot.size else 0.0)
    # union: overall
    idx = np.clip(np.searchsorted(hot, all_rows), 0, max(hot.size-1,0))
    u = 100.0*(hot[idx]==all_rows).mean()
    print(f"{mib:>7} MiB " + ' '.join(f'{v:8.2f}%' for v in row) + f" {u:8.2f}%")

print()
print("### Leave-one-out: heat map trained on the other 5 runs, evaluated on the held-out run")
for n in names:
    others = [flat(m) for m in names if m != n]
    hu, hc = heat_map(others)
    order = np.argsort(-hc)
    t = flat(n)
    out = []
    for mib in (64, 256, 1024, 4096):
        k = min(int(mib*2**20//ROW_BYTES), order.size)
        hot = np.sort(hu[order[:k]])
        idx = np.clip(np.searchsorted(hot, t), 0, hot.size-1)
        out.append(f"{mib}MiB={100.0*(hot[idx]==t).mean():5.2f}%")
    # oracle: the run's own top-k (same budget) -- upper bound for any static hot set
    u, c = heat_map([t])
    o = np.argsort(-c); orc = []
    for mib in (64, 256, 1024, 4096):
        k = min(int(mib*2**20//ROW_BYTES), o.size)
        hot = np.sort(u[o[:k]])
        idx = np.clip(np.searchsorted(hot, t), 0, hot.size-1)
        orc.append(f"{mib}MiB={100.0*(hot[idx]==t).mean():5.2f}%")
    print(f"  {n:9s} held-out: " + '  '.join(out) + "   | own-oracle: " + '  '.join(orc))

print()
print("### Within a single long run: heat from the first half, coverage on the second half (code80k)")
for n in ['code80k','prose','python','json']:
    t = flat(n)
    h = t.size//2
    hu, hc = heat_map([t[:h]])
    order = np.argsort(-hc)
    out=[]
    for mib in (64,256,1024,4096):
        k = min(int(mib*2**20//ROW_BYTES), order.size)
        hot = np.sort(hu[order[:k]])
        idx = np.clip(np.searchsorted(hot, t[h:]), 0, hot.size-1)
        out.append(f"{mib}MiB={100.0*(hot[idx]==t[h:]).mean():5.2f}%")
    print(f"  {n:9s} second half: " + '  '.join(out))

print()
print("### What share of ALL accesses live in rows seen >= N times (aggregate of 6 runs)")
tot = all_rows.size
for lo, hi in ((1,1),(2,2),(3,4),(5,8),(9,16),(17,64),(65,10**9)):
    sel = (hc>=lo)&(hc<=hi)
    print(f"  rows seen {lo:>2}-{hi if hi<10**9 else '+':>3}x : {sel.sum():8,} rows "
          f"({100*sel.sum()/hu.size:5.1f}% of distinct) -> {100*hc[sel].sum()/tot:5.1f}% of accesses")
print(f"  distinct rows: {hu.size:,}; total accesses {tot:,}; mean count {tot/hu.size:.2f}")
