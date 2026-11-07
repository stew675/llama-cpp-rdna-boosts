#!/usr/bin/env python3
"""Simulate the two candidate residency mechanisms over the measured PLE traces:
   (1) a row-granular LRU cache (90 B/row)  -- the 'hot rows in RAM' idea
   (2) a page-granular LRU cache (4 KiB/page, what mmap/page-cache does today)
Stream = per-ubatch distinct rows (a gather dedups its own rows; the page path dedups pages).
"""
import sys
from collections import OrderedDict
import numpy as np
from analyze import load, N_ROWS, ROW_BYTES, PAGE, RPP, N_PAGES

RUNS = ['threejs','zh','python','json','prose','code80k','code560k']  # small -> large, then repeat

def stream(name, unit):
    d = load(f'/tmp/ple/{name}.bin')
    out = []
    for tok, rows in d['recs']:
        r = rows.reshape(-1)
        if unit == 'page':
            r = r // RPP
        out.append(np.unique(r))
    return np.concatenate(out)

def lru_miss_rate(seq, cap):
    """cap = number of entries. returns (misses, accesses)."""
    od = OrderedDict()
    miss = 0
    for x in seq.tolist():
        if x in od:
            od.move_to_end(x)
        else:
            miss += 1
            if len(od) >= cap:
                od.popitem(last=False)
            od[x] = None
    return miss, seq.size

if __name__ == '__main__':
    rows_seq = {n: stream(n, 'row') for n in RUNS}
    page_seq = {n: stream(n, 'page') for n in RUNS}
    tot_rows = sum(s.size for s in rows_seq.values())
    tot_pages = sum(s.size for s in page_seq.values())
    print(f"lookup stream (dedup within a ubatch): {tot_rows:,} row lookups, {tot_pages:,} page lookups over {len(RUNS)} runs")
    print(f"whole workload: distinct rows {np.unique(np.concatenate(list(rows_seq.values()))).size:,}"
          f" ({np.unique(np.concatenate(list(rows_seq.values()))).size*ROW_BYTES/2**20:.1f} MiB),"
          f" distinct pages {np.unique(np.concatenate(list(page_seq.values()))).size:,}"
          f" ({np.unique(np.concatenate(list(page_seq.values()))).size*PAGE/2**20:.1f} MiB)")
    print()
    order = RUNS + RUNS  # two passes: a session that revisits the same content
    rs = np.concatenate([rows_seq[n] for n in order])
    ps = np.concatenate([page_seq[n] for n in order])
    print("LRU over %d passes of the 7 runs (%d row lookups, %d page lookups)" % (2, rs.size, ps.size))
    print(f"{'budget':>10} {'rows':>10} {'row hit%':>9} {'MiB read':>10} | {'pages':>8} {'page hit%':>10} {'MiB read':>10}")
    for mib in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        capr = max(1, int(mib*2**20//ROW_BYTES))
        m1, n1 = lru_miss_rate(rs, capr)
        capp = max(1, int(mib*2**20//PAGE))
        m2, n2 = lru_miss_rate(ps, capp)
        print(f"{mib:>7} MiB {capr:>10,} {100*(1-m1/n1):8.2f}% {m1*ROW_BYTES/2**20:10.1f} | "
              f"{capp:>8,} {100*(1-m2/n2):9.2f}% {m2*PAGE/2**20:10.1f}")
