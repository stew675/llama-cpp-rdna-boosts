#!/usr/bin/env python3
"""PLE usage characterisation: what share of the per-layer-embedding table does a
prefill/decode pass touch, and what would a static hot set buy?"""
import numpy as np, os
from analyze import load, N_ROWS, ROW_BYTES, PAGE, RPP, N_PAGES, N_HEADS

RUNS = ['code80k','prose','python','json','threejs','zh','code560k']
D = {n: load(f'/tmp/ple/{n}.bin') for n in RUNS}

def flat(n): return D[n]['rows'].reshape(-1)

def cov(hot_rows, target):
    hot = np.sort(np.asarray(hot_rows))
    if hot.size == 0: return 0.0
    i = np.clip(np.searchsorted(hot, target), 0, hot.size-1)
    return 100.0*(hot[i] == target).mean()

print("="*100)
print("A. per-run footprint")
print(f"{'run':>10} {'tokens':>8} {'row acc':>9} {'uniq rows':>11} {'row MiB':>8} {'% table':>9} {'pages':>9} {'page MiB':>9} {'% pages':>8} {'acc/row':>8}")
for n in RUNS:
    f = flat(n); nd = np.unique(f)
    pg = np.unique(f//RPP)
    print(f"{n:>10} {D[n]['rows'].shape[0]:>8,} {f.size:>9,} {nd.size:>11,} {nd.size*ROW_BYTES/2**20:>8.2f} {100*nd.size/N_ROWS:>8.4f}% "
          f"{pg.size:>9,} {pg.size*PAGE/2**20:>9.1f} {100*pg.size/N_PAGES:>7.3f}% {f.size/nd.size:>8.2f}")

print()
print("="*100)
print("B. Heaps: distinct rows as the same prefill grows (code560k, 190k tokens)")
f = flat('code560k'); n_tok = D['code560k']['rows'].shape[0]
for k in (5000, 10000, 20000, 40000, 80000, 120000, 160000, 180000, n_tok):
    sub = D['code560k']['rows'][:k].reshape(-1)
    u = np.unique(sub)
    print(f"   {k:>7,} tokens: {u.size:>9,} distinct rows ({100*u.size/N_ROWS:7.5f}% of table, {u.size*ROW_BYTES/2**20:7.1f} MiB), "
          f"={(u.size/k):6.2f} rows/token, {sub.size/u.size:5.2f} acc/row")

print()
print("="*100)
print("C. static hot set ranked by its own frequency (code560k, self-oracle ceiling)")
uf, cf = np.unique(f, return_counts=True)
order = np.argsort(-cf); hot_all = uf[order]
for mib in (0.5, 2, 8, 32, 128, 512, 2048):
    k = max(1, int(mib*2**20//ROW_BYTES))
    k = min(k, hot_all.size)
    print(f"   {mib:>7} MiB hot ({k:>9,} rows): covers {cov(hot_all[:k], f):6.2f}% of accesses, {100*k/N_ROWS:.5f}% of table")

print()
print("="*100)
print("D. cross-document: heat trained on the first 60% of the same corpus, evaluated on the last 40%")
h = int(f.size*0.6)
tr, te = f[:h], f[h:]
ut, ct = np.unique(tr, return_counts=True)
order = np.argsort(-ct); hot_tr = ut[order]
for mib in (1, 4, 16, 64, 256, 1024, 4096):
    k = min(max(1, int(mib*2**20//ROW_BYTES)), hot_tr.size)
    print(f"   {mib:>7} MiB hot ({k:>9,} rows): {cov(hot_tr[:k], te):6.2f}% of the held-out accesses;"
          f" heat set is {100*k/N_ROWS:.5f}% of the table")
print(f"   the training half touched {ut.size:,} rows ({ut.size*ROW_BYTES/2**20:.1f} MiB);"
      f" {cov(ut, te):.2f}% of the held-out accesses hit a row seen in the training half")

print()
print("="*100)
print("E. cross-run: union heat from the 6 small runs, evaluated on code560k and vice versa")
small = np.concatenate([flat(n) for n in RUNS if n != 'code560k'])
us, cs = np.unique(small, return_counts=True); o = np.argsort(-cs)
big = flat('code560k')
print(f"   small runs: {small.size:,} accesses, {us.size:,} rows ({us.size*ROW_BYTES/2**20:.2f} MiB)")
for mib in (0.25, 1, 4, 16):
    k = min(max(1,int(mib*2**20//ROW_BYTES)), us.size)
    print(f"   {mib:>6} MiB hot: {cov(us[o][:k], big):6.2f}% of code560k's accesses")
ub, cb = np.unique(big, return_counts=True); ob = np.argsort(-cb)
print(f"   code560k: {ub.size:,} rows ({ub.size*ROW_BYTES/2**20:.2f} MiB)")
for mib in (0.25, 1, 4, 16, 64, 256):
    k = min(max(1,int(mib*2**20//ROW_BYTES)), ub.size)
    covered = 0.0
    for n in RUNS:
        if n == 'code560k': continue
        covered += cov(ub[ob][:k], flat(n)) * flat(n).size
    tot = sum(flat(n).size for n in RUNS if n != 'code560k')
    print(f"   {mib:>6} MiB hot: {covered/tot:6.2f}% of the 6 small runs' accesses (weighted)")

print()
print("="*100)
print("F. n-gram census (code560k)")
tok = D['code560k']['toks']
print(f"   tokens {tok.size:,}; bigram head distinct {np.unique(D['code560k']['rows'][:,0]).size:,} (x8 heads),"
      f" trigram head distinct {np.unique(D['code560k']['rows'][:,8]).size:,} (x8 heads)")
print(f"   distinct n-grams (bigram+trigram) = {np.unique(D['code560k']['rows'][:,0]).size + np.unique(D['code560k']['rows'][:,8]).size:,}"
      f" -> rows = 8x that = {8*(np.unique(D['code560k']['rows'][:,0]).size + np.unique(D['code560k']['rows'][:,8]).size):,}")

print()
print("="*100)
print("G. page-level reality: cost of a prefill if the PLE is cold")
pg = np.unique(f//RPP)
print(f"   code560k: {pg.size:,} distinct 4 KiB pages = {pg.size*PAGE/2**20:.1f} MiB read to deliver {np.unique(f).size*ROW_BYTES/2**20:.1f} MiB of rows"
      f" ({pg.size*PAGE/(np.unique(f).size*ROW_BYTES):.0f}x amplification)")
print(f"   per 1k new tokens: {1000*N_HEADS:.0f} row gathers -> up to {1000*N_HEADS*PAGE/2**20:.1f} MiB of cold pages,"
      f" {1000*N_HEADS*ROW_BYTES/2**20:.2f} MiB of useful rows")
