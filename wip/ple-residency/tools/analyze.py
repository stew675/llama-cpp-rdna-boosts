#!/usr/bin/env python3
"""Offline analysis of a LLAMA_PLE_TRACE dump (qwen4exp per-layer-embedding gather)."""
import sys, os, glob, struct
import numpy as np

# model geometry (Qwen3.8-Flash-Next GSQ-RCO-IQ3_S, read from the GGUF)
N_ROWS      = 320001536
ROW_BYTES   = 90            # IQ4_NL, 160 elems
PAGE        = 4096
RPP         = PAGE // ROW_BYTES          # 45 rows per page
N_PAGES     = (N_ROWS + RPP - 1) // RPP
N_HEADS     = 16
PER_GRAM    = 8
OFFSETS     = None
VSIZES      = None

def parse(path):
    buf = open(path, 'rb').read()
    assert buf[:8] == b'PLETRC01', path
    o = 8
    recs = []
    while o < len(buf):
        nt, = struct.unpack_from('<I', buf, o); o += 4
        tok = np.frombuffer(buf, dtype='<i4', count=nt, offset=o); o += 4 * nt
        rows = np.frombuffer(buf, dtype='<i4', count=nt * N_HEADS, offset=o); o += 4 * nt * N_HEADS
        recs.append((tok, rows.reshape(nt, N_HEADS)))
    return recs

def load(path):
    recs = parse(path)
    toks = np.concatenate([r[0] for r in recs]) if recs else np.zeros(0, 'i4')
    rows = np.concatenate([r[1] for r in recs]) if recs else np.zeros((0, N_HEADS), 'i4')
    pre  = np.array([r[0].size > 1 for r in recs])
    return dict(recs=recs, toks=toks, rows=rows, pre=pre)

def unique_counts(rows_flat):
    return np.unique(rows_flat, return_counts=True)

def summarize(name, d):
    rows = d['rows']
    flat = rows.reshape(-1)
    n_tok = rows.shape[0]
    u, c = unique_counts(flat)
    pages = np.unique(flat // RPP)
    print(f"== {name}: {n_tok} tokens, {flat.size} row accesses ({N_HEADS}/token)")
    print(f"   unique rows {u.size:,} ({100*u.size/N_ROWS:.5f}% of {N_ROWS:,})"
          f"  = {u.size*ROW_BYTES/2**20:.1f} MiB of row bytes")
    print(f"   unique {PAGE}B pages {pages.size:,} ({100*pages.size/N_PAGES:.4f}% of {N_PAGES:,})"
          f"  = {pages.size*PAGE/2**20:.1f} MiB of pages  (x{PAGE/ROW_BYTES:.0f} page amplification)")
    print(f"   reuse: {flat.size/u.size:.2f} accesses per distinct row;"
          f" top-1% rows = {100*np.sort(c)[::-1][:max(1,u.size//100)].sum()/flat.size:.2f}% of accesses")
    # per-head
    hu = [np.unique(rows[:,h]).size for h in range(N_HEADS)]
    print(f"   unique rows per head: {hu}")
    # prefill vs decode
    if d['pre'].any():
        for tag, mask in (('prefill-ubatches', d['pre'].any()), ('decode-tokens', True)):
            pass
    pre_rows = np.concatenate([r[1].reshape(-1) for r, p in zip(d['recs'], d['pre']) if p]) if d['pre'].any() else np.zeros(0,'i4')
    dec_rows = np.concatenate([r[1].reshape(-1) for r, p in zip(d['recs'], d['pre']) if not p]) if (~d['pre']).any() else np.zeros(0,'i4')
    for tag, a in (('prefill', pre_rows), ('decode', dec_rows)):
        if a.size:
            uu = np.unique(a)
            print(f"   {tag:8s}: {a.size:9,} accesses  {uu.size:9,} unique rows"
                  f"  reuse {a.size/uu.size:.2f}  pages {np.unique(a//RPP).size:,}")
    return u, c

def coverage_curve(heat, rows_flat, label, budgets_mib=(32,64,128,256,512,1024,2048,4096)):
    """heat: dict row->count (or array of rows sorted by heat). coverage on rows_flat."""
    order = np.argsort(-heat[1])
    hot = heat[0][order]
    total = rows_flat.size
    hits = np.isin(rows_flat, hot)  # memo for the full set
    # cumulative coverage at each prefix length: compute via sorted ranks
    print(f"   {label}")
    for mib in budgets_mib:
        k = int(mib * 2**20 // ROW_BYTES)
        if k > hot.size:
            k = hot.size
        cov = hits[:0].size  # placeholder
        # coverage of the first k hot rows
        cov = np.isin(rows_flat, hot[:k]).mean() * 100 if k else 0.0
        print(f"     {mib:5d} MiB  ({k:9,} rows)  covers {cov:6.2f}% of accesses,"
              f" {100*k/N_ROWS:.4f}% of the table")
    return hot

if __name__ == '__main__':
    names = sys.argv[1:]
    data = {}
    for p in names:
        n = os.path.basename(p).replace('.bin','')
        d = load(p)
        data[n] = d
        summarize(n, d)
        print()
    # aggregate heat
    all_rows = np.concatenate([d['rows'].reshape(-1) for d in data.values()])
    u, c = np.unique(all_rows, return_counts=True)
    print(f"== ALL {len(data)} runs: {all_rows.size:,} accesses, {u.size:,} unique rows"
          f" ({100*u.size/N_ROWS:.5f}%), {np.unique(all_rows//RPP).size:,} pages"
          f" ({np.unique(all_rows//RPP).size*PAGE/2**20:.1f} MiB)")
    print(f"   scale factor: when a row IS repeated it is seen {c[c>1].mean():.2f}x on average"
          f" ({(c>1).sum():,} rows repeated, {100*(c>1).sum()/u.size:.1f}%)")
