#!/usr/bin/env python3
"""Write the bulk-load input lists for a trace: the distinct 4 KiB pages (int64 offsets)
and the distinct rows (int32 ids), plus the sector-envelope statistics.

  python3 sectors.py TRACE.bin [OUTDIR]     # default OUTDIR=/tmp/ple
"""
import sys, os
import numpy as np
from analyze import load, ROW_BYTES, RPP

DATA_OFF = 192          # per_layer_token_embd starts 192 B into ...-00002-of-00002.gguf

trace = sys.argv[1]
out   = sys.argv[2] if len(sys.argv) > 2 else '/tmp/ple'
d     = load(trace)
rows  = np.unique(d['rows'].reshape(-1)).astype(np.int64)
pages = np.unique(rows // RPP)
off   = DATA_OFF + rows * ROW_BYTES
o512  = off & ~511
span  = np.where(o512 + 512 >= off + ROW_BYTES, 512, 1024)      # a row straddling a sector needs 2

np.save(os.path.join(out, 'pages_off.npy'), DATA_OFF + pages * (RPP * ROW_BYTES))
np.save(os.path.join(out, 'rows_ids.npy'), rows.astype(np.int32))
(off := np.full(pages.size, DATA_OFF, dtype=np.int64) + pages * (RPP * ROW_BYTES)).tofile(os.path.join(out, 'pages_off.bin'))
rows.astype(np.int32).tofile(os.path.join(out, 'rows_ids.bin'))

print('%s: %d tok, %d rows (%.1f MiB), %d pages (%.1f MiB), sectors %.1f MiB, %.2f rows/page'
      % (os.path.basename(trace), d['rows'].shape[0], rows.size, rows.size*ROW_BYTES/2**20,
         pages.size, pages.size*RPP*ROW_BYTES/2**20, span.sum()/2**20, rows.size/pages.size))
