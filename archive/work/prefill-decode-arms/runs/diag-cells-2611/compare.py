#!/usr/bin/env python3
"""Compare the two-arm cell census / logits dumps of the diag-cells-2611 run (RESULTS-2026-11-22.md).

    ./compare.py                     # everything in this directory
    ./compare.py --f32               # also the numeric row diffs (needs the .f32 bodies, which are NOT committed)

The logs come from two_arms with LLAMA_KV_DEBUG_CELLS=1 LLAMA_RS_DEBUG_CELLS=1 (the compact per-ubatch
censuses of llama-kv-cache.cpp / llama-memory-recurrent.cpp) and ARMS_LOGITS_DIR=<dir> (the sampled logits
rows plus a manifest with each row's sha256).  `.f32` bodies are deleted from the repo: reproduce them with
the commands in the results file.
"""
import argparse
import glob
import gzip
import os
import re
import sys

KV = re.compile(r"\[TAG_KV_CELLS\] seq=(\d+) strm=(\d+) n_tok=(\d+) head_in=(\d+) used=(\d+) size=(\d+) "
                r"pos=\[(-?\d+),(-?\d+)\] runs=(\S+)")
RS = re.compile(r"\[TAG_RS_CELLS\] (\S+?) head=(\d+) n=(\d+) min=(\d+) max=(\d+) rs_z=(-?\d+) used=(\d+) "
                r"size=(\d+) cells=(\S*) tails=(\S*)")
ROW = re.compile(r"ROW mode=(\S+) rep=(\d+) req=(\d+) step=(\d+) row=(-?\d+) n_vocab=(\d+) "
                 r"argmax=(-?\d+) sha=(\S+)")


def lines(fn):
    """The logs are committed gzipped (`runs/` is gitignored at the repo root, so they are force-added)."""
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt", errors="replace") as fp:
        yield from fp


def kv(fn):
    out = []
    for ln in lines(fn):
        m = KV.search(ln)
        if m:
            g = m.groups()
            out.append((g[0], g[1], g[2], g[4], g[8]))   # seq, stream, n_tok, used, cells (head_in dropped)
    return out


def rs(fn):
    out = []
    for ln in lines(fn):
        m = RS.search(ln)
        if m:
            out.append(m.groups())
    return out


def rows(d):
    out = {}
    p = os.path.join(d, "manifest.txt")
    if os.path.exists(p):
        for ln in open(p):
            m = ROW.search(ln)
            if m:
                out[(int(m.group(2)), int(m.group(3)), int(m.group(4)))] = m.group(8)
    return out


def halves(xs):
    h = len(xs) // 2
    return xs[:h], xs[h:]


def report_census(files):
    print("== KV cell census ==")
    for fn in files:
        r = kv(fn)
        if not r:
            continue
        a, b = halves(r)
        if len(a) != len(b):
            print(f"  {fn}: only one rep ({len(r)} lines)")
            continue
        d = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        print(f"  {fn}: {len(a)} lines/rep, {len(d)} differ between rep 0 and rep 1")
        for i, x, y in d[:6]:
            print(f"     line {i}: rep0 seq{x[0]} strm{x[1]} n{x[2]} used{x[3]} {x[4]}")
            print(f"              rep1 seq{y[0]} strm{y[1]} n{y[2]} used{y[3]} {y[4]}")

    print("== KV cell census, arms vs batch-pair (rep 1) ==")
    for a_fn, b_fn in (("arms.err.gz", "batch-pair.err.gz"), ("host-arms.err.gz", "host-batch-pair.err.gz")):
        if not (os.path.exists(a_fn) and os.path.exists(b_fn)):
            continue
        a, b = halves(kv(a_fn))[1], halves(kv(b_fn))[1]
        n = min(len(a), len(b))
        d = [(i, a[i], b[i]) for i in range(n) if a[i] != b[i]]
        print(f"  {a_fn} vs {b_fn}: {len(d)} of {n} lines differ")

    print("== RE current cell census (cell indices differ, state math does not) ==")
    for fn in files:
        r = rs(fn)
        if r:
            print(f"  {fn}: {len(r)} lines; first: {r[0][0]} head={r[0][1]} used={r[0][6]} cells={r[0][8]}")


def report_logits(pairs, with_f32):
    print("== logits rows (sha256 of the f32 row) ==")
    for a_d, b_d in pairs:
        a, b = rows(a_d), rows(b_d)
        if not a or not b:
            continue
        keys = sorted(set(a) & set(b))
        bad = [k for k in keys if a[k] != b[k]]
        print(f"  {a_d} vs {b_d}: {len(keys)} rows, {len(bad)} differ, first={bad[0] if bad else None}")
        if with_f32:
            import numpy as np
            for k in (keys[:1] + bad[:2]):
                pa = os.path.join(a_d, f"{a_d.split('logits-')[-1]}-r{k[0]}-q{k[1]}-s{k[2]}.f32")
                pb = os.path.join(b_d, f"{b_d.split('logits-')[-1]}-r{k[0]}-q{k[1]}-s{k[2]}.f32")
                if os.path.exists(pa) and os.path.exists(pb):
                    x = np.fromfile(pa, dtype=np.float32)
                    y = np.fromfile(pb, dtype=np.float32)
                    print(f"     rep{k[0]} req{k[1]} step{k[2]}: max|d|={float(abs(x-y).max()):.3e} "
                          f"argmax {x.argmax()}/{y.argmax()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f32", action="store_true", help="also the numeric row diffs (needs the .f32 bodies)")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.chdir(args.dir)

    files = sorted(os.path.basename(f) for f in sorted(set(glob.glob("*.err.gz") + glob.glob("*.err"))))
    report_census(files)
    report_logits(
        [("logits-arms", "logits-batch-pair"), ("hb0-logits-arms", "hb0-logits-batch-pair")],
        args.f32,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
