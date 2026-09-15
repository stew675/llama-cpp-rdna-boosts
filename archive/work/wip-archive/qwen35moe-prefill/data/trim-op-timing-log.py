#!/usr/bin/env python3
"""Trim a llama-cli `op timing:` dump down to its parsed aggregate.

These dumps (GGML_CUDA_OP_DEBUG-style per-op timing) are 3-7 MB of per-eval
chatter whose analysis value is entirely in the aggregate: the per-eval totals
and the per-op share of the deepest ubatch.  That is what `data/parse_ops.py`
computes and what `report.md`'s op-breakdown tables are built from -- so keep
the parsed result (a few KiB) and drop the raw lines, with a banner recording
the original size and the transform.

The per-op regexes are parse_ops.py's, verbatim, so the shares are identical to
the ones already published.

Usage:  trim-op-timing-log.py <log> [<log> ...]      (rewrites in place)
"""
import re
import sys
from collections import defaultdict

BANNER = "# TRIMMED {date} by wip/../trim-op-timing-log.py"
# -- the two regexes from data/parse_ops.py, unchanged --
RE_BLOCK = re.compile(r".*op timing: total ([\d.]+) ms over (\d+) nodes:")
RE_OP = re.compile(r"\s+([\d.]+) ms\s+([\d.]+)%\s+x(\d+)\s+(.+?)\s*$")
IDENT = re.compile(r"^(build:|llama_model_loader: loaded meta data|.*op timing: total|llama_model_load: n_)\S*")


def parse(text):
    blocks = []
    cur = None
    for line in text.splitlines():
        m = RE_BLOCK.match(line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"total": float(m.group(1)), "nodes": int(m.group(2)),
                   "ops": defaultdict(float), "cnt": defaultdict(int)}
            continue
        if cur:
            m2 = RE_OP.match(line)
            if m2 and m2.group(4) and not m2.group(4).startswith("TOTAL"):
                op = m2.group(4).split()[0]
                cur["ops"][op] += float(m2.group(1))
                cur["cnt"][op] += int(m2.group(3))
    if cur:
        blocks.append(cur)
    return blocks


def trim(path, label):
    raw = open(path, errors="replace").read()
    n_lines = raw.count("\n") + 1
    n_bytes = len(raw.encode())
    blocks = parse(raw)
    ident = [ln.strip() for ln in raw.splitlines() if IDENT.match(ln)][:4]

    out = [f"# TRIMMED {__import__('datetime').date.today().isoformat()} by tools/trim-op-timing-log.py",
           f"#   original: {n_lines} lines / {n_bytes/1048576:.2f} MiB raw `op timing:` dump",
           f"#   kept: run identity + the parse_ops.py aggregate (same regexes, same shares)",
           ""]
    out.append("## run identity")
    out.extend("  " + i[:160] for i in ident)
    out.append("")
    if not blocks:
        out.append("## aggregate\n  (no `op timing:` blocks found)")
    else:
        tot = sum(b["total"] for b in blocks)
        agg = defaultdict(float)
        cnt = defaultdict(int)
        for b in blocks:
            for k, v in b["ops"].items():
                agg[k] += v
            for k, v in b["cnt"].items():
                cnt[k] += v
        out.append(f"## aggregate -- {label}: {len(blocks)} evals, total {tot:.1f} ms")
        for k in sorted(agg, key=lambda x: -agg[x])[:14]:
            out.append(f"  {agg[k]/tot*100:5.1f}%  x{cnt[k]:5d}  {k}")
        out.append(f"  other: {100 - sum(v/tot*100 for v in agg.values()):.1f}%")
        b = blocks[-1]
        out.append("")
        out.append(f"## deepest eval (last block): total {b['total']:.1f} ms over {b['nodes']} nodes")
        for k in sorted(b["ops"], key=lambda x: -b["ops"][x])[:14]:
            out.append(f"  {b['ops'][k]/b['total']*100:5.1f}%  x{b['cnt'][k]:5d}  {k}")
    out.append("")
    open(path, "w").write("\n".join(out))
    return n_lines, n_bytes, len("\n".join(out).encode())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        lbl = p.split("/")[-1].rsplit(".", 1)[0]
        bl, bb, ab = trim(p, lbl)
        print(f"{p}: {bl} lines / {bb/1048576:.2f} MiB -> {ab/1024:.1f} KiB")
