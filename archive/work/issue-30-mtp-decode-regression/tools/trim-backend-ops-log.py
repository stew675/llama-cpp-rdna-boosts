#!/usr/bin/env python3
"""Trim a `test-backend-ops -o <op>` log down to its claim-carrying content.

A full `-o FLASH_ATTN_EXT` log is ~27k lines / ~3 MB, most of it per-case
CUDA-graph warmup chatter.  What actually backs a documented claim is:

  * the header (device, build);
  * the **coverage** -- how many cases ran per (type_K, type_V) pair;
  * the **failures** -- grouped by (error kind, hsk, type pair) with one
    verbatim sample each, so a numeric drift is distinguishable from a NaN;
  * the **totals** ("N/M tests passed").

This keeps all of that in ~40 lines (~1% of the size) and stamps a banner
naming the transform and the original size, so the trimmed file stays
auditable instead of silently shorter.

Usage:  trim-backend-ops-log.py <log> [<log> ...]      (rewrites in place)
"""
import re
import sys
from collections import Counter, OrderedDict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CASE_DESC = re.compile(r"(FLASH_ATTN_EXT\(.*?\))\s*:")
KV = re.compile(r"hsk=(\d+).*?type_K=([a-z0-9_]+),type_V=([a-z0-9_]+)")
ERR_KIND = re.compile(r"^\[\w+\]\s+(\S+)")
HEADER = re.compile(r"^(ggml_cuda_init:|  Device |Testing |build:|ggml_backend_cuda_init|.*\bCUDA\b.*devices?|load_backend:|Backend \d+/\d+)")


def trim(path):
    lines = open(path, errors="replace").read().splitlines()
    n_bytes = len(open(path, "rb").read())

    header = []
    coverage = Counter()
    failures = OrderedDict()
    totals = []
    n_cases = 0

    for raw in lines:
        line = ANSI.sub("", raw).rstrip()
        if len(header) < 6 and HEADER.match(line) and "FLASH_ATTN_EXT(" not in line:
            if line not in header:
                header.append(line)
        if "tests passed" in line or "backends passed" in line or "FA vec slice" in line:
            totals.append(line.strip())
        m = CASE_DESC.search(line)
        if not m:
            continue
        n_cases += 1
        kv = KV.search(line)
        tag = (kv.group(2), kv.group(3)) if kv else ("?", "?")
        hsk = kv.group(1) if kv else "?"
        coverage[tag] += 1
        if line.endswith("FAIL") or " FAIL" in line:
            em = ERR_KIND.match(line)
            kind = em.group(1) if em else "FAIL"
            key = (kind, hsk, tag[0], tag[1])
            if key not in failures:
                failures[key] = [0, m.group(1) + ": FAIL"]
            failures[key][0] += 1

    out = []
    out.append(f"# TRIMMED {__import__('datetime').date.today().isoformat()} by "
               f"archive/work/issue-30-mtp-decode-regression/tools/trim-backend-ops-log.py")
    out.append(f"#   original: {len(lines)} lines / {n_bytes/1048576:.2f} MiB raw test-backend-ops output")
    out.append(f"#   kept: header, coverage, failure signatures, totals  ({n_cases} cases)")
    out.append("")
    out.append("## header")
    out.extend(header)
    out.append("")
    out.append("## coverage (cases per K/V type pair)")
    for (k, v), n in sorted(coverage.items(), key=lambda x: (-x[1], x[0])):
        out.append(f"  {n:5d}  type_K={k:<7} type_V={v}")
    out.append("")
    out.append("## failures")
    if not failures:
        out.append("  (none)")
    for (kind, hsk, tk, tv), (n, sample) in sorted(failures.items(),
                                                   key=lambda x: (-x[1][0], x[0])):
        out.append(f"  {n:5d}x  {kind}  hsk={hsk}  type_K={tk} type_V={tv}")
        out.append(f"         e.g. {sample}")
    out.append("")
    out.append("## totals")
    out.extend(f"  {t}" for t in totals)
    out.append("")

    open(path, "w").write("\n".join(out))
    return len(lines), n_bytes, len("\n".join(out))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        before_l, before_b, after_b = trim(p)
        print(f"{p}: {before_l} lines / {before_b/1048576:.2f} MiB -> "
              f"{after_b/1024:.1f} KiB")
