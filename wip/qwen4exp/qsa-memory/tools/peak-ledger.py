#!/usr/bin/env python3
"""Peak live-tensor ledger from a gallocr instrumented log.

Usage: peak-ledger.py <sched.log>

Needs a build with GGML_ALLOCATOR_DEBUG (see HANDOVER.md L0): the allocator logs
  max_size[chunk] = <MiB> MB: tensors: <name> [chunk: lo-hi] (<MiB> MB) ...
on every new record; the largest record is the live set that sets the buffer size.
"""
import re, sys

def main(path):
    txt = open(path, errors='replace').read()
    recs = list(re.finditer(r'max_size\[(\d+)\] = ([\d.]+) MB: tensors:', txt))
    if not recs:
        sys.exit(f"{path}: no max_size records (built without GGML_ALLOCATOR_DEBUG?)")
    m = max(recs, key=lambda x: float(x.group(2)))
    seg = txt[m.end(): txt.find('max_size[', m.end())]
    live = [(n, float(s)) for n, a, b, s in
            re.findall(r'(\S*) \[\d+: ([0-9a-f]+)-([0-9a-f]+)\] \(([\d.]+) MB\)', seg)]
    print(f"file: {path}")
    print(f"PEAK live-set: buffer max offset = {float(m.group(2)):.2f} MB, "
          f"{len(live)} live tensors, sum of sizes = {sum(s for _, s in live):.1f} MB")
    print("\n== live tensors >= 16 MB (with op, when present in the graph dump) ==")
    nodes = {}
    for mm in re.finditer(r'node #\s*(\d+) \(\s*(\S+)\):\s+(\S+) \(\s*(\d+)([MK])\)', txt):
        nodes[mm.group(3)] = mm.group(2)
    for n, s in sorted(live, key=lambda e: -e[1]):
        if s >= 16:
            print(f"{s:9.1f} MB  op={nodes.get(n,'?'):12s}  {n}")
    import collections
    agg = collections.Counter()
    for n, s in live:
        key = re.sub(r'-?\d+$', '', n) if n else '<unnamed>'
        agg[key] += s
    print("\n== aggregate by name (trailing -il stripped) ==")
    for k, v in agg.most_common(15):
        print(f"{v:9.1f} MB  {k}")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'sched.log')
