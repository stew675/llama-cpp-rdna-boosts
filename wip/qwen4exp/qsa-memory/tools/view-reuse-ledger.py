#!/usr/bin/env python3
"""Quantify the missed through-view (reshaped-parent) reuse from an L0c-instrumented sched log.

Usage: view-reuse-ledger.py <sched.log>

Needs a build with the L0c counter: /tmp/ggml-alloc.instrumented.c (the L0 instrumentation plus a
count in ggml_gallocr_allocate_node's "is external" branch of the *true* missed through-view reuses:
view source allocator-owned, last use, same layout, aliasing view).  /tmp/bin-l0c is such a build.

Reports, per model: the buffer high-water, the "external" event counts, the number and potential total
of the true missed reuses, and - the number that matters for the reserve - how many of them are LIVE
at the peak allocation record (parse logic shared with peak-ledger.py).
"""
import re, sys

def report(tag, path):
    log = open(path, errors='replace').read()
    ev = re.findall(r'LG_LOST_VIEW_REUSE (\S+) \((\d+) bytes\) parent (\S+) src (\S+) \| running total ([\d.]+) MB n=(\d+)', log)
    nrepr = len(re.findall(r'not reusing parent .* is external', log))
    nresh = len(re.findall(r'not reusing parent \S+ \(reshaped', log))
    if not ev:
        print(f"{tag}: no events (built without the L0c counter?)"); return
    sizes = sorted(((int(b), nm) for nm, b, _, _, _, _ in ev), reverse=True)
    recs = list(re.finditer(r'max_size\[(\d+)\] = ([\d.]+) MB: tensors:', log))
    m = max(recs, key=lambda x: float(x.group(2)))
    seg = log[m.end(): log.find('max_size[', m.end())]
    live = re.findall(r'(\S*) \[\d+: ([0-9a-f]+)-([0-9a-f]+)\] \(([\d.]+) MB\)', seg)
    nr = set(nm for nm, _ in sizes)
    overlap = [(float(s), nm) for nm, a, b, s in live if nm in nr]
    print(f"== {tag} ==")
    print(f"  buffer high-water {float(m.group(2)):.2f} MB | 'not reusing' external events: {nrepr} (names tagged reshaped: {nresh})")
    print(f"  TRUE missed through-view reuses: n={ev[-1][5]}, potential total {ev[-1][4]} MiB")
    print(f"  >=16 MiB: {sum(1 for s,_ in sizes if s>=16*1048576)} | biggest: " +
          ", ".join(f"{s/1048576:.1f} MiB ({nm})" for s, nm in sizes[:4]))
    tot = sum(s for s, _ in overlap)
    print(f"  LIVE AT THE PEAK RECORD: {len(overlap)}, {tot:.1f} MiB = {100*tot/float(m.group(2)):.0f}% of the high-water"
          "   <- the reserve-relevant number")
    for s, nm in sorted(overlap, reverse=True)[:5]:
        print(f"      {s:8.1f} MiB  {nm}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        report(p.split('/')[-2] if '/' in p else p, p)
