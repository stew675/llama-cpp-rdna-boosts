import sys, re
from collections import defaultdict

def parse(path):
    blocks = []
    cur = None
    for line in open(path):
        m = re.match(r".*op timing: total ([\d.]+) ms over (\d+) nodes:", line)
        if m:
            if cur: blocks.append(cur)
            cur = {"total": float(m.group(1)), "ops": defaultdict(float), "cnt": defaultdict(int)}
            continue
        if cur:
            m2 = re.match(r"\s+([\d.]+) ms\s+([\d.]+)%\s+x(\d+)\s+(.+?)\s*$", line)
            if m2:
                op = m2.group(4).split()[0]
                cur["ops"][op] += float(m2.group(1))
                cur["cnt"][op] += int(m2.group(3))
    if cur: blocks.append(cur)
    return blocks

def summarize(blocks, label):
    # aggregate all blocks, weighted
    agg = defaultdict(float); cnt = defaultdict(int); tot = 0
    for b in blocks:
        tot += b["total"]
        for k, v in b["ops"].items(): agg[k] += v
        for k, v in b["cnt"].items(): cnt[k] += v
    print(f"=== {label}: {len(blocks)} evals, total {tot:.1f} ms ===")
    for k in sorted(agg, key=lambda x: -agg[x])[:14]:
        print(f"  {agg[k]/tot*100:5.1f}%  x{cnt[k]:5d}  {k}")
    print(f"  other: {100-sum(v/tot*100 for v in agg.values()):.1f}%")

for f, label in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    blocks = parse(f)
    if len(blocks) == 0:
        print(f"{f}: no blocks"); continue
    summarize(blocks, label)
    # last block = deepest ubatch
    b = blocks[-1]
    print(f"  --- last eval (deepest): total {b['total']:.1f} ms ---")
    for k in sorted(b["ops"], key=lambda x: -b["ops"][x])[:12]:
        print(f"    {b['ops'][k]/b['total']*100:5.1f}%  x{b['cnt'][k]:4d}  {k}")
