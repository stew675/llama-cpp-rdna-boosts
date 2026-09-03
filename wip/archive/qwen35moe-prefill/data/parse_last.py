import re
from collections import defaultdict
blocks = []
cur = None
for line in open("/tmp/op_moe_16k_ub2048.txt"):
    m = re.match(r".*op timing: total ([\d.]+) ms over (\d+) nodes:", line)
    if m:
        if cur: blocks.append(cur)
        cur = {"total": float(m.group(1)), "ops": defaultdict(float), "cnt": defaultdict(int)}
        continue
    if cur:
        m2 = re.match(r"\s+([\d.]+) ms\s+([\d.]+)%\s+x(\d+)\s+(.+?)\s*$", line)
        if m2:
            name = m2.group(4)
            # categorize: op + tensor name
            parts = name.split()
            op = parts[0]
            tname = parts[1] if len(parts) > 1 else ""
            cur["ops"][op] += float(m2.group(1))
            cur["cnt"][op] += int(m2.group(3))
b = blocks[-1]  # deepest
tot = b["total"]
print(f"deepest eval total {tot:.1f} ms, {sum(b['cnt'].values())} nodes")
for k in sorted(b["ops"], key=lambda x: -b["ops"][x]):
    print(f"  {b['ops'][k]/tot*100:5.1f}%  x{b['cnt'][k]:4d}  {k}")
