#!/usr/bin/env python3
"""Sample RSS within a specific file range during a run.
Args: NGRAM_FILE_NUM (index of file in the model set), NGRAM_OFF, NGRAM_SIZE, then -- cmd...
"""
import subprocess, sys, time, re, os

ngram_file = sys.argv[1]        # e.g. "00002-of-00004"
ngram_off = int(sys.argv[2])
ngram_size = int(sys.argv[3])
idx = sys.argv.index("--")
cmd = sys.argv[idx+1:]

def sample(pid):
    """Return (total_rss, ngram_rss, weightfile_rss) in bytes."""
    total = 0
    ngram = 0
    other = 0
    with open(f"/proc/{pid}/smaps") as f:
        cur_file = None
        cur_start = cur_end = 0
        for line in f:
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+)\s", line)
            if m:
                cur_start, cur_end = int(m.group(1), 16), int(m.group(2), 16)
                cur_file = None
                parts = line.split()
                for p in parts[2:]:
                    if p.startswith("/"):
                        cur_file = p
                        break
            elif line.startswith("Rss:") and cur_file:
                rss = int(line.split()[1]) * 1024
                total += rss
                if ngram_file in cur_file:
                    # overlap of this mapping with the ngram byte window
                    moff = cur_start  # mapping offset == vaddr offset for these mmaps
                    moff_end = cur_end
                    ov = max(0, min(moff_end, ngram_off + ngram_size) - max(moff, ngram_off))
                    if ov > 0:
                        # this mapping is (partly) the ngram range; scale rss by overlap
                        span = moff_end - moff
                        ngram += rss * ov // span if span else 0
                    else:
                        other += rss
                else:
                    other += rss
    return total, ngram, other

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
best = None
for _ in range(60):
    time.sleep(2)
    if proc.poll() is not None:
        break
    try:
        t, n, o = sample(proc.pid)
    except Exception:
        continue
    if best is None or t > best[0]:
        best = (t, n, o)
    print(f"  rss={t/2**30:6.2f} GiB  ngram={n/2**30:6.2f} GiB  other-file={o/2**30:6.2f} GiB", flush=True)
proc.wait()
print(f"peak rss={best[0]/2**30:.2f} GiB, peak ngram={best[1]/2**30:.2f} GiB")
