#!/usr/bin/env python3
"""Sample anonymous vs file-backed RSS during a run."""
import subprocess, sys, time, re

idx = sys.argv.index("--")
cmd = sys.argv[idx+1:]

def sample(pid):
    anon = 0
    fileb = 0
    with open(f"/proc/{pid}/smaps") as f:
        cur_file = None
        for line in f:
            if re.match(r"^[0-9a-f]+-[0-9a-f]+", line):
                cur_file = None
                parts = line.split()
                for p in parts[2:]:
                    if p.startswith("/"):
                        cur_file = p
                        break
            elif line.startswith("Rss:") and not line.startswith("Shared"):
                rss = int(line.split()[1]) * 1024
                if cur_file:
                    fileb += rss
                else:
                    anon += rss
    return anon, fileb

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
peak_anon = 0
for _ in range(120):
    time.sleep(2)
    if proc.poll() is not None:
        break
    try:
        a, fb = sample(proc.pid)
    except Exception:
        continue
    peak_anon = max(peak_anon, a)
    print(f"  anon={a/2**30:6.2f} GiB  file-backed={fb/2**30:6.2f} GiB  total={(a+fb)/2**30:6.2f} GiB", flush=True)
proc.wait()
print(f"peak anon = {peak_anon/2**30:.2f} GiB (the managed arena lives here)")
