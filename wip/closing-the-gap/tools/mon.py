#!/usr/bin/env python3
"""Per-thread CPU sampler for a running llama process.

Usage: mon.py <pid> [interval]
Samples until the process exits, then prints a summary + per-interval CSV.
"""
import sys, os, time, glob

pid = int(sys.argv[1])
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

def read_threads(pid):
    out = {}
    for p in glob.glob(f"/proc/{pid}/task/*/stat"):
        try:
            with open(p) as f:
                data = f.read()
        except OSError:
            continue
        rp = data.rfind(')')
        try:
            tid = int(data[5:data.index(' ')])
            fields = data[rp+2:].split()
            out[tid] = int(fields[11]) + int(fields[12])
        except (ValueError, IndexError):
            continue
    return out

HZ = os.sysconf('SC_CLK_TCK')
prev = read_threads(pid)
prev_t = time.time()
samples = []
nbusy_last = 0
lines = []
while True:
    time.sleep(interval)
    cur = read_threads(pid)
    now = time.time()
    if not cur:  # process gone
        break
    dt = now - prev_t
    if dt <= 0:
        prev, prev_t = cur, now
        continue
    total = 0.0
    busy = 0
    for tid, ticks in cur.items():
        if tid in prev:
            d = (ticks - prev[tid]) / HZ
            if d < 0:
                continue
            cores = d / dt
            total += cores
            if cores > 0.5:
                busy += 1
    samples.append(total)
    nbusy_last = busy
    lines.append(f"{now-prev_t:.2f},{total:.2f},{busy}")
    prev = cur
    prev_t = now

if samples:
    s = sorted(samples)
    mean = sum(s) / len(s)
    p50 = s[len(s)//2]
    # ignore the first 2s (load/teardown)
    core = samples[min(4, len(samples)):]
    cmean = sum(core)/len(core) if core else mean
    print(f"SUMMARY cpu_cores mean={mean:.2f} p50={p50:.2f} max={s[-1]:.2f} "
          f"steady_mean={cmean:.2f} busy_threads(last)={nbusy_last} n={len(s)}")
    for l in lines:
        print("SAMPLE " + l)
else:
    print("SUMMARY no samples")
