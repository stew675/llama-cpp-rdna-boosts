#!/usr/bin/env python3
"""Sample /proc/PID/smaps during a run; report resident bytes by file and total."""
import subprocess, sys, time, re, os

def smaps_summary(pid):
    total = 0
    by_file = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    total = int(line.split()[1]) * 1024
        with open(f"/proc/{pid}/smaps") as f:
            cur_file = None
            for line in f:
                if re.match(r"^[0-9a-f]+-[0-9a-f]+", line):
                    parts = line.split()
                    cur_file = None
                    for i, p in enumerate(parts):
                        if p == "/":
                            cur_file = parts[i] if i < len(parts) else None
                        if i > 0 and parts[i].startswith("/"):
                            cur_file = parts[i]
                elif line.startswith("Rss:"):
                    rss = int(line.split()[1]) * 1024
                    if cur_file:
                        base = cur_file.rsplit("/", 1)[-1][:12]
                        by_file[base] = by_file.get(base, 0) + rss
    except FileNotFoundError:
        return None, None
    return total, by_file

def run(args, n_samples=40, dt=2.0):
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    samples = []
    missed = 0
    for _ in range(n_samples):
        time.sleep(dt)
        try:
            t, f = smaps_summary(proc.pid)
        except Exception:
            missed += 1
            if proc.poll() is not None:
                break
            continue
        if t is None:
            break
        samples.append((t, f))
        if proc.poll() is not None:
            break
    proc.wait()
    return proc.returncode, samples, missed

if __name__ == "__main__":
    # args after "--" are the logitdump args
    idx = sys.argv.index("--")
    cmd = sys.argv[idx+1:]
    rc, samples, missed = run(cmd)
    print(f"rc={rc}, samples={len(samples)}, missed={missed}")
    if samples:
        last_t, last_f = samples[-1]
        peak_t = max(s[0] for s in samples)
        print(f"final Rss={last_t/2**30:.2f} GiB, peak Rss={peak_t/2**30:.2f} GiB")
        for base, sz in sorted(last_f.items(), key=lambda kv: -kv[1]):
            print(f"  {base:14s} {sz/2**30:8.2f} GiB")
        # anonymous memory = total - file-backed
        anon = last_t - sum(last_f.values())
        print(f"  {'(anon+other)':14s} {anon/2**30:8.2f} GiB")
