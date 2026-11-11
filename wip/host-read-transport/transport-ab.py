#!/usr/bin/env python3
"""Which transport reads a spilled host-expert chunk faster: an mmap view, or a buffered pread?

The end-to-end rig (`wip/bench-56gb`) answers "is the whole model faster", but a whole run carries routing, the
coalesced span, the arena and the pool, so the transport signal sits inside a +/-25 % run-to-run band.  This
isolates the transport with the real GGUF and the real chunk shape:

  * it reads ONLY expert-shaped chunks (640 KiB, the pool's `host_bytes` range) out of the model file;
  * it walks a working set of `--work-gib` repeatedly, so pass 1 is COLD and pass 2+ is as warm as the memory
    cgroup allows -- which is the variable that decides the answer;
  * the reader runs inside a memory-cgroup scope, so the page cache it may keep is a KNOB:
        work <= cache budget  -> the warm arm      (the mapping should win: no syscall, fault-around prefetch)
        work >  cache budget  -> the pressure arm  (every pass misses; the fault storm is the question)

Usage (root for the scope and the cache drop):
    sudo -n systemd-run --scope -p MemoryMax=16G -p MemorySwapMax=0 \
        python3 transport-ab.py --work-gib 4 --run-gib 8
Arms, same process and same bytes, each checksummed: mmap, buffered pread, O_DIRECT pread.

WHY IT IS NOT A DELIVERY NUMBER: nothing here measures llama.cpp.  It exists to answer one question -- does a
mapping read collapse under pressure relative to a buffered read, and at what working-set size do they cross.
"""

import argparse
import mmap
import random
import os
import sys
import time

CHUNK = 640 * 1024    # 0.625 MiB, a 4096 multiple: the pool's host_bytes on the field shape is 627200..921600
MODEL = "/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"


def drop_cache():
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")


def mi_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return -1.0


def walk(base, work, run, rnd=False):
    """Yield the file offset of every chunk to read, wrapping inside the working set.

    `rnd` matters more than it looks: the model's expert reads are ROUTING-driven, i.e. random at 640 KiB
    granularity.  A sequential mapping read gets the kernel's fault-around/readahead for free and ties a
    buffered read (measured), so a sequential test cannot reproduce a pressure collapse -- it has to be random.
    """
    n = run // CHUNK
    span = max(1, work // CHUNK)
    for i in range(n):
        yield base + (random.randrange(span) * CHUNK if rnd else (i % span) * CHUNK)


def arm_mmap(fd, base, work, run, rnd=False):
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    n, total = 0, 0
    t0 = time.time()
    try:
        for off in walk(base, work, run, rnd):
            data = mm[off:off + CHUNK]      # the copy the consumer does: faults every page of the chunk
            n += data[0]
            total += len(data)
    finally:
        mm.close()
    return total, n, time.time() - t0


def arm_pread(fd, base, work, run, rnd=False):
    scratch = mmap.mmap(-1, CHUNK)          # page aligned; writable, so preadv can land in it
    view = memoryview(scratch)[:CHUNK]
    n, total = 0, 0
    t0 = time.time()
    try:
        for off in walk(base, work, run, rnd):
            got = os.preadv(fd, [view], off)
            n += view[0]
            total += got
    finally:
        view.release()
        scratch.close()
    return total, n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--work-gib", type=float, default=4.0, help="working set, walked repeatedly")
    ap.add_argument("--passes", type=int, default=2, help="repeats of the [cold, warm] pair per arm")
    ap.add_argument("--base-mib", type=int, default=8192, help="file offset of the working set")
    ap.add_argument("--random", action="store_true", help="random chunk order (the routing shape)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    work = int(a.work_gib * 1024 ** 3) // CHUNK * CHUNK
    base = a.base_mib * 1024 * 1024
    fsize = os.path.getsize(a.model)
    if work < CHUNK or base + work > fsize:
        sys.exit("working set does not fit the file")
    random.seed(a.seed)
    tag = os.environ.get("ARM_TAG", "")
    lim = os.environ.get("MEMORY_MAX", "?")
    print(f"{tag} cgroup={lim} | model {fsize/2**30:.1f} GiB | working set {work/2**30:.2f} GiB at "
          f"+{a.base_mib} MiB | chunk {CHUNK/1024:.0f} KiB | {'RANDOM' if a.random else 'sequential'} | "
          f"MemAvailable {mi_available():.1f} GiB", flush=True)

    fd = os.open(a.model, os.O_RDONLY)
    fd_d = os.open(a.model, os.O_RDONLY | os.O_DIRECT)
    arms = [("mmap", arm_mmap, fd), ("pread", arm_pread, fd), ("odirect", arm_pread, fd_d)]
    cold, warm = {}, {}
    try:
        # Per arm, ONE cache drop and then a cold read followed immediately by a warm one.  Dropping per ARM
        # (not per pass) is what removes the ordering effect that dominated the first version of this
        # experiment: whichever arm ran second was reading a cache the first arm had just filled.
        for p in range(a.passes):
            for name, fn, f in (arms if p % 2 == 0 else list(reversed(arms))):
                drop_cache()
                _, _, dt = fn(f, base, work, work, a.random)   # cold: nothing of the working set is cached
                c = work / 1048576.0 / dt
                _, _, dt = fn(f, base, work, work, a.random)   # warm: straight after, as warm as the cgroup allows
                w = work / 1048576.0 / dt
                cold.setdefault(name, []).append(c)
                warm.setdefault(name, []).append(w)
                print(f"  rep {p+1}  {name:8s} cold {c:9.1f} MiB/s   warm {w:9.1f} MiB/s", flush=True)
    finally:
        os.close(fd)
        os.close(fd_d)

    print()
    for label, d in (("COLD", cold), ("WARM", warm)):
        m = {k: sum(v) / len(v) for k, v in d.items()}
        print(f"{label}: mmap {m['mmap']:.0f}  pread {m['pread']:.0f}  odirect {m['odirect']:.0f} MiB/s "
              f"-> pread/mmap = {m['pread'] / m['mmap']:.2f}x, odirect/mmap = {m['odirect'] / m['mmap']:.2f}x")
    print("WARM is the shape where the page cache can hold the working set; if this cgroup cannot, WARM will"
          "\nmatch COLD -- which is the pressure case the direction is about.")


if __name__ == "__main__":
    main()
