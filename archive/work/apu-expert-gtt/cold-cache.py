#!/usr/bin/env python3
# wip/apu-expert-gtt/cold-cache.py -- drop a GGUF's PAGE CACHE so the next arm runs cold (P4`s cold arm).
#
#   python3 cold-cache.py MODEL.gguf && <arm>
#
# It uses POSIX_FADV_DONTNEED on the file, which needs no privileges (the `sudo tee /proc/sys/vm/drop_caches`
# route drops the whole machine's cache).  CAVEAT, measured: with a FITTING model a long arm re-reads the same
# experts many times, so only the first touch of each is cold -- a "cold" arm is really "cold first touch, warm
# repeats", which UNDERSTATES the cold penalty for both arms equally, so an alternating A/B is still fair.  For a
# >RAM model (the tier`s actual workload) every repeat is cold, which is why MBIG needs no drop at all.
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
    print("dropped:", p)
