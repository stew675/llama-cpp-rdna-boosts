import os, random, sys, threading, time
F = "/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf"
SZ = 1769472        # one expert blob
N  = 64
fsz = os.path.getsize(F)
random.seed(7)
offs = [4096 * random.randrange(1, (fsz - SZ) // 4096) for _ in range(N)]

def read_many(idxs, out):
    fd = os.open(F, os.O_RDONLY | os.O_DIRECT)
    buf = bytearray(SZ)
    mv = memoryview(buf)
    t0 = time.perf_counter()
    for i in idxs:
        os.preadv(fd, [mv], offs[i])
    out.append(time.perf_counter() - t0)
    os.close(fd)

for depth in (1, 2, 4, 8):
    threads = []
    res = []
    per = N // depth
    groups = [list(range(k * per, (k + 1) * per)) for k in range(depth)]
    groups[0] += list(range(per * depth, N))
    t0 = time.perf_counter()
    for g in groups:
        t = threading.Thread(target=read_many, args=(g, res)); threads.append(t); t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    print("depth %d: wall %.3f s  (%.2f GiB in %.3f s = %.1f GiB/s)" % (depth, wall, N * SZ / 2**30, wall, N * SZ / 2**30 / wall))
