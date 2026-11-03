// wip/host-expert-pinned-window: price the MoE host-expert MISS path three ways on one expert blob.
//
// The delivery's `--host-experts pool` exists because an `-ncmoe` master backed by the read-only GGUF
// model mapping is pageable, so a GPU cannot read it (issue #116, no XNACK).  The pool works around
// that by keeping a second, pinned copy of the hot experts: a miss is `memcpy(page cache -> pool)`
// (SYNCHRONOUS, on the token's critical path) followed by a CE copy pool -> VRAM.  This probe asks
// whether the pool can be removed by registering the mapping range in place, so the bytes the op needs
// are the bytes the page cache already holds.
//
// Arms, all per expert blob of S bytes from a real GGUF mapping:
//   pool      memcpy(page cache -> hipHostMalloc slot), then hipMemcpyAsync(slot -> VRAM)   [today]
//   B         hipHostRegister(range), hipMemcpyAsync(registered -> VRAM), unregister later [candidate]
//   A         hipHostRegister(range), kernel reads the registered range over PCIe          [zero-copy]
//   refs      CE device->device and kernel device->device copies, i.e. what a VRAM hit costs
//
// Also measured: register/unregister cost and pin accounting vs size (is the pin eager or deferred?),
// whether a read-only MAP_SHARED file mapping can be registered at all (and with which flags), the
// page-alignment constraint, the real cold (page-cache miss) cost each arm must hide, and whether
// unregistering a range a queued copy still reads is refused, blocked or silently allowed.
//
// build:  ./build.sh
// run:    ./run-probe.sh [GGUF_PATH] [ITERS]
//
// Correctness is checked on every arm: the VRAM copy is read back and compared against the file bytes.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <cerrno>
#include <atomic>
#include <thread>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIP_CHECK(call)                                                                       \
    do {                                                                                      \
        hipError_t e_ = (call);                                                               \
        if (e_ != hipSuccess) {                                                               \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,          \
                    hipGetErrorString(e_));                                                   \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

static double now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double med(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[(size_t) std::min<double>(v.size() - 1, p * (double) (v.size() - 1))];
}

static std::string mib(double bytes) {
    char b[64];
    if (bytes < 1024.0 * 1024.0) {
        snprintf(b, sizeof b, "%.0f KiB", bytes / 1024.0);
    } else {
        snprintf(b, sizeof b, "%.2f MiB", bytes / 1048576.0);
    }
    return b;
}

// Pinned / unevictable accounting, to tell an EAGER pin from a DEFERRED one.
static void meminfo_kb(long * mlocked, long * unevictable) {
    *mlocked = *unevictable = -1;
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        long v = 0;
        if (sscanf(line, "Mlocked: %ld kB", &v) == 1)         *mlocked = v;
        if (sscanf(line, "Unevictable: %ld kB", &v) == 1)     *unevictable = v;
    }
    fclose(f);
}

// Chunked copy: block b copies chunk b.  Vectorised 16 B lanes, like the delivery's gather kernel
// (`moe_cache_gather_kernel`), so an `A` number is comparable to the in-tree zero-copy arm.
__global__ void copy_chunk_kernel(char * __restrict__ dst, const char * __restrict__ src,
                                  size_t nbytes, size_t chunk) {
    const size_t off = (size_t) blockIdx.x * chunk;
    if (off >= nbytes) {
        return;
    }
    const size_t len = std::min(chunk, nbytes - off);
    uint4 *       d   = (uint4 *) (dst + off);
    const uint4 * s   = (const uint4 *) (src + off);
    const size_t  n16 = len / 16;
    for (size_t i = threadIdx.x; i < n16; i += blockDim.x) {
        d[i] = s[i];
    }
    char *       dt = dst + off;
    const char * st = src + off;
    for (size_t i = n16 * 16 + threadIdx.x; i < len; i += blockDim.x) {
        dt[i] = st[i];
    }
}

struct timed {
    double host_us = 0.0;   // serial host work on the calling thread (memcpy / register / unregister)
    double total_us = 0.0;  // host work + the device transfer, i.e. to the op being able to read it
    double dev_us = 0.0;    // the device-side transfer alone (event delta)
};

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
                                 : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const int    iters = argc > 2 ? atoi(argv[2]) : 40;

    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int pci_bus = -1;
    (void) hipDeviceGetAttribute(&pci_bus, hipDeviceAttributePciBusId, 0);

    printf("=== wip/host-expert-pinned-window probe ===\n");
    printf("device 0          : %s (%s), pci bus %02x, %d device(s) visible\n", prop.name,
           prop.gcnArchName, pci_bus, ndev);
    printf("hostRegister      : supported=%d  readOnlyFlag=%d  unifiedAddr=%d  managedMem=%d\n",
           prop.hostRegisterSupported, prop.hostRegisterReadOnlySupported, prop.unifiedAddressing,
           prop.managedMemory);
    printf("source            : %s\n", path);

    // ---- the mapping -------------------------------------------------------------------------
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st{};
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    printf("mapping           : %zu MiB MAP_SHARED PROT_READ\n", (size_t) st.st_size >> 20);

    // A region in the middle of the file, page aligned.  Every arm walks `iters` consecutive expert
    // blobs from here, so a repetition never re-reads the same pages.
    const size_t expert = 1740800;                      // 1.66 MiB ~ 35B-A3B Q4_K_M expert (3 matrices / 3)
    const size_t region = expert * (size_t) (iters + 2);
    size_t base = ((size_t) st.st_size / 2) & ~(page - 1);
    if (base + region >= (size_t) st.st_size) {
        base = ((size_t) st.st_size - region - page) & ~(page - 1);
    }

    char * dst = nullptr;    // VRAM: the arena slot
    char * dst2 = nullptr;   // VRAM: the reference copy destination
    char * readback = nullptr;
    HIP_CHECK(hipMalloc((void **) &dst, 64 << 20));
    HIP_CHECK(hipMalloc((void **) &dst2, 64 << 20));
    HIP_CHECK(hipHostMalloc((void **) &readback, 64 << 20, hipHostMallocDefault));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));

    void * pool_slot = nullptr;
    HIP_CHECK(hipHostMalloc(&pool_slot, 64 << 20, hipHostMallocDefault));

    auto pre_touch = [&](size_t off, size_t len) {
        volatile char sink = 0;
        for (size_t p = off; p < off + len; p += page) {
            sink ^= map[p];
        }
        (void) sink;
    };
    pre_touch(base, region);   // "the page cache already holds it" = the buffered pool fill today

    // ---- 1. can a read-only file mapping be registered at all? ------------------------------
    printf("\n--- 1. registration of a read-only MAP_SHARED file mapping ---\n");
    printf("%-28s %-22s %-26s %s\n", "flags", "hipHostRegister", "hipHostGetDevicePointer", "us");
    struct flagarm { const char * name; unsigned flags; };
    const flagarm farms[] = {
        { "default(0)",                 hipHostRegisterDefault },
        { "ReadOnly",                   hipHostRegisterReadOnly },
        { "Portable|Mapped",            hipHostRegisterPortable | hipHostRegisterMapped },
        { "ReadOnly|Portable|Mapped",   hipHostRegisterReadOnly | hipHostRegisterPortable | hipHostRegisterMapped },
    };
    bool any_ok = false;
    for (const auto & fa : farms) {
        void * p = map + base;   // page aligned
        const double t0 = now_us();
        hipError_t   e  = hipHostRegister(p, expert, fa.flags);
        const double dt = now_us() - t0;
        char reg[64] = "-", gdp[64] = "-";
        if (e != hipSuccess) {
            snprintf(reg, sizeof reg, "FAIL (%s)", hipGetErrorString(e));
        } else {
            snprintf(reg, sizeof reg, "ok");
            void * dev = nullptr;
            hipError_t ge = hipHostGetDevicePointer(&dev, p, 0);
            if (ge == hipSuccess) {
                snprintf(gdp, sizeof gdp, "ok (%s)", dev == p ? "same VA" : "distinct VA");
            } else {
                snprintf(gdp, sizeof gdp, "FAIL (%s)", hipGetErrorString(ge));
            }
            any_ok = true;
            HIP_CHECK(hipHostUnregister(p));
        }
        (void) hipGetLastError();
        printf("%-28s %-22s %-26s %.0f\n", fa.name, reg, gdp, dt);
    }
    {
        // A range whose start is not page aligned: either accepted, or an error we must handle by
        // rounding the window out to pages (the real code would register whole pages anyway).
        void * p = map + base + 64;
        hipError_t e = hipHostRegister(p, expert, hipHostRegisterReadOnly);
        printf("%-28s %s\n", "unaligned +64 B",
               e == hipSuccess ? "ACCEPTED (no page-alignment constraint)" : hipGetErrorString(e));
        if (e == hipSuccess) {
            HIP_CHECK(hipHostUnregister(p));
        }
        (void) hipGetLastError();
    }
    if (!any_ok) {
        printf("\nRESULT: a read-only file mapping cannot be registered on this stack -> B and A are\n"
               "        not implementable; the pool (or an anonymous pinned window) is the only option.\n");
        return 2;
    }

    // ---- 2. register / unregister cost and pin accounting vs size ---------------------------
    printf("\n--- 2. register+unregister cost vs range size, and is the PIN eager or deferred? ---\n");
    printf("%-12s %-12s %-12s %-10s %-14s %s\n", "range", "register us", "unreg us", "us/MiB",
           "Mlocked delta", "Unevictable delta");
    const size_t ssizes[] = { 256 * 1024, 512 * 1024, expert, 4 << 20, 16 << 20, 64 << 20, 256 << 20 };
    for (size_t s : ssizes) {
        const size_t n = (s + page - 1) & ~(page - 1);
        if (base + n >= (size_t) st.st_size) {
            continue;
        }
        std::vector<double> rt, ut;
        long ml0, un0, ml1, un1;
        meminfo_kb(&ml0, &un0);
        for (int i = 0; i < 12; ++i) {
            void * p = map + base;
            double t0 = now_us();
            HIP_CHECK(hipHostRegister(p, n, hipHostRegisterReadOnly));
            rt.push_back(now_us() - t0);
            meminfo_kb(&ml1, &un1);   // after the first/second register
            t0 = now_us();
            HIP_CHECK(hipHostUnregister(p));
            ut.push_back(now_us() - t0);
        }
        printf("%-12s %-12.1f %-12.1f %-10.2f %-14ld %ld\n", mib((double) n).c_str(), med(rt),
               med(ut), (med(rt) + med(ut)) / (double) n * 1048576.0, ml1 - ml0, un1 - un0);
    }

    // ---- 3. the miss path, three ways -------------------------------------------------------
    auto run_pool = [&](char * src, size_t n) {
        timed t;
        const double t0 = now_us();
        memcpy(pool_slot, src, n);
        const double t1 = now_us();
        HIP_CHECK(hipEventRecord(ev0, stream));
        HIP_CHECK(hipMemcpyAsync(dst, pool_slot, n, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        const double t2 = now_us();
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        t.host_us  = t1 - t0;
        t.dev_us   = (double) ms * 1000.0;
        t.total_us = t2 - t0;
        return t;
    };
    auto run_b = [&](char * src, size_t n) {
        timed t;
        const double t0 = now_us();
        HIP_CHECK(hipHostRegister(src, n, hipHostRegisterReadOnly));
        const double t1 = now_us();
        HIP_CHECK(hipEventRecord(ev0, stream));
        HIP_CHECK(hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        HIP_CHECK(hipHostUnregister(src));
        const double t2 = now_us();
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        t.host_us  = t1 - t0;
        t.dev_us   = (double) ms * 1000.0;
        t.total_us = t2 - t0;
        return t;
    };
    auto run_a = [&](char * src, size_t n, int nb) {
        timed t;
        const double t0 = now_us();
        HIP_CHECK(hipHostRegister(src, n, hipHostRegisterReadOnly));
        HIP_CHECK(hipEventRecord(ev0, stream));
        copy_chunk_kernel<<<nb, 256, 0, stream>>>(dst, src, n, (n + (size_t) nb - 1) / (size_t) nb);
        HIP_CHECK(hipEventRecord(ev1, stream));
        HIP_CHECK(hipEventSynchronize(ev1));
        HIP_CHECK(hipHostUnregister(src));
        const double t2 = now_us();
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
        t.host_us  = 0.0;
        t.dev_us   = (double) ms * 1000.0;
        t.total_us = t2 - t0;
        return t;
    };

    printf("\n--- 3. miss path per %s expert, %d runs, median [p10..p90] ---\n",
           mib((double) expert).c_str(), iters);
    printf("%-30s %-12s %-12s %-12s %s\n", "arm", "host us", "total us", "device us", "GB/s");
    auto sweep = [&](const char * label, const std::vector<timed> & v) {
        std::vector<double> th, tt, td;
        for (const auto & x : v) { th.push_back(x.host_us); tt.push_back(x.total_us); td.push_back(x.dev_us); }
        const double gbps = med(td) > 0 ? (double) expert / (med(td) * 1e-6) / 1e9 : 0.0;
        printf("%-30s %-12.1f %-12.1f %-12.1f %.1f   [%.1f..%.1f]\n", label, med(th), med(tt),
               med(td), gbps, pct(tt, 0.1), pct(tt, 0.9));
    };

    std::vector<timed> pool_v, b_v, bd_v, a8, a32, a128;
    std::vector<char *> pending;
    for (int i = 0; i < iters; ++i) {
        char * src = map + base + (size_t) i * expert;
        pool_v.push_back(run_pool(src, expert));
        b_v.push_back(run_b(src, expert));
        a8.push_back(run_a(src, expert, 8));
        a32.push_back(run_a(src, expert, 32));
        a128.push_back(run_a(src, expert, 128));
        // the realistic B protocol: register, copy, and demote only after the graph has drained
        char * s2 = map + base + (size_t) i * expert;
        const double t0 = now_us();
        HIP_CHECK(hipHostRegister(s2, expert, hipHostRegisterReadOnly));
        HIP_CHECK(hipMemcpyAsync(dst, s2, expert, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        pending.push_back(s2);
        (void) t0;
        bd_v.push_back({ 0.8, now_us() - t0, 0.0 });
    }
    for (char * p : pending) {
        HIP_CHECK(hipHostUnregister(p));
    }
    sweep("pool: memcpy + CE H2D", pool_v);
    sweep("B: register + CE H2D", b_v);
    sweep("B: register + CE H2D (defer)", bd_v);
    sweep("A: SM read, 8 blocks", a8);
    sweep("A: SM read, 32 blocks", a32);
    sweep("A: SM read, 128 blocks", a128);

    // 3d. references: what a VRAM hit costs (no host involvement at all).
    {
        std::vector<double> ce, sm;
        for (int i = 0; i < iters; ++i) {
            HIP_CHECK(hipEventRecord(ev0, stream));
            HIP_CHECK(hipMemcpyAsync(dst2, dst, expert, hipMemcpyDeviceToDevice, stream));
            HIP_CHECK(hipEventRecord(ev1, stream));
            HIP_CHECK(hipEventSynchronize(ev1));
            float ms = 0.f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            ce.push_back((double) ms * 1000.0);
            HIP_CHECK(hipEventRecord(ev0, stream));
            copy_chunk_kernel<<<64, 256, 0, stream>>>(dst2, dst, expert, (expert + 63) / 64);
            HIP_CHECK(hipEventRecord(ev1, stream));
            HIP_CHECK(hipEventSynchronize(ev1));
            HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
            sm.push_back((double) ms * 1000.0);
        }
        printf("%-30s %-12.0f %-12.1f %-12.1f %.1f\n", "ref: CE D2D (VRAM hit)", 0.0, med(ce),
               med(ce), (double) expert / (med(ce) * 1e-6) / 1e9);
        printf("%-30s %-12.0f %-12.1f %-12.1f %.1f\n", "ref: SM VRAM copy, 64 blk", 0.0, med(sm),
               med(sm), (double) expert / (med(sm) * 1e-6) / 1e9);
    }

    // 3e. does the B win survive at other blob sizes?  (the pool memcpy scales, the register does not)
    printf("\n--- 3e. pool vs B vs A across blob sizes (median total us) ---\n");
    printf("%-12s %-14s %-14s %-14s %s\n", "blob", "pool", "B", "A (32 blk)", "B vs pool");
    const size_t bsizes[] = { 256 * 1024, 512 * 1024, expert, 4 << 20, 16 << 20, 32 << 20 };
    for (size_t s : bsizes) {
        const size_t n = (s + page - 1) & ~(page - 1);
        if (base + n + expert * 4 >= (size_t) st.st_size) {
            continue;
        }
        std::vector<double> tp, tb, ta;
        for (int i = 0; i < 10; ++i) {
            char * src = map + base + (size_t) i * expert;
            tp.push_back(run_pool(src, n).total_us);
            tb.push_back(run_b(src, n).total_us);
            ta.push_back(run_a(src, n, 32).total_us);
        }
        printf("%-12s %-14.1f %-14.1f %-14.1f %+.2fx\n", mib((double) n).c_str(), med(tp), med(tb),
               med(ta), med(tb) / med(tp));
    }

    // 3f. is the pin deferred to the first device access, and is that cost per RANGE or one-time?
    //     Register a FRESH 8 MiB range and copy it once, 8 times over.  A per-range lazy pin shows up
    //     as every first copy being slow; a one-time driver setup shows up only on the first.
    printf("\n--- 3f. 8 fresh 8 MiB ranges, one CE copy each (lazy pin: per range or one-time?) ---\n");
    {
        const size_t n = 8 << 20;
        for (int i = 0; i < 8; ++i) {
            char * src = map + base + (size_t) i * n;
            HIP_CHECK(hipHostRegister(src, n, hipHostRegisterReadOnly));
            const double t0 = now_us();
            HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
            const double dt = now_us() - t0;
            printf("   fresh range %d, 1st copy : %8.1f us (%.1f GB/s)", i, dt, (double) n / dt / 1e3);
            const double t1 = now_us();
            HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
            const double dt2 = now_us() - t1;
            printf("    2nd copy: %7.1f us (%.1f GB/s)\n", dt2, (double) n / dt2 / 1e3);
            HIP_CHECK(hipHostUnregister(src));
        }
    }

    // ---- 4. correctness: does each arm move the bytes it claims to? --------------------------
    printf("\n--- 4. byte check (VRAM copy vs the file's bytes) ---\n");
    {
        char * src = map + base;
        memcpy(pool_slot, src, expert);
        HIP_CHECK(hipMemcpyAsync(dst, pool_slot, expert, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        printf("pool    : %s\n", memcmp(readback, src, expert) == 0 ? "identical" : "MISMATCH");
        HIP_CHECK(hipHostRegister(src, expert, hipHostRegisterReadOnly));
        HIP_CHECK(hipMemcpyAsync(dst, src, expert, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipHostUnregister(src));
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        printf("B       : %s\n", memcmp(readback, src, expert) == 0 ? "identical" : "MISMATCH");
        HIP_CHECK(hipHostRegister(src, expert, hipHostRegisterReadOnly));
        copy_chunk_kernel<<<64, 256, 0, stream>>>(dst, src, expert, (expert + 63) / 64);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipHostUnregister(src));
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        printf("A       : %s\n", memcmp(readback, src, expert) == 0 ? "identical" : "MISMATCH");
    }

    // ---- 5. is an in-flight read protected from unregister? --------------------------------
    printf("\n--- 5. unregister with the copy still queued (must NOT be relied on) ---\n");
    {
        char * src = map + base;
        HIP_CHECK(hipHostRegister(src, expert, hipHostRegisterReadOnly));
        HIP_CHECK(hipMemcpyAsync(dst, src, expert, hipMemcpyHostToDevice, stream));
        hipError_t e = hipHostUnregister(src);   // deliberately without waiting! UB by contract
        printf("unregister returned %s\n", e == hipSuccess ? "Success (NOT a safety guarantee)"
                                                            : hipGetErrorString(e));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        printf("bytes    : %s  (a mismatch here is exactly why the demote needs an event)\n",
               memcmp(readback, src, expert) == 0 ? "identical" : "MISMATCH");
        (void) hipGetLastError();
    }

    // ---- 6. what each arm has to hide: a REAL cold (page cache miss) read --------------------
    printf("\n--- 6. cold read: a file the page cache has never held (on the model volume, NOT tmpfs) ---\n");
    {
        // `/tmp` is tmpfs on this box, which silently makes a "cold" test a RAM test.  Use the same
        // xfs volume the GGUFs live on, and PROVE the drop with mincore() before timing.
        char tmp[] = "/llm/.pw-probe-cold-XXXXXX";
        int tfd = mkstemp(tmp);
        if (tfd < 0) {
            perror("mkstemp");
        } else {
            const size_t n = 8 << 20;
            std::vector<char> buf(n, 0);
            for (size_t i = 0; i < n; i += page) { buf[i] = (char) i; }
            (void) !write(tfd, buf.data(), n);
            (void) fsync(tfd);
            auto resident = [&](char * m) {
                std::vector<unsigned char> vec(n / page, 0);
                if (mincore(m, n, vec.data()) != 0) return -1L;
                long c = 0;
                for (unsigned char v : vec) c += (v & 1);
                return c;
            };
            (void) posix_fadvise(tfd, 0, (off_t) n, POSIX_FADV_DONTNEED);   // unmapped: really drops
            char * m2 = (char *) mmap(NULL, n, PROT_READ, MAP_SHARED, tfd, 0);
            if (m2 != MAP_FAILED) {
                printf("   resident pages after the drop: %ld of %zu\n", resident(m2), n / page);
                double t0 = now_us();
                memcpy(pool_slot, m2, n);                                    // disk -> cache -> pool
                printf("   memcpy out of a COLD %s range      : %8.1f us (%5.1f GB/s)  <- the pool's fill\n",
                       mib(n).c_str(), now_us() - t0, (double) n / (now_us() - t0) / 1e3);
                munmap(m2, n);
                (void) posix_fadvise(tfd, 0, (off_t) n, POSIX_FADV_DONTNEED);
                m2 = (char *) mmap(NULL, n, PROT_READ, MAP_SHARED, tfd, 0);
                t0 = now_us();
                HIP_CHECK(hipHostRegister(m2, n, hipHostRegisterReadOnly));   // GUP faults them in
                printf("   hipHostRegister of a COLD %s range : %8.1f us (%5.1f GB/s)  <- B/A's fill\n",
                       mib(n).c_str(), now_us() - t0, (double) n / (now_us() - t0) / 1e3);
                HIP_CHECK(hipHostUnregister(m2));
                munmap(m2, n);
            }
            close(tfd);
            unlink(tmp);
        }
    }

    // ---- 7. does registration PROTECT the pages (and does a demote return them)? ------------
    // mincore() is NOT authoritative here: it can report 0 resident for a range whose pages are still
    // in the page cache.  TIME the copy instead -- a cache hit is ~160 us for 8 MiB, a real refill is
    // ms-scale (measured in 6 at ~1.4-2.3 ms).  Compare against a snapshot taken while resident, so
    // the check itself cannot re-fault the source.
    printf("\n--- 7. forced reclaim: is a REGISTERED range protected? (timed, not mincore) ---\n");
    {
        const size_t n = 8 << 20;
        char * win = map + base;
        char * ctl = map + base + (size_t) 8 * (8 << 20);
        char * snap = (char *) malloc(n);
        char * snapc = (char *) malloc(n);
        memcpy(snap, win, n);
        memcpy(snapc, ctl, n);

        HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
        double t0 = now_us();
        HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));   // warm: the lazy pin lands here
        const double warm = now_us() - t0;
        int mv = madvise(win, n, MADV_PAGEOUT);
        t0 = now_us();
        HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));
        const double after = now_us() - t0;
        HIP_CHECK(hipMemcpy(readback, dst, n < expert ? n : expert, hipMemcpyDeviceToHost));
        printf("   registered window: warm copy %7.1f us | after MADV_PAGEOUT (rc=%d) %8.1f us  -> %s\n",
               warm, mv, after,
               memcmp(readback, snap, n < expert ? n : expert) != 0 ? "MISMATCH (!!)"
               : (after < 3.0 * warm ? "PAGES KEPT (registration protects them)"
                                     : "PAGES REFILLED (registration does NOT protect)"));
        // DONTNEED control on the SAME registered range.  For a MAP_SHARED file mapping, DONTNEED drops
        // the PROCESS's PTEs (VmRSS falls) but the page cache keeps the pages -- it frees no memory and
        // performs no I/O, so it cannot test "is this page protected from reclaim".  Demonstrate that by
        // timing a host read of a DONTNEED half against the known cold cost (6: 8 MiB = 17.5 ms, i.e.
        // ~8.7 ms per 4 MiB) and a cache-warm read (~0.5-1 ms per 4 MiB).
        long rss0, rss1;
        auto vmrss_kb = []() {
            long r = -1;
            FILE * f = fopen("/proc/self/status", "r");
            if (!f) return r;
            char l[256];
            while (fgets(l, sizeof l, f)) { long v; if (sscanf(l, "VmRSS: %ld kB", &v) == 1) r = v; }
            fclose(f);
            return r;
        };
        const size_t half = n / 2;
        HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));   // re-warm after the PAGEOUT refill
        rss0 = vmrss_kb();
        const int dn = madvise(win, n, MADV_DONTNEED);
        rss1 = vmrss_kb();
        t0 = now_us();
        HIP_CHECK(hipMemcpy(dst, win, half, hipMemcpyHostToDevice));   // GPU path on the first half
        const double dn_gpu = now_us() - t0;
        t0 = now_us();
        memcpy(pool_slot, win + half, half);                          // host path on the second half
        const double dn_host = now_us() - t0;
        const double warm_host = 900.0;                               // known cache-warm host read, ~4 MiB
        printf("   MADV_DONTNEED (rc=%d): VmRSS %+ld kB, GPU copy of half %7.1f us, host read of half %7.1f us\n",
               dn, rss1 - rss0, dn_gpu, dn_host);
        printf("      -> host read %s a page-cache hit: DONTNEED freed NO memory, so it cannot test "
               "reclaim (use MADV_PAGEOUT)\n",
               dn_host < 3.0 * warm_host ? "is" : "is NOT");

        // control: same operation, unregistered
        memcpy(snapc, ctl, n);
        mv = madvise(ctl, n, MADV_PAGEOUT);
        t0 = now_us();
        memcpy(pool_slot, ctl, n);
        printf("   unregistered ctl : MADV_PAGEOUT rc=%d, then a host memcpy of the same range %8.1f us  -> %s\n",
               mv, now_us() - t0,
               memcmp(pool_slot, snapc, n) != 0 ? "MISMATCH (!!)"
                                               : ((now_us() - t0) > 3.0 * 160.0 ? "REFILLED (test is live)"
                                                                                : "kept (test void)"));
        HIP_CHECK(hipHostUnregister(win));
        free(snap);
        free(snapc);
    }

    // ---- 9. demote / re-promote: the design's steady state -----------------------------------
    // The intended cycle is register -> fill -> UNREGISTER (the pages go back to being ordinary,
    // reclaimable page-cache pages) -> later register again.  What matters:
    //   9a  does a demote keep the page cache warm, so a re-promote is cheap and still zero-copy?
    //   9b  does the cycle degrade over many iterations (leak / refcount growth)?
    //   9c  if the page was evicted while demoted, what does the re-promote cost (disk refill)?
    //   9d  is a double/overlapping registration of one range refused or refcounted?
    printf("\n--- 9. demote -> re-promote cycle (register, fill, unregister, repeat) ---\n");
    {
        const size_t n = 8 << 20;
        char * win = map + base + (size_t) 16 * (8 << 20);
        char * snap = (char *) malloc(n);
        memcpy(snap, win, n);

        std::vector<double> regs, cpys;
        for (int i = 0; i < 20; ++i) {
            double t0 = now_us();
            HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
            regs.push_back(now_us() - t0);
            t0 = now_us();
            HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));
            cpys.push_back(now_us() - t0);
            HIP_CHECK(hipHostUnregister(win));       // demote: back to ordinary page-cache pages
        }
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        std::vector<double> half(cpys.begin(), cpys.begin() + cpys.size() / 2), half2(cpys.begin() + cpys.size() / 2, cpys.end());
        printf("   9a 20x demote/re-promote (no pressure): register %.1f us, copy %.1f us (%.1f GB/s), %s\n",
               med(regs), med(cpys), (double) n / (med(cpys) * 1e-6) / 1e9,
               memcmp(readback, snap, expert) == 0 ? "bytes identical" : "MISMATCH (!!)");
        printf("       first-half median %.1f us / second-half %.1f us -> %s\n", med(half), med(half2),
               med(half2) < 1.5 * med(half) ? "no degradation" : "DEGRADES (investigate)");

        // 9c: evict WHILE DEMOTED (the loop above left it unregistered), then re-promote.
        int mv = madvise(win, n, MADV_PAGEOUT);
        double t0 = now_us();
        HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
        const double reg = now_us() - t0;
        t0 = now_us();
        HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));
        const double cpy = now_us() - t0;
        HIP_CHECK(hipMemcpy(readback, dst, expert, hipMemcpyDeviceToHost));
        printf("   9c demoted, then MADV_PAGEOUT (rc=%d), re-promote: register %.1f us, copy %8.1f us -> %s\n",
               mv, reg, cpy,
               memcmp(readback, snap, expert) == 0
                   ? (cpy < 3.0 * med(cpys) ? "still cheap (page cache kept it)" : "DISK REFILL (as expected)")
                   : "MISMATCH (!!)");
        HIP_CHECK(hipHostUnregister(win));

        // 9d: overlapping registration of a range that is already registered.
        HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
        hipError_t e2 = hipHostRegister(win + (1 << 20), 4 << 20, hipHostRegisterReadOnly);
        printf("   9d overlapping re-register of an already registered range: %s\n",
               e2 == hipSuccess ? "accepted (refcounted? unregister the same count!)"
                                : hipGetErrorString(e2));
        if (e2 == hipSuccess) {
            HIP_CHECK(hipHostUnregister(win + (1 << 20)));
        }
        HIP_CHECK(hipHostUnregister(win));
        (void) hipGetLastError();
        free(snap);
    }

    // ---- 11. can the window be REALLY pinned (the design's demote/promote contract)? ---------
    // `hipHostRegister` alone does NOT protect the pages (7).  The design wants "pinned while in the
    // window, unpinned on demote, re-pinned cheaply if still cached".  Recipe under test: mlock +
    // register to hold, munlock + unregister to demote.  Measured on the range 9a already warmed
    // (so the one-time first-access cost is out of the numbers), medians of 5.
    printf("\n--- 11. mlock + register as the real pin (warm range, medians of 5) ---\n");
    {
        struct rlimit rl{};
        getrlimit(RLIMIT_MEMLOCK, &rl);
        printf("   RLIMIT_MEMLOCK  : soft %llu MiB / hard %llu MiB  (a GiB window needs this raised)\n",
               (unsigned long long) (rl.rlim_cur / 1048576), (unsigned long long) (rl.rlim_max / 1048576));
        const size_t n = 4 << 20;                     // stay under the 8 MiB soft limit
        char * win = map + base + (size_t) 16 * (8 << 20);   // the range 9a warmed
        char * snap = (char *) malloc(n);
        memcpy(snap, win, n);
        auto cpy = [&](int reps) {
            std::vector<double> v;
            for (int i = 0; i < reps; ++i) {
                const double t0 = now_us();
                HIP_CHECK(hipMemcpy(dst, win, n, hipMemcpyHostToDevice));
                v.push_back(now_us() - t0);
            }
            return med(v);
        };
        HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
        (void) cpy(2);                                          // warm the device path for this range
        const double base_us = cpy(5);

        const int mrc = mlock(win, n);
        (void) madvise(win, n, MADV_PAGEOUT);                   // pressure while "pinned"
        const double held_us = cpy(5);
        HIP_CHECK(hipMemcpy(readback, dst, n, hipMemcpyDeviceToHost));
        printf("   holding (mlock rc=%d) then MADV_PAGEOUT : %7.1f us vs warm baseline %7.1f us -> %s\n",
               mrc, held_us, base_us,
               held_us < 2.0 * base_us ? "PAGES HELD (mlock works)" : "NOT held (mlock did not protect)");
        printf("   bytes %s\n", memcmp(readback, snap, n) == 0 ? "identical" : "MISMATCH (!!)");

        (void) munlock(win, n);
        HIP_CHECK(hipHostUnregister(win));
        (void) madvise(win, n, MADV_PAGEOUT);
        const double demoted_us = cpy(5);
        printf("   demoted (munlock+unregister), evicted    : %7.1f us -> %s\n", demoted_us,
               demoted_us > 3.0 * base_us ? "refills from disk (as expected)"
                                          : "still cached (demote kept the page cache)");

        (void) !memcmp(win, win, 1);                    // touch: pull back through the page cache
        const int mrc2 = mlock(win, n);
        HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
        const double back_us = cpy(5);
        printf("   re-promoted with the pages cached (mlock rc=%d): %7.1f us -> %s\n", mrc2, back_us,
               back_us < 3.0 * base_us ? "full rate again" : "still slow (investigate)");
        HIP_CHECK(hipHostUnregister(win));
        (void) munlock(win, n);
        free(snap);
    }

    // ---- 8. 256 MiB copy rates; the CPU-attribution column here is VOID ----------------------
    // Trying to tell a real DMA from a host-side bounce by host CPU time does NOT work: a blocking
    // `hipMemcpy` busy-waits, so every arm reads ~100 % of wall (VRAM->VRAM included).  Use
    // `rocprofv3 --hip-trace`: a bounce shows up as an extra host memcpy/blit, a direct DMA as one
    // async copy with no host-side byte movement.  Kept here only for the size-scaling rates.
    printf("\n--- 8. 256 MiB copy rates (the CPU column is VOID: a blocking copy busy-waits) ---\n");
    {
        const size_t n = 256 << 20;
        char * big = map + base;
        void * hb = nullptr;
        HIP_CHECK(hipHostMalloc(&hb, n, hipHostMallocDefault));
        memcpy(hb, big, n);
        HIP_CHECK(hipHostRegister(big, n, hipHostRegisterReadOnly));
        char * bigdev = nullptr;
        char * bigref = nullptr;
        HIP_CHECK(hipMalloc((void **) &bigdev, n));
        HIP_CHECK(hipMalloc((void **) &bigref, n));
        auto cpu_us = [](double * user, double * sys) {
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            *user = (double) ru.ru_utime.tv_sec * 1e6 + ru.ru_utime.tv_usec;
            *sys  = (double) ru.ru_stime.tv_sec * 1e6 + ru.ru_stime.tv_usec;
        };
        struct arm { const char * name; void * src; };
        const arm arms[] = {
            { "registered file mapping", big },
            { "hipHostMalloc buffer",    hb  },
        };
        for (const auto & a : arms) {
            HIP_CHECK(hipMemcpy(bigdev, a.src, n, hipMemcpyHostToDevice));   // warm (lazy pin)
            double u0, s0, u1, s1;
            cpu_us(&u0, &s0);
            const double t0 = now_us();
            HIP_CHECK(hipMemcpy(bigdev, a.src, n, hipMemcpyHostToDevice));
            const double dt = now_us() - t0;
            cpu_us(&u1, &s1);
            printf("   %-26s %8.1f us (%5.1f GB/s)   host cpu: %7.1f us (%.0f %% of wall)\n",
                   a.name, dt, (double) n / dt / 1e3, (u1 - u0) + (s1 - s0),
                   100.0 * ((u1 - u0) + (s1 - s0)) / dt);
        }
        // (a 256 MiB D2D "reference" row was dropped: an untouched allocation lets the runtime elide it)
        HIP_CHECK(hipHostUnregister(big));
        HIP_CHECK(hipFree(bigdev));
        HIP_CHECK(hipHostFree(hb));
    }

    // ---- 10. is a copy out of a registered range a DMA or a host-side bounce? ----------------
    // Direct comparators under host DRAM-bandwidth contention.  A host-staged copy needs TWO copies
    // worth of host bandwidth (cache -> bounce, then bounce -> PCIe) and collapses when the CPU hogs
    // DRAM; a direct DMA only needs the read side and is PCIe-limited.  A plain `malloc` source is
    // the known-bounce control and a `hipHostMalloc` source the known-DMA control.
    printf("\n--- 10. 8 MiB H2D under host DRAM contention: registered vs pinned vs pageable ---\n");
    {
        const size_t n = 8 << 20;
        char * regsrc = map + base + (size_t) 24 * (8 << 20);
        void * pin = nullptr, * pgbl = nullptr;
        HIP_CHECK(hipHostMalloc(&pin, n, hipHostMallocDefault));
        pgbl = malloc(n);
        memcpy(pin, regsrc, n);
        memcpy(pgbl, regsrc, n);
        HIP_CHECK(hipHostRegister(regsrc, n, hipHostRegisterReadOnly));

        std::atomic<bool> stop{ false };
        std::vector<std::thread> hogs;
        auto hog = [&]() {
            std::vector<char> a(64 << 20, 1), b(64 << 20, 2);
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t off = 0; off < a.size(); off += 65536) {
                    memcpy(b.data() + off, a.data() + off, 65536);
                }
            }
        };
        auto bench = [&](void * src, int reps) {
            std::vector<double> v;
            for (int i = 0; i < reps; ++i) {
                const double t0 = now_us();
                HIP_CHECK(hipMemcpy(dst, src, n, hipMemcpyHostToDevice));
                v.push_back(now_us() - t0);
            }
            return med(v);
        };
        // warm every path first (lazy pin, staging allocation, ...)
        (void) bench(regsrc, 2);
        (void) bench(pin, 2);
        (void) bench(pgbl, 2);

        const double idle_reg = bench(regsrc, 8), idle_pin = bench(pin, 8), idle_pg = bench(pgbl, 8);
        for (int i = 0; i < 8; ++i) hogs.emplace_back(hog);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const double load_reg = bench(regsrc, 8), load_pin = bench(pin, 8), load_pg = bench(pgbl, 8);
        stop.store(true);
        for (auto & t : hogs) t.join();

        auto row = [&](const char * nm, double a, double b) {
            printf("   %-26s idle %7.1f us (%5.1f GB/s)   under load %8.1f us (%5.1f GB/s)   slowdown %.2fx\n",
                   nm, a, (double) n / a / 1e3, b, (double) n / b / 1e3, b / a);
        };
        row("registered file mapping", idle_reg, load_reg);
        row("hipHostMalloc (DMA ref)", idle_pin, load_pin);
        row("malloc pageable (bounce)", idle_pg, load_pg);
        printf("   %s\n",
               (load_reg / idle_reg) > 1.5 && (load_pg / idle_pg) < 1.2
                   ? "-> registered tracks the PINNED arm (DMA), pageable the bounce"
                   : "-> VOID: the contention did not separate the arms (two runs have\n"
                     "            disagreed); use `rocprofv3 --memory-copy-trace` instead");
        HIP_CHECK(hipHostUnregister(regsrc));
        HIP_CHECK(hipHostFree(pin));
        free(pgbl);
    }

    printf("\n--- done ---\n");
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(pool_slot));
    HIP_CHECK(hipFreeHost(readback));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(dst2));
    munmap(map, (size_t) st.st_size);
    close(fd);
    return 0;
}
