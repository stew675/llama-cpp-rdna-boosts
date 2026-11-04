// probe-overcommit -- the test the maintainer asked for: a file BIGGER THAN RAM, read in registered windows,
// cycled several times in ONE process, asking whether the previously-pinned memory really GOES AWAY.
//
// WHY: `hipHostRegister` over a model mapping makes a fill native (141 us -> 0.3-0.5 us/copy) with no
// duplicate, but the mode exists for the band where the weights do NOT fit in RAM.  `hipHostUnregister`
// alone does not release the pin (P4-ROOT-CAUSE.md 5a), so the released shape is `unregister + munmap` over a
// PRIVATE window.  That has only been tested at 64-256 MiB on a box with plenty of free RAM.  This is the
// over-commitment test: a 100 GiB file against 91 GiB of RAM, 20 GiB windows, 3 full loops in one process,
// with an explicit "are the pages droppable" check per window and a watchdog so a leak kills the test
// instead of the box.
//
// TWO READ MODES, because they are two different bands of the delivery:
//   READ=mmap   the file is read through the PAGE CACHE and the registered window is a private mapping of the
//               same pages (the page-cache band).  The page cache is deliberately NOT dropped between loops:
//               that is the whole point of the mode -- if the pages are still there the read is WARM and
//               costs no disk, and the kernel reclaims them only under pressure.  The per-window report says
//               which it was (disk MiB from /proc/self/io read_bytes), and DROPPABILITY is checked
//               separately on a 64 MiB sample of each released window (fadvise that sample, then a fresh map
//               must re-read it from disk).  That way "the pin leaked" cannot hide behind the page cache
//               holding the pages anyway.
//   READ=dio    the file is read with O_DIRECT into an ANONYMOUS window buffer (the not-fitting band: the
//               page cache is bypassed entirely, nothing is cached, the buffer IS the memory).  Verification
//               is that the buffer's memory comes back after unregister + munmap -- MemAvailable must
//               return to its baseline after every window, and a 20 GiB leak per window would exhaust the
//               box within the first loop.
//
// Per window: [read] -> hipHostRegister -> H2D copy of the whole window (the fill) + device-side sample
// check -> hipHostUnregister -> munmap -> release/verify -> report MemAvailable / swap.
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-overcommit probe-overcommit.cpp
//   run:
//     FILE=/var/tmp/overcommit-100g.bin READ=mmap /tmp/probe-overcommit 20480 3
//     FILE=/var/tmp/overcommit-100g.bin READ=dio  /tmp/probe-overcommit 20480 3
//     NOREG=1 ...                                   # control arm: same loop, no registration
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static const size_t PAGE = 4096;
static const size_t DIO_CHUNK = 8u << 20;

static long meminfo_kib(const char * key) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    const size_t klen = strlen(key);
    long v = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (!strncmp(line, key, klen)) { v = atol(line + klen); break; }
    }
    fclose(f);
    return v;
}
static long long read_bytes() {
    FILE * f = fopen("/proc/self/io", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    long long v = -1;
    while (fgets(line, sizeof(line), f)) { if (!strncmp(line, "read_bytes:", 11)) { v = atoll(line + 11); break; } }
    fclose(f);
    return v;
}
static long cached_mib() { return meminfo_kib("Cached:") / 1024; }
static long swap_used_mib() {
    const long t = meminfo_kib("SwapTotal:");
    const long f = meminfo_kib("SwapFree:");
    return (t < 0 || f < 0) ? -1 : (t - f) / 1024;
}
static long self_rss_mib() {
    FILE * f = fopen("/proc/self/status", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    long v = -1;
    while (fgets(line, sizeof(line), f)) { if (!strncmp(line, "VmRSS:", 6)) { v = atol(line + 6) / 1024; break; } }
    fclose(f);
    return v;
}
static double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            std::chrono::steady_clock::now() - t0).count();
}
// Fault every page in: one load per 4 KiB pulls the whole page from disk/page cache and is ~64x cheaper
// than touching every cache line.
static void fault_in(const char * p, size_t len) {
    volatile unsigned long long acc = 0;
    for (size_t o = 0; o < len; o += PAGE) { acc += (unsigned char) p[o]; }
    (void) acc;
}
// Read a file mapping EFFICIENTLY.  The naive "one load per page" walk depends on the kernel's readahead to
// be fast: it maps ~64 KiB per fault while readahead works, but under memory pressure the readahead
// allocations fail and it degenerates to one 4 KiB read per fault (~25 MB/s measured on this box) -- which
// looks like a broken kernel/HIP and is neither.  MADV_WILLNEED queues readahead for the whole range up
// front, and the walk then only pays the minor faults.
static void read_mapping(const char * name, char * p, size_t len, size_t chunk) {
    const auto t0 = std::chrono::steady_clock::now();
    size_t done = 0;
    while (done < len) {
        const size_t n = (len - done) < chunk ? (len - done) : chunk;
        (void) madvise(p + done, n, MADV_WILLNEED);
        fault_in(p + done, n);
        done += n;
        if (len >= (size_t) 8 << 30) {   // only narrate multi-GiB reads
            printf("    %s: %5zu / %5zu MiB  %6.1f s  %5.2f GB/s  MemAvailable %ld MiB  swap %ld MiB\n",
                   name, done >> 20, len >> 20, ms_since(t0) / 1000.0, (double) done / 1e9 / (ms_since(t0) / 1000.0),
                   meminfo_kib("MemAvailable:") / 1024, swap_used_mib());
            fflush(stdout);
        }
    }
}

static bool odirect_read(int fd, char * buf, size_t len, size_t off) {
    size_t done = 0;
    while (done < len) {
        const size_t chunk = (len - done) < DIO_CHUNK ? (len - done) : DIO_CHUNK;
        const ssize_t n = pread(fd, buf + done, chunk, (off_t) (off + done));
        if (n <= 0) { perror("pread(O_DIRECT)"); return false; }
        done += (size_t) n;
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * file = getenv("FILE");
    if (file == nullptr) { fprintf(stderr, "set FILE=<big file>\n"); return 1; }
    const size_t win_mib = argc > 1 ? (size_t) atol(argv[1]) : 20480;
    const int    loops   = argc > 2 ? atoi(argv[2]) : 3;
    const bool   reg     = getenv("NOREG") == nullptr;
    // COPYMIB: how much of each window to actually DMA (default: the whole window).  COPYMIB=0 skips the
    // copy entirely, which is how the cache-eviction question is bisected: is it the read, the registration,
    // or the DMA that makes the next loop cold?
    const size_t copy_mib = getenv("COPYMIB") ? (size_t) atol(getenv("COPYMIB")) : win_mib;
    const char * mode    = getenv("READ") ? getenv("READ") : "mmap";
    const bool   dio     = strcmp(mode, "dio") == 0;
    const long   min_avail_mib = getenv("MINAVAIL_MIB") ? atol(getenv("MINAVAIL_MIB")) : 4096;
    const long   max_swap_mib  = getenv("MAXSWAP_MIB")  ? atol(getenv("MAXSWAP_MIB"))  : 24576;

    const int fd = open(file, O_RDONLY | (dio ? O_DIRECT : 0));
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    const size_t fsize = (size_t) st.st_size;
    const size_t win   = win_mib << 20;
    size_t nwin  = fsize / win;
    // NWIN=N uses only the first N windows per loop: with 3 x 20 GiB in a box that has ~84 GiB available the
    // pages FIT, so the next loop's reads must come back WARM from the page cache (that is the check that the
    // page cache is genuinely usable again after the pin is released -- not just that the pin is gone).
    if (getenv("NWIN") != nullptr) {
        const size_t cap = (size_t) atol(getenv("NWIN"));
        if (cap > 0 && cap < nwin) { nwin = cap; }
    }
    if (nwin == 0) { fprintf(stderr, "file smaller than one window\n"); return 1; }
    printf("  (using %zu window(s) = %.1f GiB per loop)\n", nwin, (double) (nwin * win) / 1073741824.0);

    void * d = nullptr;
    HIPCK(hipSetDevice(0));
    HIPCK(hipMalloc(&d, win));
    hipStream_t s;
    HIPCK(hipStreamCreate(&s));

    // PIN_MIB: a permanent registration, kept for the life of the process, before any window is read.
    char * pinned = nullptr;
    size_t pinned_len = 0;
    if (getenv("PIN_MIB") != nullptr && atol(getenv("PIN_MIB")) > 0) {
        pinned_len = (size_t) atol(getenv("PIN_MIB")) << 20;
        const size_t poff = (size_t) (getenv("PIN_OFF_MIB") ? atol(getenv("PIN_OFF_MIB")) : 51200) << 20;
        printf("  PIN_MIB: registering %zu MiB at file offset %zu MiB and NEVER releasing it ...\n",
               pinned_len >> 20, poff >> 20);
        pinned = (char *) mmap(nullptr, pinned_len, PROT_READ, MAP_SHARED, fd, (off_t) poff);
        if (pinned == MAP_FAILED) { perror("mmap pin"); return 1; }
        // REGISTER FIRST, then read.  Registering a range whose pages were just faulted in with readahead
        // can come back `hipErrorInvalidValue` (measured: a 50 GiB read-then-register in the same process),
        // while the same range registers fine when the runtime is the one that faults the pages in -- and a
        // retry loop covers the transient case.  Registration itself pulls the pages in, so the explicit
        // read is only for the "is it really resident" accounting.
        const auto tp0 = std::chrono::steady_clock::now();
        hipError_t pe = hipHostRegister(pinned, pinned_len, hipHostRegisterReadOnly | hipHostRegisterPortable);
        for (int attempt = 1; pe != hipSuccess && attempt <= 3; attempt++) {
            printf("  PIN_MIB: attempt %d -> %s; retrying in 1 s\n", attempt, hipGetErrorString(pe));
            fflush(stdout);
            sleep(1);
            pe = hipHostRegister(pinned, pinned_len, hipHostRegisterReadOnly | hipHostRegisterPortable);
        }
        printf("  PIN_MIB: register %.2f s -> %s; MemAvailable %ld MiB, swap used %ld MiB\n",
               ms_since(tp0) / 1000.0, pe == hipSuccess ? "OK (permanently pinned)" : hipGetErrorString(pe),
               meminfo_kib("MemAvailable:") / 1024, swap_used_mib());
        fflush(stdout);
        if (pe == hipSuccess && getenv("PIN_READ") != nullptr && atol(getenv("PIN_READ")) != 0) {
            read_mapping("pin read", pinned, pinned_len, (size_t) 256 << 20);
        }
        if (pe != hipSuccess) { return 2; }
    }

    printf("probe-overcommit: %s (%.1f GiB), READ=%s, %zu MiB windows x %zu = %.1f GiB per loop, %d loop(s), "
           "registration %s\n", file, (double) fsize / 1073741824.0, mode, win_mib, nwin,
           (double) (nwin * win) / 1073741824.0, loops, reg ? "ON" : "OFF (control)");
    printf("  RAM: MemTotal %ld MiB, MemAvailable %ld MiB, swap used %ld MiB; watchdog: abort below %ld MiB "
           "available or above %ld MiB swap\n", meminfo_kib("MemTotal:") / 1024,
           meminfo_kib("MemAvailable:") / 1024, swap_used_mib(), min_avail_mib, max_swap_mib);
    const long avail0 = meminfo_kib("MemAvailable:") / 1024;
    printf("\n  %-9s %-4s %9s %12s %-4s %8s %8s %7s %10s %9s %8s %8s %8s\n", "window", "reg", "read_ms",
           "read_disk", "src", "reg_ms", "copy_ms", "GB/s", "dropcheck", "Avail_MiB", "Cached_MiB", "Swap_MiB",
           "RSS_MiB");

    long long verify_fail = 0, verify_ok = 0, copy_fail = 0;
    long min_avail = 1L << 30, max_swap = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (int lp = 0; lp < loops; lp++) {
        long loop_disk_mib = 0;
        for (size_t w = 0; w < nwin; w++) {
            const size_t off = w * win;
            const auto t_w = std::chrono::steady_clock::now();
            char * m = nullptr;
            const long long rb0 = read_bytes();
            if (dio) {
                m = (char *) mmap(nullptr, win, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (m == MAP_FAILED) { perror("mmap anon"); return 1; }
            } else {
                m = (char *) mmap(nullptr, win, PROT_READ, MAP_SHARED, fd, (off_t) off);
                if (m == MAP_FAILED) { perror("mmap file"); return 1; }
            }
            if (dio) {
                if (!odirect_read(fd, m, win, off)) { return 1; }
            } else {
                read_mapping("window", m, win, (size_t) 256 << 20);   // READ through the page cache
            }
            const long long rb1 = read_bytes();
            const double read_ms = ms_since(t_w);
            const long read_mib = (long) ((rb1 - rb0) >> 20);   // what the READ actually pulled from disk

            double reg_ms = 0.0, unreg_ms = 0.0, copy_ms = 0.0;
            if (reg) {
                const auto t0 = std::chrono::steady_clock::now();
                const hipError_t e = hipHostRegister(m, win, hipHostRegisterReadOnly | hipHostRegisterPortable);
                reg_ms = ms_since(t0);
                if (e != hipSuccess) {
                    printf("  window %zu: hipHostRegister FAILED: %s\n", w, hipGetErrorString(e));
                    return 2;
                }
            }
            const size_t copy_bytes = copy_mib == 0 ? 0 : (copy_mib << 20) < win ? (copy_mib << 20) : win;
            if (copy_bytes > 0) {
                const auto t1 = std::chrono::steady_clock::now();
                HIPCK(hipMemcpyAsync(d, m, copy_bytes, hipMemcpyHostToDevice, s));   // the FILL
                HIPCK(hipStreamSynchronize(s));
                copy_ms = ms_since(t1);
            }
            for (int k = 0; k < 4 && copy_bytes == win; k++) {   // device-side sample check, 4 x 4 KiB
                const size_t so = (win / 4) * (size_t) k;
                std::vector<char> back(4096);
                HIPCK(hipMemcpy(back.data(), (char *) d + so, 4096, hipMemcpyDeviceToHost));
                if (memcmp(back.data(), m + so, 4096) != 0) {
                    copy_fail++;
                    printf("  window %zu: DEVICE SAMPLE MISMATCH at +%zu\n", w, so);
                }
            }
            if (reg) {
                const auto t2 = std::chrono::steady_clock::now();
                HIPCK(hipHostUnregister(m));
                unreg_ms = ms_since(t2);
            }
            munmap(m, win);                        // release the VMA: the pin dies with it (P4 5a)

            long long vmb = -1;
            long ret_mib = 0;
            long avail = 0;
            long swapm = 0;
            if (dio) {
                // nothing was cached: the buffer's memory must simply be GONE.  Check MemAvailable against
                // the run's baseline (a leak accumulates 20 GiB per window, so it cannot hide).
                avail  = meminfo_kib("MemAvailable:") / 1024;
                swapm  = swap_used_mib();
                ret_mib = avail - avail0;
                const bool returned = ret_mib > -(long) (win_mib / 4);
                if (returned) { verify_ok++; } else { verify_fail++; }
                printf("  %2d/%-6zu %-4s %9.0f %8ld MiB %-4s %8.1f %8.0f %7.2f %8s %9ld %8ld %8ld\n", lp, w,
                       reg ? "yes" : "no", read_ms, read_mib, "DIO", reg_ms, copy_ms,
                       (double) win / 1e9 / (copy_ms / 1000.0), returned ? "ok" : "NO", avail, swapm,
                       self_rss_mib());
            } else {
                // DROPPABILITY CHECK, on a sample only so the rest of the window stays in the page cache and
                // the next loop's read shows the cache's real behaviour: drop a 64 MiB sample of the range we
                // have just released, then a fresh map of it must re-read those pages FROM DISK.
                const size_t vsz = win < (64u << 20) ? win : (64u << 20);
                const size_t voff = off + win - vsz;   // the tail of the window
                (void) posix_fadvise(fd, (off_t) voff, (off_t) vsz, POSIX_FADV_DONTNEED);
                const long long vb0 = read_bytes();
                char * v = (char *) mmap(nullptr, vsz, PROT_READ, MAP_SHARED, fd, (off_t) voff);
                fault_in(v, vsz);
                const long long vb1 = read_bytes();
                munmap(v, vsz);
                vmb = (vb1 - vb0) >> 20;
                const bool dropped = vmb >= (long long) (vsz >> 21);
                if (dropped) { verify_ok++; } else { verify_fail++; }
                avail  = meminfo_kib("MemAvailable:") / 1024;
                swapm  = swap_used_mib();
                ret_mib = avail - avail0;
                printf("  %2d/%-6zu %-4s %9.0f %8ld MiB %-4s %8.1f %8.0f %7.2f %6lld MiB %9ld %8ld %8ld %8ld\n",
                       lp, w, reg ? "yes" : "no", read_ms, read_mib,
                       read_mib > (long) (win_mib / 2) ? "COLD" : read_mib < 8 ? "warm" : "part",
                       reg_ms, copy_ms, (double) win / 1e9 / (copy_ms / 1000.0), vmb, avail, cached_mib(),
                       swapm, self_rss_mib());
            }

            if (avail < min_avail) { min_avail = avail; }
            if (swapm > max_swap) { max_swap = swapm; }
            if (avail < min_avail_mib || swapm > max_swap_mib) {
                printf("\n  WATCHDOG: aborting at loop %d window %zu -- MemAvailable %ld MiB (floor %ld), swap "
                       "%ld MiB (ceiling %ld).  The memory is NOT being released.\n", lp, w, avail,
                       min_avail_mib, swapm, max_swap_mib);
                return 5;
            }
            loop_disk_mib += read_mib;
            (void) unreg_ms;
        }
        const double loop_mib = (double) (nwin * win) / 1048576.0;
        printf("  loop %d total: read %ld MiB from disk of %.0f MiB (%.0f%% from the page cache)\n\n", lp,
               loop_disk_mib, loop_mib, 100.0 * (1.0 - (double) loop_disk_mib / loop_mib));
    }

    const double total_s = ms_since(t_start) / 1000.0;
    printf("\n  %s: %zu windows, %lld verified released, %lld NOT released, %lld device-sample mismatches\n",
           verify_fail == 0 && copy_fail == 0 ? "PASS" : "FAIL", nwin * (size_t) loops, verify_ok, verify_fail,
           copy_fail);
    printf("  MemAvailable floor %ld MiB (baseline %ld), peak swap used %ld MiB, total %.1f s over %.1f GiB read\n",
           min_avail, avail0, max_swap, total_s, (double) (nwin * win * (size_t) loops) / 1073741824.0);
    return (verify_fail == 0 && copy_fail == 0) ? 0 : 3;
}
