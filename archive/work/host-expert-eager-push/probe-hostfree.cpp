// probe-hostfree -- does `hipHostFree` give the PAGES back, or only the VMA?
//
// WHY (wip/host-expert-eager-push): a genuine `hipHostMalloc` master is the fastest transport measured, at
// every device count (prefill 645.9/1105.4 t/s, decode 53.5/72.0 against registered 478.7/951.6 and
// 53.4/71.8, and against the pool path 616.6/707.1 and 49.0/51.3).  So the plan is to use `hipHostMalloc`
// everywhere -- `pinned` (the whole set) and `pool` (a bounded, rotated set) -- and the one thing that must
// hold for the bounded mode is that a freed region's pages really return to the kernel, otherwise the
// bounded tier is a slow leak and the host ends up swapping (exactly what `hipHostUnregister` does: it
// returns success and leaks the pin -- P4 5a).
//
// METHOD, per cycle:
//   allocate N chunks of M MiB with hipHostMalloc -> touch every page (forces the pin) -> record
//   /proc/meminfo  -> hipHostFree every chunk -> record again.  Then REUSE test: read a large file through
//   the page cache and confirm the freed bytes are actually available to it.
//
// The signal is MemFree/MemAvailable (a pinned allocation is non-swappable, so it MUST show up there; note
// VmLck does not account HIP pins on this stack, which is what made the earlier probe useless).
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-hostfree probe-hostfree.cpp
//   run:
//     /tmp/probe-hostfree 184 102 3        # 184 MiB chunks, 102 of them (~18.3 GiB), 3 cycles
//     REUSE_FILE=/var/tmp/big.bin /tmp/probe-hostfree 184 102 2
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <chrono>

static long meminfo_mib(const char * key) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    const size_t klen = strlen(key);
    long v = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (!strncmp(line, key, klen)) { v = atol(line + klen) / 1024; break; }
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
static void show(const char * what) {
    printf("  %-30s MemFree %6ld MiB   MemAvailable %6ld MiB   swap used %5ld MiB\n", what,
           meminfo_mib("MemFree:"), meminfo_mib("MemAvailable:"),
           (meminfo_mib("SwapTotal:") - meminfo_mib("SwapFree:")));
}

int main(int argc, char ** argv) {
    const size_t chunk_mib = argc > 1 ? (size_t) atol(argv[1]) : 184;
    const int    nchunks   = argc > 2 ? atoi(argv[2]) : 102;
    const int    cycles    = argc > 3 ? atoi(argv[3]) : 3;
    const size_t chunk = chunk_mib << 20;

    hipError_t e0 = hipSetDevice(0);
    if (e0 != hipSuccess) { printf("hipSetDevice: %s\n", hipGetErrorString(e0)); return 1; }

    printf("probe-hostfree: %d chunks of %zu MiB = %.2f GiB, %d cycle(s)\n",
           nchunks, chunk_mib, (double) nchunks * chunk_mib / 1024.0, cycles);
    show("baseline");
    const long free0 = meminfo_mib("MemFree:");
    long worst_return = 1L << 30;

    for (int c = 0; c < cycles; c++) {
        std::vector<void *> bufs;
        bufs.reserve((size_t) nchunks);
        for (int i = 0; i < nchunks; i++) {
            void * p = nullptr;
            const hipError_t e = hipHostMalloc(&p, chunk, hipHostMallocDefault);
            if (e != hipSuccess) {
                printf("  cycle %d: hipHostMalloc FAILED after %d chunks (%zu MiB): %s\n", c,
                       (int) bufs.size(), bufs.size() * chunk_mib, hipGetErrorString(e));
                break;
            }
            memset(p, (int) (i & 0xff), chunk);   // touch: the pin has to be materialised
            bufs.push_back(p);
        }
        const long held = meminfo_mib("MemFree:");
        printf("  --- cycle %d: held %zu / %d chunks (%.2f GiB)\n", c, bufs.size(), nchunks,
               (double) bufs.size() * chunk_mib / 1024.0);
        show("  while allocated");
        for (void * p : bufs) {
            const hipError_t e = hipHostFree(p);
            if (e != hipSuccess) { printf("  hipHostFree FAILED: %s\n", hipGetErrorString(e)); }
        }
        bufs.clear();
        bufs.shrink_to_fit();
        const long after = meminfo_mib("MemFree:");
        const long returned = after - held;
        if (after - free0 < worst_return) { worst_return = after - free0; }
        printf("  cycle %d: freed; MemFree returned %ld MiB (%.2f GiB held) -> %s; %ld MiB vs baseline\n",
               c, returned, (double) (held - free0) / 1024.0,
               returned > (long) (nchunks * chunk_mib / 4) ? "PAGES RETURNED" : "*** PAGES NOT RETURNED ***",
               after - free0);
        show("  after free");
    }

    printf("\n  worst MemFree vs baseline across cycles: %ld MiB  -> %s\n", worst_return,
           worst_return > -(long) (nchunks * chunk_mib / 8) ? "no accumulation" : "ACCUMULATING (leak)");

    // REUSE: can the kernel actually use those bytes for page cache again?
    const char * rf = getenv("REUSE_FILE");
    if (rf != nullptr) {
        const int fd = open(rf, O_RDONLY);
        if (fd < 0) { perror("open REUSE_FILE"); return 1; }
        struct stat st;
        if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
        const size_t want = (size_t) nchunks * chunk;                 // roughly what we just freed
        const size_t len  = (size_t) st.st_size < want ? (size_t) st.st_size : want;
        const long long b0 = read_bytes();
        const auto t0 = std::chrono::steady_clock::now();
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        (void) madvise(m, len, MADV_WILLNEED);
        volatile long long acc = 0;
        for (size_t o = 0; o < len; o += 4096) { acc += m[o]; }
        (void) acc;
        const double s = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - t0).count();
        printf("\n  REUSE: read %.2f GiB of %s through the page cache in %.1f s (%.2f GB/s), "
               "%.2f GiB from disk\n", (double) len / 1073741824.0, rf, s, (double) len / 1e9 / s,
               (double) (read_bytes() - b0) / 1073741824.0);
        show("  after the reuse read");
        munmap(m, len);
        close(fd);
    }
    return 0;
}
