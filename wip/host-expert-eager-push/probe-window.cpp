// probe-window -- CAN registered host pages be DROPPED once unregistered?  The question the whole campaign
// hinges on, so it is measured on the REAL model file, repeatedly, with the RAM accounting.
//
// WHY (wip/host-expert-eager-push P4): registering the host mapping in place makes a fill native
// (141 us -> 0.3 us/copy; the 2-device tensor split 71.6 t/s against the pool's 51.5 with 9.2 GiB less pinned
// RAM).  But the mode exists for the band where the weights do NOT fit in RAM: if a pin cannot be dropped,
// the page cache stops being reclaimable, the host swaps, and the mode is dead.  `hipHostUnregister` alone
// does NOT release the pin while the mapping is live (measured: MADV_PAGEOUT re-reads 2 of 256 MiB, control
// 256 of 256).
//
// So this establishes, on the model file itself:
//   1. FLAG MATRIX  -- is the leak specific to `hipHostRegisterReadOnly`?  (a cheap fix, if so)
//   2. RELEASE PATH -- which of unregister / unregister+munmap / munmap frees the pages
//   3. CYCLES       -- a sliding registered window, 20 cycles, each released and verified re-readable FROM
//                      DISK, with MemAvailable tracked so an accumulating leak would show
//   4. the released pages must still serve the MODEL'S OWN mapping (the bytes are shared page cache)
//
// Each arm runs in its own process (driver state is sticky), selected by argv[1].
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-window probe-window.cpp
//   run:
//     FILE=<gguf> /tmp/probe-window flags 1 64      # hipHostRegisterReadOnly
//     FILE=<gguf> /tmp/probe-window cycle 64 20     # the sliding-window proof
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static long long read_bytes() {
    FILE * f = fopen("/proc/self/io", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    long long v = -1;
    while (fgets(line, sizeof(line), f)) { if (!strncmp(line, "read_bytes:", 11)) { v = atoll(line + 11); break; } }
    fclose(f);
    return v;
}
static long mem_available_mib() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    long v = -1;
    while (fgets(line, sizeof(line), f)) { if (!strncmp(line, "MemAvailable:", 13)) { v = atol(line + 13) / 1024; break; } }
    fclose(f);
    return v;
}
static unsigned long long cksum(const char * p, size_t len) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t o = 0; o < len; o += 64) { h = (h ^ (unsigned char) p[o]) * 1099511628211ULL; }
    return h;
}
static void touch(const char * p, size_t len) {
    volatile long long acc = 0;
    for (size_t o = 0; o < len; o += 64) { acc += p[o]; }
    (void) acc;
}
static double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    const char * arm  = argc > 1 ? argv[1] : "cycle";
    const size_t mib  = argc > 2 ? (size_t) atoi(argv[2]) : 64;
    const int    reps = argc > 3 ? atoi(argv[3]) : 20;
    const char * file = getenv("FILE") ? getenv("FILE")
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const size_t len  = mib << 20;
    const int    fd   = open(file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    HIPCK(hipFree(nullptr));   // bring the runtime up before any measurement

    // A window well inside the file, at a 1 GiB-aligned offset: a separate VMA over the SAME page-cache pages
    // the model's own mapping uses (that is the whole point -- no duplicate).
    const size_t base_off = 1L << 30;

    if (!strcmp(arm, "flags")) {
        const unsigned flags = (unsigned) (argc > 3 ? atoi(argv[3]) : 4);
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) base_off);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);
        const hipError_t r = hipHostRegister(m, len, flags);
        printf("flags=0x%x (%s): register %s\n", flags,
               flags == 0 ? "Default" : flags == 1 ? "Portable" : flags == 2 ? "Mapped" :
               flags == 4 ? "ReadOnly" : flags == 5 ? "ReadOnly|Portable" : "custom",
               r == hipSuccess ? "OK" : hipGetErrorString(r));
        if (r != hipSuccess) { return 0; }
        HIPCK(hipHostUnregister(m));
        (void) posix_fadvise(fd, (off_t) base_off, (off_t) len, POSIX_FADV_DONTNEED);
        (void) madvise(m, len, MADV_PAGEOUT);
        const long long b0 = read_bytes();
        touch(m, len);
        printf("  after unregister: MADV_PAGEOUT re-read %.1f / %zu MiB  -> %s\n",
               (double) (read_bytes() - b0) / 1048576.0, len >> 20,
               (read_bytes() - b0) > (long long) (len / 2) ? "RELEASED" : "STILL PINNED");
        munmap(m, len);
        return 0;
    }

    if (!strcmp(arm, "release")) {
        // which release path frees the pages?  (fresh process, one path per run: argument 3 = munmap?)
        const bool unmap = argc > 3 && atoi(argv[3]) != 0;
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) base_off);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);
        HIPCK(hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable));
        HIPCK(hipHostUnregister(m));
        if (unmap) { munmap(m, len); }
        (void) posix_fadvise(fd, (off_t) base_off, (off_t) len, POSIX_FADV_DONTNEED);
        long long b0, b1;
        if (unmap) {
            char * m2 = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) base_off);
            b0 = read_bytes(); touch(m2, len); b1 = read_bytes();
            munmap(m2, len);
        } else {
            (void) madvise(m, len, MADV_PAGEOUT);
            b0 = read_bytes(); touch(m, len); b1 = read_bytes();
        }
        printf("release=%s: re-read %.1f / %zu MiB -> %s\n", unmap ? "unregister+munmap" : "unregister only",
               (double) (b1 - b0) / 1048576.0, len >> 20, (b1 - b0) > (long long) (len / 2) ? "RELEASED" : "STILL PINNED");
        return 0;
    }

    if (!strcmp(arm, "shared")) {
        // THE end-to-end shape a delivered feature would have: llama.cpp holds its own long-lived read-only
        // mapping of the model (A); the transport owns a SECOND mapping of the same file range (B) and
        // registers THAT.  Releasing B must leave A's pages reclaimable -- otherwise the mode starves RAM.
        char * a = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) base_off);
        if (a == MAP_FAILED) { perror("mmap A"); return 1; }
        touch(a, len);
        const unsigned long long h_before = cksum(a, len);
        char * b = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) base_off);
        if (b == MAP_FAILED) { perror("mmap B"); return 1; }
        HIPCK(hipHostRegister(b, len, hipHostRegisterReadOnly | hipHostRegisterPortable));
        void * d = nullptr;
        HIPCK(hipSetDevice(0));
        HIPCK(hipMalloc(&d, len));
        hipStream_t s2;
        HIPCK(hipStreamCreate(&s2));
        HIPCK(hipMemcpyAsync(d, b, len, hipMemcpyHostToDevice, s2));
        HIPCK(hipStreamSynchronize(s2));
        HIPCK(hipHostUnregister(b));
        munmap(b, len);                       // release the transport's window, A stays
        // now the SHARED pages must be reclaimable through A, the mapping llama.cpp keeps
        const long long b0 = read_bytes();
        (void) madvise(a, len, MADV_PAGEOUT);
        touch(a, len);
        const long long b1 = read_bytes();
        const unsigned long long h_after = cksum(a, len);
        printf("shared: after the window was released, A's pages re-read %.1f / %zu MiB -> %s; data %s\n",
               (double) (b1 - b0) / 1048576.0, len >> 20,
               (b1 - b0) > (long long) (len / 2) ? "RECLAIMABLE" : "NOT reclaimable",
               h_after == h_before ? "identical" : "CORRUPTED");
        munmap(a, len);
        return (b1 - b0) > (long long) (len / 2) ? 0 : 4;
    }

    // ---- cycles: a SLIDING registered window, the shape a releasable tier would use
    void * d = nullptr;
    HIPCK(hipSetDevice(0));
    HIPCK(hipMalloc(&d, len));
    hipStream_t s;
    HIPCK(hipStreamCreate(&s));
    const bool reg = !strcmp(arm, "cycle");
    printf("%s: %d cycles of a %zu MiB sliding window over %s (registration %s)\n",
           arm, reps, mib, file, reg ? "ON" : "off (control)");
    printf("  start: MemAvailable %ld MiB\n", mem_available_mib());
    long long reclaimed_mib = 0;
    double reg_ms = 0.0, unreg_ms = 0.0, copy_ms = 0.0;
    int bad_reclaim = 0;
    for (int i = 0; i < reps; i++) {
        const size_t off = base_off + (size_t) i * len * 2;   // never overlap, always a fresh range
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);                        // page cache: this reads from disk the first time
        if (reg) {
            auto t0 = std::chrono::steady_clock::now();
            const hipError_t e = hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable);
            reg_ms += ms_since(t0);
            if (e != hipSuccess) { printf("  cycle %d: register FAILED %s\n", i, hipGetErrorString(e)); break; }
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < 8; k++) {   // use it as a DMA source, like a fill
                HIPCK(hipMemcpyAsync((char *) d + (size_t) k * 4096, m, len - 8 * 4096, hipMemcpyHostToDevice, s));
            }
            HIPCK(hipStreamSynchronize(s));
            copy_ms += ms_since(t1);
            auto t2 = std::chrono::steady_clock::now();
            HIPCK(hipHostUnregister(m));
            unreg_ms += ms_since(t2);
        }
        munmap(m, len);                       // release the VMA (the pin lives exactly as long as it does)
        (void) posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
        // verify: the pages must be droppable -> mapping them again has to read them back from disk
        const long long b0 = read_bytes();
        char * m2 = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
        const unsigned long long h = cksum(m2, len);
        const long long b1 = read_bytes();
        munmap(m2, len);
        const long long mb = (b1 - b0) >> 20;
        reclaimed_mib += mb;
        if (mb < (long long) (mib / 2)) { bad_reclaim++; }
        if (i < 3 || i >= reps - 3 || mb < (long long) (mib / 2)) {
            printf("  cycle %2d: re-read %4lld / %zu MiB %s  MemAvailable %ld MiB  cksum %016llx\n", i, mb, mib,
                   mb >= (long long) (mib / 2) ? "(dropped and re-read)" : "*** NOT DROPPED ***",
                   mem_available_mib(), h);
        }
    }
    printf("  %d/%d cycles reclaimed their pages (%lld MiB re-read in total)\n", reps - bad_reclaim, reps, reclaimed_mib);
    if (reg) {
        printf("  cost per %zu MiB window: register %.1f ms, 8x H2D %.2f ms, unregister %.1f ms  ->  "
               "%.2f ms/cycle for %.1f GiB/s of registration\n", mib, reg_ms / reps, copy_ms / reps, unreg_ms / reps,
               (reg_ms + copy_ms + unreg_ms) / reps,
               (double) mib / 1024.0 / ((reg_ms / reps) / 1000.0));
    }
    printf("  end: MemAvailable %ld MiB\n", mem_available_mib());
    return bad_reclaim == 0 ? 0 : 3;
}
