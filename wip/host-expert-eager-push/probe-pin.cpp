// probe-pin -- after `hipHostUnregister`, can the kernel still EVICT the pages?  (the question that
// decides whether registering the model mapping in place is a residency TIER or a one-way RAM grab)
//
// WHY (wip/host-expert-eager-push): `MOE_HOST_PUSH_REGISTER_MIB` registers the read-only model mapping in
// place, which makes it a native DMA source (a 360 KiB fill 91-146 us pageable -> 17-23 us registered; the
// 2-device tensor split 26 -> 71 t/s, and it needs NO duplicate of the bytes).  But a residency tier that
// cannot be undone is worse than the pool in the only band the mode is for (the set does NOT fit).  The
// maintainer's finding on the other system: after `hipHostUnregister` the pages stayed pinned.
//
// VmLck does NOT see HIP's pins on this stack and `mincore` reports page-cache residency for a file mapping,
// so neither can answer it.  This probe uses the one signal that is honest: /proc/self/io read_bytes.
//
//   touch the range            -> the pages are in the page cache
//   [register / unregister]    -> whatever the runtime did to them
//   POSIX_FADV_DONTNEED        -> the kernel is asked to drop those pages
//   touch the range again      -> read_bytes grows by ~the range IF they were really dropped
//
// A pinned (GUP-refcounted) page cannot be dropped, so a zero delta after unregister means the pin LEAKED.
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-pin probe-pin.cpp
//   run:  /tmp/probe-pin [MiB]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static long long read_bytes() {
    FILE * f = fopen("/proc/self/io", "r");
    if (f == nullptr) { return -1; }
    char line[256];
    long long v = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, "read_bytes:", 11) == 0) { v = atoll(line + 11); break; }
    }
    fclose(f);
    return v;
}
static void touch(const char * p, size_t len) {
    volatile long long acc = 0;
    for (size_t o = 0; o < len; o += 64) { acc += p[o]; }   // one load per cache line
    (void) acc;
}

// Time one H2D copy of the whole window: the ONLY observable that distinguishes "the runtime still knows
// this range is registered" from "it does not" (a registered source measured 17-23 us per 360 KiB fill
// against 91-146 us pageable).
static double copy_us(int dev, void * dst, const char * src, size_t len, int reps) {
    hipStream_t s2;
    HIPCK(hipSetDevice(dev));
    HIPCK(hipStreamCreate(&s2));
    HIPCK(hipMemcpyAsync(dst, src, len, hipMemcpyHostToDevice, s2));
    HIPCK(hipStreamSynchronize(s2));
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    for (int i = 0; i < reps; i++) {
        HIPCK(hipMemcpyAsync(dst, src + (size_t) (i % 8) * 4096, len - 8 * 4096, hipMemcpyHostToDevice, s2));
    }
    HIPCK(hipStreamSynchronize(s2));
    const double us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(clk::now() - t0).count();
    HIPCK(hipStreamDestroy(s2));
    return us / reps;
}

// one arm, mapping stays LIVE (a pin cannot be observed on a range we unmap -- the driver loses the pages):
//   touch -> [register] -> [unregister] -> MADV_PAGEOUT -> re-touch, count read_bytes
// MADV_PAGEOUT asks the kernel to reclaim the range NOW; it silently skips a page it cannot isolate (i.e. a
// GUP-pinned one), so a zero re-read after unregister is the leak the maintainer found.
static void arm(const char * name, int fd, size_t off, size_t len, bool reg, bool unreg) {
    char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    touch(m, len);
    if (reg) {
        const hipError_t e = hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable);
        if (e != hipSuccess) {
            printf("  %-42s hipHostRegister FAILED: %s\n", name, hipGetErrorString(e));
            munmap(m, len);
            return;
        }
    }
    if (unreg) { HIPCK(hipHostUnregister(m)); }
    (void) posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    const int rc = madvise(m, len, MADV_PAGEOUT);
    const long long b0 = read_bytes();
    touch(m, len);
    const long long b1 = read_bytes();
    const double mib = (double) (b1 - b0) / 1048576.0;
    printf("  %-42s re-read %7.1f MiB of %zu MiB %s   (madvise rc=%d)\n", name, mib, len >> 20,
           mib > 0.5 * (double) (len >> 20) ? "EVICTED (no pin survived)" : "NOT evicted (STILL PINNED)", rc);
    if (reg && !unreg) { (void) hipHostUnregister(m); }
    munmap(m, len);
}

int main(int argc, char ** argv) {
    const size_t mib  = argc > 1 ? (size_t) atoi(argv[1]) : 256;
    const char * file = getenv("FILE") != nullptr ? getenv("FILE")
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const size_t len  = mib << 20;
    const int    fd   = open(file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    HIPCK(hipFree(nullptr));

    printf("probe-pin: %zu MiB windows of %s, evicted with POSIX_FADV_DONTNEED and re-read\n\n", mib, file);
    const size_t win = len;        // one window per arm, back to back inside the temp file
    arm("A. no registration (control)",              fd, win * 0, len, false, false);
    arm("B. registered, NOT unregistered",           fd, win * 1, len, true,  false);
    arm("C. registered, then hipHostUnregister'd",   fd, win * 2, len, true,  true);
    arm("D. control again (repeatability)",          fd, win * 3, len, false, false);

    // is the pin tied to the MAPPING (a rotating window could then be released by munmap) or permanent?
    {
        printf("\n  pin lifetime vs the mapping:\n");
        const size_t off = win * 5;
        // E: register -> unregister -> munmap -> fadvise -> NEW mmap -> touch
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);
        HIPCK(hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable));
        HIPCK(hipHostUnregister(m));
        munmap(m, len);
        (void) posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
        long long b0 = read_bytes();
        char * m2 = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
        touch(m2, len);
        printf("    E. register -> UNREGISTER -> munmap -> fadvise:      re-read %7.1f / %zu MiB  %s\n",
               (double) (read_bytes() - b0) / 1048576.0, len >> 20,
               (read_bytes() - b0) > (long long) (len / 2) ? "EVICTED (pin released with the mapping)"
                                                           : "NOT evicted (pin OUTLIVED the mapping)");
        munmap(m2, len);
        // F: unregister, then sleep, then fadvise on the LIVE mapping (is the release deferred?)
        m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) (win * 6));
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);
        HIPCK(hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable));
        HIPCK(hipHostUnregister(m));
        usleep(500 * 1000);
        (void) posix_fadvise(fd, (off_t) (win * 6), (off_t) len, POSIX_FADV_DONTNEED);
        (void) madvise(m, len, MADV_PAGEOUT);
        b0 = read_bytes();
        touch(m, len);
        printf("    F. unregister + 0.5 s, mapping LIVE, PAGEOUT:        re-read %7.1f / %zu MiB  %s\n",
               (double) (read_bytes() - b0) / 1048576.0, len >> 20,
               (read_bytes() - b0) > (long long) (len / 2) ? "EVICTED (deferred release)" : "NOT evicted (permanent)");
        munmap(m, len);
    }

    // behaviour: does the copy path still treat the range as registered after unregistering?
    {
        char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) win * 4);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch(m, len);
        void * d = nullptr;
        HIPCK(hipSetDevice(0));
        HIPCK(hipMalloc(&d, len));
        const double before = copy_us(0, d, m, len, 8);
        HIPCK(hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable));
        const double during = copy_us(0, d, m, len, 8);
        HIPCK(hipHostUnregister(m));
        const double after  = copy_us(0, d, m, len, 8);
        printf("\n  copy of %zu MiB: pageable %.0f us -> registered %.0f us -> unregistered %.0f us  (%s)\n",
               len >> 20, before, during, after,
               after > 0.5 * (before + during) ? "the runtime FORGOT the registration" : "STILL treated as registered");
        munmap(m, len);
    }

    // Can the runtime register the SAME range twice / after an unregister?  (a rotating window would)
    printf("\n  rotation: register -> unregister -> register the same range again\n");
    char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) win * 4);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    touch(m, len);
    for (int i = 0; i < 3; i++) {
        hipError_t e = hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable);
        printf("    cycle %d register: %s\n", i, e == hipSuccess ? "OK" : hipGetErrorString(e));
        if (e != hipSuccess) { break; }
        e = hipHostUnregister(m);
        printf("    cycle %d unregister: %s\n", i, e == hipSuccess ? "OK" : hipGetErrorString(e));
    }
    munmap(m, len);
    return 0;
}
