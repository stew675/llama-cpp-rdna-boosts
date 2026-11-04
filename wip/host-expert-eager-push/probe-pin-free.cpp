// probe-pin-free -- is there a call that RELEASES a `hipHostRegister` pin while the mapping survives?
//
// WHY (wip/host-expert-eager-push): `hipHostRegister` over the model mapping makes it a native DMA source
// (fills 141 us -> 0.3-0.5 us/copy; the 2-device tensor split 26 -> 71.6 t/s, byte-identical) but on this
// stack `hipHostUnregister` does NOT give the pages back: with the mapping still live the kernel cannot
// reclaim them (probe-pin arm C/F: 2 MiB of 256 MiB re-read after MADV_PAGEOUT, i.e. still pinned), while
// unregister + munmap DOES release them (arm E: 256/256 re-read).  So the pin lives as long as the VMA.
//
// This asks the follow-up: does `hipFreeHost` -- the other "give the host memory back" call -- release a
// REGISTERED range, and does it take the caller's mapping with it?  (It is documented for `hipHostMalloc`
// pointers only, so the interesting part is what it does anyway, and whether the VMA survives.)
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-pin-free probe-pin-free.cpp
//   run:  FILE=/var/tmp/pin-test.bin /tmp/probe-pin-free [MiB]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
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
static void touch(const char * p, size_t len) {
    volatile long long acc = 0;
    for (size_t o = 0; o < len; o += 64) { acc += p[o]; }
    (void) acc;
}
// is the VMA still there?  mincore fails with ENOMEM for an unmapped range and never faults.
static bool vma_alive(void * p, size_t len) {
    std::vector<unsigned char> v(len / 4096 + 1);
    return mincore(p, len, v.data()) == 0;
}
static void arm(const char * name, int fd, size_t off, size_t len, int mode) {
    char * m = (char *) mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, (off_t) off);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    touch(m, len);
    const hipError_t r = hipHostRegister(m, len, hipHostRegisterReadOnly | hipHostRegisterPortable);
    if (r != hipSuccess) { printf("  %-46s register FAILED: %s\n", name, hipGetErrorString(r)); munmap(m, len); return; }
    hipError_t rel = hipSuccess;
    const char * relname = "none";
    switch (mode) {
        case 1: relname = "hipHostUnregister"; rel = hipHostUnregister(m); break;
        case 2: relname = "hipFreeHost";       rel = hipFreeHost(m);       break;
        case 3: relname = "unregister + munmap"; rel = hipHostUnregister(m); munmap(m, len); break;
        default: break;
    }
    const bool alive = mode == 3 ? false : vma_alive(m, len);
    (void) posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_DONTNEED);
    long long b0 = 0, b1 = 0;
    if (alive) {
        (void) madvise(m, len, MADV_PAGEOUT);
        b0 = read_bytes();
        touch(m, len);
        b1 = read_bytes();
    }
    printf("  %-46s %-20s %-9s VMA %-3s  re-read %7.1f / %zu MiB  %s\n", name, relname,
           rel == hipSuccess ? "OK" : hipGetErrorString(rel), alive ? "up" : "gone",
           (double) (b1 - b0) / 1048576.0, len >> 20,
           !alive ? "n/a (mapping gone -> pages released)"
                  : (b1 - b0) > (long long) (len / 2) ? "EVICTED (pin released)"
                                                      : "NOT evicted (still pinned)");
    if (alive) { munmap(m, len); }
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) atoi(argv[1]) : 256;
    const char * file = getenv("FILE") ? getenv("FILE") : "/var/tmp/pin-test.bin";
    const size_t len = mib << 20;
    const int fd = open(file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    HIPCK(hipFree(nullptr));
    printf("probe-pin-free: %zu MiB windows of %s\n\n", mib, file);
    arm("A. register, release with hipHostUnregister", fd, len * 0, len, 1);
    arm("B. register, release with hipFreeHost",       fd, len * 1, len, 2);
    arm("C. register, release with unregister+munmap", fd, len * 2, len, 3);
    arm("D. register, release with nothing (control)", fd, len * 3, len, 0);

    // The control for the whole idea: hipHostMalloc -> hipHostFree really does return anonymous pages.
    printf("\n  control: hipHostMalloc/hipHostFree (anonymous host memory, the documented pair)\n");
    void * hp = nullptr;
    HIPCK(hipHostMalloc(&hp, len, hipHostMallocDefault));
    memset(hp, 1, len);
    const bool alive = vma_alive(hp, len);
    HIPCK(hipHostFree(hp));
    printf("    VMA %s after hipHostFree, munmap-after-free still possible: %s\n",
           alive ? "up before free" : "?", vma_alive(hp, len) ? "yes" : "no");
    return 0;
}
