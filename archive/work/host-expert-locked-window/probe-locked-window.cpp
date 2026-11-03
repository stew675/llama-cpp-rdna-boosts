// wip/host-expert-locked-window P0 probe: is "the mapping is the residency tier" mechanically sound, and
// what does the hold cost?
//
// The design (DESIGN.md) says a host expert's bytes can be read IN PLACE by a kernel if the host (a) has
// installed a device mapping over the range once per tensor (`hipHostRegister`) and (b) has HELD the pages
// the op is about to read (`mlock2(MLOCK_ONFAULT)` + `MADV_POPULATE_READ`).  The prior campaign
// (archive/work/host-expert-pinned-window/) established that (a) alone is not enough: an unheld page faults
// fatally on a no-XNACK GPU.  This probe measures the missing half and prices it.
//
// One ARM per process (`--arm <name>`), because the negative control is expected to die: the runner script
// sequences them and records each status.  An arm that reads an unheld pageable range is FATAL by design, so
// nothing may share a process with it.
//
//   base          register cost vs size, alias identity, in-place bandwidth (populated, hot)
//   hold-hot      mlock2(MLOCK_ONFAULT) + MADV_POPULATE_READ on a page-cache-hot expert, then GPU verify
//   hold-cold     MADV_PAGEOUT first (proven with mincore), then the same -> this is the disk read
//   survive       held + populated, then PAGEOUT + 4 GiB anonymous pressure -> GPU read must be identical
//   granularity   expert-granular vs chunk-granular locks: VMA count, per-op cost, the refusal path
//   unheld        registered + populated + NOT held, then PAGEOUT -> GPU read must FAULT (the gate's proof)
//
// Every GPU read is verified by copying the range to VRAM, reading it back and memcmp'ing against the file
// bytes, so a wrong result is never reported as a fast one.
//
// build:  ./build.sh
// run:    ./run-probe.sh [GGUF_PATH] [EXPERT_MIB]
//
// Traps this probe is written around (all measured by the prior campaign, do not "simplify" them away):
//   * MADV_DONTNEED and POSIX_FADV_DONTNEED are NOT reclaim tests on a mapped file (DONTNEED drops only the
//     PTEs, FADV_DONTNEED skips pages that have PTEs).  MADV_PAGEOUT is the only user call that asks the
//     kernel to reclaim, and every cold claim here is proven with mincore.
//   * /tmp is tmpfs on this box, so a "cold" test written there measures RAM.  The source is the real GGUF.
//   * `Mlocked` is not a pinning probe (`hipHostMalloc` pins without mlocking; `hipHostRegister` maps
//     without pinning).  mincore + a real fault are the evidence, not /proc accounting.
//   * a pageable mapping is not device-addressable at all: reading it from a kernel is the refuted arm, so
//     the `base` arm populates before it reads and says so.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MLOCK_ONFAULT
#define MLOCK_ONFAULT 1
#endif
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

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

template <typename T> static inline T tmin(T a, T b) { return a < b ? a : b; }
template <typename T> static inline T tmax(T a, T b) { return a > b ? a : b; }

static double med(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
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

// ---------------------------------------------------------------------------------------------
// the in-place reader
// ---------------------------------------------------------------------------------------------

// 16-byte vector loads: a byte-at-a-time stride pattern measures the kernel, not the link.
__global__ void kern_copy4(const uint4 * __restrict__ src, uint4 * __restrict__ dst, size_t n4) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void kern_copy(const unsigned char * __restrict__ src, unsigned char * __restrict__ dst,
                          size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        dst[i] = src[i];
    }
}

static void kern_launch(const unsigned char * src, unsigned char * dst, size_t n) {
    const int threads = 256;
    if ((n % 16) == 0 && ((uintptr_t) src % 16) == 0) {
        const size_t n4 = n / 16;
        const int blocks = (int) tmin<size_t>(4096, tmax<size_t>(1, (n4 + threads - 1) / threads));
        kern_copy4<<<blocks, threads>>>((const uint4 *) src, (uint4 *) dst, n4);
        return;
    }
    const int blocks = (int) tmin<size_t>(4096, tmax<size_t>(1, (n + threads - 1) / threads));
    kern_copy<<<blocks, threads>>>(src, dst, n);
}

// One throwaway launch so the first timed read does not pay the module load.
static void warmup() {
    // Pinned (hipHostMalloc): a pageable host pointer is not device-addressable at all, and a warmup that
    // faults asserts exactly the thing this campaign exists to avoid.
    unsigned char * h = nullptr;
    unsigned char * d = nullptr;
    HIP_CHECK(hipHostMalloc((void **) &h, 4096, hipHostMallocDefault));
    memset(h, 0, 4096);
    HIP_CHECK(hipMalloc(&d, 4096));
    kern_copy<<<1, 256>>>(h, d, 4096);
    HIP_CHECK(hipDeviceSynchronize());
    (void) hipFree(d);
    (void) hipHostFree(h);
}

// Copy `n` bytes of the (possibly registered) host range `src_host` through the GPU and verify.
static bool gpu_read_verify(const unsigned char * src_host, size_t n, double * us_out) {
    unsigned char * d = nullptr;
    HIP_CHECK(hipMalloc(&d, n));
    unsigned char * back = (unsigned char *) malloc(n);

    const double t0 = now_us();
    kern_launch(src_host, d, n);
    HIP_CHECK(hipDeviceSynchronize());
    const double t1 = now_us();

    HIP_CHECK(hipMemcpy(back, d, n, hipMemcpyDeviceToHost));
    const bool ok = memcmp(back, src_host, n) == 0;

    if (us_out) *us_out = t1 - t0;
    free(back);
    (void) hipFree(d);
    return ok;
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

static size_t resident_pages(void * addr, size_t n) {
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    const size_t np = (n + pg - 1) / pg;
    std::vector<unsigned char> vec(np ? np : 1, 0);
    if (mincore(addr, n, vec.data()) != 0) {
        fprintf(stderr, "mincore: %s\n", strerror(errno));
        return 0;
    }
    size_t r = 0;
    for (size_t i = 0; i < np; i++) {
        if (vec[i] & 1) r++;
    }
    return r;
}

// Residency of a file range as the PAGE CACHE sees it: map it fresh and mincore.  mincore on the
// registered mapping reports the process' PTEs, which the driver may have dropped while the pages
// themselves stay put -- that distinction is the whole question `arm_pin` exists to answer.
static size_t fresh_resident(int fd, size_t off, size_t n) {
    void * p = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, (off_t) off);
    if (p == MAP_FAILED) return (size_t) -1;
    const size_t r = resident_pages(p, n);
    munmap(p, n);
    return r;
}

static long meminfo_kb(const char * key) {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, strlen(key)) == 0) {
            v = atol(line + strlen(key) + 1);
            break;
        }
    }
    fclose(f);
    return v;
}

static long vmlck_kb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmLck:", 6) == 0) { v = atol(line + 6); break; }
    }
    fclose(f);
    return v;
}

static bool page_out(void * addr, size_t n) {
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    void * a = (void *) (((uintptr_t) addr) & ~(uintptr_t)(pg - 1));
    const size_t len = ((uintptr_t) addr - (uintptr_t) a) + n;
    if (madvise(a, len, MADV_PAGEOUT) != 0) {
        fprintf(stderr, "MADV_PAGEOUT: %s\n", strerror(errno));
        return false;
    }
    return true;
}

static int mlock_onfault(void * addr, size_t n) {
    return (int) syscall(SYS_mlock2, addr, n, (unsigned int) MLOCK_ONFAULT);
}

static int vma_count() {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    int n = 0;
    char line[4096];
    while (fgets(line, sizeof line, f)) n++;
    fclose(f);
    return n;
}

static size_t memlock_budget() {
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0) return 8ull << 20;
    return (rl.rlim_cur == RLIM_INFINITY) ? (8ull << 30) : (size_t) rl.rlim_cur;
}

static void report_limits() {
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        printf("RLIMIT_MEMLOCK      soft=%llu KiB  hard=%llu KiB%s\n",
               (unsigned long long) (rl.rlim_max == RLIM_INFINITY ? 0 : rl.rlim_cur / 1024),
               (unsigned long long) (rl.rlim_max == RLIM_INFINITY ? 0 : rl.rlim_max / 1024),
               rl.rlim_cur == RLIM_INFINITY ? " (unlimited)" : "");
        if (rl.rlim_cur == rl.rlim_max && rl.rlim_cur != RLIM_INFINITY) {
            printf("                    (soft == hard: cannot be raised without CAP_IPC_LOCK/privilege)\n");
        }
    }
    printf("max_map_count       ");
    fflush(stdout);
    FILE * f = fopen("/proc/sys/vm/max_map_count", "r");
    if (f) { char b[64] = {0}; if (fgets(b, sizeof b, f)) printf("%s", b); fclose(f); } else printf("?\n");
    printf("uid                 %d (%s)\n", (int) geteuid(), geteuid() == 0 ? "root: can raise the limit" : "cannot raise RLIMIT_MEMLOCK");
}

struct region {
    int             fd     = -1;
    unsigned char * base   = nullptr;
    size_t          off    = 0;
    size_t          bytes  = 0;
    size_t          expert = 0;   // page-aligned expert blob
    bool            registered = false;
};

static region map_and_register(const char * path, double expert_mib) {
    region r;
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    r.expert = (size_t) ((expert_mib * 1048576.0) + (double) pg - 1) / pg * pg;

    r.fd = open(path, O_RDONLY);
    if (r.fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); exit(1); }
    struct stat st;
    fstat(r.fd, &st);
    const size_t fsize = (size_t) st.st_size;
    r.bytes = tmin<size_t>(512ull << 20, fsize / 2);
    r.off   = (fsize / 3) & ~(pg - 1);
    r.base  = (unsigned char *) mmap(nullptr, r.bytes, PROT_READ, MAP_SHARED, r.fd, (off_t) r.off);
    if (r.base == MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); exit(1); }
    return r;
}

// LW_REG_FLAGS: the last door on a releasable hold.  `hipHostRegisterReadOnly` is documented as the
// read-only registration; the prior campaign found it claimed unsupported yet accepted.  If a release
// exists at all it is most likely here.
static unsigned int reg_flags() {
    static unsigned int f = [] {
        const char * e = getenv("LW_REG_FLAGS");
        if (e == nullptr) return (unsigned int) hipHostRegisterDefault;
        if (strcmp(e, "ro") == 0) return (unsigned int) hipHostRegisterReadOnly;
        if (strcmp(e, "portable") == 0) return (unsigned int) hipHostRegisterPortable;
        if (strcmp(e, "mapped") == 0) return (unsigned int) hipHostRegisterMapped;
        return (unsigned int) hipHostRegisterDefault;
    }();
    return f;
}

static void do_register(region & r, double * us_out) {
    const double t0 = now_us();
    HIP_CHECK(hipHostRegister(r.base, r.bytes, reg_flags()));
    const double t1 = now_us();
    r.registered = true;
    if (us_out) *us_out = t1 - t0;
}

static void release_region(region & r) {
    if (r.registered) (void) hipHostUnregister(r.base);
    if (r.base) munmap(r.base, r.bytes);
    if (r.fd >= 0) close(r.fd);
}

// ---------------------------------------------------------------------------------------------
// arms
// ---------------------------------------------------------------------------------------------

static int arm_base(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    printf("source        %s\n", path);
    printf("expert blob   %s (%zu pages)\n", mib((double) r.expert).c_str(), np);
    printf("region        off=%zu len=%s\n", r.off, mib((double) r.bytes).c_str());

    const size_t region_pages = r.bytes / (size_t) sysconf(_SC_PAGESIZE);
    printf("region resident before register  %zu/%zu pages\n", resident_pages(r.base, r.bytes), region_pages);

    double t_reg = 0;
    do_register(r, &t_reg);
    void * alias = nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&alias, r.base, 0));
    printf("register      %8.1f us  (%.4f us/MiB)\n", t_reg, t_reg / (r.bytes / 1048576.0));
    printf("host ptr      %p\ndevice alias  %p -> %s\n", (void *) r.base, alias,
           (const void *) alias == (const void *) r.base ? "IDENTITY (UVA)" : "distinct");
    printf("region resident after register   %zu/%zu pages  (a mapping, or a hold?)\n",
           resident_pages(r.base, r.bytes), region_pages);

    // Second registration of the same range: separates one-time driver init from a per-byte cost.
    (void) hipHostUnregister(r.base);
    r.registered = false;
    double t_reg2 = 0;
    do_register(r, &t_reg2);
    printf("register again (warm)  %8.1f us  (%.4f us/MiB)\n", t_reg2, t_reg2 / (r.bytes / 1048576.0));
    warmup();

    // Populated (hot) BEFORE the read: a registered-but-not-resident page is the fatal arm, and this arm is
    // about bandwidth, not about the hazard (the `unheld` arm is the hazard).
    const int prc = madvise(r.base, r.expert, MADV_POPULATE_READ);
    double us = 0;
    const bool ok = prc == 0 && gpu_read_verify(r.base, r.expert, &us);
    printf("in-place read %8.1f us  %.1f GB/s  %s\n", us, r.expert / (us * 1e-6) / 1e9,
           ok ? "DATA IDENTICAL" : "*** MISMATCH ***");

    // sustained: back-to-back kernels, one allocation, no per-iteration readback (that measures the harness).
    const size_t n = tmin<size_t>(64, r.bytes / r.expert);
    for (size_t i = 0; i < n; i++) (void) madvise(r.base + i * r.expert, r.expert, MADV_POPULATE_READ);
    unsigned char * dv = nullptr;
    HIP_CHECK(hipMalloc(&dv, r.expert));
    const double t0 = now_us();
    for (size_t i = 0; i < n; i++) {
        kern_launch(r.base + i * r.expert, dv, r.expert);
    }
    HIP_CHECK(hipDeviceSynchronize());
    const double t1 = now_us();
    // verify the last one (the readback is outside the timed section)
    unsigned char * back = (unsigned char *) malloc(r.expert);
    HIP_CHECK(hipMemcpy(back, dv, r.expert, hipMemcpyDeviceToHost));
    const bool last_ok = memcmp(back, r.base + (n - 1) * r.expert, r.expert) == 0;
    free(back);
    (void) hipFree(dv);
    printf("sustained     %zu x %s read+write in %.1f us -> %.1f GB/s useful (%.1f GB/s link)  last %s\n", n,
           mib((double) r.expert).c_str(), t1 - t0, (double) (n * r.expert) / ((t1 - t0) * 1e-6) / 1e9,
           2.0 * (double) (n * r.expert) / ((t1 - t0) * 1e-6) / 1e9,
           last_ok ? "DATA IDENTICAL" : "*** MISMATCH ***");
    release_region(r);
    return 0;
}

static int arm_hold_hot(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    const size_t budget = memlock_budget();
    const size_t n_hold = tmax<size_t>(1, tmin<size_t>(budget / r.expert, r.bytes / r.expert));
    printf("budget        %s locked -> %zu expert blobs holdable\n", mib((double) budget).c_str(), n_hold);

    std::vector<double> lock_us, pop_us, unl_us;
    size_t ok_reads = 0;
    for (size_t i = 0; i < n_hold; i++) {
        unsigned char * p = r.base + i * r.expert;
        const double a = now_us();
        const int rc = mlock_onfault(p, r.expert);
        const double b = now_us();
        if (rc != 0) { printf("mlock2 failed: %s\n", strerror(errno)); break; }
        const double c = now_us();
        const int prc = madvise(p, r.expert, MADV_POPULATE_READ);
        const double d = now_us();
        if (prc != 0) printf("populate failed: %s\n", strerror(errno));
        if (gpu_read_verify(p, r.expert, nullptr)) ok_reads++;
        const double e = now_us();
        munlock(p, r.expert);
        const double f = now_us();
        lock_us.push_back(b - a);
        pop_us.push_back(d - c);
        unl_us.push_back(f - e);
    }
    printf("mlock2(ONFAULT)  median %8.2f us  (%d ops)\n", med(lock_us), (int) lock_us.size());
    printf("populate (hot)   median %8.2f us  (%.2f us/MiB)\n", med(pop_us), med(pop_us) / (r.expert / 1048576.0));
    printf("munlock          median %8.2f us\n", med(unl_us));
    printf("in-place reads   %zu/%zu DATA IDENTICAL\n", ok_reads, lock_us.size());
    release_region(r);
    return 0;
}

static int arm_hold_cold(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    unsigned char * p = r.base + (r.bytes / 2 & ~((size_t) sysconf(_SC_PAGESIZE) - 1)) - r.expert;

    (void) madvise(p, r.expert, MADV_POPULATE_READ);   // make sure it is in RAM first...
    const size_t before = fresh_resident(r.fd, (size_t) (p - r.base) + r.off, r.expert);
    if (!page_out(p, r.expert)) { release_region(r); return 1; }
    const size_t after = fresh_resident(r.fd, (size_t) (p - r.base) + r.off, r.expert);
    printf("PAGEOUT proven by mincore   %zu -> %zu of %zu pages resident\n", before, after, np);
    if (after != 0) {
        printf("*** the page cache did not drop (fresh mapping still %zu/%zu): VOID ***\n", after, np);
        release_region(r);
        return 1;
    }

    const double a = now_us();
    const int rc = mlock_onfault(p, r.expert);
    const double b = now_us();
    const double c = now_us();
    const int prc = madvise(p, r.expert, MADV_POPULATE_READ);
    const double d = now_us();
    printf("mlock2(ONFAULT)  %8.2f us  %s\n", b - a, rc == 0 ? "ok" : strerror(errno));
    printf("populate (cold)  %8.2f us  %.2f GB/s  %s\n", d - c,
           (d - c) > 0 ? (r.expert / ((d - c) * 1e-6)) / 1e9 : 0.0, prc == 0 ? "ok" : strerror(errno));
    double us = 0;
    const bool ok = (rc == 0 && prc == 0) && gpu_read_verify(p, r.expert, &us);
    printf("in-place read    %8.1f us  %s\n", us, ok ? "DATA IDENTICAL" : "*** MISMATCH ***");
    printf("page cache after %zu/%zu (fresh mapping)\n",
           fresh_resident(r.fd, (size_t) (p - r.base) + r.off, r.expert), np);
    munlock(p, r.expert);
    release_region(r);
    return ok ? 0 : 1;
}

static int arm_survive(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    unsigned char * p = r.base + r.expert * 2;

    const int rc  = mlock_onfault(p, r.expert);
    const int prc = rc == 0 ? madvise(p, r.expert, MADV_POPULATE_READ) : -1;
    if (rc != 0 || prc != 0) {
        printf("cannot hold (%s): the guarantee cannot be tested on this box\n", strerror(errno));
        release_region(r);
        return 1;
    }
    printf("held             %zu/%zu resident\n", resident_pages(p, r.expert), np);

    // Evict what is evictable, then pressure RAM with anonymous pages.  PAGEOUT over the held expert's VMA
    // is refused (EINVAL: a VM_LOCKED VMA is not pageout-able), so evict around it -- which is exactly what
    // the design must do at eviction time anyway.
    for (size_t o = 0; o + r.expert <= r.bytes; o += r.expert) {
        if (r.base + o == p) continue;
        (void) page_out(r.base + o, r.expert);
    }
    const size_t press_bytes = 4ull << 30;
    unsigned char * press = (unsigned char *) mmap(nullptr, press_bytes, PROT_READ | PROT_WRITE,
                                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (press != MAP_FAILED) {
        for (size_t i = 0; i < press_bytes; i += (size_t) sysconf(_SC_PAGESIZE)) {
            press[i] = (unsigned char) ((i * 2654435761u) >> 24);
            touched += (size_t) sysconf(_SC_PAGESIZE);
        }
    }
    printf("pressure         %s anonymous pages touched\n", mib((double) touched).c_str());
    printf("held expert      %zu/%zu resident after pressure (the guarantee)\n",
           resident_pages(p, r.expert), np);

    // The control: a neighbour in the same region, not held, must have been reclaimed.
    unsigned char * nb = r.base + r.expert * 8;
    const size_t nb_res = resident_pages(nb, r.expert);
    printf("unheld neighbour %zu/%zu resident (control: pressure is real)\n", nb_res, np);

    double us = 0;
    const bool ok = gpu_read_verify(p, r.expert, &us);
    printf("in-place read    %8.1f us  %s\n", us, ok ? "DATA IDENTICAL" : "*** MISMATCH ***");
    munlock(p, r.expert);
    if (press != MAP_FAILED) munmap(press, press_bytes);
    release_region(r);
    return (ok && nb_res < np) ? 0 : 1;
}

static int arm_granularity(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t budget = memlock_budget();
    printf("budget        %s\n", mib((double) budget).c_str());

    const int v0 = vma_count();
    std::vector<unsigned char *> locked;
    size_t k = 0;
    while ((k + 1) * r.expert <= budget && (k + 1) * r.expert <= r.bytes / 4) {
        unsigned char * p = r.base + (r.bytes / 4) + k * r.expert;
        if (mlock_onfault(p, r.expert) != 0) break;
        locked.push_back(p);
        k++;
    }
    const int v1 = vma_count();
    const double a = now_us();
    for (auto * q : locked) munlock(q, r.expert);
    const double b = now_us();
    printf("expert-granular  %zu locks of %s  VMAs %d -> %d (+%d)  per-lock %.1f us  munlock total %.1f us\n",
           k, mib((double) r.expert).c_str(), v0, v1, v1 - v0, k ? (b - a) / (double) k : 0.0, b - a);

    const size_t chunk = tmin<size_t>(budget & ~((size_t) sysconf(_SC_PAGESIZE) - 1), 8ull << 20);
    const int v2 = vma_count();
    const double c = now_us();
    const int rc = chunk ? mlock_onfault(r.base + r.bytes / 4, chunk) : -1;
    const double d = now_us();
    const int v3 = vma_count();
    printf("chunk-granular   1 lock of %s  VMAs %d -> %d (+%d)  mlock2 %.1f us  %s\n",
           mib((double) chunk).c_str(), v2, v3, v3 - v2, d - c, rc == 0 ? "ok" : strerror(errno));
    if (rc == 0) munlock(r.base + r.bytes / 4, chunk);

    // The refusal path: more than the budget.  This is what the fallback (bounce) exists for.
    const size_t over = tmin<size_t>(budget + (64ull << 20), r.bytes / 2);
    const double e = now_us();
    const int rrc = mlock_onfault(r.base + r.bytes / 2, over);
    const double f = now_us();
    printf("over-budget      lock %s  %.1f us  -> %s\n", mib((double) over).c_str(), f - e,
           rrc == 0 ? "*** SUCCEEDED (RLIMIT_MEMLOCK is not being enforced) ***" : strerror(errno));
    if (rrc == 0) munlock(r.base + r.bytes / 2, over);

    // Eager mlock for comparison: it holds AND populates, by faulting page-by-page.
    const size_t plen = tmin<size_t>(budget & ~((size_t) sysconf(_SC_PAGESIZE) - 1), 8ull << 20);
    (void) page_out(r.base + r.bytes / 2, plen);
    const double g = now_us();
    const int prc = mlock(r.base + r.bytes / 2, plen);
    const double h = now_us();
    printf("mlock (eager)    %s  %.1f us  %s  (populates by faulting page-by-page)\n",
           mib((double) plen).c_str(), h - g, prc == 0 ? "ok" : strerror(errno));
    if (prc == 0) munlock(r.base + r.bytes / 2, plen);
    release_region(r);
    return 0;
}

// Does registration (or a first device read) PIN the pages?  This decides whether a per-window hold is
// even the right mechanism: if `hipHostRegister` GUP-pins everything resident at registration time, then
// "register the tensor once, hold only the window" is already pinning the tensor, and the campaign's
// premise changes.  Evidence is mincore through a FRESH mapping (page cache), never the registered one.
static int arm_pin(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);   // mapped, NOT registered
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    const size_t off = r.off;

    auto report = [&](const char * what) {
        printf("  %-34s own-ptes %3zu/%zu   page-cache %3zu/%zu   VmLck %4ld KiB  Mlocked %6ld MiB  Unevict %6ld MiB\n",
               what, resident_pages(r.base, r.expert), np, fresh_resident(r.fd, off, r.expert), np,
               vmlck_kb(), meminfo_kb("Mlocked") / 1024, meminfo_kb("Unevictable") / 1024);
    };

    printf("-- A. control: NOT registered --\n");
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    report("populated");
    if (!page_out(r.base, r.expert)) printf("  PAGEOUT refused\n");
    report("after PAGEOUT");

    printf("-- B. registered, never device-read --\n");
    do_register(r, nullptr);
    report("after register");
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    if (!page_out(r.base, r.expert)) printf("  PAGEOUT refused\n");
    report("populate + PAGEOUT");
    (void) hipHostUnregister(r.base);
    r.registered = false;

    printf("-- C. registered, then device-read --\n");
    do_register(r, nullptr);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    double us = 0;
    const bool ok = gpu_read_verify(r.base, r.expert, &us);
    printf("  device read %s (%.1f us)\n", ok ? "ok" : "MISMATCH", us);
    report("after device read");
    if (!page_out(r.base, r.expert)) printf("  PAGEOUT refused\n");
    report("after PAGEOUT");

    printf("-- D. the same range, held (mlock2), then PAGEOUT --\n");
    const int rc = mlock_onfault(r.base, r.expert);
    printf("  mlock2 %s\n", rc == 0 ? "ok" : strerror(errno));
    if (!page_out(r.base, r.expert)) printf("  PAGEOUT refused (EINVAL is expected over a locked VMA)\n");
    report("held + PAGEOUT");

    printf("\n  verdict: if 'page-cache' stays at %zu/%zu after PAGEOUT while 'own-ptes' drops, then the\n"
           "  pages are RETAINED (pinned by the driver or simply not reclaimed) and a cold claim needs\n"
           "  real pressure, not madvise; if it drops, PAGEOUT is a valid eviction control here.\n", np, np);
    release_region(r);
    return 0;
}


// ---- P0.5: cgroup v2 accounting (a registered range's bytes have to show up SOMEWHERE) ----
static std::string cg_dir() {
    FILE * f = fopen("/proc/self/cgroup", "r");
    if (!f) return "";
    char line[512];
    std::string path;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '0' && line[1] == ':' && line[2] == ':') {
            path = line + 3;
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
        }
    }
    fclose(f);
    return path.empty() ? "" : ("/sys/fs/cgroup" + path);
}

static long cg_kb(const char * key) {
    const std::string d = cg_dir();
    if (d.empty()) return -1;
    FILE * f = fopen((d + "/memory.stat").c_str(), "r");
    if (!f) return -1;
    char k[128];
    long v = -1;
    while (fscanf(f, "%127s %ld", k, &v) == 2) {
        if (strcmp(k, key) == 0) { fclose(f); return v; }
    }
    fclose(f);
    return -1;
}

static long cg_current_mib() {
    const std::string d = cg_dir();
    if (d.empty()) return -1;
    FILE * f = fopen((d + "/memory.current").c_str(), "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v < 0 ? -1 : v / 1048576;
}

static void ram_report(const char * what) {
    printf("  %-26s cgroup cur %5ld MiB  file %5ld  anon %5ld  unevict %5ld  slab %4ld MiB   MemAvailable %5ld MiB\n",
           what, cg_current_mib(), cg_kb("file") / 1024, cg_kb("anon") / 1024, cg_kb("unevictable") / 1024,
           cg_kb("slab_reclaimable") / 1024, meminfo_kb("MemAvailable") / 1024);
}

// ---- P0.5 #4: does hipHostUnregister RELEASE the hold?  The window can only slide if it does. ----
static int arm_release(const char * path, double expert_mib, bool unregister_first) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    double us = 0;
    if (!gpu_read_verify(r.base, r.expert, &us)) { printf("baseline read MISMATCH\n"); return 1; }
    printf("baseline: registered, populated, device read ok\n");

    if (unregister_first) {
        const double t0 = now_us();
        HIP_CHECK(hipHostUnregister(r.base));
        r.registered = false;
        const double t1 = now_us();
        printf("unregister: %.2f us\n", t1 - t0);
    } else {
        printf("control arm: NOT unregistered\n");
    }

    size_t fresh = (size_t) -1;
    for (int attempt = 0; attempt < 5; attempt++) {
        if (!page_out(r.base, r.expert)) return 3;
        fresh = fresh_resident(r.fd, r.off, r.expert);
        if (fresh == 0) break;
    }
    printf("page cache after PAGEOUT: %zu/%zu (fresh mapping)\n",
           fresh, r.expert / (size_t) sysconf(_SC_PAGESIZE));
    fflush(stdout);
    alarm(20);
    const bool ok = gpu_read_verify(r.base, r.expert, &us);
    printf("device read after PAGEOUT: %s\n", ok ? "SUCCEEDED (the hold was NOT released)" : "mismatch");
    printf("%s\n", ok ? "  -> unregister did not release (or PAGEOUT cannot drop it): the window cannot\n"
                        "     slide and the mode is a whole-tensor hold"
                      : "  -> unheld + dropped is unreadable");
    return ok ? 2 : 0;
}

// Does the release actually give the memory BACK, or does the device keep a stale mapping to pages that
// the kernel may now reuse?  Register, verify, unregister, PAGEOUT, then flood the machine with a
// distinctive anonymous pattern and read the range again: identical = the driver held the page (RAM not
// returned); mismatch = the page was reused while the device still mapped it (silent corruption, the worst
// answer); fault = the mapping went away with the release (what the design needs).
static int arm_reuse(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    double us = 0;
    if (!gpu_read_verify(r.base, r.expert, &us)) { printf("baseline MISMATCH\n"); return 1; }
    HIP_CHECK(hipHostUnregister(r.base));
    r.registered = false;
    if (!page_out(r.base, r.expert)) return 3;
    printf("unregistered + PAGEOUT: page cache %zu/%zu\n",
           fresh_resident(r.fd, r.off, r.expert), r.expert / (size_t) sysconf(_SC_PAGESIZE));

    const size_t press = 8ull << 30;
    unsigned char * m = (unsigned char *) mmap(nullptr, press, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (m != MAP_FAILED) {
        for (size_t i = 0; i < press; i += (size_t) sysconf(_SC_PAGESIZE)) { m[i] = 0xC5; touched += (size_t) sysconf(_SC_PAGESIZE); }
        printf("reuse pressure: %s of 0xC5 anonymous pages\n", mib((double) touched).c_str());
    }
    printf("page cache now %zu/%zu\n", fresh_resident(r.fd, r.off, r.expert),
           r.expert / (size_t) sysconf(_SC_PAGESIZE));
    fflush(stdout);
    alarm(20);
    const bool ok = gpu_read_verify(r.base, r.expert, &us);
    printf("device read after release + reuse pressure: %s\n",
           ok ? "IDENTICAL (the driver still holds the page: RAM is not returned)"
              : "*** DATA CHANGED (the page was reused under a live device mapping: corruption) ***");
    if (m != MAP_FAILED) munmap(m, press);
    return ok ? 2 : 1;
}

// P0.5 escape hatch: if `hipHostUnregister` does not release the hold, does unmapping the VMA?  The window
// could then be a fixed-address mapping that is remapped per slide (the delivery's tensors keep their
// pointers; the design re-maps AT THE SAME VA).  The measure is cgroup `file`: it must go back down.
static int arm_remap(const char * path, double expert_mib, double win_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t W = tmin<size_t>((size_t) (win_mib * 1048576.0), r.bytes / 4);
    unsigned char * va = (unsigned char *) mmap(nullptr, W, PROT_NONE,
                                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (va == MAP_FAILED) { printf("reserve failed: %s\n", strerror(errno)); return 3; }
    const size_t off1 = r.off;
    const size_t off2 = r.off + r.bytes / 2;

    auto map_at = [&](size_t off) {
        void * p = mmap(va, W, PROT_READ, MAP_SHARED | MAP_FIXED, r.fd, (off_t) off);
        if (p == MAP_FAILED) { printf("mmap at VA failed: %s\n", strerror(errno)); exit(1); }
        return (unsigned char *) p;
    };
    auto dev_read = [&](unsigned char * p) {
        double us = 0;
        if (!gpu_read_verify(p, r.expert, &us)) { printf("  read MISMATCH\n"); return false; }
        return true;
    };

    printf("window %s at a fixed VA (%p)\n", mib((double) W).c_str(), (void *) va);
    ram_report("baseline");
    unsigned char * w1 = map_at(off1);
    double t = 0;
    const double t0 = now_us();
    HIP_CHECK(hipHostRegister(w1, W, hipHostRegisterDefault));
    t = now_us() - t0;
    printf("register window: %.1f ms\n", t / 1000.0);
    (void) madvise(w1, W, MADV_POPULATE_READ);
    ram_report("window 1 registered");
    if (!dev_read(w1)) return 1;

    // Release route 1: unregister only.
    HIP_CHECK(hipHostUnregister(w1));
    ram_report("unregistered (VMA still mapped)");

    // Release route 2: unmap the VMA.
    munmap(w1, W);
    usleep(300 * 1000);
    ram_report("unmapped");

    // Re-map the SAME VA to a different file offset: pointer-stable windows.
    unsigned char * w2 = map_at(off2);
    (void) madvise(w2, W, MADV_POPULATE_READ);
    ram_report("window 2 (different offset)");
    if (!dev_read(w2)) return 1;   // unregistered: a device read here must be fatal if the release worked
    printf("  (an unregistered window was device-readable: the old registration is still in effect)\n");
    munmap(w2, W);
    munmap(va, W);
    release_region(r);
    return 0;
}

// The question that decides the campaign: after a window is unmapped, is its RAM really returned?  Use an
// offset no other arm has touched, so cgroup `file` grows by the window before the test and must come back
// down by it.  Pressure is real (3/4 of MemAvailable), because PAGEOUT is not.
static int arm_remapram(const char * path, double expert_mib, double win_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t W = tmin<size_t>((size_t) (win_mib * 1048576.0), r.bytes / 2);
    unsigned char * va = (unsigned char *) mmap(nullptr, W, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (va == MAP_FAILED) return 3;
    void * p = mmap(va, W, PROT_READ, MAP_SHARED | MAP_FIXED, r.fd, (off_t) (r.off + r.bytes / 2));
    if (p == MAP_FAILED) { printf("mmap: %s\n", strerror(errno)); return 3; }
    unsigned char * w = (unsigned char *) p;

    const long file0 = cg_kb("file") / 1024;
    ram_report("before the window");
    HIP_CHECK(hipHostRegister(w, W, hipHostRegisterDefault));
    (void) madvise(w, W, MADV_POPULATE_READ);
    ram_report("window registered+populated");
    const long file1 = cg_kb("file") / 1024;
    printf("  cgroup file grew by %ld MiB (the window is %s)\n", file1 - file0, mib((double) W).c_str());
    double us = 0;
    if (!gpu_read_verify(w, r.expert, &us)) { printf("read MISMATCH\n"); return 1; }
    printf("  device read of the window: ok\n");

    munmap(w, W);                       // release the VMA (the registration's lifetime is the VMA)
    usleep(300 * 1000);
    ram_report("window unmapped");

    const long avail_mib = meminfo_kb("MemAvailable") / 1024;
    const size_t press = (size_t) (avail_mib > 4096 ? (avail_mib * 3 / 4) : 1024) << 20;
    unsigned char * m = (unsigned char *) mmap(nullptr, press, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (m != MAP_FAILED) {
        for (size_t i = 0; i < press; i += (size_t) sysconf(_SC_PAGESIZE)) { m[i] = 0xA5; touched += (size_t) sysconf(_SC_PAGESIZE); }
    }
    printf("  pressure: %s\n", mib((double) touched).c_str());
    ram_report("after real pressure");
    const long file2 = cg_kb("file") / 1024;
    printf("  cgroup file after unmap+pressure: %ld MiB = baseline %+ld MiB\n", file2, file2 - file0);
    printf("  %s\n", (file1 - file2) >= (long) (W / 1048576) * 3 / 4
          ? "RELEASED: the window's bytes came back (>= 75% of it) -> a sliding window is viable"
          : "NOT RELEASED: the driver kept the pages; the hold is not per window");
    if (m != MAP_FAILED) munmap(m, press);
    munmap(va, W);
    release_region(r);
    return 0;
}

// Does the documented "tears under back-to-back use" reproduce for a 1-D expert fill, and does any host-side
// preparation stop it?  The hypotheses are separable:
//   pageable    plain page-cache pages                     (the delivery's suspect)
//   locked      mlock'ed first (resident, but NOT registered -- the driver does not know)
//   registered  hipHostRegister'd (the driver DOES know)
//   pinned      cudaMallocHost (the pool's source: the control that must never tear)
// and for each, N copies issued back-to-back over S streams with ONE sync at the end -- the overlap the
// delivery's async fill actually has.  Every copy is verified against its own source; when it is wrong we
// also report WHICH source's bytes arrived, because a shared bounce gives you copy j's bytes in copy i's
// destination, which is the signature to look for.
// LW_TEAR=pageable|locked|registered|pinned  LW_TEAR_N=8  LW_TEAR_STREAMS=2  LW_TEAR_ITERS=200
static int arm_tear(const char * path, double expert_mib) {
    const char * mode = getenv("LW_TEAR") ? getenv("LW_TEAR") : "pageable";
    const int ncopy  = getenv("LW_TEAR_N") ? atoi(getenv("LW_TEAR_N")) : 8;
    const int nstr   = getenv("LW_TEAR_STREAMS") ? atoi(getenv("LW_TEAR_STREAMS")) : 2;
    const int iters  = getenv("LW_TEAR_ITERS") ? atoi(getenv("LW_TEAR_ITERS")) : 200;

    region r = map_and_register(path, expert_mib);
    const size_t n = r.expert;
    const size_t need = (size_t) ncopy * n;
    if (need > r.bytes / 2) { printf("region too small for %d copies\n", ncopy); return 3; }

    const unsigned char * src = r.base;
    unsigned char * pinned = nullptr;
    if (strcmp(mode, "pinned") == 0) {
        HIP_CHECK(hipHostMalloc((void **) &pinned, need, hipHostMallocDefault));
        memcpy(pinned, r.base, need);
        src = pinned;
    } else if (strcmp(mode, "registered") == 0) {
        HIP_CHECK(hipHostRegister(r.base, need, hipHostRegisterDefault));
        r.registered = true;
    } else if (strcmp(mode, "locked") == 0) {
        const int rc = mlock((void *) r.base, need);
        if (rc != 0) { printf("mlock(%zu): %s -- cannot test this arm\n", need, strerror(errno)); return 3; }
        printf("mlock'ed %s (VmLck %ld KiB); the driver has NOT been told\n", mib((double) need).c_str(), vmlck_kb());
    } else if (strcmp(mode, "pageable") != 0) {
        printf("unknown LW_TEAR mode '%s'\n", mode); return 64;
    }

    std::vector<hipStream_t> st(nstr);
    for (int i = 0; i < nstr; i++) HIP_CHECK(hipStreamCreate(&st[i]));
    std::vector<unsigned char *> d(ncopy);
    for (int i = 0; i < ncopy; i++) HIP_CHECK(hipMalloc(&d[i], n));
    std::vector<std::vector<unsigned char>> back(ncopy, std::vector<unsigned char>(n));

    printf("mode=%s  copies=%d  streams=%d  iterations=%d  blob=%s  in flight=%s\n", mode, ncopy, nstr, iters,
           mib((double) n).c_str(), mib((double) need).c_str());
    fflush(stdout);

    int torn = 0, clean = 0;
    std::vector<std::pair<int,int>> sig;   // (copy i got j's bytes)
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < ncopy; i++) {
            HIP_CHECK(hipMemcpyAsync(d[i], src + (size_t) i * n, n, hipMemcpyHostToDevice, st[i % nstr]));
        }
        for (int i = 0; i < nstr; i++) HIP_CHECK(hipStreamSynchronize(st[i]));
        bool bad = false;
        for (int i = 0; i < ncopy; i++) {
            HIP_CHECK(hipMemcpy(back[i].data(), d[i], n, hipMemcpyDeviceToHost));
            if (memcmp(back[i].data(), src + (size_t) i * n, n) == 0) continue;
            bad = true;
            for (int j = 0; j < ncopy; j++) {
                if (j != i && memcmp(back[i].data(), src + (size_t) j * n, n) == 0) { sig.push_back({i, j}); break; }
            }
        }
        if (bad) torn++; else clean++;
        if (bad && torn <= 3) {
            printf("  iteration %d: TORN", it);
            for (auto & pr : sig) {
                if (pr.first >= 0) { printf("  (copy %d carries copy %d's bytes)", pr.first, pr.second); pr.first = -1; }
            }
            printf("\n");
        }
    }
    printf("result: %d/%d iterations clean, %d torn\n", clean, iters, torn);
    if (!sig.empty()) printf("       shared-bounce signature seen %zu times (copy i carrying copy j's bytes)\n", sig.size());
    printf("%s\n", torn == 0
        ? "  -> no tearing in this configuration: this source is safe for an overlapped 1-D fill"
        : "  -> TEARING REPRODUCED: this source cannot feed overlapped copies");

    for (int i = 0; i < ncopy; i++) (void) hipFree(d[i]);
    for (int i = 0; i < nstr; i++) (void) hipStreamDestroy(st[i]);
    if (pinned) (void) hipHostFree(pinned);
    if (strcmp(mode, "locked") == 0) munlock((void *) r.base, need);
    if (strcmp(mode, "registered") == 0) HIP_CHECK(hipHostUnregister(r.base));
    release_region(r);
    return torn ? 1 : 0;
}

// The delivery's warning names the 2-D path explicitly ("a pageable-source hipMemcpy2DAsync is
// pathologically slow AND can fault").  A 1-D expert fill does not use it, but the LOADER does -- and that is
// where the -sm tensor host-expert crash came from.  Measure it: fault, tear, or just slow?
// LW_TEAR2D=pageable|locked|registered  LW_TEAR_N=8  LW_TEAR_STREAMS=2  LW_TEAR_ITERS=20  LW_TEAR_W=4096
static int arm_tear2d(const char * path, double expert_mib) {
    const char * mode = getenv("LW_TEAR2D") ? getenv("LW_TEAR2D") : "pageable";
    const int ncopy = getenv("LW_TEAR_N") ? atoi(getenv("LW_TEAR_N")) : 8;
    const int nstr  = getenv("LW_TEAR_STREAMS") ? atoi(getenv("LW_TEAR_STREAMS")) : 2;
    const int iters = getenv("LW_TEAR_ITERS") ? atoi(getenv("LW_TEAR_ITERS")) : 20;
    const size_t w  = getenv("LW_TEAR_W") ? (size_t) atol(getenv("LW_TEAR_W")) : 4096;

    region r = map_and_register(path, expert_mib);
    const size_t n = r.expert - (r.expert % w);
    const size_t rows = n / w;
    if ((size_t) ncopy * n > r.bytes / 2) { printf("region too small\n"); return 3; }
    if (strcmp(mode, "registered") == 0) { HIP_CHECK(hipHostRegister(r.base, (size_t) ncopy * n, hipHostRegisterDefault)); r.registered = true; }
    else if (strcmp(mode, "locked") == 0) {
        if (mlock((void *) r.base, (size_t) ncopy * n) != 0) { printf("mlock: %s\n", strerror(errno)); return 3; }
    }
    const unsigned char * src = r.base;

    std::vector<hipStream_t> st(nstr);
    for (int i = 0; i < nstr; i++) HIP_CHECK(hipStreamCreate(&st[i]));
    std::vector<unsigned char *> d(ncopy);
    for (int i = 0; i < ncopy; i++) HIP_CHECK(hipMalloc(&d[i], n));
    std::vector<std::vector<unsigned char>> back(ncopy, std::vector<unsigned char>(n));

    printf("2-D copy: mode=%s copies=%d streams=%d %zux%zu (pitch %zu) iters=%d\n", mode, ncopy, nstr, rows, w,
           w, iters);
    fflush(stdout);
    int torn = 0;
    double t_first = 0;
    for (int it = 0; it < iters; it++) {
        const double a = now_us();
        for (int i = 0; i < ncopy; i++) {
            HIP_CHECK(hipMemcpy2DAsync(d[i], w, src + (size_t) i * n, w, w, rows, hipMemcpyHostToDevice, st[i % nstr]));
        }
        for (int i = 0; i < nstr; i++) HIP_CHECK(hipStreamSynchronize(st[i]));
        const double b = now_us();
        if (it == 0) t_first = b - a;
        bool bad = false;
        for (int i = 0; i < ncopy; i++) {
            HIP_CHECK(hipMemcpy(back[i].data(), d[i], n, hipMemcpyDeviceToHost));
            if (memcmp(back[i].data(), src + (size_t) i * n, n) != 0) bad = true;
        }
        if (bad) torn++;
    }
    printf("first iteration: %.1f us for %d copies (%s)  -> %.1f GB/s per copy\n", t_first, ncopy,
           mib((double) ((size_t) ncopy * n)).c_str(), n / ((t_first / ncopy) * 1e-6) / 1e9);
    printf("result: %d/%d iterations torn\n", torn, iters);
    printf("%s\n", torn ? "  -> the 2-D pageable path tears" : "  -> no tearing, but compare the rate above");
    for (int i = 0; i < ncopy; i++) (void) hipFree(d[i]);
    for (int i = 0; i < nstr; i++) (void) hipStreamDestroy(st[i]);
    release_region(r);
    return torn ? 1 : 0;
}

// Can the HOST push an expert straight out of the pageable mapping -- no pinned slot, no registration?
// That is the "host pushes" direction without the pool's `memcpy`.  ROCm stages a pageable H2D through an
// internal bounce, so the answer is an empirical one; measure it against the pool's memcpy+CE and against a
// registered-range CE copy on the SAME bytes.
static int arm_dmapage(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    const int n = 16;
    unsigned char * d = nullptr;
    HIP_CHECK(hipMalloc(&d, r.expert));
    unsigned char * back = (unsigned char *) malloc(r.expert);
    std::vector<double> hot, cold;
    size_t ok = 0;

    // CE copy straight from pageable (page-cache) pages.
    for (int i = 0; i < n; i++) {
        unsigned char * p = r.base + (size_t) i * r.expert;
        (void) madvise(p, r.expert, MADV_POPULATE_READ);
        const double a = now_us();
        HIP_CHECK(hipMemcpyAsync(d, p, r.expert, hipMemcpyHostToDevice, 0));
        HIP_CHECK(hipDeviceSynchronize());
        const double b = now_us();
        HIP_CHECK(hipMemcpy(back, d, r.expert, hipMemcpyDeviceToHost));
        if (memcmp(back, p, r.expert) == 0) ok++;
        hot.push_back(b - a);
    }
    printf("CE H2D from PAGEABLE, hot   median %8.2f us  (%.1f GB/s)  %zu/%d identical\n",
           med(hot), r.expert / (med(hot) * 1e-6) / 1e9, ok, n);

    ok = 0;
    int skipped = 0;
    unsigned char * coldbase = r.base + r.bytes / 2;
    for (int i = 0; i < n; i++) {
        unsigned char * p = coldbase + (size_t) i * r.expert;
        (void) madvise(p, r.expert, MADV_POPULATE_READ);
        if (!page_out(p, r.expert)) { skipped++; continue; }
        const size_t fresh = fresh_resident(r.fd, (size_t) (p - r.base) + r.off, r.expert);
        // PAGEOUT leaves a handful of pages (measured 6-29 of 425); >= 90 % dropped is a cold source for a
        // TIMING measurement -- the copy faults the rest in exactly as it would the whole range.
        if (fresh > 0 && fresh * 10 > r.expert / (size_t) sysconf(_SC_PAGESIZE)) {
            printf("  (cold skip: %zu/%zu still cached)\n", fresh, r.expert / (size_t) sysconf(_SC_PAGESIZE));
            skipped++;
            continue;
        }
        const double a = now_us();
        HIP_CHECK(hipMemcpyAsync(d, p, r.expert, hipMemcpyHostToDevice, 0));
        HIP_CHECK(hipDeviceSynchronize());
        const double b = now_us();
        HIP_CHECK(hipMemcpy(back, d, r.expert, hipMemcpyDeviceToHost));
        if (memcmp(back, p, r.expert) == 0) ok++;
        cold.push_back(b - a);
    }
    if (cold.empty()) printf("  (cold arm: %d/%d blobs could not be proven non-resident)\n", skipped, n);
    if (!cold.empty()) {
        printf("CE H2D from PAGEABLE, cold* median %8.2f us  (%.1f GB/s)  %zu/%zu identical  (*>=90%% paged out)\n",
               med(cold), r.expert / (med(cold) * 1e-6) / 1e9, ok, cold.size());
    }

    // Now the same bytes from a REGISTERED range, for the comparison (registration = the permanent hold).
    HIP_CHECK(hipHostRegister(r.base, r.expert * 64, hipHostRegisterDefault));
    std::vector<double> reg;
    for (int i = 0; i < n; i++) {
        unsigned char * p = r.base + (size_t) (16 + i) * r.expert;
        (void) madvise(p, r.expert, MADV_POPULATE_READ);
        const double a = now_us();
        HIP_CHECK(hipMemcpyAsync(d, p, r.expert, hipMemcpyHostToDevice, 0));
        HIP_CHECK(hipDeviceSynchronize());
        const double b = now_us();
        reg.push_back(b - a);
    }
    printf("CE H2D from REGISTERED      median %8.2f us  (%.1f GB/s)  (the prior campaign's arm B)\n",
           med(reg), r.expert / (med(reg) * 1e-6) / 1e9);
    printf("   pool for the same blob: memcpy 50.1 us + CE 66.4 us = 137.3 us (prior campaign, same box)\n");
    free(back);
    (void) hipFree(d);
    release_region(r);
    return 0;
}

// The clean version of the release test: a SCRATCH file on real disk whose pages cannot already be in the
// page cache, so `file` in cgroup accounting grows by the window and must come back down when it is
// released.  Everything the earlier arms got wrong (pages already cached, KiB vs MiB labels) is fixed here.
static int arm_scratch(const char * path, double expert_mib, double win_mib) {
    (void) path; (void) expert_mib;
    const char * sp = getenv("LW_SCRATCH");
    const std::string file = sp ? sp : "/home/stew675/lw-scratch.bin";
    const size_t W = (size_t) (win_mib * 1048576.0);
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);

    const int fd = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { printf("open %s: %s\n", file.c_str(), strerror(errno)); return 3; }
    if (ftruncate(fd, (off_t) W) != 0) { printf("ftruncate: %s\n", strerror(errno)); return 3; }

    auto mib_of = [&](const char * k) { return cg_kb(k) / 1048576.0; };
    auto report = [&](const char * what) {
        printf("  %-30s file %8.1f MiB  anon %8.1f MiB  unevict %6.1f MiB  cur %6ld MiB  MemAvailable %6ld MiB\n",
               what, mib_of("file"), mib_of("anon"), mib_of("unevictable"), cg_current_mib(),
               meminfo_kb("MemAvailable") / 1024);
    };

    const double f0 = mib_of("file");
    report("baseline (empty scratch)");
    unsigned char * w = (unsigned char *) mmap(nullptr, W, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (w == MAP_FAILED) { printf("mmap: %s\n", strerror(errno)); return 3; }
    HIP_CHECK(hipHostRegister(w, W, reg_flags()));
    (void) madvise(w, W, MADV_POPULATE_READ);
    report("window registered+populated");
    const double f1 = mib_of("file");
    printf("  file grew by %.1f MiB for a %.1f MiB window (expected ~ the window)\n", f1 - f0, W / 1048576.0);

    unsigned char * d = nullptr;
    HIP_CHECK(hipMalloc(&d, 1 << 20));
    kern_launch(w, d, tmin<size_t>(1 << 20, W));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  device read of the window: ok\n");
    (void) hipFree(d);

    munmap(w, W);
    usleep(300 * 1000);
    report("window unmapped");

    const long avail = meminfo_kb("MemAvailable") / 1024;
    const size_t press = (size_t) (avail > 4096 ? (avail * 3 / 4) : 1024) << 20;
    unsigned char * m = (unsigned char *) mmap(nullptr, press, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (m != MAP_FAILED) {
        for (size_t i = 0; i < press; i += pg) { m[i] = 0xA5; touched += pg; }
    }
    printf("  pressure: %.0f MiB anonymous\n", touched / 1048576.0);
    report("after real pressure");
    const double f2 = mib_of("file");
    printf("  file: baseline %.1f -> window %.1f -> released %.1f MiB (delta from baseline %+.1f)\n",
           f0, f1, f2, f2 - f0);
    const bool released = (f1 - f2) >= (W / 1048576.0) * 0.75;
    printf("  %s\n", released
          ? "RELEASED: the window's bytes came back -> a sliding window is viable (re-map per slide)"
          : "NOT RELEASED: the bytes stayed -> the hold is not per window, and the mode cannot bound RAM");
    if (m != MAP_FAILED) munmap(m, press);
    close(fd);
    unlink(file.c_str());
    return released ? 0 : 1;
}

// P0.6 (for `wip/host-expert-eager-push/`): the mirror of `arm scratch`, with **mlock instead of
// registration**.  The push design wants a bounded PINNED WORKING SET over the mapping, with the remainder
// reclaimable -- which is only possible if mlock is releasable.  Measure: does an mlocked window stay resident
// under real pressure, and does `munlock` give the pages back (cgroup `file` must drop)?
//   LW_SCRATCH=<path>   scratch file (default /home/stew675/lw-scratch.bin)
static int arm_mlockram(const char * path, double expert_mib, double win_mib) {
    (void) path; (void) expert_mib;
    const char * sp = getenv("LW_SCRATCH");
    const std::string file = sp ? sp : "/home/stew675/lw-scratch.bin";
    const size_t W = (size_t) (win_mib * 1048576.0);
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    auto mib_of = [&](const char * k) { return cg_kb(k) / 1048576.0; };

    const int fd = open(file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { printf("open %s: %s\n", file.c_str(), strerror(errno)); return 3; }
    if (ftruncate(fd, (off_t) W) != 0) { printf("ftruncate: %s\n", strerror(errno)); return 3; }
    auto report = [&](const char * what) {
        const size_t npg = W / pg;
        printf("  %-28s WINDOW %6zu/%zu resident (fresh map)  VmLck %8.1f MiB  MemAvailable %6ld MiB\n", what,
               fresh_resident(fd, 0, W), npg, vmlck_kb() / 1024.0, meminfo_kb("MemAvailable") / 1024);
    };
    unsigned char * w = (unsigned char *) mmap(nullptr, W, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (w == MAP_FAILED) { printf("mmap: %s\n", strerror(errno)); return 3; }

    report("baseline (empty scratch)");
    const double f0 = mib_of("file");
    const double t0 = now_us();
    const int rc = mlock(w, W);
    const double t1 = now_us();
    printf("  mlock(%s) -> %s  (%.1f us, %.2f us/MiB)\n", mib((double) W).c_str(),
           rc == 0 ? "ok" : strerror(errno), t1 - t0, (t1 - t0) / (W / 1048576.0));
    if (rc != 0) { printf("  cannot test mlock at this size (RLIMIT_MEMLOCK/VmLck)\n"); return 3; }
    (void) madvise(w, W, MADV_POPULATE_READ);
    report("mlocked + populated");
    const double f1 = mib_of("file");

    // Pressure must be HELD while the residency is measured: an allocate-then-free loop leaves the page
    // cache no reason to reclaim, which is the trap that made an earlier version of this arm say
    // "not released" for both mlock and registration.
    const long avail = meminfo_kb("MemAvailable") / 1024;
    const size_t press = (size_t) (avail > 4096 ? (long) (avail * 3 / 4) : 1024) << 20;
    unsigned char * m = (unsigned char *) mmap(nullptr, press, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (m != MAP_FAILED) {
        for (size_t i = 0; i < press; i += pg) { m[i] = 0x3C; touched += pg; }
    }
    printf("  pressure held: %.0f MiB anonymous\n", (double) touched / 1048576.0);
    report("mlocked, under pressure");
    const size_t held = fresh_resident(fd, 0, W);

    // The deterministic release proof: MADV_PAGEOUT is REFUSED over a VM_LOCKED VMA and must succeed after
    // munlock.  (Real pressure alone cannot prove it: the anon pages are already touched, so the kernel has
    // no new demand to reclaim for.)
    printf("  MADV_PAGEOUT while mlocked -> %s\n",
           madvise(w, W, MADV_PAGEOUT) == 0 ? "accepted (unexpected)" : strerror(errno));
    report("PAGEOUT attempted while mlocked");

    const double t2 = now_us();
    const int urc = munlock(w, W);
    const double t3 = now_us();
    printf("  munlock(%s) -> %s  (%.1f us, %.2f us/MiB)\n", mib((double) W).c_str(),
           urc == 0 ? "ok" : strerror(errno), t3 - t2, (t3 - t2) / (W / 1048576.0));
    printf("  MADV_PAGEOUT after munlock -> %s\n",
           madvise(w, W, MADV_PAGEOUT) == 0 ? "accepted" : strerror(errno));
    report("PAGEOUT after munlock");
    const size_t rel = fresh_resident(fd, 0, W);
    printf("  window residency: mlocked %zu/%zu -> after munlock %zu/%zu (pressure held)\n", held, W / pg, rel,
           W / pg);
    const bool released = rel * 2 < (W / pg);
    printf("  %s\n", (held == W / pg && released)
          ? "RELEASED: mlock holds under pressure AND munlock gives the pages back -> a bounded pinned "
            "working set over the mapping is viable"
          : (held != W / pg ? "mlock did not hold under pressure"
                            : "munlock did not release while pressure was held"));
    if (m != MAP_FAILED) munmap(m, press);
    munmap(w, W);
    close(fd);
    unlink(file.c_str());
    return released ? 0 : 1;
}

// ---- P0.5 #3: where do a registered range's bytes live? ----
static int arm_ram(const char * path, double expert_mib, double ram_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t big = tmin<size_t>((size_t) (ram_mib * 1048576.0), r.bytes);
    printf("region %s\n", mib((double) big).c_str());
    ram_report("start");
    (void) madvise(r.base, big, MADV_POPULATE_READ);
    ram_report("populated (unmapped-by-dev)");
    do_register(r, nullptr);
    ram_report("registered");
    (void) page_out(r.base, big);
    ram_report("PAGEOUT (driver holds)");
    printf("  page cache now %zu/%zu (fresh mapping)\n", fresh_resident(r.fd, r.off, big),
           big / (size_t) sysconf(_SC_PAGESIZE));
    HIP_CHECK(hipHostUnregister(r.base));
    r.registered = false;
    ram_report("unregistered");
    usleep(300 * 1000);
    ram_report("unregistered + 300ms");
    for (int i = 0; i < 3; i++) {
        (void) page_out(r.base, big);
        usleep(200 * 1000);
        ram_report("PAGEOUT after release");
    }
    printf("  page cache now %zu/%zu (fresh mapping)\n", fresh_resident(r.fd, r.off, big),
           big / (size_t) sysconf(_SC_PAGESIZE));
    // Real pressure: anonymous pages, just over half of what is available.
    const long avail_mib = meminfo_kb("MemAvailable") / 1024;
    const size_t press = (size_t) (avail_mib > 4096 ? (avail_mib * 3 / 4) : 1024) << 20;
    unsigned char * m = (unsigned char *) mmap(nullptr, press, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    size_t touched = 0;
    if (m != MAP_FAILED) {
        for (size_t i = 0; i < press; i += (size_t) sysconf(_SC_PAGESIZE)) { m[i] = 0x5A; touched += (size_t) sysconf(_SC_PAGESIZE); }
    }
    printf("  pressure: %s anonymous touched\n", mib((double) touched).c_str());
    ram_report("after real pressure");
    printf("  page cache now %zu/%zu (fresh mapping)\n", fresh_resident(r.fd, r.off, big),
           big / (size_t) sysconf(_SC_PAGESIZE));
    if (m != MAP_FAILED) munmap(m, press);
    printf("  (cgroup `file` is the page-cache tier: a %s drop means the range was really released)\n",
           mib((double) big).c_str());
    release_region(r);
    return 0;
}

// ---- P0.5 #1: a registration LARGER than the memory available to it.  Run it under a memory.max
// cgroup: if the driver faults the whole range in, the registration must fail or the process must be
// OOM-killed; if it faults in only what it can, some page must be unreadable later.  Either answer is
// what the design needs to know.
static int arm_big(const char * path, double expert_mib, double big_mib) {
    region r = map_and_register(path, expert_mib);
    // Its OWN region: this arm must be able to register more than the 512 MiB the other arms map.
    const size_t S = tmin<size_t>((size_t) (big_mib * 1048576.0),
                                  tmin<size_t>(r.bytes * 16, 8192ull << 20));
    const size_t np = S / (size_t) sysconf(_SC_PAGESIZE);
    printf("range to register %s   MemAvailable %ld MiB   cgroup max %ld MiB\n", mib((double) S).c_str(),
           meminfo_kb("MemAvailable") / 1024, ({ FILE * f = fopen((cg_dir() + "/memory.max").c_str(), "r");
               long v = -1; if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); } v < 0 ? -1L : v / 1048576; }));
    ram_report("before register");
    // Map the whole S from the file (the shared region is only 512 MiB).
    unsigned char * whole = (unsigned char *) mmap(nullptr, S, PROT_READ, MAP_SHARED, r.fd, (off_t) r.off);
    if (whole == MAP_FAILED) { printf("mmap %s: %s\n", mib((double) S).c_str(), strerror(errno)); return 3; }
    r.base = whole;
    r.bytes = S;
    r.registered = false;
    const double t0 = now_us();
    const hipError_t e = hipHostRegister(r.base, S, reg_flags());
    const double t1 = now_us();
    printf("hipHostRegister -> %s  (%.1f ms)\n", e == hipSuccess ? "ok" : hipGetErrorString(e), (t1 - t0) / 1000.0);
    if (e != hipSuccess) { (void) hipGetLastError(); release_region(r); return 10; }
    r.registered = true;
    ram_report("after register");

    // Sample residency at 16 evenly spaced expert windows: faulted in, or not?
    const int ns = 16;
    size_t resident = 0;
    for (int i = 0; i < ns; i++) {
        const size_t off = (size_t) ((double) (S - r.expert) * (double) i / (double) (ns - 1));
        if (fresh_resident(r.fd, r.off + off, r.expert) == r.expert / (size_t) sysconf(_SC_PAGESIZE)) resident++;
    }
    printf("residency sample: %zu/%d windows fully resident\n", resident, ns);
    if (resident == 0) { printf("nothing faulted in; nothing to read\n"); return 11; }

    // Read the LAST sampled window (highest offset).  If the registration could not fault it in, this is
    // the fatal case the prior campaign saw.
    const size_t off = S - r.expert;
    printf("reading the last window (offset %zu, %s) ...\n", off, mib((double) r.expert).c_str());
    fflush(stdout);
    alarm(20);
    double us = 0;
    const bool ok = gpu_read_verify(r.base + off, r.expert, &us);
    printf("device read of the last window: %s\n", ok ? "SUCCEEDED (faulted in)" : "mismatch");
    (void) hipHostUnregister(r.base);
    release_region(r);
    return ok ? 0 : 1;
}

// Is PAGEOUT alone enough to make a REGISTERED range unreadable?  Its own process, because the answer may
// be a fatal fault.  This is the sharpest open question in P0: `arm_pin` proves PAGEOUT drops the page
// cache (through a fresh mapping), yet a device read after it has been seen to succeed -- which would mean
// the driver keeps the physical page mapped, and then "cold" is not the same as "unmapped".
static int arm_evicted(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    // A single PAGEOUT sometimes leaves a handful of pages (measured 0/425 and 8/425 on two runs), so retry
    // until the range is provably empty or the budget of attempts runs out.
    size_t fresh = (size_t) -1;
    for (int attempt = 0; attempt < 5; attempt++) {
        if (!page_out(r.base, r.expert)) return 3;
        fresh = fresh_resident(r.fd, r.off, r.expert);
        if (fresh == 0) break;
    }
    printf("registered, populated, PAGEOUT -> page-cache %zu/%zu resident (fresh mapping), NOT held\n",
           fresh, np);
    if (fresh != 0) {
        printf("*** the page cache did not drop: this arm cannot test the hazard ***\n");
        return 3;
    }
    fflush(stdout);
    alarm(20);
    double us = 0;
    const bool ok = gpu_read_verify(r.base, r.expert, &us);
    printf("device read after PAGEOUT: %s (%.1f us)\n", ok ? "SUCCEEDED" : "mismatch", us);
    printf("%s\n", ok ? "  -> a registered range whose pages PAGEOUT dropped is still device-readable here:\n"
                        "     the driver keeps a mapping, so 'cold' != 'unmapped'.  The hazard needs a\n"
                        "     real reclaim pressure (the prior campaign used a set larger than RAM)."
                      : "  -> not readable");
    return ok ? 2 : 0;
}

// The GRANT cost, which is the number the simulator needs: register one expert range (which faults it in
// and holds it), verify the read, unregister.  Hot and cold.  This is the primitive the design's window
// would use per granted expert, so it replaces the pool's `memcpy` + CE fill in the cost model.
static int arm_regcost(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    const int n = 16;
    std::vector<double> hot_reg, hot_read, hot_unreg, cold_reg;
    size_t ok = 0;

    for (int i = 0; i < n; i++) {
        unsigned char * p = r.base + (size_t) i * r.expert;
        (void) madvise(p, r.expert, MADV_POPULATE_READ);
        const double a = now_us();
        HIP_CHECK(hipHostRegister(p, r.expert, hipHostRegisterDefault));
        const double b = now_us();
        if (gpu_read_verify(p, r.expert, nullptr)) ok++;
        const double c = now_us();
        HIP_CHECK(hipHostUnregister(p));
        const double d = now_us();
        hot_reg.push_back(b - a);
        hot_read.push_back(c - b - 0.0);
        hot_unreg.push_back(d - c);
    }
    printf("grant (hot)   register %8.2f us   verify %8.2f us   unregister %8.2f us   (%zu/%d identical)\n",
           med(hot_reg), med(hot_read), med(hot_unreg), ok, n);
    printf("              register %.2f us/MiB  = %.2f ms/MiB\n", med(hot_reg) / (r.expert / 1048576.0),
           med(hot_reg) / (r.expert / 1048576.0) / 1000.0);

    for (int i = 0; i < n; i++) {
        unsigned char * p = r.base + (size_t) (32 + i) * r.expert;
        (void) madvise(p, r.expert, MADV_POPULATE_READ);
        if (!page_out(p, r.expert)) { printf("  (cold: PAGEOUT refused at %d)\n", 32 + i); continue; }
        const size_t fresh = fresh_resident(r.fd, (size_t) (p - r.base) + r.off, r.expert);
        if (fresh != 0) { printf("  (cold: page cache still %zu/%zu at %d -- another reader?)\n", fresh, np, 32 + i); continue; }
        const double a = now_us();
        HIP_CHECK(hipHostRegister(p, r.expert, hipHostRegisterDefault));
        const double b = now_us();
        if (gpu_read_verify(p, r.expert, nullptr)) ok++;
        (void) hipHostUnregister(p);
        cold_reg.push_back(b - a);
    }
    if (!cold_reg.empty()) {
        printf("grant (cold)  register %8.2f us (%.2f ms/MiB)  over %zu provably non-resident blobs\n",
               med(cold_reg), med(cold_reg) / (r.expert / 1048576.0) / 1000.0, cold_reg.size());
    }
    printf("              a grant is: the driver faults the range in and holds it (see arm 'absent')\n");
    release_region(r);
    return 0;
}

// The pageable arm: NO registration.  This is the refuted shape (issue #116) and it must be fatal; it is
// here so the campaign's own record proves the mapping is what makes an in-place read legal.
static int arm_pageable(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    printf("pageable (NOT registered), populated \n");
    fflush(stdout);
    alarm(20);
    double us = 0;
    (void) gpu_read_verify(r.base, r.expert, &us);
    printf("device read SUCCEEDED -> the pageable arm is not fatal here?!\n");
    return 2;
}

// Register a range that is NOT resident.  If registration faults the pages in (a silent hold), the read
// works; if it maps only what is present, the read faults.  This decides what "register the window on
// grant" actually guarantees.
static int arm_absent(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    if (!page_out(r.base, r.expert)) return 3;
    const size_t fresh = fresh_resident(r.fd, r.off, r.expert);
    printf("paged out FIRST: page cache %zu/%zu (fresh mapping)\n", fresh, np);
    if (fresh != 0) return 3;
    do_register(r, nullptr);
    printf("registered over absent pages; page cache now %zu/%zu\n", fresh_resident(r.fd, r.off, r.expert), np);
    fflush(stdout);
    alarm(20);
    double us = 0;
    (void) gpu_read_verify(r.base, r.expert, &us);
    printf("device read SUCCEEDED -> registration faulted the pages in (a silent hold)\n");
    return 2;
}

// registered + populated + NOT held, then paged out: the read must fault.  Runs as its own process.
static int arm_unheld(const char * path, double expert_mib) {
    region r = map_and_register(path, expert_mib);
    do_register(r, nullptr);
    const size_t np = r.expert / (size_t) sysconf(_SC_PAGESIZE);
    (void) madvise(r.base, r.expert, MADV_POPULATE_READ);
    if (!page_out(r.base, r.expert)) return 3;
    printf("registered + populated + PAGEOUT -> %zu/%zu resident, NOT held\n", resident_pages(r.base, r.expert), np);
    fflush(stdout);
    alarm(20);   // a hang is the campaign's other measured failure mode; bound it
    double us = 0;
    const bool ok = gpu_read_verify(r.base, r.expert, &us);
    if (ok) {
        printf("*** the unheld read SUCCEEDED: this probe is not evicting its own range, so every\n"
               "    'survive' result is VOID ***\n");
        return 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);   // a fatal fault must not swallow the log

    const char * arm  = argc > 1 ? argv[1] : "base";
    double big_mib = argc > 4 ? atof(argv[4]) : 6144.0;
    const char * path = argc > 2 ? argv[2] : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const double emib = argc > 3 ? atof(argv[3]) : 1.66;

    printf("=== probe-locked-window: arm=%s expert=%.2f MiB ===\n", arm, emib);
    report_limits();

    if (strcmp(arm, "base") == 0)        return arm_base(path, emib);
    if (strcmp(arm, "hold-hot") == 0)    return arm_hold_hot(path, emib);
    if (strcmp(arm, "hold-cold") == 0)   return arm_hold_cold(path, emib);
    if (strcmp(arm, "survive") == 0)     return arm_survive(path, emib);
    if (strcmp(arm, "granularity") == 0) return arm_granularity(path, emib);
    if (strcmp(arm, "pin") == 0)         return arm_pin(path, emib);
    if (strcmp(arm, "evicted") == 0)     return arm_evicted(path, emib);
    if (strcmp(arm, "regcost") == 0)     return arm_regcost(path, emib);
    if (strcmp(arm, "release") == 0)     return arm_release(path, emib, true);
    if (strcmp(arm, "release-held") == 0) return arm_release(path, emib, false);
    if (strcmp(arm, "reuse") == 0)       return arm_reuse(path, emib);
    if (strcmp(arm, "ram") == 0)         return arm_ram(path, emib, big_mib > 0 ? big_mib : 2048.0);
    if (strcmp(arm, "remap") == 0)       return arm_remap(path, emib, big_mib > 0 ? big_mib : 256.0);
    if (strcmp(arm, "remapram") == 0)    return arm_remapram(path, emib, big_mib > 0 ? big_mib : 512.0);
    if (strcmp(arm, "dmapage") == 0)     return arm_dmapage(path, emib);
    if (strcmp(arm, "tear") == 0)        return arm_tear(path, emib);
    if (strcmp(arm, "tear2d") == 0)      return arm_tear2d(path, emib);
    if (strcmp(arm, "mlockram") == 0)    return arm_mlockram(path, emib, big_mib > 0 ? big_mib : 1024.0);
    if (strcmp(arm, "scratch") == 0)     return arm_scratch(path, emib, big_mib > 0 ? big_mib : 1024.0);
    if (strcmp(arm, "big") == 0)         return arm_big(path, emib, big_mib);
    if (strcmp(arm, "pageable") == 0)    return arm_pageable(path, emib);
    if (strcmp(arm, "absent") == 0)      return arm_absent(path, emib);
    if (strcmp(arm, "unheld") == 0)      return arm_unheld(path, emib);

    fprintf(stderr, "unknown arm '%s' (base|hold-hot|hold-cold|survive|granularity|pin|evicted|pageable|absent|regcost|release|release-held|reuse|ram|remap|remapram|scratch|mlockram|dmapage|tear|tear2d|big|unheld)\n", arm);
    return 64;
}
