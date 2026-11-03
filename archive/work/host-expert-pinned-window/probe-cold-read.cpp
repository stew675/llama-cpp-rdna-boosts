// wip/host-expert-pinned-window: is a KERNEL (SM) read of a registered-but-reclaimed range safe?
//
// The CE recovers from a reclaimed page in a registered range (measured: the copy refills, slowly).  The
// COLD read arm (`moe_cache_get_cold`, id = n_res + e) instead has an SM read the expert in place, which on a
// no-XNACK GPU is the case issue #116 says is fatal.  If an SM read recovers too, a window-backed cold read
// needs no mlock at all; if it faults or returns garbage, the cold read must be gated on mlock.
//
//   build: ./build.sh    run: ./run-probe-cold.sh
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIP_CHECK(call)                                                                       \
    do {                                                                                      \
        hipError_t e_ = (call);                                                               \
        if (e_ != hipSuccess) {                                                               \
            fprintf(stderr, "HIP error %s at %d: %s\n", #call, __LINE__, hipGetErrorString(e_)); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

// Sum every byte, 16 at a time; write the sum to `out`.
__global__ void sum_kernel(const char * __restrict__ src, size_t n, unsigned long long * out) {
    const uint4 * p = (const uint4 *) src;
    const size_t  n16 = n / 16;
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[1] = 0;   // unused
    }
    atomicAdd(out, acc);   // out[0] = the whole-range checksum
}

static unsigned long long host_sum(const char * p, size_t n) {
    unsigned long long acc = 0;
    const uint4 * q = (const uint4 *) p;
    for (size_t i = 0; i < n / 16; i++) {
        uint4 v = q[i];
        acc += v.x + v.y + v.z + v.w;
    }
    return acc;
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1]
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const size_t n = 8 << 20;
    HIP_CHECK(hipSetDevice(0));
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    struct stat st{};
    fstat(fd, &st);
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    char * map = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    char * win = map + (((size_t) st.st_size / 3) & ~(page - 1));

    unsigned long long * dev_out = nullptr;
    unsigned long long   host_ref = host_sum(win, n);   // also pulls the range into the page cache
    HIP_CHECK(hipMalloc((void **) &dev_out, 4096));

    auto run = [&](const char * label) {
        HIP_CHECK(hipMemset(dev_out, 0, 16));
        sum_kernel<<<64, 256>>>(win, n, dev_out);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("%-34s DEVICE ERROR: %s\n", label, hipGetErrorString(e));
            (void) hipGetLastError();
            return;
        }
        unsigned long long got = 0;
        HIP_CHECK(hipMemcpy(&got, dev_out, sizeof got, hipMemcpyDeviceToHost));
        printf("%-40s %s (0x%llx vs ref 0x%llx)\n", label,
               got == host_ref ? "DATA IDENTICAL" : "DATA CORRUPT", got, host_ref); fflush(stdout);
    };

    // Registration is what makes the mapping device-addressable at all (issue #116): an unregistered
    // pageable mapping read by a kernel is a fatal fault, so phase 1 must be the registered+resident case.
    HIP_CHECK(hipHostRegister(win, n, hipHostRegisterReadOnly));
    run("kernel read, registered + resident");

    // ... then reclaimed through the driver's own advice
    int mv = madvise(win, n, MADV_PAGEOUT);
    printf("MADV_PAGEOUT rc=%d\n", mv); fflush(stdout);
    run("kernel read, REGISTERED + reclaimed");

    // mlocked, then reclaimed: the pages cannot go, so this is the safe reference
    const int ml = mlock(win, n);
    mv = madvise(win, n, MADV_PAGEOUT);
    printf("mlock rc=%d, MADV_PAGEOUT rc=%d\n", ml, mv); fflush(stdout);
    run("kernel read, MLOCKED + reclaimed");
    if (ml == 0) { munlock(win, n); }

    // REAL pressure: allocate and touch most of free RAM, forcing the kernel to actually reclaim
    {
        long total_kb = 0;
        FILE * mf = fopen("/proc/meminfo", "r");
        if (mf) { char l[256]; while (fgets(l, sizeof l, mf)) { long v; if (sscanf(l, "MemAvailable: %ld kB", &v) == 1) total_kb = v; } fclose(mf); }
        const size_t press = (size_t) (total_kb * 1024) * 80 / 100;
        char * hog = (char *) mmap(nullptr, press, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hog != MAP_FAILED) {
            for (size_t i = 0; i < press; i += page) { hog[i] = 1; }
            printf("pressed %zu MiB of anonymous RAM\n", press >> 20); fflush(stdout);
            run("kernel read, REGISTERED under REAL pressure");
            munmap(hog, press);
        }
    }

    char * dev = nullptr;
    HIP_CHECK(hipMalloc((void **) &dev, n));
    HIP_CHECK(hipMemcpy(dev, win, n, hipMemcpyHostToDevice));
    printf("%-34s copied\n", "CE read after reclaim");
    HIP_CHECK(hipHostUnregister(win));
    printf("\nhost checksum of the reference: %llu\n", host_ref);
    return 0;
}
