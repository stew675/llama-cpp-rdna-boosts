// hipHostMalloc(hipHostMallocMapped) + hipHostGetDevicePointer: does the GPU read it safely, and does it
// survive memory pressure?  (wip, 2026-11-02, answering a suggestion to use it against no-XNACK.)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define CKD(x) do { hipError_t e=(x); if (e) { printf("%-46s FAILED rc=%d (%s)\n", #x,(int)e,hipGetErrorString(e)); } } while (0)
__global__ void sum_k(const char * __restrict__ p, size_t n, unsigned long long * out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 16; i += (size_t) gridDim.x * blockDim.x)
        s += ((const unsigned long long *) (p + i * 16))[0];
    atomicAdd(out, s);
}
static unsigned long long host_sum(const char * p, size_t n) {
    unsigned long long s = 0;
    for (size_t i = 0; i < n / 16; i++) s += ((const unsigned long long *) (p + i * 16))[0];
    return s;
}
static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }
struct arm { const char * name; void * host; unsigned flags; };
int main() {
    const size_t n = 256u << 20;
    hipSetDevice(0);
    hipDeviceProp_t prop{}; hipGetDeviceProperties(&prop, 0);
    printf("device %s | unifiedAddressing(UVA) = %s | canMapHostMemory = %s\n", prop.name,
           prop.unifiedAddressing ? "yes" : "NO", prop.canMapHostMemory ? "yes" : "no");

    void * def = nullptr, * map = nullptr;
    CKD(hipHostMalloc(&def, n, hipHostMallocDefault));
    CKD(hipHostMalloc(&map, n, hipHostMallocMapped));
    // deterministic, non-zero contents so a skipped read cannot look identical
    for (size_t i = 0; i < n; i += 4096) { ((char *) def)[i] = (char) (i >> 12); ((char *) map)[i] = (char) ((i >> 12) ^ 0x5a); }
    memset((char *) def + n/2, 0xa5, 4096); memset((char *) map + n/2, 0x3c, 4096);

    void * def_dev = nullptr, * map_dev = nullptr;
    CKD(hipHostGetDevicePointer(&def_dev, def, 0));
    CKD(hipHostGetDevicePointer(&map_dev, map, 0));
    printf("alias def: host %% 4096 = %zu, dev==host? %s\n", (size_t) def & 4095, def_dev == def ? "yes (identity)" : "no");
    printf("alias map: host %% 4096 = %zu, dev==host? %s\n", (size_t) map & 4095, map_dev == map ? "yes (identity)" : "no");

    unsigned long long * out = nullptr; hipMalloc((void **) &out, 8);
    {   // POSITIVE CONTROL: kernel-read `def` but compare with `map`'s host sum -> must MISMATCH
        const unsigned long long ref_map = host_sum((const char *) map, n);
        hipMemset(out, 0, 8); sum_k<<<512,256>>>((const char *) def_dev, n, out);
        hipDeviceSynchronize(); unsigned long long got = 0; hipMemcpy(&got, out, sizeof got, hipMemcpyDeviceToHost);
        printf("CONTROL (read def, expect map's sum): %s\n", got == ref_map ? "WRONGLY IDENTICAL" : "correctly MISMATCH");
    }
    char * devbuf = nullptr; CKD(hipMalloc((void **) &devbuf, n));

    arm arms[2] = { {"hipHostMallocDefault", def, 0}, {"hipHostMallocMapped", map, 0} };
    for (int a = 0; a < 2; a++) {
        const char * host = (const char *) arms[a].host;
        void * alias = a == 0 ? def_dev : map_dev;
        const unsigned long long ref = host_sum(host, n);
        hipMemset(out, 0, 8);
        sum_k<<<512, 256>>>((const char *) alias, n, out);
        hipError_t e = hipDeviceSynchronize();
        unsigned long long got = 0; hipMemcpy(&got, out, sizeof got, hipMemcpyDeviceToHost);
        // bandwidth: kernel read and CE copy
        double best_k = 0, best_c = 0;
        for (int it = 0; it < 3; it++) {
            hipDeviceSynchronize(); double t0 = now_s();
            sum_k<<<512, 256>>>((const char *) alias, n, out); hipDeviceSynchronize();
            double g = (double) n / (now_s() - t0) / (1u << 30); if (g > best_k) best_k = g;
            t0 = now_s(); hipMemcpyAsync(devbuf, host, n, hipMemcpyDefault, 0); hipDeviceSynchronize();
            g = (double) n / (now_s() - t0) / (1u << 30); if (g > best_c) best_c = g;
        }
        printf("%-22s kernel read: rc=%d %s | SM %.1f GiB/s | CE %.1f GiB/s\n", arms[a].name, (int) e,
               got == ref ? "DATA IDENTICAL" : "DATA MISMATCH", best_k, best_c);
    }

    // real pressure: touch ~70% of MemAvailable, then re-read both (pinned pages must survive)
    long avail_kb = 0; { FILE * f = fopen("/proc/meminfo", "r"); char l[256];
        while (f && fgets(l, sizeof l, f)) { long v; if (sscanf(l, "MemAvailable: %ld kB", &v) == 1) avail_kb = v; } if (f) fclose(f); }
    const size_t press = (size_t) avail_kb * 1024 * 70 / 100;
    char * hog = (char *) mmap(nullptr, press, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hog != MAP_FAILED) { for (size_t i = 0; i < press; i += 4096) hog[i] = 1; }
    printf("\npressed %zu MiB of anonymous RAM\n", press >> 20);
    for (int a = 0; a < 2; a++) {
        const char * host = (const char *) arms[a].host;
        void * alias = a == 0 ? def_dev : map_dev;
        const unsigned long long ref = host_sum(host, n);
        hipMemset(out, 0, 8); sum_k<<<512, 256>>>((const char *) alias, n, out);
        hipError_t e = hipDeviceSynchronize();
        unsigned long long got = 0; hipMemcpy(&got, out, sizeof got, hipMemcpyDeviceToHost);
        printf("under pressure, %-22s kernel read: rc=%d %s\n", arms[a].name, (int) e,
               got == ref ? "DATA IDENTICAL" : "DATA MISMATCH");
    }
    { FILE * f = fopen("/proc/meminfo", "r"); char l[256];
      while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "Mlocked:", 8) || !strncmp(l, "Unevictable:", 12)) fputs(l, stdout);
      if (f) fclose(f); }
    return 0;
}
