// linkprobe: isolate the in-place host-read rate and the copy/overlap behaviour.
//
//  A. the delivery's probe exactly (128 MiB, first pass on a fresh mapping) vs its steady state
//  B. H2D copy vs in-place read, each alone
//  C. each running CONCURRENTLY with a compute kernel of the same duration (does it overlap, at what rate)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "FAIL %s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

static double wall_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

__global__ void read_probe(const uint4 * __restrict__ p, size_t n, unsigned * sink) {
    size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    const size_t st = (size_t) gridDim.x*blockDim.x;
    unsigned acc = 0;
    for (; i < n; i += st) { const uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x9e3779b9u) { *sink = acc; }
}

__global__ void burn_probe(float * __restrict__ dst, int iters) {
    float a = (float) (threadIdx.x + 1) * 1e-7f;
    float b = 1.0000001f;
    for (int i = 0; i < iters; ++i) { a = fmaf(a, b, 1e-7f); }
    if (a == -1.0f) { dst[threadIdx.x] = a; }
}

static double time_read(void * hd, size_t bytes, hipStream_t s, unsigned * sink) {
    const double t = wall_ms();
    read_probe<<<2048, 256, 0, s>>>((const uint4 *) hd, bytes/16, sink);
    CK(hipStreamSynchronize(s));
    return wall_ms() - t;
}

int main(int argc, char ** argv) {
    size_t mib = 512;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) { mib = (size_t) atoi(argv[++i]); }
    }
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    printf("device: %s (%s), L2 %.1f MiB\n", prop.name, prop.gcnArchName, prop.l2CacheSize/1048576.0);

    hipStream_t s0, s1;
    CK(hipStreamCreate(&s0));
    CK(hipStreamCreate(&s1));
    unsigned * sink = nullptr;
    CK(hipMalloc((void **) &sink, 1024));
    float * scratch = nullptr;
    CK(hipMalloc((void **) &scratch, 65536));

    // ---------- A. the delivery's probe shape: 128 MiB, FIRST pass on a fresh mapping ----------
    {
        const size_t sz = (size_t) 128 << 20;
        char * h = nullptr;
        CK(hipHostMalloc((void **) &h, sz, hipHostMallocMapped | hipHostMallocPortable));
        memset(h, 0x5a, sz);
        void * hd = nullptr;
        CK(hipHostGetDevicePointer(&hd, h, 0));
        CK(hipDeviceSynchronize());
        double t[5];
        for (int i = 0; i < 5; ++i) { t[i] = time_read(hd, sz, s0, sink); }
        printf("\n128 MiB mapped pinned host, five consecutive passes (GB/s):\n   ");
        for (int i = 0; i < 5; ++i) { printf("%6.2f ", sz/1e9/(t[i]*1e-3)); }
        printf("\n   first pass is what the delivery's calibration quotes; the rest is the steady state\n");
        CK(hipFreeHost(h));
    }

    // ---------- B/C. big buffer, cold vs warm, alone vs concurrent with compute ----------
    const size_t bytes = mib << 20;
    char * h = nullptr;
    CK(hipHostMalloc((void **) &h, bytes, hipHostMallocMapped | hipHostMallocPortable));
    memset(h, 0x5a, bytes);
    void * hd = nullptr;
    CK(hipHostGetDevicePointer(&hd, h, 0));
    char * d = nullptr;
    CK(hipMalloc((void **) &d, bytes));
    CK(hipMemset(d, 0x11, bytes));
    CK(hipDeviceSynchronize());

    const double ms_read1 = time_read(hd, bytes, s0, sink);
    const double ms_read2 = time_read(hd, bytes, s0, sink);
    printf("\n%zu MiB in-place host read : pass1 %6.2f GB/s, pass2 %6.2f GB/s (%6.2f / %6.2f ms)\n",
            mib, bytes/1e9/(ms_read1*1e-3), bytes/1e9/(ms_read2*1e-3), ms_read1, ms_read2);

    const double t_copy = wall_ms();
    CK(hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s0));
    CK(hipStreamSynchronize(s0));
    const double ms_copy = wall_ms() - t_copy;
    printf("%zu MiB H2D copy            : %6.2f GB/s (%6.2f ms)\n", mib, bytes/1e9/(ms_copy*1e-3), ms_copy);

    // a burn kernel tuned to roughly the duration of one transfer
    int iters = 40000;
    double ms_burn = 0;
    for (int tries = 0; tries < 6; ++tries) {
        const double b0 = wall_ms();
        burn_probe<<<1024, 256, 0, s1>>>(scratch, iters);
        CK(hipStreamSynchronize(s1));
        ms_burn = wall_ms() - b0;
        if (ms_burn > ms_read2*0.8 && ms_burn < ms_read2*1.25) { break; }
        iters = (int) (iters * (ms_read2*1.0/std::max(ms_burn, 0.1)));
    }
    printf("burn kernel                : %6.2f ms (%d fma/thread, matched to one transfer)\n", ms_burn, iters);

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));

    // copy on s0 while compute runs on s1
    double w0 = wall_ms();
    CK(hipEventRecord(e0, s0));
    burn_probe<<<1024, 256, 0, s1>>>(scratch, iters);
    CK(hipMemcpyAsync(d, h, bytes, hipMemcpyHostToDevice, s0));
    CK(hipEventRecord(e1, s0));
    CK(hipStreamSynchronize(s0));
    CK(hipStreamSynchronize(s1));
    double ms_wall = wall_ms() - w0;
    float ms_copy_c = 0.0f;
    CK(hipEventElapsedTime(&ms_copy_c, e0, e1));
    printf("\ncopy + compute             : wall %6.2f ms  (copy %6.2f ms = %6.2f GB/s, burn %6.2f ms)\n",
            ms_wall, (double) ms_copy_c, bytes/1e9/((double) ms_copy_c*1e-3), ms_burn);
    printf("                             %s\n",
            ms_wall < (ms_copy_c + ms_burn)*0.7 ? "OVERLAPPED"
            : ms_wall > (ms_copy_c + ms_burn)*0.9 ? "SERIALISED" : "PARTLY OVERLAPPED");

    // in-place read on s0 while compute runs on s1
    const double ms_readw = time_read(hd, bytes, s0, sink); // warm
    w0 = wall_ms();
    CK(hipEventRecord(e0, s0));
    burn_probe<<<1024, 256, 0, s1>>>(scratch, iters);
    read_probe<<<2048, 256, 0, s0>>>((const uint4 *) hd, bytes/16, sink);
    CK(hipEventRecord(e1, s0));
    CK(hipStreamSynchronize(s0));
    CK(hipStreamSynchronize(s1));
    ms_wall = wall_ms() - w0;
    float ms_read_c = 0.0f;
    CK(hipEventElapsedTime(&ms_read_c, e0, e1));
    printf("read + compute             : wall %6.2f ms  (read %6.2f ms = %6.2f GB/s, burn %6.2f ms; alone %6.2f GB/s)\n",
            ms_wall, (double) ms_read_c, bytes/1e9/((double) ms_read_c*1e-3), ms_burn,
            bytes/1e9/(ms_readw*1e-3));
    printf("                             %s\n",
            ms_wall < (ms_read_c + ms_burn)*0.7 ? "OVERLAPPED"
            : ms_wall > (ms_read_c + ms_burn)*0.9 ? "SERIALISED" : "PARTLY OVERLAPPED");

    CK(hipFreeHost(h));
    CK(hipFree(d));
    CK(hipFree(scratch));
    CK(hipFree(sink));
    return 0;
}
