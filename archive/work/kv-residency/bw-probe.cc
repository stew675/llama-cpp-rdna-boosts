// wip/kv-residency/bw-probe.cc -- the link constant a KV-residency design has to live with.
//
// Measures, on the attached device and at a size that matters for KV pages:
//   H2D  pinned    hipMemcpyAsync  host(pinned) -> device
//   H2D  pageable  hipMemcpyAsync  host(malloc) -> device
//   D2H  pinned    hipMemcpyAsync  device -> host(pinned)
//   D2D            hipMemcpyAsync  device -> device
//   read pinned    a kernel sums a pinned, device-mapped host buffer (the zero-copy read a
//                  Strata-style `kv_stream_resolve` copy kernel does, i.e. what a miss costs)
//   read native    the same kernel over device memory (the VRAM reference)
//
// Build (no CMake, no project state):
//   /opt/rocm-10.1.0-gfx120X/bin/hipcc -O3 --offload-arch=gfx1201 bw-probe.cc -o /tmp/bw-probe
// Run:
//   HIP_VISIBLE_DEVICES=0 /tmp/bw-probe [MiB] [reps]

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); return 1; } } while (0)

__global__ void sum_kernel(const uint4 * __restrict__ p, size_t n4, unsigned long long * __restrict__ out) {
    unsigned long long s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        s += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0xdeadbeefull) { atomicAdd(out, s); }   // never true: keeps the loop alive
}

static float time_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0.0f;
    hipEventElapsedTime(&ms, a, b);
    return ms;
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) atoll(argv[1]) : 512;
    const int    rep = argc > 2 ? atoi(argv[2]) : 5;
    const size_t nbytes = mib << 20;

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s), %zu MiB per arm, %d reps\n", prop.name, prop.gcnArchName, mib, rep);

    void * d = nullptr, * d2 = nullptr, * h_pin = nullptr, * h_pg = nullptr;
    unsigned long long * d_out = nullptr;
    CK(hipMalloc(&d, nbytes));
    CK(hipMalloc(&d2, nbytes));
    CK(hipHostMalloc(&h_pin, nbytes, hipHostMallocDefault));
    CK(hipHostMalloc(&h_pg, nbytes, hipHostMallocDefault));   // pageable would be malloc(); we want a *pinned* arm
    CK(hipMalloc(&d_out, sizeof(*d_out)));
    std::memset(h_pin, 0x5a, nbytes);

    hipStream_t s;
    CK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));

    // warm up (first touch, page table population)
    CK(hipMemcpyAsync(d, h_pin, nbytes, hipMemcpyHostToDevice, s));
    CK(hipStreamSynchronize(s));

    auto report = [&](const char * name, double ms) {
        // GB/s, decimal, payload bytes
        std::printf("  %-22s %8.2f ms  %7.1f GB/s\n", name, ms, (double) nbytes / (ms * 1e6));
    };

    double best;
    best = 1e9;
    for (int i = 0; i < rep; ++i) {
        CK(hipEventRecord(e0, s));
        CK(hipMemcpyAsync(d, h_pin, nbytes, hipMemcpyHostToDevice, s));
        CK(hipEventRecord(e1, s));
        CK(hipStreamSynchronize(s));
        best = std::min(best, (double) time_ms(e0, e1));
    }
    report("H2D pinned", best);

    // pageable source: timed with the host clock, because an async pageable copy is staged by the driver and
    // the device-side events do not cover it
    {
        void * h_malloc = std::malloc(nbytes);
        std::memset(h_malloc, 0x5a, nbytes);
        best = 1e9;
        for (int i = 0; i < rep; ++i) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            CK(hipMemcpyAsync(d, h_malloc, nbytes, hipMemcpyHostToDevice, s));
            CK(hipStreamSynchronize(s));
            clock_gettime(CLOCK_MONOTONIC, &t1);
            const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            best = std::min(best, ms);
        }
        report("H2D pageable (wall)", best);
        std::free(h_malloc);
    }

    best = 1e9;
    for (int i = 0; i < rep; ++i) {
        CK(hipEventRecord(e0, s));
        CK(hipMemcpyAsync(h_pin, d, nbytes, hipMemcpyDeviceToHost, s));
        CK(hipEventRecord(e1, s));
        CK(hipStreamSynchronize(s));
        best = std::min(best, (double) time_ms(e0, e1));
    }
    report("D2H pinned", best);

    CK(hipMemcpyAsync(d2, d, nbytes, hipMemcpyDeviceToDevice, s));
    CK(hipStreamSynchronize(s));
    best = 1e9;
    for (int i = 0; i < rep; ++i) {
        CK(hipEventRecord(e0, s));
        CK(hipMemcpyAsync(d2, d, nbytes, hipMemcpyDeviceToDevice, s));
        CK(hipEventRecord(e1, s));
        CK(hipStreamSynchronize(s));
        best = std::min(best, (double) time_ms(e0, e1));
    }
    report("D2D", best);

    // a kernel reading pinned host memory through its device mapping: what a resolve copy costs per byte
    void * d_map = nullptr;
    CK(hipHostGetDevicePointer(&d_map, h_pin, 0));
    const size_t n4 = nbytes / 16;
    const dim3 grid(1024), block(256);
    for (int arm = 0; arm < 2; ++arm) {
        const uint4 * src = arm == 0 ? (const uint4 *) d_map : (const uint4 *) d;
        sum_kernel<<<grid, block, 0, s>>>(src, n4, d_out);
        CK(hipStreamSynchronize(s));
        best = 1e9;
        for (int i = 0; i < rep; ++i) {
            CK(hipEventRecord(e0, s));
            sum_kernel<<<grid, block, 0, s>>>(src, n4, d_out);
            CK(hipEventRecord(e1, s));
            CK(hipStreamSynchronize(s));
            best = std::min(best, (double) time_ms(e0, e1));
        }
        report(arm == 0 ? "kernel read pinned" : "kernel read VRAM", best);
    }

    std::printf("note: the card sits behind a PCIe switch; the host uplink is what this probes.\n"
                "      read the *root port's* max_link_width, not the card's (here: card x16, uplink x8).\n");
    return 0;
}
