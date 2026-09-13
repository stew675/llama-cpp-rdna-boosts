// ar_signal_bench: measure the arrival-token round-trip in isolation.
//
// Replicates the internal AR signal mechanism: two GPUs each write a token
// to a mapped-pinned host slot (64B-strided) and busy-wait on the peer's
// slot, recording clock64() spin durations per direction.  No llama.cpp, no
// graph, no RCCL -- this is the pure PCIe/host-memory signal path.
//
// Build: hipcc --offload-arch=gfx1201 ar_signal_bench.cpp -o ar_signal_bench
// Run:   HIP_VISIBLE_DEVICES=<a>,<b> ./ar_signal_bench [iters] [spin_iters]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include <chrono>

#define HK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP FAIL %d: %s\n", __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static constexpr int SLOT_STRIDE = 64;  // bytes between arrival ints

// Kernel: write token to my_slot, fence, spin on other_slot (16-iter dummy
// spin between polls, like the fork), record clock64 spin window in spin_out.
__global__ void ar_signal_kernel(int * my_slot, const int * other_slot, int token,
                                 int spin_iters, long long * spin_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const long long t0 = clock64();
        *(volatile int *)my_slot = token;
        __threadfence_system();
        while (*(const volatile int *)other_slot != token) {
            #pragma unroll
            for (int i = 0; i < spin_iters; ++i) { asm volatile("" ::: "memory"); }
        }
        spin_out[0] = t0;
        spin_out[1] = clock64();
    }
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 2000;
    const int spin_iters = argc > 2 ? atoi(argv[2]) : 16;

    int devcount = 0; HK(hipGetDeviceCount(&devcount));
    if (devcount < 2) { fprintf(stderr, "need >= 2 devices\n"); return 1; }

    // mapped pinned host memory: 2 slots, 64B apart (cache-line separated)
    int * host_slots = nullptr;
    HK(hipHostMalloc((void**)&host_slots, 2 * SLOT_STRIDE, hipHostMallocPortable));
    int * dev_slots = host_slots;   // GTT: host pointer is device-accessible on AMD
    HK(hipMemset(dev_slots, 0, 2 * SLOT_STRIDE));

    long long * spin0 = nullptr, * spin1 = nullptr;
    hipStream_t s0, s1;
    HK(hipSetDevice(0)); HK(hipMalloc((void**)&spin0, 2 * sizeof(long long))); HK(hipStreamCreate(&s0));
    HK(hipSetDevice(1)); HK(hipMalloc((void**)&spin1, 2 * sizeof(long long))); HK(hipStreamCreate(&s1));

    // device pointers into the mapped slots, per rank
    int * slot0 = dev_slots;             // rank 0's slot
    int * slot1 = (int *)((char *)dev_slots + SLOT_STRIDE);  // rank 1's slot

    // warmup
    for (int i = 0; i < 10; i++) {
        int tok = 1000 + i;
        hipSetDevice(0);
        ar_signal_kernel<<<1, 32, 0, s0>>>(slot0, slot1, tok, spin_iters, spin0);
        hipSetDevice(1);
        ar_signal_kernel<<<1, 32, 0, s1>>>(slot1, slot0, tok, spin_iters, spin1);
    }
    HK(hipStreamSynchronize(s0)); HK(hipStreamSynchronize(s1));

    // measured: N paired signals.  Rank0's spin = time for rank1's token to
    // become visible; rank1's spin = time for rank0's token.
    double sum0 = 0, sum1 = 0, min0 = 1e30, min1 = 1e30, max0 = 0, max1 = 0;
    long long host0[2], host1[2];
    std::vector<double> s0s, s1s; s0s.reserve(iters); s1s.reserve(iters);
    int rate_khz = 0; hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeClockRate, 0);

    for (int i = 0; i < iters; i++) {
        const int token = i + 1;   // strictly increasing; never 0
        hipSetDevice(0);
        ar_signal_kernel<<<1, 32, 0, s0>>>(slot0, slot1, token, spin_iters, spin0);
        hipSetDevice(1);
        ar_signal_kernel<<<1, 32, 0, s1>>>(slot1, slot0, token, spin_iters, spin1);
        // sync both to lockstep each iteration
        HK(hipStreamSynchronize(s0));
        HK(hipStreamSynchronize(s1));
        HK(hipSetDevice(0)); HK(hipMemcpy(host0, spin0, 2*sizeof(long long), hipMemcpyDeviceToHost));
        HK(hipSetDevice(1)); HK(hipMemcpy(host1, spin1, 2*sizeof(long long), hipMemcpyDeviceToHost));
        double d0 = (double)(host0[1]-host0[0]) * 1e3 / rate_khz;
        double d1 = (double)(host1[1]-host1[0]) * 1e3 / rate_khz;
        if (d0 > 0 && d1 > 0) { s0s.push_back(d0); s1s.push_back(d1); }
    }

    std::sort(s0s.begin(), s0s.end()); std::sort(s1s.begin(), s1s.end());
    auto pct = [&](std::vector<double>&v, double q){ return v[(size_t)(q*(v.size()-1))]; };
    printf("pair (HIP0=%d bus?, HIP1=%d bus?)  iters=%zu  clock=%d kHz\n", 0, 1, s0s.size(), rate_khz);
    printf("rank0->rank1 token visibility (rank1 spin):  p50 %.2f us  p90 %.2f us  max %.2f us\n",
           pct(s1s,0.50), pct(s1s,0.90), s1s.back());
    printf("rank1->rank0 token visibility (rank0 spin):  p50 %.2f us  p90 %.2f us  max %.2f us\n",
           pct(s0s,0.50), pct(s0s,0.90), s0s.back());
    printf("asymmetry: %.2f us\n", pct(s1s,0.50) - pct(s0s,0.50));
    return 0;
}
