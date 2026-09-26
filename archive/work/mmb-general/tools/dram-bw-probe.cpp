// DRAM bandwidth probe (Strix Halo / gfx1151) -- establishes the ceiling that the
// dsv4_hc kernels are judged against, instead of inferring it from those kernels.
//
// Build:  hipcc --offload-arch=gfx1151 -O3 -o /tmp/dram-bw-probe dram-bw-probe.cpp
// Run:    LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib /tmp/dram-bw-probe
//
// Measured 2026-09-19 (gfx1151, Ryzen AI Max+ 395, ROCm 7.14, 34 C, no other GPU work):
//   pure sequential read, grid SWEPT (2048..16384)    234 -> 241.5 GB/s   91-94 % of 256
//   copy (read+write)                                 ~208-216 GB/s       81-84 % of 256
//   write-only                                        ~217 GB/s           85 % of 256
//   EXACT dsv4_hc_pre pattern, grid SWEPT (2048..32768) 221 -> 232.6 GB/s  86-91 % of 256
//
// **The grid size matters by ~5 %, so sweep it; single-point probe numbers understate the part.**
// A first pass used grid=4096 everywhere and reported 231.6 (read) / 227.2 (pattern); the same
// kernels at grid=16384 give 241.5 / 232.6.  Instruction sequence matters too: a 4-accumulator
// x4-unrolled read variant measured *worse* (221-230) than a simple single-accumulator grid-stride
// loop at the same grid, so "more ILP" is not automatically more bandwidth here.
//
// Against those numbers, the real `dsv4_hc_pre_f32` does 189 MB unique in 0.98 ms = **197 GB/s**:
// ~18 % below what its own access pattern achieves (232.6), and ~23 % below the best pure read
// (241.5).  A separate report puts this part at ~255 GB/s achievable, which would make the gap a
// little larger again -- either way the honest statement is "~18-23 % kernel-local headroom", NOT
// "already at the memory wall" (which is what inferring the ceiling from our own kernels produced).
//
// The bigger lever remains BYTES: bf16 HC intermediates take 189 -> ~105 MB (1.8x), ~2.3 % of prefill,
// and that is what matches the reference's 1.44 -> 0.93 s.
//
// Traps this probe cost, all worth keeping:
//   * size the buffers by BYTES and index by float4 count -- mixing them gives a 4x out-of-bounds
//     read and a GPU memory fault that looks like a driver problem;
//   * count write bytes as the iterations actually performed, not one whole buffer (counting
//     3x192 MB when 50 MB was written reported 302 GB/s on a 256 GB/s part -- exceeding spec is
//     the tell that the accounting is wrong);
//   * sweep the grid before quoting a ceiling.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){printf("HIP error: %s (%s:%d)\n",hipGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)

static void row(const char * n, double ms, double bytes) {
    const double bps = bytes/(ms/1e3);
    printf("  %-42s %8.3f ms  %7.1f GB/s  %5.1f%% of 256\n", n, ms, bps/1e9, 100.0*bps/1e9/256.0);
}

__global__ void k_read(const float4 * __restrict__ in, float * __restrict__ sink, size_t n4) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x*blockDim.x;
    float s = 0.f;
    for (; i < n4; i += st) { const float4 v = in[i]; s += v.x+v.y+v.z+v.w; }
    if (s == 12345.678f) sink[0] = s;   // never taken; defeats DCE
}

__global__ void k_copy(const float4 * __restrict__ in, float4 * __restrict__ out, size_t n4) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x*blockDim.x;
    for (; i < n4; i += st) out[i] = in[i];
}

__global__ void k_write(float4 * __restrict__ o, size_t n4) {
    size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
    const size_t st = (size_t)gridDim.x*blockDim.x;
    const float4 z = make_float4(1.f, 2.f, 3.f, 4.f);
    for (; i < n4; i += st) o[i] = z;
}

// the exact dsv4_hc_pre shape: x and gate are each [n_embd, hc, n_tokens] and a thread
// takes one (i0, it) pair, reading all hc streams of both and writing one dst element
__global__ void k_hcpre(const float4 * __restrict__ x, const float4 * __restrict__ g,
                        float4 * __restrict__ d, const long n_embd4, const long n_tok) {
    const long total = n_embd4*n_tok;
    long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
    const long st = (long)gridDim.x*blockDim.x;
    for (; i < total; i += st) {
        const long i0 = i % n_embd4, it = i / n_embd4;
        const float4 * bx = x + it*4*n_embd4 + i0;
        const float4 * bg = g + it*4*n_embd4 + i0;
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int ih = 0; ih < 4; ++ih) {
            const float4 a = bx[(long)ih*n_embd4], b = bg[(long)ih*n_embd4];
            s.x += a.x*b.x; s.y += a.y*b.y; s.z += a.z*b.z; s.w += a.w*b.w;
        }
        d[i] = s;
    }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t NB = 192ull*1024*1024;      // bytes
    const size_t n4 = NB/16;                 // float4 count -- NOT NB/4
    float4 *in = nullptr, *in2 = nullptr, *out = nullptr; float * sink = nullptr;
    CK(hipMalloc(&in, NB)); CK(hipMalloc(&in2, NB)); CK(hipMalloc(&out, NB)); CK(hipMalloc(&sink, 16));
    CK(hipMemset(in, 1, NB)); CK(hipMemset(in2, 1, NB));

    int d; CK(hipGetDevice(&d)); hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, d));
    printf("device %s | spec 256.0 GB/s (LPDDR5X-8000 x 256 bit)\n", p.name);

    const int B = 4096, T = 256;
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    auto timeit = [&](auto fn, double bytes, const char * nm) {
        fn(); CK(hipDeviceSynchronize());
        CK(hipEventRecord(a));
        for (int r = 0; r < 20; ++r) fn();
        CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
        row(nm, ms/20, bytes);
    };

    timeit([&]{ k_read<<<B,T>>>(in, sink, n4); }, (double) NB, "1. pure sequential read");
    timeit([&]{ k_copy<<<B,T>>>(in, out, n4); },  (double) NB*2, "2. copy (read + write)");
    timeit([&]{ k_write<<<B,T>>>(out, n4); },     (double) NB, "3. write-only");

    const long e4 = 640;                       // 640 float4 = 2560 floats = n_embd
    const long tok = (long)(n4/(4*e4));        // 4 streams per token
    const double rd = 2.0*(double)e4*4*16*(double)tok;   // x + gate
    const double wr = (double)e4*(double)tok*16;         // dst, one element per iteration
    printf("    (kernel 4: read %.0f MB + write %.0f MB = %.0f MB)\n", rd/1e6, wr/1e6, (rd+wr)/1e6);
    // sweep the grid: this is worth up to 5 % and is what made a first pass understate the part
    for (const int GB : { 2048, 4096, 8192, 16384, 32768 }) {
        char nm[96];
        snprintf(nm, sizeof nm, "4. EXACT dsv4_hc_pre pattern, grid=%d", GB);
        timeit([&]{ k_hcpre<<<GB,T>>>(in, in2, out, e4, tok); }, rd+wr, nm);
    }

    printf("\n  for reference: the real dsv4_hc_pre_f32 does 189 MB unique in 0.98 ms = 197 GB/s\n");
    printf("  best pattern rate above = the fair denominator; the gap to 197 GB/s is the headroom.\n");
    return 0;
}
