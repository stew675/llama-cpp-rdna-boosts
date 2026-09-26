// Isolate the Q8_0 per-block-scale epilogue cost vs a "straight-copy" FP8-style inner loop.
// Both use the same RDNA4 16x16x16 WMMA; EPI adds the 8 FMA + scale loads MMQ does per 2 WMMA.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

using v2i = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using v8i = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

#define NACC 8

template <bool EPI>
__global__ void __launch_bounds__(256) peak(v8f * out, int reps) {
    extern __shared__ float sscale[];          // per-lane dA / dB stand-ins (LDS-resident)
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) sscale[i] = 1.0f + (i & 7) * 0.001f;
    __syncthreads();

    v2i a = {(int) threadIdx.x, (int) threadIdx.x + 1};
    v2i b = {(int) threadIdx.x + 2, (int) threadIdx.x + 3};
    v8i acc[NACC];
    v8f sum[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) { acc[i] = (v8i){0,0,0,0,0,0,0,0}; sum[i] = (v8f){0,0,0,0,0,0,0,0}; }

    for (int r = 0; r < reps; ++r) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) {
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[i], true);
            if (EPI) {
                const float dB = sscale[(r + i) & 31];
#pragma unroll
                for (int l = 0; l < 4; ++l) {                  // MMQ does 8 FMA per 2 WMMA
                    const float dA = sscale[((threadIdx.x + l) & 31) + ((i & 1) << 5)];
                    sum[i][l] += (float) acc[i][l] * dA * dB;   // the Q8_0 epilogue
                }
            }
        }
    }
    v8f o = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) o[l] += (float) acc[i][l] + sum[i][l];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = o;
}

template <bool EPI>
static double bench(v8f * d, int blocks, int threads, int reps) {
    peak<EPI><<<blocks, threads, 64 * 32 * sizeof(float)>>>(d, reps);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    peak<EPI><<<blocks, threads, 64 * 32 * sizeof(float)>>>(d, reps);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    double s = ms / 1e3;
    double warps = (double) blocks * (threads / 32);
    double macs = warps * NACC * (double) reps * 4096.0;
    return macs / s / 1e12;
}

int main(int argc, char ** argv) {
    int blocks = argc > 1 ? atoi(argv[1]) : 4096;
    int reps   = argc > 2 ? atoi(argv[2]) : 20000;
    v8f * d; hipMalloc(&d, sizeof(v8f));
    double plain = bench<false>(d, blocks, 256, reps);
    double epi   = bench<true >(d, blocks, 256, reps);
    printf("int8 WMMA plain         : %6.1f T-MAC/s\n", plain);
    printf("int8 WMMA +Q8_0 epilogue: %6.1f T-MAC/s\n", epi);
    printf("epilogue cost           : %5.1f %% of the plain rate\n", 100.0 * (1.0 - epi / plain));
    return 0;
}
