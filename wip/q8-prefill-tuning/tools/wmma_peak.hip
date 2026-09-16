// Microbenchmark: peak throughput of the RDNA4 INT8 WMMA instruction
//   __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
// Each invocation is 16*16*16 = 4096 int8 MACs.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

#ifndef ITERS
#define ITERS 4096
#endif
#ifndef NACC
#define NACC 8
#endif

extern "C" __global__ void __launch_bounds__(256) peak_kernel(int32x8_t * out, int reps) {
    int32x2_t a = {(int)threadIdx.x, (int)threadIdx.x + 1};
    int32x2_t b = {(int)threadIdx.x + 2, (int)threadIdx.x + 3};
    int32x8_t acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = (int32x8_t){0,0,0,0,0,0,0,0};

    for (int r = 0; r < reps; ++r) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) {
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[i], true);
        }
    }
    int32x8_t s = (int32x8_t){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < NACC; ++i) {
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] += acc[i][l];
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = s;
}

int main(int argc, char ** argv) {
    int blocks = 4096;
    int threads = 256;
    int reps = ITERS;
    if (argc > 1) blocks = atoi(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);

    int32x8_t * dout = nullptr;
    hipMalloc(&dout, sizeof(int32x8_t));
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int cu = prop.multiProcessorCount;
    printf("device=%s CUs=%d blocks=%d threads=%d reps=%d NACC=%d\n",
           prop.name, cu, blocks, threads, reps, NACC);

    // warmup
    peak_kernel<<<blocks, threads>>>(dout, reps);
    hipDeviceSynchronize();

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    int rounds = 5;
    double best = 1e30;
    for (int i = 0; i < rounds; ++i) {
        hipEventRecord(e0);
        peak_kernel<<<blocks, threads>>>(dout, reps);
        hipEventRecord(e1);
        hipEventSynchronize(e1);
        float ms = 0;
        hipEventElapsedTime(&ms, e0, e1);
        double s = ms / 1e3;
        // MACs = blocks * warps_per_block * NACC * reps * 4096
        double warps = (double)blocks * (threads / 32);
        double macs = warps * NACC * (double)reps * 4096.0;
        double tops = macs / s / 1e12;
        if (s < best) best = s;
        printf("round %d: %.3f ms  -> %.1f T-MAC/s (%.1f TOPS)\n", i, ms, tops, tops * 2.0);
    }
    return 0;
}
