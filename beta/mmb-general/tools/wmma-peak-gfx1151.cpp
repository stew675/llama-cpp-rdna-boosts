// Microbenchmark: gfx11 (RDNA3.5 / gfx1151) WMMA peak for the two MMB instruction choices.
//
// The MMB dense Q8_0 kernel dequantizes Q8_0 -> bf16 then issues bf16 WMMA
// (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32).  The section-9 idea is to skip the
// dequant and feed int8 straight to the int8 tensor core
// (__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32) with a Q8_0 per-32-block scale epilogue.
// This measures the ceiling of each on the actual target part.
//
// build: hipcc --offload-arch=gfx1151 -O3 -o /tmp/wmma-peak wmma-peak-gfx1151.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef short  v16s __attribute__((ext_vector_type(16)));
typedef int    v4i  __attribute__((ext_vector_type(4)));
typedef int    v8i  __attribute__((ext_vector_type(8)));
typedef float  v8f  __attribute__((ext_vector_type(8)));

#ifndef NACC
#define NACC 8
#endif

__device__ __forceinline__ v8f wbf16(v16s a, v16s b, v8f c) {
#if defined(__gfx11__) || !defined(RDNA4)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    (void)a; (void)b; return c;
#endif
}

template <bool EPI>
__global__ void __launch_bounds__(256) peak_bf16(v8f * out, int reps) {
    extern __shared__ float ss[];
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) ss[i] = 1.0f + (i & 7) * 0.001f;
    __syncthreads();
    v16s a, b;
#pragma unroll
    for (int i = 0; i < 16; ++i) { a[i] = (short)(threadIdx.x + i); b[i] = (short)(threadIdx.x - i); }
    v8f acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = (v8f){0,0,0,0,0,0,0,0};
    for (int r = 0; r < reps; ++r) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) {
            acc[i] = wbf16(a, b, acc[i]);
            if (EPI) {
                const float dB = ss[(r + i) & 31];
#pragma unroll
                for (int l = 0; l < 8; ++l) { const float dA = ss[((threadIdx.x + l) & 31) + ((i & 1) << 5)]; acc[i][l] *= dA * dB; }
            }
        }
    }
    v8f o = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) o[l] += acc[i][l];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = o;
}

template <bool EPI>
__global__ void __launch_bounds__(256) peak_iu8(v8f * out, int reps) {
    extern __shared__ float ss[];
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) ss[i] = 1.0f + (i & 7) * 0.001f;
    __syncthreads();
    v4i a = {(int)threadIdx.x, (int)threadIdx.x + 1, (int)threadIdx.x + 2, (int)threadIdx.x + 3};
    v4i b = {(int)threadIdx.x + 4, (int)threadIdx.x + 5, (int)threadIdx.x + 6, (int)threadIdx.x + 7};
    v8i acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = (v8i){0,0,0,0,0,0,0,0};
    for (int r = 0; r < reps; ++r) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) {
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[i], true);
            if (EPI) {
                const float dB = ss[(r + i) & 31];
#pragma unroll
                for (int l = 0; l < 8; ++l) { const float dA = ss[((threadIdx.x + l) & 31) + ((i & 1) << 5)]; acc[i][l] = (int)((float)acc[i][l] * dA * dB); }
            }
        }
    }
    v8i s = (v8i){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] += acc[i][l];
    v8f o = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int l = 0; l < 8; ++l) o[l] = (float)s[l];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = o;
}

template <typename K>
static double bench(K kern, v8f * d, int blocks, int reps) {
    kern<<<blocks, 256, 64 * 32 * sizeof(float)>>>(d, reps);
    hipDeviceSynchronize();
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    double best = 1e30;
    for (int i = 0; i < 3; ++i) {
        hipEventRecord(e0);
        kern<<<blocks, 256, 64 * 32 * sizeof(float)>>>(d, reps);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1);
        if (ms / 1e3 < best) best = ms / 1e3;
    }
    const double warps = (double)blocks * 8.0;
    const double macs  = warps * NACC * (double)reps * 4096.0;
    return macs / best / 1e12;
}

int main(int argc, char ** argv) {
    int blocks = argc > 1 ? atoi(argv[1]) : 2048;
    int reps   = argc > 2 ? atoi(argv[2]) : 20000;
    v8f * d; hipMalloc(&d, sizeof(v8f));
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("device=%s CUs=%d blocks=%d reps=%d NACC=%d\n", prop.name, prop.multiProcessorCount, blocks, reps, NACC);
    printf("bf16 WMMA plain          : %6.1f T-MAC/s\n", bench(peak_bf16<false>, d, blocks, reps));
    printf("int8 WMMA plain          : %6.1f T-MAC/s\n", bench(peak_iu8 <false>, d, blocks, reps));
    printf("bf16 WMMA +Q8_0 epilogue : %6.1f T-MAC/s\n", bench(peak_bf16<true >, d, blocks, reps));
    printf("int8 WMMA +Q8_0 epilogue : %6.1f T-MAC/s\n", bench(peak_iu8 <true >, d, blocks, reps));
    return 0;
}
