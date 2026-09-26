// Can the tensor-parallel all-reduce traffic hide behind compute on this box?
//  t_comp  = compute alone        (WMMA loop)
//  t_xfer  = N peer transfers alone
//  t_both  = both concurrently, wall-clocked
//  hidden  = 1 - (t_both - t_comp)/t_xfer   (1.0 = fully hidden, 0.0 = fully serial)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

using v2i = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using v8i = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
#define NACC 8

__global__ void __launch_bounds__(256) compute(v8f * out, int reps) {
    v2i a = {(int) threadIdx.x, (int) threadIdx.x + 1};
    v2i b = {(int) threadIdx.x + 2, (int) threadIdx.x + 3};
    v8i acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = (v8i){0,0,0,0,0,0,0,0};
    for (int r = 0; r < reps; ++r)
#pragma unroll
        for (int i = 0; i < NACC; ++i)
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[i], true);
    v8f o = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) o[l] += (float) acc[i][l];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = o;
}

__global__ void sm_xfer(const int * __restrict__ src, int * __restrict__ dst, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

static double wall_ms(const std::chrono::steady_clock::time_point & a,
                      const std::chrono::steady_clock::time_point & b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char ** argv) {
    int mb     = argc > 1 ? atoi(argv[1]) : 21;
    int reps   = argc > 2 ? atoi(argv[2]) : 20000;
    int nxfers = argc > 3 ? atoi(argv[3]) : 80;
    int sm_grid = argc > 4 ? atoi(argv[4]) : 1024;
    long n_int = ((long) mb << 20) / sizeof(int);

    hipSetDevice(0);
    v8f * dc; hipMalloc(&dc, sizeof(v8f));
    int * src; hipMalloc(&src, n_int * sizeof(int)); hipMemset(src, 1, n_int * sizeof(int));
    int can = 0; hipDeviceCanAccessPeer(&can, 0, 1); hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1); int * dst; hipMalloc(&dst, n_int * sizeof(int));
    hipSetDevice(0);
    printf("peer=%d  payload=%d MB  xfers=%d  reps=%d\n", can, mb, nxfers, reps);

    hipStream_t sA, sB;
    hipStreamCreateWithFlags(&sA, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&sB, hipStreamNonBlocking);

    for (int kind = 0; kind < 2; ++kind) {   // 0 = copy-engine, 1 = SM-driven
        auto xfer = [&]() {
            if (kind == 0) hipMemcpyPeerAsync(dst, 1, src, 0, (size_t) n_int * sizeof(int), sB);
            else           sm_xfer<<<sm_grid, 256, 0, sB>>>(src, dst, n_int);
        };

        // compute alone
        hipDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        compute<<<4096, 256, 0, sA>>>(dc, reps);
        hipStreamSynchronize(sA);
        auto t1 = std::chrono::steady_clock::now();
        double t_comp = wall_ms(t0, t1);

        // xfer alone
        hipDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nxfers; ++i) xfer();
        hipStreamSynchronize(sB);
        t1 = std::chrono::steady_clock::now();
        double t_xfer = wall_ms(t0, t1);

        // both
        hipDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        compute<<<4096, 256, 0, sA>>>(dc, reps);
        for (int i = 0; i < nxfers; ++i) xfer();
        hipStreamSynchronize(sA); hipStreamSynchronize(sB);
        t1 = std::chrono::steady_clock::now();
        double t_both = wall_ms(t0, t1);

        const char * names[2] = {"copy-engine (SDMA)", "SM-driven (NCCL-like)"};
        double hidden = 1.0 - (t_both - t_comp) / t_xfer;
        printf("%-21s: comp %7.2f  xfer %7.2f  both %7.2f ms  | hidden %5.1f%%  (comp slowdown %4.1f%%)\n",
               names[kind], t_comp, t_xfer, t_both, 100.0 * hidden, 100.0 * (t_both - t_comp) / t_comp);
    }
    return 0;
}
