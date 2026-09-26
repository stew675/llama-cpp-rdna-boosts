// Why doesn't the token-chunk pipeline overlap on this box?
//
// overlap.hip proved that a bare `hipMemcpyPeerAsync` (SDMA) hides 100% behind WMMA on a second
// stream.  The real all-reduce (archive/work/q8-prefill-tuning/tools/ce-allreduce.patch) does NOT overlap in
// llama.cpp even with every host-side block removed and no cross-chunk dependencies.  The two
// structural differences from overlap.hip are:
//
//   (1) the AR phases are chained by CROSS-DEVICE stream-wait events
//       (device 0's stream waits on an event recorded on device 1's stream, and vice versa), and
//   (2) two chains (the two chunk parities) SHARE one set of AR scratch buffers and events.
//
// This program models both and measures whether the AR hides behind compute.
//
// Two "chains" = the two chunk parities.  Each chain runs on BOTH devices (tensor-parallel), on its
// own stream per device (stream 1 for chain A, stream 2 for chain B), exactly like
// ggml_backend_cuda_comm_set_stream_no(parity+1) in the real thing.
//
// Per chain, NSTEPS times:  [GEMM on both devices] then [peer copy with a cross-device event wait].
//
// Reported:
//   t_gemm1  = 1 chain, GEMM only            (no AR at all)
//   t_full1  = 1 chain, GEMM + AR
//   t_full2s = 2 concurrent chains, GEMM + AR, SEPARATE scratch (independent)
//   t_full2x = 2 concurrent chains, GEMM + AR, SHARED   scratch (like the real CE code)
//
// Interpretation: the AR is hidden iff t_full2* ~= 2*t_gemm1 (SMs shared, DMA free).
//                 the AR is exposed if t_full2* ~= 2*t_full1.
//
// usage: overlap_ar [mb] [reps] [nsteps]
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

__global__ void addk(int * d, const int * s, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        d[i] += s[i];
}

// bandwidth-bound stand-in for MMQ: sweep a large array (reads streaming from HBM).
// The real workload's MUL_MAT_Q streams quantized weights from HBM, so the prefill compute is
// memory-bandwidth-bound, NOT tensor-unit-bound.  This kernel reproduces that property.
__global__ void bw_sweep(const int * __restrict__ d, long n, int reps, int * out) {
    int acc = 0;
    for (int r = 0; r < reps; ++r) {
        for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
            acc += d[i];
    }
    if (acc == 0x7fffffff && threadIdx.x == 0) out[0] = acc;
}

// one set of AR scratch + events (per chain, or shared between the two chains)
struct ar {
    int *       tmp1[2];   // per device: receive region for the phase-1 slice
    int *       tmp2[2];
    hipEvent_t  ev[2];     // per device: cross-device dependency event
    int         n_int;     // elements per rank's buffer
};

static void ar_init(ar & a, int n_int) {
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipMalloc(&a.tmp1[d], (size_t) n_int * sizeof(int));
        hipMalloc(&a.tmp2[d], (size_t) n_int * sizeof(int));
        hipEventCreateWithFlags(&a.ev[d], hipEventDisableTiming);
    }
    a.n_int = n_int;
}

// one "all-reduce step": two peer copies with a cross-device event wait between them, on the
// per-device streams held in `streams[]` (the chain's parity streams).
static void ar_step(ar & a, int * buf[2], hipStream_t streams[2], int chunk) {
    const size_t bytes = (size_t) chunk * sizeof(int);
    const size_t off   = (size_t) chunk;   // rank 1's half starts at element `chunk`

    // phase 1: each rank ships the OTHER rank's half to the peer, after waiting on the peer's event
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipStream_t s = streams[d];
        const int   p = 1 - d;
        hipStreamWaitEvent(s, a.ev[p], 0);
        // rank d owns half d; ship half (1-d) of my buffer to peer p's tmp1
        hipMemcpyPeerAsync(a.tmp1[p], p,
                           (char *) buf[d] + (d == 0 ? off : 0) * sizeof(int), d,
                           bytes, s);
        hipEventRecord(a.ev[d], s);
    }

    // phase 2: read the peer's slice out of my own tmp2 (second cross-device dependency)
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipStream_t s = streams[d];
        const int   p = 1 - d;
        hipStreamWaitEvent(s, a.ev[p], 0);
        addk<<<64, 256, 0, s>>>(buf[d] + (d == 0 ? 0 : off), a.tmp1[d], chunk);
        hipMemcpyPeerAsync(a.tmp2[p], p,
                           (char *) buf[d] + (d == 0 ? 0 : off) * sizeof(int), d,
                           bytes, s);
        hipEventRecord(a.ev[d], s);
    }
}

static double run(bool two_chains, int reps, int nsteps, ar & scratch_a, ar & scratch_b,
                  int * buf[2], int * buf2[2], hipStream_t s1[2], hipStream_t s2[2], bool with_ar,
                  int kind, const int * sweep, long sweep_n) {
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipStreamSynchronize(s1[d]);
        if (two_chains) hipStreamSynchronize(s2[d]);
    }
    auto do_compute = [&](hipStream_t st, int d) {
        if (kind == 0) {
            compute<<<2048, 256, 0, st>>>((v8f *) buf[d], reps);
        } else {
            bw_sweep<<<1024, 256, 0, st>>>(sweep, sweep_n, reps, (int *) buf[d]);
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nsteps; ++i) {
        for (int d = 0; d < 2; ++d) {
            hipSetDevice(d);
            do_compute(s1[d], d);
        }
        if (with_ar) ar_step(scratch_a, buf, s1, 0);

        if (two_chains) {
            for (int d = 0; d < 2; ++d) {
                hipSetDevice(d);
                do_compute(s2[d], d);
            }
            if (with_ar) ar_step(scratch_b, buf2, s2, 0);
        }
    }
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipStreamSynchronize(s1[d]);
        if (two_chains) hipStreamSynchronize(s2[d]);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char ** argv) {
    int mb     = argc > 1 ? atoi(argv[1]) : 21;
    int reps   = argc > 2 ? atoi(argv[2]) : 6000;
    int nsteps = argc > 3 ? atoi(argv[3]) : 40;
    int kind   = argc > 4 ? atoi(argv[4]) : 0;   // 0 = WMMA (compute-bound), 1 = HBM sweep (bandwidth-bound)

    int can = 0;
    hipDeviceCanAccessPeer(&can, 0, 1);
    hipSetDevice(0); hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1); hipDeviceEnablePeerAccess(0, 0);
    if (!can) { printf("no peer access\n"); return 1; }

    const long n_int = ((long) mb << 20) / sizeof(int);

    int * buf[2]  = {nullptr, nullptr};
    int * buf2[2] = {nullptr, nullptr};
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipMalloc(&buf[d],  (size_t) n_int * sizeof(int));
        hipMalloc(&buf2[d], (size_t) n_int * sizeof(int));
        hipMemset(buf[d],  1, (size_t) n_int * sizeof(int));
        hipMemset(buf2[d], 1, (size_t) n_int * sizeof(int));
    }

    hipStream_t s1[2], s2[2];
    for (int d = 0; d < 2; ++d) {
        hipSetDevice(d);
        hipStreamCreateWithFlags(&s1[d], hipStreamNonBlocking);
        hipStreamCreateWithFlags(&s2[d], hipStreamNonBlocking);
    }

    // a large array to sweep for the bandwidth-bound compute model (256 MB per device)
    const long sweep_n = 256L << 20;
    int * sweep = nullptr;
    hipSetDevice(0);
    hipMalloc(&sweep, (size_t) sweep_n);
    hipMemset(sweep, 3, (size_t) sweep_n);

    // every scratch set owns its own per-rank chunk size
    const int chunk = (int) (n_int / 2);

    ar sa;   ar_init(sa,     n_int);
    ar sb_sep; ar_init(sb_sep, n_int);   // separate scratch: fully independent chains
    ar sb_shr = sa;                      // shares sa's buffers/events: like the real CE code

    printf("payload=%d MB (chunk %d MB/rank)  reps=%d  steps=%d  peer=%d\n",
           mb, (int) ((size_t) chunk * sizeof(int) >> 20), reps, nsteps, can);

    const double t_gemm1 = run(false, reps, nsteps, sa, sb_sep, buf, buf2, s1, s2, false, kind, sweep, sweep_n / sizeof(int));
    const double t_full1 = run(false, reps, nsteps, sa, sb_sep, buf, buf2, s1, s2, true,  kind, sweep, sweep_n / sizeof(int));
    const double t_sep2  = run(true,  reps, nsteps, sa, sb_sep, buf, buf2, s1, s2, true,  kind, sweep, sweep_n / sizeof(int));
    const double t_shr2  = run(true,  reps, nsteps, sa, sb_shr, buf, buf2, s1, s2, true,  kind, sweep, sweep_n / sizeof(int));

    const double ar_cost = t_full1 - t_gemm1;

    printf("\n");
    printf("1 chain, GEMM only        : %8.2f ms\n", t_gemm1);
    printf("1 chain, GEMM + AR        : %8.2f ms   (AR cost %.2f ms)\n", t_full1, ar_cost);
    printf("2 chains, AR SEPARATEscr  : %8.2f ms\n", t_sep2);
    printf("2 chains, AR SHARED scratch: %8.2f ms\n", t_shr2);
    printf("\n");
    printf("ideal (AR fully hidden, SMs shared) = 2*%0.2f = %8.2f ms\n", t_gemm1, 2.0 * t_gemm1);
    printf("worst (AR fully exposed)            = 2*%0.2f = %8.2f ms\n", t_full1, 2.0 * t_full1);
    printf("\n");
    printf("SEPARATE scratch: %6.1f%% of the AR hidden   (0%% = fully exposed, 100%% = free)\n",
           100.0 * (2.0 * t_full1 - t_sep2) / (2.0 * ar_cost));
    printf("SHARED   scratch: %6.1f%% of the AR hidden\n",
           100.0 * (2.0 * t_full1 - t_shr2) / (2.0 * ar_cost));
    return 0;
}
