// Prototype: 2-rank all-reduce over SDMA copy-engine P2P (no SM kernel on the transfer path).
// Both ranks are threads in one process, so cross-device ordering is done with hipStreamWaitEvent.
//   round 1 (reduce-scatter): rank0 sends [half,N); rank1 sends [0,half)  -> each reduces its own half
//   round 2 (all-gather):     each rank sends its reduced half back
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <thread>

using v2i = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using v8i = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__global__ void __launch_bounds__(256) gemm_like(v8f * out, int reps) {
    v2i a = {(int) threadIdx.x, (int) threadIdx.x + 1};
    v2i b = {(int) threadIdx.x + 2, (int) threadIdx.x + 3};
    v8i acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (v8i){0,0,0,0,0,0,0,0};
    for (int r = 0; r < reps; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[i], true);
    v8f o = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int l = 0; l < 8; ++l) o[l] += (float) acc[i][l];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = o;
}

__global__ void add_k(const int * __restrict__ a, const int * __restrict__ b, int * __restrict__ o, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        o[i] = a[i] + b[i];
}

struct Rank {
    int rank, peer;
    int *buf, *tmp, *out;
    int *peer_tmp, *peer_out;
    hipStream_t s;
    hipEvent_t ev_send, ev_gather;
    hipEvent_t peer_ev_send, peer_ev_gather;
    v8f * dummy; int * ok_flag;
};

static void run_rank(Rank * t, long N, int iters, int iters_max, double * ms_out, int * ok_out) {
    hipSetDevice(t->rank);
    const long half = N / 2;
    const long send_off   = (t->rank == 0) ? half : 0;   // round 1: opposite half
    const long gather_off = (t->rank == 0) ? 0 : half;   // round 2: own reduced half

    hipStreamCreateWithFlags(&t->s, hipStreamNonBlocking);
    hipEventCreateWithFlags(&t->ev_send,   hipEventDisableTiming);
    hipEventCreateWithFlags(&t->ev_gather, hipEventDisableTiming);

    auto step = [&](int it) {
        // round 1: hand the peer our opposite half
        hipError_t rc = hipMemcpyPeerAsync((char *) t->peer_tmp + send_off * 4, t->peer,
                           (char *) t->buf + send_off * 4, t->rank, (size_t) half * 4, t->s);
        if (rc != hipSuccess) { printf("memcpyPeer1 failed: %s\n", hipGetErrorString(rc)); }
        hipEventRecord(t->ev_send, t->s);
        hipStreamWaitEvent(t->s, t->peer_ev_send, 0);   // peer's round-1 send landed
        // reduce our own half
        if (t->rank == 0) add_k<<<256, 256, 0, t->s>>>(t->buf,       t->tmp,       t->out,       half);
        else              add_k<<<256, 256, 0, t->s>>>(t->buf + half, t->tmp + half, t->out + half, half);
        // round 2: send our reduced half back
        hipError_t rc2 = hipMemcpyPeerAsync((char *) t->peer_out + gather_off * 4, t->peer,
                           (char *) t->out + gather_off * 4, t->rank, (size_t) half * 4, t->s);
        if (rc2 != hipSuccess) { printf("memcpyPeer2 failed: %s\n", hipGetErrorString(rc2)); }
        hipEventRecord(t->ev_gather, t->s);
        hipStreamWaitEvent(t->s, t->peer_ev_gather, 0);
    };
    for (int i = 0; i < 3; ++i) step(i);          // warmup
    hipStreamSynchronize(t->s);

    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0, t->s);
    for (int i = 0; i < iters; ++i) step(i);
    hipEventRecord(e1, t->s);
    hipStreamSynchronize(t->s);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    *ms_out = (double) ms / iters;
    (void) iters_max;

    std::vector<int> h(N);
    hipMemcpy(h.data(), t->out, (size_t) N * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (long i = 0; i < N; ++i) if (h[i] != 2) { if (bad < 3) printf("  rank%d bad out[%ld]=%d\n", t->rank, i, h[i]); bad++; }
    *ok_out = (bad == 0);

    // ---- overlap phase: run the same AR loop concurrently with a WMMA-saturating kernel ----
    hipStream_t sc; hipStreamCreateWithFlags(&sc, hipStreamNonBlocking);
    const int reps = 8000;
    hipEvent_t c0, c1; hipEventCreate(&c0); hipEventCreate(&c1);
    hipEventRecord(c0, sc); gemm_like<<<4096, 256, 0, sc>>>(t->dummy, reps); hipEventRecord(c1, sc);
    hipStreamSynchronize(sc);
    float comp_ms = 0; hipEventElapsedTime(&comp_ms, c0, c1);

    hipStreamSynchronize(t->s);
    hipEventRecord(c0, sc);
    gemm_like<<<4096, 256, 0, sc>>>(t->dummy, reps);
    for (int i = 0; i < iters; ++i) step(i);
    hipEventRecord(c1, sc);
    hipStreamSynchronize(sc); hipStreamSynchronize(t->s);
    float both_ms = 0; hipEventElapsedTime(&both_ms, c0, c1);

    if (t->rank == 0) {
        printf("overlap: gemm-alone %.2f ms, gemm+AR %.2f ms  ->  gemm slowdown %.1f%% (AR %.2f ms x%d = %.2f ms)\n",
               comp_ms, both_ms, 100.0 * (both_ms - comp_ms) / comp_ms, *ms_out, iters, *ms_out * iters);
    }
}

int main(int argc, char ** argv) {
    int mb    = argc > 1 ? atoi(argv[1]) : 21;
    int iters = argc > 2 ? atoi(argv[2]) : 50;
    long N = ((long) mb << 20) / 4;

    Rank t[2];
    for (int r = 0; r < 2; ++r) {
        hipSetDevice(r);
        t[r].rank = r; t[r].peer = 1 - r;
        { hipError_t e = hipDeviceEnablePeerAccess(1 - r, 0);
          if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) printf("peer access %d->%d: %s\n", r, 1-r, hipGetErrorString(e)); }
        hipMalloc(&t[r].buf, (size_t) N * 4);
        hipMalloc(&t[r].tmp, (size_t) N * 4);
        hipMalloc(&t[r].out, (size_t) N * 4);
        std::vector<int> ones(N, 1);
        hipMemcpy(t[r].buf, ones.data(), (size_t) N * 4, hipMemcpyHostToDevice);
        hipMemset(t[r].tmp, 0, (size_t) N * 4);
        hipMemset(t[r].out, 0, (size_t) N * 4);
        hipMalloc(&t[r].dummy, sizeof(v8f));
    }
    // share peer events
    t[0].peer_ev_send = t[1].ev_send;   t[0].peer_ev_gather = t[1].ev_gather;
    t[1].peer_ev_send = t[0].ev_send;   t[1].peer_ev_gather = t[0].ev_gather;
    t[0].peer_tmp = t[1].tmp; t[0].peer_out = t[1].out;
    t[1].peer_tmp = t[0].tmp; t[1].peer_out = t[0].out;

    double ms[2] = {0,0}; int ok[2] = {0,0};
    std::thread a(run_rank, &t[0], N, iters, iters, &ms[0], &ok[0]);
    std::thread b(run_rank, &t[1], N, iters, iters, &ms[1], &ok[1]);
    a.join(); b.join();

    double t_ms = ms[0] > ms[1] ? ms[0] : ms[1];
    double bytes = (double) N * 4;
    printf("copy-engine P2P allreduce %d MB: %.3f ms  -> %.2f GB/s (N bytes/rank)  correct=%d/%d\n",
           mb, t_ms, bytes / (t_ms / 1e3) / 1e9, ok[0], ok[1]);
    printf("  NCCL reference on this box: ~2.08 ms for a 21 MB bf16 payload (128 calls/forward)\n");
    return 0;
}
