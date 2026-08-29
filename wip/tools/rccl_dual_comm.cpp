// Test: can ncclConfig_t.maxP2pPeers=0 force the SHM transport per-comm,
// while a default-config comm keeps P2P?  If yes, llama.cpp can hold both
// comms and dispatch by tensor size (P2P for prefill, SHM for decode).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <chrono>

#define CK(x) do { auto e = (x); if (e != ncclSuccess) { \
    fprintf(stderr, "RCCL FAIL %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)
#define HK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP FAIL %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static double bench(ncclComm_t * comms, hipStream_t * streams, void ** bufs, int ngpus, size_t nelems, int iters) {
    for (int i = 0; i < ngpus; i++)
        CK(ncclAllReduce(bufs[i], bufs[i], nelems, ncclFloat, ncclSum, comms[i], streams[i]));
    for (int i = 0; i < ngpus; i++) HK(hipStreamSynchronize(streams[i]));
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ngpus; i++)
            CK(ncclAllReduce(bufs[i], bufs[i], nelems, ncclFloat, ncclSum, comms[i], streams[i]));
        for (int i = 0; i < ngpus; i++) HK(hipStreamSynchronize(streams[i]));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s < best) best = s;
    }
    return best;
}

int main(int argc, char** argv) {
    int ngpus = argc > 1 ? atoi(argv[1]) : 2;
    int devcount = 0;
    HK(hipGetDeviceCount(&devcount));
    if (ngpus > devcount) { fprintf(stderr, "want %d GPUs, have %d\n", ngpus, devcount); return 1; }

    std::vector<ncclComm_t> comms_default(ngpus), comms_nop2p(ngpus);
    std::vector<void*> bufs(ngpus);
    std::vector<hipStream_t> streams(ngpus);
    std::vector<int> devs(ngpus);
    for (int i = 0; i < ngpus; i++) devs[i] = i;

    // comm set 1: default config (P2P as RCCL decides)
    CK(ncclCommInitAll(comms_default.data(), ngpus, devs.data()));

    // comm set 2: maxP2pPeers = 0 via ncclCommInitRankConfig
    ncclUniqueId id;
    CK(ncclGetUniqueId(&id));
    for (int i = 0; i < ngpus; i++) {
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        cfg.maxP2pPeers = 0;
        HK(hipSetDevice(i));
        CK(ncclCommInitRankConfig(&comms_nop2p[i], ngpus, id, i, &cfg));
        HK(hipMalloc(&bufs[i], 512ULL << 20));
        HK(hipStreamCreate(&streams[i]));
        HK(hipMemset(bufs[i], 0, 512ULL << 20)); // touch memory
    }

    printf("GPU count: %d\n", ngpus);
    printf("test                | default (P2P?)     | maxP2pPeers=0      | ratio\n");
    printf("--------------------+--------------------+--------------------+------\n");
    struct { const char* name; size_t ne; int iters; } cases[] = {
        {"allreduce 64 KB  ", 16384, 200},
        {"allreduce 1 MB   ", 262144, 100},
        {"allreduce 128 MB ", 33554432, 16},
        {"allreduce 512 MB ", 134217728, 4},
    };
    for (auto & c : cases) {
        double t_def = bench(comms_default.data(), streams.data(), bufs.data(), ngpus, c.ne, c.iters);
        double t_np  = bench(comms_nop2p.data(), streams.data(), bufs.data(), ngpus, c.ne, c.iters);
        // per-rank bidirectional GB/s: 2 * bytes / time
        double bw_def = 2.0 * c.ne * 4 / t_def / 1e9;
        double bw_np  = 2.0 * c.ne * 4 / t_np  / 1e9;
        printf("%s | %11.1f GB/s (%6.0f us) | %12.1f GB/s (%6.0f us) | %4.2fx\n",
               c.name, bw_def, t_def*1e6, bw_np, t_np*1e6, bw_def/bw_np);
    }

    for (int i = 0; i < ngpus; i++) {
        HK(hipSetDevice(i));
        HK(hipFree(bufs[i])); HK(hipStreamDestroy(streams[i]));
        CK(ncclCommDestroy(comms_default[i]));
        CK(ncclCommDestroy(comms_nop2p[i]));
    }
    return 0;
}
