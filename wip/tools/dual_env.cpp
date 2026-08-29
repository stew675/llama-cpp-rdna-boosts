// Proof-of-concept: two comms in one process, one P2P (default env) and one
// SHM (NCCL_P2P_DISABLE=1 set around its init). If RCCL honors the env at
// comm-init time, the SHM comm must show ~half bandwidth at large sizes and
// the NCCL_DEBUG channel lines must say "via SHM" instead of "via P2P".
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include <rccl.h>
#include <chrono>
#define CK(x) do { auto e = (x); if (e != ncclSuccess) { \
    fprintf(stderr, "RCCL FAIL %d: %s\n", __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)
#define HK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP FAIL %d: %s\n", __LINE__, hipGetErrorString(e)); exit(1); } } while (0)

static double bench(ncclComm_t * comms, hipStream_t * streams, void ** bufs, int ngpus, size_t ne, int iters) {
    for (int i = 0; i < ngpus; i++) CK(ncclAllReduce(bufs[i], bufs[i], ne, ncclFloat, ncclSum, comms[i], streams[i]));
    for (int i = 0; i < ngpus; i++) HK(hipStreamSynchronize(streams[i]));
    double best = 1e30;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ngpus; i++) CK(ncclAllReduce(bufs[i], bufs[i], ne, ncclFloat, ncclSum, comms[i], streams[i]));
        for (int i = 0; i < ngpus; i++) HK(hipStreamSynchronize(streams[i]));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s < best) best = s;
    }
    return best;
}

int main() {
    const int ngpus = 2;
    std::vector<ncclComm_t> comms_p2p(ngpus), comms_shm(ngpus);
    std::vector<void*> bufs(ngpus);
    std::vector<hipStream_t> streams(ngpus);
    int devs[2] = {0, 1};

    // comm set 1: default env -> P2P expected
    CK(ncclCommInitAll(comms_p2p.data(), ngpus, devs));

    // comm set 2: NCCL_P2P_DISABLE=1 set around init -> SHM expected
    setenv("NCCL_P2P_DISABLE", "1", 1);
    ncclUniqueId id; CK(ncclGetUniqueId(&id));
    ncclGroupStart();
    for (int i = 0; i < ngpus; i++) {
        hipSetDevice(i);
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        CK(ncclCommInitRankConfig(&comms_shm[i], ngpus, id, i, &cfg));
    }
    CK(ncclGroupEnd());
    unsetenv("NCCL_P2P_DISABLE");

    for (int i = 0; i < ngpus; i++) {
        HK(hipSetDevice(i));
        HK(hipMalloc(&bufs[i], 256ULL << 20));
        HK(hipMemset(bufs[i], 0, 256ULL << 20));
        HK(hipStreamCreate(&streams[i]));
    }

    printf("transport test (2 GPUs): P2P comm vs SHM comm\n");
    struct { const char* n; size_t ne; int it; } cs[] = {
        {"64 KB ", 16384, 200}, {"1 MB  ", 262144, 100}, {"128 MB", 33554432, 16}};
    for (auto & c : cs) {
        double tp = bench(comms_p2p.data(), streams.data(), bufs.data(), ngpus, c.ne, c.it);
        double ts = bench(comms_shm.data(), streams.data(), bufs.data(), ngpus, c.ne, c.it);
        printf("%s | P2P %7.1f GB/s | SHM %7.1f GB/s | ratio %.2fx\n",
               c.n, 2.0*c.ne*4/tp/1e9, 2.0*c.ne*4/ts/1e9, tp/ts);
    }
    return 0;
}
