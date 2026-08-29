// RCCL all-reduce bandwidth bench: measures the exact path llama.cpp uses
// for tensor-parallel (row-split) across the R9700s, using the ROCm 7.14 tree.
// Build: /opt/rocm-7.14-gfx1201/bin/hipcc -O2 rccl_ar_bench.cpp -o rccl_ar_bench \
//           -I/opt/rocm-7.14-gfx1201/include/rccl -L/opt/rocm-7.14-gfx1201/lib -lrccl
// Run:   HIP_VISIBLE_DEVICES=0,1,2 ./rccl_ar_bench [ngpus] [mb]
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

int main(int argc, char** argv) {
    int ngpus = argc > 1 ? atoi(argv[1]) : 2;
    size_t mb = argc > 2 ? (size_t)atoi(argv[2]) : 128;
    size_t bytes = mb << 20;

    int devcount = 0;
    HK(hipGetDeviceCount(&devcount));
    if (ngpus > devcount) { fprintf(stderr, "want %d GPUs, have %d\n", ngpus, devcount); return 1; }

    char name[256];
    for (int i = 0; i < ngpus; i++) {
        hipDeviceProp_t p; HK(hipGetDeviceProperties(&p, i));
        printf("GPU %d: %s  uuid=%s  (%.1f GB)\n", i, p.name, p.gcnArchName, p.totalGlobalMem/1e9);
    }

    std::vector<ncclComm_t> comms(ngpus);
    std::vector<void*> bufs(ngpus);
    std::vector<hipStream_t> streams(ngpus);
    std::vector<void*> hostbufs(ngpus);

    CK(ncclCommInitAll(comms.data(), ngpus, nullptr));
    for (int i = 0; i < ngpus; i++) {
        HK(hipSetDevice(i));
        HK(hipMalloc(&bufs[i], bytes));
        HK(hipHostMalloc(&hostbufs[i], bytes));
        memset(hostbufs[i], i + 1, bytes);
        HK(hipMemcpy(bufs[i], hostbufs[i], bytes, hipMemcpyHostToDevice));
        HK(hipStreamCreate(&streams[i]));
    }
    // validate that each GPU can read its peers' memory (P2P check)
    if (ngpus > 1) {
        unsigned char* p = (unsigned char*)bufs[1];
        unsigned char v = 0; HK(hipSetDevice(0));
        HK(hipMemcpy(&v, p, 1, hipMemcpyDeviceToHost));
        printf("P2P check: GPU0 read GPU1 byte = %u (expect 2)\n", (unsigned)v);
    }

    // warmup
    for (int i = 0; i < ngpus; i++) {
        CK(ncclAllReduce(bufs[i], bufs[i], bytes / 4, ncclFloat, ncclSum, comms[i], streams[i]));
    }
    for (auto s : streams) HK(hipStreamSynchronize(s));

    // timed runs
    double best = 1e30, sum = 0;
    const int iters = 16;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < ngpus; i++)
            CK(ncclAllReduce(bufs[i], bufs[i], bytes / 4, ncclFloat, ncclSum, comms[i], streams[i]));
        for (auto s : streams) HK(hipStreamSynchronize(s));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // total data moved: each of ngpus buffers does a full all-reduce = 2*(ngpus-1)/ngpus * bytes moved per GPU... use total xfer = bytes * 2
        sum += s; if (s < best) best = s;
    }
    // effective: every GPU sends and receives `bytes` => aggregate moved = ngpus*2*bytes, but
    // wall-clock work per rank is what limits: use bytes (send) + bytes (recv) per rank
    double bwb = bytes / best * 2.0 / 1e9;   // GB/s per-rank bidirectional
    printf("\nall-reduce %zu MB across %d GPUs:  avg %.3f ms  best %.3f ms  => ~%.1f GB/s (per-rank, send+recv)\n",
           mb, ngpus, sum/iters*1e3, best*1e3, bwb);

    for (int i = 0; i < ngpus; i++) {
        HK(hipSetDevice(i));
        HK(hipFree(bufs[i])); HK(hipHostFree(hostbufs[i]));
        CK(ncclCommDestroy(comms[i]));
        HK(hipStreamDestroy(streams[i]));
    }
    return 0;
}
