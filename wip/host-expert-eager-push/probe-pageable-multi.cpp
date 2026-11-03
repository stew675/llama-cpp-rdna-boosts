// probe-pageable-multi — does the driver's pageable-copy staging scale with device count?
//
// WHY (wip/host-expert-eager-push, 2026-11-03): the P1 switch (`MOE_HOST_PUSH_SRC=mapping`) DMAs straight
// from the read-only model mapping, which is PAGEABLE.  On one GPU it measured equal to the pool's
// memcpy-into-pinned + CE (P1-RESULTS.md).  On a `-sm tensor` split it is a large REGRESSION (35B-A3B
// Q4_K_M, -ncmoe 40, --host-experts pool: 2 GPUs 39.2 -> 13.8 t/s, 3 GPUs 19.0 -> 11.8), while `-sm layer`
// on the same 3 devices loses only 6 %.  The per-fill overhead implied by those runs grows with the number
// of devices ISSUING CONCURRENTLY from one host thread (28 us on 1 GPU, 204 us on 2, 302 us on 3 tensor,
// 41 us on 3 layer).
//
// This probe isolates the one mechanism that can explain that shape: a pageable source is staged through a
// shared internal bounce, and the pool's explicit memcpy is not.  It is NOT a llama.cpp test; it reproduces
// the issue pattern of one MoE op (each device issues its own run of ~0.6 MiB fills, on its own stream, from
// one host thread) for 1, 2 and 3 devices.
//
//   arm  pageable   hipMemcpyAsync straight from a malloc'd buffer        (the push candidate)
//   arm  pinned     hipMemcpyAsync straight from a hipHostMalloc'd buffer (the transport ceiling)
//   arm  pool       host memcpy -> per-device pinned slot, then CE from it (the delivered pool shape)
//
// Build (ROCm 10.1, gfx1201):
//   /opt/rocm-10.1.0-gfx120X/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1201 -O2 \
//       -I/opt/rocm-10.1.0-gfx120X/include -L/opt/rocm-10.1.0-gfx120X/lib -lamdhip64 \
//       -o probe-pageable-multi probe-pageable-multi.cpp
// Run:  HIP_VISIBLE_DEVICES=0,1,2 ./probe-pageable-multi [n_dev] [chunk_KiB] [chunks_per_round] [rounds]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

struct arm_t {
    const char * name;
    bool src_pinned;   // the CE source is hipHostMalloc'd (vs pageable malloc)
    bool host_memcpy;  // copy page cache -> per-device pinned slot first (the pool shape)
};

int main(int argc, char ** argv) {
    const int    ndev    = argc > 1 ? atoi(argv[1]) : 3;
    const size_t chunk   = (argc > 2 ? (size_t) atoi(argv[2]) : 600) << 10;   // ~0.6 MiB, a real expert fill
    const int    per_dev = argc > 3 ? atoi(argv[3]) : 64;                    // fills issued per device per op
    const int    rounds  = argc > 4 ? atoi(argv[4]) : 200;

    std::vector<hipStream_t> stream((size_t) ndev);
    std::vector<void *>      dst((size_t) ndev), slot((size_t) ndev);
    for (int d = 0; d < ndev; d++) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipStreamCreate(&stream[(size_t) d]));
        HIPCK(hipMalloc(&dst[(size_t) d], chunk * (size_t) per_dev));
        HIPCK(hipHostMalloc(&slot[(size_t) d], chunk * (size_t) per_dev));   // the pool's pinned ring
        HIPCK(hipMemsetAsync(dst[(size_t) d], 0, chunk * (size_t) per_dev, stream[(size_t) d]));
    }
    // ONE pageable source, touched so every page is resident: this is the model mapping's situation with a
    // warm page cache.  Every device reads the same bytes (they are slices of one host tensor).
    void * page = nullptr;
    const size_t page_bytes = chunk * (size_t) per_dev;
    if (posix_memalign(&page, 4096, page_bytes) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    memset(page, 1, page_bytes);
    // Same bytes in a pinned buffer, for the transport ceiling arm.
    void * pin_src = nullptr;
    HIPCK(hipHostMalloc(&pin_src, page_bytes));
    memcpy(pin_src, page, page_bytes);
    for (int d = 0; d < ndev; d++) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipDeviceSynchronize());
    }

    const arm_t arms[] = {
        { "pageable", false, false },
        { "pinned",   true,  false },
        { "pool",     true,  true  },
    };
    printf("probe-pageable-multi: %d device(s), %.0f KiB per fill, %d fills/device/round, %d rounds"
           " (%.1f MiB per round)\n",
           ndev, chunk / 1024.0, per_dev, rounds, (double) (chunk * (size_t) per_dev * (size_t) ndev) / 1048576.0);
    for (const arm_t & a : arms) {
        // warm-up round (faults the device pages / primes the staging path)
        for (int d = 0; d < ndev; d++) {
            HIPCK(hipSetDevice(d));
            if (a.host_memcpy) {
                memcpy(slot[(size_t) d], a.src_pinned ? pin_src : page, page_bytes);
            }
            HIPCK(hipMemcpyAsync(dst[(size_t) d], a.host_memcpy ? slot[(size_t) d] : (a.src_pinned ? pin_src : page),
                                 page_bytes, hipMemcpyHostToDevice, stream[(size_t) d]));
        }
        for (int d = 0; d < ndev; d++) {
            HIPCK(hipSetDevice(d));
            HIPCK(hipStreamSynchronize(stream[(size_t) d]));
        }
        const double t0 = now_ms();
        for (int r = 0; r < rounds; r++) {
            // One "op": each device issues its run of fills in turn, exactly as the Meta forward does.
            for (int d = 0; d < ndev; d++) {
                HIPCK(hipSetDevice(d));
                if (a.host_memcpy) {
                    // the pool: the host copies the expert out of the page cache into the pinned ring ONCE,
                    // and the CE reads the pinned ring.  One memcpy per fill, as `expert_host_src` does.
                    for (int c = 0; c < per_dev; c++) {
                        memcpy((char *) slot[(size_t) d] + (size_t) c * chunk,
                               (const char *) page + (size_t) c * chunk, chunk);
                    }
                    for (int c = 0; c < per_dev; c++) {
                        HIPCK(hipMemcpyAsync((char *) dst[(size_t) d] + (size_t) c * chunk,
                                             (const char *) slot[(size_t) d] + (size_t) c * chunk,
                                             chunk, hipMemcpyHostToDevice, stream[(size_t) d]));
                    }
                } else {
                    for (int c = 0; c < per_dev; c++) {
                        HIPCK(hipMemcpyAsync((char *) dst[(size_t) d] + (size_t) c * chunk,
                                             (const char *) (a.src_pinned ? pin_src : page) + (size_t) c * chunk,
                                             chunk, hipMemcpyHostToDevice, stream[(size_t) d]));
                    }
                }
            }
            for (int d = 0; d < ndev; d++) {
                HIPCK(hipSetDevice(d));
                HIPCK(hipStreamSynchronize(stream[(size_t) d]));
            }
        }
        const double ms     = now_ms() - t0;
        const double bytes  = (double) page_bytes * (double) ndev * (double) rounds;
        const double per_fill_us = 1000.0 * ms / ((double) per_dev * (double) ndev * (double) rounds);
        printf("  %-9s %8.2f ms   %7.2f GB/s   %7.1f us/fill\n", a.name, ms, bytes / 1e9 / (ms / 1000.0),
               per_fill_us);
    }
    return 0;
}
