// probe-pageable-threads — is the pageable staging PER THREAD, and is SHARING A STREAM between two
// enqueueing threads safe?  (P3: the real path wedged twice; the probe never did.  The one thing the real
// path does that this probe did not is let TWO threads enqueue on the SAME stream.)
//
// WHY (wip/host-expert-eager-push, P3): pushing straight from the pageable model mapping is at parity on ONE
// device (55.1 vs 53.6 t/s) and on a 3-device `-sm layer` split (55.6 vs 53.7), but a 2-device `-sm tensor`
// split costs 52.4 -> 26.0 t/s.  Instrumentation (`MOE_HOST_PUSH_PROBE=1`) puts the whole loss inside the
// copy call: 44 / 107 / 191 us per H2D copy at 1 / 2 / 3 devices (a pinned or D2D copy from the same thread
// is 1.4 us), i.e. aggregate staging throughput FALLS from ~12 to ~3 GB/s as concurrent devices rise.
// `MOE_HOST_PUSH_SETDEV=1` (make the copying backend's device current) changes nothing.
//
// That scaling is the signature of a shared staging resource.  If it is per PROCESS, one thread issuing for
// every device serialises and no amount of batching helps; the answer is our own small pinned ring (tens of
// MiB, not GiB).  If it is per THREAD, then ONE UPLOAD THREAD PER DEVICE restores the throughput with no
// pinned duplicate at all -- which is exactly the design's P6 rule ("one push queue per device").
//
//   build:
//     clang++ -x hip --offload-arch=gfx1201 -O2 -I/opt/rocm-10.1.0-gfx120X/include \
//       -L/opt/rocm-10.1.0-gfx120X/lib -lamdhip64 -pthread -o /tmp/probe-pageable-threads probe-pageable-threads.cpp
//   run (compare the two THREADS settings at the same device count):
//     HIP_VISIBLE_DEVICES=0,1 THREADS=0 /tmp/probe-pageable-threads 2 600 64 150
//     HIP_VISIBLE_DEVICES=0,1 THREADS=1 /tmp/probe-pageable-threads 2 600 64 150
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int    ndev    = argc > 1 ? atoi(argv[1]) : 2;
    const size_t chunk   = (argc > 2 ? (size_t) atoi(argv[2]) : 600) << 10;
    const int    per_dev = argc > 3 ? atoi(argv[3]) : 64;
    const int    rounds  = argc > 4 ? atoi(argv[4]) : 150;
    const bool   threads = getenv("THREADS") != nullptr && atoi(getenv("THREADS")) != 0;
    // SHARED=1: the worker threads enqueue on the MAIN thread's stream for that device (two enqueueing
    // threads per stream, which is what the in-tree queue did) instead of one stream per thread.
    const bool   shared  = getenv("SHARED") != nullptr && atoi(getenv("SHARED")) != 0;

    std::vector<hipStream_t> stream((size_t) ndev);
    std::vector<void *>      dst((size_t) ndev);
    const size_t page_bytes = chunk * (size_t) per_dev;
    void * page = nullptr;
    // MMAPF=<file>: source the copies from a FILE-BACKED read-only mapping (the model's situation) instead of
    // anonymouse memory -- a pageable copy that has to fault file pages in is a different driver path, and it
    // is the one difference between the in-tree queue (which wedged) and this probe (which does not).
    const char * mmapf = getenv("MMAPF");
    if (mmapf != nullptr) {
        const int fd = open(mmapf, O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        page = mmap(nullptr, page_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (page == MAP_FAILED) { perror("mmap"); return 1; }
        printf("  source: %s (file-backed, first %zu MiB, NOT pre-faulted)\n", mmapf, page_bytes >> 20);
    } else {
        if (posix_memalign(&page, 4096, page_bytes) != 0) {
            return 1;
        }
        memset(page, 1, page_bytes);
        printf("  source: anonymous memory\n");
    }
    std::vector<hipStream_t> wstream((size_t) ndev);
    for (int d = 0; d < ndev; d++) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipStreamCreate(&stream[(size_t) d]));
        HIPCK(hipStreamCreate(&wstream[(size_t) d]));
        HIPCK(hipMalloc(&dst[(size_t) d], page_bytes));
        HIPCK(hipDeviceSynchronize());
    }

    // one round = every device issues its run of fills (the pattern of one MoE op)
    const auto round_issue = [&](int d) {
        HIPCK(hipSetDevice(d));
        // the stream the push lands on: shared with the main thread under SHARED=1, else private to this thread
        hipStream_t s = shared ? stream[(size_t) d] : wstream[(size_t) d];
        for (int c = 0; c < per_dev; c++) {
            HIPCK(hipMemcpyAsync((char *) dst[(size_t) d] + (size_t) c * chunk, (const char *) page + (size_t) c * chunk,
                                 chunk, hipMemcpyHostToDevice, s));
        }
    };
    const auto round_sync = [&](int d) {
        HIPCK(hipSetDevice(d));
        HIPCK(hipStreamSynchronize(stream[(size_t) d]));
        if (!shared) { HIPCK(hipStreamSynchronize(wstream[(size_t) d])); }
    };

    // warm-up
    for (int d = 0; d < ndev; d++) { round_issue(d); }
    for (int d = 0; d < ndev; d++) { round_sync(d); }

    const double t0 = now_ms();
    for (int r = 0; r < rounds; r++) {
        if (threads) {
            std::vector<std::thread> th;
            for (int d = 0; d < ndev; d++) {
                th.emplace_back([&, d] { round_issue(d); });
            }
            for (auto & t : th) { t.join(); }
        } else {
            for (int d = 0; d < ndev; d++) { round_issue(d); }
        }
        for (int d = 0; d < ndev; d++) { round_sync(d); }
    }
    const double ms = now_ms() - t0;
    const double bytes = (double) page_bytes * (double) ndev * (double) rounds;
    printf("  %d device(s), %s%s: %8.2f ms  %7.2f GB/s  %6.1f us/fill\n", ndev,
           threads ? "one thread PER DEVICE" : "ONE thread for all devices",
           shared ? " (SHARED stream)" : " (private stream)", ms, bytes / 1e9 / (ms / 1000.0),
           1000.0 * ms / ((double) per_dev * (double) ndev * (double) rounds));
    return 0;
}
