// probe-copysync -- is a copy from a `hipHostRegister`ed mapping SYNCHRONOUS (the host waits for the DMA),
// while the same copy from `hipHostMalloc` memory is asynchronous?  And does copying from the UVA ALIAS
// (device-to-device, as the pool path already does) restore asynchrony?
//
// WHY (wip/host-expert-eager-push P4 5d): registering the model mapping makes fills fast (141 us -> 0.3-0.5
// us/copy) and beats the pinned master at 2 devices, but at ONE device prefill drops 22 % (478 vs 615 t/s)
// with an IDENTICAL copy census.  `perf` puts 15.6 % of the host's time in `BusyWaitSignal::WaitRelaxed` in
// the registered arm (0.8 % in the pool arm, which instead spends 71 % in `__memmove`) -- i.e. the registered
// copy blocks the host.  This measures that directly, and the candidate fix.
//
// METHOD: put a long D2D copy on the stream first (so the stream has ~10 ms of work queued), then time the
// HOST side of one small copy issued behind it:
//   async  -> the call returns in microseconds (it only enqueued)
//   sync   -> the call takes about as long as the queued work (the runtime waited)
//
//   build:
//     /opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-copysync probe-copysync.cpp
//   run:  FILE=<gguf> /tmp/probe-copysync [MiB_copy] [MiB_queue]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <vector>

#define HIPCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static double us_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    const size_t copy_mib  = argc > 1 ? (size_t) atol(argv[1]) : 1;    // the fill we time
    const size_t queue_mib = argc > 2 ? (size_t) atol(argv[2]) : 256;  // the work queued ahead of it
    const char * file = getenv("FILE") ? getenv("FILE")
        : "/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    const size_t copy_len = copy_mib << 20;
    HIPCK(hipSetDevice(0));
    hipStream_t s;
    HIPCK(hipStreamCreate(&s));

    // sources
    void * pinned = nullptr;
    HIPCK(hipHostMalloc(&pinned, copy_len, hipHostMallocDefault));
    memset(pinned, 1, copy_len);

    const int fd = open(file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    char * mapping = (char *) mmap(nullptr, copy_len, PROT_READ, MAP_SHARED, fd, 1L << 30);
    if (mapping == MAP_FAILED) { perror("mmap"); return 1; }
    volatile long acc = 0;
    for (size_t o = 0; o < copy_len; o += 4096) { acc += mapping[o]; }
    (void) acc;
    HIPCK(hipHostRegister(mapping, copy_len, hipHostRegisterReadOnly | hipHostRegisterPortable));
    void * alias = nullptr;
    const bool alias_ok = hipHostGetDevicePointer(&alias, mapping, 0) == hipSuccess && alias != nullptr;
    printf("probe-copysync: %zu MiB copy, %zu MiB of D2D queued ahead of it\n", copy_mib, queue_mib);
    printf("  hipHostGetDevicePointer over the REGISTERED mapping: %s\n",
           alias_ok ? "available (the UVA alias exists)" : "NOT available");

    void * d = nullptr, * q1 = nullptr, * q2 = nullptr;
    HIPCK(hipMalloc(&d, copy_len));
    HIPCK(hipMalloc(&q1, queue_mib << 20));
    HIPCK(hipMalloc(&q2, queue_mib << 20));
    HIPCK(hipMemset(q1, 0, queue_mib << 20));
    HIPCK(hipStreamSynchronize(s));

    // one arm: queue the work, then time the host side of the copy call
    const auto arm = [&](const char * name, const void * src, hipMemcpyKind kind, void * dst) {
        HIPCK(hipMemcpyAsync(q2, q1, queue_mib << 20, hipMemcpyDeviceToDevice, s));   // ~10 ms queued
        const auto t0 = std::chrono::steady_clock::now();
        HIPCK(hipMemcpyAsync(dst, src, copy_len, kind, s));
        const double host_us = us_since(t0);
        const auto t1 = std::chrono::steady_clock::now();
        HIPCK(hipStreamSynchronize(s));
        const double total_ms = us_since(t1) / 1000.0;
        printf("  %-42s host %9.1f us   (%s)   stream drained afterwards in %.1f ms\n", name, host_us,
               host_us > 2000.0 ? "SYNCHRONOUS - the host waited" : "asynchronous - it only enqueued",
               total_ms);
        HIPCK(hipStreamSynchronize(s));
    };

    arm("pageable anon -> D",        mapping, hipMemcpyHostToDevice, d);
    arm("hipHostMalloc pinned -> D", pinned,  hipMemcpyHostToDevice, d);
    arm("REGISTERED mapping -> D (H2D)", mapping, hipMemcpyHostToDevice, d);
    if (alias_ok) {
        arm("REGISTERED mapping alias -> D (D2D)", alias, hipMemcpyDeviceToDevice, d);
    }
    return 0;
}
