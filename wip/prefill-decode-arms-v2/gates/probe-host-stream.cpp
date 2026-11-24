// Non-vacuity probe for STEP 1's per-thread host stream.
//
// WHAT IT PROVES: `ggml_cuda_host_stream()` returns a REAL private stream, not the
// `hipStreamPerThread` sentinel it replaces, and it returns the SAME handle for the same
// (thread, device) on every call -- so a host copy really moved off the legacy default stream that
// a graph capture claims.  Without this, "the text did not change" is equally consistent with the
// helper having returned the sentinel, so a text-only oracle is vacuous for this fix.
//
// WHY IT MATTERS: HIP maps `hipStreamPerThread` onto the legacy default stream, and a graph
// capture claims that stream.  A host copy issued by a second context is then refused with
// `operation not permitted when stream is capturing`, and every later launch on the device fails.
// The sentinel's numeric value on HIP is 2 (`(hipStream_t) 2`); a private stream is a real handle.
//
// Build/run (see the STEP 1 record for the exact line):
//   HIP_VISIBLE_DEVICES=0 <clang++> -x hip --offload-arch=gfx1201 probe-host-stream.cpp \
//       -I<rocm>/include -L<build>/bin -lggml-hip -Wl,-rpath,<build>/bin -o /tmp/probe-host-stream
//   /tmp/probe-host-stream

#include <hip/hip_runtime.h>

// ggml maps the CUDA spellings onto HIP in ggml/src/ggml-cuda/vendors/hip.h; the probe speaks HIP
// directly (cudaStream_t == hipStream_t, cudaStreamPerThread == ((hipStream_t) 2)).

#include <cstdio>
#include <thread>
#include <vector>

// defined in ggml-cuda.cu (declared in common.cuh there)
hipStream_t ggml_cuda_host_stream();

int main() {
    // the sentinel's value on HIP; CUDA's is 0x2 as well, but the CHECK below is "not the sentinel
    // AND not the legacy default stream", which is what actually matters
    const hipStream_t per_thread = hipStreamPerThread;

    const int device_count = [] {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess) { return 0; }
        return n;
    }();
    if (device_count <= 0) {
        printf("FAIL: no HIP devices\n");
        return 1;
    }

    std::vector<hipStream_t> first(device_count, nullptr), second(device_count, nullptr);
    for (int d = 0; d < device_count; d++) {
        if (hipSetDevice(d) != hipSuccess) { printf("FAIL: hipSetDevice(%d)\n", d); return 1; }
        first[d]  = ggml_cuda_host_stream();
        second[d] = ggml_cuda_host_stream();
        printf("device %d: host_stream = %p (repeat %p), hipStreamPerThread = %p, legacy default = %p\n",
               d, (void *) first[d], (void *) second[d], (void *) per_thread, (void *) nullptr);
        if (first[d] == nullptr) {
            printf("FAIL: device %d returned a null stream\n", d);
            return 1;
        }
        if (first[d] == second[d]) {
            printf("  device %d: stable handle (ok)\n", d);
        } else {
            printf("FAIL: device %d: the handle changed between two calls in one thread\n", d);
            return 1;
        }
        if (first[d] == per_thread) {
            printf("FAIL: device %d: got the hipStreamPerThread sentinel -- the fix is not in effect\n", d);
            return 1;
        }
        if (first[d] == (hipStream_t) nullptr) {
            printf("FAIL: device %d: got the legacy default stream\n", d);
            return 1;
        }
        for (int e = 0; e < d; e++) {
            if (first[e] == first[d]) {
                printf("FAIL: devices %d and %d share one stream (the stream must be per device)\n", e, d);
                return 1;
            }
        }
    }

    // per THREAD: a second thread must get its own handle, or two contexts would still queue behind
    // each other on one stream (and a capture on one thread would still block the other's copies)
    hipStream_t other = nullptr;
    std::thread t([&] {
        if (hipSetDevice(0) == hipSuccess) { other = ggml_cuda_host_stream(); }
    });
    t.join();
    if (other == nullptr || other == first[0]) {
        printf("FAIL: a second thread got the same stream as this one (%p) -- the stream is not per thread\n",
               (void *) other);
        return 1;
    }
    printf("second thread: host_stream = %p (distinct from thread 1's %p): ok\n", (void *) other, (void *) first[0]);

    printf("PASS: a private, stable, per-(thread, device) host stream, not the captured default\n");
    return 0;
}
