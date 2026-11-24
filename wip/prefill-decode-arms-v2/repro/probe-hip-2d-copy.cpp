// probe: which D2H hipMemcpy2DAsync argument combinations does this ROCm line refuse?
//
// Cluster-free, model-free, RCCL-free: it isolates the transport constraint behind the 3-device `-sm tensor`
// abort
//   "2-D read of 'result_output' failed (invalid argument): 331092 bytes x 5 copies, tensor pitch 331092,
//    host pitch 993280"
// The meta backend's chunked 2-D path hands each device's row-chunk a host destination offset accumulated from
// the per-device chunk widths (0, 331092, 662184), so device 1's destination is 4-byte aligned; and the
// destination is llama.cpp's PINNED output buffer, allocated from ONE device's pinned host buffer type
// (`llama_context::output_host_buft()`), while the copy is issued on EVERY device of the split.
//
// Usage: HIP_VISIBLE_DEVICES=0,1,2 ./probe-hip-2d-copy
// build: hipcc -O2 -o probe-hip-2d-copy probe-hip-2d-copy.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

static const char * memtype(int t) {
    switch (t) {
        case hipMemoryTypeUnregistered: return "unreg";
        case hipMemoryTypeHost:         return "host";
        case hipMemoryTypeDevice:       return "device";
        case hipMemoryTypeManaged:      return "managed";
        case hipMemoryTypeArray:        return "array";
        case hipMemoryTypeUnified:      return "unified";
        default:                        return "?";
    }
}

#define E(x) (x)

// dst pinned host memory allocated while `dst_dev` is current; src device memory on `src_dev`; the copy is
// issued on `copy_dev` (which is what `cuda_ctx->device` is in the real thing).
static void arm(int dst_dev, int src_dev, int copy_dev, unsigned host_flags, size_t width, size_t spitch,
                size_t dpitch, size_t n_copies, size_t dst_off, const char * tag) {
    void * src  = nullptr;
    void * host = nullptr;

    (void) hipSetDevice(dst_dev);
    const hipError_t eh = hipHostMalloc(&host, dpitch*n_copies + 4096 + dst_off + 64, host_flags);
    if (eh != hipSuccess) {
        printf("  %-46s -> hipHostMalloc %s\n", tag, hipGetErrorString(eh));
        return;
    }
    (void) hipMemset(host, 0xAB, 64);

    (void) hipSetDevice(src_dev);
    (void) hipMalloc(&src, spitch*n_copies + 4096);
    (void) hipMemset(src, 0, spitch*n_copies);

    hipPointerAttribute_t pa = {};
    const hipError_t epa = hipPointerGetAttributes(&pa, (char *) host + dst_off);

    (void) hipSetDevice(copy_dev);
    const hipError_t e = hipMemcpy2DAsync((char *) host + dst_off, dpitch, src, spitch, width, n_copies,
            hipMemcpyDeviceToHost, nullptr);
    (void) hipDeviceSynchronize();

    printf("  %-46s dst(pinned on %d, %s, off %zu) <- src dev %d, on dev %d, w=%zu n=%zu | %s\n",
           tag, pa.device, epa == hipSuccess ? memtype((int) pa.type) : "?", dst_off, src_dev, copy_dev,
           width, n_copies, hipGetErrorString(e));

    (void) hipSetDevice(dst_dev);  (void) hipHostFree(host);
    (void) hipSetDevice(src_dev);  (void) hipFree(src);
}

int main(void) {
    int ndev = 0;
    (void) hipGetDeviceCount(&ndev);
    printf("devices: %d\n", ndev);
    if (ndev < 2) {
        printf("need 2 devices\n");
        return 1;
    }
    const int d0 = 0, d1 = 1, d2 = ndev > 2 ? 2 : 1;

    // the shape of the failing read: an f32 [248320, 5] split 82773/82773/82774 on axis 0
    const size_t width = 331092, spitch = 331092, dpitch = 993280, n = 5;
    const unsigned mapped = hipHostMallocMapped | hipHostMallocPortable;

    printf("\n== 1. the copy issued on the device that owns the pinned destination (the probe that works)\n");
    arm(d1, d1, d1, mapped, width, spitch, dpitch, n, 0,     "same device, dst off 0");
    arm(d1, d1, d1, mapped, width, spitch, dpitch, n, width, "same device, dst off = width");

    printf("\n== 2. THE FIELD CASE: the pinned destination is a mapping of ANOTHER device\n");
    for (int dd = 0; dd < (ndev < 3 ? 2 : 3); ++dd) {
        arm(dd, d1, d1, mapped, width, spitch, dpitch, n, 0,     "cross-device, n=5, dst off 0");
        arm(dd, d1, d1, mapped, width, spitch, dpitch, n, width, "cross-device, n=5, dst off = width");
        arm(dd, d1, d1, mapped, width, spitch, dpitch, 1, width, "cross-device, n=1, dst off = width");
    }

    printf("\n== 3. n_copies sweep on the cross-device destination (dev %d -> copy on %d)\n", d2, d1);
    for (size_t nn = 1; nn <= 6; ++nn) {
        char tag[64];
        snprintf(tag, sizeof tag, "cross-device n=%zu", nn);
        arm(d2, d1, d1, mapped, width, spitch, dpitch, nn, width, tag);
    }

    printf("\n== 4. is it the MAPPED flag? (portable only / plain pinned) on the same cross-device pair\n");
    arm(d2, d1, d1, hipHostMallocPortable,                          width, spitch, dpitch, n, width, "portable only, n=5");
    arm(d2, d1, d1, hipHostMallocDefault,                           width, spitch, dpitch, n, width, "default (portable+mapped), n=5");
    arm(d2, d1, d1, hipHostMallocDefault,                           width, spitch, dpitch, n, 0,     "default (portable+mapped), n=5, off 0");

    printf("\n== 5. pageable host destination, cross-device source\n");
    {
        void * src = nullptr;
        (void) hipSetDevice(d1);
        (void) hipMalloc(&src, spitch*n);
        void * host = malloc(dpitch*n + 4096);
        const hipError_t e = hipMemcpy2DAsync((char *) host + width, dpitch, src, spitch, width, n,
                hipMemcpyDeviceToHost, nullptr);
        (void) hipDeviceSynchronize();
        printf("  %-46s -> %s\n", "pageable dst off=width, on dev 1", hipGetErrorString(e));
        free(host); (void) hipFree(src);
    }

    printf("\n== 6b. TWO IN-FLIGHT pitched copies from different devices into the same pinned destination\n");
    for (int nn = 1; nn <= 5; nn += 4) {
        // one fresh pinned destination owned by device `d2`, then device 0's copy and device 1's copy queued
        // back to back with no synchronise in between -- exactly the order the meta chunked path issues them
        void * h = nullptr;
        void * s0 = nullptr;
        void * s1 = nullptr;
        (void) hipSetDevice(d2);
        (void) hipHostMalloc(&h, dpitch*nn + 4096, mapped);
        (void) hipSetDevice(d0);
        (void) hipMalloc(&s0, width*nn);
        (void) hipSetDevice(d1);
        (void) hipMalloc(&s1, width*nn);
        hipStream_t st0 = nullptr, st1 = nullptr;
        (void) hipStreamCreateWithFlags(&st0, hipStreamNonBlocking);
        (void) hipStreamCreateWithFlags(&st1, hipStreamNonBlocking);
        (void) hipSetDevice(d0);
        hipError_t e0 = hipMemcpy2DAsync((char *) h + 0,     dpitch, s0, width, width, nn, hipMemcpyDeviceToHost, st0);
        (void) hipSetDevice(d1);
        hipError_t e1 = hipMemcpy2DAsync((char *) h + width, dpitch, s1, width, width, nn, hipMemcpyDeviceToHost, st1);
        (void) hipStreamSynchronize(st0);
        (void) hipStreamSynchronize(st1);
        printf("  in-flight pair, n=%d: dev %d -> %s ; dev %d -> %s\n", nn, d0, hipGetErrorString(e0), d1, hipGetErrorString(e1));
        // and the same pair with ONE copy per device (n=1) for contrast
        (void) hipStreamDestroy(st0); (void) hipStreamDestroy(st1);
        (void) hipFree(s0); (void) hipFree(s1); (void) hipHostFree(h);
    }

    printf("\n== 6c. per-device copies into ONE destination, but each issued on the NULL stream\n");
    {
        void * h = nullptr;
        (void) hipSetDevice(d2);
        (void) hipHostMalloc(&h, dpitch*n + 4096, mapped);
        for (int dd = 0; dd < (ndev < 3 ? 2 : 3); ++dd) {
            void * s = nullptr;
            (void) hipSetDevice(dd);
            (void) hipMalloc(&s, width*n);
            const hipError_t e = hipMemcpy2DAsync((char *) h + dd*width, dpitch, s, width, width, n,
                    hipMemcpyDeviceToHost, nullptr);
            printf("  dev %d -> %s\n", dd, hipGetErrorString(e));
            (void) hipFree(s);
        }
        (void) hipDeviceSynchronize();
        (void) hipHostFree(h);
    }

    printf("\n== 6. a device-local pinned BOUNCE then a host memcpy (the staging shape)\n");
    {
        void * src = nullptr, * bounce = nullptr, * host = nullptr;
        (void) hipSetDevice(d1);
        (void) hipMalloc(&src, spitch*n);
        (void) hipHostMalloc(&bounce, width*n, mapped);
        (void) hipSetDevice(d2);
        (void) hipHostMalloc(&host, dpitch*n + 4096, mapped);
        (void) hipSetDevice(d1);
        hipError_t e = hipMemcpy2DAsync(bounce, spitch, src, spitch, width, n, hipMemcpyDeviceToHost, nullptr);
        printf("  %-46s -> dev %d local 2-D into its own pinned bounce: %s\n", "bounce step", d1, hipGetErrorString(e));
        for (size_t i = 0; i < n; ++i) {
            memcpy((char *) host + i*dpitch + width, (const char *) bounce + i*spitch, width);
        }
        printf("  %-46s -> host memcpy of the bounce: ok\n", "restage step");
        (void) hipDeviceSynchronize();
        (void) hipHostFree(bounce); (void) hipHostFree(host); (void) hipFree(src);
    }
    return 0;
}
