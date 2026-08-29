// Standalone unit test for the ggml_cuda_ar_kernel (extracted from
// allreduce-hip.cu) with n=3, both F32 and BF16 wire.  Verifies:
//   (a) every device's recvbuf is bit-identical across devices
//   (b) recvbuf == sum of all ranks' contributions (rounded through T_wire)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define HK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s: %s\n", #x, hipGetErrorString(e)); exit(1);} } while(0)

// --- helpers the kernel needs (mirroring ggml's) ---
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() { return 16; }
template <typename T>
static __device__ __forceinline__ T ggml_cuda_cast(const float x) {
    if constexpr (std::is_same<T, float>::value) return x;
    if constexpr (std::is_same<T, hip_bfloat16>::value) return hip_bfloat16(x);
    return (T) x;
}
template <int nbytes>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    __builtin_memcpy(dst, src, nbytes);
}

#define NDEV 3
#define GGML_CUDA_MAX_DEVICES 8

#include "/tmp/kernel_fragment.h"
#define NE 5120
#define SLOT_STRIDE_BYTES 64

struct mapping {
    unsigned char * host = nullptr;
    unsigned char * dev  = nullptr;
    hipError_t alloc(size_t bytes) {
        hipError_t rc = hipHostMalloc((void**)&host, bytes, hipHostMallocPortable | hipHostMallocMapped);
        if (rc != hipSuccess) return rc;
        rc = hipHostGetDevicePointer((void**)&dev, host, 0);
        if (rc != hipSuccess) { hipHostFree(host); host = nullptr; }
        return rc;
    }
};

int main(int argc, char ** argv) {
    const int n_devices = argc > 2 ? atoi(argv[2]) : NDEV;
    if (n_devices < 2 || n_devices > 3) { printf("usage: ar_kernel_unit <wire> [n_devices]\n"); return 2; }
    const int wire = argc > 1 ? atoi(argv[1]) : 0; // 0=f32, 1=bf16
    printf("n_devices=%d wire=%s NE=%d\n", n_devices, wire ? "bf16" : "f32", NE);

    // host wire buffer: contiguous (rank*POOL + slot) * MAX_BYTES
    const size_t buf_bytes = GGML_CUDA_AR_MAX_BYTES;
    const size_t wire_bytes = (size_t)GGML_CUDA_AR_POOL_SIZE * n_devices * buf_bytes;
    mapping wire_map, arrival_map;
    HK(wire_map.alloc(wire_bytes));
    HK(arrival_map.alloc(GGML_CUDA_AR_POOL_SIZE * n_devices * 8 * SLOT_STRIDE_BYTES));
    memset(wire_map.host, 0, wire_bytes);
    memset(arrival_map.host, 0, GGML_CUDA_AR_POOL_SIZE * n_devices * 8 * SLOT_STRIDE_BYTES);

    // per-device buffers + streams
    float * devbuf[NDEV]; hipStream_t streams[NDEV];
    float * hbuf[NDEV];
    for (int d = 0; d < n_devices; ++d) {
        HK(hipSetDevice(d));
        HK(hipMalloc(&devbuf[d], NE * sizeof(float)));
        HK(hipStreamCreate(&streams[d]));
        hbuf[d] = (float *) malloc(NE * sizeof(float));
        for (int i = 0; i < NE; ++i) hbuf[d][i] = 100.0f + d * 10.0f + (i % 7) * 0.125f;
        HK(hipMemcpy(devbuf[d], hbuf[d], NE * sizeof(float), hipMemcpyHostToDevice));
    }

    const int slot = 0, token = 1;
    for (int d = 0; d < n_devices; ++d) {
        HK(hipSetDevice(d));
        if (wire) {
            ggml_cuda_ar_kernel<float, hip_bfloat16><<<dim3(8), dim3(256), 0, streams[d]>>>(
                devbuf[d], devbuf[d], (hip_bfloat16 *)wire_map.dev, (int *)arrival_map.dev,
                d, n_devices, slot, NE, token, nullptr, true);
        } else {
            ggml_cuda_ar_kernel<float, float><<<dim3(8), dim3(256), 0, streams[d]>>>(
                devbuf[d], devbuf[d], (float *)wire_map.dev, (int *)arrival_map.dev,
                d, n_devices, slot, NE, token, nullptr, true);
        }
        HK(hipGetLastError());
    }
    for (int d = 0; d < n_devices; ++d) { HK(hipSetDevice(d)); HK(hipDeviceSynchronize()); }

    // expected: sum over ranks of round_Tw(x_d)
    std::vector<float> expect(NE);
    for (int i = 0; i < NE; ++i) {
        float s = 0;
        for (int d = 0; d < n_devices; ++d) {
            float v = hbuf[d][i];
            if (wire) v = (float)(float)hip_bfloat16(v);
            s += v;
        }
        expect[i] = s;
    }


    // --- debug: dump wire buffer contents per rank slot ---
    {
        float * w = (float *)wire_map.host;
        printf("wire dump (rank slot 0/1/2, first 8 elems each):\n");
        for (int r = 0; r < n_devices; ++r) {
            printf("  rank%d slot%d: ", r, 0);
            float * base = w + (size_t)(r * GGML_CUDA_AR_POOL_SIZE + slot) * (GGML_CUDA_AR_MAX_BYTES / sizeof(float));
            for (int k = 0; k < 8; ++k) printf("%.3f ", base[k]);
            printf("\n");
        }
        printf("  input check hbuf[0][0..3]: %.3f %.3f %.3f %.3f\n", hbuf[0][0], hbuf[0][1], hbuf[0][2], hbuf[0][3]);
    }
    int mism = 0;
    for (int d = 0; d < n_devices; ++d) {
        HK(hipSetDevice(d));
        HK(hipMemcpy(hbuf[d], devbuf[d], NE * sizeof(float), hipMemcpyDeviceToHost));
        for (int i = 0; i < NE; ++i) {
            if (hbuf[d][i] != expect[i]) {
                if (mism < 10) printf("dev%d [%d]: got %f expect %f\n", d, i, hbuf[d][i], expect[i]);
                mism++;
            }
        }
    }
    // cross-device identity
    int xmism = 0;
    for (int i = 0; i < NE; ++i)
        if (hbuf[0][i] != hbuf[1][i] || hbuf[0][i] != hbuf[2][i]) xmism++;
    printf("RESULT: mismatches=%d (%s)  cross-device-mismatches=%d\n",
           mism, mism ? "FAIL" : "PASS", xmism);
    return mism || xmism ? 1 : 0;
}
