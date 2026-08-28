// kern_gap.cpp — measure HIP inter-kernel launch gap on gfx1201.
// Loaded via: hipcc kern_gap.cpp -o kern_gap
//
// Measures three configurations and reports wall time and per-launch cost:
//   1. "seq"  : N tiny kernels launched sequentially on one stream (gap-bound).
//   2. "seq2" : same, but each kernel does a tiny amount of actual work so we
//               can separate pure-launch from launch+work.
//   3. "graph": the same N kernels captured into a HIP graph and replayed once.
//
// The inter-kernel gap dominates in #1. If ROCm 10 reduced it, the per-launch
// cost in #1/#2 will be measurably lower than on 7.14.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CK(...) do { hipError_t _err = (__VA_ARGS__); if (_err != hipSuccess) { \
    printf("HIP error %d at line %d: %s\n", _err, __LINE__, hipGetErrorString(_err)); exit(1);} } while (0)

// Tiny kernel: touches a few floats so it's not entirely empty, but does
// essentially nothing — so wall time is dominated by launch/gap.
__global__ void tiny(float * out, float a) {
    int i = threadIdx.x;
    if (i < 4) {
        out[i * blockIdx.x] = a * (float) i;
    }
}

// Slightly more work: reads/writes a small buffer to model a real (small) op.
__global__ void small(float * out, const float * in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 1.000001f + 0.5f;
    }
}

static double time_ms(hipEvent_t s, hipEvent_t e) {
    CK(hipEventSynchronize(e));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, s, e));
    return ms;
}

int main(int argc, char ** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
    CK(hipSetDevice(dev));

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, dev));
    printf("device=%d %s  gcn=%s  wave=%d\n", dev, prop.name, prop.gcnArchName, prop.warpSize);
    printf("hip version: %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);

    const int N = 4096;          // number of kernels to launch
    size_t buf_bytes = 4 * sizeof(float);
    float * d_out; CK(hipMalloc(&d_out, buf_bytes));

    hipStream_t stream;
    CK(hipStreamCreate(&stream));

    // ---- mode 1: sequential tiny kernels (gap-bound) ----
    {
        hipEvent_t s, e;
        CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        CK(hipEventRecord(s, stream));
        for (int i = 0; i < N; i++) {
            tiny<<<1, 32, 0, stream>>>(d_out, 1.0f);
        }
        CK(hipEventRecord(e, stream));
        double ms = time_ms(s, e);
        printf("seq_tiny   : %8.3f ms  (%6.3f us/launch)  N=%d\n", ms, ms * 1000.0 / N, N);
        CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
    }

    // ---- mode 2: sequential kernels with a bit of work ----
    {
        const int n = 4096;
        float * d_in; CK(hipMalloc(&d_in, n * sizeof(float)));
        float * d_out2; CK(hipMalloc(&d_out2, n * sizeof(float)));
        CK(hipMemset(d_in, 0, n * sizeof(float)));
        hipEvent_t s, e;
        CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        CK(hipEventRecord(s, stream));
        for (int i = 0; i < N; i++) {
            small<<<(n + 63)/64, 64, 0, stream>>>(d_out2, d_in, n);
        }
        CK(hipEventRecord(e, stream));
        double ms = time_ms(s, e);
        printf("seq_small  : %8.3f ms  (%6.3f us/launch)  N=%d\n", ms, ms * 1000.0 / N, N);
        CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
        CK(hipFree(d_in)); CK(hipFree(d_out2));
    }

    // ---- mode 3: graph capture + single replay ----
    {
        hipEvent_t s, e;
        CK(hipEventCreate(&s)); CK(hipEventCreate(&e));
        hipGraph_t graph; hipGraphExec_t exec;
        CK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        for (int i = 0; i < N; i++) {
            tiny<<<1, 32, 0, stream>>>(d_out, 1.0f);
        }
        CK(hipStreamEndCapture(stream, &graph));
        CK(hipGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CK(hipEventRecord(s, stream));
        CK(hipGraphLaunch(exec, stream));
        CK(hipEventRecord(e, stream));
        double ms = time_ms(s, e);
        printf("graph_tiny : %8.3f ms  (%6.3f us/kernel)  N=%d  (single graph launch)\n", ms, ms * 1000.0 / N, N);
        CK(hipGraphExecDestroy(exec)); CK(hipGraphDestroy(graph));
        CK(hipEventDestroy(s)); CK(hipEventDestroy(e));
    }

    CK(hipFree(d_out));
    CK(hipStreamDestroy(stream));
    return 0;
}
