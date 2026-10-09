// Per-kernel cost inside a captured HIP graph on this card: N dependent tiny kernels, timed with events.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while (0)
__global__ void tiny(float * p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * 1.0001f + 1.0f; }
int main() {
    float * buf; CK(hipMalloc(&buf, 64 << 20)); CK(hipMemset(buf, 0, 64 << 20));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const int N = 2000;
    const int grids[] = {192, 256, 320, 384, 448, 480, 496, 504, 512, 520, 544, 576, 640, 768, 1024, 1536};
    for (int g : grids) {
        for (int mode = 1; mode < 2; ++mode) {   // 0 = stream launches, 1 = captured graph
            hipGraphExec_t ge = nullptr;
            if (mode == 1) {
                hipGraph_t gr; CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
                for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(g), dim3(256), 0, s, buf, g * 256);
                CK(hipStreamEndCapture(s, &gr)); CK(hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0)); CK(hipGraphDestroy(gr));
            }
            float best = 1e9;
            for (int rep = 0; rep < 5; ++rep) {
                CK(hipEventRecord(a, s));
                if (mode == 1) CK(hipGraphLaunch(ge, s));
                else for (int i = 0; i < N; ++i) hipLaunchKernelGGL(tiny, dim3(g), dim3(256), 0, s, buf, g * 256);
                CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b));
                float ms; CK(hipEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
            }
            printf("grid %5d x 256  %-6s  %.2f us per kernel\n", g, mode ? "graph" : "stream", best * 1000.0f / N);
            if (ge) CK(hipGraphExecDestroy(ge));
        }
    }
    return 0;
}
