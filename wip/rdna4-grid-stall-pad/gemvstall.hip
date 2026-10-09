// Does the grid-size stall survive real work? A GEMV-shaped kernel: one block per row, 2560 bf16 per row.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s\n", hipGetErrorString(e)); exit(1); } } while (0)
__global__ void gemv(const unsigned short * w, const float * x, float * y, int ncols, int nrows) {
    const int row = blockIdx.x;
    if (row >= nrows) return;                       // the padding block
    float acc = 0.f;
    for (int c = threadIdx.x; c < ncols; c += blockDim.x) {
        unsigned int b = ((unsigned int) w[(size_t) row * ncols + c]) << 16;
        acc += __uint_as_float(b) * x[c];
    }
    __shared__ float s[256]; s[threadIdx.x] = acc; __syncthreads();
    for (int k = 128; k > 0; k >>= 1) { if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) y[row] = s[0];
}
int main() {
    const int ncols = 2560, N = 500;
    unsigned short * w; float * x, * y;
    CK(hipMalloc(&w, (size_t) 4096 * ncols * 2)); CK(hipMalloc(&x, ncols * 4)); CK(hipMalloc(&y, 8192 * 4));
    CK(hipMemset(w, 0x3c, (size_t) 4096 * ncols * 2)); CK(hipMemset(x, 0, ncols * 4));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const int rows[] = {256, 512, 1024};
    for (int nr : rows) for (int pad = 0; pad < 2; ++pad) {
        hipGraph_t gr; hipGraphExec_t ge;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        for (int i = 0; i < N; ++i) hipLaunchKernelGGL(gemv, dim3(nr + pad), dim3(256), 0, s, w, x, y, ncols, nr);
        CK(hipStreamEndCapture(s, &gr)); CK(hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0)); CK(hipGraphDestroy(gr));
        float best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            CK(hipEventRecord(a, s)); CK(hipGraphLaunch(ge, s)); CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
        }
        printf("rows %4d  grid %4d  %.2f us per GEMV (%.2f MB read)\n", nr, nr + pad, best * 1000.f / N, nr * ncols * 2 / 1e6);
        CK(hipGraphExecDestroy(ge));
    }
    return 0;
}
