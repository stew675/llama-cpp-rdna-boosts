// rocBLAS INT8 GEMM throughput for Q8_0-prefill-like shapes on gfx1201.
// C[m,n] = A[m,k] * B[k,n], row-major, int8 in, int32 accum.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main(int argc, char ** argv) {
    int m = argc > 1 ? atoi(argv[1]) : 8704;   // output rows (weights)
    int n = argc > 2 ? atoi(argv[2]) : 512;    // tokens
    int k = argc > 3 ? atoi(argv[3]) : 5120;   // contraction
    int iters = argc > 4 ? atoi(argv[4]) : 50;

    rocblas_handle h;
    rocblas_create_handle(&h);

    int8_t * dA; int8_t * dB; int32_t * dC;
    hipMalloc(&dA, (size_t)m * k);
    hipMalloc(&dB, (size_t)k * n);
    hipMalloc(&dC, (size_t)m * n * sizeof(int32_t));
    hipMemset(dA, 1, (size_t)m * k);
    hipMemset(dB, 1, (size_t)k * n);

    int32_t alpha = 1, beta = 0;

    rocblas_status st = rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
        m, n, k, &alpha,
        dA, rocblas_datatype_i8_r, k,
        dB, rocblas_datatype_i8_r, n,
        &beta,
        dC, rocblas_datatype_i32_r, n,
        dC, rocblas_datatype_i32_r, n,
        rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, uint32_t(0));
    hipDeviceSynchronize();
    printf("status=%d (%s)\n", (int) st, rocblas_status_to_string(st));

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < iters; ++i) {
        rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
            m, n, k, &alpha,
            dA, rocblas_datatype_i8_r, k,
            dB, rocblas_datatype_i8_r, n,
            &beta,
            dC, rocblas_datatype_i32_r, n,
            dC, rocblas_datatype_i32_r, n,
            rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, uint32_t(0));
    }
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1);
    double s = ms / 1e3 / iters;
    double macs = (double)m * n * k;
    printf("rocBLAS i8  m=%d n=%d k=%d : %.3f ms  -> %.1f T-MAC/s (%.1f TOPS)\n",
           m, n, k, s * 1e3, macs / s / 1e12, 2.0 * macs / s / 1e12);
    return 0;
}
