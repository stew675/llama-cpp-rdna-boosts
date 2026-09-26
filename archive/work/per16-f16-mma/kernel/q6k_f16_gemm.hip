// Phase-1 spike: fused Q6_K -> F16 WMMA GEMM at the 27B ffn shape on gfx1201.
//
// Question: can dequantizing Q6_K weights to F16 in shared memory and running the F16 WMMA with
// f32 accumulation beat the current per-16 int8 MMQ (57 TFLOPS at m=17408,n=512,k=5120)?
// This is a measurement harness, not a correctness oracle, but it does check a slice against a CPU
// dequant reference.
//
// Build/run (see ../tools/):
//   hipcc --offload-arch=gfx1201 -O3 -std=c++17 -o q6k_f16_gemm q6k_f16_gemm.hip
//
// Tile: BM=128, BN=64, BK=128; 256 threads (8 warps); each warp 16 rows x 64 cols.
//
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define QK_K 256

struct block_q6_K {
    uint8_t ql[128];   // 2 x 64
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;        // fp16 bits
};

// Canonical ggml ordering (dequantize_row_q6_K).
static inline void dequant_q6k_block(const block_q6_K * x, float * y) {
    const float d = (float)*(const _Float16 *) &x->d;
    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t  * sc = x->scales;
    y += 0;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l/16;
            const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 128;              // half a Q6_K super-block
constexpr int SA_STRIDE = BK + 8;    // halves; even -> 16B-aligned fragment reads
constexpr int SB_STRIDE = BN + 8;
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;

// gfx12 WMMA fragment is 8 halves/lane in "two runs of four": k = 4*hi + {0..3} and
// 4*hi + 8 + {0..3} (see mmb.cu mmb_ld_frag).  C is J-major here: lane holds m = 8*hi + e,
// n = lane%16.
typedef _Float16 halfx8 __attribute__((ext_vector_type(8)));
typedef float    floatx8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ halfx8 ld_frag8(const _Float16 * row_k, const int hi) {
    const _Float16 * q = row_k + 4*hi;
    return __builtin_bit_cast(halfx8,
        (uint2[2]){*reinterpret_cast<const uint2 *>(q), *reinterpret_cast<const uint2 *>(q + 8)});
}

// Dequantize the 128-value half-super-block at (row, k0) into the sram A tile.
// 128 rows * 128 values / 256 threads = 64 values/thread.
__device__ __forceinline__ void load_a(const block_q6_K * A, int stride_blocks, int m0, int k0,
                                       _Float16 * sA) {
    const int t = threadIdx.x;             // 0..255
    const int row = t >> 1;                // 0..127
    const int half = t & 1;                // which 64-value half of the 128
    const int m = m0 + row;

    const int kb   = k0 >> 8;              // super-block index in the row
    const int noff = k0 & 255;             // 0 or 128
    const block_q6_K * b = A + (size_t) m * stride_blocks + kb;
    const float d = (float)*(const _Float16 *) &b->d;

    const uint8_t * ql = b->ql + (noff ? 64 : 0);
    const uint8_t * qh = b->qh + (noff ? 32 : 0);
    const int8_t  * sc = b->scales + (noff ? 8 : 0);

    _Float16 * dst = sA + (size_t) row * SA_STRIDE;
#pragma unroll
    for (int l0 = 0; l0 < 16; ++l0) {
        const int l  = half * 16 + l0;
        const int is = l/16;
        const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
        dst[l +  0] = (_Float16)(d * sc[is + 0] * q1);
        dst[l + 32] = (_Float16)(d * sc[is + 2] * q2);
        dst[l + 64] = (_Float16)(d * sc[is + 4] * q3);
        dst[l + 96] = (_Float16)(d * sc[is + 6] * q4);
    }
}

// Load the BK x BN F16 activation tile (global row-major [K][N]) into sram as [N][K] — the WMMA B
// fragment's get_i is the N index and get_j is the K index (see mmq-vec-dot.cuh: tile_y is [J][K]).
__device__ __forceinline__ void load_b(const _Float16 * B, int n, int k0, int n0, _Float16 * sB) {
    for (int i = threadIdx.x; i < BK * BN; i += NTHREADS) {
        const int kk = i / BN;
        const int nn = i % BN;
        sB[nn * SB_STRIDE + kk] = B[(size_t)(k0 + kk) * n + n0 + nn];
    }
}

__global__ void q6k_f16_gemm(const block_q6_K * __restrict__ A, int stride_blocks,
                             const _Float16 * __restrict__ B, float * __restrict__ C,
                             int M, int N, int K) {
    __shared__ _Float16 sA[BM * SA_STRIDE];
    __shared__ _Float16 sB[BK * SB_STRIDE];

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int mr = warp * 16;              // 8 warps x 16 rows = 128
    const int r  = lane & 15;              // lane's row inside the 16-row tile (weight M, act N)
    const int hi = lane >> 4;              // 0/1: the K run pair

    floatx8 acc[BN/16];
#pragma unroll
    for (int i = 0; i < BN/16; ++i) {
        acc[i] = (floatx8){0,0,0,0,0,0,0,0};
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_a(A, stride_blocks, m0, k0, sA);
        load_b(B, N, k0, n0, sB);
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK/16; ++ks) {
            const halfx8 a = ld_frag8(&sA[(size_t)(mr + r) * SA_STRIDE + ks*16], hi);
#pragma unroll
            for (int nt = 0; nt < BN/16; ++nt) {
                const halfx8 b = ld_frag8(&sB[(size_t)(nt*16 + r) * SB_STRIDE + ks*16], hi);
                acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, a, acc[nt]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < BN/16; ++nt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            C[(size_t)(m0 + mr + 8*hi + e) * N + n0 + nt*16 + r] = acc[nt][e];
        }
    }
}

// ---------------------------------------------------------------- host

static void fill_random_q6k(std::vector<block_q6_K> & v, int nblocks) {
    srand(1234);
    for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < 128; ++j) v[i].ql[j] = rand() & 0xFF;
        for (int j = 0; j < 64;  ++j) v[i].qh[j] = rand() & 0xFF;
        for (int j = 0; j < 16;  ++j) v[i].scales[j] = (int8_t)((rand() % 255) - 127);
        v[i].d = (uint16_t)0x2200; // ~0.1 fp16
    }
}

int main(int argc, char ** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 17408;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 5120;
    const int iters = argc > 4 ? atoi(argv[4]) : 50;

    if (K % QK_K != 0 || M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "bad dims\n"); return 1;
    }
    const int stride_blocks = K / QK_K;

    std::vector<block_q6_K> hA((size_t)M * stride_blocks);
    fill_random_q6k(hA, (int)hA.size());
    std::vector<_Float16> hB((size_t)K * N);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (_Float16)((int)(i * 2654435761u % 1000) / 1000.0f - 0.5f);
    std::vector<float> hC((size_t)M * N, 0.0f);

    void *dA, *dB, *dC;
    hipMalloc(&dA, hA.size() * sizeof(block_q6_K));
    hipMalloc(&dB, hB.size() * sizeof(_Float16));
    hipMalloc(&dC, hC.size() * sizeof(float));
    hipMemcpy(dA, hA.data(), hA.size() * sizeof(block_q6_K), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), hB.size() * sizeof(_Float16), hipMemcpyHostToDevice);

    dim3 grid(M / BM, N / BN);
    dim3 block(NTHREADS);

    for (int i = 0; i < 3; ++i) q6k_f16_gemm<<<grid, block>>>((block_q6_K *)dA, stride_blocks, (_Float16 *)dB, (float *)dC, M, N, K);
    hipDeviceSynchronize();

    // ---- correctness: recompute the first 128x64 output tile on the CPU from dequantized A ----
    hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost);
    std::vector<float> dq((size_t)stride_blocks * QK_K);
    float maxerr = 0.0f, maxref = 0.0f;
    int em = -1, en = -1;
    for (int m = 0; m < BM; ++m) {
        for (int kb = 0; kb < stride_blocks; ++kb) dequant_q6k_block(&hA[(size_t)m*stride_blocks + kb], dq.data() + (size_t)kb*QK_K);
        for (int nn = 0; nn < BN; ++nn) {
            double s = 0.0;
            for (int kk = 0; kk < K; ++kk) s += (double)dq[kk] * (double)hB[(size_t)kk*N + nn];
            const float r = (float)s;
            const float e = fabsf(hC[(size_t)m*N + nn] - r);
            if (e > maxerr) { maxerr = e; em = m; en = nn; }
            maxref = fmaxf(maxref, fabsf(r));
        }
    }
    printf("correctness first tile: max abs err %.4f at (%d,%d) (max |ref| %.1f), rel %.4f\n",
           maxerr, em, en, maxref, maxerr / (maxref > 0 ? maxref : 1.0f));
    printf("  sample C[0][0..3] = %.3f %.3f %.3f %.3f\n", hC[0], hC[1], hC[2], hC[3]);
    {
        std::vector<float> dq0((size_t)stride_blocks * QK_K);
        for (int kb = 0; kb < stride_blocks; ++kb) dequant_q6k_block(&hA[(size_t)0*stride_blocks + kb], dq0.data() + (size_t)kb*QK_K);
        for (int nn = 0; nn < 4; ++nn) {
            double s = 0.0;
            for (int kk = 0; kk < K; ++kk) s += (double)dq0[kk] * (double)hB[(size_t)kk*N + nn];
            printf("  ref[0][%d] = %.3f  kernel = %.3f\n", nn, (float)s, hC[nn]);
        }
        printf("  A[0][0..3] = %.4f %.4f %.4f %.4f | B[0..3][0] = %.4f %.4f %.4f %.4f\n",
               dq0[0], dq0[1], dq0[2], dq0[3], (float)hB[0], (float)hB[N], (float)hB[2*N], (float)hB[3*N]);
        std::vector<float> refg((size_t)4*4);
        for (int m = 0; m < 4; ++m) {
            std::vector<float> dqm((size_t)stride_blocks*QK_K);
            for (int kb = 0; kb < stride_blocks; ++kb) dequant_q6k_block(&hA[(size_t)m*stride_blocks+kb], dqm.data()+(size_t)kb*QK_K);
            for (int nn = 0; nn < 4; ++nn) {
                double s = 0.0;
                for (int kk = 0; kk < K; ++kk) s += (double)dqm[kk]*(double)hB[(size_t)kk*N+nn];
                refg[m*4+nn] = (float)s;
            }
        }
        printf("  ref grid (m rows, n cols):\n");
        for (int m = 0; m < 4; ++m) printf("    %8.3f %8.3f %8.3f %8.3f\n", refg[m*4+0], refg[m*4+1], refg[m*4+2], refg[m*4+3]);
        printf("  kernel grid:\n");
        for (int m = 0; m < 4; ++m) printf("    %8.3f %8.3f %8.3f %8.3f\n", hC[(size_t)m*N+0], hC[(size_t)m*N+1], hC[(size_t)m*N+2], hC[(size_t)m*N+3]);
    }

    // ---- timing ----
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0);
    for (int i = 0; i < iters; ++i) q6k_f16_gemm<<<grid, block>>>((block_q6_K *)dA, stride_blocks, (_Float16 *)dB, (float *)dC, M, N, K);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);
    const double flop = 2.0 * (double)M * N * K;
    printf("q6k_f16_gemm M=%d N=%d K=%d : %.3f ms/run  %.2f TFLOPS  (iters=%d)\n",
           M, N, K, ms / iters, flop / (ms / iters / 1000.0) / 1e12, iters);

    hipFree(dA); hipFree(dB); hipFree(dC);
    return 0;
}
