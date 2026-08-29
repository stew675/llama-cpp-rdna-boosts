#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

// ---------------------------------------------------------------------------------------------
// Fused chunked GGML_OP_GATED_DELTA_NET prefill on the gfx1201 bf16 WMMA units.
// RDNA4 (gfx12) ONLY: this is the fork's validated RDNA4-only kernel file
// (chunked-gdn tip 99bbd1b20, 46/46 on gfx1201 per the d222767c7 record), restored
// verbatim after the in-place dual-arch refactor regressed the gfx12 path (NaN/inf
// on the bf16 chunked GDN, llama-cpp-rdna-boosts report). The gfx11 (RDNA3/RDNA3.5)
// first-gen WMMA port lives in its own file (gated_delta_net_chunked_bf16_gfx11.cu)
// with its own dispatch; the two architectures never share kernel code, so they
// cannot cross-regress. This file compiles to no-op stub kernels on non-RDNA4 device
// passes (the gfx12 wmma intrinsic does not exist there - issue #1); the runtime-cc
// dispatch in gated_delta_net.cu launches it only when the device is RDNA4.
//
// This is the tensor-core half of the chunked-GDN lever: the SAME chunked algorithm as the
// fp32 path (gated_delta_net_chunked.cu), but every GEMM runs on the RDNA4 WMMA instruction
// with bf16 operands and fp32 accumulation, and the recurrent state lives in WMMA
// accumulators across the whole chunk loop. Ported from libr4d's r4d_gdn_kkt_solve_k128_c64
// and r4d_gdn_chunk_scan_k128_v128_c64 (validated on this exact hardware against FLA), adapted
// to llama.cpp: fp32 tensor I/O (converted to bf16 on the fly), llama.cpp layouts, the
// h % H_k GQA mapping, and a compact bf16 A scratch.
//
// Accuracy model (the same as the fork's BF16 KV-cache flash-attn and PR#26001): bf16/FP16
// operands, fp32 accumulation, fp32 prefix sums, fp32 H-state carried across chunks, bf16
// only for the staged WMMA operands. Near-lossless, NOT bit-exact -- the dispatch in
// gated_delta_net.cu makes bf16 the S_v == 128 default on RDNA4 (opt-OUT via
// GGML_CUDA_GDN_CHUNKED_BF16=0); the bit-exact fp32 chunked path stays the fallback.
//
// Decay is carried on a per-chunk reference c = (g_first + g_last)/2 so the (i,j) factor
// separates exactly:
//     e^{g_i-g_j} = e^{g_i-c} * e^{c-g_j}
// Folding the left half into the score matrix (rs) and the right half into V' (gv) means the
// ungated Vd is never needed. Splitting at the midpoint bounds each half by e^{span/2}; the
// halves clamp at 80 (r4d's measured value -- a chunk span past ~176 would attenuate, which
// never happens with real gates). The KKT inverse (the kkt kernel) uses the DIRECT form
// e^{g_i-g_j}, which is provably <= 1 on its kept pairs.
//
//   ge[j]=e^{g_j}   rs[i]=scale*e^{g_i-c}   gv[j]=e^{c-g_j}   lam=e^{g_last-c}
//   U  = K@S            W  = beta*(V - ge*U)      V' = gv*(A@W)
//   P  = Q@K^T          sA = rs*tril(P)
//   O  = scale*ge*(Q@S) + sA@V'
//   S  = lam*(e^c*S + V'^T@K)
//
// Layouts (llama.cpp fp32 in/out, bf16 only in the WMMA staging):
//   q, k:  [S_v, H_k, n_tokens, n_seqs] contiguous rows; q/k share strides
//   v:     [S_v, H,   n_tokens, n_seqs] contiguous rows
//   g,beta:[1,  H,   n_tokens, n_seqs] contiguous (scalar-per-head gate)
//   state: [S_v, S_v, H, n_seqs] contiguous, stored transposed: state[v*S_v + k] = S[k][v]
//   out:   attn [S_v, H, n_tokens, n_seqs] followed by the new state (K == 1: one slot)
//   A_sc:  [n_chunks][H][n_seqs][64][64] bf16 row-major (the KKT inverse, unit-lower,
//          decay folded in). The scan's fragment reads use the r4d [token][head][64] stride
//          convention: A_sc[((chunk*64 + r)*H + h)*64 + c].
//
// The v-head h reads q/k from k-head h % H_k (llama.cpp's GQA mapping) and g/beta from h.
// Geometry is fixed to the target model: BT=64, KD=VD=S_v=128. Other S_v take the fp32 path.
// ---------------------------------------------------------------------------------------------

#include "gated_delta_net_chunked.cuh"
#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#define GDN_BF16_BT  64
#define GDN_BF16_KD  128
#define GDN_BF16_VD  128
#define GDN_BF16_BV  64                 // v-columns per scan workgroup
#define GDN_BF16_KP  132                // LDS row pitch of the [.][KD] buffers, in shorts
#define GDN_BF16_AP  68                 // LDS row pitch of the [.][BT] buffers, in shorts
#define GDN_BF16_WP  GDN_BF16_AP
#define GDN_BF16_NW  8                  // waves per scan workgroup
#define GDN_BF16_NTHR (GDN_BF16_NW * 32)
#define GDN_BF16_NTV 2                  // n tiles per wave in the scan's main / V' / O loops
#define GDN_BF16_SKT 2                  // state k-tiles per wave (2x2 over the 8x4 grid)
#define GDN_BF16_SVT 2
#define GDN_BF16_NST (GDN_BF16_SKT * GDN_BF16_SVT)
#define GDN_BF16_KKT_NW 4               // waves per kkt workgroup
#define GDN_BF16_KKT_NTHR (GDN_BF16_KKT_NW * 32)

#if defined(RDNA4)

// ---------------------------------------------------------------------------------------------
// gfx1201 wave32 bf16 WMMA fragment helpers (libr4d's r4d_gdn_wmma.h, layouts probed on this
// hardware, not documented by AMD):
//   16-bit A/B : idx = lane%16 , k = 8*(e>>2) + 4*(lane>>4) + (e&3)   ("two runs of four")
//   f32   C/D  : n   = lane%16 , m = 8*(lane>>4) + e
// A wants [M][K] and B wants [N][K] -- both K-contiguous -- and mma(X, Y) = X . Y^T.
// ---------------------------------------------------------------------------------------------
typedef __bf16 gdn_v8bf __attribute__((ext_vector_type(8)));
typedef float  gdn_v8f  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ unsigned short gdn_f2bf(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
__device__ __forceinline__ unsigned gdn_f2bf2(float a, float b) {
    unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
    unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
    return __builtin_amdgcn_perm(ub, ua, 0x07060302u);   // {b_hi16, a_hi16}
}
__device__ __forceinline__ uint2 gdn_f2bf4(float a, float b, float c, float d) {
    return make_uint2(gdn_f2bf2(a, b), gdn_f2bf2(c, d));
}
__device__ __forceinline__ float gdn_bf2f(unsigned short h) {
    return __builtin_bit_cast(float, (unsigned) h << 16);
}

// K-contiguous fragment from a row pointer already advanced to this lane's row and half-K.
// The four dwords are two 8-BYTE groups 16 bytes apart: read as uint2 to get one
// ds_load_2addr_b64 per group (LLVM pairs dwords only when the pointer proves 8-byte aligned,
// which every fragment address here is).
__device__ __forceinline__ gdn_v8bf gdn_fragP(const unsigned short * p) {
    const uint2 * pw = (const uint2 *) p;
    union { uint2 w[2]; gdn_v8bf f; } u;
    u.w[0] = pw[0]; u.w[1] = pw[2];
    return u.f;
}
// Row pointer for the fragment above: LDS source, `pitch` in elements.
__device__ __forceinline__ const unsigned short * gdn_rowP(const unsigned short * src,
                                                           int pitch, int base, int lane) {
    return src + (base + (lane & 15)) * pitch + 4 * (lane >> 4);
}
// Fragment from a row pointer already advanced; `pitch` in elements.
__device__ __forceinline__ gdn_v8bf gdn_frag(const unsigned short * src, int pitch,
                                             int base, int kbase, int lane) {
    return gdn_fragP(gdn_rowP(src, pitch, base, lane) + kbase);
}
// 32-BIT element offset for a global fragment source, split from a UNIFORM base. Keeping the
// offset in one 32-bit register lets the whole k loop be global_load_b64 with immediates.
__device__ __forceinline__ unsigned gdn_rowGo(unsigned rowstride, int row0, int lim, int lane) {
    int r = row0 + (lane & 15); r = r < lim ? r : lim;
    return (unsigned) r * rowstride + 4u * (unsigned) (lane >> 4);
}
__device__ __forceinline__ gdn_v8bf gdn_fragGo(const unsigned short * base, unsigned off) {
    return gdn_fragP(base + off);
}
// Same, but the source is fp32: 8 floats at the element offset, packed to bf16. NOTE: the
// fragment layout is the "two runs of four" -- registers 0..3 = k 0..3, registers 4..7 =
// k 8..11 -- so the source reads are pf[0] (elements 0..3) and pf[2] (elements 8..11), NOT
// pf[1]! The LDS version (fragP) reads pw[0] and pw[2] for the same reason; a contiguous
// pf[1] read feeds the wrong k's into the WMMA and scrambles every contraction (the P gram
// and the O state term came out ~0.53x before this was fixed).
__device__ __forceinline__ gdn_v8bf gdn_fragGo32(const float * base, unsigned off) {
    const float4 * pf = (const float4 *) (base + off);
    union { uint2 w[2]; gdn_v8bf f; } u;
    u.w[0] = gdn_f2bf4(pf[0].x, pf[0].y, pf[0].z, pf[0].w);
    u.w[1] = gdn_f2bf4(pf[2].x, pf[2].y, pf[2].z, pf[2].w);
    return u.f;
}
// Fragment from a 32-BIT LDS BYTE OFFSET off the shared-block base (see libr4d's fragO).
__device__ __forceinline__ gdn_v8bf gdn_fragO(const char * sb, unsigned off) {
    return gdn_fragP((const unsigned short *) (sb + off));
}
#define GDN_LDSOPAQUE(x) asm("" : "+v"(x))
// byte offset of buffer[base + lo][4*hi] from the base of the shared block
#define GDN_ROWO(buf, pitch, base) \
    ((unsigned) ((const char *) &(buf)[0][0] - (const char *) &s) + \
     (unsigned) ((((base) + lo) * (pitch) + 4 * hi) * 2))

// Same fragment but the source is stored transposed: src[k][idx]. Costs 8 scalar reads.
__device__ __forceinline__ gdn_v8bf gdn_fragT(const unsigned short * src, int pitch,
                                              int base, int kbase, int lane) {
    int idx = base + (lane & 15);
    int k0  = kbase + 4 * (lane >> 4);
    union { unsigned short h[8]; gdn_v8bf f; } u;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        u.h[e]     = src[(size_t) (k0 + e) * pitch + idx];
        u.h[4 + e] = src[(size_t) (k0 + 8 + e) * pitch + idx];
    }
    return u.f;
}

__device__ __forceinline__ gdn_v8f gdn_mma(gdn_v8bf a, gdn_v8bf b, gdn_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// Barrier scoped to the LDS address space only: __syncthreads() is an ALL-address-space fence,
// so the barrier that follows a global store compiles to s_wait_storecnt_dscnt -- every wave
// waits for its output stores to reach cache before an LDS-only rendezvous.
#define GDN_BAR() do {                                                          \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");         \
        __builtin_amdgcn_s_barrier();                                           \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");         \
    } while (0)

// ---------------------------------------------------------------------------------------------
// kkt_solve (bf16): A = (I + strict_lower(diag(beta) . K K^T . e^{g_i-g_j}))^{-1} per chunk.
// The gram is computed ONCE per (chunk, k-head) via WMMA (upper tiles only -- the strictly
// lower half of A is read out of the upper tiles, the orientation where one accumulator row
// is one contiguous 16-byte LDS store). The blocked inverse is fp32 forward substitution for
// the four 16x16 diagonal blocks + two WMMA 32-merges + a WMMA 64-merge, using
// [[P,0],[R,Q]]^-1 = [[P^-1,0],[-Q^-1 R P^-1,Q^-1]]. Only the bf16 inverse is stored.
// grid (n_chunks, H_k, n_seqs); the HG_RATIO v-heads sharing a k-head loop inside the block.
// ---------------------------------------------------------------------------------------------
#define GDN_BF16_NBK 4                     // 16x16 blocks per chunk edge
#define GDN_BF16_NTILE 10                  // upper-triangular tiles of a 4x4 tiling
#define GDN_BF16_MAXT ((GDN_BF16_NTILE + GDN_BF16_KKT_NW - 1) / GDN_BF16_KKT_NW)

struct gdn_bf16_kkt_smem {
    unsigned short A[GDN_BF16_BT][GDN_BF16_AP];  // A on the way in, its inverse on the way out
    unsigned short TMP[32][36];                  // one 32x32 intermediate, transposed
    float gb[GDN_BF16_BT];                       // the chunk's raw gate
    float gcs[GDN_BF16_BT];                      // chunk-local inclusive cumsum
    float bt[GDN_BF16_BT];                       // beta
};

__device__ __forceinline__ void gdn_kkt_tile(int ti, int * mt, int * nt) {
    int m = 0, n = ti;
    while (n >= GDN_BF16_NBK - m) { n -= GDN_BF16_NBK - m; ++m; }
    *mt = m; *nt = n + m;
}

template <int HG_RATIO>
__global__ static void __launch_bounds__(GDN_BF16_KKT_NTHR)
gdn_bf16_kkt_cuda(
        const float * __restrict__ k,       // [S_v, H_k, T, N] fp32
        const float * __restrict__ g,       // [1, H, T, N] fp32
        const float * __restrict__ beta,    // [1, H, T, N] fp32
        unsigned short * __restrict__ A_sc, // [n_chunks][H][n_seqs][64][64] bf16
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int dbg)
{
    __shared__ gdn_bf16_kkt_smem s;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int lo = lane & 15, hi = lane >> 4;
    const int c  = blockIdx.x;      // chunk
    const int hg = blockIdx.y;      // k-head
    const int nq = blockIdx.z;      // seq
    const int c0 = c * GDN_BF16_BT;
    const int rows = min(GDN_BF16_BT, (int) n_tokens - c0);
    const int lim = rows - 1;

    const float * kp = k + (int64_t) nq * sq3 + (int64_t) c0 * sq2 + (int64_t) hg * sq1;
    const unsigned krow = (unsigned) sq2;   // token stride, in float elements (permuted-safe)

    // ---- zero the parts no store ever reaches: the upper tiles of A and the TMP slack ----
    for (int i = tid; i < GDN_BF16_BT * GDN_BF16_AP / 8; i += GDN_BF16_KKT_NTHR)
        ((uint4 *) &s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

    // ---- the gram P[m][n] = sum_d K[mt*16+m][d] K[nt*16+n][d], upper tiles only -------------
    gdn_v8f P[GDN_BF16_MAXT];
#pragma unroll
    for (int t = 0; t < GDN_BF16_MAXT; ++t) P[t] = (gdn_v8f) {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < GDN_BF16_MAXT; ++t) {
        const int ti = warp + t * GDN_BF16_KKT_NW;
        if (ti < GDN_BF16_NTILE) {
            int mt, nt; gdn_kkt_tile(ti, &mt, &nt);
            // One 32-bit element offset per operand row, hoisted out of the k loop.
            const unsigned om = gdn_rowGo(krow, mt * 16, lim, lane);
            const unsigned on = gdn_rowGo(krow, nt * 16, lim, lane);
#pragma unroll
            for (int sk = 0; sk < GDN_BF16_KD / 16; ++sk)
                P[t] = gdn_mma(gdn_fragGo32(kp, om + sk * 16),
                               gdn_fragGo32(kp, on + sk * 16), P[t]);
        }
    }

    for (int hh = 0; hh < HG_RATIO; ++hh) {
        const int hv = hg + hh * (int) H_k;  // llama.cpp GQA: v-heads {hg, hg+H_k, ...} share k-head hg

        // ---- per-head gate and beta for this chunk (flat cumsum on the padding) -------------
        if (tid < GDN_BF16_BT) {
            const int64_t o = ((int64_t) nq * n_tokens + c0 + (int64_t) min(tid, rows - 1)) * H + hv;
            if (tid < rows) {
                s.gb[tid] = g[o];
                s.bt[tid] = beta[o];
            } else {
                s.gb[tid] = 0.0f;
                s.bt[tid] = 0.0f;
            }
        }
        GDN_BAR();
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_BF16_BT; ++t) { acc += s.gb[t]; s.gcs[t] = acc; }
        }
        GDN_BAR();

        // ---- scale, mask and place the gram as the strictly lower half of A ------------------
        // Tile (mt, nt) holds P[m][n]; the element written is A[n][m] = bt[n] gv[m] P[m][n],
        // in the lower triangle exactly when n > m. n is fixed per lane and m runs over the
        // eight accumulator elements, so this is one row of A and one 16-byte store.
#pragma unroll
        for (int t = 0; t < GDN_BF16_MAXT; ++t) {
            const int ti = warp + t * GDN_BF16_KKT_NW;
            if (ti < GDN_BF16_NTILE) {
                int mt, nt; gdn_kkt_tile(ti, &mt, &nt);
                const int i  = nt * 16 + lo;                  // row of A
                const int j0 = mt * 16 + 8 * hi;              // first column of A
                const bool live = (i < rows);
                const float bi = live ? s.bt[i] : 0.0f;
                const float gi = live ? s.gcs[i] : 0.0f;
                float vv[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = j0 + e;
                    // i > j means g_i <= g_j, so this exponent is never positive -- but ONLY
                    // for a live row. On a padding row gi is forced to 0 while gb[j] keeps its
                    // real, negative cumsum, so gi - gb[j] is large POSITIVE and __expf
                    // overflows to +INF: bi * INF = 0 * INF = NaN. Those NaNs sit in padding
                    // rows and the blocked inverse merges the whole tile with WMMA, so they
                    // reach live rows. (r4d's measured Qwen3.8-27B bug, documented in the kkt.)
                    const float d = (live && i > j && j < rows) ? (gi - s.gcs[j]) : -INFINITY;
                    vv[e] = bi * __expf(d) * P[t][e];
                }
                *(uint2 *) (&s.A[i][j0])     = gdn_f2bf4(vv[0], vv[1], vv[2], vv[3]);
                *(uint2 *) (&s.A[i][j0 + 4]) = gdn_f2bf4(vv[4], vv[5], vv[6], vv[7]);
            }
        }
        GDN_BAR();

        // ---- 1. the four diagonal blocks, by fp32 forward substitution ----------------------
        // D = (I + N)^-1 solves D = I - N D one row at a time. Lane l owns column l of D, so
        // the dot product over j < i is 16 registers deep and N[i][j] is a broadcast LDS read.
        for (int p = warp; p < GDN_BF16_NBK; p += GDN_BF16_KKT_NW) {
            const int b0 = p * 16;
            float d[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    if (j < i) acc += gdn_bf2f(s.A[b0 + i][b0 + j]) * d[j];
                d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
            }
            if (lane < 16) {
#pragma unroll
                for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = gdn_f2bf(d[i]);
            }
        }
        GDN_BAR();

        // ---- 2. the two 32-merges: M10 = -D1 A10 D0 ----------------------------------------
        // X = A10 . D0     (left A10 row-major, right D0 transposed: fragT of the diagonal)
        // Y = -D1 . X      (left D1 row-major, right X transposed: the TMP staging)
        for (int blk = warp; blk < 2; blk += GDN_BF16_KKT_NW) {
            const int r0 = blk * 32;                            // 32-block origin
            gdn_v8f X = (gdn_v8f) {0, 0, 0, 0, 0, 0, 0, 0};
            X = gdn_mma(gdn_frag(&s.A[0][0], GDN_BF16_AP, r0 + 16, r0, lane),
                        gdn_fragT(&s.A[0][0], GDN_BF16_AP, r0, r0, lane), X);
            // stage X transposed for the next product. The two merges run on two waves against
            // the same scratch, so each owns 16 rows of it.
            *(uint2 *) (&s.TMP[blk * 16 + lo][8 * hi])     = gdn_f2bf4(X[0], X[1], X[2], X[3]);
            *(uint2 *) (&s.TMP[blk * 16 + lo][8 * hi + 4]) = gdn_f2bf4(X[4], X[5], X[6], X[7]);
            // fragT reads M column-wise below, so the merge results only need the row-major copy
            __builtin_amdgcn_wave_barrier();
            gdn_v8f Y = (gdn_v8f) {0, 0, 0, 0, 0, 0, 0, 0};
            Y = gdn_mma(gdn_frag(&s.A[0][0], GDN_BF16_AP, r0 + 16, r0 + 16, lane),
                        gdn_frag(&s.TMP[0][0], 36, blk * 16, 0, lane), Y);
            const int i = r0 + 16 + 8 * hi;                    // rows of M10
            const int j = r0 + lo;                             // column of M10
#pragma unroll
            for (int e = 0; e < 8; ++e) s.A[i + e][j] = gdn_f2bf(-Y[e]);
        }
        GDN_BAR();

        // ---- 3. the 64-merge: M[32:,:32] = -Q^-1 (R P^-1) -----------------------------------
        // Y = R . P^-1, R = A[32:,:32] row-major, P^-1 transposed = TMP[:32,:32]
        for (int t = warp; t < 4; t += GDN_BF16_KKT_NW) {
            const int mt = t >> 1, nt = t & 1;
            gdn_v8f Y = (gdn_v8f) {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Y = gdn_mma(gdn_frag(&s.A[0][0], GDN_BF16_AP, 32 + mt * 16, sk * 16, lane),
                            gdn_fragT(&s.A[0][0], GDN_BF16_AP, nt * 16, sk * 16, lane), Y);
            *(uint2 *) (&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi])     = gdn_f2bf4(Y[0], Y[1], Y[2], Y[3]);
            *(uint2 *) (&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi + 4]) = gdn_f2bf4(Y[4], Y[5], Y[6], Y[7]);
        }
        GDN_BAR();   // every wave has read R into TMP; now it may be overwritten
        // Z = -Q^-1 . Y, Q^-1 = A[32:,32:] row-major, Y transposed = TMP
        for (int t = warp; t < 4; t += GDN_BF16_KKT_NW) {
            const int mt = t >> 1, nt = t & 1;
            gdn_v8f Z = (gdn_v8f) {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int sk = 0; sk < 2; ++sk)
                Z = gdn_mma(gdn_frag(&s.A[0][0], GDN_BF16_AP, 32 + mt * 16, 32 + sk * 16, lane),
                            gdn_frag(&s.TMP[0][0], 36, nt * 16, sk * 16, lane), Z);
            const int i = 32 + mt * 16 + 8 * hi;
            const int j = nt * 16 + lo;
#pragma unroll
            for (int e = 0; e < 8; ++e) s.A[i + e][j] = gdn_f2bf(-Z[e]);
        }
        GDN_BAR();

        // ---- store: one row of the inverse is 64 bf16, contiguous in HBM --------------------
        {
            unsigned short * op = A_sc + ((int64_t) (c * GDN_BF16_BT) * H + hv) * GDN_BF16_BT
                                      + (int64_t) nq * (GDN_BF16_BT * GDN_BF16_BT * H);
            const size_t orow = (size_t) H * GDN_BF16_BT;
            for (int r = tid / 8; r < rows; r += GDN_BF16_KKT_NTHR / 8) {
                const int c8 = (tid & 7) * 8;
                *(uint4 *) (op + (size_t) r * orow + c8) = *(const uint4 *) (&s.A[r][c8]);
            }
        }
        if (hh + 1 < HG_RATIO) GDN_BAR();
    }
}

// ---------------------------------------------------------------------------------------------
// chunk_scan (bf16): the recurrent state lives in WMMA accumulators across the whole chunk
// loop. grid (VD/BV, H, n_seqs); each block owns a BV=64-wide v slice of one v-head.
// ---------------------------------------------------------------------------------------------
struct gdn_bf16_scan_smem {
    unsigned short k[GDN_BF16_BT][GDN_BF16_KP];   // A: K / B: K^T for P = Q@K^T
    unsigned short A[GDN_BF16_BT][GDN_BF16_AP];   // B-source for sA (written as [i][j])
    unsigned short S[GDN_BF16_BV][GDN_BF16_KP];   // B: state, V-major (matches FLA's [V,K])
    unsigned short W[GDN_BF16_BV][GDN_BF16_WP];   // B: W (transposed)
    unsigned short D[GDN_BF16_BV][GDN_BF16_AP];   // staged V as [token][v]; then V' as [v][token]
    float ge[GDN_BF16_BT], gv[GDN_BF16_BT], rs[GDN_BF16_BT], bt[GDN_BF16_BT];
    float gcs[GDN_BF16_BT];
};

#define GDN_BF16_PFW 8                   // shorts staged per lane per step
#define GDN_BF16_PFV_T uint4
#define GDN_BF16_PFZERO make_uint4(0u, 0u, 0u, 0u)
#define GDN_BF16_NKB (GDN_BF16_BT * GDN_BF16_KD / (GDN_BF16_NTHR * GDN_BF16_PFW))
#define GDN_BF16_NVB (GDN_BF16_BT * GDN_BF16_BV / (GDN_BF16_NTHR * GDN_BF16_PFW))

__global__ static void __launch_bounds__(GDN_BF16_NTHR)
gdn_bf16_scan_cuda(
        const float * __restrict__ q,       // [S_v, H_k, T, N] fp32
        const float * __restrict__ k,       // [S_v, H_k, T, N] fp32
        const float * __restrict__ v,       // [S_v, H, T, N] fp32
        const float * __restrict__ g,       // [1, H, T, N] fp32
        const float * __restrict__ beta,    // [1, H, T, N] fp32
        const float * __restrict__ state_in,// [S_v, S_v, H, N] fp32, transposed: state[v][k]
        float * __restrict__ attn_out,      // [S_v, H, T, N] fp32
        float * __restrict__ state_out,     // [S_v, S_v, H, N] fp32
        const unsigned short * __restrict__ A_sc,   // [n_chunks][H][n_seqs][64][64] bf16
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_seq_stride, int dbg)
{
    __shared__ gdn_bf16_scan_smem s;
    const int tid = threadIdx.x;
    const int bv = blockIdx.x, h = blockIdx.y, nq = blockIdx.z;
    const int w = tid >> 5, lane = tid & 31;
    const int lo = lane & 15, hi = lane >> 4;
    const int v0 = bv * GDN_BF16_BV;
    const int nchunk = (int) ((n_tokens + GDN_BF16_BT - 1) / GDN_BF16_BT);

    // llama.cpp GQA mapping: v-head h reads q/k from k-head h % H_k (and the seq index iq3)
    const int64_t hq  = fastmodulo((uint32_t) h, neqk1_magic);
    const int64_t iq3 = fastdiv((uint32_t) nq, rq3_magic);
    const int64_t state_seq_base = (int64_t) nq * state_seq_stride + (int64_t) h * GDN_BF16_VD * GDN_BF16_KD;

    // wave tile mapping: [BT][BV] = 4x4 tile grid, 2 tiles/wave; state 2x2 over the 8x4 grid
    const int mt  = w >> 1, ntb = (w & 1) * 2;
    const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;

    // ---- the state accumulator: C[m=k-dim][n=v], transposed so the writeback to s.S is
    // ---- four packed 8-short stores instead of 32 scalar ds_store_b16 -----------------------
    gdn_v8f St[GDN_BF16_NST];
#define KT(t) (kt0 + (t) / GDN_BF16_SVT)
#define VT(t) (vt0 + (t) % GDN_BF16_SVT)
#define SROW(t) ((size_t) (v0 + VT(t) * 16 + lo) * GDN_BF16_KD + KT(t) * 16 + 8 * hi)
    {
        const float * p = state_in + state_seq_base;
#pragma unroll
        for (int t = 0; t < GDN_BF16_NST; ++t) {
            const float4 a = *(const float4 *) (p + SROW(t));
            const float4 b = *(const float4 *) (p + SROW(t) + 4);
            St[t][0]=a.x; St[t][1]=a.y; St[t][2]=a.z; St[t][3]=a.w;
            St[t][4]=b.x; St[t][5]=b.y; St[t][6]=b.z; St[t][7]=b.w;
        }
    }
#define SWRITE()                                                                                  \
    do {                                                                                          \
        _Pragma("unroll") for (int t = 0; t < GDN_BF16_NST; ++t) {                                \
            unsigned short * sp = &s.S[VT(t) * 16 + lo][KT(t) * 16 + 8 * hi];                     \
            *(uint2 *) sp       = gdn_f2bf4(St[t][0], St[t][1], St[t][2], St[t][3]);              \
            *(uint2 *) (sp + 4) = gdn_f2bf4(St[t][4], St[t][5], St[t][6], St[t][7]);              \
        }                                                                                         \
    } while (0)
    SWRITE();

    // ---- staging indices: WAVE-UNIFORM rows + loop-invariant lane columns (r4d PFV=4) -------
    const int wu = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int kc = (tid * GDN_BF16_PFW) & (GDN_BF16_KD - 1), vc = (tid * GDN_BF16_PFW) & (GDN_BF16_BV - 1);
    const int kr0 = 2 * wu + ((tid >> 4) & 1), vr0 = 4 * wu + ((tid >> 3) & 3);
#define GDN_BF16_KRU (2 * GDN_BF16_NW)
#define GDN_BF16_VRU (4 * GDN_BF16_NW)
    GDN_BF16_PFV_T kb[GDN_BF16_NKB], vb[GDN_BF16_NVB];
    const unsigned krs = (unsigned) sq2, vrs = (unsigned) sv2;   // token strides (permuted-safe)
    unsigned koff0[GDN_BF16_NKB], voff0[GDN_BF16_NVB];
    {
        _Pragma("unroll") for (int u = 0; u < GDN_BF16_NKB; ++u)
            koff0[u] = (unsigned) (kr0 + GDN_BF16_KRU * u) * krs + (unsigned) kc;
        _Pragma("unroll") for (int u = 0; u < GDN_BF16_NVB; ++u)
            voff0[u] = (unsigned) (vr0 + GDN_BF16_VRU * u) * vrs + (unsigned) vc;
    }
    // the chunk-uniform half: clamp the row reads to the last real token
#define GDN_BF16_PFBASE(CH)                                                                       \
        const int pc0_ = (CH) * GDN_BF16_BT, prl_ = min(GDN_BF16_BT, (int) n_tokens - pc0_) - 1;  \
        const float * kb_ = k + (int64_t) iq3 * sq3 + (int64_t) pc0_ * sq2 + hq * sq1;            \
        const float * vb_ = v + (int64_t) nq * sv3 + (int64_t) pc0_ * sv2 + (int64_t) h * sv1 + v0; \
        const unsigned kcl = (unsigned) prl_ * krs + (unsigned) kc;                               \
        const unsigned vcl = (unsigned) prl_ * vrs + (unsigned) vc;
#define GDN_BF16_PFONE(u)                                                                         \
        do {                                                                                      \
            if ((u) < GDN_BF16_NKB) {                                                             \
                const unsigned koff = min(koff0[(u)], kcl);                                       \
                const float4 * pf = (const float4 *) (kb_ + koff);                                \
                kb[(u)] = make_uint4(gdn_f2bf2(pf[0].x, pf[0].y), gdn_f2bf2(pf[0].z, pf[0].w),    \
                                     gdn_f2bf2(pf[1].x, pf[1].y), gdn_f2bf2(pf[1].z, pf[1].w));   \
            } else if ((u) - GDN_BF16_NKB < GDN_BF16_NVB) {                                       \
                const unsigned voff = min(voff0[(u) - GDN_BF16_NKB], vcl);                        \
                const float4 * pf = (const float4 *) (vb_ + voff);                                \
                vb[(u) - GDN_BF16_NKB] = make_uint4(gdn_f2bf2(pf[0].x, pf[0].y),                  \
                                                    gdn_f2bf2(pf[0].z, pf[0].w),                  \
                                                    gdn_f2bf2(pf[1].x, pf[1].y),                  \
                                                    gdn_f2bf2(pf[1].z, pf[1].w));                 \
            }                                                                                     \
        } while (0)
#define GDN_BF16_PREFETCH(CH)                                                                     \
        do {                                                                                      \
            GDN_BF16_PFBASE(CH)                                                                   \
            _Pragma("unroll") for (int u = 0; u < GDN_BF16_NKB + GDN_BF16_NVB; ++u)               \
                GDN_BF16_PFONE(u);                                                                \
        } while (0)
#define GDN_BF16_PFSTORE(dst, val)                                                                \
        do {                                                                                      \
            unsigned short * dp_ = (dst);                                                         \
            *(uint2 *) dp_ = make_uint2((val).x, (val).y);                                        \
            *(uint2 *) (dp_ + 4) = make_uint2((val).z, (val).w);                                  \
        } while (0)

    GDN_BAR();
    GDN_BF16_PREFETCH(0);

    const char * const sbase = (const char *) &s;
    for (int ci = 0; ci < nchunk; ++ci) {
        const int c0 = ci * GDN_BF16_BT, nval = min(GDN_BF16_BT, (int) n_tokens - c0);
        const int lim = nval - 1;
        const unsigned short * ag = A_sc + ((int64_t) c0 * H + h) * GDN_BF16_BT
                                        + (int64_t) nq * (GDN_BF16_BT * GDN_BF16_BT * H);
        const size_t ars = (size_t) H * GDN_BF16_BT;
        const float * qg = q + (int64_t) iq3 * sq3 + (int64_t) c0 * sq2 + hq * sq1;
        const size_t qrs = (size_t) sq2;   // token stride of q (permuted-safe)
        const float * gbase = g + ((int64_t) nq * n_tokens + c0) * H + h;
        const float * bbase = beta + ((int64_t) nq * n_tokens + c0) * H + h;
        float * obase = attn_out + ((int64_t) nq * n_tokens + c0) * H * GDN_BF16_VD + (int64_t) h * GDN_BF16_VD;

        // ---- gates: chunk-local inclusive cumsum, then ge/gv/rs/bt (flat on padding) --------
        if (tid < GDN_BF16_BT) {
            if (tid < nval) {
                s.bt[tid] = bbase[(int64_t) tid * H];
                s.gcs[tid] = 0.0f;   // placeholder; filled by the cumsum
                s.ge[tid] = gbase[(int64_t) tid * H];
            } else {
                s.bt[tid] = 0.0f;
                s.ge[tid] = 0.0f;
                s.gcs[tid] = 0.0f;
            }
        }
        GDN_BAR();
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_BF16_BT; ++t) { acc += s.ge[t]; s.gcs[t] = acc; }
        }
        GDN_BAR();
        if (tid < GDN_BF16_BT) {
            const float gcs_t = s.gcs[tid];
            const bool ok = tid < nval;
            s.ge[tid] = __expf(gcs_t);
            // DIRECT decay (no chunk-midpoint split, no clamp): el = e^{g_last - g_t} and
            // rs = scale*e^{g_t - g_last} are provably <= 1 for every t (gates are negative,
            // padding keeps the cumsum flat), so they can never overflow or attenuate -- the
            // same form the fp32 chunked path uses. r4d's split form (e^{g-c}, e^{c-g} with an
            // 80-clamp) is exact for real-model spans (<= 200) but attenuates the state path
            // catastrophically for the test harness's pathological gates (spans ~ 640+): the
            // last token's own e^{g_last-g_t} ~ 1 weight becomes e^{80-(c-g_t)} ~ e^-240.
            s.gv[tid] = ok ? __expf(s.gcs[nval - 1] - gcs_t) : 0.0f;
            s.rs[tid] = ok ? scale * __expf(gcs_t - s.gcs[nval - 1]) : 0.0f;
        }
        GDN_BAR();

        // consume the previous iteration's prefetch: stage K/V to LDS (bf16)
        {
            _Pragma("unroll") for (int u = 0; u < GDN_BF16_NKB; ++u)
                GDN_BF16_PFSTORE(&s.k[kr0 + GDN_BF16_KRU * u][kc], kb[u]);
            _Pragma("unroll") for (int u = 0; u < GDN_BF16_NVB; ++u)
                GDN_BF16_PFSTORE(&s.D[vr0 + GDN_BF16_VRU * u][vc], vb[u]);
        }
        GDN_BAR();                                                   // (1)
        GDN_BF16_PREFETCH(min(ci + 1, nchunk - 1));                  // issue next chunk's loads

        const float glast = s.ge[nval - 1];   // e^{g_last} (the state carry factor)

        // ---- U = K@S, O^T = S@Q^T and P^T = K@Q^T share the q and k fragments --------------
        gdn_v8f O[GDN_BF16_NTV], u[GDN_BF16_NTV], p[GDN_BF16_NTV];
#pragma unroll
        for (int i = 0; i < GDN_BF16_NTV; ++i) {
            O[i] = (gdn_v8f) {}; u[i] = (gdn_v8f) {}; p[i] = (gdn_v8f) {};
        }
        {
            const char * sb = sbase;
            unsigned oSn[GDN_BF16_NTV], okn[GDN_BF16_NTV];
            unsigned okm = GDN_ROWO(s.k, GDN_BF16_KP, mt * 16);
            GDN_LDSOPAQUE(okm);
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i) {
                oSn[i] = GDN_ROWO(s.S, GDN_BF16_KP, (ntb + i) * 16);
                okn[i] = GDN_ROWO(s.k, GDN_BF16_KP, (ntb + i) * 16);
                GDN_LDSOPAQUE(oSn[i]); GDN_LDSOPAQUE(okn[i]);
            }
#define pSnf(i, ko) gdn_fragO(sb, oSn[i] + (ko) * 2u)
#define pkmf(ko)    gdn_fragO(sb, okm + (ko) * 2u)
#define pknf(i, ko) gdn_fragO(sb, okn[i] + (ko) * 2u)
            const unsigned qo = gdn_rowGo((unsigned) qrs, mt * 16, lim, lane);
#define pqf(ko) gdn_fragGo32(qg, qo + (ko))
#pragma unroll
            for (int sk = 0; sk < GDN_BF16_KD / 16; ++sk) {
                const int ko = sk * 16;
                gdn_v8bf ak = pkmf(ko), aq = pqf(ko);
#pragma unroll
                for (int i = 0; i < GDN_BF16_NTV; ++i) {
                    gdn_v8bf b = pSnf(i, ko);
                    u[i] = gdn_mma(ak, b, u[i]);                          // U: m = token
                    O[i] = gdn_mma(b, aq, O[i]);                          // O^T: m = v
                    p[i] = gdn_mma(pknf(i, ko), aq, p[i]);                // P^T: m = j
                }
                
            }
            // group the k loop: per 2 steps ask for the reads against the previous WMMAs
#pragma unroll
            for (int i = 0; i < GDN_BF16_KD / (16 * 2); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x020, 2 * 2, 0);    // VMEM q reads
                __builtin_amdgcn_sched_group_barrier(0x100, (1 + 2 * GDN_BF16_NTV) * 2, 0);  // LDS
                __builtin_amdgcn_sched_group_barrier(0x008, (3 * GDN_BF16_NTV) * 2, 0);      // WMMA
            }
#undef pSnf
#undef pkmf
#undef pknf
#undef pqf
        }
        {
            float g = scale * s.ge[mt * 16 + lo];   // token index is constant per lane
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e) O[i][e] *= g;
            
        }

        float mxU = 0.0f;
        // ---- W = beta * (V - ge*U)  (beta folded here: (A*beta_col)@W == A@(beta_row*W)) ----
        {
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i)
#pragma unroll
                for (int e = 0; e < 8; ++e) mxU = fmaxf(mxU, fabsf(u[i][e]));

            float wa[GDN_BF16_NTV][8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {           // beta / ge read once for both tiles
                const int tok = mt * 16 + 8 * hi + e;
                const float f = s.bt[tok], g = s.ge[tok];
#pragma unroll
                for (int i = 0; i < GDN_BF16_NTV; ++i)
                    wa[i][e] = f * (gdn_bf2f(s.D[tok][(ntb + i) * 16 + lo]) - g * u[i][e]);
            }
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i) {
                unsigned short * wp = &s.W[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
                *(uint2 *) wp       = gdn_f2bf4(wa[i][0], wa[i][1], wa[i][2], wa[i][3]);
                *(uint2 *) (wp + 4) = gdn_f2bf4(wa[i][4], wa[i][5], wa[i][6], wa[i][7]);
            }
        }
        // ---- sA = scale * tril(P) with the DIRECT pair decay e^{g_i-g_j}: one exp per ------
        // ---- element (8 per tile), provably <= 1 (i >= j means g_i <= g_j). The r4d split  --
        // ---- form (e^{g_i-c} * e^{c-g_j}) overflows its halves at the test harness's       --
        // ---- pathological gate spans; the pair form cannot. --------------------------------
        {
            const int ii = mt * 16 + lo;            // now constant per lane
            const float gi = s.gcs[ii];
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i) {
                const int j0 = (ntb + i) * 16 + 8 * hi;
                float a[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int j = j0 + e;
                    a[e] = (j <= ii) ? p[i][e] * scale * __expf(gi - s.gcs[j]) : 0.0f;
                }
                *(uint2 *) &s.A[ii][j0]     = gdn_f2bf4(a[0], a[1], a[2], a[3]);
                *(uint2 *) &s.A[ii][j0 + 4] = gdn_f2bf4(a[4], a[5], a[6], a[7]);
            }
        }
        GDN_BAR();                                                   // (2)

        // ---- V' = gv * (A @ W) -> D (transposed store is one contiguous pair of dwords) ----
        {
            gdn_v8f d[GDN_BF16_NTV];
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i) d[i] = (gdn_v8f) {};
            const unsigned ao = gdn_rowGo((unsigned) ars, mt * 16, lim, lane);
#define paf(ko) gdn_fragGo(ag, ao + (ko))
            const unsigned short * pwn[GDN_BF16_NTV];
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i)
                pwn[i] = gdn_rowP(&s.W[0][0], GDN_BF16_WP, (ntb + i) * 16, lane);
#define pwnf(i, ko) gdn_fragP(pwn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < GDN_BF16_BT / 16; ++sk) {
                if (sk > mt) break;                 // A is unit lower triangular
                const int ko = sk * 16;
                gdn_v8bf a = paf(ko);
#pragma unroll
                for (int i = 0; i < GDN_BF16_NTV; ++i) d[i] = gdn_mma(a, pwnf(i, ko), d[i]);
            }
#pragma unroll
            for (int i = 0; i < GDN_BF16_BT / (16 * 2); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x020, 2 * 2, 0);   // A global reads
                __builtin_amdgcn_sched_group_barrier(0x100, GDN_BF16_NTV * 2, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, GDN_BF16_NTV * 2, 0);
            }
            float gvv[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) gvv[e] = 1.0f;   // V' is stored RAW; the decay lives in
                                                         // the sA (pair) and the state K scaling
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i) {
                unsigned short * dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
                *(uint2 *) dp       = gdn_f2bf4(d[i][0] * gvv[0], d[i][1] * gvv[1],
                                                d[i][2] * gvv[2], d[i][3] * gvv[3]);
                *(uint2 *) (dp + 4) = gdn_f2bf4(d[i][4] * gvv[4], d[i][5] * gvv[5],
                                                d[i][6] * gvv[6], d[i][7] * gvv[7]);
            }

#undef paf
#undef pwnf
        }
        GDN_BAR();                                                   // (3)

        // ---- O += sA @ V' ; store ------------------------------------------------------------
        {
            const unsigned short * pA = gdn_rowP(&s.A[0][0], GDN_BF16_AP, mt * 16, lane);
            const unsigned short * pdn[GDN_BF16_NTV];
#pragma unroll
            for (int i = 0; i < GDN_BF16_NTV; ++i)
                pdn[i] = gdn_rowP(&s.D[0][0], GDN_BF16_AP, (ntb + i) * 16, lane);
#define pAf(ko)     gdn_fragP(pA + (ko))
#define pdnf(i, ko) gdn_fragP(pdn[i] + (ko))
#pragma unroll
            for (int sk = 0; sk < GDN_BF16_BT / 16; ++sk) {
                if (sk > mt) break;                 // sA is masked to lower triangular
                const int ko = sk * 16;
                gdn_v8bf b = pAf(ko);
#pragma unroll
                for (int i = 0; i < GDN_BF16_NTV; ++i) O[i] = gdn_mma(pdnf(i, ko), b, O[i]);
            }

#pragma unroll
            for (int i = 0; i < GDN_BF16_BT / (16 * 2); ++i) {
                __builtin_amdgcn_sched_group_barrier(0x100, (1 + GDN_BF16_NTV) * 2, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, GDN_BF16_NTV * 2, 0);
            }
#undef pAf
#undef pdnf
        }
        {
            const int tok = mt * 16 + lo;
            if (tok < nval) {
                float * op = obase + (int64_t) tok * H * GDN_BF16_VD + v0;
#pragma unroll
                for (int i = 0; i < GDN_BF16_NTV; ++i) {
                    float * q2 = op + (ntb + i) * 16 + 8 * hi;
                    *(float4 *) q2       = make_float4(O[i][0], O[i][1], O[i][2], O[i][3]);
                    *(float4 *) (q2 + 4) = make_float4(O[i][4], O[i][5], O[i][6], O[i][7]);
                }
            }
            
        }

        // ---- S' = e^{g_last} * S + V''^T @ K  (V'' carries the direct e^{g_last-g_t} decay) --
        {
#pragma unroll
            for (int t = 0; t < GDN_BF16_NST; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) St[t][e] *= glast;
            const unsigned short * pvn[GDN_BF16_SVT];
#pragma unroll
            for (int j = 0; j < GDN_BF16_SVT; ++j)
                pvn[j] = gdn_rowP(&s.D[0][0], GDN_BF16_AP, (vt0 + j) * 16, lane);
#define pvnf(j, ko) gdn_fragP(pvn[j] + (ko))
#pragma unroll
            for (int sk = 0; sk < GDN_BF16_BT / 16; ++sk) {
                gdn_v8bf bv[GDN_BF16_SVT];
#pragma unroll
                for (int j = 0; j < GDN_BF16_SVT; ++j) bv[j] = pvnf(j, sk * 16);
#pragma unroll
                for (int ki = 0; ki < GDN_BF16_SKT; ++ki) {
                    // K^T [k][t] with the DIRECT decay el[t] = e^{g_last-g_t} folded into the
                    // operand (<= 1, no clamp): S' += sum_t el[t] * V'[t] * K[t]^T
                    gdn_v8bf ak = gdn_fragT(&s.k[0][0], GDN_BF16_KP, (kt0 + ki) * 16, sk * 16, lane);
                    {
                        // el[t] folded into the K^T operand, rebuilt via a union. NOTE: ak[e]
                        // element-indexed reads are MISCOMPILED on this compiler (every index
                        // reads element 0 -- the same bug class as the __bf16 assignment), so
                        // the el-scaled fragment must be assembled from the raw ushort bits.
                        const int k0 = sk * 16 + 4 * hi;
                        union { unsigned short h[8]; gdn_v8bf f; } ab, au;
                        ab.f = ak;
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            au.h[e] = gdn_f2bf(gdn_bf2f(ab.h[e]) * s.gv[k0 + e]);
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            au.h[4 + e] = gdn_f2bf(gdn_bf2f(ab.h[4 + e]) * s.gv[k0 + 8 + e]);
                        ak = au.f;
                    }
#pragma unroll
                    for (int j = 0; j < GDN_BF16_SVT; ++j)
                        St[ki * GDN_BF16_SVT + j] = gdn_mma(ak, bv[j], St[ki * GDN_BF16_SVT + j]);

                }
            }
#pragma unroll
            for (int i = 0; i < GDN_BF16_BT / 16; ++i) {
                __builtin_amdgcn_sched_group_barrier(0x100, (8 * GDN_BF16_SKT + GDN_BF16_SVT) * 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, GDN_BF16_NST * 1, 0);
            }
#undef pvnf
        }
        // no barrier: s.S aliases nothing, and barrier (2) already proves every wave finished
        // its U/O reads of s.S this chunk
        SWRITE();
        GDN_BAR();                                                   // (4)
        
    }

    // ---- write the final state: state[v][k] = S[k][v] ---------------------------------------
    {
        float * p = state_out + state_seq_base;
#pragma unroll
        for (int t = 0; t < GDN_BF16_NST; ++t) {
            *(float4 *) (p + SROW(t))     = make_float4(St[t][0], St[t][1], St[t][2], St[t][3]);
            *(float4 *) (p + SROW(t) + 4) = make_float4(St[t][4], St[t][5], St[t][6], St[t][7]);
        }
    }
#undef SWRITE
#undef KT
#undef VT
#undef SROW
}

#else // !defined(RDNA4)

// No-op stub definitions: the gfx12 wmma intrinsic does not exist on older targets (issue #1).
// The runtime-cc dispatch never launches these on non-RDNA4 hardware; they exist only so the
// host-pass launch stubs in the launchers below link on every arch.
template <int HG_RATIO>
__global__ static void gdn_bf16_kkt_cuda(
        const float * __restrict__ k, const float * __restrict__ g, const float * __restrict__ beta,
        unsigned short * __restrict__ A_sc, int64_t H, int64_t H_k, int64_t n_tokens,
        int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3, int dbg) {}
__global__ static void gdn_bf16_scan_cuda(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ state_in,
        float * __restrict__ attn_out, float * __restrict__ state_out,
        const unsigned short * __restrict__ A_sc, int64_t H, int64_t H_k, int64_t n_tokens,
        int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        uint3 neqk1_magic, uint3 rq3_magic, float scale, int64_t state_seq_stride, int dbg) {}

#endif // defined(RDNA4)


// ---------------------------------------------------------------------------------------------

template <int HG_RATIO>
static bool launch_gdn_bf16_kkt(
        const float * k_d, const float * g_d, const float * b_d, unsigned short * A_sc,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, cudaStream_t stream, int dbg) {
    const int n_chunks = (int) ((n_tokens + GDN_BF16_BT - 1) / GDN_BF16_BT);
    const dim3 grid((unsigned) n_chunks, (unsigned) H_k, (unsigned) n_seqs);
    const dim3 block(GDN_BF16_KKT_NTHR);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    return ggml_cuda_kernel_launch_try(gdn_bf16_kkt_cuda<HG_RATIO>, launch_params,
        k_d, g_d, b_d, A_sc, H, H_k, n_tokens, n_seqs, sq1, sq2, sq3, dbg);
}

static bool launch_gdn_bf16_scan(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, const unsigned short * A_sc,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_seq_stride, cudaStream_t stream, int dbg) {
    const dim3 grid((unsigned) (GDN_BF16_VD / GDN_BF16_BV), (unsigned) H, (unsigned) n_seqs);
    const dim3 block(GDN_BF16_NTHR);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    return ggml_cuda_kernel_launch_try(gdn_bf16_scan_cuda, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc,
        H, H_k, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
        neqk1_magic, rq3_magic, scale, state_seq_stride, dbg);
}

bool ggml_cuda_op_gated_delta_net_chunked_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;

    float * dst_d     = (float *) dst->data;
    // state_d_ext: fused-cache slot base when the GDN->cpy is fused (K == 1: single slot)
    float * state_d   = state_d_ext ? state_d_ext : dst_d + S_v * H * n_tokens * n_seqs;   // K == 1: single slot

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);
    const int64_t state_seq_stride = H * S_v * S_v;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(S_v == GDN_BF16_KD && S_v == GDN_BF16_VD);
    GGML_ASSERT(H % neqk1 == 0);

    const int64_t n_chunks = (n_tokens + GDN_BF16_BT - 1) / GDN_BF16_BT;
    ggml_cuda_pool_alloc<unsigned short> A_sc(ctx.pool(),
        (size_t) n_chunks * H * n_seqs * GDN_BF16_BT * GDN_BF16_BT);

    const int64_t hg_ratio = H / neqk1;
    const int dbg = getenv("GDN_DBG_P") != nullptr ? 1 : (getenv("GDN_DBG_ONES") != nullptr ? 2 : 0);
    switch (hg_ratio) {
        case 1:  if (!launch_gdn_bf16_kkt<1>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        case 2:  if (!launch_gdn_bf16_kkt<2>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        case 3:  if (!launch_gdn_bf16_kkt<3>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        case 4:  if (!launch_gdn_bf16_kkt<4>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        case 6:  if (!launch_gdn_bf16_kkt<6>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        case 8:  if (!launch_gdn_bf16_kkt<8>(k_d, g_d, b_d, A_sc.get(), H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, stream, dbg)) return false; break;
        default: GGML_ABORT("gated_delta_net_chunked_bf16: unsupported GQA ratio");
    }
    if (getenv("GDN_DBG_A") != nullptr) {
        // temp debug: dump the first A matrix per (chunk, head) for the first seq
        const size_t m = (size_t) std::min((int64_t) 2, n_chunks) * H * GDN_BF16_BT * GDN_BF16_BT;
        std::vector<unsigned short> dump(m);
        GGML_ASSERT(hipStreamSynchronize(stream) == hipSuccess);
        GGML_ASSERT(hipMemcpy(dump.data(), A_sc.get(), m * 2, hipMemcpyDeviceToHost) == hipSuccess);
        const auto h2f = [](unsigned short h) {
            union { unsigned u; float f; } x; x.u = (unsigned) h << 16; return x.f;
        };
        for (int64_t cc = 0; cc < 2 && cc < n_chunks; ++cc)
            for (int64_t hh = 0; hh < H; ++hh) {
                // layout: A_sc[((chunk*64 + r)*H + h)*64 + c]
                const auto at = [&](int r, int c) -> float {
                    return h2f(dump[((size_t) (cc * GDN_BF16_BT + r) * H + hh) * GDN_BF16_BT + c]);
                };
                if (hh < 4 || hh % 8 == 0)
                    printf("  A[c=%lld][h=%lld]: a00=%f a10=%f a11=%f a63=%f\n",
                           (long long) cc, (long long) hh, at(0, 0), at(1, 0), at(1, 1), at(63, 63));
            }
    }
    return launch_gdn_bf16_scan(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc.get(),
        H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3,
        scale, state_seq_stride, stream, dbg);
}

#endif // GGML_USE_HIP && __HIP_PLATFORM_AMD__
