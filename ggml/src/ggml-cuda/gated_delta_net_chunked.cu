#include "gated_delta_net_chunked.cuh"
#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// ---------------------------------------------------------------------------------------------
// Fused chunked GGML_OP_GATED_DELTA_NET for prefill (n_tokens > 1, K == 1, non-KDA).
//
// The chunked recurrence is sequential in the state, but the state is the ONLY sequential
// thing: everything else -- the KKT solve, the gram products, the attention output -- is
// embarrassingly parallel over (chunk, head). Two kernels split it:
//
//   kkt_solve: grid (n_chunks, H_k, n_seqs). One block per (chunk, k-head) computes the gram
//              K^T K ONCE in registers and forms the KKT inverse A for each of the
//              HG_RATIO = H_v/H_k v-heads that share that k-head, storing only the inverse
//              (fp32) to scratch. It also stages Q and stores the intra-chunk attention gram
//              K^T Q_s per k-head. Nothing here touches the state, so it is fully parallel.
//   chunk_scan: grid (S_v/16, H, n_seqs). Each block owns a 16-column slice of one v-head's
//              state and loops the chunks SEQUENTIALLY, keeping that state slice resident in
//              LDS across the whole loop -- the state is read from HBM once and written once,
//              never per chunk (the sequential kernel's per-token state round-trip is exactly
//              the traffic that made it HBM-bandwidth-bound). Per chunk it reads A and KQ_gram
//              from scratch, computes U, v_new, the attention output and the state update in
//              one pass.
//
// The per-chunk math is the algebra of llm_build_delta_net_base::build_delta_net_chunking,
// with gcs the chunk-local inclusive cumsum of the gate, K_b = K*beta, V_b = V*beta,
// Q_s = scale*Q, and decay[i][j] = e^{gcs[j]-gcs[i]} on the upper triangle (i <= j):
//
//     L    = strict_upper(K^T K_b . decay)                 (the gram, masked)
//     A    = (I + L)^-1                                    (unit upper; back substitution)
//     KQ   = upper_diag((K^T Q_s) . decay)
//     U    = K_b^T S
//     v_n  = V_b^T A - diag(e^g) A^T U                     (v_new = v_corr - predicted)
//     o    = e^g . S^T Q_s + KQ^T v_n
//     S'   = e^{g_last} S + K diag(e^{g_last - g}) v_n
//
// Decay is computed DIRECTLY per kept (i, j) pair, not split around a midpoint: the test
// harness feeds gates as low as -20 per token, so a chunk's cumsum can span thousands and any
// split-form half clamps. The direct exponent gcs[j]-gcs[i] is <= 0 on every kept pair (gates
// are negative; padding keeps the cumsum flat), so it can never overflow. Rows/columns past
// the sequence end are handled by clamping the token READ to the last real token (zero-padding
// would read out of bounds) and by bt[pad] = el[pad] = 0, which zero the gram entries and the
// state-update factors the way ggml_pad's zero columns do.
//
// Layouts (fp32, matching the sequential kernel):
//   q, k:  [S_v, H_k, n_tokens, n_seqs] contiguous rows, q/k share strides
//   v:     [S_v, H,   n_tokens, n_seqs] contiguous rows
//   g,beta:[1,  H,   n_tokens, n_seqs] contiguous (non-KDA scalar-per-head gate)
//   state: [S_v, S_v, H, n_seqs] contiguous, stored transposed: state[v*S_v + k] = S[k][v]
//   out:   attn [S_v, H, n_tokens, n_seqs] followed by the new state (K == 1: one slot)
//   scratch: A [n_chunks][H][n_seqs][PACKED_CS] (per v-head) and KQ_gram [n_chunks][H_k][n_seqs][PACKED_CS]
//
// The v-head h reads q/k from k-head h % H_k (fastmodulo on the same magics as the sequential
// kernel) and its g/beta from head h.
// ---------------------------------------------------------------------------------------------

#define GDN_CHUNKED_CS 64
#define PACKED_CS (GDN_CHUNKED_CS * (GDN_CHUNKED_CS + 1) / 2)   // 2080: packed strict-upper+diag
#define GDN_CHUNKED_NTHREADS 256
#define GDN_CHUNKED_VT 16   // state v-rows per scan block (S_v/16 blocks per head)

// State-phase K prefetch depth (float4s in flight per thread). The state phase's chunk K
// re-read is the scan's bottleneck (one un-overlapped HBM load per token); batching the
// independent per-token loads into a rotating window lets the scheduler overlap their
// latency with the 64-FMA accumulator chains. CS=64 divides evenly by 2/4/8; each step adds
// 4 floats of registers per thread (4 = 16, 8 = 32).

template <int S_v>
struct gdn_chunked_kkt_smem {
    // (the gram contracts over d with float4-over-d reads straight from HBM -- the LDS staging
    // was 32KB of unnecessary shared memory that capped this kernel at 1 block/CU; the A and
    // the gate arrays alone are ~17KB, so 3 blocks fit per CU)
    float A[GDN_CHUNKED_CS][GDN_CHUNKED_CS];   // the KKT inverse (built per v-head)
    float gc[GDN_CHUNKED_CS];   // raw gate, pre-cumsum
    float gcs[GDN_CHUNKED_CS];  // chunk-local inclusive cumsum
    float bt[GDN_CHUNKED_CS];   // beta
};

template <int S_v>
struct gdn_chunked_scan_smem {
    // A and KQ are unit-upper; only the upper triangle + diagonal is stored, packed by
    // COLUMN: pack[j][i] for i <= j lives at j*(j+1)/2 + i, so each column is a contiguous
    // run and the v_corr/o column contractions read linearly. The kkt writes the pack
    // directly (scattered stores are fine); the scan copies it contiguously.
    float AK[PACKED_CS];                       // A (v_corr/P), then KQ (intra-chunk attention)
    float S[S_v][GDN_CHUNKED_VT];              // this block's state slice, resident across chunks
    float UV[GDN_CHUNKED_CS][GDN_CHUNKED_VT];  // U, then v_new
    float QS[GDN_CHUNKED_CS][GDN_CHUNKED_VT];  // S^T Q_s (the o phase's state term), computed
                                               // in the U phase so Q is read once per chunk
    float Vb[GDN_CHUNKED_CS][GDN_CHUNKED_VT];  // V_b slice [token][v], staged for the v_corr
    // gate arrays (per chunk, per head)
    float gc[GDN_CHUNKED_CS];   // raw gate, pre-cumsum
    float gcs[GDN_CHUNKED_CS];  // chunk-local inclusive cumsum
    float eg[GDN_CHUNKED_CS];   // e^{gcs} (<= 1: gates are negative)
    float el[GDN_CHUNKED_CS];   // e^{g_last - gcs} (<= 1; 0 on padding)
    float bt[GDN_CHUNKED_CS];   // beta
};

// ---------------------------------------------------------------------------------------------
// kkt_solve: gram once per (chunk, k-head, seq), A per v-head, KQ_gram per k-head.
// ---------------------------------------------------------------------------------------------
template <int S_v>
__global__ static void __launch_bounds__(GDN_CHUNKED_NTHREADS)
gdn_chunked_kkt_cuda(
        const float * __restrict__ k,
        const float * __restrict__ q,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        float * __restrict__ A_sc,    // [n_chunks][H_v][n_seqs][PACKED_CS] fp32
        float * __restrict__ KQ_sc,   // [n_chunks][H_k][n_seqs][PACKED_CS] fp32 (upper + diag)
        int64_t H_k, int64_t H_v, int64_t hg_ratio, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        float scale)
{
    __shared__ gdn_chunked_kkt_smem<S_v> s;
    const int tid = threadIdx.x;
    const int c   = blockIdx.x;   // chunk
    const int vh  = blockIdx.y;   // v-HEAD (grid is now H_v wide: each block does ONE v-head's
                                  // A, so the hg_ratio back-substitutions run in parallel
                                  // blocks instead of serially inside one block)
    const int seq = blockIdx.z;   // sequence (uniform length: no cu needed)
    const int hg  = vh % H_k;     // the k-head whose K/Q this v-head reads
    const int m   = vh / H_k;     // 0..hg_ratio-1; only the m==0 block of a k-head stores KQ

    const int c0 = c * GDN_CHUNKED_CS;
    const int nval = min(GDN_CHUNKED_CS, (int) n_tokens - c0);
    const int tok_lim = nval - 1;
#define TK(t) min((t), (tok_lim))

    const float * kbase = k + (int64_t) seq * sq3 + (int64_t) c0 * sq2 + hg * sq1;
    const float * qbase = q + (int64_t) seq * sq3 + (int64_t) c0 * sq2 + hg * sq1;
    const float * gbase = g + ((int64_t) seq * n_tokens + c0) * H_v + vh;

    // gram tile ownership: 4x4 tiles, upper triangle only (i0 <= j0)
    const int i0 = (tid & 15) * 4;
    const int j0 = (tid >> 4) * 4;
    const bool upper = (i0 <= j0);

    // ---- gram G[i][j] = sum_d K[d][i] K[d][j] in registers (upper only; symmetric) ---------
    // d is contiguous in HBM, so the contraction reads float4-over-d straight from L2/L1 --
    // no LDS staging needed (the transposed-staging form was a leftover from the debugging
    // saga: the original [d][t] layout flipped the float4 orientation, the fix was to read
    // direct, not to add a 32KB staging buffer).
    float G[4][4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
#pragma unroll
        for (int f = 0; f < 4; ++f) G[e][f] = 0.0f;
    if (upper) {
        for (int d0 = 0; d0 < S_v; d0 += 4) {
            float ki[4][4], kj[4][4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                *(float4*) &ki[e][0] = *(const float4 *) (kbase + d0 + (int64_t) TK(i0 + e) * sq2);
#pragma unroll
            for (int f = 0; f < 4; ++f)
                *(float4*) &kj[f][0] = *(const float4 *) (kbase + d0 + (int64_t) TK(j0 + f) * sq2);
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f)
#pragma unroll
                    for (int dd = 0; dd < 4; ++dd)
                        G[e][f] += ki[e][dd] * kj[f][dd];
        }
    }

    // ---- this block's v-head: A = (I + strict_upper(beta . G . decay))^-1, store -----------
    // (grid is H_v wide, so there is no per-block v-head loop; the gates are this v-head's)
    {
        const float * bb = beta + ((int64_t) seq * n_tokens + c0) * H_v + vh;

        if (tid < GDN_CHUNKED_CS) {
            if (tid < nval) {
                s.gc[tid] = gbase[(int64_t) tid * H_v];
                s.bt[tid] = bb[(int64_t) tid * H_v];
            } else {
                s.gc[tid] = 0.0f;   // ggml_pad semantics: zero-padded gate keeps the cumsum flat
                s.bt[tid] = 0.0f;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CHUNKED_CS; ++t) { acc += s.gc[t]; s.gcs[t] = acc; }
        }
        __syncthreads();

        // build A from the gram (the exponent is <= 0 on every kept pair, see the header)
        if (upper) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = i0 + e;
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int j = j0 + f;
                    s.A[i][j] = (i == j) ? 1.0f
                              : (i < j)  ? s.bt[j] * __expf(s.gcs[j] - s.gcs[i]) * G[e][f]
                              : 0.0f;
                }
            }
        }
        __syncthreads();

        // (I+L)x = e_col by back substitution, lane per column (unit upper: x[i]=0 for i>col).
        // The substitution READS the original upper-triangular L out of s.A while the in-place
        // column overwrites below replace s.A with the inverse. Without a barrier between the
        // two phases this is a cross-warp data race: thread col reads s.A[i][j] (j > i) while
        // thread j overwrites it, and the outcome depends on the driver's warp scheduler. Split
        // the block in two with a full-block barrier so every thread finishes reading the
        // original L before any thread stores its column (llama-cpp-rdna-boosts#2).
        float x[GDN_CHUNKED_CS];
        if (tid < GDN_CHUNKED_CS) {
            const int col = tid;
#pragma unroll
            for (int i = GDN_CHUNKED_CS - 1; i >= 0; --i) {
                float acc = 0.0f;
#pragma unroll
                for (int j = i + 1; j < GDN_CHUNKED_CS; ++j)
                    acc += s.A[i][j] * x[j];
                x[i] = ((i == col) ? 1.0f : 0.0f) - acc;
            }
        }
        __syncthreads();   // all threads' reads of the original s.A are complete
        if (tid < GDN_CHUNKED_CS) {
            const int col = tid;
#pragma unroll
            for (int i = 0; i < GDN_CHUNKED_CS; ++i)
                s.A[i][col] = x[i];
        }
        __syncthreads();

        // store A -> scratch, PACKED by column (j*(j+1)/2 + i for i <= j)
        {
            const int64_t off = ((int64_t) c * H_v + vh) * n_seqs * PACKED_CS
                              + (int64_t) seq * PACKED_CS;
            if (tid < GDN_CHUNKED_CS) {
                const int col = tid;
                const int64_t base = off + (int64_t) col * (col + 1) / 2;
                for (int i = 0; i <= col; ++i)
                    A_sc[base + i] = s.A[i][col];
            }
        }
        __syncthreads();
    }

    // ---- KQ gram per k-head: KQ[i][j] = scale * sum_d K[d][i] Q[d][j] (direct reads) ------
    // (only the m == 0 block of a k-head computes and stores it; the scan reads the per-k-head
    // copy and applies its own v-head's decay)
    if (upper && m == 0) {
        float KQ[4][4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) KQ[e][f] = 0.0f;
        for (int d0 = 0; d0 < S_v; d0 += 4) {
            float ki[4][4], qj[4][4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                *(float4*) &ki[e][0] = *(const float4 *) (kbase + d0 + (int64_t) TK(i0 + e) * sq2);
#pragma unroll
            for (int f = 0; f < 4; ++f)
                *(float4*) &qj[f][0] = *(const float4 *) (qbase + d0 + (int64_t) TK(j0 + f) * sq2);
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f)
#pragma unroll
                    for (int dd = 0; dd < 4; ++dd)
                        KQ[e][f] += ki[e][dd] * qj[f][dd];
        }
        // store KQ_gram -> scratch PACKED per k-head, WITHOUT the decay: the decay is per
        // v-head (GQA: 3 v-heads share one k-head with different gates), so the scan applies
        // its own head's decay after the contiguous copy.
        const int64_t off = ((int64_t) c * H_k + hg) * n_seqs * PACKED_CS
                          + (int64_t) seq * PACKED_CS;
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const int i = i0 + e, j = j0 + f;
                if (i <= j)
                    KQ_sc[off + (int64_t) j * (j + 1) / 2 + i] = scale * KQ[e][f];
            }
    }
#undef TK
}

// ---------------------------------------------------------------------------------------------
// chunk_scan: state slice resident in LDS across the chunk loop; U / v_new / o / S' in one pass.
// ---------------------------------------------------------------------------------------------
template <int S_v>
__global__ static void __launch_bounds__(GDN_CHUNKED_NTHREADS)
gdn_chunked_scan_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ state_in,
        float * __restrict__ attn_out,
        float * __restrict__ state_out,
        const float * __restrict__ A_sc,    // [n_chunks][H_v][n_seqs][PACKED_CS]
        const float * __restrict__ KQ_sc,   // [n_chunks][H_k][n_seqs][PACKED_CS]
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_seq_stride)
{
    __shared__ gdn_chunked_scan_smem<S_v> s;
    const int tid = threadIdx.x;
    const int vt  = blockIdx.x;   // state v-tile (S_v/16 of them)
    const int h   = blockIdx.y;   // v-head
    const int seq = blockIdx.z;
    const int v0g = vt * GDN_CHUNKED_VT;   // state v-offset within the head's state

    const int n_chunks = (int) ((n_tokens + GDN_CHUNKED_CS - 1) / GDN_CHUNKED_CS);

    // q/k head for this v-head and the q/k seq index (matches the sequential kernel)
    const int64_t hq  = fastmodulo((uint32_t) h, neqk1_magic);
    const int64_t iq3 = fastdiv((uint32_t) seq, rq3_magic);

    const int64_t state_seq_base = (int64_t) seq * state_seq_stride + (int64_t) h * S_v * S_v;

    // ---- resident state slice: state[v][k] (v = v0g..v0g+VT-1), S[k][v] in LDS --------------
    for (int i = tid; i < S_v * GDN_CHUNKED_VT / 4; i += GDN_CHUNKED_NTHREADS) {
        const int k0 = (i / GDN_CHUNKED_VT) * 4, v = i % GDN_CHUNKED_VT;
        const float4 sv = *(const float4 *) (state_in + state_seq_base + (int64_t) (v0g + v) * S_v + k0);
        s.S[k0 + 0][v] = sv.x;
        s.S[k0 + 1][v] = sv.y;
        s.S[k0 + 2][v] = sv.z;
        s.S[k0 + 3][v] = sv.w;
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; ++c) {
        const int c0 = c * GDN_CHUNKED_CS;
        const int nval = min(GDN_CHUNKED_CS, (int) n_tokens - c0);
        const int tok_lim = nval - 1;
#define TK(t) min((t), (tok_lim))

        const float * kbase = k + iq3 * sq3 + (int64_t) c0 * sq2 + hq * sq1;
        const float * qbase = q + iq3 * sq3 + (int64_t) c0 * sq2 + hq * sq1;
        const float * vbase = v + (int64_t) seq * sv3 + (int64_t) c0 * sv2 + h * sv1;
        const float * gbase = g + ((int64_t) seq * n_tokens + c0) * H + h;
        const float * bbase = beta + ((int64_t) seq * n_tokens + c0) * H + h;

        const int64_t A_off = ((int64_t) c * H + h) * n_seqs * PACKED_CS
                            + (int64_t) seq * PACKED_CS;
        const int64_t KQ_off = ((int64_t) c * H_k + hq) * n_seqs * PACKED_CS
                             + (int64_t) seq * PACKED_CS;
        float * obase = attn_out + ((int64_t) seq * n_tokens + c0) * H * S_v + h * S_v;

        // ---- phase 1: gate, beta, chunk-local inclusive cumsum ------------------------------
        if (tid < GDN_CHUNKED_CS) {
            if (tid < nval) {
                s.gc[tid] = gbase[(int64_t) tid * H];
                s.bt[tid] = bbase[(int64_t) tid * H];
            } else {
                s.gc[tid] = 0.0f;
                s.bt[tid] = 0.0f;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CHUNKED_CS; ++t) { acc += s.gc[t]; s.gcs[t] = acc; }
        }
        __syncthreads();
        if (tid < GDN_CHUNKED_CS) {
            const float gc = s.gcs[tid];

            s.eg[tid] = __expf(gc);
            // el feeds the state update S' += el[t] * K[k][t] * v_new[t][v]. K reads are clamped
            // to the last real token (zero-padding would read OOB), so the padding factor must
            // be 0 or the clamped K columns leak into the state.
            s.el[tid] = (tid < nval) ? __expf(s.gcs[tok_lim] - gc) : 0.0f;
        }
        __syncthreads();

        // ---- load A into s.AK (the v_corr / predicted-v operand): contiguous packed copy ----
        {
            const float4 * src = (const float4 *) (A_sc + A_off);
            float4 * dst = (float4 *) &s.AK[0];
            for (int i = tid; i < PACKED_CS / 4; i += GDN_CHUNKED_NTHREADS)
                dst[i] = src[i];
        }
        __syncthreads();

        // ---- stage the V_b slice: Vb[t][v] = V[v0g+v][h][c0+t] * beta[t] -------------------
        {
            const int t = tid >> 2;            // 64 tokens
            const int vv = (tid & 3) * 4;      // 16 v-rows (4 per thread)
            const float bt = s.bt[t];
            const float4 v4 = *(const float4 *) (vbase + (int64_t) TK(t) * sv2 + (v0g + vv));
            s.Vb[t][vv + 0] = v4.x * bt;
            s.Vb[t][vv + 1] = v4.y * bt;
            s.Vb[t][vv + 2] = v4.z * bt;
            s.Vb[t][vv + 3] = v4.w * bt;
        }


        // ---- U[j][v] = sum_k K_b[k][j] S[k][v]  -> s.UV  AND  QS[j][v] = sum_k Q_s[k][j] S[k][v]
        // ---- -> s.QS. All 256 threads participate (thread (j, v0) owns v = v0, v0+4, v0+8,
        // ---- v0+12): 8 warps issue the Q column reads, hiding the HBM latency that stalled
        // ---- the old 2-warp U phase; the S reads broadcast across the j threads of a warp.
        {
            const int j  = tid >> 2;        // 0..63 (token)
            const int v0 = tid & 3;         // v = v0, v0+4, v0+8, v0+12
            float u[4], q[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) { u[e] = 0.0f; q[e] = 0.0f; }
            const float bj = s.bt[j];
            // K and Q are both read as float4 over k; 8 warps issue them, hiding the L2/HBM
            // latency that stalled the old 2-warp phase. Q is software-pipelined 8 deep
            // (read once per chunk, folded into the o phase's state term via s.QS).
            float4 qb[8];
            int kg = 0;
            for (; kg < 8 && kg * 4 < S_v; ++kg)
                qb[kg] = *(const float4 *) (qbase + kg * 4 + (int64_t) TK(j) * sq2);
#pragma unroll
            for (int k0 = 0; k0 < S_v; k0 += 4) {
                const float4 qq = qb[0];
                const float4 kk = *(const float4 *) (kbase + k0 + (int64_t) TK(j) * sq2);
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const float kb = kk[f] * bj;
                    const float qs = qq[f] * scale;
                    const float s0 = s.S[k0 + f][v0];
                    const float s1 = s.S[k0 + f][v0 + 4];
                    const float s2 = s.S[k0 + f][v0 + 8];
                    const float s3 = s.S[k0 + f][v0 + 12];
                    u[0] += kb * s0;  u[1] += kb * s1;  u[2] += kb * s2;  u[3] += kb * s3;
                    q[0] += qs * s0;  q[1] += qs * s1;  q[2] += qs * s2;  q[3] += qs * s3;
                }
                qb[0] = qb[1]; qb[1] = qb[2]; qb[2] = qb[3]; qb[3] = qb[4];
                qb[4] = qb[5]; qb[5] = qb[6]; qb[6] = qb[7];
                if (k0 + 32 < S_v)
                    qb[7] = *(const float4 *) (qbase + (k0 + 32) + (int64_t) TK(j) * sq2);
            }
            s.UV[j][v0]      = u[0];  s.UV[j][v0 + 4]  = u[1];  s.UV[j][v0 + 8]  = u[2];  s.UV[j][v0 + 12] = u[3];
            s.QS[j][v0]      = q[0];  s.QS[j][v0 + 4]  = q[1];  s.QS[j][v0 + 8]  = q[2];  s.QS[j][v0 + 12] = q[3];
        }
        __syncthreads();

        // ---- W[i][v] = V_b[i][v] - eg[i] U[i][v]  (in place: Vb is dead after this) --------
        // ---- v_new[j][v] = sum_{i<=j} A[i][j] W[i][v] -> s.UV --------------------------------
        // ---- factoring diag(eg) into W halves the v_corr's per-iteration LDS traffic and
        // ---- FMAs (10 loads + 8 FMA -> 5 loads + 4 FMA per iteration) -----------------------
        {
            const int i  = tid >> 2;
            const int v0 = tid & 3;
            const float ei = s.eg[i];
            s.Vb[i][v0]      = s.Vb[i][v0]      - ei * s.UV[i][v0];
            s.Vb[i][v0 + 4]  = s.Vb[i][v0 + 4]  - ei * s.UV[i][v0 + 4];
            s.Vb[i][v0 + 8]  = s.Vb[i][v0 + 8]  - ei * s.UV[i][v0 + 8];
            s.Vb[i][v0 + 12] = s.Vb[i][v0 + 12] - ei * s.UV[i][v0 + 12];
        }
        __syncthreads();
        {
            const int j  = tid >> 2;
            const int v0 = tid & 3;
            float vc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) vc[e] = 0.0f;
            const int pbase = j * (j + 1) / 2;   // packed-upper base: column j is contiguous
            for (int i = 0; i <= j; ++i) {
                const float a = s.AK[pbase + i];
                vc[0] += a * s.Vb[i][v0];
                vc[1] += a * s.Vb[i][v0 + 4];
                vc[2] += a * s.Vb[i][v0 + 8];
                vc[3] += a * s.Vb[i][v0 + 12];
            }
            // every thread's last read of s.UV (the U data) must land before any thread
            // overwrites it with v_new: the barrier is inside the (uniform) block scope so
            // every thread reaches it before the store
            __syncthreads();
            s.UV[j][v0]      = vc[0];
            s.UV[j][v0 + 4]  = vc[1];
            s.UV[j][v0 + 8]  = vc[2];
            s.UV[j][v0 + 12] = vc[3];
        }
        __syncthreads();
        // ---- load KQ_gram into s.AK (contiguous packed copy), then apply this v-head's ------
        // ---- decay in place: the gram is shared per k-head (GQA), the decay is per v-head ----
        {
            const float4 * src = (const float4 *) (KQ_sc + KQ_off);
            float4 * dst = (float4 *) &s.AK[0];
            for (int i = tid; i < PACKED_CS / 4; i += GDN_CHUNKED_NTHREADS)
                dst[i] = src[i];
            __syncthreads();
            // 4x4 upper tiles over the 64x64 logical matrix, each entry packed at j(j+1)/2 + i
            const int i0t = (tid & 15) * 4;
            const int j0t = (tid >> 4) * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int i = i0t + e, j = j0t + f;
                    if (i <= j)
                        s.AK[j * (j + 1) / 2 + i] *= __expf(s.gcs[j] - s.gcs[i]);
                }
        }
        __syncthreads();

        // ---- o[t][v] = eg[t] * QS[t][v] + sum_j KQ[j][t] v_new[j][v]  (LDS only) -----------
        if (tid < GDN_CHUNKED_CS * GDN_CHUNKED_VT / 4) {
            const int t = tid >> 2;         // 0..63
            const int vv = (tid & 3) * 4;   // 0..12
            const float4 qs = *(const float4 *) &s.QS[t][vv];
            const float egt = s.eg[t];
            float acc[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[e] = egt * qs[e];
            const int pbase = t * (t + 1) / 2;   // packed-upper base: column t is contiguous
            for (int j = 0; j <= t; ++j) {
                const float kq = s.AK[pbase + j];
                const float4 vn = *(const float4 *) &s.UV[j][vv];
                acc[0] += kq * vn.x;
                acc[1] += kq * vn.y;
                acc[2] += kq * vn.z;
                acc[3] += kq * vn.w;
            }
            if (t < nval) {
                *(float4 *) (obase + (int64_t) t * H * S_v + v0g + vv) =
                    make_float4(acc[0], acc[1], acc[2], acc[3]);
            }
        }

        // the o phase reads s.S (the chunk's INCOMING state) and s.UV; the state update below
        // overwrites s.S in place -- every thread must finish its last read before any write
        __syncthreads();

        // ---- S'[k][v] = e^{g_last} S[k][v] + sum_t el[t] K[k][t] v_new[t][v] ----------------
        {
            // 32 k-tiles x 4 v-tiles = 128 tiles; threads 128..255 idle (uniform guard)
            const int k0 = (tid >> 2) * 4;        // k-tile row offset
            const int vv = (tid & 3) * 4;
            const float eg0 = s.eg[tok_lim];      // e^{g_last}
            if (k0 < S_v) {
                float acc[4][4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        acc[e][f] = eg0 * s.S[k0 + e][vv + f];
#define GDN_CHUNKED_STATE_PIPE 8   // tune: 2/4/8
                float4 kk[GDN_CHUNKED_STATE_PIPE];
#pragma unroll
                for (int u = 0; u < GDN_CHUNKED_STATE_PIPE; ++u)
                    kk[u] = *(const float4 *) (kbase + k0 + (int64_t) TK(u) * sq2);
                for (int t = 0; t < GDN_CHUNKED_CS; t += GDN_CHUNKED_STATE_PIPE) {
#pragma unroll
                    for (int u = 0; u < GDN_CHUNKED_STATE_PIPE; ++u) {
                        const float4 vn = *(const float4 *) &s.UV[t + u][vv];
                        const float elt = s.el[t + u];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float kke = kk[u][e] * elt;
                            acc[e][0] += kke * vn.x;
                            acc[e][1] += kke * vn.y;
                            acc[e][2] += kke * vn.z;
                            acc[e][3] += kke * vn.w;
                        }
                    }
                    // prefetch the next group into the (now dead) window: the loads are
                    // independent of the FMAs above, so the scheduler overlaps them with the
                    // accumulator chains; the last group's guard skips the OOB re-read.
                    if (t + GDN_CHUNKED_STATE_PIPE < GDN_CHUNKED_CS) {
#pragma unroll
                        for (int u = 0; u < GDN_CHUNKED_STATE_PIPE; ++u)
                            kk[u] = *(const float4 *) (kbase + k0 + (int64_t) TK(t + GDN_CHUNKED_STATE_PIPE + u) * sq2);
                    }
                }
#undef GDN_CHUNKED_STATE_PIPE
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        s.S[k0 + e][vv + f] = acc[e][f];
            }
        }
        __syncthreads();
#undef TK
    }

    // ---- write the final state slice: state[v][k] = S[k][v] ---------------------------------
    for (int i = tid; i < S_v * GDN_CHUNKED_VT / 4; i += GDN_CHUNKED_NTHREADS) {
        const int k0 = (i / GDN_CHUNKED_VT) * 4, v = i % GDN_CHUNKED_VT;
        const float4 sv = make_float4(s.S[k0 + 0][v], s.S[k0 + 1][v], s.S[k0 + 2][v], s.S[k0 + 3][v]);
        *(float4 *) (state_out + state_seq_base + (int64_t) (v0g + v) * S_v + k0) = sv;
    }
}

// ---------------------------------------------------------------------------------------------

template <int S_v>
static bool launch_gdn_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        float * A_sc, float * KQ_sc,
        int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_seq_stride, cudaStream_t stream) {
    const int n_chunks = (int) ((n_tokens + GDN_CHUNKED_CS - 1) / GDN_CHUNKED_CS);
    const int64_t hg_ratio = H / H_k;

    if (getenv("GDN_DBG_SKIP_KKT") == nullptr) {
        dim3 grid((unsigned) n_chunks, (unsigned) H, (unsigned) n_seqs);
        const dim3 block(GDN_CHUNKED_NTHREADS);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
        if (!ggml_cuda_kernel_launch_try(gdn_chunked_kkt_cuda<S_v>, launch_params,
                k_d, q_d, g_d, b_d, A_sc, KQ_sc, H_k, H, hg_ratio, n_tokens, n_seqs,
                sq1, sq2, sq3, scale)) {
            return false;
        }
    }
    if (getenv("GDN_DBG_SKIP_SCAN") != nullptr) return true;
    {
        dim3 grid((unsigned) (S_v / GDN_CHUNKED_VT), (unsigned) (H_k * hg_ratio), (unsigned) n_seqs);
        const dim3 block(GDN_CHUNKED_NTHREADS);
        const uint3 neqk1_magic = init_fastdiv_values(neqk1);
        const uint3 rq3_magic   = init_fastdiv_values(rq3);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
        return ggml_cuda_kernel_launch_try(gdn_chunked_scan_cuda<S_v>, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc, KQ_sc,
            H, H_k, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            neqk1_magic, rq3_magic, scale, state_seq_stride);
    }
}

bool ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext) {
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
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

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
    // state_d_ext: fused-cache slot base when the GDN->cpy is fused (K == 1: single slot);
    // nullptr keeps the default tail-of-output layout.
    float * state_d   = state_d_ext ? state_d_ext : dst_d + S_v * H * n_tokens * n_seqs;

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);
    const int64_t state_seq_stride = H * S_v * S_v;

    cudaStream_t stream = ctx.stream();

    const int64_t n_chunks = (n_tokens + GDN_CHUNKED_CS - 1) / GDN_CHUNKED_CS;
    ggml_cuda_pool_alloc<float> A_sc (ctx.pool(), (size_t) n_chunks * H     * n_seqs * PACKED_CS);
    ggml_cuda_pool_alloc<float> KQ_sc(ctx.pool(), (size_t) n_chunks * neqk1 * n_seqs * PACKED_CS);

    switch (S_v) {
        case 16:  return launch_gdn_chunked<16 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc.get(), KQ_sc.get(),
                    H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream);
        case 32:  return launch_gdn_chunked<32 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc.get(), KQ_sc.get(),
                    H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream);
        case 64:  return launch_gdn_chunked<64 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc.get(), KQ_sc.get(),
                    H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream);
        case 128: return launch_gdn_chunked<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, A_sc.get(), KQ_sc.get(),
                    H, neqk1, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream);
        default:
            GGML_ABORT("fatal error");
            return false;
    }
}
