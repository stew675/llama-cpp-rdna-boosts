// Fused chunked kernel for GGML_OP_GATED_DELTA_NET prefill (n_tokens > 1, K == 1, non-KDA).
//
// The chunked recurrence is sequential in the state, so it runs as TWO launches (see the .cu):
// a state-scan pass (block per (v-head, seq), looping chunks, state through a scratch buffer)
// and an output pass (same grid, reading the per-chunk states, writing only the attention out).
//
// The per-chunk math is the algebra of llm_build_delta_net_base::build_delta_net_chunking, with
// gcs the chunk-local inclusive cumsum of the gate, K_b = K . beta, V_b = V . beta, Q_s = scale . Q,
// and decay[i][j] = e^{gcs[j] - gcs[i]} on the upper triangle (i <= j):
//
//     A    = (I + strict_upper(K^T K_b . decay))^-1      (unit upper, the KKT solve)
//     U    = K_b^T S                                     (state times keys)
//     v_n  = V_b^T A - diag(e^g) A^T U                   (v_new = v_corr - predicted)
//     o    = e^g . S^T Q_s + KQ^T v_n                    (decay + intra-chunk attention)
//     S'   = e^{g_last} S + K diag(e^{g_last - g}) v_n   (state update)
//
// Decay is computed DIRECTLY per kept pair: the exponent gcs[j]-gcs[i] is <= 0 on every kept
// pair (gates are negative, padding keeps the cumsum flat), so it can never overflow even for
// the test harness's pathological gates. Padding is handled by clamping token reads to the last
// real token and bt[pad] = el[pad] = 0, which zero the gram entries and the state-update
// factors the way ggml_pad's zero columns do.

#pragma once
#include "common.cuh"

// Non-aborting launch used by the chunked GDN ops. Returns false (and consumes the sticky
// error) when the HSA/CUDA runtime rejects the dispatch SYNCHRONOUSLY -- e.g. ROCr refusing
// an AQL dispatch whose code object carries a hidden hostcall buffer on platforms without
// PCIe atomics (surfaces as hipErrorIllegalState at launch, llama-cpp-rdna-boosts#2). A
// rejected launch has no side effects, so the caller can safely recompute the same op with
// the sequential kernel instead of aborting. (An ASYNC fault inside a launched kernel is not
// visible here and still surfaces at the next synchronization point.)
template<typename Kernel, typename... Args>
static __inline__ bool ggml_cuda_kernel_launch_try(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    return cudaGetLastError() == cudaSuccess;
}

// Both chunked ops return true when their kernels were accepted for launch (or skipped by a
// GDN_DBG_* env hook) and false when a launch was rejected; the dispatcher then falls back
// to the sequential gated_delta_net kernel.
bool ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext = nullptr);

// bf16/WMMA tensor-core variant (S_v == 128 only; near-lossless: PPL +0.056% / KL 0.0036 on
// wikitext-2 vs the fp32 path). DEFAULT for S_v == 128 on the HIP build (opt out with
// GGML_CUDA_GDN_CHUNKED_BF16=0); the fp32 chunked path stays the fallback and the
// bit-exact option. Two arch-specific kernels, fully segregated (they share no code and
// cannot cross-regress):
//   gated_delta_net_chunked_bf16.cu       - gfx12 (RDNA4): the fork's validated RDNA4-only
//     kernel, restored verbatim (chunked-gdn tip 99bbd1b20, 46/46 on gfx1201).
//   gated_delta_net_chunked_bf16_gfx11.cu - gfx11 (RDNA3/RDNA3.5): the first-gen WMMA port
//     (16 bf16/lane, validated on gfx1100), on its own path.
bool ggml_cuda_op_gated_delta_net_chunked_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext = nullptr);

// gfx11 (RDNA3/RDNA3.5) first-gen WMMA variant - same contract as the gfx12 entry above.
bool ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext = nullptr);
