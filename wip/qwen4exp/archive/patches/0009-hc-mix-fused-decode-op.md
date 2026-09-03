# 0009 - Fused hc_mix decode op (GGML_OP_HC_MIX)

Applies: ggml (enum/builder/symbols), ggml-cpu (ref impl + dispatch),
ggml-cuda (hc-mix.cu/.cuh + dispatch), ggml-backend-meta (MIRRORED split
case), llama-cparams/llama-context (fused_hc_mix flag + env), qwen4exp.cpp
(decode-only branch in build_hc_mix).

What it does: fuses the hyper-connection mix tail of qwen4exp decode
(down LoRA Q8_0 mm, scale 1/hc, silu, up LoRA mm, sigmoid, gate mul,
4-stream collapse) into ONE op dispatch at nt == 1. Prefill (nt > 1)
keeps the unfused chain, so prefill-path gates stay valid by
construction. Toggle: LLAMA_FUSED_HC_MIX=0 env (default on). The rms+
gamma nodes stay in the graph (the inject MUL_MAT consumes the same xn).

Bit-exactness: the CUDA kernels replicate the unfused decode chain byte
for byte. Three subtleties found by micro A/B (see handover-2026-09-04-
decode.md, session 3):
1. vec_dot_q8_0_q8_1 takes the Q8_1 block pointer as-is (kbx offsets only
   the weight).
2. the mmvq dispatch applies a rows-per-block override on RDNA2+ for
   short-K GEMMs (K=320 -> rpb=16) that changes the per-row accumulation
   tree; the kernel must clone mul_mat_vec_q with the matching rpb.
3. the collapse must round each xn*gate product before adding (no FMA
   contraction: the reference runs separate MUL and ADD ops).

Result: decode logitdump A/B byte-identical vs the unfused path (16 nt=1
positions, all 97 fused calls/token); llama-bench tg128 41.11 vs 39.15
t/s (+5.0%), the first decode win on qwen4exp.

Commit: bbd976ceb on the qwen4exp branch (parent 9dff5cc4f). Verified to
apply cleanly on the parent. Also fixes a pre-existing GGML_OP_SYMBOL
off-by-one (INDEXER_TOPK had no symbol entry) and bumps GGML_OP_COUNT
103 -> 104.
