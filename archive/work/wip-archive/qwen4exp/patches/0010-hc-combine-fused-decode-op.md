# 0010 - Fused hc_combine decode op (GGML_OP_HC_COMBINE)

Applies: ggml (enum/builder/symbols), ggml-cpu (ref impl + dispatch),
ggml-cuda (hc_combine_kernel in hc-mix.cu + dispatch + supports_op),
ggml-backend-meta (MIRRORED split case), llama-cparams/llama-context
(fused_hc_combine flag + env LLAMA_FUSED_HC_COMBINE), qwen4exp.cpp
(decode-only branch in build_hc_combine).

What it does: fuses the hyper-connection residual combine of qwen4exp
decode (SCALE inject by 1/hc, SIGMOID, SCALE by 2, REPEAT of the block
output over the hc streams, MUL, ADD residual - the tail of
build_hc_combine) into ONE dispatch at nt == 1. Two combines per layer
(attn + ffn side) x 48 layers = 96 calls/token. Prefill (nt > 1) keeps
the unfused chain. Env toggle: LLAMA_FUSED_HC_COMBINE=0 (default on).
The op reads the same shapes the chain produces: residual [n_embd, hc,
nt], block_out [n_embd, nt], inject [hc, nt].

Bit-exactness: one kernel, one thread per row with the hc columns in
smem-broadcast w[c]. The 1/hc and 2.0 scales are exact in f32, so the
only value rounding is the sigmoid (inline formula matches the unary
op). Each block_out[r]*w[c] product is stored to an array before the
residual add so the compiler cannot contract it into an FMA (the
reference runs separate MUL and ADD ops) - same trap as the hc_mix
collapse. Result: decode logitdump A/B byte-identical vs the unfused
path (16 nt=1 positions, all 96 combines/token).

Result: llama-bench tg128 43.05 vs 41.19 t/s (+4.5% on top of the
hc_mix +5%; 39.15 unfused baseline -> 43.05 = +10% total). pp512
unchanged (1492-1500, decode-only construction).

Commit: 6a4e2c766 on the qwen4exp branch (parent bbd976ceb). Verified
to apply cleanly on the parent. GGML_OP_COUNT 104 -> 105 (ggml-rpc.h
patch version 2 -> 3).
