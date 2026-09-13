# 0013 - hc_mix decode: fold the grouped RMSNorm + gamma into the fused op

Applies: ggml (builder), ggml-cpu (ref + wdata), ggml-cuda (hc-mix.cu +
supports_op), ggml-backend-meta (MIRRORED), llama-cparams/llama-context
(unchanged), qwen4exp.cpp (build_hc_mix branch).

What it does: the op now takes the raw hc residual x [n_embd, hc, nt] and
w_norm, runs the grouped RMSNorm over each stream internally (a 1024-
thread block_reduce clone of rms_norm_f32, with the gamma MUL's separate
rounding) and PERSISTS xn in the dst tail: dst [n_embd + hc_dim, nt]
with mixed at the head and xn at the tail. The model views both out of
the one result; the F32 inject MUL_MAT consumes the xn view and is left
untouched (its mmvf path is not replicated - F32 weights, not Q8_0).
The graph loses the RMS + MUL dispatches per mix call (~194 node evals
per device per token); the decode logitdump A/B stays byte-identical
(the rms replication is exact).

Why not fold the inject too: the inject weights are F32 in the GGUF, so
the reference inject runs the mmvf kernel (float2-pair FMA accumulation)
- replicating it is a separate exactness chase for ~10us/call.

Result: tg128 44.81 vs 44.31 (+1%). Commit 50ae9d134 (parent b5f7fd598).
