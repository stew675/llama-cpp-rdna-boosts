# Strix Halo (RDNA3.5 / gfx1151) — weighted-down fusion port (decode MoE tail)

Date: 2026-09-10 session (continuation of the equal/surpass-B campaign). A commit `a18e24f97`
on `1682d32a9`; beta patch 8 (`ws3-weighted-down-fusion.patch`). Environment: same as the
2026-09-10 records (IQ4_XS UD Qwen3.8-Flash-Next, llama-bench `-ngl 99 -t 15 -b 2048 -ub 2048
-fa on -ctk f16 -ctv f16 --load-mode none`).

## What and why

B (halo-box) collapses the decode MoE down-projection tail into one kernel
(`ggml_cuda_mul_mat_id_weighted_rdna3_5`, mmvq.cu): the graph emits
`down = mul_mat_id(down_exps, cur, ids)` [2560 x 10 x n_tokens] -> `ggml_mul(weights)` ->
10 VIEWs -> 9 ADDs (A's `build_moe_ffn` weight-after-FFN tail — already the A graph, no change
needed; A's existing MWR fusion already collapses the mul+views+adds part, but the mmid GEMM +
the [2560,10] intermediate + its write/read stayed). B's kernel computes all 10 selected
experts for one output row in a wave, applying the routing weights in the GEMM epilogue
(rn-rounded mul/add, matching the unfused graph's per-op rounding), quantizing the y input to
Q8_1 once. Ported verbatim into A: kernels + `_ok()` + dispatch appended to mmvq.cu, declared
in mmvq.cuh, and the CUDA fusion matcher inserted in `ggml_cuda_try_fuse` (before the MWR
match; opt-out `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`; RDNA3.5 gate + shape fingerprint: w
[640, 2560, 512] IQ4_NL or Q8_0, ids = 10, dst = 2560 elements -> single token, i.e. DECODE
only, n_tokens == 1; prefill falls through to the normal path).

## Verification

- FIRES in decode (rocprofv3 over 60 decode tokens): `mul_mat_id_iq4_nl_weighted_rdna3_5<10>`
  1505 calls (~25 IQ4_NL down-proj layers per token; the model's remaining down layers are
  IQ3_S, which B's kernel does not cover either — those stay unfused on both sides).
- Coherence: llama-cli seed-42 text fused == `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1` (40 decode
  tokens, byte-identical generated text; deterministic substrate).
- Perf (same-session pairs): tg128@0 25.03 vs 24.97 disabled (+0.2%, ~noise); tg128 @ d12288
  22.50 vs 22.22 disabled (+1.3%) — the fusion pays at depth where the per-token decode work
  (and launch savings: 19 kernel launches/layer x ~25 layers/token elided) matters more.
- No prefill regression (fusion is decode-only; same-order r3 depth-0 ladder unchanged:
  pp16384 573.6, pp8192 544.2, pp4096 486.7, pp2048 398.9 vs pre-port 575.1/543.6/487.3/400.1).

## Reading

The port = B-parity on this decode component (B runs the same fusion). Measured value is small
on this box (decode is host-launch/attention-bound, not the down-tail); it is NOT the source of
the remaining tg@0 gap (A ~25.0 vs B ~25.96). A's decode profile is dominated by thousands of
small `mul_mat_vec_q`/ksplit launches per token — the next decode lever is a kernel-mix
comparison A-vs-B (per-op decode profile), not this tail. The prefill shallow-row gap (depth-0
pp2048 1.93x) is a separate per-token cost (see the 2026-09-10 gap record).
