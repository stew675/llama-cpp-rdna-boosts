# Strix Halo (gfx1151) — hc elementwise fusion-window launches: scale→unary fixed (item (c))

Item (c) of the launch ledger ROOT-CAUSED + FIXED (fork f5ac11903, patch 21, env
GGML_CUDA_SCALE_UNARY=0, BIT-IDENTICAL).

## Root cause

A's newer upstream tree (b10837) never had the scale→unary peek-ahead fusion that B's
dispatcher carries (B ggml-cuda.cu ~5000, "scale -> silu/sigmoid (e.g. the qwen4exp
hyper-connection low-rank gate: silu(x / hc))"). No upstream model has that gate pattern
(qwen4exp's build_hc_mix prefill path: lo = silu(scale(lo, 1/hc)); hc_combine inject:
sigmoid(scale(inject, 1/hc))), so upstream never implemented the window and the refactor to
the try_fuse architecture dropped it.

A fired scale_f32 + unary silu as TWO kernels per pair (190-192 pairs/eval); B fired one
fused scale_unary (dst[i] = op(scale*x[i]+bias)).

## Fix

- unary.cu: ggml_cuda_op_scale_unary + scale_unary_kernel (halo-box port, A's kernel style
  with pdl calls).
- try_fuse: 2-node window placed AFTER all the larger hc windows (scale→sigmoid→scale→
  mul→add→rms... chains must match first) and after the tanh softcap window. Env opt-out
  GGML_CUDA_SCALE_UNARY=0.
- Memory-safety: NO ggml_cuda_check_fusion_memory_ranges gate — the fused kernel is purely
  elementwise (dst[i] reads only x[i]), so even the in-place case is safe. Census showed the
  general check fails for ALL 190 silu pairs (mem_ok=0): their scale runs IN-PLACE on the
  wide lo input, so the allocator legitimately reuses lo's buffer for the unary dst. An
  elementwise fused kernel handles dst==src safely (per-index read/write; whole-buffer
  allocator reuse is base-aligned, never shifted). The 188 sigmoid pairs (out-of-place
  scales) passed the general check and fused even under it.

## Validation

- Numerics: fusion ON and OFF both logitcmp BIT-IDENTICAL to the stored baseline (same
  per-element expression; verified pre/post cleanup).
- Counts (pp2048 r1 same-command rocprof): scale_f32 526 -> 336, unary silu 192 -> 2;
  scale_unary silu 190 + sigmoid 188 fire. Total kernels 7755 -> 7565 (-190/eval, exactly
  the silu pairs; the sigmoid arm was already firing in the first build).
- Same-session A-on vs A-off (GGML_CUDA_SCALE_UNARY=0): pp2048 +0.34% (747.4 vs 744.9),
  pp512 +0.42% (655.4 vs 652.7) — the fixed per-ubatch cost matters most at small pp.
- Deterministic: A-on == A-off == baseline.

## What remains on the launch ledger axis

A 7565 vs B 7001 kernels (+564 structural): scale_f32 A 336 vs B 298 (+38), plus deep tree
differences (B's unary_gated silu 94 vs A 0 — B's gated-silu variant covers silu chains A
handles as fused windows; A fuses MORE scale_unary sigmoid than B, 188 vs 2). These are
fusion-surface architecture differences, not single-window gaps. The +380 unary silu
component of the original +800 is CLOSED; the +416 scale_f32 component is mostly closed
(+88 -> +38/eval). Next candidates on this axis: rms_norm<256,true> count diff
(decode: A 61/step vs B 24.6/step), k_bin_bcast-add (A 55/step vs B 337/step — A FEWER).
