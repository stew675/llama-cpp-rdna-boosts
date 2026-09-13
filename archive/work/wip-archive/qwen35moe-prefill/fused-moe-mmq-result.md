# Milestone 2/3 result: fused gate+up+GLU MMQ kernel (patch 0003)

## What was built

The prefill `{MUL_MAT_ID(gate), MUL_MAT_ID(up), GLU}` triple now runs as ONE
MMQ kernel that reads both weight streams and applies the GLU epilogue in
registers. Verified bit-exact: 536M logits, 0 differing floats vs the 3-op
path.

- `mmq.cuh`: `mul_mat_q_process_tile` / `mul_mat_q` gained a `has_gate`
  template bool + `x_gate`/`glu_op`/`glu_limit` args. The gate K-loop reuses
  the SAME `tile_x`/`tile_y` shared buffers (no shared memory increase); the
  gate accumulation order per K-block is identical to the unfused run, so the
  GLU result is bit-exact. GLU epilogue supports SWIGLU/GEGLU/SWIGLU_OAI/
  SWIGLU_CLAMP.
- `mmq.cu`: `ggml_cuda_mul_mat_q` takes an optional
  `ggml_cuda_mm_fusion_args_host *`; when `fusion->gate` is set it routes to
  a new `ggml_cuda_mul_mat_q_switch_type_gate` (Q6_K enabled; extendable).
  One ids-sort, one src1 quantize, one launch for the pair.
- `ggml-cuda.cu` try_fuse: the `{op, op, GLU}` branch gained an MMQ arm after
  the mmvq/mmvf checks, gated on `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` opt-out.
- template-instances: `DECL_MMQ_CASE_GATE` + q6_k instance + generator.

## The register problem (the interesting part)

The first cut was SLOWER than the 3-op path at ub=512 (2797 vs 2978 t/s)
even though the decode-size test (pp8) won. Cause: the doubled accumulator
(`sum` + `sum_gate`) plus WMMA tile registers pushes the J=128 config
(nthreads=256, I=128) to 256 VGPR + 14 spills. The fix is a J cap: the fused
kernel skips J > 64 configs, forcing more, smaller blocks (2x work per block
is throughput-neutral once spills are avoided).

J cap sweep (pp512 ub=512 / pp2048 ub=2048, t/s):
| cap | pp512 | pp2048 |
|-----|-------|--------|
| 32  | 3620  | 5081   |
| 48  | 3559  | 5169   |
| 64  | 3458  | 5244   |
| 96  | 3156  | 5074   |
| uncapped | 2797 | (regression) |
| unfused | 2998 | 4892   |

cap 64 chosen: best at ub=2048 (the deployment config), still +15% at ub=512.

## Results (final, cap 64)

| test | fused | unfused | delta |
|------|-------|---------|-------|
| pp512 ub=512 | 3417-3470 | 2998 | +14-16% |
| pp2048 ub=512 | 3323 | 2895 | +14.8% |
| pp2048 ub=1024 | 4479 | 4027 | +11.2% |
| pp2048 ub=2048 | 5207 | 4899 | +6.3% |
| pp16384 ub=2048 | 4557 | 4309 | +5.8% |
| pp32768 ub=2048 | 3892 | 3711 | +4.9% |
| tg128 (decode) | 98.0 | 97.9 | unchanged |

Depth-flat ~5-8% at ub=2048; the win is larger at small ub (14-16% at
ub=512), confirming the user's rationale (smaller ub for VRAM-constrained or
high-expert-count workloads). This BEATS the milestone-1 revision (~3%):
the shared sort/quantize/launch savings were the minor part; the real win is
the fused kernel's better tile behavior at small J plus eliminating the GLU
pass and round-trip.

## Validation

- Bit-exact: logitdump 536M floats, max abs diff 0, 0 differing (both the
  J=64 and the pre-cap builds).
- Generation (temp 0, seed 42, 40 tokens): byte-identical text.
- test-backend-ops: all ops + MUL_MAT/MUL_MAT_ID pass (full suite OK).
- Decode: 98.0 t/s at KV=512 and KV=16K, unchanged (the fusion correctly
  does not fire on decode; mmvq still owns n_tokens <= 5).

## Remaining / notes

- Only Q6_K is enabled in `ggml_cuda_mul_mat_q_switch_type_gate`; add other
  k-quants by extending that switch + MMQ_GATE_TYPES in the generator.
- The J cap is a blunt instrument; a per-shape rule (cap relative to
  ncols_max) might squeeze a bit more at ub=512, but cap 64 is near the
  observed optimum at both ends.
- CUDA graphs: the fused kernel is a single launch, capturable (decode uses
  mmvq anyway; prefill skips graphs per block 11).
- The down mmid was NOT folded (kept scope per plan); the down+weights MUL +
  expert-sum ADDs stay as-is.
