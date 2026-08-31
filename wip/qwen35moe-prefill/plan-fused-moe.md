# PLAN (b): fused MoE gate+up+GLU kernel for the routed experts

## Status: Milestone 1 (graph verification) DONE (2026-08-31)

Verified against the real prefill graph (pp16384, ub=2048, op timing dump):

1. **Separate tensors, separate path**: the GGUF has separate
   `ffn_gate_exps`/`ffn_up_exps` (no `ffn_gate_up_exps` merged tensor), so
   `create_tensor_gate_up_exps` falls back and `build_moe_ffn` takes the
   separate path: `MUL_MAT_ID(gate)` + `MUL_MAT_ID(up)` + `SWIGLU_SPLIT`
   per layer. Node order is gate-first in the cgraph (the decode dump shows
   the FUSED triple anchored at `ffn_moe_gate`), matching the
   `mul_mat_id_glu_ops` = {MUL_MAT_ID, MUL_MAT_ID, GLU} assumption.
2. **Shapes**: gate/up weights [K=2048, n_ff=512, n_expert=256] Q6_K
   (note: K=2048 = n_embd, NOT 2560 as this plan originally said); both
   share src1=`cur` and src2=`selected_experts`. `ggml_cuda_should_fuse_mul_mat`
   passes (same type/shape/stride, same src1/src2, swiglu not swapped).
3. **Weights-scale MUL is separate**: `ffn_moe_weighted` (expert probs x
   down output) is applied AFTER the GLU/down, so the fused kernel does not
   touch it. Confirmed in build_moe_ffn.
4. **qwen4exp**: same build_moe_ffn hook (same call args, same fallback).
5. **Dispatch gap confirmed**: try_fuse's {op, op, GLU} branch only
   dispatches `mul_mat_vec_q`/`mul_mat_vec_f` (mmvq: ne[1]<=8, or F32/F16
   src0). Prefill (ne=2048, Q6_K) falls through to 3 separate ops. This is
   the gap the fused MMQ kernel fills.
6. **Win estimate REVISED DOWN (measured, not modeled)**: at ub=2048 the
   deepest eval (485.8 ms) shows gate 12.8% + up 13.3% + down 13.9% =
   ~40% mmid block, but SWIGLU_SPLIT itself is only 1.54% (7.5 ms) and the
   gate/up round-trip traffic is ~2.3% of eval. The MMQ kernels are
   compute/bandwidth-bound, NOT launch/sort-bound (first gate op 5.9 ms =
   cold, later ones 2.2-2.8 ms: overhead is once per eval, not per op).
   Realistic fused win: ~2-3% at ub=2048 (eliminate the GLU pass + gate
   round-trip + shared sort/quantize), more at ub=512 where the fixed
   costs are proportionally larger. The plan's original 6-10% was too
   optimistic; do not oversell the prototype result.

## Why there is room

Per layer the routed experts currently run, as separate graph nodes with
independent dispatch:

```
ffn_moe_gate: MUL_MAT_ID [n_ff=512, n_embd=2560, n_expert=256] x tokens, ids
ffn_moe_up:   MUL_MAT_ID [n_ff=512, n_embd=2560, n_expert=256] x tokens, ids
GLU (swiglu): silu(gate) * up
```

Each MUL_MAT_ID independently:
1. runs `mm_ids_helper` (ids sort / inverse map / expert bounds),
2. quantizes the SAME src1 (token activations) to q8_1 - twice,
3. launches the batched MMQ kernel (grid.z = n_experts),
4. writes the [512 x n_tokens x 8] intermediate,
then the GLU pass reads both and writes again.

The `mul_mat_id_glu_ops` pattern is already recognized by
`ggml_cuda_can_fuse()` (ggml-cuda.cu ~3192) but the result is never consumed
- there is no fused kernel. Both this fork and the upstream base
(`a7cc83bba`) lack the fused experts forward (vLLM has it; llama.cpp does
not, at this base).

## What to build

### 1. Fused kernel (MMQ-style, one launch)

Reuse the mmid machinery, once:
- one `mm_ids_helper` sort,
- one src1 q8_1 quantize (the `dedup_bcast` path already quantizes each
  token once; keep it),
- one batched kernel: grid.z = n_experts, each CTA processes its expert's
  tokens x N-tile by streaming K over BOTH weight matrices
  (`ffn_gate_exps` and `ffn_up_exps`), accumulating two dot products, then
  epilogue `out = silu(gate_acc) * up_acc` written to the GLU dst.

Quant type support starts at Q6_K (the target model) and extends to the
other MMQ types (Q4_K, Q5_K, Q8_0, ...) via the existing
`mmq_switch_type`-style dispatch.

### 2. Dispatch hook

In the node loop of `ggml_cuda_graph_evaluate_and_capture()`, when
`ggml_cuda_can_fuse(cgraph, i, {MUL_MAT_ID, MUL_MAT_ID, GLU}, {})` matches
(conditions verified in milestone 1), dispatch the fused kernel and skip the
two following nodes. Gate behind an env toggle (`GGML_CUDA_FUSED_MOE=0`
opts out) for A/B and regression testing. Fall back to the current 3-op
path whenever `ggml_cuda_should_fuse_mul_mat()` or the type table rejects.

### 3. Correctness

- Bit-exactness: the fused kernel must reproduce the accumulation order of
  the separate MMQ runs per weight (same K tiling per weight inside the
  shared loop). If that forces an unnatural kernel, accept
  near-bit-exact and prove it via the test suites; do not silently change
  outputs.
- The per-token expert-weights scale (if the graph has a separate MUL after
  GLU for `expert_weights_scale`) stays a graph node - the fused kernel
  does not touch it. Verify the graph first (milestone 1).
- CUDA-graph capture: the fused op is a single kernel launch, so it stays
  capturable (decode) and direct-evaluable (prefill).

## Milestones

1. **Graph verification** (0.5 d): dump the qwen35moe routed-expert
   subgraph for a prefill ubatch; confirm the node order
   `MUL_MAT_ID, MUL_MAT_ID, GLU`, the src/dst shapes and strides, whether
   the weights-scale MUL is a separate node, and that
   `ggml_cuda_should_fuse_mul_mat()` passes for the Q6_K tensors. Also
   check the qwen4exp graph (same hook or different).
2. **Prototype** (1-2 d): Q6_K fused kernel + dispatch + env toggle.
   Validate bit-exactness against the unfused path with the existing
   fixtures (`test-llama-archs` qwen35moe/qwen4exp legs, and
   `test-backend-ops` mul_mat_id + glu cases if shapes allow).
3. **Measure** (0.5 d): `GGML_CUDA_OP_TIMING` and t/s at ub=512 and 2048,
   pp 2048/16384, fused on vs off. Expect the gate+up pair to lose one
   ids-sort, one quantize, one launch, the intermediate round-trip and the
   GLU pass: optimistic 15-25% of the mmid block (~6-10% of prefill); the
   win should be larger at ub=512 where the fixed overheads dominate.
4. **Generalize** (1-2 d): other quant types; the non-`dedup_bcast` case;
   the host-sync fallback path must stay on the old 3-op sequence.
5. **Cross-validate** (1 d): qwen4exp (Flash-Next) real model - the routed
   experts there have the same pattern; re-check the prefill-vs-35B-A3B
   ratio with `--ubatch-size 2048` + fused MoE; full arch-test suite
   (608 ROCm configs) for regressions.

## Risks

- **Accumulation-order differences** -> not bit-exact; mitigation: enforce
  per-weight K tiling identical to the unfused MMQ config.
- **Kernel complexity / register pressure**: two weight streams per CTA
  doubles the loaded blocks per K step; may need a larger K block or a
  two-pass structure (gate fully, then up) to fit registers. The two-pass
  variant still saves the sort/quantize/launch/intermediate but not the
  weight read.
- **Fallback correctness**: every fused condition must mirror
  `should_fuse_mul_mat` exactly; the env toggle must be a hard opt-out.
- **Scope creep**: do NOT touch the shared-expert chain or the router
  (topk-moe) in this work item - they are separate (the topk-moe fusion
  already exists; the shared-expert chain has its own fused commit).

## Reference material in-tree

- `ggml/src/ggml-cuda/mmq.cu` (mmid MMQ path incl. `dedup_bcast`,
  `ggml_cuda_mul_mat_q`), `ggml/src/ggml-cuda/mmid.cu` (sort).
- `ggml_cuda_can_fuse()` / `ggml_cuda_should_fuse_mul_mat()` in
  `ggml/src/ggml-cuda/ggml-cuda.cu` (~3175, ~1732) - the existing dead
  pattern check.
- `src/models/delta-net-base.cpp` `build_moe_ffn` (graph shape).
- Upstream reference if useful: vLLM `fused_moe` kernel design (grouped
  GEMM with two weight tensors and a fused epilogue).
