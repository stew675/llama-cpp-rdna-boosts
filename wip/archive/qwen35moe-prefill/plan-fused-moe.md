# PLAN (b): fused MoE gate+up+GLU kernel for the routed experts

## Status: Milestones 1-3 DONE, Milestone 4 in progress (2026-08-31)

Built, measured, committed (patch 0003, boosts `72605d1`): the prefill
{MUL_MAT_ID(gate), MUL_MAT_ID(up), GLU} triple now runs as ONE MMQ kernel
reading both weight streams + GLU epilogue in registers. Q6_K only.

Results (J<=64 cap): pp512 ub=512 +14-16%, pp2048 ub=2048 +6.3%, pp16384
+5.8%, pp32768 +4.9% - depth-flat, beats the milestone-1 ~3% revision.
Bit-exact (536M floats 0 diff), full test-backend-ops passes, decode
unchanged (98 t/s, mmvq owns it). Full detail: fused-moe-mmq-result.md,
milestone4-handoff.md.

Key implementation facts (complete recipe in milestone4-handoff.md):
has_gate template on mul_mat_q_process_tile/mul_mat_q; J<=64 cap for the
fused kernel (doubled accumulator + WMMA tiles spill at J=128: 256 VGPR +
14 spills); gate switch in mmq.cu routes on fusion->gate; try_fuse MMQ arm
with GGML_CUDA_DISABLE_MOE_MMQ_FUSION opt-out; DECL_MMQ_CASE_GATE +
generator MMQ_GATE_TYPES list.

## Remaining

- M4 (next): enable Q4_K, Q5_K, Q8_0 - 3 steps per type (gate switch case,
  DECL in the instance file + generator list, bit-exact + bench validation).
- M5: qwen4exp cross-validation; server --ubatch-size 2048 deployment test.

## Historical record

### Milestone 1 (graph verification, boosts `9f05cf8`)

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
6. **Win estimate REVISED DOWN then UP**: milestone-1 measured SWIGLU_SPLIT
   at only 1.54% and the round-trip ~2.3%, so the plan predicted ~2-3%.
   The actual fused kernel measured 6-16% because the win is the fused
   kernel's tile behavior at small J (J<=64 cap avoids the J=128 register
   spill) plus eliminating the GLU pass/round-trip, not just the shared
   sort/quantize/launch.

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
