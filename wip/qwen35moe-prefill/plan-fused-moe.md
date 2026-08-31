# PLAN (b): fused MoE gate+up+GLU kernel for the routed experts

Target: the `MUL_MAT_ID` block - 41-53% of qwen35moe prefill GPU time
(report.md sections 2, 5). The same pattern exists in every MoE arch in the
fork (qwen4exp, gpt-oss, glm4-moe, ...), so the work transfers.

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
