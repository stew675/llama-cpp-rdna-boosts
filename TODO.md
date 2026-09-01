# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.
Each entry: status, why it matters, what "done" looks like, where the work lives.

## Active (in progress now)

### [IN-PROGRESS] Block 13: MoE WIP -> official patch (qwen4exp split-out)
- Goal: promote the non-qwen4exp MoE work (mmvq item-split + fused gate+up+GLU MMQ
  + M4 quant extension) into the main 12-patch set as a new `0013-*` block.
  qwen4exp work moves OUT to `wip/qwen4exp/`.
- Decision (2026-09-01): NEW 13th block, NOT folded into 04/08/10. Reasons:
  verified 12-set must stay untouched; MoE is a distinct feature; item-split needs
  re-base against the 0eadefebd tree anyway (fusion in mul_mat_vec_q_moe); the
  MoE work is still maturing (M4/M5 open) so it must not pollute release blocks.
- Also: rename block 12 to the 000N convention
  (`12-hybrid-allreduce-hip.patch` -> `0012-rdna-boosts-block-12-*.patch`).
- Re-base already done in ~/llama.cpp working tree (uncommitted):
  - mmvq.cu: item-split merged with has_fusion (tmp_gate accumulation, launcher
    dispatches rpb 2/4/8 x has_fusion true/false).
  - 0004 consolidated (fused MoE MMQ + dispatch + M4 types + op-timing) applied
    as one patch; the separate 0003-fused-moe + 0003-dispatch patches are
    SUPERSEDED by 0004 (it contains both + the RDNA4 gate refinement).
  - test-backend-ops: 2/2 backends OK after fixes (see two fixes below).
- Remaining: build llama-cli/bench, verify decode +6-7% and prefill +5-16%
  (qwen35moe), verify same-seed coherence IDENTICAL, commit on fork after block
  12, extend make-patches.sh + apply-all.sh, update patches/README.md +
  MANIFESTS.md + README.md, regenerate rdna-boosts-all.patch, verify apply on
  fresh checkout at 0eadefebd, then move qwen4exp to wip/qwen4exp/.

### [IN-PROGRESS] ITEM B - sparse QSA flash attention (qwen4exp branch)
- Op + kernel committed on `~/prs/llama.cpp` qwen4exp branch @ `554691a72`
  (tiled + all-heads-per-block). Depth-constant attention: 11.9ms @64K vs dense
  158ms. +6% @16K, +24% @32K, +56% @64K over dense.
- Remaining: latency optimization (~35% GPU, per-warp serial chain, ~3x
  headroom), package as boosts patches, final commits, handoff UPDATE 12+.
- Move to `wip/qwen4exp/` as part of the block-13 split-out (above).

## Flagged follow-ups (not started)

### [OPEN] Multi-token MUL_MAT_ID mmvq `x_scale_channel_dst` fusion
- What: the `ffn_moe_weighted = moe_down * topk_weights` MUL fold (block 08's
  arm in ggml-cuda.cu try_fuse) only supports SINGLE-token decode (n=1). The
  kernel epilogue applies `x_scale[channel_dst]` = one scalar per expert, which
  is correct only when all batch tokens share one expert weight. Multi-token
  topk weights are per-(expert, token) (shape [1, n_used, n_tokens]), which the
  kernel cannot express (assert: `nelements(x_scale) == dst->ne[1]`).
- Why it surfaced NOW: upstream 0eadefebd changed the mmvq gate from
  `dst->ne[2] != 1` (single-token only) to `dst->ne[2] > get_mmvq_mmid_max_batch(...)`
  (admits multi-token), so the block-08 arm now fires for n>1 and the kernel
  assert aborts. Old tree never hit it (old gate blocked n>1).
- Interim fix APPLIED (in the block-13 working tree): gate the arm with
  `mm_node->ne[2] == 1` - restores old verified semantics (multi-token keeps the
  separate MUL op, bit-exact). test-backend-ops green.
- "Done" would be: extend both kernels to index x_scale by (expert, token) and
  remove the gate; validate bit-exactness + perf on spec-dec verify batches
  (n=2..8). Real feature work, NOT a trivial gate removal.

### [OPEN] MXFP4 (and NVFP4) fused gate+up+GLU MMQ (block 13)
- What: the fused MoE MMQ kernel (`ggml_cuda_mul_mat_q_switch_type_gate`) is
  instantiated for Q3_K/Q4_K/Q5_K/Q8_0/Q6_K only. MXFP4/NVFP4 and the other
  mmq_supported types (Q4_0/Q4_1/Q5_0/IQ*/...) fall back to the 3-op sequence.
- Why: `ggml_cuda_should_use_mmq` returns true for these on RDNA4, so the fused
  arm WOULD fire for them, but the gate switch aborts
  (`GGML_ABORT("fused gate MMQ not implemented")`). Interim fix in the block-13
  working tree: arm gated on the exact instantiated type list. MXFP4 is the
  interesting one for future MoE models (deepseek-style native MXFP4 experts).
- "Done": add MXFP4 (+ maybe NVFP4/Q4_0-class) cases to switch_type_gate with
  DECL_MMQ_CASE_GATE instances + generator list + bit-exact + bench validation.
  Follow the 0004 recipe (3 steps per type: gate switch case, instance file +
  generator, validation).

### [OPEN] Port block 13 (fused MoE MMQ) to Strix Halo + 7900XTX
- The fused gate+up+GLU MMQ arm is gated to `GGML_CUDA_CC_IS_RDNA4(cc)`
  (tuned J tile-width caps on gfx1201). Port = validate the fused kernel's
  tile caps + bit-exactness on gfx1151 (Strix Halo, RDNA3.5) and gfx1100
  (7900XTX, RDNA3), then relax the gate per-arch. Follow the 0004 validation
  recipe per arch (bit-exact + bench). Note: Strix Halo also matters for the
  unified-memory decode work (bandwidth-bound). See
  `wip/qwen35moe-prefill/true-q3-rdna4-gating-2gpu.md` for the RDNA4-gating
  history.

### [OPEN] Decode push (tg128 40.1, memory-bandwidth bound)
- Parked after ITEM B. mmvq short-K item-split (block 13) is part of it
  (+6-7% decode on qwen35moe already measured). 2-GPU deploy + Strix Halo
  back-port still parked.

## Closed / archived
- ITEM A (node_56/node_570 one-time hipBLAS JIT): RESOLVED.
- Indexer head-sum fusion (GGML_OP_INDEXER_HEAD_SUM): REVERTED - bit-correct but
  end-to-end regressed (graph splitter perturbation). Lesson: measure
  end-to-end, not op timing, on qwen4exp.
- qwen35moe: dense GQA (no indexer) - QSA op NOT applicable (model change, not
  inference change).
