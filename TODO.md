# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.
Each entry: status, why it matters, what "done" looks like, where the work lives.

## Priority (next session)

### [NEXT TASK] Multi-token MUL_MAT_ID mmvq `x_scale_channel_dst` fusion
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
- Interim fix APPLIED (in the block-13 tree): gate the arm with
  `mm_node->ne[2] == 1` - restores old verified semantics (multi-token keeps the
  separate MUL op, bit-exact). test-backend-ops green.
- "Done": extend both kernels to index x_scale by (expert, token) and remove
  the gate; validate bit-exactness + perf on spec-dec verify batches (n=2..8).
  Real feature work, NOT a trivial gate removal.
- Where: `ggml-cuda.cu` try_fuse (block-08 arm) + the mmvq kernel epilogue in
  `mmvq.cu` (`mul_mat_vec_q` / `mul_mat_vec_q_moe` x_scale handling).
- Note: this is the block-08 arm, NOT the block-13 fused-gate MMQ (that one is
  the prefill path). Verify the arm is still the mmvq single-token fold and
  that multi-token spec-dec verify batches (n=2..8) are the target.

### [OPEN] File upstream llama.cpp issue: unaligned-width split-load on ROCm
- The block-13 load fix (below) works around an UPSTREAM bug: H2D 2D copies
  with width not multiple of 4 (Q6_K/Q3_K quant blocks = 210/110B) are
  ~1000x slower on ROCm (present in pristine 0eadefebd). Fix is local in
  `set_tensor_2d` (staging via aligned H2D + unaligned D2D). Candidate to
  contribute upstream once local validation settles; needs a minimal repro
  (standalone hipMemcpy2DAsync: width 210 -> 2300ms, width 212 -> 0ms).

## Active (in progress now)

### [IN-PROGRESS] Block 13: MoE WIP -> official patch (qwen4exp split-out)
- Goal: promote the non-qwen4exp MoE work (mmvq item-split + fused gate+up+GLU MMQ
  + M4 quant extension) into the main 12-patch set as a new `0013-*` block.
  qwen4exp work moves OUT to `wip/qwen4exp/`.
- Decision (2026-09-01): NEW 13th block, NOT folded into 04/08/10. Reasons:
  verified 12-set must stay untouched; MoE is a distinct feature; item-split needs
  re-base against the 0eadefebd tree anyway (fusion in mul_mat_vec_q_moe); the
  MoE work is still maturing (M4/M5 open) so it must not pollute release blocks.
- DONE (2026-09-01): block 12 renamed to the 000N convention
  (`0012-rdna-boosts-block-12-hybrid-HIP-all-reduce-RDNA4-gat.patch`);
  block 13 committed on the fork (`a14257996`), 13-patch series regenerated
  and verified (clean git am apply, applied tree byte-identical to fork
  tip, zero whitespace warnings).
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
- DONE (2026-09-01): block 13 released (fork a14257996, 13-patch set
  regenerated + verified, docs updated, all.patch regenerated); qwen4exp
  moved to `wip/qwen4exp/` (plan + handoffs + patches 0005-0007).
- DONE (2026-09-01): block-13 commit amended with the unaligned-width
  split-load fix (fork 834a8d3ff), 13-patch set + all.patch regenerated,
  fresh-apply re-verified byte-identical, both repos pushed.
- VALIDATION DONE (2026-09-01, 1-GPU qwen35moe = the verified config):
  - test-backend-ops: 2/2 backends OK (after 2 fixes below)
  - Coherence: fusion on/off same-seed greedy output IDENTICAL
  - Prefill pp16384: Q6_K +5.1% (3344 vs 3181), Q4_K_M +3.6% (3488 vs 3367)
  - Decode tg128: +5.6% (97.28 vs 92.15 pristine) - item-split + fused gate
  - Fused path fires: FUSED MUL_MAT_ID ffn_moe_down-* on all layers
- The 2 fixes vs the verified 0004 patch (both in ggml-cuda.cu):
  1. x_scale_channel_dst arm gated to mm_node->ne[2] == 1 (multi-token falls
     back to separate MUL - restores old verified behavior; upstream 0eadefebd
     gate change admits multi-token which the n=1-only kernel can't express).
  2. fused MoE MMQ arm gated to the instantiated type list (Q3_K/Q4_K/Q5_K/
     Q8_0/Q6_K) - q4_0/q4_1/q5_0/IQ/MXFP4 were aborting
     (GGML_ABORT "fused gate MMQ not implemented").
  Both are correctness fixes required for the multi-token tests upstream added.

### [IN-PROGRESS] ITEM B - sparse QSA flash attention (qwen4exp branch)
- Op + kernel committed on `~/prs/llama.cpp` qwen4exp branch @ `554691a72`
  (tiled + all-heads-per-block). Depth-constant attention: 11.9ms @64K vs dense
  158ms. +6% @16K, +24% @32K, +56% @64K over dense.
- Remaining: latency optimization (~35% GPU, per-warp serial chain, ~3x
  headroom), package as boosts patches, final commits, handoff UPDATE 12+.
- Move to `wip/qwen4exp/` as part of the block-13 split-out (above).

## Flagged follow-ups (not started)

### [OPEN] Port + validate block 12 (hybrid HIP all-reduce) on dual 7900XTX
- What: block 12's internal AR is hard-gated to RDNA4 (`gfx1200`/`gfx1201`
  only, `strncmp` on gcnArchName in allreduce-hip.cu; refuses to init
  elsewhere and falls back to RCCL). 7900XTX = gfx1100 (RDNA3), which
  upstream RCCL supports but whose peer-to-peer / queue-preemption behavior
  the internal AR's in-kernel spin + hybrid dispatch were never validated
  against (the spin timeout work, PR #8, was tuned on 2x gfx1201).
- Work: relax the arch gate for gfx1100, validate the hybrid dispatch
  matrix (internal vs nccl vs none) + the bounded-spin path on a 2x7900XTX
  box, confirm no MES REMOVE_QUEUE/MODE1-reset regressions at depth-16384.
- **Needs community assistance**: the author has NO machine with dual
  7900XTX. Ask for help in the blocks repo (issue or a 7900XTX-owner
  thread); the gate makes the fallback safe (RCCL) so a volunteer can test
  with `GGML_CUDA_ALLREDUCE=internal` and report the matrix.

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
- MOVED to Priority (next task) 2026-09-01.

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
- qwen35moe MoE + multi-GPU "hang" (Q5_K/Q6_K): RESOLVED + FIXED. Root cause:
  pathologically slow 2-GPU tensor-split weight load for quants with block
  size % 4 != 0 (Q6_K=210B, Q3_K=110B) - H2D 2D copies with unaligned width
  take ~2300ms vs ~10ms on ROCm; the ~3min load looked like a hang. UPSTREAM
  bug (present in pristine 0eadefebd); the earlier "hangs at EVERY point"
  bisect was wrong (timeouts < slow load). Fixed in block 13 (fork
  834a8d3ff, amended): `set_tensor_2d` stages via aligned H2D + unaligned
  D2D. Verified: Q6_K/Q3_K 2-GPU load <15s, pp512 ~4200-4500 / tg32
  ~66-82 (matches docs-era 4526/82); Q8_0 unchanged; fresh-apply of the
  13-patch set byte-identical. Pushed to stew675/llama.cpp rdna-boosts +
  delivery repo main. See `wip/qwen35moe-prefill/handoff-2026-09-01-
  session4.md` for the full investigation.
- ITEM A (node_56/node_570 one-time hipBLAS JIT): RESOLVED.
- Indexer head-sum fusion (GGML_OP_INDEXER_HEAD_SUM): REVERTED - bit-correct but
  end-to-end regressed (graph splitter perturbation). Lesson: measure
  end-to-end, not op timing, on qwen4exp.
- qwen35moe: dense GQA (no indexer) - QSA op NOT applicable (model change, not
  inference change).
