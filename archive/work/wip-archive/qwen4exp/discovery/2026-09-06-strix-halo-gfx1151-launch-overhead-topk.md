# Strix Halo (gfx1151) — launch-overhead ledger: topk-moe fusion disabled by memory alias (numerics fork found)

Fork tip 376f02aa0 (clean). Investigation of "A fires ~1900 more kernels/pass than B".

## Kernel-count diff (pp2048 r1 same-session captures, A 7943 vs B 7001 = +942)

| kernel | A_c | B_c | d | notes |
|---|---|---|---|---|
| scale_f32 | 714 | 298 | +416 | tiny (~0.006ms); B's 298 are 6x longer - different structure |
| unary_op_kernel | 628 | 248 | +380 | tiny |
| k_argsort | 94 | 0 | +94 | **25ms/capture (the big TIME item)** |
| soft_max (router) | 94 | 0 | +94 | 3ms (folded into B's fused topk) |
| k_get_rows | 174 | 80 | +94 | |
| reduce_rows / op_clamp / l2_norm / rope / moe_weighted | ~94/72/24 each | 0/2 | ... | moe_weighted = B's weighted_expert_sum (same work, name diff) |
| gdn_bf16_scan+kkt | 144 | 72 (tiled) | +72 | the 2-kernel GDN structure (already faster overall post-NW16) |

scale/unary/rms counts differ partly by template naming (B splits rms_norm into more variants) and
partly by real fusion-window differences in the hc elementwise chains - NOT yet root-caused.

## The big one: A's MoE routing runs a FULL 512-expert argsort per token

A: k_argsort_f32_i32 (bitonic sort, 512-thread block per token, full 512 sort) 94 calls @ 0.264ms
= **25ms/capture ~ 12.5ms/eval ~ 0.5% wall**. B: topk_moe_cuda<512,false,true> 96 calls @ 0.022ms
= ~3ms total (partial top-10, one warp per token). A full-sorts 512 scores to take 10.

## Root cause: the CUDA topk-moe FUSION is disabled in A by the memory-ranges check

All the fusion code (ggml_cuda_topk_moe_fusion matcher, ggml_cuda_should_use_topk_moe,
try_fuse call site) is byte-identical between A and B. Instrumented gates: can_fuse=1,
can_fuse_subgraph=1, should_use=1, but **ggml_cuda_check_fusion_memory_ranges = FALSE**.

A's check (newer upstream, line 3222) refuses because the gallocr allocates the fused kernel's
output (ffn_moe_weights_norm, 80KB) INTO the dead ffn_moe_logits buffer (4MB, same address) -
verified: "dst=ffn_moe_weights_norm-N src=ffn_moe_logits-N same 0x...". The check permits
logits/output aliasing ONLY when nrows <= TOPK_MOE_ROWS_PER_BLOCK (one block reads all logits
before writing). qwen4exp pp2048 = 2048 rows >> threshold -> multi-block reads would race the
writes -> the refusal is CORRECT for A's layout. B's older check special-cased only nrows==1 and
its layout happened not to alias, so the fusion fired there.

## The unlock (experimental, REVERTED): pin ffn_moe_logits

ggml_set_output(logits) on the router logits in build_ffn_moe_outer (llama-graph.cpp) prevents
the allocator from reusing the logits buffer -> no alias -> fusion fires:
- k_argsort 94 -> 0; topk_moe_cuda<512,false> 96 calls; capture 4.852 -> 4.826s (-26ms ~ -0.5%)
- Memory cost: all 47 logits [512x2048x4B = 4MB] pinned = ~188MB held for the graph duration.

## NUMERICS FORK (why it was NOT adopted): the fused topk is NOT transparent for qwen4exp

logitcmp on the fixed 838-token prompt: A-unfused top1 logit 18.424 -> A+fused 18.690 (0.27
scale = real behavioral change, NOT ulp; whole token stream diverges from step 0). B = 18.086 -
NEITHER A variant matches B. The fused kernel's internal softmax/top-k/weights differ from the
plain ggml chain (softmax -> argsort -> get_rows -> sum/clamp/div) for qwen4exp's router shape.
Three-way divergence means adopting the fusion = a numerics re-baseline with an UNKNOWN
correctness direction (the kernel may be subtly wrong for this config, or the unfused chain
may be the deviant - needs a CPU-reference + PPL/KL quality gate to decide).

## Verdict + follow-up

The launch-overhead item's biggest component (~0.5%) is available ONLY via a numerics decision
(the fused topk adoption, quality-gated) or a layout fix that keeps the unfused numerics. Open
leads for a future session:
1. topk fusion: quality-gate the fused kernel vs the plain-op chain (CPU ref + PPL/KL). If the
   fused behavior validates as equivalent-or-better, adopt the pin (with a scoped cost ~188MB).
   If not, look for a layout-only fix (non-pinning way to avoid the logits/output alias so the
   UNFUSED chain is untouched and only the wasted full-sort is... not possible without the
   fusion - the fusion IS the fix for the sort waste).
2. scale_f32/unary_op excess (+800 launches): compare the hc elementwise fusion windows A vs B
   (the repeat-absorb window is [repeat,mul,add,rms,mulg]; B's window may include the upstream
   scale/sigmoid/scale chain).
3. GDN 2-kernel +72 launches: cosmetic once the NW16 win is banked.

Tree clean at 376f02aa0, bit-identical. All probes reverted.
