# Strix Halo (gfx1151) — mmq q8_1 feed quantize: block-dispatch-bound; halo-box 2-slice chunking merged (GATED to gfx1151)

Date: session continuing the shared-path campaign. Tip: 0a3a2b498 (clean).

## Fresh same-session re-derivation (A 6d457634e vs B c7af5c6c2, pp2048 r3, rocprofv3 pair)

Captures: /tmp/prof/Anow_results.db (A 10.445s / 15181 calls) vs /tmp/prof/Bnow_results.db
(B 9.663s / 13457). A/B kernel totals +0.782s/4 decodes (~+195ms/pass) with mul_mat_q (all
variants) now AT PARITY after the split_j fix (2.08 vs 2.07s), mmid/mwr fixed, GDN/scan
A-faster. Remaining real deltas (paired by semantics; name mangling differs because A's
kernels carry extra fusion/glu args):

| family | A tot | B tot | delta | cause |
|---|---|---|---|---|
| quantize_mmq_q8_1<*,false> (D4 feed) | 0.600 | 0.392 | +0.208 | 2x blocks/call (dispatch-bound) -> FIXED |
| k_bin_bcast (op_repeat family) | 0.181 | 0.008 | +0.173 | A 384 calls vs B 12 (graph structure) |
| hc_combine_norm_f32 | 0.550 | 0.468 | +0.082 | A +22% per call (grid 4096, sh 128) |
| Cijk GEMM (grid256/sh2048 bucket) | 0.548 | 0.453 | +0.094 | A 1.44 vs B 1.19ms/call (rocblas shapes) |
| flash_attn_ext_f16 (48 calls) | 0.560 | 0.468 | +0.092 | A 11.66 vs B 9.75ms/call, sh 33792 vs 51328 |
| quantize_mmq_q8_1_swiglu / <*,true> | 0.264 | 0.203 | +0.061 | A fires 2x the calls (376 vs 188 each) |

## Root cause of the quantize gap (FIXED in 0a3a2b498)

1768 identical calls both trees on identical tensors (position-aligned gridDim.y was EXACTLY
2x B's at every call: A {5,20,12,1} vs B {3,10,6,1} x-groups). A (upstream b10837) launches
one 512-float slice per 128-thread block (block_num_y = ceil(ne0/512)); B loops
CUDA_QUANTIZE_MMQ_CHUNKS_PER_BLOCK=2 slices per block. Feed tensors have up to 262144 rows,
so unchunked calls launch ~10-40M tiny blocks and are block-DISPATCH-bound on RDNA3.5 (that
is why 2x fewer blocks = ~1.5x faster wall even with identical per-element work; also why the
earlier "A +52% per call, unexplained" attribution was wrong - it was never per-element work).

Fix design (BEST OF BOTH, gated): B's 2-slice chunk loop merged into A's kernel with A's
surroundings preserved. Traps found:
- The per-channel y-stride must switch from the grid-derived upstream formula
  (gridDim.x*gridDim.y*blockDim.x/QK8_1) to the exact tensor formula (gridDim.x*ne0/QK8_1_MMQ):
  the chunked grid can no longer express it (upstream formula would under-run by ~2x ->
  z-channel overlap). At n_chunks==1 the upstream formula is kept verbatim.
- The chunk count CANNOT be a compile-time __gfx1151__-gated define: the HIP host pass does
  not define __gfx*__ macros, so host gridDim.y (1) and device loop (2) disagreed -> correct
  but wasteful early-exit blocks (0.600 -> 0.398s, gridDim.y unchanged). Fixed by making
  n_chunks a runtime kernel argument threaded from mmq.cu's cc via
  ggml_cuda_quantize_mmq_q8_1_n_chunks(cc) = (cc == GGML_CUDA_CC_RDNA3_5+1) ? 2 : 1.

Validation: logitcmp DETERMINISTIC + BIT-IDENTICAL to the pre-change reference (per-slice
element math unchanged); cli text byte-identical. quantize_mmq_q8_1<*,false> 0.600 -> 0.394s
(B: 0.392); gridDim.y now exactly B's set {3,10,6,1}.

## Depth-0 matrix AFTER the merge (same-session r3, A/B t/s)

| row | A | B | A/B | (was A/B) |
|---|---|---|---|---|
| pp512 | 634.5 | 649.3 | 0.977 | (0.986) |
| pp1024 | 712.5 | 729.6 | 0.977 | (0.958) |
| pp2048 | 736.2 | 773.7 | 0.952 | (0.936) |
| pp4096 | 725.9 | 732.6 | 0.991 | (0.976) |
| pp8192 | 702.5 | 679.0 | 1.035 | (1.022) |
| pp16384 | 702.1 | 599.3 | 1.171 | (1.148) |
| tg128 | 25.92 | 25.94 | parity | |

## Remaining open deltas (next attacks, per-call proof pending)

1. k_bin_bcast op_repeat: A fires 384 calls/pass (2 per layer) vs B 12 - find the graph node;
   if A's newer graph broadcasts something per layer B hoists/fuses, safe restructure worth
   ~+45ms/pass and -96 launches/pass.
2. hc_combine_norm_f32 +22%/call and Cijk grid256 bucket +21%/call: both per-call kernel/graph
   differences (A newer hc/gdn vs B older single-kernel GDN; rocblas shape feeding).
3. flash_attn_ext_f16 A 11.66 vs B 9.75ms/call on identical grid: A's newer flash uses 33KB vs
   B's 51KB shared; needs a source-level comparison.
4. swiglu/scatter quantize A fires 2x calls (376 vs 188): graph-level (B merges has_gate work
   into fewer quantize passes).

## Repeat-bcast investigation (2026-09-13 cont.): hc_combine prefill fusion REVERTED - net-negative

The per-layer repeat (A 96/pass at grid 1280x4 ~0.47ms = the build_hc_combine b-broadcast of
block_out (n_embd,1,nt) -> (n_embd,hc,nt)) looked removable: ggml's k_bin_bcast indexes src0
with RAW dst indices (no mod) - only src1 broadcasts - so the repeat is NOT redundant; the
real lever is the fused GGML_OP_HC_COMBINE op (hc-mix.cu hc_combine_kernel), which was
decode-only (nt==1 assert + inject read at t=0 only), while the op builder (ggml.c) and CPU
ref (ops.cpp) were already token-general with bit-exact RN math.

Attempted: (1) generalized hc_combine_kernel to nt>1 (blockIdx.y = token, per-token w_s,
stride-based block_out indexing - layout note: prefill block_out is a view with nb[1]=n_embd*4
regardless of ne[1]); (2) lifted the nt==1 gate in build_hc_combine. Result: logitcmp
BIT-IDENTICAL (the fused kernel IS numerically exact), repeat gone (384->8) BUT net WORSE:
+372 rms_norm_f32<1024,false> AND +368 k_bin_bcast<op_mul> appeared (+0.63s) - the custom
GGML_OP_HC_COMBINE node broke the scheduler's pattern fusion at ggml-cuda.cu ~4919, which
matches the STANDARD chain scale(sigmoid)->scale->repeat->mul->add->rms->mul(gamma) and
consumes the repeat into ONE hc_combine_norm_f32 dispatch (its bo/GGML_OP_REPEAT handling at
~4985). With the chain replaced by the custom op the following rms_norm+gamma unglued into
separate rms_norm_f32<1024> + op_mul kernels. REVERTED (tree back at 0a3a2b498).

KEY UNRESOLVED: A fires BOTH ~93 hc_combine_norm_f32/pass (2/layer - the codebase fusion IS
active at prefill, n_tok>=1 allowed) AND ~96 op_repeat/pass (2/layer). Only ~half the
combines fuse; need a per-pass op census (single-pass graph walk) to see which combine
instances the fusion misses and why (node-order contiguity? bo/repeat edge cases? PLE or
layer-boundary combines). B: 94 hc_combine_norm_f32 but only 12 repeats/4pass -> B's graph
never emits the wide block_out repeat (B keeps block_out hc-wide or its fusion consumes every
combine). A's hc_combine_norm_f32 is also +22%/call slower than B's (1.48 vs 1.21ms med) on
identical counts. Both remain open.

## gfx1151 gate / deferred validation ledger

- CHUNKED QUANTIZE: gated to gfx1151 ONLY (cc == GGML_CUDA_CC_RDNA3_5+1). gfx1150 (Strix
  Point) = same RDNA3.5 dispatcher, likely benefits, NOT validated. gfx1201/RDNA4 uses its
  own build - n_chunks=1 there (upstream form verbatim); when the gfx1201 box arrives, test
  ggml_cuda_quantize_mmq_q8_1_n_chunks on it (and gfx1150) and re-validate with logitcmp.
- The quantize chunking is arch-agnostic by mechanism (block-dispatch bound) - the gate is
  conservative; NVIDIA/Pascal-dp4a etc. unaffected (n_chunks=1 path byte-identical).

Raw: /tmp/prof/{Anow,Bnow,Achunk,Ac2}_results.db, /tmp/gateA/lg-{chunk,chunk2,head}-*.txt.
