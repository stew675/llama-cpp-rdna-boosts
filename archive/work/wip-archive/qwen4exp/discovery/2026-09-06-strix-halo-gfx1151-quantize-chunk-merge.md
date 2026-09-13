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

## Repeat-bcast FIXED (commit b987877d7): absorb the block_out REPEAT into hc_combine_norm

Census (first-schedule walk): every pass has 94 combine chains and ALL structurally match the
hc_combine_norm fusion pattern (k=MUL, m=ADD, q=RMS, g=MUL); ~93 fuse - yet 96 standalone
op_repeat kernels ALSO fire (2/layer). The repeats are NOT unfused combines: the graph expands
them BEFORE the fusion window (repeat node index < the scale chain). The fusion deliberately
does not absorb them: (a) reading the repeat INPUT after a standalone dispatch is the
allocator-reuse hazard of the 2026-09-06 fix; (b) a scale-anchored window that included the
in-list repeat must also include the [n_embd,1,T] reshape view, whose external view_src
(block_out, non-constant) fails ggml_can_fuse_subgraph_ext. B avoids the materialization
entirely: its model ggml_set_output's block_out/inject (pins the buffers alive) and
pre-expands block_out + the w scale chain so scale->sigmoid->scale->reshape->repeat->mul->add
is contiguous, and its fusion absorbs the repeat reading the NARROW base (block_out_hc=false).

Fix (ported, both halves): qwen4exp build_hc_combine pins + pre-expands (B's exact lines); a
new repeat-ANCHORED fusion entry in ggml-cuda.cu absorbs [repeat, mul, add, rms, mulg] into
hc_combine_norm_f32 with bo = the pinned narrow base (the scale/sigmoid/scale dispatch
standalone, tiny). Old scale-anchored entry kept for unpinned/old-order graphs (byte-identical
fallback). TRAP: scale1 = scale2->src[0]->src[0] (sigmoid between the two scales); the base
pin check must unwrap the reshape view to block_out (views don't carry the OUTPUT flag).
Correctness: logitcmp deterministic + BIT-IDENTICAL; cli text byte-identical. Kernel deltas
per 4 pp2048 decodes: op_repeat 384->8 (0.182->0.004s), hc_combine_norm 372/0.547s ->
376/0.445s (narrow-bo reads; now FASTER per call than B's 0.468s), capture total 10.250 ->
9.980s. Same-session A/B t/s: pp2048 762.9/773.1 (0.987x, was 0.936/725.4 pre-chunk +
pre-absorb), pp4096 0.996x, pp16384 1.20x, pp512/pp1024 at parity. NOTE: machine-level drift
between sessions reached ~2.5% on B itself (773->754) - always judge by same-session pairs.

## Repeat-bcast investigation (2026-09-06 cont.): hc_combine prefill fusion REVERTED - net-negative

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

## Remaining prefill gaps (post-repeat-absorb capture: A 9.980s vs B 9.663s, +79ms/pass)

Ranked per 4 pp2048 decodes (semantic twins paired; Aabs4_results.db vs Bnow):
1. flash_attn_ext_f16 A 0.562s vs B 0.468s (+94ms; 11.7 vs 9.75ms/call; A smem 33792 vs B 51328) -
   A's NEWER upstream flash vs B's older; needs a source-level diff (A has extra mask/param
   support; B's bigger smem suggests larger tiles).
2. quantize_mmq_q8_1<*,true>+swiglu A 376+376 calls 0.275s vs B 188+188 0.203s (+72ms) - A fires
   2x the CALLS; B's swiglu kernel does 2x rows/call (0.74 vs 0.38ms med): B merges the gated
   (has-gate) feed into the swiglu quantize; A splits. Graph-level merge candidate.
3. GDN scan+kkt A 0.392+0.082 vs B gated_delta_net_tiled 0.428 (+46ms) - A's newer GDN = 2
   kernels (scan + kkt) vs B's single tiled; A per-call faster but two launches.
4. Cijk grid256 bucket A 1.30 vs B 1.18ms/call (+45ms total) - rocblas dense-GEMM shape feeding.
5. k_get_rows A 348 vs B 160 calls (+15ms); launches A 15905 vs B 13457 (+612/pass host).
ALREADY A-FASTER: hc_combine_norm 0.445 vs 0.468, rms_norm family, mul_mat_q parity, mwr parity.
Closing 1+2 (~42ms/pass) puts pp2048 past B (773); 3+4 are the rest of the +79ms.

## Generality of the campaign work (packaging)

- ARCH-LEVEL (gfx1151, any model, gated): split_j Q8_0 J128 config (6d457634e), chunked mmq
  q8_1 quantize (0a3a2b498). Both inert byte-identically off-gfx1151.
- QWEN4EXP-ONLY (this model family): hc fusions + repeat-absorb (b987877d7), QSA shortcuts, PLE
  host-gather, concat/swiglu ports, mmid 512x10, mwr float4, swiglu quantize kernel. The ggml-cuda
  hc_combine/hc_mix fusion code is pattern-gated (scale-sigmoid-scale-repeat-mul-add-rms-gamma)
  and DORMANT for Llama/Qwen-dense/moe architectures.

## gfx1151 gate / deferred validation ledger

- CHUNKED QUANTIZE: gated to gfx1151 ONLY (cc == GGML_CUDA_CC_RDNA3_5+1). gfx1150 (Strix
  Point) = same RDNA3.5 dispatcher, likely benefits, NOT validated. gfx1201/RDNA4 uses its
  own build - n_chunks=1 there (upstream form verbatim); when the gfx1201 box arrives, test
  ggml_cuda_quantize_mmq_q8_1_n_chunks on it (and gfx1150) and re-validate with logitcmp.
- The quantize chunking is arch-agnostic by mechanism (block-dispatch bound) - the gate is
  conservative; NVIDIA/Pascal-dp4a etc. unaffected (n_chunks=1 path byte-identical).

Raw: /tmp/prof/{Anow,Bnow,Achunk,Ac2}_results.db, /tmp/gateA/lg-{chunk,chunk2,head}-*.txt.
