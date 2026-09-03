# Patch 0002: mmvq short-K item-split (mmvq.cu)

Status: WIP - NOT integrated into the patch set. Applied in the
`~/llama.cpp` working tree; decide later whether to integrate.

## Problem

The decode mmvq kernels split the K dimension across fixed thread groups
(16 groups of 16 threads for Q6_K on RDNA4, nwarps=8). Short-K decode
GEMMs have few K-blocks per row, so most groups idle:

| kernel | K | K-blocks | groups active | measured |
|---|---|---|---|---|
| routed down (MMID, 8 experts) | 512 | 2 | ~12% | 53 us @166 GB/s |
| router logits | 2048 | 8 | 50% | 19 us |
| attention/recurrent qkv | 2048 | 8 | 50% | 40 us |
| gate+up (MMID) | 2560 | 10 | ~63% | 37 us @506 GB/s |
| lm_head | 248320 | 970 | 100% | 864 us @600 GB/s |

The routed down was the only compute-bound victim (53 us/layer x 40 = 2.1
ms of decode). The rest are launch/overhead-bound and stayed flat.

## Change

1. `mul_mat_vec_q` (and `mul_mat_vec_q_moe`): the K-split loop becomes an
   item-split loop that distributes (token, row, kblock) items over ALL
   thread groups: `it = tid/(qi/vdr); it < n_items; it += n_groups` with
   `n_items = ncols_dst * rows_per_block * blocks_per_row_x`.

2. The ncols_dst 1..8 dispatch (RDNA2+ tables only): rows_per_block chosen
   to fill the thread groups AND to keep the per-row chunk->lane mapping
   identical for every ncols_dst (see Greedy Purity below):
   `rpb = max(ceil(n_groups/(ncols*blocks)), n_groups/gcd(blocks,n_groups))`
   rounded up to a power of two, capped at 16. The old `use_rpb_moe`
   hardcode (rpb=2) is replaced; other tables keep their tuned config via
   the rpb=0 template path.

3. The cross-warp reduction in `mul_mat_vec_q` is restructured: each warp
   reduces its own lanes first (fixed warp_reduce tree), then one value
   per warp is summed serially by warp 0. The shared footprint drops from
   `[nwarps-1][ncols][rpb][32]` to `[nwarps-1][ncols][rpb]`, which is what
   makes the uniform rpb feasible at ncols 5..8 (the old 4D shared blew
   the 64 KB limit at rpb>=8, ncols>=5). This is a new rounding path
   (deterministic; not bit-identical to the pre-change build).

4. `mul_mat_vec_q_moe_launch`: same rpb selection for the multi-token mmid
   kernel (rpb 2/4/8). Its mapping is per-warp kbx-only, so it was already
   ncols-invariant.

## Results (gfx1201, qwen35moe Q6_K, single GPU)

- Decode tg128: 92.8 -> 98.2-99.0 t/s (**+6-7%**), flat with depth
  (98.4 at KV=16K). Same with ub=512 (98.4).
- Routed down kernel: 53.2 -> 37.7 us (-29%); gate+up unchanged (already
  full); router/qkv/shared experts flat (launch-bound, need fusion not
  mapping).
- Prefill: unchanged or slightly better (pp16384 ub2048: 4325 -> 4367).
- Correctness: test-backend-ops MUL_MAT + MUL_MAT_ID all pass (5e-4
  tolerance); real-model generation coherent.

## Greedy Purity (see GREEDY-PURITY.md)

Same class as block 10: an mmvq accumulation-order change. The dot
products are exact (dp4a), only the fp32 association order changes ->
last-bit logit drift, bounded (block 10 measured <=0.184 logits), PPL
unchanged, greedy deterministic within a build, diverges from other
builds only at near-ties.

Difference vs block 10: block 10's VDR mapping depends only on (kbx, vdr),
so the speculative verify batch (ncols 2..8) stays per-row bit-identical
to single-token decode within a build (GREEDY-PURITY.md section 6,
invariant 2). The item-split's item index is (j*rpb + i)*blocks + kbx,
whose j-offset j*rpb*blocks must vanish mod n_groups - the rpb formula
above enforces rpb*blocks == 0 (mod n_groups), which restores the
invariant for the plain-MUL_MAT path. (The mmid path n>=2 already used a
different kernel - mul_mat_vec_q_moe - with a per-warp reduction, so its
n=1 vs n>=2 rows were never bit-identical; this patch does not change
that.)

Caveat unchanged: outputs differ in the last bits vs the pre-change build
(any mmvq rounding change does, block 10 included). Excluding the patch
restores stock bits.
