# Patch 0002: mmvq short-K item-split (mmvq.cu)

Status: WIP - NOT integrated into the patch set. Applied in the
`~/llama.cpp` working tree; decide later whether to integrate.

## Problem

The decode mmvq kernels split the K dimension across fixed thread groups
(16 groups of 16 threads for Q6_K). Short-K decode GEMMs have few K-blocks
per row, so most groups idle:

| kernel | K | K-blocks | groups active | measured |
|---|---|---|---|---|
| routed down (MMID, 8 experts) | 512 | 2 | ~12% | 53 us @166 GB/s |
| router logits | 2048 | 8 | 50% | 19 us |
| attention/recurrent qkv | 2048 | 8 | 50% | 40 us |
| gate+up (MMID) | 2560 | 10 | ~63% | 37 us @506 GB/s |
| lm_head | 248320 | 970 | 100% | 864 us @600 GB/s |

The routed down was the only compute-bound victim (its 53 us/layer x 40 =
2.1 ms of decode). The rest are launch/overhead-bound and stayed flat.

## Change

1. `mul_mat_vec_q` (and `mul_mat_vec_q_moe`): the K-split loop
   (`kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter`)
   becomes an item-split loop that distributes (token, row, kblock) items
   over ALL thread groups: `it = tid/(qi/vdr); it < n_items; it += n_groups`
   with `n_items = ncols_dst * rows_per_block * blocks_per_row_x`.

2. The ncols_dst==1 dispatch (RDNA2+ tables only): rows_per_block is
   chosen to fill the thread groups: `rpb = roundup2(n_groups / blocks)`
   capped at 8 (empirically best; caps above 8 lose occupancy to the
   cross-warp shared reduction). The old `use_rpb_moe` hardcode (rpb=2)
   is replaced by this general selection; other tables keep their tuned
   config via the rpb=0 template path.

3. `mul_mat_vec_q_moe_launch`: same rpb selection for the multi-token
   mmid kernel (rpb 2/4/8 runtime switch).

## Results (gfx1201, qwen35moe Q6_K, single GPU)

- Decode tg128: 92.8 -> 98.2-99.0 t/s (**+6-7%**), flat with depth
  (98.4 at KV=16K).
- Routed down kernel: 53.2 -> 37.7 us (-29%); gate+up unchanged (already
  full); router/qkv/shared experts flat (launch-bound, need fusion not
  mapping).
- Prefill: unchanged (mmvq not used at ub>=512).
- Correctness: test-backend-ops MUL_MAT + MUL_MAT_ID 2065/2065 pass
  (5e-4 tolerance); real-model generation coherent.

## Caveat

The item-split changes the per-row accumulation order, so mmvq outputs
differ in the last bits vs the pre-change build (same class as the bf16
GDN path). The speculative-verify constraint (decode vs n_draft+1 batch
must share the kernel config) is preserved - both use the new mapping.
