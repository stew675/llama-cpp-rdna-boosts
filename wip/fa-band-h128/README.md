# wip/fa-band-h128 — the RDNA4 decode/verify band for 128-dim heads whose GQA ratio is not a multiple of 8

Written by Claude (AI).  One `git am` file, one commit, made on `v17-4f9dfa0e3-r305` (tree `ef8a4adc`, rebuilt here from the
release's block patches on `4f9dfa0e3`; the tree and tip match `release.json`).

| file | touches | sha256 |
|---|---|---|
| `0001-FA-band-128-dim-heads-with-a-GQA-ratio-that-is-not-a.patch` | `ggml/src/ggml-cuda/fattn-rdna.cuh`, `fattn.cu`, `fattn-mma-f16.cuh`, `tests/test-backend-ops.cpp` (+58 −3) | `639dcb16aca5dbd83986fa43ddad63d7406799d7dfe3ea0ab98dcfed85873063` |

## What it changes

Outside the band, a decode or verify step (1..8 query rows) of a GQA model folds only the largest power of two that divides the
GQA ratio into the tile (`gqa_ratio_eff`), and the tile kernel needs every column of a block on one K/V head.  For an odd ratio
nothing folds: each query head reads its K/V head on its own, so a GQA 7 model reads the K/V cache seven times per step (five
times for GQA 5).  The band already folds the group into `ncols2 = 8` and masks the unused columns (GQA 6 runs that way today),
but it only takes 256-dim heads.

Models this reaches: 128-dim heads with a GQA ratio of 5, 6 or 7, or above 8 but not a multiple of 8, for example
Nanbeige4.3-33B (56 query / 8 K/V heads, the one we measured), Qwen2.5-7B (28 / 4), Qwen2.5-14B / -32B and Qwen3-14B (40 / 8).
Ratios that are a multiple of 8 fold whole in the generic dispatch and stay there.

## Change

- `fattn-rdna.cuh`, `ggml_cuda_fattn_band_wmma_applies`: also 128-dim heads (`DKQ == DV == 128`) when the ratio is not a multiple
  of 8; the other conditions are unchanged (ratio > 4 and <= `GGML_HIP_FA_BAND_WMMA_GQA_MAX`, mask, alignment, K/V types).
  Kill switch: `GGML_HIP_FA_BAND_WMMA_H128=0` (read once).
- `fattn.cu`, `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`: for 128-dim heads in the band, the 32-column block
  (`ncols1 = 4`, or 2 with `GGML_HIP_FA_BAND_WMMA=2`) for every query width; widths 5..8 take two output tiles.  There is no
  64-column band row (`band_wide`) for this head size, and no q8_0 fetch-ahead variant: both are left out on purpose.
- `fattn-mma-f16.cuh`: the band's 2-D launch path is also compiled into the 128-dim instances with `ncols1 <= 4`.  No new config
  rows and no new template instances: the existing 128-dim `ncols1 2 / 4, ncols2 8` instances and config rows are used.
- `tests/test-backend-ops.cpp`: eval cases for 128-dim heads with GQA 5, 6, 7 and 12 on 8 and 4 K/V heads (q8_0 and f16 K/V,
  kv 512 / 1024 / 4096 with a fully masked 256-cell group, 1..16 rows), and perf cases for GQA 5, 7 and 8.

## Results

Linux, Ryzen 9 3900X, one R9700 (gfx1201), ROCm 10.0.0.  Measured on r295, where `fattn*.cu(h)` are the same as in r305; on r305
the patch was rebuilt and the correctness, width and microbench checks rerun (below).  `ROCBLAS_USE_HIPBLASLT=0` in every arm.

**Correctness.**  test-backend-ops `FLASH_ATTN_EXT`, every 128-dim case: 1981/1981 with the band on and off.

**Width purity.**  k query rows at once against each row alone, bit for bit (head 128, 56 / 8 and 40 / 8 heads, q8_0 and f16 K/V,
positions 1000..32766): no difference for k = 1..8 with the band on or off.  Whole model (Nanbeige4.3-33B, 16K and 32K prefixes,
400 teacher-forced tokens): width 8 equals width 1 bit for bit in all 16 runs with the band on.  9..16 rows do not take the band,
as before.

**Other models unchanged.**  Teacher-forced log-probs (8K prefix, 400 tokens, widths 1 and 8) against stock r295: identical bit
for bit with the band on for Qwen3.8-27B, Qwen3.6-27B (head 256, GQA 6), Gemma 4 12B and Nemotron 3.5 30B-A3B (head 128, GQA 16);
Nanbeige4.3-33B identical with the band off.

**Attention microbench** (`test-backend-ops perf`, us per op, stock S vs the band on; the band off is within 0.4 % of S, and so
is GQA 8 with the band on):

| GQA | kv | rows | S | band | band / S |
|---|---|---|---|---|---|
| 7 | 4096 | 1 | 68.1 | 49.6 | 0.73 |
| 7 | 16384 | 1 | 293.9 | 85.4 | 0.29 |
| 7 | 16384 | 8 | 884.1 | 147.6 | 0.17 |
| 7 | 45056 | 1 | 697.0 | 232.2 | 0.33 |
| 5 | 16384 | 1 | 200.0 | 85.1 | 0.43 |
| 5 | 45056 | 8 | 1789.5 | 356.9 | 0.20 |

**Decode end to end** (llama-bench, Nanbeige4.3-33B Q4_K_M, `-fa 1`, 64 tokens, 3 repetitions, arms interleaved; tokens/s):

| K/V | depth | stock | band off | band on |
|---|---|---|---|---|
| q8_0 | 0 | 27.77 / 28.07 | 28.09 / 28.07 | 26.58 / 26.59 |
| q8_0 | 4096 | 25.72 / 25.81 | 25.85 / 25.85 | 25.47 / 25.48 |
| q8_0 | 16384 | 20.78 / 20.77 | 20.78 / 20.77 | 24.02 / 24.02 |
| q8_0 | 32768 | 16.36 / 16.37 | 16.36 / 16.36 | 22.25 / 22.26 |
| f16 | 0 | | 28.17 | 27.21 / 27.22 |
| f16 | 4096 | | 26.41 | 26.09 / 26.10 |
| f16 | 16384 | | 22.36 | 23.49 / 23.53 |
| f16 | 32768 | | 18.32 | 20.83 / 20.82 |

The trade-off: below about 4-8K cells the band is slower (−5 % at depth 0 with q8_0, −3 % with f16).  Its K/V split is a fixed
P = nsm (nsm / 2 for f16), so a short cache leaves most blocks of a tile with nothing to do and adds the fixup combine per layer.
P cannot follow the K/V length without making decode and verify split differently near a length boundary, so this patch keeps the
band's rule; a gate on the K/V length would need the same care.

**KLD** (Nanbeige4.3-33B, 400 teacher-forced tokens, the reference is the band off with an f16 cache; mean without the largest):
six zh / en texts at a 16K prefix and two at 32K.  With a q8_0 cache the band and the tile kernel land on either side of each
other (4 texts each; mean 0.0157 vs 0.0224 over the eight).  With an f16 cache the band against the tile kernel is 0.001-0.029,
smaller than stock's own 16-row kernel against its 1-row kernel on the same texts (0.006-0.07).

**On r305** (this patch on `v17-4f9dfa0e3-r305`, rebuilt): test-backend-ops `FLASH_ATTN_EXT`, every 128-dim case, 1981/1981
with the band on and off; k = 1..8 rows bit-identical to single rows in every shape above; the microbench within 1 % of the r295
numbers (GQA 7, 16K cells, 1 row: 293.5 → 85.3 us; GQA 5, 45K, 8 rows: 1788.5 → 357.4 us; GQA 8 and the band off within 0.3 % of
stock r305).  The end-to-end, KLD and other-model runs above are r295's.

## Notes

- The 128-dim `ncols2 = 8` MMA instances were already spilling (90 VGPRs at `ncols1 = 4` in r295); compiling the band path in
  raises that to 122-127 (`ncols1 = 4`) and 74 (`ncols1 = 2`).  The GQA 8 cases that use these instances measure the same as stock
  (table note above).
- Not tested: Windows, RDNA3 / RDNA3.5, several GPUs, bf16 K/V, GQA 5 end to end (synthetic only), models other than the ones
  listed.
