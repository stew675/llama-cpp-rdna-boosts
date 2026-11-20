# `rdna2-moe-rows-abreast` — optional block `0104`: RDNA2 (`gfx1030`) `mul_mat_vec_q_moe` walks the K-blocks with the rows side by side

**Optional artifact, NOT part of the delivery.**  RDNA2 (`gfx1030`) only: the new loop is compiled for the RDNA2
device pass and armed by the launch from the device's compute capability.  Nothing here is in `patches/`,
`release.json.tree`, `rdna-boosts-all.patch` or the default `scripts/apply-all.sh` apply.

* `0104-rdna2-moe-rows-abreast.patch` — the artifact.  sha256 `6051fb4e62fb2bcee2b263434913fe1bbe40db74f385f7f7ba1eefe16198d160`
* Cut against the r325 default tree; on r327 the same bytes apply strictly (`git am`) on the default tree
  `4792f98b82a4c6433bf11c629fdf01c21a143ce9` (`e159adfc2`), producing tree `23a76dae1a649b494c3832ea05aabe770c1c7801`,
  and in numeric order after `0100` + `0101` + `0102` + `0103` (cumulative tree
  `8dfb3d65948de4369b86a83fa18b95c1630a0482`).

## What it changes

**`ggml/src/ggml-cuda/mmvq-moe.cu`** and one field in **`ggml/src/ggml-cuda/common.cuh`**.

`mul_mat_vec_q_moe` takes one (row, K-block) item at a time: a divide and a modulo per item, and one dot product
in flight per thread.  Every routed `MUL_MAT_ID` width up to the band limit goes through this kernel, `n == 1`
included (the has_ids branch that keeps decode and verify on one accumulation order), so on `gfx1030` the small
widths were slower than ggml-org's kernel, which walks the K-blocks with two rows unrolled.

* On RDNA2 the kernel walks the K-blocks once and keeps the rows of the block side by side.  A thread group `g`
  takes the items `it == g (mod n_groups)`, so in row `i` it takes the K-blocks
  `kbx == g - i*blocks_per_row (mod n_groups)` in rising order.  The new loop visits exactly those K-blocks in
  that order for each row, so every partial sum is built the same way and **the output is bit-identical**.
  * `blocks_per_row % n_groups == 0`: every row keeps the same K-blocks, so one `y` block serves all rows.
  * otherwise: each row starts at its own first K-block (`(g - i*blocks_per_row) & (n_groups - 1)`).  `IQ2_XS`
    keeps the item loop in this case — its two-items-in-flight unroll is faster there.
* The switch is `bool moe_rows_abreast` in `ggml_cuda_mm_fusion_args_device` (default `false`); only
  `mul_mat_vec_q_moe_launch` sets it, from `GGML_CUDA_CC_IS_RDNA2(cc)`.  The kernel reads it inside
  `#if defined(RDNA2)`; every other device pass gets `constexpr bool rows_abreast = false` and compiles the item
  loop alone.
* `GGML_CUDA_MOE_ROWS_ABREAST=0` restores the item loop.
* Rows per block, the grid, the warp reduction and the write-back are untouched.

## Measured on a Radeon Pro V620 (`gfx1030`, ROCm 7.14, idle card)

Correctness, the r325 cut with `0100`..`0104` applied, built with the default HIP flags
(`test-backend-ops test -b ROCm1`): `-o MUL_MAT_ID` 1056/1056 with the switch on and with
`GGML_CUDA_MOE_ROWS_ABREAST=0`; `-o RMS_NORM` 121/121, `-o GATED_DELTA_NET` 55/55, `-o FLASH_ATTN_EXT` 7405/7405.

Bit-identity, raw `MUL_MAT_ID` output bytes of a standalone ggml graph (20 quantized types — `q4_0 q4_1 q5_0 q5_1
q8_0 mxfp4 q2_K q3_K q4_K q5_K q6_K iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m iq4_nl iq4_xs` — on shapes
with `n` = 1, 3, 4, 5, 8, 16, odd and even row counts, and `k` = 96..3328 so that the blocks per row are both a
multiple and not a multiple of the group count):

* default HIP flags, switch on against `GGML_CUDA_MOE_ROWS_ABREAST=0` in the same build: 194 cases, 0 differences;
* a `-funsafe-math-optimizations` build, switch on and off against a build without this block: 194 cases,
  0 differences.

(The 256-wide block types cannot be given a `k` that is not a multiple of 256; those combinations are not in the
counts.)

`test-backend-ops perf -o MUL_MAT_ID`, geometric mean over the quantized cases at each `n`, the block against the
same tree without it (two runs of each within 0.2 % of each other), and both against ggml-org `c406f4367` on the
same card:

| `n` | this block vs without | without vs ggml-org | with vs ggml-org | cases |
|---|---|---|---|---|
| 1 | 1.32x | 0.98x | 1.26x | 22 / 15 |
| 4 | 1.30x | 0.83x | 1.07x | 17 / 15 |
| 5 | 1.26x | | | 21 |
| 6 | 1.29x | | | 16 |
| 7 | 1.30x | | | 16 |
| 8 | 1.31x | 1.29x | 1.68x | 17 / 15 |
| 32..2048 | 1.00x | | | 91 |

(`n >= 32` does not run this kernel.)  Single cases: `q4_K` `[1792 x 2048]` `n = 1` 23.8 -> 12.2 us, `n = 4`
57.9 -> 39.7 us; `q8_0` `[768 x 2048]` `n = 1` 23.1 -> 14.5 us; `mxfp4` `[2880 x 2880]` (90 blocks per row,
16 groups) `n = 4` 102.9 -> 73.8 us.

Still behind ggml-org with the block applied (0.87-0.95x): `q4_0` and `q8_0` at `n = 8`, `mxfp4`
`[2880 x 2880]` at `n = 4` and `8`, `q8_0` `[1792 x 2048]` at `n = 1`.  ggml-org's kernel shares the K-block
across rows in every case; when the blocks per row are not a multiple of the group count that is a different
accumulation order from this tree's, so it is not taken here.

## Not measured

Any arch other than `gfx1030` (the loop is not compiled there); a model-level decode or verify run.  The perf
numbers were taken on the r324 default tree + `0100`..`0103`, in a build whose HIP flags included
`-funsafe-math-optimizations`; the r325 cut was built with the default flags and run through the correctness and
byte checks only.  On r327 the strict apply and the trees above were re-checked, and the r327 default tree +
`0100`..`0104` + the fix of PR #339, built with `-funsafe-math-optimizations`, passes `-o MUL_MAT_ID` 1056/1056,
`-o RMS_NORM` 121/121, `-o RMS_NORM_MUL_ADD` 113/113, `-o GATED_DELTA_NET` 55/55 and `-o FLASH_ATTN_EXT` 7405/7405
on `ROCm1`; the byte and perf runs were not repeated on r327.
