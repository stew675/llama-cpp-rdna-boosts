# `rdna2-fattn-tile-caps` — optional block `0103`: lift stale RDNA2 (`gfx1030`) FlashAttention tile caps

**Optional artifact, NOT part of the delivery.**  RDNA2 (`gfx1030`) only: the caps it removes were already
RDNA2-conditional.  Nothing here is in `patches/`, `release.json.tree`, `rdna-boosts-all.patch` or the default
`scripts/apply-all.sh` apply.

* `0103-rdna2-fattn-tile-caps.patch` — the artifact.  sha256 `bb6f2256537d14efda9724552faaa2c4099a22e6f9dc4bd926bea3dee0e94ded`
* Cut against the r324 default tree; on r327 the same bytes apply strictly (`git am`) on the default tree
  `4792f98b82a4c6433bf11c629fdf01c21a143ce9` (`e159adfc2`), producing tree `9f492674f79a1f9c7731e07d9bc2dd4b684971ca`,
  and in numeric order after `0100` + `0101` + `0102` (cumulative tree `3d6d7b953633620981b520af1c23a59623f5f95e`).

## What it changes

**`ggml/src/ggml-cuda/fattn-tile.cuh`** only.

* `launch_fattn_tile_switch_ncols1`: the RDNA2 `cols_per_block` caps (`rdna512` D=512 <= 8, `rdna_d128` D=128 <= 8,
  `rdna576` D=576 <= 16, `rdna256_mha` D=256 MHA <= 8) are lifted, so prefill uses the full 32/64-column tiles.
  `launch_fattn` computes RDNA2 occupancy itself, so the occupancy-0 abort these caps worked around no longer
  happens.
* `launch_fattn_tile_switch_ncols2`: the `ncols2 <= 2` cap for D=128 is lifted (it already was for D=256), so
  GQA-8 D=128 decode no longer re-reads the KV cache 4x.
* `ggml_cuda_fattn_tile_get_config_amd_rdna2` is untouched.
* The five flags stay as `false` constants on purpose: editing the tests that read them would put this
  artifact's hunks on lines optional block `0101` changes, and it could then not apply strictly both alone and
  after `0101`.

## Measured on a Radeon Pro V620 (`gfx1030`, ROCm 7.14, idle card)

Correctness (`test-backend-ops test -b ROCm1 -o FLASH_ATTN_EXT`, the r324 cut with `0100`..`0103` applied, built with
the default HIP flags): 7405/7405, no GPU memory fault.

`test-backend-ops perf -o FLASH_ATTN_EXT`, geometric mean over the cases in each group:

| head size, K type, mode | vs the capped tiles | vs ggml-org `c406f4367` | cases |
|---|---|---|---|
| D=128 f16 prefill | 1.88x | 1.06x | 15 |
| D=128 q8_0 prefill | 1.73x | 0.95x | 2 |
| D=128 f16 decode (nb=1) | 1.21x | 0.98x | 15 |
| D=512 f16 prefill | 2.15x | 1.02x | 10 |
| D=576 f16 prefill | 1.33x | 1.05x | 11 |
| D=576 q8_0 prefill | 1.47x | 0.99x | 1 |
| D=256 f16 prefill | 1.09x | 0.92x | 21 |
| D=256 q8_0 prefill | 1.12x | 0.87x | 5 |

Every other group (D=64, D=256/512/576 decode) is within 1 % of the capped tiles; the slowest single
`FLASH_ATTN_EXT` case is 0.95x.

## Not measured

Any arch other than `gfx1030` (the change is inert there); a model-level prefill run.  The perf numbers above
were taken on the r321 default tree + `0100` + `0101` + `0102`, in a build whose HIP flags included
`-funsafe-math-optimizations`; r322 changed block 05's band fixup period after that, and the r324 cut was built
with the default flags and run through the correctness tests only.
