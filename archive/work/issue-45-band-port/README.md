# archive/work/issue-45-band-port: porting the RDNA4 GQA-6 decode/verify FA band to RDNA3

**Status: WIP handoff, not part of the delivery.**

Issue #45 (reported by @overdoingism, folded into the r4 delivery as a block-15 amendment) fixed a
real inefficiency on RDNA4: at head 256 with GQA 6 (Qwen3.8-27B: 24 Q / 4 KV heads) the tile flash
attention kernel can only fold `ncols2 = 2`, because its `ncols2` must divide the GQA ratio, so the
whole `n_q <= 8` decode/verify band fetched and dequantized every K/V element once per head pair,
three times per query row.

The r4 fix routes the band to the WMMA kernel with the GQA group folded into `ncols2 = 8` and splits
the KV round-robin over a fixed `P = nsm` blocks per output tile, which keeps decode and every verify
width accumulation-identical. The gate is **RDNA4-only**:

```c
// ggml/src/ggml-cuda/fattn-common.cuh, ggml_cuda_fattn_band_wmma_applies()
if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) {
    return false;
}
```

`amd_wmma_available()` is true for RDNA3 too, so **gfx1151 (RDNA3_5) and gfx1100 (RDNA3_0) still run
the tile kernel with `ncols2 = 2`** and still pay the 3x re-fetch. Whether the WMMA fold *wins* on
those two arches is unmeasured:

- **gfx1100 (RDNA3_0, 96 CU, GDDR6):** most likely to win, since the gfx1201 result came from being
  instruction-issue bound and gfx1100 is the same class of discrete card. Head 256 is exactly at the
  RDNA3_0 WMMA cap (256), so it passes.
- **gfx1151 (RDNA3_5, ~40 CU, shared LPDDR5X):** plausible, especially at verify widths, but RDNA3
  WMMA throughput is lower and memory is shared, so it must be measured.

Neither arch is regressed today (the band is simply off there). The two prompts below are
self-contained handoffs for an agent running on each device.

| prompt | device | what it does |
|---|---|---|
| [`gfx1100-prompt.md`](gfx1100-prompt.md) | RX 7900 XTX, gfx1100, RDNA3_0 | measure the band vs tile, then port into block 15 if it wins |
| [`gfx1151-prompt.md`](gfx1151-prompt.md) | Strix Halo, gfx1151, RDNA3_5 | measure the band vs tile, then port into block 15 if it wins |

Both prompts use the same A/B method: a one-line local relaxation of the gate to
`if (!amd_wmma_available(cc))` (the chooser, the ncols dispatcher and `launch_fattn` all share the
predicate, so this one edit enables the band consistently), then the r4 op-level and end-to-end
gates. A negative result is a valid deliverable.

**If either port wins**, fold it into block 15 as a per-arch amendment exactly like r4, then re-base
`beta/mmb-general` the same way (it should again be a pure SHA/offset re-cut, since the beta set does
not touch the dense `GGML_OP_FLASH_ATTN_EXT` files).
