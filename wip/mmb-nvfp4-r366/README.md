# wip/mmb-nvfp4-r366: cuda MMB -- NVFP4 dequant with the E2M1 table in registers, and NVFP4 blocks prefetched in load_regs

Thank you for MMB and its NVFP4 support. Its per-weight-type structure made these two changes easy to keep small. We measured them while running NVFP4 MoE models and are sharing the data and two patches in case they're useful. Nothing is urgent on our side; we run them in our own build in the meantime.

**What we saw.** In `mmb_tile_gemm`, the NVFP4 path (WTYPE 15) reads its weight bytes inside `store_lds_a`, between the barriers around the LDS write. Each E2M1 value is one load from the `kvalues_mxfp4` table, and `mmb_pack2` adds rounding to a product that is already exact in bf16. The other block types (e.g. WTYPE 0) fetch their block into registers in `load_regs` instead, so the load overlaps the previous K step's WMMA.

**0001** (`mmb.cu`, +44/-15): the E2M1 values come from a table held in four registers (`mmb_fp4_lut4`) via byte permutes, not one table load per value. The bf16 pack drops the rounding adds (`mmb_pack2_exact`). A UE4M3 x E2M1 product has at most 6 significant bits (4 + 2) and stays inside the bf16 exponent range, so bf16 holds it exactly. `GGML_CUDA_MMB_FP4_PERM=0` selects the old code at all four NVFP4 dispatch sites (dense 128-wide, the RDNA4 256x128 dense tile, routed, routed GLU). The old code is kept under a separate template value (WTYPE 19); it is not a new weight format.

**0002** (`mmb.cu`, +32/-1): WTYPE 15 fetches its 36-byte block in `load_regs` like WTYPE 0 (two `uint4` loads and one `uint32`), and the UE4M3 scale is converted without branches. `GGML_CUDA_MMB_FP4_PERM=0` turns this off too, so it selects stock behaviour for both patches.

**Exactness.**
- UE4M3: we checked the conversion against `ggml_cuda_ue4m3_to_fp32` on all 256 codes. That covers zero, subnormals, 0x7F (0, as in ggml), the ignored sign bit and the 0.5 factor ggml applies.
- E2M1: we checked the register table against `kvalues_mxfp4` on all 16 indices, and `mmb_pack2_exact` against `mmb_pack2` on every scale x value product.
- Loads: by inspection, the three loads read bytes 0-35 of the block and nothing past it. Block bases are 4-byte aligned (36-byte stride). This is the same `uint4` idiom WTYPE 0 already uses for its 36-byte blocks.
- Registers, gfx1100: on the large tiles WTYPE 15 uses 8-9 more VGPRs than the old code (e.g. dense 128x128 246 -> 254, routed 128x128 173 -> 182). On the small routed tiles it uses fewer (77 -> 63). There are no spills.

**Results.** "Old" and "new" are the same build (r366 `e10879b0b3` + 0001 + 0002) with `GGML_CUDA_MMB_FP4_PERM=0` and with the default. Setup: one GPU, `llama-bench -fa 1 -ub 512 -ctk q8_0 -ctv q8_0`. Greedy replies from `logs/depth2.py` (a ~2-3K and a ~23-30K token prompt depending on the tokenizer, up to 256 tokens) are byte-identical between old and new in every row below and in the UD-IQ4_XS run. There were no GPU faults.

| GPU | model | path | pp512 old -> new | pp2048 old -> new |
|---|---|---|---|---|
| RX 7900 XTX (gfx1100) | Qwen3.8-27B NVFP4 | dense, default | 931 -> 1125 (+20.8 %) | 917 -> 1108 (+20.8 %) |
| R9700 (gfx1201) | Qwen3.8-27B NVFP4 | dense, `MMB_TYPES=nvfp4 MMB_DENSE_TYPES=nvfp4` | 1141 -> 1433 (+25.6 %) | 1133 -> 1429 (+26.1 %) |
| R9700 (gfx1201) | Inkling-2-Mini UD-Q8_K_XL (NVFP4 experts) | routed, `MMB_TYPES=nvfp4 MMB_ROUTED=1` | 2150 -> 2587 (+20.3 %) | 2007 -> 2374 (+18.3 %) |

- **Stock vs old.** On the R9700, stock r366 with MMB forced gave 1155 / 1145, about 1 % above the old path (1141 / 1133; three repeats each). The greedy replies are identical to the old and new arms.
- **Decode.** MMB runs only for prefill ubatches of 512 tokens or more, so decode isn't touched.
- **RDNA4 default.** On RDNA4 the patches change nothing by default. NVFP4 isn't in the RDNA4 MMB type mask, so it runs on MMQ unless MMB is opted in as above. Where the mask is open the gain applies by default: gfx1100, and we'd expect gfx1151, though we haven't measured it there.
- **Inkling rows.** These come from our own branch, because stock r366 doesn't load `inkling2_loop`. Its NVFP4 MMB code is identical to these two patches. On gfx1100 the same patches gave Inkling-2-Mini UD-IQ4_XS +13.2 % at pp4096 (1960 -> 2219).
- **MMB vs MMQ (not part of this PR).** With the patches, MMB beats MMQ for NVFP4 on the R9700 at pp512: dense 1433 vs ~1115 (+29 %), routed 2587 vs 1868 (+38 %). The MMQ figures are runs from the same session without MMB opted in (`logs/r9700-nvab.log`, `logs/r9700-nvab3.log`). Switching the RDNA4 mask would change output against MMQ, so we haven't proposed it. We'd measure KLD first if that's of interest.

The patches apply cleanly to r366 `e10879b0b3`, and we're glad to rebase onto whatever you're on. We're happy to re-run any of this on a build or settings you'd like, or to test any change you make to the patches.

**Logs.** `logs/gfx1100.log` (+ `gfx1100-q38.sh`, `depth2.py`): the 7900 XTX row. `logs/r9700-nvab2.log`: the R9700 dense row and stock MMB. `logs/r9700-nvab4.log`: the R9700 routed row. `logs/r9700-nvab.log` / `r9700-nvab3.log`: the MMQ arms (MMB not opted in, where the switch makes no difference).
