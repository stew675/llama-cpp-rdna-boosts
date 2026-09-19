# gfx1100 (RDNA3_0) WMMA flash-attention head cap — 2026-09-18 (r5, issue #30)

Single **RX 7900 XTX** (gfx1100, Navi 31, PCIe, 24 GiB), ROCm 7.14, `HIP_VISIBLE_DEVICES=0`,
build flags `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_HIP_RCCL=1 -DGGML_HIP_GRAPHS=ON
-DGGML_HIP_MMQ_MFMA=ON`, Release.  Reference build = stock `ebbb18522` (`~/llama-stock/build-stock`).
All rows are `llama-bench`, `-ngl 99`, `-p 2048`, `-r 2`-`3`, no other work on the GPU.

## Why this was looked at

The 2026-09-14 block-04 amendment made the RDNA WMMA head-256 config arch-aware and adopted
upstream #28102's AMD `ncols2` rule, but it also overwrote the **RDNA3_0** side of
`ggml_cuda_fattn_mma_get_config_rdna` (heads 320/512/576) with the **RDNA4/gfx1201** rows and lifted
the RDNA3_0 WMMA head cap from stock's 256 to 576.  Neither was re-validated on gfx1100 (the r4
follow-up only fixed the tensor-split `ncols2`).  The test models both have head > 256:
Qwen3.5-9B has head 256 (gqa 4) and Gemma-4-26B-A4B has head 512 (gqa 8 global) + head 256
(gqa 2 SWA).

## Finding 1 (fixed in r5): head-512 WMMA vs tile

`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`, `pp2048 @ d98304` (t/s):

| K/V type | r4 (RDNA4 #28102 row, WMMA) | upstream base row (WMMA) | tile kernel (r5) |
|---|---:|---:|---:|
| f16  | —     | 791.0 | 818.7 |
| bf16 | 655.9 | 774.6 | 851.1 |
| q8_0 | 661.4 | 784.0 | 772.8 |

r4 was **-14 % (q8_0) / -23 % (bf16)** below the tile kernel; the base-row WMMA recovers q8_0 but
still leaves bf16 9 % low.  The tile kernel is exactly what stock `ebbb18522` chooses for head > 256
(`Q->ne[0] <= 256`).  The same ordering held at d65536 and (for f16/bf16) at d32768.

Head 256 must keep WMMA — forcing tile on Qwen3.5-9B (head 256, q8_0), `pp2048`:

| depth | WMMA | tile | delta |
|---|---:|---:|---:|
| 0      | 3777.7 | 3688.5 | +2.4 % |
| 65536  | 1780.3 | 1231.7 | **+44.5 %** |
| 98304  | 1407.4 |  929.2 | **+51.5 %** |

**Fix:** `GGML_CUDA_CC_IS_RDNA3_0(cc) ? 256` (was `? 576`) in `ggml_cuda_get_best_fattn_kernel`;
RDNA4 (576) / RDNA3_5 (320) untouched.  `GGML_CUDA_FA_WMMA_MAX_HEAD` still overrides the cap.

Post-fix (r5) gemma-4-26B-A4B `pp2048`, d0 / d65536 / d98304:

| K/V type | r5 | stock | note |
|---|---:|---:|---:|
| q8_0 | 3635.7 / 1051.3 / 778.2 | 3563 / 1090 / 810 | residual -4 % at depth = finding 2 |
| bf16 | 3627.7 / 1139.6 / 853.3 | 3594 / 1081 / 803 | ahead |
| f16  | 3660.4 / 1112.8 / 818.2 | — | |

Dense 9B (head 256) q8_0 `pp2048 @ d98304` = **1404.7** t/s (r4 1408.1); unchanged, WMMA retained.
`test-backend-ops -o FLASH_ATTN_EXT` **5952/5952** after the change.

## Finding 2 (documented, not changed): q8_0 native-arm prefill trade

With the head-512 path on tile, `GGML_CUDA_FA_KV_NATIVE` still moves gemma-4-26B-A4B q8_0 at d98304:

| setting | pp2048 @ d98304 | tg128 @ d65536 |
|---|---:|---:|
| auto (native arm on) | 773 | 109.9 |
| `GGML_CUDA_FA_KV_NATIVE=0` | **813** (stock 810) | 76.4 (stock 71.5) |

So the block-15 q8_0 native arm costs ~5 % of head-512 deep prefill on gfx1100 but buys **+44 %**
decode at depth — it stays on.  The dense head-256 model shows no prefill effect (1405.9 vs 1404.4),
so the cost is head-512-specific.  Candidate future fix: keep the native decode path but restore the
node-scratch F16 staging for prefill on RDNA3_0.  Tracked in `TODO.md` item 6 and `WORKLOG.md`
2026-09-18 (r5).

## Env knobs used

- `GGML_CUDA_FA_WMMA_MAX_HEAD` — force the tile kernel for heads above the cap (A/B).
- `GGML_CUDA_FA_WMMA_256=0` — force the tile kernel for all heads > 128.
- `GGML_CUDA_FA_KV_NATIVE=0` — force the F16 staging path (no q8_0/q4_0 native arm).
