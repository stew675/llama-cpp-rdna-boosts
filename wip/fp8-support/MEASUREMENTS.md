# MEASUREMENTS — frozen FP8 baseline (from the `cllm` campaign, 2026-08-06)

Carried from `reference/PERF_HANDOVER.md` / `LEVERS.md`.  Box: 1× AMD Radeon AI PRO R9700 (gfx1201,
64 CUs, 2 SIMDs/CU, wave32, **64 KB LDS/CU**, 8 MB L2), 2350 MHz.  Model: `Qwen3.5-4B StewFP8`
(`/llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf`, 4.17 GiB / 4.33 B params).

Bench rule from the source docs: **bench on a free `-dev ROCm#`** (an occupied GPU makes llama-bench
OOM and silently CPU-offload, which looks like a regression), and prefer **rocprof kernel times
(±3 %)** over `llama-bench` (±60-170 t/s noise).

---

## 1. The scoreboard (Qwen3.5-4B, gfx1201, pp512 unless noted)

| config | pp512 t/s | tg64 t/s | vs Q8_0 pp |
|---|---:|---:|---:|
| **Q8_0 GGUF (target base)** | **6177** | **90.9** | — |
| fp8 baseline (`821d423ac`) | 4930 | 71.0 | -20 % |
| fp8 + aiter GEMM port (`26b7be281`) | 5546-5596 | 71.4 | -5.6 % |
| fp8 + chunked GDN | 6068-6072 | 89.8-90.0 | -1.7 % |
| fp8 + phase A/B rewrite | ~6100-6200 | 89.8-90.1 | ~-1 % |
| fp8 + L2 weight repack | 6319-6334 | 89.4-90.0 | ~+1.5 % |
| fp8 + L5 conv fusion | 6538-6554 | 89.4-90.0 | ~+5 % |
| fp8 + L6 quantize rewrite | 6818-6841 | 89.2-89.6 | ~+7 % |
| fp8 + L7 in-proj fp8 | **7184-7260** | 88.3-88.5 | **~+13 %** |
| fp8 + L9 single-copy embd (final) | **7162-7170** | 88.3-88.6 | **~+13 %** |
| target | 8770 | ~100 | +50 % / +10 % |

PPL 6.2250 (L9) vs 6.2528 (L7) vs Q8_0 6.2464 on `/tmp/corpus_pride.txt` — within error bars.
File: 4.17 GiB / 4.33 B params vs Q8_0's larger footprint.

## 2. Where fp8 pp512 time goes (per-pp512, warmup removed)

| kernel | ms/pp512 | share |
|---|---:|---:|
| `mul_mat_fp8_wmma` | ~39.4 | ~50 % |
| GDN chunked (phase A `gdn_chunk_prepare` ~4.9 + phase B `gdn_chunk_state` ~5.9) | ~10.9 | ~14 % |
| `quantize_fp8` (warp rewrite) | ~2.8 | ~4 % |
| `flash_attn_tile` (8 attn layers) | ~3.0 | ~4 % |
| `silu` | ~2.9 | ~4 % |
| `rms_norm` | ~4.4 | ~6 % |
| `ssm_conv_long_token_2src` (fused) | ~0.6 | ~1 % |
| misc (k_bin_bcast/cpy/get_rows/rope/…) | ~3.3 | ~4 % |

`mul_mat_fp8_wmma` effective ≈ **77 TFLOP/s** (~92-98 on the big shapes in a hot-loop rig).  The
roof: gfx1201 raw WMMA ~370 TFLOP/s → the cllm kernel is at ~21-26 %.

## 3. AITER reference (what vLLM runs on ROCm) — from `reference/AITER_FINDINGS.md`

AITER is vLLM's default ROCm kernel backend.  On gfx1201 CK/ASM are CDNA-only, so AITER dispatches to
**Triton JIT**.  Its `gemm_a8w8_blockscale` matches llama.cpp's fp8 mul_mat 1:1 (per-128 K block,
per-token activation scale, f32 acc).

| fp8 GEMM, M=512, 4B shapes | TFLOP/s |
|---|---:|
| q_proj (4096×2560) | 121.2 |
| o_proj (2560×4096) | 123.4 |
| gate/up (9216×2560) | 136.9 |
| down (2560×9216) | 124.5 |
| lm_head (248320×2560) | 128.7 |
| AMD-tuned shapes (N=8192 K=8192 etc.) | 124.9-131.2 |

⇒ AITER ≈ **121-137 TFLOP/s** (≈35 % of raw) vs the cllm kernel's **77-98**.  That delta is the
remaining lever.  Configs: `~/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE*.json`
(30 files, `M_LEQ_x` selection).  **Negative result to respect:** the `bpreshuffle` (16,16)-shuffled
path is **5-8× slower** on gfx1201; the row-major non-shuffled layout — ours — is correct.

## 4. Post-re-base baselines to (re)capture

The 4B numbers above predate the current delivery base.  `PLAN.md` requires:
* reproduce the 4B +13-17 % on the current base (gate 1);
* the **27B** fp8 number vs the int8 baselines, which are in
  `../prefill-gap-attribution/MEASUREMENTS.md` §1 (Q8_0 1371 / Q6_K 975 / Q4_K_XL 1254 t/s pp8192).
