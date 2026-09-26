# Phase-2 item 10 — MMB quant coverage: Q4_0 / Q4_1 / Q5_0 / MXFP4 / NVFP4, ported 2026-09-22

**Status:** ported, validated, **default ON** on non-RDNA4 (`patches/0017`).  Fork `~/llama.cpp` branch
`gap-closing-r13`, commit `abf3bff76`.

**What it is.** The `mmb` (dequant-to-bf16 WMMA) weight GEMM grew five more GGUF types.  Codes 0–10
were already taken (`0` IQ4_NL, `1` Q8_0, `2` BF16, `3` Q4_K, `4` Q5_1, `5` IQ3_S, `6` Q5_K, `7` Q6_K,
`8` IQ4_XS, `9` Q3_K, `10` IQ3_XXS); the new codes are **11 = Q4_0, 12 = Q4_1, 13 = Q5_0,
14 = MXFP4, 15 = NVFP4**.

Each dequantizer writes bf16 into the LDS A-panel in the layout the existing types use, loading its
exact byte count straight from `Wbase` (the `WTYPE >= 6` "store_lds loads from Wbase" route — no
register over-read):

| type | code | block | bytes / K-step (64 values) | value |
|---|---:|---|---:|---|
| Q4_0 | 11 | `half d; qs[16]` (18 B, 32 vals) | 36 | `d * (nib - 8)` |
| Q4_1 | 12 | `half2 dm; qs[16]` (20 B) | 40 | `d * nib + m` |
| Q5_0 | 13 | `half d; qh[4]; qs[16]` (22 B) | 44 | `d * ((nib \| (qh bit << 4)) - 16)` |
| MXFP4 | 14 | `uint8 e; qs[16]` (17 B) | 34 | `2^(e-127) * kvalues_mxfp4[nib] * 0.5` |
| NVFP4 | 15 | `uint8 d[4]; qs[32]` (36 B, 64 vals) | 36 | `ue4m3(d[sub]) * kvalues_mxfp4[nib]` |

`mmb_dq_row_q4_0` mirrors `mmb_dq_row36` (IQ4_NL), `mmb_dq_row_q5_0` mirrors `mmb_dq_row_q5_1`
(the 5th-bit selection is the same expression), `mmb_dq_row_mxfp4` mirrors Q4_0's nibble positions,
and `mmb_dq_row_nvfp4` handles the one 64-value block per K-step (four 16-value sub-blocks).

## Wiring

Added to **all** dispatch sites, not just the dense one — the first Q4_0 cut missed the inline gfx1151
launch chain and faulted in the `default` (WTYPE 2) arm with a bogus BF16 stride.  The sites a future
type must touch:

* the three `wrow_bytes` ternaries (`mmb_dense_kernel`, `mmb_routed_kernel`, `mmb_tile_gemm_glu`);
* `store_lds_a` in `mmb_tile_gemm` (dense/routed) and `mmb_tile_gemm_glu` (gate+up);
* `mmb_dense_launch_t` (RDNA4 geometry) **and** the inline gfx1151 chain in `ggml_cuda_mul_mat_mmb`;
* `mmb_routed_kernel_dispatch` and `mmb_routed_glu_kernel_dispatch`;
* `mmb_wtype_mask()` (+ a `Q4Q5` default arm for non-RDNA4) and the `GGML_CUDA_MMB_TYPES` /
  `GGML_CUDA_MMB_DENSE_TYPES` name tables.

RDNA4 is deliberately unchanged: `mmb_wtype_mask()` keeps its per-type `IQ_FAMILY` default, so the new
types stay inert there (the same policy as Q8_0/Q5_1/K-quants).

## Gates (gfx1151)

| gate | q4_0 | q4_1 | q5_0 | mxfp4 | nvfp4 |
|---|---:|---:|---:|---:|---:|
| `test-backend-ops -o MUL_MAT` (MMB forced on, `GGML_CUDA_MMB_MIN_T=1`) | **48/48** | **47/47** | **14/14** | **46/46** | **45/45** |
| `test-backend-ops -o MUL_MAT_ID` (same) | **74/74** | **75/75** | **3/3** | **74/74** | **73/73** |
| PPL off → on (16 chunks) | 27.5916 → 27.5439 | 25.7408 → 25.1566 | 22.0234 → 21.6935 | — | — |
| pp throughput off → on | +19.5 % (pp8192) | +22.4 % (pp8192) | +25.4 % (pp8192) | +5.2 % (gpt-oss-20b, pp4096) | — |

* The `MUL_MAT`/`MUL_MAT_ID` oracles are the dequant-vs-CPU-reference gate; MMB interception was
  verified with `GGML_CUDA_MMB_LOG=1`.  **NVFP4 has no model on this box**, so only the oracles gate it
  (its throughput is unmeasured).
* PPL models: Nanbeige-4.2-3B requantized from BF16 to Q4_0/Q4_1/Q5_0; gpt-oss-20b-MXFP4 (8 chunks,
  677.0 → 670.4).  The small shift is the MMB bf16-rounding numerics (the oracle proves the dequant).
* MoE arms: 35B-A3B Q4_1 pp4096 **1500.2 → 2449.1 (+63 %)** exercises `mmb_routed_kernel` and
  `mmb_routed_glu_kernel`; gpt-oss-20b MXFP4 exercises them too.
* Existing types unregressed: `iq4_nl` 14/14, `q8_0` 50/50 (MUL_MAT, MMB forced on).
* Decode is untouched by construction (MMB is prefill-only, `T >= mmb_min_t` = 512).

## Follow-ups (not done here)

* **gfx1100/gfx1201** — the dequant code is arch-neutral (no fragment-layout dependency) and gfx1201
  keeps its per-type policy (inert); the gfx1151 `Q4Q5` default applies to gfx1100 too, which has not
  been re-run.
* `Q2_K`/`IQ1_*`/`IQ2_*` stay deliberately out of scope (quality), per the beta README.

## Files

`ggml/src/ggml-cuda/mmb.cu` only.  Patch: [`patches/0017-…`](patches/).
