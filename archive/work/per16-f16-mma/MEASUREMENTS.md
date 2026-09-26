# Baseline measurements — 2026-09-26

All numbers on `qwen35` 27B (Qwen3.8-27B), **single R9700 gfx1201** unless stated.
Fork tree `~/llama.cpp` @ `527d39401` (block-15 tip) + build `build-rocm` (ROCm 7.14 gfx1201).
Reproduce with `tools/repro-gemm-perf.sh` and the commands inline below.

> `/home/stew675/llama.cpp` and `build-rocm` (not `build-rocm-hybrid`) are the paths on this box.

---

## 1. Whole-model prefill (`llama-bench`, `-ngl 99 -n 0`)

Run: `tools/repro-gemm-perf.sh model` (see the script), or directly:

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
~/llama.cpp/build-rocm/bin/llama-bench -ngl 99 -p 512,8192,32768 -n 0 -r 3 \
  -m /llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf
```

### 1.1 Single GPU

| model | pp512 | pp2048 | pp8192 | pp32768 |
|---|---:|---:|---:|---:|
| Q6_K | 1034.8 | 1020.2 | 981.6 | 897.3 |
| Q4_K_XL (UD) | — | — | — | 1129.4 |
| Q8_0 | 1444.1 | — | — | 1207.0 |

Ratios: Q6_K/Q8_0 = **0.74** (pp512 0.72, pp32768 0.74); Q6_K/Q4_K_XL = **0.79** (pp32768).
Visible at pp512 ⇒ GEMM cost, not context.

### 1.2 Two GPUs, `-sm tensor` (the maintainer's cross-check)

`HIP_VISIBLE_DEVICES=0,1 … -sm tensor -p 8192`:

| model | pp8192 |
|---|---:|
| Q6_K | 1466.5 |
| UD-Q4_K_XL | 1804.2 |
| Q8_0 | 1955.1 |

Q6_K/Q8_0 = **0.75**, Q6_K/Q4_K_XL = **0.81** — the same deficit, so the all-reduce is not involved.

---

## 2. Isolated prompt GEMM (`test-backend-ops perf`)

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
~/llama.cpp/build-rocm/bin/test-backend-ops perf -b ROCm0 -o 'MUL_MAT.*' \
  -p 'q6_K.*m=17408,n=512,k=5120'
```

### 2.1 27B ffn shape `m=17408, n=512, k=5120`

| type | µs/run | TFLOPS |
|---|---:|---:|
| F16 | 852 | 107.1 |
| Q8_0 | 1101 | 82.9 |
| Q6_K | 1599 | **57.1** |

`m=5120, n=512, k=17408` (ffn_out): Q6_K 1627 µs = 56.1 TFLOPS, F16 902 µs = 101.2 TFLOPS.

### 2.2 Full type landscape `m=4096, n=512, k=14336`

| type | TFLOPS | | type | TFLOPS |
|---|---:|---|---|---:|
| f16 | 92.0 | | q2_K | 2.6 (fallback) |
| bf16 | 89.9 | | **q3_K** | **57.4** |
| q4_0 | 80.9 | | q4_K | 74.9 |
| q4_1 | 82.5 | | q5_K | 70.5 |
| q5_0 | 73.9 | | **q6_K** | **55.1** |
| q5_1 | 51.9 | | iq2_xxs | 79.6 |
| q8_0 | 83.9 | | **iq2_xs** | **58.4** |
| q1_0 | 63.4 | | **iq2_s** | **57.6** |
| q2_0 | 84.8 | | iq3_xxs | 78.5 |
| mxfp4 | 81.2 | | iq1_s | 75.1 |
| nvfp4 | 54.4 | | iq1_m | 69.3 |
| | | | iq4_nl | 84.1 |
| | | | iq3_s | 70.1 |
| | | | iq4_xs | 84.5 |

**Every per-16-scale type (bold) is in the 55-58 TFLOPS band; every per-32 type is 70-85.**

---

## 3. Experiments run this session (all reverted; tree left clean)

### 3.1 The J tile does not help
`GGML_CUDA_MMQ_J_MAX = 16,32,48,64,80,96,112,128` ⇒ 56.3-56.7 TFLOPS flat.  (At `J_MAX=64` the
config is `nthreads=128, I=64`; identical, so nthreads/I do not help either.)

### 3.2 Block 04's register hoist is a real +47 % win
Reverting just the AMD body of `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` to the fork point
(`84e76d8a2`) drops the ffn shape to **38.9 TFLOPS** (from 57.1); `m=4096,k=14336` → 37.6.
⇒ the current kernel is *already* the optimised one; this campaign must beat 57, not 39.

### 3.3 `mmb`/bf16 loses for Q6_K
`GGML_CUDA_MMB_TYPES=q6_k GGML_CUDA_MMB_DENSE=1 GGML_CUDA_MMB_DENSE_TYPES=q6_k`, `pp8192`:

| `GGML_CUDA_MMB_GEOM` | t/s |
|---|---:|
| `1` (RDNA4 256×128) | 638.7 |
| `0` (gfx11 split) | 733.2 |
| MMQ (reference) | **981.6** |

The on-the-fly dequant (`mmb_dq_row_q6k`) is scalar ⇒ dequant-bound.  Matches the delivery's
"Q6_K loses on RDNA4" policy (`mmb_wtype_mask`).

### 3.4 A 2-sub-block restructure is slower *and* wrong
Restructuring the AMD branch to process two 16-K sub-blocks per `j0` pass (mirroring the Turing
branch) measured **49.7 TFLOPS** and failed `test-backend-ops` ⇒ reverted.  The RDNA4 WMMA fragment
layout interleaves K across warp halves, so a merged tile misassigns the per-16 scales.

---

## 4. What the kernel actually does (code map)

| file | line | role |
|---|---|---|
| `ggml/src/ggml-cuda/mmq.cuh` | 795 | dispatch: `GGML_TYPE_Q6_K` → `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` |
| `ggml/src/ggml-cuda/mmq.cuh` | 795 | `Q3_K`/`IQ2_S`/`IQ2_XS` → `ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma` |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | 1029 | `q6_K_q8_1_mma`: `tile<16,4,int>` A/B, `k01 += 4`, per-16 `x_s2` epilogue |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | 496 | `q8_0_16_q8_1_mma`: same shape, per-16 `x_df` epilogue |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | 946 | `load_tiles_q6_K`: expands 6-bit → int8 in sram (`__vsubss4`) + scales |
| `ggml/src/ggml-cuda/mma.cuh` | ~1280 | RDNA4 F16 WMMA: `tile<16,16,float> D, tile<16,8,half2> A, tile<16,8,half2> B` → `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` |
| `ggml/src/ggml-cuda/mmb.cu` | 704 | `mmb_dq_row_q6k`: the existing scalar Q6_K→bf16 dequant reference |
| `ggml/src/ggml-cuda/mmf.cuh` | 50 | `mul_mat_f`: the F16 WMMA GEMM framework (K-column A access → wrong for Q6_K blocks) |

### 4.1 RDNA4 WMMA instruction ceilings (from `archive/work/q8-prefill-tuning/`)

| instruction | measured |
|---|---:|
| `v_wmma_i32_16x16x16_iu8` (int8) | 174 T-MAC/s |
| `v_wmma_f32_16x16x16_fp8_fp8` | 171 T-MAC/s |
| current MMQ (whole model, Q8_0) | 59.4 T-MAC/s = **34 %** of ceiling |
| Q6_K isolated ffn (57 TFLOPS) | ≈ 28.5 T-MAC/s = **16 %** of ceiling |

Both the missing double-buffering (general) and the per-16 epilogue (this campaign) are in play.
