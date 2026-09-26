# Baseline measurements — 2026-09-26

Frozen reference for the MMQ-pipeline campaign.  Measured on the same box as the rest of the repo:
AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.14, `~/llama.cpp` @ `527d39401`, `build-rocm`.
Reproduce with `tools/repro.sh`.

---

## 1. Instruction-level ceiling (from `archive/work/q8-prefill-tuning/`)

Per-GPU synthetic WMMA loops (8 independent accumulators, operand-free):

| instruction | measured |
|---|---:|
| `v_wmma_i32_16x16x16_iu8` (int8) | **174 T-MAC/s** (348 TOPS) |
| `v_wmma_f32_16x16x16_fp8_fp8` | 171 T-MAC/s |

Whole-model Q8_0 MMQ (`rocprofv3`, 2 GPU `-sm tensor`):

| config | MMQ rate | % of ceiling |
|---|---:|---:|
| pp512, ub512 | 53.6 T-MAC/s/GPU | **31 %** |
| pp2048, ub2048 | 59.4 T-MAC/s/GPU | **34 %** |

Kernel-time share (ub2048, pp2048, per GPU): `mul_mat_q` **50 %**, `ncclDevKernel` 28 %,
`quantize_mmq_q8_1` 3.7 %, rest 18 %.  The kernel-time sum equals wall time → **nothing overlaps**.

## 2. Isolated prompt GEMM (this box, `test-backend-ops perf`)

`m=17408, n=512, k=5120` (27B ffn):

| type | µs/run | TFLOPS | T-MAC/s | % of 174 |
|---|---:|---:|---:|---:|
| F16 | 852 | 107.1 | — | — (hipBLAS, not MMQ) |
| Q8_0 | 1101 | 82.9 | 41.5 | **24 %** |
| Q4_K | — | 74.9* | 37.5 | 22 % |
| Q6_K | 1599 | 57.1 | 28.5 | **16 %** |

\* Q4_K at `m=4096,n=512,k=14336`; see the full landscape in
`../per16-f16-mma/MEASUREMENTS.md` §2.2.

F16/bf16 reference for the headroom (rocBLAS, `m=4096,n=512,k=14336`): f16 93.7, bf16 90.9 TFLOPS.

## 3. LDS budget on gfx1201

`hipGetDeviceProperties`: `sharedMemPerBlock = sharedMemPerBlockOptin = 65536` (64 KiB);
`warpSize = 32`, `multiProcessorCount = 32`.

Current MMQ shared memory at the Q8_0 `I=128, J=128` config:

| piece | bytes |
|---|---:|
| `tile_x` = `I * sram_stride * 4` (stride 70) | 35 840 |
| `tile_y` = `J * MMQ_TILE_Y_K * 4` (MMQ_TILE_Y_K = 33) | 16 896 |
| `data_mul_mat_q` header (`+J` ints) | 512 |
| **total** | **~53 248** (~11 KiB free) |

⇒ a second `tile_x` (35.8 KiB) or `tile_y` (16.9 KiB) does **not** fit at this geometry.
`I=64` (`tile_x` 17.9 KiB) leaves room for `2×tile_x + tile_y = 52.7 KiB`.

## 4. The serial k-loop (source map)

| file | line | role |
|---|---|---|
| `ggml/src/ggml-cuda/mmq.cuh` | 895-1005 | `mul_mat_q` outer k-loop: `load_tiles` → sync → `vec_dot(k00=0)` → sync → load y half 1 → sync → `vec_dot(k00=32)` → sync |
| `ggml/src/ggml-cuda/mmq.cuh` | 864-885 | `ggml_cuda_mmq_get_util_funcs` (per-type `load_tiles`/`vec_dot` selection) |
| `ggml/src/ggml-cuda/mmq.cuh` | 1657 | `launch_mul_mat_q` (config, `nbytes_shared`, block dims) |
| `ggml/src/ggml-cuda/mmq.cuh` | 434 | `ggml_cuda_mmq_get_nbytes_shared_x` |
| `ggml/src/ggml-cuda/mmq.cuh` | 1595 | `mul_mat_q_routed_compact` (MoE; same load/vec_dot shape — must keep working) |
| `ggml/src/ggml-cuda/mmq.cuh` | 1030 | `mul_mat_q` template signature (`<type, J, fallback, has_gate>`) |
| `ggml/src/ggml-cuda/mmb.cu` | ~850-875 | the delivery's existing LDS double-buffer (`DBUF`) pattern, as a reference |

## 5. Why bit-identity is expected to hold

`vec_dot` writes into `sum[]` per output element; the order of contributions is fixed by the
`kb0` / `k00` call sequence and the `k01` loop inside each `vec_dot`.  Moving a *load* earlier in time
does not change that sequence, so the floating-point accumulation order — and the output — is
preserved.  Any design that changes the `vec_dot` call grouping (e.g. collapsing the two `y` halves
into one call) must re-verify the same-seed text hash.
