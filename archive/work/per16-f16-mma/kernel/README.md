# Standalone spike (optional fast-iteration path)

`PLAN.md` Phase 1 is the **in-tree** spike, which is preferred because its number is directly
comparable to `test-backend-ops perf` and its loader/vec_dot become the real implementation.

This directory is the fallback if the in-tree wiring turns out to be fiddly: a self-contained
`q6k_f16_gemm.hip` compiled with `hipcc` (no CMake, no llama.cpp), reproducing the same tiling and
measuring pure kernel TFLOPS at the ffn shape.  It answers only "can a fused Q6_K→F16 WMMA beat
57 TFLOPS on gfx1201?"; it is not a correctness oracle.

## Spec

| | |
|---|---|
| arch | `--offload-arch=gfx1201` |
| shape | `M=17408, N=512, K=5120` (27B ffn), fp32 output |
| A | synthetic Q6_K weights (`block_q6_K`, 256 values / 210 B), M×K row-major |
| B | F16 activation, K×N, or F32 cast once in a preamble (measure both) |
| tile | `BM=128, BN=128`, K-step 64 or 128, `nthreads=256`, LDS double-buffered |
| intrinsics | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` |
| reference | all-zeros / noise input + a CPU dequant check on a few tiles |
| report | µs/run and TFLOPS, ≥ 10 warmup + 100 timed iterations |

## Pinned facts (do not re-derive)

* RDNA4 F16 WMMA fragment: `tile<16,16,float> D`, `tile<16,8,half2> A/B`, 8 halves (4 `half2`) per
  lane for A/B and 8 floats for D (`mma.cuh:1292`).
* `load_ldmatrix(tile<16,8,half2>)` reads 16 B from row `tid%16` at `half2` offset `4*(tid/16)`
  (`mma.cuh:844`), i.e. lanes 0-15 hold K 0..7 and lanes 16-31 hold K 8..15 of a 16×16 A fragment.
* Q6_K dequant: `value[j] = d * scales[j/16] * (q6[j] - 32)`, `q6 = (ql nibble) | ((qh 2-bit) << 4)`.
  Verified reference: `mmb_dq_row_q6k` in `ggml/src/ggml-cuda/mmb.cu:704`.
* gfx1201 int8 WMMA ceiling is 174 T-MAC/s and F16 is comparable — 57 TFLOPS is ~16 % of it, so
  there is room; the target is 70+.

## Status (2026-09-26): **parked — incorrect, and no longer the preferred path**

`q6k_f16_gemm.hip` exists and compiles/runs, but its output is wrong: the gfx12 F16 WMMA fragment is
not a contiguous shared-memory read but "two runs of four" per lane (`k = 4*hi+{0..3} ∪ 4*hi+8+{0..3}`)
with a J-major C tile (`m = 8*hi+e`, `n = lane%16`).  Two layout attempts failed.  **Do not extend
this file further** — the in-tree `PLAN.md` Phase 1 reuses `mma.cuh`'s `load_ldmatrix`/`mma` and
avoids the trap.  If this microbench is ever revived, copy `mmb.cu`'s `mmb_ld_frag`/`MMB_ACC_M`
verbatim and validate a 16×16×16 product first.  Full write-up: `../RESULTS-2026-09-26.md` §2.

- [x] kernel written (naive)
- [ ] CPU reference check — **fails**
- [ ] TFLOPS measured and recorded
