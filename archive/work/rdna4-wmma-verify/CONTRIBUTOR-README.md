# rdna4-wmma-verify: the 2..16-column speculative verify in WMMA iu8 (RDNA4, opt-in)

**Status: candidate (contributor patch), not part of the delivery.  Off by default** (`GGML_CUDA_WMMA_MMVQ=2`
enables it).  One patch, cut against the release tree `v16-4f9dfa0e3-r289` (`release.json`: base `4f9dfa0e3`,
tree `7181d396977e90270528d8478ee7b04894a3f39e`); strict `git am` on that tree.

| file | sha256 |
|---|---|
| `0001-mmvq-speculative-verify-2-16-columns-in-WMMA-iu8-on-.patch` | `982dfb3b5f2d2f31278651a3358f3867fdd068e7e324e8d207498c058ba5a664` |

## Why

With a deep draft (DFlash2 at 7 drafts, a deep MTP) the verify matvecs dominate the cycle.  On a Radeon AI PRO
R9700 with Qwen3.8-27B UD-Q4_K_M, the target's matvecs cost, per pass:

| verified tokens | 1 | 4 | 7 | 8 |
|---|---|---|---|---|
| ms | 25.9 | 28.4 | 32.4 | 34.3 |

At 1 token the ksplit kernel runs at ~93 % of the read bandwidth; at 7-8 tokens it becomes instruction-bound,
because dequantization and the int8 dot products are redone for every column.  An ATT trace of the Q5_K
17408×5120 ksplit kernel shows 92.7 % of its cycles waiting on memory at 1 column, 10.8 % at 7.

## What it does

A new translation unit (`mmvq-wmma.cu/.cuh`, ~330 lines) and a 7-line early return in `ggml_cuda_mul_mat_vec_q`:
for a dense matrix without fusion, 2..16 columns, `ne01 % 16 == 0`, `ne00 % 256 == 0`, on RDNA4, the product is
computed with `wmma_i32_16x16x16_iu8`.  Every other case falls through to the existing path.

- C^T = X^T · W^T: the q8_1 activations are the signed A operand (16 columns × 16 K), the weights the B operand
  (16 K × 16 rows), so each lane keeps its weight row and its block scales stay local.  Each weight is
  dequantized **once for all the columns**: the cost becomes nearly flat in the column count.
- **Q4_K, Q5_K, Q6_K, IQ4_XS** in their original GGUF layout: the order of the K positions inside a 32-block is
  free as long as both operands use it, so a lane reads 16 contiguous bytes of `qs` (Q4_K/Q5_K) or the native
  8-byte runs (Q6_K, IQ4_XS).  Q6_K's 210-byte blocks are read with unaligned 8-byte loads.
- Activations: the existing `block_q8_1` cache (no new quantize pass); each lane loads one column scale and a DPP
  `row_share` broadcast hands the 8 column scales it accumulates.
- The Q4_K/Q5_K min term uses d·Σxq (the sum of the **quantized** activations, one extra WMMA against a ones
  matrix), exactly as `vec_dot_q4_K_q8_1` does.  Using `ds.y` (the float Σx) instead moves the outputs by ~1 %
  and drops the DFlash2 acceptance by ~6 points; `test-backend-ops` does not catch that.
- One workgroup per 16-row tile, 10/8/4/2/1 waves (the largest that divides the super-block count) splitting K,
  next super-block prefetched, LDS reduction, direct store.

## Numerics

- `test-backend-ops -o MUL_MAT` (with `GGML_CUDA_WMMA_MMVQ=2`): no failure.
- A run-time cross-check against the ksplit kernel on every call of a real DFlash2 generation: max relative
  difference ≈ 2e-7 on every shape and type.
- KLD against the 1-token decode (llama-perplexity, code text, 4 × 512): 0.0023 at ub 8 — the same as the
  delivery's own mmq path at ub 16.
- **Not bit-identical to the single-token path** (a different accumulation order).  With it enabled the
  decode/verify greedy-purity guarantee (`GREEDY-PURITY.md`) does not hold: greedy DFlash2 text matched the
  non-speculative text in one test and diverged at a near-tie (line 32 of 256 tokens) in another.  That is why it
  is offered off by default.

## Measured

Radeon AI PRO R9700 (gfx1201, ROCm 10.1), Qwen3.8-27B UD-Q4_K_M, 256 tokens, temperature 1, `-c 8192`, KV f16:

DFlash2 `--spec-draft-n-max 7`, 30 seeds, delivery of commit `81e969e` (2026-11-03), toggled by the env var:

| | ms per cycle | tokens per cycle | tok/s |
|---|---|---|---|
| code, off | 47.95 | 5.80 | 121.1 |
| code, on | 43.53 | 5.72 | 131.4 |
| prose, off | 47.89 | 2.43 | 50.7 |
| prose, on | 43.30 | 2.34 | 54.1 |

On r289 (10 seeds, together with the `gdn-conv-norm-deferred-gather` fix): code 124.0 → 130.7 tok/s, prose
51.0 → 55.6, cycle 47.4 → 42.6 ms.  With the flat verify cost, 7 drafts beat 6 (code 131 vs 125).

Standalone kernel, 8 columns, rotating weight copies (> 512 MB, outside the Infinity Cache), GB/s:
Q4_K 17408×5120 549, Q5_K 17408×5120 559 / 5120×17408 583, IQ4_XS 17408×5120 554, Q6_K 248320×5120 608 —
against a pure-read ceiling of ~610-620 GB/s for a single launch of that size.

## Not covered

- Q8_0 and the other types (tried for Q8_0: correct, but no gain — its ksplit path is already at the memory
  ceiling at 4 columns), `MUL_MAT_ID`, fused GLU / bias calls, 1 column.
- Only gfx1201 was available; gfx1200 shares the ISA but was not run.
