# Packed-QSA — P3 implementation notes

**Status:** P3 part 2 done (2026-09-13) — the packed WMMA kernel is correct and ~1.35x faster than
the VEC kernel at the op level.  Fork `packed-qsa` `4f464941a` (on `91f5e41a0` = part 1).
Design: `P3-DESIGN.md`.  P1/P2: `P1-NOTES.md`, `P2-NOTES.md`.

## What landed

| file | what |
|---|---|
| `qsa-packed.cu/.cuh` | gfx12 f16 WMMA primitive + layout self-test; `qsa3_attn_kernel` + `ggml_cuda_qsa3_attn` launcher; env-gated score/P dump |
| `fattn-qsa.cu` | dispatch: build the P2 descriptor, run the packed path when `dst->src[7]` is set (`GGML_CUDA_QSA_PACKED_ATTN=0` forces VEC), `GGML_CUDA_QSA_ATTN_CHECK=1` A/B + op timing |

Kernel structure: `grid=(ngroups, n_kv_heads)`, 256 threads = 8 waves; wave `w` owns head-dim tiles
`2w, 2w+1` (KQ partials) and V-dim tiles `2w, 2w+1` (output), keeping `O[3][2]` in registers.
Waves 0..2 own the softmax for row-tiles 0..2 and publish `alpha`/`l`/`P` via LDS.  A 16-key chunk
is 4 union blocks, so 16 keys per iteration.  The P fragment is staged through an LDS transpose
instead of the reference's gfx11 shuffle network.

## Correctness (gfx1201, 3x R9700)

`LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_PACKED=1 GGML_CUDA_QSA_ATTN_CHECK=1`, IQ3_XXS, f16 KV —
the packed kernel and the VEC kernel run over the **same** inputs and are compared elementwise:

```
PACKED-QSA attn check: n=16932864 max_abs=0.000616 rel=2.98e-05 mean_packed=0.2663 mean_vec=0.2663
```

i.e. agreement to f16 fragment rounding; no structural error.  `PACKED-QSA WMMA self-test: OK`
(the gfx12 fragment layout).  `test-backend-ops FLASH_ATTN_QSA` **22/22** on the default path.

Sampled output is coherent and text-level close to the VEC build (it is a prefill re-baseline, so
the greedy stream diverges, as expected).

## Two bugs found on the way (both worth remembering)

1. **`#if defined(RDNA4)` around the host launcher compiles it out.**  `RDNA4` comes from
   `vendors/hip.h` and is only set in the **device** pass; in the host pass the whole launcher body
   (including the `<<<>>>` launch) is removed, so the kernel never ran and the A/B compared
   uninitialised memory (rel≈1.0).  The fix is to guard only the **intrinsic** (`qsa_mma_f16` gets a
   `NO_DEVICE_CODE` fallback) and leave the kernels and the host launcher unguarded.  The earlier
   "WMMA self-test OK" was vacuous for the same reason — it returned 0 without running.
2. **The gfx12 D layout holds 8 rows per lane** (`m = 8*(lane>>4)+e`, `n = lane%16`), so the softmax
   statistics are **per-row** and must be reduced over the 16 key-lanes `r` (8 separate shuffle
   reductions of 4 steps).  A whole-array reduction (the natural port of the gfx11 code, where a
   lane holds one row) sums 8 rows at once and silently shrinks the output to ~1/6 magnitude with a
   flat-looking result.

## Performance (gfx1201, IQ3_XXS, ns=2051)

Op-level, `GGML_CUDA_QSA_ATTN_CHECK=1` timing (`x20`, same inputs):

| shape | packed | VEC | speedup |
|---|---:|---:|---:|
| n_q=1656, ns=1792 | 8.23 ms | 11.19 ms | **1.36x** |
| n_q=2756, ns=2051 | 16.47 ms | 21.99 ms | **1.34x** |

End-to-end `llama-bench -p 4096` (the same model, `-sm layer`, f16 KV): 893 t/s VEC vs 895 t/s
packed — **flat within noise**.  On this box the QSA op is only ~5 % of a prefill pass, so a 1.35x
op win is ~1.5 % end-to-end, below the ±2 % bench noise.  (The archived gfx1151 attribution put QSA
at 14 % of the pass; the delivery's VEC kernel is relatively faster there, and the box is slower
overall, so the fraction differs.)

The plan's op target was `>= 2x`; 1.35x is short of it.  The kernel is correct and lands the
architecture, but the remaining headroom is real: the per-row softmax costs 8 separate shuffle
reductions, the P transpose goes through LDS, there is no K/V prefetch, and only 16 keys are
processed per iteration.  Treat those as the P3.5 optimisation list, not as blockers.

## What remains

- **P3.5 (optimisation, optional):** fewer softmax shuffles (shuffle the 8-row vector once per
  step), shuffle-based P transpose, K/V prefetch, wider key chunks.  Re-run the A/B + timing.
- **P4:** support predicate / dispatch fallback; `-sm tensor` pack layout (currently asserts the
  pack is mirrored); RDNA3.5 (gfx1151) needs a 16-half fragment instantiation.
- **P5:** PPL vs the VEC build, `W=1..8` logits matrix (packed is prefill-only, so it should be
  unchanged), MTP acceptance, and the end-to-end perf A/B at pp8192/16384.

## One more note on the A/B harness

`GGML_CUDA_QSA_ATTN_CHECK=1` is the reference-grade gate for any future kernel change: it runs both
kernels on identical inputs and prints max-abs/rel plus the op timing, once per process.  Keep it.

`GGML_CUDA_QSA_ATTN_DBG=1` dumps the first group/tile/chunk's post-mask score tile, P tile and `l`
values — that is what localised the softmax bug.
