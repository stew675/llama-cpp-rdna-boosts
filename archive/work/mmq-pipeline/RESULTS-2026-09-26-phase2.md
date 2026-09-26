# RESULTS 2026-09-26 (phase 2/3) — software pipeline **rejected** (negative result)

**Headline: design (2) loses on both axes and is rejected.**  Two independent effects, each large
enough on its own to kill the campaign:

1. Phase 1's "**`I=64` costs no base rate**" was based on `GGML_CUDA_MMQ_J_MAX`, which the RDNA4
   J-selection switch **does not consult** — it was a dead knob (the earlier 83.47-vs-83.66 "free"
   reading was run-to-run noise).  Measured properly, `I=64/J=64` is **~21 % slower**.
2. Even at the **same** geometry, the pipelined loop (2 barriers/tile, double-buffered) is **15-18 %
   slower** than the original 4-barrier loop.  The extra LDS halves occupancy, and the in-order
   `load_tiles` (global→reg→smem) chain means the prefetched loads do **not** overlap the `vec_dot`.

Fork reverted; the attempt is preserved as `kernel/mmq-pipeline-attempt1.patch` (246 lines).  No
delivery change.

---

## 1. The `J_MAX` trap (corrects Phase 1)

`ggml_cuda_mmq_get_J_max()` (which reads `GGML_CUDA_MMQ_J_MAX`) is not called on the RDNA path: the
RDNA4 switch does its own `for (J = 8; J <= 128; J += 8)` tile-count loop (`mmq.cuh` ~1915).  So
Phase-1's `I=64/J=64` numbers were really the default `I=128/J=128` config twice.  A real override
(`GGML_CUDA_MMQ_FORCE_J`, added for this experiment) gives the truth, isolated ffn shape
`m=17408,n=512,k=5120`:

| type | `I=128/J=128` (default) | `I=64/J=64` | delta |
|---|---:|---:|---:|
| Q8_0 | **83.7** TFLOPS | 63.7 | **−24 %** |
| Q6_K | **57.1** TFLOPS | 44.1 | **−23 %** |

## 2. The pipeline loses at equal geometry too

Same `I=64/J=64`, pipeline off (4 barriers, single buffers) vs on (2 barriers, double-buffered
`tile_x` + `tile_y2`):

| type | pipeline OFF | pipeline ON | delta |
|---|---:|---:|---:|
| Q8_0 | 63.7 TFLOPS | 53.9 | **−15 %** |
| Q6_K | 44.1 TFLOPS | 33.9 | **−23 %** |

LDS at `I=64/J=64`: 28 928 B single-buffered vs **57 600 B** pipelined (measured via the
`GGML_CUDA_MMQ_PIPELINE_DEBUG` print), and the default `I=128/J=128` kernel already sits at 53 248 B
with 216 VGPRs — so every extra buffer is bought out of occupancy.

## 3. Correctness is fine — it is purely a perf loss

The restructured loop is **numerically correct** (so the barrier restructure itself is safe):

* `test-backend-ops test -o MUL_MAT` → **1299/1299 OK, 0 FAILED**, gate on and gate off.
* The `vec_dot(k00=0)` / `vec_dot(k00=MMQ_TILE_NE_K)` sequence and the `sum[]` accumulation order are
  preserved (the y halves are loaded into two buffers and passed to the same two calls).

## 4. Side effect on the default path

Because the pipeline branch is `if constexpr`-compiled into every `mul_mat_q` instantiation, the
*default* path regressed **~3.5 %** even with the gate off (Q8_0 80.85 vs 83.68).  Reverting the tree
restored 83.68 / 57.13.  Any future attempt must keep the default path's codegen byte-identical.

## 5. Why the premise was wrong (bottleneck re-analysis)

The kernel is **not global-load/barrier-bound**.  The int8 MMQ spends most of its issue slots on the
feed and drain of the WMMA, not on moving weights:

* the per-output-element **scalar epilogue** (`sum[...] += C.x[l]*dA*dB`) does one FMA plus one LDS
  load of `x_df` per accumulator element, per 16×16×16 MMA (≈256 VALU + ≈256 LDS per 4096 MAC);
* every A/B operand must be **`ldmatrix`'d** out of LDS first.

Those depend on the *compute*, not on the global load, so moving the global load earlier cannot hide
them — it only costs occupancy.  The "174 T-MAC/s synthetic issue ceiling" (operand-free loops) is
not reachable for a quantized-MMQ structure; the measured 31-34 % of it is an overhead limit, not a
latency limit.  `rocprofv3` counter collection returns 0 on gfx1201, so the VALU/LDS split is
analytic, not measured.

## 6. Recommendation

* **Do not pursue MMQ software pipelining** on RDNA4.  The LDS budget cannot hide a second tile
  without dropping occupancy, and the in-order `load_tiles` provides no overlap even when it fits.
* If the goal is still "make prefill faster", the lever is removing MMQ **overhead**, not the
  barriers: fewer/wider MMAs with no int8 epilogue is exactly the `per16-f16-mma` idea
  (`archive/work/per16-f16-mma/`), which this result strengthens rather than weakens — the int8 MMQ's problem
  is its per-element epilogue and fragment traffic.
* Before any further kernel engineering, get a **usable profiler** for gfx1201 (rocprofv3 counters
  are empty here); an unmeasured stall attribution is how this campaign went wrong.

## Artifacts

* `kernel/mmq-pipeline-attempt1.patch` — the full attempt (double-buffered `tile_x`, `tile_y2`,
  2-barrier loop, `GGML_CUDA_MMQ_PIPELINE` gate, the `FORCE_J` debug override and the `DEBUG` print).
  Applies to `mmq.cuh` at `527d39401`.  Not for promotion.
