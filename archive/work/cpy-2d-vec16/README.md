# cpy_2d_vec16 — a vectorised fallback for the 2-D copy the r183 guard refuses — PROMOTED in r189

**Origin:** the KV-residency campaign (`wip/kv-residency`), session A12 (2026-10-31), commit `f8e47e392` on the
fork branch `wip-kv-residency-b`.  Staged for promotion as `beta/0016-copy-2d-vec16/` (this directory's
`copy-2d-vec16.patch`, `README-beta.md`, `BETA-TESTING.md`), folded into **block 10** and released as
`v16-6050b467c-r189` (`ggml/src/ggml-cuda/cpy.cu`).

## Why it ships

r183 (block 10) gave `ggml_cuda_cpy_as_memcpy_2d` a conservative driver-refusal guard.  Its false positive sends
a *large* padded block copy to `cpy_scalar<cpy_1_scalar<float,float>>`, which does three integer divisions and
three modulos per element.  The delivery reaches that: with a **quantized indexer key cache** (`-ctk q8_0`,
which the fused `INDEXER_SCORE`/`INDEXER_FILL` path cannot read) qwen4exp takes its per-op chain, which makes one
refused 2-D copy per layer per prefill chunk (`w=512`, `sp=2048`, `dp=512`, `h` = the QSA index-block count).

The fix is a **regression fix, not a perf win on our arms**: it cuts the refused copies' kernel time
(1010 -> 637 ms of a 105k-token prefill on gfx1201) while the wall time is flat (prefill and decode both measure
unchanged — the copies are prefill-phase and ~0.3 % of the prefill's kernel time).  Its real value is the
campaign's sparse/pool decode arm, where the same copy is made per step (5.20 -> 1.83 ms/token there), plus not
leaving a 4x-slower fallback under a guard any padded destination can trip.

## What changed relative to the beta patch

The fold is the beta patch **plus one hardening**: "the guard refused" is now an explicit `guard_refused`
out-parameter instead of being inferred from `width != 0`.  `ggml_cuda_cpy_as_memcpy_2d` writes `width` before its
`spitch < width || dpitch < width` rejection too, and that refusal is a *different* case (self-overlapping rows,
which need the flat element mapping); inferring the refusal from `width != 0` would send such a geometry to a
row-per-row copy that reads and writes the wrong bytes.  `GREEDY-PURITY.md` §76.  The kernel, the alignment
conditions and **the guard itself** are exactly as submitted (r183's load fix is untouched, re-verified).

## Validation (gfx1201, 3x R9700 + 1x 7900 XTX where noted)

Independent of the campaign's numbers (which the beta README carries):

* **Reach** proven with a throw-away instrumented build: 48 refused copies per prefill chunk (`w=512`), all
  taking the vec16 path; the r183 `w=2` case correctly stays on `cpy_scalar`; **no decode-phase refused copy**
  on the resident arm or with `LLAMA_KV_RESIDENT_CELLS=2048`, i.e. the delivery's reach is prefill.
* **Copy cost** (105k prompt, `-n 8`, rocprofv3): `cpy_scalar<float,float>` 1433.75 ms / 12428 calls ->
  423.91 ms / 2732 calls, plus `cpy_2d_vec16` 636.77 ms / 9696 calls; whole run 329.16 s -> 329.28 s.
* **Wall time**: `pp16384` 339.3 both; `tg64 @ d49152` 26.7 both; deep-arm generation 30.9 vs 30.4 t/s.
* **Purity**: text byte-identical at 4k (`905 6b9e214d3b59`) and **105k depth** (`94 fe7775042f82`), on 3-GPU
  `-sm tensor -ncmoe 48` qwen4exp q8_0 (`611 15561552e5d1`) and dense 27B 3-GPU (`678 692c3f7ca639`);
  prefill-logit **0.000000 / 100.000 %** on qwen4exp q8_0 and **-0.000000 / 100.000 %** on the canonical dense
  27B; `test-backend-ops -o CPY` 251/251; no GPU faults.
* **The r183 load fix still holds**: Flash-Next GSQ-IQ3_XXS 3 GPUs `-sm tensor -ncmoe 48` loads and generates
  (that arm aborted at load on r182 and earlier).
* One measurement trap worth noting: a fresh-process text run **without** `ROCBLAS_USE_HIPBLASLT=0` read a
  different hash (`953 46bd85e08529`) — the per-start hipBLASLt solution pick of issue #67 (§44), not a change.

Numbers: `benchmarks/2026-10-31-cpy-2d-vec16.md`.  Variable catalogue: none (no new env var).  Release record:
`WORKLOG.md` 261031.007.
