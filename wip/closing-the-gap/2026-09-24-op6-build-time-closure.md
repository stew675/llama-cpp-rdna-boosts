# OP-6 — `fattn-mma-f16` build-time: **CLOSED, no further change**

**Date:** 2026-09-24 · **Box:** soar (gfx1201, Ryzen 9 9950X3D, 16 cores, `-j16`).

**Verdict.**  The OP-6 premise ("7.26 MB / 229 s per instance TU") is the **pre-r6** state.  r6's
per-head split already fixed it: each MMA instance TU is now one `(ncols1, ncols2, head)` and the
largest is **2.9 MB / 81.8 s** to compile.  The clean `ggml-hip -j16` build is ~236 s, i.e. **3× the
largest single TU — the build is throughput-bound, not tail-bound**, so the remaining "one MMA TU per
KV type" split (which leaves the total instantiation work unchanged) **cannot reduce the makespan**.
Close OP-6.  ccache remains the answer for the edit/rebuild loop.

## Measurements

| | value |
|---|---|
| largest MMA instance TU (solo, ccache disabled) | `fattn-mma-f16-instance-ncols1_16-ncols2_2-dkq512-dv512.cu` → **81.8 s**, object **2.9 MB** |
| dkq512 MMA TU objects (6) | 2.8–2.9 MB each (was 7.26 MB pre-r6 for the 8-head file) |
| template-instance objects | 268 files, 236 MB total |
| clean `ggml-hip -j16` build (r6, reference) | ~236 s |
| → makespan vs largest TU | 236 s ≫ 82 s ⇒ **throughput-bound** |

Because the makespan ≈ total-work / 16 and the tail (82 s) is already a third of it, splitting each
per-head TU into per-type TUs (≈7× more TUs, each ≈7× smaller) leaves total work identical and only
shortens the tail — a change with no upside.

## What is already delivered (r5/r6)

* **tile**: one TU per `(head size, KV type)` (`DECL_FATTN_TILE_CASE_TYPE`), macros expand per type —
  clean `ggml-hip -j16` **538 → 330 s**; `fattn-tile.cu` < 10 s.
* **MMA**: one TU per `(ncols1, ncols2, head size)` with the head-512 instances listed first — the
  **reorder, not the split, is the win** (clean `ggml-hip -j16` **323.4 → 236.0 s**; the six dkq512
  TUs moved off the tail).
* **-Wpass-failed flood** (10 362 unroll warnings, all from `fattn-mma-f16.cuh`) suppressed in the HIP
  CMake flags (diagnostic-only).

## What was tried and rejected (r6)

* **runtime KV-type dispatch** (one loader, runtime switch): object 2.80 → 2.46 MB but compile
  **236 → 304 s** (worse — one giant CFG optimises more slowly than six specialised functions).
* **`__noinline__` native loader**: **236 → 136 s** but a universal **-1.5…-2.5 % prefill** and
  -0.3…-0.6 % decode (the force-inlined loaders are what makes native staging fast).
* **hybrid** (inline q8_0/q4_0, outline the rest): no perf recovery at 168 s.

## Follow-up

* Only worth revisiting if a *throughput* reduction is found (i.e. fewer instantiations, not more TUs):
  the upstream-worthy `#pragma unroll` cleanup and a runtime KV-type dispatch in the loader are the
  candidates, both filed under `upstream/`.  Neither is a campaign gap.
* `TODO.md` / `wip/build-time-regression/README.md` already carry the analysis; keep ccache enabled.
