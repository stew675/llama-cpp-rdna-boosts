# Per-16 superblock quants on RDNA4: fold the sub-scales into an F16 WMMA tile

**Status: PARKED (2026-09-26) in favour of `archive/work/mmq-pipeline`.**  The maintainer chose the MMQ
software-pipelining lever, which is safer (bit-identical) and bigger (helps every quant type).  The
findings here — the gfx12 fragment trap, the hipBLAS-not-tile-kernel correction, and the `mmb`
precedent — are recorded in `RESULTS-2026-09-26.md` and should be read before anyone revives this.

**NOT delivery work** —
nothing here is in `patches/`, `apply-all.sh` does not touch it, and the `~/llama.cpp` tree is only
used as a scratch build for measurements.  Nothing gets promoted to `main` without the maintainer.

**Scope:** every prompt-processing (prefill) quant whose scale block is **16 values**: `Q6_K`,
`Q3_K`, `IQ2_XS`, `IQ2_S`.  **Q6_K is the proof of concept**; the other three follow only if Q6_K
clears the gates in `PLAN.md`.

> **Decision (2026-09-26):** parked.  bf16 is not a workaround for the F16 findings (same WMMA
> shape/rate/fragment; the delivery's `mmb` is already the bf16 version of this idea and loses;
> bf16 is lower precision), so the campaign was superseded by `archive/work/mmq-pipeline`.  Revisit only if
> the pipeline work stalls and a numerics-changing prefill kernel becomes acceptable.

**Companion docs**

| doc | what |
|---|---|
| `MEASUREMENTS.md` | the full 2026-09-26 baseline evidence (model + isolated-GEMM numbers) |
| `PLAN.md` | the implementation plan checklist — **this is the thing to follow** |
| `HANDOVER.md` | cold-start brief (read this first next session) |
| `tools/` | repro scripts (`repro-gemm-perf.sh`, `sweep-j.sh`, `counts.sh`) |
| `kernel/` | the standalone spike kernel (Phase 1), built before touching llama.cpp |

**Related existing work (do not duplicate)**

* `archive/work/q8-prefill-tuning/README.md` — the measured context that makes this campaign worth doing:
  the MMQ kernel reaches only **31-34 %** of the gfx1201 int8-WMMA issue ceiling
  (`v_wmma_i32_16x16x16_iu8` = 174 T-MAC/s per GPU), its k-loop has **no double-buffering**, and
  "the k-quants pay for their superblock dequant".  A software-pipelined MMQ is the *general* fix
  for every type; **this campaign attacks the per-16-specific deficit**, which is separate and
  additive.
* `archive/work/prefill-arrangements/README.md` — the "transform once, consume with a WMMA kernel" family
  (`mmb` bf16 shadow, packed QSA, …).  This campaign is a fifth instance of that same move: the
  transform is **Q6_K → F16 in shared memory** and it is fused, so it costs no extra DRAM.

---

## 1. The problem, in one table

`llama-bench`, single R9700 (gfx1201), `-ngl 99 -n 0` (prefill only), `pp32768`:

| model | file | pp512 | pp32768 | 2-GPU `-sm tensor` pp8192 |
|---|---:|---:|---:|---:|
| **Q6_K** | 21.30 GiB | 1035 | **897** | **1466** |
| UD-Q4_K_XL | 16.34 GiB | — | 1129 | 1804 |
| Q8_0 | 27.04 GiB | 1444 | 1207 | 1955 |

Q6_K is **74 % of Q8_0** and **79 % of Q4_K_XL**, *already at pp512* — so this is a weight-GEMM
cost, not a context/attention cost, and the identical 2-GPU ratio proves it is not the all-reduce.
Q8_0 moves **27 % more weight bytes** than Q6_K and is still 35 % faster.

## 2. Root cause (measured, `MEASUREMENTS.md`)

Isolated prompt GEMM, 27B ffn shape `m=17408,n=512,k=5120` (`test-backend-ops perf`):

| type | TFLOPS | scale block | MMQ inner kernel |
|---|---:|---|---|
| F16 | 107 | — | (hipBLAS / `mmf`) |
| Q8_0 | 83 | 32 | `q8_0_q8_1_mma` |
| Q4_K | 75 | 32 | `q8_1_q8_1_mma` |
| **Q6_K** | **57** | **16** | `q6_K_q8_1_mma` |

Across all types at `m=4096,n=512,k=14336`, **every per-16 type is at the bottom and every per-32
type at the top**:

* per-32: Q8_0 84, IQ4_XS 85, IQ4_NL 84, IQ3_XXS 78, Q4_K 75, Q5_K 71
* **per-16: Q6_K 55, Q3_K 57, IQ2_XS 58, IQ2_S 58**

**Why.**  A per-16 scale forces the int8 WMMA to use **16-K fragments** (`tile<16,4,int>`): per 32 K
the kernel issues 2 A-loads + 2 B-loads + 2 `mma` and runs a **doubled scalar epilogue**
(2× int→float + 4 FMA per output element), where the per-32 kernels issue one A/B load pair and
1 int→float + 2-3 FMA.  On gfx1201 the int8 WMMA is fast enough that this doubled issue/epilogue is
the limiter.  The RDNA4 fragment layout interleaves K across the two warp halves, so the per-16 and
per-32 kernels **cannot** be merged without misassigning scales (verified by attempted
restructuring — slower *and* wrong).

**This is upstream code, not a delivery regression.**  The delivery's block 04 already optimises it
by +47 % (its register hoist lifts the ffn GEMM 38.9 → 57.1 TFLOPS); reverting that one function
reproduces the slower fork-point kernel.  Nothing in `patches/` made Q6_K worse.

**Why not `mmb`/bf16 (the delivery's existing transform).**  Forcing `GGML_CUDA_MMB_TYPES=q6_k`
gives **639 t/s** (RDNA4 256×128 tile) / **733 t/s** (gfx11 split tile) vs MMQ **981 t/s**: the
on-the-fly Q6_K→bf16 dequant (`mmb_dq_row_q6k`) is scalar and dequant-bound.  That matches the
delivery's documented "Q6_K loses on RDNA4" type policy.

## 3. The hypothesis

> Fold the per-16 sub-scale (and the per-256 super-scale) into an **F16 weight tile in shared
> memory**, dequantize the activation to F16 too, and run the existing **F16 WMMA with f32
> accumulation**.  The per-16 scalar epilogue then disappears entirely, and the mma runs at the F16
> rate (90-107 TFLOPS isolated) instead of the per-16 int8 rate (55-57).

> **Corrected ceiling (2026-09-26, see `RESULTS-2026-09-26.md`).**  The 107 TFLOPS F16 figure is
> **rocBLAS**, not a tile kernel — `ggml_cuda_should_use_mmf()` returns false for prefill
> (`src1_ncols > 16`), so F16 prefill goes to hipBLAS.  There is no fast in-tree F16 prefill tile
> GEMM to inherit, so the realistic competitor for a fused kernel is the **int8 MMQ at 57**, and the
> delivery's own fused dequant tile kernel (`mmb`, bf16) currently **loses** (639-733 t/s vs 981).
> Expected landing zone is therefore **57 → maybe 65-75 TFLOPS**, not a near-doubling.  The gate is
> "correct **and** beats 57", with ≥ 70 as the stretch.

Expected landing zone: Q6_K **65-75 TFLOPS** → model prefill **~1100-1300 t/s** single-GPU (vs 981),
i.e. around Q4_K_XL.  The dequant is fused into the sram loader, so DRAM traffic is unchanged.

**What makes this different from `mmb`:** `mmb` dequantizes to **bf16 into registers/global** and
uses a bespoke kernel; this reuses the **MMQ tiling** (256-K super-block per tile, which is exactly
Q6_K's block size), keeps everything in sram, and uses the F16 `tile`/`mma` primitives already in
`mma.cuh`.

## 4. Success criteria (the gates)

| gate | threshold |
|---|---|
| Phase-1 spike, isolated `m=17408,n=512,k=5120` | correct **and** > **57 TFLOPS** (stretch 70) else stop |
| correctness, `test-backend-ops` MUL_MAT over Q6_K shapes | green within the fp16 tolerance |
| single-GPU Q6_K `pp8192` | **≥ 1100 t/s** (vs 981) |
| 2-GPU `-sm tensor` Q6_K `pp8192` | **≥ 1800 t/s** (vs 1466) |
| quality | same-seed text coherent; perplexity within noise of the MMQ path |
| purity | prefill-only gate; `W = 1..8` decode/verify untouched (mmvq path) |

> **Numerics warning (read before Phase 3).**  F16 accumulation is *not* bit-identical to the
> current int8-MMA + f32-epilogue path.  The delivery's "same-seed output IDENTICAL" gate is
> therefore **not satisfiable** by this change as-is; the acceptable bar for a prefill-only kernel
> is coherence + perplexity-within-noise, and the maintainer decides whether that is a permitted
> trade.  This is called out in `PLAN.md` Phase 3 and is the single biggest reason this may end up
> an archived experiment rather than a delivery block.

## 5. Rollout

1. **Q6_K PoC** — Phases 1-3.  Stop here if the spike or the quality gate fails.
2. **Q3_K** — same per-16 problem, inner kernel `q8_0_16_q8_1_mma`; expect the same win.
3. **IQ2_XS, IQ2_S** — same `q8_0_16` kernel family; lower priority (they are already rare).
4. **Promote/archive** — a validated win becomes a delivery-block amendment (maintainer go-ahead,
   env-gated, documented); a failure is archived under `archive/work/`.
