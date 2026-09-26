# MMQ software pipeline on RDNA4: overlap the k-tile loads with the WMMA

**Status: CLOSED — NEGATIVE RESULT (2026-09-26), branch `archive/work/mmq-pipeline`.**  The software pipeline
was implemented, gated and measured, and it **loses** (see `RESULTS-2026-09-26-phase2.md`): the
`I=64` geometry is ~21 % slower (Phase 1's "free" reading used a dead env knob) and the 2-barrier
double-buffered loop is another 15-18 % slower.  The kernel is epilogue/LDS-traffic-bound, not
global-load/barrier-bound.  Fork reverted; the attempt is kept as `kernel/mmq-pipeline-attempt1.patch`.
**NOT delivery work** — nothing here is in `patches/`, `apply-all.sh` ignores it.  Nothing reaches
`main` without the maintainer.

**Scope:** the shared `mul_mat_q` (MMQ) prefill kernel — **every quant type**, not just the per-16
family.  This supersedes `archive/work/per16-f16-mma` (parked): the per-16→F16 idea was a numerics-changing
tile kernel whose ceiling turned out to be the int8 MMQ, not hipBLAS; this one is a pure scheduling
change and stays **bit-identical**.

**Companions**

| doc | what |
|---|---|
| `PLAN.md` | the phased checklist — **follow this** |
| `MEASUREMENTS.md` | the frozen baseline + LDS budget + the 3x headroom evidence |
| `HANDOVER.md` | cold-start brief and the next-session prompt |
| `tools/` | repro scripts and an A/B harness |
| `../q8-prefill-tuning/README.md` | the measurement that motivates this (MMQ at 31-34 % of the int8 ceiling; the AR analysis) |
| `../per16-f16-mma/RESULTS-2026-09-26.md` | why the F16 alternative was parked |

---

## 1. The problem

The MMQ kernel's k-loop is a strict serial chain with **no cross-tile overlap** (`mmq.cuh`,
`mul_mat_q`, ~line 895):

```
for (kb0 ...) {                 // one MMQ_ITER_K (256-value) tile per iteration
    load_tiles(x, tile_x, ...); // weights global -> sram
    { load y half 0 -> tile_y; }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);            // int8 WMMA
    __syncthreads();
    { load y half 1 -> tile_y; }
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
}
```

Nothing overlaps: the next tile's global→sram weight load starts only after the previous tile's last
`vec_dot` and barrier.  Measured consequence (`q8-prefill-tuning`): the MMQ runs at **31-34 %** of the
gfx1201 int8-WMMA issue ceiling (`v_wmma_i32_16x16x16_iu8` = **174 T-MAC/s** per GPU), and the
isolated ffn GEMMs sit at 16-24 % of it:

| type | isolated `m=17408,n=512,k=5120` | % of 174 T-MAC/s |
|---|---:|---:|
| Q8_0 | 83 TFLOPS = 41.5 T-MAC/s | 24 % |
| Q4_K | 75 TFLOPS | 22 % |
| Q6_K | 57 TFLOPS | 16 % |
| F16 (hipBLAS) | 93.7 TFLOPS | — (not the same units) |

Whole-model Q8_0 prefill (2 GPU `-sm tensor`, ub2048, pp2048) was profiled at **59.4 T-MAC/s = 34 %**
of the ceiling.  There is ~3x instruction-level headroom sitting behind the barrier chain.

## 2. Goal

Pipeline the k-loop so that the **next** tile's global→sram weight load (and activation load)
overlaps the **current** tile's `vec_dot` WMMA.  Keep the arithmetic (and therefore the output)
**bit-identical** — this is a scheduling change only.

Expected: **+30-60 %** MMQ-bound prefill across all quant types.

**Outcome (2026-09-26): the expectation was wrong.**  Neither lever materialised — see
`RESULTS-2026-09-26-phase2.md`.  The MMQ kernel is bound by its per-element scalar epilogue and the
`ldmatrix` fragment traffic feeding the WMMA, not by the global weight load or the barriers, so
moving the load earlier cannot help and the LDS a second tile needs is taken out of occupancy.
Phase 1's supporting claim that `I=64` is free was based on `GGML_CUDA_MMQ_J_MAX`, which the RDNA4
J-switch ignores — with a real override `I=64/J=64` is ~21 % slower.

## 3. The binding constraint: LDS is 64 KiB/block

Measured on gfx1201 (`hipDeviceProp_t`): `sharedMemPerBlockOptin = 65536`.  The current configs
already spend almost all of it:

| tile | Q8_0 `I=128, J=128` |
|---|---:|
| `tile_x` weights (`I * sram_stride * 4`, stride 70) | 35.8 KiB |
| `tile_y` activations (`J * 33 * 4`) | 16.9 KiB |
| `data_mul_mat_q` header (`+J`) | 0.5 KiB |
| **total** | **~53 KiB** (~11 KiB free) |

So a second `tile_x` (35.8 KiB) does **not** fit at this geometry, and neither does a second `tile_y`
(16.9 KiB).  Candidate designs (pick after Phase 1 measurements):

1. **Double-buffer `tile_x` at a smaller `I`.**  `I=64` gives `tile_x = 17.9 KiB`, so `2×tile_x +
   tile_y = 52.7 KiB` fits.  Costs rows-per-block (more blocks, more redundant `tile_y` loads) —
   measure the base rate at `I=64` first (`GGML_CUDA_MMQ_J_MAX=64` reaches it; it was flat for Q6_K,
   but re-measure every type).
2. **Register-staged prefetch.**  Stage the next tile's global data in registers during `vec_dot`,
   then copy to sram after the barrier.  No extra LDS, but the MMQ `sum[]` array is already large
   (`J*I/(nwarps*32)` = 64 floats/thread at `I=128,J=128`), so this risks spills — measure.
3. **Half-tile `y` double-buffering.**  Keep `tile_x` single and pipeline the two 32-K `y` halves
   (and collapse to one `vec_dot` per tile), overlapping only the activation load.
4. **`global_load_lds` (direct-to-LDS global loads).**  RDNA can issue `buffer_load ... lds` without
   a register round-trip; a two-stage pipeline built on that may need less LDS headroom.

The `mmb` kernel already double-buffers its A tile in LDS and records it as a win
(`mmb_dense_kernel`, `DBUF`), so the pattern is proven on this arch — it just has a different tile
budget.

## 4. Why this is worth doing (and low risk)

* **Bit-identical.**  Only the load/compute scheduling changes; each output element's accumulation
  order is untouched.  The delivery's purity gates apply unchanged.
* **Broad.**  Every quant type, every model.
* **The headroom is measured, not assumed** (31-34 % of a 174 T-MAC/s issue ceiling, with the
  load→sync→compute→sync shape visible in the source).
* **Complementary** to the per-16 idea (if that is ever revived, a pipelined MMQ is its floor).

## 5. Success criteria

| gate | threshold |
|---|---|
| Phase-1 microbench, isolated `m=17408,n=512,k=5120` | pipeline demonstrably overlaps (rocprof shows load+mma concurrent) |
| isolated ffn GEMM, all types | **≥ +20 %** TFLOPS (Q8_0 83 → ≥ 100) |
| whole-model Q8_0 `pp2048` 2-GPU | **≥ 2400 t/s** (vs 2128) |
| whole-model Q6_K `pp8192` 1-GPU | **≥ 1150 t/s** (vs 981) |
| purity | same-seed text **bit-identical** with the gate on/off; `W = 1..8` unchanged |
| no regression | MoE / fused-gate / routed-compact paths and every quant type |
