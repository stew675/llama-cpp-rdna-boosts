# HANDOVER — MMQ software pipeline (RDNA4)

**Branch:** `archive/work/mmq-pipeline` (delivery repo `~/llama-cpp-rdna-boosts`).  **Fork scratch:**
`~/llama.cpp` @ `527d39401`, build `build-rocm`, ROCm 7.14 gfx1201.

**Read order:** this file → `README.md` (why) → `MEASUREMENTS.md` → `PLAN.md` (the checklist).

## State at handover (2026-09-26) — campaign CLOSED, negative result

* **Design (2) was implemented, gated, measured and rejected** (`RESULTS-2026-09-26-phase2.md`).
  Fork tree reverted and clean; `build-rocm` back at the delivery tip (Q8_0 ffn 83.7, Q6_K 57.1
  TFLOPS); the attempt is kept as `kernel/mmq-pipeline-attempt1.patch` (246 lines, not for promotion).
* Two independent losses: `I=64/J=64` is ~21 % slower (Phase 1's "free" reading used a dead env
  knob), and the 2-barrier double-buffered loop is another 15-18 % slower (LDS → occupancy, and the
  in-order `load_tiles` gives no overlap).  The kernel is epilogue/`ldmatrix`-bound, not barrier-bound.
* This branch supersedes `archive/work/per16-f16-mma` (parked — see its `README.md`/`RESULTS-2026-09-26.md`),
  and the result actually points *back* at it: the fix is removing MMQ overhead, not the barriers.

## The one-paragraph summary

The MMQ prefill kernel's k-loop is `load_tiles → __syncthreads → vec_dot → __syncthreads` with no
cross-tile overlap, and it measures only **31-34 %** of the gfx1201 int8-WMMA issue ceiling
(174 T-MAC/s).  The goal is to pipeline each tile's global→sram load behind the previous tile's
WMMA, which should lift **every** quant type (Q6_K 57→~70+, Q8_0 83→100+, whole-model pp2048
2128→2400+) while staying **bit-identical**.  The binding constraint is that the current geometry
already uses ~53 KiB of the 64 KiB LDS/block, so the design may need a register-staged prefetch or a
smaller tile (see `PLAN.md` candidates).

## Next actions

Nothing to continue here.  If the maintainer wants to revive prefill work:

1. Get a **working profiler for gfx1201** first — `rocprofv3 --pmc` returns 0 here, so the
   VALU/LDS split in `RESULTS-2026-09-26-phase2.md` §5 is analytic.  Do not start another kernel
   campaign without measured stall attribution.
2. The promising direction is reducing MMQ *overhead* (per-element int8 epilogue + `ldmatrix`),
   i.e. the `../per16-f16-mma/` route, not the barrier/load schedule.
3. Keep any future experiment's **default-path codegen byte-identical** — the always-compiled
   `if constexpr` branch here cost ~3.5 % even with the gate off.

## Traps

* **Bit-identity**: never merge the two `y`-half `vec_dot` calls or change the `k00` grouping without
  re-running the same-seed and `W=1..8` gates.
* **LDS is full**: 64 KiB/block, ~53 KiB used.  A naive `2×tile_x` does not fit at `I=128`.
* `mul_mat_q` is shared with `mul_mat_q_routed_compact` (MoE) and the fused gate+up+GLU MMQ; a
  pipeline that breaks those is a correctness bug.
* The pre-existing `MUL_MAT_ID(type_a=q6_K,m=64,n=16,k=768)` FAIL is not ours.
* `build-rocm` (not `build-rocm-hybrid`); incremental `cmake --build … -j16` is ~50 s when only the
  mmq files change.

## Useful commands

```bash
archive/work/mmq-pipeline/tools/repro.sh counts
archive/work/mmq-pipeline/tools/repro.sh gemm q8_0
archive/work/mmq-pipeline/tools/ab.sh q8_0
cd ~/llama.cpp && cmake --build build-rocm --target test-backend-ops llama-bench -j 16
```

## Next-session prompt

> The `archive/work/mmq-pipeline` campaign in `~/llama-cpp-rdna-boosts` is **closed with a negative result**
> (`archive/work/mmq-pipeline/RESULTS-2026-09-26-phase2.md`): the MMQ software pipeline is slower, the `I=64`
> geometry is ~21 % slower, and the reason is that the kernel is per-element-epilogue/`ldmatrix`
> bound rather than global-load/barrier bound.  The fork is reverted.  Do not restart this campaign.
> If asked to improve prefill, read `archive/work/per16-f16-mma/` (the overhead-removal route) and first
> establish a working gfx1201 profiler, because rocprofv3 PMC returns 0 on this box.
