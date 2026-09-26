# PLAN — MMQ software pipeline (RDNA4)

Follow top to bottom.  Every item is a checkbox; **do not skip a gate**.  New measurements go in a
dated `RESULTS-YYYY-MM-DD.md` next to this file (never edit `MEASUREMENTS.md`).

Work in `~/llama.cpp` (fork checkout; its `rdna-boosts` branch is disposable, never push it).  Every
in-tree change is **env-gated OFF by default** until Phase 4 passes.  The delivery repo only carries
this `wip/` tree.

---

## Design

Turn the serial `load → sync → compute → sync` k-loop into a **two-stage pipeline**: while the
WMMA consumes tile *k*, the global→sram load for tile *k+1* is in flight.  Keep the `vec_dot` call
sequence and the `sum[]` accumulation order **exactly** as they are (bit-identity).

### Candidate designs (Phase 1 picks one)

| # | design | LDS | changes geometry? | risk |
|---|---|---|---|---|
| 1 | **register-staged prefetch**: load tile *k+1* global→regs during `vec_dot`, store regs→sram after the barrier | none extra | no | VGPR spills (`sum[]` is already 64 f32/thread at `I=128,J=128`) |
| 2 | **LDS double-buffer `tile_x` at a smaller tile** (`I=64`, or `MMQ_ITER_K=128`) | 2× `tile_x` | **yes** | base-rate change; per-type re-tune; routed-compact/fused-gate variants |
| 3 | **half-tile `y` double-buffer** (pipeline only the activation, collapse to one `vec_dot`) | + 1 `tile_y`-half | no, but changes `vec_dot` grouping | breaks bit-identity unless carefully ordered |
| 4 | **`global_load_lds` direct-to-LDS** two-stage | + stage buffer | no | ROCm codegen support/quality on gfx12 |

**Phase 1 chose (2).**  Design (1) is ruled out: `mul_mat_q<Q8_0,128>` already uses 214 VGPRs and a
register-staged prefetch adds ~35 → spill.  `I=64` costs no base rate and its LDS budget allows
`2×tile_x + 2×tile_y = 53.2 KiB`.  `mmb`'s `DBUF` is a working reference for LDS double-buffering
(`mmb.cu` ~line 850).

### Interface

* New env gate: `GGML_CUDA_MMQ_PIPELINE=0` disables (default on only after Phase 4).
* The pipeline must apply to `mul_mat_q`; decide explicitly what to do with
  `mul_mat_q_routed_compact` and the `has_gate` fused path (either pipeline them too or leave them on
  the old path — the latter is fine, but document it).

---

## Phase 0 — Setup  ✅ (2026-09-26)

- [x] branch `archive/work/mmq-pipeline`
- [x] `archive/work/mmq-pipeline/` (README/PLAN/MEASUREMENTS/HANDOVER/tools)
- [x] baseline frozen (`MEASUREMENTS.md`), LDS budget measured

## Phase 1 — Quantify the headroom and pick a geometry  **(go/no-go)**  ✅

- [ ] `rocprofv3` the isolated `test-backend-ops perf` Q8_0 ffn case; confirm the
      `load_tiles`/`vec_dot` split and the barrier stalls.  *(deferred — the arithmetic in
      `RESULTS-2026-09-26.md` §4 already shows 62 % of the time unaccounted for; a stall-counter
      confirmation is a refinement, not a blocker.)*
- [x] MMQ kernel VGPR/spill usage: `mul_mat_q<Q8_0,128,false>` = **214 VGPRs** (max 256) → design (1)
      would spill; design (2) needs no extra registers.
- [x] base rate at `I=64/J=64`: Q8_0 83.47 vs 83.66, Q6_K 57.17 vs 56.97 → **free**.
- [x] **Go/no-go: GO, design (2)** (LDS double-buffer at `I=64`; `2×tile_x + 2×tile_y = 53.2 KiB`
      fits).  Recorded in `RESULTS-2026-09-26.md`.

## Phase 2 — Implement the pipeline (env-gated)  ❌ DONE, REJECTED

- [x] add the prefetch path for `tile_x`/`tile_y` behind `GGML_CUDA_MMQ_PIPELINE`
- [x] preserve the `load_tiles` → `vec_dot(k00=0)` → `vec_dot(k00=MMQ_TILE_NE_K)` call sequence per
      tile; only move the *loads* earlier
- [x] gate off the fallback / `stream_k` / `has_gate` paths; routed-compact untouched
- [x] build clean (`-j16`)
- [x] **result: −15-18 % at equal geometry, and the default path regressed ~3.5 %** (reverted)

## Phase 3 — Correctness  ✅ (numerically correct; perf disqualifies it)

- [x] `test-backend-ops test -b ROCm0 -o MUL_MAT` → **1299/1299 OK, 0 FAILED**, gate on and off
- [ ] same-seed text / `W=1..8` purity — *not run; the perf is disqualifying so the campaign stops
      here.*

## Phase 4 — Performance  ❌ FAILED (all gates missed)

- [x] isolated ffn: Q8_0 **53.9** (needed ≥ 100), Q6_K **33.9** — both **worse** than the default
- [x] whole-model gates not reached

## Phase 5 — Decision

- [x] **Reject design (2); record the negative result** (`RESULTS-2026-09-26-phase2.md`), revert the
      fork, leave `main` untouched.
- [x] Recommendation recorded: the lever is MMQ *overhead* (epilogue + `ldmatrix`), not barriers;
      that points back at `../per16-f16-mma/`.  Get a working gfx1201 profiler first (rocprofv3 PMC
      returns 0 here).
- [ ] move the tree to `archive/work/` when the maintainer closes the campaign.

---

## Risks

1. **LDS is the wall.**  64 KiB/block and the current geometry already uses ~53 KiB; the pipeline may
   force a geometry change that costs more than the overlap buys.  Phase 1 must measure this first.
2. **Register pressure.**  The `sum[]` accumulator (`J*I/(nwarps*32)` f32) is already large;
   design (1) adds a prefetch buffer and may spill, which can be worse than the barrier.
3. **Bit-identity is easy to break** by reordering the `vec_dot` calls or changing the `k00`
   grouping.  Never merge the two `y` halves into one `vec_dot` without re-running the purity gates.
4. **Shared with other paths.**  `mul_mat_q` is also used by `mul_mat_q_routed_compact` and the
   fused gate+up+GLU MMQ; a broken pipeline there is a correctness bug, not just a perf one.
5. **The real limiter may be elsewhere.**  If Phase 1 shows the loads are already hidden (e.g. by the
   warp scheduler) or that the kernel is DRAM-bound, the barrier is not the problem and the campaign
   should pivot (the per-16 idea would then be the remaining lever).

## Reference commands

```bash
archive/work/mmq-pipeline/tools/repro.sh counts      # full per-type TFLOPS landscape
archive/work/mmq-pipeline/tools/repro.sh gemm q8_0   # isolated ffn GEMM
archive/work/mmq-pipeline/tools/repro.sh model       # whole-model prefill (1 GPU)
archive/work/mmq-pipeline/tools/ab.sh q8_0           # gate-on/off A/B at the ffn shape
```
