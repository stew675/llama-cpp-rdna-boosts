# HANDOVER — per-16 → F16 WMMA (Q6_K PoC)

**Branch:** `archive/work/per16-f16-mma` (delivery repo `~/llama-cpp-rdna-boosts`).  **Fork scratch:**
`~/llama.cpp` @ `527d39401`, build `build-rocm`, ROCm 7.14 gfx1201.

**Read order:** this file → `README.md` (why) → `PLAN.md` (what to do) → `MEASUREMENTS.md` (evidence).

## State at handover (2026-09-26)

* Investigation complete, **no implementation started**.  The fork tree is clean (`git status`
  empty) and `build-rocm` is rebuilt to the delivery tip (Q6_K ffn back at 57.1 TFLOPS).
* Every experiment run so far was reverted; nothing is left in the fork and nothing is in
  `patches/`.
* The campaign branch and this `wip/` tree are committed (see `git log`).

## The one-paragraph summary

Q6_K prefill is 74-79 % of Q8_0/Q4_K_XL because its MMQ int8-WMMA kernel (`q6_K_q8_1_mma`) is forced
to 16-K fragments by Q6_K's per-16 sub-scales, doubling the A/B ldmatrix issue and the scalar
epilogue.  It measures 57 TFLOPS where Q8_0 gets 83 and F16 107.  All per-16 types share the deficit
(Q6_K 55, Q3_K 57, IQ2_XS 58, IQ2_S 58) while all per-32 types are 70-85.  It is upstream code and
block 04 already optimised it +47 %; the candidate fix is to **fold the per-16 scales into an F16
sram weight tile and run the F16 WMMA with f32 accumulation**, removing the epilogue.  Q6_K is the
PoC.

**But the 107 F16 figure is hipBLAS (prefill F16 has no in-tree tile kernel), and the delivery's own
fused-dequant tile kernel `mmb` (bf16) currently loses to MMQ.**  So the realistic bar is 57, not 90,
and the PoC may fail.  `archive/work/q8-prefill-tuning/`'s MMQ double-buffering is the larger, safer lever.

## Next actions (in order)

1. **Maintainer decision** (see `RESULTS-2026-09-26.md` §6): continue the per-16→F16 PoC, or pivot to
   MMQ double-buffering.  Do not start the kernel until this is settled.
2. **Confirm the numerics contract** (`PLAN.md` risk #1): F16 accumulation is *not* bit-identical to
   the int8 path, so the delivery's "same-seed IDENTICAL" gate cannot hold.  If bit-identity is
   mandatory, stop and archive — do not spend the kernel effort.
3. Start `PLAN.md` Phase 1 (in-tree, env-gated), **using `mma.cuh`'s `load_ldmatrix`/`mma`** — not a
   hand-rolled gfx12 fragment (see the trap in `RESULTS-2026-09-26.md` §2).  Phase-1 gate is
   "correct **and** > 57 TFLOPS", not 90.
4. Record every measurement in a new `RESULTS-<date>.md` (never edit `MEASUREMENTS.md`).

## Traps already paid for

* **Do not** try to merge the two 16-K sub-blocks into one 32-K int8 mma — the RDNA4 fragment
  layout interleaves K across warp halves and the scales get misassigned.  Attempted: 49.7 TFLOPS
  **and** wrong.
* **Do not** credit the J tile / nthreads / I — a full sweep is flat (56.3-56.7).
* **Do not** reach for `mmb` for Q6_K on RDNA4 — it measured 639-733 vs MMQ 981 (scalar dequant).
* The `MUL_MAT_ID(type_a=q6_K,m=64,n=16,k=768)` case already FAILs `test-backend-ops` in the
  baseline; it is pre-existing, not caused by this work.

## Useful commands

```bash
archive/work/per16-f16-mma/tools/counts.sh                 # full per-type TFLOPS landscape
archive/work/per16-f16-mma/tools/repro-gemm-perf.sh gemm q6_K
archive/work/per16-f16-mma/tools/repro-gemm-perf.sh model
archive/work/per16-f16-mma/tools/sweep-j.sh q6_K
```

Fork rebuild loop (incremental, ~50 s when only mmq files change):

```bash
cd ~/llama.cpp && cmake --build build-rocm --target test-backend-ops llama-bench -j 16
```

## Next-session prompt

> Continue the `archive/work/per16-f16-mma` campaign in `~/llama-cpp-rdna-boosts`.  Read
> `archive/work/per16-f16-mma/HANDOVER.md`, then `PLAN.md`.  Phase 0 is done; start Phase 1 (the env-gated
> F16 weight loader + F16 `vec_dot` for Q6_K in `~/llama.cpp`), measure the isolated ffn shape
> `m=17408,n=512,k=5120`, and write the result to `archive/work/per16-f16-mma/RESULTS-<date>.md`.  First
> confirm with me whether the F16-accumulation numerics trade is acceptable.
