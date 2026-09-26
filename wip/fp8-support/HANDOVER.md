# HANDOVER — FP8 support campaign (prepared, not started)

**Branch:** `wip/fp8-support` (delivery repo).  **Not delivery work.**  **Encouraged to start cold.**

Read order: this file → `README.md` → `MEASUREMENTS.md` → `PLAN.md` → `ROCmFPX-ASSESSMENT.md` → the
`reference/` docs.

## State at handover (2026-09-26)

* **Prepared, not started.**  The full 23-commit FP8 series is vendored in `patches/`, the net diff in
  `cllm-fp8-full.diff`, and the branch's campaign docs in `reference/`.
* The FP8 port is the maintainer's own work on the `cllm` branch of the `stew675/llama.cpp` fork,
  last touched **2026-08-06**.  It works and it wins (+13-17 % prefill over Q8_0 on a 4B / gfx1201)
  — but it sits on a **2026-08-05 master**, ~7 weeks behind the current delivery base.
* `~/cllm` (the live clone, tip `7c17faffc`) and `~/aiter` (the AITER inspection tree) are both still
  on disk.  `~/llama.cpp` has only the remote-tracking `origin/cllm` = `535d3bcb1`.
* `~/ROCmFPX` (Ciru's AMD low-bit fork) assessed in `ROCmFPX-ASSESSMENT.md`: no fp8 kernels, but it
  confirms the native-WMMA constraint and offers the DualView / ActiveFPX prefill ideas to cross-check.

## The one-paragraph summary

Single-request prefill is **70-81 % `mul_mat_q`** (sibling campaign `../prefill-gap-attribution/`), so
the vLLM gap is a GEMM problem.  The current int8 MMQ is epilogue-bound; **FP8 E4M3 removes the int8
dequant epilogue** and already measured **+13-17 % over Q8_0** on RDNA4.  The plan is to re-base that
port onto the current delivery base (inheriting the MMB/GEMM/FA work of blocks 08/13/15), convert the
27B FP8 safetensors to `F8_E4M3` GGUF, measure it against the int8 baselines, then lift **AITER's
gfx1201 fp8 GEMM tuning** (121-137 vs 77-98 TFLOP/s) to close the rest.

## Next actions (in order)

1. **Phase 1:** build `~/cllm`, reproduce the 4B `stewfp8-ow.gguf` pp512 ≥ 7184 t/s on a free GPU.
2. **Phase 2:** re-base the 23-commit series onto the current base, dropping the chunked-GDN commits
   that already landed as block 02; gate on the 4B number holding.
3. **Phase 3:** convert the 27B FP8 checkpoint and measure pp8192 vs Q8_0 (1371 t/s) — the real test.
4. **Phase 4:** port AITER's gfx1201 configs into `mul_mat_fp8_wmma`.

## Traps

* **`~/cllm` is the source of truth, not `origin/cllm`** — the tip `7c17faffc` is unpushed and absent
  from `~/llama.cpp`.
* **Bench on a free GPU** — llama-bench on an occupied device OOMs and silently CPU-offloads.
* **Prefer rocprof kernel times (±3 %) over `llama-bench` (±60-170 t/s noise)** for A/B.
* **The conversion scale mapping is the correctness cliff** — HF per-tensor/per-channel weight scales
  → the 128-block `block_f8_e4m3` grid.  Gate on PPL/oracle, not "it runs".
* **Do not adopt AITER's `bpreshuffle`** (5-8× slower on gfx1201; row-major wins).
* Keep the shared `~/llama.cpp` clean; do the FP8 re-base in its own clone/worktree.

## Next-session prompt

> Start the `wip/fp8-support` campaign in `~/llama-cpp-rdna-boosts`.  Read
> `wip/fp8-support/HANDOVER.md`, then `README.md`, `MEASUREMENTS.md` and `PLAN.md`.  This is a prepared
> campaign: the 23-commit FP8 E4M3 port from the maintainer's `cllm` branch is vendored in
> `wip/fp8-support/patches/` (source of truth `~/cllm`, tip `7c17faffc`), measured +13-17 % prefill
> over Q8_0 on a 4B / gfx1201.  Work in a fresh clone/worktree.
>
> Phase 1 first: build `~/cllm` for gfx1201 and reproduce `stewfp8-ow.gguf` pp512 ≥ 7184 t/s on a
> free GPU.  Then re-base the series onto the current base, dropping the chunked-GDN commits already
> in block 02.  Report each gate result before continuing; keep everything in `wip/` — nothing reaches
> `patches/` without the promotion path.
