# Delivery revalidation — 2026-09-13 (pre-GFX1201 gate)

Final consumer-flow revalidation of the restructured delivery before the GFX1201 port work.
Passed all stages; caught and fixed one delivery defect (below).

## Stage 1: the "Big 13" top-level blocks on master tip

- Master tip = `8b4b3558f` (the blocks base; origin/master == blocks base, zero drift).
- Applied `patches/0001..0013` (amended set: 0002/0004/0008/0013 carry the model-neutral
  Strix folds) — 13/13 clean with plain `git apply`.
- Built clean (HIP/gfx1151, Release): ggml-hip + llama + llama-bench/cli.
- Model Qwen3.6-35B-A3B Q8_0 (`/llm/models/Qwen3.6/35B-A3B/Q8_0/...gguf`, 35.19 GiB):
  - Coherence: sane, coherent output (llama-cli single-turn, thinking-mode model).
  - Perf (same protocol, r3): **pp2048 2136.7 t/s, tg128 52.7 t/s**.

## Delivery defect found by this validation + fixed

`beta/qwen4exp/qwen4exp-support.patch` had been generated as `diff(0013-fold-tree, fork-tip)`
and so still carried the 0002/0004/0008 fold content (gdn NW16, fattn row, scale-unary
unary.cu/cuh — ~117 lines duplicated) — the delivery did NOT compose (support patch
conflicted over the amended blocks). Regenerated as `diff(full-amended-blocks state, tip)`
(6754 lines vs 6871). Delivery commit `e2cb477`.

## Stage 2: qwen4exp branch recreation

- `8b4b3558f` + amended blocks + fixed `qwen4exp-support.patch` → tree == fork tip
  `f5ac11903` byte-identically (0 diff, plain git apply from scratch).
- Built incrementally (validation tree /tmp/val-master, full content == fork tip).
- Model Qwen3.8-Flash-Next IQ4_XS (87.24 GiB):
  - **logitcmp: BIT-IDENTICAL** to the canonical fingerprint (`/tmp/gateA/lg-head-1.txt`,
    fixed 838-token prompt, FNV-40 + top5@9dp, 46 lines).
  - Coherence: sane (llama-cli single-turn).
  - Perf vs B (same-session r3): pp512 647.5 vs 648.8 (parity), **pp2048 759.1 vs 751.0
    (+1.1%)**, pp4096 724.1 vs 726.6 (parity), tg128 ~25.9 both (parity).
  - Depth spot-check (r1): pp2048@d12288 **641.6 vs 531.0 (1.21x)**, tg128@d12288
    23.06 vs 22.08 (1.04x) — reproduces the campaign's depth numbers.

## Notes for the GFX1201 phase

- The delivery artifacts (13 blocks + 1 support patch) now compose from scratch and reproduce
  the validated fork tip. Keep `/tmp/val-master` (built validation tree) until the gfx1201
  work supersedes it.
- Revalidation method note: llama-cli needs `--single-turn` AND `</dev/null`; stray processes
  from aborted runs must be checked (`pkill llama-cli`/`llama-bench`) before GPU runs.
- Recorded in `wip/archive/qwen4exp/` (post-2026-09-13 dated records live there per the
  archive convention).
