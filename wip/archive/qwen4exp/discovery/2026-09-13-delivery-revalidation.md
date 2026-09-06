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

## Qwen3.6-35B-A3B Q8_0 same-session ladder A vs B (follow-up)

The web page's community numbers were measured on a faster machine; on THIS box the
machine-local ladder (r3, warm cache, A/B interleaved) shows A >= B everywhere:

| row | A (Big-13) | B (strix) | A/B |
|---|---|---|---|
| pp512 | 1911.4 | 1824.6 | 1.048 |
| pp1024 | 2222.3 | 2143.7 | 1.037 |
| pp2048 | 2335.4 | 2306.9 | 1.012 (bracket: A 2335.4 ± 0.2%, B 2306.9) |
| pp4096 | 2281.6 | 2274.4 | 1.003 |
| pp8192 | 2176.6 | 2125.8 | 1.024 |
| tg128 | 52.4 | 51.6 | 1.016 |

Machine-state note: A pp2048 measured 2136 in the earlier stage-1 single-shot vs 2335 in
this ladder = ~9% cross-session swing; same-session A/B is the only valid comparison.

## Qwen3.6-27B Q8_0 (DENSE) same-session ladder A vs B — small dense-pp deficit found

The dense model diverges from the MoE results: A trails B by a consistent ~1.3-1.7% on
prefill but leads decode by +5.1%. Bracketed at pp2048 (A spread 0.2%) -> real, not drift.

| row | A (Big-13) | B (strix) | A/B |
|---|---|---|---|
| pp512 | 464.1 | 472.2 | 0.983 |
| pp1024 | 458.2 | 467.4 | 0.981 |
| pp2048 | 450.8 | 456.9 | 0.987 (bracket: A 450.8 +/- 0.2%) |
| pp4096 | 442.9 | 448.3 | 0.988 |
| pp8192 | 428.7 | 430.8 | 0.995 |
| tg128 | 7.89 | 7.51 | 1.051 |

Hypothesis: the recent work was MoE-heavy (routed mmq, swiglu, split_j, pair, weighted-down)
so the dense-pp paths (dense Q8_0 mmq, GDN chunked, flash, elementwise) were only partially
tuned vs B. OPEN: profile dense pp2048 per-op A vs B to locate the ~1.3% (candidates: dense
mul_mat_q per-call, GDN scan, elementwise/launch structure); decide whether to chase before
the GFX1201 phase or defer. NOTE: much of the qwen4exp-support patch is MoE/qwen4exp-gated
and likely INERT on this dense model (only sched-sync + the generic scale-unary window could
fire) - a blocks-only-vs-blocks+support A/B on this model would isolate whether the gap is
in the top-level blocks themselves.

## Dense-pp isolation (Qwen3.5-9B Q8_0 profile + 27B/9B ladders)

Location: the dense Q8_0 J=128 mmq (`mul_mat_q<8,128,false>`, 400 calls/eval, 4-7.6ms each =
~75% of dense-pp time). A is +2.7%/call (2034 vs 1980ms/eval at pp2048) with IDENTICAL grid
geometry and identical split_j config (both trees run the type-Q8_0/J128/I64/nwarps8 split
variant). Register profile: A 136 vgpr vs B 192 vgpr (256 thr, sgpr 128 both) - B's kernel
uses deeper register blocking (more ILP, less traffic); A's leaner kernel loses ~2% on this
workload. Everything else ~parity or A-faster (GDN NW16 scan 91.7 vs B tiled 118.8ms = -27ms;
flash, quantize, elementwise ~parity; rope/rms parity).

Signature: A WINS tiny pp (128/256: +2.7/+1.6% - fewer fixed kernels/launches) and pp4096
(multi-ubatch, scheduling artifacts), loses 512-2048 (-0.9..-1.4%). NOT a fixed per-eval
overhead; it is the per-token dense-mmq cost being ~2% higher mid-range.

Close option (defer): re-block A's Q8_0 J=128 tile toward B's 192-vgpr register profile (port
B's inner loop into A's mmq.cuh) ~1.3-1.5% dense-pp; needs bit-identity re-gate (mma order
may shift). Data: /tmp/prof/q9{A,B}.db_results.db, /tmp/q9{A,B}-*.log, /tmp/d27*.log,
/tmp/q36*.log.
