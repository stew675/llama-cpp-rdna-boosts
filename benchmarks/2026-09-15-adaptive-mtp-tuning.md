# 2026-09-15 — tuning the adaptive-MTP controller (issue #35)

**Result:** `v16-d1d3c3396-r2`, block-01 amendment. The adaptive draft-depth controller is now the
credit-bucket form, tuned so that a higher ceiling is never worse than a lower one and a
phase-switching workload is not punished.

## Hardware / configuration

| | |
|---|---|
| model (reporter's cell) | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| model (delivery reference) | `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| GPU | 2x RDNA4 (`gfx1201`), `-sm tensor -ts 1/1`, f16 KV (reporter's cell); 1 GPU (reference) |
| sampling | `--temp 0 --seed 42 --single-turn`, greedy |
| length | **`-n 3000`** (a short run measures the warm-up, not the mode) |

## The pinned-depth oracle

`--spec-draft-n-min-adaptive D --spec-draft-n-max D` pins the controller at `D` with zero
transitions, which isolates the depth cost from the controller. Code axis, t/s:

| D | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 80.8 | 88.5 | 94.8 | 96.1 | 96.7 | 99.1 | 98.1 | 99.0 | 97.2 | 94.1 |

The optimum is a broad plateau at 8-10. The same sweep on the other workloads puts the optimum at
the **floor** for reasoning (58.1 at depth 3, 43.7 at 12), prose and the code <-> reasoning
phase-switch (64.3 at 3, 50.0 at 12), and at the **ceiling** for verbatim recall.

## Four-axis gate (reporter's cell, `-n 3000`)

| axis | fixed MTP-3 | adaptive cap 7 | adaptive cap 12 (before) | adaptive cap 12 (after) |
|---|---:|---:|---:|---:|
| reasoning (R) | 57.9 | 57.7 | 57.9 | **60.8** |
| prose (P) | 73.0 | 79.5 | 78.8 | **81.2** |
| code (C) | 80.8 | 95.8 | 92.8 | **96.0** |
| recall (K) | 86.5 | 115.5 | 118.9 | **137.3** |
| code <-> reasoning (phase switch) | 64.3 (pinned opt.) | — | 64.2 | **64.0** |
| code depth changes | — | 8 | 40 | **4** |

Constraints: R within 3 % of fixed-3 (met: +5.0 %), P >= fixed-3 (met), C >= 1.10 x fixed-3 (met:
+18.8 %), **C(cap 12) >= C(cap 7)** (met: 96.0 >= 95.8; was 92.8 < 96.3), K rides to the ceiling
(met: mean depth 10.6).

On the 1-card UD-Q4_K_XL reference: code cap-12 **84.7** vs cap-7 61.4 (+37.9 %), reasoning 47.3 vs
47.7 fixed-3 (-0.8 %), prose 57.0 vs 56.8 (+0.4 %).

## Purity

Same seed, greedy, no `-lv` (the extractor needs a clean stream): `--spec-type none`, fixed
`draft-mtp`, adaptive cap 7 and adaptive cap 12. The three **speculative** configurations are
byte-identical (`sha=6f846d6c2458`, 4514 chars at `-n 1000`); `none` differs, which is the
pre-existing `plain != draft-mtp` residual (`GREEDY-PURITY.md` §36), not a tuning effect. The
adaptive depth never changes the accepted prefix, so the tuning is purity-neutral by construction.

## Method notes (what the investigation established)

1. **The credit function was already right.** The drift's zero-crossing lands on each workload's
   throughput optimum, so the failure was in the *constants*, not the economics.
2. **The ramp was the whole deficit at cap 7 parity.** From the floor the controller spent ~106 of
   477 round-equivalents climbing 3 -> 8 (~2.3 %), which is the entire headroom of a higher ceiling.
3. **Integral windup, not noise, drove the overshoot.** A six-round lucky streak at depth 8 filled
   the bucket and — because the credit grows with depth — cascaded the depth to 12 in 16 rounds.
4. **Churn is the second cost**, ~0.1 % per depth change: the stock configuration made 40 changes in
   477 rounds on a slow 6 <-> 12 limit cycle.
5. **Fewer changes always won.** Every small-threshold variant (fast ramp + fast drop) scored 89-92
   with 106-161 changes; the tuned controller parks within +/-1 of the optimum with 4.
6. **A near-ratchet is wrong.** `drop max(120, 30d)` won pure code (94.7) but lost 2 % on the
   phase-switching prompt, which is why that prompt is now part of the gate.

## Residual

An unexplained ~2 % gap between an adaptive run and a *pinned* run at the same mean depth, in
per-round wall time (the transitions themselves cost only +1.0 ms each). Maintainer hypothesis:
graph invalidation on the depth change. `cap - 3` and the constants are tuned on the reporter's
cell and should be re-tuned per shape.
