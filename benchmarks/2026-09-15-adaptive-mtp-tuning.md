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

## Single-card (1 x RDNA4, f16 KV, `-n 3000`)

Both weight types, because the controller's cost/benefit shifts with the per-token cost:

| axis | Q8_0 fixed-3 | Q8_0 cap 7 | Q8_0 cap 12 | UD-Q4_K_XL fixed-3 | UD-Q4_K_XL cap 7 | UD-Q4_K_XL cap 12 |
|---|---:|---:|---:|---:|---:|---:|
| code | — | 63.3 | **71.3** | 63.4 | 61.4 | **84.7** |
| reasoning | 39.4 | — | 38.9 | 47.7 | — | 47.3 |
| prose | 47.7 | — | 53.6 | 56.8 | — | 57.0 |
| code <-> reasoning | — | — | 41.5 | — | — | — |
| recall | — | — | — | — | — | 127.6 |

Cap-12 vs cap-7 on the reporter's weight type, single card: **+12.6 %** (71.3 vs 63.3).  A pinned
depth 10 on that cell reads 67.6, so the depth the controller settles at (~9) is at least as good as
a pinned deep draft.  Reasoning is -1.3 % against fixed-3 (inside the 3 % bound) and prose +12.4 %.
Settled trajectories: code cap-12 `9 -> 10 -> 9`, code cap-7 `4 -> 5 -> 6 -> 7` (the cold start is
`cap - 3`), reasoning `9 -> 8 -> ... -> 3` staying at the floor.

**Caveat to carry forward:** the code <-> reasoning phase-switching axis costs **2.4 %** on
single-card Q8_0 (41.5 against a 42.5 floor optimum), where it cost only 0.5 % on the two-card cell.
What is paid is the cold start plus the `9 -> 3` descent, and the descent is deliberately slow --
`drop_pressure(d) = max(60, 10*d)` is the same constant that stops the `6 <-> 12` churn on the code
axis.  A workload that spends most of its tokens at the floor *on a single card* is therefore the
worst case for this tuning; its trajectory was `9 -> 8 -> 7 -> 6 -> 5 -> 4 -> 3` with `3 <-> 4`
chatter afterwards.  Compare with the two-card mixed reading (64.0 against 64.3) before deciding
whether a shape-specific `drop_pressure` slope is worth it.

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
