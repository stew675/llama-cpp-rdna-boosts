# MTP journey measurement — 2026-09-17 (scratch record)

**Status: scratch/WIP. Not part of the delivery. Not pushed.**

This record exists to resolve a real inconsistency: the delivery's current adaptive-MTP
build does **not** reproduce the repo's "current" four-axis table on the dense 1-card cell,
and the upstream+PR#27210 arm looks *faster* on prose. Everything below was measured on one
machine, one methodology, one day, so the three arms are directly comparable.

## Arms

| arm | source | commit | controller |
|---|---|---|---|
| **stock** | `/home/stew675/stock-9113` | `9113cc188` (the PR's exact base) | none (static MTP only) |
| **pr** | `/home/stew675/pr27210` | `d236d41a2` (PR #27210 review head) | **mean-reverting table** (starts at the floor) |
| **deliv** | `/home/stew675/llama.cpp/build-rocm-current` | `31b179037` (current delivery tip) | **credit bucket** (tuned) |

All three built with the same toolchain (ROCm 7.14 gfx1201, clang 23, `gfx1201` target,
`GGML_HIP_GRAPHS=ON`) and only `llama-cli` + server impl. The older canonical delivery build
(`8465f08b9`, previous base `d1d3c3396`) was cross-checked and gives identical results.

> **The table constants are identical between `pr` and the delivery's pre-tuning code**
> (`climb_threshold` = 2/4/10/6/3/2 for depths 1..6, 2 for ≥7; `drop_pressure = max(5·d, 20)`).
> So `pr` is a valid stand-in for "the delivery before the 2026-09-15 controller rewrite"
> *on stock kernels*.

## Cells

| cell | model | layout | KV |
|---|---|---|---|
| `dense1` | Qwen3.8-27B `UD-Q4_K_XL` | 1 GPU | f16 |
| `moe1` | Qwen3.6-35B-A3B `UD-Q4_K_M` | 1 GPU | f16 |
| `q8t2` | Qwen3.8-27B `Q8_0` | 2 GPU `-sm tensor -ts 1/1` | f16 |

Methodology = the delivery's own four-axis gate: `-n 3000`, `--seed 42 --temp 0
--single-turn --no-display-prompt`, `-c 32768 -b 2048 -ub 2048 -fa auto -ngl 99`,
`--reasoning on` for R and `off` for P/C/K, one process at a time (no parallel benches).
Values are `t/s(mean accepted length)`; `--` = not measured / not supported by that arm.

## Headline numbers

### `dense1` — 27B UD-Q4_K_XL, 1 GPU (`t/s(meanlen)`)

**prose** — the anomaly

| spec | stock | pr | deliv |
|---|---|---|---|
| plain | 27.86 | 27.87 | 28.49 |
| static n3 | 58.74 (3.54) | 58.66 (3.54) | 56.77 (3.43) |
| adaptive n12 | — | **68.15 (5.96)** | 57.04 (5.11) |
| adaptive n7 | — | — | 52.25 (4.67) |
| mtp n9 (default start) | — | 69.38 (5.63) | 51.56 (4.78) |
| mtp n9 s9 | — | — | 57.02 (5.11) |
| combo n9 nm45 | — | 69.37 (5.63) | 51.48 (4.78) |
| combo n9 s9 nm45 | — | — | 57.12 (5.11) |
| **pinned 8** | — | 74.89 (5.99) | 69.13 (5.45) |
| **pinned 12** | — | 67.36 (6.32) | **70.97 (6.57)** |

**code**

| spec | stock | pr | deliv |
|---|---|---|---|
| plain | 28.24 | 28.24 | 28.87 |
| static n3 | 62.73 (3.70) | 62.66 (3.70) | 63.56 (3.72) |
| adaptive n12 | — | 81.82 (7.09) | **84.66 (6.71)** |
| adaptive n7 | — | — | 60.85 (5.94) |
| mtp n9 s9 | — | — | 85.07 (6.71) |
| pinned 8 | — | 85.79 (6.61) | 85.76 (6.56) |
| pinned 12 | — | 80.85 (7.33) | 84.35 (7.59) |

**recall**

| spec | stock | pr | deliv |
|---|---|---|---|
| plain | 28.34 | 28.35 | 28.97 |
| static n3 | 67.69 (3.98) | 67.65 (3.98) | 68.26 (3.98) |
| adaptive n12 | — | 106.27 (8.95) | **127.88 (11.13)** |
| mtp n9 s9 | — | — | 120.33 (9.58) |
| combo n9 nm45 | — | **272.04 (26.26)** | 212.31 (33.27) |
| combo n9 s9 nm45 | — | — | 236.48 (38.38) |
| pinned 12 | — | 132.21 (12.22) | 134.95 (12.22) |

**reasoning** — all arms ~46–48; adaptive and static n3 are equivalent. Pinned 12 collapses
(38.4–38.8): deep drafting is actively bad here, and the controller correctly parks at the floor.

### `moe1` — 35B-A3B UD-Q4_K_M, 1 GPU (`t/s(meanlen)`)

| axis | spec | stock | pr | deliv |
|---|---|---|---|---|
| reasoning | static n3 | 162.41 (3.45) | 163.18 (3.45) | 156.12 (3.14) |
| reasoning | adaptive n12 | — | 124.91 (4.73) | **158.98 (4.18)** |
| prose | static n3 | 155.13 (3.33) | 155.87 (3.33) | 157.71 (3.24) |
| prose | adaptive n12 | — | 134.20 (3.57) | **148.85 (3.91)** |
| code | static n3 | 172.10 (3.64) | 172.42 (3.64) | 180.59 (3.63) |
| code | adaptive n12 | — | 133.49 (5.04) | **167.16 (5.87)** |
| recall | static n3 | 185.39 (3.96) | 185.77 (3.96) | 195.74 (3.96) |
| recall | adaptive n12 | — | 191.59 (8.79) | **239.17 (10.66)** |

Deliver **wins adaptive everywhere** (+11 % to +27 %) and is ahead on plain too. The table
controller (`pr`) rides deep, the acceptance collapses, and it loses to static n3 on
reasoning/code — exactly the failure the delivery's acceptance + verify-width work fixes.

### `q8t2` — 27B Q8_0, 2-GPU tensor (`t/s(meanlen)`) — the tuning cell

| axis | spec | pr | deliv |
|---|---|---|---|
| prose | adaptive n12 | 66.41 (5.00) | **80.89 (5.10)** |
| prose | mtp n9 s9 | — | 80.85 (5.10) |
| prose | pinned 8 | 72.62 (5.33) | 77.37 (5.38) |
| prose | pinned 12 | 69.50 (6.23) | 70.13 (5.76) |
| code | adaptive n12 | 83.73 (7.16) | **95.76 (7.11)** |
| code | mtp n9 s9 | — | 97.12 (7.01) |
| code | pinned 8 | 90.56 (6.51) | 98.16 (6.68) |
| code | pinned 12 | 83.64 (7.37) | 93.99 (7.65) |

On the cell the tuning targeted, the bucket **dominates** the table: code +14 %, prose +22 %,
and the delivery adaptive beats its own pinned depths (the controller moves with the content).

## Reconciliation with the recorded 4-axis table

The repo's "current" four-axis record is
[`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`](../../../benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md),
and it was measured **before** the 2026-09-15 controller rewrite. Its acceptance signature is
the table's, and the current build's is the bucket's:

| run | prose acceptance | prose meanlen |
|---|---|---|
| 2026-09-13 record (delivery, pre-tuning) | **0.50654** | 5.10 |
| this record, `pr` (table, stock kernels) | **0.50805** | 5.96 |
| this record, `deliv` (bucket) | **0.60232** | 5.11 |
| 2026-09-15 tuning record, 1-card UD-Q4_K_XL | (bucket) | — |

The current build reproduces the **2026-09-15 tuning record** exactly:

| axis | tuning record (1-card UD-Q4_K_XL, cap 12) | this record `deliv` |
|---|---|---|
| code | 84.7 | 84.66 |
| reasoning | 47.3 | 47.19 |
| prose | 57.0 | 57.04 |

So the confusion is **not** a measurement error and **not** upstream getting faster:

* **stock static MTP is unchanged** vs the 2026-09-13 record (46.8/58.7/62.7/67.7 vs
  47.3/56.3/62.9/68.0) and vs the 2026-09-15 rebase validation. Upstream did not improve.
* The recorded delivery `adaptive n12` row was the **old table**, not today's bucket. The repo
  still points to that record as current, which makes today's build look like it regressed on
  prose while improving on code/recall.

## Is the `pr` prose result real?

Yes. It is reproducible (68.15 with `-lv 4`, 68.4 without) and coherent:

* extracted text (repo's `scripts/extract-generated.py`): `pr` adaptive **7037 chars**
  vs its own plain **7138 chars** — same length, normal prose continuation;
* per-position acceptance decays normally: `(0.902, 0.781, 0.703, 0.521, 0.435, 0.381,
  0.329, 0.290, 0.248, 0.179, 0.130, 0.061)` — no repetition/degeneracy signature;
* `#gen tokens = 3974`, `#acc tokens = 2019` over 407 verify rounds at 35.5 s = 87 ms per
  13-row target forward, consistent with the pinned-12 measurement on the same arm.

Hash cross-check: `deliv` plain prose = `29e02911baed`, which matches the recorded 2026-09-15
purity hash for the same axis — the extractor and cell are the recorded ones.

**Why `pr` is fast:** its table controller rides deep on this content (meanlen 5.96). The
delivery's bucket deliberately sits shallower (5.11). At the **same pinned depth the delivery
is faster** (pinned 12: 70.97 vs 67.36; pinned 8: 69.13 vs 74.89 on prose; code pinned 12
84.35 vs 80.85) — so this is a controller-depth difference, **not** a kernel regression.

## Controller A/B on one build (isolates the controller from the kernels)

Cross-build comparisons (`pr` = stock kernels + table, `deliv` = delivery kernels + bucket) cannot
say whether a difference is the controller or the kernels.  To settle it, the delivery tip was built
once in a scratch worktree with the pre-2026-09-15 **table** controller re-added behind
`GGML_ADAPTIVE_TABLE=1` ([`raw/ab.py`](raw/ab.py); the switch is **not** in the delivery, only in
`/home/stew675/deliv-ab`).  Same binary, same kernels, same numerics — only the controller differs.

`adaptive n12`, `t/s` (mean accepted length):

| cell | axis | bucket (default) | table (`GGML_ADAPTIVE_TABLE=1`) | winner |
|---|---|---|---|---|
| `dense1` | prose | 57.42 (5.11) | **65.58 (5.58)** | **table +14 %** |
| `dense1` | code | **84.68 (6.71)** | 81.00 (6.91) | bucket +4.5 % |
| `dense1` | recall | **127.70 (11.13)** | 106.12 (8.79) | bucket +20 % |
| `q8t2` | prose | **81.62 (5.10)** | 76.20 (5.24) | bucket +7 % |
| `q8t2` | code | **95.30 (7.11)** | 90.13 (6.72) | bucket +5.7 % |
| `moe1` | code | **167.68 (5.87)** | 153.70 (5.79) | bucket +9 % |
| `moe1` | recall | **238.97 (10.66)** | 232.76 (8.64) | bucket +2.7 % |

`adaptive n7` (all differences small): bucket wins `dense1` prose +5 %, `q8t2` prose +2 %,
`q8t2` code +1.3 %; table wins `moe1` recall +2.3 %; `dense1` code/recall and `moe1` code are flat.

**Conclusion:** the tuned bucket is better than the table on **six of the seven** `n12` axes tested
— including both other cells *and* `dense1` code/recall.  The single exception is **`dense1` prose**,
and there it is the controller, not the kernels: the table rides deeper (meanlen 5.58 vs 5.11) on
that content.  So the correct characterisation is "one **axis**", not "one cell".

## Prose-prompt sensitivity (is the `dense1` prose gap a prompt artifact?)

The `orig` prose prompt is the repo's own documentation, whose continuation is a highly predictable
summary.  To test whether the table's `dense1` prose win is a property of *prose* or of *that prompt*,
the single-build A/B was repeated on eight independent Wikipedia (wikitext-2) articles.

| prompt | bucket t/s (ml) | table t/s (ml) | table/bucket |
|---|---:|---:|---:|
| orig (repo docs) | 57.42 (5.11) | 65.62 (5.58) | **+14.3 %** |
| wk1 | 50.50 (4.12) | 55.23 (4.19) | +9.4 % |
| wk2 | 59.24 (5.31) | 56.03 (3.93) | **−5.4 %** |
| wk3 | 48.43 (3.46) | 50.11 (3.06) | +3.5 % |
| wk4 | 51.37 (3.75) | 53.26 (3.35) | +3.7 % |
| wk5 | 50.96 (4.07) | 53.05 (3.32) | +4.1 % |
| wk6 | 48.19 (3.83) | 49.79 (3.51) | +3.3 % |
| wk7 | 54.53 (4.83) | 52.29 (4.32) | **−4.1 %** |
| wk8 | 51.08 (4.44) | 58.74 (4.81) | **+15.0 %** |

Across the eight independent prompts the table's advantage averages **+3.7 %** (range −5.4 % to
+15.0 %), and the bucket wins **2 of 8**.  The repo-docs prompt's +14.3 % is near the top of the
range but not unique (wk8 +15.0 %).

**Conclusion:** the `dense1` prose gap is strongly **prompt/content-sensitive** — exactly the
"one accepted token moves the trajectory ±10 %" behaviour observed while developing the table.
There is a small systematic tilt toward the table on prose (≈ +3.7 % on average), but it is not the
14–18 % the single prompt suggested, and it reverses on some content.  Any re-tuning must be judged
on the *distribution* of prose prompts, not one.

## Extended-corpus ground state (4 prompts per axis, 1 GPU)

The four-axis gate used **one** prompt per axis (`r1`/`p1`/`c1`/`k1`).  To test the overfitting
hypothesis, a 16-prompt corpus (4 per axis; see
[`corpus/MANIFEST.md`](corpus/MANIFEST.md)) was run against the **table** and the **current bucket**,
same build, 1 GPU, adaptive cap 12 (`raw/ground.tsv`).

| axis | prompt | table t/s | bucket t/s | bucket/table | table ml | bucket ml |
|---|---|---:|---:|---:|---:|---:|
| reasoning | r1 | 47.03 | 47.33 | 1.006 | 2.85 | 3.01 |
| reasoning | r2 | 44.36 | 44.38 | 1.000 | 2.64 | 2.74 |
| reasoning | r3 | 48.03 | 45.90 | 0.956 | 2.91 | 2.84 |
| reasoning | r4 | 46.36 | 47.64 | 1.028 | 2.77 | 2.95 |
| prose | p1 | 65.55 | 57.01 | **0.870** | 5.58 | 5.11 |
| prose | p2 | 55.16 | 50.46 | 0.915 | 4.19 | 4.12 |
| prose | p3 | 55.95 | 59.18 | **1.058** | 3.93 | 5.31 |
| prose | p4 | 50.07 | 48.29 | 0.964 | 3.06 | 3.46 |
| code | c1 | 80.89 | 84.56 | **1.045** | 6.91 | 6.71 |
| code | c2 | 71.86 | 68.12 | 0.948 | 5.83 | 5.95 |
| code | c3 | 64.85 | 59.84 | 0.923 | 5.31 | 5.15 |
| code | c4 | 67.45 | 55.07 | **0.816** | 5.42 | 4.78 |
| recall | k1 | 105.62 | 127.63 | 1.208 | 8.79 | 11.13 |
| recall | k2 | 106.05 | 123.24 | 1.162 | 8.81 | 10.68 |
| recall | k3 | 107.41 | 127.77 | 1.190 | 8.86 | 11.22 |
| recall | k4 | 102.98 | 124.48 | 1.209 | 8.43 | 10.83 |

Per-axis geometric mean of bucket/table:

| axis | geo ratio | min | max |
|---|---:|---:|---:|
| reasoning | 0.997 | 0.956 | 1.028 |
| prose | **0.949** | 0.870 | 1.058 |
| code | **0.930** | 0.816 | 1.045 |
| recall | **1.192** | 1.162 | 1.209 |

Overall (geo of the four axis geos): **1.012**.

**This is the overfitting, demonstrated.**

* **Code is the big miss.** The tuning evidence was a single code prompt (`c1`), where the bucket
  wins +4.5 %. Across four code prompts the bucket **loses on three** (c2 −5.2 %, c3 −7.7 %,
  **c4 −18.4 %**) and sits **−7.0 %** on the axis. On `c4` the bucket's mean accepted length is
  4.78 against the table's 5.42 — it under-drafts code by a full token per round.
* **Prose** is −5.1 % on the axis (p1 is worst at −13 %), but wins `p3` +5.8 %.
* **Reasoning** is flat (the bucket parks at the floor, as does the table).
* **Recall** is the bucket's genuine, robust win: **+19.2 %** on every one of the four passages.

So the bucket's true profile is: a real recall win, flat reasoning, and an under-drafting deficit on
code and prose that a one-prompt gate could not see.  The next phase tunes against this distribution.

## The journey, decomposed

On the tuning cell (`q8t2`, code, adaptive cap 12):

| step | t/s |
|---|---|
| PR #27210 (table) on **stock kernels** | 83.7 |
| table on **delivery kernels** (2026-09-15 tuning record "before") | 92.8 |
| **tuned bucket** on delivery kernels ("after" / this record) | 95.8–96.0 |

So the "significant" win the journey is remembered for is **two** contributions: the delivery's
numerics + batch-verification kernels (≈ +11 %) and the controller rewrite/tuning (≈ +3–4 %),
totalling ≈ +15 %. On `moe1` the kernel contribution is much larger (the table's acceptance
collapses; the delivery keeps it).

## The sliding-mean controller (a simpler design we tested)

Idea: drop the climb/drop constants entirely; keep a sliding window of the last N rounds' **accepted
draft counts** and set `n_cur = ceil(mean) + offset`. One knob (the window), plus a small offset.
Implemented behind `GGML_ADAPTIVE_MEAN` (+ `GGML_MTP_MEAN_WINDOW`, `GGML_MTP_MEAN_OFFSET`,
`GGML_MTP_MEAN_PREFILL`) in the same scratch build.

**Prefill matters, and the first cut got it wrong.**  An empty window makes the first round's mean a
single sample, so one early low-accept round collapses the depth.  Pre-filling the window changes the
picture sharply — e.g. `k1` recall goes from `0.94` (empty, w20) to `1.27` (pre-filled with `cap`).
Modes: `cap` (start at the ceiling), `cold` (`cap - 3`), `floor`, `none`.

Best configurations on the 5-prompt set (ratio vs the table; higher is better):

| controller | geo | worst | r1 | p1 | c3 | c4 | k1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| current credit bucket (base) | 0.954 | 0.815 | 1.00 | 0.87 | 0.92 | 0.81 | 1.21 |
| **best tuned credit bucket** (`drop 250/40`, `climb 10/3`) | **1.026** | **0.924** | 1.03 | 1.02 | 0.97 | 0.92 | 1.21 |
| mean, empty, w5 o2 | 0.944 | 0.861 | 0.89 | 0.88 | 0.94 | 0.86 | 1.18 |
| mean, empty, w20 o3 | 0.968 | 0.790 | 0.79 | 0.98 | 1.00 | 0.93 | 1.18 |
| mean, prefill `cap`, w10 o3 | 0.980 | 0.851 | 0.85 | 0.92 | 1.00 | 0.90 | 1.27 |
| mean, prefill `cold`, w10 o3 | 0.959 | 0.760 | 0.76 | 0.93 | 1.00 | 0.90 | 1.27 |

**Conclusion: the pure mean controller is close but not better than the tuned credit bucket**
(0.980 vs 1.026), and it has a structural flaw: with target `ceil(mean) + offset`, the equilibrium is
`d = offset / (1 - p)`, so one offset cannot serve both ends.  `offset 1` is right for reasoning
(p≈0.58 → d≈2.4, the floor) but far too shallow for code (p≈0.66 → d≈3); `offset 3` fixes code and
recall but drives reasoning to `d≈7` and costs it 15-24 %.  The credit bucket's per-depth climb
barrier (the hardened 3→4 step) is precisely the mechanism that pins reasoning at the floor while
still letting code and recall ride deep, and the mean rule has no equivalent.

**Next candidate worth trying: a target-acceptance-rate controller** — increment/decrement the depth
toward a set point `p*` (`d += 1 if observed acceptance > p* else d -= 1`).  Its equilibrium is
`p = p*` *independent of the content*, so it fixes the diverging-equilibrium problem with one knob and
no per-depth constants.

## MoE cross-check (same 4x4 corpus, 1 GPU, 35B-A3B UD-Q4_K_M)

Same build and method; table vs current bucket across all 16 prompts (`raw/ground-moe.tsv`).

| axis | dense geo (27B) | **MoE geo (35B-A3B)** | dense per-prompt | MoE per-prompt |
|---|---:|---:|---|---|
| reasoning | 0.997 | **1.090** | 1.006 1.000 0.956 1.028 | 1.036 1.004 1.133 1.197 |
| prose | 0.949 | **0.943** | 0.870 0.915 1.058 0.964 | 1.034 0.920 0.931 0.894 |
| code | 0.930 | **1.089** | 1.045 0.948 0.923 0.816 | 1.086 1.089 1.094 1.088 |
| recall | 1.192 | **1.014** | 1.208 1.162 1.190 1.209 | 1.024 1.003 1.008 1.020 |
| overall (geo of axis geos) | 1.012 | **1.032** | | |

**The cross-check invalidates two of the dense conclusions:**

* **The code deficit does not reproduce.**  Dense: bucket −7.0 %, losing 3 of 4 prompts (c4 −18 %).
  MoE: bucket **+8.9 %**, winning **all four**.  The dense code deficit is dense-specific (1-GPU
  Q4_K dense verify shape).
* **The recall win does not reproduce.**  Dense: +19.2 %.  MoE: **+1.4 %**.
* **The prose deficit does reproduce** (−5.1 % dense, −5.7 % MoE, 3 of 4 prompts on both): the one
  consistent weakness.
* Reasoning flips from flat (dense) to +9.0 % (MoE, bucket ahead).

So the controller's profile is **model-class-specific as well as prompt-specific**.  Tuning against one
model class would overfit across classes exactly as the single prompt overfit across prompts.


## The target-acceptance-rate controller (implemented and tested)

Implemented as `GGML_ADAPTIVE_TARGET=1` in the same scratch build: a sliding window of the last `W`
rounds' `(accepted, drafted)` pairs, and a unit step of the depth toward a set point `p*`
(`n_cur++` while the pooled rate `p = sum(acc)/sum(dft) > p* + deadband`, `--` while below), with an
optional `stride` (minimum rounds between changes) and a neutral prefill (`GGML_MTP_TARGET_PREFILL`)
so the empty-window bias that hurt the mean rule cannot recur.  Its equilibrium is `p = p*`,
independent of the content.

**It does not beat the bucket, and the reason is a statistic.**  The pooled acceptance rate is not a
sufficient statistic for the optimal depth, because it is an *average*, not the *marginal*.  From the
ground-state logs the operating pooled rate is:

| prompt | table p | table depth | base bucket p | base bucket depth |
|---|---:|---:|---:|---:|
| c4 code | 0.536 | 10.1 | 0.644 | 7.4 |
| c1 code | 0.581 | 11.9 | 0.663 | 10.1 |
| p1 prose | 0.510 | 10.9 | 0.602 | 8.5 |
| r1 reasoning | 0.587 | **4.9** | 0.568 | 5.3 |
| k1 recall | 0.937 | 9.4 | 0.960 | 11.6 |

Reasoning's pooled rate at its *floor* (0.587) is **higher** than code's at its *optimum* (0.536).
Any single `p*` low enough to hold code deep therefore over-drafts reasoning -- the same single-knob
conflict the mean rule had, expressed in rate rather than mean length.  (The table itself escapes it
because its per-depth climb thresholds approximate the *marginal* full-accept probability, not the
average.)

The 5-prompt sweep looked like a win -- `p*=50, w32, stride 4` gave geo 1.038, worst 0.936, versus the
tuned bucket's 1.026/0.923.  The full 16-prompt corpus, dense:

| controller | overall | worst axis | reasoning | prose | code | recall |
|---|---:|---:|---:|---:|---:|---:|
| base bucket (for reference) | 1.012 | 0.930 | 0.997 | 0.949 | 0.930 | 1.192 |
| **`ref_d` credit bucket** (`drop 250/40`, `climb 10/3`) | **1.061** | **0.992** | 1.030 | 1.024 | 0.992 | 1.211 |
| target `t50w32` | 1.048 | 0.945 | 0.945 | 1.004 | 1.013 | 1.257 |
| target `t55w64` | 1.042 | 0.958 | 0.958 | 0.976 | 1.004 | 1.258 |

and MoE: `ref_d` 1.033, `t50w32` 0.972, `t55w64` 0.987.  The target controller is **dominated** on the
full corpus by the retuned bucket; its 5-prompt lead was the same overfit this investigation set out
to expose.

## The multi-cell Pareto finding

`ref_d` -- the one credit-bucket retune that beat the incumbent on the full dense corpus -- is the
best controller found, but it is **not a general improvement** (ratio vs the base bucket):

| cell | prose | code | reasoning | recall |
|---|---:|---:|---:|---:|
| dense 27B Q4_K_XL 1 GPU | **1.079** | 1.067 | 1.033 | 1.016 |
| MoE 35B-A3B 1 GPU | 0.981 | 0.991 | 1.000 | 1.033 |
| 27B Q8_0 2 GPU tensor | 0.951 | 1.013 | 0.985 | 1.004 |
| 27B Q8_0 3 GPU tensor | **0.913** | 1.017 | 1.008 | 1.020 |
| dense phase-switch guard | **0.964** | | | |

`ref_d` improves **all 16** dense prompts (min +0.3 %) and is MoE-neutral, but it **drafts shallower
on every multi-GPU tensor cell** (mean len 3-GPU prose 4.33 vs the base's 5.16) and loses prose
there -- 4.9 % at 2 GPU and 8.7 % at 3 GPU -- plus 3.6 % on the phase-switch guard.

Every conservative variant trades the same way:

| candidate | phase-switch | q8t2 prose | q8t2 code | MoE prose | MoE code |
|---|---:|---:|---:|---:|---:|
| `climbonly` (`climb 10/3`) | 0.991 | 0.973 | 0.973 | **1.034** | **1.017** |
| `drop120` (`drop 120/20`) | 0.973 | 0.978 | 0.995 | 0.987 | 1.009 |
| `climb12` (`climb 12/4`) | 0.984 | 0.986 | 0.980 | **1.033** | **1.025** |
| `hi2` (`drop 200/30`) | 0.957 | 0.996 | 1.001 | 1.006 | 0.998 |
| `hi3` (`drop 200/30` + `climb 10/3`) | 0.982 | 0.960 | 1.014 | 1.013 | 0.966 |

**No tested constant set is a Pareto improvement** across {1 GPU, 2 GPU tensor, 3 GPU tensor} x 16
prompts x the phase-switch guard.  The delivery's base bucket is therefore a good multi-cell
compromise, not a badly-overfit choice: the original concern (the delivery "underperforming the
recorded numbers") was the stale pointer, not the constants.

## The "split the controller by GPU mode" proposal -- tested and rejected

Single-GPU and tensor-split are both easy to detect, so the obvious follow-up is to run `ref_d` on one
GPU and keep the delivery default on 2+.  Two tests killed that.

**(1) The phase-switch objection dissolved.**  The only 1-GPU regression was the one phase-switch
prompt (0.969).  Two new mixed prompts (`corpus/ps2-phase.txt`, `ps3-phase.txt`) put `ref_d` at
**1.039** and **1.016**, so the -3.6 % was that prompt's content, not the mixed-workload case
(3-prompt geo 1.008).

**(2) The discriminator is the quantization, not the split.**  Every multi-GPU cell was Q8_0 while the
1-GPU win was Q4_K_XL -- and `ref_d` behaves the *same* on **1-GPU Q8_0** as on 2/3-GPU Q8_0:

| prose, `ref_d`/base | p1 | p2 | p3 | p4 | axis geo |
|---|---:|---:|---:|---:|---:|
| **1 GPU Q8_0** | 0.957 | 0.944 | 0.980 | -- | **0.960** |
| 2 GPU tensor Q8_0 | 0.951 | 0.900 | 0.942 | 0.953 | 0.936 |
| 2 GPU **layer** (no AR) Q8_0 | 0.957 | 0.943 | -- | -- | ~0.950 |
| 3 GPU tensor Q8_0 | 0.913 | 0.899 | 0.935 | 0.909 | 0.914 |
| 1 GPU Q4_K_XL | 1.174 | 1.062 | 1.022 | 1.064 | 1.079 |

`-sm layer` removes the cross-device AllReduce and the regression is unchanged, so it is **not** the AR
numerics either.  A GPU-mode split would therefore not protect Q8_0 -- and Q8_0 is what the
maintainer's 3-GPU server runs, so `ref_d` would regress it on one GPU or three.

**Conclusion: do not split by GPU mode.**  The only split the data would support is
per-quantization (`ref_d` for ~Q4_K_XL/Q4_K_M, the base bucket for Q8_0), which multiplies the tuning
surface, is unprincipled, and breaks the delivery's "one controller, constants not per-config"
property.  Keep the base bucket.

## The ngram-mod combo (`c9 s9 nm45`) on the 4x4 corpus (Q8_0, 2 GPU)

The 2026-09-17 combo record (`benchmarks/2026-09-17-mtp-ngram-combo.md`) was measured on one prompt per
axis.  Re-run on the full 4x4 corpus, Q8_0 2-GPU tensor, against the delivery default (`mtp12`) and
against `c9s9` (same cap/start, no ngram, to separate the ngram from the cap change):

| arm | reasoning | prose | code | recall | overall |
|---|---:|---:|---:|---:|---:|
| `mtp12` (delivery default) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `c9s9` (cap 9 / start 9, no ngram) | 0.995 | 1.002 | 1.007 | 0.973 | 0.994 |
| **`c9s9nm45` (the combo)** | **0.981** | 1.005 | 1.008 | **1.675** | **1.136** |

(geometric mean of the per-prompt ratio to `mtp12`.)

The recall win **reproduces and is large**: k1 +69.9 %, k2 +78.2 %, k3 +70.8 %, k4 +52.0 %.  The k1
figures match the original record almost exactly (234.8 t/s vs 234.3, acceptance `0.96813` identical).
Prose and code are flat-to-slightly-positive.  The ngram is **silent off recall**: acceptance/meanlen
are identical between `c9s9` and `c9s9nm45` on c1/c2/c4 and every prose prompt, while on k* the mean
len jumps `9.6-11.1 -> 34.9-38.7`.

**The single-prompt gate hid a reasoning cost.**  r1 -- the one reasoning prompt the combo record used
-- is the *only* one that does not regress (+1.0 %).  The other three lose 2.5-3.3 %, for a
**reasoning axis of -1.9 %** -- the same overfit pattern this investigation found for the bucket's code
deficit.  The ngram fires occasionally on reasoning and lowers acceptance there (r3 `0.602 -> 0.569`),
and the cap/start change costs a further ~0.5 % (`c9s9` alone is 0.995).  So the combo's +13.6 %
overall is entirely recall, bought with a small -- but now visible -- reasoning cost.

## Tuning the bucket inside the ngram-mod combo (Q8_0, 2 GPU)

**Question.**  With ngram-mod serving recall, the adaptive MTP no longer has to reach the deep recall
plateau -- so can its drop/climb constants be tuned differently?  (All arms are the combo; ratios are
geometric means over the 12 R/P/C prompts vs the base combo `c9 s9 nm45`.)

**Answer 1 -- the drop/climb constants: no.**  Every variant is worse on R/P/C, exactly as with plain
MTP -- ngram-mod is silent off recall, so the R/P/C depth dynamics are unchanged:

| constants (cap 9 / start 9) | env | reasoning | prose | code | **R/P/C geo** |
|---|---|---:|---:|---:|---:|
| **base** | (default) | 1.000 | 1.000 | 1.000 | **1.000** |
| `ref_d` | `DROP 250/40 CLIMB 10/3` | 0.978 | 0.948 | 0.966 | 0.964 |
| `climbonly` | `CLIMB 10/3` | 0.991 | 0.974 | 1.000 | 0.988 |
| `hi2` | `DROP 200/30` | 0.973 | 0.986 | 0.995 | 0.985 |

**Answer 2 -- the cap: yes, modestly.**  The lever your reasoning points at is real, but it is the
**cap**, not the constants.  Lowering it (start lowered with it) gives a small R/P/C gain with recall
essentially intact:

| cap/start | reasoning | prose | code | recall (k1) | **R/P/C geo** |
|---|---:|---:|---:|---:|---:|
| 3 / 3 | 1.002 | 0.942 | 0.875 | 0.956 | 0.938 |
| 4 / 4 | 1.000 | 0.995 | 0.952 | 0.978 | 0.982 |
| 5 / 5 | 1.002 | 1.008 | 1.000 | 0.985 | 1.003 |
| **6 / 6** | 1.004 | 1.014 | 1.005 | 0.991 | **1.008** |
| 7 / 7 | 1.003 | 1.009 | 1.001 | 1.007 | 1.004 |
| 8 / 8 | 0.990 | 1.005 | 1.000 | 1.012 | 0.998 |
| 9 / 9 (base) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

A lower cap means the MTP drafts are short enough to be fully accepted more often (c1 acceptance
`0.668 -> 0.768`, p4 `0.560 -> 0.600`) so the per-verify cost drops; below cap 4 the code/prose
throughput collapses.

**Answer 3 -- it is the cap, not the start.**  `cap 9 / start 6` scores **0.999** on R/P/C (code 0.984),
while `cap 6 / start 6` scores **1.008** (code 1.005) -- so lowering the initial depth alone does not
reproduce the gain; the ceiling has to come down.

**Verdict:** the drop/climb constants should **not** change under the combo; the cap can optionally
come down from 9 to ~6 for about +0.8 % on R/P/C (recall unchanged, ngram-dominated).  The effect is
~1 % and partly content, so it is a marginal tweak, not a headline.

## Cap 6/7 with `n_match 42` on dense Q4_K_XL 1 GPU (the requested cell)

Combo `c6 s6 nm42` and `c7 s7 nm42` over the 16-prompt corpus, `--reasoning` pinned as usual:

| axis | `c6s6nm42` / mtp12 | `c7s7nm42` / mtp12 |
|---|---:|---:|
| reasoning | 1.015 | 1.006 |
| prose | 0.953 | 0.930 |
| **code** | **0.888** | **0.858** |
| recall | **1.597** | **1.571** |
| overall | 1.082 | 1.060 |

The recall win transfers to dense 1-GPU (+55 to +63 %), but the cap costs code badly -- `c1` alone goes
`84.56 -> 62.84` (-26 %) and `60.05` (-29 %).  **The cause is the cap, not the ngram:** `c6s6` with no
ngram gives `c1 = 62.75` (identical to `62.84` with it), and at cap 9/12 the ngram is code-neutral
(`c9s9nm45` `85.13`, `c12s12nm42` `83.93`, vs `mtp12` `84.56`).

Mechanism: the dense Q4_K_XL drafter naturally accepts `mean len ~6.7` on `c1`, so a cap of 6 **truncates**
it (`5.63` accepted/round) and the 1-GPU verify cost is nearly flat in batch size, so the extra rounds
are pure loss.  On 2-GPU the same cap is ~flat because the verify cost falls with the batch.  So the
cap optimum is split-dependent: **~9 on dense 1 GPU, ~6 on 2-GPU**.

**The `n_match` multiple-of-cap rule holds.**  Adding ngram at cap 12 with `nm42` (12 is not a divisor
of 42) *crashes* the prose p3 acceptance `0.623 -> 0.522` and costs 5 % throughput, while `nm45` at cap 9
(`9x5`) and `nm42` at cap 6 (`6x7`) leave acceptance at/above the no-ngram value -- so `nm42` is the right
choice for 6 and 7, and the dense prose loss above is the cap, not the match.

**Recommendation for dense Q4_K_XL 1 GPU: keep the cap at 9** (the ngram combo with `c9 s9 nm45` is
code-neutral); lowering it to 6/7 buys nothing and costs 11-14 % on code.

## Findings

1. **The repo's "current" four-axis pointer is stale.** `benchmarks/README.md` and
   `benchmarks/mtp-adaptive-methodology.md` still call the 2026-09-13 record current, but it
   was measured with the pre-tuning table controller. Today's numbers live in the 2026-09-15
   tuning record. This is the direct cause of the original confusion.
2. **The controller was overfit to one prompt per axis.** The tuning evidence was
   `r1`/`p1`/`c1`/`k1`. On the 16-prompt corpus the bucket's dense profile is a real recall win
   (+19.2 %), flat reasoning (0.997), a prose deficit (−5.1 %) and a **code deficit (−7.0 %,
   worst −18.4 %)** that `c1` alone hid — the bucket wins `c1` (+4.5 %) and loses `c2`/`c3`/`c4`.
3. **The profile is model-class-specific too.** The MoE cross-check *reproduces* the prose
   deficit (−5.7 %) but **reverses code** (bucket +8.9 %, all four) and **removes the recall
   win** (+1.4 %). Reasoning flips from flat to +9.0 %. So any tuning must be validated on both
   model classes, not just both prompts.
4. **A sliding-mean controller (one window knob) was tested and is close but not better.** Best
   geo 0.980 vs the tuned credit bucket's 1.026. Pre-filling its window matters a lot (recall
   `k1` 0.94 empty -> 1.27 pre-filled). Its structural limit is the `d = offset/(1-p)` equilibrium:
   one offset cannot serve reasoning (floor) and code (deep) at once; the credit bucket's
   per-depth climb barrier is what does. A target-acceptance-rate controller is the better
   simple design.
5. **The kernel work and the controller are separable, and both matter.** On `q8t2` code
   (adaptive n12): PR (table, stock kernels) 83.7 -> table on delivery kernels 90.1 (**+7.6 %**,
   kernels/numerics) -> bucket on delivery kernels 95.3 (**+5.7 %**, controller). On `moe1` the
   kernel/acceptance share is larger still (the table's acceptance collapses there).
6. **Static MTP is bit-identical between stock and PR** (identical acceptance/meanlen/rounds on
   all four axes), confirming the PR adds only the adaptive mode; the single-build A/B is the
   direct controller isolation.
7. **`deliv` plain decode is ~2 % faster** than stock/PR (block 10 + decode work), and its
   prefill is ~20 % faster (block 04/08/13/14) — visible in the prompt-eval rates
   (1181 vs 974 t/s on prose).
8. **The target-acceptance-rate controller is dominated.** Implemented, swept and rejected: its
   pooled-rate criterion is an *average*, not the *marginal* full-accept probability, so a `p*` that
   holds code deep over-drafts reasoning (reasoning's floor rate 0.587 > code's optimum rate 0.536).
   Full-16 dense 1.048 vs the retuned bucket's 1.061; MoE 0.972 vs 1.033.
9. **No retune is a Pareto improvement, and the split is the quantization.** The best candidate
   `ref_d` gives a uniform Q4_K_XL-dense-1-GPU win (+4.8 %, all 16 prompts) but loses **Q8_0 prose on
   every configuration tried** -- 1 GPU 0.960, 2 GPU tensor 0.936, 2 GPU layer 0.950, 3 GPU tensor
   0.914 -- while the Q4_K_XL prose axis is 1.079. Every conservative variant trades the same way.
   **A GPU-mode split does not help**: the loss is the weight quantization, not the split or the AR
   (`-sm layer` shows it too), and Q8_0 is the maintainer's server quant. **No block-01 change is
   recommended** from this work.
10. **The recommendation is "no change".** The apparent underperformance that motivated this
    investigation was the stale 2026-09-13 record pointer (measured with the *table*), not the
    delivery's credit bucket.
11. **The phase-switch "regression" was one prompt.** The original guard fell 3.6 %, but two new mixed
    prompts rose 3.9 % and 1.6 % (geo 1.008); the guard does not object to `ref_d`. The quant (9) is
    the only real obstacle, and it is not addressable by a mode branch.
12. **Tuning inside the combo: the constants cannot move, the cap can.** With ngram-mod on recall the
    bucket's drop/climb constants are still optimal at their base values (ref_d/climb/hi2 all cost
    1-4 % on R/P/C); the lever is the **cap**, which can drop 9 -> ~6 for +0.8 % on R/P/C with recall
    intact (and it is the cap, not the start). Effect ~1 %, partly content -- a marginal tweak.
13. **The cap optimum is split-dependent; `n_match` should be a multiple of it.** On dense Q4_K_XL
    1 GPU, `c6 s6 nm42` / `c7 s7 nm42` win recall (+60 %) but cost code 11-14 % (`c1` -26/-29 %) and
    prose 5-7 % -- and the cause is the **cap**, not the ngram (`c6s6` without ngram is identical on
    code). The dense drafter accepts `~6.7`/round, so cap 6 truncates it and the 1-GPU verify cost is
    flat in batch, making the extra rounds pure loss.  Cap ~9 on dense 1 GPU, ~6 on 2-GPU.
    Separately, the **multiple-of-cap `n_match` rule is confirmed**: `nm42` at cap 12 (42 is not a
    multiple of 12) crashes p3 acceptance `0.623 -> 0.522`, while `nm45`/cap 9 and `nm42`/cap 6 are clean.
14. **The ngram-mod combo lands as advertised on recall and hides a reasoning cost.** On the Q8_0
    2-GPU 4x4 corpus `c9 s9 nm45` is recall **+67.5 %** (per prompt +52 to +78 %), prose +0.5 %, code
    +0.8 %, overall **+13.6 %**, and the ngram is provably silent off recall. But reasoning is
    **-1.9 %**: the original gate's r1 is the only reasoning prompt that does not regress. The combo
    remains a per-context opt-in (not a default) and its reasoning cost should be documented with it.

## Follow-ups (not done here)

* The stale "current record" pointers in `benchmarks/README.md` and
  `benchmarks/mtp-adaptive-methodology.md`.
* Re-tune the bucket's `drop_pressure`/`climb_budget`/cold-start for the 1-card dense cell
  (or make the credit verify-cost-aware, since the bucket's token-count credit ignores the
  per-cell cost of a wide verify). Gate on `dense1` prose **and** the phase-switching prompt.
* Measure `stock` on `q8t2` (plain + static n3) to complete that cell's baseline — currently
  filled from the 2026-09-15 tuning record instead.

## Files

* `raw/canonical.tsv` — every accepted run (cell, arm, axis, spec, tps, tokens, acc, meanlen).
* `raw/journey.tsv`, `raw/dense1-first-pass.tsv` — the two raw sources merged into canonical.
* `raw/coh_*.log` — the coherence/purity runs (no `-lv 4`) with extracted text.
* `raw/logs/` — the `-lv 4` logs for the prose axis.
* `raw/matrix.py`, `raw/extra.py`, `raw/pinned.py`, `raw/ab.py`, `raw/prompt_ab.py`, `raw/analyze.py` — the harnesses.
* `raw/ab-logs/` — the single-build controller A/B logs.
* `raw/prose-ab-logs/` — the prose-prompt sensitivity logs (9 prompts x bucket/table).
* `raw/prose_p1..p8.txt` — the wikitext-2 prose prompts used.
