# WIP: adaptive-MTP ceiling scaling (quant × GPU split)

**Tracked as issue [#35](https://github.com/stew675/llama-cpp-rdna-boosts/issues/35) — "Adaptive MTP
behaviour after recent performance tuning".**  The original report was posted onto issue #30 (comments
[5683949195](https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5683949195) /
[5684693398](https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5684693398)), but
it is a different class of problem and was split out into **#35** on 2026-09-15 — **this dossier tracks
#35, not #30**.  Reporter **@1337hero**.  The data below was collected 2026-09-15 after the
`v16-d1d3c3396-r1` re-base, from the `v16-790cf51aa-r5` reporter evidence plus our own reproduction.

**Status: CONFIRMED (narrowly).**  The headline cell — Qwen3.8-27B **Q8_0**, 2-card `-sm tensor`,
adaptive `--spec-draft-n-max 7` vs **12**, code prompt, `-n 3000` — reproduces on the re-based
delivery: **n7 95.1 → n12 89.6 t/s (−5.8%)**, reproducible across back-to-back runs and on **BF16**
KV as well as f16.  The broader claim ("tensor split erases the k-quant gain") did **not** reproduce
here.  This is a **decode/verify performance & tuning** issue, **not** a purity bug (the reporter's
purity is clean, and our `n3` purity gate is unaffected).

## The report

The adaptive-MTP ceiling **12** is what the delivery recommends
([`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`](../../benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md),
[`benchmarks/mtp-adaptive-methodology.md`](../../benchmarks/mtp-adaptive-methodology.md)).  That table
is **UD-Q4_K_XL on one card**.  The reporter swept ceiling 7 → 12 across three quants and two layouts
and reported (their r5 build, their box):

| quant | 1 card, prose / code | 2-card `-sm tensor`, prose / code |
|---|---|---|
| Q4_K_M | +32 % / +31 % | 0 % / −6 % |
| Q6_K | +23 % / +29 % | −2 % / −3 % |
| Q8_0 | −1 % / +4 % | **−13 % / −9 %** |

Their ask: the "recommended ceiling 12" needs a caveat (7 is the right default on Q8_0 / any tensor
split), optionally enforced by capping the adaptive default at 7 when `n_gpu > 1` or the weight type
is Q8_0 — the same pattern as the `ncols2` split gate.  They also note ROCm **7.2.4** breaks the
purity guarantee (filed separately below) and that the AR is not the cause (they A/B'd
`GGML_CUDA_P2P=1` / `GGML_CUDA_ALLREDUCE=internal`), and the `ggml_set_fa_tensor_parallel` hint is
not either (it steers prefill `ncols2` only; forcing it 0/1 left decode flat).

## Maintainer hypothesis (stew675, #35) — the climb/drop table is tuned for stock

The maintainer reads the ceiling-scaling symptom as a **controller-tuning** issue rather than an
upstream regression: the delivery's drafting-accuracy work raised acceptance well above the mainline
behaviour the adaptive controller was originally tuned against, so the controller now climbs faster and
higher than its constants assume — and the extra depth hurts when the (improved, but not perfect)
drafter misses.  The prime suspect is therefore the per-depth **climb/drop cost table** in
[`common/speculative-adaptive.h`](../../common/speculative-adaptive.h) (block 01):

| depth | `climb_threshold` (consecutive full accepts needed to climb one step) |
|---:|---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 10 (the hardened 3→4 barrier) |
| 4 | 6 |
| 5 | 3 |
| 6 | 2 |
| ≥ 7 | **2** |

`drop_pressure(depth) = max(depth * 5, 20)`.

Above depth 6 the climb needs only **2** consecutive full accepts per step, so a high-acceptance
workload walks straight up to the cap and sits there.  The measured acceptance / mean-length table
below is exactly that signature: acceptance falls (0.737 → 0.586) while mean accepted length saturates
(5.70 → 6.72).  The Q8_0 × tensor-split cell is simply where the wide verify is most expensive relative
to its marginal acceptance.  **Retuning the `>= 7` band (or adding a verify-cost term) is the leading
fix, ahead of a blanket `n_gpu > 1` cap.**  The controller has a unit test
([`tests/test-speculative-adaptive.cpp`](../../tests/test-speculative-adaptive.cpp)) to drive the
retune.

## Reproduction on `v16-d1d3c3396-r1`

Local box: 3× R9700 (gfx1201), ROCm 7.14 (`/opt/rocm-7.14-gfx1201`), build from the re-based set
(applied tree `c6896785a5fefdf9438d26974c0274bf99f43263`).  Model `Qwen3.8-27B-Q8_0.gguf`,
`HIP_VISIBLE_DEVICES=1,2 -sm tensor -ts 1/1`, `-c 32768 -b 2048 -ub 2048 -fa auto -ngl 99`,
`--spec-type draft-mtp-adaptive`, `--reasoning off`, `-n 3000 --seed 42 --temp 0`, greedy.  Metrics
are the llama-cli footer t/s and the `-lv 4` acceptance / mean accepted length (one run per cell; the
key cell reproduced twice).

### The ceiling curve (code, 2-card tensor, f16 KV)

| ceiling | tg t/s | acceptance | mean accepted len |
|---:|---:|---:|---:|
| 7  | **95.1** | 0.73733 | 5.70 |
| 8  | 92.2 | 0.71388 | 6.07 |
| 9  | 93.5 | 0.68159 | 6.37 |
| 10 | 92.8 | 0.65413 | 6.55 |
| 11 | 92.4 | 0.62341 | 6.69 |
| 12 | 89.6 | 0.58636 | 6.72 |

### BF16 KV (the maintainer's default) — same shape, ~2 % faster absolute

| ceiling | tg t/s (bf16) | acceptance | mean len |
|---:|---:|---:|---:|
| 7  | **97.0** | 0.74344 | 5.96 |
| 10 | 94.5 | 0.65418 | 6.94 |
| 12 | 90.6 | 0.58509 | 7.07 |

Forcing native-bf16 FA (`GGML_CUDA_FA_KV_NATIVE=1`) made no difference here (97.0 / 90.7).  **BF16 is
therefore not the cause** — the regression shape is identical to f16.

### Quant × split, ceiling 7 → 12 (f16 KV, code unless noted)

| quant | 2-card tensor | 1-card |
|---|---|---|
| Q8_0 | code 95.1 → 89.6 (**−5.8 %**); prose 79.5 → 75.7 (**−4.8 %**) | code 61.9 → 66.4 (+7.3 %, q8_0 KV) |
| Q4_K_XL | code 94.7 → 102.5 (+8.2 %); prose 87.0 → 92.7 (+6.6 %) | (n12 record: 1-card code +35 % vs ceiling 7) |
| Q6_K | code 83.5 → 92.9 (+11.3 %); prose 69.9 → 74.7 (+6.9 %) | — |

## What is confirmed vs not

* **Confirmed:** Q8_0 + 2-card tensor split loses going 7 → 12 (both code and prose, f16 and bf16).
  Depth **10 recovers only part** of it (92.8 f16 / 94.5 bf16) and is still below ceiling 7.  The knee
  is at 8, not 12.
* **Confirmed:** 1-card Q8_0 still gains from deeper drafting (+7.3 % code), so the loss is specific to
  the **Q8_0 × tensor-split** combination, not Q8_0 alone.
* **Not reproduced:** the k-quant 2-card loss.  On this box Q4_K_XL and Q6_K *gain* at ceiling 12 on a
  tensor split on both axes.  The reporter's Q4_K_M/Q6_K numbers may differ because of the quant file
  (their HF-converted Q4_K_M/Q6_K vs our Unsloth UD-Q4_K_XL / Q6_K) or their ROCm build.

## Absolute values — a caution for the next session

Cross-machine absolutes are not comparable here:

| source | 2-card Q8_0, code, adaptive |
|---|---|
| historical Protocol B (server, 2-GPU tensor, f16 KV, 2026-09-02 13-block) | C3 (adaptive) **79.1** |
| reporter's r5 cli | n7 87.8 / n12 **79.5** |
| our `v16-d1d3c3396-r1` cli | n7 **95.1** / n12 **89.6** |

Our numbers are ~14 % above the reporter's and the older server record on the same cell.  Both
harnesses and builds differ (server `predicted_per_second` vs cli footer; their TheRock ROCm; our
newer upstream base and local 7.14).  **Trust the n7-vs-n12 ratio, not the absolute t/s**, when
comparing against the reporter.

## Mechanism (evidence, not yet root-caused)

* Acceptance falls monotonically with ceiling (0.737 @7 → 0.586 @12) while mean accepted length
  saturates (5.70 → 6.72).  The deeper draft's marginal acceptance no longer pays for the wider
  verify.
* The wide verify is exactly where the delivery's kernel families switch: >8 query rows changes the
  FA chooser (tile/MMA) and `ncols == 8` changes the matmul family (MMVQ/MMVF → MMQ) — the same
  boundary that makes purity above `n_max 7` a warned trade
  ([`GREEDY-PURITY.md`](../../GREEDY-PURITY.md) §11).  Q8_0's weight mat-vec path is the reporter's
  suspect and is the natural place to look.
* Ruled out so far: the all-reduce (reporter A/B'd P2P/internal), and the
  `ggml_set_fa_tensor_parallel` hint (prefill `ncols2` only).

## Investigation log — controller dynamics, pinned depth, and the bucketed alternative

**The design constraints (maintainer, #35).**  The adaptive controller must hold all of these at
once: **R** ≤ 1.03 × fixed MTP-3; **P** ≥ fixed MTP-3; **C** ≥ 1.10 × fixed MTP-3 **and
C(n12) ≥ C(n7)** (ideally a little better); **K** must climb to depth 12 quickly.

**Four-axis gate, Q8_0 27B × 2-card `-sm tensor`, f16 KV, `-n 3000`** (our box; 1 run/cell):

| axis | fixed n3 | adaptive n7 | table adaptive n12 | bucketed adaptive n12 |
|---|---:|---:|---:|---:|
| reasoning | 58.3 | 57.7 | 57.7 (−0.5%) | 57.8 (−0.9%) |
| prose | 73.2 | 79.5 | 76.5 (+5.1%) | 78.8 (+7.7%) |
| code | 80.5 | **95.0** | 91.3 (+13.4%) | 92.7 (+15.4%) |
| recall | 86.9 | 115.5 | 124.9 (+43.7%) | 119.6 (+37.5%) |

Against fixed MTP-3 **all four constraints pass** — but **C(n12) < C(n7)** on the code axis is the
violation the maintainer added, and the reason this is a tuning problem rather than a doc caveat.

**Pinned depth isolates the cost.**  `--spec-draft-n-min-adaptive D --spec-draft-n-max D` pins the
controller at exactly `D` (0 transitions).  Same cell: 7 → 97.0, **10 → 99.5**, 11 → 98.0,
12 → 94.5 t/s.  **Depth 10 is the code optimum**, and every adaptive run is *below* the pinned
throughput at its own mean depth — i.e. the controller's ramp + wander costs ~2-5%, and the
ceiling-12 loss is the controller settling at 11–12 rather than at 10.

**Mechanism (measured per round).**  At depth 7 code's full-accept rate is 0.51; at 11–12 it is
0.21–0.28.  The table controller's `update()` **zeroes `n_drop` on every full accept**, so a
workload that fully accepts every few rounds never accumulates the drop pressure and parks at
11–12 — where the wide verify is dearest.  Removing the reset (partial relief) shifted the balance
by only ~1 round and did not fix the cell.

**The bucketed alternative (maintainer's preferred direction).**  `bucketed-port/` carries the
maintainer's `bucketed-adaptive-mtp` controller (single credit bucket, no hard resets, surplus/
deficit carried across depth changes) ported onto block 01, plus its measurements.  It beats the
table on prose and code but under-climbs on recall, and it still gives C(n12) < C(n7) — it was
tuned against upstream acceptance rates.  A depth-weighted credit that helps R/P/K pushes code's
equilibrium *above* 10 toward 12, so the lever is controller **stability**, not the climb rate.

## Next steps (handover)

1. **Tune the bucketed controller** (`bucketed-port/`, the maintainer's preferred approach): the
   win is holding code near depth 10 (pinned 99.5 vs an adaptive ~92) while recall rides at 12.
   The table's `climb_threshold`/`drop_pressure` path was tried and does not fix the cell
   (the `+1` climb and the partial drop-relief both leave C(n12) < C(n7)); a depth-weighted credit
   helps R/P/K but overshoots code to 12.  Focus on hysteresis/stability and a depth-dependent
   climb budget, not on the raw climb rate.  Validate against **all five** constraints
   (R ≤ 1.03×, P ≥, C ≥ 1.10× **and C(n12) ≥ C(n7)**, K → 12) on both the reporter's cell and the
   delivery's 1-card reference.
2. Root-cause the Q8_0 wide-verify cost before changing the controller: profile the verify batch
   (`n_q > 8`, Q8_0 weights, Q8_0/f16 KV) with `test-backend-ops perf` at the verify widths and a
   `llama-batched-bench`/kernel-family A/B.  Is it the FA kernel, the Q8_0 `MUL_MAT`, or the
   tensor-split dispatch?
3. Distinguish "2 cards" from "tensor split": compare `-sm layer` vs `-sm tensor` at n7/n12 on the
   same 2 cards.
4. Re-check the k-quant 2-card claim with a **Q4_K_M** file (matching the reporter) before trusting
   either side.
5. Ship a doc caveat regardless (the reporter's explicit ask): the "recommended ceiling 12" line in
   [`benchmarks/README.md`](../../benchmarks/README.md), [`benchmarks/mtp-adaptive-methodology.md`](../../benchmarks/mtp-adaptive-methodology.md)
   and [`README.md`](../../README.md) should say 7 is the default on Q8_0 / tensor split.

## Separate item in the same report: ROCm 7.2.4 purity

The reporter also reports that on ROCm **7.2.4** the delivery's purity contract breaks (plain ≠ spec
on 3 of 5 prompts, acceptance ~20 % lower) and that it is clean on 7.14.  That is a *toolchain*
observation, unrelated to the ceiling scaling; if confirmed it belongs in the README as a supported-
toolchain note (the delivery's validation is ROCm 7.14 only).  **Not investigated here.**

## Files

* `repro.sh` — the reproduction above (parameterised; `new`/`stock`).
* [`HANDOVER.md`](HANDOVER.md) — the turnkey brief for the next session.
