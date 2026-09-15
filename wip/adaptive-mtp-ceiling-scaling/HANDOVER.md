# HANDOVER — adaptive-MTP ceiling scaling

Turnkey brief for the next session.  Read [`README.md`](README.md) first (the finding, the data, the
confirmed/not-confirmed split, the four-axis and pinned-depth data, and the controller dynamics).
**The maintainer prefers the bucketed controller** — start at
[`bucketed-port/`](bucketed-port/README.md) (his `bucketed-adaptive-mtp` algorithm ported onto
block 01, with its numbers and the tuning direction), not at the table.  **Tracked as issue
[#35](https://github.com/stew675/llama-cpp-rdna-boosts/issues/35).**

## TL;DR

Confirmed on `v16-d1d3c3396-r1`: **Qwen3.8-27B Q8_0, 2-card `-sm tensor`, adaptive
`--spec-draft-n-max 12` loses ~5-6 % to `n_max 7`** (code and prose, f16 and BF16 KV).  Depth 10 is
between the two.  1-card Q8_0 still wins from 12, and Q4/Q6 2-card still win here — so the loss is
specific to **Q8_0 × tensor split**, and it is a *tuning/performance* issue, not a purity bug.

**Hypothesis, now refined:** the controller's constants were tuned for mainline (low) acceptance, and
the delivery's higher acceptance moves the operating point.  Measured: the table parks code at 11–12
(too high) while the bucketed controller parks it at ~7.8 (too low) — the issue is the controller's
**spread**, and the maintainer's bucketed design is the preferred base to tune.  See `README.md` and
`bucketed-port/`.

## Setup

```sh
# Delivery build (already built once):
cd ~/llama.cpp && git worktree add -f ~/llama-cpp-rebase rdna-boosts-v17
cd ~/llama-cpp-rebase && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714     # ~6 min, gfx1201
# Stock reference at the same fork point, if needed:
cd ~/llama.cpp && git worktree add -f ~/stock-d1d3 $(git rev-parse d1d3c3396)
cd ~/stock-d1d3 && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
```

Models: `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`, `.../Q6_K/...`,
`.../Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf`.  Prompts under `prompts/`.

## Reproduce

```sh
wip/adaptive-mtp-ceiling-scaling/repro.sh new        # the whole sweep (~20 min)
# the key cell by hand:
HIP_VISIBLE_DEVICES=1,2 build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf \
  -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
  -p "$(cat prompts/code-python.txt)" --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
  -sm tensor -ts 1/1 -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
# repeat with --spec-draft-n-max 12; compare the footer Generation t/s and the -lv 4 acceptance.
```

**llama-cli MUST always be invoked with `--single-turn`** (else it blocks in the chat loop).  Use a
separate run per cell; do not run benches in parallel with anything else.

## Measured data on `v16-d1d3c3396-r1` (self-contained; do not re-derive)

Q8_0 27B, 2-card `-sm tensor -ts 1/1`, code prompt, f16 KV:

| ceiling | tg t/s | acceptance | mean accepted len |
|---:|---:|---:|---:|
| 7  | **95.1** | 0.73733 | 5.70 |
| 8  | 92.2 | 0.71388 | 6.07 |
| 9  | 93.5 | 0.68159 | 6.37 |
| 10 | 92.8 | 0.65413 | 6.55 |
| 11 | 92.4 | 0.62341 | 6.69 |
| 12 | 89.6 | 0.58636 | 6.72 |

BF16 KV (same cell): 7 → 97.0, 10 → 94.5, 12 → 90.6.  Native-bf16 FA forced (`GGML_CUDA_FA_KV_NATIVE=1`)
is a no-op here.  Quant × split, 7 → 12: Q8_0 2-card code −5.8 %, prose −4.8 %; **Q4_K_XL and Q6_K
2-card still gain** (+8.2/+6.6 %, +11.3/+6.9 %); 1-card Q8_0 still gains (+7.3 %, q8_0 KV).  So the
confirmed loss is exactly **Q8_0 × tensor split**.

## The five constraints a tuned controller must hold at once (all at `-n 3000`)

1. **R** (reasoning) ≤ 1.03 × fixed MTP-3.
2. **P** (prose) ≥ fixed MTP-3.
3. **C** (code) ≥ 1.10 × fixed MTP-3 **and C(n12) ≥ C(n7)** (ideally a little better).
4. **K** (recall) climbs to depth 12 quickly.

**RESULT (2026-09-15): the tuned bucketed controller meets all five** — see
`bucketed-port/tuned-port.patch` + its README.  Code cap-12 **96.0** vs cap-7 **95.8** on the
reporter's cell (was 92.8 vs 96.3), R +5.0 %, P +11.2 %, recall +58.7 % riding at depth 12, and the
phase-switching prompt 64.0 vs its 64.3 pinned optimum.  The three tuning changes are a **cold start**
at `cap - 3`, a **depth-growing climb budget** `20 + 6*(depth-1)`, and a **steeper drop pressure**
`max(60, 10*depth)`.  What remains is the ~2 % adaptive-vs-pinned per-round gap, per-shape re-tuning
of `cap - 3`, and re-deriving `tests/test-speculative-adaptive.cpp` for the new defaults.

On Q8_0 × 2-card tensor all pass except **C(n12) < C(n7)** (n7 95.0, table n12 91.3, bucketed
n12 92.7).  Pinned depth says the code optimum is **10** (99.5 t/s; 7 → 97.0, 12 → 94.5), so the fix
is not a blanket ceiling — it is a controller that holds ~10 on code and rides 12 on recall.  Judge
candidates on all four axes at once; a control change that helps one axis but breaks another is not
a fix.

## Controller internals (the prime suspect)

| item | location |
|---|---|
| the climb/drop table | `common/speculative-adaptive.h` — `climb_threshold()`, `drop_pressure()` |
| state + transition | `common/speculative-adaptive.h` — `reset()`, `update()` |
| per-round feedback | `common/speculative.cpp` — `adaptive_feedback()` (~L1819), called from `accept()`/`accept_partial()` |
| depth applied to the draft cap | `common/speculative.cpp` (~L1679): `n_cap[seq_id] = adaptive ? adaptive_ctrl[seq_id].n_cur : params.n_max` |
| init / range validation | `common/speculative.cpp` (~L1456-1480); floor is `--spec-draft-n-min-adaptive` (default 3) |
| CLI knob | `common/arg.cpp` (~L4170) `--spec-draft-n-min-adaptive` |
| unit test (drive the retune) | `tests/test-speculative-adaptive.cpp` |

Quick experiments to try (one variable at a time, re-run the 7/10/12 curve for Q8_0 2-card):

1. Raise the `>= 7` climb from **2** to 4/6 (match the depth-4/3 barriers).
2. Raise `drop_pressure` above `depth * 5` (e.g. `depth * 8`) so misses eject depth sooner.
3. Make the climb/drop depend on the *verify cost* (e.g. penalise climbing when `n_gpu > 1` or the
   dominant weight type is Q8_0, which is where the wide verify is dearest) — the principled version
   of the reporter's cap.

If the retune cannot close the gap, the fallback is the reporter's blanket rule: default the adaptive
cap to 7 when `n_gpu > 1` (tensor) or the dominant weight type is Q8_0.  Detect `n_gpu` /
split-mode at the same place the context is created and pass it into the controller; the weight type is
on `llama_model`/the layer tensors.

## Where to look (delivery blocks)

| area | files / notes |
|---|---|
| **adaptive controller — the climb/drop table** | **`common/speculative-adaptive.h`** (`climb_threshold` / `drop_pressure`), block 01; unit test `tests/test-speculative-adaptive.cpp`; `patches/README.md` block-01 notes |
| wide-verify matmul family (MMVQ/MMVF → MMQ at `ncols == 8`) | block 08/10/13; `patches/README.md`; `GREEDY-PURITY.md` §11/§19 |
| FA chooser (tile/MMA at `n_q > 8`) | block 00/03/04/08; `GREEDY-PURITY.md` §11 |
| tensor-split dispatch / AR | block 12; but note the reporter **ruled the AR out** (P2P/internal A/B) |
| prefill `ncols2` split hint | `ggml_set_fa_tensor_parallel` — **ruled out** (prefill only; forcing 0/1 left decode flat) |

## Gates

Before proposing a change, run:

1. The key cell above, `n7` vs `n10` vs `n12`, on **Q8_0 2-card** — the change must remove the loss.
2. `benchmarks/mtp-adaptive-methodology.md` rule 5 (verify-width): `llama-batched-bench -npp 16 -ntg 32
   -npl 1,4,8` with a quantized KV on a dense K-quant — no regression at B=4/B=8.
3. Purity: `--spec-type none == draft-mtp --spec-draft-n-max 3` byte-identical (27B, all supported KV
   types).  A ceiling cap must not change single-token decode or the `<= 7` band.
4. The four-axis gate (`-n 3000`) to confirm the Q4/Q6 single-card wins are untouched.
5. Cross-arch spot check on `halo` (gfx1151) — the ceiling default is arch-independent, but the
   `n_gpu` cap interacts with split mode.

## Acceptance criteria for a fix

* Q8_0 2-card, code/prose, `n12` no longer below `n7` (ideally within noise; a cap at 7 is acceptable
  since `n7` is the observed optimum there).
* Q4/Q6 single-card ceiling-12 wins unchanged (the block-01 feature is not lost).
* No purity change anywhere in the `n_max <= 7` band.
* Document the rule (and, regardless of a code fix, add the README/methodology caveat the reporter
  asked for).

## Do not re-derive

* The maintainer's working hypothesis (#35) is that the **climb/drop table is tuned for mainline (low)
  acceptance** and the delivery's improved drafting made the controller over-climb — retune
  `common/speculative-adaptive.h` before reaching for a blanket `n_gpu > 1` / Q8_0 cap.
* The maintainer prefers the **bucketed** controller (no hard resets, running credit bucket); the
  table in `common/speculative-adaptive.h` is the *old* design.  The bucketed port is
  `bucketed-port/port.patch` (apply in `~/llama-cpp-rebase`); measurements in its README.
* **Pinned depth** (`--spec-draft-n-min-adaptive D --spec-draft-n-max D`) is the depth-cost oracle:
  code 7 → 97.0, **10 → 99.5**, 11 → 98.0, 12 → 94.5 t/s.  Every adaptive config is *below* pinned at
  its own mean depth (ramp + wander) — but the residual is per-round wall time, **not** the depth
  changes: measured transition overhead is only **+1.0 ms** each.
* The bucketed **credit's zero-crossing lands on the throughput optimum of every axis** (code 9,
  reasoning/prose/mixed at the floor, recall at the ceiling), so the credit function needs no change;
  the tuning is entirely in the start depth and the two thresholds.
* Three orthogonal probes are needed, not two: **phase switching** (`prompts/code-reasoning-mixed.txt`)
  is *maximal at the floor* (64.3 → 50.0 at depth 12), so a controller that is slow to drop after a
  code phase loses there even when it wins on pure code.
* The table controller parks at 11–12 because `n_drop` is **zeroed on every full accept** (code's
  full-accept rate at 11–12 is 0.21–0.28).  Partial drop-relief and the `+1` climb were both tried
  and do not fix the cell; a depth-weighted credit helps R/P/K but overshoots code to 12.
* The AR is not the cause (reporter A/B).  The `ggml_set_fa_tensor_parallel` hint is prefill-only.
* BF16 vs f16 KV is not the cause (same shape; bf16 ~2 % faster absolute).
* The absolute t/s in this dossier are our local cli footers; the reporter's and the historical server
  record are 10-15 % lower on the same cell.  Compare ratios, not absolutes.
* A separate, untriaged item from the same report: ROCm **7.2.4** breaks purity (clean on 7.14) — a
  toolchain note, not this investigation.
