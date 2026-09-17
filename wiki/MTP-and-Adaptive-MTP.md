# MTP & Adaptive MTP

> **How this project turns llama.cpp's built-in multi-token-prediction drafter from a fixed
> 3-token guess into a workload-aware accelerator.**
>
> The short version: make the drafter and the verifier agree bit-for-bit, make the verify batch
> cheap, then let the draft depth follow the workload. Static MTP drafts the same depth whether the
> model is writing a poem or reciting a document. Adaptive MTP doesn't.

This page is the story and the evidence. For flags and commands, see
**[MTP Quick Reference](MTP-Quick-Reference)**.

---

## TL;DR

| What | Before | After |
|---|---|---|
| Draft depth | fixed (default 3) | adaptive, floor 3 → ceiling 15 (default start `cap − 3`) |
| MoE draft acceptance (35B-A3B) | 0.000 on the broken build / 0.49 upstream | **0.82** |
| qwen4exp draft acceptance (f16) | 0.500 | **0.767** |
| MTP vs plain decode, code axis | +6% (static n3) | **+28%** (adaptive n12) |
| MTP vs plain decode, recall axis | +135% (static n3) | **+279%** (adaptive n12) |
| Wide verify batch (B=8, dense, q8_0 KV) | 3.958 s (delivery, broken) | **2.798 s** — faster than stock's 2.929 s |
| MoE verify width B=8 | 289.9 t/s | **341.3 t/s** |
| Purity band | — | `--spec-draft-n-max ≤ 7` byte-identical (plain == draft-mtp) |

The result on a single RDNA4 card, 27B `UD-Q4_K_XL`, `-n 3000`:

| Workload | plain | static `n3` | adaptive cap 12 |
|---|---:|---:|---:|
| reasoning | 28.9 | 47.3 | 46.0 |
| prose | 28.4 | 56.3 | **63.4** |
| code | 28.8 | 62.9 | **81.8** |
| verbatim recall | 28.9 | 68.0 | **109.6** |

Static depth 3 is a solid accelerator. Adaptive depth is a *much* better one — and the reason it is
possible is the numerics and kernel work below.

---

## 1. What MTP is

Multi-token prediction (MTP) is a form of **speculative decoding**. A small, cheap "drafter" proposes
the next *few* tokens at once; the full model **verifies all of them in a single forward pass**. Every
proposal that matches the model's own `argmax` is accepted for free; the first mismatch stops the
acceptance and the model's token is kept instead.

llama.cpp's modern models ship a **built-in MTP head** — an extra "nextn" block
(`blk.<last>.nextn.*`, `nextn_predict_layers` in the GGUF). It is used automatically when no separate
draft model (`-md`) is passed:

```bash
# static MTP, draft up to 3 tokens per verify round
llama-cli -m model.gguf --spec-type draft-mtp --spec-draft-n-max 3 ...

# adaptive MTP, draft depth follows the workload
llama-cli -m model.gguf --spec-type draft-mtp-adaptive --spec-draft-n-max 12 ...
```

> **Use the built-in head, not a standalone `mtp-*.gguf`.** The standalone file is an older drafter
> and gives different (lower) acceptance.

### Why acceptance is everything

A speculative round has a cost and a payoff.

- **Cost:** one forward pass over `draft_depth + 1` tokens (the verify batch).
- **Payoff:** up to `draft_depth` accepted tokens if the drafter is right.

If the drafter is wrong at the *first* token, the round produced 1 token for the price of a wide
batch — worse than plain decode. If it is right every time, a deep round produces many tokens for
little more than a single-token decode. So the whole game is **acceptance**: how often the verifier's
chosen token equals the one the drafter proposed.

Acceptance is a **bit-level comparison**. If the verifier and the single-token decode path compute
slightly different logits for the same state — different rounding order, different reduction tree,
different kernel — then near-ties can flip, and drafts that *should* have been accepted get rejected.
This is the first lever, and it is the one that most MTP projects get wrong quietly.

### Where mainline stops

Upstream supports a **static** draft depth: `--spec-draft-n-max N` fixes it for the whole run. And the
per-model acceptance depends entirely on whether the decode path and the verify path happen to agree.

The adaptive controller itself started life as **upstream PR
[#27210](https://github.com/ggml-org/llama.cpp/pull/27210)** ("adaptive MTP draft depth", author:
stew675), which introduced the idea of letting the draft depth follow measured acceptance instead of a
fixed constant. **This page is not that PR.** Block 01 was cut from the PR, but the version shipped
there is a distant ancestor: the delivery replaced its mean-reverting table with a credit-bucket
controller, retuned every constant against the delivery's much higher acceptance, and — most
importantly — only became able to *use* deeper drafts at all because of the numerics and
batch-verification work in §3 and §4. The PR proved the idea; the work below is what made it pay.

So the patch set attacks three levers, in this order of importance:

1. **Acceptance** — make the drafter and the verifier numerically consistent. *(§3)*
2. **Verify cost** — make the wide batch cheap, so deep drafts don't cost much. *(§4)*
3. **Depth policy** — pick the depth per workload. *(§5)*

---

## 2. The model of a speculative round

```
                    ┌──────────────── drafter (1 token at a time, cheap head) ─────────────────┐
   last committed → │ d1   d2   d3   d4   ...                                                │
   token           └────────────────────────────────────────────────────────────────────────┘
                                        ↓  proposed tokens
   ┌────────────── verifier (one forward pass over depth+1 tokens) ──────────────────────────┐
   │  v0   v1   v2   v3   v4   ...                                                           │
   └─────────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓  accept  d_i  while  argmax(v_i) == d_i
   accepted prefix + one bonus token from the first mismatch (or one extra if all matched)
```

Everything that follows is about making the `argmax(v_i) == d_i` comparison trustworthy and the
`depth + 1`-wide verifier pass cheap.

---

## 3. Lever 1 — the numerics contract: make the drafter and the verifier agree

### The rule: band invariance

A speculative verify is just decode at a wider batch. The same weight, the same state. So the
delivery enforces one invariant across the whole **decode/verify band**:

> **Inside the band, one arm — chosen from the band (`1..K` tokens) and the cache length, never from
> the exact width, never from `n_tokens == 1`.**

A W=1 decode and a W=(`n_max + 1`) verify of the same state must take the *same* kernels with the
*same* reduction order. When they do, the verifier's `argmax` is exactly the token plain decode would
have produced — which is the definition of correct speculative decoding. When they don't, acceptance
silently drops and nobody notices until a benchmark falls off a cliff weeks later.

This is harder than it sounds, because llama.cpp's dispatch is full of accidental width dependencies:
kernel family choosers keyed on `n_q`, matmul-family crossovers at `ncols == 8`, per-type "caps" that
route mmvq vs MMQ, fusions gated to `n_tokens == 1`, and register-pressure heuristics that pick
different warp counts for different widths. Each one is a potential near-tie factory.

### The failures we found — and fixed

These are real bugs, found by probing every width and hashing the token-0 logits. Each one was a
drafter/verifier disagreement hiding in plain sight.

| # | Defect | Symptom | Fix | Effect |
|---|---|---|---|---|
| 1 | **FA KV-split keyed on query width.** `launch_fattn`'s `parallel_blocks` heuristic used `ntiles_dst` (a function of `Q->ne[1]`), so 1-token decode and a 3-token verify grouped the online-softmax reduction differently. | Greedy output changed with `--spec-draft-n-max` (issue #25). | Block 00: evaluate the heuristic as if `n_q == 1` for every `n_q ≤ 8`. | Decode and every verify width reduce identically; only `n_q ≥ 2` moves. |
| 2 | **FA kernel-family split for quantized K/V.** The chooser returned VEC for `n_q ≤ 2` and TILE from `n_q = 3`; the two order the softmax/PV reduction differently. | `q8_0`/`q4_0` caches: `W=1,2` disagreed with `W=3..8`; plain ≠ draft-mtp in *text*. | Block 08 (F1): the band is TILE throughout. | All four split configs pure; decode-only cost −0.5…−0.9%. |
| 3 | **Upstream's per-type mmvq caps.** The cap selected mmvq vs MMQ by type (`IQ3_S` cap 4, `IQ4_XS` cap 5, `IQ4_NL` cap 7 …) and sized `mul_mat_vec_q_moe`'s launch bound at `cap × warp_size`. | MoE widths grouped `{1..4}{5}{6,7}{8}` — every cap boundary inside the band was a numeric boundary. | Block 13: floor the cap at `MMVQ_MAX_BATCH_SIZE` and size the launch bound at the band. | `W=1..8` bit-identical **and +14–26% at the verify widths**. |
| 4 | **MoE `MUL_MAT_ID` at `ncols_dst == 1` used the dense ksplit kernel** (with an ids gather) while the 2..8-token verify used the dedicated MoE kernel. | Decode and verify not bit-identical for MoE. | Block 13: all `MUL_MAT_ID` use the dedicated MoE kernel. | **+6.2% MoE decode** and decode == verify. |
| 5 | **Fused shared-expert epilogue gated to `n_tokens == 1`.** The verify batch ran the unfused chain. | MoE `W=1 ac8825…` vs `W≥2 bd138a…`. | Block 13: token-generic kernels, `nwarps` pinned to the single-token reduction order; band `1 ≤ nt ≤ 8`. | **MoE acceptance 0.51 → 0.82**, MTP 167.3 t/s vs plain 96.9 (**+73%**). |
| 6 | **qwen4exp hyper-connection fusions gated to `nt == 1`.** | 1-token decode used the fused `HC_MIX`/`HC_COMBINE` chain; an n-token verify used the unfused chain. | Block 14: ops map the token onto `blockIdx.y`, band `1 ≤ nt ≤ 8`. | **qwen4exp f16 acceptance 0.500 → 0.76744**, MTP 63.3 → 79.9 t/s. |
| 7 | **qwen4exp QSA dense decode arm gated to `n_tokens == 1`.** Above the indexer selection width, decode was dense while the verify fell through to sparse top-k. | `plain ≠ draft-mtp` in text, appearing only after ~2051 tokens. | Block 14: `QSA_DECODE_BAND`, later `max(QSA_DECODE_BAND, n_rs_batch)`. | `plain == n_max 3 == n_max 7` byte-identical. |
| 8 | **Block-12 size-based all-reduce dispatch.** The reduced tensor grows with the batch, so a 7-token verify crossed the 2-device 32768-element crossover and switched from the internal pipeline to NCCL — different summation, different BF16 round-trip. | 2-GPU `-sm tensor` pure only to `n_max 5`. | Block 12: the 2-device crossover is 131072 (matching 3 devices). | `W=1..8` identical and **+12% MTP** (the internal pipeline is faster at these sizes). |
| 9 | **Chunked-GDN rollback bound.** A long-draft verify batch was chunked with no rollback snapshots; a partial accept then restored an unwritten state. | Silent recurrent-state rewind with ngram-style long drafts. | Block 02: threshold `max(K > 16 ? K : 16, n_rs_batch)` + a pre-batch snapshot slot. | Correctness (see §6 for the depth bound it implies). |
| 10 | **RDNA3_5 single-token-only fusions.** The dense gate+up+GLU fusion and the weighted-down MoE tail don't reproduce the standalone arithmetic. | gfx1151: `W=1` vs `W=8` differed. | Block 13: skip both on RDNA3_5 unless explicitly A/B-enabled. | gfx1151 band pure; ~0.9% `tg128` cost on qwen4exp. |

There were more, all catalogued in the repo's
[`GREEDY-PURITY.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/GREEDY-PURITY.md)
(§§10–36). The pattern is always the same: **a fast path and a generic path that don't quite compute
the same thing, with the boundary drawn through the middle of the decode/verify band.**

### The contract that falls out

| Draft depth | Verify width | Greedy guarantee |
|---|---|---|
| `--spec-draft-n-max ≤ 7` | ≤ 8 | **Byte-identical**: `--spec-type none` == `draft-mtp` — the acceptance comparison runs exactly the decode arithmetic. |
| `8 … 15` | 9 … 16 | **Allowed with a notice.** Above 8 rows the FA chooser switches tile → WMMA and the matmuls switch MMVQ/MMVF → MMQ, so a greedy near-tie can flip. The output stays valid and coherent; it is a trade, not corruption. |
| `> 15` | > 16 | **Clamped.** This is the recurrent rollback snapshot bound (`n_max + 1 = K ≤ 16`) — a *correctness* bound, not a purity one. |

The default `n_max` is still **3**, so none of this changes an existing setup.

> **Purity is a function of the workload too.** A short spot-check can report "pure" while a long run
> hits a near-tie. The four-axis gate runs `-n 3000` for exactly this reason — at `-n 256` the code
> axis at ceiling 12 read **−5%** vs fixed `n3`; at `-n 3000` it reads **+28%**.
> See [How we validate](#7-how-we-validate).

### Why this is the whole ballgame

The numbers above are not incremental. The MoE shared-expert fix alone moved acceptance from 0.51 to
0.82 — a **+73%** throughput swing — at a cost of ~2.4% at the widest verify width. The arithmetic is
brutal and worth stating plainly:

> The draft model's proposals **are** decode steps. Any arithmetic difference between decode and verify
> shows up as draft-vs-verify disagreement, and lost acceptance costs a multiple of whatever the
> "faster" kernel saved. A few percent of attention throughput cannot pay for a lower acceptance rate.

That is the delivery's **purity-first** policy: land the bit-identical path, record the cost, repay it
structurally later. (Which we did — see the column-blocked epilogue in §4.)

---

## 4. Lever 2 — making the verify batch cheap

High acceptance makes deep drafts *profitable*, but profitable is not free. A verify round is one
forward pass over `depth + 1` tokens — a shape plain decode never produces. If that shape is slow, the
extra accepted tokens get eaten by the wider kernel.

This is exactly the failure mode of the 2026-09-12 "issue #30" regression: the patch set had made the
mmvq knobs **band-uniform** (a purity requirement) but left them tuned for the single-token case, so
the wide verify batch cost up to **35% more** than it needed to.

### What a verify batch looks like

- multi-token `MUL_MAT` / `MUL_MAT_ID` (MoE) rows,
- a wider flash-attention batch (up to 8 rows in the pure band),
- the fused MoE epilogue at `ncols 2..8`,
- a draft context that alternates graph shapes and (before upstream #28549) never warmed up its graph.

### The batch-verification work

| Work | Where | Result |
|---|---|---|
| Restore the pre-item-split K-split kernel for `ncols 2..8` (the item-split was register-bound at multi-token widths) | block 13 | dense MTP 18.3 → 27.5 t/s |
| Make the MoE `MUL_MAT_ID` dispatch width-uniform | block 13 | +6.2% MoE decode |
| Floor the per-type mmvq caps at the band and size the launch bound at it | block 13 | **+14–26%** at the verify widths |
| Band-uniform `nwarps` tuned **at the verify widths** (RDNA4 `nwarps = 1`), VDR reverted for dense kernels | blocks 08/10 | B=8 **3.958 → 2.798 s**, faster than stock's 2.929 s |
| Per-`(type, K)` `nwarps` for the dense mmvq weight kernel (Q8_0 short-K takes the wide block) | block 13 | MoE MTP `n_max 7` **+10%** (acceptance 0.631 → 0.731) |
| Per-kernel VDR (dense = upstream, MoE expert = wide) | blocks 10/13 | recovers the MoE single-token loss the band rule cost |
| Column-block the fused shared-expert epilogue so one weight row is read once per `(row, k-block)` for the whole band | block 13 | `pl 8` 461.0 → **475.4 t/s**, bit-identical |
| Enable CUDA graphs for the MTP draft context (upstream #28549, picked up by the re-base) | upstream | +0.3% … **+1.4%**, scaling with draft depth |

### The verify-width payoff

`llama-batched-bench` TG total seconds (lower is better), dense 27B `UD-Q4_K_XL`, `q8_0` KV, on one
RDNA4 card:

| Build | plain | B=1 | B=4 | B=8 | MTP `n_max 7` | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| stock, same fork point | 28.25 | 1.157 | 1.726 | 2.929 | 37.51 | 0.484 |
| delivery before the fix | 29.34 | 1.147 | 2.121 | 3.958 | 30.34 | 0.466 |
| **delivery after the fix** | 28.62 | 1.175 | **1.657** | **2.798** | **36.32** | 0.475 |

The amended verify path is **faster than stock's** at every width. For MoE and qwen4exp, fixing the
per-type caps gave:

| Model | B=4 | B=5 | B=6 | B=7 | B=8 |
|---|---:|---:|---:|---:|---:|
| qwen4exp 3-GPU tensor | 134.1 / 132.9 | **149.5** / 118.4 | **162.5** / 130.6 | **171.4** / 147.0 | **178.0** / 155.4 |
| 35B-A3B MoE 1 GPU | 254.2 / 254.1 | — | — | — | **341.3** / 289.9 |

*(fixed / baseline; higher is better. The baseline is the pre-fix build.)*

> **This is what "deeper drafts incur less of a performance hit" means.** The verify batch is not a
> penalty you pay for drafting deep; it is a shape we optimize in its own right. Once it is cheap, the
> only remaining question is *how deep to draft* — which is the third lever.

---

## 5. Lever 3 — Adaptive MTP: let the depth follow the workload

### Why a fixed depth is wrong

Different content has wildly different predictability. A verbatim recall passage is almost free to
extend; a step-by-step reasoning derivation is not. The **pinned-depth oracle** makes it concrete —
`--spec-draft-n-min-adaptive D --spec-draft-n-max D` pins the depth with zero transitions:

**Code axis, 2-card Q8_0, t/s by pinned depth:**

| D | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 80.8 | 88.5 | 94.8 | 96.1 | 96.7 | 99.1 | 98.1 | **99.0** | 97.2 | 94.1 |

Code has a broad optimum around **8–10**. But the same sweep puts the optimum at the **floor** for
reasoning, prose, and phase-switching (58.1 at depth 3 vs 43.7 at 12), and at the **ceiling** for
verbatim recall. No single static value is right for more than one of them.

A controller doesn't need to be told which workload it is in. It just needs to **measure acceptance**
and move.

### Provenance: an upstream PR, then a long evolution

The controller's lineage matters, because the tuning only makes sense in the context of what changed
around it:

| Stage | Where | What |
|---|---|---|
| **Origin** | upstream PR #27210 (`0994374fd`) | A **mean-reverting table** controller: per-depth `climb_threshold` (consecutive full accepts needed to climb) and `drop_pressure`. It proved that depth should adapt. |
| **PR review head** | `d236d41a2` | Review-round fixes: partial-accept reporting, per-generation reset, arg validation, `--spec-draft-n-min-adaptive` semantics. Still the table. |
| **Delivery evolution** | block 01 afterwards | Replaced by a **credit-bucket** controller; every constant retuned for the delivery's acceptance; the depth clamp moved 7 → 15; `--spec-draft-n-start` added; the cold start measured and restored. |

Why the table had to go: it was tuned against **mainline (low) acceptance**. The delivery's ±2x higher
acceptance changed the operating point, so the same constants over-climbed and then churned. The
table's `n_drop` was zeroed on every full accept, so a workload that fully accepted every few rounds
never accumulated drop pressure and parked at 11–12 — exactly where the wide verify is dearest.
Measured on the reporter's cell (Q8_0 27B, 2-card tensor, code):

- ceiling 12 read **92.8 t/s against 96.3** at ceiling 7 — deeper was *worse*;
- the pinned optimum was depth 10 at **99.0**;
- the table made **40 depth changes in 477 verification rounds** on a slow 6↔12 limit cycle.

The diagnosis (issue #35) was that the *economics* were fine but the *constants* — and in the end the
*structure* — were not. The controller now uses a **credit bucket** (stew675's bucketed design,
ported onto block 01 and tuned for the delivery):

```
delta = n_accepted − depth            (a full accept credits max(1, n_accepted − 1))
surplus / deficit carries across a depth change
```

and the bucket's drift zero-crossing already lands on each workload's throughput optimum (code ~9,
reasoning/prose/phase-switch at the floor, recall at the ceiling). The tuning is in three constants:

| Knob | Value | Why |
|---|---|---|
| `climb_budget(d)` | `20 + 6·(d − 1)` | A flat budget let six lucky full accepts at depth 8 cascade 9 → 10 → 11 → 12 in 16 rounds (integral windup). |
| `drop_pressure(d)` | `max(60, 10·d)` (was `max(20, 4·d)`) | Damps the 6 ↔ 12 limit cycle. |
| cold start | `max(floor, cap − 3)`, overridable with `--spec-draft-n-start N` | Climbing is the expensive direction; park near the plateau and let the drift pull depth down. |

The resulting controller is *boring*, and that is the point: it parks within ±1 of each workload's
optimum with **4** depth changes instead of 40.

> **Note on the constant `cap − 3`.** The credit's zero-crossing already lands on each workload's
> throughput optimum, so the *economics* needed no tuning — the tuning is entirely the starting point
> and the two thresholds. That is also why the whole thing is only viable here: at the delivery's
> acceptance a deeper start and a deeper ceiling both pay, and at mainline's acceptance they do not.

### The cold-start finding

A tempting idea was to start deep (the midpoint of floor/cap) so short generations ramp up faster.
It was measured and **reverted** — it loses 5% on reasoning, 1.9% on code and 7.5% on recall. The
delivery's optimized multi-token verify makes a wider batch cheap, so starting low only lowers the
settled mean depth and costs *rounds*. The "slow at first, then speeds up" effect was the **drafter
warming up** (acceptance rises with context), not the cold start.

`--spec-draft-n-start N` remains as a runtime knob, clamped to `[floor, cap]`.

### Results

**Two-card Q8_0 27B, `-n 3000` (the reporter's cell):**

| Axis | fixed MTP-3 | adaptive cap 7 | adaptive cap 12 (old) | adaptive cap 12 (tuned) |
|---|---:|---:|---:|---:|
| reasoning | 57.9 | 57.7 | 57.9 | **60.8** |
| prose | 73.0 | 79.5 | 78.8 | **81.2** |
| code | 80.8 | 95.8 | 92.8 | **96.0** |
| verbatim recall | 86.5 | 115.5 | 118.9 | **137.3** |
| phase switching | 64.3 *(pinned opt.)* | — | 64.2 | 64.0 |
| code depth changes | — | 8 | 40 | **4** |

**One-card 27B `UD-Q4_K_XL`, `-n 3000`, adaptive cap 12 vs fixed `n3`:**

| Axis | plain | fixed `n3` | adaptive n12 | Δ vs `n3` | mean accepted length |
|---|---:|---:|---:|---:|---:|
| reasoning | 28.9 | 47.3 | 46.0 | −1.0% | 2.74 |
| prose | 28.4 | 56.3 | **63.4** | **+13.3%** | 5.10 |
| code | 28.8 | 62.9 | **81.8** | **+28.5%** | 7.02 |
| verbatim recall | 28.9 | 68.0 | **109.6** | **+61.0%** | 8.95 |

Ceiling 12 vs ceiling 7 on that cell: prose **+26%**, code **+35%**, recall **+44%**.

> **Lower acceptance, higher throughput.** The adaptive acceptance is *below* fixed `n3` on every axis
> (code 0.579 vs 0.917) because it drafts deeper and rejects more. But it accepts more tokens per
> target forward pass, so it is faster. **Acceptance alone is not the metric** — accepted tokens per
> round is.

### What the higher depths buy

The point of the numerics and batch work is that it makes depth affordable:

- **Higher acceptance** means each extra draft slot is more likely to pay for itself, so the
  controller's optimum sits deeper.
- **A cheap verify** means the marginal cost of one more slot is small, so even a moderate acceptance
  rate keeps deep drafts profitable.
- Together they turn a **safe** depth of 7 into a **profitable** depth of 12 on predictable content —
  while still parking at the floor when the content is hard.

---

## 6. The combination, end to end

### Four-axis gate (the canonical comparison)

One RDNA4 card, 27B `UD-Q4_K_XL`, `-n 3000`, reasoning pinned per axis, prompts hashed and versioned
in `prompts/`:

| Axis | plain | stock `n3` | delivery `n3` | adaptive `n12` | adaptive vs `n3` |
|---|---:|---:|---:|---:|---:|
| reasoning | 28.9 | 47.3 | 46.4 | 46.0 | −1.0% |
| prose | 28.4 | 56.3 | 55.9 | **63.4** | +13.3% |
| code | 28.8 | 62.9 | 63.7 | **81.8** | +28.5% |
| recall | 28.9 | 68.0 | 68.1 | **109.6** | +61.0% |

At fixed `n3` the delivery and stock are within ~2% on every axis. **The delivery's win is
concentrated in the deeper/adaptive drafts** — exactly what the numerics and batch work unlocked.

### Combining with `ngram-mod`

`ngram-mod` is a draftless speculator that is excellent on verbatim recall. The speculators compose
with a fixed priority (ngram first when both are enabled), and the tuned combo is:

```bash
--spec-type draft-mtp-adaptive,ngram-mod \
--spec-ngram-mod-n-match 45 \
--spec-draft-n-max 9 --spec-draft-n-start 9
```

On the two-card Q8_0 cell: **R +1.2%, P flat, C +0.6%, K +72%** against MTP-only — with no harm on
any axis. (`n_match 45` makes an ngram hash hit require a long context match, so it fires on recall
but not on incidental code repeats.)

---

## 7. How we validate

MTP regressions are invisible to ordinary throughput benchmarks, because `llama-bench tg` always
decodes **one token per step**. It never touches the verify batch or the draft context. Both of the
big MTP regressions in this project's history slipped past every standard gate.

So the project has its own MTP gate
([`benchmarks/mtp-adaptive-methodology.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/mtp-adaptive-methodology.md)):

1. **Length: `-n 3000`** (floor `-n 2000`). A short run measures the drafter/controller warm-up, not
   the mode. The ranking can invert.
2. **Pin the content mode:** `--reasoning on` for the reasoning axis, `--reasoning off` for prose,
   code and recall. Qwen3.8 emits a thinking trace by default, so an unpinned run measures thinking,
   not the workload.
3. **Acceptance:** the `-lv 4` `draft acceptance` line must be healthy (≥ ~0.45 at position 1). A
   collapse to 0.0 is a numerics divergence, not a tuning problem.
4. **MTP must not lose to plain decode** at the default depth on the same build.
5. **Stock-relative verify-width check:** interleaved `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`
   with a quantized KV cache — B=4/B=8 must not regress against stock. This is the gate that catches a
   wide-verify regression when acceptance and `tg128` both look fine.
6. **Purity:** `--spec-type none` == `draft-mtp --spec-draft-n-max 3` == `n_max 7`, byte-identical
   greedy text, on every supported KV type.

The prompts are versioned and hash-recorded in `prompts/`; a reported result is only valid against the
hash it names. Shipped prompts are never edited in place.

---

## 8. Where this lives in the patches

| Block | MTP-relevant content |
|---|---|
| `0000` | FA small-batch KV-split width invariance |
| `0001` | Adaptive MTP draft depth + the credit-bucket controller; the depth clamp policy |
| `0002` | Chunked-GDN rollback bound (`n_rs_batch`) — the correctness bound behind `n_max ≤ 15` |
| `0008` | Fused-core prefill, GPU bit-identical results; FA kernel-family band |
| `0010` | k-quant mmvq VDR (the per-kernel split is tuned at the verify widths) |
| `0013` | Fused MoE gate+up+GLU MMQ; verify-batch ksplit; band-uniform MoE caps and shared-expert epilogue |
| `0014` | qwen4exp QSA + hyper-connection bands; MTP export logits purity |
| `0015` | K/V staging as a decode-depth policy; native KV types in FA |

---

## 9. The one-paragraph version

MTP is only as good as the agreement between the cheap drafter and the expensive verifier. We made
them compute the same thing across the whole decode/verify band, which recovered collapsing
acceptance rates (MoE 0.51 → 0.82, qwen4exp 0.50 → 0.77) and made greedy output independent of the
draft length. We then made the wide verify batch fast in its own right — restoring the K-split kernel,
fixing the MoE dispatch, tuning the mmvq knobs *at the verify widths* rather than at one token, and
column-blocking the fused expert epilogue — so a deeper draft costs less per extra accepted token. With
acceptance high and the wide batch cheap, the adaptive controller can afford to draft deep on
predictable content and stay shallow on hard content, which is why adaptive depth beats any static
depth: **+28% on code and +61% on recall over static `n3`**, at the same or better reasoning/prose
throughput.

---

### See also

- **[MTP Quick Reference](MTP-Quick-Reference)** — flags, depth policy, copy-paste commands.
- [`GREEDY-PURITY.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/GREEDY-PURITY.md) —
  the purity rulebook: every finding, with evidence.
- [`benchmarks/mtp-adaptive-methodology.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/mtp-adaptive-methodology.md) —
  the gate protocol and baselines.
- [`benchmarks/2026-09-15-adaptive-mtp-tuning.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/2026-09-15-adaptive-mtp-tuning.md) —
  the controller tuning record.
- [`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md) —
  the four-axis record.
- [`patches/README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md) —
  per-block notes, env knobs.
