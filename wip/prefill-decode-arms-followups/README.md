# prefill-decode-arms follow-ups — what block 17 did NOT ship, and the next round of work

**Status: OPEN (2026-11-22).**  Block 17 (the two-arm server) shipped in `v18-919794500-r346`, **default OFF / opt-in
only**.  None of the work below is in `patches/`, none of it is applied by `scripts/apply-all.sh`, and none of it is
covered by a release gate until an item is promoted into a block **with the maintainer's approval**.

**Where the delivered feature lives:** `patches/README.md` → block 17, `ENVIRONMENT.md` §23,
`INVARIANTS.md` r346, `GREEDY-PURITY.md` §105, `MANIFESTS.md`'s block-17 row, `WORKLOG.md` 261122.027.
**The campaign that produced it:** [`archive/work/prefill-decode-arms/`](../../archive/work/prefill-decode-arms/README.md)
(the record, the 18 pass reports, the instruments, the measured losers).

**This document is the handover for a cold session.**  Every item states what exists today, what is missing, how it
would be designed, and what "done" means, so nobody has to re-derive the state from the campaign record.

---

## 0. What shipped, in one paragraph, so the items below make sense

`llama-server --prefill-decode-arms` runs **two `llama_context`s on one model**: a prefill arm (the main loop,
`ctx_tgt`) and a decode arm (`ctx_dec`, its own thread) that **co-own one KV cache**.  A slot crosses between the arms
only at a **request boundary**, and a hybrid model's recurrent state follows it device to device.  It buys a decoder's
interactivity for a prefill cost: measured **47-49 ms ITL p50 / 64-78 ms p95** through a 2-5 s prefill, where the single
context stalls 300-880 ms, for **-21…-24 %** prefill.  It is off by default because that is a trade an operator makes
for their own shape.

Everything below is either (a) a capability the pair refuses today, (b) a cost the pair pays and should not, (c) a
validation gap, or (d) tuning.  **The tuning item (W6) is the most exploratory and the only one that may end in a
documented "no"; it is also the one the maintainer asked for by name.**

---

## 1. Item index

| id | item | class | `TODO` | size | depends on |
|---|---|---|---|---|---|
| **W1** | speculative decoding with the arms (**MTP first**) | capability | #364 | large | W2 helps, not blocks |
| **W2** | a **meta-aware device copy** for the recurrent handover | cost | #365 | medium | **DONE 2026-11-22 (r347)**: the Meta buffer's per-child `cpy_tensor_n` — **1.4-1.9 ms** against 72-77 ms serialised, `LLAMA_RS_VERIFY_TAKE=1` 0 of 118 039 420 bytes differ, and the two arms' per-take dumps are byte-identical |
| **W3** | derive the **decode context's reserve** from the decode width | cost | #366 | small | **DONE 2026-11-22**: floor 3584 + a width-keyed tier (0 / 1536 MiB), which prints 5120 on the host tier and needs no knob |
| **W4** | **GLM5-Next** on the pair | validation | #367 | small (needs a checkpoint) | none |
| **W5** | a **recurrent-only** memory handover | capability | #368 | medium | needs a Mamba-class model |
| **W6** | the **prefill/decode speed balance**: the prefill arm SKIPS turns while a decoder is active | tuning | #371 | medium | none; measure first (§7.1) |
| **W7** | the pair's **slot erase and cell accounting** | correctness (narrow) | #370 | small | none |
| **W8** | **multi-slot prefill batching** ("layer-major" across active prefill slots) | tuning / architecture | #372 | medium | read the premise check first (§11.1) |
| **W9** | **P0: the pair mutates FIVE shared structures from TWO threads with no lock** -- the KV cell map (`init_batch`'s snapshot + `apply()`), the **Meta split-state cache** (per buffer, so shared through the KV cache's buffer: the released build's SIGSEGV), the **async-input ring's** device-wide sync (illegal during a peer's graph capture), the **VMM pool's peer grant**, and the **synchronous host paths'** default-stream coupling.  **REPRODUCED 2026-11-22** in ~30 s (`repro-373/`); **SHIPPED in the r347 block-17 amendment** (`WORKLOG.md` 261122.036) with the owed gate arm (`gate-arms-soak.sh boundary`) | **correctness (P0)** | closed | memory lock = small; Meta + ring + pool + host streams = done | **done**: `repro-373/` (3 slots, two prefills + a decode) |
| **G1** | gate debt: three gates still anchored to the retired Flash-Next model | tooling | #369 | small | none |

Ordering suggestion (after r347 closed **W2**, **W3** and **W9**): **G1, W7** (small, close out loose ends) →
**W6** (the one the maintainer wants measured) → **W1** (the big one) with **W4** whenever a checkpoint appears and
**W5** parked until a model exists.

---

## 2. W1 (TODO #364) — speculative decoding with the arms, MTP first

### What exists

The pair **refuses** a speculator.  That is deliberate and measured, not an oversight: with
`--prefill-decode-arms-mode auto` the server falls back to the single-context server *and runs your speculator*, and
with the explicit `--prefill-decode-arms` it refuses to start with a message naming the reason
(`--prefill-decode-arms: a speculative decoder is not supported (it owns the draft ...)`).  Pass 18 of the campaign
("does MTP work now?") re-measured exactly this: shipped MTP is unchanged and fine, MTP *with the arms* is refused.

### What is missing

1. **The verify batch on the decode arm.**  A speculator turns a decode step into a multi-row verify (the target
   decodes `n_draft+1` rows).  Today the decode arm's batch is **one row per active slot, never a prompt row** by
   construction, because its `n_batch = n_ubatch = n_parallel`.  A verify batch needs `n_draft+1` rows per slot, so
   the decode context's width and the per-step batch composition both change, and the width is also the graph reserve
   term (see W3: the flat reserve was sized for a 1-row decode graph, and a verify width is bigger).
2. **The draft context on a shared KV.**  MTP runs the target's nextn layer in a second context (`ctx_other` in
   `common.cpp`).  The pair already creates a second context through the same gate, so three contexts on one model
   become possible; the co-owner allocator (`struct llama_kv_alloc`, `llama_memory_alloc_sync()`) currently reasons
   about **one** co-owner pair, so it needs to stay consistent with a third participant (or the draft keeps its own
   private cache, which is the simpler first cut).
3. **The draft ring and the shared cursor.**  Block 14's `--spec-draft-kv-window` is a per-sequence ring.  On the pair
   the attention cells are shared and a handover moves the recurrent half, so the ring's cells must be unparked and
   the cursor handshake must cover them (block 14's r200 rule: the window is granted per sequence; a second slot's
   batch must not invalidate the first's ring).
4. **The guards.**  `-ib`/`LLAMA_SERVER_PREFILL_DUTY` are reported deprecated and inert on this path; a speculator
   changes the decode step's own width, so the "deprecated" message must not hide a real interaction.  Re-derive
   which of the two (if either) still has meaning under a drafter.

### Design sketch

Start with the smallest honest step, which is **MTP with the draft on its own private KV** and the decode arm at
`n_batch = n_ubatch = n_parallel * (n_draft + 1)`.  That keeps the co-owner allocator reasoning about exactly one pair
and makes the only new question "which rows does the decode arm verify".  The draft window support is a second step.

### Acceptance ("done")

* `scripts/gate-spec-purity.sh` on the **pair** (not just the single context): `none` == `draft-mtp` byte-identically on
  both cache types at 3-device `-sm tensor`, plus the width probe.
* The MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) **on the pair**: text identical to the plain
  arm and acceptance not collapsed; the recorded reference is `0.34306 / 0.96652 / 0.63593 / 0.73478` with text
  `218515b89b31 / a87c4318b649 / 27d74f92ec5c / 6fa43035b8d9` (f16) at `-n 3000`.
* The pair's text still equals the single-context server's on a class-matched shape (`GREEDY-PURITY.md` §105).
* `scripts/gate-prefill-fairness.sh --arms-band` and `gate-prefill-duty.sh --inert-band` still green.
* The refusal message is replaced by a supported path, and `ENVIRONMENT.md` §23 says what a speculator costs on the
  pair (a wider graph, hence a bigger reserve).

### Risks

The verify width is the numerics carrier the whole purity rulebook is about (§36, §63, §96, §105): a co-batched pair
already is not a text oracle, and a verify batch on the decode arm makes the *pair* a co-batched shape.  Expect the
first cut to be text-unstable across warm reps and plan for the `TWO_SLOT`-style sequential arm as the oracle.

---

## 3. W2 (TODO #365) — a meta-aware device copy for the handover

**DONE 2026-11-22 (r347, `WORKLOG.md` 261122.036), shipped in the block-17 amendment.**  The Meta buffer exposes a
per-child, split-aware **`cpy_tensor_n`**, so `llama_memory_recurrent::can_take_batched()` stops refusing and the
take is a device copy: **1.4-1.9 ms** (3.0-3.3 ms with the verify capture; ~5.5 ms on the first take in a process)
against **72-77 ms** serialised, `LLAMA_RS_VERIFY_TAKE=1` **0 of 118 039 420 bytes differ**, and the same turn
driven once per arm (`LLAMA_RS_FORCE_SERIALISED=1`, serially, greedy) gives **byte-identical** per-take destination
dumps and identical text.  What made it work: the state of Qwen3.8-Flash-Next is **one segment repeated 3 or 5
times** (interleaved inside the row), and both sides of a copy are children of the SAME layout, so a child's covered
rows are one contiguous range on each side — a planner that only knew the single-repeat layout left 72 of 73
tensors on the host path (measured with `GGML_META_DEBUG_CPY=1`).  The design record, kept because the next copy
hook will face the same choices:

### What exists

`llama_memory_recurrent::seq_state_take()` performs the recurrent handover as **one batched device-to-device copy**
(measured **0.45-0.66 ms** for 52.7 MiB).  Where a single device owns the rows that is what runs.  On a **`-sm tensor`
split of a hybrid model** the backend exposes no device copy hook for that buffer, so the code takes the
**serialised fallback**: `112.57 MiB` through the host, **~74.6 ms per request boundary**, and it lands in TTFT.
`LLAMA_RS_FORCE_SERIALISED=1` forces that arm deliberately (the A/B control), and `LLAMA_RS_VERIFY_TAKE=1` re-reads and
compares the taken rows (the instrument to prove a new copy path is faithful).

### What is missing

A **meta/S2D copy path for `seq_state_take()`**: the meta backend's per-device children each hold a slice of the state
buffer, so the take has to iterate children and issue one copy per child (the shape block 12's segmented async tail
already uses for the tensor split).  The reason it is not just "call the same copy" is that the take's source and
destination live in different buffers **and different devices**, with the split boundary in the middle of a row.

### Acceptance ("done")

* A `-sm tensor` hybrid boundary (27B Q8_0, 3 GPUs, the maintainer's shape) moves the state in **well under 5 ms**,
  measured with the existing timings, against the 74.6 ms today.
* `LLAMA_RS_VERIFY_TAKE=1` reports every tensor identical to the serialised arm's result (byte for byte).
* Text identical to the serialised arm on the same shape, and to the single context on a class-matched shape.
* `scripts/gate-ckpt-reuse.sh` and the campaign's reuse oracle still PASS (the reverse handover uses the same take).

### Risks

A wrong state is **fluent but wrong text** (the campaign's §89/§44 carrier class), so the faithful-copy instrument is
not optional.  Do not "optimise" by copying only the rows the destination claims: the destination's metadata is what
decides which rows matter, and it is set by the same pass.

---

## 4. W3 (TODO #366) — derive the decode context's declared reserve from the decode width

**DONE 2026-11-22** (`WORKLOG.md` 261122.035).  The reserve the pair declares is now the flat **3584 MiB floor plus a
WIDTH-keyed tier** (`0` at `-ub <= 1024`, **1536 MiB** at `-ub >= 2048`, +256 per doubling above that) for the *second*
late narrow view the pair creates -- the prefill arm's **post-prefill re-reserve**, which the flat value did not cover.
Measured (Flash-Next, `-ncmoe 48 -c 131072`, arena 13.3 GiB, 3 slots, 71k + 23k + 800, two rounds per point):
**3584 fails, 4096 fails, 5120 clean** -- and 5120 is what the derivation prints (`aux=5.00 GiB`), so the host tier
needs no knob.  `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` still wins outright.  The ordering hazard the pre-read warns about
(the slab's reserve is whatever the LAST preflight carried) is unchanged and still wants fixing when the draft path is
next touched.  The original item, kept for the design record:

### What exists

Block 17 declares the decode context's graph reserve **before** the arena claims the space, through
`MOE_EXPERT_CACHE_AUX_RESERVE_MIB`, at a **flat 3584 MiB** (the same flat number an embedded MTP head gets).  Without
that declaration a host-expert tier refused the second context (`SLAB_INIT ... aux=0`, then `the decode arm's step
failed`).  It is never charged when a speculator is configured, because the pair refuses one.

The measured need is far smaller and scales with the **decode width**: **1792 MiB** on Flash-Next IQ3_S, **9.52 MiB**
on the 4B.  A flat reserve is the safe direction (it over-declares, so the context is never refused), but on a shape
where the expert arena is the thing being sized it costs arena capacity for nothing.

### What is missing

The draft path's own derivation, applied to the decode context: estimate from `n_ubatch`/`n_batch`, the per-layer
activation terms and the model's own layer shapes, with the flat 3584 MiB as the **floor** (never smaller), and
`MOE_EXPERT_CACHE_AUX_RESERVE_MIB` still winning when set explicitly.

**Read `wip/slab-draft-aux-reserve/` before touching this.**  That campaign found that
`ggml_cuda_slab_set_aux_reserve()` is called from *every* preflight that returns "cache on", so the slab's reserve is
whatever the LAST call carried, and a draft model's own preflight can overwrite the target's derived value with the
flat 3584 MiB (a measured 2.4 GiB arena loss).  A derived decode reserve inherits that ordering hazard exactly: two
contexts, two preflights, one slab field.

### Acceptance ("done")

* `SLAB_INIT ... aux=` is non-zero and equal to the derived value on a host tier, and the pair still serves
  (`scripts/gate-arms-soak.sh pool` and `gate-prefill-server-slots.sh`-class arms).
* The derived value is **>= the measured need** on every validated shape (the dangerous direction is under-declaring).
* The arena gains the difference on a budgeted shape, measured, and the hit census matches the single context's class.
* Both kill switches still behave: explicit `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` wins, manual
  `GGML_CUDA_SLAB_RESERVE_MIB` wins over both.

---

## 5. W4 (TODO #367) — GLM5-Next on the pair

`GLM5-Next` uses the **same `llama_memory_hybrid_idx`** memory as Qwen3.8-Flash-Next: attention KV, a recurrent state
and a lightning indexer KV.  That module is exactly where block 17's P0 lived (it never received the shared view, so
the decode arm ran on an empty attention **and** indexer cache while the server printed "KV shared").  The fix is
module-level, so GLM5-Next should be covered by construction, but "should by construction" is not a measurement and
**no GLM5-Next checkpoint exists on this box**.

**Done =** a coverage-matrix row for GLM5-Next: both arms, `layer` and `tensor`, 2 and 3 GPU, text byte-identical, and
the `implements no shared view` warning **absent** from the log.  Whoever has a checkpoint owes this row; nothing else
is needed.

---

## 6. W5 (TODO #368) — a recurrent-only memory

`llama_memory_seq_state_take()` refuses anything that is not a **hybrid**: it moves the recurrent half and leaves the
attention cells alone (they are shared).  A pure recurrent model (Mamba class, no attention KV at all) therefore gets
**no state at all** on the second arm, so the pair would either refuse it or silently run it on an empty state.

This is a gap in the *take*, not in the sharing: there is no attention KV to share, so the whole state is the recurrent
state and the correct shape is the same take with the recurrent half as the entire state.  **Parked** until a model of
that class exists on the box: implementing it without a model to measure would be a claim with no oracle.

---

## 7. W6 (TODO #371, NEW) — the prefill/decode SPEED BALANCE: the prefill arm skips turns while a decoder is active

This is the item the maintainer asked for, and it is the only one on this list that may correctly end in **"no, the
curve does not justify a knob"**.  It is written up as a measurement campaign for that reason.

### 7.1 The problem, stated with the numbers we have

Today the pair's Pareto point is fixed: **-21…-24 % prefill** buys a decoder that keeps moving.  The worst in-window
gap is what a user still feels, and it does not vanish.  Every pair below is the **same shape and the same run family**, so the two columns are comparable (the step series is
`arms/smooth.py`'s `arms*/pf` row, the gap is `arms/interactive.py`'s `gap_max` for the same ubatch):

| shape | prefill arm's own step, p50 (step count) | decoder's worst gap during the document prefill |
|---|---|---|
| host tier, `-ub 512` (`-ncmoe 40`, arena 5000) | not recorded as a step series | **95.5 ms** |
| host tier, `-ub 2048` | **771 ms** (7 steps) | **86.5 ms** |
| host tier, `-ub 4096` | 1227 ms (5 steps) | **60.5 ms** |
| fully resident, `-c 16384`, `-ub 2048` | **339 ms** (4 steps) | **32.8 ms** |

**Read that table carefully before designing anything: the decoder's worst gap is much SMALLER than one prefill step**,
by a factor of four to twenty on every row where both are recorded.

An earlier draft paired the gap of one ubatch with the step of another, which is exactly the mis-pairing a shape-keyed
reference makes easy (a timing reference is per model, per split, per device count **and per ubatch**, `AGENTS.md`
rule 6 and `GREEDY-PURITY.md` §38); the table above is the corrected pairing.
So the naive model ("the decoder waits for one whole prefill ubatch") is **wrong on this build**, and whatever the real
mechanism is, it must be identified before a policy is written.  Candidate mechanisms, in the order they should be
tested (each is cheap):

1. **the device time-slices the two contexts' streams**, so the decoder is interleaved into the prefill step rather
   than queued behind it, and the gap is one *chunk* of it (block 16's `-ib` cap admits a bounded number of prompt
   tokens per step, and the pair runs that admission policy on the prefill arm);
2. **the prefill arm's step is internally chunked** (the ubatch is split across graph evaluations), so the decoder gets
   the device between chunks;
3. **the recorded "step" granularity in the harness** is the *loop iteration*, not the device occupancy, so the p50
   above over-states what the decoder actually waits for.

The census that decides it: per-step timings on both arms plus `LLAMA_KV_DEBUG_CELLS=1` (the compact per-ubatch cell
census), correlated by timestamp.  `smooth.py` already records `pf_step_p50`, `pf_step_max`, `itl_d_p50/p95/max` and
`dec_during`; what is missing is a per-step alignment.

### 7.2 The idea to evaluate

**The prefill arm skips one or more submission opportunities while the decode arm has pending work**, so the decode arm
gets **several uninterrupted steps** before the prefill arm grabs the device again.

Concretely: the prefill arm's loop, before it builds and submits a batch, asks a counter on the shared arms state "how
many decode steps have completed since my last submission, and is a decode step pending now?".  If a decoder is
actively generating and fewer than `K` decode steps have run since the last prefill submission, the prefill arm
**does not submit this turn** (it parks on the handshake it already uses at request boundaries, or it takes a step that
only advances bookkeeping).  After `K` decode steps it resumes and submits its own work.  The effect a user should see:
the decoder gets bursts of `K` tokens at its own uncontended step time, instead of one token per prefill gap.

### 7.3 Why this is NOT one of the measured losers

The campaign already measured and rejected three host-side policies, and this must not repeat them:

| rejected | what it did | why it failed |
|---|---|---|
| `3f2768131` per-gather split yield point | a **yield point inside** a host-to-device gather | the gather is not the contended resource; the yield added bookkeeping to a path that was already pipelined |
| `540d442a8` stream priority (-1/+1) | asked HIP to prioritise one context's streams | measured neutral or worse: the two contexts' work is not competing on the same queue in a way a priority changes |
| block 16's `-ib` cap and `LLAMA_SERVER_PREFILL_DUTY` | bounded a decoder's stall **inside a shared step** | **inert by construction on the pair**: the decoder has its own context and thread, so there is no shared step to bound (measured: duty 50 -> 50 tokens, 1.00x) |

The skip-turn idea differs in kind: it is a **cross-arm submission policy at the batch level**, not a point inside a
data path and not a device hint.  That is exactly why the campaign's "no policy in the hot path" conclusion does not
already settle it, and also why it must be measured rather than assumed.

### 7.4 Design constraints (non-negotiable, from the purity rulebook)

* **Counters, never wall clocks.**  A policy keyed on time is not reproducible: the same request sequence would take a
  different number of submission opportunities per run and the recorded text would move for no reason.  The decision
  must be a function of (decode steps completed since the last prefill submission, whether a decoder is active, the
  outstanding prefill work).
* **It changes WHEN the decode arm runs, never WHAT it batches.**  The decode batch stays one row per active slot; the
  prefill batch stays whatever the fair-share admission chose.  The moment the policy changes a batch's *composition*
  it becomes a numerics carrier (`GREEDY-PURITY.md` §36/§63) and the text oracle is gone.  Guard: assert in a debug
  build that the per-step batch shape is identical with the policy on and off for a single-slot run.
* **Never skip with nothing decoding** (a lone prompt must not be slowed for no one) and **never skip on a request's
  first prefill step** (that would be a pure TTFT loss).  Block 16 solved the same question for the single context as
  "a step with NOTHING decoding always takes the whole `-b` window"; reuse the wording and the predicate.
* **Bound the streak.**  `K` is a cap, not an unbounded yield: under continuous multi-slot load the prefill arm must
  still make progress, or admission starves and a newcomer's TTFT becomes unbounded.  A total-skip budget per request
  (or per admitted chunk) is the guard.
* **Interaction with the fair-share admission.**  The prefill arm runs block 16's max-min fair share across the prompt
  slots.  A skip must not make one slot's admission order depend on the other slot's decode activity, or the fairness
  gate's own criterion (short-prompt TTFT) becomes a function of the neighbour.
* **Do not touch the deprecated knobs.**  `-ib` and `LLAMA_SERVER_PREFILL_DUTY` stay inert on this path; the new
  control is its own knob (below) so an operator cannot accidentally re-enable a mechanism that does nothing.

### 7.5 Knob shape (when it exists)

`--prefill-decode-arms-skip N` (default **0 = off**, so the delivery's behaviour does not move), plus the diagnostic
census that lets the campaign measure it: a `-lv 4` line per skip with the streak length and the decode steps it
bought.  If it lands, it lands as a **tuning** knob in `ENVIRONMENT.md` §23 next to the pair's other switches, and the
default stays 0 until a Pareto curve justifies something else.  A beneficial default is the maintainer's call, on the
curve, not this document's.

### 7.6 Measurement plan (this is the item, not an appendix)

1. **Establish the mechanism** (§7.1): per-step alignment on both arms, one host-tier shape and one fully resident
   shape.  No policy until the gap's composition is known.
2. **Sweep `K` in {0, 1, 2, 4, 8}** on the same shapes with `arms/smooth.py` (the campaign's metric: `itl_d_p50`,
   `itl_d_p95`, `itl_d_max`, `dec_during`, `pf_tps`, `pf_step_*`) and `arms/interactive.py` for the user-facing number
   (document TTFT, document rate, reply gaps).  Warm reps only: a cold host-tier rep is a §89 residency class, not an
   oracle.
3. **Purity at every `K`**: the tier-1 text equality (`arms/gate-server-arms.sh`), the pair vs single-context text on
   a class-matched shape, and `gate-prefill-fairness.sh --arms-band` (the arms criterion) plus
   `gate-prefill-duty.sh --inert-band`.
4. **The Pareto conclusion.**  Report the curve, not a point: prefill cost against worst in-window gap.  Accept the
   feature only if some `K` buys a **>= 50 % reduction in the worst gap for <= 3 percentage points of extra prefill
   cost** (state the bound up front, so a flat curve is recorded as a refutation rather than a shrug).  The baseline to
   beat is the shipped default: worst gap 95 / 60 / 34.9 ms at 1252 / 3111 / 4792 t/s.
5. **A negative result is a result.**  If the curve is flat or dominated, record it here and in `GREEDY-PURITY.md`, and
   retire #371.  That is the same standard the three rejected policies were held to.

### 7.7 Honest expectation

The headroom is **bounded by the mechanism**: if the decoder is already interleaved into the prefill step, the worst
gap is a *chunk* of foreign work and a skip policy can only remove the part that is the prefill arm's own submission
latency (a few ms to tens of ms), not the 60-95 ms tail.  The larger prize, if the mechanism turns out to be "the
decoder waits for a full ubatch", is a real one, and the table in §7.1 exists precisely because that cannot be settled
by reasoning.

**Do not** reintroduce a host-side rate gate, a mid-gather yield or a stream priority to "help" this item: all three
are measured losers and their records are in the campaign.

---

## 8. W7 (TODO #370) — the pair's slot erase and cell accounting

Measured with `scripts/gate-arms-soak.sh soak` (4B, `-np 3 -c 8192`, six rounds of distinct ~2600-token prompts): the
cache fills to **8174 of 8192** cells and stays bounded (the server keeps answering), and after erasing all three slots
the next tiny request's ubatch reads **used=5446**, so the erases released about **one** resident prompt's cells rather
than three.

Two facts, only one of which is a defect.  `POST /slots/N?action=erase` erases a slot's **saved state** (block 12's slot
cache), not its KV cells; and on the pair the idle slots' cells legitimately stay resident because **parking is declined
by the co-owner guard** (`idle slot parking: not supported by this KV cache, the idle slots stay where they are`).  What
is open is whether the pair *should* release all three (a slot whose saved state is erased has no prompt to keep) and
what the accounting should read afterwards.  The gate records the number rather than asserting it; turning it into an
assertion is this item.

Related, and worth reading while here: `try_clear_idle_slots()` and the KV-full recovery path are **unreachable on the
pair by construction** (both decline by their own guard, and admission is chunked against the space the co-owner
handshake reserved).  The soak asserts the property the clear path exists to provide instead.

---

## 9. W8 (TODO #372, NEW) — multi-slot prefill batching: is the prefill arm's ubatch already a multi-slot batch, and what does the ubatch SPLIT cost?

**The question (maintainer, 2026-11-22).**  "If there are concurrent slots actively prefilling, each slot scans through
the full set of layer weights per ubatch chunk.  What if, with multiple active prefill slots, we processed the prefilling
layer by layer with each active prefill slot processing its ubatch for the active layer?  Could that give batched or
concurrent prefilling for better throughput?"

### 9.1 The premise check: within one step, the slots ALREADY share the weight pass -- **since r300, and because of the fair share**

**The history matters here, because "prefill is batched" is a DELIVERY property, not an upstream one.**  Before r300
(block 16) `update_slots()` added the decode tokens and then filled the rest of the batch **in slot index order**, so the
**lowest slot took the whole window** and every other prompt waited for that slot's entire prompt: the fairness gate's own
header records the measurement (a 7-token prompt behind a 13204-token one waited **1.91 s**), and notes that `-b` does not
help because it changes the grab size, not the order.  Under that scheduler a prefill step carried **one** sequence, so
every step re-read the whole weight set for that one slot's chunk.  The graph/ubatch layer always supported several
sequences (that is the parallel-request machinery), but the **scheduler never composed it for prefills**.

r300 replaced the fill order with `share_prompt_budget()`, and that is what makes the multi-slot prefill batch exist at
all.  So the answer to "does this repo now support batching on prefills" is yes, and it arrived **with the fix for the
serialisation, not separately from it**.

Read the rest of the scheduler before designing anything, because the mechanism the question proposes is what the prefill
arm now does **up to one ubatch**:

* `share_prompt_budget()` (`tools/server/server-context.cpp`, the `wip/session-scheduling` fair share that block 16
  shipped) splits the step's prompt window max-min across **every slot that still has prompt tokens**, with the stated
  result: *"nobody fills n_batch first and every active prompt is admitted to every step"*.  So two prefilling slots
  put rows into **one** `llama_batch`, and the mask machinery keeps each sequence's attention separate.
* that batch is then split into `n_ubatch`-sized ubatches (`split_equal`), and **each ubatch is one graph**: the linear
  ops see a single 2-D tensor of all rows in it, so the weight read is **once per ubatch for all slots' rows in it**,
  not once per slot.
* on the arms path the chunk cap that would shrink a slot's contribution is **inert by construction**: `-ib` bounds a
  step only when a **decoder is waiting on this arm**, and a slot on the decode arm does not (the code says so at the
  `-ib` comment: *"the prefill takes the whole window and `-ib` is inert"*).

**And the shortest-prompt priority is not dead code, but it is not the default either.**  Two different things were
bundled into "the fix", and they have different status today:

* the **serialisation fix**: `share_prompt_budget()`, i.e. every active prompt is admitted to **every** step, with a
  one-token floor (so starvation is impossible at any `n_parallel`) and a rotating tie-break (`slot_next`, "rotation makes
  the tie-break fair across steps").  This is what creates the multi-slot batch.  **ON by default**, with
  `LLAMA_SERVER_PREFILL_FAIR=0` as its kill switch.
* the **shortest-prompt-priority droop**: `LLAMA_SERVER_PREFILL_SKEW`, a `N:1` weighting of the shortest-remaining slot,
  clamped 1..64, explicitly *"not SRPT: every other slot still advances every step, so starvation is impossible"*.
  **Default `1` = the plain equal share**, i.e. OFF.  It is a live tuning knob, not dead code: measured on four prompts of
  6000/3000/1500/400 tokens at `-np 4`, mean TTFT is **1.401 s at 1**, 1.082 s at 5 and 1.040 s at 20, and the *longest*
  prompt also improves (2.201 -> 1.975 s) because short slots retire and stop competing.

So the priority mechanism's job changed rather than disappeared: with the fair share in place the question is no longer
"which slot gets to run" (everyone does, every step) but "who gets the larger share of one shared window", which is a
TTFT-versus-throughput tuning.  The default stayed at the plain share because the pathological serialisation was what
needed fixing, and the droop's gain was not promoted.  Anything that wants the droop back turns one knob, and the
composition census below is what would tell you whether a droop is what a given shape needs.

So "each slot scans the full weight set per chunk" is **not** what happens for co-batchable slots.  What *is* true, and
is the real cost, is the **ubatch split**: a step whose admitted tokens exceed `n_ubatch` becomes N ubatches, and each
one re-reads the whole weight set.  That is the axis `-ub` moves, and it is already measured: `-ub` 512 -> 2048 -> 4096
lifted the pair's prefill **1440 -> 3041 -> 3884 t/s** on the reference arm, and 8192 buys only a further +8 % with a
worse decoder tail (124.5 vs 95.1 ms).  A bigger ubatch is exactly "process more slots' rows per weight pass", so the
proposal, applied inside the prefill arm, **is the ubatch knob**.

### 9.2 Where the idea does have teeth

1. **Prove the co-batching actually happens.**  The fair share *admits* every active prompt but does not make them
   arrive together: a newcomer is admitted to the next step, and a short prompt finishes while a long one runs.  If in
   practice the long prompt is alone in its ubatches for most steps, the aggregate gain the idea promises is not being
   collected, and the fix is scheduler-side (admit together) rather than kernel-side.  **This is cheap to measure and
   is the first task:** `LLAMA_KV_DEBUG_CELLS=1` prints one line per sequence per ubatch
   (`[TAG_KV_CELLS] seq=.. n_tok=..`), so grouping those lines by ubatch gives the real composition.
2. **Host-expert MoE is where batching rows multiplies.**  A routed MoE reads the **union** of the experts its rows
   route to, so doubling the rows per ubatch does not double the expert traffic; it is the one place where multi-slot
   batching is superlinear (`GREEDY-PURITY.md` §89/§105: the arena hit rate and the t/s both scale with prompt length).
   A two-slot concurrent prefill on the host tier is the arm that would show it.
3. **A cross-chunk layer-major schedule** (for layer L, serve several *chunks* before moving to L+1) would only help if
   the chunks' activations can be held simultaneously, which is the definition of a bigger `-ub`.  It therefore
   degenerates into (1) and `-ub`, and the worth-measuring version is "how far does `-ub` scale before activations bind
   on this card", which the delivery has partly measured (4096 sweet spot; 8192 worse on decode).

### 9.3 Where it must NOT be applied: across the two ARMS

The tempting extension is to do the layer pass once for **both** arms' rows (decode rows plus prefill rows), which would
genuinely read the weights once instead of twice per layer.  **That is the shared step the feature exists to remove**:
it puts the decode row's latency back inside the prefill layer's work, which is precisely the 300-880 ms stall block 17
was built to delete.  Recorded here so it is not proposed as an optimisation later.

### 9.4 Measurement plan

1. **Composition census** (the premise check): two slots with long prompts on the pair, `LLAMA_KV_DEBUG_CELLS=1`, and
   count distinct `seq=` per ubatch.  If the median ubatch holds one sequence, the headroom is scheduler-side.
2. **Aggregate scaling**: two concurrent prefills against one at the same total tokens, at `-ub` 512 / 2048 / 4096 /
   8192, on a dense model and on the host-expert MoE.  Report tokens/s for the *pair of requests* and the per-request
   TTFT, because a two-slot prefill trades one request's latency for the other's throughput.
3. **Expert-reuse effect** (§9.2 item 2): the same sweep on `-ncmoe 48` with the arena census, so the claim is a
   measured hit-rate change, not an inference.
4. **Purity**: batching more sequences per ubatch changes the ubatch **composition**, which is a numerics carrier
   (`GREEDY-PURITY.md` §36, §63, §105).  A two-slot concurrent prefill is **not** a text oracle; the oracle is each
   slot run **alone** at the same shape, compared across the policy change, plus `gate-prefill-fairness.sh --arms-band`
   and `gate-prefill-logits.sh` for the prefill path.
5. **Coupling to W6.**  Both items move along the **same Pareto axis**: more prefill work submitted per step (bigger
   ubatch) versus fewer, shorter prefill steps (turn skipping, smaller ubatch).  Measure them on one curve, not as two
   independent wins: the deliverable of the pair is "worst in-window gap against prefill rate", and each item is a point
   on it.

### 9.5 Acceptance ("done")

A composition census showing where the rows actually land, an aggregate-scaling table for a two-slot prefill at four
ubatches on two model classes, and **either** a scheduler change that keeps more slots in the same ubatch (with purity
green) **or** a recorded conclusion that the ubatch knob already captures the available gain and no scheduler change is
owed.  As with W6, **a refutation is a legitimate result**.

---

## 9b. W9 (TODO #373, P0) — a qwen4exp ABORT at a request boundary: the indexer's padded extent and the attention cache's disagree

**Reported by the maintainer 2026-11-22, on the released block 17 (`--prefill-decode-arms-mode auto`).**  Log and
backtrace: `/llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL/server-logs`.

### The crash

```
/home/stew675/llama.cpp/src/models/qwen4exp.cpp:1138: GGML_ASSERT(mctx_idx->get_n_kv() ==
    inp->mctx->get_attn()->get_n_kv() && "the indexer cache must track the attention cache cell for cell") failed
```

Backtrace: `ggml_abort` <- `llama_model_qwen4exp::graph::graph` <- `build_arch_graph` <- `build_graph` <-
`llama_context::process_ubatch` <- `decode` <- **`server_context_impl::update_slots()`** (the PREFILL arm's thread, not
the decode arm's).  Shape: `--ctx-size 204800 --parallel 3 --kv-unified`, `-sm tensor` on 3 x R9700, `q8_0` K/V,
`--ubatch-size 2048`, the 94 GB STEW675 IQ4_NL model, all layers on the GPU, `--prefill-decode-arms-mode auto`.
Workload at the time: slot 0 decoding, slot 1 prefilling a **70 986**-token prompt, slot 2 prefilling **23 003** tokens.
The abort came within ~0.4 s of slot 1 printing `progress = 1.00` (its prompt finished and was handed to the decode arm).
The assertion is **pre-existing**, from upstream #27742 (commit `6c84c7d5d`, 2026-08-27), carried in by block 11: this
configuration is violating an invariant the model has always demanded, not a new one introduced by the block.

### Why the invariant is worth keeping, and why it is fragile

`get_n_kv()` is the **padded** extent the attention views use:
`max over streams of min(cache.size(), max(pad, PAD(used, pad)))`, with `pad` = 256 normally and
`LLAMA_KV_N_PAD_PREFILL` (**8192 by default on a multi-device split**) for prompt-sized ubatches.  The indexer cache must
mirror the attention cache cell for cell, so the two padded extents must be equal.

They are computed **separately and in sequence**, and each is computed *after that child's own* `apply_ubatch`:

```
llama_memory_hybrid_idx_context::apply()
    llama_memory_hybrid_context::apply();   // attention child: apply_ubatch(), then n_kv = get_n_kv(...)
    ctx_idx->apply();                       // indexer child:   apply_ubatch(), then n_kv = get_n_kv(...)
```

In a single-context server nothing runs between those two lines, so the two reads see one state.  **With the pair they
are two threads on one shared cache**: the decode arm may `apply_ubatch` between the attention child's `n_kv` read and
the indexer child's, which moves the used-cell count (the co-owner allocator publishes the decode arm's cells into the
same caches).  The two padded extents then differ whenever the movement crosses a **pad boundary** (a multiple of 8192
for a prompt ubatch), which is why this is rare and why it needs a large prefill plus a request boundary: the boundary is
when the decode arm gains a sequence and starts applying, and 70K of cells is what puts `used` next to an 8192 multiple.

> **NEXT SESSION:** see [`HANDOVER-365-and-release.md`](HANDOVER-365-and-release.md) -- the self-contained plan for
> `TODO` #365 (the meta-aware device copy) and the block-17 amendment + release wrap-up, with the current repo state,
> the fix branch, the re-validation commands and the landmines.

### UPDATE 2026-11-22 (final): SHIPPED in the r347 block-17 amendment, with the owed gate arm

`WORKLOG.md` 261122.036.  The whole fix set is in **block 17** as a same-block amendment (the maintainer's decision;
no new block, no renumbering), and it carries two things this campaign listed as owed:

5. **The synchronous host paths stop using the `cudaStreamPerThread` sentinel** (the fifth defect, `#366`'s commit):
   one private non-blocking stream per **(thread, device)** — `ggml_cuda_host_stream()` — for every synchronous
   host↔device path (`set_tensor`, `get_tensor`, memset, the bounce and peer-staging copies, the KV-region row
   moves), with the async ring's origin sharing it so the ordering the sentinel used to provide is preserved.  A
   peer arm's graph capture claims the device's legacy default stream, so those copies were refused
   (`operation not permitted when stream is capturing`) on the host tier.
6. **`#366` closed**: the pair's declared reserve is `common_arms_dec_reserve()` — the flat 3584 MiB floor plus a
   **width-keyed tier** (0 at `-ub <= 1024`, 1536 MiB at `>= 2048`), because the pair creates a SECOND late narrow
   view (the prefill arm's post-prefill re-reserve).  Measured 3584 fails, 4096 fails, **5120 clean** — which is
   what the derivation prints, so the host tier needs no knob.
7. **The owed gate arm exists now**: `scripts/gate-arms-soak.sh boundary` — two prefills in flight plus a decode on
   one pair, pad buckets shrunk to 256, `BB_ROUNDS >= 3`, and a fourth sequential fresh request compared against a
   single-context server (`MANIFESTS.md` gate 17b).  It also found that a **small model over a 3-way split is not a
   shape anyone runs** (a 4B on 3 devices faults in `llama_kv_cache::find_slot()` on the *released* build too), so
   its default is the validated 35B-A3B hybrid split.

### UPDATE 2026-11-22 (late): BOTH configurations clean -- the fourth defect (the VMM pool's peer grant) fixed

`WORKLOG.md` 261122.034.  Adding to the three fixes below:

4. **The VMM pool's peer grant is best effort**, exactly as the slab's has been since it hit the same wall.  `alloc()`
   granted every peer descriptor in ONE `cuMemSetAccess` and `CU_CHECK`ed it, so one refused descriptor (the driver
   cannot build a peer mapping once the arena has taken the VRAM) aborted the process -- the host tier's
   `HipVMM Failure: invalid argument` at ~2 s, pair-only and arena-on only.  Now: owner alone (located explicitly),
   then each peer alone, a per-pool `peer_mask`, ranges in a registry, and the copy predicate widened from
   `ggml_cuda_slab_peer_copy_ok()` to **`ggml_cuda_peer_copy_ok()`** covering pools too, so a refused pair stages
   through pinned host memory.  **And** the async-input origin uploads on a **private non-blocking stream** instead of
   the `cudaStreamPerThread` sentinel (HIP couples that to the legacy default stream, so a peer's capture refused it --
   measured as a failed `llm_graph_input_ple::set_input` on the host tier).  Ordering was already event-carried.

| configuration | result |
|---|---|
| fully resident, 3 slots x (71k + 23k + 800-token gen), 2 rounds | clean |
| host tier `-ncmoe 48 -c 131072`, arena 13.3 GiB, `MOE_EXPERT_CACHE_AUX_RESERVE_MIB=6144`, 2 rounds | clean (`aux=6.00 GiB`); one earlier round survived **6 refused peer grants** by staging |
| `gate-arms-soak.sh soak --quick` | PASS 11/11, text oracle included |

**`TODO` #366 is now a hard blocker for the host tier**: at r346's flat 3584 MiB the decode arm's post-prefill
re-reserve (1430 MiB) cannot be allocated next to the arena, at both `-c 204800` and `-c 131072`; 6144 MiB is clean.
This is the maintainer's "is the non-pair approach's pre-allocation cleaner?" question, answered: the non-pair path is
not pre-allocating *buffers* (the two arms must own separate compute buffers -- they compute concurrently with different
shapes), it simply attempts its grants and its reserves **before** the arena claims the space.  What the pair should
borrow is that *timing discipline* (`#366`'s derived reserve), not the buffers.

### UPDATE 2026-11-22 (evening): the THREE fixes, and the one the host tier still needs

`WORKLOG.md` 261122.033.  Implemented on the scratch branch (`~/llama.cpp` `crash373-fix`), exported for review as
`repro-373/FIX-candidate-373.patch` (2 commits, 206 insertions, 7 files), **not shipped** (block 17 is the target, so it
is a same-block amendment, and that is the maintainer's call):

1. **One apply lock per memory GROUP** (`llama_memory_i::apply_mutex()`, shared through `create_memory()`), taken at the
   engine's memory entry points: `init_batch` (the rollback snapshot!), `apply`, `clear`, `state_write/read`, the
   `llama_memory_*` seq ops and park.  Worst critical section **783-1161 us**, once per prefill step.
2. **The Meta split-state cache is per VIEW SLOT** (`split_state_cache[4][8]` + `cur[4]`/`used[4][8]`/`tick[4]`,
   indexed by `g_meta_view_slot`), and the slot is bound for **every** graph build/alloc
   (`llama_context::meta_bind_view_slot()`, from `graph_reserve()` and `sched_alt_select()`) instead of only when the alt
   scheduler state changed.  This is the class that SIGSEGVs the *released* build: two arms, one per-buffer
   `unordered_map`, `clear()`/rehash under a peer's `find`.
3. **The async-input ring waits per ORIGIN stream**, not `cudaDeviceSynchronize()` -- which is illegal while the peer arm
   captures a graph (`operation not permitted when stream is capturing`, then every launch on the device fails).  If even
   the per-origin wait is refused the copy reports "not staged" and the caller takes its synchronous path.

**Validated:** 3 slots x (71k + 23k + 800-token generation), fully resident, **3 rounds clean** (0 asserts, 0 ROCm
errors, 0 segfaults; pre-fix it died in 20-55 s every run); pair OFF clean; `gate-arms-soak.sh soak --quick` **11/11**
with the class-matched text oracle, i.e. no number moved.

**Still open, and it blocks the host tier:** with `-ncmoe 48` the pair aborts ~2 s in inside
`ggml_cuda_pool_vmm::alloc` -> `cuMemSetAccess` ("HipVMM Failure: invalid argument") -- the **P2P grant** for the second
context's pool growth, refused because the arena has taken the VRAM.  The slab already treats that grant as best-effort
(`peer_mask` + `ggml_cuda_slab_peer_copy_ok()` staging); the VMM pool still `CU_CHECK`s it.  Pair OFF at the same shape
is clean.  Two designs, **a decision for the maintainer**: (A) the slab's pattern applied to the pool (best-effort grant,
per-pool `peer_mask`, pool ranges in the peer-copy predicate), or (B) warm the decode arm's pools at context creation, the
way r346 already declares its graph reserve before the arena claims the space.  `TODO` #365 (the device copy for the
state handover) comes after that.

### UPDATE 2026-11-22: REPRODUCED, DIAGNOSED, and it is bigger than one bug

The full record is `WORKLOG.md` 261122.032; the harness is `repro-373/`.  What changed from the analysis below:

* **It is not rare and the pad boundary is a red herring.**  Three slots (a short prompt + a long generation, a 20k prompt,
  a 6k prompt) abort on the assert **within ~30 s, first try**; the 71k/23k shape in 40-55 s.  Instrumenting the two
  children at the abort measured `attn used=22921` vs `idx used=23999` (`n_kv` 23040 vs 24064, both `size=204800`): the
  divergence is **one whole peer PREFILL ubatch**, so any interleaving shows up and nothing had to cross an 8192 bucket.
  Pair OFF: clean (`E1` confirmed).
* **The window is wider than "two reads".**  `apply()` and `init_batch()` BOTH mutate the shared cell map, and neither was
  serialised: `init_batch()` -> `llama_kv_cache::prepare()` **copies the whole cell map** for its rollback snapshot, so a
  peer `apply_ubatch()` during that copy corrupts `std::set<uint32_t> used` -- measured as a SIGSEGV inside libstdc++'s
  red-black tree.  That is the *same* defect with a memory-safety symptom, and it is pre-existing on the release build.
* **Fix (written, validated, NOT shipped):** one apply lock per memory **group** (`llama_memory_i::apply_mutex()`, shared
  through `create_memory()`), taken at the engine's memory entry points -- `init_batch`, `apply`, `clear`,
  `state_write/read`, the `llama_memory_*` seq ops and park.  `alloc_sync`/`seq_state_take` stay unlocked (request-boundary
  only).  Worst critical section measured **783-1161 us** (a 2048-token `init_batch`), one per prefill step;
  `gate-arms-soak.sh soak --quick` 11/11.
* **Two more defects sit behind it, so the pair is not stable on this shape yet:** (1) a SIGSEGV inside
  `ggml-backend-meta.cpp` (`set_tensor_async`'s copy lambda / `get_split_state`) that the **released** r346 build already
  has -- the shared KV buffer's per-graph VIEWS are registered in the view source's buffer, which both arms use; a
  candidate hardening (set `g_meta_view_slot` on every `sched_alt_select`, not only on an alt-state change -- measured
  arms at slots 0 and 3 afterwards) did not fix it alone.  (2) a HIP `operation not permitted when stream is capturing`
  from `ggml_cuda_async_input_set`, i.e. one arm's async input staging against the other's capture; `GGML_CUDA_DISABLE_GRAPHS=1`
  (the real knob -- there is no `GGML_CUDA_GRAPHS`) still crashes, so it is not capture alone.
* **Owed:** the maintainer's block decision; the Meta and capture/staging defects; and the gate arm below, which is now
  `repro-373/` plus assertions.

**Leading hypothesis, stated as such (kept for the record; superseded by the UPDATE above):** a **co-owner race on the two
children's `n_kv` computation**, not a stale indexer pool and not the pad formula itself.  Two facts support it and one would settle it:

* the abort is on the prefill arm's thread while the decode arm is live, and only ONE
  `allocation handshake clean at the 'handover' boundary` line precedes it (the boundary that gave the decode arm its
  new sequence, immediately before the crash);
* parking is declined on this cache (`idle slot parking: not supported by this KV cache`), so the park bound in
  `get_n_kv()` is not involved, and both children are constructed from the SAME `sinfos` and `ubatches`
  (`heads_idx = heads_attn` in `init_batch`), so the *inputs* are identical and only the *timing* can differ;
* **the discriminator:** the same 3-slot workload with `--prefill-decode-arms-mode off`.  If it stops aborting, the race
  is the cause.  A repro can also be made *more* likely by shrinking the pad buckets
  (`LLAMA_KV_N_PAD_MIN=256 LLAMA_KV_N_PAD_PREFILL=0`), which turns smaller timing differences into a visible mismatch.

### Fix direction (preferred first)

1. **Derive the indexer's `n_kv` from the attention's** for the pair instead of computing it twice.  The invariant already
   *says* they are equal, and `init_batch` already gives the indexer the attention's slot layout (`heads_idx = heads_attn`),
   so making the value follow the layout removes the possibility of disagreement by construction.  The read-side already
   behaves that way elsewhere (`qsa_n_kv_window()` reads `get_idx()->get_n_kv()`).
2. If (1) turns out to be wrong for a legitimate reason (a real case where the two caches differ), then the quiet window
   must cover the whole `llama_memory_hybrid_idx_context::apply()`, i.e. the arms' handshake held across both children,
   and the assert becomes a check inside that window.
3. Either way, a **gate arm**: 3 slots, one long prefill and one medium prefill and one decode on a qwen4exp hybrid, with
   the pad buckets shrunk so the window is wide, repeated N times.  Nothing in the current gate set drives three slots
   through a boundary with two prefills in flight, which is why this reached a user.

### Severity and scope

It is an **abort, not silent corruption**: the server dies instead of emitting wrong text, which is the benign failure
mode.  It affects **opt-in users only** (block 17 is default off, and the shipped single-context server has no second
thread to race), so it does not invalidate `v18-919794500-r346`.  It **is** a P0 for the feature and a blocker for any
default-on decision, and it is the strongest argument yet for a soak gate that drives a real mixed workload rather than
two slots.

---

## 10. G1 (TODO #369) — gate debt: three gates still anchored to the retired Flash-Next model

`gate-park-idle-slots.sh`, `gate-moe-promise.sh` and `gate-host-pool.sh` were **repointed** at r346 to the maintainer's
quality-first replacement, `Qwen3.8-Flash-Next/STEW675-IQ4_NL/Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf`, and
re-run on it (59/59, 238/238, and a park PASS at `CTX=16384` with `--lazy-mode on`).  Three more still name the retired
`GSQ-IQ3_XXS` tree and were **deliberately not** repointed blind, because each carries **recorded text/KLD anchors for
the BF16-hc / all-VRAM class**, and a new model is a new class:

* `gate-hc-base-copy.sh` (its `MODEL` default),
* `gate-hc-fusions.sh` (`MODEL_ROOT/GSQ-IQ3_XXS/...`, and its header records the measured hashes for that class),
* `gate-host-expert-text.sh` (its header sizes the run at 75.8 GB / ~71 GiB peak RSS).

**Done =** repoint each, **re-record its anchors on that class** (the `--record`/`--kld` paths already exist), correct
the headers' numbers, and run each once.  Until then they refuse at the model check, which is the correct failure mode
rather than a wrong anchor.

---

## 11. Ordering, dependencies, and the path to a default-on decision

1. **G1 + W7 + W3** are small, independent, and close out everything that is currently "recorded but not asserted".
2. **W6 and W8 together**, because they move along the **same Pareto axis** (more prefill work per step versus fewer,
   shorter prefill steps) and their results only mean something measured as one curve: "worst in-window gap against
   prefill rate".  W6 is the maintainer's turn-skip idea, W8 is the maintainer's layer-major multi-slot idea; each is a
   point on that curve, and a refutation of either is a legitimate outcome that gets recorded as such.
3. **W2** then **W1**: the device copy is a prerequisite in practice for a speculator on a `-sm tensor` hybrid
   (a 75 ms boundary plus a verify batch is a TTFT problem), though not formally blocking.
4. **W4** whenever a GLM5-Next checkpoint appears; **W5** parked until a Mamba-class model exists on the box.
5. **Default-on** is a separate decision, taken on evidence after 1-4, and not implied by any green gate here: the
   feature costs prefill, so "on by default" needs a statement about which shape it is good for.  The `auto` form
   exists to make that decision expressible without changing the shipped default.

**Housekeeping for whoever picks this up:** every item above is `TODO` #364-#371, `wip/CAMPAIGNS.md` carries the open
row, and any item that lands in `patches/` **needs the maintainer's approval for a block** (AGENTS.md: a block is never
a gate outcome).  Items that turn out wrong move to `archive/work/` with a `WORKLOG.md` entry; nothing here becomes a
stub.
