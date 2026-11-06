# PLAN — the rest of the `host-expert-split` campaign (post-investigation, pre-implementation)

**Status: the plan of record from 2026-11-05 onward.**  The investigation is closed; this is what gets built,
in what order, with what gates, and what would make us stop.  Nothing here is in `patches/` yet.

Read with: [`README.md`](README.md) (the scoping note), [`DESIGN.md`](DESIGN.md) (the PR review + convergence
design), [`STEP1-REPLAY.md`](STEP1-REPLAY.md) (the evidence, incl. §7 the ranking finding and §8 the
write-back-vs-warmer pricing), [`P1-POOL-RANK.md`](P1-POOL-RANK.md) (the one arm already implemented and
measured).

---

## 1. Settled — do not re-derive these

| # | fact | value | where |
|---|---|---|---|
| S1 | The pool's content is ranked by `t.heat`, the combined prefill+decode counter, which is **worse than chance** for decode routing (top-64 overlap 10.5 % against 12.5 % chance); `heat_dec` is 24.2 % | — | STEP1 §7 |
| S2 | Ranking the pool by `heat_dec` instead is **+10.6 % generation**, +1.1–2.1 % prompt, 0 copies, 0 bytes | 22.6 → 25.0 t/s | P1 |
| S3 | The ranking is **provably inert at a mirror** (identical counters to the digit; 833.7/835.0/833.4 t/s) | — | P1 |
| S4 | Our deficit is **holes**, not transport: at a full-complement budget (94 % of the set) the pool answers 0.795 of arena misses and **23–30 MiB/token** still comes from storage | — | STEP1 §7, PLAN §5 |
| S5 | An exclusive tier (the PR's shape) has **no holes by construction**; that — not D2H economics — is its entire advantage | — | STEP1 §8 |
| S6 | r113's refusal of the literal swap was measured **with a covering pool** (0 MiB of L3 available to win) **and** 2-D strided `cudaMemcpy2DAsync` copies on a tensor split. It does not bound the complement-scarce, single-device case | — | STEP1 §8 |
| S7 | ~~A write-back is slot-neutral~~ **WRONG (2026-11-06, from the code)** — see the W2 precondition below.  The incoming fill is *reading* that slot and the victim's arena bytes are overwritten by the same fill, so the victim cannot take it | needs 2 copies (staging) or a pool-eviction policy | W2 precondition |
| S8 | A page-cache **warmer** is worth −15…−20 % of storage reads at **zero** copies, but **only in mmap/buffered mode** — with `MOE_HOST_POOL_DIO=1` the page cache is bypassed and it is a no-op | — | STEP1 §8 |
| S9 | A better prior and a demote are **not independent**: P1 already removed 12–19 % of the hole traffic for free | — | P1, STEP1 §7 |
| S10 | A periodic conditional swap (N rounds, K per round, hottest-pool vs coldest-arena) is **4–5× less copy-efficient** than an on-demand one and is negative on a page-cached box. **Not worth a knob.** | 0.12–0.18 vs 0.65 MiB saved/MiB copied | STEP1 §6 |
| S11 | The copy direction and vector order are **correct** (an `invert` probe collapses to h 0.114 / 16.7 t/s) | — | P1 |

## 2. The one number that decides the rest of the campaign

Everything now reduces to an exchange rate between **pinned RAM** and **PCIe copies**, and it hangs on a
single unmeasured constant: the marginal cost of our own 1-D swap.

* To match an exclusive tier's L3 on our no-copy layout we must pin **+9.2 … +17.7 GiB** (mean **+12.8 GiB**);
  to reach *zero* storage reads we need the full mirror (46.8 GiB) where the tier needs ~29 GiB — **17.8 GiB
  apart**.
* At **equal L3**, the demote costs those copies instead: **1.10 ms/token** at `c_pcie ≈ 0.04 ms/MiB` — about
  9 % of a 12.8 ms token — and frees the 12.8 GiB.
* At `c_pcie ≈ 0.4 ms/MiB` (the r113 2-D shape scaled to 1-D) the same trade costs **11 ms/token** and the
  demote is worthless.

**Therefore: measure `c_pcie` for a 1-D contiguous D2H on this tree BEFORE writing the policy.**  That is W1,
and it is ~30 lines rather than ~150.

> **ANSWERED (2026-11-06) — see [`W1-SINK-TAX.md`](W1-SINK-TAX.md).**  Measured `c_pcie` = **0.0140 ms/MiB**
> at `-lme 36` (**9.6 µs** per 679 762 B copy, ~2/3 of it hidden behind compute) and 0.0324 ms/MiB at
> `-lme 44`.  Decode cost **-2.3 %** (1.06 ms/token) and -5.4 % (2.06 ms/token) respectively, at **111**
> and **94** D2H copies per decode token.  Against a read that must cost >= ~0.3 ms on this box, the copy is
> ~30x cheaper (~570x on a 125 MB/s device), so the break-even reuse rate is **~3 % here and ~0.2 % on slow
> storage**: the demote does not need a better prior, only that a few percent of demoted victims are wanted
> again.  **W2 is GO.**  The cost is roughly constant in pool size while the benefit falls as the pool grows,
> so the demote's best cell is the below-complement one -- exactly where the holes are.  Caveat: the tax pays
> transport only, so W2 must re-measure with its own bookkeeping.

## 3. Work items, in order

### W0 — close out P1 (cheap, and it may be the campaign's real deliverable)

| item | why | pass criterion |
|---|---|---|
| MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`) | MANIFESTS demands it for any decode-path change; **not run** | within noise of `combined` on R/P/C/K, and text byte-identical |
| `scripts/gate-prefill-logits.sh` | needs a base; none exists for this model | record the base **from the delivery tip** (one build of the unmodified tree) first, then run — recording from the modified build is vacuous |
| the ranking default, decided on **single-user cells only** | the batched utility is a gate: it is prefill-dominated and its sign inverts against the real workload (see below) | `llama-cli` single-stream prose at >= 2 budgets, 3 alternating reps, plus `llama-server` at 1 slot |
| `-npl 1,4,8` batched-bench | **gate only** — "did we regress the multi-slot path?", never "what should the default be" | recorded as an observation; a regression there blocks a default flip, it does not choose one |
| `floor` on both cells | the only candidate that can serve single-user and parallel | wins most of `decode`'s single-user gain and recovers `combined` on the 4-slot path |
| `ENVIRONMENT.md` | rule 9: `MOE_HOST_POOL_RANK`, `MOE_HOST_POOL_RANK_FLOOR`; `invert` must not ship | entry written, classes assigned |
| default-on decision | repo policy is "beneficial ⇒ on, env only disables" | maintainer's call; `floor` is the defensible flip value because of the staging duty |

**Decision-set rule (added 2026-11-06, after a methodology error).**  The campaign's decision inputs are the
**single-user cells**: `llama-cli` single-stream (the cell S2's +10.7 % was measured in) and `llama-server` at
1 slot.  `llama-batched-bench` is a *gate* on the multi-slot path and is never a tie-breaker.  The reason is
measured, not stylistic: at `-lme 36` the ranking is **+10.7 %** single-stream and **-11.4 %** at 4 batched
slots, because `npp 512 -ntg 128 -npl 8` is prefill-dominated (4 096 prefill per 1 024 decode tokens) and the
pool's content is one early decision that `combined` heat predicts better under that mix.  Quoting a batched
number as evidence *for* the ranking is what produced the retracted +20.8 % in `P1-POOL-RANK.md`.

### W1 — instrument the swap's marginal cost (`MOE_HOST_POOL_SINK_TAX=1`, diagnostic)

The demote adds **exactly one D2H per admission** (the H2D to the arena already happens on that miss, and the
slot is the one the promoted expert vacated).  So the marginal cost can be measured *without* implementing the
policy: on every admitted miss, also issue one 1-D D2H of `expert_bytes` into a scratch pinned buffer, and
change nothing else.

* Arms: `SINK_TAX=0` vs `1`, same binary, same config, `MOE_HOST_POOL_DIO=1` so misses are real.
* Report: decode t/s delta at two pool budgets (a below-complement one and a full-complement one), plus the
  observed admissions/token, so the cost is quoted **per copy** and **per token**.
* Measurements under `scripts/gpu-watch.sh --cmd`, 3 runs per arm in both orders (the P1 protocol).
* **Go/no-go:** cost ≤ ~1.5 ms/token at the full-complement cell ⇒ build W2.  ≥ ~3 ms/token ⇒ stop, close the
  campaign with "buy the RAM" and hand the analysis to the PR.

### W2 — implement the demote (`MOE_HOST_POOL_SINK`, default **0** until gated) — **GO on cost, BLOCKED on slot policy**

> **Precondition found 2026-11-06, before any W2 code.**  S7 is false.  Tracing the miss path
> (`expert_host_src` → `pool_peek_locked` | `pool_fill_empty_locked`) shows the pool slot that serves an
> admission holds the **incoming** expert's bytes until its H2D drains, and the same H2D has already
> overwritten the victim in the arena.  So "the victim takes the slot the promoted expert vacated" cannot be
> implemented as stated: the victim's copy must be issued *before* the fill, into a slot that is free at that
> moment.  Consequences, in order of severity:
>
> 1. **The demote is 2 copies, not 1** (arena → staging → pool slot, ≈19 µs at the measured 9.6 µs/copy), so
>    the break-even reuse roughly doubles (~6 % on this box's NVMe, still ~0.4 % on a 125 MB/s device).  W1's
>    cost stands as a *bound*: the demote cannot be cheaper than one copy.
> 2. **The pool has nowhere to put a victim in the two clean cases.**  In the exact-complement case it is full
>    of residents *and* coverage is exact, so there are no holes to fix; in the surplus case (`keep_all`) it
>    mirrors residents and coverage is again near-exact.  The demote only has work to do where the prior is
>    *wrong* (our measured 79 % coverage at a 94 % budget), and there the only slot a victim can take belongs
>    to another expert — i.e. a **pool eviction policy**, exactly the "churning cache" §87 rejected when it
>    measured the periodic variant.
> 3. So W2 is not a 150-line local change: it is a **pool slot-policy decision** plus a staging path.  The
>    open question for the maintainer is which of these to implement: (i) staging + evict the pool's
>    *worst-value* resident on demote (on-demand churn, which S10 says is 4-5x more copy-efficient than the
>    periodic variant it rejected); (ii) staging + demote only into genuinely free slots (cheap, but covers
>    only the transient not-yet-full window); or (iii) stop, and take "buy the RAM" as the answer for the
>    constrained box, since the demote's cost has doubled and its slot policy contradicts a standing design.
>
> **RESOLVED (2026-11-06) — the maintainer's design, and it is better than all three options.**  Put the
> bounce buffer *inside* the pool: plan K slots of **exchange headroom** below the budget (`slots -= K`, so
> nothing grows), let a demote write the victim straight into a free slot (ONE D2H, no staging, no second
> copy), and retire the promoted expert's slot back into the free list only when the headroom has run dry
> (`pool_free_slot_locked() < 0`).  That keeps #87's keep-the-duplicate win in the steady state and makes the
> duplicate-retirement harmless when it does happen, because the vacancy is refilled by a just-evicted hot
> expert instead of being left empty.  Implemented as `MOE_HOST_POOL_SINK` + `MOE_HOST_POOL_SINK_SPARE`
> (default 2).  First run at `-lme 36`: **3 712 demoted (2 401 MiB), 6 497 slots retired, 4 983 declined** —
> the decline rate (57 %) is the open tuning item: incoming misses also draw on the headroom, so K=2 per
> table is not enough to keep the rotation ahead of the fills.
>
> **MEASURED (2026-11-06), `-lme 36`, single-user, pool active, text byte-identical at `880c932888f8`:**
>
> | arm | gen t/s | vs 22.6 | demoted | declines |
> |---|---|---|---|---|
> | `combined`, no demote | 22.6 | -- | -- | -- |
> | `decode`, no demote | 25.1 | +11.1 % | -- | -- |
> | `combined` + demote K=16 | 24.3 | +7.5 % | 9 922 | 9 % |
> | `decode` + demote K=8 | 27.0 | +19.5 % | 8 921 | 15 % |
> | **`decode` + demote K=16** | **27.5-27.6** (3 runs) | **+21.7 %** | 10 110 | **1.8 %** |
> | `decode` + demote K=32 | 27.0 | +19.5 % | 10 398 | 0 |
>
> The two changes **compose**: the ranking decides what the pool holds at layout time, the demote decides
> what it learns from evictions.  K=16 is tuned, not derived (16/512 = 3.1 % of a table's slots); K=32
> removes the last declines and costs 0.5 t/s because the reserve's coverage loss overtakes the extra
> demotes.  Two bugs had to be fixed first, and both looked like "the knob does nothing": (1) reserving
> headroom by *shrinking `slots`* reserves nothing -- `slots` is the capacity everywhere, so the fill paths
> just filled the smaller pool, which is why declines were identical at K=2/4/8 and *rising*; the reserve
> must be a `keep_free` argument that fills respect and only the demote may dip below.  (2) A miss frees one
> slot but wants two (demote + incoming fill): `retired == demoted + declined` to the digit.  Under the sink
> the incoming expert is refused an empty slot and arrives via the pinned bounce ring.
>
> **Reuse measured, and the decay swept (2026-11-06).**  A per-slot flag marks demote-sourced content, so
> the run reports how often a demoted victim is wanted again: **7.9 % at the default
> `MOE_EXPERT_CACHE_PERIOD=32`**, i.e. well above W1's ~3 % break-even on this box's NVMe (0.2 % on slow
> storage) -- the mechanistic reason the demote is worth +7.6 % on top of the ranking, not a coincidence.
> Sweeping the decay at `decode`+K=16: 32 -> **27.6 t/s** / 7.9 %; 64 -> 27.5 / 12.0 %; 128 -> 26.9 /
> 16.1 %; 256 -> 26.1 / 20.9 %.  Refined 16-56 by 4 to locate the knee (2 runs each at 24/28/32):
> 16 -> 27.0, 20 -> 27.1, 24 -> 27.2, 28 -> 27.2, **32 -> 27.6**, 36 -> 27.3, 40 -> 27.4, 44 -> 27.3,
> 48 -> 27.3, 56 -> 27.3.  So it is a **plateau from ~32 to ~64 with 32 at its top edge**, and the fewest
> demotions of any fast configuration (10 110 against 11 798 at 64).  A longer half-life raises reuse but costs throughput *and* raises the
> demotion count, so the extra reuse is low-value churn.  **32 stays.**

> **Do not read "fewer demotions" as "fewer disk reads".**  `fills` is **60 513 in every period** from 16 to
> 128 -- identical to the digit -- so the pool's own fill traffic does not move with the decay; what moves is
> arena churn.  And a *rejected* admission is not a read avoided: it is served cold **per access**, for as
> long as the expert keeps being routed, which is why period 16 has both the fewest demotions (8 683) and the
> lowest throughput (27.0) -- a fast decay shrinks `ghost`, the `touch`-2 gate rejects more, and those
> experts are re-read on every use instead of admitted once.  Fewer demotions is better **at equal
> coverage**, which is why 32 beats 16 as well as 56.  Unproven part: the logs do not emit the cold counters
> (`takeover=`/`cold_reaches=` are absent at this verbosity), so the cold-traffic half of this argument
> rests on the mechanism plus throughput, not on a printed counter.  Making the report print `cold_reaches`/
> `cold_bytes` unconditionally and re-running 16 vs 32 would settle it.
>
> **Batched gate re-run WITH the demote (2026-11-06)** -- the ranking's 4-slot loss is only half repaired, so
> the default decision splits in two.  `-lme 36`, `-lzm auto`, npl 1/4/8:
>
> | arm | npl 1 | npl 4 | npl 8 |
> |---|---|---|---|
> | `combined`, no demote | 26.22 | **50.07** | 33.72 |
> | `decode`, no demote | 24.89 (-5.1 %) | 43.46 (**-13.2 %**) | 33.64 |
> | `decode` + demote | 25.21 (-3.8 %) | 45.90 (-8.3 %) | 33.53 |
> | `combined` + demote | **26.70** (+1.8 %) | 49.60 (-0.9 %) | 33.81 |
>
> * **The demote is safe in every cell** (+7.5 % single-stream, +1.8/-0.9/+0.3 % at npl 1/4/8) -> a
>   defensible **default-on** with an env kill-switch, once its gates pass.
> * **The ranking stays opt-in**: +11.1 % single-stream but -13.2 % at 4 slots.  Its sign is workload
>   dependent and the demote only recovers half of the 4-slot loss.  That conclusion survives the demote.
>
> **Methodology trap, recorded because it cost a whole measurement round:** the batched arm must not run
> `-lzm off`.  PLE becomes resident (~24 GiB) and with a 36.8 GiB pool + 23 GiB arena the box
> over-subscribes, so `npl 8` collapses (33.7 -> 8-18 t/s, MemAvailable min 46 -> 18 GiB).  The collapse
> looks exactly like a policy regression.  `arm-batched.sh` therefore defaults `LZM=auto`; the single-user
> arms keep `-lzm off` because there the PLE confound matters and the memory does not.
>
> **DECISION (2026-11-06, maintainer): both ship ON by default, with kill switches.**  The reasoning is
> recorded because it is a judgement, not a measurement: +21.7 % single-stream outweighs -5.1 %/‑13.2 % at
> npl 1/4, and a single-user stream is the common case.  **`npl 4` is an accepted limitation**, to be written
> up as such at promotion (not hidden).  Implementation on the fork:
> `MOE_HOST_POOL_RANK` default `decode` (`combined` = kill switch), `MOE_HOST_POOL_SINK` default on
> (`0` = kill switch), and the headroom derived as **4 % of a table's slots** (floor 2, cap slots/8,
> `MOE_HOST_POOL_SINK_SPARE` absolute override, -1 = auto).  Verified with no env vars at all: **27.5 t/s
> twice** (old defaults 22.6), 268/269 declines, 7.7-7.9 % reuse, text `880c932888f8`; the kill switches
> restore 22.6 exactly.  Note the fraction must be *of slots*, not of experts: at `-lme 36` a table has
> ~393 slots, so K=16 is 4.1 % of slots -- 3 % under-shot to 11 slots and 26.8 t/s.
>
> **Mirror invariants (2026-11-06, requested).**  (1) **spare = 0 and the exchange INERT** when a table's
> pool can hold every expert (`slots >= n_experts`): coverage is exact, so the copies are pure cost -- and,
> worse, a demote there would find no free slot, ask for one, and **retire a resident, destroying the
> mirror**.  `pool_sink_active()` gates the demote, the incoming-fill competition and the retirement on
> `slots < n_experts`.  (2) **pool bytes + headroom never exceed a full mirror**: `slots` is clamped to
> `n_experts` at sizing, so the headroom is always drawn from that budget.  Verified: `-lme 36` -> 56 592
> slots / 36 815 MiB / 10 033 demoted / **27.5 t/s**; `-lme 48` -> 73 728 slots (144x512) / **47 962.5 MiB =
> the whole host expert set** / 0 demoted / 0 retired / 144 mirror tables / **31.2 t/s**.  So a mirror that
> fits is still the fastest configuration, as it must be -- and 31.2 is the asymptote W3's curve should
> approach.
>
> Owed before promotion: `MOE_HOST_POOL_RANK=invert` must be REMOVED (a diagnostic, must not ship);
> `ENVIRONMENT.md` entries for `MOE_HOST_POOL_RANK`/`_FLOOR`/`SINK`/`_SPARE`/`_SPARE_PCT`/`_SPARE_MIN`
> (classes: kill switch / tuning) with `SINK_TAX` excluded; the `INVARIANTS.md` r113 "no write-back" entry
> corrected in place; and the reserve as a *fraction* is now done.; an `allow_empty_fill` A/B
> now that the reserve protects the headroom; the batched gate; the MTP four-axis; the prefill-logit gate;
> and **W3**, the `-lme` 30->47 sweep, which is the only place the "does the demote buy back pinned bytes"
> question is actually answered.

Design constraints, all of which come from something measured:

1. **Trigger** — in `access_locked`, on an admitted miss with a victim, when the table is pool-backed and a
   pool slot is available (the vacated one first, then a free slot, then nothing).
2. **Direction and ordering** — copy the victim D2H into the pool slot *before* the incoming H2D overwrites the
   arena slot, on the compute stream.  Use the module's kernel-copy form, not `hipMemcpyAsync` (the PR's
   finding: a small pinned copy can leave later queued work unsubmitted on ROCm, which stalls a peer past its
   all-reduce spin budget).
3. **Retire, then publish** — the slot is `slot_state == 2` (retiring) until the copy's event fires; only then
   is `expert_slot[victim]` published.  One ordering state, reused, not a second one.
4. **Decline, never lie** — no available slot, a failed record, or a geometry it cannot express ⇒ **decline the
   demote** (drop the victim as today).  A failed demote must not take the op over — the same rule
   `gate-host-pool.sh` already enforces for a failed fill (issue #274).
5. **1-D only.**  `split_axis == -1 && n_slices == 0`.  A `-sm tensor` axis-0 slice is the 2-D strided copy
   that measured 2× decode in r113; it declines and behaves exactly as today.
6. **No new memory.**  The demote may never grow the pool; if the bookkeeping cannot find a slot it declines.
7. **Counters** — sinks, declines (per reason), bytes; reported once at teardown beside the pool line.

### W3 — the three-arm split curve (the campaign's answer)

One binary, one model, `MOE_HOST_POOL_DIO=1`, **`-lzm off`** (see §4), `gpu-watch.sh` on every arm:

* **arm A** `MOE_HOST_POOL_RANK=combined` (today)
* **arm B** `MOE_HOST_POOL_RANK=decode` (P1)
* **arm C** arm B + `MOE_HOST_POOL_SINK=1`

swept over `-lme` **30 … 47** (below complement → mirror) at two arena shapes (a small one and the field one),
reporting **pinned GiB vs decode t/s vs storage MiB/token vs copy MiB/token**.  Plot the three curves with the
mirror as the asymptote.

**Accept for C to ship:** it beats B by ≥ ~12.8 GiB of equivalent pinned RAM (i.e. reaches B's L3 at a budget
≥ 12.8 GiB smaller), **or** it reaches zero storage reads within ~2 GiB of the tier's ~29 GiB point; and at the
mirror it is **inert or disabled** (S3: there is nothing to win and the copies are pure cost).
**Accept for B to ship:** it is ≥ 5 % better than A at the below-complement budgets and no worse anywhere —
which the P1 record already suggests.

### W4 — prefill read granularity — **DONE 2026-11-06** (see `W4-SPAN-READS.md`)

Coalesced the per-expert prefill reads into one read per run-of-missing-experts: **+81 % prefill at `-lme` 12,
+57 % at 18, +14 % at 36**, generation unchanged, text byte-identical, 0 fallbacks, no GPU faults.  Default on
(`MOE_HOST_POOL_STAGE_SPAN=0` = kill switch).  The record also corrects where the reads were (`moe_cache_stage_table`
never ran; the gather pre-pass and the pool prepopulate are the sites) and found that every ggml counter line
needs `-lv 4`.

### W4b — the mmap-only warmer (optional, deferred)

`MOE_HOST_POOL_WARM`: on arena eviction, `posix_fadvise(POSIX_FADV_WILLNEED)` on the victim's file range.  No
thread (the tree already does this in `llama-lazy-reader.cpp`, `llama-mmap.cpp`, `qwen4exp.cpp`), **armed only
in buffered fill mode** (`pool_decide_dio_locked` resolves it), self-disabling on: warm-hit rate ≈ 0 over N
evictions, warmed pages found non-resident (`mincore`), or reclaim pressure (`/proc/vmstat` `pgscan`/`pgsteal`)
rising while armed.  Long window, one-way latch per session — the idiom `g_pinned_off` / `g_heat_dead` already
use.  **Never presented as a fix for the constrained case**: it moves latency, not bytes (S8).

### W5 — promotion path — **started 2026-11-06**
### D1 reopened (2026-11-06, W4 session)

The ranking's `+11.1 %` was **prior leakage**: it was measured against a heat map that the session's own short
benchmark runs had rewritten, so the pool had been ranked for the test prompt.  On a frozen real-world map the
same decomposition at `-lme 36` is **control 24.3 → ranking only 24.6 (+1.2 %) → demote only 25.6 (+5.3 %) →
shipping 26.2 (+7.8 %)**, while the batched gate still costs the ranking `-11.5 %` prefill at npl 4.
**Recommendation: default `MOE_HOST_POOL_RANK=combined` (demote stays on).  Maintainer's call.**

### W5 — promotion path, if and only if one arm wins

Fold into the owning block (09 for the cache, 12 for the loader/pool policy), regenerate `patches/` from a
canonical tree via `scripts/make-patches.sh`, `validate-set.sh`, per-block gate, then the full release ladder.
New env vars → `ENVIRONMENT.md` with their class (`SINK`/`RANK` as kill switches once default-on). Every
invariant this campaign changed (the r113 "no write-back" entry) gets **corrected in place** in
`INVARIANTS.md`, with the r-number of the release that did it.

## 4. Honest caveats carried into every number above

* One trace (Flash-Next GSQ-RCO IQ3_S, prose prompt, 806 430 decode reaches) and one arena ratio (19 GiB ≈ 43 %).
  The 2-D grid says the tier's advantage **grows** as the arena shrinks, so +12.8 GiB is conservative for small
  arenas.
* `c_pcie = 0.04 ms/MiB` is an assumption until W1 measures it.  It is the whole decision.
* **`-lzm off` is mandatory for every decision arm** (2026-11-06).  Under `-lzm auto` the CPU-offloaded PLE
  tensor is mmap-lazy and page-fault loaded *during* the measurement, competing for the page cache the pool's
  `MOE_HOST_POOL_DIO=1` makes it bypass; the same single-user cell moved from +10.7 % (`auto`) to +9.6 %
  (`off`).  Corollary: **process `read_bytes` is not comparable across lazy modes** (78 GB vs ~102 GB on the
  same cells) — quote the pool's own DIO counters, never the process total.
* Every model number is a **hit/byte** model: the device measured P1's effect at +10.6 % while the model
  predicted 12–19 % of the *L3 traffic* — same direction, different currency.  Judge arms on device
  measurements; use the model only for ranking candidate policies.
* The persisted prior is **flat** for this model+workload (S1).  Results here may not transfer to a model whose
  heat map is well-conditioned — which would *shrink* the demote's case, since its advantage is foreknowledge.

## 5. Risks

* **A hard machine lockup (2026-11-05)**: a R9700 pinned at 100 %, power-cycled, **no kernel-journal record**.
  No benchmark from this session was running.  `HUNG_TASK` detection is not in this kernel
  (`# CONFIG_DETECT_HUNG_TASK is not set`), so every GPU arm runs under `scripts/gpu-watch.sh --cmd` with a
  bounded `timeout`.  Until the cause is known: **single-device arms only**.
* **`-sm tensor`**: any demote there is a 2-D strided copy — the r113 2×-decode mechanism.  Declines by design.
* **CUDA graph capture**: a demote is a stream write inside a captured graph unless it is only taken on the
  eager path.  Decide before coding: the sink is a decode-band decision, and the band is not captured for the
  eager path — verify, do not assume.
* **Windows**: the demote itself is portable (no `mmap` alias, no `mlock`); the warmer is POSIX-only and must be
  a no-op there, like the existing stubs.
* **The mistake pattern of this session, worth naming**: three times a plausible number turned out to be my
  own arithmetic (pool_h read as "fraction answered", the copy volume divided by the wrong token count, a K
  knob silently ignored by a failed patch).  Every claim in this plan that survives must lean on a printed
  counter or a device run, not on a hand-divided column.

## 6. Where the records live

| what | where |
|---|---|
| the campaign's index entry and status | `wip/CAMPAIGNS.md` |
| the scoping note (what was open before) | `archive/work/host-expert-split/README.md` |
| the PR review + convergence design | `archive/work/host-expert-split/DESIGN.md` |
| the evidence (trace, replay, surface, pricing) | `archive/work/host-expert-split/STEP1-REPLAY.md` |
| the one implemented arm + its gates | `archive/work/host-expert-split/P1-POOL-RANK.md` |
| the fork arm | `~/llama.cpp` branch `p1-pool-rank-r223` (`b2a8d2e76`) |
| the replayer's new knobs | `scripts/moe-trace-replay.py` (`pool-layout`, `--pool-band`, `--rebalance*`, `--evict-warm`) |
| the flight recorder | `scripts/gpu-watch.sh` |
