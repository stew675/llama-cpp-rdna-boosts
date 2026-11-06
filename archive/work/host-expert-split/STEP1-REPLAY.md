# Step 1 — the offline replay, PR #267 vs our layout (2026-11-05)

Status: **DONE.  The sink is worth it exactly where an arena miss would be expensive, and worth nothing
where the two tiers already cover the set.**  Nothing implemented, nothing in `patches/`.  Design:
[`DESIGN.md`](DESIGN.md).

> **Correction to an earlier draft of this file.**  It said "most of the sink's apparent win is a tool
> artefact".  That was wrong, and the mistake is worth keeping visible: the artefact is real but *narrow* —
> it only bites when the pool budget exceeds the arena's complement (see §2) — and the sink's separation at
> budgets *below* the complement is genuine.  The fix is a new policy, `pool-layout`, which is now the
> faithful model of the delivered cache.

## 1. The trace, and the model's calibration

One bounded run (`gpu-watch.sh --cmd`, 1 R9700, no all-reduce, no drafter, `timeout 1200`; rc=0 in 83 s):

```
MOE_EXPERT_CACHE_MIB=4096  MOE_EXPERT_TRACE=/tmp/fn-prose.trace  MOE_EXPERT_TRACE_MIB=1024
-cli -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf -ncmoe 48 -ngl 99 -lme 4
     -ot 'per_layer_token_embd.*=CPU' -lm mmap -lzm auto -fa on -c 8192 -ub 4096
     -f prompts/prose-rdna-boosts.txt -n 512 --seed 42 --temp 0 --no-display-prompt --single-turn
```

144 tables × 512 experts, `expert_bytes = 627200` (the full 46.84 GiB host set, one device), 74 016 decode
records (514 passes) + 564 prefill records, 33.7 MB.  Run: prompt 5298 tok at **121.0 t/s**, generation
512 tok at **23.65 t/s** (a deliberately starved arm: 4 GiB arena + 4 GiB pool).

**Calibration.**  Replay at the traced geometry gives `h = 0.6117` against the run's own turn counter
`MoE arena = 0.6300 (468924/744360)` — 2.9 % low.  The `h=0.0133` in `moe_cache_report` is the **pool's**
hit rate, not the arena's; the model's `pool_h = 0.0745` is the same order.  Two counters, never mixed.

**A second, independent calibration.**  At a 100 % pool budget the new `pool-layout` policy reads **0.00
MiB/token of L3** — which is what `wip/bench-56gb/RESULTS.md` measured for the mirror ("0 MiB of drive
reads").  The old `pool-none` read 16.6 MiB/token for the *same* budget, and that is the artefact.

## 2. What was wrong with the tool

`--pool-init sidecar-complement` seeds the arena from the prior and then gives the pool only the arena's
**non-residents**.  That is right while `slots_p <= |complement|`, and wrong above it: at a 100 % budget it
filled 292 of 512 slots and left 220 **empty**, which the shipped cache never does — `pool_rerank_from_heat_locked`
fills the remaining capacity with the arena's **mirrored residents** ("the complement first, then the arena's
residents in the same rank order while capacity remains").  The empty slots were then filled one at a time by
L3 misses, so the pool never caught up and the residual L3 was attributed to the policy.

New policy **`pool-layout`** = the delivered cache: heat-seed the arena, lay the pool out as the complement
**plus mirrored residents**, and re-lay it out at the first scored pass from the arena's *actual* residency
(one relayout, because the demanded path never evicts).  Use it for every "ours" row from now on;
`pool-none`/`pool-retire` are lower bounds and should be labelled as such.

| arena 19 GiB | `pool-none` | `pool-layout` | `pool-swap-slot` |
|---|---:|---:|---:|
| frac 50 (below the complement) | 26.1 | 26.1 | 6.7 |
| frac 80 (above it) | 17.0 | **10.3** | 0.0 |
| frac 100 (mirror) | 16.6 | **0.0** | 0.0 |

MiB/token of L3.  Below the complement the two agree (the pool is full of the complement either way), so the
frac-50 comparison was always honest; above it, only `pool-layout` is.

## 3. The surface — arena ratio x pool budget (MiB/token of L3)

`--admit touch --touch 2` (the shipped admission).  `ours` = `pool-layout`, `sink` = `pool-swap-slot`.

| arena | frac 20 | frac 30 | frac 40 | frac 50 | frac 60 | frac 80 |
|---|---|---|---|---|---|---|
| 4 GiB (8 %) | 286 / 274 | 238 / 224 | 194 / 177 | 148 / 128 | 106 / 85 | 45 / 21 |
| 10 GiB (23 %) | 163 / 152 | 134 / 118 | 105 / 85 | 79 / 55 | 58 / 30 | 30 / **0** |
| 19 GiB (43 %) | 90 / 80 | 67 / 53 | 47 / 30 | 32 / 12 | 22 / **0.4** | 4 / **0** |
| 26 GiB (59 %) | 50 / 42 | 33 / 21 | 20 / 6 | 10 / **0** | 2.5 / **0** | 0.1 / **0** |

Your intuition was right on both counts: **the win grows as the arena shrinks** (the sink removes 24-25
MiB/token at a 4 GiB arena against 12-22 at 19 GiB) and **it vanishes as coverage completes** — at 26 GiB the
sink is worth nothing at frac >= 50, and at 19 GiB it is worth nothing at frac >= 60.

**The trade is one constant across the whole surface: ~1 MiB/token of L3 removed per ~1.7 MiB/token of extra
PCIe copies.**  That number is the decision:

* a page-cache miss costs ~0.12 ms/MiB (RAM read + H2D) against ~0.04 ms/MiB for the extra copy, so the swap
  is worth ~`0.12 - 1.7 x 0.04 = 0.05 ms` per MiB removed — at 19 MiB removed, ~1 ms on a 12.8 ms token (~+8 %);
* a **disk** miss on the 56 GiB rig costs 8.4 ms/MiB (125 MB/s), so the same trade is a landslide — and it is
  why the PR exists;
* at a budget that covers the set there is nothing to remove, and the sink pays 5-6 MiB/token for nothing:
  **r113 reproduces, and so does Brian's own -3 %.**

## 4. The touch counter is not a detail — it becomes load-bearing

`MOE_EXPERT_CACHE_ADMIT` (`always|value|touch`) and `MOE_EXPERT_CACHE_TOUCH` (default **2**, floored at 2) are
shipped; the PR adds its own `GGML_MOE_TIER_TOUCH`, default **3**, to damp a tier's churn.  The replay says
the two interact with the sink in the opposite direction from what the PR assumes:

| arena 19 GiB, frac 50 | `ours` | `sink` | sink's advantage |
|---|---:|---:|---:|
| `always` / `value` | 26.1 | 6.7 | **19.4** |
| `touch 2` (shipped) | 31.7 | 12.3 | **19.3** |
| `touch 3` (the PR's tier default) | 32.8 | 19.8 | **13.0** |

A stricter admission raises L3 for *both* arms (16.5 -> 20.3 -> 12.5 MiB/token of advantage at arena 10 GiB,
frac 40) and **shrinks the sink's advantage by a third**.  The reason is structural: in our cache a refused
admission is *free* (the pool is a layout), whereas once a demote exists every admission writes a victim into
RAM, so the gate stops being free and `touch 3` becomes coherent — but only as a way to bound the *copy*
volume, never as a way to save L3.  **C1 must therefore be measured with the admission rule as a swept knob,
not inherited.**  (`--admit value` measured identical to `always` on this trace, which is suspicious enough
to check before relying on it.)

## 5. Verdict

1. **There is a real benefit to swapping** — in the band where the pinned budget cannot cover the arena's
   complement and a miss is expensive.  It is not a throughput win at the budgets we ship (a mirror or a
   near-complement pool), which is where r113's "2x decode for 0 MiB of L3" was measured and where it stands.
2. **The honest framing is bytes per token, not tokens per second**: a sink buys ~1 MiB of disk read away per
   1.7 MiB of PCIe copy.  On a page-cached box that is worth ~8 %; on a disk-bound box it is decisive; on a
   covered box it is negative.
3. **The next number is the copy cost on this tree**, single device, 1-D — the only thing that can turn the
   table into a decision.  That is step 2 (C1: one call site plus a one-spare floor, `DESIGN.md` §4), and it
   must be run on the 56 GiB rig with `-lme` below the complement, `touch` swept, and the mirror row as the
   control.
4. **Not yet explained, and worth a look before C1 ships**: our arena holds 93.5 % at 43 % capacity while the
   pool — a 50 %-of-the-set prior-ranked complement — only answers 56 % of the arena's misses, where the
   coverage arithmetic predicts ~88 %.  Either the persisted prior is much weaker than the arena's own F-LRU
   tracking (in which case a *recency*-tracked tier's advantage is really "a better prior", not "a swap"), or
   the replay's arena drift is not what the real cache does.  Deciding which is the difference between
   "implement a demote" and "improve `pool_rerank_from_heat_locked`".

## 6. Addendum — the maintainer's periodic conditional swap, and a denominator bug in §3's prose

**Proposal (2026-11-05):** every N rounds, swap each table's *coldest arena resident* with its *hottest
non-resident pool expert*, but only if the pool expert is hotter.  Then a variant with K = 1..4 swaps per round.
Implemented in the tool as `--rebalance N --rebalance-k K --rebalance-heat carry|reset|prior|blind` (the guard's
comparison heat is selectable because the shipped bookkeeping does not carry it — see below).

**The bookkeeping answer (asked directly).**  In the *shipped* cache the hotness does **not** travel: on
admission `count[expert]++` and `ghost[expert] = 0`; on eviction the victim's counters are simply abandoned
(its `count` keeps decaying, its `ghost` stays at the 0 from its last admission).  My first implementation
transferred it (`a_count <- max(a_count, a_ghost)` on promote, `a_ghost <- max(a_ghost, old count)` on
demote).  On this trace **`carry`, `reset` and `prior` produce byte-identical runs** — after a few decay
periods every resident's `count` sits at 0-1, so the guard collapses to "has this expert been demanded since
it left", and the transfer detail never changes a decision.  It is *not* inert, though: dropping the guard
entirely (`--rebalance-heat blind`) costs **3.3x the copies** (16.6 against 5.0 at N=1) for slightly less L3
saved.  So the "only if hotter" condition is doing real work; the counter-transfer subtleties are not.

| arena 19 GiB, frac 50 | L3 MiB/tok | copy/tok | L3 saved | MiB saved per MiB copied |
|---|---:|---:|---:|---:|
| shipped (layout, no swap) | 31.65 | 0.00 | — | — |
| periodic N=1, K=1 | 25.46 | 51.99 | 6.19 | **0.119** |
| periodic N=3, K=1 | 27.89 | 25.29 | 3.76 | **0.149** |
| periodic N=3, K=4 | 26.81 | 42.33 | 4.84 | 0.114 |
| sink (on demand, `pool-swap-slot`) | 12.33 | 29.93 | 19.32 | **0.646** |

Same ranking at (19 GiB, frac 40) and (26 GiB, frac 40); the sink's efficiency is remarkably stable at
**0.65** everywhere, the periodic policy's at **0.11-0.18**.

**Verdict on the proposal.**  *Batching does not swing the balance — it slightly hurts it*: at a fixed N,
K=4 buys ~1 MiB/token more L3 for ~17 MiB/token more copies, i.e. a worse ratio, because each additional
swap in a round acts on information the previous swap just invalidated.  And the periodic shape is **4-5x
less copy-efficient than the on-demand sink**, for a structural reason: a proactive swap pays 2 copies for an
expert that may never be demanded again, whereas the sink's extra copy happens exactly when a miss proves the
demand exists.  What the periodic shape *does* buy is latency shaping — the copies are spread over round
boundaries instead of landing in a burst on the miss — which matters only if a single swap's latency is
visible in a token, and at 0.6 MiB per copy on one device it is not.

**Correction to §3's prose (a denominator bug of mine, not the tool's).**  I first divided the copy volume by
the trace's *total* token count (5813) instead of the decode-band count the tool uses (~881), which inflated
every "MiB saved per MiB copied" figure ~5.7x and inverted the ranking above.  The tool now prints
`copy/tok` / `link/tok` computed from its own denominator (`handover_mib_per_tok`, `hostlink_mib_per_tok`);
**quote those columns, never a hand-divided one.**  §3's "1.7 MiB of PCIe per MiB of L3 removed" is the
*sink's* figure and stands (1/0.646 = 1.55); the periodic policy needs **5.6-8.4**.

**Converted to time** (`c_l3` 0.12 ms/MiB page-cache, 8.4 ms/MiB disk; `c_pcie` 0.04 ms/MiB):

| arm, arena 19 GiB frac 50 | page-cached box | 56 GiB rig (disk) |
|---|---:|---:|
| sink | +1.1 ms/token (~+9 %) | +161 ms/token |
| periodic N=1 | **-1.3 ms/token** | +50 ms/token |

So the periodic swap is negative on a box where the miss was already cheap, and positive but dominated on a
box where it is not.  The sink is the shape to build; the periodic variant is not worth a second knob.

## 7. The anomaly, resolved: the pool is ranked by a counter that is worse than chance for decode

§5.4 left one number unexplained: the arena holds 93.5 % of reaches at **39.5 %** of the set (202 of 512
slots/table at a 19 GiB budget), while the pool — a 50 %-of-the-set, prior-ranked complement — answers only
**56.5 %** of the arena's misses, where covering 256 of the 310 complement experts predicts ~83 %.

**Two of my own errors first, because they framed the question wrongly.**

1. `pool_h` is `p_hits / misses`, and `p_hits` counts only misses **promoted from the pool**.  A miss whose
   expert is pool-resident but which is *rejected* by the admission gate is served in place and counted as
   `pool_cold`, not `p_hits`.  So under `--admit touch` the column is 0.088 at the very cell where
   `--admit always` gives 0.565 — neither is "the fraction of misses the pool answered".  The 83 % figure was
   never a like-for-like comparison.
2. The 83 % assumed misses are uniform over the complement.  They are not: the arena's misses are, by
   construction, the experts it does *not* hold, i.e. the cold tail.

**The actual measurement** (decode band, 806 430 reaches, 144 tables):

| reach mass by | decile 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| the persisted prior's rank (0 = its hottest 10 %) | 10.3 | 11.0 | 11.4 | 10.7 | 12.1 | 12.0 | 10.9 | 8.8 | 8.0 | 4.8 |
| expert id | 10.0 | 10.5 | 9.7 | 10.1 | 10.0 | 10.7 | 10.0 | 9.3 | 9.0 | 10.8 |

A decile of a **random** ranking is 10 %.  The persisted prior is therefore almost perfectly flat against this
workload's decode routing.  Overlap of the prior's top-K with the trace's *own* hot set, averaged per table:

| K | combined counter (what the pool uses) | decode counter (`heat_dec`, what the arena seeds from) | expert id | chance |
|---|---|---|---|---|
| 16 | 3.1 % | **9.5 %** | 3.5 % | 3.1 % |
| 64 | **10.5 %** (worse than chance) | **24.2 %** | 12.8 % | 12.5 % |
| 202 | 47.9 % | **54.8 %** | 40.9 % | 39.5 % |

**So: the arena's 93.5 % comes entirely from its own F-LRU tracking, not from the prior — and the pool is
ranked by the *combined* (prefill-dominated) heat counter, which is anti-predictive for decode**
(`pool_rank_experts_locked` sorts by `t.heat`; the design comment defends that because the pool also stages
whole tables for prefill — a real tension, but it means the decode half is being paid for by the wrong vector).

**What that is worth, modelled (arena 19 GiB, frac 50, no copies either way):**

| admission | pool ranked by `heat` (shipped) | pool ranked by `heat_dec` | change |
|---|---:|---:|---:|
| always | 26.13 MiB/tok | 23.32 | **-10.7 %** |
| touch 2 | 31.65 | 29.69 | -6.2 % |
| touch 2, frac 40 | 46.81 | 38.99 | **-16.7 %** |

i.e. **12-19 % of the decode L3 traffic for zero copies** — the cheapest candidate this campaign has produced,
and it ships on its own if it measures on the device.

**And it bounds the demote's value.**  With an *oracle* pool content the sink reaches 3.0 MiB/token
(touch 2) where no fixed layout gets below 23-30.  A layout is limited by the prior's ability to predict which
experts the arena will not hold; a demote gets that knowledge by construction.  So the residual gap is
**foreknowledge**, not transport — which is the same conclusion r113 reached from the other direction, now with
the number to go with it.

**Revised order of work.**  (1) The pool's ranking vector is its own change: rank the complement by
`heat_dec` (or keep a prefill-hot floor and rank the remainder by decode), A/B against the shipped vector on
both bands, and ship it alone if the prefill row is not regressed.  (2) The demote, if it is still wanted,
must be measured **against the decode-ranked layout**, not against the combined-ranked one — otherwise most of
its advantage is a bug in the baseline.

## 8. Addendum — write-back vs page-cache warmer (maintainer's question, 2026-11-05)

The question: an admitted expert's arena victim is dropped and re-read from storage later; D2H is far faster
than a storage read, so why not write the victim back — and if the profile is O_DIRECT, is a D2H swap not
*always* better?  And, separately: with mmap, could a background reader pull the dropped expert into the page
cache, so a later miss is a page-cache hit?

**The observation is correct and the code does exactly what was described.**  On a miss the arena victim's
bytes are abandoned (bookkeeping only, no write-back), and in the complement case the pool does **not** hold
the victim (it holds the complement), so it is genuinely lost to storage.

**Why the delivery does not do it — and why that reason does not cover this case.**  r113 implemented the
write-back (`pool-swap-slot`) and dropped it at **2× decode**, and `INVARIANTS.md` carries the rule.  Two
properties of that arm matter and neither holds here: the pool **covered** the complement, so the swap saved
**0 MiB** of L3 (nothing to win), and the copies were **2-D strided `cudaMemcpy2DAsync`** on a 2-device tensor
split (6 259 swaps).  The complement-scarce, single-device case is 1-D contiguous, and there the replay says
the swap removes ~65 % of the L3 at 1.55 copies per MiB removed.  **A write-back is slot-neutral**: the
promoted expert came from the pool, so the slot it vacates takes the victim — no extra memory, which is what
makes it structurally cheap.  It needs the ordering the pool already has (`slot_state == 2` retiring) plus a
copy-ahead on a side stream to keep the D2H off the admission's path (the PR's 0006 precopy).

**Your second proposal, measured.**  `--evict-warm` models it exactly: nothing is moved by us, each arena
victim is read into the page cache off the critical path, and its next miss is a page-cache hit.

| arena / frac (`--admit touch 2`) | no warmer | `--evict-warm` | change | device copies |
|---|---:|---:|---:|---:|
| 19 GiB / 50 % | 29.69 | **23.60** | **-20.5 %** | 0 |
| 19 GiB / 40 % | 38.99 | **32.87** | -15.7 % | 0 |
| 10 GiB / 50 % | 65.98 | **56.09** | -15.0 % | 0 |

**And that measurement makes the distinction that decides the design:**

* a **write-back** *eliminates* the storage read — it saves **bandwidth**, and costs PCIe copies;
* a **warmer** keeps the storage read and only takes its **latency** off the token path — it saves **no
  bandwidth at all**, and on a box where the fill path is bandwidth-bound (the 56 GiB rig measures ~125 MB/s
  effective) it competes with the critical path for the same storage queue.

So the warmer is close to free when there is spare storage bandwidth and spare page cache (a big box, where a
miss was already cheap), and it is the *wrong* instrument on a memory-tight box, which is the only place the
campaign cares about.  A write-back is the one that helps the constrained case.  Both are worth having, for
different boxes, and neither exists in the delivery.
