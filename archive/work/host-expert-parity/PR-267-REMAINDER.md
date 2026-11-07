# PR #267 — what is left, per patch (re-assessment 2026-11-06, against the r224 tree)

**Status: assessment + the r228 follow-up.**  Nothing of the PR is applied, and no patch of it can be applied
as-is (they are cut against r204 and touch code that moved in r209-r228; `archive/work/host-expert-split/DESIGN.md`
§5 has the port analysis).  **r228 changed the frame** (§3.0): the demote that 0004/0006/0008 target is now a
decode-class mechanism that only runs where it pays, so every remaining comparison is made at a budget *below*
the near-mirror boundary.  **Two mechanisms closed on 2026-11-06** (`soar`, 3 × R9700): 0004/0006/0007's hazard is
**unreachable by construction** and 0003b costs **≤0.2-1.2 % of a prompt** — [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md).
**The 2026-11-07 passes (`radix` + `soar`) closed the rest**: 0003a's narrow-2 preference is **measured / do not
implement** ([`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md)) and 0008's touch value was swept on
device with its default unchanged, which also closes 0002 as *not needed*
([`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md)).  So of the ten patches: 0001/0005/0010 absorbed or
superseded, 0002 closed as not needed, 0003a/0003b/0004/0006/0007 closed by measurement or by construction,
0008 measured with no change, and 0009 split off to `wip/host-expert-coverage/`.  **Nothing of the branch is
applied and nothing is outstanding**, which is what makes the close as *integrated rather than merged* honest.
This file exists so the PR can be closed as **integrated rather than merged**, and so a future session does not
re-review a mechanism we have already measured.

Read with [`DESIGN.md`](../host-expert-split/DESIGN.md) (the original review + the convergence
design) and [`archive/work/host-pool-speed/README.md`](../host-pool-speed/README.md) §4.3/§4.4/§6
(the earlier per-mechanism verdicts, still correct).  The r224 tree is what "our state" means below; the file
`ggml/src/ggml-cuda/moe-expert-cache.cu` is the single file every one of his patches edits.

## 1. The ten patches

| # | his mechanism | our state at r224 | verdict |
|---|---|---|---|
| 0001 | `--host-experts mmap` puts the `-ncmoe`/`-cmoe` experts back in the pageable mapping; GPU-keyed host-expert accounting | the **mode is gone and must not come back**: since r209-r212 the placement is one budget, `-lme [auto\|N]` (below the set = a per-expert pinned pool over the read-only mapping; at/above it = a mirror), and `MOE_HOST_TIER=tensor` is the whole-tensor escape hatch | **superseded** |
| **0002** | sparse registration: a second read-only `MAP_SHARED` alias, `mlock` + `hipHostRegister(ReadOnly)` for the expert ranges a GPU may read, `madvise(WILLNEED)`, trim after a device sync, `MAP_FIXED` remap of KFD's leftover range; owner-device routing; whole-table prefill staged from arena+alias | **not adopted, and now closed.**  The job is done by two mechanisms we already ship: a pool-resident miss is an **in-place PCIe read of pinned RAM** (`MOE_EXPERT_CACHE_COLD_UVA`, r203/r210) and a declined read goes through the per-table **bounce ring** (`expert_host_src`, the `gate-host-pool.sh` arms).  It works rather than merely existing: on the 2026-11-07 live-demote arms the cold path served **3.27-4.58 %** of all reaches (**15 147-21 206** misses) with **zero** declines, without registering anything.  `-sm layer` owner routing is the semantic key (r215+) | **CLOSED — not needed** ([`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md) §2) |
| 0003a | the narrow-2 (draft compute) pin prefers its `cudaMalloc` fallback when it fits **outside** the slab instead of evicting arena tables | **measured 2026-11-07 → `TODO` #279 CLOSED, do not implement**: the eviction branch is unreachable on every shape we ship (the pin's VA spare IS the draft's reserve minus the headroom, so it exceeds the compute region by construction), and when forced the eviction is re-absorbed — identical final arena MiB and decode hit rate in both arms.  The preference cannot avoid it (its fallback eats the free VRAM its own guard checks ⇒ the pin is retried 13 ms later with the same eviction) and a latched version trades the evicted bytes of cushion for one ~20 ms re-fill ([`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md)) |
| 0003b | `GGML_CUDA_SLAB_RING_PREFILL=0` (implied by mmap): keep the expert-cache arena across a decode→prefill transition | **absent** — no such variable exists.  The slab's H2D ring *is* reclaimed on the prefill→decode drop, and the arena **is** dropped going the other way when the work region outgrows its reservation (`moe_cache_evict_slab_range` + `re-armed N stood-down`) | **measured 2026-11-06 → `TODO` #278 CLOSED, do not implement**: 12 tables / 1 062 MiB per transition at the field ubatch, ≤0.2-1.2 % of the prompt and invisible in paired rows; 0 events on the 3-GPU tensor shape ([`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) §1) |
| **0004** | the swap is two custom **copy kernels**, not `hipMemcpyAsync`, because a small pinned copy can leave later queued work unsubmitted on ROCm | our demote is one `cudaMemcpyAsync(D2H)` on the compute stream into a free pool slot (block 12).  It measured +5.3 % net on r224's box, so no stall was visible there — but that box's link is not this one | **closed 2026-11-06 → `TODO` #276: unreachable by construction.**  The hazard needs a copy *and* a paired all-reduce; on `-sm tensor` (RCCL banner present) the demote declines **0 demoted / 14 480 geometry (structural)**, and on `-sm layer` (7 862 copies) there is no all-reduce |
| 0005 | tier × device-remap, and a tier table never uses the pool's cold read | ours already declines on a **partial** pool and serves a mirror (`dr_pool_ok`, r221) | **absorbed** |
| **0006/0007** | precopy: K spare RAM slots + a side stream that copies the policy's likely victims ahead of time; a failed event record drains the side stream, a failed wait fails the swap | **absent** (`MOE_HOST_POOL_PREFETCH`, the knob the earlier review named as the survivor, no longer exists).  Our demote only pays the D2H when the miss proves the demand | **retired 2026-11-06 → `TODO` #277**: its condition was "0004 leaves a residual on a multi-device arm" and 0004 closed as unreachable |
| **0008** | a tier admits from the **3rd** touch (`GGML_MOE_TIER_TOUCH=3`) to damp churn; 8 spares | the knob exists (`MOE_EXPERT_CACHE_TOUCH`, default **2**) and the spares are our exchange headroom.  **Measured 2026-11-07**: touch-3 damps churn exactly as the replay predicted (**-17.3 %/-18.3 %** demotions, **-27 %** sink uses) and the decode outcome is a wash (**+1.3 to +2.2 %** at 93.8 % coverage, inside this box's spread), for **-0.78/-0.81 pt** of arena hit rate and **+35 %** cold-served misses | **CLOSED — default stays 2** ([`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md)) |
| 0009 | **one** pin budget for tier + registration (`GGML_MOE_PIN_BUDGET_MIB`), a **partial tier by hotness**, bounded warm-up registrations, a refused registration not retried until the next trim | half landed elsewhere: `-lme` **is** the one budget (r209-r212) and since r207 a shrunken plan is a per-table **distribution**, not a tail shave, so a partial tier is already hotness-ordered by content.  The registration half is moot.  What is left is the *policy/visibility* half — and it is owned by [`wip/host-expert-coverage/`](../../../wip/host-expert-coverage/README.md) steps 1-3 (cgroup-aware `MemAvailable`, an authoritative explicit `-lme`, one visible `MIRROR/PARTIAL/NONE` verdict) | **split off, not here** |
| 0010 | a declined read goes through an **8-entry** per-table bounce ring (capped so a whole-master declined read cannot pin a copy of the master) | the ring exists and is **deliberately uncapped**: measured, a cap of 2/4/8 costs **8-25 % of prefill** at FRAC 80 (`bounce_acquire`'s comment), and r224's span coalescing addresses his *concern* (fewer, larger reads ⇒ fewer entries in flight) instead | **rejected, with a number** |

## 2. Where the mechanisms differ by design (ours is not a port of his)

Worth stating once, because a reviewer of the PR will ask:

* **In the saturated case ours *is* his swap.**  His `tier_swap_locked` (0004) sends the victim D2H into a
  **spare tier slot**, repoints `cold_ptrs[victim]` at it, and parks the promoted expert's vacated slot in
  `ram_pending` until `swap_ev` fires -- so his "spare" is just the promoted expert's old slot, one event later.
  Ours retires the promoted expert's pool slot, waits on `slot_state == 2`, and publishes the victim into it, and
  the measurement shows the pairing (`demoted == retired`: 9 933/9 933 at `-lme` 44, 10 037/10 681 at 36).  Same
  mechanism, same **one extra copy** per admission (the victim D2H; the incoming expert's fill happens anyway).
  The difference is the **accounting**, and it is the one that mattered:
  - his spares are **on top of** the budget (`ram_free.assign(1, j)`, K+1 slots per table with precopy), so the
    complement can never be starved;
  - our headroom is **inside** it (`K = 4 %` of a table's slots), i.e. coverage given up -- which is exactly why
    `K > gap` displaced the complement at 97.7 % coverage and cost 43-53 % of the field prefill until r228 capped
    it by the gap and disengaged the exchange below 4 %.
  Our third path has no analogue in his design: a **duplicate** (an expert the arena also holds) is reclaimed
  when the pool is saturated (`g_sink_evicted`) because that copy is worthless -- his tier is the complement by
  construction, so he can only decline (`g_tier_nospare`).
* **`K` is a transient reserve, his spares are a permanent rotation** -- which decides #277: a precopy cannot
  live in our headroom (a prefill fill may consume it, and r228 deliberately lets it), so precopy needs a
  dedicated K-slot ring with a side stream, i.e. his 0006/0007 shape.  `S7` in `PLAN.md` records why the literal
  slot-neutral swap cannot be built *from the incoming fill's own slot* (the fill is reading it and has already
  overwritten the victim) -- the promoted expert's *retired* slot is the one that works.
* **Ours declines rather than lies**: no slot, a geometry it cannot express (a `-sm tensor` axis-0 slice), or a
  failed copy ⇒ the victim is dropped exactly as before, nothing is published, and the admitted expert's own copy
  is unaffected (`gate-host-pool.sh`'s `sinkfail` arm asserts it).
* **Ours keeps the duplicate-retirement win** (#87) and retires a duplicate only when the headroom has run dry,
  because reclaiming an LRU resident instead was measured as a net loss (18.7 → 12.4 t/s at `-lme` 24).

## 3. What remains, in one list

### 3.0 r228 changed the frame (read this before ranking the items)

r228 (`TODO` #281, block 12) made the demote a **decode-class** mechanism: its headroom is capped by the pool's
coverage gap, a prefill fill ignores it, and a **near-mirror pool disengages it entirely**.  Three consequences
for the PR's remaining mechanisms:

1. **A demote comparison at 97.7 % coverage is no longer meaningful** -- the exchange does not fire there, by
   design (`sink near-mirror tables ... 144`, 0 demoted).  Judge 0004/0006/0008 at the cells where the sink is
   live: `-lme 44` (93.7 %, gap 32) and `-lme 36` (76.9 %, gap 119) on the CLI, and a *partial* field row
   (`-lme 44` / a low `MOE_HOST_LIMIT_FRAC`) on the rig.
2. **The field shape's first-pass prefill is no longer a demote question.**  At `auto` (97.7 %) the shipping
   configuration is now the `SINK=0` path to the counter (1 907-1 938 t/s measured).  What is left of that
   number is the miss path itself (the near-mirror gap), **not 0003b**: measured 2026-11-06 on a 3 × R9700 box,
   the arena transition evicts 12 tables / 1 062 MiB per event and costs ≤0.2-1.2 % of the prompt, and it does
   not fire at all on the 3-GPU tensor shape — so the staging transport and the coverage gap are the items, and
   0003b is closed ([`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) §1).
3. **The demote's transport (0004) matters on fewer cells** but still matters on the ones where it runs: it is
   worth +19-21 % generation at 94 %, so its D2H is on the hot path exactly there — but the hazard 0004 exists
   for needs the copy *and* a paired all-reduce, and the delivery never has both: the demote declines 100 % of
   offers on the tensor split (`n_slices != 0`, no contiguous pool image) and `-sm layer` has no all-reduce to
   poison.  **0004 is a transport optimisation with no reachable defect; 0006/0007 retire with it.**

| # | item | what decides it | where | state after r228 |
|---|---|---|---|---|
| 1 | **parity/crossover table** (`TODO` #280): pinned GiB vs decode vs first-pass prefill vs storage MiB/token, mirror as the asymptote | the field matrix at a budget the sink is *live* at, plus the DIO/buffered pair | PLAN P0.2/P0.3 | **CLOSED 2026-11-07 -- the table exists** ([`RESULTS-2026-11-07-radix.md`](RESULTS-2026-11-07-radix.md)): mirror 83.8 / 1 895 t/s at 46.8 GiB pinned, `auto` 97.9 % 75.6 / 1 431 at 45.5, `-lme 44` 93.8 % 76.2 / 735 at 43.9, `-lme 36` 76.8 % 76.1 / 1 037 at 36.0; the DIO/buffered pair (+43-81 % prefill), the span pair (+8 %) and the `-lme 44` field row are all measured.  **Decode-parity at 77 %, prefill-parity nowhere.**  It also found and shipped (r231) the band-transition tail: the pool's re-layout fill order |
| 2 | **the demote's copy submission** (`TODO` #276): his copy-kernel + drain rationale is the *paired-device* stall, not 1-GPU cost (the demote measures net +19-21 % at `-lme` 44, so the copy does not dominate) | a 2-GPU arm, a copy-kernel A/B, and a stall in the flight record right after an admission -- not a throughput delta | [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) §2 | **CLOSED 2026-11-06 -- unreachable by construction**: the demote declines **100 %** of offers on the tensor split (the only layout with the paired all-reduce: `0 demoted / 14 480 geometry (structural)`) and `-sm layer` (7 862 copies) has no all-reduce.  Re-entry: only if the demote is extended to an `n_slices != 0` table |
| 3 | **precopy** (`TODO` #277, 0006/0007/0008): a **dedicated** K-slot ring + a side stream + the drain semantics (our headroom cannot hold a pre-copied victim -- a prefill fill consumes it) | only if #276 leaves a residual on a multi-device arm | PLAN P1.2 | **RETIRED 2026-11-06** with #276; the condition fails, and the shape was already decided (his 0006/0007, not more headroom) |
| 4 | **the slab keeps the arena across a decode→prefill transition** (`TODO` #278, 0003b) | PLAN P1.4 -- first count the `evicted … re-armed` census on the field shape | [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md) §1 | **CLOSED 2026-11-06 -- measured, do not implement**: the drop is real (12 tables / 1 062 MiB per event at `-c 204800 -ub 6144`), the re-fill is bounded by the evicted bytes (≤0.2-1.2 % of the prompt) and invisible in paired rows, and the 3-GPU tensor shape never evicts |
| 5 | **narrow-2 prefers `cudaMalloc` over evicting arena tables** (`TODO` #279, 0003a) | PLAN P1.5 -- one A/B against `GGML_CUDA_SLAB_MIN_ARENA_MIB` | [`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md) | **CLOSED 2026-11-07 -- measured / do not implement**: the branch is unreachable on every shipping shape and its eviction is re-absorbed (identical arena MiB, hit rate, decode); the preference is a churn without a latch, and latching costs more cushion than it saves |
| 6 | **`MOE_EXPERT_CACHE_TOUCH` 2 vs 3 with the demote on** (0008) -- a knob, no code | PLAN P1.3, on a live cell | [`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md) | **CLOSED 2026-11-07 -- default stays 2**: touch-3 damps churn (-17.3 %/-18.3 % demotions, -27 % sink uses) and buys no decode (+1.3 to +2.2 % at 93.8 % coverage, inside the spread) for -0.8 pt of arena residency and +35 % cold-served misses; text identical (`0e3ed3d279f6`) |
| 7 | **the tier's policy and visibility** (0009's non-registration half): cgroup-aware figure, authoritative `-lme`, one verdict line | already scoped, do not duplicate | [`wip/host-expert-coverage/`](../../../wip/host-expert-coverage/README.md) | **open there** |
| 8 | **sparse registration** (0002) | **closed, not needed**: r228 + the r224 span made the partial tier's prefill cheap, and the tier's own in-place cold read is live and carrying 3.3-4.6 % of reaches with zero declines (measured 2026-11-07) | [`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md) | **CLOSED 2026-11-07** |

Every code item is a `TODO` handle: **#276** (transport, **closed**), **#277** (precopy, **retired**), **#278**
(arena across a prefill transition, **closed**), **#279** (narrow-2, **closed 2026-11-07 -- measured, do not
implement**) and **#280** (parity/crossover, **closed -- the table exists**).  The touch sweep needs no handle: it
is a knob, and it closed on 2026-11-07 with its default unchanged.  **Every item in this table is now closed**, so
everything of the PR is absorbed, superseded, rejected with a number, or closed by measurement or construction;
see [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md),
[`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md) and
[`RESULTS-2026-11-07-touch.md`](RESULTS-2026-11-07-touch.md).

## 4. What would close the PR (the reply, once the numbers exist)

1. **Absorbed or superseded**: 0001, 0005, 0010 (with its measurement), and 0009's placement half.
2. **Measured, with no change taken**: 0008's value (touch-3 damps churn -18 % and buys no decode, so the default
   stays 2).  0003a is **closed** (measured 2026-11-07: unreachable on our shapes, re-absorbed when forced, and
   the preference cannot avoid the eviction without a latch that costs the evicted bytes of cushion for one
   ~20 ms re-fill); 0003b is **closed** (measured, no addressable cost).
3. **Closed as not needed**: 0002.  Both halves of its condition are answered: r228 made the partial tier's
   prefill cheap, the decode half is the +19-21 % the tier already has, and the tier's in-place cold read carried
   3.3-4.6 % of all reaches with zero declines in the 2026-11-07 arms.
4. **The numbers to quote** (r228, this box): his **82.7 t/s / 1 949 t/s prefill** row against our **mirror
   83.7 / 1 932-1 938** at the same 56 GiB container shape (parity on the mirror), and the tier curve that the
   pool delivers without any registration -- **76.9 % = 26.0, 93.7 % = 33.9 (+19-21 % from the demote),
   100 % = 38.2** (CLI, one R9700).  **The two transport items are answered as of 2026-11-06** (0004's hazard is
   unreachable -- the demote declines 100 % beside a paired all-reduce; 0003b costs ≤0.2-1.2 % of a prompt), so
   what the reply still owes is the crossover table (#280) and 0003a's A/B (#279).
   [`RESULTS-2026-11-06-soar.md`](RESULTS-2026-11-06-soar.md).
   **Both are now answered**: #280's table is in [`RESULTS-2026-11-07-radix.md`](RESULTS-2026-11-07-radix.md) and
   0003a is a measured "do not implement" in [`RESULTS-2026-11-07-narrow2.md`](RESULTS-2026-11-07-narrow2.md)
   (unreachable on our shapes; forced, the eviction is re-absorbed; the preference churns without a latch and the
   latch costs more cushion than it saves).
5. **Posted 2026-11-06** (the config + numbers above, the r228 rules, the repro conditions and the
   taken/rejected/open mapping):
   <https://github.com/stew675/llama-cpp-rdna-boosts/pull/267#issuecomment-6698266713> -- awaiting the
   reporter's confirmation before the close.  The reply also flags that the fill mode follows `--load-mode`
   (his box measured `dio` decisively better; the pool and the span read follow it) and that the demote
   declines on a `-sm tensor` axis-0 slice, which matters for the second GPU he is bringing up.
6. **Then**: post that as the PR's closing summary ("integrated rather than merged": every mechanism is either
   in the delivery in a different shape, measured and rejected, or recorded with a handle) and close it.  Do not
   mark it merged -- nothing of the branch is applied.
