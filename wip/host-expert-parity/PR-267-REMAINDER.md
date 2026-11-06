# PR #267 — what is left, per patch (re-assessment 2026-11-06, against the r224 tree)

**Status: assessment + the r228 follow-up.**  Nothing of the PR is applied, and no patch of it can be applied
as-is (they are cut against r204 and touch code that moved in r209-r228; `archive/work/host-expert-split/DESIGN.md`
§5 has the port analysis).  **r228 changed the frame** (§3.0): the demote that 0004/0006/0008 target is now a
decode-class mechanism that only runs where it pays, so every remaining comparison is made at a budget *below*
the near-mirror boundary.  This file exists so the PR can be closed as **integrated rather than merged**, and so
a future session does not re-review a mechanism we have already measured.

Read with [`DESIGN.md`](../../archive/work/host-expert-split/DESIGN.md) (the original review + the convergence
design) and [`archive/work/host-pool-speed/README.md`](../../archive/work/host-pool-speed/README.md) §4.3/§4.4/§6
(the earlier per-mechanism verdicts, still correct).  The r224 tree is what "our state" means below; the file
`ggml/src/ggml-cuda/moe-expert-cache.cu` is the single file every one of his patches edits.

## 1. The ten patches

| # | his mechanism | our state at r224 | verdict |
|---|---|---|---|
| 0001 | `--host-experts mmap` puts the `-ncmoe`/`-cmoe` experts back in the pageable mapping; GPU-keyed host-expert accounting | the **mode is gone and must not come back**: since r209-r212 the placement is one budget, `-lme [auto\|N]` (below the set = a per-expert pinned pool over the read-only mapping; at/above it = a mirror), and `MOE_HOST_TIER=tensor` is the whole-tensor escape hatch | **superseded** |
| **0002** | sparse registration: a second read-only `MAP_SHARED` alias, `mlock` + `hipHostRegister(ReadOnly)` for the expert ranges a GPU may read, `madvise(WILLNEED)`, trim after a device sync, `MAP_FIXED` remap of KFD's leftover range; owner-device routing; whole-table prefill staged from arena+alias | **not adopted.**  The job is done by two mechanisms we already ship: a pool-resident miss is an **in-place PCIe read of pinned RAM** (`MOE_HOST_POOL_COLD` + `_MULTIDEV`, r203/r210) and a declined read goes through the per-table **bounce ring** (`expert_host_src`, the `gate-host-pool.sh` arms).  `-sm layer` owner routing is the semantic key (r215+) | **non-goal**, re-entry condition in §3 |
| 0003a | the narrow-2 (draft compute) pin prefers its `cudaMalloc` fallback when it fits **outside** the slab instead of evicting arena tables | **partially.**  `ggml_cuda_slab_narrow2_pin_locked` prefers the unmapped spare, **refuses** when the region would leave less than `GGML_CUDA_SLAB_MIN_ARENA_MIB`, maps the VA above `mapped`, and falls back to `cudaMalloc` only on a refusal — but once the arena floor is satisfied it *will* evict arena tables for the draft's buffer | **delta remains** → PLAN P1.5 (an A/B, ~1 branch) |
| 0003b | `GGML_CUDA_SLAB_RING_PREFILL=0` (implied by mmap): keep the expert-cache arena across a decode→prefill transition | **absent** — no such variable exists.  The slab's H2D ring *is* reclaimed on the prefill→decode drop; whether arena **tables** are dropped going the other way on the field shape is unmeasured | **measure first** → PLAN P1.4 |
| **0004** | the swap is two custom **copy kernels**, not `hipMemcpyAsync`, because a small pinned copy can leave later queued work unsubmitted on ROCm | our demote is one `cudaMemcpyAsync(D2H)` on the compute stream into a free pool slot (block 12).  It measured +5.3 % net on r224's box, so no stall was visible there — but that box's link is not this one | **open** → PLAN P1.1 |
| 0005 | tier × device-remap, and a tier table never uses the pool's cold read | ours already declines on a **partial** pool and serves a mirror (`dr_pool_ok`, r221) | **absorbed** |
| **0006/0007** | precopy: K spare RAM slots + a side stream that copies the policy's likely victims ahead of time; a failed event record drains the side stream, a failed wait fails the swap | **absent** (`MOE_HOST_POOL_PREFETCH`, the knob the earlier review named as the survivor, no longer exists).  Our demote only pays the D2H when the miss proves the demand | **conditional** → PLAN P1.2 (only if P1.1 leaves a residual cost) |
| **0008** | a tier admits from the **3rd** touch (`GGML_MOE_TIER_TOUCH=3`) to damp churn; 8 spares | the knob exists (`MOE_EXPERT_CACHE_TOUCH`, default **2**), the spares are our exchange headroom; the *value* was never swept with the demote on.  The replay says touch-3 **shrinks the sink's advantage by a third** (it bounds copies, it does not save L3 — `STEP1-REPLAY.md` §4) | **measurement** → PLAN P1.3 |
| 0009 | **one** pin budget for tier + registration (`GGML_MOE_PIN_BUDGET_MIB`), a **partial tier by hotness**, bounded warm-up registrations, a refused registration not retried until the next trim | half landed elsewhere: `-lme` **is** the one budget (r209-r212) and since r207 a shrunken plan is a per-table **distribution**, not a tail shave, so a partial tier is already hotness-ordered by content.  The registration half is moot.  What is left is the *policy/visibility* half — and it is owned by [`wip/host-expert-coverage/`](../host-expert-coverage/README.md) steps 1-3 (cgroup-aware `MemAvailable`, an authoritative explicit `-lme`, one visible `MIRROR/PARTIAL/NONE` verdict) | **split off, not here** |
| 0010 | a declined read goes through an **8-entry** per-table bounce ring (capped so a whole-master declined read cannot pin a copy of the master) | the ring exists and is **deliberately uncapped**: measured, a cap of 2/4/8 costs **8-25 % of prefill** at FRAC 80 (`bounce_acquire`'s comment), and r224's span coalescing addresses his *concern* (fewer, larger reads ⇒ fewer entries in flight) instead | **rejected, with a number** |

## 2. Where the mechanisms differ by design (ours is not a port of his)

Worth stating once, because a reviewer of the PR will ask:

* **His swap is slot-neutral-ish (victim → the promoted expert's slot) with a spare per table; ours is a demote
  into exchange headroom planned *below* the budget** (`K = 4 %` of a table's slots, floor 2, cap slots/8), so
  nothing grows, the mirror stays impossible to exceed, and the demote is **inert** against a mirror
  (`slots >= n_experts`).  `S7` in `PLAN.md` records why the literal slot-neutral swap could not be built (the
  incoming fill is reading that slot and has already overwritten the victim).
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
   number is the miss path itself -- which is exactly what **0003b** (`TODO` #278, the arena across a
   prefill transition) and the staging transport are about, and that is now the highest-value prefill item in
   this list.
3. **The demote's transport (0004) matters on fewer cells** but still matters on the ones where it runs: it is
   worth +19-21 % generation at 94 %, so its D2H is on the hot path exactly there.  The A/B is unchanged in
   shape, only cheaper to justify.

| # | item | what decides it | where | state after r228 |
|---|---|---|---|---|
| 1 | **parity/crossover table** (`TODO` #280): pinned GiB vs decode vs first-pass prefill vs storage MiB/token, mirror as the asymptote | the field matrix at a budget the sink is *live* at, plus the DIO/buffered pair | PLAN P0.2/P0.3 | **half done**: CLI 76.9 % = 26.0, 93.7 % = 33.9, mirror = 38.2; field mirror 1 932-1 938 / 83.7 vs `auto` 1 837 / 76.6.  Owed: the DIO-vs-buffered field pair, the span-reach pair, and a partial field row (`-lme 44`) |
| 2 | **the demote's transport** (`TODO` #276): is the `cudaMemcpyAsync` D2H on the token path where the sink is live?  Then a **copy-kernel form + a drain on a failed event record** (0004/0006/0007) | PLAN P1.1 -- the `SINK=0/1` delta at `-lme 44`, not at 97.7 % | this campaign | **open**, now measurable on a live cell |
| 3 | **precopy** (`TODO` #277, 0006/0007/0008): K spares + a side stream, incl. the drain semantics | only if #276 leaves a residual >~1 % of the step | PLAN P1.2 | **open, conditional** |
| 4 | **the slab keeps the arena across a decode→prefill transition** (`TODO` #278, 0003b) | PLAN P1.4 -- first count the `evicted … re-armed` census on the field shape | PLAN P1.4 | **open, promoted** -- it is now the field prefill's main residual, not a demote artefact |
| 5 | **narrow-2 prefers `cudaMalloc` over evicting arena tables** (`TODO` #279, 0003a) | PLAN P1.5 -- one A/B against `GGML_CUDA_SLAB_MIN_ARENA_MIB` | PLAN P1.5 | **open** |
| 6 | **`MOE_EXPERT_CACHE_TOUCH` 2 vs 3 with the demote on** (0008) -- a knob, no code | PLAN P1.3, on a live cell | this campaign | **open** (the replay predicts the sink's advantage shrinks a third; never measured on device) |
| 7 | **the tier's policy and visibility** (0009's non-registration half): cgroup-aware figure, authoritative `-lme`, one verdict line | already scoped, do not duplicate | [`wip/host-expert-coverage/`](../host-expert-coverage/README.md) | **open there** |
| 8 | **sparse registration** (0002) | **non-goal.**  Re-entry only if a partial tier cannot be made cheap by the tier itself -- and r228 + the r224 span made the partial tier's *prefill* cheap, so this is closer to closed than open | -- | **non-goal, unchanged** |

Every code item is a `TODO` handle: **#276** (transport), **#277** (precopy), **#278** (arena across a prefill
transition), **#279** (narrow-2), **#280** (parity/crossover).  The touch sweep needs no handle: it is a knob.

## 4. What would close the PR (the reply, once the numbers exist)

1. **Absorbed or superseded**: 0001, 0005, 0010 (with its measurement), and 0009's placement half.
2. **Borrowed, pending the A/Bs here**: 0004 (+0007's drain), 0008's value, 0003's two halves.
3. **Deferred with a condition**: 0006 (precopy) behind 0004's result; 0002 behind "a partial tier cannot be
   made cheap" (r228 makes the *prefill* half of that cheap; the decode half is the +19-21 % the tier already
   has, so the condition looks answered in the negative for the registration mechanism).
4. **The numbers to quote** (r228, this box): his **82.7 t/s / 1 949 t/s prefill** row against our **mirror
   83.7 / 1 932-1 938** at the same 56 GiB container shape (parity on the mirror), and the tier curve that the
   pool delivers without any registration -- **76.9 % = 26.0, 93.7 % = 33.9 (+19-21 % from the demote),
   100 % = 38.2** (CLI, one R9700).  Plus the two remaining transport items (#276, #278) or their measured
   refutation, and the crossover table (#280).
5. **Then**: post that as the PR's closing summary ("integrated rather than merged": every mechanism is either
   in the delivery in a different shape, measured and rejected, or recorded with a handle) and close it.  Do not
   mark it merged -- nothing of the branch is applied.
