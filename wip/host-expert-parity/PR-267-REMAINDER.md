# PR #267 — what is left, per patch (re-assessment 2026-11-06, against the r224 tree)

**Status: assessment only.  Nothing here is applied, and no patch of the PR can be applied as-is** (they are cut
against r204 and touch code that moved in r209-r224; `archive/work/host-expert-split/DESIGN.md` §5 has the port
analysis).  This file exists so the PR can be closed as **integrated rather than merged**, and so a future
session does not re-review a mechanism we have already measured.

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

| # | item | what decides it | where |
|---|---|---|---|
| 1 | **the demote's transport** on a Gen5 x16 host: is the `cudaMemcpyAsync` D2H still ~free, or does it sit on the token path?  Then (and only then) a **copy-kernel form + a drain on a failed event record** | PLAN P0.1 (the constant) → P1.1 (the code) | this campaign |
| 2 | **precopy** (K spares + a side stream; 0006/0007, incl. the drain) | only if P1.1 leaves a residual >~1 % of the step | this campaign |
| 3 | **touch 2 vs 3 with the demote on** — a knob, no code: it bounds the copy volume and the measured says it costs a third of the sink's advantage | PLAN P1.3 | this campaign |
| 4 | **the slab keeps the arena across a decode→prefill transition** (0003b): first count whether the arena is actually dropped on the field shape (`evicted … re-armed` census), then decide | PLAN P1.4 | this campaign |
| 5 | **narrow-2 prefers `cudaMalloc` over evicting arena tables** (0003a) | PLAN P1.5, one A/B against `GGML_CUDA_SLAB_MIN_ARENA_MIB` | this campaign |
| 6 | **the tier's policy and visibility** (0009's non-registration half): a cgroup-aware figure, an authoritative `-lme`, one verdict line | already scoped, do not duplicate | [`wip/host-expert-coverage/`](../host-expert-coverage/README.md) |
| 7 | **sparse registration** (0002) | **non-goal.**  Re-entry only if a partial tier cannot be made cheap by the tier itself.  That is the question P0.2/P0.3 answer, so this is decided *by* this campaign, not inside it | — |

Every item above is a `TODO` handle, so none of them can be lost: **#276** (transport), **#277** (precopy),
**#278** (arena across a prefill transition), **#279** (narrow-2 preference), **#280** (the parity/crossover
measurement itself).  The touch sweep needs no handle: it is a knob.

## 4. What would close the PR (the reply, once the numbers exist)

1. **Absorbed or superseded**: 0001, 0005, 0010 (with its measurement), and 0009's placement half.
2. **Borrowed, pending the A/Bs here**: 0004 (+0007's drain), 0008's value, 0003's two halves.
3. **Deferred with a condition**: 0006 (precopy) behind 0004's result; 0002 behind "a partial tier cannot be
   made cheap".
4. **The numbers to quote**: his 82.7 t/s / 1949 t/s prefill row against the shipping tier on this box at the
   same 56 GiB container shape, plus the crossover table (`PLAN.md` P0.2).
