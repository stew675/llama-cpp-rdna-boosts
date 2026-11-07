# RESOLUTION — `host-expert-transport` closed 2026-11-07 with r234

**`TODO` #271: RESOLVED (visibility shipped, and the "cost" refuted).  `TODO` #272: CLOSED as documented
(no default changed).**  The delivery change is one file, `ggml/src/ggml-cuda/moe-expert-cache.cu`, block
12, +106/−10: the declined tables are named once per run and the census stops counting the priming pass
as a defect.  Numbers, protocol and raw runs: [`README.md`](README.md); the delivery record is
`WORKLOG.md` 261107.011.

## 1. Claim (#271)

The prefill staging path declines a pinned-master table when it has no arena
(`t.arena == nullptr || t.slots <= 0`), and on a real pinned master that is **50 of 144 tables for the
whole prefill**, silently, on every ubatch.

**The predicate was right and the fallback is free.**  The reason is a **transient, designed** state: the
prefill's compute layout takes the slab chunks a table's arena lived in
(`moe_cache_evict_slab_range: evicted 7536.0 MiB … 50 table(s) stood down`), and `moe_cache_rearm()`
rebuilds them at the compute-buffer drop, before any decode.  Measured with balanced ABBA
(1 GPU, pinned master, `-ub 6144`, 52 512-token prompt): **1826 t/s declined against 1824 t/s staged**
(±0.1 %).  The scheduler's fallback is one bulk async copy per table on the **same copy stream** the
staging ring uses, so "per-tensor copies serialised on the compute stream" was a misreading of a trace
from a different arm.

The two candidate "fixes" were built and measured, and both are **worse**:

* relax the arena requirement and keep the pinned tier's engine (`moe_cache_stage_kernel`, one 512-block
  launch per table): **-10 %** (1572/1595 against 1757/1765) — the kernel is the wrong engine for a
  whole-table copy on this stack, exactly the CU-monopolisation mechanism
  [`../host-expert-mirror/LEAD-pool-staging-gap.md`](../host-expert-mirror/LEAD-pool-staging-gap.md) §2
  traced on the pool tier;
* the same relaxation with one bulk copy per table: **±0 %** (1824 against 1826) — correct and no faster,
  so it does not earn a new code path next to the fallback the scheduler already has.

## 2. Rule (#271)

1. **A staging decline is a decision, so it gets a line.**  One WARN per run names the count, the reason
   and up to 12 tables (the rest at INFO), and the census now splits `unsized=` (the priming pass, by
   design) from `unusable=` (a table that will never be staged).  Without it, "which tables, and why" cost
   a patch and two instrumented builds.
2. **The eviction line names the table count, not only the bytes.**  The two halves of the story
   (`evicted N MiB` and `prefill staging declined M tables`) were in different places and neither said the
   other existed.
3. **A fallback is not a defect until it is measured against the alternative.**  Here the alternative was
   10 % *slower* and the second alternative was neutral: the silent state was a visibility bug, not a
   performance bug.  Re-deriving that took two ABBA cycles; assuming it took two wrong conclusions first
   (both from block-ordered runs — see rule 5).
4. **Never require an arena to fill a copy.**  The arena is a *source* for the staging copy, not a
   precondition for it.  (Not acted on: with the engine it would select, it is not faster.)
5. **Judge a transport with `A B B A` and quote the spread.**  Identical consecutive runs on one unchanged
   arm measured 1873 / 2177 / 1865 / 2180 t/s (period 2, amplitude 16 %, cause unidentified).  The gather
   is immune (±0.1 %), every copy-engine arm is not.  A plain `A B A B` reads as a 6 % win for whichever
   arm sits in the even positions.

## 3. Claim and disposition (#272)

The archived transport band (`archive/work/host-expert-mirror/LEAD-pool-staging-gap.md` §1) said the
copy-engine arm won at 1 GPU / `-ub` 5120-7168, lost at 2048 **and by 8192** ("its pipeline stalls:
kernel/wall 0.80"), and lost on a 3-device tensor split.

**Re-measured with ABBA on r233/r234, the band's upper edge does not exist.**  At `-ub 8192` the
copy-engine arm beat the gather in **4 of 4** runs (1829-2133 against 1788-1793) and at 16384 the two tie
(2002 against 1987); the gather's win at 2048 (+20 %) and the copy engine's at 5120 (+18 %) both
reproduce.  **The "8192 stall" is the low phase of the bimodality, not a stall**: the suspected cause
(ring depth `GGML_SCHED_STAGE_SLOTS` = 8 against the split count) was tested at `=16`, the compile-time
maximum, and changes nothing.

**No default was changed, deliberately:**

* the band still needs a **width gate** (the gather wins by 20 % at 2048) and the crossover between 2048
  and 5120 was not measured, so a rule would be tuned to an unmeasured edge;
* it needs **two coupled decisions** (the gather/staging choice *and* the staging engine), i.e. a transport
  selector, not a default flip;
* the winning arm is the one that is **bimodal** (17-32 % spread), so a consumer can land in its low
  phase — the gather is 20 % worse in one cell and never unstable;
* on the **pinned** tier the same question has a different answer again (+12.8 % for not staging at all,
  -16 % for forcing the gather), i.e. the ranking is tier-shaped as well as width-shaped;
* the r114 lesson (`ENVIRONMENT.md`: the single-launch kernel's +34 % was real only against a *pageable*
  master) is exactly what happens when a transport default is flipped from one cell.

**What a consumer can do today**, with switches that already exist and are now documented with their
prefill effect (`ENVIRONMENT.md`):

* pool tier, 1 GPU, `-ub` ≥ ~5120: `GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0` → +10-18 %;
* pinned tier, 1 GPU, `-ub` 6144: `GGML_MOE_CACHE_STAGE=0` → +12.8 % (at the cost of sending the arena's
  bytes over PCIe too);
* every width: the default gather, which is the stable choice.

## 4. What would reopen it

1. **The crossover width between 2048 and 5120** (a 4-arm ABBA set), which is the one number a width-gated
   rule needs.
2. **The cause of the period-2 bimodality.**  Until it is named, a copy-engine default is a coin flip and
   any A/B of one is a measurement of the box's phase.  Candidates that were ruled out: paging,
   `MemAvailable`, the arena/slot geometry, THP accounting.  Not ruled out: the pinned tier's physical page
   backing after a re-grow (each run re-pins 48 GiB), the GPU's clock/power state, and the PCIe link's
   power state on the **x8** slot gpu0 sits in.
3. **A second device count and `-sm layer`** for the pool band (the archived 3-device tensor row says the
   gather wins by 14 %, unverified on r233+).
4. **A hardware or ROCm change** that alters the copy-engine/kernel-read asymmetry.

## 5. Not promoted, and why

* No default change (see §3) — nothing in `patches/` was altered apart from the census/decision line.
* No `optional/` artifact: no patch was produced for a feature that has no validated win on a supported
  configuration.
* The per-expert copy-engine arm for the **pinned** tier was prototyped (`MOE_HOST_POOL_STAGE_BATCH=0`
  reaching the pinned branch) and **not shipped**: its two runs straddled the bimodality in opposite
  directions (1620 low / 2142 high), so it is not established as better than the shipped kernel, and a
  default-off arm that is not better violates the default-on policy.
