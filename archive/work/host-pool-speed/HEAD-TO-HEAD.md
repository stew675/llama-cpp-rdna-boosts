# HEAD-TO-HEAD — pitting the two host-expert approaches against each other

**Status: prepared 2026-11-02, not run.**  This campaign (`host-pool-speed`) is **approach A**.  A parallel session
is exploring **approach B**, the `--host-experts mmap` / pageable-master family (PR #267's lineage: the host
experts stay in the pageable model mapping and the GPU reads them directly, with the pinning done per expert /
per slice on demand).  When B is ready, run this protocol once on both builds and let the numbers decide.

The point is not “which bench number is higher” in the abstract — it is the campaign's acceptance criterion
(`README.md` §1): **approach B must offer no performance advantage over a pool-mode build that has been tuned with
these mechanisms.**  If it does offer one, that is a finding about pool mode, not a verdict on B.

---

## 0. The two approaches in one paragraph each (for whoever runs this)

**A — pool mode, an explicit intermediary (this campaign).**  The host expert tensors stay in the read-only model
mapping; *every* GPU read of them is staged through a bounded **pinned** host pool (`MOE_HOST_POOL_*`), sized as a
percentage of each tensor's bytes and clamped to `MOE_HOST_POOL_RAM_PCT` of `MemAvailable`.  The pool's content is
the arena's complement ranked by a persisted decode-heat prior; the arena is the VRAM tier (`MOE_EXPERT_CACHE_*`).
This campaign's changes: **full coverage by default** (`MOE_HOST_POOL_FRAC=100`) and **`cov90` per-table sizing**
(`MOE_HOST_POOL_SIZING=1`), see [`patches/README.md`](patches/README.md).  Claimed shape: works for every split
type (the intermediary is a buffer type, not an address trick), bounded non-swappable footprint, and a diagnostic
story for the disk tier.

**B — mmap / pageable master (the parallel session).**  Pool mode's intermediary is dropped: the GPU reads the
mapping directly, and the requisite pages are made *pinned and resident* per expert/slice on demand (sparse
registration / aliasing), with a pin budget.  Claimed shape: fewer copies, and the page cache doubles as the cache.
The known fragility (recorded in `README.md` §8 from the architecture review) is geometry: `n_slices == 0` for a
fused table, axis-0 vs axis-1 slices, POSIX-only aliases, and cross-device behaviour.

Both may be right for different boxes; the head-to-head exists to find out *where*.

---

## 1. The fixed protocol (copy of `README.md` §9 — both builds must use the same one)

- **Model**: `/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf`
  (73.2 GiB, 49 blocks × 256 experts, 8 used, hybrid SSM, embedded MTP, **no PLE**).
- **Arm**: `-ncmoe 48 -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 -fa auto -b 2048 -ub 2048 --spec-type none
  --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt -n 2000 -p "$(cat prompts/prose-rdna-boosts.txt)"`.
- **Pins**: `HIP_VISIBLE_DEVICES=0` for 1 GPU; `HIP_VISIBLE_DEVICES=0,2` + `-sm tensor` for the x8 pair (**never**
  `HIP 1`, the x4 card).  Without the pin the iGPU is enumerated and the load aborts (`invalid kernel file`).
- **Decode ≥ 2000 tokens** for any policy number (1000 is the floor; short runs are correctness smoke only — the
  arena ramps over the first few hundred tokens).
- **Alternate arms in one run** and always include a `pinned` reference in the same run: the comparison is a
  *ratio to pinned measured on the same binary, same box state*, never an absolute from another day.
- **Never run arms in parallel.**  One benchmark at a time on the box.
- **Log level `-lv 4`** for anything that comes out of libllama (`LLAMA_LOG_INFO` maps to verbosity 4); **no `-lv`
  at all** when hashing generated text.
- Record the page-cache state before each arm (`MemAvailable`, `Cached`) and the disk bytes read per arm
  (`/proc/diskstats`, field 6 of the device line — the campaign's runners do this).

## 2. The arm matrix (run for each build, then diff)

| # | arm | why |
|---|---|---|
| 1 | `pinned` | the reference; also the “no pool overhead” bound |
| 2 | **A only**: `pool` at its shipped default (post-0002 = FRAC 100) | A's headline |
| 3 | **A only**: `pool` at a sub-full budget (`MOE_HOST_POOL_FRAC=60`, and again with `MOE_HOST_POOL_SIZING=1`) | the field case: a budget that cannot hold the set |
| 4 | **B only**: its shipped default (mmap/pageable master, pin budget at its default) | B's headline |
| 5 | **B only**: its answer to #3 — the same *sub-full* regime (its pin budget cut to ≈40 GiB of pinned pages) | the regime B's design is most distinctive in |
| 6 | each build's **own two strongest variants** | so neither side is judged on a strawman |

**Do not cross-apply the knobs** (A's `MOE_HOST_POOL_*` do not exist in B and vice versa).  Do keep the *arena*
identical: same `MOE_EXPERT_CACHE_MIB` / auto sizing and the same `-ncmoe`.  If B changes the loader's host buffer
type, note it — that is part of B.

## 3. Metrics to collect (per arm, all of them)

1. **decode t/s** and **prefill t/s** (from `slot print_timing`; never from a shell `time`).
2. **Ratio to the same run's `pinned` decode** — the acceptance number.
3. **Non-swappable host footprint**: pool bytes pinned (the pool's own report), or B's pin budget/`VmLck`
   equivalent — *and* `RssShmem` if it differs.  Report both, since `hipHostMalloc` does not show in `VmLck`.
4. **Disk bytes read per run** (`/proc/diskstats` delta) and, if wanted, per decode token after subtracting the
   model load the `pinned` arm pays on the same box state.
5. **Arena hit rate** — the per-turn `slot print_timing: MoE arena = h (hits/reaches)` line (it is `COM_INF`, so
   it shows at any `-lv`).  **Do not** use the teardown `moe_cache_report` for this on a pre-0001 build: it prints
   post-teardown zeros (`patches/README.md`, patch 0001).
6. **Pool/cold split** where the build reports it (`expert access:`, `expert H2D traffic:`, pool `h=`).
7. **Wall clock** for load + prefill + decode separately (a design that wins decode but pays 3 minutes of load is
   not a win for an interactive box).
8. **Failure modes**: OOM kill, swap growth (`VmSwap`, `pswpout`), abort, `invalid kernel file`, GPU faults.

## 4. Gates both builds must pass before any number counts

- **Same-seed greedy text** byte-identical across the arms of that build (`scripts/extract-generated.py`, hashed
  with **no** `-lv`) — and, for a *cross-build* claim, identical to the same arm of the other build where the
  configuration is the same (A vs B will differ if B changes numerics; say so explicitly rather than hiding it).
- `scripts/check-gpu-faults.sh` on every arm's log (never grep a log for `page fault` by hand).
- `scripts/gate-prefill-logits.sh` if the build changes a prefill kernel or a default-on approximate path.
- The MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) if the build changes decode numerics.
- `MOE_HOST_POOL_VERIFY=1` (A) / B's equivalent self-check, once per configuration.
- A 30-minute soak with `VmSwap`/`MemAvailable` sampling on the sub-full arm, if either side claims field
  viability.

## 5. Scoring rules (agree these *before* looking at the numbers)

1. **A wins** if B's best arm is within ~2 % of A's best arm on decode *and* A is not worse on (a) resident
   footprint, (b) the sub-full regime, (c) split-type coverage.  (Then B has no advantage and is not adopted —
   the campaign's original criterion.)
2. **B wins** if it beats A by more than ~5 % decode on the *same* pinned footprint, in *both* the full-coverage and
   the sub-full regimes, and it passes every gate on at least the 1-GPU and the 2-GPU x8 tensor arms.
3. **Split decision** (the likely outcome) — record per regime: which wins at full coverage, which at a sub-full
   budget, which survives `-sm tensor`, and which has the better failure story.  Then the maintainer decides
   whether to keep both (A as the default, B as an opt-in for the boxes where it wins) or to fold B's winning
   mechanism into pool mode (the campaign has a standing rule for this: a mechanism is adopted *in pool mode* if
   its advantage is real — `README.md` §1).
4. **Inconclusive** if either side's numbers move more than the arm-to-arm spread on a repeat (the campaign's
   observed spread is ±0.06 t/s on `pinned` and ±0.03 on a pool arm at 2000 tokens; a difference smaller than that
   is not a difference).

## 6. Traps that will silently ruin the comparison (all bitten in this campaign)

- **`-lv 3` hides every libllama INFO line** — including `host_experts_mode=POOL`, `table_pool_locked`,
  `pool_decide_dio_locked` and the cache's `h=` report.  Use `-lv 4` (and no `-lv` for text hashing).
- **The teardown cache report is zero-by-construction on any build before patch 0001** — in *both* modes.
- **Page-cache state is part of the measurement.**  A pool build's disk reads (and its `DIO` choice) depend on what
  the previous arm left in RAM; record `MemAvailable`/`Cached` per arm, and re-run any arm whose state differs
  wildly.  Full coverage (A's default) pins 67.1 GiB — on a box with a 73 GiB model that is *look* tight, and the
  second arm in a row behaves differently from the first.
- **Do not judge a tensor-split arm on the 122B with short runs**: on `-sm tensor` the pinned shape currently gets
  no expert cache at all and a full pool can stall the prepopulate (`HANDOVER.md` §4/F7).  Gate that arm
  separately and say so.
- **Never `git apply` a concatenated series** — `git am` only (the delivery repo's rule, and it applies to these
  campaign patches too).
- **`/tmp` is shared on the maintainer's box**: check patch filenames before globbing them into `git am` (a
  stale `/tmp/0001-*.patch` from another session silently goes first — it happened once while validating this
  set).

## 7. Where B's patches currently live (housekeeping, not a review)

While validating this campaign's set, a **different** patch series was found in `/tmp` on the maintainer's box
(dated 02:03, not mine, left untouched): `/tmp/0001-llama-restore-host-experts-mmap-*.patch` through
`/tmp/0009-moe-cache-one-pin-budget-*.patch` — i.e. the `--host-experts mmap` + pageable-master + RAM-tier +
precopy + pin-budget family, nine patches, which reads like the parallel session's approach B (PR #267 lineage).

**`/tmp` is not durable.**  Before the head-to-head, archive that series where the repo's rules put an external
candidate (`beta/<name>/` for a promotion candidate, or `wip/mmap-host-experts-*/` / `archive/work/<name>/` for a
campaign record, with a `README.md`), and do not mix it into `patches/`.  This campaign has not read, applied or
modified those patches; the only thing recorded here is where they were last seen.

## 8. Where the numbers go

Append a dated entry to [`HANDOVER.md`](HANDOVER.md) §3 (or a `benchmarks/2026-11-..-host-expert-showdown.md` if it
grows beyond a table), keep the per-arm logs under `data/<date>-head-to-head/`, and state which gate each arm
passed.  Do **not** paste the result into `patches/README.md`, `README.md` (delivery) or `WORKLOG.md` until the
maintainer has decided — approach B is not in the delivery set.
