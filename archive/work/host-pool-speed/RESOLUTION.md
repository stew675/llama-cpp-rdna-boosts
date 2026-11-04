# RESOLUTION — `host-pool-speed` retired (2026-11-04, r213 delivery session)

**Status: CLOSED AND ARCHIVED — promoted, then superseded by its own successor campaign.**  The directory moved
here from `wip/host-pool-speed/` and no stub was left behind (the `wip/` rule: a finished campaign does not stay in
`wip/`, not even as a redirect; see `AGENTS.md` → WIP and promotion rules).  Its index row was removed from
`wip/CAMPAIGNS.md`, whose narrative section records the retirement.

## Why it moved

The campaign's charter (2026-11-02) was to make `--host-experts pool` *simpler / more automatic / faster* until
**PR #267's `--host-experts mmap` offered no advantage**.  Two things finished it:

1. **Its results shipped in r207** (block 12, `WORKLOG.md` 261103.009-012): a POOL-backed table gathers in the
   prefill band (`GGML_SCHED_DEVGATHER`, three-valued, unset = pool-backed only), the `MOE_HOST_POOL_RAM_PCT`
   ceiling scales every table, `MOE_HOST_POOL_SIZING`/`_COV_PCT` (opt-in), the run diagnostics, and the dead
   `MOE_HOST_POOL_PREFETCH` removed.
2. **Its successor subsumed and obsoleted the rest**: `archive/work/host-expert-eager-push/` (r210/r211, shipped
   as block 09 + the r212 repair) replaced the tier *policy* the campaign was tuning — the host-resident MoE expert
   tier is now a **budget** (`-lme auto|N`, `MOE_HOST_LIMIT_FRAC`) over **one** mechanism, the pinned per-expert
   pool laid out as the arena's complement (`MOE_HOST_TIER=pool|tensor` forces either shape), read in place on any
   device, with a covering pool exempt from the eager take-over's arena bound.  Everything this campaign was
   measuring `--host-experts pool` *for* — "which host tier, sized how, read by whom" — is now decided by `-lme`.

That also **obsoleted the campaign's transports**: the direct pageable push and its `mlock` working set were
measured and **dropped** from the shipped design (they are not in `patches/`), so the two TODO items that existed
only to make `push` viable were retired with this directory (see below).

## What was retired with it

| item | why it cannot be kept |
|---|---|
| `TODO` **#137** — "make the pageable push work under `-sm tensor`" | the pageable push is **not in the delivery**; it was dropped from `host-expert-eager-push` after measurement.  There is no code to make faster. |
| `TODO` **#138** — "bound the per-table bounce ring so it can be a `push` transport" | the transport it exists for is gone.  (The bounce ring itself still exists as the pool-declined fallback; a *cap* on it would be a new item with its own measurement, not this one.) |
| `TODO` **#139** — "a partial `-lme` budget wants a per-EXPERT region" | **closed by r211**: the expert-indexed region *is* the bounded pool, it reads in place on a multi-device split (`MOE_HOST_POOL_COLD_MULTIDEV`, +11.5 % at `-lme 8`), and `-lme N` spends per expert below the host expert set. |
| `TODO` **#140** — "a bounded pool could SPILL instead of dropping", incl. the host-side READ-AHEAD variant | **built and measured as a LOSS in both bands** and dropped (page cache FRAC 50: 56.4 -> 52.8 t/s; `O_DIRECT` disk FRAC 50: 15.9 -> 15.1, prefill 24.1 -> 16.3).  The band it targeted is served by the surplus/mirror rule instead (`P6-POOL-REGION.md` §8/§10). |

## What stays open, and where

Three items opened from this campaign's measurements are **not** retired — they describe the *shipped* modes and
still have a live mechanism (`--host-experts pool` / `pinned` / `GGML_SCHED_DEVGATHER` all ship), so they live in
`TODO.md` with their records repointed here:

* **#131** — the ≥3-device pool prefill collapse is **fixed** (r207, the pool-backed gather is the default); what
  remains open is the unidentified root cause *inside the staging path*, now reachable only through
  `GGML_SCHED_DEVGATHER=0`.
* **#132** — the pool's decode deficit on a tensor split is the synchronous demand read (86 % of it), i.e. the
  overlap-worker question at a ~25-30 % ceiling on a split (it was correctly declined at ~6 % on one GPU).
* **#133** — a fully-resident (identity) expert table is still re-copied out of the arena per prefill ubatch in the
  **default** (`pinned`) mode, once per table per ubatch, serialised against compute (~3x prefill at 3 devices).
* **#136** — `GGML_SCHED_DEVGATHER`'s prefill-band default costs 33 % prefill on 35B-A3B at `FRAC 50`, where r207
  chose the default from a 122B/FRAC-95 measurement.

## Where the evidence is

The measured record is this directory's `README.md` (§1-§18, the per-session numbers and the gates),
`HANDOVER.md` / `HANDOVER-2026-11-02.md` / `HANDOVER-MULTIGPU.md` (the state at each handover), `HEAD-TO-HEAD.md`
(pool vs PR #267), `PROPOSAL-pipelined-staging.md`, `data/` (the runs) and `patches/` + `proposals/` (nothing in
either was applied to the fork).  The delivery-side statements are `patches/README.md` block 12 (r207) and
`INVARIANTS.md` r203/r207; the successor's are `archive/work/host-expert-eager-push/` and `INVARIANTS.md`
r210/r211.
