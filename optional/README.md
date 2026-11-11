# `optional/` — opt-in artifacts that are NOT part of the delivery

This directory holds **self-contained patches for features that are worth having on some hardware but
are not part of the delivery block set**.  Nothing here is in `patches/`, `release.json` or
`rdna-boosts-all.patch`, and a fresh `scripts/apply-all.sh` checkout never contains it.

**Why it exists — and how it differs from `beta/`.**  `beta/` is a *validation window*: a candidate for
the delivery blocks, awaiting a reporter's test, which will be promoted or archived.  `optional/` is a
*finished* artifact: it has been reviewed and is kept on purpose, but it does **not** earn a block
because it has no validated win on the delivery's own configurations.  The two cases that land here:

* the feature needs hardware the delivery does not validate (a heterogeneous GPU pair), or
* it is unreachable in the splits the delivery validates (`-sm tensor` is one backend — see
  `INVARIANTS.md`, "a backend-count gate is inert under `-sm tensor`"), so it cannot be measured here.

Taking such a patch into a block would put unvalidated code in everyone's build and owe the full release
gate ladder for a feature most consumers cannot use.  Taking it here gives the consumer who *does* have
the hardware a supported, rebased artifact, and keeps the delivery honest about what it can measure.

## Conventions

* One sub-directory per artifact: `optional/<name>/` with `README.md`, and `<name>.patch`.
* `<name>.patch` is a `git format-patch` cut against **the current `release.json` tree**, and nothing
  else.  It must apply strictly (`git am`) on a fresh `scripts/apply-all.sh` checkout of *this* release.
  Record its **sha256** and the tree it produces in the artifact's `README.md`.
* **Rebase-only-against-the-fully-applied-delivery-set.**  An optional patch is rebased onto a new
  release whenever the delivery moves; it is never cut against a working branch, a part-applied set or
  an older release.  If it no longer applies, that is a rebase task, not a reason to relax the strict
  apply (`apply-all.sh --include-optional` refuses a 3-way merge for exactly this reason).
* **Nothing here is applied by default.**  `scripts/apply-all.sh --include-optional` steps through every
  `optional/*/*.patch` (sorted) **on top of the verified release tree** and asks `[y/N]` for each one —
  `N` is the default — applying the accepted ones with strict `git am`, one commit each, and reporting the
  resulting tree, which is *not* `release.json`'s tree.  `--include-optional-all` applies every artifact
  without asking; it is the non-interactive form (a scripted apply, CI, a gate) and is required when there
  is no terminal to ask on.  The prompt names each artifact's patch subject, so a consumer opts in to the
  specific feature they want instead of the whole directory.
* **Not covered by the delivery's gates.**  An optional artifact is reviewed (it must build, must not
  change the default path, and must not break another backend), but its own claims are the artifact's
  responsibility, and its `README.md` must say which layouts it was measured on and which it cannot
  reach.  A consumer who applies one is off the supported delivery configuration.

## Promotion

An artifact is promoted into the delivery — folded into the block that owns the code it changes, with
the full release gate ladder — only when it has a **validated win on one of the delivery's own
configurations** (gfx1201 RDNA4, gfx1151 RDNA3.5, gfx1100 RDNA3; the splits `patches/README.md` names).
Until then it stays here.  An artifact that turns out to be wrong is archived to `archive/work/<name>/`
and recorded in `WORKLOG.md`; no stubs are left behind.

## Current artifacts

| artifact | what | measured | status |
|---|---|---|---|
| `host-pool-fill-src/` | `MOE_HOST_POOL_FILL_SRC`: the host pool's **buffered fill** reads the file (a buffered `pread` through the OS page cache) instead of one `memcpy` per expert out of the model mapping — the second half of issue #281 (the first half, the coalesced span, is block 12 since r232) | **+38 % prefill on top of the coalesced span** at 26 % pool coverage (821-875 → 1138-1146 t/s) and 95 → 23.8 GiB of storage reads, **−41 % on its own** (a 0.7 MiB `pread` per fill is latency-bound at depth 1) and **exactly neutral once the pages are cacheable**.  A page-cache-headroom bet: it pays where the span pays (a sub-complement pool) and nowhere else, and `MOE_HOST_POOL_STAGE_SPAN=0` drops it into the slow arm | **optional**, r241 cut, 2026-11-09 |
| `two-lane-prefill/` | `LLAMA_PREFILL_LANES=2`: two consecutive prefill ubatches on two backend schedulers with interleaved splits, plus the sub-ubatch split that makes a single wide batch pair (from PRs #268/#279/#296/#301, @slavap).  The MTP (nextn) **target** also pairs now; a **draft** context (MTP via `ctx_type`, DFlash/EAGLE3 via `ctx_other`) is refused at the arm gate | **+23.4 % MoE 35B-A3B pp2048, +47.9 % at the default `-ub 512`, +22.3 % dense 27B prefill, +29.2 % on an MTP target** (r259 re-measure, 3 x R9700 gfx1201, `-sm layer -ot`, the only layout it reaches); nil decode, text identical in every arm, default path **KLD 0.000000**.  The gain needs both halves -- the split alone is a loss, the pair alone nil.  Costs: a second compute-buffer set (~1.4 GiB/armed context), ~8 % on a second context (r229), **−6.1 % at `-npl 8`**, −0.9 % on a layer-input draft's target, and on the **host-expert tier** (`-ncmoe`) **−11.8 % at a wide ubatch** (+5.6 % at `-ub 512`) -- leave the knob off there.  The r259 cut also **fixes an abort** this artifact had on `-ncmoe` since r246 (a lane-0 placement block re-ran for the lane-1 ubatch) | **optional**, r259 re-cut (PR #301 = the r254 rebase, plus this repo's host-expert abort fix and base-drift record), 2026-11-11 |
