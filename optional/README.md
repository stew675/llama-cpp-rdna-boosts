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
* **Nothing here is applied by default.**  `scripts/apply-all.sh --include-optional` applies every
  `optional/*/*.patch` (sorted) on top of the verified release tree, one commit each, and reports the
  resulting tree — which is *not* `release.json`'s tree.
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
| `two-lane-prefill/` | `LLAMA_PREFILL_LANES=2`: two consecutive prefill ubatches on two backend schedulers with interleaved splits, plus the sub-ubatch split that makes a single wide batch pair (from PRs #268/#279, @slavap) | **+27.4 % MoE 35B-A3B pp2048, +41.0 % dense 27B pp2048, +48.3 % at pp8192/-b 8192** on 3 x R9700 gfx1201 in the `-sm layer -ot` layout (the only one it reaches); nil decode, default path bit-identical.  The gain needs both halves -- the split alone is −15 %, the pair alone nil.  Costs: a second compute-buffer set (~1.4 GiB/context), ~8 % on a second context, −3.5 % at `-npl 8`, ~4 % if armed with a draft | **optional**, r229 re-cut, 2026-11-06 |
