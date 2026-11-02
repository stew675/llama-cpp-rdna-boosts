# `beta/` — promotion-candidate patches (NOT part of the delivery)

This directory holds **self-contained patches that are candidates for folding into the delivery block
set**, staged for a validation window before the maintainer decides.  Nothing here is part of
`patches/`, `release.json` or `rdna-boosts-all.patch`, and a fresh `scripts/apply-all.sh` checkout
never contains it.

**Why it exists.**  A candidate that changes a file a delivery block owns cannot be shipped as a
separate `wip/` item (it would conflict with the release flow) and cannot be committed straight into
the blocks (it has not been validated on the reporter's platform yet).  `beta/` is the holding area
for exactly that: the patch is cut against the current delivery tree, handed to the reporter for a
beta test, and only folded into the owning block once the result comes back.  This is the same area
that carried `beta/mmb-general` and `beta/tensor-fit-fix` (both promoted; see `archive/work/`).

## Conventions

* One sub-directory per candidate: `beta/<name>/`.  A numeric prefix orders it relative to the block
  set (`0014-...` applies after block 13).
* `README.md` — what the candidate is, why, the patch, and the status (dated entries, newest first).
* `BETA-TESTING.md` — the apply + build + gate checklist handed to the tester.
* `<name>.patch` — the patch itself, cut against the current `release.json` tree.  It must `git apply`
  and `git am` cleanly on a fresh `apply-all.sh` checkout; record its **sha256** and the applied tree
  in `README.md`.
* **Nothing here is applied by default.**  `scripts/apply-all.sh` ignores `beta/` unless the tester asks
  for it explicitly with `--include-beta`, which applies every `beta/*/*.patch` (sorted) on top of the
  verified release tree, one commit each, and then reports the resulting tree (which is *not*
  `release.json`'s tree -- that is the point).  Without the flag the script behaves exactly as before.

## Promotion

When the tester confirms the candidate, it is folded into the block that owns the code it changes (or
another block, at the maintainer's discretion), the block set is regenerated and re-validated, and the
candidate moves to `archive/work/<name>/` and is recorded in `WORKLOG.md`.  If it is refuted, the same
move happens with a "refuted" status.  A finished candidate leaves `beta/`; no stubs.

## Current candidates

* **`0016-host-experts-window`** (2026-11-02) — `--host-experts window`: a third host-expert mode that
  registers each host expert tensor **in place** so the page cache itself is device-readable, and the
  pinned pool stops being a tier at all.  1 GPU: **+17-35 % decode** and **−24.8 GB peak RSS** on the
  cells that spill, MTP four-axis acceptance identical to 5 decimals, every kill-switch arm
  byte-identical, correct inside a 10 GiB cgroup.  **Held for the multi-GPU / `-sm tensor` arm**: the
  per-host-tensor registration refcount (three per-device children over one master) has never run on
  more than one device.  Patch sha256 `3502e334…`, applied tree `5101f0e4…`.  Record:
  [`0016-host-experts-window/README.md`](0016-host-experts-window/README.md) and
  `WORKLOG.md` 261102.006.

(Empty before this: both earlier candidates were promoted in `v12-097cf33d9-r86`, 2026-10-19.)_

## Recently promoted

* **`0012-slab-transient-absorption`** — promoted into **r86**, folded per owning block: **block 09**
  (derived draft reserve, draft model loaded before the target context, slab-served common/all-reduce
  transients), **block 10** (wide-phase auxiliary region, workspace-pool redirect, capture safety,
  per-device diagnostics, headroom `640 + 128 x n_gpu`) and **block 11** (loaded-draft ownership handoff).
  Record: `archive/work/slab-transient-absorption/RESOLUTION.md`; `WORKLOG.md` 261019.001.

* **`0013-meta-hot-dispatcher`** — promoted into **r86**, folded into **block 00** (issue #184).  The
  split-state dispatcher is the exact pre-#170 shape; the #170 alias/host resolution is a `noinline, cold`
  helper.  Record: `archive/work/issue-184-meta-r80r82/0013-meta-hot-dispatcher/RESOLUTION.md`;
  `WORKLOG.md` 261019.001.

* **`slab-narrow-floor`** — promoted into **r85**, blocks 09/10 (the slab narrow band is learned from the
  real decode graph).  Record: `archive/work/slab-narrow-floor/`; `WORKLOG.md` 261018.007.

* **`fa-stage-prealloc`** (issue #138) — promoted into **r53**, blocks 09/10: the FA prefill staging arena
  is allocated once at its eventual size; default-on with `GGML_CUDA_FA_STAGE_PREALLOC=0` as the kill
  switch.  Record: `archive/work/fa-stage-prealloc/`; `WORKLOG.md` 261013.003.

* **`rdna2`** (issue #158) — folded into **r70**, blocks 06/10; record `archive/work/rdna2/`.
