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

* **`prefill-decode-arms`** — **the two-arm server (the withdrawn delivery block 17), REBUILT as v2** (2026-11-24).
  **ONE `llama_memory_i`, TWO `llama_context`s**: the prefill arm owns the memory and the decode arm borrows it
  through `llama_context_params::ctx_memory`; nothing is copied between them.  This **replaces the earlier
  artifact in its entirety** (the maintainer's decision of 2026-11-24 — the old design's second context that
  *viewed* and *co-owned* the first's memory, and the recurrent state copied between two modules, are gone at
  the root); on promotion it is **delivery block 17 as ONE self-contained patch**, a modification to the
  lower-numbered blocks.  Applied on the **r355** default tree (`4244d077e`) it produces tree
  **`dc9c657e60`** (no re-cut was needed: the patch files are unchanged, and every produced tree moved with
  the default tree).  Files: [`prefill-decode-arms/README.md`](prefill-decode-arms/README.md) (what it is, the
  measured numbers, the residuals, the shapes it refuses),
  [`prefill-decode-arms/BETA-TESTING.md`](prefill-decode-arms/BETA-TESTING.md) (the apply, build and arm ladder)
  and [`prefill-decode-arms/prefill-decode-arms.patch`](prefill-decode-arms/prefill-decode-arms.patch)
  (**THREE** patch files, applied in filename order, one commit each: `prefill-decode-arms.patch` — block 17,
  sha256 `a2badad1…e366fe3c`, 42 files, +3073/−741 — then two delivery-level fixes the candidate carries for
  now, `state-split-2d-read.patch` (STEP 15, sha256 `b3f0645a…c2a7336b`, 1 file, +85/−46) and
  `two-arm-drop-boundary.patch` (STEP 16, sha256 `fd50d736…d31030963`, 4 files, +97/−4)).
  Campaign record:
  [`../wip/prefill-decode-arms-v2/`](../wip/prefill-decode-arms-v2/README.md); its own gate set is
  `wip/prefill-decode-arms-v2/gates/gate-arms-*.sh` (beta-only, never delivery gates).

The `arena-steady-sizing` v2 candidate was **promoted into block 09 at r292** (2026-11-15);
see *Recently promoted* below and `archive/work/arena-steady-sizing/`.
`0016-host-experts-window` was **withdrawn 2026-11-02**: the mode is unimplementable on a
no-XNACK GPU (a kernel read of a reclaimable registered page is a fatal fault, and holding it is pinning —
the RAM saving disappears).  It is archived with a resolution at
[`../archive/work/host-experts-window-beta/`](../archive/work/host-experts-window-beta/README.md) and
[`../archive/work/host-expert-pinned-window/RESOLUTION.md`](../archive/work/host-expert-pinned-window/RESOLUTION.md);
the two mechanisms worth keeping moved to
[`../archive/work/host-pool-speed/`](../archive/work/host-pool-speed/HANDOVER-2026-11-02.md) (the campaign has
since retired there).

(Empty before this: both earlier candidates were promoted in `v12-097cf33d9-r86`, 2026-10-19.)_

## Recently promoted

* **`arena-steady-sizing` v2** — promoted into **block 09** at **r292** (2026-11-15), no new block.  The
  expert arena is now settled against the capacity a prefill -> decode drop returns, demand-gated at the
  first drop on the marginal decode-band prior mass (`MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP`, default 1.0 pp)
  and sticky per layer with the capacity it was achieved at (two-slot transitions stable).  Measured: 1 x
  R9700 +8.2 % at `-ub 16384`, +31.9 % at 32768, partial pool +36.8 %; 24 GiB `gfx1100` +19.5 %; the
  saturated 2-device splits and a small `-ub 2048` are byte-identical to base.  Record:
  `archive/work/arena-steady-sizing/`; `WORKLOG.md` 261115.005/.006 and the r292 release entry.

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
