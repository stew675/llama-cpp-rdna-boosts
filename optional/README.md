# `optional/` — opt-in artifacts that are NOT part of the delivery

This directory holds **self-contained patches for features that are worth having on some hardware but
are not part of the delivery block set**.  Nothing here is in `patches/`, `release.json.tree`,
`release.json.tip`, `rdna-boosts-all.patch` or the fork's `rdna-boosts` branch, and a fresh
`scripts/apply-all.sh` checkout never contains it.

**The numbering (r288).**  The delivery uses three four-digit ranges, and only the first two are in the
default apply:

| range | meaning | default apply? | in `v<N>`? |
|---|---|---|---|
| `00xx` | official supported blocks (`0000`-`0015`) | yes | **yes — `N` = the count** |
| `03xx` | TEMPORARY block (`0300`, the `-sm relaxed-tensor` port) | yes, after the `00xx` blocks | no |
| `01xx` | **optional** block (this directory: `0100`, `0101`) | no — `--include-optional` only | no |

So an artifact here is **optional block `01xx`**, applied in **numeric order** after the verified default
tree.  `release.json` records `n_blocks` (official), `n_temporary`, `n_default` (the apply count) and
`n_optional`, with a separate `optional` map holding each artifact's sha256 and produced tree.

**Why it exists — and how it differs from `beta/`.**  `beta/` is a *validation window*: a candidate for
the delivery blocks, awaiting a reporter's test, which will be promoted or archived.  `optional/` is a
*finished* artifact: it has been reviewed and is kept on purpose, but it does **not** earn a default block
because it has no validated win on the delivery's own configurations, or because the maintainer has
decided the default set should not carry it.  The three cases that land here:

* the feature needs hardware the delivery does not validate (a heterogeneous GPU pair), or
* it is unreachable in the splits the delivery validates (`-sm tensor` is one backend — see
  `INVARIANTS.md`, "a backend-count gate is inert under `-sm tensor`"), so it cannot be measured here, or
* it is a genuine trade rather than a win (smaller KV at a few percent of decode — `mixed-kv`) and the
  default set stays at the official blocks plus the temporary port.

Taking such a patch into a block would put unvalidated code in everyone's build and owe the full release
gate ladder for a feature most consumers cannot use.  Taking it here gives the consumer who *does* have
the hardware a supported, rebased artifact, and keeps the delivery honest about what it can measure.

## Conventions

* One sub-directory per artifact: `optional/<name>/` with `README.md` and **`<NNNN>-<name>.patch`**
  (e.g. `optional/mixed-kv/0101-mixed-kv.patch`).  The stable name does not follow the commit subject, so
  a message edit does not silently rename the artifact; `release.json` keys the map on that basename.
* The patch is a `git format-patch` cut against **the current `release.json` default tree** (the applied
  `00xx` + `03xx` chain), and nothing else.  It must apply strictly (`git am`) on a fresh
  `scripts/apply-all.sh` checkout of *this* release.  Record its **sha256** and the tree it produces in
  the artifact's `README.md`.
* **Rebase-only-against-the-fully-applied-default-set.**  An optional patch is rebased onto a new release
  whenever the delivery moves; it is never cut against a working branch, a part-applied set or an older
  release.  If it no longer applies, that is a rebase task, not a reason to relax the strict apply
  (`apply-all.sh --include-optional` refuses a 3-way merge for exactly this reason).  **Enforced since
  r285:** `scripts/validate-set.sh` check 8 applies every `optional/*/01xx-*.patch` strictly on the
  freshly built default tree, checks each one alone produces its recorded tree, and checks that all of
  them applied in numeric order produce `release.json.optional_tree`, so a drifted artifact fails CI
  instead of a consumer's apply (`TODO` #324 -- both artifacts had gone 40+ releases stale before that
  check existed).
* **Nothing here is applied by default.**  `scripts/apply-all.sh --include-optional` steps through every
  `optional/*/01xx-*.patch` in **numeric block order** (`0100` before `0101`, not directory order) **on top
  of the verified default tree** and asks `[y/N]` for each one — `N` is the default — applying the accepted
  ones with strict `git am`, one commit each, and reporting the resulting tree, which is *not*
  `release.json`'s tree.  `--include-optional-all` applies every artifact without asking; it is the
  non-interactive form (a scripted apply, CI, a gate) and is required when there is no terminal to ask
  on.  The prompt names each artifact's patch subject, so a consumer opts in to the specific feature they
  want instead of the whole directory.  A later artifact may build on an earlier one, so the numeric order
  is the contract (and what CI verifies).
* **Not covered by the delivery's gates.**  An optional artifact is reviewed (it must build, must not
  change the default path, and must not break another backend), but its own claims are the artifact's
  responsibility, and its `README.md` must say which layouts it was measured on and which it cannot
  reach.  A consumer who applies one is off the supported delivery configuration.  An artifact whose
  feature has a dedicated gate names it (`mixed-kv` → `scripts/gate-mixed-kv.sh`), run on the
  optional-applied build.

## Promotion and demotion

An artifact is promoted into the delivery — folded into the block that owns the code it changes, with the
full release gate ladder — only when it has a **validated win on one of the delivery's own configurations**
(gfx1201 RDNA4, gfx1151 RDNA3.5, gfx1100 RDNA3; the splits `patches/README.md` names).  Until then it stays
here.  An artifact that turns out to be wrong is archived to `archive/work/<name>/` and recorded in
`WORKLOG.md`; no stubs are left behind.  An artifact whose **premise has been promoted** into the delivery
is archived the same way (`host-pool-fill-src` → `archive/work/`, r285: the delivery's
`MOE_HOST_POOL_TRANSPORT` is the general form of it), because a second copy of delivered code is drift
with no consumer.  The reverse also happens: a feature promoted into a block can be **demoted** to an
optional block by the maintainer (the r288 block-number re-org moved `mixed-kv` out of the default set as
`0101`); the artifact keeps the block's bytes and the record names the move.

## Current artifacts

| artifact | block | what | status |
|---|---|---|---|
| `two-lane-prefill/` | **`0100`** | `LLAMA_PREFILL_LANES=2`: two consecutive prefill ubatches on two backend schedulers with interleaved splits, plus the sub-ubatch split that makes a single wide batch pair (from PRs #268/#279/#296/#301, @slavap).  The MTP (nextn) **target** also pairs; a **draft** context (MTP via `ctx_type`, DFlash/EAGLE3 via `ctx_other`) is refused at the arm gate | **optional**, r288 re-cut (mechanical: the r285 diff applied strictly, unchanged).  Measured on the r285 cut: **+25.6 % prefill** (35B-A3B Q8_0, the 5 298-token prose prompt, `-sm layer -ot`, 3 pairs logged: 5426.6 → 6815.6 t/s), **+23.4 % MoE pp2048, +47.9 % at the default `-ub 512`, +22.3 % dense 27B prefill, +29.2 % on an MTP target** (r259 record, 3 x R9700 gfx1201); nil decode, text identical in every arm, default path **KLD 0.000697 / 98.784 %**.  The gain needs both halves.  Costs: a second compute-buffer set (~1.4 GiB/armed context), ~8 % on a second context (r229), **−6.1 % at `-npl 8`**, −0.9 % on a layer-input draft's target, and on the **host-expert tier** (`-ncmoe`) **−11.8 % at a wide ubatch** (+5.6 % at `-ub 512`; the r285 re-check reads −13.6 %) -- leave the knob off there.  Tree alone (r360) `831bc116a47559c4892b6f66f95fffbf869c76df` (unchanged bytes; re-cut against the r334 default tree, which is r333's plus the three RDNA2 items folded into blocks 05/08/10) |
| `mixed-kv/` | **`0101`** | K `q8_0` with V `q4_0` / `q5_0` / `q5_1` on a model whose attention layers are all 256-wide, read natively by the tile kernel's second V type instead of through the whole-cache F16 scratch; qwen4exp (QSA) and every other head size/pair stay refused; `GGML_CUDA_FA_KV_MIXED=0` restores the staged path (PR #312, @mjohanning99) | **optional** since r288 (it was **block 17** in r287 and moved out by the block-number re-org; the bytes are unchanged — applied alone it reproduces the r287 release tree `89b25552…`; the r360 cut alone produces `f455c3121bd4e1acd0a518c90eb947136cc5cd4c`).  `test-backend-ops -o FLASH_ATTN_EXT` 7021/7021 on gfx1201/gfx1100/gfx1151, W=1..8 purity maxdiff 0, KV 4352 → 3328-3712 MiB at `-c 131072` on 27B 3-GPU, decode 0.3-3.2 % below q8_0/q8_0 at 32768 cells.  Gate: `scripts/gate-mixed-kv.sh` (optional-applied build).  Tree alone (r360) `f455c3121bd4e1acd0a518c90eb947136cc5cd4c` (re-cut: block 05's stale RDNA2 caps are **deleted** in r334, and this artifact's two `!rdna_cap_ncols2` conditions were the only collision — the merge keeps the `type_V` argument and drops the prefix, so the mixed-KV matcher is unchanged) |

Cumulative tree with both applied in numeric order (`0100` then `0101`): `8bd172df5d98e698a6868d645afa626cdfc6c19a`
(`release.json.optional_tree`).
