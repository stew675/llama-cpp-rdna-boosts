# `closing-the-gap` → `beta/mmb-general` patch consolidation — plan

**Status:** PLAN — not started.  Created 2026-09-25.
**Branches:** delivery repo `gap-consolidation` (this file + the regenerated beta patches); fork
`~/llama.cpp` `gap-consolidation` (disposable mechanical branch, currently parked at `rdna-boosts-r13`).

**Purpose:** make `beta/mmb-general/patches/` the **single consolidated patch set**.  After this work,
applying `beta/mmb-general/patches/*.patch` to a fresh `rdna-boosts-r13` must reproduce the *final
combined* tree that today requires r13 + the 12 beta patches + the 30 `wip/closing-the-gap/patches/`
patches.  The closing patches then remain only as the historical/research record.

> Session-continuation note: everything needed is in §1 (immutable reference state), §2 (the mapping)
> and §4 (validation).  Read the "Execution checklist" (§7) last.

---

## 0. Definition of done

1. `~/llama.cpp` `gap-consolidation` carries a rebuilt series of **N consolidated beta commits** whose
   tip tree is **`468c64963ae45e72367c73809efa7cc038217e8a`** (T_final).
2. A **fresh** `rdna-boosts-r13` + `git am beta/mmb-general/patches/*.patch` (strict, no 3-way)
   reproduces T_final exactly, with the right commit count.
3. The consolidated set **builds clean** (gfx1151 `~/bin/build-llama-rocm-714`, 0 errors) and passes
   the standard gates (coherence, width probe, op oracles, MTP acceptance; the P1–P3 A/Bs are
   optional re-confirmation).
4. `beta/mmb-general/{commits.txt,README.md,BETA-TESTING.md,combined-set-verification.md,HANDOVER.md}`
   and `mmb-general.patch` are regenerated/updated; `wip/closing-the-gap/README.md` marks the
   consolidation.
5. `wip/closing-the-gap/patches/` is kept as history (marked superseded/consolidated), **not** deleted.

**Out of scope:** promoting the beta set to a delivery block, `release.json`, the r13 delivery patches.

---

## 1. Immutable reference state (record before touching anything)

| name | commit | tree |
|---|---|---|
| base | `8491bf2bff8eb3a56e5120c3c9c17533a94ea6bf` (branch `rdna-boosts-r13`) | `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12` |
| beta tip (r13 + 12 beta) | `aa1e87b8f827aa258a1faf947ba29f1ff36b627c` | `79136a15cac1920c0dd334b4c119a9cb42f9143b` |
| final combined (r13 + 12 beta + 30 closing) | `e72c2a17d2f0c89ba9ff4e0bbd193ce21ec3e20a` | **`468c64963ae45e72367c73809efa7cc038217e8a`** |
| beta series | `8a3ef224e..aa1e87b8f` (12 commits) | — |
| closing series | `18a9008d8..e72c2a17d` (30 commits) | — |

Fork branch `gap-closing-denseband` @ `e72c2a17d` is the **reference tree**; do not rewrite it.  Tag it
`ref-combined` (and `ref-beta` at `aa1e87b8f`) before starting so the comparison points survive.

The closing set is **30 patches** (`0001..0014`, `0016..0031`; `0015` was removed — it is r13 block 00).

## 2. The mapping — each closing patch's fold target

Two outcomes per closing patch:
* **FOLD** — its change is squashed into an existing beta commit (the beta patch whose feature it
  extends).  The beta patch file then *contains* the closing update.
* **NEW** — it is a feature that no beta patch owns (base-code or an independent capability); it is
  appended as a new beta patch after `0012`.

The "basis" column is the file/feature evidence from the combined branch; **the exact fold target is
confirmed at execution time by `git blame` on each modified hunk** (a mismatch is a plan bug, fix the
mapping, do not force it).

| closing | subject (short) | proposed target | basis |
|---|---|---|---|
| 0001 | `18a9008d8` revive hc_combine_norm | **FOLD → beta 0004** (HC16) | MMB graph-optimizer HC fusion; touches beta-0004 `ggml-cuda.cu` |
| 0002 | `a1e4332f3` default beneficial features ON | **FOLD → beta 0008** (per-arch defaults) | flips the mmb/HC16/gate-mix defaults |
| 0003 | `c86bf6c19` hc_gate_mix fusion | **FOLD → beta 0004** | HC BF16 producer |
| 0004 | `f94c87d1b` depthwise conv1d (GDN+PLE) | **NEW** (prefill fusions) | `gdn-conv.*`/`ple-conv.*` base files |
| 0005 | `d6c3ea7b6` QSA block window | **FOLD → beta 0002** (qsa3) | qwen4exp / QSA |
| 0006 | `35da4bb0f` narrow-row RMS norm | **NEW** (prefill fusions) | `norm-gated.*` base file |
| 0007 | `36de17cac` QSA visibility fold | **FOLD → beta 0002** | `fattn-qsa3.cu` |
| 0008 | `37038da53` M=4 inject / TALL_MIN_M | **FOLD → beta 0007** (dense tile) | tile geometry |
| 0009 | `816424b7d` `-lzm auto` semantics | **NEW** | CLI/model-loader (base) |
| 0010 | `39a7934fc` MoE BF16 epilogue (DOWN16) | **FOLD → beta 0004** | HC16/MoE BF16 producer |
| 0011 | `c474e128b` HC BF16 streams | **FOLD → beta 0004** | HC BF16 producer |
| 0012 | `f7612213c` mmb_cvt `out_xn` | **FOLD → beta 0004** | HC BF16 producer |
| 0013 | `98eaafa1b` prefill indexer relu+head-sum | **FOLD → beta 0005** (indexer top-k) | `indexer-score.*` |
| 0014 | `cc0357fe2` QSA scorer trim | **FOLD → beta 0005** | `indexer-topk.cu` |
| 0016 | `42380542c` QSA_SCORE_WMMA | **FOLD → beta 0005** (or NEW if `lightning-indexer.cu` is base-owned) | `lightning-indexer.cu` + qwen4exp |
| 0017 | `2404d3f54` MMB quant Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 | **FOLD → beta 0001** (mmb) | weight types |
| 0018 | `88247a13d` MMB quant IQ2_S/IQ2_XS/IQ2_XXS | **FOLD → beta 0001** | weight types |
| 0019 | `d461b905b` HC16 eval-callback F32 fix | **FOLD → beta 0004** | HC16 marking correctness |
| 0020 | `a37f6ed6e` sparse MTP-draft attention | **NEW** (MTP group) | qwen4exp / memory |
| 0021 | `17ca09d81` QSA derived indexer default ON | **FOLD → beta 0005** (or → NEW MTP group) | default of a beta-0005 cache |
| 0022 | `3a5278008` gfx1151 QSA decode crossover 32K | **FOLD → beta 0005** (or → NEW) | qwen4exp arch policy |
| 0023 | `f2a217ec3` HC16 per-backend-context | **FOLD → beta 0004** | HC16 state |
| 0024 | `24153c735` input layer on GPU (single device) | **NET-FOLD with 0025** | superseded by 0025 |
| 0025 | `2b86e8b8b` host-buffer input layer | **NEW** (input-layer group) | backend + model |
| 0026 | `5ba0d2c33` sparse MTP prefill default ON | **FOLD → 0020's new patch** | default flip |
| 0027 | `84e527c48` meta `graph_optimize` under `-sm tensor` | **NEW** | meta backend |
| 0028 | `83abc1eff` MMVQ↔MMQ band boundary | **NEW** (MMVQ band group) | `mmvq.*` |
| 0029 | `f6e56b1a8` tiny CPU split graphs single-thread | **NEW** | `ggml-cpu.cpp` |
| 0030 | `726b37b76` opt-in device input placement | **NET-FOLD with 0025** | opt-in on 0025 |
| 0031 | `e72c2a17d` gfx1151 MMVQ band policy | **FOLD → 0028's new patch** | override of 0028 |

### Proposed new beta patches (appended after 0012)

Order to satisfy dependencies; each is one commit (or a net-fold where noted):

1. `0004` depthwise conv1d (GDN+PLE)
2. `0006` narrow-row RMS norm
3. `0009` `-lzm auto` semantics + managed PLE reader gate
4. `0020`+`0026` sparse MTP draft (net: opt-in then default-ON)  ← decide: keep two commits or one
5. `0021` derived indexer default (if not folded into beta 0005)
6. `0022` gfx1151 QSA decode crossover (if not folded into beta 0005)
7. `0024`+`0025`+`0030` input-layer placement (net: host-buffer + optional device input; `0024` folded
   into `0025` per the "supersedes" relationship)
8. `0027` meta `graph_optimize` forwarding
9. `0028`+`0031` MMVQ↔MMQ band (net: gfx1201 bands + gfx1151 policy)
10. `0029` tiny CPU split graphs single-thread

Resulting beta set ≈ **12 folded + ~8–10 new ≈ 20–22 patches**.  Exact count is a function of the
fold decisions confirmed in step 3.

---

## 3. Mechanics — fresh linear rebuild (preferred over interactive rebase)

Work in `~/llama.cpp` on `gap-consolidation` (recreate it from `rdna-boosts-r13` if it was moved):

```sh
cd ~/llama.cpp
git tag -f ref-combined e72c2a17d
git tag -f ref-beta     aa1e87b8f
git checkout -B gap-consolidation rdna-boosts-r13
```

For each **consolidated beta patch** `B` in order `0001..0012`:

```sh
git cherry-pick -n <beta_B_commit>                 # stage the beta change
for c in <closings mapped to B, original order>; do
    git cherry-pick -n "$c"                        # stage each closing; resolve conflicts here
done
git commit -C <beta_B_commit>                       # one commit; subject reworded to the combined feature
```

Use `git commit -C <beta_B_commit>` to keep author/date, then `git commit --amend` to reword the
subject so it describes the combined feature (e.g. "WIP mmb: ... + the closing HC B…).

For each **appended closing** `C`: `git cherry-pick "$C"` (one commit each), except the explicit
net-folds (`0024` into `0025`; `0026` into `0020`; `0030` into `0025`; `0031` into `0028`) where the
pair is one commit.

**Conflict guidance.**  A closing patch was developed on the *full* beta tree, so a few will conflict
when replayed earlier.  Resolve to the **net combined content**, then let §4 checkpoint validation
prove nothing was lost.  If a conflict makes the fold unsafe, demote that closing to **NEW** rather
than force it (a demoted patch is still a valid consolidation).

---

## 4. Validation — incremental, at every checkpoint

The invariant is **net-content preservation**, not commit-shape preservation.  After each consolidated
beta patch (or each appended patch) is committed at checkpoint `k`, build a throwaway tree and assert:

```
base
  + consolidated patches 1..k            # the new series so far
  + original beta patches (k_beta+1 .. 12)   # the betas not yet folded into
  + closing patches not yet processed       # in original order
  == T_final (468c6496…)
```

Mechanised as a helper (`/tmp/consol-check.sh`), run in a scratch worktree:

```sh
W=$(mktemp -d); git worktree add --detach "$W" rdna-boosts-r13
( cd "$W" \
  && git am <consolidated patches so far> \
  && git am <remaining beta patches> \
  && git am <remaining closing patches in original order> ) || { echo APPLY-FAIL; }
got=$( cd "$W" && git rev-parse HEAD^{tree} )
[ "$got" = 468c64963ae45e72367c73809efa7cc038217e8a ] && echo CHECKPOINT-OK || echo TREE-MISMATCH
git worktree remove --force "$W"
```

* **Early checkpoints** (only a few closings folded) exercise the "remaining closing patches still
  apply" path — a failure here means a fold changed the net content or the ordering.
* Once all closings are folded/appended, the invariant degenerates to `base + consolidated == T_final`.

**Per-fold audit:** `git range-diff rdna-boosts-r13..e72c2a17d rdna-boosts-r13..gap-consolidation`
must show every closing commit as a content-preserving `=` mapped into its beta commit (or a `pick`),
with **no** unexplained content deltas.  This is the human-readable proof that a fold moved code
without changing it.

**Final gates:**
1. `git rev-parse HEAD^{tree}` == `468c64963ae45e72367c73809efa7cc038217e8a`.
2. Re-export the series (`git format-patch --start-number 1 rdna-boosts-r13..gap-consolidation`), copy
   to `beta/mmb-general/patches/`, then **fresh-apply** a clean r13 clone with strict `git am` (no
   `-3`) and re-assert the tree + commit count.
3. Build clean (gfx1151) and run: same-seed coherence; `test-logits-width-probe` PASS; op oracles
   (`FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46); MTP acceptance gate
   (`benchmarks/mtp-adaptive-methodology.md` rule 0); and the P1/P2 spot checks
   (`llama-batched-bench` qwen4exp B=8→12 monotone, P2 `n7`/`n8` matrix).
4. Confirm the closing-only default flips still landed: `GGML_CUDA_MMB_CFG=1` on gfx1151 reads the
   same `cc=0x1001151 … dense_geom=0 … routed=1` row, and `GGML_CUDA_DISABLE_MMVQ_MOE_BAND` is a
   no-op there while `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND` still reverts.

---

## 5. Deliverables / repo updates

* `beta/mmb-general/patches/` — the regenerated consolidated series (replace the current 12 files).
* `beta/mmb-general/mmb-general.patch` — regenerate (`git diff rdna-boosts-r13..gap-consolidation`).
* `beta/mmb-general/commits.txt` — regenerate from the new series.
* `beta/mmb-general/README.md` — rewrite "What is in the beta set" + the per-patch table.
* `beta/mmb-general/BETA-TESTING.md`, `combined-set-verification.md`, `HANDOVER.md` — add a
  consolidation section (the new apply command is simply r13 + `git am beta/...`).
* `wip/closing-the-gap/README.md` — mark the patches as consolidated into `beta/mmb-general` (keep
  `patches/` as history; do not delete).
* This file — append a dated "executed" section with the checkpoint log and the final hashes.
* Commit on `gap-consolidation` and push only if the maintainer asks (per `AGENTS.md` pushing policy).

---

## 6. Risks & traps

* **Reordering conflicts** — closings were written on the full beta tree; replaying them earlier can
  conflict.  Resolve + checkpoint; demote to NEW if unsafe.
* **GSM/`-sm tensor` correctness gate (`0004` conv1d, `0019`)** — the conv1d fusion is gated to
  single-device graphs; that gate must survive its fold.  Re-run the gfx1201 `-sm tensor` check if the
  fold touches it.
* **MMB HC16 state (`0023`) and the eval-callback fix (`0019`)** — both touch the shared
  `ggml_backend_graph_optimize_params` plumbing; folding both into beta 0004 must keep the
  `has_eval_callback`/`full_graph` fields and the per-context activation state.
* **Default-on policy (`0002`)** — was written to flip the *closing* defaults; folding it into beta
  0008 must not flip the beta's *opt-in beta-tree* defaults (the beta README documents that MMB is
  opt-in in the beta tree).  Decide explicitly: the consolidated set is the **campaign** state
  (beneficial features ON), so the closing defaults win — record that.
* **The gfx1201/gfx1151 MMVQ split (`0028`/`0031`)** — fold `0031` without flattening the per-arch
  gate; gfx1151 must keep the 8-wide routed MoE band and the RDNA3_5 dense band.
* **`0015`** — must stay absent (r13 block 00 owns that fix).
* **Build-time / matrix** — the consolidated beta set is larger; keep the FA instance discipline
  (`nm` check) and do not leave an un-instantiated KV-type arm.
* **Don't rewrite `gap-closing-denseband`/`e72c2a17d`** — it is the reference.

---

## 7. Execution checklist

1. [ ] Tag `ref-combined`/`ref-beta` in `~/llama.cpp`; recreate `gap-consolidation` from `rdna-boosts-r13`.
2. [ ] Confirm each closing patch's fold target by `git blame` on its hunks; finalise §2.
3. [ ] Fold `0001..0012` in order (cherry-pick + commit per beta), checkpoint after each.
4. [ ] Append the NEW closings (net-fold `0024/0025/0030`, `0020/0026`, `0028/0031`); checkpoint after each.
5. [ ] `git range-diff ref-combined gap-consolidation` — every closings's content accounted for.
6. [ ] Assert tip tree == T_final.
7. [ ] Export the new series to `beta/mmb-general/patches/` (replace the 12).
8. [ ] Fresh strict `git am` on a clean r13 → tree == T_final.
9. [ ] Build clean; run the gates (§4 final).
10. [ ] Regenerate `commits.txt` + `mmb-general.patch`; update the beta + closing READMEs and this file.
11. [ ] Commit on `gap-consolidation`; hand to the maintainer for review/push.
