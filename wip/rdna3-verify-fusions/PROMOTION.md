# Promotion: rdna3-verify-fusions (PR #192 + PR #193)

This is the fold recipe and release-gate checklist for the main development box.  The candidate is
validated on gfx1100 against the current r94 tree; see `README.md` for the evidence.

## 1. Fold targets in the 12-block set

The two patches are independent.  Fold each hunk into the block commit that owns its pre-image line.
On the canonical r94 chain (`release.json` base `097cf33d9`, tip
`c6b8107a37c4350cb283e9ae4f1fdb2d7a3046c1`, tree `40d43e3044051ea8168c02cd1a899877b5d8574d`) that is:

| patch / hunk | file | block |
|---|---|---|
| `0001` (PR #192), all hunks | `ggml/src/ggml-cuda/mmvq.cu` (the `verify_band` gate and comment) | **block 10** |
| `0001` (PR #192), comments | `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` | **block 10** |
| `0002` (PR #193), `gdn-conv.cu` hunk | `ggml/src/ggml-cuda/gdn-conv.cu` (the `verify_ok` line and comment) | **block 10** |
| `0002` (PR #193), `cpy-batch` hunk | `ggml/src/ggml-cuda/cpy-batch.cu`, `cpy-batch.cuh` | **block 09** |

Notes:

- The pre-V2.0 `patches/README.md` text calls the verify-band norm/gated unary fusions "block 08"; in
  the V2.0 map the current `mmvq.cu`/`norm.cu`/`unary.cu`/`ggml-cuda.cu` lines are block 10's, and
  `cpy-batch.*` is block 09's.  Confirm with `git blame <file>` on the **pre-fold** canonical tree before
  folding; if a line has moved, use the owning block, not the table above.
- `0002` is one commit covering two blocks.  Split it by file when folding: `cpy-batch.*` to block 09,
  `gdn-conv.cu` to block 10.  The delivered tree must be identical either way; do not hand-edit the
  committed `patches/`.
- The `verify_band_arch` and `verify_ok` expressions are `A || B` with the existing RDNA4 term kept, so
  RDNA4 and every non-RDNA3_0 backend stay on the same path.  RDNA3_5 is deliberately not included.

## 2. Fold procedure

1. In a clean `~/llama.cpp`, check out upstream `master` at `release.json.base` (`097cf33d9`), not the
   tip of `master`.  Run `scripts/apply-all.sh .` to create the 12-block `rdna-boosts` branch.
2. Apply the two candidate patches as two extra commits (`git am`), so the pre-fold tree is the r94
   tree plus the two fusions.  Build it and run the quick gates below once.  Record the pre-fold tree.
3. Fold the hunks into the owning block commits (block 09 for `cpy-batch.*`, block 10 for the rest).
   Use `git rebase -i` or apply the 13th/14th commits and `git commit --amend` the owning blocks; keep
   both sides on a replay conflict.  The folded 12-block tree must equal the pre-fold tree.
4. Regenerate the delivery artifacts from the canonical chain, never by hand:
   - `scripts/make-patches.sh`
   - `scripts/make-release.sh`
   - `scripts/validate-set.sh` on a fresh tarball (strict 12/12 `git am`, applied tree == `release.json`)
5. Move this directory to `archive/work/rdna3-verify-fusions/` (no stub in `wip/`), remove its row from
   `wip/CAMPAIGNS.md`, and add a `WORKLOG.md` entry with the release tag and gate results.

## 3. Documentation to update in the same release

- `ENVIRONMENT.md` §7.1: the three kill switches already exist; extend their arch note from "RDNA4" to
  "RDNA4 and RDNA3_0 (gfx1100-1102), RDNA3_5 excluded".  No new environment variable is introduced.
- `patches/README.md`: add a line to the block 09 and block 10 notes (the batched f32 CPY on RDNA3_0;
  the verify-band Q8_1 fusions, GDN conv verify path on RDNA3_0).
- `GREEDY-PURITY.md` §11 (the verify-band purity range) and §19 (purity-first policy): record that the
  RDNA3_0 band now takes the same fusions, with the gfx1100 W=1..8 hashes and the same-binary
  kill-switch identity from `README.md`.  This is a purity claim, not only a performance claim.
- `benchmarks/` if a new dated bench record is wanted; the numbers in `README.md` are sufficient for a
  WORKLOG entry.

## 4. Release gates (run on the folded tree)

The gfx1100 checks in `README.md` are the acceptance evidence for the candidate tree.  Re-run the
decisive ones on the folded tree, and run the full gate set before tagging:

**Every platform**

- `scripts/apply-all.sh` on a fresh checkout at `release.json.base`, build with
  `~/bin/build-llama-rocm` (or the box's canonical build), and `scripts/validate-set.sh`.

**gfx1201 (RDNA4) scope check, must be byte-identical to r94**

- Same-seed coherence text, `test-logits-width-probe` (27B, `W=1..8`), and
  `scripts/gate-prefill-logits.sh`.  Any diff means the RDNA3_0 scope leaked into RDNA4.

**gfx1100 (RDNA3_0), the platform the change targets, if a box is available**

The candidate tree already has the acceptance evidence in `README.md`.  Re-run the decisive gates on
the folded tree if this box (or a sibling) has a gfx1100.  If not, the candidate-tree evidence stands
and the WORKLOG entry must say the folded-tree gfx1100 re-run was not performed; do not re-label the
candidate evidence as a folded-tree result.

- `test-logits-width-probe` on 27B, 30B-A3B and 35B-A3B (prefix 256 and 3584, `q8_0` KV), plus the
  same-binary kill-switch A/B (`GGML_CUDA_FUSE_Q8_1_VERIFY=0`, `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`,
  `GGML_CUDA_FUSE_CPY_BATCH=0`).  Expect all hashes identical and maxdiff 0.
- Greedy text, `--spec-type none` == `--spec-type draft-mtp` at `n_max <= 7`, on a dense and an MoE
  model (400 tokens is enough for the text check; the four-axis gate is below).
- Four-axis MTP gate at `-n 3000` (R/P/C/K, reasoning pinned) and the phase-switching prompt, per
  `benchmarks/mtp-adaptive-methodology.md`.  The author measured fixed n3 only; confirm adaptive.
- Stock-relative `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (dense, `q8_0` KV): B=1 within noise,
  B=4/B=8 no worse.
- `scripts/gate-prefill-logits.sh` (expect rounding-level, as on the unfolded tree).
- `test-backend-ops -b ROCm0 -o RMS_NORM,ADD,MUL_MAT,CPY,CONCAT,SSM_CONV` (3130/3130 on the unfolded
  tree) and `scripts/check-gpu-faults.sh` on the run logs.

**gfx1151 (RDNA3_5), if a box is available**

- Unchanged by design (the macros exclude 0x1150-0x11ff).  Build and run the coherence gate; the full
  gate set is optional.  Record the "not enabled" state rather than a performance result.  If no box is
  available, record that the folded tree was not built on gfx1151.

**Multi-device (open gap)**

- The GDN conv fusion keeps its single-device restriction, so it is not a multi-device change.
- The batched f32 CPY is **not** single-device restricted.  On an RDNA3_0 multi-GPU box, run a GDN
  model with MTP under `-sm tensor` and `-sm layer`, and the width probe with `SPLIT=tensor`.  If no
  such box is available, record the gap explicitly in the WORKLOG entry; do not claim multi-device
  validation from a single-GPU run.

## 5. Release mechanics

After the folded tree is green: regenerate `release.json`, tag `v12-<base>-rNN`, push the delivery,
refresh the personal fork's `rdna-boosts` branch (strict apply at `release.json.base`, build, gates,
`--force-with-lease`), and let the `radiance` worker watcher build/push the ROCm images.  Follow
`AGENTS.md` "Pushing policy" and `CONTAINERS.md`; do not build the full container set on the primary
box.
