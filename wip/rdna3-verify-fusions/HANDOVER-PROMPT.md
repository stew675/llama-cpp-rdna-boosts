# Handover prompt: promote rdna3-verify-fusions into the next release

Copy the block below to the main development box.

---

You are taking over a validated promotion candidate in the delivery repo `~/llama-cpp-rdna-boosts` on
branch `rdna3-pr-192-193`.  Read `AGENTS.md` first, then `wip/rdna3-verify-fusions/README.md`,
`wip/rdna3-verify-fusions/PROMOTION.md` and the two patches in that directory.  The candidate is
PR #192 plus PR #193, two RDNA3_0 arch-gate extensions for existing verify-band fusions.  It is
accepted for promotion into the 12-block V2.0 delivery.  Your job is to fold it into the owning block
commits, regenerate the delivery, run the release gates, and prepare the next full release.

## Context

- Current release: `v12-097cf33d9-r94` (`release.json`: base `097cf33d9`, 12 blocks, tip
  `c6b8107a37c4350cb283e9ae4f1fdb2d7a3046c1`, tree `40d43e3044051ea8168c02cd1a899877b5d8574d`).
- Candidate: `wip/rdna3-verify-fusions/0001-mmvq-the-verify-band-Q8_1-fusions-on-RDNA3_0-too.patch`
  (PR #192) and `wip/rdna3-verify-fusions/0002-cuda-the-GDN-conv-verify-path-and-the-batched-f32-CP.patch`
  (PR #193).  Both are re-cut against the r94 tree and apply with `git am -3`.
- The gfx1100 acceptance evidence is in `wip/rdna3-verify-fusions/README.md`: W=1..8 probe maxdiff 0 on
  27B, 30B-A3B and 35B-A3B, greedy text identical for `none` and `draft-mtp`, same-binary kill-switch
  identity for all three fusions, prefill-logit KLD 0.000000 / 100.000 %, `test-backend-ops` 3130/3130
  on ROCm0, no GPU faults, +4.1 % prose / +4.2 % code MTP n3 on 27B on r94, and batched-bench B=4/B=8
  no worse.
- The change is RDNA3_0-only.  RDNA4 and RDNA3_5 must be unchanged.  The GDN conv fusion keeps its
  pre-existing single-device restriction; the batched f32 CPY does not, and that half has no
  multi-device measurement yet.

## Do this, in order

1. **Report findings first.**  If you find anything in the patches, the fold, or the gates, stop and
   tell the maintainer before the delivery commit and before any release or tag (AGENTS.md "Report
   findings before shipping").  Do not bury a finding in a commit message or a post-push summary.
2. **Build the pre-fold tree.**  In a clean `~/llama.cpp`, check out upstream `master` at
   `release.json.base` (`097cf33d9`), run `scripts/apply-all.sh .` to create the 12-block branch, then
   `git am` the two candidate patches as two extra commits.  Build and confirm the pre-fold tree is
   the r94 tree plus exactly these two changes.
3. **Fold.**  Put the `cpy-batch.cu` / `cpy-batch.cuh` hunk into **block 09**; put the `mmvq.cu`,
   `gdn-conv.cu`, `ggml-cuda.cu` and `ggml-cuda-try-fuse.cu` hunks into **block 10**.  Confirm the
   owning block with `git blame` on the pre-fold tree if a line has moved; the V2.0 map above is what
   r94 has.  Keep both sides on a replay conflict.  The folded 12-block tree must equal the pre-fold
   tree.
4. **Regenerate and validate.**  Run `scripts/make-patches.sh`, then `scripts/make-release.sh`, then
   `scripts/validate-set.sh` on a fresh tarball.  Never hand-edit the committed patches or
   `release.json`.
5. **Run the release gates** in `wip/rdna3-verify-fusions/PROMOTION.md` section 4:
   - **gfx1201 (RDNA4):** same-seed coherence, `test-logits-width-probe`, `gate-prefill-logits.sh`.
     All must be byte-identical to r94; any diff means the RDNA3_0 scope leaked.
   - **gfx1100 (RDNA3_0), if a box is available:** `test-logits-width-probe` on 27B, 30B-A3B and
     35B-A3B at prefixes 256 and 3584 with `q8_0` KV, plus the same-binary kill-switch A/B
     (`GGML_CUDA_FUSE_Q8_1_VERIFY=0`, `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`,
     `GGML_CUDA_FUSE_CPY_BATCH=0`); greedy text `none` == `draft-mtp` at `n_max <= 7`; the four-axis
     MTP gate at `-n 3000` (R/P/C/K, reasoning pinned) and the phase-switching prompt; stock-relative
     `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`; `gate-prefill-logits.sh`;
     `test-backend-ops -b ROCm0 -o RMS_NORM,ADD,MUL_MAT,CPY,CONCAT,SSM_CONV`; and
     `check-gpu-faults.sh` on the logs.  The candidate-tree evidence is already in `README.md`; if no
     gfx1100 is reachable, say so in the WORKLOG instead of presenting the candidate evidence as a
     folded-tree result.
   - **gfx1151 (RDNA3_5), if a box is available:** build and coherence only; the change must be
     inactive.  Record the "not enabled" state, not a performance result.  If no box is available,
     record that the folded tree was not built on gfx1151.
   - **Multi-device RDNA3_0 (if a box is available):** GDN model with MTP under `-sm tensor` and
     `-sm layer`, and the width probe with `SPLIT=tensor`, for the batched-CPY half.  If no box is
     available, record the gap explicitly; do not claim multi-device validation from a single-GPU run.
6. **Update the docs in the same change.**  `ENVIRONMENT.md` §7.1 (RDNA4 and RDNA3_0, RDNA3_5
   excluded; no new variable), the block 09/10 notes in `patches/README.md`, and `GREEDY-PURITY.md`
   §11/§19 (the RDNA3_0 verify band now takes the same fusions; keep the gfx1100 hashes and the
   kill-switch identity).
7. **Close the WIP.**  Move `wip/rdna3-verify-fusions/` to `archive/work/rdna3-verify-fusions/`,
   remove its row from `wip/CAMPAIGNS.md` (no stub, per the WIP rule), and add a `WORKLOG.md` entry
   (newest first, `YYMMDD.VVV` header) with the release tag and the gate results.
8. **Release.**  Tag `v12-097cf33d9-rNN`, push the delivery, refresh the personal fork's
   `rdna-boosts` branch (strict apply at `release.json.base`, build, gates, `--force-with-lease`,
   personal fork only), and let the `radiance` worker watcher build and push the container images.
   See AGENTS.md "Pushing policy" and `CONTAINERS.md`.

## Do not

- Do not push to upstream `ggml-org/llama.cpp`.
- Do not apply `wip/` material to the fork or the delivery before the fold.
- Do not run the full container image build on the primary box.
- Do not claim gfx1101/gfx1102 or multi-device validation that was not run.
