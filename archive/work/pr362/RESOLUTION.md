# pr362 -- RESOLVED: promoted into block 05 as the padded GQA-6 tile fold (r376)

**Status: CLOSED 2026-11-28.  The artifact is not here; it is IN the delivery** (`patches/0005-*.patch`,
`ggml/src/ggml-cuda/fattn-tile-gqa6-fold.cuh` + one generated TU), **default ON** with
`GGML_HIP_FA_TILE_GQA6_FOLD=0` as the kill switch, gated by `scripts/gate-gqa6-fold.sh` (`MANIFESTS.md`
gate 50).  The files next to this note (`README.md`, `pr-submitted.patch`, `assessment-delta.patch`) are the
2026-11-25 REJECTION record and stay as written -- a dated record keeps the path that was correct when it was
written, so "wip/pr362/" inside them means this directory.

## What changed between the rejection and the promotion

The resubmission (2026-11-27) met R1-R6 by replacing the runtime branch with a **dedicated kernel TU**, so no
pre-existing instantiation moves at all.  Re-verified here before promotion:

* **R1 on the arch where the old form regressed**: gfx1201, `--kv q8_0 --ub 1`, the kill-switch arm reads mean KLD
  `-0.000000` / same-top-p `100.000 %` against an unpatched r375 build (the old form read 0.000391 / 98.97 %), and
  `tg64` is unmoved (27.95 / 26.24 vs 27.95 / 26.27).
* **The fold's own point, both arches**: the fold arm reads the same numbers as the kill-switch arm and as the
  unpatched build; on gfx1100 the raw u16 log-prob files are byte-identical between all three.
* **Value on the delivery's own RDNA3 part** (gfx1100, 27B IQ3_S, q8_0 K/V, tg64): +0.5 / +0.9 / +2.6 / +4.3 /
  +5.7 % at d2048/8192/32768/65536/98304 -- smaller than the submission's 2-GPU layer-split +9.9 % at 96 K, as a
  single-card arm should be.
* **R4/R5**: the prefill-KLD claim is gone (the decode-width point is quoted), the census exists, the three
  coverage additions are in (`test-backend-ops` 7581/7581 with the fold on, both arches).

## What was NOT closed and must not be re-derived as a win

* **multi-RDNA3 splits**: no such box on this fleet, so the fold is validated on a SINGLE RDNA3_0 device only.
* **a speculator verify batch** on a q8_0/GQA-6 target: our 27B carries no MTP head, so the submitter's
  "acceptance unchanged" arm (2-GPU layer split) is unverified here.  The fold DOES engage at `n_q <= 8`, i.e. at
  verify width, so this is the coverage gap to close on a box that can.
* **the kernel copy is a maintenance obligation**, not a one-off (`INVARIANTS.md` r376): re-sync it whenever
  `flash_attn_tile` / `flash_attn_tile_iter` change.

**Record.** `WORKLOG.md` 261128.004, `INVARIANTS.md` r376, `MANIFESTS.md` gate 50, `DIAGNOSTIC.md` 8,
`patches/README.md` block 05, `CONTRIBUTORS.md` (`@kapojko`, co-author).
