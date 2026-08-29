# BASELINE - provenance and drift policy

Current state: `main` is the delivery branch carrying the **12-patch set**
(blocks 01-11 + the hybrid all-reduce block 12) generated against the fork
point **llama.cpp master `17252c769`**. The `baseline/<sha>` branches below
are HISTORICAL checkpoints of the old pre-block-12 structure (patch
numbering 01-11 against older upstream ranges, `git apply` flow); they
remain as known-good records for those upstream versions.

> **Naming collision warning:** in the OLD records below, "block 12"
> sometimes means the old *k-quant umbrella* (folded into what is now block
> 10) and sometimes the *hybrid all-reduce* (the current block 12). In the
> current delivery, block 12 = the hybrid all-reduce, period.

Older branches (historical): `baseline/fe235f434` (the gfx12/gfx11
segregation baseline, validated on gfx1100, gfx1151, gfx1201),
`baseline/192067b72` (same patch files as `d222767c7`, zero-fuzz-validated
at `192067b72`), `baseline/d222767c7` (validated against `d222767c7`) and
`baseline/758443071` (the original set for the older upstream range).

## Baseline (current delivery)


All 12 patches are generated against **llama.cpp upstream master at
`17252c769`**: blocks 01-11 = the fork's `rdna-boosts` block commits
(originally `2b7a135cb..f6f8f6778`, preserved on `old-rdna-boosts`; the
current whitespace-clean regeneration is `3209e83b4..cc985ba9a`, tip
`12d10267b` with block 12 committed); block 12 = the hybrid HIP all-reduce
delta over four files (`allreduce-hip.cu` new, `allreduce.cu`,
`allreduce.cuh`, `ggml-cuda.cu`), including the RDNA4-only gate.
`scripts/make-patches.sh` regenerates both. Verified 2026-08-29: clean
apply (`git am` 01-11 + `git apply` 12) on a fresh checkout at
`17252c769`, full build clean, llama-cli same-seed coherence IDENTICAL to
the fork build, tg64 38.12 / tg512 41.08 — and the apply is
**whitespace-free** (zero git warnings) after the whitespace-clean
regeneration of 2026-08-29.

## Two fixes vs the fork

The patch set carries two fixes that are NOT on `chunked-gdn`; both come from
the fork's `rdna-boosts` branch (the production lineage) or from this
validation:

1. **Test-harness seeding (folded into block 02).** The fork's chunked-GDN
   work carried a deterministic seed into `init_tensor_uniform` in
   `tests/test-backend-ops.cpp` (added while debugging the bf16 GDN kernel):

   ```cpp
   // fork (chunked-gdn): static std::atomic<unsigned> g_seed(12345); (void) g_seed;
   // thread_local std::default_random_engine gen(12345 + (unsigned) start * 101);
   // this branch:         thread_local std::default_random_engine gen(std::random_device{}());
   ```

   The fixed seed correlates tensor data across rows and deterministically
   exposes a pre-existing numerical fragility in `rms_norm_back` and
   `cross_entropy_loss_back` on RDNA4 (CPU/GPU comparison fails with fixed
   seeds; passes with `random_device` seeding). GDN results are unaffected
   (46/46 in all configs either way). Full diagnostic: fixed seeds 12345 and
   54321 both fail those 9 cases; only the seeding line differs in the
   passing build.

2. **Meta-buffer compute-container headroom (block 09, `f2a22a71`).**
   `compute_headroom` 16x -> 128x in `ggml-backend-meta.cpp`. Hybrid
   recurrent models (GDN/SSM) create ~2*(n_rs_seq+1) conv-state snapshot
   views per recurrent layer during graph allocation, exceeding 16x and
   aborting with "not enough space in the context's memory pool"
   (ggml.c:1804). Without it, speculative MTP drafting under
   `--split-mode tensor` crashes on first decode. Source commit is on the
   fork branch `rdna-boosts`, not `chunked-gdn`; upstream has not fixed it
   either (reproduced on pristine `d222767c7`).

## Per-block provenance

The CURRENT delivery patches (0001-0011) are the fork's `rdna-boosts` block
commits exported with `git format-patch` (one commit per block,
`17252c769` as the base) — originally `2b7a135cb..f6f8f6778`, currently
the whitespace-clean regeneration `3209e83b4..cc985ba9a` (tip `12d10267b`
with block 12 committed; the original fork history is preserved on
`old-rdna-boosts`). Block 12 is the hybrid HIP all-reduce delta over four
files (RDNA4-gated). The ORIGINAL source commits on the fork branch
`chunked-gdn` (the pre-consolidation lineage) and the old `baseline/*`-branch
checkpoint history moved to `archive/docs/baseline-history.md`.

## Drift policy

The patches are static against the fork point `17252c769`. If a patch fails
to apply against a newer upstream master:

1. Try `git am -3` / `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails, rebase the failing hunks manually against the current
   master and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when more
   than one block needs manual re-base hunks, regenerate the whole set from
   the fork with `scripts/make-patches.sh` (which re-exports blocks 01-11
   from `17252c769..<blocks-tip>` and the block-12 delta; defaults target
the current clean blocks tip `cc985ba9a`), then re-verify the clean-apply
simulation (fresh worktree at the new fork point, `scripts/apply-all.sh`,
build, coherence) and update the fork point + verification numbers in
`patches/README.md` and `README.md`.

