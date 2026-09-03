# BASELINE - provenance and drift policy

Current state: `main` is the delivery branch carrying the **13-patch set**
(blocks 01-13) generated against the fork
point **llama.cpp master `9cffdcc80`** (re-based 2026-09-02 from
`0eadefebd`). The `baseline/<sha>` branches below
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


All 13 patches are generated against **llama.cpp upstream master at
`9cffdcc80`** (re-based 2026-09-02 from `0eadefebd`; dated records at the
bottom of this file): blocks 01-13 = the fork's `rdna-boosts` block
commits (the current 13-commit branch on `9cffdcc80` is
`04122bfb5..8f2838d1`; block 13 amended 2026-09-04 with two MTP
regression fixes — see `MANIFESTS.md` / `patches/README.md` block-13
notes; the previous `0eadefebd`-based
regeneration `b25bc8a9c..482837e5a` is superseded and preserved on the
fork remote's history).
`scripts/make-patches.sh` regenerates both. Verified 2026-09-02 and
re-verified 2026-09-04 after the block-13 amendment: clean
apply (`git am` 01-13) on a fresh checkout at
`9cffdcc80`, full build clean, llama-cli same-seed coherence IDENTICAL
(hybrid vs RCCL) — and the
apply is **whitespace-free** (zero git warnings).

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

The CURRENT delivery patches (0001-0013) are the fork's `rdna-boosts` block
commits exported with `git format-patch` (one commit per block, the
re-based set against `9cffdcc80`: `04122bfb5..8f2838d1`, block 13
amended 2026-09-04 with the two MTP regression fixes; previously the
re-based set against `0eadefebd`: `b25bc8a9c..a14257996`;
re-based regeneration `4c0f30dec..8fbf10e5b` against `a7cc83bba`; the
whitespace-clean regeneration `3209e83b4..cc985ba9a` against
`17252c769`; originally `2b7a135cb..f6f8f6778`; the original fork history
is preserved on `old-rdna-boosts`). Block 12 is the hybrid HIP all-reduce delta over four
files (RDNA4-gated). The ORIGINAL source commits on the fork branch
`chunked-gdn` (the pre-consolidation lineage) and the old `baseline/*`-branch
checkpoint history moved to `archive/docs/baseline-history.md`.

## Drift policy

The patches are static against the fork point `9cffdcc80`. If a patch fails
to apply against a newer upstream master:

1. Try `git am -3` / `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails (or the fork-state pre-image blobs are not in the local
   clone — the normal case for a fresh puller), rebase the failing hunks
   manually against the current
   master and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when more
   than one block needs manual re-base hunks, regenerate the whole set from
   the fork with `scripts/make-patches.sh` (re-exports blocks 01-13 from
   `9cffdcc80..<blocks-tip>`; defaults target
the current blocks tip `8f2838d1`), then re-verify the clean-apply
simulation (fresh worktree at the new fork point, `scripts/apply-all.sh`,
build, coherence) and update the fork point + verification numbers in
`patches/README.md` and `README.md`.

---

## Re-baseline to a7cc83bba (2026-08-30, dated record)

Upstream master moved 24 commits past the fork point `17252c769` (6 of
them touching ggml-cuda). The fork's `rdna-boosts` branch was rebuilt from
the delivery patches on the new base (blocks 01-07 and 09-12 applied
cleanly; block 08 needed a manual merge) and the set regenerated with
`scripts/make-patches.sh` (base `a7cc83bba`, blocks tip `8fbf10e5b`,
block 12 committed as `4fa92f0ae`).

The one real conflict: upstream's **SWIGLU_CLAMP (#27930)**, landed one
day after the old fork point, added `glu_limit` plumbing to the same
mm-fusion machinery block 08 rewrites (the `ggml_cuda_mm_fusion_args_*`
structs in `common.cuh`; four regions of `mmvq.cu` — the `active_glu`
decls, the fusion-assign block, the GLU-switch/result-write restructure,
and the `fusion_local` copy). Resolution: upstream's `glu_limit`/
`SWIGLU_CLAMP` additions were kept alongside block 08's fields, with the
SWIGLU_CLAMP case relocated inside block 08's restructured switch
(`result_val`). Verified by diffing the merged files against block 08's
post-image blobs: the difference is exactly upstream's additions, nothing
else.

Verified end-to-end 2026-08-30: clean-apply sim on a fresh clone at
`a7cc83bba` (`scripts/apply-all.sh`, zero conflicts + zero whitespace
warnings), full build clean, llama-cli same-seed coherence IDENTICAL to
the pre-re-base known-good build. tg64 38.12 / tg512 41.08 (sim build)
unchanged — the re-base is code-identical to the 2026-08-29 set plus
upstream's SWIGLU_CLAMP additions.

---

## Cross-version apply to 0eadefebd (2026-09-01, dated record)

Upstream master moved **22 commits** past the fork point `a7cc83bba`; only
**3 touched ggml-cuda** — `e4b9af007` (XOR-swizzle flash-attn K/V smem
fp16 tiles, #25635), `f8dbcd618` (ROCm radix TOP_K for long rows,
#27466), `41ef91f7c` (MOE fusion extended to specdec, #27621) — all in
block 08 / block 10 territory. Applied the 12-patch set to a fresh clone
checked out at `0eadefebd` (branch `rdna-boosts`):

- Blocks 01-07, 09-11: `git am` clean. Block 12: `git apply` clean.
- Block 08 (fused core): the ONE conflict — `git am -3` 3-way merge
after fetching the fork's blobs (the clone lacked the patch's index
blobs); **auto-resolved, zero manual hunks**. Verified per the
post-image-blob protocol: the two merged files (`ggml-cuda.cu`,
`mmvq.cu`) diff vs block 08's post-image blobs = exactly upstream's
additions (content-identical after stripping index/hunk headers).

**Full-tree zero-drift check:** `fork-tip → HEAD` differs from
`a7cc83bba → 0eadefebd` in exactly the same **51 files**, and all 51
diffs are content-identical — the applied tree is byte-faithful to the
fork delivery tip `4fa92f0ae` (blocks tip `8fbf10e5b` + block 12
`4fa92f0ae`) plus exactly the upstream drift.

**Verified end-to-end 2026-09-01:** full build clean (ROCm 7.14 gfx1201,
`GGML_HIP_RCCL=1`, graphs+native; zero patch-related compiler warnings);
llama-cli same-seed coherence **IDENTICAL between hybrid and RCCL**
(3-GPU tensor split, `GGML_CUDA_ALLREDUCE=nccl` comparison) — the
coherence gate passes on the new master.

**Fork point decision:** the delivery set **remains static against
`a7cc83bba`** — per the drift policy, regeneration / formal re-baseline
is triggered only when *more than one* block needs manual re-base hunks;
here only block 08 needed a 3-way merge and it auto-resolved with zero
drift. The verified applied state is preserved on the `rdna-boosts`
branch of the `~/prs/llama.cpp` clone (upstream `0eadefebd` + 12 blocks,
tag `rdna-boosts-0eadefebd`). If a future drift event ever needs the fork
point moved, follow the regeneration path above (`scripts/make-patches.sh`
with base `0eadefebd`, blocks tip `9c2463ff8`/`221b0c804` in that clone).

---

## Re-baseline to 0eadefebd (2026-09-01, dated record)

**Maintainer decision (same day): move the fork point to `0eadefebd`** —
`scripts/apply-all.sh` must apply cleanly against a fresh upstream
checkout, and with the old base's patch context it does not (block 08
fails with plain `git am`; only `git am -3` works). The record above's
"keep static" recommendation is superseded. Full re-baseline performed
per the drift policy step 3:

- **Fork rebuild:** `~/llama.cpp`'s `rdna-boosts` was deleted and
  rebuilt on `0eadefebd` (worktree; blocks 01-07 + 09-12 `git am`
  clean, block 08 `git am -3` auto-3way, block 12 `git apply` +
  commit). New commits: blocks `217e33ba4..d7bdd0a91`, block 12
  `ce9182473`. The rebuilt tree is **byte-identical** to the verified
  2026-09-01 cross-version apply above. The old `a7cc83bba`-based fork
  state (tip `4fa92f0ae`) is preserved on the `rdna-boosts-a7cc83bba`
  branch.
- **Set regenerated:** `scripts/make-patches.sh` (base `0eadefebd`,
  blocks tip `d7bdd0a91`) re-exported blocks 01-11 + the block-12
  delta; `rdna-boosts-all.patch` regenerated as
  `git diff 0eadefebd..ce9182473`. Folding upstream's changes into the
  patch context means the regenerated block 08 now applies with plain
  `git am` — **`apply-all.sh` is clean again on fresh master**.
- **Re-verified 2026-09-01:** clean-apply sim on a fresh clone at
  `0eadefebd` (`scripts/apply-all.sh`: **zero conflicts, zero
  whitespace warnings**; sim tree byte-identical to the fork tip), full
  build clean (ROCm 7.14 gfx1201, RCCL+graphs+native), llama-cli
  same-seed coherence IDENTICAL to the pre-re-base known-good build,
  tg64 38.12 / tg512 41.08 (numbers unchanged — code-identical
  content). Re-measured 2026-09-01 on the sim build (27B Q8_0,
  3-GPU tensor, r2): tg64 36.87 ± 4.83 / tg512 40.72 ± 1.02 — matches
  the documented numbers within noise (prs build: 37.92 ± 4.67 /
  40.90 ± 0.86).
- **Tooling fix:** `scripts/make-patches.sh`'s checkout check now
  accepts git worktrees (`[ ! -e "$FORK/.git" ]` instead of `-d`),
  which is how the fork rebuild is hosted.

## AR_PROFILE devices[] init fix + fork re-sync (2026-09-01)

- **Fix:** block 12's `allreduce-hip.cu` now fills `p->devices[]` from the
  caller list before the per-device profiler hipMallocs (PR #8).  With
  `GGML_CUDA_AR_PROFILE=1` the buffers were allocated while `devices[]`
  was still zero-filled, so all landed on GPU 0 and MTP's second pipeline
  (draft context) faulted/hung GPU 1 on gfx1201.  Pre-fix A/B reproduced
  the fault on 3x R9700 (2-GPU, internal AR, MTP n-max 3, `-c 32768`);
  post-fix runs clean with profiler teardown dumps on every device;
  coherence IDENTICAL to the pre-fix golden.  Integrated into the fork's
  block-12 commit and regenerated into
  `patches/0012-…-hybrid-HIP-all-reduce-RDNA4-gat.patch` + `rdna-boosts-all.patch`.
- **Fork re-sync:** the fork branch was rebuilt as a 12-commit branch
  directly on `0eadefebd` (block 01 `b25bc8a9c` .. block 11 `43f5ab71d`,
  block 12 `7d5d3f77b`, block 13 `a14257996`), dropping the upstream
  `kleidiai` docs commit
  `518b76236` that had crept into the previous rebuild (upstream-only;
  remains in `origin/master`).  The delivery contract is unchanged: 13
  blocks applied to a fresh checkout at `0eadefebd`.
- **Re-verified 2026-09-01:** clean-apply sim (`apply-all.sh` on a fresh
  clone at `0eadefebd`): zero whitespace warnings, applied tree
  byte-identical to the fork tip (`d42fc80…`).  Fork build clean (ROCm
  7.14 gfx1201, `cmake --build build-rocm --config Release -j 16 --
  VERBOSE=1`) + coherence + the A/B above.

## MTP chunked-GDN prefix folded into block 02 (2026-09-01, PR #9)

- **Change:** block 02 now also runs its chunked WMMA GDN on long
  single-sequence MTP prefills (`K > 1`): chunked on the prefix
  (`n_tokens - K`), sequential GDN only on the last K snapshot slots
  (PR #9; folded into the block-02 commit, NOT a new patch block).  The
  chunked ops take an `n_tokens_limit`; opt out `GGML_CUDA_GDN_CHUNKED=0`.
- **Verified 2026-09-01 (3x R9700, 2-GPU, internal AR, Qwen3.8-27B Q8,
  ubatch 1024, MTP n-max 3):** path fire `n=1024 K=4 prefix=1020`;
  prefill +7.5% (~5.5k) / +7.7% (~38k) vs sequential; 64-token
  same-seed output token-identical; non-MTP coherence unchanged.
- **Fork state:** block 02 amended (`cbc219af4`), blocks 03-12 replayed
  unchanged; blocks tip `43084332f`, block 12 `7d5d3f77b`.  Set
  regenerated (blocks 01, 03-11 content-identical; 0002 = old 0002 +
  PR #9 hunks); clean-apply sim re-verified (applied tree byte-identical
  to the fork tip `b90eb525e`).  Full clean build passes.


## Re-baseline to 9cffdcc80 (2026-09-02, dated record)

Upstream master moved **42 commits** past the fork point `0eadefebd`. This
re-base was performed as a **clean-room exercise** (simulating a first-time
puller of this repo): the fork was rebuilt in `~/llama.cpp` (a fresh
upstream clone at `9cffdcc80`) by applying the old 13-patch set with plain
`git am` — no fork-blob 3-way merges, no access to the previous fork
checkouts. Failed hunks were resolved **by hand** per the drift policy step
2. Three blocks needed manual re-base hunks (all in fattn-tile.cuh /
ggml-cuda.cu):

- **Block 03 vs upstream #27970 (sparse-fa, `8e93a9773`):** a 4th bool
  (`use_sparse`) was added to `launch_fattn` and the fattn-tile call sites
  updated; block 03 rewrites the same sites (type_KV threading + runtime
  `need_f16_K`/`need_f16_V`). Merged: `need_f16_K, need_f16_V, false,
  false, warp_size` per site (upstream's `stream_k`/`use_sparse` stay
  false); the `launch_fattn_tile_switch_ncols2` template gained `type_KV`.
- **Block 08 vs upstream #25952 (fused MoE expert reduction,
  `3466812d1`):** upstream's new `GGML_OP_MUL` weighted-reduction arm in
  `ggml_cuda_try_fuse` shifted block 08's rms_norm->mmvq quantize-fold arm
  context; the arm now sits after upstream's (order-independent — arms are
  mutually exclusive on `node->op`).  A second, latent issue was caught by
  the coherence gate, not the build: block 08's spec-verify `launch_fattn`
  call site in fattn-tile.cuh still passed the pre-#27970 3-bool arg list,
  binding the `warp_size` int into the new `use_sparse` bool slot
  (compiles silently; `use_sparse=true`) -> runtime
  `GGML_ASSERT(n_kv_max > 0)` in fattn-common.cuh. Fixed to the 4-bool
  form and folded into the block-08 commit.
- **Block 13 vs upstream #25952:** the `disable_moe_mmq` opt-out static +
  `const int cc` decls at the top of `ggml_cuda_try_fuse` (context shifted
  by upstream's inserted arm) restored after the MoE arm.

**Set regenerated:** `scripts/make-patches.sh` (base `9cffdcc80`, blocks
tip `92f09e80a`) re-exported blocks 01-13; `rdna-boosts-all.patch`
regenerated as `git diff 9cffdcc80..92f09e80a`. Folding upstream's
changes into the patch context means **`scripts/apply-all.sh` applies all
13 blocks with plain `git am` — zero conflicts, zero whitespace warnings**
on a fresh checkout at `9cffdcc80`.

**Re-verified end-to-end 2026-09-02:** clean-apply sim on a fresh worktree
at `9cffdcc80` (applied tree byte-identical to the fork tip `92f09e80a`),
full build clean (ROCm 7.14 gfx1201, `GGML_HIP_RCCL=1`, graphs+native,
zero errors — build note: `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="` is
required with CMake >= 4.3, whose HIP compiler test injects
`--cuda-host-only` directly after the build script's bare `-mllvm`),
llama-cli same-seed coherence **IDENTICAL between hybrid and RCCL**
(3-GPU tensor split, Qwen3.5-4B Q8_0). tg64 38.12 / tg512 41.08 unchanged
— the re-base is content-identical to the `0eadefebd` set plus upstream's
additions. The previous `0eadefebd`-based fork state (tip `482837e5a`)
remains on the `stew675/llama.cpp` fork remote (`rdna-boosts`); older
reference checkpoints are preserved in `~/prs/llama.cpp`.
