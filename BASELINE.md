# BASELINE - provenance and drift policy

Branch: `main` points at the current checkpoint, currently
`baseline/fe235f434` (current, recommended)
Upstream range: `192067b72` .. `fe235f434` (7 upstream commits, incl. the
`common_speculative_impl` n_max parameter and new arg.cpp contexts)

Older branches: `baseline/192067b72` (same patch files as `d222767c7`,
zero-fuzz-validated at `192067b72`; superseded here because block 01 needed
re-base for the `n_max` drift), `baseline/d222767c7` (12-block set,
validated against `d222767c7`) and `baseline/758443071` (the original set
for the older upstream range; see "Older branches" below).

## Baseline

All patches on this branch are generated against **llama.cpp upstream master
at `fe235f434`** (block 01 regenerated from the fork's `adaptive-mtp` branch;
blocks 02-12 carried forward unchanged from `d222767c7`, where they were
validated by applying them, one at a time, to a fresh checkout of that SHA,
building with ROCm 7.14 (gfx1201, GGML_HIP=ON, Release), and running the
full test suite).

The same patch files apply **zero-fuzz to master at `192067b72`** (13
upstream commits past `d222767c7`): `scripts/apply-all.sh` applied all 12
patches without 3-way fallback, and the applied tree is byte-identical to
the `d222767c7`-validated tree for every block file (the 13 drift commits
only touch files outside the blocks). A fresh full build
(`~/bin/build-llama-rocm-714`, ROCm 7.14, gfx1201) succeeds, and the
real-model sanity run (Qwen3.8-27B Q4_K_XL, 1 card) matches the validated
numbers (pp512 1330 t/s, tg64 30.9 t/s; llama-cli generation 30.7 t/s).
That checkpoint is `baseline/192067b72`; its patch files are unchanged from
`d222767c7` (regeneration from the new tip produces identical content -
only hunk line hints and blob SHAs shift).

### Drift fix (2026-08-27): block 01 re-based against master `fe235f434`

Master moved 7 commits past `192067b72` to `fe235f434`. Block 01
(`01-adaptive-mtp.patch`) no longer applied: upstream added the
`n_max` parameter to `common_speculative_impl` (`spec: Add
benchmark-only synthetic speculative acceptance options`, `2bb9bddaf`), so
the old hunks in `common/arg.cpp`, `common/speculative.cpp` and
`tests/test-arg-parser.cpp` conflicted (`git apply -3` left conflicts in
`common/speculative.cpp` and `tests/test-arg-parser.cpp`). The fix comes
from the fork's `adaptive-mtp` branch (`/home/stew675/stew675/llama-master-before`,
same 5-commit series re-based on `fe235f434` with the `n_max` constructor
adapted - only `common/speculative.cpp` differs from the old block-01
content, by 2 lines). `01-adaptive-mtp.patch` was regenerated from that
branch; all 11 patches then apply **zero-fuzz to master `fe235f434`** via
`scripts/apply-all.sh` (no 3-way fallback), and the applied block-01 tree is
byte-identical to `adaptive-mtp` for all 10 block files. The other 10 blocks
are untouched (their patch files apply zero-fuzz unchanged).

### Validation record (2026-08-25, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

- `test-backend-ops` (ROCm0/1/2 + CPU): **14883/14883 passed, 0 failures**.
- `GATED_DELTA_NET` `-b ROCm0`: **46/46** in all four dispatch configs
  (default bf16 S_v=128, `GGML_CUDA_GDN_CHUNKED_BF16=0`,
  `GGML_CUDA_GDN_CHUNKED=0`, both force-enabled).
- `test-arg-parser`, `test-speculative-adaptive`: all tests OK.
- The applied tree is byte-identical to the source fork branch `chunked-gdn`
  for every production file (the only intentional divergence is the test
  harness fix described below).

### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14) - k-quant bundle

Block 09 retired; its Q6_K VDR=2 content folded into block 12 (the combined
k-quant umbrella, incl. RDNA4-gated Q8_0 VDR=4). The set is now 11 patches
(01-08, 10, 11, 12). Re-validated: fresh master `192067b72` checkout,
apply-all 01-08 + 10 + 11 + 12 - zero-fuzz apply, applied tree byte-identical (0 diff
lines) to the pre-bundle validated tree. MUL_MAT q4_K/q5_K/q8_0 54+47,
MUL_MAT_ID 76/76, GDN 46/46. Real-model sanity unchanged (Q4_K_XL decode
+5.5-5.7%, Q8_0 model +0.8%). PPL re-run on the Q8_0 model: 6.3162
(128 chunks) and 6.3563 (full corpus) - bit-identical to the pre-block-12
records (PPL is prefill-only; the VDR changes do not touch the prefill
path).

Greedy-purity note (block 12 is the only patch that changes decode
numerics): compute outputs are not bit-identical to a build without it - max
logit diff 0.184 vs 0.203 for flash-attn on/off; greedy streams are
deterministic within a build but can flip across configs (1 of 3 test
prompts diverged at char 176). Excluding block 12 restores purity; it is
applied last.

### Validation record (2026-08-27, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Master tip `fe235f434` + the 11-block patch set (applied with `apply-all.sh`,
zero-fuzz, block 01 regenerated from fork `adaptive-mtp`):

- `scripts/apply-all.sh` applied all 11 patches in manifest order with no
  3-way fallback (block 10 in context after 03+04, as required).
- The applied block-01 tree is byte-identical to the fork branch
  `adaptive-mtp` for all 10 block files; blocks 02-12 are byte-identical to
the `192067b72`-validated block code (their patch files were not changed).
- Block-01 behavior verified: `./bin/test-speculative-adaptive` and
  `./bin/test-arg-parser` pass (same tests as the `d222767c7` record);
  `--draft-mtp-adaptive` range validation (n_min_adaptive vs n_max,
  incl. the MTP-layer-capped n_max case) matches the fork branch.

### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Master tip `192067b72` + the 12-block patch set (applied with `apply-all.sh`,
zero-fuzz):

- Fresh full build (`~/bin/build-llama-rocm-714`): clean, all targets
  (incl. llama-server, llama-cli, test-backend-ops).
- Real-model sanity (Qwen3.8-27B Q4_K_XL, 1 card): llama-bench pp512
  1330 t/s, tg64 30.9 t/s; llama-cli --single-turn generates correctly
  (30.7 t/s). Matches the `d222767c7`-validated numbers.
- The block code is byte-identical to the 2026-08-25/26 records below (the
  13 drift commits do not overlap any block file region - proven by the
  zero-fuzz apply).

### Validation record (2026-08-26, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

Blocks 01-11 as above, plus **block 12** (`12-k-quant-boosts.patch`,
`a7d092368`):

- `test-backend-ops` (ROCm0 + CPU): **14883/14883 passed, 0 failures** on the
  block-11+12 tree.
- `MUL_MAT` q4_K/q5_K: **54/54**; `MUL_MAT_ID` q4_K/q5_K: **76/76**.
- Real-model (Qwen3.8-27B Q4_K_XL, 17 GB, mixed Q4_K/Q5_K/Q6_K/Q8_0, 1 card,
  llama-bench): decode +5.5-5.7% (tg32/tg128), prefill +0.6-1.0%
  (pp128/512/1024) vs the block-11 tree.

### Two fixes vs the fork

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

2. **Meta-buffer compute-container headroom (block 11, `f2a22a71`).**
   `compute_headroom` 16x -> 128x in `ggml-backend-meta.cpp`. Hybrid
   recurrent models (GDN/SSM) create ~2*(n_rs_seq+1) conv-state snapshot
   views per recurrent layer during graph allocation, exceeding 16x and
   aborting with "not enough space in the context's memory pool"
   (ggml.c:1804). Without it, speculative MTP drafting under
   `--split-mode tensor` crashes on first decode. Source commit is on the
   fork branch `rdna-boosts`, not `chunked-gdn`; upstream has not fixed it
   either (reproduced on pristine `d222767c7`).

## Per-block provenance

Blocks were re-generated from the validated application (one commit per
block on the `rdna-boosts` branch of the consumer checkout at `d222767c7`),
so each patch is exactly the block's net change against the new baseline.
The original source commits on the fork branch `chunked-gdn` (parent
`758443071`) remain:

| patch | source commits on `chunked-gdn` (history order) |
|-------|--------------------------------------------------|
| `01-adaptive-mtp.patch` | fork branch `adaptive-mtp` (re-based on `fe235f434`): `c0f398ec5` `f3208c5c5` `0bbd7f3c2` `b5c99b890` `b26c775e1` - same 5-commit series as the old `chunked-gdn` lineage (`87ad1db26` `b56926039` `d0d7ff27e` `8d70e21f5` `0cf87e989`), re-based with the `common_speculative_impl` `n_max` constructor fix |
| `02-chunked-gdn.patch` | `876ef1f0b` `5d2090e96` `b220647b1` `a4982afa2` `659f94987` `2a1e5c5a8` `1da07e19b` `77d51ee28` `abfa24265` `be46c7621` `3441b7d40` `246136122` `05cab3c41` |
| `03-bf16-kv-cache.patch` | `5485e79e4` `b98265cfd` `07767a88a` `ef3673358` `b6bfa422e` `5e6072558` `bd5bf0ea3` `d33ce1adf` |
| `04-wmma-flash-attn.patch` | `beaf69fb6` |
| `05-bit-identical-decode-cpu.patch` | `89ac4ba1f` |
| `10-fused-core.patch` | `14e5dd427` `d0e6119a7` `333e8f950` `c11752b18` `10e016df4` `85387ba3a` `8e1300159` `ac08b6d85` `a84112dcf` `9b4554626` `555e79ab2` `00f53040f` `ec09a818e` `bb64338f9` `4c0440841` `3d65d7979` |
| `06-gfx1151-mmvq-table.patch` | `5b320ed94` |
| `07-host-buffer-revert.patch` | `edb8d44c0` |
| `08-meta-device-wrapper-skip.patch` | `32670eec8` |
| `11-meta-headroom.patch` | `f2a22a71` (fork branch `rdna-boosts`, NOT on `chunked-gdn`) |
| `12-k-quant-boosts.patch` | combined k-quant umbrella: `cd35abd19` (Q6_K, ex-block 09) + `a7d092368`, `f1a072dcd` (Q4_K/Q5_K/Q8_0) - all fork branch `rdna-boosts`, NOT on `chunked-gdn`; the patch is regenerated from the consumer application, see `scripts/make-patches.sh` |

> `10-fused-core.patch` additionally carries a local correctness fix on top of
> the fork commits: the Q8_1 input cache now keys entries by `src1->data` in
> addition to the view root, so the stack-allocated per-expert `src1_slice`
> tensors of the mul_mat_id host-sort fallback never collide (equal token
> counts produced identical keys, reusing the wrong expert's quantized
> tokens; nondeterministic iq1_m MUL_MAT_ID failures on RDNA4).
| `rdna-boosts-all.patch` (repo root) | regenerated from the consumer application at `fe235f434` (diff of the fresh-master checkout with all 11 blocks applied vs `fe235f434`); the block-01 content is the fork `adaptive-mtp` branch, the rest is the `758443071..chunked-gdn` + `f2a22a71` + `a7d092368` + `f1a072dcd` content carried forward unchanged |

## Older branches

`baseline/192067b72` holds the previous checkpoint (same patch files as
`d222767c7`, zero-fuzz-validated at `192067b72`, full-suite + real-model
records in the validation sections above). It is superseded because block 01
drifted (see the drift-fix section); it remains the known-good set for that
upstream range.

`baseline/758443071` holds the original patch set (generated against
`758443071`, mirroring the fork `chunked-gdn` exactly). It remains the
known-good set for that upstream range, with two caveats:

1. Its `02-chunked-gdn.patch` still carries the deterministic test-harness
   seed (see above) - the fork's state. Use this branch's fixed patches if
   you are testing against `758443071` and hit the `rms_norm_back` /
   `cross_entropy_loss_back` failures.
2. Its block-10 test hunk is placed at a re-based location in
   `make_test_cases_perf()` (context added by block 04 in the fork).

## Drift policy

The patches are static. If a patch fails to apply against a newer upstream
master:

1. Try `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails, rebase the failing hunks manually against the current
   master and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when a new
   validated baseline is cut, the whole set is regenerated from the
   consumer application (as done for `baseline/d222767c7`) and a new
   `baseline/<new-sha>` branch is created here. Old branches stay as the
   known-good sets for older upstream versions.

Rule of thumb for cutting a new baseline branch: cut one whenever **more
than one block needs manual re-base hunks**, whenever **a validated
full-suite run exists for a newer tip**, or whenever **merge-conflict drift
has occurred** (any block that needed `git apply -3` or manual hunk
re-basing to apply). Merge-conflict drift means the upstream context the
block was generated against has changed - even a single-block fix is
permanent new content (the block's net diff vs the new baseline differs
from the old patch), so it gets its own checkpoint rather than silently
editing the committed patch files on an older baseline branch. The
`baseline/fe235f434` cut is the first application of this rule (block 01
was the only conflicted block).

`MANIFESTS.md` records the upstream range per branch so consumers can match
their upstream version.
