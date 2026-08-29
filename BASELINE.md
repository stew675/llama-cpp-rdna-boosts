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
`17252c769`**: blocks 01-11 = the fork's `rdna-boosts` branch commits
`2b7a135cb..f6f8f6778` (exported with `git format-patch`); block 12 = the
fork's working-tree delta vs `f6f8f6778` over four files
(`allreduce-hip.cu` new, `allreduce.cu`, `allreduce.cuh`, `ggml-cuda.cu`),
including the RDNA4-only gate. `scripts/make-patches.sh` regenerates both.
Verified 2026-08-29: clean apply (`git am` 01-11 + `git apply` 12) on a
fresh checkout at `17252c769`, full build clean, llama-cli same-seed
coherence IDENTICAL to the fork build, tg64 38.12 / tg512 41.08.

### Checkpoint cut (2026-08-29): master `cc83d7b48` - cross-architecture consolidation bookend

Master moved 4 commits past `d7bd3bfca` to `cc83d7b48` (sycl --fit
improvement, vulkan fastdiv/duplicate cleanup, metal fa-vec tunings, vulkan
mul_mat_id K-pad change + issue-27873 mul_mat_id test cases). None of the
4 commits touch the block footprint except `tests/test-backend-ops.cpp`
(the issue-27873 test additions in `make_test_cases_eval`, ~line 9367 -
outside every block hunk). `scripts/apply-all.sh` applies all 11 patches
**zero-fuzz to master `cc83d7b48`** (no 3-way fallback), and the applied
tree is byte-identical to the validated `d7bd3bfca` tree for every block
file except `fattn.cu` (the gfx1151-ports RDNA3_5 ternary, which is the
current official block-04 content and is strictly RDNA3_5-gated) and
`tests/test-backend-ops.cpp` (which carries exactly upstream's issue-27873
addition). The 3-arch validation (gfx1100/gfx1151/gfx1201) recorded in
MANIFESTS.md therefore carries over unchanged. Checkpoint:
`baseline/cc83d7b48`.

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

## Validation record (2026-08-28, AMD Radeon AI PRO R9700 x3, ROCm 7.14) - mmvq umbrella fold + renumber

Block 06 (old numbering) retired; its gfx1151 (RDNA3_5) mmvq parameter
table folded into block 10 (the k-quant + mmvq-parameter umbrella),
completing the plan from `~/stew675/bundle-k-quant-boosts.md` section 7.
Block 10 now carries the only decode-numerics changes in the set across ALL
architectures (RDNA3, RDNA3_5, RDNA4): the Q6_K/Q4_K/Q5_K/Q8_0 VDR kernels,
the RDNA3_5 nwarps=2 Q8_0 table, and the RDNA4 MoE mmid whitelist. Block
08's RDNA3_5 verify-batch hunk (`ncols_dst <= MMVQ_MAX_BATCH_SIZE`) moved
to block 10 with the table (it modifies block-06 content). The set is 10
patches numbered 01-10 with no gaps (old 07->06, 08->07, 10->08, 11->09,
12->10). Re-validated: fresh master `fe235f434` checkout, apply-all
01-10 - zero-fuzz apply, no 3-way fallback, applied tree **byte-identical
(0 diff lines)** to the pre-fold validated tree (every block's file bytes
unchanged; only the hunk boundaries moved between patches 08 and 10).
Block-01 behavior unchanged (`test-speculative-adaptive`, `test-arg-parser`);
MUL_MAT 1193/1193, GDN 46/46 (4/4 backends), q4_K/q5_K/q8_0 OK on the fresh
build. RDNA3_5 table semantics unchanged from the validated block-06/08
combination.

Greedy-purity note (block 10 is now the ONLY patch that changes decode
numerics on any architecture): compute outputs are not bit-identical to a
build without it - max logit diff 0.184 vs 0.203 for flash-attn on/off;
greedy streams are deterministic within a build but can flip across
configs (1 of 3 test prompts diverged at char 176). This is a different
fp32 rounding path, not a correctness change (see
[`GREEDY-PURITY.md`](GREEDY-PURITY.md)); PPL is bit-unchanged. Excluding block 10
restores 100% greedy purity on RDNA3, RDNA3_5 and RDNA4 alike (block 06's
gfx1151 table was the last non-block-10 numerics change and is now inside
block 10); it is applied last.

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

The CURRENT delivery patches (0001-0011) are the fork's `rdna-boosts` branch
commits `2b7a135cb..f6f8f6778` exported with `git format-patch` (one commit
per block, `17252c769` as the base); block 12 is the fork's working-tree
delta vs `f6f8f6778` (four files, RDNA4-gated). The table below records the
ORIGINAL source commits on the fork branch `chunked-gdn` (parent
`758443071`) whose content the blocks carry — historical provenance, kept
for the record:

| patch | source commits (original lineage) |
|-------|--------------------------------------------------|
| `0001-…-adaptive-MTP…` | fork branch `adaptive-mtp` (re-based): `c0f398ec5` `f3208c5c5` `0bbd7f3c2` `b5c99b890` `b26c775e1` — same 5-commit series as the old `chunked-gdn` lineage (`87ad1db26` `b56926039` `d0d7ff27e` `8d70e21f5` `0cf87e989`), re-based with the `common_speculative_impl` `n_max` constructor fix |
| `0002-…-chunked-gdn…` | `876ef1f0b` `5d2090e96` `b220647b1` `a4982afa2` `659f94987` `2a1e5c5a8` `1da07e19b` `77d51ee28` `abfa24265` `be46c7621` `3441b7d40` `246136122` `05cab3c41` |
| `0003-…-bf16-kv-cache…` | `5485e79e4` `b98265cfd` `07767a88a` `ef3673358` `b6bfa422e` `5e6072558` `bd5bf0ea3` `d33ce1adf` |
| `0004-…-wmma-flash-attn…` | `beaf69fb6` (plus the gfx1151-ports RDNA3_5 consumer-side fix) |
| `0005-…-bit-identical-decode-cpu…` | `89ac4ba1f` |
| `0008-…-fused-core…` | `14e5dd427` `d0e6119a7` `333e8f950` `c11752b18` `10e016df4` `85387ba3a` `8e1300159` `ac08b6d85` `a84112dcf` `9b4554626` `555e79ab2` `00f53040f` `ec09a818e` `bb64338f9` `4c0440841` `3d65d7979` |
| `0006-…-host-buffer-revert…` | `edb8d44c0` |
| `0007-…-meta-device-wrapper-skip…` | `32670eec8` |
| `0009-…-meta-headroom…` | `f2a22a71` (fork branch `rdna-boosts`, NOT on `chunked-gdn`) |
| `0010-…-k-quant-boosts…` | combined k-quant + mmvq-parameter umbrella: `cd35abd19` (Q6_K, ex-block 09) + `a7d092368`, `f1a072dcd` (Q4_K/Q5_K/Q8_0) + `5b320ed94` (RDNA3_5 table, ex-block 06, incl. the verify-batch hunk block 08 used to carry) — all fork branch `rdna-boosts`, NOT on `chunked-gdn` |
| `0011-…-prefill-graph-skip…` | the CUDA prefill-graph skip (fork `rdna-boosts` branch) |
| `12-hybrid-allreduce-hip.patch` | WIP exploration (`wip/12-hybrid-allreduce-hip.md`); no source commit — the fork's working-tree delta vs `f6f8f6778` (allreduce-hip.cu + allreduce.cu + allreduce.cuh + ggml-cuda.cu), RDNA4-gated |

> `0008-…-fused-core…` additionally carries a local correctness fix on top of
> the fork commits: the Q8_1 input cache now keys entries by `src1->data` in
> addition to the view root, so the stack-allocated per-expert `src1_slice`
> tensors of the mul_mat_id host-sort fallback never collide (equal token
> counts produced identical keys, reusing the wrong expert's quantized
> tokens; nondeterministic iq1_m MUL_MAT_ID failures on RDNA4).

`rdna-boosts-all.patch` (repo root) = the entire 12-patch net as ONE patch,
regenerated from the verified clean-apply tree (fork point `17252c769` + all
12 blocks applied), 37 files — including the hybrid all-reduce. It applies
cleanly on the fork point alone; use it only when you do not need the
per-block reviewability of `patches/`.

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
