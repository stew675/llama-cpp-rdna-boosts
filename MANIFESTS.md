# MANIFESTS - apply order and verification contract

Squashed, standalone diff blocks of RDNA-specific performance and correctness
work from the `chunked-gdn` branch of the
[llama.cpp fork](https://github.com/stew675/llama.cpp), packaged for easy
application to mainline llama.cpp.

The branch is 48 commits of RDNA work on top of upstream master
(`758443071`). Those commits decompose into **10 functional blocks**: 9 clean
standalone diffs and 1 "fused core" (16 mutually-entangled commits extracted as
one combined diff, applied last). One follow-up fix was added later as
**block 11** (`11-meta-headroom.patch`, meta-buffer compute-container
headroom) and the k-quant kernel boosts landed as **block 12**
(`12-k-quant-boosts.patch`, Q6_K VDR=2 + Q4_K/Q5_K/Q8_0 VDR=4 decode, mmq
scale-load hoist, RDNA4 MoE mmid whitelist), so the full set is 11 patches
in apply order. **Block 09 was retired**: its Q6_K VDR=2 decode work was
folded into block 12 so that all k-quant VDR/decode work lives in the one
umbrella patch - excluding it restores 100% greedy-purity (see the
Greedy-purity note below). Numbers run 01-08, 10, 11, 12.

This is the authoritative apply order and the verification contract for the
patch set. It is written for humans AND LLM coding agents. Follow it exactly;
do not skip blocks.

Branch: `main` points at the current checkpoint, currently
`baseline/fe235f434` (current, recommended)
Upstream range: `192067b72` .. `fe235f434` (7 commits; block 01 was
re-based for the `common_speculative_impl` n_max drift, blocks 02-12 are
unchanged - see BASELINE.md. Older branches: `baseline/192067b72`,
`baseline/d222767c7`, `baseline/758443071`)

## Validation record (2026-08-27, ROCm 7.14, gfx1201) - drift fix

All 11 patches applied zero-fuzz to master at `fe235f434`
(`scripts/apply-all.sh`, no 3-way fallback; block 01 regenerated from the
fork's `adaptive-mtp` branch). Applied block-01 tree byte-identical to
`adaptive-mtp`; blocks 02-12 byte-identical to the `192067b72` record.
Block-01 behavior: `test-speculative-adaptive` and `test-arg-parser` pass.

## Validation record (2026-08-26, ROCm 7.14, gfx1201) - new checkpoint

Same patch files applied zero-fuzz to master at `192067b72`
(`scripts/apply-all.sh`, no 3-way fallback); applied tree byte-identical to
the `d222767c7` record for every block file. Fresh full build clean;
real-model sanity (Qwen3.8-27B Q4_K_XL, 1 card): pp512 1330 t/s, tg64
30.9 t/s, llama-cli generation 30.7 t/s - matches the validated numbers.
Full 14883/14883 suite record carries from the identical block code.

## Validation record (2026-08-25, ROCm 7.14, gfx1201)

All 11 patches applied in the order below to a fresh checkout of
`d222767c7`, built `GGML_HIP=ON Release`:

- `test-backend-ops` (ROCm0/1/2 + CPU): **14883/14883 passed, 0 failures**
- `GATED_DELTA_NET` `-b ROCm0`: **46/46** in all four dispatch configs
- `test-arg-parser`, `test-speculative-adaptive`: all tests OK

**One fix vs the fork:** block 02 restored upstream `random_device` seeding
in `init_tensor_uniform` (the fork's deterministic seed, added while
debugging the bf16 GDN kernel, deterministically fails `rms_norm_back` /
`cross_entropy_loss_back` on RDNA4). GDN results are unaffected. See
`BASELINE.md` for the full diagnostic.

## Apply order

| # | patch | files | deps |
|---|-------|-------|------|
| 01 | `01-adaptive-mtp.patch` | common/arg.cpp, common/common.{h,cpp}, common/speculative.cpp, common/speculative-adaptive.h, tools/server/server-context.cpp, src/models/delta-net-base.cpp, tests/test-speculative-adaptive.cpp, tests/test-arg-parser.cpp, tests/CMakeLists.txt | none |
| 02 | `02-chunked-gdn.patch` | ggml/src/ggml-cuda/gated_delta_net.cu, gated_delta_net_chunked.{cu,cuh}, gated_delta_net_chunked_bf16.cu, tests/test-backend-ops.cpp | none |
| 03 | `03-bf16-kv-cache.patch` | fattn-tile.{cu,cuh}, fattn.cu, common.cuh, rope.cu, ggml-cuda.cu (IMRoPE fuse), tests/test-backend-ops.cpp | none |
| 04 | `04-wmma-flash-attn.patch` | fattn-mma-f16.cuh, mmq-vec-dot.cuh, mmq.cuh, fattn.cu, ggml-cuda.cu (WMMA dispatch), tests/test-backend-ops.cpp | none |
| 05 | `05-bit-identical-decode-cpu.patch` | ggml/src/ggml-cpu/llamafile/sgemm.cpp | none |
| 06 | `06-gfx1151-mmvq-table.patch` | ggml/src/ggml-cuda/mmvq.cu | none |
| 07 | `07-host-buffer-revert.patch` | ggml/src/ggml-cuda/ggml-cuda.cu | none |
| 08 | `08-meta-device-wrapper-skip.patch` | src/llama.cpp | none |
| 10 | `10-fused-core.patch` | mmvq.{cu,cuh}, ggml-cuda.cu (try_fuse), norm.{cu,cuh}, unary.{cu,cuh}, common.cuh, fattn.cu, fattn-tile.cuh | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory) |
| 11 | `11-meta-headroom.patch` | ggml/src/ggml-backend-meta.cpp | none (independent; apply last) |
| 12 | `12-k-quant-boosts.patch` | ggml/src/ggml-cuda/{ggml-cuda.cu,mmq-vec-dot.cuh,mmvq.cu,vecdotq.cuh}, tests/test-backend-ops.cpp | none (apply last; omit for greedy purity) |

Block numbers are the apply order: `01` is the smallest number and applies
first, `11` last. All blocks are mutually independent except **block 10
(fused core) requires blocks 03 and 04 in the tree** - so it is applied at
position 10, just before block 11.

## Block stacking tags (git-native path)

The repo also carries lightweight tags `block/01-… block/10-…`, each pointing
at a squashed commit whose diff-vs-parent is exactly that block's change (the
commits are stacked in apply order, rooted at an orphan commit whose tree is
the upstream baseline `fe235f434`). The tags track the CURRENT baseline
branch; they are force-moved whenever a new validated baseline is cut.

**Tags are numbered by APPLY order** and so are the patch filenames and the
block commit labels: `01` applies first, `11` last. The fused core needs
blocks 03+04 in the tree, so it sits at position 10:

| apply | tag | patch file |
|-------|-----|-----------|
| 1 | `block/01-adaptive-mtp` | `01-adaptive-mtp.patch` |
| 2 | `block/02-chunked-gdn` | `02-chunked-gdn.patch` |
| 3 | `block/03-bf16-kv-cache` | `03-bf16-kv-cache.patch` |
| 4 | `block/04-wmma-flash-attn` | `04-wmma-flash-attn.patch` |
| 5 | `block/05-bit-identical-decode-cpu` | `05-bit-identical-decode-cpu.patch` |
| 6 | `block/06-gfx1151-mmvq-table` | `06-gfx1151-mmvq-table.patch` |
| 7 | `block/07-host-buffer-revert` | `07-host-buffer-revert.patch` |
| 8 | `block/08-meta-device-wrapper-skip` | `08-meta-device-wrapper-skip.patch` |
| 10 | `block/10-fused-core` | `10-fused-core.patch` |
| 11 | `block/11-meta-headroom` | `11-meta-headroom.patch` |
| 12 | `block/12-k-quant-boosts` | `12-k-quant-boosts.patch` |

`block/11-meta-headroom` fixes the meta-buffer compute-container headroom
(16x -> 128x) for hybrid recurrent models: the GDN/SSM conv-state snapshot
views create ~2*(n_rs_seq+1) views per recurrent layer, which previously
aborted graph allocation with "not enough space in the context's memory
pool" (ggml.c:1804) - exactly what `--split-mode tensor` + MTP draft hit
in the server. Source: fork branch `rdna-boosts` commit `f2a22a71` (not on
`chunked-gdn`); needed by any recurrent model under tensor split, applied
last or anywhere (independent file).

`block/12-k-quant-boosts` is the k-quant umbrella: all k-quant VDR/decode
work in one patch. It carries the Q6_K VDR=2 decode kernel (ex-block 09,
plus the `GGML_CUDA_OP_TIMING` graph-capture guard), the Q4_K/Q5_K mmvq
VDR=4 kernels (32 elements/thread, shared scale/d8 loads), the Q8_0 mmvq
VDR=4 kernel (RDNA4-gated, `__GFX12__`), the mmq prefill scale-load hoist
in `vec_dot_q8_1_q8_1_mma`, the RDNA4 MoE mmid whitelist Q4_K 4->7, and the
matching perf-harness cases. Measured on gfx1201: decode n=1 -13%..-18%
(q4_K), -8%..-15% (q5_K), -10%..-33% (q8_0 compute-bound shapes); verify
batch n=2..8 -2%..-13%; MoE n=5/6/7 per expert -22%..-28%; real-model
(Qwen3.8-27B Q4_K_XL, mixed quants) decode +5.5-5.7%. Future k-quant
optimizations land in this block. Combined patch regenerated from the
consumer application (see BASELINE.md); source commits on fork branch
`rdna-boosts`: `cd35abd19` (Q6_K), `a7d092368` + `f1a072dcd` (Q4_K/Q5_K +
Q8_0).

> **Greedy-purity note:** this is the ONLY patch in the set that changes
> decode numerics (VDR reorders the fp32 cross-thread reduction). Compute
> outputs are not bit-identical to a build without it: max logit diff 0.184
> vs 0.203 for flash-attn on/off; greedy streams are deterministic within a
> build but can flip across configs (1 of 3 test prompts diverged). PPL is
> unaffected (prefill path untouched): 6.3162/6.3563 on wikitext-2, matching
> the pre-block-12 records. Excluding this patch restores 100% greedy
> purity; it is applied last, so `apply-all.sh` needs only the one ORDER
> entry dropped.
Consumer (git-native alternative to `git apply`):

```
git remote add rdna-boosts git@github.com:stew675/llama-cpp-rdna-boosts.git
git fetch rdna-boosts --tags
# on the llama.cpp checkout at the baseline, in apply order:
git cherry-pick block/01-adaptive-mtp
...
git cherry-pick block/10-fused-core     # last
...
git cherry-pick block/11-meta-headroom
git cherry-pick block/12-k-quant-boosts   # omit for greedy purity
```

Cherry-pick uses 3-way merge, so each block degrades gracefully when upstream
master drifts past the recorded baseline. The tags are derived artifacts -
`scripts/make-patches.sh` rebuilds the whole lineage and force-moves the tags
deterministically (identical trees, messages and SHAs across runs). The patch
files remain the primary, reviewable artifact.

## Verified apply sequence (this branch)

On a fresh checkout of `fe235f434`, the sequence

```
git apply patches/01-adaptive-mtp.patch
git apply patches/02-chunked-gdn.patch
git apply patches/03-bf16-kv-cache.patch
git apply patches/04-wmma-flash-attn.patch
git apply patches/05-bit-identical-decode-cpu.patch
git apply patches/06-gfx1151-mmvq-table.patch
git apply patches/07-host-buffer-revert.patch
git apply patches/08-meta-device-wrapper-skip.patch
git apply patches/10-fused-core.patch
git apply patches/11-meta-headroom.patch
git apply patches/12-k-quant-boosts.patch   # omit for greedy purity
```

applies with zero fuzz and, after the block-02 test-harness fix, passes the
full backend test suite (14883/14883). The patch set was validated exactly
this way; see BASELINE.md for the byte-identity and validation details.

## Verification per block

| block | verify command | expected |
|-------|----------------|----------|
| 01 | `./bin/test-speculative-adaptive && ./bin/test-arg-parser`; llama-server `--draft-mtp-adaptive` smoke | pass |
| 02 | `./bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET` | 46/46 on all four dispatch configs (default bf16, `GGML_CUDA_GDN_CHUNKED_BF16=0`, `GGML_CUDA_GDN_CHUNKED=0`, +/- graphs) |
| 03 | `./bin/test-backend-ops -b ROCm0 -o FA_ATTN_*` (BF16 KV cases) + bf16-KV model run | pass |
| 04 | `./bin/test-backend-ops -b ROCm0` (attention correctness) + decode/prefill perf on gfx1201 | pass / perf |
| 05 | speculative-decoding determinism test with the CPU backend | identical decode vs verify batches |
| 07 | build + Q8_0 decode on gfx1151 | pass / perf |
| 08 | build + decode perf on integrated-GPU HIP target | pass / perf |
| 09 | `llama-server --list-devices` with one GPU | no Meta wrapper |
| 10 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT Q6_K cases) + Q6_K decode on gfx1201 | 1194/1194 MUL_MAT OK |
| 11 | build + server `--split-mode tensor` + MTP draft smoke | loads/serves, no graph-alloc abort |
| 12 | `./bin/test-backend-ops -b ROCm0` (MUL_MAT + MUL_MAT_ID q4_K/q5_K cases) + Q4_K/Q5_K decode on gfx1201 | 54/54 MUL_MAT, 76/76 MUL_MAT_ID OK |
| 06 | full `./bin/test-backend-ops`; speculative-decode bit-identity check; SSM/MoE model smoke | pass |

Convenience: `rdna-boosts-all.patch` (repo root) is the entire net diff in
one patch (applies cleanly on the baseline SHA alone; not a substitute for the
per-block flow in `patches/` when you want reviewable increments).

## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the latest llama.cpp master tip. After each patch, run its
> verification command. If a patch fails to apply, use `git apply -3`; if
> 3-way fails, manually rebase the hunks against the current master and
> continue. Do not skip blocks. The fused core (`06-...`) goes last.

## Known notes

- **iq1_m MUL_MAT_ID flake - FIXED (block 10 / `10-fused-core.patch`).** The
  nondeterministic ~75%-per-run failure of
  `MUL_MAT_ID(type_a=iq1_m,...,m=64,n=16,k=768)` (ERR ~0.4 vs 5e-4 tolerance,
  b=0 variant only) was a real bug in the fused-core Q8_1 input cache, not a
  numerical quirk: the cache keyed matmuls by the view root of `src1` only,
  but the mul_mat_id host-sort fallback reuses one stack-allocated
  `src1_slice` tensor for every expert. Experts with equal token counts
  produced identical keys, so the second expert reused the first one's
  quantized tokens. Fix: add the `src1->data` pointer to the cache key
  (same-tensor reuse across qkv/alpha/beta projections is preserved).
  Verified: 20/20 clean single-case runs, 15/15 full MUL_MAT_ID groups,
  full suite 14883/14883, GDN 46/46 in all dispatch configs.
- **MTP draft + `--split-mode tensor` crash - FIXED by block 11.** The
  `ggml.c:1804` graph-allocation abort seen with speculative MTP drafting
  under tensor split was the meta-buffer compute-container headroom (16x)
  being exceeded by hybrid-recurrent (GDN/SSM) conv-state snapshot views
  (~2*(n_rs_seq+1) per recurrent layer). Block 11 raises it to 128x,
  verified: full server config (adaptive MTP + ngram, tensor split, 262K
  ctx, BF16 KV, mmproj) loads, serves, and survives requests. Without block
  11, pristine upstream `d222767c7` still crashes (upstream has not fixed
  it); the fix originates from the fork's `rdna-boosts` branch (`f2a22a71`),
  which is not part of `chunked-gdn`. Timeline: adaptive MTP is new (created
  ~1 week before this crash report) - the crash surfaced on fix-less builds
  (chunked-gdn lineage) only; the prior fixed-depth MTP config ran for ~3
  months on the fork's `rdna-boosts` branch, which has always carried
  `f2a22a71`.
- **Block 02 carries the test-harness seeding fix** (restores upstream
  `random_device` seeding in `init_tensor_uniform`; the fork's deterministic
  seed exposed pre-existing `rms_norm_back` / `cross_entropy_loss_back`
  numerical fragility on RDNA4). The GDN kernels are untouched. Full
  diagnostic in BASELINE.md.
- **Block 10 test hunk** was re-based against the original baseline: its
  original context lines (Q6_K perf cases) were added by block 04. On this
  branch the hunk applies with the re-based placement; content is identical
  to the fork, only position differs.
- **Block 12 test hunks** anchor on block-09 content (the Q6_K/Q8_0
  decode-shape rows and the `qwen3-30b-a3b` loops). Applied in manifest order
  they sit on the re-based block-09 hunk; do not re-order the test hunks when
  re-basing.
- **`test-backend-ops.cpp` is shared** by blocks 02/03/04/09/10/12. The hunks
  are in different case regions; if upstream adds cases in those regions,
  re-base the affected hunks (each patch applies independently on the
  baseline, so re-basing is local to the failing file).
- **Block 06 is genuinely inseparable**: its 16 commits co-developed the fused
  mmvq kernel region, the `ggml-cuda.cu` try_fuse machinery, and the `fattn.cu`
  dispatch cluster. It is extracted as ONE combined diff on purpose. Do not try
  to split it.
- When upstream master moves past the recorded range and more than one block
  needs manual re-base hunks, use `scripts/make-patches.sh` from the fork to
  regenerate the set against the new tip and cut a new `baseline/<sha>`
  branch here instead of patching this branch's files by hand.
