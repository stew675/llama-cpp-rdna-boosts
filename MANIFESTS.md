# MANIFESTS - apply order and verification contract

This is the authoritative apply order and the verification contract for the
patch set. It is written for humans AND LLM coding agents. Follow it exactly;
do not skip blocks.

Branch: `baseline/758443071`
Upstream range: `7584430716ee229751771ed0d6bbcb780d105eeb` (single point)

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
| 09 | `09-q6k-mmvq-vdr2.patch` | ggml/src/ggml-cuda/{ggml-cuda.cu,mmvq.cu,vecdotq.cuh}, tests/test-backend-ops.cpp | none (test hunk re-based to stock) |
| 10 | `10-fused-core.patch` | mmvq.{cu,cuh}, ggml-cuda.cu (try_fuse), norm.{cu,cuh}, unary.{cu,cuh}, common.cuh, fattn.cu, fattn-tile.cuh | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory) |
| 11 | `11-meta-headroom.patch` | ggml/src/ggml-backend-meta.cpp | none (independent; apply last) |

Block numbers are the apply order: `01` is the smallest number and applies
first, `11` last. All blocks are mutually independent except **block 10
(fused core) requires blocks 03 and 04 in the tree** - so it is applied at
position 10, just before block 11.

## Block stacking tags (git-native path)

The repo also carries lightweight tags `block/01-… block/10-…`, each pointing
at a squashed commit whose diff-vs-parent is exactly that block's change (the
commits are stacked in apply order, rooted at an orphan commit whose tree is
the upstream baseline `758443071`).

**Tags are numbered by APPLY order.** The patch filenames keep the plan's
apply order: the fused core is `block/10-fused-core`, applied last
encodes the sequence - so the fused core, applied last, is `block/10-fused-core`:

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
| 9 | `block/09-q6k-mmvq-vdr2` | `09-q6k-mmvq-vdr2.patch` |
| 10 | `block/10-fused-core` | `10-fused-core.patch` |
| 11 | `block/11-meta-headroom` | `11-meta-headroom.patch` |

Consumer (git-native alternative to `git apply`):

```
git remote add rdna-boosts git@github.com:stew675/llama-cpp-rdna-boosts.git
git fetch rdna-boosts --tags
# on the llama.cpp checkout at the baseline, in apply order:
git cherry-pick block/01-adaptive-mtp
...
git cherry-pick block/10-fused-core     # last
```

Cherry-pick uses 3-way merge, so each block degrades gracefully when upstream
master drifts past the recorded baseline. The tags are derived artifacts -
`scripts/make-patches.sh` rebuilds the whole lineage and force-moves the tags
deterministically (identical trees, messages and SHAs across runs). The patch
files remain the primary, reviewable artifact.

## Verified apply sequence (this branch)

On a fresh checkout of the baseline SHA, the sequence

```
git apply patches/01-adaptive-mtp.patch
git apply patches/02-chunked-gdn.patch
git apply patches/03-bf16-kv-cache.patch
git apply patches/04-wmma-flash-attn.patch
git apply patches/05-bit-identical-decode-cpu.patch
git apply patches/06-gfx1151-mmvq-table.patch
git apply patches/07-host-buffer-revert.patch
git apply patches/08-meta-device-wrapper-skip.patch
git apply patches/09-q6k-mmvq-vdr2.patch
git apply patches/10-fused-core.patch
git apply patches/11-meta-headroom.patch
```

applies with zero fuzz and produces a tree byte-identical to the source fork
branch `chunked-gdn` for every production file.

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
| 06 | full `./bin/test-backend-ops`; speculative-decode bit-identity check; SSM/MoE model smoke | pass |

Convenience: `rdna-boosts-all.patch` (repo root) is the entire 48-commit net diff in
one patch (applies cleanly on the baseline SHA alone; not a substitute for the
per-block flow when you want reviewable increments).

## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the latest llama.cpp master tip. After each patch, run its
> verification command. If a patch fails to apply, use `git apply -3`; if
> 3-way fails, manually rebase the hunks against the current master and
> continue. Do not skip blocks. The fused core (`06-...`) goes last.

## Known notes

- **iq1_m MUL_MAT_ID flake - FIXED (block 10 / `10-fused-core.patch`).** The
  nondeterministic failures of `MUL_MAT_ID(type_a=iq1_m,...,m=64,n=16,k=768)`
  (ERR ~0.4 vs 5e-4 tolerance) on this patch set were a real bug in the
  fused-core Q8_1 input cache, not a numerical quirk: the cache keyed matmuls
  by the view root of `src1` only, but the mul_mat_id host-sort fallback
  reuses one stack-allocated `src1_slice` tensor for every expert. Experts
  with equal token counts produced identical keys, so the second expert
  reused the first one's quantized tokens. Fix: add the `src1->data` pointer
  to the cache key (same-tensor reuse across qkv/alpha/beta projections is
  preserved). Verified on the newer baseline (`d222767c7`): 20/20 clean
  single-case runs, 15/15 MUL_MAT_ID groups, full suite 14883/14883, GDN
  46/46 in all dispatch configs.
- **Block 11 (`11-meta-headroom.patch`)**: meta-buffer compute-container
  headroom 16x -> 128x for hybrid recurrent models. Fixes the llama-server
  graph-allocation abort (ggml.c:1804, "not enough space in the context's
  memory pool") seen with speculative MTP drafting under `--split-mode
  tensor`; source is fork branch `rdna-boosts` commit `f2a22a71` (not on
  `chunked-gdn`). Apply last; verified clean on this baseline.
- **Block 10 test hunk** was re-based against the baseline: its original
  context lines (Q6_K perf cases) were added by block 04, so the 12
  Qwen3.6-27B decode-shape cases are placed after the generic `mul_mat` perf
  loop in `make_test_cases_perf()` instead. Content is identical to the fork;
  only position differs.
- **`test-backend-ops.cpp` is shared** by blocks 02/03/04/10. The hunks are in
  different case regions; if upstream adds cases in those regions, re-base the
  affected hunks (each patch applies independently on the baseline, so
  re-basing is local to the failing file).
- **Block 06 is genuinely inseparable**: its 16 commits co-developed the fused
  mmvq kernel region, the `ggml-cuda.cu` try_fuse machinery, and the `fattn.cu`
  dispatch cluster. It is extracted as ONE combined diff on purpose. Do not try
  to split it.
- When upstream master moves past the recorded range and more than one block
  needs manual re-base hunks, use `scripts/make-patches.sh` from the fork to
  regenerate the set against the new tip and cut a new `baseline/<sha>`
  branch here instead of patching this branch's files by hand.
