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
| 07 | `07-gfx1151-mmvq-table.patch` | ggml/src/ggml-cuda/mmvq.cu | none |
| 08 | `08-host-buffer-revert.patch` | ggml/src/ggml-cuda/ggml-cuda.cu | none |
| 09 | `09-meta-device-wrapper-skip.patch` | src/llama.cpp | none |
| 10 | `10-q6k-mmvq-vdr2.patch` | ggml/src/ggml-cuda/{ggml-cuda.cu,mmvq.cu,vecdotq.cuh}, tests/test-backend-ops.cpp | none (test hunk re-based to stock) |
| 06 | `06-fused-core.patch` | mmvq.{cu,cuh}, ggml-cuda.cu (try_fuse), norm.{cu,cuh}, unary.{cu,cuh}, common.cuh, fattn.cu, fattn-tile.cuh | **blocks 03 and 04 MUST be applied first** (fattn-tile.cuh / fattn.cu territory) |

Blocks 01-05 and 07-10 are mutually independent and may be applied in any
order; **block 06 goes last, always**. The order above is the verified order.

## Verified apply sequence (this branch)

On a fresh checkout of the baseline SHA, the sequence

```
git apply patches/01-adaptive-mtp.patch
git apply patches/02-chunked-gdn.patch
git apply patches/03-bf16-kv-cache.patch
git apply patches/04-wmma-flash-attn.patch
git apply patches/05-bit-identical-decode-cpu.patch
git apply patches/07-gfx1151-mmvq-table.patch
git apply patches/08-host-buffer-revert.patch
git apply patches/09-meta-device-wrapper-skip.patch
git apply patches/10-q6k-mmvq-vdr2.patch
git apply patches/06-fused-core.patch
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

Convenience: `patches/rdna-boosts-all.patch` is the entire 48-commit net diff in
one patch (applies cleanly on the baseline SHA alone; not a substitute for the
per-block flow when you want reviewable increments).

## Failure handling (agent instruction)

> Apply the patches in `patches/` in the order given in this file to a fresh
> branch from the latest llama.cpp master tip. After each patch, run its
> verification command. If a patch fails to apply, use `git apply -3`; if
> 3-way fails, manually rebase the hunks against the current master and
> continue. Do not skip blocks. The fused core (`06-...`) goes last.

## Known notes

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
