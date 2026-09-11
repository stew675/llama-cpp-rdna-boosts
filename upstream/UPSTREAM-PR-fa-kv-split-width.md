# Upstream-PR proposal — ggml-cuda: make the FA KV split query-width independent for small batches

Status: PREPARED 2026-09-10.  Verified on the delivery fork rebuilt at the fork point
(`9113cc188`): the patch is the first commit of the `rdna-boosts` `structural-fixes` branch, builds, and
is covered by the full delivery validation below.  Applies clean to `9113cc188` by construction (it is
`git diff 9113cc188 <block-00>`).  Not yet filed.  Patch file:
`UPSTREAM-PR-fa-kv-split-width.patch` (1 file, `ggml/src/ggml-cuda/fattn-common.cuh`, +9/−1).

**Value: a real greedy-determinism bug on every CUDA/HIP backend.**  Speculative decoding verifies
`n_draft+1` tokens in one target forward.  Two verify widths (e.g. `--spec-draft-n-max 2` vs `4`) can
produce different greedy output because the flash-attention tile kernel's KV-split heuristic
(`parallel_blocks`) is keyed off `ntiles_dst`, which is a function of `Q->ne[1]` (the number of query
rows).  Decode (`n_q = 1`) and every verify width then get a different number of partial reductions,
the online-softmax / PV partial sums associate differently, and near-ties flip.  This is not new to the
fork — it is upstream code; the fork hit it while validating MTP (issue #25).

## Root cause

In `launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`), the non-stream-K `parallel_blocks` heuristic
maximises wave efficiency over

```
ntiles_dst = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3]
ntiles_x   = ceil(Q->ne[1] / ncols1)
```

`Q->ne[1] > 1` (a verify batch) therefore picks a different `parallel_blocks` than decode, even though
both run the same kernel instantiation.  Measured on gfx1201 (Qwen3.8-27B Q8_0, f16 KV), at the same KV
length:

| `Q->ne[1]` | `ntiles_dst` | chosen `parallel_blocks` |
|---|---|---|
| 1 (decode) | 12 | 16 |
| 3 (verify n-max 2) | 36 | 16 |
| 5 (verify n-max 4) | 60 | 11 |

`n_q=3` and `n_q=5` therefore reduce the same KV in a different grouping and return logits that differ
in the last bits; greedy (temp 0) then streams apart at a near-tie.

## Change

Evaluate the heuristic as if `n_q == 1` for every small batch:

```cpp
const int ntiles_dst_eff = Q->ne[1] <= 8 ? (ntiles_z_gqa * K->ne[2] * Q->ne[3]) : ntiles_dst;
...
const int nblocks_total = ntiles_dst_eff * parallel_blocks_test;
```

`n_q > 8` (prefill) is unchanged.  `parallel_blocks` only selects the reduction grouping — it is not a
correctness parameter — so this is a numerics-only change.

## Validation

* Issue #25 reproduction (Qwen3.8-27B Q8_0, f16 KV, greedy seed 42, 512 tokens): `--spec-draft-n-max 2`
  == `--spec-draft-n-max 4` on 3 prompts × 2-GPU and on 3-GPU; adaptive == both.  Without the patch the
  same arms diverge (char 2018 / 1578 / 1901).
* Plain decode (`--spec-type none`) is **byte-identical** with and without the patch (2-GPU and 1-GPU) —
  `n_q = 1` is unchanged.
* MTP acceptance gate unchanged/healthy (dense Qwen3.8-27B: 0.479; MoE Qwen3.6-35B-A3B: 0.669).
* Clean-apply: fresh `9113cc188` + the 15-block delivery → strict `git am`, zero whitespace warnings,
  applied tree == canonical.

**Not validated here:** NVIDIA/turing+ paths (`stream_k` is untouched; the affected non-stream-K branch
is shared, but only AMD was exercised), quantized-KV caches, and long-context (16k) FA cost of the
pinned split.  The change is numerics-only and `n_q <= 8`, so regressions are not expected, but the
draft PR should re-run the CUDA `test-backend-ops FLASH_ATTN_EXT` matrix and a depth-16384 decode A/B.

## Notes for filing

* Repro is deterministic, not a race: same build + seed → same text; only the draft length changes.
* The same class of bug exists if `cols_per_block`/`nthreads` are chosen from `Q->ne[1]` (the fork's
  block 08 also pins those for `n_q <= 8`; upstream does not).  A reviewer may prefer a single small
  helper that makes the whole small-batch config query-width independent rather than pinning one field.
* Base: `9113cc188`.  Re-create the branch from current master before filing and re-run the validation.
