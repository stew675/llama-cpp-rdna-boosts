# Issue #25 — greedy output changes with MTP draft length (verify batch width)

**Status:** root-caused on 3x gfx1201 (3x R9700, ROCm 7.14), 2026-09-10.
**Fix:** DELIVERED as `patches/0000-rdna-boosts-block-00-structural-and-architecture-fix.patch`
(2026-09-10) — the first block of the 15-patch set.  This document is the
investigation record that led there.
**Scope:** investigation record for the open `regression` issue
[stew675/llama-cpp-rdna-boosts#25](https://github.com/stew675/llama-cpp-rdna-boosts/issues/25).

## Symptom

Same build, same prompt, greedy (`temp 0`): `--spec-draft-n-max 2` and
`--spec-draft-n-max 4` produce different text.  Deterministic within an arm.
`--spec-type none` differs from both.  Reporter: 2x gfx1201.  Reproduced here
on 2x gfx1201 (matching) **and** on 1x gfx1201 (so it is not the block-12 AR).
Model: Qwen3.8-27B Q8_0 (dense `qwen35`, `full_attention_interval=4`,
`nextn_predict_layers=1`, f16 KV, `-fa auto`).

## Root cause

The HIP flash-attention **tile** kernel (used for every `n_q <= 8` batch, i.e.
both single-token decode and the speculative verify batch) splits the KV
dimension into `parallel_blocks` partial reductions.  The heuristic that picks
`parallel_blocks` in `launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`,
non-stream-K branch) is a function of

```
ntiles_dst = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3]
ntiles_x   = ceil(Q->ne[1] / ncols1)
```

so it depends on `Q->ne[1]` — the number of query rows, i.e. the verify batch
width.  Different `parallel_blocks` groups the fp32 partial sums (online
softmax + PV) differently, so the logits differ in the last bits.  Greedy
near-ties then flip, and the two arms stream apart.

Block 08's guard (`launch_fattn_tile_switch_ncols1`, `Q->ne[1] <= 8`) pins the
**kernel template / cols_per_block / nwarps / nbatch_fa** to the decode config
so decode and verify share a kernel, but it does **not** pin
`parallel_blocks`.  The guard is therefore incomplete: the tile kernel is
still batch-width dependent through the KV split.

### Direct evidence

`FA_DEBUG` instrumentation (see `fa-parallel-diagnostic.patch`) on the tile
launch, grouped by `ntiles_KV`:

| arm | n_q | ntiles_dst | ntiles_KV | parallel |
|---|---|---|---|---|
| n-max 2 | 3 | 36 | 16 | **16** |
| n-max 4 | 5 | 60 | 16 | **11** |

At `ntiles_KV = 8` both arms pick `parallel = 8` (identical); at
`ntiles_KV = 16` the heuristic diverges (16 vs 11).  That is exactly when the
greedy streams split (char ~1539 on 1 GPU / ~2018 on 2 GPUs for prompt p0).
Other `n_q` in the same arm also pick different splits (e.g. n-max 4 at
`ntiles_KV = 16`: n_q=1 -> 16, n_q=2 -> 14, n_q=3 -> 16, n_q=5 -> 11).

### Isolation matrix (1 GPU, prompt p0, 512 tokens)

| condition | none vs n-max2 | n-max2 vs n-max4 |
|---|---|---|
| FA on, GDN chunked on (default) | differ | **differ** |
| FA on, `GGML_CUDA_GDN_CHUNKED=0` | equal | **differ** |
| `--flash-attn off`, chunked on | differ | equal |
| `--flash-attn off`, chunked off | equal | equal |

So:
* the **n-max 2 vs n-max 4** split is caused by FA (needs FA on; chunked GDN
  irrelevant);
* a **separate** `none` vs spec difference is caused by the GDN chunked
  prefill path (the plain path runs full-chunked GDN on the prompt, the spec
  path runs chunked-prefix + sequential tail) — it disappears with
  `GGML_CUDA_GDN_CHUNKED=0`;
* with FA off **and** chunked off, decode and verify are byte-identical.

`LLAMA_KQ_MASK_DERIVED=0` (block-15 V3 off) changes nothing — block 15 is not
involved.

### Fix confirmation

Forcing the KV split to a batch-width-independent value makes the arms
identical:

* `FA_FORCE_PARALLEL=1` (or `4`), 1 GPU: `n-max 2 == n-max 4` for p0.
* `FA_PIN_PARALLEL=1` = `parallel_blocks = min(max_blocks_per_sm, ntiles_KV)`
  for `Q->ne[1] <= 8`, 2 GPU: `n-max 2 == n-max 4` on p0, p2, p3 (the three
  prompts that split in the baseline).

### Fix (delivered as block 00)

The KV split is made query-width independent for every small batch in
`launch_fattn`:

```cpp
const int ntiles_dst_eff = Q->ne[1] <= 8 ? (ntiles_z_gqa * K->ne[2] * Q->ne[3]) : ntiles_dst;
```

so decode and every verify width pick the same `parallel_blocks` as decode.
This is `patches/0000` (block 00) of the delivery; see `patches/README.md`
(the 2026-09-10 block-00 section) and the WORKLOG entry for the validation
record.  The earlier `FA_FORCE_PARALLEL` / `FA_PIN_PARALLEL` experiments below
are what proved the mechanism.

## Why upstream has it too

The `parallel_blocks` heuristic is stock upstream code.  Vanilla has **no**
`Q->ne[1] <= 8` guard at all, so it additionally picks different
`cols_per_block` for n_q 1 vs 3 vs 5 — its divergence is larger, not smaller.
This matches the maintainer's note that vanilla MTP n-max 2 != n-max 4.

## Reproduction

Baseline (clean 14-block/block-15 build, 2x gfx1201):

```sh
HIP_VISIBLE_DEVICES=0,1 llama-cli -m Qwen3.8-27B-Q8_0.gguf -ngl 99 \
  -sm tensor -ts 1/1 -c 8192 -ctk f16 -ctv f16 -fa auto \
  -p '<prompt>' -n 512 --seed 42 --temp 0 --top-k 1 \
  --no-display-prompt --single-turn --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max {2,4}
```

* baseline: p0 diverges @2018, p2 @1578, p3 @1901 (p1 identical)
* with `FA_PIN_PARALLEL=1`: p0/p2/p3 all identical

Diagnostic patch: `fa-parallel-diagnostic.patch` in this directory
(`FA_DEBUG=1` logs the tile launch config; `FA_FORCE_PARALLEL=N` /
`FA_PIN_PARALLEL=1` override the KV split).  It was applied to a scratch
`~/llama.cpp` build and reverted; the fork checkout is clean.

## Open questions

* Long-context (`~16k`) FA cost of pinning `parallel_blocks` for `n_q <= 8`
  (the per-launch env override in the diagnostic makes the A/B easy).
* Does the same heuristic dependence affect the WMMA/MMA prefill kernels via
  `stream_k`/`nblocks_stream_k`?  Not needed for this issue (prefill is not
  compared across widths), but the same class of bug.
