# Phase-1 item 15 — QSA prefill scorer trim (`QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`), ported 2026-09-22

**Status:** ported, default ON (`patches/0014`), bit-identical, small win at pp8192 / neutral at
depth.  Fork `~/llama.cpp` branch `gap-closing`, commit `10a839639`.

**What it is.** The reference scores the prefill indexer in `min(n_tokens, 512)`-token query strips
and, per strip, trims the scorer to the complete-block columns the strip can actually see.  Ported to
our tree, with one deviation forced by our fused indexer top-k (below).

## The bound

The QSA indexer pool enumerates **complete** blocks in ascending logical block order
(`llama_memory_hybrid_idx::set_input_qsa` groups cells by `pos/r` and assigns `n_bid++`), so a
complete block's ordinal cannot exceed its logical block number.  A block visible to a query at
position `p` therefore has ordinal `<= p/r`; every column past `(max_query_pos + 1)/ratio` is `-inf`
in the visibility metadata and can be dropped from the score.  The `+1` keeps the incomplete tail
block, which the fused top-k carries as ordinary cells (the reference appends it separately).

Preconditions (`qsa_position_prefix`): one sequence, unique non-negative positions, and — our
addition — `cell index == position` for every occupied cell, because our fused top-k's cell-index
partition needs it.  M-RoPE images (several cells per position) and multi-stream prefill fail this
and stay unbounded.

`qsa_score_key_limits` also leaves the **graph-reserve synthetic ubatch** (all positions equal)
unbounded, so the compute buffers are sized for a full-width scorer rather than a 4 %-wide one.

## The fused-top-k deviation

The reference's complete-block selector takes a trimmed `score` **and** a trimmed block→cell map, so
nothing indexes the trimmed columns.  Our fused `ggml_indexer_top_k` reads
`score[cell_blk[c]]` over the whole cell array, so a trimmed score would index out of range.  Two
options were measured:

* **(A) pad the trimmed columns back to full width with `-inf`** (no kernel change) — measured
  **neutral**: the extra full-width fill + `ggml_concat` per strip costs what the matmul/relu trim
  saves.
* **(B) hand the trimmed score straight to the top-k and clamp the cell range** — the shipped
  version.  `indexer_topk_radix_cuda_blocks` and the CPU oracle clamp the processed cells to
  `n_blocks*ratio`; because the single-sequence precondition makes the cache contiguous, the trimmed
  cells are exactly `c >= n_blocks*ratio`.  When nothing is trimmed (decode/verify, multi-stream,
  no bound) `n_blocks*ratio >= n_kv`, so the clamp is the identity.  No op-API change: the trimmed
  `blk_idx`/`blk_cells` views work because the bound only applies with `n_stream == 1`, where their
  stream stride is irrelevant.

(B) is what recovered the win; **the strip size is the other half of the knob.**

## Strip size (`LLAMA_QSA_SCORE_STRIP`)

Default **1024**.  Measured gfx1151, qwen4exp IQ4_NL, `-ctk/-ctv f16`, `-b 8192 -ub 8192`, interleaved:

| strip | pp8192 (3 rounds) | pp32768 (3 rounds) |
|---:|---:|---:|
| 256 | 1333.2 (1 run) | 1303.2 (1 run) |
| 512 | 1344.7 / 1352.0 / 1351.7 | 1310.4 / 1309.6 / 1310.9 |
| **1024** | **1358.2 / 1360.0 / 1358.8** | **1311.6 / 1310.8 / 1312.3** |
| 2048 | 1358.6 / 1361.4 / 1362.4 | 1307.6 / 1308.3 / 1308.3 |
| off | 1352.4 / 1352.1 / 1351.8 | 1311.8 (interleaved) / 1312.4 |

Reading: 256 is clearly worst (too many strip/top-k launches, smaller GEMM tiles); 512 is slightly
below off at pp8192 (same trim, 2× the launches); **1024/2048 are ~+0.5/+0.65 % at pp8192 and neutral
at depth**, with 1024 the better depth point.  The bound's own contribution (strip-only vs
strip+bound) is +0.6 % at pp8192.

`-b/-ub 4096` (the A/B protocol) is a wash: A 1293.4/1265.6, C 1294.7/1263.9 — the trim fraction is
lower there and the per-strip overhead proportionally larger.  Use the `-b/-ub 8192` target to see it.

## Gates

* **Bit-identity (the real gate, since the change is selection-neutral):**
  `test-logits-width-probe <IQ4_NL> prompts/prose-rdna-boosts.txt 3072 2048` ->
  `width_purity=PASS (worst maxdiff 0)` on **f16 / bf16 / q8_0**, and the per-W hashes are
  **byte-identical** with bounds-on vs `LLAMA_QSA_SCORE_STRIP=0` vs `LLAMA_QSA_SCORE_BOUNDS=0`.
* **Same-seed greedy text:** `61cebc1d31a9` (648 chars) identical bounds-on vs bounds-off.
* **Coherence:** coherent thinking trace, 31.6 t/s generation (1 GPU, gfx1151).
* **`plain == draft-mtp`:** **not run** — the qwen4exp IQ4_NL MTP head needs the shared sidecar,
  which this tree cannot load (`nextn_shared_target_tensors`, Phase 3); the prefill-only change
  cannot touch the decode/verify band, and the W=1..8 probe covers that half.
* **Op oracle:** no `test-backend-ops -o INDEXER_TOPK` case exists in this tree (the BETA-TESTING
  reference is aspirational); the kernel change is validated end-to-end by the probe and the
  strip/bounds A/B identity.

## Kill switches

* `LLAMA_QSA_SCORE_STRIP=0` — disables the strip **and** the bound (whole-batch full-width scorer).
* `LLAMA_QSA_SCORE_STRIP=<N>` — strip size in tokens.
* `LLAMA_QSA_SCORE_BOUNDS=0` — keeps the strips, disables the trim (overhead-only A/B arm).
* `LLAMA_QSA_SCORE_STRIP_DEBUG=1` — one `QSA score bounds: …` line per graph build.

## Remaining follow-up

`QSA_SCORE_WMMA` — fuse the prefill score into one WMMA op (`n_tps >= 128`, `idx_dim == 128`,
`n_idx_h == 4`).  Ours is `n_tokens == 1` only; a numerics change, so it needs the width probe +
same-seed text plus its own A/B.  It composes with this trim (the WMMA arm would run on the trimmed
width).  See [`2026-09-22-qsa-graph-flags-audit.md`](2026-09-22-qsa-graph-flags-audit.md).

## Files

`src/llama-memory-hybrid-idx.{h,cpp}` (`qsa_position_prefix`, `qsa_score_key_limits`),
`src/models/qwen4exp.cpp` (`qwen4exp_query_strip`, `qwen4exp_score_key_limits`, the strip loop in
`build_qsa_top_k`, `can_reuse`), `ggml/src/ggml-cuda/indexer-topk.cu` (cell-range clamp),
`ggml/src/ggml-cpu/ops.cpp` (CPU oracle clamp).  Patch:
[`patches/0014-…`](patches/).
