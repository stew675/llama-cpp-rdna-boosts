# Phase-1 item 8 — the reference's 9 QSA graph-side flags audited (2026-09-22)

**Status:** DONE (audit, no code change).  Outcome: **7 of the 9 flags are already present or
superseded** in our block-14/15 QSA; **2 are genuinely un-ported prefill-score optimizations** and are
handed to the next session as follow-ups:

* **`QSA_SCORE_WMMA`** — a fused WMMA indexer score for the prefill band.  Ours is the per-op chain at
  `n_tokens > 1`; our fused `GGML_CUDA_QSA_INDEXER_SCORE` op is gated to `n_tokens == 1` (decode).
* **`QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP`** — prefill queries are scored in 512-token strips, each
  trimmed to the blocks it can actually see (the reference's compiled-in value is
  `min(n_tokens, 512)` for one stream).

Tree audited: fork `~/llama.cpp` branch `gap-closing` @ `6e5f34ebf`.  Reference:
`~/pwilkin-llama-cpp` @ `b0f31f587`.  The flags themselves were **deleted** by the reference's
`ac1ebb4e0` ("compile in the tuned defaults and drop the env gating"): all 9 are now hard-coded to the
value its measurements chose.  The pre-removal behaviour was read from `ddaf5214b`.

## What each flag did, and where it stands on our tree

| reference flag | behaviour | ours |
|---|---|---|
| `QSA_WHOLE_ATTN` | in the packed prefill layout, attend the whole strip instead of a query strip (the qsa3 packed-block path) | **present** — our `qwen4exp_qsa_sparse()` + qsa3 (`q->ne[1] >= 128`) |
| `QSA_BLOCK_SELECTION` | select whole key blocks (per-block score) rather than per cell | **present** — our block bias + `indexer_top_k` (the `blk_bias` path) |
| `QSA_COMPACT_METADATA` | upload the per-block half of the bias only (block index + tail start) when the visibility is scalar | **present** — our `blk_bias` / `blk_idx` / `blk_tail` derived bias (`GGML_QSA_DERIVED_BIAS`, W2's sibling) |
| `QSA_DIRECT_INDICES` | feed the selected cell indices straight to the attention (no dense-mask expansion) | **present** — our `indexer_top_k` output + the qsa3 cell-union |
| `QSA_NO_DENSE_MASK` | drop the dense mask and let the selected indices carry visibility | **superseded** — our derived visibility (`GGML_QSA_DERIVED_VIS` / `cell_vis`/`q_vis`); the reference's own `40c0b9c38` bug is N/A for us (item 3.5 audit) |
| `QSA_QUERY_STRIP` | prefill queries scored in strips (`0` = off; compiled-in `min(n_tokens, 512)`) | **absent** — we score all `n_tps` in one op |
| `QSA_SCORE_BOUNDS` | per strip, trim the scorer to the first `(max_query_pos + 1)/ratio` blocks (safe: trimmed blocks are `-inf` in the visibility metadata) | **absent** — we always score the full `n_blocks` |
| `QSA_SCORE_WMMA` | fused WMMA indexer score for `n_tps >= 128`, `idx_dim == 128`, `n_idx_h == 4` | **partial** — `GGML_CUDA_QSA_INDEXER_SCORE` (default on) is fused but gated to `n_tokens == 1`; prefill uses the per-op chain |
| `QSA_TOKEN_EMBD` | the direct/lazy token-embedding (PLE) reader feeding the sparse prefill | **present** — block 14's managed lazy reader / PLE loading |

## Why the two absent ones are worth a follow-up

The indexer chain is the largest F32 graph region in the qwen4exp prefill.  Measured on **our** build
(qwen4exp IQ4_NL, pp8192, `-ub 8192`, r=1):

| kernel family | ms |
|---|---:|
| `indexer_topk_*` (histogram/select/scan) | ~129 |
| `unary_op_kernel<op_relu>` (the score relu) | ~56 |
| the score `mul_mat` (folded into the F32 dense path) | not isolated |
| **indexer score/top-k total** | **~185 + the matmul** |

The reference's trim bounds the scorer to the visible prefix: for one sequence scored in 512-token
strips, strip `i` only needs `~512*(i+1)/ratio` of the `n_tokens/ratio` columns, so the mean width is
**~50 %** of full — i.e. roughly half the score matmul, relu and top-k work.  That is a
**low-single-digit %** prefill (recall-speed) candidate, and it is *selection-neutral* (the reference
documents the trimmed blocks as `-inf` in the visibility metadata, so the selection cannot change).

`QSA_SCORE_WMMA` is the other half: fusing the prefill score into one WMMA op (the reference builds
shared weights + a zero mask and runs it for `n_tps >= 128`).  Ours already has the fused-score *op*
for decode, so extending it (or adding a WMMA prefill arm) is a bounded change — but it must reproduce
the per-op chain bit-for-bit or be gated as a numerics review, and the width-purity gate must stay
green.

## What a port needs (scoping, not done)

1. `llama_memory_hybrid_idx_context::qsa_position_prefix(ubatch)` — true for one sequence with unique
   non-negative positions (the bound's precondition).
2. `qsa_prefix_limits(pos, n_tokens, strip, ratio, blocks, budget)` — the per-strip block bound.
3. `qwen4exp_query_strip(n_tokens, n_stream) = min(n_tokens, 512)` for `n_stream == 1`.
4. Thread `score_strip` / `score_key_limits` through the QSA graph input and `can_reuse` (the limits
   are baked into the graph, so a reused graph must agree on them).
5. In `build_qsa_top_k`, build one score per strip (concat) and trim each to its limit; the
   reserve-time synthetic ubatch (all positions equal) must stay unbounded, or the reserve sizes for a
   narrow scorer and then executes a full-width one.

Gate it as every other change: width probe PASS, same-seed text, A/B at `-b/-ub 4096`.  Because the
bound is selection-neutral, the width probe and the same-seed text are the bit-identity gate.

## Conclusion

Item 8 closes with **7/9 present-or-superseded**, and adds two prefill-score follow-ups to the plan
(`QSA_SCORE_BOUNDS`+`QSA_QUERY_STRIP`, then `QSA_SCORE_WMMA`) — both recall-speed items, both
selection/bounded-width changes whose gate is the existing width-probe + same-seed pair.
