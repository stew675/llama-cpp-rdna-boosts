# L1 step 1 — derived per-block bias: results + an open MTP numerics question

Status: **implemented & measured (2026-09-10).** Patch:
`patches/0002-derived-qsa-block-bias.patch` (271+/44−, 8 files, applies cleanly on the L2 base).
Env gate: `GGML_QSA_DERIVED_BIAS=0` restores the uploaded tensor (A/B + validation).

## 1. What it does

`build_qsa_top_k` no longer uploads the `[n_blocks, n_tps, n_stream]` F32 per-block bias
(`leaf_117`: 400 MiB at ctx 204800 / ub 2048) and no longer folds it into the block score with
`ggml_add`. Instead the *prefill* graph (n_tokens > 1) creates two compact I32 inputs, filled by
`set_input_qsa`:

- `blk_idx [n_blocks, n_stream]` — `INT32_MAX` for the spare block that holds the unpooled
  (incomplete-tail) cells, `-1` for a block that is not complete for this stream, else
  `bid_idx[b]` (the block's first-cell position, or its rank when the memory layer ranked the
  cells).
- `blk_tail [n_tps, n_stream]` — the per-token tail start `((q+1)/ratio)*ratio`.

`ggml_indexer_top_k` takes them as optional srcs 5/6 and computes, in the kernel:
`sc = score[b]; sc += (bi < 0 ? -inf : (bi >= blk_tail[t] ? 1e9f : 0.0f)); return sc + additive[c]`
— the same two IEEE f32 adds, in the same order, as the previous `add(score, bias)` + top-k pair.

The **per-sequence half of the original bias test is deliberately dropped** (`seq_has(bid_cell,
seq_id) → -inf`). It is redundant: a block's cells all share one sequence set (that is the grouping
key), so a foreign block has every cell masked by the attention mask, and `-inf + -inf = -inf`.
This is what makes the compact state per-*stream* rather than per-*token*.

Not gated on `!is_pos_2d()`: qwen4exp is IMROPE, so `n_pos_per_embd() == 4` and `is_pos_2d()` is
*always* true (it describes the position tensor, not the data). `bid_idx` and `tail_start` both
come from the same code path (rank units when the memory layer ranked the cells for mrope images,
position units otherwise), so their comparison is unit-consistent either way.

## 2. Measured result (ctx 204800, ub 2048, 3x R9700, q8_0 KV)

| build | compute buf /GPU | host buf |
|---|---|---|
| pristine (pre-L2) | 6690.40 MiB | 1262.70 MiB |
| L2 only | 4450.40 MiB | 1262.70 MiB |
| **L2 + derived bias** | **4050.60 MiB** | **862.90 MiB** |
| L2 ub1024 | 2274.35 MiB | 432.85 MiB |
| **L2 + derived bias ub1024** | **2074.55 MiB** | **432.85 MiB** |

−399.80 MiB/GPU (−1.2 GiB box) and the host buffer drops by the same amount (the bias was a
host-resident input), so the per-ubatch host build of `n_blocks × n_tps` floats and its upload
disappear as well — that cost scaled with depth.

llama-bench parity (`-p 20480 -n 256 -r 3 -b 2048`, 3-GPU, `-sm tensor`):

| build | pp20480 | tg256 |
|---|---|---|
| L2 ub2048 | 2480.87 ± 1.18 | 50.69 ± 1.41 |
| L2+derived ub2048 | 2482.43 ± 3.75 | 50.61 ± 1.41 |
| L2 ub1024 | 2209.08 ± 1.92 | 50.72 ± 1.42 |
| L2+derived ub1024 | 2213.65 ± 1.17 | 50.68 ± 1.39 |

Coherence (same-seed, temp 0, filtered for loader/timing lines): **byte-identical** against the L2
build on a 40000-token prompt (24 tokens), on the 3k prompt at ctx 32768, and on the 3k prompt with
`-ub 8` (small-batch prefill geometry). Every run exercised the derived path on every prefill
ubatch (the buffer delta proves it is active).

A host-side self-check (temporary, removed) compared `blk_idx`/`blk_tail` against the *original*
tensor formula — including the `seq_has` term — for **every (block, token) pair** of every QSA
batch at n_tokens ≤ 4096: 2180 calls, **0 mismatches**. The compact state is exactly the tensor's
content.

## 3. OPEN: the MTP acceptance drifts (generated text still identical)

The one thing that does *not* reproduce is the adaptive-MTP probe
(`benchmarks/mtp-adaptive-methodology.md` Protocol A / the `draft-mtp` path). Same commands
(3k prompt, `-n 96 --seed 42 --temp 0 -c 32768 -b/-ub 2048`, draft
`mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, `--spec-type draft-mtp`, `--verbosity 4`):

| build | draft acceptance | accepted/generated | acc per pos | #draft calls |
|---|---|---|---|---|
| pristine | 0.64583 | 62 / 96 | (0.844, 0.625, 0.469) | 32 |
| L2 only | 0.64583 | 62 / 96 | (0.844, 0.625, 0.469) | 32 |
| L2 + derived (full) | **0.61616** | 61 / 99 | (0.818, 0.606, 0.424) | 33 |
| L2 + derived, prefill only (`MAX=16`) | **0.61616** | 61 / 99 | (0.818, 0.606, 0.424) | 33 |
| L2 + derived, verify only (`MIN=17`) | 0.64583 | 62 / 96 | (0.844, 0.625, 0.469) | 32 |

**The generated text is byte-identical in every case** (both runs emit the same 96 target tokens,
stop at the same `n_tokens = 2269`, and produce the same 2174-token prompt eval). Only the
*draft/acceptance* pattern moves. The derived path is MTP-clean for the verify geometry and drifts
only for the prefill geometry. L2 itself is MTP-clean, so this is **specific to this change**, not
a systemic property of QSA graph restructuring.

Ruled out so far:

- **Not the bias values.** The state check above (0/2180), plus the arithmetic identity of the two
  adds. The plain paths are text-identical across three different geometries.
- **Not CUDA graph capture.** `GGML_CUDA_DISABLE_GRAPHS=1` gives the same drift (0.61616 vs
  0.64583) on both sides.
- **Not a different context sizing.** Both contexts keep `n_ctx = 32768` in both runs; the 192 MiB
  extra free memory is just this change's own savings (2 contexts × 64 MiB compute + 64 MiB host).
- **Not the meta-backend graph split count.** `sched_reserve: graph splits = 2` in both.

Working hypothesis (unproven): the ulp is *not* in the indexer values but in how the surrounding
multi-GPU graph is cut. Removing one `ggml_add` node (and swapping a 400 MiB host input for two
tiny ones) changes the tensor/node order and the buffer layout, and the meta-backend splitter /
AllReduce grouping for the indexer's `mul_mat` (reduction over `idx_dim`) is shape- and
structure-sensitive; a different grouping rounds differently. That is argmax-invisible (identical
text) but the MTP head's argmax over near-ties is sensitive, so one draft token flips and the
acceptance pattern moves.

### The next experiment (cheap, decisive)

Discriminate "layout/structure ⇒ ulp" from "values" by restoring the old *layout* with the derived
*values*: in the derived path, additionally create the `bias` tensor at its original shape and
consume it as `score = ggml_add(score, inp->bias)` with `set_input_qsa` filling it with **zeros**
(adding `+0.0f` first is value-preserving for the three bias constants `-inf`, `0.0f`, `1e9f`
and for the mask values, since none of them is `-0.0f`), while the top-k still derives the real
bias from `blk_idx`/`blk_tail`. If the MTP numbers return to 62/96 the cause is the graph
structure/buffer layout (and the follow-up is the peak-ledger/`ggml-alloc` liveness audit, since a
host-buffer aliasing bug would look exactly like this: host-side state correct, device-side
different). If they stay at 61/99, the cause is in the values after all, and the next step is a
device-side dump of `blk_idx`/`blk_tail` inside `ggml_cuda_indexer_top_k` compared against the
host-computed expectation.

Until then the honest classification is: **memory win real, plain coherence byte-identical,
MTP acceptance 0.646 → 0.616 with identical output text** — which passes the documented gate
(acceptance ≫ 0.45, MTP still accelerates) but is an *unexplained* numerics change and must not be
packaged into the delivery before it is understood.

## 4. Consequences for step 2 (the mask / derived visibility)

The design in `L1-visibility-bias-derivation.md` §2–§3 needs two corrections found here:

1. `!is_pos_2d()` **cannot** be used as a gate (always true for IMROPE). The visibility must be
   derived in a way that is exact for both plain text and mrope images.
2. The mask's predicate is `!empty && seq_has(cell, token_seq) && !(pos_c > pos_q) &&
   !(pos_c == pos_q && ext_c.is_2d_gt(qx, qy))` (llama-kv-cache.cpp `set_input_kq_mask_impl`). The
   per-*token* `seq_has` cannot be dropped here (it is exactly what step 1 relied on the mask for),
   and the 2-D tie rule needs the ext values.

The workable compact form is the memory layer's **rank in the mask's own total order** (pos, then
ext.y, then ext.x — the existing validated comparator) plus a sequence bitmask:

- `cell_rank [n_kv, n_stream]` I32: rank in that order, `-1` for an empty cell.
- `q_rank [n_tps, n_stream]` I32: the query's rank (the existing ranked branch already computes it
  by binary search; the sort must then run on every ubatch — the removed mask build is
  O(n_kv × n_tps) host work, so an O(n_kv log n_kv) sort is a large net win).
- `cell_seq [n_kv, n_stream]` I32 (`seq_get_all` bitmask) and `tok_seq [n_tps, n_stream]` I32.
- visibility = `rank >= 0 && rank <= q_rank && (cell_seq & (1u << tok_seq)) != 0`.

That is 4 compact srcs; with step 1's blk_idx/blk_tail, score and cell_blk the op reaches 8 srcs
(GGML_MAX_SRC = 10). The FA side keeps the cheap trick from the design: pass a **compact mask**
`[width, n_tps, 1, n_stream]` (built by a new op from `top_k` + those arrays) so that both
`M_smem` staging sites in `fattn-qsa.cu` collapse to `maskh[tile0 + flat]` (the `idx` indirection
disappears). Expected: 4050.60 → ~3250 MiB/GPU, i.e. ub2048 speed below ub1024 memory.

Given §3, step 2 must be validated with **both** the plain same-seed A/B *and* the MTP probe, and
the step-2 work should start by resolving (or at least bounding) the §3 question, because a
400 MiB state change and an 800 MiB state change will be far harder to reason about together.

## 5. Files touched by the patch

| file | change |
|---|---|
| `ggml/include/ggml.h` | `ggml_indexer_top_k` signature: +`cell_pos`, `q_pos`, `blk_idx`, `blk_tail` (nullable) with the documented semantics |
| `ggml/src/ggml.c` | constructor: optional-src asserts, store srcs 3–6 |
| `ggml/src/ggml-cpu/ops.cpp` | CPU reference handles the derived bias + a `nullptr` additive |
| `ggml/src/ggml-backend-meta.cpp` | INDEXER_TOPK: assert *every* non-null src is mirrored (was hard-coded 0..2) |
| `ggml/src/ggml-cuda/indexer-topk.cu` | `indexer_topk_extra` threaded through the 3 kernels + launcher; host reads srcs 3–6; support check |
| `src/llama-memory-hybrid-idx.{h,cpp}` | `set_input_qsa` takes `blk_idx`/`blk_tail`; fills them instead of the tensor when asked |
| `src/models/qwen4exp.cpp` | `qwen4exp_derived_bias_enabled()` gate, compact inputs, `ggml_add` dropped on that path, top-k call |
