# L1 — derive the QSA visibility and per-block bias instead of uploading tensors

Status: **design (2026-09-10), not implemented.** Companion to `L2-score-chain-findings.md`.
Target: close the last **−1104 MiB/GPU** needed for "ub2048 speed at ≤ ub1024 memory"
(4450.40 → ~3250 MiB at ctx 204800 / ub 2048).

## 1. What is being removed

| tensor | size @ctx204800/ub2048 | consumers |
|---|---|---|
| `attn_inp_kq_mask` F16 `[n_kv, n_tps, 1, n_stream]` | 800 MiB | `ggml_indexer_top_k` (additive) **and** `ggml_flash_attn_qsa` (per-cell mask) |
| `leaf_117` per-block bias F32 `[n_blocks, n_tps, n_stream]` | 400 MiB | the graph `ggml_add(score, bias)` (prefill) and the fused `ggml_indexer_score` (decode, `[n_blocks,1]` — negligible) |

Both are **inputs** (live graph-wide) and both are pure functions of cell/query positions and
block bookkeeping that `set_input_qsa` already computes. Uploading them costs 1200 MiB of VRAM per
GPU plus, per ubatch, a host-side build of `n_kv × n_tps` F16/F32 plus its H2D transfer — that
cost grows with depth (at full ctx: ~42 GB of host writes + uploads over a 204800-token prefill).

Note: removing the top-k's *use* of the mask saves nothing on its own — the FA consumes the same
tensor. Both consumers must switch together.

## 2. The FA side: a compact mask (cheap)

`fattn-qsa.cu` reads the mask in exactly two device sites, both:

```cuda
const half * maskh = (const half *)(mask + nb33*(sequence % ne33) + nb31*col);  // column for this token
...
M_smem[flat] = maskh[identity ? tile0 + flat : idx[tile0 + flat]];   // two sites: ~L214, ~L362
```

The mask is read **only at the selected cells**, only to stage `M_smem`, and never used to skip
work. So if the mask is passed in a **compact slot layout** `[width, n_tps, 1, n_stream]`
(`width = min(n_kv, indexer_top_k + r - 1)`, the top-k's own array length — 16 MiB instead of
800 MiB), the row index becomes the *slot* and both sites collapse to:

```cuda
M_smem[flat] = maskh[tile0 + flat];
```

(the `identity` distinction disappears for the mask; the K/V gathers keep using `idx`). The column
pointer arithmetic (`nb31` row stride, `nb33` sequence stride, `col` = token) is unchanged because
the compact tensor keeps the same dim order.

The compact mask is built by a new small op:

```
ggml_indexer_mask(top_k I32 [width, n_tps, 1, n_stream],
                  cell_pos I32 [n_kv, n_stream],
                  q_pos    I32 [n_tps, n_stream]) -> F16 [width, n_tps, 1, n_stream]
  out[i,t,s] = vis(cell_pos[top_k[i,t,s], s], q_pos[t,s]) ? 0.0f : -INFINITY
```

## 3. The top-k side: derive the additive and the bias in-kernel

`ggml_indexer_top_k(score, cell_blk, additive, width)` currently computes
`value(c,t,s) = (score[b] + bias[b,t]) + additive[c,t]` where `additive` is the mask and the bias
was added by the graph into `score`. Extend the op so that when `additive == nullptr` it takes
compact srcs instead and computes, in the same order (bit-identical):

```
value = (score[b] + bias_val) + vis_val
```

with, using the *exact* predicates of `set_input_qsa` (llama-memory-hybrid-idx.cpp):

- **bias** (blk_bias path, per block b, token t, stream s):
  ```
  blk_idx  [n_blocks, n_stream] I32   // bid_idx[b] = pb*r for real blocks;
                                      // INT32_MAX for the dead/tail block; -1 when the block is
                                      // not full (b >= n_bid) or its cell is foreign (!seq_has)
  bias_val = (blk_idx[b,s] < 0)              ? -INFINITY
           : (blk_idx[b,s] >= tail_start(t)) ? 1e9f
           :                                   0.0f
  tail_start(t) = ((q_t + 1)/r)*r             // computed in-kernel from q_pos and r
  ```
  `blk_idx` folds in `n_bid` (invalid → -1), `seq_has` (foreign → -1), `have_dead`/`dead_bid`
  (the dead block is always the tail: `bid_idx = INT32_MAX`) and the "block holds the tail"
  test, so a single I32 array (`n_blocks × n_stream`) replaces the `[n_blocks × n_tps]` F32.
  Verified against `llama-memory-hybrid-idx.cpp:722-741`.
- **visibility**:
  ```
  cell_pos [n_kv, n_stream] I32   // position of the cell in this stream, -1 if not in the
                                  // stream's sequence (the mask's seq_has test)
  q_pos    [n_tps, n_stream] I32
  vis_val  = (cell_pos[c,s] >= 0 && cell_pos[c,s] <= q_pos[t,s]) ? 0.0f : -INFINITY
  ```
  `cell_pos <= q_pos` is the causal predicate of `set_input_kq_mask_impl` (mask_keep = `0.0f`,
  mask_drop = `llama_cast<T>(-INFINITY)`).

Exactness notes (from `HANDOVER.md` §4.4): keep `+ 0.0f` (it normalises `-0.0f`, and the top-k's
ordered-float key distinguishes the two zeros); do not reorder the adds.

## 4. Gating (keep the tensor path everywhere else)

Derived path only when all hold, else fall back to the existing tensors:

- `blk_bias == true` (already implies causal, no alibi, mask shape matches, `n_swa == 0`).
- `!ubatch->is_pos_2d()` — the 2-D/mrope mask uses the `pos, ext.y, ext.x` order; the QSA memory
  layer already builds `rank`/`ordered` for that case, so gate on the 1-D path only.
- `!ranked` (i.e. no 2-D positions were seen in this batch) — `tail_start` is then in the same
  units as `cell_pos`/`q_pos` (raw positions).
- 1-D `n_stream` handling unchanged (each stream gets its own `cell_pos` column).

Decode keeps the tensors only where they are cheap: the decode graph's mask is `[n_kv, 1]`
(400 KiB) and the bias is `[n_blocks, 1]` (200 KiB), so even an unconditional switch is harmless —
but the fused `ggml_indexer_score` (decode) applies the bias internally, so the simplest split is:
**the derived inputs replace the mask for both the top-k and the FA in every graph; the bias
derivation replaces only the prefill graph's `add(score, bias)` + the blk_bias top-k additive**
(the decode path keeps the small bias tensor for the fused op).

## 5. Work items

1. `ggml_indexer_mask` op (ggml.c + ggml-cpu/ops.cpp reference + CUDA kernel) ~120 lines.
2. `ggml_indexer_top_k`: optional compact srcs (`cell_pos`, `q_pos`, `blk_idx`) + kernel path
   (~40 lines in `indexer-topk.cu`); keep the `additive` path for all other callers.
   src count: score, cell_blk, additive = 3 → +3 = 6 (GGML_MAX_SRC = 10).
3. `fattn-qsa.cu`: two mask-staging lines → `maskh[tile0 + flat]`.
4. `llama-memory-hybrid-idx.cpp`: build `cell_pos`, `q_pos`, `blk_idx` in `set_input_qsa`
   (it already walks the cells, `rank`/`bid_idx` and the `seq_has` bookkeeping); add fields to
   `llm_graph_input_qsa`.
5. `qwen4exp.cpp`: pass the compact srcs; drop the `add(score, bias)` and the `kq_mask`
   dependence in the derived case.
6. Validation: same-seed 40k A/B (bit-identical) **plus** a direct mask-equivalence check
   (dump `ggml_indexer_mask` output vs the gathered full mask for a few layers/positions).

## 6. Risks

- The FA's `M_smem` staging is on the hot path; the change removes an indirection (should be
  neutral or faster), but the compact mask must match the *slot* order the kernel iterates
  (`tile0 + flat` over `n_top_k`), which is the top-k's own ordering.
- A mask row (`width`) shorter than `n_top_k` used by the FA would be a silent mis-index; assert
  `mask->ne[0] >= top_k->ne[0]` in the op.
- The `1e9f` tail sentinel and `-INFINITY` must be produced as exact f32 constants; the top-k's
  tie handling and the ordered-float key are untouched, so selection is bit-identical *if* the
  values are.
- Expected gain: 800 MiB (mask) + 400 MiB (bias) → **~3250 MiB/GPU at ub 2048**, plus the
  per-ubatch host build/upload removal (a depth-scaling prefill win).
