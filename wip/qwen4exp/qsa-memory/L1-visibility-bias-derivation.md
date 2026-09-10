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

## 7. Session state / where to pick up (2026-09-10 end of session)

**STEP 1 IS IMPLEMENTED — see `L1-step1-derived-block-bias-findings.md`** (results, the measured
4050.60 MiB at ub2048, the env gate `GGML_QSA_DERIVED_BIAS`, the corrected step-2 design, and an
**open MTP-acceptance question that must be resolved before packaging**). The notes below are the
original pre-implementation plan; §2-§4 of the design need the two corrections listed in the
findings' §4 (`!is_pos_2d()` cannot be used as a gate, and step 2's visibility must carry the
per-token `seq_has` test plus the 2-D tie rule).

Tree: `~/llama.cpp` = rdna-boosts `e2380eb67` **plus the L2 patch and the step-1 patch applied in
the working tree** (`../patches/0001-...L2a-L2m...` and `../patches/0002-derived-qsa-block-bias.patch`,
uncommitted on purpose — `make-patches.sh` treats the fork's tip as canonical). To rebuild the L1
patch after edits: scratch worktree at HEAD, `git apply` the L2 patch, commit, copy the 8 touched
files over it, `git diff`. Rebuild with
`cmake --build build-rocm --target llama-cli llama-bench -j 16` after
`export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH`.

Builds (volatile, `/tmp`): `bin-pristine` (no patch), `bin-l2a`, `bin-l2` (= l2a+l2m),
`bin-l0base` (pristine + `GGML_ALLOCATOR_DEBUG`). Re-measure any of them with
`tools/bufsize.sh`, `tools/ub-sweep.sh`, `tools/ab-coherence.sh`, `tools/peak-ledger.py` (the
last one needs the instrumented build; the instrumentation patch is described in
`L2-score-chain-findings.md` §1 — `ggml-alloc.c` `GGML_ALLOCATOR_DEBUG` + `AT_PRINTF`, array bumped,
O(N²) sort dropped, and a >1 GiB record filter).

Suggested order for the implementation (smallest self-contained piece first, each verified before
the next):

1. **Bias only** (−400 MiB, no new op, no FA change): add optional srcs `blk_idx`
   `[n_blocks, n_stream]` I32 and `blk_tail` `[n_tps, n_stream]` I32 to `ggml_indexer_top_k`; in
   `indexer_topk_value` add `if (blk_idx) sc += bias(b, t, s)`. Build both arrays in
   `set_input_qsa` (it already computes `n_bid`, `bid_idx`, `have_dead`/`dead_bid` and `seq_has`;
   encode invalid → `-1`, tail/dead → `INT32_MAX`, and `blk_tail[t] = ((q+1)/r)*r`), add the fields
   to `llm_graph_input_qsa`, and drop the graph's `ggml_add(score, inp->bias)` on that path.
   Expected 4450 → ~4050 MiB. Verify with the 40k same-seed A/B (must be byte-identical) plus a
   direct comparison of the derived `bias_val` against the uploaded `dst_bias` values for a few
   tokens/blocks (dump both for one layer at a probe depth).
2. **Mask** (−800 MiB): the `ggml_indexer_mask` op, the two `fattn-qsa.cu` staging lines →
   `maskh[tile0 + flat]`, the top-k's in-kernel `vis_val` from `cell_pos`/`q_pos`, and the
   `cell_pos` build in `set_input_qsa`. Same A/B plus a mask-equivalence dump
   (`indexer_mask` output vs `kq_mask` gathered at the top-k slots).
3. Re-measure the reserve (`tools/bufsize.sh`) and the ub sweep; update
   `L2-score-chain-findings.md` §6 and this file with the achieved numbers.

The next lever after that is the chunked top-k (removes the ~700 MiB concat peak and the 400 MiB
final score; the radix-select tie order must be reproduced exactly — see `L2-...findings.md` §6).
