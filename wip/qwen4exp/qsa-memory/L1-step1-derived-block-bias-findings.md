# L1 step 1 — derived per-block bias: results + the MTP-acceptance question

Status: **implemented & measured (2026-09-10).** Patch:
`patches/0002-derived-qsa-block-bias.patch` (314+/44−, 8 files, applies cleanly on the L2 base).
Env gate: `GGML_QSA_DERIVED_BIAS` = 0 (uploaded tensor) | 1 (derived, default) | 2, 3 (diagnostics).
The diagnostics are scaffolding for §3 and must be stripped if/when this is packaged.

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

## 2b. Step 2 (derived visibility) — state landed and validated (this session)

The mask predicate has the same shape as the per-cell bias the memory layer **already** computes, so
step 2 needed **no new op and no compact-mask tensor**. `set_input_qsa` now also fills

- `cell_vis [n_kv, n_stream]` I32 — a cell's compaction key: the cell's position, or its rank in
the mrope order when `ranked`, and `-1` when the cell is empty or belongs to another sequence;
- `q_vis [n_tps, n_stream]` I32 — the query token's key (`q`, the same value the per-cell bias
  already compares against).

Visibility is `0 <= cell_vis[cell] <= q_vis[token]` — the predicate `set_input_kq_mask_impl`
materializes (`!empty && seq_has && !future`, with the mrope 2-D order handled by ranking), which is
why the values are identical by construction. The top-k already accepted these as optional srcs 3/4
from step 1 (`extra.cell_pos`/`extra.q_pos`), so the graph now passes them and the kernel derives
the visibility. Gate: `GGML_QSA_DERIVED_VIS=0` keeps the mask as the additive (A/B).

**Validated**: generated text byte-identical with `GGML_QSA_DERIVED_VIS=1` vs `=0` on the 3k prompt
and the 40k prompt (seed 42, temp 0, ctx 204800, ub 2048) — i.e. identical top-k selections.
Reserve: 4050.60 → **4051.39 MiB/GPU** (+0.79 MiB = the two arrays, 800 KB + 8 KB), host 863.69.
The mask is *still* allocated because the FA reads it, so this step alone saves nothing (as the L2
findings predicted) — it is the prerequisite for the −800 MiB.

**Remaining work for the −800 MiB — the FA switch, also needing no new op.** `fattn-qsa.cu` already
has the selected slot indices `idx[]` for the tile it is staging, so both `M_smem` sites can compute
the value inline instead of gathering it from the mask (the query token and stream are already in
scope there):

| site | now | instead |
|---|---|---|
| `fattn-qsa.cu` ~L214 (F16/Q8_0 gather) | `M_smem[flat] = maskh[identity ? tile0 + flat : idx[tile0 + flat]];` | `const int g = identity ? tile0 + flat : idx[tile0 + flat];` then `M_smem[flat] = (cell_vis[g] >= 0 && cell_vis[g] <= q_vis_t) ? 0.0f : -INFINITY;` |
| `fattn-qsa.cu` ~L362 (bf16 gather) | same | same |

Concretely: add `cell_vis`/`q_vis` as optional srcs of `GGML_OP_FLASH_ATTN_QSA` (`ggml.h` + the
`ggml.c` ctor), thread them through the host entry/launcher and `fattn-qsa.cuh`, allow
`kq_mask == nullptr` when they are set (then the graph passes them and stops building the mask, so
the 800 MiB tensor is pruned), and update the CPU reference (`ggml-cpu/ops.cpp` ~9366/9373/9494) plus
the meta-backend mirrored-src case for the new src count. Expected after that: **~3250 MiB/GPU at
ub 2048** (ub 1024 ~2100) — the campaign's BEST outcome: ub2048 speed below ub1024's memory.
Validate with the same `GGML_QSA_DERIVED_VIS` A/B (text must stay byte-identical) + `tools/bufsize.sh`.

## 2c. Step 3 — the flash-attention switch (support landed; the prune measured, then blocked — **LANDED, see §2d**)

`GGML_OP_FLASH_ATTN_QSA` now takes two optional srcs 5/6 (`cell_vis`, `q_vis`; `ggml.h` + the
`ggml.c` ctor, which also accepts a **null** mask when they are present). Both `M_smem` staging
sites in `fattn-qsa.cu` (~L214 F16/Q8_0 gather and ~L362 bf16) now call one inline helper
(`qsa_cell_mask`) that returns `__float2half(0.0f)` / `__float2half(-INFINITY)` from
`cell_vis[g] in [0, q_vis[token]]` when the keys are present, else the gathered mask — the mask's
two fp16 constants exactly. The CPU reference aborts with a clear message on the derived form
(CUDA-only), and the meta backend accepts a null src slot (`SPLIT_AXIS_UNKNOWN`).

**Validated (mask still present):** generated text byte-identical with `GGML_QSA_DERIVED_VIS=1` vs
`=0` on the 3k and 40k prompts; reserve unchanged at 4051.39 MiB/GPU.

**The −800 MiB prune works at the reserve level and was measured**: passing no mask (to the top-k
*and* the FA) gives **3251.39 MiB/GPU, host 63.69 MiB** (from 4051.39 / 863.69) — i.e. **−799.21
MiB/GPU**, exactly the campaign's BEST target (ub2048 at *less* memory than pristine ub1024's 3347
MiB, with ub2048 speed; ub1024 would land near 1475). It also removes the O(n_kv × n_tps) host mask
build from every prefill ubatch.

**Blocker (two attempts, both reverted).** The prune is not a 2-line flip: leaving the mask
*created but unreachable* is invalid in this backend, and removing its creation trip-wires a chain
of unguarded null-buffer probes.

- **Attempt 1** (mask still created by `build_attn_qsa`, only passed to nothing): aborts with
  `ggml-backend.cpp GGML_ASSERT(buffer)` inside `ggml_backend_buffer_get_usage`, called from the
  sched's split-input copy loop (`ggml-backend.cpp:1730`).
- **Attempt 2** (the proper fix: `qwen4exp_want_derived_vis(cparams, hparams, n_tokens)` — the same
  predicate `blk_bias` computes from the mask's shape, without the mask — threaded into the top-k
  call site and `build_attn_qsa`, which then never call `inp->get_kq_mask()`; the dense
  `kq_mask_all` chain skipped; the attn input's `can_reuse` null-guarded; the FA launch and `maskh`
  made null-safe): **the reserve is still 3251.39 MiB/GPU with a 63.69 MiB host buffer**, i.e. the
  mask really is gone and the −800 MiB is confirmed — but the run then aborts on a *second*
  unguarded probe, `GGML_ASSERT(buffer)` in **`ggml_backend_buffer_get_type`**
  (`ggml-backend.cpp:211`). Tolerating `get_usage(nullptr)` (returning COMPUTE, semantically right:
  weights are always allocated) only moved the failure to the next probe.

The probes that touch a tensor buffer without a null check, in the meta/sched path:
`ggml-backend.cpp:1730` (`get_usage`), `ggml-backend-meta.cpp:856` (`get_type` inside
`calculate_split_state`), `ggml-backend-meta.cpp:1249` and `2138`/`2275`/`2314`
(`ggml_backend_buffer_is_host`). So *some* tensor with a null buffer survives into the meta
split/assign phase: a tensor that is created but never consumed, hence never allocated — expected
for an input nothing reads, which the meta backend does not tolerate.

**Next step for that (small, decisive):** name it. Put a `backtrace()`/`backtrace_symbols_fd` (or a
`GGML_ABORT` under gdb) in the `GGML_ASSERT(buffer)` of `ggml_backend_buffer_get_usage` /
`get_type`, or instrument `ggml_backend_meta_get_split_state` to print `tensor->name`/`tensor->op`
before its buffer deref, and run the derived path once. Then either (a) give that tensor a buffer
(allocate/consume it) or (b) guard the probes — a null buffer simply means "not a weight, not yet
allocated", so returning COMPUTE/skipping is safe for all of them. The derived-path plumbing itself
needs no further work: it is written, compiles, and produces the measured 3251.39 MiB.

**Kept from attempt 2** (harmless, and needed for the flip): `fattn-qsa.cu` now tolerates a null
mask in the kernel (`maskh` guard) and in both launch argument lists, and the `qsa_sparse` /
`cell_vis` lookup block sits slightly earlier in `build_attn_qsa` than before.

## 3. The MTP-acceptance drift — RESOLVED as a buffer-layout effect, not arithmetic

The adaptive-MTP probe (`benchmarks/mtp-adaptive-methodology.md` Protocol A; `tools/mtp-ab.sh`)
does not reproduce: acceptance 0.64583 (62/96) → 0.61616 (61/99) on the derived path, with
**byte-identical generated text** in every configuration. Four configurations of the *same binary*
isolate it:

| `GGML_QSA_DERIVED_BIAS` | bias tensor in graph | derived in kernel | acceptance | acc/pos |
|---|---|---|---|---|
| 0 | 400 MiB, real values | no | 62/96, **0.64583** | (0.844, 0.625, 0.469) |
| 1 (default) | absent | yes | 61/99, **0.61616** | (0.818, 0.606, 0.424) |
| 2 | 400 MiB, **all zeros** | yes | 62/96, **0.64583** | (0.844, 0.625, 0.469) |
| 3 | 1 broadcast element, zero | yes | 61/99, **0.61616** | (0.818, 0.606, 0.424) |

Read off that table:

- **The derived arithmetic is exactly neutral.** Mode 2 runs the *derived* top-k (identical kernel,
  identical `blk_idx`/`blk_tail`) and reproduces the tensor path bit-for-bit in the metric that is
  sensitive to ulps — while the only difference is that the graph also carries a *zero* bias
  tensor. Zeros cannot change an add, so the derived path's values are not the cause. This matches
  the arithmetic argument (same two IEEE adds, same order) and the 0/2180 host-side state check.
- **The `add` node and the node count are irrelevant.** Mode 3 keeps the node (the add is in-place,
  so it costs nothing: mode 3's reserve equals mode 1's 4050.60 MiB) and still drifts.
- **What moves the result is the presence of the 400 MiB host input**, i.e. a *buffer layout* /
  *buffer size* change, with no semantic content at all: mode 2 vs mode 3 differ only in the size
  of a tensor that holds zeros. The effect is deterministic (2 configurations with the tensor →
  62/96 twice; 2 without → 61/99 twice) and reproducible, so it is not a free-running race.

Interpretation: the graph's numerical result depends on *unrelated allocation sizes* somewhere in
the multi-GPU path — a pre-existing property of the engine (the L2 patch is MTP-clean, so it is not
"any layout change"; the 400 MiB host input with its per-device mirrored copy is a specific
trigger). Candidates, in order of plausibility: a copy/AR/kernel path that branches on pointer
alignment (a different vector width changes the summation order → ulp), or an allocator
liveness/aliasing defect (this allocator demonstrably has one: the `cpy`-into-view `n_views`
underflow found while doing L2). **Not diagnosed further; it is the same class of question as the
`archive/work/` AR work and is not a blocker for the memory campaign** — the derived path is
arithmetically exact, output text is identical in all modes, and 0.61616 is far above the repo's
MTP gate (≥ ~0.45 with MTP still faster than plain decode).

If it is ever chased, the cheapest next probes are: (a) the same four-mode table with
`GGML_CUDA_ALLREDUCE=nccl` and with the meta butterfly (does the trigger follow the AR backend?),
(b) the same table on a pristine tree *without* L2 (does the L2 concat participate?), (c) the
instrumented `ggml-alloc` peak ledger at the prefill ubatch with mode 1 vs mode 2 (look for two
live tensors sharing a host-buffer range, and for any tensor whose offset differs between the
modes beyond the bias slot itself).

## 4. Consequences for step 2 (the mask / derived visibility)

The design in `L1-visibility-bias-derivation.md` §2–§4 needs three corrections found here:

1. `!is_pos_2d()` **cannot** be used as a gate (always true for IMROPE). Nothing may be gated on it.
2. The mask's predicate is `!empty && seq_has(cell, token_seq) && !(pos_c > pos_q) &&
   !(pos_c == pos_q && ext_c.is_2d_gt(qx, qy))` (llama-kv-cache.cpp `set_input_kq_mask_impl`). The
   per-token `seq_has` cannot be dropped here (it is exactly what step 1 relied on the mask for),
   and the 2-D tie rule must be reproduced.
3. Because of §3, step 2 must not be judged by the MTP probe alone.

Simpler exact encoding than that doc, and the one to implement — **two I32 arrays, one comparison
pair, no bitmask ops**:

- `cell_key [n_kv, n_stream]` I32: for each cell `c` of the stream's cell array, `-1` unless the
  cell is non-empty *and* `seq_has(c, seq_of_stream)`; otherwise its **rank among the stream's own
  qualifying cells** in the existing (pos, ext.y, ext.x) order.
- `q_rank [n_tps, n_stream]` I32: the query's rank in that same filtered sequence (the number of
  qualifying cells ordered before it — the existing `ranked` branch already locates it by binary
  search, just on the filtered list).
- visibility `vis = (cell_key >= 0) && (cell_key <= q_rank)`; values `0.0f` / `-INFINITY`, and the
  top-k must keep the exact add order `(score + bias) + vis + additive`.

Why this is enough: filtering by sequence and re-ranking *within* the query's sequence makes the
single `<=` comparison correct for every case — a foreign-sequence cell is `-1` (excluded by
`>= 0`), a same-sequence cell of another subsequence that interleaves in cell order is impossible
by construction, and a future cell has a rank above `q_rank`. `-1` needs the `>= 0` test, so it is
two comparisons and an AND; both are per-(cell,token) in the top-k kernel (cheap, no tensor), and
the FA's compact mask is built from the same two arrays.

Consumers, both of which must switch together (neither alone saves anything):

- **top-k**: two more optional srcs (`cell_key`, `q_rank`) — srcs become score, cell_blk, additive,
  blk_idx, blk_tail, cell_key, q_rank = 7 ≤ GGML_MAX_SRC (10).
- **FA**: `fattn-qsa.cu` reads the mask at *selected* cells only, so pass a **compact mask**
  `[width, n_tps, 1, n_stream]` F16 (~8 MiB instead of 800 MiB) for which row index == slot, and
  both `M_smem` staging sites collapse to `maskh[tile0 + flat]` (the `idx` indirection disappears).
  Building that tensor needs a gather of `cell_key` by `top_k` plus the compare — either a small new
  op `ggml_indexer_mask(top_k, cell_key, q_rank)` (~120 lines: ggml.c + CUDA kernel + CPU ref) or a
  chain of existing ops (`ggml_get_rows` on a `[1, n_kv]` view of `cell_key`, then a compare/shift
  that maps `{true,false} → {0.0f,-INFINITY}`); the chain avoids new kernel code and its
  intermediates are small (`width × n_tps × n_stream` I32 ≈ 16 MiB), so try it first.

Expected: 4050.60 → ~3250 MiB/GPU at ub 2048, i.e. ub2048 speed below ub1024's memory (the
campaign's BEST outcome; ub1024 would land near 2100).

## 5. Files touched by the patch

| file | change |
|---|---|
| `ggml/include/ggml.h` | `ggml_indexer_top_k` signature: +`cell_pos`, `q_pos`, `blk_idx`, `blk_tail` (nullable) with the documented semantics; `ggml_flash_attn_qsa`: +`cell_vis`, `q_vis` (nullable) |
| `ggml/src/ggml.c` | constructor: optional-src asserts, store srcs 3–6 |
| `ggml/src/ggml-cpu/ops.cpp` | CPU reference handles the derived bias + a `nullptr` additive |
| `ggml/src/ggml-backend-meta.cpp` | INDEXER_TOPK: assert *every* non-null src is mirrored (was hard-coded 0..2) |
| `ggml/src/ggml-cuda/indexer-topk.cu` | `indexer_topk_extra` threaded through the 3 kernels + launcher; host reads srcs 3–6; support check |
| `ggml/src/ggml-cuda/fattn-qsa.cu/.cuh` | optional `cell_vis`/`q_vis` srcs; both `M_smem` sites derive via `qsa_cell_mask` |
| `ggml/src/ggml-cpu/ops.cpp` | INDEXER_TOPK reference (derived bias + visibility, nullable additive); FLASH_ATTN_QSA reference aborts on the derived form (CUDA-only) |
| `ggml/src/ggml-backend-meta.cpp` | INDEXER_TOPK: every non-null src mirrored; FLASH_ATTN_QSA: srcs 4–6 mirrored or UNKNOWN (null mask) |
| `src/llama-memory-hybrid-idx.{h,cpp}` | `set_input_qsa` takes `blk_idx`/`blk_tail` **and `cell_vis`/`q_vis`**; fills them instead of the tensor/mask when asked (and zero-fills the tensor in the diagnostic modes) |
| `src/models/qwen4exp.cpp` | `qwen4exp_derived_bias_mode()` + `qwen4exp_derived_vis_enabled()` gates, compact inputs, `ggml_add` dropped on that path, top-k call |

## 6. State at the end of this session

- `~/llama.cpp` working tree = rdna-boosts `e2380eb67` + the L2 patch + this patch (uncommitted by
  design; `make-patches.sh` treats the fork tip as canonical). Rebuild:
  `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH && cmake --build build-rocm --target llama-cli llama-bench -j 16`.
- Binaries: `/tmp/bin-l1d` (this patch, all four modes), `/tmp/bin-l1` (the earlier build with the
  same derived path + removed instrumentation), `/tmp/bin-l2`, `/tmp/bin-pristine`.
- The step-1 lever is *landed in the WIP tree* with a documented, arithmetically-exact basis; the
  remaining campaign milestone is step 2 above (mask, −800 MiB).

## 2d. The prune LANDED — 3251.39 MiB/GPU (session 2)

**Result.** The mask is gone: `Meta() compute buffer size = 3251.39 MiB`, `ROCm_Host compute buffer
size = 63.69 MiB` (from 4051.39 / 863.69). The *same* 800 MiB leaves the per-GPU compute buffer
**and** the shared host buffer, so the flip is worth **3 x 800 MiB of VRAM + 800 MiB of host RAM**.
ub1024: 2074.55 -> **1675.33**; ub512: 1188.56 -> **889.54**. The O(n_kv x n_tps) host mask build
per prefill ubatch is gone with it.

**Root cause of the "null-buffer probe chain" — and its correction.** The block was **not** in the
meta/sched backend. Two *input-fill* call sites assumed a buffer that the prune deliberately leaves
unallocated; each aborted at its callee's `GGML_ASSERT(ggml_backend_buffer_is_host(...))`:

1. `llm_graph_input_mem_hybrid::set_input` (llama-graph.cpp) called
   `llama_kv_cache::set_input_kq_mask(inp_attn->self_kq_mask, ...)` unguarded. Backtrace:
   `ggml_backend_buffer_is_host <- llama_kv_cache::set_input_kq_mask <-
   llm_graph_input_mem_hybrid::set_input <- llm_graph_result::set_inputs <-
   llama_context::process_ubatch`. Fixed with the guard its siblings already use
   (`if (self_kq_mask && self_kq_mask->buffer)`, as in `llm_graph_input_attn_kv::set_input`).
2. `llm_graph_input_qsa::set_input` (qwen4exp.cpp) called
   `llama_memory_hybrid_idx::set_input_qsa(cell_blk, ...)` unguarded. The K-store-only policies
   (decode + short context) build no top-k node, so `cell_blk`/`blk_cells`/`blk_pos` are left
   unallocated — and `set_input_qsa` dereferences `cell_blk->data` unconditionally. Only reachable
   with `LLAMA_QSA_SPARSE_FA=0`, which is why the first validation round missed it. Fixed with an
   early return when `cell_blk == nullptr || cell_blk->buffer == nullptr`.

The earlier "an unreachable tensor is not a valid state in this backend" conclusion was an artifact
of *where* the assert surfaced: `ggml_backend_buffer_get_type`'s assert (ggml-backend.cpp:209) was
reached from `ggml_backend_buffer_is_host`, called by the *input plumbing* above — not by the meta
or sched code. No meta/sched/allocator change is needed, every diagnostic has been removed, and
`ggml/src/ggml-backend.cpp` is back to the upstream state. The `get_usage`/`get_type` "tolerances"
explored in session 1 are **not** in the patch. (Method note: the decisive tool was a `backtrace()`
print in `ggml_backend_buffer_get_type` — the assert's own file:line and the thread dump were both
misleading because the aborting worker thread's dump was not in the log.)

**Gate hardening.** The sparse-FA flag is now a shared helper, `qwen4exp_qsa_sparse(cparams)`, used
by both the FA branch and the prune predicate so they cannot disagree: with `LLAMA_QSA_SPARSE_FA=0`
the dense `build_attn_mha` reads the mask and must keep it.

**Validation (final binary, 3x R9700, ctx 204800, ub 2048, q8_0 KV).**

| check | result |
|---|---|
| same-seed text A/B, sparse FA, derived (V=1) vs mask (V=0) | **byte-identical** — 3k and 40k prompts, one binary |
| same-seed text A/B, dense FA (`LLAMA_QSA_SPARSE_FA=0`) | **byte-identical** V=0 vs V=1; both complete |
| MTP probe (Protocol A) | 0.61616 in both modes (unchanged; gate >= 0.45) |
| llama-bench ub2048 (r=3) | pp20480 2509.63 +/- 4.06 (pre-flip 2488.36 +/- 4.99, **+0.9%**); tg256 50.66 +/- 1.38 (pre-flip 50.63 +/- 1.40) |
| reserve | 3251.39 / 63.69 (ub2048); 1675.33 / 33.64 (ub1024); 889.54 / 18.61 (ub512) |

With the mask deleted, identical same-seed text *is* the proof of the FA's derived path: there is no
mask left to read. Sparse and dense legitimately produce different text (different FA kernels give a
different fp accumulation order), and each is internally consistent across the derived/mask A/B.

**Patch state.** `patches/0002-derived-qsa-block-bias.patch` (10 files, +552/-79 over 0001) now
carries the whole L1 state — steps 1-3 + the prune + both guards;
`patches/0003-prune-mask-flip-NOT-APPLIED.patch` is **deleted** (folded in).

**The full ubatch table (compute buffer / GPU, MiB | pp20480 t/s | tg256 t/s).** Pristine = the
upstream fork point; every point is ~half the memory and slightly faster after the prune:

| ub | pristine | + L2 | + steps 1-3 | **+ prune (final)** | pp20480 final (pre-flip) | tg256 final |
|---|---|---|---|---|---|---|
| 2048 | 6690.40 | 4450.40 | 4051.39 | **3251.39** | 2515.35 +/- 5.30 (2488.36 +/- 4.99) | 50.64 +/- 1.37 |
| 1024 | 3346.50 | 2274.35 | 2074.55 | **1675.33** | 2235.03 +/- 2.88 (2209.80 +/- 3.56) | 50.68 +/- 1.42 |
| 512 | 1724.56 | 1188.56 | - | **889.54** | 1712.85 +/- 0.54 (1692.50 +/- 1.80) | 50.67 +/- 1.40 |

Host buffer: 63.69 / 33.64 / 18.61 MiB. Pristine references: ub2048 2461 t/s, ub1024 2201 t/s,
ub512 1691 t/s (all at ub2048's 6690/3347/1725 MiB). **Verdict: BEST** - ub2048 now needs less
memory than *pristine ub1024* (3251.39 vs 3346.50) while being faster than pristine ub2048
(2515 vs 2461), and the prune lifts every ub point by ~1%.

**Next lever.** The score chain's own peak (the ~700 MiB concat in L2; see
`L2-score-chain-findings.md` §6) and the host-side top-k build.
