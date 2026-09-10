# V3 — derived kq mask: verified predicate + implementation plan

**Status (2026-09-10).**  *Phase 1 (the predicate) is DONE and proven bit-exact on the host.*
*Phases 2a (the op + a CPU reference), 2b (the CUDA MMA kernel) and 2c (the graph plumbing, the backend
probe and the enable) are DONE and validated — see §4.1-§4.4.  V3 is complete: the feature is on by
default (`LLAMA_KQ_MASK_DERIVED=0` forces the packed mask) and measured at -799 MiB/GPU plus
-799 MiB host on the dense and SWA models.  Everything in §1–§4 is measured, not derived.

Read order: this file → `DERIVED-MASK-DESIGN.md` (the general brief) →
`../../beta/block-15-campaign-wins/HANDOVER.md` §3.1 (the campaign plan).

---

## 1. The verified predicate (phase 1)

The packed kq mask is `F16 [n_kv, n_tps, 1, n_stream]` (F32 without FA), written by
`llama_kv_cache::set_input_kq_mask` → `set_input_kq_mask_impl<>` (`src/llama-kv-cache.cpp`, was
~L1555, now ~L1555+208).  Per (stream `s`, token `ii`, cell `j`), with
`seq_id = ubatch->seq_id[i][0]`, `cells = v_cells[seq_to_stream[seq_id]]`, `p1 = ubatch->pos[i]`:

| # | the mask drops the cell when | reduced to |
|---|---|---|
| 1 | `cells.is_empty(j)` | `cell_pos[j] = INT32_MIN` |
| 2 | `!cells.seq_has(j, seq_id)` | `cell_pos[j] = INT32_MIN` |
| 3 | `causal && p0 > p1` | `cell_pos[j] <= hi`, `hi = causal ? p1 : INT32_MAX` |
| 4 | `is_2d && p0 == p1 && p0_ext.is_2d_gt(p1_x, p1_y)` | the ext clause (see below) |
| 5 | `swa && !in_span && is_masked_swa(n_swa, swa_type, p0, p1)` | `cell_pos[j] >= lo` |
| - | else the value is `0` (alibi: `-abs(p0-p1)`) | |

so, with `cell_pos[j] = is_empty(j) || !seq_has(j, seq_id) ? INT32_MIN : cells.pos_get(j)`:

```
visible = cell_pos[j] >= lo && cell_pos[j] <= hi
```

**`lo` is the only nontrivial part, and every SWA variant reduces to a per-token floor:**

| `swa_type` | fill condition | floor |
|---|---|---|
| `LLAMA_SWA_TYPE_STANDARD` | `p1 - p0 >= n_swa` | `lo = p1 - n_swa + 1` |
| `LLAMA_SWA_TYPE_CHUNKED` | `p0 < (p1/n_swa)*n_swa` | `lo = (p1/n_swa)*n_swa` |
| `LLAMA_NON_CAUSAL_TYPE_SWA_FULL` in-span exception (`!causal && p0 >= seq_pos_min[seq]`) | a *union* of two ranges | `lo = min(lo, seq_pos_min[seq_id])` |
| no SWA | — | `lo = 0` |

`lo` is clamped to `>= 0`, which also makes `INT32_MIN` (the drop sentinel) fail `>= lo` without an
extra branch.  `seq_pos_min[seq_id]` is the ubatch-wide minimum position of that sequence, exactly
as the fill computes it.  The `SWA_ONLY` rule
(`if (!causal_attn && non_causal_type == SWA_ONLY) causal_attn = swa_type == NONE`) is applied by
the caller before the predicate, as the fill does.

**The M-RoPE clause** needs no reduction and cannot be folded into `lo`/`hi` (it is conditional on
`p0 == p1`): `visible = visible_core && !(p0 == p1 && cell_ext.is_2d_gt(p1_x, p1_y))`.  Two facts
make it cheap in practice:

* `qwen35` (the 4B and the 27B) is **IMROPE** — `LLM_ARCH_QWEN35 → LLAMA_ROPE_TYPE_IMROPE` in
  `llama_model_rope_type` — so `n_pos_per_embd() == 4`, `ubatch.n_pos == 4` and
  `ubatch.is_pos_2d()` is **true even for text**.  The clause is therefore *live* for the target
  models and must be handled (it is a no-op only because the positions are degenerate).
* for text input all four position sections are equal, so cells are written with
  `ext = (x = y = pos)` (`apply_ubatch`), and `is_2d_gt(p1_x = p1, p1_y = p1)` is false whenever
  `p0 <= p1` — measured: `2d: cells=0`, `dropped_pairs=0` on every text run.

Therefore phase 2 uses **the core predicate + a degeneracy guard** (fall back to the packed mask
when any non-empty cell has `ext.x != ext.y || ext.x != pos`, or any token has
`pos[i+n_tokens] != pos[i] || pos[i+2*n_tokens] != pos[i]`), and adds the clause itself only if a
real M-RoPE workload later needs it (phase 3.3, `DERIVED-MASK-DESIGN.md`).

### 1.1 The proof (phase-1 deliverable)

`patches/0002-DIAGNOSTIC-verify-derived-kq-mask-predicate.patch` (208 lines, `llama-kv-cache.cpp`,
applies clean on top of W1+W2) recomputes the mask from both the core and full (core + ext)
predicates and compares them cell-by-cell with the packed fill.  Gate
`LLAMA_KQ_MASK_DERIVED_VERIFY=1`; it runs **only** in the diagnostic build and is not part of the
eventual patch (it is the oracle for phase 2, like `GGML_QSA_DERIVED_BIAS=2|3` was for L1).

Measured (`logs/v3-*.extract.txt` — every record is `mismatches: core=0 ext=0`):

| run | model | config | what it proves |
|---|---|---|---|
| `v3-27b` | Qwen3.8-27B-Q8_0, dense | ctx 45056, ub 2048, 41.5k-token prompt, `-fa on`, q8_0 KV | 21 real prefill ubatches with `n_tps=2048` and `n_kv` 2304→39680, plus decode `n_tps=1` — the exact case that sets the reserve (`pairs=8912896` at `n_kv=4352`) |
| `v3-gemma` | gemma-4-E4B-it-Q8_0 (**ISWA**) | same shape | **the 3.2 (SWA) case**: 26 fills on the base cache + 26 on the SWA cache (`swa=1, swa_type=1 (STANDARD), n_swa=512`) |
| `v3-noncausal` | gemma-4-E4B (llama-embedding) | `--attention non-causal` | `causal=0 swa=1 n_swa=512` (`hi = INT32_MAX`, window-only) **and** the `SWA_ONLY` rule flipping the base cache back to `causal=1` |
| `v3-fa0` | Qwen3.5-4B | `-fa 0` | the **F32** mask path |

Also covered incidentally: `is_2d=1` with all-degenerate cells (4B/27B), `is_2d=0` (gemma-4 — it is
not M-RoPE), and small verify batches (`n_tps` 2/4/7/8).

**Not yet exercised** (each is a fallback or a straight extension, and each belongs to the phase-2
validation matrix): `n_stream > 1` (multi-sequence / non-unified), the M-RoPE 2-D clause *firing*,
`SWA_FULL` (deepseek-4 has no model here), `CHUNKED` (no model here), alibi (deliberately excluded),
empty/foreign cells from cross-sequence cache reuse, and prompt-cache/checkpoint restore.

Reproduce: rebuild the fork tree, then
`LLAMA_KQ_MASK_DERIVED_VERIFY=1 <bin>/llama-cli ... -v` and grep `verify_derived_kq_mask_impl`
(`-v` is required — both `llama-cli` and `llama-embedding` suppress INFO logs otherwise).

---

## 2. What phase 2 has to build (the plan)

### 2.1 Where the win comes from

The mask is `n_kv × n_tps × 2 B` per GPU in the compute buffer **and** again in `ROCm_Host`
(measured 800 + 800 MiB at ctx 204800 / ub 2048; see `DERIVED-MASK-DESIGN.md` §1.2-1.3).  Phase 2
replaces the tensor with `cell_pos I32[n_kv, n_stream]` (0.8 MiB at ctx 204800), `tok_lo I32[n_tps,
n_stream]` and `tok_hi I32[n_tps, n_stream]` (16 KiB) — the whole 800 + 800 MiB, and it removes the
per-ubatch host fill (`O(n_kv × n_tps)`) in favour of `O(n_kv)`.

### 2.2 Gate: the backend, not the graph

The graph cannot know in advance which backend will run the FA node.  Use the repo's existing
**fused-op probe** mechanism (`llama_context::resolve_fused_ops`, `llama-context.cpp` ~L525, which
reserved a graph and checks that each `add_fused_node` landed on the layer's own device):

1. `cparams.kq_mask_derived` (new, `llama-cparams.h`), default from the env (`LLAMA_KQ_MASK_DERIVED`,
   default 1) and resolved by a new probe entry (`LLM_FUSED_OP_FLASH_ATTN_DERIVED`) whose
   `n_tokens_per_seq` is large enough to select the MMA kernel (> 8, e.g. 64) — the mask win only
   exists for prefill-shaped batches anyway.
2. `build_attn_mha` records `{LLM_FUSED_OP_FLASH_ATTN_DERIVED, cur, il}` when it builds a derived FA
   node.
3. `ggml_cuda_flash_attn_ext_supported()` returns **false** for a derived op unless
   `ggml_cuda_get_best_fattn_kernel()` returns `BEST_FATTN_KERNEL_MMA_F16` — so any other backend (or
   a CUDA shape that would pick the tile/vec kernel) fails the placement check and the flag is
   turned off for the whole context, falling back to the packed mask.

### 2.3 Backend capability: keep the mask *policy* identical

`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:491`) uses `mask` for **policy**, not just for reading:
`gqa_opt_applies = gqa_ratio >= 2 && mask && ...`, and the AMD-WMMA arm is gated on `gqa_opt_applies`.
A null mask therefore silently changes kernel selection (and numerics/perf).  Introduce
`const bool has_mask = mask != nullptr || dst->src[5] != nullptr;` and use `has_mask` in every
*policy* test (`gqa_opt_applies`, `if (mask && mask->ne[2] != 1)`, the `use_gqa_opt` assertions in
`ggml_cuda_flash_attn_ext_mma_f16`), while the *reading* sites keep using `mask`.

### 2.4 The op

Follow the qwen4exp precedent exactly (`ggml_flash_attn_qsa`, patch 0002 of the qwen4exp campaign):

* `ggml/include/ggml.h` + `ggml/src/ggml.c`: a setter in the `add_sinks` style —
  `ggml_flash_attn_ext_add_kq_derived(a, cell_pos, tok_lo, tok_hi)` → `src[5]`, `src[6]`, `src[7]`
  (`src[4]` is `sinks`; `GGML_MAX_SRC == 10`).  The ctor already tolerates `mask == NULL` ✓
  (`if (mask) {...}`), and `max_bias > 0` (alibi) still asserts a mask, which is the fallback rule.
* The CUDA host (`ggml_cuda_flash_attn_ext`) passes the three device pointers down; the meta backend
  already treats null src slots as `SPLIT_AXIS_UNKNOWN`.
* The CPU reference (`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_flash_attn_ext`) must
  `GGML_ABORT` with a clear message when `src[5] != nullptr` — but its `supports_op` must stay
  permissive so the probe's placement check works (the L1/QSA precedent: CPU *accepts*, aborts only
  if actually executed).

### 2.5 The kernel (HIP/CUDA, MMA path only)

`ggml/src/ggml-cuda/fattn-mma-f16.cuh`:

* a POD `kq_derived_t { const int32_t * cell_pos; const int32_t * tok_lo; const int32_t * tok_hi; int32_t n_kv; }`
  passed **by value** down the existing call chain (it ends up in the kernel's parameter space);
* `flash_attn_ext_f16_load_mask` (L480) gains the struct and, at the top, a **runtime** branch that
  computes `tile_mask[j_sram*(nbatch_fa+8) + i] = half(0.0f | -INFINITY)` from
  `k_VKQ_0 + i` and `j_vram` — no new template parameter is needed, because the branch precedes the
  `if constexpr (use_cp_async)` chain and returns; the derived path therefore never uses `cp_async`
  (which cannot synthesize data anyway);
* update the three call sites' guards (`if (ncols2 > 1 || mask_h)` → `|| derived.valid`) at L636,
  L985, L1341 and the mask-add guards at L735, L797, L983, L1020, L1339, plus the "dead column"
  detector at L1027 (it reads `tile_mask[...] <= -1e30f` and must see the same values);
* `flash_attn_ext_f16_process_tile` / `_iter` / the kernel and its fixup kernel just forward the
  struct;
* **decode (`n_tps == 1`) keeps the packed mask** (~400 KiB) — the gate keeps `n_tps > 8`, which also
  means the vec and tile kernels need no change at all (they are only selected below that).

### 2.6 The llama graph plumbing

* A new self-contained input class `llm_graph_input_kq_derived` (owns the three tensors,
  `ggml_set_input`, and fills them in its `set_input` via the memory context) — it needs **no**
  changes to the existing input classes.
* `build_attn_inp_kq_mask` (`llama-graph.cpp:29`) gains an `allow_derived` flag: when true *and* the
  runtime gates pass, it creates the derived input object and returns a **1-byte handle tensor**
  instead of the mask; the returned tensor is recorded in a `llm_graph_context` member
  `std::unordered_map<const ggml_tensor *, ...> kq_derived`, keyed by that handle.
  All 12 call sites keep working; pass `allow_derived = true` **only** at the six standard sites
  (L2832 `attn_kv`, L2939 `attn_k`, L3351 the dsa-iswa swa sub-cache, L3409/L3439 `*_iswa`,
  L3641 hybrid-iswa) and leave the DSA/MSA/DSV4/LID/MLA ones alone — those models consume the mask
  as a *bias* (`deepseek4.cpp`, `minimax-m3.cpp`), not only as an FA input.
* `build_attn_mha` (`llama-graph.cpp:2639`) looks the mask up: if it is a handle, it passes
  `mask = nullptr` and calls `ggml_flash_attn_ext_add_kq_derived(...)`; otherwise it is the unchanged
  path.  This is the **only** place a dense mask reaches the FA op, so the substitution is local.
* `llama_kv_cache_context` gains the two entry points:
  `bool kq_mask_derivable(const llama_ubatch &, bool causal) const` (cache type + the degeneracy
  guard + `n_stream == 1`, `n_seqs_unq == 1`, `n_tps > 8`, `flash_attn`, `!use_alibi`) and
  `void set_input_kq_derived(cell_pos, tok_lo, tok_hi, ubatch, causal) const` (the loop from §1).

### 2.7 Fallbacks (all decided at graph-build time)

alibi · `!flash_attn` · `n_tps <= 8` (decode and small verify batches) · `n_stream > 1` ·
`n_seqs_unq > 1` · any non-degenerate 2-D position (M-RoPE with real x/y) · every cache type other
than the plain `llama_kv_cache` (dsa/msa/dsv4/ml hybrid-idx keep the packed mask; the ISWA base/swa
sub-caches are plain caches and are covered) · every backend whose best FA kernel is not the MMA path
(via the §2.2 probe).

### 2.8 Validation bar (phase 2)

Same-seed generated text **byte-identical**, derived vs packed, in one binary that flips the gate:
Qwen3.5-4B, Qwen3.8-27B, gemma-4-E4B (SWA) and gemma-4-31B, plus qwen4exp as a regression control;
the reserve matrix (expect the -800/-800 MiB at ctx 204800 / ub 2048; ub 1024 to confirm it is
really `n_kv × n_tps`); `tools/mtp-ab.sh` ≥ 0.45; bench parity (`tools/ub-sweep.sh`, the L1 flip
measured +1.1 % because the host fill disappeared); `test-backend-ops FLASH_ATTN_EXT` on CPU + ROCm0
with a derived case; and a **negative control** (`LLAMA_KQ_MASK_DERIVED=0`) plus the `LLAMA_KQ_MASK_DERIVED_VERIFY=1`
oracle on every model in the matrix.

---

## 3. Hazards (the L1 lessons, restated for this change)

* **One predicate, two places.** The graph builder's capability check and the prune/`can_reuse` sites
  must consult the *same* helper — the L1 bug was `LLAMA_QSA_SPARSE_FA=0` reading a mask that the
  prune had already removed.  Here the mask's existence and the derived tensors' existence must be
  decided by one function.
* **Every input fill must tolerate `buffer == nullptr`** — an unreachable input is simply not
  allocated by the gallocr.  The new `llm_graph_input_kq_derived::set_input` is itself such a site.
* **Keep the mask's *policy* meaning in the CUDA launcher** (§2.3) or kernel selection changes
  silently.
* Never let the derived form reach a mask *consumer* other than the FA op (§2.6 gating).
* The diagnostic (`LLAMA_KQ_MASK_DERIVED_VERIFY`) and the `GGML_QSA_DERIVED_BIAS=2|3` modes must be
  **stripped** before the patch is packaged.

---

## 4. Phase 2 progress

### 4.1 Done — phase 2a: the op + a real CPU reference (2026-09-10)

`patches/0003-phase-2a-engine-derived-kq-mask-op-and-cpu-reference.patch` (3 files, +69/-4, applies on
top of W1+W2+the phase-1 diagnostic; fork tree, coherence **byte-identical** vs the pre-phase-2 build
with the slice inert):

* `ggml.h` + `ggml.c`: `ggml_flash_attn_ext_add_kq_derived(a, cell_pos, tok_lo, tok_hi)` -> `src[5..7]`,
  with the shape asserts (`cell_pos->ne[0] == K->ne[1]`, `tok_lo/hi->ne[0] == Q->ne[1]`).
* `ggml-cpu/ops.cpp`: the derived form is **implemented, not aborted** — helper
  `kq_derived_mask_value()` in both FA compute paths (`_one_chunk` at the `mv` read, `tiled` at the
  `mask32` fill + a NULL-mask-tolerant `mp_row`).  That makes the CPU the reference oracle for
  `test-backend-ops FLASH_ATTN_EXT`, so the CUDA kernel can be validated without a full model run.

### 4.2 Done — phase 2b: the CUDA MMA kernel + the shared FA launcher (2026-09-10)

`patches/0004-phase-2b-cuda-mma-derived-kq-mask.patch` (6 files, +158/-35; `git apply --check` CLEAN on
the W1+W2+diag+2a base).

**The one open question is answered, and it changed the shape of the change for the better.**
`fattn_kernel_t` (`fattn-common.cuh`) is a *single concrete function-pointer type with a fixed
parameter list*, and `launch_fattn` is shared by the MMA, TILE and VEC kernels (all three are assigned
to `fattn_kernel_t` and launched through the same `ggml_cuda_kernel_launch(fattn_kernel, ...)` call;
the QSA op has its own launcher).  So the three pointers are threaded **once** through the shared
typedef and the one launch-argument list instead of through the MMA kernel alone: TILE and VEC accept
and ignore them (they are only selected for `n_tps <= 8`, which keeps the packed mask).  That is the
smaller and safer change — one ABI, checked by the compiler, no mma-specific launch tail.

Files:

* `fattn-common.cuh`: `fattn_kernel_t` gains `cell_pos`/`tok_lo`/`tok_hi`; `launch_fattn` picks them up
  from `dst->src[5..7]` and appends them to the launch.
* `fattn-vec.cuh`, `fattn-tile.cuh`: the three parameters, deliberately unnamed/unused.
* `fattn-mma-f16.cuh`: `struct kq_derived_t`; a **runtime** branch at the top of
  `flash_attn_ext_f16_load_mask` filling `tile_mask[j_sram*(nbatch_fa+8) + i]` from
  `cell_pos[k_VKQ_0 + i] >= tok_lo[j_vram] && <= tok_hi[j_vram]`, mirroring the packed path's
  out-of-bounds convention exactly (`(oob_check && i >= i_sup) ? half(0.0f) : ...`, i.e. `i_sup` is
  still honoured where the packed path uses it) so the dead-column detector sees identical values;
  the six `if (ncols2 > 1 || mask_h)` gates widened with `|| derived.cell_pos != nullptr`; the struct
  threaded through `iter` (4 call sites) and `process_tile` (3) up to the kernel (L1798), which takes
  the three raw pointers and builds the POD.  It precedes the `if constexpr (use_cp_async)` chain and
  returns — the derived path cannot use cp_async, and nothing else depends on the mask's cp_async group
  (every wait in the file is `cp_async_wait_all()`, verified).
* `fattn.cu`: `ggml_cuda_flash_attn_ext_has_mask()` = `src[3] || src[5]`, used by every *policy* test
  (`gqa_opt_applies` and the four `use_gqa_opt` sites) while the *reads* keep using `src[3]`;
  `ggml_cuda_flash_attn_ext_supported()` returns false for a derived op unless the best kernel is
  `BEST_FATTN_KERNEL_MMA_F16`.
* `tests/test-backend-ops.cpp`: six derived `FLASH_ATTN_EXT` cases (no mask tensor; `cell_pos` with
  every 16th cell = `INT32_MIN`; a per-token sliding window `[tok_lo, tok_hi]`), spanning F16 and Q8_0
  K/V, DKQ 64/128/192/256, a K/V padded to the tile size and a ragged one.

Validation of 2b **on its own**, before any graph change:

| check | result |
|---|---|
| the six new derived cases vs the CPU reference | **6/6 on CPU**, **3/3 on ROCm0** — the other three select the VEC/TILE kernel on this hardware and are therefore correctly reported `not supported` (that is the `supports_op` fallback working, and it is itself the evidence for the rule) |
| whole `FLASH_ATTN_EXT` suite on ROCm0 (+ VIEW/CONT/CPY/DUP/CONCAT) | **5104/5104 passed** |
| coherence, 4B, one binary, derived code in place but no caller | **byte-identical** to the pre-phase-2 build |
| the phase-1 host oracle, re-run | 7/7 records `mismatches: core=0 ext=0` |

### 4.3 Done — phase 2c: graph plumbing + backend probe + enable (2026-09-10)

`patches/0005-phase-2c-graph-plumbing-probe-enable.patch` (6 files, +487/-45; `git apply --check` CLEAN
on the W1+W2+diag+2a+2b base): `src/llama-cparams.h`, `src/llama-graph.h`, `src/llama-graph.cpp`,
`src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-context.cpp`.

**The one design change from §2.6, and it is a safety win: the packed mask tensor is still created.**

`build_attn_inp_kq_mask` (now a `llm_graph_context` method, `allow_derived` flag) still creates the
`[n_kv, n_tps, 1, n_stream]` mask exactly as before, *and* registers the compact state in
`llm_graph_context::kq_derived` (keyed by that mask tensor) with a new `llm_graph_input_kq_derived`
input object (cell_pos I32, tok_lo/tok_hi I32) added to the graph result.  `build_attn_mha` looks the
mask up: when it has an entry, the FA node is built with `mask = nullptr` plus
`ggml_flash_attn_ext_add_kq_derived()` and the fused node is registered as
`LLM_FUSED_OP_FLASH_ATTN_DERIVED`.

The mask tensor therefore has *no consumer* on the derived path, and the gallocr leaves it
unallocated - which is the entire memory win (no VRAM, no host copy) - while the already existing
`if (tensor->buffer)` guards in the two mask fills skip the per-ubatch fill on their own.

The consequence is that **no mask consumer can ever be mis-served**: deepseek4's bias `ggml_concat`,
minimax-m3's `msa_kqm`, qwen4exp's indexer, any future model - if anything else reads the mask, the
mask is materialized and filled as before, and the FA node still gets the (value-identical) derived
form.  There is no model-level allowlist to keep in sync; `allow_derived` only decides *where* the
attempt is made, and the phase-1 oracle decides *whether* the batch qualifies.

The gates, all of which must hold:

| level | gate |
|---|---|
| context | `cparams.kq_mask_derived` (`LLAMA_KQ_MASK_DERIVED` env, default 1; only an explicit 0 forces the packed mask) |
| graph | `allow_derived` at the mask-build site, `cparams.flash_attn`, `n_stream == 1`, `n_kv % 256 == 0` (must match `FATTN_KQ_STRIDE`, or the derived op would have no kernel and the node would move to the CPU) |
| batch | `llama_kv_cache::kq_mask_derivable`: no alibi, `n_seqs_unq == 1`, `n_tokens > 8`, and for 2-D batches every token position and every own cell degenerate (`x == y == pos`) - where the M-RoPE ext clause is a proven no-op |
| backend | the new `LLM_FUSED_OP_FLASH_ATTN_DERIVED` probe (`resolve_fused_ops`): a 64-token *single-sequence* probe graph must contain the derived node, on the layer's own device, which must be a GPU/IGPU/META device that implements the derived form (`ggml_backend_dev_implements_kq_derived`: CUDA/ROCm registry name; a META device is accepted because its `supports_op` forwards to its sub-devices, which is how it got the node) - and `ggml_cuda_flash_attn_ext_supported` still rejects the op unless the best kernel is `BEST_FATTN_KERNEL_MMA_F16` |

**Crash found and fixed during validation** (only reproducible under `llama-bench`, which keeps
`n_tokens` constant so the reuse check actually reaches the new input): `params.mctx` is the *memory*
context, and in this fork most models (including the 4B) run on `llama_memory_hybrid`, so
`static_cast<const llama_kv_cache_context *>` on it is a type pun and `llm_graph_input_kq_derived::
can_reuse` dereferenced a garbage `kv`.  The existing inputs get away with it because the hybrid
input class does its own mask check via `mctx->get_attn()`.  Fixed with `dynamic_cast` + reject the
reuse when it is not a kv cache context (reuse is harmless to lose: the derived form only exists for
prefill-shaped batches, where `n_kv` changes on every ubatch anyway).

**Kernel-selection coupling**: the derived op needs the MMA kernel, and on RDNA4 with a 256-wide head
that kernel needs the WMMA-256 path (`GGML_CUDA_FA_WMMA_256` unset/default; `=0` caps WMMA at head
128).  Measured: with `GGML_CUDA_FA_WMMA_256=0` the probe correctly reports "assigned to device CPU"
and disables the feature; with the default env it is enabled on the 4B, the 27B and both gemmas.
The many campaign commands that set `=0` therefore do **not** exercise the derived path.

### 4.4 Validation record (2026-09-10, ctx 204800, ub 2048, `-ctk/-ctv q8_0`, one binary flipping `LLAMA_KQ_MASK_DERIVED`)

Reserve matrix (compute buffer per GPU / `ROCm_Host`):

| model | packed | derived | delta |
|---|---|---|---|
| Qwen3.5-4B-Q8_0, 1 GPU | 1800.33 / 840.34 MiB | **1001.13 / 41.13** | **-799.20 / -799.21** |
| Qwen3.8-27B-Q8_0, 3 GPU (Meta) | 1920.33 / 880.34 | **1121.13 / 81.13** | **-799.20 / -799.21** |
| gemma-4-E4B-it-Q8_0 (ISWA), 1 GPU | 1887.35 / 935.37 | **1078.17 / 126.19** | **-809.18 / -809.18** |
| gemma-4-31B-qat-Q4_K_XL (ISWA), 2 GPU | 2753.35 / 897.36 | **1942.18 / 86.18** | **-811.17 / -811.18** |
| qwen4exp (QSA) - regression control | 3251.39 / 63.69 | 3251.39 / 63.69 | **0** (probe: "not used in the probe graph") |

The ISWA models shed **both** masks (base ~800 MiB + SWA ~9-11 MiB).  ubatch scaling on the 4B is
exactly `n_kv x n_tps x 2 B` per copy: ub 2048 -799.20/-799.21, ub 1024 -399.21/-399.21,
ub 512 -199.21/-199.21.

| check | result |
|---|---|
| same-seed generated text, derived vs packed, **identical binary** | **byte-identical** on the 4B (3k prompt), the 27B (40k prompt, 3-GPU Meta), gemma-4-E4B (3k **and** 40k prompt, ISWA) |
| phase-1 host oracle re-run with the new plumbing | 6/6 records `mismatches: core=0 ext=0` |
| MTP gate (`--spec-type draft-mtp`, 27B, ctx 32768, 3-GPU, draft depth 3) | acceptance **0.76744 (66/86, mean len 3.28) identical** on both gates - no layout drift; prompt 1123.6 -> 1229.2 t/s, generation 78.8 -> 78.3 t/s |
| bench parity, 4B pp20480 at ub 2048 (interleaved 5x) | packed 6040.6 t/s vs derived 5973.1 t/s = **-1.1 %** |
| bench parity, 4B tg256 | 99.14 -> 98.48 t/s = -0.7 % (ub 1024: pp +0.2 %, tg -0.2 %) |
| negative control | every number above is an A/B inside one binary with `LLAMA_KQ_MASK_DERIVED=0` |

**The -1 % prefill is real and consistent** (both gates have <0.5 % spread over 5 interleaved runs) but
it cannot be the mask traffic: the derived staging reads 4 B/cell instead of 2 B/cell, which is
~840 MB of extra reads over a pp20480 prefill, i.e. ~1 ms of ~3 s.  The remaining candidates are the
kernel's register/occupancy change from the runtime branch in `flash_attn_ext_f16_load_mask` and the
compute-buffer layout change (the reserve is 85 MiB smaller at bench ctx), the same class of effect
that produced the MTP probe drift.  Not investigated further: the win is 799 MiB/GPU + 799 MiB host,
and against the delivered ub 1024 baseline the derived ub 2048 is **equal speed for 300 MiB less per
GPU**.

### 4.5 Not exercised (each is a fallback or a straight extension)

* prompt-cache / state (checkpoint) save+restore with the derived path active - the derived inputs are
  not part of the KV state, but an end-to-end round trip was not run;
* `--parallel > 1` / multi-sequence batches (`kq_mask_derivable` rejects them, so the packed mask is
  used - expected to be a no-op, not measured);
* the non-causal paths (`llama-embedding --attention non-causal`, `SWA_ONLY`): covered by the phase-1
  oracle, but not re-validated end to end with the derived path active;
* `SWA_FULL` / `CHUNKED` (no model here), the M-RoPE 2-D clause *firing* (the guard rejects it), alibi
  (excluded by the predicate);
* every non-CUDA/HIP backend (rejected by the probe - and that is the intent: Vulkan/Metal would
  silently ignore the derived sources), CPU-only runs (rejected), and NVIDIA CUDA (untested; the
  sparse-mask path is unreachable there with a null mask, but no NVIDIA hardware was available);
* `GGML_CUDA_FA_WMMA_256=0` (the derived path is then unavailable for head > 128 - by design, see
  §4.3);
* the root cause of the -1 % prefill delta (see §4.4).

**State after this session**: fork tree = W1 + W2 + block-15 phase 1 + 2a + 2b + 2c = 21 modified
files, builds clean, feature **on by default** and validated.  Five separable patches under
`patches/` (0002 the phase-1 oracle, 0003 phase-2a, 0004 phase-2b, 0005 phase-2c; 0001 is the W4
ggml-alloc fix).
