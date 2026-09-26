# Phase-2 item 9 audit — sparse QSA decode + incremental indexer (`d67d58836`) vs our tree

**Status:** audit complete.  **Recommendation: do not port as-is.**  Our tree already carries both
halves (functionally), so the item is a *hold/repay* — and one concrete piece (the MTP-draft sparse
attention, below) is the only uncovered part, for a targeted A/B rather than a port.

Reference: `~/pwilkin-llama-cpp` @ `b0f31f587`, commit `d67d58836` ("hip: enable sparse QSA decode and
incremental indexer state"), 11 files / +664/-120, plus its
`docs/development/qwen4exp-decode-indexer.md`.  Ours: `~/llama.cpp` branch `gap-closing-r13`
(`575c4c091`).  This follows the handoff's explicit instruction: *"Audit first … 1:1 audit vs the
reference's design before porting (overlapping, not equivalent)."*

## What the reference commit actually contains

1. **Sparse selected-cell decode kernels** — `qsa-decode.cuh` (SIMT) + `qsa-decode-wmma.cuh` (WMMA),
   dispatched from `ggml_cuda_flash_attn_ext()` before the existing QSA path; they read the selected
   F16 K/V cells named by the top-k `ids` directly.  The WMMA arm needs 12 query heads per KV head and
   aligned keys; otherwise SIMT.
2. **Incremental indexer state** — `src/qsa-prefix-state.h` + `llama-memory-hybrid-idx.*`: a per-layer
   `qsa_keys[il] = [indexer_head_size, (kv_size+3)/4]` F32 cache (i.e. **one pooled key per
   4-cell block**), a per-cell prefix tracker (`cells`/`positions`), incremental group updates
   (`qsa_apply`/`qsa_fill_updates`), invalidation + reconstruction across lifecycle ops
   (`qsa_invalidate`/`qsa_recover`), and a `qsa_prefix_matches`/`qsa_fast` reuse predicate.
3. **MTP-draft sparse selection** — in `graph_mtp`, the top-k build was gated `n_tokens >= 128`; the
   commit adds a `sparse_decode` arm for `n_tokens <= 8` on HIP with one stream, flash attention + KQV
   offload, no ALiBi and no soft cap.
4. A failed-compute invalidation hook in `llama-context.cpp` and small `fattn.cu`/`qsa.cu` wiring.

## 1:1 mapping

| reference piece | our tree | verdict |
|---|---|---|
| `qsa-decode.cuh` (SIMT selected-cell) | `ggml/src/ggml-cuda/fattn-qsa.cu::flash_attn_qsa` (`ggml_flash_attn_qsa`, `LLAMA_QSA_SPARSE_FA` default ON) — warp-per-head, tiles the top-k list, sliced/combined | **present, arguably more general** (we also dequantize q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl staging into LDS, and support the derived-visibility + `cell_vis`/`q_vis` path) |
| `qsa-decode-wmma.cuh` (WMMA selected-cell) | none — our decode is SIMT only (`qsa3` is the *prefill* packed-block WMMA path) | **not present**; a micro-optimisation candidate |
| `qsa_keys[il]` block-key cache + `qsa_apply`/`qsa_fill_updates`/`qsa_commit`/`qsa_recover` | `llama_memory_hybrid_idx::pool_layers` (`ggml_tensor * pool = F32 [idx_dim, n_blocks, n_stream]`) + `pool_wm` watermark + `pool_invalidate_all` + the `rng` per-step fill-range leaf in `set_input_qsa` (`GGML_CUDA_QSA_INDEXER_CACHE` default ON, `src/models/qwen4exp.cpp:1394`) | **present** — same block-key cache, ours watermark-driven per stream, the reference's cell-prefix-driven single-seq; ours is *simpler* and already the default |
| `graph_mtp` sparse decode (`n_tokens <= 8`) | `graph_mtp` calls `build_attn(inp_attn, …)` **dense** and has no `build_qsa_top_k`/`build_qsa_store_k`; the comment in `build_layer_attn` says *"the MTP draft attends dense"* | **not present** — the one genuine gap |
| `ggml_backend_sched_graph_compute_async` failure → `qsa_invalidate` | none | robustness delta only |

## Reading

* The reference's measured table (`docs/development/qwen4exp-decode-indexer.md`) is **same-build
  sparse-recompute vs sparse-incremental**, i.e. it prices the *incremental indexer* — which we
  already have (`pool_wm`).  The "previous shipping dense path" delta (34.47 → 35.41, 36.87 → 38.43)
  is dense-vs-sparse *target* decode — also already ours.  So the headline +11–20 % is mostly a gap we
  closed on 2026-09-07 with the derived block-vector cache.
* Therefore item 9 is **not** the large decode win the one-line summary in the older body implies for
  our tree.  This matches the handoff's own caveat (*"a hold/repay item: our plain decode is already
  ahead … land it only if it holds that lead"*) and the session-6 MTP finding (fixed-depth speedup at
  parity).
* The **MTP-draft sparse attention** is the only substantive uncovered behaviour.  It is worth an
  experiment because a draft step currently attends the whole KV cache densely; at depth that is the
  draft's dominant attention cost.  Two cautions before touching it: (a) the draft and target must
  keep the same QSA machinery reduction order — the historical "cause 3" (`plain != draft-mtp` text)
  lived in exactly this split, and the purity contract (`plain == draft-mtp` greedy text, byte-
  identical) is the gate; (b) the reference's own guard forbids ALiBi/soft-cap and needs one stream +
  KQV offload, which is where our `graph_mtp` already is.
* The reference's WMMA decode variant is only worth pursuing if a width sweep shows our SIMT
  `flash_attn_qsa` losing at the 12-head/128-dim geometry — it has not been measured to lose.

## Recommendation

1. **No port of `d67d58836`.**  Its two headline mechanisms are already in the tree, and importing a
   second indexer-cache design would duplicate `pool_layers` for no measured gain.
2. If decode is revisited, the experiment is a **targeted MTP-draft sparse A/B**: reuse
   `build_qsa_store_k`/`build_qsa_top_k` + the existing `flash_attn_qsa` in `graph_mtp` behind the
   reference's guard (`n_tokens <= 8`, one stream, flash attention, no ALiBi/soft-cap), gated by
   `plain == draft-mtp` byte-identical text, the width probe, and the adaptive-MTP acceptance
   methodology (`benchmarks/mtp-adaptive-methodology.md`).  That is the item to schedule, not a kernel
   port.
3. The prefill half of the same dispatch is already covered by the trunk's sparse path, so no separate
   work there.

## Files

Read-only audit; no code changed.  Reference commit and doc are the sources above; our mapping lives in
`ggml/src/ggml-cuda/fattn-qsa.cu`, `src/models/qwen4exp.cpp` (`build_layer_attn`, `graph_mtp`,
`GGML_CUDA_QSA_INDEXER_CACHE`) and `src/llama-memory-hybrid-idx.{h,cpp}` (`pool_layers`, `set_input_qsa`).
