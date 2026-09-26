# Fix — the MMB HC16 F32-elision corrupts MTP (per-context activation state)

**Status:** fixed, gated, committed.  Closes **Rule 0 / open item 1** of
[`closing-the-gap.md`](closing-the-gap.md) (the campaign's only blocker).  Fork `~/llama.cpp`, branch
`gap-closing-r13`, commit **`88530f05e`**; exported as
[`patches/0023-mmb-hc16-per-context.patch`](patches/0023-mmb-hc16-per-context.patch).

## Symptom

With the campaign default (MMB on, HC16 on):

* **128K MTP is nondeterministic** — the same command produced five different greedy texts in five
  runs (`9230a59d9116`, `770e770ae7d6`, `36b818fb2985`, `7217d66ed477`, `6f3a284c98cc`), so
  `plain == draft-mtp` failed.
* At **40K** the draft (dense *or* sparse) diverged from plain, and over repeats the divergence was
  nondeterministic (41 hashes on some arms).
* The **plain** path was deterministic, and `GGML_CUDA_MMB_HC16=0` / `GGML_CUDA_MMB=0` made every arm
  deterministic *and* pure (`plain == draft == 8285d12d40ca` at 40K).  So the defect is specific to
  HC16's cross-context state.

## Root cause

HC16 keeps two pieces of file-scope state in `ggml/src/ggml-cuda/mmb.cu`:

* the per-graph BF16 activation cache (`g_mmb_cache`) and the pinned producer slots
  (`g_mmb_slots`/`g_mmb_slot_cap`), and
* the `g_mmb_bf16_only` / `g_mmb_bf16_copy` / `g_mmb_bf16_slot` marks.

Both were **shared by every CUDA backend context in the process**.  The MTP path is two
`llama_context`s (target + draft head) with separate schedulers, and they interleave
optimize/compute calls on the same GPU:

1. **Cache ownership.**  `ggml_cuda_mmb_begin_graph()` (called at the start of *every*
   `ggml_backend_cuda_graph_compute`) cleared the whole global cache and reset the slots.  When the
   draft computed, it freed the target's cache buffers.  A target producer that had been marked
   `bf16_only` (its F32 output elided) then had no cache entry left for its consumer, which fell
   back to re-converting the never-written F32 buffer — garbage, and allocator-layout dependent,
   hence the nondeterminism.
2. **Mark leakage.**  The marks were global, so a context could elide a producer based on another
   context's marks.  The lifetime key (`g_mmb_marks_first_split = cgraph->nodes[0]`, cleared when
   the key repeated or after any compute) was also global and racy.

A third, independent hole was found while fixing it: the step-3 marking pass ran **per split**, but
`all_bf16_consumers()` returned `true` when it found *no* consumer in the split.  A tensor whose only
consumer lived in the other split was therefore marked `bf16_only`, its producer elided the F32, and
the split-1 consumer (whose compute had already cleared the activation cache) read garbage.  This was
the deterministic half of the 40K sparse-draft divergence.

## Fix

All in `ggml/src/ggml-cuda/mmb.{cu,cuh}` plus one scheduler plumb:

* **Per-context state.**  The cache, the producer slots and the marks/lifetime moved into
  `mmb_ctx_state`, stored in `std::unordered_map<const ggml_backend_cuda_context *, mmb_ctx_state>`.
  A file-scope "active context" pointer (`ggml_cuda_mmb_set_active_ctx`) is set at the start of
  `ggml_backend_cuda_graph_optimize` and `ggml_backend_cuda_graph_compute`, so the op implementations
  (which run synchronously inside a compute) select the right state without threading `ctx` through
  every call site.
* **Explicit mark lifetime.**  `ggml_cuda_mmb_optimize_begin(graph_key)` clears and rebuilds the
  marks when the context starts a new graph (after-compute flag, or the same first-node seen again)
  and `ggml_cuda_mmb_compute_done()` sets the after-compute flag at the end of every compute.  The
  old global `g_mmb_marks_after_compute` / `g_mmb_marks_first_split` key hack is gone.
* **Whole-graph consumer scan.**  `ggml_backend_graph_optimize_params` gained
  `full_graph` (set by `ggml_backend_sched_split_graph` to the graph being scheduled).  The step-3
  pass now calls `classify_consumers()` over the whole graph and returns
  `0 = no consumer`, `1 = every consumer is an in-split BF16 reader`, `2 = otherwise` (a non-BF16
  consumer *or any consumer in another split*).  Only `1` may elide (`bf16_only`); `2` marks
  `bf16_copy` (F32 kept); `0` is left alone.  The old `all_bf16_consumers()``true`-on-empty result is
  gone.
* `ggml_cuda_mmb_release_all()` frees every context's cache/slots and clears the map.

The eval-callback stand-down (`has_eval_callback`, `patches/0019`) is unchanged.

## Gates (gfx1151, qwen4exp IQ4_NL + Q4_K_XL MTP sidecar, f16 KV, `--ctx-checkpoints 0`)

| gate | result |
|---|---|
| **128K MTP `n_max 1`** (documented race repro, 5 runs) | **one hash every run `770e770ae7d6`**, == plain == HC16=0 |
| **40K** plain / dense `n1`,`n3` / sparse `n1`,`n3`,`n5` | all **`8285d12d40ca`**, deterministic over repeats |
| **8192** plain / dense `n1` / sparse `n1`,`n3` | all **`3553e76d3a9e`** |
| **Width probe** (`test-logits-width-probe` P=512 ub=512, and P=32768 with the local probe extension) | **PASS (worst maxdiff 0)** |
| **`llama-imatrix`** NanBeige4.2-3B-BF16 `-c 512 -b 512 --chunks 4` | clean (no non-finite), PPL 25.0705; imatrix file **byte-identical to HC16=0** (`49828e7c1aa8dc7d`) |
| **Oracles** | `FLASH_ATTN_QSA` 2/2, `GATED_DELTA_NET` 2/2, `INDEXER_TOPK` 2/2 |

Perf (`-c 40000 -b/-ub 2048 -n 200 --spec-draft-n-max 3`, `Generation:` t/s): plain **28.4**,
dense MTP **33.4**, sparse MTP **29.6**; dense MTP with HC16 on and off are identical (33.4), i.e.
the fix keeps the HC16 win and does not tax decode.

## Notes / follow-ups

* Like `patches/0019` (the eval-callback fix), this is an HC16 **correctness** fix and belongs in the
  `beta/mmb-general` HC16 feature when it is cut into a delivery block; it is staged here as a
  gap-closing patch because the campaign branch is `r13 + beta/mmb-general + gap-closing`.
* The `full_graph` consumer scan is required for correctness even single-context: every graph here is
  **2-split** (a tiny CPU split for the token embedding + the ROCm split), and the activation cache is
  cleared between the split computes.
* The sparse MTP draft is now pure with HC16 on, so it is promotion-eligible (open item 2), but at
  40K its decode is **29.6 vs 33.4 t/s** for the dense draft — its win is prefill at depth
  (`+6.9 %` at 150K).  Any default-on promotion should keep/raise the depth gate rather than enable
  the sparse decode arm shallow.
* The CPU observation ("cores light up during sparse decode") is **not** the sparse attention: the
  only CPU-scheduled nodes in the target are `model.input_embed` (`[0,2)`) and in the draft
  `mtp_tok_embd-48` (`[0,9)`); every QSA op (`top_k_rows`, the score, the store) is on ROCm.  The
  burst is the standard token-embedding gather, present in dense and sparse alike.

## Files

`ggml/src/ggml-cuda/mmb.cu`, `ggml/src/ggml-cuda/mmb.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu`,
`ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp`.
