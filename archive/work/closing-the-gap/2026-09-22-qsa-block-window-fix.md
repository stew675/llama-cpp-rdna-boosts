# Phase-1 item 3.5 (first fix) — size the QSA block window by the highest stored position (2026-09-22)

**Status:** DONE (WIP).  Fork branch `gap-closing` @ **`9449f3446`**, exported as
[`patches/0005`](patches/0005-gap-closing-WIP-size-the-QSA-block-window-by-the-hig.patch).  This is a
**correctness** fix (the maintainer's priority order puts recall speed + correctness first).

## The bug

The QSA block metadata — `cell_blk` (block id per cell), `blk_cells`, `blk_pos`, the per-block
`bias`, `blk_idx`/`blk_tail`, `cell_vis`/`q_vis`, and the derived-cache pool — was sized from
`idx->get_n_kv()`, which counts **occupied cells**.  But blocks are keyed by **position**: a cache
whose stored positions run ahead of its occupied cells has blocks past that view.  The MTP draft
context never receives the cells an M-RoPE image pins to a single position, so after an image its
highest position leads its cell count by the image's grid size; once generation carried the highest
position past the padded view, the block fill walk ran off the end (assert / corrupt read).

This is the other solution's `b0f31f587`; our tree was at the pre-fix state, so it ported directly.

## The fix

* `src/llama-memory-hybrid-idx.{h,cpp}`: new
  `llama_memory_hybrid_idx_context::qsa_n_kv_window()` =
  `max(get_n_kv(), PAD(highest stored position + 1, 256))`.
* `src/models/qwen4exp.cpp`: the per-`n_blocks` sizing in the graph input's `can_reuse()` and in
  `build_qsa_top_k()` now uses the window; the derived-cache `get_pool()` sizing follows from the
  latter.  The per-cell tensors (`cell_blk`, `cell_vis`) stay `get_n_kv()`-sized.

One deviation from the reference, deliberate: the seq sweep is bounded by the KV stream count
(`idx->get_n_stream() > 1 ? n_stream : LLAMA_MAX_SEQ`), because `seq_to_stream` is
`LLAMA_MAX_SEQ`-sized only in the unified case and `get_cells()` asserts otherwise — the reference's
bare `for s < LLAMA_MAX_SEQ` loop would trip that assert on a multi-stream cache.

## Validation

* **Normal case is unchanged**: for a contiguous cache, `positions == cells` so
  `qsa_n_kv_window() == get_n_kv()`.  `test-logits-width-probe` (qwen4exp IQ4_NL, P=1024) still
  `width_purity=PASS` with row-0 hash **`268e0673300b7a33`** (unchanged).
* **Long-context row**: `llama-bench -p 65536 -n 0 -r 1` runs clean.
  **Important ubatch note (maintainer, 2026-09-22):** at `-b/-ub 8192` this point is right at the
  memory limit and the GPU **oscillates** (memory shortfall) at ~844 t/s; at `-b/-ub 4096` the GPU
  stays pegged at 100 % (no thrashing) and it runs **1093 t/s**.  Use `-ub 4096` for long-context
  smoke runs — the `-ub 8192` number is a memory-pressure artifact, not a kernel result.
* The exact repro (M-RoPE image + MTP shared head + >12k tokens) is **not reproducible here**: our
  build cannot load the shared MTP head the IQ4_NL checkpoint ships (see
  [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md)), so the fix is validated as a
  faithful port + a no-regression check rather than by FAIL -> PASS on the crash.

## The other two item-3.5 fixes (still an audit)

* `40c0b9c38` (maskless only where the qsa3 kernel consumes it) and `14fff4f97` (−1 sentinels) target
  the reference's `tail_idxs` / `compact` / `maskless` design and its `qsa_scalar_visibility()` gate.
  Our delivery's QSA has a **derived-visibility** design (`cell_vis`/`q_vis`, `blk_idx`/`blk_tail`,
  `qwen4exp_derived_vis_enabled()`), so each has to be mapped to our equivalent before it can be
  ported; they remain open in item 3.5.
