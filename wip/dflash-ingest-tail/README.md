# wip/dflash-ingest-tail — a DFlash draft whose layers all have a sliding window ingests only the prompt tail

One `git am` file (one commit), made on `v14-097cf33d9-r120` (tree `0fd9827d…`; applied: tree `61d95463…`). It touches
`common/speculative.cpp`, `src/llama-ext.h` and `src/llama-model.cpp`.

- Every layer of the Qwen3.8-27B DFlash2 draft attends over a 2048-position sliding window, and its KV cache is sized
  for the window. `process()` still fed it every prompt row, so a 110K prompt was encoded and written into the draft's
  cache row by row, and all but the last ~2048 rows were dropped again without being read. Each prompt batch also
  waited for the target before the draft could start, which breaks the queue of the target's ubatches.
- When every draft layer has a window, the DFlash impl now takes the prompt end from `set_prompt_end()` and skips the
  prompt rows before the last `n_swa` positions. A batch left with no rows returns before the wait on the target. The
  first generation row resets the threshold, as in the MTP impl (`LLAMA_MTP_INGEST_TAIL`, #143 / #206), so a later
  position shift never skips rows. `LLAMA_DFLASH_INGEST_TAIL=0` feeds every row.
- The window test needs the per-layer pattern, which `llama_model_meta_val_str()` cannot return because it skips array
  values. A new staging call, `llama_model_n_layer_swa()`, sits in `src/llama-ext.h` next to
  `llama_model_dflash_selector_top_k()`. `include/llama.h` is unchanged. The PR description asks whether you want this
  call or something else.
- A draft with a full-attention layer keeps feeding every row. For example, the Qwen3.6-35B-A3B DFlash draft has 5 of
  its 6 layers windowed.

The draft cannot attend to the skipped rows, so the output does not change: the target verifies every drafted token.
Measured on gfx1201 with r120, Qwen3.8-27B UD-Q4_K_XL + DFlash2-Q4_K_M and `-ub 2048`:

- prefill of 8K / 32K / 110K tokens: 6180 / 26945 / 122946 ms -> 6053 / 26600 / 121879 ms;
- the draft's share of the prefill drops by 42 / 59 / 64 %;
- the text is identical in every arm.

Details, and a cross-test together with #210, are in the PR description.

*The patch and this note were written by Claude (AI).*
