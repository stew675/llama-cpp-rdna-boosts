# 2026-09-12 — `-Wshadow` audit for `src/` (TODO item 15)

Motivation: the Block-15 dense-arm blocker (2026-09-11 (11), `GREEDY-PURITY.md` §23) was a one-token
shadowing bug — `ggml_tensor * kq_mask_top_k = ...` declared inside a block that already had an outer
declaration of the same name — which made a whole mask chain dead code.  The code compiles, the chain is
still built, and the only detector was the node dump.  `-Wshadow` reports the class outright; it is not
enabled anywhere in the build.

## Measurement (the delivery, gfx1151, 2026-09-12)

Method: replay the tree's own host compile commands for the `llama` target (`src/`, 186 TUs) from
`build-rocm/compile_commands.json`, replacing `-o <obj>` with `-fsyntax-only` and appending `-Wshadow`
(no rebuild, no tree change).

**Result: 128 `-Wshadow` warnings in 27 files.**

| class | count |
|---|---|
| `declaration shadows a local variable` — the class that caused the Block-15 bug | **46** |
| `declaration shadows a field of '…'` (constructor params and locals shadowing members) | 82 |

Top files: `llama-graph.cpp` 32, `llama-batch.cpp` 14, `llama-context.cpp` 12, `llama-model.cpp` 12,
`llama-sampler.cpp` 10, `llama-kv-cache.cpp` 9, `models/deci.cpp` 5, `llama-vocab.cpp` 4,
`llama-mmap.cpp` 3, `models/mimo2.cpp` 3, then a long tail of 1-2 each.  **`src/models/qwen4exp.cpp`
is clean** — the delivery does not carry the shadowing bug (it was beta-Block-15 code).

The 46 "shadows a local variable" entries, delivery line numbers (full list captured from the audit):
src/llama-batch.cpp:111:36: warning: declaration shadows a local variable [-Wshadow]
src/llama-batch.cpp:889:30: warning: declaration shadows a local variable [-Wshadow]
src/llama-context.cpp:1576:24: warning: declaration shadows a local variable [-Wshadow]
src/llama-context.cpp:1979:32: warning: declaration shadows a local variable [-Wshadow]
src/llama-grammar.cpp:640:34: warning: declaration shadows a local variable [-Wshadow]
src/llama-kv-cache.cpp:956:22: warning: declaration shadows a local variable [-Wshadow]
src/llama-kv-cache.cpp:1154:26: warning: declaration shadows a local variable [-Wshadow]
src/llama-kv-cache.cpp:2229:31: warning: declaration shadows a local variable [-Wshadow]
src/llama-memory-recurrent.cpp:915:27: warning: declaration shadows a local variable [-Wshadow]
src/llama-model-loader.cpp:99:14: warning: declaration shadows a local variable [-Wshadow]
src/llama-model-loader.cpp:1576:20: warning: declaration shadows a local variable [-Wshadow]
src/llama-quant.cpp:800:46: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:674:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:688:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:706:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:716:23: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:726:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:746:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:780:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:800:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-sampler.cpp:814:17: warning: declaration shadows a local variable [-Wshadow]
src/llama-vocab.cpp:43:18: warning: declaration shadows a local variable [-Wshadow]
src/models/bert.cpp:115:23: warning: declaration shadows a local variable [-Wshadow]
src/models/deci.cpp:30:23: warning: declaration shadows a local variable [-Wshadow]
src/models/deci.cpp:31:23: warning: declaration shadows a local variable [-Wshadow]
src/models/deci.cpp:32:23: warning: declaration shadows a local variable [-Wshadow]
src/models/deci.cpp:33:23: warning: declaration shadows a local variable [-Wshadow]
src/models/deci.cpp:34:23: warning: declaration shadows a local variable [-Wshadow]
src/models/eurobert.cpp:55:23: warning: declaration shadows a local variable [-Wshadow]
src/models/gemma3n.cpp:138:23: warning: declaration shadows a local variable [-Wshadow]
src/models/gemma4-assistant.cpp:50:23: warning: declaration shadows a local variable [-Wshadow]
src/models/gemma4-assistant.cpp:52:23: warning: declaration shadows a local variable [-Wshadow]
src/models/gemma4.cpp:69:23: warning: declaration shadows a local variable [-Wshadow]
src/models/jamba.cpp:48:23: warning: declaration shadows a local variable [-Wshadow]
src/models/jamba.cpp:49:23: warning: declaration shadows a local variable [-Wshadow]
src/models/kimi-k3.cpp:109:27: warning: declaration shadows a local variable [-Wshadow]
src/models/kimi-k3.cpp:110:27: warning: declaration shadows a local variable [-Wshadow]
src/models/kimi-linear.cpp:302:31: warning: declaration shadows a local variable [-Wshadow]
src/models/mimo2.cpp:44:18: warning: declaration shadows a local variable [-Wshadow]
src/models/mimo2.cpp:45:18: warning: declaration shadows a local variable [-Wshadow]
src/models/mimo2.cpp:46:18: warning: declaration shadows a local variable [-Wshadow]
src/models/neo-bert.cpp:61:23: warning: declaration shadows a local variable [-Wshadow]
src/models/openelm.cpp:26:23: warning: declaration shadows a local variable [-Wshadow]
src/models/openelm.cpp:28:23: warning: declaration shadows a local variable [-Wshadow]
src/models/wavtokenizer-dec.cpp:19:23: warning: declaration shadows a local variable [-Wshadow]
src/models/wavtokenizer-dec.cpp:85:23: warning: declaration shadows a local variable [-Wshadow]

## Disposition

* Enabling `-Wshadow` **wholesale** for `src/` is not viable without a cleanup: 82 of the 128 are the
  benign "shadows a field" class (dominated by constructor parameters, which clang reports under the
  same umbrella).
* Practical proposal for a future block: enable **`-Wshadow -Wno-shadow-field-in-constructor`** for
  `src/` on the CI/build path and fix the local-variable class (~46 sites, mechanical renames) — that is
  the class that silently kills a chain, and it is the one the Block-15 bug belonged to.  The field class
  can follow later.
* Doing the renames touches ~20 upstream `src/` files, so it belongs in its own block (or a dedicated
  cleanup commit) rather than folded into a code block: it will collide on every re-base otherwise.
* TODO item 15 keeps the flag proposal and now carries this measurement.

Repro: `cd build-rocm && python3 - <<'PY'` — replay `compile_commands.json` entries whose file is under
`src/` and whose `command` contains `llama.dir`, with `-o <obj>` → `-fsyntax-only` and `+ -Wshadow`.
