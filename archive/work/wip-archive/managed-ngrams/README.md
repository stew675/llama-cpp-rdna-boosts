# managed-ngrams — software-managed LRU loading for the qwen4exp PLE n-gram table

Work-in-progress patch set for llama.cpp (fork point `a7cc83bba`, 7 commits)
that replaces the mmap-based lazy loading of the qwen4exp PLE n-gram table
(`per_layer_token_embd.weight`, ~28.8-54.4 GB depending on quantization) with a
**software-managed LRU cache** in a user-defined-size host buffer, loaded and
evicted on demand via `pread()`.

**Status: WIP. Not part of the push distro.** The patch set is under active
testing; see `worklog.md` and `results/` for what is proven and what remains.

---

## Why

Qwen3.8-Flash-Next (qwen4exp) has a PLE n-gram hash-embedding module: each
token gathers 16 rows (one per n-gram head) out of a ~320M-row table that is
far too large to keep in VRAM. llama.cpp's stock lazy path
(`TENSOR_READ_LAZY`) mmaps the GGUF and lets the kernel page cache decide what
stays resident:

- **Unbounded residency**: on a unified-memory system (Strix Halo-style VRAM
  carve-out) the table can silently consume the whole system-RAM pool.
- **Kernel LRU, not ours**: eviction is the global page cache's, not an LRU
  over our rows.
- **Load-time pollution**: the whole-file WILLNEED prefetch at load faults
  pages we may never read.

The managed approach gives:

1. **Exact byte budget** — the ngram cache occupies at most `N` bytes of host
   RAM, user-defined.
2. **Our eviction policy** — a second-chance clock over 4 KB pages of the
   table.
3. **No mmap dependency for the table** — no WILLNEED interaction, no
   page-cache pollution.
4. **Same numerics** — rows are dequantized with the same `to_float` the
   `get_rows` op uses, so outputs are bit-identical (verified).

## What was built

| Component | File(s) | Notes |
|---|---|---|
| `llama_lazy_reader` | `src/llama-lazy-reader.{h,cpp}` | fixed-size host arena, page-granular bookkeeping (direct-indexed `page_to_slot`, no hash map), second-chance clock eviction, `pread` with row-extent clamping, dedup'd page-ensure, mid-gather re-fault, F32 copy path, stats atomics |
| Plumbing | `include/llama.h`, `src/llama-model.cpp`, `src/llama-model-loader.h`, `src/llama.cpp`, `common/` | `llama_model_params.n_lazy_buf_size`, `lazy_read::buf_size`, `--lazy-buffer-size N(K\|M\|G)` (shared parser, so llama-cli **and** llama-server) |
| qwen4exp integration | `src/models/qwen4exp.cpp`, `src/models/models.h` | reader creation + `TENSOR_SKIP_MANAGED` + `lazy_read::add_range` in `load_arch_tensors`; F32 graph-input branch in `build_inp_ple`; dual-path `llm_graph_input_ple::set_input`/`can_reuse`; `LLAMA_LAZY_READER_STATS` aggregate print at model free |
| Loader guards | `src/llama-model-loader.cpp` | user path (`llama_model_init_from_user`, no `weights_map`) works with PLE keys; `TENSOR_READ_LAZY` no-ops without files; `TENSOR_SKIP_MANAGED` skips data without the misleading "unused tensor" warning |
| ggml fix | `ggml/src/ggml-backend-meta.cpp` | tensor-split (meta backend) crashed on host-buffer graph nodes (views of a reshaped token embedding, PLE gathers); fixed by passing host nodes through unchanged and making `ggml_backend_meta_get_split_state` host-safe |
| Tests | `tests/test-lazy-reader.cpp`, `tests/test-llama-archs.cpp` | unit test (Q8_0 + F32 cases) and the qwen4exp PLE fixture whose file-backed roundtrip runs the real managed path |

## New options

```
--lazy-buffer-size N(K|M|G)   managed buffer size in bytes for on-demand tensors,
                              e.g. the qwen4exp PLE n-gram table
                              (default: 0 = mmap-based lazy loading, current behavior)
                              (env: LLAMA_ARG_LAZY_BUF_SIZE)
```

Additional diagnostics:

```
LLAMA_LAZY_READER_STATS=1      print aggregate ngram cache stats (hits/misses/bytes
                               read/budget) when the model is freed
LLAMA_GRAPH_INPUT_DEBUG=1      per-ubatch "ple managed" line with running stats
```

## How to enable

```sh
# managed mode: cap the ngram cache at 4 GiB of host RAM
llama-cli -m <model>.gguf -ngl 99 --lazy-buffer-size 4G ...

# mmap path (default, zero behavior change):
llama-cli -m <model>.gguf -ngl 99 ...

# llama-server works the same way (shared common parser):
llama-server -m <model>.gguf --lazy-buffer-size 4G
```

## What is verified (see `results/validation-summary.md` for tables)

- **Determinism**: bit-identical logits vs the mmap path on the real
  UD-Q4_K_XL model (IQ4_NL ngram) — 234-token prompts and 16K-token stress
  runs, layer-split and tensor-split, budgets from 4 MiB to 30 GiB.
- **Loading / lookups / eviction**: 16K-token random streams push a ~1 GB
  working set through budgets as small as 4 MiB (~500x reload factor) with
  zero correctness loss; a repeated window shows 95.7% hit rate (250,831 hits
  vs 11,186 misses) when the window fits the budget.
- **Residency**: the table is never mmap'd in managed mode (file-backed ngram
  pages 0.00 GiB); the anonymous arena footprint equals the budget exactly.
  The load-time WILLNEED no longer faults the table (a bug found during
  validation — `lazy_read::add_range` keeps the range out of the prefetch).
- **GPU**: 3x Radeon AI PRO R9700 — weights in VRAM (~78-80 GiB), system RAM
  steady state ~1.3 GiB; full `test-llama-archs` passes (608 configs ROCm,
  128 configs CPU).
- **Throughput**: flat (570-620 tok/s) across every budget — the managed path
  is invisible next to GPU decode.

## Layout

```
README.md                    this file
worklog.md                   session-by-session record
provenance.md                Unsloth reverse-engineering provenance
plan/
  qwen4exp-ngram-lru-loading.md   implementation plan (design, API, milestones)
  qwen38-flash-q4-q8.md           guide: graft Q8_0 ngram into UD-Q4_K_XL (frankenstein model)
provenance/unsloth-qwen.md        how Unsloth arranges/quantizes the weights
tests/                            all test + tool source code
  test-lazy-reader.cpp            unit test for llama_lazy_reader
  test-llama-archs.cpp            arch test with the PLE fixture (repo tree copy)
  logitdump.cpp                   deterministic logit dumper / stress driver
  graft-ngram-q8.py               Q8_0 ngram graft script
  gguf_parse.py                   minimal GGUF header parser
  smaps_sample.py / smaps_ngram.py / smaps_anon.py   RSS/VRAM samplers
results/
  validation-summary.md           all measured results
patches/
  0001-...0007                    the 7 commits (git format-patch)
  managed-ngrams-combined.diff    single combined diff
```

Apply: the patches apply to a clean llama.cpp checkout at `a7cc83bba`
(`git am patches/0001-*.patch`), or review the combined diff. Not yet
integrated into the push distro (blocks 01-12) — deliberately kept WIP for
further testing.
