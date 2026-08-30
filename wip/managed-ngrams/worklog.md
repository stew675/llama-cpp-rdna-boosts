# WORKLOG — managed ngram LRU loading for qwen4exp

Session-by-session record of the design, implementation and validation.
Companion to `README.md` (overview) and `results/validation-summary.md`
(numbers).

---

## Phase 0 — reverse-engineering (pre-session, compacted context)

Goal: understand how qwen4exp's PLE n-gram table is quantized by Unsloth and
how llama.cpp's lazy mechanism works, so a "frankenstein" model (4-bit weights
+ Q8_0 ngrams) and a managed loading path could be designed.

- Mapped the qwen4exp lazy mechanism end to end: `per_layer_token_embd`
  created with `TENSOR_READ_LAZY`; CPU buffer wraps the mmap (zero-copy);
  pages fault on demand (`POSIX_MADV_RANDOM`, no prefetch on lazy ranges);
  per-ubatch host-side n-gram hash in `llm_graph_input_ple::set_input`;
  gather = `ggml_get_rows` (CPU dequant op supporting Q8_0/IQ4_NL/etc.);
  predecessor tokens come from the KV cells' `ext.tok`.
- Verified GGUF type IDs against the compiled repo (`libggml-base.so`):
  Q8_0=8, Q4_K=12, IQ4_NL=20, BF16=30, ... GGUF data is tight-packed (no
  row/tensor padding); `nbytes = nelems * type_size / blck_size`.
- Reverse-engineered the Unsloth quant scheme (HF repo
  `unsloth/Qwen3.8-Flash-Next-GGUF`): the ngram table is always quantized
  separately, never at the model's level — **Q8_0 (54.4 GB)** in the
  Q8_0/Q5_K_XL/Q6_K_XL builds, **IQ4_NL (28.8 GB)** in Q4_K_XL and below
  (even 1-bit IQ1_M keeps ngram IQ4_NL). The 54.4 GB ngram file is one
  identical LFS blob across the three top builds
  (sha256 `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20`).
  See `provenance.md` and `provenance/unsloth-qwen.md`.
- Extracted the full UD-Q4_K_XL per-tensor map (82.5 GB weights + 28.8 GB
  IQ4_NL ngram): `ffn_down_exps` Q5_1 (layers 2,4,30,46,47 -> Q8_0),
  `ffn_gate/up_exps` Q4_K (layer 2 -> Q5_K), `indexer.k/q_proj` BF16, norms
  F32, rest Q8_0; 48 layers; full attention at 3,7,11,...,47; n_embd 2560.
- Real PLE parameters: `ngram_size=3`, `heads_per_ngram=8` ->
  `ple_n_heads=16` rows/token, `ple_head_dim=160` (Q8_0 row = 170 B),
  table 320,001,536 x 160 (54.4 GB), head vocabs ~20.0M each,
  layer_multipliers `[23703573157769, 20109073645365, 8052911324071]`,
  PLE on layer 1, eos 248044, image 248056.
- Wrote and validated `graft-ngram-q8.py` (byte-graft the Q8_0 ngram into a
  Q4_K_XL build; verified on synthetic gguf-py files: offsets chain,
  byte-identical tensors, GGUFReader accepts output).
- Delivered `~/qwen4exp-ngram-lru-loading.md` (implementation plan) and
  `~/qwen38-flash-q4-q8.md` (graft guide).

## Phase 1 — Milestone 1: `llama_lazy_reader` + unit test

Implemented `src/llama-lazy-reader.{h,cpp}` + `tests/test-lazy-reader.cpp`.

Bugs found and fixed during bring-up:
- `evict_one()` must return the freed slot (not void).
- Each reader owns a `dup()`ed fd; the dtor closes it (the test's original
  fd must never be handed to a reader).
- The last-page read clamp must be **row-extent based**, not byte-offset
  based: page 415's byte offset (415 x 4096) exceeds the table end
  (10000 x 170); also rows_per_page x row_bytes (24 x 170 = 4080) !=
  page_size (4096), so file offsets must come from row indices.
- A page faulted during pass 1 of a `gather()` can be evicted by a later
  page of the *same* gather when the working set exceeds the budget; pass 2
  re-faults on demand (`page_to_slot[page] == -1` -> `ensure_page`).
- F32 (non-quantized) has no `to_float` trait; the reader copies the row.

## Phase 2 — Milestone 2: plumbing

`n_lazy_buf_size` in `llama_model_params` (default 0 = mmap path),
`lazy_read::buf_size`, `--lazy-buffer-size N(K|M|G)` in the shared common
parser (so llama-cli and llama-server both accept it), `ml.lazy.buf_size`
set in `llama.cpp`.

## Phase 3 — Milestone 3: qwen4exp integration

- `load_arch_tensors`: managed branch creates the reader over the table's
  file range (`dup(ml.files[ple_w->idx]->file_id())`, `ple_w->offs`,
  tight-packed `row_bytes`) and creates the ggml tensor with
  `TENSOR_SKIP_MANAGED` (data never loaded). POSIX-gated
  (`LLAMA_LAZY_READER_POSIX`); non-POSIX falls back to the mmap path.
- User path (`llama_model_init_from_user`, no `weights_map`): `ple_rows` is
  derived from the head ranges instead of `require_weight`; `create_tensor`
  skips `require_weight` for `TENSOR_READ_LAZY` when there are no files;
  the user-path `GGML_ASSERT(buft != nullptr)` was relaxed to a nullptr
  return for skipped tensors.
- `build_inp_ple` branches on the reader: F32 graph input
  `[ple_head_dim * n_heads, n_tokens]` instead of a `get_rows` node.
- `llm_graph_input_ple::set_input` gathers the hashed rows straight into
  the input tensor (`gather` layout == `get_rows` + reshape layout, so
  `build_ple` is untouched); `can_reuse` branches on the same condition.

## Phase 4 — Milestone 4: tests

- `tests/test-lazy-reader.cpp`: Q8_0 page-boundary math, eviction sweeps,
  single-slot budget, page-size bump, threaded gathers, fd ownership, F32.
- `tests/test-llama-archs.cpp`: qwen4exp fixture now emits the PLE keys and
  the file-backed roundtrip leg reloads with a 256 KiB lazy buffer, running
  the real managed path with a bit-exact logits check.
  Fixture constraints discovered:
  - `ple_head_dim * n_heads` must equal `n_embd` (the emb feeds `ple_key`,
    a `{n_embd, hc_dim}` mul-mat).
  - The PLE layer must be a **recurrent** layer (the conv-state row `p_l`
    is only allocated inside the recurrent layer filter loop).

## Phase 5 — Milestone 5: real-model validation, and a real bug

Built `logitdump` (deterministic logit dumper; later extended with
`-n NTOKENS --seed S --sample K --repeat`). On the real UD-Q4_K_XL build
(IQ4_NL ngram, 4 splits):

- **Bit-identical logits** mmap vs managed 4G vs managed 30G (234 tokens x
  248,320 vocab).
- **Prefetch bug found**: with `TENSOR_SKIP`, the ngram range dropped out of
  the lazy-ranges bookkeeping, so `init_mappings` WILLNEED-prefetched the
  whole 28.8 GB table into the process despite the budget (smaps: 26.8 GiB
  resident; peak RSS 145.3 GiB vs 115 GiB after fix). Fix:
  `lazy_read::add_range()` records the range in the managed branch so it is
  excluded from the WILLNEED complement and marked MADV_RANDOM, exactly like
  `TENSOR_READ_LAZY`. Verified: ngram resident 0.00 GiB, logits still
  bit-identical.

## Phase 6 — polish

- `TENSOR_SKIP_MANAGED`: skip without the misleading "model has unused
  tensor ... ignoring" warning.
- llama-server: verified `--lazy-buffer-size` flows through the shared
  common parser end to end (no server code changes needed).
- `LLAMA_LAZY_READER_STATS=1`: aggregate cache stats printed at model free.

## Phase 7 — GPU validation (3x Radeon AI PRO R9700, ROCm build)

The `build/` used earlier was CPU-only; the repo's `build-rocm`
(`GGML_HIP=ON`) is the GPU build. Rebuilt it and validated:

- Real model `-ngl 99` (weights in VRAM): bit-identical logits mmap vs
  managed; VRAM ~78-80 GiB across the 3 GPUs; system-RAM steady state
  ~1.3 GiB (weight mmap pages are unmapped after offload); ngram resident
  0.00 GiB.
- `--split-mode 3` (tensor/row split): bit-identical mmap vs managed.
- **ggml bug found**: the meta backend (tensor-split) crashed on host-buffer
  graph nodes — `model.input_embed (reshaped)` (a VIEW of a GET_ROWS, the
  token-embedding gather) reached `ggml_backend_meta_buffer_simple_tensor`
  because the s_copy FIXME guard only covered views of NONE-op leafs.
  Reproduced 5/5 with the PLE fixture (also with lazy_buf=0, so not the
  managed path's fault). Fix: per-backend node loop passes host-buffer
  nodes through unchanged; `ggml_backend_meta_get_split_state` returns the
  unsplit state for host buffers. After fix: full `test-llama-archs` on
  ROCm = 608 configs OK, CPU = 128 configs OK.

## Phase 8 — ngram residency / lookup stress

16K-token runs with deliberately small budgets (see
`results/validation-summary.md` for the table): bit-identical at every
budget from 4 MiB (2.11 GB read, ~500x reload factor) to 512 MiB; repeated
window at 32 MiB -> 95.7% hit rate (250,831 hits); 4 MiB + repeated window
-> correct thrash. Throughput flat (570-620 tok/s). Residency: ngram
file-backed 0.00 GiB; anonymous arena footprint == budget.

## Remaining work

- Real-model determinism/RSS re-run on the target box (Strix-Halo-style
  unified memory) and the Q8_0 graft validation once the 54.4 GB ngram
  download is available.
- Optional: Windows fallback for the reader (currently POSIX-gated),
  gemma4 adoption (same gather pattern), post-load weight-page DONTNEED,
  prefill prefetch.
- Not yet integrated into the push distro (blocks 01-12).
