# Managed LRU loading for the qwen4exp PLE n-gram table

Implementation plan for replacing the mmap-based lazy loading of the qwen4exp
n-gram table (`per_layer_token_embd.weight`, the PLE hash-embedding table) with
a **software-managed LRU cache** in a user-defined-size host buffer, loaded and
evicted on demand via `pread()`.

Target model: Qwen3.8-Flash-Next (Unsloth UD-Q4_K_XL + Q8_0 ngram graft, or any
build with `per_layer_token_embd.weight`). The design is written generically
enough that gemma4 (the other `TENSOR_READ_LAZY` user) can adopt it later.

## Implementation status

Milestones 1-4 are implemented and committed on the `ngram-disk` branch
(4 commits, `57a79413a`..`df4411212`), ready to be extracted as a patch:

- `src/llama-lazy-reader.{h,cpp}` -- the cache class (page-granular
  bookkeeping, second-chance clock, `pread`, F32 memcpy path).
- Plumbing: `n_lazy_buf_size` (llama.h/defaults), `ml.lazy.buf_size`,
  `--lazy-buffer-size N(K|M|G)` (common).
- qwen4exp integration: reader creation + `TENSOR_SKIP` in `load_arch_tensors`,
  F32 graph-input branch in `build_inp_ple`, dual-path
  `llm_graph_input_ple::set_input`/`can_reuse`, user-path guards in
  `load_arch_tensors` and `create_tensor`.
- Tests: `tests/test-lazy-reader.cpp` (Q8_0 + F32 cases) and the qwen4exp PLE
  fixture in `tests/test-llama-archs.cpp` whose file-backed roundtrip leg runs
  the real managed path (256 KiB budget) with a bit-exact logits check.

Implementation notes (deltas vs. the plan below):

- The PLE layer of the arch-test fixture must be a **recurrent** layer: the
  conv-state row `p_l[il]` is allocated only inside the recurrent layer filter
  loop of `llama_memory_recurrent` (src/llama-memory-recurrent.cpp:110). The
  real model's PLE layer 1 is recurrent, so this is a fixture-only constraint.
- F32 has **no `to_float` trait**; the reader treats a null `to_float` as a
  plain row copy (F32). Q8_0/F16 use the trait.
- Managed mode is POSIX-only for now (`pread`/`dup`, gated by
  `LLAMA_LAZY_READER_POSIX`); non-POSIX builds fall back to the mmap path.
- `load_arch_tensors` in the user path (`ml.files.empty()`) derives `ple_rows`
  from the head ranges instead of `require_weight`; `create_tensor` skips
  `require_weight` for `TENSOR_READ_LAZY` when there are no files, and the
  user-path `GGML_ASSERT(buft != nullptr)` was relaxed to a nullptr return for
  `TENSOR_SKIP`.
- Validated: full `test-llama-archs` (128 configs, qwen4exp OK on NMSE and
  Roundtrip), `test-lazy-reader` (Q8_0 + F32), ctest. Real-model validation
  (section 9.3) still pending the user's model downloads.

### Real-model validation on the actual UD-Q4_K_XL build (IQ4_NL ngram)

Done with `logitdump` (a tiny deterministic logit dumper) on the 4-split
`/models/Qwen3.8/Flash-Next/Q4_K_XL` model:

- **Determinism**: mmap path vs `--lazy-buffer-size 4G` produce **bit-identical
  logits** (234 prompt tokens x 248,320 vocab = 58M floats, same md5). The
  managed reader ran: 3664 distinct pages, 14.8 MiB read.
- **Prefetch bug found and fixed**: with `TENSOR_SKIP` alone, the ngram range
  drops out of the lazy-ranges bookkeeping, so `init_mappings`
  WILLNEED-prefetches its whole file range and the kernel faults the 28.8 GB
  table into the process despite the managed budget (peak RSS 145.3 GiB on the
  real model). Fix: `load_arch_tensors` records the range via
  `lazy_read::add_range()` in the managed branch, which excludes it from the
  WILLNEED complement and marks it MADV_RANDOM, exactly like
  `TENSOR_READ_LAZY`. Verified with smaps sampling: ngram resident
  **26.8 GiB -> 0.00 GiB**, peak RSS 145.3 -> 115 GiB, logits still
  bit-identical.
- RSS on this CPU-only box: managed final ~73-74 GiB (82.5 GB weights in RAM
  + repack + 4 GiB arena), ngram range never faulted. On the target box the
  weights live in VRAM, so system RAM holds just the budget.
- Note: this box's CPU backend repacks the Q4_K experts to q4_K_8x8 at load
  (42.3 GiB extra anonymous buffer), which dominates the load-time peak in
  both modes and is unrelated to the ngram.

### GPU validation (3x Radeon AI PRO R9700, ROCm build)

- Real model with `-ngl 99` (weights in VRAM, layer split): **bit-identical
  logits** mmap vs managed 4G vs managed 30G. VRAM ~78-80 GiB across the 3
  GPUs; system-RAM steady state ~1.3 GiB (weight mmap pages are unmapped
  after offload); ngram stays 0.00 GiB resident.
- `--split-mode 3` (tensor/row split across all 3 GPUs): **bit-identical**
  mmap vs managed 4G.
- test-llama-archs on ROCm: 608 configs OK, incl. the qwen4exp PLE managed
  roundtrip on each single GPU (Roundtrip bit-exact) and the Meta row-split
  leg.
- **ggml fix found by the GPU test**: the meta backend (tensor-split) assumed
  every graph node lives in a meta buffer and crashed on host tensors (views
  of a reshaped token embedding, PLE gathers); the old FIXME guard only
  covered views of NONE-op leafs. Fixed by passing host-buffer nodes through
  unchanged in the per-backend node loop and making
  `ggml_backend_meta_get_split_state` return the unsplit state for host
  buffers (commit `3f27b0f38`).

### Ngram residency / lookup stress (16K random tokens, seed 42, GPU)

All runs bit-identical to the mmap baseline (sampled logits, 17 positions x
248,320 vocab). `logitdump` in `/tmp` (`-n NTOKENS --seed S --sample K`,
optionally `--repeat 512` for a repeated window).

| Budget | Misses | Hits | Bytes read | Reload factor | tok/s |
|---|---|---|---|---|---|
| 4 MiB  | 521,899 | 5      | 2.11 GB | ~503x | 610.7 |
| 32 MiB | 261,767 | 244    | 1.06 GB | ~32x  | 617.1 |
| 512 MiB| 258,556 | 3,449  | 1.05 GB | ~2x   | 604.4 |
| 32 MiB (repeated 512-block x32) | 11,186 | 250,831 (95.7%) | 45 MB | 1.3x | 574.6 |
| 4 MiB (repeated) | 519,551 | 2,177 | 2.10 GB | ~500x | 570.9 |

Interpretation:

- misses -> pread from disk; hits -> served from the arena. With a repeated
  window the clock LRU holds the working set and serves 95.7% of lookups from
  memory; with the window 8x the budget it thrashes correctly (hit rate ~0)
  and still returns bit-exact values.
- Throughput is flat across every budget (570-620 tok/s): the managed path is
  invisible next to the GPU decode cost.
- Residency: the table is never mmap'd in managed mode (file-backed ngram
  pages 0.00 GiB); the arena is anonymous and its footprint is exactly the
  budget (anon delta between a 512 MiB and a 4 MiB budget = 0.49 GiB).

---

## 1. Motivation

The current lazy path maps the whole GGUF and lets the kernel page cache decide
what stays resident:

- `per_layer_token_embd.weight` is created with `TENSOR_READ_LAZY`
  (`src/models/qwen4exp.cpp`, `load_arch_tensors`).
- The loader gives it a CPU buffer that *wraps the mmap* and never copies the
  data; the graph gathers rows with `ggml_get_rows(table, rows)`, and the CPU
  backend dequantizes rows straight out of the mmap
  (`ggml_compute_forward_get_rows_q`, `ggml/src/ggml-cpu/ops.cpp:4979`).
- Residency is unbounded and governed by the global page cache: on a unified
  memory system (e.g. Strix Halo with a VRAM carve-out) the ngram can silently
  consume the whole system-RAM pool, and the kernel's eviction is global, not
  LRU-over-our-rows.

Goals of the managed approach:

1. **Exact byte budget**: the ngram cache occupies at most `N` bytes of host
   RAM, user-defined.
2. **Our eviction policy**: second-chance clock over ngram *pages* (not the
   kernel's global LRU over everything).
3. **No mmap dependency for the table**: no `MAP_POPULATE` / `WILLNEED`
   interaction, no page-cache pollution, works even with `--load-mode none`.
4. **Same numerics**: the dequantized F32 rows are bit-identical to the mmap
   path (same `to_float` dequantizer), so outputs do not change.

Real model parameters (verified from the GGUF metadata):

| | |
|---|---|
| table shape | 320,001,536 rows x 160 dims |
| type / row size | Q8_0, 170 bytes/row (tight-packed, 5 x 34B blocks) |
| table size on disk | 54.4 GB |
| `ple_n_heads` | 16 = (ngram_size 3 - 1) x heads_per_ngram 8 |
| rows per token | 16 (2720 bytes, typically 1-2 pages) |
| 512-token ubatch | 8192 rows, ~342 distinct 4K pages worst-case |
| head vocabs | 16 heads x ~20.0M rows each (sum 320,001,446) |

---

## 2. Design overview

```
┌───────────────────────── llama.cpp ─────────────────────────┐
│                                                             │
│  load_arch_tensors (qwen4exp)                               │
│    │  creates llama_lazy_reader{fd dup, data_offs, row_* }  │
│    ▼                                                        │
│  llama_model_qwen4exp::lazy_reader  (shared, per model)     │
│    │                                                        │
│  graph::build_inp_ple                                       │
│    │  managed mode: ggml input tensor "ple_embd"            │
│    │  F32 [160*16, n_tokens]  (replaces the get_rows node)  │
│    ▼                                                        │
│  llm_graph_input_ple::set_input  (per ubatch)               │
│    │  1. hash tokens -> 16 row ids each (existing code)     │
│    │  2. lazy_reader->gather(rows, dst)  (mutex, pread,     │
│    │     clock LRU eviction, dequant)                       │
│    │  3. ggml_backend_tensor_set(ple_embd, dst)             │
└─────────────────────────────────────────────────────────────┘
```

The cache is **row-granular on disk, page-granular in bookkeeping**:

- rows are contiguous in the file: `row i` at `data_offs + i*170` bytes;
- a 4 KB page holds 24 rows (170 x 24 = 4080 <= 4096);
- `page_to_slot` is a direct-indexed `int32_t` array over all ~13.3M pages
  (53 MB, fixed cost, no hash map) -> `row -> page -> slot` is O(1);
- slots form the arena (`budget / 4096` slots) with a **second-chance clock**
  reuse bit per slot for O(1)-amortized eviction;
- the cache stores **on-disk Q8_0 bytes** (170 B/row), dequantized on gather
  with `ggml_get_type_traits(type)->to_float` (the same function `get_rows_q`
  uses) -- a 10 GB budget then holds 62M rows (~19% of the table) instead of
  16M if we stored F32, at identical gather cost.

---

## 3. New component: `src/llama-lazy-reader.{h,cpp}`

### 3.1 Public API

```cpp
// src/llama-lazy-reader.h
#pragma once

#include "ggml.h"

#include <cstdint>
#include <mutex>
#include <vector>

// Managed on-demand reader for a single "lazy" tensor: rows live on disk in a
// file, a fixed-size host arena caches a subset, and eviction is a clock LRU.
// All access goes through gather(), which is internally synchronized so that
// several llama_contexts sharing one model can call it concurrently.
class llama_lazy_reader {
public:
    struct config {
        int          fd;         // owned fd (dup of the model file), closed in dtor
        size_t       data_offs;  // absolute file offset of the tensor data
        size_t       row_bytes;  // tight-packed on-disk bytes per row (NOT ggml_row_size: no padding)
        uint64_t     n_rows;     // number of rows in the table
        enum ggml_type type;     // source quantization (e.g. GGML_TYPE_Q8_0)
        int64_t      row_nelems; // elements per row (ne[0])
        size_t       budget;     // managed buffer size in bytes (> 0)
        size_t       page_size = 4096;
    };

    llama_lazy_reader(const config & cfg);
    ~llama_lazy_reader();

    // Dequantize rows[] (n entries) into dst, laid out as dst[j*row_nelems + d].
    // Loads missing pages with pread(), evicts with the clock LRU, all under
    // one lock hold so no page can disappear mid-gather.
    void gather(const int32_t * rows, size_t n, float * dst);

    uint64_t n_pages_total() const { return n_pages; }
    size_t   budget()         const { return budget_bytes; }

    // counters for LLAMA_LOG_DEBUG / stats
    uint64_t hits()   const { return n_hits; }
    uint64_t misses() const { return n_misses; }
    uint64_t bytes_read() const { return n_bytes_read; }

private:
    void ensure_page(uint32_t page);   // called with mtx held
    void evict_one();                  // called with mtx held (clock scan)

    // config
    const int          fd;
    const size_t       data_offs;
    const size_t       row_bytes;
    const uint64_t     n_rows;
    const ggml_to_float_t to_float;
    const int64_t      row_nelems;
    const size_t       page_size;
    const size_t       n_pages;       // ceil(n_rows / rows_per_page)
    const size_t       rows_per_page; // floor(page_size / row_bytes), >= 1
    const size_t       n_slots;       // budget / page_size
    const size_t       budget_bytes;

    // state (guarded by mtx)
    std::mutex         mtx;
    std::vector<int32_t> page_to_slot; // n_pages, -1 = not resident
    std::vector<uint8_t> slot_ref;     // n_slots, second-chance bit
    std::vector<int32_t> slot_page;    // n_slots, page id in the slot
    uint8_t *          arena;          // n_slots * page_size
    size_t             clock = 0;

    // stats
    std::atomic<uint64_t> n_hits{0}, n_misses{0}, n_bytes_read{0};
};
```

Notes:

- `row_bytes` must be the **tight-packed** size `row_nelems * ggml_type_size(t)
  / ggml_blck_size(t)` (must divide evenly). `ggml_row_size()` pads rows to 32
  bytes and is wrong for file offsets; GGUF files pack tensor data tightly (no
  padding between rows or tensors).
- `page_size` is fixed at 4096 (a constant in the class; no need to expose it).
- `arena` is `malloc(n_slots * page_size)` (or `posix_memalign`); with a 10 GB
  budget that is one contiguous 10 GB allocation. Bookkeeping is `page_to_slot`
  (53 MB for this model) + `slot_ref` + `slot_page` (~5 MB), independent of the
  budget.
- Stats are cheap atomics; expose them through an env-gated debug print (see
  section 7) rather than the API surface above if preferred.

### 3.2 `gather()` algorithm

```
lock mtx

// pass 1: ensure every distinct page needed by this ubatch is resident
for each unique page p among rows[]:
    if page_to_slot[p] != -1:
        slot_ref[slot] = 1;  n_hits++
    else:
        n_misses++
        slot = evict_one()      // clock scan, O(1) amortized
        pread(fd, arena + slot*page_size, page_size,
              data_offs + p*page_size)     // clamp to table end
        page_to_slot[p] = slot;  slot_page[slot] = p;  slot_ref[slot] = 1
        n_bytes_read += page_size

// pass 2: dequantize (all needed pages are resident; no eviction between
//         pass 1 and pass 2 because we still hold the lock)
for j in [0, n):
    row = rows[j]
    page = row / rows_per_page;  off = (row % rows_per_page) * row_bytes
    to_float(arena + page_to_slot[page]*page_size + off,
             dst + j*row_nelems, row_nelems)

unlock mtx
```

Correctness property worth stating in a comment: **a page needed by the current
ubatch is never evicted during that same `gather`** -- eviction happens only in
`evict_one()` while loading a page that was not resident, so worst case a
working set larger than the budget re-loads a page in the same call (identical
behavior to the kernel page cache today, just bounded and ours).

### 3.3 `evict_one()` (second-chance clock)

```
loop:
    slot = clock % n_slots;  clock++
    if slot_ref[slot] == 0:
        if slot_page[slot] != -1: page_to_slot[slot_page[slot]] = -1
        return slot
    slot_ref[slot] = 0   // second chance
```

### 3.4 Portability

- `pread` is POSIX; on Windows use `_lseeki64` + `_read` under the mutex, or
  `ReadFile` with `OVERLAPPED`. Initial implementation: gate the managed path
  to platforms where `pread` exists; elsewhere fall back to the mmap lazy path
  (the mode is decided at load, so this is a one-line condition). fd duplication
  is `dup()` / `_dup()`.

---

## 4. qwen4exp integration (`src/models/qwen4exp.cpp`)

### 4.1 `load_arch_tensors` -- create the reader

Current (approx. line 125-141):

```cpp
if (hparams.ple_n_heads > 0) {
    const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
    const auto & ple_w = ml.require_weight(ple_name.c_str());
    const int64_t ple_rows = ple_w.tensor->ne[1];
    // sanity check on head ranges ...
    per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                       { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
}
```

New:

```cpp
if (hparams.ple_n_heads > 0) {
    const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
    const auto & ple_w = ml.require_weight(ple_name.c_str());
    const int64_t ple_rows = ple_w.tensor->ne[1];
    // sanity check on head ranges ... (unchanged)

    if (managed_lazy(params)) {   // params.n_lazy_buf_size > 0 (see 4.4)
        // reader over the raw file data; the tensor itself is never materialized
        const int64_t row_nelems = ple_w.tensor->ne[0];
        const size_t  row_bytes  = row_nelems * ggml_type_size(ple_w.tensor->type)
                                                 / ggml_blck_size(ple_w.tensor->type);
        GGML_ASSERT(row_nelems % ggml_blck_size(ple_w.tensor->type) == 0);
        lazy_reader = std::make_shared<llama_lazy_reader>(llama_lazy_reader::config{
            /*fd*/         dup(ml.files[ple_w.idx]->file_id()),
            /*data_offs*/  ple_w.offs,
            /*row_bytes*/  row_bytes,
            /*n_rows*/     (uint64_t) ple_rows,
            /*type*/       ple_w.tensor->type,
            /*row_nelems*/ row_nelems,
            /*budget*/     params.n_lazy_buf_size,
        });
        // keep the ggml tensor for metadata/counts only; do not load its data
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_SKIP);
    } else {
        per_layer_tok_embd = create_tensor(..., TENSOR_READ_LAZY);  // current path
    }
}
```

Details to get right:

- `ml.files` and `llama_tensor_weight::{idx, offs}` are public
  (`src/llama-model-loader.h`); `llama_file::file_id()` returns the fd
  (`src/llama-mmap.h`). `dup()` keeps the fd alive after the loader is
  destroyed (the model does **not** retain `llama_files` today -- only
  `llama_mmaps`).
- **User path** (`ml.files.empty()`, e.g. `llama_model_init_from_user` and
  test-llama-archs): there is no `weights_map`, so `ml.require_weight()`
  throws. Guard the whole block: when `files.empty()`, derive `ple_rows` from
  the gguf metadata (`gguf_find_tensor(ml.metadata, ...)`) or from
  `ple_head_offsets[last] + ple_head_vocab_sizes[last]` rounded up to a
  multiple of 32, keep the head-range sanity check, and **do not create the
  reader** (no fd) -- fall back to a plain `create_tensor` (no flags), which
  the user path fills via `set_tensor_data`. See section 9.2.
- `TENSOR_SKIP` (`1 << 2`) makes `create_tensor` return `nullptr` after
  registering the tensor: `n_created` is incremented, `size_data -= nbytes`
  cancels the `weights_map` contribution from `init_mappings`, the dims are
  still validated by `check_tensor_dims`, and `load_all_data` never touches it.
  It logs "model has unused tensor ... ignoring" at WARN -- cosmetically wrong
  for this case; consider a quieter variant flag (e.g. `TENSOR_SKIP_MANAGED`)
  or downgrading the log when a lazy_reader exists. Note: `TENSOR_SKIP` only
  works on the file path today (`GGML_ASSERT(buft != nullptr)` in the
  `files.empty()` branch of `create_tensor`); relax that to
  `if (buft == nullptr) return nullptr;` if managed mode should work on the
  user path too.
- The sanity check on `ple_head_offsets`/`ple_head_vocab_sizes` vs `ple_rows`
  must run **before** either branch (it reads `ple_w.tensor->ne[1]`).
- The managed flag is read from `ml.lazy.buf_size` (see 4.4): `load_arch_tensors`
  has `ml`, and this mirrors exactly how `ml.lazy.mode` is plumbed today
  (`src/llama.cpp:321`: `ml.lazy.mode = params.lazy_mode;`). No change to
  `llama_model_base` is needed.

### 4.2 `llm_graph_input_ple` -- dual-path `set_input`

Current class (src/models/qwen4exp.cpp:966, `set_input` at 990): holds `rows`
(I32 graph input), computes `idx[i*n_heads + h]` host-side, then
`ggml_backend_tensor_set(rows, idx.data(), 0, ...)`.

New: add a managed path with an F32 output tensor.

```cpp
class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx)
        : pmodel(pmodel), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override {
        // ... existing hash + predecessor code, fills idx[] (unchanged) ...
        if (pmodel.lazy_reader) {
            // managed path: dequantize the needed rows straight into the graph input
            emb_scratch.resize(idx.size() * pmodel.hparams.ple_head_dim);
            pmodel.lazy_reader->gather(idx.data(), idx.size(), emb_scratch.data());
            ggml_backend_tensor_set(emb, emb_scratch.data(), 0, emb_scratch.size()*sizeof(float));
        } else {
            ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        if (pmodel.lazy_reader) {
            return emb->ne[1] == params.ubatch.n_tokens;
        }
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // mmap path
    ggml_tensor * emb  = nullptr;   // managed path, F32 [ple_head_dim*n_heads, n_tokens]

    const llama_model_qwen4exp & pmodel;
    const llama_kv_cache_context * mctx;
    std::vector<llama_token> prev;
    std::vector<float> emb_scratch;  // reused across ubatches
};
```

Layout guarantee: `gather` writes `dst[j*row_nelems + d]` with `j = t*n_heads +
h`; the graph input tensor is column-major `[160*16, n_tokens]`, so flat index
`c + t*(160*16)` with `c = h*160 + d` -- identical ordering. This matches the
current `get_rows` + `reshape_2d` output exactly, so `build_ple` is untouched.

### 4.3 `build_inp_ple` -- two small graph branches

Current (src/models/qwen4exp.cpp:1101):

```cpp
ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
ggml_set_input(ple_inp->rows);
ggml_tensor * rows = ple_inp->rows;
res->add_input(std::move(ple_inp));
ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
cb(emb, "ple_embd", -1);
return emb;
```

New: branch on the reader at the top (the mode is fixed at load, so the graph
topology is stable per model -- no per-ubatch variation, graph reuse is safe).

```cpp
const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model);
if (pmodel.lazy_reader) {
    ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                      hparams.ple_head_dim * n_heads, n_tokens);
    ggml_set_input(ple_inp->emb);
    ggml_tensor * emb = ple_inp->emb;
    res->add_input(std::move(ple_inp));
    cb(emb, "ple_embd", -1);
    return emb;                       // no gather node in the graph
}
// ... existing mmap path unchanged ...
```

The caller (`graph` ctor) already does `ggml_build_forward_expand(gf, ple_emb)`
for both cases; the input tensor is allocated by the backend scheduler and
transferred to the GPU for the `ple_key`/`ple_value` mul-mats exactly like the
GET_ROWS output today.

### 4.4 Mode selection

Add to `llama_model_qwen4exp` (public member, `src/models/models.h:2278`):

```cpp
std::shared_ptr<llama_lazy_reader> lazy_reader;   // null = mmap lazy path
```

Managed mode is active iff `ml.lazy.buf_size > 0`. The mmap path remains the
default (`buf_size == 0`), so there is zero behavior change unless the user
opts in.

Plumbing (mirrors `lazy.mode`): add a field to `llama_model_loader::lazy_read`
(`src/llama-model-loader.h:89`) and set it next to `lazy.mode` at
`src/llama.cpp:321`:

```cpp
// llama-model-loader.h
struct lazy_read {
    enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;
    size_t buf_size = 0;   // managed buffer size in bytes; 0 = mmap-based lazy
    ...
} lazy;
```

```cpp
// llama.cpp, in llama_load_model_from_file / llama_model_create caller
ml.lazy.mode     = params.lazy_mode;
ml.lazy.buf_size = params.n_lazy_buf_size;
```

In `load_arch_tensors`, the condition is `ml.lazy.buf_size > 0`. The loader's
mmap/lazy machinery is untouched: when `buf_size > 0`, `TENSOR_SKIP` keeps the
tensor out of `load_all_data`, so no mapping, no `MADV_RANDOM` ranges, no
`WILLNEED` interaction for this tensor.

---

## 5. Parameter & CLI plumbing

- `include/llama.h` -- `llama_model_params`:
  ```cpp
  size_t n_lazy_buf_size; // managed buffer size in bytes for on-demand tensors
                          // (qwen4exp PLE n-gram table); 0 = mmap-based lazy
                          // loading (current behavior)
  ```
  Also update `llama_model_default_params()` in `src/llama-model.cpp` and the
  `llama_model_params` doc comment near `lazy_mode` (llama.h:217-220, 320-330).
- `src/llama-model-loader.h` -- add `size_t buf_size` to `lazy_read` (see 4.4).
- `src/llama.cpp:321` -- set `ml.lazy.buf_size = params.n_lazy_buf_size;`
  next to `ml.lazy.mode`.
- `common/common.h` -- `common_params`: `size_t lazy_buf_size = 0;`
- `common/arg.cpp` -- new option next to `--lazy-mode` (arg.cpp:2730-2744):
  ```
  --lazy-buffer-size N(K|M|G)   size of the managed on-demand tensor buffer;
                                0 disables the managed path (default: 0)
  ```
  Reuse the existing size parser used by `--split-max-size`-style options.
- `common/common.cpp:1691` -- `mparams.n_lazy_buf_size = params.lazy_buf_size;`
- Server: optionally expose in `tools/server/server-schema.cpp` (follow the
  pattern of `--lazy-mode` if it is exposed there; otherwise skip).

---

## 6. Build system

- `src/CMakeLists.txt`: add `llama-lazy-reader.cpp` to the `llama` sources.
- New file header includes: `ggml.h`, `<mutex>`, `<vector>`, `<atomic>`,
  `<unistd.h>` (POSIX) with a Windows guard.
- No new dependencies.

---

## 7. Diagnostics

- `LLAMA_GRAPH_INPUT_DEBUG` already dumps graph input state
  (`llm_graph_input_i` ctor, `src/llama-graph.h:102`); print the PLE input mode
  and `lazy_reader` stats there (hits/misses/bytes read) at debug level.
- On context free (or at exit), if `LLAMA_DEBUG` / the env var is set, print:
  `ngram cache: X hits, Y misses (Z%), W MiB read, budget V MiB`.
- Consider an env var `LLAMA_LAZY_READER_STATS=1` if debug-level logging is too
  noisy.

---

## 8. Performance analysis

Worst-case costs per ubatch (fully cold cache):

- decode (1 token): 16 rows -> 1-2 pages -> 4-8 KB `pread`, one `to_float` of
  2560 floats. Microseconds, far below the decode step cost.
- 512-token prefill: 8192 rows -> ~342 distinct pages -> ~1.4 MB `pread` +
  8192 dequant calls (160 floats each). With a 10 GB budget this is a one-time
  warm-up; subsequent ubatches hit.
- 262K-token prefill: working set ~11 GB cold I/O worst case (same as the mmap
  path's fault-in cost); NVMe does this in ~1-2 s. Budget exceeded -> clock
  eviction + reload, identical thrash profile to the kernel page cache but
  bounded and ours.

Steady state:

- The hash maps each token to 16 rows spread over 16 heads of ~20M rows each,
  so hit rate follows **n-gram Zipf** (rows depend on the token *and* its
  predecessors), not token Zipf. Decode over natural text hits well after a
  short warm-up; a 10 GB cache holds 62M rows (19% of the table).
- Lock: one `gather` per ubatch per context; the lock is held for the dequant
  of 16 x n_tokens rows (microseconds). Multiple contexts (speculative
  decoding) serialize on it -- acceptable; can be refined later by pinning
  pages with a refcount and copying outside the lock if contention shows up.

Memory footprint:

- arena = budget (e.g. 10 GB)
- `page_to_slot` = n_pages x 4 B = 53 MB (fixed, this model)
- `slot_ref` + `slot_page` ~ 5 MB
- `emb_scratch` = 16 x n_tokens x 4 B (e.g. 32 KB decode / 16 MB at 512 tokens)
- No mmap of the ngram range, no page-cache pollution from the 54.4 GB file.

---

## 9. Testing plan

### 9.1 New unit test: `tests/test-lazy-reader.cpp`

A standalone test for `llama_lazy_reader` (no llama.cpp integration), since the
page arithmetic and eviction are new, self-contained logic. Builds a small
temporary file of quantized rows (e.g. 10000 rows x 32 elems Q8_0, written with
gguf-py or as raw bytes) and a tiny budget.

Cases:

1. **Correctness**: `gather` matches a direct dequant of the same rows for:
   rows inside a page, rows at page boundaries (row 23/24, the last row of a
   page), the final partial page, and repeated row ids in one call (dedup does
   not corrupt output).
2. **Eviction**: budget smaller than the working set; sweep a sliding window of
   rows across the table repeatedly; every returned value is still correct
   (exercises the clock, ref bits, and `page_to_slot` invalidation).
3. **All-resident**: budget >= table size; second pass has 100% hits.
4. **Stats**: hits + misses + bytes_read are sane (bytes_read <= table size,
   hits + misses == distinct pages touched).
5. **Threaded**: several threads gathering concurrently (exercises the mutex)
   produce results identical to serial gathers.
6. **Config validation**: row_bytes not dividing page_size evenly (rows_per_page
   >= 1), budget < page_size (clamped to one slot), fd close on destruction.

### 9.2 `tests/test-llama-archs.cpp` -- PLE coverage for qwen4exp

`get_gguf_ctx` drives the model from KV metadata only (the "user" path:
`llama_model_init_from_user` + `set_tensor_data`), so adding PLE to the fixture
is a matter of KV keys plus a few small loader accommodations.

#### KV keys (in `get_gguf_ctx`, inside the existing `LLM_ARCH_QWEN4EXP` block)

```cpp
if (arch == LLM_ARCH_QWEN4EXP) {
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

    // PLE n-gram hash embeddings on layer 1
    const uint32_t ple_ngram     = 3;                 // 2- and 3-gram heads
    const uint32_t ple_per_ngram = 4;                 // heads per ngram size
    const uint32_t ple_n_heads   = (ple_ngram - 1) * ple_per_ngram;  // 8
    const uint32_t ple_row_dim   = n_embd / 4;        // 64
    const uint32_t head_vocab    = 1024;              // rows per head
    std::vector<uint32_t> ple_offsets(ple_n_heads), ple_vocabs(ple_n_heads, head_vocab);
    for (uint32_t h = 0; h < ple_n_heads; h++) {
        ple_offsets[h] = h * head_vocab;
    }
    ms.add_kv(LLM_KV_PLE_LAYERS,            std::vector<uint32_t>({1}));
    ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,        ple_ngram);
    ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,   ple_per_ngram);
    ms.add_kv(LLM_KV_PLE_CONV_KERNEL,       uint32_t(4));
    ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS, std::vector<uint64_t>({3, 5, 7}));
    ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,      ple_offsets);
    ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,  ple_vocabs);
    ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,      uint32_t(0));   // < n_vocab (128)
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, ple_row_dim);
}
```

Fixture notes:

- `ple_n_heads = 8`, head vocab 1024 -> table rows = 8192 (multiple of 32; the
  loader sanity check needs `offsets[h] + vocabs[h] <= ple_rows`).
- `LLM_KV_EMBEDDING_LENGTH_PER_LAYER` = the PLE row dim (`ple_head_dim`). The
  loader maps it to `n_embd_per_layer` and requires `ple_conv_kernel > 0`.
- Multipliers are stored as uint64 and used in the host-side hash
  `mixed = t[p]*m[0] ^ t[p-1]*m[1] ^ ...`; small values keep `mixed` well
  inside int64 (token ids < 128, products < 2^47).
- EOS = 0 is in the random token range (`get_tokens` uses `n_vocab - 1`), so
  the EOS-reset path of the hash window is exercised.
- `LLM_KV_PLE_IMAGE_TOKEN_ID` is optional (token batches only in this test).

#### Loader accommodations required for the user path

`load_arch_tensors` and `create_tensor` currently assume a file-backed loader
whenever PLE is present:

1. `load_arch_tensors` (qwen4exp.cpp): `ml.require_weight(ple_name)` throws when
   `ml.files` is empty (the user path has no `weights_map`). Guard it:
   - files present: as today (`ple_w.tensor->ne[1]` + head-range sanity check);
   - files empty: derive `ple_rows` from the gguf metadata if the tensor is
     registered (`gguf_find_tensor(ml.metadata, ...)`) or from
     `ple_head_offsets[last] + ple_head_vocab_sizes[last]` rounded up to a
     multiple of 32; still run the head-range check.
2. `create_tensor` (llama-model-loader.cpp): in the `TENSOR_READ_LAZY` branch,
   guard the `require_weight` call with `!files.empty()`; in the user path the
   tensor is then a plain F32 tensor created from the `ne` argument and filled
   by `set_tensor_data` (no lazy bookkeeping -- there is nothing to lazy-read
   from). This is also the correct behavior for `llama_model_init_from_user`
   callers in general.

With those two guards, the qwen4exp PLE fixture runs through the existing
CPU-vs-device NMSE check and the save/load roundtrip. The **roundtrip leg is
file-backed**, so it exercises the real `TENSOR_READ_LAZY` mmap path (and, with
`model_params.n_lazy_buf_size > 0` set for the roundtrip load, the real managed
path end to end) -- and the bit-exact logits assertion then doubles as a
**determinism check between the user-path gather and the file-backed lazy or
managed gather**.

### 9.3 Real-model validation (the 192 GB box)

1. Graft the Q8_0 ngram into UD-Q4_K_XL (see `~/qwen38-flash-q4-q8.md`).
2. Same prompt/seed, `--lazy-mode auto` (mmap) vs `--lazy-buffer-size 10G`
   (managed): assert bit-identical logits for the first ubatches and identical
   sampled output for a longer run (both paths dequantize the same Q8_0 bytes
   with the same `to_float`).
3. RSS: `--lazy-buffer-size 1G` + long prefill; verify RSS stays under
   budget + model + slack (`/usr/bin/time -v`).
4. Fallback: `--lazy-buffer-size 0` behaves exactly as before.
5. Repeat on a constrained box (32 GB system RAM) to demonstrate the budget
   caps RSS where the mmap path would thrash.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Numerics diverge between mmap and managed paths | Both use `ggml_get_type_traits->to_float`; add the 9.3 determinism check plus the test-llama-archs roundtrip (bit-exact) assertion |
| User path (`llama_model_init_from_user`) breaks with PLE keys | Two loader guards (section 9.2): `require_weight` in `load_arch_tensors` and in `create_tensor`'s `TENSOR_READ_LAZY` branch become no-ops when `ml.files.empty()`; TENSOR_SKIP assert relaxed |
| `TENSOR_SKIP` side effects (counts, progress, "unused tensor" WARN) | Verified accounting in `create_tensor`/`init_mappings`; downgrade or silence the WARN for the managed case |
| Graph reuse broken (topology depends on mode) | Mode is fixed at load; `can_reuse` branches on the same condition as `build_inp_ple`; add a test that graph reuse fires across ubatches in managed mode |
| fd lifetime (loader is destroyed, model keeps the reader) | `dup()` at creation; reader owns the fd |
| Windows (`pread`/`dup`) | Gate managed mode to POSIX initially; fall back to mmap path; `_read`/`_lseeki64` follow-up |
| Multi-context lock contention | One lock per ubatch; microsecond hold; pinning + copy-out-of-lock as a follow-up if needed |
| Huge-prefill thrash (working set > budget) | Same as today's kernel behavior but bounded; optional streaming/prefetch follow-up |
| mmap path regressions | Managed mode is opt-in (`n_lazy_buf_size > 0`); default unchanged |

---

## 11. Implementation order

**Milestone 1 -- cache class (no llama.cpp integration):**
`src/llama-lazy-reader.{h,cpp}` + `tests/test-lazy-reader.cpp`. Fully testable
standalone.

**Milestone 2 -- plumbing:**
`llama.h` param + default, `common` (arg, common.h/cpp), `ml.lazy.buf_size`
at `llama.cpp:321`, `src/CMakeLists.txt`.

**Milestone 3 -- qwen4exp integration + user-path guards:**
`load_arch_tensors` (reader creation + TENSOR_SKIP branch + `files.empty()`
guard), `build_inp_ple` and `llm_graph_input_ple::set_input`/`can_reuse` (dual
path), `lazy_reader` member, `create_tensor` guards (`require_weight` in the
`TENSOR_READ_LAZY` branch, TENSOR_SKIP assert), stats hook in
`LLAMA_GRAPH_INPUT_DEBUG`.

**Milestone 4 -- tests:**
PLE fixture in `tests/test-llama-archs.cpp` (KV keys per section 9.2); the
roundtrip leg runs file-backed, so it exercises the real lazy (and, with
`n_lazy_buf_size > 0`, the real managed) path with a bit-exact logits check.
Run the full arch test suite to confirm no regression on other archs.

**Milestone 5 -- real-model validation:**
9.3 determinism + RSS checks on the grafted UD-Q4_K_XL + Q8_0 ngram file on
the 192 GB box; fix any divergence.

**Milestone 6 -- polish (optional):**
Windows fallback, silent TENSOR_SKIP variant, server option, env-gated stats
print, gemma4 adoption.

---

## 12. Follow-ups (out of scope here)

- **Post-load weight-page eviction**: after copying weights to VRAM, the mmap
  pages of the weight files are dead weight; `madvise(MADV_DONTNEED)` on those
  ranges would hand ~the Q4_K_XL file size back to the system (complements the
  managed ngram on constrained boxes).
- **Prefetch**: during prefill, `set_input` could issue the next chunk's
  `pread`s on a background thread (the rows for token t+1.. are computable in
  advance).
- **gemma4**: same `TENSOR_READ_LAZY` gather pattern (`build_inp_per_layer`,
  `src/models/gemma4.cpp:423`); the reader class is generic enough to reuse.
- **Smarter cache unit**: for tables where rows are smaller/larger, let
  `rows_per_page` be configurable instead of fixed 24.
