# Upstream-PR proposal — llama: keys-only KV caches (drop the dead indexer V buffer)

Status: PREPARED 2026-09-10.  Verified against **unadulterated master** (`9cf3bf256`): the patch applies
clean (`git apply --check`), the indexer KV buffer drops to keys-only on a real qwen4exp model, the
same-seed output is byte-identical, and the allocator tests pass (see Validation).  Not yet filed.
Patch file: `UPSTREAM-PR-kv-cache-keys-only.patch` (3 files, +10/−5:
`src/llama-kv-cache.{cpp,h}`, `src/llama-memory-hybrid-idx.cpp`).

## TL;DR

The Qwen3.8-Flash-Next (qwen4exp, `llama_memory_hybrid_idx`) **indexer KV cache is keys-only in
practice**: every graph that touches it uses `build_input_k_idxs` / `cpy_k` / `get_k`, and no V-side
op (`cpy_v`, `get_v`, `set_input_v_*`) is ever issued against it — the lightning indexer scores blocks
of *keys*.  Yet the cache is constructed like an attention cache, so a V tensor is allocated, written
into the checkpoint format, and never read.  Measured on master with the actual model (CPU, ctx 8192,
F16 KV): the indexer cache occupies **72.00 MiB (K 24.00 + V 48.00)**; with this patch it is
**24.00 MiB (K 24.00, no V)** — a third of the cache for nothing.  On the fork's reference
configuration (3× R9700, ctx 204800, q8_0 KV) the same cache is **956.26 MiB/GPU without the patch and
318.76 MiB/GPU with it** (−637.5 MiB/GPU, i.e. −1.9 GiB per box across the three GPUs).

Note the shape: the indexer store overrides the *key* head dimension to the indexer's
(`hparams_idx.n_embd_head_k_full = indexer_head_size`, 128 for these GGUFs) but inherits the model's
*value* head dimension (256), so the dead V is **twice the size of the K it never accompanies**.

## Where the waste is

* `src/llama-memory-hybrid-idx.cpp` (~L60): the indexer cache is built with
  `new llama_kv_cache(model, hparams_idx, type_k, type_v, ..., "idx_")` — same constructor as an
  attention cache, so `has_v = !is_mla` is true and a V tensor is created for every layer.
* `src/llama-kv-cache.cpp` (~L230): `const bool has_v = !is_mla;` — the only keys-only case that
  exists today is MLA.
* consumers of the indexer store: `src/models/qwen4exp.cpp` uses `get_n_kv()`,
  `build_input_k_idxs()`, `cpy_k()` and `get_k()`; `src/llama-graph.cpp` (MSA) uses
  `set_input_k_idxs` on its idx store.  There is no `cpy_v`/`get_v` on any idx store in the tree.

## Change

Give `llama_kv_cache` an explicit `bool v_enabled = true` constructor parameter (`false` ⇒ no V tensor
is allocated, and no V-side op may be issued against the cache) and pass `false` for the indexer store.
Default keeps every existing caller unchanged.

## Validation

Upstream master `9cf3bf256`, CPU build (`cmake -B build -DCMAKE_BUILD_TYPE=Release`, no GPU backend),
Qwen3.8-Flash-Next UD-IQ4_XS (the real 3-shard GGUF), `-ngl 0 -c 8192 -n 1 -v`:

| build | indexer cache | K | V |
|---|---|---|---|
| master | 72.00 MiB | 24.00 MiB (f16) | **48.00 MiB (f16)** |
| master + this patch | **24.00 MiB** | 24.00 MiB (f16) | **0.00 MiB (none)** |

* same-seed generated text (`-p "The capital of France is" -n 16 --seed 42 --temp 0`) is
  **byte-identical** with and without the patch.
* `test-alloc`: all cases PASSED.  `test-batch-alloc`: 0 failures, 0 exceptions, 0 skipped.
* fork (3× R9700 gfx1201, ctx 204800, ub 2048, `-ctk/-ctv q8_0`): indexer KV 956.26 → 318.76 MiB/GPU,
  box 88.58 → 86.70 GiB (the store is replicated per GPU); the fork's broader matrix (bf16/F32 caches,
  `--parallel 2`, unified cache, prompt-cache/checkpoint round-trips, MTP acceptance 0.741, F16
  byte-identity) also passed — the fork has carried this change since 2026-09-10 as block-15 win W3.

**Not validated here / what to check before filing:** the upstream GPU (HIP/CUDA) configuration and
the flash-attention `v_trans=false` layout (this box has no upstream HIP build); other architectures
with an idx store (MSA/minimax-m3), non-unified/multi-sequence caches, and state
save/restore (`--prompt-cache`) on a keys-only cache.  The fork evidence covers those shapes on its
own tree, but the upstream code paths differ enough that they should be re-run on the PR branch.

## Notes for filing

* Review-sensitive points: the `v_enabled` default (`true`) so nothing else changes; that no V-side op
  can reach a keys-only cache (a `GGML_ASSERT` is tempting but the cache is also used for
  `state_read`/`state_write`, so the invariant is enforced by construction — the caller is the only
  place that knows); and that a keys-only cache has to keep working for prompt-cache/checkpoint I/O.
* Suggested extra test: a `tests/` case (or an assert in `get_v`/`cpy_v`) that a keys-only cache never
  gets a V op, plus a state round-trip on a keys-only cache.
* Base: applies clean to `9cf3bf256` (master at preparation time) and to `9113cc188`.
