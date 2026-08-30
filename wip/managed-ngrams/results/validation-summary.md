# VALIDATION RESULTS

All results measured on the real model:
`/models/Qwen3.8/Flash-Next/Q4_K_XL/` (Unsloth UD-Q4_K_XL, 4 splits, IQ4_NL
ngram, 28.8 GB table in split 00002 at offset 682,434,560).

Machine: **AMD Ryzen 9 9950X3D (16C/32T), 184 GB RAM, 3x Radeon AI PRO
R9700 (Navi 48 / gfx1201, 32 GiB VRAM each), ROCm 10.0.0 (gfx1201 build)**.

Builds: `build/` = CPU-only Release; `build-rocm/` = GGML_HIP=ON Release.

Method: `logitdump` (see `tests/`) writes sampled float logits; runs are
compared with `cmp` (byte-identical). Residency via `/proc/<pid>/smaps`
samplers; VRAM via `rocm-smi`.

---

## 1. Determinism — mmap vs managed

All byte-identical (58M-float comparisons for the 234-token runs).

| Run | Path A | Path B | Result |
|---|---|---|---|
| 234-token prompt, CPU | mmap | managed 4G | identical |
| 234-token prompt, CPU | mmap | managed 30G (all-resident) | identical |
| 234-token prompt, GPU layer-split | mmap | managed 4G | identical |
| 234-token prompt, GPU tensor-split (`--split-mode 3`) | mmap | managed 4G | identical |
| 1024 random tokens, GPU | mmap | managed 16M | identical |
| 16384 random tokens, GPU | mmap | managed 4M / 32M / 512M | identical |
| 16384 tokens (512-block x32), GPU | mmap | managed 4M / 32M | identical |

## 2. Ngram residency / lookup stress (16K random tokens, seed 42, GPU)

| Budget | Misses | Hits | Bytes read | Reload factor | tok/s |
|---|---|---|---|---|---|
| 4 MiB | 521,899 | 5 | 2.11 GB | ~503x | 610.7 |
| 32 MiB | 261,767 | 244 | 1.06 GB | ~32x | 617.1 |
| 512 MiB | 258,556 | 3,449 | 1.05 GB | ~2x | 604.4 |
| 32 MiB, repeated 512-block x32 | 11,186 | 250,831 (95.7%) | 45 MB | 1.3x | 574.6 |
| 4 MiB, repeated 512-block x32 | 519,551 | 2,177 | 2.10 GB | ~500x | 570.9 |

Interpretation:
- misses = `pread` from disk; hits = served from the arena.
- The reload factor tracks the working-set/budget ratio (~1 GB working set
  for 16K random tokens): 503x / 32x / 2x as budget grows 4M -> 32M -> 512M.
- The repeated window at 32 MiB (window ~= budget) gives a 95.7% hit rate:
  the clock LRU holds the hot pages and serves lookups from memory.
- At 4 MiB the window (32 MB) is 8x the budget: the LRU thrashes correctly
  (hit rate ~0.4%) and still returns bit-exact values.
- Throughput is flat across every budget (570-620 tok/s): the managed path
  is invisible next to the GPU decode cost.

## 3. Residency

| Quantity | mmap path | managed path |
|---|---|---|
| ngram file-backed resident (234-token) | 0.00 GiB (lazy, only ~2-4 MB faulted) | 0.00 GiB (never mmap'd; pread into anonymous arena) |
| ngram file-backed resident (16K tokens) | ~1.05 GiB (grows with working set) | 0.00 GiB |
| arena footprint | n/a | exactly the budget (anon delta between 512 MiB and 4 MiB budgets = 0.49 GiB) |
| system-RAM steady state (GPU, 234-token) | ~1.3 GiB | ~1.3 GiB |

Prefetch bug (found during validation, fixed): with `TENSOR_SKIP` alone the
ngram range was WILLNEED'd with the rest of the file -> 26.8 GiB resident,
peak RSS 145.3 GiB (managed 4G) vs 115 GiB after the `lazy_read::add_range`
fix. The table now stays cold until the reader actually preads it.

## 4. GPU offload (real model, `-ngl 99`)

- VRAM during run: ~78-80 GiB across the 3 GPUs (weights + KV + compute).
- System RAM steady state: ~1.3 GiB (weight mmap pages are unmapped after
  offload).
- Full `test-llama-archs` on ROCm: **608 configs OK, 0 FAIL** (each arch x
  each GPU + CPU + Meta tensor-split), including the qwen4exp PLE managed
  roundtrip on every single-GPU config (Roundtrip bit-exact).
- Full `test-llama-archs` on CPU: **128 configs OK, 0 FAIL**.
- `tests/test-lazy-reader.cpp`: all cases pass (Q8_0 page-boundary math,
  eviction sweeps, single-slot budget, page-size bump, threaded gathers,
  fd ownership, F32 rows); 20-run stress clean.

## 5. Bugs found and fixed (in this work)

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| Last-page byte-offset clamp underflow | wrong/abort on the partial final page | row-extent based clamp in `ensure_page` | 0001 |
| Same-gather eviction | assert `slot != -1` when working set > budget | pass-2 re-fault in `gather` | 0001 |
| F32 has no `to_float` trait | ctor abort for F32 tables | plain row copy | 0004 |
| Load-time WILLNEED prefetched the skipped table | 26.8 GiB resident despite budget | `lazy_read::add_range` excludes the range | 0005 |
| Meta backend crashed on host graph nodes (tensor-split) | `ggml_backend_meta_buffer_simple_tensor` assert on `model.input_embed (reshaped)` | host-buffer pass-through + host-safe `get_split_state` | 0007 |

## 6. Caveats / not yet done

- The Q8_0 graft (54.4 GB ngram) validation on real files is pending the
  Q8_0 ngram download; the managed path itself is type-agnostic and was
  validated on IQ4_NL (90 B/row) and Q8_0 (170 B/row, unit test).
- Real-model validation on the target unified-memory box (Strix-Halo-style
  carve-out) still to be re-run there.
- Tensor-split + managed combination is validated via logitdump
  (`--split-mode 3`); test-llama-archs skips the roundtrip for tensor-split
  by design.
