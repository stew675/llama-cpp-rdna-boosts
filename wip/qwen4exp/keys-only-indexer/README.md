# qwen4exp VRAM investigation — keys-only QSA indexer cache

Date: 2026-09-10 · Author: pi session · Status: **WIP experiment, NOT part of the delivery**

## Context

Qwen3.8-Flash-Next (qwen4exp, 176.94 B params, IQ4_XS, 3× R9700/gfx1201, ctx 204800,
`--cache-type-k/v q8_0`, fa, `--split-mode tensor`, `--parallel 1`) reported near-full VRAM
(~88.6 GiB of 95.6).  User suspected the KV cache was ~2× qwen35's.  Findings:

1. **KV is not the problem.**  Dense-attn KV = 2550 MiB (12 layers only — the 1-in-4 dense
   layers of 48; log `204800 cells, 12 layers`), QSA indexer KV = 956 MiB (12 layers),
   recurrent state = 113 MiB fixed.  ~3.6 GiB total, ~4% of VRAM.  Per-token geometry is
   comparable to (or cheaper than) qwen35: 2 KV heads × 256 vs qwen35-4B's 4 × 256.
2. **Weights dominate, PLE stays host-side.**  File = 87.24 GiB but
   `per_layer_token_embd.weight` (PLE, 27,465 MiB) is host/disk (`lazy read enabled`);
   GPU-resident weights ≈ 62.6–64 GiB (i-quant block/align padding), partitioned evenly
   (~21.4 GiB/GPU on 3 GPUs; 2-GPU probe: 31,321.59 MiB/GPU; 1-GPU probe OOM'd trying one
   61,222 MiB buffer).  User's ~63 GB figure is accurate.
3. **The "missing" ~8 GiB/GPU above weights** (per-GPU rocm 29.38–29.73 GiB vs weights
   ~21.4): KV+state ~1.2 + **compute/graph buffer ~6.69 GiB/GPU** + AR ~0.05.  The compute
   buffer is a **full per-GPU replica** (2-GPU probe died on `allocating 2652.07 MiB on
   device 0: cudaMalloc failed` after weights fit).  It splits into a **ctx-independent,
   ubatch-bound base (~2.65 GiB/GPU at ctx 8192, ubatch 2048; MoE 512-expert + graph
   workspace)** and a ctx×ubatch-proportional term (f32 attention/QSA staging ~4 GiB/GPU at
   ctx 204800).  The earlier "missing ~2.5 GiB/GPU" the user spotted = that ctx-independent
   base (invisible in ctx 8192→204800 deltas).
4. **ubatch lever** (ctx 204800, all else equal; pp20480 from llama-bench, tg256 at depth
   20480, -r 3):

   | ubatch | Meta compute buf/GPU | total VRAM | pp20480 t/s | tg256 t/s |
   |---|---|---|---|---|
   | 2048 | 6690 MiB | 88.6 GiB | 2509.6 ± 7.0 | 50.69 ± 1.4 |
   | 1024 | 3347 MiB | 78.8 GiB | 2207.7 ± 1.3 (−12%) | 50.70 ± 1.4 |
   | 512  | 1725 MiB | 74.0 GiB | 1691.3 ± 0.7 (−33%) | 50.68 ± 1.4 |

   tg unaffected (parallel-1 decode is single-token steps).  Prefill is the ubatch casualty,
   as expected.

## The keys-only fix

The QSA indexer store (`llama_memory_hybrid_idx::mem_idx`) was built as a generic
`llama_kv_cache`, which allocates a V tensor unconditionally (`has_v = !is_mla`), sized from
the model's value_length (256 dims).  qwen4exp only ever issues K-side ops against it
(`cpy_k`/`get_k`/`get_rows` in `build_qsa_store_k` / `build_qsa_top_k`); the sparse attention
reads actual values from the *dense* layer cache.  The 256-dim V store (272 B/token/layer at
q8_0) is never written, never read — and was **triplicated per GPU** (measured reclaim
637.5 MiB × 3 ≈ 1.9 GiB total).

### Patch

`0001-keys-only-qsa-indexer-cache.patch` (62 lines, 3 files) — applies to rdna-boosts
`e2380eb67` (fork point 9113cc188):

- `src/llama-kv-cache.{h,cpp}`: new ctor param `bool v_enabled = true`
  (`has_v = !is_mla && v_enabled`).  Null-V caches already work throughout (MLA precedent;
  `size_v_bytes`, stream copies, state_read/write all null-guard V).  V-side ops
  (`get_v`/`cpy_v`/`type_v`) must not be issued against a keys-only cache.
- `src/llama-memory-hybrid-idx.cpp`: pass `v_enabled = false` for the indexer store.

### Validation (2026-09-10, build-rocm, 3× R9700)

- Init log: indexer cache now `size = 318.75 MiB (204800 cells, 12 layers), K (q8_0):
  318.75 MiB, V (q8_0): 0.00 MiB` (was 956.25 MiB).
- Total VRAM @ ctx 204800 / ubatch 2048 / q8_0: **88.58 → 86.70 GiB** (−1.9 GiB; V was
  replicated per GPU).
- Coherence: llama-cli same-seed (seed 42, temp 0, n 24) output **byte-identical** baseline
  vs keys-only (modulo the t/s status line).

### Not yet validated (before any delivery consideration)

- MTP/draft context, state save/load round-trip (ctx checkpoints, prompt cache), multi-slot
  (`--parallel > 1`, `--no-kv-unified` with streams), `--ctx-checkpoints` rollback, kv-unified
  mode, non-q8 KV types (f16/bf16/f32), and a decode-at-depth perf check.  The null-V paths
  are shared with MLA so risk is low, but the block-14 amendment protocol applies.

Working keys-only build: `/tmp/bin-keysonly/` (llama-cli/llama-server + impl .so; fork tree
restored canonical afterwards).

## Files

- `0001-keys-only-qsa-indexer-cache.patch`
- raw logs: `/tmp/keysonly.log`, `/tmp/bench-ub{2048,1024,512}.log`, `/tmp/qwen4exp-alloc.log`
  (baseline ctx-204800 ub-2048), `/tmp/ub1024mem.log`, `/tmp/ab-base.txt` vs `/tmp/ab-keys.txt`

## Addendum (2026-09-10, same session): validation matrix results

Keys-only build (/tmp/bin-keysonly, fork rdna-boosts e2380eb67) vs baseline, 3× R9700:

| test | result |
|---|---|
| perf parity (pp20480/tg256, ub 2048, r3) | pp 2497.4 vs 2509.6 (−0.5%), tg 50.65 vs 50.69 — identical within noise |
| KV types bf16 @ ctx 204800 | loads + completions clean; indexer V = 0 |
| KV type f32 @ ctx 204800 | **VRAM capacity abort** (f32 dense KV 9.6 GiB > headroom on 3×32 GiB) — not a keys-only issue; f32 @ ctx 16384 runs clean |
| parallel 2 (n_stream 2), non-unified | clean, both slots complete |
| parallel 2 + --kv-unified | clean |
| prompt-cache / ctx-checkpoint state round-trip | checkpoints created/restored/erased/superseded, prompt saves, correct gens, "graphs reused = 7", exit 0 |
| ctx-checkpoints 64 + 4600-token gen | clean (checkpoint 1 of 64 created; see note below on gen length) |
| MTP draft (--spec-draft-model, draft-mtp) | acceptance 0.741 (43/58), 20/20 calls accepted, correct output |
| f16 KV coherence A/B | **byte-identical** (seed 42, temp 0) |
| ubatch 1536 (non-power-of-2, non-divisor of batch 2048) | works fine in inference (divisibility assert is training-only); ragged 1536+512 graphs make it slower than ub1024 at same graph count (see below) |

Notes:
- The long-gen (T7) stopped at 609 tokens on <|im_end|> despite n_predict 4600, so the
  4096-min-spacing checkpoint save was not crossed in that run; checkpoint save/restore
  was nonetheless exercised heavily by the default per-request slot checkpoints in every
  server test (create/restore/erase/supersede, size 112.571 MiB, incl. the keys-only
  indexer store in the state blobs).
- ubatch has NO power-of-2 / divisibility requirement for inference (only ubatch <= batch
  is clamped; the `n_batch % n_ubatch == 0` assert lives in the training-only opt path).
  For throughput, prefer ub that divides n_batch to avoid ragged tail graphs.

## See also

The full implementation plan for the follow-on QSA memory reductions (mask derivation, score
chunking, block-bias), plus the session environment/repro notes and the packaging/rollout steps for
this patch, is in `../qsa-memory/HANDOVER.md` (2026-09-10).
