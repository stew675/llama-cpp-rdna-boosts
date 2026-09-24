# `0016` `QSA_SCORE_WMMA` — ported to RDNA4 (gfx12), default ON (2026-09-23)

**Status:** DONE.  Ported and folded into
[`patches/0016`](patches/0016-gap-closing-WIP-fuse-the-QSA-prefill-indexer-score-Q.patch); the
25-patch set re-applies to tree **`95f916a8e015efd68ee44bcea7620187ebc70019`** (25/25 `git am`).
This is a real gfx1201 prefill win, not just non-breakage.

## What `0016` was

`0016` fuses qwen4exp's QSA prefill indexer score into one `GGML_OP_LIGHTNING_INDEXER`
(`sum_h relu(q_h·k) · w_h + mask`) instead of the `mul_mat + relu + head-sum` chain.  It shipped a
hand-written 4-head/128-dim WMMA kernel `qsa_indexer_wmma16_keyreg` for **RDNA3_5 (gfx1151)**; the
arch gate (`supports_indexer4` -> `indexer4_arch_enabled`) left RDNA4 on the **generic vec fallback**.

## The port

`qsa_indexer_wmma16_keyreg` does **not** hand-roll the gfx11 fragment layout — it uses the
`ggml_cuda_mma` `tile`/`mma()` abstraction (unlike `mmb.cu`/`fattn-qsa3.cu`, which needed the beta's
gfx12 fragment shims).  `AMD_WMMA_AVAILABLE` already covers RDNA4 and `mma.cuh` already selects the
`_gfx12` builtins + accumulator map, so the port is three small changes:

1. **A/B layout per arch.**  RDNA3's A/B tile is `DATA_LAYOUT_I_MAJOR_MIRRORED` (a gfx11-only
   layout, no RDNA4 device code); RDNA4 uses `DATA_LAYOUT_I_MAJOR` ("two runs of four" per lane).
   ```cpp
   #if defined(RDNA3)
       using AB=tile<16,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED>;
   #else
       using AB=tile<16,8,half2,DATA_LAYOUT_I_MAJOR>;
   #endif
       using C =tile<16,16,float,DATA_LAYOUT_J_MAJOR>;
   ```
   (The kernel body indexes the fragments through `AB::get_i/get_j`/`C::get_i/get_j`, so it adapts
   with no other edits.)
2. **Compile gate** `#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)` ->
   `&& (defined(RDNA3) || defined(RDNA4))`.
3. **Enable RDNA4** in `indexer4_arch_enabled`: default **ON**, with
   `GGML_CUDA_LIGHTNING_INDEXER4_GFX1201=0` forcing the generic fallback for A/B.  (The RDNA3_0
   opt-in `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100` is untouched.)

## Correctness

* **`LIGHTNING_INDEXER` 225/225** on gfx1201 (RAM: the 4-head/128-dim cases —
  `test_lightning_indexer(128, 4, …)` — now take the WMMA path and still match the CPU reference).
* **Width purity PASS** (`qwen4exp IQ4_XS`, P=1024, both arms): `row0=2a795cb1f0ee16f2`,
  `worst maxdiff 0`.
* **Intra-build purity** `plain == draft-mtp --spec-draft-n-max 3` byte-identical:
  **`90069b3ed9c4`** (753 chars, 8K, q8_0 KV, `-sm tensor`).
* The kernel is a **prefill re-baseline** vs the generic fallback: text `90069b3ed9c4` (WMMA) vs
  `36019732357e` (generic) — the same class as the `LLAMA_QSA_SCORE_WMMA=0` A/B (f16 WMMA vs vec
  accumulation order), and purity holds *within* the arm.

## Performance (gfx1201, the point)

qwen4exp IQ4_XS, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 4096`, `llama-bench -r 5`, interleaved rounds
(warm page cache — the first cold run inflated pp8192 with ±121 t/s noise):

| depth | generic fallback | RDNA4 WMMA | Δ |
|---|---:|---:|---:|
| pp8192  | 3032.5 | 3083.0 | **+1.7 %** |
| pp16384 | 3139.98 | 3239.3 | **+3.2 %** |
| pp32768 | 3038.8 | 3237.0 | **+6.5 %** |
| pp65536 | 2798.8 | 3144.1 | **+12.3 %** |

The win grows with depth (the indexer score runs over `n_kv` blocks), matching the gfx1151 record's
shape (+1.0 % @pp32768 there, but the RDNA4 win is much larger because the generic vec fallback was
the RDNA4 baseline).  Both deep arms were stable (±0.2-0.5 %).

## Verdict

`0016`'s WMMA kernel **does transfer to gfx1201 and is a substantial prefill win** — the first
gfx1151-gated closing feature that does.  Enable by default (done); the lossy-text re-baseline is
accepted as the same class already documented for the fused op.
