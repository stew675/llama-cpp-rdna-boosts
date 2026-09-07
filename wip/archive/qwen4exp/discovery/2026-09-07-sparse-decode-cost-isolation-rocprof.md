# 2026-09-07 — sparse-decode cost isolation: rocprof decode-token windows, fused vs dense @ ~30K

Question (maintainer): the [1]/[2] fusion proved the gap is NOT in the pool->score
table generation.  Where IS the sparse decode time vs dense?  Method: rocprofv3
kernel traces of a 1-token decode at ~30K context (30K-token prompt, `-c 32768`,
3x R9700 bf16), fused (GGML_CUDA_QSA_INDEXER_SCORE=1) vs dense (LLAMA_QSA_OFF=1);
the last ~6000 dispatches = one decode token (fused window wall 23.4ms == 42.7 t/s).

## Results (per decode token, ~30K depth)

| piece | fused | dense | note |
|---|---|---|---|
| flash_attn_qsa vs flash_attn_tile | ~30 us/call | ~242 us/call | **sparse FA is ~8x cheaper per call at depth** (reads the selected cells only) |
| indexer_topk pipeline | 6 kernels/layer: radix histogram+select+count+write+scan+init, ~1.5-7 us/call each | none | the only O(n_kv) step left in sparse |
| rope_multi (float) calls | 62 | 18 | the indexer q-side/k-side per-token ropes |
| k_get_rows bf16 gather + extra norms/Cijk mms + copies | ~1 ms/layer-set | ~0 | the small-kernel auxiliary chain |

## Reading (why sparse decode still loses ~2.5 ms/token at 32K despite the FA win)

- The block-sparse FA is genuinely cheaper at depth (30K: ~8x per call) - the read-cap
  premise delivers.  The FA is NOT the tax; it is the sparse WIN.
- The tax is the SELECTION + machinery: the topk is O(n_kv) per layer per token (a
  radix-select over the whole 32K-element score vector - the step block-sparse was
  supposed to avoid), plus the per-layer small-kernel chain (q-side rope/norm/mms,
  gathers, copies).  At 32K these sum to roughly the FA's saving + ~2.5 ms.
- Structural model: dense attention per token ~= a·n_kv (tiled FA reads).  Sparse ~=
  b·n_kv (topk select, b << a, ~30x cheaper per element) + c·2051 (small FA) + fixed
  machinery.  Sparse wins when (a-b)·n_kv clears the fixed machinery - the measured
  32K crossover says that is past 64K (consistent with dense staying ahead to 64K on
  gfx1201 and the fused rows closing on halo).
- The two levers that remain: (1) the topk's O(n_kv) select (any cheaper top-2051-of-32K
  structure), (2) the per-layer small-kernel chain count.  The FA and the pool->score
  stack are done.

## Method caveats
- rocprof aggregates mix per-device rows (mirrored ops x3); absolute per-name sums need
  matched windows.  Per-call and per-layer-token costs are clean; total deltas have
  ~+/-1 ms noise from window alignment.
- The fused run also shows a ~10% slower prefill (1313 vs 1449 t/s pp): the sparse
  machinery costs prefill too (amortized over the ubatch).
