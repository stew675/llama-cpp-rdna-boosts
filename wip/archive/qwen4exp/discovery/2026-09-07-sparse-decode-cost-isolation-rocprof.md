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

## Reading (CORRECTED after the maintainer challenged the topk figure)

- The whole topk pipeline (init+histogram+select+count+write+scan, all 12 layers) is
  ~0.5 ms/token wall (per-call 1.5-7 us; its decode grids are small: histogram over the
  ~8K block scores, count/write over the ~90K cell expansion).  NOT 4-6 ms - that figure
  was wrong (conflated window artifacts + the whole aux chain).
- The block-sparse FA at 30K is ~0.16 ms/device/token vs dense's ~1.2 ms (flash_attn_tile
  ~242 us/call vs qsa ~30 us/call) - the FA is the sparse WIN.
- THE REAL FINDING - device-level accounting of the 1-token windows:
    fused: wall 23.4 ms/token, per-device busy ~16.1 ms  -> ~66-70% util, ~2000 disp/dev
    dense: wall 21.1 ms/token, per-device busy ~18.5 ms  -> ~88% util,   ~1613 disp/dev
  The sparse path does LESS GPU work (16.1 vs 18.5 ms busy) yet is slower: it wastes
  ~4.5 ms/token MORE in inter-kernel gaps.  Its per-layer structure is a long chain of
  small DEPENDENT kernels (store -> q-side -> score -> 6-stage radix topk -> sparse FA),
  each stage serialized (launch latency + syncs + mirrored-op cross-device waits), vs
  dense's few fat fused kernels (one tiled FA + the mms) at ~88% util.
- Conclusion: the sparse decode deficit is NOT any kernel's work (all small; the FA
  wins) - it is the CHAIN LENGTH + dependency gaps: ~2000 dependent small kernels/token
  at ~66% utilization.  This explains why [1]/[2]/[3] (which cut per-kernel work) moved
  nothing: they optimized busy time; the cost is the serialization between kernels.
- The lever that remains: FEWER, FATTER, less-serial kernels per layer (fuse the store /
  q-side / score / topk / FA per-layer chain toward one kernel), and/or better overlap.

## Method caveats
- rocprof aggregates mix per-device rows (mirrored ops x3); absolute per-name sums need
  matched windows.  Per-call and per-layer-token costs are clean; total deltas have
  ~+/-1 ms noise from window alignment.
- The fused run also shows a ~10% slower prefill (1313 vs 1449 t/s pp): the sparse
  machinery costs prefill too (amortized over the ubatch).
