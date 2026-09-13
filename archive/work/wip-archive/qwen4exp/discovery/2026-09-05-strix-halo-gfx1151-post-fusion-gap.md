# Strix Halo (RDNA3.5 / gfx1151) — post-fusion A-vs-B depth-0 pp gap (WS4 done)

Date: 2026-09-05 (run from the 2026-09-05 session handoff)
Machine/build/model/protocol: identical to
`2026-09-06-strix-halo-gfx1151-ws4-hc-fusion-gates.md` — same session,
same warm page cache, same ladder command. A = `~/llama.cpp` `qwen4exp`
tip `248e47704` (WS4 fusions DEFAULT ON); B = `~/strix-llama.cpp`
`c7af5c6c2` (halo-box community build, upstream-derived, NO QSA sparse
op). Both built with the canonical `~/bin/build-llama-rocm-714`;
llama-bench `-ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16
-ctv f16 --load-mode none`, depth-0, prompts DESCENDING in one process
(pp16384 first = warm clock). IQ4_XS UD 87.24 GiB, non-MTP.

## Depth-0 pp t/s (mean ± stdev, r3, warm clock) — A (fusion ON) vs B

| pp | A (248e47704, fusion ON) | B (c7af5c6c2) | gap B/A |
|----|-------------------------:|--------------:|--------:|
| 512 | 315.79 ± 4.01 | 647.40 ± 26.65 | 2.05x |
| 1024 | 354.42 ± 6.76 | 723.90 ± 12.78 | 2.04x |
| 2048 | 349.52 ± 2.33 | 771.28 ± 9.09 | 2.21x |
| 4096 | 437.65 ± 2.33 | 737.37 ± 8.95 | 1.68x |
| 8192 | 497.37 ± 0.82 | 679.18 ± 1.28 | 1.37x |
| 16384 | 527.79 ± 5.28 | 600.90 ± 1.71 | 1.14x |

WS1 (pre-fusion, 2026-09-05) gap for reference: 2.02 / 2.07 / 2.25 /
1.77 / 1.46 / 1.23.  B today reproduces WS1's B within ~1% at every row
(600.9 vs 605.4 @16k; 771.3 vs 782.8 @2k), so B is a stable reference
across sessions.

## Reading

- The fusion moved the LONG rows decisively: pp16384 gap 1.23 -> 1.14,
  pp8192 1.46 -> 1.37, pp4096 1.77 -> 1.68 (A fusion-on is +7-8% over
  A fusion-off on those rows this session). At pp2048 the fusion is
  +8.8% same-session (349.5 vs 321.2) yet the gap (2.21x) is barely
  moved from WS1 (2.25x) because A's pp2048 absolute row is volatile
  across sessions (±8%: WS1 348.0 / 09-06-late 342.8 / today 349.5
  fusion-on; fusion-off 321.2 today vs 326.7 09-06-late): short single-
  ubatch rows are clock/boost sensitive on this APU. B's pp2048 row is
  NOT volatile (771 today vs 783 WS1).
- Remaining gap is now concentrated at pp512-4096 (still ~1.7-2.2x) and
  shrinks to 1.14x by pp16384. Per the WS1/WS2 attribution the leftover
  components at those sizes are: (a) expert i-quant mmq vs B's
  mul_mat_q_routed_compact (~1.2-1.5x on ~20-25% of the row), (b)
  shallow-context QSA indexer+sparse-kernel overhead vs B's dense
  shortcut (no indexer/top-k below the selection width), (c) residual
  per-ubatch elementwise/routing tail (A still ~1960 elementwise-class
  dispatches/ubatch vs B's fused ~1550 + hc_*). Follow-ons: WS3 #2
  (dense shortcut below the selection width — attacks (b) and part of
  the shallow-row fixed cost) then WS3 #3 (routed-compact MoE mmq for
  i-quants, RDNA4-gated — attacks (a)).

## Status

- WS3 #2 (dense shortcut in A's `build_layer_attn`) implementation +
  validation follows; results appended here when measured (same
  protocol, same-session A/B).

## WS3 #2 follow-on (2026-09-06, same session): QSA dense shortcut below the selection width

Implemented (port of halo-box's `LLAMA_QSA_DENSE_SHORTCUT` heuristic;
A commit `151798ed2`, OPT-IN default OFF): while
`n_kv <= indexer_top_k + ratio - 1` (= 2051 here; indexer_top_k 2048,
ratio 4 on the 12 of 48 QSA layers) every cell is selected, so the layer
attends DENSE (`build_attn`) and only caches the ubatch's raw indexer
keys (store-only `build_qsa_store_k` + a minimal `llm_graph_input_qsa_k`
holding just `k_idxs`, mirroring B's split). Past the budget the indexer
scoring + QSA sparse kernel run exactly as before.

Numerics: below the width the shortcut path is text-identical to A's
pre-existing `LLAMA_QSA_SPARSE_FA=0` masked-dense path (same attention
compute; the shortcut skips the redundant all-cells mask edit). The
difference from the sparse-kernel default is the known dense-vs-sparse
kernel signature (15.708 vs 15.973 pp-last top0 at the 7-token probe),
an env-selectable regime A already ships. `LLAMA_QSA_DENSE_SHORTCUT=0`
forces the old selection path (byte-identical known-good) either way.

### Measured (llama-bench, IQ4_XS UD, warm-clock, same build, fusion ON)

| test | shortcut ON | shortcut OFF (=0) | Δ |
|------|------------:|------------------:|--:|
| tg128 @0 | 24.53 ± 0.11 | 23.54 ± 0.08 | **+4.2%** (intended shallow-decode win) |
| pp512 @0 | 337.9 ± 3.1 | 326.7 ± 4.2 | +3.4% |
| pp1024 @0 | 376.1 ± 5.3 | 366.2 ± 5.0 | +2.7% |
| pp2048 @0 | 379.7 ± 1.2 | 363.5 ± 1.4 | **+4.5%** |
| pp2048, ub 4096 (single chunk) | 324.0 ± 4.2 | 327.1 ± 3.9 | +1% |
| pp4096 @0 | 367.3 ± 1.7 | 444.3 ± 1.5 | −17% (artifact) |
| pp8192 @0 | 354.8 ± 2.4 | 499.1 ± 2.4 | −29% (artifact) |
| pp16384 @0 | 334.2 ± 9.6 | 525.0 ± 3.2 | −36% (artifact) |
| pp2048 @ d12288 | 291.7 ± 2.4 | 294.7 ± 0.9 | flat (sparse-only) |
| pp2048 @ d32768, tg @ 12k/32k | unchanged | — | flat |

Real-path check (llama-cli, same crossing prompt ~5000 tok + 40 gen):
shortcut ON pp 259.9 t/s vs OFF 258.8 t/s — NO regression on the
llama_decode/server path; the llama-bench deep-row penalty does not
reproduce there.

### OPEN: llama-bench-only multi-ubatch artifact (root cause pending)

The pp4096+@0 rows are slower in llama-bench with the shortcut ON even
though (a) the executed GPU kernels are identical-or-smaller (rocprof
kernel traces: pp2048-on = pure dense FA + no indexer kernels, 7769
calls / 6.47 s vs 8591 / 6.90 s off; pp8192 traces identical except the
one dense chunk) and (b) llama-cli/llama-server crossing prefills show
no penalty. Suspect: a host-side graph-lifecycle cost in this fork's
hybrid-memory machinery when one context mixes store-only ubatches
(n_kv <= 2051) with scoring ubatches; appears at r1 (not a rep/
memory-clear amortization effect); grows with ubatch count (~−17% @2
chunks to −36% @8 chunks). The separate store-only input object
(`qsa_k_inp`) was suspected; a unified-input variant crashed (the full
QSA input's unused cell_blk/bias tensors get no buffer in store-only
graphs -> NULL-buffer assert in set_input_qsa), confirming B's minimal
input split is required. Not resolved this session.

### WS3 #2 status
- Implementation + numerics + perf characterization DONE; llama-bench
  artifact OPEN. Default OFF (opt-in `LLAMA_QSA_DENSE_SHORTCUT=1`; =0 or
  unset = known-good byte-identical) pending a root cause + maintainer
  adjudication (B defaults ON). Depth gates unaffected (shortcut never
  fires past the width).
- WS3 #3 (routed-compact MoE mmq for i-quants, RDNA4-gated) still
  pending (next session): the leftover gap at pp512-4096 is now
  dominated by the expert i-quant mmq vs B's `mul_mat_q_routed_compact`
  (~20-25% of the shallow rows per the WS1 attribution) plus the
  per-ubatch routing/reduction tail.
