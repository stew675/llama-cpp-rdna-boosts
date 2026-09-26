# Prefill indexer relu + head-sum fusion (`idx-relu-sum`) — item 14 (2026-09-22)

**Status:** done, **default ON**, bit-identical.  Fork branch `gap-closing` @ **`ac391cf4f`**, exported as
[`patches/0013`](patches/0013-gap-closing-WIP-fuse-the-prefill-indexer-relu-head-s.patch).

## The gap

Session 5's profile had the `indexer` family at **1254.6 ms (ours) vs 664.5 ms (reference)**: the
reference's `idx_relu_sum_f32` (1536 calls / 364 ms) fuses the per-head relu and the head sum, while
our prefill ran a separate `unary_op<relu>` (559 ms) plus the head-sum CONT/ADD chain, reading the
block scores H times.

## The graph, and why the reference matcher cannot port verbatim

Our qwen4exp prefill score chain (`src/models/qwen4exp.cpp`, the default `GGML_QSA_SCORE_MEM` path,
"L2a") is:

```
sc    = mul_mat(pooled, q3)                       // [n_blocks, H*n_tps, n_stream]
sc    = relu(sc)                                  // <-- relu BEFORE the reshape (the L2a memory win)
sc    = reshape_4d(sc, n_blocks, H, n_tps, n_stream)
summed = cont(view(sc, h=0)) ; summed = add(summed, view(sc, h=1)) ; ...   // head sum in graph order
```

The reference's matcher anchors at a RELU whose **4-D** input is already `[n_blocks, H, nt, ns]`, with
the head views pointing at the relu.  Here the relu is one node earlier and 3-D.

The physical detail that makes a verbatim kernel work anyway: `ggml_reshape_4d` reinterprets
`[nb, H*nt, ns]` as `[nb, H, nt, ns]`, and the head views use that 4-D stride (`head stride = nb`,
`token stride = H*nb`).  Reading the **pre-relu** mul_mat output with those same strides gives exactly
the values the graph's views read, and the fused kernel reapplies `fmaxf` — idempotent and identical.

One trap: `ggml_view_3d` on the reshape produces a view whose `view_src` is the **relu** (ggml
collapses a view-of-a-view to the root), while its `nb[]` come from the **reshape**.  The matcher must
therefore check `view_src == relu` against the reshape's strides, not `view_src == reshape`.  (This
was the first cut's bug: the fusion silently did not fire — `comb` matched 96 combines but the
`IDX_RELU_SUM` debug never printed.)

## Files

| file | change |
|---|---|
| `ggml/src/ggml-cuda/indexer-score.cuh` | `ggml_cuda_idx_relu_sum_args` + declarations |
| `ggml/src/ggml-cuda/indexer-score.cu` | `idx_relu_sum_f32` — `fmaxf` + the same left-to-right add order; `ggml_cuda_op_idx_relu_sum` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_match_idx_relu_sum` (anchored at the relu, optional following reshape), the call in `ggml_cuda_try_fuse`, and the alloc dependency in `graph_optimize` |

The op reuses `indexer-score.cu`/`.cuh` (already in the build) rather than adding new files, so no
CMake reconfigure is needed.  Kill switch `GGML_CUDA_DISABLE_IDX_RELU_SUM=1`; `GGML_CUDA_IDX_RELU_SUM_LOG=1`
prints the first four firings.  Gated to RDNA3_5 like the reference (this campaign's box); the kernel
itself is arch-neutral and can be widened after a gfx1201/gfx1100 validation.

## Gates (qwen4exp IQ4_NL, gfx1151)

* **Bit-identical:** width probe W=1..8 + `row0_row1_hashes 1:268e0673300b7a33/…` unchanged,
  `width_purity=PASS (worst maxdiff 0)`; `plain == draft-mtp n3` byte-identical (`984263fb8e0f`, 434 chars).
* Fires at `nb=1024, heads=4, rows=4096`.

## Performance — `-b/-ub 4096`, `-p 8192,32768 -n 0 -r 3`

| | fused | unfused (`…DISABLE=1`) | Δ |
|---|---:|---:|---:|
| pp8192 | 1293.0 | 1293.2 | flat |
| pp32768 | **1264.5** | 1241.9 | **+1.8 %** |

The gain grows with context (the indexer work scales with `n_blocks`), matching the reference's 364 ms
`idx_relu_sum_f32` versus our 559 ms `unary_op<relu>` + adds.

## Follow-ups

* The `QSA_SCORE_BOUNDS` + `QSA_QUERY_STRIP` item (trim the prefill scorer to its visible blocks) and
  then `QSA_SCORE_WMMA` are the remaining item-8 follow-ups.
* The fusion is RDNA3_5-gated; widening it is a one-line change plus a gfx1201/gfx1100 width-probe run.
