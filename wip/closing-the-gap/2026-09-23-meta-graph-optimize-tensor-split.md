# 2026-09-23 — the meta-backend `graph_optimize` gap: markings now run under `-sm tensor`

**Work item 3 of [`gfx1201-closed.md`](gfx1201-closed.md) §12.3.**  Box: 3× Radeon AI PRO R9700
(gfx1201), ROCm 7.14; campaign branch `closing-gfx1201` (r13 + `beta/mmb-general` + the closing set).

## TL;DR

Under `-sm tensor` the meta backend owns the whole graph, so the scheduler **never calls a child
backend's `graph_optimize`**: the MMB **HC16 / DOWN16 / blk16 / res16** markings and *all* the fused
kernels' gallocr alloc deps registered there were inert (`MMB_OPT=0` under `-sm tensor` vs `60`
under `-sm layer`), and only the meta's own `moe_weighted_reduction` dep loop ran.  The meta backend
now forwards the child pass in two modes — alloc deps over the whole graph before allocation, and
markings per per-device subgraph after the simple tensors exist.  New closing patch **`0027`**
(`2af88bc06`); the 26-patch set re-applies on `r13-beta-baseline` and reproduces tree
**`803e6d908ade68b02a71af9d5d0cf605aec1382a`**.

## Why it was inert, and why the marks cannot simply be set on the original graph

The scheduler calls `graph_optimize` once per split **before allocation**, and the CUDA pass is
structural (no data pointers).  Under `-sm tensor` there is one split whose backend is the meta
backend, and the graph a CUDA child actually computes is built later, inside
`ggml_backend_meta_graph_compute`, as a per-device subgraph of **simple tensors**
(`ggml_backend_meta_buffer_simple_tensor`) — separate `ggml_tensor` objects from the scheduled
graph.  The MMB marks (`ggml_cuda_mmb_mark_bf16_only` / `_copy` / `_slot`) are keyed by
`const ggml_tensor *` and consumed at compute time by the producers (MMB GEMM `store_f32`, the MoE
reduction, `dsv4_hc_pre`, the fused norm/unary) and by the fused HC combine — all of which see the
**simple** tensors.  Setting the marks on the original graph would make every lookup miss.

## The change (`0027`)

`ggml_backend_graph_optimize_params` gains two flags, `marks_only` / `allocs_only` (both default
false — the single-backend scheduler is unchanged), and `ggml_backend_cuda_graph_optimize` is split
accordingly:

* `allocs_only` runs only the `add_alloc_dep` section and returns before the CUDA-graph tail;
* `marks_only` runs only the marking passes and returns before the alloc deps;
* the default runs both, exactly as before.

`ggml_backend_meta_graph_optimize` then forwards the child pass twice:

1. **before allocation**, over the whole scheduled graph with the scheduler's collector
   (`allocs_only`).  This registers every fused-kernel dep the CUDA pass knows — `idx_relu_sum`,
   `ple_conv`, `gdn_conv`, `norm_gated`, `moe_weighted_reduction`, and the **res16** residual
   dep — not just the MoE reduction the meta handled by hand.  The meta's own
   `moe_weighted_reduction` loop is kept as the backend-agnostic fallback (duplicates are deduped by
   the scheduler).
2. **after allocation**, per `(device, subgraph)`, with `marks_only` and `full_graph = NULL`.  The
   subgraphs are already built at that point (`bcj.cgraphs[i].cgraph_main`), so the marks land on
   the simple tensors the child compute sees.  The pass runs inside the `needs_rebuild` branch, once
   per graph topology, and **all** subgraphs of a graph are marked before any is computed — the
   condition `ggml_cuda_mmb_optimize_begin` relies on (it clears on the first optimize after a
   compute; later subgraphs accumulate).

`GGML_CUDA_MMB_MARK_LOG=2` therefore now reports `MMB_OPT` for every per-device subgraph under
`-sm tensor`.

### `full_graph = NULL` is correct under tensor split

In the single-backend path `full_graph` exists so a producer whose consumer lives in another
**scheduler split** (an eval-callback split) keeps its F32 output, because the per-compute activation
cache does not span computes.  Under the meta backend the marks are per `(device, producer/consumer)`
pair, and a marked producer and every consumer that reads through its BF16 activation cache are
always on the **same device** (the tensors are MIRRORED there; the split follows the GEMM output
dimension).  Callback splits are handled one level up: the scheduler splits at the callback and each
meta `graph_compute` is callback-bounded.  `has_eval_callback` is still propagated (conservative:
elision stands down if any callback exists), matching the single-backend behaviour.

### The one mark that needs an alloc dep

`LLAMA_HC_RES16`'s in-place residual stream needs the producer/consumer on different buffers, so its
`add_alloc_dep` calls must reach the gallocr.  They were moved out of the marking block into the
alloc-deps section so the `allocs_only` pass still registers them; the marking pass re-runs the
structural matcher for the marks.

## Validation (gfx1201, 3 GPU)

| check | result |
|---|---|
| `MMB_OPT` lines, 4B `q4_1`, `-sm tensor` | **0 → 390** (was 0 before the change) |
| `MMB_OPT` lines, 27B UD-IQ3_S, `-sm tensor` | **5418** |
| `MMB mark bf16_only` lines, qwen4exp IQ4_XS, `-sm tensor`, `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1` | **837** (was 0) |
| 27B UD-IQ3_S `-sm tensor` coherence (8K, seed 42) | `6073add19dac` — **matches the pre-change record** |
| 4B `q4_1` single-backend coherence | `9f9f41270c70` — matches the record |
| 35B-A3B UD-Q3_K_M `-sm tensor` `plain == draft-mtp` (8K, `-n 64`) | `5f6f93dd9b62` both — pure |
| `test-backend-ops -o MUL_MAT` | **1297/1297**, 0 FAIL |

## Caveats / follow-ups

* This is **infrastructure, not a win on gfx1201**: the three features it enables there
  (`0010` DOWN16, `0011` blk16/res16) were already measured as neutral/negative on discrete RDNA4
  ([`2026-09-23-gfx1201-lossy-prefill-transfer.md`](2026-09-23-gfx1201-lossy-prefill-transfer.md));
  `HC16` is RDNA3_5-gated.  The value is that a future `graph_optimize`-based marking now works
  under `-sm tensor`, and the fused kernels' alloc deps are complete.
* **gfx1151 (`halo`) is the machine where this pays**: HC16 is the gfx1151 win, and it was inert
  under `-sm tensor` there for exactly this reason.  Re-gate on `halo` (HC16 on, 3-GPU `-sm tensor`,
  qwen4exp) before claiming anything.
* Observed while validating: on the tested qwen4exp IQ4_XS graph the `hc_combine_norm` fused window
  was *rejected* by `ggml_can_fuse_subgraph_ext` (`LLAMA_HC_CN_DEBUG=1` shows the op-window
  mismatch), under **both** `-sm layer` and `-sm tensor` — so the 837 marks were consumed only by
  the other BF16 readers (norm/unary/MoE reduction), not the fused combine.  That is a property of
  the feature's dispatch on this model/build, independent of the meta forwarding; noted for the
  next HC-stream session.
* The forwarding is generic (`iface.graph_optimize`), so a non-CUDA child with a pass also runs; a
  child without one is skipped.
