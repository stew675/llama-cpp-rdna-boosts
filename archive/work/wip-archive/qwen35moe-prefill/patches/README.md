# Patch 0001: op timing wraps fused dispatches (ggml-cuda.cu)

Status: WIP backup only - NOT integrated. Applied in the `~/llama.cpp`
working tree (rdna-boosts branch) to enable the decode cost analysis;
keep it as a working-tree change or apply via `git apply` when needed.
Decide later whether to integrate into the patch set.

## What it does

The `GGML_CUDA_OP_TIMING=1` instrumentation (`ggml_cuda_graph_evaluate_and_capture`
in `ggml/src/ggml-cuda/ggml-cuda.cu`) previously recorded CUDA events only
around non-fused node dispatches. Nodes consumed by `ggml_cuda_try_fuse`
(fused kernels: GDN cache, topk-moe, RMS_NORM q8_1 fold, SSM conv fold,
multi-op chains, fused expert mmvq, ...) skipped the event pair entirely,
so their GPU time was invisible. On decode, the fused work is ~70% of the
step - the cost table was impossible without this.

The patch:
- records `op_ev0` on the stream before `ggml_cuda_try_fuse`,
- when `nodes_to_skip != 0`, records `op_ev1` and pushes
  `(node, i, fused=true)`,
- `op_nodes` becomes `std::tuple<const ggml_tensor *, int, bool>`,
- the report loop prefixes fused entries with `FUSED ` (keyed by the
  first node of the fused group).

## Usage

```sh
GGML_CUDA_OP_TIMING=1 llama-bench -m <model> -p 512 -n 64 -ub 2048 -r 1 --no-warmup -v 2>&1 \
  | grep -E "FUSED|op timing"        # fused kernels now visible as "FUSED <op> <name>"
```

Caveats (unchanged from the base instrumentation):
- disables CUDA graphs (direct-path eval; on decode the direct path is
  11.1 ms/step vs 10.8 ms graph - representative),
- streamed (secondary-stream) nodes still show ~0 on the main stream
  (their cost lands in the following main-stream op's elapsed time),
- per-kernel event overhead is ~4 us; subtract when comparing to wall
  time.

## Validation

Made the decode cost table in decode-phase1-results.md possible (FUSED
MUL_MAT_ID gate+up 37 us, down+shared-gate 53 us, etc.). No behavioral
change: the patch only adds event records when `GGML_CUDA_OP_TIMING` is
set.

## Patch 0002: mmvq short-K item-split (Phase 2 item 1)

See `0002-mmvq-short-k-item-split.md`. Working-tree change in
`ggml/src/ggml-cuda/mmvq.cu`. Decode +6-7% on qwen35moe (routed down
kernel 53.2 -> 37.7 us); test-backend-ops 2065/2065 pass; prefill flat.
The gate+up/down one-kernel merge (plan-fused-moe) remains a follow-up -
the decode data showed the down's latency is the K-split idling, which the
item-split fixes more cheaply than a cooperative-launch merge.
