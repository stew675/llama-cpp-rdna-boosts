# 0022: LLAMA_ROUTE_DUMP expert-routing instrumentation (scratch, Phase-0)

Env-gated capture of the per-layer moe top-k routing (expert ids + softmax
weights) for every ubatch eval, for the Phase-0 expert-tier measurement.

## What it does
- `LLAMA_ROUTE_DUMP=<path>`: append a text record per (ubatch, layer):
  `R <step> <n_tokens> <il> <n_eu> # meta... <ids> <weights>`
  with ids = the top-n_eu selected experts (token-major) and weights = their
  raw softmax values.  A `U` line records the ubatch size + capture count.
- Capture: in `graph_get_cb`, when the graph-build cb fires for
  `ffn_moe_topk` / `ffn_moe_weights`, record the tensor pointers per layer.
  Cleared before the REAL graph build in process_ubatch (the cb also fires for
  reserve builds).
- Keep-alive: the argsort parent of the topk view and the weights tensor are
  marked `ggml_set_output` so their buffers are not reused before the
  post-ubatch readback (the intermediate-buffer-reuse bug that initially made
  the ids garbage).
- Readback: after each ubatch in the decode loop, sync + read the full
  contiguous argsort parent, take the first n_eu entries per token column.

## Files
- src/llama-context.h: 3 mutable capture vectors + dump file/step members
- src/llama-context.cpp: capture in graph_get_cb; clear before real build;
  per-ubatch readback+dump in the decode loop; lazy file open in decode

## Usage / results
Applied on the scratch tree (base 7c7d9eeff).  Analysis scripts + full
findings: wip/qwen4exp/PHASE0_ROUTING.md.  Excluded from all beta/wip feature
patches - scratch diagnostic only.  Revert with `git checkout src/...` or drop
the commit.
