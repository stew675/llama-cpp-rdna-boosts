# Item 3 result: the router->topk fusion does NOT pay off (experiment reverted)

## What was tried

The decode router (`ffn_moe_logits` MUL_MAT [2048x256], F32 weights, 19
us/layer) + the top-k kernel (~7 us) looked like the cleanest Item 3
fusion: fold the logits matmul into the top-k kernel, each lane computing
its expert's dot, skipping the logits write/read round-trip and one launch.

Implementation (reverted, not in the tree):
- `topk_moe_cuda` gained `router`/`type` template params: per-lane
  expert dot (F32 plain dot, or vec-dot over a q8_1 shared quantize for
  the k-quants), then the existing softmax/argsort path.
- try_fuse: a forward scan from the router MUL_MAT to the gating op, with
  the topk-moe fusion args/checks replicated (decode-only gate).

## Why it failed

The fused kernel serialized the 256 expert dots into ONE warp (8 rounds of
32 lanes, each lane a 2048-element dot). The parallel mmvq router uses
hundreds of warps. Measured:

| variant | fused router kernel | decode t/s |
|---|---|---|
| mmvq router + topk (status quo) | 19 + 7 us | 98 |
| fused, naive loop (register spill) | 1100 us | 28.8 |
| fused, 8-accumulator ILP | 320 us | 56.7 |

Even at 320 us the fused kernel is ~17x slower than the mmvq it replaces.
The launch savings (~5 us) are dwarfed ~60x by the serialization. A
correct fusion would need the logits computed with mmvq-style parallelism
(256 blocks) then reduced to the top-k warp - a grid-wide dependency that
needs a barrier or a second kernel, i.e. no launch savings at all.

## Conclusion for Item 3

- The router/qkv/shared-expert kernels are launch/latency-bound; the
  fusions that would remove them cost more than the launches they save.
  This is now measured, not assumed.
- The decode win stays with the mmvq short-K item-split (patch 0002,
  +6-7%). The remaining decode overhead is the ~900 kernels/step - the
  elementwise tail - which the existing multi-op fusions already collapse;
  further fusion candidates measured as net-negative on gfx1201.
- Item 2 (short-K mmvq config tuning) was covered by patch 0002's
  group-filling rpb (cap 8 measured best); the shared-expert kernels
  (shexp_down_gate, ssm_gate_beta) have the same K-split idle structure
  but are launch-bound, so mapping changes there also measured flat.

Recommendation: stop the decode fusion direction; the data says the
current fused machinery (worth ~24%) + the item-split (+6-7%) is near the
practical ceiling for the kernel-level decode work on this stack. The
remaining big lever is batch-level (multi-token decode / speculation
throughput), which is an architecture decision, not a kernel tweak.
