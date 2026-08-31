# qwen35moe prefill + decode investigation (2026-08-30)

Exploratory work on the qwen35moe (Qwen3.6-35B-A3B) path at the `rdna-boosts`
base (`4fa92f0ae`), single Radeon AI PRO R9700 (gfx1201).

## Contents

| File | What |
|---|---|
| `report.md` | The investigation: baseline curves, depth-falloff mechanism, the n_ubatch discovery, op-level cost structure, GDN/MoE/decode status |
| `plan-fused-moe.md` | Plan: fused gate+up+GLU kernel for the routed experts (the 41-53% mmid block) |
| `plan-decode.md` | Plan: decode-path investigation (92 t/s, flat with depth, bottleneck inside the graph path) |
| `data/` | Raw measurements: bench logs, op-timing dumps, GDN A/B, parsers |

## TL;DR

1. **Depth falloff = flash-attention KV traffic** - the only op that grows
   with context (3% -> 18-26% of ubatch time from 2K to 16K KV). WMMA
   flash-attn (block 04) already mitigates it on gfx1201 (FA off is -33% at
   32K); on Strix Halo the same term is bandwidth-bound (the observed 3-4x
   falloff at 64K). Q8_0 KV is neutral on discrete; re-test on unified
   memory.
2. **`--ubatch-size 2048` is a free +65% prefill** on qwen35moe (default
   `n_ubatch` is 512; per-expert GEMM N goes 16 -> 64 tokens). MoE-specific
   (dense 27B gains only +4%). Deploy it now; re-test Flash-Next with it.
3. **The routed-expert GEMMs are the top code-level target** (fused
   gate+up+GLU, see plan-fused-moe.md).
4. **Decode is flat but slow** (92 t/s); the cost is inside the CUDA-graph
   path and uncharacterized (see plan-decode.md).

## Quick reproduction

```sh
# baseline (ub=512)
llama-bench -m <Qwen3.6-35B-A3B-Q6_K.gguf> -ngl 99 -p 16384 -r 2
# the win
llama-bench -m <...> -ngl 99 -p 16384 -ub 2048 -r 2          # ~4325 t/s
# op-level picture
GGML_CUDA_OP_TIMING=1 llama-bench -m <...> -ngl 99 -p 16384 -ub 2048 -r 1 --no-warmup -v
```
