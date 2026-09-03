# Plan (c) Phase 1 results: decode cost structure (gfx1201)

Measurement session 2026-08-30. Model qwen35moe Q6_K, single R9700,
KV=512 (decode is flat with depth: 92.2 t/s at KV=16K too), ub=2048.
Method: the direct path (11.1 ms/step) is within 0.3 ms of the graph path
(10.8 ms), so the op timing on the direct path is representative. The op
timing was extended to also wrap the fused dispatches
(`ggml_cuda_try_fuse`), which were previously invisible (a per-op sync was
never the issue - fused nodes simply skipped the event pair).

## The cost model: 10.8 ms/token = 923 kernels

| Component | ms | % | Notes |
|---|---|---|---|
| Routed experts (2 fused kernels/layer) | 3.6 | 33% | gate+up: 37 us @427 GB/s; down+shared-gate+residual: 53 us @166 GB/s |
| Shared experts (up/gate/down) | 1.2 | 11% | ~200-380 GB/s |
| Recurrent qkv+z+alpha+beta+out (conv-folded) | 1.0 | 9% | 534 GB/s - good |
| lm_head [2048x248320] | 0.86 | 8% | 600 GB/s - near peak for 256-bit GDDR6 |
| RMS_NORM q8_1 folds | 0.76 | 7% | 41 kernels, ~18 us each |
| Attention layers (qkv + FA + rope) | 0.45 | 4% | FA is only ~1% of decode |
| GDN AR scan + state cpy | 0.5 | 5% | 30 sequential, ~11 us each |
| topk-moe fusion | 0.29 | 3% | 40 x ~7 us |
| Elementwise (ADD/MUL/UNARY/SCALE chains), GET_ROWS, CPY | ~1.3 | 12% | visible + fused |
| Launch/graph overhead | ~1 | 9% | 923 kernels/token |

Kernel time distribution (833 distinct kernels, event overhead ~4 us/kernel
deducted): 68% of kernels are under 20 us. The decode step is
latency-bound, not bandwidth-bound: the sum of 923 kernel critical paths
dominates. Only the lm_head (0.86 ms) and a few large GEMMs are
bandwidth-bound.

## Key findings

1. **FA is NOT the decode bottleneck** (1%, 0.11 ms). The earlier "FA 5.6
   ms / GET_ROWS 3.9 ms" decode numbers were an artifact of the old op
   timing: the streamed attention branches left join-waits that polluted
   the surrounding main-stream event pairs. The GDN AR chain is also small
   (5%).
2. **The fusion machinery is worth 24% of decode** (70.5 -> 92.8 t/s with
   GGML_CUDA_DISABLE_FUSION=1). Decode already fuses: gate+up into one
   mmvq kernel (40/layer), down+shared-gate+residual into another
   (`ggml_cuda_op_shexp_down_gate`), conv-input into the qkv mmvq, RMS_NORM
   + q8_1 quantize, GDN cache cpy, topk-moe, and multi-op elementwise
   chains. The `mul_mat_id_glu_ops` can_fuse check remains dead for the
   MMQ (prefill) path only.
3. **The routed-expert GEMMs are latency-bound, not bandwidth-bound**: the
   mmid mmvq kernel maps the 8 experts onto 8 warps with the K dimension as
   grid.x; the down experts (K=512) get ~2 K-blocks per warp and achieve
   only 166 GB/s (vs 600 GB/s near-peak). The gate+up (K=2560) does 427
   GB/s.
4. **CUDA graphs are worth only 2.5%** (90.1 t/s with graphs off); PDL is
   neutral (93.1 vs 92.8). Decode is flat with context depth.
5. **Dense calibration**: qwen35 27B Q6_K decode = 24.4 t/s (reads all
   21.3 GB/token at ~520 GB/s). The MoE's 8/256 expert read pattern is why
   it is 3.8x faster at decode (~2 GB/token). The card (Navi 48, 256-bit
   GDDR6, ~640 GB/s peak) delivers ~600 GB/s on pure weight reads.

## Phase 2 targets (ranked by evidence)

1. **One fused expert kernel per layer (gate+up+down, plan-fused-moe)**:
   current 37+53 = 90 us/layer -> a single ~60-70 us dispatch reading
   24.4 MB at ~400+ GB/s. Save ~1.2-1.3 ms/token (~12% decode) + the
   prefill win (the mmid block is 41-53% of prefill). This is the clearest
   single lever and it is the same work as plan-fused-moe.md.
2. **Short-K expert kernel config** (down: K=512 -> 166 GB/s): tune
   rpb/nwarps for the mmid decode shapes in the RDNA4 mmvq parameter table;
   a config-only change. Potential: recover ~0.5-1 ms of the down kernel.
   Must keep bit-identical per-row accumulation (the speculative-verify
   batch shares the config).
3. **Kill the small-kernel tail** (~2 ms in kernels <20 us): fold the ssm
   state GET_ROWS into the GDN AR kernel epilogue, fold residual ADD into
   the expert epilogues, and check the remaining visible GET_ROWS/CPY/
   elementwise chains. Each fusion removes ~10-15 us of kernel latency.
4. **Skip**: FA decode tuning, GDN AR chain fusion, PDL, graphs (all
   measured <5%).

## Tooling note (worth keeping)

The op timing now wraps fused dispatches (edit in
ggml/src/ggml-cuda/ggml-cuda.cu, node loop): record op_ev0 before
`ggml_cuda_try_fuse`, record op_ev1 + push `(node, idx, fused=true)` when
nodes_to_skip != 0. Fused kernels appear as "FUSED <op> <name>" rows. This
made the decode cost table possible; the same instrumentation applies to
prefill (fused GDN chunked kernel was previously invisible there too).
