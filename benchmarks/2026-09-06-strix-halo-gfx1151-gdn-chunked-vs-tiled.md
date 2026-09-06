# Strix Halo (gfx1151) — GDN prefill gap: chunked-bf16 (A) vs tiled-fp32 (B); investigation notes + handover

Date: 2026-09-06 (session continuing after the scatter-dedup landing 6a80b695c). Fork tree CLEAN at
6a80b695c. Delivery dfa90c5 (19 patches verified -> tip). This file is the working ledger for the
GDN item; a full record will be written once a direction lands.

## The gap (ledger #3, after flash + scatter-dedup closed)

A = gdn_bf16_scan_cuda (144 calls @ 2.71ms med = 0.391s) + gdn_bf16_kkt_cuda<3> (144 @ 0.58ms =
0.083s) = 0.474s vs B gated_delta_net_tiled_cuda (144 @ 2.97ms = 0.428s) per 4 pp2048 decodes:
+46ms/4dec (~+11ms/pass, ~0.45% uniform wall - GDN ~4.8% of a pp2048 pass; n_tokens per GDN call
is always the ubatch size (2048) so the fraction is identical at depth -> no depth-specific risk).

Geometry (pp2048): S_v=128, H=48 v-heads, H_k=16 k-heads (GQA 3:1), n_seqs=1, n_tokens=2048,
n_chunks=32, hg_ratio=3, K=1. Grids (rocprof inflates grid.x by blockDim.x for these kernels):
scan real (VD/BV=2, H=48, 1) block 256 (=96 blocks); kkt real (32, 16, 1) block 128 (=512
blocks); B tiled real (H=48, 1, S_v/(warps*cols)=2) 2D block (32, 8) = 256 thr.

## Per-path cost table (pp2048, per call; r1 rocprof = 2 evals)

| path | per call | numerics | notes |
|---|---|---|---|
| A sequential fp32 (GGML_CUDA_GDN_CHUNKED=0) | 17.0ms | exact | upstream fallback |
| A chunked fp32 (GGML_CUDA_GDN_CHUNKED_BF16=0) | 6.17ms (scan 4.49 + kkt 1.68) | exact-ish | |
| A chunked bf16 (DEFAULT, gfx11 file) | 3.29ms (scan 2.72 + kkt 0.57) | near-lossless deviation (doc: PPL +0.056% RDNA4 / -0.09% RDNA3 vs fp32) | gated_delta_net_chunked_bf16_gfx11.cu |
| B tiled fp32 (fired <128,8,8,16,false>) | 2.97ms | EXACT - fma-spelled bit-identical to its fp32 sequential | classic upstream tiled, deleted in A's newer base; halo config rows |

Work estimate: ~9.4 GFLOP/call -> at ~3ms only ~15% of fp32 peak: NOT compute-bound; both are
latency/serialization bound. bf16 tensor cores cannot win on raw compute here.

## KEY INSIGHT: the scan core already beats B; the gap IS the kkt tax

A scan core 2.72ms < B's ENTIRE tiled op 2.97ms. The whole +46ms = 0.32ms/call extra = the kkt
pass (0.57ms) minus B having no equivalent, roughly. Chunked pays 17% for the per-chunk inverse
A_sc (32 chunks x 16 kheads, 64x64 bf16 solves, 512 blocks; A_sc buffer 32*48*64*64*2B = 12.6MB
round trip + launch barrier). If the kkt were free, chunked = ~2.72ms < B 2.97 (A wins ~8% on
GDN).

## Probe: occupancy is NOT the lever (tested 2026-09-06)

Scan kernel resources: gsmem=61184 (61KB: k/S 16.9KB each 64x132 u16, A/W/D 8.7KB each, floats),
arch_vgpr=256 (capped). B tiled: gsmem=20608, vgpr=104. Hypothesis: 1 block/CU pinning the scan.
Test: __launch_bounds__(GDN_BF16_NTHR, 2) on gdn_bf16_scan_cuda -> NO change (2.72 -> 2.73ms).
Occupancy already adequate; the scan is bound elsewhere (serial chunk loop latency, WMMA pipe).

## Option C - retune the gfx11 chunked scan (user preference #1)

The gfx11 file is a self-described FIRST-GEN WMMA port (gated_delta_net_chunked_bf16_gfx11.cu,
ported from libr4d r4d_gdn_kkt_solve_k128_c64 + r4d_gdn_chunk_scan_k128_v128_c64, validated on
gfx1201 against FLA). Config constants at the top: GDN_BF16_BT=64, KD=VD=128, BV=64, KP=132
(LDS pitch), AP=68, NW=8 (waves/scan wg), NTHR=256, NTV=2 (tiles/wave main/V'/O loops), SKT=2,
SVT=2, NST=4, KKT_NW=4, KKT_NTHR=128. Tuning targets:
  - scan inner loop wave/tile mapping (NTV/SKT) + occupancy-independent latency hiding
  - kkt elimination/overlap (structural, likely big effort; scan serial depth kills naive
    stream pipelining: kktA | (scanA || kktB) | scanB ~ no win)
  - smem 61KB -> smaller tiles to raise blocks/CU (probe says not the lever, but revisit AFTER
    inner-loop work: register/smem may co-limit)
  - B's tiled sweep anecdote: gfx1151 H=32 2048tok 5.03->2.2ms (cfg-dependent); H=48 fired
    <128,8,8,16> at 2.97 - the fp32 kernel family has shape-tuned configs; the chunked gfx11
    may just be under-tuned for gfx1151's WMMA (first-gen port, validated on gfx1201).
  - Numerics: any retune MUST stay bit-identical to the current bf16 chunked output (or
    re-validate near-losslessness vs fp32 if the math changes - the FLA reference matters).
  - Env toggles that already exist: GGML_CUDA_GDN_CHUNKED=0 (sequential), GGML_CUDA_GDN_CHUNKED_BF16=0
    (fp32 chunked).

## Option B - port B's tiled fp32 (user preference #2)

B's gated_delta_net.cu (935 lines) = sequential + gated_delta_net_tiled_cuda (line 333) +
launch_gated_delta_net (502, incl the tiled dispatch: num_warps=32 on RDNA3.5 S_v==128
H in {32,48,64} !KDA; cfg default H==48?1:0 -> <128,8,8,16> for H=48 (grid H x 1 x 2, block
(32,8)) / <128,16,4,16> otherwise; env GGML_GDN_TILE_CFG to force). Numerics: fma-spelled
BIT-IDENTICAL to B's sequential gated_delta_net_cuda -> fp32 EXACT. In A: port kernel + launcher
+ a prefill dispatch branch in ggml_cuda_op_gated_delta_net_impl BEFORE the chunked for
(gfx1151 RDNA3_5 && S_v==128 && H in {32,48,64} && !kda && n_tokens>=16 && K==1 prefill);
honour cache/state_d_ext fused-slot handling. COST: numerics re-baseline (current default bf16
chunked -> fp32 tiled changes the fingerprint; lg-head-1.txt must be regenerated + quality gate
= top-1/token-stream stability, KL vs old). Precedent: the bf16 default itself is an accepted
near-lossless deviation; fp32 tiled returns to exact.

## Option A - fallback (user preference #3)

Accept the chunked bf16 as-is; GDN stays 0.45% behind B on every row (A is already ahead on all
measured depth-0 rows + depth). Move to the next ledger item.

## User direction (2026-09-06)

"Try C, and if that fails, then B, and if that fails, then fall back to A." While C/B are in
flight, DO NOT lose the other remaining items (ledger below). All GDN work is numerically
EXPLORATORY until a bit-identity or near-lossless gate passes - keep the tree green at
6a80b695c between probes (git checkout the gfx11 file after each probe).

## Remaining ledger (post flash + scatter-dedup): per 4 pp2048 decodes

1. GDN scan+kkt: +46ms (THIS FILE; user preference C -> B -> A).
2. Cijk grid256 bucket +45ms: dense fp16 GEMM shape feeding into rocblas (A 1.30 vs B 1.18ms
   med grid256/sh2048 bucket). Compare the graph's matmul dims/leading strides A vs B at the
   DENSE ffn/attn matmuls.
3. Launch overhead: A 15905 -> 15341 kernels (-564 from the pair); still more than B's 13457.
   k_get_rows 348 vs 160 and scale_f32 1428 vs 596 counts are structural; hunt the per-ubatch
   host submit path after 1-2.
Then: depth-12k/32k re-derivation (A strongly ahead; re-derive at -r 1 for B), decode followup
(tg parity 25.9/25.9; per-op kernel-mix + mmvq launch-bound profile at tg@0), gfx1201 validation
deferred ledger.
