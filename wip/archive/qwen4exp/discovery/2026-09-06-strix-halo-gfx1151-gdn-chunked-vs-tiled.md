# Strix Halo (gfx1151) — GDN prefill gap: chunked-bf16 scan retuned (RESOLVED via option C)

Date: session continuing after 6a80b695c. Fork commit: 376f02aa0. This record closes the GDN
ledger item (#3): A's gdn_bf16_scan_cuda + gdn_bf16_kkt_cuda now BEAT B's gated_delta_net_tiled.

## The gap (as measured at 6a80b695c)

A = gdn_bf16_scan_cuda (144 calls @ 2.71ms med) + gdn_bf16_kkt_cuda<3> (144 @ 0.58ms) =
0.474s vs B gated_delta_net_tiled_cuda (144 @ 2.97ms = 0.428s) per 4 pp2048 decodes:
+46ms/4dec (~+11ms/pass, ~0.45% uniform - GDN ~4.8% of a pp2048 pass; n_tokens per GDN call
is always the ubatch size, so the fraction is identical at depth).

Geometry (pp2048): S_v=128, H=48 v-heads, H_k=16 k-heads (GQA 3:1), n_seqs=1, n_tokens=2048,
n_chunks=32. Scan grid (VD/BV=2, H=48, 1) block 256 = 96 blocks; kkt (32,16,1) block 128 =
512 blocks; B tiled (48,1,2) 2D block (32,8).

## Per-path costs (pp2048, per call; established at 6a80b695c)

| path | per call | numerics |
|---|---|---|
| A sequential fp32 (GGML_CUDA_GDN_CHUNKED=0) | 17.0ms | exact |
| A chunked fp32 (GGML_CUDA_GDN_CHUNKED_BF16=0) | 6.17ms | exact-ish |
| A chunked bf16 (DEFAULT) @ 8 warps | 3.29ms (scan 2.72 + kkt 0.57) | near-lossless deviation |
| A chunked bf16 @ 16 warps (376f02aa0) | 2.68ms (scan 2.11 + kkt 0.57) | BIT-IDENTICAL to the above |
| B tiled fp32 | 2.97ms | fp32 EXACT (fma-spelled) |

Work ~9.4 GFLOP/call at ~15% of fp32 peak: NOT compute-bound; latency/serialization bound.

## Root cause: 1 block/CU with only 8 resident warps

- gfx1151 hardware query: MaxSharedMemoryPerMultiprocessor = 65536 (64KB LDS per CU),
  regs 196608. The scan's 61KB smem -> HARD-CAPPED at 1 block/CU by LDS.
- The earlier "negative occupancy probe" (launch_bounds minBlocks=2) was VACUOUS: minBlocks
  only caps REGISTERS, which were never the binding constraint (LDS was); the code object was
  byte-identical (vgpr 256, private 584 both builds) and the probe build did not even rebuild
  the kernel (built --target llama only, not ggml-hip). DO NOT trust that result.
- The scan is a 32-step SERIAL chunk loop (state carries across chunks) - scaling probe:
  0.573ms/8chunks vs 2.722ms/32chunks -> ~85us marginal per chunk-step, no fixed overhead.
  With 8 warps resident (2/SIMD) and 61KB LDS preventing more blocks, the serial chain's
  latency had almost nothing to hide it.
- B's tiled: 20.6KB smem -> 3 blocks/CU = 24 resident warps - 3x the latency hiding, which is
  why fp32 tiled could match/beat bf16 chunked despite the WMMA arithmetic advantage.

## The fix (376f02aa0): 16 warps x 1 tile/wave inside the single resident block

Constants: GDN_BF16_NW 8->16 (256->512 thr), NTV 2->1, SVT 2->1 (SKT stays 2; NST 4->2).
Wave->tile mapping re-derived: mt = w>>2 (4 token-row groups), ntb = w&3 (4 v-tiles,
1 tile/wave), state kt0 = (w>>2)*2, vt0 = w&3 (2x1 state slice). All tile loops self-scale
via the NTV/NST/SKT/SVT macros; only the mapping line + constants changed (7 lines).
Per-tile mma accumulation order unchanged -> BIT-IDENTICAL by construction (verified:
logitcmp lg-head-1 fingerprint, 838-token prefill + 40 decode steps, top5 @9dp + FNV).

Results (same-session, rocprof pp2048 r1):
- scan 2.722 -> 2.115ms/call (-22%), kkt unchanged 0.569. vgpr 256 -> 208, smem 61184 same.
- GDN op total 2.68ms < B's 2.97 (A now 10% faster on the whole GDN op).
- Capture total 4.905 -> 4.842s.

## Probes that did NOT land

- KKT_NW 4->8 (kkt more warps): REGRESSED 0.569 -> 0.755ms/call - reverted. The kkt (512
  blocks, 11.5KB smem, already ~5-6 blocks/CU) is not warp-starved.
- (superseded by the vacuous-minBlocks finding: do not re-run launch_bounds experiments for
  this kernel; LDS caps occupancy at 1 block/CU and the only lever is warps-per-block.)

## Bench (same-session A/B, warm cache, pp DESCENDING in one process, r3)

| row | A (376f02aa0) | B | A/B | (was at 6a80b695c) |
|---|---|---|---|---|
| pp4096 | 726.9 | 720.7 | 1.009 | 1.018 |
| pp2048 | 776.3 | 771.8 | 1.006 | 1.002 |
| pp1024 | 743.5 | 725.8 | 1.024 | 1.014 |
| pp512 | 663.0 | 645.9 | 1.026 | 1.013 |

Interleaved A1/B/A2 (A2 = 726.6/776.2/743.3/664.4). All four rows ahead; pp2048 margin grew
0.002 -> 0.006 with the GDN cut. In-ladder absolute t/s run higher than isolated single-row
runs (long descending process warms the clock) - judge only A-vs-B within a ladder.

## Numerics gate notes

- logitcmp BIT-IDENTICAL pre-commit and POST-commit (376f02aa0). The remap is numerically
  transparent: every (m,n) tile and every state tile is still accumulated by exactly one wave
  in the same k/sk order; only the wave ownership changed. cli-bytes implied by full logit
  identity (skipped: identical 9dp logits at every of 838+40 positions cannot change text).
- bf16 chunked stays the DEFAULT (near-lossless deviation vs fp32, documented in the file);
  the retune does not move numerics at all. Option B (B's fp32 tiled, exact) was NOT needed.

## gfx1100/gfx1101 caveat (deferred validation)

NW16 = 512 threads x 208 vgpr = ~106K VGPRs/CU + 61KB LDS. gfx1151 reports 196608 regs/CU
(fits). Desktop RDNA3 (gfx1100/gfx1101) must be re-validated for launch fitness; if their
regfile is the classic 64K VGPRs/CU the kernel will not launch - revert to NW8 + the 2x2
mapping there (2-line change, the original constants are in git history). Comment in-file.

## Delivery

ws6-gdn-scan-nw16.patch = git diff(6a80b695c, 376f02aa0) (10+/7-). Applies clean on a
pristine 6a80b695c (verified). Patch 20 of the series.

## Remaining ledger (unchanged - do not lose)

1. Cijk grid256/sh2048 bucket +45ms/4dec: A 1.30 vs B 1.18ms/call on the dense fp16 GEMMs -
   compare the graph matmul dims/leading strides A vs B at the DENSE ffn/attn matmuls feeding
   rocblas.
2. Launch overhead: A 15341 vs B 13457 kernels - k_get_rows 348-vs-160 and scale_f32
   1428-vs-596 counts are structural; hunt the per-ubatch host submit path after 1-2.
Then: depth-12k/32k re-derivation at -r 1 for B; decode followup (tg parity; per-op kernel-mix
+ mmvq launch-bound profile at tg@0); gfx1201/gfx1100 validation (deferred ledger).
