# Strix Halo (gfx1151) — Cijk dense-GEMM gap: full investigation (both trees F32; ssm-pair fusion constrained)

Fork tip 376f02aa0 (clean). Investigation of ledger item "Cijk grid256 +45ms/4dec" (A 1.30 vs
B 1.18ms/call on the rocblas dense fp16/fp32 GEMMs).

## The measured gap (fresh same-session pp2048 r1 captures, Acijk/Bcijk)

A 428 Cijk calls/505ms vs B 428/483ms (+22ms/capture ~ +44ms/4dec). Same kernel name
(Cijk_Alik_Bljk_SB_MT32x32x8...), same call counts, same grids. Per-grid buckets:

| grid | calls | A med | B med | notes |
|---|---|---|---|---|
| (256,64) | 190 | 1.336 | 1.193 | +12% A - the whole item |
| (512,64) | 144 | 0.362 | 0.422 | A FASTER |
| (4096,64) | 94 | 2.044 | 2.070 | parity |

## Which GEMMs (A-side census of the cublas path: shapes + weight names + addresses)

All cublas GEMMs per pp2048 eval (ct=0 F32; ct=30 = BF16 for the indexer k_proj):
- (10240,4) x2/layer = blk.N.hc_attn_inject + blk.N.hc_ffn_inject (grid256 bucket) - read the
  hc_norm wide tensor
- (2560,48) x2/recurrent-layer = blk.N.ssm_alpha + blk.N.ssm_beta (grid512 bucket) - read hc_mixed
- (2560,512) x1/layer = blk.N.ffn_gate_inp (grid4096 bucket)
- (2560,128) bf16 = blk.N.indexer.k_proj (24 calls)

## Type question SETTLED: both trees run these GEMMs in pure F32

- GGUF (split, shards 2-3 hold the infos): hc_*_inject + ssm_alpha/beta weights are ALL F32.
- B fires NO float->half conversion kernels (checked B's capture) => no fp16 staging, no fp16
  load-time downcast => B is F32 end-to-end, identical to A. There is NO precision difference to
  match or reject; the user's FP16 concern was correct to raise but is moot here (B is not doing
  FP16 on these, consistent with its fp32-tiled GDN).
- A-side placements: operands 256B-aligned (pool), phases ~0x100 mod 4096.

## Ghost-hunt results (all negative - the +12% is context-scale, no A-side lever)

- Run-order swap (B first): delta persists -> not warm/cold.
- In-model buffer-phase sweep (env GGML_CUDA_BUF_PAD 0..3072 on the compute arena): pp2048 t/s
  flat 747-751 -> NOT phase-steerable in-model (isolated microbench showed a 2x phase swing
  1.19->2.30ms on the M4 shape, but real-model memory pressure masks it).
- L2-reuse between the same-layer inject pair: impossible - the injects are 20-50ms apart in GPU
  time (the layer's attention/MoE runs between them).
- Per-call distributions: both trees degrade in eval2 (thermal) but A degrades ~4x more
  (eval1 1.285->1.432 vs B 1.173->1.206) - thermal/clock context.
- Bench vs rocprof contradiction: bench says A ahead on all rows; capture totals say A +39ms
  busy - the item is inside the machine's day-to-day variance (~2.5-3% on some rows).

## The fusion idea, investigated and constrained (attempted per user direction)

Microbench premise: two M=4 GEMMs on the SAME B = 2.37ms; a fused M=8 single walk = 1.14ms
(2.1x) -> if the attn+ffn inject pair truly shared src1, a single-walk fusion would win ~2.3%.
Investigation killed it:
- The attn/ffn injects of a layer read DIFFERENT tensor objects (census tobj differs) whose
  runtime data is byte-identical (blk.1+) only because the ffn path's hc_norm REGENERATES the
  same values into the pool-reused buffer. The combine sits between the two paths in the graph -
  the equality is a runtime property, not structural -> fusing would be a correctness trap on
  other prompts/depths. NOT SAFE. DROP the inject fusion.

REAL candidate found: blk.N.ssm_alpha + blk.N.ssm_beta (recurrent layers only, ~36/48) - both
[2560x48] F32, read the SAME src1 object (hc_mixed-N, structurally shared - the code builds
beta=mm(ssm_beta,cur), alpha=mm(ssm_alpha,cur) on the same cur). Structurally safe pair.

Measurements (pairbench): stacked rocblas M=96 on [2560x96] weights = 0.410ms vs the 2x M48
pair 0.588ms (1.43x only - rocblas M96 config reads ~51GB/s, not 2x). Stacked M96 vs 2x M48:
NOT bit-identical (~3e-7 max diff; different rocblas config/k-order) - a documented fp32
near-lossless deviation if adopted (user accepted "essentially correct" fp32 reorder).

Implementation constraints discovered:
- The SSM alpha/beta MUL_MATs are NOT adjacent in the scheduled graph (alpha-MM@52, beta-MM@58
  with alpha's ADD/softplus/mul chain 54-56 between - the cgraph order is a topological DFS of
  the expanded outputs, NOT the creation order; build_layer_attn_linear creates beta first yet
  alpha executes first). The scheduler pair matcher (nodes i,i+1) CANNOT fire.
- Getting them adjacent needs a qwen4exp.cpp restructure that builds both MMs consecutively AND
  controls the expansion DFS - unreliable given the expansion pass.
- The robust routes: (a) load-time stacked [2560x96] weight + ONE graph MM + per-half views
  (~0.27-0.4% wall; loader plumbing), or (b) graph restructure + custom single-walk kernel
  (~0.6%, the real prize, reads the 21MB once at ~150-200GB/s). Both are numerics-moving
  (near-lossless fp32) and need the quality gate + re-baseline.

Verdict: the ssm-pair fusion is real but modest (0.3-0.6%) and needs either loader plumbing or
a graph restructure with a custom kernel - a focused follow-up item, NOT a tail-of-session
change. Parked with full design notes above. Recommend a fresh session if pursued.

## Status

Cijk item: both trees F32; the +12% grid256 delta is context/thermal-scale with no clean lever;
the structural win (ssm-pair) is designed but parked. Fork clean at 376f02aa0, bit-identical.
All probe code reverted.
