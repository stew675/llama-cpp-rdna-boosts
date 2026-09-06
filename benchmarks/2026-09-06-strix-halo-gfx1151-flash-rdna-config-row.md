# Strix Halo (gfx1151) — flash_attn 20% gap = RDNA WMMA config row; B's Q_in_reg=false row adopted

Date: 2026-09-06 session (continuing the prefill ledger from the 09-13 record). Tip: e7eecb369 (clean).

## Gap (#1 in the prefill ledger): flash_attn_ext_f16 A 11.70 vs B 9.75ms/call

Identical dispatch shape both trees: 48 calls, grid (640,8,1), wg (32,8,1) = 256 threads,
DKQ=DV=256, ncols1=8, ncols2=8 (fired config key (256,256,64)). Earlier "smem 33792 vs 51328"
numbers misled: on ROCm the K/V smem swizzle is DISABLED in both trees (tile_stride = nbatch+4
everywhere), so the smem delta comes from the CONFIG ROW, not the swizzle.

RDNA WMMA config table diff (B vs A) — A = newer upstream tune, only the fired row matters here:

| key | B (halo, c7af5c6c2) | A (upstream b10837) |
|---|---|---|
| (256,256,64) | nthreads256 occ1 fa32 K2=128 V2=64 cb64 stg1 **Q_in_reg=false** | nthreads256 occ2 fa32 K2=128 V2=128 cb32 stg1 **Q_in_reg=true** |

Other differing rows (320/512/576 head dims) are inert for qwen4exp - deferred.

## Probes (same-session rocprof, flash per-call med, 48 calls/capture)

- A original row (occ2/V2=128/cb32/Q_in_reg=true): 11.70ms, capture total 9.980s
- PROBE1 = B's exact row (occ1/V2=64/cb64/Q_in_reg=false): **9.54ms** (faster than B's 9.75),
  capture 9.909s
- PROBE2 = B's row + Q_in_reg=true only: **11.19ms** -> Q_in_reg is the lever, NOT the
  V2/cb/occ geometry. Q_in_reg=true keeps the 64-col Q tile permanently in registers ->
  256-vgpr pressure (arch_vgpr_count capped 256 in both code objects) -> spills/slowdown.

## Design answer: "keep A's low smem at B's speed?" — NO, they are the same switch

smem total (nstages=1): Q_in_reg=false -> Q+KV+mask ~26KB (Q tile alone 64x132x2 = ~16.9KB);
Q_in_reg=true -> max(Q, KV+mask) ~17KB. The low-smem variant IS the slow one (Q in registers).
Q staged in smem is required for the speed; the ~16.9KB Q tile is unavoidable with
Q_in_reg=false regardless of V2/cb (probe2 kept V2=64/cb=64 and still regressed). ~26KB/block
is far inside the LDS budget -> no occupancy penalty. Verdict: adopt B's row exactly.

## Landed (e7eecb369): B's row adopted, single-line change

Fired RDNA (256,256,64) row -> (256, 1, 32, 128, 64, 64, 1, false). Logitcmp BIT-IDENTICAL.
Compile-time gating: none needed - the row lives in the shared constexpr host/device RDNA
table (applies to any AMD-WMMA build using DKQ=256/ncols=64 FA); matches B's proven halo
tuning. NOTE for gfx1201 deferred validation: RDNA4 reaches this same table - re-validate
there (the Q_in_reg tradeoff may differ by arch).

## Same-session depth-0 A/B (t/s, higher=better; A then B, warm cache, r3)

| row | A | B | A/B | (was A/B post-repeat-absorb) |
|---|---|---|---|---|
| pp512  | 656.1 | 650.4 | 1.009 | (parity) |
| pp1024 | 737.7 | 727.2 | 1.014 | (parity) |
| pp2048 | 769.2 | 775.5 | 0.992 | (0.987) |
| pp4096 | 738.8 | 723.9 | 1.021 | (0.996) |

pp2048 delta shrank 0.987 -> 0.992 same-session (A 762.9 -> 769.2 on this row change; B 773 ->
775.5 session drift). Capture total 9.980 -> 9.909s. Remaining ledger per-call deltas now:
quantize swiglu/<*,true> 2x-call structure (+72ms/4dec), GDN scan+kkt (+46ms), Cijk grid256
bucket (+45ms), k_get_rows + launches. Raw: /tmp/prof/Aflashprobe{,2}_results.db (probe1
results = committed build; probe2 = Q_in_reg experiment).

## Negative result: scheduler-level gate+up pair merge (B's mul_mat_q_pair) breaks A's numerics

Gap #2 of the ledger re-derived precisely: quantize_mmq_q8_1<*,true> (dedup-scatter feed) A 376
calls/0.125s vs B 188/0.061s (identical grids 262144x3) = +64ms/4dec. SWIGLU was actually AT
PARITY (A 0.150 vs B 0.151 incl. the shared-expert small call - earlier ledger misattributed
+72ms to swiglu; the delta is the <*,true> 2x count). Per-layer structure: ffn_moe_gate +
ffn_moe_up (MUL_MAT_ID, dedup scatter each, type 21/23) + ffn_moe_down (swiglu-fold) + shexp
dense up/gate pair. B merges each gate+up via ggml_cuda_mul_mat_q_pair (one mm_ids_helper +
one scatter, both muls read the shared q8_1 buffer) - B's <*,true> = 1/layer.

PORT ATTEMPTED + REVERTED (tree clean at e7eecb369, bit-identical): added the pair function
(ids + dense branches, with A's gfx1151 q8_1_chunks) + a try_fuse matcher (shared src1/ids,
same shapes, both mmq, equal ds layout). Fires correctly (PAIRCENSUS: blk.N gate+up per layer;
scatter 376->192, <*,false> 1768->1584, ~76ms/4dec saved) BUT logitcmp DIVERGES at 1e-4 in the
pp-last top logit - even a SINGLE layer (blk.0 only) and even re-running the two ORIGINAL
single-node muls from the pair dispatch position diverges.

ROOT CAUSE (scheduler-census): in A the up/down MUL nodes are NOT plain consecutive nodes in
the try_fuse walk - the walk shows gaps at 106/108/110/114/115/117(=up)/119(=down) - A's
existing MoE mega-matchers (topk-moe/routed, weighted-down, fused gate+up+GLU) already
partition the MoE dispatch into umbrellas that consume those indices. B's pair works because B
lacks those competing matchers (plain per-node graph). So the fix belongs INSIDE A's routed/
topk path (dedupe the scatter there), NOT at scheduler adjacency level. NEXT SESSION: hunt the
up-mul's real dispatcher (the fusion umbrella covering index 117) and dedupe the scatter
quantize there instead.

## Delivery

ws6-fattn-rdna-row-qinreg.patch = git diff(b987877d7, e7eecb369) (13 lines). Series now 18;
re-verify from-scratch -> tip e7eecb369 on commit.
