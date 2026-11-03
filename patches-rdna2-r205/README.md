# rdna-boosts r205 add-on set (blocks 16-18)

## Block 16 -- two-lane prefill (LLAMA_PREFILL_LANES)

**3 patches** that apply on top of the 16-block rdna-boosts delivery set
(`release.json`, `v16-6050b467c-r205`, applied tree `d6e71901`). Strict
`git am` on the recorded base reproduces the recorded tree `3b00f589`
byte-for-byte (0016: d6e71901 -> 3249229a; 0017: 3249229a -> c680d4f6;
0018: c680d4f6 -> 3b00f589; validated on a fresh worktree at base
`6050b467c`: 16/16 base blocks strict, tree assert, add-on strict,
tree assert; 0018 also strict standalone on the 16-block base).

## What it adds

`LLAMA_PREFILL_LANES=2` (default 1) builds two consecutive prefill ubatches
on a second backend scheduler and submits their splits alternately
(`ggml_backend_sched_graph_compute_async_pair`), so one device's routed
experts overlap the other device's attention on a wide prefill. Opt-in;
degrades to the existing single-lane path when < 2 non-CPU backends,
pipeline parallel, `no_alloc`, or a draft context (`ctx_other`) is present.
A ubatch is paired only when it is wide, output-free, and no nextn (MTP) or
F1 layer-input readback is active — MTP and DFlash stay single-lane.

The decisive part is the copy-event design. On HIP a queued
`cudaStreamWaitEvent` resolves to the LATEST `cudaEventRecord` at execution
time. The stock eager-copy path records ONE shared per-device
`copy_event`; with two lanes, the other lane's copies re-record it and
rebind the already-queued consumer waits — every pair serializes
(measured: ~0% gain in every layout tested). The eager path now takes
`cpy_tensor_async_nowait`: the copy is issued on the source stream and the
caller's per-copy ring event is recorded there; the consumer waits on its
own event. No shared event is ever re-waited. `cpy_tensor_async` keeps its
old behavior for its two remaining callers.

Surface: `llama_context` second scheduler + lane-aware
prepare/graph_reserve/compute; `ggml_backend` lane-aware split build and
pair submission; stream-ordered host input copies; `llama_n_prefill_lanes()`
accessor; `GGML_SCHED_PAIR_DEBUG` pair/hazard diagnostics with per-lane
per-backend GPU event spans (`ggml_backend_event_elapsed_time`, an optional
device-iface hook, `hipEventElapsedTime` on the unified CUDA/HIP backend).

## Reference layout (MoE on eGPU + APU)

`-sm layer -ts 1,0 --fit off -fa on -ub 2048 -b 4096 -np 1`, routed-expert
layers steered to the second card via
`-ot 'blk.<moe-idxs>.ffn_(gate|up|down)_exps=ROCm1'`. Lanes never move
work: `-ts` placement decides which card owns which tensors, the interleave
only changes submission order.

## Validation (slavap-pd-7900: RX 7900 XTX gfx1100 + Radeon Pro V620 gfx1030,
gfx1030;gfx1100 dual build, llama-cli prompt-eval, p8192, ABBA 3 rounds)

| shape | 1 lane | 2 lanes | gain |
|---|---|---|---|
| dense (Qwen3.8-27B Q6_K_XL, `-ts 1,1`) | 511.9 t/s | 580.2 t/s | **+13.3%** |
| MoE prod-mirror (Qwen3.6-35B-A3B Q8_0, 38 expert layers → v620) | 2113.9 t/s | 2835.7 t/s | **+34.1%** |

- the pair engages on every 2-lane arm (`pair decided`, `interleaved=1`);
  intra-arm spread < 0.4%, no ABBA-order drift
- greedy trajectories: the 2-lane runs stay in the 1-lane jitter family.
  Box-level FP non-determinism: the 1-lane arm is itself non-deterministic
  across runs on this box, so byte-identity is not a valid gate here
- MTP (`-md`): the draft context (`ctx_other != nullptr`) stays single-lane,
  the target engages; draft acceptance unaffected, no aborts
- decode: the 2-lane arm runs the stream-ordered input copies on the main
  scheduler (halo parity) — tg128 70.96 → 81.87, no regression

## Block 17 -- RDNA2 (gfx1030) FA prefill: the q_global threshold and the D=256 ncols2 cap lift

**1 file** (`ggml/src/ggml-cuda/fattn-tile.cuh`, +16/−14), two RDNA2-only
levers for the TILE flash-attention kernel:

1. **q_global threshold 16 → 64** (11 sites: the two kernel-side
   conditions, the `Q_tmp_count` stub condition, 7 launcher `q_h2` args,
   1 comment). The global-half2 Q path only pays off at ≥ 64 columns; at
   the prefill config (16×2 = 32 cols) it makes the kernel re-read Q
   from global per K-tile pass and the kernel spills 140–145 VGPRs.
   Below 64 cols Q goes back to LDS: 0 spills.
2. **`rdna_cap_ncols2` lifted for `DKQ == 256`** (D=128 keeps the cap).
   The cap's comment claims the D=256 ncols2≥4 variants "overflow the
   register file (occupancy query returns 0 → abort)"; measured 4×8
   D=256 launches run fine, including native q8_0 KV (no abort) — the
   manual WGP occupancy fallback keeps ≥ 1 block resident even when the
   kernel spills scratch. Lifting the cap lets BOTH prefill and decode
   take the GQA-8 fold: KV read once per block instead of 4×.

### Why (root cause)

On gfx1030 the block-05 TILE kernel is ~15–16% slower than the stock
kernel at long context (pp2048 @ d32768: 1041.0 vs 1231.7 t/s on
Qwen3.6-35B-A3B, head-256 GQA-8). A/B on the r205 tree: a verbatim
stock-kernel transplant measures 1376.0 t/s, and r205's own body forced
to the stock 4×8 config gives 375 vs the stock kernel's 463 at d131072 —
identical config, the body is the whole gap (not host, not occupancy).

### Validation (slavap-pd-7900, V620 gfx1030 = Rocm2, -lm dio -b 8192 -ub 2048 -fa 1)

| test | r205 release | with block 17 | delta |
|---|---|---|---|
| pp2048 @ d0 | 2330.2 | 2347.1 | ~0 |
| pp2048 @ d32768 | 1041.0 | **1106.5** | +6.3% |
| pp2048 @ d131072 | 368.2 | **411.7** | +11.8% |
| tg256 @ d131072 | 50.9 | **58.9** | +16.0% |

- greedy tokens: two `-fa 1` runs (-s 42 -n 30) byte-identical to each
  other and to `-fa 0`
- gfx1100 (Rocm0) pp/tg within 0.4% of the pre-fix build (both levers
  are compile-time / device-CC gated; other targets byte-identical)
- r206: `fattn-tile.cuh` is byte-identical between r205 and r206
  (reconstructed from the r206 patch set: 0004/0005/0010 touch it, net
  zero drift), so 0017 applies to the current r206 release as well
- residual vs stock at 32k (1106.5 vs 1231.7): the r205 4×8 body's
  register pressure (256 VGPR + 488B scratch vs stock 212/0) — a
  body-trimming follow-up, not a launch hazard

## Block 18 -- RDNA2 (gfx1030) FA tile tail: the clamped in-bounds read

**1 file** (`ggml/src/ggml-cuda/fattn-tile.cuh`, +24/−12): on gfx1030 the
FA tile kernel's checked tail loads out-of-range rows through a per-row
`if (!oob_check || i < i_sup)` branch plus a local zero array. The branch
form is forced by the flat-scratch hazard (a pointer-select between the
global KV pointer and a local zero array can fault at a near-null
address for unaligned KV lengths), and it is also the cost: it pushes
the checked-tail instantiation past the 256-VGPR cap, spilling 488 B of
scratch through global memory in the D=256 4x8/16x2 prefill kernels
(HSACO: 256 sat / 167 spills / 488 B vs stock 212 / 0 / 0).

The tail values never reach the result (the k_VKQ_sup guards discard
them before the softmax), so on RDNA2 the out-of-range rows are clamped
to the last in-bounds row (`i_c = i_sup - 1`): a single in-bounds global
load, no branch, no zero array. Every read stays inside the KV cache
(an unchecked read past k_VKQ_max faults at a full 131k context). All
three sites (half2 loader, native dequant loader head/tail) are in
`#if defined(RDNA2)` (gfx103x-only compile-time macro); every other
target keeps the existing pointer-select, codegen unchanged.

### Validation (slavap-pd-7900, V620 gfx1030 = Rocm2, -lm dio -b 8192 -ub 2048 -fa 1)

| test | build14 (block 17) | with block 18 | delta |
|---|---|---|---|
| pp2048 @ d0 | 2347.1 | **2375.39** | ~0 |
| pp2048 @ d32768 | 1106.5 | **1259.83** | +13.8% (stock: 1231.7) |
| pp2048 @ d131072 | 411.7 | **474.70** | +15.3% |
| tg256 @ d32768 | 82.46 | **85.49** | +3.7% |
| tg256 @ d131072 | 58.9 | **62.40** | +5.9% |
| q8_0 pp2048 @ d32768 | 1098.47 | **1196.41** | +8.9% (no abort) |

- HSACO (dkq256-dv256, gfx1030): 4x8 229 VGPR / 0 spills / 0 B, 16x2
  236 / 0 / 0, q8_0 4x8 224 / 0 / 0
- greedy tokens: two -fa 1 runs (-s 42 -n 30) byte-identical to each
  other and to the block-17 build
- gfx1100 (Rocm0) pp/tg vs the pre-fix build: +0.6% (r=1 noise); the
  RDNA2 macro is gfx103x-only, so non-RDNA2 codegen is structurally
  unchanged
- 131k caveat: this box intermittently faults at d131k (HSA memory
  aperture) independent of this block -- the block-17 build faults
  identically; it clears and reappears with the V620 power profile
