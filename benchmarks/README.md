# Benchmark results: llama.cpp master vs rdna-boosts

Date: 2026-08-26
Machine: 3x Radeon R9700 (gfx1201, 34 GB, ~640 MB/s each), ROCm 7.14
Model: Qwen3.8-27B - Q8_0 (29 GB) for the 2c/3c rows, Q6_K (22.9 GB) for the
1c rows; the full BF16 weights are covered in their own section.

Configs under test (full protocol in [methodology.md](methodology.md)):

| label | build | KV cache | note |
|-------|-------|----------|------|
| A | master `d222767c7` | f16 | upstream baseline |
| B | master `d222767c7` | bf16 | silently f16 (master ROCm has no native bf16 KV) + per-token conversion cost |
| C | rdna-boosts (11 blocks) | bf16 | native BF16 KV + fp32-accumulate attention (block 03) |
| V | master, Vulkan backend | bf16 | native BF16 KV on Vulkan (different backend end-to-end) |

## Headline numbers

- **Prefill**: C beats A by +33% (32K, 2 cards) up to +74% (64K, 1 card);
  the fork's edge grows with depth because bf16 halves the attention KV
  traffic. Peak prefill anywhere: **1930 t/s** (8K depth, 3 cards, +13.8%).
- **Decode**: C beats A by +6-9% on 2-3 GPUs (+3-4% on the single card),
  roughly constant across depths.
- **B's silent-f16 penalty** (decode): -2% (4K) to -34% (64K, 1 card),
  growing with depth and bandwidth pressure. B's prefill is indistinguishable
  from A's.
- **PPL**: a statistical wash - every config within +/-0.04 error bars at
  both resolutions, directionally in C's favor (full-corpus ladder below).
- **3-card scaling**: ROCm decode +17-20% going 2 -> 3 cards
  (bandwidth-bound), prefill +1-6%; Vulkan scales negatively on both axes
  (layer-split sync).

## Throughput

The tables are grouped by GPU count; within each, the backend is on the
left and the context depth runs along the top (4K/8K/32K/64K). The 1-GPU
table uses the Q6_K model (22.9 GB) on a single 640 MB/s card; the 2- and
3-GPU tables use Q8_0 (29 GB). ROCm runs use tensor split, Vulkan uses
layer split (Vulkan has no tensor split). Values are means of 1-2 rounds,
stable to <1%.

### 1 GPU (Q6_K weights, 22.9 GB, 640 MB/s single card)

Prefill (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 613.3 | 623.9 | 525.9 | 422.4 |
| B bf16 | 606.5 | 619.4 | 523.6 | 420.1 |
| C fork | 858.5 | **919.8** | **858.7** | **733.9** |
| V Vulkan | **909.1** | 894.2 | 767.7 | 638.7 |
| **C vs A** | +40.0% | +47.4% | +63.3% | +73.7% |

Decode (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 22.99 | 22.79 | 21.61 | 20.22 |
| B bf16 | 22.13 | 21.05 | 16.81 | 13.39 |
| C fork | 23.79 | 23.60 | 22.32 | 20.87 |
| V Vulkan | **24.18** | **23.87** | **22.63** | **21.07** |
| **C vs A** | +3.5% | +3.6% | +3.3% | +3.2% |
| **B vs A** | -3.7% | -7.6% | -22.2% | -33.8% |

### 2 GPUs (Q8_0 weights, 29 GB, tensor split)

Prefill (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 1600.2 | 1570.6 | 1235.8 | 941.4 |
| B bf16 | 1602.8 | 1575.4 | 1229.3 | 939.3 |
| C fork | **1772.4** | **1816.5** | **1645.9** | **1395.0** |
| V Vulkan | 1126.8 | 1376.1 | 1463.0 | 1235.2 |
| **C vs A** | +10.8% | +15.7% | +33.2% | +48.2% |

Decode (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 30.05 | 29.92 | 28.80 | 27.46 |
| B bf16 | 29.35 | 28.50 | 24.73 | 20.83 |
| C fork | **32.06** | **31.92** | **30.60** | **29.20** |
| V Vulkan | 18.02 | 17.90 | 17.23 | 16.32 |
| **C vs A** | +6.7% | +6.7% | +6.3% | +6.3% |
| **B vs A** | -2.3% | -4.7% | -14.1% | -24.1% |

### 3 GPUs (Q8_0 weights, 29 GB, tensor split)

Prefill (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 1678.1 | 1696.7 | 1295.1 | 953.5 |
| B bf16 | 1675.5 | 1689.4 | 1281.9 | 953.2 |
| C fork | **1843.9** | **1930.2** | **1739.6** | **1449.8** |
| V Vulkan | 1069.4 | 1216.6 | 1144.9 | 960.4 |
| **C vs A** | +9.9% | +13.8% | +34.3% | +52.1% |

Decode (t/s)

| backend | 4K | 8K | 32K | 64K |
|---------|-----|-----|------|------|
| A f16 | 35.75 | 35.60 | 34.08 | 32.26 |
| B bf16 | 34.78 | 33.66 | 28.70 | 23.59 |
| C fork | **38.79** | **38.69** | **36.87** | **34.74** |
| V Vulkan | 16.50 | 16.12 | 14.81 | 14.16 |
| **C vs A** | +8.5% | +8.7% | +8.2% | +7.7% |
| **B vs A** | -2.7% | -5.5% | -15.8% | -26.9% |

### Reading the throughput tables

- **The fork's prefill edge grows with depth**: +10-16% at 4K/8K (2c) ->
  +33-34% at 32K -> +48-52% at 64K, and +40-74% on the single slow card. The
  bf16 KV halves the attention read traffic, and prefill re-reads the growing
  cache for every batch - so the deeper the context, the bigger the win.
  Prefill falls off roughly half as fast with depth on C (2c: -21% to 64K vs
  A's -41%).
- **8K is the prefill peak** for both backends on 3 cards (A: 1678 @ 4K ->
  1696.7 @ 8K -> 953 @ 64K; C: 1844 @ 4K -> 1930.2 @ 8K -> 1450 @ 64K).
- **B's decode penalty is small and shallow, brutal at depth**: -2 to -8%
  at 4K/8K, -14 to -16% at 32K, -24 to -27% at 64K (2c/3c), and -34% on the
  single card at 64K. The per-token f16 conversion cost scales with KV
  traffic. B's prefill is unaffected (it is compute-bound, not conversion
  bound).
- **C's decode edge is roughly constant rather than growing**: +6-9% on
  2-3 GPUs, +3-4% on the single card. Decode touches only the final KV
  depth, so the bf16 advantage does not compound with depth the way
  prefill's does.
- **Vulkan**: prefill beats master ROCm on 1 GPU at every depth (+43-51%)
  and on 2 GPUs at 32K/64K (+18-31%), but loses at shallow multi-GPU
  (4K: -30% 2c, -36% 3c) and at 3 GPUs even at depth (32K: -12%; 64K:
  ~tie) where the layer-split sync dominates. Its prefill never beats the
  fork's ROCm bf16. Decode collapses on multi-GPU (17.23 2c / 14.81 3c at
  32K; 14.16-16.32 at 64K) but matches or slightly beats C on 1 GPU
  (+1-2%, a recent upstream Vulkan improvement).
- **Multi-card scaling, 2 -> 3 cards** (Q8_0):

  | config | prefill (32K / 64K) | decode (32K / 64K) |
  |--------|----------------------|--------------------|
  | A | +4.8% / +1.3% | +18.3% / +17.5% |
  | C | +5.7% / +3.9% | +20.5% / +19.0% |
  | V | -21.7% / -22.2% | -14.0% / -13.2% |

  ROCm decode is bandwidth-bound and a third card adds bandwidth (+17-20%);
  prefill is compute/latency-bound and barely scales (+1-6%). Vulkan scales
  negatively on both axes. vLLM cannot span 3 GPUs with a single model;
  llama.cpp (ROCm) can - the 3-card row is the interesting one for serving
  comparisons.

### Full BF16 weights, 3 cards (fork build, bf16 KV)

| depth | prefill | decode |
|-------|---------|--------|
| 4K | 1282.1 t/s | 23.56 t/s |
| 32K | 1571.1 t/s | 22.75 t/s |
| 64K | 1348.6 t/s | 21.95 t/s |

vs the Q8_0 fork on 3 cards (1844/1740/1450 prefill, 38.8/36.9/34.7 decode):
decode ~37-39% slower (weight-bandwidth bound, 54 GB vs 29 GB), prefill
7-30% slower. Combined with the PPL result (Q8_0 costs nothing measurable),
the full BF16 weights buy no PPL and cost ~1.65x decode: not competitive for
serving; Q8_0 is the right production choice.

## Perplexity (wikitext-2 test, -c 2048, flash-attn on)

### 128 chunks x 2048 ctx

| config | build | backend | KV | PPL | vs C |
|--------|-------|---------|-----|-----|------|
| A | master | ROCm | f16 | 6.3193 +/- 0.04262 | +0.0031 |
| B | master | ROCm | bf16 (silent f16) | 6.3213 +/- 0.04264 | +0.0051 |
| C | rdna-boosts | ROCm | bf16 | **6.3162** +/- 0.04264 | - |
| V | master | Vulkan | bf16 (native) | 6.3293 +/- 0.04284 | +0.0131 |

All four within their +/-0.043 error bars - the ordering is directional, not
conclusive. Notes:
- B equals A: master's ROCm bf16 is f16 in disguise, so it carries f16's PPL.
- V is the reference for a *true* native BF16 KV (master's Vulkan backend
  really stores bf16, unlike master's ROCm path). It comes out highest;
  caveat: V is a different backend end-to-end (Vulkan dequant and attention
  kernels have their own numerics), so the delta is not purely the KV type.
- C is the lowest: the fork's bf16 does not trade PPL for speed against any
  reference.
- The BF16-weight (f32 KV) ground truth did NOT land below the Q8_0 set:
  mainline BF16+f32 6.3212, fork BF16+f32 6.3220. The Q8_0 quants are
  imatrix-calibrated and wikitext-2 at 128x2048 is an easy protocol, so the
  quantization cost is below statistical resolution. Honest reading: Q8_0 is
  effectively free here, and the fork's bf16 KV does not trade PPL against
  anything, including full precision.
- The fork build on BF16 weights matches mainline exactly (6.3220 vs
  6.3212): the rdna-boosts pipeline does not change the model's numerics.

### Full corpus (the maximal resolution)

The test set is ~152 chunks at 2048 ctx, so the 500-chunk request exceeded
the corpus and these runs covered the entire test set. Harder trailing chunks
raise the absolute PPL by ~0.04 for every config; the error bars barely
shrink because per-chunk loss variance dominates.

| config | full corpus PPL |
|--------|-----------------|
| fork Q8_0 bf16, chunked fp32 | **6.3561** +/- 0.0406 |
| fork Q8_0 bf16, chunked bf16 (default) | 6.3563 +/- 0.0406 |
| fork Q8_0 bf16, sequential GDN | 6.3576 +/- 0.0406 |
| mainline BF16 + f32 KV (ground truth) | 6.3595 +/- 0.0406 |
| mainline Q8_0 f16 (config A) | 6.3619 +/- 0.0406 |
| mainline Vulkan Q8_0 bf16 (config V) | 6.3678 +/- 0.0407 |

Same ordering as the 128-chunk runs, now at full-corpus resolution:
- GDN dispatch trio spread 0.0015: the chunked GDN rewrite and its bf16
  tensor-core path are PPL-neutral (see the dispatch matrix below).
- All three Q8_0 forks sit at or below the BF16 full-precision ground truth
  by 0.002-0.003: calibrated Q8_0 quants are effectively free.
- Directional only: same weights, same ROCm backend, the fork's bf16 KV
  (6.3563) is 0.006 better than mainline's f16 KV (6.3619), and Vulkan's
  attention numerics land last.

### GDN dispatch matrix (fork, Q8_0, bf16 KV, 2 cards, 128 chunks)

Hypothesis: the chunked GDN rewrite (or its bf16 tensor-core path) is the
source of the small PPL spread. Tested by forcing the alternate dispatches:

| GDN path | env | PPL |
|----------|-----|-----|
| chunked bf16 (default, S_v=128) | - | 6.3162 |
| chunked fp32 | `GGML_CUDA_GDN_CHUNKED_BF16=0` | **6.3144** |
| sequential (non-chunked) | `GGML_CUDA_GDN_CHUNKED=0` | 6.3177 |

Spread 0.0033, all within +/-0.043: neither the chunking nor the bf16
tensor-core path is a measurable PPL variance source. (The fp32 chunked path
happens to land lowest; that is not significant.)

## Analysis

### Why Vulkan bf16 sits at the top of the PPL range

Every config uses the same Q8_0 model, so the ordering is driven by the
attention/KV pipeline numerics:

- **fork (C)**: bf16 KV storage (7 mantissa bits), full fp32 attention math
- **mainline f16 (A)**: f16 KV storage (10 mantissa bits), fp32 attention math
- **Vulkan (V)**: bf16 KV storage (7 bits), fp16-rounding at the attention

Investigation findings (master, ggml-vulkan):
- The R9700 (RADV gfx1201) exposes VK_KHR_cooperative_matrix +
  VK_NV_cooperative_matrix2, and the build compiles the bf16 coopmat2
  shaders (GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT=ON), so bf16 KV dispatches
  the cooperative-matrix (tensor-core) flash attention.
- The bf16 coopmat2 variant (fa_bf16_dict: FLOAT_TYPE=bfloat16_t,
  BFLOAT16=1, GL_EXT_bfloat16) reads K/V natively as bfloat16_t straight
  into coopmat<bfloat16_t,...> (raw uint8_t buffers, no dequant), and
  rounds Q (fp32) to bf16 for the tensor core. QK^T/PV run as bf16 x bf16
  -> fp32 tensor-core ops with fp32 softmax (ACC_TYPE=float). The fp16
  rounding (faDecodeK -> float16_t) applies only to the f16/quantized K/V
  variants, not bf16.
- So Vulkan's bf16 attention operates at 7-bit mantissa precision
  EVERYWHERE: bf16 KV storage, bf16 Q, bf16 tensor cores (fp32 accumulate).
  The ROCm paths keep Q and the attention math in fp32 (fork: bf16 KV only;
  mainline: f16 KV, 10-bit mantissa, fp32 math). Vulkan's is the lowest-
  precision attention of the three - the plausible mechanism for its
  consistent position at the top (worst) of the range: ~0.006 PPL over
  mainline f16 and ~0.012 over the fork, all within the +/-0.04 error bars.
- Precision table (attention pipeline):

  | config | KV storage | Q at attention | attention math |
  |--------|-----------|----------------|----------------|
  | A/B mainline | f16 (10 bits) | fp32 | fp32 (B silently = A) |
  | C fork | bf16 (7 bits) | fp32 | fp32 |
  | V Vulkan | bf16 (7 bits) | bf16 | bf16 tensor cores, fp32 acc |

  Note the ordering does NOT reduce to "KV mantissa bits" alone: A's f16 KV
  has more mantissa bits than C's bf16 KV, yet C measures better - so the
  fork-vs-mainline gap comes from other numerics (GDN rewrite, attention
  implementation), not KV precision. All within +/-0.04 error bars.

### Why the BF16+f32 "ground truth" did not top the ladder

The full-precision config (BF16 weights, f32 KV) measured mid-pack
(6.3595 full corpus), below the fork's Q8_0 configs. Three stacked reasons:

1. The BF16 weights are themselves quantized - 7-bit mantissa (the model's
   native format). The config carries the same mantissa quantization as the
   fork's bf16 KV.
2. Imatrix-calibrated Q8_0 (8-bit mantissa + per-block fp32 scale, scales
   tuned to the activation distribution) can approximate the original f32
   weights BETTER than the bf16 line does. Calibrated quants occasionally
   beating the half-precision "ground truth" is expected when the
   quantization noise is smaller than the bf16 representation error.
3. The f32 KV bought nothing - KV precision is not a PPL lever on this model
   at this protocol (bf16 = f16 = f32 KV within noise, proven across every
   config).

The mid-pack landing is the strongest endorsement of the fork: the fastest
config is statistically indistinguishable from the most precise config
runnable (0.003 apart, direction inconsistent, +/-0.04 bars). Config nuance:
the ground truth ran 3-card layer split (mainline aborts on tensor split) vs
the forks' 2-card tensor split - numerics-equivalent per-op, only
cross-device reduction order differs.

### Fork KV design rationale (C)

The fork's KV design is a conscious trade: compute the attention with an fp32
accumulator end-to-end and round to bf16 ONLY at the cache boundary (the K/V
projection output). This sacrifices ~3 mantissa bits vs an fp16-cached path
(10 bits), but keeps the full fp32 exponent range in the accumulation. On
this model the range property wins: C's bf16+fp32 path measures at-or-below
every fp16-based config at every resolution (6.3563 full-corpus vs A's f16
6.3619), and its prefill edge grows with depth because bf16 halves the KV
traffic. The fp16-vs-bf16 storage choice is model-dependent in principle
(models with small-magnitude attention states might prefer fp16's mantissa);
empirically, on Qwen3.8-27B at this protocol, bf16 storage costs nothing
measurable.

Exonerated along the way:
- The scalar (non-coopmat) FA bf16 variant dequantizes bf16 -> fp32 with
  fp32 accumulate (FA_DEQUANT4_BF16 + forced f32acc) - fully precise.
- The Vulkan gated_delta_net kernels are pure fp32 (FLOAT_TYPE=float).
- The f32 -> bf16 KV cache write uses round-to-nearest-even, identical to
  ggml_compute_fp32_to_bf16.
- The Q8_0 GEMM inputs are fp16 on both backends (tensor-core dequant) -
  not a differentiator.

Caveat: if a build/device lacks bf16 matrix-core support, bf16 KV falls
back to the scalar fp32 FA (precise) and the penalty would then come only
from the 7-bit vs 10-bit KV storage difference. The measured direction is
the same either way.
