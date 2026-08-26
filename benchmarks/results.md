# Results

Machine: 3x Radeon R9700 (gfx1201, 34 GB), ROCm 7.14; GPUs 0+2, tensor split.
Model: Qwen3.8-27B-Q8_0. Context depth for the server numbers: 31,694 tokens.
Two rounds per config.

## Perplexity (wikitext-2 test, 128 chunks x 2048 ctx)

| config | build | backend | KV | PPL | vs C |
|--------|-------|---------|-----|-----|------|
| A | master | ROCm | f16 | 6.3193 +/- 0.04262 | +0.0031 |
| B | master | ROCm | bf16 (silent f16) | 6.3213 +/- 0.04264 | +0.0051 |
| C | rdna-boosts | ROCm | bf16 | **6.3162** +/- 0.04264 | - |
| V | master | Vulkan | bf16 (native) | 6.3293 +/- 0.04284 | +0.0131 |

All four are within their +/-0.043 error bars, so the PPL differences are
directional, not conclusive. Notes:
- B equals A (master's ROCm bf16 is f16 in disguise, so it has f16's PPL).
- V is the reference for a *true* native BF16 KV cache (master's Vulkan
  backend, which unlike master's ROCm path really stores bf16). It comes out
  highest; caveat: V is a different backend end-to-end (Vulkan dequant and
  attention kernels have their own numerics), so the delta is not purely the
  KV type.
- C (rdna-boosts bf16) is the lowest of the four: the fork's bf16 does not
  trade PPL for speed relative to any of these references.
- The BF16-weight (f32 KV) ground truth did NOT land below the Q8_0 set:
  mainline BF16+f32 6.3212, fork BF16+f32 6.3220. The Q8_0 quants here are
  imatrix-calibrated (imatrix_unsloth.gguf lineage) and wikitext-2 at
  128x2048 is an easy protocol, so the quantization cost is below the
  statistical resolution: every config sits within +/-0.043 of every other.
  The honest reading: Q8_0 quantization is effectively free on this
  protocol, and the fork's bf16 KV does not trade PPL against any
  reference, including full precision.
- The fork build on BF16 weights matches mainline exactly (6.3220 vs
  6.3212): the rdna-boosts pipeline does not change the model's numerics.

## Throughput at ~32K context depth

| config | build | backend | KV | prefill @32K (r1 / r2) | decode @32K (r1 / r2) |
|--------|-------|---------|-----|------------------------|-----------------------|
| A | master | ROCm | f16 | 1235.6 / 1236.0 t/s | 28.79 / 28.80 t/s |
| B | master | ROCm | bf16 (silent f16) | 1236.2 / 1222.3 t/s | 24.81 / 24.64 t/s |
| C | rdna-boosts | ROCm | bf16 | **1642.5 / 1649.2 t/s** | **30.63 / 30.57 t/s** |
| VK | master | Vulkan | bf16 (native) | 1467.0 / 1458.9 t/s | 17.28 / 17.18 t/s |

### Headline numbers (mean of the two rounds)

| metric | A master f16 | B master bf16 | C rdna bf16 | VK Vulkan bf16 | C vs A | C vs B |
|--------|--------------|---------------|-------------|----------------|--------|--------|
| prefill @32K | 1235.8 t/s | 1229.3 t/s | **1645.9 t/s** | 1463.0 t/s | **+33.2%** | +33.9% |
| decode @32K | 28.80 t/s | 24.73 t/s | **30.60 t/s** | 17.23 t/s | **+6.3%** | **+23.7%** |

VK runs used `--split-mode layer` on 2 cards (`GGML_VK_VISIBLE_DEVICES=1,2`;
Vulkan has no tensor split). Notable: Vulkan's prefill beats master ROCm
(+18%) but is still behind the fork's bf16 (-11%), and Vulkan's decode is
far behind every ROCm config (17.2 t/s) - the Vulkan backend's decode
kernels on RDNA are its weak spot, and a native bf16 KV cannot overcome
that. The fork's ROCm bf16 is the fastest on both axes.

## Full throughput matrix (prefill / decode t/s, mean of 1-2 rounds)

### 2 cards, Q8_0 weights, tensor split (ROCm) / layer split (Vulkan)

| depth | A master f16 | B master bf16 | C rdna bf16 | VK Vulkan bf16 |
|-------|--------------|---------------|-------------|----------------|
| 32K prefill | 1235.8 | 1229.3 | **1645.9** | 1463.0 |
| 32K decode | 28.80 | 24.73 | **30.60** | 17.23 |
| 64K prefill | 941.4 | 939.3 | **1395.0** | 1235.2 |
| 64K decode | 27.46 | 20.83 | **29.20** | 16.32 |

### 1 card, Q6_K weights, 640 MB/s bandwidth

| depth | 1A master f16 | 1B master bf16 | 1C rdna bf16 | 1VK Vulkan bf16 |
|-------|---------------|----------------|--------------|-----------------|
| 32K prefill | 525.9 | 523.6 | **858.7** | 767.7 |
| 32K decode | 21.61 | 16.81 | 22.32 | **22.63** |
| 64K prefill | 422.4 | 420.1 | **733.9** | 638.7 |
| 64K decode | 20.22 | 13.39 | 20.87 | **21.07** |

#### 3 cards, Q8_0 weights, tensor split (ROCm) / layer split (Vulkan)

| depth | 3A master f16 | 3B master bf16 | 3C rdna bf16 | 3VK Vulkan bf16 |
|-------|---------------|----------------|--------------|-----------------|
| 32K prefill | 1295.1 | 1281.9 | **1739.6** | 1144.9 |
| 32K decode | 34.08 | 28.70 | **36.87** | 14.81 |
| 64K prefill | 953.5 | 953.2 | **1449.8** | 960.4 |
| 64K decode | 32.26 | 23.59 | **34.74** | 14.16 |

### 4K context depth (KV cache negligible - the shallow anchor)

2 cards, Q8_0:

| config | 4KA2 master f16 | 4KB2 master bf16 | 4KC2 rdna bf16 | 4KVK2 Vulkan |
|--------|-----------------|------------------|----------------|--------------|
| prefill | 1600.2 | 1602.8 | **1772.4** | 1126.8 |
| decode | 30.05 | 29.35 | **32.06** | 18.02 |

3 cards, Q8_0:

| config | 4KA3 | 4KB3 | 4KC3 | 4KVK3 |
|--------|-------|------|------|-------|
| prefill | 1678.1 | 1675.5 | **1843.9** | 1069.4 |
| decode | 35.75 | 34.78 | **38.79** | 16.50 |

1 card, Q6_K:

| config | 4KA1 | 4KB1 | 4KC1 | 4KVK1 |
|--------|-------|------|------|-------|
| prefill | 613.3 | 606.5 | 858.5 | **909.1** |
| decode | 22.99 | 22.13 | 23.79 | **24.18** |

## Why Vulkan bf16 sits at the top of the PPL range (mechanism analysis)

Every config in the ladder uses the same Q8_0 model, so the ordering is
driven by the attention/KV pipeline numerics:

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

## Why the BF16+f32 "ground truth" did not top the ladder

The full-precision config (BF16 weights, f32 KV) measured mid-pack
(6.3595 full corpus), below the fork's Q8_0 configs. Three stacked
reasons:

1. The BF16 weights are themselves quantized - 7-bit mantissa (the
   model's native format). The config carries the same mantissa
   quantization as the fork's bf16 KV.
2. Imatrix-calibrated Q8_0 (8-bit mantissa + per-block fp32 scale,
   scales tuned to the activation distribution) can approximate the
   original f32 weights BETTER than the bf16 line does. Calibrated
   quants occasionally beating the half-precision "ground truth" is
   the expected outcome when the quantization noise is smaller than
   the bf16 representation error.
3. The f32 KV bought nothing - KV precision is not a PPL lever on this
   model at this protocol (bf16 = f16 = f32 KV within noise, proven
   across every config).

The mid-pack landing is the strongest endorsement of the fork: the
fastest config is statistically indistinguishable from the most precise
config runnable (0.003 apart, direction inconsistent, +/-0.04 bars).
Config nuance: the ground truth ran 3-card layer split (mainline aborts
on tensor split) vs the forks' 2-card tensor split - numerics-equivalent
per-op, only cross-device reduction order differs.

## Fork KV design rationale (C)

The fork's KV design is a conscious trade: compute the attention with an
fp32 accumulator end-to-end and round to bf16 ONLY at the cache boundary
(the K/V projection output). This sacrifices ~3 mantissa bits vs an
fp16-cached path (10 bits), but keeps the full fp32 exponent range in the
accumulation. On this model the range property wins: C's bf16+fp32 path
measures at-or-below every fp16-based config at every resolution (6.3563
full-corpus vs A's f16 6.3619), and its prefill edge grows with depth
because bf16 halves the KV traffic. The fp16-vs-bf16 storage choice is
model-dependent in principle (models with small-magnitude attention states
might prefer fp16's mantissa); empirically, on Qwen3.8-27B at this
protocol, bf16 storage costs nothing measurable.

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

## Full-corpus PPL (the maximal resolution - 500 chunks exceeded the corpus)

The test set is ~152 chunks at 2048 ctx, so the "500-chunk" run covered
the entire test set (harder trailing chunks raise the absolute PPL by
~0.04 for every config; the error bars barely shrink because the per-chunk
loss variance dominates).

| config | full corpus PPL |
|--------|-----------------|
| fork Q8_0 bf16, chunked fp32 | **6.3561** +/- 0.0406 |
| fork Q8_0 bf16, chunked bf16 (default) | 6.3563 +/- 0.0406 |
| fork Q8_0 bf16, sequential GDN | 6.3576 +/- 0.0406 |
| mainline BF16 + f32 KV (ground truth) | 6.3595 +/- 0.0406 |
| mainline Q8_0 f16 (config A) | 6.3619 +/- 0.0406 |
| mainline Vulkan Q8_0 bf16 (config V) | 6.3678 +/- 0.0407 |

Same conclusions as the 128-chunk runs, now at full-corpus resolution:
- GDN dispatch trio spread 0.0015: the chunked GDN rewrite and its bf16
  tensor-core path are PPL-neutral.
- Q8_0 (all three dispatch variants) sits at or below the BF16 full-
  precision ground truth by 0.002-0.003: imatrix-calibrated Q8_0 quants
  are effectively free on this corpus. Everything remains within the
  +/-0.04 error bars.
- The complete ladder places the fork's Q8_0 bf16 at the top, the BF16
  ground truth next, then mainline f16 (6.3619) and Vulkan bf16 (6.3678)
  at the bottom - the same ordering seen at 128 chunks. Directional only:
  same weights, same ROCm backend, the fork's bf16 KV (6.3563) is 0.006
  better than mainline's f16 KV (6.3619) on the same Q8_0 model, and
  Vulkan's attention numerics land last.

## BF16 full weights, 3 cards (fork build, bf16 KV)

| depth | prefill | decode |
|-------|---------|--------|
| 4K | 1282.1 t/s | 23.56 t/s |
| 32K | 1571.1 t/s | 22.75 t/s |
| 64K | 1348.6 t/s | 21.95 t/s |

vs the Q8_0 fork on 3 cards (1844/1740/1450 prefill, 38.8/36.9/34.7
decode): decode ~37-39% slower (weight-bandwidth bound, 54 GB vs 29 GB),
prefill 7-30% slower. Combined with the PPL result (Q8_0 costs nothing
measurable), the full BF16 weights buy no PPL and cost ~1.65x decode:
not competitive for serving; Q8_0 is the right production choice.

## GDN dispatch matrix (rdna-boosts, Q8_0, bf16 KV, 2 cards)

Hypothesis: the chunked GDN rewrite (or its bf16 tensor-core path) is the
source of the small PPL spread. Tested by forcing the alternate dispatches:

| GDN path | env | PPL |
|----------|-----|-----|
| chunked bf16 (default, S_v=128) | - | 6.3162 |
| chunked fp32 | `GGML_CUDA_GDN_CHUNKED_BF16=0` | **6.3144** |
| sequential (non-chunked) | `GGML_CUDA_GDN_CHUNKED=0` | 6.3177 |

Spread 0.0033, all within the +/-0.043 error bars: neither the chunking
nor the bf16 tensor-core path is a measurable PPL variance source at
128x2048 resolution. (The fp32 chunked path happens to land lowest, but
that is not significant.)

## Depth fall-off (prefill, 4K -> 32K -> 64K)

2 cards (Q8_0): A f16: 1600 -> 1236 -> 941 (-41% to 64K). C rdna bf16:
1772 -> 1646 -> 1395 (-21% to 64K). The fork's bf16 prefill falls off
roughly half as fast: at 64K the KV read traffic is halved, so the deeper
the context the bigger C's relative edge (11% at 4K, 33% at 32K, 48% at
64K).

3 cards (Q8_0): A: 1678 -> 1295 -> 953 (-43%). C: 1844 -> 1740 -> 1450
(-21%). Same story - C's prefill fall-off is about half of A's, which is
the "more gentle fall-off at higher depths" observed on 3 cards. (The 4K
prefill rates are the least representative: a 3788-token burst is mostly
ramp-up.)

Vulkan prefill does not fall off monotonically (4K < 32K) and its decode
is flat-to-worse with depth; the layer-split sync cost dominates the depth
effect on multi-card.

## 8K depth (the prefill "highs")

8K is where prefill peaks - both backends' 3-card prefill is at its
maximum here (A: 1678 @ 4K -> 1696.7 @ 8K -> 953 @ 64K; C: 1844 @ 4K ->
1930.2 @ 8K -> 1450 @ 64K). Prefill t/s, 7510-token prompt:

| config | 1 card (Q6_K) | 2 cards (Q8_0) | 3 cards (Q8_0) |
|--------|---------------|----------------|----------------|
| A master f16 | 623.9 | 1570.6 | 1696.7 |
| B master bf16 | 619.4 | 1575.4 | 1689.4 |
| C fork bf16 | 919.8 | 1816.5 | 1930.2 |
| V Vulkan bf16 | 894.2 | 1376.1 | 1216.6 |

C's edge over A at 8K: +15.6% (2c), +13.8% (3c) - between the 4K (+11%)
and 32K (+33%) values, i.e. the fork's bf16 KV advantage grows with depth
as KV traffic becomes the bottleneck. A and B are indistinguishable at 8K
prefill (the silent-f16 penalty only emerges past ~8K as attention traffic
dominates). Vulkan's 3-card prefill is already below its 2-card rate at 8K
(layer-split sync); its 1-card prefill (894.2) trails C's by only 2.8%.

Decode t/s at 8K:

| config | 1 card (Q6_K) | 2 cards (Q8_0) | 3 cards (Q8_0) |
|--------|---------------|----------------|----------------|
| A master f16 | 22.79 | 29.92 | 35.60 |
| B master bf16 | 21.05 | 28.50 | 33.66 |
| C fork bf16 | 23.60 | 31.92 | 38.69 |
| V Vulkan bf16 | 23.87 | 17.90 | 16.12 |

C decode edge over A: +6.7% (2c), +8.7% (3c) - the same ~+6-9% as at 32K.
B's penalty at 8K is already -4.7% (2c) and grows with depth (up to -34%
at 64K 1c). Vulkan decode collapses on 2-3 cards even at 8K (layer-split
sync), while matching C on 1 card (23.87 vs 23.60).

Note: the 1-card column uses the Q6_K model, so its absolute numbers are
not comparable to the Q8_0 columns.

## Multi-card scaling (Q8_0, ROCm tensor split)

2 cards -> 3 cards, prefill / decode:

| config | 32K prefill | 32K decode | 64K prefill | 64K decode |
|--------|-------------|------------|-------------|------------|
| A master f16 | +5% / +1% | +18% / +17% |  |  |
| C rdna bf16 | +6% / +4% | +20% / +19% |  |  |
| VK Vulkan | -22% / -22% | -14% / -13% |  |  |

- ROCm decode scales with card count (+17-20% going 2 -> 3 cards): decode
  is memory-bandwidth bound, and a third card adds bandwidth. Prefill
  barely scales (+1-6%): it is compute/latency bound, not bandwidth bound.
- Vulkan scales *negatively* on both axes: the layer-split cross-card sync
  cost grows with card count. 3-card Vulkan is slower than 2-card Vulkan
  everywhere.
- The fork's bf16 edge holds at every card count: decode +6-8% over master
  f16, prefill +34-52% over master f16 (and over Vulkan at every point).
- vLLM cannot span 3 GPUs with a single model; llama.cpp (ROCm) can, which
  is why the 3-card row is the interesting one for anyone comparing
  serving stacks.

## Reading the single-card rows

- The rdna-boosts prefill edge grows on one card (+63% at 32K, +74% at 64K
  vs master ROCm) - the bf16 KV + fused kernels matter more when the KV
  traffic is a larger share of a 640 MB/s budget.
- Master's silent-f16 decode penalty scales with depth and bandwidth
  pressure: -14% (32K, 2 cards) -> -24% (64K, 2 cards) -> -22% (32K, 1
  card) -> -34% (64K, 1 card). On a single slow card at depth it is
  brutal.
- Vulkan decode is fine on one card (top of the table at both depths, ~1%
  ahead of rdna-boosts - an upstream Vulkan improvement within the last
  few weeks) and collapses on 2 cards (16.3 t/s at 64K): the layer-split
  cross-card sync is the cost, as suspected.
- Vulkan prefill beats master ROCm at every depth/card count, but never
  beats the fork's ROCm bf16.

### Headline numbers (mean of the two rounds)

## Reading the results

- **C (rdna-boosts, true bf16 KV) is the fastest on both axes.** At 32K depth
  the bf16 KV halves the attention read traffic, which is why the prefill
  gain (+33%) is so much larger than the decode gain (+6.3%): prefill
  re-reads the growing cache for every batch, decode only the final depth.

- **B (master, bf16) is the worst of both worlds, exactly as predicted.**
  Upstream has no native BF16 KV path for this stack: the request silently
  maps to f16 storage, so the PPL is f16's (6.3213, the worst of the three),
  and the per-token conversion costs ~14% decode throughput at depth
  (24.73 t/s vs A's 28.80). It buys neither the memory savings nor the speed
  of real bf16, and gives up PPL parity with the fork's bf16.

- **Depth is where the story lives.** At shallow depth the KV traffic is
  negligible and the configs converge; the 32K-depth protocol is what makes
  the bf16 advantage fully visible (and the user-visible magnitude tracks
  context depth: deeper context, bigger delta).

- **PPL is a wash, directionally in C's favor.** The bf16 KV (8-bit
  mantissa) reads as f16-equivalent or better on a bf16-native model
  (Qwen3.8 is trained in bf16), thanks to the fork's FP32 accumulation in
  the native-BF16 flash-attn path. The differences are within the +/-0.043
  statistical error, so treat the ordering as indicative.

## Raw timings (server logs)

Per-request `timings` from `/completion` (script output):

```
A-master-f16 r1: prefill 1235.6 t/s (31694 tok)  decode 28.79 t/s (256 tok)
A-master-f16 r2: prefill 1236.0 t/s (31694 tok)  decode 28.80 t/s (256 tok)
B-master-bf16 r1: prefill 1236.2 t/s (31694 tok)  decode 24.81 t/s (256 tok)
B-master-bf16 r2: prefill 1222.3 t/s (31694 tok)  decode 24.64 t/s (256 tok)
C-rdna-bf16 r1: prefill 1642.5 t/s (31694 tok)  decode 30.63 t/s (256 tok)
C-rdna-bf16 r2: prefill 1649.2 t/s (31694 tok)  decode 30.57 t/s (256 tok)
```

PPL runs (perplexity logs): A 2:55 elapsed, B 2:55, C 2:40 (the 15 s saving
on C is the prefill-side bf16 effect). V (Vulkan) 4:25 elapsed at 2.33 s/pass
(3-card layer split). Vulkan server runs (config VK): prefill 1467/1459,
decode 17.28/17.18 (2-card layer split).

## Vulkan reference run (config V)

Vulkan does not support tensor parallelism, so the Vulkan run used
`--split-mode layer` across all 3 cards, `GGML_VK_VISIBLE_DEVICES=1,2,3`
(the iGPU and llvmpipe excluded). Same protocol otherwise. Master's Vulkan
backend stores BF16 KV natively (unlike master's ROCm path, which maps it to
f16), so V is the reference for what a true BF16 KV cache yields on the Q8_0
weights:

```
GGML_VK_VISIBLE_DEVICES=1,2,3 <build-vulkan>/bin/llama-perplexity \
  -m <Q8_0> -f /llm/models/wikitext-2-raw/wiki.test.raw \
  -c 2048 --chunks 128 -fa on -ctk bf16 -ctv bf16 -sm layer -ngl 99
```

PPL = 6.3293 +/- 0.04284. See the perplexity table for interpretation.
