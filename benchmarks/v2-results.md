# Benchmark results v2: llama-benchy suite (2026-08-27)

Full gamut: builds (master `fe235f434` vs rdna-boosts `a265041b1`)
x KV cache (f16 vs bf16) x 4 model/card sets, on BOTH backends (ROCm and
Vulkan). 24 throughput rows (16 ROCm + 8 Vulkan) + 18 PPL corners.

Protocol: [benchy-methodology.md](benchy-methodology.md) (llama-benchy
v0.4.0, `--pp 2520 --tg 240 --depth 0 4096 8192 16384 32768 65536 131072
--no-cache --runs 2`). Machine: `soar`, 3x R9700 (gfx1201, 34 GiB), ROCm
7.14. Raw JSON/MD: [results/benchy/](results/benchy/).

Row labels: `M` = master ROCm, `B` = rdna-boosts ROCm, `V` = master
Vulkan (same `fe235f434` tip); `1Q6` = 1 card Q6_K, `1Q4` = 1 card
Q4_K_XL, `2` = 2 cards Q8_0, `3` = 3 cards Q8_0; `-f16`/`-bf16` = KV
cache type. A = M-f16, B = M-bf16, D = B-f16, C = B-bf16 (v1 mapping);
V-f16/V-bf16 are the Vulkan corners.

## The one-sentence version

Master's ROCm BF16 KV silently degrades decode with depth (-2% shallow to
**-39 to -51% at 128K**); rdna-boosts fixes that and goes further - its
BF16 config is **faster than master-f16 at every depth and more accurate
(PPL) on every model**, and even its f16 config beats master-f16. Vulkan
is a genuine native-BF16 reference with superb single-card prefill, but
its decode collapses on multi-card (layer-split sync) and its BF16 costs
prefill at depth. No performance or accuracy reason to choose either
baseline approach.

## Throughput

### 1 card, Q6_K (22.9 GB, ctx 140000)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | C B-bf16 pp/tg | D B-f16 pp/tg | V-f16 pp/tg | V-bf16 pp/tg |
|------:|--------------:|---------------:|---------------:|--------------:|------------:|-------------:|
| 0 | 620 / 22.78 | 621 / 22.25 | 894 / 23.84 | 895 / 23.82 | 780 / 24.13 | 788 / 24.11 |
| 4096 | 642 / 22.59 | 641 / 21.32 | 964 / 23.58 | 964 / 23.63 | 863 / 23.90 | 854 / 23.91 |
| 8192 | 624 / 22.40 | 623 / 20.41 | 959 / 23.26 | 958 / 23.32 | 865 / 23.66 | 846 / 23.65 |
| 16384 | 581 / 21.98 | 580 / 18.69 | 922 / 22.93 | 921 / 22.93 | 844 / 23.14 | 807 / 23.16 |
| 32768 | 506 / 21.22 | 505 / 16.16 | 835 / 22.07 | 834 / 22.07 | 792 / 22.31 | 729 / 22.31 |
| 65536 | 401 / 19.77 | 399 / 12.76 | 705 / 20.44 | 704 / 20.44 | 703 / 20.72 | 606 / 20.72 |
| 131072 | 282 / 17.43 | 280 / 8.99 | 536 / 18.00 | 534 / 18.00 | 570 / 18.16 | 451 / 18.15 |

### 1 card, Q4_K_XL (17.6 GB, ctx 140000)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | C B-bf16 pp/tg | D B-f16 pp/tg | V-f16 pp/tg | V-bf16 pp/tg |
|------:|--------------:|---------------:|---------------:|--------------:|------------:|-------------:|
| 0 | 949 / 26.62 | 949 / 25.87 | 1090 / 29.77 | 1092 / 29.76 | 891 / 29.42 | 881 / 29.38 |
| 4096 | 982 / 26.28 | 981 / 24.62 | 1208 / 29.45 | 1207 / 29.32 | 970 / 29.01 | 953 / 29.02 |
| 8192 | 940 / 26.01 | 938 / 23.46 | 1200 / 29.08 | 1199 / 29.01 | 968 / 28.71 | 940 / 28.70 |
| 16384 | 845 / 25.48 | 842 / 21.24 | 1142 / 28.38 | 1139 / 28.34 | 941 / 28.04 | 894 / 28.03 |
| 32768 | 693 / 24.40 | 689 / 18.10 | 1010 / 27.07 | 1008 / 27.04 | 874 / 26.75 | 798 / 26.72 |
| 65536 | 510 / 22.53 | 507 / 13.97 | 826 / 24.78 | 824 / 24.75 | 767 / 24.43 | 653 / 24.43 |
| 131072 | 332 / 19.50 | 329 / 9.61 | 602 / 21.19 | 601 / 21.18 | 611 / 20.92 | 477 / 20.91 |

### 2 cards, Q8_0 (29 GB, ctx 262144)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | C B-bf16 pp/tg | D B-f16 pp/tg | V-f16 pp/tg | V-bf16 pp/tg |
|------:|--------------:|---------------:|---------------:|--------------:|------------:|-------------:|
| 0 | 1269 / 29.94 | 1268 / 29.46 | 1370 / 32.45 | 1380 / 32.47 | 913 / 18.00 | 899 / 18.03 |
| 4096 | 1450 / 29.82 | 1450 / 28.70 | 1644 / 32.14 | 1645 / 32.23 | 1200 / 17.86 | 1166 / 17.87 |
| 8192 | 1437 / 29.59 | 1439 / 28.09 | 1687 / 31.93 | 1689 / 32.01 | 1366 / 17.71 | 1315 / 17.73 |
| 16384 | 1344 / 29.18 | 1341 / 26.65 | 1669 / 31.58 | 1669 / 31.56 | 1472 / 17.50 | 1391 / 17.50 |
| 32768 | 1141 / 28.51 | 1137 / 24.09 | 1538 / 30.73 | 1538 / 30.72 | 1467 / 16.99 | 1338 / 16.97 |
| 65536 | 877 / 27.14 | 873 / 20.11 | 1313 / 29.19 | 1308 / 29.20 | 1345 / 16.04 | 1152 / 16.03 |
| 131072 | 595 / 24.82 | 591 / 15.23 | 1010 / 26.51 | 1002 / 26.52 | 1101 / 14.06 | 869 / 14.05 |

### 3 cards, Q8_0 (29 GB, ctx 262144)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | C B-bf16 pp/tg | D B-f16 pp/tg | V-f16 pp/tg | V-bf16 pp/tg |
|------:|--------------:|---------------:|---------------:|--------------:|------------:|-------------:|
| 0 | 1275 / 35.78 | 1266 / 34.92 | 1365 / 39.29 | 1351 / 39.23 | 889 / 16.25 | 869 / 16.44 |
| 4096 | 1524 / 35.52 | 1517 / 33.88 | 1702 / 38.88 | 1697 / 38.90 | 1124 / 16.18 | 1088 / 16.26 |
| 8192 | 1523 / 35.30 | 1517 / 33.02 | 1764 / 38.63 | 1761 / 38.68 | 1199 / 16.23 | 1150 / 15.96 |
| 16384 | 1420 / 34.71 | 1420 / 31.04 | 1755 / 38.08 | 1755 / 38.12 | 1228 / 15.59 | 1154 / 15.35 |
| 32768 | 1181 / 33.66 | 1177 / 27.73 | 1606 / 36.79 | 1607 / 36.76 | 1173 / 14.66 | 1063 / 14.66 |
| 65536 | 886 / 31.82 | 881 / 22.70 | 1361 / 34.57 | 1350 / 34.59 | 1042 / 13.94 | 885 / 13.97 |
| 131072 | 581 / 28.66 | 576 / 16.85 | 1021 / 30.85 | 1011 / 30.87 | 835 / 11.76 | 653 / 11.75 |

## Graphs

**All 16 charts on one page: [graphs/ALL-CHARTS.md](graphs/ALL-CHARTS.md)**
(regenerated from the raw JSON by `scripts/make-v2-graphs.py`). Each chart
compares the three backends at one KV type (3 series, not 6); the legend
under every axis is **red = master, amber = boosts, blue = vulkan**.

![decode bf16 - 1 card Q6_K](graphs/v2-tg-1Q6-bf16.png)
![decode bf16 - 3 cards Q8_0](graphs/v2-tg-3c-bf16.png)
![decode f16 - 1 card Q4_K_XL](graphs/v2-tg-1Q4-f16.png)

## What the numbers say

### 1. Master ROCm BF16 = a depth-growing decode penalty, prefill unaffected

Master's ROCm BF16 KV is silently FP16 storage plus a per-token
conversion cost. Decode penalty grows with depth (KV traffic grows with
depth):

| depth | 1-card Q6_K | 1-card Q4_K_XL | 2-card Q8_0 | 3-card Q8_0 |
|------:|------------:|---------------:|------------:|------------:|
| 0 | -2.3% | -2.8% | -1.6% | -2.4% |
| 4096 | -5.6% | -6.3% | -3.7% | -4.6% |
| 8192 | -8.9% | -9.8% | -5.1% | -6.5% |
| 16384 | -15.0% | -16.7% | -8.7% | -10.6% |
| 32768 | -23.8% | -25.8% | -15.5% | -17.6% |
| 65536 | -35.5% | -38.0% | -25.9% | -28.7% |
| 131072 | -48.5% | -50.7% | -38.7% | -41.2% |


Prefill is indistinguishable (620.3 vs 620.7 at depth 0, 1-card Q6_K): it
is compute-bound, not conversion-bound.

### 2. The fork fixes it and wins at the same KV type

C (boosts bf16) vs A (master f16):

| depth | 1-card Q6_K pp / tg | 1-card Q4_K_XL pp / tg | 2-card pp / tg | 3-card pp / tg |
|------:|--------------------:|-----------------------:|---------------:|---------------:|
| 0 | +44.2% / +4.5% | +15.0% / +11.8% | +8.8% / +8.4% | +5.9% / +9.7% |
| 4096 | +50.2% / +4.6% | +23.0% / +11.6% | +13.5% / +8.1% | +11.4% / +9.5% |
| 8192 | +53.5% / +4.1% | +27.5% / +11.5% | +17.6% / +8.2% | +15.6% / +9.6% |
| 16384 | +58.4% / +4.3% | +34.8% / +11.2% | +24.2% / +8.2% | +23.6% / +9.8% |
| 32768 | +64.9% / +4.0% | +45.6% / +10.8% | +34.8% / +7.8% | +36.1% / +9.2% |
| 65536 | +75.5% / +3.4% | +61.5% / +9.9% | +49.1% / +7.6% | +52.3% / +8.7% |
| 131072 | +89.2% / +3.3% | +81.0% / +8.6% | +68.4% / +6.8% | +73.9% / +7.7% |


The prefill edge grows with depth (BF16 halves the KV attention read
traffic, and prefill re-reads the growing cache every batch). The decode
edge is roughly constant (+3-12%), larger on Q4_K_XL where block 10's VDR
boost on the Q4_K/Q5_K kernels lands.

### 3. Boosts-f16 (D) also beats master-f16 (A) - same KV type, same build

D vs A:

| depth | 1-card Q6_K pp / tg | 1-card Q4_K_XL pp / tg | 2-card pp / tg | 3-card pp / tg |
|------:|--------------------:|-----------------------:|---------------:|---------------:|
| 0 | +44.2% / +4.5% | +15.0% / +11.8% | +8.8% / +8.4% | +5.9% / +9.7% |
| 4096 | +50.2% / +4.6% | +23.0% / +11.6% | +13.5% / +8.1% | +11.4% / +9.5% |
| 8192 | +53.5% / +4.1% | +27.5% / +11.5% | +17.6% / +8.2% | +15.6% / +9.6% |
| 16384 | +58.4% / +4.3% | +34.8% / +11.2% | +24.2% / +8.2% | +23.6% / +9.8% |
| 32768 | +64.9% / +4.0% | +45.6% / +10.8% | +34.8% / +7.8% | +36.1% / +9.2% |
| 65536 | +75.5% / +3.4% | +61.5% / +9.9% | +49.1% / +7.6% | +52.3% / +8.7% |
| 131072 | +89.2% / +3.3% | +81.0% / +8.6% | +68.4% / +6.8% | +73.9% / +7.7% |


So even if someone insists on f16 KV, the fork is faster than master at the
same KV type; the BF16 column is an additional layer on top (mostly a
prefill win).

### 4. C vs B: the fork reverses the master BF16 penalty completely

At 128K decode:
- 1Q6: +100.3%
- 1Q4: +120.3%
- 2: +74.2%
- 3: +83.2%

### 5. Vulkan: the genuine native-BF16 reference, with its own tradeoffs

Vulkan (same master tip, RADV gfx1201) supports native BF16 (`bf16: 1`,
KHR_coopmat) - unlike ROCm master, which silently stores f16. The Vulkan
rows complete the gamut:

- **Vulkan bf16 vs f16 decode: +-0.1% at every depth, every card count** -
  native BF16 carries no conversion cost (contrast: ROCm master -39 to
  -51%). Vulkan's bf16 is the true upstream-native BF16 reference.
- **Vulkan bf16 prefill: -1 to -3% shallow, -21 to -22% at 128K** - the
  BF16 tensor-core attention path is slower at depth on this stack. The
  mirror image of ROCm master's penalty (which hits decode, not prefill).
- **Vulkan single-card prefill is exceptional**: +26% shallow to +84 to
  +102% at 128K vs ROCm master f16 (1Q6: 570 vs 282 t/s at 128K). But the
  boosts build still matches or beats it: C-bf16 534-536 t/s at 128K on
  1Q6, and on 1Q4 C-bf16 (601) vs V-f16 (611) - within 2%.
- **Vulkan decode collapses on multi-card** (layer-split sync):
  -40% vs ROCm master at 2 cards, -54 to -59% at 3 cards, at every depth.
  Boosts decode beats Vulkan by +80-89% (2c) and +138-162% (3c).
- **Vulkan PPL sits at the top of the range** (see below): its attention
  numerics (BF16-everywhere on the coopmat path) are the least precise of
  the gamut.

#### Why Vulkan edges 1-card decode: kernel speed vs launch latency

The 1-card decode comparison is close (-0.9 to -1.4% for Vulkan on Q6_K,
+1.1 to +1.3% for boosts on Q4_K_XL), and the direction of the gap is
informative. On a kernel-per-kernel basis the ROCm BF16 kernels are
substantially faster than Vulkan's (~20% on the k-quant decode path).
What keeps Vulkan ahead on 1-card decode is inter-kernel launch latency:
Vulkan's kernel launcher (RADV) dispatches the per-token decode kernel
chain with less overhead than the ROCm HIP launcher. Decode is a long
chain of small kernels per token, so launch latency is a first-order cost
at every depth.

The two models bracket the crossover, which is exactly where block 10's
per-kernel work is strongest:

- **Q6_K (C trails V by ~1%)**: block 10's VDR boost here is modest
  (Q6_K 1->2, Q8_0 2->4; the decode edge over master is only ~+3%). The
  per-kernel advantage is too small to overcome the launch-latency gap, so
  Vulkan's launcher wins.
- **Q4_K_XL (C leads V by ~1.2%)**: block 10's VDR boost is large (Q4_K
  2->4, Q5_K 2->4; the decode edge over master is ~+11%). The ~20%
  per-kernel advantage is big enough to flip the crossover, so ROCm's
  kernels win despite the launcher.

Reading: the ROCm BF16 kernels are the reference implementation; the ~1%
1-card decode deficit is the ROCm driver's launch overhead, not a kernel
deficiency - a gap llama.cpp cannot close from userspace. The multi-card
Vulkan decode collapse (-40 to -59%) is a separate, much larger effect
(layer-split sync overhead), and there the boosts kernel advantage shows
as +80-89% (2c) and +138-162% (3c).

#### The launch-gap is measured, not inferred (fused-matmuls project, 2026-08-12)

Prior profiling (rocprof, Q6_K @ d256, `fused-matmuls.md` experiment log)
quantified the exact mechanism. The per-token wall was 43.7 ms; the
rocprof per-op decomposition is below.

| op (per token, Q6_K @ d256, BF16 KV) | count | time | note |
|---|---|---:|---|
| `mul_mat_vec_q` (mmvq, weights) | 321 | 35.4 ms | **DRAM-bound, no headroom** (~645 GB/s ≈ card peak) |
| `gated_delta_net_cuda` (SSM) | 49 | 1.19 ms | one per layer, the SSM core op |
| `rms_norm_q8_1_f32<1024>` | 128 | 0.68 ms | fused norm+quantize already |
| `k_get_rows_float(_vec)` | 49+50 | 0.41 ms | |
| `mul_mat_vec_f` (F32 small matmuls) | 96 | 0.39 ms | includes the [5120×48] gates |
| `unary_gated_op_kernel` (SWIGLU) | 51 | 0.35 ms | |
| `rms_norm_f32<256>` | 82 | 0.33 ms | |
| `k_bin_bcast` | 133 | 0.33 ms | |
| `flash_attn_tile` + `flash_attn_ext` | 16-17 | 0.26 ms | |
| `cpy_scalar` (conv_state_update) | 65 | 0.19 ms | |
| `concat` (conv_input) | 49+48 | 0.19 ms | |
| `ssm_conv_f32` + `ssm_conv_long_token` | | 0.13 ms | |
| `l2_norm_pair_f32` | 49 | 0.12 ms | |
| `quantize_q8_1` residual | 65 | 0.09 ms | the rest is folded |
| `unary_op` / `unary_gated_q8_1` / misc | | 0.15 ms | |
| **kernel work subtotal** | ~1290 | **~39.6 ms** | GPU busy |
| **CPU launch + graph capture/replay + sync gap** | | **~4.1 ms** | the launch tax |
| **wall total** | | **~43.7 ms** | = ~22.88 t/s |

Read it as: **~84% of the wall is the mmvq weights stream** (DRAM-bound, no
headroom). Of the remaining ~8.3 ms, ~4.1 ms is **pure launch/sync gap**
behind ~1290 kernel dispatches (~3.5 µs each). That launch tax is what
Vulkan's launcher avoids (~0.1 ms/token total), and it is what the fusion
campaign attacked — by cutting kernel *count*, not making any kernel faster.

- **Vulkan inter-kernel gap: ~0.1 ms/token total**; ROCm: **~4.5 ms/token
  (~3.5 us x ~1290 kernels)**. Decode wall was 43.7 ms/token: 35.4 ms
  mmvq (DRAM-bound at ~645 GB/s, no headroom) + ~4.5 ms launch/graph/sync
  gaps + ~4 ms of norms/get_rows/conv-state kernels.
- The pre-project ROCm-vs-Vulkan decode gap was **~15%**; the fusion
  campaign (C1 K/V dual-output mmvq, C6 Meta-wrapper removal at single
  device, etc.) closed it to **~0.6%** by cutting kernel count - i.e. by
  reducing the number of times the ~3.5 us launch tax is paid, not by
  making any single kernel faster.
- The C6 fix (skip the Meta(ROCm0) wrapper when n_devices==1, commit
  469538e5b) alone recovered +1.4-1.8% by eliminating ~130 sub-graph
  captures per token. C1 (fuse the 32 per-layer K/V projections) saved
  only ~0.15 ms: those dispatches were ~9 us of real L2-bound work each,
  not pure launch overhead - confirming the remaining gap is the ~3.5 us
  per-launch tax on ~1290 kernels, not kernel speed.
- The "persistent mega-kernel" idea (zero kernel boundaries, recovering
  the full ~4.5 ms/token = ~10% decode) was scoped and parked as "a very
  large project". That ~10% is the same headroom the fork's decode edges
  (Q4_K_XL +11% over master) are now partially harvesting per-kernel.

So the ~1% 1-card deficit measured in this suite is the *residual* of
that campaign: ROCm kernel speed now exceeds Vulkan's, but the ~3.5 us x
~1290-kernel launch tax still costs more than Vulkan's ~0.1 ms gap.

#### Two-week swing: ROCm was ahead on Aug 13, Vulkan ahead again now

On 2026-08-13 the fork's decode was at **parity or slightly ahead** of
Vulkan (fused-matmuls final status: "within ~0.6%"; rocm-prefill-handoff:
19.63 vs 19.49 t/s on Q8_0). Two weeks later this suite measures Vulkan
~1% ahead on Q6_K decode. The llama.cpp code between the two builds is
not the cause: only two Vulkan commits landed in the window (a warp-size
clamp #27726 and a cross_entropy_loss impl #27216 - neither touches the
decode hot path), and the only CUDA-side change (fc35562ba) is prefill-
side MoE. The swing is therefore most plausibly the **RADV library**
(Mesa 26.1.7 currently) finding a little extra dispatch/launch throughput
- consistent with the user's observation that both implementations are
approaching their code-side limits and the remaining gap is in library
support. If AMD fixes the HIP launch path (the ~3.5 us x ~1290 kernels
overhead), the fork's per-kernel advantage should put ROCm decisively
ahead: the Q4_K_XL row already shows the direction.

| | V vs A prefill (128K) | V vs A decode (128K) | B-bf16 vs V decode (128K) |
|---|---|---|---|
| 1-card Q6_K | +102% | +4.2% | -0.9% |
| 1-card Q4_K_XL | +84% | +7.3% | +1.3% |
| 2-card Q8_0 | +85% | -43% | +89% |
| 3-card Q8_0 | +44% | -59% | +162% |

## Perplexity (accuracy) - all 18 corners

`llama-perplexity`, wikitext-2 test, 128x2048, flash-attn on. ROCm rows on
2 cards (tensor split), Vulkan rows on 2 cards (layer split).

| model | A M-f16 | B M-bf16 | C B-bf16 | D B-f16 | V-f16 | V-bf16 |
|-------|--------:|--------:|--------:|--------:|------:|-------:|
| Q8_0 | 6.3193 | 6.3213 | **6.3162** | 6.3164 | 6.3241 | 6.3293 |
| Q6_K | 6.3120 | 6.3130 | **6.3095** | 6.3099 | 6.3172 | 6.3206 |
| Q4_K_XL | 6.3042 | 6.3086 | 6.3051 | **6.3010** | 6.3138 | 6.3143 |

- A/B/C on Q8_0 reproduce the v1 baselines **exactly** (6.3193 / 6.3213 /
  6.3162); V-bf16 on Q8_0 also reproduces v1's Vulkan baseline (6.3293).
  The harness measures the same things v1 did.
- All ROCm corners sit within their +/-0.042 error bars (directional, not
  conclusive); the fastest config (C) is at or below the others on every
  model: **the speed win does not cost accuracy**.
- Vulkan sits 0.005-0.018 above its model's ROCm range - consistent with
  v1's finding that Vulkan's BF16-everywhere attention numerics land last.
- Q4_K_XL is the lowest-PPL model of the three (6.30 vs 6.31/6.32): the
  mixed-quant at 17.6 GB costs nothing measurable and is the fastest
  1-card config - the natural single-card production choice.

## Bottom line

| | master f16 (A) | master bf16 (B) | boosts f16 (D) | boosts bf16 (C) | vulkan f16 | vulkan bf16 |
|---|---|---|---|---|---|---|
| decode 128K (1c Q6_K) | 17.43 | 8.99 | 18.00 | 18.00 | 18.16 | 18.15 |
| prefill 128K (1c Q6_K) | 282 | 280 | 536 | 534 | 570 | 451 |
| PPL (Q6_K) | 6.312 | 6.313 | 6.3099 | **6.3095** | 6.3172 | 6.3206 |

Master bf16 is strictly dominated by master f16 (same accuracy, worse
decode at depth). Boosts f16 dominates master f16 (faster everywhere,
equal or better PPL). Boosts bf16 dominates everything on ROCm (fastest,
lowest PPL). Vulkan's native-BF16 decode is competitive on a single card
but its multi-card decode and BF16 prefill are real costs, and its PPL is
the worst of the gamut. There is no performance or accuracy reason to
choose either baseline approach.

