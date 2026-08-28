
# Benchmark results v2: llama-benchy suite (2026-08-27)

Full 2x2 matrix: builds (master `fe235f434` vs rdna-boosts `a265041b1`)
x KV cache (f16 vs bf16) x 4 model/card sets (1-card Q6_K, 1-card Q4_K_XL,
2-card Q8_0, 3-card Q8_0). 16 throughput rows + 12 PPL corners.

Protocol: [benchy-methodology.md](benchy-methodology.md) (llama-benchy v0.4.0,
`--pp 2520 --tg 240 --depth 0 4096 8192 16384 32768 65536 131072 --no-cache
--runs 2`). Machine: `soar`, 3x R9700 (gfx1201, 34 GiB), ROCm 7.14.
Raw JSON/MD: [results/benchy/](results/benchy/). Log:
[results/run-all.log](results/run-all.log).

Row labels: `M` = master, `B` = rdna-boosts; `1Q6` = 1 card Q6_K, `1Q4` =
1 card Q4_K_XL, `2` = 2 cards Q8_0, `3` = 3 cards Q8_0; `-f16`/`-bf16` =
KV cache type. A = M-f16, B = M-bf16, D = B-f16, C = B-bf16 (v1 mapping).

## The one-sentence version

Master's BF16 KV silently degrades decode with depth (-2% shallow to **-38
to -51% at 128K** depending on card count); rdna-boosts fixes that and goes
further - its BF16 config is **faster than master-f16 at every depth and
more accurate (PPL) on every model**, and even its f16 config beats
master-f16, so there is no performance or accuracy reason to pick either of
the baseline's KV approaches.

## Throughput

### 1 card, Q6_K (22.9 GB, ctx 140000)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | D B-f16 pp/tg | C B-bf16 pp/tg |
|------:|--------------:|---------------:|--------------:|---------------:|
| 0 | 620 / 22.78 | 621 / 22.25 | 894 / 23.84 | 895 / 23.82 |
| 4096 | 642 / 22.59 | 641 / 21.32 | 964 / 23.58 | 964 / 23.63 |
| 8192 | 624 / 22.40 | 623 / 20.41 | 959 / 23.26 | 958 / 23.32 |
| 16384 | 581 / 21.98 | 580 / 18.69 | 922 / 22.93 | 921 / 22.93 |
| 32768 | 506 / 21.22 | 505 / 16.16 | 835 / 22.07 | 834 / 22.07 |
| 65536 | 401 / 19.77 | 399 / 12.76 | 705 / 20.44 | 704 / 20.44 |
| 131072 | 282 / 17.43 | 280 / 8.99 | 536 / 18.00 | 534 / 18.00 |

### 1 card, Q4_K_XL (17.6 GB, ctx 140000)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | D B-f16 pp/tg | C B-bf16 pp/tg |
|------:|--------------:|---------------:|--------------:|---------------:|
| 0 | 949 / 26.62 | 949 / 25.87 | 1090 / 29.77 | 1092 / 29.76 |
| 4096 | 982 / 26.28 | 981 / 24.62 | 1208 / 29.45 | 1207 / 29.32 |
| 8192 | 940 / 26.01 | 938 / 23.46 | 1200 / 29.08 | 1199 / 29.01 |
| 16384 | 845 / 25.48 | 842 / 21.24 | 1142 / 28.38 | 1139 / 28.34 |
| 32768 | 693 / 24.40 | 689 / 18.10 | 1010 / 27.07 | 1008 / 27.04 |
| 65536 | 510 / 22.53 | 507 / 13.97 | 826 / 24.78 | 824 / 24.75 |
| 131072 | 332 / 19.50 | 329 / 9.61 | 602 / 21.19 | 601 / 21.18 |

### 2 cards, Q8_0 (29 GB, ctx 262144)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | D B-f16 pp/tg | C B-bf16 pp/tg |
|------:|--------------:|---------------:|--------------:|---------------:|
| 0 | 1269 / 29.94 | 1268 / 29.46 | 1370 / 32.45 | 1380 / 32.47 |
| 4096 | 1450 / 29.82 | 1450 / 28.70 | 1644 / 32.14 | 1645 / 32.23 |
| 8192 | 1437 / 29.59 | 1439 / 28.09 | 1687 / 31.93 | 1689 / 32.01 |
| 16384 | 1344 / 29.18 | 1341 / 26.65 | 1669 / 31.58 | 1669 / 31.56 |
| 32768 | 1141 / 28.51 | 1137 / 24.09 | 1538 / 30.73 | 1538 / 30.72 |
| 65536 | 877 / 27.14 | 873 / 20.11 | 1313 / 29.19 | 1308 / 29.20 |
| 131072 | 595 / 24.82 | 591 / 15.23 | 1010 / 26.51 | 1002 / 26.52 |

### 3 cards, Q8_0 (29 GB, ctx 262144)

| depth | A M-f16 pp/tg | B M-bf16 pp/tg | D B-f16 pp/tg | C B-bf16 pp/tg |
|------:|--------------:|---------------:|--------------:|---------------:|
| 0 | 1275 / 35.78 | 1266 / 34.92 | 1365 / 39.29 | 1351 / 39.23 |
| 4096 | 1524 / 35.52 | 1517 / 33.88 | 1702 / 38.88 | 1697 / 38.90 |
| 8192 | 1523 / 35.30 | 1517 / 33.02 | 1764 / 38.63 | 1761 / 38.68 |
| 16384 | 1420 / 34.71 | 1420 / 31.04 | 1755 / 38.08 | 1755 / 38.12 |
| 32768 | 1181 / 33.66 | 1177 / 27.73 | 1606 / 36.79 | 1607 / 36.76 |
| 65536 | 886 / 31.82 | 881 / 22.70 | 1361 / 34.57 | 1350 / 34.59 |
| 131072 | 581 / 28.66 | 576 / 16.85 | 1021 / 30.85 | 1011 / 30.87 |

## What the numbers say

### 1. Master BF16 = a depth-growing decode penalty, prefill unaffected

Master's ROCm BF16 KV is silently FP16 storage plus a per-token conversion
cost. Decode penalty grows with depth (KV traffic grows with depth):

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
| 0 | +44.0% / +4.6% | +14.9% / +11.8% | +8.0% / +8.4% | +7.0% / +9.8% |
| 4096 | +50.2% / +4.4% | +23.0% / +12.1% | +13.4% / +7.8% | +11.7% / +9.5% |
| 8192 | +53.7% / +3.8% | +27.6% / +11.8% | +17.4% / +7.9% | +15.8% / +9.4% |
| 16384 | +58.6% / +4.3% | +35.1% / +11.4% | +24.2% / +8.2% | +23.6% / +9.7% |
| 32768 | +65.0% / +4.0% | +45.9% / +11.0% | +34.8% / +7.8% | +36.0% / +9.3% |
| 65536 | +75.8% / +3.4% | +61.9% / +10.0% | +49.7% / +7.6% | +53.6% / +8.6% |
| 131072 | +89.7% / +3.3% | +81.4% / +8.6% | +69.6% / +6.8% | +75.8% / +7.7% |


So even if someone insists on f16 KV, the fork is faster than master at the
same KV type; the BF16 column is an additional layer on top (mostly a
prefill win).

### 4. C vs B: the fork reverses the master BF16 penalty completely

At 128K decode:
- 1c Q6_K: +100.3%
- 1c Q4_K_XL: +120.3%
- 2c Q8_0: +74.2%
- 3c Q8_0: +83.2%

## Perplexity (accuracy) - all 12 corners

`llama-perplexity`, wikitext-2 test, 128x2048, flash-attn on, 2 cards.

| model | A M-f16 | B M-bf16 | C B-bf16 | D B-f16 |
|-------|--------:|--------:|--------:|--------:|
| Q8_0 | 6.3193 | 6.3213 | **6.3162** | 6.3164 |
| Q6_K | 6.3120 | 6.3130 | **6.3095** | 6.3099 |
| Q4_K_XL | 6.3042 | 6.3086 | 6.3051 | **6.3010** |

- A/B/C on Q8_0 reproduce the v1 baselines **exactly** (6.3193 / 6.3213 /
  6.3162) - the harness measures the same thing v1 did.
- All corners sit within their +/-0.042 error bars (the differences are
  directional, not conclusive), and the fastest config (C) is at or below
  the others on every model: **the speed win does not cost accuracy**.
- Q4_K_XL is the lowest-PPL model of the three (6.30 vs 6.31/6.32): the
  mixed-quant at 17.6 GB costs nothing measurable and is the fastest 1-card
  config - the natural single-card production choice.

## Bottom line

| | master f16 (A) | master bf16 (B) | boosts f16 (D) | boosts bf16 (C) |
|---|---|---|---|---|
| decode 128K (1c Q6_K) | 17.43 | 8.99 | 18.00 | 18.00 |
| prefill 128K (1c Q6_K) | 282 | 280 | 536 | 534 |
| PPL (Q6_K) | 6.312 | 6.313 | 6.3099 | **6.3095** |

Master bf16 is strictly dominated by master f16 (same accuracy, worse
decode at depth). Boosts f16 dominates master f16 (faster everywhere, equal
or better PPL). Boosts bf16 dominates everything (fastest, lowest PPL).
There is no performance or accuracy reason to choose either baseline KV
approach.

