# `v16-d1d3c3396-r1` re-base validation (2026-09-15)

End-to-end validation of the 2026-09-15 re-base onto upstream master `d1d3c3396`, on **gfx1201**
(local, 3× R9700) and **gfx1151** (Strix Halo, `halo`).  Companion to the `WORKLOG.md` 2026-09-15
(re-base) entry; the re-base itself, its three conflict files and the MTP `nextn.hc_head_norm` crash it
exposed are described there.

## gfx1201 — adaptive-MTP four-axis gate (`-n 3000`)

The gate of [`mtp-adaptive-methodology.md`](mtp-adaptive-methodology.md) / the
[2026-09-13 n12 record](2026-09-13-adaptive-mtp-4-axis-n12.md), re-run on the re-based delivery and on a
**stock build at the same fork point** (`d1d3c3396`).  One R9700 (`HIP_VISIBLE_DEVICES=0`), Qwen3.8-27B
UD-Q4_K_XL, f16 K/V, `-fa auto -ngl 99`, `-c 32768 -b 2048 -ub 2048 --seed 42 --temp 0 --single-turn
--no-display-prompt`, `--reasoning on` for R and `off` for P/C/K, one run per cell.

| axis | plain t/s | stock `n3` t/s | stock `n3` acc | deliv `n3` t/s | deliv `n3` acc | adaptive `n12` t/s | adaptive `n12` acc | adaptive mean len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reasoning (R) | 28.91 | 47.78 | 0.59659 | 47.94 | 0.60150 | 46.95 | 0.58655 | 2.85 |
| prose (P)     | 28.49 | 56.64 | 0.79948 | 57.46 | 0.81099 | 65.52 | 0.51018 | 5.58 |
| code (C)      | 28.87 | 63.20 | 0.91079 | 63.35 | 0.90770 | 81.04 | 0.58111 | 6.91 |
| recall (K)    | 29.00 | 67.55 | 0.99200 | 68.14 | 0.99200 | 91.48 | 0.93671 | 8.79 |

* **Rule 1 (acceptance)**: fixed `n3` acceptance 0.60 / 0.81 / 0.91 / 0.99, all far above the ~0.45
  floor; no draft-vs-verify collapse.
* **Rule 2 (MTP >= plain)**: `n3` is 1.66× / 2.02× / 2.19× / 2.35× plain, adaptive `n12` 1.62× / 2.30×
  / 2.81× / 3.15×.  Delivery `n3` is within 1.4 % of stock `n3` on every axis (the delivery's win is
  the deeper/adaptive drafts, which is what block 001 is for), matching the n12 record's finding.
* **Purity at the gate length** (separate `-lv 4`-free pass): `plain == draft-mtp n3` byte-identical on
  **all four** axes — R `1a44b968d1ae`, P `29e02911baed`, C `90fae3e635a9`, K `a87c4318b649`.  Adaptive
  `n12` is byte-identical on R/K and diverges on P/C, the documented above-7 kernel-family trade.
* The n12 record's delivery columns reproduce within a few percent (R 47.94 vs 46.4, P 65.52 vs 63.4,
  C 81.04 vs 81.8); the stock `n3` throughput and acceptance columns reproduce the n12 record's stock
  numbers exactly.

## gfx1201 — delivery vs stock at the same base

`llama-bench -r 2`, 3-GPU `-sm tensor`.

| model / KV | test | new | stock | delta |
|---|---|---:|---:|---:|
| 27B UD-Q4_K_XL, f16 | pp512 | 2064 | 1904 | +8.4 % |
| 27B UD-Q4_K_XL, f16 | tg128 | 47.9 | 42.6 | +12.5 % |
| 27B UD-Q4_K_XL, f16 | pp512 @ d16384 | 1740 | 1569 | +10.9 % |
| 27B UD-Q4_K_XL, f16 | tg128 @ d16384 | 46.96 | 41.91 | +12.0 % |
| 27B UD-Q4_K_XL, q8_0 | pp512 | 2023 | 1886 | +7.3 % |
| 27B UD-Q4_K_XL, q8_0 | tg128 @ d16384 | 45.96 | 40.85 | +12.5 % |
| 35B-A3B UD-Q4_K_M | pp512 | 4929 | 4510 | +9.3 % |
| 35B-A3B UD-Q4_K_M | tg128 @ d16384 | 96.5 | 82.4 | +17.1 % |
| Flash-Next Q4_K_XL, `-sm layer` | pp512 | 1038 | 326 | +218 % |
| Flash-Next Q4_K_XL, `-sm layer` | tg128 | 36.7 | 25.5 | +44 % |

Rule-5 verify-width gate (`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, 27B q8_0 KV):
B=1 35.76 vs 34.97 t/s, B=4 140.7 vs 86.9, **B=8 195.1 vs 120.1**.  qwen4exp `-sm tensor`
(1181 pp / 50.8 tg) is delivery-only; stock rejects tensor split for the architecture.

## gfx1151 — Strix Halo pass

Built from the same patch set (strict `git am`, applied tree
`c6896785a5fefdf9438d26974c0274bf99f43263`), ROCm 7.14 gfx1151, one device, `-ngl 99`, versus a stock
build at `d1d3c3396`.

| model / KV | test | new | stock | delta |
|---|---|---:|---:|---:|
| 27B Q8_0, q8_0 | pp512 | 470.6 | 367.0 | +28.2 % |
| 27B Q8_0, q8_0 | tg128 | 7.85 | 7.78 | +0.9 % |
| 27B Q8_0, q8_0 | pp512 @ d16384 | 393.6 | 314.0 | +25.4 % |
| 27B Q8_0, q8_0 | tg128 @ d16384 | 7.65 | 7.55 | +1.3 % |
| 27B Q8_0, f16 | pp512 @ d16384 | 393.9 | 316.0 | +24.6 % |
| 27B Q8_0, f16 | tg128 @ d16384 | 7.59 | 7.53 | +0.8 % |
| 35B-A3B Q4_K_M | pp512 | 1716.7 | 945.5 | +81.6 % |
| 35B-A3B Q4_K_M | tg128 | 56.86 | 56.01 | +1.5 % |
| 35B-A3B Q4_K_M | pp512 @ d16384 | 1303.2 | 890.5 | +46.3 % |
| 35B-A3B Q4_K_M | tg128 @ d16384 | 52.54 | 51.84 | +1.4 % |

The delivery wins the prefill/long-context arms (block 04/08/13/14) and is flat-to-slightly-ahead on
decode; **no regression on any gfx1151 row**.

Correctness on gfx1151:

* `test-backend-ops`: `FLASH_ATTN_QSA`, `GATED_DELTA_NET`, `TOPK_MOE`, `HC_MIX`, `HC_COMBINE` pass;
  `FLASH_ATTN_EXT` **0 failures** (2/2 backends passed).
* Purity: 27B Q8_0 f16 KV `plain == draft-mtp` byte-identical (`2bde6e01c95f`, 1658 chars); qwen4exp
  Flash-Next Q4_K_XL `plain == draft-mtp` byte-identical (`07219ff0c119`, 1136 chars), MTP acceptance
  0.48867.
* SWA / kq-mask (block-15 V3 engages on the HIP iGPU): Gemma-4-E4B-it Q8_0 greedy output
  `5dd272b4f316` — **identical to the gfx1201 build**, a cross-architecture coherence check.
* 3-GPU `-sm tensor` gemma-4-E4B remains the documented pre-existing meta-splitter abort (2 KV heads <
  3 devices); single/2-device is fine.
