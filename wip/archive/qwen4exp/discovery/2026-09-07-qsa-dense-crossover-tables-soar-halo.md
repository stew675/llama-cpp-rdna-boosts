# 2026-09-07 — QSA vs Dense crossover tables: Soar (3xR9700 gfx1201) + Strix Halo (gfx1151)

Method: llama-bench pp2048 + tg64 at fixed depth D (context pre-filled to D first), bf16
KV on both boxes, interleaved single runs per (config, depth) on a quiet box.  Configs:
qsa = GGML_CUDA_QSA_INDEXER_SCORE=1 (+ GGML_CUDA_QSA_INDEXER_CACHE=1 for the decode path;
the derived cache is decode-only, so prefill is identical with/without it), dense =
LLAMA_QSA_OFF=1.  Model: Qwen3.8-Flash-Next-UD-IQ4_XS.  The per-op sparse prefill path is
measured as-is (prefill has no fused score; n_tokens>1 gating).  Box drift: Soar tg numbers
vary +/-1 t/s across the session; the ratios are the signal.

## Soar (3x R9700 AI PRO, gfx1201, tensor split, bf16 KV)

| depth | pp2048 qsa | pp2048 dense | pp delta | tg64 qsa | tg64 dense | tg delta |
|---|---|---:|---:|---:|---:|---:|---:|
| 8K    | 2223 | 2140 | **+4%**  | 43.3 | 46.4 | -6.7% |
| 16K   | 2154 | 1757 | **+23%** | 42.9 | 45.8 | -6.5% |
| 32K   | 1982 | 1268 | **+56%** | 41.5 | 44.8 | -7.3% |
| 64K   | 1709 |  842 | **+103%**| 39.5 | 42.7 | -7.5% |
| 96K   | 1496 |  637 | **+135%**| 37.8 | 41.0 | -7.8% |
| 131K  | 1341 |  515 | **+161%**| 36.2*| 39.4 | -8.0% |
| 160K  | 1207 |  429 | **+181%**| 34.7 | 37.8 | -8.1% |

*131K tg qsa = 36.21 with cache=1; fused WITHOUT the derived cache was 32.53 (-17.5%) -
the derived cache recovers +11% at 131K (the O(depth) score rescan it eliminates is 4x the
32K cost there; it measured flat at <=64K because the rescans were still overlapped).

## Strix Halo (Ryzen AI MAX+ 395, gfx1151, 1 GPU, bf16 KV)

| depth | pp2048 qsa | pp2048 dense | pp delta | tg64 qsa | tg64 dense | tg delta |
|---|---|---:|---:|---:|---:|---:|---:|
| 16K  | 603 | 527 | **+14%**  | 24.11 | 24.48 | **-1.5%** |
| 24K  | 607 | 478 | **+27%**  | 23.81 | 23.85 | **-0.2%** |
| 32K  | 593 | 424 | **+40%**  | 23.56 | 23.27 | **+1.2%** |
| 49K  | 564 | 336 | **+68%**  | 23.07 | 22.18 | **+4.0%** |
| 64K  | 540 | 266 | **+103%** | 22.59 | 21.20 | **+6.6%** |
| 96K  | 493 | 183 | **+169%** | 21.66 | 19.46 | **+11.3%** |

## Conclusions

- **Soar: QSA for prefill ALWAYS (wins from ~8K, monotonically to +181% @160K); dense for
  decode ALWAYS (qsa trails a flat ~7-8% at every depth 8K-160K).**  Simple, deterministic.
  The old "dense wins prefill at 30K" record is obsolete (predates the QSA prefill
  improvements; also a non-comparable whole-prompt llama-cli banner).
- **Halo: QSA for prefill always (already +14% @16K, grows to +169%); QSA for decode above
  ~26K context** (CORRECTED: the decode crossover is between 24K, where dense leads by a
  hair -0.2%, and 32K where qsa leads +1.2% - i.e. ~24-28K, NOT ~40K as first stated; at
  32K qsa is already ahead, so the cross is below it).  The qsa decode lead then grows
  +4.0% @49K, +6.6% @64K, +11.3% @96K.  Halo is the outlier because 1 GPU has no
  mirror/dispatch/AR penalty - the sparse FA's read savings surface on the wall.
  NOTE: this bf16/current-build halo table flips the old f16-era regime data (dense led
  +7.8% @12K / +11.9% @32K per-op era; +2.3% @32K fused-f16): both the KV type and the
  newer build (fused score + derived cache + round2 topk) moved the halo picture.
- Why Soar decode never crosses: the 3-GPU decode is MoE-mmv-bound + ~30% per-kernel
  dispatch floor (~3.16us x ~1900 kernels/token - GPU-side, graphs can't remove it), so the
  attention share is too small for QSA's FA savings to overcome the indexer overhead.  The
  dense decode drops only 47->37.8 t/s from 32K->160K while fused+cache1 follows at ~8%.
- Deep-context enablers found along the way: (1) the derived cache ([3], parked as a wash at
  <=64K) is vindicated at depth - it recovers +11% of the fused decode at 131K; (2) the
  llama-bench deep-context config works to 160K+ on Soar (32 GB GPUs, ctx-size up to 180K).

## Raw logs
Soar: /tmp/sweep-{qsa,den}-{8192,16384,32768,49152,65536,98304,131072,163840}.log (131K in
/tmp/deep131-*.log), /tmp/pp2048-131-*.log.  Halo: /tmp/hs-*.log, /tmp/hs2-*.log on the
halo box (~/llama-delivery/build-fused at 4ff65247c, bf16 KV).
