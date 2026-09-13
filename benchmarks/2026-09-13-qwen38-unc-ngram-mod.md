# 2026-09-13 — Qwen3.8-27B Unc Q8 HIP: ngram-mod 16/2/96 vs MTP n-max 3

Credit: stew675 rdna-boosts **12-set** (`a7cc83bba` + blocks 01–12, HIP). This is a **server recipe**, not a new kernel. Stock llama.cpp `--spec-type ngram-mod`.

**Lab:** 2× Radeon AI PRO R9700 (gfx1201), ROCm 7.14 / HIP 7.14, tensor-split 1,1, `GGML_CUDA_ALLREDUCE=internal`, `orcarouter_Qwen3.8-27B-Uncensored-Q8_0.gguf`, KV q8_0, FA on, ubatch 1024, `-c 131072` P=1 (same matrix as the HIP vs Vulkan night). Daily restore is P=2 × 262144/slot.

**Why this exists:** farm ngram **24/48/64** on this Unc never drafted (`draft_n` null, ~31 tok/s = nospec). MTP n-max 3 won decode and became daily. MTP prefill is **−23–28%** vs nospec. Sweeping **n-min down to 2** (match 16 / max 96) fills the hash pool on Unc.

Not qwen4exp `wip/managed-ngrams` (PLE table LRU).

## Decode tg128 median (best) · prefill from same block

| arm | unique d0 | unique 16k | unique 32k | code | pp 32k |
|--|--:|--|--|--:|--:|
| nospec | 32 | 30 | 29 | — | **1785** |
| MTP n-max 3 | **83** | **61 / 64** | **62 / 63** | — | **1283** |
| ngram-mod 16/32/96 | 168* | 50 / 70 | 55 / 78 | 342 | **1787** |
| **ngram-mod 16/2/96** | 157* | 62 / **92** | 55 / **78** | **343** | **1719–1775** |
| ngram-mod 24/2/2 | 35 | 31 | 31 | 36 | 1787 |

\*tiny “Say OK” + 113/113 drafts — lottery, not essay speed. Unique-prose **median** still MTP; **best** and **code** are ngram. Prefill is the ngram win.

## Flags (ngram daily)

```
--spec-type ngram-mod --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 96
```

Do not mix ngram + MTP. n-min **48** on this dense Unc ≈ no drafts.
