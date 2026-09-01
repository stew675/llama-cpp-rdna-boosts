# 2026-08-31 — MTP chunked GDN prefix (block 13)

Credit: block 02 chunked WMMA GDN is stew675. This patch only runs it on MTP prefills.

**Lab:** 2× R9700 (gfx1201), ROCm 7.14, tensor-split 1,1, `GGML_CUDA_ALLREDUCE=internal`, Qwen 3.8 27B Q8, ubatch 1024, MTP n-max 3 (`K=4`), FA on. Path fire: `MTP chunked GDN prefix n=1024 K=4 prefix=1020`.

## Prefill tok/s (median of 3, `n_predict=16`, VOID=false)

| prompt tokens | before (K==1-only chunked) | after (prefix+tail) |
|--|--|--|
| ~4141 | ~1441 | **1526–1527** (+5.9%) |
| 40960 | 1160 | **1254** |
| 81920 | 984 | **1043** |
| 102400 | 911 | **960** (spread 0.03%, no cliff) |

4k decode-32 ~78 both (not the win). 100k decode-16 ~46.5–46.8 both.

## vs sequential (`GGML_CUDA_GDN_CHUNKED=0`, same binary)

Token identity (greedy, temp 0): **not identical** — first diverge at token 6 (4k) / 11 (8k). Expected: bf16 chunked vs fp32 sequential.

| task | prefix KEEP | sequential |
|--|--|--|
| GSM8K 50, max_tokens 256 | 19/50 | 18/50 |
| HumanEval 20 | 20/20 | 20/20 |

GSM8K +1 is noise. HE-20 is a ceiling.

Opt out: `GGML_CUDA_GDN_CHUNKED=0`.
