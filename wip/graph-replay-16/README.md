# wip/graph-replay-16 — graph replay up to 16 tokens (TODO #347)

Written by Claude (AI).  Item 2 of #334, re-sent on its own as asked.  One `git am` file made on `v17-4f9dfa0e3-r324`
(tip `923b910eb`, tree `d3babb65a`).

| file | touches | kill switch |
|---|---|---|
| `0001-CUDA-HIP-graph-replay-up-to-16-tokens-GGML_CUDA_GRAP.patch` | `ggml/src/ggml-cuda/ggml-cuda.cu` (+10 −1) | `GGML_CUDA_GRAPH_MAX_TOKENS=8` |

## What it does

`ggml_cuda_graph_is_multi_token()` compares the graph's token count with `GGML_CUDA_GRAPH_MAX_TOKENS` (default 16) instead of
`MMVQ_MAX_BATCH_SIZE` (8), so 9..16-token graphs take the graph path like the 2..8 widths.  The kernels are the same.  The graph
cache already keys each width separately (`ggml_cuda_graph_key` carries `n_tokens`), so a width that repeats step after step is
captured once and replayed; a width that moves is captured once per new width.  `GGML_CUDA_GRAPH_TRACE=1` counts both.

Changed from #334 after your review:

- The "bit for bit" sentence is gone.  Widths 9..16 are outside this set's purity range (`GREEDY-PURITY.md` §11), so the change
  is judged on acceptance and throughput below.  In every arm here the text and the accepted-draft count were the same between
  the two switch settings and across repeats; that is an observation on these shapes, not a claim.
- The code comment no longer says the shape is fixed every step: it pays where a 9..16-row width repeats (several slots
  verifying together under `LLAMA_SPEC_ALIGN`, a fixed-depth draft); a moving width pays one capture per new width.

## Measurements

One R9700 (gfx1201), Linux, ROCm 10.0.0, one binary = stock r324 + this commit + the #348 re-submission (the FA KV-group bitmap,
left at its default in every arm here); each arm sets only `GGML_CUDA_GRAPH_MAX_TOKENS` (A = 8, B = unset = 16).
`ROCBLAS_USE_HIPBLASLT=0` and `GGML_CUDA_GRAPH_TRACE=1` everywhere, q8_0 K/V, `-fa on -b 2048 -ub 2048`, one fresh process per
run, a short untimed warm-up before the first run.  Models: Qwen3.8-27B UD-Q4_K_XL and Qwen3.8-4B Q8_0 (both carry the MTP
layer).

**1. The fixed-width arm** (`-np 4 --kv-unified`, `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0`, so the
aligned verify is 4 x 4 = 16 rows every step; `-c 14048`).  Four concurrent greedy `/completion` requests of 3000 tokens
(`ignore_eos`, `cache_prompt false`, `id_slot` 0..3, the prompt of `archive/work/spec-batch-budget/multislot-mtp-probe.py` with a
per-slot prefix), aggregate t/s as in that probe (sum of tokens / the longest slot's decode time).  Order A B B A A B.

| model | A (8) | B (16) | B vs A | graph calls A → B (direct / replay / capture) |
|---|---|---|---|---|
| 27B | 181.17 / 181.77 / 181.79 | 184.69 / 184.65 / 183.80 | **+1.5 %** | 1966 / 3054 / 59 → 170 / 4756 / 153 |
| 4B | 431.79 / 428.45 / 431.42 | 440.92 / 436.63 / 440.83 | **+2.1 %** | 2174 / 3223 / 61 → 182 / 5118 / 158 |

Accepted drafts 8121 / 11620 (27B) and 7807 / 12551 (4B) in all six runs of each; the four texts hash the same in all six.
The spread between repeats of one arm was within 0.5 % (27B) and 1 % (4B) here.

**2. `llama-batched-bench`** (`-c 8192 -b 2048 -ub 512 -npp 128 -ntg 128 -npl 1,4,8,16`), S_TG t/s, order A B B A:

| model | npl | A (8) | B (16) | B vs A |
|---|---|---|---|---|
| 27B | 1 / 4 / 8 | 31.63, 31.65 / 104.55, 104.90 / 160.81, 161.73 | 31.48, 31.62 / 104.88, 104.95 / 161.33, 161.31 | within ±0.5 % |
| 27B | 16 | 249.89, 251.67 | 256.56, 257.50 | **+2.5 %** |
| 4B | 1 / 4 / 8 | 98.71, 98.70 / 353.51, 351.92 / 549.62, 550.51 | 98.73, 98.77 / 352.64, 353.21 / 550.40, 551.31 | within ±0.3 % |
| 4B | 16 | 787.95, 786.89 | 819.46, 819.91 | **+4.1 %** |

Graph calls per run: 140 / 378 / 3 (A) → 13 / 504 / 4 (B).  Widths 1..8 are untouched, as expected.

**3. A moving width** (your `--spec-draft-n-max 15` arm, one slot, `-np 1 -c 8192`, `--spec-type draft-mtp --spec-draft-n-max 15
--spec-draft-p-min 0`, one 1000-token greedy request, 27B), order A B B A: A 44.82 / 44.89, B 45.48 / 45.50 t/s (**+1.4 %**);
graph calls 543 / 3649 / 30 → 69 / 4113 / 40; accepted drafts 749 / 3681 and the same text in all four.  On this shape the extra
captures (30 → 40) cost less than the replays they buy; your three-GPU run counted 138 → 924 captures, which this one-GPU,
one-slot run does not reproduce.

**The small model**: the 4B was not slower in any arm (+2.1 % fixed width, +4.1 % at `-npl 16`, inert at 1..8), so no size gate
is proposed; WORKLOG 260928.003's ~2.5 % (r20, ROCm 7.14, widths 2..8) did not show up here for 9..16.

## Not tested

Windows; RDNA3 / 3.5; NVIDIA; several GPUs (`-sm tensor`, the shape of your 138 → 924 captures); the n-gram drafters; a
`-n 3000` arm at more than 4 slots.

## ENVIRONMENT.md row, if it lands

| `GGML_CUDA_GRAPH_MAX_TOKENS` | **16** | kill-switch / tuning | the most tokens a graph may have and still take the graph path (was `MMVQ_MAX_BATCH_SIZE` = 8).  9..16-token verifies are replayed like 2..8; a repeating width (4 slots x `draft-mtp n3` under `LLAMA_SPEC_ALIGN`) +1.5 % (27B) / +2.1 % (4B), `llama-batched-bench -npl 16` +2.5 % / +4.1 %, widths 1..8 untouched.  `8` restores the previous limit. |
