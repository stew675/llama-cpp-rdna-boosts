# 03 — gfx1201 validation record (prototype port)

Date: 2026-09-13.  Box: `soar`, 1× AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GiB), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`), wave32.  Model: `Qwen3.8-27B-Q6_K` (qwen35, 27.32 B, 48 value heads /
16 key heads / S_v=128 / conv 4 / 4-layer full-attn interval), 21.30 GiB.

Scratch fork: `~/llama.cpp` on `rdna-boosts` with the **prototype port applied uncommitted**
(`reference/port-spike-gated_delta_net.patch`, +234/−1 in `ggml/src/ggml-cuda/gated_delta_net.cu`).
The port adds pwilkin's DPP helpers, the `gated_delta_net_tiled_cuda` template, and an env-gated
dispatch (`GGML_CUDA_GDN_TILED`: `1` = 16×4, `2` = 8×8; `0`/unset = off) that also makes the
chunked block skip so the tiled path can be A/B'd directly.  **Revert** with
`git -C ~/llama.cpp checkout -- ggml/src/ggml-cuda/gated_delta_net.cu`.

Build: `cmake --build build-rocm --target llama-bench llama-cli test-backend-ops llama-perplexity -j 16`
(recompiles only `gated_delta_net.cu` + relinks).

> **Measurement hygiene note.**  Two early perf runs were launched in the same tool block and
> therefore ran concurrently, contaminating those numbers (sequential and chunked-fp32).  All
> numbers below are from runs executed **strictly one process at a time** in a single command
> chain.  The contaminated figures are not quoted anywhere in this tree.

## 1. Op-level GDN performance — `test-backend-ops perf -o GATED_DELTA_NET`

Shape `head_count=16, head_size=128, v_repeat=3` (H_v=48), `n_seqs=1`, `K=1` (µs/run):

| n_tokens | chunked bf16 (default) | sequential (`CHUNKED=0`) | chunked fp32 (`BF16=0`) | tiled 16×4 (`TILED=1`) | tiled 8×8 (`TILED=2`) |
|---:|---:|---:|---:|---:|---:|
| 64   | **28.49** | 129.04 | 99.98 | 77.30 | 73.37 |
| 256  | **64.59** | 489.63 | 300.92 | 280.06 | 267.75 |
| 512  | **107.32** | 977.73 | 593.00 | 551.15 | 525.79 |
| 1024 | **230.36** | 1970.68 | 1184.18 | 1130.31 | 1077.84 |

Speed-up vs the sequential kernel:

| n_tokens | chunked bf16 | chunked fp32 | tiled 16×4 | tiled 8×8 |
|---:|---:|---:|---:|---:|
| 64   | 4.53× | 1.29× | 1.67× | 1.76× |
| 256  | 7.58× | 1.63× | 1.75× | 1.83× |
| 512  | 9.11× | 1.65× | 1.77× | 1.86× |
| 1024 | 8.55× | 1.66× | 1.74× | 1.83× |

Reading: the tiled kernel is a real ~1.8× scan improvement, and it beats the **fp32** chunked
path — but it is **~5× slower than the bf16 chunked default**, and the gap grows with `n_tokens`
(the scan's fixed parallelism versus the chunked kernel's chunk parallelism).

## 2. Correctness — tight oracle gate

`test-backend-ops test -o GATED_DELTA_NET`, with the chunked env off so the test uses its **tight
`1e-7` NMSE** gate rather than the relaxed bf16 gate:

```
GGML_CUDA_GDN_TILED=1 GGML_CUDA_GDN_CHUNKED=0 GGML_CUDA_GDN_CHUNKED_BF16=0 \
  ./build-rocm/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0
  -> 46/46 tests passed   (and the same for TILED=2)
```

## 3. End-to-end prefill — `llama-bench` (1 GPU)

`-ngl 99 -fa on -ctk f16 -ctv f16 -r 3 -o md`:

| config | pp2048 (ub 2048) | pp8192 (ub 8192) |
|---|---:|---:|
| sequential (`CHUNKED=0`, exact reference) | 944.85 ± 1.05 | 882.41 ± 0.16 |
| **tiled 16×4** (`TILED=1`) | 988.67 ± 0.65 | — |
| **tiled 8×8** (`TILED=2`) | 989.06 ± 0.80 | 923.24 ± 2.84 |
| default **chunked bf16** | **1028.62 ± 1.15** | **953.51 ± 0.32** |

- chunked bf16 vs sequential: **+8.9 %** (pp2048), **+8.1 %** (pp8192).
- tiled vs sequential: **+4.6 %** (pp2048), **+4.6 %** (pp8192).
- tiled vs chunked bf16: **−4.0 %** (pp2048), **−3.3 %** (pp8192).

This also bounds the whole GDN prefill lever on this box/model at ~9 %: even a zero-cost GDN
could not buy more.

## 4. Quality — wikitext-2 PPL

`llama-perplexity -m Qwen3.8-27B-Q6_K -f wiki.test.raw -c 512 -b 512 -ub 512 -ngl 99 -fa on
-ctk f16 -ctv f16 --chunks 64`:

| path | per-chunk PPL (first few) | Final estimate |
|---|---|---|
| sequential (exact) | `4.3053, 6.0685, 5.8186, 5.7792, …` | **6.5078 ± 0.12353** |
| tiled 16×4 | `4.3053, 6.0685, 5.8186, 5.7792, …` (identical vector) | **6.5078 ± 0.12353** |
| tiled 8×8 | `4.3053, 6.0685, 5.8186, 5.7792, …` (identical vector) | **6.5078 ± 0.12353** |
| chunked bf16 (default) | `4.3101, 6.0653, 5.8182, 5.7799, …` (differs) | 6.5092 ± 0.12357 (**+0.0215 %**) |

The tiled kernel reproduces the sequential kernel's PPL **exactly, chunk for chunk** at printed
precision — the empirical confirmation that it is bit-neutral on gfx1201 (DPP pairing + explicit
FMA spellings).  The deviation lives entirely in the chunked bf16 path, and is the expected
near-lossless magnitude.

> Caveat: PPL to four decimals is very strong but not a bitwise proof.  A direct logits-hash
> comparison (`W = 1..8` and long-prefill state hashes) is the delivery-grade confirmation and
> should be run before any promotion.

## 5. What was *not* tested (scope gaps)

- `S_v` other than 128 (the kernel's `static_assert(S_v % block_cols == 0)` excludes 16/32 with
  block_cols=64; needs per-`S_v` configs).
- KDA (the port gates on `!KDA`; KDA stays on the sequential kernel).
- `n_seqs > 1` in the real model path.
- RDNA3 / gfx1100 / gfx1151 (no hardware here; the DPP guard compiles for RDNA3 too).
- Multi-GPU tensor split (GDN is per-layer and replicated; not expected to differ, but untested).
- The MTP `keep_rs` snapshot path (`K > 1`): the perf/PPL runs are `K=1`; the kernel implements
  `keep_rs_t`, but the delivery's snapshot sweep was not run against it.

## 6. Reproduction commands

```bash
cd ~/llama.cpp
git apply ~/llama-cpp-rdna-boosts/archive/work/tiled-gdn/reference/port-spike-gated_delta_net.patch
LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0 \
  cmake --build build-rocm --target llama-bench test-backend-ops llama-perplexity -j 16

# op perf (one config at a time!)
GGML_CUDA_GDN_TILED=2 ./build-rocm/bin/test-backend-ops perf -o GATED_DELTA_NET -b ROCm0
# correctness (tight gate)
GGML_CUDA_GDN_TILED=2 GGML_CUDA_GDN_CHUNKED=0 GGML_CUDA_GDN_CHUNKED_BF16=0 \
  ./build-rocm/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0
# end-to-end
GGML_CUDA_GDN_TILED=2 ./build-rocm/bin/llama-bench -m Qwen3.8-27B-Q6_K.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -p 2048 -n 0 -b 2048 -ub 2048 -r 3
# PPL
GGML_CUDA_GDN_TILED=2 ./build-rocm/bin/llama-perplexity -m Qwen3.8-27B-Q6_K.gguf \
  -f /llm/models/wikitext-2-raw/wiki.test.raw -c 512 -b 512 -ub 512 -ngl 99 \
  -fa on -ctk f16 -ctv f16 --chunks 64
# revert the prototype
git checkout -- ggml/src/ggml-cuda/gated_delta_net.cu
```
