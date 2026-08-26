# Benchmark methodology

The results this methodology produced are in [README.md](README.md). This
file is the how/what: builds, configs, launch commands, measurement
protocol, and raw per-run readings.

Date: 2026-08-26
Machine: 3x Radeon R9700 (gfx1201, 34 GB, ~640 MB/s each), ROCm 7.14
Model: `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (Q8_0, 29 GB;
imatrix-calibrated lineage), Q6_K 22.9 GB for the 1-card rows, BF16 full
weights 54 GB for the ground-truth section.

## Builds under test

| label | tree | build | binary commit |
|-------|------|-------|---------------|
| A | upstream `master` | `~/llama.cpp/build-rocm` | `d222767c7` |
| B | upstream `master` | `~/llama.cpp/build-rocm` | `d222767c7` |
| C | `rdna-boosts` (all 11 blocks) | `~/llama.cpp/build-rdna-boosts` | `a56b2ffe2` |
| V | upstream `master` (Vulkan) | `~/llama.cpp/build-vulkan` | `d222767c7` |

## Configs

| config | build | `--cache-type-k/v` | what this shows |
|--------|-------|--------------------|-----------------|
| A | master | `f16` | upstream baseline |
| B | master | `bf16` | upstream silently maps bf16 -> f16 (KV is stored f16) and pays a per-token conversion cost |
| C | rdna-boosts | `bf16` | native BF16 KV cache + native-BF16 flash-attn (block 03) |
| V | master (Vulkan) | `bf16` | true native BF16 KV on Vulkan (unlike master ROCm) |

Card counts: 1 card (`HIP_VISIBLE_DEVICES=0`, Q6_K), 2 cards (`0,2`, Q8_0),
3 cards (`0,1,2`, Q8_0), all ROCm tensor split. Vulkan:
`GGML_VK_VISIBLE_DEVICES=1,2[,3]` (devices 1-3 are the R9700s; 0 is the
iGPU), layer split (Vulkan has no tensor split).

## Server launch command (identical across configs except cache type)

```
HIP_VISIBLE_DEVICES=0,2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_DISABLE_GRAPHS=0 NCCL_P2P_DISABLE=1 \
llama-server --model /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf \
  --alias Qwen3.8-27B-Q8_0 --prio 2 --fit false --top-k 20 --port 8033 --threads 8 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --cpu-range 0-7 \
  --predict 98304 --threads-http 4 --load-mode mlock --cache-ram 16384 --ctx-size 262144 \
  --flash-attn auto --temperature 0.0 --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all \
  --no-kv-unified --cache-type-k <K> --cache-type-v <V> --ctx-checkpoints 64 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
  --repeat-penalty 1.0 --presence-penalty 1.5 --seed 675 --split-mode tensor
```

MTP is excluded from this benchmark (it is a separate speedup layered on top
of these baselines).

## Measurement protocol

Prefill and decode are measured at context depths 4K (3788-token prompt),
8K (7510), 32K (31,694), and 64K (61,684). The deep depths are the primary
measurement: at shallow depth the KV cache traffic is negligible and the
bf16-vs-f16 difference is invisible; at 32K/64K the KV reads are a material
fraction of memory traffic and the difference is fully expressed. 4K is the
shallow anchor (largely ramp-up noise); 8K is where prefill peaks.

1. Start the server with the launch command above.
2. Wait for `/health` to report `{"status":"ok"}` (this server version has
   no `model_loaded` field; it returns a 503 `{"error":{"message":"Loading
   model"}}` while loading).
3. Request 1: `/completion` with the depth prompt, `n_predict: 1`.
   `timings.prompt_n/prompt_ms` = prefill rate averaged over the full
   prefill (context grows 0 -> depth during it).
4. Request 2: `/completion` with the same prompt, `cache_prompt: true`,
   `n_predict: 256`. The prompt is served from cache, so
   `timings.predicted_n/predicted_ms` = steady-state decode at depth.
5. Two full rounds per config; numbers are stable to <1%.

Scripts: `scripts/bench-deep.sh` (server harness, 2 cards),
`scripts/bench-deep-vulkan.sh` (Vulkan variant), `scripts/bench-server.sh`
(shallow variant), `scripts/run-ppl.sh`. The 1-card and 3-card variants are
the same harness with the env/model swapped.

## Perplexity protocol

`llama-perplexity` over `wikitext-2-raw/wiki.test.raw`, flash-attn on, same
GPU env as the server. PPL is deterministic (no sampling). Two resolutions:

- 128 chunks x 2048 ctx (`--chunks 128`).
- Full corpus: the test set is only ~152 chunks at 2048 ctx, so a
  `--chunks 500` request covers the entire test set (the maximal
  resolution; the harder trailing chunks raise the absolute PPL ~0.04 for
  every config, and the error bars barely shrink because per-chunk loss
  variance dominates).

```
HIP_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 <build>/bin/llama-perplexity \
  -m <model> -f /llm/models/wikitext-2-raw/wiki.test.raw \
  -c 2048 --chunks 128 -fa on -ctk <K> -ctv <V> -sm tensor -ngl 99
```

GDN dispatch env vars used in the dispatch matrix: `GGML_CUDA_GDN_CHUNKED=0`
(sequential fallback), `GGML_CUDA_GDN_CHUNKED_BF16=0` (fp32 chunked path).

BF16 full weights (f32 KV ground truth) run with `-sm layer` on 3 cards:
mainline's `llama_params_fit` aborts on tensor split, so the ground truth
used layer split; the fork build on the same weights/tensor split was also
run and matched (6.3220 vs 6.3212, within noise).

## Raw readings (32K depth, 2 cards, per-round)

```
A-master-f16 r1: prefill 1235.6 t/s (31694 tok)  decode 28.79 t/s (256 tok)
A-master-f16 r2: prefill 1236.0 t/s (31694 tok)  decode 28.80 t/s (256 tok)
B-master-bf16 r1: prefill 1236.2 t/s (31694 tok)  decode 24.81 t/s (256 tok)
B-master-bf16 r2: prefill 1222.3 t/s (31694 tok)  decode 24.64 t/s (256 tok)
C-rdna-bf16 r1: prefill 1642.5 t/s (31694 tok)  decode 30.63 t/s (256 tok)
C-rdna-bf16 r2: prefill 1649.2 t/s (31694 tok)  decode 30.57 t/s (256 tok)
```

PPL run wall times: A 2:55, B 2:55, C 2:40 (the 15 s saving on C is the
prefill-side bf16 effect), V (Vulkan) 4:25 at 2.33 s/pass (3-card layer
split). Vulkan server runs (config VK): prefill 1467/1459, decode
17.28/17.18 (2-card layer split).
