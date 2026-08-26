# A/B/C benchmark: master vs rdna-boosts on Qwen3.8-27B-Q8_0

Date: 2026-08-26
Machine: 3x Radeon R9700 (gfx1201, 34 GB), ROCm 7.14
GPUs used: 2 (device 0 and 2, `HIP_VISIBLE_DEVICES=0,2`, tensor split)

## Builds under test

| label | tree | build | binary commit |
|-------|------|-------|---------------|
| A | upstream `master` | `~/llama.cpp/build-rocm` | `d222767c7` |
| B | upstream `master` | `~/llama.cpp/build-rocm` | `d222767c7` |
| C | `rdna-boosts` (all 11 blocks) | `~/llama.cpp/build-rdna-boosts` | `a56b2ffe2` |

Same model: `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (Q8_0, 29 GB).
Same GPU allocation for every run: devices 0+2, tensor split, 262144 ctx.

## Configs

| config | build | `--cache-type-k/v` | what this shows |
|--------|-------|--------------------|-----------------|
| A | master | `f16` | upstream baseline |
| B | master | `bf16` | upstream silently maps bf16 -> f16 (KV is stored f16) and pays a per-token conversion cost |
| C | rdna-boosts | `bf16` | native BF16 KV cache + native-BF16 flash-attn (block 03) |

## Server launch command (identical for A, B, C except cache type)

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

MTP is excluded from this benchmark (it is a separate speedup layered on top of
these baselines).

## Measurement protocol

Prefill and decode are measured at **~32K context depth** (31,694-token
prompt), because at shallow depth the KV cache traffic is negligible and the
bf16-vs-f16 difference is invisible; at 32K depth the KV reads are a material
fraction of memory traffic and the difference is fully expressed.

1. Start the server with the launch command above.
2. Wait for `/health` to report `{"status":"ok"}` (server ready in ~7 s).
3. Request 1: `/completion` with a 31,694-token prompt, `n_predict: 1`.
   `timings.prompt_n/prompt_ms` = prefill rate averaged over the full 32K
   prefill (context grows 0 -> 32K during it).
4. Request 2: `/completion` with the same prompt, `cache_prompt: true`,
   `n_predict: 256`. The prompt is served from cache, so
   `timings.predicted_n/predicted_ms` = steady-state decode at 32K depth.
5. Two full rounds per config; numbers are stable to <1%.

Scripts: `scripts/bench-deep.sh` (server harness), `scripts/bench-server.sh`
(shallow variant), PPL commands in `scripts/run-ppl.sh`.

## Perplexity protocol

`llama-perplexity` over `wikitext-2-raw/wiki.test.raw`, `-c 2048 --chunks 128`,
flash-attn on, tensor split, same GPU env as the server:

```
HIP_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 <build>/bin/llama-perplexity \
  -m <model> -f /llm/models/wikitext-2-raw/wiki.test.raw \
  -c 2048 --chunks 128 -fa on -ctk <K> -ctv <V> -sm tensor -ngl 99
```

PPL is deterministic (no sampling); each config was run once (B's estimate is
byte-identical to A's within noise, which is itself the finding).
