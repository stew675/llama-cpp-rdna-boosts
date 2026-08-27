# Benchmark methodology: llama-benchy protocol (v2)

This is the v2 performance-testing plan for the `rdna-boosts` project. It
replaces the hand-rolled `/completion` curl protocol of
[methodology.md](methodology.md) (v1, 2026-08-26) with
[`llama-benchy`](https://pypi.org/project/llama-benchy/) (v0.4.0), a live-server
benchmark tool that speaks the OpenAI-compatible API against a running
`llama-server`. Everything here is designed so the same commands produce a
standardized, reproducible set of results across builds, quantizations, and
GPU counts.

- v1 (curl + `/completion`): [methodology.md](methodology.md) — kept as the
  historical record.
- v2 (llama-benchy): this file.

Date: 2026-08-27
Machine: `soar`, 3x AMD Radeon AI PRO R9700 (gfx1201, 34 GiB each, ~640 MB/s
each), ROCm 7.14, 9950X3D host. Note: `rocm-smi` reports 32624 MiB usable per
card (34 GiB class).

## What changes vs v1

| | v1 (curl) | v2 (llama-benchy) |
|---|---|---|
| target | `/completion` | `/v1/chat/completions` (OpenAI compat) |
| prefill metric | server-reported `timings.prompt_ms` | client-side `est_ppt` = ttfr − api-latency, over the full prompt |
| decode metric | server-reported `timings.predicted_ms` | client-side token timestamps from the SSE stream (interpolated from `usage.completion_tokens`; see Measurement semantics) |
| depth ladder | 4K / 8K / 32K / 64K | 0 / 4K / 8K / 16K / 32K / 64K / 128K |
| prompt | fixed prebuilt prompt files | random slice of the Sherlock Holmes corpus + random UUID suffix |
| KV reuse | `cache_prompt: true` for decode phase | `cache_prompt: false` for every request (`--no-cache`) |
| run count | 2 rounds | `--runs 2` (+1 automatic warmup) |

## Builds under test

| label | tree | build | binary commit |
|-------|------|-------|---------------|
| M | upstream `master` | `~/llama.cpp/build-rocm/bin/llama-server` | `fe235f434` |
| B | `rdna-boosts` (all 10 blocks) | `~/llama.cpp/build-rdna-boosts/bin/llama-server` | `a265041b1` |

Both builds are ROCm 7.14 / Clang 23 Release, gfx1201.

## Models

| model tag | path | size | used with |
|-----------|------|------|-----------|
| `Qwen3.8-27B-Q8_0` | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | 29 GB | 2- and 3-card rows |
| `Qwen3.8-27B-Q6_K` | `/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf` | 22.9 GB | 1-card rows |
| `Qwen3.8-27B-Q4_K_XL` | `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17.6 GB | 1-card rows (new) |

The `Qwen3.8-27B-Q4_K_XL` file is a mixed-quant model: its weight tensors
carry Q4_K (69), Q5_K (191), Q6_K (56), Q8_0 (110), plus IQ4_XS/IQ4_NL/IQ3_S
slivers. **Every dominant quant type is one block 10 boosts**
(Q4_K VDR 2→4, Q5_K 2→4, Q6_K 1→2, Q8_0 2→4), so the Q4_K_XL row exercises
the full k-quant decode path on a single card, at a size (17.6 GB) that
leaves ample headroom for the 128K depth ladder under BF16 KV.

The model is `qwen35` hybrid attention (`full_attention_interval = 4`: only
every 4th layer carries full KV; the rest use sliding-window KV), which is
why a 34 GiB card fits `--ctx-size 140000` with BF16 KV despite the
per-layer KV math. Verified: 1-card server (Q6_K, BF16 KV, ctx 140000)
loads and answers `/health` OK; Q4_K_XL loads and runs on the boosted build
(`llama-bench`: pp32 429.9 t/s, tg8 24.9 t/s).

## Config matrix (8 rows)

| row | build | GPUs (`HIP_VISIBLE_DEVICES`) | model | ctx | KV |
|-----|-------|------|-------|-----|----|
| M-1Q6 | master | `0` | Q6_K | 140000 | bf16/bf16 |
| B-1Q6 | boosts | `0` | Q6_K | 140000 | bf16/bf16 |
| M-1Q4 | master | `0` | Q4_K_XL | 140000 | bf16/bf16 |
| B-1Q4 | boosts | `0` | Q4_K_XL | 140000 | bf16/bf16 |
| M-2 | master | `0,2` | Q8_0 | 262144 | bf16/bf16 |
| B-2 | boosts | `0,2` | Q8_0 | 262144 | bf16/bf16 |
| M-3 | master | `0,1,2` | Q8_0 | 262144 | bf16/bf16 |
| B-3 | boosts | `0,1,2` | Q8_0 | 262144 | bf16/bf16 |

Notes:

- KV cache type is bf16/bf16 for every row, per the project's standard
  configuration. On master (ROCm) this silently falls back to f16 storage
  plus a per-token conversion cost (v1 config B behavior); on the boosted
  build it is native BF16 (v1 config C behavior). This is intentional: it is
  the fork's headline differentiation and the comparison the benchmarks are
  designed to show. An optional master-f16 row (true upstream default, v1
  config A) can be added for parity — see "Optional rows".
- Single-card rows use the two quantizations that fit with the full depth
  ladder: Q6_K (the v1 standard) and Q4_K_XL (added 2026-08-27).
- 2/3-card rows use Q8_0 as in v1.
- GPU sets `0,2` (2 cards) and `0,1,2` (3 cards) match v1; ROCm tensor split
  (`--split-mode tensor`).

### Optional rows

- `M-1Q6-f16` / `M-1Q4-f16`: master with f16/f16 KV (upstream default), to
  isolate the silent-f16-conversion penalty from the fork comparison.
- A 1-card Q8_0 row is NOT possible at ctx 140000 (29 GB weights + KV
  exceeds 34 GiB); Q6_K is the v1 single-card standard for that reason.

## Server launch command

Identical across configs except model/alias/ctx/GPU env. Port 8033
(`--base-url http://localhost:8033/v1`). The server MUST be started with
`--alias <model-tag>` — llama-benchy sends the `--model` value in every
request and the server resolves it against loaded model names/aliases.

```
HIP_VISIBLE_DEVICES=<gpus> NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 \
GGML_CUDA_DISABLE_GRAPHS=0 NCCL_P2P_DISABLE=1 \
~/llama.cpp/build-<which>/bin/llama-server \
  --model <model-path> --alias <model-tag> --prio 2 --fit false --top-k 20 \
  --port 8033 --threads 8 --parallel 1 --top-p 0.95 --min-p 0.001 \
  --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --cpu-range 0-7 \
  --predict 98304 --threads-http 4 --load-mode mlock --cache-ram 16384 \
  --ctx-size <ctx> --flash-attn auto --temperature 0.0 \
  --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all --no-kv-unified \
  --cache-type-k bf16 --cache-type-v bf16 --ctx-checkpoints 64 \
  --cache-idle-slots --reasoning-budget 65536 --reasoning-preserve \
  --checkpoint-min-step 4096 --repeat-penalty 1.0 --presence-penalty 1.5 \
  --seed 675 --split-mode tensor
```

- `--fit false` is required: `llama_params_fit` aborts on tensor split.
- `--temperature 0.0` keeps generation deterministic (benchmarking is about
  throughput, not sampling quality; a deterministic target also makes the
  coherence check stable).
- MTP is excluded (as in v1): these are the baseline decode/prefill numbers;
  MTP is a separate speedup layered on top.
- The slot proxy (`llama-slot-proxy` on ports 8037–8039, upstream
  `:8033`) is idle during the suite; llama-benchy talks to `:8033` directly.
  It may be left running — its `/slots` polling is harmless.

## llama-benchy command

Fixed command line; only `--model` and the output path change per row:

```
uvx llama-benchy \
  --base-url http://localhost:8033/v1 \
  --tg 240 --pp 2520 \
  --model <model-tag> \
  --tokenizer Qwen/Qwen3.8-27B \
  --no-cache --runs 2 \
  --depth 0 4096 8192 16384 32768 65536 131072 \
  --save-result benchmarks/results/benchy/<row>.json --format json \
  --exit-on-first-fail
```

With `--model`:

| row set | `--model` |
|---------|-----------|
| 1-card | `Qwen3.8-27B-Q6_K` or `Qwen3.8-27B-Q4_K_XL` |
| 2/3-card | `Qwen3.8-27B-Q8_0` |

Details:

- `--pp 2520 --tg 240`: each request prefills `depth + 2520` tokens (the
  depth context is the system message, the pp prompt the user message) and
  then generates 240 tokens. So at `--depth 131072` the prefill is 133,592
  tokens — within the 140000 ctx of the 1-card rows, and far within 262144
  on 2/3 cards.
- `--depth 0 4096 ... 131072`: the depth ladder. `--depth 0` is the shallow
  anchor (2520-token prefill, decode at ~2.5K ctx).
- `--no-cache`: every request gets a random UUID suffix and
  `cache_prompt=false`, so each run is a full fresh prefill — no KV reuse,
  no prefix caching. This is what makes the prefill number a true
  0→depth measurement.
- `--runs 2`: 2 measured runs per (depth) cell, plus an automatic warmup run
  (llama-benchy always warms up unless `--no-warmup`; keep the warmup).
- `--tokenizer Qwen/Qwen3.8-27B`: HF tokenizer used to build the corpus
  (Sherlock Holmes, cached at `~/.cache/llama-benchy/`); identical to the
  server's embedded tokenizer, so local and server token counts agree.
- `--exit-on-first-fail`: fail the suite loudly instead of producing
  half-results.

The md table is rendered post-hoc from the saved JSON by
`scripts/benchy-json-to-md.py` (llama-benchy prints the md table only in
`--format md` mode; in `--format json` mode it prints a status line), so
`benchy-run.sh` saves `<row>.json` and derives `<row>.md` from it.

## Measurement semantics (what the numbers mean)

llama-benchy measures at the client. From `llama_benchy/results.py`:

- **Prefill t/s** = `prompt_tokens / est_ppt` where `est_ppt = ttfr − api
  latency` (ttfr = time to first *response* chunk). Because `cache_prompt`
  is false and the whole `depth + pp` prompt is one request, this is the
  full prefill rate averaged over context growing 0 → depth (+2520). This
  matches v1's "prefill rate averaged over the full prefill" semantics.
  Note: est_ppt includes one decode step (the first generated token is
  streamed after prefill completes), a constant ~tens of ms across configs —
  systematic, not a differentiator.
- **Decode t/s** = `(total_tokens − 1) / (last_token_ts − first_token_ts)`
  from the SSE stream. llama.cpp's OpenAI-compat stream does not emit
  per-chunk `token_ids`, so llama-benchy falls back to
  `usage.completion_tokens` (in the final chunk; the server supports
  `stream_options.include_usage`) and interpolates token timestamps evenly
  across the observed chunk span. The aggregate rate is still a true
  wall-clock decode rate — the interpolation only affects the peak/rolling
  series, not the mean.
- **`--tg 240` without `--exact-tg`**: generation runs until EOS or 240
  tokens. The corpus prompts are mid-book slices, so EOS is not hit in
  practice; the measured token counts confirm this per run.
- **Reported rows**: `pp2520 @ d<depth>` and `tg240 @ d<depth>` for each
  depth. `est_ppt`, `ttfr`, `e2e_ttft` are reported per prefill row; the
  JSON output keeps every per-run value (mean ± std) for aggregation.

## Depth-ladder cost and memory envelope

- Prefill tokens per row ≈ 3 runs × (7 × 2520 + Σ depths) ≈ 3 × 275,688 ≈
  827K tokens; at 1-card rates (~600–900 t/s prefill) that is ~15–17 min of
  prefill per single-card row, plus ~5 min of decode (7 × 240 × 3 tokens at
  ~24 t/s). 2/3-card rows are faster (up to ~1900 t/s prefill).
  Full suite ≈ 2–2.5 h.
- KV envelope at the deepest cell: 133,592 tokens × 0.254 MiB/token (BF16,
  hybrid-attention layout) ≈ 34 GiB of KV on the 1-card rows — this is why
  ctx 140000 is the 1-card ceiling and why Q6_K/Q4_K_XL are the 1-card
  quantizations. The ladder fits by construction (verified: 1-card server
  at ctx 140000 with BF16 KV comes up OK).

## Harness

`scripts/benchy-run.sh` starts the server for a row, waits for `/health`,
runs the fixed llama-benchy command, saves JSON + md, and tears the server
down. `scripts/run-benchy-suite.sh` drives all 8 rows in sequence. Both are
described in [benchmarks/README.md](README.md).

## Validation of the harness (2026-08-27)

- llama-benchy v0.4.0 installs via `uvx`; corpus fetch + tokenize works
  (144,480 corpus tokens, cached).
- The server accepts every field llama-benchy sends: `cache_prompt: false`
  (schema field), `stream_options.include_usage: true` (supported; final
  chunk carries `usage.completion_tokens`), unknown `return_token_ids`
  silently ignored (no strict validation), no API key required,
  `/v1/models` present (latency mode `api`).
- 1-card server (Q6_K, bf16/bf16, ctx 140000, tensor split) reaches
  `/health` OK; `--fit false` required on tensor split.
- Q4_K_XL loads and runs on the boosted build.
- Block 10's VDR table covers the Q4_K_XL dominant quant types
  (Q4_K/Q5_K/Q6_K/Q8_0), so the new row exercises the boosted path.
