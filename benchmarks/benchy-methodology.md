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
| V | upstream `master` (Vulkan) | `~/llama.cpp/build-vulkan/bin/llama-server` | `fe235f434` |

All three are built from the same source lineage: V is the same master tip
as M but on the Vulkan backend (RADV), giving a full cross-backend gamut.

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

## Config matrix (24 rows: 2 ROCm builds × 2 KV × 4 model/card sets + 1 Vulkan build × 2 KV × 4 sets)

ROCm rows (as above, 16):

| row | build | GPUs (`HIP_VISIBLE_DEVICES`) | model | ctx | KV |
|-----|-------|------|-------|-----|----|
| M-1Q6-f16 | master | `0` | Q6_K | 140000 | f16/f16 |
| M-1Q6-bf16 | master | `0` | Q6_K | 140000 | bf16/bf16 |
| B-1Q6-f16 | boosts | `0` | Q6_K | 140000 | f16/f16 |
| B-1Q6-bf16 | boosts | `0` | Q6_K | 140000 | bf16/bf16 |
| M-1Q4-f16 | master | `0` | Q4_K_XL | 140000 | f16/f16 |
| M-1Q4-bf16 | master | `0` | Q4_K_XL | 140000 | bf16/bf16 |
| B-1Q4-f16 | boosts | `0` | Q4_K_XL | 140000 | f16/f16 |
| B-1Q4-bf16 | boosts | `0` | Q4_K_XL | 140000 | bf16/bf16 |
| M-2-f16 | master | `0,2` | Q8_0 | 262144 | f16/f16 |
| M-2-bf16 | master | `0,2` | Q8_0 | 262144 | bf16/bf16 |
| B-2-f16 | boosts | `0,2` | Q8_0 | 262144 | f16/f16 |
| B-2-bf16 | boosts | `0,2` | Q8_0 | 262144 | bf16/bf16 |
| M-3-f16 | master | `0,1,2` | Q8_0 | 262144 | f16/f16 |
| M-3-bf16 | master | `0,1,2` | Q8_0 | 262144 | bf16/bf16 |
| B-3-f16 | boosts | `0,1,2` | Q8_0 | 262144 | f16/f16 |
| B-3-bf16 | boosts | `0,1,2` | Q8_0 | 262144 | bf16/bf16 |

Vulkan rows (8, same master tip `fe235f434`, layer split):

| row | build | GPUs (`GGML_VK_VISIBLE_DEVICES`) | model | ctx | KV |
|-----|-------|------|-------|-----|----|
| V-1Q6-f16 | master (Vulkan) | `1` | Q6_K | 140000 | f16/f16 |
| V-1Q6-bf16 | master (Vulkan) | `1` | Q6_K | 140000 | bf16/bf16 |
| V-1Q4-f16 | master (Vulkan) | `1` | Q4_K_XL | 140000 | f16/f16 |
| V-1Q4-bf16 | master (Vulkan) | `1` | Q4_K_XL | 140000 | bf16/bf16 |
| V-2-f16 | master (Vulkan) | `1,2` | Q8_0 | 262144 | f16/f16 |
| V-2-bf16 | master (Vulkan) | `1,2` | Q8_0 | 262144 | bf16/bf16 |
| V-3-f16 | master (Vulkan) | `1,2,3` | Q8_0 | 262144 | f16/f16 |
| V-3-bf16 | master (Vulkan) | `1,2,3` | Q8_0 | 262144 | bf16/bf16 |

The Vulkan leg completes the gamut: the same master source on a different
backend end-to-end (RADV gfx1201). Vulkan device indices differ from HIP:
Vulkan 0 = the iGPU, **1/2/3 = the three R9700s** (verified 2026-08-27 via
`llama-bench --list-devices` and `vulkaninfo`). Vulkan uses layer split
(`--split-mode layer`; it has no tensor split) and genuinely supports
native BF16 KV (`bf16: 1`, KHR_coopmat) — unlike ROCm master, which
silently stores f16. So V-bf16 is the true native-BF16 reference from the
upstream tree, and V-f16 is the Vulkan default.

Note on the Vulkan GPU sets: 1-card = `1` (first R9700), 2-card = `1,2`,
3-card = `1,2,3` (matching v1).

The 2×2 build × KV-type design is deliberate; each corner answers one
question:

| | f16 KV | bf16 KV |
|---|---|---|
| **master** | the true upstream default (v1 config A) — the baseline | upstream BF16 (v1 config B): ROCm silently stores f16 and pays a per-token conversion cost — the degradation being measured |
| **boosts** | the fork on the conservative path: same KV type as the baseline, showing the other blocks' contribution in isolation | the fork's headline config (v1 config C): native BF16 KV + FP32-accumulate attention |

The story the suite tells: master-bf16 is strictly worse than master-f16
(decode penalty growing with depth); boosts-bf16 is faster than master-f16
AND more accurate (PPL) — no reason to pick either baseline corner. The
boosts-f16 column exists so that if someone insists on f16 KV for their own
reasons, they can see the fork still beats master at the same KV type
(and quantify what giving up BF16 costs on the fork).

Notes:

- KV cache types: f16/f16 is the conservative baseline; bf16/bf16 is the
  project's standard. On master (ROCm) bf16 silently falls back to f16
  storage plus a per-token conversion cost; on the boosted build it is
  native BF16. This is exactly the difference the suite measures.
- Single-card rows use the two quantizations that fit with the full depth
  ladder: Q6_K (the v1 standard) and Q4_K_XL (added 2026-08-27).
- 2/3-card rows use Q8_0 as in v1.
- GPU sets `0,2` (2 cards) and `0,1,2` (3 cards) match v1; ROCm tensor split
  (`--split-mode tensor`). NOTE: HIP enumerates the R9700s in a rotated
  order vs `rocm-smi` (HIP 0 = physical GPU[1] = bus 06, HIP 1 = GPU[2] =
  bus 09, HIP 2 = GPU[0] = bus 03, verified 2026-08-27). All three cards
  are identical, so results are unaffected — but "GPU 0" in the logs is
  the physical card at bus 03 only when HIP_VISIBLE_DEVICES maps it so.
- A 1-card Q8_0 row is NOT possible at ctx 140000 (29 GB weights + KV
  exceeds 34 GiB); Q6_K is the v1 single-card standard for that reason.

## Server launch command

Identical across configs except model/alias/ctx/GPU env/KV types. Port 8033
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
  --cache-type-k <K> --cache-type-v <V> --ctx-checkpoints 64 \
  --cache-idle-slots --reasoning-budget 65536 --reasoning-preserve \
  --checkpoint-min-step 4096 --repeat-penalty 1.0 --presence-penalty 1.5 \
  --seed 675 --split-mode tensor
```

`<K>/<V>` = `f16/f16` or `bf16/bf16` per the matrix.

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

With `--model` (same for both KV types of a model set):

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
  Full suite (16 rows) ≈ 4–5 h.
- KV envelope at the deepest cell: this is a hybrid-attention model
  (`full_attention_interval = 4`), so only 16 of the 65 layers carry a full
  KV cache; the rest use sliding-window KV. Measured at ctx 140000 on a
  single card: total VRAM 32.2 GB with Q6_K (21.3 GiB) + BF16 KV + compute
  buffers, i.e. ≈ 8.6 GiB of KV for 140,032 cells ≈ 0.064 MiB/token (vs
  0.254 for a full-attention layout). At the deepest cell (133,592 tokens)
  the KV is ≈ 8.3 GiB — comfortably inside the 34 GiB card for both Q6_K
  (21.3 GiB) and Q4_K_XL (16.3 GiB). Verified: 1-card servers at ctx
  140000 come up OK on both builds with bf16/bf16 AND f16/f16 KV.
- Why ctx 140000 then? It is the round number above the ladder's deepest
  cell (131072 + 2520 = 133,592) that the user verified end-to-end on a
  single card; it also leaves room for the 240 generated tokens.

## Perplexity (accuracy) protocol

Throughput is only half the story; the 2×2 matrix exists to show the fork
is faster AND at least as accurate. PPL is deterministic (no sampling), so
it is run once per model via `llama-perplexity` (same protocol as v1,
128 chunks × 2048 ctx over `wikitext-2-raw/wiki.test.raw`, flash-attn on):

```
HIP_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 <bin>/llama-perplexity \
  -m <model> -f /llm/models/wikitext-2-raw/wiki.test.raw \
  -c 2048 --chunks 128 -fa on -ctk <K> -ctv <V> -sm tensor -ngl 99
```

with the same 4 corners (scripts/run-ppl.sh):

| corner | build | KV | v1 mapping |
|--------|-------|----|------------|
| A | master | f16 | config A (upstream default) |
| B | master | bf16 | config B (silently f16 on ROCm + conversion cost) |
| C | boosts | bf16 | config C (native BF16, FP32-accumulate attention) |
| D | boosts | f16 | (new) |
| V | master (Vulkan) | f16 | (new) |
| V | master (Vulkan) | bf16 | config V (true native BF16 on Vulkan) |

Run per model: Q8_0 (2/3-card), Q6_K and Q4_K_XL (1-card). The Vulkan
corners use `GGML_VK_VISIBLE_DEVICES=1,2` and `-sm layer` (Vulkan has no
tensor split). Expected v1 baseline: A ≈ B ≈ C within ±0.04 (all wash),
directionally C lowest; Vulkan landed highest in v1 (6.3293 Q8_0 BF16,
attributed to BF16-everywhere attention numerics). The full matrix will
show whether D differs from C on the fork, whether V-f16 vs V-bf16 differ,
and whether the Q4_K_XL quants carry a PPL cost vs Q6_K/Q8_0 at the same
corners. The message the suite is designed to support: the fastest config
is also at-or-below the others in PPL, so neither baseline corner is ever
justified.

## Harness

`scripts/benchy-run.sh` starts the server for a row (build, model, alias,
GPUs, ctx, KV types, backend rocm|vulkan), waits for `/health`, runs the
fixed llama-benchy command, saves JSON + md, and tears the server down.
`scripts/run-benchy-suite.sh` drives the 16 ROCm rows;
`scripts/run-benchy-vulkan-suite.sh` drives the 8 Vulkan rows. Both are
described in [benchmarks/README.md](README.md).

## Validation of the harness (2026-08-27)

- llama-benchy v0.4.0 installs via `uvx`; corpus fetch + tokenize works
  (144,480 corpus tokens, cached).
- The server accepts every field llama-benchy sends: `cache_prompt: false`
  (schema field), `stream_options.include_usage: true` (supported; final
  chunk carries `usage.completion_tokens`), unknown `return_token_ids`
  silently ignored (no strict validation), no API key required,
  `/v1/models` present (latency mode `api`).
- 1-card servers at ctx 140000 (Q6_K, tensor split) reach `/health` OK on
  both builds with bf16/bf16 AND f16/f16 KV; `--fit false` required on
  tensor split.
- Q4_K_XL loads and runs on the boosted build.
- Block 10's VDR table covers the Q4_K_XL dominant quant types
  (Q4_K/Q5_K/Q6_K/Q8_0), so the new row exercises the boosted path.
- HIP↔rocm-smi device mapping verified (rotation: HIP 0 = physical GPU[1]).
