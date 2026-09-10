# Block 0015 beta — A/B test checklist

For beta testers with a machine that can build the Block-15 tree.  Purpose: confirm the campaign's
memory wins on *your* models and hardware, and — if something looks wrong — isolate it to a single win
without rebuilding five times.  Every win except W4 is switchable by environment variable.

> **Status: template.**  The W1–W3 rows are final (those wins are validated).  The V3/V4 rows are
> provisional until those land; a session that stages Block 15 must finalise the names/defaults here and
> in `patches/README.md` **before** the beta window opens.

## 0. What Block 15 promises

With everything **on**: the generated text (same seed, `--temp 0`) is **byte-identical** to the previous
build, the buffers are smaller, and throughput does not drop.  Sizes at ctx 204800 / ub 2048 /
`-ctk q8_0 -ctv q8_0`, per GPU:

| model | before Block 15 | with Block 15 |
|---|---|---|
| Qwen3.8-Flash-Next IQ4_XS (qwen4exp) | 6690.40 MiB compute + 1262.70 host | **3251.39 + 63.69** (W1+W2) |
| Qwen3.8-27B-Q8_0 (dense) | 1920.33 + 880.34 | ~1120 + ~80 (V3), ~290 (V3+V4) |
| Qwen3.5-4B-Q8_0 (dense) | 1800.33 + 840.34 | ~1000 + ~40 (V3), ~170 (V3+V4) |

## 1. The gates

Set to `0` to disable a win (or to the value noted) and re-measure.  `LLAMA_QSA_SPARSE_FA=0` restores the
dense masked flash-attention path (the packed mask is then kept automatically).

| gate | default | disables | expected effect of disabling (ctx 204800 / ub 2048) |
|---|---|---|---|
| `GGML_QSA_SCORE_MEM` | `1` | W1 — the QSA score-chain memory cuts | qwen4exp compute **+2240 MiB** |
| `GGML_QSA_DERIVED_BIAS` | `1` | W2 — the derived per-block bias (`0` = uploaded 400 MiB tensor) | **+400 MiB** |
| `GGML_QSA_DERIVED_VIS` | `1` | W2 — the derived visibility (the packed `n_kv × n_tps` mask comes back) | **+800 MiB** compute **and +800 MiB host** |
| `LLAMA_QSA_SPARSE_FA` | `1` | the fused sparse QSA flash-attn (dense masked fallback) | slower prefill; the mask is required and kept |
| `LLAMA_QSA_KEYS_ONLY` | `1` | W3 — the keys-only QSA indexer cache (V buffer allocated again) | **+638 MiB** indexer KV |
| `LLAMA_KQ_MASK_DERIVED` | `1` | V3 — the derived kq mask for dense models | **+800 MiB** compute **and +800 MiB host** |
| *V4 gate (name TBD)* | `1`, or `0` if V4 regressed | V4 — native quantized K/V in the MMA FA path (the F16 staging scratch returns) | **+832 MiB** (q8_0 KV, exactly ctx-linear) |
| **W4** | always on | — | it is a bug fix, not a policy.  To A/B it: `git apply ab/w4-revert.patch`, rebuild |

## 2. The three measurements per model

```bash
# environment used for every run
export HIP_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib GGML_CUDA_FA_WMMA_256=0

# (1) coherence: same seed, temp 0, and compare ONLY the generated text
./build/bin/llama-cli -m MODEL -ngl 99 -sm tensor -mg 0 -p "The capital of France is" -n 24 \
    --seed 42 --temp 0 --no-display-prompt --single-turn

# (2) buffers (compute + host reserve) — llama-bench does NOT print these
./build/bin/llama-cli -m MODEL -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub 2048 -fa auto \
    -ctk q8_0 -ctv q8_0 -n 1 -v --single-turn --no-display-prompt 2>&1 |
    grep -E "compute buffer size|memory breakdown"

# (3) speed — ONE bench at a time, never in parallel/background
./build/bin/llama-bench -m MODEL -p 20480 -n 256 -r 3 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 \
    -fa on -ngl 99 -sm tensor

# (3b) if you use draft-MTP, also report the draft acceptance rate
./build/bin/llama-cli -m MODEL -md DRAFT -p "$(head -c 4000 /tmp/prompt3k.txt)" -n 96 \
    --seed 42 --temp 0 --no-display-prompt --single-turn 2>&1 | grep -i "draft acceptance"
```

## 3. Report template

Copy this into your message/issue and fill it in:

```
model/quant:            e.g. Qwen3.8-27B-Q8_0
hardware + GPUs:        e.g. 3x R9700 (gfx1201), 32 GB each, unpinned
flags:                  -ngl 99 -sm tensor -mg 0 -c 204800 -b 2048 -ub 2048 -fa auto -ctk/-ctv q8_0
context / ubatch:       204800 / 2048
block-15 commit/patch:  <sha or patch name>
gate under test:        e.g. LLAMA_KQ_MASK_DERIVED=0
coherence:              IDENTICAL / DIFFERS (attach the two texts)
compute / host:         X MiB / Y MiB      (previous build: X' / Y')
pp20480 / tg256:        A / B t/s          (previous build: A' / B')
MTP acceptance:         0.xxx              (previous build: 0.xxx)   [only with draft-MTP]
what broke / what looks off:
```

## 4. Known and accepted differences — do not report these

* the `[ Prompt: … | Generation: … ]` footer always differs (compare the text, not the process log);
* sparse vs dense flash-attention produce legitimately different floats (different kernels) — compare
  sparse-vs-sparse, dense-vs-dense;
* the adaptive-MTP acceptance probe may read e.g. `0.61616` where another build reads `0.64583`: that is
  a deterministic **buffer-layout** sensitivity of the engine, not an arithmetic difference (proven by
  running the same binary with a zero placeholder tensor).  The gate is **≥ ~0.45**;
* a *higher* reserve for a model that does not use a given win is not a regression if that win's gate
  was off.

## 5. What is not in Block 15

* **3a** (allocator through-view reuse) — measured zero reserve win on the dense models;
* **V4** if it costs prefill throughput: it then ships **opt-in** (default off) for people who need the
  last 832 MiB, and its gate row above must be marked `default 0`;
* anything under `archive/work/`.

## 6. If you find a real regression

1. Re-run with the suspected gate off.  If the regression disappears, it belongs to that win — say so
   and the session that owns it will fix or gate it.
2. If it persists with every gate off, it is probably not a campaign win: report it as a
   block-15-vs-14-block difference with the §3 template.
3. Always attach the *generated text* for coherence issues (a diff of the two outputs is enough) and the
   exact command lines — they are usually enough to reproduce it without your model.
