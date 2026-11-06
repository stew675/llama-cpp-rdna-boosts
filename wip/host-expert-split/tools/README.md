# Measurement harnesses for `wip/host-expert-split/`

One script per arm shape, so W1/W3 in [`../PLAN.md`](../PLAN.md) are reproduced rather than re-derived.
Every arm is meant to be run **under `scripts/gpu-watch.sh`**, which supplies the `timeout` and the flight
record:

```bash
SAMPLE_NOTE=<tag> ../../../scripts/gpu-watch.sh --out /tmp/<tag>.csv --interval 3 --rocm-every 0 \
    --timeout 1200 --cmd 'bash arm-decay.sh decode 36 lme36-decode'
```

| script | shape | used for |
|---|---|---|
| `arm-decay.sh RANK LME TAG` | `-n 64`, `-ub 4096`, the trace config | P1's +10.6 % generation A/B (`P1-POOL-RANK.md`) |
| `arm-prefill.sh RANK LME TAG` | `-n 20`, `-ub 6144` | the prefill A/B across `-lme` 36/44/47 |
| `arm-coherence.sh RANK LME TAG` | `-p "The capital of France is" -n 24`, **no `-lv`** | text identity (9/9 byte-identical).  **Never hash an `-lv 3` log** — its timing lines interleave into the generated text |
| `arm-batched.sh RANK TAG` | `llama-batched-bench -npl 1,4,8`, `-lme 44` | the batched gate |

Fixed for every arm, and each for a measured reason:
`MOE_HOST_POOL_DIO=1` (a pool miss must be a real storage read, not a page-cache hit), `MOE_EXPERT_CACHE_MIB=4096`
(a partial arena, so the pool is where misses go), `MOE_HOST_POOL_RAM_PCT=0` (never let a ceiling shave the
arm), `LLAMA_KV_REGION_POOL=0` (not this campaign's question, and it costs 2.3 GiB), `-lzm auto -fa on`.

**The trace is not committed** (33.7 MB): regenerate it with the command in [`../STEP1-REPLAY.md`](../STEP1-REPLAY.md)
§1 — `MOE_EXPERT_TRACE=/tmp/fn-prose.trace MOE_EXPERT_CACHE_MIB=4096 …llama-cli -n 512`.  Record the trace's
own run header when you quote a replay number, because the model's calibration depends on the geometry
(`--arena-mib`/`--pool-frac` must be set to what the traced run used, or the hit rate is not comparable).

Host-specific by nature: `$HOME/llama.cpp/build-rocm/bin/...`, `/opt/rocm-10.1.0-gfx120X/lib`, and the model
directory `/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/`.  Adjust, or set the env the script reads.
