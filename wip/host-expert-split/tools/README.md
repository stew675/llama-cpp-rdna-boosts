# Measurement harnesses for `wip/host-expert-split/`

One script per arm shape, so W1/W3 in [`../PLAN.md`](../PLAN.md) are reproduced rather than re-derived.
Every arm is meant to be run **under `scripts/gpu-watch.sh`**, which supplies the `timeout` and the flight
record:

```bash
SAMPLE_NOTE=<tag> ../../../scripts/gpu-watch.sh --out /tmp/<tag>.csv --interval 3 --rocm-every 0 \
    --timeout 1200 --cmd 'LME=36 bash arm-decay.sh decode lme36-decode'
```

| script | shape | used for |
|---|---|---|
| `arm-decay.sh RANK TAG` (`LME=`, `NGEN=`, `CACHE_MIB=`, `RANK_FLOOR=` in env) | `-n 64`, `-ub 4096`, the trace config | P1's +10.6 % generation A/B (`P1-POOL-RANK.md`) |
| `arm-prefill.sh RANK TAG` (`LME=` in env) | `-n 20`, `-ub 6144` | the prefill A/B across `-lme` 36/44/47 |
| `arm-coherence.sh RANK TAG` (`LME=`, `SINK=` in env) | the **prose prompt**, `-n 24`, **default verbosity** | text identity.  Two traps, both measured 2026-11-06: (1) **never hash an `-lv 3` log** — its per-token timing interleaves into the generated text (1691 chars instead of 126); `-lv 2` is also unusable (WARN lines land in the captured region), so hash at **default verbosity**; (2) **the pool is built at the first decode-band pass**, so a short prompt (`-p "The capital of France is"`) never engages it — the original 9/9 claim was measured **with no pool at all**.  The arm now uses the prose prompt; pool engagement is verified out-of-band at `-lv 2` on the identical config (it cannot change the computation, only the logging), because the pool report is a WARN and default verbosity hides it.  Reference hash with the pool active: **`sha=880c932888f8`** (126 chars) |
| `arm-batched.sh RANK TAG [LME]` (`LME=` default 44) | `llama-batched-bench -npl 1,4,8` | the batched gate.  **Runs `-lzm auto`, not `off`**: with PLE resident (~24 GiB) plus a 36.8 GiB pool and a 23 GiB arena the box over-subscribes and `npl 8` collapses from ~33.7 to 8-18 t/s (MemAvailable min 46 -> 18 GiB), which reads exactly like a policy regression and is not one |

Fixed for every arm, and each for a measured reason:
`-lm mmap -lzm off` (**`LZM` overrides**; the batched arm defaults to `auto` for the memory reason above) — under `-lzm auto` the CPU-offloaded PLE tensor is mmap-lazy and
page-fault loads *during* the measurement, competing for the page cache the pool deliberately bypasses; the
same cell measured +10.7 % (`auto`) vs +9.6 % (`off`).  Never compare process `read_bytes` across lazy modes.

`MOE_HOST_POOL_DIO=1` (a pool miss must be a real storage read, not a page-cache hit), `MOE_EXPERT_CACHE_MIB=4096`
(a partial arena, so the pool is where misses go), `MOE_HOST_POOL_RAM_PCT=0` (never let a ceiling shave the
arm), `LLAMA_KV_REGION_POOL=0` (not this campaign's question, and it costs 2.3 GiB), `-lzm auto -fa on`.

**The trace is not committed** (33.7 MB): regenerate it with the command in [`../STEP1-REPLAY.md`](../STEP1-REPLAY.md)
§1 — `MOE_EXPERT_TRACE=/tmp/fn-prose.trace MOE_EXPERT_CACHE_MIB=4096 …llama-cli -n 512`.  Record the trace's
own run header when you quote a replay number, because the model's calibration depends on the geometry
(`--arena-mib`/`--pool-frac` must be set to what the traced run used, or the hit rate is not comparable).

Host-specific by nature: `$HOME/llama.cpp/build-rocm/bin/...`, `/opt/rocm-10.1.0-gfx120X/lib`, and the model
directory `/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/`.  Adjust, or set the env the script reads.
