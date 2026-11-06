# Measurement harnesses for `archive/work/host-expert-split/`

One script per arm shape, so W1/W3 in [`../PLAN.md`](../PLAN.md) are reproduced rather than re-derived.
Every arm is meant to be run **under `scripts/gpu-watch.sh`**, which supplies the `timeout` and the flight
record.  **Put every env prefix on the WRAPPER, never inside `--cmd`**: gpu-watch word-splits `--cmd`, so
`FOO=1 cmd` becomes argv[0] and the arm silently never runs:

```bash
SAMPLE_NOTE=<tag> ../../../scripts/gpu-watch.sh --out /tmp/<tag>.csv --interval 3 --rocm-every 0 \
    --timeout 1200 --cmd 'LME=36 bash arm-decay.sh decode lme36-decode'
```

| script | shape | used for |
|---|---|---|
| `arm-decay.sh RANK TAG` (`LME=`, `NGEN=`, `CACHE_MIB=`, `RANK_FLOOR=`, `SINK=` in env) | `-n 64`, `-ub 4096`, the trace config | the headline decode A/B and the 4-arm decomposition (`P1-POOL-RANK.md`, `HANDOVER.md` §4.2) |
| `arm-prefill.sh RANK LME TAG` (`NGEN=`, `SINK=` in env) | `-n ${NGEN:-20}`, `-ub 6144` | the prefill A/B across `-lme`.  **Use `NGEN=64` for any gen number**: a 20-token generation under-reports gen by 13-15 % (41.6 vs 35.8 ms/token — warm-up plus the pool's first ~20 decoded tokens) |
| `arm-coherence.sh RANK LME TAG` (`NGEN=`, `SINK=` in env) | the **prose prompt**, `-n 24`, `-lv 3`, **streams split** | text identity.  Three traps, all measured: (1) **the pool is built at the first decode-band pass**, so a short prompt never engages it — the original 9/9 claim was measured **with no pool at all**; (2) this CLI logs **nothing** at its default threshold (not even ggml `WARN`), so the old vacuity check reported VACUOUS on every run *while the pool was engaged* — the arm now runs `-lv 3`; (3) the old "never hash an `-lv 2/3` log" rule was about a **merged** `2>&1` file (the per-token timing interleaves into the captured region).  The arm now runs `> OUT 2> LOG`, and because the logger writes every level to stderr while the generated text is written raw to stdout, the hashed text is byte-clean at any verbosity while the pool report lands in the log — one run proves both.  Reference hash with the pool active: **`sha=880c932888f8`** (126 chars) |
| `arm-batched.sh RANK TAG [LME]` (`LME=` default 44, `SINK=` in env) | `llama-batched-bench -npl 1,4,8` | the batched gate.  **Runs `-lzm auto`, not `off`**: with PLE resident (~24 GiB) plus a 36.8 GiB pool and a 23 GiB arena the box over-subscribes and `npl 8` collapses from ~33.7 to 8-18 t/s (MemAvailable min 46 -> 18 GiB), which reads exactly like a policy regression and is not one |

## Two knobs every arm honours, and both are measurement-critical

* **`LV=`** (default 3).  **Any counter you intend to quote needs `LV=4`**: `common_log_get_verbosity()` maps
  ggml `INFO` to `LOG_LEVEL_TRACE` (4), so at `-lv 3` every ggml counter line (`pool fills`, `prefill cache
  staging`, `coalesced prefill reads`, `per-expert storage reads`, `scheduler pageable-master fallback`) is
  invisible.  That is how this campaign once blamed the wrong function.  `-lv 4` costs nothing measurable.
* **`HEATMAP=`** (default **`load`**).  The persisted prior is an **input**, not scratch state: a writable one
  is rewritten by every run, and if the workload is the benchmark prompt the prior converges on *that prompt*
  and the number stops measuring the delivery (measured at `-lme 36`: 26.2 t/s on a frozen real-world map vs
  27.9 on one drifted by the benchmark's own runs).  The arms default to `MOE_EXPERT_HEATMAP=load` and the
  sidecar is additionally `chmod 444` in the model directory, which the code itself treats as the freeze
  signal.  `HEATMAP=1` is only for regenerating a map on purpose.

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
