# Running an oversized Q8_0 MoE on a single card — a reproducible config guide

**Audience:** someone with one RDNA4 card (this was measured on a single R9700 / gfx1201, ROCm 7.14.1)
and a model + KV cache that does **not** fit in VRAM, so some MoE layers must be offloaded.

**Scope:** the decode-side MoE expert cache (campaign record: `archive/work/moe-expert-cache/`).  `MOE_EXPERT_CACHE_MIB` arms a
persistent per-device VRAM cache of the hot experts; `-ncmoe N` decides how many MoE layers' experts
live in the host pool.  All numbers below are single-GPU unless stated.

## The method (read this before comparing numbers)

* **One process, both sides.**  Measure prefill and decode in the **same** `llama-bench` invocation
  (`-p 8192 -n 1024`).  A warm cache from the prefill is part of the result; a standalone `-p 0 -n 1024`
  run reports a different decode.
* **Warm the model.**  Repeat (`-r 2+`) and quote the **last** rep.  qwen4exp has a 26.8 GiB
  `per_layer_token_embd.weight` (PLE) that is mmap-lazy by default; it warms with every pass.
* **`--lazy-mode off` for qwen4exp.**  This loads the PLE into host RAM and removes the warm-up: prefill
  becomes **flat** (~2650 t/s at `-p8192 -ub2048` instead of a cold ~1600 climbing to ~1790).  Every
  qwen4exp prefill comparison must use it.  qwen35moe/Q4_K_M have no lazy tensor.
  (`LLAMA_LAZY_BUF_MB` sizes a managed RAM buffer instead.)
* **Assess at the real context.**  The target is **>= 131072 tokens with a q8_0 KV cache** (`-ctk q8_0
  -ctv q8_0`, which needs `-fa 1`).  `llama-bench` has no `-c`; it uses `-d <depth>`
  (`n_ctx = -p + -n + -d`), and `tg@depth` is the deep decode.  A shallow f16 run leaves far more VRAM
  for the expert arena and reports numbers that do not apply.
* **Pin threads:** `-t 8`.  The GPU IRQs on this host live on the top cores; a default `-t` starves the
  cards.
* **Never run two benches at once.**
* **`-f`/`--single-turn --no-display-prompt`** for `llama-cli`; `-v` only for diagnostics (it corrupts
  the hash extractor).

## The axes

| axis | what it does | notes |
|---|---|---|
| `-ncmoe N` | host-resident MoE layers | higher N = more experts in RAM; prefill falls, decode is capped by residency |
| `MOE_EXPERT_CACHE_MIB=<MiB>` | per-device VRAM cache budget | the decode win; `≥ slots×expert_bytes` = full residency (`h=1`) |
| `-b` / `-ub` | batch / ubatch | `-ub 4096` is the sweet spot here; `-ub 8192` is usually too big (VRAM/OOM) |
| `-sm layer` vs `-sm tensor` | split | 1 GPU: `layer`.  Multi-GPU: `tensor` beats `layer` at every offload level (the r16 record), and the gap is large at low residency: 2x R9700 on the **PCIe 5.0 x8** root ports, Flash-Next IQ4_NL, `-ncmoe 48`, 2000-token decode — **35.5 t/s (`tensor`) vs 25.7 (`layer`)**, i.e. +38 %, at the *same* expert hit rate (99.74 % vs 99.48 %, so it is the per-layer cross-device split and its split/event/handoff cost, not residency).  `layer` is the CLI default, so pass `-sm tensor` explicitly.  (`GGML_CUDA_DISABLE_GRAPHS=1` measured within noise of `tensor` here until r149, because the decode graph was captured but never replayed on this shape — `TODO #107`, closed by the r149 graph-cache key; the `tensor` arm is **37.7 -> 42.8 t/s** with it, so re-measure `DISABLE_GRAPHS` as the *control* now, not as an equal.) |
| gather vs staging | how the used experts are uploaded | **staging is the default.**  The device gather is OFF by default (2026-10-03): its one-time expert-head zero does not survive a multi-ubatch prefill, so its benchmarks were corrupt (NaN routing skips work — measured 2650-3060 t/s, above the PCIe limit for the bytes copied).  The staging/host path copies the MMQ guard pad every pass and is correct at ~655-1400.  `GGML_SCHED_DEVGATHER=1` re-enables the gather for A/B only. |

## The headline config — Qwen3.6-35B-A3B Q8_0 (37.8 GB) on one R9700 (32 GiB)

This is the representative case: the weights **do not fit** the 32 GiB card, so `-ncmoe` is mandatory and
the expert cache is what makes decode usable.  The realistic target is a **131072-token context with a
q8_0 KV cache** — that KV is what the model + arena must leave VRAM for, and it changes the tuning
versus a shallow f16 setup.

### 128K context, q8_0 KV (the assessment target)

`-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -t 8`, `MOE_EXPERT_CACHE_DEVMAP=1`,
measured with `llama-bench -p 8192 -n 1024 -d 131072 -r 1` (`n_ctx = p + n + d` = 140288; `tg128k` is
1024-token decode at depth 131072):

| `-ncmoe` | `MIB` | pp8192 (t/s) | tg128k (t/s) | note |
|---:|---:|---:|---:|---|
| 12 | 8192 | **967** | 39.3 | max prefill |
| 16 | 8192 | 919 | 53.3 | |
| **20** | **8192** | **884** | **57.1** | **recommended balanced default** |
| 24 | 12288 | 843 | **57.7** | best decode |
| 40 | 12288 | 732 | 48.7 | |

The arena beyond 8192 does not help at `-ncmoe 16-20` (MIB 12288 gives the same 920/53.4 and
884/57.1), and `-ub 8192` trades deep decode for prefill (986 / 45.2).  `-ncmoe 8` does not fit at this
context.  Deep decode at 128K (~53-58 t/s) is naturally below the shallow number (~81).

**Recommendation:** `-ncmoe 20 MOE_EXPERT_CACHE_MIB=8192 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 -t 8`
→ **884 prefill / 57.1 deep decode**.  `-ncmoe 12` if prefill matters more (967/39.3); `-ncmoe 24` for
the last bit of decode (843/57.7).

**Coherence verified at this setting**: a ~110K-token reference prompt + the essay task at `-c 131072`
q8_0 KV produced all 12 requested sections plus a conclusion (6253 words, fluent, zero `////`, rc=0) -
the model stays coherent at deep context with the cache active.

### Shallow reference — `-c 8192`, f16 KV (older records)

`-ngl 99 -sm layer -fa 1 -b 4096 -ub 4096 -t 8 -p 8192 -n 1024 -r 2`:

| `-ncmoe` | `MIB` | pp8192 | tg1024 |
|---:|---:|---:|---:|
| 12 | 8192 | **3122** | 79.9 |
| **16** | **8192** | **2706** | 81.1 |
| 20 | 12288 | 2404 | 81.8 |
| 24 | 12288 | 2139 | 78.6 |
| 32 | 12288 | 1786 | 70.5 |
| 40 | 12288 | 1554 | 61.8 |

These are **not** representative of the 128K+q8_0 use case (the small f16 KV leaves much more VRAM for
the arena); they are kept only to compare against older records.  `-ub 4096` still lifts prefill a lot
versus `-ub 2048` at the same `-ncmoe`/`MIB` (e.g. `-ncmoe 16 MIB=8192`: 2706 vs 1751) with the same
decode.  The gather is neutral for Q8_0 at every width (272 MiB tables, above the 224 MiB gate).

### ub 2048 reference

| `-ncmoe` | `MIB` | pp8192 | tg1024 |
|---:|---:|---:|---:|
| 8 | 4096 | 2635 | 72.1 |
| 16 | 8192 | 1751 | 81.0 |
| 16 | 16384 | 1753 | 81.8 |
| 24 | 12288 | 1326 | 78.1 |

## The qwen4exp point — Qwen3.8-Flash-Next IQ4_NL (100 GB) on one R9700

`-ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode off --load-mode none -t 8`, `MIB=12288`:

| `-b`/`-ub` | pp8192 | tg1024 |
|---:|---:|---:|
| 2048 | 659 | 39.7 |
| 4096 | 1018 | 40.3 |
| 8192 | 1396 | 35.9 |

**These are staging/host-copy numbers (the default).**  The previously-recorded gather numbers
(2650-3060 t/s) were **corrupt**: the device gather's one-time expert-head zero does not survive a
multi-ubatch prefill, so the MMQ tail over-read read stale NaN, and NaN MoE routing short-circuited
experts.  A bandwidth check confirms it — ~68 GB is read from host RAM per ubatch, so 3086 t/s at
`-ub 8192` would need ~25.6 GB/s, above this box's PCIe link, while 1396 t/s needs ~11.6 GB/s.  The
gather is now OFF by default; `GGML_SCHED_DEVGATHER=1` re-enables it for A/B.  The correct gather
(persistent destination, gate-verified) is ~683/1532, i.e. about the same as staging — so the
long-claimed "2-4x gather prefill win" was the corruption.  Full record: `archive/work/moe-mmq-overread/RESOLUTION.md`.

The prefill is PCIe-bound (~11-12 GB/s effective); the realistic target is a >= 131072-token context
with a q8_0 KV cache, and `-b 4096 -ub 4096` (1018) is the sweet spot on this box.

## Repeat it yourself

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
BIN=./build-rocm-b3/bin/llama-bench
MQ8=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
# the assessment target: 128K context, q8_0 KV (n_ctx = 8192 + 1024 + 131072)
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_DEVMAP=1 \
  $BIN -m "$MQ8" -ngl 99 -ncmoe 20 -sm layer -fa 1 -ctk q8_0 -ctv q8_0 -t 8 \
       -p 8192 -n 1024 -d 131072 -b 4096 -ub 4096 -r 1

# qwen4exp (PLE fully in RAM, gather automatic):
IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  $BIN -m "$IQ4" -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode off --load-mode none \
       -t 8 -p 8192 -n 1024 -b 2048 -ub 2048 -r 2
```

## Gotchas

* `-ncmoe 8` on the Q8_0 35B fits at `-ub 2048` but **OOMs at `-ub 4096`** - fewer offloaded layers
  means more expert bytes on the card.  Use `-ncmoe >= 12` with `-ub 4096`, or `-ub 2048`.
* `-ncmoe` and `MOE_EXPERT_CACHE_MIB` are independent: `-ncmoe` is a static placement, the arena is a
  persistent decode cache over the same host pool.  Set `MIB` large enough and the arena *is* `-ncmoe 0`
  for decode while prefill still streams from host.
* `--fit` knows nothing about the arena (it is allocated from free VRAM after `--fit`), so `-c`/`--fit`
  and a large `MIB` can both claim the same headroom.
* **A big MoE on 2 cards: pick `-ncmoe` from the RAM budget, not from the card.**  Flash-Next
  GSQ-RCO IQ3_XXS (71 GB) on 2x R9700, 4 slots / unified KV at `-c 40960`: **`-ncmoe 44`** keeps four
  layers' experts on the cards and pins **37.8 GB** of host experts (the arena then holds 100 % of them,
  22.1 + 21.2 GiB of slab per device), which leaves headroom on a 91 GB box; `-ncmoe 48` pins ~85 GB and
  the OOM killer takes the run (measured 2026-10-26, `WORKLOG.md` 261026.004).  Same rule for any quant:
  pinned host experts = (n_cpu_moe / n_layer_moe) x the expert bytes, and the arena needs them resident.
* **Windows: check which HIP runtime is loaded before you trust a pinned-host number.**  If the startup
  line says `HIP runtime ... from C:\Windows\System32`, that is the **display driver's** copy, not the
  ROCm/TheRock runtime, and its pinned host tier stops at ~9.3 GiB of adapter shared usage (issue #278)
  regardless of RAM -- so the pool caps out or latches off.  Put the ROCm/TheRock `amdhip64_7.dll` (and
  the rest of its runtime DLLs) beside the executable; the same binary then pins 16-40 GiB.  If you cannot
  switch runtimes, `MOE_HOST_POOL_PIN=0` closes the pinned tier (correct, slower).  See `README.md` →
  "Windows: use the ROCm (TheRock) HIP runtime".

## Verify your build (recommended)

The always-on expert **gather** must not corrupt the tile: the r29 regression was an under-sized gather
head pad that only bit IQ4_XS (the gate set had used IQ4_NL).  The r30 gate checks, for **every** qwen4exp
quant present, that the gather-ON output is coherent (no `////`) and matches the gather-OFF output:

```bash
# both runs per quant (IQ3_XXS/IQ4_NL/IQ4_XS/Q4_K_M by default)
QUANTS="IQ4_NL IQ4_XS" ./scripts/gate-qwen4exp-quant-coherence.sh
```

It needs no `MOE_EXPERT_CACHE_MIB` (the gather is the always-on half) and exits non-zero on a corrupted
or non-transparent quant.  A `////` in the gather-ON output is a hard failure; a hash mismatch is a WARN
by default (qwen4exp `-sm tensor` is run-to-run nondeterministic at temp 0) and `STRICT=1` makes it a
hard failure.  See `WORKLOG.md` 2026-10-02 (r30).
