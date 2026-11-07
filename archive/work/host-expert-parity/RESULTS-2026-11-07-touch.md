# RESULTS 2026-11-07 (`radix`) — PLAN P1.3: `MOE_EXPERT_CACHE_TOUCH` 2 vs 3 on a live-demote cell

**Status: measured; P1.3 closes with the DEFAULT UNCHANGED (2).  No delivery change, no code.**  The knob does
exactly what PR #267's 0008 claimed mechanically — it damps churn — and the decode/prefill outcome is a wash,
so there is nothing to move.  This is the campaign's last open measurement.

**The question.**  `admit=touch` (the default admission mode) refuses a fill while the expert has been touched
fewer than `MOE_EXPERT_CACHE_TOUCH` times since it was last resident, so a rejected miss is served from the pool's
in-place cold read.  PR #267's 0008 changes `GGML_MOE_TIER_TOUCH` from 2 to 3 and claims the higher value damps
churn, which the campaign's offline replay predicted would **shrink the sink's (demote's) advantage by a third**
(it bounds the copies rather than saving L3).  The replay's warning was on the record; the on-device arm set was
not.

## 1. Environment of record

* `radix`: Ryzen 7 9800X3D, **one R9700** (gfx1201, 34.2 GB), 93 GiB RAM.
* `~/llama.cpp` @ **`a19aab55e`** = `release.json` `v16-6050b467c-r231`, tree `5aeb007deac`, `build-rocm` build
  `11903`, ROCm 10.1.0-gfx120X, `HIP_VISIBLE_DEVICES=0`.
* Field config, one load per arm: Flash-Next GSQ-RCO-IQ3_S + `mtp-…-Q4_K_M`, `-c 204800 -ub 6144`,
  `-ncmoe 48`, `GGML_CUDA_SLAB_HEADROOM_MIB=1536`, `LLAMA_KV_REGION_POOL=0`, `MOE_HOST_POOL_RAM_PCT=0`,
  `MOE_HOST_POOL_DIO=1` (a sub-complement tier should set it, r231), heat prior `MOE_EXPERT_HEATMAP=load`
  (frozen, never learns the prompt), `-f prompts/prose-rdna-boosts.txt` (5 298 tokens), greedy `-n 192`,
  `-lv 4` (**required** for the census: `common_log_get_verbosity` maps `GGML_LOG_LEVEL_INFO` to
  `LOG_LEVEL_TRACE`, so every census line above is `-lv 4`-only — see §5).
* Two **live-demote** cells, i.e. budgets below the near-mirror boundary where r228 engages the sink:
  `-lme 36` (pool 36 815 MiB of the 46 841.8 MiB set, 56 592 slots) and `-lme 44` (44 964.8 MiB, 69 120 slots).
  Arms **interleaved** (the order flips every rep), `scripts/check-gpu-faults.sh` clean afterwards.
* Harness: [`touch-ab.sh`](touch-ab.sh) / [`touch-ab.py`](touch-ab.py) (`./touch-ab.sh 3`; `LME_LIST`,
  `TOUCH_LIST`, `VERBOSE=` to override).

## 2. The arms (3 reps per cell per value, plus 3 more at `-lme 44`)

The cache census is **identical across reps** — this run is deterministic in its admission behaviour, so the
mechanism columns are one number, not a mean, and the only noise is in the rates:

| cell | `TOUCH` | arena hit | demoted | demo MiB | sink uses | reuse % | cold-served misses | cold % | fills |
|---|---|---|---|---|---|---|---|---|---|
| `-lme 36` | **2** (default) | 0.9217 | **10 102** | 6 391.5 | 1 567 | 15.5 | 15 147 | 3.27 | 70 340 |
| `-lme 36` | **3** | 0.9139 | **8 353** | 5 293.8 | 1 138 | 13.6 | 20 481 | 4.43 | 70 340 |
| `-lme 44` | **2** (default) | 0.9213 | **9 758** | 6 171.4 | 1 537 | 15.8 | 15 680 | 3.39 | 73 656 |
| `-lme 44` | **3** | 0.9132 | **7 973** | 5 048.6 | 1 135 | 14.2 | 21 206 | 4.58 | 73 657 |

**`TOUCH=3` does damp churn, exactly as claimed:** **-17.3 % / -18.3 %** of demotions (-17.2 % / -18.2 % of the
bytes copied), **-27.4 % / -26.2 %** of sink uses.  It pays for that with residency: the arena hit rate loses
**0.78 / 0.81 pt**, the cold-served share rises **+35.2 %** (3.27 -> 4.43 %, 3.39 -> 4.58 %), and the doorkeeper
rejects that many more fills (`cold admission=touch, admission-rejected (served cold)=15 147` vs `20 481`).  The
*number of fills* is unchanged (70 340 / 73 656), i.e. the same work is admitted, just later.

### 2.1 What the rates do

| cell | metric | `TOUCH=2` | `TOUCH=3` | delta |
|---|---|---|---|---|
| `-lme 36` | prefill t/s | 582.8 (583.1 / 580.6 / 584.7) | 593.5 (621.4 / 578.3 / 581.0) | +1.8 % (one 621 outlier; -0.5 % without it) |
| `-lme 36` | decode t/s | 44.1 (45.7 / 44.4 / 42.2) | 44.8 (48.0 / 41.9 / 44.4) | +1.6 %, no direction (2.4 t/s spread inside one arm) |
| `-lme 44` | prefill t/s | 938.7 (6 reps) | 933.5 (6 reps) | -0.6 % |
| `-lme 44` | decode t/s | 47.7 (6 reps: 46.0 / 48.0 / 48.0 / 48.0 / 47.9 / 48.2) | 48.7 (6: 49.0 / 48.2 / 48.7 / 48.2 / 48.5 / 49.7) | +2.2 % raw, **+1.3 %** dropping each cell's first (warm-up) rep |

So the on-device outcome is a wash: **+1.3 to +2.2 % decode** at 94 % coverage (inside this box's ±3-5 %
single-run spread, though the sign held over six interleaved reps), **-0.6 %** prefill, for 0.8 pt of arena
residency and 35 % more cold reads.  The demote stays worth having under both settings: **reuse 13.6-15.8 %**
against W1's break-even ~3 % on NVMe / ~0.2 % on slow storage.

**Verdict: the default stays 2.**  A knob with no measured win does not move, and the replay's warning ("touch-3
shrinks the sink's advantage by a third") is confirmed in the mechanism (-18 % of the copies) without any
decode benefit to buy.

## 3. Coherence

`TOUCH=2` vs `TOUCH=3` at `-lme 36`, default verbosity, same seed: **925 chars, `sha=0e3ed3d279f6` in both arms**.
The knob changes which experts are *resident*, never what is computed — as expected for a cache policy, and the
reason this item's only gate is text equality.

## 4. What closes, and what it means for PR #267

* **PLAN P1.3 — closed.**  `MOE_EXPERT_CACHE_TOUCH` keeps its default (2); `ENVIRONMENT.md`'s row now carries
  the measured trade (§2) so an operator choosing 3 knows what it buys (-18 % demotions) and what it costs
  (0.8 pt residency, +35 % cold reads, no decode win).
* **PR #267 0008 — answered**: the value does damp churn, and it is not a performance lever here.  With that,
  every mechanism in the PR is closed (see [`PR-267-REMAINDER.md`](PR-267-REMAINDER.md) §3) and the PR is ready
  to be closed as *integrated rather than merged*.

## 5. The trap this pass corrected

r231 recorded that `-lv 4` interleaves llama-cli's statistics block into the generated text.  The sharper rule,
and the one that matters for a census arm:

* `common_log_get_verbosity()` maps `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (= 4), so **every ggml census
  line needs `-lv 4`** ("MoE expert cache:", "expert access:", "takeover=", "cold admission=");
* llama-cli's own `LLAMA_LOG_INFO` is `LOG_LEVEL_INFO` (= 3), so `-lv 3` shows its per-slot statistics block —
  which is exactly the text pollution — while hiding the census.

So: **census/mechanism arms take `-lv 4`; text-identity arms take no `-lv` at all.**  `-lv 3` is the one setting
that gives you the pollution without the numbers.
