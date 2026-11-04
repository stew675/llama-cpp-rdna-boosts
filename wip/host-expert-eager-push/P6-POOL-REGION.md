# P6 — the expert-indexed region IS the pool; a partial `-lme` budget now buys residency, not tensors

**Status: IMPLEMENTED and MEASURED (2026-11-04), awaiting promotion with `P5`.**
`~/llama.cpp` branch `p5-host-expert-tier`, commit `72bc81b88` (on top of P5's `c1478f911`), exported as
[`patches/0011-p6-pool-region-cold-multidev.patch`](patches/0011-p6-pool-region-cold-multidev.patch).
This closes [`TODO.md`](../../TODO.md) `#139` in the campaign; nothing is promoted into `patches/` yet.

Read [`P5-RESULTS.md`](P5-RESULTS.md) §3 first: P5 measured that **at equal RAM a partial budget is better spent
per EXPERT than per tensor** (8 064 MiB of pinned whole tensors = 505 prefill / 50.1 decode against 713 / 51.8
for a 9 331 MiB pool) and deferred the per-expert region as `#139`. This is that work.

---

## 1. `#139` was already built — it is the bounded pinned host pool

The design of record asks for *"one pinned host region, addressed by expert index, with a presence bitmap, so
the two staging branches collapse into one hole-aware path"*.  The pool is exactly that object:

| design element | the pool today |
|---|---|
| a pinned region per host tensor | `host_pool_t::base` (`cudaHostAlloc(Portable\|Mapped)`), one per host tensor, process-wide |
| addressed by expert index | `expert_slot` (expert → slot) / `slot_expert` (slot → expert) |
| a presence bitmap | `slot_state` (populated) + `slot_hot`/`slot_lru` |
| a device alias | `base_dev` (`cudaHostGetDevicePointer`) |
| filled in descending hotness | `pool_prepopulate_locked` ranks by the persisted heat prior / the prompt's prefill routing |
| native fill transport | a resident fill is a **1.3 µs D2D** from `base_dev`; a hole is a page-cache (`memcpy` to the bounce ring) or `O_DIRECT` fill |

What it did **not** have was the right to be a **cold-read source away from one GPU**: `pool_cold_slot_locked()`
and `moe_cache_get_cold()` both required `device_count == 1 && src_off == 0`.  On a tensor split that meant
every resident expert still paid an arena fill plus an H2D, which is the whole of the pool's decode deficit on
this box.

Both restrictions are geometry accidents, not properties of the pool:

* `base_dev` is a `Portable|Mapped` UVA alias that is **already** used as the cross-device source for fills;
* a tensor split's slice offset is **already** folded into the cold base for a pinned master
  (`*cold_base = host_dev + t.src_off`), so fold it for the pool too.

So this is a two-line behavioural change plus a safety refactor: the two restrictions now live in **one**
predicate, `pool_cold_geometry_ok()`, used by the host-side admission decision *and* the kernel-side cold
description.  They must agree, or the host emits a cold id that the kernel resolves from a different base —
a silently wrong matmul, not a crash.  Fused tables stay excluded (one stride cannot express several ranges).
`MOE_HOST_POOL_COLD_MULTIDEV=0` restores the old restriction exactly (the A/B arm).

## 2. The budget now goes where it is worth more

`-lme N` spends its budget by the measured shape of the curve:

| `-lme N` | placement |
|---|---|
| `N` **below 90 %** of the host expert set | a pinned per-**expert** pool of N GiB (residency, heat-ranked) — the masters stay in the mapping |
| `N` at or above 90 %, or `auto`/`pinned` | the whole-tensor pinned-master tier of P5 (only a pinned *master* gets the single-launch prefill staging branch and the native fill) |

`MOE_HOST_TIER=pool|tensor` forces either shape (the A/B arm; not needed by a user).  The explicit budget
reaches the pool through a new device-API call, `moe_cache_set_host_pool_budget()`, so `-lme` no longer has to
abuse `MOE_HOST_POOL_MIB`.

The 90 % switch is measured on both sides but the exact crossover was **not** swept:

| 2 devices, 35B-A3B, `-n 300` | pool (per expert) | whole tensors (P5) |
|---|---|---|
| `-lme 8` (43 %) | **54.5** | 50.1 |
| `-lme 12` (64 %) | **61.6** | 56.8 |
| `-lme 16` (86 %) | 65.6 | 65.7 (tie) |
| `-lme 17` (91 %) | 67.0 | **68.2** |
| `-lme 64` / `auto` (100 %) | 63.9 | **71.6-71.9** |

## 3. The measurements

`Qwen3.6-35B-A3B-UD-Q4_K_M`, `-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=2048`, `--seed 42 --temp 0 --ignore-eos`,
`HIP_VISIBLE_DEVICES=0,2`, serial.  Prefill = `prompts/prose-rdna-boosts.txt` (5 246 tok), `-n 1`.

| 2 devices | prefill | decode (`-n 300`) | cold reaches | text |
|---|---|---|---|---|
| `pinned` masters (the fit reference) | 1105.4 | **71.9** | 125 286 | `sha=231512fc77e5` |
| `-lme 64` (≥ the set → tensors) | 1106.0 | 71.6 | — | `sha=231512fc77e5` |
| `-lme 8`, **`MOE_HOST_POOL_COLD_MULTIDEV=0`** (old restriction) | — | 48.9 | 0 | `sha=231512fc77e5` |
| `-lme 8`, MULTIDEV=1 (**new**) | 654.0 | **54.5** | 76 878 | `sha=231512fc77e5` |
| `-lme 12`, MULTIDEV=1 | — | 61.6 | — | `sha=231512fc77e5` |
| `-lme 17`, MULTIDEV=1 | — | 67.0 | — | `sha=231512fc77e5` |
| not-fitting band (`auto` → pool, FRAC 50 = 9 331 MiB) | 714.2 | **57.0** (was **51.8**) | 84 714 | `sha=231512fc77e5` |
| full coverage (`MOE_HOST_POOL_MIB` = the whole set, 18 662 MiB) | **1166.4** | 63.9 | 105 342 | `sha=231512fc77e5` |

So the multi-device cold read is worth **+11.5 %** decode at `-lme 8` and **+10 %** for the *shipped* not-fitting
pool, byte-identically — and a **fully resident pool now prefills faster than a pinned master (1166.4 against
1105.4)**, which nothing else in the campaign has achieved.  Decode at full coverage still trails a pinned
master (63.9 against 71.9), which is §5.

| other shapes | result |
|---|---|
| 1 device, `-lme 8` (pool) | 48.3, `sha=4ae33f7bd65c` (the 1-device cold path existed already; per-expert vs per-tensor is a tie there) |
| 1 device, full coverage | 51.4, `sha=4ae33f7bd65c` (the pool's documented 51.3) |
| 3 devices `-sm layer`, full coverage | 47.4 at `-lme 12`... `-n 100`, 8 191 cold reaches — **previously refused entirely** (device_count != 1) |
| 3 devices `-sm tensor`, `-lme 12` | 19.5 (`TODO` #135's degenerate split); **`sha=d589ad909b67` with MULTIDEV on and off** — the sliced/2-D geometry check |
| n=500, 2 devices: `pinned` vs full-coverage pool | **`sha=c6e6d709a6db` both** (a 757-token forced generation) |
| Flash-Next IQ3_XXS (`qwen4exp`), `-lme 12` pool vs `pinned` | **`sha=5fa645a83257` both**, `check-gpu-faults` clean (the pool arm is 20.7 t/s there because that config does not fit in RAM — the disk band, `TODO` #132) |

Every arm: `scripts/check-gpu-faults.sh` clean.

## 4. What this changes in the records above it

* `P5-RESULTS.md` §3 ("the partial band belongs to the pool") is now **actioned**, not deferred: `-lme N` routes
  the sub-90 % band to the pool, so the per-tensor tier only serves the fit case (where it is bit-for-bit
  `pinned`).
* `IMPLEMENTATION-PLAN.md` step b2 / c `[PARTIAL]`: the cold encoding did **not** need to change.  The encoding
  is `id - n_res_slots -> a slot index into one pinned base`, and the pool's base with its slot stride *is*
  such a base.  What was needed was the pool being allowed to be that base on a multi-device split, and the
  host-side decision and the kernel-side description agreeing (`pool_cold_geometry_ok`).
* `TODO` #132 (the pool's synchronous demand read) is untouched: non-resident experts still fill synchronously.

## 5. What is left — and it is now a *prefill-vs-decode* choice, not a hole

A **fully resident pool is not a pinned master**, so it keeps the pool-backed (two-tier + per-expert copy)
prefill staging branch and a pageable master.  Measured, that trades −8 decode points for +5.5 % prefill against
`pinned`.  The remaining step, if it is ever wanted, is to let a 100 %-covered pool **become** the master:

* fill slots in **expert order** when `slots >= n_experts` (so the region is expert-indexed: `slot == e`);
* rebind the table's master to the region (`region` / `region_dev` fields — **not** `t.host`, which is the key
  into `g_host_src`/`g_host_pools` and is shared by every per-device table of the same tensor);
* then `host_pinned` is true, the single-launch staging branch and the native fill apply, and the whole-tensor
  tier becomes unnecessary — one mechanism, coverage deciding the behaviour.

That is a real change with a narrow payoff (decode parity with `pinned` at 100 % coverage, plus deleting the
per-tensor tier); it is recorded rather than attempted, because the fit case already hits the `pinned` numbers
through P5's tier and the *interesting* band (a budget below the set) is now the pool's and measured better.

## 6. Reproduce

```bash
cd wip/host-expert-eager-push
P=../../prompts/prose-rdna-boosts.txt
# the A/B that isolates the multi-device cold read (identical text, +11.5 % decode)
GPU=0,2 SM=tensor EOS=1 N=300 LME=8 MOE_HOST_POOL_COLD_MULTIDEV=0 ./p3-matrix.sh lme8-old
GPU=0,2 SM=tensor EOS=1 N=300 LME=8                               ./p3-matrix.sh lme8-new
# the tier crossover (force either shape at the same budget)
GPU=0,2 SM=tensor EOS=1 N=300 LME=12 MOE_HOST_TIER=pool   ./p3-matrix.sh lme12-pool
GPU=0,2 SM=tensor EOS=1 N=300 LME=12 MOE_HOST_TIER=tensor ./p3-matrix.sh lme12-tensor
# full coverage: the first arm to beat a pinned master on prefill
GPU=0,2 SM=tensor N=1 PROMPT_FILE=$P POOL_FRAC=100 MOE_HOST_LIMIT_FRAC=1 ./p3-matrix.sh full-pre
```

Logs: `runs/p3-*-p5b-*` and `runs/p3-*-p6-*`.
