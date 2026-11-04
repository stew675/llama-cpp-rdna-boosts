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

## 7. Why a fully resident pool still decodes slower at `-n 300` — it is warm-up, not the pool

Asked after the first round: *if the pool can hold every weight, why is anything still cold, and does the pool
need to be armed better at start-up?*  Short answer: nothing is missing from the pool, "cold" is the **VRAM
arena**, not the host tier, and the decode gap is a **warm-up transient that both arms share** — at `-n 2000` it
is 2.5 %, and the *marginal* rate differs by 0.5 %.

### 7a. The pool is fully loaded, and it never churns

`MOE_HOST_POOL_MIB` = the whole set (full coverage), 2 devices, `-n 300`:

```
host pool (L2): 120 tensors, 30720 slots, 18662.0 MiB pinned; h=0.6303, fills=30720, evictions=0
```

30 720 slots for 30 720 experts, each filled exactly once (the prepopulate), **zero evictions, zero storage
reads** — the duplicate-free guarantee is intact and nothing is re-read from disk.

### 7b. "Cold" is the arena, and its mix is the same in both arms

| access outcome | pinned master | full-coverage pool |
|---|---|---|
| **hit** — the expert is in the **VRAM arena** | 74.06 % | 75.01 % |
| **cold** — not in VRAM; read **in place** over PCIe (no copy) | 20.69 % (38 075 MiB) | 17.99 % (31 990 MiB) |
| **fill** — not in VRAM; copied into an arena slot | 5.26 % (9 680 MiB) | 7.00 % (12 448 MiB) |
| total H2D | 47 755 MiB | **44 438 MiB** |

The arena is `MOE_EXPERT_CACHE_MIB` (2 041 MiB/device here) against an 8.8 GiB/device expert set, i.e. **56 of
256 experts per table**, and the `touch` doorkeeper deliberately serves a one-shot expert in place instead of
evicting a hot resident (`GREEDY-PURITY`): "cold" is a *policy choice about VRAM*, present in both arms.  The
pinned arm is **not** less cold — it does *more* in-place reads and moves *more* bytes.

### 7c. The gap is the first few hundred tokens

decode t/s, 2 devices, identical text at every n:

| `-n` | pinned master | full-coverage pool | gap | wall difference |
|---|---|---|---|---|
| 50 | 66.5 | 51.7 | −22 % | +0.22 s |
| 100 | 68.1 | 55.9 | −18 % | +0.32 s |
| 300 | 71.9 | 63.9 | −11 % | +0.52 s |
| 500 | 75.5 | 69.7 | −7.7 % | +0.55 s |
| 600 | 76.9 | 71.8 | −6.6 % | +0.56 s |
| 1000 | 80.1 | 76.4 | −4.6 % | +0.61 s |
| 2000 | 82.9 | 80.8 | **−2.5 %** | +0.63 s |

**Both arms warm up** (pinned 66.5 → 82.9; pool 51.7 → 80.8) as the arena's LFU/heat placement converges, so the
interesting quantity is the *difference*: ~0.13 s fixed (the n=50 point) plus ~0.5 %/token (300 → 2000 gives
**85.2 t/s pinned against 84.8 t/s pool** on the marginal 1 700 tokens).  So yes: the gap nearly closes at
`-n 2000`, and what is left is not a pool-residency problem.

### 7d. Candidates tested and ELIMINATED (all measured, 2 devices, `-n 300`)

| candidate | test | result |
|---|---|---|
| the fill copy kind (D2D from the pool alias vs H2D from pinned) | `MOE_HOST_POOL_H2D=1` | 63.7 vs 63.9 — no |
| the sync-free device remap (only a pinned master qualifies for it) | it never engages in *either* arm (no `device remap: N calls` line); `GGML_MOE_DEVREMAP=0` | 63.7 — no |
| the cold-read availability (fewer in-place reads in the pool arm) | `MOE_HOST_POOL_COLD=0` | 58.2 → cold reads are worth **+5.7 t/s**, i.e. a win, not the cost |
| per-access host work: `table_pool_locked()` was a `std::map` (tree) lookup on the per-ACCESS path | cached the pointer on the table + `std::unordered_map` (implemented, measured, then **reverted**: no measurable effect) | 63.8 / 81.1 against 63.9 / 80.8 — no |
| the pool's 18.22 GiB population (1 033 ms) landing inside the measured window | `MOE_HOST_POOL_PREWARM=0` | +3.0 t/s at n=300, **+0.3 at n=2000** → it is ~0.09 s of the window, not the gap |
| traffic | the census above | the pool arm moves **less** H2D |

So the pool does not need "better arming": its contents are complete and static, and moving the population to
the preflight (the only remaining arming step) is worth ~0.09 s of a short run.  The residual ~0.5 %/token is
**unattributed**; the two structural differences left are that a pool-backed table keeps the pool-backed
*prefill* staging branch (irrelevant to decode) and that its arena fills read the pool alias rather than a pinned
master (tested above, a wash).  Recorded so nobody re-derives these six negatives.

## 8. Why the pool bounced while "fully resident": it is the arena's COMPLEMENT by design (and the fix)

**This supersedes §7c/§7d's "unattributed ~0.5 %/token".**  Asked: *if the pool holds every weight, why is anything
bouncing?*  Because **"fully resident" is a claim about the pool's contents right after prepopulation, and the pool
then deliberately un-loads experts** — its design contract (TODO #87) is *the arena's COMPLEMENT*, because pinned
RAM is the scarce resource.  Two code paths enforce it, both after the prompt:

| where | what it does at full coverage |
|---|---|
| `fill_slot_from_host` (the arena sizing/seed) | copies an expert into VRAM, then `expert_src_done(..., retire=true)` marks its **pool slot retiring** (`slot_state = 2`) — the expert is dropped from the pool *because* it is now in VRAM. 13 440 of 30 720 slots |
| `pool_rerank_from_heat_locked` (heat re-rank, first decode-band hook) | keeps only the wanted NON-resident experts — *"an expert already in the arena needs no pool slot"* — and frees the rest.  Logged per table: `0 filled, 56 freed, 200 wanted non-resident` |

So from the first decode step the pool is the *complement* of the arena: it holds the 200 of 256 experts that are
not in VRAM and has **no copy of the 56 that are**.  When the arena's `touch` policy later evicts one of those 56
and its bytes are needed again, `expert_host_src` peeks the pool (miss), `pool_fill_empty_locked` needs a *free*
slot (a pool full of other residents has none), and the fill falls through to the **bounce ring**: a synchronous
**288 KiB `memcpy` out of the model mapping** plus an H2D.  Measured **20 116 such bounce fills in the first 300
decode tokens** (~5.5 GiB of host memcpy at ~7 GB/s ≈ 0.5 s).

Why every earlier attribution missed it: the fill probe (`MOE_HOST_PUSH_PROBE`) times the `cudaMemcpyAsync` only
(1.2-1.4 µs for a bounce fill) — the ~25 µs host memcpy that precedes it is in `expert_host_src`, invisible to
every copy-timing A/B.  A pinned master never bounces: `table_master_pageable()` is false, so `expert_host_src`
returns `master + e*stride` (0.3 µs H2D) for every expert, always — it never forgets.

**Fix (patch `0012`): when the budget covers the whole tensor (`slots >= n_experts`) the pool is a MIRROR.**
Neither path may un-load an expert because VRAM also has it.  A bounded pool (`slots < n_experts`) keeps the
complement policy byte-for-byte, so the not-fitting band and `-lme N` are unchanged *by construction*.

| 2 devices | n=50 | n=300 | n=500 | n=2000 | prefill |
|---|---|---|---|---|---|
| full pool, before | 51.7 | 63.9 | 69.7 | 80.8 | 1166.4 |
| + retire fix | 56.9 | 67.2 | 71.1 | 81.6 | — |
| **+ rerank fix (MIRROR)** | **64.5** | **70.4** | **73.6** | **82.6** | **1315.9** |
| pinned whole-tensor master | 66.6 | 71.9 | 75.5 | 82.9 | 1105.4 |

Byte-identical text in every pair (`231512fc77e5` @n=300, `c6e6d709a6db` @n=500, `4ae33f7bd65c` @1 device,
`d589ad909b67` @3 devices), `check-gpu-faults` clean, bounded pools unchanged (FRAC 50 56.5, `-lme 8` 54.0),
1 device 52.1, 3 devices `-sm layer` 48.1 at n=100.  All §8 numbers were taken with the persisted heat prior
**frozen** (read-only sidecar) so both arms seed from the same prior.

**Consequence, and it is a policy question**: a full-coverage pool now prefills **19 % faster than a pinned
master** (1315.9 against 1105.4) and decodes within 0.4-3 % of it (n=2000 82.6 against 82.9; n=300 70.4 against
71.9) **for the same 18 662 MiB of pinned RAM**.  So `-lme auto`'s fit case could be one mechanism (the pool as a
mirror) instead of the whole-tensor pinned-master tier — the trade is +19 % prefill for ~-2 % decode at
n≤500.  Not changed here: it is the maintainer's call, and it would remove P5's per-tensor tier entirely.

## 9. The read-ahead back-fill was built and it LOSES (patch `0013`, opt-in)

The maintainer's refinement of the spill — *eagerly back-fill the evicted expert into the slot the promoted expert
vacates, from the page cache/disk, on a worker thread, so the union `{VRAM ∪ pool}` stays closed without any PCIe* —
was implemented exactly as described and **measured a loss in both bands** (text byte-identical, faults clean):

| band | read-ahead off | on | jobs | host memcpy moved off the critical path |
|---|---|---|---|---|
| page cache, FRAC 50, n=300 | **56.4** | 52.8 | 12 252 | 640 ms |
| `O_DIRECT` disk, FRAC 50, n=100 | **15.9** (prefill 24.1) | 15.1 (prefill **16.3**) | 7 984 | 2 534 ms |

The three reasons, all measured rather than argued:

1. **The victim is the arena's least-demanded resident by construction.**  `access_locked` evicts the smallest
   decaying `count` (tie: oldest use), so an eager copy of it is a bet against the very signal that chose it.  The
   obvious predicate — the demand-side `ghost` counter — cannot be used: it is reset to 0 on admission, so it is 0
   for *every* resident, and requiring `ghost >= 1` rejected 100 % of victims (measured: zero jobs, baseline rate).
2. **The copy competes with the demand path for the same resource.**  In the page-cache band the host is already
   running 46 871 bounce memcpys for the coverage misses, so 12 252 more do not shorten the critical path; in the
   disk band the eager reads fight the demand reads for the NVMe queue and **prefill collapses 24.1 → 16.3 t/s**.
3. **The ceiling is low**: only ~26 % of the bounded band's fills can come from the pool at all (the rest are
   coverage misses that no policy can help), so even a perfect prediction could not pay for the extra reads.

It is the campaign's own L3 lesson in a fourth guise: *the pool's CONTENT matters far more than the hand-over
policy*.  The mechanism is kept, **default OFF** (`MOE_HOST_POOL_READAHEAD=0`), as an experiment — and one hard-won
implementation fact is recorded with it: **the worker must do host-only work**.  ROCm lazily spins up its runtime
threads, and doing that from a foreign thread aborts the process (`std::system_error: Operation not permitted`), so
the main thread polls each job's drain event (a bounded `cudaEventQuery` per access) and only then hands the job to
the worker; that also keeps the worker off `g_mutex` entirely, so a teardown can join it while holding the lock.

What *would* be needed: a predicate that predicts re-admission.  The campaign already owns the tool for that —
`scripts/moe-trace-replay.py` scores placement/fill policies against a real routing trace offline — so score
`heat-top-k`, recency and LRU-x predicates there before building another variant.

## 10. The surplus rule: one layout policy for every budget (and the pool is now the one mechanism)

The maintainer's generalisation, implemented as patch `0013` — it **subsumes** the mirror of §8:

> lay the pool out as **the complement first, then the arena-resident experts while capacity remains**.

| pool capacity | layout |
|---|---|
| `slots <  |complement|` | a bounded complement (the historical layout, unchanged) |
| `slots == |complement|` | exactly the complement |
| `slots >  |complement|` | the complement **plus** the hottest arena residents — a full mirror when the budget covers the tensor, i.e. "basically pinned" |

Two places used to un-load an expert merely because VRAM held it; both now apply the rule: the arena-seed retire
decision (`keep_all = slots > n_experts - n_arena`) and the heat re-rank's wanted set.  And with the pool carrying
every case, the loader no longer routes `-lme N >= 90 %` to the whole-tensor pinned-master tier: `auto` (fit) spends
the **whole set** on the pool, `-lme N` spends N GiB, both land in the same rule, and `--host-experts pinned` /
`MOE_HOST_TIER=tensor` remain the explicit pinned-master escape hatch.

Measured, 2 devices (0,2), 35B-A3B Q4_K_M, `-ncmoe 40`, decode n=300 / prose-5246 prefill, text byte-identical
(`231512fc77e5` @n=300, `6ede7c724f4e` @n=2000), `check-gpu-faults` clean:

| arm | prefill | decode n=300 | pool hit rate | pinned / pageable (MiB) |
|---|---|---|---|---|
| **`-lme auto` (fit → MIRROR)** | **1317.6** | 70.4 | **1.0000** | 18 662 / 0 |
| `--host-experts pinned` (escape hatch) | 1105.4 | **71.9** | — | 18 662 / 0 |
| `-lme 16` (86 %) | — | **66.5** (was 65.6) | 0.8368 | 16 329 / 2 333 |
| `-lme 12` (64 %) | — | 61.7 (was 61.6) | 0.5658 | 12 247 / 6 415 |
| `-lme 8` (43 %) | — | 54.1 (was 54.5) | 0.3631 | 8 165 / 10 497 |

**So the default is one mechanism** that beats a pinned master by **19 % on prefill**, is within 0.5-2 % on decode
(n=2000: 82.5 against 82.9), costs the same pinned RAM, and reports **`h=1.0000` with zero evictions and no
page-cache re-read at all** (30 960 pool fills = the one-time prepopulate).  The intermediate band gains where the
surplus appears (`-lme 16`: +0.9 t/s); a genuine sub-complement budget is untouched, as designed.

### Dropped: the read-ahead back-fill (§9), at the maintainer's request

It was built, measured as a loss in both bands, and removed — it raises complexity and adds a synchronisation
surface for no measured gain.  The code remains retrievable in the fork's history (`a7dfe8871`) and the measurement
is in §9 + `TODO` #140; nothing of it is in the patch chain.

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
