# `wip/host-pool-speed/patches/` — the campaign's patch set (NOT the delivery)

**These are not part of the delivery.** They are not in the repo's top-level `patches/`, they are not applied by
`scripts/apply-all.sh`, and `release.json` does not know about them.  They exist so the experiment can be rebuilt
elsewhere (a review, another machine, the head-to-head in [`../HEAD-TO-HEAD.md`](../HEAD-TO-HEAD.md)).

## The set

| patch | what it does | files |
|---|---|---|
| `0001-moe-cache-report-the-RUN-*.patch` | **Diagnostics.** `moe_cache_report()` is an `atexit` hook that runs after `moe_cache_forget_all` has un-sized the cache, so every run ended with `tables=0, slots=0, arena=0.0 MiB` and `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned` — in *both* host-expert modes.  That line is what made “the pool never activated on the 122B” look like a delivery bug (`../HANDOVER.md` §4/F3).  The patch snapshots the report's structural aggregates at the first forget and has the report prefer them.  It also makes a forced `MOE_HOST_POOL_DIO=0\|1` log its effective mode: the *“(forced by MOE_HOST_POOL_DIO=N)”* branch was unreachable dead code because the env parse pre-set `g_pool_dio_decided`. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |
| `0002-host-pool-full-coverage-default-*.patch` | **The tuning changes.** (a) `MOE_HOST_POOL_FRAC` now defaults to **100** (full coverage of each host expert tensor), still clamped by `MOE_HOST_POOL_RAM_PCT` of `MemAvailable`.  (b) New **`MOE_HOST_POOL_SIZING=1`**: per-table pool slots from routing *concentration* (`cov_t` = the slots a table needs to hold `MOE_HOST_POOL_COV_PCT` of its own prior mass), fitted one-shot to the same total byte budget at the first pool build.  (c) The `--host-experts pool` help/log text that named the old 80 % default. | `ggml/src/ggml-cuda/moe-expert-cache.cu`, `common/arg.cpp`, `src/llama-model-loader.cpp` |
| `0003-host-pool-the-device-gather-resolves-arena-POOL-boun.patch` | **Mechanism (b): the device gather resolves arena -> POOL -> bounce.**  A pool-backed table's gather used to decline (no device-readable master), so the pool arm fell to the host path or to whole-table staging.  The kernel now takes the staging path's two-tier residency encoding (`>= 0` arena slot, `<= -2` pool slot `-v-2`), the gather builds its own copy of it on the compute stream (separate buffers from staging's `prs_`, uploaded only when it changes), and an expert in neither tier is copied into the destination from a pinned source before the launch — no kernel ever reads the pageable mapping.  Measured: srv-cell prefill 641-647 -> **832-878 t/s**, decode flat, pool ahead of `pinned` on prefill; 122B neutral; `MOE_HOST_POOL_COLD` no longer needed (`COLD=0` 15.54 -> 19.57 t/s).  Also adds the two diagnostic counters it needs (pageable-master fallback exposure, gather calls) and the gather accounting line. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |
| `0004-host-pool-account-for-the-pinned-bounce-ring-and-the.patch` | **Diagnostics + one owner, no behaviour change.** `bounce_acquire` becomes the pinned bounce ring's only allocator (`moe_cache_upload_pageable`'s duplicate acquire/grow loop is deleted), so the ring total is finally true — 2209-3108 MiB on the 122B arm, from the prefill miss path (measured; capping it costs 8-25 % prefill at FRAC 80, so it is deliberately uncapped, `../README.md` §16.2). Also times `pool_fill_slot_locked`'s two sources and prints calls/ms/GiB/effective rate — the instrument that sized the coalesce/overlap decision (**27.3-34.5 s of synchronous read per run, 3-4 ms per decode token**, `../README.md` §16.3). | `ggml/src/ggml-cuda/moe-expert-cache.cu` |
| `0005-host-pool-the-RAM-ceiling-scales-every-table-proport.patch` | **W3 (2026-11-03): the `MOE_HOST_POOL_RAM_PCT` ceiling is a DISTRIBUTION, not a tail shave.**  The old clamp cut the *registration-order tail* to the one-slot floor: on the 122B at `RAM_PCT=70` that was 26 of 144 tables at one slot and **10.94 t/s**, while 10 % more ceiling decoded 19.52.  Now the process-wide plan is compared with the ceiling **once** and *every* table's slot count is scaled by that factor (the one-slot floor only where the ceiling cannot cover a slot, with the old shave kept as a rounding backstop), and `MOE_HOST_POOL_SIZING=1` fits its concentration plan **to** the ceiling (a uniform rescale of a fitted plan collapses it back to uniform).  Measured (1 × R9700, 122B, `-ncmoe 48`, FRAC 100, 2000-token decode): `RAM_PCT 70` **10.94 → 17.44 t/s** (+73 % with cov90: 18.92) at identical pinned bytes, prefill 343 → 598; `RAM_PCT 85` **19.52 → 20.97**.  Nothing changes on a box where the ceiling does not bind. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |
| `0006-host-pool-a-POOL-backed-expert-table-gathers-in-the-.patch` | **#131: a POOL-backed table gathers in the PREFILL band, by default.**  The pool arm's whole-table *staging* is the slow path at every width (its source is the pageable model mapping plus the pool's host alias, and it degrades as coverage rises): 122B across three RDNA4 GPUs, 5258-token prompt, `-ub 2048`, 2000-token greedy decode — `pinned` 471.8 prefill / 67.6 decode; `pool` FRAC 80 **1502.6 / 67.1**; `pool` FRAC 95 **1569.2 / 67.1**; and with the gather switched OFF the same FRAC-95 arm collapses to **16.9** t/s prefill.  Mechanism: a new optional device-iface query `moe_cache_pool_backed(backend, weight)` (the CUDA cache answers "this table's master is the read-only mapping"; the Meta backend forwards it), used by `sched_input_gatherable`; a band gate (`n_tokens > moe_cache_band`) keeps the decode/verify band on the arena take-over (the gather costs that an order of magnitude: decode 1.2-4.1 t/s without it).  `GGML_SCHED_DEVGATHER` becomes **three-valued** (see the knob table).  Text: 257 chars `sha=5531ae33f0f7` on all eight mode/device arms, and 5835 chars `sha=dc178c29fe89` across `pinned`/`pool` F80/F95 at 3 GPUs with a 2000-token decode behind a 5258-token multi-ubatch gather-served prefill. | `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/moe-expert-cache.{h,cu}` |
| `h2d-experiment.patch` | **Not part of the set — a finished experiment, kept for the record.** `MOE_HOST_POOL_H2D=1` switches a pool hit/fill from the historical `cudaMemcpyDeviceToDevice` (from the pool's mapped UVA alias) to `cudaMemcpyHostToDevice` (from its pinned host pointer).  Measured: 17.79/17.78 vs 17.76/17.80 t/s — **no effect**, so the copy kind is excluded as an explanation of the pool's decode deficit (`../HANDOVER.md` §3.1).  Do not apply it unless you want to re-run that A/B. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |

**Set size: six patches** (three from session 2, `0004` from session 3, and `0005` (the `RAM_PCT` ceiling) + `0006` (the pool gather default) from session 4).

## Prerequisites and apply order

The set applies on the **delivery's r206 tip**, `efaa60db0` (`release.json.tip`), *in numeric order*:

```bash
# 1. a checkout at the r206 tip: either the fork's rdna-boosts branch, or the delivery set applied
git clone git@github.com:stew675/llama.cpp.git && cd llama.cpp && git checkout rdna-boosts   # or:
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   git checkout "$(jq -r .base <delivery>/release.json)" && bash <delivery>/scripts/apply-all.sh .

# 2. the campaign patches (strict git am — never `git apply` a concatenated series)
git am <delivery>/wip/host-pool-speed/patches/0001-moe-cache-report-the-RUN-not-an-emptied-cache-snapsh.patch \
       <delivery>/wip/host-pool-speed/patches/0002-host-pool-full-coverage-default-MOE_HOST_POOL_FRAC-1.patch \
       <delivery>/wip/host-pool-speed/patches/0003-host-pool-the-device-gather-resolves-arena-POOL-boun.patch

# 3. build
~/bin/build-llama-rocm          # or: cmake --build build-rocm --target llama-cli -j 16
```

(All six in numeric order: `git am .../000[123456]-*.patch`.  On the delivery tip `efaa60db0` the six give tree
**`07660557c05bf0a6c462212c4746c5bc6ce40afe`** (verified 2026-11-03 in a fresh worktree).  Compare TREES, not
commit ids: `0001`-`0006` are campaign artifacts, not the deterministic `apply-all.sh` flow, so an ad-hoc `git am`
stamps its own committer dates/ids.)

**Validated 2026-11-02 (session 2):** a fresh worktree at `efaa60db0` takes both patches with `git am` (no fuzz,
no 3-way) and produces tree **`0b8cc5a8bd7477747723faa45c6de38992877f3a`**, identical to the fork's local branch
`wip/host-pool-speed-pool-sizing` tip.  (That branch is local only — never pushed; the patches are the artifact.)

**Re-based and extended 2026-11-02 (session 3, the 1-GPU host).**  The "re-base onto `efaa60db0`" was a no-op:
`0b8cc5a8b` is the *applied tree*, and the patches' pre-image blob at `efaa60db0` (`334931ecd`) is the patch
`index` line, so 0001/0002 apply with no conflict (the window campaign's work exists only in its archived
reference diff, never in this tree).  With `0003` added, a fresh worktree at `efaa60db0` + `git am` on all three
gives applied tree **`ac580a8e3a6f12ccb5f673dace82b16ec4f1f275`** (`0004` added 2026-11-02, session 3); per-patch
build is 4/4 clean and same-seed greedy text is `sha=4ac3a523160f`.  See `../README.md` §14.

## Knobs this set introduces or changes

| variable | default after 0002 | class | meaning |
|---|---|---|---|
| `MOE_HOST_POOL_FRAC` | **100** (was 80) | tuning | pool size as a % of each host expert tensor's bytes; still clamped by `MOE_HOST_POOL_RAM_PCT` (85 % of `MemAvailable`) |
| `MOE_HOST_POOL_SIZING` | **0** (uniform) | opt-in | `1` = per-table slots from routing concentration instead of a flat FRAC |
| `MOE_HOST_POOL_COV_PCT` | 90 | tuning | the mass fraction `cov_t` targets when `MOE_HOST_POOL_SIZING=1` |
| `GGML_SCHED_DEVGATHER` | **unset = pool-backed tables only** (was: off) | tuning / A/B (`0006`) | three-valued: **unset** = a `--host-experts pool` table gathers in the prefill band and everything else keeps staging; **`0`** = kill switch, the pre-`0006` behaviour (whole-table staging); **`1`/`force`** = the historical A/B arm (gather below the staging width gate and for large tables, in either host-expert mode) — not the default, because it costs a pinned master ~47 % prefill at two devices.  **At promotion this row replaces `ENVIRONMENT.md`'s "gather default-off" text.** |
| `MOE_HOST_POOL_RAM_PCT` | 85 (unchanged) | tuning | **behaviour changed by `0005`**: the ceiling now scales every table's slots instead of starving the last-registered ones, and `SIZING=1` fits its plan to the ceiling.  At promotion, `ENVIRONMENT.md`'s row needs that sentence — its current text describes the one-slot-floor tail. |

Everything else (`MOE_HOST_POOL_MIB`, `_RAM_PCT`, `_DIO`, `_COLD`, `_PREFETCH`, `_PREWARM`, `_STAGE_BATCH`,
`_VERIFY`, `MOE_EXPERT_CACHE_*`) is unchanged.  **At promotion** these three rows must be added to
`ENVIRONMENT.md`, and the per-block note in `patches/README.md` (block 12) updated — that is deliberate: nothing
here touches the delivery docs yet.

## What it is expected to do (measured, 1 × R9700, 122B-A10B, `-ncmoe 48`, 2000-token decodes)

| arm | decode t/s | prefill t/s | disk read / run | pool hit |
|---|---|---|---|---|
| `pinned` | 22.38-22.39 | 677-689 | 62-65 GiB | — |
| `pool` default (FRAC 100) | **20.62** | 613 | 112 GiB | 0.9169 |
| `pool` FRAC 80 uniform (pre-0002 behaviour) | 17.80 | 637 | 154 GiB | 0.6200 |
| `pool` FRAC 80 **`MOE_HOST_POOL_SIZING=1`** | **19.14** | 554 | 131 GiB | 0.6785 |
| `pool` FRAC 60 uniform (sub-full) | 13.60 | 443 | 272 GiB | 0.3405 |
| `pool` FRAC 60 **`MOE_HOST_POOL_SIZING=1`** | **15.83** | 377 | 210 GiB | 0.4416 |

So: full coverage turns half of the pool's old −20.5 % decode gap into a **−7.9 % raw-overhead** number, and `cov90`
buys **+7.5 %** (FRAC 80) / **+16.4 %** (FRAC 60) at *identical* pinned bytes.  **Costs and caveats, read before
quoting any of it:**

- **`cov90` costs prefill** (−13 % at FRAC 80, −15 % at FRAC 60): the pool is ranked for decode, and a prefill
  ubatch routes nearly uniformly.  Net wall-clock win for a 2000-token request; not yet isolated.
- **The full-coverage default is validated on one host-expert device only.**  On the 2-GPU `-sm tensor` pair the
  same arm over-commits (67.1 GiB pool + a 73 GiB model on 96 GiB) and **stalled once** in the pool prepopulate
  under page-cache pressure, and `pinned` on that shape gets no expert cache at all
  (`../HANDOVER.md` §4/F7).  Keep `MOE_HOST_POOL_FRAC=80` (or lower) on a multi-device tensor split until that is
  re-measured.
- All numbers are the campaign's reference arm on the maintainer's box; the protocol (pins, `-lv 4`, alternation,
  ≥2000-token decodes, `-b/-ub 2048`) is in `../README.md` §9.  Gates run: greedy text byte-identical across
  pinned / FRAC 60 uniform / FRAC 60 cov90 / FRAC 100 (`sha=cacea85d5705`), `MOE_HOST_POOL_VERIFY=1` 0
  mismatches, `scripts/check-gpu-faults.sh` clean, no swap growth.

## Dropping the set

`git reset --hard efaa60db0` in the scratch checkout (nothing else depends on it), or `git checkout rdna-boosts`
in the fork and rebuild.  The fork's `build-rocm` is currently built **from this set** — rebuild after switching
back to `rdna-boosts`, and note that on the patched build `MOE_HOST_POOL_FRAC` defaults to 100, so re-measuring the
old default needs `MOE_HOST_POOL_FRAC=80` explicitly.
