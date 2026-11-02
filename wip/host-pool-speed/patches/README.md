# `wip/host-pool-speed/patches/` — the campaign's patch set (NOT the delivery)

**These are not part of the delivery.** They are not in the repo's top-level `patches/`, they are not applied by
`scripts/apply-all.sh`, and `release.json` does not know about them.  They exist so the experiment can be rebuilt
elsewhere (a review, another machine, the head-to-head in [`../HEAD-TO-HEAD.md`](../HEAD-TO-HEAD.md)).

## The set

| patch | what it does | files |
|---|---|---|
| `0001-moe-cache-report-the-RUN-*.patch` | **Diagnostics.** `moe_cache_report()` is an `atexit` hook that runs after `moe_cache_forget_all` has un-sized the cache, so every run ended with `tables=0, slots=0, arena=0.0 MiB` and `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned` — in *both* host-expert modes.  That line is what made “the pool never activated on the 122B” look like a delivery bug (`../HANDOVER.md` §4/F3).  The patch snapshots the report's structural aggregates at the first forget and has the report prefer them.  It also makes a forced `MOE_HOST_POOL_DIO=0\|1` log its effective mode: the *“(forced by MOE_HOST_POOL_DIO=N)”* branch was unreachable dead code because the env parse pre-set `g_pool_dio_decided`. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |
| `0002-host-pool-full-coverage-default-*.patch` | **The tuning changes.** (a) `MOE_HOST_POOL_FRAC` now defaults to **100** (full coverage of each host expert tensor), still clamped by `MOE_HOST_POOL_RAM_PCT` of `MemAvailable`.  (b) New **`MOE_HOST_POOL_SIZING=1`**: per-table pool slots from routing *concentration* (`cov_t` = the slots a table needs to hold `MOE_HOST_POOL_COV_PCT` of its own prior mass), fitted one-shot to the same total byte budget at the first pool build.  (c) The `--host-experts pool` help/log text that named the old 80 % default. | `ggml/src/ggml-cuda/moe-expert-cache.cu`, `common/arg.cpp`, `src/llama-model-loader.cpp` |
| `h2d-experiment.patch` | **Not part of the set — a finished experiment, kept for the record.** `MOE_HOST_POOL_H2D=1` switches a pool hit/fill from the historical `cudaMemcpyDeviceToDevice` (from the pool's mapped UVA alias) to `cudaMemcpyHostToDevice` (from its pinned host pointer).  Measured: 17.79/17.78 vs 17.76/17.80 t/s — **no effect**, so the copy kind is excluded as an explanation of the pool's decode deficit (`../HANDOVER.md` §3.1).  Do not apply it unless you want to re-run that A/B. | `ggml/src/ggml-cuda/moe-expert-cache.cu` |

## Prerequisites and apply order

The set applies on the **delivery's r206 tip**, `efaa60db0` (`release.json.tip`), *in numeric order*:

```bash
# 1. a checkout at the r206 tip: either the fork's rdna-boosts branch, or the delivery set applied
git clone git@github.com:stew675/llama.cpp.git && cd llama.cpp && git checkout rdna-boosts   # or:
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   git checkout "$(jq -r .base <delivery>/release.json)" && bash <delivery>/scripts/apply-all.sh .

# 2. the campaign patches (strict git am — never `git apply` a concatenated series)
git am <delivery>/wip/host-pool-speed/patches/0001-moe-cache-report-the-RUN-not-an-emptied-cache-snapsh.patch \
       <delivery>/wip/host-pool-speed/patches/0002-host-pool-full-coverage-default-MOE_HOST_POOL_FRAC-1.patch

# 3. build
~/bin/build-llama-rocm          # or: cmake --build build-rocm --target llama-cli -j 16
```

**Validated 2026-11-02:** a fresh worktree at `efaa60db0` takes both patches with `git am` (no fuzz, no 3-way) and
produces tree **`0b8cc5a8bd7477747723faa45c6de38992877f3a`**, identical to the fork's local branch
`wip/host-pool-speed-pool-sizing` tip.  (That branch is local only — never pushed; the patches are the artifact.)

## Knobs this set introduces or changes

| variable | default after 0002 | class | meaning |
|---|---|---|---|
| `MOE_HOST_POOL_FRAC` | **100** (was 80) | tuning | pool size as a % of each host expert tensor's bytes; still clamped by `MOE_HOST_POOL_RAM_PCT` (85 % of `MemAvailable`) |
| `MOE_HOST_POOL_SIZING` | **0** (uniform) | opt-in | `1` = per-table slots from routing concentration instead of a flat FRAC |
| `MOE_HOST_POOL_COV_PCT` | 90 | tuning | the mass fraction `cov_t` targets when `MOE_HOST_POOL_SIZING=1` |

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
