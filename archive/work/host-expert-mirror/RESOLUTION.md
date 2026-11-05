# RESOLUTION — `wip/host-expert-mirror` closed 2026-11-05 (no release, no delivery change)

**Status: CLOSED as CAPPED — the campaign's premise was refuted, its target was re-measured and moved, and
nothing of it belongs in `patches/`.**  The delivery is unchanged at `v16-6050b467c-r221`.  The record is this
directory: [`README.md`](README.md) (the original scoping note, kept for the reasoning), [`spike-261105/`](spike-261105/SPIKE.md)
(step 1: which copy runs, its share, its rate, the byte accounting) and
[`LEAD-pool-staging-gap.md`](LEAD-pool-staging-gap.md) (the three-transport comparison and its mechanism).

## Claim

The campaign was opened to close the last ~8 % of prefill against `briansp2020`'s branch (1802 vs 1949 t/s,
`-ub 6144`, one R9700, a mirror host-expert tier) by **handing a covering pool's op the pool's device alias**
instead of staging a copy — i.e. an in-place expert read in the prefill band (`MOE_HOST_POOL_ALIAS`).

**Refuted as a lever, and superseded by a cheaper one.**  On a mirror the prefill transport is not the staging
engine at all: it is the **device gather** (`GGML_SCHED_DEVGATHER` mode 1 routes every pool-backed table above
the cache band to the gather, so `GGML_MOE_CACHE_STAGE` and `moe_cache_upload_pageable` never run — measured: the
staging census is absent and `device gather: N calls (N pool-backed)` covers every table).  That gather reads the
pool's pinned alias **in place, at ~33 GB/s** — above this stack's 28.6 GB/s bulk-H2D calibration — so the bytes
are already moved at the link's rate and an alias cannot move them faster.  What the gather costs is its
**stream**: it is a node in the compute graph, so its time is added to the prefill instead of hidden under it.

## Rule

1. **On a mirror, measure the transport, not the copy function.**  A prefill A/B of an expert transport must
   read the exit census first (`device gather:` vs `prefill cache staging:`) — "no gain" and "not engaged" look
   identical otherwise.  A staging-byte figure from `GGML_MOE_CACHE_STAGE`-era records does not describe the
   shipped default path.
2. **An in-place expert read is bounded by the link, so it can only ever buy overlap.**  The gather measured
   34.9 % of prefill GPU time at `-ub 6144` (55 % at `-ub 2048`) at the link's full rate; the whole prize would
   be hiding that inside compute, and the comparable in-place implementation on this hardware
   (`briansp2020`'s branch) measures +8 %.
3. **The same byte copy is 12-30 % faster or slower depending on which engine and stream carry it**, and the
   ordering inverts with the shape (`LEAD-pool-staging-gap.md` §1-§2).  Never conclude "transport X is faster"
   from one device count or one ubatch width.
4. **A block-ordered set of "reps" is not a measurement on this box.**  A 16 % monotonic drift (and a 36 % swing
   on the pinned tier) appeared across block-ordered arms while the interleaved control moved ±0.3 %; only
   interleaved ABAB is trustworthy (`spike-261105/abba.sh`).  This extends `INVARIANTS.md` r221 fact 7.

## Numbers (all 2026-11-05, r221 tree, 1 x R9700 unless stated, 52 512-token prompt, greedy, no cgroup)

* **The transport on a mirror is the gather**: 1410 calls, all pool-backed, **0** experts served from the
  neither-tier path; `GGML_MOE_CACHE_STAGE=0` is a no-op (1616.2 vs 1618.8 t/s).
* **Its share and rate**: `moe_cache_gather_kernel` 10.91 s of a 32.74 s prefill (34.9 %; 55.0 % at `-ub
  2048`), 329.19 GiB pool + 126.50 GiB arena per pass, 250.7 MB of pool reads per call in 7.52 ms = **33 GB/s**.
  GPU-busy 0.95x of wall, i.e. **serialised**.
* **Transport ranking, interleaved** (gather / r114 single-launch staging kernel / per-expert copy-engine
  staging): `-ub 2048` 1016-1021 / 872-875 / 742-907; `-ub 5120` 1511 / - / **1694**; `-ub 6144` 1627-1629 /
  1412-1425 / **1840-1845**; `-ub 7168` 1685 / - / **1889**; `-ub 8192` **1748** / 1546 / 1696; `-ub 16384`
  **1945** / 1736 / 1788; 3 GPU `-sm tensor` `-ub 6144` **2359** / 2075 / 2053.
* **The mechanism, traced** (identical dispatch counts): the single-launch staging kernel monopolises the CUs
  and starves the compute kernels — `mm_ids_helper_512_10` 41x, `quantize_mmq_q8_1` 4.9x,
  `mul_mat_q_routed_compact<20>` 3.4x slower — for a total of 43.55 s of GPU time against 31.22 s (gather) and
  27.57 s (copy engine).  Its r114 rationale (+34 %) was real only against a *pageable* master that paid a host
  memcpy into a pinned bounce per copy; with a readable pool alias there is nothing to amortise.
* **Purity of the transport change**: same-seed greedy text byte-identical across all three transports
  (sha256 `68704e68…`, 135 chars).
* **Levers that are not the answer**: the `-ub` width (bytes/token ∝ 1/ub while every expert routes: 1019 ->
  1942 t/s from `-ub 2048` to `16384`, at the cost of arena, 21.5 -> 16.4 GiB); a bigger arena alone (`-c 65536`:
  pool 329 -> 312 GiB, +1.2 %, because only ~28 % of routed offers are served from the arena — the slots fill
  on demand); `MOE_HOST_POOL_ALLOC` shape and `MOE_HOST_POOL_H2D` (no effect on either arm); the pinned tier's
  own transport (1820-1825 t/s, i.e. the same as the pool's best, for a 46.84 GiB duplicate).

## What would reopen it

1. **A transport-selection rule that is device-count and width shaped** (`TODO` #272): the copy-engine arm wins
   only at 1 GPU / `-ub` 5120-7168 and *stalls* by 8192 (kernel/wall 0.80, 20 % of the wall idle with no kernel
   running) while the gather wins on a tensor split by 14 %.  Naming that stall (ring depth
   `GGML_SCHED_STAGE_SLOTS` vs the split count is the suspect) could turn a band into a rule — that is the only
   route from this record to a delivery change.
2. **The pinned tier's 54 declined tables** (`TODO` #271): `prefill cache staging: ok=783 unusable=477` — 54 of
   144 tables never enter the staging kernel on a pinned master and are served by compute-stream per-tensor
   copies, with no log line naming them.  That is a silent prefill cost on a tier `-lme auto` can select.
3. **A second device count's mirror rows** (2 GPU `-sm tensor`, and `-sm layer` at `-ncmoe`), which the
   `bench-56gb` rig still owes; and the logit gate A/B
   (`scripts/gate-prefill-logits.sh --ab "GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0"`) before any of
   this can be called a shippable transport choice.
4. **A hardware or ROCm change** that alters the copy-engine/kernel-read asymmetry (the whole ranking is one
   stack's behaviour: ROCm 10.1.0 / gfx1201).

## Not promoted, and why

* No `patches/` change: the only measurable win is a **two-env-var configuration**
  (`GGML_SCHED_DEVGATHER=0 MOE_HOST_POOL_STAGE_BATCH=0`) on one shape, and both switches are already documented
  kill switches.  Making it a default would repeat r114's mistake in the other direction — one cell, one width,
  one device count.
* No `optional/` artifact: nothing was built, so there is no patch to carry.
* No `INVARIANTS.md` fact was "owned" by this campaign and contradicted — the facts it learned were appended
  there as its record (the transport ranking, the starvation mechanism, the ABBA obligation).
