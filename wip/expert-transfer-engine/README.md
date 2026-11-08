# expert-transfer-engine — ONE prefill transport for every shape (OPEN, scoping + a parked experiment)

**Status: OPEN — decision pending on which of the two live transports to consolidate onto.  Nothing in
`patches/`; the delivery is unchanged at `v16-6050b467c-r234`.  The experiment that produced the numbers below
is parked as [`0001-no-pinned-staging.patch`](0001-no-pinned-staging.patch) (never applied to the fork).**

## 1. The problem, in the maintainer's words

> "Ideally we'd want just one system, one way of handling the transfers, rather than the 3 we have right now.
> It seems like we still don't have that single best one for all scenarios … trying to get away from this
> ad-hoc confusion of choose this engine for this scenario, and that one for that special case, and a third
> one for the general case, all with their own kill switches."

Today a host-resident expert table's prefill upload is served by **three** transports, selected by two
environment variables and a tier test:

| # | transport | selected by | reaches |
|---|---|---|---|
| 1 | **device gather** (`moe_cache_gather_kernel`) — reads only the *routed* experts, in place from the arena/pool | default for a POOL-backed master above the cache band | pool tier |
| 2 | **staged fill** (`moe_cache_stage_table`) — per-expert copies into a ring slot, then a D2D into the split input | default for a PINNED master (`GGML_MOE_CACHE_STAGE=1`), and for a pool master with `GGML_SCHED_DEVGATHER=0` | both tiers (`MOE_HOST_POOL_STAGE_BATCH` used to pick its engine; r234 deleted the one-launch kernel, so it is per-expert copies only) |
| 3 | **bulk upload** — the scheduler's own single `cudaMemcpyAsync` per table | `GGML_MOE_CACHE_STAGE=0` ("kill switch") | both tiers (on a pool tier it degenerates to the per-expert pinned **bounce** path, 5-10x slower) |

The r88/r114/r224 lineage added (2) and its switches; r207/r221 added (1); (3) is what the scheduler does when
neither takes the table.  Each has a kill switch and each wins somewhere.

## 2. The measurements that make this a decision (2026-11-07, r234 tree + the parked patch)

Config: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, 52 512-token prompt, `-n 1`, `-ub 6144`, greedy, `-sm layer` unless
stated, R9700s on Gen5 **x8** slots, balanced ABBA (4 runs/arm unless noted).  `[ Prompt: … ]` footer as the
figure.  Raw logs: [`../../archive/work/host-expert-transport/runs/`](../../archive/work/host-expert-transport/runs/)
(`res43-2dev`, `np-*`, `nopinned*`, `x-c*`, `p*`, `ml*`, `off*`, `thp*`, `sdmaoff*`, `tr*`, `span-*`, `text/`).

### 2a. Staged vs bulk against arena residency (2 devices, layer split, arena capped via `MOE_EXPERT_CACHE_MIB`)

| residency | staged (per-expert copies) | bulk (no staging) | winner |
|---|---|---|---|
| 44 % | 1704-2070 (bimodal) | **1929-1969** | **bulk +16 %** |
| 67 % | 1774-2093 | 1825-1969 | **bulk +6 %** |
| 83 % | 1858-2093 | 1823-1967 | staged +2..10 % |
| 99 % | 1957-2088 | 1821-1964 | staged +4 % (tie-ish) |

With the **retired one-launch kernel** as the staged engine the same cells read 1665 / 1854 / 2007 / 1960 — i.e.
the staged arm's *mean* rises with residency while the bulk arm is flat (~1900) at every residency, because bulk
always moves 100 % of the table over PCIe and staging moves `1 − residency`.

### 2b. 1 GPU and tensor split

| cell | staged | bulk | winner |
|---|---|---|---|
| 1 GPU, layer, 43 % res, `-ub 6144` | 1761 / 1876 (kernel, low/high phase); 1624 / 2147 (per-expert copies) | **1866 / 2180** | **bulk, both phases** |
| 2 dev, tensor, 99 % res | **2375** (kernel), 2372 (copies) | 2265 | **staged +4.6 %** |

### 2c. The pool tier is a separate case

There the staged fill is **mandatory**: without it a fill falls to `moe_cache_upload_pageable`'s per-expert
pinned bounce and measures **103-281 t/s against 1052-1556** (5-10x).  The gather is the default transport on
that tier and wins at `-ub` ≤ 4096 (+12 % at 4096, +20 % at 2048, vs the copy-engine staging arm); the r234
"copy engine wins from 5120" did **not** reproduce in the next session (gather +5.3 % at 5120).

### 2d. The box's run-to-run phase (a measurement hazard, not a transport property)

Identical work alternates between two levels with period 2: 1873/2177/1865/2180 t/s on one unchanged arm.
Amplitude is a property of the transport's **DMA op count**: gather ±0.1 %, one-launch kernel 6.5 %, one bulk
DMA per table 17 %, 512 per-expert copies 32 %.  Traced: the slow phase is 67 % GPU-busy against 85 %, with the
copy engine *less* busy — the copies stall rather than transfer; `HSA_ENABLE_SDMA=0` removes the oscillation
(stable, at the slow level).  So it is ROCm/SDMA dispatch latency.  **Ruled out**: PLE lazy-mode (`auto`/`off`/
`mlock` all oscillate identically), memory pressure, GPU clock, `GGML_SCHED_STAGE_SLOTS` (8 vs 16), the GPU slot.
Records: `WORKLOG.md` 261107.012, `INVARIANTS.md` (r234, second addendum).

## 3. The hole, stated plainly

* **bulk** is the simplest and the most stable, and it wins at ≤ 67 % residency (the shape any 24-32 GB card
  produces against a 40-48 GB host set).  It loses 2-10 % when the arena holds **≥ 83 %** of the set, because it
  re-fetches over PCIe the experts the arena already holds in VRAM.
* **staging** collects that saving (at 99 % residency it sends **0.0 %** of the bytes over PCIe) but only by
  moving the host part as **512 small transactions per table** (or, in the retired kernel, as GPU-generated
  loads), which is what costs it at low residency.
* The two are therefore **the same trade seen from both ends**: bytes saved vs transactions issued.  The resident
  set is scattered (heat-seeded), so the complement cannot be coalesced into one DMA — that is why "stage only
  what is required" is not a small change.

## 4. The two candidate directions (and what would have to be measured)

**Direction A — consolidate on BULK (the maintainer's leaning).**  One `cudaMemcpyAsync` per table.
* Needs: the high-residency hole closed *without* an in-place/remap read.  Two candidates, both cheap to test:
  1. **Decode-sized arena.**  The arena exists for the decode band; today it auto-sizes to eat all free VRAM,
     which is the *only* reason residency ever reaches ≥ 83 %.  Sizing it for decode (heat coverage) keeps
     residency ≤ ~50 %, where bulk is uniformly best, and hands the freed VRAM to the KV cache / a wider ubatch.
     Measure: decode hit-rate and tg t/s at a capped arena vs auto, plus the prefill cells above.
  2. **One bulk copy from the tier that holds the table contiguously.**  A *mirror* pool is expert-indexed and
     contiguous (`pool->base`/`base_dev`), so a pool-tier table could be one DMA too — deleting the per-expert
     copies, the arena D2D assembly, and the span read on that tier.  A sub-complement pool would keep the span
     read (it is the only coalescing source there).
* Risk: the 2-device tensor / 99 % residence cell (-4.6 %) — which (1) removes by construction.

**Direction B — consolidate on STAGING.**  Fix its low-residency loss: coalesce the complement into few DMAs.
* Blocked by the scattered resident set: runs average ~1.3 experts at 43 % residency, so coalescing buys almost
  nothing.  The only real fixes are the ones the archive already flagged and the maintainer does **not** want to
  re-enter lightly: let the GEMM read a resident expert **in place** (`mmq_routed_src`, "the seam exists and is
  dead; the caller was dropped in the r50 rebase") or hand the op the arena wholesale (`MOE_HOST_POOL_ALIAS`,
  refuted for the pool tier in `archive/work/host-expert-mirror/`).  Both are remap-shaped and big.

**Recommendation: Direction A (bulk).**  It is the smallest, the most stable, already wins where most consumers
live, and its one hole is closable by a *sizing* decision rather than by a new copy path — with one transport,
one switch (`GGML_MOE_CACHE_STAGE`, kept only as the A/B kill switch), and the gather left as the pool tier's
default (it is a decodable, measured win there and costs nothing to keep).

## 5. The parked experiment

[`0001-no-pinned-staging.patch`](0001-no-pinned-staging.patch) (against the r234 tree, tree
`ca208aed7763985904568ccd093cacafe70d995e`): the pinned master is never staged (bulk), the pool tier keeps the
per-expert copy-engine fill + the span read, and the one-launch kernel, the PRS device snapshot,
`MOE_HOST_POOL_STAGE_BATCH` and the `order_event` plumbing are deleted (**−328/+129 in three files**).
Measured: `gate-host-pool.sh` **41/41**, text byte-identical (`d3d554954cba` 1 GPU, `301b143c1cf5` 2-device
tensor), span read reproduced exactly (59 659 reads / 153 491.3 MiB / 2.57 MiB/read), 1 GPU/43 % **+6..16 %**,
2 dev/44 % **+16 %**, 2 dev tensor/99 % **-4.6 %**.  A binary built from it is kept as
`~/llama.cpp/build-rocm/bin-nostage` so the cells can be re-measured without a rebuild.

**It is not proposed for merge as-is**: it ships the one sided trade (regressing the ≥ 83 %-residency
multi-device cell by 2-4.6 %) without closing it.  Direction A's step 1 (a decode-sized arena) is what would
close it; until that is measured, this patch is evidence, not a delivery candidate.

## 6. Next steps, in order

1. Measure **decode** at a capped arena vs the auto arena (hit rate, fills, tg t/s) — does the arena's
   prefill-driven size buy anything for the band it exists for?  (Cheap: one `llama-bench`/`llama-cli -n 256`
   sweep at 2-3 arena caps.)
2. If decode is insensitive above ~50 % residency, re-run §2a's 44/67/83/99 % cells with the arena capped to the
   decode-optimal size and confirm bulk wins every cell.
3. Only then cut the delivery change (Direction A step 1 + the parked patch) as one block-12 amendment.
4. Keep the phase (SDMA dispatch) as a measurement protocol note; do not tune to it.
