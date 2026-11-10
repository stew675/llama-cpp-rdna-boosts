# expert-transfer-engine — ONE prefill transport for every shape (OPEN, scoping + a parked experiment)

**Status: OPEN — the transport question is now bounded, and the answer is not a transport.  Nothing in
`patches/`; the delivery is at `v16-6050b467c-r235` and unchanged by this campaign.  Two experiments are parked
here and **neither is applied to the fork**: [`0001-no-pinned-staging.patch`](0001-no-pinned-staging.patch)
(the bulk-direction merge, evidence not a candidate) and
[`0002-stage-copy-streams.patch`](0002-stage-copy-streams.patch) (a selectable fill-stream count — wins on
dispatch-bound fills, **loses 4 % on the field config**).  Read §7 first: it measures on a **gen5 x16** host, and
it says the prefill gap is bytes and first-completion latency, not the transport *engine* and not the number of
VRAM copies.**

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

## 2e. Where the ≥ 83 %-residency cell actually comes from — it is a MISCONFIGURATION, not a target

The 99 % row above is not a configuration anyone needs; it is an axis point I constructed (by capping the arena
at three other values) and it exists at all only because of how the arena is sized:

* `alloc_all_locked` sizes the arena from **what is free NOW** — after `--fit`, the compute reserve and the KV
  cache — minus `MOE_EXPERT_CACHE_RESERVE_MIB` (default 1024 MiB), capped at the host set's slot count.
* So residency ≈ `min(1, (free VRAM − reserve) / host set)`.  **High residency therefore means "there was lots
  of VRAM free"**, i.e. **the user offloaded more experts to the host than they needed to**.  The 99 % case was
  2 × 32 GB = 64 GB of VRAM with *every* layer's experts forced onto the host (`-ncmoe 48`, a 48 GB set): the
  obvious right answer there is `-ncmoe 0` (fully resident) or a much smaller `-ncmoe`.
* Conversely a model that **just** does not fit has almost no free VRAM after its weights → a small arena →
  **low residency → bulk wins**.  So the regimes where high residency appears are precisely the ones where the
  host tier should not have been used for that much of the set.

**The lever is not `-ncmoe`.**  Lowering `-ncmoe` keeps residency *high*: fewer host layers means a *smaller*
host set with the same free VRAM, so the arena still covers all of it (`min(1, free/host)` → 1).  What lowers
residency is (a) **capping the arena** (`MOE_EXPERT_CACHE_MIB`), or (b) raising `-ncmoe`, which is not a fix —
it just moves more bytes.  So the honest deliverable is a **sizing policy** (what the arena should be *for*),
not a transport rule.

That reframes step 1: the question is not "can we close the hole?" but **"does the arena earn its auto size at
all, given that the prefill cannot use it (below ~83 % residency) and only the decode band can?"** — a decode
times prefill trade in pure VRAM bytes, with the freed VRAM going to KV / a wider ubatch.

## 2f. `-ncmoe` DOES move residency — measured, and it reframes the guidance

I claimed it did not (arguing that a pinned layer removes the same bytes from the host set and from free
VRAM).  That is only true while `F ≥ H`; with a real context it is wrong.  Measured, 1 GPU, `-c 204800`,
`-sm layer`, `--host-experts pinned`, a `-n 1` load:

| `-ncmoe` | arena | host set | residency |
|---|---|---|---|
| 48 | 21 639.3 MiB | 47 962.5 | **45.1 %** |
| 46 | 19 925.6 | 46 162.5 | 43.2 % |
| 44 | 17 971.5 | 44 237.5 | 40.6 % |
| 42 | 16 115.1 | 42 312.5 | 38.1 % |
| 40 | 14 084.9 | 40 287.5 | **35.0 %** |

Each pinned layer removes ~950 MiB from the host set **and** ~950 MiB from free VRAM (the layer's VRAM copy),
and the arena was already the smaller quantity, so the ratio falls ~1.25 points per layer.  In the `F ≥ H`
regime (2 devices, `-c 4096`) residency instead stays at **100 %** for every `-ncmoe` from 48 down to 32,
because the arena just covers the shrinking host set.  So:

* `residency = F / H`, and a pinned layer subtracts the same bytes from both → residency **falls when F < H**,
  stays at 100 % while F ≥ H;
* "offload the minimum" therefore lands in **bulk's regime by construction** — the maintainer's lever works,
  and it is now measured rather than asserted.

## 2g. Maintainer proposal: permanently pin half the residency (a static tier)

For a > 80 %-residency config: pin a static fraction of the set in VRAM, keep the arena for the rest, so a
prefill never pays PCIe for the pinned part.  Evaluated against the transport, the answer depends on who reads
the pinned bytes:

* **Under bulk: no effect.**  Bulk copies 100 % of the host set per ubatch because the op reads one contiguous
  `input_cpy`; its PCIe bytes are `H` whatever is resident (that is why bulk measured flat, ~1900 t/s, in every
  residency cell).
* **Under staging: it *raises* the host traffic** if it merely converts adaptive residency into static
  residency — staging sends `1 − residency`, so "90 % adaptive" → "45 % static + 45 % dynamic" moves the local
  fraction from 90 % to 45 % (10 % → 55 % over PCIe).  The gain is **stability** (a static tier cannot be
  evicted mid-prefill, which is what stood 50 tables down in #271), not bytes.
* **The version that would actually win** pins a **contiguous** expert range rather than the hottest set: the
  non-resident complement is then one contiguous run in the master → **one DMA per table** (bulk-like
  transaction count) *and* a partial byte saving.  Viability depends on expert popularity being flat enough that
  index order ≈ heat order.  The r234 heat census is encouraging but not decisive: the in-run adaptive set held
  27.95 % of the routed mass while a static top-N placement of the same slots would hold **84.66 %**.
* **Decode cost**: at equal residency a (heat-seeded) static set loses ~1.8 points of hit rate to the adaptive
  arena (r74, `wip/moe-static-hot-tier/`), and the freeze fights the movable-boundary slab — "a frozen slot
  inside a lendable chunk is not frozen".  Both are the named hard parts of that (still unimplemented) campaign.

**So the static-tier idea is a separate stage, not a transport fix**, and it is only worth building in the
contiguous form.  If the arena-sizing step (§6.1) shows a large arena does not earn its bytes for decode, this
proposal is moot on the delivery's supported shapes.

## 2h. Corrected: bulk DOES benefit from VRAM residency — measured, and it names the leading candidate

My §2g claim ("under bulk it changes nothing") was true only for **duplicate** VRAM residency (the arena).  It
is wrong for **exclusive** VRAM residency, i.e. moving a layer's experts out of the host tier: bulk's prefill
traffic is exactly the host-set bytes per ubatch.  Measured with bulk itself (`GGML_MOE_CACHE_STAGE=0`), 1 GPU,
`-c 204800`, `-ub 6144`, 52 512-token prompt:

| `-ncmoe` | host set | bulk prefill |
|---|---|---|
| 48 | 47 962.5 MiB | 1872.0 t/s |
| 44 | 44 237.5 | 1912.8 (+2.2 %) |
| 40 | 40 287.5 | 1950.6 (+4.2 %) |
| 36 | 36 887.5 | **1989.1 (+6.3 %)** |

So the hierarchy is:

1. **Host-set size is the primary lever** — and it is *transport-independent*: every transport's PCIe bytes
   scale with it (~1 % of prefill per layer moved to VRAM, measured on bulk).
2. **Arena residency (duplicate VRAM) is secondary** — only a transport that *reads* the duplicate can use it
   (staged: `1 − residency` of the bytes; in-place: none).  Under bulk it is a decode cache and nothing else.
3. **Transaction count** is the third axis, and it is what the SDMA phase punishes (§2d).

### The candidate that satisfies all three: a contiguous static tier + a two-transaction fill

If the VRAM-resident set is a **contiguous expert range**, a table's prefill is:

1. **one bulk DMA** of the contiguous *complement* from the host master into `input_cpy` — bytes = the
   complement only (the saving), one transaction (bulk's transaction count, no SDMA exposure);
2. **one D2D** (or one small kernel) overlaying that contiguous range from the static tier.

At 0 % residency it degenerates to plain bulk; at ~100 % to pure D2D.  It needs no per-expert copies, no
512-transaction fill, no PRS map and no remap into the GEMM.  The resident range should be picked by a
**heat-maximising sliding window** over the expert axis (cheap: the prior map is already loaded), so decode
keeps close to a heat-optimal set while the complement stays contiguous.

**What it costs, and the number to measure**: a heat-seeded *static* set already loses ~1.8 points of hit rate
to the adaptive arena at equal residency (r74, `wip/moe-static-hot-tier/`); an index-contiguous set will lose
some more.  Plus that campaign's recorded hard part — a frozen slot inside the slab's lendable chunk is not
frozen — has to be solved for the static tier to exist at all.

This is now the campaign's leading candidate for "one transport for every shape"; the arena-sizing step (§6.1)
is still worth doing first because it is nearly free, and it tells us how much residency there even is to
exploit on the delivery's supported shapes.

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

**Recommendation: Direction A (bulk).**  It is the smallest, the most stable, already wins at ≤ 67 % residency
on 1 **and** 2 devices, and — since the ≥ 83 % cells are misconfigurations (§2e) — its one hole is not a shape a
consumer needs: a host expert set that the arena covers is a set that should have stayed resident.  What is left
is a **sizing policy** for the arena (decode hits vs prefill bytes), not a transport rule.  One transport, one
switch (`GGML_MOE_CACHE_STAGE`, A/B only), and the gather left as the pool tier's default (a measured win there
at `-ub` ≤ 4096, and free to keep).

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

1. Measure **decode** at a capped arena vs the auto arena (tg t/s, hit rate, fills, `h=`) at 2-3
   `MOE_EXPERT_CACHE_MIB` values — the real question of §2e: does the auto size earn its VRAM for the band the
   arena exists for?  (One model, `-n 256`, 2 devices, interleaved.)
2. If decode is flat above ~50 % residency, adopt a decode-sized arena default and confirm that residency then
   lands ≤ ~67 % in every realistic shape — at which point **bulk wins every configuration the delivery
   supports** and the ≥ 83 % rows are unreachable by construction.
3. Then cut the delivery change (the parked patch + the sizing default) as one block-12 amendment, and delete
   `MOE_HOST_POOL_STAGE_BATCH`.
4. Keep the phase (SDMA dispatch) as a measurement protocol note; do not tune to it.
5. Only if step 1 shows decode genuinely needs the big arena: the trade is real, and it is a workload policy
   (prefill bytes vs decode hits) — to be decided by the maintainer per shape, not defaulted from one cell.
6. **Separate stage — a static tier, contiguous form only** (§2g): pin experts `0..N` in VRAM outside the
   movable boundary, keep the arena for the complement, and measure (a) decode hit rate vs the adaptive arena at
   equal bytes, (b) the prefill with the complement as one DMA.  Read `wip/moe-static-hot-tier/` first — the
   freeze-vs-slab problem and the r74 -1.8-point static penalty are that campaign's recorded results.


## 7. Measured 2026-11-07 on a gen5 x16 host: the gap is not transport (and not the copies)

Artifacts: [`h2d-probe.cpp`](h2d-probe.cpp) (bare `hipcc`, no model: host-kind sweep, granularity x stream count,
pull kernel, interference, time-to-first-completion), [`heat-run-layout.py`](heat-run-layout.py) (offline
run-layout analyser over the persisted `MHT2` heat maps), the pipeline arms below (a `llama-cli` harness,
`-ncmoe 40`/`46`, one R9700).

### 7.1 The box differs from every number in §§2-4, so the crossover must be re-derived

The campaign's transport and residency numbers come from PCIe5 **x8** (`soar` x4 = 14.45 GB/s, `fingon` gen4 x16
= 28.6 GB/s).  This session's host reports **16 lanes @ 32 GT/s** at the endpoint and the delivery's own
calibration agrees: `H2D bandwidth calibration: 51.7-52.1 GB/s`, which collapses the staging width gate to
`min_tokens = 64`.  Measured directly: **55.8 GB/s** for one 333 MiB pinned H2D.  At 2x the link, bulk's PCIe cost
per ubatch is half what §§2-4 assume, so any "which transport wins" conclusion has to be re-measured here (or
labelled an x8 number).

### 7.2 Copy-engine mechanics (`h2d-probe.cpp`)

| shape | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| one 333 MiB copy, pinned | 55.8 GB/s | - | - |
| one 333 MiB copy, registered mmap / write-combined / pageable | 55.8 / 55.8 / 55.4 | - | - |
| **290 per-expert copies** (0.65 MiB each, the 43 %-resident complement) | **41.1** | **54.8** | 54.9 |
| 125 run-coalesced copies | 48.2 | 53.9 | 53.7 |
| vectorised **pull kernel** (int4, 1 CTA/run) | 50.7 | - | - |
| 8 x 343 MiB copies (one per table) | 55.8 | 55.8 | 55.8 |

* **Allocation kind does not matter for async copies** (all within 1 %).  But a **synchronous `hipMemcpy` out of a
  `hipHostMalloc` allocation is 27 % slower** (40.7 vs 55.8 GB/s) while sync-from-malloc/registered/WC are full
  speed — a benchmarking trap, and the delivery's `stage_h2d_gbps()` times a synchronous copy (it uses `malloc`,
  i.e. the fast path, so the shipped number is honest; change it to pinned and the width gate silently moves).
* **Run-coalescing the complement is a free 1.17x** (41.1 -> 48.2 GB/s) at zero mass cost — the complement has
  ~2.3-expert runs, not the ~1.3 §4 assumed.
* **A second stream is the big lever for dispatch-bound fills** (+33 % on per-expert copies: 41.1 -> 54.8 GB/s,
  i.e. back to single-big-copy speed; 4 streams add nothing).
* **A kernel that pulls is not faster than the copy engine on this link** (50.7 vs 55.8 GB/s), and it showed no
  interference with a concurrent compute kernel in isolation (+0.3 %).  The r221 record's "33 GB/s kernel read vs
  28.6 GB/s DMA" advantage does not reproduce on a fast link.

### 7.3 The rule: one stream per consumer chain — extra streams delay the FIRST completion

8 x 343 MiB copies, same total bytes every arm:

| streams | **time to first copy** | all copies |
|---|---|---|
| 1 | **6.26 ms** | 49.95 ms |
| 2 | **10.94 ms** (+75 %) | 49.99 ms |
| 4 | **10.93 ms** (+75 %) | 49.98 ms |

Aggregate bandwidth is invariant (link-bound), but the copy the consumer waits on arrives much later when the link
is shared with a copy it does not need yet.  So: **extra streams pay only when a fill is DISPATCH-bound; they cost
you first-completion latency when it is LINK-bound.**  That single rule explains every number in 7.2 and 7.4.

### 7.4 What it does end-to-end

| arm | 1 stream | 2 streams | verdict |
|---|---|---|---|
| 35B-A3B, pool mirror, 43 % arena, per-expert copy-engine fill, `-ub 8192` | 3551 | **4702** | **+32 %** (dispatch-bound) |
| 35B-A3B, same, `-ub 2048` | 1647 | 2060 | +25 %, still 45 % behind the gather (3728) |
| 35B-A3B, pinned master, staging kernel | 5090 | 5096 | flat (one launch per table) |
| 35B-A3B, pinned master, bulk | 5219 | 5238 | flat (one copy per table) |
| 35B-A3B, pool gather (shipped default) | 4931 | 4957 | flat |
| **Flash-Next, the maintainer's `runme` shape** (pageable master, bulk, 48.2 % residency) | **1538** | **1475** | **-4.1 %** |
| Flash-Next, same master, default staging | 1449 | 1380 | -4.7 % |
| Flash-Next, pinned master + bulk, `-ub 2048` / `-ub 8192` | 1380 / 1495 | 1325 / 1436 | -4.0 % both |

Text is byte-identical in every arm and stream count (`262 chars sha=e9db5cb8db0b`), and the staged bytes are
identical (`ok=117; arena=7733.9 MiB host=10430.1 MiB (H2D 57.4 %)`), so the delta is pure transport.
**The global stream knob is therefore NOT promotable**: it wins by 32 % on a config whose fill is dispatch-bound
and loses 4 % on the field config, where the fill is link-bound.  A scoped version (second stream only inside the
many-small-copy fill loop, whole-table bulk and the staging kernel left on `copy_stream()`) is the only defensible
form, and it is not worth a block on a 55 GB/s link.

**Also found:** `--load-mode none` silently **disables** the host-expert pool (`a pool needs a mapped load`), so
a `runme` built on it runs the *pageable bulk* path its own comments do not describe.

### 7.5 The three VRAM copies, and why unifying them does not buy arena

For a host-resident table at prefill, VRAM holds three copies, and each is charged to a **different budget pool**
(Flash-Next, `--n-cpu-moe 46`, `-c 32768`, 23.69 GiB slab):

| copy | pool | size |
|---|---|---|
| arena slots (duplicates of the resident `r x H`) | the arena region, sized at init from `free - reserve - aux` | 22.06 GiB (48.2 % residency) |
| ring slot pool (`GGML_CUDA_SLAB_RING_MIB`, auto = slots x largest table, capped 6144) | a work sub-region of the slab | `ring_req` 4-6 GiB |
| `input_cpy` (the consumer's real `src0`) | the work region (narrow floor 1.62 GiB) | ~1.0 GiB (3 tables of a layer) |

Measured (`GGML_CUDA_SLAB_RING_MIB` 6144 -> 2560, two runs each): ring 6.00 -> 2.50 GiB, **arena unchanged at
22269.8 MiB (48.2 %)**, prefill **1490 -> 1324 (-11 %)**, decode **48.3 -> 49.6 (+2.7 %)**.  So shrinking the
transport footprint **strands** VRAM instead of converting it into residency (the arena is sized once; `--fit off`
never re-derives it), and the ring's VRAM is buying prefill, not idle.  Converting freed VRAM into arena needs the
arena *sizing policy* (the §6.1 question), not a structural merge.

### 7.6 A unified expert store — DECLINED at r252 (`TODO` #285, retired; kept as the arithmetic)

Representing **every** expert is cheap: 138 x 512 = 70 656 experts, 4 B residency id each = **276 KiB**; with
{VA offset, host offset} per expert, **1.08 MiB**; VMM range entries per expert, the same; the ids remap
(`remap_dev`) 276 KiB.  **~1.5 MiB all-in**, 0.005 % of the card.  What costs is the *backing* (46.2 GiB to hold
them all), so the only viable form is a **VA reservation with VRAM backing for a resident subset** — block 15's
KV-window pattern at expert granularity.

What it would fix *structurally*: the arena's stand-down after a slab chunk reclamation makes `arena == nullptr &&
slots == 0`, i.e. **the whole table leaves the staging path** (`TODO` #271: 50 of 144 tables), whereas a page table
would flip the evicted experts to host-backed and keep the table addressable.  It also collapses the three
allocations into one (the fill lands in the table's own VA, the op reads `table + e*stride`, so the ids remap
disappears) and replaces three sizing rules with one.

What it does **not** fix: the bytes.  Every prefill ubatch routes essentially the whole table (measured: ~15.5 GB
of *used* experts per ubatch, independent of ubatch size), so a paged store has no cross-ubatch reuse at prefill
widths; and a host-backed page read in place is 33 GB/s serialised on the compute stream, which loses to the copy
engine's 55.8 GB/s overlapped.  It is therefore a complexity/robustness win, and the measured baseline it must
beat is the current fallback, which is **free** (r234: 1826 t/s declined vs 1824 staged, +-0.1 %).

**Constraint that kills the naive form:** it must stay *yieldable*.  A raw `cudaMalloc` outside the slab is the
pre-slab crash mode (`INVARIANTS.md`, issue #136): a wide prefill's transient work needs that VRAM.  VA outside the
slab is fine; *backing* outside the movable boundary is not.

**Resolution (r252, `WORKLOG.md` 261110.008): declined, not scheduled.**  Re-measured what a stand-down still
costs on the delivery's own arms (r251 build, `-c 32768`, `-ub 8192`, a 16 k-token prompt): it fires **only** on
the high-residency arm (Flash-Next GSQ-RCO IQ3_S, 1 GPU, `-ncmoe 48`: 47.87 % resident, 32 + 1 tables for
5 926 MiB in the prefill, `re-armed 33` at the compute-buffer drop), never on 35B-A3B `-ncmoe 40` (100 %),
`MOE_EXPERT_CACHE_MIB=8000` (56.5 %) or the 2-device tensor split (100 %) — and the fallback it replaces is
free (r234).  So the store is robustness for a state that is transient, prefill-only and re-armed, at the price
of a rewrite that must stay yieldable; the reopen conditions are in `INVARIANTS.md` r252 / that WORKLOG entry.

### 7.7 Next state

1. `TODO` #284 — defer the ring's redirect restore to `stage_free_ev` so the third copy (`input_cpy`, ~1 GiB) can
   go; then re-measure it on Flash-Next (the record's one clean A/B put the D2D at +0.3 %, on a shape 4x smaller).
2. The arena-sizing policy question (§6.1) — now with the number: ~4-5 GiB of transport footprint that only a
   policy change can turn into residency.
3. Only if a config with a dispatch-bound fill matters: the scoped stream change (`0002` minus the global knob).
   **Re-cut onto the r251 tree (r251)**: `0002-stage-copy-streams.patch` is regenerated against
   `release.json` tree `5fbe6aff5` (`git apply --check` clean) and its numbering is unchanged — its copy streams
   are index 1 and 2, inside the copy range r251 reserves (`copy_stream()` = 1, streams 1..4 reserved, concurrent
   branch streams based at 5), so it can no longer collide with `GGML_CUDA_GRAPH_OPT`'s branches.  Its
   `MOE_STAGE_COPY_STREAMS` clamp of 4 is exactly that reservation; raising it would need the layout constant
   changed in `common.cuh` too.  One hunk adapted to r250's staging-redirect retirement
   (`/*defer =*/ !redirect` -> `true`).  Reviewed, never applied.
4. Do **not** re-enter the contiguous static tier (§2g) — 7.4's run-layout analysis prices it at 7-14 points of
   decode mass (see `heat-run-layout.py`; §2's "one bulk DMA from the contiguous complement" needs a resident set
   whose heat-weighted mass survives being made contiguous, and it does not).
