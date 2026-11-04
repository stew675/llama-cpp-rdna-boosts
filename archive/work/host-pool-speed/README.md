# host-pool-speed — make `--host-experts pool` faster, simpler and automatic

> **RETIRED AND ARCHIVED (2026-11-04).**  Promoted in r207 and then superseded by
> [`archive/work/host-expert-eager-push/`](../host-expert-eager-push/README.md), which replaced the host-expert
> tier policy this campaign was tuning.  Read [`RESOLUTION.md`](RESOLUTION.md) first: it says what shipped,
> what was obsoleted (the `push` transports and the `TODO` items that existed for them) and what stayed open.

**Seeded 2026-11-02 at the maintainer's call.  Nothing here is implemented, and nothing here is in
`patches/` or gets applied to the fork** (`AGENTS.md`, WIP rules: a campaign is a handover, and promotion is a
separate decision).

**The charter, in the maintainer's words:**

- The whole host-pool system should be **simpler / more automatic / faster**.
- **Decode is where the pain is**, above all in **single-GPU** mode.  Even with a pool big enough to hold the
  *entire* host-expert set, decode still trails just pinning the host experts directly.
- If the pool **cannot** be grown to hold everything, cold experts are seen to be **streamed from disk over and
  over even though they should be in the page cache**.
- The mechanisms Brian's PR #267 offers that pool mode lacks should be examined and, where they are real,
  implemented **in pool mode**.
- **RAM_PCT 85 % looks too high** — it has produced swap thrashing on some models and was never re-measured.
- **Acceptance criterion: pool mode should get to the point where PR #267 (`--host-experts mmap` + sparse
  registration) offers no performance advantage.**  Pool mode is the *fundamentally less fragile* design — an
  explicit **intermediary** that works for every split type — so it is the one to invest in.
- If an `mmap`-style mode is ever wanted, the shape to build is **pool-with-a-guarantee**: the same
  intermediary, but it guarantees the requisite pages are **pinned *and* resident** before handing the go-ahead
  back to the GPU (see §7).  Not a direct GPU read of pageable memory.
- **The tuning model is `Qwen3.5-122B-A10B` UD-Q4_K_XL, not Flash-Next** (2026-11-02, §9.1): it is large enough
  that it does not fit cleanly in the 64 GiB RAM class the mode exists for, it has **no PLE engine** to confound
  memory/disk measurements, and it carries its MTP head **embedded**.  Flash-Next is a **final validation**
  model only.  Earlier pool tuning on 35B-A3B (which fits) is treated as smoke data, not as evidence.
- **First pass = one GPU and `-sm tensor` on the two x8 GPUs** (§9.4).  The 3-GPU arms and the Flash-Next /
  PLE / swap / `RAM_PCT` tuning are the campaign's **last** phase, not its first.
- **`-ncmoe 48` is the primary arm** (2026-11-02): it stresses the pool hardest and is the VRAM-constrained case —
  nearly all experts on the host, so the arena is as large as VRAM allows and the arena hit rate is at its best,
  which is what makes a *pool* policy the remaining variable.
- **No policy conclusion from a short decode.**  Policy validation runs generate **≥ 2000 tokens** (1000 the
  floor).  Short runs (128-192 tokens) are for correctness/oracle smoke only: the arena ramps up over the first
  few hundred tokens, which flatters or disguises a pool policy (§9.5).
- **Simulate before implementing** (§9.6): capture a routing trace at the campaign's own arm and score candidate
  pool policies with `scripts/moe-trace-replay.py` (the deliverable already has the replayer and the two-tier
  pool model).  A candidate that the simulated oracle does not separate is not implemented.

**Status of PR #267:** reviewed at the architecture level (see §8) and **not adopted**.  The maintainer's
decision is to hold it, gather these numbers first, and reply to the submitter later.  Its record lives under
`wip/mmap-host-experts-r204/` on the submitter's branch — it is *not* in this repo's `patches/`.

---

## 1. What exists today (do not re-derive)

`--host-experts pool` (r111/r112, block 12): the host expert tensors live in the **read-only model mapping**
(a device-carrying `ROCm_HostMap` host buffer type) and no full pinned master exists.  Every GPU read of a
pooled table is staged through pinned memory by one of three intermediaries:

| intermediary | used for |
|---|---|
| the bounded **host pool** (`pool_fill_slot_locked`: a `memcpy` out of the page cache, or a `pread` under `MOE_HOST_POOL_DIO=1`) | the decode/verify band's fills, and the whole-table prefill staging |
| a per-table pinned **bounce ring** (`expert_host_src` / `expert_src_done`, grown on demand, never reused until its event drains) | a fill the pool declines |
| `moe_cache_upload_pageable` | the scheduler's used-expert copies (`set_tensor_async` / `stage_upload` / `stage_gather`) |

Never a pageable `cudaMemcpyAsync` from the master: ROCm stages a pageable upload through a shared internal
bounce and back-to-back copies tear (that is what made the first cut nondeterministic).

The pool is a **layout**, not a churning cache (r113): its content is the **arena's complement**, ranked by the
persisted heat prior (band-split since r114: `heat_dec` drives the arena's seed, the combined vector drives the
pool), laid out at sizing and again at the first decode-band pass.  The **demand path never evicts** and never
copies the arena victim into the pool.  `MOE_HOST_POOL_COLD` (r203) reads a pool-resident miss **in place**
through the pool's device alias instead of filling an arena slot first.

Knobs today (see `ENVIRONMENT.md` for each): `MOE_HOST_POOL_{MIB,FRAC,RAM_PCT,COLD,PREFETCH,PREWARM,DIO,
STAGE_BATCH,VERIFY}` next to `MOE_EXPERT_CACHE_{MIB,MIN_MIB,MIN_RES_PCT,RESERVE_MIB,ADMIT,TOUCH,HEATMAP*,
IDENTITY_PREF}`.  That is a lot of surface for "start it and it picks the right thing".

Records: `patches/README.md` block 12 (the r111/r112/r113/r114 notes), `archive/work/host-expert-master-drop/`,
`archive/work/host-pool-layout/`, `archive/work/host-pool-swap-and-prefill/`,
`archive/work/moe-cache-coverage/`, `benchmarks/2026-11-01-host-expert-device-remap.md`.

## 2. The starting measurements (what the gap actually is)

**Decode, and it is not where it looks.**  1 × R9700, gfx1201, 35B-A3B UD-Q4_K_M `-ncmoe 40`,
`MOE_EXPERT_CACHE_MIB=8000`, `-npp 64 -ntg 512` (r203's own release table, one binary, alternating rounds):

| arm | t/s | note |
|---|---|---|
| `pinned`, `GGML_MOE_DEVREMAP=0` (eager) | 60.6 | |
| **`pinned`, remap on (default)** | **65.7** | the r203 sync-free remap |
| `pool`, `MOE_HOST_POOL_COLD=0` | 57.6 | pre-r203 pool behaviour |
| **`pool`, `MOE_HOST_POOL_COLD=1` (default)** | **60.4** | |

So **the pool arm is already within 0.3 % of `pinned` eager** (60.4 vs 60.6) on the same arena, and the *whole*
remaining decode deficit (60.4 vs 65.7 = **−8.1 %**) is that **r203's `GGML_MOE_DEVREMAP` cannot run on a
pool-backed table**: `devremap_ok_locked()` needs `table_cold_ok()` → a pinned master (`t.host_dev != nullptr`),
and a pool table's master is the pageable mapping.  The pool arm also cannot use the remap's in-place cold read
of a *pinned* master because there isn't one — but `MOE_HOST_POOL_COLD` already proved the pool alias is a
valid in-place read source (`moe_cache_get_cold` returns the pool base, `slots + pool_slot`).

That also reframes the maintainer's "a pool big enough for the whole set still trails `pinned`": on this
evidence the pool *content* is not the problem (60.4 vs 60.6 at the default pool fraction), the **sync-free
routing path is** — and at a full pool the model predicts parity with `pinned` cold reads except for exactly
that path.  §3.1 tests both halves (a full-pool arm and the remap on pool tables).

**Prefill is at parity** (2 × R9700, 35B-A3B Q4_K_M `-ncmoe 40`, 70 K prompt, `-ub 2048`, r114): pool
**2103** t/s vs `pinned` **2099**; decode 46.4 vs 49.3.  Forced `DIO=1` 2088/**22.7**; per-copy staging
(`MOE_HOST_POOL_STAGE_BATCH=0`) 1564/46.2.

**Two structural soft spots already quantified:** *"one extra host memcpy per staged byte"* (prefill −1…−6 %)
and *"a single-ubatch `pp512` on the 2-GPU tensor config −34 %"* (r111).  Both live in the experts that are in
**neither tier** — copied one at a time, because each is a distinct read from the mapping/disk.

**RAM + hit rate the mode buys:** `RssShmem` 18.6 → **6.8 GiB** at FRAC 25 on 35B-A3B `-ncmoe 40` (r111); a
12 GiB / 8 GiB cgroup that OOM-kills `pinned` runs pool at ~93 t/s.  Field gfx1100 (30 GiB RAM, 35B-A3B Q8_0,
`-ncmoe 40` = 31.9 GiB of host experts, r112): FRAC 80 → **78.3 t/s decode / 0.3 GiB swap / 26.0 GiB pinned**;
FRAC 90 → 71.8 t/s **and swapping**.  Hit rate 0.7999 → **0.9183**, evictions 4600 → **0** (r113, IQ4_NL
Flash-Next 2 GPU tensor `-ncmoe 48`, MTP n7).

**What is *not* measured:** `MOE_HOST_POOL_RAM_PCT` 80 vs 85 with the r114 binary on a memory-tight box
(`ENVIRONMENT.md` says so explicitly), the FRAC × RAM_PCT interaction, and the pinned total *actually* held —
pool pins with `hipHostMalloc`, which **does not show in `VmLck`** (PR #267's note, and the reason its table
reports stock `pinned` bytes from the log rather than from `/proc`).

**The 35B-A3B cells above are smoke data.**  They fit this box (46.9 GiB of host experts on 96 GiB RAM), so
they cannot show the sub-full-pool, page-cache and swap behaviour the mode exists for.  §9.1 replaces them for
tuning.

## 3. P0 — the work that decides the campaign

> **Disposition decided 2026-11-02 (maintainer):** the remap survives the `#266` reversion and stays a delivery
> capability; the rest of `#266` (including `MOE_HOST_POOL_COLD`, subsumed by the gather) is dropped.  **The
> condition is that the remap is tensor-split aware** — a per-device map whose cold-id encoding is per-device-slice
> aware and whose freshness invariant holds per device.  Tracked as `TODO` #269.  Its measured ceiling on this
> host is **+3.3 %** (§15), not the +8.1 % the r203 cell reported, so it is scheduled *behind* the coalesced fill
> (§16.3/§17) and gated on the multi-device arm.

### 3.1 P0-a: `GGML_MOE_DEVREMAP` on a pool-backed table (the decode gap, ≈ +8 %)

The mechanism is nearly there — `pool_cold_slot_locked()` + `moe_cache_get_cold()`'s pool branch already
produce `n_res + pool_slot` and the pool base, and the eager path already consumes it.  What is missing is the
**device-side** map for pool tables, and one invariant:

- `devremap_ok_locked()` is gated on `table_cold_ok()` (pinned master).  A pool table needs its own cold
  *source*: the map's cold entries must encode the **pool slot**, which is a *host-side* fact at map-build time
  (the policy runs on the host, one call late), so the map can carry `slots + pool_slot` rather than `slots + e`.
- **The blocker is freshness, not resolution**: r203's invariant 3 (`archive/work/pinned-decode-remap/
  README.md`, `ENVIRONMENT.md` `MOE_HOST_POOL_COLD`) says the pool-cold read is safe *because the eager path
  synchronizes before the hook*, and `pool_rerank_from_heat_locked` rewrites pool slots with plain host copies
  from inside the eager `moe_cache_update_host` hook, i.e. after `ggml_backend_synchronize(input_backend)` and
  the ids readback's sync.  A remap that lags one call has **no** sync.  So the design must make a re-rank
  impossible while a lagging map can still be read: a **pool layout epoch** carried in the cold id (and checked
  on the host at the lag callback, with the map rebuilt on an epoch change), or a hard **freeze** of the pool
  layout for the duration of the decode band.  The re-rank has exactly two call sites today —
  `alloc_table_locked` and the first decode-band pass (`pool_rank_from_prefill_locked`) — which is what makes a
  freeze plausible.
- Acceptance: same-seed greedy text byte-identical across `pool` (`COLD` × `DEVREMAP` crossed), MTP acceptance
  counts identical, prefill KLD 0.000000, and the r203 invariant list extended rather than weakened.
- Ceiling: **+8.1 %** decode on the cell above (recovering `pinned`'s remap gain).  Judge it on 1 GPU *and* on
  a 2/3-GPU tensor arm where the remap still declines for its own reason (the Meta iface is NULL) — the pool
  campaign does not change that.

### 3.2 P0-b: `MOE_HOST_POOL_RAM_PCT` and swap thrashing — **final phase, on Flash-Next**

The maintainer's report: the thrash showed up with **Flash-Next**, where the model weights *and* the PLE weights
are both streaming and the page cache evicts heavily into swap.  That makes it a **Flash-Next / PLE-interaction**
measurement, so this item is scheduled for the campaign's **final validation phase** (§9.4), on the field box and
then on the 122B as the PLE-free control.  It is not a Phase-1 blocker.  Properties to attack when we get there:

- 85 % of `MemAvailable` **measured once, at the first pool build** — a box whose availability later shrinks
  (another model, a server, a browser) keeps the old ceiling.
- The clamp keeps a **one-slot floor per table**, so the pinned total can exceed the ceiling by
  `n_tables × host_bytes` (documented in `ENVIRONMENT.md`; never measured on a real model).
- `FRAC` and `RAM_PCT` interact and only `FRAC` was swept (r112, and only at 80/90 on the field box).
- The pinned bytes are **not visible to the operator** (`hipHostMalloc` is not `VmLck`), which is exactly the
  blindness that makes a thrash hard to attribute.
- **The ceiling's denominator is host-wide, not the cgroup's.**  `RAM_PCT` is a percentage of
  `/proc/meminfo`'s `MemAvailable`, so inside any container or `memory.max`-limited cgroup (the normal
  deployment, and the only way to emulate a smaller box here — §9.2) the pool is sized from RAM it is not
  allowed to use, and the ceiling never binds until the kernel OOM-kills.  The cgroup knows its own limit
  (`/sys/fs/cgroup/memory.max` + `memory.current`); reading it is both the bug fix and what makes the
  emulation faithful.

Work: re-measure `RAM_PCT` 80 vs 85 (and 70) on the tight box with the current binary, watching `VmSwap`,
`pswpout`, `MemAvailable` and the decode rate; then either lower the default or make the ceiling **dynamic**
(re-check `MemAvailable` at sizing and at the first decode-band pass, never mid-band).  Also add a
truthful pinned-bytes report (a `WARN` at pool build and a counter in the cache's traffic report).

## 4. P1 — the mechanisms worth implementing in pool mode

### 4.1 P1-a: disk streaming when the set does not fit (the maintainer's page-cache complaint)

`MOE_HOST_POOL_DIO=auto` is **all-or-nothing per box**, decided once: buffered iff `planned pool + the whole
host expert set fits in MemAvailable`, else `O_DIRECT` for every fill.  On a box that takes the `O_DIRECT` arm,
a **repeated** cold expert is a fresh `pread` every time, even when the page cache could have kept it — which
is the observed "streamed from disk over and over".  Candidates, cheapest first:

1. **A residency check on the fill source** (`mincore`/`MADV_WILLNEED` before the `pread`): read buffered when
   the pages are already resident (free), `O_DIRECT` when they are not (no page-cache pollution).  This is a
   per-read decision instead of a per-box one, and it is the honest version of "it should have been in the page
   cache".
2. **Readahead on the next fills** (a small `MADV_WILLNEED`/`posix_fadvise` queue on a worker for the experts
   the routing is about to need) — note `MOE_HOST_POOL_PREFETCH` exists and is **default 0** because it hurt
   until the pool nears full residency; re-measure it against the r113 layout before adding anything.
3. **A small page-cache tier** (a second-chance list of recently-used experts that the page cache keeps, with
   the pool admitting only what survives) — only if (1) and (2) are insufficient; it reintroduces a churning
   structure that r113 deliberately removed, so it needs the offline model's numbers first.

**An external "pipelined asynchronous staging" proposal (pinned ring + `O_DIRECT` + a background fill thread)
was reviewed 2026-11-02: see [`PROPOSAL-pipelined-staging.md`](PROPOSAL-pipelined-staging.md)** — the ring, the
`O_DIRECT`-into-pinned-memory fill and the overlap are already shipped (r111/r114) and the CPU-copy half is
measured at <=0.3 % of wall clock, so only the *demand-fill* worker is open, with a ~19 % decode ceiling that
exists only on a sub-full pool and two measured negative precedents (r113's worker, `MOE_HOST_POOL_PREFETCH`).
Screen it with the replayer first, per §9.6.

Measure with `/proc/diskstats` (bytes read per decode token) and `MOE_HOST_POOL_VERIFY=1`, not throughput alone.
The reference model's full pass is **166 GB of shard bytes** (§9.1), so "a repeated cold expert re-reads from
disk" is not a rounding error — at ~1.8 GB/s it is ~92 s of a pass.

### 4.2 P1-b: per-table pool sizing from heat, not bytes

Today each table's pool slots are `n_experts × FRAC/100` (or proportional to the tensor's *byte* share for an
explicit `MOE_HOST_POOL_MIB`) — **uniform**, and a one-slot floor.  The *content* is heat-ranked, but the
*size* is not, so a table with a long cold tail starves a table with a flat hot set.  PR #267's 0009 has the
same insight ("if the whole tier does not fit, the hottest tables get one first") and its 20 GB-budget limit
(59 tables with no tier, decode 28.4 t/s) is the failure mode in reverse.  This is a **sizing** change with the
existing offline instrument: `scripts/moe-trace-replay.py` models a two-tier (arena + pool) budget and reports
MiB of L3 reads per decode token, so a candidate allocation can be screened before any code changes.  Candidate:
allocate by each table's **decode heat mass** (`heat_dec`) with a floor of one slot, subject to the same
`RAM_PCT` ceiling.

The **acceptance path** for a sizing candidate is §9.6: simulate it on a real trace first, and implement only what
the simulated oracle separates.

**Session 2 result — the candidate as written is degenerate; re-specify it as coverage.**  A top-k router gives
every table exactly `n_used` experts per token, so a table's *total* decode routed mass is the same constant for
all of them and "allocate by heat mass" reproduces the uniform rule exactly (measured in
`tools/pool-sizing-sim.py`).  What differs is each table's **concentration**, so the candidate becomes
`cov_t` = the slots a table needs to hold `t` of its own mass (t = 0.90), fitted to the same byte budget.  On the
reference trace that is **-42 % L3 (disk) bytes per decode token at `FRAC` 80** (-29 % at `FRAC` 50) with `h`
unchanged — **screening-passed**, and then **implemented and measured**: `MOE_HOST_POOL_SIZING=1` gives **+7.5 %**
decode at `FRAC` 80 (17.80 -> 19.14 t/s) and **+16.4 %** at `FRAC` 60 (13.60 -> 15.83), at identical pinned bytes,
with the disk reads down 15/23 % and the pool's own hit rate up 6/10 points — at a cost of 13-15 % prefill.
Numbers, trade and gates: `HANDOVER.md` §3.3 and §4/F6.

### 4.3 P1-c: precopy / spare slots (PR #267's 0006-0008), in pool terms

#267 copies the *likely victims* back to the RAM tier ahead of time on a side stream into K spare slots, so an
admission rarely waits for its copy-back.  Pool mode's analogue is a **demand prefetch into empty pool slots**
on a side stream — not an eviction swap: r113 measured the eviction-side version (bounce-staged swap with the
incoming's slot) at **2× decode for 0 MiB** of L3, and deleted the background eviction-prefetch worker for it.
`MOE_HOST_POOL_PREFETCH` is the surviving knob (default 0).  Re-measure it with (a) the r113 layout, (b) the
heat-split prior, (c) a `-sm tensor` arm; keep it only if it wins when the pool is *not* full.
#267's "admission from the 3rd touch" is already `MOE_EXPERT_CACHE_TOUCH` (default 2) — a knob, no new code.

### 4.4 P1-d: adopt the two mode-agnostic pieces of #267 outright (no `mmap` needed)

1. **0010 — cap the bounce ring per table (8 entries, wait for one copy when full).**  This is pool mode's own
   path (`moe_cache_upload_pageable`), and the delivery's ring is *grown on demand*, so a declined whole-table
   read can in principle grow it toward the master's size — outside any pin budget.  Small, testable, and it
   needs no new knob (the old path is wrong, so 0010 deliberately has none).
2. **0003 — keep the expert-cache arena across prefill transitions**, and the narrow-2 fallback (the draft's
   compute region keeps its `cudaMalloc` fallback when it fits outside the slab instead of evicting arena
   tables).  The submitter measured no eviction change on the pinned arm (17 per run, both builds), so the
   pinned risk is low; it is the *pool* arm that should show the win (no arena re-fill after every prompt).

Both must be re-cut against the current tree (they are against r204; `main` is r206) and gated as their own
changes, not folded into a campaign-sized patch.

## 5. P2 — simpler and automatic

Goal: **start it and it picks the right thing.**  Inventory first (every knob above, what it decides, when),
then collapse:

- One `--host-experts auto` decision that owns `pool | pinned` from the model, the host RAM and the arena size,
  instead of `FRAC` × `RAM_PCT` × `MIB` × `DIO` × `MIN_RES_PCT` × `ADMIT/TOUCH`.
- Keep the **kill switches** (a beneficial default is on; the variable exists to disable it) and the
  **diagnostics** (`MOE_HOST_POOL_VERIFY`, the traffic report); delete/merge the *tuning* knobs that only exist
  because a measurement once needed two arms.
- The pattern to follow is the delivery's own: when a gate passes, flip the default in the same change and turn
  the opt-in into an opt-out (`AGENTS.md`, Default-on policy).

This is the item with the least technical risk and the most user-visible benefit, and it should not wait for
P0/P1 to finish.

## 6. Non-goals

- **No direct GPU read of pageable memory** (the r33/#116 removal stands), and **no sparse/alias registration**
  of the model mapping (`hipHostRegister` on demand) — that is PR #267's mechanism and the source of its
  unfalsifiable pin growth (`VmLck` ~43.8 GB before its own budget patch; POSIX-only; `n_slices == 0`; and the
  "scheduled copies from a pageable master" fallback that its own notes call fault-prone on ROCm 7.14).
- **No device-side policy / KSLOT / promotion** — the r40 family stays removed (`ENVIRONMENT.md` §1.2).  A
  pool-slot *id* in the remap map is a host decision encoded for the device, which is what r203 already is.
- **No new mode** that is not the intermediary.

## 7. P3 — the shape an `--host-experts mmap` would take if it is ever wanted

The maintainer's framing, recorded so it is not re-derived: *pool mode's mechanism, but the intermediary
**guarantees** the requisite pages are pinned **and** resident before it hands the go-ahead back to the GPU.*

Sketch (design note only, no work scheduled):

- The pool already **is** the pinning guarantee (a bounded pinned slot for every expert a GPU may read).
- What it does **not** guarantee is that the *source* pages are **resident** before the copy: an `O_DIRECT`
  fill on a sub-full pool is a synchronous disk read in the middle of the demand path.  So the missing half is
  **residency + readahead**, which §4.1 already covers inside pool mode.
- A direct-read variant would add "pin *and* keep resident, per expert, on demand" — precisely what #267's 0002
  does, and its measured costs are the warning: an unbounded `VmLck` until a budget was added, an all-device
  synchronize in the trim, a per-expert register/unregister path, and a decline path that still has to copy.
  So: only with a **hard per-table pin budget** (the pool's `RAM_PCT` shape), a **residency check** before the
  read is released, and **no** fallback that copies from pageable memory.

> **Cross-reference (2026-11-02, added from the 1-GPU host).**  §7's rule was violated by the parallel
> `archive/work/host-expert-pinned-window/` campaign (closed 2026-11-02), which let SM kernels read the mmap'd master in place: on a model
> larger than RAM that is a fatal GPU page fault (`Page not present`, no XNACK, `WORKLOG.md` 261102.012) and with
> `mlock` it stops being a RAM saving at all (261102.013).  Two of its pieces are **sound under §7 as written**
> and worth taking here: CE (copy-engine) copies out of a *registered* mapping instead of the per-table bounce
> `memcpy` -- measured there as prefill 375.9 -> 532.6 t/s and -6.7 GB RSS, and independently priced by Avalon at
> fills 115 -> 61 us -- and an arena-aware gather **gated on a hold** (for pool mode: gathering from the pinned
> pool slots, a D2D).  Neither involves a kernel reading pageable memory.

## 8. PR #267 — the architecture-level assessment (recorded, not adopted)

`--host-experts mmap` (0001) + sparse file-alias registration (0002) + arena-across-prefill (0003) + a per-table
RAM tier with `cold_ptrs` (0004-0005) + precopied spare slots (0006-0008) + one pin budget (0009) + a bounce
cap (0010).  Its measured envelope: 1 × R9700, Flash-Next GSQ-RCO IQ3_S, MTP, q8_0 KV, 256K ctx, ROCm 10.1,
`pinned` 74.7 vs `mmap` 72.4 greedy t/s, RAM available 6.3-6.9 → 16.4-17.2 GB, prefill 1189 vs 1278 t/s,
20 GB budget → **28.4 t/s** (59 tables without a tier).  Their own limits: *"single device only"*, *"CUDA side
untested"*, POSIX only.

Why it is the fragile shape: every place that must know geometry has to know it (`n_slices == 0` for a fused
table → the cache declines → the scheduler copies from a pageable master → the meta 2-D spliced upload that
their own 0001 comment says faults on ROCm 7.14, while 0010's bounce covers only the 1:1 unsplit geometry),
axis-1 vs axis-0 slices register differently, and the remap still declines under `-sm tensor` (the Meta iface
is NULL) so that arm gains nothing.  Pool mode's validation envelope, by contrast, is *"text identical to
`pinned` on 1/2/3-GPU tensor, gemma4 (fused tables), Flash-Next and MTP"*.

**Conclusion:** adopt 0010 and 0003 (§4.4), take the *ideas* behind 0009 and 0006 where pool mode has the same
hole (§4.2, §4.3), and decline the mode.

## 9. Measurement protocol

### 9.1 The tuning model — `Qwen3.5-122B-A10B` UD-Q4_K_XL (decided 2026-11-02)

**Primary tuning model:** `/llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-0000{1,2,3}-of-00003.gguf`.
Verified from the GGUF metadata on this box (`general.architecture = qwen35moe`):

| property | value | why it matters here |
|---|---|---|
| size | **73.2 GiB** (78,646,479,904 B, 3 shards) | does not fit cleanly in the 64 GiB RAM class the mode exists for — the point the earlier 35B-A3B tuning missed |
| experts | 256 experts × 8 used, `expert_ff` 1024, shared expert 1024, 49 blocks | ~118 B of its 122 B parameters are expert weights, i.e. **~66 GiB of the 73 GiB is the host-expert set** — `-ncmoe` is a wide, well-resolved dial, and the pool can be made genuinely sub-full |
| PLE | **none** (no `ple.*` / per-layer-embedding keys anywhere in the file) | the confound the maintainer called out is absent by construction |
| MTP | **`nextn_predict_layers = 1`** (embedded head in the same GGUF) | exercises the *embedded*-head path — block 15's `LLAMA_SPEC_DRAFT_WINDOW` default and the `-ncmoe` nextn-layer guard — not a separate draft file |
| structure | `full_attention_interval = 4` (hybrid SSM + attention), 2 KV heads × 256, ctx 262144 | a real decode shape (GDN/SSM + MoE), not a toy |
| prior data | `-ncmoe 99 -sm tensor` pp8192 **675.5** (stage-all) / 615.2 (gather) / 618.0 (off) vs all-GPU **3378.8**, and *"the 122B and IQ4 move the same shard bytes per pass (166 GB)"* (`archive/work/moe-expert-cache/item3-findings-session20.md` §session 20) | already validated as runnable fully host-resident on 3 × R9700, and it supplies the disk-streaming denominator: **166 GB of shard bytes per full pass** (~92 s at 1.8 GB/s) |

**Flash-Next (qwen3.8 / `qwen4exp`) is a FINAL-VALIDATION model only.**  Verified from its GGUF: it carries a
real PLE engine (`qwen4exp.embedding_length_per_layer_input`, `qwen4exp.ple.{layers,ngram_size,head_vocab_sizes,
head_offsets,heads_per_ngram,layer_multipliers,conv_kernel,eos_token_id}`), which the delivery reads as
`TENSOR_READ_LAZY` (PLE/engram) tables — a **second lookup/stream subsystem inside the model** (see
`ENVIRONMENT.md`'s `LLAMA_LAZY_*` row).  Any pool measurement there mixes the expert stream with the PLE stream
and produces conclusions that do not survive contact with a PLE-free model.  It also uses a **separate** MTP
draft file (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, plus a DFlash2 draft), i.e. a different draft path.
Use it at the **end**, for cross-architecture validation of a conclusion that was reached on the 122B.

Kept, but demoted to smoke/regression arms (they fit, so they hide the effects under study): 35B-A3B UD-Q4_K_M
(`-ncmoe 40`), Qwen3.8-27B Q4_K_XL (dense control), gemma4 26B-A4B (fused-table decline path, `-sm layer` only —
`-sm tensor` is rejected for `gemma4` arch-wide).

### 9.2 Emulating the 64 GiB class on a 96 GiB box

A `memory.max` cgroup is the honest emulation, and the delivery has used exactly that before (a 12 GiB / 8 GiB
cgroup OOM-kills the `pinned` arm while pool mode runs at ~93 t/s, r111): `systemd-run --scope -p MemoryMax=64G`
(or a dedicated user slice) around the whole benchmark process, with `VmSwap`, `memory.max`,
`memory.events`/`memory.stat` (incl. `pgscan`, `oom_kill`) and `MemAvailable` logged.  Pinned pages are charged
and unreclaimable, so the cgroup reproduces the OOM/pressure side faithfully.

**It does not reproduce the pool's *decisions* on its own**: `MOE_HOST_POOL_RAM_PCT` reads host-wide
`MemAvailable`, so inside the cgroup it sizes from RAM the cgroup may not use (§3.2).  Until that is fixed, run
the emulation with the ceiling expressed explicitly — an equivalent `MOE_HOST_POOL_MIB`, or a `RAM_PCT`
recomputed from the cgroup limit — and record which was used.  The code fix (read `/sys/fs/cgroup/memory.max`)
is itself a P0 item, because containers are the normal deployment shape.

### 9.4 Phases, and the first pass's hardware (decided 2026-11-02)

**Phase 1 — first pass (this is the work to start):**

- **One GPU** (the single-GPU pain point) and **`-sm tensor` on two GPUs**, 122B-A10B, embedded MTP on/off,
  `GGML_MOE_DEVREMAP` × `MOE_HOST_POOL_COLD` crossed, pool **sub-full** and **full**, plus the all-host
  (`-ncmoe 99`) disk arm.  This is where P0-a (the remap on pool tables) and P1-a/P1-b/P1-d get measured.
- **`-ncmoe 48` is the primary arm** (49 blocks in the 122B, so 48 hosted layers ≈ **62 GiB** of host experts by
the per-layer estimate below).  It is the case the pool exists for: nearly every expert lives on the host, so
VRAM is spent almost entirely on the arena, the arena hit rate is at its best, and the *pool* policy is
consequently the remaining variable.  It is also the tight one for this box — `FRAC` 80 % asks for ~50 GiB of
pinned pool out of ~76 GiB `MemAvailable`, leaving ~26 GiB of page cache for a 73 GiB model, i.e. `-ncmoe 48`
on the 96 GiB box *already* exercises the sub-full-pool and disk-streaming path; the 64 GiB cgroup (§9.2) is the
field case.  Surrounding arms: `-ncmoe 40` (smaller host set, larger arena) and `-ncmoe 99` (all-host disk arm)
as gradient points, not as the primary.
- **The pair must be the two x8 GPUs, never the x4 one.**  Verified on this box (2026-11-02):

| HIP / `-dev` index | PCI | CPU root port | CPU link | uses |
|---|---|---|---|---|
| **0** | `03:00.0` | `00:01.1` | **x8** | **yes — first GPU of the pair** |
| 1 | `06:00.0` | `00:01.2` | **x4** | **never in this campaign** |
| **2** | `09:00.0` | `00:01.3` | **x8** | **yes — second GPU of the pair** |
| 3 | `1a:00.0` | `00:08.x` | iGPU `gfx1036` | never (the delivery is gfx1201-only) |

  How it was established: `rocm-smi --showbus` maps HIP index -> PCI, and the **CPU root ports**
  (`/sys/bus/pci/devices/0000:00:01.{1,2,3}/current_link_width`) read **8 / 4 / 8** — the GPU's own
  `current_link_width` reads 16 because it sits behind a switch's downstream port, so *do not* trust that file
  for this question.  This matches `AGENTS.md` (the x8 pair is `HIP_VISIBLE_DEVICES=0,2`).
- **So the 2-GPU command pins `HIP_VISIBLE_DEVICES=0,2`**, after which the in-process indices are 0 and 1 —
  `--device 0,1`, `-ts`, `-mg` and any log line refer to those, *not* to physical 0 and 2.  A bare `--device 0,2`
  under that pin silently selects one device.
- 1-GPU arms take index 0 of whatever set is visible (`HIP_VISIBLE_DEVICES=0`), always pinned: an unpinned
  multi-device arm enumerates the iGPU and aborts at load with `invalid kernel file` (the `gfx1036` trap).
- **Iterating on the 122B is cheap now**: `/llm` is on the faster mount, ~**15 s of disk reads** to load the
  model, so the big model is the iteration vehicle and the small models are not needed for speed.
- The 122B on **one** 32.6 GiB card is a genuine host-expert workload: its expert weights are ~1.4 GB per layer
  (256 x 3 x 1024 x 3072 at UD-Q4_K_XL, derived), so `-ncmoe N` puts roughly `1.4 x N` GB on the host —
  `-ncmoe 32` is ~45 GB (≈ 42 GiB) and `-ncmoe 48` ~67 GB (≈ **62 GiB**) of host experts against a ~32 GB card.
  On 1 GPU the arena is then ~25 GiB for 62 GiB of experts (~40 % aggregate), and on the 2-GPU x8 pair ~50-55 %:
  large pool, high-but-not-identity arena hit rate — the churn regime the campaign is about.

**Phase 2 — the last phase (do not start here):** Flash-Next final validation, i.e. the PLE interaction and the
`RAM_PCT` / page-cache / swap thrash (§3.2): measure with the model weights **and** the PLE stream active, on the
field box and under the 64 GiB cgroup, against the 122B as the PLE-free control.  The 3-GPU arms also belong
here (they add the all-reduce and the iGPU trap, and nothing in the first pass needs them).

### 9.5 Arms, commands, gates

- Boxes: **primary** (3 × R9700 gfx1201, 96 GiB, ROCm 10.1) — the tuning box, with the 64 GiB cgroup variant
  for the tight case.  The field box (gfx1100, ~30 GiB) belongs to Phase 2 (§9.4) and is the ground truth for the
  mode's original purpose.
- Model arms (Phase 1): 122B-A10B at `-ncmoe` values that give a **sub-full pool** (the interesting regime), a
  **full pool** (the maintainer's "big enough for the whole set" case), and one **all-host** (`-ncmoe 99`) arm for
  the disk-streaming case; **1 GPU** and **`-sm tensor` on the x8 pair**; `--spec-type draft-mtp` on the embedded
  head vs none.  The 3-GPU arms and the Phase-2 model (Flash-Next) are not Phase 1.
- **Decode length: ≥ 2000 generated tokens for any policy number (1000 is the floor).**  The arena ramps up over
  the first few hundred tokens, so a 128-192 token decode measures the ramp, not the policy — short runs are for
  correctness/oracle smoke only.  Under MTP report both the generated and the accepted count, since acceptance
  changes how many steps the pool sees.
- Always include the **`pinned` arm in the same run** (the acceptance criterion is a ratio to `pinned`); arms
  alternated, never parallel (`AGENTS.md`); pin `HIP_VISIBLE_DEVICES` on every multi-device arm (the iGPU trap).
- **Full feature set, every arm** (`AGENTS.md` Default-on policy): no `FOO=1 <bench>` unless the arm *is* the
  A/B.
- Oracles: same-seed greedy text (byte-identical — a pool/layout defect is a wrong answer, not a crash), MTP
  four-axis acceptance counts, `gate-prefill-logits.sh` for any change to the pool fill/staging path,
  `llama-batched-bench -npl 1,4,8`, `MOE_HOST_POOL_VERIFY=1`, `check-gpu-faults.sh`.
- Instruments: the cache's own traffic report, `MOE_HOST_POOL_VERIFY`, `/proc/diskstats` (and the cgroup's
  `io.stat`), `VmSwap`/`pswpout`/`memory.events`, plus `scripts/moe-trace-replay.py` (two-tier offline model)
  to screen a layout/sizing candidate before any code changes.
- **Log level: `-lv 4` for anything out of libllama; `-lv 3` is not enough** (corrected 2026-11-02, session 2).
  `common/log.cpp` maps a raw `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4), so every `LLAMA_LOG_INFO` line —
  `host_experts_mode=POOL`, `table_pool_locked`, `pool_decide_dio_locked`, `pool_clamp_slots_to_ram`,
  `apply_heat_seed_locked`, `moe_cache_report: MoE expert cache: h=` — is invisible at `-lv 3`.  `-lv 3` shows
  `COM_INF`/`COM_WRN` only (`slot print_timing: MoE arena =` is the usable per-turn `h=`).  This is what made
  session 1 believe the pool never activated (`HANDOVER.md` §4/F3).
- **The teardown cache report is zero in every mode** (same cause as above, a second time): `moe_cache_report` is
  an `atexit` hook that runs *after* `moe_cache_forget_all`, so `tables=0, slots=0, arena=0.0 MiB` and
  `host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned` are printed whether or not a pool existed.  Read the
  cumulative counters (`takeover=`, `expert access:`, `expert H2D traffic:`) instead, or the per-turn line.
- Page cache is a *state to control*, not a nuisance: record it (or drop it) before every arm, because on a
  96 GiB box with a 73 GiB model `DIO=auto` picks buffered while on a 64 GiB box it picks `O_DIRECT` — the same
  binary, two different behaviours.  That divergence is exactly what P1-a replaces with a per-read decision.

### 9.6 Simulate before implementing — the routing trace and the replayer (decided 2026-11-02)

One GPU run produces a trace; the replayer scores candidate **pool policies** on it in seconds, with no model and
no GPU.  This is the campaign's candidate filter, and it exists already (r110, `TODO` #61 P1).

**Capture** (the trace is a diagnostic, off by default; `ENVIRONMENT.md` has the format):

```bash
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_TRACE=/tmp/trace-122b-ncmoe48.txt MOE_EXPERT_CACHE_MIB=4096 \
  ./build-rocm/bin/llama-cli -m /llm/models/Qwen3.5/122B-A10B/Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf \
  -ncmoe 48 -ngl 99 -c 8192 -ctk q8_0 -ctv q8_0 --temp 0 --single-turn --no-display-prompt \
  -n 2000 -p "$(cat prompts/prose-rdna-boosts.txt)"
```

Always pass `-c`: the model's training context is 262144, and the KV is cheap here (13 KV layers x 2 heads x 256
x f16 → ~27 KB/token, ~220 MiB at `-c 8192`, half that at `q8_0`), so an unset `-c` silently spends GiB on context
instead of arena.  The 5246-token prose prompt plus 2000 generated tokens needs `-c 8192`.

**The four gotchas (read before trusting a trace):**

1. A **fully resident (identity) table is not traced on the decode band** — it cannot miss.  Capture from a
   *partially* resident config (e.g. `MOE_EXPERT_CACHE_MIB=4096` on the `-ncmoe 48` shape); the routing is the
   model's, the arena size is what the replayer varies, so the stream stays valid while the residency is not.
2. The trace **dedupes the per-device record** a tensor split produces: the table set is ONE device's and
   `--arena-mib` is a **per-device** budget → pass `--devices 2` for the x8-pair arms.
3. **Validate the model first**: `--measured-h <the run's logged h=>`.  If the replayer's `h` does not land on the
   logged value, no comparison means anything (this is the same trap the r114 replayer defects came from).
4. **After P0-a the pool tables stop being traced on the decode band** (the remap path never reads the routing
   back to the host), so traces for the pool work must be captured with `GGML_MOE_DEVREMAP=0` — or before P0-a
   lands — else they lose exactly the records the campaign needs.

**Score:** `scripts/moe-trace-replay.py <trace> --arena-mib <per-device> --devices <1|2> --pool-frac <N>
--pool-init heat --mass-band decode --policies shipped,static,quota,window`, then read the two-tier numbers as
**MiB of L3 reads per decode token** and convert to time with the measured disk/page-cache bandwidth (the
reference model's full pass is 166 GB, §9.1).  `static` is the oracle upper bound for any placement that cannot
re-rank at run time; `window` is what a prefetch-on-miss scheme can hope for.

**What it can and cannot answer.**  It scores placement / admission / prefetch / pool size — P1-b, P1-c, P1-d and
§2's sub-full boundary.  It **cannot** score P0-a, which is a per-op sync cost: that needs real runs, and its
ceiling is already measured (pinned eager 60.6 → remap 65.7 t/s).  For P1-a it cannot score the copy cost, but it
*does* predict the benefit (fewer disk bytes per token), which is the half that decides go/no-go.

**Use it to kill candidates, not to bless them.**  The precedents are exactly that: the per-layer capacity quota
measured **+0.0025 h / −1.6 % traffic** (r110) and long-session adaptation's oracle ceiling was **~6 %** of the L3
traffic per topic shift (r113) — both dropped on the simulation, before any code.  A candidate the simulated
oracle does not separate is not implemented.

## 10. Open questions for the maintainer (decide before the first code change)

1. ~~**Decode target**~~ — **answered 2026-11-02**: iterate on the 122B (the faster `/llm` mount makes its load
   ~15 s), and the bar is parity with `pinned` *on the arms the campaign runs*, which means P0-a still has to
   land before that comparison is meaningful.
2. ~~**`RAM_PCT` / swap**~~ — **answered 2026-11-02**: the thrash is a **Flash-Next weights + PLE streaming**
   effect, so it is measured at the **end** of the campaign against Flash-Next (§3.2, §9.4), not in Phase 1.
   The cgroup-aware denominator (§9.2) is still worth fixing independently, because containers hit it without any
   PLE involvement — say whether that should be its own small change, or ride with the Phase-2 work.
3. **Knob collapse (§5)**: is a "one `auto` decision, everything else diagnostic" shape the target, or should
   the existing per-knob defaults stay user-visible for the field configs?
4. **P3 (§7)**: is the "guaranteed pinned **and** resident" intermediary worth a design note of its own, or is
   §4.1's residency check enough?
5. **PR #267**: hold the reply until §3.1 and §4.1 report numbers (current plan), and if it is declined, does
   the reply offer a `wip/` home for the submitter's tier work so it is not lost?
6. **`TODO` handle**: do you want one for the swap-thrash report / cgroup-aware `RAM_PCT`, so they are tracked
   outside `wip/`?

## 11. References

- `AGENTS.md` — the x8 pair (`HIP_VISIBLE_DEVICES=0,2`) and the iGPU load-abort trap; also `COMMUNITY-CONFIG.md`
  for the `-sm tensor`-vs-`-sm layer` numbers on the x8 ports.
- `patches/README.md` block 12 — the pool's shipped design and its r111-r114 numbers.
- `ENVIRONMENT.md` — `MOE_HOST_POOL_*` and `MOE_EXPERT_CACHE_*` rows; the L0/L1/L2 memory-tier model; the
  `RAM_PCT` "not re-measured on the tight box" note.
- `benchmarks/2026-11-01-host-expert-device-remap.md` — the r203 table that gives §2 its decode attribution;
  `archive/work/pinned-decode-remap/README.md` — the remap's invariants (3 is the pool interaction).
- `archive/work/host-expert-master-drop/` (r111), `archive/work/host-pool-layout/` (r113),
  `archive/work/host-pool-swap-and-prefill/` (r114), `archive/work/moe-cache-coverage/` (the replayer and the
  admission-quality result), `archive/work/moe-cpu-overlap/` (the miss-traffic measurement: 0.1-0.4 % of a
  token at the delivered arena, 7 % at the 2 048 MiB floor — do not re-open CPU/GPU overlap).
- [`HANDOVER-MULTIGPU.md`](HANDOVER-MULTIGPU.md) — the cold-start handover for the multi-GPU box (state, W1-W4,
  traps, gates, promotion checklist); [`HANDOVER.md`](HANDOVER.md) is the 1-GPU session-2 record.
- `scripts/moe-trace-replay.py` — the offline two-tier model (band-split heat, `--pool-frac`).
- `tools/run-arms-multigpu-261103.sh` / `tools/run-text-purity-multigpu-261103.sh` — the session-4 multi-device
  arm and text-oracle runners (`VIS=0,2|0,1,2`, `SM=tensor|layer`, `NCMOE`, `NGEN`, `LV`).
- `archive/work/moe-expert-cache/item3-findings-session20.md` — the 122B-A10B `-ncmoe 99 -sm tensor` numbers,
  the 166 GB/pass shard-byte figure, and the staging-vs-gather architecture rule (§9.1).
- PR #267 files (on the submitter's branch, not in this repo): `wip/mmap-host-experts-r204/0001..0010`,
  `README.md`.

## 12. Measured so far — first session, 2026-11-02

Box: primary, **one R9700** (32.6 GiB), gfx1201, ROCm 10.1, build `b11903-efaa60db0` (the r206 tip).
Model: the campaign's reference 122B-A10B (§9.1).  Arm: `-ncmoe 48 -ngl 99 -c 8192 -ctk/ctv q8_0 -fa auto
-b 2048 -ub 2048`, `--spec-type none --seed 42 --temp 0 --reasoning off --single-turn`,
`prompts/prose-rdna-boosts.txt` (5258 prompt tokens), `-n 2000`.  Host experts: **67,110 MiB** (65.5 GiB).

| arm | arena | residency | `h=` | prompt | decode |
|---|---|---|---|---|---|
| **shipped default** (`MOE_EXPERT_CACHE_MIB` unset, auto; `GGML_MOE_DEVREMAP` on) | **22,544.8 MiB** | **33.6 %** | **0.7543** | **682.4 t/s** (1.47 ms/tok) | **22.36 t/s** (44.72 ms/tok) |
| trace arm (`MOE_EXPERT_CACHE_MIB=4096`, `GGML_MOE_DEVREMAP=0`) | 3,932.2 MiB | 5.5 % | 0.3488 | 157.2 t/s | 12.88 t/s |

Wall clock: 114 s shipped (19 s load + 7.7 s prompt + 83 s decode), 212 s for the trace arm.
Auto sizing takes the slab's arena region (23.00 GiB): `alloc_all_locked` sizes the cache from what is free
*after* `--fit`, the graph reserve and the KV cache, so "unset" is not "small" — it is **22.5 GiB**, and
`MOE_EXPERT_CACHE_MIB=4096` was costing **4.3x the prefill and 42 % of the decode**.  Same for the trace arm:
restricting the arena is what makes the tables traceable, and it also changes the regime by a factor of two.

**Finding 1 — a decode trace requires `GGML_MOE_DEVREMAP=0` (today, on any pinned table).**  The `D` records
are emitted by `moe_cache_update_host`, the *eager* hook; the r203 remap path never brings the routing back to
the host, and the scheduler returns before the eager hook.  With the shipped default the trace stops after the
two sizing passes (**288 `D` records** for 1864 decode steps).  With `GGML_MOE_DEVREMAP=0`: **268,416 `D`
records**, 38 MB, 1864 steps × 144 tables ✓.  This is a doc gap in `ENVIRONMENT.md`'s `MOE_EXPERT_TRACE` row and
in the replayer's docstring (both explain the identity-table caveat but not this one) — worth a one-line fix
when the campaign next touches docs.

**Finding 2 — the replayer does not calibrate against the delivered allocator.**  Same trace, `--devices 1`,
`--seed sidecar:<the model's .moe-heatmap>`, scored against each arm's own logged `h=`:

| arena | slots/table | replayer `shipped` | run's `h=` | error |
|---|---|---|---|---|
| 3,932 MiB | 14-16 | 0.19-0.20 | 0.349 | **−0.16** |
| 3,932 MiB, `--churn off` | 15 | 0.2855 | 0.349 | −0.06 |
| 22,518 MiB | 87-92 | 0.642 | 0.754 | **−0.11** |

The delivered placement (verified from a `-lv 4` run) is a **uniform 15 slots/table** at 4 GiB, seeded from the
persisted map (`apply_heat_seed_locked: ... heat-seeded 15/15 slots from the persisted map (256 experts, ranked
by the DECODE band)`), i.e. exactly the shape the replayer models — so the gap is in the *dynamics*, and the
`--churn off` arm recovers two thirds of it at the low-arena point.  `--seed sidecar` reports only **31.4 % of
the prior's mass in the seeded slots** at 14 slots, while the run's `h=0.349` is *higher* than that, so the real
cache is losing less of its seed to eviction than the model does.  Candidates, not yet tested: the seeded-resident
eviction order (`count=0` → first to be given up), the `MOE_EXPERT_CACHE_PERIOD`/decay interaction, and whether
the real `access_locked` charge of `t.hits`/`t.misses` includes the band-wide union of a multi-token pass.
**Until this is reconciled, the replayer's policy deltas are not trustworthy** — it is the campaign's P1 tool
item, and it is the one blocker §9.6's "simulate before implementing" rule depends on.

**Finding 3 — the pinned arm's cost, measured.**  `-ncmoe 48` pinned holds **65.5 GiB of non-swappable RAM**
(the cache logs `67110 MiB of host experts`) for 22.36 t/s on one card.  That is the number pool mode has to be
judged against: not a throughput ratio to a leaky baseline, but 65.5 GiB of pinned RAM versus a bounded pool.

**Session 2 corrections to this section (read it before quoting any number above):** two of the five rows in
§12 were mis-labelled.  The `--host-experts pool` row at 22.4 t/s came from a run with no pool in it, and the
"`host pool (L2): 0 tensors`" line that made the pool look dead is printed by *every* run in *every* mode (it
is an `atexit` report that runs after the cache is un-sized).  The pool **does** activate on the 122B; the real
session-1 pool numbers are 17.73 t/s (`O_DIRECT`) and 18.21 (buffered).  And Finding 2 is **resolved**: the
replayer's error was the score band, not its dynamics — the runtime's `h=` is decode-band only, and
`scripts/moe-trace-replay.py` now defaults to that band and calibrates to <0.3 %.  Details and evidence:
`HANDOVER.md` §4/F2/F3.

### 12.1 Session 2 (2026-11-02, overnight) — the pool baseline, measured properly

1 GPU, `-ncmoe 48`, 5258-token prompt, 2000-token greedy decode, arms **alternated in one run**
(`HANDOVER.md` §3 has the full table and the counters):

| | decode t/s | prefill t/s | disk read / run |
|---|---|---|---|
| **`pinned`** (3 arms) | **22.33 +/- 0.06** | 680 +/- 1 | 63-72 GiB |
| **`pool`, `DIO` auto = `O_DIRECT`** (3 arms) | **17.76 +/- 0.03** | 629 +/- 3 | 152-170 GiB |
| `pool`, `DIO=0` buffered (2 arms) | 15.22 / 18.10 (unstable) | 543 / 588 | 152-162 GiB |

So: **the pool is -20.5 % on decode and -7.5 % on prefill on the reference arm, and the `DIO` boolean is not
what costs it** (forced `O_DIRECT` == auto to 0.3 %; the buffered arm is the *unstable* one).  The pool streams
~90 GiB more from disk per run (53.7 of it the one-shot prepopulate), yet it moves the same host-link bytes as
`pinned`.  Two hypotheses were then tested and one closed: the **copy kind** out of the pool (D2D from the mapped
alias vs the pinned arm's H2D) makes **no difference** (17.76/17.80 vs 17.79/17.78, `HANDOVER.md` §3.1), while
**pool coverage** is decisive — see below.

**Pool coverage is the decode lever (measured, `FRAC` swept on the same arm).**  `pinned` 22.34; `pool` at
`FRAC` 100 (256/256 slots, 67.1 GiB pinned) **20.59**; at `FRAC` 80 (the shipped default, 204/256) **17.76**; at
`FRAC` 50 (128/256) **11.75**.  Disk reads per run: ~70 GiB (`pinned`, i.e. the model load) / 100 / 165 / **366**
GiB — so the deficit is *demand reads*, ~163 / ~53 / ~13 MiB per decode token at `FRAC` 50 / 80 / 100, and the
replayer's two-tier model now predicts 160.1 / 48.7 / 0.0 on the same trace.  Two consequences: a **full pool still
trails `pinned` by 7.8 %** (the maintainer's observation, now a number, with the pool pinning exactly what the
pinned master uses), and **the shipped `FRAC 80` default costs 16 % decode against a full pool on this box** (the
`RAM_PCT` clamp did not bind).  `HANDOVER.md` §3.2.

**The sizing changes, measured** (`HANDOVER.md` §3.3, 7 alternating arms): the **full-coverage default**
(`MOE_HOST_POOL_FRAC=100`) gives **20.62** t/s against `pinned` 22.38 — so the pool's raw overhead, with the pool
pinning exactly what the pinned master does (67,110 MiB), is **-7.9 %** and its own hit rate is 0.9169.  **`cov90`
per-table sizing** (`MOE_HOST_POOL_SIZING=1`) is **+7.5 %** at `FRAC` 80 (17.80 -> 19.14) and **+16.4 %** at
`FRAC` 60 (13.60 -> 15.83) at *identical* pinned bytes (disk reads -15/-23 %, pool hit rate 0.62 -> 0.68 and
0.34 -> 0.44), for **-13/-15 % prefill** — the one trade, recorded in §3.3.  Both implemented as
`patches/0001-0002` (gated, not yet in the delivery set), with the two diagnostic fixes: the teardown report now
describes the run instead of an emptied cache, and a forced `DIO` logs its mode.

**Greedy text across modes:** pinned / `pool` D2D / `pool` H2D all give `270 chars sha=cacea85d5705` at `-n 64`
(hash with **no** `-lv`; `-lv 3` pollutes the extracted span), and so do the FRAC 60 uniform / FRAC 60 cov90 /
FRAC 100 default arms — the sizing changes are layout-only.

What session 2 changed in the campaign's plan:

- **F5 re-scoped**: per-read residency (`mincore`) is a *sub-full-pool / cgroup* feature, not the reference-arm
  lever; the page-cache **budget** is the load-bearing half (the unbounded buffered arm is a thrash source).
- **F2 fixed** in `scripts/moe-trace-replay.py` (`--score-band` default `decode` + the docstring's caveat), so
  §9.6's simulate-before-implement is unblocked and a sizing candidate can finally be screened.
- **Protocol**: `-lv 4`, not `-lv 3`, for anything out of libllama (§9.5).
- **§4.2's sizing candidate was killed as specified and replaced** by a concentration-based one that screens at
  -42 % disk bytes/token (§4.2 note, `HANDOVER.md` §4/F6) — the first candidate the replayer has been able to
  bless since it calibrated.

## 13. Session state

**Next session: [`HANDOVER-MULTIGPU.md`](HANDOVER-MULTIGPU.md)** — a cold-start handover for the multi-GPU box:
state, the ordered work list (W1 multi-device arms → W2 the tensor-split-aware remap → W3 the `RAM_PCT` clamp →
W4 the head-to-head), the traps, the gates and the promotion checklist.  The 1-GPU host's work is finished as far
as it can go.

**Current state (2026-11-03, session 4 — the MULTI-GPU box; see §18):** the campaign set is **five** patches on
the r206 tip (`0005` = W3, the `RAM_PCT` proportional ceiling), applied tree
`5fb99c1c9e45a485fc8780898fecfcb77b0544af`; F7 is **resolved as an
environment artefact** (the "`pinned` gets no cache under `-sm tensor`" claim was `MOE_EXPERT_CACHE_MIB=0`); W3 is
**done** (`RAM_PCT=70`: +59 %/+73 % decode at identical pinned bytes); W1's gather assertion **fails** on the pair
(`pinned` −47 % prefill) and the gather is **mandatory** for `pool` at 3 devices (prefill 16.85 → 1126.8 t/s,
`TODO` #131); and the pair's remaining −34 % decode is **86 % synchronous demand reads** (`TODO` #132, ~25-30 %
overlap ceiling).  Nothing is promoted: `patches/` (delivery) and `release.json` are untouched.  The 1-GPU
session-3 record below is unchanged and remains the reference for the single-GPU regime.

**Session 3 state (2026-11-02 — the 1-GPU host; see §14 for the measured record):** the campaign set is
now **four** patches on the r206 tip (applied tree `ac580a8e3`): the teardown-report / forced-`DIO` diagnostics,
the full-coverage default (`FRAC 100`) + `cov90` per-table sizing, and — new this session — **mechanism (b), the
pool-aware device gather** (`patches/0003`), which takes the srv cell's prefill from 641-647 to **832-878 t/s**
(pool ahead of `pinned`'s 739) and removes the `MOE_HOST_POOL_COLD` dependency.  Mechanism (a) was **measured and
dropped** (exposure ≤0.3 % of wall clock), the pool's old `285.3` srv-cell reference is **withdrawn** (a
window-beta artefact), `RAM_PCT` 85 % does **not** thrash on the PLE-free control, and decode is still **-11 %**
(srv cell) to **-22 %** (122B) behind `pinned`, so the acceptance criterion is **not met yet** (§14.9).  Nothing
is promoted: `patches/` (delivery) and `release.json` are untouched.

**Where the work is saved:** [`patches/README.md`](patches/README.md) (the two patches `0001`/`0002`, apply order on
the r206 tip, knobs, expected numbers, caveats — validated: the applied tree is `0b8cc5a8b`) and
[`HANDOVER.md`](HANDOVER.md) §3.3/§4.  **Not promoted**: `patches/` (delivery) and `release.json` are untouched.
The next step the maintainer has planned is a head-to-head against the parallel `--host-experts mmap` /
pageable-master approach, for which the protocol is prepared in [`HEAD-TO-HEAD.md`](HEAD-TO-HEAD.md).

The cold-start handover is [`HANDOVER.md`](HANDOVER.md): §0 is the one-line status, §3 the measured baseline,
§4 the findings (F3 resolved, F2 resolved, F1/F4/F5 current), §5 the `DIO` analysis plus the measured result,
§6 the ordered next steps (copy kind -> residency+budget -> remap-on-pool), §8 the open questions and the
decisions taken overnight, §9 the in-repo data/tools (`data/20261102-session2-*`, `tools/run-arms-*.sh`,
`tools/replayer-calib.py`).

## 14. Measured — session 3, 2026-11-02, the 1-GPU host (re-based, mechanism (b) landed)

Box: **one** R9700 (32.6 GiB), gfx1201, ROCm 10.1.0, 9800X3D, 93 GiB RAM, `/llm` on NVMe.  Tree: the
delivery tip `efaa60db0` + the campaign set, **re-based and extended** (see §14.1).  Data:
`data/20261102-session3-*`, runners `tools/run-arms-1gpu-261102b.sh`.  **This host has ONE GPU** — the
`-sm tensor` x8-pair arm of §9.4 is not runnable here (see §14.5).

### 14.1 The re-base was a no-op, and that is the finding

The set was said to be "validated against tree `0b8cc5a8b`, current tip `efaa60db0`".  `0b8cc5a8b` is the
**applied tree**, not the base: the patches' pre-image blob at `efaa60db0`
(`ggml/src/ggml-cuda/moe-expert-cache.cu` = `334931ecd`) *is* the patch `index` line, so the two patches apply
to `efaa60db0` with no fuzz, no 3-way and **no conflict** (the window campaign never touched this tree — its work
lives only in the archived reference diff).  Verified: fresh worktree at `efaa60db0` + `git am` = tree
`0b8cc5a8b` exactly.

The set now has four patches (README in [`patches/`](patches/README.md)); applying all four to a fresh
`efaa60db0` gives **applied tree `ac580a8e3a6f12ccb5f673dace82b16ec4f1f275`** (four patches after §16).  Per-patch build: 4/4 clean
(`llama-cli`, `llama-server`, `llama-batched-bench`, `llama-perplexity` at each level).  Greedy text on the
re-based build: 150 chars `sha=4ac3a523160f` (the campaign reference).

### 14.2 Phase 1 baseline, 1 GPU — the srv cell (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`)

`srv-test.sh`, persistent `llama-server`, prompt cache off, 3 sequential requests of the 5258-token prose
prompt, 2000-token greedy decode, **request 3** (steady state).  `pool` arms measured **before** any `pinned`
arm (the page-cache-order trap); the interleaved `pool-frac80-afterpinned` control cost only ~1 %, not 4 %.

| arm | prefill req3 | gen | RSS | disk read req3 |
|---|---|---|---|---|
| `pinned`, staging (the charter's reference) | **739.2** | **60.0** | 41.9 GB | 0 |
| `pinned`, gather (`GGML_SCHED_DEVGATHER=1`) | 688.9 | 60.1 | 41.9 GB | 0 |
| `pool` FRAC 80, staging | 646.8 / 641.2 | 52.6 / 53.6 | 75.5 GB | 0 |
| `pool` FRAC 100, staging | 624.5 | 52.8 | 82.9 GB | 0.06 GB |
| **`pool` FRAC 80, gather** | **877.8 / 874.8 / 832.3** | 53.5 / 53.6 / 53.5 | 75.2-75.5 GB | 0 |
| `pool` FRAC 80, gather, `-sm tensor` (1 visible device, i.e. through the meta backend) | 877.8 | 53.6 | 75.4 GB | 0 |

The charter's reference (734.5 / 59.5) reproduces on `pinned`.  **The charter's `pool` reference (285.3 / 48.8)
does NOT reproduce: it is a window-beta artefact.**  That number was taken on the window campaign's beta tree
(`moe_cache_stage_table` dispatching on device-readability, which WORKLOG 261102.008 records as a *regression*
for the pool arm); on the delivery + campaign-patch tree the same arm is **641-647 t/s**.  Treat 285.3 as
withdrawn — pool's prefill was never 2.6x behind `pinned` on this cell.

Two things follow.  (a) `FRAC 100` is **worse** than `FRAC 80` on this cell (624.5 vs 646.8, and 82.9 GB RSS
against a 75.8 GB model): full coverage over-commits here, which is the multi-device caveat in
[`patches/README.md`](patches/README.md) showing up on a single GPU too.  (b) with the gather, **the pool is
faster than `pinned` on prefill** (+18.7 %) and 10.8 % behind on decode.

### 14.3 Phase 1 baseline, 1 GPU — the campaign's reference arm (122B-A10B UD-Q4_K_XL, `-ncmoe 48`)

`llama-cli`, 5258-token prompt, **1862-token greedy decode** (`-n 2000`, the model stops at 1862),
`-c 8192 -ctk/-ctv q8_0 -b/-ub 2048`, arms alternating, ≥2000 tokens per §9.5.  Host experts **67110 MiB**.

| arm | decode t/s | prefill t/s | disk read / run | pool hit |
|---|---|---|---|---|
| `pinned` (2 arms) | **32.82 / 34.20** | **1025.7 / 1067.2** | 64-72 GiB | — |
| `pinned` + gather (2) | 32.72 / 34.15 | 912.2 / 951.2 | 66-73 GiB | — |
| `pool` FRAC 80 (3) | 19.31 / 19.94 / 19.25 | 761.1 / 763.2 / 759.0 | 160-165 GiB | 0.606-0.610 |
| `pool` FRAC 80 + gather (3) | 19.25 / 19.86 / 19.77 | 752.6 / 754.1 / 747.3 | 157-164 GiB | 0.606 |
| `pool` FRAC 100 | 26.74 | 925.7 | 95.4 GiB | 0.9111 |
| `pool` FRAC 100 + gather | 26.67 | 921.1 | 95.2 GiB | 0.9123 |
| `pool` FRAC 80 `MOE_HOST_POOL_COLD=0` | 15.54 | 761.7 | 253.2 GiB | 0.8039 |
| `pool` FRAC 80 `COLD=0` + gather | **19.57** | 755.2 | 157.1 GiB | 0.6048 |

So on the primary arm: `FRAC 80` is **-42 % decode / -27 % prefill** against `pinned`, and `FRAC 100` (the
patch-0002 default) is **-22 % decode / -14 % prefill**.  The gather is neutral here (256-expert tables are
below the 224 MiB always-gather gate, §14.4).  **These are not the soar numbers**: this host's `pinned` arm is
~1.5x session 2's (32.8-34.2 vs 22.3) at the same h=0.754, so the *ratio* is what carries over, not the t/s.

### 14.4 Mechanism (b) — the pool-aware gather: implemented, and it works where tables are large

`patches/0003`.  The kernel gains the third source the handover specified (arena slot / **pool slot** / caller-
filled destination), the gather builds its own two-tier residency map on the compute stream (separate buffers
from staging's `prs_`, uploaded only when the map changes), and an expert in neither tier is copied into the
destination from a pinned source (`expert_host_src`) before the launch — so **no kernel reads the pageable
mapping**, in either mode.  The pinned-master arm keeps its old behaviour (`prs == nullptr`).

* srv cell: prefill **641-647 → 833-878 t/s (+29…36 %)**, decode flat, and it puts the pool ahead of `pinned`.
* 122B: neutral (pool -1 % prefill, decode unchanged); `pinned` **-11 % prefill** when the gather fires (its
  routed union over a 2048-token ubatch is the whole 180 MB table, so the gather just moves the same bytes
  through kernel copies instead of the DMA engine).
* The discriminator is `SCHED_GATHER_TABLE_MIN_BYTES` (224 MiB): Flash-Next's 512-expert tables (360 MB) gather
  at every width, the 122B's 256-expert tables (180 MB) only below the staging width gate.  The win follows the
  always-gather path, not sparsity.
* **OPEN A — answered.**  With the gather, `MOE_HOST_POOL_COLD=0` reads **19.57** against `COLD=1`'s 19.25-19.86,
  i.e. the gather fully substitutes for the in-place pool-slot read (without it `COLD=0` was 15.54, -20 %).  The
  gather does not care about `device_count`, so this is the portable form of that win and a `#266` dependency the
  maintainer can drop.
* Gates: greedy 150 chars `sha=4ac3a523160f` both arms; prefill-logit **mean KLD -0.000000 / same-top-p
  100.000 %** (host-expert arm `-ncmoe 40 --host-experts pool`, gate script `--extra`); four-axis **acceptance
  identical to 5 decimals** (0.81370 / 0.75408 / 0.89175 / 0.98148) with `none` == `draft-mtp` text on all four
  axes; `llama-batched-bench -npl 1,4,8` within run-to-run noise; `check-gpu-faults.sh` clean; and the
  **`GGML_MOE_GATHER_ONCE=1` control reproduces Hole B** (2000 chars, `2ef00a0e243f`, 1326.9 t/s = 1.5x the
  honest arm), so the clean arm is evidence rather than an unfalsifiable pass.

### 14.5 Mechanism (a) — registered-range CE copies: measured, and dropped for pool mode

`moe_cache_upload_pageable` is pool mode's fallback for a table the cache staging declines, and the handover
priced it at prefill 375.9 → 532.6 t/s and -6.7 GB RSS **on the window arm**.  Both numbers are window-staging
artefacts, because in pool mode the staging path does not use this function at all (it has its own pool branch).
Exposure, counted (the counter is part of `patches/0003`):

| cell | ranges | bytes through the bounce | pinned bounce ring |
|---|---|---|---|
| srv cell, 3 requests incl. 5258-token prompt x3 | 44 | **56.4 MiB** | 43.3 MiB |
| 122B, 3 requests (the primary arm) | 1 878 | **3 664 MiB** (0.26 % of a 142 s run) | 831-1 190 MiB |

Implementing (a) would register the whole model mapping (`hipHostRegister`) to save 56 MiB-3.6 GiB of host
`memcpy` per run — ≤0.3 % of wall clock — while taking on the pin-budget/fragility tail §6 and §8 exist to
refuse.  **Recommendation: drop (a) from the pool scope**; it is a window-staging fix, not a pool one.

### 14.6 OPEN B — `RAM_PCT` 85 % does not thrash here; lowering it is a cliff

Bounded probe (not a soak — the soak is the charter's final Flash-Next phase): 122B, `-ncmoe 48`, `FRAC 100`,
2000-token decode, sampling the process's `VmSwap` and the box's `pswpout`.

| `MOE_HOST_POOL_RAM_PCT` | ceiling | tables at full slots | decode | process `VmSwap` | `pswpout` |
|---|---|---|---|---|---|
| **85** (shipped default) | 73.5 GiB | **144/144** | **26.29** | 0 | 125 pages (0.5 MiB) |
| 70 | 60.4 GiB | **129/144** (14 tables starved to **1 slot**) | 15.86 (**-40 %**) | 0 | 0 |

So on the PLE-free control the 85 % default does **not** thrash — and turning it *down* is far worse than
leaving it: 10 % fewer slots costs 40 % of decode, because the per-table clamp starves the *registration-order
tail* tables to the one-slot floor instead of shaving every table proportionally.  The defect is the ceiling's
**distribution**, not its level.  (The Flash-Next/PLE thrash report stays Phase 2.)

### 14.7 `-sm tensor` on the x8 pair — NOT runnable on this host

`rocm-smi`/`lspci` show **one** R9700 (`03:00.0`, x8) plus the Granite Ridge iGPU: `HIP_VISIBLE_DEVICES=0,2`
enumerates a single ROCm device, so the pair arm cannot be built here.  What *was* runnable is the **meta
backend with one simple backend** (`-sm tensor` on the one visible device), and the gather is served through it
(4059 pool-backed calls, 877.8 t/s — identical to the `-sm layer` arm).  **Still owed on `soar`:** a
multi-device `-sm tensor` prefill of `-p 2048+` (never a 32-token smoke), asserting gather ≥ staging per split
mode and `check-gpu-faults.sh` clean.  Note the capability-predicate caveat the handover describes does **not**
exist at `efaa60db0` (§14.8), so that arm is the first thing that would exercise a multi-device decline.

### 14.8 Handover corrections (the code wins)

1. **`moe_cache_gather_ok()` does not exist in this tree.**  That capability predicate was window-campaign code
   (WORKLOG 261102.010/.011); at `efaa60db0` the scheduler's only test is `iface.moe_cache_gather == NULL` plus
   `GGML_SCHED_DEVGATHER` (default **off**).  Consequences: the gather is opt-in, and the "predicate answers no
   after staging was skipped" collapse is reachable only with the switch on — which is exactly why the pool path
   had to be made to *serve* rather than to decline (§14.4).
2. **Mechanism (a)'s numbers do not transfer to pool mode** (§14.5).
3. **The `pool` 285.3 / 48.8 reference is withdrawn** (§14.2).
4. The handover's `-sm tensor` claim stands as a *plan*, not a measurement: pool is split-native and the meta
   forwarding works (measured with one device), but the multi-device arm is unrun.

### 14.9 Recommendation

**The acceptance criterion is not met on decode yet.**  On this host, with mechanism (b) landed: prefill is at
or above `pinned` on the large-table cell (pool 877.8 vs `pinned` 739.2) and only -14 % on the 122B, but decode
is still **-11 %** (srv cell, FRAC 80) to **-22 %** (122B, FRAC 100) behind `pinned` — and PR #267's `mmap` was
*equal* to `pinned` (74.7 vs 72.4 t/s, 1189 vs 1278 prefill), so `mmap` still offers an advantage on decode.
What changed is that the *prefill* half of the acceptance argument is settled and the gather removed a `#266`
dependency (`MOE_HOST_POOL_COLD`).  The remaining decode gap is the pool's own demand path, i.e. P0-a
(`GGML_MOE_DEVREMAP` over pool slots), which this session did not start.

## 15. Step 0 — where the decode gap actually is (measured 2026-11-02, later the same session)

The plan was "close the ~8 % `GGML_MOE_DEVREMAP` gap on pool tables (P0-a) and then re-measure".  **Step 0 says
that is the wrong order**: the remap is worth a third of what the campaign assumed on this host, and ~18 % of the
pool's decode deficit is not any of the explanations on the table.  All arms below are the 122B reference arm
(`-ncmoe 48`, 5258-token prompt, 1862-token greedy decode), alternating, same binary (the three-patch tree, plus
`patches/h2d-experiment.patch` applied to the working tree for the H2D arm and reverted after).

| arm | decode t/s | prefill t/s | disk/run | arena hit |
|---|---|---|---|---|
| `pinned` (remap on, the delivery default) | **32.77** | 1024.0 | 70.1 GiB | 0.7544 |
| `pinned`, `GGML_MOE_DEVREMAP=0` ×2 | **31.73 / 31.63** | 1026.2 / 1007.1 | 72.7 / 64.5 | 0.7537 |
| `pool` FRAC 100 ×2 | **26.12 / 25.73** | 918.1 / 920.7 | 101.2 / 101.0 | 0.9131 / 0.9136 |
| `pool` FRAC 100 + `MOE_HOST_POOL_H2D=1` ×2 | 25.90 / 25.85 | 915.6 / 914.8 | 95.1 / 95.1 | 0.9132 / 0.9139 |

1. **The remap is worth ~+3.3 % here, not +8.1 %** (32.77 vs 31.68 mean).  P0-a's whole ceiling on this arm is
   ~1.1 t/s.  It stays worth doing for `#266`'s sake, but it is **demoted from the campaign's top item** — the
   r203 +8.1 % was measured on a different box, a different model and the 35B-A3B cell.
2. **The copy kind is nil** (25.90/25.85 vs 26.12/25.73), confirming session 2's H2D A/B — the D2D-from-alias
   hypothesis is dead a second time, now at full coverage.
3. **~18 % remains unexplained** (25.9 against 31.7 with the remap excluded).  And it is *not* traffic: identical
   arena (22517.6 MiB), identical reaches (2 150 688/turn), the pool arm has **more** arena hits, moves **fewer**
   bytes (943.8 GiB vs 962.2 GiB of expert traffic) and reads **fewer** bytes from disk.  So it is not coverage,
   not copies, not the page cache.

### 15.1 The first instrument that localises it: a per-kernel trace

`perf` on the host thread is uninformative (the process is mostly idle: 64 % of its CPU cycles are ROCr's
`AsyncEventsLoop`, 22 % `__memmove`, i.e. no host bottleneck to remove).  `rocprofv3 --kernel-trace` on a
**pure-decode** workload (`-c 4096`, `-n 128`, short prompt) does localise it — same graph, same dispatch
counts, per-kernel durations compared:

| kernel | pool s | pinned s | n (both) | per-dispatch pool / pinned |
|---|---|---|---|---|
| `mul_mat_vec_q_moe<q4_K,2>` | 0.492 | 0.435 | 6 296 | 78.1 / 69.1 µs |
| `mul_mat_vec_q_ksplit<Q8_0,…>` | 0.434 | 0.413 | 5 295 | 81.9 / 78.0 µs |
| **`__amd_rocclr_fillBufferUnAligned`** | **0.162** | **0.035** | 441 | **367.2 / 78.5 µs (4.7x)** |
| `__amd_rocclr_copyBuffer` | 0.065 | 0.029 | 20 691 / 13 281 | 3.1 / 2.2 µs |
| `mul_mat_q<q8_0,16>` | 0.029 | 0.013 | 297 | 97.0 / 42.8 µs |
| total GPU kernel time | **1.932** | **1.629** | 104 116 / 96 706 | (+18.6 %, matching the wall gap) |

Read it as: **the wall gap is GPU work, and 42 % of the delta is one runtime kernel** (`fillBufferUnAligned`,
+0.127 s of the +0.303 s) **whose dispatch count does not change at all** — plus **58 extra `copyBuffer` calls
per decode step** (161 vs 103; the pool's fetch path is per-expert) and a 5-16 % per-dispatch slowdown on the two
routed matvec kernels.  `fillBufferUnAligned` at 367 µs is far too big to be a guard pad; identifying what fills
~MBs per step in the pool arm and why the same 441 dispatches cost 4.7x is **the next concrete action**, ahead of
P0-a.

### 15.2 What this means for the plan

- **P0-a is demoted** (~1.1 t/s ceiling here).  It is still the piece `#266`'s removal needs, so it is worth doing
  eventually — but it is no longer the thing that moves the acceptance criterion.
- **The new P0 is the residual 18 %**, starting with `fillBufferUnAligned`.
- The measurement also retires one more campaign assumption: `MOE_EXPERT_CACHE_COLD=uva` is default-on and
  **20.5 % of reaches are served cold** (441 399 cold reaches/turn, 801 GiB of the 944 GiB of expert traffic) in
  *both* modes — so the cold path, not the fill path, is where the bytes are.  Anything that changes cold-read
  behaviour is a decode-band change and needs the same gates.

The trace CSVs are 40-75 MB each and are **not** committed; regenerate a pair and summarise them with
`tools/rocprof-summarise.py k-pool_kernel_trace.csv k-pinned_kernel_trace.csv`, which reproduces the table above
(use a short prompt and `-n 128`, or a prefill floods the trace with `moe_cache_stage_kernel`).

**Instrument note (for the next session):** `perf record` works here without `--call-graph` (`perf_event_paranoid`
is 2, so `-g` is refused — do not raise it silently); `rocprofv3 --kernel-trace -f csv` works and the per-kernel
durations are comparable **only** as durations (the tracer distorts the achieved rate badly: 13.3 vs 27.2 t/s on
the same two arms, so never quote a t/s from a traced run).

## 16. The bounce ring and the synchronous fill — measured, and what they decide (2026-11-02, session 3)

Chasing the §15 residual (a fill kernel 4.7x slower in the pool arm) produced two measurements that matter and
one negative result.  Data: `data/20261102-session3-fillcost`, `data/20261102-session3-ring`, `kt*` traces.

### 16.1 The 4.7x fill is not ours — it is ROCclr internal, and it is a timeline artifact

`rocprofv3 --kernel-trace` reports 441 `__amd_rocclr_fillBufferUnAligned` dispatches in **both** arms (144 with
grid 8192 ~ 2 MiB, 288 with grid 256 ~ 64 KiB, 8 on a worker stream), in **identical counts**, in pairs, spread
over the whole run, on stream 0 (`cudaStreamPerThread`), with **no overlap** against compute (0/50 sampled) and a
GPU concurrency factor of 1.00.  Only the *durations* differ (2.4 vs 1.4 us; median 367 vs 231 us; pool has a
heavy tail to 3.5 ms).

It is **not** a ggml memset: env-gated instrumentation of both `ggml_backend_cuda_buffer_memset_tensor` and
`..._buffer_clear` logs **8 calls in an 8-token run and zero `memset_tensor` calls**, while the tracer counts 441
fills on the same arm.  So the fills are inside the ROCr runtime (the same family as `__amd_rocclr_copyBuffer`,
which the pool arm calls **58 more times per decode step**: 161 vs 103).  **Conclusion: §15.1's "42 % of the
residual is one runtime kernel" is a per-stream duration sum, and the honest reading is "the pool arm issues more
copy work and its timelines are longer", not "a fill kernel regressed".** Do not chase it further without a
per-stream busy-union analysis; `tools/rocprof-summarise.py` flags the "same count, different duration" rows.

### 16.2 The pinned bounce ring: 2.2-3.1 GiB, from the prefill miss path, and capping it is a loss

`moe_cache_upload_pageable` had its own acquire/grow loop, so the reported ring total was under-counted (it never
saw `bounce_acquire`, which `expert_host_src` also uses).  Instrumented honestly:

| arm | ring total | notes |
|---|---|---|
| FRAC 100 | 2209.7 MiB | 732 `cudaHostAlloc`s in an **8-token** run, ~274 us median |
| FRAC 80 | 3107.7 MiB | |

`bounce_acquire` is now the ring's only owner (`patches/0004`), so the number is true.  **A cap was implemented,
swept and reverted** — it is the charter's §4.4 / PR #267's 0010 idea, and the measurement says no:

| cap | decode (FRAC 100) | prefill (FRAC 100) | prefill (FRAC 80) | ring |
|---|---|---|---|---|
| none | 26.12 / 25.73 | 918 / 921 | **759-763** | 2209-3108 MiB |
| 8 | 25.99 / 26.07 | 887 / 916 | **573-644** | 1719 MiB |
| 4 | 25.51 / 25.66 | 870 / 891 | — | 1009 MiB |
| 2 | 25.64 / 25.63 | 846 / 839 | — | 524 MiB |

The ring's memory is the **prefill miss path** legitimately holding one entry per expert in flight while a
whole table's miss set is read; the scheduler-copy path never exceeds ~5 entries, so capping *it* is inert
(waits = 0).  A control build of the parent commit measures FRAC 80 at 720.1 prefill / 18.52 decode against this
tree's 721.5 / 18.56, i.e. the surviving change is neutral.  (The 759-763 baseline was taken hours earlier; the
~5 % drift between them is box state, not code — hence the control.)

### 16.3 The synchronous fill: 3-4 ms per decode token, ~6 % of the step, fully serialised

`pool_fill_slot_locked` now times its two sources (`patches/0004`), which finally sizes §4.1/C's candidates:

| arm | storage reads | read wall time | bytes | effective | mapping memcpys |
|---|---|---|---|---|---|
| FRAC 80 | 38 836 | **27.3 s** | 69.1 GiB | 2.6 GB/s | 0 |
| FRAC 100 | 44 980 | **34.5 s** | 80.0 GiB | 2.4 GB/s | 0 |

Every fill is one expert (1.8 MiB on this model), the reads are 100 % serialised with compute (the host thread
blocks inside the fill), and per decode token that is **~3-4 ms of a 52 ms step (~6 % at FRAC 80)** — after
subtracting the pool's one-shot prepopulate (29 376 / 36 864 of the fills), not the ~19 % an earlier
`/proc/diskstats` estimate suggested (that estimate wrongly counted the model load and the prepopulate).

**What this decides for the two candidates:**

* **Coalescing (deterministic, known-sequence readahead): SCREENED AND REFUTED — see §17.**  O_DIRECT on the
  real shard (`00002-of-00003`, 46 GiB, 256 MiB spans) reads **587 MB/s at our 1792 KiB fill size** and
  **1.5-1.7 GB/s at >= 4 MiB** — so per-read size does matter at the device level, even though the in-situ
  sequential stream already achieves 2.6 GB/s (queue depth / locality).  The instrument is now in place
  (`pool fills:` line): a coalesced arm must beat 27.3 s, and it also shrinks the ring by the same factor, which
  is the only *bounded* way to fix §16.2.  This is the same shape as the external "pipelined staging" proposal
  ([`PROPOSAL-pipelined-staging.md`](PROPOSAL-pipelined-staging.md)) and it is the part of it that survives.
* **Overlap (a demand-fill worker): a hard ~6 % ceiling at FRAC 80 and ~0 at FRAC 100** (hit rate 0.91, the pool
  is full and the residual fills are the tier's own churn), against two measured negative precedents (r113's
  worker; `MOE_HOST_POOL_PREFETCH`, now known to be a dead parse — `TODO.md` #268).  Not worth a thread on this
  evidence; revisit only if the coalesced arm shows the fill still dominating after the read time halves.

## 17. Coalesced multi-expert fills — built, measured, REFUTED (2026-11-02, session 3)

§16.3 said "worth screening, not obviously free".  It was screened, and it loses.  Record so nobody builds it
again: `data/20261102-session3-coalesce`.

**What was built** (and then reverted — nothing of it remains in the tree): the staging path's miss loop walks
experts in ascending order, so a *run* of consecutive misses was read with ONE aligned pread of up to 16 MiB into
a shared 4-slot pinned scratch (64 MiB process-wide, bounded, event-tracked) and issued as ONE H2D, instead of
one pread + one per-expert bounce entry each.  Kill switch `MOE_HOST_POOL_COALESCE=0`; the 1:1 geometry only.

**Why it fails: the miss set is not contiguous.**  The pool's residency is **heat-ranked**, not expert-ordered, so
"the experts this table does not hold" is a *scattered* subset of the file — and the run finder therefore produced
runs of ~1.2 experts:

| arm | prefill | decode | coalesced coverage | read time (all fills) | ring |
|---|---|---|---|---|---|
| FRAC 80, coalesce **on** | **590.1** | **16.24** | 3 503 runs, 7.7 GiB of 69 GiB | 41 229 ms | 2761 MiB |
| FRAC 80, `=0` | **629.8** | **17.65** | — | 29 415 ms | 2946 MiB |
| FRAC 100, on | 896.2 | 25.48 | 385 runs, 0.7 GiB of 80 GiB | 40 191 ms | 2202 MiB |
| FRAC 100, `=0` | 912.9 | 25.25 | — | 40 628 ms | 2255 MiB |

So at FRAC 80 it costs **-6 % prefill and -8 % decode** (the scratch churn plus a mostly-single-expert fallback),
and at FRAC 100 it is inert (0.7 GiB of 80 GiB coalescable).  And the second half of the idea — the *prepopulate*,
which is 29 376-36 864 of the fills — cannot coalesce either: it fills slots in **rank** order, so its sources are
scattered by construction.

**And the read path is already at device speed.**  In situ the fills run at **2.4-2.6 GB/s**, which is *above* the
`dd iflagn=direct` ceiling measured on the real 46 GiB shard (587 MB/s at our 1792 KiB size, 1.5-1.7 GB/s at
>= 4 MiB): thousands of back-to-back preads already present the device with a deep sequential stream.  There is no
bandwidth left to recover by making individual reads bigger.

**Consequences.**  §4.1's P1-a is closed on the evidence: neither coalescing (refuted above) nor an overlap
worker (~6 % ceiling at FRAC 80, ~0 at FRAC 100, two measured negative precedents — §16.3) is worth building.
The remaining decode work is P0-a (`GGML_MOE_DEVREMAP` over pool slots, `TODO` #269, ceiling +3.3 % here, and it
must be **tensor-split aware**), the `soar` multi-device `-sm tensor` arm, and the §5 knob collapse.

## 18. Measured — session 4, 2026-11-03, the MULTI-GPU box (`soar`): W1 and W3 done, W2 still open

Box: `soar`, 3 × R9700 gfx1201 (32.6 GiB each) + the Granite Ridge iGPU (never pinned in), 91 GiB RAM,
ROCm 10.1.0, `/llm` on NVMe (`nvme2n1`).  Tree: the delivery tip `efaa60db0` + the campaign set (now **five**
patches, `0005` = W3 added this session; per-patch build clean, applied tree
`5fb99c1c9e45a485fc8780898fecfcb77b0544af`).  Protocol: the
campaign's reference 122B-A10B UD-Q4_K_XL, `-ncmoe 48 -ngl 99 -c 8192 -ctk/-ctv q8_0 -fa auto -b/-ub 2048`,
5258-token prose prompt, **2000-token greedy decode** (`-n 2000`, the model stops at ~1860-1793), `-lv 4`,
`--spec-type none --seed 42 --temp 0`, arms alternating in one sequence, `pool` arms before `pinned` arms in
each group (page-cache order), never two benchmarks at once.  Data:
`data/20261103-multigpu/{w1-pair,w1-3gpu,w3-before,w3-after,w3-after2}` and `data/20261103-text-purity` (both
device counts in one directory — the text runner takes the output directory as a positional argument); runner
`tools/run-arms-multigpu-261103.sh` (`VIS`/`SM` select the device set and split), text oracle
`tools/run-text-purity-multigpu-261103.sh`.

### 18.1 F7 is RESOLVED — it was an env artefact, there is no `-sm tensor` registration gap

F7 (`HANDOVER.md` §4) said that under `-sm tensor` on the pair `pinned` never sizes the expert cache, so its
14.46 t/s was a no-cache rate.  Reproduced exactly, and the cause is **`MOE_EXPERT_CACHE_MIB=0` in that arm's
environment**, not a device/Meta registration defect:

| arm (pair, `-sm tensor`, `-ncmoe 48`, 5258-token prompt, `-n 8`) | prefill | decode | decode-graph splits | arena | slots/table |
|---|---|---|---|---|---|
| `pinned`, `MOE_EXPERT_CACHE_MIB=0` (what F7 measured) | 887.2 | **14.40** | 98 | **none** | — |
| `pinned`, shipped default (AUTO) | 1614.5 | **30.06** | 50 | **49808.2 MiB** | **190** (×2 devices) |

`MOE_EXPERT_CACHE_MIB=0` disables the cache, and with the cache off `offload_op` falls back to
`get_op_batch_size(op) >= op_offload_min_batch_size`, i.e. a 1-token decode leaves the routed `MUL_MAT_ID` on the
CPU (98 splits instead of 50, no slab, no arena, no `alloc_all_locked`).  With the shipped default the whole
multi-device path is healthy — instrumented on this tree:

```
TEMPINSTR preflight: moe_host_expert_dev_count=2      (ROCm0 33555.0 MiB, ROCm1 33555.0 MiB)
TEMPINSTR account blk.2.ffn_down_exps.weight: buft=ROCm_Host buft_dev=ROCm0 layer_dev=Meta() meta=1 n_experts=256
alloc_all_locked: device 0/1: sized 190 slots/table from 144 tables / 131.1 MiB per expert
alloc_all_locked: per-device sizing: 2 device(s), arena total 49808.2 MiB
```

So the split accounting, the preflight and the per-device arena all work; `-sm tensor` + host experts is **not**
a no-cache shape.  Two consequences: (a) the pair's `pinned` baseline in §18.2 is a *cached* baseline, which is
what W1 needed; (b) **`MOE_EXPERT_CACHE_MIB=0` on a multi-device tensor split is a trap worth documenting** — it
silently moves every routed expert to the CPU and halves decode, and the only visible symptom is the split count
and the missing arena.  (`TORCH`-style "disable the cache to isolate it" A/Bs must use `-ncmoe 0`, not `MIB=0`.)

### 18.2 W1 — the multi-device arms, and the gather's real shape

Pair (`HIP_VISIBLE_DEVICES=0,2`, in-process `0,1`), `-sm tensor`:

| arm | decode t/s | prefill t/s | arena `h` | disk / run |
|---|---|---|---|---|
| `pinned` × 4 | **39.72 / 39.72 / 39.85 / 39.87** | 1603.8 / 1568.0 / 1574.0 / 1647.8 | 0.9477 | 71-73 GiB |
| `pinned` + gather | 39.81 | **832.5 (-47 %)** | 0.9476 | 74.2 GiB |
| `pool` FRAC 80, staging | 26.08 | **1024.1** | 0.9544 | 212.8 GiB |
| `pool` FRAC 80 + gather | 26.11 | 991.5 (-3 %) | 0.9544 | 214.2 GiB |
| `pool` FRAC 80, `SIZING=1` (cov90) | **28.02 (+7.4 %)** | 698.7 (-32 %) | 0.9544 | 202.1 GiB |
| `pool` FRAC 100 (RAM ceiling clamped it to 35944/36864 slots — **pre-W3 build**) | 26.58 | 1333.9 | 0.9544 | 203.2 GiB |
| `pool` FRAC 100 (same arm, **W3 build**, ceiling needs no shave: 36864 slots) | **29.08 (+9.4 %)** | 1316.9 | 0.9547 | 185.5 GiB |
| `pool` FRAC 95 (W3 build, 34992 slots / 63702 MiB) | **32.02** | **1397.1** | 0.9547 | 143.7 GiB |

3 GPUs (`HIP_VISIBLE_DEVICES=0,1,2`), `-sm tensor`:

| arm | decode t/s | prefill t/s | arena `h` | disk / run |
|---|---|---|---|---|
| `pinned` × 2 | **51.97 / 52.04** | 564.7 / 606.1 | 0.9863 | 75-76 GiB |
| `pinned` + gather | 51.71 | 587.8 (+4 %) | 0.9865 | 73.6 GiB |
| `pool` FRAC 80, staging | 49.99 | **16.85 (!!)** | 0.9859 | 330.9 GiB |
| `pool` FRAC 80 + gather | **49.48** | **1126.8 (+66×)** | 0.9855 | 226.8 GiB |
| `pool` FRAC 80 + cov90, staging | 48.97 | 24.01 | 0.9839 | 301.4 GiB |
| `pool` FRAC 95, staging | **51.03** | 15.03 | 0.9835 | 317.5 GiB |
| `pool` FRAC 80, `STAGE_BATCH=0` | 49.25 | 31.99 | 0.9854 | 291.3 GiB |
| `pool` FRAC 80, `DIO=0` (buffered) | 50.75 | 8.12 | 0.9833 | 227.8 GiB |
| `pool` FRAC 30, staging | 47.25 | 206.30 | 0.9853 | 294.3 GiB |
| `pool` FRAC 100 (clamped to 63832 MiB), staging | 45.44 | 11.20 | 0.9868 | 357.6 GiB |

**Finding A — W1's assertion fails on the pair, for the documented reason.**  `gather ≥ staging per split mode`
does **not** hold at 2 devices: on `pinned` the gather costs **-47 % prefill** (832 vs 1604), and on the pool it
is neutral (-3 %).  This is the same mechanism the 1-GPU 122B showed at -11 % (§14.4): a 256-expert 180 MB table
is under `SCHED_GATHER_TABLE_MIN_BYTES` (224 MiB), so only sub-gate ubatches gather, and the gather then moves
the same bytes through kernel copies instead of the DMA engine.  **The gather is therefore not a general prefill
win — it is a *large-table* win and a *pool-staging rescue*.**  On the pair it must stay opt-in.

**Finding B — the pool staging path collapses at THREE devices, and the gather is the fix (new P0).**  At 2
devices the pool's prefill is 1024 t/s (fine); at 3 devices the same code path stages the *same* volume (846
stage ops, 122 GiB "arena" + 9.4 GiB "host") in **312 s instead of 5 s** — 16.85 t/s against `pinned`'s 564.65.
The `pinned` arm at 3 devices stages the identical 122 GiB through the same staging kernel at full speed (790 ops,
0.8 GiB from the host), so the defect is in the **pool branch of `moe_cache_stage_table`** (`!t.host_pinned`), not
in the staging kernel, the split, or the arena.  Turning the gather on takes the same arm to **1126.8 t/s**
(+66x, +100 % over `pinned`).  It was then probed with every knob that could plausibly be the cause, and the
answer is "it is not any of them" — all at 3 devices, FRAC 80 unless stated (`data/20261103-multigpu/w1-3gpu`):

| probe | prefill t/s | what it rules out |
|---|---|---|
| `pool`, staging (baseline) | **16.85** | — |
| `pool`, `STAGE_BATCH=0` (no staging kernel; per-expert copies) | 31.99 | the single-launch kernel / the pool-alias read *inside* it |
| `pool`, `MOE_HOST_POOL_DIO=0` (buffered) | **8.12** | `O_DIRECT`; buffered is *worse* (page-cache thrash) |
| `pool`, `FRAC 95` (5 % uncovered) | 15.03 | the per-uncovered-expert miss loop (nearly everything is covered) |
| `pool`, `FRAC 30` (19.9 GiB pool) | **206.30** | — but see below: it scales with the POOL SIZE |
| `pinned` (same 3 devices) | 564.65 | the split, the all-reduce, the staging kernel itself |
| `pool` + gather | **1126.80** | the whole path: the gather does not use it |

So the cost is monotone in the **pinned pool size** (19.9 GiB → 206 t/s, 53.5 GiB → 16.85, 63.7 GiB → 15.03) and
it is confined to the first ubatches (310 s for the first 3210 of 5258 prompt tokens, 1.8 s for the rest).  **This
is a delivery defect in block 12's pool mode under a ≥3-device tensor split**, not a campaign-patch regression
(the arm is the pre-existing staging path).  Tracked as `TODO` #131; the gather (`patches/0003`,
`GGML_SCHED_DEVGATHER=1`) is the working remedy, and `--host-experts pool` on 3+ devices should be documented as
needing it until #131 is fixed.

**Finding B, diagnosis (2026-11-03, later — and it is NOT the staging copy).**  Instrumented the staging entry
point per call (device, pool-backed or not, bytes, host wall split into map-build / miss-loop / kernel, and an
optional forced `cudaStreamSynchronize` on the copy stream), and cross-checked with a per-kernel trace.  Same 3
devices, 512-token prompt, `-ub 2048`:

| arm | prefill t/s |
|---|---|
| `pinned`, cache on | **140.1** |
| `pool` FRAC 80, cache on | **11.8** |
| `pool` FRAC 80, `GGML_MOE_CACHE_STAGE=0` (staging path off) | 16.1 |
| `pool` FRAC 80, `MOE_EXPERT_CACHE_MIB=0` (no arena, no cache) | **50.4** |

* **The staging function is not the cost.**  423 staging calls cost **1.04 s of host time in total** (miss loop
  0.76 s); with `GGML_INSTR_STAGE_SYNC=1` the staged copies themselves add **0.66 s**.  Of a 273-419 s prefill
  that is noise.
* **The GPU kernel time is the same in both modes**: `rocprofv3 --kernel-trace`, same prompt, total kernel time
  **5.04 s (pool) vs 4.08 s (pinned)** over 70 283 / 66 842 dispatches, with matching dispatch counts per kernel
  (only `__amd_rocclr_fillBufferUnAligned` differs: 3574 vs 1846 — ROCclr's buffer *fill*, i.e. ~1700 extra
  allocation/fill operations in the pool arm, ≈4 per staging call).
* **The host burns 2.3x the CPU** for the same work: `perf stat` task-clock **83.2 s (pool) vs 35.8 s (pinned)`,
  with major faults 44 138 vs 870 (that half is a *load-time* artefact: in pool mode the experts stay in the
  page-cache mapping, in `pinned` they are pread'd into the pinned master).
* **Turning the arena off recovers 4.3x** (`MIB=0` → 50.4) while turning only the staging path off does not
  (16.1), so the cost lives in the cache's **per-(table, device) prefill work around** the staging copy.

That "next measurement" was superseded: the per-table `cudaMalloc(t.prs_dev)` cost 0.00 s in total (423 calls,
max 64 us) and the staging copy 1.6 s, so neither is the cost.  With the gather as the default the *prefill* is
fixed (see the resolution below); the staging path's own cost is still unexplained and is only reachable through
`GGML_SCHED_DEVGATHER=0` now — `TODO` #131 keeps that half open rather than closing it as "worked around".

**Finding C — the pool is at pinned parity on decode exactly where its tiers are large enough.**  The 3-GPU
`-sm tensor` shape gives the arena everything (`arena 67110 MiB = 100 % of the host expert set`, `h≈0.986`) and
the pool arm then decodes **49.48-49.99 vs `pinned` 51.97-52.04 (-4.5 %)**, i.e. inside PR #267's own
`mmap`-vs-`pinned` envelope (72.4 vs 74.7 = -3.1 %).  On the pair the arena is 49.8 GiB for 67.1 GiB of experts
(`h=0.95`), the pool carries the rest, and the deficit is **-34 %** — see §18.4.

**Finding D — coverage is still the pair's decode lever, but the ceiling must leave the model a page cache.**
On the W3 build the pair's `pool` arms are FRAC 80 **26.08** < FRAC 100 **29.08** < FRAC 95 **32.02** (against
`pinned` 39.85): more coverage helps until the pinned pool starts evicting the model's own pageable set —
FRAC 95 pins 63.7 GiB and reads 143.7 GiB from disk per run, FRAC 100 pins 67.1 GiB and reads 185.5 GiB, i.e.
4 GiB more pinned costs 42 GiB more disk and 9 % of decode.  That is §3.2's "the ceiling does not reserve the
model's own page cache" showing up as a decode effect on a multi-device split, and it also means the *pre-W3*
`FRAC 100` arm understated the pool by 9.4 % because the old tail shave had taken 920 slots off the last tables.

**Finding B, RESOLVED (2026-11-03, `patches/0006`) — a pool-backed table now gathers in the prefill band, by
default.**  The staging path is the defect (above); the device gather is the remedy, and it is now routed
automatically for `--host-experts pool` tables in the prefill band.  Same box, same 122B, 5258-token prompt,
`-ub 2048`, **2000-token greedy decode** (`data/20261103-multigpu/w1-fix/`):

| arm (3 × R9700, `-sm tensor`, `-ncmoe 48`) | prefill t/s | decode t/s |
|---|---|---|
| `pinned` | 471.8 | **67.6** |
| `pool` FRAC 80 (gather, new default) | **1502.6** | 67.1 |
| `pool` FRAC 95 (gather, new default) | **1569.2** | 67.1 |
| `pool` FRAC 95, `GGML_SCHED_DEVGATHER=0` (kill switch) | **16.9** | 27.3 |

so on the box's own three-device shape the pool is now **-0.7 % decode and 3.2x `pinned`'s prefill** — inside
(and past) PR #267's own published envelope (§1: `pinned` 74.7/1189 vs `mmap` 72.4/1278).  The change is
neutral elsewhere: pair `pinned` 1671.6 prefill (unchanged, it never gathers), pair `pool` 948.6 (staging was
1024.1, i.e. -7 % of a path that was never broken there), 1-GPU `pool` FRAC 100 734.2 vs `pinned` 755.5.  The
first attempt routed *all* pool-table gathers with no band gate and decode fell to 1.2-4.1 t/s — the decode band
is served from the arena by the slot-remap read, and the gather must not take it over (`sched_input_gatherable`
now requires `n_tokens > moe_cache_band`).

**The pair, with the fix (2026-11-03, 2000-token greedy, 5258-token prompt, `-sm tensor`):** `pinned` 1643.9
prefill / **49.0** decode; `pool` FRAC 80 948.2 / **29.6**; `pool` FRAC 95 1506.2 / **38.5** — text identical across
all three (`6079 chars sha=01607c16b483`).  So the fix lifts the pair's pool prefill 1024 -> 1506 t/s (+47 %) but
**the acceptance criterion is met only on the three-device shape** (-0.7 % decode there) and **not on the pair
(-21 % at FRAC 95)**, where the residual is the synchronous demand read (`TODO` #132), not the gather.

**Is the prefill number real, or is it corruption?**  The code's own history says a gather that looks faster
than the link can carry is skipping work (`SCHED_GATHER_TABLE_MIN_BYTES`'s comment: the 2026-10-04 numbers were
"from a *corrupted* gather pass"), so it was attacked with the delivery's prefill oracle rather than with
throughput:

| check (`soar`, 122B `-ncmoe 48`, `-sm tensor` over 3 GPUs, pool FRAC 95) | result |
|---|---|
| **`scripts/gate-prefill-logits.sh --ab "GGML_SCHED_DEVGATHER=0"`** — prefill logits of the gather default vs the staging path, `-ub 2048`, ctx 2048 × 6 chunks, kv f16 | **mean KLD −0.000000, same-top-p 100.000 %**, base PPL 3.5537 on both arms → **bit-identical prefill** |
| greedy text, 2000 tokens, behind a 5258-token multi-ubatch gather-served prefill | `5835 chars sha=dc178c29fe89`, identical for `pinned` / `pool` F80 / `pool` F95 |
| greedy text, 64 tokens, 2 device counts × 4 modes | `257 chars sha=5531ae33f0f7`, all eight |
| physics | the gather **does not copy** the routine expert bytes — it reads them in place from pinned memory through the pool alias; only the neither-tier experts are copied (1200 fills = **1014.5 MiB** H2D for the whole prefill).  Worst-case in-place read 144 tables × 3 devices × 3 ubatches × 256 experts × 0.56 MiB = 186 GiB over 3.33 s = **19 GB/s per device**, against a measured per-device H2D calibration of **28.6 GB/s** → inside the link, so no work is being skipped |
| `GGML_MOE_GATHER_ONCE=1` corruption control | **VACUOUS at this width/quant** — it only restores the once-only expert *head-pad* zeroing and the text stays `5531ae33f0f7`.  Do not cite it as evidence; it is a gate that no longer bites here (§14.4's control was measured on the 1-GPU srv cell) |

**Repeatability.**  Pool+gather prefill across runs: 1063 / 1072 / 1129.6 / 1144 / 1300.4 / 1387.8 / 1471 / 1507 /
1569 / 1581 t/s.  `pinned` in the **same group, same prompt, saved logs, three back-to-back runs: 394.1 / 425.7 /
425.2 t/s** (prompt 3049 tokens, `-n 4`), and 471.8-606.9 across the 5258-token runs — i.e. **pinned is stable at
±4 % within a group too**, and the pool is 2.7-3.7x it.  One earlier `pinned` reading of **75.2 t/s did NOT
reproduce in three attempts** and is withdrawn: it came from an inline pair whose logs were not kept (the other
run of that pair read 380.8), and with logs saved the arm is well behaved.  So the honest claim is "**~1130-1580
t/s against a stable `pinned` at 394-426 (same 3049-token prompt) or 471-607 (5258-token prompt)**", and any ratio
must quote the pinned control from the same group.

Two `llama-bench` cross-checks remain open: `pp2048 -b/-ub 2048` gives pool **1063.4 ± 1.0** (the `pinned` arm
*core-dumps* at that tool setting, so there is no pair), and `pp1024 -b/-ub 1024` gave `pinned` 271.4 ± 2.5 against
pool **130.0 ± 73.2** — the opposite ordering, in the one regime where the CLI and `llama-bench` disagree.  That is
**not** explained and must be re-measured (a `-r 3` repeat at `-ub 1024`) before the head-to-head quotes a
`-ub 1024` number; the ±73 variance itself points at a first-iteration effect rather than a steady state.

**Correctness of the new default:** `257 chars sha=5531ae33f0f7` on all eight mode/device arms (pair and 3-GPU,
gather default and forced: `data/20261103-text-purity-b-{pair,g3}`), and **`5835 chars sha=dc178c29fe89` across
`pinned` / `pool` F80 / `pool` F95 at 3 GPUs with a 2000-token decode behind a 5258-token multi-ubatch
gather-served prefill** — i.e. the gather's prefill is text-exact at exactly the multi-ubatch width the code's
`SCHED_GATHER_TABLE_MIN_BYTES` comment warns the old "one-time expert-head zero" bug lived in.
`check-gpu-faults.sh` clean.  Note the *staging path itself* is still unidentified (TODO #131 keeps that item:
it is now only reachable via the kill switch, and it should either be fixed or deleted for pool tables).

**Split-mode dependence (measured 2026-11-03, same box, same prompt 3049 tokens, `-n 4`, `-ncmoe 48`, 3 GPUs).**
Every number quoted above is **`-sm tensor`** (the campaign's Phase-1 shape).  The `-sm layer` counterpart, same
arms and same redundant staging volume:

| 3 GPUs | `-sm tensor` | `-sm layer` |
|---|---|---|
| arena (`alloc_all_locked`) | 67110.0 of 67110.0 = 100 % | 67110.0 of 67110.0 = 100 % |
| `pinned` prefill | **394.1 / 425.7 / 425.2** | **944.3** |
| `pool` prefill (gather default) | 1129.6 / 1300.4 / 1387.8 | **1000.7 (+6 % over pinned)** |
| `pinned` / `pool` decode | 46-67 / 67 | 16.7 / 24.9 (the known `-sm layer` decode penalty) |
| staging calls per prefill | 396-846 (`-sm tensor`: one per table **per device**) | 244 (one per table: each device owns its own layers) |

Three consequences, and they narrow the finding: (a) the 100 % arena residency and the redundant re-copy of
resident tables happen in **both** splits — it is not a tensor-split artefact; (b) its *cost* is dominated by the
**number of staging calls**, which tensor split triples (a slice of every table on every device) while layer split
issues one per table, so `-sm tensor` pinned prefill is 2.3x slower than `-sm layer` (394-426 vs 944.3) for the
same bytes; (c) therefore the **pool's prefill advantage is a `-sm tensor` phenomenon** — under `-sm layer` the
two modes are within 6 % (1000.7 vs 944.3), because the staging is already cheap there.  Layer split remains
unusable for decode in either mode (16.7 / 24.9 t/s), as the delivery already documents.

**Why `pinned` prefill is so slow when every weight is host-resident (`-sm tensor`, measured 2026-11-03).**  Because at this arm
the experts are **also 100 % VRAM-resident**, and the prefill copies them out of VRAM per ubatch anyway:

| | 1 GPU | 3 GPUs |
|---|---|---|
| arena (`alloc_all_locked: MoE expert cache (auto)`) | 22544.8 MiB of 67110.0 = **33.6 %** | 67110.0 MiB of 67110.0 = **100 %** |
| slots/table | 85 | **256 (identity)** |
| `pinned` prefill, 5258-token prompt | **755.5 t/s** | 471.8-606.9 t/s |

Three devices with three times the VRAM and three times the compute prefill **slower than one** — so prefill is not
compute-bound and not host-residency-bound.  The cache's own counter says where it goes: on the 3049-token prompt
the 3-GPU `pinned` arm logs `prefill cache staging: ok=396 ... arena=61031.5 MiB host=510.5 MiB (H2D 0.8 % of
staged)` — i.e. **61 GiB is copied out of the ARENA (VRAM) into the op's per-split weight buffer, once per table
per ubatch**, and only 0.8 % comes from host RAM.  That copy is ~8 GB/s aggregate (2.6 GB/s per device) over a 7.7 s
prefill: far below local D2D speed, because it is a 256-block kernel copy per table issued 396 times and
*serialised against the compute* (copy table → compute table → copy next), so it scales with the number of tables
per device and never overlaps.  For a fully resident table the copy buys nothing — the op could read the arena in
place, which is exactly what the gather does in the pool arm (and why that arm scales normally: 1 GPU 734 → 3 GPUs
1300-1600 t/s).  Forcing the gather on `pinned` does **not** help (587.8 vs 564.7 t/s) because a pinned table's
gather source is the *pinned host master* over PCIe, not the arena.

That makes the pool arm's "3x prefill win" really "**no-copy prefill vs re-copy-into-the-op's-buffer prefill**", and
it means the same defect is still present in the **default (pinned) mode** at a large-arena configuration — see
`TODO` #133: let an identity (fully resident) table serve the prefill op in place, the aliasing
`moe_cache_get_table()` already does in the decode band.

### 18.2b REVALIDATED after the box restart (2026-11-03, clean box) — every headline number holds

**Why it had to be redone:** `/tmp` on `soar` is **tmpfs**.  This session had put ~6.6 GB of run artefacts in it
(rocprof CSVs, a 2.9 GB prefill-logit base, a llama.cpp worktree), an aborted run left a `llama-cli` holding
**70 GiB RSS**, and the box was running with 12 GiB of swap in use — after which the numbers were erratic (a 75.2
t/s `pinned` prefill, `llama-bench` core-dumping, a run aborted mid-flight).  The box was restarted; everything
below was measured on it with `swap 0`, `MemAvailable ≥ 83.7 GiB`, no stray processes, logs on **disk** (the repo's
`data/`, never `/tmp`), and each arm verified clean before and after (`tools/arm-run-261103.sh` refuses to record an
arm on a dirty box).  Data: `data/20261103-multigpu/reval/` (`summary.tsv` has the per-arm memory/swap/disk state).

| group | arm | decode t/s | prefill t/s | text |
|---|---|---|---|---|
| 3 GPUs, `-sm tensor` | `pinned` ×2 | **67.7 / 67.3** | **638.2 / 634.1** | `dc178c29fe89` |
| | `pool` FRAC 95 (gather) ×2 | **67.7 / 67.2** | **1631.3 / 1507.7** | `dc178c29fe89` |
| | `pool` FRAC 80 | 66.6 | 1343.8 | `dc178c29fe89` |
| pair, `-sm tensor` | `pinned` ×2 | **49.2 / 49.2** | **1651.4 / 1642.3** | `01607c16b483` |
| | `pool` FRAC 95 (gather) ×2 | **38.5 / 38.5** | 1446.2 / 1461.3 | `01607c16b483` |
| 1 GPU, `-sm layer` | `pinned` ×2 | **25.4 / 25.4** | 724.6 / 723.2 | `5421cf18bed5` |
| | `pool` FRAC 100 ×2 | **23.4 / 23.4** | 718.3 / 718.1 | `5421cf18bed5` |

So, on a clean box and with two runs per arm: **3 GPUs the pool is at decode parity (-0.6 %) and 2.4-2.6x
`pinned`'s prefill; the pair is -21.7 %; one GPU is -7.9 %** (exactly the campaign's long-standing "pool raw
overhead" figure).  Every group's text is byte-identical across modes.  The prefill-logit gate was re-run here too:
**mean KLD -0.000000 / same-top-p 100.000 %** (base PPL 3.5537).

**One anomaly survives the clean box, and it is `llama-batched-bench`-specific**, so the gate must be read with
care: at `-npp 64 -ntg 64 -npl 1` the pool arm reports **~1 t/s** prefill (63-66 s for 64 tokens) against
`pinned`'s 9.74 — and it is **1.00 with the gather on and 0.96 with it off**, i.e. not a path question; the same
pool arm at the same 64-token width in `llama-cli` gives **71.2 t/s** against `pinned`'s 33.8 (and `pinned` itself
is 3.5x depressed in the tool at `npl=1`).  At `npl` 4 and 8 the tool behaves sanely and the pool leads by 4-6x
(**338.5 vs 81.8**, **835.8 vs 136.0**).  Treat the `npl=1` cell as a tool artefact to be explained, not as a
host-expert measurement; the earlier report that this "1.02 t/s" was caused by the memory contamination is
**withdrawn** — it reproduces on the clean box, so it is a separate, still-open question (recorded under `TODO`
#131, not a regression from `patches/0006`).

### 18.3 W3 — the `RAM_PCT` ceiling now scales every table (implemented, measured, `patches/0005`)

`pool_clamp_slots_to_ram()` shaved the **registration-order tail** to the one-slot floor.  Measured on `soar`,
1 GPU, `-sm layer`, `FRAC 100`, 2000-token decode (`data/20261103-multigpu/w3-*`):

| arm | decode t/s | prefill t/s | pinned | pool slots/table |
|---|---|---|---|---|
| **before**, `RAM_PCT=70` | **10.94** | 343.5 | 54.7 GiB | 117×256, 1×34, **26×1** |
| **after**, `RAM_PCT=70` | **17.44 (+59 %)** | 597.8 | 54.5 GiB | 144×208 |
| **after**, `RAM_PCT=70`, `SIZING=1` (ceiling-fitted) | **18.92 (+73 %)** | 612.1 | 54.4 GiB | cov inside the ceiling |
| **before**, `RAM_PCT=85` | 19.52 | 617.5 | 66.1 GiB | 141×256, 1×181, **2×1** |
| **after**, `RAM_PCT=85` | **20.85 / 20.97 (+7 %)** | 607-671 | 65.8-66.3 GiB | 144×253 |
| `pinned` (1-GPU control, same session) | 22.36 | 671.7 | — | — |

So the level was never the problem (the 1-GPU session's OPEN B finding, §14.6) — the *distribution* was, and it
is worse than measured before: at `RAM_PCT=85` the shave already bites **from layer 18 on** (2 tables at 1 slot),
and at 70 it takes out **26 of 144 tables**.  The fix (patch `0005`): compare the process-wide plan with the
ceiling **once** and scale *every* table's slot count by that factor, keeping the one-slot floor only where the
ceiling cannot cover a slot, with the old shave as a rounding backstop; and let the concentration fit spend the
ceiling as its budget (a uniform rescale of a fitted plan collapses it back to uniform — measured identical
144×208 before that second half).  `RAM_PCT` is now a usable knob: 70 % + cov90 (18.92) sits 15 % behind the 85 %
arm instead of 44 % behind it.

### 18.4 The decode gap on the pair is the SYNCHRONOUS DEMAND READ — 2.2x the 1-GPU cost

§16.3 sized the demand-fill overlap candidate at a ~6 % ceiling on one GPU and correctly declined it.  On the
pair that ceiling is much larger, and the fill accounting (`patches/0004`) says why:

| arm | pool fills | read wall | bytes | demand fills (minus prepopulate) | demand read / decode token |
|---|---|---|---|---|---|
| 1 GPU, FRAC 80 (§16.3) | 38 844 | 29.3 s | 69.1 GiB | 9 468 / 1 862 = **5.1** | **~5.2 ms of a 56 ms step (9 %)** |
| pair, FRAC 80 | 69 777 | 35.1 s | 124.0 GiB | 40 401 / 1 950 = **20.7** | **~11.3 ms of a 38 ms step (29 %)** |
| pair, FRAC 100 (clamped) | 47 446 | 25.0 s | 84.3 GiB | ~10 600 / 1 950 = 5.4 | ~6 ms |
| pair, FRAC 80 + cov90 | 59 321 | 30.4 s | 105.5 GiB | 29 900 / 1 950 = 15.3 | ~8 ms |

At `FRAC 80` the pair's whole deficit is 25.1 ms/token (pinned) vs 38.3 (pool) = 13.2 ms, of which the demand
fill is ~11.3 ms — **86 % of the gap is one synchronous O_DIRECT read per uncovered expert per decode step**,
not copies and not traffic (identical arena, identical reaches, the pool moves fewer bytes).  A multi-device
split multiplies the miss set (each device's table misses its own slice) while the fill stays on one host thread.
**So the "overlap worker" that §16.3 declined is worth re-opening for the multi-device case** (ceiling ≈ 25-30 %
there, ~0 at full coverage), and it is a strictly better first target than W2's remap (+3.3 %).

### 18.5 The box was wedged at the end of the session (my mistake, recorded so it is not repeated)

A `pkill -f "seq 1 90"` intended for a sampler matched its own shell, killed the running `llama-cli`, and left a
**stale KFD entry holding ~99 GB of VRAM** across the three R9700s (`rocm-smi --showpids` shows a dead pid as
`UNKNOWN`; every subsequent load fails with `unable to allocate ROCm0 buffer`).  It needs
`sudo rocm-smi --gpureset -d 0,1,2` or a reboot to clear — it cannot be reclaimed from user space.  The handover's
own trap list says to kill by pid, never by pattern; this is that trap, self-inflicted.

### 18.6 Gates run this session

* **Greedy text**, `-n 64`, no `-lv`: `257 chars sha=5531ae33f0f7` on **all eight** arms — pair and 3-GPU,
  `pinned` / `pool` FRAC 80 / `pinned`+gather / `pool`+gather (`data/20261103-text-purity`).  Identical *across*
  device counts as well as within them, which is stronger than the per-device-count reference the campaign
  assumes.
* `scripts/check-gpu-faults.sh --since "04:10"` → **clean** (the only kernel faults in the window's journal are a
  burst at 03:58:53, i.e. before this session's first run; `--since "5 hours ago"` fails on them, so the window
  matters).
* W3 changes were measured with the *same* binary pair (before = the five-patch-minus-0005 build, after = the
  six-commit build), arms alternating, identical protocol.
* Not run this session (and why): `gate-prefill-logits.sh` (needs a base recorded on this box; the gather and W3
  are layout/diagnostic changes whose *text* identity is measured above — but a promotion needs it, see §6),
  the MTP four-axis protocol (no MTP arm was run: `--spec-type none` throughout), `llama-batched-bench` (no
  server path changed).

### 18.7 What this session changes in the plan

0. **The pair is at −19.7 % and the 3-GPU shape at −1.8 %** with the campaign set on the W3 build (pair:
`pool` FRAC 95 32.02 vs `pinned` 39.85; 3 GPUs: `pool` FRAC 95 51.03 vs `pinned` 51.97, gather on).  The
acceptance criterion (§1) is therefore **met on the box's own three-device shape and not on the pair**, and on
the pair the residual is the demand fill (§18.4) plus the page-cache starvation in finding D.
1. **F7 is closed** (§18.1) — the handover's inherited blocker was an environment mistake, and the multi-device
   arms mean something now.
2. **W3 is done** (§18.3, `patches/0005`): `RAM_PCT` scales proportionally, `+59…73 %` decode at `RAM_PCT=70`,
   `+7 %` at 85 %, at identical pinned bytes.
3. **New P0: the pool's staging path under a ≥3-device tensor split** (§18.2 finding B, `TODO` #131) — prefill
   16.85 t/s against `pinned` 564.65, fixed in-place by the gather (1126.8).  A delivery defect, not a campaign
   one.
4. **P1-a's overlap half is re-opened for the multi-device case** (§18.4): the demand fill is 29 % of a pair
   decode step, so the worker's ceiling is ~25-30 %, not ~6 %.  Re-screen with the replayer before building
   (`§9.6`), then measure.
5. **The gather's promotion shape is now implemented and measured** (§7 open decision 2): a **pool-only**
   default, gated by the truthful per-device query `moe_cache_pool_backed()` and band-limited to the prefill
   (`patches/0006`).  `pinned` never gathers, so the 2-device −47 % regression is gone; the kill switch is
   `GGML_SCHED_DEVGATHER=0`.  What remains for the maintainer is promotion only (the `ENVIRONMENT.md` row).
6. **W2 (the tensor-split-aware remap) is unchanged and still last**: its ceiling on the pair is ≈ +3.3 % of
   decode, against the fill's ~25-30 % and the staging defect's prefill collapse.  It is not blocked by anything
   here; it is simply no longer the biggest item, and it still needs the maintainer's `§7` decisions before it
   lands.
