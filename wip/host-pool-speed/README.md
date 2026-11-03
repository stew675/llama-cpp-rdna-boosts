# wip/host-pool-speed — make `--host-experts pool` faster, simpler and automatic (OPEN)

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
- `scripts/moe-trace-replay.py` — the offline two-tier model (band-split heat, `--pool-frac`).
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

**Current state (2026-11-02, session 3 — the 1-GPU host; see §14 for the measured record):** the campaign set is
now **three** patches on the r206 tip (applied tree `6e2b58c83`): the teardown-report / forced-`DIO` diagnostics,
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

The set now has three patches (README in [`patches/`](patches/README.md)); applying all three to a fresh
`efaa60db0` gives **applied tree `6e2b58c83e4f550f6a0d537216c05ac01754e083`**.  Per-patch build: 3/3 clean
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
