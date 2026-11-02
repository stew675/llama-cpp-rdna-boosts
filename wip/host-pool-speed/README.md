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

### 3.2 P0-b: `MOE_HOST_POOL_RAM_PCT` and swap thrashing

The maintainer reports swap thrashing on some models at the current default.  Known properties to attack:

- 85 % of `MemAvailable` **measured once, at the first pool build** — a box whose availability later shrinks
  (another model, a server, a browser) keeps the old ceiling.
- The clamp keeps a **one-slot floor per table**, so the pinned total can exceed the ceiling by
  `n_tables × host_bytes` (documented in `ENVIRONMENT.md`; never measured on a real model).
- `FRAC` and `RAM_PCT` interact and only `FRAC` was swept (r112, and only at 80/90 on the field box).
- The pinned bytes are **not visible to the operator** (`hipHostMalloc` is not `VmLck`), which is exactly the
  blindness that makes a thrash hard to attribute.

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

Measure with `/proc/diskstats` (bytes read per decode token) and `MOE_HOST_POOL_VERIFY=1`, not throughput alone.

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

- Boxes: the primary (3 × R9700 gfx1201, ROCm 10.1) and the tight one (**gfx1100 / Strix-class, ~30 GiB RAM**)
  — the second is the point of pool mode and the only place `RAM_PCT`/`DIO` bind.
- Models: 35B-A3B UD-Q4_K_M (`-ncmoe 40`), Qwen3.8-27B Q4_K_XL (dense control), Flash-Next GSQ-RCO IQ3_XXS
  (`-ncmoe 24`, fused tables + qwen4exp), gemma4 26B-A4B (fused, fused-table decline path).
- Arms: `pinned` / `pool` with the relevant knobs crossed; **always include the `pinned` arm in the same run**
  (the campaign's acceptance criterion is a ratio to `pinned`, not an absolute).
- **Full feature set, every arm** (`AGENTS.md` Default-on policy): no `FOO=1 <bench>` unless the arm *is* the
  A/B, and pin `HIP_VISIBLE_DEVICES` on every multi-device arm (the iGPU trap).
- Oracles: same-seed greedy text (byte-identical — a pool/layout defect is a wrong answer, not a crash), MTP
  four-axis acceptance counts, `gate-prefill-logits.sh` for any change to the pool fill/staging path,
  `llama-batched-bench -npl 1,4,8`, `MOE_HOST_POOL_VERIFY=1`, `check-gpu-faults.sh`.
- Instruments: the cache's own traffic report, `MOE_HOST_POOL_VERIFY`, `/proc/diskstats`, `VmSwap`/`pswpout`,
  and `scripts/moe-trace-replay.py` (two-tier offline model) to screen a layout/sizing candidate before code.

## 10. Open questions for the maintainer (decide before the first code change)

1. **Decode target**: parity with `pinned` *with* the remap (65.7 on the r203 cell, i.e. P0-a must land first),
   or is the honest bar "pool must beat `pinned` eager" (60.6, already met)?
2. **`RAM_PCT`**: lower the default opportunistically now (a one-line change + a soak), or hold the default
   until the dynamic-ceiling design lands?  (There is no `TODO` handle for the swap report yet — say the word
   and one is allocated.)
3. **Knob collapse (§5)**: is a "one `auto` decision, everything else diagnostic" shape the target, or should
   the existing per-knob defaults stay user-visible for the field configs?
4. **P3 (§7)**: is the "guaranteed pinned **and** resident" intermediary worth a design note of its own, or is
   §4.1's residency check enough?
5. **PR #267**: hold the reply until §3.1 and §4.1 report numbers (current plan), and if it is declined, does
   the reply offer a `wip/` home for the submitter's tier work so it is not lost?

## 11. References

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
- PR #267 files (on the submitter's branch, not in this repo): `wip/mmap-host-experts-r204/0001..0010`,
  `README.md`.
