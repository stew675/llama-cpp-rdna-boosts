# wip/apu-expert-gtt — a UMA-only, heat-mapped **GTT expert store** (no host pool)

**Status: OPEN — design REVISED 2026-11-11 (`halo`, gfx1151); P0 **and** the P1.0 keystone probes DONE (all four
assumptions hold, `MEASUREMENTS.md` §13); nothing implemented, nothing in `patches/`, nothing applied to the
fork.**  The plan is [`IMPLEMENTATION.md`](IMPLEMENTATION.md); the probe log and every number are
[`MEASUREMENTS.md`](MEASUREMENTS.md); the probe sources are [`probe/`](probe/README.md) so each claim can be
re-run on any UMA box.

Renamed from `wip/apu-expert-pool/` on 2026-11-11: **there is no pool in this design any more** (§1), and the
thing the tier holds is a store of resident experts, not a cache tier under a host pool.  Cut against the
**r259** tree (`release.json` tree `7995db44…`, tip `c80fa8bda…`); file:line references are that tree.

## 0. The shape, in the maintainer's words (2026-11-11)

> a UMA-only variant of the heat-mapped approach, but for experts resident in the GTT arena.  No slab/arena
> involved.  Just a mass of HIP registered transparent huge pages to hold the GTT resident experts, and rather
> than have a host pool at all, if there is a spill over because the MoE experts don't fit, then evict the
> coldest expert by removing it from the registration table and have the GPU initiate a load from storage
> directly into now vacated memory region, and once the load is complete, then continue.  So, sort of like what
> discrete GPUs do today with host-experts, but without the pool at all.  No host-pool at all, just a
> zero-copy load into the vacated expert slot.  Prefilling/prompt processing should be handled in a similar way
> to what occurs with host experts for discrete GPUs.  The wide-prefill table should only stream in from
> storage just the layer weights that aren't already resident.

In one table:

| element | the design |
|---|---|
| **the tier** | ONE anonymous region: `mmap` + `MADV_HUGEPAGE`, `hipHostRegister(Mapped\|ReadOnly)`, sliced into per-expert slots.  It **is** the resident expert set — there is no pinned master, no host pool, no arena, no slab below or beside it |
| **what it is made of** | registered THPs, so the GPU reads the resident set at **230-233 GB/s** (measured, §2) and the host fills it through its own writable mapping with **no flush** (measured FRESH) |
| **residency** | the delivery's existing **heat table + slot map** (block-09/12 machinery).  On UMA nothing decides *location* — host RAM and "VRAM" are one DRAM — so residency is the whole policy |
| **a miss** | evict the coldest slot (a **table** operation), then load straight into the vacated bytes, then continue.  No copy-out, no `madvise`, no unregister on this path |
| **the one knob** | **`-lme N`** (GiB) sizes the store, and `-lme auto` derives it -- the *same* flag, the same units and the same "ranked by the heat prior" policy the discrete box uses for its pinned per-expert pool.  **It is the one command line that carries across from discrete mode to APU mode** (§5) |
| **admission** | **Nth touch**, not first -- the delivery's existing `admit=touch` policy (`MOE_EXPERT_CACHE_TOUCH`, default **2**; the 2-vs-3 A/B is measured, see §7 Q1).  A non-admitted read must land *somewhere* GPU-readable, so the store keeps a small **transient band** (the analogue of the pool's in-place cold read, §5.1) |
| **eviction** | **LFRU over the resident set** (the delivery's existing policy), as confirmed by the maintainer on 2026-11-11 |
| **the load** | zero-copy into the vacated slot: one `pread` into it when the source is cold, or one copy-engine pull from the registered read-only **file mapping** when it is in the page cache — **and pipelined behind the previous layer's compute**, because a synchronous per-layer fill cannot decode (§3.2) |
| **prefill** | per layer, stream in only the experts that are **not already resident**, coalesced — the discrete host-expert wide-prefill table's shape, without a pool to stage through |
| **what stays refused** | `-ncmoe` / `-cmoe` / `--host-experts pinned` / `--host-experts pool` on an integrated GPU stay exactly as r236 left them — this tier is not the host-expert tier and does not re-open it |

## 1. What this replaced, and why (2026-11-11)

The campaign opened (2026-11-09) with a different plan, and the maintainer's directive is that it was the
wrong one.  The differences, so a reader of the old record can map it:

| old plan (superseded) | now |
|---|---|
| **P1**: enable `--host-experts pool` / `-lme auto` on `IGPU` (the `ROCm_HostMap` mapping master + a bounded pinned pool) | **dropped.**  No host pool on UMA.  `llama_model_loader::set_host_experts_umem()` (`src/llama-model-loader.cpp:130`) keeps refusing the host tier there, unchanged |
| **P2**: the arena as the disk-backed cache; slab sized from a RAM budget; fills into the arena slot; a staging/bounce ring for the disk tier | **same shape, different memory.**  No slab, no arena, no staging ring: the region is registered anon THP and the disk fill is zero-copy into the slot |
| **P0.5**: `MOE_HOST_POOL_RO` — read-only registration for the *pool's* device alias | becomes `MOE_EXPERT_GTT_RO`: the **store** is registered `ReadOnly` from creation.  Same 60 -> 233 GB/s fact, applied to the tier that now exists |
| **P3/P4** (one budget across pool + arena + context; policy unification) | **P3 survives as one budget** (there is only one tier left to budget); P4 becomes "prefill streaming", which is a mechanism, not a policy |

**Why the old plan was wrong, from its own probes:** `pread(O_DIRECT)` into a `hipMalloc` GTT buffer is
**`EINVAL`** (`MEASUREMENTS.md` §7) — so the old plan had to keep the staging/bounce ring for every disk fill,
and the "eviction moves no data" win only ever applied to the *host* tier.  `pread(O_DIRECT)` into an
**anonymous** buffer works at **3.6 GB/s** (`MEASUREMENTS.md` §4).  An anonymous, registered region is
therefore the only backing on which *"a zero-copy load into the vacated expert slot"* is possible at all — and
choosing it deletes the pool, the staging ring and the arena from the UMA path at the same time.

## 2. What the design stands on — measured (`halo`, P0)

| fact | measurement | probe |
|---|---|---|
| no fault-and-retry exists: an unregistered mapping is not in the GPU's page tables, at all, hot or cold | `HSA_STATUS_ERROR_MEMORY_FAULT` … `Page not present or supervisor privilege`, GPU coredump, exit 134; `HSA_XNACK=1` inert; `PageableMemoryAccess = 0` | `02-fault` |
| gfx1151 has no `xnack` target to turn on | `gfx1151:xnack+` -> `invalid target ID` | `MEASUREMENTS.md` §1 |
| "VRAM" is the host DRAM | 2 GiB carve-out, 130 GB GTT, HIP `totalGlobalMem` 121.1 GiB, `Memory Properties: APU` | `01-props` |
| a registered **read-only** host range reads at 230-233 GB/s; registered **writable** at 60 | anon RO **230.8**, file RO **230.5**, `hipMalloc` 232.6, anon writable **60.2** | `03-bandwidth` |
| a *filled* `PROT_WRITE` buffer can be registered RO afterwards | `hipHostRegister(Mapped\|ReadOnly)` OK -> GPU reads it at **233.2** | `04-fill-and-register` |
| the host can write a registered range in place and the GPU reads the new bytes, both modes, no flush | 4 alternations, FRESH every time | `10-coherence` |
| disk -> host memory, zero copy, no staging | `pread(O_DIRECT)` 1 GiB into anon **3.6 GB/s**; into `hipMalloc` **`EINVAL`** | `04`, `07` |
| warm page-cache -> GTT through the existing transport | `hipMemcpy` **31.5** cold / **77.1** warm; host `memcpy` 12.8 | `08` |
| registration is a hold, and it outlives `hipHostUnregister` while the VMA lives | 4 GiB: control reclaims 0/1 048 576 pages; registered holds 1 048 576 | `09` |
| **trap**: `MADV_DONTNEED` on a registered region the GPU is reading is a **silent zero** | GPU sum `0333…` -> `0000…`, no fault, no error | `06` |
| **trap**: a GGUF tensor offset is **32-byte** aligned, not 4096 | `dd iflag=direct` at a real expert offset copies **0 bytes**; at `align_down(off, 4096)` it works (the bytes then land `+2 688` into the slot) | `14`, `dd` |

P1.0 (§13) re-measured several of these at the store's own shapes; where a figure differs (e.g. `pread` into
anon: **3.6 GB/s** at 1 GiB in §4, **4.75-6.4 GB/s** at 256 MiB in §13) the newer, store-shaped number is the
one §3 uses.  Nothing in §§0-12 was falsified — those are dated records and stand as measured.

### P1.0 — the four keystone assumptions (MEASURED 2026-11-11, all four HOLD; `MEASUREMENTS.md` §13)

| assumption | result |
|---|---|
| THPs survive `hipHostRegister` | **yes** — `AnonHugePages` 1024.0 MiB of 1024 before *and* after registering (`11-thp-register`); registration of a pre-faulted region is **17.4 ms/GiB**, GPU read **232.1 GB/s** |
| `pread` **into** a registered RO region, then a GPU read | **yes** — `O_DIRECT` accepted (4.75 GB/s cold), the GPU reads the fresh bytes **including on the reuse/eviction arm**; buffered is faster still (**15.7 GB/s**) |
| the copy-engine pull from a registered RO file mapping | **yes** — one 0.68 MiB slab: 0.012 ms warm (**60.9 GB/s**), 0.083 ms first; a 16 MiB layer span: 0.57 ms warm (**29.4 GB/s**); bytes verified.  Faster than a host `memcpy` at slab granularity |
| per-fill cost at the real granularity | see §3.2 — **0.41-0.57 ms** for a cold 0.88 MiB slab (35 ms after idle), **0.07-0.38 ms** warm, **0.42 ms** for `pread -> launch -> sync`; a 14.1 MiB span streams at **5.75 GB/s** |

One rule fell out of it: **a GGUF tensor offset is 32-byte aligned, not 4096-byte aligned**, so an `O_DIRECT`
fill needs a round-out to the block size plus a per-fill delta into the slot (2 688 bytes on the layer measured),
or the fill must be buffered.  See §6.

The one probe still owed is a **workload** one: the **per-session touch distance** (how often an expert is used
1x, 2x, 3x in one session), which is what sets the admission threshold `N`.  The persisted heat map gives the
long-horizon concentration (§3.2) but not a session sequence.

## 3. How a step and a prefill run

**Decode.**  The routing is known after the gate/softmax, before the `MUL_MAT_ID`:

1. for each selected expert, look it up in the heat table; a hit is a touch (heat) and nothing else;
2. a miss claims a slot: the coldest **non-resident-safe** entry's table entry is dropped, the slot is free,
   nothing is copied out (its bytes become garbage the fill overwrites);
3. the load is issued **into that slot** and the launch is ordered behind it (§3.1);
4. the `MUL_MAT_ID` reads the slot in place at registered-read-only speed.

**Prefill.**  A prompt chunk routes over most of a layer's experts, so prefill is *streaming*, not caching:
per layer, compute the layer's used-expert set, subtract what is already resident, and bring in only the
difference — coalesced by expert (an expert's rows are contiguous in the GGUF), and never re-fetching a bank
the previous chunk already has.  This is the discrete host-expert wide-prefill path's shape
(`moe_cache_stage_table()`, `MOE_HOST_POOL_STAGE_SPAN`, `moe-expert-cache.cu:8134`), with the store replacing
the pool.  It is also where the zero-copy property pays most: a prefill fill is a bulk sequential read, which
is what the 3.6 GB/s pread arm is good at.

### 3.1 The two load arms

| arm | source state | mechanism | measured |
|---|---|---|---|
| **cold** | not in the page cache | host `pread` **straight into the vacated slot** (buffered, or `O_DIRECT` + a block round-out and delta) | 0.88 MiB slab: **0.413 ms** median `O_DIRECT`, **0.567 ms** buffered, **34.9 ms** worst first-read-after-idle.  Buffered is the faster arm at this granularity on `halo` and leaves the page as a future warm source |
| **warm** | in the page cache | `hipMemcpyAsync` **on the compute stream** from the **registered, read-only `MAP_SHARED` mapping of the GGUF** into the slot | 0.68 MiB slab: **0.012 ms** (60.9 GB/s); 16 MiB span: **0.570 ms** (29.4 GB/s); bytes verified; beats a host `memcpy` at slab granularity |

The GPU cannot issue file I/O: "GPU-initiated load" is the **warm** arm (a copy engine pulling from a
registered mapping).  The cold arm is host-issued, but it is still a *single* copy that lands in the slot, which
is what "zero-copy into the vacated slot" means here.  Which arm runs is a property of the page cache, not a
knob — the census must report both (`fills_cold` / `fills_warm`) so a measurement says which one it measured.

### 3.2 The cost model the policy follows (all measured, `MEASUREMENTS.md` §13.4-13.5)

A decode step touches top-k × 3 tensors × n_layer slots.  On Flash-Next (10 × 3 × 48) that is **1 440
slots**; at the measured **0.42 ms** fill round trip, *every* slot missing would cost **~0.6 s per step**, 90 %
hit rate **~60 ms**, 98 % **~12 ms**.  Three consequences, and they are the design:

1. **The hit rate must be very high** — or the fills must be pipelined.  Pipelining the next layer's fills behind
   the current layer's compute is the only way to decode at a merely-good hit rate; it is a P4 requirement, not
   an optimisation (`TODO` #132 is the same defect on the discrete pool: "the pool's pair decode deficit is the
   synchronous demand read").
2. **Admission costs one read and saves it on every later use**, so a 2nd-touch promotion pays for itself at two
   uses in a session, and a 3rd-touch policy only wins while the resident budget is the binding constraint
   (i.e. it is a *footprint* lever, not a cost lever).  Hence `N` is a knob with a measured default, and the
   per-session touch distance is the one probe still owed.
3. **Prefill is cheap and decode is not**: a layer span streams at 5.75 GB/s, a scattered 0.88 MiB slab is
   latency-bound — so prefill streams spans (P5) and decode must pipeline.

**How big must the store be?**  From the delivery's own persisted heat maps (MHT2, `moe-heatmap-read.py`), over
*(tensor, expert)* slots: on **35B-A3B Q4_K_M** (18.22 GiB of experts, 7.5 M accumulated accesses) **21 %** of
the slots were **never touched**, and 90 % of traffic is covered by **35.5 %** of the slots = **6.46 GiB** (95 %:
8.05 GiB).  The Flash-Next map looks far more concentrated (13.1 % at 90 %), but it has 27x less history, so it
is undersampled — **plan on the long-history number**.  The never-touched mass is what an Nth-touch policy keeps
on disk for free, and it is why "hot subset" (below) is the right default.

## 4. Code map: what is reused, what is new, what is deleted on the UMA arm

**Reused as-is (do not re-derive):** the **`-lme` budget and its heat ranking** (`plan_host_expert_placement()`, `src/llama-model-loader.cpp:277-310`, and `pool_rerank_from_heat_locked()`, `moe-expert-cache.cu:2905`) -- the APU store is the same decision at a finer granularity, so the size and the order come from that code, not from a second policy -- plus the heat table and its MHT2 sidecar prior (`apply_heat_seed_locked`), the
slot/remap indirection (`g_dr_key_to_id`, `moe_cache_remap`), the per-layer access plan
(`access_locked()`, `moe-expert-cache.cu:3495`), the census lines, the prefill in-place table read
(`moe_cache_in_place`, r258, `ggml-cuda.cu:9455`), and the staging/span machinery
(`moe_cache_stage_table()`, `:8134`) — the last one reused for **prefill** (P5) rather than for decode fills.

**New:** the store itself —
* allocation: `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)` + `MADV_HUGEPAGE`, then
  `hipHostRegister(Mapped|ReadOnly)` over the budgeted extent (§5), sliced into per-expert slots;
* a **buffer type** for expert tables with the delivery's KV-region pattern
  (`ggml_backend_cuda_kv_region_buft_split_for_dev`, `ggml-cuda.cu:10697`): a device-carrying buft with the
  plain device type's vtable contract, so the scheduler places expert tensors in it and `MUL_MAT_ID` reads it
  in place with no new dispatch and no new op;
* the fill: `fill_slot_from_host()` (`moe-expert-cache.cu:3452`) retargeted to the two arms of §3.1.

**Deleted *on the UMA arm only*:** the pinned host pool and its master, the `access_locked()` arena->pool
hand-over, `bounce_acquire()`'s staging ring for the store's fills, and any slab sizing for this tier.  The
discrete path keeps all of them: this is a UMA branch, not a replacement.

**Not touched:** `set_host_experts_umem()` and `host_expert_selected()` (the host-expert tier stays refused on
UMA, r236); the `-ncmoe` refusal; `ggml_cuda_slab_*` (this tier does not use the slab — see the trap in §6
about the slab's *own* free-VRAM sizing, which stays a separate, pre-existing item).

## 5. Sizing: `-lme N`, the one knob that carries across from discrete mode (2026-11-11)

On a discrete box `-lme` (`--limit-host-experts`) already means *"cap the NON-SWAPPABLE memory used by
host-resident MoE experts, in GiB, spent in descending hotness from the persisted heat map"*
(`common/arg.cpp:2767`).  On an APU it means **the size of the registered GTT expert store** -- the same semantic
shape, the same units, the same ranking, and (in the pool band) the same per-**expert** granularity the discrete
path already uses.  So the design keeps **one** CLI knob across both modes and adds no sizing flag:

| `-lme` | discrete box (today) | APU (this design) |
|---|---|---|
| **`auto`** | fits under `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable` -> mirror the whole set; otherwise a bounded per-expert pool | **fits** -> the whole expert set is resident in the store (zero evictions); **does not fit** -> `MOE_HOST_LIMIT_FRAC` x `MemAvailable` of store, i.e. a hot subset with evictions on |
| **`N`** | pin N GiB, spent on whole tensors in descending hotness (fit case) or as a per-expert pool (not-fitting case) | exactly **N GiB** of registered store, spent at **one (tensor, expert) slot** granularity in descending hotness -- finer than the discrete fit case, so the budget is used byte-for-byte |

Three things the APU arm has to decide differently, and they are P1's work:

1. **The ceiling is different.**  `MemAvailable - KV - compute - OS reserve`, read **once**, floored -- never
   "free VRAM" (the 2026-10-19 over-commit: 121 GiB resident + 13 GiB swap).  The store is *registered host*
   memory, so it is not charged to the GTT pool (`amdgpu.gttsize`, 121 GiB on `halo`) -- what bounds it is the
   pinning rule and the non-swappable ceiling.  A 46 GiB userptr registration is measured to work (§9), so the
   store can be large; no ceiling above that has been measured.
2. **`-lme` must select the store without `-ncmoe`.**  On a discrete box host experts exist only because
   `-ncmoe`/`-cmoe` moved them there, so `-lme` is a *cap on a placement someone else asked for*.  On an APU
   there is no host tier (`-ncmoe` is refused, r236), so `-lme` becomes the **only** expert-residency knob and
   has to engage the store by itself.  Its help text needs the UMA branch; the wording carries over almost
   verbatim ("N GiB, spent on the hottest slots, the rest stays on storage").
3. **Only the not-fitting band is a win.**  A set that already fits needs no store: the delivery's plain
   placement reads GTT at **232.6 GB/s** and the registered store reads at **232.1** -- the same.  The store's
   value is entirely in the set that does **not** fit, where the alternative is a mapping read through the page
   cache (or a fault).  So `auto` engages in the not-fitting band, and an explicit `-lme N` is the middle
   ground.  Default-on policy: the store is **on** with `auto` (there is nothing to opt into), and
   `MOE_EXPERT_GTT=0` is the kill switch.

Whatever the budget leaves is the page cache, which is where the warm fill arm's source comes from -- so the
number implicitly decides the cold:warm ratio, which is why the census must print both counters.

A resident slot is **not** reclaimable while registered (§2, `09`), which is what makes the budget
load-bearing rather than advisory, and the reason `MADV_DONTNEED` is only ever reached *after*
`hipHostUnregister` -- and never while a launch can address the region.

### 5.1 The transient band (what makes `N > 1` implementable)

`admit=touch` on the discrete path serves a non-admitted expert "from the pool's in-place cold read"
(`ENVIRONMENT.md`, `MOE_EXPERT_CACHE_TOUCH`) -- it needs no destination, because the bytes are already mapped.
On an APU a non-admitted expert is **only on storage**, so the read has to land somewhere the GPU can read:
a small **transient band** of registered slots, one (or two) per table, filled by the cold or warm arm, read by
the `MUL_MAT_ID`, and reused by the next non-admitted expert of the same table.  It costs
`n_tables x slab` ~ 130 MiB on Flash-Next (144 tables x ~0.9 MiB) and must be counted in the budget; without it
a touch-count policy degenerates into "admit on first touch", i.e. no policy at all.  Whether one slot per table
is enough (tables are processed in sequence) or two are needed (a batch can have two in flight) is a P3
measurement.

## 6. What must NOT be done (each is a measured trap)

* **Do not use `MADV_DONTNEED` as the eviction primitive.**  On a registered region the GPU reads it is a
  silent zero-fill (measured).  Eviction is a table operation; the only discard is `hipHostUnregister` **then**
  `madvise`, on the budget-shrink path, with the region out of every graph.
* **Do not `hipHostUnregister` on the hot path at all.**  It does not release the pinned pages while the VMA
  lives (measured), and with `iommu=off` the registration *is* the GPU's mapping for that VA, so unregistering
  turns a read into a fatal fault (no XNACK, no retry) instead of a fill.  Removing an entry from the
  residency table is what makes a slot reusable.
* **Do not make the file mapping the thing the GPU reads.**  `PageableMemoryAccess = 0`: an unregistered
  page-fault-capable mapping is not in the GPU's page tables, hot or cold (measured fatal).  A pageable mapping
  is a *source* (a `pread` target or, registered, a copy-engine source); never a read target.
* **Do not size the store from "free VRAM".**  Measured over-commit; the guard stays.
* **Do not `O_DIRECT` a GGUF offset directly.**  Tensor offsets are aligned to `general.alignment` (32 bytes),
  so an `O_DIRECT` fill must round the read out to the block size and carry a per-fill delta into the slot — or
  be buffered.  Measured: an unaligned `O_DIRECT` read copies **0 bytes**.
* **Do not use `VmPin`/`VmLck` to detect the registration hold** — both read 0; use `mincore()`.
* **Do not touch the `integrated` host-buffer (zero-copy UMA weights) path.**  It has its own history and its
  own guard (`ggml_backend_sched_buffer_supported()`); `GGML_FORCE_NO_INTEGRATED=1` is the A/B that proves the
  store does not depend on it.
* **Do not probe UMA by host-dereferencing a device pointer.**  Gate on `prop.integrated`.

## 7. Open questions for the maintainer

1. **Admission threshold `N`** -- *already measured once, on the discrete path, and 2 won.*  The delivery's
   `admit=touch` A/B (2026-11-07, Flash-Next + MTP, `-lme 36`/44, 3+3 interleaved reps,
   `archive/work/host-expert-parity/RESULTS-2026-11-07-touch.md`): `MOE_EXPERT_CACHE_TOUCH=3` damps churn exactly
   as intended (demotions **-17.3/-18.3 %**, sink uses -27 %) and buys **nothing measurable** (decode +1.3 to
   +2.2 %, inside the box's spread, prefill flat) while giving up 0.78/0.81 pt of hit rate and serving **+35 %**
   more misses cold -- so `2` stayed the default.  **What changes on UMA is the price of "served cold"**: on the
   discrete path it is an in-place read of an already-mapped page; here it is a **0.4-0.6 ms storage read** -- so
   `3` deserves the re-run, with `2` as the starting default and the existing knob (no new variable).  The
   per-session touch distance (probe 15) is what would make the answer predictive rather than empirical.
2. **Does the store hold the hot subset or the whole expert set?**  The heat-map measurement (§3.2) says the hot
   subset is the right default: on 35B-A3B, 90 % of traffic comes from 35.5 % of the slots and 21 % are never
   touched at all.  The whole set only makes sense on a model whose expert set fits the budget outright.
3. **Eviction policy**: LFRU is confirmed.  The remaining question is whether to extend
   `g_prov_evict` (`MOE_EXPERT_CACHE_PROVISIONAL`, `moe-expert-cache.cu:722`) to "evicted but intact" — on UMA
   the bytes are already in DRAM, so a second chance is free in a way a discrete card cannot have.
4. **How much pipelining is enough?**  Fills must be issued behind the previous layer's compute (§3.2); the
   open question is whether one-layer lookahead is enough or the plan needs a two-layer ring, which is a
   measurement (P4 arm), not a preference.
5. **Does this become one placement for both APU shapes** (Linux gfx1151: 2 GiB carve-out / 130 GB GTT) and a
   Windows APU with a fixed VRAM partition, or does the carve-out keep `MOE_EXPERT_CACHE_UNIFIED`?  Different
   detections, measured on one box so far.
6. **gfx1150** (Strix Point, smaller pool): nothing measured.  Take it or leave it.

### 7.1 Slot granularity — the pros and cons (asked 2026-11-11; the answer is "per expert, coalesced I/O")

The delivery keys its cache per *tensor* already: its heat map has 144 tables on Flash-Next (48 layers x
`gate`/`up`/`down`), so the natural slot is **one (tensor, expert) slab** — 0.45-0.92 MiB, measured.  The three
options:

| unit | pros | cons |
|---|---|---|
| **per (tensor, expert) slab** — the recommendation | matches the heat map's own unit, so residency, LFRU and the MHT2 prior all work unchanged; a miss costs 0.4-0.6 ms and brings in exactly what the `MUL_MAT_ID` will read; a hot expert keeps its slot while a cold one beside it does not | ~3 reads per expert (gate/up/down are separate tensors at separate offsets — 1.78 MiB per expert per layer); scattered I/O is latency-bound (measured), hence the coalescing rule below |
| **per layer-bank** (all 512 experts of one layer = ~450 MiB on Flash-Next) | one huge sequential read per layer (5.75-6.2 GB/s measured), trivial coalescing, and a bank is an all-or-nothing residency decision | a *single* cold expert forces streaming the whole bank; a decode step uses top-k (2 %) of it, so the read is 50x the useful bytes; banks that large also collide with the budget — and it destroys the heat locality that makes the small store possible |
| **per expert, all three tensors fused** (only possible if the store is filled at load, not from the GGUF) | one 1.78 MiB fill per expert | needs a repacked file (a second copy of the weights) — out of scope for a read-only GGUF |

**And the I/O unit is not the slot unit**: a fill is a *coalesced span over the layer's used set* (the discrete
`MOE_HOST_POOL_STAGE_SPAN` shape, chunks of ~16 experts), so the scattered-latency problem is paid once per
layer rather than once per expert.  That combination — fine slots, coarse reads — is the recommendation.

## 8. Promotion criteria (what turns this into a block)

* A model whose expert set **exceeds RAM** runs on `halo`, same-seed byte-identical text against the best
  fitting arm, with the census showing real fills (cold and warm counted separately) and real evictions.
* Prefill streams only non-resident experts: the census's per-layer read bytes fall when a bank is already
  resident, on a repeated-prompt arm.
* `scripts/gate-uma-experts.sh` (spec in [`IMPLEMENTATION.md`](IMPLEMENTATION.md) §7) green, including the
  cold-page-cache arm, and `check-gpu-faults.sh` clean.
* No regression on the discrete boxes (the store is unreachable there; the pool/arena path must be unchanged
  bit for bit).
* Every new variable in `ENVIRONMENT.md` with its class; beneficial features **on by default** with a kill
  switch; a dated `WORKLOG.md` entry; the block note in `patches/README.md` (block 09/12 own this code).

## 9. File map

| file | what |
|---|---|
| `README.md` | this file: the shape, what it replaced, the measured foundation, the step/prefill protocol, the code map, the traps |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | the phased plan: P1.0 keystone probes, the store, residency/eviction, the fills, prefill streaming, preflight/fit, the gate spec, docs, effort, rollback |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | the probe log (dated record, not edited in place): box config, every number, reproduction commands; §13 is the 2026-11-11 P1.0 record and the expert-traffic concentration |
| [`probe/`](probe/README.md) | the ten P0 probes, the four P1.0 probes (11-14), and two measurement helpers (`gguf-expert-geom.py`, `moe-heatmap-read.py`); one workload probe still owed (15) |
