# archive/work/apu-expert-gtt — a UMA-only, heat-mapped **GTT expert store** (no host pool)

> **PROMOTED as block 16 in r263 (`v17-a691b2345-r263`).**  This directory is now the campaign's *record*
> (design, probes, every measurement, the trap list) — the code it describes ships in
> [`patches/0016-rdna-boosts-block-16-the-UMA-GTT-expert-store-an-APU.patch`](../../patches/), its knobs are in
> [`ENVIRONMENT.md` §20](../../ENVIRONMENT.md), its block note in [`patches/README.md`](../../patches/README.md),
> its rules in [`INVARIANTS.md`](../../INVARIANTS.md) ("Block 16"), and its gate is
> [`scripts/gate-uma-experts.sh`](../../scripts/gate-uma-experts.sh).  `gtt-expert-store.patch` here is the
> pre-promotion artifact, kept for provenance; do not apply it — apply the delivery set.
> Status below is the campaign's own, as written while it was WIP.

**Status: P0-P5 COMPLETE and measured; P6 (gate + docs) DONE except the promotion itself, which waits on the
maintainer's two decisions (README §8).**  Nothing in `patches/`, nothing applied to the fork, nothing pushed.
`halo`, gfx1151.  DONE: P0 (the ten probes), the P1.0 keystone probes, P1.1 + P2 (the GTT store itself),
P3a/P3b (the store as the expert cache's ARENA -- it serves, it evicts, and a 123.45 GiB model runs on this
123 GiB box), P3c (the MTP/nextn head pinned at full residency outside `-lme`), P4a/P4b/P4e/P4f (the fills on
r262's adaptive transport, the direct fill, the "no fill worker" answer, and the RDNA3_5 UMA start arm) and
**P5 (prefill streaming: only what is not resident, coalesced and read in parallel)**.  The plan is
[`IMPLEMENTATION.md`](IMPLEMENTATION.md); the probe log and every number are
[`MEASUREMENTS.md`](MEASUREMENTS.md) (§25 is P5); the probe sources are [`probe/`](probe/README.md); the arms are
[`arms.sh`](arms.sh); the gate is [`../../scripts/gate-uma-experts.sh`](../../scripts/gate-uma-experts.sh).

The implementation is ONE **cumulative** patch, cut against the **r262** `release.json` tree (`base` `a691b2345`,
tip `d65b0e889`) and applied with `git am`; it produces tree `5b74fa1b…`.  Its sha256 advances as the campaign
does — **the value below is always the current artifact** (older `WORKLOG` entries quote the value that was
current then):

| artifact | sha256 |
|---|---|
| [`gtt-expert-store.patch`](gtt-expert-store.patch) — P1.1 (`-lme` sizes the GTT store) + P2 (the region, the device buffer type, the load-time fill, the lock policy, the census, the kill switches) + P3a/P3b (the store as the cache's arena: `try_arena()`, the mapping holes, the forced mapped load, the budget/cap guards, the `GTT arena tier:` census) + P3c (the MTP/nextn head pinned outside `-lme` and never evicted; the SEPARATE drafter model sharing the one store safely) + P4a (the fills run on r262's adaptive `map`/`file`/`dio` host-read transport) + P4b (the buffered arm reads straight into the arena slot: +15.6 % decode warm) + P4f (on RDNA3_5 UMA the ladder starts on the buffered arm: +13.7 % warm, +56-67 % on a >RAM model) + **P5 (prefill streaming: the residents by D2D, the rest coalesced and parallel -- neutral on a fitting model, +63-67 % on a >RAM model, logits unmoved)** | `0049156434112f49133b791ddc30a0852f99b338b3e898c7ee3b5c0f468105a0` |

> **Next session: [`NEXT-SESSION-PROMPT-P5-P6.md`](NEXT-SESSION-PROMPT-P5-P6.md)** is now a RECORD (it drove this
> session); what is left is the promotion decision (§8) and the measurement debt named at the end of the newest
> `WORKLOG.md` entry.  The re-runnable tools are [`arms.sh`](arms.sh), [`guard.sh`](guard.sh),
> [`cold-cache.py`](cold-cache.py) and [`../../scripts/gate-uma-experts.sh`](../../scripts/gate-uma-experts.sh).

**Cross-machine (2026-11-11, `MEASUREMENTS.md` §27):** the discrete side is measured, not claimed.  On `soar`
(3x R9700 gfx1201, ROCm 10.1.0) the campaign's envs forced ON are **inert** (zero GTT lines, identical text
`a99ccbd97ecb`), `-lme` still means the host-expert budget, **`scripts/gate-host-pool.sh` is 42/42 green** with the
patched build, and the pool path's throughput is within noise (pp512 1750/1774 against 1745/1755, tg128 62.8/61.3
against 62.9/60.7 t/s).

**Where the wins are, in one line each:** the store itself is a *fittability* tier (a 123.45 GiB model on a 123 GiB
box); the decode wins are P4b (+15.6 %) and P4f (+13.7 % warm, +56-67 % on a >RAM model); the prefill win is P5
(+63-67 % on a >RAM model, neutral on a fitting one).  **`MOE_EXPERT_GTT_ARENA` stays default-OFF** — it is a
fittability feature (it costs decode on a config whose experts fit and it forces a mapped load), and that is the
one recorded exception to the default-on policy, pending the maintainer's decision (§8).

**The campaign's environment variables** (not in `ENVIRONMENT.md`: nothing here is in `patches/` yet, so these
are documented here and will move there on promotion — per AGENTS.md rule 9, with their class):

| variable | default | class | what |
|---|---|---|---|
| `MOE_EXPERT_GTT` | `1` | **kill switch** | `0` = no GTT store, i.e. today's behaviour and log, byte for byte |
| `MOE_EXPERT_GTT_RO` | `1` | **kill switch** | `0` = register the region writable instead of ReadOnly (the measured 232 → 60 GB/s A/B) |
| `MOE_EXPERT_GTT_MLOCK` | `1` | **kill switch** | `0` = do not `mlock`: the tier is then reclaimable by design (diagnostic / bisection) |
| `MOE_EXPERT_GTT_ARENA` | **unset = `-lme` decides** | **kill switch / override (tri-state)** | unset = **arena iff the store cannot hold the whole expert set** (i.e. `-lme N` asked for a cap below the set; `-lme auto` on a fitting model does not), placement otherwise; `1` forces the arena, `0` forces placement.  `-lme`'s promise is *"cap the NON-SWAPPABLE memory spent on MoE experts"*, and only the arena keeps it: the excess streams from the **file-backed page cache (reclaimable)** instead of becoming anonymous device memory -- measured, 13.6 GiB less non-reclaimable memory on a 35B at `-lme 4`, at 2x the decode cost (maintainer decision 2026-11-11; `MEASUREMENTS.md` §26.1) |
| `MOE_EXPERT_GTT_FILL_DIRECT` | `1` | **kill switch** | `0` = the buffered arm fills through the bounce ring instead of straight into the arena slot (P4b's A/B) |
| `MOE_EXPERT_GTT_PREFILL` | `1` | **kill switch** | `0` = the delivery's per-expert used-expert copy for the prefill band (P5's A/B) |
| `MOE_EXPERT_GTT_PREFILL_READERS` | `8` | tuning | the P5 reader-pool width; `1` reads inline on the caller (the parallel-transport A/B) |
| `MOE_EXPERT_GTT_PREFILL_VERIFY` | unset | diagnostic | compare every coalesced span against the model mapping and log the first mismatch |
| `MOE_EXPERT_GTT_PREFILL_FAULT` | unset | diagnostic | make every span read fail, so every window takes the per-expert source (the gate's fault arm) |
| `MOE_EXPERT_GTT_PREFILL_DEBUG` | unset | diagnostic | one line per (table, ubatch): used / resident / read / arena / fallback |
| `MOE_HOST_LIMIT_FRAC` | `80` | tuning (existing) | the `auto` ceiling as a % of `MemAvailable`, shared with the discrete arm |
| `MOE_EXPERT_CACHE_TOUCH` | `2` | tuning (existing) | the admission threshold — a **documented no-op on this tier** (every miss fills; asserted by the gate's `touch1/2/3` arm) |
| `MOE_HOST_POOL_STAGE_SPAN(_EXPERTS)` | `1` (16) | tuning (existing) | the coalesced-read shape P5 reuses; `=0` turns it off and the tier's `file` transport is then refused by design |

The store's **size is not a variable**: it is `-lme`.  **`-lme` is also the opt-in for STREAMING** (maintainer
decision, 2026-11-11): with it the store may become the arena and stream the part of a set larger than RAM from the
file-backed page cache; without it the store is only a *placement* (a fitting model's hot subset registered, which
is the delivery's default-on behaviour) and a model whose weights do not fit RAM is **refused at load, in 0.2 s,
with the fix** -- never placed and left to swap.  `MEASUREMENTS.md` §26.1a.  Nothing is added for `-fit`, the reach limit, or admission.

**Naming rule (maintainer, 2026-11-11): never write a bare "store".**  It is always a **GTT store** (or
"GTT expert store"), so it can never be read as disk/SSD storage, and so it is never confused with the
slab's **arena** -- which it is *functionally* the same thing as, with a different implementation.  This
applies to code comments, `-lv` census text and every record in this directory.

**Scope: Linux only, for now** (maintainer, 2026-11-11).  The GTT store declines on Windows (the single-address
premise does not hold on WDDM) and the run keeps today's behaviour there.  Nothing in the patch forecloses a
Windows arm later: the platform-specific part is **one function** (`gtt_store_create_locked()` -- `mmap` +
`MADV_HUGEPAGE` + the one registration, already `#if !defined(_WIN32)`-guarded), the buffer type is a plain
device buft whose backing only has to be host-writable, and the loader's fill is `read_raw()` into
`tensor->data`, which Windows has too.  A Windows arm would replace that one function and either teach the
buffer type a host/device alias pair or drop the "one address" assertion with a second pointer.

Renamed from `wip/apu-expert-pool/` on 2026-11-11: **there is no pool in this design any more** (§1), and the
thing the tier holds is a GTT store of resident experts, not a cache tier under a host pool.  Cut against the
**r262** tree unless a file:line reference names an older one.

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
| **the tier** | ONE anonymous region: `mmap` + `MADV_HUGEPAGE`, `hipHostRegister(Mapped\|ReadOnly)`, sliced into per-expert slots.  It **is** the resident expert set — there is no pinned master and no host pool below or beside it.  **Naming, agreed 2026-11-11: the GTT store IS the expert arena** — the slot provider the expert cache's tables allocate from — and it is **explicitly NOT the slab's "arena"**, which is the discrete boxes' VRAM slab allocator.  Same role, different implementation; the two words must not be used interchangeably |
| **what it is made of** | registered THPs, so the GPU reads the resident set at **230-233 GB/s** (measured, §2) and the host fills it through its own writable mapping with **no flush** (measured FRESH) |
| **residency** | the delivery's existing **heat table + slot map** (block-09/12 machinery).  On UMA nothing decides *location* — host RAM and "VRAM" are one DRAM — so residency is the whole policy |
| **a miss** | evict the coldest slot (a **table** operation), then load straight into the vacated bytes, then continue.  No copy-out, no `madvise`, no unregister on this path |
| **the one knob** | **`-lme N`** (GiB) sizes the GTT store, and `-lme auto` derives it -- the *same* flag, the same units and the same "ranked by the heat prior" policy the discrete box uses for its pinned per-expert pool.  **It is the one command line that carries across from discrete mode to APU mode** (§5) |
| **admission** | **Nth touch**, not first -- the delivery's existing `admit=touch` policy (`MOE_EXPERT_CACHE_TOUCH`, default **2**; the 2-vs-3 A/B is measured, see §7 Q1).  A non-admitted read must land *somewhere* GPU-readable, so the GTT store keeps a small **transient band** (the analogue of the pool's in-place cold read, §5.1) |
| **eviction** | **LFRU over the resident set** (the delivery's existing policy), as confirmed by the maintainer on 2026-11-11 |
| **the load** | zero-copy into the vacated slot: one `pread` into it when the source is cold, or one copy-engine pull from the registered read-only **file mapping** when it is in the page cache — **and pipelined behind the previous layer's compute**, because a synchronous per-layer fill cannot decode (§3.2) |
| **prefill** | per layer, stream in only the experts that are **not already resident**, coalesced — the discrete host-expert wide-prefill table's shape, without a pool to stage through |
| **what stays refused** | `-ncmoe` / `-cmoe` / `--host-experts pinned` / `--host-experts pool` on an integrated GPU stay exactly as r236 left them — this tier is not the host-expert tier and does not re-open it |
| **the nextn (MTP) head** | its expert tables are **pinned in the GTT store and NOT counted against `-lme N`** (decision 2026-11-11: the head's weights are treated as metadata overhead -- read on every draft step, so they must never be the thing a budget evicts).  P2 does not implement this yet: the head's 3 tables are simply part of the candidate set (see §7 Q10) |

## 1. What this replaced, and why (2026-11-11)

The campaign opened (2026-11-09) with a different plan, and the maintainer's directive is that it was the
wrong one.  The differences, so a reader of the old record can map it:

| old plan (superseded) | now |
|---|---|
| **P1**: enable `--host-experts pool` / `-lme auto` on `IGPU` (the `ROCm_HostMap` mapping master + a bounded pinned pool) | **dropped.**  No host pool on UMA.  `llama_model_loader::set_host_experts_umem()` (`src/llama-model-loader.cpp:130`) keeps refusing the host tier there, unchanged |
| **P2**: the arena as the disk-backed cache; slab sized from a RAM budget; fills into the arena slot; a staging/bounce ring for the disk tier | **same shape, different memory.**  No slab, no arena, no staging ring: the region is registered anon THP and the disk fill is zero-copy into the slot |
| **P0.5**: `MOE_HOST_POOL_RO` — read-only registration for the *pool's* device alias | becomes `MOE_EXPERT_GTT_RO`: the **GTT store** is registered `ReadOnly` from creation.  Same 60 -> 233 GB/s fact, applied to the tier that now exists |
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
| registration is a hold against a *user-initiated* discard, and it outlives `hipHostUnregister` while the VMA lives | 4 GiB: control reclaims 0/1 048 576 pages; registered holds 1 048 576 | `09` |
| **…but it is NOT a hold against the kernel's reclaim under memory pressure** — see §15 | registered anon, 2 048 MiB under a 6 GiB cgroup: 0-436 MiB swapped out, huge pages split every run, the next GPU read 12-37x slower (one-off), `Unevictable` 0.  `mlock`ed: 0 taken, huge pages intact, `Unevictable` 2 048 | `16` |
| **trap**: `MADV_DONTNEED` on a registered region the GPU is reading is a **silent zero** | GPU sum `0333…` -> `0000…`, no fault, no error | `06` |
| **trap**: a GGUF tensor offset is **32-byte** aligned, not 4096 | `dd iflag=direct` at a real expert offset copies **0 bytes**; at `align_down(off, 4096)` it works (the bytes then land `+2 688` into the slot) | `14`, `dd` |

P1.0 (§13) re-measured several of these at the GTT store's own shapes; where a figure differs (e.g. `pread` into
anon: **3.6 GB/s** at 1 GiB in §4, **4.75-6.4 GB/s** at 256 MiB in §13) the newer number at the GTT store's own shape is the
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

Two probes are still owed, and neither blocks the design's memory:

* **`15` -- the per-session touch distance** (how often an expert is used 1x, 2x, 3x in one session), which
  sets the admission threshold `N`.  The persisted heat map gives the long-horizon concentration (§3.2) but
  not a session sequence.
* **`16` -- reclamation under pressure.**  Probe `09` shows a registered range ignores `fadvise`/`madvise`;
  whether the kernel's reclaim can still take those pages under real memory pressure -- and what
  `Unevictable` / `MemAvailable` / `mincore` each report while it does -- is **unmeasured**.  §5's "the budget
  is a reservation" rests on that, so measure it before a >RAM model trusts the budget.

### P1.1 + P2 -- the GTT store exists and is measured (BUILT 2026-11-11; `MEASUREMENTS.md` §14)

The patch is [`gtt-expert-store.patch`](gtt-expert-store.patch) (sha256 in the status header above).  What
landed, and the number that says it works:

| what | where | measured |
|---|---|---|
| the region: `mmap(PROT_READ\|PROT_WRITE, MAP_PRIVATE\|MAP_ANONYMOUS\|MAP_NORESERVE)` + `MADV_HUGEPAGE`, then ONE `cudaHostRegister(Mapped\|ReadOnly)` over the budgeted extent; `cudaHostUnregister` **then** `munmap` at teardown | `ggml/src/ggml-cuda/gtt-expert-store.cu` | **26 ms/GiB** to create+register (19.1 GiB in 482-486 ms), **100 %** of it huge pages *after* the load, and the host VA == the device VA is **asserted** at creation (a second address declines the GTT store) |
| the buffer type: a DEVICE buft with the plain device type's vtable contract, a per-device unique name, one page of alignment, and a clean NULL decline when a carve does not fit | `ggml-cuda.cu` (`ggml_backend_cuda_gtt_store_buffer_type*`), the KV-region pattern | one carve per load (`19 126.5 MiB` / `40 925.2 MiB`); the expert tables stay in that device's split, so `MUL_MAT_ID` reads them in place -- **no new op, no dispatch, no per-layer copy, and no scheduler input-copy of the table** |
| `-lme` is the GTT store's size (P1.1), resolved before any tensor is placed, with the discrete arm's own budget and heat ranking | `llama-model-loader.cpp` `plan_gtt_expert_store()`, called from `plan_host_expert_placement()`'s UMA branch | `-lme 6` → **exactly 6 144.0 MiB** realised (6 116.0 MiB carved); `auto` → the whole set when it fits (123 or 144 tables, **zero evictions**), the ceiling as a hot subset when it does not (`MOE_HOST_LIMIT_FRAC=5` → 34 tables) |
| the load-time fill: the buffered `pread` arm straight into the slot, no alignment round-out | `llama-model-loader.cpp` `load_data_for()` | every expert byte of the carve is read from the GGUF (18 662.0 / 40 925.0 MiB, the carve's remainder being the buffer type's per-tensor padding); **13.9 GB/s** with the file warm, **2.33 GB/s** on a cold 46 GiB file (a *load* cost, not a decode one) |
| one `-lv 4` census (extent requested vs realised, budget source, registration mode, huge-page share, registration cost, slots, fill bytes, MemAvailable after) + the "ALL-RESIDENT arm only" scope line | `ggml_cuda_gtt_store_report()` | every arm prints it; `MemAvailable` falls by exactly the extent |
| the kill switch `MOE_EXPERT_GTT=0` | the CUDA device hook returns NULL, the loader plans nothing | same-seed greedy text **identical**, and the `-lv 4` log normalises to the **same 329 lines** as the pre-P2 build |
| `MOE_EXPERT_GTT_RO` (default 1) | the registration flags | RO is within noise of the plain device buffer (-0.4 / -1.25 % pp512); **WRITABLE is -5.9 / -6.2 %**, i.e. the 232-vs-60 GB/s fact applies to this region |

**The design's memory is now confirmed at GTT store scale, with two corrections:**

1. **Registration is cheaper than P1.0 predicted** -- 26 ms/GiB at the real extent against the probe's
   30.5 ms/GiB, so the `memset` pre-fault walk is not worth taking.
2. **The load fill is page-cache-bound, and that is the one number that got worse as the GTT store got bigger**:
   18.7 GiB in 1.34 s when the GGUF is resident (13.9 GB/s), 40.9 GiB in 17.6 s when it is not (2.33 GB/s).  It
   changes nothing about decode, but it is the GTT store's **startup** cost on a >RAM set and P4/P5 do not address it.

And **one measured non-fact to keep straight**: decode throughput does not move at all between the GTT store and the
plain device buffer (63.22-63.34 t/s over six runs).  While the set fits, the GTT store's value is what it makes
*possible*, not what it makes faster (§5 item 3).

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
(`moe_cache_stage_table()`, `MOE_HOST_POOL_STAGE_SPAN`, `moe-expert-cache.cu:8134`), with the GTT store replacing
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

1. **The hit rate is the lever, and overlap is NOT a given.**  The obvious escape — issue the next layer's fills
   behind the current layer's compute — was **built and measured on the discrete pool and it is a wash** (r261,
   `archive/work/host-pool-demand-overlap/`, `benchmarks/2026-11-11-host-pool-slice-read.md`): the exact
   within-layer lookahead measured **±1.5 %** (noise), because the host copies each split's host-weight inputs —
   beginning with a synchronize — *before* it dispatches that split, so at the instant the routing (and the miss
   set) is known the device has nothing in flight; and the pool's own placing fill (0.131 ms) is cheaper and
   earlier than any prefetch.  `TODO` #132 retired on that finding.  **Our fill is 3.2x that round trip (0.420
   ms) because the source is storage**, so the trade is not settled here — but P4 must *test* whether any
   overlap is reachable on this path and must not assume it.  Until it is shown, plan on the hit rate:
   `-lme`/coverage (§5) and admission (§5.1, §7 Q1) are the only knobs that move the cost.
2. **Admission costs one read and saves it on every later use**, so a 2nd-touch promotion pays for itself at two
   uses in a session, and a 3rd-touch policy only wins while the resident budget is the binding constraint
   (i.e. it is a *footprint* lever, not a cost lever).  Hence `N` is a knob with a measured default, and the
   per-session touch distance is the one probe still owed.
3. **Prefill is cheap and decode is not**: a layer span streams at 5.75 GB/s, a scattered 0.88 MiB slab is
   latency-bound — so prefill streams spans (P5), and decode lives or dies on the hit rate (item 1).

**How big must the GTT store be?**  From the delivery's own persisted heat maps (MHT2, `moe-heatmap-read.py`), over
*(tensor, expert)* slots: on **35B-A3B Q4_K_M** (18.22 GiB of experts, 7.5 M accumulated accesses) **21 %** of
the slots were **never touched**, and 90 % of traffic is covered by **35.5 %** of the slots = **6.46 GiB** (95 %:
8.05 GiB).  The Flash-Next map looks far more concentrated (13.1 % at 90 %), but it has 27x less history, so it
is undersampled — **plan on the long-history number**.  The never-touched mass is what an Nth-touch policy keeps
on disk for free, and it is why "hot subset" (below) is the right default.

## 4. Code map: what is reused, what is new, what is deleted on the UMA arm

**Reused as-is (do not re-derive):** the **`-lme` budget and its heat ranking** (`plan_host_expert_placement()`, `src/llama-model-loader.cpp:277-310`, and `pool_rerank_from_heat_locked()`, `moe-expert-cache.cu:2905`) -- the APU GTT store is the same decision at a finer granularity, so the size and the order come from that code, not from a second policy -- plus the heat table and its MHT2 sidecar prior (`apply_heat_seed_locked`), the
slot/remap indirection (`g_dr_key_to_id`, `moe_cache_remap`), the per-layer access plan
(`access_locked()`, `moe-expert-cache.cu:3495`), the census lines, the prefill in-place table read
(`moe_cache_in_place`, r258, `ggml-cuda.cu:9455`), and the staging/span machinery
(`moe_cache_stage_table()`, `:8134`) — the last one reused for **prefill** (P5) rather than for decode fills.

**New:** the GTT store itself —
* allocation: `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)` + `MADV_HUGEPAGE`, then
  `hipHostRegister(Mapped|ReadOnly)` over the budgeted extent (§5), sliced into per-expert slots;
* a **buffer type** for expert tables with the delivery's KV-region pattern
  (`ggml_backend_cuda_kv_region_buft_split_for_dev`, `ggml-cuda.cu:10697`): a device-carrying buft with the
  plain device type's vtable contract, so the scheduler places expert tensors in it and `MUL_MAT_ID` reads it
  in place with no new dispatch and no new op;
* the fill: `fill_slot_from_host()` (`moe-expert-cache.cu:3452`) retargeted to the two arms of §3.1.

**What P1.1 + P2 actually landed (2026-11-11, `gtt-expert-store.patch`), so a later reader does not have to
re-derive the shape from the plan above:**

| file | what is in it |
|---|---|
| `ggml/src/ggml-cuda/gtt-expert-store.cu/.cuh` (new, ~470 lines) | the region's lifetime (mmap + THP + ONE registration, `unregister` then `munmap` at exit; the single-address assertion), the bump allocator the buffer type carves from, the `/proc` helpers, and the one `-lv 4` census |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the **buffer type** (`ggml_backend_cuda_gtt_store_*`: per-device name, page alignment, the plain device type's `get_alloc_size`, memcpy `set`/`get`/`memset`/`clear`, a NULL decline when a carve does not fit), the two device hooks, and one line in `supports_buft` |
| `src/llama-model-loader.{cpp,h}` | `plan_gtt_expert_store()` (the UMA arm of `-lme`: candidates, budget, ceiling, heat ranking, whole-table selection), the placement hook in `buft_for_tensor()`, the `pread`-into-the-slot fill in `load_data_for()`, and the census call |
| `common/arg.cpp` | the UMA branch of `-lme`'s help text and of its "90 % of available memory" refusal |
| `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend.cpp` | the two appended device hooks (`gtt_store_open` / `gtt_store_report`) and their wrappers |

**Not built yet, deliberately:** `fill_slot_from_host()` is untouched (P4 owns the two fill arms); the eviction
path does not exist (P3), so the residency unit is one whole expert table; `moe-expert-cache.cu` is unchanged --
its integrated-device branch never runs on this box anyway (the preflight is host-expert-gated and an APU
registers no host-expert bytes), which is why `MOE_EXPERT_GTT=0` and `=1` differ only in the new code.

**Deleted *on the UMA arm only*:** the pinned host pool and its master, the `access_locked()` arena->pool
hand-over, `bounce_acquire()`'s staging ring for the GTT store's fills, and any slab sizing for this tier.  The
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
| **`auto`** | fits under `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable` -> mirror the whole set; otherwise a bounded per-expert pool | **fits** -> the whole expert set is resident in the GTT store (zero evictions); **does not fit** -> `MOE_HOST_LIMIT_FRAC` x `MemAvailable` of GTT store, i.e. a hot subset with evictions on |
| **`N`** | pin N GiB, spent on whole tensors in descending hotness (fit case) or as a per-expert pool (not-fitting case) | **exactly N GiB of registered GTT store**, spent in descending hotness, **plus the MTP/nextn head's tables on top of it** (P3c: the head is read every draft step, so the budget does not trade it away).  **What landed (P2) spends it on WHOLE expert tables**; in the arena tier (P3) the unit is one (tensor, expert) slot |

Three things the APU arm has to decide differently, and they are P1's work:

1. **The ceiling is different.**  `MemAvailable - KV - compute - OS reserve`, read **once**, floored -- never
   "free VRAM" (the 2026-10-19 over-commit: 121 GiB resident + 13 GiB swap).  The GTT store is *registered host*
   memory, so it is not charged to the GTT pool (`amdgpu.gttsize`, 121 GiB on `halo`) -- what bounds it is the
   pinning rule and the non-swappable ceiling.  A 46 GiB userptr registration is measured to work (§9), so the
   GTT store can be large; no ceiling above that has been measured.
2. **`-lme` must select the GTT store without `-ncmoe`.**  On a discrete box host experts exist only because
   `-ncmoe`/`-cmoe` moved them there, so `-lme` is a *cap on a placement someone else asked for*.  On an APU
   there is no host tier (`-ncmoe` is refused, r236), so `-lme` becomes the **only** expert-residency knob and
   has to engage the GTT store by itself.  Its help text needs the UMA branch; the wording carries over almost
   verbatim ("N GiB, spent on the hottest slots, the rest stays on storage").
3. **Only the not-fitting band is a win.**  A set that already fits needs no GTT store: the delivery's plain
   placement reads GTT at **232.6 GB/s** and the registered GTT store reads at **232.1** -- the same.  The GTT store's
   value is entirely in the set that does **not** fit, where the alternative is a mapping read through the page
   cache (or a fault).  So `auto` engages in the not-fitting band, and an explicit `-lme N` is the middle
   ground.  Default-on policy: the GTT store is **on** with `auto` (there is nothing to opt into), and
   `MOE_EXPERT_GTT=0` is the kill switch.

Whatever the budget leaves is the page cache, which is where the warm fill arm's source comes from -- so the
number implicitly decides the cold:warm ratio, which is why the census must print both counters.

**Measured, P1.1 + P2 (2026-11-11, `MEASUREMENTS.md` §14.4).**  The three arms the gate asks for, on the two field
models, with the census's own numbers:

| arm | extent requested -> realised | tables / slots | zero evictions? |
|---|---|---|---|
| 35B-A3B, `-lme auto` (set fits) | 19 126.5 -> **19 126.5 MiB** | 123 / 31 488 (the whole set) | **yes**, and the census says so |
| 35B-A3B, `-lme 6` | 6 144.0 -> **6 144.0 MiB** | 36 / 9 216 (6 116.0 MiB carved) | n/a -- a hot subset |
| Flash-Next, `-lme auto` (set fits) | 40 925.6 -> **40 925.6 MiB** | 144 / 73 728 (the whole set) | **yes** |
| 35B-A3B, `auto` with `MOE_HOST_LIMIT_FRAC=5` | 5 804.7 -> **5 804.7 MiB** | 34 / 8 704 | n/a -- the not-fitting arm |

So `-lme N` **is** the extent (page-rounded), `auto` is a cap not a floor, and the waste between the extent and
the carved bytes is under one expert table (155 MiB on 35B-A3B) -- the granularity of this phase.

**Settled (maintainer, 2026-11-11): a best fit, not a strict fit.**  The extent is the budget and the carved
bytes may sit up to one expert table under it; the registered-but-unoccupied tail is the price of the region
being exactly the size that was asked for, and it is the right trade.  P3 does not change the intent -- a
slot-granular residency unit simply makes the carve converge on the extent, so the tail disappears by itself.

**A resident slot is not *discardable* by the usual means** — probe `09` measured a registered range keeping
1 048 576 of 1 048 576 pages after `fadvise(DONTNEED)` **and** `madvise(DONTNEED)`, where the unregistered
control released all of them, and the hold outlives `hipHostUnregister` while the VMA lives.  So the only
*user-initiated* discard is `hipHostUnregister` **then** `madvise`, never while a launch can address the region.

**But the budget is NOT a reservation, and that was measured on 2026-11-11 (probe `16`, §15).**  `09` tested a
user-initiated discard; the kernel's own reclaim under pressure is a different mechanism, and it **does** take
this tier's pages: `hipHostRegister` on an `mmap` region does not pin it, so under real pressure the kernel splits
the huge pages and swaps out 0-436 MiB of a 2 048 MiB region (4 of 8 runs at one pressure level), with the next
GPU read paying a **one-off 12-37x** stall (524 ms and 2 191 ms against 42-59 ms) before going back to normal.  It
never faults and never corrupts — the driver re-faults the pages in — and **`Unevictable` reports 0 the whole
time**, so the tier is reclaimable *and invisible* to a memory monitor.

**`mlock()` is the answer, and it is now IMPLEMENTED (2026-11-11, `MOE_EXPERT_GTT_MLOCK`, default 1).**  An
`mlock`ed region kept **all** its pages and **all** its huge pages in every probe run (4 of 4), read at the same
26-28 ms, and is the only arm the accounting reports (`Unevictable`/`Mlocked` = 2 048 MiB).  It costs
`RLIMIT_MEMLOCK` — unlimited on `halo` (its `/etc/security/limits.conf` raises it), **8 MiB** on a stock systemd
box (`#DefaultLimitMEMLOCK=8M`), 64 KiB for an unraised non-root process.  So the extent is clamped to the limit
and a model that only fits *because* of the GTT store is **refused** rather than silently degraded (§5.2, §7 Q11).

**mlock is free at creation**: 26.3 ms/GiB for a 19 GiB store against 25.8-26.0 before it, and 26.0 vs 25.9 for
the 41 GiB one — the registration has already faulted every page in, so the `mlock` is a bookkeeping pass over an
existing VMA, not a second walk of DRAM.

### 5.2 The lock, and what happens when it cannot be honoured (implemented 2026-11-11)

The GTT store is registered **and `mlock`ed**; that is what makes `-lme` a reservation (probe 16, §15).  The lock
limit is `RLIMIT_MEMLOCK`, and the extent is resolved against it **before any tensor is placed or any byte is
read**, in `plan_gtt_expert_store()`:

| | |
|---|---|
| `extent <= lockable` | honour it.  The census reports `mlocked X MiB (RLIMIT_MEMLOCK …): a real reservation` |
| `extent > lockable`, the **model fits in RAM without the store** (`weights <= 90 % of MemAvailable`) | **decline the store** (or clamp it to the limit, when the clamp still holds one expert table), WARN once with the numbers and the fix, and carry on with today's behaviour.  It is an optimisation here, not a lifeline |
| `extent > lockable`, the **model cannot fit in RAM** | **REFUSE**, with the shortfall, the limit, and the fix.  Measured on `Qwen3.5-122B-A10B-Q8_0` (weights 126 403 MiB against 114 656 MiB available): **0.76 s** and one message, instead of the 2026-10-19 over-commit (121 GiB resident + 13 GiB swap) |
| `MOE_EXPERT_GTT_MLOCK=0` | no lock, no policy: the tier is reclaimable by design, the census says `NOT locked … RECLAIMABLE under pressure`.  The kill switch for A/B and bisection |

The refusal is gated on a capability flag (`gtt_store_can_stream`, `false` until P3/P4) so that a clamp which
*would* cover the shortfall still refuses today: without eviction and streaming fills the experts that did not fit
would take their layer's device buffer anyway, and the load would over-commit.  The message says so explicitly
rather than promising that raising the limit is enough.

`weights` is every tensor in the loader's index (`ggml_nbytes` summed), and the 90 % is the same reserve the
`-lme` argv check already applies — one constant, not two.  It is a *lower bound* check, deliberately not a second
`--fit` (which sizes the context against free device memory — a different question).

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
* **Do not size the GTT store from "free VRAM".**  Measured over-commit; the guard stays.
* **Do not clamp the ARENA tier's budget against free VRAM or the slab either** (2026-11-11, `MEASUREMENTS.md`
  §16.1): `alloc_all_locked`'s clamp is meaningless once every slot comes from the store, and on an APU the slab's
  arena capacity is 0, so it sized the arena to **zero slots** (`layer 0 cannot hold a uniform arena above 0
  slots (requested 0)`) with the store wide open behind it.  The extent is the bound; a 1.25x cap is asserted.
* **Do not arm the movable-boundary slab on the arena tier.**  The backend's preflight wrapper arms it whenever
  the cache is on, and a UMA slab sizes itself as `free VRAM - reserve` = the whole machine.
* **Do not let the arena tier run on an unmapped load.**  An IGPU device reports `mmap_support = false`, so AUTO
  resolves to `--load-mode none` and the "mapping holes" become a **pageable full copy of the expert set**
  (measured 18 662 MiB on 35B-A3B) -- the exact memory the tier exists to avoid.  The arm forces `use_mmap`.
* **Do not let a second model in one process size, grow or take the arena.**  The GTT store is ONE region per
  device for the whole process (a separate MTP drafter, a reload, `llama-bench -m A -m B`).  A second load sizes
  its placement against the store's **free** bytes only, never takes the arena (its extent is the trunk's budget,
  and carving a head out of it drops the trunk's residency -- measured 100 % -> 69 %), and adopts the region only
  up to what is free (`MEASUREMENTS.md` §18).
* **Do not treat a failed carve as a soft decline.**  `alloc_buffer` returning null **fails the model load**
  (`unable to allocate ROCmGTTStore0 buffer`); the old comment claimed otherwise and a `-lme 1` + separate-drafter
  arm died on it.  Size against the free bytes, and treat a null carve as the fatal sizing bug it is.
* **Do not let anything evict the draft weights** (maintainer, 2026-11-11).  The embedded head is pinned at one
  slot per expert, so the policy cannot evict it, and the three out-of-policy paths (pressure stand-down, partial
  shrink, post-drop re-arm) must skip it.  The census line `MTP/nextn head residency: … X eviction(s)` must read
  0 and `slots == experts`; measured at 7 670 trunk evictions against 0 on the head (`MEASUREMENTS.md` §19).
* **Do not assume r262's transport ladder reaches the tier.**  It is initialised at the first POOL build and the
  tier has no pool, and the per-table file source was registered only when the pool was on -- so a pin or
  `--load-mode buffered|dio` was silently ignored and an escalated `file` arm could read the mapping anyway
  (`MEASUREMENTS.md` §20.1).  Both are fixed; keep them fixed.
* **Do not read the transport numbers as decode numbers.**  They are per-read rates into the bounce entry; the H2D
  and the op's own cost sit on top, and P4 has no throughput claim (`MEASUREMENTS.md` §20.4).
* **Do not run a measurement arm without freezing the heat-map prior** (2026-11-11, `MEASUREMENTS.md` §23).
  The sidecar is `0644` and every run adds its own tally to it and rewrites it, so arms contaminate each other and
  **run order becomes part of the result** -- measured: the `auto` ladder's latch rate looked like 4/5 with a
  drifting prior and is **0/5** with a frozen one, which inverted a conclusion.  Set `MOE_EXPERT_HEATMAP=load`
  (or `chmod a-w` the sidecar -- the file's mode IS the freeze signal, latched and logged); the delivery's own
  gates already default to `load` because "the persisted prior is an INPUT".  `arms.sh` does it for every arm.
* **Do not mark a span-ring buffer busy when its H2D is issued -- mark it at ACQUISITION** (2026-11-11, P5,
  `MEASUREMENTS.md` §25.5).  `span_buf_acquire` only refuses a buffer whose copies have drained, so a ring slot
  claimed for a prefill window and marked busy later is handed out AGAIN to the next window in the same call: the
  last read wins, every H2D copies it, and the greedy text comes out short (509 vs 522 chars).  Three symptoms
  from one cause -- the file read verifies clean, and the *used set* looks degenerate (8 experts/table) only
  because the model has already degenerated, so its later routings really are experts 0..7.  The gate asserts
  `PREFILL_VERIFY` + text + the read/used ratio together for exactly this reason.
* **Do not read the routing ids again in a backend hook.**  Take the scheduler's used-expert BITSET.  The ids
  buffer the scheduler read is reused by the graph allocator, and a later read of it served a stale routing (P5's
  first implementation did exactly that; the bitset is also cheaper and cannot disagree with the copy it replaces).
* **Do not expect the split-input copy to hold anything between ubatches.**  The graph allocator gives a prefill
  graph ONE `input_cpy` per copy id for every table (their lifetimes do not overlap), so "do not re-fetch a bank
  the previous chunk already holds" cannot be served there -- measured 113 owner changes over 117 calls.  The
  residency that survives is the GTT store's (P5 subtracts that).
* **Do not judge this tier by text alone, and never by a `-lv` run's text.**  The `tables=0` census produced
  byte-identical text with the mechanism completely inert (`MEASUREMENTS.md` §16.1), and a `-lv 4` census of the
  122B gave two different hashes on the same seed.  Text arm: no `-lv`; the mechanism is proved by the census.
* **Do not `O_DIRECT` a GGUF offset directly.**  Tensor offsets are aligned to `general.alignment` (32 bytes),
  so an `O_DIRECT` fill must round the read out to the block size and carry a per-fill delta into the slot — or
  be buffered.  Measured: an unaligned `O_DIRECT` read copies **0 bytes**.
* **Do not port the tensor-split slice-read logic** (r261's `MOE_HOST_POOL_SLICE_READ`: on a `-sm tensor` split
  each device owns a contiguous slice, and reading the whole expert doubled the bytes).  The GTT store is
  single-device, so the device's slice IS the expert -- r261 measured the flag inert at 1 GPU "as the
  geometry predicts".  The class only returns if the GTT store is ever split across devices.
* **Do not use `VmPin`/`VmLck` to detect the registration hold** -- both read 0 (measured, `09`); use
  `mincore()`, and do not confuse "a registered range ignores `fadvise`/`madvise`" (measured) with "the kernel
  cannot reclaim it under pressure" (probe `16`, owed).
* **Do not touch the `integrated` host-buffer (zero-copy UMA weights) path.**  It has its own history and its
  own guard (`ggml_backend_sched_buffer_supported()`); `GGML_FORCE_NO_INTEGRATED=1` is the A/B that proves the
  GTT store does not depend on it.
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
2. **Does the GTT store hold the hot subset or the whole expert set?**  The heat-map measurement (§3.2) says the hot
   subset is the right default: on 35B-A3B, 90 % of traffic comes from 35.5 % of the slots and 21 % are never
   touched at all.  The whole set only makes sense on a model whose expert set fits the budget outright.
3. **Eviction policy**: LFRU is confirmed.  The remaining question is whether to extend
   `g_prov_evict` (`MOE_EXPERT_CACHE_PROVISIONAL`, `moe-expert-cache.cu:722`) to "evicted but intact" — on UMA
   the bytes are already in DRAM, so a second chance is free in a way a discrete card cannot have.
4. **Is *any* fill overlap reachable on this path?**  r261 refuted the premise for the discrete pool and measured
   the exact lookahead at ±1.5 % (§3.2); the structural obstacle — the host builds a split's host-weight inputs
   before dispatching it — is the same shape here.  Our fill is 3.2x more expensive (storage, 0.420 ms), so the
   question is open, and it is the first thing P4 should settle: either an overlap exists and the hit-rate
   requirement relaxes, or it does not and `-lme` coverage carries the whole design.
5. **Does this become one placement for both APU shapes** (Linux gfx1151: 2 GiB carve-out / 130 GB GTT) and a
   Windows APU with a fixed VRAM partition, or does the carve-out keep `MOE_EXPERT_CACHE_UNIFIED`?  Different
   detections, measured on one box so far.
6. **gfx1150** (Strix Point, smaller pool): nothing measured.  Take it or leave it.
7. **The load fill is page-cache-bound, and nothing in P3-P5 fixes it** (measured, §14.3): a 41 GiB GTT store takes
   17.6 s to populate off a cold 46 GiB file (2.33 GB/s) against 1.34 s when the file is resident (13.9 GB/s).
   It is a *startup* cost, not a decode one, but on a >RAM model whose file can never be fully cached it is the
   floor on every start.  **Owned elsewhere** (maintainer, 2026-11-11): a separate development line is adding
   **explicit buffered reads** instead of the mmap-only load, precisely because the page cache chokes under
   low-free-memory conditions.  This GTT store inherits that work for free -- its fill is a buffered `pread` into the
   slot, so it only needs the *read* to stop depending on the page cache; nothing here needs to change.
8. **`--fit` does not account for the GTT store** (P1.3 of the plan).  The probe load is `no_alloc`, so the GTT store is
   deliberately not planned there and the fit pass reserves nothing for it -- correct for the probe, but it means
   `--fit` sizes the context against a memory budget the GTT store will later take 19-41 GiB out of.  It did not bite
   on this box (the set fits with 45 GiB to spare); it will bite on a set that does not.
9. **The whole-table granularity of `-lme N`** (measured §14.4: 6 144.0 MiB requested, 6 116.0 MiB carved) is a
   P2 artefact, not the design, and **the best-fit behaviour is what was chosen** (maintainer, 2026-11-11).  P3 is
   what turns the residency unit into one (tensor, expert) slot; until then a budget smaller than one table buys
   nothing, and `-lme N` is exact to the extent, not to the carve.
10. **The MTP (nextn) head's expert tables -- RESOLVED (maintainer, 2026-11-11): pinned in the GTT store, and NOT
   counted against `-lme N`.**  The head's weights are treated as metadata overhead: read on every draft step, so
   they must never be the thing a budget evicts, and a budget must never be able to buy itself out of them.  On an
   APU there is no second placement to give them anyway (one DRAM; the delivery's "keep the speculative head on
   the device" rule exists to stop `-ncmoe` moving it to *host* memory, which is not a separate thing here).
   **P2 does not implement the accounting yet**: the head's 3 tables are part of the candidate set and therefore
   *are* counted against the budget (35B-A3B: 123 = 40x3 trunk + 3 nextn).  The change is small -- take them out
   of `cands`, place them unconditionally, and size the extent as `nextn + min(budget, trunk)` -- and it belongs
   with P3, where "pinned" first means something (in P2 everything resident is pinned).  Until then a hot-subset
   `-lme N` can rank a head table out of the GTT store; it then keeps its layer's device buffer, so it stays readable
   and the text does not change.

11. **Should the GTT store `mlock` its region? -- RESOLVED (maintainer, 2026-11-11): yes, and it is
   user-actionable.**  Implemented as: `mlock` by default (`MOE_EXPERT_GTT_MLOCK=0` is the kill switch), the
   extent clamped to `RLIMIT_MEMLOCK`, a **decline** when the model fits in RAM anyway, and a **refusal** (with
   the fix) when the model cannot run without the store and the store cannot be locked.  So "cannot lock" is a
   startup-time, documented, actionable condition rather than a silent stall — §5.2 for the table, §15 for the
   measurement behind it.

### 7.1 Slot granularity — the pros and cons (asked 2026-11-11; the answer is "per expert, coalesced I/O")

The delivery keys its cache per *tensor* already: its heat map has 144 tables on Flash-Next (48 layers x
`gate`/`up`/`down`), so the natural slot is **one (tensor, expert) slab** — 0.45-0.92 MiB, measured.  The three
options:

| unit | pros | cons |
|---|---|---|
| **per (tensor, expert) slab** — the recommendation | matches the heat map's own unit, so residency, LFRU and the MHT2 prior all work unchanged; a miss costs 0.4-0.6 ms and brings in exactly what the `MUL_MAT_ID` will read; a hot expert keeps its slot while a cold one beside it does not | ~3 reads per expert (gate/up/down are separate tensors at separate offsets — 1.78 MiB per expert per layer); scattered I/O is latency-bound (measured), hence the coalescing rule below |
| **per layer-bank** (all 512 experts of one layer = ~450 MiB on Flash-Next) | one huge sequential read per layer (5.75-6.2 GB/s measured), trivial coalescing, and a bank is an all-or-nothing residency decision | a *single* cold expert forces streaming the whole bank; a decode step uses top-k (2 %) of it, so the read is 50x the useful bytes; banks that large also collide with the budget — and it destroys the heat locality that makes the small GTT store possible |
| **per expert, all three tensors fused** (only possible if the GTT store is filled at load, not from the GGUF) | one 1.78 MiB fill per expert | needs a repacked file (a second copy of the weights) — out of scope for a read-only GGUF |

**And the I/O unit is not the slot unit**: a fill is a *coalesced span over the layer's used set* (the discrete
`MOE_HOST_POOL_STAGE_SPAN` shape, chunks of ~16 experts), so the scattered-latency problem is paid once per
layer rather than once per expert.  That combination — fine slots, coarse reads — is the recommendation.

## 8. Promotion criteria (what turns this into a block)

* A model whose expert set **exceeds RAM** runs on `halo`, same-seed byte-identical text against the best
  fitting arm, with the census showing real fills (cold and warm counted separately) and real evictions.
* Prefill streams only non-resident experts: **DONE (P5)** -- `read` is strictly below "fetch every used expert"
  (73.8 % on a fitting model, 41.3 % on a >RAM model), measured on a prefill-band arm with the prior frozen.  The
  reading of "a bank the previous chunk already holds" that the plan assumed (the DESTINATION buffer) is
  impossible by construction, and the one that works (the STORE's residents) is what P5 subtracts -- see
  `MEASUREMENTS.md` §25.5.
* `scripts/gate-uma-experts.sh` (spec in [`IMPLEMENTATION.md`](IMPLEMENTATION.md) §7) green, including the
  cold-page-cache arm, and `check-gpu-faults.sh` clean.  **WRITTEN AND GREEN: 35 checks, 16 arms**
  (`WORKLOG.md` 261111.026).  The prefill-logit KLD gate is owed as well (a prefill-path change) and reads mean
  KLD -0.000000 / same-top-p 99.990 % on `--ab MOE_EXPERT_GTT_PREFILL=0`.
* **The two maintainer decisions are taken** (the arena tier's default, and whether to revive the transient
  band): they are the only thing between this set and `patches/`.
* No regression on the discrete boxes (the GTT store is unreachable there; the pool/arena path must be unchanged
  bit for bit).
* Every new variable in `ENVIRONMENT.md` with its class; beneficial features **on by default** with a kill
  switch; a dated `WORKLOG.md` entry; the block note in `patches/README.md` (block 09/12 own this code).

## 9. File map

| file | what |
|---|---|
| `README.md` | this file: the shape, what it replaced, the measured foundation, the step/prefill protocol, the code map, the traps |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | the phased plan: P1.0 keystone probes, the GTT store, residency/eviction, the fills, prefill streaming, preflight/fit, the gate spec, docs, effort, rollback |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | the probe log (dated record, not edited in place): box config, every number, reproduction commands; §13 is the 2026-11-11 P1.0 record and the expert-traffic concentration, §14 the same-day P1.1+P2 record (keystone re-run, registration cost at real extent, the fill rates, the `-lme` arms, the read-rate A/B) |
| [`guard.sh`](guard.sh) | run ONE arm under the campaign's barriers (address space, `RLIMIT_MEMLOCK`, a `SwapFree` monitor that kills the run on the first drop) -- every arm from P3 on used it |
| [`cold-cache.py`](cold-cache.py) | drop a GGUF's page cache (no root) for a cold arm, with the caveat that a fitting model's long arm is really "cold first touch, warm repeats" |
| [`NEXT-SESSION-PROMPT-P5-P6.md`](NEXT-SESSION-PROMPT-P5-P6.md) | the cold-start handover for P5/P6 |
| [`gtt-expert-store.patch`](gtt-expert-store.patch) | **the implementation so far**: P1.1 + P2 + P3a/P3b/P3c as ONE patch cut against the r262 `release.json` tree `891d4acb…` (`git am`; sha256 in the status header) |
| [`arms.sh`](arms.sh) | the P1.1+P2 arms: `text` (same-seed greedy, no `-lv`), `census` (`-lv 4`), `rate` (`llama-bench`, the ReadOnly/writable/absent A/B), plus the GPU-fault scan.  `TREE=~/llama-scratch ./arms.sh all` |
| [`probe/`](probe/README.md) | the ten P0 probes, the four P1.0 probes (11-14), and two measurement helpers (`gguf-expert-geom.py`, `moe-heatmap-read.py`); probes 15 (touch distance) and 16 (reclamation under pressure) still owed |
| [`NEXT-SESSION-PROMPT-P3.md`](NEXT-SESSION-PROMPT-P3.md) | **the cold-start prompt for the P3 session** (the one to paste): where the branch is, the two open blockers, the barrier safety protocol, and the definition of done |
| [`NEXT-SESSION-PROMPT.md`](NEXT-SESSION-PROMPT.md) | the cold-start prompt that drove the P1.1/P2 session (what it is, the numbers not to contradict, the definition of done).  It is now a **record**, not the current prompt: the next session starts from `IMPLEMENTATION.md` P3 |
