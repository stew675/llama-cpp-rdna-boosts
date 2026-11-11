# IMPLEMENTATION — `wip/apu-expert-gtt`

Phased plan for the UMA-only heat-mapped **GTT expert store** (no host pool).  Read [`README.md`](README.md)
first — §0 is the shape, §2 the measured foundation (P1.0 is now answered), §3.2 the cost model, §6 the traps.
Numbers cited here are from [`MEASUREMENTS.md`](MEASUREMENTS.md); file:line references are the **r259** tree
(`release.json` tree `7995db44…`, tip `c80fa8bda…`).

**Rules for the whole campaign.**  Every phase ships behind a switch whose default is decided by measurement;
every phase's accept criteria include a byte-identical-text arm; **P1.0 is answered** (2026-11-11: all four assumptions hold, `MEASUREMENTS.md` §13), so implementation may
start -- with the two rules that fell out of it: fills must be **pipelined** (P4) and `O_DIRECT` needs an
alignment round-out (P4).

---

## P0 — the facts (DONE, 2026-11-09)

Ten probes, no root, nothing changed in the fork.  [`MEASUREMENTS.md`](MEASUREMENTS.md),
[`probe/`](probe/README.md).  What the new design takes from them:

1. `prop.integrated` is one DRAM: 2 GiB visible carve-out, 130 GB GTT, HIP `totalGlobalMem` 121.1 GiB.
2. A registered **read-only** range reads at **230-233 GB/s**; the same range registered writable at **60**.
   A `PROT_WRITE` buffer can be registered read-only *after* it is filled.
3. `pread(O_DIRECT)` into **anonymous** memory works (**3.6 GB/s**); into a `hipMalloc` GTT buffer it is
   **`EINVAL`** — which is why the store must be anonymous, and why the old arena-shaped plan needed a staging
   ring this design does not have.
4. Host writes into a registered range are visible to the GPU with **no flush**, in both registration modes.
5. Registration pins, and the hold survives `hipHostUnregister` while the VMA lives; `MADV_DONTNEED` on a
   registered region the GPU is reading is a **silent zero**.
6. There is no fault-and-retry on this part (`PageableMemoryAccess = 0`, no `xnack` target): an unregistered
   mapping is not in the GPU's page tables, so **every byte the GPU reads must be registered**.

## P1.0 — the four keystone probes (DONE 2026-11-11, all four HOLD)

`probe/11-14`, results in `MEASUREMENTS.md` §13.  Nothing was implemented before these answered, and nothing was
falsified:

| # | question | answer |
|---|---|---|
| 11 | do `MADV_HUGEPAGE` THPs survive `hipHostRegister(Mapped\|ReadOnly)`? | **yes** -- `AnonHugePages` 1024.0 MiB of 1024 before *and* after registering, both fault orders; 17.4 ms/GiB pre-faulted; GPU read 232.1 GB/s |
| 12 | **`pread` into the registered RO region, then a GPU read of exactly those bytes** | **yes** -- `O_DIRECT` accepted (4.75 GB/s), sums match on every round *including* the reuse/eviction arm; buffered is faster (15.7 GB/s) |
| 13 | the copy-engine pull from a registered RO `MAP_SHARED` file mapping | **yes** -- 0.012 ms for a 0.68 MiB slab warm (60.9 GB/s), 0.570 ms for a 16 MiB span (29.4 GB/s), bytes verified |
| 14 | per-expert cost at the real shape | cold **0.413 ms** median (0.567 buffered, 34.9 ms worst first-read-after-idle), warm **0.069-0.381 ms**, `pread -> launch -> sync` **0.420 ms**, 14.1 MiB span **5.75 GB/s** |

**Two rules fell out, and they are requirements, not notes:**

1. **A GGUF tensor offset is 32-byte aligned, not 4096**, so an `O_DIRECT` fill must round the read out to the
   block size and carry a per-fill delta into the slot (measured: 2 688 bytes on `blk.2`); an unaligned
   `O_DIRECT` read copies **0 bytes**.  The buffered arm needs no delta and is faster at slab granularity on
   `halo`, so both arms stay.
2. **A fill cannot be synchronous per layer** — see the cost model (README §3.2).  P4 owns pipelining.

**Still owed, two probes:** `15`, the per-session touch distance (a workload measurement) -- it sets the
admission default `N`, and nothing is blocked on it because the cost model says `N = 2` already pays for
itself; and `16`, **reclamation under pressure** -- whether the kernel's reclaim can take a registered
region's pages under real memory pressure.  Probe `09` only shows that a *user-initiated* `fadvise` +
`madvise` cannot, so P1.2's "the budget is a real reservation" needs `16` before a >RAM model leans on it.

## P1 — preflight, detection and **`-lme`** sizing (the gate becomes a policy)

> **STATUS 2026-11-11: P1.1 and P1.2 are BUILT** (`p2-gtt-expert-store.patch`, `MEASUREMENTS.md` §14).  What the
> accept criteria measured: `MOE_EXPERT_GTT=0` reproduces today's log byte for byte and the same text;
> `-lme 6` realises **exactly 6 144.0 MiB** and the census names the source; `auto` in the fitting band is the
> whole set with zero evictions (and says so) and with `MOE_HOST_LIMIT_FRAC=5` it is a bounded hot subset
> instead; no `-ncmoe` is required anywhere.  **Two deviations, both recorded:** (a) the integrated branch in
> `moe_cache_shared_pool_device()`/`moe_cache_preflight()` needed **no change at all** -- that preflight is gated
> on host-expert bytes, and an APU registers none, so the guard never runs on this box and the cache is already
> inert; `MOE_EXPERT_GTT=0` and `=1` therefore differ only in the new code (`moe-expert-cache.cu` is untouched);
> (b) `--fit` does **not** see the store (a `no_alloc` probe load skips the plan by design), which is the
> "does not double-reserve" criterion met the weak way -- P1.3 remains open (README §7 Q8).

Today `moe_cache_shared_pool_device()` (`ggml/src/ggml-cuda/moe-expert-cache.cu:5623`) and
`moe_cache_preflight()` (`:5641`) take the integrated branch and **disable the whole cache**
(`g_enabled = false`).  That branch is what this campaign replaces — not "relaxes".

### P1.1 — the size comes from `-lme` (the one knob that carries over from discrete mode)

**Decision (maintainer, 2026-11-11): on an APU `-lme N` is the size of the registered GTT expert store** — the
one command-line entry that carries across from discrete GPU mode.  It is already spelled
`--limit-host-experts auto|N` and already means *"cap the NON-SWAPPABLE memory used by host-resident MoE experts,
N GiB, spent in descending hotness from the persisted heat map"* (`common/arg.cpp:2767`), so this is a policy
branch, not a new flag:

| `-lme` | behaviour on an APU |
|---|---|
| `auto` | the set fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable` -> the whole set is resident (mirror, zero evictions); otherwise that same ceiling becomes a **bounded hot-subset** store with evictions on |
| `N` (GiB) | exactly N GiB of store, spent at one **(tensor, expert) slot** at a time in descending heat order — finer than the discrete fit case, so `-lme 6` means 6 GiB used |
| (unset) | = `auto` (default-on policy: nothing to opt into).  `MOE_EXPERT_GTT=0` is the kill switch and restores today's behaviour verbatim |

* **Reuse the ranking, do not write one.**  `plan_host_expert_placement()` (`src/llama-model-loader.cpp:277-310`)
  already turns `-lme` into a byte budget and ranks tensors by the persisted heat prior; the APU store needs the
  same decision at a finer granularity (per slot rather than per tensor), so the budget and the ordering come
  from that function and `pool_rerank_from_heat_locked` (`moe-expert-cache.cu:2905`), not from a second policy.
* **Finish it before any tensor is placed**, like the discrete path: the store's extent, the transient band
  (README §5.1) and the not-fitting decision are all load-time, logged once, and visible to `--fit` (P1.3).
* **A set that already fits is not a win** (232.6 GB/s plain GTT vs 232.1 registered, README §5 item 3), so `auto`
  engages the store only in the not-fitting band; an explicit `-lme N` is the middle ground and the A/B lever.
* The arg needs the **UMA branch in its help text and its validation**: the existing text is written entirely in
  terms of `-ncmoe` and a pinned master, and its "N above 90 % of available memory is refused" rule becomes the
  store's own non-swappable ceiling (README §5 item 1).
* `MOE_EXPERT_CACHE_UNIFIED=1` + explicit MIB + `GGML_CUDA_SLAB_RESERVE_MIB` stays as the **carve-out** arm
  (Windows APUs with a fixed VRAM partition).  UMA and carve-out are different detections; do not fold them.

### P1.2 — the budget's two ceilings

* `MemAvailable - KV - compute - OS reserve`, read **once**, floored, and **asserted** in the census
  (`store + KV + compute < MemTotal`).  `MemAvailable` is not cgroup-aware in either call site —
  `wip/host-expert-coverage/` owns that; until it lands, stay with `MemAvailable` and document it.
* **Not** the GTT pool: the store is registered *host* memory, so `amdgpu.gttsize` does not bound it (a 46 GiB
  userptr registration is measured, `MEASUREMENTS.md` §9).  Do not apply the slab's free-VRAM rule here — and do
  not let the slab see the store either (this tier takes nothing from it).

**Accept (P1):** with `MOE_EXPERT_GTT=0`, `halo` reproduces today's log and behaviour byte for byte; `-lme 6`
produces a store within a slot of 6 GiB and the census names the budget and its source; `-lme auto` in the
fitting band keeps everything resident and in the not-fitting band reports evictions; `--fit` does not
double-reserve the store; no `-ncmoe` is required (and `-ncmoe` is still refused).

## P2 — the store itself (allocation, buffer type, in-place reads)

> **STATUS 2026-11-11: BUILT and MEASURED** (`MEASUREMENTS.md` §14).  The accept criteria: same-seed greedy
> text **byte-identical** to `MOE_EXPERT_GTT=0` on 35B-A3B *and* Flash-Next (and to the pre-P2 build); the GPU
> read rate is **within noise** of the plain device buffer (-0.4 / -1.25 % of pp512, against a run-to-run drift of
> ~1 %); `MOE_EXPERT_GTT_RO=0` is **materially slower** (-5.9 / -6.2 %), so the flag does something; no fault, and
> `check-gpu-faults.sh` clean over every arm.  Two things the plan did not anticipate, both recorded in §14:
> registration is **26 ms/GiB** at the real extent (cheaper than the 30.5 ms/GiB P1.0 predicted, so no pre-fault
> walk), and the **load fill is page-cache-bound** (13.9 GB/s warm against 2.33 GB/s off a cold 46 GiB file), which
> is a startup cost P4 does not address (a separate development line is adding explicit buffered reads for the
> load, README §7 Q7).  **The fill here is the load-time one**; P4's miss fill is untouched.
>
> **Scope: Linux only** (maintainer, 2026-11-11) -- the store declines on Windows and the run keeps today's
> behaviour there; the platform-specific part is one `#if !defined(_WIN32)` function, so a Windows arm later
> replaces that function and nothing else (README, status header).  **Granularity: best fit** (same decision) --
> the extent is the budget and the carve may sit up to one expert table under it; P3 makes the two converge.

* Allocation: `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)` + `MADV_HUGEPAGE` over the
  budgeted extent, then `hipHostRegister(Mapped|ReadOnly)`; slice into per-expert slots.  Windows keeps a
  one-shot `hipHostAlloc` arm (the growable shape is POSIX-only), as block 15's KV mirror already does.
* Buffer type: follow the delivery's **KV-region pattern** — a device-carrying buft with the plain device
  type's vtable contract (`ggml_backend_cuda_kv_region_buft_split_for_dev`, `ggml-cuda.cu:10697`), so the
  scheduler places expert tensors in it and `MUL_MAT_ID` reads them **in place**: no new op, no new dispatch,
  no copy per layer.  It must answer `get_name` per device (`llama_kv_cache` learnt that the hard way) if it is
  ever doubled, and it must decline cleanly where a shape does not fit.
* Census: one `-lv 4` line with the extent, registered RO or not, huge-page granularity (P1.0), and the budget
  source.
* `MOE_EXPERT_GTT_RO` (default `1`): the read-only registration.  `0` = writable, the A/B that proves the
  60 -> 233 GB/s claim applies to *this* region and not just to a probe.

**Accept:** an all-resident config on `halo` is **byte-identical** (same-seed greedy text, hashed with no
`-lv`) to the same run with the store off; the store's GPU read rate is within noise of a `hipMalloc` arena on
the same model; no fault.

## P3 — residency and eviction (table only, no data movement)

Reuse, do not rebuild: the heat table and its MHT2 prior (`apply_heat_seed_locked`), the slot/remap
indirection (`g_dr_key_to_id`, `moe_cache_remap`), and `access_locked()` (`moe-expert-cache.cu:3495`) as the
per-layer access plan.  **The eviction policy is the delivery's LFRU, over the resident set** (confirmed by the
maintainer 2026-11-11) -- do not invent a second policy.

* **Admission reuses the delivery's `admit=touch` policy and its knob** — `MOE_EXPERT_CACHE_TOUCH`, default
  **`2`**, no new variable.  `N = 3` was already measured on the discrete path (2026-11-07 A/B: churn damping but
  no measurable decode win, -0.8 pt hit rate, +35 % cold misses, `README.md` §7 Q1) and stays the candidate to
  re-test **here**, where "served cold" is a 0.4-0.6 ms storage read rather than a mapped in-place read.
* **A non-admitted read needs a destination: the transient band** (README §5.1) — one or two registered slots per
  table, filled by the cold or warm arm, read by the `MUL_MAT_ID`, reused in place.  Without it, touch-count
  admission degenerates into first-touch admission.  Size it, count it in the budget, and measure whether one
  slot per table is enough (tables are resolved in sequence).
* **The slot is per (tensor, expert)** -- the heat map's own unit, 0.45-0.92 MiB measured -- while an **I/O is a
  coalesced span over the layer's used set** (README §7.1).  Fine slots, coarse reads.

* A miss drops the **coldest** entry's table mapping and reuses its slot.  **No copy-out**: the
  `access_locked()` arena->pool hand-over is skipped on this arm (there is no pool to preserve the bytes).
* **No `madvise`, no unregister on this path.**  They belong to the budget-shrink path only, with the region out
  of every graph (README §6).
* Optional second chance (P3.5, only if measured): extend `g_prov_evict`
  (`MOE_EXPERT_CACHE_PROVISIONAL`, `moe-expert-cache.cu:722`) to "evicted but intact" with a generation
  counter, so a re-admission is free.  On UMA the bytes are already in DRAM, so this is genuinely free — but it
  must not be able to mistake a stale entry for a fill.
* Census: `evictions`, `reuses`, `unregisters` (must be 0 on a steady-state arm), `store MiB` vs budget.

**Accept:** a config that **evicts** still produces byte-identical text to the same config with the store
large enough that nothing evicts (the eviction path changes *where* bytes are, never *which*); census shows
evictions > 0 and unregisters == 0.

## P4 — the two load arms

`fill_slot_from_host()` (`moe-expert-cache.cu:3452`) is the seam.  Two arms, chosen by the page cache, not by a
knob (README §3.1):

* **cold**: `pread(O_DIRECT)` straight into the vacated slot (aligned; the GGUF's expert rows are contiguous).
  Ordering: the launch follows the `pread`; assert the visibility with the P1.0-12 arm on every fill in a
  `SELFTEST` mode.
* **warm**: `hipMemcpyAsync` on the **compute stream** from a registered, read-only `MAP_SHARED` mapping of the
  GGUF.  The copy engine pulls, so the fill is in the same submit as the `MUL_MAT_ID` that needs it.
* **Alignment (measured):** a GGUF tensor offset is 32-byte aligned, so the `O_DIRECT` arm must round the read
  out to 4096 and write it to `slot_base` such that the expert's bytes land at `slot + (off - align_down(off,
  4096))`; the slot carries a per-fill delta.  An unaligned `O_DIRECT` read copies 0 bytes.  The **buffered**
  arm is the simpler zero-delta path and is *faster* at slab granularity on `halo` (0.567 vs 0.413 ms median,
  and no 35 ms idle-wake tail) -- keep both, and choose per arm by measurement, not by principle.
* **Overlap is an experiment, not an assumption (README §3.2).**  A step touches ~1 440 slots at 0.42 ms each, so
  a synchronous per-layer fill only decodes at a ~98 % hit rate -- which is why the escape matters.  But r261
  **built the exact within-layer lookahead on the discrete pool and measured ±1.5 %**, because the host
  materialises a split's host-weight inputs (synchronize included) before dispatching it, so nothing is in
  flight when the miss set is known, and the pool's placing fill (0.131 ms) is cheaper and earlier than a
  prefetch (`archive/work/host-pool-demand-overlap/`; `TODO` #132 retired on it).  So: **first reproduce the
  question here** -- does this path ever have work the fill can hide behind, given the store's fill is 3.2x the
  pool's -- and only then build a worker (`copy_stream` / `order_event`, `moe-expert-cache.cu:8406` is the
  shape).  If it does not, say so and drop the requirement: `-lme` coverage and admission carry the design.
  The gate's arm 9 asserts whichever answer is recorded, not a pipeline.
* Do **not** reuse the discrete staging ring (`bounce_acquire()`, `:3082`) or the coalesced-span staging
  (`moe_cache_stage_table()`, `:8134`) for the decode fills: both exist to serve a *pool*.  The span logic is
  reused for **prefill** (P5) and for the coalescing of a layer's used set (P3/P4), where it is the right shape.
* Diagnostic parity with the shipped path: a fill-fault arm (`MOE_EXPERT_GTT_FILL_FAULT=1`) that makes the load
  fail must **decline** the access (or abort the run cleanly) and never let a kernel read an unfilled slot.
* Census: `fills_cold`, `fills_warm`, bytes each, and per-layer read bytes — a measurement must be able to say
  which arm it measured.

**Accept:** same-seed byte-identical text on a **cold page-cache** arm (drop the file's cache first) against
the warm arm; both arms' census counters move as expected; `check-gpu-faults.sh` clean; the fill-fault arm
declines and does not corrupt.

## P5 — prefill: stream only what is not resident

* Per layer: compute the chunk's used-expert set (the existing per-layer access plan already does this), subtract
  the resident set, and stream the difference coalesced — this is exactly
  `moe_cache_stage_table()`'s span-read shape with the store as the destination.
* Keep the `MOE_HOST_POOL_STAGE_SPAN`-class coalescing (an expert's rows are contiguous; a layer span is
  sequential) — it is the same reason W4 measured +38 % on the discrete pool.
* Do not re-fetch a bank the previous chunk already holds; the census's per-layer read bytes must fall on a
  repeated-prompt arm.
* Ordering: a prefill fill is a bulk fill, so the warm arm is the copy engine and the cold arm is a batched
  `pread` (parallel, not one `pread` per expert) — measure both before choosing.

**Accept:** prefill logits identical to the all-resident arm (`scripts/gate-prefill-logits.sh` — this *is* a
prefill-path change); text identical; read bytes strictly below "fetch every used expert" on the
repeated-prompt arm; pp rate reported against the all-resident arm.

## P6 — gate, docs, promotion

* `scripts/gate-uma-experts.sh` (§7 below), `check-gpu-faults.sh`, `gate-prefill-logits.sh`,
  `gate-host-pool.sh` (the discrete path must be untouched) and `validate-set.sh` if anything regenerated.
* Docs per README §8.  `INVARIANTS.md` gains the measured facts it does not already have (no `xnack`; APU
  "VRAM" is the host pool; registered RO 233 vs 60 writable; registration pins and outlives unregister;
  `MADV_DONTNEED` on a GPU-visible registered region is a silent zero; a pageable mapping may be a fill source
  and never a read target; `O_DIRECT` targets anon and refuses `hipMalloc`).
* `ENVIRONMENT.md` gets the new variables with their class: `MOE_EXPERT_GTT` (kill switch, default `1` on an
  integrated GPU), `MOE_EXPERT_GTT_RO` (kill switch for the read-only registration), `MOE_EXPERT_GTT_FILL_FAULT`
  (diagnostic).  **The size is not a variable -- it is `-lme`**, and admission is not a variable either: it is the
  existing `MOE_EXPERT_CACHE_TOUCH`.  `-lme` itself gains the UMA branch of its help text in `common/arg.cpp`.

---

## §7 The gate: `scripts/gate-uma-experts.sh`

Modelled on `gate-host-pool.sh` (seven arms, byte-identical text, a fault arm that must decline).  Text is
hashed with **no `-lv`**; censuses are read at **`-lv 4`** — two runs per arm, never one.

| # | arm | pass criterion |
|---|---|---|
| 1 | **text identity** — store on vs `MOE_EXPERT_GTT=0`, on a config that **evicts** | same-seed greedy text **byte-identical** |
| 2 | **all-resident** — store big enough that nothing evicts | identical text to the plain-device arm; read rate within noise of a `hipMalloc` arena |
| 3 | **>RAM** — an expert set larger than RAM | completes; census shows fills (cold and warm) and evictions; no `HSA_STATUS_ERROR_MEMORY_FAULT`; `check-gpu-faults.sh` clean |
| 4 | **cold page cache** — the same run after dropping the file's cache | the cold arm is really exercised (its counters move); text identical to arm 1 |
| 5 | **budget** — explicit `MOE_EXPERT_GTT_MIB=N` | the store's extent ≤ N; `MemAvailable` after load above the OS reserve; the census names the source |
| 6 | **no-movement** — steady-state arm | `unregisters == 0` and no `madvise` on the hot path; evictions > 0 with text identical |
| 6b | **admission** — `MOE_EXPERT_CACHE_TOUCH=1/2/3`, same prompt set | a higher threshold never changes the text; the census shows fewer promotions, a smaller footprint and more transient fills; `2` must be no slower than `1` on a cold-start session (that is the default's justification) |
| 6c | **`-lme`** — `-lme 6`, `-lme 12`, `-lme auto`, same model/prompt | the store's extent matches the request (+-1 slot); the census's budget source names `-lme N` vs `auto`; `auto` in the fitting band reports zero evictions and in the not-fitting band non-zero; text identical across all four |
| 7 | **kill switch** — `MOE_EXPERT_GTT=0` | reproduces today's behaviour (cache disabled on IGPU) and its exact log |
| 8 | **fill-fault** — `MOE_EXPERT_GTT_FILL_FAULT=1` | declines cleanly; correct text; never a GPU read of an unfilled slot |
| 9 | **prefill** — repeated prompt with a resident bank | per-layer read bytes fall; prefill logits identical (from `gate-prefill-logits.sh`) |
| 10 | **host-buffer A/B** — `GGML_FORCE_NO_INTEGRATED=1` | throughput unchanged within noise, text identical: the store does not depend on the zero-copy UMA weights path |

Nothing in this gate runs on a discrete box; the delivery's existing gates are the discrete-side proof (the
store must be unreachable there).

## §8 Effort and sequencing (estimate, not a commitment)

| phase | scope | risk |
|---|---|---|
| P1.0 | four probes, no code — **DONE 2026-11-11**, all four hold; probe 15 (touch distance) still owed | none |
| P1 | preflight branch + budget + kill switch | low |
| P2 | store allocation + buft + census | medium: a new buffer type the scheduler must accept |
| P3 | residency/eviction wiring | low: the policy already exists |
| P4 | the two fill arms + ordering | medium: the cold arm's visibility must be proven, not assumed |
| P5 | prefill streaming | medium: prefill is where the read volume lives |
| P6 | gate + docs | low |

## §9 Rollback

`MOE_EXPERT_GTT=0` restores today's behaviour without a rebuild (the cache disabled on an integrated GPU, the
host-expert tier refused).  If a phase cannot be made switch-safe it does not ship — that is the lesson of the
guard this campaign replaces.  The store must never be the only path to a working configuration on a discrete
box, and it must be provably inert there (`GGML_FORCE_NO_INTEGRATED`, arm 10).
