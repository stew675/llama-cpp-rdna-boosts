# IMPLEMENTATION — `archive/work/apu-expert-gtt`

> **PROMOTED as block 16 of the delivery in r263 (`v17-a691b2345-r263`).**  This document is the dated
> campaign record; the shipping code is `patches/0016-…`, the knobs `ENVIRONMENT.md` §20, the rules
> `INVARIANTS.md` ("Block 16"), and the gate `scripts/gate-uma-experts.sh`.

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
   **`EINVAL`** — which is why the GTT store must be anonymous, and why the old arena-shaped plan needed a staging
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

> **STATUS 2026-11-11: P1.1 and P1.2 are BUILT** (`gtt-expert-store.patch`, `MEASUREMENTS.md` §14).  What the
> accept criteria measured: `MOE_EXPERT_GTT=0` reproduces today's log byte for byte and the same text;
> `-lme 6` realises **exactly 6 144.0 MiB** and the census names the source; `auto` in the fitting band is the
> whole set with zero evictions (and says so) and with `MOE_HOST_LIMIT_FRAC=5` it is a bounded hot subset
> instead; no `-ncmoe` is required anywhere.  **Two deviations, both recorded:** (a) the integrated branch in
> `moe_cache_shared_pool_device()`/`moe_cache_preflight()` needed **no change at all** -- that preflight is gated
> on host-expert bytes, and an APU registers none, so the guard never runs on this box and the cache is already
> inert; `MOE_EXPERT_GTT=0` and `=1` therefore differ only in the new code (`moe-expert-cache.cu` is untouched);
> (b) `--fit` does **not** see the GTT store (a `no_alloc` probe load skips the plan by design), which is the
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
| `auto` | the set fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable` -> the whole set is resident (mirror, zero evictions); otherwise that same ceiling becomes a **bounded hot-subset** GTT store with evictions on |
| `N` (GiB) | exactly N GiB of GTT store, spent at one **(tensor, expert) slot** at a time in descending heat order — finer than the discrete fit case, so `-lme 6` means 6 GiB used |
| (unset) | **PLACEMENT ONLY**: the store is still sized from `MemAvailable`, so a fitting model's hot subset is registered (neutral-to-positive, the delivery's default-on behaviour), but the **arena never engages and nothing is streamed** -- `-lme` is the streaming opt-in (maintainer decision, 2026-11-11).  A model whose weights do not fit RAM is **refused at load (0.2 s) with the fix**, not placed and left to swap.  `MOE_EXPERT_GTT=0` is the kill switch and restores today's behaviour verbatim.  See `MEASUREMENTS.md` §26.1a |

* **Reuse the ranking, do not write one.**  `plan_host_expert_placement()` (`src/llama-model-loader.cpp:277-310`)
  already turns `-lme` into a byte budget and ranks tensors by the persisted heat prior; the APU GTT store needs the
  same decision at a finer granularity (per slot rather than per tensor), so the budget and the ordering come
  from that function and `pool_rerank_from_heat_locked` (`moe-expert-cache.cu:2905`), not from a second policy.
* **Finish it before any tensor is placed**, like the discrete path: the GTT store's extent, the transient band
  (README §5.1) and the not-fitting decision are all load-time, logged once, and visible to `--fit` (P1.3).
* **A set that already fits is not a win** (232.6 GB/s plain GTT vs 232.1 registered, README §5 item 3), so `auto`
  engages the GTT store only in the not-fitting band; an explicit `-lme N` is the middle ground and the A/B lever.
* The arg needs the **UMA branch in its help text and its validation**: the existing text is written entirely in
  terms of `-ncmoe` and a pinned master, and its "N above 90 % of available memory is refused" rule becomes the
  GTT store's own non-swappable ceiling (README §5 item 1).
* `MOE_EXPERT_CACHE_UNIFIED=1` + explicit MIB + `GGML_CUDA_SLAB_RESERVE_MIB` stays as the **carve-out** arm
  (Windows APUs with a fixed VRAM partition).  UMA and carve-out are different detections; do not fold them.

### P1.2 — the budget's two ceilings

> **MEASURED 2026-11-11 (probe 16, `MEASUREMENTS.md` §15): the "non-swappable" half of this is FALSE as built.**
> The ceiling arithmetic (`MemAvailable - KV - compute - OS reserve`, read once) is implemented and holds, and the
> census asserts it against `MemTotal`.  But a region that is `hipHostRegister`ed and NOT `mlock`ed is reclaimable
> by the kernel: under real pressure it loses 0-436 MiB per 2 GiB (and its huge pages, every run), the next GPU
> read stalls 12-37x while the driver faults the pages back, and `Unevictable` reports 0 throughout.  **`mlock`
> is the difference between a reservation and a sizing intent** (README §7 Q11, an open decision).  Nothing else
> in this section changes: `MemAvailable` is charged the same either way.
>
> **NOW IMPLEMENTED (same day):** the store is `mlock`ed (`MOE_EXPERT_GTT_MLOCK`, default 1), the extent is clamped
> to `RLIMIT_MEMLOCK`, a fitting model **declines** the store with the fix in the message, and a model that cannot
> fit **refuses** at load with the shortfall and the fix.  `mlock` is free at creation (26.3 vs 25.8-26.0 ms/GiB --
> the registration already faulted the pages in).  README §5.2 is the table; `MEASUREMENTS.md` §15.3 the arms.

* `MemAvailable - KV - compute - OS reserve`, read **once**, floored, and **asserted** in the census
  (`GTT store + KV + compute < MemTotal`).  `MemAvailable` is not cgroup-aware in either call site —
  `wip/host-expert-coverage/` owns that; until it lands, stay with `MemAvailable` and document it.
* **Not** the GTT pool: the GTT store is registered *host* memory, so `amdgpu.gttsize` does not bound it (a 46 GiB
  userptr registration is measured, `MEASUREMENTS.md` §9).  Do not apply the slab's free-VRAM rule here — and do
  not let the slab see the GTT store either (this tier takes nothing from it).

**Accept (P1):** with `MOE_EXPERT_GTT=0`, `halo` reproduces today's log and behaviour byte for byte; `-lme 6`
produces a GTT store within a slot of 6 GiB and the census names the budget and its source; `-lme auto` in the
fitting band keeps everything resident and in the not-fitting band reports evictions; `--fit` does not
double-reserve the GTT store; no `-ncmoe` is required (and `-ncmoe` is still refused).

## P2 — the GTT store itself (allocation, buffer type, in-place reads)

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
> **Scope: Linux only** (maintainer, 2026-11-11) -- the GTT store declines on Windows and the run keeps today's
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
`-lv`) to the same run with the GTT store off; the GTT store's GPU read rate is within noise of a `hipMalloc` arena on
the same model; no fault.

## P3 — residency and eviction (table only, no data movement)

> **DIRECTION (maintainer, 2026-11-11): the GTT store IS the expert cache's ARENA.**  The existing tables allocate
> their slots from the GTT store region instead of from the slab/`hipMalloc` arena, and admission, LFRU, the
> remap/indirection, `access_locked()`'s plan and the censuses are reused unchanged -- which is also what keeps the
> campaign's "do not write a second policy" rule.  Concretely that means the integrated-device branch **selects the
> GTT store** rather than disabling the cache (the thing P2 found does not even execute on an APU today, because that
> preflight is host-expert-gated -- so P3 has to un-gate it deliberately rather than just edit it).
>
> **Vocabulary, and this has bitten before: the APU GTT store is the expert ARENA, but it is explicitly NOT the
> "arena" the slab allocator means.**  Same role (a slot provider for the cache's tables), different
> implementation (a registered anonymous region vs the discrete boxes' movable-boundary VRAM slab).  Never use the
> two words interchangeably in code, comments or records; say "GTT expert store" or "slab arena".
>
> **The nextn (MTP) head's tables are pinned in the GTT store and are NOT counted against `-lme N`** (same decision):
> the head is metadata overhead, read every draft step.  P2 counted them until P3c moved the accounting
> (2026-11-11, `MEASUREMENTS.md` §17) -- the head is no longer part of the set when it is not loaded, and both
> tiers now treat it as pinned overhead on top of the budget.

### P3.0 — the seams, read before writing anything (2026-11-11)

Four findings from reading `moe-expert-cache.cu` against the store.  They fix the shape of P3, and three of them
**simplify** the plan:

1. **The device-path interception already exists.**  `ggml_cuda_mul_mat_id()` (`ggml-cuda.cu:6471`) asks
   `moe_cache_get_table(dst, src0, ...)` at its entry and, on a hit, re-dispatches with `src0c.data = arena` and
   the slot-remapped ids.  So a table whose arena is the GTT store needs **no new op, no dispatch change and no
   copy**: the redirect is base-pointer-only.  Its lookup even has a semantic fallback
   (`(layer, role, device)`), which is what lets a table be found when the scheduler never aliased it -- the exact
   situation on an APU, where the expert weight is not a host-buft input.
2. **`access_locked()` is the whole residency policy and it is reusable as-is.**  Hit → `arena + slot*stride`;
   free slot → take it; full → **LFRU victim** (smallest decaying count, tie → oldest use) with the optional
   `admit=value|touch` doorkeeper.  It is exactly the design's §3 step, already written and measured on the
   discrete boxes.
3. **The transient band is NOT needed for the core loop** (and §5.1's rationale does not survive here).
   `access_locked` only *rejects* a miss when a cold path exists (`cold_ok`, i.e. `MOE_EXPERT_CACHE_COLD_UVA` +
   a UVA-dereferenceable master); with no cold path it says so in as many words -- "every mode fills and the
   policy is a no-op".  On an APU the in-place cold read of a pageable mapping **is** the fault (no
   `PageableMemoryAccess`, no XNACK), so `g_cold_uva` must be false for this tier and every miss evicts+fills.
   That is P3a; the band (and with it Nth-touch admission) becomes a P3b refinement, priced against a measured
   hit-rate loss rather than assumed.
4. **The source of truth on an APU is the model mapping, not a pinned master.**  `fill_slot_from_host()` →
   `expert_host_src()` resolves a `table_t` master with `t.host`/`host_bytes`/`src_off`; a pageable master is
   copied with `cudaMemcpyAsync(H2D)`, which is the delivery's known-bad path on RDNA.  So the APU arm must fill
   from the **mapping with a plain host `memcpy` into the registered slot** (the measured warm arm, 12.8 GB/s;
   a cold page faults the mapping in, which is the §3.1 "cold" arm).  That is the fill change -- and it is
   **NOT in yet**: the P3 arms run their fills through the delivery's existing pinned bounce ring
   (`expert_host_src` -> `memcpy` into a pinned entry -> H2D), which is correct and ordered but pays a second
   copy.  The direct `memcpy` needs an ordering argument first (a host write must follow the previous consumer's
   reads of the same slot, which the stream-ordered copy gets for free), so it travels with P4's fill work and
   its 12.8 GB/s warm arm (`MEASUREMENTS.md` §16.6).

So P3 is: **the store is the arena, the model mapping is the master, the cache is the policy.**  In the discrete
path's own vocabulary that is `--host-experts pool` with the pool replaced by the GTT store and no pinned master
-- which is precisely what this campaign set out to be.

**Stages, each independently verifiable:**

| stage | what | verify |
|---|---|---|
| **P3a-1** (DONE 2026-11-11) | the GTT store becomes a real allocator: first-fit over a sorted, coalesced free list (`reserve`/`release`/`largest_free`), replacing the bump pointer; the all-resident carve keeps working | P2's arms unchanged: text `627b2ebd7792` / `f586785d7183`, same carves, no faults |
| **P3a-2** (DONE 2026-11-11, PM) | `try_arena()` gains the store as a third arm (and the free path returns the range), `g_cold_uva = false` for the tier, and a preflight entry that enables the cache on an IGPU with the store's extent as its budget (no slab).  **It serves**: the 3 defects that made the census read `tables=0` are fixed (see the box below) | `MEASUREMENTS.md` §16.2: `M35 -lme 4` -> **120 tables / 6 720 slots / 4 082.3 MiB of a 4 096 MiB extent**, `h=0.5901`, `evictions=4 279`, `arena releases=0`; `MFN -lme 8` -> 144 / 14 688 / 8 153.0 MiB |
| **P3a-3** (DONE 2026-11-11, PM) | the loader places the expert tensors as **holes in the read-only model MAPPING** (`ROCm_HostMap` when the arena tier is on -- `host_master_mapped` in `llama-model.cpp` does the placement), and the arena arm now FORCES a mapped load (an iGPU reports no `mmap_support`, so AUTO resolved to `none` and the master was a pageable 18.6 GiB COPY) | `MEASUREMENTS.md` §16.1 (3): the `ROCm_HostMap 18 662.00 MiB` anonymous copy is gone, `MemAvailable` after the load 93 -> 106 GiB on `M35` |

> **THE DRAFT-SIDE INVARIANT (maintainer, 2026-11-11): the draft weights, embedded or a separate file, are never
> evicted** (`MEASUREMENTS.md` §19).  The embedded head holds one slot per expert, so the policy itself can never
> evict it; the three paths that take a table down outside the policy -- the pressure stand-down, the partial
> shrink and the post-drop re-arm -- now skip it explicitly, and the census asserts the result
> (`MTP/nextn head residency: N table(s), S/E slots resident, X eviction(s)`, ERROR if `X != 0` or `S != E`).
> A separate drafter's tables are a whole-set device allocation, not cache tables, so nothing can evict them.
>
> **TWO MORE, found while testing the SEPARATE MTP drafter (2026-11-11, PM, `MEASUREMENTS.md` §18) and both
> CLOSED**: (i) `ggml_backend_cuda_gtt_store_buffer_type_alloc_buffer` returning null **fails the model load** --
> its comment claimed the loader fell back to the device buffer, and it does not (`unable to allocate
> ROCmGTTStore0 buffer`, exit 1, on `-lme 1` + a separate drafter); the loader now sizes its placement against the
> store's real free bytes (`ggml_backend_dev_gtt_store_free_bytes`).  (ii) The store buffer's `free_buffer` released
> **nothing**, so a second model in one process found the region exhausted and silently ran on device buffers --
> `llama-bench -m A -m B` measured its two rows under different expert tiers.  And with them, the rule for a second
> load: it never sizes or grows the store, never takes the arena tier (that would carve the head out of the trunk's
> budget), and adopts the live region only up to what is free.
>
> **OPEN BUGS, both CLOSED (2026-11-11, PM).  The `tables=0` census was three seams, not one** -- full numbers in
> `MEASUREMENTS.md` §16.1:
>
> 1. `moe_cache_preflight()` returned **above** the per-device host-expert byte recording, so the completeness
>    gate saw `expected=0` (complete on the FIRST table), and both transport queries answered "no opinion"
>    (`moe_cache_pending` false, `moe_cache_stage_ready` true).  Record, then return.
> 2. `alloc_all_locked()` **clamped the tier's budget** against free VRAM / the slab's arena capacity, which on an
>    APU is 0 -> `layer 0 cannot hold a uniform arena above 0 slots (requested 0)`.  The store's extent already IS
>    the bound; the clamp is skipped for the tier.
> 3. The load was **not MAPPED** (`mmap_support` is false for an IGPU device), so the "holes" were a pageable full
>    copy of the expert set.  The arena arm forces the mapped load now (one WARN).
>
> Plus: the tier no longer arms the movable-boundary slab (it armed it through the preflight wrapper, and on a
> shared pool the slab sizes itself as the whole machine).
>
> **The unbounded allocation did NOT reproduce on the fixed build**, including the literal case-2 configuration
> (`-lme 4`, `-lv 4`, default ctx, `MemoryMax=48G`): exit 0, `SwapFree` unchanged for the whole run, no cgroup
> kill (`MEASUREMENTS.md` §16.4).  The `--fit` hypothesis is **not confirmed** (`--fit on` is clean too).  Barrier
> 1 (`ulimit -v 96 GiB`) did fire, on ONE arm and on an allocation the tier has nothing to do with: Flash-Next's
> 27 465 MiB resident `per_layer_token_embd.weight` CPU buffer, on top of a 41 GiB file mapping.  The store's
> extent cannot be exceeded (`ggml_cuda_gtt_store_reserve` refuses), so an explosion is never the arena; if it
> recurs, collect `/proc/<pid>/smaps_rollup` per mapping first.  `MOE_EXPERT_GTT_ARENA` stays **default-OFF**
> (it is a fittability tier, not a speed tier, and it costs decode on a config whose KV/experts fit).

| **P3b** (DONE 2026-11-11, PM) | `-lme` smaller than the set: evictions, fills, the hit-rate census, the transient band only if the measured admission policy wants it (it does not: every miss fills by construction, so touch admission stays a no-op here) | `MEASUREMENTS.md` §16.3/§16.5: **a >RAM model runs** -- 123.45 GiB on this 123 GiB box at `-lme 60` (3.5 t/s) and `-lme 20` (3.0 t/s), `evictions > 0`, `arena releases == 0`, and the text is identical across a 3x residency change and to the store-off arm on both fitting models |
| **P3c** (DONE 2026-11-11, PM) | the MTP/nextn head's tables pinned outside `-lme`: partitioned by layer (`>= n_layer_trunk`) and only when `load_mtp` (otherwise the head's tensors are not part of the set at all), the head's bytes added ON TOP of the budget, its tables placed first in placement mode, and at full slot count in arena mode | `MEASUREMENTS.md` §17: 35B-A3B `-lme 4 --spec-type draft-mtp` -> extent **4 560.0** = 4 096 + 464, **123 tables / 7 488 slots** (the head's 3 tables at 256/256 on top of the trunk's 56), trunk slots unchanged, `arena releases=0`, `draft acceptance=1.00000`; the 122B `-lme 60 --spec-type draft-mtp` -> 63 888.0 = 61 440 + 2 448, 147 tables / 19 920 slots.  Same-seed text unchanged in all 10 35B arms and all 4 122B arms (with and without MTP).  The **separate** drafter shape (Flash-Next's `mtp-...-shared-*.gguf`) is `MEASUREMENTS.md` §18: one store, two models -- the second load sizes against the live store's free bytes, never takes the arena, adopts the region when its set fits, and declines cleanly otherwise (**fixing a fatal null-carve path and a never-released carve**); text `f586785d7183` in every arm |

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
* Census: `evictions`, `reuses`, `unregisters` (must be 0 on a steady-state arm), `GTT store MiB` vs budget.

**Accept:** a config that **evicts** still produces byte-identical text to the same config with the GTT store
large enough that nothing evicts (the eviction path changes *where* bytes are, never *which*); census shows
evictions > 0 and unregisters == 0.

## P4 — the two load arms

> **P4a DONE 2026-11-11 (after the r262 rebase)** -- the transport half needed NO new design: r262 shipped the
> `map -> file -> dio` ladder (stall detector + probe-fill A/B) and this stage only had to put the tier back **on**
> it (`MEASUREMENTS.md` §20).  Two bugs had to be fixed for that to be real: the ladder was initialised only at a
> POOL build (the tier has no pool, so a pin and `--load-mode buffered|dio` were silently ignored), and the
> per-table file source was registered only when the pool was on (so the `file`/`dio` arms could be *selected* and
> then fall back to the mapping memcpy).  Measured: the bufferered arm is **+45 %** on a cold page cache and
> **+24 %** even warm, `dio` is 11x slower (hence the 3x guard), `auto` latches `file` on the probe, and the text is
> identical in all three arms, in all four `--load-mode` values, and in an arm where **every fill fails**.
>
> **P4b DONE 2026-11-11** -- the destination side: on the `file` (buffered) arm a fill reads **straight into the
> arena slot**, no bounce entry and no H2D (`MOE_EXPERT_GTT_FILL_DIRECT=0` is the A/B and the kill switch;
> `dio` keeps the bounce path because `O_DIRECT` needs the 4096 round-out plus a tail splice, `map` because a host
> `memcpy` would need the ordering drain with no transfer to hide behind; a capturing stream declines).

| **P4b** (DONE 2026-11-11, PM) | the buffered arm reads into the slot: one `pread`, no bounce, no H2D; the host write drained behind the slot's previous consumer (counted: 0.005 ms/fill); the sizing fills get the same read with no drain at all | `MEASUREMENTS.md` §21.2: **+15 % decode warm** (28.2 vs 24.5 t/s, 3 reps each at `-n 96`) and **+12 % cold** (21.3/21.6 vs 19.1/19.1), text `627b2ebd7792` in every arm including the fill-fault one |
| **P4e** (TODO) | the overlap experiment: reproduce "does the fill ever have work to hide behind" before any worker (r261 measured ±1.5 % for the exact within-layer lookahead) | unaffected by P4a/P4b; still the last P4 item |

> **OPEN QUESTION for the maintainer (§21.3):** with the direct fill, `file` beats `map` **warm** as well
> (28.2 against 25.1-25.3 t/s), but r262's ladder starts optimistically on `map` and only falls on a probe or a
> stall -- and on this tier the probe compares transports only (both samples land in the bounce entry), so a warm
> box rarely latches `file`.  Two candidate fixes, both policy changes to an r262 decision, neither implemented: a
> tier-specific start arm when the direct fill is available, or a direct-aware probe (the sample served into the
> slot, so the comparison is the end-to-end one).

| **P4e** (DONE 2026-11-11, PM) | the overlap experiment: is there GPU work in flight when a demand fill is issued (i.e. would a fill worker pay)? | `MEASUREMENTS.md` §22: **NO** -- the drain in front of a direct fill is the in-flight work by construction, and it is **<= 263 us, never >= 1 ms**, across 22-36 k fills; the arm that *overlaps* the H2D is slower than the one that does not (§21.2); and +49 % decode from `-lme 4` to `8` proves the fill is on the critical path, not hidden. **Do not build a fill worker**; `IMPLEMENTATION.md` §7 arm 9 asserts this answer.  Prefill is P5's question |

> **P4 is now COMPLETE.**  Its answer, in one line: the fills are on the critical path, so make them
> fewer/cheaper (residency, the direct fill, coalescing) -- never a pipeline.
>
> **P4f (the start arm, DONE 2026-11-11, maintainer's Option 1)** -- and it is the campaign's biggest win: on
> **RDNA3_5 with the GTT arena tier and the direct fill available**, the transport ladder now **starts on the
> buffered arm** instead of r262's optimistic `map`.  Three gates (`g_gtt_arena`, `GGML_CUDA_CC_IS_RDNA3_5`, the
> direct fill being available) keep it off every other configuration, and an explicit
> `MOE_HOST_POOL_TRANSPORT`/`--load-mode` still wins.  Measured (`MEASUREMENTS.md` §24): **+13.7 % warm** on a
> fitting model (29.0 against 25.5 t/s, deterministic -- before it the ladder latched `file` 0 of 5 times), and
> **+56-67 %** on a >RAM model (MBIG 7.5/7.8 against 4.5/5.0 t/s), where the set can never be cached.  The cost is
> the **first** run of a *fitting* model on a cold cache (-11.5 %), accepted on the maintainer's steer: *"the cold
> condition only happens once at the start; better to tune for warm"*.

| **P4f** (DONE 2026-11-11, PM) | the RDNA3_5 UMA start arm: `file` when the direct fill is available | `MEASUREMENTS.md` §24.2/§24.3: +13.7 % warm, +56-67 % on the >RAM model, text identical in every arm, every gate verified (env pin, `--load-mode`, direct-fill switch, placement mode) |

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
  question here** -- does this path ever have work the fill can hide behind, given the GTT store's fill is 3.2x the
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

## P5 — prefill: stream only what is not resident  (DONE 2026-11-11, `MEASUREMENTS.md` §25)

> **DONE, and it is the second-biggest win of the campaign: +63-67 % prefill on a >RAM model, neutral on a fitting
> one, with the prefill logits unmoved (mean KLD -0.000000, same-top-p 99.990 %).**  One new backend hook
> (`moe_cache_prefill_stream`) fills the scheduler's split-input copy where the delivery's used-expert copy would
> run: a **D2D** for every used expert the store holds, ONE **coalesced span read + ONE H2D** per maximal run of
> the rest (a run is contiguous in the file AND in the destination), the reads **in parallel** (8 readers) and
> overlapping the previous batch's H2Ds, and the delivery's per-expert source for anything a span could not serve.
> The used set is the scheduler's own **bitset**, never a re-read of the routing.
>
> Two plan assumptions did not survive contact with the code, and both are recorded rather than papered over:
> (a) **the destination buffer cannot carry residency between chunks** -- the graph allocator gives a prefill graph
> ONE `input_cpy` per copy id for every table, so nothing survives a pass (113 owner changes over 117 calls), and
> the "do not re-fetch a bank the previous chunk holds" clause is served by the STORE's residents instead; (b)
> **the coalescing is a call-count win, not a byte win** -- `stage_span_fill`'s per-expert H2D out of the pinned
> chunk was the dominant cost (12 031 H2Ds for 7 293 MiB) and one H2D per span is the same bytes in 4 605 calls.
> `stage_span_fill` itself is NOT reused (its pool admission, its `MOE_STAGE_SPAN_MIN` skip and its per-expert
> copy are all wrong for this tier); its SHAPE is, and the discrete pool's body is untouched.
> Gate: `scripts/gate-uma-experts.sh` arms `prefill` / `pfoff` / `pfverify` / `fault`.


* Per layer: compute the chunk's used-expert set (the existing per-layer access plan already does this), subtract
  the resident set, and stream the difference coalesced — this is exactly
  `moe_cache_stage_table()`'s span-read shape with the GTT store as the destination.
* Keep the `MOE_HOST_POOL_STAGE_SPAN`-class coalescing (an expert's rows are contiguous; a layer span is
  sequential) — it is the same reason W4 measured +38 % on the discrete pool.
* Do not re-fetch a bank the previous chunk already holds; the census's per-layer read bytes must fall on a
  repeated-prompt arm.
* Ordering: a prefill fill is a bulk fill, so the warm arm is the copy engine and the cold arm is a batched
  `pread` (parallel, not one `pread` per expert) — measure both before choosing.

**Accept:** prefill logits identical to the all-resident arm (`scripts/gate-prefill-logits.sh` — this *is* a
prefill-path change); text identical; read bytes strictly below "fetch every used expert" on the
repeated-prompt arm; pp rate reported against the all-resident arm.

## P6 — gate, docs, promotion  (gate + docs DONE 2026-11-11; the PROMOTION waits on the maintainer)

> **`scripts/gate-uma-experts.sh` is written and green: 16 arms, 35 checks** (`WORKLOG.md` 261111.026).  Every arm
> freezes the heat-map prior, hashes the text from a `llama-cli -v` run with stdout/stderr split (one run per arm),
> and the prefill arm asserts BOTH P5's read-byte fall and P4e's "no fill worker" answer from the drain census.
> Two assertions had to be corrected against what this tier actually reports (`MOE_EXPERT_GTT=0` is "cache on,
> registers nothing", not "cache OFF"; `takeover=` counts the decode take-overs here, not `gate-host-pool.sh`'s
> identity hand-off) -- copying either over from the discrete gate would have failed a correct arm.
> **Both decisions are TAKEN (2026-11-11):** the SHAPE follows `-lme`'s promise -- a budget below the set takes the
> arena (the excess streams from the reclaimable page cache, which is what "cap the NON-SWAPPABLE memory" means),
> a budget that covers the set takes placement -- with `MOE_EXPERT_GTT_ARENA` as a tri-state override (`=1` force
> arena, `=0` force placement, unset = auto), implemented and asserted by the gate's `shapefit` / `ram` arms; and
> the transient band stays DROPPED, because the campaign's own policy simulator predicts the Nth-touch admission is
> *worse* on this tier at every slot count (0.708 against LFRU's 0.727 at S=56, 0.880 against 0.918 at S=133) and
> the tier already implements the predicted-best policy (`MEASUREMENTS.md` §26).
>
> **The discrete side is measured too** (`MEASUREMENTS.md` §27, 2026-11-11): on `soar` (3x R9700 gfx1201) the tier is
> inert with its envs forced ON, `-lme` still means the host-expert budget, `scripts/gate-host-pool.sh` is **42/42**
> with the patched build, and the pool path's throughput is within noise.  §9's "must be provably inert there" is
> now a measurement.
>
> **What is left is the promotion itself.**  The docs are staged: the env table is in `README.md` with
> each variable's class, the measured facts are in §6/`MEASUREMENTS.md`, and the `ENVIRONMENT.md` /
> `INVARIANTS.md` / `patches/README.md` texts are written out in the newest `WORKLOG.md` entry.


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
| 1 | **text identity** — GTT store on vs `MOE_EXPERT_GTT=0`, on a config that **evicts** | same-seed greedy text **byte-identical** |
| 2 | **all-resident** — GTT store big enough that nothing evicts | identical text to the plain-device arm; read rate within noise of a `hipMalloc` arena |
| 3 | **>RAM** — an expert set larger than RAM | completes; census shows fills (cold and warm) and evictions; no `HSA_STATUS_ERROR_MEMORY_FAULT`; `check-gpu-faults.sh` clean |
| 4 | **cold page cache** — the same run after dropping the file's cache | the cold arm is really exercised (its counters move); text identical to arm 1 |
| 5 | **budget** — explicit `MOE_EXPERT_GTT_MIB=N` | the GTT store's extent ≤ N; `MemAvailable` after load above the OS reserve; the census names the source |
| 6 | **no-movement** — steady-state arm | `unregisters == 0` and no `madvise` on the hot path; evictions > 0 with text identical |
| 6b | **admission** — `MOE_EXPERT_CACHE_TOUCH=1/2/3`, same prompt set | a higher threshold never changes the text; the census shows fewer promotions, a smaller footprint and more transient fills; `2` must be no slower than `1` on a cold-start session (that is the default's justification) |
| 6c | **`-lme`** — `-lme 6`, `-lme 12`, `-lme auto`, same model/prompt | the GTT store's extent matches the request (+-1 slot); the census's budget source names `-lme N` vs `auto`; `auto` in the fitting band reports zero evictions and in the not-fitting band non-zero; text identical across all four |
| 7 | **kill switch** — `MOE_EXPERT_GTT=0` | reproduces today's behaviour (cache disabled on IGPU) and its exact log |
| 8 | **fill-fault** — `MOE_EXPERT_GTT_FILL_FAULT=1` | declines cleanly; correct text; never a GPU read of an unfilled slot |
| 9 | **prefill** — repeated prompt with a resident bank | per-layer read bytes fall; prefill logits identical (from `gate-prefill-logits.sh`) |
| 10 | **host-buffer A/B** — `GGML_FORCE_NO_INTEGRATED=1` | throughput unchanged within noise, text identical: the GTT store does not depend on the zero-copy UMA weights path |

Nothing in this gate runs on a discrete box; the delivery's existing gates are the discrete-side proof (the
GTT store must be unreachable there).

## §8 Effort and sequencing (estimate, not a commitment)

| phase | scope | risk |
|---|---|---|
| P1.0 | four probes, no code — **DONE 2026-11-11**, all four hold; probe 15 (touch distance) still owed | none |
| P1 | preflight branch + budget + kill switch | low |
| P2 | GTT store allocation + buft + census | medium: a new buffer type the scheduler must accept |
| P3 | residency/eviction wiring | low: the policy already exists |
| P4 | the two fill arms + ordering | medium: the cold arm's visibility must be proven, not assumed |
| P5 | prefill streaming | medium: prefill is where the read volume lives |
| P6 | gate + docs | low |

## §9 Rollback

`MOE_EXPERT_GTT=0` restores today's behaviour without a rebuild (the cache disabled on an integrated GPU, the
host-expert tier refused).  If a phase cannot be made switch-safe it does not ship — that is the lesson of the
guard this campaign replaces.  The GTT store must never be the only path to a working configuration on a discrete
box, and it must be provably inert there (`GGML_FORCE_NO_INTEGRATED`, arm 10).
