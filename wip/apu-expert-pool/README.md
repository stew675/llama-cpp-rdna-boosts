# wip/apu-expert-pool — opening the host-expert / expert-arena tier on UMA APUs (gfx1150/1151 and newer)

**Status: OPEN — SCOPING NOTE + a completed measurement foundation (2026-11-09).  Nothing implemented,
nothing in `patches/`, nothing applied to the fork.**  The plan is [`IMPLEMENTATION.md`](IMPLEMENTATION.md);
the probe log and the exact numbers are [`MEASUREMENTS.md`](MEASUREMENTS.md); the probe sources are
[`probe/`](probe/README.md) so every claim here can be re-run on any UMA box.

Cut against the **r243 tree** (`release.json` `tree` = `e9300f1`, tip `72f7753f`).  r243 changes only
`src/models/qwen4exp.cpp`, so every file this campaign cites is identical to r242's.

## The charter, in the maintainer's words (2026-11-09, `halo` = Ryzen AI MAX+ 395 / gfx1151)

> at present all attempts to create a host-expert pool on the GFX1151 are presently hard-gated off, so we're
> essentially opening up that path for UMA architectures (gfx1150/1151 and newer).

And the goal, stated against a working discrete configuration:

> single R9700, 32 GB VRAM + 32 GB host RAM: I can load and run models that don't fit, via the host reading
> the weights through the page cache (from disk), and it works really well … the hottest expert weights are on
> the GPU, the warm ones in RAM, and the cold ones on disk which are only pulled in rarely.

The APU version of that: keep the existing arena/residency-table machinery, but on UMA **an eviction stops
being a data movement** — the arena is host-writable and device-readable, so a "hot" expert is a table entry
over bytes, and a "cold" one is a `pread`.

## 1. What is gated today (three places, all deliberate)

| # | where | what it does on an integrated GPU |
|---|---|---|
| 1 | `src/llama-model.cpp` (~L1888, the `IGPU` device scan) → `llama_model_loader::set_host_experts_umem(true)` (`src/llama-model-loader.cpp:130`) | sets `host_experts_umem`, which **zeroes `n_cpu_moe_from_end`** and makes `host_expert_selected()` (`:149`) return false, so `-ncmoe`/`-cmoe`/`-lme`/a CPU `--override-tensor *_exps.*` are **ignored**: every expert stays on its layer's device. |
| 2 | `moe_cache_shared_pool_device()` + `moe_cache_preflight()` (`ggml/src/ggml-cuda/moe-expert-cache.cu:5604`, `:5622`) | turns the **whole expert-cache stack** off (`g_enabled = false`): arena, pool, heat map, cache-band fusions.  Escape hatch only: `MOE_EXPERT_CACHE_UNIFIED=1` **plus** an explicit `MOE_EXPERT_CACHE_MIB` **plus** `GGML_CUDA_SLAB_RESERVE_MIB`. |
| 3 | `common/fit.cpp` (~L700, `cache_reserve.arena_refused`) | withholds no slab headroom and no arena in the `--fit` margin, matching #2. |

The `--host-experts mmap` spelling is separately refused everywhere (`common/arg.cpp:2697`, issue #116) because
a **pageable host master cannot be read by a no-XNACK GPU** — that refusal is *correct and stays* (see §6).

## 2. Why the gate was put there (both reasons are real, and neither is a verdict on UMA)

1. **Over-commit by AUTO sizing.** The arena's slab sizes itself as `free VRAM − reserve`.  On a shared pool
   "free VRAM" is *host RAM*, so AUTO maps the machine: measured 2026-10-19 (35B-A3B Q4_K_M, `-ncmoe 40`), a
   123 GiB box went to **121 GiB resident + 13 GiB swap** before the load finished.  Also measured:
   `SLAB_INIT … size(VA)=98.44 GiB mapped=98.19 GiB` on the same box.  This is a **sizing** defect and the
   guard against it must stay — it is not an argument that the tier cannot exist.
2. **A real fault.** On gfx1151, `-ncmoe 24`/`48` on Qwen3.8-Flash-Next faulted in
   `mul_mat_q<Q2_0,16>` **at a host address**: the GPU was reading a host master in place and there is no
   XNACK on this part.  **Which mechanism left that page unpresent is not established** — the campaign must
   not inherit the assumption that it was unavoidable.  Three candidates, and P1.0 in
   [`IMPLEMENTATION.md`](IMPLEMENTATION.md) separates them before any code changes:
   (a) the host-expert buft registers its backing `Portable|ReadOnly` with **no `Mapped`**
   (`ggml/src/ggml-cuda/ggml-cuda.cu:9178`) — r215 §4's documented failure mode on a discrete box;
   (b) the load-time placement handing the GPU a **pageable** pointer (the mapping) instead of a pinned one;
   (c) genuine eviction/unpinning under pressure.  Evidence *against* a blanket "registration needs `Mapped`
   here": the KV pool mirror registers with **flags 0** (`ggml-cuda.cu:4734`) and is read by kernels on this
   same box (`INVARIANTS.md`, block 15), because with `iommu=off` the host VA *is* the device VA.

**The campaign's thesis: keep guard #2's *sizing* rule, replace the blanket refusal with a UMA policy.**

## 3. The measured foundation (this changes the conclusion for UMA)

Full log in [`MEASUREMENTS.md`](MEASUREMENTS.md).  All on `halo` (gfx1151, 40 CU, 123 GiB, kernel 7.2.13,
ROCm 10.1.0):

| fact | measurement |
|---|---|
| gfx1151 has **no XNACK** — not "off", absent from the ISA | `hipcc --offload-arch=gfx1151:xnack+` → `invalid target ID`; `gfx90a:xnack+`, `gfx942:xnack+` accepted |
| no fault-and-retry, at any setting | unregistered page-cache mapping, **fully hot** → `HSA_STATUS_ERROR_MEMORY_FAULT` … `Page not present or supervisor privilege`, core dump; `HSA_XNACK=1` changes nothing; `PageableMemoryAccess = 0` |
| "VRAM" is the same DRAM as host RAM | `mem_info_vram_total` **2 GiB** carve-out, `mem_info_gtt_total` **130 GB**, HIP `totalGlobalMem` **121.1 GiB**, rocminfo `Memory Properties: APU` |
| **the host can read and write the GTT arena** | host read/write of a `hipMalloc` pointer: OK; host `memset` into it **13.5 GB/s**; GPU reads it **FRESH** |
| fills through the existing transport are already good | `hipMemcpy` page cache → GTT arena **31.5 GB/s** cold / **77.1 GB/s** warm; host `memcpy` **12.8 GB/s** |
| GPU read bandwidth is not tiered on an APU | `hipMalloc` **232.6** GB/s; registered-RO file/page-cache mapping **230.5**; registered-RO anon **230.8**; registered **writable** **60.2** |
| registration pins, so "mapped but absent" does not exist | 4 GiB: control (unregistered) reclaims **4096/4096** MiB on `fadvise`+`madvise`; registered holds **4096/4096**, and still holds after `hipHostUnregister` while the VMA lives; a 46 GiB shard registers in **8.69 s** and holds 11.9M pages |
| a filled buffer can be registered read-only *afterwards* | `hipHostRegister(Mapped\|ReadOnly)` on a `PROT_WRITE` buffer: OK → GPU reads at **233 GB/s** (vs 60 writable) |
| refills need no register cycle and no flush | host overwrite in place → GPU reads **FRESH** (4 alternations, both registration modes); register/unregister/re-register of 1 GiB = 2.67 / 0.00 / 0.00 ms |
| `O_DIRECT` cannot target the GTT arena | `pread(O_DIRECT)` into a `hipMalloc` buffer → `EINVAL`; into an anon pool → **3.6 GB/s**.  Disk fills keep the staging buffer. |
| dirty pages are a *policy* question, not an I/O cost | anon pool is `Private_Dirty` 4096/4096 MiB, but `unregister`+`MADV_DONTNEED` releases 4 GiB in **0.26 ms** with `pswpout=0`/`pgpgout=0`; a `PROT_READ` file mapping is `Private_Dirty` **0** and never writeback |
| **trap**: discarding a *registered* anon region the GPU is reading is a **silent zero** | GPU sum `0333333330000000` → `0000000000000000`, no fault, no error — so never use `MADV_DONTNEED` as an eviction primitive on a GPU-visible region |

Consequences for the design:

* The arena **does not need to move its data on eviction**, and on UMA the "host pool → VRAM" copy has no
  tier to cross: the pool and the arena are one DRAM.  So the pool tier's *only* remaining job is the one the
  mapping-master mode already gives it (be the pinned, bounded, always-present source of a GPU read).
* The two tiers that remain physically distinct are **resident** and **not resident (page cache / disk)**.
  The heat/residency table is what decides residency; nothing decides *location*.
* Therefore the cheapest correct thing to build first is **not** the arena — it is enabling the *existing*
  pool/mapping tier on UMA (P1), because that is what makes a >RAM model run at all, and it is already
  written, gated and gated-off.

## 4. Two candidate shapes, sequenced (details and accept criteria in `IMPLEMENTATION.md`)

* **P1 — UMA pool tier (smallest diff, unblocks the goal).** Allow the `pool`/`auto` host-expert mode on
  `IGPU`, keep `pinned` refused until P1.0 explains the fault.  The mechanism already exists and is shipped
  for discrete: `ROCm_HostMap` (`ggml-cuda.cu:4970`, "wip/host-expert-master-drop", `TODO` #55) is a host
  buft that carries its device, whose tensors are placed **directly in the read-only model mapping**, with
  every GPU read served through the bounded pinned pool — written to make "a model whose host-expert set
  does not fit in RAM runnable".  On UMA only the umem gate stops it.  Sizing from a RAM budget, never from
  "free VRAM".  Proves the >RAM case end to end with today's machinery.
* **P2 — UMA arena as the disk-backed cache (the maintainer's shape).** `MOE_EXPERT_CACHE_UNIFIED` becomes a
  real UMA mode: slab/arena sized from an explicit RAM budget; fills written straight into the slot (the GTT
  arena is host-writable, so no pool→arena copy and no second allocation); **eviction = invalidate the table
  entry** (the `access_locked` arena→pool hand-over is skipped on UMA); soft-evict (keep bytes, drop the
  entry) as a free second-chance policy that a discrete card cannot have.

P1 first, then measure P2 against it at equal RAM.  If P1's in-place pool path is within a few percent of P2,
**stop at P1** — two tiers need two budgets, and this campaign exists to make one box behave like one pool.

## 5. The one non-obvious fix that applies to both

The shipped pool registers its backing with `cudaHostRegisterPortable | cudaHostRegisterMapped`
(`moe-expert-cache.cu:2017`, mirroring `cudaHostAlloc(Portable|Mapped)` at `:2250`) — **writable**, and on this
box a writable registration is the **60 GB/s** read arm against **233 GB/s** for a read-only one.  On a
discrete card the PCIe link (28.6-55.8 GB/s measured in the delivery's own records) hides the difference; on
a UMA APU it is the whole difference between "the pool is RAM at DRAM speed" and "the pool is a slow tier".
Adding `cudaHostRegisterReadOnly` to the device alias (the host still writes through the original mapping —
measured FRESH) is a one-line-class change that must be A/B'd on **both** boxes.

## 6. What must NOT be done (each is a measured trap)

* **Do not use registration as the residency tier.**  Registering pins (measured above), so "the arena is the
  whole file" means "the whole file is `mlock`ed" — 46 GiB/8.69 s here.  A *window* is the only expression of
  this that is not a RAM regression.
* **Do not `MADV_DONTNEED` anything the GPU may read.**  Measured silent zero-fill with no fault.  Eviction is
  a table operation; discarding is `hipHostUnregister` **then** `madvise`.
* **Do not let AUTO size the arena from "free VRAM"** — that is the 2026-10-19 over-commit, and it stays
  guarded.  A UMA budget is `MemAvailable` minus KV, compute and an OS reserve, taken once, and explicit
  `MOE_EXPERT_CACHE_MIB` always wins.
* **Do not touch the `integrated` host-buffer (zero-copy UMA weights) path.**  It is already enabled on HIP
  and it has its own history (`ggml-cuda.cu:646`: PPL 5.9243 → 8.51+ without `HIP_LAUNCH_BLOCKING`, root-caused
  to the scheduler reading a host-resident *graph input* in place, fixed by the host-input guard in
  `ggml_backend_sched_buffer_supported()`).  The arena is *device* memory; this campaign must not need that
  path, and `GGML_FORCE_NO_INTEGRATED=1` is the A/B that proves it.
* **Do not probe UMA by host-dereferencing a device pointer.**  On a discrete GPU that is UB/SIGSEGV; gate on
  `prop.integrated` (the same value `ggml_backend_cuda_device_get_type()` and the fit reserve use).
* **Do not make a `pageable` mapping the thing the GPU reads.**  `--host-experts mmap` stays refused
  (issue #116).  A pageable mapping is a **fill source** (host `memcpy`/`pread`), never a read target.

## 7. Open questions for the maintainer

1. **P1 or P2 first?** This note recommends P1 (cheap, unblocks >RAM) then P2 measured against it.
2. **Does the pool tier survive on UMA at all** once the arena can be host-filled, or should `-lme` become
   "the arena budget" and the pinned pool disappear on UMA?  (One budget is much easier to reason about; the
   counter-argument is that the pool is the only *bounded, never-swapped* structure today.)
3. **What is the RAM reserve policy** on a shared pool — `MOE_HOST_POOL_RAM_PCT` (85) / `-lme auto` (80 %)
   were derived on a discrete box where host RAM had no VRAM competitor.  This campaign would unify them.
4. **Windows APUs with a fixed VRAM carve-out** are the case `MOE_EXPERT_CACHE_UNIFIED` was written for.  Keep
   that arm exactly as is, or fold UMA and carve-out into one detection (Linux gfx1151 is *not* a carve-out:
   2 GiB visible / 130 GB GTT).
5. Should the campaign also take the **`gfx1150`** target (Strix Point, smaller pool)?  Nothing measured here.

## 8. Promotion criteria (what turns this into a block)

* A model whose expert set **exceeds RAM** runs on `halo` with `-ncmoe`/`-cmoe` + the tier, at
  same-seed byte-identical text against the best fitting arm, and the census shows real fills/evictions.
* `scripts/gate-uma-arena.sh` (spec in `IMPLEMENTATION.md` §7) green, including the cold-page-cache arm and a
  `check-gpu-faults.sh` clean run.
* No regression on the discrete boxes at equal config (the pool registration change must be neutral or better).
* Every new variable documented in `ENVIRONMENT.md` with its class, beneficial features **on by default** per
  the default-on policy, and a kill switch.
* Dated `WORKLOG.md` entry; block note in `patches/README.md` (block 09/12 own this code); `TODO.md` item if
  any part is deliberately deferred.

## 9. File map

| file | what |
|---|---|
| `README.md` | this file: charter, the gate, the measured foundation, the two shapes, the traps |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | the phased plan: code touch points, env surface, gate script spec, docs, rollback, effort |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | the probe log: box config, every number, reproduction commands |
| [`probe/`](probe/README.md) | the ten probe programs behind those numbers (they change nothing and need no root) |
