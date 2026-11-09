# IMPLEMENTATION — `wip/apu-expert-pool`

Phased plan for opening the host-expert / expert-arena tier on UMA APUs.  Read
[`README.md`](README.md) first (§3 is the measurement foundation, §6 the traps).  Numbers cited here are from
[`MEASUREMENTS.md`](MEASUREMENTS.md); file:line references are the **r243** tree (`e9300f1`).

**Rule for the whole campaign:** every phase ships behind a switch whose default is decided by measurement, and
every phase's accept criteria include a byte-identical-text arm.  Nothing here is allowed to become a
"temporarily disabled" default like the guard it replaces.

---

## P0 — the facts (DONE, 2026-11-09)

Ten probes, no root, nothing changed in the fork.  See [`MEASUREMENTS.md`](MEASUREMENTS.md) and
[`probe/`](probe/README.md).  The three facts the plan depends on:

1. `prop.integrated` (gfx1151) memory is one DRAM pool: 2 GiB visible VRAM carve-out, 130 GB GTT, HIP
   `totalGlobalMem` 121.1 GiB.
2. A `hipMalloc` (GTT) buffer is **host-writable** (13.5 GB/s host stores, GPU sees it FRESH) and
   device-readable at 232.6 GB/s — so **a fill can target the arena slot directly and an eviction moves no
   data**.
3. A registered **writable** host range reads at 60.2 GB/s; registered **read-only** at 230.8 GB/s.  The
   shipped pool registers writable.

## P0.5 — the standalone win: read-only registration for the pool's device alias

Ships on its own, needs no UMA gating, and is measurable today.

* `ggml/src/ggml-cuda/moe-expert-cache.cu:2017` — the r215 `mmap` + `cudaHostRegister` backing (selected by
  `MOE_HOST_POOL_ALLOC=thp`, the default) currently passes
  `cudaHostRegisterPortable | cudaHostRegisterMapped`.  Add `cudaHostRegisterReadOnly` there, behind a new
  `MOE_HOST_POOL_RO` (default `1`, the kill switch `0` that restores today's writable registration).
* The legacy arms have no read-only flag: `MOE_HOST_POOL_ALLOC=portable|mapped|plain` (`:2239`, allocation at
  `:2250`/`:2258`/`:3125`) stay as the A/B, so the comparison is `thp`+RO vs `thp`+writable vs `portable`.
* **Why it is safe:** the flag describes *device* access.  The host still writes the pool through the original
  mapping, and measured, the GPU sees those writes with no flush (FRESH, both registration modes).
* **Measured expectation:** UMA 233 vs 60 GB/s on the device read of pool-resident experts.  On the R9700 the
  PCIe link (28.6-55.8 GB/s) should mask it — so the discrete arm is a *neutrality* check, not a win.
* Accept: `gate-host-pool.sh` green; same-seed text byte-identical; on `halo` a pool-resident MoE config
  improves materially; on a R9700 unchanged within noise.
* If the discrete arm regresses, keep the flag UMA-only (`prop.integrated`) — do not revert the find.

## P1 — enable the **pool** tier on UMA (what actually unblocks a >RAM model)

`--host-experts pool` resolves to `host_experts_mode = AUTO` (`common/arg.cpp:2693`) and selects the
`ROCm_HostMap` buft (`ggml/src/ggml-cuda/ggml-cuda.cu:4970`, `wip/host-expert-master-drop`, `TODO` #55): the
expert tensors are placed **directly in the read-only model mapping** and every GPU read of them is served
through the bounded pinned pool.  That is exactly the tier this campaign wants, it is already shipped and gate-
covered **for discrete**, and on UMA only gate #1 stops it.

### P1.0 — find the root cause of the gfx1151 fault first (no code)

The gate cites `-ncmoe 24`/`48` on Qwen3.8-Flash-Next faulting in `mul_mat_q<Q2_0,16>` at a host address.
Reproduce it, then separate the three candidates:

| hypothesis | how to test | if true |
|---|---|---|
| (a) registration flags: the host-expert buft registers `cudaHostRegisterPortable\|cudaHostRegisterReadOnly`, **no `Mapped`** (`ggml-cuda.cu:9178`) | add `Mapped` behind an env and re-run the faulting arm | a one-line fix; `pinned` becomes usable too |
| (b) a **pageable** pointer reached the GPU (the mapping, not the pinned master) — `pool` placement taken while the loader believed `pinned` | `MOE_HOST_POOL_FILL_FAULT=1`-style decline arms + the address in the fault message vs `/proc/<pid>/maps` (attribute by address-to-buffer, never by kernel name — r214 §2) | a placement bug, not an architecture limit |
| (c) a genuine unpin/eviction | run with the pool pinned and large enough that nothing is reclaimed; `mincore()` the master before the faulting launch | a residency defect (then `INVARIANTS.md` gains an entry) |

Do **not** skip this step: the whole campaign's risk assessment depends on whether the tier "cannot work on UMA"
or "has a fixable defect that was read as 'cannot work'".  Record the answer in this file before P1.1.

### P1.1 — make the loader's refusal a policy, not a blanket

* `src/llama-model.cpp` (~L1888): the `IGPU` scan calls `ml.set_host_experts_umem(true)` unconditionally before
  `plan_host_expert_placement()`.  Pass the resolved mode instead:
  `ml.set_host_experts_umem(true, params.host_experts_mode)`.
* `src/llama-model-loader.cpp:130 set_host_experts_umem()`: keep the function as the single place that decides,
  and split the two cases:
  * `LLAMA_HOST_EXPERTS_MODE_PINNED` → **keep refusing**.  Two reasons: a whole-set pin is 2x the RAM on a
    pooled box, and until P1.0 explains it, the arm that faulted on gfx1151 is the pinned master read in
    place.  Keep the existing message, add the remedy: "use `--host-experts pool` (or `-lme auto`)".
  * `AUTO`/`POOL` → **allow**: do not zero `n_cpu_moe_from_end`, and let `host_expert_selected()`
    (`:149`) answer normally.
* `src/llama-model-loader.cpp` (~L1969, the "keep it on the device" fallback) must test the same policy.
* **On an `IGPU`, `AUTO` resolves to the pool unconditionally** — it must *not* fall back to "pin the whole
  set when it fits", which is `AUTO`'s discrete behaviour and the one arm that has faulted here.  A whole-set
  pin stays reachable only through an explicit `--host-experts pinned`, and only once P1.0 clears it.
* The default on an IGPU therefore becomes the pool tier with no flags — this is the
  default-on policy, and it is what makes the campaign's goal reachable without a flag.  Guard it with one
  kill switch, `MOE_HOST_EXPERTS_UMA` (default `1`; `0` restores today's refusal verbatim), so a bisect can put
  the box back exactly as it is now.

### P1.2 — sizing on a shared pool

* `MOE_HOST_LIMIT_FRAC` (80 %) and `MOE_HOST_POOL_RAM_PCT` (85 %) were derived where host RAM had no VRAM
  competitor.  On UMA the pool is the *only* non-swappable tier and the page cache is the rest, so the figure
  that matters is `MemAvailable − KV − compute − OS reserve`.  For P1, keep the existing fractions (they are
  already `MemAvailable`-based) and **assert** in the census that `pool + KV + compute < MemTotal`; do not
  invent a second budget until P3.
* The known defect to fix in the same change: `MemAvailable` is not cgroup-aware in either call site
  (`src/llama-model-loader.cpp`, `moe-expert-cache.cu`) — see `wip/host-expert-coverage/`.  On UMA this matters
  more (the pool is the whole tier), so P1 depends on that campaign's verdict; if it is not ready, stay with
  `MemAvailable` and document it.

### P1.3 — verify the two things that faulted

* **The `pinned`-master in-place read** (gate #1's fault): must be unreachable in pool mode — assert in the
  gate that no `mul_mat_q<…>` fault appears in the log and that `check-gpu-faults.sh` is clean.
* **`Portable`-only registration faults** (r215 §4: returns success, then `Page not present or supervisor
  privilege` on the first in-place cold read).  Pool mode depends on the alias being `Mapped`; the gate must
  include a cold-page-cache arm so the first read is genuinely cold.

### P1 accept criteria

* P1.0's answer is written down, and the arm that faulted before now either runs or has a named, recorded
  reason why it cannot.
* A model whose **expert set exceeds RAM** runs on `halo` with `-ncmoe`/`-cmoe` + `-lme auto`, and the census
  reports real pool fills.
* Same-seed greedy text **byte-identical** to the best all-resident arm on a config where both fit (the
  `--host-experts pinned` / `-ncmoe 0` arms), hashed with **no `-lv`**.
* `MOE_HOST_EXPERTS_UMA=0` reproduces today's behaviour exactly (the A/B that proves the new path is the only
  difference).
* `scripts/gate-host-pool.sh` green; a UMA arm added to it (§7).
* No `HSA_STATUS_ERROR_MEMORY_FAULT` in any arm.

## P2 — the UMA **arena** as the disk-backed cache (the maintainer's shape)

Only start this after P1's numbers are in: it must beat P1 at **equal RAM**, or P1 is the answer.

### P2.1 — preflight: replace "OFF" with a RAM-sized arena

`ggml/src/ggml-cuda/moe-expert-cache.cu:5604` (`moe_cache_shared_pool_device`) and `:5622`
(`moe_cache_preflight`).  Today the integrated branch *disables the whole cache*.  New branch, taken when
`prop.integrated`:

* budget = explicit `MOE_EXPERT_CACHE_MIB` if set, else `MOE_EXPERT_CACHE_UMA_RAM_PCT` (new; start at 60,
  tune) × `MemAvailable` read **once, at preflight**, floored by `MOE_EXPERT_CACHE_MIN_MIB`.  **Never** from
  "free VRAM" — that is the 2026-10-19 over-commit (121 GiB resident + 13 GiB swap) and the reason the gate
  exists.
* `MOE_EXPERT_CACHE_UNIFIED=1` + explicit MIB + `GGML_CUDA_SLAB_RESERVE_MIB` stays as the **carve-out** arm
  (Windows APUs with a fixed VRAM partition).  UMA and carve-out are different detections and must not be
  folded together.
* Kill switch `MOE_EXPERT_CACHE_UMA` (default `1` on integrated); `0` = today's OFF, verbatim.

### P2.2 — the slab must map only the budget

`ggml/src/ggml-cuda/ggml-cuda-slab.cu:441` (`ggml_cuda_slab_reserve_mib`), `:413`
(`ggml_cuda_slab_reserve_bytes`), `:579` (`ggml_cuda_slab_arena_limit`), `:384` (`ggml_cuda_slab_seed_floor`).  The slab sizes itself as
`free VRAM − reserve`; on a shared pool that is the machine (`SLAB_INIT size(VA)=98.44 GiB` measured).  Add
`ggml_cuda_slab_set_uma_total(size_t)` fed from P2.1's budget so `SLAB_INIT` cannot exceed it, and make the
`SLAB_INIT` census line print which source chose the total (`uma-budget` / `free-vram`).  **This is the
load-bearing safety change of the campaign** — it is what makes reason #1 impossible rather than unlikely.

### P2.3 — fills go straight into the slot

* `fill_slot_from_host()` (`moe-expert-cache.cu:3433`) and the `access_locked` (`:3476`) miss path already
  `cudaMemcpyAsync` from a host source into `dst`; on UMA `dst` is the arena slot in GTT, which is
  host-writable, so the path works unchanged.  Measured: 31.5 GB/s cold / 77.1 GB/s warm through
  `hipMemcpy`; a host `memcpy` alternative is 12.8 GB/s (do not prefer it).
* The **disk** tier keeps the existing staging/bounce ring: `O_DIRECT` into a GTT buffer is refused
  (`EINVAL`), so `pread` → pinned staging → DMA.  Reuse `bounce_acquire()` (`:3079`), `MOE_HOST_POOL_STAGE_SPAN`
  (coalesced reads) and `MOE_HOST_POOL_DIO`; add no new transport (see `wip/expert-transfer-engine/`).
* Verify (assumption to test, not proven): that `expert_host_src()` (`:3234`) resolves a **pageable mapping**
  master in pool/UMA mode without ever handing that pointer to a kernel; the fill is a host read, which is
  legal.  Assert it in the gate by faulting the source (`MOE_HOST_POOL_FILL_FAULT`) and requiring decline, not
  a GPU read.

### P2.4 — eviction moves no data

* In `access_locked` (`:3476`), the **arena→pool hand-over** (the `bounce_acquire` (`:3082`) user at `:3079`)
  exists because the victim's bytes must survive in the pool.  In UMA mode there is no pool to preserve them: skip it, leave the
  victim's slot stale, and let the incoming fill overwrite it.  Residency is the table, nothing else.
* Nothing must `free`/`madvise`/`unregister` on the eviction path — the arena is a plain `hipMalloc` slab (the
  measured `MADV_DONTNEED` silent-zero trap applies to registered host regions; do not introduce one here).

### P2.5 — soft evict (optional, and free on UMA)

Dropping the table entry while the bytes are intact makes a re-admission free.  The cache already has the
primitive: `MOE_EXPERT_CACHE_PROVISIONAL` / `g_prov_evict` (`:722`) — "pre-filled entries are evictable like
empty slots until first hit".  Extend that state to "evicted but intact" with a generation counter so a stale
entry cannot be mistaken for a fill.  Sequence this last; treat it as a policy experiment, not a requirement.

### P2 accept criteria

* A >RAM model runs with the arena sized from the RAM budget, and the census proves it: `arena X MiB` ≤
  budget, non-zero fills, non-zero evictions, and **zero** arena→pool hand-overs.
* Same-seed text byte-identical to P1's at equal RAM; prefill and decode measured against P1 (interleaved
  arms, no other work on the box).
* `SLAB_INIT` size ≤ the budget in every documented config; the whole-machine mapping is unreachable.
* `scripts/gate-uma-arena.sh` (§7) green; `check-gpu-faults.sh` clean.

## P3 — one budget on a shared pool

* `common/fit.cpp` (~L700): with a UMA arena the `--fit` margin must **reserve from `MemAvailable`** rather
  than skip the reserve as "refused" (`cache_reserve.arena_refused(dev)`).  The `refused_shared` warning text
  goes away on the UMA arm; the carve-out arm keeps it.
* `--fit` context sizing must use the shared-pool figure on `IGPU` (the `GGML_BACKEND_DEVICE_TYPE_IGPU`
  branch already exists in `common/fit.cpp`; verify it, do not assume).
* Collapse the two knobs into one on UMA: `-lme auto` (pool) and `MOE_EXPERT_CACHE_*` (arena) draw from a
  single `MOE_UMA_RAM_PCT` envelope, and the census prints the split.  Whatever is left is the page cache.
* Document the whole thing in `ENVIRONMENT.md` with classes: `MOE_EXPERT_CACHE_UMA` (kill switch),
  `MOE_EXPERT_CACHE_UMA_RAM_PCT` (tuning), `MOE_HOST_EXPERTS_UMA` (kill switch), `MOE_HOST_POOL_RO` if P0.5
  needs one.

## P4 — policy unification (only if measured)

A single heat table over "everything resident" instead of "arena = pool's complement"
(`pool_rerank_from_heat_locked`, `:2932`).  Do this only if P2's census shows the two-tier split
losing mass it should hold; otherwise it is churn on a working policy.  The heat prior (MHT2 sidecar,
`apply_heat_seed_locked`) is already the right input.

---

## §7 The gate: `scripts/gate-uma-arena.sh`

Modelled on `gate-host-pool.sh` (seven arms, byte-identical text, `takeover=0` under faults).  Arms, in order:

| # | arm | pass criterion |
|---|---|---|
| 1 | **text identity** — UMA tier on vs `MOE_HOST_EXPERTS_UMA=0` (P1) / `MOE_EXPERT_CACHE_UMA=0` (P2), on a config that **evicts** | same-seed greedy text **byte-identical**, hashed with **no `-lv`** (the `-lv 3` trap: it pollutes the text and hides the censuses) |
| 2 | **>RAM** — an expert set larger than RAM | run completes; census shows fills **and** evictions; no `HSA_STATUS_ERROR_MEMORY_FAULT`; `check-gpu-faults.sh` clean |
| 3 | **cold page cache** — same run after dropping the file's page cache | the disk tier really is exercised (fill count and bytes move); text identical to arm 1 |
| 4 | **budget** — explicit `MOE_EXPERT_CACHE_MIB=N` | `SLAB_INIT` total ≤ N + the documented reserve; `MemAvailable` after load is above the OS reserve |
| 5 | **kill switch** — `MOE_EXPERT_CACHE_UMA=0` / `MOE_HOST_EXPERTS_UMA=0` | the run reproduces the pre-campaign behaviour (arena off / host experts refused) and the warning text is unchanged |
| 6 | **carve-out arm** — `MOE_EXPERT_CACHE_UNIFIED=1` + MIB + `GGML_CUDA_SLAB_RESERVE_MIB` | still works exactly as today (it is the Windows fallback) |
| 7 | **host-buffer A/B** — `GGML_FORCE_NO_INTEGRATED=1` | throughput unchanged within noise, text identical: proves the tier does not depend on the zero-copy UMA weights path |
| 8 | **fill-fault** — `MOE_HOST_POOL_FILL_FAULT=1` | declines (correct text, `takeover=0`), never a GPU read of a pageable master |

Census lines are read at `-lv 4` (llama/ggml INFO); the text is hashed at no `-lv`.  Where a check needs the
allocation path, use the existing `SLAB_INIT` / `MoE expert cache:` / `expert access:` lines and add the
`uma-budget` source tag from P2.2.

Also required before a tag (per `MANIFESTS.md`): `gate-per-block-build.sh --changed-from <prev tip>`, the
prefill-logit gate if the fill path touches prefill, and `validate-set.sh` for any regenerated patch.

## §8 Docs this campaign must touch when it lands

* `ENVIRONMENT.md` — every new variable, class, default, and the "on because it helps / off because risky"
  distinction.
* `INVARIANTS.md` — append: (a) gfx1151 has no `xnack` target and a fault is fatal; (b) an APU's "VRAM" is
  the host pool (2 GiB carve-out / 130 GB GTT measured here); (c) a GTT allocation is host-writable, an
  eviction moves no data; (d) read-only registration is 233 GB/s vs 60 writable; (e) registered ranges are
  pinned and a `MADV_DONTNEED` on a GPU-visible anon range is a **silent zero**; (f) a pageable mapping may be
  a fill source and never a read target.
* `patches/README.md` — block 09/12 note (whoever owns the tier after promotion).
* `MANIFESTS.md` — gate table row for `gate-uma-arena.sh`.
* `WORKLOG.md` — one dated entry per phase that lands.
* `TODO.md` — anything deliberately deferred (e.g. gfx1150, Windows carve-out folding).

## §9 Effort and sequencing (estimate, not a commitment)

| phase | scope | risk |
|---|---|---|
| P0.5 | one flag + an A/B on both boxes | low; falls out on its own |
| P1.0 | investigation only, no code: reproduce the `mul_mat_q` fault and separate (a)/(b)/(c) | low effort, high information — it can shrink or kill P1 |
| P1 | 2 files + a policy + sizing assertions | medium: it re-opens a path that has faulted twice |
| P2 | preflight, slab budget, eviction, census | high: the arena is the code the gate called "shared pool" |
| P3 | fit/reserve + one envelope | medium |
| P4 | policy | low, optional |

## §10 Rollback

Each phase is behind its own switch, so rollback is `MOE_EXPERT_CACHE_UMA=0` / `MOE_HOST_EXPERTS_UMA=0` without
a rebuild.  If a phase cannot be made switch-safe, it does not ship — that is the lesson of the guard this
campaign replaces (`wip/umem-arena-guard`, whose comments are still in the tree as the record).
