# archive/work/host-expert-locked-window — the pool's policy, the mapping's storage, an explicit hold (CLOSED: blocked by the driver)

> **VERDICT 2026-11-03 (P0.5): the sliding window is BLOCKED by the driver — do not build it.**
> `hipHostRegister` is a hold that lives for the process's life: `hipHostUnregister`, `munmap`, a MAP_FIXED
> remap and every registration flag (`ro`/`portable`/`mapped`) all leave the window's bytes non-reclaimable
> under 67 GiB of real pressure (cgroup `file` unchanged).  A bounded *sliding* window therefore cannot exist,
> and without one the footprint is the whole registered set — `pinned` with a longer load.  The full evidence
> is `MEASUREMENTS.md` §4; what survives is there too (the pool's copy is the only *releasable* residency
> primitive here, and a **static** bounded registered subset is the honest pivot).  Read this campaign as a
> refutation, not as a plan.

**Seeded 2026-11-03 at the maintainer's call.  Nothing here is implemented, and nothing here is in
`patches/` or gets applied to the fork** (`AGENTS.md`, WIP rules: a campaign is a handover, and promotion is a
separate decision).

**The charter, in the maintainer's words:**

- **Goal: the pool approach, without the bounce buffer.**  The host `mlock`s the region, reads it in so it is
  resident, and only then gives the go-ahead back to the GPU to continue.  The pool already hands the go-ahead
  back — but *after copying*.  Remove the copy; keep the read (the read is what guarantees residency).
- **Resurrect `--host-experts window`** as the mode that selects it.
- **`pinned`, `pool` and `window` co-exist** until `pool` is deliberately deprecated and dropped.
- **First arm is 1 GPU**, but **`-sm tensor` must remain supportable**: do not take a decision that backs the
  design into a corner, the way PR #266's device remap did (it is single-device by construction and declines
  under a split — see §8).
- The closed `archive/work/host-expert-pinned-window/` campaign is **not** taken as settled: its hold half was
  dropped mid-flight (and `MOE_EXPERT_WINDOW_MLOCK` was, on the reference box, refused outright by an 8 MiB
  `RLIMIT_MEMLOCK`), so the "one residency tier" shape never got to run with the guarantee it was designed
  around.  This campaign re-runs that shape **with the hold re-attached, at window granularity**.

## 1. The claim

A host expert's bytes can be made device-readable **in place**, in the read-only GGUF model mapping, provided
the host has (a) installed a device mapping over the range once per tensor and (b) **held** the specific pages
the GPU is about to read.  Once that is true, the pool's pinned copy is redundant: the intermediary becomes a
**residency table over the mapping**, not a second set of bytes.

That replaces, per miss:

```
today (pool)   page cache --memcpy--> pinned pool slot --CE--> VRAM arena --> op
proposed       page cache --populate--> (held in place) --SM read over PCIe--> op
```

and deletes the per-miss host `memcpy` from the token's critical path, the pool's duplicated RAM
(`slots x host_bytes`, 6.8-18.6 GiB on the measured arms) and one of the two copies a miss pays for.

## 2. What is already established (do not re-derive)

From `archive/work/host-expert-pinned-window/` (`RESOLUTION.md`, `PROTOTYPE.md`) and `INVARIANTS.md`:

- **A registered range is addressable, not resident.**  `cudaHostRegister`/`hipHostRegister` installs page
  tables (UVA alias, host pointer == device pointer) and leaves the pages ordinary reclaimable page cache.
  `Mlocked` stays 0.  Cost is ~0.6 µs, flat to 256 MiB, and there is no page-alignment constraint.  It must be
  done **once per tensor**, never per miss.
- **Without XNACK there is no fault-and-retry.**  A device read of a page that is not present is a fatal
  `Page not present` fault on the first completion (`mul_mat_vec_q_moe`).  Measured on a set larger than RAM:
  the unheld arm aborts, `pinned` on the same cell runs.
- **A partly-lockable set hangs rather than faults** (no fault, no progress, GPU at 100 %) unless *every*
  in-place read is gated on the hold.  The gate being exhaustive is a correctness requirement, not a detail.
- **In-place reads are at pinned-class bandwidth**: 51.8 GB/s (`cudaMallocHost`) and 55.7 GB/s sustained on a
  256 MiB registered window, vs 52 GiB/s for pinned (`probe-aperture`).  The hold costs nothing in bandwidth.
- **A cold in-place populate is 7-13x cheaper than the pool's cold `memcpy`**: 8 MiB with 0/2048 pages resident,
  2.3 ms (3.6 GB/s) vs 17.0 ms (0.5 GB/s), because the faults batch through GUP instead of faulting
  page-by-page inside a `memcpy`.  This is the tier the maintainer's charter calls the actual pain.
- **A registered range must never be read by an H2D copy**: ROCm stages a `cudaMemcpyHostToDevice` from one
  through an internal bounce (v1 cause 1).  Resolve the alias and copy **D2D**.
- **Do not build a second LRU over the window** (v1 cause 4: 118 hits in 11 776 promotes).  The arena is the
  cache; the window is a **storage** tier whose content the existing policy decides (v1 causes 2 and 3 — the
  epoch/event-gated demote — are how the eviction side must be written).

What this campaign adds to that record: the hold has only ever been tried **per whole tensor**, all-or-nothing,
which is why it either faulted (unheld) or held the entire set (38.7-59 GB, no RAM saving).  A **sliding,
window-granular hold** has never been measured.

## 3. What must be measured

1. **The lock/populate/unlock cost** per expert and per chunk, hot (page-cache) and cold (real disk), against
   the `memcpy` + CE copy it replaces.  → `probe-locked-window.cpp`
2. **The residency guarantee is load-bearing**: a held+populated range must be data-identical through real
   eviction pressure, and the **negative control** (registered, populated, then paged out **unheld**) must
   fault.  A gate that cannot fail is not a gate.  → `probe-locked-window.cpp --evict-control`
3. **The churn rate at a real miss rate**: how many locks/populates/unlocks a decode token pays at a given
   window budget.  → `sim-locked-window.py` (the existing two-tier policy model, priced with the measured
   costs; no new policy)
4. **That the numbers beat the pool** on the campaign arm, with byte-identical text.

Measured so far: see `MEASUREMENTS.md` (probe + simulation logs in `runs/`).

## 4. The shape of the mode

```
--host-experts <pinned|auto|pool|window>
```

| mode | host master | GPU reads |
|---|---|---|
| `pinned` (default) | `ROCm_Host` full pinned copy | in place, always (`cudaMallocHost` pinned) |
| `pool` | read-only model mapping (`ROCm_HostMap`), **no** pinned master | through the bounded pinned pool (`MOE_HOST_POOL_MIB`/`FRAC`) |
| **`window`** (new) | read-only model mapping (`ROCm_HostMap`) + one `hipHostRegister` per tensor | **in place**, gated on a per-expert residency bit backed by a **held** (`mlock2`+populate) sliding window |

`window` is `pool` with the copy removed: it reuses the pool's **policy** (heat/LFRU admission, bulk admit +
rerank, the arena's existing slot map) and its **budget knob** (`MOE_HOST_POOL_MIB`/`FRAC` now means "how many
MiB of the mapping are held at once") rather than introducing a new tuning surface.  `MOE_EXPERT_CACHE_*`,
`GGML_MOE_DEVREMAP` and the arena are unchanged and orthogonal.

`window` implies `use_mmap`; without a file mapping there is nothing to register, so the loader logs and falls
back to `pinned` (the same way `pool` refuses without mmap).

## 5. Mechanism (the contract)

`DESIGN.md` is the normative document: the state machine, the ordering rules, the granularity, the residency
table, the eviction gate, and the failure modes.  Short version:

```
populate(expert):  mlock2(page range, MLOCK_ONFAULT)  ->  MADV_POPULATE_READ  ->  residency[expert] = HELD
grant:             the op may now read; a miss is impossible for a HELD expert
evict(expert):     stream has finished with the chunk (event-gated)  ->  munlock  ->  residency[expert] = COLD
```

The GPU never asks.  The policy (host) decides what to hold **before** the op that reads it — that is the
information the pool already uses, so the "guided XNACK" round-trip is not needed and would only add latency
(see `DESIGN.md` §9 for why a device-side request leg is rejected).

## 6. Why a grant can be refused

**P0 has already changed this section**: `hipHostRegister` turns out to be the hold (it faults the range in
and pins it — `MEASUREMENTS.md` §1.2), so the grant primitive is a registration and the `RLIMIT_MEMLOCK`
column below is **provisional** — it applies only if `mlock` survives P0.5 as part of the hold.

| cause | what happens | handling |
|---|---|---|
| **`RLIMIT_MEMLOCK`** (soft and hard, per process) — *only if `mlock` is kept* | `mlock2` fails `ENOMEM` once the process' locked bytes pass the limit.  **On this box: 8192 KiB soft == hard, i.e. ~4 experts of 1.66 MiB** | *deployment requirement*: `LimitMEMLOCK=` (systemd) or `CAP_IPC_LOCK`.  The mode must detect it at startup, log it once and refuse loudly rather than run partly held |
| **memory cgroup / OOM** | `mlock` fails `ENOMEM` when the cgroup cannot miss-account the page | the miss falls back (§7); the window shrinks; a WARN names the budget |
| **`vm.max_map_count`** | expert-granular locks split a VMA per range; at 1.66 MiB x 512 experts per tensor this exhausts the default 65530 | lock in **chunks** (default 64 MiB), track residency per expert inside a held chunk (§DESIGN 4) |
| **registration failure** (I/O error, `ENOMEM`/`EPERM` from the driver, a range the driver will not map) | the range is not held, so an in-place read would be fatal | refused grant: fall back (bounce), record the cause |
| **the kernel or the mapping says no** (page not mappable, `PROTECT_NONE`, DAX/device memory) | `mlock` returns `EPERM`/`EINVAL` | refuse the table (degrade that table to the pool path) |

**And yes — your reading is right about the fallback**: when a grant cannot be made, the correct last-ditch
recovery is to **bounce that expert into a known-pinned buffer** (today's pool slot, or the existing per-table
bounce ring when the pool has declined), which is exactly the path `pool` mode already has.  It costs what the
pool costs, only on the misses where the hold could not be obtained — and it is what keeps the mode correct
instead of fatal on a box where the lock budget is smaller than the window wants to be.

## 7. `-sm tensor` — the non-negotiable design rules

The campaign's first arm is 1 GPU, but the design must not repeat PR #266's corner:

- **One residency table per host tensor, not per device.**  Under a split the same host bytes are read by
  several devices through the same UVA alias (`cudaHostGetDevicePointer` returns the identity pointer on this
  platform: `canMapHostMemory` yes, `unifiedAddressing` no), so a per-device hold would multiply the budget and
  a per-device bit would let one device read what another had released.
- **The eviction gate must be a device-set event, not a single stream.**  "The GPU has finished with this
  chunk" is a cross-device question under `-sm tensor`; gate on an event proxy fanned out to every device that
  can read the tensor (the same shape the loader's `set_tensor_async` Meta arm uses), never on device 0's
  stream.  Building it single-device now and "porting it later" is what corners a design.
- **No new per-simple-device backend iface member.**  `GGML_MOE_DEVREMAP` declines under a split because the
  Meta backend's iface member is NULL (TODO #269); the window's residency must live where the Meta forward can
  reach it — the global, per-host-tensor table the existing `moe_cache_*` hooks already use.
- **The budget is per host tensor**, so a 2-GPU tensor arm does not double the held bytes.
- **Do not inherit `pool_cold_slot_locked`'s `device_count != 1` guard.**  It is the current pool in-place
  path's single-device restriction (a deliberate narrowing in `moe-expert-cache.cu`); the window's in-place
  read is not allowed to be single-device-only, or `-sm tensor` gets nothing.

## 8. Plan

| phase | what | state |
|---|---|---|
| **P0** | probes: grant/release cost (hot/cold), what registration actually guarantees, granularity, the eviction controls, alias + bandwidth, `RLIMIT_MEMLOCK` reality | **DONE — `MEASUREMENTS.md`; two results changed the design (registration is the hold; `mlock` is 23x the grant)** |
| **P1** | simulation: churn at a real miss rate, window budget sweep, break-even vs the pool's `memcpy`+CE, on the campaign trace | **DONE (ratio only; absolute calibration owed) — `MEASUREMENTS.md` §2** |
| **P0.5** | the four open questions `MEASUREMENTS.md` §3 lists: does the register's hold survive real pressure, does `hipHostUnregister` release it, what it costs in RAM accounting, and the gate's per-chunk negative control | **NEXT** |
| **P2** | the mode: `LLAMA_HOST_EXPERTS_MODE_WINDOW`, loader registration, the residency table, the policy hook reusing the pool's, the `bounce` fallback | not started (needs P0/P1 to agree the costs) |
| **P3** | 1-GPU arm: 122B-A10B `-ncmoe 48`, decode ≥ 2000 tokens, `pinned` / `pool` / `window` head-to-head; text byte-identical | not started |
| **P4** | the split: `-sm tensor` on the x8 pair, then the 3-GPU arm — the gate that decides whether the design is worth keeping | not started |

P2 must not start before P3's arm is priced: the whole question is whether the lock churn is cheaper than the
copy it removes, and P1 answers that from the trace without writing the mode.

## 9. Gates (when it becomes code)

- Same-seed greedy text vs `pinned` and vs `pool` (byte-identical), MTP four-axis
  (`benchmarks/mtp-adaptive-methodology.md`), `-sm tensor` coherence on the x8 pair.
- **A residency gate**, which is the one that matters and which no existing gate covers: the in-place read must
  be correct under real eviction pressure — multi-ubatch prefill on a set that competes for RAM, inside a
  `memory.max` cgroup, with a **negative control** (hold released behind the op's back) that faults.  The
  prefill-logit KLD gate is blind to this class (`RESOLUTION.md`), and a short prompt does not exercise
  eviction.
- `scripts/gate-per-block-build.sh` and `scripts/check-gpu-faults.sh` before any tag (per `AGENTS.md`).

## 10. What would kill it

- **The lock churn costing more than the copy.**  If a decode token's miss set is large enough that
  `mlock2`+populate+`munlock` per miss exceeds the `memcpy`+CE it replaces, the mode is a loss on this box and
  the answer is "the pool's copy is the cheaper residency primitive here".  P1 prices this before P2.
- **`RLIMIT_MEMLOCK` being unraisable in the deployment.**  8 MiB holds ~4 experts; a window needs hundreds of
  MiB to GiB.  If the target installs cannot set `LimitMEMLOCK`, the mode is dev-only.
- **The hold not surviving real pressure** (the campaign's §3 negative control failing in the *wrong*
  direction: held pages faulting, or unheld pages not faulting — the latter would mean the probe is not
  actually evicting, which is the trap `probe-cold-read` fell into).
- **A `-sm tensor` arm that cannot be made cross-device** without a Meta iface member (then it is PR #266
  again and the honest outcome is "1 GPU only, documented as such, not a default").

## 11. Instruments in this directory

| file | what |
|---|---|
| `DESIGN.md` | the normative mechanism: state machine, ordering, granularity, tables, failure modes, rejected alternatives |
| `MEASUREMENTS.md` | what P0/P1 measured, in numbers, with the log paths |
| `probe-locked-window.cpp` | the P0 probe: hold cost (hot/cold), granularity/VMA cost, the eviction control, alias/bandwidth, `RLIMIT_MEMLOCK` reality |
| `build.sh`, `run-probe.sh` | build/run it against the canonical ROCm line, device-pinned (`AGENTS.md` rule 3) |
| `sim-locked-window.py` | the P1 simulator: prices the window tier against the pool tier from a real routing trace, reusing `scripts/moe-trace-replay.py`'s two-tier policy model (no new policy) |
| `runs/` | probe and simulation logs (dated) |

Provenance for the prior art: `archive/work/host-expert-pinned-window/` (probes, `PROTOTYPE.md`,
`reference-window-261102.diff`) and `archive/work/host-experts-window-beta/` (the withdrawn candidate);
the constraint is also `INVARIANTS.md` §"Device-visible host memory must be RESIDENT".  The campaign trace is
`wip/host-pool-speed/data/traces/trace-122b-ncmoe48-devremap0.txt.gz` (read-only; that campaign's arm, used
here as the workload model).  The parallel `wip/host-pool-speed/` campaign owns the pool's own tuning and edits
the same `moe-expert-cache.cu` region — the reconciliation point is the cold-read hook
(`table_cold_ok` / `moe_cache_get_cold` / `expert_host_src`), and neither campaign may rewrite it unilaterally.
