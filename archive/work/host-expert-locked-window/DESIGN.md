# DESIGN — `--host-experts window`: the mapping as the residency tier

Normative document for `archive/work/host-expert-locked-window/`.  It specifies the mechanism, the ordering rules and
the tables; it does **not** specify the policy (that is the pool's, unchanged) and it is **not** a patch.
Everything here is a design under test — where it says "must", it means "the design is wrong if it does not",
not "the code does this today".

## 0. The problem, stated exactly

A host-resident MoE expert (`-ncmoe`/`-cmoe`) must be readable by a GPU kernel.  On gfx1201 (`XNACK enabled:
NO`) a kernel read of a **pageable** page faults fatally, and there is no fault-and-retry.  So the GPU may only
read a page that is, at the instant of the read, **present and not reclaimable**.

The delivery currently satisfies that with a *copy*: the pool keeps a pinned slot per hot expert, copied from
the page cache on a miss.  A pinned slot is present-and-unreclaimable **by construction**, which is why nothing
else is needed.  The window satisfies it *in place*: the page is made present-and-unreclaimable in the mapping,
and the GPU reads the original bytes.

So the mode is not "read pageable memory" (that is the refuted arm).  It is **"make these pages pinned, then
read them where they are"** — a *hold*, with an explicit table saying which pages are held.

## 1. Tiers

```
L1   VRAM expert-cache arena        (unchanged; the real cache; the policy's target)
L2   the held window over the       (NEW: the pool's *storage* role, without the copy)
     model mapping
L3   page cache -> GGUF on disk     (unchanged; reclaimable)
```

`L2` has the same capacity shape and the same policy as today's pool (`MOE_HOST_POOL_MIB`/`FRAC`), and the same
event set (`promote`, `handover`, `p_evict`, `cold`); only the storage differs.  That is deliberate: v1 of the
prior campaign built a second LRU over the window and measured 118 hits in 11 776 promotes, because the arena
is the cache.  **The window must be the pool's storage, not a second policy.**

## 2. Granularity: hold experts (chunks only if the register needs them)

Two levels, and they are not the same level:

| level | unit | what it is for |
|---|---|---|
| **hold** | one expert blob, as first written; a chunk (`WINDOW_CHUNK_MIB`, default 64 MiB) **only if** P0.5 shows per-expert registration is too expensive or the driver does not merge adjacent registrations |
| **residency** | one expert blob | the bit the cold path consults.  Set only when that expert's own pages are populated inside a held chunk |
| **read** | the expert blob | what the kernel indexes; the alias is the identity pointer on this platform |

A chunk is held while **any** expert in it is granted.  Populating is per expert
(`MADV_POPULATE_READ` over the expert's own range), so a held chunk can contain cold and hot experts at once.
Chunk release is a policy decision (the pool's eviction, at chunk granularity: release a chunk when none of its
experts is resident), event-gated (§5).

Why not expert-granular locks: measured in `MEASUREMENTS.md` (VMA count and per-syscall cost) — the chunk form
is the one that keeps `max_map_count` and the syscall count sane, and it loses nothing because the residency bit
is per expert anyway.

## 3. State machine

Per host tensor: the source of truth is the **mapping**, not a separate allocation.  The grant primitive is a
registration over the granted range (see §4 and `MEASUREMENTS.md` §1.2 — on this platform `hipHostRegister`
**faults the range in and holds it**, so it is the populate *and* the pin, not merely a mapping).

```
GRANTED(expert)  --hipHostRegister(expert range)-->  HELD      (resident + pinned + device-addressable)
HELD             --(op submitted; the policy retires the expert)-->  RELEASING
RELEASING        --(every device's read of the range complete)--> hipHostUnregister --> COLD
```

The invariant that makes the whole design safe:

> **I1. A kernel may read an expert's pages only while its residency bit is HELD.**  HELD implies: the driver
> holds the range (`hipHostRegister` succeeded) **and** the holder has not been released.  Any code path that
> can produce an in-place read must consult the bit; a path that cannot consult it must take the bounce.

and its consequence, which must be enforced in the alias resolution (not in the policy):

> **I2. A page that is not HELD is never handed to a kernel as a device pointer.**  `moe_cache_get_cold`-style
> leakage of `t.host_dev + offset` for a COLD expert is the fatal-fault bug; the fallback is the bounce ring.

## 4. Ordering (each step is a rule, not a suggestion)

```
grant(t, e):
  1. hipHostRegister(expert range, size)         # faults the pages in, holds them, installs the mapping
  2. if (1) failed: treat the grant as REFUSED -> fallback (README §6/§7), record the cause
  3. residency[t][e] = HELD                      # release-store, host side, after the register returns
  4. the op is submitted / the alias may be returned
```

- **I3. The register is the populate and the hold.**  `MEASUREMENTS.md` §1.2: registering a range whose pages
  are absent faults them in (page cache 0/425 → 425/425) and a later `MADV_PAGEOUT` cannot take them back.
  A separate `mlock2`+`MADV_POPULATE_READ` step is therefore **provisional**: it is kept in the model below
  only until P0.5 shows (a) the driver's hold survives real pressure and (b) `hipHostUnregister` releases it.
  Measured cost if it is kept: 375 µs per 1.66 MiB hot — 23x the register — and 18.6 ms cold, so the register
  is the better primitive on both paths.
- **I4. The residency bit is set after the hold exists and before the op.**  Never before (a kernel could read
  an unheld page) and never after (the op could run with the bit clear and take the bounce needlessly).
- **I5. Register the *window*, not the tensor.**  ⚠ Corrected by measurement: a registration is an eager hold
  of everything it covers, so "register once per tensor" pins the whole host expert set — the state that made
  the prior campaign conclude the RAM saving was gone.  The grant is `hipHostRegister` over the range being
  granted, and the release is `hipHostUnregister` over the same range (16.35 µs / 0.49 µs at 1.66 MiB, hot).
- **I6. A copy out of a registered range is D2D from the alias.**  An H2D `cudaMemcpyAsync` from a registered
  range is staged through ROCm's internal bounce (prior campaign v1 cause 1).

```
evict(t, e):
  1. the policy has decided the expert is no longer resident
  2. every device that can read the tensor has completed every op that read it  (§5)
  3. residency[t][e] = COLD
  4. hipHostUnregister(expert range)     # releases the hold; the pages become reclaimable again
```

- **I7. Never release a range a device may still be reading.**  The prior campaign measured that
  unregistering a range a queued copy still reads returns `Success` — *not* a safety guarantee.  Here the
  consequence is the fatal fault rather than a stale byte (a released range that is then reclaimed and read is
  `Page not present`), so the gate is mandatory, not defensive.
- **I8. The bit clears before the unlock**, so a racing grant sees COLD and takes the bounce rather than
  reading in place.

## 5. The eviction gate (the `-sm tensor` requirement)

"The GPU has finished with this chunk" is, under a split, a **set** of devices, not one stream.  The gate is
therefore an **event proxy fanned out to every device that can read the tensor**, resolved with one
`event query` on the host side at a policy boundary — the epoch scheme the prior campaign needed anyway (a
per-entry device query inside the rescan loop cost ~90 ms/token; one boundary event per policy call is the
fix).

- **I9. The gate is a device set from day one.**  Building it on device 0's stream and "porting it later" is
  exactly how PR #266's remap cornered itself (`device_count != 1` guard in `pool_cold_slot_locked`, and the
  Meta iface member that does not exist under `-sm tensor`).  The first measured arm is 1 GPU; the *code
  shape* is N devices with a fan-out of size 1.
- **I10. One residency table and one held-byte budget per host tensor.**  Under a tensor split both devices read
  the same bytes through the same UVA alias (`cudaHostGetDevicePointer` == the host pointer here:
  `canMapHostMemory` yes, `unifiedAddressing` no, measured in the prior campaign).  A per-device table would
  both multiply the `RLIMIT_MEMLOCK` accounting and let device 1 read what device 0 released.
- **I11. No new per-simple-device backend iface member.**  State lives in the global per-host-tensor table the
  existing `moe_cache_*` hooks already reach, so the Meta/split forward path can consult it.

## 6. The residency table

Per host tensor (process-wide, keyed by the host weight the way `g_key_to_id` already is):

| field | meaning |
|---|---|
| `host`, `host_dev` | the mapping base and its device alias (`cudaHostGetDevicePointer`, resolved once — this is what makes `table_cold_ok` true) |
| `n_experts`, `host_bytes`, `host_pitch` | geometry, as `table_t` already carries |
| `chunk_bytes`, `n_chunks` | the hold granularity (§2) |
| `chunk_state[]` | `COLD`/`LOCKED` + the epoch of its last release |
| `residency[]` | per-expert `COLD`/`HELD` (**the bit I1/I2 turn on**) |
| `grant_epoch[]`, `devices` | the eviction gate's inputs: the last op epoch that read the chunk, and the devices that can read the tensor |
| `grant_fail` | refused grants by cause (`ENOMEM`/`EPERM`/register failure), for the startup/degrade decision |

The kernel reads the **expert → residency** decision indirectly: the host resolves it in the op's setup
(`moe_cache_alias_get`-style) and passes either the in-place pointer or an arena/pool-bounce slot id.  Nothing
new needs to be visible to the kernel itself, which is what keeps the design split-portable.

## 7. Failure handling: the bounce is the fallback, not the mechanism

A refused grant (any cause in `README.md` §6) degrades **that expert** to the existing bounce path — the pool
slot if the pool exists in this mode, the per-table pinned ring otherwise — and records the cause.  The run is
correct; it simply pays the copy for the entries the box would not let it hold.  A **table** whose registration
fails degrades entirely (same path).  A **process** whose `RLIMIT_MEMLOCK` cannot be raised at all logs one
WARN and refuses the mode at startup rather than running permanently degraded.

That is also why the correct answer to "why would a lock be refused" is *not* "a bug": the mode must have a
refused-grant path, because the budget is a deployment property (measured on this box: `RLIMIT_MEMLOCK` soft ==
hard == 8 MiB, so ~4 experts).

## 8. Environment surface (wip only, none of this is in `patches/`)

| variable / flag | default | class |
|---|---|---|
| `--host-experts window` | off (`pinned`) | the mode switch (new enum value `LLAMA_HOST_EXPERTS_MODE_WINDOW`) |
| `MOE_HOST_POOL_MIB` / `_FRAC` | auto | **reused** as the held-byte budget (no new knob) |
| `LLAMA_WINDOW_CHUNK_MIB` | 64 | tuning: lock granularity (§2) |
| `LLAMA_WINDOW_POPULATE` | `read` | tuning: `MADV_POPULATE_READ` vs a read-touch loop (A/B only) |
| `LLAMA_WINDOW_EVICT_GATE` | on | kill-switch: `0` = ungated `munlock` (the unsafe arm; A/B only, never valid) |
| `LLAMA_WINDOW_FALLBACK` | `bounce` | `refuse` = fail the op instead of bouncing (debug only) |
| `LLAMA_WINDOW_TRACE=1` | off | diagnostic: per-grant/refuse/evict log line (the simulator's input on the host side) |

## 9. Rejected alternatives (and why)

| alternative | why not |
|---|---|
| **A device-side "on miss, stop the op, ask the host, retry"** | The GPU has no information the host lacks: at decode the routing ids already reach the host every step (the eager admission path needs them for LFRU/heat), so a request carries data the host received one op earlier.  The two implementations are both worse: an in-kernel spin stalls the whole device (each miss ≥60 µs plus host turnaround, on a token that touches ~n_used x n_layers blobs), and a pre-pass + relaunch is a host sync per op per layer (48-96/token) that also forbids CUDA-graph capture.  This is the machinery `moe_cache_promote`/`moe_cache_inplace` was, deleted for cause.  The one cheap version — the GPU *reports* a miss one op late — is a prefetch prior, and the host already has the routing one op early. |
| **Fault-and-retry (the implicit XNACK shape)** | not available on gfx1201; a fault is fatal, not a trap the driver can service.  (On `gfx90a`/`gfx942` with `HSA_XNACK=1` it is available and this campaign is unnecessary — out of scope per `AGENTS.md`.) |
| **`hipMemAdvise`/`hipMemPrefetchAsync` on the registered range** | accepted and useless: a hint is not a hold (prior campaign, measured). |
| **Per-miss register/unregister as a cost to avoid** | refuted by measurement: 16.35 µs to grant and 0.49 µs to release a 1.66 MiB blob hot (`MEASUREMENTS.md` §1.4), i.e. 3x cheaper than the pool's `memcpy` — and it is the hold. |
| **`mlock` as the hold** | measured at 375 µs hot (mlock2 + populate) vs 16 µs for the equivalent grant, and 18.6 ms cold vs 1.2-3.2 ms: right semantics, wrong primitive, and `RLIMIT_MEMLOCK` makes it a deployment requirement.  Kept only until P0.5 confirms the register's hold is durable *and* releasable. |
| **A window-side LRU policy** | v1's 118/11 776 result.  Use the pool's policy. |
| **Holding the whole set at startup** | the state that killed the prior campaign: 38.7-59 GB held is pinning, i.e. no RAM saving.  The window must slide. |
| **Expert-granular `mlock`** | VMA explosion (§2). |

## 10. What the code has to touch (when P2 starts)

| file | change |
|---|---|
| `include/llama.h` | `LLAMA_HOST_EXPERTS_MODE_WINDOW` next to `PINNED`/`POOL` |
| `common/arg.cpp` | `--host-experts window` (the option already exists; add the value) |
| `src/llama-model-loader.cpp` | `host_expert_buft()`: `window` takes the `ROCm_HostMap` master like `pool`; plus the once-per-tensor registration |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | the residency table + `populate`/`evict`; the **gate in the alias resolution** (I2); reuse `pool`'s policy and budget; the fallback hook |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | the `ROCm_HostMap` device alias after registration (`host_dev` non-null → `table_cold_ok` true) |
| `ggml/src/ggml-backend.cpp` | nothing new: the scheduler already keeps a host-weight `MUL_MAT_ID` on its layer's device |

Non-negotiable at review: I1, I2, I7, I9, I10, I11 — each is either a fatal-fault (I1/I2/I7) or a
`-sm tensor` dead end (I9/I10/I11).
