# The pinned working set: `mlock` over the mapping, and why it works where registration did not

Design note for `wip/host-expert-eager-push/`, seeded by the maintainer's proposal and **measured the same
day** (2026-11-03).  Instrument: `archive/work/host-expert-locked-window/probe-locked-window.cpp`, arm
`mlockram` (added for this note; log below).

## 1. The proposal

> Do away with the pool's bounce buffer — its true cost is **RAM** (a pinned duplicate of the hot experts).
> Enforce the FRAC policy by **`mlock`ing portions of the weights** so the system cannot reclaim them: keep
> pinned what is valuable to the arena, cycle the remainder.  Then `push` acts as a true extension of the
> arena cache.

That splits the pool into its two functions and keeps only the one that is needed:

| the pool's job | kept? | what replaces it |
|---|---|---|
**transport** (a pinned, safely-DMA-able source) | **dropped** | the direct pageable push (`patches/0001`) — measured equal to memcpy+pinned H2D when the source is warm |
**residency** (the hot set cannot be reclaimed, so a fill does not become a disk read) | **kept** | **`mlock` over the mapping's own pages** — same guarantee, no second copy of the bytes |

## 2. Does `mlock` give a *releasable* guarantee?  Yes — measured

This was the open question, because `hipHostRegister` is an unreleasable process-lifetime hold (that is what
closed the window campaign).  `mlock` is a different primitive and behaves correctly.  `arm mlockram`
(scratch file on the real xfs volume, so nothing is cached by accident; residency measured through a **fresh**
mapping, never the locked one; pressure **held** while measuring, because an allocate-then-free loop gives the
kernel no reason to reclaim — the trap that produced a false "not released" in the first version of this arm):

4 MiB window, unprivileged (uid 1000, `RLIMIT_MEMLOCK` = 8 MiB), and 1024 MiB as root:

| step | 4 MiB window | 1024 MiB window |
|---|---|---|
baseline | 0/1024 resident | 0/262144 resident |
`mlock` + populate | 1024/1024, VmLck 4 MiB | 262144/262144, VmLck 1024 MiB |
**67 GiB of anonymous pressure held** | **1024/1024** | **262144/262144** |
`MADV_PAGEOUT` while locked | **refused (`EINVAL`)** | **refused (`EINVAL`)** |
`munlock` | 12.4 µs (3.09 µs/MiB) | 831 µs (0.81 µs/MiB) |
`MADV_PAGEOUT` after `munlock` | accepted | accepted |
after `munlock`, pressure still held | **0/1024** | **0/262144** |

**Cost:** `mlock` (eager, which also populates) **48.7 µs/MiB** — ~50 ms per GiB; `munlock` **0.81 µs/MiB**.
Note this is ~4.6x cheaper than the `mlock2(MLOCK_ONFAULT)`+populate pair measured earlier (225 µs/MiB), so
the eager call is the primitive to use.

**So: `mlock` holds under real pressure and releases on `munlock`.**  The blocking property of the window
campaign does not apply here — the two primitives look similar and are opposites on the one axis that matters.

## 3. The tier model this produces

```
L1  VRAM arena                      (unchanged; the cache)
L2  the mapping's PINNED WORKING SET  mlock'd pages of the model mapping -- bounded by a FRAC/MiB budget,
    (NEW, replaces the pool)          heat-ranked by the same policy that fills the arena, rotated
                                      incrementally; NO second copy of the bytes
L3  the rest of the page cache       reclaimable, shared with anything else reading the GGUF
L4  the GGUF on disk                 the cold source
```

- A fill's source is always the mapping; whether it was `mlock`ed only decides **whether the push pays a disk
  read**: L2 → page-cache read (µs), L3 → page-cache read, L4 → disk (measured 1.32 ms per 1.66 MiB).
- `MOE_HOST_POOL_FRAC`'s budget keeps its shape, but it now counts **bytes locked**, not bytes copied.
- The pinned set is the arena's *complement of interest*: what the arena would want next, so the pushes that
  matter never go cold.  Rotation is incremental (a few chunks per policy boundary) because a full re-mlock of
  8 GiB costs ~400 ms — the policy's existing decay period (`MOE_EXPERT_CACHE_PERIOD`, default 32) is the
  natural cadence.

## 4. Consequences and caveats

- **`RLIMIT_MEMLOCK` becomes a deployment requirement** (8 MiB here, unraisable as uid 1000): `LimitMEMLOCK=`
  in the unit or `CAP_IPC_LOCK`.  It is *bounded* (the FRAC budget, a few GiB), unlike the window campaign's
  whole-set hold, and the mode must degrade to L3/L4 (slow, correct) rather than refuse.
- **Locked pages cannot be reclaimed by the kernel**, so the mode must `munlock` before an eviction can free
  RAM — the eviction path is ours, which is fine, but an over-large pinned set is an OOM hazard exactly as the
  pool is.  The difference is that it is the *only* pinned tier: no duplicate.
- **`mlock` does not change the transport.**  The driver is not told about it, so a push out of an `mlock`ed
  page is still a pageable `cudaMemcpyAsync` (staged internally by ROCm).  The 1-D tearing evidence
  (`P1-RESULTS.md`) applies unchanged, as does the 7.14 gate.
- **Prefill is the winner.**  The pool's `O_DIRECT` decision exists because a whole-table prefill would fault
  the entire set into the page cache; with a pinned working set, the hot part is already resident and only the
  remainder is cycled — which is the maintainer's point about not round-robining every page.
- **This is the mode ladder's middle rung**, and it is what makes `push` a true extension of the arena rather
  than a streamer: `pinned` (whole set non-reclaimable) → **`push` + mlock'd working set (bounded, releasable)**
  → `pool`/DIO (streams from disk).

## 5. What must be measured next (before any of it is code)

1. **A pinned working set over a *real* model mapping, not a scratch file**: `mlock` the top-`F` experts of a
   `--host-experts pool` master (heat-ranked), push from them, and confirm the push does zero disk reads for
   the pinned set and that a rotation does not disturb the arena's hit rate.
2. **Rotation cost under load**: mlock/munlock churn per token at a realistic `PERIOD`, and whether it is
   visible against decode.
3. **The L3 reclaim interaction**: with the set larger than RAM, does the *unpinned* part of the page cache
   actually get reclaimed (and re-read), or does the pinned set push the streaming path into OOM?
4. **`-sm tensor`**: one pinned budget per host tensor (not per device), and the eviction/release gate fanned
   out to every device that reads it — the campaign's P6 rule applied to the mlock path.

## 6. P2 increment 1 — implemented and measured (2026-11-03)

`patches/0002` (136 lines, `moe-expert-cache.cu`; branch `p2-pinned-working-set` @ `7699a492e`, on top of the
P1 push switch):

```
MOE_HOST_PUSH_MLOCK_MIB=N     pin up to N MiB of the host expert set (0/unset = off)
```

- **Policy enforcement, not transport.**  `mlock` over the mapping's own pages on **first use** of an expert
  (the fill path is the routing), tracked per table with a one-byte mask, counted in a process-wide budget.
  The driver is not told, so the transport is unchanged — this only makes the warm set unreclaimable.
- **Graceful degradation, exactly as specified.**  When the kernel refuses (the normal case on a box without
  `CAP_IPC_LOCK`), the mode logs **one** message naming the remedy and the actual limit, keeps what was
  already pinned, and carries on with the remainder served from the page cache — correct, just less warm:

  ```
  P1-PUSH: mlock refused (Cannot allocate memory) after 7 MiB pinned in 11 experts (requested 64 MiB).
  The pinned working set stops here and the remainder stays reclaimable page cache -- correct, just less
  warm.  To pin more, raise the limit for this process: `LimitMEMLOCK=<MiB>` in the unit, or grant
  CAP_IPC_LOCK (RLIMIT_MEMLOCK is 8192 KiB here).
  ```
- **Measured**, 35B-A3B, 1 GPU, `-ncmoe 40 --host-experts pool`, forced misses (`MOE_EXPERT_CACHE_MIB=2048`,
  `FRAC=50`), `MOE_HOST_PUSH_SRC=mapping`:

  | arm | pinned | refusals | text | result |
  |---|---|---|---|---|
  unprivileged, request 64 MiB | **7 MiB / 11 experts** | 1 (one message) | — | run correct, 17 421 mapping fills continue |
  root, request 4096 MiB | **4095 MiB / 6 744 experts** | 0 | — | run correct |
  request 0 (off) | 0 | 0 | `sha=0cf4a62aa63f` | control |
  request 64 MiB | 7 MiB | 1 | `sha=0cf4a62aa63f` | **byte-identical to the control** |

  `check-gpu-faults.sh --since 12 min` clean.  (No throughput claim from these runs: single-shot, and the
  first arm of a pair pays the cold page cache.)
- **Periodic, not continual** — honoured in the sense that pinning happens only on a first-use fill (a
  boundary event); there is no per-token re-pin, no polling, and no `munlock` in this increment.

**Not yet implemented, and it is the next increment:** rotation.  This pins **monotonically on first use** and
stops at the budget, so a session that starts cold keeps a suboptimal set; rotation needs
(i) a demotion trigger at the cache policy's boundary (its decay period already exists), (ii) `munlock` on
eviction, and (iii) — the reason it is not a five-line change — the **device-set event gate**: a locked range
must not be released while any device's copy is still reading it, which under `-sm tensor` is a set of devices,
not one stream (the campaign's P6 rule, and the same hazard the closed window campaign measured: unregistering
a range a queued copy still reads returns success and is not safe).

## 7. P2 increment 2 — the pinned set is seeded from the heat prior (2026-11-03)

`patches/0003` (98 lines, same branch, on top of `0002`):

```
MOE_HOST_PUSH_MLOCK_SEED=0     disable seeding (first-use only) -- kill switch / A/B arm
```

At sizing, immediately after the arena's own `apply_heat_seed_locked()`, `push_mlock_seed_locked()` pins the
mapping's top-ranked experts **in the same order the arena seeds itself** (`heat_rank_vec`, the decode-band
ranking), until the budget is exhausted.  So:

- **both host tiers agree on the ranking** — the arena's slots and the pinned set are the same hot set, which
  is what makes `push` an extension of the arena rather than a separate cache;
- the set is **right from token 1** instead of being whatever the router touched first (the monotonic
  first-use ordering that increment 1 shipped);
- **no prior → no seeding**, and first-use admission fills the set exactly as before (a first run, or
  `MOE_EXPERT_HEATMAP=0`);
- the graceful refusal path is shared: seeding stops at `RLIMIT_MEMLOCK` with the same single message, and a
  partly-seeded set is a normal state.

**Measured** (35B-A3B, 1 GPU, `-ncmoe 40 --host-experts pool`, forced misses, budget 4096 MiB, as root, with a
prior from earlier runs on the box):

```
pinned working set 4095 MiB in 6769 experts (budget 4096 MiB, seeded=6013 first_use=756, refusals=0)
```

`seeded=6013` from the prior, `first_use=756` for tables that had no usable prior entry — the designed
mixture.  And the invariant that matters: **the pinned set is a performance artifact and never a result** —
text is byte-identical with seeding on and off (`sha=0cf4a62aa63f`), `check-gpu-faults` clean, and t/s is equal
(48.6 vs 48.7) on this box, where the whole set is page-cache hot anyway.  The seed's value is on a
cold / not-fitting arm, which is the next measurement.

**Still owed:** rotation by decay (demote + `munlock`, gated on a device set), then the 122B cold arm, then the
7.14 correctness gate, then the A/B with the pool actually built.
