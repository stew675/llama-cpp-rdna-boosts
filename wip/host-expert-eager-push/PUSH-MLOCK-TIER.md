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

## 8. P2 increment 3 — the device-set eviction gate, and rotation (2026-11-03)

`patches/0004` (293 lines, same branch, on top of `0003`):

```
MOE_HOST_PUSH_MLOCK_ROTATE=0        never release (pin monotonically)
MOE_HOST_PUSH_MLOCK_ROTATE_MAX=8    experts released per sweep
MOE_HOST_PUSH_MLOCK_PERIOD_MS=2000  the sweep cadence
```

**The gate.**  A pinned range may be `munlock`ed only when **no device can still be reading it**.  Under
`-sm tensor` the same host pages are read by several devices (each `table_t` is a device's slice of one host
tensor), so "finished" is a **set of streams**.  Implementation: one boundary event per device per op, recorded
on that device's stream at the host hook; a per-expert epoch of its last read; release only when the read is at
least one full boundary behind *and* every recorded event for it (and later ones in the 4-deep ring) has
completed.  Ring reuse is safe in the direction that matters (a newer event on the same stream completing
implies the older read did), and a missing/stale slot is treated conservatively.  The device set is cached per
table (the sweep must not rescan `g_tables` per expert).

**Rotation.**  Releases only what the **policy** says is cold — not arena-resident *and* its decayed count is
0 — at most `ROTATE_MAX` experts per sweep, on a wall-clock cadence.

### Three lessons, all measured, all now encoded

1. **The staleness signal must be the policy's, not "was it filled lately."**  The arena absorbs most reads, so
   a hot-but-arena-resident expert looks unfilled; the first cut released those and the pinned set **collapsed
   to 33 experts, generation 1.0 t/s** (30 419 releases over 500 tokens).
2. **The cadence must not be derived from an op/table count.**  The table set grows lazily, so "every
   `g_period` boundaries" fired every 1-2 tables and turned a periodic sweep into a continual one: **5.7 then
   9.2 t/s** (156 ms/token).  Wall clock fixed it.
3. **`mlock`/`munlock` on populated page-cache pages is ~ms/MiB**, far above the 48.7 µs/MiB measured on a
   sparse scratch file — so the pinned set must be **near-static**: 4 232 releases over 500 tokens was
   unaffordable, 16 is free.

**Measured after the fixes** (35B-A3B, 1 GPU, `-ncmoe 40`, forced misses, budget 512 MiB, 500 tokens, root):

| | generation | pinned set | churn |
|---|---|---|---|
rotation off | 47.9-48.9 t/s | 511 MiB / 838 experts | — |
**rotation on (2 s, max 8)** | **47.9 t/s** | **511 MiB / 837 experts** | `seeded=725 first_use=128 released=16 release_denied=1` over 17 037 boundaries |

So the gate and the rotation are free at this cadence, and the set stays stable.  `release_denied=1` is weak
evidence that the gate *denies* under contention, though — the real test is a **multi-device** arm, where the
device set is > 1 and an in-flight read on device 1 must block a release decided against device 0.  That is the
next measurement.

## 8b. Published for the multi-GPU gate test (2026-11-03)

**Update (P3, 2026-11-03): the branches above are on the r208 chain.**  `rdna-boosts` is now **r209**
(`27271aa91`, not `efaa60db0`), so re-cut the four commits with
`git rebase --onto 27271aa91 efaa60db0 origin/p2-pinned-working-set` — 4/4, **no conflicts** (389
insertions, one file) — which is branch **`p2-eager-push-r209`** on the fork.  The gate below is reachable
**only on `-sm layer` or 1 GPU** (a tensor split never reaches the hook: `P3-MULTIGPU-RESULTS.md` §5), and
the "one message" diagnostics are now once **per process** — before that, rotation re-opened the budget and
re-printed them *inside* the generated text on a `2>&1` capture, which made a byte-identical arm hash
differently (`P3-MULTIGPU-RESULTS.md` §1b).

The experiment branches are on the **personal fork** (`origin` = `git@github.com:stew675/llama.cpp.git`), so a
multi-GPU box can check out the state directly instead of re-applying patches.  **`rdna-boosts` is untouched**
(still `efaa60db0`); these are experiment branches only:

| branch | content |
|---|---|
`p1-push-src` @ `be83da8f0` | `0001` — the push transport switch |
`p2-pinned-working-set` @ `f131cef2f` | `0001`-`0004` — + working set, heat seed, device-set gate + rotation |

```bash
git fetch origin && git checkout p2-pinned-working-set
~/bin/build-llama-rocm            # ~/bin/build-llama-rocm-714 for the older ROCm line
cd <delivery>/wip/host-expert-eager-push
SUDO=1 ./p2-gate-3gpu.sh 0,1,2 500     # the gate test; see the script header for PASS/FAIL
```

The script runs the rotate arm and the rotation-off control, prints the pinned-set report (`released`,
`release_denied`), the text hash and t/s for each, and runs `check-gpu-faults.sh` on both logs.

## 8c. First 3-GPU run (2026-11-03 16:41, control arm only) — what it proved, and what it did not

The log (`~/gate3-20261103-164125-norotate.log`) is the **rotation-off control** on the 3-GPU box, built from
*that box's* tree (`build 11907 (3cecf3bd8)` — not this repo's history) with `0001`-`0004` applied on top.

**Confirmed on 3 GPUs:** the push transport (35 806 fills DMA'd from the mapping), the pinned working set
(**511 MiB in 907 experts, seeded=664 from the heat prior, first_use=243, refusals=0**), the rotation kill
switch (`released=0 release_denied=0`), and no faults or aborts (`check-gpu-faults` clean; the one `abort`
string is the benign `--fit` message).

**Not proved — the gate.** The arm ran as **llama-server** (a single ~6-token request: `over ~6 tokens`,
`over 3 boundaries`), so the boundary counter never advanced; and the `rotate` arm was not run at all.  The
gate test needs a real decode run of both arms.

**And a valuable surprise: the pool is LIVE on that box** —
`host pool (L2): 61 tensors, 7808 slots, **4712.0 MiB pinned**; h=0.5239, fills=12817`, with
`pool fills: 12817 mapping memcpys 480 ms, 7.56 GiB (7742 MiB/s effective)`.  On one GPU it reported
`0 tensors, 0 slots` in every arm, which is why the earlier "push vs pool" comparisons were really against
the **bounce** path.  So the multi-GPU box is where the real comparator lives — and the numbers make the
campaign's case for it: **4.7 GiB of pinned pool to delete, against a 512 MiB pinned set that replaces it.**

The script now warns when the resolved binary is really `llama-server` and when the run decoded fewer than
`GATE_MIN_TOKENS` (default 100) tokens, so a vacuous arm cannot be mistaken for a result again.

**Two corrections to the reading above, from the same-day P3 runs on the primary box** (real `llama-cli`,
`--ignore-eos -n 500`, both arms) — recorded here rather than by editing the paragraph:

1. **`over ~N tokens` is not the decode length, and `srv` log lines do not mean llama-server.**  `over ~N
   tokens` is the cache's traffic-accounting counter on the `expert H2D traffic` line and it reads `~6` (or
   `~3`) on a full 500-token decode; and in this tree `llama-cli` is built on the server implementation, so it
   emits `srv`/`slot print_timing` lines of its own — the "ran as llama-server" reading came from those two
   strings, and both are normal here.  The harness now decides from the **binary** (basename must be
   `llama-cli`) and from `slot print_timing ... eval time = ... / N tokens` (500 on these arms), and fails the
   arm below `GATE_MIN_TOKENS`.
2. **A real 500-token run under `-sm tensor` still reports `over 3 boundaries`**, so a short request was not
   the (only) reason the counter did not advance.  The cause is structural: on a tensor split the whole graph
   is one Meta split, the routing ids are computed inside it, and the issue-#129 guard stands the
   host-readback branch down — `moe_cache_update_host`, where increment 3 records its per-device boundary,
   never runs (P3-MULTIGPU-RESULTS.md §5).  The counterexample is the same build on `-sm layer`:
   **60 357** boundaries, `released=40 release_denied=528`.

## 9. Additive by construction (the pool path is not touched)

The whole diff against the release tip is **189 insertions, 0 deletions**, and every line is behind an
`MOE_HOST_PUSH_*` variable that defaults **off**:

| hook | guard |
|---|---|
`expert_host_src`'s mapping branch | `MOE_HOST_PUSH_SRC=mapping` (unset → the pool/bounce path exactly as before) |
`push_mlock_expert` | only reachable from that branch |
`push_gate_boundary_locked` in `moe_cache_update_host` | `push_mlock_on()` (budget > 0) |
`push_mlock_seed_locked` at sizing | budget > 0 (early return otherwise) |
every log line | budget > 0 / mapping mode |

Verified at runtime: a default-arm run with the pool configured logs **zero** `P1-PUSH` lines.  The parallel
`wip/host-pool-speed/` campaign therefore owns the pool path untouched, and nothing here changes a default.
