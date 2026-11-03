# wip/host-expert-eager-push — never device-pull: the host pushes, eagerly (OPEN)

> **P1 STATUS (2026-11-03): correctness GREEN on ROCm 10.1, performance leg confounded — see
> [`P1-RESULTS.md`](P1-RESULTS.md).**  `patches/0001` adds `MOE_HOST_PUSH_SRC=mapping` (default off): on
> 35B-A3B with forced misses, the mapping arm made **33 393 direct pageable DMA fills** with **byte-identical
> text** (`sha=0cf4a62aa63f`), an identical access census, and **no GPU faults** — the 1-D tearing claim is
> refuted in the real path.  Performance found two things instead: where the page cache is hot the `memcpy`
> is **already hidden behind the DMA** (102.9 vs 102.6 t/s warm/warm), and on the 122B arm the mapping source
> **silently bypasses the pool's `O_DIRECT` decision**, so that 5.7x is a page-cache-vs-DIO comparison, not a
> copy win.  **`push` is therefore not a replacement for `pool` but the missing middle rung** of a memory
> ladder: `pinned` (locked) → **`push` (reclaimable page cache)** → `pool` (streams from disk).  P2 must first
> make the mapping source honour the DIO decision, and the 7.14 correctness gate is still owed.

> **THE DESIGN SHAPE (2026-11-03, after P1 and the `mlock` test): the pool splits into transport and residency,
> and only residency is worth keeping.**  Transport = the direct pageable push (`patches/0001`, measured equal
> to memcpy+pinned H2D when warm).  Residency = a **pinned working set `mlock`'d over the mapping itself**,
> bounded by the FRAC budget and rotated by the arena's own policy — measured **releasable** (`mlock` holds
> through 67 GiB of pressure; `MADV_PAGEOUT` is refused while locked and succeeds after `munlock`, dropping the
> window to 0/262144; 48.7 µs/MiB in, 0.81 µs/MiB out).  So the pool's **RAM duplicate** disappears while its
> *guarantee* is kept, and `push` becomes a real extension of the arena.  See
> [`PUSH-MLOCK-TIER.md`](PUSH-MLOCK-TIER.md).
>
> **P2 increment 1 is implemented and measured** (`patches/0002`, `MOE_HOST_PUSH_MLOCK_MIB=N`): first-use
> pinning over the mapping with a one-byte per-expert mask and a process-wide budget; **refusal is not fatal**
> (one message naming `LimitMEMLOCK=`/`CAP_IPC_LOCK` and the actual limit, keep what was pinned, carry on from
> the page cache).  Measured: 7 MiB / 11 experts unprivileged with one refusal message and a correct run;
> **4095 MiB / 6 744 experts as root, zero refusals**; text byte-identical with and without pinning
> (`sha=0cf4a62aa63f`); no GPU faults.  **Rotation is not implemented** — it pins monotonically on first use,
> and rotation needs `munlock` on eviction gated on a *device set* under `-sm tensor`.
>
> **P2 increment 2** (`patches/0003`): the pinned set is **seeded from the heat prior** at sizing, in the same
> ranked order the arena seeds itself (`MOE_HOST_PUSH_MLOCK_SEED=0` disables it) — measured `seeded=6013
> first_use=756` of 6 769 pinned experts on 35B-A3B, text byte-identical with and without seeding
> (`sha=0cf4a62aa63f`), no faults.  Both host tiers now agree on the hot set, which is what makes `push` an
> extension of the arena rather than a second cache.
>
> **P2 increment 3** (`patches/0004`): the **device-set eviction gate** (a pinned range is released only when no
> device can still be reading it — one boundary event per device per op, a per-expert read epoch, a 4-deep ring)
> and **rotation** (policy-cold experts only, ≤ 8 per sweep, wall-clock cadence).  Measured free after three
> fixes: **47.9 t/s with rotation vs 47.9-48.9 without**, pinned set stable at 511 MiB / 837 experts,
> `released=16 release_denied=1`.  Two traps encoded in the design now: the staleness signal must be the
> *policy's* (not the fill count — that collapsed the set to 33 experts at 1.0 t/s), and the cadence must be
> wall-clock (an op/table-derived one made the sweep continual: 156 ms/token).
>
> **Additive by construction**: the entire diff against the release tip is 189 insertions, 0 deletions, all
> behind default-off `MOE_HOST_PUSH_*` variables; a default-arm run logs zero `P1-PUSH` lines and the
> pool/bounce path that `wip/host-pool-speed/` owns is untouched.

**Seeded 2026-11-03 at the maintainer's call.  Nothing here is implemented, and nothing here is in `patches/`
or gets applied to the fork** (`AGENTS.md`, WIP rules: a campaign is a handover, and promotion is a separate
decision).

**The charter, in the maintainer's words:**

- *"Perhaps we'd been going about it the wrong way all along by asking for the GPUs to pull when the full set
  isn't resident."*  **The alternative MUST be host-eager-push, and never device-pull.**
- **`pinned` stays the mode of choice when the whole expert set fits in RAM** (today's default, unchanged).
- The push mode is what runs when it does not fit — where `pool` is the only option today.
- The CLI surface reuses the resurrected **`--host-experts window`** intent: a bounded residency window over
  the experts, whose contents the host pushes.  `window` is accepted as an alias for `push`.

## 1. The reframing, and why it is the right one

The GPU is **never** given a host address to read.  Every host-resident expert byte reaches VRAM through a copy
the host (or its copy engine) issues, ordered by an event, with a known destination slot.  That was already the
delivery's behaviour; this campaign makes it the **rule**, deletes the in-place-read design space, and asks the
question the closed campaign left open: *how cheap can the push be when it reads the page cache directly?*

Why this shape is now the interesting one:

| | device-pull (refuted) | pool (today) | **eager push (this campaign)** |
|---|---|---|---|
source | kernel reads host RAM in place | memcpy into a pinned slot, then CE | **CE straight from the page cache** |
needs a hold | yes (**unreleasable** — closed campaign) | the slot is the hold, and it is reusable | **none** |
host RAM beyond VRAM | whole registered set | the pool budget | **the page cache (reclaimable)** |
fill cost, hot | 0 (if it worked) | 137 µs / 1.66 MiB | **66-93 µs** (measured) |
fill cost, cold | fatal | ~3.5 ms / 1.66 MiB | **1.32 ms** (measured) |
wire traffic | 1x | 1x | 1x |

The numbers are from `archive/work/host-expert-locked-window/MEASUREMENTS.md` §5 (arm `dmapage`), measured on
1x R9700, ROCm 10.1: **the memcpy into a pinned slot is the entire gap**, and removing it is worth ~2x hot and
~2.7x cold without introducing any new residency primitive.

## 2. The one thing standing in the way, and it is a claim without a record

`ggml-cuda.cu:6255` (and the prior campaign's `PROTOTYPE.md:47`) says a pageable `cudaMemcpyAsync` "is staged
through a shared internal bounce and **tears under back-to-back use** (measured: nondeterministic greedy
text)".  That is why `moe_cache_upload_pageable()` exists and why the pool's copy is load-bearing today.

**It does not reproduce** for a 1-D push on this stack: 8-64 copies in flight (13-128 MiB), 2-4 streams, from
one host thread, every copy verified against its own source, plain pageable / `mlock`ed / registered / pinned —
**0 torn in every arm**, 20-100 iterations each.  The 2-D path (which the warning actually names) is 4-5x
slower but also clean.  The claim has **no dated record, no ROCm line and no reproduction** anywhere in the
repo; the delivery ships **two** ROCm lines (7.14.1 and 10.1), and the async-input *ring* is a third possible
origin (a shared **destination**, not a source bounce).

So the campaign's first deliverable is **reproduce-or-refute in the real path**, not a design.

## 3. Shape of the mode

```
--host-experts <pinned|auto|pool|push>        # `window` = alias for `push`
```

| mode | host master | fill | when |
|---|---|---|---|
`pinned` (default) | full `ROCm_Host` copy | direct H2D from pinned (no staging) | **the set fits in RAM** — still the best mode, unchanged |
`pool` | read-only mapping, no master | memcpy into a bounded pinned pool, then CE | the set does not fit; today's answer |
**`push`** (new) | read-only mapping (`ROCm_HostMap`) | **CE straight from the page cache, issued early, event-ordered** | the set does not fit; deletes the memcpy |
`auto` | — | picks `pinned` when `host_expert_bytes` fits `MemAvailable` with margin, else `push` | the default once gated |

`auto` gaining the predicate is the campaign's product-level goal: *nobody should have to know which mode their
box needs*.  `pool` is kept because it is the fallback and the control arm, and because PR #267's
`--host-experts mmap` shape must not be resurrected to serve a case this covers.

## 4. The mechanism

`DESIGN.md` is normative: the push pipeline (source → staging → destination slot → event), the invariants
(**P1 never device-pull**, P2 eager, P3 slot ownership, P4 no whole-set hold, P5 counted fallback, P6
split-portable, P7 byte-identical), the open ordering questions, and the rejected alternatives.

## 5. Plan

| phase | what | gate before advancing |
|---|---|---|
**P0** | inherited: the archived probes' numbers (fill costs, the tear matrix, the grant/hold facts) | — |
**P1** | **reproduce-or-refute the tearing claim in the real path**: a switch that makes the fill bypass the pinned bounce and DMA from the mapping (`MOE_HOST_PUSH_SRC=pinned|mapping`), then the correctness gate: same-seed greedy vs `pinned`, MTP four-axis, a back-to-back fill stress, `check-gpu-faults.sh`, on **10.1 and 7.14**, 1 GPU **and** the `-sm tensor` pair | it is either clean on both lines or the claim is confirmed with a dated record — the campaign stops at P1 if it tears |
**P2** | **eagerness**: issue op N+1's pushes while op N computes (the host already has the routing), slot ownership + event ordering, bounded lookahead | push ≥ pool on decode and prefill, 1 GPU, 122B `-ncmoe 48`, decode ≥ 2000 tokens |
**P3** | the not-fitting arm: 64 GiB emulated by `memory.max`, the disk tier on the critical path, `RAM_PCT`/DIO interplay | text byte-identical, no faults, and the charter's acceptance criterion (below) |
**P4** | the split: `-sm tensor` on the x8 pair, then 3 GPUs | the tensor-split arm is an acceptance gate, not a follow-up |

**Acceptance (the charter's existing criterion, carried over):** `push` must be at least as fast as `pool` on
both bands, and must leave PR #267's `--host-experts mmap` shape with no performance advantage, on a config
where `pinned` cannot hold the set.

## 6. Gates

- **Correctness first, everywhere**: same-seed greedy text byte-identical to `pinned` and `pool`; MTP four-axis
  (`benchmarks/mtp-adaptive-methodology.md`); a **back-to-back fill stress** (the only thing that would expose a
  torn fill); `scripts/check-gpu-faults.sh`; `scripts/gate-prefill-logits.sh` for any prefill-path change.
  Text gates alone are weak here — the failure mode being tested is silent corruption, so the stress matters.
- `scripts/gate-per-block-build.sh` and `scripts/validate-set.sh` at promotion, as always.

## 7. What would kill it

- **The tearing reproduces in the real path** on one or both lines.  Then the copy is load-bearing for a reason
  with a record, `pool` is the answer for the not-fitting case, and this campaign closes the same day (as its
  predecessor did).
- **Ordering cost swallows the win**: if the pushes must be serialized per device to be safe, the overlap that
  makes them cheap is gone and the measured 2x evaporates.  P1 must measure *overlapped* fills, not serialized
  ones.
- **The disk tier dominates** at the campaign's arm: a cold push is 1.32 ms/1.66 MiB, so if the miss set is
  large the fill path is I/O-bound and no transport change helps (`host-pool-speed` measured exactly this).
- **`-sm tensor` cannot be made safe**: a per-simple-device push queue that the Meta forward cannot reach is
  PR #266's corner again; the push must live where the existing `moe_cache_*` hooks live.

## 8. Instruments

- Reuse `archive/work/host-expert-locked-window/probe-locked-window.cpp` — it already carries the arms that
  matter (`dmapage`, `tear`, `tear2d`, `regcost`, `pin`, `evicted`) and `sim-locked-window.py` prices a policy
  against the measured constants.  **Do not rebuild them.**
- The delivery's own paths to touch in P1: `moe_cache_upload_pageable()` / `moe_cache_source_pageable()`
  (`ggml/src/ggml-cuda/moe-expert-cache.cu`), the fill site in `ggml-cuda.cu:6254`, and the prefill staging
  (`GGML_SCHED_STAGE`) for the batch band.
- The parallel `wip/host-pool-speed/` campaign owns the pool's own tuning and the same code region: the
  reconciliation point is the fill/cold-read hook, and neither campaign may rewrite it unilaterally.
