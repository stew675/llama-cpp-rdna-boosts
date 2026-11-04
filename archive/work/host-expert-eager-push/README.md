> **PROMOTED AND ARCHIVED (2026-11-04, `v16-6050b467c-r211`).**  The host-expert tier this campaign built is
> **in the delivery** — folded into **block 09** (blocks 10-15 re-cut on top), repaired so every block tree
> builds, and released as r211 (r210 was never tagged).  What shipped: `-lme auto|N` as the budget over ONE
> pinned per-expert pool, laid out as the arena's **complement first** and then the arena-resident experts in
> heat order while capacity lasts; a pool-resident expert read **in place** by any device
> (`MOE_HOST_POOL_COLD_MULTIDEV`); and, in r211, `pool_covers_table()` — a **covering** pool is exempt from the
> eager take-over's arena bound, which is what fixes `llama-batched-bench -npl 8` (49.2 -> 270.3 tg).
> **Dropped, all measured**: the direct pageable push and its `mlock` working set, in-place `hipHostRegister`,
> the tier/attribution probes, and the eager read-ahead back-fill.
> Read in this order: [`P5-RESULTS.md`](P5-RESULTS.md) (the tier and its numbers) ->
> [`P6-POOL-REGION.md`](P6-POOL-REGION.md) (§8 complement vs mirror, §9 the dropped read-ahead, §10 the surplus
> rule) -> `WORKLOG.md` 261104.001 (r210: the fold) and 261104.002 (r211: the guard fix, the fold repair, every
> gate).  `INVARIANTS.md` r210/r211 is the "do not re-derive" list.  Open follow-ons: `TODO` #141, #142.
> **This campaign subsumed and obsoleted two `wip/` campaigns, both retired on 2026-11-04:**
> [`archive/work/host-pool-speed/`](../host-pool-speed/RESOLUTION.md) (the `--host-experts pool` mode's own
> speed campaign, whose remaining tier question `-lme` answers, and whose `push` transport is the one dropped
> here — so its `TODO` #137/#138 were retired with it) and
> [`archive/work/no-xnack-and-avalon/`](../no-xnack-and-avalon/RESOLUTION.md) (the no-XNACK analysis, whose
> answer this tier embodies).

# wip/host-expert-eager-push — never device-pull: the host pushes, eagerly (ROOT CAUSE FOUND)

> **P8 STATUS (2026-11-04): the pool is the ONE mechanism, and it spends spare capacity mirroring the arena** —
> `P6-POOL-REGION.md` §10, patch `patches/0013`, branch commit `d87713e36`.  The layout rule is now *one* rule for
> every budget: the **complement first** (every expert the arena does not hold, in heat rank order), **then the
> arena-resident experts while capacity remains**.  `slots < |complement|` is the historical bounded complement;
> `slots > |complement|` keeps the hottest arena residents too — a full mirror when the budget covers the tensor,
> i.e. "basically pinned".  The loader therefore no longer routes `-lme N >= 90 %` to the whole-tensor tier:
> `-lme auto` (fit) spends the whole set on the pool, `-lme N` spends N GiB, and `--host-experts pinned` /
> `MOE_HOST_TIER=tensor` are the explicit escape hatch.  Measured 2 devices, byte-identical text, faults clean:
> **`auto` = 1317.6 prefill / 70.4 decode (n=300) / 82.5 (n=2000), pool hit rate 1.0000, 18 662 MiB pinned, zero
> evictions** against a pinned master's 1105.4 / 71.9 / 82.9 — so **+19 % prefill for −0.5…−2 % decode at the same
> pinned RAM**; the intermediate band gains where the surplus appears (`-lme 16`: 65.6 → 66.5) and a genuine
> sub-complement budget is unchanged (`-lme 8`: 54.1).  The read-ahead back-fill was **dropped** (measured a loss in
> both bands, and a needless synchronisation surface; §9 keeps the numbers, commit `a7dfe8871` the code).
>
> **P6b STATUS (2026-11-04): a full-coverage pool is a MIRROR, not the arena's complement — and it now BEATS a
> pinned master at prefill** (patch `patches/0012`, branch commit `2f8947bbd`, record `P6-POOL-REGION.md` §8).
> The pool IS fully populated before the prompt, but two paths then *un-load* experts from it purely because they
> also exist in VRAM (`fill_slot_from_host` retires the pool slot; `pool_rerank_from_heat_locked` frees it —
> "an expert already in the arena needs no pool slot"), 13 440 of 30 720 slots at full coverage.  When the arena
> evicts one of those experts the pool peek misses, `pool_fill_empty_locked` has no free slot, and the fill falls
> through to the **bounce ring: a synchronous 288 KiB `memcpy` from the model mapping** plus an H2D — 20 116 of
> them in the first 300 decode tokens, invisible to the fill probe (which times only the `cudaMemcpyAsync`).
> With the mirror rule the bounce goes to **zero** and, at 2 devices, decode goes 51.7 → **64.5** (n=50),
> 63.9 → **70.4** (n=300), 69.7 → **73.6** (n=500), 80.8 → **82.6** (n=2000) against a pinned master's
> 66.6 / 71.9 / 75.5 / 82.9, while prefill goes 1166.4 → **1315.9 against the pinned 1105.4 (+19 %)** for the same
> 18 662 MiB of pinned RAM.  Text byte-identical everywhere, faults clean, bounded pools unchanged by
> construction (FRAC 50 56.5, `-lme 8` 54.0), 1 device 52.1, 3 devices `-sm layer` 48.1.  **Open policy call:**
> the fit case could therefore be the pool-as-mirror instead of P5's whole-tensor tier (+19 % prefill, ~-2 %
> decode at n≤500) — one mechanism, and the per-tensor tier disappears.
>
> **P6 STATUS (2026-11-04): the expert-indexed region IS the pool, and it now works on a multi-device split —
> [`P6-POOL-REGION.md`](P6-POOL-REGION.md)** (branch commit `72bc81b88`, patch `patches/0011`).  `TODO` #139 is
> closed in the campaign: the pool is a pinned per-host-tensor region addressed by (expert -> slot) with
> `slot_state` as the presence bitmap, so the design of record was already built — what it lacked was the right
> to be a **cold-read source away from one GPU** (`device_count == 1 && src_off == 0` in both
> `pool_cold_slot_locked` and `moe_cache_get_cold`).  Both restrictions are geometry accidents (the alias is
> already the cross-device fill source; a slice offset is already folded into the cold base for a pinned master),
> and they now live in ONE predicate both sides use — if they disagreed the host would emit a cold id the kernel
> resolves from the wrong base, a silently wrong matmul.  `MOE_HOST_POOL_COLD_MULTIDEV=0` is the kill switch.
> **Measured +11.5 % decode at `-lme 8` (48.9 -> 54.5) and +10 % for the shipped not-fitting pool (51.8 -> 57.0),
> byte-identically**, and a **fully resident pool now prefills FASTER than a pinned master (1166.4 against
> 1105.4)** — a first for the campaign — though its decode still trails (63.9 against 71.9).  With that measured,
> `-lme N` spends the budget per EXPERT below 90 % of the host expert set (the pool, heat-ranked) and keeps P5's
> whole-tensor pinned-master tier at or above it (the fit case, where only a pinned *master* gets the
> single-launch staging branch): 8 GiB 54.5/50.1, 12 GiB 61.6/56.8, 16 GiB 65.6/65.7, 17 GiB 67.0/68.2,
> 100 % 63.9/71.9 — pool/tensors.  `MOE_HOST_TIER=pool|tensor` forces either shape for an A/B.  Validated on 1,
> 2 and 3 devices (`-sm layer` and the `#135` degenerate `-sm tensor`), n=500 forced generation, and a second
> arch (Flash-Next), all byte-identical where the arms must match and all fault-free.
>
> **P5 STATUS (2026-11-04): the unified host-expert tier is IMPLEMENTED and MEASURED, awaiting the
> maintainer's go/no-go before promotion — the record is [`P5-RESULTS.md`](P5-RESULTS.md)** (branch
> `p5-host-expert-tier`, patch `patches/0010`, 26 arms in `runs/p3-*-p5-*`).  `-lme` is now the **budget for
> the pinned host-master tier**, resolved at model load before any tensor is placed: `auto` = pin the whole
> set when it fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable`, else no pinned master (the mapping + the
> bounded pool, i.e. today's `pool`); `pinned` = the whole set; `-lme N` = N GiB of **whole tensors in
> descending hotness** (the persisted heat prior ranks them).  A wholly pinned tensor IS the region (the
> master is expert-indexed by stride), so the fit case is bit-for-bit today's `pinned`: **1105.4 prefill /
> 71.9 decode at 2 devices, 645.2 / 53.4 at 1**, byte-identical text (`sha=231512fc77e5` / `4ae33f7bd65c`),
> no GPU faults, and the not-fitting band unchanged (713.7 / 51.8).  Two findings change the plan's shape:
> **(1) at equal RAM a partial budget belongs to the POOL, not to per-tensor pinning** — 8 064 MiB of pinned
> whole tensors measures 505.1 prefill / 50.1 decode against 9 331 MiB of per-EXPERT pool at 713.7 / 51.8
> (decode a tie, prefill 29 % worse), because routing is spread across every layer, so no whole tensor is
> hot; `auto` is therefore a BINARY decision and a per-expert *region* (the plan's step b2) is deferred to
> `TODO` #139.  **(2) two silent traps**: the `-ncmoe` tail bound is `hparams.n_layer()` (not
> `n_layer_all`, which would plan the MTP head and miss trunk layer 0), and `moe_cache_set_host_pool()` must
> re-resolve `g_pool_enabled` because the loader decides after `parse_env()` (without it the not-fitting band
> silently lost the pool: 37.4 vs 51.8 t/s).  `--host-experts pool` is now a deprecated shorthand for
> `auto`; nothing is promoted into `patches/` yet.

> **P4 STATUS (2026-11-04): the `-sm tensor` defect is DIAGNOSED and a fix is MEASURED — the record is
> [`P4-ROOT-CAUSE.md`](P4-ROOT-CAUSE.md).**  The loss is the driver's **pageable staging, a per-CALL cost**:
> in situ at 2 devices the mapping arm spends **12.82 s of a 27.7 s run** inside **90 636 1-D pageable fill
> copies at 141 µs/copy (99.8 % of the H2D time)**, while the same arm's 2-D/sliced fills — served from the
> pinned bounce ring — are **1.3 µs/copy**.  A tensor split halves the per-copy size and doubles the call
> count, so the fixed staging cost is paid twice per byte; that is the whole defect, and it is why 1 device
> and `-sm layer` were at parity.  **The fix is `hipHostRegister` over the read-only model mapping, in
> place** (`patches/0007`, `MOE_HOST_PUSH_REGISTER_MIB=N`): no copy, no duplicate, native transfer for every
> geometry — **0.3-0.5 µs/copy**, and 2 devices go from **21.5 to 71.6 t/s** against the pool's **51.5**
> (byte-identical text `sha=231512fc77e5`); 1 GPU 53.5 vs 49.0, 3 GPU `-sm layer` 54.3 vs 50.5, 3 GPU tensor
> 20.5 vs 19.4 (that arm is `TODO` #135's degraded config).  Two measured caveats decide the shape: the pin
> is **not releasable** — `hipHostUnregister` leaves the pages pinned while the mapping lives (`hipFreeHost`
> rejects a registered range; `unregister + munmap` *does* release), so the in-place form is a **permanent**
> pin usable as an all-or-nothing fit-in-RAM mode, and the **releasable** tier is a private registered
> window** over the same page-cache pages.  **Reclaimability is now PROVEN, not assumed** (`probe-window.cpp`,
> the requirement for the > RAM band): `hipHostUnregister` alone does not release the pin on ANY flag set, but
> `unregister + munmap` does, and a sliding registered window over a **private** mapping of the model file
> reclaimed **40/40 cycles** (`MemAvailable` flat, no accumulation) while a long-lived model mapping over the
> same pages stayed fully reclaimable with identical data — so the rule is **never register the model's own
> mapping; register a private window and release it by `munmap`**.  **Over-commitment tested at scale**
> (`probe-overcommit.cpp`, 100 GiB file vs 91 GiB RAM, 20 GiB registered windows, 3 loops, 300 GiB read, in
> one process): every window released and **verified droppable**, registration 31-33 ms per 20 GiB, DMA 28.8
> GB/s, `MemAvailable` and swap **flat** — no leak, no thrash, no lockup, in both read modes (`mmap` and
> `O_DIRECT`).  Releasing a pin does **not** discard the page cache: a resident 8 GiB window re-reads in
> 13-21 ms and re-registers in ~13 ms (72 GiB in 3.3 s over 3 loops), and with a **50 GiB region pinned
> permanently** a 5 GiB cycling set still climbs from 40 % to 87 % cache-served over five loops (the no-pin
> control behaves the same — the convergence is the host's cache policy, not the pin).  Two host caveats
> recorded: `hipHostRegister` of a single 50 GiB range works and faults the pages in itself (register
> **first**, not after a read), and on this box `/proc/meminfo` `Cached`/`MemAvailable` do not account these
> pages usefully — judge residency by `read_bytes` and a second pass.  **And `--host-experts pinned` is NOT this
> mechanism**: it is `cudaMallocHost` of a *second full copy* of the host expert set (≤4 GiB per buffer, no
> limit but RAM, silent pageable fallback on failure), i.e. the same transport at the wrong RAM cost — measured
> 71.6-72.1 t/s with the duplicate against **71.5-71.9 t/s for the registered mapping with no duplicate**, and
> 36.0 t/s when the pinned allocation falls back to pageable.  The in-place `hipHostRegister` call exists in
> the tree (`ggml_backend_cuda_register_host_buffer`) but nothing calls it.  See `P4-ROOT-CAUSE.md` §5c.  Also recorded: the destination (plain `hipMalloc` vs a
> VMM/slab slot) is **irrelevant** (HANDOVER experiment 1 closed negative), and on this stack a pageable
> **1-D** copy is 2-3x *slower* than a pageable 2-D copy of the same bytes (the opposite of the earlier
> single-GPU observation; the transferable conclusion is "stop it being pageable", not "reshape the copy").

> **P3 STATUS (2026-11-03): the transport WORKS, except under `-sm tensor` — now diagnosed in P4.**  The
> pre-diagnosis record is [`P3-MULTIGPU-RESULTS.md`](P3-MULTIGPU-RESULTS.md); the ruling-out list and the
> trap list are in [`HANDOVER.md`](HANDOVER.md).  Summary: **correctness GREEN** on 1, 2 and 3 devices
> (`mapping` and the pool produce byte-identical greedy text over a forced 500-token generation, up to
> **65 260** direct pageable fills, identical access census, `check-gpu-faults` clean — the 1-D tearing
> claim refuted a third time, now multi-device; 7.14 still owed); decode at **parity** on 1 device
> (55.1 vs 53.6 t/s) and on a 3-device **`-sm layer`** split (55.6 vs 53.7), and the 2-device tensor split
> **52.5 -> 26.1** — which P4 attributes to the pageable staging per call, not to the split.  Also from P3,
> still true: the **device-set gate is INERT on a tensor split** (3 boundaries vs 60 357 on `-sm layer`),
> so P2's eagerness needs a boundary source that exists there.

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
