# host-expert-mirror (archived) - when the pool covers the set, stop staging a copy

> **CLOSED 2026-11-05 (no release, no delivery change) -> [`RESOLUTION.md`](RESOLUTION.md).**  The campaign's
> premise was refuted by measurement: on a mirror the prefill transport is the **device gather** (not the staging
> engine), it reads the pinned pool alias **in place at 33 GB/s** (at the link's ceiling), and it is **serialised**
> on the compute stream — so an in-place alias can only ever buy overlap.  What *was* recoverable is the
> **engine/stream** choice (`LEAD-pool-staging-gap.md`): per-expert copy-engine staging measures +12-13 % on one
> GPU at `-ub` 5120-7168, while the r114 single-launch staging kernel is the *worst* transport at every width and
> device count measured.  Nothing here is in `patches/`.  The text below is the original scoping note, kept for the
> reasoning (its §2 numbers are superseded by `spike-261105/SPIKE.md` and `LEAD-pool-staging-gap.md`).

**Status when written: scoping note with a measured target (2026-11-05).  Nothing implemented, nothing in
`patches/`, nothing applied to the fork.**

**What this is.**  A mirror pool holds every expert of every host table in pinned RAM.  The GPU still receives a
copy of the routed experts every ubatch, because the scheduler's host-weight upload is the only way an op
currently gets a weight tensor.  Brian's branch reads the registered mapping in place and does the same pass 8 %
faster.  This campaign is about closing that gap WITHOUT giving up the pinned, bounded, multi-device tier -
i.e. handing the op the pool's own device alias when, and only when, the pool covers the table.

## 1. What you inherit (do not re-derive)

* **The mirror predicate already exists.**  `pool_covers_table()`
  (`ggml/src/ggml-cuda/moe-expert-cache.cu`) is "pool slots cover the tensor AND every expert is pool-resident",
  and its comment states the property: a pool whose capacity covers the tensor has the pinned master's property,
  so coverage is a property of the budget, not of the run.  It is used today for the decode/verify completeness
  guard.  Do not invent a second predicate.
* **The in-place transport for ONE expert read already exists.**  `MOE_HOST_POOL_COLD` (default on) serves a
  pool-resident expert in place through the pool's device alias `base_dev`, and
  `MOE_HOST_POOL_COLD_MULTIDEV` (default on) removes the old `device_count == 1 && src_off == 0` restriction,
  because `base_dev` is a `Portable|Mapped` UVA alias and a tensor split's slice offset is folded into the cold
  base exactly as for a pinned master.  What does NOT exist is using it for a whole table / a whole op.
* **The mirror fill is one sequential stream and the pin is folded into it** (r215, `MOE_HOST_POOL_ALLOC=thp`):
  `AnonHugePages` is 48.97 GiB while running and 0 after teardown, `MemAvailable` returns to baseline.
  `MOE_HOST_POOL_ALLOC=portable` restores the r214 shape.  The alias exists under both.
* **A registered source is not automatically a prefill loss** (`INVARIANTS.md`, r215 item 7): the archived -22 %
  1-device prefill regression was a *pageable mapping registered in place*, whose fills became synchronous DMA
  ([`archive/work/host-expert-eager-push/P4-ROOT-CAUSE.md`](../host-expert-eager-push/P4-ROOT-CAUSE.md)
  §5d).  That precedent is the main risk here, and it is close enough that this campaign must re-measure rather
  than assume, in both directions.
* **Any new reader of a host expert master must go through `expert_host_src` (pool slot / per-table pinned
  bounce) or be added to `moe_cache_upload_pageable`** (`INVARIANTS.md`).  That is where a new transport plugs in.
* **The staging copy must stay batched** (`MOE_HOST_POOL_STAGE_BATCH`, r114) and the master-drop design pays one
  extra host memcpy per staged byte, which is why the copy is batched.  An in-place read does not pay that at
  all, which is the point.

## 2. Evidence (2026-11-05, mirror config on the `bench-56gb` rig)

| | decode | prefill 50k | prefill 179.5k | wall |
|---|---|---|---|---|
| `-lme 47` + `RAM_PCT=0`, `-ub 6144` | 78.5 t/s | 1941 t/s | 1802 t/s | 99.6 s |
| the same at `-ub 2048` | 80.0 | 1798 | 1565 | 114.7 s |
| `briansp2020/llama-cpp-rdna-boosts` (his box) | 82.7 / 81.7 | - | 1949 t/s | 104 s |

* 0 MiB of drive reads during prefill (`/proc/<pid>/io` delta over a 50k pass), so the tier is doing its job and
  the gap is not I/O.
* the `-ub` dependence is the tell.  At `-ub 6144` one ubatch routes 6144 x 10 draws over 512 experts, so
  essentially the WHOLE table is routed and the staging copy is 46.84 GiB per ubatch: 29 ubatches per 179.5k
  pass, about 1.36 TB over PCIe in 100 s (13.6 GB/s), on top of the reads a kernel would issue anyway.
* an in-place read does not move fewer bytes over PCIe; it moves them as kernel loads instead of as a bulk copy.
  So the win, if any, is "no VRAM write, no double buffer, no copy-engine sync", not "less traffic".  The 8 % is
  the size of that prize, and this campaign's first step is to find out how much of it is reachable at all.

> **2026-11-05: step 1 is DONE and steps 2-3 are RE-SCOPED.**  The spike is
> [`spike-261105/SPIKE.md`](spike-261105/SPIKE.md) and its follow-up lead is
> [`LEAD-pool-staging-gap.md`](LEAD-pool-staging-gap.md).  Headline: on a mirror the prefill transport is the
> **device gather** (not the staging engine), it is **serialised** on the compute stream and costs **34.9 %** of
> prefill GPU time at `-ub 6144` (55 % at `-ub 2048`) at the link's full 33 GB/s -- so the bytes are not
> recoverable and the in-place alias below is the wrong lever.  What *is* recoverable is the stream: per-expert
> copy-engine staging measures **+13 %** over the gather on one GPU at `-ub 6144` (and the r114 single-launch
> staging kernel is the *worst* of the three, because it starves the compute kernels).  Steps 3-5 below are kept
> for the reasoning, but the campaign's live work is now the lead, not `MOE_HOST_POOL_ALIAS`.

## 3. The work, in the order I would take it

1. **Spike, no code.**  Instrument what the prefill band copies today for a mirror table: staged bytes per
   ubatch, `D2D` from `base_dev` vs `H2D` (`MOE_HOST_POOL_H2D=0|1`), the time inside
   `moe_cache_upload_pageable()`, and the A/B pair `GGML_MOE_CACHE_STAGE=0|1` with `-ub 2048` and `6144` (at 2048
   the routed set is a fraction of the tables, so the copy is smaller and cheaper relative to compute).
   *Accept:* "staging bytes per pass" and "share of prefill wall time in the copy".  Either number can end the
   campaign before any code is written, and that is a good outcome.
2. **Prove a kernel read of the pool alias on this stack.**  The pool is pinned and mapped, so issue #116 (a
   *pageable* master faulting) does not apply by construction; prove it with a one-off test that reads a pool
   slot from a kernel (both alloc shapes, `thp` and `portable`).  *Accept:* passes, or records "no" with the
   error.
3. **Hand a covered table's op the alias.**  Reuse `pool_covers_table()` as the condition, and prefer the
   per-expert remap shape (the cold-id path the kernels already take) over aliasing `src0->data` wholesale: the
   latter puts a pool pointer into graph keys and fusion guards, and a rerank would then move bytes under a
   captured graph.  Behind `MOE_HOST_POOL_ALIAS=1` (default OFF, opt-in until measured).
4. **Measure on the rig.**  Mirror config, `-ub 6144` and 2048, alias on/off, plus `MOE_HOST_POOL_H2D` and
   `MOE_HOST_POOL_ALLOC` rows.  *Accept:* greedy output bit-identical to the staged path (this is a transport
   change, not a placement change); prefill within 3 % of 1949 t/s; decode not worse; `MOE_HOST_POOL_ALIAS=0`
   byte-identical to today.
5. **Only then a default.**  A bit-identical and faster transport is safe to default; if it only wins at large
   ub, keep it conditional and say so in `ENVIRONMENT.md`.
6. **`-sm tensor` with the pair.**  The pool is process-wide and shared by every device that owns tables of a
   host tensor, and `MOE_HOST_POOL_COLD_MULTIDEV` already reads residents in place from any device, so the
   multi-device shape is the SAME code with the slice offset folded in (`src_off`, `host_pitch`, `split_axis`,
   `slices`).  *Accept:* single GPU is the degenerate case, no second path.

## 4. What would kill it, and what to watch

* If a bulk `cudaMemcpyAsync` is simply more efficient per byte than per-access load/store over PCIe, the 8 % is
  not recoverable this way and the win has to come from `host-expert-split/` (a bigger VRAM arena) instead.
  Step 1 decides this cheaply.
* The -22 % precedent (`P4-ROOT-CAUSE.md` §5d) was a *registration* change that made fills synchronous.  An
  alias that keeps the fill path for misses must not touch the fill path for hits; measure the arm where the
  pool is partial as well, to prove the fallback still holds.
* Graph capture: a captured graph bakes whichever pointer the op read.  The per-expert remap is republished each
  band, which is why step 3 prefers it.  If the wholesale alias is chosen anyway, the graph key must carry the
  pool identity (the r149 lesson: a graph key that lost weight identity was worth +13.5 % once fixed).
* Rerank invalidation: `pool_rerank_from_heat_locked` re-derives the pool as the arena's complement at the first
  decode-band pass.  In a mirror there is nothing to rerank, but the code path must still be safe if a table is
  one slot short (i.e. exactly the case the predicate excludes).
* Prefill kernel coverage: the alias has to work for the kernels the prefill band actually uses (`MMQ`/`MMF`
  paths, the MoE fusions), not only the routed `MMVQ` band the cold read was built for.  If a kernel assumes
  device memory, the alias must decline for that op rather than fail.

## 4b. r221 (2026-11-05) shipped the shape this note recommends, in the DECODE band -- read this before step 3

`TODO` #269 landed as r221 (`WORKLOG.md` 261105.007, `INVARIANTS.md` r221, block 09).  It does **not** touch the
prefill staging copy or `MOE_HOST_POOL_ALIAS` -- your steps 1, 2, 4 and 5 are untouched, and the 8 % you are
chasing is still a prefill question.  What it did do is take **step 3's advice verbatim** ("prefer the per-expert
remap shape ... over aliasing `src0->data` wholesale") for the routed `MUL_MAT_ID` decode band, and in doing so
answers four of your §4 worries with measurements:

* **A covering pool is now a legitimate cold source: cold id `slots + pool slot`.**  `pool_covers_table()` gained
  a second consumer (`dr_pool_ok`, latched once at sizing) and the map builder emits the pool slot instead of the
  expert index, so the kernel resolves a miss through `base_dev`.  Measured on the default `-lme auto` mirror,
  1 GPU, 35B-A3B Q8_0 `-ncmoe 40`: 57.6 -> 63.3 t/s decode (**+9.9 %**), byte-identical text.  Nothing was
  invented twice: `pool_covers_table()` is still the one predicate, it is just stricter (see the next bullet).
* **The "one slot short" case in your §4 was NOT hypothetical -- it was a live bug.**  `kept_all` in
  `expert_host_src` retired a slot for a mirror whenever `pool->slots > n_experts - n_arena`, which is FALSE
  during the seeding pass (the arena is empty), so a mirror could end at 255/256 live residents and stand the
  remap down.  Fixed in block 15 (`keep_all` never retires a mirror), and `pool_covers_table` now counts LIVE
  slots (`slot_state == 1`): a retiring slot still has an `expert_slot` index but cannot serve a cold read.
* **The graph-capture worry is answered for this shape.**  The per-expert remap is safe under capture because
  the *decision* is latched per shape at sizing and the map is republished each call on the device stream;
  measured over 300- and 1000-token greedy runs with graphs on, bit-identical text on 1/2/3 devices.  Your
  rerank-invalidation fear is closed structurally: `pool_rerank_from_heat_locked` now returns before any
  bookkeeping when the pool covers the table (it was a no-op by coincidence), so a covering pool's
  `expert -> slot` map is **write-once after sizing**.  That is the invariant to reuse in step 3, whatever the
  transport.
* **The -22 % precedent did not reproduce in the decode band.**  For a *covering* pool the in-place read beat
  the eager fill path (part of the +9.9 %); a **sub-complement** pool cannot be remapped at all (`0 of 3840
  offers served`, by construction -- the map would have a hole) so it keeps the eager path, which is exactly
  your "keep the fill path for misses" requirement, enforced structurally rather than by measurement.

Still open for you, and unchanged by r221: the prefill band's kernels (`MMQ`/`MMF`, the MoE fusions) reading an
alias at all, the staging-share spike, and whether the bytes are better moved as a bulk copy.  The
`-ub` dependence you recorded is the thing to attack first.

## 5. Records

| where | what |
|---|---|
| [`wip/bench-56gb/RESULTS.md`](../../../wip/bench-56gb/RESULTS.md) | the matrix (mirror rows, ub sweep) |
| [`archive/work/host-expert-eager-push/P4-ROOT-CAUSE.md`](../host-expert-eager-push/P4-ROOT-CAUSE.md) | §5d, the -22 % registered-pageable-mapping precedent |
| [`archive/work/host-expert-eager-push/P6-POOL-REGION.md`](../host-expert-eager-push/P6-POOL-REGION.md) | the pool-as-expert-indexed-region design that `MOE_HOST_POOL_COLD_MULTIDEV` implements |
| [`archive/work/host-pool-speed/PROPOSAL-pipelined-staging.md`](../host-pool-speed/PROPOSAL-pipelined-staging.md) | the other way to pay less for the copy (pipelining); read before choosing in-place |
| [`archive/work/host-pool-speed/HEAD-TO-HEAD.md`](../host-pool-speed/HEAD-TO-HEAD.md) | the earlier head-to-head of the pool against a pinned master |
| `INVARIANTS.md` | the r215 item 7 (a registered source is not automatically a loss), the `expert_host_src` rule |
| `ENVIRONMENT.md` | `MOE_HOST_POOL_COLD`, `MOE_HOST_POOL_COLD_MULTIDEV`, `MOE_HOST_POOL_STAGE_BATCH`, `MOE_HOST_POOL_ALLOC`, `MOE_HOST_POOL_H2D` |
