# Proposal review — "Pipelined Asynchronous Staging" (external suggestion, 2026-11-02)

An outside proposal (Google Gemini, relayed by the maintainer) for the host-expert path:

> Decouple disk I/O, CPU staging and GPU execution into a multi-buffered ring: pinned host buffer A is copied
> to VRAM while a background host thread fills pinned buffer B from disk; open files with `O_DIRECT` so the
> read bypasses the page cache and lands **directly in the pinned host memory**, "removing the CPU memory-copy
> overhead" and hiding the "bounce buffer" penalty behind GPU execution.  Framed as an alternative to "letting
> the runtime manage page faults via managed memory".

**Assessment: three quarters of this is already the delivery's shipped design, and the one quarter that is
not (a *demand-fill* worker) is a P1-a question with two measured negative precedents and a 19 % ceiling that
only exists on a sub-full pool.  Recorded here so it is not re-proposed as new; screen it with the replayer
before any code (§9.6).**

## What already exists (do not re-implement)

| the proposal's element | where it is in the tree | record |
|---|---|---|
| a multi-buffered ring, filled ahead of the compute | the scheduler's staging ring: `GGML_SCHED_STAGE_SLOTS` (default **8** slots), per-slot upload events, an **independent copy stream**, `GGML_SCHED_STAGE_MAX_MB` for the byte cap, and a per-split plan issued before any upload | `ggml/src/ggml-backend.cpp` (`sched_stage_issue`); r111/r114 |
| the "bounce" is already gone from the staging path | `moe_cache_stage_table` fills a pool-backed table's tier-resident experts with **one kernel launch** (`MOE_HOST_POOL_STAGE_BATCH=1`, default) instead of one `cudaMemcpyAsync` per expert | measured 1564 -> 2103 t/s prefill (r114) |
| `O_DIRECT` straight into **pinned** host memory, no bounce | `pool_fill_slot_locked` -> `pool_dio_read(p.fd, p.base + slot*host_bytes, ...)`, and `p.base` is `cudaHostAlloc(Portable\|Mapped)`.  A thread-local alignment bounce exists only for a non-page-aligned tail | `MOE_HOST_POOL_DIO` (`auto` by default, r114) |
| overlap of the fill with compute | the pool staging issues its miss fills and the tier kernel on the copy stream, ordered after the compute; the copy engine runs them beside the GPU | r114 |
| a speculative prefetch of upcoming fills | `MOE_HOST_POOL_PREFETCH` — **default 0**, because a long-prompt sweep showed it *hurts* until the pool nears full residency | `ENVIRONMENT.md` |
| a low-priority background prefetch worker | **tried and removed** (r113, `MOE_HOST_POOL_BGFETCH`): the eviction-side swap with a demand prefetch cost **2x decode for 0 MiB of L3** | `ENVIRONMENT.md` (`removed (r113)`) |

Two framing points do not apply at all:

* **"managed memory / page faults"** — the delivery has never used managed memory.  The faulting design was
  `--host-experts window`, closed 2026-11-02: on this no-XNACK hardware a kernel read of a reclaimable page is
  a fatal `Page not present` fault (`INVARIANTS.md`, "device-visible host memory must be RESIDENT").  Nothing
  in the pool path lets a kernel read pageable memory, and mechanism (b) of this campaign (§14.4) removed the
  last reader that could have.
* **"removing the CPU memory-copy overhead"** — priced this session: everything that still goes through
  `moe_cache_upload_pageable`'s pinned bounce is **56.4 MiB over a 3-request srv-cell run** and **3664 MiB on
  the 122B** (0.26 % of a 142 s run, §14.5).  With `MOE_HOST_POOL_DIO=1` (or `auto` on a box the set does not
  fit in) there is no host copy on the fill path at all — the read lands in the pinned slot.

## The part that is genuinely open

**A demand-fill worker that overlaps the host-side `pread`/`memcpy` of the *current* ubatch's misses with the
GPU's compute of the previous one.**  Today a miss is filled synchronously by whichever thread is inside
`expert_host_src`/`pool_fill_slot_locked`, so on a sub-full pool the host thread blocks on storage while the
device has work it could be doing.  The proposal is right that this is the one place where the pipeline has a
hole.

Sizing it first (this host, 122B, `-ncmoe 48`, FRAC 80, 1862-token decode, §14.3):

| quantity | value |
|---|---|
| disk read per run | 160-165 GiB |
| model load (the `pinned` arm reads the same 64-72 GiB) | ~73 GiB |
| one-shot pool prepopulate (session 2's figure) | ~53.7 GiB |
| **residual = demand reads** | **~33 GiB over 1862 tokens ≈ 18 MiB/token** |
| decode step at 19.31 t/s | 51.8 ms |
| 18 MiB at the measured ~1.8 GB/s sustained | **~10 ms = ~19 % of the step** |

So the **ceiling is ~19 % of decode, and only on a sub-full pool** — at FRAC 100 the run reads 95 GiB, the
pool hit rate is 0.91 and the demand residual is ~0.  It buys nothing on the maintainer's "big enough for the
whole set" case, which is exactly the case the acceptance criterion is about.

Two measured precedents say the obvious implementations lose:

1. `MOE_HOST_POOL_PREFETCH` is a routing-driven speculative fill and it **hurt** until the pool was nearly
   full — a prefetched slot is a slot the demand path cannot use, and the fill competes for the same disk.
2. The r113 background worker measured **2x decode for 0 MiB** of L3 benefit.  That one was eviction-side, so
   it is not a direct refutation — but it is the same "add a thread that touches the pool behind the policy"
   shape, and it is why the layout is static (r113: the pool is a *layout*, not a churning cache).
3. `archive/work/moe-cpu-overlap/RESOLUTION.md` measured the cold-read pressure at **0.1-0.4 % of a decode
   token** at the delivered arena and **7 %** at the smallest arena the cache permits, and parked the
   CPU-compute-beside-GPU shape above a ~1.2 GiB arena (its cost was scheduler dispatch serialisation, 8-10
   ms/token, not the compute).

## If it is pursued, this is the shape and the order

1. **Screen first, per §9.6.**  `scripts/moe-trace-replay.py` predicts MiB of L3 reads per decode token for a
   given arena/pool split; run it at the campaign's arm and convert with the measured 1.8 GB/s.  If the model
   does not separate "18 MiB/token demand" from "18 MiB/token demand, overlapped", there is nothing to build.
   (It cannot score the overlap itself — only the bytes — so the decision needs one instrumented run.)
2. **Instrument before building**: count demand fills and their wall time (the counters added in
   `patches/0003` are the shape: `pageable-master fallback`, `device gather`).  A `pread` histogram over a
   2000-token decode is the cheapest possible experiment and it either shows a blocking 10 ms/token or it does
   not.
3. **If it shows up, the implementation is a demand-fill worker writing into POOL SLOTS** — never a second
   bounce ring (that is the window mode, and its `-6.7 GB` was a pinned duplicate of the host set), and never
   eviction-side (r113).  The pool slot is the pinned destination the proposal wants; a fill only ever takes
   an **empty** slot, so a background fill cannot tear a live read.
4. **`io_uring` is not the first step.**  It is a new dependency for a lever whose ceiling is 19 % and whose
   two nearest precedents are negative.  A plain `pread` on one worker thread, with the pool's existing
   `O_DIRECT` entry point, measures the same overlap on one queue depth; go asynchronous only if the
   instrumented run shows multiple in-flight fills are needed.
5. Gates if it becomes code: same-seed greedy text, `MOE_HOST_POOL_VERIFY=1` (0 mismatches — the O_DIRECT
   alignment tail is the corruption mode this path has actually produced), `check-gpu-faults.sh`, the
   prefill-logit gate, and a >= 2000-token decode against the synchronous arm.

**Status: recorded, not scheduled.**  It stays P1-a's territory (`README.md` §4.1: a per-read residency
decision plus readahead), which is where the campaign's own plan already puts it.
