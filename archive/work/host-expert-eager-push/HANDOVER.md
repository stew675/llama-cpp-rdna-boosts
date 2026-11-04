# HANDOVER — superseded for the refactor by `IMPLEMENTATION-PLAN.md`

**If you are here to implement the unified host-expert tier, read [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md)
instead (or first): it is the executable plan, with the decisions marked, the acceptance numbers, the tests and
the trap list, and it is designed to be carried out without waiting for the maintainer.  This file remains the
record of the investigation that led to it, and [`P4-ROOT-CAUSE.md`](P4-ROOT-CAUSE.md) is the evidence.**


**Status: the `-sm tensor` defect is DIAGNOSED and a fix is MEASURED.**  `P4-ROOT-CAUSE.md` is the record:
the loss is the driver's **pageable staging, a per-CALL cost** that a tensor split pays twice as often per
byte (in situ: 90 636 1-D pageable fills at **141 µs/copy = 12.82 s of a 27.7 s run**, against **1.3 µs/copy**
for the same arm's pinned-ring 2-D fills); registering the host mapping **in place**
(`MOE_HOST_PUSH_REGISTER_MIB`, patch `0007`) makes it **0.3-0.5 µs/copy**, and 2 devices go from 21.5 to
**71.6 t/s** against the pool's 51.5, byte-identical text.  One measured caveat decides the shape of the
fix: `hipHostUnregister` does **not** release the pin while the mapping lives (`hipFreeHost` rejects a
registered range; `unregister + munmap` does release), so the in-place form is a **permanent** pin and the
**releasable** form is a private registered **window** over the same file pages.

**Read `P4-ROOT-CAUSE.md` first; the sections below are the pre-diagnosis state of the investigation and are
kept because the ruling-out list (§4) and the trap list (§7) are still valid.**

---

Read this whole file before touching anything.  Sections 1-2 are the problem and the goal; 3 is every number
measured so far; 4 is what is ruled out and **what that rules out for you**; 5 is the ranked experiment list
with commands; 6 is the state of the tree; 7 is the trap list (several of these cost real time and two of them
produced wrong conclusions in the record).

---

## 1. The problem, stated exactly

`--host-experts pool` makes the **read-only model mapping** the expert master: the expert bytes live only in
the page cache (reclaimable) and are copied into VRAM arena slots on demand.  The campaign
(`wip/host-expert-eager-push`) wants the copy engine to read **straight from that mapping**, so that the
~4.7 GiB pinned host pool (a *duplicate* of the hot set) can be deleted.  The RAM is the point of the whole
campaign; the maintainer's bar is stated in **§2**.

`MOE_HOST_PUSH_SRC=mapping` (campaign patch `0001`) does exactly that for the 1-D geometry: it returns the
mapping as the `cudaMemcpyAsync` source and skips the pool slot / bounce ring entirely.

| config | pool path (`ctrl`) | direct mapping push | verdict |
|---|---|---|---|
| 1 GPU | 55.1 t/s | 53.6 | **parity (-3 %)** |
| 3 GPU `-sm layer` | 55.6 | 53.7 | **parity (-3 %)** |
| 2 GPU `-sm tensor`, devices 0,2 (both upstream x8) | **56.0** | **28.2** | **-50 %  <-- THE OPEN DEFECT** |
| 2 GPU `-sm tensor`, devices 0,1 (x8 + x4) | 52.5 | 26.1 | -50 % |
| 3 GPU `-sm tensor` (contains the x4 card) | 20.6 | 16.3 | -21 %, confounded, see §7 |

So the mechanism is **proven where fills are issued for one device at a time** (1 GPU, and `-sm layer`, which
puts whole layers on one device).  It breaks only where **several devices' fills are issued inside one op**
(`-sm tensor`), and only in the *fill path*, not in the bytes (see §3).

All numbers: 35B-A3B Q4_K_M, `-ncmoe 40 --host-experts pool`, `MOE_EXPERT_CACHE_MIB=2048`,
`MOE_HOST_POOL_FRAC=50`, `--seed 42 --temp 0 --ignore-eos -n 500`, ROCm 10.1.0, gfx1201 x3.

## 2. The goal, and the acceptance bar

- **Goal: delete the pinned pool's duplicate.**  4 712 MiB observed at FRAC 50 / 3 devices; 9 331 MiB planned.
  The bytes should live once, in the page cache, with residency guaranteed by **`mlock` over the mapping**
  (bounded, releasable — `PUSH-MLOCK-TIER.md`), not by a second copy of them.
- **Acceptance: parity.**  The maintainer's words: *"Let's call parity as 'within 4%' for both prefill/decode
  here.  I know that host push is always going to be a little slower, but given that it's only meant to be used
  when `pinned` doesn't fit, then the disk overheads are likely to dominate anyway."*
- **Fallback the maintainer would tolerate:** a staging buffer **under 2 GiB** that keeps performance.  The
  pool-size sweep in §3 says a *small* pool is worthless (512 MiB and 1 GiB are indistinguishable from none),
  so a 2 GiB *residency* pool is probably not viable either — but a 2 GiB ceiling on a **transport** buffer is
  easily met (the bounce ring is tens of MiB).
- **Target band:** the mode exists for the **not-fitting** case (set > RAM, disk on the critical path).  Every
  number here is on the *fit-in-RAM* band, where `auto` should not choose push at all.  **The not-fitting band
  is UNMEASURED** and is the band where the gap is expected to shrink most — see §5 experiment 5.

## 3. Everything measured so far (all reproducible, logs in `runs/`)

### 3a. The cost is host time INSIDE the copy call

`MOE_HOST_PUSH_PROBE=1` (patch `0006`) times every fill copy on the host and records the current device:

| devices | ctrl arm: host time in the copy | mapping arm: host time in the copy | per H2D copy (mapping) |
|---|---|---|---|
| 1 | 52 437 µs / 53 495 copies (**1.0 µs/fill**) | 2 353 275 µs | **44 µs** |
| 2 (0,1) | 243 258 µs / 173 708 (**1.4 µs/fill**) | **15 150 708 µs** | **107 µs** |
| 3 (0,1,2) | 100 246 µs / 68 078 (**1.5 µs/fill**) | **11 066 908 µs** | **191 µs** |

At 2 devices **15.2 s of a 34.3 s process** is blocked inside `cudaMemcpyAsync`.  A pinned/D2D copy issued from
the same thread is **1.4 µs**.  So it is the *pageable staging* inside the call, and it **degrades with device
count**: 44/107/191 µs ≈ **12 / 5.6 / 3.1 GB/s aggregate**.

### 3b. It is not the bytes, and not the traffic

The census is **identical between the arms** at every device count (`expert H2D traffic` 48 637.6 MiB at 2
devices, 19 934.1 MiB at 3; same hits/fills/evictions).  What grows is the **number of copies**: a tensor split
gives each device its own slice, so per-copy size halves (0.57 MiB at 1 device -> 0.28 MiB at 2) and the call
count doubles for the same bytes — and each smaller copy costs *more*.

### 3c. What the pinned RAM is actually buying (pool swept by size, 0,2, `-n 500`)

| arm | pinned pool | decode t/s | vs pool |
|---|---|---|---|
| pool FRAC 50 | 4 712 MiB | **55.9** | 100 % |
| bounce only (`MOE_HOST_POOL_MIB=1`, `PUSH_SRC=pinned`) | ~0 | 41.9 | 75 % |
| pool 512 MiB | 512 MiB | 42.2 | 75 % |
| pool 1024 MiB | 1 GiB | 42.7 | 76 % |
| direct mapping push | 0 | 28.4 | 51 % |

Two readings, both important: **a pool smaller than the hot set is worth nothing** (512 MiB == 1 GiB == none),
and the 25 % between the pool and the bounce path is the per-fill host memcpy that a page-cache source always
pays (`mlock` prevents *reclaim*, not the copy).  The bounce path — page cache -> pinned ring -> pinned H2D —
is therefore the **current best page-cache-sourced transport: 75 %**, and the direct push has to beat it.

### 3d. Standalone probes (both in this directory, both build clean)

`probe-pageable-multi.cpp` — pageable vs pinned vs pool shape (host memcpy into a pinned slot, then CE):

| devices | pageable | pinned | pool | per-fill pageable / pool |
|---|---|---|---|---|
| 1 | 16.2 GB/s | 23.6 | 17.2 | 38 / 36 µs |
| 2 | 12.0 | 25.7 | 15.0 | 51 / 41 µs |
| 3 | 13.3 | 37.9 | 20.4 | 46 / 30 µs |

`probe-pageable-threads.cpp` — one issuing thread for all devices vs **one thread per device**:

| devices | one thread | one thread per device |
|---|---|---|
| 1 | 15.0 GB/s (41 µs) | 15.1 (41 µs) |
| 2 | 11.8 (52 µs) | **20.6 (30 µs)** |
| 3 | 12.8 (48 µs) | **29.4 (21 µs)** |

It also supports `SHARED=1` (workers enqueue on the main thread's stream) and `MMAPF=<file>` (source is a
file-backed, **not** pre-faulted mapping).  **Neither makes it slower or hangs.**

## 4. Ruled out — and what each result means for you

1. **Bytes / traffic.**  Identical census between arms (§3b).  Do not chase traffic.
2. **The device context.**  46 % of fill calls run with the wrong current device (80 134 / 173 708 at 2
   devices) — **but equally in both arms**, and `MOE_HOST_PUSH_SETDEV=1` (make `t.device` current before the
   copy) gives **26.0 vs 26.1 t/s**.  Fixing it is still correct hygiene; it is not the cause.
3. **A per-device issuing thread.**  Implemented and **verified working** — `115 808 fills issued by worker
   threads, 0 caller fallbacks, 0 stalled devices` — and **performance-neutral: 27.5 t/s against 28.2**.  So the
   real path does *not* reproduce the probe's thread win.  Do not simply re-add the thread; find out why it does
   nothing here (that is experiment 3 in §5).
4. **A pageable copy from a foreign thread is NOT unsafe on this stack.**  This was asserted in the record and
   is **wrong** — retracted in `P3-MULTIGPU-RESULTS.md` §3a.  Both "wedges" were a **teardown deadlock in the
   campaign's own code** (a static array of `std::condition_variable` destroyed while detached workers waited on
   it: `pthread_cond_destroy` blocks with waiters).  Heap-allocate it and never destroy it — or join at exit.
   Generation had completed every time; only `exit()` hung.
5. **`hipBLASLt`.**  Every measurement in this file ran with `ROCBLAS_USE_HIPBLASLT=0`.

## 5. What is NOT ruled out — the ranked experiment list

The core unexplained number: **the probe does 21-52 µs per 0.6 MiB pageable fill, the real path 107-191 µs.**
Everything about thread, stream and source has been reproduced in the probe without reproducing the cost, so
look at what the probe does NOT model.  Ranked by (cheapness × discrimination):

1. **The DESTINATION.**  The probe copies into a plain `hipMalloc` buffer; the path copies into an
   **expert-cache arena slot** (the delivery's arena / `ggml-cuda-vmm` machinery, `cudaMalloc`-backed arena with
   its own mapping/slab behaviour).  If pageable staging into VMM-mapped/arena memory takes a slower driver
   path, this is the whole answer.  **Test:** add a `VMM=1` mode to `probe-pageable-threads.cpp` that allocates
   the destination with `hipMemAddressReserve` + `hipMemCreate` + `hipMemMap` (and, if that is clean, with the
   delivery's own arena allocator via a tiny C test harness), and compare 21 µs vs ?.  **Do this first.**
2. **Per-call fixed cost vs per-byte cost on the real geometry.**  Sweep the probe's `chunk` (32 KiB, 64 KiB,
   128 KiB, 256 KiB, 512 KiB, 600 KiB, 2 MiB) at 2 devices with one thread per device, and plot µs/fill.  The
   path's copies are **0.28 MiB** at 2 devices, so if the curve is dominated by a fixed per-call cost, that
   alone explains the gap and points at reducing the call count (coalescing consecutive experts — the fix
   `bounce_acquire`'s own note already names, and the ring RAM win comes with it).
3. **Why the per-device thread does nothing in the path but a lot in the probe.**  Instrument the queued path
   (`MOE_HOST_PUSH_PROBE` already times `copy_slot_from_src`, but the queue bypasses it — move the probe
   *inside* `uq_worker` so the queued copies are timed too).  Then compare the per-copy time with the queue on
   and off at 2 devices: if the copy time drops but the total does not, the loss moved to the handshake (batch
   the handoff: **one job per op**, not per fill); if the copy time does not drop, the staging is serialised by
   something outside the thread.
4. **Staging granularity / page alignment.**  Every expert is 720 896 B (a 4 KiB multiple) but starts at
   different page offsets in the mapping; check whether the driver stages whole pages (i.e. more bytes than
   asked) — compare the *effective* staging throughput of a 0.28 MiB copy at an arbitrary offset vs a
   page-aligned one.  Also try `cudaMemcpyAsync` on a `cudaMemcpyDefault` kind, and `hipMemcpy2DAsync`
   avoidance.
5. **The not-fitting band (the band that matters).**  The campaign's own P3: emulate 64 GiB with `memory.max`,
   so every miss is a disk read.  Then pool (DIO) vs bounce vs direct push.  The expectation is that the
   transport difference collapses when the read is `~1.32 ms` instead of `~40 µs` — if it does, the direct push
   is already acceptable **for the band the mode exists for**, and the fit-in-RAM gap becomes an interesting
   wrinkle instead of a blocker.  **Run this even if experiments 1-4 are inconclusive: it may close the
   campaign's acceptance question on its own.**
6. **Coalesce consecutive expert fills** (after 1-4 tell you whether it is worth it): one staging copy per run
   of consecutive experts instead of one per expert.  Helps the pageable path *and* bounds the bounce ring's
   RAM (uncapped today: 2.2-3.0 GiB on the 122B prefill miss path).

## 6. State of the tree, branches and patches

- `~/llama.cpp`: branch **`p2-eager-push-r209`** = r209 (`27271aa91`) + campaign patches `0001`-`0006`.
  Built with `~/bin/build-llama-rocm` (ROCm 10.1, `build-rocm`).  **Working tree clean.**  `bin-p2r209` holds a
  copy of the same build.
- Pushed to the personal fork: `origin/p2-eager-push-r209` @ `f065c96e5`.  `rdna-boosts` untouched.
- Loose patches in `wip/host-expert-eager-push/patches/`: `0001` push source switch, `0002` mlock working set,
  `0003` heat-seeded pinned set, `0004` rotation + device-set gate, `0005` once-per-process diagnostics,
  `0006` attribution probe + setdev A/B.  They replay in order onto a clean r209 (verified: the result matches
  the branch file bit-for-bit).
- All switches are **default off**: `MOE_HOST_PUSH_SRC`, `MOE_HOST_PUSH_MLOCK_MIB`/`_SEED`/`_ROTATE`,
  `MOE_HOST_PUSH_PROBE`, `MOE_HOST_PUSH_SETDEV`.  With them unset the arms are byte-identical to pristine r209
  (verified: 19.0/18.9 t/s at 3 devices, identical census).
- The per-device **upload queue is NOT in the tree** — it worked but was performance-neutral (§4.3), and its
  code is recorded in the git history of this repo's `P3-MULTIGPU-RESULTS.md` §3a rather than left in the tree.

### Reproduce the headline measurement

```bash
cd <this repo>/wip/host-expert-eager-push
# clean x8 pair -- see §7.1; NEVER use 0,1 or all three for a scaling claim
GPU=0,2 SM=tensor EOS=1 N=500 ./p3-matrix.sh ctrl
GPU=0,2 SM=tensor EOS=1 N=500 ./p3-matrix.sh mapping  MOE_HOST_PUSH_SRC=mapping
GPU=0,2 SM=tensor EOS=1 N=500 ./p3-matrix.sh probe    MOE_HOST_PUSH_SRC=mapping MOE_HOST_PUSH_PROBE=1 LV=4
```

`p3-matrix.sh` prints t/s, the text hash, the census and a fault check; `LV=4` adds the census but then the
**text hash is not usable** (see §7.2).  `p3-busy.sh` samples GPU busy% and the process's host CPU time.

## 7. Traps — each of these cost real time, two produced wrong conclusions

1. **The PCIe topology.**  Two cards are upstream **x8**, one is upstream **x4** (HIP device **1**, BDF
   `06:00.0`; the H2D calibration in the log says so: **14.5 GB/s against 28.7**).  Use
   `HIP_VISIBLE_DEVICES=0,2` for any pair or scaling claim; a 2-device arm on `0,1` mixes x8 and x4, and the
   3-device arm cannot be used as a scaling point at all.  All three cards report `current_link_width 16` in
   sysfs — that is the card-side link through the switch, not the upstream bandwidth; do not trust it.
2. **Never hash a log that has stderr mixed in or `-lv 4` on.**  llama-cli writes the generated text to
   **stdout** and every log line to **stderr**; `scripts/extract-generated.py` needs them separate.  Redirect
   `>log.out 2>log.err`, hash `log.out`.  A `-lv 4` or `2>&1` log interleaves stats into the text and will show
   two byte-identical arms as "different text" (this happened, twice, and once from the campaign's own
   `fprintf(stderr, ...)` diagnostics landing mid-generation).
3. **A static array of `std::condition_variable` + detached worker threads = a teardown deadlock.**  The
   process finishes all its work and hangs in `exit() -> __cxa_finalize -> pthread_cond_destroy`, with the
   workers parked in `cv.wait()`.  It looks exactly like a driver wedge.  Heap-allocate and never destroy, or
   stop and join the workers before `exit`.
4. **ROCm's `AsyncEventsLoop` threads normally sit in `kfd_wait_on_events`.**  That is *not* evidence of stuck
   GPU work (it was misread as such).  If you need the truth, attach gdb (`ptrace_scope=0` here, `gdb` and
   `eu-stack` both installed) and read the stacks.
5. **`over ~N tokens` in `expert H2D traffic` is not the decode length** — it is the cache's traffic counter and
   reads `~6` on a full 500-token decode.  The decode length is `slot print_timing ... eval time = ... / N
   tokens`.  (`p2-gate-3gpu.sh` now reads the right one.)
6. **`llama-cli` in this tree is built on the server implementation** and emits `srv`/`slot print_timing` lines
   of its own — those do not mean "this ran llama-server".
7. **The pool is only live at >= 2 devices.**  On 1 GPU it reports `0 tensors, 0 slots`, so a 1-GPU "push vs
   pool" arm is really push vs the bounce ring.  Check `host pool (L2)` in the log before believing an A/B.
8. **Under `-sm tensor` the per-op boundary hook never runs.**  The whole graph is one Meta split, so the MoE
   op's routing ids are computed inside it and the issue-#129 guard stands the scheduler's host-readback branch
   down; `moe_cache_update_host` (where the campaign's rotation and its device-set gate live) is reached only at
   sizing — measured **3 boundaries** against **60 357** on `-sm layer`.  The arena is still driven (take-over
   plus demand fills), but any per-op host epoch must come from somewhere else.
9. **`--ignore-eos -n 500` for any throughput comparison.**  An EOS-stopped ~150-token answer averages in the
   cold head and understates throughput; and text-purity needs the longer fill stream anyway.
10. **Serial runs only.**  Never two arms at once (`AGENTS.md`), and keep `-t` off the GPU IRQ cores.
11. **`GGML_SCHED_DEVGATHER`** is on (prefill band) by default for a pool-backed table, and it serves the routed
    table itself — so **any prefill A/B of the fill path must set it to `0`**, otherwise the thing you are
    measuring is bypassed.  For decode it is gated off (0 gather calls measured) and irrelevant.

## 8. Where the record lives

- `P3-MULTIGPU-RESULTS.md` — the measurement record for everything in §3 (§3a carries the retraction of the
  wrong "driver wedge" conclusion).
- `PUSH-MLOCK-TIER.md` — the residency design (mlock over the mapping: releasable, measured) and §8b/§8c.
- `DESIGN.md` — the invariants (P1 never device-pull, P3 slot ownership, P6 one push queue per device, P7
  byte-identical) and the rejected alternatives.
- `README.md` — the campaign's status banners.
- `TODO.md` #137 (this issue) and #138 (bound the bounce ring's RAM).
- `INVARIANTS.md` — the four r209 entries this campaign established (search `r209, wip/host-expert-eager-push`).
- `runs/` — every log quoted above, plus the gate harness output.
