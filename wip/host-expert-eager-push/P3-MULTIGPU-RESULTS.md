# P3 RESULTS — the multi-GPU assessment: `-sm tensor` is an OPEN defect, not a closed question (2026-11-03)

> **OPEN.  This file is the measurement record; the live investigation is
> [`HANDOVER.md`](HANDOVER.md).**  Nothing here closes the campaign: the direct page-cache push is at parity on
> one device and on a 3-device `-sm layer` split, and **50 % short on a 2-device `-sm tensor` split**.  The
> cause is *partially* attributed (§3a) and the remaining gap is unexplained — the ranked experiment list, the
> exact commands and the trap list for the next session are in `HANDOVER.md`.  Do not read "the driver's staging
> is slow" as "this cannot be fixed": what has been ruled out is listed in `HANDOVER.md` §4, and the prime
> untested candidate (the copy DESTINATION: plain `hipMalloc` in the probes vs an arena/VMM slot in the path) is
> `HANDOVER.md` experiment 1.

**Verdict: correctness GREEN on 1, 2 and 3 devices.  The TRANSPORT — a direct CE source on the pageable
mapping, not the host-push principle — WORKS: parity on 1 device (55.1 vs 53.6 t/s) and parity on a 3-device
`-sm layer` split (55.6 vs 53.7), with the pool live and byte-identical text.  It breaks in exactly one
configuration: **`-sm tensor`, where several devices' pageable fills are issued inside one op** (2 devices
52.5 -> 26.1, 3 devices 20.6 -> 16.3).  That is the open engineering question, not a refutation of the
mechanism.  Separately, the DEVICE-SET GATE (P2 increment 3) cannot be exercised on a tensor split at all.
All of it is on the fit-in-RAM band, ROCm 10.1 — the not-fitting band the charter targets is NOT measured.**  Nothing here is in `patches/`; this
is a wip campaign record.  Measured on gfx1201 x3 (R9700), ROCm 10.1.0, `~/bin/build-llama-rocm`,
`Qwen3.6-35B-A3B-UD-Q4_K_M`, `-ncmoe 40 --host-experts pool`, `MOE_EXPERT_CACHE_MIB=2048` (per device),
`MOE_HOST_POOL_FRAC=50`, `--seed 42 --temp 0`, greedy.

## 0. What was built

The four campaign commits (`patches/0001`-`0004`, the fork branches `p1-push-src` @ `be83da8f0` and
`p2-pinned-working-set` @ `f131cef2f`) were cut on the **r208** chain (`efaa60db0`).  They were **rebased onto
r209** (`27271aa91`, `release.json.tip`) as branch **`p2-eager-push-r209`** in `~/llama.cpp`:

```
git rebase --onto 27271aa91 efaa60db0 origin/p2-pinned-working-set
# 4/4 applied, NO conflicts; diff vs r209 = 389 insertions, 0 deletions, ggml/src/ggml-cuda/moe-expert-cache.cu
```

The rebase is clean because r209's changes to that file (the WDDM RAM figures, block 09/12) land in code
regions the campaign does not touch.  Built green; `bin-p2r209` is kept alongside so the pristine r209 build
can be A/B'd without a rebuild.

**Unlike P1, the pool is actually built in every arm below** (`61 host tensors, 18662 MiB host expert bytes,
7808 slots, 4712 MiB pinned` at 3 devices) — P1's `0 tensors` trap is gone, so these arms compare the push
transport against a live pool.

## 1. Correctness — GREEN, and a much stronger stress than P1

`--ignore-eos -n 500`, so the text is a forced 500-token greedy continuation (748-757 chars) rather than a
~150-token answer that stops at EOS:

| arm | ctrl (`pool`) | `mapping` (push) | direct pageable fills | text |
|---|---|---|---|---|
| 1 GPU | 748 ch | 748 ch | 32 326 | **identical** `sha=f92b4e201207` |
| 2 GPU `-sm tensor` | 757 ch | 757 ch | 65 260 | **identical** `sha=c6e6d709a6db` |
| 3 GPU `-sm tensor` | 748 ch | 748 ch | 35 806 | **identical** `sha=f92b4e201207` |
| 3 GPU `-sm layer` | 748 ch | 748 ch | 36 311 | **identical** `sha=f92b4e201207` |

- The 2-device answer differs from the 1/3-device answer — that is the documented per-device-count reference
  rule (`AGENTS.md` gate 6), not a campaign difference: within the same device count every arm agrees.
- The access census is **identical** between the arms at every device count (`h=0.7664 (125374/163584
  reaches), fills=49550, H2D 13446.9 MiB, takeover=19880 decline_all=142` at 3 devices), i.e. the switch
  changes the transport and nothing else.
- `scripts/check-gpu-faults.sh` clean on every one of the 20+ runs; no `Page not present`, no aborts.
- The pinned working set changes nothing either: 3-device decode 11.8 (`mlock` 512 MiB, rotate on) vs 11.9
  (transport only) on the 300-token arm, text identical.

**So the "a pageable `cudaMemcpyAsync` tears under back-to-back use" claim (`ggml-cuda.cu:6255`) is refuted a
third time, now at 2 and 3 devices with up to 65 260 consecutive direct fills.**  It is still refuted on
**ROCm 10.1 only** — the 7.14 arm remains owed (separate build), exactly as `P1-RESULTS.md` says.

### 1b. The gate script's text hashes were an artefact, not a divergence

`p2-gate-3gpu.sh` ran with `-lv 4` *and* extracted the text from the same log.  `scripts/extract-generated.py`
says in its own header that it must be run **without** `-lv 4` because the verbose log interleaves statistics
lines into the generation — so the rotate/norotate hashes it printed (`f3a2667e3c68` / `d882767767e6`, 5017 vs
5016 chars) are meaningless.

**There was a second, independent source of the same artefact, in the campaign's own code.**  On the
`-sm layer` arm the rotate run hashed **1373 chars `cc9ce502e184`** against the control's 748 chars
`f92b4e201207` — and the "divergence" was the once-per-event `fprintf(stderr, "P1-PUSH: pinned working set
full: ...")` landing *inside* the streamed generation, twice (836 then 835 experts), because rotation frees
budget and `push_rotate_locked` then re-opens first-use admission, which re-triggers the message.  Fixed in
`patches/0005`: both P1-PUSH diagnostics ("working set full" and "mlock refused") are now **once per
process**.  This was an instrumentation defect, not a rotation bug — `released=40 release_denied=528` is
correct behaviour.

**The gate now separates the streams** (`>out 2>err`): llama-cli writes the generated text to stdout and
everything else to stderr, so the hash is valid *always*, `-lv 4` can stay on, and one run per arm is enough.

## 2. Performance — the transport is at PARITY except under `-sm tensor`

Decode t/s, `--ignore-eos -n 500` (a 150-token EOS-stopped answer averages in the cold head and is a worse
sample; the shorter arms are quoted in §2b because they show the same shape, larger):

| config | ctrl (`pool`) | `mapping` (push) | delta |
|---|---|---|---|
| 1 GPU `-sm tensor` | 55.1 | 53.6 | **−3 % (noise)** |
| **2 GPU `-sm tensor`** | **52.5** | **26.1** | **−50 % (2.0x)** |
| 3 GPU `-sm tensor` | 20.6 | 16.3 | −21 % |
| 3 GPU `-sm layer` | 55.6 | 53.7 | **−3 % (noise)** |

### 2b. The shorter arms (`-n 300`, EOS-stopped, two reps each) — same shape

| config | ctrl | mapping | delta |
|---|---|---|---|
| 3 GPU `-sm tensor` | 19.0 / 18.9 | 11.9 / 11.7 | −38 % |
| 2 GPU `-sm tensor` | 39.2 / 39.8 | 13.8 / 13.9 | −65 % (2.8x) |
| 1 GPU | 40.7 | 39.1 | −4 % |
| 3 GPU `-sm layer` | 44.7 | 42.0 | −6 % |

The pristine **r209** build (campaign patch absent) reproduces the ctrl arm exactly — 19.0/18.9 t/s and the
identical census — so the switch is inert with `MOE_HOST_PUSH_SRC` unset and every difference above is the
switch.

**Read this against the design's own rule (§5 of `DESIGN.md`): this config is a `pinned` config.**  The host
set is 18662 MiB against 81 GiB `MemAvailable`, `pool_decide_dio_locked` logs `buffered (page cache as L3)`,
and `auto` should never choose push here.  So the −3 % arms are "a rung that should not have been used costs
nothing", and the −50 %/−65 % arms are "and on a split it costs a lot".  The band this campaign was chartered
for — the set does **not** fit, disk on the critical path, `pool`/DIO — is **not measured by P3** and remains
the open one.

## 3. Mechanism — the host is the critical path, and pageable staging makes it worse

**The GPUs really are starved** (this is a measurement, not an impression).  `gpu_busy_percent` is unusable as
a utilization figure on this box (all-VRAM, 138 t/s reads 14 % mean), so the signal is the process's own CPU:
`/proc/<pid>` utime+stime sampled alongside the run (`p3-busy.sh`):

| arm | decode t/s | llama-cli host CPU | host CPU per token |
|---|---|---|---|
| 3 GPU `-sm tensor`, ctrl | 19.0 | 1.77 cores | **93 ms** |
| 3 GPU `-sm tensor`, mapping | 11.9 | 1.27 cores | 107 ms |
| 3 GPU `-sm layer`, ctrl | 44.8 | 1.55 cores | **35 ms** |
| 1 GPU, ctrl | 40.8 | 1.70 cores | 42 ms |

A host-expert `-sm tensor` split spends ~1 core-second of CPU per generated token, ~3x the `-sm layer` figure
at the same device count, so the transport's cost lands directly on the critical path rather than under a DMA.

**The marginal cost of a pageable fill over a pool fill**, from the wall-time difference divided by the fill
census (derived, 300-token arms): **28 µs (1 dev) → 205 µs (2 dev) → 302 µs (3 dev tensor) → 41 µs (3 dev
layer)**.  Meanwhile the pool's explicit memcpy is only **483 ms of a 14 s run** (3.4 %), so deleting it should
have been a small win, not a 2x loss.

**`probe-pageable-multi.cpp`** (new, standalone HIP; pageable `malloc` source vs `hipHostMalloc` vs the pool
shape — host memcpy into a pinned slot then CE from it, 600 KiB fills, 64 fills/device/round):

| devices | pageable | pinned | pool | per-fill pageable / pool |
|---|---|---|---|---|
| 1 | 16.2 GB/s | 23.6 | 17.2 | 38 / 36 µs |
| 2 | 12.0 | 25.7 | 15.0 | 51 / 41 µs |
| 3 | 13.3 | 37.9 | 20.4 | 46 / 30 µs |

The driver's pageable staging is **real and gets worse with device count** (pageable/pool 1.06x → 1.26x →
1.54x) — the "staged through a shared internal bounce" half of the `ggml-cuda.cu:6255` comment is measurably
correct, even though the *tearing* half is not.  But it is **~20x smaller** than what the llama path pays at 3
devices (46 µs vs 302 µs marginal), so the driver bounce is **not the whole story**.  What remains is per-fill
host overhead inside the multi-device fill path, which needs a profile, not another microbenchmark.  The shape
of the evidence points at the one difference between the arms that work and the arm that does not: `-sm layer`
and 1 GPU issue a fill batch for **one device at a time**; `-sm tensor` issues the batch for **every device of
the op**, from one host thread, back to back.  Candidate causes, cheapest test first:

| # | candidate | why it fits | test |
|---|---|---|---|
| a | **copy granularity**: the mapping branch issues one `hipMemcpyAsync` per expert (0.375-0.6 MiB), paying the driver's fixed staging setup per call | consecutive experts are contiguous host bytes with a uniform stride, so a run of them is one 1-D copy; the probe shows per-fill cost dominated by fixed setup | coalesce a run of consecutive expert fills into one copy, A/B at 2 devices |
| b | **per-call device context**: `ggml_backend_meta_moe_cache_update` loops over the child backends with no `ggml_cuda_set_device` between them | free at 1 device, invisible under `-sm layer` (one device per op), paid 2 of 3 calls on a 3-device op — the observed pattern | log `hipGetDevice()` at the copy, then add the guard |
| c | driver staging contention across the devices of one op | the probe bounds it at ~1.54x | already bounded; at most a part |

## 3a. ATTRIBUTED: the pageable copy call itself, and the staging resource is PER THREAD

`MOE_HOST_PUSH_PROBE=1` (`patches/0006`) times every fill copy on the host and records whether the copying
backend's device was current.  Same arms as §2, `--ignore-eos -n 500`, `-lv 4`:

| devices | ctrl arm: host time in the copy | mapping arm: host time in the copy | per H2D copy (mapping) |
|---|---|---|---|
| 1 | 52 437 µs (53 495 copies, **1.0 µs/fill**) | 2 353 275 µs | **44 µs** |
| 2 | 243 258 µs (173 708 copies, **1.4 µs/fill**) | **15 150 708 µs** | **107 µs** |
| 3 | 100 246 µs (68 078 copies, **1.5 µs/fill**) | **11 066 908 µs** | **191 µs** |

At 2 devices **15.2 s of a 34.3 s run is spent blocked inside `cudaMemcpyAsync`** — that is the whole 52.4 ->
26.0 t/s.  A pinned/D2D copy from the same thread costs 1.4 µs, so the driver's pageable staging is ~75x
slower per call and **degrades** with concurrent devices: 44 / 107 / 191 µs is ~12 / 5.6 / 3.1 GB/s aggregate.

**Not the bytes.**  The census is identical between the arms at every device count (48 637.6 MiB at 2
devices, 19 934.1 MiB at 3), so the switch moves the same bytes.  What changes is the number of **copies**: a
split gives each device its own slice, so per-copy size halves (0.57 MiB at 1 device -> 0.28 MiB at 2) and the
call count doubles — and each of those smaller copies costs *more*.

**Eliminated: the device-context candidate.**  46 % of all fill calls run with the wrong current device
(80 134 / 173 708 at 2 devices) — but equally in **both** arms, and `MOE_HOST_PUSH_SETDEV=1` (make
`t.device` current before the copy) changes nothing: **26.1 -> 26.0 t/s**.  So the per-call device switch is
a real wart but not this cost.

**The staging resource is per THREAD, and that is the fix.**  `probe-pageable-threads.cpp` issues the same
fills from one thread versus one thread per device:

| devices | one issuing thread | one thread per device |
|---|---|---|
| 1 | 15.0 GB/s (41 µs/fill) | 15.1 GB/s (41 µs) |
| 2 | 11.8 GB/s (52 µs) | **20.6 GB/s (30 µs)** |
| 3 | 12.8 GB/s (48 µs) | **29.4 GB/s (21 µs)** |

A single host thread serialises every device's pageable copies through its own bounce; one thread per device
removes the serialisation entirely.  **This is the design's own P6 rule ("one push queue per device")**, and
it means the direct pageable push needs **no pinned host duplicate at all** — so the transport can keep the
RAM property that is the campaign's actual goal (delete the pool's 4.7-9.3 GiB, `PUSH-MLOCK-TIER.md`), with
the residency guarantee coming from the mlock tier rather than from pool slots.

**FIRST attempt: a teardown deadlock in my code, not a driver wedge (see the correction below).**  A
per-device upload thread with "caller waits until the copy has been ISSUED" wedged on its first run: the
process sat with GPUs and CPU idle, 6 threads — main in `futex_do_wait`, two in `kfd_wait_on_events`, and the
two per-device workers parked on their condvars with no job in flight — i.e. the copy was never issued and
nothing was pending (killed after ~2 min; the tree was reverted, nothing was committed).  The standalone
pattern does **not** hang (`probe-pageable-threads.cpp` runs it for 1-3 devices repeatedly), so the difference
is the surrounding state — the waiting thread holds the cache mutex, and the destination is a live arena slot
rather than a private buffer.  **Not diagnosed**; do not re-attempt this shape without (i) a timeout-guarded
handshake that falls back to the caller's-thread copy and logs it, and (ii) a stack trace of the wedge.

**Second attempt wedged again — and the cause turned out to be MINE, not the driver's (RETRACTION, see
below).**

**CORRECTION (same day, after a gdb trace): the "wedge" was a TEARDOWN deadlock in my own code.**  The traces
showed the main thread inside \`exit()\` -> \`_dl_fini\` -> \`__do_fini\` -> \`__cxx_global_array_dtor\` ->
\`std::condition_variable::~condition_variable\` -> \`pthread_cond_destroy\`, blocked, with the two workers parked
in \`uq_worker\`'s \`cv.wait()\`: destroying a condvar that still has waiters is undefined and blocks.  The two
"stuck GPU work" threads I had read as \`kfd_wait_on_events\` are ROCm's own \`AsyncEventsLoop\` threads in their
normal idle state, not evidence of anything.  **So nothing was ever wedged in the driver: the generation had
completed and the process hung on the way out.**  Fix: allocate the per-device queue on the heap and never
destroy it.  With that, **the queue runs cleanly: 115 808 fills issued by the per-device threads, 0 caller
fallbacks, 0 stalled devices** — and it is **performance-neutral: 27.5 t/s against 28.2 without it** (pool
56.0).  So the per-thread-de-contention hypothesis is **not** the explanation in the real path, and the
remaining gap between the probe's 21-30 us/fill and the path's ~107 us/fill is unexplained by the thread, the
stream, or the source (the probe reproduces none of it with a file-backed source, a shared stream, or
per-device threads).

**The one difference left is the DESTINATION**: the probe copies into a plain \`hipMalloc\` buffer, the real
path into an expert-cache ARENA slot (the delivery's arena/VMM machinery).  That is the next test: the same
probe with a VMM-mapped destination.

**Superseded (kept for the record): the second attempt and the reasoning that led to it.**  Re-implemented with
a **bounded**
wait (1.5 s, falling back to the caller's-thread copy) so the arm could not hang on my code.  It froze again
after ~25 s, and the process state rules the handshake out: **the two workers were parked on their condvars
(no job in flight) while two other threads sat in `kfd_wait_on_events`** — i.e. the copies WERE issued and
returned, and their GPU completion never arrives.  A timeout on the host cannot reach that.  Two wedges, two
distinct signatures (the first had nothing in flight; the second leaves work that never completes), both
reproducible, neither tunable from here: **a pageable `cudaMemcpyAsync` issued from a foreign thread is not
safe on this stack.**  The copies must be issued from the thread that owns the device/stream — the same
conclusion DESIGN §6 records for device-initiated requests ("adds a sync"), arriving from the other direction.

**Coalescing is the cheaper lever, and it was already named in this codebase.**  `bounce_acquire` carries a
measured note: the grow-on-demand bounce ring is **deliberately UNCAPPED** and holds **2.2-3.0 GiB of pinned
bounce on the 122B prefill miss path**, and "the fix for the memory is fewer, LARGER fills (coalesce a run of
consecutive missing experts into one pinned read), not a smaller ring" (`wip/host-pool-speed` §16).  That
matters twice over here: it is the same fix as (a) the pageable path's per-call cost (one staging copy per RUN
of consecutive experts instead of one per expert) *and* (b) the maintainer's RAM constraint — a `push` mode
using the bounce as its transport must bound that ring, and the maintainer's bar is **under 2 GiB at parity**.
Coalescing needs no threads, so it carries none of the ordering risk above.

**Implementation shape (original plan, paused):** a per-device upload queue — the fill hands (dst, src, len, stream) to that
device's worker, the worker issues the copy on the device's stream and records an event, and the host thread
that will enqueue the consumer waits for the worker to have *issued* (not completed) it.  Ordering is
unchanged because both the fill and the op land on the same device stream.  The fallback, if that proves
awkward, is a **per-device pinned ring of tens of MiB** (our own memcpy, which measures 1-2 µs per call) —
bounded far below the "under 2 GiB" the maintainer would accept, and 2 orders of magnitude below the pool.

## 3b. What the RAM is actually buying: the pool-size / transport curve at 2 devices (0,2)

The maintainer's constraint is **RAM** (delete the pinned pool if at all possible; a staging buffer under
2 GiB would be tolerated).  So the pool — which IS a pinned staging buffer — was swept by size, on the clean
upstream-x8 pair (`HIP_VISIBLE_DEVICES=0,2`), `--ignore-eos -n 500`, 35B-A3B, `-ncmoe 40`:

| arm | pinned pool | decode t/s | vs pool |
|---|---|---|---|
| A pool FRAC 50 | 4 712 MiB | **55.9** | 100 % |
| B bounce only (`MOE_HOST_POOL_MIB=1`, `PUSH_SRC=pinned`) | ~0 | 41.9 | 75 % |
| C pool 512 MiB | 512 MiB | 42.2 | 75 % |
| D pool 1024 MiB | 1 024 MiB | 42.7 | 76 % |
| E direct page-cache push (`PUSH_SRC=mapping`) | 0 | 28.4 | 51 % |

**Read this carefully, because it is not a transport story.**  C and D are indistinguishable from B: a pool
that does not hold the *hot* set is the same as no pool.  The pool's benefit is all-or-nothing — residency of
the working set — and no transport can substitute for it.  The 25 % gap between A and B/C/D is **the per-fill
host memcpy that a page-cache source always pays**: with 4.7 GiB resident, ~75 % of fills hit a pinned slot and
cost nothing; without it, every fill copies page-cache bytes into the ring first.  `mlock` does **not** remove
that cost — it prevents *reclaim* (the disk read), not the copy, because the bytes stay pageable.

So the RAM question and the transport question are separable, and the answer to each is:

- **Transport: the bounded bounce ring, not the direct CE.**  41.9-42.7 t/s against 28.4, and the direct path
  is neutral once each device has its own issuing thread (§3a) — so this is NOT closed by that lever, and it is
  not closed at all: see `HANDOVER.md`.
- **Residency: `mlock` over the mapping** — the same pages, locked, with **no duplicate**.  That is what
  replaces the 4.7 GiB pool's *guarantee* at a fraction of its RAM, and it is the piece
  `PUSH-MLOCK-TIER.md` designed.

**Under the 4 % parity bar, none of the low-RAM options pass on this band** (75 % and 51 %).  Two things soften
that and must be measured before any decision: (i) this is the **fit-in-RAM** band, where the mode is not meant
to be used — on the not-fitting band every miss is a disk read anyway and the transport difference should
shrink, which is exactly the maintainer's own argument; (ii) the 25 % is *one* number for one config — the
fraction of fills that hit the pool is what sets it, so it depends on FRAC, arena budget and model.

## 4. `GGML_SCHED_DEVGATHER` — irrelevant to decode, decisive for prefill

The gather returns from the scheduler **before** the fill path, so an A/B of anything in the fill path is only
a transport comparison with the gather off.  Dated facts, 3-GPU `-sm tensor`, 5246-token
`prompts/prose-rdna-boosts.txt`, `-n 1`, two reps each (agreement within 1 %):

| arm | `GGML_SCHED_DEVGATHER` default | `=0` |
|---|---|---|
| ctrl (`pool`) | 282.2 / 282.7 t/s | **422.2 / 421.4 t/s** |
| `mapping` (push) | 200.7 / 200.9 t/s | **413.4 / 411.6 t/s** |

- **Decode: the gather never fires.**  It is band-gated (`n_tok <= band` keeps the arena read); every decode
  arm above reports **zero** gather lines (`g_gather_calls == 0`).  Our decode conclusions stand as measured.
- **Prefill: the default mostly hides the path being tested.**  With the gather on, the mapping arm's 29 %
  "prefill loss" is not the transport (the fill census is identical between the arms); with it off the push
  costs **−2 % (noise)**.
- **Separate delivery finding (not the campaign): the default costs 33 % prefill here** — 422 → 282 t/s on
  35B-A3B at FRAC 50.  r207 justified the pool-prefill-gather default on a 122B at FRAC 95 (`ENVIRONMENT.md`
  `GGML_SCHED_DEVGATHER`); this config disagrees.  Tracked as `TODO` #136.

## 5. The device-set gate is UNREACHABLE on a tensor split (increment 3's headline claim)

`moe_cache_report` on the 3-device `mlock` arm:

```
P1 push-src: pinned working set 511 MiB in 907 experts (budget 512 MiB, seeded=664 first_use=243 cumulative,
refusals=0; released=0 release_denied=0 over 3 boundaries)
```

Three boundaries.  Versus **17 037** boundaries on the same binary at 1 GPU (`released=24 release_denied=0`) and
on 3-GPU `-sm layer` (`released=8 **release_denied=528**`).

**Why, from the code:** the boundary event lives in `moe_cache_update_host`, reached only from the scheduler's
host-readback branch (`ggml-backend.cpp:3028`, behind `host_expert_input && node_n_consumers == 1 &&
!ids_computed_here`).  With `-sm tensor` the whole graph is **one Meta split**, so a MoE op's routing ids are
computed *inside that split* → `ids_computed_here` is true → the branch is stood down by the issue-#129 guard
(and the take-over fast path returns before it as well).  The hook therefore runs only at sizing (one call per
device = 3).  With `-sm layer` each device owns whole layers, the host-weight op starts its own split, and the
branch runs.

**Consequence, and it is structural:** the only configuration that gives a host tensor a **device set > 1** is
the tensor split — and that is exactly the configuration where the per-op boundary never fires.  So
(i) the rotation never runs on a split (`released=0`), (ii) `release_denied` can never prove a *cross-device*
denial, and (iii) `push_read_epoch` is refreshed only by the demand fill, which stamps `g_push_epoch-1` and is
therefore never old enough to pass.  The `-sm layer` arm exercises the gate (`release_denied=528`) but each
table's device set is 1 there, so it proves event-completion, not cross-device safety.  **Increment 3's gate is
decorative on the arm it was written for** — the exact FAIL the old script header described, now with a cause.

P2's eagerness has the same problem: **there is no per-op host boundary on a tensor split to hang "one op
early" on.**  Whatever P2 builds must take its ordering from the take-over/arena path or from a device-side
signal, not from `moe_cache_update_host`.

## 6. What changed in the campaign tooling

| file | what |
|---|---|
| `p3-matrix.sh` | one arm of this assessment: fixed config, `GPU`/`SM`/`N`/`EOS`/`LV`/`PROMPT_FILE` knobs, prints t/s + text hash + the census + a GPU-fault check.  Serial runs only. |
| `p3-busy.sh` | samples every card's `gpu_busy_percent` and the llama process's CPU time while an arm runs (`gpu_busy_percent` alone is not a utilization figure — see §3). |
| `probe-pageable-multi.cpp` | the standalone staging probe in §3 (pageable / pinned / pool, 1-3 devices). |
| `p2-gate-3gpu.sh` | **fixed**: stdout (`log.out`, hashed) and stderr (`log.err`, counters) are separate files, so `-lv 4` no longer invalidates the hash and one run per arm suffices; explicit PASS/FAIL verdict including "the boundary hook never ran — the gate is inert on this arm"; nonzero exit on FAIL.  Validated both ways below. |
| `patches/0005-p2-once-only-diagnostics.patch` | the "one message" diagnostics are now **once per process** (§1b).  Append-only so `0001`-`0004` stay as published. |

**Gate validation after the fixes** (3 devices, 500 tokens, `MLOCK_MIB=512`):

| arm | rotate | control | verdict |
|---|---|---|---|
| `-sm layer` | 748 ch `f92b4e201207`, `released=40 denied=528` over **60 357** boundaries | 748 ch, `released=0` | **PASS** |
| `-sm tensor` | 748 ch `f92b4e201207`, `released=0 denied=0` over **3** boundaries | 748 ch, `released=0` | **FAIL (inert)** — exit 1, §5 |

## 7. Pre-existing behaviour this exposes (NOT the campaign)

3-GPU `-sm tensor` + `-ncmoe 40` on this box is **20-21 t/s while 1 GPU gives 55 and 2 GPUs 52** — identical
for `--host-experts pool` (20.6) and `--host-experts pinned` (20.5), reproducible on pristine r209, while the
same 3 GPUs with all experts in VRAM give 138 t/s.  So on this box the 3-device tensor split is **not** a
scaling point for host-resident experts, and the campaign's P4 "then 3 GPUs" arm measures a degraded
configuration.  Tracked as `TODO` #135.

## 8. What this means for the campaign

- **Transport (`MOE_HOST_PUSH_SRC=mapping`) — the SOURCE hop, NOT the push principle.**  **The mechanism is
  proven: parity on one device and on a 3-device `-sm layer` split**, with byte-identical text — pushing
  straight from the reclaimable page cache costs nothing where it is issued one device at a time.  **The one
  open defect is `-sm tensor`**: fills for several devices are issued inside one op and the pageable path
  degrades (2 devices 2.0x, 3 devices 1.26x, §2).  That is a fixable fill-path problem (§3), and it is the
  campaign's next engineering task — not its verdict.  Not promotable **until that is fixed**, because
  `-sm tensor` is a gate arm for every release.  Two things this does not touch: (a) the charter (the host
  issues every copy, no kernel ever reads a host address) is unchanged and is what the delivered pool does
  today, so nothing here argues for device-pull; (b) the band the charter actually targets — the set does
  **not** fit, `pool`/DIO with the disk on the critical path — is **not measured**, and that is where P0's
  1.32 ms cold direct vs ~3.5 ms pool lives.
- **Residency (`MOE_HOST_PUSH_MLOCK_*`): untouched by any of this.**  It pins over the mapping and does not
  change the transport, so the §3 cost applies equally to both arms (`mlock` vs transport-only is a wash, as
  measured).  It is still the RAM win the campaign was after, and its natural home is the pool's transport.
- **Eagerness (P2): needs a boundary source that exists on a tensor split.**  Today's does not (§5).
- **Still owed:** the not-fitting band (P3 in the plan: `pool`/DIO with the disk on the critical path) and the
  ROCm 7.14 correctness arm.  Everything above is 10.1 + a fitting config.
