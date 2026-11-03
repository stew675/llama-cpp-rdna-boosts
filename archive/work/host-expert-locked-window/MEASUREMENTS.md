# MEASUREMENTS — P0 (mechanism) and P1 (churn), 2026-11-03

Box: 1x R9700 (gfx1201), PCIe 5.0 x16, ROCm 10.1.0, kernel 7.2.15-300.fc45, GGUF on `/llm` (xfs), uid 1000
(`RLIMIT_MEMLOCK` soft == hard == 8192 KiB, `max_map_count` 1 000 000, sudo available).  Device pinned
(`HIP_VISIBLE_DEVICES=0`, `AGENTS.md` rule 3).  Logs: `runs/`.

**Everything below is measured, except where it says otherwise.  Two results change the design in
`DESIGN.md`; they are marked ⚠.**

## 1. P0 — the mechanism

### 1.1 `hipHostRegister` is what makes an in-place read legal (and nothing else is)

| arm | setup | result |
|---|---|---|
`pageable` | mapped, populated, **not** registered | **fatal**: `Page not present` in `kern_copy4` |
`base` | registered, populated | read + memcmp identical |

Confirms issue #116 on the current stack: a pageable mapping is not device-addressable, and registration is
the only thing that changes that.  Alias: `hipHostGetDevicePointer` returns the **identity pointer**
(`0x7f95…` == the host pointer), so one pointer serves every device.

### 1.2 ⚠ Registration is an **eager hold**, not just a mapping

`arm absent`: page cache proven **0/425** pages resident through a *fresh* mapping, then
`hipHostRegister` → page cache **425/425**, then the device read succeeds and is data-identical.

`arm pin` (page cache read through a fresh mapping, never the registered one's PTEs):

| step | own PTEs | page cache | VmLck |
|---|---|---|---|
not registered, populated | 425/425 | 425/425 | 0 |
not registered, after `MADV_PAGEOUT` | 0/425 | **0/425** | 0 |
registered, never device-read, after PAGEOUT | 0/425 | **0/425** | 0 |
registered → device read → PAGEOUT | 0/425 | **0/425** | 0 |
`mlock2` + PAGEOUT | 0/425 | 0/425 | **1700 KiB** |

and `arm evicted`: registered + populated + PAGEOUT (**page cache 0/425, proven**) → **the device read
succeeds, data-identical**.  So once a range is registered the driver keeps a reference to the pages and
`PAGEOUT` cannot take them away, yet "registering over absent pages faults them in".

Registration cost is consistent with that reading: 512 MiB **resident** = 12.9 ms (25 µs/MiB, GUP);
512 MiB **cold** = 1.10 s (2.14 ms/MiB, i.e. it read them from disk); re-registering the same range warm =
**2.7 µs**.

**Consequences (both are changes to the design as first written):**

1. **`DESIGN.md` I5 is wrong.**  "Register once per tensor, hold per window" would pin the **whole tensor**
   — the exact state that made the prior campaign conclude "the RAM saving is gone".  The grant primitive is
   **register the window, unregister on evict**.
2. **`mlock` may be unnecessary.**  If registration populates *and* pins, the hold is free with the mapping
   and `RLIMIT_MEMLOCK` (8 MiB here, unraisable as uid 1000) stops being a deployment requirement.  What must
   still be measured before that claim is made: that the driver's hold *survives real reclaim pressure* and
   that `hipHostUnregister` actually **releases** it (the budget is only real if release is real).

### 1.3 `mlock` costs more than the copy it would replace

`arm hold-hot`, 1.66 MiB expert, budget 8 MiB (≈4 experts):

| op | median |
|---|---|
`mlock2(MLOCK_ONFAULT)` | **364.8 µs** |
`MADV_POPULATE_READ` (hot) | 10.3 µs (6.2 µs/MiB) |
`munlock` | 5.8 µs |
`mlock` eager (8 MiB, cold pages) | 72.5 µs |
over-budget `mlock2` | refused in 0.4 µs (`ENOMEM`) |

A hot `mlock2`+populate is ~375 µs per 1.66 MiB — **worse than the pool's 50 µs `memcpy`** and 23x worse than
a registration.  If §1.2 holds, `mlock` is not the primitive.

`MADV_PAGEOUT` over a range that contains a `VM_LOCKED` VMA returns **`EINVAL`** — the eviction side must
page out per chunk, excluding held chunks (which is what the design does anyway).

### 1.4 The grant cost, which is the churn constant

`arm regcost`, one expert blob (1.66 MiB), 16 iterations:

| | register (the grant) | verify (device read) | unregister |
|---|---|---|---|
**hot** (range in page cache) | **16.35 µs** (9.85 µs/MiB) | 182 µs | **0.49 µs** |
**cold** (page cache proven 0/425) | **1166 µs** (0.70 ms/MiB) | — | — |

So a grant is 3x cheaper than the pool's hot `memcpy` (30 µs/MiB) and 3x cheaper than its cold fill
(2.13 ms/MiB), and it *is* the disk read on a cold miss instead of a page-by-page fault inside a `memcpy`.

### 1.5 The hold survives pressure (with a caveat on the control)

`arm survive`: held expert stays **425/425** resident after 4 GiB of anonymous pressure and the in-place read
is data-identical.  **Caveat:** the intended control (an unheld neighbour evicted by the same pressure) was
void — `MADV_PAGEOUT` returned `EINVAL` because it was applied across the held VMA.  The control must be
per-chunk; until it is run, "pressure is real" is **unproven** in that arm.

### 1.6 Granularity

4 expert-granular locks and 1 chunk lock of 8 MiB each produce **+2 VMAs** (adjacent `mlock`ed ranges merge),
~0.7 µs per lock.  So the VMA concern in `DESIGN.md` §2 is **mild for `mlock`** and needs re-checking for
per-window **registration** (a driver mapping may not merge).

### 1.7 Bandwidth

Single 1.66 MiB in-place read+write: 68.8-73.9 µs = 23-25 GB/s useful ≈ **45-49 GB/s of link traffic**,
consistent with the prior campaign's ~52 GB/s aperture measurement.  The sustained-loop figure (49 GB/s
useful) is **not** quoted: it exceeds an x16 Gen5 link and is a measurement artefact (likely GPU-side caching
of the destination / driver staging); re-measure with `rocprofv3` before using it.

## 2. P1 — the churn, priced on a real trace

`sim-locked-window.py` runs `scripts/moe-trace-replay.py`'s existing two-tier policy (the shipped admission /
eviction, unchanged — the window is the pool's *storage*) and prices the same event stream two ways with the
§1.4 constants.  Trace: `wip/host-pool-speed/data/traces/trace-122b-ncmoe48-devremap0.txt.gz` (144 tables,
1867 decode passes), arena 4096 MiB/device, `--cold-pagecache 0.5`.

| window | slots/tbl | held MiB | pool host/tok | window host/tok | delta |
|---|---|---|---|---|---|
2 048 MiB | 7 | 1 888 | 1 425 ms | 450 ms | **-68 %** |
8 192 MiB | 30 | 8 020 | 1 060 ms | 330 ms | **-69 %** |
24 576 MiB | 94 | 24 478 | 499 ms | 146 ms | **-71 %** |
49 152 MiB | 188 | 49 057 | 146 ms | 30 ms | **-79 %** |

**Read this as a ratio, not as a prediction.**  The absolute host µs/token is not yet calibrated (the
measured 122B arm is ~56 ms/token; this model says more), and the dominant term is the L3 tier, whose cost
depends entirely on `--cold-pagecache`.  What the run does establish:

- the window's host cost is **0.2-0.32x** the pool's at every budget, and the gap **widens** as the window
  covers the set (at full coverage the pool is still paying a CE copy per expert consumed; the window pays
  none);
- the advantage is not the fill count (identical event stream) but the **per-grant cost**, which is the
  §1.4 measurement;
- `l3 MiB/token` falls from 1 258 to 78 MiB as the window grows — the disk tier is what a bigger window buys,
  which is the charter's stated pain.

## 3. What P0 established, what it did not

**Established:** the in-place read is legal only with a mapping; a mapping is an eager, silent **hold** of
what it covers; `PAGEOUT` cannot reclaim a registered range; a grant costs 9.85 µs/MiB hot and 0.70 ms/MiB
cold; `mlock` costs 3-20x more than the grant and is probably unnecessary; a pageable read is fatal.

**Not established (P0.5, next):**
1. Does the driver's hold survive **real** reclaim pressure (not `PAGEOUT`)?  The prior campaign's fatal
   fault came from a set larger than RAM; if registration faults in only what it can, some pages must fault
   later.  Register a range larger than free RAM and read a page the registration could not populate.
2. Does `hipHostUnregister` **release** the hold (page cache droppable afterwards)?  Without this the
   "window" is a whole-tensor budget again.
3. What does a registered range cost in **RAM accounting** (`/proc/meminfo`, cgroup `memory.current`)?
4. The §1.5 control, per-chunk, and with the hold released behind the op's back (the residency gate's
   negative control that must fault).

## 4. P0.5 — the release question, and the verdict

**Question:** the design needs a *releasable* hold (a window that slides).  Is there one?

**Answer: no.  `hipHostRegister` takes a hold that lives for the life of the process, and no user-space call
tested releases it.**  Measured two ways — cgroup `memory.stat:file` deltas (the page-cache tier) and real
reclaim pressure (3/4 of `MemAvailable` as anonymous pages), never `MADV_PAGEOUT` alone:

`arm scratch` (a 1 GiB scratch file on `/llm`, so the pages cannot already be cached; window 512 MiB-1 GiB):

| step | cgroup `file` |
|---|---|
baseline (empty scratch) | 826.5 MiB |
window registered + populated | 1854.4 MiB (**+1027.9** for a 1024 MiB window) |
after `hipHostUnregister` | unchanged |
after `munmap` + 300 ms | **1863.5 MiB — unchanged** |
after **67 GiB** of anonymous pressure | **1863.5 MiB (delta from baseline +1037.0)** |

and every registration flag behaves the same (`LW_REG_FLAGS=ro|portable|mapped`), all `+~512 MiB`, all
`NOT RELEASED`.

`arm release` / `release-held` (control): after `hipHostUnregister` + PAGEOUT (page cache proven 0/425) the
device read **still succeeds**; and `arm remap` shows the device mapping *does* die with the VMA — an
unregistered window re-mapped at the same address faults.  So the lifetime is:

> **the VMA, not the registration** — but the pages themselves stay referenced (unreclaimable) either way.

`arm big` under `sudo systemd-run --scope -p MemoryMax=4G` registering **6144 MiB**: `hipHostRegister`
returns **ok** (5.8 s), `memory.current` pegs at the 4095 MiB limit, only 2/16 sampled windows are fully
resident, and the device read of the last (non-resident) window still **succeeds**.  So the driver neither
refuses an over-budget registration nor faults — it silently fronts the read (an unbounded, unguaranteed
read path, not a residency guarantee).

### Verdict

**The campaign's stated goal is blocked by the driver, not by the code.**  A bounded *sliding* window cannot
exist, because the hold cannot be released: registering a window makes its bytes permanently non-reclaimable
for the process, so the footprint is the whole registered set.  That is `pinned` with a longer load and no
`RLIMIT_MEMLOCK` — and it cannot help the not-fitting case, which is the only case `pool` exists for.

**Do not build P2/P3 for the sliding window.**  What survives:

1. **The pool's copy is not waste.**  It is the only *releasable* residency primitive on this stack: a pinned
   slot is bounded and reusable, and releasing it costs nothing because it is an ordinary `cudaMallocHost`
   buffer.  The bounce exists precisely because the alternative cannot be given back.
2. **A static, bounded registered subset is viable and is the honest pivot**: register a *fixed*,
   heat-selected set of experts once (bounded by a budget) and keep the pool for the rest.  That bounds
   memory (registered set + pool budget), removes the copy for the registered set, and needs no release.  It
   is the host-RAM analogue of `wip/moe-static-hot-tier/` (the VRAM static tier) and it uses the existing
   heat map.  It is a different campaign, not this one.
3. **A re-examination note for the archive**: the withdrawn `beta/0016-host-experts-window` was withdrawn
   because a registered range faulted fatally on a set larger than RAM.  That fault did **not** reproduce
   here for a partially-registered range (the driver fronted the read).  If anyone revives the whole-set
   `window` mode, re-run that case first — but it is not this campaign's business and it does not give a
   bounded footprint either way.

Cost of finding out: one session, 12 probe arms, ~30 runs.  That is the campaign paid for itself.

## 5. The alternative the measurements point at: let the copy engine read the page cache

Asked after the verdict: instead of a pinned slot, can the HOST push an expert straight out of the pageable
mapping (a CE copy from page-cache pages)?  `arm dmapage`, 1.66 MiB blob, 16 iterations, same box:

| source of the CE copy | hot (page cache) | cold (≥90 % paged out) |
|---|---|---|
**pageable mapping (no pin, no registration)** | **65.8 / 93.2 µs** (2 runs) | **1 321 µs** (1.3 GB/s) |
registered mapping (the permanent hold) | 42.9 / 48.5 / 104.2 µs | not measured |
pool: `memcpy` into a pinned slot + CE | 137.3 µs (prior campaign: 50.1 + 66.4) | ~3.5 ms (17.0 ms/8 MiB, prior campaign) |

So a direct pageable push is **~1.5-2x cheaper hot and ~2.7x cheaper cold** than the pool's
memcpy-into-a-slot, and it needs no pin, no registration and no hold.  All copies verified byte-identical.

**Why the delivery does not already do it — and this is the whole answer: correctness, not speed.**  A
pageable `cudaMemcpyAsync` is staged through a **shared internal bounce** in ROCm and *tears under
back-to-back use*: the delivery measured **nondeterministic greedy text** from it, which is why
`moe_cache_upload_pageable()` exists (`moe-expert-cache.cu:5509`), why `ggml-cuda.cu:6254` routes a
pool-backed master through the pinned bounce instead, and why a stray pageable upload logs "this is slow and
may tear".  This probe synchronises every copy, so it never exercises the overlap that tears.

Also documented and relevant: a **2-D** pageable-source H2D is worse than slow — it can fault
(`Page not present`) and crashed the `-sm tensor` host-expert load (`ggml-cuda.cu:2152`).  The expert fill is
1-D, so that is not the blocker here; the tearing is.

**Conclusion for the next campaign:** the tractable question is not "remove the copy" but "can a pageable
(or registered) push be made *safe* under overlap" — with a **correctness gate first** (same-seed greedy text
under a back-to-back fill stress, `check-gpu-faults.sh`), because the failure mode is silent corruption, and
the upside is 1.5-2.7x on the fill path.  If it cannot be made safe without a per-device staging buffer, then
that buffer *is* the pool and the copy is load-bearing — which is the current design, for the reason now
measured rather than assumed.

## 6. Does the "tearing" claim hold? (it does not reproduce for a 1-D push)

§5 was going to leave the pool's copy standing on a correctness claim: "a pageable `cudaMemcpyAsync` … tears
under back-to-back use (measured: nondeterministic greedy text)" (`ggml-cuda.cu:6255`, and the prior
campaign's `PROTOTYPE.md:47`).  So the claim was tested directly.  `arm tear` / `arm tear2d`: N copies issued
back-to-back with **one** sync at the end (the overlap an async fill actually has), from one host thread over
S streams, every copy verified against its own source, and the wrong bytes identified when they arrive.

| source | copies x blob | in flight | streams | iterations | result |
|---|---|---|---|---|---|
pageable | 8 x 1.66 MiB | 13 MiB | 2 | 100 | **0 torn** |
pageable | 64 x 1.66 MiB | 106 MiB | 2 | 20 | **0 torn** |
pageable | 16 x 8 MiB | 128 MiB | 4 | 12 | **0 torn** |
`mlock`ed, unregistered | 4 x 1.66 MiB | 6.6 MiB | 2 | 100 | **0 torn** |
`hipHostRegister`ed | 16 x 8 MiB | 128 MiB | 4 | 12 | **0 torn** |
`cudaMallocHost` (the pool's own source) | 8 x 1.66 MiB | 13 MiB | 2 | 100 | **0 torn** |

2-D pageable (`hipMemcpy2DAsync`, 425 x 4096, the path the warning names): 8 copies x 1.66 MiB, 2 streams →
**0 torn, no fault**, but **376 µs per 1.66 MiB (4.6 GB/s)** vs 66-93 µs for the same bytes 1-D, i.e. ~4-5x
slower — the "pathologically slow" half of the warning confirms, the "can fault"/"tears" half does not.

**What the three hypotheses in the question turn out to be, answered:**

- **"does locking first stop it"** — no difference, but the question is moot: the plain pageable case did not
  tear either, so there is nothing for the lock to fix.  (And the driver is not told about `mlock`; only
  `hipHostRegister` changes its path.)
- **"what if each push is one-shot"** — one-shotness is not the variable; *concurrency* is, and 106-128 MiB in
  flight over 2-4 streams is clean.  These loops also re-read the same source regions 20-100 times, so
  re-reading is not a factor either.
- **"re-read requires another request"** — not needed: repeated pushes from the same page-cache region verify
  clean, every iteration.

**So the claim needs re-sourcing before it is used to justify the copy.**  It has no dated record in the
delivery (no ROCm line, no reproduction), it does not reproduce here for the 1-D CE push on ROCm 10.1, and
the code's own warning is attached to the **2-D** loader path — where the failure it describes ("crashed the
`-sm tensor` host-expert load", `ggml-cuda.cu:2152`) is a different symptom (a fault) from the one a fill
would hit.  Two candidate origins remain, and a microbench cannot separate them: (a) ROCm **7.14** (the
delivery ships both lines), and (b) the **async-input ring** (`GGML_CUDA_ASYNC_INPUT`), whose hazard is a
shared *destination* ring, not a source bounce.

**This is now the most promising direction the campaign has produced**: if the tearing claim is 7.14-only or
ring-only, then the pool's `memcpy` is removable for 1-D fills on 10.1 — 1.5-2x hot and 2.7x cold on the fill
path, with no pinning, no registration, no window and no release problem.  That is "the pool without the
bounce", reached by deleting the copy rather than the mapping.  It wants its own campaign, and its first
deliverable is reproduce-or-refute in the REAL path (the loader/scheduler, not a microbench), on both ROCm
lines, with a same-seed greedy gate and `scripts/check-gpu-faults.sh`.

## 7. P0.6 — the `mlock` mirror (added for `wip/host-expert-eager-push/`, 2026-11-03)

The successor's design replaces the pool's pinned duplicate with a **pinned working set `mlock`'d over the
mapping**, so the one property it needs is the opposite of the registration's: a hold that can be *released*.
`arm mlockram` (new) measures it on a scratch file, with residency read through a **fresh** mapping and the
pressure **held** while measuring (an allocate-then-free loop leaves the kernel no reason to reclaim — that
trap produced a false "not released" on the first cut of this arm, and on the earlier `arm scratch` reading).

| step | 4 MiB window (unprivileged) | 1024 MiB window (root) |
|---|---|---|
`mlock` + populate | 1024/1024 | 262144/262144 |
67 GiB anonymous pressure, held | **1024/1024** | **262144/262144** |
`MADV_PAGEOUT` while locked | refused (`EINVAL`) | refused (`EINVAL`) |
`munlock` | 12.4 µs (3.09 µs/MiB) | 831 µs (0.81 µs/MiB) |
after `munlock`, pressure still held | **0/1024** | **0/262144** |

`mlock` eager: **48.7 µs/MiB** (~50 ms/GiB) — ~4.6x cheaper per MiB than the `mlock2(ONFAULT)`+populate pair
measured in §1.3 (225 µs/MiB), so eager is the primitive to use.

**Verdict: `mlock` is a releasable, enforceable residency knob — the property `hipHostRegister` does not
have.**  The design that uses it is `wip/host-expert-eager-push/PUSH-MLOCK-TIER.md`.  Deployment requirement:
`RLIMIT_MEMLOCK` (8 MiB here, unraisable as uid 1000) → `LimitMEMLOCK=` / `CAP_IPC_LOCK`.
