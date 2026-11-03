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
