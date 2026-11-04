# P4 — ROOT CAUSE FOUND: the `-sm tensor` loss is the PAGEABLE STAGING per fill copy (2026-11-04)

> **This closes the open question in [`HANDOVER.md`](HANDOVER.md) §5.**  The mechanism was never in doubt:
> the multi-device loss is **host time inside the copy call** (the record already placed it there), and what
> was missing is *why* the copy call costs 107-191 µs.  It is not the bytes, not the geometry, not the
> destination, not the device context, not the queue: **the driver stages every pageable upload, and the
> staging is a per-CALL cost, so a tensor split (which halves the copy size and doubles the call count) pays
> it twice as often per byte.**  Measured in situ, at the real geometry, in one run:
>
> | arm (2 devices, 0,2, n=300) | 1-D pageable fills | 2-D/sliced fills (pinned ring) | host time in copies | decode |
> |---|---|---|---|---|
> | `MOE_HOST_PUSH_SRC=mapping` | 90 636 × **141.0 µs** | 18 995 × 1.3 µs | **12.82 s** of 27.7 s | 21.5 t/s |
> | `MOE_HOST_PUSH_REGISTER_MIB=20000` | 90 636 × **0.3 µs** | 45 314 × 0.5 µs | **0.052 s** | **71.6 t/s** |
>
> The fix is `hipHostRegister` over the **read-only model mapping, in place**: no copy, no duplicate, and the
> fill becomes a native transfer.  Cost of the pin: measured, below (§5) — **`hipHostUnregister` does not
> release it while the mapping lives**, so the in-place form is a permanent pin and the *releasable* shape is
> a private registered **window** (`unregister + munmap` does release; §5).

Measured on gfx1201 x2 (R9700, `HIP_VISIBLE_DEVICES=0,2`, both upstream x8), ROCm 10.1.0,
`~/bin/build-llama-rocm` (`build-rocm`), `Qwen3.6-35B-A3B-UD-Q4_K_M`, `-ncmoe 40 --host-experts pool`,
`MOE_EXPERT_CACHE_MIB=2048`, `MOE_HOST_POOL_FRAC=50`, `--seed 42 --temp 0 --ignore-eos -n 300`, serial runs.
Tree: `p2-eager-push-r209` + campaign patches `0001`-`0007` (`0007` is new: the registered source + the
geometry split in the attribution probe).

---

## 1. What a tensor split actually does to the fill (the missing fact)

`GGML_CUDA_CACHE_GEOM_DEBUG=1` prints one line per table.  At 2 devices the 240 tables come in exactly three
shapes (counts from a full run):

| tables | `expert_bytes` | `host_bytes` | `src_off` | `host_pitch` | `split_axis` | per-fill copy |
|---|---|---|---|---|---|---|
| 80 | 294 912 | 589 824 | 0 | **0** | 1 | **1-D**, 288 KiB |
| 80 | 294 912 | 589 824 | 294 912 | **0** | 1 | **1-D**, 288 KiB |
| 37 | 360 448 | 720 896 | 0 | **352** | 0 | **2-D**: 2048 rows × **176 B** |
| 37 | 360 448 | 720 896 | 176 | **352** | 0 | **2-D**: 2048 rows × 176 B |
| 3 | 430 080 | 860 160 | 0 / 210 | 420 | 0 | 2-D: 2048 rows × 210 B |

So a tensor split gives `ffn_gate_exps`/`ffn_up_exps` a **contiguous 1-D slice** (`split_axis=1`, packed) and
`ffn_down_exps` a **strided 2-D slice** (axis 0 split: 2048 segments of 176 B at pitch 352).  One device and
`-sm layer` are 1-D everywhere — which is exactly why they were at parity and `-sm tensor` was not: not
because of the split *per se*, but because the split changes the copy the fill issues.

**And the 2-D table is not the expensive one.**  The old push gate required `host_pitch == 0 && n_slices == 0`,
so the 2-D down fills fell through to the pinned bounce ring — and they cost **1.3 µs/copy**.  The whole loss
was in the 1-D gate/up fills.  That inverted the campaign's working assumption (and is the opposite of what
the same code's own comment predicted, and of the earlier single-GPU observation that 2-D was the slow case —
see §6).

## 2. The cost is the pageable STAGING, and it is per CALL, not per byte

In situ, `MOE_HOST_PUSH_PROBE=1` (patched in `0007` to split the H2D time by copy geometry):

```
mapping (pageable)  P1 probe: 135950 fill copies, 12819977 us host total (94.3 us/fill); H2D 109631 in 12804960 us
                    P1 probe geometry: 1-D 90636 copies in 12779519 us (141.0 us/copy, 99.8% of the H2D time);
                                       2-D/sliced 18995 copies in 25441 us (1.3 us/copy, 0.2%)
registered          P1 probe: 135950 fill copies, 51511 us host total (0.4 us/fill);   H2D 135950 in 51511 us
                    P1 probe geometry: 1-D 90636 copies in 30857 us (0.3 us/copy, 59.9%);
                                       2-D/sliced 45314 copies in 20654 us (0.5 us/copy, 40.1%)
```

Same bytes (census identical between arms: `37221.8 MiB` H2D, `h=0.7907`, `fills=122510`), same destination
(the slab/VMM arena), same stream, same thread, same call count — the **only** difference is whether the
source range is registered.  141 µs → 0.3 µs.

**Why it scales with the device count rather than the bytes.**  The staging cost is dominated by a fixed
per-call component, so what matters is the *number* of calls for a given amount of data.  A tensor split gives
each device its own slice, so for the same expert bytes there are N times as many copies of 1/N the size —
the cost per byte rises with N.  This also explains the two other facts already in the record without any new
mechanism: that 1 device is at parity (one call per expert, and the fixed cost is small next to the 0.57 MiB
transfer), and that a *smaller* pool does not help (the pool's benefit is residency, not the transport).

## 3. The fix: register the mapping in place (`MOE_HOST_PUSH_REGISTER_MIB`)

`hipHostRegister(base, len, hipHostRegisterReadOnly|Portable)` over the table's host master, once at sizing,
page-aligned outwards, shared by every device's table for the same tensor, bounded by a MiB budget
(`patches/0007`).  Nothing is copied or moved: the page cache keeps the only copy of the bytes, and the driver
is simply told they are pinned, so the fill becomes a native transfer, for **every** geometry (1-D, strided
2-D, fused multi-slice) — the old 1-D-only restriction is lifted for a registered master because the staging
is what made the other geometries unsafe/slow.

All arms below are the same build, `--ignore-eos -n 300`, `--seed 42 --temp 0`, serial, control = `pool`:

| config | ctrl (`pool`) | registered in-place push | delta | text |
|---|---|---|---|---|
| 1 GPU, `-sm tensor` | 49.0 | **53.5** | +9 % | identical `4ae33f7bd65c` |
| **2 GPU (0,2), `-sm tensor`** | **51.5** (two reps: 51.5/51.5) | **71.6** (two reps: 71.6/71.8) | **+39 %** | identical `231512fc77e5` |
| 3 GPU, `-sm tensor` | 19.4 | 20.5 | +6 % | identical `4ae33f7bd65c` |
| 3 GPU, `-sm layer` | 50.5 | 54.3 | +8 % | identical `4ae33f7bd65c` |

- **Correctness: byte-identical greedy text in every pair** (680/681 chars; the 2-device answer differs from
  the 1/3-device answer, which is the documented per-device-count reference rule, not a campaign difference).
- `scripts/check-gpu-faults.sh`: clean on every run.
- Prefill also improves where it is measured (2 devices 36.6 -> 72.0 t/s prompt processing).
- The 3-device tensor arm remains the degraded configuration of `TODO` #135 (20 t/s with the pool too, 138
  with all experts in VRAM); the campaign's relative result there is +6 %, not a scaling point.
- Registration cost at sizing: **977 ms for 18 662 MiB** (~19 GB/s of registration; the whole host expert set,
  240 tables, 0 refusals).  One-time, logged.

### 3a. The campaign's actual goal: **no pool at all** (`MOE_HOST_POOL_FRAC=0`)

The arms above still build the pool (FRAC 50 = 9 331 MiB of *duplicate* pinned RAM), so they do not answer
the RAM question.  With the pool switched off entirely — no duplicate, no bounce ring, the page cache is the
only copy — the registered push does not just hold up, it is the same speed:

| 2 devices (0,2), `-n 300` | pool (FRAC 50) | **registered push, NO pool** |
|---|---|---|
| decode | 51.5 / 51.5 | **71.9 / 71.7** |
| `host pool (L2)` in the log | 9 331.0 MiB pinned | **145.8 MiB** (the minimum, 240 slots) |
| text | `sha=231512fc77e5` | `sha=231512fc77e5` (identical) |
| GPU faults | clean | clean |

So **9.2 GiB of the duplicate disappears and throughput goes up 40 %**, byte-identically.  What replaces it is
the 18 662 MiB of page-cache pages made non-evictable (§5) — the same bytes, once, which is exactly the
campaign's charter ("the bytes should live once, in the page cache"), with the residency guarantee carried by
registration instead of by `mlock` + a copy.  For comparison, the same arm with no pool and no registration
(bounce ring only) is **36.8 t/s**.

## 4. Why the earlier probes missed it (so the next session does not repeat it)

`probe-pageable-multi.cpp` / `probe-pageable-threads.cpp` issued 600 KiB 1-D copies from a *single repeatedly
re-read* window into a `hipMalloc` destination and reported aggregate GB/s.  Two things hid the cost:

1. **They report throughput, not host time in the call.**  Overlapping copies make the aggregate look like
   12-20 GB/s even when each call blocks for ~90 µs on the host; the in-situ number is the one that matters.
2. **The window repeats**, so the pages the driver stages are the same 600 KiB every round.  The real path
   reads a different 704 KiB of a 19 GiB mapping every fill.

`probe-memcpy2d.cpp` (new) fixes both and reproduces it (360 KiB, 2 devices, one issuing thread):

| source | 1-D copy, idle stream | 1-D, 256 MiB D2D queued ahead | 2-D copy, idle | 2-D, queue ahead |
|---|---|---|---|---|
| file mapping, **pageable** | **91.4 µs** host/call | **146.2 µs** | 44.6 µs | 41.5 µs |
| file mapping, **registered** | 22.8 µs | **16.8 µs** | 19.3 µs | 23.8 µs |
| `hipHostMalloc` (pinned) | 18.3 µs | — | 19.7 µs | 25.4 µs |

Three facts fall out of it, all of record for later:

- **A pageable 1-D copy is the expensive one** (91-146 µs), and it is *worse* under a live queue.  A pageable
  2-D copy of the same bytes is 2-3x *cheaper* on this stack (44 µs) — see §6.
- **Registration removes the cost entirely and removes the queue sensitivity** (17-23 µs regardless of load).
- **The destination is irrelevant**: `DST=vmm` (page-reserve + create + map, as the slab arena is) equals
  `DST=hipMalloc` to within noise in every arm.  HANDOVER experiment 1 is therefore **closed negative**.

## 5. The RAM question — the maintainer's `hipHostUnregister` finding REPRODUCES, and the answer is `munmap`

A residency tier that cannot be released is worse than the pool in the band the mode exists for (the set does
not fit).  `probe-pin.cpp` / `probe-pin-free.cpp` measure the pin's lifetime with the only honest signal on
this stack (`VmLck` does not see HIP pins; `mincore` reports page-cache residency): touch the range, ask the
kernel to reclaim it, re-touch, and count `/proc/self/io read_bytes`.

| arm (256 MiB window of a 1.4 GiB file on disk) | re-read after `FADV_DONTNEED`+`MADV_PAGEOUT` | verdict |
|---|---|---|
| control, never registered | 256.0 / 256 MiB | evicted (the method works) |
| registered, **`hipHostUnregister`**, mapping live | **2.0 / 256 MiB** | **still pinned — the pin LEAKED** |
| registered, `hipHostUnregister` + 0.5 s, mapping live | 2.0 / 256 MiB | still pinned (not deferred) |
| registered, **`hipFreeHost`** | `hipErrorInvalidValue` | not applicable — it only takes `hipHostMalloc` pointers |
| registered, `hipHostUnregister` + **`munmap`** | 256.0 / 256 MiB | **released** |
| registered, nothing (control) | 2.0 / 256 MiB | pinned, as designed |
| `hipHostMalloc` -> `hipHostFree` | VMA gone | the documented pair behaves |

So, on this stack:

- **`hipHostUnregister` returns `hipErrorSuccess` and does NOT give the pages back** while the mapping lives —
  exactly the maintainer's finding on the other system.  It only makes the *runtime* forget the registration
  (a copy from the range reverts to the slow pageable path).
- **`hipFreeHost` is not a workaround**: it rejects a registered range.
- **The pin is tied to the VMA.**  `unregister + munmap` releases it completely, and register/unregister
  cycles on the same range work repeatedly (3 cycles measured).

Which splits the design cleanly in two:

| shape | RAM | release | verdict |
|---|---|---|---|
| register the **model mapping in place** (`MOE_HOST_PUSH_REGISTER_MIB`, as measured in §3) | the page-cache pages, pinned, **no duplicate** | **none** — permanent for the process | usable as an **all-or-nothing** fit-in-RAM mode; 18.6 GiB pinned against the pool's 9.3 GiB *duplicate* on the same run, and +39 % throughput.  Not usable where the set does not fit |
| a **private, separately-mapped registered WINDOW** over the model file (`mmap` the range again, register it, DMA from it, then `unregister + munmap` to release) | the same page-cache pages, pinned while the window is open | **`unregister + munmap`** — proven droppable, 40/40 cycles, §5a | **this is the releasable tier the campaign wanted** — no duplicate, native speed, bounded and rotatable.  Needs (i) the existing P2 epoch gate for safe release while fills are in flight, (ii) **coarse** windows (register is 0.6 ms/64 MiB ≈ 104 GiB/s, so a 256 MiB-2 GiB file range that covers many tables, rotated infrequently — a per-expert window would cost about as much as the fill it serves), (iii) a guard gap between windows (adjacent registrations returned `part or all of the requested memory range is already mapped`) |

**Guard in the tree** (`patches/0007`): because the in-place mode cannot release, it refuses to register past
**half of `MemAvailable`** measured at the first registration, and says so once on stderr and in the census
line.  Verified: a 20 GiB budget registers the full 18 662 MiB and keeps 71.7 t/s with `sha=231512fc77e5`, and
a deliberately oversized budget (60 GiB) stops at the cap instead of pinning the box into swap.

`mlock` is **not** a substitute: it keeps the pages but does not make them a DMA source, which is why the P2
`mlock` tier was transport-neutral (measured in P3 §3b) — the campaign's residency idea was right, it just
needed the runtime told as well.

### 5a. PROOF that the pages can be dropped (the requirement for the > RAM band)

The mode exists for configs whose weights do **not** fit in RAM, so "the pin is releasable" is a hard
requirement, not a nicety: a pin that cannot be dropped turns the page cache into unswappable resident memory
and the host starts swapping, which is worse than any transport win.  `probe-window.cpp` establishes it on
the **real model file**, in one measurement per arm (fresh process each: driver state is sticky):

| measurement (real GGUF, 64-256 MiB ranges) | result |
|---|---|
| `hipHostUnregister`, **all four flag sets** (`Default`, `Portable`, `ReadOnly`, `ReadOnly|Portable`), mapping live, then `MADV_PAGEOUT` | **STILL PINNED** — 2-10 MiB of 64-256 MiB re-read.  Not a flags problem |
| `hipFreeHost` instead of unregister | `hipErrorInvalidValue` — only accepts `hipHostMalloc` pointers |
| `hipHostUnregister` **+ `munmap`**, then drop and re-read | **RELEASED** — 256.0 / 256 MiB re-read from disk |
| **40 cycles** of a sliding registered window (mmap -> register -> 8 H2D fills -> unregister -> munmap -> drop -> verify) | **40/40 reclaimed**, 2 560 MiB re-read in total, `MemAvailable` flat (83 219 -> 83 076 MiB; the *unregistered* control drifts the same way) — **no accumulation** |
| the delivered shape: a **long-lived model mapping A** + a registered **second mapping B** of the same pages, B released | A's pages **RECLAIMABLE** (256/256 MiB re-read through A) and A's data **identical** by checksum |
| cost of a window | register **0.6 ms / 64 MiB** (~104 GiB/s on resident pages), unregister ~0.0 ms, plus `munmap`+drop; i.e. a 2 GiB window re-registers in ~20 ms |

So the pin's lifetime is **exactly the VMA's**, and that is the design rule:

> **Never register the model's own mapping.  Register a private, separately-`mmap`'d window over the same
> file range, and release it with `hipHostUnregister` + `munmap`.**

That shape is releasable, keeps the page cache reclaimable (proven end-to-end through a second, long-lived
mapping of the same pages), needs no duplicate, and keeps the native-transfer speed — the copy source is just
another VA range over the same page-cache pages.  Window granularity should be **coarse** (file ranges of
256 MiB-2 GiB covering many experts/tables, not per expert: a 704 KiB window costs ~6-10 us of registration,
i.e. as much as a fill).  What it needs from the delivery is the machinery the campaign already designed for
`mlock` rotation: the P2 per-device epoch gate, so a window is only unmapped once no device can still be
reading it, plus the loader's `path`/`offs` plumbing (`moe_cache_set_host_source`, the same data the DIO tier
uses) to know which file range a table lives in.

## 5b. The over-commitment test (a 100 GiB file against 91 GiB of RAM), and what the page cache does

The > RAM band is where a pin that cannot be dropped would be fatal, so `probe-overcommit.cpp` runs the
production shape at scale: a **100 GiB file**, read in windows that are **registered, DMA'd from, and
released**, cycled several times in **one process**, with a memory watchdog.  Two read modes, because they are
two bands of the delivery: `READ=mmap` (read through the page cache, register a private mapping of the same
pages) and `READ=dio` (`O_DIRECT` into an anonymous window buffer -- the page cache bypassed entirely).

**Read mode `mmap`, 20 GiB windows, 3 loops = 300 GiB read:**

| loop | read from disk | per-window register | DMA | verified droppable | MemAvailable | swap |
|---|---|---|---|---|---|---|
| 0 / 1 / 2 | 101 674 MiB each (1 % from cache) | 31-33 ms per 20 GiB | 746 ms (28.8 GB/s) | **15 / 15 windows** | 84 073-84 174 MiB (flat) | 447-449 MiB (**flat**) |

**Read mode `dio`, 8 GiB windows, 3 loops:** every read from disk by design (0 % cache), 138-197 ms to
register 8 GiB, 299 ms DMA at 28.7 GB/s, buffers released every window (`MemAvailable` 83 840-85 637, flat),
9/9 windows verified.

So: **a 100 GiB working set on a 91 GiB box cycles three times with the pin released every window, no swap
growth, no OOM, no lockup, and every released window verified re-readable from disk.**  That is the safety
property the mode needs, and it holds in both read modes.

### The page cache DOES serve the reads again (and it converges, it does not snap)

The same harness with a working set the cache can hold shows the warm behaviour directly -- 3 x 8 GiB, full
cycle (register -> 8 GiB DMA -> unregister -> munmap) x 3 loops:

| loop | read from disk | per-window read | per-window register | DMA |
|---|---|---|---|---|
| 0 (cold) / 1 / 2 | 0 MiB (the regions were already resident) | **13-21 ms per 8 GiB** (vs ~4.3 s cold) | **12.8-14.2 ms per 8 GiB** | 299-303 ms (28.75 GB/s) |

**72 GiB read in 3.3 s**, 9/9 windows released and verified droppable, `MemAvailable` flat.  Releasing the
pin does not discard the page cache: reading and re-registering resident pages is ~600 GB/s.

**With a 50 GiB region permanently pinned** (`PIN_MIB`, registered once and never released) and a 15 GiB
cycling set (3 x 5 GiB) on top, 5 loops:

| loop | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| from the page cache | 40 % | 52 % | 67 % | 83 % | **87 %** |

per-window reads fall from >1.1 s to 225-654 ms, every window still released and verified droppable, and
`MemAvailable`/swap stay flat -- so a large permanent pin does **not** stop the cache from serving ordinary
reads, it just leaves less room.  The same test **without** the pin at the same starting state is the same
shape (36-67 % early, converging), i.e. **this is the host's cache behaviour, not the pin.**

Two host facts worth writing down, because they make this box look broken and are neither rare nor HIP's:

- **`/proc/meminfo` does not account these pages usefully here.**  `Cached` sits at ~32-45 GiB and
  `MemAvailable` stays ~85 GiB while 24 GiB of mapped file pages are demonstrably resident (second pass at
  463 GB/s) and a 50 GiB registration is live.  Do not judge residency from `Cached`/`MemAvailable` on this
  box -- use `read_bytes` and the timing of a second pass.
- **The cache converges over minutes, not instantly** (btop shows the same slow growth), and the practical
  ceiling for a file working set is ~24-45 GiB even with ~50 GiB free.  6 x 4 GiB is 100 % warm on the second
  pass; 6 x 8 GiB (48 GiB) is 100 % cold; the same 15 GiB set climbs 40 -> 87 % over five loops.
  `lru_gen/enabled = 0x0007` with **`min_ttl_ms = 0`** is the most likely tunable here (upstream default
  1000 ms); it needs root to test.

**One harness defect worth recording, because it cost 12 minutes of wall clock and looked like a kernel
problem:** walking a file mapping with one load per *page* depends on readahead (the kernel maps ~64 KiB per
fault).  Under memory pressure the readahead allocations fail and the walk degenerates to one 4 KiB read per
fault -- **~25 MB/s measured**.  `read_mapping()` now issues `MADV_WILLNEED` per 256 MiB chunk and reports
progress, which restores ~2 GB/s cold and >50 GB/s when resident.

### Registering a 50 GiB range: works, and it is what makes the pages resident

| test | result |
|---|---|
| `hipHostRegister` of 1 / 4 / 8 / 16 / 20 / 24 / 32 / 40 / **50 GiB** in ONE call | **OK** at every size (fresh process) |
| 50 GiB registration, pages already in page cache | **0.84 s** (~60 GB/s) |
| 50 GiB registration with **no host read first** (`PIN_READ=0`) | **OK** -- registration faults the pages in itself, so the host does **not** have to read them first |
| register **after** `MADV_WILLNEED`-driven readahead, same process | one **`hipErrorInvalidValue`** observed; the retry succeeds.  **Register-first is the robust order** |

## 5c. What `--host-experts pinned` actually is (and why the registered path replaces it)

The question the next design phase turns on: is `pinned` the same mechanism, just unlimited?  **No.**  From the
tree:

- `llama_model_loader::host_expert_buft()` returns `ggml_backend_dev_host_buffer_type(dev)` for
  `LLAMA_HOST_EXPERTS_MODE_PINNED` (`src/llama-model-loader.cpp:56-68`), i.e. the `ROCm_Host` buffer type, whose
  allocation is `cudaMallocHost` = **`hipHostMalloc`** (`ggml-cuda.cu:4203-4235`) -- a **second, complete copy**
  of the host expert set, into new pinned pages, loaded from the GGUF.
- There is **no size limit**: `get_max_size` is **4 GiB per buffer**, so the loader splits the set into
  several pinned buffers, and the only bound is available RAM.
- It is **not** `hipHostRegister`.  The tree *has* the in-place call --
  `ggml_backend_cuda_register_host_buffer()` -> `cudaHostRegister(buffer, size, Portable|ReadOnly)`
  (`ggml-cuda.cu:8660`) -- and it is exposed through `get_proc_address`, but **nothing in the tree calls it**.
- Failure is **silent and expensive**: if `cudaMallocHost` fails, the allocation falls back to
  `ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size)` -- **pageable** memory.  A pageable
  master has no device alias, so the in-place cold read stands down and every miss becomes a fill.  Measured
  by forcing it (`GGML_CUDA_NO_PINNED=1`, 2 devices, `-n 300`):

| `--host-experts pinned` | decode | cold read | fills | H2D traffic |
|---|---|---|---|---|
| allocator AVAILABLE | **71.6-72.1 t/s** | `cold=uva`, 125 286 in-place reaches (20.7 %) | 31 874 | 9 680 MiB fills + 38 075 MiB cold |
| allocator DISABLED (pageable fallback) | **36.0 t/s** | `cold=uva`, **0 reaches** | 124 350 | 37 781 MiB fills |

  So "no limit" is exactly the problem: on a box that cannot supply the whole duplicate, the mode degrades by
  half **without an error**, and nothing tells the user which of the two states they are in.

**Performance parity, RAM not parity** -- all 2 devices, `-n 300`, clean runs, byte-identical text
(`sha=231512fc77e5`) across every arm:

| arm | decode | what the host expert set costs in RAM |
|---|---|---|
| `--host-experts pinned` (fully pinned) | 71.6 / 72.1 | **the whole set duplicated in pinned pages: 18 662 MiB, never released** |
| `--host-experts pool` (bounded pool) | 50.7 / 51.4 | 9 331 MiB pinned duplicate + page cache |
| `--host-experts pool` + `MOE_HOST_PUSH_REGISTER_MIB` (registered in place) | **71.5 / 71.7 / 71.9** | the page-cache pages, registered in place -- **no new RAM** |
| registered in place + `MOE_EXPERT_CACHE_COLD_UVA=0` (no device-pull) | 71.6 | as above |

**So `pinned` is "the right transport at the wrong RAM cost", and the registered mapping is the same
performance with no duplicate at all.**  That is the design statement: the campaign is not asking for a new
transport, it is asking `pinned` to stop copying the weights.  Registration also keeps the charter's
"never device-pull" available at no cost (71.6 with the cold read off, against 71.6 with it on).

Two side notes for whoever implements it:

- Registration needs the model mapping to exist (`use_mmap`); a non-mmap load has no page-cache master to
  register, so `pinned`'s allocation stays the fallback there.
- A pageable fallback must be **loud** (the existing `GGML_LOG_DEBUG` in `ggml_cuda_host_malloc` is invisible
  at the default log level), otherwise a user can silently lose half the throughput.

## 5d. Prefill: registered wins at 2 devices (+34 %), loses at 1 device (-22 %) -- and why

Prefill is a different animal from decode here, so it is measured separately: `prose-rdna-boosts.txt`
(5 246 tokens), `-n 1`, `-sm tensor`, 2 reps each (`prompts/README.md` has the prompt's sha256).

| 2 devices (0,2) | prefill | vs pool | decode (`-n 300`) |
|---|---|---|---|
| `pool` (ctrl) | 708.5 / 708.4 | -- | 50.7-51.4 |
| `pinned` (`hipHostMalloc` duplicate) | 708.8 / 708.2 | 0 % | 71.6-72.1 |
| **`pool` + registered in place** | **951.0 / 950.0** | **+34 %** | **71.5-71.9** |
| registered + `MOE_EXPERT_CACHE_COLD_UVA=0` | 951.9 / 950.0 | +34 % | 71.6 |
| registered + `GGML_SCHED_DEVGATHER=0` | 950.8 / 950.5 | +34 % | -- |
| registered, **pool off** (`MOE_HOST_POOL_FRAC=0`) | **953.0 / 958.9** | **+35 %** | 71.7-71.9 |

| 1 device (GPU 0) | prefill |
|---|---|
| `pool` | 615.6 (with `GGML_SCHED_DEVGATHER=0`: 495.7) |
| `pinned` | 614.7 |
| **registered in place** | **477.7** (-22 %) |

So the answer to "what are prefill speeds like with registered" is **much better at 2 devices and worse at 1
device**, and the two cases are explained by *which device-pull path is available*, not by the copy bytes (the
census is identical between the arms: `fills=18596`, `arena fills=5156`, `H2D 1563.3 MiB`, `cold=0`).

### The mechanisms, and the gates that differ on a tensor split

1. **The scheduler's device gather** (`GGML_SCHED_DEVGATHER`, `ggml-backend.cpp:3351`).  Three-valued:
   default `1` = *"pool-backed tables gather"*; `0` = kill switch; `1`/`force` = mode 2, the historical A/B
   arm that also gathers for a **pinned** master -- and the comment there records that mode 2 *"costs a pinned
   master ~47 % prefill at two devices"*, which is exactly what the numbers above show from the other side
   (a pool master 708 vs the registered/pinned path 950 at 2 devices).
   At **1 device the gather is worth +24 % to the pool arm** (495.7 -> 615.6) and a registered/pinned master
   cannot use it: `force` measured 477.5, i.e. no recovery.  That is the bulk of the 1-device gap.
2. **The cold in-place read** (`MOE_EXPERT_CACHE_COLD_UVA`), i.e. the GPU reading host memory in place
   through the UVA alias.  Its two gates are **not** the same:
   - the *arena overflow* cold read (`table_cold_ok`) works per device slice (`cold_safe` for a 2-D strided
     slice; not for a fused table);
   - the *pool* cold read (`pool_cold_slot_locked`) is **single-device only** -- it returns -1 when
     `ggml_cuda_info().device_count != 1`, by construction (the pool is process-wide and holds whole experts).
   Measured: at 2 devices the registered arm served **125 286 reaches cold (20.7 %)**; at 1 device the prefill
   runs served **0** cold reaches.  Turning it off (`COLD_UVA=0`) changes nothing at either device count, so
   it is not what makes the difference -- but it *is* the thing that only exists on a split.
3. **The per-op host boundary** (`moe_cache_update_host`) runs on 1 device and on `-sm layer`, and **never** on
   a tensor split (3 boundaries vs 60 357 -- P3 §5): the whole graph is one Meta split, so the pool's per-op
   accounting, the gather decision cadence and the campaign's epoch machinery are all inert there.
4. **Prefill staging** (`moe_cache_stage_table`): a pinned master stages with ONE kernel launch; a pool-backed
   master uses the two-tier snapshot plus per-expert copies.  In these runs its counters stayed 0 -- the
   prefill went through the demand fills, not the staging path -- so it is not the explanation here.

### Why 1 device loses: REGISTRATION ITSELF, not the memory's provenance (and it is not the copy calls)

The discriminator that settles it -- the SAME master, with and without `hipHostRegister` over it (1 device,
prefill, `-n 1`, cached-on):

| arm | prefill |
|---|---|
| `pinned` (a `hipHostMalloc` master, as today) | **616.5** |
| the same master, **also registered** | **476.9** |
| `pool` + a registered file mapping | 475.9 |

So the trigger is **the range being registered**, not where it came from -- which also answers "why should
`hipHostMalloc` be faster than `hipHostRegister`": they are different *driver objects*.  `hipHostMalloc` hands
the driver its own allocation, which it keeps mapped for device access and registered with the copy engines;
`hipHostRegister` pins *user* pages and the driver has to treat that range as a user pointer.  The
microbenchmarks say the *copy call* is async in both cases (1.2 us from a registered mapping against 3.4 us
from the same bytes via `hipHostMalloc`, and 15.0 vs 15.2 GB/s sustained), so this is not "registered copies
are synchronous".

**Everything that has been excluded as the cause** (1 device, prefill, `-sm tensor`, 35B-A3B):

| excluded | measurement |
|---|---|
| the fill copies' host cost | `MOE_HOST_PUSH_PROBE`: **4 994 copies in 2 378 us** (0.5 us/copy, 2-D 0) -- the fills are free |
| the copy API / route | `MOE_HOST_PUSH_SRC_COPY=alias` (D2D from the UVA alias, confirmed engaged by a one-time log) = 477.4 vs 478.7 H2D |
| the destination | `probe-memcpy2d` at 1 device: `DST=vmm` (slab-style) == `DST=hipMalloc` for both sources (15.0-15.3 GB/s) |
| the copy geometry | 1-D 100 % of the fills at 1 device |
| the registration flags | all of `0,1,2,3,4,5,6,7` -> 476-480 (Mapped makes no difference) |
| the scheduler gather | `GGML_SCHED_DEVGATHER=1/force` -> 477.5 (and it *is* worth +24 % to the pool arm) |
| the cold in-place read | `MOE_EXPERT_CACHE_COLD_UVA=0` -> 474.4 |
| the pool's presence / waste | `MOE_HOST_POOL_FRAC=0` -> 475.3 |
| the arena size | 2048/4096/6144 MiB: both arms improve, the GAP stays 2.46 / 2.73 / 2.92 s -> **not per fill** |
| the slab / VMM arena | `GGML_CUDA_SLAB=0` -> 478.1 (unchanged) |
| warm-up / page cache | 3 consecutive reps: 477.7 / 478.6 / 479.3 against 615.9 / 615.9 / 616.2 |
| the model load / placement | slab and buffer lines byte-identical between the arms |
| the expert cache entirely | `MOE_EXPERT_CACHE_MIB=0` -> **all three arms 673.7-674.0 t/s** |

### The mechanism: the ROCm SVM path for `hipHostRegister` (and why it is not a bug to fix)

A web search turned up two documented mechanisms and one of them is the answer:

- **`hipHostRegister` has been an *SVM* operation since ROCm 5.6** (`HSA_USE_SVM` enabled silently):
  [ROCm/ROCm issue #2433](https://github.com/ROCm/ROCm/issues/2433) documents the switch and shows its most
  visible symptom (registration taking 0.7 s -> 14-54 s for the same range, fixed only by `HSA_USE_SVM=0`).
- **AMD's own docs do not promise `hipHostRegister` == `hipHostMalloc`.**  The HIP runtime reference says of
  registration *"on some systems, registered memory may not actually be pinned but uses OS or hardware
  facilities to allow the GPU to access it"*, and the ROCm/AMD lab-notes say registered memory is *"
  treated as `hipHostMallocCoherent`"* -- the **coherent** (not L2-cached) pinned flavour, explicitly
  different from the non-coherent allocation `hipHostMalloc` hands out.

**Measured here, and the result is an ASYMMETRY, not a bug:**

| `HSA_USE_SVM` | 1 device prefill | 1 device decode | 2 devices prefill | 2 devices decode | registration |
|---|---|---|---|---|---|
| default (SVM on, ROCm 5.6+) | **477** | **53.4** (+9 % vs pool) | **949** (+34 %) | **71.8** (+39 %) | 18 662 MiB in 0.46 s |
| `HSA_USE_SVM=0` (legacy path) | 615.7 | -- | 707.2 | 51.2 | **REFUSED**: `registered 0 MiB ... refusals=1`, master stays pageable |

So `HSA_USE_SVM=0` is **not a workaround**: with SVM off the runtime refuses to register the range at all and
the arm silently degenerates to the pool path (that is why it "matches `pinned`" -- it *is* the pool path; the
census shows `host pool h=0.03` and 20 289 pool fills).  **Registration on this stack *is* the SVM path**, and
the same property that makes it a big win on a tensor split (949/71.8) costs 22 % prefill on one device (477).

`pinned` at 2 devices is bimodal in the record (71.6-72.1 twice, 51.2 once): that is the silent per-buffer
pageable fallback documented in §5c (4 GiB buffers; a failed `cudaMallocHost` falls back to pageable CPU
memory with only a DEBUG log).  Treat `pinned`'s 2-device number as "72 when every buffer pinned, 51 when it
fell back".

**So the design answer is a policy, not a fix**: use the registered master where it wins -- **2+ devices,
where it is +34 % prefill / +39 % decode against the pool and no duplicate RAM** -- and keep today's pinned
master (or accept +9 % decode / -22 % prefill) at **1 device**.  Nothing in the public record describes this
particular symptom (a registered source under an expert cache costing a fixed per-token prefill cost on one
device); the *mechanism* is documented, the behaviour is not.

**The shape of what is left**: the penalty is ~0.47 ms **per prefill token** (0.10 s at 533 tokens, 2.47 s at
5 246), it needs the expert cache to be ON, and it needs the master range to be registered.  Whole-process
CPU: user +3.6 s, sys -1.6 s, minor faults +2.5 M, maxrss -7.5 GiB.  It is not the fill *count* (arena sweep),
so the next measurement must localise the per-token cost **inside the op**: per-phase in-situ timers
(`access_locked` vs the remap/map upload vs the hook total) in the prefill band, which is a small patch and
the only remaining hypothesis with a cheap test.

**Design consequence, and it is a good one**: `pinned` -> registered is the right refactor, but it must be
gated on **device count** until this is solved -- or the 1-device path must keep the `hipHostMalloc` master
(i.e. register only where it wins: 2+ devices, where the same arm measures +34 % prefill and +39 % decode
against the pool and parity against `pinned`).  At 1 device the registered path is a 22 % prefill regression
*with no decode benefit* (53.5 vs 49.0 measured earlier at 1 device is decode, so: gate on phase or count).

### (superseded) Why 1 device was thought to lose: the registered copies are SYNCHRONOUS

`perf record` of the same 1-device prefill, self time in the main thread:

| arm | host `__memmove_avx512` | `BusyWaitSignal::WaitRelaxed` | `AsyncEventsLoop` |
|---|---|---|---|
| `pool` (616 t/s) | **71.3 %** | 0.8 % | 3.6 % |
| registered (477 t/s) | 27.7 % | **15.6 %** | 23.0 % |

The pool arm burns the host *copying* (its fills are host memcpys, and they are useful).  The registered arm
instead **spin-waits inside the runtime** for each copy -- the transfer is issued synchronously, so the host
is serialised with the DMA.  Recovering the 1-device number therefore needs one of:

- **asynchronous fill issue**: one copy stream (and issuing thread) per device, with the compute stream
  ordered by an event -- the campaign's own P6 rule.  It was measured *performance-neutral* in the pageable
  world (where the loss was the staging, not the wait), which is exactly why it is the right lever now;
- **or** a device gather that works for a registered master at 1 device (mode 2 does not, measured).

Both are small, testable changes.  Whichever is done, the decision to move `pinned` onto registration should
be taken **with the 1-device prefill number in hand**, because 1 GPU is a supported configuration.

## 5d-ter. THE MEASURED MATRIX, with a *real* pinned master (a harness bug had hidden it)

**The harness bug:** `HOSTMODE=pinned` was being passed *after* the script name, so it reached `env` but not
`p3-matrix.sh`'s own `--host-experts "$HOSTMODE"` -- **every earlier "pinned" run in this record was
`--host-experts pool`**.  That is why `pool` and `pinned` measured identically everywhere, and why `pinned`'s
2-device number was bimodal.  Fixed (the variable must be set on the script invocation line; the script now
documents it), and re-measured.  A genuine `hipHostMalloc` master works at this size because the loader
allocates **per tensor** (144-176 MiB), and chunked allocation was verified independently: 18 584 MiB in 101
chunks of 184 MiB, 1.86 s, **101/101 with a device alias**.

| arm | prefill 1 dev | decode 1 dev | prefill 2 dev | decode 2 dev | host RAM |
|---|---|---|---|---|---|
| `pool` | 616.6 | 49.0 | 707.1 | 51.3 | 9.3 GiB pool + bounce ring |
| **`pinned` (real `cudaMallocHost`)** | **645.9** | **53.5** | **1105.4** | **72.0** | the whole set duplicated, never released |
| registered in place | 478.7 | 53.4 | 951.6 | 71.8 | none (the page-cache pages) |
| cache off: pool / pinned / registered | 673.4 / 686.1 / 672.9 | | 949.4 / 1043.6 / 948.9 | | |

Three things follow, and they change the picture:

1. **Registered == pinned on DECODE at both device counts** (53.4 vs 53.5, 71.8 vs 72.0).  Decode is settled:
   one mechanism, no duplicate.
2. **A real pinned master is the FASTEST prefill transport** (645.9 / 1105.4), so `pinned` is no longer just
   "the duplicate we want to delete" -- it is the performance bar.  Registered trails it by 26 % (1 dev) and
   14 % (2 dev).
3. **That prefill gap is NOT the transfer**: the staging census is identical between the two arms at 2 devices
   (`ok=2574`, `arena=43707.1 MiB`, `host=156096.9 MiB`, `H2D 78.1 %`, `PRS rebuilds=2574`, same fills), and
   sustained bulk H2D is identical from both sources (27.93/28.68/28.76 vs 27.94/28.69/28.77 GB/s at 4/32/128
   MiB blocks).  So it is the prefill *staging path*, not the copy -- the one remaining open item.

## 5d-bis. (superseded in part by 5d-ter) `--host-experts pinned` on a *pageable* master

The 1-device comparison in §5d ("registered 477 vs pinned 615") rests on a **false premise**.  With
`--host-experts pinned` at 1 device, the delivery itself says:

```
bind_host_dev_locked: the host expert master is pageable (pool-backed `--host-experts pool`);
    the in-place cold read is disabled and misses are served from the pinned host pool
moe_cache_report: scheduler pageable-master fallback: 228 ranges, 288.8 MiB served through the per-table
    bounce; pinned bounce ring total 7489.6 MiB
moe_cache_report: host pool (L2): 120 tensors, 15360 slots, 9331.0 MiB pinned; h=0.0300 (4342/144664),
    fills=20293
```

`cudaMallocHost(18 662 MiB)` **fails**, `ggml_backend_cuda_host_buffer_type_alloc_buffer` silently falls back
to a *pageable* CPU buffer, and the expert pool then serves every fill.  That is why `pool` and `pinned`
measure **identically** (615.6 vs 615.6 t/s, same census).  Consequences:

1. **`--host-experts pinned` is not a pinned master for a large expert set.**  At 18.6 GiB it is a pageable
   master + a 9.3 GiB pool + up to a **7.5 GiB bounce ring** -- i.e. it uses *more* RAM than `pool` mode and
   performs the same.  This also explains the bimodal 2-device `pinned` numbers in §5d (71.6-72.1 vs 51.2):
   they track whether the pinned allocation (partly) succeeded.  The fallback must be **loud** (TODO: it is
   one `GGML_LOG_INFO`/`DEBUG` line today).
2. **Registration (the SVM path) is the only way to obtain a genuinely device-accessible host master of this
   size** on this stack -- which is exactly the argument for the maintainer's plan ("register everywhere").
3. **The 1-device comparison is therefore registered-in-place vs the POOL**, not vs a pinned master:

| 1 device | prefill | decode | host RAM |
|---|---|---|---|
| registered in place (SVM) | 477 | **53.4** | the page-cache pages, no duplicate |
| `pool` / nominal `pinned` | **615** | 48.8 | 9.3 GiB pool + 7.5 GiB bounce ring |

So the 1-device trade is: **-22 % prefill, +9 % decode, and ~17 GiB less RAM**.  The maintainer's instinct
("`HSA_USE_SVM` makes registration uniform everywhere") is the better reading of the data -- the alternative
at 1 device is not a pinned master, it is the pool with its duplicate + bounce ring.

## 5e. Can the master policy be automated at runtime?  Mostly yes -- and the residue is a *preference*.

The decision is "which master does the expert cache fill from", and every input it needs is already known at
sizing, before any fill happens:

| input | where it comes from | what it decides |
|---|---|---|
| device count | the tables' device set (already used by `pool_cold_slot_locked`) | 1 device -> registered costs prefill; 2+ -> registered wins both phases |
| `use_mmap` | the loader (a page-cache master must exist to register) | no mmap -> registration impossible, pinned allocation is the only master |
| host expert bytes vs `MemAvailable` | the preflight already computes `g_preflight_host_bytes` for AUTO | fit -> a duplicate is affordable; not fit -> it is not, so it must be the window form |
| registration *capability* | `hipHostRegister` + `cudaHostGetDevicePointer` at sizing | already handled: refusal leaves `host_pinned=false` and the pool path serves (`g_push_reg_stop`, `refusals`) |

So the policy table is derivable, with no benchmarking and no guessing:

| band | 1 device | 2+ devices |
|---|---|---|
| **fits in RAM** | **registered in place** (the only real pinned master at this size, §5d-bis: 477 prefill / 53.4 decode against the pool's 615 / 48.8 *with* 9.3 GiB pool + 7.5 GiB bounce ring) | **registered in place** (no duplicate, and strictly better: prefill 950 vs 709, decode 71.8 vs 51.2-72.1) |
| **does not fit** | registered **window** (a duplicate cannot fit; window = releasable, §5a) | registered **window** (same) |

What is **not** automatable is the 1-device fit-band row: registered is +9 % decode and -22 % prefill there.
That is a workload preference (long prompts vs long generations), not something the runtime can infer, so it
belongs behind a knob with a documented default, not behind a guess:

- `MOE_HOST_MASTER=auto|registered|pinned` (default `auto` = the table above), and the choice and its reason
  logged once at sizing, so a user can see which master they got and override it.
- The delivery has precedent for both halves: an H2D bandwidth calibration already runs at startup, and the
  prefill/decode *band* machinery (the gather's band gate, `g_period`) already exists if we later want the
  fill source to differ per band rather than per run.

**What I would NOT automate is an autotune probe.**  The 1-device penalty is invisible to a copy benchmark --
the copies measure 0.5 us each and 15 GB/s in both arms; the cost only appears per prefill *token*, under a
live op, with the cache on (§5d).  A startup probe would therefore confidently pick the wrong arm on one
device, and a probe big enough to see the difference is a prefill the user pays for on every start.

## 5f. `hipHostFree` gives the pages back -- PROVEN (so `hipHostMalloc` can be the one mechanism)

The plan is one allocation mechanism everywhere -- `hipHostMalloc` for both `pinned` (the whole set) and
`pool` (a bounded, rotated set) -- so the load-bearing question is whether a freed region really returns its
pages, or leaks them the way `hipHostUnregister` does (§5a).  `probe-hostfree.cpp` answers it two ways.

**Test 1 -- does the page count come back?**  102 chunks of 184 MiB (18.33 GiB, exactly the expert set of
35B-A3B), touched (the pin materialised), then `hipHostFree`d, twice:

| cycle | held | MemFree returned on free | vs baseline | swap |
|---|---|---|---|---|
| 0 | 18.33 GiB | **18 329 MiB** | -365 MiB | 1 831 MiB (flat) |
| 1 | 18.33 GiB | **18 376 MiB** | -323 MiB | 1 831 MiB (flat) |
| (earlier, 20.00 GiB x 3 cycles) | 20.00 GiB | 19 688 / 20 340 / 20 067 MiB | -893 / -811 / -902 MiB | flat |

No accumulation across cycles, and swap never moves -- the ~0.3-0.9 GiB residue is page-cache noise, not a
leak.

**Test 2 -- can the kernel actually REUSE them?**  Immediately after the last free, an 18.33 GiB file was read
through the page cache: it pulled **18.33 GiB from disk** at 2.02 GB/s and `MemFree` fell by ~18.3 GiB
(61 095 -> 42 425 MiB).  The freed bytes were genuinely available to become page cache again.

So `hipHostFree` is a real release -- the opposite of `hipHostUnregister`, which returns success and keeps the
pin (§5a).  That decides the design:

| mechanism | fastest transport | release |
|---|---|---|
| **`hipHostMalloc`** | **yes** (prefill 645.9 / 1 105.4 t/s, decode 53.5 / 72.0 -- the best at every device count) | **`hipHostFree` returns the pages (proven above)** |
| `hipHostRegister` (SVM) | 26 % / 14 % behind on prefill, decode at parity | **not releasable while the mapping lives** (§5a) |
| pageable + pool/bounce (today's `pool`) | 616.6 / 707.1 prefill, 49.0 / 51.3 decode | n/a |

**Therefore: use `hipHostMalloc` in both modes.**  `pinned` = the whole set in pinned chunks (the loader
already allocates per tensor, 144-176 MiB, and chunked allocation was verified at 18 584 MiB / 101 chunks /
101 aliased); `pool` = the *same tier* with a budget -- pin what the budget allows, rotate the contents, and
`hipHostFree` when shrinking.  That is exactly the maintainer's framing: **`pool` is `pinned` with a pin
limit and a policy**.  The registration experiment (patch 0007) is not needed for it and should stay out of
the delivery.

**The load phase is a real copy, and the tier costs its budget (both measured).**  `hipHostMalloc` returns
driver-owned pinned pages, so the loader must move the bytes into them -- it is not a page remap, and it is
measurable:

| arm | load phase (`loading model tensors` -> next line) | physical memory consumed |
|---|---|---|
| `pinned` | **1.9 s** (~9.8 GB/s for 18.6 GiB) | **19 651 / 18 476 / 18 583 MiB** over three consecutive runs |
| `pool` | 0.04 s (the bytes never leave the mmap) | ~10.1 GiB (the 9.3 GiB pool) |

`pool`'s 0.04 s is what a page remap would look like -- which is why the 1.9 s in `pinned` is proof of a copy
(it also *cannot* be a remap: hipHostMalloc's pages are pinned and handed to the GPU as stable physical
memory, so mapping reclaimable file pages into that VA would defeat the pin).

**RAM accounting, and it is the good answer: the tier costs its budget, NOT double.**  Three consecutive
`pinned` runs consumed 18 651 / 18 476 / 18 583 MiB for an 18 662 MiB tier -- no accumulation, and the
page-cache copy of the same bytes is not added on top (it gives way as the tier is populated, then stays
unread).  The measurement is **order-independent**: running `pool` first then `pinned` gives the same
per-mode figures (19.3 GiB / 10.1 GiB) as the reverse order.  The only ordering-sensitive quantity is what
each run leaves in the page cache when it exits (~11 GiB after `pinned`, ~1.3 GiB after `pool`), and that is
reclaimable.

The one-time 1.9 s is also optimisable if it ever matters: it is a double copy today (page cache -> user
buffer -> pinned), and a direct `read()`/`O_DIRECT` into the pinned buffer would halve it.

## 5g. Why 3-GPU host experts are 20 t/s for EVERY transport: the split mirrors an expert set on dev 0

A quick 3-GPU check (the maintainer asked for one) reproduces `TODO` #135 -- and shows it is **not a transport
problem at all**, because all three transports land in the same place:

| 3 GPUs (0,1,2), `-sm tensor` | decode (`-n 300`) | prefill (prose) |
|---|---|---|
| `pool` | 19.5 | 281.2 |
| `pinned` (real) | 20.1 | 443.5 |
| registered in place | 20.4 | 422.4 |

20 t/s is *worse than one GPU* (53.5), while all-VRAM on the same 3 cards is 138.  The geometry explains it --
with `GGML_CUDA_CACHE_GEOM_DEBUG` now printing the device id:

```
layer 0, 3 devices:
  dev=0  ffn_gate_exps  eb=589824 (WHOLE tensor)     ffn_down_exps eb=720896 (WHOLE)
  dev=1  ffn_gate_exps  eb=294912 (half, src_off 0)  ffn_down_exps eb=360448
  dev=2  ffn_gate_exps  eb=294912 (half, src_off 294912)  ffn_down_exps eb=360448 (src_off 176)
```

**Device 0 is MIRRORED (the whole expert) while dev 1 and dev 2 hold the two halves** -- 2x the expert bytes
and the whole expert's compute on dev 0 (so the critical path is one device, with extra traffic on top).
Every transport measures the same because the transport is not the bottleneck.

Why: it is **not** `-sm relaxed-tensor` (we never selected it).  `src/llama-impl.h:46-50` says *"TENSOR and
RELAXED_TENSOR place the weights identically ... they differ only in the replication contract"*, so the
placement code is shared, and `GGML_META_SPLIT_COPY` (default **1**: every scheduler weight copy inherits its
weight's split) is what makes the per-split copies follow the weight's state.  The split state itself is what
hands dev 0 a MIRRORED expert: the split is *"granularity-rounded, so it need not be even"*
(`ggml_backend_meta_split_dev_bytes`), and this model's expert rows are 512 wide against a 256-element quant
granularity, i.e. **exactly two valid chunks** -- so a 3-way split cannot be expressed and the leftover device
falls back to MIRRORED (the default arm of the split-state switch, `llama-model.cpp:680`).  That is why 2
devices is the last clean tensor split for this family, and why the campaign's own record says the 3-device
tensor split is not a scaling point.

**`-sm relaxed-tensor` changes nothing** -- measured, because the placement code is shared: same geometry
(dev 0 whole, dev 1/2 halves) and the same speed (decode 20.1 vs 20.0, prefill 442.4 vs 438.2).  The mode only
changes the replication contract, as `llama-impl.h` says.

**Root cause, now visible per layer.**  The split is *equal-division only, rounded DOWN to the quant
granularity* (`llama-model.cpp`, the extent loop: `high = ne_s*(j+1)/n_devices`, then
`high -= high % granularity`; the last device takes the remainder).  For this model's experts (axis width 512,
granularity 256 for q4_K/q5_K) and 3 devices that gives `high = 170 -> 0`, `341 -> 256`, remainder `256`, i.e.
extents **0 / 256 / 256**: one device is *given a zero extent*.  What happens to that device depends on the
layer's rotation (`rotation = il_eff % n_devices`), and the two outcomes are not equivalent:

| layers | extents observed | verdict |
|---|---|---|
| 1, 2, 5 | `dev0=294912, dev2=294912` (no table on the zero device) | a clean 2-way split, total = 1x |
| 0, 3, 4 | `dev0=589824 (WHOLE), dev1=294912, dev2=294912` | **mirrored: total = 2x bytes, whole-expert compute on one device** |

So two thirds of the layers replicate the expert and one third do not -- that is the 20 t/s, and it is why every
transport measures the same.  It is a **split-path defect** (block 14 territory / `TODO` #135): a zero-extent
device should be skipped consistently, never left holding a mirrored copy of the whole tensor.

### Why the split cannot simply be uneven (170/170/172 or 128/128/256)

Because a device's slice must contain **whole quantization blocks**.  These expert tensors are q4_K/q5_K, whose
super-block is **256 elements**, and the dequant/matmul kernels index blocks from the slice's own base -- so
every slice boundary has to be a multiple of 256.  The split axes here are 512 wide (host_bytes 589 824 /
element size = 2048x512 per expert), and 512 admits exactly **two** block-aligned chunks:

| candidate | valid? | why |
|---|---|---|
| 256 / 256 / 0 | **yes** | each slice is one whole super-block |
| 170 / 170 / 172 | no | no slice contains a whole block |
| 128 / 128 / 256 | no | 128 < one block |

So a 3-way split is *not expressible* for this model, and the correct fallback is the 256/256/0 one -- a 2-way
split with the third device idle (or a batch-parallel split of the tokens).  Replicating the whole tensor into
the zero-extent device is the bug.

### Why Qwen3.8-27B scales on 3 GPUs and the 35B-A3B experts do not

Measured, all VRAM, `-sm tensor` (the maintainer's example):

| Qwen3.8-27B Q8_0 | 1 GPU | 2 GPUs (0,2) | 3 GPUs (0,1,2) |
|---|---|---|---|
| prefill (5 246 tok) | 1 396.9 | **1 964.4** | 1 900.3 (-3 % vs 2) |
| decode (`-n 300`) | 20.4 | 35.7 | **45.5** (+27 % vs 2) |

So it does scale -- monotonically on decode, and prefill peaks at 2 GPUs (the 3-GPU arm carries the x4 card,
worth ~9 % prefill, so it is roughly flat there).  **The difference from the 35B-A3B experts is the quant
granularity, not "an even divisor":**

| model | splittable width | quant block | slices expressible | 3-way? |
|---|---|---|---|---|
| Qwen3.8-27B (attention/FFN) | 5 120 / 17 408, n_head 24 (= 3 x 8) | **32** (Q8_0) | 5 120 = 160 blocks, 17 408 = 544 blocks | **yes**, rounded to 1 664/1 664/1 792 and 5 792/5 792/5 824 |
| 35B-A3B experts | **512** | **256** (q4_K/q5_K super-block) | 512 = exactly **2** blocks | **no** |

So the rule is `splittable_width / quant_block_size >= n_devices` (plus the head count divisible for the
attention split), **not** exact divisibility: the 27B's 5 120 does not divide by 3 either, but its blocks are
32 elements wide so three real slices exist.  A finer quant (Q8_0/Q4_0, 32) splits well; a K-quant
(q4_K/q5_K, 256) needs a much wider axis to reach three devices -- which is why a MoE model's per-expert FFN
(512 wide) is stuck at two, and why the fallback must be "use fewer devices", never "mirror one".

Scale matters too: the dense 4B Q8_0 has the same fine granularity and still does not scale
(prefill 7 405 / 7 977 / 6 862), because the per-device work is too small against the all-reduce.

### Correction: the mirror explains the HOST path, not the VRAM numbers

I over-attributed earlier.  Measured, all-VRAM `-sm tensor`:

| all experts in VRAM | prefill | decode |
|---|---|---|
| 1 GPU | 4 641.9 | 114.6 |
| 2 GPUs (0,2) | 4 778.6 | 136.0 |
| 3 GPUs (0,1,2) | **4 036.5** | **138.3** |

So 3 GPUs is +1.7 % over 2 at decode and *16 % slower* at prefill -- the third device buys essentially nothing
in VRAM, but that is **not** the mirror.  Two controls settle it:

- a **dense 4B** model (no expert split at all) shows the same shape: prefill 7 405 / 7 977 / 6 862 and decode
  103.8 / 137.3 / 153.8 for 1 / 2 / 3 GPUs;
- the x4 card is worth only ~9 % prefill / ~4 % decode in a clean pair A/B (dense 4B, all VRAM: 0,2 = 8 068 /
  137.2 against 0,1 = 7 327 / 131.8).

So the 3-GPU **prefill** regression is the **all-reduce over three PCIe peers** (the activations per layer grow
with the batch), not the split geometry -- it is a property of `-sm tensor` on this box and applies to dense
models too.

What the mirror *does* explain is the **host-expert** case: 2x the PCIe bytes on two thirds of the layers, plus
the whole expert's compute serialised on one device, on top of the x4 link -- that is the 3.6x collapse from
72 t/s (2 devices, both x8) to 20 t/s.  Separating the mirror from the x4 link cleanly would need three x8
cards, which this box does not have (0 and 2 are x8, device 1 is x4).

**Practical consequence for the refactor:** do not chase the 3-GPU tensor number -- it is `TODO` #135/#104
territory (`relaxed-tensor` is a TEMPORARY block).  The usable 3-GPU configuration is `-sm layer` (a whole
tensor per device):

| 3 GPUs, `-sm layer` | decode | prefill |
|---|---|---|
| `pool` | 50.3 | 671.4 |
| `pinned` (real) | **54.4** | **622.3** |
| registered in place | 54.3 | 440.9 |

**One diagnostic worth adding to the delivery**: a table arriving with `split_axis` such that one device holds
`expert_bytes == host_bytes` while another holds a slice means the split is mirroring -- it silently costs
about 3.5x throughput and 2x the bytes.  That deserves a one-line WARN at sizing, next to the loud-fallback
requirement.

## 5h. The 3-GPU weakness is MODEL-SPECIFIC (proven on a second model), not a mechanism failure

The maintainer's question: does a larger model that fits 2 *and* 3 GPUs behave differently?  Measured on
**Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS** (arch `qwen4exp`, 48 layers, 512 experts / 10 used, n_embd 2560,
n_head 24, 70.6 GiB on disk of which **~21 GiB is PLE weights that stay on the HOST** -- which is why it fits
two 32 GiB cards at all).  Every arm ran with **`--lazy-mode off`** (added as a harness knob) so lazy PLE
loading does not confound the numbers.

| Flash-Next IQ3_XXS | 2 GPUs (0,2) | 3 GPUs (0,1,2) |
|---|---|---|
| experts in VRAM, prefill | 1 191.5 | 1 195.5 |
| experts in VRAM, decode | 84.4 | 86.9 |
| **all 48 expert layers pinned on host**, prefill | **297.6** | 196.7 |
| **all 48 expert layers pinned on host**, decode | 47.9 | 45.9 |

**And the split geometry is HEALTHY at 3 devices** -- unlike 35B-A3B, there is no mirrored table:

```
Flash-Next IQ3_XXS, layer 0, 3 devices (host_bytes = 473 600 for gate/up, 921 600 for down):
  dev=0  eb=94720  (1 granularity unit)   dev=1  eb=189440  (2 units)   dev=2  eb=189440  (2 units)
  dev=0  eb=184320 (1 unit)              dev=1  eb=368640  (2 units)   dev=2  eb=368640  (2 units)
```
i.e. the expert tensors carry **five** block-aligned units (against 35B-A3B's **two**), so a 3-way partition
exists -- it is just uneven (a 1/2/2 -> 20/40/40 compute split), and `eb != hb` on every device, so nothing is
mirrored.

So the 35B-A3B 3-GPU collapse is a **model-specific defect of the split fallback** (a two-unit axis cannot be
divided three ways, and the zero-extent device is given a mirrored whole copy instead of being skipped), not a
property of `-sm tensor`, the host-expert path, or any transport.  That is exactly the maintainer's framing:
**document 35B-A3B as the model that does not scale to 3 tensor-split devices, and fix the fallback so it uses
fewer devices instead of mirroring.**

The 2-vs-3 comparison here is still not a clean scaling claim: the 3-device set contains the **x4** card
(device 1) while the 2-device set is both x8, and the 1/2/2 partition is unbalanced.  What the measurement does
establish is that the *split is well-formed* for this model and *malformed* for 35B-A3B.

## 5i. The refactor: what is landed, what is left, and the numbers that justify each step

The design is settled (maintainer, 2026-11-04): **`hipHostMalloc` everywhere**, `pinned` = the whole set,
`pool` = the same tier with a budget and a policy, `auto` = pinned when it fits else a bounded pool.  Nothing
from the registration experiment goes into the delivery.

### DESIGN OF RECORD: one expert-indexed pinned host region (NOT yet implemented)

The transport is uniform (pinned host memory -> GPU, same CE path).  What is *not* uniform today is
`t.host_pinned`, the flag the prefill staging branch keys on, which is derived from **where the master lives**
rather than from **whether the bytes are pinned**:

```
bind_host_dev_locked():  host_pinned = (cudaHostGetDevicePointer(t.host) succeeded)
  pinned mode : master = the per-tensor ROCm_Host buffer  -> host_pinned = true  -> single-launch staging,
                                                                                   cold read available
  pool   mode : master = the pageable model mapping       -> host_pinned = false -> pool-backed staging
                                                                                   (two-tier snapshot rebuilt
                                                                                   per op + per-expert copies),
                                                                                   cold read unavailable
```

Measured consequence, with the SAME full 18.6 GiB pool (30 720 slots) in both arms at 2 devices:

| | prefill | decode |
|---|---|---|
| pinned master, pool present (`LLAMA_HOST_MASTER_PINNED=1`) | **1 105.3** | **71.9** |
| pageable master + full pool (`-lme 64`) | 532.9 | 22.6 |
| full pool + pinned master (`-lme 64` + `LLAMA_HOST_MASTER_PINNED=1`) | 1 009.4 | 71.9 |

So a fully-pinned *slot set* does not behave like a pinned *master*: the mode-keyed switch, not the transport,
is the 3x.  A pool **beside** a pinned master also costs ~9 % prefill for nothing (1 009.4 vs 1 105.4).

**Target shape** -- the maintainer's design, and what the refactor must produce:

1. **One pinned host region, addressed by expert index, with a presence bitmap** (`present[e]`).  Today's pool
   stores a *permutation* of whole experts in slots, so it cannot serve as a master (the fill indexes weights
   as `expert * host_bytes`, slot != expert) -- that is precisely why `host_pinned` stays false and the slow
   branch is taken.  An expert-indexed region makes a full set identical to today's `pinned`, and a partial set
   a pinned master **with holes**.
2. **Filled greedily at load in descending hotness**, after VRAM has been filled with the hottest weights up to
   the first wide prefill boundary: put each host expert in the region while it fits, else leave it in the
   pageable model mapping.
3. **Sized by `-lme`**: `auto` = `MOE_HOST_POOL_FRAC` (80 %) of `MemAvailable` at load; `N` = N GiB (clamped to
   the host expert set; refused above 90 % of available memory).  Same mechanics for both -- only the cap
   differs.
4. **Releasable**: `hipHostFree` returns the pages (proven, §5f).
5. **The pageable mapping is the per-expert fallback through the holes**, so `host_pinned` becomes a per-table
   truth about *the weights* and the two staging branches collapse into one hole-aware path.  The cost of a
   non-resident expert is quantified: ~532 prefill / 22.6 decode at 2 devices, which is what the hotness
   ordering exists to minimise.

**Implementation steps** (none of this is written yet):

| step | where | acceptance |
|---|---|---|
| a | expert-indexed region + presence bitmap replacing the slot permutation | a full region reproduces the `pinned` numbers (1 105.4 / 72.0 at 2 devices) |
| b | greedy load-time placement in hotness order, capped by `-lme` | reporting shows `pinned X of Y GiB`; a set that fits reports `X == Y` and matches step a |
| c | hole-aware staging branch: one path, `host_pinned` per table | a partial set keeps the pinned numbers for resident experts; non-resident ones take the pageable path |
| d | retire the mode distinction (`--host-experts` stays a deprecated shorthand) | one mode in the code; `-lme` is the only knob |
| e | census: `pinned X GiB (cap C), pageable Y GiB`; keep the degenerate-split and pinned-fallback warnings | the tier split is visible in one line |

### Landed (fork commit `e30ba75b0`, patch `0008`): the two silent failures, made loud

| diagnostic | why | verified |
|---|---|---|
| `PINNED HOST ALLOCATION FAILED ... falling back to PAGEABLE memory` (stderr, once) | the fallback is **not device-readable**, so `--host-experts pinned` stops being pinned, the pool serves the fills, a multi-GiB bounce ring grows, and throughput lands at the pool's -- this is what hid the whole thing, since the only trace was a `GGML_LOG_DEBUG` line | forced with `GGML_CUDA_NO_PINNED=1`: fires; normal run: silent |
| `DEGENERATE TENSOR SPLIT: ... one device holds the WHOLE expert (MIRRORED) and another a slice` (stderr, once) | a split that cannot be expressed hands the zero-extent device the whole tensor; measured ~20 t/s against 72 at 3 devices for every transport | 35B-A3B at 3 devices: fires (naming layer/role/devices).  At 2 devices, and on Flash-Next at 3 devices: silent, as it should be |

Both go through `fprintf(stderr, ...)` **not** `GGML_LOG_WARN`, because ggml's WARN is filtered at
llama-cli's default level -- the same trap that hid the campaign's own P1-PUSH messages in P3.

### Landed (fork commit `a644115a0`, patch `0009`): ONE knob for the tier

The mechanism is uniform, so the interface is now uniform too -- the maintainer's design:

```
-lme, --limit-host-experts auto|N        (auto is the default once -ncmoe/-cpu-moe is given)
    auto : pin the whole host expert set when it fits in RAM, else a bounded pinned pool at
           MOE_HOST_POOL_FRAC (80%)
    N    : pin a pool of N GiB, clamped to the size of the host expert set; N above 90% of
           MemAvailable is REFUSED, because that memory cannot be swapped out
--host-experts pinned|pool               DEPRECATED shorthands (warn): pinned = pin the whole
                                         set, pool = auto
```

Verified end to end on this box:

| invocation | result |
|---|---|
| `-lme 200` | refused: *"more than 90% of the available memory (82 GiB): that memory is non-swappable"* |
| `-lme 8` | `host pool (L2): 13440 slots, 8164.6 MiB pinned` |
| `-lme 64` (host set is 18.2 GiB) | clamped: `30720 slots, 18662.0 MiB pinned` -- the whole set |
| `--host-experts pool` | deprecation warning, then `limit-host-experts: auto -- pin the whole host expert set ...` |

**Honest gap, and it is the one remaining design piece**: `auto` resolves to *pinned* today.  The pool fallback
for a set that does not fit needs the host-expert byte total **before** buffer assignment, which the loader does
not currently compute.  Two options, in preference order:

1. **a metadata pre-pass** in the loader (sum the nbytes of the tensors the `-ncmoe` selection will host, before
   `host_expert_buft` assigns any of them) -- exact, and it makes `auto` a pure decision;
2. **switch on failure**: keep pinned, and let the (now loud) `PINNED HOST ALLOCATION FAILED` warning trigger a
   retry in pool mode.  Cheaper, but a half-loaded model has to be torn down first, and the user pays the failed
   allocation.

### Left to do (in order), with the acceptance bar for each

| # | change | acceptance |
|---|---|---|
| 1 | `pinned`: keep the per-tensor `ROCm_Host` (`hipHostMalloc`) master; report the tier's pinned MiB in the census | 1/2 devices: `pinned` >= today's numbers (prefill 645.9 / 1 105.4, decode 53.5 / 72.0) and zero fallback warnings |
| 2 | `pool` becomes *pinned with a budget*: the pinned tier is the pool (already `hipHostMalloc`), the overflow stays pageable from the model mapping, shrink/free via `hipHostFree` | budget and pinned bytes both reported; freeing returns the pages (proven, §5f); no change to the not-fitting band's behaviour |
| 3 | `auto`: pinned when the host set fits `MemAvailable` (with headroom), else the bounded pool | the mode choice logged once with its reason |
| 4 | retire the registration path from the deliverable (patches `0001`-`0007` stay experimental in `wip/`) | `scripts/validate-set.sh` green after regenerating the block set |
| 5 | docs: `ENVIRONMENT.md` entries (default + class per variable), the 3-GPU tensor-split rule (`width / quant_block >= n_devices`, with 35B-A3B named as the case that does not divide and `-sm layer` as its answer), `TODO` for the mirroring-fallback fix | a reader can predict 3-GPU behaviour from the docs alone |
| 6 | gates: coherence (same-seed greedy, byte-identical) + 2 reps of decode and prefill at 1/2 devices for `pool`/`pinned` + `check-gpu-faults.sh` | as `MANIFESTS.md` requires before any tag |

**Deliberately out of scope**: the registered-window tier (parked in `wip/`, releasable but 26 %/14 % behind on
prefill), fixing the splitter's mirroring (documented + warned, not fixed here), and any autotuning probe (the
1-device signature is invisible to a copy benchmark, §5e).

## 6. Dated facts that contradict older ones (recorded, not reconciled)

- **On this stack a pageable 1-D copy is slower than a pageable 2-D copy** of the same bytes (91-146 vs 41-45
  µs/call, §4).  The maintainer's observation from the single-GPU system was the opposite.  Both can be true
  across ROCm lines/geometries; what is *not* geometry-dependent is the conclusion: **a pageable source is a
  staging cost per call, and the fix is not to reshape the copy but to stop it being pageable.**
- The code comment above `expert_host_src` (`a 2-D pageable H2D is the pathological ROCm case`) is **not**
  what this box measures — the pathological case here is the 1-D pageable upload.  Left in place (it is a
  wip patch's comment); do not cite it as current.

## 7. Reproduce

```bash
cd wip/host-expert-eager-push

# the attribution (per-copy geometry split + the reporting line)
GPU=0,2 SM=tensor EOS=1 N=300 LV=4 ./p3-matrix.sh map  MOE_HOST_PUSH_SRC=mapping MOE_HOST_PUSH_PROBE=1
GPU=0,2 SM=tensor EOS=1 N=300 LV=4 ./p3-matrix.sh reg  MOE_HOST_PUSH_REGISTER_MIB=20000 \
                                                         MOE_HOST_PUSH_PROBE=1 MOE_EXPERT_CACHE_COLD_UVA=0

# the headline pairs (clean runs: text hash valid, no -lv)
GPU=0,2 SM=tensor EOS=1 N=300 ./p3-matrix.sh ctrl
GPU=0,2 SM=tensor EOS=1 N=300 ./p3-matrix.sh reg MOE_HOST_PUSH_REGISTER_MIB=20000

# the geometry a tensor split produces
GPU=0,2 SM=tensor EOS=1 N=4 LV=4 GGML_CUDA_CACHE_GEOM_DEBUG=1 ./p3-matrix.sh geom   # grep CACHE_GEOM

# the standalone reproduction + the pin's lifetime
/opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-memcpy2d probe-memcpy2d.cpp
HIP_VISIBLE_DEVICES=0,2 MODE=1d SRC=file    /tmp/probe-memcpy2d 2 10 16   # 91 us/copy
HIP_VISIBLE_DEVICES=0,2 MODE=1d SRC=reg     /tmp/probe-memcpy2d 2 10 16   # 23 us/copy
HIP_VISIBLE_DEVICES=0,2 MODE=1d SRC=file BUSYMB=256 /tmp/probe-memcpy2d 2 10 16   # 146 us/copy
/opt/rocm-10.1.0-gfx120X/bin/hipcc --offload-arch=gfx1201 -O2 -o /tmp/probe-pin probe-pin.cpp
FILE=/var/tmp/pin-test.bin /tmp/probe-pin 256        # pin survives unregister, dies with the mapping
FILE=/var/tmp/pin-test.bin /tmp/probe-pin-free 256   # hipFreeHost is invalid for a registered range
```

## 8. Open questions for the maintainer

1. **Two shapes, now both measured — which do we build?**
   (a) *in-place registration* (`MOE_HOST_PUSH_REGISTER_MIB`, as it stands): the only no-duplicate transport
   measured at parity, and with the pool off **71.7-71.9 t/s against the pool's 51.5 with 9.2 GiB less pinned
   RAM** (§3a) — but the pin is permanent, so it is an **all-or-nothing fit-in-RAM mode** and must be
   default-off / refused where the set does not fit.
   (b) *the registered window* (§5a): releasable, proven droppable over 40 cycles and through a long-lived
   second mapping, same speed, no duplicate — **the only shape that can serve the > RAM band**, and the one
   the campaign's charter was written for.
2. **Or go straight to the registered window** (§5) — the releasable shape, which needs the epoch gate wired
   to `unregister + munmap`, a rotation cadence, and a guard gap.  That is the campaign's charter answer, and
   it is now a bounded engineering task rather than an unexplained gap.
3. **The registered master makes the delivery's cold in-place read (UVA) legal** (`cudaHostGetDevicePointer`
   succeeds), i.e. a *device-pull* path becomes reachable where the charter says "never device-pull".  With
   `MOE_EXPERT_CACHE_COLD_UVA=0` the arms above measure the pure push and are unchanged (71.6 vs 71.7), so
   this is a policy choice, not a performance one — but it must be an explicit one.
4. `MOE_HOST_PUSH_SRC=mapping` (pageable) should probably be retired as a candidate: it is strictly worse than
   the bounce ring it was meant to replace (§2 shows 141 µs against 1.3 µs), and the campaign's answer is
   registration, not the pageable push.
