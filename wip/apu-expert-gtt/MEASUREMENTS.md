# MEASUREMENTS — `wip/apu-expert-pool`

Every number in [`README.md`](README.md) and [`IMPLEMENTATION.md`](IMPLEMENTATION.md) comes from here.
(This file was written under the campaign's opening name `wip/apu-expert-pool/`; the campaign was renamed
`wip/apu-expert-gtt/` on 2026-11-11 and its design replaced -- see `README.md` §1 and §13 below.)  Nothing
below changed the fork or the delivery: all probes are standalone HIP/C programs under
[`probe/`](probe/README.md), run as ordinary users.

## 0. The box, at the time of measurement

| | |
|---|---|
| host | `halo` — AMD **Ryzen AI MAX+ 395** w/ Radeon 8060S (**gfx1151**, RDNA3.5 APU) |
| compute | 40 CU, 2 SIMD/CU, wavefront 32, L2 2 MiB, L3 32 MiB |
| RAM | 123 GiB usable; 175 GiB swap (16 GiB zram + NVMe); THP `madvise` |
| kernel | 7.2.13-200.fc44.x86_64 (`CONFIG_HMM_MIRROR=y`, `CONFIG_DEVICE_PRIVATE=y`, `CONFIG_ZONE_DEVICE=y`) |
| ROCm | 10.1.0 at `/opt/rocm-10.1.0-gfx1151` (HIP 7.16.26385) |
| cmdline | `amdgpu.gttsize=124000 ttm.pages_limit=31744000 iommu=off pcie_aspm=off transparent_hugepage=madvise` |
| storage | nvme0n1 CT4000T500SSD3 (3.6 T, `/llm`), nvme1n1 CT2000P310SSD8 (1.8 T, root) |
| reference model | `/llm/models/DeepSeek/Q4_K_XL/DeepSeek-V4-Flash-…-00004-of-00005.gguf` (46 GiB shard) |

Build line used for every probe:

```
export PATH=/opt/rocm-10.1.0-gfx1151/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
hipcc -O2 -o probe <src>.cpp && ./probe
```

(`hipcc` from `/usr/bin` links against ROCm 7.1 and fails with `undefined symbol: hipGetDeviceCount` for these
sources — use the 10.1 toolchain explicitly.)

## 1. Topology and capability (`probe/01-props.cpp`, `rocminfo`, sysfs)

```
dev 0: AMD Radeon 8060S Graphics | arch=gfx1151 | integrated=1 | canMapHostMemory=1 | unifiedAddressing=1
  totalGlobalMem=121.1 GiB isLargeBar=1 memBusWidth=256
  ManagedMemory                      1
  ConcurrentManagedAccess            1
  PageableMemoryAccess               0        <-- no pageable device access
```

| rocminfo / sysfs | value |
|---|---|
| `Memory Properties` | **APU** |
| `Coherent Host Access` | **FALSE** |
| `IOMMU Support` | None (`iommu=off`) |
| Pool 1 | GLOBAL COARSE GRAINED, 126 976 000 KB (GTT) |
| Pool 2 | GLOBAL EXTENDED FINE GRAINED |
| Pool 3 | GROUP 64 KB |
| `mem_info_vram_total` | **2 147 483 648 (2 GiB)** |
| `mem_info_gtt_total` | 130 023 424 000 (121 GiB) |
| kfd node 1 | `gfx_target_version 110501`, `simd_count 80`, `capability 671326848` |

**XNACK is not a setting here, it is absent from the ISA** (ROCm 10.1 target list):

| target | result |
|---|---|
| `gfx1151` | accepted |
| `gfx1151:xnack+` / `:xnack-` | **`invalid target ID`** |
| `gfx1100:xnack+`, `gfx1201:xnack+` | `invalid target ID` |
| `gfx90a:xnack+`, `gfx942:xnack+` | accepted |

## 2. Does the GPU fault-and-retry? (`probe/02-fault.cpp`)

Kernel reads a 64 MiB region (`sum_bytes`); `mincore()` counts resident pages; each arm is a separate process.

| arm | residency | result |
|---|---|---|
| `hot` — page-cache mapping, **not** registered, all pages resident | 16 384/16 384 | **`HSA_STATUS_ERROR_MEMORY_FAULT`**, `Memory access fault … faulting addr 0x…e04000`, `Reason: Page not present or supervisor privilege`, GPU coredump, exit **134** |
| `hot` with `HSA_XNACK=1` | same | **identical fatal fault** (flag inert) |
| `cold_unreg` — `fadvise`+`madvise` DONTNEED, not registered | 16 384/16 384 (this run's DONTNEED did not drop them) | same fatal fault |
| `cold_reg` — `hipHostRegister(Mapped\|ReadOnly)` | 16 384/16 384 | **survives**, GPU sum = CPU sum |

Conclusion: residency is irrelevant without registration; an unregistered pageable mapping is not in the GPU's
page tables at all, and there is no retry.

## 3. GPU read bandwidth (`probe/03-bandwidth.cpp`)

1 GiB, read-only sum kernel, 512×256, best of 6, `hipEvent` timing:

| backing | GB/s |
|---|---|
| `hipMalloc` (device = GTT) | **232.6** (232.7 in a second run) |
| anonymous mmap, `hipHostRegister(Mapped)` writable | **60.2** |
| anonymous mmap, `hipHostRegister(Mapped\|ReadOnly)` | **230.8** |
| file / page-cache mmap (hot), `Mapped\|ReadOnly` | **230.5** (230.2 in a second run) |

The 60-vs-230 split tracks the **`ReadOnly` registration flag**, not the backing: the writable registration is
mapped coherent/uncached.  (Mechanism inferred from the flag experiment; the flag-level difference is measured
four times.)

## 4. Fill paths and the register cycle (`probe/04-fill-and-register.cpp`)

| operation | result |
|---|---|
| `pread(O_DIRECT)` 1 GiB into an anonymous buffer | 0.29 s = **3.6 GB/s** |
| `hipHostRegister(Mapped\|ReadOnly)` on a `PROT_WRITE` buffer | **OK** |
| GPU read of that buffer, registered ReadOnly | **233.2 GB/s** |
| GPU read of the same buffer registered writable | **60.8 GB/s** |
| register / unregister / re-register (1 GiB) | 2.67 ms / 0.00 ms / 0.00 ms |
| `memcpy` page cache → buffer (1 GiB) | 0.040 s = **27.0 GB/s** |

## 5. Dirty pages, discard, and writeback (`probe/05-dirty-and-discard.cpp`)

`smaps` of the containing VMA, plus `/proc/vmstat` deltas:

| arm | Rss | Private_Dirty | note |
|---|---|---|---|
| anon pool 4 GiB, host-filled | 4096 MiB | **4096 MiB** | dirty by definition |
| …registered `Mapped\|ReadOnly` | 4096 MiB | 4096 MiB | unchanged |
| …`MADV_DONTNEED` **while registered** | 4096 → 196 MiB in 6.14 ms | 196 MiB | most pages *are* released — see §6 for why that is a trap |
| …`hipHostUnregister` + `MADV_DONTNEED` | 2 MiB in 0.26 ms | 2 MiB | **`pswpout` 0, `pgpgout` 0** |
| file `PROT_READ` `MAP_SHARED`, hot | 1 GiB | **0** | never dirty (`Private_Clean` 1 GiB, `FilePmdMapped` 1007 616 kB) |

`VmPin` and `VmLck` read **0** in every one of these arms — do not use them to detect a registration hold;
use `mincore()`.

## 6. The silent-zero trap (`probe/06-dontneed-zero-trap.cpp`)

Registered `Mapped|ReadOnly` 1 GiB anon region, filled with `0x33`, GPU summed, then `madvise(DONTNEED)`:

```
GPU sum before DONTNEED: 0333333330000000
GPU sum after  DONTNEED: 0000000000000000    no fault, no error, no warning
```

So a discard of a registered region the GPU is reading is **silent weight corruption**, not a fault.  Eviction
must be a table operation; discarding is `hipHostUnregister` **then** `madvise`.

## 7. Host access to the GTT arena (`probe/07-host-write-gtt.cpp`)

| operation on a `hipMalloc` 1 GiB buffer | result |
|---|---|
| host read | **OK** |
| host write | **OK** |
| host `memset` (1 GiB) | 0.079 s = **13.5 GB/s** |
| GPU sum after the host memset | **FRESH** (matches the pattern) |
| `pread(O_DIRECT)` into it | **`EINVAL`** (−1) |

## 8. DMA fill rates (`probe/08-dma-fill.cpp`)

| transport, page-cache source → GTT arena | GB/s |
|---|---|
| `hipMemcpy` H2D, first (cold-ish cache) | **31.5** |
| `hipMemcpy` H2D, warm | **77.1** |
| host `memcpy` | **12.8** |

## 9. Registration is a hold, and it survives unregister (`probe/09-register-pins.cpp`)

4 GiB slice of the 46 GiB shard, `mincore()` after `posix_fadvise(DONTNEED)` + `madvise(DONTNEED)`:

| arm | resident |
|---|---|
| **control** — never registered | **0 / 1 048 576 pages** (all 4096 MiB reclaimed) |
| registered `Mapped\|ReadOnly` | **1 048 576 / 1 048 576** (held) |
| registered → `hipHostUnregister`, VMA kept | **1 048 576 / 1 048 576** (still held) |

Whole shard (**46 GiB, 11 906 828 pages**):

| operation | result |
|---|---|
| `hipHostRegister(Mapped\|ReadOnly)` | **8.69 s** (5.6 GB/s — the whole file is read), all pages resident |
| `fadvise`+`madvise` DONTNEED | no effect (all held) |
| `hipHostUnregister` | 0.00 s, pages still held while the VMA lives |
| process exit (`munmap`) | release: `MemAvailable` back to baseline |

Hot 1 GiB: first register **14.0 ms**, subsequent register/unregister cycles **0.0 ms** (the KFD keeps the
range).

## 10. Coherence of host writes into a registered range (`probe/10-coherence.cpp`)

1 GiB anon buffer, registered, GPU sums it; host then overwrites all of it in place (**no** unregister, no
flush, no sync beyond the launch), GPU sums again — four alternations:

| registration | iter 0 (0x11) | iter 1 (0x22, after host overwrite) | iter 2 | iter 3 |
|---|---|---|---|---|
| `Mapped\|ReadOnly` | FRESH | **FRESH** | FRESH | FRESH |
| `Mapped` (writable) | FRESH | **FRESH** | FRESH | FRESH |

So "fill in place, no flush" holds for both modes — which is what makes a host-filled slot legal.

## 11. Disk (reference only)

`dd bs=16M count=64 iflag=direct` on the 46 GiB shard: **4.4** and **5.5 GB/s**; the same read buffered:
**5.7 GB/s**.  (The delivery's own records have the latency-bound figure for a 0.65 MiB expert read:
~2.2 GiB/s at depth 1 — that is the disk tier's real shape, and it is unchanged by this campaign.)

## 12. Things that were checked and did **not** help

* `HSA_XNACK=1` — inert (§2).
* `PageableMemoryAccess` — 0, so no HMM path to pageable memory exists from HIP on this part.
* The kernel *has* HMM (`hmm_range_fault` appears in the journal, reached from
  `amdgpu_amdkfd_restore_userptr_worker`) — that is the driver re-validating pinned userptr, **not** a GPU
  fault-retry path.

---

## 13. 2026-11-11 — the P1.0 probes (11-14) for the GTT expert store, and the expert-traffic concentration

The 2026-11-09 sections above were written for the campaign's **first** design (a host pool + arena).  The
design was replaced on 2026-11-11 (`README.md` §1) by a single registered anonymous **expert store**, and that
design rests on four assumptions the ten probes above did not measure.  This section answers them.  Records are
appended, not rewritten: §§0-12 stand as measured.

Same box, same build line (`/opt/rocm-10.1.0-gfx1151`, HIP 7.16.26385, gfx1151, `iommu=off`).  Sources:
`probe/11-thp-register.cpp`, `12-pread-registered.cpp`, `13-dma-from-mapped-file.cpp`,
`14-expert-cost.cpp`; helpers `probe/gguf-expert-geom.py`, `probe/moe-heatmap-read.py`.

### 13.1 Do THPs survive registration?  (`11-thp-register`, 1 GiB) — **YES**

| step | AnonHugePages | Rss | Private_Dirty |
|---|---|---|---|
| anon `mmap` + `MADV_HUGEPAGE`, faulted in, not registered | **1024.0 MiB of 1024** | 1024.0 | 1024.0 |
| `hipHostRegister(Mapped\|ReadOnly)` | **1024.0 MiB** (unchanged) | 1024.0 | 1024.0 |
| after `hipHostUnregister` | 1024.0 MiB (unchanged) | 1024.0 | 1024.0 |

* registration of a **pre-faulted** region: **17.4 ms/GiB** (61.7 GB/s); of an **untouched** region: 30.5 ms/GiB
  (the later fault then still lands as huge pages: 1024.0 MiB).
* GPU read of the registered region: **232.1 GB/s** (pre-faulted) / **227.3 GB/s** (registered then faulted).

So the tier really is a huge-page tier, registration does **not** split it, and a 40 GiB store costs ~0.7 s of
registration at load — not the 8.69 s §9 measured for registering a 46 GiB **file** mapping (that number
included faulting the file in).

### 13.2 `pread(O_DIRECT)` into a registered region, then a GPU read (`12-pread-registered`, 256 MiB) — **YES**, with one rule

| arm | result |
|---|---|
| registered `Mapped\|ReadOnly`, pre-filled `0xAA` | CPU sum == GPU sum (45 634 027 520) |
| `pread(O_DIRECT)` **into** the registered region | **accepted** (not `EINVAL`) — 4.75 GB/s first, 6.37 / 6.39 GB/s after |
| GPU read after each pread | CPU sum == GPU sum **every round**, including rounds that overwrite the previous round's bytes (the eviction/reuse case) |
| buffered `pread` into the same region | **15.68 GB/s** (page-cache source), sums match |

**Rule (measured separately, `dd` on the same file): a GGUF tensor offset is 32-byte aligned, not 4096-byte
aligned.**  An `O_DIRECT` read at a real expert offset copies **0 bytes**; at `align_down(off, 4096)` it works,
and the expert's bytes then land at `slot + (off - align_down(off, 4096))` (2 688 bytes for the layer measured
below).  So an `O_DIRECT` fill needs a per-fill delta and a slot head-pad of one block.

Also worth recording: the buffered arm is **faster** than `O_DIRECT` for a single slab on this box, and it
leaves the source in the page cache (which is the warm arm's source).  "Zero-copy" and "`O_DIRECT`" are not the
same decision.

### 13.3 The copy-engine pull from a registered file mapping (`13-dma-from-mapped-file`, Flash-Next GSQ-IQ3_XXS)

`hipHostRegister(Mapped|ReadOnly)` on a `MAP_SHARED` file mapping: **OK** (as §3 measured for read bandwidth).

| transfer | first | warm | bytes verified | host `memcpy` reference |
|---|---|---|---|---|
| one expert slab, 0.680 MiB | 0.083 ms = **8.55 GB/s** | 0.012 ms = **60.91 GB/s** | match | 0.017 ms = 41.01 GB/s |
| layer span, 16 MiB | 0.903 ms = **18.57 GB/s** | 0.570 ms = **29.42 GB/s** | match | 0.369 ms = 45.45 GB/s |

So the warm arm is real, and at expert granularity the copy engine beats a host `memcpy` into the same region
(0.012 vs 0.017 ms).  This is the arm that makes the load part of the same submit as the `MUL_MAT_ID`.

### 13.4 Per-fill cost at the real granularity (`14-expert-cost`, Flash-Next `blk.2.ffn_down_exps`)

Geometry (from `gguf-expert-geom.py`): 48 layers with `*_exps` tensors, **512 experts**, per-expert slab
0.452-0.921 MiB per tensor, **1.782 MiB per expert across the three tensors** of one layer; the whole expert set
is **≈40 GiB** (33.37 GiB counted over known quants).  The delivery's cache keys each tensor separately
(144 heat tables = 48 layers × gate/up/down), so a slot is one *(tensor, expert)* slab.

| arm (slab 0.879 MiB, at a real 32-byte-aligned offset) | min | median | mean | max |
|---|---|---|---|---|
| COLD one slab, `O_DIRECT` (rounded out) | 0.380 ms | 0.413 ms | 4.752 ms | **34.885 ms** |
| COLD one slab, buffered | 0.480 ms | 0.567 ms | 0.623 ms | 0.892 ms |
| warm one slab, `O_DIRECT` | 0.155 ms | 0.381 ms | 0.375 ms | 0.488 ms |
| warm one slab, buffered | 0.068 ms | **0.069 ms** | 0.070 ms | 0.079 ms |
| COLD 16-slab span, `O_DIRECT` | — | 2.563 ms = **5.75 GB/s** (14.1 MiB) | | |
| warm 16-slab span, `O_DIRECT` | — | 2.370 ms = **6.22 GB/s** | | |
| `pread` -> launch -> `hipStreamSynchronize` floor | 0.407 ms | **0.420 ms** | 0.489 ms | 0.984 ms |

Three consequences:

1. **A cold single-slab read is ~0.4-0.6 ms, with a 35 ms first-read-after-idle tail.**  (The `ple-residency`
   record's 4.7 ms first-cold-read figure is the same phenomenon on the same drive, at a different idle gap.)
2. **Bulk is fine, scatter is not**: 14.1 MiB sequential = 5.75 GB/s; one 0.879 MiB slab = 0.4 ms (~2.1 GB/s).
   Prefill's span shape is cheap; decode's scattered single-slab shape is latency-bound.
3. **The fill is on the critical path per layer.**  A decode step touches top-k × 3 tensors × 48 layers =
   **1 440 slots**.  At the 0.42 ms floor: every slot missing = **~0.6 s/step**; 90 % hit rate = ~60 ms/step;
   98 % = ~12 ms/step.  **The store only pays at a very high hit rate, or with the next layer's fills pipelined
   behind the current layer's compute** (`TODO` #132 is the same defect on the discrete pool: "the pool's pair
   decode deficit is the synchronous demand read").

### 13.5 How concentrated is expert traffic?  (`moe-heatmap-read.py` on the delivery's own persisted heat maps)

This is the input to "how big must the store be" and to the admission policy.  MHT2 has two bands: `all` and
`dec` (decode only).  Coverage is over *slots* = (tensor, expert) pairs.

| model | slots | traffic | never touched | 90 % coverage | 95 % | 99 % |
|---|---|---|---|---|---|---|
| **35B-A3B Q4_K_M** (120 tables, 18.22 GiB) | 30 720 | 7 522 320 accesses | **21.0 %** | 10 891 slots (35.5 %) = **6.46 GiB** | 44.2 % = 8.05 GiB | 57.7 % = 10.52 GiB |
| … its `dec` band (93 % of its accesses) | | 6 998 160 | 32.1 % | 32.2 % = 5.87 GiB | 40.1 % = 7.31 GiB | 52.2 % = 9.51 GiB |
| **Flash-Next GSQ-IQ3_XXS** (144 tables, 39.97 GiB) | 73 728 | 278 250 accesses | **75.1 %** | 9 648 slots (13.1 %) = **5.23 GiB** | 18.0 % = 7.18 GiB | 23.0 % = 9.19 GiB |
| … its `dec` band (36.2 % of its accesses) | | 100 590 | 97.2 % | 0.3 % = 0.11 GiB | 0.3 % = 0.14 GiB | 1.5 % = 0.58 GiB |

**The concentration depends on how long the history is, and the second row is the honest one.**  The 35B-A3B map
has 27x the accesses of the Flash-Next map, and its 90 % coverage needs **35 % of the slots**, not 13 % — the
Flash-Next figure is largely *undersampling* (278 k accesses over 73 728 slots).  Plan on the long-history
number: **a resident store of ~6-8 GiB covers 90-95 % of expert traffic on an 18-22 GiB expert set, and ~20-25 %
of the slots are never touched at all** — that never-touched mass is what an Nth-touch admission policy leaves
on disk for free.

### 13.6 What this changes in the plan

* P1.0 assumptions **all hold**; the store's memory is confirmed (registered anon THP) and the two fill arms are
  both real.  `IMPLEMENTATION.md` P1.0 is closed by this section.
* **The fills must be pipelined**, not synchronous per layer, or the design cannot decode at a 90 % hit rate
  (§13.4).  This becomes a P4 requirement, not an optimisation.
* **Admission is measurable now**: a promotion costs one cold read (0.4-0.6 ms) and saves that read on every
  later use, so a 2nd-touch promotion pays for itself at two uses in a session.  The *per-session touch
  distance* is still unmeasured (the heat map is a long-horizon counter, not a session sequence) — that is the
  one probe still owed (`probe/README.md`, 15).
* `O_DIRECT` at a real GGUF offset needs an alignment round-out and a per-fill delta (§13.2); the buffered arm
  is a legitimate (and here faster) alternative and should stay an arm, not be dropped.
