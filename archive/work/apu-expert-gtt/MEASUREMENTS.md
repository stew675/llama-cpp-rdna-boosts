# MEASUREMENTS — `wip/apu-expert-pool`

> **PROMOTED as block 16 of the delivery in r263 (`v17-a691b2345-r263`).**  This document is the dated
> campaign record; the shipping code is `patches/0016-…`, the knobs `ENVIRONMENT.md` §20, the rules
> `INVARIANTS.md` ("Block 16"), and the gate `scripts/gate-uma-experts.sh`.

Every number in [`README.md`](README.md) and [`IMPLEMENTATION.md`](IMPLEMENTATION.md) comes from here.
(This file was written under the campaign's opening name `wip/apu-expert-pool/`; the campaign was renamed
`archive/work/apu-expert-gtt/` on 2026-11-11 and its design replaced -- see `README.md` §1 and §13 below.)  Nothing
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

---

## 14. 2026-11-11 — the GTT store EXISTS (P1.1 + P2): the keystone re-run, and what the registered region costs and reads at real scale

The design's memory was answered in §13.  This section is the same box measuring the **implemented** GTT store
(`gtt-expert-store.patch`, `README.md` §2.1): the keystone re-run first (because a design premise that no
longer holds changes everything), then the numbers only an implementation can give — the registration cost of a
real 19-41 GiB region, the load-time fill rate, and the model-level read-rate A/B the flag exists for.

Box unchanged: `halo`, Ryzen AI MAX+ 395 / gfx1151, UMA, 123 GiB, kernel 7.2.13, ROCm 10.1.0
(`/opt/rocm-10.1.0-gfx1151`), `iommu=off`, `HIP_VISIBLE_DEVICES=0`.  Arms: `../arms.sh`
(`TREE=~/llama-scratch OUT=/tmp/gtt-arms`), text hashed with **no `-lv`**, censuses at **`-lv 4`**.

### 14.1 The keystone still holds on today's driver (`probe/12-pread-registered`, re-run)

`./12-pread-registered <35B-A3B gguf> 256`, first run of the session (a **cold-ish** page cache):

| arm | result |
|---|---|
| registered `Mapped\|ReadOnly`, prefilled `0xAA` | CPU sum == GPU sum (45 634 027 520) |
| `O_DIRECT` pread **into** the registered region | accepted; 2.09 GB/s first round, **6.44 / 6.42** after |
| GPU read after each pread | sums match **every** round, including the rounds that overwrote the previous expert |
| buffered pread into the same region | **4.50 GB/s** (this session; §13.2 recorded **15.68** on a warm file) |

**The buffered number is page-cache state, not a regression.**  It is the same file, the same region and the
same code; §13.2's run had the file warm.  That is why §14.4 measures the fill at both ends rather than quoting
one figure.

### 14.2 Registration at the GTT store's real extent — cheaper than P1.0 predicted

The GTT store registers **one** untouched `mmap`+`MADV_HUGEPAGE` region (`cudaHostRegister(Mapped|ReadOnly)`), which
is the pre-faulting-free order: registering faults the pages in and the kernel forms the huge pages (§13.1, second
row).  Measured inside the GTT store creation, from the census (extent, mode: 2 runs each on 35B-A3B):

| extent | registration mode | mmap + MADV_HUGEPAGE + register | per GiB |
|---|---|---|---|
| **19 126.5 MiB** (18.7 GiB) | ReadOnly | **482 / 485 ms** | **25.8 / 26.0** |
| 6 144.0 MiB (6 GiB) | ReadOnly | 160 / 161 ms | 26.7 / 26.9 |
| 19 126.5 MiB | WRITABLE (`MOE_EXPERT_GTT_RO=0`) | 483 / 486 ms | 25.9 / 26.0 |
| **40 925.6 MiB** (39.9 GiB) | ReadOnly | **1 034 ms** | **25.9** |

* **~26 ms/GiB**, against P1.0's **30.5 ms/GiB** for an untouched region (§13.1) and **17.4 ms/GiB** for a
  pre-faulted one.  So the `memset` pre-fault walk the probe needed is not worth taking at the GTT store's scale:
  registering the untouched region costs about what pre-faulting alone did.
* Registration mode makes **no difference to the cost** (25.8 vs 25.9 ms/GiB) — only to the read rate (§14.5).
* The **whole** GTT store's registration is **0.48 s on a 19 GiB tier and 1.03 s on the 41 GiB Flash-Next tier** -- and
  the per-GiB figure does not drift with size (25.9 ms/GiB at 6, 19 and 41 GiB), so it is a linear, predictable
  startup cost.
* **All 19 126 MiB of it lands as huge pages**: `AnonHugePages` 19 126.0 MiB of 19 126.5 (100.0 %), and 6 144.0 of
  6 144.5 on the `-lme 6` arm.  `AnonHugePages` is read from `/proc/self/smaps` **after the load**, i.e. after the
  loader wrote every byte — so the fill does not split them.

### 14.3 The load-time fill (P2's, not P4's)

The fill is the buffered `pread` arm straight into the slot (`read_raw` into `cur->data`, `llama-model-loader.cpp`
→ `load_data_for`): no alignment round-out, no bounce buffer, no staging ring.  The measured window is
`carve → census` (the other tensors of the load are `mmap`-mapped, so they are nearly free in that window):

| model | GTT store read from the GGUF | fill window | rate | page cache |
|---|---|---|---|---|
| 35B-A3B Q4_K_M | **18 662.0 MiB** | 1.34 s | **13.9 GB/s** | warm (the file had just been read by earlier arms) |
| Flash-Next GSQ-IQ3_XXS | **40 925.0 MiB** | 17.6 s | **2.33 GB/s** | cold-ish (46 GiB file on a box holding 37 GiB of cache) |

**The fill is page-cache/disk-bound, and that is a LOAD cost, not a decode cost.**  A 41 GiB GTT store takes ~18 s to
populate when its file is not resident, ~3 s when it is; it is the same 2.3-15.7 GB/s band §13.2 measured for
`pread` into a registered region, and P4's pipelining does not touch it (P4 pipelines *miss* fills, this is the
one-shot load fill).  The bytes read account for every expert byte of the carve (the carve is 0.2 MiB larger on Flash-Next: the
buffer type's per-tensor page padding), and `size_done`'s progress is unaffected.

### 14.4 `-lme` is the GTT store's size (P1.1) — realised, on both models

Census, `-lv 4` (`arms.sh census`).  "Slots" = `(tensor, expert)` slabs; `auto` on a fitting set is the whole set.

| arm | budget source | extent requested → realised | tables / slots | carved | evictions |
|---|---|---|---|---|---|
| 35B-A3B, `-lme auto` | `auto: 80% of MemAvailable -> a 92 413.5 MiB cap` | 19 126.5 → **19 126.5 MiB** | 123 / 31 488 (the **whole** set) | 18 662.0 MiB | **0** |
| 35B-A3B, `-lme 6` | `-lme 6` | 6 144.0 → **6 144.0 MiB** | 36 / 9 216 (a hot subset) | 6 116.0 MiB | n/a (none in this phase) |
| 35B-A3B, `MOE_HOST_LIMIT_FRAC=5` | `auto: 5% ... -> a 5 804.7 MiB cap` | 5 804.7 → **5 804.7 MiB** | 34 / 8 704 (a hot subset) | 5 796.0 MiB | n/a |
| Flash-Next, `-lme auto` | `auto: 80% ... -> a 92 613.3 MiB cap` | 40 925.6 → **40 925.6 MiB** | 144 / 73 728 (the **whole** set) | 40 925.2 MiB | **0** |

* `-lme 6` realises **exactly 6 144.0 MiB** — the extent IS the request (page-rounded, and the request is a whole
  number of pages).  The **carved** bytes are 6 116.0 MiB, i.e. within **one expert table** (155 MiB) of the
  budget, which is the granularity of this phase: a whole table per residency decision, because there is no
  eviction yet (P3 is what makes one *slot* the unit).
* `auto` is `min(MOE_HOST_LIMIT_FRAC x MemAvailable, the expert set)` — a **cap, not a floor**: 35B-A3B's
  19 126.0 MiB set fits the 92 GiB ceiling so the whole set is resident and nothing can evict; Flash-Next's
  40 925.0 MiB set also fits on this 123 GiB box.  `MOE_HOST_LIMIT_FRAC=5` exercises the **not-fitting** band: a
  5 804.7 MiB hot subset 34 tables deep.
* The Flash-Next counts match §13.5's heat-map geometry exactly — **144 tables, 73 728 slots, 39.97 GiB** — which
  is the cross-check that the GTT store holds the same set the heat map ranks.
* 35B-A3B's 123 tables / 19 126 MiB are 3 x 41 layers: the **MTP (nextn) head's** three expert tables are included
  here, where the delivery's cache keys only the 40 trunk layers (§13.5's 120 tables / 18.22 GiB).

### 14.5 The read rate at model level, and what `MOE_EXPERT_GTT_RO` buys

`llama-bench -m 35B-A3B -ngl 99 -p 512 -n 128 -r 3`, two independent runs of the whole set of arms:

| arm | pp512 (run 1) | pp512 (run 2) | tg128 |
|---|---|---|---|
| `MOE_EXPERT_GTT=0` (plain device buffer, today) | 2 268.65 ± 26.67 | **2 289.39 ± 4.78** | 63.22-63.25 |
| GTT store, registered **ReadOnly** (default) | 2 259.63 ± 15.37 | 2 260.77 ± 8.68 | 63.24 |
| GTT store, registered **WRITABLE** (`RO=0`) | 2 135.82 ± 1.34 | 2 146.85 ± 12.13 | 63.30-63.34 |

* **The registered ReadOnly GTT store is within noise of the plain device buffer**: -0.40 % in run 1, -1.25 % in run 2,
  against a drifts between runs of +0.9 % on the plain arm itself.  The GTT store is **not a slow tier**, which is the
  acceptance criterion P2 set.
* **`MOE_EXPERT_GTT_RO=0` is materially slower**: -5.9 % and -6.2 % of the plain arm, -5.5 % and -5.0 % of the
  ReadOnly GTT store, at a spread of ±1.34 / ±12.13 — i.e. well outside the noise.  The 232-vs-60 GB/s fact of probes
  03/04 **does apply to this region**.
* The magnitude is *masked by compute*: a 3.9x slower read costs 5-6 % of prefill, not 74 %, because the expert
  read overlaps arithmetic and only a fraction of prefill is weight-bandwidth-bound.  Do not read the 60 GB/s
  probe figure as "writable costs 74 % at the model".
* **Decode does not move at all** (63.22-63.34 t/s across all six runs): a decode step reads top-k experts, so it
  never approaches the read bandwidth.  A GTT store's value is in what it holds, not in how fast it reads while
  fitting.

### 14.6 Identity, and the kill switch

Same-seed greedy text, **no `-lv`** (`scripts/extract-generated.py`), prompt `The capital of France is`, `-n 20`:

| model | `MOE_EXPERT_GTT=0` | GTT store `auto` | GTT store `-lme 6` | GTT store `RO=0` | pre-P2 build (`release.json` tree) |
|---|---|---|---|---|---|
| 35B-A3B Q4_K_M | `89 chars sha=627b2ebd7792` | **identical** | **identical** | **identical** | **identical** |
| Flash-Next GSQ-IQ3_XXS | `113 chars sha=f586785d7183` | **identical** | — | — | **identical** |

`-ncmoe 8` on the same box with the GTT store on: identical too (the `-ncmoe` refusal is unchanged, and the GTT store
simply holds the whole set).

The kill switch's log is **byte-for-byte** the release tree's: the two `-lv 4` logs (pre-P2 build vs
`MOE_EXPERT_GTT=0`) normalise to the same 329 lines, differing only in the ephemeral server port and the four
throughput figures.  `scripts/check-gpu-faults.sh --since "90 min ago"` over every arm's log plus the kernel
journal: **clean**, and no `HSA_STATUS_ERROR_MEMORY_FAULT` in any arm.

### 14.7 What this does not measure (and must not be read as measuring)

* **Eviction, the streaming refill, and the `>RAM` case** — P3/P4/P5.  Every arm above has its expert set
  *fitting*; the census prints "a hot subset" when it does not, and says there is no eviction.
* **Pipelining.**  §13.4's requirement (fills behind compute) is untouched: P2 has exactly one fill, at load.
* **`--fit`'s view of the GTT store.**  A `--fit` probe load is `no_alloc`, so the GTT store plan is skipped entirely
  (`llama_model_loader.cpp`: `if (no_alloc) return;`) and the probe reserves nothing — confirmed by exactly ONE
  `plan_gtt_expert_store` line per run with `--fit on` (the default).  But `--fit` therefore does **not** account
  for the GTT store when it sizes the context; that is P1.3's item, and on this box it did not matter.
* **Reclamation under memory pressure** — probe `16`, still owed, and now more load-bearing: the GTT store is
  registered and `MemAvailable` fell by exactly the GTT store's extent on every arm (e.g. 19 126 MiB on the 35B arm).
  Nothing here tests whether the kernel can take it back.
* **gfx1150 / Windows / a carve-out APU**, and any second device.

---

## 15. 2026-11-11 — probe 16: reclamation under pressure.  **`hipHostRegister` does not pin; the GTT store is reclaimable; the "budget is a reservation" claim is FALSE as measured**

`probe/16-reclaim-pressure.cpp`.  The question §5 and §12 built a claim on: probe `09` showed a *user-initiated*
`fadvise(DONTNEED)` + `madvise(DONTNEED)` cannot release a registered range, and the campaign read that as "the
budget is a reservation".  A user discard is not the kernel's reclaim under pressure.  This is the measurement.

Box unchanged (`halo`, gfx1151, 123 GiB, 183.9 GiB swap, kernel 7.2.13, ROCm 10.1.0, `iommu=off`,
`TRANS_HUGEPAGE=madvise`).  Each arm runs in its own process **inside a memory cgroup** --
`systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=4G` -- so the pressure is real but contained (a
machine-wide hog needs ~100 GiB and invites the OOM killer); `PROBE16_HOG_MIB=4096` bounds the hog.  Region
2048 MiB, control (unregistered anon) 2048 MiB, both filled with an offset-dependent pattern.  `ulimit -l` is
**unlimited** on this box, which is why the `mlock` arm exists.

### 15.1 The arms, and the verdict

Every arm's pressure was **real**: the unregistered control lost 17-2048 MiB of pages and the cgroup reached its
`memory.max` with 2.1-2.2 GiB in `memory.swap.current`.

| arm | what the region is | pages the kernel took (MiB), across runs | huge pages after | GPU read after the pressure | `Unevictable` |
|---|---|---|---|---|---|
| `ctl` | plain anon, nothing registered | 2048 (all of it) | lost | n/a | 0 |
| **`reg`** | `mmap`+`MADV_HUGEPAGE`+`hipHostRegister(Mapped\|ReadOnly)` -- **the GTT store's own backing** | **1.1, 28, 212, 436** (and 0 in 2 runs) of 2048 | **split every run**: 2048 → 626-1066 MiB | 29-36 ms when ≤ 28 MiB was taken, **524 ms** when 436 MiB was | **0** |
| `reg_w` | the same, registered writable | 286, 394 (and 0 in 1 run) | split: 2048 → 812-1150 MiB | 78 ms, 395 ms, **2 191 ms** | **0** |
| `hostalloc` | `hipHostMalloc(Mapped\|Portable)` -- a *pinned* allocation | 86 (and 0 in 2 runs) | **never had any** (`AnonHugePages` 0) | 48 ms, 183 ms | **0** |
| **`mlock`** | `mmap`+`MADV_HUGEPAGE`+**`mlock`**+`hipHostRegister(ReadOnly)` | **0, 0, 0, 0** | **2048 MiB preserved** | **26-28 ms** | **2048** (`Mlocked` 2048) |

Four things, in order of how much they change the design:

1. **`hipHostRegister` on an `mmap` region does not pin it on this box.**  Under real pressure the kernel *splits
   the transparent huge pages* (every run: 2048 MiB → 626-1 150 MiB of `AnonHugePages`) and takes 0-436 MiB of the
   region's pages into swap.  The delivery's `ROCm_Host` master is `hipHostMalloc`'d, which is a different
   mechanism; this region is *registered*, not pinned.  (A `hipHostMalloc` region held in 2 of 3 runs and lost
   86 MiB in the third, so it is not an absolute guarantee either -- it is simply not THP-backed.)
2. **It is a slow failure, not a fatal one -- and it is one-off.**  The GPU read after the pressure returned the
   right bytes in **every** arm and run: no `HSA_STATUS_ERROR_MEMORY_FAULT`, because the driver revalidates the
   userptr range at submission and faults the pages back in.  The cost is that first read: **524 ms against 42 ms**
   for 2 GiB when 436 MiB had been taken (12x; 2 191 ms / 37x in the worst `reg_w` run), and the **second** read is
   back to 41-44 ms.  So a swapped-out GTT store does not corrupt and does not crash: it stalls.
3. **The only reliable reservation is `mlock`** -- and it is the only arm that is *visible* to a monitor:
   `Unevictable` / `Mlocked` / `smaps Locked` all report the 2 048 MiB, where **every** other arm reports 0 while
   being reclaimable to some degree.  It also keeps the huge pages (2048 MiB of `AnonHugePages` *after* the
   pressure) and the read rate (26-28 ms), which is what the GTT store wants; `hipHostMalloc` is the opposite
   trade (never reclaimable-ish, never huge).
4. **THP splitting alone costs nothing measurable**: the runs where the kernel split the huge pages but took 0-28 MiB
   read at 29-36 ms, the same as before the pressure.

### 15.2 What this does to the campaign

* **`README.md` §5's "the budget is a reservation" is wrong** and is corrected there: the GTT store as built (P2)
  is *reclaimable under pressure*.  A >RAM configuration cannot lean on the budget for anything except its
  *sizing* intent -- the pages can be swapped, and the GTT store then pays a one-off 10-40x stall on the first read
  that touches the swapped-out part.
* **The fix is one line and it needs the maintainer's call:** `mlock()` the region after the fill (or before the
  registration).  It costs `RLIMIT_MEMLOCK` -- unlimited on this box, but a stock desktop ships 8 MiB, so a real
  GTT store would be refused there unless the process raises the limit or has `CAP_IPC_LOCK`.  The fallback must
  therefore be explicit: either decline the GTT store, or accept "reclaimable, with a stall under pressure" and
  say so in the census.
* **Nothing else changes**: `MemAvailable` is charged the same either way, so `-lme auto`'s arithmetic stands, and
  no fill path, no registration mode and no census field depends on this.
* **Unmeasured, and named as such:** whether a large (`-lme auto`, 19-41 GiB) `mlock`ed region behaves like the
  2 GiB one; whether `mlock` + registration changes the fill rate (the fill writes already-resident pages, so it
  should not); and what `mlock`'s interaction with `hipHostUnregister` is at teardown.

### 15.3 The fix, implemented and measured (same day, P2 follow-up)

`mlock()` the region (`MOE_EXPERT_GTT_MLOCK`, default 1), clamp the extent to `RLIMIT_MEMLOCK`, decline when the
model fits in RAM anyway, refuse when it does not.  Four arms, all on `halo`:

| arm | result |
|---|---|
| unlimited limit (this box), `-lme 6` / `auto` / 41 GiB Flash-Next | **`mlocked 6144.0 / 19 126.5 / 40 925.6 MiB` — a real reservation**, and the census prints the limit it was measured against |
| `mlock`'s cost at creation | **free**: 26.3 ms/GiB at 19 GiB against 25.8-26.0 before it (and 26.0 vs 25.9 at 41 GiB) — the registration has already faulted every page in, so this is a bookkeeping pass over the VMA |
| `ulimit -l 8192` (8 MiB), fitting model, `-lme 6` and `-lme auto` | **declined**, one WARN with the numbers and the five ways to raise the limit; the run continues all-resident and the greedy text is **identical** (`627b2ebd7792`) |
| `ulimit -l 8192` and `33554432` (32 GiB), `Qwen3.5-122B-A10B-Q8_0 -lme 80` | **REFUSED in 0.76 s**, before any byte is read: weights 126 403 MiB, `MemAvailable` 114 656 MiB, shortfall 23 212 MiB, limit 8 MiB / 32 GiB, and (at 32 GiB, where the clamp *would* cover it) the extra line that even a locked store would not make it runnable until P3/P4 |
| `MOE_EXPERT_GTT_MLOCK=1` vs `0` at an 8 MiB limit | `0` skips the whole policy and the census says `NOT locked … RECLAIMABLE under pressure` — the old behaviour, deliberately |

Two notes for the record.  **The refusal is the point**: the same 122B load without it goes to the 2026-10-19
over-commit (121 GiB resident + 13 GiB swap on this 123 GiB box).  And **the decline is not silent**: a fitting
model still gets its GTT store when the limit allows it, and is told exactly why it did not when it does not.

## 16. 2026-11-11 — P3: the GTT store SERVES as the expert ARENA.  `tables=120`, `evictions>0`, and a 123.45 GiB model runs on a 123 GiB box

The P3 session answered the two open bugs `IMPLEMENTATION.md` P3.0 recorded (a `tables=0` census, and an
unbounded allocation) and then measured the tier.  Box unchanged (`halo`, gfx1151, 123 GiB, 123 GiB RAM + 175 GiB
swap, kernel 7.2.13, ROCm 10.1.0, `iommu=off`, `RLIMIT_MEMLOCK` unlimited).  Every arm below is a guarded run
(`/tmp/p3/guard.sh`: `ulimit -v 96 GiB`, `ulimit -l 8 GiB`, plus a 1 s monitor that kills the run the moment
`SwapFree` starts dropping — see §16.4 for which barrier fired and on what).

Models: `M35` = Qwen3.6-35B-A3B Q4_K_M (18.22 GiB of experts, 120 tables), `MFN` = Qwen3.8 Flash-Next GSQ-IQ3_XXS
(40.9 GiB of experts, 144 tables), `MBIG` = Qwen3.5-122B-A10B Q8_0 (119.9 GiB of experts, 144 tables, 126.4 GiB of
weights: it cannot fit in 123 GiB of RAM).

### 16.1 Why `tables=0` — three defects, all in the seam between the loader and the cache

Instrumented with a temporary `MOE_EXPERT_GTT_DEBUG=1` one-shot `fprintf` at each declining seam (removed before
the patch was re-cut).  The first census with the tier on registered **no tables at all**, and these are the three
causes, each independently fatal:

1. **`moe_cache_preflight()` returned above the per-device byte recording.**  The arena tier's early `return true`
   sat *before* the `g_preflight_device_seen` / `g_preflight_device_bytes` block, so on an APU the cache recorded
   "0 host-expert bytes expected".  Consequences, all silent: `moe_cache_tables_complete_locked()` answered
   **complete on the first registered table** (measured: `registered=144.0 MiB expected=0.0 MiB tables=1`, then
   `g_sized=1`), and the two transport queries answered "no opinion" — `moe_cache_pending()` false (so the
   scheduler's registration pre-pass never ran) and `moe_cache_stage_ready()` true.  The fix is to record, THEN
   return: the tier's expected total is the whole host-expert set (18 662 MiB on `M35`), so the arena sizes once,
   on the complete set.
2. **`alloc_all_locked()` clamped the tier's budget against free VRAM / the slab.**  `g_budget` is the store's
   extent on this tier, but the clamp below it subtracted `cudaMemGetInfo` free or the slab's arena capacity — and
   on an APU the slab's arena capacity is **0** (the slab is armed but has no work region), so the budget came out
   0 and every table got `layer 0 cannot hold a uniform arena above 0 slots (requested 0)`.  The clamp is
   meaningless here: the store's extent is already bounded by `-lme` and by `MOE_HOST_LIMIT_FRAC`, and every slot
   comes from `ggml_cuda_gtt_store_reserve()`, which cannot exceed the region.  Skipped for the tier.
3. **The load was not MAPPED.**  `ggml_backend_cuda_device_get_props` reports `mmap_support = false` for an IGPU
   device (upstream's rule), so the AUTO load mode resolves to `none` and the expert master became a **pageable
   copy of the whole set**: `ROCm_HostMap model buffer size = 18 662.00 MiB` of `ggml_aligned_malloc` on `M35`,
   with a one-time WARN.  For a set that does not fit that is the entire memory the tier exists to avoid.  The
   arena arm now forces `use_mmap` (one WARN, `--load-mode mmap` silences it) — after which the same buffers are
   `ROCm_HostMap` ranges over the file mapping and the anonymous copy is gone (`MemAvailable` after the load:
   **93 GiB before, 106 GiB after** on `M35`, i.e. the 18.6 GiB copy).

Also found and fixed while there: the tier was *arming the movable-boundary slab* (through
`ggml_backend_cuda_device_moe_cache_preflight`, which arms on `on == true`), whose size is `free VRAM - reserve` —
on a shared pool that is the whole machine (`SLAB_INIT size(VA)=98.44 GiB` is the recorded over-commit).  The tier
disarms it, and the "cache enabled but the slab is not armed" WARN now excludes the tier (it is correct there).

### 16.2 The tier serves: three arms, three models

Census = `-lv 4` (mechanism); text = a separate run with **no `-lv`** at all, hashed by
`scripts/extract-generated.py` (correctness only — §16.1's `tables=0` is the proof that identical text alone is
not evidence).

| arm | extent | resident | residency | census |
|---|---|---|---|---|
| `M35`, `-lme 4` | 4 096.0 MiB, **80 % evicting** | **120 tables / 6 720 slots / 4 082.3 MiB** | 56 of 256 experts per table | `h=0.5901 (6161/10440 reaches), fills=10 999, evictions=4 279, arena releases=0` |
| `MFN`, `-lme 8` | 8 192.0 MiB | **144 tables / 14 688 slots / 8 153.0 MiB** | 102 of 144 experts per table | `h=0.9447 (14823/15690), fills=15 555, evictions=867, arena releases=0` |
| `MBIG`, `-lme 60` | 61 440.0 MiB | **144 tables / 19 152 slots / 61 047.0 MiB** | 133 of 256 experts per table | `h=0.8786 (11028/12552), fills=20 676, evictions=1 524, arena releases=0` |

`arena=` is the resident bytes by construction (slots × expert_bytes) and sits inside the extent in every arm
(4 082.3 ≤ 4 096, 8 153.0 ≤ 8 192, 61 047.0 ≤ 61 440); the 1.25× cap added to `alloc_all_locked` never fired.  The
tier's own census line prints the two acceptance numbers (`GTT arena tier: N table(s) resident, 0 arena
release(s)`), and `arena alloc failures=0` in every arm.

### 16.3 Text: the eviction path changes *where* the bytes are, never *which*

Same-seed greedy (`--seed 42 --temp 0 -n 20 --no-display-prompt --single-turn`), hashes from
`scripts/extract-generated.py`:

| model | arm | hash |
|---|---|---|
| `M35` | `MOE_EXPERT_GTT=0` (no store) | `627b2ebd7792` |
| `M35` | store `auto` (P2 placement) | `627b2ebd7792` |
| `M35` | `-lme 6` (P2 placement) | `627b2ebd7792` |
| `M35` | **arena `-lme 4`** (80 % evicting) | **`627b2ebd7792`** |
| `M35` | **arena `-lme 6`** | **`627b2ebd7792`** |
| `MFN` | `MOE_EXPERT_GTT=0` / store `auto` | `f586785d7183` |
| `MFN` | **arena `-lme 8`** | **`f586785d7183`** |
| `MBIG` | arena `-lme 60`, two runs | `9ccf62ed39c3`, `9ccf62ed39c3` |
| `MBIG` | arena `-lme 20` (**one third** the residency: 44 slots/table, not 133) | **`9ccf62ed39c3`** |

So the tier's text is identical to the store-off reference on both fitting models, and on `MBIG` a 3× change of
the resident set (`-lme 20` = **44 slots/table**, `h=0.5975`, `evictions=5 052`, against `-lme 60` = **133
slots/table**, `h=0.8786`, `evictions=1 524`) changes no token — which is the eviction invariant (`IMPLEMENTATION.md`
P3 accept) measured directly.  A `-lv 4` census arm of the same model does **not**
reproduce a hash (4663 vs 4797 chars across two `MBIG` censuses): the stream is polluted, as `AGENTS.md` says, so
every hash above comes from its own no-`-lv` run.

### 16.4 The unbounded-allocation bug does NOT reproduce, and the two barriers say why it was not the arena

The pre-fix arms are gone, so this is stated as what was measured on the fixed build, not as a root cause.

* **The literal case-2 configuration now completes**: `M35`, arena `-lme 4`, `-lv 4`, default `-c 262144`, inside
  `systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=4G`, under `ulimit -v 96 GiB` — exit 0, 21.9 t/s
  decode, `SwapFree` **unchanged at 179 958 MiB for the whole run**, and the same with `-c 4096`.  No OOM, no
  cgroup kill, no swap movement.
* **`--fit off` was not the trigger**: the default `--fit on` arm is clean too (fit projected 25 864 of 122 773 MiB
  and changed nothing).  The maintainer's `--fit` hypothesis is recorded as **not confirmed**: the fit probe is
  skipped for `no_alloc` and the tier's own budget never consults it.
* **Barrier 1 (`ulimit -v 96 GiB`) fired on ONE arm, on an allocation the tier has nothing to do with**: `MFN`
  needs `per_layer_token_embd.weight` (27 465 MiB, `add_resident`, `--lazy-mode off`) as a plain CPU buffer, and
  41 GiB of file mapping + ROCm's VA + the 8 GiB store exhausted the cap (`ggml_aligned_malloc: insufficient
  memory (attempted to allocate 27465.95 MB)`).  At `AS_KB=0` the same arm runs (22.6 t/s census, 13.6 t/s text,
  `f586785d7183`).  So the 96 GiB cap is **not** usable on the Flash-Next class of model, and the `-v` barrier's
  "anonymous growth" reading must not be attached to the store.
* **Barrier 2 (the cgroup) never fired** in any fixed-build arm, and the `SwapFree` monitor never had to kill a run.
* **What the original allocation most plausibly was, and what is now true instead**: with `tables=0` the whole
  18.6 GiB expert set was re-uploaded per op through the scheduler's pageable-master bounce path (measured on the
  pre-fix arm: `scheduler pageable-master fallback: 414 ranges, 520.2 MiB served through the per-table bounce;
  pinned bounce ring total 157.8 MiB`) *and* the load held the pageable 18.6 GiB copy *and* the slab was armed.
  All three are now gone for this tier (the copy by forcing the mapped load, the slab by disarming it, the
  per-op bounce by the tables actually registering).  If it recurs, the suspects in `IMPLEMENTATION.md` P3.0
  still stand, but the observation to collect first is the `GTT arena tier:` panic line and `/proc/<pid>/smaps_rollup`
  per mapping — the store's extent cannot be exceeded (`ggml_cuda_gtt_store_reserve` refuses), so an explosion is
  never the arena.

### 16.5 `MBIG`: a 123.45 GiB model on a 123 GiB box (the design's whole point)

`MBIG` is 126 403 MiB of weights against ~115 GiB of `MemAvailable`: it cannot be loaded resident.  With
`MOE_EXPERT_GTT_ARENA=1` and a mapped load, the expert tensors are mapping holes, the store holds the hot subset,
and the rest is read from the page cache / NVMe on demand:

| arm | arena | decode | prompt | result |
|---|---|---|---|---|
| `-lme 60 --fit off -c 4096` | 61 047.0 MiB | **3.5 t/s** | 1.3 t/s | runs to completion, `h=0.8786`, text `9ccf62ed39c3` (×2) |
| `-lme 20 --fit off -c 4096` | **20 196.0 MiB** (144 tables / 6 336 slots) | 3.0 t/s | 2.9 t/s | runs to completion, `h=0.5975`, `evictions=5 052`, **same text** |
| `-lme 60`, `ulimit -l 8 GiB` (the §15.3 clamp path) | — | — | — | would **REFUSE** (a >RAM model needs a locked store); these arms ran at `ulimit -l 96 GiB` |

These are 4-20-token arms on a cold page cache, **not benchmarks**: the 3.5 vs 3.0 t/s difference is the
direction the design predicts (more residency, fewer disk fills) and nothing more.  What they establish is the
qualitative claim the campaign exists for: **a model that cannot fit in RAM runs, and the resident subset is what
carries it.**

### 16.6 What is NOT measured, and is named as such

* **No throughput claim of any kind.**  The tier is opt-in and every arm above is short; the fill path is still
  the scheduler's pinned-bounce H2D (`expert_host_src` → bounce → `copy_slot_from_src`), **not** the plain host
  `memcpy` into the registered slot that P3.0 finding 4 calls for — that needs the ordering argument (a host
  write must follow the previous consumer's reads on the same slot) and belongs to P4 with the measured 12.8 GB/s
  warm arm.
* **P3c (the nextn/MTP head's tables pinned outside the `-lme` arithmetic) is not done.**  Every arm above is a
  plain `llama-cli` run, no MTP drafter, so the accounting question never came up.
* **Multi-device is untested**: this box has one GPU.  The tier reserves per device from that device's store and
  the cap is per device, but a `-sm tensor` split over two APUs has not been run.
* **No eviction *rate* study**: `evictions=4 279` on the `M35` arm is the counter, not a hit-rate-vs-budget curve;
  `MOE_EXPERT_CACHE_TOUCH` is a no-op on this tier by construction (every miss fills), so admission is not tuned.
* **The mlock arm at MBIG's 61 GiB** is a real reservation under §15.3's model, but its own pressure behaviour
  (a 61 GiB `mlock`ed region) was not probe-tested.

## 17. 2026-11-11 — P3c: the MTP/nextn HEAD is pinned at full residency and outside `-lme`'s budget

Maintainer decision (2026-11-11, `IMPLEMENTATION.md` P3): *the nextn (MTP) head's tables are pinned in the GTT
store and are NOT counted against `-lme N`, because the head is metadata overhead read every draft step.*  This is
that accounting, measured.

### 17.1 What the head is, and where it lives

On an MoE model with an embedded nextn block the head is **not a second model**: the MTP draft is a
`LLAMA_CONTEXT_TYPE_MTP` **context on the target's own weights** (`common/speculative.cpp`'s
`/*.shares_model =*/ !has_draft`), and the block's tensors are `layers[n_layer()]` of the same file.  So its
expert tables (`blk.40.ffn_{gate,up,down}_exps` on 35B-A3B, `blk.48.*` on the 122B) are ordinary candidates of
`plan_gtt_expert_store()`'s set, and the scheduler registers them like any other table once a draft context
exists.  Two consequences the code had to fix:

1. **They were in the set even when they were never loaded.**  `weights_map` holds the GGUF index, so
   `padded_total` counted the head's tables on every load (`19 126.5 MiB` for 35B-A3B) while `load_mtp` false
   meant their tensors were never even created: `-lme auto` then carved **464 MiB of store, and read it from the
   GGUF, for three tables no graph would touch** (P2's recorded "123 tables / 19 126.5 MiB" is that waste).
2. **A shared budget would evict them.**  On the arena tier they would get the trunk's uniform slot count
   (56 of 256 on the `-lme 4` arm), so a draft step would miss and evict on the head — for tables that are read
   *every* draft step and cost 464 MiB (35B-A3B) or 2 448 MiB (the 122B).

### 17.2 The rule, as implemented

| | |
|---|---|
| partition | a candidate is the HEAD when `load_mtp` and its layer index is `>= n_layer_trunk` (`hparams.n_layer()`); otherwise it is the trunk.  `load_mtp` false ⇒ the head's tensors are not part of the set at all |
| budget | `-lme N` (or `auto`) buys **trunk** residency only; the store's extent is `min(budget, trunk padded) + head padded` |
| placement mode (P2) | the head's tables are placed **first and unconditionally**, and are marked in a `host_experts_store_pinned` set the census names |
| arena mode (P3) | the loader passes the head's **layer indices** to `moe_cache_preflight_store()`; the cache sizes every table of such a layer at `slots = n_experts`, excludes the head's per-expert bytes from `dev_one_bytes`, subtracts its bytes from the per-device budget, and keeps head layers out of the identity-aware drop pass (they are already full, and charging them to `remain` twice would drop trunk layers) |
| no trunk | a model whose expert tables are **all** head (a standalone MTP drafter loaded as the target) keeps the uniform arithmetic — otherwise `-lme` would stop being a cap |

### 17.3 Measured

`halo`, one GPU, every arm guarded (`ulimit -v`, `ulimit -l`, the `SwapFree` monitor).  Text = a separate run with
**no `-lv`**; census = `-lv 4`.

| arm | extent | resident | head | census |
|---|---|---|---|---|
| 35B-A3B, `-lme 4`, **no drafter** | 4 096.0 MiB | 120 tables / 6 720 slots / 4 082.3 MiB | none (set = 18 662.5 MiB) | `arena releases=0`, `0 head layer(s)` |
| 35B-A3B, `-lme 4`, `--spec-type draft-mtp` | **4 560.0** = 4 096 + 464 | **123 tables / 7 488 slots / 4 546.3 MiB** | **1 layer, 3 tables × 256/256, 464.0 MiB** | `evictions=1 080`, `arena releases=0`, `draft acceptance=1.00000` |
| 122B-A10B, `-lme 60`, no drafter | 61 440.0 | 144 / 19 152 / 61 047.0 | none (set = 117 504.6 MiB) | `evictions=1 413`, releases 0 |
| 122B-A10B, `-lme 60`, MTP | **63 888.0** = 61 440 + 2 448 | **147 / 19 920 / 63 495.0** | **1 layer, 3 × 256/256, 2 448.0 MiB** | `evictions=519`, releases 0, `draft acceptance=1.00000`, 3.4 t/s |

`7 488 = 6 720 + 768` and `19 920 = 19 152 + 768`: the head's three tables hold **all 256 experts each** on top of
the trunk's uniform 56 (35B-A3B) — the trunk's own slot count is unchanged by the head's presence, which is the
whole point of taking its bytes out of the budget.  `arena=` tracks it exactly (4 546.3 = 4 082.3 + 464.0;
63 495.0 = 61 047.0 + 2 448.0) and the `GTT arena tier:` line names the pinning.

**Text** (same-seed greedy, hashed from no-`-lv` runs):

| model | arms | hash |
|---|---|---|
| 35B-A3B | store-off / `auto` / `-lme 6` / arena `-lme 4` / arena `-lme 6`, **each with and without `--spec-type draft-mtp`** (10 arms) | `627b2ebd7792` in all of them |
| 122B-A10B | arena `-lme 60` and `-lme 20`, with and without MTP | `9ccf62ed39c3` in all of them |

So pinning the head changes no token, the 3× residency change still changes none, and the head's presence does not
disturb the trunk's text.  Placement mode's census names it too (`-lme 6 --spec-type draft-mtp`): *"a registered GTT
expert store of 6 608.0 MiB of a 19 126.0 MiB expert set: 39 tables / 9 984 slots … 3 MTP/nextn head table(s) of
the 464.0 MiB head are placed FIRST and pinned outside the budget (P3c)"* — 6 614.0 − 464.0 = 6 150 MiB of trunk,
the same 36 trunk tables the P2 arm had.

**Flash-Next cannot test this**: `--spec-type draft-mtp` fails there with `context type MTP requested but model
doesn't contain MTP layers`, with the store **off** as well, so it is a property of that GGUF, not of the tier.

### 17.4 What is NOT measured

* **No throughput claim.**  The head's tables are 464 MiB / 2 448 MiB and would have been read from the device
  buffer anyway; the case for pinning is *fit* (the extra bytes) and *no churn on a table read every draft step*,
  and neither was measured as a speedup.  The `draft acceptance=1.00000` lines above are a 2-token sanity
  signal, not the MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`), which this change owes
  before any promotion.
* **The trunk's own numbers moved for a non-MTP load**: the store's set is now the trunk's (`18 662.5` MiB rather
  than `19 126.5` on 35B-A3B, `117 504.6` rather than `119 952.6` on the 122B), so `-lme auto` carves the trunk's
  whole set and no longer spends 464 MiB / 2 448 MiB on unloaded tensors.  That is a P2 behaviour change, and it
  is the reason the `auto` census in §16 changes from 123 tables to 120.
* **Multi-device** is untested (one GPU here); the head's bytes are subtracted per device, which is right under
  `-sm layer` and untested under `-sm tensor`.

## 18. 2026-11-11 — the SEPARATE MTP drafter: one GTT store, two models.  Two real bugs found and fixed, and why the head is *correct* on its own device buffer

`MEASUREMENTS.md` §17 measured the head **embedded** in the target's own file (35B-A3B, the 122B).  The other
shape is a **separate drafter model**, and it is the Flash-Next one:
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` — arch `qwen4exp` with `nextn_shared_target_tensors = true`, 32 tensors,
all of them `blk.48.*` (one nextn block: `ffn_{gate,up,down}_exps` at 850 MiB each = **2 550 MiB**, the whole of
its expert set), used as `--spec-draft-model` with `--spec-type draft-mtp`.  The target
(`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`) reports `n_layer = n_layer_all = 48`: it has no nextn block of its own.

So the two shapes differ in exactly the way that matters for this tier: the embedded head shares the target's
**arena**, the separate one arrives as a **second load** into the same process, where the GTT store is ONE region
per device.

### 18.1 What the separate drafter did before the fix

| arm (`halo`, one GPU) | what happened |
|---|---|
| target arena `-lme 8` + drafter | the drafter's plan tried `gtt_store_open(2 550 MiB)` against the live 8 192 MiB region → **refused** (`a GTT store of 8192.00 MiB is already live on device 0; 2550.01 MiB cannot replace it in place`) → "no GTT expert store on this device; the expert weights keep their layer's device buffer".  Text `f586785d7183` (correct), target arena untouched (144 tables / 8 153.0 MiB) |
| target placement `-lme 6` + drafter | same refusal: the target's placement carves 6 087.6 MiB of a 6 144 MiB extent, so 56.5 MiB free and no store for the head.  Text `f586785d7183` |
| target `auto` + drafter | the whole 40.9 GiB set is carved; nothing left.  Text `f586785d7183` |
| **target arena `-lme 1` + drafter** | **the run FAILED, exit 1**: `-lme 1` is below the cache's 2 048 MiB arena floor, so *both* models fell back to whole-tensor placement; the drafter's `open` matched the live 1 024 MiB extent exactly (the idempotent branch), it carved, and `ggml_backend_cuda_gtt_store_buffer_type_alloc_buffer` returned **null** → `llama_model_load: error loading model: unable to allocate ROCmGTTStore0 buffer` → `failed to load draft model` |

**Two real bugs**, both pre-existing (P2, not P3c) and both in the same seam:

1. **A failed carve is FATAL, and the code said the opposite.**  `alloc_buffer`'s comment claimed "a DECLINE, not an
   abort: the loader places the tensors it cannot fit in the plain device buffer".  It does not:
   `llama_model.cpp` turns a null buffer into `unable to allocate ... buffer` and the model does not load.  Fixed
   in two places: the loader now sizes its placement against the store's **real free bytes**
   (`ggml_backend_dev_gtt_store_free_bytes`, a new hook) so it never asks for a carve it cannot get, and the
   message is now an ERROR that names the largest free run and says the load cannot proceed.
2. **The store buffer released nothing** (`free_buffer` was a no-op, "never released"), so a second model in one
   process found the region exhausted and **silently ran on device buffers** — i.e. `llama-bench -m A -m B`
   measured its two rows under *different* expert tiers.  `free_buffer` now hands the range back to the store's
   allocator (the region, its registration and its extent are untouched).

### 18.2 The rule for a second load, and what it does

The loader asks the device how much the **live** store can still hand out (`SIZE_MAX` = nothing live, so it sizes
its own).  When a store is live, that free space is the entire budget, and the load:

* **never sizes the store and never grows it** — the extent is the first model's, clamped for that model against
  `RLIMIT_MEMLOCK` and 90 % of `MemAvailable`;
* **never takes the arena tier** — the hook reports **0** free when the live store is the cache's arena, because
  its extent *is* the trunk's budget and the cache reserves its slots from the same free list.  Measured
  consequence of getting this wrong: the head's 2 550 MiB would come out of an 8 192 MiB arena, i.e. the trunk's
  residency silently drops from 100 % to 69 %;
* **places within what is free, or declines cleanly** with a message that says which case it is;
* **adopts** the live region when its set does fit the free space (`ggml_cuda_gtt_store_open` accepts a *smaller*
  request when the free run covers it; a *larger* one is still refused — the region cannot grow in place).

### 18.3 Measured after the fix (all text from separate no-`-lv` runs)

| arm | result |
|---|---|
| Flash-Next + separate drafter: store off / `auto` / `-lme 6` / arena `-lme 8` | text **`f586785d7183`** in all four (identical to the store-off arm and to the no-drafter reference) |
| Flash-Next + separate drafter, arena `-lme 8` (census) | the target's arena is **untouched**: 144 tables / 14 688 slots / **8 153.0 of 8 192 MiB**, `evictions=0`, `arena releases=0`; the head is declined with the new message ("the region is either this run's ARENA … or a placement carve that filled it") |
| Flash-Next + separate drafter, placement `-lme 6` (census) | target carves 23 tables / 6 087.6 MiB of 6 144; the head is declined with "has 56.5 MiB free, less than the smallest expert table (850.0 MiB)" |
| **Flash-Next + separate drafter, arena `-lme 1`** | **exit 0** (was exit 1): "has 0.0 MiB free … this load places nothing in it", head on its device buffer, target unchanged |
| `llama-bench -m Flash-Next -m 35B-A3B -p 32 -n 8` (two models, one process) | **adoption measured**: model 1 creates 40 925.6 MiB (`auto`) and carves its whole set; at teardown the carve is released; model 2 sees "40 925.6 MiB free" and `this load adopts the live 40925.56 MiB GTT store, placing 18662.47 MiB of expert tables inside it` — so **both bench rows are measured under the same expert tier** |
| 35B-A3B regressions (store off / `auto` / `-lme 6` / arena `-lme 4` / arena `-lme 4` + `--spec-type draft-mtp`) | `627b2ebd7792` in all five |
| 122B-A10B arena `-lme 60` regression | 144 tables / 19 152 slots / 61 047.0 MiB, `evictions=1 380`, `arena releases=0` |

`check-gpu-faults.sh` clean over every arm; no `HSA_STATUS_ERROR_MEMORY_FAULT`.

### 18.4 So: is the separate-drafter head handled correctly?  Yes — and on UMA it needs no pinning

The head's 2 550 MiB stays in **its layer's device buffer**, and on this platform that is the *right* answer rather
than a fallback:

* UMA means the "device buffer" is **the same host RAM** the store would have held — there is no second copy and no
  memory win from moving it into the store;
* it is **already fully resident** (a plain device allocation of the whole set), so there is no residency policy to
  pin it against and **no churn** — the two things P3c exists for apply to the *embedded* head, whose tables share
  the trunk's arena and its uniform slot count;
* taking it out of the arena would *shrink the trunk's residency*, which is the wrong trade (§18.2).

That last point is the reason the tier does not try to serve a separate head from the arena.  If a future
configuration needs the head's bytes *inside* the region (a box where RAM, not the region, is the binding
constraint), the size has to be decided when the **target** sizes the store — the drafter loads afterwards and the
region cannot grow — so that is a `common`/`--fit` sizing term (it already knows the draft path and estimates the
draft context, `common_draft_reserve_estimate`), not a loader change.  Recorded as the open item it is.

### 18.5 What is NOT measured

* No throughput claim of any kind, here or for §17 (the `llama-bench` numbers above were run for the adoption
  behaviour, not as a perf claim).
* The **multi-device** case of a second load (one GPU here); the free-bytes query is per device and the loader
  takes the minimum over the IGPU devices.
* The adoption path's *failure* arm (a second model whose set does not fit the live store) is covered only by the
  Flash-Next drafter declines above, not by a synthetic two-large-models arm.

## 19. 2026-11-11 — the draft weights are never evicted (the invariant, and the census that asserts it)

Maintainer's rule (2026-11-11): **"the draft weights (whether embedded or a separate file) are not subject to
being evicted."**  This is how that is enforced and what proves it.

### 19.1 Why the embedded head cannot be evicted, and the three paths that had to be told

The head is sized at `slots = n_experts` (§17.2), i.e. **one slot per expert**.  `access_locked()` evicts only when
a table has no free slot, and a table with one slot per expert always has a free slot for an expert it does not
hold — so *the policy itself can never evict a head entry*.  What needed explicit work is everything that takes a
table down **outside** the policy, and there were three:

| path | what it does | now |
|---|---|---|
| `moe_cache_shrink_step()` | stands down the LARGEST table arena to satisfy a failed compute allocation (on the APU that is a host-memory "arena", so it buys nothing anyway) | skips pinned head tables: the trunk gives up its bytes instead, and a head layer is a whole layer, so standing one table down would also break slot-uniformity |
| `moe_cache_shrink_arena()` | releases the largest table arenas until `need_bytes` is free | the candidate order excludes pinned head tables |
| `moe_cache_rearm()` | re-arms stood-down tables after a compute-buffer drop | head tables are excluded explicitly (not merely implied by `slots == 0`), so no future path can re-arm one at a lower count |

Plus the two already in place: the identity-aware drop pass (`MEASUREMENTS.md` §17.2) and the budget arithmetic
(the head's bytes are outside `-lme`, so the trunk's shortfall never falls on the head).

### 19.2 The census asserts it

A new line, printed with the arena tier's census (`-lv 4`), and an ERROR on either violation:

```
report_locked: MTP/nextn head residency: 3 table(s), 768/768 (tensor, expert) slots resident, 0 eviction(s)
               (MUST be 0: the draft weights are pinned and never evicted)
```

* `slots == experts` — every expert of every head table is resident (0 churn on the draft path);
* `evictions == 0` — the rule the maintainer asked for, asserted rather than assumed;
* an `ERROR` line if either fails, naming the config, so a gate (or a reader) can grep for it.

### 19.3 Measured under trunk pressure (the point of the exercise)

Same-seed arms at a deliberately starved trunk, so the policy is churning hard while the head must not move:

| arm | trunk | head |
|---|---|---|
| 35B-A3B, arena `-lme 2`, `--spec-type draft-mtp`, 32 tokens | 123 tables / 4 128 slots / 2 505.2 MiB, `h=0.7945`, **508 evictions** | **3 tables, 768/768, 0 evictions** |
| 35B-A3B, arena `-lme 4`, `--spec-type draft-mtp`, 32 tokens | 123 tables / 7 488 slots / 4 546.3 MiB, `h=0.7929`, **7 670 evictions** | **3 tables, 768/768, 0 evictions** |
| 122B-A10B, arena `-lme 20`, `--spec-type draft-mtp`, 8 tokens | 147 tables / 7 104 slots / 22 644.0 MiB, `h=0.7683`, **3 036 evictions** | **3 tables, 768/768, 0 evictions** (the head is 2 448 MiB here) |

7 670 trunk evictions against 0 on the head, and the same at the 2.4 GiB head size, is the invariant holding under
exactly the condition it exists for.  Text unchanged (`627b2ebd7792` for 35B-A3B in the store-off / arena `-lme 4`
/ arena `-lme 2` + MTP arms re-run after this change); `check-gpu-faults.sh` clean.

### 19.4 The separate file

A separate drafter's tables are a whole-set **device** allocation and are **not cache tables at all** (§18.4), so
no cache path can evict them — the census correctly reports `MTP/nextn head residency: 0 table(s)` for that shape,
and the loader's decline message says so in as many words ("it is already fully resident, and it is NOT a cache
table -- so these draft weights cannot be evicted").

### 19.5 What is NOT measured

* The three pressure paths are **guarded but not force-tested**: `moe_cache_shrink_step` and
  `moe_cache_shrink_arena` need a failed compute allocation on this 123 GiB APU, and the re-arm needs a
  compute-buffer drop.  What is tested is the *policy* path (which is the one that runs every step) and the
  code-level reason the three others can no longer select a head table.
* `moe_cache_release_arena()` — the last-resort whole-arena release — still takes **every** table, head included,
  and disables the cache.  It is unchanged and out of scope here (it is the "the crate is on fire" path); on the
  arena tier it would also break the mapping holes generally, which is a pre-existing property of the tier and not
  specific to the draft side.

## 20. 2026-11-11 — P4a: the tier's fills run on **r262's adaptive host-read transport**, and two bugs had to die first

r262 (block 12 amendment, `TODO` #306) shipped the transport P4 was going to invent: a `map -> file -> dio`
ladder driven by `--load-mode` (`auto` = adaptive), a stall detector (K x this arm's own EWMA) and a probe-fill
A/B (every Nth fill served by the next arm, same bytes).  So P4 stops being a *design* and becomes a *routing*
job -- which is also the campaign's own rule ("do not write a second policy").  This is the rebase onto r262 plus
that routing, measured.

### 20.1 What was already there, and the two places the tier fell off it

`pool_transport_apply_locked()` maps the ladder onto the legacy flags the bounce path reads
(`g_pool_dio`, `g_pool_fill_src_file`, `g_pool_demand_src_file`), so the read path itself needed no change.  Two
things did:

1. **The ladder was never INITIALISED on the tier.**  r262 applies it at the first **pool** build
   (`pool_transport_init_locked`, called from the pool sizing), and an APU has no pool -- pinning is refused on an
   integrated device, and the tier serves pageable fills from the per-table bounce ring instead.  Consequence,
   measured: `MOE_HOST_POOL_TRANSPORT=file` and `=dio` both reported `map`, and `--load-mode buffered|dio` did
   nothing.  Fixed by applying the mode at the tier's own sizing (`alloc_all_locked`, `g_gtt_arena` only): a pin is
   applied as itself, and `auto` starts optimistically on the mapping — which is what the bounce path already did,
   so the first apply changes no behaviour, it makes the decision visible and keeps the legacy flags consistent
   with the arm the census reports.
2. **The `file`/`dio` arms could be selected and then read the mapping anyway.**  `moe_cache_set_host_source()`
   returned early when the host pool was off, so `t.src_path`/`src_known` were never set on the tier, and the
   bounce path's storage branch requires them (`t.src_known && !t.src_path.empty()`).  Measured: the `auto` ladder
   escalated to `file` and the census then showed **zero storage reads** — the arm was a label.  Fixed by recording
   the source regardless of the pool (one map entry per expert tensor; nothing is pinned, the pool stays inert) and
   letting the reader decide.  This is inert by default: `g_pool_dio` and both `*_src_file` flags stay off, so the
   delivered arm is still the mapping memcpy.

Plus census: the `host read transport:` line was gated on **pool fills** (an APU never has any), so the tier's arm
was invisible; it now prints when a mode is set and any read happened, the `GTT arena tier:` line names the arm
(`read transport %s (%s, %d switch(es))`), and the bounce path's probe samples are counted in
`probe N calls / M ms` like the pool's (they were not, so a run whose ladder moved on probes said "probe 0 calls").

### 20.2 The three arms, measured (`halo`, 35B-A3B, arena `-lme 4`, 0.61 MiB/read, `-n 8`)

Per-expert **fill** reads into the bounce entry (the H2D follows and is unchanged):

| arm | warm page cache | cold page cache (`fadvise(DONTNEED)` on the GGUF before the arm) |
|---|---|---|
| `map` (the mapping memcpy — the delivered arm) | 11 637 reads / 7 062.2 MiB / 421 ms = **16 776 MiB/s** | 11 613 / 7 048.1 MiB / 5 306 ms = **1 328 MiB/s** |
| `file` (buffered `pread`) | 11 637 / 7 062.1 MiB / 318 ms = **22 211 MiB/s** | 11 612 / 7 047.3 MiB / 3 650 ms = **1 931 MiB/s** (**+45 %**) |
| `dio` (`O_DIRECT`) | 11 633 / 7 060.1 MiB / 4 645 ms = **1 520 MiB/s** | — (bypasses the cache by definition) |

Two things this says about *this tier*, and they are not what the pool measured:

* **The cold-cache collapse is the tier's problem too** (1 328 MiB/s against 16 776 warm, 12.6x), and the buffered
  arm is **45 % faster** there — the page-fault storm is the mapping's cost, exactly the mechanism r262 documented.
* **The fill (unlike the pool's fill) does not favour the mapping even when warm**: a buffered `pread` of
  already-cached pages beats a memcpy out of a perfectly warm mapping (22 211 against 16 776 MiB/s, 24 %), because
  the mapping still pays a PTE lookup per 4 KiB page while `pread` copies in one syscall with readahead.  `dio` at
  1 520 MiB/s is the reason `MOE_HOST_POOL_TR_PROBE_GAIN_DIO` (3x) exists: bypassing the cache costs 11x here.

**`auto` on a cold page cache** (the ladder, end to end): states `map` at 1.24 s ("optimistic start"), the
probe-fill A/B then reports `file` served the same read in **28 us against this arm's 294 us, 3 samples running**
and latches `file` — after which the run's reads are 10 256 storage reads at **2 086 MiB/s** plus the 1 270 mapping
reads from before the switch.  One switch, one direction, no flapping.

### 20.3 Correctness: the transport never changes a byte

Same-seed greedy text, hashed from **no-`-lv`** runs, all `627b2ebd7792` (the store-off reference):

* the three pinned arms (`MOE_HOST_POOL_TRANSPORT=map|file|dio`);
* all four loader modes (`--load-mode mmap|buffered|dio|auto`);
* `map` and `file` **on a cold page cache** (cache dropped before each arm);
* **an arm where every fill FAILS** (`MOE_HOST_POOL_FILL_FAULT=late`): 1 886 failed fills, each rolled back and its
  routed op declined (`FAILED fills: 1886 (each one was rolled back -- the slot is not resident and the routed op
  was declined…)`), `h=0.0053`, `evictions=0`, and the text is **byte-identical** — the fault arm P4 owed, and it
  does not corrupt, does not fault and does not need a new env var (r262's knob is the tier's knob).

Regressions on the same build: 35B-A3B store-off / `auto` / `-lme 6` / arena `-lme 4` / arena `-lme 4` + MTP all
`627b2ebd7792`; Flash-Next arena `-lme 8` and Flash-Next + the SEPARATE drafter `f586785d7183`; the 122B arena
`-lme 60` `9ccf62ed39c3`.  `check-gpu-faults.sh` clean over every arm.

### 20.4 What P4 still owes, and what changed about it

* **P4b (the destination side) is now the win, and it is newly attractive**: the `file`/`dio` arms read into a host
  bounce entry and then H2D into the arena — but the arena IS registered host memory, so a `pread` can land
  **straight in the slot** (probe `12` measured that as legal and coherent; 4096-aligned with the per-fill delta).
  That deletes a copy and a transfer for two of the three arms, and it is now *reachable* because the source is
  registered (§20.1 (2)) — before this change the file arms could not read at all.
* **The `map` arm keeps bounce + H2D**: a direct host `memcpy` into a slot still needs the ordering argument (a host
  write must follow the previous consumer's reads of that slot; the stream-ordered copy gets it for free).  The
  ladder makes this optional rather than required, which is the right order to do it in.
* **P4e (the overlap experiment) is unchanged** and still comes before any worker: r261 measured ±1.5 % for the
  exact within-layer lookahead on the discrete pool.
* **Not measured**: the discrete-side effect of the source-registration change (no discrete GPU on `halo`;
  `scripts/gate-host-pool.sh` is the gate, and the change is inert unless a transport arm is pinned — the pool's
  own path already registered its sources); multi-device (one GPU); and any throughput claim (the numbers above are
  per-read rates, not decode).

## 21. 2026-11-11 — P4b: the buffered arm fills **straight into the arena slot** (no bounce entry, no H2D) — the campaign's first decode win

The premise P3.0 wrote down and P4a made reachable: the arena **is** registered host memory (one address,
`iommu=off`), so a per-expert read can land in it instead of going `pread` -> pinned bounce entry -> H2D.  Probe
`12` had already measured that as legal *and* coherent (`pread` into the registered region, GPU read, including
the reuse arm).

### 21.1 The rule, and the two arms that keep the bounce path

| arm | direct fill? | why |
|---|---|---|
| `file` (buffered `pread`) | **YES** | no alignment constraint, so the read is exactly `expert_bytes` from `src_offs + e*host_bytes + src_off` into `dst` — no delta, no tail |
| `dio` (`O_DIRECT`) | no | `O_DIRECT` needs 4096-aligned offset *and* length: the round-out makes the report overrun the slot's end by up to 4095 bytes into the **next** slot's head, which holds a resident expert.  A safe version needs a tail splice; `dio` measured 11x slower than the mapping here (§20.2), so it is not worth it yet |
| `map` (the mapping `memcpy`) | no | the source is host memory anyway, so the copy would have to be a host `memcpy` into the slot, and *that* needs the ordering drain with no transfer to hide behind — strictly worse than the stream-ordered copy it replaces |

Guards: `g_gtt_arena` (the destination must be registered host memory — the same helper on the discrete pool path
would be handed a device pointer), a known source, one contiguous run (`n_slices == 0`, not a strided axis-0 split),
`src_off + expert_bytes <= host_bytes`, and **a stream that is not capturing** (a `cudaStreamSynchronize` is illegal
inside a capture; those fills take the bounce path and the census counts them).  `MOE_EXPERT_GTT_FILL_DIRECT=0` is
the A/B arm and the kill switch.  The same read serves the **sizing** fills (heat seed, identity, prefill seed),
which pass stream 0 before any graph exists — so they need no drain at all.

**The ordering drain**: a host write must follow any in-flight read of the slot (the fused MoE kernels take the
arena address as a launch parameter, so the previous ubatch may still be reading the slot being refilled).  The
bounce path gets that for free from the stream-ordered copy; the direct path drains the stream and **counts the
drain separately**, because that cost is the whole question.  Measured: **0.005 ms/fill** (122-131 ms across
26 536 fills), i.e. noise.

### 21.2 Measured (`halo`, 35B-A3B, arena `-lme 4`, buffered arm, `-n 96`, 3 reps warm / 2 cold)

| arm | warm page cache | cold page cache |
|---|---|---|
| bounce (`FILL_DIRECT=0`) | **24.5 / 24.5 / 24.4** t/s | **19.1 / 19.1** t/s |
| **direct (`FILL_DIRECT=1`)** | **25.8 / 28.2 / 28.2** t/s (**+15 %**, median 28.2 vs 24.5) | **21.3 / 21.6** t/s (**+12 %**) |

Text identical in every arm (`627b2ebd7792`): direct on/off, warm and cold, and in the **fill-fault** arm
(`MOE_HOST_POOL_FILL_FAULT=late` with direct on) — the fault arm still declines and still answers correctly.

Why it wins: the read rate *alone* looks worse (10 988 MiB/s direct against 18 010 through the bounce on a warm
cache — the bounce entry is one hot 0.61 MiB buffer while a direct fill writes a different slot spread across the
4 GiB arena, so it pays TLB/cache misses), but the bounce path also moves the **same bytes again** as an H2D
(16 GiB per 96-token arm) on the compute stream, and that transfer is what the direct fill deletes.  End to end
the deletion is worth more than the read-rate loss: +15 % warm, +12 % cold.

### 21.3 What this says about the ladder's start arm (a finding, not yet a change)

With `--load-mode auto` (the default) the ladder starts on `map` and only falls to `file` on a probe or a stall.
On this tier that start arm is now the *wrong* one:

| default (`auto`), warm, 3 reps | arm it settled on | t/s |
|---|---|---|
| rep 1 | `map` (bounce) | 25.1 |
| rep 2 | `map` (bounce) | 25.3 |
| rep 3 | `file` (escalated on a probe) -> **direct** | **28.2** |

So `auto` leaves ~12 % on the table until the probe happens to latch `file`, and the probe rarely does on a warm
box: it compares the *transport* only (both samples land in the bounce entry), where the mapping and a buffered
read are near-equal once the pages are cached (`§20.2`: 16 776 against 22 211 MiB/s, and the probe needs a 1.5x
win, 3 times running).  Two candidate fixes, both of which are **policy decisions for the maintainer** because
r262 chose the optimistic-map start deliberately:

1. a tier-specific start arm (`file` when the direct fill is available), or
2. a direct-aware probe on this tier — the sample served into the slot, so the comparison is the end-to-end one
   (map+bounce+H2D against file+direct).

Neither is implemented; documented so the choice is visible (`IMPLEMENTATION.md` P4b).

### 21.4 Not measured

* **Multi-device** (one GPU); the guard is per table and per device, untested on a split.
* **`dio`'s direct path**: not implemented (the tail splice), and `dio` is 11x slower here anyway.
* **A long arm**: `-n 96` at ~25-28 t/s is ~3.5 s of decode; the percentages are consistent across reps but this
  is not the MTP four-axis protocol, and there is still **no throughput claim** for the tier as a whole — this is
  the first *decode* number the campaign has, on one box, and it is reported as such.
* The discrete-side effect of nothing here: the direct fill is `g_gtt_arena`-only.

## 22. 2026-11-11 — P4e, the overlap experiment: there is **nothing to hide a fill behind**.  Drop the worker.  (And what the ladder's latch rate means for the start-arm question)

IMPLEMENTATION.md P4e set the rule: *reproduce the question before building anything* — is there GPU work in flight
when a demand fill is issued?  r261 had refuted the premise for the discrete pool (the exact within-layer
lookahead measured ±1.5 %, `archive/work/host-pool-demand-overlap/`).  This is the measurement for this tier, and
the answer is the same one, for a different reason.

### 22.1 The measurement is the drain, and the mechanism says why

A fill has work to hide behind only if the stream it must order behind is busy.  P4b's direct fill already
drains that stream before writing the slot (§21.1), so **the drain IS the instrument**: it is the in-flight work,
by construction, per fill.  P4b also showed the drain is tiny (0.005-0.01 ms/fill), which was the first hint.

The mechanism, from the code: on the decode band the fills are issued from `devremap_prep_locked()`, the one-call-late
policy of the sync-free device remap, and that path only waits for the **previous call's remap kernel** to publish
its lag (`__threadfence_system` + a flag, or an event).  It never drains the previous token's MoE compute for the
fill.  The previous graph's compute has nonetheless completed by the time the policy runs — the measurement below
is the proof, and it matches r261's finding (the host materialises a split's host-weight inputs, synchronize
included, before dispatching it).

### 22.2 The drain distribution (35B-A3B, buffered + direct fills, `-n 96`)

| arm | fills | longest drain | fills that waited >= 100 us | >= 1 ms | >= 5 ms |
|---|---|---|---|---|---|
| `-lme 4`, cold page cache | 35 962 | **263 us** | 120 | **0** | **0** |
| `-lme 4`, warm | 35 966 | **173 us** | 26 | **0** | **0** |
| `-lme 8` / `12` / `24`, warm | 22 608-22 630 | **187-199 us** | 13-24 | **0** | **0** |

So at the instant a fill is issued there is **at most ~0.2 ms** of GPU work queued, and never a millisecond.  A
fill costs 30-200 us.  The maximum a worker could hide is therefore ~0.2 ms per token, against a 23-36 ms token:
**<= 0.9 %**, and that upper bound assumes the worker is free and perfectly timed.

### 22.3 The cross-check: the fill is ON the critical path, not hidden by it

If the fills were overlapped, spending more residency would buy little; if they are serialized, it buys their
whole cost.  Residency sweep, buffered + direct, warm, `-n 96`:

| `-lme` | decode | fills | host read |
|---|---|---|---|
| 4 | **28.7 t/s** | 35 966 | 1 454 ms |
| 8 | **42.9 t/s** (+49 %) | 22 608 | 889 ms |
| 12 | 43.0 t/s | 22 628 | 885 ms |
| 24 | 43.0 t/s | 22 630 | 874 ms |

+49 % from 4 GiB to 8 GiB, then flat: the fills are exactly the cost being removed, i.e. **on the critical path**
(~44 % of the wall clock at `-lme 4`: 35 966 fills at 40 us against 96 tokens at 34.8 ms).  And the flat tail
says the plateau beyond ~44 % residency is the compute bound, not the fill bound.

Together with §21.2 (*the arm that overlaps the H2D is slower than the one that does not*), the answer is not
ambiguous:

> **There is no overlap to be had on the demand/decode path.  Do not build a fill worker** (`copy_stream` /
> `order_event`).  The fill is on the critical path; the levers are FEWER and CHEAPER fills -- residency (`-lme`),
> the direct fill (P4b, +15 %), and read coalescing -- not a pipeline.  `IMPLEMENTATION.md` §7 arm 9 must assert
> THIS answer (no pipeline), which is what P4e asked for.

Prefill is a different band and stays P5's question: there the GPU *is* busy, the reads are bulk, and the tier has
no staged prefill path yet.

### 22.4 The ladder's latch rate, measured (this prices the start-arm options)

`--load-mode auto` starts on `map` and falls to `file` on a probe win or a stall.  With the direct fill, `file` is
the better arm **warm** as well, so how often does `auto` actually get there?  Same seed/prompt, 8 runs total on a
warm cache, `-n 96`:

| batch | runs that latched `file` | decode | runs stuck on `map` | decode |
|---|---|---|---|---|
| first (3 reps, right after a rebuild) | 1 | 28.2 t/s | 2 | 25.1 / 25.3 t/s |
| second (5 reps, warm cache) | **4** | 28.8 t/s | 1 | 25.6 t/s |

> **SUPERSEDED by §23.2/§23.3**: the drift these runs measured was the HEAT MAP warming between them, not
> cache state.  With the prior frozen, `auto` latched `file` **0 of 5** times, i.e. the gap is systematic.
> The "~80 %" below is an artifact; kept for the record.

So ~80 % of runs reach the winning arm and ~20 % stay on `map` and lose **~11 %** decode.  The probe needs 3
consecutive 1.5x wins at a 1/128 cadence, and on a warm box the two transports are nearly level (§20.2: 16 776
against 22 211 MiB/s into the bounce entry), so whether it latches depends on the cache state it happens to see.
Expected loss over the mix: ~0.6 t/s (**~2.4 %**).

### 22.5 The start-arm options, with the numbers behind them

| | what it is | pros | cons |
|---|---|---|---|
| **Option 1** -- tier-specific start arm | when the direct fill is available, start the ladder on `file` instead of `map` | one line; **deterministic** (every run gets 28.8, not 28.8-or-25.6); trivial to gate | forks an r262 decision (`map` is the optimistic start *by design*) on evidence from **one box**; wrong-shaped if the direct fill is off (`FILL_DIRECT=0`) or inapplicable (fused/strided tables keep the bounce path) -- though measured neutral there (24.5 vs 25.1-25.3); a slow-NVMe APU would want `map` |
| **Option 2** -- direct-aware probe | when the direct fill is available, serve the probe's *sample* into the slot too, so the probe compares end-to-end (map+bounce+H2D against file+direct) | keeps r262's **one policy, evidence-driven, one-way** design; self-adapting per box (slow storage stays `map`); no new knob; the probe is a real fill, so no extra I/O | changes what the probe *means* (r262's is deliberately transport-only, shared with the pool) -- a future reader must know the tier's sample includes the destination; **delays the win** (still needs 3 wins at 1/128 cadence, so short runs may never latch); more coupling between the ladder and the store |
| **Option 3** -- leave it | `auto` as it is; `--load-mode buffered` for the win on demand | **zero risk**, r262's policy untouched, nothing to explain later; under memory pressure `auto` latches `file` anyway (that is when the mapping stalls), i.e. a *fittability* tier gets the win exactly where it matters | a warm box loses ~11 % on ~20 % of runs (~2.4 % expected); the fastest configuration needs a flag the user has to know about |

> **SUPERSEDED by §23.3** (frozen-prior measurement): the latch rate is 0/5, so "leave it" is not adequate, and
> Option 2 turns out to need end-to-end fill timing *and* a re-calibrated threshold to work at all.  The revised
> recommendation is **Option 1**.

**Recommendation: Option 3 now, Option 2 at promotion if the tier ever ships default-on** — Option 2 is the only
one that does not fork r262's decision, and the cost of it (a win that arrives after a few hundred fills) is the
price a *fittability* tier can pay; Option 1 is the cheapest deterministic fix and is worth it only if the 1-in-5
variance on a warm box turns out to matter in practice.  Nothing is implemented.

### 22.6 Not measured

* Multi-device, and prefill (`P5`): the overlap answer above is the **demand/decode** path only.
* The drain was instrumented on the direct fill because that is where a drain exists; the bounce path has no drain
  to measure (its copy is stream-ordered), so "the stream is idle at fill time" is inferred for it from the arm
  comparison (§21.2) rather than measured directly.
* The latch rate is 8 runs on one box and one prompt; it is reported as a rate, not as a stable property.

## 23. 2026-11-11 — the heat-map prior was WARMING during my arms: two numbers corrected, one trap for the campaign, and what the corrected latch rate means for the start-arm question

Maintainer's question: *"Is the `moe-heatmap` file marked as read-only?  Is it adapting to the tests being run?"*
Answer: **no, and yes** — and that invalidated one of my own conclusions (§22.4) until it was re-measured.

### 23.1 The facts

* The sidecars are `0644`, owner-writable: `Qwen3.6-35B-A3B-Q4_K_M.moe-heatmap`, `...Flash-Next...moe-heatmap`,
  `...122B-A10B...moe-heatmap`.  **Every arm I ran wrote its own**, atomically (`<path>.tmp.<pid>` -> `fsync` ->
  `rename`) at teardown: the 35B file's mtime was 30 s old when this was asked.
* **It adapts, cumulatively**: `heat_merge_into_loaded_locked()` adds this session's per-expert tally into the
  loaded prior and saves it, and the next run seeds the arena's residency from it
  (`heat-seeded 56/56 slots from the persisted map`).  Measured drift: the census's `heat (R2): total=` grew by
  **exactly 98 760 per run** across nine consecutive arms (10 443 912 -> 11 233 992).
* **The delivery already supports freezing it, and its own gates use that**: `MOE_EXPERT_HEATMAP=load` (aliases
  `ro`, `read-only`) loads and seeds but never writes, and a **read-only file is the freeze signal**
  (`heat_map_open_locked`: *"is read-only; loading and seeding from it, not updating it"*, latched in
  `g_heat_frozen`, and re-checked at every save).  `ENVIRONMENT.md` documents all of it, and
  `scripts/gate-host-pool.sh` defaults **every arm** to `MOE_EXPERT_HEATMAP=load` with the reason written down:
  *"the persisted prior is an INPUT"*.
* **So the campaign's arms were the ones out of contract**, not the delivery: `/tmp/p3/guard.sh` and my ad-hoc
  invocations never set it.  `archive/work/apu-expert-gtt/arms.sh` is being fixed (§23.4).

### 23.2 What it invalidated, and what it did not (all re-measured with the prior FROZEN)

Frozen with `chmod a-w` on the sidecar, verified by the "read-only; loading and seeding" line and an unchanged
`md5sum` afterwards (the file was restored to `0644` and byte-identical).

| number | drifting prior (§21/§22) | **frozen prior** | verdict |
|---|---|---|---|
| P4b direct vs bounce, warm, `-n 96`, alternating 3+3 | 25.8/28.2/28.2 vs 24.4/24.5/24.5 (+13 %) | **28.7/28.8/28.8 vs 24.8/24.9/25.1 (+15.6 %)** | **confirmed**, and tighter -- the earlier spread was prior noise |
| residency sweep, `-lme 4/8/12/24` | 28.7 / 42.9 / 43.0 / 43.0 | **28.8 / 43.2 / 43.1 / 43.0** | **confirmed** (+50 %, then flat); the prior's *ranking* had already converged, only its counts drifted |
| `auto` latch rate, 5 reps warm | **4 of 5 latched `file`** (28.8 t/s) | **0 of 5** (all 25.3-25.5, all on `map`) | **CORRECTED -- the earlier "~80 % latch" was an artifact of the drifting map** |
| text identity (all §21/§22 arms) | `627b2ebd7792` | `627b2ebd7792` | **immune by construction**: the prior changes *which* experts are resident, never the bytes (`ENVIRONMENT.md`: "bit-identical"), and the hashes agree in ~40 arms |

New datum while the switch was in hand: **`MOE_EXPERT_HEATMAP=0` (cold start), buffered+direct, `-lme 4` = 28.8 t/s**
-- the same as the warm prior with the direct fill.  On this tier the prior is **neutral at `-lme 4`**: it decides
*which* experts are resident, and `-lme` decides *how many*, and it is the coverage that carries decode (consistent
with the campaign's own "`-lme` coverage carries the design").

### 23.3 The corrected conclusion for the start-arm question (§22.4/§22.5 superseded)

With a **stable** prior, `--load-mode auto` never reaches `file` on this box: 0 of 5 runs, all 25.4 t/s, against
28.8 achievable.  So the gap is **systematic (~11 %), not 1-in-5**, and the earlier "≈2.4 % expected loss" is
wrong.

That also kills the "Option 3 is adequate" recommendation, and it weakens **Option 2** more than §22.5 admitted:

* r262's probe needs `probe_us * 1.5 < cur_us` on **per-read times into the bounce entry**.  A direct-aware sample
  would still compare *reads*: the tier's win comes from the **deleted H2D**, which no read timer sees.  Served into
  the slot, the sample would measure ~1.15x against a 1.5x threshold -> **it would still not latch**.  A real
  Option 2 therefore needs an *end-to-end* measurement (time the fill until the slot's bytes are usable, i.e. an
  event after the copy), plus a tier-specific threshold -- i.e. a much bigger change to a shared mechanism than
  "serve the sample into the slot".
* **Option 1** (start the ladder on `file` when the direct fill is available) is therefore the proportionate fix:
  one line, deterministic, and structurally safe because it is **neutral on the bounce path** (`file`+bounce
  24.8-25.1 t/s against `map`+bounce 24.4-25.3), so a box whose direct fill is off or inapplicable loses nothing.

**Recommendation (revised): Option 1**, documented as a deliberate, tier-scoped exception to r262's optimistic
`map` start, with `MOE_HOST_POOL_TRANSPORT` still able to pin either arm for an A/B.  Not implemented -- it is a
policy change to an r262 decision and needs the maintainer's nod.

### 23.4 The trap, and the fix in the campaign's own tooling

**Every measurement arm must freeze the prior** (`MOE_EXPERT_HEATMAP=load`, or a read-only sidecar, or a frozen
`MOE_EXPERT_HEATMAP_DIR` copy as `benchmarks/2026-10-26-graph-replay-keys.md` shows).  Otherwise arms contaminate
each other, **run order becomes part of the result**, and a comparison can silently invert -- which is exactly what
happened to the latch rate above.  `arms.sh` now exports `MOE_EXPERT_HEATMAP=load` for every arm it runs, and the
campaign README carries it as a trap; the eventual `gate-uma-experts.sh` must do the same per arm (the delivery's
`gate-host-pool.sh` already does).

## 24. 2026-11-11 — Option 1 landed: on **RDNA3_5 UMA** the transport ladder STARTS on the buffered arm, and for a >RAM model it is +56-67 %

Maintainer decision (2026-11-11): *"Option 1, but gate it specific to RDNA3.5 because it's more of a UMA specific
optimization"*, and *"always remember that the cold condition only happens once at the start; better to tune for
warm"*.

### 24.1 What is implemented, and the three gates

`pool_transport_init_locked()` no longer always applies r262's optimistic `map` start.  One deviation, deliberately
narrow:

| gate | why |
|---|---|
| `g_gtt_arena` (the GTT arena tier) | the **discrete host pool keeps r262's `map` start exactly** -- nothing about a discrete box changes |
| `GGML_CUDA_CC_IS_RDNA3_5(cc)` on the table's device | the maintainer's call: this is a UMA-shaped optimisation and gfx1151 is the APU this campaign measures.  A future APU on another family keeps the delivered start until it is measured |
| `direct_fill_wanted() && table_direct_readable(t)` | the win being chased **is** the direct fill (P4b).  With it off the two arms are level on the bounce path, so a `file` start would be a fork of r262's decision with no evidence behind it |

Applied, it logs its own reason: `host read transport -> file (RDNA3_5 UMA start: the buffered arm fills straight
into the arena slot (no bounce, no H2D), which the probe cannot see and the one-way ladder would otherwise never
reach)`.  An explicit arm still wins over the deviation -- `MOE_HOST_POOL_TRANSPORT=map|file|dio` and
`--load-mode mmap|buffered|dio` are pins by definition (verified below).

### 24.2 Measured (35B-A3B, arena `-lme 4`, `-n 96`, prior FROZEN, alternating 3+3, and MBIG in both orders)

**A fitting model, warm page cache — the steady state (the maintainer's "tune for warm"):**

| arm | reps | median |
|---|---|---|
| **default (`auto` -> RDNA3_5 UMA start -> `file` + direct)** | 29.0 / 28.8 / 29.0 t/s | **29.0** |
| pinned `map` (r262's start) | 25.5 / 25.5 / 25.4 t/s | 25.5 |

**+13.7 %**, and now deterministic: before the deviation the same configuration latched `file` **0 of 5** times
(§23.3).

**A fitting model, cold page cache — paid once per model load, by decision:**

| arm | reps | median |
|---|---|---|
| default (`file` + direct) | 21.5 / 21.6 / 21.6 t/s | 21.6 |
| pinned `map` | 24.4 / 24.4 / 24.6 t/s | 24.4 |

**-11.5 %**, and because the ladder is one-way (an r262 decision: "a box that fell short once is likely to again")
there is no path back to `map` -- this is a commitment, not an oscillation.  Accepted on the maintainer's steer:
the cold state exists only at the start of a session, the warm state is every token after it, and the deviation
also *warms* the set faster (a buffered read populates the page cache the mapping then reads).

**The tier's actual workload -- a >RAM model, where the set can never be cached (MBIG `-lme 60`, `-n 16`):**

| order | `file` + direct | pinned `map` |
|---|---|---|
| default first, then `map` | **7.5 t/s** (direct pread 4 221 MiB/s) | 4.5 t/s (24 442 ms of mapping reads) |
| `map` first, then default | **7.8 t/s** | 5.0 t/s |

**+56 % / +67 %.**  With the page cache unable to hold the set, the mapping pays a fault path per read where a
`pread` of the same bytes does not, so the deviation is not a warm-only trick: on the model this tier exists for it
is the largest single win the campaign has measured.  Text identical in both orders (`ee95c45231c3`, no-`-lv` runs);
the fitting-model arms are all `627b2ebd7792`.

### 24.3 The gates, verified on the box

| arm | result |
|---|---|
| `MOE_EXPERT_GTT_FILL_DIRECT=0` | starts on **`map`** -- the deviation requires the direct fill (and the ladder may still escalate to `file`, which then uses the bounce path: 23.9 t/s) |
| `MOE_HOST_POOL_TRANSPORT=map` | `map` (25.5 t/s) -- an explicit pin wins |
| `MOE_HOST_POOL_TRANSPORT=dio` | `dio` (7.1 t/s -- `O_DIRECT` is 11x slower here, as §20.2 measured) |
| `--load-mode mmap` | `map` (26.2 t/s) -- `--load-mode` is a pin too |
| placement mode (`-lme 6`, no arena tier) | no transport applied at all -- inert |
| text, all of the above | `627b2ebd7792` |

`check-gpu-faults.sh` clean.

### 24.4 Not measured

* **A non-RDNA3_5 device**: none on this box.  That is precisely what the arch gate is for -- the deviation cannot
  reach a family nobody measured it on.
* **The discrete host pool**: `g_gtt_arena` is false there, so the code path is unreachable and r262's start is
  untouched; `scripts/gate-host-pool.sh` (which cannot run here -- no discrete GPU) is the gate.
* **A truly cold >RAM model**: MBIG's runs are inherently disk-bound, which is why they are the representative arm,
  but the cache was not explicitly dropped before them.
* **The cold-for-fitting trade is a decision, not a measurement**: the maintainer's steer is recorded above so a
  later reader sees *why* the cold arm loses.

## 25. 2026-11-11 — P5: PREFILL STREAMING. The fill only what is not resident, coalesced and parallel: **+63-67 % on the >RAM model**, neutral on a fitting one, and the logits do not move

The plan's P5: *per layer, subtract the resident set and stream the difference coalesced*, never re-fetching a
bank the previous chunk holds.  What was actually wrong, measured first (`-lv 4`, 35B-A3B, `-lme 4`, a 533-token
prompt): the prefill does **not** go through `moe_cache_stage_table` at all on this tier -- the scheduler's
used-expert pruning issues one `set_tensor_async` per run of consecutive used experts, and on a pageable master
that is `moe_cache_upload_pageable`: a memcpy out of the model mapping into the per-table pinned bounce in
`host_bytes` (0.6 MiB) chunks, then one pinned H2D per chunk.

    scheduler pageable-master fallback: 3984 ranges, 11608.1 MiB served through the per-table bounce
    [ Prompt: 356.8 t/s ]   (the delivery's prefill on this tier)

### 25.1 What P5 is (and what it is not)

A new backend hook, `moe_cache_prefill_stream`, called where the scheduler's own copy would run, one call per
(table, ubatch), **prefill band only**.  It fills the split-input copy itself and returns true, so the caller
skips its copy:

| the used expert is... | what happens |
|---|---|
| **resident in the GTT store** | one **D2D** out of its slot (232 GB/s, no storage read) -- this is the residency subtraction |
| not resident | one **coalesced span read** per maximal run of missing used experts, `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` (16) experts max, on the tier's own buffered transport, into a pinned span ring -- and **one H2D per span**, because a run is contiguous in the FILE and in the DESTINATION |
| read came up short | the delivery's per-expert source serves it (`fallback`) |

* the used set is the scheduler's **own bitset**, not the routing buffer: the ids buffer the scheduler read is
  reused by the graph allocator, and re-reading it in the backend served a STALE routing (see §25.5);
* the reads run **in parallel** (`MOE_EXPERT_GTT_PREFILL_READERS`, default 8) and overlap the previous batch's
  H2Ds;
* the MMQ finite-head guard is laid down first, one zero per used run's successor slot;
* a failed or partial fill returns **false** and the caller's copy overwrites the destination, so the path can
  only ever be slower, never wrong.

### 25.2 The fitting model (35B-A3B Q4_K_M, arena `-lme 4`, prior FROZEN, alternating 3+3)

| arm | pp | reads |
|---|---|---|
| `MOE_EXPERT_GTT_PREFILL=0` (the delivery's copy) | 352.1 / 352.9 t/s | 11608.1 MiB (every used expert) |
| **P5 (default)** | **359.7 t/s** | **9451.6 MiB of 12809.2 used (73.8 %)**, in 4605 spans of 2.05 MiB; 3357.6 MiB served from the arena by D2D; 0 fallback |

**+2 %** -- inside the box's spread.  26.2 % of the used bytes are never read, but on a *fitting* model the reads
come out of the page cache and the copy-issue cost is what it is, so the byte saving does not convert into
throughput.  That is the honest result and it is why the tier's real workload is measured separately.

### 25.3 The tier's ACTUAL workload (>RAM model, MBIG 122B-A10B Q8_0, `-lme 60`, alternating 2+2)

| arm | pp (533-token prompt) |
|---|---|
| `MOE_EXPERT_GTT_PREFILL=0` | **9.8 / 10.6 t/s** |
| **P5 (default)** | **16.5 / 17.6 t/s** |

**+63 % / +66 %.**  Census: `used 89418.9 MiB`, `read 36949.5 MiB (41.3 %)` in 8673 spans of 4.26 MiB,
`arena D2D 52469.4 MiB (58.7 %)` -- the store's residents are more than half of what the delivery re-read on every
ubatch, and the reads that are left are parallel, which is what a disk-bound arm wants.  Text identical in all
four runs (`9b259759672f`, no `-lv`).

### 25.4 Correctness

* **Text**: identical in every arm, both models: `347f9f6f715f` (35B, 522 chars) and `9b259759672f` (MBIG).  The
  arms include `PREFILL=0` (the exact pre-P5 behaviour), `MOE_EXPERT_GTT=0` (no store at all), the prefill
  fault arm, the cold-cache arm, and the whole gate.
* **Prefill logits**: `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_GTT_PREFILL=0"` (40 chunks, ctx 512,
  `--extra "-lme 4 --fit off"`): **mean KLD -0.000000** (limit 0.005), **same-top-p 99.990 %** (floor 98 %), GPU
  journal clean.  Base PPL 5.8238 -- the same number for the fallback arm and for P5.
* **The file read equals the model mapping**: `MOE_EXPERT_GTT_PREFILL_VERIFY=1` byte-compares every coalesced
  span against the pageable mapping (a shard-relative offset in a multi-part GGUF, or an `O_DIRECT` alignment
  slip, shows up there and nowhere else).  No mismatch on either model.
* **A failed coalesced read is a slowdown, not an answer**: `MOE_EXPERT_GTT_PREFILL_FAULT=1` fails every span
  read; all 117 prefill fills then decline and the per-expert source serves them.  Same text, no GPU fault.

### 25.5 Two traps found the hard way (both are now in the gate)

1. **The split-input copy is ONE buffer per copy id for every table, not one per table.**  The graph allocator
   gives a prefill graph one `input_cpy` for tables whose lifetimes do not overlap, so a table's destination in
   ubatch 2 holds the previous ubatch's LAST table -- measured: 113 owner changes over 117 calls, i.e. nothing
   survives a pass.  **The plan's "do not re-fetch a bank the previous chunk already holds" therefore cannot be
   served by the destination**; the residency that survives is the store's, which is what P5 subtracts.  The
   design was built, measured, and dropped.
2. **`span_buf_acquire` only refuses a buffer whose copies have drained** -- so a ring slot must be marked busy AT
   ACQUISITION, not when its H2D is issued.  With the mark in the wrong place every window in a call got the SAME
   pinned buffer: the last read won, every H2D copied it, and the greedy text came out 509 chars instead of 522.
   The file read verified clean and the *used set* looked degenerate (8 experts per table) -- because the model
   had already degenerated, so its later routings really were "experts 0..7".  One root cause, three symptoms; it
   cost the session an afternoon and it is why the gate asserts `verify` + text + the read/used ratio together.

The second one also retro-explains the "stale routing" red herring: a bad destination corrupts the hidden states,
and the routing that follows is genuinely degenerate.  The hook still takes the scheduler's bitset rather than the
ids buffer -- it is cheaper, it cannot disagree with the copy it replaces, and it is the same set under every arm.

### 25.6 Not measured

* **A cold page cache on a fitting model**: one cold P5 sample read 88 t/s against ~360 warm, but the baseline
  was not measured cold in the same series, so no ratio is claimed (P4f already owns the cold trade: `file`
  starts the ladder and costs -11.5 % cold, +13.7 % warm).
* **Reader width**: 1/4/8 readers measured 9.7/15.2/18.2 GiB/s of *in-run* read rate (much flatter than a bare
  `pread`, which is 25 GiB/s at one thread and 60 at eight, because the reads contend with the GPU prefill for
  DRAM).  8 is the default and 1 is the A/B.
* **Multi-device**: untested, as everywhere in this campaign.
* **A `-sm tensor` slice or a fused table**: the hook declines by geometry (no single run per expert) and the
  delivery's copy serves it -- by construction, not by measurement.

## 26. 2026-11-11 — the two promotion decisions: the SHAPE is chosen by the fit, and admission needs nothing (the campagne's own policy simulator says so)

Maintainer, 2026-11-11: *"MOE_EXPERT_GTT_ARENA's default shape should be arena if the weights would spill onto
disk, otherwise placement.  If `MOE_EXPERT_GTT_ARENA` is set, then honour the user's choice"* and *"we can drop
the `MOE_EXPERT_CACHE_TOUCH=N` for now … don't we have a policy simulator already written somewhere for exactly
this?"*

### 26.1 The shape default: `-lme` is a CAP, so the shape follows the budget (and the first implementation was wrong)

**What the flag promises.**  `-lme`'s own help text: *"cap the NON-SWAPPABLE memory spent on MoE experts"*.  On an
APU that promise decides the shape, and the campaign's first attempt keyed it on the wrong thing -- the *model's
fit in RAM* rather than the *budget*.  The maintainer caught it with a one-line counter-example: `-lme 4` against a
21 GiB model is a 4 GiB cap, so the excess has to go somewhere, and "somewhere" is the arena.

**The measurement that settles it** (35B-A3B, `-lme 4`, one 1 s `MemAvailable` monitor per run, same prompt):

| shape | lowest `MemAvailable` during the run | non-reclaimable expert memory | pp | decode |
|---|---|---|---|---|
| placement | **92 489 MiB** (126 482 MiB total) | 18.66 GiB, **anonymous / device** | 1453 / 1435 t/s | 62.5 / 62.6 t/s |
| arena | **106 064 MiB** | ~4 GiB in the store, the rest **file-backed page cache (reclaimable)** | 382.6 / 375.1 t/s | 31.6 / 31.6 t/s |

So the arena really does hold **13.6 GiB less non-reclaimable memory** -- `MemAvailable` counts reclaimable page
cache as available, and that is exactly what "non-swappable" means.  The first implementation's claim that "it is
the same RAM either way" was wrong in the way that matters: the same DRAM is *touched*, but only one shape makes
the excess reclaimable.  The price of the cap is real and is the churn: placement does no fills at all, the arena
misses 25.5 % (`hits=73548 fills=25212` over 96 tokens).

**The implemented rule** (`plan_gtt_expert_store()`), which is also what the flag documents:

    the store CAN hold the whole expert set   -> PLACEMENT   (nothing to stream, nothing evicted, fastest)
    it CANNOT (a `-lme N` below the set)      -> ARENA       (the store is the cache's slot storage, the expert
                                                             tensors are holes in the read-only model mapping, LFRU
                                                             residency serves every read -- the excess is
                                                             reclaimable page cache, i.e. the cap is honoured)
    `-lme auto` (the default) asks for NO cap: on a fitting model it sizes to the whole set, so it lands on
    placement; on a >RAM model MemAvailable's ceiling is below the set, so it lands on the arena.

`MOE_EXPERT_GTT_ARENA` is the override: `=1` forces the arena (where there is something to stream), `=0` forces
placement.  It is no longer read as `getenv(...) != nullptr`, which meant `MOE_EXPERT_GTT_ARENA=0` *enabled* the
tier -- a switch that cannot spell "off" is not a switch.

Verified on the box, with the reason in the census:

| model | `-lme` | env | census |
|---|---|---|---|
| 35B-A3B (18.2 GiB set) | 4 | unset | `[auto: -lme asks for a 4096.0 MiB cap on non-swappable expert memory against a 18662.5 MiB set, so the excess streams from the file-backed page cache (reclaimable) instead of becoming anonymous memory] -> the GTT store is the MoE ARENA` |
| 35B-A3B | auto (95 584.6 MiB cap) | unset | `a registered GTT expert store of 18662.5 MiB of a 18662.0 MiB expert set: 120 tables / 30720 slots (18662.5 MiB) resident … the WHOLE set` -- **placement** |
| 35B-A3B | 4 | `=0` | placement |
| 122B-A10B (117.5 GiB set) | 60 | unset | `[auto: -lme asks for a 61440.0 MiB cap …] -> the MoE ARENA` |

The gate asserts both directions: `shapefit` (a cap below the set and the env UNSET must take the arena and name
the cap) and `resident` (a budget that covers the whole set must land on placement).  `ram` runs with the env UNSET.

### 26.1a `-lme` is the opt-in for STREAMING (maintainer, 2026-11-11)

*"if `-lme [N|auto]` is set, then `gtt_store_can_stream` should be true, as this then allows for a memory > RAM to
be served via the arena.  If `-lme` is not set, then it should decline (i.e. the current non-lme behaviour).  The
whole idea is that `-lme` turns on the arena."*

Implemented as three linked facts, because `-lme` is what distinguishes "the operator asked for an expert residency
budget" from "nobody did" (`host_experts_limit_gib`: -1 = never given, 0 = `auto`, >0 = N GiB):

1. **`gtt_store_can_stream = -lme was given.**  That is the flag the LOCK policy reads, so a >RAM model under a
   `RLIMIT_MEMLOCK` too small for the request but big enough for the RAM shortfall is now CLAMPED AND RUN instead of
   refused -- which is the whole point of the tier.  Measured: 122B-A10B, `-lme 60`, `RLIMIT_MEMLOCK=20 GiB` ->
   `20480.0 MiB of slots for a 117504.6 MiB expert set … the MoE ARENA`, 144 tables resident, 4032 evictions,
   runs.  (The clamp must still fund one slot per expert table -- `honour >= arena_min` -- or the arena could not
   serve a single table and placement would over-commit.)
2. **The ARENA also requires `-lme`**, not just a budget that resolves below the set.  Without the flag the store
   is still sized from MemAvailable and PLACEMENT may engage (a fitting model keeps its hot subset registered --
   neutral-to-positive, and it is the delivery's default-on behaviour), but nothing is streamed.
3. **A model whose weights do not fit in RAM is REFUSED when streaming is unavailable**, instead of being placed
   and left to swap (the 2026-10-19 precedent: 121 GiB resident + 13 GiB swap on a 123 GiB box).  Measured:
   122B-A10B with no `-lme` fails in **0.2 s, before a byte of weights is read**, with the reason and the fix:

       GTT expert store: this model's weights (126403.0 MiB) do not fit in RAM (118113.2 MiB available,
       106301.9 MiB is the ceiling they may occupy), so the part that does not fit has to be streamed --
       and streaming is not available here:
         `-lme` was not given, and the GTT arena -- the tier that streams the part of the expert set that
         does not fit -- is opted in BY `-lme`
         asked for           auto: 80% of MemAvailable -> a 95321.3 MiB cap
       Pass `-lme N` (N GiB of non-swappable expert memory) or `-lme auto` to opt into the GTT arena …

   Three other paths reach the same refusal, each with its own sentence: a GTT store already live in the process
   (a second model), a budget below one slot per table, and `MOE_EXPERT_GTT_ARENA=0` (an explicit request for
   placement).  A *fitting* model never refuses: its weights are under the ceiling, so placement is fine.

The gate asserts all three: `shapefit` (arena), `resident` (placement), `ram` (arena with the env unset) and
`nolme` (the refusal, its reason, and its fix).  **40 checks, green.**

**Recorded as an open item, NOT changed in passing:** `plan_gtt_expert_store()` still hard-codes
`gtt_store_can_stream = false` in its lock-policy branch, whose comment says "P3/P4 flip this to true".  Relaxing it
would let a >RAM model run on a CLAMPED (smaller) locked store instead of being refused -- but it needs its own
arms, because the dangerous combination is a clamp that covers the RAM shortfall yet is still below one slot per
table: the arena would decline and placement would over-commit.  That is a separate change with its own failure
mode.

### 26.2 Admission: the campaign's own simulator predicts that NOTHING should change

`archive/work/moe-expert-cache/policy_sim.py` (the `moe-expert-cache` campaign's admission/replacement simulator)
replays a token-indexed routing trace under each candidate policy and reports the steady-state hit rate per S.
It was re-used here, unchanged, with a fresh trace from THIS tier: `MOE_EXPERT_TRACE=/tmp/moe35b.trace`, 35B-A3B,
arena `-lme 4`, a 407-token greedy decode over 120 tables, converted to the simulator's
`route,tok,table,expert` form (one "table" = one (layer, role), i.e. the arena's own unit; the converter is 30
lines and is recorded in the WORKLOG entry).

**h_steady (last 200 tokens), the arena's real slots/table in brackets:**

| policy | S=56 (`-lme 4`) | S=133 (MBIG `-lme 60`) | S=192 |
|---|---|---|---|
| `static` top-S (no adaptation) | 0.398 | 0.510 | 0.512 |
| **`lru` / the tier's LFRU** | **0.727** | **0.918** | **0.938** |
| `second_touch` (the delivered `TOUCH=2`) | **0.708** | **0.880** | **0.891** |
| `second_touch_fill` | 0.719 | 0.917 | 0.938 |
| `lfru_decay:64` | 0.738 | 0.919 | 0.938 |
| `slru:0.5` | 0.745 | 0.918 | 0.938 |
| `lru_window:32` (a recency admission filter) | 0.541 | -- | -- |

Three things fall out, and they settle the question:

1. **The Nth-touch admission the delivery ships is predicted WORSE here**, at every slot count (-1.9 pts at S=56,
   -3.8 at S=133).  The reason is structural: on this tier a non-admitted miss cannot be *served* cold (there is
   no in-place GPU-readable source), so it still costs a full read AND it declines the slot -- the churn it saves
   is worth less than the hit rate it gives up.  That is the maintainer's intuition, now with numbers:
   **we would rather fill it.**
2. **The tier already implements the predicted-best policy.**  `access_locked()`'s eviction is min-count with an
   LRU tie-break and a decay period -- `LfruDecay` -- which is the best or tied-best row at every S (0.738 /
   0.919 / 0.938 against plain LRU's 0.727 / 0.918 / 0.938).
3. **SLRU is the only policy that beats it**, by +1.8 pts at S=56 and ~0 at S>=128: it trades a protected/probation
   split, which is a new mechanism for a gain that disappears in the range the tier actually runs at.  Not worth
   building on this evidence.

**So the transient band stays dropped, and `MOE_EXPERT_CACHE_TOUCH` stays a no-op** -- now on a prediction as well
as on the gate's `touch1/2/3` identical-residency assertion.  If a future workload makes admission worth
revisiting, the simulator is the first stop, and the trace capture is one env var.

**Calibration check:** the simulator predicts a 0.727 steady-state hit rate at S=56; the box's own whole-run
census at `-lme 4` reads `h=0.60-0.63` over a short run, which includes the cold start.  Consistent in direction
and magnitude -- the simulator is not obviously mis-modelling this tier.

### 26.3 Not measured

* **MBIG's own decode trace**: the trace above is the 35B's, replayed at MBIG's slot count (S=133).  A trace from
  the 122B would be a better bound on its own workload; the *ranking* of the policies is what the question asked
  for, and it is unchanged across S.
* **SLRU / decay-period tuning**: the +1.8 pt is one trace, three prompts, one box.

## 27. 2026-11-11 — CROSS-MACHINE: the campaign on a discrete RDNA4 box (`soar`, 3x R9700 gfx1201). Inert by measurement, host-expert pool green, throughput unchanged

The promotion criteria demand "no regression on the discrete boxes (the GTT store is unreachable there; the
pool/arena path must be unchanged bit for bit)".  Until now that was a *claim* from `g_gtt_arena` / `prop.integrated`
gating; this is the measurement.

**Box** (`ssh soar`): 3x **R9700, gfx1201 (RDNA4)** + a gfx1036 iGPU, Ryzen 9 9950X3D2, **91 GiB RAM**, kernel
**7.3.3**, ROCm **10.1.0** (`/opt/rocm-10.1.0-gfx120X`).  `~/llama.cpp` is a clean checkout of the fork at the r262
tip `d65b0e889` and was **not touched**: the patched build lives in a throwaway `git clone --shared` at
`~/apu-gtt-check` (patch applied with `git am`, tree `9e972b50…` -- the same tree as the campaign's own), built with
`~/bin/build-llama-rocm` into its own `build-rocm`.  The unpatched reference is `~/llama.cpp/build-rocm`.

### 27.1 The GTT tier is INERT on a discrete GPU

The campaign's envs forced on, on the discrete box, on the host-expert pool config (`-ncmoe 47 --host-experts pool`,
Flash-Next GSQ-IQ3_XXS, `MOE_HOST_LIMIT_FRAC=40`, prior frozen):

| forced env | text | `GTT` lines in the log | host pool |
|---|---|---|---|
| `MOE_EXPERT_GTT_ARENA=1` | `a99ccbd97ecb` (182 chars) | **0** | 141 tensors, 56 964 slots |
| `+ MOE_EXPERT_GTT_PREFILL_READERS=8` | `a99ccbd97ecb` | **0** | 141 tensors, 56 964 slots |

Zero `GTT expert store` / `GTT arena tier` / `GTT prefill stream` / `plan_gtt_expert_store` lines: `plan_gtt_expert_store()`
is only reached from the UMA branch of `plan_host_expert_placement()`, `gtt_store_open()` is never called, and
`moe_cache_prefill_stream()` declines on `!g_gtt_arena` before it looks at anything.  The env vars are inert, not
merely harmless.

### 27.2 `-lme` on a discrete box still means what it always meant

`-lme 4` on the same config is the **host-expert budget**, and it behaves exactly as r262 left it:

    host experts: pinned 0.0 MiB of 40012.5 MiB (cap 4096.0 MiB), pageable 40012.5 MiB, 141/141 pool sources
    registered; -lme 4 -> a pinned per-EXPERT poo[l] ...
    host pool (L2): 141 tensors, 7332 slots

Text identical (`a99ccbd97ecb`), zero GTT lines.  The APU reading of `-lme` (the GTT store's size) is confined to the
UMA branch, so the one command line still carries across in the direction it was designed to.

### 27.3 `scripts/gate-host-pool.sh` -- 42/42 with the patched build

The discrete host-expert pool gate, run against `~/apu-gtt-check/build-rocm/bin/llama-cli` (nine arms: `ref`, `dio`,
`nodisk`, `fail`, `late`, `sink`, `sinkfail`, `pinlate`, `pin0`):

* **`42/42 checks passed`** -- every arm's greedy text is byte-identical to `ref` (`a99ccbd97ecb`, 182 chars), which
  is also the hash recorded in that gate's own header;
* `sink`: 5128 demoted; `sinkfail`: 5128 demote copies failed and **nothing published** (0 demoted);
* `fail` / `late`: **takeover=0** (a failed fill declines its op), every failed fill reports the slot as NOT resident,
  the cause-hiding `async fill failed` message is absent;
* `pinlate`: the latch stopped the tier GROWING, not its use (5434 fills served from a pool built before the
  refusal) -- that arm's text is `b981e742aa13` (171 chars), the documented issue-#282 mid-run-mixture condition;
* the exit census prints once, from the model teardown, in every arm; no arm aborts; **kernel journal and every arm
  log clean** (`scripts/check-gpu-faults.sh`).

### 27.4 Text identity patched vs unpatched, same config

| build | text (pool arm, `-n 32`, `--temp 0 --seed 42`) | pool |
|---|---|---|
| `~/llama.cpp/build-rocm` (unpatched, r262 tip) | **`a99ccbd97ecb`**, 182 chars | 141 tensors, 56 541 slots |
| `~/apu-gtt-check/build-rocm` (patched) | **`a99ccbd97ecb`**, 182 chars | 141 tensors, 56 823 slots |

(The pool's slot count is sized from free memory at build time, so it varies run to run; the gate asserts `> 0`.)

### 27.5 Throughput, patched vs unpatched, alternating (`-ncmoe 20 --host-experts pool`, 35B-A3B Q8_0, `-r 2`)

| arm | pp512 | tg128 |
|---|---|---|
| unpatched | 1745.13 ± 7.19 / 1754.79 ± 10.79 t/s | 62.88 ± 5.69 / 60.71 ± 6.44 t/s |
| **patched** | 1750.40 ± 3.12 / 1774.16 ± 9.51 t/s | 62.76 ± 5.45 / 61.27 ± 5.20 t/s |

**Within noise on both axes** (+0.3 / +1.1 % prefill, -0.2 / +0.9 % decode, with overlapping spreads).  No throughput
claim is made beyond that: the point is that the discrete host-expert path is not disturbed.

### 27.6 Not measured

* **`gate-uma-experts.sh` on soar**: it is the UMA gate and has nothing to test there (its `off` arm would pass
  trivially); the discrete side's gate is `gate-host-pool.sh`, above.
* **Two or three GPUs at once**: this box has three R9700s, but the pool path here is single-device (`GPU=0`), which
  is what the gate defines.  A multi-device pool/tensor-split regression check is still owed (it is owed on the APU
  side too -- see §25.6/§26.3).
* **A long-running arm**: the gate's arms are 32-token greedy runs; a throughput/acceptance comparison over the
  MTP four-axis protocol is still the campaign's outstanding debt on any box.

## 28. 2026-11-11 — the MTP four-axis gate for block 16 (the release gate that was owed): acceptance is BIT-IDENTICAL in every arm

`benchmarks/mtp-adaptive-methodology.md` Protocol A, on `halo` (gfx1151), Qwen3.6-35B-A3B Q4_K_M (its embedded
nextn head, no `-md`), `-n 3000` (2000 floor), `--reasoning on` for R and `off` for P/C/K, seed 42, temp 0, bf16 KV,
`-c 16384`, `-t 8`, the versioned prompts (`reasoning.txt`, `prose-rdna-boosts.txt`, `code-python.txt`,
`recall.txt`), prior frozen.  Three arms per axis:

* **OFF** = `MOE_EXPERT_GTT=0` -- the delivery's previous behaviour, the reference;
* **ARENA** = `MOE_EXPERT_GTT_ARENA=1 -lme 4` -- the store as the cache's arena, ~25 % miss churn: the stressing arm;
* **PLACEMENT** = `-lme auto` with `RLIMIT_MEMLOCK=unlimited` -- the **fitting-model default a consumer gets** (the
  store holds the whole set, nothing is evicted), measured on P and K;
* plus a **plain** (`--spec-type none`, `-n 1000`) arm per axis on the ARENA configuration, to show MTP still
  accelerates.

| axis | OFF (t/s) | ARENA (t/s) | PLACEMENT (t/s) | plain, ARENA (t/s) | **acceptance (all arms)** | mean len |
|---|---|---|---|---|---|---|
| R reasoning | 91.72 | 38.49 | -- | 30.44 | **0.72082** (2050 / 2844) | 3.16 |
| P prose | 92.85 | 39.06 | 93.13 | 28.22 | **0.75659** (1321 / 1746) | 3.27 |
| C code | 104.69 | 42.18 | -- | 28.47 | **0.88057** (2175 / 2470) | 3.64 |
| K recall | 112.54 | 46.23 | 112.03 | 29.18 | **0.99462** (370 / 372) | 3.98 |

**The acceptance is bit-identical on every axis -- not "within noise": the accepted/generated counts are the same
integers** (2050/2844, 1321/1746, 2175/2470, 370/372) and so is the mean accepted length.  That is the strongest
possible statement for this block: the store changes WHERE the bytes are, never WHICH, so the draft-vs-verify
numerics relationship is untouched.  It is also on the right side of the protocol's threshold (acceptance > ~0.45
everywhere) and **MTP still accelerates in every arm** -- +27 % to +58 % over plain on the arena arm, and the
placement arm is level with the store OFF (P 93.13 against 92.85, K 112.03 against 112.54).

The ARENA arm's throughput (38-46 t/s against 91-112) is the **price of a 4 GiB cap on an 18.7 GiB set**, not a
block-16 regression: the same configuration's *plain* decode is 28-30 t/s, i.e. the tier is still a large win over
no tier at all, and the default (`-lme auto`, placement) is level with the previous behaviour.

**Two things this gate run also showed, both already documented and both real:**

* `llama-batched-bench -npl 1,4,8` produces **no data rows on this box in either arm** (rc=0, header only) -- on the
  block-16 build *and* on the other build in the checkout, so it is the tool/host, not the block.  The width arm is
  therefore **inconclusive here** and no comparison is claimed; `llama-batch`/`llama-bench` arms were used instead
  (`MEASUREMENTS` §25.2/§25.3).  The 3x R9700 box is where that arm has always been run.
* **`-lme auto` under an 8 GiB `RLIMIT_MEMLOCK` is an ARENA, not placement**: the clamp branch takes the limit when
  the model fits ("this model fits in RAM anyway, so this is an optimisation"), the extent lands at 8 GiB, and
  `extent_trunk < padded_total` therefore engages the arena -- loudly, with the clamp WARN.  The placement arm above
  needs `LOCK_KB=unlimited`, and the gate's `resident`/`shapefit` arms already assert both directions.
