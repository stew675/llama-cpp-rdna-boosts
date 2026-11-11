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
