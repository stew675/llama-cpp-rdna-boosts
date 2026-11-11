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

## 14. 2026-11-11 — the store EXISTS (P1.1 + P2): the keystone re-run, and what the registered region costs and reads at real scale

The design's memory was answered in §13.  This section is the same box measuring the **implemented** store
(`p2-gtt-expert-store.patch`, `README.md` §2.1): the keystone re-run first (because a design premise that no
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

### 14.2 Registration at the store's real extent — cheaper than P1.0 predicted

The store registers **one** untouched `mmap`+`MADV_HUGEPAGE` region (`cudaHostRegister(Mapped|ReadOnly)`), which
is the pre-faulting-free order: registering faults the pages in and the kernel forms the huge pages (§13.1, second
row).  Measured inside the store creation, from the census (extent, mode: 2 runs each on 35B-A3B):

| extent | registration mode | mmap + MADV_HUGEPAGE + register | per GiB |
|---|---|---|---|
| **19 126.5 MiB** (18.7 GiB) | ReadOnly | **482 / 485 ms** | **25.8 / 26.0** |
| 6 144.0 MiB (6 GiB) | ReadOnly | 160 / 161 ms | 26.7 / 26.9 |
| 19 126.5 MiB | WRITABLE (`MOE_EXPERT_GTT_RO=0`) | 483 / 486 ms | 25.9 / 26.0 |
| **40 925.6 MiB** (39.9 GiB) | ReadOnly | **1 034 ms** | **25.9** |

* **~26 ms/GiB**, against P1.0's **30.5 ms/GiB** for an untouched region (§13.1) and **17.4 ms/GiB** for a
  pre-faulted one.  So the `memset` pre-fault walk the probe needed is not worth taking at the store's scale:
  registering the untouched region costs about what pre-faulting alone did.
* Registration mode makes **no difference to the cost** (25.8 vs 25.9 ms/GiB) — only to the read rate (§14.5).
* The **whole** store's registration is **0.48 s on a 19 GiB tier and 1.03 s on the 41 GiB Flash-Next tier** -- and
  the per-GiB figure does not drift with size (25.9 ms/GiB at 6, 19 and 41 GiB), so it is a linear, predictable
  startup cost.
* **All 19 126 MiB of it lands as huge pages**: `AnonHugePages` 19 126.0 MiB of 19 126.5 (100.0 %), and 6 144.0 of
  6 144.5 on the `-lme 6` arm.  `AnonHugePages` is read from `/proc/self/smaps` **after the load**, i.e. after the
  loader wrote every byte — so the fill does not split them.

### 14.3 The load-time fill (P2's, not P4's)

The fill is the buffered `pread` arm straight into the slot (`read_raw` into `cur->data`, `llama-model-loader.cpp`
→ `load_data_for`): no alignment round-out, no bounce buffer, no staging ring.  The measured window is
`carve → census` (the other tensors of the load are `mmap`-mapped, so they are nearly free in that window):

| model | store read from the GGUF | fill window | rate | page cache |
|---|---|---|---|---|
| 35B-A3B Q4_K_M | **18 662.0 MiB** | 1.34 s | **13.9 GB/s** | warm (the file had just been read by earlier arms) |
| Flash-Next GSQ-IQ3_XXS | **40 925.0 MiB** | 17.6 s | **2.33 GB/s** | cold-ish (46 GiB file on a box holding 37 GiB of cache) |

**The fill is page-cache/disk-bound, and that is a LOAD cost, not a decode cost.**  A 41 GiB store takes ~18 s to
populate when its file is not resident, ~3 s when it is; it is the same 2.3-15.7 GB/s band §13.2 measured for
`pread` into a registered region, and P4's pipelining does not touch it (P4 pipelines *miss* fills, this is the
one-shot load fill).  The bytes read account for every expert byte of the carve (the carve is 0.2 MiB larger on Flash-Next: the
buffer type's per-tensor page padding), and `size_done`'s progress is unaffected.

### 14.4 `-lme` is the store's size (P1.1) — realised, on both models

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
  is the cross-check that the store holds the same set the heat map ranks.
* 35B-A3B's 123 tables / 19 126 MiB are 3 x 41 layers: the **MTP (nextn) head's** three expert tables are included
  here, where the delivery's cache keys only the 40 trunk layers (§13.5's 120 tables / 18.22 GiB).

### 14.5 The read rate at model level, and what `MOE_EXPERT_GTT_RO` buys

`llama-bench -m 35B-A3B -ngl 99 -p 512 -n 128 -r 3`, two independent runs of the whole set of arms:

| arm | pp512 (run 1) | pp512 (run 2) | tg128 |
|---|---|---|---|
| `MOE_EXPERT_GTT=0` (plain device buffer, today) | 2 268.65 ± 26.67 | **2 289.39 ± 4.78** | 63.22-63.25 |
| store, registered **ReadOnly** (default) | 2 259.63 ± 15.37 | 2 260.77 ± 8.68 | 63.24 |
| store, registered **WRITABLE** (`RO=0`) | 2 135.82 ± 1.34 | 2 146.85 ± 12.13 | 63.30-63.34 |

* **The registered ReadOnly store is within noise of the plain device buffer**: -0.40 % in run 1, -1.25 % in run 2,
  against a drifts between runs of +0.9 % on the plain arm itself.  The store is **not a slow tier**, which is the
  acceptance criterion P2 set.
* **`MOE_EXPERT_GTT_RO=0` is materially slower**: -5.9 % and -6.2 % of the plain arm, -5.5 % and -5.0 % of the
  ReadOnly store, at a spread of ±1.34 / ±12.13 — i.e. well outside the noise.  The 232-vs-60 GB/s fact of probes
  03/04 **does apply to this region**.
* The magnitude is *masked by compute*: a 3.9x slower read costs 5-6 % of prefill, not 74 %, because the expert
  read overlaps arithmetic and only a fraction of prefill is weight-bandwidth-bound.  Do not read the 60 GB/s
  probe figure as "writable costs 74 % at the model".
* **Decode does not move at all** (63.22-63.34 t/s across all six runs): a decode step reads top-k experts, so it
  never approaches the read bandwidth.  A store's value is in what it holds, not in how fast it reads while
  fitting.

### 14.6 Identity, and the kill switch

Same-seed greedy text, **no `-lv`** (`scripts/extract-generated.py`), prompt `The capital of France is`, `-n 20`:

| model | `MOE_EXPERT_GTT=0` | store `auto` | store `-lme 6` | store `RO=0` | pre-P2 build (`release.json` tree) |
|---|---|---|---|---|---|
| 35B-A3B Q4_K_M | `89 chars sha=627b2ebd7792` | **identical** | **identical** | **identical** | **identical** |
| Flash-Next GSQ-IQ3_XXS | `113 chars sha=f586785d7183` | **identical** | — | — | **identical** |

`-ncmoe 8` on the same box with the store on: identical too (the `-ncmoe` refusal is unchanged, and the store
simply holds the whole set).

The kill switch's log is **byte-for-byte** the release tree's: the two `-lv 4` logs (pre-P2 build vs
`MOE_EXPERT_GTT=0`) normalise to the same 329 lines, differing only in the ephemeral server port and the four
throughput figures.  `scripts/check-gpu-faults.sh --since "90 min ago"` over every arm's log plus the kernel
journal: **clean**, and no `HSA_STATUS_ERROR_MEMORY_FAULT` in any arm.

### 14.7 What this does not measure (and must not be read as measuring)

* **Eviction, the streaming refill, and the `>RAM` case** — P3/P4/P5.  Every arm above has its expert set
  *fitting*; the census prints "a hot subset" when it does not, and says there is no eviction.
* **Pipelining.**  §13.4's requirement (fills behind compute) is untouched: P2 has exactly one fill, at load.
* **`--fit`'s view of the store.**  A `--fit` probe load is `no_alloc`, so the store plan is skipped entirely
  (`llama_model_loader.cpp`: `if (no_alloc) return;`) and the probe reserves nothing — confirmed by exactly ONE
  `plan_gtt_expert_store` line per run with `--fit on` (the default).  But `--fit` therefore does **not** account
  for the store when it sizes the context; that is P1.3's item, and on this box it did not matter.
* **Reclamation under memory pressure** — probe `16`, still owed, and now more load-bearing: the store is
  registered and `MemAvailable` fell by exactly the store's extent on every arm (e.g. 19 126 MiB on the 35B arm).
  Nothing here tests whether the kernel can take it back.
* **gfx1150 / Windows / a carve-out APU**, and any second device.
