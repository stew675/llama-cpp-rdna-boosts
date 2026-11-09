# MEASUREMENTS — `wip/apu-expert-pool`

Every number in [`README.md`](README.md) and [`IMPLEMENTATION.md`](IMPLEMENTATION.md) comes from here.  Nothing
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
