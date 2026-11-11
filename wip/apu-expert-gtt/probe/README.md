# probe/ — the programs behind `MEASUREMENTS.md`

Fourteen standalone HIP/C programs plus two measurement helpers (Python).  They **measure only**: nothing here
writes to a model, nothing needs root, and nothing touches the fork, the delivery or `patches/`.  Each one was
the arm that produced a number in [`../MEASUREMENTS.md`](../MEASUREMENTS.md); the sources are byte-identical to
what was run on `halo` (2026-11-09 for 01-10, 2026-11-11 for 11-14, see that file's §13).

## Build

`hipcc` from `/usr/bin` links ROCm 7.1 and fails these with `undefined symbol: hipGetDeviceCount`.  Use the
10.1 toolchain explicitly:

```bash
export PATH=/opt/rocm-10.1.0-gfx1151/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx1151/lib:$LD_LIBRARY_PATH
hipcc -O2 -o 01-props 01-props.cpp          # each file builds on its own
```

Verified: all ten build on `halo` with ROCm 10.1.0 / clang 20 / gfx1151.

## Run

```bash
mkdir -p /tmp/xnack-probe && cd /tmp/xnack-probe
GGUF=/llm/models/DeepSeek/Q4_K_XL/DeepSeek-V4-Flash-…-00004-of-00005.gguf    # 46 GiB shard
head -c 67108864 /dev/urandom > /tmp/xnack-probe/blob.bin                    # for 02

./01-props
./02-fault hot ; ./02-fault cold_reg           # each arm in its own process: a fault kills it
HSA_XNACK=1 ./02-fault hot                     # proves the flag is inert
./03-bandwidth
./04-fill-and-register "$GGUF"
./05-dirty-and-discard  "$GGUF"
./06-dontneed-zero-trap
./07-host-write-gtt "$GGUF"
./08-dma-fill "$GGUF"
./09-register-pins ctl   "$GGUF" $((4*1024*1024*1024))
./09-register-pins reg   "$GGUF" $((4*1024*1024*1024))
./09-register-pins cost  "$GGUF" $((1*1024*1024*1024))
./09-register-pins reg   "$GGUF" $(stat -c%s "$GGUF")    # the whole 46 GiB shard: ~9 s, 46 GiB held
./10-coherence ro ; ./10-coherence writable
```

Set `HIP_LAUNCH_BLOCKING=1` when you want a fault to be attributed to the launch that caused it.

| file | what it establishes | arguments / caveats |
|---|---|---|
| `01-props.cpp` | `prop.integrated`, `totalGlobalMem`, `PageableMemoryAccess=0`, `ManagedMemory`, `ConcurrentManagedAccess` | none |
| `02-fault.cpp` | registered vs unregistered read; `HSA_XNACK=1` inert; the exact fault text and exit code | `hot` / `cold_unreg` / `cold_reg` / `anon_*`; needs `/tmp/xnack-probe/blob.bin`; **a fault arm core-dumps by design** |
| `03-bandwidth.cpp` | GPU read bandwidth matrix: `hipMalloc` / anon writable / anon ReadOnly / file ReadOnly | creates `/tmp/xnack-probe/bw.bin` (1 GiB); needs that directory |
| `04-fill-and-register.cpp` | `O_DIRECT` into a plain buffer; `ReadOnly` registration of a `PROT_WRITE` buffer; 233 vs 60 GB/s; the register cycle; page-cache `memcpy` | `<gguf>` |
| `05-dirty-and-discard.cpp` | `smaps` `Private_Dirty` for an anon pool vs a `PROT_READ` file mapping; discard with zero writeback (`pswpout`, `pgpgout`) | `<gguf>` |
| `06-dontneed-zero-trap.cpp` | `MADV_DONTNEED` on a registered anon region the GPU is reading returns **zeros**, silently | none; **this is the trap, not a benchmark** |
| `07-host-write-gtt.cpp` | host read/write of a `hipMalloc` pointer; host `memset` rate; GPU sees it FRESH; `O_DIRECT` into GTT is `EINVAL` | `<gguf>` |
| `08-dma-fill.cpp` | `hipMemcpy` H2D into GTT vs host `memcpy` | `<gguf>` |
| `09-register-pins.cpp` | control vs registered residency under `fadvise`+`madvise`; the hold survives `hipHostUnregister`; register cost; whole-file registration | `ctl\|reg\|cost <file> <bytes>`; the `reg` arm on a 46 GiB file holds 46 GiB for the process lifetime |
| `10-coherence.cpp` | host overwrites a registered range in place and the GPU reads the new bytes, both registration modes | `ro` (default is the ReadOnly arm only if you pass `ro`) / `writable` |

## The P1.0 probes (11-14, 2026-11-11) — the GTT expert store's four assumptions

Written and run on `halo`; results in `MEASUREMENTS.md` §13.  Build them exactly like 01-10.

| # | probe | question it answers | verdict |
|---|---|---|---|
| 11 | `11-thp-register.cpp` | do `MADV_HUGEPAGE` THPs survive `hipHostRegister(Mapped\|ReadOnly)`?  `smaps` `AnonHugePages` before/after, registration cost/GiB, GPU read rate, both fault orders | **yes**, 1024.0 MiB of 1024 stays huge; 17.4 ms/GiB pre-faulted; 232.1 GB/s |
| 12 | `12-pread-registered.cpp` | `pread` **into** a registered RO anon region, then a GPU sum of exactly those bytes — fresh region and the eviction/reuse case | **yes**, `O_DIRECT` accepted, every round's sums match; buffered is faster |
| 13 | `13-dma-from-mapped-file.cpp` | the copy-engine pull from a registered RO `MAP_SHARED` file mapping into a registered anon slot, at slab and span granularity, compute-stream ordered | **yes**, 0.012 ms/slab warm (60.9 GB/s), bytes verified |
| 14 | `14-expert-cost.cpp` | the per-fill cost at real GGUF offsets: cold/warm slab latency, span rate, and the `pread -> launch -> sync` floor  (takes `<file> <offset> <slab_bytes> [count]`) | cold **0.413 ms** median (34.9 ms worst), warm **0.069 ms**, round-trip floor **0.420 ms**; `O_DIRECT` needs a 4096 round-out |

Two helpers feed them and are re-runnable on their own:

| helper | what |
|---|---|
| `gguf-expert-geom.py` | reads a GGUF's tensor table (a minimal GGUF parser) and prints each layer's `*_exps` offsets, the contiguous per-expert slab size and the whole-set footprint.  `--layer N --shell` prints `OFFSET SLAB COUNT` for probe 14 |
| `moe-heatmap-read.py` | reads the delivery's persisted `MHT2` heat map and prints the traffic concentration (coverage curve, never-touched share, resident-set footprint at 50/80/90/95/99 %) per band |

## Owed: probe 15 — the per-session touch distance

The one thing still unmeasured, and it is a *workload* measurement rather than a mechanism one: how often an
expert is used **1x / 2x / 3x within one session**, which is what sets the admission threshold `N`
(`../README.md` §3.2, §7).  The persisted heat map is a long-horizon counter, so it gives concentration but not
session sequences.  Two candidate sources: the delivery's own `heat`/`ghost` census on a discrete box (it
prints per-table hit/eviction tallies), or a routing trace from a CPU-only run (`-ngl 0`) replayed offline.
Nothing is blocked on it: the cost model says `N = 2` already pays for itself, so it tunes a default rather
than deciding the design.

## What these do not cover

* Any `llama.cpp` behaviour or throughput — these are mechanism probes only.  Throughput and text-identity arms
  belong to `scripts/gate-uma-experts.sh` (`../IMPLEMENTATION.md` §7).
* The per-session touch distance (probe 15, above), and any end-to-end decode/prefill rate.
* Windows / a fixed VRAM carve-out APU, and gfx1150.  Unmeasured (see `../README.md` §7).
* Multiple devices.  gfx1151 is one device here; a `-sm tensor` arm on a UMA + discrete pair is out of scope.
