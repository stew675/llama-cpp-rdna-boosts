# probe/ — the programs behind `MEASUREMENTS.md`

Ten standalone HIP/C programs.  They **measure only**: nothing here writes to a model, nothing needs root, and
nothing touches the fork, the delivery or `patches/`.  Each one was the arm that produced a number in
[`../MEASUREMENTS.md`](../MEASUREMENTS.md); the sources are byte-identical to what was run on `halo`
(2026-11-09).

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

## What these do not cover

* Any `llama.cpp` behaviour or throughput — these are mechanism probes only.  Throughput and text-identity arms
  belong to `scripts/gate-uma-arena.sh` (`../IMPLEMENTATION.md` §7).
* Windows / a fixed VRAM carve-out APU, and gfx1150.  Unmeasured (see `../README.md` §7).
* Multiple devices.  gfx1151 is one device here; a `-sm tensor` arm on a UMA + discrete pair is out of scope.
