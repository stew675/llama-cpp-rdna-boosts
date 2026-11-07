# Storage oracle report

Generated 2026-11-07 15:46:39 UTC on `radix` by `storage-oracle.sh` v1.
**Scope: storage and load mechanism only.** It says nothing about VRAM fit, tensor splits, quantization
or architecture -- check those separately.

## 1. Recommendation

### First (cold) load

**`--load-mode none --lazy-mode mlock`** — the working set fits in RAM (16005 MiB) and RLIMIT_MEMLOCK allows pinning it, so the first load's populate is paid once and every restart is served from pinned page cache.

### Restarts

**`--load-mode none --lazy-mode mlock`** — unchanged. The pin survives the process (r230 measured a page-table walk of **0.19 s for 27 GiB** to re-establish it on a warm cache), so a restart reads nothing from the device.
Warm-cache rate measured: 30.37 GB/s against 2.97 GB/s from the device.

| configuration | first cold load (s) | restart (s) | residency | on-demand safety | verdict | evidence |
|---|---|---|---|---|---|
| `--load-mode mmap --lazy-mode mlock` | 6.0 | 0.3 | lazy tables pinned, survive the process | safe (pinned, never reclaimed) | **ALTERNATIVE** | non-lazy bulk at the mapped fault-in rate 1.28 GB/s (3.4 s more than dio on the first load); the pin re-establishes in a page-table walk once the pages are warm (r230: 0.19 s / 27 GiB), so a restart reads nothing |
| `--load-mode none --lazy-mode mlock` | 3.1 | 0.3 | lazy tables pinned, survive the process | safe (pinned, never reclaimed) | **RECOMMENDED** | non-lazy bulk at the buffered rate 2.44 GB/s plus the pin's one-time populate; a restart reads nothing from the device (warm 30.37 GB/s) -- the form r230's fastest restart used (8.26 s ready-to-serve, 0.19 s to re-pin 27 GiB) |
| `--load-mode none --lazy-mode off` | 3.1 | 0.3 | page cache only (reclaimable) | safe while streaming | **AVOID** | buffered pread 2.44 GB/s cold, 1.22x dio; fills the cache, so a restart is warm while the pages live |
| `--load-mode dio --lazy-mode off` | 2.6 | 2.6 | none (bypasses the cache) | safe (streams ahead, never faults mid-run) | **ALTERNATIVE** | O_DIRECT 2.97 GB/s, 1.00x dio; every restart re-reads the file, and alignment can read more bytes than the file holds |
| `--load-mode mmap --lazy-mode off` | 6.0 | 0.3 | page cache only (reclaimable) | UNSAFE if tables stay on demand (see AVOID below) | **AVOID** | the fault-in path runs at 1.28 GB/s -- pin it (mlock) to pay that once, or stream it; leaving it on demand pays it on every access |

## 2. Avoid

- **AVOID** `--load-mode mmap --lazy-mode off (mapped, never pinned, never reused)` — fault-in 1.28 GB/s vs O_DIRECT 2.97 GB/s, i.e. 3.4 s more on this 7.6 GiB file -- and it pays that again whenever the pages are reclaimed. _Do instead:_ either pin it (--lazy-mode mlock: one populate, then every restart is free) or stream it (--load-mode dio/none); what is wrong is the middle ground
- **AVOID** `--lazy-mode off when the working set fits (an in-process copy is not a cache)` — 16005 MiB fits beside MemAvailable 68295 MiB and RLIMIT_MEMLOCK allows pinning; the warm page cache serves 30.37 GB/s, and a 'none'/off load reads the file into a buffer that dies with the process. _Do instead:_ use --lazy-mode mlock: the pinned pages survive the process, so the next start reads nothing from the device
- **AVOID** `any on-demand disk tier (--lazy-mode on|auto on large tables, MOE_HOST_POOL_DIO=1, a KV spill/window with a host tail)` — 4 KiB read after a 250 ms pause: 16313.0 us; 4 KiB p99 while streaming: 5853.0 us. _Do instead:_ everything must be resident before serving, or read ahead of the stall; there is no cheap late read here
- **AVOID** `assuming this file streams at its best rate` — bands range 0.57 GB/s (at 0 MiB) to 2.62 GB/s (at 4915 MiB). _Do instead:_ re-write or move the file to relocate it; probe the band map again after
- **TRADEOFF** `--load-mode dio on every start` — warm page cache serves 30.37 GB/s against 2.97 GB/s from the device. _Do instead:_ dio for the cold load only; let the page cache serve the restarts (or pin them with --lazy-mode mlock)

## 3. Facts

| | |
|---|---|
| model file | `/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf` |
| model set | 1 file(s), 7.6 GiB |
| filesystem | xfs on `/llm` from `/dev/nvme0n1p1` (rw,noatime,seclabel,inode64,logbufs=8,logbsize=32k,noquota) |
| free space | 405.9 GiB |
| block device | `/dev/nvme0n1` Lexar SSD NM1090 PRO 2TB (fw Y0527A00) |
| PCIe link | unknown (lspci -vv needs root for LnkSta) at `0000:00:01.2` |
| file layout | 1 extent(s), st_blksize 4096 |
| queue | scheduler none, rotational 0, max_sectors 512 KiB, read_ahead 128 KiB |
| memory | MemTotal 93.5 GiB, MemAvailable 66.7 GiB |
| working set | 16005 MiB = model files + 8 GiB (KV/pools/compute, `--resident-gib`) |
| RLIMIT_MEMLOCK | unlimited |
| lazy tables (all --lazy-mode mlock can pin) | 1 table(s) -- pinned 1 table(s), 2856 MiB |
| kernel / cpu | 7.2.15-300.fc45.x86_64 / 16 threads, page size 4096 |
| probe cost | about 14 GiB read |

## 4. Measurements

### 4.1 Band map — is this file uniformly fast on the device?

976 MiB `O_DIRECT` reads, 8 evenly spaced offsets. A band that is much slower than
the best one is a placement problem, not a filesystem problem, and the fix is to re-write the file.
Note these chunks are one `pread` each, which the block layer splits into 512 KiB commands and
pipelines deeply, so they read *higher* than the 4 MiB sequential arm in 4.3 (which is closer to how
the loader issues reads). Read this table for the shape, 4.3 for the rate to predict from.

| offset (GiB) | GB/s |
|---|---|
| 0.0 | 0.57 |
| 1.0 | 0.88 |
| 1.9 | 0.76 |
| 2.9 | 1.67 |
| 3.8 | 1.56 |
| 4.8 | 2.62 |
| 5.7 | 1.57 |
| 6.7 | 1.16 |

### 4.2 Cold per-read latency

Every sample evicted and verified cold (`mincore`); skipped-warm samples are counted, never timed.

| mechanism | p50 | p90 | p99 | max | >1 ms | skipped |
|---|---|---|---|---|---|---|
| `O_DIRECT` 4 KiB | 81.0 us | 152.0 us | 163.0 us | 4393.0 us | 2 | 0 |
| buffered pread 4 KiB | 150.0 us | 490.0 us | 540.0 us | 542.0 us | 0 | 0 |
| mmap fault 4 KiB | 150.0 us | 162.0 us | 535.0 us | 1707.0 us | 2 | 0 |
| `O_DIRECT` 1 MiB | 299.0 us | 627.0 us | 2753.0 us | 3977.0 us | 11 | 0 |

### 4.3 Sequential throughput

| path | GB/s |
|---|---|
| `O_DIRECT`, 4 MiB blocks | 2.97 |
| buffered pread, 1 MiB (after eviction) | 2.44 |
| buffered pread, 1 MiB (warm page cache) | 30.37 |
| `mmap` + `MAP_POPULATE` (the fault-in path) | 1.28 (0% was still cached) |
| page-cache residency at the start / before the buffered arm | 93.4% / 85.2% |
| page-cache residency after the probes | 97.5% |

### 4.4 Idle-gap ladder — cost of a read after a pause

This is the number no specification publishes, and it decides whether a read *during* decode is safe.

| idle before the read | p50 latency |
|---|---|
| 5 ms | 2533.0 us |
| 50 ms | 4743.0 us |
| 250 ms | 16313.0 us |
| 1000 ms | 16320.0 us |

### 4.5 Cold 4 KiB while the same file streams

| p50 | p90 | p99 | samples > 1 ms |
|---|---|---|---|
| 314.0 us | 2083.0 us | 5853.0 us | 66 |

## 5. Verification

Binary: `/home/stew675/llama.cpp/build-rocm/bin/llama-cli` — args: `-ngl 99 -c 4096` — `--single-turn --no-warmup -lv 4`, 1 token(s).

| arm | result | wall clock (s) | device read (MiB) | effective (GB/s) | page cache before | extra args |
|---|---|---|---|---|
| shipped default (no --load-mode) | OK | 7.05 | 4958 | 0.69 | 0.0% | `` |
| recommended (--load-mode none --lazy-mode mlock), cold | OK | 7.08 | 7813 | 1.08 | 0.0% | `--load-mode none --lazy-mode mlock` |
| recommended, WARM page cache (a restart) | OK | 1.30 | 0 | n/a | 100.0% | `--load-mode none --lazy-mode mlock` |

**Prediction inconclusive.** The two arms differ by -0.4% of wall clock (threshold +/-20%), which is noise for a single run. Interleave the arms, or use `--cold` for a stricter comparison.

**Restart effect.** Cold 7.08 s, warm restart 1.30 s -- the second start is 18% of the first. That is the page cache doing the work; a pin decides whether those pages are still there.

**How to read this.** Both arms load the same bytes from the same file and generate the same one
token, so the difference is the mechanism. *Device read* is the byte count the kernel attributes to
this block device (`/sys/block/.../stat`) across the arm — it proves the arm really read the file and
how much of it — and *effective* is that divided by the wall clock. The page cache is evicted before
each arm so the first one does not race the second one's cache. Compare **ranks**, not absolute seconds — device upload and graph setup are in
both columns and this report does not model them. One run per arm: interleave (`A/B/A/B`) manually
if you are deciding on a noisy box.

## 6. Decision rules used

| threshold | value | meaning |
|---|---|---|
| `T_DIO_BEATS` | 1.30 | O_DIRECT must beat the mapped fault path by this factor to be the cold load mode |
| `T_SAME` | 1.15 | a mode within this factor of the best is ACCEPTABLE rather than RECOMMENDED |
| `T_SPREAD` | 0.50 | worst band below this fraction of the best => placement warning |
| `T_WAKE_US` | 1000 | idle-gap wake above this forbids on-demand reads |
| `T_STREAM_P90_US` / `T_STREAM_P99_US` | 500 / 1000 | 4 KiB p90/p99 under a stream above this forbids on-demand reads |
| `T_CACHE_FIT` | 0.80 | a resident plan must need no more than this fraction of MemAvailable |
| `T_WARM_GAIN` | 2.0 | warm/cold rate above this makes `dio` a tradeoff, not a free win |

## 7. What this did not test

- VRAM fit, tensor splits (`-sm`), `-ncmoe`, quantization, architecture: out of scope by design.
- Multi-process interference: probes run one at a time, and never in parallel with a benchmark.
- Throughput under *multiple* concurrent readers (one stream only).
- Rotational or network storage beyond what the facts table shows.

Reproduce any arm with:

```sh
/tmp/storage-oracle.1000/storage-probe map  "/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf" 976 8 0
/tmp/storage-oracle.1000/storage-probe rand "/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf" dio 4096 3000 1024 4915
/tmp/storage-oracle.1000/storage-probe bulk "/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf" dio 4 244 4915
/tmp/storage-oracle.1000/storage-probe pop  "/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf" 1024 4915
```
