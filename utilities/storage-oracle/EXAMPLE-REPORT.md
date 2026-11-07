# Storage oracle report

Generated 2026-11-07 17:28:57 UTC on `radix` by `storage-oracle.sh` v1.
**Scope: storage and load mechanism only.** It says nothing about VRAM fit, tensor splits, quantization
or architecture -- check those separately.

## 1. Recommendation

### First (cold) load

**`--load-mode none --lazy-mode mlock`** — the working set fits in RAM (16005 MiB) and RLIMIT_MEMLOCK allows pinning it, so the first load's populate is paid once and every restart is served from pinned page cache.

### Restarts

**`--load-mode none --lazy-mode mlock`** — unchanged. The pin survives the process (r230 measured a page-table walk of **0.19 s for 27 GiB** to re-establish it on a warm cache), so a restart reads nothing from the device.
Warm-cache rate measured: 30.39 GB/s against 2.42 GB/s from the device.

| configuration | first cold load (s) | restart (s) | residency | on-demand safety | verdict | evidence |
|---|---|---|---|---|---|
| `--load-mode mmap --lazy-mode mlock` | 4.2 | 0.3 | lazy tables pinned, survive the process | safe (pinned, never reclaimed) | **ALTERNATIVE** | non-lazy bulk at the mapped fault-in rate 1.83 GB/s (1.0 s more than dio on the first load); the pin re-establishes in a page-table walk once the pages are warm (r230: 0.19 s / 27 GiB), so a restart reads nothing |
| `--load-mode none --lazy-mode mlock` | 3.5 | 0.3 | lazy tables pinned, survive the process | safe (pinned, never reclaimed) | **RECOMMENDED** | non-lazy bulk at the buffered rate 2.17 GB/s plus the pin's one-time populate; a restart reads nothing from the device (warm 30.39 GB/s) -- the form r230's fastest restart used (8.26 s ready-to-serve, 0.19 s to re-pin 27 GiB) |
| `--load-mode none --lazy-mode off` | 3.5 | 0.3 | page cache only (reclaimable) | safe while streaming | **ACCEPTABLE** | buffered pread 2.17 GB/s cold, 1.12x dio; fills the cache, so a restart is warm while the pages live |
| `--load-mode dio --lazy-mode off` | 3.2 | 3.2 | none (bypasses the cache) | safe (streams ahead, never faults mid-run) | **ALTERNATIVE** | O_DIRECT 2.42 GB/s, 1.00x dio; every restart re-reads the file, and alignment can read more bytes than the file holds |
| `--load-mode mmap --lazy-mode off` | 4.2 | 0.3 | page cache only (reclaimable) | UNSAFE if tables stay on demand (see AVOID below) | **AVOID** | the fault-in path runs at 1.83 GB/s -- pin it (mlock) to pay that once, or stream it; leaving it on demand pays it on every access |

## 2. Avoid

- **AVOID** `--load-mode mmap --lazy-mode off (mapped, never pinned, never reused)` — fault-in 1.83 GB/s vs O_DIRECT 2.42 GB/s, i.e. 1.0 s more on this 7.6 GiB file -- and it pays that again whenever the pages are reclaimed. _Do instead:_ either pin it (--lazy-mode mlock: one populate, then every restart is free) or stream it (--load-mode dio/none); what is wrong is the middle ground
- **AVOID** `--lazy-mode off when the working set fits (an in-process copy is not a cache)` — 16005 MiB fits beside MemAvailable 67729 MiB and RLIMIT_MEMLOCK allows pinning; the warm page cache serves 30.39 GB/s, and a 'none'/off load reads the file into a buffer that dies with the process. _Do instead:_ use --lazy-mode mlock: the pinned pages survive the process, so the next start reads nothing from the device
- **AVOID** `any on-demand disk tier (--lazy-mode on|auto on large tables, MOE_HOST_POOL_DIO=1, a KV spill/window with a host tail)` — 4 KiB read after a 250 ms pause: 4715.0 us; 4 KiB p99 while streaming: 9777.0 us. _Do instead:_ everything must be resident before serving, or read ahead of the stall; there is no cheap late read here
- **AVOID** `assuming this file streams at its best rate` — bands range 0.93 GB/s (at 6860 MiB) to 3.44 GB/s (at 4915 MiB). _Do instead:_ re-write or move the file to relocate it; probe the band map again after
- **TRADEOFF** `--load-mode dio on every start` — warm page cache serves 30.39 GB/s against 2.42 GB/s from the device. _Do instead:_ dio for the cold load only; let the page cache serve the restarts (or pin them with --lazy-mode mlock)

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
| memory | MemTotal 93.5 GiB, MemAvailable 66.1 GiB |
| working set | 16005 MiB = model files + 8 GiB (KV/pools/compute, `--resident-gib`) |
| RLIMIT_MEMLOCK | unlimited |
| power policy (APST) | nvme_core.default_ps_max_latency_us=0 (applied at controller creation; the live table needs root to read) |
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
| 0.0 | 1.65 |
| 1.0 | 2.71 |
| 1.9 | 1.51 |
| 2.9 | 1.67 |
| 3.8 | 1.87 |
| 4.8 | 3.44 |
| 5.7 | 1.22 |
| 6.7 | 0.93 |

### 4.2 Cold per-read latency

Every sample evicted and verified cold (`mincore`); skipped-warm samples are counted, never timed.

| mechanism | p50 | p90 | p99 | max | >1 ms | skipped |
|---|---|---|---|---|---|---|
| `O_DIRECT` 4 KiB | 54.0 us | 999.0 us | 1000.0 us | 1344.0 us | 102 | 0 |
| buffered pread 4 KiB | 54.0 us | 65.0 us | 70.0 us | 332.0 us | 0 | 0 |
| mmap fault 4 KiB | 56.0 us | 138.0 us | 229.0 us | 1538.0 us | 2 | 0 |
| `O_DIRECT` 1 MiB | 288.0 us | 1551.0 us | 4843.0 us | 5217.0 us | 31 | 0 |

### 4.3 Sequential throughput

| path | GB/s |
|---|---|
| `O_DIRECT`, 4 MiB blocks | 2.42 |
| buffered pread, 1 MiB (after eviction) | 2.17 |
| buffered pread, 1 MiB (warm page cache) | 30.39 |
| `mmap` + `MAP_POPULATE` (the fault-in path) | 1.83 (0% was still cached) |
| page-cache residency at the start / before the buffered arm | 100.0% / 88.3% |
| page-cache residency after the probes | 100.0% |

### 4.4 Idle-gap ladder — cost of a read after a pause

This is the number no specification publishes, and it decides whether a read *during* decode is safe.
**It is also a property of the power policy, not just the drive:** APST transitions can multiply it
several-fold (measured on this box: 16.3 ms with APST at its default against 4.7 ms with APST off).
APST here: nvme_core.default_ps_max_latency_us=0 (applied at controller creation; the live table needs root to read).

| idle before the read | p50 latency |
|---|---|
| 5 ms | 89.0 us |
| 50 ms | 53.0 us |
| 250 ms | 4715.0 us |
| 1000 ms | 5360.0 us |

### 4.5 Cold 4 KiB while the same file streams

| p50 | p90 | p99 | samples > 1 ms |
|---|---|---|---|
| 735.0 us | 3891.0 us | 9777.0 us | 79 |

## 5. Verification

Binary: `/home/stew675/llama.cpp/build-rocm/bin/llama-cli` — args: `-ngl 99 -c 4096` — `--single-turn --no-warmup -lv 4`, 1 token(s).

| arm | result | wall clock (s) | device read (MiB) | effective (GB/s) | page cache before | extra args |
|---|---|---|---|---|
| shipped default (no --load-mode) | OK | 6.21 | 4958 | 0.78 | 0.0% | `` |
| recommended (--load-mode none --lazy-mode mlock), cold | OK | 7.28 | 7813 | 1.05 | 0.0% | `--load-mode none --lazy-mode mlock` |
| recommended, WARM page cache (a restart) | OK | 1.30 | 0 | n/a | 100.0% | `--load-mode none --lazy-mode mlock` |

**Prediction inconclusive.** The two arms differ by -17.2% of wall clock (threshold +/-20%), which is noise for a single run. Interleave the arms, or use `--cold` for a stricter comparison.

**Restart effect.** Cold 7.28 s, warm restart 1.30 s -- the second start is 18% of the first. That is the page cache doing the work; a pin decides whether those pages are still there.

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
