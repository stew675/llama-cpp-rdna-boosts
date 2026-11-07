# storage-oracle — what load/residency configuration to use for *this* model file on *this* device

Point it at a model file; get a markdown report that says which `--load-mode`/`--lazy-mode` to use,
what the cold load will cost, and — the more useful half — **which configurations to avoid, each with
the measurement that condemns it**.

```sh
./storage-oracle.sh /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
./storage-oracle.sh --verify --verify-args "-ngl 99 -sm tensor" --report oracle.md MODEL.gguf
```

It changes nothing, needs no root, and never clears the page cache (unless you ask for `--cold`).

## Why this exists

Two facts make storage advice impossible to guess:

1. **The same file is not the same speed on the same drive.** Measured on the maintainer's box: one
   51 GiB shard reads at **0.23 GB/s** in its first GiB, **2.0 GB/s** at 8 GiB, and **12 GB/s** at
   17 GiB and beyond. A whole-file average hides exactly the number that decides the recommendation,
   and *which* band a model lands in depends on when it was written, not on the drive's spec sheet.
2. **Idle gaps cost more than the reads.** On the same box, a cold 4 KiB read costs **74 us** while
   the drive is busy and **16.2 ms** after 250 ms of idle. That single number decides whether any
   on-demand read during decode is survivable — and no datasheet publishes it.

So the tool measures the file, not the drive.

## What it measures, and what each measurement decides

| probe | decides |
|---|---|
| band map: N x chunk `O_DIRECT` reads across the file | throughput of *this file*; a band < 50 % of the best is a placement problem, and re-writing the file is the fix |
| cold 4 KiB latency, `O_DIRECT` / buffered `pread` / `mmap` fault | page granularity: which mechanism is cheapest when a read is one page of weights |
| cold 1 MiB latency | the granularity most tensor reads actually use |
| sequential throughput: `O_DIRECT` 4 MiB, buffered 1 MiB, buffered warm, `mmap` fault-in | the predicted cold-load time per candidate mode, and the value of the page cache |
| idle-gap ladder (5 / 50 / 250 / 1000 ms) | whether an on-demand read during decode is safe at all |
| cold 4 KiB latency *while the same file streams* | the other half of that question: what a late read costs when prefetch is running |
| working set vs `MemAvailable` and `RLIMIT_MEMLOCK` | whether "keep it resident" plans (`--lazy-mode mlock`) are viable, and whether the kernel will refuse |
| `--verify` restart arm (cache deliberately not evicted) | what a second start actually costs, which is the number the residency decision turns on |
| page-cache residency (`mincore`) | how much of the file is already hot, so a number is never misread as a cold one |
| **APST state** (`nvme_core.default_ps_max_latency_us` + the controller's table when readable) | recorded, not judged: the idle-gap ladder is a property of the *power policy* as much as of the drive (see below) |

## How each verdict is decided

| threshold | value | rule |
|---|---|---|
| `T_DIO_BEATS` | 1.30 | `O_DIRECT` must beat the mapped fault-in path by this factor to be the cold load mode |
| `T_SAME` | 1.15 | a mode within this factor of the best is ACCEPTABLE rather than RECOMMENDED |
| `T_SPREAD` | 0.50 | worst band below this fraction of the best => placement warning |
| `T_WAKE_US` | 1000 | idle-gap wake above this forbids every on-demand tier |
| `T_STREAM_P90_US` / `T_STREAM_P99_US` | 500 / 1000 | 4 KiB p90/p99 under a stream above this forbids every on-demand tier |
| `T_CACHE_FIT` | 0.80 | a resident plan must need no more than this fraction of `MemAvailable` |
| `T_WARM_GAIN` | 2.0 | warm/cold ratio above this makes `dio` a tradeoff rather than a free win |
| `T_VERIFY_MATCH` | 0.20 | `--verify` calls the prediction inconclusive if the two arms differ by less than this |

And one rule that is not a threshold: **the recommendation is chosen on two axes, not one** — the first
cold load, and every restart. A cold-load penalty is worth paying if it converts restarts into
zero-device-read loads, which is what `--lazy-mode mlock` does (see below). The matrix therefore carries
`first cold load (s)`, `restart (s)` and `residency` as separate columns, and a plan that wins one axis
and loses the other is shown as `ALTERNATIVE`, never silently ranked.

Every report prints the measured value behind every cell. A probe that cannot run prints `UNKNOWN`;
it is never guessed. The row vocabulary is:

| verdict | meaning |
|---|---|
| `RECOMMENDED` | wins the axis the plan is being asked about, on a measurement |
| `ALTERNATIVE` | viable, but loses on cold load or on a restart by more than `T_SAME` |
| `INERT` | the mode would do nothing on this model (measured: 0 tables pinned) |
| `UNCONFIRMED` | would be recommended, but the thing it depends on was not measurable without loading the model — run `--verify` |
| `AVOID` | measurably worse on this file/device; the evidence column says by how much |
| `BROKEN` | cannot work here (`RLIMIT_MEMLOCK` too small, `dio` refused) |
| `UNKNOWN` | no probe ran |

## Usage

```
storage-oracle.sh [options] MODEL [MODEL...]

  --report PATH        report path (default ./storage-oracle-<host>-<UTCSTAMP>.md)
  --resident-gib N     RAM needed BESIDES the files (KV cache, pinned pools, compute)   [8]
  --budget-gib N       maximum I/O the probes may read                                  [24]
  --quick              smaller probes; implied when --budget-gib < 12
  --no-io              facts and rules only (no measurements; verdicts become UNKNOWN)
  --verify             also run llama-cli: shipped default vs the recommendation, plus that
                       recommendation again with the cache warm (the restart number)
  --cold               --verify arms drop the page cache first (needs root)
  --binary PATH        llama-cli for --verify (default: ./build/bin/llama-cli,
                       ~/llama.cpp/build-rocm/bin/llama-cli, then $PATH)
  --verify-args "..."  extra llama-cli args; pass the flags you actually run with
                       (default "-ngl 0 -c 4096"; a context is pinned small on purpose)
  --verify-n N         tokens per verify arm                                           [1]
  --probe PATH         use an already-built storage-probe
```

Requires `bash`, `cc`, util-linux (`findmnt`), and optionally `lspci` (link speed) and `filefrag`
(extent count). Linux/POSIX only. The probe is compiled on first use into `${TMPDIR:-/tmp}/storage-oracle.<uid>/`.

## The report

Sections: **1 Recommendation** (first cold load, and restarts; the restart line is "unchanged" when the
recommendation is a pinned plan), **2 Avoid** (each entry marked `AVOID` / `BROKEN` / `UNCONFIRMED` /
`UNKNOWN` / `TRADEOFF` / `ADVISORY`), **3 Facts**, **4 Measurements**, 
**5 Verification**, **6 Decision rules**, **7 What this did not test**. `EXAMPLE-REPORT.md` is a real run.

## Why the pin wins (r230), and when it does nothing

`--lazy-mode mlock` is not a throughput feature — on the delivery's own measurement
(`benchmarks/2026-11-07-lazy-mode-mlock.md`, r230) `mlock`/`auto`/`off` are at parity
(pp2048 2570.4 / 2549.1 / 2574.8, tg32 77.41 / 77.55 / 76.27) with byte-identical greedy text.
It is a **residency** feature:

- `off` copies the lazy table into a private host buffer: 27 GiB of anonymous, swappable memory, and
  **nothing is reusable after the process exits** (`RssAnon` 32.2 GB).
- `mlock` keeps the lazy data path and pins the *mapping*: `VmLck` 26.82 GiB with `RssAnon` 1.3 GB. On
  exit the lock is released but the pages stay in page cache, so the next start **re-pins them in a
  page-table walk — 0.19 s for 27 GiB** — and reads nothing from the device. That is the maintainer's
  fastest restart (`--load-mode none --lazy-mode mlock`, ready to serve in 8.26 s against 13.6 s).
- Cost: one populate at the kernel's `__mm_populate` rate (~4.2 GB/s cold, ~7 s for 27 GiB), and the
  table is unevictable while the process runs.

**It only pins `TENSOR_READ_LAZY` tables** (`per_layer_token_embd` and friends — deepseek41, gemma4,
qwen4exp). On a model without one the flag is inert, which this tool measures rather than assumes: the
`--verify` arms parse the loader's `pinned N table(s), X MiB` line, and a model that pins nothing gets
the mlock rows marked **INERT** with the recommendation moved back to the streaming axis (measured on a
4B model here: `mmap --lazy-mode mlock` 5.17 s vs `mmap --lazy-mode off` 5.18 s cold — the mode did
nothing). Two other traps from the same record: the range must not receive `POSIX_MADV_RANDOM`
(readahead is what the one populate needs), and `mlock` is per process while the *page cache* is shared.

## `--verify`: prediction vs measurement

Runs `llama-cli` twice with identical work (same file, same prompt, one token) and reports wall clock,
**bytes this block device actually read**, and the effective rate, with the page cache evicted before
each arm so the first arm cannot warm the second. Then it states whether the prediction held, within
±`T_VERIFY_MATCH`.

Without `--verify` the pinning plans cannot be recommended: `--lazy-mode mlock` pins only a
`TENSOR_READ_LAZY` table, and nothing measurable on disk says whether this model has one — so they are
reported `UNCONFIRMED` and the streaming mode is recommended instead. With `--verify` the third arm
shows the pin (or its absence) and the restart time it buys.

Use `--verify-args` with the flags you actually run (tensor split, `-ncmoe`, number of GPUs). On a
multi-GPU box pin the devices (`HIP_VISIBLE_DEVICES`) — an unpinned split can enumerate an iGPU and
fail at model load for reasons that have nothing to do with storage. The absolute numbers include
device upload and graph setup, which this report does not model: **compare the arms, not the seconds.**

## Why the probe is written the way it is

- **Cold by construction, without root.** `O_DIRECT` bypasses the cache. The page-cache arms evict and
  then **verify with `mincore`** before timing anything, escalating the eviction window (2 MiB -> 32 MiB
  -> 256 MiB) until the page is really gone. `POSIX_FADV_DONTNEED` works at folio granularity: on a
  filesystem caching large folios (XFS does), dropping a 4 KiB range out of a multi-megabyte folio
  **silently does nothing**, and the "cold" read comes back from the cache in 0.2 us. That bug was real
  here and produced a plausible-looking 0.2 us p50 on a file with 39 GiB of hot page cache. A sample
  that cannot be made cold is counted (`skipped_warm=`), never timed.
- **Unwritten extents are labelled, not measured.** A trimmed/unwritten chunk is answered by the
  controller's zero-fill path at bus speed (10 us at QD1 on this box), which looks like a spectacular
  drive. The band map flags those steps `zeros=1` and the driver ignores them.
- **One line of `key=value` per measurement.** The probe never knows what a verdict is; the driver
  never parses prose.
- **Single-threaded, QD1, one probe at a time** — except the deliberate under-stream arm, which runs a
  sequential reader and a random reader against the same file at once. Never run it beside a benchmark.

## The power policy changes the answer, so the report records it

The idle-gap ladder is a property of the drive **and** of its APST policy, and on the maintainer's box
the policy is worth more than any load-mode choice: the Lexar NM1090 PRO's APST idle state cost
**16.3 ms** per cold read, and turning APST off takes that to **50-90 us** for gaps up to ~50 ms and
**~5 ms** after a longer pause (the drive's own firmware level, which no host setting reaches):

| idle gap | APST at its default | APST off |
|---|---|---|
| 5 ms | 2 533 us | 88 us |
| 50 ms | 4 743 us | 50 us |
| 250 ms | 16 313 us | 4 716 us |
| 1000 ms | 16 320 us | 5 359 us |

It is host-managed, but three plausible shortcuts do **not** work, and all three were tried:
`nvme set-feature -f 0x0c -v 0` is refused with `EPERM` (the kernel driver owns the APST feature);
`nvme reset` does not re-read `nvme_core.default_ps_max_latency_us` (a controller *reset* is not a
re-probe); and a user-space admin passthrough of Set Features is refused for the same reason as the
ioctl. What works is writing the parameter and **re-probing that one controller**:

```sh
echo 0 | sudo tee /sys/module/nvme_core/parameters/default_ps_max_latency_us
sudo umount /llm
echo 0000:04:00.0 | sudo tee /sys/bus/pci/drivers/nvme/unbind
echo 0000:04:00.0 | sudo tee /sys/bus/pci/drivers/nvme/bind
sudo mount /llm
```

Because the parameter is read at controller *creation*, re-probing one function scopes the change to
that drive (the SN850X keeps APST and its 0.6 ms wake). On this box that runs at boot from a
`Before=local-fs-pre.target` oneshot, so `/llm` mounts after the re-probe — the tool itself still needs
no root and changes nothing. The cost is that the drive never drops below PS0: TechPowerUp measured
this model at **1.454 W idle with ASPM disabled**
([review](https://www.techpowerup.com/review/lexar-nm1090-pro-4-tb/7.html)), and PS0 adds sub-watt to
that. Note the same review measured **7.0 W reading / 8.5 W writing** against a 10 W `id-ctrl`
descriptor — i.e. the descriptor is a ceiling, which is why the report treats every watt figure as a
declaration and only trusts its own latencies.

So the facts table prints the policy, and a ladder from a box with APST at its default is not
comparable with one from a box with it off.

## What it does not do

No VRAM fit, no `-sm`, no `-ncmoe`, no quantisation or architecture advice, no throughput under
multiple concurrent readers, no rotational/network storage tuning beyond the facts table. It also
does not apply anything — it prints the flags it recommends.

## Relationship to the delivery

This is a consumer/reporter utility under `utilities/`; it is **not part of the patch set** in
`patches/` and is not covered by the delivery's gates. Its recommendations name the delivery's
mechanisms (`--load-mode dio|none|mmap|mlock`, `--lazy-mode off|mlock`, and the disk-backed tiers that
must stay off when `T_WAKE_US`/`T_STREAM_*` say so), so a reader can copy them onto a `llama-cli` or
`llama-server` command line directly.
