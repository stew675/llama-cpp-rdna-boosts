# DESIGN — an adaptive host-read transport (`--load-mode`, `MOE_HOST_POOL_TRANSPORT`)

**Status: designed + prototyped (2026-11-11), WIP. `0002-adaptive-host-read-transport.patch`, built and
exercised; nothing in `patches/`, no gate run.**

## Why

The campaign measured a **crossover**, not a winner (`README.md` §Experiment 1, `RESULTS.md`):

| page cache holds the host set? | mapping | buffered `pread` | O_DIRECT |
|---|---|---|---|
| yes (40G cgroup, 4 GiB set) | **3 040 cold / 17 102 warm**\* | 2 491 / 7 068 | 2 177 |
| no (4G cgroup, 8 GiB set) | 1 050 / 1 132 | 2 240 / 2 580 | 2 125 |
| no (1G cgroup) | **593 / 596** | 1 737 / 1 720 | **2 127** |

\* That warm row is a *re-read* microbenchmark (page-table entries already established).  A MODEL run touches
each expert once per pass, and there the mapping pays a PTE fault per 4 KiB page: the gate measured the real
delivered paths on this box and the **buffered arm won both** — fills 29 450 against 14 725 MiB/s, demand
16 348 against 15 197 (`RESULTS.md` §4).  So the mapping's advantage is narrower than the microbenchmark
suggests, and `auto` reaching `file` on an unpressured box is the correct outcome, not a false positive.

Today the arm is a **guess**: `pool_decide_dio_locked` reads `MemAvailable` once, at the first pool build, and
compares it with pool + host set. `MemAvailable` is host-wide (wrong inside a cgroup) and the state it predicts
changes during a long run. When the guess says "fits" and the cache cannot hold the set, the buffered arm is the
mapping — the sub-100-MB/s collapse the maintainer hit.

The safety property that lets us switch **during** a run: all three transports read the same page-cache pages into
the same buffer, and greedy text is byte-identical on every arm (`de81b28cab7d`). A transport change can only
change timing. That must stay true: no arm may change the bytes, the length or the destination offset of a read.

## The load modes and the transport

`--load-mode` is the operator's caching decision, so it now names the transport too. `buffered` is **new**: a
mapped master whose host reads are buffered `pread`s — the combination that used to be unrepresentable, because
r212 welded the fill transport to `--load-mode` so only `dio` ever got a non-mapping read.

| `--load-mode` | master | host-read transport | adaptive? |
|---|---|---|---|
| `auto` (default) | mmap | **adaptive** (starts on the mapping) | **yes** |
| `mmap` | mmap | mapping (`memcpy` out of the page cache) | locked |
| `buffered` **(new)** | mmap | buffered `pread` per expert | locked |
| `dio` | O_DIRECT | O_DIRECT | locked |
| `mmap+mlock` / `mlock` / `none` | as today | mapping (out of a resident master) | locked |

Explicit modes **lock** the transport; only `auto` falls back. `MOE_HOST_POOL_TRANSPORT=map|file|dio` and the
legacy `MOE_HOST_POOL_DIO=0|1` still override (an A/B or a bisect must be the arm it names), and a library
consumer that never calls the loader's transport hook keeps the delivered heuristic.

## 1. Start optimistically

`auto` always starts on the **mapping**: it is what every delivered run does today, it is the arm whose
advantage is cache-dependent (so it should be sampled, not predicted), and being wrong costs one probe cadence
(128 fills). No capacity prior selects the arm, and no probe is taken ahead of time.

The cgroup-aware `pool_available_ram_bytes()` (`min(MemAvailable, cgroup memory.max − memory.current)`) stays, but
as the **root-cause fix for the pinned-pool RAM ceiling**, which had the same host-wide bug — not as a transport
selector.

## 2. Two relative triggers — no MiB/s, no ms, nothing box-specific

**Stall detection.** A read that took `TR_STALL_K` times this arm's own running mean, `TR_STALL_N` times running,
escalates at once. This is the "the fill blocked on a fault storm while its neighbours were fine" case, and it
fires inside one ubatch.

**The probe-fill A/B.** Every `TR_PROBE_FILLS` fills, the **next arm serves that fill instead**. The bytes are
identical, so the pool contents are identical either way, and the sample costs **no extra I/O** — it is a fill
that had to happen. That makes it an exact A/B: two latencies for the same bytes, on the same box, in the same
cache state, at the same moment. `TR_PROBE_N` consecutive wins by `TR_PROBE_GAIN` latch the next step of the
ladder (`map → file → dio`). The probe runs on the demand/bounce path too, because on a `-sm tensor` split that
read is 54 % of a decode step (r261).

Why a probe rather than an ahead-of-time calibration: a cold probe **cannot** be made honest in-process —
`posix_fadvise(DONTNEED)` skips pages the process has mapped (the probe then read 32 452 MiB/s, i.e. straight out
of the cache) and `madvise(MADV_DONTNEED)` drops the PTEs but not the page cache. And a cold-rate ranking is the
wrong question anyway: the mapping's advantage is the warm case, so it would pick O_DIRECT on every box.

## 3. One way, deliberately

Escalation never reverses inside a run. This is a **decision**, not a gap: a box that fell short once is likely to
again, and the page cache a reverse switch would go on to rebuild may itself be what caused the shortfall — so a
slower consistent point beats a flapping optimum. A reverse direction is a possible later **opt-in**; it needs a
window measured in minutes and a hysteresis argument that does not exist yet.

## 4. Controls (pinning, never enabling)

| knob | class | meaning |
|---|---|---|
| `--load-mode auto` | the adaptive arm (default) | optimistic start + stall/probe monitor |
| `--load-mode mmap\|buffered\|dio` | pin | lock the transport |
| `MOE_HOST_POOL_TRANSPORT=map\|file\|dio\|auto` | override / A/B | wins over `--load-mode` |
| `MOE_HOST_POOL_DIO=0\|1` | legacy override | still honoured; maps onto `file`/`dio` |
| `MOE_HOST_POOL_TR_{STALL_K,STALL_N,PROBE_FILLS,PROBE_GAIN,PROBE_N}` | tuning | dimensionless thresholds only |
| `MOE_HOST_POOL_TR_START` | **campaign only** | start on a named arm (staging an escalation test) |

## 4a. The gate (`gate-host-read-transport.sh`, 24/24)

Seven text arms (hashed with no `-lv`) and seven census arms (`-lv 4`), so no assertion is read off the log a
hashing arm produced.  It proves, per arm, that the transport named was the transport used (the counters differ
by path: `mapping memcpys` against `storage reads`, and the same split on the demand path), that `auto`
**converges on the arm the pinned arms prove fastest in the same run** (or stays put within 15 %), that a
transition cites its evidence, that a starving cgroup makes the ladder actually move, and that the −41 %
coupling is unreachable.  The convergence check replaced an assumption of mine ("a RAM-rich box must stay on the
mapping") that the pinned arms disproved on this hardware — see `RESULTS.md` §4.

## 5. What the prototype established

| check | result |
|---|---|
| text, `mmap` / `buffered` / `auto` / `dio`, and the static four-way instrument | **identical**, `144 chars sha=de81b28cab7d` |
| `--load-mode buffered` | `host read transport -> file (pinned by --load-mode)` |
| `--load-mode mmap` | `-> map (pinned by --load-mode)` |
| `--load-mode auto` | `-> map (optimistic start …)` |
| `--load-mode dio` + a pool | unchanged: `NOT APPLIED: a pool needs a mapped load` |
| probe A/B inside a 32G scope | latched on evidence: `map -> file (probe fill: file served the same read in 33 us against this arm's 50 us, 3 samples running)` |
| the removed absolute floor / epoch-rate rule | both fired wrongly (a warm box switched off the mapping after one 16 444 → 9 624 MiB/s epoch), which is why they are gone |

## 6. Owed before this ships

0. **The probe's gain threshold** (`TR_PROBE_GAIN`, currently 1.5) is NOT set from measurement yet, and the
   interleaved A/B shows why it matters: under starvation the probe measured `dio` 1.68x faster on the read
   while `file` was **12 % better end to end** (the buffered arm's page-cache population is what a single-read
   latency cannot see).  The threshold must be measured the way `RESULTS.md` §5 describes — interleaved, ballast
   held, cache dropped per arm, `file` preferred on a tie.  The first cut of that matrix also caught a real bug
   (the probe cadence counter was incremented only by the path the probe bypasses, so every fill became a
   probe), which is fixed: probes are 1 in 128 and `auto` is neutral when it does not latch.
1. **The gate.** A `gate-host-pool.sh` arm at a cache-starved cgroup that must log a transition with N > 0
   reads on each arm (proving the ladder is not vacuous), the three pinned arms for text identity, and TODO
   #306's original condition — a cell varying the transport **and** `MOE_HOST_POOL_STAGE_SPAN` together, since a
   `FILE` transport with the span off is the measured −41 % pair (`auto` must force the span on, or drop to the
   mapping).
2. **The interleaved two-direction A/B** (ABBA, 44G/32G, transport × span) to set `TR_PROBE_*` from measurement
   rather than reasoning; the 44G band is ±25 %, so single-pass arms cannot set them.
3. **The decode ladder.**  `llama-batched-bench -npl 1,4,8` is **run** (`RESULTS.md` §5): `auto` stays on `map`
   and is within noise at every width on the cacheable 35B-A3B, a pinned `file` costs ~4 % at `npl 1` and
   nothing at `npl 8`.  The **MTP four-axis protocol** is still owed before a promotion — the change is
   host-side with no kernel or numerics, but that is the contract for a decode-path change.
4. **Windows.** The buffered arm there is `ReadFile`, the direct arm `FILE_FLAG_NO_BUFFERING`; the reporter
   measured them within ~15 % (616-1 114 vs 638 MiB/s), and the capacity test uses `GlobalMemoryStatusEx` (no
   cgroup). `--load-mode` pins it, but neither arm has been measured in this form.
5. **Attribution.** A benchmark that latches mid-run is not reproducible; every transition is logged for that
   reason, and a measurement arm must pin the transport.
