# DESIGN — an adaptive host-read transport (`MOE_HOST_POOL_TRANSPORT=auto`)

**Status: designed + prototyped (2026-11-11), WIP. `0002-adaptive-host-read-transport.patch`, built and
exercised; nothing in `patches/`, no gate run.**

## Why

The campaign measured a **crossover**, not a winner (`README.md` §Experiment 1, `RESULTS.md`):

| page cache holds the host set? | mapping | buffered `pread` | O_DIRECT |
|---|---|---|---|
| yes (40G cgroup, 4 GiB set) | **3 040 cold / 17 102 warm** | 2 491 / 7 068 | 2 177 |
| no (4G cgroup, 8 GiB set) | 1 050 / 1 132 | 2 240 / 2 580 | 2 125 |
| no (1G cgroup) | **593 / 596** | 1 737 / 1 720 | **2 127** |

And today the arm is a **guess**: `pool_decide_dio_locked` reads `MemAvailable` once, at the first pool build,
and compares it with pool + host set. `MemAvailable` is **host-wide** — inside a memory cgroup it answers about
RAM the process cannot use (`wip/host-expert-coverage/`) — and the state it predicts can change under a long
run. When the guess says "fits" and the cache cannot hold the set, the buffered arm is the mapping, which is
the sub-100-MB/s collapse the maintainer hit.

**Replace the guess with a capacity measurement, and keep measuring while the run goes.**

## The safety property that makes a LIVE switch acceptable

All three transports read the same page-cache pages into the same buffer, and the campaign proves byte-identical
text for every combination (`de81b28cab7d`, 144 chars — re-verified on the adaptive build, including a run that
switches). A transport change therefore **cannot change output**, only timing. Without that, a mid-run switch
would be unacceptable; with it, the transport is a scheduling decision like a cache policy. This must stay true:
no transport may change the bytes, the length, or the destination offset of a read (the r261 slice read already
fixed the per-arm length semantics).

## State and switch mechanics

```
transport ∈ { MAP, FILE, DIO }
g_pool_dio                    = (transport == DIO)
g_pool_{fill,demand}_src_file = (transport >= FILE)
```

The existing branches (`stage_span_fill`, the staging path, `pool_mirror_bulk_fill_locked`, `expert_host_src`,
`pool_fill_slot_locked`) already select on those globals, so a switch is one assignment — plus the part that
bites: **the `O_DIRECT` bit lives in the open flags** (`pool_open_file(path, dio)`), so a switch must close the
pool's fd and every `table_t::src_fd` first (`pool_transport_reopen_locked`). Everything runs under `g_mutex`,
and every host read is synchronous, so no read can straddle a switch.

## 1. The initial arm: a CAPACITY test, not a rate

`pool_available_ram_bytes()` becomes **cgroup-aware** (Linux): take the smaller of `/proc/meminfo`'s
`MemAvailable` and the process's own cgroup (`/proc/self/cgroup` → `memory.max − memory.current`). Then the
delivered test (`planned pool + host set > available`) finally means what it says inside a container or a
systemd scope:

* **fits** → `MAP` (the fast arm when warm: 17 GiB/s random against 7 for a buffered read)
* **tight** → `FILE` (the general-purpose baseline: it degrades gracefully, 2.6 → 1.7 GB/s, where the mapping
  falls 3.0 → 0.6 GB/s)
* the monitor may escalate from either of them (`>= FILE` → `DIO`).

### Why the arm is NOT chosen by a probe rate (measured, not assumed)

A cold probe is impossible in-process: `posix_fadvise(DONTNEED)` **skips pages the process has mapped** (the
probe then reported 32 452 MiB/s, i.e. straight out of the page cache), and `madvise(MADV_DONTNEED)` drops the
PTEs but not the page-cache pages. A *warm* probe measures the cache, not the transport. And even a honest cold
ranking is the wrong question: the mapping's advantage is the warm case, so a cold-rate ranking picks O_DIRECT on
every box, RAM-rich ones included. **Capacity decides MAP-vs-disk; runtime evidence decides everything after.**

## 2. The runtime monitor: escalation on evidence

Every host read reports `(bytes, µs)` — the pool fill (through a wrapper around `pool_fill_slot_locked`) and the
bounce/demand read (the dominant host cost on a tensor split: r261 measured 54 % of a decode step). Two triggers:

1. **A slow-read guard.** `N` consecutive reads (default 8) longer than `MOE_HOST_POOL_TR_SLOW_MS` (default
   40 ms ≈ 15 MiB/s for a 0.6 MiB expert) escalate immediately — the maintainer's "the fill took far too long"
   case, inside one ubatch.
2. **An epoch rate check.** Every `MOE_HOST_POOL_TR_EPOCH_MIB` (default 64 MiB) of reads, compare this arm's
   throughput with **the best rate this arm itself achieved in this run**. Below `MOE_HOST_POOL_TR_RATIO`
   (default 0.50) for **two consecutive epochs** → escalate.

The reference is *relative on purpose*: a slow box and a fast box both work, and a merely cold start never fires
it. Two consecutive epochs is not caution theatre — measured: a warm map arm dipped 16 444 → 9 624 MiB/s for one
epoch and a single-epoch rule switched a RAM-rich box off the mapping, which is the arm it wants.

Escalation is one way (`MAP → FILE → DIO`), with two epochs of warm-up after a switch, one INFO line per
transition (arm, observed rate, reference, trigger), and `DIO` as the terminal safe baseline (it does not depend
on the page cache at all, and on Linux it was the fastest arm under the hardest pressure).

**Owed: de-escalation.** A cache that frees up again (a job finishes) currently keeps the escalated arm. It needs
a much longer window (minutes) than escalation and it is what makes `auto` symmetric; it is not in `0002`.

## 3. Controls (pinning, never enabling)

| knob | class | meaning |
|---|---|---|
| `MOE_HOST_POOL_TRANSPORT=auto` | the proposed default (after gating) | capacity prior + runtime monitor |
| `MOE_HOST_POOL_TRANSPORT=map\|file\|dio` | diagnostic / A/B | pin one arm (no monitoring) |
| `MOE_HOST_POOL_DIO=0\|1` | legacy, kept | the delivered knob; still honoured when TRANSPORT is unset |
| `MOE_HOST_POOL_{FILL,DEMAND}_SRC` | campaign instrument (`0001`) | the static two-way split; not proposed |
| `MOE_HOST_POOL_TR_{EPOCH_MIB,RATIO,SLOW_MS,SLOW_N}` | tuning | thresholds above |
| `MOE_HOST_POOL_TR_START` | **campaign only** | start on a named arm (the only way to stage an escalation test) |

## 4. What the prototype established (2026-11-11, `0002`)

| check | result |
|---|---|
| text across `map` / `file` / `dio` / `auto` | **identical**, `144 chars sha=de81b28cab7d` |
| text across a run that SWITCHES | identical (the escalation run hashes the same) |
| 32G cgroup, `auto` | starts on **`file`** (`capacity prior: the set cannot be cached (cgroup-aware …)`) |
| plain run (91 GiB host), `auto` | starts on **`map`** and stays there |
| escalation | `map -> file (epoch rate 9252 MiB/s < 0.50x this arm's own 21185 MiB/s for 2 epochs)`, fds reopened, no fault |
| cold probe | **not possible in-process** (above) — so no rate-based selection |

## 5. Still owed before this ships

1. **De-escalation** (above) — otherwise `auto` is one-way.
2. **Threshold calibration from the interleaved A/B.** 0.50 and two epochs are *reasoned*, not tuned: the
   interleaved two-direction arm matrix (44G/32G, transport × span) is what should set them.
3. **A gate that proves the ladder is not vacuous**: a `gate-host-pool.sh` arm at a cache-starved cgroup that
   must log a transition, with N > 0 reads on each arm, plus the pinned arms for text identity. TODO #306's
   original second condition — "a cell that varies the transport and `STAGE_SPAN` together" — is still unmet and
   must stay: a `FILE` transport with `MOE_HOST_POOL_STAGE_SPAN=0` is the measured −41 % pair, so `auto` must
   refuse that combination (force the span on, or fall back to `MAP`).
4. **The decode ladder** (`benchmarks/mtp-adaptive-methodology.md`, `llama-batched-bench -npl 1,4,8`): this
   touches the decode path of every pool user.
5. **Windows.** The buffered arm there is `ReadFile`; the direct arm is `FILE_FLAG_NO_BUFFERING`. The reporter
   measured them within ~15 % (616-1 114 vs 638 MiB/s), so the ladder mostly does not matter there — but the
   WDDM/aperture behaviour (issue #278) means `DIO` must not be assumed safe until measured, and the Windows
   capacity test already uses `GlobalMemoryStatusEx` (no cgroup equivalent). `MOE_HOST_POOL_TRANSPORT` pins it.
6. **Attribution.** A benchmark that escalates mid-run is not a reproducible number. Every transition is logged
   for that reason; a measurement arm must pin the transport.
