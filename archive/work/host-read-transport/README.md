# `host-read-transport` — which transport serves a per-expert host read (TODO #306, widened)

**Status: OPEN — measurement campaign, 2026-11-11. The instrument is built and both experiments are run;
nothing here is in `patches/`, nothing is applied to the fork, no release gate is owed yet.**

## The question

TODO #306 asks whether the *pool fill* should read the FILE (`pread` through the page cache) instead of one
`memcpy` out of the read-only model mapping. The maintainer's direction (2026-11-11) widens it:

> for weights that spill onto disk, we presently prevent anything but mmap from being used as the fill
> mechanism … we really need to be using buffered reads … the mmap fill really does perform abysmally
> (< 100 MB/s) once the page cache is under pressure from high system memory usage.

At r261 that is literally true, and it is an inconsistency we shipped ourselves: the **coalesced prefill span**
already reads the file in both modes (r232), while the two **per-expert** reads take the file arm *only* under
`MOE_HOST_POOL_DIO=1`:

| host read path | buffered (mapped) load today | O_DIRECT |
|---|---|---|
| prefill span (`stage_span_fill`) | buffered `pread` | O_DIRECT `pread` |
| prefill staging (bulk) | buffered `pread` | O_DIRECT `pread` |
| pool fill (`pool_fill_slot_locked`) | `memcpy` out of the mapping | O_DIRECT `pread` |
| demand / bounce (`expert_host_src`) | `memcpy` out of the mapping | O_DIRECT `pread` |

The pool **master** must stay mapped on a box where the set spills (the loader refuses a pool without a mapped
load, `src/llama-model-loader.cpp:1022`) — this campaign is about the *transport the reader uses*, not where
the weights live.

## The instrument — `0001-per-expert-host-read-transport.patch`

Cut against the r261 tree (`030c8997d`); **inert by default** (both knobs unset = the delivered mapping arm):

| knob | effect |
|---|---|
| `MOE_HOST_POOL_FILL_SRC=file` | the pool **fill** reads the file (buffered `pread`) |
| `MOE_HOST_POOL_DEMAND_SRC=file` | the **bounce/demand** read reads the file (buffered `pread`) |

It also adds a `per-expert mapping reads (bounce)` census line, because a map arm has no bounce counter at all
today — without it an A/B cannot prove which transport an arm actually took. And it WARNs when a file
transport is combined with `MOE_HOST_POOL_STAGE_SPAN=0` (the measured −41 % pair).

Apply it to a warm tree with `scripts/scratch-tree.sh arm wip/host-read-transport/0001-*.patch`
(built here at `~/llama-scratch`, `build-rocm`).

## Experiment 1 — the transport in isolation (`transport-ab.py`)

Real GGUF, random 640 KiB chunks (the routing-driven shape; a *sequential* test cannot reproduce the collapse —
the mapping gets fault-around/readahead for free and ties a buffered read), reader inside a memory cgroup so the
page cache it may keep is a knob. MiB/s:

| cgroup | working set holds the cache? | mmap | buffered `pread` | O_DIRECT | pread/mmap |
|---|---|---|---|---|---|
| 40G | yes (4 GiB) | 3 040 cold / 17 102 warm | 2 491 / 7 068 | 2 177 / 2 177 | 0.82x / 0.41x |
| 4G | no (8 GiB) | 1 050 / 1 132 | 2 240 / 2 580 | 2 125 / 2 127 | **2.13x / 2.28x** |
| 1G | no (8 GiB) | **593 / 596** | 1 737 / 1 720 | 2 127 / 2 126 | **2.93x / 2.89x** |

Sequential control (40G, 4 GiB): cold mmap 6 764 vs pread 6 808 (a tie), warm 39 818 vs 35 902.

**This is a crossover, not a verdict:** the mapping is the *fastest* transport while the cache holds the set,
and the *worst* once it cannot (593 MiB/s at a 1 GiB cache budget — the sub-100-MB/s shape the maintainer
describes is the same mechanism, harder). A buffered read degrades gracefully (2.6 → 1.7 GB/s); O_DIRECT is
flat (2.1 GB/s) and is the best arm under the hardest pressure.

## Experiment 2 — end to end on the 56 GiB rig (`wip/bench-56gb`)

Field config (Flash-Next IQ3_S, `-ncmoe 48`, `-lme 12` = 131/512 slots, `--load-mode mmap`,
`MOE_HOST_POOL_DIO=0` to force the buffered arms, page cache dropped per arm), `parity-arm.sh`; the cgroup size
is the pressure knob.

| arm | fills | bounce read | gen1/2/3 t/s | pp50k #1/#2 |
|---|---|---|---|---|
| **56G** map/map | 2 160 MiB/s | 9 512 MiB/s | 35.4 / 55.1 / 59.1 | 846.9 / 874.1 |
| 44G map/map | 1 599 | 4 656 | 21.6 / 43.2 / 55.8 | 694.0 / 655.7 |
| 44G file/map | **2 299** | 5 973 | 28.8 / 45.1 / 55.1 | 693.7 / 738.3 |
| 44G file/file | 2 168 | 3 792 | 31.6 / 47.4 / 55.6 | 593.4 / 648.1 |
| **32G** map/map | **898** | **1 913** | **14.6 / 21.6 / 22.4** | 514.5 / 655.4 |
| **32G** file/file | **2 047** | **2 923** | **28.7 / 36.6 / 38.9** | 560.8 / 626.8 |

* At **44G** (the set is partly cacheable) the transport signal is inside the ±25 % run-to-run band and the two
  arms split: file is better for the fill, the mapping is better for the bounce read. Do not read a winner here.
* At **32G** (the set is NOT cacheable) the mapping collapses exactly as the microbenchmark predicts —
  898 MiB/s fills, 1 913 MiB/s bounce — and **file/file is +96 / +69 / +74 % on the three decode passes**,
  +9 / −4 % on prefill (the span already reads the file in both arms, so prefill barely moves).
* Text is byte-identical in all four arms: `de81b28cab7d`, 144 chars, `wip/host-read-transport/text-ab.sh`
  (`llama-cli`, greedy, no `-lv`; the transport moves bytes, never values).

## What this settles, and what it changes

1. **The maintainer's observation reproduces and is quantified.** A mapping read collapses to ~0.6-1.1 GB/s
   (and lower with more reclaim) where a buffered read holds 1.7-2.6 GB/s. End to end that was ~2x the decode
   rate at 32G.
2. **It is a POLICY, not a preferred transport.** The mapping is 2.4x faster than a buffered read when the cache
   holds the set (17.1 vs 7.1 GiB/s random warm), so a blanket switch to `pread` regresses the RAM-rich box.
   The rule the data supports is a three-arm choice: cache holds the set → mapping; it does not → buffered
   `pread` (or O_DIRECT, which is best where the disk tier is available).
3. **The delivery's weak link is the FIT TEST, not the transport.** `pool_decide_dio_locked` already picks
   between "buffered" and "O_DIRECT" from `MemAvailable` vs pool+set — and `MemAvailable` is **cgroup-unaware**
   and taken **once**, at the first pool build (see `wip/host-expert-coverage/`). When it says "fits" but the
   cache cannot hold the set (a container, another process growing later, Windows/WDDM, the PLE table also
   competing), the buffered arm is the mapping — which is exactly the collapse. That is the bug behind "mmap
   fills perform abysmally once the page cache is under pressure": the box was silently in the wrong arm.
4. **So the promotion is not "flip the fill to file".** It is: make the *buffered* arm's transport a buffered
   read (not the mapping) **when the fit test cannot be trusted**, keep the mapping where the cache provably
   holds the set, keep O_DIRECT as the tight arm, and make the whole thing one decision (the coupling with
   `MOE_HOST_POOL_STAGE_SPAN` stays mandatory — a file transport without the span is the measured −41 %).

## The adaptive arm (`0002`, `DESIGN.md`)

`--load-mode` now names the transport as well as the master: `mmap` / `buffered` (**new**) / `dio` **lock** the
arm, and `auto` (the default) **starts optimistically on the mapping** and falls down the ladder on evidence.
Two triggers, both dimensionless — **no MiB/s and no millisecond anywhere**:

* **stall**: a read over `TR_STALL_K` (4x) this arm's own running mean, `TR_STALL_N` times running;
* **probe-fill A/B**: every `TR_PROBE_FILLS` (128) fills, the **next** arm serves that fill instead. Same bytes,
  so it costs no extra I/O — it is a fill that had to happen — and it compares two latencies for the same bytes
  in the same cache state at the same moment. Three consecutive wins by 1.5x latch the next step.

**One way is the decision**, not a gap: a box that fell short once is likely to again, and rebuilding the page
cache may itself be what shortens it, so a slower consistent point beats a flapping optimum. A reverse direction
is a possible later opt-in. The cgroup-aware `pool_available_ram_bytes()` stays as the root-cause fix for the
pinned-pool RAM ceiling, not as a transport selector.

Measured: text `de81b28cab7d` for `mmap` / `buffered` / `auto` and the static four-way instrument; `buffered` logs
`-> file (pinned by --load-mode)`, `mmap` logs `-> map`, `auto` logs `-> map (optimistic start)`; inside a 32G
scope `auto` latches with `map -> file (probe fill: file served the same read in 33 us against this arm's 50 us,
3 samples running)`.

**On this box the buffered arm wins BOTH paths even unpressured** (fills 29 450 vs 14 725 MiB/s, demand 16 348 vs
15 197), so `auto` converging on `file` is correct here — the mapping's edge is limited to re-reads with
established page-table entries, which a model pass does not do. The gate grades `auto` against the pinned arms'
own measurement of the same run rather than against an assumption (`RESULTS.md` §4).

## Owed before anything is promoted

* **A two-direction A/B per shape, interleaved**, on the 56G/44G/32G ladder (this campaign ran one arm per
  shape; the ±25 % band at 44G means only the 32G result is a clear sign today).
* A `gate-host-pool.sh` cell that varies the transport and the span **together** (TODO #306's original second
  condition, still unmet), plus the decode ladder (`benchmarks/mtp-adaptive-methodology.md`,
  `llama-batched-bench -npl 1,4,8`) because this touches the decode path for every pool user.
* The fit-test fix itself (cgroup-aware, re-evaluated) is `wip/host-expert-coverage/`'s item, not this one.
* Windows: the buffered arm there is the `ReadFile` path (`pool_open_file(path, false)`), already delivered for
  the O_DIRECT arm's shape — but it has not been measured on a Windows box in this form.

## Files

| file | what |
|---|---|
| `0001-per-expert-host-read-transport.patch` | the static two-way instrument (inert by default) |
| `0002-adaptive-host-read-transport.patch` | the **adaptive** layer on top of `0001`: cgroup-aware capacity prior + monitored ladder, `MOE_HOST_POOL_TRANSPORT=auto` |
| `DESIGN.md` | the adaptive design, what the prototype proved, and what it still owes |
| `test-adaptive.sh` | the prototype's checks (`text`, `escalate`, `slowguard`) |
| `gate-host-read-transport.sh` | **the gate**: 7 text arms + 7 census arms, 24 assertions (see `RESULTS.md` §4) |
| `transport-ab.py` | experiment 1: the transport in isolation, under a cgroup-shaped page-cache budget |
| `text-ab.sh` | the four-arm text-identity check (greedy, hashed with `scripts/extract-generated.py`) |
| `RESULTS.md` | the raw tables and the run directory names under `wip/bench-56gb/runs/` |

## Close-out / handover (2026-11-11) — DONE AS WIP, awaiting a promote-or-park decision

**Where it stands.** Everything below is proven on `soar`; nothing is in `patches/` and nothing is applied to the
fork. `0001` (static instrument, inert by default) + `0002` (adaptive transport + `--load-mode buffered`) apply
strictly to the r261 tip. Gate **24/24**. Greedy text identical on every arm (`de81b28cab7d`).

**What the data says, in one paragraph.** On an unpressured box the three transports are *equal* end to end
(decode 25.9-26.2 t/s; `dio` 11.2), so `auto` staying on the mapping costs nothing. Under starvation the mapping
collapses (fills 1.5-1.6 GiB/s, decode **5.5-6.3 t/s**) while a buffered read holds **12.8-12.9 t/s** and
O_DIRECT 11.3 — that is the maintainer's `<100 MB/s` complaint, reproduced and quantified. On width
(`batched-bench npl 1/4/8`) a pinned `file` costs ~4 % at npl 1 and nothing at npl 8, and `auto` is neutral
because it never latches there. So the change is **a win exactly where the page cache cannot hold the set and
neutral everywhere else**, which is the shape a default-on change should have.

**If you promote it (my recommendation):**
1. Fold `0001` + `0002` into block 12 as one amendment (the fill/demand transport + `--load-mode buffered` +
   the cgroup-aware ceiling), default `auto`.
2. `ENVIRONMENT.md`: the six `MOE_HOST_POOL_TR_*` knobs and the load-mode table; `patches/README.md`: a block-12
   note; `MANIFESTS.md`: `gate-host-read-transport.sh` into the per-block table.
3. Move `gate-host-read-transport.sh` to `scripts/` (it is written against `wip/bench-56gb/ballast` for its
   starving arms, so that path needs a home or the gate needs an in-gate ballast).
4. Gates still owed by the contract: the **MTP four-axis** protocol and a **Windows-form** measurement. The
   four-axis is the one I would not skip silently; it is ~30 min of runs on the 27B/35B and does not need new
   hardware.
5. Then the usual: `make-patches.sh`, `make-release.sh`, `validate-set.sh`, `fork-refresh.sh --build --gates`,
   WORKLOG entry.

**If you park it:** the campaign is self-contained — instrument, gate, both A/Bs, the mechanism record and the one
bug the matrix caught. Nothing rots except the `optional/host-pool-fill-src/` artifact it supersedes (re-cut at
r243, still applies, stale output tree recorded).

**Not worth chasing now.** The probe's `TR_PROBE_GAIN` (1.5) / `TR_PROBE_GAIN_DIO` (3.0) are *reasoned*: the
one-off runs meant to tune them were inconclusive because the box drifted (the same pinned arm read 12.8 t/s in
the interleaved matrix and 11.3 ten minutes later). They are env knobs, they are documented, and the gate grades
the outcome (decode within 5 % of the best pinned arm) rather than the thresholds — so this is tuning, not risk.
