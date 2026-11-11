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
| `0001-per-expert-host-read-transport.patch` | the instrument (inert by default) |
| `transport-ab.py` | experiment 1: the transport in isolation, under a cgroup-shaped page-cache budget |
| `text-ab.sh` | the four-arm text-identity check (greedy, hashed with `scripts/extract-generated.py`) |
| `RESULTS.md` | the raw tables and the run directory names under `wip/bench-56gb/runs/` |
