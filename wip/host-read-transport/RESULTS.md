# `host-read-transport` — the raw numbers (2026-11-11)

Box: `soar`, 32 cores, 91 GiB RAM, ROCm 10.1.0, one R9700 gfx1201 visible (`HIP_VISIBLE_DEVICES=0`),
`/llm` NVMe. Instrument: `0001-per-expert-host-read-transport.patch` on the r261 tree, built at
`~/llama-scratch/build-rocm` (incremental, warm ccache). Read `README.md` first for what it means.

## 1. Text identity (`text-ab.sh`, llama-cli, greedy, no `-lv`)

`-ncmoe 48 --limit-host-experts 8 --host-experts pool -c 4096 -b/-ub 2048 --load-mode mmap
--lazy-mode on --spec-type none --seed 42 --temp 0 -n 24`, `wip/bench-56gb/prompt_short.txt`:

| arm | FILL_SRC | DEMAND_SRC | text |
|---|---|---|---|
| mapp | map | map | 144 chars `de81b28cab7d` |
| fillf | file | map | 144 chars `de81b28cab7d` |
| demf | map | file | 144 chars `de81b28cab7d` |
| bothf | file | file | 144 chars `de81b28cab7d` |

All four identical — the transport moves no value. Runs: `wip/host-read-transport/runs/text/*.log`.

## 2. Experiment 1 — `transport-ab.py` (real GGUF, random 640 KiB chunks)

`sudo systemd-run --scope -p MemoryMax=<X> -p MemorySwapMax=0 python3 transport-ab.py --work-gib <W>
--passes 2 --random`. Per arm: drop the page cache, read the working set once (COLD), read it again (WARM).
Order alternates per repeat. MiB/s:

| cgroup | working set | arm | COLD | WARM |
|---|---|---|---|---|
| 40G | 4 GiB | mmap | 3 040 | 17 102 |
| | | pread | 2 491 | 7 068 |
| | | O_DIRECT | 2 177 | 2 177 |
| 4G | 8 GiB | mmap | 1 050 | 1 132 |
| | | pread | 2 240 | 2 580 |
| | | O_DIRECT | 2 125 | 2 127 |
| 1G | 8 GiB | mmap | 593 | 596 |
| | | pread | 1 737 | 1 720 |
| | | O_DIRECT | 2 127 | 2 126 |
| 40G, **sequential** | 4 GiB | mmap | 6 764 | 39 818 |
| | | pread | 6 808 | 35 902 |
| | | O_DIRECT | 2 442 | 2 444 |

Reading: with the cache covering the working set, the mapping is the fastest transport (2.4x a buffered read on
the random shape). Without it, the mapping collapses (593 MiB/s at a 1 GiB budget, still falling with harder
pressure) while a buffered read holds 1.7-2.6 GB/s and O_DIRECT is flat at 2.1 GB/s. Sequential access is the
wrong test: fault-around makes the mapping tie a buffered read cold and win warm.

## 3. Experiment 2 — the 56 GiB rig (`wip/bench-56gb/parity-arm.sh`)

Field config: Flash-Next IQ3_S + its MTP draft, `-lme 12` (131/512 slots), `--load-mode mmap`,
`MOE_HOST_POOL_DIO=0` (forces the buffered arms), `MOE_HOST_POOL_RAM_PCT=0`, `HEATMAP=load`, page cache
dropped per arm, ballast 21 GiB outside the scope (MemAvailable ~57 GiB). `-c 204800 -b/-ub 6144`.

| run dir | cgroup | FILL_SRC | DEMAND_SRC | `pool fills` | bounce read | gen1 | gen2 | gen3 | pp50k #1 | pp50k #2 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `runs/mapp` | 56G | map | map | 2 160 MiB/s | 9 512 MiB/s | 35.42 | 55.13 | 59.09 | 846.9 | 874.1 |
| `runs/mapp44` | 44G | map | map | 1 599 | 4 656 | 21.57 | 43.20 | 55.84 | 694.0 | 655.7 |
| `runs/fillf44` | 44G | file | map | **2 299** | 5 973 | 28.75 | 45.10 | 55.13 | 693.7 | 738.3 |
| `runs/demf44` | 44G | map | file | 1 838 | 3 464 | 32.02 | 45.05 | 53.70 | 600.1 | 685.9 |
| `runs/bothf44` | 44G | file | file | 2 168 | 3 792 | 31.55 | 47.40 | 55.60 | 593.4 | 648.1 |
| `runs/mapp32` | 32G | map | map | **898** | **1 913** | **14.63** | **21.59** | **22.36** | 514.5 | 655.4 |
| `runs/bothf32` | 32G | file | file | **2 047** | **2 923** | **28.69** | **36.55** | **38.88** | 560.8 | 626.8 |

Storage read per stage (`/proc/<pid>/io read_bytes`), same runs, MiB:

| run | gen | pp50k #1 | pp50k #2 |
|---|---:|---:|---:|
| `mapp` (56G) | 18 884 | 87 122 | 89 124 |
| `mapp44` | 78 272 | 153 836 | 210 383 |
| `fillf44` | 54 698 | 181 451 | 157 593 |
| `demf44` | 28 993 | 177 253 | 142 082 |
| `bothf44` | 22 957 | 221 307 | 194 349 |
| `mapp32` | 246 225 | 234 116 | 202 369 |
| `bothf32` | 40 716 | 236 374 | 211 964 |

**Honest reading.** The 44G rows are NOT separable: the mapping arms measured 1 599-2 160 MiB/s for fills and
4 656-9 512 for the bounce read across repeats, i.e. a ±25 % band that is larger than the arm-to-arm delta, and
the file arms split (better fills, worse bounce). The **32G row is a clear sign**: with the set not cacheable,
the mapping collapses to 898 / 1 913 MiB/s and the file arms are ~2x better on every decode pass. A proper
promotion decision needs the interleaved two-direction A/B per shape (owed, README §"Owed").

## 4. The gate (`gate-host-read-transport.sh`) — 24/24 on `soar`, 2026-11-11

Seven TEXT arms (no `-lv`, hashed) and seven CENSUS arms (`-lv 4`, separate runs), so no assertion is read off
a log that the hashing arm produced. All seven texts: `144 chars sha=de81b28cab7d`.

**The pinned arms, measured on the warm 91 GiB host — this is the number that matters, and it corrects §1:**

| arm | pool fills | demand (bounce) |
|---|---:|---:|
| `ref` — mapping | 14 725 MiB/s | 15 197 MiB/s |
| `file` — buffered `pread` | **29 450 MiB/s** | **16 348 MiB/s** |
| `dio` — O_DIRECT | 2 461 MiB/s | 2 235 MiB/s |

**The buffered arm wins on BOTH paths on a warm box**, 2x on fills and 8 % on the demand read — it is not the
"cold-only" arm the campaign's microbenchmark implied. The microbenchmark (`transport-ab.py`) measured a
4 GiB working set *re-read* in a tight loop, i.e. with the mapping's page-table entries established; a model
run overwhelmingly touches each expert once per pass, where the mapping pays a PTE fault per 4 KiB page. That
distinction is the whole story of "mmap fills perform abysmally", including on an unpressured box.

What the gate asserts (each arm must prove itself, none of it is vacuous):

| assertion | result |
|---|---|
| all seven texts identical to `ref` | PASS |
| each pinned arm reports ITS transport (`map`/`file`/`dio`) | PASS |
| `ref` fills show `N mapping memcpys`, `file` fills show `N storage reads` | PASS (45 717 / 45 718) |
| `ref` demand path shows `per-expert mapping reads (bounce)`, `file` shows `storage reads (bounce)` | PASS (14 067 / 14 066) |
| `auto` converges on the arm the PINNED arms prove fastest (or stays put within 15 %) | PASS — file 15 705 vs map 15 008, ended on `file` |
| a transition must cite its evidence (`probe fill`/`stall`) | PASS |
| `auto` inside a 32G scope must latch (ladder is not dead code) | PASS — `map -> file`, 1 step |
| `buffered` + `MOE_HOST_POOL_STAGE_SPAN=0` must be REFUSED (the −41 % coupling) | PASS — falls back to `map` |
| `auto` + span off must never select `file` | PASS — ends on `map` |

The gate's first two runs (22/24, then 24/24) are the reason its convergence check is written this way: it
asserted "a RAM-rich box must stay on the mapping", which was an **assumption of mine**, and the pinned arms
disproved it on this hardware. The check now derives the expected arm from the same run's own pinned
measurements, so it is box-independent.

## 5. The interleaved (ABBA) A/B and the decode ladder, 2026-11-11

`abba-transport.sh <44G|32G>`: four arms in the order map / file / dio / auto, then auto / dio / file / map, so a
monotone drift cancels in the average. **The page cache is dropped before every arm** (the rig's hygiene rule —
without it the previous arm's warm cache hides the transport entirely, which is how the first cut of this matrix
measured 32G as if it were unpressured). Never parallel.

**First, a bug the matrix caught in the probe itself.** The probe's cadence was gated on
`g_pool_tr_reads % N`, and that counter is incremented by the observe path that a probe **returns before** — so
once a probe succeeded the counter stopped moving, `0 % N == 0` stayed true, and **every fill became a probe**
(measured: 42 746 probes serving 42 746 fills, 3 106 ms of host read time; and once the ladder had reached
`file` every probe was a `dio` read, which cost **67 ms/token against 39**). Fixed with a counter incremented on
the wrapper line itself (`g_pool_tr_fill_calls`); probes are now 1 in 128 and cost 15-66 ms per run. With that
fix `auto` is **neutral** when it does not latch.

### Shape A — unpressured (44G cgroup, warm cache, cache dropped per arm)

| arm | fills | bounce | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|
| map (a / b) | 27 544 / 27 530 | 16 020 / 16 275 | 1 336 / 1 350 | 25.95 / 26.19 |
| file (a / b) | 27 601 / 27 593 | 14 697 / 15 090 | 1 347 / 1 360 | 25.97 / 26.09 |
| dio (a / b) | 2 313 / 2 312 | 2 329 / 2 356 | 564 / 566 | 11.15 / 11.20 |
| **auto** (a / b) | 27 491 / 27 604 | 16 269 / 16 363 | 1 347 / 1 352 | 26.04 / 25.87 |

`auto` stays on `map` (332-333 probes, no latch) and is indistinguishable from the pinned arms. **No regression**
where the cache holds the set; `dio` is 2.3x worse, which is why it is terminal-only.

### Shape B — starving (32G cgroup + 21 GiB mlocked ballast outside it, cache dropped per arm)

| arm | fills | bounce | prefill t/s | decode t/s |
|---|---:|---:|---:|---:|
| map (a / b) | **1 628 / 1 537** | 2 287 / 2 328 | 487 / 497 | **6.25 / 5.49** |
| file (a / b) | 2 129 / 2 128 | 2 186 / 2 179 | 463 / 461 | **12.84 / 12.92** |
| dio (a / b) | 2 311 / 2 311 | 2 352 / 2 313 | 560 / 556 | 11.26 / 11.28 |
| auto (a / b) | 2 133 / 2 133 | 2 265 / — | 551 / 565 | 11.28 / 11.19 |

The mapping collapses (fills 1 537-1 628 MiB/s, decode **5.5-6.3 t/s**) and both disk arms are ~2x better. `auto`
reached `file` **on the stall detector** at 6.5 s (`stall: 4 consecutive reads over 4.0x this arm's own 498 us`),
then stepped to `dio` on a probe (`dio served the same read in 272 us against this arm's 456 us`).

### The knob question this exposed (NOT yet settled — recorded as owed)

On the read level `dio` won that probe 1.68x, yet **end-to-end `file` was 12 % better** (12.8 against 11.3 t/s):
the buffered arm populates the page cache that other paths then re-read, which no single-read latency can see. So
`TR_PROBE_GAIN=1.5` can step off a better arm, and the threshold needs to be measured, not reasoned. The
one-off `gain=1.5/2.0/3.0` runs that followed were **inconclusive because the box drifted** (the same pinned
`file` arm read 12.8-12.9 t/s in the interleaved matrix and 11.3-11.7 in the one-offs 10 minutes later), so the
threshold is left at 1.5 and the measurement is owed: interleaved, ballast held, cache dropped per arm, and
`file` preferred on a tie.

### The decode ladder — `llama-batched-bench -npl 1,4,8`

35B-A3B Q8_0 (fits in RAM: the cacheable case), `-ngl 99 -sm layer --n-cpu-moe 24 --limit-host-experts 8`,
`-npp 512 -ntg 128`, cache dropped per arm:

| arm | npl 1 | npl 4 | npl 8 |
|---|---:|---:|---:|
| map | **63.72** | **186.52** | 276.84 |
| file | 61.30 (**-3.8 %**) | 183.93 (-1.4 %) | 276.72 (-0.04 %) |
| auto | 62.21 | 185.24 | 276.00 (stays `map`) |

`auto` never latches on a cacheable set and is within noise of the pinned mapping arm at every width; a *pinned*
`file` costs ~4 % at `npl 1` and nothing at `npl 8`.

**Not run: the MTP four-axis protocol.** The change is host-side (no kernel, no numerics) and the four-axis gate
is the contract for decode-kernel/speculator changes; it stays owed before a promotion, together with the
interleaved threshold measurement above.

## 6. The MTP four-axis gate (2026-11-11) — PASS

Full record: [`benchmarks/2026-11-11-mtp-four-axis-host-read-transport.md`](../../benchmarks/2026-11-11-mtp-four-axis-host-read-transport.md).
Run on the MoE arm of the protocol (Qwen3.6-35B-A3B Q8_0, `-ncmoe 24 -lme 8`) because the canonical dense 27B has
no host experts and cannot see a transport change; `-n 3000`, the four axis prompts with their hashes recorded.

| axis | plain | n3 map | n3 file | acceptance (all three transports) |
|---|---:|---:|---:|---|
| R | 64.05 | 105.77 | 105.51 | 0.75118 |
| P | 62.81 | 107.21 | 106.97 | 0.75379 |
| C | 62.59 | 108.26 | 107.25 | 0.87798 |
| K | 61.31 | 129.22 | 129.30 | 0.98667 |

* acceptance **identical to five decimal places across transports on every axis**
* MTP beats plain everywhere (+65 / +71 / +73 / +112 %)
* purity: `none` == `draft-mtp n3` **byte-identical** across transports (prose `2de120b8aeec`, code `42c309c538dd`)
* adaptive n12: 85.7 / 89.6 / 92.3 / 144.1 t/s, acceptance 0.61-0.91; the single latch (recall -> `file`) landed
  on an arm that measures **146.21 (map) against 146.22 (file)**, so it picked no worse than the pin; the 1.5 %
  it cost is the transition, inside the protocol's noise band.

With that, every release-gate item for this change except a Windows-form measurement is run.
