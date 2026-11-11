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
