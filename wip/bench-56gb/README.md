# wip/bench-56gb - the small-box rig and its run matrix

**Status: OPEN - the rig works and produced every number in this `wip/` tree (2026-11-05).  The matrix is
incomplete (see "runs we still owe").  Nothing here is in `patches/` and this is NOT a release gate.**

**What this is.**  The machine that has to reproduce `briansp2020`'s 56 GB-container numbers has 93 GiB of RAM;
his box has 59 GiB.  A cgroup alone is not enough, because every sizing decision in the delivery reads
`/proc/meminfo`, which inside a memory cgroup reports the HOST.  So the rig constrains both:

1. the server runs in a systemd scope with `MemoryMax=56G` and `MemorySwapMax=0` (the container shape);
2. a ballast process OUTSIDE the scope allocates and MLOCKs a chosen amount of RAM, which moves `MemAvailable`
   to the value the target box would report.  31 GiB of ballast puts this host at 57.3 GiB, i.e. a 59 GiB box.

Part 2 is also a standing demonstration of the defect `host-expert-coverage/` exists to fix: once the loader reads
its own cgroup, the ballast stops being needed for sizing and stays only as a memory-pressure control.

## Files

| file | what it does |
|---|---|
| `run.sh` | starts a launch script inside scope `<tag>`, logs to `runs/<tag>/server.log`, samples `memory.current` into `runs/<tag>/mem.txt` |
| `measure.sh` | health, three 600-token decode passes, one 50k prefill, one 179.5k prefill, then `memory.peak` and the cgroup anon/file split |
| `bench.py` | `/completion` client that prints the server's own `timings` (prompt and predicted t/s), plus a bounded `health` poll |
| `gen_prompt.py` | writes `prompt_short.txt` and `prompt_long.txt` (179.5k tokens on this model).  `prompt_50k.txt` (used by `measure.sh`) is its first ~250 000 characters, cut at a line boundary: `awk 'BEGIN{n=0} {if(n>=250000) exit; print; n+=length($0)+1}' prompt_long.txt > prompt_50k.txt` |
| `runme-56gb-tier.sh` | the parameterised launch script: `[LME] [SINK] [DIO] [SPAN] [RANK] [HEATMAP] [SERVER]` as **positional arguments** (see the trap below) |
| `parity-arm.sh` | one arm end to end: drop the page cache, start the scope, wait healthily (bounded), three 600-token decode passes, **two** 50k prefill passes, `/proc/<pid>/io` storage deltas per stage, the census, stop.  Writes `runs/<tag>/{parity.txt,server.log,mem.txt}` |
| `parity-matrix.sh` | an arm list (`TAG LME SINK DIO SPAN RANK SERVER` per line), one arm at a time, never in parallel |
| `parity-summary.py` | the whole matrix as one markdown table (`gen3`, `pp50k#1/#2`, the **transition tail**, MiB/request, fills, demoted, reuse) |
| `pp-analysis.py` | one log: per-pass processing vs `prompt eval time`, and the events (rearm/evict/rerank) that fill the gap |
| `ballast.c` | allocates and `mlock`s N GiB of anonymous memory, then sleeps; `gcc -O2 -o ballast ballast.c` |
| `runme-56gb-mirror.sh` | the host-expert config the campaigns compare against (`-lme 47`, `RAM_PCT=0`, KV pool off) |
| `RESULTS.md` | the measured matrix |

`SERVER` in `runme-56gb-mirror.sh` points at the scratch checkout's build
(`$HOME/llama.cpp/build-rocm/bin/llama-server`), because this repo is the delivery set and not the tree you
build.  Override it with `SERVER=` for any other build.

## Procedure

```sh
gcc -O2 -o ballast ballast.c
python3 gen_prompt.py

./ballast 31 &                          # outside the scope: MemAvailable -> about 57 GiB
grep MemAvailable /proc/meminfo

# before any start-time comparison, drop the page cache (INVARIANTS.md, r215 item 8)
echo 3 | sudo tee /proc/sys/vm/drop_caches

./run.sh H ./runme-56gb-mirror.sh
python3 bench.py 8033 health            # returns as soon as the server answers, bounded
./measure.sh H long
timeout 20 sudo systemctl stop bench-H.scope
kill %1                                 # the ballast
```

No blind `sleep`s: `bench.py <port> health` polls and returns on the first healthy answer, `measure.sh` prints
each row as it lands, and every blocking command above is wrapped in `timeout`.  A full row is about 4 minutes on
a warm page cache (13 to 30 s of load plus pool fill, 3 x 8 s decode, 26 s for 50k, 100 s for 179.5k).

## Invariants

* one model load per run: the pool, the arena and the KV pool are all decided at load;
* `-lv 4` always, otherwise the loader's placement lines are INFO and invisible and the tier is unknown;
* greedy output compared across configs; a transport change must be bit-identical, only a placement change may
  differ in t/s;
* keep the order fixed (decode, 50k, 179.5k): the 179.5k is a SECOND pass, so a non-mirror tier gets its
  remainder page-cached by then and the 50k row is the honest first-pass number;
* record from `runs/<tag>/server.log`: `plan_host_expert_placement`, `pool_clamp_slots_to_ram`,
  `host pool ... -> n/N slots`, `alloc_all_locked` (arena size), `SLAB_INIT`, the KV pool stats line;
* `LLAMA_KV_POOL_STATS` and `/proc/<pid>/io` (needs root, the server runs in the scope) are the two ways to show
  that a tier is doing its job; a mirror does 0 MiB per pass.

## Runs we still owe

* `-sm tensor` rows for the mirror config (`archive/work/host-expert-mirror/` step 6, REFUTED -- that campaign closed 2026-11-05);
* the KV pool high-hit-rate rows at 32k and 64k (`kv-residency-cleanup/` §2.4, and the A17 numbers it already
  has at 5 246 and 102 800 tokens);
* the split sweep (pool, arena) = (47, auto), (44, auto), (40, 26), (34, 32) (`host-expert-split/` step 3);
* `MOE_HOST_POOL_H2D=0|1` and `MOE_HOST_POOL_ALLOC=thp|portable` rows (`archive/work/host-expert-mirror/` step 4, CLOSED -- the alloc shape does not move any transport);
* a 93 GiB control row (no cgroup, no ballast) so the effect of the emulation itself is measured once;
* a cold-page-cache arm for the mirror row, since every row above was warm.

## Traps (measured, both cost a pass)

* **`sudo` resets the environment**, and `run.sh` starts the launch script inside `sudo systemd-run --scope`:
  an exported `LME=44 SINK=0 ./run.sh ...` reaches **nothing** and silently runs the defaults (which looks
  plausible).  That is why the knobs are positional arguments, why `parity-arm.sh` prints
  `applied tier: slots=N/512` read back from the server's own placement log, and why `SERVER` is argument 7.
* **Never hash llama-cli text taken with `-lv 4`**: its statistics lines interleave into the generated text, so
  identical output hashes differently.  `-lv 4` is for the census, not for `scripts/extract-generated.py`.
* The page cache decides a non-mirror tier's first prefill (the band-transition tail): quote the pair and
  never one rep -- the same arm moved 1 810 -> 1 329 t/s between reps in one session
  (`archive/work/host-expert-parity/RESULTS-2026-11-07-radix.md` §3).
* `ballast` takes a **fractional GiB** (`./ballast 32.4`): a whole-GiB step is ~1 GiB of `MemAvailable`, and the
  emulation needs 57.2 GiB (a 59 GiB box), not 55.3.

## Caveats

* swap is enabled on this host (157 GiB).  Anything meant to be non-reclaimable must be `mlock`ed or it is not an
  emulation of a pinned byte.
* page cache is shared between the scope and the host, so a warm cache flatters the first pass of a non-mirror
  tier.  Drop it when the number matters, and say in the row whether it was cold.
* the rig emulates a small box, not a slow one: the NVMe here is faster than most, so absolute prefill numbers
  are optimistic; the tier comparison is what it is for.
