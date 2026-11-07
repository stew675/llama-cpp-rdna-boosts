# `--lazy-mode mlock` -- page-cache residency for the PLE/engram tables, 2026-11-07 (r230)

Block 00, loader only (no kernel, no numerics). Built at the r230 tree
`b336da22e732804168fa0a2a444a5bded7ef123a` (working chain `dd1b3b784`, canonical tip `2028e44cd`).
Box: 3 x R9700 (gfx1201), ROCm 10.1.0, 91 GiB RAM, `ulimit -l` 80 GiB, THP `madvise`, XFS.

Model: `Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S` (77.9 GiB, A3B).  Its `per_layer_token_embd` is
IQ4_NL `160 x 320001536` = 27,465 MiB, and it is the **entire** second shard
(`...-00002-of-00002.gguf`, 28,800,138,240 B at offset 192) -- so "lock the file" and "lock the PLE"
are the same call for this model.

## The three arms

| arm | data path | residency |
|---|---|---|
| `auto` / `on` | rows read from the mapping on demand | none (evictable page cache) |
| `off` | whole table read into a private host buffer (`read_raw`) | 27 GiB anonymous, swappable, per process |
| `mlock` | rows read from the mapping on demand | **populated + `mlock`ed at load**, per process |

`mlock` is deliberately not recorded in the lazy `ranges` set: that set gets `POSIX_MADV_RANDOM`, and
readahead is what the one bulk populate needs.  There is no 4 GiB `auto` threshold -- the mode pins
every marked table.

## Residency (the point of the mode)

`llama-server`, `-ngl all -sm tensor --load-mode mmap --lazy-mode mlock` (Flash-Next, 3 GPUs):

```
add: tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled, to be pinned in the page cache (--lazy-mode mlock)
mlock_ranges: pinned 1 table(s), 27465 MiB in the page cache (--lazy-mode mlock); the pin is released when the process exits, ...

VmLck:     28125136 kB     <- 26.82 GiB = the whole 27,465 MiB table
RssAnon:    1337464 kB     <- 1.3 GB: no copy of the table
RssFile:   28414708 kB
/proc/meminfo: Unevictable == Mlocked == 28125136 kB
```

At exit (`SIGTERM`, 2 s): `Mlocked`/`Unevictable` -> 0 while `Cached` stays ~52-83 GB, i.e. the lock is
gone and the pages are ordinary reclaimable page cache.  `off` for comparison: `RssAnon` 32.2 GB,
nothing reusable after exit.

## Load

| arm (`--load-mode mmap`) | wall (cli, `-n 24`) | maxrss |
|---|---:|---:|
| `mlock` | 32.5 s | 32.2 GB (RssFile 28.4 GB) |
| `auto` | 29.1 s | 5.0 GB |
| `off` | 25.7 s | 32.2 GB |

The ~7 s over `off` is `mlock`'s populate (`__mm_populate`, ~4.2 GB/s); `off` pays the same order for
its `read_raw` + copy.  With the shard already in page cache the pin is a page-table walk: **0.19 s** in
the smoke run (log timestamps 1.119 s -> 1.309 s), which is where the warm-restart number comes from.

Maintainer's run (2026-11-07, `--load-mode none --lazy-mode mlock`, PLE warm in page cache):
**ready to serve in 8.26 s**.

## Throughput -- parity, as expected

`llama-bench -p 2048 -n 32 -r 2 -t 7`, Flash-Next 3 GPU `-sm tensor`:

| arm | pp2048 | tg32 |
|---|---:|---:|
| `mlock` | 2570.40 ± 18.69 | 77.41 ± 3.19 |
| `auto` | 2549.11 ± 19.00 | 77.55 ± 3.22 |
| `off` | 2574.82 ± 29.87 | 76.27 ± 2.00 |

The gather path and the bytes are identical; only the residency mechanism differs.

Same-seed greedy text, `-p "The capital of France is" -n 24 --seed 42 --temp 0`: `125 chars
sha=7d898d3d3e58` for `mlock`, `auto` and `off` (and for `--load-mode none --lazy-mode mlock`).

## The maintainer's long-prefill run (headline)

Flash-Next GSQ-RCO-IQ3_S, 3 GPU, `--load-mode none --lazy-mode mlock`, 70K-token prefill:
**~4500 t/s at the start and ~3000 t/s at the 70K mark** -- both are speed records for this box --
with generation ~10 % above the usual figure (not attributable: MTP/draft variance and prompt shape
move that number on their own).

## Limits

- Needs `RLIMIT_MEMLOCK` >= the table.  Refused (tested at `ulimit -l 2048`): one WARN, the table stays
  on the evictable page-cache path, text unchanged.  There is no partial pin.
- 27 GiB is unevictable while the process runs.  Fine on a 91 GiB box; on a unified-memory APU it is
  taken from the same pool as "VRAM".
- `mlock` is not DMA pinning: the PLE gather is on the host (`host_gather`), so nothing here depends on
  it.  A pinned-host (`ROCm_Host`) placement would be a different trade.
- Windows uses `VirtualLock` (untested here).

## Reproduce

```bash
M=/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
HIP_VISIBLE_DEVICES=0,1,2 ./build-rocm/bin/llama-server -m $M -ngl all -sm tensor \
  --load-mode mmap --lazy-mode mlock -lv 4 --ctx-size 8192 --no-kv-unified
grep -E '^(VmLck|RssAnon):' /proc/$(pgrep -f 'no-kv-unified')/status
# greedy text: swap --lazy-mode for auto / off and compare scripts/extract-generated.py output
```
