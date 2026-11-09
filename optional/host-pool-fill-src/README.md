# `host-pool-fill-src` — the host pool's buffered fill reads the file, not the model mapping

**Optional artifact, not part of the delivery.  Re-cut on `r243`** (against `release.json` tree
`e9300f1dbd0b2f0b96c1828829d2072d73f34bca`; first cut on `r241`).

The second half of **issue [#281](https://github.com/stew675/llama-cpp-rdna-boosts/issues/281)** (@DanoPTT,
Windows 11, one R9700, 64 GB, Flash-Next IQ3_XXS, `--limit-host-experts 4 MOE_HOST_POOL_RAM_PCT=10`).
The first half shipped in **r232** as a block-12 amendment (`A`: the coalesced prefill read
`MOE_HOST_POOL_STAGE_SPAN` was gated on `MOE_HOST_POOL_DIO`, so the field default — `--load-mode mmap`,
buffered — could not coalesce at all).  This is `B`: **the pool fill reads the file too.**

* `host-pool-fill-src.patch` — the artifact.  sha256
  `eb1bee8257ae3e3e2abfd2c196c2dbdb697f6c0bb0406f6f51a85f008af036be`
* applies at `release.json` tree `e9300f1dbd…`, produces tree `faa85ced24786bcd780022585b28c86a3e531a18`
  (verified by strict `git am` on a checkout of `.tip` `72f7753fd`; both artifacts together produce
  `74458b8bfd25575d86a31d69c21a16a3a5e837b7`)
* one file, 25 insertions / 8 deletions: `ggml/src/ggml-cuda/moe-expert-cache.cu`
* ```bash
  bash scripts/apply-all.sh <clean-llama.cpp-at-release.json.base>   # then accept the prompt for this one
  ```

## What it does

`stage_span_fill()` (r232) already reads a *run* of "neither tier" experts from the file.  The **fill** of an
individual pool slot still `memcpy`s out of the read-only model mapping, which is the same page-cache pages but
a different transport:

| | buffered fill (current) | file fill (this artifact) |
|---|---|---|
| syscall | none | one `pread` per expert (~0.65–0.7 MiB) |
| page-cache source | the mapping | the same pages, via `read()` |
| Windows | a **demand fault per 4 KiB page** — measured **715 MiB/s** on the reporter's box | the OS readahead |
| Linux, warm | 2 416–2 725 MiB/s (a cache `memcpy`) | 2 400–2 900 MiB/s (a cache `read`) |

`MOE_HOST_POOL_FILL_SRC=map` restores the pre-r232 mapping memcpy (the A/B arm); unset means the pool's own
file handle with a plain buffered `pread` (an O_DIRECT-mode fill is unchanged — it already read the file).

## What it measured (carried from the campaign, `r231 + A`, not re-measured at r241)

56 GiB cgroup rig (`wip/bench-56gb/`), one R9700, Flash-Next GSQ-RCO-IQ3_S + its MTP draft,
`-c 204800 -ub 6144 --load-mode mmap`, page cache dropped per arm, 50k prefill ×2, cold:

| arm | fill source | pp50k #1 | pp50k #2 | storage reads, pass 1 |
|---|---|---:|---:|---:|
| `A` only (r232, the shipping default) | mapping memcpy | 821.9 | 874.6 | 95 GiB |
| **`A` + this artifact** | buffered `pread` | **1138.1** | **1146.1** | **23.8 GiB** |
| this artifact with the span off | buffered `pread` | 313.7 | — | 71.8 GiB |

**+38 % on top of the shipped default at 26 % coverage, 4x fewer storage reads** — and **−41 % on its own**,
because a 0.7 MiB `pread` per fill is latency-bound at depth 1 where a mapping `memcpy` gets fault-around
prefetch.  Once the pages *are* cacheable it is exactly neutral: RAM-rich cell (35B-A3B UD-Q4_K_M,
`-ncmoe 40 -lme 4`, no cgroup, 91 GiB), mapping fills 8 893 MiB/s against file fills 8 984 MiB/s, prompt
4765 / 4733 / 4745 / 4740 t/s across the four arms.

So it is a **page-cache headroom bet, not a warming trick**: a buffered `pread` populates pages but does not pin
them, so unlike `--lazy-mode mlock` (r230) the "warm" does not survive cgroup pressure.  It pays exactly where
the span pays (a sub-complement pool, i.e. the reporter's 64 GB / 70 GB `l4` shape) and nowhere else.

> **Keep the coalesced prefill read on.**  The −41 % standalone arm is reachable with
> `MOE_HOST_POOL_STAGE_SPAN=0`, which is a delivery kill switch: with this artifact applied, that switch now
> drops you into the slow fill.  That coupling is the reason this is an artifact and not a block.

## Which layouts it reaches

| layout | effect |
|---|---|
| `--host-experts pool`, buffered (`MOE_HOST_POOL_DIO` unset/auto and `auto` resolving buffered, i.e. `--load-mode mmap`/`none`) | **the target**: the fill changes transport |
| `--host-experts pool` with the disk tier on (O_DIRECT: `MOE_HOST_POOL_DIO=1`, or `auto` in the memory-tight band) | **inert** — an O_DIRECT fill already reads the file |
| `--host-experts cache` / arena (`-ncmoe` without the pool) | **inert** — no pool fill runs |
| a platform whose pool file I/O is a stub (the disk tier unavailable) | **inert** — the fill falls back to the mapping and warns once, unchanged (issue #274) |

### r241 spot check (this cut)

One R9700 gfx1201, 35B-A3B Q8_0, `-sm layer -ncmoe 24 --host-experts pool -lme 4` (a near-complement pool,
so most experts are faulted in), greedy 20 tokens, the two arms interleaved:

| arm | `pool fills:` | text |
|---|---|---|
| unset (file) | **10 595 storage reads** 659 ms, **0** mapping memcpys, 10.99 GiB (11 257 MiB/s) | `627b2ebd7792` |
| `MOE_HOST_POOL_FILL_SRC=map` | 0 storage reads, **10 589 mapping memcpys** 705 ms, 10.99 GiB (11 251 MiB/s) | `627b2ebd7792` |

So the switch does exactly what it says (the fill really moves between the two transports), the text is
byte-identical, and on this warm Linux page cache the throughput is identical too — which is the "exactly
neutral when the pages are cacheable" row above, measured on the delivery's own hardware.  The win needs the
cold/sub-complement case, i.e. the reporter's box.

## Correctness

The two arms read the same bytes from the same page-cache pages, so the slot contents are identical by
construction, and `MOE_HOST_POOL_VERIFY=1` (the pool's own pool-read mismatch counter) is the instrument that
proves it.  The campaign's pool arms were text-identical between `A` and `A+B`; the strict `git am` apply above
is the same tree the delivery's `moe-expert-cache.cu` builds from with only the fill source changed.

## What it does NOT do

* It does not make buffering beat O_DIRECT in the memory-tight band (there the page cache is the thing you do
  not have); it narrows the case for `MOE_HOST_POOL_DIO=1` at a *sub-complement* budget, which is the case the
  issue reported.
* It does not fix, or relate to, `TODO` #282 (a mid-run pinned latch's ~1e-5 path difference) — that was found
  while gating `A` and is independent of the fill source.
