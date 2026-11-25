# Issue #368 — the pool tier's slab ring hole is arena the prefill streams back from the host (r362)

**Claim.** On a POOL-backed host-expert tier the movable-boundary slab reserved its whole-table H2D staging ring as
a fixed hole above the narrow floor, and nothing ever read it: the default `GGML_SCHED_DEVGATHER` mode gathers a
pool-backed routed table at *every* prefill width.  The hole's whole extent therefore came out of the expert arena
every time a wide prefill view was placed — 8500 MiB / 51 whole tables stood down at every prefill, plus their
54-table re-arm at the drop — and that is residency the prefill then re-reads from the host on every ubatch.
Declining the hole when the tier is the pool reads **+3.4 % prefill / +2.2 % decode** and **+24 %** on the
conversation's cache-hit revisit, with byte-identical text.

**Why this is the issue's axis.** GitHub issue #368 (`@DanoPTT`) reports two deterministic states on
`Qwen3.8-Flash-Next IQ3_XXS` at 1 GPU, `-ncmoe 47 --host-experts pool`, `-ub 8192`, `-np 2 --kv-unified`: L1 prefill
2960-3000 t/s (tg 52) against 1615-1620 (tg 30-32), two different greedy texts, and `-ub 4096` never switching.
Their slow state is slow "from the first 8192-token chunk", and their *good* visit has no extra eviction at L2 while
the slow one re-evicts 7043 MiB / 42 tables.  That is this mechanism: whether the arena keeps its tables through the
prefill decides what the prefill has to stream, and every whole-table stand-down is paid again on the next pass.

## 0. The box and the shape

| item | value |
|---|---|
| GPU | 1 x R9700 (gfx1201, 32 624 MiB), ROCm 10.2.0, HIP_VISIBLE_DEVICES=0 |
| CPU / RAM | 9950X3D, 91 GiB (64 GiB available; the pool pins 40 012.5 MiB) |
| model | `/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/...-00001-of-00002.gguf` (75.8 GB, `per_layer_token_embd` 26.8 GB lazy) |
| flags | the reporter's: `-ngl all --n-cpu-moe 47 --host-experts pool --load-mode auto --lazy-mode on -c 131072 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q8_0 -b 8192 -ub 8192 -t 8 --cache-ram 4096` |
| env | `MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=85 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_GATHER_NSPLIT=4 ROCBLAS_USE_HIPBLASLT=1` |
| visit | S1 (1102 tok, slot 1, n=30) -> L1 (89 199 tok, slot 0, n=300) -> S2 (slot 1) -> L2 (L1 + S2, a cache hit), temp 0, seed 1, a fresh server per visit |
| arena | 20 397 MiB of 40 012.5 (51 % residency), `L1..L48 261/512`, no identity table, pool `h=1.0000` |

The prompts are generated, not the reporter's: `l1-90k.txt` is `prompts/prose-rdna-boosts.txt` repeated 17x (273 275 B
-> 89 199 tokens), `short.txt` is `recall.txt` twice, `l2.txt` is `l1-90k.txt + short.txt`.

## 1. What reproduces, and what does not

**Nine fresh starts read one state.** L1 1890-1900 t/s, tg 34.06-34.24, text `98c4f67d6ed9` every time; S2 46.1-46.8;
L2 721-728 / 38.5.  The eviction pattern matches the reporter's closely (6602 MiB / 40 tables then 8500 / 51, re-arm
42 then 54, against their 6811 / 41 and 8546 / 51, re-arm 42 and 54), so the shape is the same machine.  The
**bimodality did not reproduce** on this box, and neither did the two texts, so this record claims nothing about
their cause; what it does claim is the mechanism that makes both states *cost* what they do.

**Eleven one-variable arms, all on the same L1 prompt and all reading the same text:**

| arm | L1 pp | tg | what it rules out |
|---|---:|---:|---|
| default (as reported) | 1895 | 34.18 | -- |
| `MOE_EXPERT_CACHE_MIB=0` (no cache, no pool) | 2559 | 12.83 | the pool gather is the prefill transport, not the arena |
| `MOE_EXPERT_CACHE_MIB=512 / 12000 / 26000` | 1870 / 1958 / 1895 | 25.5 / 34.2 / 34.2 | residency breadth is not the prefill lever |
| `GGML_SCHED_DEVGATHER=0` (ring staging instead of the gather) | 1709 | 34.11 | staging is the slower transport here, as designed |
| `GGML_SCHED_STAGE=0` (no ring at all) | 1958 | 35.03 | the ring carve costs the arena, not the gather |
| `GGML_SCHED_STAGE=0 GGML_SCHED_DEVGATHER=0` (serial) | 452 | 18.57 | the per-op serial copy is not a fallback anyone wants |
| `GGML_CUDA_MMB=0` | 1888 | 34.05 | MMB is not a state |
| `GGML_CUDA_SLAB=0` | 1942 | 35.27 | the slab itself is not the cost |
| `GGML_CUDA_SLAB_HEADROOM_MIB=3072` | 1897 | 34.18 | headroom is not a state |
| `MOE_GATHER_NSPLIT=8 / 32` (the reporter sets 4) | 1913 / 1953 | 34.14 / 34.11 | the gather's launch shape is worth 3 % |
| `--lazy-mode off` / `-ot per_layer_token_embd.*=CPU` (the PLE) | 1834 / 1886 | 31.6 / 34.2 | the PLE's lazy reads are not a state here (46 GiB of page cache) |
| `GGML_CUDA_SLAB_RING_MIB=0` | 1943 | 35.00 | **the ring hole** |
| `--host-experts pinned` / `-lme mirror` / no flag | 1887-1892 | 33.9-34.2 | the pool tier is what the loader resolves on this shape |

**Every arm read `98c4f67d6ed9`.**  On this box the expert cache's residency is not a text input on this shape —
which is a useful negative for the reporter's question 2, and consistent with §89's corrected doctrine.

## 2. The mechanism

The delivery carries two independent statements, and on a pool-backed host-expert tier they contradicted:

1. `sched_input_gatherable`, mode 1 (`GGML_SCHED_DEVGATHER` unset — the default): a POOL-backed routed table is
   gathered at **every** prefill width, because that tier has no device-readable host master and whole-table staging
   of it is the slow path by design.
2. `h2d_stage_region_bytes()`: the ring hole is `h2d_stage_slots() x (largest host weight + 512)`, capped by an
   auto-size of `1800/n + ub/(1.7 n^2)` MiB — **6656 MiB at `-ub 8192`**, 1800 at `-ub 4096`.  It is carved above
   the narrow floor, so the wide prefill view is placed behind it.

So at `-ub 8192`:

```
slab 23.44 GiB: work 3.19 GiB (narrow floor 1792 MiB) + arena 20.25 GiB, ring region pending
  narrow 1536 MiB buffer   -> boundary   3328 MiB
  wide   3328 MiB buffer   -> boundary  11776 MiB   <- 1792 narrow + 6656 ring + 3328 wide
  moe_cache_evict_slab_range([3264, 11776)) -> 8500 MiB / 51 tables STOOD DOWN, every prefill
  ...54-table re-arm at the drop, then it happens again on the next pass
```

`-ub 4096` carves 1800 MiB instead, which is why the reporter's 4096 arm never switches state: there is almost
nothing to take from the arena at that width.  **The hole is the `-ub 8192` cliff**, and it is arena that nothing
reads.

## 3. The fix

`slab_ring_region_bytes()` in `ggml/src/ggml-cuda/ggml-cuda.cu` (both call sites: the preflight and the context's
`slab_set_ring_cap`) returns 0 when `moe_cache_host_pool_requested()` — a new read-only query on the cache's
`g_pool_requested`, which the loader sets for `--host-experts pool` and for an `-lme` budget that resolves to the
per-expert pool.  It still defers to `GGML_CUDA_SLAB_RING_MIB` (an explicit size wins) and to an explicit
`GGML_SCHED_DEVGATHER=0` (that arm really stages).  A non-pool tier, a dense model and every non-expert host weight
are untouched.

Same change, the diagnostic the issue asks for (question 3): `MOE_EXPERT_CACHE_TABLES=1` prints, at exit, the
per-layer residency map and this run's per-layer slab-reclaim stand-down / re-arm tally.

## 4. Measured, after

Four fresh visits, interleaved against the r361 arms above (same server flags, fresh heat-map directory so the state
is cold):

| arm | S1 pp/tg | L1 pp | L1 tg | L1 text | S2 tg | L2 pp | L2 tg | stand-downs |
|---|---|---|---:|---:|---|---:|---:|---|
| r361 (4 visits) | 525-530 / 23.5 | 1890-1900 | 34.06-34.21 | `98c4f67d6ed9` | 46.1-46.8 | 721-728 | 38.5-38.6 | 8500 MiB / 51 tables per prefill |
| r362 (this change) | 540 / 24.02 | **1958.5** | **34.94** | `98c4f67d6ed9` | 47.12 | **902.5** | **40.58** | **none** |

+3.4 % L1 prefill, +2.2 % L1 decode, +24 % on the cache-hit revisit (which no longer re-evicts 6778 MiB / 41 tables),
and the arena keeps 20 397 MiB through the whole run (`residency by layer: L1..L48 261/512`).

## 5. Gates

* `scripts/gate-prefill-logits.sh` — **0.000000 / 100.000 %** (gfx1201, ROCm 10.2 base), no GPU faults.
* `scripts/gate-host-expert-text.sh --quick` — 2/2 identical (`328/7c5a512a8447` at its own flags); the flipped-library
  arm still reads a different text, so the pin is live.  **Its `--kld` leg has no base recorded on this box for that
  point** (the existing file is the r310 name/pin set) — the r361-vs-r362 text identity on this record's own shape is
  the neutrality evidence used instead, and recording that base is owed.
* `scripts/check-shlib-symbols.sh` — 16/16; `scripts/check-gen-docs.sh` — ok (no option added);
  `scripts/check-gpu-faults.sh` — clean; `scripts/validate-set.sh` — green, rung reconciliation `0 unexplained`
  (the two new env keys counted as additions).
* Not run in this session: `scripts/gate-host-pool.sh` and `scripts/gate-kv-residency.sh`.  The change moves no
  kernel, no routing and no fill; it moves the *slab layout*, and their arms pin the residency explicitly.

## 6. What is NOT claimed

The reporter's two states and two texts are **not** reproduced here (nine starts, one state, one text), so this
record does not say what flips between their visits.  What it says is that their slow state is paying the pool
tier's gather for tables the slab did not need to stand down, that `-ub 8192` is where that hole is largest, and
that `MOE_EXPERT_CACHE_TABLES=1` is the reading that will name the difference on their box.
