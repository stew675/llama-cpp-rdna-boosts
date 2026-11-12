# The expert seed: is the sizing fill a cost to defer, or the prefetch? (`TODO` #318, r270)

Block-16 amendment.  Both expert tiers start with a bulk fill at sizing: the **arena** is seeded from the persisted
heat prior and the prompt tally (`apply_heat_seed_locked` / `apply_prefill_seed_locked`), and the **discrete host
pool** is prewarmed (`pool_prepopulate_locked`) and re-laid-out as the arena's complement
(`pool_rerank_from_heat_locked`).  `TODO` #318 observed that this fill dominates a small request (75 GiB / ~30 s on
`halo`, 43.9 GiB / 57 s on `soar`, against a 1.2 s prompt eval) and proposed making it **lazy** — fill on first miss
— because r266's in-place prefill band removed the reason the arena had to be pre-populated: *"the prompt's own
used set is admitted on the pass that needs it"*.

The item also says the two tiers "must be ONE policy, not two".  They are now one knob, `MOE_EXPERT_SEED`, because
the QUESTION is shared.  The **answer** is not: measured on both tiers, deferring the fill does not save bytes — it
moves the same prefetch into the request at a worse rate, and on the discrete tier it costs the device remap.  So
`eager` ships as the default and `auto` (the APU policy, ported verbatim) and `lazy` are the measured arms.

## Arms

| `MOE_EXPERT_SEED` | what it does |
|---|---|
| **`eager`** (default, and the pre-r270 behaviour) | every tier is filled at sizing |
| `auto` | a tier that holds a **subset** (`slots < n_experts`) fills on first miss; a tier that **covers** its tensor is PLACED at sizing.  This is the APU policy the item proposed, applied to both tiers |
| `lazy` | no front-loading anywhere, including a covering tier |

The prior is loaded in every arm: under `auto`/`lazy` it still ranks the prefill ladder's evictions
(`heat_rank_vec`) and the pool's slot order (`pool_rank_experts_locked`).  Only the FILL is deferred.

## Setup

* `soar`: 3 x R9700 (gfx1201), ROCm 10.1.0, ONE GPU, 91 GiB RAM, `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`
  (70.6 GiB file, 141 host expert tensors, **40 012 MiB** host expert set), `-ncmoe 47`, `-b/ub 1024`,
  `-c 4096`, `-fa on -ctk/ctv q8_0`, `--spec-type none`, greedy `--temp 0 --seed 42 --ignore-eos`,
  `MOE_EXPERT_HEATMAP=load` (a frozen prior), **page cache dropped before every run**.
  The pool budget is pinned (`-lme 32` or `-lme auto`, `MOE_HOST_POOL_RAM_PCT=0`) because the default AUTO sizing
  reads `MemAvailable` *at the first pool build*, so an arm that has already faulted the model into the page cache
  would otherwise size a smaller pool than the arm it is compared with (this bit the first cut of this measurement:
  33.6 GiB against 40.0 GiB, i.e. two different tiers).
* `halo`: Ryzen AI MAX+ 395 (gfx1151), ROCm 10.1.0, ONE GPU, 123 GiB RAM, `Qwen3.5-122B-A10B-UD-Q5_K_XL`
  (**81 318 MiB** expert set) on a `-lme 70` GTT store (71 680 MiB = 225 of 256 slots/table), `-b/ub 1024`,
  `-c 4096`, greedy, `MOE_EXPERT_HEATMAP=load`, page cache dropped before every run.
* Every arm is three runs: `-n 1` (TTFT, wall clock), `-n N` with no `-lv` (wall + text hash + `/proc/io`),
  and `-n N` with `-lv 4` (the timing footer and the census).  `read` is `read_bytes` (real storage reads)
  sampled from `/proc/<pid>/io` while the run is live.  Harness: `archive/work/expert-seed-policy/arm.sh`.

## 1. The discrete pool tier — deferring the fill moves it into the prompt

Bounded pool (`-lme 32` = 419 of 512 slots/table), 57-token prompt, 500-token greedy decode:

| arm | TTFT | wall | prompt eval | decode | pool h | arena h | storage reads | text |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| r269 (`eager`, rank-order prewarm) | 20.39 / 19.10 s | 30.92 / 30.69 s | 0.88 / 0.84 s | 40.80 / 40.93 t/s | 0.9615 | 0.9701 | 45 GiB | `6e6d241acc25` |
| **`eager` + file-order prewarm (r270)** | **16.13 / 15.54 s** | **27.54 / 26.55 s** | 1.17 / 1.20 s | 40.91 / 40.94 t/s | 0.9597 | 0.9713 | 47 GiB | `6e6d241acc25` |
| `auto` (defer both) | 17.59 s | 29.89 s | **7.24 s** | 40.25 t/s | 0.9601 | 0.9485 | 46 GiB | `6e6d241acc25` |

Mirror pool (`-lme auto` = the whole 40 012 MiB set, 512/512 slots):

| arm | TTFT | wall | prompt eval | decode | device remap | storage reads | text |
|---|---:|---:|---:|---:|---|---:|---|
| `eager` | **13.58 s** | **22.21 s** | **0.68 s** | **49.17 t/s** | **served** (70 779 ops) | 46 GiB | `6e6d241acc25` |
| `auto` | 13.30 s | 22.19 s | 0.66 s | **49.44 t/s** | **served** (70 779 ops) | 45 GiB | `6e6d241acc25` |
| `lazy` | 18.71 s | 31.14 s | 8.01 s | 40.01 t/s | **stands down** (0 served) | 51 GiB | `6e6d241acc25` |

`auto` is **inert** on a covering pool: the policy PLACES it (its content is not a choice), so only the *arena*
seed is deferred, and the prefill re-earns that within noise.  The whole loss belongs to `lazy`, which defers a
tier that must be complete.

Long prompt (`prompts/prose-rdna-boosts.txt`, 5 246 tokens, `-n 32`, bounded pool, `-c 8192`):

| arm | TTFT | wall | prompt eval | decode |
|---|---:|---:|---:|---:|
| `eager` | **27.10 s** | **38.23 s** | **7.63 s (694 t/s)** | 39.26 t/s |
| `auto` | 27.70 s | 40.86 s | **17.18 s (308 t/s)** | 38.26 t/s |

**Why it does not port.**  The APU premise is that the prefill band is served out of *residency*
(`pf_in_place_locked`, r266), so the prompt needs only the experts it routes.  The discrete pool tier has no
in-place band: its prefill **stages** the whole expert table into the op's `input_cpy`, and the staging path
prefetches the **complement** whether the pass routes it or not (a routing-aware staging needs the host ids
readback the whole gather design exists to avoid — recorded at r265).  So a prefill on this tier needs the whole
host set resident every pass, and the seed is exactly the prefetch that makes that a one-time read.  Deferring it
does not save the bytes — total storage reads are 45-47 GiB in **every** arm — it only moves them into the
request, one cold read at a time, 8x slower on a short prompt and 2.3x on a long one.

**On a covering pool, deferring it also costs the device remap** (`lazy` only — `auto` places a covering tier),
which is the second half of the same story: a mirror pool with holes cannot express the cold ids the remap kernel
resolves (`devremap_prep_locked` refuses a map with a hole), so every table stands the remap down and decode loses
the sync-free path — measured **49.2 -> 40.0 t/s (-18.6 %)**.

## 2. The APU tier — the same trade, softer, but still not a default

`-lme 70` (225 of 256 slots/table = 88 %), 15-token prompt (the chat template counts 15 tokens on this tokenizer):

| arm | TTFT | wall (`-n 250`) | prompt eval | decode | arena h | storage reads | text |
|---|---:|---:|---:|---:|---:|---:|---|
| r269 binary (control) | 47.69 s | 70.06 s | 13.43 s | 11.60 t/s | 0.9796 | 171 GiB | `960bdbc9e2dc` |
| `eager` (this build) | 47.77 s | 69.46 s | 13.95 s | 11.73 t/s | 0.9796 | 166 GiB | `960bdbc9e2dc` |
| `auto` (defer the subset arena) | **28.99 s** | 80.53 s | 12.54 s | **4.83 t/s** | 0.9064 | **236 GiB** | `960bdbc9e2dc` |

A **32-token** answer (the "small prompt" the item is about):

| arm | TTFT | wall (`-n 32`) |
|---|---:|---:|
| `eager` | 49.65 s | 50.29 s |
| `auto` | **29.26 s** | **44.48 s** |

So on an APU the deferred seed buys **-39 % TTFT** and, for a very short answer, -12 % of the whole run — and pays
**2.4x decode** and +70 GiB of reads on a longer one.  That is a real window (a one-shot desktop prompt with a short
answer), but it is a window, not a default: the same box serving a 250-token answer is 16 % worse, and the decode
collapse is the mechanism, not noise.

**Neutrality on gfx1151:** the r269 binary and this build with `MOE_EXPERT_SEED=eager` are byte-identical in text
(`960bdbc9e2dc`, 821 chars), identical in arena hit rate (0.9796 both), and within 1.1 % on decode — the shipped
default does not move anything on the APU.  `scripts/gate-uma-experts.sh --quick --no-ram` on the new build:
**36 checks, 0 failures** (the in-place arm still serves 108 tables / 16 272 admissions).

## 3. What the seed's cost actually was: the fill TRANSPORT

The bounded tier's eager arm was **12.5 s** of pool prewarm for 42.7 GiB, i.e. 3.67 GB/s, while the *mirror* tier's
bulk fill of the same order of bytes runs at **6.6 GB/s** (306 MiB sequential in 46 ms).  The difference is not the
bytes and not the drive: it is the **issue order**.  `pool_populate_locked` filled its slots in *rank* order — a
random walk over a 40 GiB GGUF under a page cache that cannot hold it — which is the same defect r231 fixed for the
band-transition re-layout (`MOE_HOST_POOL_FILL_ORDER`), still present in the load-time prewarm.

Filling the same **set** in ascending expert id (ascending file offset) measures, on the bounded tier:

| | prewarm | effective fill rate | TTFT | wall | decode | text |
|---|---:|---:|---:|---:|---:|---|
| rank order (r269) | 12 546 ms | 3.67 GB/s | 19.10-20.39 s | 30.69-30.92 s | 40.80-40.93 t/s | `6e6d241acc25` |
| **file order (r270)** | **7 774 ms** | **6.25 GB/s** | **15.54-16.13 s** | **26.55-27.54 s** | 40.91-40.94 t/s | `6e6d241acc25` |

**-38 % of the seed, -20 % TTFT, -13 % of the whole run**, decode unchanged, text byte-identical, and the SET is
identical (the slot an expert lands in is arbitrary by construction — `pool_fill_empty_locked` takes the first free
slot — so this changes nothing but the order the reads are issued in).  The prompt eval pays +0.3 s of it back
(1.17 s against 0.88 s, reproducible across two runs per arm; the file-order arm leaves the *coldest* pages in the
cache instead of the hottest, and the prefill's staging misses ~2 GiB more).  Net wall is 4 s better.

Two other transports were measured and are **not** adopted:
* `MOE_HOST_POOL_DIO=1` on the same arm: **21 252 ms** of storage reads for 43 GiB (2.1 GB/s), TTFT 29.3 s, 52 GiB
  read.  A ~0.55 MiB `O_DIRECT` read at depth 1 is latency-bound — the documented shape; the prewarm needs *fewer,
  larger* reads, not a read-around.
* The seed cannot simply be smaller: it is the working set (the arena's 310 of 512 slots per table at the delivered
  auto budget; measured arena `h` 0.97 with it, 0.95 without).

## What ships

* `MOE_EXPERT_SEED` (`eager` default | `auto` | `lazy`) — one knob, both tiers, with the numbers above recorded in
  `ENVIRONMENT.md`.
* `MOE_HOST_POOL_FILL_ORDER` now governs the **prewarm** as well as the re-layout (its r231 default, `offset` =
  file order, is the win above; `rank` is the A/B arm).
* Nothing else moves: `eager` reproduces r269 exactly (text, hit rates, decode), the prefill-logit KLD gate reads
  **0.000000 / 100.000 %**, and `scripts/gate-host-pool.sh` passes **42/42**.

## Not measured here (and owed)

* A **multi-request** APU arm: the window above is per-run, and a server's second request inherits a warm arena, so
  `auto`'s true server cost is bounded by the first request.  The lazy arm's decode collapse says it is still a
  residency question, not a warm-up one.
* `auto`/`lazy` on a **`-sm tensor`** split: the pool's per-device budget and the remap's covering-pool source are
  per device, and this measurement is single-GPU on both tiers.
