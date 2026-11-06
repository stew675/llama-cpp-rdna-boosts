# PR #267 review + the convergence design — `archive/work/host-expert-split/`

**Status: OPEN — review complete, NOTHING IMPLEMENTED, nothing applied to the fork, no `patches/` change.**
Companion to [`README.md`](README.md) (the scoping note).  This file is the review of
[PR #267](https://github.com/stew675/llama-cpp-rdna-boosts/pull/267) (`@briansp2020`,
branch `wip-mmap-host-experts-r204`, head `691108ffab8d`, 10 patches cut against **r204**) and the
convergence design against **r223** (`release.json` tip `37148d118`).  The verbatim PR patches are
**not** committed here; they are reproduced with
`git archive <remote>/wip-mmap-host-experts-r204 wip/mmap-host-experts-r204`.

---

## 1. What the PR actually adds

Ten patches, ~2 600 lines, all under `--host-experts mmap` (default stays `pinned`).  Grouped by job:

| # | patch | mechanism | lines |
|---|---|---|---|
| 0001 | loader restore | `--host-experts mmap` puts the `-ncmoe`/`-cmoe` expert weights back in the pageable model mapping (`CPU_Mapped`), and keeps host-expert accounting keyed by the **GPU** so the MoE preflight (and the slab arming) still runs | 43/30 |
| 0002 | sparse registration | a second read-only `MAP_SHARED` alias of the model file; `mlock` + `hipHostRegister(ReadOnly)` **only the expert ranges a GPU may read** (the non-resident set); `madvise(WILLNEED)` before each; a trim unregisters what no reader can reach (after `cudaDeviceSynchronize` on every device) and a `MAP_FIXED` remap drops KFD's leftover SVM range; `-sm layer` owner-device routing; whole-table prefill staged from arena+alias | 873 |
| 0003 | slab/context | the narrow-2 (draft compute) pin stays on its `cudaMalloc` fallback when it fits **outside** the slab instead of evicting arena tables; `GGML_CUDA_SLAB_RING_PREFILL=0` (implied by mmap) keeps the arena across a decode→prefill transition | 152 |
| **0004** | **RAM tier** | **the core**: a pinned (mlocked) host buffer holding **exactly the experts the arena does not hold**, in the arena's compact layout, plus **one spare**; `cold_ptrs[e]` gives each cold expert its own source address; an admission is a **swap** — victim D2H into the spare, incoming H2D into the arena slot, both as custom copy **kernels** (not `hipMemcpyAsync`); a freed slot is reusable only after `swap_ev`; dropped wholesale on budget/allocation/re-size/stand-down/arena-shrink/slab-evict | 368 |
| 0005 | tier × devremap | the device remap works for a tier table; a tier table **never** uses the pool cold read | 25 |
| **0006** | **precopy** | K spare RAM slots (`GGML_MOE_TIER_SPARES`, default 8) + a **side stream** that copies the residents the policy would evict next into spares ahead of time, so a swap needs only the H2D on the compute path; one spare always stays free for the in-stream fallback | 226 |
| 0007 | precopy drain | a failed event record drains the side stream; a failed wait fails the swap | 5 |
| 0008 | defaults | 8 spares, and a tier table admits from the **3rd** touch (`GGML_MOE_TIER_TOUCH`) to damp churn | 16 |
| 0009 | pin budget | **one** budget for the tier + the registration (`GGML_MOE_PIN_BUDGET_MIB`, default `min(60 % RAM, RAM-8 GiB)`); a **partial tier** (hottest tables first) instead of all-or-nothing; warm-up registrations bounded; a refused registration is not retried until the next trim | 114 |
| 0010 | declined read | a read the cache declines goes through the **8-entry** per-table pinned bounce ring (capped: a whole-master declined read must not pin a copy of the master).  No kill switch — the old pageable path produced wrong output and then KFD's `exceeds resident system memory limit` | 17 |

The load-bearing idea, in one sentence: **the pinned host tier is not a separate cache of the arena's
complement chosen up front — it is the arena's own eviction sink, and an admission is a two-way swap.**

## 2. What r223 already has (feature-by-feature)

| PR feature | our state | delta |
|---|---|---|
| `--host-experts mmap` as a mode | **the mode is gone and must not come back**: since r223 `--host-experts` is a **deprecated alias** and the placement is one budget, `-lme [auto|N]`.  Below the host expert set, `-lme N` **is** a per-expert pinned pool over the read-only model mapping; at/above it, a **mirror**.  `--host-experts pinned` / `MOE_HOST_TIER=tensor` is the whole-tensor escape hatch only | none — `mmap` would be a third spelling of a shape we already own |
| GPU-safe read of a pageable master | **pool in-place cold read** (`MOE_HOST_POOL_COLD`/`_MULTIDEV`, cold id `n_res + pool slot`, `moe_cache_get_cold` returns the pool base) so a pool-resident miss is a **PCIe read of pinned RAM**, not a copy and not a disk read; plus the per-table **bounce ring** for what the pool declines (`expert_host_src`, the `gate-host-pool.sh` arms) | the PR's route is *registration in place*; ours is *read in place from the pool, else bounce*.  For a pool-resident expert these are the same cost |
| bounded pinned footprint | `-lme N` **is** the pin budget, and `MOE_HOST_POOL_RAM_PCT` (85 %, r114) caps the whole pool as a % of `MemAvailable` | the PR's `PIN_BUDGET_MIB` also covers registration; ours covers the pool only — and C3's job is to keep it one number, not two |
| **tier = arena's eviction sink** | **deliberately not**: `INVARIANTS.md` — "the demand path never evicts and never copies the arena victim into the pool — do not reintroduce one: the literal swap cost **2x decode** for 0 MiB of L3" | **this is the one real delta, and it is now a single direction**: promote (pool→arena) already exists as the fill; what does not exist is **demote** (arena victim → the pool slot the promoted expert vacated) |
| pool content policy | **arena's complement first, then the hottest arena residents while capacity lasts** (`pool_rerank_from_heat_locked`), ranked by the persisted heat prior, re-laid out at sizing and at the first decode-band pass; write-once for a covering pool (TODO #269) | ours is a *layout* (15× better than any policy on L3 in the r113 replay); the PR's is dynamic-recency |
| prefill staging | device-side assembly from arena + pool (`moe_cache_stage_table`), one launch, both tiers (r114) | equivalent job |
| partial tier | **the whole point of the open question**: a budget below the complement means an expert in neither tier is a file read — and on the 56 GiB rig that is the whole cliff (`-lme 51` at 93.6 % costs 5 % decode / 9 % prefill; 92.4 % halves the first prefill, `wip/bench-56gb/RESULTS.md`) | **real** |
| precopy / copy-back ahead | nothing — we never copy back, so there is nothing to hide | only meaningful once demote exists |

## 3. Why this is not simply "r113 again"

The r113 refutation is real but it was measured on a shape that is the **worst case for a swap**, and the
campaign's own evidence points at a different shape:

1. **The copy was 2-D strided.**  r113's `pool-swap-slot` arm ran `-sm tensor` on 2 devices, so an
   axis-0 slice is a fine strided scatter issued as `cudaMemcpy2DAsync` — **6 259 swaps × 2 copies = 2× decode
   (25.9 vs 52.8 t/s)**.  The PR is **single-device only**, where an expert is one contiguous 1-D block and
   the swap is two `copy_kernel` launches.  r113's numbers do not transfer to that shape, and the PR does
   not claim they do.
2. **The r113 pool already covered the complement.**  At `--pool-init sidecar-complement --pool-frac 80`
   the L3 axis had nothing left to win (237 MiB; every policy sat within 0-246 MiB) — a swap can only lose
   there.  The open case is the one our measurement produced *after* r113: a pool **below** the complement,
   where a miss is a file read at ~125 MB/s and the prefill halves.
3. **The box changed.**  `wip/bench-56gb/` recreates a 59 GiB box: a mirror pool is 48.9 GiB of host anon
   with 6.3-6.9 GB of `MemAvailable` left, i.e. 100/0 is the only split that does not cliff **today**.  A
   cheap miss is what would make 80/20 viable — that is exactly campaign step 2 in `README.md`.

So the question the review leaves open is narrow and testable: **on a single device, with a pinned budget
below the arena's complement, does an eviction sink beat our layout on L3 traffic by more than its copies
cost?**

## 4. Convergence decision (proposed)

Adopt, in this order, and only as far as the measurements justify:

* **C1 — demote on eviction** (`-lme`'s pool only; no new mode, no `--host-experts mmap`, no second tier).
  Reuse everything that already exists: the pool is per-expert, has an LRU `last` map, an eviction path, a
  `slot_state == 2` (retiring) ordering state, an in-place read of a resident (`pool_cold_slot_locked`) and a
  fill (promote) path.  The delta is **one call site**: where the arena evicts a resident, copy the victim
  D2H into the pool slot the promoted expert just vacated (today: drop it) and repoint that expert's pool
  index.  No new allocator, no `cold_ptrs` array, no second slot map.  One floor is needed, and it is the
  PR's 0004 shape: **a demote needs a free pool slot**, so each table keeps one spare (the promoted expert's
  slot is not always free — a *mirrored* resident is already in the pool).  With no spare, the victim is
  dropped exactly as today (Brian's "misses served cold because no spare was ready"), so this can only ever
  add a read that we already have.
* **C2 — precopy** (the PR's 0006/0008), only if C1 measures positive: the D2H is the part that lands on
  the admission path, and one spare per table plus a side stream is cheap.  1-D only; skip it on a
  strided/`-sm tensor` slice.
* **C3 — the pin budget as one number** (the PR's 0009, minus the registration half): fold
  `MOE_HOST_POOL_MIB`/`FRAC`/`RAM_PCT` into one documented budget with a **partial tier by hotness**
  instead of a silent per-table shave.  This is the `wip/host-expert-coverage/` ask too, and it is
  independent of C1.
* **not adopting D0 (sparse registration)** as the first move: it is 873 lines, it is the half of the PR
  that owns `mlock`/`hipHostRegister`/`MAP_FIXED` churn on a file alias, and the win we are chasing (a cheap
  miss) is bought by the tier alone.  It stays a re-entry option *only* if a partial tier cannot be made
  cheap — the PR's own numbers say a partial tier is still a cliff (20 GB budget → 28.4 t/s, 59 tables
  without a tier), which is an argument for C3, not for D0.
* **keep refusing** the demand-path *re-read-the-victim-into-the-pool* half of `pool-shipped` (the r113
  mechanism, 2× decode).

## 5. Port analysis vs r223 (not a cherry-pick)

The PR is cut against r204 and touches code that moved in r209-r223.  Its 0002 alone conflicts with every
r215+ change in `moe-expert-cache.cu` (8557 lines now).  Specifically:

| PR hunk | r223 state |
|---|---|
| `ggml-cuda-moe-cache-hooks.cu` 2-arg `moe_cache_stage_table` | r223 has **one** signature; the sparse overload would be a third entry point for a mode we do not add |
| `ggml_backend_cuda_device_offload_op` owner-device check | r223's `-sm layer` path already takes the owner through the semantic key; needs re-derivation, not a patch |
| `table_t` additions (`host_sparse`, `stage_dev`, `route_dev`, `ram_*`, `cold_ptrs`) | `table_t` has grown (prs/gprs snapshots, `dr_*`, `fill_failed`, slice array) — the tier fields are additive, the sparse ones are not |
| `mmvq-moe.cu` `cold_ptrs` kernel argument | our `mul_mat_vec_q_moe` carries `cold_base`/`n_res_cold`/strides **and** the gate lane; adding a `uintptr_t *` pair is the intrusive part and is exactly what C1 avoids (the pool slot IS the address) |
| 0003 `slab_narrow2_pin_locked` + `GGML_CUDA_SLAB_RING_PREFILL` | block 10's slab has changed; the narrow-2 fallback idea is independently good and is *not* mmap-specific (his own note: it applies to every mode) — **assess it on its own merits**, it is a candidate for a separate block-10 amendment |
| `MOE_HOST_POOL_RAM_PCT` vs `GGML_MOE_PIN_BUDGET_MIB` | two ceilings now; C3 must merge them, not stack them |

**Consequence: no patch in this PR can be applied as-is.**  C1 is ~150-250 lines in
`moe-expert-cache.cu` against our existing pool; C2 ~100; C3 ~120 (mostly the loader/`-lme` side).
That is a re-implementation guided by the PR, which is what `AGENTS.md` asks of a `wip/` item.

## 6. Measurement plan

### Step 1 — offline replay, three miss models (no GPU change, needs one trace)

`scripts/moe-trace-replay.py` already models Brian's shape: `pool-swap-slot` is "the victim takes the
incoming's own slot" and `pool-swap-weakest` is "the victim overwrites the pool's weakest resident" — both
are the PR's swap, and `pool-none`/`pool-retire` are ours.  It reports `l3_mib` (file reads), `promote_mib`
(host→arena), `handover_mib` (the extra copies a sink policy pays) and `l3_mib_per_tok`, so the three miss
models are a **conversion** of the same run, not three runs:

| miss model | who | cost per MiB |
|---|---|---|
| L3 = disk at the measured rate | our partial pool today | ~125 MB/s (1.08 GiB / 8.8 s, `bench-56gb/RESULTS.md`) |
| L3 = a single pinned PCIe copy | the PR's tier (a "free" miss, page-cache resident) | ~26 GB/s host-link |
| L3 = 0 (free) | the shape's upper bound | — |

Trace to capture (one bounded run, **1 GPU**, no all-reduce, no drafter, partial arena so every table's
**decode** band is traced — an identity table's decode routing is never read back):

```bash
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 \
       GGML_CUDA_SLAB_HEADROOM_MIB=1536 \
       MOE_EXPERT_CACHE_MIB=4096 \
       MOE_EXPERT_TRACE=/tmp/fn-prose.trace \
       MOE_EXPERT_TRACE_MIB=1024 \
       LLAMA_KV_REGION_POOL=0              MOE_HOST_POOL_RAM_PCT=0
~/llama-cpp-rdna-boosts/scripts/gpu-watch.sh --out /tmp/fn-prose.csv --cmd "
  ~/llama.cpp/build-rocm/bin/llama-cli \
    -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    -ncmoe 48 -ngl 99 -lme 4 \
    -ot 'per_layer_token_embd.*=CPU' \
    -lm mmap -lzm auto -fa on -c 8192 -ub 4096 \
    -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt   # 5246 tokens, sha256 fabdec65…
    -n 512 --seed 42 --temp 0 --no-display-prompt --single-turn
" --timeout 1200
```

Why each knob: `-lme 4` is a 4 GiB **per-expert pool** (below the host expert set, so the masters stay in
the model mapping instead of pinning 46.84 GiB — a `-lme 47`/`auto` arm would pin the whole set and make the
run heavier for no change in the routing); `MOE_EXPERT_CACHE_MIB=4096` makes the arena partial, which is what
puts **decode** records in the trace;
`-n 512` is ~4× the 128-token field decode; the prompt is the shipped hash-stable one, so the trace is
reproducible; `gpu-watch.sh --cmd` puts the whole thing under `timeout --kill-after=10` and records the
flight data (see §7).

Expected size: ~56 MB of prefill ids (2 ubatches × 144 tables × 4096×8) + ~3 MB of decode — well under
the 1 GiB cap.

Replay grid (`--devices 1`, the full expert, arena ≈ 19 GiB as measured):

```
--arena-mib 19000 --pool-frac {80,92,100} \
--pool-init sidecar-complement:/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.moe-heatmap \
--admit always --policies pool-none,pool-retire,pool-shipped,pool-swap-weakest,pool-swap-slot
```

(`--pool-init sidecar-complement` is the real layout: the arena is seeded from the same prior and the pool
takes the non-resident rest.  `--churn on` is the shipped behaviour.  The command line itself was smoke-
tested on a synthetic trace — every policy name, the `L3 MiB/tok` and `handover MiB` columns and the
two-tier accounting all resolve.)

*Accept:* the sink policies beat `pool-none` on `l3_mib_per_tok × (1/125MB/s − 1/26GB/s)` by more than their
`handover_mib` costs, at a budget where `pool-frac < complement`.  If not, the campaign parks here — the
same shape of result r113 produced, and the cheap honest outcome `README.md` §3 step 1 asks for.

### Step 2 — C1 alone, on the GPU

Only if step 1 separates.  One `pool` arm with `-lme` below the set, on the 56 GiB rig (the box that
forces the split), against the same arm with the sink disabled (`MOE_HOST_POOL_SINK=0`, the kill switch).
*Accept:* first-pass 50k prefill within 5 % of the mirror, decode not below `pool-none`, greedy text
byte-identical, `check-gpu-faults.sh` clean.

### Step 3 — the split sweep

Only if step 2 passes: (pool, arena) = (47, auto), (44, auto), (40, 26), (34, 32) at `-c 204800`,
`-ub 6144` and 2048.  *Accept:* a crossover table and a rule that can become a documented default.

### Gates (all of them, if this ships)

Same-seed greedy text at every budget; the MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`)
for any decode-path change; `scripts/gate-host-pool.sh` (a fill fault must DECLINE, not take over — the
sink adds a new failure mode to that gate's arms, which must be extended, not bypassed);
`scripts/gate-prefill-logits.sh` if the staging source changes; `scripts/check-gpu-faults.sh` after every
arm.

## 7. Risk register

* **A hard machine lockup on 2026-11-05.**  The box wedged with a R9700 at 100 % and had to be power-cycled;
  the kernel logged **nothing** (the previous boot's journal stops ~70 min before the end), so no ring
  timeout, no reset, no OOM was recorded.  Two mechanisms in this codebase match: a spinning compute kernel
  (the block-01 all-reduce class — no ring timeout fires because the queue makes progress) and a
  driver-side wait.  **No benchmark or build was running from this session.**  Until the cause is known:
  single-device arms only (no all-reduce at all), every run under `timeout`, one arm at a time, and
  `scripts/gpu-watch.sh` recording GPU use / VRAM / `MemAvailable` / `VmLck` so a recurrence is diagnosable.
* **`mlock` + `hipHostRegister` + `MAP_FIXED` churn (D0)**: this is the mechanism we are *not* adopting
  first, and it is the one most likely to interact with the KFD SVM range/restore worker.  The PR's own
  0010 exists because the first version of it produced 47 771 KFD refusals and then "no kernel image is
  available".  C1/C2 need none of it.
* **The sink is a write to pinned host memory from the D2H path inside a capture.**  r113's swap also had
  to be ordered against every reader; a sink inside a CUDA graph would be a capture-time hazard.  Our pool
  already carries `slot_state == 2` (retiring) for exactly this reason — the sink must reuse it, not add a
  second ordering state.
* **Windows**: the PR is POSIX-only for the sparse half.  C1/C2/C3 are portable (no `mmap` alias, no
  `mlock`), which is a second reason to prefer them.
* **`-sm tensor`**: the sink's copy is 2-D strided there, which is the r113 2×-decode result.  Decide up
  front: **1-D only, `-sm tensor` declines the sink** (`README.md` §3 step 5).

## 8. What would kill it

* Step 1 reproduces "the split barely matters once the layout is right" → converge on C3 only, tell the
  reporter the transport is the difference, close the campaign.
* Step 1 separates but step 2 does not (model overestimates the sink's benefit, as it did in r113 by 0 MiB
  on the L3 axis).
* The sink cannot be made graph-safe without a per-token synchronize.
* The lockup turns out to be in the pool/tier path — then the whole `--host-experts pool` family gets a
  correctness review before any of this ships.
