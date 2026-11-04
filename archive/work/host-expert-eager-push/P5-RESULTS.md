# P5 — the `-lme` budget is now a per-TENSOR pinned-master tier, filled in descending hotness

**Status: IMPLEMENTED and MEASURED (2026-11-04), waiting on the maintainer's go/no-go before it is promoted
into the delivery.**  Branch `p5-host-expert-tier` in `~/llama.cpp` (commit `c1478f911`, parent
`a644115a0`), exported as [`patches/0010-p5-host-expert-tier.patch`](patches/0010-p5-host-expert-tier.patch).
Nothing in this file is part of the delivery set yet.

This is [`IMPLEMENTATION-PLAN.md`](IMPLEMENTATION-PLAN.md) steps **a-e** executed; read that plan and
[`P4-ROOT-CAUSE.md`](P4-ROOT-CAUSE.md) for the decisions and the evidence they rest on.  What follows is
what was built, what it measures, and the two findings that change the plan's shape.

---

## 1. What was built

`-lme, --limit-host-experts` is now the **budget for the pinned host-master tier**, resolved at MODEL LOAD,
before a single host-expert tensor is placed:

| request | resolved placement |
|---|---|
| `--host-experts pinned` | the whole host expert set (unchanged meaning) |
| `-lme auto` (the default with `-ncmoe`) | the whole set when it fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable`; otherwise **no** pinned master — every host expert stays in the read-only model mapping and the bounded pinned pool serves the reads (the not-fitting band, i.e. the old `--host-experts pool`) |
| `-lme N` | N GiB, spent on **whole host-expert tensors in descending hotness order**; the rest stays in the model mapping |
| `--host-experts pool` | deprecated shorthand for `auto` |

Three properties make this safe to sit on the existing engine:

1. **A pinned host tensor IS the region.**  The master is addressed by expert index (`base + e * host_bytes`),
   so a wholly pinned tensor is byte-for-byte today's `pinned` master: `host_pinned` becomes a per-TABLE truth
   about the weights, the table keeps the single-launch staging branch, the in-place cold read and the UVA
   alias, and `P1-PUSH: fill source: master=pinned` still reads the same.  No presence bitmap, no hole
   encoding, nothing shared with the KV/arena paths is touched.
2. **The budget is decided before placement**, from a metadata pre-pass over the GGUF index
   (`weights_map`): the candidate set (the `-ncmoe` tail rule or an explicit caller `--cpu-moe`/
   `--override-tensor` match), each tensor's `nbytes`/`nb[2]`/`n_experts`, and the byte total.  `auto` needs
   that total to answer "does it fit"; `-lme N` needs it to rank.
3. **The ranking is the persisted heat prior**, opened through two new device-API entries
   (`ggml_backend_dev_moe_heat_open/_score`) instead of at the first tensor's *data* copy (which happens long
   after every buffer-type decision).  The key is the same `(layer, name, n_experts, host_bytes)` the
   residency prior uses, so a pinned set and a pool's contents are ranked by the same evidence.  No sidecar (a
   fresh model) falls back to the model's own tensor order, and says so in the census.

Census, always at INFO (a user must be able to see which tier they got — the pageable fallback costs about
half the throughput and the only other warning fires when a pinned *allocation* fails, not when the policy
chose not to pin):

```
plan_host_expert_placement: host experts: auto: the host expert set (18662.0 MiB) fits 80% of MemAvailable
    (66093.5 MiB) -> pin all of it: 18662.0 of 18662.0 MiB pinned (120 of 120 tensors, descending hotness);
    0.0 MiB pageable (per-table bounce ring)
report_host_expert_placement: host experts: pinned 18662.0 MiB of 18662.0 MiB (cap 18662.0 MiB),
    pageable 0.0 MiB; auto: ...
```

`-lme` no longer touches `MOE_HOST_POOL_MIB` (patch `0009` set it; the plan required re-targeting), and the
per-tensor decision consumes no new environment variable except **`MOE_HOST_LIMIT_FRAC`** (default 80, the
`auto` ceiling as a percentage of `MemAvailable` — deliberately NOT `MOE_HOST_POOL_FRAC`, which is a
percentage of the host EXPERT BYTES and sizes a pool).

## 2. The measurements

`Qwen3.6-35B-A3B-UD-Q4_K_M`, `-ncmoe 40`, `MOE_EXPERT_CACHE_MIB=2048`, `--seed 42 --temp 0 --ignore-eos`,
serial runs, `HIP_VISIBLE_DEVICES=0,2` (both upstream x8 — never `0,1`, that pair contains the x4 card),
prefill = `prompts/prose-rdna-boosts.txt` (5 246 tok, sha256 `fabdec65…`) with `-n 1`.

| 2 devices | prefill | decode (`-n 300`) | pinned / pageable (MiB) | text |
|---|---|---|---|---|
| `-lme auto` (the set fits) | **1105.4** | **71.9** | 18 662 / 0 | `sha=231512fc77e5` |
| `-lme 64` (clamped to the set) | 1106.0 | 71.7 | 18 662 / 0 | `sha=231512fc77e5` |
| `-lme 16` | — | 65.7 | ~16 100 / ~2 500 | `sha=231512fc77e5` |
| `-lme 8` (whole-tensor tier; **now routed to the pool**, see P6) | 505.1 | 50.1 | 8 064 / 10 598 | `sha=231512fc77e5` |
| `-lme 4` | — | 42.4 | ~4 000 / ~14 600 | `sha=231512fc77e5` |
| `auto` **not** fitting (`MOE_HOST_LIMIT_FRAC=1`) | 713.7 | 51.8 | 0 / 18 662 + pool 9 331 | `sha=231512fc77e5` |
| `--host-experts pinned` (shorthand) | — | 71.8 | 18 662 / 0 | `sha=231512fc77e5` |

| 1 device | prefill | decode | text |
|---|---|---|---|
| `-lme auto` | **645.2** | **53.4** | `sha=4ae33f7bd65c` |
| `-lme 64` (clamped to the set) | — | 53.5 | `sha=4ae33f7bd65c` |
| `-lme 8` | 447.1 | 48.4 | `sha=4ae33f7bd65c` |

Acceptance from the plan (§3 a/b, §4): the fit case must reproduce `pinned` within 2 % (1 105.4 / 72.0 at 2
devices, 645.9 / 53.5 at 1) — **met** (1105.4 / 71.9 and 645.2 / 53.4), text **byte-identical** in every pair
at a given device count, `check-gpu-faults` clean on every arm, and the not-fitting band unchanged
(713.7 / 51.8 against the record's 707-708 / 51.3-51.5).

Two more shapes were checked because they exercise different code:

- **A second model.**  Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (`qwen4exp`, `--lazy-mode off`, `-ncmoe 48`): 144
  tensors, **40 925 MiB** pinned by `auto`, decode **47.8** against the record's 47.9 for the same arm.
- **No mmap** (`--load-mode none`, a 22 GB staged load): pinned the whole set, no faults, and the tier is
  correctly a no-op there (there is no mapping to place a hole in).
- **No host experts** (`-ncmoe 0`): the plan does not run at all (no message), 124.7 t/s decode, i.e. inert.

## 3. Finding 1 — the partial band belongs to the POOL, not to per-tensor pinning

The tier behaves exactly as a budget should: decode interpolates linearly between the bounce-only rate and the
pinned rate on the fraction of bytes pinned —

| pinned fraction | decode | linear model `36.8 + (71.9-36.8) x f` |
|---|---|---|
| 0 % (`MOE_HOST_LIMIT_FRAC=1`, mapping + bounce) | 37.4 | 36.8 |
| 21 % (`-lme 4`) | 42.4 | 44.3 |
| 43 % (`-lme 8`) | 50.1 | 51.9 |
| 86 % (`-lme 16`) | 65.7 | 67.0 |
| 100 % (`auto`/`-lme 64`) | 71.9 | 71.9 |

So the mechanism is right, and the cost of a hole is proportional to its bytes.  **But at equal RAM the pool
is the better use of a partial budget**:

| ~equal non-swappable budget, 2 devices | prefill | decode |
|---|---|---|
| 8 064 MiB as pinned whole TENSORS (`-lme 8`) | **505.1** | 50.1 |
| 9 331 MiB as a per-EXPERT pool (pageable master, `MOE_HOST_POOL_FRAC=50`) | **713.7** | **51.8** |

Decode is a tie; prefill is **29 % worse** for the per-tensor tier.  The reason is structural: expert routing
is spread across every layer, so no *whole tensor* is hot — pinning a tensor pins 256 experts of which only a
few are read often, while the pool pins the experts that are actually read.  A partial budget therefore wants
**per-expert** residency, which is what the pool already is.

**Consequence for the policy (and it is the one now implemented):** `auto` is a *binary* decision — pin the
whole set when it fits, else leave everything in the mapping and let the pool do the work.  A sized pinned
master tier is for the two cases where it is unambiguous: "the whole set" (the fit case, which is the bar the
plan set and which the tier now hits exactly) and a user asking for a coarse, tensor-granular cap.  The
mid-band is the pool's, sized with `MOE_HOST_POOL_MIB`, and `-lme 8`'s job is to *bound* the non-swappable
footprint at 8 GiB, not to spend it optimally.

**What that left open** was `IMPLEMENTATION-PLAN.md` step b2 — an expert-indexed pinned region with a presence
bitmap.  **That is now done, and it turned out to be the pool itself**: see [`P6-POOL-REGION.md`](P6-POOL-REGION.md).
The pool already *is* a pinned per-host-tensor region addressed by (expert -> slot) with a presence bitmap; what
it lacked was the right to be a cold-read source away from one GPU (`device_count == 1 && src_off == 0`, both
geometry accidents).  With that enabled the cold-read encoding needs **no** change at all — `id - n_res_slots`
is a slot index into one pinned base, and the pool's base with its slot stride is such a base.  Measured:
+11.5 % decode at `-lme 8` (48.9 -> 54.5) and +10 % for the not-fitting pool (51.8 -> 57.0), byte-identically.
So `-lme N` now spends the budget per EXPERT below 90 % of the host expert set and keeps the whole-tensor
pinned-master tier at or above it (the fit case, where a pinned *master* still wins decode outright).

## 4. Finding 2 — two traps, both silent, both cost a wrong measurement

1. **The `-ncmoe` tail bound is `hparams.n_layer()`, not `hparams.n_layer_all`.**  `n_layer_all` also counts
   the appended nextn/MTP head, so the pre-pass's tail was one layer off: it planned 120 tensors of which one
   was the MTP head (never hosted: `create_tensor` refuses to offload the speculative head) and it missed
   trunk layer 0.  The first measurement read `pinned 18 198.0 MiB of 18 662.0 MiB` — 464 MiB of "pinning"
   that went to VRAM — and looked like a rounding artefact.  The plan is now exact: 18 662 / 18 662.
2. **`moe_cache_set_host_pool()` must re-resolve `g_pool_enabled`.**  The loader's mode decision lands at
   model load, which can be AFTER `parse_env()` has already resolved the pool from the environment.  The
   first version of the not-fitting branch therefore had **no pool at all** and served every hole through the
   per-table bounce ring: 37.4 t/s decode where the pool gives 51.8.  The old code did not hit this because it
   enabled the pool from the loader's CONSTRUCTOR (before any `parse_env`).

Both are the same class of failure as the campaign's earlier ones: a silent fallback that looks like a
performance property of the mechanism under test.

## 5. What was NOT done (and why)

| plan item | state |
|---|---|
| step a (an expert-indexed region with a presence bitmap) | **superseded by the measurements**: at whole-tensor granularity a pinned tensor already IS the region (property 1 above), and the bitmap is only needed for per-expert residency, which is step b2 |
| step b2 / a per-expert region | deferred (finding 1 + the plan's own `[STOP]`); `TODO.md` #139 |
| step c (a `PARTIAL` per-table staging branch) | **collapsed**: the placement is per tensor, so a table is ALL_RESIDENT (pinned master, the existing single-launch branch) or has holes (pageable master, the existing pool/bounce branch).  `host_pinned` is already per table, so no third state exists to implement.  A per-expert region (b2) is what would reintroduce it |
| step d (retire the mode distinction) | done at the CLI: `-lme` is the only knob, `--host-experts pinned` = the whole set, `pool` = `auto` (the plan's mapping).  `LLAMA_HOST_EXPERTS_MODE_POOL` survives only as the *library* spelling of the pageable master, which the loader still honours for a caller that never ran a plan |
| step e (the census) | done (both at plan time and after the load) |
| the delivered `patches/0001-0004, 0006, 0007` | still present in `wip/` as history; none of them is needed by this tier and none is in the deliverable |
| the delivery promotion itself (`release.json`, `WORKLOG`/`INVARIANTS`/`ENVIRONMENT`, block cut, gates) | **not done**: `AGENTS.md` requires the findings to be reported and a decision taken first.  §6 of `IMPLEMENTATION-PLAN.md` lists the promotion checklist; this file is its evidence |

## 6. Reproduce

```bash
cd wip/host-expert-eager-push
P=../../prompts/prose-rdna-boosts.txt
# fit case (the acceptance bar): 2 devices -> prefill 1105.4 / decode 71.9, sha=231512fc77e5
GPU=0,2 SM=tensor EOS=1 N=300            ./p3-matrix.sh auto-dec
GPU=0,2 SM=tensor N=1 PROMPT_FILE=$P     ./p3-matrix.sh auto-pre
# partial budget: 8 GiB of whole tensors, hottest first -> 8064 MiB pinned, 505/50.1
GPU=0,2 SM=tensor EOS=1 N=300 LME=8      ./p3-matrix.sh lme8-dec
# the census (a -lv run has no usable text hash, by design)
GPU=0,2 SM=tensor EOS=1 N=300 LME=8 LV=4 ./p3-matrix.sh lme8-census
# the not-fitting band (pool, unchanged): no pinned master, 713.7/51.8
GPU=0,2 SM=tensor EOS=1 N=300 MOE_HOST_LIMIT_FRAC=1 ./p3-matrix.sh nofit
```

`HOSTMODE=` (empty, the new default) means "no `--host-experts` flag": the `-lme` policy decides.  Set it on
the script invocation line (`HOSTMODE=pinned ./p3-matrix.sh …`) — an assignment after the tag silently does
not reach the script (trap 6.1 of the plan, which cost the earlier campaign its whole pinned column).

Logs: `runs/p3-*-p5-*.log` (26 arms, all of the table above plus the shape checks).
