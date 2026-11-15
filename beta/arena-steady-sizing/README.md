# Arena steady-state settle — the expert arena stops shrinking when `-ub` grows

> **Status: BETA candidate (2026-11-15).** Not in `patches/`, not in the release tree, not covered by the
> delivery's gates. One commit, one file (`ggml/src/ggml-cuda/moe-expert-cache.cu`), against the
> `release.json` tree `8423b2b5eac33f97ad530229b8c35240aa64fa08` (`tip ec721312c`, **r291**).
>
> Why `beta/` and not `wip/`: it edits a file blocks **09/10/15** own, so it cannot be a standalone `wip/`
> artefact — and where it should be *folded* (an amendment to block 09, which owns the re-arm, or a new
> temporary block) is the maintainer's decision, not a gate outcome. See §6.

* patch: `arena-steady-sizing.patch` — sha256 `66af35b2ddb6c28890903d589934045c691be4a09d2d9dd3892d0ca556862f70`
* applied tree (re-verified 2026-11-15 on r291): **`9848ad69ae8b`** — one commit on top of the r291 tree
  (`8423b2b5eac3`), via `scripts/apply-all.sh . --include-beta` on a fresh checkout of
  `release.json.base` (`4f9dfa0e3`).  `release.json` is unchanged; folding this into a block (§6) is what
  changes `.tip`/`.tree`.  (Earlier cut against r288 gave applied tree `aa7e09dabe32`; the fix and its
  numbers are identical on both — it re-applied verbatim to r291.)
* measured on: 1 × R9700 (`gfx1201`, 32 624 MiB, PCIe 5.0 x16, 93 GiB RAM) **and** 1 × RX 7900 XTX
  (`gfx1100`, 24 560 MiB, **PCIe 4.0 x16**, 61 GiB RAM), ROCm 10.1, Qwen3.8-Flash-Next `GSQ-RCO-IQ3_S`,
  `-ub 16384`, `--parallel 1`, 4 k prompt + **2 000** generated tokens

---

## 1. The bug

`-ub` grows → the expert arena gets **permanently smaller**, and nothing in the log says so.

The arena is sized **once**, from `ggml_cuda_slab_arena_total()` = `arena_top - boundary` sampled at the
moment the cache sizes (`alloc_all_locked`, `moe-expert-cache.cu`). That moment is while the **wide
(prefill) compute view is live**: the slab's birth `boundary` is the widest `--fit`/`sched_reserve` layout,
i.e. `narrow + ring + wide`, so the figure is ubatch-proportional. `g_sized = true` then latches it, and
`moe_cache_rearm` only ever restores the frozen per-layer count — TODO #70's "a deficit layer is never
revisited" rule is correct for an arena that genuinely shrank and wrong for the band a
prefill→decode drop hands back.

Measured loss = `max(0, boundary_at_birth - narrow_floor)`, on all four rows to within one arena unit:

| `-ub` | birth boundary | narrow floor | arena region post-drop | arena the tables were sized to | **thrown away** |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 1 600 MiB | 2 048 | 23 744 MiB | 23 908.0 | **0** (saturated) |
| 8 192 | 3 200 | 2 048 | 23 616 | 22 167.6 | **1 448** (6 %) |
| 16 384 | 6 656 | 2 048 | 23 360 | 18 411.9 | **4 948** (21 %) |
| 32 768 | 12 928 | 2 048 | 22 848 | 11 633.4 | **11 215** (49 %) |

It is not the phase and not the sizing *timing*: `MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` gives byte-identical
results (the slab already exists, wide, before any sizing can run), and an explicit
`MOE_EXPERT_CACHE_MIB=24000` is **silently free-clamped** — the exit census then reads
`requested=24000 MiB/device, free-clamped=1` next to `arena=18137.1 MiB total (budget 24000 MiB/device)`
while the cache ran `evictions=10781`. A user who raises `-ub` and loses hit rate has no signal.

## 2. The change

At each prefill→decode drop (where the narrow re-reserve has just shrunk the boundary back to
`narrow_floor`, so `arena_total()` **is** the decode steady-state arena), settle the per-layer slot count
by scaling the sizing count by the capacity ratio:

```
ratio = min( avail_now / avail_sizing ,  request / avail_sizing )   // clamped to >= 1 for growth
want  = sizing_slots x ratio                                        // capped at the layer's n_experts
```

* **No table scan.** The sizing's inputs are recorded once (`g_size_slots / _avail / _request / _hold`),
  so the settle cannot disagree with the sizing about which tables count (a head table, the per-table
  arena-unit rounding) — disagreeing is how a **non-uniform layer**, the one state that reads the wrong
  expert, gets built.
* **Growth only**, so a genuinely smaller arena keeps the sizing pass's answer and this cannot oscillate.
* **`request` is the pre-clamp budget**: `auto` grows with the capacity; an explicit
  `MOE_EXPERT_CACHE_MIB` below the capacity keeps the operator's count. (Recording the *post*-clamp budget
  was the first attempt and it silently disabled the whole thing — for `auto` the clamp makes
  `request == avail`, so the ratio pinned to 1.0.)
* Layers that survived the prefill intact are grown too; a table already at its full expert count cannot
  grow, so the pinned MTP head drops out without needing a head predicate (that predicate lives in block
  15, which is why the fold target matters — §6).
* Kill-switch: `MOE_EXPERT_CACHE_STEADY_REARM=0` (documented in `ENVIRONMENT.md`).

## 3. Measurements

Interleaved A/B, two separate builds (`llama-server` + its `.so`s snapshotted per arm), two cycles each,
same prompt/seed, `MOE_EXPERT_HEATMAP=load` so the prior is frozen:

| host tier | arm | arena | hit | **decode t/s** | **Δ** |
|---|---|---:|---:|---:|---:|
| 100 % mirror (all 46.9 GiB pinned) | base | 18 411.9 MiB | 0.8337 | 51.50 / 51.39 | — |
| | **settle** | **22 705.1 MiB** | **0.8866** | **53.56 / 53.52** | **+4.0 %** |
| same, re-measured on the **r291** tree | base | 18 411.9 MiB | 0.8337 | 51.40 | — |
| | **settle** | **22 705.1 MiB** | **0.8866** | **53.59** | **+4.3 %** |
| partial pool (`-lme 8`, `DIO=1`) | base | 18 411.9 MiB | 0.8515 | 18.30 / 18.29 | — |
| | **settle** | **22 705.1 MiB** | **0.8940** | **23.60 / 24.01** | **+29 %** |

**Expert miss traffic** (the hardware-independent figure — the same bytes on any link):

| arm | expert fills | cold reads | total | Δ |
|---|---:|---:|---:|---:|
| mirror base | 40 735 MiB | 261 459 | 302 194 MiB | — |
| mirror settle | 35 794 | 168 612 | **204 406 MiB** | **−32 %** |
| pool base | 175 561 | 95 838 | 271 398 | — |
| pool settle | 113 205 | 79 430 | **192 635 MiB** | **−29 %** |

So the fix removes ≈ **29-32 % of the expert miss traffic**, and how much wall time that buys depends
only on whether the expert path is the bottleneck: ~13 % of the wall on this box's mirror tier (+4.0 %),
the whole wall on a partial pool (+29 %). A PCIe4 x16 card with a real mirror sits between; the portable
estimate is `49 MiB/token saved ÷ link rate`, i.e. ~2 ms/token ≈ **+10 %** at 25 GB/s.

### Validated on the target hardware — RX 7900 XTX, 24 GiB (`gfx1100`, PCIe 4.0 x16)

Run on the maintainer's 24 GiB box at its own field `runme` (QFN `GSQ-RCO-IQ3_S`, `-ncmoe 48`, 61 GiB RAM so
`-lme auto` resolves to a **mirror**, `-ub 16384`, ctx 32 768, **2 000** generated tokens, 2 interleaved
cycles, heat prior pinned). `b1f559df7` tree, patch applied on top:

| | base | settle | Δ |
|---|---:|---:|---:|
| slab (work + arena) | 14.62 GiB (6.50 + 8.12) | same | — |
| arena sized at the drop | 7 962.5 MiB (16.6 %) | **12 888.2 MiB (26.9 %)** | **+62 %** |
| arena hit | 0.6038 / 0.6039 | **0.7337 / 0.7337** | +13 pp |
| host pool (L2) hit | 0.9849 | 0.9962 | |
| expert fills | 128 433 | 88 487 | −31 % |
| expert cold reads | 1 043 443 | 682 850 | −35 % |
| **expert miss traffic** | **757 391.7 MiB** | **500 773.0 MiB** | **−33.9 %** |
| **decode** | **45.20 / 45.68 → 45.44 t/s** | **53.33 / 54.85 → 54.09 t/s** | **+19.0 %** |
| greedy text | `sha 4df76b729a6ba020` | **identical** | — |

(Measured on the `b1f559df7` tree = the release before r291; **re-measurement on the r291 tree on that host is
pending** — it needs one `apply-all.sh` + two builds there and was not completed.  The change itself is
tree-independent: it re-applied verbatim to r291 and the 32 GiB numbers below were re-measured on r291.)

The defect there is `6.50 - 1.875 ≈ 4.6 GiB` of an **8.12 GiB** arena, i.e. the fix recovered essentially the
whole region (12 888 of a ~12.7 GiB post-drop region).

**This is the scaling that matters.** The same −32..−34 % traffic reduction is worth **+4.0 %** on the 32 GiB
box (arena 23 GiB) and **+19.0 %** on the 24 GiB box (arena 8 GiB), because the budget the arena gets is what
decides whether the expert path is the bottleneck: base miss traffic on the 24 GiB box is 379 MiB/token
(≈17 GB/s of effective read at 45 t/s, i.e. the card *is* link-bound), and the settle removes 128 MiB/token
of it. **Arena capacity is worth roughly `1 / arena_size`** — so smaller VRAM cards gain most, which is
exactly where the defect is worst (49 % of the arena lost at `-ub 32768`).

Two snapshot builds are left on that host as `/tmp/tree-base/bin` and `/tmp/tree-fix/bin` (drop-in
`SERVER=` arguments to its `runme`), so the measurement can be repeated without rebuilding.

**Correctness:** same-seed greedy text is **byte-identical** (`sha256 af0ec4edd999a414`, 489 chars,
`--temp 0 --seed 42`, the settle provably firing: arena 18 411.9 → 22 705.1 MiB in the same run). The
extra slots are *capacity*: they start empty and fill on demand, so the bytes served for a given expert
and their order are unchanged. This is the result that makes the change shippable rather than a
fidelity change.

### Methodology note (worth keeping)

The first attempt measured **128** generated tokens and read +1.8 % on the mirror tier. At **2 000** it is
+4.0 %: the arena only reaches its steady state after a few hundred tokens, so a short decode understates
residency wins. **Use ≥ 1 000 generated tokens, 2 000 preferred, for any arena/residency measurement** —
and note the earlier 128-token number is what made the whole defect look unimportant at first.

## 4. What it does *not* do

* It does not make the arena bigger than the decode steady state can hold — the `narrow_floor` is still
  reserved, and a pinned `narrow2` (MTP draft compute) is still at the top of the VA.
* It does not touch the fill transport, the seed policy (`heat_seed_above_floor`), or the eviction policy.
* It does not shrink anything: `MOE_EXPERT_CACHE_STEADY_REARM=0` is the exact pre-change behaviour.
* It does not fix the *other* half of the symptom: an explicit `MOE_EXPERT_CACHE_MIB` that is clamped is
  still reported inconsistently at the exit census (the `arena settle:` line added here at least names the
  effective figure; reconciling `requested` vs `arena total` is a separate, smaller change).

## 5. Gates to run before promotion

| gate | why |
|---|---|
| `scripts/validate-set.sh` | the patch must apply strictly to the release tree |
| same-seed greedy text, `MOE_EXPERT_HEATMAP=load`, both arms | **done here** — byte-identical |
| the MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) | the arena serves the verify band; MTP-on is the field config |
| `scripts/gate-host-pool.sh` | the fill path is exercised harder (bigger re-arm) |
| `llama-batched-bench -npl 1,4,8` | more residency must not cost slots |
| depth-varied decode (2 k / 16 k / positional 16 k) | the win is residency-shaped; a shallow-only arm will understate it |
| `MOE_EXPERT_CACHE_STEADY_REARM=0` A/B | proves the switch restores the old behaviour exactly |

Not run yet: everything except the text check and the throughput A/B in §3.

## 6. The open decision: where it folds

The patch touches one file that three blocks own. Three options:

* **A — amend block 09** (host-resident MoE: the re-arm). Semantically right. Costs a chain rewrite
  (09-15 all get new trees/ids) because the sizing loop's context at 09 predates block 15's GTT branch —
  the patch does **not** apply cleanly there as cut, so the hunks need hand-adapting.
  *Note the settle as written needs no block-15 code — `table_pinned_head` is avoided by skipping tables
  already at their full expert count.*
* **B — amend block 15** (the last block that owns the file, and the one with the GTT/pinned-head context
  already in place). One commit rewritten, no chain cascade. Poor semantic fit (block 15 is the APU GTT
  store).
* **C — a new temporary block `0301`** in the `03xx` range (default-applied, not counted in `v<N>`).
  Cleanest mechanically, but a new block is the maintainer's call.

Recommendation: **A**, done at the next re-base rather than as an urgent rewrite, with the patch staged
here meanwhile so it can be tested by anyone.
