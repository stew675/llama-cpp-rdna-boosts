# Arena steady-state settle — the expert arena stops shrinking when `-ub` grows

> **Status: PROMOTED into block 09 at r292 (2026-11-15).** The final v2 cut was amended into **block 09**,
> not a new block (maintainer decision: no `0301`); the default set is still 17 patches and the r292 release
> tree is **`4a67e7c9d7b1925081db04a814f03ae40627325f`**. This record is the candidate as it was tested: the
> patch/applied-tree lines below name the pre-fold beta cut (`9cd799ca`); the promoted tree differs from it
> only in comment/line wrapping and a redundant scope, and the behaviour is identical.
> One commit, one file (`ggml/src/ggml-cuda/moe-expert-cache.cu`), cut against the r291 tree
> (`8423b2b5eac3`, `tip ec721312c`).
>
> **What v2 changed.** The v1 candidate settled unconditionally at every drop. That is a loss on a
> saturated split: a 2-device `-sm tensor` split already holds 437/512 experts per table and covers
> 99.6 % of the routed mass, so taking the band bought **0.44 pp** of decode and cost a whole-table
> prefill eviction (measured prefill 1625 -> 903 t/s, decode -5 %). v2 decides at the **first drop** on
> the **marginal mass the growth actually buys** (`MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP`, default 1.0 pp)
> and keeps the settled count **sticky per layer with the capacity it was achieved at**, so a two-slot
> decode/prefill transition cannot ratchet the count down and oscillate. The 1-device and 24 GiB wins
> are unchanged; the saturated split and a small `-ub 2048` growth are now inert.
>
> Why `beta/` and not `wip/`: it edits a file blocks **09/10/15** own, so it cannot be a standalone `wip/`
> artefact — and where it should be *folded* is the maintainer's decision, not a gate outcome. See §6.

* patch: `arena-steady-sizing.patch` — sha256 `dcbfce70c04ad6cb0d819ba92fcd1831e8123e757d3f1570a7f63f8b675362b6`
* applied tree: **`9cd799ca3e86435eacd1d1b8e9492f1130bc0354`** — one commit on top of the r291 tree
  (`8423b2b5eac3`), verified by `scripts/apply-all.sh --include-beta` on a fresh checkout of
  `release.json.base` (`4f9dfa0e3`). `release.json` is unchanged; folding this into a block (§6) is what
  changes `.tip`/`.tree`.
* measured on: 1 × R9700 (`gfx1201`, 32 624 MiB, PCIe 5.0 x16, 93 GiB RAM) **and** 1 × RX 7900 XTX
  (`gfx1100`, 24 560 MiB, **PCIe 4.0 x16**, 61 GiB RAM), ROCm 10.1, Qwen3.8-Flash-Next `GSQ-RCO-IQ3_S`,
  `-ncmoe 47/48`, `--parallel 1`, a 5 246-token prompt, **2 000** generated tokens, greedy.

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
by scaling the count the layer holds by the capacity ratio:

```
ratio = min( avail_now / ref ,  request / avail_sizing )   // clamped to >= 1 for growth
want  = held x ratio                                        // capped at the layer's n_experts
```

* **No table scan.** The sizing's inputs are recorded once (`g_size_slots / _avail / _request / _hold`),
  so the settle cannot disagree with the sizing about which tables count (a head table, the per-table
  arena-unit rounding) — disagreeing is how a **non-uniform layer**, the one state that reads the wrong
  expert, gets built.
* **Demand-gated at the first drop.** The sizing pass only knows the wide capacity; the first drop is the
  first moment the decode capacity is visible. v2 computes the **marginal** prior mass the offered growth
  actually buys (the persisted decode-band prior, the vector the arena seeds from) and takes the band only
  when that is at least `MOE_EXPERT_CACHE_STEADY_MIN_GAIN_PP` (default **1.0 pp**). This is the difference
  between v1 and v2: taking a 1.9 % growth at `-ub 2048` bought 0.5 pp and cost a whole-table re-seed
  (measured -2.7 % decode), and on a 2-device tensor split the extra 75 slots per table buy 0.44 pp while
  costing a whole-table prefill eviction. Both now leave the band alone and are byte-identical to base.
* **Sticky, multi-session safe.** Each layer remembers the count it actually achieved and the capacity it
  achieved it at. A drop that returns a smaller arena (a second sequence's wider verify band, a live
  prefill) re-arms at the achieved count — it never ratchets **down** — and only a capacity that **grows**
  past that reference re-derives it. The per-layer count is uniform across the layer's tables and across
  `-sm tensor` devices; the re-arm still converges or stands the whole layer down, and it only runs at a
  compute-buffer drop with no graph in flight (a re-allocation bumps the graph-mem generation).
* **Growth only**, so a genuinely smaller arena keeps the sizing pass's answer and this cannot oscillate.
* **`request` is the pre-clamp budget**: `auto` grows with the capacity; an explicit
  `MOE_EXPERT_CACHE_MIB` below the capacity keeps the operator's count.
* Kill-switch: `MOE_EXPERT_CACHE_STEADY_REARM=0` (documented in `ENVIRONMENT.md`) is bit-for-bit the old
  behaviour.

## 3. Measurements

Interleaved A/B, same prompt/seed, `MOE_EXPERT_HEATMAP=load` so the prior is frozen, `ROCBLAS_USE_HIPBLASLT=0`,
**2 000** generated tokens. Text is from a no-`-lv` run; the census from a `-lv 4` run. All gaps are
byte-identical greedy text (`sha eb2816edc8e6` on the 1-device arms, `4ff95271242c` on the 24 GiB box).

### 1 × R9700 (32 GiB), the band is worth taking

| arm | arena | hit | **decode t/s** | **Δ** | miss traffic |
|---|---:|---:|---:|---:|---:|
| base `-ub 16384` | 18 411.9 MiB | 0.8830 | 41.59 / 41.84 | — | 211 498.5 MiB |
| **settle** | **22 819.7 MiB** | **0.9297** | **45.17 / 45.11** | **+8.2 %** | **129 117.3 MiB (-39 %)** |
| base `-ub 32768` | 12 183.0 | 0.7647 | 34.33 / 34.33 | — | 425 828.6 |
| **settle** | **23 018.0** | **0.9307** | **45.23 / 45.31** | **+31.9 %** | **126 134.3 (-70 %)** |
| base partial pool (`-lme 8`, `DIO=1`) | 18 411.9 | 0.8809 | 16.89 / 16.89 | — | 216 256.0 |
| **settle partial pool** | **22 819.7** | **0.9260** | **23.10 / 23.10** | **+36.8 %** | **136 033.9 (-37 %)** |

The `-ub 32768` row is the defect at its worst (49 % of the arena was thrown away) and the settle
recovers essentially all of it.

### 24 GiB RX 7900 XTX (`gfx1100`, PCIe 4.0 x16, mirror), the class that gains most

At its field `runme` shape (QFN `GSQ-RCO-IQ3_S`, `-ncmoe 48`, `-ub 16384`, ctx 32 768), re-measured on the
**r291** tree, 2 interleaved cycles:

| | base | settle | Δ |
|---|---:|---:|---:|
| arena at the drop | 11 428.6 MiB | **15 995.3 MiB** | **+40 %** |
| arena hit | 0.7418 | **0.8412** | +10 pp |
| expert miss traffic | 479 658.3 MiB | **296 483.2 MiB** | **-38.2 %** |
| **decode** | **36.16 / 36.84 → 36.50 t/s** | **43.66 / 43.56 → 43.61 t/s** | **+19.5 %** |
| greedy text | `sha 4ff95271242c` | **identical** | — |

Arena capacity is worth roughly `1 / arena_size`: the same fix is +8.2 % on a 23 GiB arena and +19.5 %
on a 16 GiB one. **Validate this class on the smallest card available.**

### The gate leaves the band on a saturated or tiny-growth config (byte-identical to base)

| config | first-drop growth | marginal prior gain | decision | decode base vs settle |
|---|---:|---:|---|---|
| 2-device `-sm tensor`, `-ub 16384` | 24.7 % of the arena (437 -> 512 slots/table) | **0.44 pp** | **DISABLED** | 51.90 vs 51.90 |
| 2-device `-sm layer`, `-ub 16384` | 9.7 % | **0.14 pp** | **DISABLED** | 39.34 vs 39.37 |
| 3-device `-sm tensor`, `-ub 16384` | 0 % (already full) | **0.00 pp** | **DISABLED** | 57.45 vs 57.51 |
| 1-device, `-ub 2048` (batched bench) | 1.9 % | **0.50 pp** | **DISABLED** | 49.61 vs 49.56 (B=1, ntg 2000) |

This is the v1 regression closed: the 2-device tensor split's prefill was 1625 -> 903 t/s (-44 %) and its
decode -5 % when the band was taken. The whole-table eviction is the mechanism: the band is filled by the
lowest-VA tables, and a prefill boundary move stands those tables down **whole**, so the prefill re-reads
every expert of them from the host instead of the 85 % that were resident.

### MTP four-axis (field config, 1 × R9700, `draft-mtp` n3, 2 000 tokens)

Text and draft acceptance are **identical** on/off on every axis; the settle is decode-positive where the
band is taken:

| axis | acceptance on/off | mean len | decode base -> settle |
|---|---|---|---|
| R (reasoning) | 0.62267 = 0.62267 | 2.87 | 61.8 -> 70.2 t/s (+13.6 %) |
| P (prose) | 0.82563 = 0.82563 | 3.48 | 65.2 -> 75.6 (+15.9 %) |
| C (code) | 0.91975 = 0.91975 | 3.76 | 70.6 -> 80.3 (+13.7 %) |
| K (recall) | 0.99200 = 0.99200 | 3.98 | 80.8 -> 83.1 (+2.8 %) |

`llama-batched-bench -npl 1,4,8` at the gate's `-ub 2048`: B=1 47.89 -> 48.04, B=4 127.36 -> 127.67,
B=8 176.52 -> 176.61 t/s (ntg 512; the settle is disabled there). `scripts/gate-host-pool.sh` **54/54**,
`scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_CACHE_STEADY_REARM=0"` **KLD 0.000000 / top-p 99.990 %**,
no GPU faults, same-seed greedy text byte-identical in every arm.

### Two-slot server — the transitions the field actually sees

`llama-server --parallel 2`, 1 × R9700, `-ub 16384`, one slot decoding while the other prefills, then the
prefill completes to decode, then the short slot **starts a new prefill** while the long one still decodes,
then a fresh prefill after both are idle. Three runs: all four responses coherent, no
`could not be re-armed slot-uniformly`, no `is not slot-uniform`, no stood-down layer, no GPU fault. The
settle fires **once** (47 layers, arena 22 388.2 MiB at the two-slot narrow floor) and the later drops
re-arm at the achieved count without re-growing or shrinking. Concurrent text is deterministic in this
harness on this box; the repo does not require that (KV placement can change it).

**Methodology note (worth keeping).** A short decode understates every residency change: the same fix
read +1.8 % at 128 generated tokens and +8.2 % at 2 000, and a growth that is only a few slots still pays a
whole-table re-seed. **Use >= 1 000 generated tokens, 2 000 preferred, and judge the growth's marginal
mass, not full residency.**

## 4. What it does *not* do

* It does not make the arena bigger than the decode steady state can hold — the `narrow_floor` is still
  reserved, and a pinned `narrow2` (MTP draft compute) is still at the top of the VA.
* It does not **partially** evict a table: the slab's arena is still per-table atomic, so a prefill that
  takes the band stands those tables down whole. That is why the 2-device split leaves the band: taking it
  would make the next prefill re-read whole tables. A two-segment table (stable hot slots + a transient
  cold extension in the band) would recover that space without the prefill cost, and is a design change of
  its own.
* It does not grow mid-decode: the decision and the re-arm happen at a compute-buffer drop. A single
  prefill followed by a decode that runs to completion is the case the first drop already handles.
* It does not shrink anything: `MOE_EXPERT_CACHE_STEADY_REARM=0` is the exact pre-change behaviour.
* It does not fix the *other* half of the symptom: an explicit `MOE_EXPERT_CACHE_MIB` that is clamped is
  still reported inconsistently at the exit census (the `arena settle:` line names the effective figure;
  reconciling `requested` vs `arena total` is a separate, smaller change).

## 5. Gates

| gate | status |
|---|---|
| same-seed greedy text, `MOE_EXPERT_HEATMAP=load`, on/off, every arm | **DONE — byte-identical** |
| `scripts/gate-host-pool.sh` | **DONE — 54/54, no faults** |
| `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_CACHE_STEADY_REARM=0"` | **DONE — 0.000000 / 99.990 %** |
| MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) | **DONE — text and acceptance identical** |
| `llama-batched-bench -npl 1,4,8` | **DONE — neutral at the gate's `-ub 2048`** |
| 2-slot server, both transition directions | **DONE — coherent, no non-uniform layer, stable arena** |
| 1-device `-ub 16384` / `-ub 32768` / partial pool, 24 GiB `gfx1100` | **DONE — +8.2 / +31.9 / +36.8 / +19.5 %** |
| `scripts/validate-set.sh`, `gate-per-block-build.sh`, `check-gpu-faults.sh` | **once it is in `patches/`** |

## 6. The open decision: where it folds

The patch touches one file that three blocks own. Three options:

* **A — amend block 09** (host-resident MoE: the re-arm). Semantically right. Costs a chain rewrite
  (09-15 all get new trees/ids) because the sizing loop's context at 09 predates block 15's GTT branch —
  the patch does **not** apply cleanly there as cut, so the hunks need hand-adapting.
* **B — amend block 15** (the last block that owns the file). One commit rewritten, no chain cascade.
  Poor semantic fit (block 15 is the APU GTT store).
* **C — a new temporary block `0301`** in the `03xx` range (default-applied, not counted in `v<N>`).
  Cleanest mechanically, but a new block is the maintainer's call.

Recommendation: **A**, done at the next re-base rather than as an urgent rewrite, with the patch staged
here meanwhile so it can be tested by anyone. **The maintainer owns the block structure; this is not a
gate outcome.**

**RESOLVED (r292): A — amended block 09, no `0301`.** The maintainer chose block 09. The block 09 commit
now carries the settle (demand gate, sticky per-layer count) and the 09-15 chain was re-cut; the release
tree is `4a67e7c9...`. The feature is now part of the default set.
