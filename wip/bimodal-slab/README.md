# A strictly bimodal slab: prefill band **or** expert arena, never both — viability assessment

> **Status: SCOPING NOTE — 2026-11-15, revised the same day: S is **promoted into block 09** at r292 (`archive/work/arena-steady-sizing/`).**
> This is a measurement-backed feasibility assessment of one proposal, plus the numbers that decide it.
> Bimodality itself is not implemented and nothing of it is in `patches/`; the **sizing fix (S)** that the
> assessment found is a beta candidate (one file, one commit, demand-gated — see §3.5's revision and the
> candidate's README). The early `+1.8 % / +4 %` figures below were measured at **128 generated tokens**;
> the 2 000-token re-measurement is **+8.2 % at `-ub 16384` and +31.9 % at `-ub 32768`**, and the
> 2-device/multi-session gate is in the candidate's README.
>
> **The proposal.** On a `--parallel 1` server the expert arena and the prefill compute band never need
> to hold VRAM *at the same time*. So treat the slab strictly **bimodally**: when prefill is active,
> evict the arena completely and spend the bytes on the widest ubatch the card can take (up to 32 768);
> when the prefill ends, drop the prefill band and reload the evicted experts in one modal burst from a
> host mirror. Motive: small-VRAM cards, where the arena's *metadata-free* share of VRAM is what makes
> large ubatches impossible.
>
> **What the assessment found instead (§3.5).** The bimodality mechanism already exists and is already
> engaged; the symptom that the proposal is really chasing — *the arena getting smaller as `-ub` grows* —
> is a **sizing defect, not a policy gap**: the arena is sized **once**, against the slab's *birth*
> boundary, which is the widest `--fit` probe layout and therefore ubatch-proportional. The band the
> prefill→decode drop hands back is never given to the tables. Measured: **4.9 GiB (21 %) of the arena
> region is permanently unusable at `-ub 16384`, and 11.2 GiB (49 %) at `-ub 32768`** — and an explicit
> `MOE_EXPERT_CACHE_MIB=24000` is silently free-clamped to the same figure. This is a much smaller, much
> cheaper, much higher-value target than a bimodal mode.
>
> All measurements below are on the maintainer's single-GPU box: 1 x R9700 (`gfx1201`, 32 624 MiB),
> 16-core, 93 GiB RAM, build `b12386 (14f3749be)` = delivery r288 tip, ROCm 10.1 (`gfx120X`),
> Qwen3.8-Flash-Next `GSQ-RCO-IQ3_S` + `--n-cpu-moe 47`. Reproduction commands in §10.

---

## 0. Verdict in one paragraph

**The mechanism you are asking for already exists and is already engaged — what is missing is not a
mechanism but a *policy*, and on top of that there is a plain sizing defect that is doing most of the
harm.** The slab's movable boundary already evicts the arena, whole tables at a time, when the prefill
band needs the bytes (`moe_cache_evict_slab_range`), and the drop already re-arms those tables and
**eagerly bulk-fills them from the pinned mirror at 17-33 GB/s** (`moe_cache_rearm` ->
`alloc_table_locked` -> `apply_heat_seed_locked`). At `-ub 32768` the current code *surrenders the slab to
prefill on its own*: measured arena **1 024 MiB of a 24.3 GiB slab**, i.e. it is already 96 % bimodal by
accident. Three things stop it being bimodal *by design*, in descending order of value:

1. **The arena's slot count is sized once, against the slab's birth boundary, and never re-settled
   (§3.5).** The birth boundary is the widest `--fit` probe layout, which is ubatch-proportional — so a
   bigger ubatch permanently shrinks the arena, and the bytes the drop returns are never handed to the
   tables. **Free to fix, and worth ≈ +1.8 % decode at `-ub 16384` / ≈ +4 % at `-ub 32768` when measured
   (§3.5) — a real defect with a modest throughput value, because on a 100 % mirror pool a miss is a
   local read.**
2. A handful of tables is deliberately left empty in the band the next prefill will reclaim
   (`heat_seed_above_floor()`), which is the "we only refill on demand" you are seeing.
3. The evicted bytes are never a *policy* decision, only an arithmetic consequence.

And the reason to be careful about (3): the on-demand refill is **mostly free**, because it hides behind
decode compute — measured 44.6 t/s over a 128-token decode against 27.7 t/s over an 8-token one on the
same evicted arena. A modal reload moves the same bytes at the same link rate; it does not remove them.
**So the phase question is a capacity question, and the capacity is being thrown away.**

---

## 1. The proposal is three separable questions

Worth separating, because they have different answers, different costs and different owners:

| # | question | answer | evidence |
|---|---|---|---|
| **Q1** | Can the arena yield *everything* so a prefill can have the whole slab? | **Yes, and it already does at large `-ub`** — it is the boundary's normal behaviour. Only the *decision* to do it is missing. | §3.1 |
| **Q2** | Can the evicted experts be reloaded in one modal burst instead of on demand? | **Yes, and 90 % of it already is** — the re-arm bulk-fills heat-ranked slots eagerly. The lazy remainder is a deliberate, documented rule. | §3.2 |
| **Q3** | Is ubatch 32 768 reachable on a 16 GiB card? | **No — arithmetically impossible for this model, at any arena size.** The prefill band at `-ub 32768` is 23.3 GiB; the wide compute buffer alone is 12.6 GiB. | §4 |

Q1 and Q2 are policy; Q3 is arithmetic. The interesting proposal is Q1+Q2 on a card where Q3's ceiling
actually bites — i.e. 16-24 GiB, not 32 GiB.

---

## 2. Today's mechanism (what is already built)

The important background for anyone picking this up: **this is not a new allocator.** Every piece the
proposal needs exists; what does not exist is a *mode*.

### 2.1 The boundary move is a free eviction

`ggml-cuda-slab.cu` keeps one VMM reservation per device split by a movable `boundary`:
`work = [0, boundary)`, `arena = [boundary, mapped)`. Growing the work region is a boundary move inside
already-mapped VA — no `cuMemCreate`/`cuMemMap`/`cuMemUnmap` at runtime. Tables whose arena lies in the
taken range are dropped whole by `moe_cache_evict_slab_range`
(`moe-expert-cache.cu:8187`), which stands the table down recursively until the compute-buffer drop
re-arms it. `ggml_cuda_graph_mem_freed()` is bumped, because a captured graph bakes arena addresses.

### 2.2 There is already a prefill band, with named parts

`ggml_cuda_slab_work_alloc` (`ggml-cuda-slab.cu:1253`) places three things, in this order, above each other:

| part | what | sized by |
|---|---|---|
| `narrow_floor` | the always-resident decode/verify compute view | the narrow re-reserve (`ggml_cuda_slab_set_narrow_floor`, from a `split_only` probe in `llama-context.cpp:2706`) |
| `ring_region` | the H2D staging ring (prefill expert uploads) | `common.cuh:2224` `h2d_stage_region_bytes`: **`1800/n + ub/(1.7 n^2)` MiB**, capped at ¼ of the smallest device's VRAM |
| `wide` | the prefill compute view | the `pp` graph itself, `~0.42 MiB/token` on this model |
| `wide_aux` | bump-allocated prefill transients (FA staging) | demand |

`wide` and `ring` are dropped on the prefill→decode transition (`llama-context.cpp:2657` `drop_now`,
`:2764` `ggml_backend_sched_drop_buffers`, `ggml_cuda_slab_ring_set(false)`); the boundary then shrinks
back to `narrow_floor` on the next narrow reserve.

### 2.3 The reload path exists, and it is eager and batched

`moe_cache_rearm()` (`moe-expert-cache.cu:8040`, called from `llama-context.cpp:2853`, *after* the drop)
re-arms every stood-down table **layer-uniformly** (`moe_cache_rearm_layer_locked`, TODO #70: a mixed-slot
layer silently reads the wrong expert, so a layer converges on one slot count or stands down whole).
Each re-armed table goes through `alloc_table_locked` (`:6141`), which ends with

* `apply_heat_seed_locked` (`:5256`) — rank all `n_experts` by the persisted/prefill heat prior, then
  **bulk-fill the top `min(slots, n_experts)` into the arena**, using the placement read queue
  (`fill_slots_read_locked`, `:4655`) where available and per-slot `fill_slot_from_host` otherwise, then
  one `cudaDeviceSynchronize`.

That **is** the modal reload. Measured rate at the drop, on a complete pinned mirror: **17-33 GB/s**
(`apply_heat_seed_locked: ... 175.2 MiB in 6 ms (29605 MiB/s)`).

### 2.4 What is deliberately *not* reloaded — and why that is what you are seeing

`heat_seed_above_floor()` (`:5082`) returns false for a table whose arena sits inside
`[narrow_floor, narrow_floor + ring + wide)` — "the band a wide prefill immediately reclaims". Those
slots are allocated and left **empty**, to fill lazily on first miss during decode
(`ggml_cuda_slab_seed_floor`, `ggml-cuda-slab.cu:384`). The comment states the intent: *"the seed would
be thrown away on the very next prefill."*

That is a rational rule for a *repeated-prefill* workload and the wrong one for a decode-dominated
steady state — and it is exactly the behaviour the proposal wants to invert. **It is a one-function
policy change, not a mechanism change.**

---

## 3. Measured behaviour on the field shape

### 3.1 The prefill band and what it costs the arena

`llama-cli`, `-ncmoe 47`, `-c 2*ub+512`, a prose prompt of ~2·ub tokens, `GGML_CUDA_SLAB_STATS=1`:

| `-ub` | narrow | ring | wide | **prefill band** (= boundary at the peak) | arena left | arena at sizing |
|---:|---:|---:|---:|---:|---:|---:|
| 4 096 | 2 048 | 4 224 | 2 048 | **8 320 MiB** (8.1 GiB) | 17 472 | 23 908 (51.0 %) |
| 8 192 | 2 048 | 6 656 | 3 776 | **12 480 MiB** (12.2 GiB) | 13 184 | 22 168 |
| 16 384 | 2 048 | 8 192 | 7 232 | **17 472 MiB** (17.1 GiB) | 7 936 | 18 412 (39.3 %) |
| 32 768 | 2 048 | 8 192¹ | 13 632 | **23 872 MiB** (23.3 GiB) | **1 024** | 11 633 |

¹ the ring auto-cap (¼ of VRAM = 8 156 MiB) binds from `-ub 16384` up.

Three things to take from this table:

1. **`wide` is ~0.42 MiB/token** — the only part that scales purely with ubatch on this model.
2. **The ring is 47 % of the prefill band at `-ub 16384`** and is *prefetch depth*, not a requirement.
   It is the single largest discretionary term, and shrinking it shrinks the eviction 1:1.
3. **At `-ub 32768` there is 1 024 MiB of arena left of a 24.3 GiB slab.** The code already behaves
   bimodally there. "Strictly bimodal" would change that 1 024 to 0 — a rounding error, not a mechanism.

### 3.2 The per-turn churn (the actual target, `--parallel 1`, `-ub 16384`)

`llama-server --parallel 1`, four turns (16 k / 24 k / cached / 16 k prompts, 8-128 tokens out):

```
SLAB_STATS wide-arm   arena=23 360 MiB   (boundary 1 792)
SLAB_STATS wide-drop  arena= 7 744 MiB   (boundary 17 408)   <- 15.1 GiB left the arena
moe_cache_rearm:      re-armed 81 stood-down tables (arena now 18 137.1 MiB)
                        ^ 81 of 141 tables stood down; 6 of them heat-seeded eagerly
MoE arena = 0.4630 (6 126 hit / 13 230 reaches), arena 18 137.1 MiB     [first 8 decode tokens]
...next turn...
MoE arena = 0.5776        ...next turn (cached)...
MoE arena = 0.8435 (151 812 hit / 179 980 reaches)   [128-token decode]
```

So per fresh turn: **15.1 GiB of arena evicted, 81/141 tables stood down, 6 re-seeded eagerly, 75 left to
fill on demand**; the decode then re-earns residency while it runs, converging on 0.84 hit rate.

### 3.3 The measurement that decides the proposal: the on-demand refill is *mostly hidden*

Three decodes in one server process. A and B each follow a fresh 16 k-token prefill (the same 15.1 GiB
eviction); C follows a 4-token cached prompt (nothing evicted, arena warm):

| arm | prompt | decode length | wall decode rate | arena hit |
|---|---|---:|---:|---:|
| A — cold arena | 16 k, fresh | 8 tokens | **27.7 t/s** | 0.463 |
| B — cold arena | 16 k, fresh | 128 tokens | **44.6 t/s** | 0.844 |
| C — warm arena | 4 tokens (cached) | 8 tokens | **42.7 t/s** | 0.842 |

B pays the *same* 15 GiB of fills as A and still reaches the warm rate, because the fills overlap with
compute; C shows that an 8-token decode is not inherently slow. **A's 35 % deficit against C is the fill,
and it is a function of decode length, not of the arena's contents.** A modal reload does not save a byte
— same bytes, same link, same 17-33 GB/s. It converts "spread across the tokens that need them,
overlapped with compute" into "one synchronous stall at the transition". For a decode shorter than the
fill that is a wash; for a longer one it is strictly worse.

**Consequence: on this shape (32 GiB, arena 23 GiB) a modal reload is worth ~nothing.** Its payoff has
to come from somewhere else — and there is one place it appears (§3.4).

### 3.4 Where the churn *does* cost: short turns on a long prefill

The tax scales with `evicted_bytes / decode_length`. A real chat turn on this card is 16 k prefill
(6.5 s) + 8 tokens out. That turn runs at 27.7 t/s instead of 44.6 — a 35 % decode regression that is
entirely fill. If the turn were 128 tokens it would be invisible. So the case for loading the arena
"modally" is really a case for *being able to keep it* — i.e. Q1's decision to evict, not Q2's reload.
And §3.5 shows that "being able to keep it" is currently broken in a much more basic way than any phase
policy could fix.

### 3.5 The real defect: the arena is sized **once**, against the *prefill* boundary

Reported symptom (maintainer, 2026-11-15): *"the usable arena size when in decode-only mode seems to
shrink when ubatch grows"*. It does, and neither the phase nor the reload is responsible.

**Measured — the arena region in the decode steady state vs the arena the cache sized itself to**
(`llama-cli`, `GGML_CUDA_SLAB_STATS=1`; the last `SLAB_STATS` line of a run is post-drop, i.e. `work ==
narrow`; "sized" is the cache's own `arena ... MiB` census line):

| `-ub` | width the slab was **born** with (`boundary` at `SLAB_INIT`) | narrow floor | arena region, post-drop | arena the cache sized to | **thrown away** |
|---:|---:|---:|---:|---:|---:|
| 4 096 | 1 600 MiB | 2 048 | 23 744 MiB | 23 908.0 | **0** (saturated) |
| 8 192 | 3 200 MiB | 2 048 | 23 616 MiB | 22 167.6 | **1 448** (6 %) |
| 16 384 | 6 656 MiB | 2 048 | 23 360 MiB | 18 411.9 | **4 948** (21 %) |
| 32 768 | 12 928 MiB | 2 048 | 22 848 MiB | 11 633.4 | **11 215** (49 %) |

The loss is a closed form, and it holds on all four rows to within one arena unit of chunk/rounding
noise: **`loss = max(0, boundary_at_birth − narrow_floor)`**. Which is why `-ub 4096` is unaffected (its
birth boundary, 1 600 MiB, is *below* the 2 048 MiB narrow floor the later probe pins) and why a
prefill-first workload degrades monotonically from there.

**Mechanism, exactly:**

1. The slab is created at the *first compute-buffer allocation* — the `sched_reserve` of the `pp` graph —
   and `ggml_cuda_slab_init_locked` sets its initial `boundary` from that request. So the slab is **born
   with the prefill band already reserved**: `SLAB_INIT ... boundary=6.50 GiB` at `-ub 16384`,
   `12.62 GiB` at `-ub 32768`. (That is deliberate for the reserve — the first graph must fit without a
   boundary move — but it means the birth state is the *wide* state.)
2. The cache sizes itself from `ggml_cuda_slab_arena_total()` = `arena_top - boundary`
   (`ggml-cuda-slab.cu:729`) **at that instant** — the comment says so explicitly: *"it must reflect what
   is allocatable NOW"*. `alloc_all_locked` (`moe-expert-cache.cu:6570-6620`) clamps **both** the auto
   budget and an explicit `MOE_EXPERT_CACHE_MIB` to it, then latches `g_sized = true` forever.
3. The prefill→decode drop returns the ring + wide band to the arena (the region really does grow back:
   `23 360` / `22 848` MiB above), but **nothing re-derives the slot count**: `moe_cache_rearm`
   (`:8040`) re-arms to `g_rearm_slots[layer]` — the frozen sizing count — and the deficit rule is
   deliberate (*"a shortfall means the arena really is smaller from now on ... so a retry would only
   re-fill the layer for a count it cannot hold"*, TODO #70). That premise is **false at a prefill→decode
   drop**: the shortfall there is precisely the band the drop gives back.

**Two arms that confirm it is sizing, not the phase:**

```
AP1  (default)                                    arena 18137.1 MiB (38.7 % residency), tg 43.0 t/s
DEF0 (MOE_EXPERT_CACHE_SIZE_AT_PRIME=0)           arena 18137.1 MiB (38.7 % residency), tg 42.8 t/s
EXPL (MOE_EXPERT_CACHE_MIB=24000)                 arena 18137.1 MiB (38.7 % residency), tg 42.8 t/s
      exit census: "VRAM budget: requested=24000 MiB/device, free-clamped=1"
                   "MoE expert cache: ... slots=27918, arena=18137.1 MiB total (budget 24000 MiB/device)"
                   "... fills=40992 evictions=10781"   <- evicting for space with 5 GiB idle
```

Deferring the sizing (`SIZE_AT_PRIME=0`) changes nothing, because by the time *any* sizing can run the
slab already exists with its wide birth boundary. And an explicit budget **larger** than the birth-time
arena is silently free-clamped — the census then reports a 24 000 MiB budget next to an 18 137 MiB
arena while the cache evicted 10 781 times. (That second part is its own defect against `ENVIRONMENT.md`,
which calls an explicit budget "honoured".)

**How much is it worth? Measured — and far less than the campaign record's slope predicts.** Interleaved
A/B on one server shape, two cycles each, fixed workload (4 k-token prompt, 128 greedy tokens, `-ub
16384`, 4 k depth so the MoE share of decode is at its largest), vary only the arena:

| arm | `MOE_EXPERT_CACHE_MIB` | arena | arena hit | decode t/s (run 1 / run 2) |
|---|---|---:|---:|---:|
| **A** | auto | 18 595.1 MiB | 0.8197 | 43.87 / 43.60 |
| **B** | 12 000 | 11 999.8 MiB | 0.7158 | 42.95 / 42.96 |

**+6.6 GiB of arena (+10.4 pp residency) is worth +0.78 t/s = +1.8 %**, i.e. **≈ 0.17 %/pp** — not the
0.87 %/pp I first extrapolated from `wip/slab-draft-aux-reserve/README.md` (+6.3 % for +7.2 pp; that
record is MTP-on, where a verify step does ~4x the MoE work per token, which is the likely reason it is
5x steeper). The two cycles agree to 0.6 % / 0.02 %, so this is not this box's period-2 noise.

**So the value of the whole fix is ≈ +1.8 % decode at `-ub 16384` and ≈ +4 % at `-ub 32768` — real,
robust, cheap, and nowhere near "huge".** Correcting my own earlier estimate: I extrapolated a 10-21 %
win from the campaign record's slope before measuring it, and the measurement does not support it.

> **REVISION (2026-11-15, 2 000-token measurement).** The 128-token arm above understates the effect by
> roughly 4x, because the arena only reaches its steady state after a few hundred tokens and the settle's
> whole-table re-seed is a fixed cost. At **2 000** greedy tokens on the same shape the same fix measures
> **+8.2 % at `-ub 16384`** (arena 18 411.9 -> 22 819.7 MiB, miss traffic -39 %) and **+31.9 % at
> `-ub 32768`** (12 183.0 -> 23 018.0 MiB, traffic -70 %); the 24 GiB `gfx1100` card is **+19.5 %**. The
> 2-device `-sm tensor` band is **not** worth taking (0.44 pp of prior mass, -44 % prefill when taken)
> and the settle now leaves it alone; see `INVARIANTS.md` r292 items 8-10 and the beta candidate's README.

The reason it is so flat is worth recording: this config's host tier is a **100 % MIRROR**, and a mirror
answers a miss *in place* through the pool's device alias (`MOE_HOST_POOL_COLD` / `GGML_MOE_DEVREMAP`'s
cold encoding `slots + pool slot`, r203) — so a "miss" is a local device read, not a copy. The arena is
then mostly buying **kernel shape** (resident routing), not bytes. **The arena's value should be
re-derived on a `MOE_HOST_POOL_DIO=1` disk tier or a partial pool** before anyone sizes an arena policy on
the mirror number — that is the arm where a miss costs a PCIe copy or a drive read, and it is the
`wip/expert-transfer-engine/` §6.1 open question arriving from the other side.

**Why the capacity still matters despite the small slope:** the extra slots cost nothing (the slab owns
the VA, the drop has already happened, the tables are already being re-allocated there), and the defect
is also an *accounting* defect — `MOE_EXPERT_CACHE_MIB` is silently clamped while the census reports the
unclamped request, and a stood-down/re-armed layer ratchets its slot count **down** for the life of the
process on the first wide prefill it sees. A user who raises `-ub` and cannot see why their hit rate
fell has no signal to follow.

### 3.6 So where does the *phase* question land?

Once the capacity is fixed, the arena is ~23 GiB in the decode steady state at any `-ub` that fits, and
the remaining bimodal question is only "should prefill be allowed to evict it?". Answer per §4: yes, it
already does, it must, and the churn is cheap (§3.3). **That is why §9 recommends fixing the sizing and
not building a mode.**

---

## 4. Q3: is ubatch 32 768 possible? (arithmetic, not policy)

The prefill band is `2 048 (narrow) + ring + 0.42·ub` MiB. On a 16.4 GiB card (scaling this box's
outside-slab inventory: weights 5 045 MiB + KV 689 + SSM 113 + `ROCm_Host` model 322 + driver ≈ 6.8 GiB,
plus the 1.25 GiB headroom, plus the MTP draft's 1.8 GiB weights when `--spec-draft-model` is on):

| card | slab available (no draft) | slab available (with MTP draft) |
|---|---:|---:|
| 32.6 GiB (this box) | 24.6 GiB | 22.8 GiB |
| 24.6 GiB (the wiki's card) | 16.6 GiB | 14.8 GiB |
| 16.4 GiB (the target) | **8.4 GiB** | **6.0 GiB** |

Against a prefill band of 8.1 / 12.2 / 17.1 / 23.3 GiB at `-ub` 4096 / 8192 / 16384 / 32768:

* **32 GiB**: everything fits today, arena 1 GiB at `-ub 32768`. No bimodality needed.
* **24 GiB**: `-ub 16384` fits *only* with the arena at ~0; `-ub 32768` does not fit at all.
* **16 GiB**: `-ub 4096` fits with the arena at 0 (6.0 GiB slab, 8.1 GiB band — needs the ring cut to
  ~1 GiB to fit at all); `-ub 32768` is **impossible by 13 GiB**, because `wide` alone is 12.6 GiB.

**So the honest answer to "ub 32768, is that even possible?" is no on the cards that motivate it, and
yes on the cards that do not need it.** What bimodality buys a 16 GiB card is a *larger* ub (from ~4 k to
~8-9 k on this model with the ring cut), not 32 k. That is still a real win — 8 k ubatches roughly
halve the per-token expert traffic (INVARIANTS r219 fact 6: every prefill ubatch routes essentially the
whole table, so bytes/token ∝ 1/ub) — but it should be scoped as "one doubling", not "32 k".

---

## 5. Q1/Q2 vs "the slab could be much larger"

The premise behind the proposal — that the slab is small because prefill and decode structures must
co-exist *outside* it — is worth checking, because it changes where the effort should go.

**On this box the slab is already ≈ the whole card.** At `-ub 16384`:

```
load_tensors:      ROCm0 model buffer size = 5 045.50 MiB     <- outside (weights, permanent)
llama_kv_cache:    ROCm0 KV buffer size    =   612.00 MiB     <- outside (ctx-proportional, permanent)
llama_memory_recurrent: RS buffer size     =   112.57 MiB     <- outside (permanent)
llama_kv_cache:    indexer KV buffer       =    76.51 MiB
SLAB_INIT dev=0 free=25.86 GiB reserve=1.25 GiB headroom=1.25 GiB size(VA)=24.56 GiB mapped=24.56 GiB
```

`mapped = free − headroom`: the slab takes **24.56 of the 25.86 GiB** still free when it is created. The
"co-existence" cost is not a slab-size problem; it is these three fixed claims:

| fixed claim | this box | 16 GiB box | mode-selectable? |
|---|---:|---:|---|
| dense weights | 5.0 GiB | 5.0 GiB | no |
| KV + SSM + indexer | 0.8 GiB (2.7 at 204k) | ~1.0 GiB | no |
| MTP draft weights | 1.8 GiB (off in these runs) | 1.8 GiB | no |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | 1.25 GiB | 1.25 GiB | **partly** — see below |
| **mode-selectable set** | **arena 23.1 GiB (decode) / ring+wide 15.4 GiB (prefill), both already inside the slab** | same idea | yes |

Two consequences:

1. **Everything the proposal wants to swap is already inside the slab.** The bimodal union on this box
   is decode-mode 25.1 GiB (arena 23.1 + narrow 2.0) and prefill-mode 17.4 GiB; both fit the 24.6 GiB
   slab. The conflict is *churn*, not capacity — which is why §3.3 matters so much.
2. **The one claim that genuinely could be mode-dependent is the headroom.** It exists for the transients
   that are *not* slab-served: hipBLASLt/Tensile workspace (~300 MiB/device measured), the Q8_1 arena.
   The 1 280 MiB default is ~1.2x the measured `T` (r206 raised it from 640 for a `-c 262144` OOM). On a
   32 GiB card that is 4 % of the slab; on a 16 GiB card it is **15 %** — worth attacking, but it is a
   separate campaign (the `aux`/headroom accounting, cf. `wip/slab-draft-aux-reserve`), not bimodality.

So: **bimodality does not make the slab bigger. It decides who owns the slab.** Reframing the proposal
that way is what makes it affordable to implement.

---

## 6. Implementation sketch (if it is opened)

Deliberately small, and staged so each phase can be abandoned.

### Phase 0 — **the capacity fix** (the actual defect, and it is free)

Two small changes in `moe-expert-cache.cu` + one accessor in `ggml-cuda-slab.cu`. No new transport, no
new eviction policy, no mode.

1. **A steady-state arena accessor.** `ggml_cuda_slab_arena_steady(dev)` returns
   `arena_top - max(narrow_floor, narrow2_floor)` — the arena the *decode* steady state has — alongside
   the existing `arena_total()` (which is `arena_top - boundary`, i.e. "what is allocatable NOW"). Use it
   as the cache's **budget**; keep `arena_total()` as the bound on what can be *placed* at this instant.
   `alloc_all_locked` then computes the slot count from the steady budget and clamps only the *placement*.
2. **Settle at the drop.** In `moe_cache_rearm` (called after the narrow re-reserve in
   `llama-context.cpp:2853`, i.e. with the boundary already back at `narrow_floor`), recompute the
   per-device slot count from the **live** capacity and allow `g_rearm_slots[layer]` to be raised, up to
   that count. Bound it one-shot per drop and only when the capacity grew, so it cannot ratchet or
   oscillate. The TODO #70 "never revisit a deficit layer" rule stays intact for its actual meaning (a
   permanently smaller arena); this adds the missing case.
3. **Do not silently clamp an explicit budget without the census agreeing.** Either re-clamp at the drop,
   or report `requested=X, clamped=Y (prefill band)` on the sizing line and in the exit census. Today the
   census says `requested=24000 ... arena total 18137.1 MiB` and nothing reconciles them.

Cost: no extra H2D if the fill policy is left alone (the extra slots stay empty and fill on demand, which
§3.3 showed is nearly free). **Measured value: ≈ +1.8 % decode at `-ub 16384`, ≈ +4 % at `-ub 32768`**
(§3.5 — not the 10-21 % an extrapolation from the campaign record's slope suggested). Verification: a
residency/hit-rate census before and after, the eviction count (`evictions=` in the exit census should
fall), same-seed text with the heat prior pinned (`MOE_EXPERT_HEATMAP=load`, since residency selects the
MoE kernel set and the sets are not bit-identical — `GREEDY-PURITY.md` §89), plus `gate-host-pool.sh` and
the MTP four-axis protocol. This is a **block 09/10 amendment** and therefore needs the maintainer's
go-ahead.

### Phase 1 — one env var, no plumbing: force the band to be seeded

Add `MOE_EXPERT_CACHE_BAND_SEED=1` (kill-switch/opt-in per `ENVIRONMENT.md` rules) that:

* makes `heat_seed_above_floor()` return true when the slab is **not** in a wide pass, and
* orders the per-table seed by heat **across** tables (not per table), so a partially complete reload is
  still the right set.

~30 lines in `moe-expert-cache.cu`. This alone tests Q2 with no slab change. **Expected outcome per
§3.3: no win on 32 GiB, a small win on the 8-token-turn shape, possibly a loss on back-to-back prefills
(the seed is thrown away).** Cheap enough to be worth running as the falsification test.

### Phase 2 — the mode

`--parallel 1` -> `common_params.n_parallel == 1` -> a new `ggml_backend_dev_slab_set_mode(dev, mode)`
alongside the existing `slab_set_ring_region` / `slab_set_aux_reserve` hooks (precedent and plumbing are
identical). In bimodal mode:

* `GGML_CUDA_SLAB_MIN_ARENA_MIB` is not merely a *creation* floor — the boundary is allowed to consume
  the arena to 0 when the wide layout asks for it (it already is; the change is that the **sizing** pass
  stops seeding the arena at all when it knows the first request will be a wide prefill);
* the drop calls the Phase-1 refill unconditionally rather than the `heat_seed_above_floor` subset.

`--parallel 1` is the right *coarse* gate but not a sufficient one: with `--spec-draft-model` there are
two contexts and two graphs in flight, and the MTP draft's compute buffer is the pinned `narrow2` region
(never in the arena), so the arena/band exclusivity still holds — worth asserting rather than assuming.

### Phase 3 — the reload becomes selective

Do not reload on every transition; reload when `evicted_bytes / link_rate < expected_decode_seconds`.
The server knows `n_predict` for the request, which is the honest signal — plumb it as a hint
(`ggml_backend_dev_slab_bimodal_hint(n_expected)`) rather than guessing from `-c`. Without this, a
draft-token turn (`-n 8`) pays 0.5 s to save 0.2 s.

### What I would *not* build

* **A separate "prefill only" allocation path.** The boundary already does it; a second path is the
  pre-slab crash mode (`INVARIANTS.md`, `cache_unusable`/`declined`).
* **A second copy stream for the refill.** Measured before and rejected: extra streams delay the first
  completion without adding bandwidth on this shape (`wip/expert-transfer-engine/README.md`).
* **Reloading from disk.** The premise is the *mirror*; a partial pool is `-lme < n_experts` and the
  refill would be a drive read — the campaign's whole design constraint.

---

## 7. Risks the assessment has to name

1. **Graph recapture is the hidden cost of a *bigger* eviction.** `moe_cache_evict_slab_range` bumps
   `ggml_cuda_graph_mem_freed`, and the re-arm frees/re-allocs arenas, so every capture that baked an
   arena address dies. Today that happens every turn for 81/141 tables; a *fully* bimodal slab would do
   it for all 141, and the recorded cost of losing the replay benefit on this shape is **13-16 % decode**
   (the `ring_carved` note in `llama-context.cpp:2755`). The cheap fingerprint is in the slot timing:
   `graphs reused` vs `graph reallocs`. **Measure this before anything else** — it can eat the whole win.
2. **Residency selects the kernel set, and the sets are not bit-identical** (`GREEDY-PURITY.md` §89, the
   MoE expert-cache triage rule). Bimodal mode is a *distinct configuration*: same-seed text will differ
   from the non-bimodal arm, and the prefill-logit gate must be re-recorded for it. Within the mode it
   must be deterministic (the refill set is a deterministic function of the heat prior, and the heat map
   is writable by default — pin it with `MOE_EXPERT_HEATMAP=load` for any A/B).
3. **A stood-down arena stands down the fusions.** With every table at 0 slots,
   `moe_cache_has_arena_locked()` is false and every cache-aware fused kernel declines (the fuse-guard
   invariant). At prefill that is measured free (r234: 1826 t/s declined vs 1824 t/s staged) — good. But
   it means the prefill's expert transport silently becomes "the scheduler's own whole-table upload", and
   any prefill-logit comparison across the mode change is comparing two transports, not one.
4. **`--parallel 1` does not mean "one graph".** `--spec-draft-model` adds a second context; a checkpoint
   restore, a state save, or the server's KV-cache maintenance can each hold a graph. The gate has to be
   "no second *compute* buffer is live", which the slab already tracks (`work_needs`, `narrow2_live`).
5. **The 1 024 MiB at `-ub 32768` is the real conservative bound.** Pushing it to 0 needs the wide-aux
   region and the FA staging to be *inside* the band rather than above it, and `SLAB_WIDE_AUX ...
   refused: slab=1 ring_armed=0` already appears at the head of a prefill in these logs. Do not promise
   arena==0 without re-deriving that.

---

## 8. Where bimodality genuinely pays (and where to point it)

**First, though: §3.5 dominates this table.** Until the arena is sized to its steady state, every row
below is describing a *sizing* loss dressed as a phase conflict — "arena already 18 GiB after the drop"
means "18 GiB of the 23 GiB the drop returned".

| configuration | evicted per turn | ub gain from arena==0 | verdict |
|---|---|---|---|
| 32 GiB, `-ub 16384` | 15.1 GiB | ~0 (arena is 23 GiB after the drop — once §3.5 is fixed) | **no win.** Churn is hidden (§3.3). Spend the effort on the ring instead. |
| 32 GiB, `-ub 32768` | ~23 GiB | none (already 1 GiB) | **already bimodal.** And it throws away 11.2 GiB of its own arena (§3.5). |
| 24 GiB (wiki card), `-ub 8192` | ~10 GiB | `-ub 16384` becomes reachable | **marginal** — the slab holds 17.1 GiB of band, so the arena loses 17 GiB and the reload is 0.5-1 s. |
| 16 GiB, `-ub 4096` | ~4-8 GiB | `-ub 8192` reachable **only with the ring cut** | **where the proposal pays** — but the ring, not the arena, is the binding constraint |

The recurring theme: **the ring is the discretionary 47 %**. Any small-card campaign that starts by
moving the arena should start by asking why the ring is `1800 + ub/1.7` MiB, i.e. 6.5 GiB at `-ub 8192`
for a 341 MiB table. The recorded trade is "ring 6.00 -> 2.50 GiB = prefill −11 %, decode +2.7 %, arena
unchanged" (`wip/expert-transfer-engine/README.md` §7) — that needs re-deriving on the current tree but
it is the same lever, one order of magnitude cheaper than a bimodal rewrite.

---

## 9. Decision needed from the maintainer

Nothing here is a `TODO` item and nothing is in `patches/`. Five options, in descending order of what I
would argue for:

* **S — fix the arena sizing (§3.5 / Phase 0).** `arena_steady()` + a one-shot settle at the drop + stop
  the silent explicit-budget clamp. **This is the finding the session actually produced**: 4.9 GiB of a
  23 GiB arena thrown away at `-ub 16384` and 11.2 GiB at `-ub 32768`, worth **≈ +1.8 % / ≈ +4 % decode**
  as measured (§3.5 — my first estimate of +10 % / +21 % was an extrapolation and is wrong). It is a
  **block 09/10 amendment**, so it needs an explicit go-ahead, and it wants its own `TODO` handle.
* **D — fold the *findings* into `INVARIANTS.md`** (the sizing/birth-boundary rule, the `heat_seed_above_floor`
  rule, the `§3.3` fill numbers, the explicit-budget clamp). Durable regardless of what ships.
* **C — reopen as the small-card campaign:** bimodal + ring re-derivation, targeted at 16 GiB, with the
  ub goal stated as "one doubling", not 32 k. Only worth it *after* S, because S changes the number it
  would be optimising.
* **B — Phase 1 only** (`MOE_EXPERT_CACHE_BAND_SEED`): ~30 lines, tests Q2 on the real shape. Note that
  §3.3 predicts a *small* win on 24-32 GiB and a loss on back-to-back prefills, so this is now the
  weakest of the options.
* **A — decline the mode, record it.** The modality mechanism exists, the reload is already eager for
  90 % of the tables, and the incremental win from a *mode* (as opposed to the sizing fix) is ~0.

> **OUTCOME (2026-11-15, v2): S was implemented, re-measured, demand-gated and promoted.** It was folded
> into **block 09** at r292 (candidate record
> [`archive/work/arena-steady-sizing/`](../../archive/work/arena-steady-sizing/README.md); one file, one
> commit, kill-switch `MOE_EXPERT_CACHE_STEADY_REARM=0`): arena 18 411.9 -> 22 819.7 MiB, miss traffic
> **-39 %**, decode **+8.2 %** at `-ub 16384`, **+31.9 %** at 32768, **+36.8 %** on a partial pool, and
> **+19.5 %** on the 24 GiB card, same-seed greedy text byte-identical. The settle is gated on the
> marginal prior mass the drop offers, so a saturated 2-device split and a small `-ub` are byte-identical
> to base, and the settled count is sticky per layer with its capacity so a two-slot decode/prefill
> transition is stable. This document stays as the assessment that found it.

My recommendation: **S + D.** The phase/mode question is downstream of a sizing bug; fixing the sizing
closes out the symptom that motivated the proposal. Bimodality is then worth re-asking only on a 16 GiB
card, and the *arena-value* question (`mirror` vs `DIO` tier, §3.5) is worth more than either.

**One thing to measure before S is called done:** the same arena sweep on a disk tier
(`MOE_HOST_POOL_DIO=1`) or a partial pool. The +1.8 % above is a *mirror* number, where a miss is served
in place; if the slope is 5x steeper on the tier where a miss really costs a copy, then the same one-line
sizing fix is worth 5x more on the configurations that need it most.

---

## 10. Reproduction

```bash
# 0. the build this was measured on
cd ~/llama.cpp && git log --oneline -1          # 14f3749be = r288 tip
~/bin/build-llama-rocm

# 1. the prefill band vs -ub  (llama-cli, 1 GPU, census arm so -lv 4 is correct here)
cd /llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S
export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0
export GGML_CUDA_SLAB_STATS=1 MOE_HOST_POOL_RAM_PCT=85 MOE_HOST_LIMIT_FRAC=95 GGML_MOE_CACHE_STAGE=0
UB=16384
~/llama.cpp/build-rocm/bin/llama-cli -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --n-gpu-layers all --n-cpu-moe 47 --ctx-size $((2*UB+512)) \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap --lazy-mode mlockr --threads 7 \
  --batch-size $UB --ubatch-size $UB --temp 0 --seed 42 \
  --no-display-prompt --single-turn -n 8 -f <prompt of ~2*UB tokens> -lv 4 > /tmp/probe.log 2>&1

grep -E "SLAB_INIT|slab .*: work|SLAB_STATS wide|re-armed|evicted [0-9]+ MiB" /tmp/probe.log

# 2. the per-turn churn and the decode-length dependence (the decisive one)
#    same flags on llama-server --host 127.0.0.1 --port 8099 -lv 4, then:
#      POST /completion n_predict=8   -> ~27.7 t/s, hit 0.463
#      POST /completion n_predict=128 -> ~44.6 t/s, hit 0.844   (same 15 GiB of fills)
#    the heat map is read-only on this box (chmod 444), so the prior is stable across runs.

# 2b. §3.5 — the sizing defect, in three arms of the same server command (-ub 16384):
#      default                                  -> arena 18137.1 MiB (38.7 %)
#      MOE_EXPERT_CACHE_SIZE_AT_PRIME=0         -> arena 18137.1 MiB (38.7 %)   [identical]
#      MOE_EXPERT_CACHE_MIB=24000               -> arena 18137.1 MiB (38.7 %)   [silently clamped]
#    the three numbers that prove it:
#      grep 'SLAB_INIT'                     -> boundary is the -ub work region (the slab's BIRTH state)
#      grep 'SLAB_STATS' | tail -1          -> arena(region) post-drop = 23360 MiB, work == narrow
#      grep 'VRAM budget'                   -> 'requested=24000 MiB/device, free-clamped=1'

# 3. what is outside the slab (the §5 claim)
grep -E "model buffer size|KV buffer size|RS buffer size|SLAB_INIT" /tmp/probe.log | head -20
```

---

## 11. See also

* `patches/README.md` — block 10 (the slab + fusion dispatcher), block 09 (host-resident MoE iface) and
  block 15 (the `-lme` host-expert tiers) are the three blocks this campaign touches.
* `INVARIANTS.md` — r219 fact 6 (bytes/token ∝ 1/ub, the arena cost of a larger ub), r234 (a fully
  stood-down arena at prefill is measured free), r206 (the headroom default), and the slab/arena sizing
  entries.
* `MANIFESTS.md` -> Release gates — the gates a Phase 1 would have to pass:
  `gate-host-pool.sh` (the fill transport), `gate-prefill-logits.sh`, the MTP four-axis protocol, and
  `gate-per-block-build.sh`.
* `wip/slab-draft-aux-reserve/` — the open 2.4 GiB `aux` accounting error in the same sizing path; a
  bimodal mode would be sized against the same `aux` and should land after it is fixed.
* `wip/expert-transfer-engine/README.md` §6/§7 — the ring sizing and the "shrinking the ring strands
  VRAM" measurement, i.e. the cheaper version of this lever.
