# wip/moe-static-hot-tier — a STATIC VRAM expert tier filled by heat, frozen outside the movable boundary

**Status: OPEN — BLOCKED (2026-11-15) on the `--load-mode dio`/`mmap` rework on the second machine; resume from
the record's §12 (handover).**  Revised 2026-11-15 after r292 and measured the same evening
([`RESULTS-2026-11-15-2gpu-3gpu.md`](RESULTS-2026-11-15-2gpu-3gpu.md), harness [`band-ab.sh`](band-ab.sh),
prototype [`drop-band-store.patch`](drop-band-store.patch) / `~/llama.cpp` `wip/drop-band-store` `e550d842e`).
The measurement **does not** kill the idea — it names what is missing and caps the prize:

* **The prize is the shape's, not the idea's.**  On a 2-device `-sm tensor` split the band is 24.4 % of the
  arena and buys **0.08-0.42 pp** of routed mass, because 85.6 % residency already covers 99.58 % of routings
  and only **0.75 % of decode reaches miss** (`hits=5 598 121 fills=42 101` over 2 000 tokens).  The shape
  where a band is worth several pp is a **short arena** (residency <= 60 %), still unmeasured.
* **Taking it with the code in the tree is a loss**: `-41 %` of a 5.2 k prompt's prefill (-4.6 % of an 84 k
  one) and 3-6 % of decode.  That cost is the **mechanism**, not the policy: the r292 settle grows every table
  at the drop (an `free` + whole-table reseed) and `moe_cache_evict_slab_range()` stands a table down **whole**,
  so the band destroys the *hot* residency of ~34 tables and re-acquires it through the **demand** transport
  (~4 GB/s on the critical path) instead of the batched seed (17-33 GB/s).
* **The lazy half already exists and is not free by itself.**  `MOE_EXPERT_SEED=lazy|auto` (TODO #318) starts a
  subset tier's slots empty and fills on first miss, but *deferring* a fill costs **12x the prefill** on the
  flat arm (1 455 -> 115 t/s): a move that was already scheduled is worth retaining, a move that is *postponed
  to first touch* is paid on the critical path.  So the free part of the policy is **retention**, not lazy
  placement.
* **The reclaim-granularity answer is built and measured** (§10 of the record): a per-device **drop-band VRAM
  expert store** (`MOE_EXPERT_L2`, `~/llama.cpp` `wip/drop-band-store`, not in `patches/`) that retains evicted
  arena residents with a D2D copy and serves a miss from VRAM instead of the pinned host pool, and is **freed
  whole** at a prefill boundary move — so it costs the prefill **0.0 %** (measured) and no table is stood down.
  17.2 % (2 k) / 15.4 % (20 k) of misses were served from VRAM and host fill traffic fell 16 %, but the decode
  rate did not move (52.3 → 52.0 / 52.4 → 52.4): on this shape the miss path is already fully overlapped
  (0.29-0.62 % of reaches, 0.36 GB/s against a 28 GB/s link), which is the 0.42 pp ceiling seen from the other
  end.  The mechanism is proven free; the prize needs a **short-arena** shape or a genuinely agentic long
  generation, neither of which is measured yet.  **The crossover is measured now (§11 of the record)**: the
  store pays exactly the fill it avoids, so it is +4.4 % when demand fills are **storage**-backed (~300 µs
  each, `-lme 4` + `FILL_SRC=file`) and 0.0 % when they come from a **pinned mirror** (~12 µs, hidden by the
  async input batch) — 20 000 generated tokens included.  And a heavier model does **not** help: the 122B at
  70.6 % residency has *no band at all* (its tables fill the mapped arena), because "plenty of band" and "the
  hot set does not fit" do not co-occur on two cards.  The unbought cost on a storage-backed box is the
  **bounce** path (134 749 reads, 61.1 GiB, 29.4 s on the disk arm) which the store does not see.
* **What upstream would otherwise be missing: reclaim granularity**, exactly as the 2026-11-15 revision said: cold slots in
  the band must be droppable at a prefill **without standing their table down**.  Two ways: two-segment /
  per-slot indirection (the revision's plan), or a **band-resident retention store** that is one slab
  allocation holding copies of experts the demand path already moved, dropped wholesale at a boundary move and
  hit by a **D2D copy** instead of a host read (no kernel addressing change) — see the record's §7.
* **3 GPUs have no region at all**: the whole 46.9 GiB set is resident at every ubatch (`evictions=0`), and the
  wide ubatch is free.  Two defects surfaced on the way and are recorded, not fixed: the default
  `GGML_CUDA_SLAB_RESERVE_MIB` **aborts** a 3-GPU wide prefill (`HipVMM Failure`, rc 134; 2048+ runs), and the
  `-ub 8192` (99.6 % residency) `MOE_EXPERT_CACHE_IDENTITY_PREF` cliff costs **8.5 %** of decode against the
  uniform placement.

**Nothing of the tier itself is implemented.**  The revision below turns the "frozen tier" into a
*placement*; that placement was measured on 2026-11-15 and does not pay.

## 2026-11-15 revision — placement, not a freeze (after r292)

Block 09 now carries the arena settle (r292): at a prefill -> decode drop the per-layer slot count is
scaled into the band the drop returns, demand-gated on the marginal decode-band prior mass the growth buys.
On a short arena (1 x R9700, 24 GiB, a partial pool) it takes the band (+8.2 / +19.5 / +36.8 % decode); on
a saturated 2-device split it **leaves it** (2-device tensor 0.44 pp, `-sm layer` 0.14 pp) because taking
the band fills the lowest VA chunks with **whole** tables and the next prefill stands them down whole,
forcing a full-table host re-read (prefill 1625 -> 903 t/s, decode -5 %).

That leaves exactly this campaign's target: use the band on a saturated split without paying the prefill.
The convergence:

1. **Placement, not a freeze.**  The hot set goes **above the widest prefill boundary** (`narrow + ring +
   wide` at its maximum).  The slab boundary never reaches it, so it is safe **by construction**: no
   frozen flag, no victim-selection change, no "never promoted/demoted" policy.  The campaign's §7
   make-or-break risk -- *"a frozen slot inside a lendable chunk is not frozen"* -- **dissolves**; the
   hot region is simply not in a lendable chunk.
2. **The band is the cold, transient tier.**  Below the boundary, the rest of the arena (and any
   cold extension) is lazy-filled and sacrificial.  Tossing it at a prefill is cheap, and the misses it
   served are the ones the heat prior says are rare.
3. **The real blocker is slot addressing, not the slab.**  A table's slots are one contiguous block and
   the kernels compute `arena + slot * expert_bytes`.  Per-(layer, expert) hot/cold placement means
   **two segments per table** (stable hot base + transient cold base) or a per-slot indirection.  That is
   the work; the campaign's §4 "allocate frozen slots outside the movable boundary" becomes "allocate the
   hot segment above the maximum boundary and the cold segment below it".
4. **What it buys.**  On a saturated 2-device split the 0.44 pp / 0.14 pp become **free wins** once the
   prefill keeps every table's hot K slots above the boundary and takes only the cold segment.  On 1
   GPU / 24 GiB the same split removes the whole-table push and the per-drop re-seed, so the short-decode
   transient improves.  The content policy is still the campaign's: heat, per (layer, expert), spread over
   all layers -- layer-tail placement is 5x worse per byte.  Keep the r74 caveat: protect only the
   **robust core** (top ~5-10 % of pairs, ~35-49 % of the mass); the rest stays adaptive.
   **MEASURED 2026-11-15 (2 and 3 GPUs): the "free win" needs reclaim granularity.**  The cold segment's
   bytes are *not* billed at the next prefill (it streams them anyway), but the band's VA is a contiguous
   range at the arena's bottom and the eviction is per-table atomic, so a lazily-retained band can only be
   taken by standing tables down whole — which is what costs 0.25 s/GiB of prefill.  See
   [`RESULTS-2026-11-15-2gpu-3gpu.md`](RESULTS-2026-11-15-2gpu-3gpu.md) §2, §2b, §7.
5. **Two smaller steps first (contained, no addressing change).**  (a) **Heat-targeted growth**: the
   settle grows only the layers whose tables stay above the seed floor, so the cold tables are not pushed
   into the band; (b) **preserve-on-grow**: grow a table by D2D-copying its existing residents and leave
   the new slots empty (lazy), instead of `free` + host re-seed.  Both improve the enabled arms now; the
   two-segment placement is what unlocks the saturated split.
6. **Measurement plan** is unchanged in shape (§5), with one addition: the fingerprint is now the
   settle's own `the first drop offers ... gain ... pp` line plus `re-armed ... N layer(s) settled UP` and
   the evicted/stood-down counts.  A successful hot/cold split keeps the prefill's stands-down to the cold
   segment only and leaves the `-ub 16384` / `-ub 32768` / partial-pool decode wins intact.

**The measurement was run on 2026-11-15** — [`RESULTS-2026-11-15-2gpu-3gpu.md`](RESULTS-2026-11-15-2gpu-3gpu.md).
Headline: the band is worth 0.08-0.42 pp **on this shape** (a shape property, not the idea's ceiling), and
taking it with the mechanism in the tree costs 0.25 s of prefill per GiB of band — because the band is a
whole-table stand-down that destroys *hot* residency, and the bytes come back through the 4 GB/s demand
transport instead of the 17-33 GB/s batched seed.  The lazy half of the policy is already built
(`MOE_EXPERT_SEED`, TODO #318) and shows that *deferring* a fill is expensive (1 455 -> 115 t/s) while
*retaining* a fill that was already scheduled is free.  What is missing is **reclaim granularity** (§7.3 of the
record).  Still unmeasured: this campaign's 1-GPU / 24 GiB tier (§5's cell), a short-arena multi-GPU shape
(residency <= 60 %), and the two mechanisms the record proposes.

**The charter, in the maintainer's words** (2026-11-02, 1-GPU host, after the `host-expert-pinned-window`
campaign was closed):

- `-ncmoe 0` is **fully VRAM resident** — no host experts, so no expert cache at all.  People (the maintainer
  included) pin *some* experts as layer-resident because it gives the GPU **work to hide the streaming prefill
  I/O behind**: while the GPU computes the residents, the ring buffer streams the rest of that layer's experts.
  For **small ubatches** that is a noticeable win; for large ubatches the streaming overhead amortises.
- The problem: large ubatches **steal from the arena** and evict valuable experts.  So in practice it really does
  help to have some residents pinned.
- Therefore: keep a static tier, but **fill it by heat instead of by layer**, and make it **non-evictable**.

## 1. Why the fill policy is the problem (the arithmetic)

`-ncmoe N` in this repo is **tail placement** (`src/llama-model-loader.cpp`: `tn.bid >= n_layer - N`, the
nextn/MTP head excluded by construction; `LLAMA_HOST_EXPERTS_FROM_END=0` restores the historical forward
placement).  So `-ncmoe 44` on a 48-layer model keeps layers 0-3 fully device-resident.  That spends **1/12 of
the expert bytes** and buys exactly **1/12 of the routed mass** — every token routes `k` experts through *every*
layer, so a layer is 1/48 of the mass, exactly (measured: each table is 0.694 % = 1/144 of the mass on
Flash-Next; `archive/work/moe-cache-coverage/FINDINGS-2026-10-17-coverage-vs-defrag.md` §3.1).

Heat does not have that property.  From the same measured map (clean r80 sidecar, 8.38e6 tallies, `H(0.5)=0.952`):

| pairs held | 10 % | 20 % | 25 % | 38.3 % | 50 % | 63.7 % | 80 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| share of routed mass | **48.5 %** | 68.6 % | 75.8 % | 88.7 % | 95.2 % | 98.7 % | 99.9 % |

**The same bytes, spent by heat, buy roughly 5x the coverage** (~40-45 % of the mass for the top 1/12, against
8.33 % for four whole layers).  The static tier is not wrong — its *fill policy* is.

## 2. What exists today (do not re-derive)

- **The heat prior is already persisted and loaded**: the `MHT2` `.moe-heatmap` sidecar (block 09 / r60),
  written at the end of a decode turn, read at load by `apply_heat_seed_locked`, which seeds the top-`slots`
  experts of each table.  **Those slots are ordinary arena slots**: still in the LRU, still evictable.
- **A transient protection list** exists (`access_locked(..., protect, n_protect)`): experts of the token
  currently being staged must not be chosen as victims.  It is per call, not a static pin.
- **There is NO freeze partition.**  The staging comment "(no freeze partition)" is literal: the snapshot covers
  the whole arena.  The word "frozen" in the code refers to the *heat map file* being load-only
  (`g_heat_frozen`), not to slot residency.
- **The movable-boundary slab evicts wholesale under ubatch pressure**, and it is visible on this host:
  `moe_cache_evict_slab_range: evicted 5132.0 MiB of arena tables from the taken slab chunks`,
  `moe_cache_rearm: re-armed 30 stood-down expert-cache tables after a compute-buffer drop`.
- Per-table slot counts are allocated **uniformly** (`re-armed uniformly at 254 slots`); concentration-based
  sizing is the pool campaign's P1 (*cov90*), not done yet.
- `-ncmoe N` semantics: **tail placement**, MTP head excluded.  `-ncmoe 0` = no host experts = no cache.

## 3. The design to evaluate

Keep the static tier.  Replace its fill policy and its eviction protection:

1. **Content = heat.**  The frozen set is the top-`K` (layer, expert) pairs *by the persisted prior*, spread over
   all layers — not the four head layers.
2. **Frozen**: those slots are never LRU victims, never promoted, never demoted.
3. **Outside the movable boundary.**  This is the load-bearing part: if a frozen slot lives in a chunk the slab
   may hand to the compute pool, a wide ubatch takes it anyway and the whole point is lost.
4. **Sized in bytes, per table, by concentration** (`MOE_EXPERT_CACHE_FREEZE_MIB` or a fraction; per-table `K`
   from each layer's own coverage curve, i.e. cov90), re-seeded from the `MHT2` sidecar.
5. The **arena keeps the rest** adaptively, and the **ring keeps streaming** the remainder — so the latency
   hiding the maintainer describes is preserved, and the residents that hide it are the *hottest* ones.

**How much to freeze** — and this is a real constraint, not a tuning preference.  r74 measured a
**conversation-local F-LRU + heat seed beating a static cross-run top-`f` set by +1.8 pts** at equal residency
(`wip/host-expert-pinned-window/README.md` §2/`moe-cache-coverage`).  Freezing the *whole* budget therefore costs
~2 points of coverage.  The shape that follows: freeze the **robust core** (the pairs hot regardless of
workload — top ~5-10 % of pairs, ~35-49 % of the mass) and let the arena own the rest adaptively.

## 4. Implementation entry points (to inspect, not yet to change)

| step | where to look |
|---|---|
| frozen flag per slot, excluded from victim selection and from LRU | `access_locked` (victim choice), `pool`/`arena` slot state, `moe_cache_rearm` |
| allocate frozen slots outside the movable boundary | `alloc_table_locked` / the movable-boundary slab (`ggml_cuda_slab_*`, `moe_cache_evict_slab_range`) |
| seed the frozen set from the prior, once | `apply_heat_seed_locked` (already ranks by the sidecar) |
| per-table `K` by concentration | the cov90 sizing helper that `archive/work/host-pool-speed/` built (§P1, promoted in r207) |
| keep the identity/cold encoding honest | `prs` (staging) and the DEVREMAP cold band — a frozen set that is *contiguous from id 0* is what would let a partially resident table approach the `identity` band |

## 5. Measurement protocol (this is the campaign's real deliverable)

The effect being chased is a **small-ubatch** phenomenon, so the whole ubatch axis is the instrument:

* **prefill** at `-ub 512 / 2048 / 4096` × three arms:
  `A` no static tier (arena only, auto-sized), `B` the layer tier (`-ncmoe 44` = 4 of 48 layers),
  `C` the heat-frozen tier sized to **the same expert bytes** as `B`.
* **decode ≥ 2000 tokens** for all three (a policy conclusion from a shorter run is not a conclusion: the arena
  ramps over the first few hundred tokens).
* **the fingerprint, not just the outcome**: the arena's `evicted … MiB of arena tables` and
  `re-armed N stood-down` counts, plus `h` and the L3/disk bytes per token.  A working frozen set *reduces*
  evictions and re-arms; that is the mechanism, and it is printed on the current build.
* Instruments: `archive/work/host-expert-pinned-window/srv-test.sh` (persistent server, prompt cache off, three
  sequential requests, per-request `/proc/<io>` bytes + RSS + timings; `NCMOE=`/`-ub` select the arm),
  `llama-batched-bench -npl 1,4,8`, and `scripts/moe-trace-replay.py` to simulate first (`--seed sidecar:<path>`
  for the prior; `--churn on` models a static set that only fills empty slots, the default models the adaptive
  arena — the delta between them is exactly this campaign's question).
* Model/cell: `Qwen3.5/122B-A10B/UD-Q4_K_XL` is the pool campaign's reference; the **Flash-Next GSQ-Q3_XXS srv
  cell** is where the layer tier's own numbers are known (`pinned` 734.5 prefill / 59.5 gen / 68.6 GB RSS;
  `pool` 285.3/48.8), so use it for the A/B/C comparison and 122B for the sub-residency confirmation.

## 6. Gates owed when this becomes code

Per `MANIFESTS.md` and the delivery's own rules: same-seed greedy (150 chars `sha=4ac3a523160f` on Q4_K_M,
in-mode), **MTP four-axis** (policy change → `-n 3000`, acceptance identical, `none` == `draft-mtp` text),
`gate-prefill-logits.sh` (a prefill-path change), `llama-batched-bench -npl 1,4,8`, the `-ncmoe` matrix,
`gate-per-block-build.sh --changed-from`, and `check-gpu-faults.sh` on every log.  Note the gate's blind spot:
the prefill-logit gate cannot see a *residency* defect, so the eviction-count fingerprint above is the real
evidence for this feature.

## 7. Risks, traps, and things not to do

* **Prior overfit**: the frozen set comes from one workload's map.  Mitigation is §3's "freeze the robust core
  only"; a fully frozen budget measurably loses to the adaptive arena (r74, +1.8 pts against it).
* **Frozen slots shrink the adaptive working set.**  That is the trade; it is only worth it where the heat pick
  is much better than the LRU pick, which is exactly the measured 5x.
* **The movable boundary is the whole difficulty.**  A frozen slot inside a lendable chunk is not frozen.  If
  step 3 cannot be done without gutting the slab's lending, the campaign's honest outcome is "not viable" —
  record that rather than shipping a frozen flag that a wide ubatch ignores.  **2026-11-15 update:** the
  placement form (hot region above the *widest* boundary) removes this objection; the difficulty moved to
  the two-segment slot addressing (see the revision above).
* **Do not reinterpret `-ncmoe`.**  Keep it a *host-capacity* knob (which layers stay off the host).  The VRAM
  content knob is a byte budget (`MOE_EXPERT_CACHE_MIB` for the arena, a new freeze budget for the tier).
* **Do not repack the model file** to get the hot set contiguous without first checking the router
  `argsort` tie-break order (`upstream/moe-router-tie-break/`) — a permutation is a relabelling, but ties break
  by id order.
* **Do not adopt tail-expert skipping** (`GREEDY-PURITY.md` §83) — it changes the model, and it is the thing
  that makes a static hot set *look* better than it is.
* Routing sidecars are **MHT2**; the campaign directory's old `heatmap-audit.py` is MHT1-only and rejects them.
  Use `scripts/moe-trace-replay.py --seed sidecar:<path>`.
* No host-page residency, no `mlock`, no registration: this is a VRAM tier, so none of the no-XNACK exposure of
  the closed `window` campaign applies.

## 8. Relationship to the other work

* **`archive/work/host-pool-speed/`** (retired) owns the *host* tier (bounded pinned pool) and the cov90 sizing this campaign
  would consume.  The two are orthogonal: this one decides what lives in VRAM, that one decides what lives in
  RAM and how it gets there.
* **`archive/work/host-expert-pinned-window/`** (closed) contributes two things that are irrelevant *here* except
  as warnings: registration does not give residency, and the prefill-logit gate is blind to residency defects.
* If this works, the **layer tier loses its content-policy role** and `-ncmoe` becomes purely a host-capacity
  bound — one fewer residency tier, which is the delivery's direction of travel (drop `pool`, drop `pinned`,
  keep one mechanism).

## 9. First experiment (no code)

1. Simulate: replayer over a captured routing trace, `A` = adaptive default, `C` = `--churn on --seed
   sidecar:<map>` sized to `B`'s expert bytes; score `h`, L3/disk bytes per token, distinct ranges touched.
   Prediction from §1: `C` >> `B` on coverage at equal bytes.  If the simulation does not separate them, stop.
2. If it separates: implement steps 1-3 of §3 behind a disabled-by-default byte budget, measure the §5 matrix,
   and judge on the eviction fingerprint plus decode at ≥ 2000 tokens.
