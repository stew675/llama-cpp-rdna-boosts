# archive/work/host-expert-split - the split between pool, arena and page cache (and whether a swap can ever beat a layout)

**Status: OPEN - review + design done (2026-11-05), nothing implemented, nothing in `patches/`, nothing applied
to the fork.  Re-opens a shape that was measured and refuted in r113, for a narrower reason and with a cheaper
gate.**  PR #267 is reviewed and the convergence plan is written in [`DESIGN.md`](DESIGN.md): the delta worth
testing is the **arena's eviction sink** (the tier is the arena's complement, an admission is a swap), *not* the
PR's `--host-experts mmap` mode or its sparse registration.  Step 1 (the offline replay) is waiting on one
bounded trace run — the only GPU run this campaign needs before deciding.

**What this is.**  `briansp2020`'s host tier is an extension of the VRAM arena: experts are swapped to it on
evict, so a miss never leaves RAM.  Ours is a LAYOUT: the pool's content is chosen up front as the arena's
complement, and the demand path never evicts, never copies an arena victim into the pool.  We measured the
literal swap and rejected it.  What is genuinely open is not "swap or not" but **the split**: how many of the
46.84 GiB of host expert bytes should sit in pinned host RAM, how many in VRAM, and whether a miss can be made
cheap enough that a partial pool stops being a cliff.  On a 56 GiB container that is the difference between
Brian's ~40 GiB host footprint and our 48.9 GiB.

## 1. What you inherit (do not re-derive)

* **The layout beat churn, measured.**  `INVARIANTS.md`: pool ranked by the **persisted heat prior as the arena's
  complement** costs 237 MiB of L3 reads over 451 decode tokens, ranked by expert id costs 3571 MiB, oracle is
  also 237 MiB, and **every** eviction/hand-over policy lands within 0-246 MiB of the others.  The demand path
  therefore does not evict and does not copy an arena victim into the pool, and the rule is explicit: "do not
  reintroduce one: the literal swap cost **2x decode** for 0 MiB of L3".  Record:
  [`archive/work/host-pool-layout/`](../../archive/work/host-pool-layout/).
* **Use the replayer before proposing another pool policy.**  `scripts/moe-trace-replay.py` (`--pool-frac`, the
  two-tier shape, a real `.moe-heatmap`) is the tool that decided the layout question.  Any new split proposal
  starts there, not in C++.
* **The tier shape question is already answered for the "fits" case.**
  [`archive/work/host-expert-eager-push/P5-RESULTS.md`](../../archive/work/host-expert-eager-push/P5-RESULTS.md)
  §7 has the anchors: at 86 % of the bytes the pool and whole-tensor tiers tie on decode and the pool wins
  prefill; at 100 % the pool is a mirror and carries `pinned`'s memory with a better transport.  Do not re-run
  those; extend them.
* **The VRAM tier's content policy is a different campaign.**  `wip/moe-static-hot-tier/` owns "freeze the hot
  subset outside the movable boundary" (heat-filled, static, immune to ubatch eviction).  This campaign is about
  the RAM/VRAM **split** and the **miss cost**, not about freezing slots.  Cross-reference, do not duplicate.
* **The arena auto-sizes.**  On this model in a 32 GB card it is 19 to 20 GiB (39 to 42 % of the host set) at
  `-c 204800`, ranked by the same prior.  A mirror pool plus that arena is 67 GiB of expert bytes, which does
  not fit in a 56 GiB container: on a small box the split is forced, and today it is forced to 100/0 because any
  miss is fatal.

## 2. Evidence (2026-11-05, `bench-56gb` rig, mirror pool, `-ub 6144`)

| coverage | decode | first 50k prefill | what the miss costs |
|---|---|---|---|
| 100 % (mirror) | 78.5 | 1941 | nothing: 0 MiB of drive reads |
| 97.7 % (`-lme auto` on a 59 GiB box) | 75.6 | 1443 | bounce ring |
| 93.6 % (`-lme 51`, ceiling) | 74.9 | 1770 | bounce ring |
| 92.4 % (`RAM_PCT=50`) | 76.6 | 1061 | bounce ring |

A miss today is a per-table bounce-ring read at about **125 MB/s** (derived from the first-pass delta: 1.08 GiB
of uncovered bytes cost 8.8 s).  If a miss were a single PCIe copy of one expert, the same 2.3 % would cost about
2 %.  That is the whole campaign in one sentence: **the cliff is the miss path, not the split.**

## 3. The work, in the order I would take it

1. **Offline replay first, three miss models — DONE 2026-11-05**, see [`STEP1-REPLAY.md`](STEP1-REPLAY.md).
   On this model's heat map, replay the decode trace with the two-tier split under (a) today's bounce-ring
   cost, (b) a single-expert PCIe copy cost, (c) a free miss (the page-cache shape Brian's design gets).
   *Result:* where the pool covers the arena's complement the sink gains nothing and pays 5-6 MiB/token of
   extra copies (r113 reproduces); below the complement the sink cuts L3 by 33-74 %, but most of that gap is
   a **tool artefact** (`pool-none` never models the shipped "complement, then mirror residents" re-rank),
   and the one genuine advantage is a **degraded prior** (no `.moe-heatmap`: 35.9 -> 26.0 MiB/token).  So the
   surviving cases are the PR's own (same speed, fewer pinned bytes) and robustness — neither is throughput.
2. **Make a miss cheap, measured alone.**  A single-expert staging path (one reusable pinned staging buffer per
   device, or the pool alias when the expert happens to be resident) instead of the per-table bounce ring.
   *Accept:* 50k prefill at 92 % coverage within 5 % of the mirror, with the mirror row unmoved.
3. **Split sweep on the rig**, only if 1 and 2 both say yes: (pool, arena) = (47, auto), (44, auto), (40, 26),
   (34, 32) subject to VRAM, at `-c 204800`, `-ub 6144` and 2048.  *Accept:* a crossover table and a rule that can
   become a documented default ("pool = min(set, RAM - X)").
4. **A swap on evict stays OUT unless step 1 separates it.**  If the replay shows a hand-over policy beating the
   layout by a margin outside its own noise band (the r113 spread was 0-246 MiB of L3 across every policy), then
   and only then is a writeback worth code, and it needs the full gate list: bit-identical greedy text at every
   budget, a completion event per swap, and no regression to the mirror row.
5. **`-sm tensor` at every step.**  Two devices hold disjoint slices of the same host tensor, so the pool's
   per-expert slot is the natural unit and the arena's per-device slot is not.  State that in any design before
   writing it.

## 4. What would kill it, and what to watch

* The r113 refutation itself.  If step 1 reproduces "the split barely matters once the layout is right", the
  answer to Brian is that our transport is the difference, not our tier list, and this campaign closes.
* `wip/moe-static-hot-tier/` may already own the VRAM side of the same arithmetic; if it lands a static frozen
  tier, the split question changes shape (the frozen bytes are not in the movable arena, so "arena size" stops
  being one number).  Read it before step 3.
* A cheap miss makes a partial pool viable, which makes the *host* footprint smaller - but only if the cheap miss
  does not re-read the mapping from disk.  On a 56 GiB box the page cache cannot hold 46.84 GiB beside a 40 GiB
  pool, so "cheap" must mean "from VRAM or from the pool", not "from the page cache".  That constraint is exactly
  why Brian's shape works on his box and would not work here at 100 % page-cache reliance.
* Do not let a swap reintroduce a demand-path eviction inside a decode band: the band is 8-16 tokens on RDNA4, so
  the working set per band is small and a per-band eviction churns.  That is the r113 "2x decode" mechanism.

## 5. Records

| where | what |
|---|---|
| [`archive/work/host-pool-layout/`](../../archive/work/host-pool-layout/) | the r113 layout-vs-churn result; read §on the swap before proposing one |
| [`archive/work/host-expert-eager-push/P5-RESULTS.md`](../../archive/work/host-expert-eager-push/P5-RESULTS.md) | the tier-shape anchors (86 % tie, 100 % mirror) |
| [`archive/work/host-pool-speed/RESOLUTION.md`](../../archive/work/host-pool-speed/RESOLUTION.md) | why the previous tier campaign closed, and what subsumed it |
| [`wip/moe-static-hot-tier/`](../moe-static-hot-tier/README.md) | the adjacent campaign: frozen hot tier, content policy, not the split |
| [`bench-56gb/RESULTS.md`](../bench-56gb/RESULTS.md) | the coverage-vs-cost rows above |
| `scripts/moe-trace-replay.py` | the replayer step 1 must use |
| `INVARIANTS.md` | the layout rule, the r113 numbers, the "0-246 MiB across policies" spread |
