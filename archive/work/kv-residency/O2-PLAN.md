# O2 — windowed / streamed KV: the ask, the only plan, and why the last two sessions drifted

Written 2026-10-29, after the maintainer said plainly: *"we've done everything BUT the thing that I asked
for. All I know about O1 and O3 is that I never asked for them."*

That is correct. This file exists so it cannot happen a third time.

---

## 1. What was asked for, and why (the maintainer's words, 2026-10-29)

> a **Windowed KV** … for **fully resident dense/MoE models**, because doing so would allow users to run large
> contexts with a Q8_0 quantization of Qwen3.8-27B, or a Q6_K quantization of 35B-A3B, on a 32 GB card, and
> **not run out of VRAM** … and for **host-resident MoE's (slab mode)**, because it would free up more VRAM
> for the decode arena, allowing for higher expert hit rates, leading to higher decode speeds.  Once that is
> in place, we could then look at improving the drafter via a ring too.

Three goals, in this order:

| # | goal | who it is for | status |
|---|---|---|---|
| **A** | **fittability**: a large context on one 32 GiB card instead of OOM; and free VRAM for the MoE arena in slab mode | dense 27B Q8_0, 35B-A3B Q6_K | **the ask — not started** |
| **B** | a *sparse* attended set streamed cheaply (QSA/DSA) | Flash-Next | bonus, same machinery, needs more hardware |
| **C** | the drafter ring | MTP draft | **built (O3), last in priority** |

That is **O2**, and nothing else in this campaign is it.  `README.md` §5's option table (O1-O6) was a menu,
and the two options that got built (`O1 --kv-grow`, `O3` drafter ring) were picked out of that menu by the
agent, never requested.

**And the goal is measurable HERE, which the record wrongly denied.**  The failure is two log lines:

| config (this box, 1 GPU) | KV | result |
|---|---|---|
| 27B Q8_0 (27.1 GiB) `-c 262144`, q8_0 KV | 8704 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, f16 KV | 8192 MiB | **`cudaMalloc failed: out of memory`** |
| 27B Q8_0 (27.1 GiB) `-c 131072`, q8_0 KV | 4352 MiB | fits, with ~nothing left over |

A windowed KV turns the first two rows into running configs.  That is the "before" number and it needs no
Flash-Next and no second GPU.

## 2. Why it kept drifting (the real cause, worth stating)

**The record itself said the ask was a Flash-Next-only feature, so each session believed the ask was
unmeasurable here and quietly substituted something that was.**  That belief was wrong (Goal A is measurable
on one card — the OOM table in §1 proves it), and it came from §4's *speed* eligibility table being read as the
whole story.

* O2's payoff exists only where the per-token attended set is **bounded** (`README.md` §4).  For full
  attention it is not: a token attends the whole used context, so every non-resident byte crosses the link
  every step — measured, `-nkvo` on a 4B at depth 4096: **100.67 -> 38.71 t/s**.  The eligible families are
  **qwen4exp QSA** (`indexer.top_k 2048 x compress_ratio 4` = 8192 cells of 262144, 3.1 %) and **DSA**, plus
  the drafter.
* The qwen4exp model on this fleet is Flash-Next at **75 GB**, and **one 32 GiB R9700 is attached**.  The
  shape that would *show* the win (small window, arena big enough for the expert set, 2-3 GPUs) cannot be run.
* So session 1 built the VMM probe + O1's primitive and called O1 "the cheap win, do it first"; session 2
  built O1 (the KV region buft) and measured a regression; session 3 built O3 (measurable on one card,
  tiny prize).  All three were measurable.  None was the ask.

**Rule from here: no substitute, and no speed-only framing.**  Fittability (Goal A) and speed (Goal B) are
different questions with different mechanisms; a decision that cites the speed rule to exclude dense models is
wrong by construction.  If the ask cannot be measured, say so and either build it for
*correctness* (explicitly, with no perf claim) or stop.  Do not pick a different option off the menu.

## 3. What O2 actually is — and Goal A needs a much smaller version than Goal B

**Goal A (full attention) needs NO page table, NO CLOCK victims and NO epoch stamps.**  Every decode step
reads the whole used context, so there is no temporal reuse to exploit and no eviction policy that can help:
every non-resident cell is read every step regardless.  The optimal mechanism is a **static split**:

* a **host master** (pinned, device-mapped) holds all K/V cells — authoritative;
* a **fixed resident window** holds the most recent `W` cells (the size is the budget the user/`--fit` grants);
* attention is split over the two segments (resident + host tail, read zero-copy at 29.0 GB/s from the pinned
  mapping) and combined — the FA band already splits over the KV axis and combines partials, so the shape
  exists;
* writers write the host master always and the resident cell when it is inside the window.

Cost, exactly: `(used - W) cells x bytes/cell / 28.8 GB/s` per token.  At `-c 131072` on the 27B with q8_0 KV
that is 2 KiB/cell... i.e. **the resident fraction IS the speed**, linearly, and the knob is `W`.

**Goal B (sparse attended set) is where the page table below is the point** — and there the reader already
walks a block list, so a residency map costs nothing at read time.

Per `STRATA-KV-STREAMING.md` (§1, §3) mapped onto this tree, with the Goal-A column marked:

| piece | this tree |
|---|---|
| **host master** — pinned, device-mapped, authoritative K/V (+ indexer keys), laid out exactly like the resident cache | new: we have no host KV at all |
| **writer rule** — write the host copy *always*, the resident slot only if the cell is resident | the KV writers are `SET_ROWS` / fused rope kernels; a streamed writer must do both |
| **residency map** — `cell-run -> slot`, or `-1` | new; `page = compress_ratio` cells for QSA |
| **in-graph resolve** — one kernel per attention call: stamp the touched cells, pick victims for the misses, copy them in | new; it must be **in the graph** (our decode is a captured HIP graph, so a host-side remap between steps is impossible) |
| **the reader** | the whole cost: free for QSA (its reader already walks a block list), expensive for full attention (this is the eligibility rule, not an implementation detail) |
| **VRAM** | a fixed pool of `W` cell-runs, known at context creation -> trivially reserved before the slab, no VMM needed (provider **P0**) |
| **budget cost** | a residency map is ~`n_cells/page x 20 B` = 1.2 MiB at 245760 with a 4-cell page (`INVENTORY.md` §3) |

**Constraint recorded from O3's session:** do *not* extend O2's page table / host master to the MTP draft
context.  The drafter only reads its last `W` cells, so a placement window on a small resident cache is
enough (`O3-DRAFTER-RING.md` §2.2.2) — a ring and a page table on the same cache would double the
indirection for nothing.

## 4. First deliverable — Goal A, on this box

1. **The split host master for the target's K/V**: pinned, device-mapped, authoritative, holding every cell;
   writers write it always, plus the resident cell when the cell is inside the window.
2. **The resident window `W`** (cells or MiB), sized by the user / `--fit`, resident in VRAM exactly like
   today's cache — so a config whose KV fits is byte-for-byte today's behaviour (`W = -c`).
3. **Two-segment attention**: resident segment as today, host tail read from the pinned mapping, partials
   combined.  `W` forced small is the test lever (the `LLAMA_KV_REGION_MAP_MAX_MIB` pattern).
4. **Gates**: (a) **byte-identical text** to the fully-resident arm at `W << context` — the window moves bytes,
   never values; (b) the two OOM rows above now *run*, and their t/s is reported against the modelled
   `(used-W)/28.8 GB/s` cost; (c) no GPU fault; (d) the host master's RAM cost, and its effect on slab mode's
   `SLAB_INIT` arena (the arena must get the freed VRAM — that is Goal A's second half).
5. **No claim beyond what is measured**: on this card, Goal A's claim is "runs where it used to OOM, at this
   t/s, with this resident window".  Goal B (the sparse page table) waits for Flash-Next + 2-3 GPUs, and the
   O1-vs-O2 crossover question (`README.md` §4.2) rides on the same measurement.
6. **O3 is already done** and is the last priority (goal C); it is closed as a standalone project and only
   returns as a companion once Goal A lands.

## 5. Closed, do not restart

| option | state |
|---|---|
| **O1** `--kv-grow` (lazy physical mapping) | **CLOSED as a detour.** Built (slice 1+2), measured, and it *regresses* a config that uses most of `-c` on a slab-armed MoE model (`IMPLEMENTATION.md` "KNOWN LIMIT").  Not on O2's path.  Keep only its **buffer-type skeleton** as the container O2 needs. |
| **O3** drafter ring | **Goal C — built, and last in priority (not a detour any more, once stated).** Built (patch `0003`), pure, measured: ~free at a usable window, but the prize is one draft layer (~1 GiB at `-c 262144`) and the freed VRAM does not reach the arena (`O3-DRAFTER-RING.md` §2.2).  Not a step toward O2; revisit only as a *companion* to O2 (§2.2.2), never as a standalone project. |
| **O4** indexer diet, **O5** non-KV metadata, **O6** provider P0 | not started; O6 is partly subsumed by O2's pool. |

Nothing from `archive/work/kv-residency/` is in `patches/`, `release.json` or any block, and none of it may be
promoted.  The fork branch `wip-kv-residency` is the record.

---

## 6. Goal A build contract (agreed 2026-10-29)

### 6.1 The opt-in rule (the maintainer's spec)

* **Default off.**  With the flag unset the behaviour is **byte-identical to today**, *including the OOM*:
  "it either fits or it doesn't".  Nothing about the resident path changes; no extra allocation, no extra
  tensor, no warning.
* **Opt-in.**  The user chooses to let the KV's tail spill into system RAM and accepts the consequence.  The
  flag says "do not deny me because the tail will spill"; it does not silently enable anything.
  This is the `--n-cpu-moe` category (a user-requested escape hatch), **not** the default-on category: it is
  not "on because it helps", it is "on because the alternative is not running at all"
  (`AGENTS.md`, default-on policy — a known-risky tradeoff may stay opt-in).
* **Loud.**  One warning at start-up, always (not only under `-v`), naming: resident MiB of total KV MiB,
  spilled MiB, the resident window in cells, and the **modelled** cost at the current depth
  (`spilled_bytes / 28.8 GB/s` per token).  Plus one line whenever the split is recomputed.
* **`--fit` reports, never repairs.**  With the flag on, an over-budget KV is not a refusal: `--fit` prints
  the split it would produce and the modelled cost.  With the flag off, `--fit` keeps refusing as today.

### 6.2 Where the split is decided

* **Before `SLAB_INIT`**, because the slab maps `free VRAM - reserve` once and the KV's resident part must be
  accounted for first — the arena then gets the remainder **by construction**.  (This is the O1 lesson,
  done deterministically: O1 grew lazily and the arena took the VRAM first.)
* Two ways to size the resident window: **auto** (as much as fits after the weights and the slab floor) or
  explicit (`--kv-resident-mib N`).  Auto is the maintainer's framing ("full speed until we run out of VRAM").
* **The host master is pinned** (a kernel reads it, and gfx1201 has XNACK off), so the spilled part is
  **non-swappable RAM**: 4.25 GiB at `-c 131072` q8_0 on the 27B, 8.5 GiB at 262144.  It must be gated on
  `MemAvailable` (the `MOE_HOST_POOL_RAM_PCT` pattern) and refuse loudly before `cudaHostAlloc` fails.

### 6.3 Mechanism — CORRECTED AGAIN 2026-10-29 (`MIXED-BACKING.md` is the design of record)

> **Read [`MIXED-BACKING.md`](MIXED-BACKING.md) instead of this section.**  The two-base read below was built
> (A1, gated, exact — [`A1-TWO-BASE-KV.md`](A1-TWO-BASE-KV.md)) and then superseded the same day: a probe
> (`mixed-backing-probe.cc`) showed that **one** VMM reservation can be backed partly by pinned host memory
> (device read 28.4 GB/s, device-writable, copy-engine-visible, re-mappable, 3264 chunks at real size).  That
> needs **no kernel and no op change** and leaves the in-place rope shift (defrag/`seq_add`), state
> save/restore, `clear` and the writers untouched — which the two-base read would each have had to split.  The
> two-base read stays as the fallback design and as the proof that a split KV read is an exact equality.

### 6.3 (historical) Mechanism — CORRECTED 2026-10-29 (A1 feasibility check; the combine route is not viable)

**The two-FA-calls-plus-combine design does not work, for a concrete reason: `ggml`/CUDA FA exposes no
log-sum-exp.**  Grepped: there is no `lse` / `softmax_lse` anywhere in `ggml/src/ggml-cuda/fattn*`.  FA
returns only the *normalised* output, and two normalised halves cannot be combined without each half's row
sum.  So the combine route needs an lse output from **every** FA family's epilogue (tile, vec, RDNA/WMMA, MMA,
QSA) — an epilogue change in the numerics of the most tuned code in the tree.  Rejected.

**Chosen instead: a two-base K/V read inside the existing kernels.**  The FA kernels already take one `KV`
base pointer plus a stride and index rows as `KV + i*stride_KV` (`fattn-tile.cuh:456-715`), and the *caller*
forms the per-tile base.  A split is therefore an **addressing** change, not a numerics change:

* the op carries a second K/V base (`k_tail`, `v_tail`) and the cell count of the tail segment (`n_tail`);
* wherever a tile's `KV` base is formed, the segment is chosen: rows below `n_tail` come from the tail base,
  rows at or above it from the head base (each segment contiguous in its own tensor);
* **the KV index space stays global**, so the mask, `cell_pos`, `tok_lo`/`tok_hi` and the derived-mask form
  are untouched — no mask slicing, no per-segment `cell_pos`, no second combine;
* with `n_tail = 0` every kernel indexes exactly as it does today, so the default path is bit-identical by
  construction, and with `n_tail > 0` the *accumulation order is unchanged* too (same rows, same order, only
  a different base address) — which is what makes A1's byte-identical oracle reachable rather than
  approximate.

**And A1 needs no change to the cache representation at all:** the two segments are two **views of the same
KV tensor** — tail = cells `[0, s)`, head = cells `[s, n_kv)`.  State io, defrag, `clear`, `seq_rm` and every
existing cell-range path keep working untouched, because there is still one tensor and one global cell index
space.  Only in **A2** does the tail's *storage* move (its view resolves to a tensor in the pinned host
master), and only then do the whole-cache walking paths (state save/restore, defrag) need a second pass.

Coverage plan: implement the two-base read in the families the target shapes actually select — the RDNA
**band** (RDNA4, head 256, GQA 5..12, `n_q <= 8`: that is the 27B's decode) and the **tile** family (prefill
and everything else).  Any other family, and any shape where the split cannot be served, **declines the
split** — and declining means "no spill", i.e. today's fit-or-fail path, which is the correct failure mode
for an opt-in feature.

### 6.4 Slice order, with the oracle at each step

| slice | what | oracle |
|---|---|---|
| **A1** | ~~the split mechanically, both segments still in VRAM~~ — **DONE, superseded** (`A1-TWO-BASE-KV.md`): byte-identical text at every split width on both families + MTP verify widths, KLD 0.000000 / 100.000 %, and a negative control that proves the second base is read | n/a (kept as the fallback) |
| **A2** | **mixed backing** (`MIXED-BACKING.md`): one VA range, resident cells mapped from VRAM, the tail from pinned host chunks, at 2 MiB bulk granularity with the per-tensor boundary at 4 KiB.  Split decided before `SLAB_INIT`. | text identical at several window widths (with a mis-mapped-chunk negative control), KLD 0.000000 / **the two OOM configs now run** at the measured `(used−W)/28.8 GB/s` cost, `gate-ckpt-reuse.sh` + `gate-state-pipe.sh` (state io now crosses the boundary), `SLAB_INIT` shows the arena gaining what the window frees |
| **A3** | auto-size from free VRAM, the opt-in flag, the loud warning, `--fit` reporting, the `MemAvailable` gate | the two OOM configs in §1 now run; the flag-off run is byte-identical to today's binary; slab mode's `SLAB_INIT` shows the arena gaining what the window frees |

A1 is the whole risk and it is entirely in the resident path, so it can be gated before a single byte goes to
host memory.  A2 is then a genuine one-line move, which is the strongest evidence that the split is right.
