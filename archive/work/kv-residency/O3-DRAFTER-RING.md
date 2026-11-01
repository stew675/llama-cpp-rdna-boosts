# O3 — the drafter ring (windowed draft KV): design note

Started 2026-10-29, per the scope decision in [`README.md`](README.md) §4.1 (O1 parked; **O3 first, then
O2**) and `TODO.md` §117.

**Nothing here is delivery work.**  It describes the campaign's next slice before any of it is written.

---

## 1. What O3 is, and why it is first

The MTP draft context carries a KV cache of its own, sized `-c` cells (1 layer): measured 256 MiB at
`-c 65536` on the 27B, 1 GiB at `-c 262144` (`INVENTORY.md` §1.A/§2).  The drafter only ever needs its
**recent** prefix — Strata's `--mtp-window` (default 32768) is exactly this — so O3 replaces
`O(n_ctx)` draft KV with a **fixed window** of `W` cells and a static residency map
(`block -> block % n_slots` in Strata; see `STRATA-KV-STREAMING.md` §1).

It is *approximation* by design, and legal only because **the target verifies every token**: a
window-limited draft can change speed, never an answer.  That makes its cost measurable as a single number
(draft acceptance / accepted-per-step), which is why it is the right rehearsal for O2's policy.

Honest value statement: O3's *standalone* VRAM win on the box we have is modest (256 MiB at `-c 65536`,
1 GiB at `-c 262144` on the 27B — one draft layer).  Its real value is that it is the **smallest complete
instance** of the machinery O2 needs: a fixed window, a residency/placement map, a documentable acceptance
cost, and a policy that must survive rollback (a rejected draft), `seq_rm`, and a prompt-cache resume.
O2's K/V page table is the same three pieces with a bigger payload.

## 2. The finding that should make this cheap: the SWA cache is already a windowed cache

`llama_kv_cache_iswa` sizes its sliding-window cache to the window, not to the context —
`llama-kv-cache-iswa.cpp:73`:

```c
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
```

So `llama_kv_cache` **already** runs correctly with fewer cells than the positions it holds: cells are
recycled as the window slides, positions stay absolute, and the mask is limited by
`llama_hparams::is_masked_swa()` (`llama-kv-cache.cpp:1476`, `:2137`, `:2803`).  That is precisely the ring
O3 wants, and it is already gated, tested and shipped (block 05/12 territory).

**Therefore the first thing to try is not a new ring implementation — it is to construct the draft
context's cache with a sliding window of `W` cells.**  If that holds, O3 becomes a *configuration* change
plus an env knob, and the interesting work moves to proving the acceptance cost and that rollback is
sound.

## 2.1 Step A result (2026-10-29) — the ring is STRUCTURAL; do not route it through SWA

Step A asked "can the draft cache take a window at all?"  Reading the code answered it, and changed the
plan for the better:

* **The mask is a per-*context* term, not a per-*model* one.**  `fill_mask(...)` is fed `hparams.n_swa` /
  `hparams.swa_type` (`llama-graph.cpp:610-620`, `:593` uses `cparams.causal_attn`), and `hparams` is shared
  between the target and the draft context (MTP builds the draft from the same model).  So a draft-only
  window *can* be expressed — but only by adding a `cparams`-level window term and threading it into the
  mask fill, the `self_kq_mask_swa` builder (`hparams.swa_type != NONE` gates its very existence), the cache
  sizing, and the cache's own `n_swa`.  That is 4 sites and a new cparams field.
* **We do not need that.**  Strata's drafter is "a ring, not a page table: a static table
  `block -> block % n_slots` and **no resolve**" (`STRATA-KV-STREAMING.md` §1).  A ring enforces the window
  *structurally*: a cell that has been recycled simply holds a different token, so the drafter attends
  whatever is in the cache and the approximation comes from the ring, not from a mask.  **No mask change and
  no `n_swa` are required.**  The mask stays what it is (absolute positions, so causality is still exact);
  the only loss is that tokens older than the ring are gone.
* **What the ring does need is a placement policy**: `cell = pos % n_slots` instead of `find_slot`'s free-cell
  search, plus `kv_size = PAD(W * streams + n_ubatch)` (the formula `llama_kv_cache_iswa` already uses for its
  SWA cache — reuse the sizing, not the masking).  That is the same shape as the idle-slot parking feature
  (a cell-placement policy inside `llama_kv_cache`), not a new cache class.

**The one real hazard this uncovered.**  `kq_mask_derivable()` (`llama-kv-cache.cpp:2307`) builds the derived
mask from a **`tok_lo`/`tok_hi` position *range*** over cell indices, i.e. it assumes cell order ≈ position
order.  A ring breaks that, so a ringed cache must **not** take the derived path (force the packed mask) —
otherwise the range silently mis-masks.  Note also that `kq_mask_derivable()` does **not** reject
`n_swa != 0`: the derived predicate has no window clause at all, so *if* anyone ever does route a window
through SWA, that path would attend outside the window with no error.  Worth a guard when this is
implemented, independent of which route is taken.

Consequence for the plan: **Step A is done and Step B is smaller than expected** (no cparams field, no mask
plumbing); the derived-mask exclusion is a new, explicit item.  Revised order: B (placement + sizing) ->
C (rollback) -> D (acceptance).

## 2.2 Step B implemented and measured (2026-10-29) — it works, and it is a poor trade

Implemented as a single change plus a helper: the MTP `create_memory` branch of the hybrid Qwen3.5 path
(`llama-model.cpp`, the `GGML_ASSERT(!hparams.is_swa_any())` plain `llama_kv_cache` for the nextn layer) builds
the draft cache at `GGML_PAD(min(n_ctx_seq, win + n_ubatch), 256)` cells with `n_swa = win`,
`swa_type = STANDARD`, when `LLAMA_SPEC_DRAFT_WINDOW` (cells) is set.  No mask change, no new allocator, no
`cparams` field: the reclaim in `find_slot` is the ring, and `set_input_kq_mask` already reads the cache's own
`n_swa`.  Floor: `GGML_PAD(n_ubatch + 256, 256)` (a whole ubatch must always be placeable).

**Purity — green.**  Target greedy text is byte-identical with the window off and on: 27B 303 chars
`sha=c437759e06e3` at off / 4096 / 2048 (`-c 8192`, 5256-token prompt so the window is genuinely active);
35B-A3B `-ncmoe 24` 262 chars `sha=e9db5cb8db0b`.  `check-gpu-faults.sh` clean.

**The draft cache shrinks as designed:**

| model / `-c` | draft KV off | window 4096 | window 2048 |
|---|---:|---:|---:|
| 27B, 16384 | 64 MiB | **18 MiB** | — |
| 27B, 12288 | 192 MiB | **72 MiB** | **36 MiB** |
| 35B-A3B, 131072 | 256 MiB | **9 MiB** | — |

The ceiling is structural: the draft is **one layer**, so the saving is one layer's worth of KV
(~2-4 KiB/token) — about **1 GiB at `-c 262144` on the 27B**, not GiBs.

**The cost, 27B `-c 12288`, 5256-token prompt, `-n 1000`, `--reasoning on`, `--spec-draft-n-max 3` (1 GPU):**

| window | acceptance | gen t/s |
|---|---|---|
| off | 0.6381 | 63.24 |
| 8192 (never truncates — the sanity arm) | 0.6381 | 63.13 |
| 4096 | 0.5712 | **58.79 (-7.0 %)** |
| 2048 | 0.5519 | **57.61 (-8.9 %)** |

35B-A3B `-ncmoe 24 -c 131072`, `-n 200`: acceptance 0.7611 -> 0.6927 (-9.0 %), generation 193.6 -> 181.8 t/s
(-6.1 %), **prefill 1956 -> 2116 t/s (+8.2 %)** — the smaller draft cache makes the draft's own prefill cheaper,
and that is the only thing that got faster.

**The finding that decides O3's value: the freed VRAM does not reach the arena.**  On the MoE arm
`SLAB_INIT` reads `arena 14.44 GiB` with the window **off and on**, because the draft context's KV lives inside
the slab's `aux_reserve` (4.25 GiB on that arm) — a reserve sized *for* the draft and otherwise unused, so
shrinking the draft just leaves part of the reserve unspent.  To convert O3 into an arena win the draft-reserve
estimate has to know about the window (`common/draft-reserve.h` / the `--fit` measurement of the extra model),
which is a separate, small change — and even then the prize is one layer (~220 MiB there).

### 2.2.1 CORRECTION — at a *usable* window O3 is essentially free

The numbers above were measured at windows (2048/4096) far below the context, i.e. on the **tail** of the
cost curve, and the blunt "poor trade" verdict drawn from them was wrong.  Re-measured with the window at a
realistic size (**27B, `-c 65536`, a 46k-token context, `-n 400`, `--reasoning on`, `-nmax 3`, 1 GPU**):

| window | context truncated to | acceptance | gen t/s | prefill t/s |
|---|---|---:|---:|---:|
| off | 46k | 0.5499 | 50.61 | 990.9 |
| 16384 | 35 % of the context | 0.5371 | **50.79** | 1001.4 |
| 32768 | 72 % of the context | 0.5383 | **50.34** | 994.1 |

Acceptance -2.3 % relative, **generation flat** (50.61 vs 50.79 / 50.34, i.e. within the run-to-run spread) and
prefill flat.  The cost of a window is therefore a function of *how much of the context it drops relative to
what the drafter actually uses*, and the drafter works off the recent ~10-30 k tokens: truncating a 5 k context
to 2-4 k is expensive, truncating a 46 k context to 32 k is free.  **With `W` at or above the working context
the feature costs nothing measurable and still removes the whole draft KV** (256 MiB at `-c 65536`, ~1 GiB at
`-c 262144`) — which is exactly Strata's reason for shipping `--mtp-window` default-on at 32768.

So the two numbers to carry forward are:
* **`W` must be chosen from the working context, not the context limit.**  A window well below it is a real
  -7..-9 % decode regression (the 4096/2048 rows); a window at/above it is free.
* the VRAM prize is one draft layer (2-4 KiB/token): ~1 GiB at `-c 262144` on the 27B.

### 2.2.2 What O2 changes for O3 (and why O3 "comes into its own" then)

Two independent reasons, both architectural rather than magnitude:

1. **The budget.**  Today the draft's `-c`-sized KV is invisible beside a 16 GiB target KV.  Once O2 makes the
   target's *resident* KV a window, the draft's fully-resident `-c` cache becomes the largest remaining fixed
   KV term in the same budget -- so removing it stops being a rounding error and starts being the next thing
   the arena wants.
2. **O3 is the cheap answer for the one cache O2 should NOT page-table.**  If the target gets a residency page
   table, a resolve kernel and a host master, the drafter would otherwise need all three as well (a second page
   table, a second resolve, a second master) or stay as a resident `-c` exception.  O3 says a *placement policy
   on a small resident cache* is sufficient, because the drafter only ever reads its last `W` cells -- which is
   precisely Strata's split (page table for the K/V, ring for the drafter, `STRATA-KV-STREAMING.md` §1).

**Constraint this puts on O2:** do not extend O2's page table / host master to the draft context.  The draft
cache stays a plain resident `llama_kv_cache` with a window; a ring and a page table on the same cache would
double the indirection for no gain.

**The one thing that still does not follow from O2:** the freed VRAM reaches the arena only if the draft-reserve
estimate knows about the window (`common/draft-reserve.h` / the `--fit` measurement of the extra model).  O2
does not fix that; it makes it matter more.

## 2.3 A15 (2026-10-31) -- the adaptive policy, and the reserve conversion that makes it pay

> **DELIVERED in r193** (`v16-6050b467c-r193`, block 15 amended): `LLAMA_SPEC_DRAFT_WINDOW`, default ON with
> the adaptive policy.  The gates listed as owed at the end of this section were run and passed before the
> fold; `WORKLOG.md` 261031.014 has the numbers and `patches/README.md` block 15 ("r193") is the shipped
> description.  What remains of the campaign is `TODO` #123-#125 and the decided-but-undocumented `#122`.

Branch `wip-kv-residency-c` (off the r191 tip, commit `fa914676e`, pushed).  The maintainer's decisions:
**MTP only** (DFlash2 later), the window is **adaptive** -- a quarter of the context, clamped to
**[32768, 65536] cells** -- the user override stays, and the default follows the delivery's own rule: **on when
it measures positive, opt-in when it measures negative**.  It is measured positive, so the unset default is
the adaptive policy and `LLAMA_SPEC_DRAFT_WINDOW=0` is the kill switch.

**Two pieces, and the second is what makes the first worth shipping.**
1. `llama_spec_draft_window_cells()` (`src/llama-ext.h`, implemented beside `get_rope_factors`) holds the
   policy; `create_memory()`'s MTP branch builds the draft cache at `min(n_ctx, W + n_ubatch)` cells with
   `n_swa = W`.  Unchanged from §2.2 otherwise: the reclaim in `find_slot` *is* the ring.
2. **The O3 §2.2 blocker is now fixed.**  An MTP draft has no GGUF of its own, so `draft_reserve` was never
   derived for it: the slab held the flat `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` (3584 MiB) and the freed
   VRAM stayed unspent (`SLAB_INIT` arena identical, which is the note's own finding).  For an MTP draft whose
   window is active, `common_draft_reserve_estimate()` now runs on the **target's** metadata with the weights
   zeroed (`shares_model`; they are the target's) and the KV term sized by the window: **aux 3.50 GiB ->
   0.76 GiB**, i.e. the arena gains ~2.8 GiB.  Without an active window nothing changes (the flat constant
   stays), so no existing run moves.

**Measured (one R9700, 1 GPU, `-n 2000`, static depth 3, `--single-turn --no-display-prompt`, interleaved pairs).**

| arm | arena | residency | generation | drafts accepted |
|---|---|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 48 -c 262144` q8_0 KV, 51 433-token prompt, window **0** | 20 910 MiB | 64.1 % | 108.5 / 111.8 t/s | 1315/2052 |
| the same, **auto** (65536) | **23 715** | **72.7 %** | **114.0 / 114.9** (+4.9 / +2.8 %) | 1315/2052 (**identical**) |
| the same, `W=32768` (the truncating case) | 23 715 | 72.7 % | 110.3 (+1.7 %) | 1308/2071 (-1.4 % rel) |
| dense 27B UD-Q4_K_XL `-c 102400`, window 0 -> auto | -- | -- | 66.6 -> **68.6** (+3.0 %) | 1925/1357 (identical) |

Prefill moved +3.4 % on the arena arm (1095.1 -> 1132.8).  Target greedy text is **byte-identical** across
off / auto / 4096 (27B, `-c 102400`, `-n 64`: 303 chars `sha=c437759e06e3`, the §2.2 hash).  At `-c 16384` the
adaptive window is absent (32768 >= `n_ctx`) and no log line appears -- "nothing to save" is the designed case.

**A correction to §2.1.**  The "one real hazard" there -- that a ring must be excluded from
`kq_mask_derivable()` because its `tok_lo`/`tok_hi` assume cell order ≈ position order -- **is not a hazard**:
the predicate is `tok_lo <= cell_pos <= tok_hi` over the **per-cell position** (`cells.pos_get(j)`), and `lo`
is a position bound (`p1 - n_swa + 1`), so a recycled cell reports its new position.  The measured text
identity is the same statement from the other side.  Nothing is owed here; do not add the exclusion.

**Still owed before it is delivered into block 15** (the maintainer's rule for any Goal-C promotion):
* MTP Protocol A at `-n 3000` (`-n 2000` floor), reasoning-pinned, **both split modes**, device count quoted --
  the §2.2.1 "free" reading came from `-n 400` arms, which is below the methodology's floor;
* the **Step C rollback oracle** (§5), never run: W=512 against a 3000-token run, `seq_rm`/`seq_cp`, "no
  reused cell attended after a rollback", target text unchanged;
* `gate-park-idle-slots.sh` / `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` on a **windowed draft** (the draft
  context is a second context with its own cache, so a state path could still reject it);
* the `--fit` half: its preflight still books the flat 3584 MiB for the draft (`common_get_device_memory_data`
  -> the probe preflight), so `--fit` reserves ~2.8 GiB more than the slab now does.  Conservative (it can
  only under-size the arena), but it is the same number and should be the same value.

## 3. Invariants the slice must keep

1. **The target is untouched.**  Only the draft context's cache gets a window.  A wrong draft is a slower
   run, never a wrong answer; the target still verifies every token.
2. **Positions stay absolute.**  A window limits *which cells are attended*, never the position a token is
   placed at (RoPE, and `get_n_kv`'s comparison against the cache's positions, both depend on this).
3. **Rollback must still work.**  A rejected draft is rolled back with `seq_rm` (+ `seq_cp` for the accepted
   prefix).  A windowed cache must not resurrect cells whose position has been re-used by a newer token —
   this is the one place a ring can be silently wrong, and it is what the gate has to check.
4. **The window must be at least the verify width.**  The draft attends the cells it just wrote plus the
   prefix the target's verify step needs; `W < n_max + 1` cannot work.  Clamp to a floor (a few hundred
   cells) rather than trusting the knob.
5. **Default OFF, and no effect when off.**  Same rule as every WIP slice: `LLAMA_SPEC_DRAFT_WINDOW` (0 =
   today's behaviour, byte-identical).
6. **`-ctkd`/`-ctv` interaction (r125).**  The draft KV follows the main `-ctk`/`-ctv` unless pinned.  A
   window changes the cell *count*, so any VRAM arithmetic in the docs/`--fit` must use the windowed size.

## 4. What it must not do

* Do not touch the main (target) cache's `n_swa`/`swa_type` — a dense model's full attention must stay full.
* Do not make the window the default in `wip/` (the promotion rule inverts a default only after the gates).
* Do not claim an acceptance number without the MTP methodology's rules: `-n 3000` (`-n 2000` floor),
  `--reasoning on` for R and off for P/C/K, and the device count quoted with the number
  (`benchmarks/mtp-adaptive-methodology.md` rule 0; AGENTS.md "MTP gates must be long and reasoning-pinned").

## 5. Plan

**Step A — DONE (see §2.1).**  The ring is structural; do not use the SWA mask.  New sub-item: a ringed cache
must be excluded from `kq_mask_derivable()` (its `tok_lo`/`tok_hi` range assumes cell order ≈ position
order).

**Step B — the placement policy, the knob and the floor.**  `cell = pos % n_slots` for the draft cache only,
`LLAMA_SPEC_DRAFT_WINDOW=N` (cells, 0 = off), floored at `n_max + n_ubatch`; exclude the cache from the
derived mask; log the draft cache's size so the VRAM win is visible in the KV log line.

**Step C — correctness of rollback under a window.**  The oracle is not text: run at a window small enough
to overlap constantly (e.g. 512 cells against a 3000-token run) and require (i) no abort, (ii) the target's
greedy text unchanged from the unwindowed arm (it must be — the target verifies), (iii) no cell whose
position was reused is attended after a rollback.  A purpose-built check beats a text diff here; the
`LLAMA_RS_ROLLBACK_DEBUG` / `LLAMA_KV_CACHE_DEBUG=2` style of trapdoor is the precedent.

**Step D — the acceptance cost, measured properly.**  `llama-cli --spec-type draft-mtp -n 3000` on the 27B,
window off vs `W` in {4096, 16384, 32768, 65536}, reporting acceptance and t/s on both split modes, on the
device counts the box allows (**one R9700 today — say so**).  Compare against the recorded reference
(3 devices: layer 51.7 / tensor 90.7 t/s, acceptance 0.62 / 0.60; `AGENTS.md`) only where the device count
matches.

**Step E — the crossover question that parked O1.**  With a window in hand, the O1-vs-O2 question from
`README.md` §4.1 becomes measurable *in principle*: past the window, is `W`-windowed streaming cheaper than
a shrinking arena?  Answering it properly still needs O2's host master, so record the windowed arm's numbers
now and leave the comparison to the O2 session.

## 6. Gates this slice owes

| gate | bar |
|---|---|
| coherence / purity | target greedy text **byte-identical** with the window off and on (the target verifies; nothing may move) |
| MTP Protocol A | `-n 3000`, reasoning-pinned, both split modes, device count quoted; report acceptance and t/s against the unwindowed arm |
| rollback | the Step C oracle: no abort, no reused cell attended after `seq_rm` |
| VRAM | the draft context's `llama_kv_cache: size = …` line scales with `W`, not `-c` |
| prefill logits | not required for a draft-only change (`gate-prefill-logits.sh` measures the *target*'s prefill), but state that explicitly rather than leaving it implicit |

## 7. Open questions for the maintainer (do not guess)

1. **Is the window an approximation we ship on by default** (Strata ships `--mtp-window 32768` default-on and
   reports the cost), or opt-in only?  `README.md` O3 lists it as a shipping candidate; the delivery's
   default-on policy says a beneficial, gated feature is ON — so this needs the gate numbers first.
2. **One window size, or per-model guidance?**  Strata's 32768 is their tuned value; ours would be
   payload- and acceptance-dependent (`blocks 13`'s `--spec-draft-mtp-vocab` had the same shape of answer).
3. **Does the same window belong on a DFlash2 draft** (a separate draft model, `-md`), or is this MTP-only?
