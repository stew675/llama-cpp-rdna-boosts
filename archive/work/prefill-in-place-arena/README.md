# prefill-in-place-arena — serve the wide band IN PLACE from the GTT arena (`TODO` #317)

**Status: CLOSED (r266) — PROMOTED into block 16.**  The in-place prefill band shipped as the block-16 r266
amendment: the admission ladder (empty slot -> landing slot -> coldest NON-routed resident by the decode heat
prior), the prefill-width remap, the finite-head guard over slot order, and the coalesced fill engine (which also
carries the `O_DIRECT` offset round-out, P2a).  Numbers and gates:
[`benchmarks/2026-11-12-prefill-in-place-arena.md`](../../../benchmarks/2026-11-12-prefill-in-place-arena.md),
`WORKLOG.md` 261112.001.  The handover prompt below is kept as the record of what was scoped, not as live work.

**Status: OPEN — scoping note + a measured reproduction.  Nothing implemented; nothing in `patches/`.**
**Handover prompt for driving it: [`NEXT-SESSION-PROMPT.md`](NEXT-SESSION-PROMPT.md)** (self-contained; paste it
into a fresh session).

Opened 2026-11-11 (r265), the same session that fixed the block-16 prefill **read plan**
(`benchmarks/2026-11-11-gtt-prefill-span-set.md`, `WORKLOG.md` 261111.036).  That fix removed the reads of
experts the ubatch never routes to.  This campaign is about the cost that remains.

## 0. The measured problem

On `halo` (gfx1151 APU), `Qwen3.5-122B-A10B Q5_K_XL` (81.06 GiB expert set, 147 tables = 144 trunk + the 3-table
MTP head), `-lme 75` ⇒ **ARENA** shape, **241/256 slots per trunk table** (78 239.3 MiB, 94.3 % of the set):

| a **38-token** prompt, one pass | value |
|---|---:|
| used experts (of the tables' 144) | 31 639.1 MiB |
| read from storage (the true misses, after r265) | **1 578.6 MiB (5.0 %)** |
| **D2D-copied out of the arena** into the split-input copy | **30 060.5 MiB (95.0 %)** |

So the *per-pass* cost is `O(used)` bytes copied, not `O(missing)` read.  At `-c 102400 -ub 4096` a prompt is ~25
passes ⇒ **~1.85 TiB of D2D copies** for one long prompt, plus a WebUI-style request that lands in 3 passes still
copies ~90 GiB for a 34-token prompt.  The copies are on the compute stream, so they are serialised against the
kernels and they compete for the APU's DRAM bandwidth.

## 1. Why it cannot be avoided today (the three gates)

The **identity** tier (slot == expert, whole table resident) is already served in place at *any* width — r258 /
`TODO` #133, `moe_cache_get_table()`'s identity branch, redirected in `ggml-cuda.cu`'s `MUL_MAT_ID` dispatch.  The
**arena** tier (partial residency) is not, in three places:

| gate | where | today |
|---|---|---|
| the eager admission + remap declines a prefill ubatch | `moe_cache_update_host()`, `if (n_tok > MOE_EXPERT_CACHE_MAX_TOK) { ...tally...; return false; }` | `# decode/verify band only` |
| the consumer refuses the remap above the band | `moe_cache_get_table()`, `if (op->ne[2] > MOE_EXPERT_CACHE_MAX_TOK) return false;` | only the identity branch is width-free |
| the remap staging buffer is sized for a decode step | `t.remap_cap = t.n_experts * 8` (2 048 int32) | `n_used x n_tok` at `-ub 4096` is **32 768** |

Keep them in this order: `MOE_EXPERT_PREFILL_IN_PLACE` (default ON since r258) is what already allows *identity*
in place above the band; `moe_cache_width_serves_in_place()` is its one entry point.  The remap path additionally
needs the routing, which the scheduler **already** reads back before the prefill hook runs (it builds the
used-expert bitset and hands it over), so no new synchronise is owed.

## 2. The shape (the plan)

**P1 — admit + remap at prefill width (the whole win).**  For each table: for every `used && !resident` expert,
claim a slot (the existing LFRU admission, one buffered `pread` straight into it), then publish the slot-remapped
ids for the whole `n_used x n_tok` routing and let `MUL_MAT_ID` read the arena in place.  Per pass that is
**0 copies + the miss fills** instead of 30 GiB of D2D.  The remap staging is **already per table**
(`table_t::remap_host`/`remap_dev`, published through `remap_n_used`/`remap_n_tok` and consumed by
`moe_cache_get_table`), so the wide band needs *capacity*, not a second buffer: `remap_cap` is `n_experts * 8`
(2 048 int32) and must hold `n_used x n_tok` = **32 768** at `-ub 4096` -- 128 KiB per table, ~19 MiB across the
147 tables of the field config.  **Do not share one remap buffer across tables to save that memory**: the scheduler
issues every table's input work before the split's graph runs, so a shared buffer would be overwritten while the
previous table's op is still reading it (that is the case that would need a ping-pong; at 19 MiB it is not worth
it).

**P1a — the MMQ finite-head guard is mandatory here.**  Above the band the `MUL_MAT_ID` dispatcher may pick MMQ, whose
tail over-read runs 512 B past a routed expert's end into the *next* slot.  In the identity tier the next slot holds
the next expert (finite weights); in a partial arena it can be an **uninitialised** slot ⇒ NaN.  The prefill stream
already lays down one zero per used-run successor (`g_pf_head_zeroes`); the in-place path needs the same, over
*slot* runs.

**P1b — the eviction RANK is the open decision.**  A full arena + a 100 %-used pass cannot admit a miss without
evicting a resident, and both were used.  Two policies, and they must be chosen by measurement, not taste:
* **(a) rank by the DECODE heat prior** for prefill evictions (`heat_rank_vec()`), so a prefill pass evicts the
  experts decode cares least about.  Residency stays 94 % and the same ~11 misses/pass are re-read (the byte cost is
  already irreducible at this budget); the D2D disappears.  Cheapest to build, and it reuses the prior that already
  sizes the arena.
* **(b) a per-table "landing" reserve** (e.g. 16 slots carved out of the 256) that only prefill misses may use, so
  the decode resident set is never touched.  Costs `16/256 = 6.25 %` of the arena (≈ 4.8 GiB of 75 GiB) and buys
  a decode hit-rate that is provably unperturbed by a prefill.  Measure both on the MTP four-axis protocol.

**P2 — the fill shape.**  The store is host-addressable, so a table's miss set is one `preadv` per table (an iovec
per run) straight into the slots it will be read from — no span ring, no bounce, no H2D (`MOE_EXPERT_GTT_FILL_DIRECT`
already reads into the slot; this batches it, and removes the per-run `pread` latency that dominates a *small*
prompt's miss set: measured 684 spans / 61 ms for 1.6 GiB after r265, at 2.31 MiB/span).

**P2a — a MEASURED defect this rewrite must also fix: the span read does not honour `O_DIRECT` alignment.**
`moe_cache_prefill_stream()` computes an aligned *destination* (`span_buf_acquire(..., align)`) but issues
`pread(fd, dst, len, t.src_offs + e*host_bytes)` with the raw GGUF offset, which is only `general.alignment`
(32 B) aligned.  With the `dio` arm active every such read is `EINVAL`/short, so **every span silently falls back
to the per-expert source** — measured on the 122B (`-lme 75`, 24 576-token prefill, `--load-mode dio`):
`used 519 970.8 MiB | read 0.0 MiB in 23 985 coalesced span(s) of 0.00 MiB | ... fallback 24 486.7 MiB`, hook
10 253 ms against 3 146 ms on the buffered arm.  `stage_span_fill()` (the discrete pool path) already does this
correctly — it rounds the offset down, reads `blen`, and carries a `skip` delta — so the fix is to reuse that shape
(round-out + delta, or `preadv` into the destination slots, which needs no rounding at all when the destination is
the registered region).  Until it lands, the `dio` arm is a "read the mapping one expert at a time" arm on this
tier, and the archive's "O_DIRECT is 11x slower here" reading is that, not the drive.

**P3 — overlap (only what is left).**  The hook is called per MoE split (144 splits per pass, measured), so the
host is already ahead of the GPU when it issues a table's work; with P1 the work per table is a handful of fills, so
an explicit worker/event pipeline is a refinement, not the win it would have been on the copy path.

**P4 — the eager seed (separate, and it is the *small prompt*'s real fixed cost).**  `apply_heat_seed_locked()`
fills every slot at sizing: **75 GiB / ~30 s** on this box, against 1.3 s of prompt eval.  With P1 the seed can be
lazy (fill on first miss; the prior only needs to *rank* evictions).  The discrete tier has the same shape — measured
on `soar`: 43.90 GiB / 80 446 memcpys = **57.0 s** cold, 3.7 s warm — so whatever the decision, it should be one
policy, not two.

## 3. What this is NOT

`TODO` #273 refuted an **in-place/remap transport for the partly resident discrete pool tier**, because there the
predictor guessed the *complement* (which experts are missing) and the transactions did not pay.  Here the arrow is
reversed: the residents are **94 % of the table and already in one addressable region**, so this removes a *copy of
the residents*, and the misses are filled through the same path that already fills them.  r258 already did exactly
this for the identity case at any width and measured +171 % prefill (1489.6 -> 4033.9 t/s, 3 GPUs, 35B-A3B).  Do not
merge the two discussions.

## 4. Traps (each one is already documented somewhere in the tier's record)

* **The scheduler gives a prefill graph ONE `input_cpy` per copy id for every table.**  Measured here again
  (144 splits, one destination): so any design that keeps the copy must own its own destination; P1 sidesteps it.
* **One address.**  The store's premise is `iommu=off` ⇒ host VA == device VA, asserted at creation.  Host writes
  (a `pread` straight into a slot) are visible to the GPU with no flush (measured, and the arena's fills rely on
  it), which is what lets P1's remap be a plain host upload.
* **Do not re-read the routing ids in a hook.**  Take the scheduler's used-expert bitset (the ids buffer is reused
  by the graph allocator; a stale read was measured as a 509-vs-522-char text difference).
* **The fusion guard must agree per table.**  `moe_cache_table_serves()` is identity-only today; a remap-served
  table makes `ggml_cuda_try_fuse`'s `moe_cache_get_table` redirect sensitive, so any fused op that pre-dispatches
  a node has to stand down exactly where the remap does.  `scripts/gate-fusion-alias-copy.sh` is the instrument.
* **Multi-device is untested** (the APU box has one GPU).  The remap buffers, the landing reserve and the budget
  arithmetic are all per-device.
* **The census must stay honest.**  Add a `prefill in place:` line (tables served, slots admitted, evictions, remap
  bytes) — a mechanism that removes 95 % of the traffic is invisible to text and to the existing counters.

## 5. Gates (when it becomes code)

`scripts/gate-uma-experts.sh` (the tier's own; add an in-place prefill arm asserting `arena D2D -> ~0` **and**
unchanged text) + `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` (mandatory: prefill path) +
the MTP four-axis protocol (`benchmarks/mtp-adaptive-methodology.md`) for the decode hit-rate cost of P1b +
`scripts/gate-prefill-logits.sh`'s identity-arm comparison + a **short-prompt** arm (the arm that exposes this
class: a wide prefill cannot, see `INVARIANTS.md` r265) + `scripts/gate-fusion-alias-copy.sh` if any fused op
reaches the arena + `check-gpu-faults.sh`.

## 6. Open questions for the maintainer

1. **P1b: (a) heat-ranked eviction or (b) a landing reserve?**  (a) is smaller and keeps the whole arena for decode;
   (b) is provably non-perturbing and costs ~6 % of the arena.  Both are measurable in one session.
2. **Should prefill admit at all**, or should the misses be filled into the *landing* slots and never promoted
   (i.e. prefill neither pollutes nor benefits from residency)?  That is (b) with a fixed landing set.
3. **P4: lazy seed, smaller seed, or keep the 30 s front-load?**  It is the dominant cost of a *small* prompt on a
   fresh load, and the same question exists on the discrete tier (57 s).

**Effort.**  P1 + P1a: ~150 lines in `moe-expert-cache.cu` (the admission/remap branch, the two remap buffers, the
slot-run guard, the census) — no scheduler change, no new op, no new env (`MOE_EXPERT_PREFILL_IN_PLACE` is the kill
switch).  P2: ~60 lines (a `preadv` fill).  P1b is a measurement + a ranking function.  P4 is a policy change in the
seed path.
