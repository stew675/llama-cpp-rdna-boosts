# CLOSEOUT — `kv-residency` archived 2026-11-01 (r197)

This directory is the **closed** record of the KV-residency campaign (`TODO` #118, the KV cache's VRAM
footprint: windowed/streamed residency, the fixed-region half, and the cache's own bookkeeping).  It was
archived when all three of its goals had shipped and everything left was either a decision, a gate or an
adjacent measurement — those leftovers are now their own campaign,
[`wip/kv-residency-cleanup/`](../../../wip/kv-residency-cleanup/README.md).

**Nothing here is applied to the fork, and nothing here may be** (`AGENTS.md`, WIP/promotion rules).  The
shipped form of the campaign's work is `patches/` **block 15**, described in `patches/README.md` and
`ENVIRONMENT.md` §17.

## What shipped (do not re-measure, do not re-promote)

| goal | what | shipped |
|---|---|---|
| **A — the resident window** | `LLAMA_KV_RESIDENT_CELLS` (opt-in): the newest `W` cells in VRAM, the tail in pinned system RAM in one VA reservation, no kernel change; `--fit` sizes `auto`; `LLAMA_KV_SPILL_RAM_PCT` caps the pinned spill | block 15, **r178** (the `auto` policy r179-r181, the spill-purity fix r182) |
| **B — the attendance pool** | `LLAMA_KV_REGION_POOL` (opt-in): the authoritative KV in a pinned host mirror plus a compact device pool of indexer-block pages, resolved by `GGML_OP_KV_POOL`; a streamed cache takes the sparse QSA decode arm | block 15, **r190** (folded into block 15 in r191; the mirror grows on demand r196) |
| **C — the MTP draft window** | `LLAMA_SPEC_DRAFT_WINDOW`, **default ON**: the embedded MTP head's draft KV is a window, and its slab reserve is derived with it (aux 3.50 → 0.76 GiB, arena +2.8 GiB) | block 15, **r193** |
| the B2 policy slice | `is_streaming()` makes a streamed cache take the sparse arm (+34 % at 103k, +47 % at 180k) | block 15, r190 |
| A12 / A13 (campaign sessions) | the vectorised 2-D copy fallback (block 10, r189) and the fused indexer scorer accepting quantized keys (block 11, r190) | delivery fixes, not campaign code |
| r197 siblings | a host expert table's routing-readiness under `-nkvo` (`TODO` #129, closed) and `-nkvo` declining the window | blocks 09 / 15, **r197** |

## What moved out of this campaign

Everything still open is in **[`wip/kv-residency-cleanup/`](../../../wip/kv-residency-cleanup/README.md)**:
the residency/pool gate (`README.md` §8 **P0**), O1's go/no-go (`§8 P1`, `§11.5`), a window for a *separate*
draft model (`O3-DRAFTER-RING.md` §7.3), a pool-size policy above ~200k, and the record hygiene those
imply.  `TODO` #124 (Goal A across `-sm tensor`) stays a TODO row with its design here
(`A6-TENSOR-SPLIT-WINDOW.md`); it is hardware-blocked on a one-card box.

## What was measured and declined (do not restart)

* **The pool's page policy (B-A step 3).**  `BA-POOL-DESIGN.md` §8.4: an LRU at this capacity is 90.2 % vs
  90.05 % measured — already at the plateau; the pool's own machinery is ~0.1 ms/token.  The lever is the
  indexer chain, which A13 closed.
* **`LLAMA_MTP_SPARSE_DECODE=1` under a pooled target**: −7.8 % with identical text and acceptance, stays
  off (`BA-POOL-DESIGN.md` §8.5).
* **The `-sm tensor` window/pool** without hardware: both placements decline the split with one message
  (r181 crash fix).  `A6-TENSOR-SPLIT-WINDOW.md` is the design and its four `common/fit.cpp` findings are
  owed by any implementation.
* **Streaming a full-attention (dense/MoE) model**: a non-goal (`README.md` §4, §10) — every cell is read
  every step, so Goal A is a *fittability* feature, not a speed one.
* **The non-KV device-resident budget** (expert-cache tables, F1 store, q8_1 arena, scheduler inputs, FA
  staging): deferred by the 2026-10-29 scope decision; the numbers stay in `INVENTORY.md` §3.
* **k-quant indexer keys**: an accepted limitation (a 256-element superblock does not divide `idx_dim` =
  128), recorded in `HANDOVER.md` §12.6 and `BA-POOL-DESIGN.md` §10.3; not worth re-deriving.

## Where to read first (in this directory)

`HANDOVER.md` (how A and B were built, the measurement rules, the exact gate commands in §6) →
`HANDOVER-A19.md` (the A17-era state and the two "decide whether to open them" candidates) →
`BA-POOL-DESIGN.md` §7-§10 (the pool) → `A2-MIXED-BACKING.md` §3.7 (the window, its purity hazard and the
arena half) → `O3-DRAFTER-RING.md` (the ring, §2.3 is the delivered session) → `C1`/`C2`/`C3` (the r196/r197
promotions).
