# wip/kv-residency-cleanup — the KV-residency campaign's leftovers (OPEN)

**What this is.**  The KV-residency campaign (`TODO` **#118**: the KV cache's VRAM footprint — windowed /
streamed residency, the fixed-region half, the cache's own bookkeeping) **finished its mechanism work and was
archived** on 2026-11-01 as
[`archive/work/kv-residency/`](../../archive/work/kv-residency/CLOSEOUT.md).  Goals A (the resident window),
B (the attendance pool) and C (the MTP draft window) all ship in **block 15**; the r197 release is
`v16-6050b467c-r197` (tip `ba7c2b7b8`, tree `eaf673210`), which also closed `TODO` #129.

**This campaign is only the residue**: one go/no-go, one measurement-heavy code item, one policy question, and
the record hygiene they imply (**the missing gate, P0, is BUILT — 2026-11-01, `scripts/gate-kv-residency.sh`**).
Nothing here is a new mechanism for a *shipped* feature, and **nothing here is in `patches/` or gets applied to
the fork** (`AGENTS.md`, WIP rules).

---

## 1. What you inherit (do not re-derive)

Block 15 is **three opt-in placements of the attention KV, one default-on draft window**, all sharing
"the KV does not have to be `O(-c)` of VRAM":

| knob | default | what it does |
|---|---|---|
| `LLAMA_KV_RESIDENT_CELLS` | off (`0`/unset) | the **window**: newest `W` cells in VRAM, the tail in pinned system RAM in one VA reservation (no kernel change).  `auto` = `--fit` sizes it, and only for a `-c` the user pinned (the decision `TODO` #122 records: P1, fittability only).  `-nkvo` declines it (r197); a `-sm tensor` split carries it, one region per child device (r204, `TODO` #124); a MIRRORED dsv4/HRM cache declines it |
| `--kv-attendance-pool [on\|off]` (alias `--kv-region-pool`; was `LLAMA_KV_REGION_POOL` before r246) | **off** | the **pool**: the authoritative KV in a pinned host mirror + a compact device pool of indexer-block pages, resolved in-graph by `GGML_OP_KV_POOL`.  qwen4exp/QSA only (it is the sparse reader's page table); declines while a window is active, on `!offload` and on a `-sm tensor` split.  **Since r247 the size is derived from the model** (one decode band of the indexer working set), not a knob |
| `--spec-draft-kv-window` (was `LLAMA_SPEC_DRAFT_WINDOW`, r246) | `auto` | the **MTP draft window**: the embedded MTP head's draft KV is a window instead of `-c` cells.  Since r246 `auto` = *the cap or nothing* (65536 cells, or a logged decline), `0` = off, `N` = N cells.  MTP + embedded head only |
| `LLAMA_KV_SPILL_RAM_PCT` / `LLAMA_KV_POOL_MIRROR_UNIT_MIB` | 85 % / 1 MiB | the pinned-RAM ceilings and the mirror's grow unit |

Two facts that make the rest of this file readable: a streamed cache reports `is_streaming()` and therefore
takes the **sparse** QSA decode arm (that is a *placement* decision, keyed on `kv_region`, not on the plan);
and **every** streamed arm is byte-identical to its plain arm by contract — a residency defect is a wrong
answer, not a crash, so the oracle is text *plus* prefill logits, never throughput.

Shipped description: `patches/README.md` (block 15), `ENVIRONMENT.md` §17.  History and the full design:
[`archive/work/kv-residency/`](../../archive/work/kv-residency/CLOSEOUT.md) — start at its `HANDOVER.md`
(§5 measurement rules, §6 the exact gate commands) and `BA-POOL-DESIGN.md` §7-§10.

**Out of scope here** (delivery items, not campaign work): `TODO` #128 (the gfx1201 decode-module MTP
cost), #120 (`q4_0`/`q5_0`/`iq4_nl`/`bf16` KV write folds), #127 (`cpy_2d_vec16` is F32-only), #116 (FP8).
`TODO` #124 (Goal A across `-sm tensor`) **shipped in r204** — the window runs on a tensor split (one region per
child device), so its design note
[`archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`](../../archive/work/kv-residency/A6-TENSOR-SPLIT-WINDOW.md)
is history and its arms are `scripts/gate-kv-window-split.sh` (plus the `LLAMA_KV_RESIDENT_CELLS` run of
`scripts/gate-state-pipe.sh`); a MIRRORED dsv4/HRM cache still declines the window, deliberately.

---

## 2. The work, in the order I would take it

### 2.1 The residency gate — **DONE (2026-11-01)**, `scripts/gate-kv-residency.sh`

**Why it existed.**  The plan said the streamed KV "cannot ship without" a residency gate, and it shipped on a
*substitute*: `LLAMA_KV_POOL_VERIFY=1` (verify kernel, 0 mismatches) plus a hand-run acceptance table at
5.3k / 103k / 180k.  There was **no `scripts/gate-*.sh` for residency or for the pool**, and the release
flow's park / ckpt / state gates do not measure hit rates — so a pool regression between releases was
invisible unless someone re-ran that table by hand.  This was the only shipped block-15 feature with no
release gate of its own.

**What shipped.**  `scripts/gate-kv-residency.sh` (indexed in `scripts/README.md`, `MANIFESTS.md` release gate
17, `AGENTS.md`'s change-to-gate table), two arms per placement, text as the oracle and the residency numbers
as the proof that the arm left the resident placement: window `wref`/`wres` (`W=8192`, above the run depth)/
`wspill` (`W=2048`) on the 4B at `-c 16384`; pool `pref`/`pres` (`=32768`, the whole cache)/`pfloor` (`=512`)
on Flash-Next + MTP at `-c 32768`.  It also prints a per-cache **KV accounting** block from the log, which is
the "cells / bytes / resident share / bookkeeping / fit consequence from the log rather than a guess" half of
the plan's P0.

**Three things the plan's own arm design got wrong — do not re-derive.**
1. **A pool cannot be made "deliberately undersized" by cells.**  `--kv-region-pool N` is floored to one
decode band (`8 x ceil(top_k / page)` pages, 4 104 pages on Flash-Next), so the plan's `512`-cell arm becomes
exactly that floor: 204 MiB of pool, not 128 pages.  The forced-miss arm has to be expressed against the
**working set** instead — the same pool at a depth that does not fit it (25 763 tokens = 6 441 blocks against
4 104 pages).
2. **A resident pool does not report ~0 misses.**  The claim cursor is round-robin (it hands out the next slot
whether or not the block is resident) and a decode step's top-k selection moves, so every step pulls in blocks
it did not read before: 144 claims/resolve at `-c 32768` with the pool holding the WHOLE cache.  The
separation is claim pressure and hit rate (144 -> 218 claims/resolve, 90.95 -> 89.64 % hits), not an absolute.
3. **`hits %` can never read 100 %**: an id whose block another thread of the SAME launch just claimed is
neither a hit nor a miss.
4. **The counters move run to run**: the claim cursor is an `atomicAdd`, so the same two arms read 144.3/218.4
claims per resolve in one run and 144.2/216.5 in the next (hits 90.95/89.64 vs 90.91/90.03).  The gate asserts
a strict inequality between the arms, never an absolute threshold.

**Negative test (the plan's "shown to fail").**  An off-by-one pool slot in `kv_pool_map_fill_kernel` (a
wrong-page resolve) makes `pres`/`pfloor` diverge in text AND in acceptance, and the gate FAILs on all three.
Note `LLAMA_KV_POOL_VERIFY=1` stays `bad 0` under it: VERIFY compares the pool's BYTES against the mirror, so
it cannot see a wrong READER index — the text is the only oracle for that class, which is the whole reason the
gate exists.

**Measured (this box, one R9700, r205/r206 build).**  Window: `wref == wres == wspill` (`be904e86f248`), 0
flips / 0 host MiB on `wres`, 16 flips / 64 MiB host-backed on `wspill`.  Pool: `pref == pres == pfloor`
(`f260d1824ab3`), draft 71/43 on all three, `bad 0`, 8 192 pages (408.0 MiB) on `pres` vs 4 104 (204.4 MiB) on
`pfloor`, 252 resolves on both, 144.3 -> 218.4 claims/resolve.  Runtime: ~15 s window, ~4m45 s pool.
**Not wired into `scripts/validate-gpu.sh`** (the worker has one gfx1151 card and no Flash-Next GGUF); it is a
by-hand release gate, like the park/ckpt/state gates.

### 2.2 O1 "KV-grow" — decide it, then either ship or close (`README.md` §8 **P1**, §11(5))

**What it is.**  The grow-only VMM region used as the KV buffer *without* a window: reserve the whole `-c` as
VA, commit physical lazily as the used extent grows, never unmap (the growth is monotone, so it is legal
between steps, not in-graph).  It saves nothing at a full context and helps exactly the config that
over-reserves `-c` — which is most server configs.

**State.**  The provider exists (`llama_kv_region_buft`) and is used **only** by `--fit`'s measurement probe
(`g_kv_region_probe`).  A real run passes it by, for a recorded reason: it changes *when* the KV's physical
is committed, so on a `-ncmoe` shape the slab takes the difference, and a run that later needs the whole
context can then refuse where the plain arm completes (O1's KNOWN LIMIT — the comment on the buffer choice
in `src/llama-kv-cache.cpp`).

**The decision** (the maintainer's): (a) offer it opt-in and gate it — `SLAB_INIT` layout identical when
the used context equals `-c`; a *small* used context must leave the arena **larger** than today; `--fit`
must see the term; and the refusal arm (grow past the reserve) must be reported, not an abort; or (b) close
it in the plan as "not offered: the refusal mode is worse than the OOM it avoids".

A cheap experiment before deciding: on a `-c`-generous `-ncmoe` server shape, compare arena MiB / residency
% / `identity fast path ON` between plain and grow-only at the same real depth.  Size: S to decide, M to
ship and gate.

### 2.3 A window for a *separate* draft model (`O3-DRAFTER-RING.md` §7(3); the archive's `HANDOVER-A19.md` §2)

**State.**  `--spec-draft-kv-window` is **MTP-only and embedded-head-only** by the r193 decision: an
embedded head has no GGUF of its own, which is what lets the slab reserve be derived with the window.  A
*separate* draft GGUF keeps the full `-c` cache: Flash-Next's MTP sidecar
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`), DFlash2 (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`) and the old
standalone `mtp-*.gguf`.

**Why it is not a one-liner.**  Such a draft context is not the embedded head's shape: it alternates a KV
*injection* graph (target features, up to `-ub` rows, main scheduler) with a noise-block decoder graph
(≤ `n_seq × block` rows, the **alt** scheduler chosen by batch kind — see the draft-reserve rule in
`INVARIANTS.md`).  A window must be sized for the injection's width and must not disturb the alt-scheduler
split, or graph reuse collapses.

**Gates.**  Target greedy text **byte-identical** with the window off and on (the target verifies, so a
draft change can only cost proposals); MTP Protocol A (`-n 3000`, reasoning pinned, both split modes,
device count quoted); acceptance **and** t/s against the unwindowed arm; the draft cache's log line must
scale with `W`, not `-c`; and `gate-ckpt-reuse.sh` / `gate-state-pipe.sh` if the draft's state path is
touched.  Size: M (measurable on one card; both models are on disk).

### 2.4 A pool-size policy above ~200k (the archive's `HANDOVER-A19.md` §2)

**State.**  `--kv-region-pool` is a **fixed cell count per layer**.  A17 measured `=16416`
(4 104 pages, 204 MiB) at 97.9 % hit at 5 246 tokens and 86.0 % at 102 800, and `=32768` bought 0.2 points
and no speed.  The pool's VRAM is inside the KV buffer's size, while its mirror is pinned **host** RAM that
`--fit` charges to the host — so a bigger pool trades device VRAM for hit rate only.

**Question.**  At the runme's `-c 262144` (a 200k+ working set) is a fixed count still right, or should it
scale with `-c` / the indexer's top-k?  Nobody has asked; do not change the default without the arm.

**Cross-arm, 2026-11-05 (`wip/bench-56gb/`, 1 x R9700, host experts on a mirror pool so the host tiers are
not a variable, `-c 204800`, `-ub 6144`, `-lv 4`).**  New data point, and it does NOT sit on the A17 curve:

* the measured `--kv-region-pool 16384` (4 104 pages, 204.4 MiB of device pool, **12** full-attention
  layers on this model) reports `hits 8408 (4.27 %)` of 196 896 ids at a 179.5k prompt, with
  `misses 29041 / claims 29041`, i.e. every claim missed and only 4.3 % of the ids were served from a
  resident page.  A17's 86.0 % was at 102 800.  Half the context, a 20x difference in hit rate: reconcile
  this BEFORE changing any default (the candidates are the prompt cache, `--parallel`, `-ub`, and the
  indexer's `top_k=2048` against a fixed 16 416 cells; the arms above did not control for them).
* measured cost of the pool at that context, everything else identical: **+2.3 GiB host anon** (the pinned
  mirror, up to 2 550 MiB at full context), **+2.3 GiB of arena VRAM** (the KV is in VRAM when the pool is
  off, so the expert arena shrinks by the same amount), decode neutral within noise, and prefill **0 to -17 %**:
  0 % at `-ub 6144` with `-cache-ram 1024`, -7 % at `-ub 2048` with the same, and -17 % when the pool runs
  beside a 6 GiB prompt cache (`-cache-ram 6144`), which is page-cache competition, not the resolve path.  A
  65536-cell pool is worse
  than 16384 on both bands.
* so on this box the pool is currently a RAM-for-VRAM trade that pays a PCIe cost for a 4 % hit rate.  That
  does not make it wrong (it is opt-in, and the contract is that it is inert when unset); it makes the sizing
  question the whole question.

**The work, in the order I would take it** (each step is cheap and can end the campaign):

1. **Reconcile the two hit rates** on one rig, one build, one config, at 32k / 64k / 102 800 / 204800: same
   `--parallel`, same `-ub`, same `--cache-ram`, page cache dropped before each start.  *Accept:* a curve, not a
   point, and a stated reason for the 86 % versus 4.3 % gap.  If the gap is a config artefact in A17's favour,
   the default is probably fine; if it is real, the pool is a large-context tool that stops working exactly
   where it is needed and that must be said in `ENVIRONMENT.md`.
2. **`auto` sizing from the working set**, not from a constant: `top_k (2048) x full-attention layers x the
   decode band`, plus the sink/recent tail, clamped by the VRAM budget (`kv_pool_bytes()` already reports the
   VRAM/RAM pair).  Keep the explicit cell count for A/B.  *Accept:* at every context, `auto` is within 3 % of
   the best hand-picked value on both bands, and no row is worse than pool-off.
3. **Report the property, not the counters.**  Predicted coverage of the attended working set at the target
   context (startup, visible without `-lv 4`) plus the measured hit rate at shutdown from
   `LLAMA_KV_POOL_STATS`.  "4.27 % hits" is only meaningful beside "of a working set 24x the pool".
4. **Decline when it cannot pay.**  Below a coverage threshold (start at 60 %), do not build the pool and say
   why: a low-coverage pool is pure cost (+2.3 GiB host, +2.3 GiB of arena, per-token PCIe refills).
5. **Prefill bypass**, decided by A/B: the prefill loss appears at `-ub 2048` with `-cache-ram 6144` and
   disappears at `-ub 6144` with `-cache-ram 1024`, so it may be page-cache competition rather than the resolve
   path.  Measure before changing anything.
6. **Document the decision table by context length** (16k-64k plain VRAM KV; 128k-200k a working-set-sized pool
   if it fits; above that the window or accept the streamed cache) wherever the knob is documented.

**Gates.**  The A11/A14 three-depth shape (interleaved arms, `-n 512`, MTP-4), hit rate from
`LLAMA_KV_POOL_STATS`, `LLAMA_KV_POOL_VERIFY=1`, text identity, and §4 below as written.  Size: S-M (needs a
200k+ arm; the prefill is minutes, not hours).

**Records for the cross-arm:** [`bench-56gb/RESULTS.md`](../bench-56gb/RESULTS.md) (rows A-E, K) and
[`bench-56gb/README.md`](../bench-56gb/README.md) (the rig; it is the only way to reproduce a 200k context on a
small box without a small box).

### 2.5 Close-outs — decide and record, no code

These are decisions a reader currently has to make for themselves, because the archive still states them as
open:

* **O6 / a portable (non-RDNA) provider** (`README.md` §8 P5): close it — the delivery is RDNA-first by
  policy and a non-RDNA provider has no consumer.
* **"The target context"** (`B1-STRATA-PLAN.md` §7(2)): record the answer — the single-GPU `-c 262144`
  Flash-Next runme shape is this campaign's target; the multi-device `-ncmoe` server shape is now supported by
  the split window (r204, `TODO` #124) and measured there.
* **The per-cache window exemption** (`B1-STRATA-PLAN.md` §7(3), a wip-only `LLAMA_KV_RESIDENT_CELLS_IDX`
  knob): close it — never built, and moot, because the pool keeps the indexer resident by construction.
* **k-quant indexer keys** (accepted limitation: a 256-element superblock does not divide `idx_dim` = 128;
  `HANDOVER.md` §12.6, `BA-POOL-DESIGN.md` §10.3): add it to `TODO.md`'s **accepted limitations** list, so a
  reader of `TODO.md` alone can find it.

---

## 3. Rules that cost sessions (carried forward, still true)

1. **`HIP_VISIBLE_DEVICES=0` always** — an unpinned `-sm tensor` enumerates the iGPU and aborts at model load
   with `invalid kernel file`, which reads like a defect in whatever you were testing.
2. **One GPU job at a time**, and **interleave arms** in one batch: this box's clocks drift.
3. **Arena numbers:** `-n 2000` and `-ncmoe` at the model's **full** layer count (35B-A3B and Flash-Next 48,
   122B-A10B 48).  Never quote a 128-token arena.
4. **Speculative arms:** `-n 512`+ and check `draft_n` / `draft_n_accepted` before reading t/s; static MTP
   depth ≤ 4 for reference arms; always quote the device count (the `INVARIANTS.md` reference arm is 3 devices;
   this box is 1).
5. **`ROCBLAS_USE_HIPBLASLT=0`** for every fresh-process numeric comparison (the library picks its solution
   per process start), and never judge run-to-run determinism on a greedy text hash.
6. **Text from runs WITHOUT `-v`** (`scripts/extract-generated.py`); timings from a separate `-v` run.
7. **Flash-Next arms always carry** `-ot per_layer_token_embd.*=CPU -lm none -lzm off`, and at
   `-c 262144 -ncmoe 48` also `GGML_CUDA_SLAB_HEADROOM_MIB=1536` (768 MiB is one hipBLASLt workspace short
   and the failure is a hard, order-dependent prefill OOM).
8. **`GGML_QSA_DERIVED_VIS=0` is not a no-op A/B** (it also changes the prefill scorer's strip trimming and
   can flip the `LLAMA_QSA_SCORE_WMMA` auto choice).  Use `LLAMA_INDEXER_NOBLOCK=1` for a top-k isolation
   arm, `GGML_CUDA_QSA_INDEXER_SCORE=0` for the fused-vs-chain indexer arm.
9. **A pooled cache's park declines by design** (the server falls back to save-and-clear); the park gate
   grades that as a PASS.  Do not "fix" it by growing the mirror to the park bound.
10. **`LLAMA_KV_POOL_STATS` takes a path**, never stderr, and `LLAMA_KV_POOL_VERIFY=1` must read `bad 0`.
11. **A spill arm must prove it spills**: `-c` too small proves nothing (the 4B needs `-c 16384` to reach a
    flip at `W=2048`); look for `FLIP > 0` / `KVREGION_FREE … host=`.
12. **The provider's own selftest** is the only sensitive primitive check: `GGML_CUDA_VMM_REGION_SELFTEST=1`
    must **PASS**, and must **FAIL** with `LLAMA_KV_REGION_QUARANTINE=0` — otherwise you have run a
    formality.
13. Always `--single-turn --no-display-prompt`, always `timeout`, and read the journal with
    `scripts/check-gpu-faults.sh` rather than grepping a log for `page fault`.

## 4. Gates owed by anything that changes block 15

* **Window change:** plain vs windowed same-seed text on an arm that provably spills, **prefill-logit**
  `0.000000 / 100.000 %` on that arm (`scripts/gate-prefill-logits.sh`), the selftest pair above, and the
  state/checkpoint/park gates if the memory type is touched.
* **Pool change:** text identity against the pool-off arm, `LLAMA_KV_POOL_VERIFY=1`, `LLAMA_KV_POOL_STATS`,
  `scripts/gate-kv-residency.sh` (r206: both placements, each with a resident and a forced-miss arm, plus the KV
  accounting block), `scripts/gate-park-idle-slots.sh`, `scripts/gate-ckpt-reuse.sh`, `scripts/gate-state-pipe.sh`
  on a **pooled** cache, `test-backend-ops -o FLASH_ATTN_QSA`, `check-gpu-faults.sh`.
* **Delivery change** (anything that reaches `patches/`): `scripts/validate-set.sh`, the per-block build gate
  `--changed-from <previous tip>` (and an `all` build at the tip), and the standard prefill-logit point at the
  27B Q8_0 before a tag.  `scripts/gate-prefill-logits.sh` bases are **per ROCm line** — a cross-line compare
  is refused, not a failure.
* **Promotion:** `wip/` → `patches/` needs the `AGENTS.md` promotion rule (validated win, an env kill switch,
  the *combination* re-validated, a block with the maintainer's go-ahead).  Nothing here is pre-approved.

## 5. Records

| where | what |
|---|---|
| [`archive/work/kv-residency/CLOSEOUT.md`](../../archive/work/kv-residency/CLOSEOUT.md) | what shipped, what was declined, where to read first |
| [`archive/work/kv-residency/`](../../archive/work/kv-residency/) | the whole campaign: `HANDOVER.md` (§5 rules, §6 gate commands), `HANDOVER-A19.md` §2 (the two candidates this file formalises), `BA-POOL-DESIGN.md` §7-§10, `A2-MIXED-BACKING.md` §3.7, `O3-DRAFTER-RING.md`, `A6-TENSOR-SPLIT-WINDOW.md` (`TODO` #124), `C1`/`C2`/`C3` (the r196/r197 promotions) |
| `patches/README.md` block 15, `ENVIRONMENT.md` §17 | the shipped description and every knob |
| `WORKLOG.md` 261031.008 / .009 / .014, 261101.001 / .002 | the Goal B / Goal C / r196 / r197 release records |
| `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md` | the two gates any change here must run |
