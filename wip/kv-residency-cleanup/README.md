# wip/kv-residency-cleanup — the KV-residency campaign's leftovers (OPEN)

**What this is.**  The KV-residency campaign (`TODO` **#118**: the KV cache's VRAM footprint — windowed /
streamed residency, the fixed-region half, the cache's own bookkeeping) **finished its mechanism work and was
archived** on 2026-11-01 as
[`archive/work/kv-residency/`](../../archive/work/kv-residency/CLOSEOUT.md).  Goals A (the resident window),
B (the attendance pool) and C (the MTP draft window) all ship in **block 15**; the r197 release is
`v16-6050b467c-r197` (tip `ba7c2b7b8`, tree `eaf673210`), which also closed `TODO` #129.

**This campaign is only the residue**: one missing gate, one go/no-go, one measurement-heavy code item, one
policy question, and the record hygiene they imply.  Nothing here is a new mechanism for a *shipped* feature,
and **nothing here is in `patches/` or gets applied to the fork** (`AGENTS.md`, WIP rules).

---

## 1. What you inherit (do not re-derive)

Block 15 is **three opt-in placements of the attention KV, one default-on draft window**, all sharing
"the KV does not have to be `O(-c)` of VRAM":

| knob | default | what it does |
|---|---|---|
| `LLAMA_KV_RESIDENT_CELLS` | off (`0`/unset) | the **window**: newest `W` cells in VRAM, the tail in pinned system RAM in one VA reservation (no kernel change).  `auto` = `--fit` sizes it.  `-sm tensor` declines it; `-nkvo` declines it (r197); one device only |
| `LLAMA_KV_REGION_POOL` | off (`0`/unset) | the **pool**: the authoritative KV in a pinned host mirror + a compact device pool of indexer-block pages, resolved in-graph by `GGML_OP_KV_POOL`.  qwen4exp/QSA only (it is the sparse reader's page table); declines while a window is active, on `!offload` and on a `-sm tensor` split |
| `LLAMA_SPEC_DRAFT_WINDOW` | **on** (`auto`) | the **MTP draft window**: the embedded MTP head's draft KV is a quarter of the context clamped to [32768, 65536] cells, `0` = off.  MTP + embedded head only |
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

### 2.1 The residency gate — build it (`README.md` §8 **P0** in the archive)

**Why.**  The plan said the streamed KV "cannot ship without" a residency gate, and it shipped on a
*substitute*: `LLAMA_KV_POOL_VERIFY=1` (verify kernel, 0 mismatches) plus a hand-run acceptance table at
5.3k / 103k / 180k.  There is **no `scripts/gate-*.sh` for residency or for the pool**, and the release
flow's park / ckpt / state gates do not measure hit rates — so a pool regression between releases is
invisible unless someone re-runs that table by hand.  This is the only shipped block-15 feature with no
release gate of its own.

**What to build.**  `scripts/gate-kv-residency.sh`, two arms per placement, both with counters:
1. **resident** — sized so the placement holds everything the run touches (window: `W` at or above the run's
   depth, so nothing flips; pool: `LLAMA_KV_REGION_POOL` in **cells** — 4 cells = one page = one indexer
   block — large enough that nothing is claimed out twice), and
2. **forced misses** — deliberately undersized (window: `W` well below the depth; pool: a small cell count,
   e.g. `512` cells = 128 pages, against a 103k-token run).

The counters are part of the assertion, not a diagnostic: the resident arm must report ~0 misses and the
undersized arm must report **misses > 0 with hits > 0** — otherwise it never actually left the cache and the
comparison proved nothing.

Assert: same-seed greedy text **byte-identical** between the two arms (text from runs **without** `-v`,
`scripts/extract-generated.py`), identical `draft_n`/`draft_n_accepted` on an MTP arm (`-n 512`+), `bad 0`
from `LLAMA_KV_POOL_VERIFY=1`, the hit/miss line from `LLAMA_KV_POOL_STATS=<path>` (a **path**, never
stderr), and the window arm's spill proof (`KVREGION_FREE … host=`, `FLIP > 0`) — **a gate that cannot
leave the resident arm is not a gate**, so fail if the two arms report the same hit rate.

**Model.**  Flash-Next GSQ-IQ3_XXS (`qwen4exp` — the only pool-capable model), 1 GPU, `-ncmoe 48`, q8_0 KV,
the PLE flags below.  A window arm can use a smaller model (the 4B at `-c 16384` really spills; at
`-c 4096` it never does).

**Done when** the script is in `scripts/README.md`, wired into the release flow (and optionally
`scripts/validate-gpu.sh`), and has been shown to *fail* on a deliberately broken arm (e.g. a pool whose
resolve is made to read the wrong page).  Size: S-M.

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

**State.**  `LLAMA_SPEC_DRAFT_WINDOW` is **MTP-only and embedded-head-only** by the r193 decision: an
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

**State.**  `LLAMA_KV_REGION_POOL` is a **fixed cell count per layer**.  A17 measured `=16416`
(4 104 pages, 204 MiB) at 97.9 % hit at 5 246 tokens and 86.0 % at 102 800, and `=32768` bought 0.2 points
and no speed.  The pool's VRAM is inside the KV buffer's size, while its mirror is pinned **host** RAM that
`--fit` charges to the host — so a bigger pool trades device VRAM for hit rate only.

**Question.**  At the runme's `-c 262144` (a 200k+ working set) is a fixed count still right, or should it
scale with `-c` / the indexer's top-k?  Nobody has asked; do not change the default without the arm.

**Gates.**  The A11/A14 three-depth shape (interleaved arms, `-n 512`, MTP-4), hit rate from
`LLAMA_KV_POOL_STATS`, `LLAMA_KV_POOL_VERIFY=1`, text identity.  Size: S-M (needs a 200k+ arm; the prefill
is minutes, not hours).

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
* **`TODO` #122** (the Goal A `auto` policy): decided (P1) and its owed `patches/README.md` line is present,
  so the row can be retired to one `WORKLOG.md` entry whenever the maintainer wants.

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
  `scripts/gate-park-idle-slots.sh`, `scripts/gate-ckpt-reuse.sh`, `scripts/gate-state-pipe.sh` on a
  **pooled** cache, `test-backend-ops -o FLASH_ATTN_QSA`, `check-gpu-faults.sh`.
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
