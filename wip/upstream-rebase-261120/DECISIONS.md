# Critical decision points — 2026-11-20 re-base

Every entry here is a place where **upstream now implements, or is refactoring, something the delivery
owns**.  The policy is `AGENTS.md` → *Report findings before shipping*: the delivery's code is never dropped
because upstream's "looked newer".  A rung that reaches an `OPEN` entry **stops and asks**.

Two classes, because they need different treatment:

- **A — subsumption.**  Upstream landed a feature the delivery also ships.  Dropping ours is a code deletion
  and needs the maintainer's word, even when the answer is obvious.
- **B — structural collision.**  Upstream refactored the surface our feature sits on.  Resolving it in-rung
  is the ladder's job; it becomes a decision only if the refactor leaves our feature no home (or gives it a
  better one).

Entry numbers are **campaign-local** (`D1`..`D14`); they do not continue the r278 campaign's numbering
(which lives in `archive/work/upstream-rebase-261113/DECISIONS.md` and must not be confused with this file).

Status: `OPEN` = waiting on the maintainer · `DECIDED` = answered, with the date and the choice ·
`MERGE` = resolved inside the rung, recorded in `RUNGS.md` (not a question).

---

## D1 — `#32505` duplicate expert IDs subsumes block 00's carried upstream-PR port  ·  **DECIDED 2026-11-20: take upstream's, retire the candidate**

**Claim.** Block 00 carries a *port* of the delivery's own upstream-PR candidate
(`upstream/UPSTREAM-PR-mul-mat-id-dup.patch`) so the CUDA `MUL_MAT_ID` handles a repeated expert id.  Upstream
`b94959a64` (*CUDA: implement support for duplicate expert IDs (#32505)*, 2026-11-19) lands the same feature
in the backend, removes the `[TAG_MUL_MAT_ID_DUPLICATE_ID]` caveat from `ggml.h`, and reworks `mmid.cu`
(+209) and `mmf.cuh` (−245/+…: the `mmf.cuh` few-row machinery is restructured).

**Evidence.**

| item | value |
|---|---|
| our port | block 00 commit message: "the upstream MUL_MAT_ID duplicate-ids fix … `mmids_helper` keeps every occurrence … `test-backend-ops -o MUL_MAT_ID` 931/931" |
| upstream | `b94959a64`, 9 files, +251/−271; also deletes the `ggml.h` caveat block (6 lines) |
| conflict it causes | block 00: `mmid.cu`, `mmf.cuh`; blocks 06/08: `mmid.cu` (`mmf.cuh` at 08) — **5 of the 61 conflicted paths** |
| our `upstream/` record | `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` + its README |

**Options.**

1. **Take upstream's `mmid.cu`/`mmf.cuh`/`ggml.h`, drop the block-00 port** (recommended).  Upstream's is the
   canonical implementation, it is what every later block will be rebased against, and it removes ~5 conflict
   paths from the climb.  The `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` candidate is then **retired** (it
   has been merged upstream; the record moves to `archive/work/`).
2. Keep ours and re-port on top of upstream's rework.  Cost: a rewrite of `mmid.cu` against a restructured
   `mmf.cuh`, for no gain — the features are the same one.
3. Keep both behind a switch.  Rejected on sight: two duplicate-id implementations is a correctness hazard,
   not a preference.

**Needs the maintainer because:** option 1 deletes delivery code and retires an `upstream/` candidate.

**DECISION (maintainer, 2026-11-20): option 1 — take upstream's `mmid.cu`/`mmf.cuh`/`ggml.h`, drop the
block-00 port, retire the `upstream/` candidate.**  Consequences for R00:

- `git checkout 919794500 -- ggml/src/ggml-cuda/mmid.cu ggml/src/ggml-cuda/mmf.cuh` is the starting point for
  those two paths, then the delivery's `mm_ids_helper_512_10` is re-ported onto upstream's new helper
  signature (`nbytes_shared` / `ne_store`).  **That fast path is not upstream's and must survive**: 1024-thread
  single block for the 512-experts / 10-used shape, RDNA3_5 + RDNA4, +4.5 % pp16384 / +5.2 % pp2048 on
  gfx1201, kill switch `GGML_CUDA_DISABLE_MMID_512`.
- `ggml.h`'s `[TAG_MUL_MAT_ID_DUPLICATE_ID]` caveat is gone on both sides, so that hunk becomes upstream's.
- `upstream/mul-mat-id-dup/` is **retired** (moved to `archive/work/upstream-mul-mat-id-dup-merged/`, with the
  reason in its README and the index row removed from `upstream/README.md`).  Recorded in `WORKLOG.md`
  261120.006.
- Everything else block 00 touches in `ggml-cuda.cu` (the band-uniformity / KV-split width work, the Meta
  wrapper, the allocator fixes) is untouched by this decision and lands as planned.

**Verification for option 1 (R00 exit):** `test-backend-ops -o MUL_MAT_ID` on gfx1201, plus the block-00
`quantize_mmq_q8_1` item the port also fixed.

---

## D2 — `#30542` HIP VMM pool: carry block 00's revert a second time  ·  **DECIDED 2026-11-20: carry it (with a tracked re-test)**

**Claim.** The r278 campaign's **BASE BLOCKER B1** is still live at `919794500`: the pristine base, built the
way the delivery builds (`GGML_HIP_NO_VMM=OFF`, ROCm 10.1.0, gfx1201), never returns from its first GPU
compute.  The standing r278 decision was "carry the revert in block 00; drop it at the next re-base".

**Evidence, measured today on the pristine base (`build-base`).**

| arm | result |
|---|---|
| `HIP_VISIBLE_DEVICES=0 llama-cli -m Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -n 20` | **hang** — SIGKILLed at 150 s (rc 137), no `load time` line |
| upstream code | `ggml-cuda.cu:709-712` still has the `#30542` reserve-before-`cuMemCreate` + halve-on-`OUT_OF_MEMORY` retry loop, byte-for-byte the trigger the r278 campaign pinned |
| (`GGML_HIP_NO_VMM` is a **CMake** option, not an env var — a runtime `GGML_HIP_NO_VMM=ON` changes nothing; a `build-base-novmm` control build was made to prove the mechanism) |

**Options.**

1. **Carry the revert again, folded into block 00** (recommended — it is the r278 standing decision, the code
   upstream ships is unchanged, and it is the only configuration that restores the VMM path the delivery's
   host-expert tier needs).
2. HIP-gate the revert.  Same cost, no validated benefit, and it forks the VMM pool per backend.
3. Open an upstream issue and build the whole climb with `GGML_HIP_NO_VMM=ON`.  Blocks 09/15 (host-expert
   arena, GTT store) cannot be verified that way, so the climb would carry unverified rungs.

**Recorded at r278:** `archive/work/upstream-rebase-261113/BASE-BLOCKER.md` — nothing in it has expired.

**DECISION (maintainer, 2026-11-20): carry the revert again — "for now".**  The re-test that would let it go
is tracked as **`TODO` #351**: *does upstream now properly support `-sm tensor` for `flint`?*  That is the
`0300` (`-sm relaxed-tensor`) drop condition, and the maintainer named it as the reason the escape hatch stays
alive; `#351` records the exact arms (3-GPU `-sm tensor` load, the refusal arm, and `-sm layer` vs
`-sm relaxed-tensor` text identity on `/llm/models/RecursiveAI/Flint/Q8_0/flint-Q8_0.gguf`).  It is run at
R17, with the revert in place, so the result cannot be an artifact of the revert's absence.  Note the two
mechanisms are *separate*: the revert restores the VMM pool the host-expert arena needs, and `0300` is the
Flint/Mellum/Mistral4 split mode — `#351` says so explicitly, for whoever reads it next.

---

## D3 — `#32387` BF16 ROPE/SET_ROWS and the bf16-cache rope fusion vs block 04  ·  **RESOLVED at R04: union of arms, upstream's guard dropped, measured**

**Claim.** Upstream `7fce5439f` adds BF16 to `ROPE` and `SET_ROWS` on CPU and CUDA **and** "`cuda : fuse BF16
rope into a BF16 cache`" — the same fused `ROPE + VIEW + SET_ROWS`-into-`BF16` path block 04 added, and it
lands on the same assertion (`src0 F32 -> dst F16 **or BF16**`) in `rope.cu`.

**Evidence.**

| item | value |
|---|---|
| ours (block 04, `rope.cu`) | changes the fused-path assertion to `src0->type == F32 && (dst == F16 \|\| dst == BF16)` and adds the `rope_multi<forward,float,bf16>` dispatch arm |
| upstream | `7fce5439f`, 13 files, "cuda : fuse BF16 rope into a BF16 cache"; touches `rope.cu` (+63), `set-rows.cu` (+53), `ggml-cuda.cu` (+13), `ggml.c`, `ops.cpp`, tests |
| conflict it causes | block 04 `rope.cu`, `ggml-cuda.cu`; block 08 `set-rows.cu`; block 10 `rope.cu` — 4 paths |
| the rest of block 04 | untouched by upstream: `fattn-tile.cu(h)` (+623/−191 `f16`-tile bf16-native work), `fattn-mma-f16.cuh` (+30), `common.cuh` (+53) |

**Options.**

1. **Take upstream's `rope.cu` bf16 arm + assertion; keep block 04's `fattn-tile`/`fattn-mma` bf16 attention**
   (recommended if the arms are equivalent — to be checked at R04 by diffing the two dispatches and confirming
   the bf16 KV cache + native bf16 FA still measures its kill-switch delta).
2. Keep ours and drop upstream's — wrong: upstream's also serves CUDA-on-NVIDIA and other backends, and the
   delivery's scope policy says a non-AMD path must not be injured.
3. Keep both arms if they are not the same code path (possible: ours may fuse *during* the tile kernel,
   upstream's in `rope.cu`).

**Ask before R04** because option 1 may delete our `rope.cu` change (the small part of block 04).

---

## D4 — `#32454` CUDA NORM + MUL (+ ADD) fusion vs block 08's `norm-gated.cu`  ·  **RESOLVED at R08: MERGE — additive, no precedence change, nothing dropped**

**Claim.** Upstream now ships a NORM+MUL(+ADD) fusion in `norm.cu` + a `ggml-cuda.cu` dispatch decision.
Block 08 ships `norm-gated.cu` — a narrow-row (ncols ≤ 256) wave-per-row RMS norm with an optional sigmoid
gate and a bit-exact reduction order — plus the delivery's fusion dispatcher.  These are two mechanisms for
"do not launch a separate MUL after a norm".

**Evidence.**

| item | value |
|---|---|
| ours | `ggml/src/ggml-cuda/norm-gated.cu` (block 08), guarded by `GGML_CUDA_DISABLE_NORM_ROWS`, gate `scripts/gate-norm-rows-alias.sh`; 8 rows/256-thread block, `rms_norm_f32<256>`-identical reduction |
| upstream | `03ec5b432` — `ggml-cuda.cu` +30, `norm.cu` +72/−26, `norm.cuh` +5; a fused epilogue on the *general* kernel |
| conflict it causes | block 08 `norm.cu`, `ggml-cuda.cu` |
| why they are not the same | ours is a *launch-shape* win for the ~786 k narrow rows (block-scheduling bound); upstream's is a *fusion* win (one fewer kernel).  Both can hold, but only if the dispatcher picks exactly one. |

**These are probably complementary** — the loud question is which one takes the fused case, and whether the
delivery's fusion dispatcher (block 10) still owns the decision.

**Options.**

1. Keep both, with a documented precedence (ours for narrow rows, upstream's otherwise) and a gate arm proving
   each still fires (recommended, pending the R08 diff).
2. Drop `norm-gated.cu` if upstream's fusion + the narrow-row launch equals its measured win at R08.
3. Drop upstream's fusion on the narrow-row path only.

Measured at R08 before any deletion, with `GGML_CUDA_DISABLE_NORM_ROWS=0/1` and the prefill-logit gate.

**RESOLUTION (R08, 2026-11-21) — option 1, and it was already the shipped arrangement.**  The question turned
out to be answerable from the r327 tree rather than by choosing:

- The delivery's matcher `ggml_cuda_norm_rows_try_fuse()` is called **~1000 lines ahead** of every generic
  `ggml_cuda_can_fuse(RMS_NORM, MUL[, ADD])` branch in `ggml-cuda-try-fuse.cu` (line 2199 vs 3314/3319), so
  where it matches it wins and upstream's branch is the fall-through — exactly option 1, shipped.
- The claims are disjoint: both delivery matchers bail on `rms->op != GGML_OP_RMS_NORM` and require narrow rows
  (ncols ≤ 256, ≥ 4096 rows, F32, single-use); `#32454` adds the **layer-norm** `GGML_OP_NORM` variant (never
  claimed by the delivery) plus a `mul`/`add` epilogue on the general kernel.  Nothing was deleted.
- At R08 the delivery's matcher is not reachable at all (its dispatch lives in block 10's TU), which is
  measurable: `GGML_CUDA_DISABLE_NORM_ROWS=1` changes neither the text (`1c5d32ac537d` either way) nor the rate
  (7409.28 / 7388.04 pp2048, inside noise) on the R08 tree, while the fusion surface as a whole is worth
  +5.4 % pp2048 / +18.7 % tg128 (`GGML_CUDA_DISABLE_FUSION=1`, identical text).

**Obligation (R10):** block 10 must place the matcher **ahead** of upstream's generic norm branches in
`ggml-cuda-try-fuse.cu` — that position *is* this resolution — and `scripts/gate-norm-rows-alias.sh` at R18 must
show both the narrow-row kernel and upstream's fused `NORM`+`MUL` still firing.

---

## D5 — DFlash speculative-decoding refactor (`#32016`, `#31904`) vs blocks 02/12  ·  **RESOLVED at R02: MERGE, no delivery code dropped**

**Claim.** Upstream restructured the DFlash-based speculative decoding methods (`32bdc3a89`) and added
DFLASH2 (`de32b268c`).  That is the surface blocks 02 (adaptive MTP draft depth) and 12 (DFlash) sit on, and
block 13's draft-vocabulary subset touches the same `common/speculative.cpp` plumbing.

**Evidence.** `32bdc3a89` touches `common/common.h`, `common/speculative.cpp`, `docs/speculative.md`, both
READMEs, `src/models/dflash.cpp` — the five block-02 conflicts plus block 12's `speculative.cpp` and
`dflash.cpp`.  `652d559ed` also rewrites DFlash's node-budget rule in `dflash.cpp` (see D8).

**Options / what to measure.** The refactor is *structural*, so the first question is mechanical: does the
delivery's adaptive-depth control still have a seam?  Then: does upstream's structure change the four-axis
MTP protocol's baseline (`benchmarks/mtp-adaptive-methodology.md`)?  Stop at R02 and present the diff.

---

## D6 — `llama_batch_ext` migration (`#32356`, `#32306`)  ·  **MERGE (high cost)**

The old batch API is deprecated and `llama_batch_ext` introduced.  Blocks 02, 12, 16 (and every gate arm that
builds a `llama_batch`) must migrate.  Not a subsumption — the ladder does it — but it is the single biggest
*mechanical* cost in this re-base and the reason R12/R16 should not be scheduled beside a benchmark.
Note it also affects the **gate scripts** (`llama-batched-bench`, the CLI arms), not only the sources.

---

## D7 — server `slot_sched` refactor (`#32154`) vs block 16 / block 12  ·  **MERGE (structural)**

`429460dc0` adds `server_slot_sched` and a prompt running hash and rewrites `server-context.cpp` /
`server-task.{cpp,h}`.  Block 16 (max-min fair admission, per-step chunk, prefill/decode duty cycle) and
block 12 (idle-slot parking, server prompt cache) both live there.  Both blocks apply **clean** in the probe,
which means the probe is *not* seeing this one — read the refactor at R12 and R16 and check that:
the admission policy still runs per step, the parking path still has its seam, and `gate-prefill-fairness.sh`
/ `gate-park-idle-slots.sh` still reach their code.

---

## D8 — graph node budget into the model classes (`#32648`) + MoE-cache node reserve (`#32636`)  ·  **MERGE**

Upstream moved the nodes-per-tensor factor into `llama_model::graph_nodes_per_tensor()` (defaults 8; 32/64
overrides in `models.h`; DFlash keeps its hparams rule in `dflash.cpp`) and added
`res += 32*n_layer_all` when `cparams.moe_cache_size > 0`.  The delivery (block 10) owns the budget's other
half: the KV-rotation leaf accounting and the hard "over its node budget" refusal.  The merge is additive —
keep upstream's factor function, keep our rotation leaves and refusal, and re-run `scripts/gate-node-budget.sh`.
**Note:** upstream's MoE-cache reserve is for *their* `--moe-cache`, not for block 09's arena; check whether
block 09 needs its own term.

---

## D9 — indexer key rotation into the model classes (`#32653`)  ·  **MERGE**

`4783b1145` replaces the architecture list in `llama-kv-cache.cpp` with `llama_model::lid_k_hdmd()` (and fixes
DeepSeek4.1).  Blocks 09 (KV cache) and 11 (qwen4exp/indexer) both touch that decision.  Merge: keep the
virtual, make qwen4exp's answer explicit in its model class rather than in the cache's list.

---

## D10 — `--tensor-split` up to `llama_max_devices()` (`#31435`) vs block 0300  ·  **MERGE**

`7804be063` relaxes the argument limit for `--tensor-split`/`--fit-target`.  Block 0300 (`-sm relaxed-tensor`)
is adjacent but independent.  Check that the relaxed-tensor path's own device-count assumption still holds, and
that `test-arg-parser` (block 02 touches it) still agrees.

---

## D11 — upstream HIP/GCN MMQ + MMVQ tuning vs blocks 06/07/08  ·  **MERGE, measure — the maintainer wants both A/B'd and the better of each picked**

Upstream retuned the AMD MMQ configs in a **new file** `mmq-config-gcn.cuh` (`b79696f1f` +128/−83,
`2e53a0bab` +456/−237 incl. stream-K on GCN) and the MMVQ `ne11` gate for RDNA3_5 (`c7518dbec`).  The
delivery's blocks 06/07/08 tune `mmq.cuh`, `mmq.cu`, `mmvq.cu` (wide-VDR MoE, Q6_K MMQ, verify-band WMMA) and
carry their own per-arch defaults.  **Risk: double tuning** — a delivery default that silently overrides
upstream's new table, or vice versa.

**Maintainer's instruction (2026-11-20): assess both, run the performance tests, and pick the better of each
change — do not assume either side.**  The measurement plan, run at R06/R07/R08 on the *same* build (one tree,
kill switches instead of two builds wherever the switch already exists, which is most of them):

| knob | arm A | arm B | measured by |
|---|---|---|---|
| MMQ config table (per type/J/cc) | upstream's `mmq-config-gcn.cuh` | the delivery's table | `llama-bench -p 512,2048,16384` + the 35B-A3B prefill shape, 3 reps, one device count |
| MMVQ `ne11` gate (RDNA3_5) | upstream's | ours | `llama-bench -n 128` decode + the MoE decode shape |
| Q6_K MMQ (block 07) | on (ours) | off (`GGML_CUDA_*` switch, else a two-build arm) | `llama-bench -p 512,2048` on a Q6_K-quantised model |
| wide-VDR MoE (`vec_dot_*_moe`, block 07) | on | off on RDNA3_5 (already the default there) | the 1-GPU host-expert decode, `-n 128` |
| MMB (block 06) | on | `GGML_CUDA_DISABLE_MMB`-equivalent | `llama-bench -p 2048,16384` |
| few-row MMQ for `MUL_MAT_ID` (`#32065`) | upstream's dispatch | ours/MMB | the 35B-A3B prefill + `test-backend-ops -o MUL_MAT_ID` |

Rules that decide the winner, in order: (1) the text is byte-identical on both arms; (2) prefill logits pass
`gate-prefill-logits.sh` for a kernel change; (3) the faster arm wins **per knob**, not per file — the
maintainer's instruction is "the best one of each change", so a mixed result (upstream's table, our VDR) is the
expected shape and is a perfectly good outcome.  Never average two tables into a third without re-measuring.
Record the numbers in `benchmarks/` and the choice in the rung log.

---

## D12 — staging buffer from the CPU buffer type (`#31438`) vs block 10  ·  **MERGE**

`3e41ae69d` allocates `llama_context`'s tensor staging buffer from the CPU buffer type.  Block 10's async
input ring is a staging path with its own lifetime rules.  Check whether it now allocates twice or whether
upstream's change is the better home for it.

---

## D13 — qwen4exp upstream work (`#32483`, `#32476`) vs block 11  ·  **MERGE, read**

Upstream fixed the qwen4exp graph reallocation from the n-gram table scale and FP8 checkpoint conversion.
Both are in `src/models/qwen4exp.cpp` — a file block 11 owns.  Take upstream's fixes; confirm the delivery's
QSA/indexer/lazy-reader additions still sit on top, and re-run `gate-qwen4exp-quant-coherence.sh` at R11.

---

## D14 — UI moved `tools/ui/` -> `ui/`  ·  **DECIDED AND APPLIED 2026-11-20: the HF-asset hack is retired; no build fetches a UI bundle**

The old workaround was `scripts/ui-dist-cache.sh`: one validated bundle per machine in
`~/.cache/rdna-boosts/ui-dist`, mirrored into `${checkout}/tools/ui/dist` so `ui-assets.cmake`'s Priority 1
never let a build reach the HuggingFace bucket (`TODO` #342: a per-build-dir cache re-fetched per rung, and r326
rung 14 died on a timeout; `latest` also shipped a bundle with no `workbox*.js` on 2026-11-17, which the
validator rejects).

**The hack is gone and nothing replaces it**, because the delivery no longer builds a UI where it does not ship
one.  Asked at the start of the re-base (the maintainer's stated reason for doing this re-base now), and the
answer is the same on both bases — `LLAMA_USE_PREBUILT_UI` exists at `4f9dfa0e3` too:

- `-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF` in `gate-per-block-build.sh`, `scripts/scratch-tree.sh`
  and **`~/bin/build-llama-rocm`** (`UI=1` opts back in; `BUILD_UI=1` adds the npm build).
- `ui-assets.cmake` then fetches nothing and runs no npm: `-- UI: embedded 0 assets`, `libllama-ui.a` 1.8 KB
  (an embedded-UI build of the same tree is 3.1 MB from a 13.5 MB generated `ui.cpp`).
- `scripts/ui-dist-cache.sh` **deleted**; `HF_UI_VERSION` is no longer a workaround the gate needs.
- `TODO` #342 **closed** (`WORKLOG.md` 261120.006); docs corrected in place (`INVARIANTS.md` r303 fact 6,
  `MANIFESTS.md` gate 5, `AGENTS.md`, `scripts/README.md`).
- **The container build still needs a real UI** and gets it from its own `web` stage (npm), which satisfies
  Priority 1 without touching the bucket or the validator.  Its paths still target the pre-r328 `tools/ui/`
  layout: moving them to `ui/apps/chat` (+ `ui/lib/ui`, `ui/lib/voice`) is **`TODO` #352**, done at R18
  because a Dockerfile edit is only correct once the tree has the new layout.

---

## D15 — where the VMM-pool top-k workaround lives  ·  **DECIDED 2026-11-20: block 00, as a rung repair**

`upstream's top_k_argsort_cuda` does a strided `cudaMemcpy2DAsync` out of its sorted-index temporary, and
this stack refuses a padded D2D 2-D copy (`spitch != dpitch`) whose destination is VMM-backed.  The delivery
requires the VMM pool (host-expert arena, slab, GTT store), so the refusal is reachable on the shipped
configuration and is **not** caused by any delivery source change: a build of the pristine base carrying
**only** the `#30542` revert aborts identically (`RUNGS.md` R00.6 has the four-arm table).

**The question** (raised by the maintainer): block 00 is where the symptom starts, block 10 is where the
delivery already fixed it — which block owns the fix?

**Decision: block 00**, i.e. the first block that needs it, because block 00's own revert is what enables the
pool.  Implemented as the rung repair `3ea8b0212`, which adopts block 10's `top_k_gather_rows` /
`top_k_gather_cuda` verbatim.  Consequences, all recorded in `RUNGS.md` R00.6:

- block 10 **drops its copy of the hunk** when R10 is climbed (keep-local on that file), so the final tree
  carries the copy once;
- the repair is **squashed into block 00 at R18**;
- the `mm_ids_helper_512_10` A/B is unblocked and reads **+0.7 %** prefill at 5298 tokens, **not** the
  recorded +4.5 %/+5.2 % — text identical, so the re-port is correct, and the claim of record is re-measured
  at R18 before anyone quotes it.

**Alternatives rejected.** Leaving it red to R10 (loses the field-model oracle for R02-R09, and with it the
only measurement that reaches `mm_ids_helper_512_10`); a `#define`'d minimal fix in block 00 that block 10
then deletes (a switch that exists only to be removed, plus a delete-hunk, and the 1-D loop is more code than
the kernel); and moving the **revert** to block 09 so R00-R08 never enable the pool (that would verify eight
rungs on an allocator the release does not use — the r278 "not a rung" trap — and this very bug is proof that
the allocator changes op behaviour, which is exactly the class a rung exists to catch).

**Why the delivery owns this rather than upstream:** upstream's HIP default is `GGML_HIP_NO_VMM=ON`, so no
upstream HIP user has the VMM pool and none of them can hit it.  The refusal itself may be worth an upstream
or ROCm report as a runtime question, but the delivery's answer is the device-side gather, which it already
had.

**Resolution (R02, 2026-11-20).**  `#32016` refactored by **adding sibling classes** (`DSpark`/`DFlash2`/
`LiliCorr` derive from `impl_draft_dflash`) and left `common_speculative_impl_draft_mtp` as its own class —
which is exactly where every block-02 hunk lands.  All five conflicts were unions (enum, name map,
`type_to_str`, the `COUNT` static_assert 13 -> 14, `add_config_if_enabled`), plus a 420-line region where the
two sides inserted at the same point and both belong.  The arm is not merely built but **measured**:
`--spec-type none` 29.3 t/s vs `draft-mtp-adaptive` 65.2 t/s (+122 %) with **identical text**
(`55824e640aa0`), acceptance 0.95238, per-position `(1.000, 1.000, 0.857)`.  The one thing to keep in step is
`need_n_rs_seq()`, which upstream made an explicit list: a new upstream type that needs an rs_seq must be
added there as well as to the enum.  Block 12's DFlash hunks re-open this at R12 and will be checked against
the same finding.

**Resolution (R04, 2026-11-20).**  `rope.cu` was a pure union — upstream added `BF16->BF16`/`BF16->F16`, block
04 adds `F16->BF16` (its `F32->BF16` merged cleanly), so all {F32,F16,BF16} -> {F32,F16,BF16} pairs have a
kernel.  `ggml_cuda_should_fuse_rope_set_rows()` was the real collision: upstream refuses every non-BF16 rope
into a BF16 cache, which is exactly the delivery's bf16-KV path, and it refuses it only because *its* matrix
lacks that arm.  With the unioned matrix the guard is what would disable the feature, so **the guard is
dropped** and the cache-type list stays as the test.  Measured with `GGML_CUDA_DISABLE_FUSION=0/1` at
`-ctk bf16 -ctv bf16`: **6786.5 / 88.8 t/s against 6417.0 / 76.2** (+5.8 % prefill, +16.5 % decode), text
identical (`71 chars sha=1de99b7e6ee6`); `test-backend-ops -o ROPE` 515/515; f16-KV coherence unchanged at
`1c5d32ac537d`.  The alternative — keeping upstream's guard — is recorded in `RUNGS.md` R04.2 with the cost it
would carry, so the choice is auditable rather than silent.  In R04.2's terms this was option 1 *plus* the
union of option 3: no delivery arm was deleted.

---

## D16 — two incompatible FA iteration models: `#32407` (rotation) vs block 05 (band)  ·  **DECIDED 2026-11-20 (maintainer): option 2 — the delivery's model wins, the rotation is not carried**

`flash_attn_ext_f16_process_tile` is the tile loop of the MMA FA kernel.  Upstream `054a1babf` (`#32407`,
*Rotate K/V read order for better L2 cache hit rates*) and the delivery's block 05 **both rewrite it**, and the
two rewrites are different partitions of the same work:

- upstream: `nkb = nkb_pre + nkb_post` over a contiguous run plus a rotated remainder, stepping `kb0 + 1` and
  wrapping at `kb0_max`; the loop is the `nstages > 1` double-buffered pipeline;
- the delivery: a strided band, `for (; kb0 < kb0_stop; kb0 += kb0_step)`, with a `kq_blocks` bitmap skip, a
  neutral partial for a band block that owns no iteration, and the RDNA4 fetch-ahead ring.

`#32407` is not a prefetch hint — it changes which blocks a tile visits and in what order, and its bookkeeping
must account for every block exactly once, so a merge is a **third model**, not a union.  A wrong model is
invisible to short-prompt text and silently wrong at depth.

**Options** (full detail and the 22-region classification: `RUNGS.md` R05.1/R05.2):

1. **Combined model** — highest fidelity, highest risk; the oracle is `test-backend-ops -o FLASH_ATTN_EXT`
   (7381 cases, ~25 min per iteration) + the prefill-logit KLD gate + text, at every step. 2-4 sessions.
2. **Keep block 05's model, drop the rotation** (recommended) — the other 17 regions still get the union, so
   `#32217`'s per-head-mask fix and the type changes survive; only the rotation is not carried.  Its own commit
   message records no number, and the band/stream-K split it would displace is a shipped, gated delivery
   feature.  Reversible later as a rung-repair or an `upstream/` candidate.
3. **Keep upstream's model, re-band later** — equivalent to redoing block 05.

**Why it is asked rather than decided:** option 2 drops an upstream performance change, which is the
maintainer's call under `AGENTS.md`, and option 1 is a multi-session design task that must not be started
without agreement on the oracle and the risk.

**Decision (maintainer, 2026-11-20).**  Option 2, with the standing rationale stated: *in every re-base so far
upstream has never done something better than the delivery*, so where the two are alternatives the delivery's
side is taken and the cost is recorded and measured rather than assumed absent.  Implemented at R05
(`1c9bfa37c`): the band iteration model is kept, upstream's rotation is not carried, and its absence is
verified rather than assumed (no `nkb_pre`/`nkb_post`/`kb0_max`/`kb0_rot` in the FA sources; every `kb0_next`
left is the delivery's own fetch-ahead).  The `#32217` per-head-mask fix and the `#31617` typed-mask conversion
were still adopted, because they are unrelated to the rotation — dropping them would have removed a
*correctness* fix for Inkling 2, which is a different thing from preferring our performance model.

**Evidence the choice is safe:** `test-backend-ops -o FLASH_ATTN_EXT` **7428/7428, 0 failures, 2/2 backends**;
4B coherence unchanged (`1c5d32ac537d`); and the block's own switch is non-vacuous where the feature lives
(q8_0 KV: `GGML_CUDA_FA_KV_NATIVE=1` 88.6 vs 83.9 t/s decode, +5.6 %, identical text).  If the rotation is ever
wanted back it must arrive as a *combined* model with that matrix green — i.e. as option 1, which remains
available and is what this decision declines for now.

**Addendum (2026-11-20, R05 repair `c3860af1e`):** the maintainer asked whether anything relies on `nb32` being
64-bit wide, and it does.  The per-head term adopted from upstream, `nb32*(zt_Q % ne32)`, is an `int`
multiplication when `nb32` is `int32_t`; the head index scales it by up to 71, so at 128k context with a
2048-token ubatch the product is 35.5 GiB and wraps (`nb32` alone, 512 MiB, fits — which is why the first
answer said "safe" and was wrong).  Overflow starts around 28.8 MiB of `nb32` (~15.1M cell-tokens).  `nb32` is
now `int64_t` in the typedef and in all four kernels, exactly as `#32217` does.  This is the general lesson for
the rest of the climb: **"take the delivery's side" applies to a shared *mechanism*, not to a type widening
that upstream introduced to make its own new code correct** — the per-head mask is upstream's feature and its
arithmetic is upstream's to get right.

## D17 — the DFlash feature set vs upstream's behaviour, after R12 merged the class onto `init(opt)`  ·  **DECIDED 2026-11-20 (measurement, per the maintainer's instruction): the delivery's feature set stays**

Maintainer, 2026-11-21: *"the best way to make a decision on which way to go with the dflash choice is to
measure which is more performant.  I know that we spent a lot of effort on optimizing DFlash2 within the
delivery repo.  I am reticent to drop its work for upstream's."*  So the delivery's work was the default and a
**loss** had to be shown before anything was dropped.  The question has three layers and only one is
measurable as a rate:

1. **The graph builders are not a fork.**  `build_dspark_markov_head` and `build_lilicorr_scores` are
   byte-identical between upstream's new per-campaign files and the delivery's `src/models/dflash.cpp`
   (7,037 B / 11,151 B), `build_dflash2_conv` differs only structurally, and the one real difference —
   `build_dflash2_selector`'s all-gather of the vocab-split logits — is a **correctness** fix: the pristine
   base aborts the same arm under `-sm tensor` (`ggml-backend-meta.cpp`'s `handle_per_row`:
   `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`), while the plain control on that base runs and
   the re-based tree runs every arm.  Upstream's file layout is kept; the all-gather is ported into it.
2. **The seam is structural**: every delivery feature is a switch or a state field, not an alternative
   implementation, so it is implemented on upstream's `dflash_options`/`init(opt)` base and the pre-split
   duplicate state (`is_dspark`/`is_dflash2`/`is_lilicorr`/`sample_from_anchor`/`selector_top_k`/
   `dflash2_rng`/`lilicorr_*`) is deleted from the base, where upstream's subclasses already own it.  Nothing
   of the delivery's feature set is lost by that deletion.
3. **The feature set was measured** (three targets, four arms, 6K and 26K prompts, text + acceptance + pp + tg;
   `benchmarks/2026-11-20-r328-r12-dflash-merge.md`).  Text and per-position acceptance are **identical in every
   pair**; the ingest tail is **+29.8 % pp at 26K on the 2-GPU 35B** and +11.7 % on 3-GPU Flash-Next, ~neutral
   only on a 1-GPU `-sm layer` cell (where there is nothing to overlap, as the delivery's own r130 record
   says); F1 is +6-7 % pp on the dense 27B and inert on both MoE heads; decode is flat everywhere.

**Decision: keep upstream's file layout and class split with the delivery's whole feature set on top** — ingest
tail default-on (`LLAMA_DFLASH_INGEST_TAIL=0` kills it), F1 default-on (`LLAMA_DFLASH_DEV=0`), the ported
all-gather, DFlash2's selector/probabilistic paths and DSpark's anchor layout as upstream now has them.  No
number anywhere argued for dropping a delivery feature; the only thing dropped is the build's second,
pre-split copy of the class.  Recorded at R12 (`RUNGS.md` §R12.4).

## D10 addendum (2026-11-20, R17) — verified at the completed ladder

Upstream `#31435` (`--tensor-split` up to `llama_max_devices()`) merged structurally with block `0300`; nothing in
the rung needed a choice.  The verification is the block's own two arms on this base: **flint Q8_0, 3 GPU,
`-sm relaxed-tensor` == `-sm layer` byte-identical** (`8ec5510e801e`, 85 chars) — the amax arch the mode exists
for — and **35B-A3B Q8_0, 3 GPU, `-sm tensor` == `-sm relaxed-tensor`** (`86b7f9b5aa80`) — the mode is inert for a
non-amax arch.  Both run through the split path `#31435` touches.

## `TODO` #351 addendum (2026-11-20, R17) — **`0300` cannot be dropped yet**

The re-test `0300` awaited: on `919794500` **flint still refuses `-sm tensor` at load** —
`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'flint'; this arch's MoE amax rescale needs
'-sm relaxed-tensor'` (upstream's `#31081` deny with the block's own pointer).  So upstream has **not** landed a
real per-device amax fix, the temporary block stays in the default apply for r328, and `-sm tensor` keeps
upstream's strict behaviour.  Re-test again at the next re-base, not before.
