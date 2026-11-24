> **THIS IS THE CAMPAIGN RECORD AS OF 2026-11-24 (STEPs 0-14), KEPT FOR REFERENCE.**  The *live* handover is
> [`HANDOVER.md`](HANDOVER.md), which is about **one task**: the real fix for the 2-D state-save read
> (`WORKLOG.md` 261124.004/005, `INVARIANTS.md` r381/r382/r383-wip).  Nothing in this file is a work order any
> more — read it when a fix needs the reasoning, the traps or the anchors behind a landed step
> (`RUNS.md` has the numbers, `WORKLOG.md` the dated entries).

# HANDOVER — `prefill-decode-arms` v2, campaign state + the next tasks (STEP 11, then STEP 12)

**Read this file first, then work top to bottom.**  Written for a session with *no* memory of the campaign: what
exists, what is green, what is open, and the exact ladder to run.  Everything below was measured on this box; a
plan that has not been run says so.  **STEPs 0-10 are DONE — your work is §3: the delivery sweep (STEP 11) and the
candidate patch + the ask (STEP 12).**  Read `§0.0 START HERE` first; it is the short version of §3.

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`prefill-decode-arms-v2`** (**pushed to origin**), HEAD = the commit that carries this file; **`main` untouched** at `91a81505` |
| release the candidate is cut against | `release.json`: base `919794500`, tip `089f90067`, tree `27db9bfd4ef7f41c7df60f704410de28863beac3`, `n_blocks` 17 + 1 temporary (`0300`) = `n_default` 18, `n_optional` 2 |
| work tree | `~/llama.cpp`, branch **`arms-v2`** (**local only, never pushed**), tip **`ffaee2c52`** (STEP 10b), tree clean |
| build | `~/bin/build-llama-rocm` → `build-rocm` (ROCm **10.2.0**); fast loop `cmake --build build-rocm --target llama-cli llama-server -j 16` |
| extra build dirs | **`build-ref`** = a build of the delivery tip `089f90067` (the "pre-change build": the reference for a regression question), **`build-bisect`** = the last bisect rung's build.  Both at the same source path, so ccache stays warm |
| landings | STEP 0-6e, **7a**, **7d**, **7c**, **7e** (STEP 4 broke the external draft head on a tensor split), **8** (the gate set), **7f** (a borrowing context carried an uninitialized `cparams.type_k`), **9** (the shape/latency/memory matrix), **10a/10/10b** (the speculator on the pair, six memory defects, the per-aux-context reserve) — §1.2 |
| **next action** | **STEP 11 and STEP 12 are DONE (2026-11-24)** — see the STATUS block below.  The delivery sweep is green (16/16 gates on `ffaee2c52`) and `beta/prefill-decode-arms/` has been REPLACED by this design's candidate (tree `84ffb1ef1c`).  **The only open item is the maintainer's promotion decision (block 17)**, plus the residuals in §7 |
| what is NOT done | nothing in `patches/` changes; **no block exists**; the candidate is a `beta/` artifact awaiting an external validation window (`BETA-TESTING.md`).  The campaign's own gates are green but are NOT delivery gates |

---

## STATUS (2026-11-24, the STEP 11/12 session) — `WORKLOG.md` 261124.003, `RUNS.md` 2026-11-24

**Both closing steps are complete; this HANDOVER is now a record, not a work order.**

* **STEP 11 — the delivery regression sweep: 16/16 gates green on `ffaee2c52`** (`validate-set`, `check-shlib-symbols`,
  `check-windows-portability`/`-paths`, `gate-prefill-logits` **KLD 0.000000**, `gate-state-pipe`, `gate-park-idle-slots`,
  `gate-kv-residency`, `gate-host-pool` 59/59, `gate-mtp-dev-rows`, `gate-spec-purity`, `gate-draft-ctx`,
  `gate-moe-promise`, `gate-node-budget`, `gate-graph-replay-width`, `gate-ckpt-reuse`), plus the whole-set `all` build
  and a clean fault scan.  `check-gen-docs` FAILs on the campaign tree by design (the candidate owes the
  regeneration) and is green on the applied candidate tree.
* **Two findings, both environmental** — the two segfaults were **stale tool binaries** (build the whole `all` profile
  BEFORE the ladder, not after), and `gate-kv-residency.sh`'s default `MODEL_POOL` (+ its `-Q8_0` sidecar) was
  **missing from the box**, not stale gate debt: the gate PASSed on `GSQ-RCO-IQ3_S` and then on its own defaults once
  the model tree was restored, and `TODO` #369 stays at three gates.  `INVARIANTS.md` r380-wip.
* **STEP 12 — the candidate**: `beta/prefill-decode-arms/` replaced whole (three files), ONE `format-patch` commit,
  sha256 `a2badad1…e366fe3c`, 42 files, +3073/−741, cut against the **release tree** `27db9bfd4` and applying to
  tree **`e24d88b31e`** — re-cut twice since (STEP 13's cancel fix, STEP 14's one-RCCL-per-process fix), each verified
  by a fresh clone + strict `git am` and by `apply-all.sh --include-beta`.  The 21 `steps/` replay reproduces the
  campaign tree `1ac30b82f` **exactly**.
* **Correction recorded:** §12.1's `git diff 919794500` is wrong — a `beta/` candidate is applied AFTER the blocks and
  must be release-tree-relative `format-patch` output.
* **STEP 13 (added 2026-11-24, from the maintainer's beta testing) — a crash the STEP 11 sweep could not see:**
  aborting one of two concurrent generations killed the server (`GGML_ASSERT(t_start > 0)`), because the CANCEL
  path released a slot the decode arm was mid-round on.  Fixed by `release_slot()` (quiesce the arm, then
  release), gated by the new `gates/gate-arms-cancel.sh` (fails pre-fix, passes after).  A second tester crash
  (`-sm tensor` on 3 devices: `FD_SETSIZE` exhausted by RCCL's IPC rings, doubled by the second context) is
  **diagnosed and recorded as a Known issue in the beta artifact, not yet fixed** — `GGML_CUDA_ALLREDUCE=ce` is
  the measured workaround (11 fds), and the maintainer's decision is a true fix.  `INVARIANTS.md` r381/r382-wip.
* **STEP 14 (added 2026-11-24, same beta testing)** — a `-sm tensor` split on 3 devices aborted with
  `*** bit out of range 0 - FD_SETSIZE on fd_set ***`: **not memory, descriptors**.  RCCL's IPC rings cost 646
  fds per comm layer, the delivery alone is at 63 % of the ceiling, and the pair's decode arm and/or an MTP draft
  head crossed it (2026 fds measured with both).  Fixed: an auxiliary context asks for the copy-engine chain and
  the backend enforces ONE RCCL comm layer per process — measured 686 fds for base / pair+drafter / pair+drafter,
  i.e. **0 extra**, text unchanged (`gate-arms-spec` 36/36, `gate-spec-purity` same hashes).  Gate:
  `gates/gate-arms-fd-budget.sh`.  **`GGML_CUDA_ALLREDUCE=ce` is no longer needed** as a workaround.
  **Behind it: an OPEN, independent defect** — the state/checkpoint save's 2-D read of a row-split `result_output`
  fails with a pitch of 1/3 the host pitch on a 3-device split (the Meta backend's chunked 2-D path); recorded in
  the beta artifact's README, not fixed.
* **NOT taken:** promotion into `patches/` as block 17.  That is the maintainer's decision, and a validated window
  comes first.

---

---

## 0.0 START HERE — the STEP 11/12 session, in order

**You have two jobs and they are ordered: prove the delivery did not move (STEP 11), then package the work as one
candidate patch and ASK (STEP 12).**  Nothing here needs a design decision; if you find yourself redesigning, you
are off the plan.  The full detail is in §3 — this is the checklist.

**Before anything: the state you must verify** (2 minutes)

```bash
cd ~/llama.cpp && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD   # arms-v2 ffaee2c52, tree clean
cd ~/llama-cpp-rdna-boosts && git rev-parse --abbrev-ref HEAD && git log --oneline -1  # prefill-decode-arms-v2
tail -5 WORKLOG.md | head -3            # 261124.002 = the newest entry
cmake --build ~/llama.cpp/build-rocm --target llama-server llama-cli -j 16   # must be a no-op
```
If `~/llama.cpp` is not at `ffaee2c52` with a clean tree, STOP and read §1-§3 before touching it — the whole
campaign's evidence is tied to that tree and its `build-rocm` (ROCm 10.2.0).

**STEP 11 — the sweep (commands: §3 STEP 11, 11.2; the touched-file map is 11.1)**

1. `scripts/validate-set.sh` — the DELIVERY set is unchanged, so this must be green as it was at r353.
2. The 16 delivery gates the touched files owe — 11.1 maps every touched file to the gates that can see it, and
   11.2 is the run order.  Run them **one GPU job at a time** (the campaign's harnesses take a flock; the
   delivery's gates mostly do not) and **record each verdict**, because a green campaign gate does not imply a
   green delivery gate.
3. `scripts/check-gen-docs.sh`, `check-windows-portability.sh`, `check-windows-paths.sh`,
   `scripts/check-shlib-symbols.sh`, one whole-set `all` build, `scripts/check-gpu-faults.sh` over the logs.
4. **A failure is a finding, not a work item you invent.**  Diff it against the delivery tip build
   (`build-ref`, built from `089f90067` at the same source path) with the same command **before** you change
   anything: if `build-ref` fails the same way, it is not this campaign's defect (record it, report it).  If it
   passes there and fails here, it IS this campaign's — fix it, re-run the gate, and add a `WORKLOG.md` entry.
5. **Report the sweep's result to the maintainer BEFORE STEP 12's commit** (§6, the reporting rule).

**STEP 12 — the candidate (commands: §3 STEP 12, 12.1; the directory contract is 12.2)**

6. Cut the patch from a **canonical** chain: `scripts/apply-all.sh` on a clean `~/llama.cpp` at `release.json.base`
   → apply the campaign's own steps → `git diff <base> > beta/prefill-decode-arms/prefill-decode-arms.patch`.
7. **Replace `beta/prefill-decode-arms/` entirely** (`README.md`, `BETA-TESTING.md`, the one `.patch`; delete the
   old design text — the maintainer's decision 2026-11-24).  It is ONE self-contained patch: on promotion it is
   **delivery block 17**, a modification to the lower-numbered blocks.
8. Report: the measured trade/memory/shapes, what is validated and what is not, the residuals (§7), and
   **ASK about promotion**.  A green ladder is not permission; the block is the maintainer's (§6).

**Do not:** push `~/llama.cpp` anywhere; touch `patches/`, `MANIFESTS.md` gates or `release.json`; renumber or
create a block; "fix" a delivery gate by changing the gate; or read the archived beta design (§0).

---

## 0. Reading rules (inherited from `README.md` §0 — they are part of the design)

**In scope:** this directory (`README.md`, `DESIGN.md`, `PLAN.md`, `HANDOVER.md`, `RUNS.md`), the code in
`~/llama.cpp`, and the delivery records that describe the *shipping* system (`INVARIANTS.md`,
`GREEDY-PURITY.md`, `MANIFESTS.md`, `ENVIRONMENT.md`, `scripts/README.md`, the two methodology files in
`benchmarks/`).

**DO NOT READ, for any reason, "to see what the old one did":** `archive/work/prefill-decode-arms/`,
`archive/work/prefill-decode-arms-followups/` (incl. its `DESIGN-REVIEW-2026-11-23.md`),
`archive/work/prefill-transfer-overlap/`, and the design / "hard-won rules" halves of
`beta/prefill-decode-arms/README.md`.  That material is **wrong at the root** (a second context that *views* and
*co-owns* the first's memory, recurrent state copied between two modules) and an agent that reads it rebuilds
those assumptions.  The one exception was used and closed at STEP 1 (three backend hunks of the beta patch).

If a step ever seems to need that material, you are misreading the step: re-read `DESIGN.md` §2 (the two memory
models side by side) and §5 (the defect register V1-V11 — every entry is a defect v2 deletes structurally).

---

## 1. Where the campaign stands

### 1.1 The design in one sentence

**ONE `llama_memory_i`, TWO `llama_context`s: the prefill arm OWNS the memory, the decode arm BORROWS it, and
nothing is copied between them.**  `llama_context_params::ctx_memory` is the only relationship the pair needs
(`DESIGN.md` I1-I9; I9 was added by this session).

### 1.2 Landed — one reviewable patch per landing in `steps/`, each with its oracle

| landing | patch | what it is |
|---|---|---|
| STEP 0 | — | baseline anchors (`baseline/r353-anchors.md`), harnesses, supervision contract |
| STEP 1 | `STEP-1-0001-backend-concurrency.patch` | per-(thread,device) host stream incl. `batch-copy.cu` + the async-input ring, best-effort VMM pool peer grants + a range registry, `ggml_cuda_pool_peer_copy_ok()` defined unconditionally |
| STEP 2 | `STEP-2-0002-ctx-memory-borrow.patch` | `llama_context_params::ctx_memory`, `memory_own`/`memory` split, I1/I2 |
| STEP 3 | `STEP-3-0003-one-allocator-one-lock.patch` | the reservation bitmap (I4) + one plain mutex on the memory (I3) |
| STEP 4 | `STEP-4-0004-meta-view-slot-identity.patch` | the Meta backend view slot is a per-**context** identity, `thread_local` selector |
| STEP 5 | `STEP-5-0005-recurrent-placement.patch` | `llama_memory_recurrent` contextified (per-batch `placement`), `rs_idx` reset out of the graph-time getter |
| STEP 6a/6c | `STEP-6a-0006-pair-predicate-and-context.patch` | `common_arms_pair_possible()` (pure) + `--prefill-decode-arms[-mode off\|on\|auto]` + the decode arm's context (a borrow, inert) |
| STEP 6d-1 | `STEP-6d-0007-slot-step-extraction.patch` | `slot_step()` extracted from `post_decode()` |
| STEP 6d-2/6e/6f | `STEP-6d2-0008-decode-arm-routing.patch` | the decode thread, `slot.arm` ownership, the handover, the step barrier, the five invariants |
| STEP 6e | `STEP-6e-0009-reserve-and-census.patch` | the decode arm's reserve, the reservation census on stderr, the (now removed) `-ncmoe` refusal |
| STEP 7a | `STEP-7a-0010-aux-narrow-regions.patch` | **the aux narrow region stack**: one pinned slab work region per extra context (slot 0 = draft, slot 1 = a borrowing context), `thread_local` routing from the allocating context's identity, the iface rename; the `-ncmoe` refusal **removed** (§2.1-2.5) |
| STEP 7d | `STEP-7d-0011-pair-drop-disable.patch` | `llama_context_disable_compute_buffer_drop()` for both arms: the prefill→decode drop races the other arm's graph capture (§2.7) |
| STEP 7c | `STEP-7c-0012-aux-reserve-calibration.patch` | the aux region is charged to the slab **reserve** (approximation), the slab logs the measured size + the reserve to set **once per slot**, and `mapped_arena` separates the arena's extent from the slab's (§2.8) |
| STEP 7e | `STEP-7e-0013-meta-slot-per-graph.patch` | **STEP 4's defect**: the `thread_local` Meta view slot must be bound where a graph is built/resolved, not where a flag changes — `sched_alt_select()` bound only after its no-op early return, so on `-sm tensor` a second context's graph resolved under the first's view containers (`GGML_ASSERT(tensor)`) (§3, and `INVARIANTS.md` r368-wip) |
| STEP 8 | — (gates + anchors, no code) | `gate-arms-default-inert.sh` + `gate-arms-recurrent-shared.sh`, `baseline/inert-anchors.tsv`, `baseline/inert-shapes/`, `mtp-axis.sh` shape flags + `--no-lock` (§3) |
| STEP 7f | `STEP-7f-0014-borrower-kv-types.patch` | **STEP 2's defect**: `cparams.type_k`/`type_v` (no default initializer) were assigned only in the own-memory branch, so a BORROWING context carried stack garbage that qwen4exp's QSA probe fed to `ggml_new_tensor_4d` -- Flash-Next + the pair aborted 2 of 3 runs (`GGML_ASSERT(type >= 0 ...)`).  A borrower now mirrors the OWNER's types |
| STEP 7g | `STEP-7g-0015-meta-split-state-per-slot.patch` | the Meta backend's split-state cache is a per-**buffer** member (`split_state_cache[8]` + one version pointer) on a buffer every context shares, so two arms computing at once raced it -- SEGV in `ggml_backend_meta_get_split_state`, 2 of 3 runs on a tensor-split pair arm.  Now indexed by the same `thread_local` view slot as the containers |
| STEP 7h | `STEP-7h-0016-capture-blas-barrier.patch` | **the tensor-split blocker, fixed as the maintainer chose (a barrier, neither feature disabled)**: a graph capture holds one process-wide mutex exclusively, library GEMMs hold it shared, and `cublas_handle()`/`cublaslt_handle()` **creation** holds it exclusively — creation (`hipblasCreate` -> hipBLASLt -> legacy-stream sync) is what actually tripped, not the GEMM (`INVARIANTS.md` r374-wip).  Flash-Next tensor-split pair: **23.31 ms/token vs 53.99** (−56.9 %), 7 of 7 clean |
| STEP 10a | `STEP-10a-0017-draft-round-extraction.patch` | `slot_accept_draft_round(slot, ctx, seq_rm_type)` extracted out of `post_decode` so BOTH arms run ONE verify/accept/emit round (the context is a parameter: the decode arm verifies on `ctx_dec`, which borrows the same memory).  Verified as a pure refactor against STEP 0 A5: axis K `a87c4318b649`, acceptance **74/73/435/423** |
| STEP 10b | `STEP-10b-0019-per-aux-context-reserve.patch` | the maintainer's decision: the arena reserve is a **per-aux-context COUNT** (`common_arms_reserve_derive()` = `n_aux x COMMON_ARMS_RESERVE_MIB_PER_AUX`, report in the log) instead of one lump for "the pair"; `n_aux` counts the contexts no other term covers (today the decode arm -- the drafter's region is its own reserve's `compute` term).  Reserve 4864 MiB on the measured shape, unchanged, with the count now explicit; text unchanged (`RUNS.md` 2026-11-24 §8) |
| STEP 10 | `STEP-10-0018-speculator-on-the-pair.patch` | the DECODE arm drives the whole draft/verify/accept round for the slots it owns; ONE speculator lock over both arms' paths (recursive: the round ends by printing the statistics); a slot's speculative state reset at BOTH handover directions; the predicate admits `draft-simple`/`draft-mtp`/`-adaptive`/ngram and refuses the layer-input families + a gemma4 assistant (asked of the GGUF arch INSIDE the predicate, so the reserve and the pair decision still cannot disagree).  Six memory defects fell out of it: the draft's target rows follow the arm that ran the batch (`common_batch::ctx`); `ctx_aux` gives a qwen35 MTP draft its own slab region; an allocation and a cross-stream event wait may not overlap the other arm's capture; an aux region maps its own hole, can GROW, and its floor call had the slab slot off by one.  Oracle: `gates/gate-arms-spec.sh` 30/30 on 3 shapes + 4 axes, `a87c4318b649` on the pair-off drafter path, default-inert PASS (`RUNS.md` 2026-11-24) |
| STEP 9 | — (`baseline/shape-matrix.sh` + `server-arm.sh --vram-probe`) | the shape/latency/memory matrix: 7/7 shapes text-identical (3 repeats per arm), the cost table (+53..+154 MiB, 1280 MiB aux region on a host tier), the ITL/prefill trade (ITL −22.8 % at `-ub 2048`, prefill **+24 %**), and two corrections (the ITL gate's text check, one-run-vs-rate) |

Code tips: 7a `e460f0dea`, 7d `55aaf5398`, 7c `16131c077`, 7e `3013e880a`, 7f `f61fcb03f`, 7g `ffbde9a41`,
7h `f4540afd0`, 10a `e7ae0b470`, 10 `f3e34859d`, 10b `ffaee2c52` (branch `arms-v2`, local only).

### 1.3 Green, with the number that proves it (re-measured across STEPs 7c-10b; see the tip per row)

| claim | evidence |
|---|---|
| **the speculator ON the pair (STEP 10)** | `gates/gate-arms-spec.sh` **30/30 PASS**: `pair+draft == pair+none == single+draft == single+none` on 3 dev f16 (`f2753e13e419`), 3 dev q8_0 (`f2753e13e419`), 2 dev f16 (`0f06ed90e017`), drafts accepted on the decode arm (34/25, 32/26, 32/26); the four axes on the pair R `3a41e71cfaab` / K `8fe00a125d38` / P `fb4519c618ef` / C `4d46d5ab509d` with 74-181 drafts accepted; the pair-off drafter path unchanged (`a87c4318b649`, 74/73/435/423) |
| **concurrency with a drafter** | `gate-arms-multislot.sh` + `--spec-type draft-mtp-adaptive` on the 35B-A3B host tier (np 4, 3 devices): PASS at `ROUNDS=20` 2/2 (`6 step(s), 2 handovers, widest batch 2`); `baseline/itl-arm.sh` pair+drafter **5/5 clean, ITL 1.26-1.28 ms/token** through a 5246-token prefill (70 ms/token solo) |
| default path inert (pair OFF) — re-verified on the final tip | A1 4B `be904e86f248`, A2 27B-3GPU-tensor `10a82b5559a9`, A3 35B-A3B-host `e9db5cb8db0b`, A4 prefill-logit **KLD 0.000000 / same-top-p 100.000 %**, A5 K axis `a87c4318b649` (151.4 t/s, acc 0.98649/0.97241, pos-1 0.986), A6 server `{7bd542418112, 6ce1f88dd5f7}` + the **13-line startup shape byte-identical** to `baseline/A6-startup-shape.txt` |
| the pair reads the single-context text | 4B `7bd542418112`/`6ce1f88dd5f7`; `gate-arms-n1-memory.sh` PASS (`01c9cbe2e895`/`5e69982c7d94` both arms, 0 of 8192 cells reserved); 27B-3GPU-tensor `f2753e13e419`/`c5f2fdc43642` |
| the decode arm really runs | census `the decode arm is generating: N step(s), M handover(s), widest batch W slot(s)` — e.g. 8/3/3 and 15/3/**4** in the soaks |
| the pair's win (latency) | `gate-arms-itl.sh`, 4B, 5246-token prefill on B while A decodes 64 tokens, `-ub 2048`: **16.33 → 15.09 ms/token** (1044.8 → 965.5 ms / 64 tokens), prefill **5191 → 6165 t/s**, text identical (`36216e77db27`).  **Flat at `-ub 512`** — the win needs the wide chunk |
| **the host-expert shape** | `gate-arms-multislot.sh`, 35B-A3B Q8_0 `-ncmoe 40`, np 4, 20 rounds: **PASS 7/7** (5 on the 7c retake, 2 after the `mapped_arena` fix), 0 errors, **0** `arena alloc failed`, aux region `narrow-3` pinned from the **unmapped VA spare**, journal clean; pair-OFF **control** on the same shape PASS |
| **the draft slot** | Gemma4 26B-A4B Q8_0 (MoE) `-ncmoe 20` + `-md mtp-gemma-4-26B-A4B-it.gguf`: **both** aux regions (`narrow-2` draft, `narrow-3` borrowing) map from the spare at `GGML_CUDA_SLAB_RESERVE_MIB=2560` (the slab's own recommendation), 0 evictions; `none` == `draft-mtp` == **481 chars `a195d3d6953c`**, stable over 3 cold `none` runs |
| GPU-resident soak | 4B, 4 slots × 20 rounds, 80 requests, 0 errors, **widest decode batch 4** |
| the probes | `probe-kv-reservation.cpp` (I4), `probe-recurrent-shared.cpp` (I5), `probe-ctx-memory.cpp`, `probe-meta-slots.cpp`, `probe-host-stream.cpp` |
| **default path untouched (the gate)** | `gates/gate-arms-default-inert.sh` **PASS** on 4 shapes: cli text `be904e86f248` / `db5ab8e63e11` / `10a82b5559a9` / `e9db5cb8db0b` (the first, third and fourth are **pre-change** STEP 0 anchors) and startup shapes identical (the 4B one to `baseline/A6-startup-shape.txt` **byte for byte**), plus the pair-ON sensitivity arm (4 added lines, same responses) |
| **one memory, two contexts on a RECURRENT model (the gate)** | `gates/gate-arms-recurrent-shared.sh` **PASS** on 35B-A3B and Flash-Next/qwen4exp + the shared MTP head, **both on a 3-device `-sm tensor -mg 0` split** (that shape is part of this campaign and is the one STEP 7e broke): placements differ (`head=0 n=1` vs `head=1 n=1`), concurrent digests == `--solo`, and the four-axis `none` == adaptive text with 73..213 drafts accepted on every axis.  `--axis-shape host` runs the delivery's host-expert arm for the external head instead |
| **the shape matrix (STEP 9)** | `baseline/shape-matrix.sh`: **7/7 shapes text-identical** with the pair OFF vs ON, 3 fresh processes per arm, one request in flight (4B, phi-4, 27B-3GPU-tensor, 35B-A3B host tier np 4, Flash-Next-3GPU-tensor np 3, Gemma4 iswa) + the designed REFUSAL row (Gemma4 + its shared MTP head).  Cost **+53..+154 MiB** of VRAM (+149/154 per device on the tensor splits) and one 1280 MiB aux region on a host tier |
| **the ITL/prefill trade (STEP 9)** | 4B, A decodes 64 tokens while B prefills 5246 tokens, **3 repeats per arm**: `-ub 512` 14.34 -> 13.36 ms/token (−6.8 %) and 4732 -> 5676 t/s; `-ub 2048` 20.38 -> 15.74 (−22.8 %) and 4909 -> 6090 (**+24.1 %**); `-ub 4096` 20.83 -> 15.85 (−23.9 %) and 5092 -> 6079 (+19.4 %).  On tensor splits (2-3 repeats per arm): **Flash-Next 3 dev `-sm tensor`** `-ub 4096` **53.83 -> 22.89 ms/token (−57.5 %)** with capture off (+21.8 % prefill), or 47.82 -> 41.20 (−13.8 %) with `ROCBLAS_USE_HIPBLASLT=0`; **27B 3 dev `-sm tensor`** `-ub 2048` 64.45 -> 60.30 (−6.4 %, +35.6 % prefill) |
| the 7f fix | 5/5 pair-ON runs clean on Flash-Next-3GPU-tensor (was 1/3), reading the pair-OFF answers; `-ctk q8_0 -ctv q8_0` identical on both arms |
| the 7e fix | the reproducer (Flash-Next + external head, 3 devices `-sm tensor`) reads the delivery's text `be69e274b66d` again; `gate-draft-ctx.sh` PASS; the pair on 27B tensor identical (`8f396b237f0f`/`04116b9a99ad`); the host-tier soak PASS |
| the build | whole set builds; `scripts/check-shlib-symbols.sh` **OK** (16 libs) after the iface rename |

### 1.4 Open

* **STEP 11 (next):** the delivery regression sweep — §0.0 steps 1-5, §3 STEP 11 (11.1 the touched-file map,
  11.2 the ladder, 11.3 a failure).  Always: the campaign landed changes in files the DELIVERY owns even with the
  pair off (`include/llama.h`/`llama-cparams.h`/`llama-context.{h,cpp}`, `ggml-cuda-slab.cu`, `ggml-cuda.cu`,
  `common/`), every one inert for a single context — the sweep is what proves it, and the three gates that
  actually cover those edits are `gate-draft-ctx.sh`, `gate-mtp-dev-rows.sh`, `gate-spec-purity.sh`.
* **Residuals a tester must be told about** (all in §7): `LLAMA_MTP_DEV_ROWS` inert on the pair; the aux pin grows
  only while nothing is live at its slot; the reserve count is one constant per context, re-measure if the verify
  batch is much wider than `4 x (7+1)` rows; the slab reserve to set comes from the calibration line.
* **Closed at STEP 10:** the speculator on the pair (§1.3 row 1, `RUNS.md` 2026-11-24).  Six memory defects came
  out of it, four fixed here and two recorded as follow-ups (§7.2/§7.8).
* **Closed at STEP 8:** the five-gate set (`gate-arms-{default-inert,n1-memory,itl,multislot,recurrent-shared}.sh`)
  plus STEP 10's `gate-arms-spec.sh`, each printing its own non-vacuity line.  `RUNS.md` has the numbers;
  `INVARIANTS.md` r368/r369-wip and r375..r378-wip the facts.
* **STEP 12:** the candidate patch + the promotion ask — §0.0 steps 6-8, §3 STEP 12.  The artifact REPLACES
  `beta/prefill-decode-arms/` whole and is ONE patch; promotion is block 17 and the maintainer's decision.

---

## 2. The host-expert chapter (closed, but read this before touching the slab, the pair or any third context)

TASK 1 ("the pair faults a host-expert shape") is **closed**, and it took **three** separate defects.  They are
independent; fixing one never fixed another.

### 2.1 The fault: a SLAB WORK-BASE ALIAS (STEP 7a)

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION ... kernel: rms_norm_mul_rope_f32_qk_pair<false,float,__half,0,0>
```

The previous session blamed "the prefill arm remaps the MoE expert tables".  **Wrong.**  With
`GGML_CUDA_WORK_ALLOC_DEBUG=1` (prints every compute-buffer alloc/release + base offset):

```
0.07.583 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the PREFILL arm's compute view
0.07.589 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the DECODE arm's view: SAME base 0
0.08.683 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the prefill arm re-takes base 0
0.08.686 GPU FAULT
```

`ggml_cuda_slab_work_alloc()` has ONE narrow (decode/verify) base and assumed **one live narrow view**.  One arm's
prefill wrote the other's activations; the K rope then wrote KV rows at a garbage index.  The delivery already
fixed this class once for the MTP draft (the `narrow2` region — its own comment: *"without this they both aliased
at base 0"*).  The pair is simply the **third** context.

Rung 1 of the ladder (`GGML_CUDA_ROPE_QK_PAIR=0 GGML_CUDA_QK_PAIR_FWHT=0`) **still faulted**, in the *unfused*
kernel — so the Q/K deferral was never the mechanism.  (Trap: `GGML_CUDA_QK_PAIR_FWHT=0` cannot disable the
faulting instantiation at all; `fwht_n=0` is the **plain** pair, switch `GGML_CUDA_ROPE_QK_PAIR`.)

### 2.2 The fix: one pinned region per extra context, routed by identity (I9)

* `narrow2` became a **stack of 2 pinned regions** at the top of the reserved VA — slot 0 = a draft context,
  slot 1 = a context that **borrows** another's memory; the arena ceiling is the **lowest** pinned offset; the
  commit takes the pinned offset as an argument (the pin drops the lock to evict, so the window is validated);
* routing is a **`thread_local` slot from the allocating context's own identity**
  (`llama_context::slab_aux_slot()`: `cparams.ctx_other` → 1, `memory_borrowed` → 2).  Same pattern as STEP 4's
  Meta view slot.  A **process-wide** flag was the second defect: the prefill arm's reserve inherited the decode
  arm's flag, was routed into the aux region, found it live, fell back to `cudaMalloc` and failed the graph
  reserve with a 1280 MiB OOM;
* iface: `slab_narrow2_floor` → `slab_aux_floor(dev, slot, bytes)`, `slab_compute_narrow2` →
  `slab_compute_aux(dev, slot)` across cuda/cpu/rpc/meta (`check-shlib-symbols.sh` is the check);
* `GGML_CUDA_SLAB_NARROW2=0` still bypasses the slab for such a context (A/B arm — **not** a working
  configuration on a slab-armed host-tier shape, it aborts; §2.8/`INVARIANTS.md` r365-wip);
* the `-ncmoe > 0` refusal is gone.

### 2.3 The intermittency: the drop races the other arm's capture (STEP 7d) — **the important one**

After 7a the soak passed *sometimes*.  Same binary: PASS 3/3, then FAIL 3/3, then FAIL 1/PASS 1.  Two causes:

1. **the gate was unpinned** — `MOE_EXPERT_HEATMAP` defaults to load+**save**, so each soak rewrote the persisted
   map and the next started from a different residency prior, and residency picks the MoE kernel set.  The gate
   now pins `MOE_EXPERT_HEATMAP=load`;
2. **the real defect** — the prefill→decode **compute-buffer drop** is a *process-wide* mutation (slab work
   boundary move + MoE arena re-arm).  Issued from the prefill arm while the decode arm is inside
   `llama_process()` capturing:

```
0.19.024 I compute buffer 1280.00 MiB from the slab work region (boundary 1536.00 MiB)
0.19.026 W moe_cache_rearm: re-armed 15 stood-down expert-cache tables after a compute-buffer drop
0.19.026 E ROCm error: operation failed due to a previous error during capture   (ggml_cuda_kernel_launch)
0.19.058 E operation would make the legacy stream depend on a capturing blocking stream  <- consequence
```

Proof: `LLAMA_DROP_COMPUTE_BUFFERS=0` → **3/3 clean**.  Fix: `llama_context_disable_compute_buffer_drop()`,
called for **both** arms when the pair is created (`server-context.cpp`).  Measured cost: **nil** (arena
25537 MiB and hit-rate 1.0 identical to the drop-on arm, because the slab reclaims the wide view on the next wide
pass).  **This is exposure removal, not the general answer** — the general answer is a barrier that keeps the
other arm out of `llama_process()` around a drop (DESIGN §8 invariants 3-4).  A text/hash oracle cannot see this
class: the run either aborts or is correct.

### 2.4 The reserve, and the number the slab tells you to use (STEP 7c)

The aux context's compute region is charged to the **slab reserve**, not to the arena's headroom.  The headroom
variant (allocate it "anonymous, before the slab") **aborts**: the outside cushion is what `slab_extend` maps the
arena's physical *from*, so spending it starves arena growth (measured: 42 `arena alloc failed` → capture abort;
`INVARIANTS.md` r365-wip).  The unreserved arena variant costs one eviction pass at load.

* `COMMON_ARMS_RESERVE_MIB = 1280` (an **approximation**; measured 1280 MiB on 35B-A3B `-ncmoe 40` np 4, 512 MiB
  per aux context on Gemma4 26B-A4B + MTP).  With `reserve = headroom + aux`, the region maps from the **unmapped
  VA spare** → 1 → **0** `arena alloc failed` lines;
* once per slot the slab logs the measured size and the reserve that would host every aux region
  (`headroom + sum(floors)`) — the operator sets that on the next run:

```
the aux narrow-3 region (1280 MiB) came from the unmapped VA spare -- no arena table was evicted.
The reserve in force is 2816 MiB; 2816 MiB would host every auxiliary region (headroom 1536 + aux 1280)

... had to TAKE arena tables (1024 MiB evicted) ... Set GGML_CUDA_SLAB_RESERVE_MIB=2560 ... (headroom 1536 + aux 1024)
```

* verified end to end on Gemma4 (two aux contexts of 512 MiB): default 2496 → one eviction + "set 2560"; at 2560
  → **both** regions from the spare, zero eviction, same text;
* a defect 7c exposed in 7a's accounting is fixed with it: `ggml_cuda_slab::mapped` is the slab's **VMM** extent,
  and pinning region 0 above the arena grows it, so the second pin's eviction test read the slab instead of the
  arena.  `mapped_arena` (seeded at init, grown by `slab_extend`, never moved by a pin) is the arena's extent and
  is what `arena_top()` and the pin read.

### 2.5 Which arm reaches which slot (do not re-derive)

| shape | slot 0 (draft) | slot 1 (borrowing) |
|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40` + the pair (the soak) | — | **pinned** (`aux narrow-3`) |
| **Gemma4 26B-A4B Q8_0 `-ncmoe 20` + `-md mtp-gemma-4-26B-A4B-it.gguf`** | **pinned** (`aux narrow-2`, 0.50 GiB) | **pinned** (`aux narrow-3`, 0.50 GiB) |
| 27B (dense, no slab) | n/a — no slab is armed | n/a |
| 35B-A3B + `--spec-type draft-mtp-adaptive` (embedded head) | no second compute buffer at all | — |
| 35B-A3B + `--spec-type draft-dflash2` + `-md` | no second compute buffer, **and the text is not run-stable** (4 runs, 4 hashes at 2 fixed lengths — residency picks the kernel set) | — |

### 2.6 Debug recipes for "two contexts, a GPU fault, or wrong numbers"

1. `GGML_CUDA_WORK_ALLOC_DEBUG=1` — every compute-buffer alloc/release with its base offset (named the alias in
   one run);
2. `-lv 4` for censuses, **no `-lv`** for a text/hash arm (a `-lv 3` run corrupts the text and hides the censuses);
3. `scripts/check-gpu-faults.sh --since "<start>" --log <log>` — never grep for `page fault` by hand;
4. `MOE_EXPERT_HEATMAP=load` on every host-expert arm, or the residency prior drifts between runs;
5. `rm -f gpucore.*.gpu` (32 GB each) after any fault, before anything else.

---

## 3. The rest of the campaign

**If you are the STEP 11/12 session: skip to `### STEP 11` and `### STEP 12` at the END of this section.**  The
blocks before them are the CLOSED steps, kept because they carry the reasoning, the traps and the anchors a
regression question will send you back to (STEP 8 the gate set, STEP 9 the shape/latency matrix, STEP 10 the
speculator).  `PLAN.md` has each step's own detail; `RUNS.md` the numbers.

### STEP 8 — the gate set (CLOSED 2026-11-23)

**Done.**  Five gates, each printing its own non-vacuity line, all green:

| gate | what it asserts |
|---|---|
| `gate-arms-default-inert.sh` | pair OFF on **4 shapes** == the anchors: cli text vs `baseline/inert-anchors.tsv` (provenance printed: 3 of 4 rows are PRE-CHANGE STEP 0 rows), server startup shape vs `baseline/inert-shapes/` (the 4B one IS `baseline/A6-startup-shape.txt`).  Sensitivity: pair ON must print a DIFFERENT shape (4 lines) and read the SAME responses |
| `gate-arms-n1-memory.sh` | one memory, byte text, the decode-arm census, the I4 reservation census |
| `gate-arms-itl.sh` | the latency claim at a stated `-ub` |
| `gate-arms-multislot.sh` | the 4-slot soak (`ARMS=` for the pair-OFF control, the heat prior pinned, fails if a slab-armed shape gave the decode arm no region) |
| `gate-arms-recurrent-shared.sh` | **new**: the probe promoted -- placements DIFFER (`head=0 n=1` vs `head=1 n=1`), concurrent == `--solo`, and the four-axis `none` == adaptive with a non-zero draft/accept count, on **35B-A3B and Flash-Next/qwen4exp**, both on a **3-device `-sm tensor`** split (`--axis-shape host` for the delivery's host-expert arm) |

Two arms the new gate caught while being written (both in `RUNS.md`, and both the HARNESS's bug): an adaptive
drafter driven at `--spec-draft-n-max 2` -- an invalid pair, since the adaptive floor is 3 (delivery block 02);
the same arm on a build of the delivery tip behaves identically, so nothing was broken and **no delivery change
is proposed** (the maintainer's decision) -- and a gate that holds the GPU flock across `mtp-axis.sh` deadlocks
on its own fd (`mtp-axis.sh --no-lock`).

Also closed here, because STEP 8's shape survey found it: **STEP 7e** -- STEP 4's `thread_local` Meta view slot
was bound only after `sched_alt_select()`'s no-op early return, so on `-sm tensor` a second context's graph
resolved under the first's view containers (`GGML_ASSERT(tensor)`; the delivery build runs that shape).  Fixed,
re-verified, `INVARIANTS.md` r368-wip.

The original STEP 8 scope, for the record: three gates existed and were green: `gates/gate-arms-n1-memory.sh` (one memory, byte text, decode-arm census, the I4
reservation census), `gates/gate-arms-itl.sh` (the latency claim at a stated `-ub`), `gates/gate-arms-multislot.sh`
(the 4-slot soak; it now takes **`ARMS=`** for the pair-OFF control, pins `MOE_EXPERT_HEATMAP=load`, and **fails
if a slab-armed shape did not give the decode arm its own region**).

What was owed (both now written):

* **`gate-arms-default-inert.sh`** — the pair **OFF** must (a) print a startup shape **byte-identical** to
  `baseline/A6-startup-shape.txt` (normalise with `baseline/norm-log.sh`, cut at the `listening on` line) and
  (b) read A1/A2/A3 identical, on **4 shapes** (4B 1 GPU, phi-4 1 GPU dense, 27B 3 GPU `-sm tensor`, 35B-A3B
  `-ncmoe 40`).  The anchor file and the shapes exist; this gate is plumbing + the shape list + the hashes.
* **`gate-arms-recurrent-shared.sh`** — promote `gates/probe-recurrent-shared.cpp` into a gate with its
  **non-vacuity** line (the two live contexts' placements must **differ**: `A head=0 n=1` vs `B head=1 n=1`; the
  STEP-4 control prints `head=1 n=1` for both) on **two** recurrent models (35B-A3B and Flash-Next/qwen4exp).
* **Every gate must print its non-vacuity line** — a gate that could pass on the single-context server is
  defective.  Two hard-won rules for the gate scripts: pin `MOE_EXPERT_HEATMAP=load`, and state the repeats (a
  "PASS 3/3" on an intermittent arm is a lottery ticket, not a rate).

### STEP 9 — the shape/latency matrix + the records (CLOSED 2026-11-23)

**Done**: `baseline/shape-matrix.sh` (7 shapes x 3 repeats x 2 arms, one request in flight, VRAM probed while
loaded) and `server-arm.sh --vram-probe`.  Results in `RUNS.md` and `ENVIRONMENT.md` §24: 7/7 text-identical, the
cost is +53..+154 MiB (+149/154 per device on a tensor split; one 1280 MiB slab aux region on a host tier), and
the trade is ITL **−22.8 %** at `-ub 2048` with prefill **+24.1 %** (3 repeats per arm).  It also found **STEP
7f** (a borrowing context carried an uninitialized `cparams.type_k`; Flash-Next + the pair aborted 2 of 3 runs)
and corrected `gate-arms-itl.sh`'s text check (a byte claim under concurrency is a lottery).

The original scope, for the record:

4B (1 GPU), phi-4 IQ4_XS (1 GPU, dense), 27B Q8_0 (3 GPU `-sm tensor`), 35B-A3B Q8_0 (`-ncmoe 40`, host tier),
Flash-Next IQ4_NL (3 GPU tensor, 3 slots `--kv-unified`), Gemma4 (iswa), **plus** the external-MTP-head shape
(`Gemma4 26B-A4B + -md`, §2.5) as the draft-context row.  **Byte text claims at ONE request in flight only** — a
concurrent arm's composition picks the kernels and even the unmodified server varies run to run
(`INVARIANTS.md` r362-wip item 5).  Then the `ENVIRONMENT.md` rows for `--prefill-decode-arms`,
`--prefill-decode-arms-mode`, `LLAMA_ARG_PREFILL_DECODE_ARMS[_MODE]`, `LLAMA_RS_DEBUG_PLACEMENT`, plus the
measured reserve and the VRAM delta per extra slot (and the fact that an aux context's compute region costs one
chunk-rounded layout, §2.4).

### STEP 10 — re-admit the speculator on the pair (CLOSED 2026-11-24)

**Done, in two landings.**  10a (`e7ae0b470`) extracted `slot_accept_draft_round(slot, ctx, seq_rm_type)`; 10
(`f3e34859d`, `steps/STEP-10-0018-speculator-on-the-pair.patch`) is everything else:

* `slot_draft_setup(slot, ctx, seq_rm_type)` and `slot_draft_ckpt(slot, ctx, seq_rm_type)` extracted from
  `update_slots` the same way, so BOTH arms run one draft round -- and the whole round (setup -> draft -> ckpt) is
  ONE critical section, because `common_speculative_draft()` clears every `drafting` flag when it returns,
  including the flags the other arm just set;
* the decode arm's round: draft setup -> draft -> checkpoints -> `handle_last_sampled_token(batch_dec)` (it already
  appends the sampled token plus every draft row and fills `spec_i_batch`) -> `llama_process(ctx_dec)` ->
  `common_speculative_process` -> `slot_accept_draft_round(..., can_finish=false)` per slot, else `slot_step`;
* `arms_spec_mtx`, a **recursive** mutex, held by both arms' speculator paths AND across the decode arm's own
  `llama_process(ctx_dec)` (the draft context is decode-band, DESIGN §8 rule 3).  Recursive because a round can end
  by printing the statistics line through `print_timings()`.  The prefill arm's WIDE pass does not take it;
* the slot's speculative state is reset at every handover, with an ASSERT that nothing is pending -- and a
  speculating slot is handed over only after its accept round resolved (a REPLAY round keeps `spec_draft` and must
  stay on the arm whose checkpoint produced it);
* `common_arms_pair_possible()` admits `draft-simple` / `draft-mtp` / `draft-mtp-adaptive` / the n-gram family and
  refuses the layer-input families and a gemma4 assistant -- the latter asked of the draft GGUF's
  `general.architecture` INSIDE the predicate, so the arena reserve and the pair decision still cannot disagree
  (V2); the server asks the loaded model the same question as a belt-and-braces check;
* `ctx_dec_seq_rm_type`, and the decode arm is reserved ONCE at its widest batch at pair creation.

**The oracle** is `gates/gate-arms-spec.sh` (new): the four-arm cold text claim on 3 shapes + the four-axis
protocol on the pair, with non-vacuity (the decode-arm census, drafts AND accepted drafts) -- 30/30 PASS, plus
`a87c4318b649` on the pair-off drafter path (the A5/10a value) and default-inert PASS.  Numbers: `RUNS.md`
2026-11-24.

**What re-admitting it cost, and where it now lives** (`RUNS.md` §6 has the table): six memory defects, four fixed
in this landing and two recorded as follow-ups (§7.2/§7.8).  Three of them needed two contexts live at once and
would have been invisible to any single-request arm: the embedded-MTP draft sharing the target's base-0 slab view,
a compute-buffer allocation / a cross-stream event wait overlapping the other arm's graph capture, and a second aux
region landing in the hole the first one's pin left mapped-but-absent.

**Traps this step paid for** (keep them):
* `accept_special_token` was a *lambda local to two different functions* (now a member helper);
* `slot.ctx_dft` is null with an embedded MTP head (both ckpt helpers handle that);
* the decode arm's batch must be constructed with `ctx_dec`, because the drafter's target-side reads follow the
  context that RAN the batch (`common_batch::ctx`);
* `populate_token_probs()` read `ctx_tgt` for the logits -- wrong on the decode arm (a `n_probs` request would have
  reported another context's distribution);
* a control arm's PASS is not proof a defect is gone: the ITL and soak arms below faulted intermittently, so every
  fix here was re-run at least twice (and the soak's overlap check needs `ROUNDS=20` with a drafter, because a
  verify step emits up to 8 tokens and two decoders rarely share a 16-token batch).

### STEP 11 — the delivery regression sweep  (11.1 what was touched, 11.2 the ladder, 11.3 a failure)

**Goal:** prove that everything this campaign landed is either INERT for the delivery's default path or covered by a
delivery gate, and that the delivery's own set still validates.  It is a *verification* step: the expected outcome
is "all green, nothing to change", and a change here is a finding to report first (§6).

#### 11.1 What this campaign changed in files the DELIVERY owns

Every one of these is inert with one context — but "inert by argument" is not the standard, a gate is:

| delivery-owned file | what landed | inert because | the gates that can see it |
|---|---|---|---|
| `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.{h,cpp}` | `ctx_memory` (the borrow, STEP 2), `cparams.type_k/v` for a borrower (7f), `ctx_aux` + `is_draft` (10), the aux floor slot / pin growth / region mapping (10), `slab_aux_slot()` | every field defaults to the old behaviour: `ctx_memory = null`, `ctx_aux = false`, and `slab_aux_slot()` returns 0 for a normal context | `gate-state-pipe.sh`, `gate-park-idle-slots.sh`, `gate-kv-residency.sh`, `gate-mtp-dev-rows.sh`, `gate-draft-ctx.sh`, `gate-spec-purity.sh` |
| `ggml/src/ggml-cuda/ggml-cuda.cu`, `common.cuh` | the capture barrier's two new users (an allocation EXCLUSIVE, a cross-stream event wait / peer copy SHARED) | a single context never has an allocation or an event wait beside a capture | `gate-graph-replay-width.sh`, `gate-kv-residency.sh`, `gate-host-pool.sh` |
| `ggml/src/ggml-cuda/ggml-cuda-slab.cu` | the aux pin's slot off-by-one, growth-when-idle, the region-below-the-extent mapping (10) | no aux context, so no region, so no pin path | `gate-host-pool.sh`, `gate-kv-residency.sh`, `gate-graph-replay-width.sh` |
| `common/common.cpp`, `common/speculative.{h,cpp}`, `common/arg.cpp` | `common_arms_pair_possible()`, the per-aux-context reserve, `ctx_tgt_alt`, the per-round target context, `common_speculative_path_shares_target_kv()`, the `--prefill-decode-arms*` options | the pair is OFF by default, so `common_arms_reserve_derive()` returns not-charged and the drafter keeps `ctx_tgt` (it must read the same rows) | `gate-mtp-dev-rows.sh`, `gate-spec-purity.sh`, `gate-draft-ctx.sh`, `check-gen-docs.sh` |
| `tools/server/server-context.cpp` | the two arms, the slot ownership flag, `slot_step`/`slot_draft_setup`/`slot_draft_ckpt`/`slot_accept_draft_round`, the handover resets | the prefill arm is the old code path when `ctx_dec == nullptr`, and the extracted helpers are the SAME code | `gate-park-idle-slots.sh`, `gate-spec-purity.sh`, `gate-ckpt-reuse.sh`, `gate-spec-ckpt-device.sh`, `gate-draft-ctx.sh` |
| `src/llama-kv-cache.*`, `src/llama-memory.{h,cpp}`, `src/llama-memory-recurrent.*`, `ggml/src/ggml-backend-meta.cpp` | one lock + the reservation bitmap (STEP 3), the contextified recurrent placement (5), the Meta view slot (4/7e/7g) | unchanged for one context: one owner, no reservation, one slot | `gate-kv-residency.sh`, `gate-state-pipe.sh`, `gate-seqpar.sh` (`-sm tensor` arms), `gate-meta-seg-drain.sh` |

#### 11.2 The ladder (run in this order; ONE GPU JOB AT A TIME)

```bash
cd ~/llama-cpp-rdna-boosts
./scripts/validate-set.sh                    # 1. the delivery set: checksums, fresh-tarball strict apply, file sets
                                             #    EXPECTED: green.  It cannot see the campaign (nothing in patches/), so
                                             #    a failure here means the tree or release.json moved -- stop and report.
./scripts/check-shlib-symbols.sh             # 2. every shipped .so resolves (already OK at 16 libs on this tip)
./scripts/check-gen-docs.sh                  # 3. the three generated docs (the pair ADDS options -> they must be
                                             #    regenerated IN THE CANDIDATE; see 12.2)
./scripts/check-windows-portability.sh        # 4. POSIX-only calls/types in inserted lines (the campaign inserts
./scripts/check-windows-paths.sh              #    server + llama-context code, so both are in scope)
./scripts/gate-prefill-logits.sh              # 5. the prefill-logit KLD: mean KLD <= 0.005, same-top-p >= 98 %
                                             #    (needs a base recorded on THIS ROCm line: --record from build-ref)
./scripts/gate-state-pipe.sh                 # 6. state save/restore: every arm restores byte-identical text
./scripts/gate-park-idle-slots.sh             # 7. idle-slot parking: park == lazy-off text + the indexer cache
./scripts/gate-kv-residency.sh                # 8. block 15's window + attendance pool, incl. the slab aux path
./scripts/gate-host-pool.sh                   # 9. the host-expert pool: 13 arms, byte-identical text
./scripts/gate-mtp-dev-rows.sh                # 10. the MTP h-rows gate: DEFAULT = host readback, no h_nextn device
                                             #     buffer; both arms the same text + acceptance (the drafter path)
./scripts/gate-spec-purity.sh                 # 11. plain == draft-mtp on q8_0 and bf16 at 3-device -sm tensor
./scripts/gate-draft-ctx.sh                   # 12. the context constructor's ctx_other adoption rule (this campaign
                                             #     edited exactly that layer: ctx_aux / is_draft)
./scripts/gate-moe-promise.sh                 # 13. the take-over promise's lifetime (the slab/MoE layer)
./scripts/gate-node-budget.sh                 # 14. graph_max_nodes / scheduler sizing
./scripts/gate-graph-replay-width.sh          # 15. the capture decision + the capture-time FA staging rule (the two
                                             #     new barrier users landed next to it)
./scripts/gate-ckpt-reuse.sh                  # 16. the context-checkpoint reuse path (server-context.cpp)
# then, and only then, the whole-set build + the fault scan:
~/bin/build-llama-rocm                       # the `all` profile at the tip (6-10 min cold, ~10 s warm)
for f in <the logs the gates wrote>; do ./scripts/check-gpu-faults.sh --since "<start>" --log "$f"; done
```

Notes that cost this campaign time:
* **`gate-prefill-logits.sh` needs a base recorded on THIS ROCm line** — and one IS on this box:
  `~/.cache/rdna-boosts/gates/prefill-logit-gfx1201-Qwen3.8-27B-Q8_0-rocm-10.2.0-gfx120X.kld` (5.1 GiB, recorded
  2026-11-23, base PPL 5.9325).  If it is missing, record it from `build-ref` with `--record` first; a base recorded
  on another line is refused, not silently compared.  Never point a base at `/tmp` (tmpfs here).
* **`scripts/gate-draft-ctx.sh`, `gate-mtp-dev-rows.sh` and `gate-spec-purity.sh` are the three that actually cover
  this campaign's edits** (the draft-context layer, the drafter's row path, and the verify/accept plane).  If you
  have time for only three, those are the three.
* **What is actually green, and on which tip** (this matters: do not read "the campaign is green" as "every gate
  passed on the last binary"):
  * on the FINAL tip `ffaee2c52`: `gates/gate-arms-spec.sh` 30/30 (3 shapes x 4 arms + 4 axes) and 19/19 on the
    post-10b re-run, `gates/gate-arms-default-inert.sh` PASS (4 shapes, then `--no-cli` after 10b),
    `gates/gate-arms-multislot.sh` PASS with the drafter at 20 rounds, `baseline/mtp-axis.sh` K axis
    `a87c4318b649` 74/73/435/423, `scripts/check-shlib-symbols.sh` OK (16 libs);
  * on a slightly EARLIER campaign tip: `gates/gate-arms-n1-memory.sh` PASS, the pair-only
    `gates/gate-arms-multislot.sh` (no drafter, `widest batch 3`), and 5 clean runs of `baseline/itl-arm.sh` with
    the drafter (ITL 1.26-1.28 ms/token).  Two later changes (the `is_draft`/`ctx_aux` test and the 10b reserve
    count) touch none of their code paths, so they stand;
  * **NOT run at all**: `gates/gate-arms-itl.sh` and `gates/gate-arms-recurrent-shared.sh` on the final tip, and
    **none of the 16 delivery gates** — that is STEP 11's job, and it is the reason STEP 11 exists.  If you change
    code during STEP 11, re-run the campaign gate that covers it.
* Anything you run at a host-expert shape: `MOE_EXPERT_HEATMAP=load`; anything fresh-process text:
  `ROCBLAS_USE_HIPBLASLT=0`; every arm: `HIP_VISIBLE_DEVICES` pinned.

#### 11.3 When a delivery gate FAILS

1. **Reproduce it on `build-ref` first.**  It EXISTS on this box: `~/llama.cpp/build-ref/bin/llama-server` reports
   `commit 089f90067` (the delivery tip), built at the same source path so ccache stays warm.  Use `LLAMA_BIN=` /
   `LLAMA_CLI=` (most campaign harnesses honour them) or the absolute path.  A scratch worktree also exists at
   `~/llama-scratch` (checked out at `089f90067`) if you need a clean delivery tree to build from.
2. `build-ref` fails too → **not this campaign's defect**.  Record it (a `RUNS.md` note + the `TODO.md`/issue it
   belongs to), report it, and move on.  Do NOT fix it inside STEP 11.
3. `build-ref` passes and the tip fails → **this campaign's defect**.  Localise before fixing:
   `scripts/gate-per-block-build.sh` exists for a *build* breakage, and `git bisect` over the campaign's `steps/`
   commits (each is one landing) for a behaviour breakage.  Then fix, re-run THAT gate plus
   `gate-arms-default-inert.sh`, and add a `WORKLOG.md` entry (`YYMMDD.VVV`, newest first) + an `INVARIANTS.md`
   entry if the lesson is general.
4. **Never** weaken a gate, skip an arm, or re-record an anchor to make a red gate green.  A gate that another
   change makes vacuous is documented in `MANIFESTS.md`; "simplify to the arm that passes" is the exact defect the
   gate exists to catch.

### STEP 12 — the candidate patch and the ask  (12.1 the cut, 12.2 the beta contract, 12.3 the report)

**The maintainer's decision (2026-11-24): this work COMPLETELY REPLACES `beta/prefill-decode-arms/`** — the old
directory's design text, README and patch are deleted, not kept beside it — **and on promotion it is delivery block
17 as ONE self-contained patch**, a modification to the lower-numbered blocks.  Promotion is still asked (§6).

#### 12.1 Cut it from a CANONICAL chain — never from the working branch

`~/llama.cpp` on `arms-v2` carries the campaign's own history on top of the delivery tree; a raw
`git diff <base>` there can export upstream commits or miss a landing.  The chain must be re-derived:

```bash
cd ~/llama.cpp
git checkout -q --detach 919794500            # release.json.base -- NEVER the tip of master
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .
#    -> prints "Applied tip: <sha>" and the applied TREE, and verifies it equals release.json.tree
git checkout -q rdna-boosts                   # the branch apply-all.sh created
```
(`--include-beta` is the TESTER's path — it applies whatever `beta/*/*.patch` holds, which at this point is the OLD
artifact.  Never use it to build the candidate; the candidate is cut from the campaign's `steps/` on a clean
delivery set.)
The canonical set has to be the delivery ALONE first (18 patches, tree = `release.json.tree`).  Then apply the
campaign's steps **in order**, each as its own `git am`, so the candidate is their union and you can still bisect:

```bash
for p in ~/llama-cpp-rdna-boosts/wip/prefill-decode-arms-v2/steps/STEP-*.patch; do
    git am -3 "$p" || { echo "FAILED on $p -- do not continue"; break; }
done
git status --short          # must be empty
```
The `steps/` files are `git diff` output (not `format-patch`), so apply them with `git apply --index` + one commit
if `git am` refuses them -- **do not** concatenate them into one file and `git apply` (the delivery's own rule:
that silently drops hunks).

Then, and this is the deliverable:

```bash
git diff 919794500 > ~/llama-cpp-rdna-boosts/beta/prefill-decode-arms/prefill-decode-arms.patch
git diff --stat 919794500 | tail -3        # record the size in the README
```

**Verify the patch round-trips before you write a word of documentation** (this is the contract a tester relies on):

```bash
cd /tmp && rm -rf cand && git clone -q ~/llama.cpp cand && cd cand
git checkout -q 919794500
git am  <beta>/prefill-decode-arms.patch
git rev-parse HEAD^{tree}        # must equal the tree you cut it from (`git rev-parse <campaign-tip>^{tree}`)
```

#### 12.2 The beta directory contract

`beta/prefill-decode-arms/` must contain **exactly three files** (the shape `scripts/apply-all.sh --include-beta`
expects, and the shape the delivery's `beta/` policy describes):

| file | what it must say |
|---|---|
| `prefill-decode-arms.patch` | the ONE patch above, cut against `919794500`, applying strictly with `git am` |
| `README.md` | what it is (one sentence: one `llama_memory_i`, two `llama_context`s — the prefill arm owns the memory, the decode arm borrows it, nothing is copied); the applied TREE; the shapes it reaches and the ones it refuses; **the measured numbers** from `RUNS.md` (`2026-11-24` — the four-arm text claim, the four axes, ITL, the cost); the options it adds (`--prefill-decode-arms`, `--prefill-decode-arms-mode`) and the fact that they DEFAULT OFF; what it does NOT cover (`LLAMA_MTP_DEV_ROWS` inert, the aux-pin growth path, the reserve count's wider-verify caveat); and that nothing here is covered by the delivery's release gates |
| `BETA-TESTING.md` | the ladder a tester runs, in order: apply → build → the 4-5 arms with the exact commands → what to report (their ROCm line, GPU model, build clean, the text hash per arm, the acceptance counts, the ITL pair) → the hygiene rules (`HIP_VISIBLE_DEVICES`, `setsid`+`timeout`+kill the process group, `rm -f gpucore.*.gpu` after a fault, `check-gpu-faults.sh`, `MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0`, no two GPU jobs) |

**Delete** the old artifact's three files and its design text.  The campaign's `README.md` §0 reading ban STAYS in
force for you (do not read the deleted material — it is in git history if a maintainer asks).  **Regenerate the three
generated docs** (`tools/cli/README.md`, `tools/completion/README.md`, `tools/server/README.md`) *inside the
candidate patch*, because the pair adds options: run `scripts/check-gen-docs.sh` on the candidate tree and fold the
regeneration into the patch (the delivery's rule: one regeneration, in the last block that touches `common/arg.cpp`
— here the candidate IS that block).

#### 12.3 The report, and the ask

Report to the maintainer in the session's final message (and as a `WORKLOG.md` entry, newest first) — with:

* the STEP 11 verdict per gate (and any finding, with the `build-ref` comparison that classified it);
* the candidate's applied TREE + the patch's sha256, the file count, and the round-trip proof;
* the measured trade: ITL, prefill, the memory/arena cost, and the shapes validated (and NOT validated);
* the residuals that a tester or a future block must know (§7 — `LLAMA_MTP_DEV_ROWS` inert on the pair, the aux
  pin's growth-only-when-idle, the reserve count's wider-verify caveat, the slab reservoir value to set);
* **the ask**: promotion into `patches/` as **block 17**, which is a maintainer decision and a block-structure
  decision — a green ladder is not permission (§6), and `AGENTS.md`'s "ALWAYS ask before creating a block" is the
  rule that applies.

Do **not**: touch `patches/`, `release.json`, `MANIFESTS.md`, `BASELINE.md` or any release gate; create or renumber
a block; push `~/llama.cpp` anywhere (a bare `push` there targets upstream `ggml-org/llama.cpp` and can get the
account banned); or push the delivery branch's candidate before the sweep is reported.

---

## 4. Environment and hygiene (each of these cost this session real time)

* **`llama-cli` always `--single-turn`** (+ `--no-display-prompt` when scripted); bound anything that can block
  with `timeout`; every server under `setsid`, and kill the **process group** afterwards.
* **`-lv` is per arm:** none for a text/hash arm, `-lv 4` for a census/mechanism arm, **never `-lv 3`**.  The
  library's own INFO and every ggml census need `-lv 4` on the server too.
* **One signal, then wait, when you intend to read a teardown census.**  `timeout` forwards its signal to the
  child, so an inner `timeout` + a group TERM makes the server see a "second interrupt" and exit without running
  a single destructor — destructor censuses then silently disappear.  `baseline/server-arm.sh` sends ONE TERM and
  waits.  Never SIGKILL a server whose teardown you are about to measure.
* **Never two GPU jobs at once** — make it a mechanism: `baseline/mtp-axis.sh` takes an exclusive `flock` on
  `~/.cache/rdna-boosts/gpu.lock`; this session wrapped every gate in `flock -w 900 … gpu.lock` as well.
* **Pin `HIP_VISIBLE_DEVICES`** (`0`, `0,2`, `0,1,2`): the 9950X3D iGPU (`gfx1036`) is otherwise enumerated and
  the load aborts with `invalid kernel file`.
* **`ROCBLAS_USE_HIPBLASLT=0` for every fresh-process text comparison**, and **never** put another ROCm install in
  `LD_LIBRARY_PATH` (it beats `RUNPATH`; `INVARIANTS.md` r359-wip).  ROCm 10.1 is retired; `scripts/rocmlib.sh`
  resolves the right lib dir.
* **`MOE_EXPERT_HEATMAP=load` on every host-expert arm** (§2.3) — otherwise the residency prior drifts and the
  same binary changes its verdict.
* **Big artifacts never go to `/tmp`** (tmpfs here): prefill-logit bases live in `~/.cache/rdna-boosts/gates`.
* **`gpucore.*.gpu` is 32 GB** — `rm -f gpucore.*.gpu` after any fault, before anything else.
* **Supervision contract** (`README.md` §5): every wait bounded, a dead *or silent* process fails fast, progress
  on stderr, and a timeout is a result, never a retry trigger.

---

## 5. Harness inventory (all under this directory)

```
baseline/  r353-anchors.md        the anchors A1-A7 and how to re-run them
           A6-startup-shape.txt   the 13-line pair-OFF startup shape (the default-inert anchor, 4B)
           inert-anchors.tsv      the default-inert cli TEXT anchors + each row's PROVENANCE (pre-change vs lock)
           inert-shapes/          one recorded startup shape per shape (+ the 4B pair-ON sensitivity fixture)
           server-arm.sh          start a server arm, drive N requests, hashes + the log; ONE TERM then wait;
                                  `--vram-probe` = whole-server VRAM/RSS while LOADED (STEP 9's memory row);
                                  `--nreq N` = how many requests (1 for a single-cold-request axis arm)
           shape-matrix.sh        STEP 9's measurement: N repeats x {pair off, on} per shape, text identity as a
                                  set + run-stability, VRAM delta per device, the reserve/aux/census lines
           itl-arm.sh             the ITL-through-a-prefill measurement (slot A decodes while B prefills)
           mtp-axis.sh            one four-axis MTP arm (GPU flock; --spec none|draft-mtp|adaptive).
                                  Shape options for another model: --sm/--ncmoe/--ctx/--load/--md/--tag-prefix;
                                  **--no-lock** when the CALLER already holds the GPU lock (a gate)
           norm-log.sh            a log's SHAPE (keeps both log line forms, masks addresses/decimals)
gates/     gate-arms-default-inert.sh  pair OFF on 4 shapes: cli text + startup shape vs the anchors, with the
                                    pair-ON sensitivity arm; `--record` writes a MISSING anchor (never overwrites)
           gate-arms-n1-memory.sh    both arms, ONE memory, byte text, decode-arm census, reservation census
           gate-arms-itl.sh          the latency claim (same text, ITL no worse) at a stated -ub
           gate-arms-multislot.sh    4 slots x 20 rounds soak; ARMS= for the pair-OFF control; pins the heat
                                     prior; FAILS if a slab-armed shape gave the decode arm no region of its own
           gate-arms-recurrent-shared.sh  the probe as a gate on 35B-A3B + Flash-Next: placements DIFFER,
                                     concurrent == --solo, four-axis none == adaptive with drafts accepted
           gate-arms-spec.sh         STEP 10's gate: the four-arm cold text claim (pair x spec) on 3 shapes,
                                     the four-axis protocol ON the pair, and the non-vacuity lines (the
                                     decode-arm census, drafts AND accepted drafts).  `--nreq 1` on the
                                     axis arms, `-lv 4` for the statistics (the text comes over HTTP)
           gate-arms-fd-budget.sh    STEP 14's gate (a beta tester's crash): a 3-device `-sm tensor` split must
                                     cost the SAME descriptors with and without the pair and with and without a
                                     drafter (an auxiliary context may add NO comm layer), the count must sit
                                     below FD_SETSIZE, and the backend must really have refused a second RCCL
                                     (`NOT bringing up a second RCCL comm layer`)
           gate-arms-cancel.sh       STEP 13's gate (a beta tester's crash): two concurrent generations, ABORT
                                     one (a client disconnect -> a CANCEL), N rounds per arm; the server must
                                     survive, the decode arm must provably have run (`finish_pending` is what
                                     prints the census, so one round lets a request FINISH), and the pair-OFF
                                     control must survive too.  FAILS on the pre-fix build (round 2)
           probe-*.cpp              the five non-vacuity probes (host stream, ctx-memory, kv reservation,
                                    meta slots, recurrent placement) -- each compiles against the built tree:
   e.g.  /opt/rocm-10.2.0-gfx120X/lib/llvm/bin/clang++ -std=c++17 -O1 -pthread probes.cpp \
           -I$HOME/llama.cpp/include -I$HOME/llama.cpp/src -I$HOME/llama.cpp/ggml/include \
           -L$HOME/llama.cpp/build-rocm/bin -lllama -lggml -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin
```

Every number of record is in `RUNS.md` (newest first) with its command and shape; the anchors are in
`baseline/r353-anchors.md` (+ `inert-anchors.tsv` for the default-inert rows).  The campaign's facts are in
`INVARIANTS.md` under `r353-wip` … `r379-wip` (r375-r379 are STEP 10's: the draft context that keeps no
`ctx_other`, the two capture-barrier users, the slab's aux regions, what a drafter-on-the-pair text oracle must
assert, and the reserve count).

The review machinery a cold STEP 11/12 session needs is in the DELIVERY repo, not here: `scripts/validate-set.sh`,
the `scripts/gate-*.sh` list in §3 STEP 11's ladder, `scripts/check-{gen-docs,shlib-symbols,windows-*}.sh`, and
`AGENTS.md` (policy: what a block is, the reporting rule, the push targets).  `MANIFESTS.md` is the delivery's
gate contract — read the entries for the gates you run, not just the gate's own `--help`.

---

## 6. Git, records and the reporting rule

* **Delivery repo** (`~/llama-cpp-rdna-boosts`), branch `prefill-decode-arms-v2`: docs, gates, harnesses, patches.
  `git push -u origin prefill-decode-arms-v2` after each landing (it is PUSHED as of this file: STEPs 0-10 +
  the review decisions).  **`main` stays untouched.**
* **Current tips:** `~/llama.cpp` `arms-v2` = **`ffaee2c52`** (STEP 10b), local only; the delivery branch's HEAD is
  the commit that carries this file.  A landing is one commit in EACH repo + a patch in `steps/`
  (`git diff > steps/STEP-<n>-<name>.patch` from `~/llama.cpp`), and the delivery commit carries the records.  STEP
  10's own landings are `e7ae0b470` (10a), `f3e34859d` (10) and `ffaee2c52` (10b); the `steps/` files are 0017-0019.
* **`~/llama.cpp`**, branch `arms-v2`: the code.  **Never push it** (a bare `push` there targets upstream
  `ggml-org/llama.cpp` and can get the account banned).
* **Records in the same landing:** a dated `WORKLOG.md` entry at the **top** (`YYMMDD.VVV`, today's real date +
  the next free number), an appended `INVARIANTS.md` entry tagged with the landing's `rNNN-wip`, and a `RUNS.md`
  section with the command, the shape and the number.
* **Report findings to the maintainer BEFORE committing a delivery-affecting change** — including this campaign's
  `wip/` landings.  **A new block, a renumbering, or a promotion into `patches/` is the maintainer's decision and
  is asked first; a green gate ladder is not permission.**
* If code and a document disagree: **the code wins for facts, the document wins for intent** — correct
  `DESIGN.md`/`PLAN.md`/this file in the same change.
* Corrections are owed, not optional: this session had to retract a "PASS" that was a lucky run (§2.3).  When a
  measurement contradicts a record, fix the record in the same landing and say so at the top of it.

---

## 7. Open questions for the maintainer (ask before deciding alone)

1. **Where should the slab changes live? -- ANSWERED by the maintainer (2026-11-24): ONE self-contained
   patch-block, and it becomes delivery block 17.**  This work **completely replaces `beta/prefill-decode-arms/`**
   (the old artifact's design/README/patch are deleted, `PLAN.md` STEP 12 / `AGENTS.md`), and on promotion it is
   **block 17 as a MODIFICATION to the lower-numbered blocks** — it edits files blocks 12/15 own
   (`ggml-cuda-slab.cu`) and the llama_context/llama.h layer, and that is exactly what a block on top is for.
   It is therefore ONE patch cut against the release tree, never a series by file-owner; the campaign's own
   `steps/` patches stay the review history.
2. **STEP 10's reserve -- ANSWERED by the maintainer (2026-11-24): the per-aux-context count (done).**
   `common_arms_reserve_derive()` now returns `{n_aux, bytes}` — **one pinned compute region per auxiliary
   context**, `n_aux x COMMON_ARMS_RESERVE_MIB_PER_AUX` — instead of a single lump for "the pair", and the log
   says which term charges which context: `the two-arm server reserves 1 auxiliary compute region(s) x 1280 MiB
   = 1280 MiB (the drafter's own region, when there is a drafter, is in the draft reserve term below)`.  `n_aux`
   counts the contexts whose region is NOT covered by another term (today: the decode arm; the drafter's region
   is its own reserve's `compute` term, derived or flat), because counting it twice would withhold ~1.3 GiB of
   arena that nothing uses — 4864 MiB of reserve against the 3584 + 1280 the two regions and the draft's KV
   actually need.  **Measured** (35B-A3B Q8_0 `-ncmoe 40` np 4 + `draft-mtp-adaptive`, single request each):
   arena 17467.5 -> 16192.5 MiB, hit rate 0.9173 -> 0.8968, generation 52.94 -> 48.22 t/s, text identical
   `cb8fa5a1790a`; the two regions measured decode arm **1280 MiB** (32 verify rows) and draft **512 MiB**, and
   the slab's calibration line states the total (`3328 MiB would host every auxiliary region (headroom 1536 + aux
   1792)`), which is the `GGML_CUDA_SLAB_RESERVE_MIB` the operator sets (zero evictions).  **Still owed:** a
   re-measurement on a shape whose verify width is much larger than `4 x (7+1)` rows, since the region grows with
   it (that is §7.8's growth path, now implemented).
3. **The drop barrier.**  STEP 7d removes the pair's exposure by never dropping.  The general fix — a barrier so a
   drop can run while the other arm is *not* inside `llama_process()` — is not implemented and would let the drop
   (and its arena win) come back.  Worth a step of its own before STEP 12, or a documented follow-up after
   promotion?
4. **The Step-9 byte-claim rule** ("one request in flight") is this campaign's reading of DESIGN V8.  If a byte
   claim *under* concurrency is wanted, the batch composition has to be pinned first — a separate instrument, not
   a gate tweak.
5. **ANSWERED by the maintainer (2026-11-23): `draft-mtp-adaptive` stays as it is — do not constrain a flagship
   feature.**  The "adaptive drafter at `--spec-draft-n-max 2` runs plain" observation was this campaign's
   HARNESS bug, not a delivery defect: the adaptive floor `n_min_adaptive = 3` landed with its validation in
   delivery **block 02**, so max 2 is a self-contradictory pair that never worked, and a build of the delivery
   tip `089f90067` behaves identically (same `E ... invalid adaptive draft range`, 0 drafts).  The gate now
   drives it at a valid ceiling (`--nmax 7`) and FAILs if a drafter arm generates 0 drafts.  Recorded in
   `INVARIANTS.md` r369-wip item 1 as a harness rule.
6. **ANSWERED by the maintainer (2026-11-23): a 3-device `-sm tensor` split IS part of this campaign.**  So the
   recurrent gate now runs both models' four-axis arms on it (`--axis-shape tensor` is the default), which makes
   the gate the regression instrument for STEP 7e.  Whether the DELIVERY's own `scripts/gate-draft-ctx.sh` should
   also gain a `-sm tensor` arm — its documented-but-unrun one, `INVARIANTS.md` r368-wip item 2 — is delivery
   gate work, not a decision this campaign takes.

7. **ANSWERED by the maintainer (2026-11-23): option 3 — build the barrier, keep both features on, do not take a
   slower decode.**  Implemented as **STEP 7h**: a capture/hipBLASLt barrier in ggml-cuda (capture exclusive,
   library GEMMs shared, `cublas_handle()`/`cublaslt_handle()` **creation** exclusive — creation is what actually
   tripped: `hipblasCreate` initialises hipBLASLt, which synchronises the legacy stream).  Flash-Next 3 devices
   `-sm tensor`, `-ub 4096`, all defaults: **53.99 -> 23.31 ms/token (−56.9 %)**, prefill +21.7 %, **7 of 7 runs
   clean** (was 2 of 3 aborts).  Text identity unchanged, the default-path startup shape unchanged, and
   `GGML_CUDA_CAPTURE_BLAS_BARRIER=0` is the kill switch.  `INVARIANTS.md` r374-wip records the three hypotheses
   that measurement killed on the way there.

8. **The aux-region SIZE -- acknowledged by the maintainer (2026-11-24): no decision needed; it is the hardening
   note below, and the growth path it names is implemented.**  Two things are now fixed (the floor call's off-by-one
   and growth-when-idle), but the mechanism is still "a region is sized by what a context asked for, and only
   grows while nothing is live at that slot".  A shape where a context's widest layout arrives *after* a view is
   live would decline it (the log says `... exceeds the pinned ... and a view is live there; falling back` and the
   buffer takes `cudaMalloc`, i.e. the reserve was charged and not used).  Not observed on any arm here (the
   decode arm's widest layout arrives inside its FIRST reserve, which the pair now forces at creation), so it is a
   hardening item, not a defect: the reserve is now a per-aux-context COUNT (§7.2) and the slab's calibration line
   states the total, so what remains is letting a pin be re-mapped while a view is idle (today it declines).

9. **`LLAMA_MTP_DEV_ROWS` is inert on the pair -- NOTED by the maintainer (2026-11-24); keep the log line.**  The device h-row path binds `ctx_dft`'s state
   source to ONE target context (`llama_set_ctx_state_src`), and re-pointing it per round marks the scheduler for a
   re-reserve -- i.e. it would cost the feature.  So the pair keeps the host readback (the delivery DEFAULT since
   r293) and now SAYS so in the log rather than silently reading the other arm's rows.  A per-arm state source, if
   wanted, is a follow-up; the delivery's own `scripts/gate-mtp-dev-rows.sh` does not cover the pair.

10. **The soak's overlap check needs `ROUNDS=20` when a drafter is live -- ACCEPTED by the maintainer (2026-11-24)
    and ENFORCED in the gate**: `gate-arms-multislot.sh` raises `ROUNDS` to 20 for a `--spec-type` arm that is not
    `none` and prints why, rather than lowering the check.  `gate-arms-multislot.sh`
    asserts that two decoders shared ONE decode batch at least once.  A verify step emits up to `n_max+1` tokens, so
    a 16-token request finishes in ~6 decode-arm steps and two slots rarely overlap: at `ROUNDS=10` one run in eight
    saw `widest batch 1` for the whole soak (no fault, no error -- the CHECK failed, not the server).  At
    `ROUNDS=20` it is 2/2.  The check stays hard; the drafter arm is run with 20 rounds.
