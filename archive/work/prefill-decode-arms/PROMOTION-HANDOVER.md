# PROMOTION HANDOVER — promote `--prefill-decode-arms` as official block `0017`

**For a fresh cold session. Read this top to bottom, then `archive/work/prefill-decode-arms/RESULTS-2026-11-22.md` passes 10-18
(the record behind every claim here) and `archive/work/prefill-decode-arms/HANDOVER.md` (the technical state of the feature).**

This document is a **job ticket**, not a technical narrative. It says what to do, in what order, what "done" means,
and which decisions are already made (so you do not re-litigate them) versus which are yours to stop and ask about.

---

## 0. The mission, and the authority for it

Promote the campaign's work as a **new official delivery block, `0017`** — "the two-arm server: a prefill arm and a
decode arm sharing one KV" — and run every gate the delivery requires before a release.

**Decisions already made by the maintainer (do not re-open, and do not treat a green gate as permission to change
them):**

1. **A new block IS authorised, and it is `0017`.** This is the one thing `AGENTS.md` says must never be assumed — it
   has been explicitly granted. `0017` is the next official number: `0000`-`0016` are the 17 official blocks, `0300` is
   the temporary relaxed-tensor block (not counted), and the optional `0100`/`0101` live under `optional/`. `n_blocks`
   in `release.json` therefore goes **17 → 18**, `n_temporary` stays 1, `n_default` goes **18 → 19**.
2. **Default OFF, opt-in only.** `--prefill-decode-arms-mode` keeps its `off` default; `--prefill-decode-arms` remains
   an explicit opt-in. Nothing may be allocated, reserved or logged differently on a run that does not ask for the pair.
   **There is no `auto` flip in this release.** Flipping the default is a *later* decision, on its own evidence.
3. **Speculative decoding with the arms stays out.** The pair still refuses `--spec-type` (measured, pass 18: `auto`
   falls back and runs MTP; `on` refuses to start). That is the follow-up campaign.
4. **The meta-aware device copy stays out** (the ~74.6 ms serialised `-sm tensor` state move). Cost item, follow-up.
5. **Scope**: opt-in server feature + the memory-module work it needs. Anything that is not that does not go in.

If a gate fails in a way that forces a change to (1)-(5), **stop and ask**. Everything else in this ticket is yours.

---

## 1. Verified state you are starting from

| what | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`prefill-decode-arms`**, HEAD **`6ed2134e`** |
| delivery `main` | r343 (`b2b400c9`); this branch is rebased on it; nothing of the feature is in `patches/` |
| the WIP fork | `~/llama.cpp`, branch **`wip/prefill-overlap`**, HEAD **`9b8ac957f`**, **clean** |
| the fork point | `release.json.base` = **`919794500`**; the r343 chain tip the WIP branch sits on = **`e6b3cd3e5`** |
| the feature delta | **`e6b3cd3e5..wip/prefill-overlap`** — 15 commits, **25 files, +2296/−106** |
| release.json today | `n_blocks 17`, `n_temporary 1`, `n_default 18`, `n_optional 2`, `tip e6b3cd3e5`, `tree e71c0b59…` |

Verify all of that before touching anything; if a HEAD is newer, read the newer `WORKLOG.md` entries first (delivery
`261122.015`-`.023` are this campaign's).

---

## 2. What the block is (and what it is not)

**Is:** an opt-in server mode with two `llama_context`s — a prefill arm (the main loop, `ctx_tgt`) and a decode arm
(`ctx_dec`, its own thread, co-owning the prefill arm's KV) — where a slot crosses only at a request boundary, and the
recurrent state (hybrid models) follows it device-to-device. Plus the library work that makes a second context of the
same model possible: a **shared view of the KV for every memory module** (plain KV, plain hybrid, `hybrid_idx`, `iswa`),
one allocation owner for the co-owner pair, the recurrent-state handover, the indexer-pool staleness note, and the
declared second-context slab reserve.

**Is not:** speculative decoding with the arms, a meta-aware device copy, any default-on change, any change to a
shipped path's behaviour when the flag is off.

**The 15 commits, and which ones the block carries.** The two oldest are **measured losers kept in the WIP branch as
the record** and must **NOT** be in the block:

```
6f2a74717  DR0  the opt-in second context that shares the first one's KV           -> IN
3f2768131  the per-gather split yield point                                        -> OUT (measured loser)
540d442a8  stream priority (-1 low / +1 high)                                      -> OUT (measured loser)
d8a696252  the recurrent-state handover (device to device)                          -> IN
53ad4a6d4  compact env-gated cell censuses                                          -> IN (diagnostic)
0aff8602f  ONE allocation owner for the co-owner KV pair                            -> IN
9bb97b741  the two-arm SERVER cut                                                   -> IN
7c114eec0  -ib no longer counts a decode-arm slot                                   -> IN
16e942d2f  the REVERSE handover (prefix reuse)                                      -> IN
51c1fb8ec  the option tri-state + auto fallback + -ib/DUTY deprecation              -> IN
b822f89fc  the -sm tensor safety work (take views, serialised fallback, view slots)  -> IN
acd705146  hybrid_idx shares its memory + the indexer pool follows a handover        -> IN
423c91619  iswa shares its memory (gemma4)                                          -> IN
a7958316c  declare the decode context's reserve like an MTP draft's                  -> IN
9b8ac957f  never charge that reserve when a speculator is configured                 -> IN
```

**`3f2768131` and `540d442a8` are independent of the rest** — verified, not assumed: no later commit references
`stream_priority`, `per_gather` or the yield API (`git log -S`), and the only remaining "yield" is a comment in the
server cut saying the path does not use one. Dropping them removes an unused public API
(`ggml_backend_set_stream_priority_t`, the `llama_context_params` stream-priority fields) and **three files drop out of
the block entirely** — `src/llama-context.h`, `ggml/src/ggml-cuda/common.cuh`, `ggml/src/ggml-cuda/ggml-cuda.cu` are
touched by the two losers and by nothing else. So the block is **22 files, not 25**; take the line count from the
cleaned chain you cut and record it (do not subtract by hand — the overlapping files make that wrong in the third
digit). **Your first real task is to cut that cleaned chain and prove it** (§3).

---

## 3. Phase 1 — cut the block

1. In a **clean** `~/llama.cpp` at `release.json.base` (`919794500`), apply the delivery set with
   `scripts/apply-all.sh .` (strict `git am`) to get the r343 chain; confirm the tip is `e6b3cd3e5` and the tree is
   `release.json.tree`. **Never** `git apply` the concatenated series.
2. Replay the **13 IN commits** on top (`git cherry-pick`), dropping the two losers. Resolve nothing by hand if you can
   avoid it: `540d442a8`'s files (`ggml/src/ggml-cuda/common.cuh`, `ggml/include/ggml-backend.h`,
   `src/llama-context.{h,cpp}`) are also touched by IN commits, so a conflict is expected there — take the IN side and
   re-`cherry-pick` the rest.
3. **Squash into ONE block commit** with a block-style message (`0017: the two-arm server — …`), or keep a small,
   reviewable series if `scripts/make-patches.sh` produces one artefact either way. The deliverable is
   **`patches/0017-…-block-17-….patch`**, one file.
4. Build the whole applied set at the tip: `~/bin/build-llama-rocm-101` (BUILD_DIR=build-rocm-101 or the script's
   default; `all` target). **A block is a layer, never a standalone patch** — the gate is one build of the whole set
   (`AGENTS.md`, r332). Only if it does not build, localise with `scripts/gate-per-block-build.sh --changed-from …`.
5. `scripts/make-patches.sh` then `scripts/make-release.sh`, then **`scripts/validate-set.sh`** (strict apply on a
   fresh tarball, tree/count, per-artefact sha256, the Windows lints, the file-set reconciliation).
6. **File-set reconciliation** (`scripts/check-rung-file-sets.py`): the new block replaces nothing, but the *tree* delta
   must be exactly the feature's 25 files — check it and record the list in `patches/README.md`.

**Expected `release.json` after this phase:** `n_blocks 18`, `n_temporary 1`, `n_default 19`, new `tip`/`tree` from
`fork-refresh.sh`'s re-cut, `patches/0017-…` in the artefact map with its sha256.

---

## 4. Phase 2 — the gate ladder

`MANIFESTS.md` → *Release gates* is the contract. This section says which rows apply to **this** block and what the
acceptance is. **Two tiers, and both are mandatory:**

### 4a. The default path must be BIT-IDENTICAL (the "opt-in only" proof)

This block touches `create_memory()`, the KV allocator, `llama-context`, `ggml-backend-meta` and `common/`. With the
flag off nothing may move. Evidence to record, all at **`flag off`** against the **r343 build**:

| gate | acceptance |
|---|---|
| `scripts/gate-prefill-logits.sh` (default) | mean KLD **0.000000 / 100 %** vs the r343 base (record the base from the r343 build on this ROCm line; the gate refuses a base from another line, and it refuses a base on tmpfs) |
| same-seed greedy text, shipped path | **byte-identical** on 4B (1 GPU), 27B (3 GPU `-sm tensor`), phi-4 (dense, 1 GPU), 35B-A3B (`-ncmoe 40` host tier) |
| **MTP four-axis protocol** (`benchmarks/mtp-adaptive-methodology.md`) | plain == `draft-mtp` and acceptance identical to r343 — this block changes the memory modules the *draft* context also uses (`ctx_other`), so this is a real risk, not a formality. Pass 18 measured one MTP shape as identical; the four-axis is the gate |
| `scripts/gate-ckpt-reuse.sh` | PASS (the shipped path's checkpoint path is untouched) |
| `scripts/gate-park-idle-slots.sh` | PASS (idle-slot parking allocates/clears cells; the allocator changed) |
| `scripts/gate-state-pipe.sh` | PASS (batched state save/restore) |
| `scripts/gate-moe-promise.sh`, `scripts/gate-host-pool.sh` | PASS on their own shapes (the expert cache and the slab reserve both changed) |
| `scripts/gate-spec-purity.sh` | PASS (the FA/verify split is untouched but the KV allocator is not) |
| **`scripts/check-gen-docs.sh`** | **GREEN** — the block adds `--prefill-decode-arms`, `--no-prefill-decode-arms`, `--prefill-decode-arms-mode` (`common/arg.cpp`), so `tools/{cli,completion,server}/README.md` must be regenerated with `llama-gen-docs` and folded into this block (r329). This is a *build artefact*, easy to forget |
| `scripts/check-windows-portability.sh`, `check-windows-paths.sh` | clean (new C++ in `src/`, `common/`, `tools/server/`) |
| `scripts/validate-set.sh` | GREEN |

### 4b. The feature's own gates (with the flag ON)

All of these already have a runner in the campaign; the job is to make them **delivery gates** (or record why not):

| gate | acceptance |
|---|---|
| `archive/work/prefill-decode-arms/runs/coverage-matrix/matrix.sh` | **16/16 PASS** — 35B-A3B, Gemma4 26B-A4B, Flint (`-sm relaxed-tensor`), 27B × {layer, tensor} × {2, 3} GPU, arms-on text == arms-off byte for byte |
| `runs/host-tier-single-gpu/verify-postfix.sh` | **8/8 PASS** — the four models at 1 GPU `-ncmoe 48` with a **default arena and no knob**, plus Flash-Next 2/3 GPU, 3 GPU tensor, and `-c 204800 --parallel 3` |
| `runs/host-tier-single-gpu/verify-3g-cap.sh` | **3/3 PASS** with `MOE_EXPERT_CACHE_MIB=13653` (40 GiB total) — `h` **identical between arms** (0.9352 / 0.9376), i.e. residency-class matched |
| `runs/coverage-matrix/` tier-1 reuse oracle (`arms/gate-server-reuse.sh`) | `GATE=strict` PASS on 4B/phi-4/35B, `GATE=parity` PASS everywhere, `TWO_SLOT=1` PASS, `FORK_REUSE=1` on a dense model, `CONCURRENT=1` as **stress only** (no text claim, `GREEDY-PURITY.md` §63) |
| `scripts/check-gpu-faults.sh` | clean over every log of the ladder (never grep a log by hand) |
| **`scripts/gate-prefill-fairness.sh`** | **needs an arms-aware ITL criterion written first** (§4c) |
| **`scripts/gate-prefill-duty.sh`** | **needs a two-sided band** (§4c) |
| `wip/.../arms/gate-server-arms.sh` (tier 1) | PASS cold and over ≥3 warm pairs on the product shape (`REPS=N`; the **warm** pairs are the oracle, the cold host-tier rep is a §89 class) |

### 4c. Gate *infrastructure* this block owes (write it, then record the PASS)

1. **Fairness**: the shipped criterion `ITL_c <= 1.75 × ITL_b × chunk/b` computes **13 ms**, below the decode arm's own
   28-30 ms step, so the gate is red on a *correct* pair. Add an **arms-aware ITL criterion** (the intent: a decoder's
   worst in-window ITL must not exceed the decode arm's own step by more than the acceptance band) and keep
   `--max-itl-ms 120` as the absolute arm. `--self-test` must still reject the unfixed build.
2. **Duty**: inert on this path (75 → 73 tokens, stretch 1.01×). Give the gate a **two-sided band** (±10 %) so an inert
   mechanism reads PASS, and record "deprecated on this path" in its help/README.
3. **A long multi-turn soak**: cells filling to compaction, `try_clear_idle_slots` executing **on the arms path** (never
   run), several concurrent slots over many turns. Assert: no leak (`used` returns), no fault, text stable, and the
   pair still equals the single-context server on a class-matched shape.
4. **A `gate-host-pool` arm on the server**: the shipped gate is `llama-cli`-only. The pair's decode arm gathers too;
   the owed arm is a server-side pool/host-expert run asserting `takeover=0`, `broken=0 repaired=0` and identical text.
5. **A dense fork arm**: `FORK_REUSE=1` on phi-4 (a fork below the prefix reuses — verified in pass 7/8, keep it
   enforced).

---

## 5. Phase 3 — the documents the block owes

| file | what it needs |
|---|---|
| `patches/README.md` | the block's note: what it contains, its option, its knobs, its purity notes, its verified shapes and its **opt-in** status |
| `ENVIRONMENT.md` | every variable the block adds, with default + class: `LLAMA_ARG_PREFILL_DECODE_ARMS` (opt-in), `LLAMA_ARG_PREFILL_DECODE_ARMS_MODE` (opt-in, default `off`), the diagnostics `LLAMA_RS_DEBUG_TAKE` / `LLAMA_RS_DEBUG_CELLS` / `LLAMA_KV_DEBUG_CELLS` / `GGML_META_DEBUG_SET` / `LLAMA_RS_FORCE_SERIALISED` / `LLAMA_RS_VERIFY_TAKE` (diagnostic, inert), `LLAMA_KV_ALLOC_SHARED` (A/B kill switch, default 1), and the new use of `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` by the pair |
| `MANIFESTS.md` | the apply-order row for `0017` + its verification rows (the §4 ladder, tier 1/2) |
| `INVARIANTS.md` | **append** the new facts: (a) *every memory module must implement the shared view or the pair is blind* — `hybrid_idx` and `iswa` did not, and the failure is silent garbage/`COCOCO`; the engine warns `implements no shared view` and that line is now load-bearing; (b) a second context's graph reserve must be **declared before the arena claims the space** (the aux path), and a partial declaration fails; (c) a **budgeted** arena keeps both arms in one residency class, an auto arena does not; (d) the indexer pool is laid out from CELLS and is per context, so a restore/handover must stale it |
| `GREEDY-PURITY.md` | the findings this campaign measured: the residency-class pair (budgeted vs auto arena), the co-batched two-slot pair not being a text oracle (§63 already covers it — cite it), and the "same text across three residency regimes is an observation, not a claim" note |
| `README.md` | the consumer overview: the option, its opt-in status, `-np 1` behaviour, the deprecations (`-ib`, `LLAMA_SERVER_PREFILL_DUTY`), what the pair refuses |
| `TODO.md` | forward-looking items only: spec decoding with the arms; the meta-aware device copy; **derive the decode reserve from the decode width instead of the flat 3584 MiB** (measured need: 1792 MiB Flash-Next, 9.52 MiB 4B); `GLM5-Next` validation (same memory module, no checkpoint on this box); the recurrent-only memory gap (`llama_memory_seq_state_take` refuses a non-hybrid; **no Mamba-class model exists on this box**) |
| `WORKLOG.md` | **one** dated entry for the release (newest first), `YYMMDD.VVV`, referencing the gate results |
| `release.json` | regenerated by `scripts/make-release.sh` — never hand-edited |

---

## 6. Phase 4 — release mechanics (only after every gate is green and recorded)

1. `scripts/validate-set.sh` GREEN at the final tip.
2. **`scripts/fork-refresh.sh --build --gates --push`** — re-cuts `rdna-boosts` on the personal fork from
   `release.json.base`, checks the tree equals `release.json.tree` and the tip equals `.tip`, then pushes
   `--force-with-lease` to **`git@github.com:stew675/llama.cpp.git` only**. Never a bare `--force`, never upstream,
   never a push from `~/llama.cpp` by hand.
3. Tag the delivery (`v18-…`), then the **container images are built and pushed off-host by the worker watcher**
   (`scripts/container-watch.sh` on `radiance`) — **do not** run the two-line image build on the primary box.
4. The branch will *look* stale on GitHub (committer dates are the author dates) — freshness is
   `gh api repos/stew675/llama.cpp/branches/rdna-boosts --jq .commit.sha` == `.tip`. Compare **trees, never SHAs**.

---

## 7. Landmines that will bite (each cost a session in this campaign)

**Measurement**
* **Never export `TEMP` to a GPU process** — the ROCm 10.1 runtime uses it; `TEMP=0` surfaces as
  `hipStreamCreateWithFlags failed` → `ROCm error: out of memory` at context creation. Half a session went into that.
* **`kill -9` on a `timeout` wrapper leaves the server alive**, and `rocm-smi` does not count VMM allocations, so a
  "VRAM drained" check passes while the leaked server answers `/health` and becomes the arm you think you are driving.
  Start arms under `setsid`, kill the process **group**, refuse to start while any `llama-server` exists.
* A GPU OOM reported at `hipStreamCreateWithFlags`/`hipMemsetAsync` is a **sticky** error: read the **first** `E` line.
* Text is a **per-device-count**, per-split, per-residency-class reference (AGENTS.md rule 6, `GREEDY-PURITY.md` §38).
  Compare within one shape; a **budgeted** arena is the class-matched shape on a host tier.
* `-lv`: a text/hash arm takes **no** `-lv`; a census arm takes **`-lv 4`**; the server hashes over HTTP so `-lv 4` is
  safe there. `-lv 3` is the trap.
* Flash-Next is a **thinking** model — read `reasoning_content` too, and give it tokens.
* A leaked server, a cold host-tier rep, and a co-batched concurrent pair are **not** oracles. Use `REPS=N` warm pairs,
  `TWO_SLOT=1`, and never `CONCURRENT=1` for a text claim.
* `-sm tensor` with a hybrid model takes the **serialised state move** (~74.6 ms/boundary, 112.57 MiB) — known, not a
  bug; it lands in TTFT.

**Shape**
* `MOE_EXPERT_CACHE_MIB` is **MiB per device** and has a **floor of 2048** (`2000` aborts at load; `0` = the CPU expert
  path).
* Flint needs **`-sm relaxed-tensor`** for a tensor split (plain `-sm tensor` is refused at load — that is block `0300`).
* **`-sm tensor` + `-ncmoe` + the pair needs the declared reserve** — now automatic (flat 3584 MiB via
  `MOE_EXPERT_CACHE_AUX_RESERVE_MIB`); a manual `GGML_CUDA_SLAB_RESERVE_MIB` still wins. If you see
  `the decode arm's step failed`, read `SLAB_INIT … aux=` first: `0` means nothing was declared.
* The Flash-Next shape is **marginal on VRAM** (93.2 GiB of weights on 95.6 GiB): one load in five can OOM at the PLE
  lazy table. Retry; do not chase it as a defect.

**Process**
* Report findings **before** any delivery-affecting commit, and before any tag.
* Nothing in `patches/` until Phase 1 is *your* deliberate cut; no block is created by a gate outcome — the
  authorisation for `0017` is §0 of this ticket and nothing else.
* Leave both repos committed and `~/llama.cpp` clean.

---

## 8. Definition of done (checklist for the release entry)

- [ ] `patches/0017-…` exists, applies strictly, and the whole set builds at the tip.
- [ ] `release.json`: `n_blocks 18`, `n_temporary 1`, `n_default 19`, new `tip`/`tree`, every sha256 refreshed.
- [ ] `scripts/validate-set.sh` GREEN; `scripts/check-gen-docs.sh` GREEN; the Windows lints clean.
- [ ] **4a** recorded: KLD 0.000000 / 100 %, shipped-path text identical on ≥4 shapes, MTP four-axis identical,
      ckpt/park/state/host-pool/moe-promise/spec-purity PASS.
- [ ] **4b** recorded: coverage 16/16, host-tier 8/8, 3-GPU capped 3/3 (`h` identical between arms), reuse
      strict+parity, faults clean, tier-1 cold + warm pairs.
- [ ] **4c** written and recorded: arms-aware fairness criterion, two-sided duty band, the multi-turn soak, the
      server-side host-pool arm, the dense fork arm.
- [ ] Docs: `patches/README.md`, `ENVIRONMENT.md`, `MANIFESTS.md`, `INVARIANTS.md` (appended), `GREEDY-PURITY.md`,
      `README.md`, `TODO.md`, one `WORKLOG.md` entry.
- [ ] **Default is still `off`**, and the release's own numbers include the "off ⇒ bit-identical" evidence.
- [ ] Fork `rdna-boosts` re-cut, tree == `.tree`, tip == `.tip`, pushed to the personal fork with
      `--force-with-lease`; container watcher picked the tag up (or you have said why not).
- [ ] Nothing pushed to upstream; no container image build on the primary box.

---

## 9. If something is genuinely broken

Stop, write it down, and tell the maintainer **before** committing or tagging: what you measured, what it means, the
options and your recommendation. The campaign's own history is the argument for that rule — the P0 that blocked r343's
promotion was found only because a cold session ran the model nobody had gated, and it was fixed only because the
finding was reported before anything shipped.
