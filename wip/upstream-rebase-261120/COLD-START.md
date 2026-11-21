# COLD START — r328 upstream re-base, R12 landed, resume R13

**Read this file top to bottom before touching anything.** It is self-contained: a fresh session should not need
the campaign's history, only this, then the two files it names in §3.

Campaign dir: `~/llama-cpp-rdna-boosts/wip/upstream-rebase-261120/` (the delivery repo).
Ladder checkout: `~/llama.cpp`, branch **`rebase-r328`** (never commit to `master` there — it is the pristine
upstream snapshot).

---

## §0 What is being done, in one paragraph

The delivery's 18-block patch set (`patches/` in `~/llama-cpp-rdna-boosts`) is being **re-based onto a new upstream
fork point**: from `4f9dfa0e3` (r327's base) to **`919794500`** — 228 upstream commits, 1944 files,
+147k/−41k. The method is a **ladder**: one `git cherry-pick` per delivery block ("a rung"), each rung resolved by
hand, built, coherence-checked, then committed, so a session can stop after any rung. Target revision: **r328**.
Nothing in `patches/`, `release.json`, `rdna-boosts-all.patch` or the fork is touched until **R18**.

## §1 Exact state — verify this first

```bash
cd ~/llama.cpp && git status --short && git branch --show-current && git log --oneline -1
```

Expect: **clean**, branch `rebase-r328`, `81523fda6 rdna-boosts: block 13: MTP draft vocabulary subset ...` (tree
`080d7ecea198417011ca068916c5deafa9f4ce14`).  If a cherry-pick is mid-flight (`git status` shows unmerged
paths), the previous session stopped inside a conflict: finish it or `git cherry-pick --abort` and redo that rung.

| item | value |
|---|---|
| rungs landed | **R00 … R13** (14 of 18) + 2 rung repairs (R00's top-k gather, R05's `nb32` widening) |
| rung in progress | **none — R13 landed 2026-11-20** (`81523fda6`, tree `080d7ecea198417011ca068916c5deafa9f4ce14`) |
| next rung | **R14** (block 14, `git cherry-pick e88d45d9e`, 3 conflicts expected) |
| remaining | R14, R15, R16, R17 (block `0300`), R18 (release) |
| delivery repo | docs-only commits so far (`patches/`, `release.json`, `rdna-boosts-all.patch` untouched until R18) |
| new fork point | `919794500` |
| build | `cd ~/llama.cpp && ~/bin/build-llama-rocm`, then `cmake --build build-rocm -j 24` for the per-rung loop (`build-rocm`: `GGML_HIP_NO_VMM=OFF`, `GGML_CUDA_FA_QUANTS` includes `iq4_nl-iq4_nl`, `LLAMA_USE_PREBUILT_UI=OFF`) |
| the shipping reference | `~/llama.cpp/build-rocm/bin-r327` (the r327 `bin/`) — **the attribution oracle for any abort** |

## §2 The numbers to beat / the anchors to match (do not re-derive)

**Coherence (every rung, and after any change):**

```bash
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 \
  --no-display-prompt --single-turn
# hash it with: ~/llama-cpp-rdna-boosts/scripts/extract-generated.py LOG
# EXPECT: 100 chars sha=1c5d32ac537d   (identical at every rung so far)
```

Note: **no `-lv`** on a text arm (it corrupts the stream). A census arm is a separate `-lv 4` run.

| arm | command sketch | expected / reference |
|---|---|---|
| 35B-A3B Q8_0, 2 dev `-sm tensor`, `prompts/prose-rdna-boosts.txt` | `-ngl 99 -sm tensor -c 8192 -ub 1024 -b 1024 -n 16 --seed 42 --temp 0` | `73 chars sha=86b7f9b5aa80`; `4850.8 / 80.6 t/s` (R08) |
| 27B UD-IQ3_S dense (default MMB), 1 GPU | `-ngl 99 -sm layer --fit false -c 8192 -fa auto -ub 1024 -b 1024 -n 8` | `50 chars sha=ebf2a5a9b00a`; `1139.4 / 38.0 t/s` (R10) |
| host-expert arena, 35B-A3B Q8_0, 1 GPU | `-ngl 99 -ncmoe 20 -c 8192 -ub 1024 -b 1024 -n 16`, `MOE_EXPERT_HEATMAP=load` | text `86b7f9b5aa80`; **decode is a RELEASE BLOCKER — see §8** |
| MUL_MAT_ID op matrix | `test-backend-ops -o MUL_MAT_ID` | 1079 OK / 0 FAIL, 16 `dup=1` |
| NORM/RMS_NORM/GROUP_NORM/SET_ROWS | `test-backend-ops -o NORM,RMS_NORM,GROUP_NORM,SET_ROWS` | 368 OK / 0 FAIL |

## §3 Read order (in this order, only what you need)

1. **This file** (§0–§9).
2. `RUNGS.md` — the per-rung ledger: every conflict and its resolution, plus each rung's findings. Read the last
   landed rung (R11) and skim the headers of R06–R10 (they carry obligations that are still open).
3. `DECISIONS.md` — the decision log. Only **D4** (resolved, additive), **D8**, **D9**, **D12**, **D13** are still
   live references; everything else is closed.
4. `~/llama-cpp-rdna-boosts/AGENTS.md` — the policy layer (apply with `scripts/apply-all.sh`, `-lv` rules,
   goldens, push policy). Its *Verify* table is what a rung's exit criteria come from.
5. For R12 (landed): `RUNGS.md` §R12 (esp. §R12.3, the two splice-orphaned server features, and §R12.4, the
   DFlash decision).  `r12-wip/` is the paused session's evidence and is no longer an entry point.

`HANDOVER.md` is session-2/3's delta and is now partly historical; this file supersedes it as the entry point.

## §4 The rules that actually bite (all of these have bitten in this climb)

- **Analyse `-lv` per arm**: a text/hash arm takes **no `-lv`**; a census arm takes **`-lv 4`**. `-lv 3` pollutes the
  text and still hides every ggml/llama census.
- **Pin `HIP_VISIBLE_DEVICES`** on every multi-device arm (`0` = one R9700, `0,2` = the x8 pair, `0,1,2` = all
  three). Otherwise the iGPU is enumerated and the load aborts with `invalid kernel file`.
- **Build-completion detection**: never poll `pgrep`/`grep` for a build. Use a sentinel:
  `(cmake --build build-rocm -j 24 > L 2>&1; echo "BUILD_RC=$?" >> L)` then poll `L` for `BUILD_RC=`.
- **Any abort gets the same arm re-run on `bin-r327` before it is called a regression.** If r327 is clean and this
  tree is not, name the block that owns the fix and record it as an obligation.
- **Never `git apply` the concatenated patch series** — `scripts/apply-all.sh` is the tested flow (strict `git am`).
- **Never push** anything from `~/llama.cpp` except the release-time `rdna-boosts` refresh to the personal fork
  (`scripts/fork-refresh.sh`). A bare `git push` there targets upstream `ggml-org/llama.cpp` — never acceptable.
- **Report findings before committing a delivery-affecting change**; a rung's own log in `RUNGS.md` is where a rung's
  findings go.
- **A rung must build and pass its coherence arm.** It is better to pause a rung (as R12 was) than to commit a
  tree that does not build — but a paused rung must capture *all* its resolved work in one patch (r12-wip did).

## §5 Ladder status (one line each)

| rung | block | state |
|---|---|---|
| R00 | 00 foundation | done — `#32505` taken upstream-side; `mm_ids_helper_512_10` re-ported; `#30542` revert carried; top-k gather repair |
| R01 | 01 all-reduce | done — clean apply + one repair (64-bit warp mask) |
| R02 | 02 adaptive MTP | done — DFlash spec refactor resolved as MERGE; adaptive arm 29.3→65.2 t/s |
| R03 | 03 chunked GDN | done — test-file union; +8.8 % prefill |
| R04 | 04 BF16 KV/FA | done — 7 rope arms; `#32387`'s guard dropped; +5.8 % pp / +16.5 % tg |
| R05 | 05 FA band + q8_0 K/V | done — D16: the delivery's band model wins, `#32407`'s rotation not carried; FA matrix 7428/0 |
| R06 | 06 MMB | done — finding R06.2 (MMB cache was pool-unsafe; fixed by block 10) |
| R07 | 07 k-quant boosts | done — MMQ/MMVQ knobs measured per knob |
| R08 | 08 fused kernels | done — D4 resolved (additive); 4 forced fixes incl. a silent `force_tiling`→`has_gate` template artifact |
| R09 | 09 host-resident MoE | done — `seq_rm` adopted; finding R09.3 (arena-off text divergence; closed at R10) |
| R10 | 10 slab + **single fusion dispatcher** | done — the dispatcher reconciled into `ggml-cuda-try-fuse.cu`; R06.2 dense half + R09.3 closed; **R10.3 decode regression opened (§8)** |
| R11 | 11 qwen4exp/QSA | done — D13 carried; `gate-hc-fusions.sh` **RED** (R11.2) |
| R12 | 12 DFlash/CPU wins/server cache/seqpar | **DONE 2026-11-20** — `a9d8e1a96`; the speculator class merged onto upstream's `init(opt)` seam (D17: the delivery's feature set kept, measured); six gates PASS (`spec-reuse-window`, `seqpar --record`, `state-pipe`, `ckpt-reuse`, `park-idle-slots`, `mtp-dev-rows`) |
| R13 | 13 MTP draft vocab subset | **DONE 2026-11-20** — `81523fda6`; clean apply, one rung repair (the meta device iface's missing trailing initializers), `test-mtp-draft-vocab` PASS, `--spec-draft-mtp-vocab` +7.6 % decode at identical text |
| R14–R17 | 14..0300 | not started |
| R18 | release | not started (a long checklist, `RUNGS.md` R18) |

## §6 R12 — landed; what it cost, and what to do first in R13

R12 is in (`a9d8e1a96`, tree `f0f9db34b1c3ae8a93f834e5c2fad472113a1af9`).  The full log is `RUNGS.md` §R12; the
short version a fresh session needs:

* The paused session's ten resolutions were applied from `r12-wip/r12-resolved-but-unbuildable.diff`, then
  `common_speculative_impl_draft_dflash` was merged onto upstream `#32016`'s `dflash_options` + `init(opt)` seam
  (upstream's split classes, the delivery's features as state `init()` sets — `is_mrope`/`strict_pos`, the
  `LLAMA_DFLASH_INGEST_TAIL` window + `set_prompt_end`, the ingest filter and `spec_draft_align_cache()`, and
  `opt.dev_features` instead of the old `!is_lilicorr` guard).  The merged class now differs from upstream by
  exactly the delivery's 207 lines.
* Two features the splice had orphaned in `tools/server/` were restored: `prompt_load_with_room()` had **no
  caller** (so the parked-slot restore retry was dead code) and `server_prompt_cache::load()` read three
  locals that had moved into `find()`.
* `ckpt_recycle()` needed a deliberate `std::const_pointer_cast` against upstream's new
  `shared_ptr<const common_prompt_checkpoint>` immutability rule; the guard (`use_count() == 1`) is what makes
  it safe and `gate-ckpt-reuse.sh` is the oracle.
* **D17** — the DFlash decision, resolved by measurement in the maintainer's direction: the delivery's feature
  set stays on upstream's seam.  Numbers: `benchmarks/2026-11-20-r328-r12-dflash-merge.md`.

**First thing in R14:** re-run `scripts/gate-hc-fusions.sh` (R11.2 is still red and block 12's `hc`/`mmb` knobs
may be what restores the `mix_op` anchor) and keep R10.3 (the host-expert arena decode, 54.4 → 20.4 t/s, a
release blocker) in view for R15.

## §6b The rungs left (one line each)

| rung | block | source commit |
|---|---|---|
| R14 | 14 | `e88d45d9e` (streamed KV window + attendance pool + MTP draft ring, 3 conflicts) |
| R15 | 15 | `35a9ae985` (UMA GTT expert store — **R10.3 is owed here**) |
| R16 | 16 | `286191b31` (server prompt admission, D7) |
| R17 | `0300` | `e159adfc2` (`-sm relaxed-tensor` + `TODO` #351, the flint re-test) |
| R18 | — | the release checklist at the end of `RUNGS.md` |

## §7 The DFlash decision — **RESOLVED (D17): the delivery's feature set stays**, measured

Maintainer's instruction (2026-11-21): *"the best way to make a decision on which way to go with the dflash
choice is to measure which is more performant … I am reticent to drop its work for upstream's."*  Resolved at
R12 by measurement, in that direction.  Full record: `benchmarks/2026-11-20-r328-r12-dflash-merge.md`;
decision: `DECISIONS.md` D17; rung log: `RUNGS.md` §R12.4.  What a later session needs from it:

* **The graph builders are not a performance fork.**  `build_dspark_markov_head` and `build_lilicorr_scores` are
  byte-identical to the delivery's; `build_dflash2_conv` differs only structurally; the one real difference —
  the delivery's all-gather of the vocab-split logits in `build_dflash2_selector` — is a **correctness** fix
  (the pristine base aborts that arm under `-sm tensor`: `ggml-backend-meta.cpp`'s `handle_per_row`,
  `GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`; the plain control on that base runs).
* **The seam is structural**: upstream's class split + `dflash_options`/`init(opt)` is kept, the delivery's
  features (`is_mrope`/`strict_pos`, the `LLAMA_DFLASH_INGEST_TAIL` window and its `set_prompt_end` hook, the
  ingest filter, `spec_draft_align_cache()`, `opt.dev_features`) are state `init()` sets.  The pre-split
  duplicate state was deleted because upstream's subclasses already own it — no feature was dropped.
* **Measured** (3 targets × U/D/D-it/D-f1, 6K and 26K, `-n 2000`, `-lv 4` with the text on stdout): text and
  per-position acceptance identical in every pair; the ingest tail is **+29.8 % pp at 26K on the 2-GPU 35B** and
  +11.7 % on 3-GPU Flash-Next, ~neutral on a 1-GPU `-sm layer` cell (nothing to overlap — the delivery's own
  r130 record says the same); **F1 is +6-7 % pp on the dense 27B and inert on both MoE heads** (their draft
  contexts fail `llama_supports_embd_row()` and log "host copies"); decode is flat everywhere.

## §8 Open obligations carried into the next session (do not lose these)

| # | item | state |
|---|---|---|
| **R10.3 — RELEASE BLOCKER** | the host-expert arena arm's decode: 35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, `MOE_EXPERT_HEATMAP=load` → **R09 54.4 t/s, R10 20.4 t/s**, reproducible, text unchanged, and **no switch recovers it** (`DISABLE_FUSION=1` 18.5, `SLAB=0` 13.8; r327 reads 49.7) | maintainer, 2026-11-21: *"the decode regression must be fixed before we release; that can happen later."* Re-check at **R15**; if still open, bisect from `pool_stack` / `mmb_yield_caches` / the MMB slot lifetime. **r328 does not ship with it open.** |
| **R11.2** | `gate-hc-fusions.sh` **RED**: `mix_op`, `dsv4_pre_op`, `ops_no_post` left their r327 anchors (comm_op is exact; the chain itself moved). Half was predicted by block 11's own commit message; the fused **mixer** arm moving is new | re-run at **R13/R14/R17**; if `mix_op` still differs, the fused mixer must not ship default-on. Re-anchoring planted values is a **maintainer decision**, arm by arm |
| **R06.2 (routed half)** | the forced routed MMB arm (`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1`, 35B-A3B) still aborts on the pool LIFO assert where r327 reads `86b7f9b5aa80` | owed at **R15**/R18; it is an A/B override, not a shipped default (RDNA4: `routed=0`, Q8_0 not in the mask) |
| **R12-owed gates** | the gates for block 12 that R12 did **not** run: `gate-spec-align.sh` (`LLAMA_SPEC_ALIGN`), `gate-spec-ckpt-device.sh` (`LLAMA_SERVER_SPEC_CKPT_DEVICE`), `gate-spec-width-band.sh`, and the `-sm tensor` arms of `gate-host-pool.sh` / `gate-prefill-logits.sh` for block 09's Meta split-state cache (which D7/R12 edits) | run at **R13**/R18; `gate-seqpar.sh --record` was refreshed at R12, so R18 must re-record the prefill-logit base on the final tree |
| **R18 (delivery repo)** | drop block 10's `top_k` hunk is DONE (de-duplicated by construction at R10); still owed: squash `3ea8b0212` into block 00, re-measure `mm_ids_helper_512_10`, regenerate the generated docs and assert no drift, `TODO` #352 (container `web` stage), `TODO` #351 (the flint `-sm tensor` re-test, at R17), the full quant-coherence + QSA section E, `gate-norm-rows-alias.sh`, `MMB_J_MAX=8`'s floor | see `RUNGS.md` R18 and `HANDOVER.md` §2 for the full list |

## §9 Recipes

```bash
# a rung
cd ~/llama.cpp && git cherry-pick <block-sha>        # see RUNGS.md's ledger for the sha
#   resolve conflicts by hand; never -Xours/-Xtheirs
(cmake --build build-rocm -j 24 > /tmp/rNN.log 2>&1; echo "BUILD_RC=$?" >> /tmp/rNN.log)
#   poll the log for BUILD_RC=; then coherence (§2), then the block's own switch/census, then:
git add -A && GIT_EDITOR=true git cherry-pick --continue
git log -1 --format='%h %T %s'                        # record the tree in RUNGS.md

# verification after any merge-shaped resolution
git grep -n '<<<<<<<' | grep -v miniaudio             # expect only the binary vocab files
git grep -n 'TODO(rebase)'                             # expect nothing

# a gate that needs the delivery repo
cd ~/llama-cpp-rdna-boosts && ./scripts/gate-hc-fusions.sh          # R11.2
./scripts/gate-qwen4exp-quant-coherence.sh                          # QUANTS=… QSA_ARM=1 for the full arm
./scripts/gate-prefill-logits.sh                                    # any prefill-kernel change, and at R18

# record
#   RUNGS.md: one section per rung (resolutions table, exit criteria table, findings, notes-for-later)
#   then: cd ~/llama-cpp-rdna-boosts && git add -A && git commit
```

**A rung's log is written before the next rung starts.** Findings go in the rung's log, never in the commit message
only.

## §10 Where things are (bookmarks)

| what | where |
|---|---|
| the ladder, per-rung detail | `RUNGS.md` (this dir) |
| live decision items | `DECISIONS.md` (this dir) — D4/D8/D9/D12/D13 |
| R12 (landed) | `RUNGS.md` §R12; its evidence patch is `r12-wip/r12-resolved-but-unbuildable.diff` (historical) |
| the DFlash decision + its numbers | `DECISIONS.md` D17, `benchmarks/2026-11-20-r328-r12-dflash-merge.md` |
| the delivery's DFlash2 record | `archive/work/qwen4exp-dflash2/README.md`, `archive/work/dflash2-draft-sched2/README.md` |
| env knobs (incl. `LLAMA_DFLASH_*`) | `~/llama-cpp-rdna-boosts/ENVIRONMENT.md` |
| delivery policy | `~/llama-cpp-rdna-boosts/AGENTS.md` |
| engineering facts / "do not re-derive" | `~/llama-cpp-rdna-boosts/INVARIANTS.md` |
| MTP/DFlash validation protocol | `~/llama-cpp-rdna-boosts/benchmarks/mtp-adaptive-methodology.md` |
| prompts (hash-stable) | `~/llama-cpp-rdna-boosts/prompts/` — `prose-rdna-boosts.txt` is the 5298-token one used above |
| the r327 binaries | `~/llama.cpp/build-rocm/bin-r327` |
