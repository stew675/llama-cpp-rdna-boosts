# COLD START — r328 upstream re-base, **all 18 rungs landed (R00-R17)**, resume for R18

**Read this file top to bottom before touching anything.** It is self-contained: a fresh session should not need
the campaign's history, only this, then the two files it names in §3.

Campaign dir: `~/llama-cpp-rdna-boosts/archive/work/upstream-rebase-261120/` (the delivery repo).
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

Expect: **clean**, branch `rebase-r328`, `73a74454b rdna-boosts: block 0300: ...` (tree
`0c46248f7371a0eee84a980a1e16ad25a6fbd38f`) — **18 commits**, the three rung repairs squashed (§R18.1) and the
block-03 radiance attribution folded in (§R18.5).  The ladder is **complete**: 18 block commits (`0000`..`0016` +
`0300`) plus two rung repairs.

| item | value |
|---|---|
| rungs landed | **R00 … R17 — all 18 blocks**, 2026-11-20, + 2 rung repairs |
| ladder tip | **`73a74454b`**, tree `0c46248f7371a0eee84a980a1e16ad25a6fbd38f` — the pre-squash tree `4e222b2e…` **plus** the +10-line Apache-2.0 attribution in block 03's two chunked-GDN headers (comments only) |
| next | **R18 — the release**, not a rung (see `RUNGS.md` R18 + §8 below) |
| repairs | **squashed 2026-11-21** (`RUNGS.md` §R18.1): R00's `3ea8b0212` → block 00, R05's `c3860af1e` → block 05, R16's `dd4fa5e22` → block 16.  18 commits = the 17 official blocks + `0300`, final tree unchanged |
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
2. `RUNGS.md` — the per-rung ledger: every conflict and its resolution, plus each rung's findings (the last four
   are R14-R17; §R14.1 is the op-count trap and §R15.2 is why block 15's gate is an APU gate).
3. `DECISIONS.md` — the decision log. **D17** (the DFlash choice) is the only one this ladder decided on
   measurement; D4/D8/D9/D10/D12/D13 are the merge-shaped ones a later reader may need.
4. `~/llama-cpp-rdna-boosts/AGENTS.md` — the policy layer (apply with `scripts/apply-all.sh`, `-lv` rules,
   goldens, push policy). Its *Verify* table is what a rung's exit criteria come from.
5. For the completed ladder: `RUNGS.md`'s ledger (one row per rung) and the four newest sections (R14-R17).  The
   two that decide R18's shape are §R14.1 (the `GGML_OP_COUNT` trap) and §R15.2 (why block 15's gate is an APU
   gate).  `r12-wip/` is the paused session's evidence and is no longer an entry point.

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
| R14 | 14 streamed KV (window + pool + draft ring) | **DONE 2026-11-20** — `9f8c11782`; op count recounted to **109** (patch 10), `gate-kv-residency` + `gate-spec-draft-window` PASS; `gate-node-budget` owed (no phi-4) |
| R15 | 15 UMA GTT expert store | **DONE 2026-11-20** — `07b293a7b`; **R10.3 CLOSED** (49.9 t/s median vs r327's 49.7), `gate-host-pool` **59/59** incl. the `-sm tensor` arms owed from R12; the GTT gate is an APU gate (inert on gfx1201 **by design**) |
| R16 | 16 server prompt admission | **DONE 2026-11-20** — `71ce858ca` (+ repair `dd4fa5e22`); `gate-prefill-fairness` PASS on three shapes + `--self-test`, `gate-prefill-duty` PASS dense and with `draft-mtp` |
| R17 | `0300` relaxed-tensor | **DONE 2026-11-20** — `ee502d397`; **`TODO` #351 answered: `0300` cannot be dropped** (flint still refuses `-sm tensor`), flint `relaxed-tensor` == `layer` byte-identical, the no-op arm holds on 35B-A3B |
| R18 | release | **not started — this is the next session's whole job** (a long checklist, `RUNGS.md` R18) |

## §6 The ladder's last four rungs, in one paragraph each (R12-R17 are all in `RUNGS.md`)

* **R12** (`a9d8e1a96`) — the paused speculator-class merge finished on upstream's `dflash_options`/`init(opt)`
  seam, two splice-orphaned server features restored, **D17** decided by measurement (the delivery's DFlash
  feature set stays).  Six gates green.
* **R13** (`81523fda6`) — MTP draft vocabulary subset, clean apply, its own test PASS, +7.6 % decode.
* **R14** (`9f8c11782`) — the streamed KV.  The one trap: `GGML_OP_COUNT` moved to **109** and every
  `static_assert` was merge *context*, so it had to be recounted (patch 10).  `gate-kv-residency` and
  `gate-spec-draft-window` PASS; `gate-node-budget` is owed (no phi-4 on this box — see §8).
* **R15** (`07b293a7b`) — the UMA GTT expert store.  **R10.3 is CLOSED** (arena decode 49.9 t/s median against
  r327's 49.7, text unchanged) and `gate-host-pool.sh` is **59/59**, including the `-sm tensor` arms owed from
  R12.  The GTT gate itself is an **APU gate**: on gfx1201 the tier is inert *by design* (the open/arena hooks
  return null unless `prop.integrated`), so its claims stay owed on the `halo` box.
* **R16** (`71ce858ca` + repair `dd4fa5e22`) — server prompt admission: fair share, `-ib`, skew, duty cycle.
  `gate-prefill-fairness` PASS on dense / `draft-mtp` / streamed-host-expert shapes plus `--self-test`, and
  `gate-prefill-duty` PASS dense and with `draft-mtp`.
* **R17** (`ee502d397`; post-squash `8406cbd86`) — block `0300`.  **`TODO` #351 is answered: `0300` cannot be
  dropped** — flint still refuses `-sm tensor` at this base with the block's own message, while
  `-sm relaxed-tensor` is byte-identical to `-sm layer` (`8ec5510e801e`) and a non-amax arch is unchanged
  (`86b7f9b5aa80` both ways).

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

## §8 Open obligations carried into R18 (this is the pre-release list)

| # | item | state |
|---|---|---|
| **R10.3** | the host-expert arena arm's decode (54.4 → 20.4 t/s at R10) | **CLOSED at R15**: 51.2 / 49.9 / 49.2 t/s, median 49.9, against r327's 49.7, text `86b7f9b5aa80`.  Nothing to fix — record it as closed |
| ~~**R11.2**~~ | **REFUTED 2026-11-21 (`RUNGS.md` R18.4)**: r327's own released binaries read the r328 numbers byte for byte on every arm, and the whole gate is red the same way on r327 — so the re-base moved NOTHING.  The gate's anchors were stale underneath a r290..r319 numerics change and the **reference arm was unanchored**, which is what made it read as "the chain moved".  The gate now anchors `chain_chain` too and its rows are re-derived from r327: **`local` and `meta` both PASS**, no delivery defect found | **CLOSED** — no maintainer decision needed; the fused mixer ships as before |
| **R06.2 (routed half)** | the forced routed MMB arm (`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1`, 35B-A3B) still aborts on the pool LIFO assert where r327 reads `86b7f9b5aa80` | an A/B override, never a shipped default (RDNA4: `routed=0`).  Owed as a measurement, not a blocker |
| **block 15's APU gates** | `gate-uma-experts.sh` (17/17 arms text-identical but its census arms cannot fire on gfx1201) and `gate-prefill-server-slots.sh` (needs the >RAM 122B model) | **owed on the `halo` (gfx1151) box** — the discrete side is `gate-host-pool.sh`, 59/59 PASS.  The 122B-A10B Q8_0 **is on `halo`**, so the `>RAM` shape is not missing, only mis-located: the gate belongs on the box where the GTT store is not inert.  r328 may ship with this recorded as an APU-unvalidated state (`AGENTS.md` allows a documented unvalidated state).  Flow: [`REMOTE-VALIDATION.md`](REMOTE-VALIDATION.md) |
| **gfx1151 + gfx1100 arch validation** | the whole ladder was climbed on **gfx1201 only**; the delivery's scope policy covers three arches, and the gfx11 chunked-GDN kernel file (`gated_delta_net_chunked_bf16_gfx11.cu`) is code gfx1201 **never runs** | **owed: one session per box, AFTER the release tag** — `ssh halo` (gfx1151 APU) and `ssh fingon` (gfx1100), both reachable, both carrying r327 as their own attribution oracle.  The per-box model inventory is now verified, and the primary box's canonical models (27B **Q8_0**, 35B-A3B **UD-Q4_K_M**) are **not** on either remote — each session names its own (halo: `Qwen3.5/122B-A10B/Q8_0` for the `>RAM` arm, `35B-A3B/F16`, `Flash-Next/Q4_K_M`; fingon: `35B-A3B/UD-Q4_K_M`, `Flash-Next/GSQ-IQ3_XXS`, `27B/UD-IQ3_S`, `Flint/Q5_K_XL`).  Inventory, what each box owns and the exact flow: [`REMOTE-VALIDATION.md`](REMOTE-VALIDATION.md).  They apply the **released** set, which also re-verifies `apply-all.sh` + `release.json` on two more machines (what r138 did) |
| **`gate-node-budget.sh`** | needs phi-4's tight shape; **no phi-4 on this box**, and the only candidate (`Inkling-2-Mini`) fails to load on the pristine base too (`blk.24.attn_q.weight`) — *and none on `halo` or `fingon` either* | owed at R18 (fetch phi-4 IQ4_XS, or accept it as a documented gap) |
| **`gate-spec-align.sh` / `gate-spec-ckpt-device.sh` / `gate-spec-width-band.sh`** | block 12/14 speculator paths | **not yet run on the completed tree** — R18's gate sweep |
| **the prefill-logit base** | `gate-seqpar.sh --record` refreshed it at R12, i.e. on an **intermediate** tree | **R18 must `--record` again on the final tree**, then run the gate |
| **`-Wdeprecated-declarations`** | `tests/test-logits-width-probe.cpp` (4), `tests/test-mtp-draft-vocab.cpp` (3) still use `llama_batch_init`/`llama_decode` | cosmetic; a `TODO` (D6's ext-batch migration of two test TUs), never a blocker |
| ~~repairs to squash~~ | **DONE** 2026-11-21 (`RUNGS.md` §R18.1) — all three folded, final tree unchanged | R18 step 4b |
| **R18 (delivery repo)** | `TODO` #352 (container `web` stage), regenerate the generated docs + assert no drift, the full quant-coherence + QSA section E, `gate-norm-rows-alias.sh`, `MMB_J_MAX=8`'s floor, `make-patches.sh` / `make-release.sh` / `validate-set.sh`, `fork-refresh.sh --build --gates --push`, the worker container build, the `WORKLOG.md` release entry | see `RUNGS.md` R18 |

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
| live decision items | `DECISIONS.md` (this dir) — D17, and the merge-shaped records D4/D8/D9/D10/D12/D13 |
| R12 (landed) | `RUNGS.md` §R12; its evidence patch is `r12-wip/r12-resolved-but-unbuildable.diff` (historical) |
| R11.2's open decision | `RUNGS.md` §R11.2 (the anchors) + the ladder-complete block at the end of `RUNGS.md` |
| the DFlash decision + its numbers | `DECISIONS.md` D17, `benchmarks/2026-11-20-r328-r12-dflash-merge.md` |
| the ladder's tip | `73a74454b`, tree `0c46248f7371a0eee84a980a1e16ad25a6fbd38f` (pre-squash `dd4fa5e22`, kept as `backup-pre-squash-r328`; blocks `0300` + the block-03 attribution fixup) |
| the R18 gate sweep's cost model | `MANIFESTS.md` → Release gates (gate 5) and `scripts/gate-per-block-build.sh`'s header |
| the delivery's DFlash2 record | `archive/work/qwen4exp-dflash2/README.md`, `archive/work/dflash2-draft-sched2/README.md` |
| env knobs (incl. `LLAMA_DFLASH_*`) | `~/llama-cpp-rdna-boosts/ENVIRONMENT.md` |
| delivery policy | `~/llama-cpp-rdna-boosts/AGENTS.md` |
| engineering facts / "do not re-derive" | `~/llama-cpp-rdna-boosts/INVARIANTS.md` |
| MTP/DFlash validation protocol | `~/llama-cpp-rdna-boosts/benchmarks/mtp-adaptive-methodology.md` |
| prompts (hash-stable) | `~/llama-cpp-rdna-boosts/prompts/` — `prose-rdna-boosts.txt` is the 5298-token one used above |
| the r327 binaries | `~/llama.cpp/build-rocm/bin-r327` |
