# COLD START — r328 upstream re-base, resume R12

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

Expect: **clean**, branch `rebase-r328`, `dbe4fdd40 rdna-boosts: block 11: qwen4exp ...` (tree `2b6aec1f0`).
If a cherry-pick is mid-flight (`git status` shows unmerged paths), the previous session stopped inside a conflict:
finish it or `git cherry-pick --abort` and redo that rung.

| item | value |
|---|---|
| rungs landed | **R00 … R11** (12 of 18) + 2 rung repairs |
| rung in progress | **R12 — paused, not landed** (§6) |
| remaining after R12 | R13, R14, R15, R16, R17 (block `0300`), R18 (release) |
| delivery repo tip | `8fb8e43f` (docs only; `patches/`/`release.json` untouched) |
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
5. For R12 only: `r12-wip/README.md` — the paused rung's state (what is settled, what is not).

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
- **A rung must build and pass its coherence arm.** It is better to pause a rung (as R12 is) than to commit a tree
  that does not build.

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
| **R12** | 12 DFlash/CPU wins/server cache | **PAUSED** — see §6 |
| R13–R17 | 13..0300 | not started |
| R18 | release | not started (a long checklist, `RUNGS.md` R18) |

## §6 R12 — how to resume (the next session's first job)

R12 = `git cherry-pick dad1442b3` (64 files, +10k). Ten conflicted files. **Ten of them are already resolved** —
captured in [`r12-wip/r12-resolved-but-unbuildable.diff`](r12-wip/r12-resolved-but-unbuildable.diff), with the
rationale per file in [`r12-wip/README.md`](r12-wip/README.md). Apply that patch over `dbe4fdd40` (or redo the
cherry-pick using that README's table as the checklist — it is 10 rows).

The one thing that does not build is `common/speculative.cpp`:

> **`common_speculative_impl_draft_dflash` is two implementations of one class.**
> * upstream `#32016` (the R02/D5 refactor): a delegating ctor pair taking `dflash_options`, a
>   `void init(const dflash_options & opt)` seam, and members `{features_buf, smpls, backend_chains,
>   n_embd_dec/enc/tgt, block_size, causal_attn, target_layer_ids(_n), use_dev_features}` — 388 lines.
> * the delivery (block 12): one ctor doing everything inline, plus `is_dspark`, `sample_from_anchor`,
>   `is_dflash2`, `is_mrope`, `strict_pos`, `selector_top_k`, `dflash2_rng`/`dflash2_seed` (probabilistic
>   drafting), `ingest_window`/`ingest_from` (the prompt-tail ingest), the `lilicorr_*` state, and methods
>   (`begin`, `set_prompt_end`, the tail-filtered `draft`) that upstream's version does not have — 620 lines.

**The intended merge shape** (decided from the code, to be validated by building + the MTP four-axis protocol):
build on **upstream's seam** (`dflash_options` + `init(opt)`), because that is what every other speculator class in
this tree now uses (R02 resolved D5 as MERGE for exactly that reason), and add the delivery's state as fields that
`init()` sets from the options, with its `begin`/`set_prompt_end`/`draft` additions as overrides on upstream's base.
**Nothing of the delivery's feature set is dropped** — see §7, which is the important part of this document.

## §7 The DFlash decision — what the maintainer asked for, and what to measure

**Maintainer's instruction (2026-11-21):** *"the best way to make a decision on which way to go with the dflash
choice is to measure which is more performant.  I know that we spent a lot of effort on optimizing DFlash2 within
the delivery repo.  I am reticent to drop its work for upstream's."*

So: **measurement decides, and the default is to preserve the delivery's work.** Before measuring, understand
which layer the question is even about, because I checked and the layers are very different:

### Layer B — the graph builders: NOT a performance fork (already answered)

Upstream (at this base) has split the speculator graphs into per-campaign files, and their builders are the
**same code** as the delivery's:

| builder | upstream | delivery (block 12) | difference |
|---|---|---|---|
| `build_dflash2_selector` | `src/models/dflash2.cpp:120`, used at 223 | `src/models/dflash.cpp` (block 12) | **the delivery's is upstream's + one block**: the all-gather of vocab-split logits (`ggml_all_gather`) under `-sm tensor`, which is a **correctness fix** (upstream's selector aborts on a per-row op over a row that straddles a split) |
| `build_dspark_markov_head` | `src/models/dspark.cpp:180` | `src/models/dflash.cpp` | **byte-identical** (7,037 bytes each, verified by diff at R12) |
| `build_lilicorr_scores` | `src/models/lilicorr.cpp:82` | `src/models/dflash.cpp` | **byte-identical** (11,151 bytes each, diffed at R12) |
| `build_dflash2_conv` | a **class member** (`dflash2.cpp:53`) | a static free function | structural only |

**Consequence for the measurement plan:** an A/B of "delivery builders vs upstream builders" would compare the same
arithmetic, and the only measurable delta is the all-gather arm, where upstream's version is not merely slower — it
**aborts**. Port the delivery's all-gather into upstream's `dflash2.cpp` (already done in the R12 WIP patch) and the
builders are a non-question: keep upstream's file layout, keep the delivery's fix.

### Layer A — the speculator class seam: structural, not a performance fork either

Every delivery feature that "we spent a lot of effort on" is a *switch or a state field* on this class, not an
alternative implementation of upstream's:

| feature | knob | measured value (delivery records — do not re-derive) |
|---|---|---|
| prompt-tail ingest (every draft layer has a sliding window) | `LLAMA_DFLASH_INGEST_TAIL` (on) | prefill **+8–12 %** at 6K/26K/88K (27B UD-Q4_K_XL + DFlash2-Q4_K_M, 3× gfx1201 `-sm tensor -mg 0`, `-fa on -ctk/ctv q8_0`: 88.6K 1271 → 1395 t/s, draft share of prefill 10.2 % → 1.4 %); 35B-A3B Q4_K_M + its DFlash2 **+14.6 %** at 6.1K; text and draft counts byte-identical |
| device-resident layer-input features (F1) | `LLAMA_DFLASH_DEV` (on) | `ENVIRONMENT.md` §LLAMA_DFLASH_DEV; since r163 the target graph writes them with an in-graph `SET_ROWS` (188.8 MiB of compute buffer saved at `-ub 2048` on a 27B) |
| probabilistic drafting (per-seq rng + seed) | state: `dflash2_rng` / `dflash2_seed` | expressiveness, not a rate |
| anchor-first sampling (DSpark) | state: `sample_from_anchor` | DSpark correctness |
| DFlash2 itself vs the alternatives (the reason any of this exists) | `--spec-type draft-dflash` | 27B UD-Q4_K_XL, 2 dev `-sm tensor -c 16384` q8_0 KV, `-n 2000`, 4 prompts: plain **48.6**, `draft-mtp` n5 **131.2**, adaptive n5 126.8, **`draft-dflash` n5 144.3, n7 153.1** (mean over R/P/C/K) — DFlash2 wins by ~20 % over the best adaptive-MTP arm (`archive/work/qwen4exp-dflash2/README.md`) |

**Consequence:** dropping the delivery's class for upstream's deletes the ingest-tail (~+10 % prefill), F1, the
probabilistic drafting state and the DSpark anchor path — i.e. exactly the work the maintainer does not want to
lose. The seam merge is therefore **structural**: implement the delivery's features on upstream's `init(opt)` base.

### The measurement that IS worth running (and its arms)

Once R12 builds, this is the honest A/B — *the delivery's feature set against upstream's behaviour on the same
targets*, plus a per-feature isolation so the difference is attributed rather than assumed:

**Targets** (this box has the heads):

| target | DFlash2 head |
|---|---|
| `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` | `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` |
| `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` | `/llm/models/Qwen3.6/35B-A3B/DFlash2/Q8_0/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` |
| `/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `…/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` |

**Arms** (one run per cell, `--seed 42 --temp 0`, `HIP_VISIBLE_DEVICES` pinned, one GPU workload at a time):

| # | what it answers | command sketch |
|---|---|---|
| U | upstream behaviour: delivery features off | `--spec-type draft-dflash … LLAMA_DFLASH_INGEST_TAIL=0 LLAMA_DFLASH_DEV=0` |
| D | the delivery's feature set (default) | `--spec-type draft-dflash …` (both env vars default on) |
| D-it | ingest-tail's own contribution | `LLAMA_DFLASH_INGEST_TAIL=0` only |
| D-f1 | F1's own contribution | `LLAMA_DFLASH_DEV=0` only |
| U-st | **correctness gate for `-sm tensor`** | arm U with `-sm tensor`: upstream's selector must be shown to abort (or the all-gather must be present) — this is the one place a rate comparison is meaningless |

Prefill and decode are separate claims: measure **pp** at 6K and 26K (the ingest-tail's win is a *prefill* win) and
**tg** with `-n 2000` (DFlash2's win is a *decode* win), plus the **acceptance** counts (accepted/generated, mean
block length, per-position rates) from `-lv 4`.

**Pass criteria, in order:** (1) where the feature sets match, text **and** acceptance counts are identical; (2) the
HARD correctness arms hold — upstream's `-sm tensor` selector abort is the proof that the all-gather must stay, and
no arm may corrupt (`////`); (3) **the faster arm wins per metric** (prefill separately from decode), per the
maintainer's standing "per knob, not per file" rule.  A mixed result is a fine outcome: e.g. keep upstream's file
layout and the delivery's builders + all-gather, with the delivery's features on upstream's seam.

Record the numbers in `benchmarks/` and the choice in `RUNGS.md` R12 — **the maintainer's reluctance is the default
direction**, so the measurement has to *show* a loss to justify dropping anything.

## §8 Open obligations carried into the next session (do not lose these)

| # | item | state |
|---|---|---|
| **R10.3 — RELEASE BLOCKER** | the host-expert arena arm's decode: 35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, `MOE_EXPERT_HEATMAP=load` → **R09 54.4 t/s, R10 20.4 t/s**, reproducible, text unchanged, and **no switch recovers it** (`DISABLE_FUSION=1` 18.5, `SLAB=0` 13.8; r327 reads 49.7) | maintainer, 2026-11-21: *"the decode regression must be fixed before we release; that can happen later."* Re-check at **R15**; if still open, bisect from `pool_stack` / `mmb_yield_caches` / the MMB slot lifetime. **r328 does not ship with it open.** |
| **R11.2** | `gate-hc-fusions.sh` **RED**: `mix_op`, `dsv4_pre_op`, `ops_no_post` left their r327 anchors (comm_op is exact; the chain itself moved). Half was predicted by block 11's own commit message; the fused **mixer** arm moving is new | re-run at **R13/R14/R17**; if `mix_op` still differs, the fused mixer must not ship default-on. Re-anchoring planted values is a **maintainer decision**, arm by arm |
| **R06.2 (routed half)** | the forced routed MMB arm (`GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1`, 35B-A3B) still aborts on the pool LIFO assert where r327 reads `86b7f9b5aa80` | owed at **R15**/R18; it is an A/B override, not a shipped default (RDNA4: `routed=0`, Q8_0 not in the mask) |
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
| the paused R12 | `r12-wip/README.md` + `r12-wip/r12-resolved-but-unbuildable.diff` |
| the delivery's DFlash2 record | `archive/work/qwen4exp-dflash2/README.md`, `archive/work/dflash2-draft-sched2/README.md` |
| env knobs (incl. `LLAMA_DFLASH_*`) | `~/llama-cpp-rdna-boosts/ENVIRONMENT.md` |
| delivery policy | `~/llama-cpp-rdna-boosts/AGENTS.md` |
| engineering facts / "do not re-derive" | `~/llama-cpp-rdna-boosts/INVARIANTS.md` |
| MTP/DFlash validation protocol | `~/llama-cpp-rdna-boosts/benchmarks/mtp-adaptive-methodology.md` |
| prompts (hash-stable) | `~/llama-cpp-rdna-boosts/prompts/` — `prose-rdna-boosts.txt` is the 5298-token one used above |
| the r327 binaries | `~/llama.cpp/build-rocm/bin-r327` |
