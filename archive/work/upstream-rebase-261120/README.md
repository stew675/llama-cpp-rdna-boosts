# Upstream re-base campaign — 18 blocks from `4f9dfa0e3` onto `919794500`

> ## CLOSED — 2026-11-21 (archived after the r329 follow-up session).  Every obligation this campaign carried is
> closed; the two residuals it surfaced are now tracked in `TODO.md`, not here.
>
> | obligation (`COLD-START.md` §8 / `RUNGS.md` §2) | final state |
> |---|---|
> | the 18 rungs, R00-R17 | **DONE** — one rung per block on `rebase-r328`, `gate-per-block-build.sh --all` **18/18**, coherence `1c5d32ac537d` at every rung |
> | the three rung repairs | **squashed** into block 00 / 05 / 03 at R18 (§R18.1, final tree unchanged) |
> | R11.2 (the "moved HC chain") | **REFUTED** (§R18.4): r327's own binaries read the r328 numbers byte for byte; the gate's anchors had been stale since r309..r319 and its reference arm was unanchored. `gate-hc-fusions` now anchors the chain too |
> | `mm_ids_helper_512_10` re-measure | **DONE at R00**: **+0.7 %** prefill at 5298 tokens with identical text — the recorded +4.5 %/+5.2 % is superseded, and block 00's note says so |
> | R10.3 (the host-expert arena collapse) | **CLOSED at R15**: median 49.9 t/s against r327's 49.7, text `86b7f9b5aa80` |
> | R06.2 (the forced routed-MMB abort) | **CLOSED 2026-11-21**: the r328 arm exits 0 with the `GGML_CUDA_MMB=0` control hash `86b7f9b5aa80` and `MMB_GLU` proven reached; the abort belonged to the intermediate ladder tree (blocks 06+09 without block 10's `pool_stack`).  `WORKLOG.md` 261121.007 |
> | block 15's APU gates | **CLOSED 2026-11-21** on `halo`: `gate-uma-experts` **49/49**, `gate-prefill-server-slots` **3/0** at `-lme 85` — both had been run against the campaign tree rather than the delivery build (`INVARIANTS.md` r329) |
> | gfx1151 + gfx1100 arch validation | **DONE** — `benchmarks/2026-11-21-r328-arch-validation.md` (apply 18/18 + tree/tip match on both boxes, op matrix green, gfx1100 `gate-spec-align` and `gate-hc-fusions` PASS) |
> | `gate-node-budget` (phi-4) | **CLOSED 2026-11-21**: phi-4 IQ4_XS fetched, four arms `a1b7eb2ee7a6` |
> | the R18 gate sweep | **DONE** — ten block-owned gates green on the final tree (`WORKLOG.md` 261121.005), including `gate-spec-align`, `gate-spec-ckpt-device`, `gate-spec-width-band`, `gate-norm-rows-alias`, `gate-park-idle-slots`, the quant-coherence + QSA section E |
> | the prefill-logit base on the final tree | **DONE** — `--record` re-pinned at R18 (`build=73a74454b`), KLD **0.000000 / 100.000 %** |
> | `TODO` #352 (the container `web` stage) | **CLOSED** — the worker published all six r328 images with `ui/apps/chat` built |
> | generate the docs / assert no drift | **NOT DONE — now `TODO` #358.**  This is the one obligation that was skipped, and it is a real (docs-only) delivery defect: `llama-gen-docs` rewrites three committed files, because everything block 06/09/12/14/15/16/0300 added to `common/arg.cpp` is missing from them.  Fix rides **block 16** on the next release, with a `validate-set.sh` check so it cannot silently recur |
> | `-Wdeprecated-declarations` in two test TUs | **now `TODO` #359** — cosmetic, recorded here as "a TODO" but never given a handle, so it would have been lost with this directory |
>
> The release itself: **`v17-919794500-r328`** — `RUNGS.md` §R18.8, `WORKLOG.md` 261121.005.  The follow-ups that
> closed the rungs' owed list: `WORKLOG.md` 261121.007 (`TODO` #352/#354/#356/#357 retired).

**Status: RELEASED — `v17-919794500-r328`** (tip `6d9d54238`, tree `0c46248f7371a0eee84a980a1e16ad25a6fbd38f`; ladder branch `rebase-r328` = `73a74454b`).  R00-R17 landed, R18 cut the set.  What is left is the **gfx1151 + gfx1100 validation** (`TODO` #354, [`REMOTE-VALIDATION.md`](REMOTE-VALIDATION.md)) — the ladder itself is closed.
`patches/`, `release.json`, `rdna-boosts-all.patch` and the fork's `rdna-boosts` branch are untouched until
the final rung (R18).  Ladder branch `rebase-r328` = `73a74454b` (tree
`0c46248f7371a0eee84a980a1e16ad25a6fbd38f`): all 18 blocks, the three rung repairs squashed into their owning
blocks at R18 (`RUNGS.md` §R18.1, final tree unchanged); every rung is green on build + coherence and each has
one entry in [`RUNGS.md`](RUNGS.md).
**The one known-red is closed at R00** (repair `3ea8b0212`): the delivery's required VMM pool refuses
upstream's strided 2-D indexer top-k copy, and it reproduces on the pristine base with *only* the `#30542`
revert, so the fix (block 10's `top_k_gather_rows`, moved down — [`DECISIONS.md`](DECISIONS.md) D15) lives in
block 00.  Flash-Next now prefills at 1692 t/s, and `mm_ids_helper_512_10`'s A/B reads **+0.7 %** with
identical text (the recorded +4.5 %/+5.2 % is re-measured at R18 before it is quoted again).  R02 is done: its 5
conflicts were all the DFlash spec refactor and every one was a union, so `D5` resolved as MERGE with the
adaptive arm at 29.3 -> 65.2 t/s on identical text.  R03 is done too: one test-file
conflict (a union of upstream's new `padded_q`/`permuted_k` layouts with the delivery's `n_rs_batch`), and the
block's switch moves its own measurement -- chunked prefill +8.8 % with identical text.  R04 is done: `D3` resolved by
unioning seven rope arms and dropping upstream's blanket guard (which the delivery's own kernels invalidate),
worth +5.8 % prefill / +16.5 % decode at bf16 KV with identical text.  R05 is done: `D16` decided the delivery's FA
iteration model wins over upstream's `#32407` rotation (verified absent, FA matrix 7428/0, native q8_0 K/V
+5.6 % decode).  R06 is done too: both conflicts were unions (upstream's few-row MMQ keeps precedence because
it is compiled out under HIP, and MMB follows it; block 06's `expert_bounds` re-ported onto the R00 store
helper), coherence unchanged, MUL_MAT_ID 1078/0, MMB +0.33 %/+0.62 % and the F32 skinny GEMM +0.9 %/+1.0 % —
and one **finding**: block 06's MMB activation cache/slot buffers are freed in an order the VMM pool cannot
accept, so a default-on RDNA4 *dense* type (IQ3_S) aborts until **block 10's `pool_stack`** lands (r327 is
clean, this intermediate tree is not — `RUNGS.md` R06.2, with the R10/R18 obligations it creates).  R07 is done
too: both conflicts were in `mmvq.cu` and both unions (upstream's Jetson-`ORIN` id + the delivery's
RDNA2-`NAVI21` id, and the two `calc_nwarps` special cases side by side), the RDNA3_5 `ne11` gate is adopted as a
dispatch threshold rather than a competitor to the delivery's VDR tables, and D11's knobs came out "mixed, per
knob" as predicted: the grid-stall pad is +1.37 % pp2048 with identical text, while capping the MMQ tile width
(`MMQ_J_MAX=64`) costs 15.3 % — so the delivery's own cap stays at its default.  R08 is done, and it was the
climb's worst rung: 8 conflicted files / 13 regions, **D4 resolved as MERGE without a choice** (the delivery's
narrow-row matcher already sits ~1000 lines ahead of upstream's generic norm branches, and `#32454` adds the
layer-norm variant the delivery never claims), plus four forced fixes — including an **auto-merge artifact**
where upstream's `force_tiling=true` landed in the delivery's `has_gate` template slot and would have
mis-dispatched silently (`RUNGS.md` R08.3).  R09 is done too: the host-expert tier's 70 files merged with only 4
conflicted paths, **D8** and **D9** both resolved as merges without a choice, and two of the delivery's extracted
functions had to be re-ported with upstream's later refinements carried in (`cc < GGML_CUDA_CC_VOLTA`, the
few-row-MMQ early-out) — but it carries a **finding**: with the arena off, this tree reads different text from the
arena arm where r327 reads one text on both, and the `GGML_MOE_HOST_OFFLOAD=0` control pins the cause on block
09's own host-offload relaxation (`RUNGS.md` R09.3, owed at R10/R18).  R10 is done, and it was the rung that owns
the most carried obligations: the tree keeps **one** fusion dispatcher (`ggml-cuda-try-fuse.cu`, reconciled
forward to this base — upstream's two `#32454` NORM+MUL arms were the only missing ones, and they are D4's arm),
the `top_k_gather_rows` hunk de-duplicated by construction, **R06.2's dense half closed** (27B UD-IQ3_S now
prefills and reads r327's exact hash) and **R09.3 closed** (arena and arena-off now read one text).  It also
produced a new finding: the arena arm's decode dropped 54.4 -> 20.4 t/s and no kill switch recovers it
(`RUNGS.md` R10.3, owned by blocks 14/15) and is a **release blocker**.  R11 is done: the qwen4exp block merged with
5 conflicts, D9 and D13 were carried (upstream's FP8 PLE scale *and* the delivery's lazy reader coexist), and both
block gates ran — quant-coherence PASS, but **`gate-hc-fusions.sh` is RED**: the fused *mixer* arm and the two
dsv4 pre/post arms left their r327 anchors while the combine arm is exact (`RUNGS.md` R11.2, owed at R13/R14/R17
and needing a maintainer call on re-anchoring).  **R12 is done** (`a9d8e1a96`): the paused session's ten resolutions were applied, and
`common_speculative_impl_draft_dflash` — upstream's `#32016` `dflash_options`/`init()` seam against block 12's
inline ctor plus the DSpark/DFlash2/LiliCorr/ingest state — was merged by keeping upstream's seam and class
split and implementing the delivery's features on it (the merged class differs from upstream by exactly the
delivery's 207 lines).  Two features the splice had orphaned in `tools/server/` were restored (the parked-slot
restore retry had no caller; `server_prompt_cache::load()` read locals that had moved into `find()`), and the
DFlash question was settled by measurement in the maintainer's direction — **D17: the delivery's feature set
stays** (ingest tail +29.8 % pp at 26K on a 2-GPU 35B, +11.7 % on 3-GPU Flash-Next, ~neutral on a 1-GPU
`-sm layer` cell; F1 +6-7 % pp on the dense 27B and inert on both MoE heads; text and acceptance identical in
every pair; the pristine base still aborts DFlash under `-sm tensor`).  Numbers:
[`../../benchmarks/2026-11-20-r328-r12-dflash-merge.md`](../../benchmarks/2026-11-20-r328-r12-dflash-merge.md).
Six gates green.  R13 is done as well (`81523fda6`): the MTP draft vocabulary subset applied clean,
its own test passes (subset logits == the full-vocab reference, six cases) and `--spec-draft-mtp-vocab 65536`
is +7.6 % decode at identical text.  R14 landed (`9f8c11782`): the streamed KV, with the one real trap of the
climb — `GGML_OP_COUNT` grew to **109** while every `static_assert` was merge *context*, so it had to be recounted
(patch 10); `gate-kv-residency` and `gate-spec-draft-window` are green, and `gate-node-budget` is owed (no phi-4 on
this box).  R15 landed (`07b293a7b`): the UMA GTT expert store, and with it the climb's **release blocker is
closed — R10.3's arena decode reads 49.9 t/s median (r327: 49.7) with the text unchanged**; `gate-host-pool.sh` is
**59/59**, including the `-sm tensor` arms owed from R12, and the GTT gate itself turned out to be an **APU gate**
(on gfx1201 the tier is inert *by design*), so its claims stay owed on the `halo` box.  R16 landed
(`71ce858ca` + repair `dd4fa5e22`): the server's prompt admission policy, green on `gate-prefill-fairness` (dense,
`draft-mtp`, and a streamed host-expert shape, plus `--self-test`) and `gate-prefill-duty` (dense and
`draft-mtp`).  R17 landed (`ee502d397`): block `0300`, and **`TODO` #351 is answered — `0300` cannot be dropped**,
because flint still refuses `-sm tensor` at this base with the block's own message, while `-sm relaxed-tensor` is
byte-identical to `-sm layer` and a non-amax arch is unchanged both ways.  **One red remains: R11.2**
(`gate-hc-fusions.sh`), still reading the R11 numbers at the completed tree — a maintainer decision before
shipping.  Target: the next
release after `v17-4f9dfa0e3-r327`, i.e. **r328**, fork point
**`919794500`**.

**Two external analyses arrived 2026-11-21 and are assessed in [`EXTERNAL-ANALYSES.md`](EXTERNAL-ANALYSES.md):** (a) a
licensing obligation — block 03's two chunked-GDN kernel files are libr4d ports, and libr4d now ships inside
`radiance` under Apache-2.0 — **fixed at the repo level** (root `NOTICE` + `third_party/radiance/`), with one
one-line in-patch attribution still needing the maintainer's approval; and (b) an offered patch set
(`llama-Frankenstein`, cut against r327) mapped group by group onto what the delivery already ships — **nothing is
taken into r328**, and the three upstream overlaps it warns about (DFlash, NORM+MUL, server slot scheduling) were
already resolved at R12/R10.1/R16.

This is a **ladder climb, not a single rebase**: one rung per delivery block, each rung ending in a resolved
commit, a green build and a coherence run, so a session can stop after any rung and the next one resumes with
`git cherry-pick <next block>`.  A rung that reaches a **critical decision point** (upstream has implemented,
or is about to subsume, something the delivery owns) **stops and asks the maintainer** before any delivery
code is dropped — see [`DECISIONS.md`](DECISIONS.md).

The method, the exit criteria and the traps are the ones proved by
[`../../archive/work/upstream-rebase-261113/`](../../archive/work/upstream-rebase-261113/README.md) (the r278
re-base, 17 blocks from `a691b2345` onto `4f9dfa0e3`) and `../../archive/work/upstream-rebase-261108/` (the
r236 re-base).  Read the r278 record's §1 before starting a session; nothing in it has stopped being true.

**A fresh session starts at [`COLD-START.md`](COLD-START.md)** (self-contained state, anchors, the R18 plan,
the DFlash measurement mandate and the carried obligations); the prompt to paste is
[`NEXT-SESSION-PROMPT.md`](NEXT-SESSION-PROMPT.md).  `HANDOVER.md` is the previous sessions' delta.

## §0  Baseline facts (as measured 2026-11-20)

| item | value |
|---|---|
| old fork point | `4f9dfa0e3` — 2026-11-13 17:50 +0100 (`requirements, Fix CI: support Python 3.15 (#32199)`) |
| new fork point | **`919794500`** — 2026-11-20 20:54 +0100 (`convert: qwen3-asr: put "language ..." output into thinking block (#32571)`), the maintainer's snapshot of upstream `master` in `~/llama.cpp` |
| upstream drift | **228 commits** (0 merges, 1 revert), **1944 files**, **+147 254 / −41 182** |
| delivery tip (old) | `e159adfc2`, tree `4792f98b8`, base tree `cfb1f99a2` (`release.json`, r327) |
| delivery shape | 18 block commits (17 official `00`-`16` + the temporary `0300`), **557 distinct files**; **96** of them upstream also touched, 461 delivery-only |
| probe result | replaying the 18 blocks onto `919794500` leaves **61 conflicted paths across 12 blocks**; blocks **01, 13, 15, 16 and 0300 apply clean**; worst are 08 (9), 10 (9), 12 (9), 05 (7) — see [`CONFLICT-MAP.md`](CONFLICT-MAP.md) |
| semantic pressure | every block shares files with upstream; the rungs that carry **subsumption** questions are 00, 04, 06, 07, 08, 09, 16 — see [`DECISIONS.md`](DECISIONS.md) |
| working branch | **`rebase-r328`** in `~/llama.cpp`, created at `919794500`; **`master` there stays the pristine snapshot — never commit to it** |
| r327 reference binary | `~/llama.cpp/build-rocm/bin-r327` (a copy of the r327 `bin/`, `build 12387, commit 163eb6cd8`) |
| pristine-base build | `~/llama.cpp/build-base` (`BUILD_DIR=build-base ~/bin/build-llama-rocm`) — the upstream-only oracle, never the ladder |
| canonical build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm/`, `GGML_HIP_NO_VMM=OFF`, ccache 30 GiB) |
| models | `/llm/models/Qwen3.5/4B/Q8_0/…`, `/llm/models/Qwen3.6/35B-A3B/Q8_0/…` (`runme-rocm` carries the host-expert config) |

This drift is **bigger than the r278 one in every dimension**: 228 commits against 164, 1944 changed files
against 751, +147 k lines against +40 k.  Two upstream events dominate the risk and neither is a text
conflict:

1. **`llama_batch_ext`** (`62a6d3a30`, deprecating the old batch API in `a06044b93`) rewrites the batch
   surface that blocks 02, 12 and 16 sit on.
2. **Upstream is now developing *in* the delivery's subsystems**: a MoE cache (`--moe-cache-cpu-share`),
   duplicate expert ids, few-row MMQ for `MUL_MAT_ID`, a NORM+MUL(+ADD) fusion, BF16 ROPE/SET_ROWS, an
   indexer key-rotation virtual, a graph-node budget moved into the model classes, a server `slot_sched`
   refactor and a DFlash spec refactor.  Each of those is a decision, not a merge.

## §1  Method

**Replay one block commit at a time onto the new base.**  `git rebase --onto` would do the same merges but
leaves an un-committed rebase state that is fragile across sessions; a cherry-pick per rung leaves a
committed, buildable branch at every stop.

```bash
cd ~/llama.cpp
git status                                  # must be clean, on rebase-r328
git cherry-pick <block sha>                 # rung Rnn  (shas in RUNGS.md / release.json order)
# conflict:
git status                                  # the conflicted paths are the rung's work
#   resolve; read both sides; never blanket -Xours / -Xtheirs
git add -A && GIT_EDITOR=true git cherry-pick --continue
~/bin/build-llama-rocm                      # incremental, all targets
```

Then the rung's **exit criteria** (recorded per rung in [`RUNGS.md`](RUNGS.md)):

1. **Every resolution is written down** in that rung's log: what upstream changed, what we kept, why.  A hunk
   that resolves *naturally* because upstream fixed the same thing is recorded as "upstream's, ours dropped"
   — that is a decision, not a silent delete.
2. **Build green** (`~/bin/build-llama-rocm`, no new warnings in touch-path files).
3. **Coherence**: same-seed greedy text on `Qwen3.5/4B/Q8_0` (`-ngl 99`, `-sm layer`, one GPU,
   `HIP_VISIBLE_DEVICES=0`, `--seed 42 --temp 0 --no-display-prompt --single-turn -n 20`), hashed with
   `scripts/extract-generated.py`; plus the block's own mechanism line where it has one (`-lv 4`, a census
   arm — see the `-lv` table in `AGENTS.md`).
4. **No leftover markers / no stub resolutions**: `git grep -n '<<<<<<<'` empty; any `TODO(rebase)` is listed
   in the rung log, not left in the code.
5. `git log -1 --format=%h` recorded in `RUNGS.md`.

**The text reference is NOT r327.**  Upstream moved numerics on purpose in this range too (`7fce5439f` bf16
rope/set_rows, `b94959a64` the MMID rework, `03ec5b432` the norm fusion, `054a1babf` KV read order,
`2cbfc3cf3` accumulation gating, …), so r327-identical greedy text is the **wrong oracle for a rung**.  A rung
is judged by: build + coherence + the block's own kill-switch A/B (a delivery feature that is default-on must
still move its own measurement) and, where the block owns a gate, "the gate reaches the code it is supposed
to reach".  r327-identical text is only used as a *smoke* check when a block has no kill switch and touches no
numerics; a difference there is recorded, not "fixed".

### Traps that already exist here (carried from the r278 climb; all still live)

- **Never commit to `master` in `~/llama.cpp`** — it is the pristine new-base snapshot (and a bare `git push`
  there targets upstream; see `AGENTS.md`).
- **One long-lived source path.**  The replay happens in `~/llama.cpp` because that is where `ccache` is warm.
  Do not make a rebase worktree at a new path: the first build there is a full cold compile.
- **Do not run the per-block gate or a container build beside a benchmark** (and never in parallel with
  another benchmark).
- **A rung is a cherry-pick, not a re-cut.**  Do not regenerate `patches/` mid-climb; `make-patches.sh` runs
  once, at R18, from the finished branch.
- **`-sm tensor` needs `HIP_VISIBLE_DEVICES` pinned** (the iGPU is otherwise enumerated and the load aborts
  with `invalid kernel file`).
- **A rung build may need a clean configure** if a block changes the CMake surface; a reused
  `CMakeCache.txt` carries the *tip's* options.  `GGML_CUDA_FA_QUANTS` is set by hand on this box and is the
  live example (the `iq4_nl-iq4_nl` instance belongs to block 05).
- **Build with `GGML_HIP_NO_VMM=OFF`** — the delivery's host-expert tier needs VMM and upstream's HIP default
  is `ON`.  `~/bin/build-llama-rocm` passes it.
- **A rung's diff may shrink, and that is the point.**  Judge a rung by its resolutions, not its line count.
- **Report before deciding.**  Any rung that reaches a [`DECISIONS.md`](DECISIONS.md) item stops there:
  present the measurement and the options to the maintainer and wait.  Never drop delivery code because
  "upstream's looked newer" (see `AGENTS.md` → report findings).

### 2026-11-20 tooling deltas found while laying this ladder out

- **The web UI moved `tools/ui/` -> `ui/`** (`3a7660371`, then `ui/apps/chat`, `ui/lib/ui`).  Upstream's
  `scripts/ui-assets.cmake` now resolves `SRC_DIST_DIR = ${UI_SOURCE_DIR}/dist` with `UI_SOURCE_DIR = ui`,
  so an untracked `tools/ui/dist` in the checkout is **no longer Priority-1 provisioned**.  A ladder rung
  whose build predates the delivery's own UI asset step will therefore fetch from HF.  `scripts/ui-dist-cache.sh`
  (`TODO` #342) must learn the new source path before the **R18** ladder, otherwise the per-block gate pays
  ~18 downloads again; a mid-climb rung build is unaffected as long as the bundle is cached in its build dir.
- **`scripts/gate-per-block-build.sh --fast` is the default** (`--jobs 28`) and `--all` is only for a re-base
  or a `tests/`-TU / build-graph change (`AGENTS.md` → Build the fork).  A re-base ladder wants `--all` at
  the *tip* only; `--fast` is the per-rung default.
- **The reference binary must be captured before the first rebuild of `build-rocm`** — done
  (`build-rocm/bin-r327`).

## §2  Files in this campaign

| file | what |
|---|---|
| `README.md` | this handover: facts, method, traps, state |
| [`CONFLICT-MAP.md`](CONFLICT-MAP.md) | the **measured** conflict map (per block: text conflicts, upstream churn on the block's files, the upstream commits that own the collision) |
| [`RUNGS.md`](RUNGS.md) | the ladder: one entry per rung, status, resolutions, exit criteria, the commit id |
| [`DECISIONS.md`](DECISIONS.md) | the **critical decision points**: where upstream subsumes or re-implements a delivery feature.  Each entry is a question for the maintainer with the evidence and the options |
| `probe-conflicts.py` | regenerates `conflict-map.json`: replays the blocks onto the new base with `git merge-tree` (no worktree touched) |
| `probe-overlap.py` | regenerates `overlap-map.json`: per-block upstream churn + the upstream commits whose subject enters a delivery space |
| `conflict-map.json`, `overlap-map.json` | the machine-readable probe output |

## §3  Session protocol (how to resume)

1. Read this file's §0 and §1, then the **last filled rung** in `RUNGS.md` and any `OPEN` item in
   `DECISIONS.md` that names your next rung.
2. `cd ~/llama.cpp && git status` — must be clean, on `rebase-r328`.  If a cherry-pick is mid-flight, the
   previous session stopped inside a conflict: finish or `git cherry-pick --abort` and redo that rung.
3. Climb the next rung, meet the exit criteria, append the rung's log **before** committing the next one.
4. If the rung reaches an `OPEN` decision: **stop, present, wait.**  Record the outcome in `DECISIONS.md`
   with the date and the decision, then continue.
5. `WORKLOG.md` gets **one** entry per *session* only when something delivery-affecting lands (an amendment,
   a promotion, the release).  The rung log lives here, not in `WORKLOG.md`.

## §4  Answers received (2026-11-20) and what changed in the tree as a result

- **D1 — decided: take upstream's `#32505`, drop block 00's carried port.**  The
  `upstream/mul-mat-id-dup/` candidate is **retired** (moved to
  `../../archive/work/upstream-mul-mat-id-dup-merged/`, index row removed), block 00 no longer carries it, and
  the delivery's own `mm_ids_helper_512_10` fast path must be re-ported onto upstream's new helper signature at
  R00.  `DECISIONS.md` D1 has the details.
- **D2 — decided: carry the `#30542` revert again, "for now".**  The re-test that would let it go is tracked as
  **`TODO` #351**: *does upstream now properly support `-sm tensor` for `flint`?* — the `0300`
  (`-sm relaxed-tensor`) drop condition, tested at **R17** with the revert in place and the real model this box
  has.  `DECISIONS.md` D2 records that the two mechanisms are separate (the revert is about the VMM pool the
  host-expert arena needs).
- **D11 — the maintainer wants both AMD MMQ/MMVQ tunings A/B'd and the better of each picked**, not assumed:
  a concrete measurement plan (per knob, not per file) is in `DECISIONS.md` D11, run at R06/R07/R08.
- **D14 — the HF-UI asset hack is retired and replaced by "no UI built where none is shipped".**  Done
  2026-11-20, before R00: `scripts/ui-dist-cache.sh` deleted, `~/bin/build-llama-rocm` + both gate scripts
  configure with `-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF` (`UI=1` opts back in), the machine's
  `~/.cache/rdna-boosts/ui-dist` and the checkout's `tools/ui/dist` removed, `TODO` #342 closed
  (`WORKLOG.md` 261120.006).  The container `web` stage move is `TODO` #352 at R18.  This work is valid on
  **both** bases (`LLAMA_USE_PREBUILT_UI` exists at `4f9dfa0e3`), so it did not have to wait for the climb.
