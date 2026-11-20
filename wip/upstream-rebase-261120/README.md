# Upstream re-base campaign — 18 blocks from `4f9dfa0e3` onto `919794500`

**Status: OPEN — scaffolded 2026-11-20. Nothing is delivery yet. Zero rungs climbed.**
`patches/`, `release.json`, `rdna-boosts-all.patch` and the fork's `rdna-boosts` branch are untouched until
the final rung (R18).  Target: the next release after `v17-4f9dfa0e3-r327`, i.e. **r328**, fork point
**`919794500`**.

This is a **ladder climb, not a single rebase**: one rung per delivery block, each rung ending in a resolved
commit, a green build and a coherence run, so a session can stop after any rung and the next one resumes with
`git cherry-pick <next block>`.  A rung that reaches a **critical decision point** (upstream has implemented,
or is about to subsume, something the delivery owns) **stops and asks the maintainer** before any delivery
code is dropped — see [`DECISIONS.md`](DECISIONS.md).

The method, the exit criteria and the traps are the ones proved by
[`../../archive/work/upstream-rebase-261113/`](../../archive/work/upstream-rebase-261113/README.md) (the r278
re-base, 17 blocks from `a691b2345` onto `4f9dfa0e3`) and `../../archive/work/upstream-rebase-261108/` (the
r236 re-base).  Read the r278 record's §1 before starting a session; nothing in it has stopped being true.

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

## §4  Open questions for the maintainer (asked 2026-11-20, before R00)

- **D1 (#30542 VMM revert)** — the standing decision at r278 was "carry the revert in block 00 until upstream
  fixes it; drop it at the next re-base".  Upstream has **not** fixed it (`ggml-cuda.cu:709-712` still has the
  reserve-and-halve retry), so the plan is to carry it again.  Confirm.
- **D2 (#32505 duplicate expert ids)** — upstream landed the feature the delivery was carrying as an
  upstream-PR port in block 00.  Recommendation: **take upstream's, drop ours**.  Needs the maintainer's word
  because it deletes delivery code.
- The other subsumption candidates (D3-D9 in `DECISIONS.md`) are stated now and resolved at their rung.
