# Upstream re-base campaign — 17 blocks from `a691b2345` onto `4f9dfa0e3`

**Status: OPEN — R00..R08 done 2026-11-13 (blocks 00-07); R09 (block 08, fused kernels) is next.**  `D17`
(the base blocker) is decided: block 00 carries the `#30542` revert.  Rungs R01..R17 are the replay; R18 is
the release ladder.  Nothing here is delivery: `patches/`, `release.json` and the fork are untouched until
R18.  Target release is the next one after r277 (r278 unless an amendment intervenes) and the next fork
point is **`4f9dfa0e3`**.

This campaign is a **ladder climb**, not a single rebase: one rung per delivery block, each rung ending in
a resolved commit, a green build and a coherence run, so a session can stop after any rung and the next one
resumes with `git cherry-pick <next block>`.  The rungs are in [`RUNGS.md`](RUNGS.md), the "upstream may
subsume us" questions in [`DECISIONS.md`](DECISIONS.md), and the measured conflict map (the evidence the
ladder was laid out from) in [`CONFLICT-MAP.md`](CONFLICT-MAP.md) — regenerate it with
[`probe-conflicts.py`](probe-conflicts.py) (machine-readable output: `conflict-map.json`).

**Read [`BASE-BLOCKER.md`](BASE-BLOCKER.md) before starting a session.**  Upstream `ac931b5bc`
(`#30542`) breaks the HIP VMM pool on this box: the unpatched new base does not return from its first GPU
compute with the delivery's required `GGML_HIP_NO_VMM=OFF` build.  The ladder branch carries a candidate
revert folded into block 00 and it is `AWAITING` the maintainer's decision.

This campaign is a continuation of the method proved by
[`../../archive/work/upstream-rebase-261108/`](../../archive/work/upstream-rebase-261108/README.md) (the
r236 re-base); its traps and exit criteria are carried forward verbatim below because they still apply.

## §0  Baseline facts (as measured 2026-11-13)

| item | value |
|---|---|
| old fork point | `a691b2345` — 2026-11-08 11:50 +0200 (`webgpu : add COUNT_EQUAL support (#31578)`) |
| new fork point | **`4f9dfa0e3`** — 2026-11-13 17:50 +0100 (`requirements, Fix CI: support Python 3.15 (#32199)`), the maintainer's snapshot of upstream `master` in `~/llama.cpp` |
| upstream drift | **164 commits**, **751 files**, +40 612 / −24 226 |
| delivery tip (old) | `9d7c0502c`, tree `0104361a0`, base tree `9d7b9a02f` (`release.json`, r277) |
| delivery shape | 17 blocks (00 + 01-16), 506 distinct files; **113** of them upstream also touched |
| probe result | replaying the 17 blocks onto `4f9dfa0e3` leaves **55 conflicted paths**; blocks **01, 04, 13, 14 and 16 apply clean**; worst are 10 (10), 05 (8), 09 (8) — see `CONFLICT-MAP.md` |
| working branch | **`rebase-r278`** in `~/llama.cpp`, created at `4f9dfa0e3`; **`master` there stays the pristine snapshot — never commit to it** |
| r277 reference binary | `~/llama.cpp/build-rocm/bin-r277` (a copy of the r277 `bin/`, version line `build 12222, commit 9d7c0502c`) |
| canonical build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm/`, `GGML_HIP_NO_VMM=OFF`, ccache 30 GiB) |
| models | `/llm/models/Qwen3.5/4B/Q8_0/…`, `/llm/models/Qwen3.6/35B-A3B/Q8_0/…` (`runme-rocm` carries the host-expert config) |

## §1  Method

**Replay one block commit at a time onto the new base.**  `git rebase --onto` would do the same merges but
leaves an un-committed rebase state that is fragile across sessions; a cherry-pick per rung leaves a
committed, buildable branch at every stop.

```bash
cd ~/llama.cpp
git status                                  # must be clean, on rebase-r278
git cherry-pick <block sha>                 # rung Rnn  (shas in RUNGS.md / release.json order)
# conflict:
git status                                  # the conflicted paths are the rung's work
#   resolve; read both sides; never blanket -Xours / -Xtheirs
git add -A && GIT_EDITOR=true git cherry-pick --continue
~/bin/build-llama-rocm                      # incremental, all targets
```

Then the rung's **exit criteria** (each rung records them in `RUNGS.md`):

1. **Every resolution is written down** in that rung's log line: what upstream changed, what we kept, why.
   A hunk that resolves *naturally* because upstream fixed the same thing is recorded as "upstream's, ours
   dropped" — that is a decision, not a silent delete.
2. **Build green** (`~/bin/build-llama-rocm`, no warnings added in touch-path files).
3. **Coherence**: same-seed greedy text generated on `Qwen3.5/4B/Q8_0` (`-ngl 99`, `-sm layer`, one GPU,
   `HIP_VISIBLE_DEVICES=0`, `--seed 42 --temp 0 --no-display-prompt --single-turn -n 20`); plus the block's
   own mechanism line where it has one (`-lv 4`, a census arm — see the `-lv` table in `AGENTS.md`).
4. **No leftover markers / no stub resolutions**: `git grep -n '<<<<<<<'` empty; any `TODO(rebase)` is
   listed in the rung log, not left in the code.
5. `git log -1 --format=%h` recorded in `RUNGS.md`.

**The text reference is NOT r277.**  Upstream moved numerics on purpose in this range too (`31953bcf7` bf16
RMS_NORM, `998200491` SSM_SCAN d_state=64, `459ab1cdd` MMVF bf16 few-row dispatch, `ccd360f06` GDN graph
pruning, the FP8 KV/FA additions, …), so r277-identical greedy text is the **wrong** oracle for a rung.  A
rung is judged by: build + coherence + the block's own kill-switch A/B (a delivery feature that is
default-on must still move its own measurement) and, where the block owns a gate, "the gate reaches the
code it is supposed to reach".  r277-identical text is only used as a *smoke* check when a block has no
kill switch and touches no numerics; a difference there is recorded, not "fixed".

### Traps that already exist here (carried from r278's predecessor; all still live)

- **Never commit to `master` in `~/llama.cpp`** — it is the pristine new-base snapshot (and a bare `git push`
  there targets upstream; see `AGENTS.md`).
- **One long-lived source path.**  The replay happens in `~/llama.cpp` because that is where `ccache` is
  warm (`scripts/scratch-tree.sh`'s measured rule).  Do not make a rebase worktree at a new path: the first
  build there is a full cold compile (measured 5m56s / 85 CPU-min).
- **Do not run the per-block gate or a container build beside a benchmark** (and never in parallel with
  another benchmark).
- **A rung is a cherry-pick, not a re-cut.**  Do not regenerate `patches/` mid-climb; `make-patches.sh` runs
  once, at R18, from the finished branch.
- **`-sm tensor` needs `HIP_VISIBLE_DEVICES` pinned** (the iGPU is otherwise enumerated and the load aborts
  with `invalid kernel file`).
- **A rung build needs a CLEAN configure** (`~/bin/build-llama-rocm clean`; ~1m20s wall on warm ccache).  A
  reused `CMakeCache.txt` carries the *tip's* options, and some of them are illegal at an earlier rung — the
  live example is `GGML_CUDA_FA_QUANTS`, which holds block 05's `iq4_nl-iq4_nl` instances and makes the
  configure fail for blocks 00-04 with `unknown type "iq4_nl"`.  The `iq4_nl` instance must be re-set by
  hand when R06 lands, because that variable is set by hand on this box, not by the delivery.
- **Build with `GGML_HIP_NO_VMM=OFF`** — the delivery's host-expert tier needs VMM and upstream's HIP default
  is `ON`.  `~/bin/build-llama-rocm` passes it.
- **A rung's diff may shrink, and that is the point.**  R01 of the previous climb delivered 34 files / +986−340
  from a block of 56 files / +1366−909 because the difference was rename-only `-Wshadow` ballast.  Judge a
  rung by its resolutions, not by its line count.
- **Report before deciding.**  Any rung that reaches a `DECISIONS.md` item stops there: present the
  measurement and the options to the maintainer and wait.  Never drop delivery code because "upstream's
  looked newer" (see `AGENTS.md` → report findings).
- **New this climb:** upstream `8ca5a8008 llama : prune (#30565)` deletes `src/llama-ext.h` content and
  `29dbbad2b ggml : prune (#29801)` deletes dead ggml ops.  Both blocks that add to `src/llama-ext.h`
  (02, 10) therefore meet a **modify/delete** conflict, and a prune may delete a symbol a delivery block
  calls.  Resolve by re-adding *only the delivery's* declarations to the file's new home — never by
  restoring the pruned file wholesale.

## §2  The ladder

17 replay rungs, grouped into sessions.  Per-rung conflict counts are the 2026-11-13 probe's (an upper
bound: the probe chained through marker-carrying trees).

| session | rungs | blocks | conflicts | notes |
|---|---|---|---|---|
| 0 | R00 | — | — | **DONE 2026-11-13** — campaign, branch, `bin-r277`, conflict map, decision register |
| 1 | R01 | 00 foundation | 1 | **DONE** — resolved as an upstream-wins rename; green with the folded base fix; **`D9`**, **`D17`/B1** |
| 1 | R02 | 01 hybrid all-reduce | **0** | **DONE** — clean; hybrid all-reduce pipeline proven on 3 GPU |
| 1 | R03 | 02 adaptive MTP | 3 | **DONE** — `D7` measured (device path is `-sm layer`-only today), `D8` include lesson |
| 1 | R04 | 03 chunked GDN | 1 | **DONE** — `D2` **not** a subsume: upstream deleted its own graph builders/tags; our chunked kernels are the surviving path |
| 1 | R05 | 04 BF16 KV | **0** | **DONE** — clean; `D3` audit passed (bf16 + f8_e4m3 + q8_0 all coherent) |
| 2 | R06 | 05 flash attention | 8 (+16 modify/delete) | **DONE** — `D5` FA half complementary: union instance matrix + upstream's `DKQp`/`DVp`; 6996/0 FA oracle |
| 2 | R07 | 06 MMB prefill | 1 | **DONE** — our blocks on upstream's refactored MMVF/fusion predicates; MMB on, 6915 t/s prefill |
| 2 | R08 | 07 k-quant boosts | 2 | **DONE** — graph-reuse union + test-matrix union; MUL_MAT 2681/0 |
| 3 | R09 | 08 fused kernels | 7 | **NEXT** — analysed (conflict table in `RUNGS.md`); **`D6`** + **`D12`** (`norm.cu` dual-store merge, needs its own build+gate cycle) |
| 3 | R10 | 09 host-resident MoE | 8 | **`D1`** (upstream moved the MoE cache out of `llama-graph`), `D9`, `D15`, `D17` |
| 4 | R11 | 10 infrastructure | 10 | `D6`, `D8` (prune/`llama-ext.h`), `D10` (scheduler upload/prefetch) |
| 4 | R12 | 11 qwen4exp | 5 | `D11` (the `ggml.c`/RPC/test refactors) |
| 4 | R13 | 12 campaign remainder | 4 | `D10` (backend upload/prefetch), `D15` (external PLE) |
| 5 | R14 | 13 MTP vocab subset | **0** | `D7` verify only |
| 5 | R15 | 14 relaxed-tensor (**temporary**) | **0** | re-check it is still not subsumed |
| 5 | R16 | 15 streamed KV | 5 | `D3` (FP8 KV instance set), `D19` (recurrent cache rename) |
| 5 | R17 | 16 UMA GTT expert store | **0** | re-check the VMM pool change (`D17`) |
| 6 | R18 | — | — | release ladder: regenerate, validate, gates, `release.json`, fork refresh, records |

Rung mechanics, per-rung conflict file lists and the resume command: [`RUNGS.md`](RUNGS.md).

## §3  Decision points

Nineteen upstream/delivery questions are registered in [`DECISIONS.md`](DECISIONS.md).  The ones that can
**drop delivery code** are `D2` (GDN graph pruning), `D5` (FA restructure + FP8), `D6` (graph state move),
`D7` (MTP hidden-state device buffer), `D9` (mmap/Win-load rewrite) and `D12` (BF16 RMS_NORM); every one of
them stops for a maintainer decision with a measurement in hand.  The rest are "coexist, verify the
dispatch" (`D1`, `D3`, `D4`, `D10`, `D11`, `D13`, `D14`, `D16`, `D17`, `D18`, `D19`), "re-derive after a
refactor" (`D8`, `D15`) or "which of our wins survive" (`D18`).

**Nothing in `DECISIONS.md` is decided by this campaign alone.**  The delivery's standing principle (from
`AGENTS.md` and the maintainer): the set aims to be *complementary* — a second preferred path, not a
replacement — so most of these should resolve to "both paths ship, ours stays the default where it is
faster, upstream's stays reachable".  A drop only happens where the two cannot coexist.

## §4  End of the ladder (R18)

Not a rung in the climb — the release gate set, in the order the traps fire:

1. `scripts/make-patches.sh` from the rebased branch (17 files, block 00 keeps `0000`), then
   `scripts/make-release.sh --base 4f9dfa0e3 --tip <tip> --tree <tree>`.
2. `scripts/validate-set.sh` — strict apply on a fresh tarball at `4f9dfa0e3`, tree equality, sha256s,
   Windows path/portability lints.
3. `scripts/gate-per-block-build.sh` — a clean build per block commit (all 17).
4. Coherence: dense (`Qwen3.5-4B`), the host-expert config (`Qwen3.6-35B-A3B` `runme-rocm`), and a hybrid
   recurrent model.
5. `scripts/gate-prefill-logits.sh --record` **from a clean unpatched `4f9dfa0e3` build** (the re-base's own
   base record, same ROCm line), then the patched build against it; plus the `--sm tensor --device 0,1,2`
   arm for block 01, the `--kv q8_0` arm, and — **new this climb** — an `--kv fp8`/BF16 arm because upstream
   added FP8 KV (`#28392`).
6. MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) + `llama-batched-bench -npl 1,4,8`.
7. Every block-specific gate whose code moved: `gate-host-pool.sh`, `gate-kv-residency.sh`,
   `gate-kv-window-split.sh`, `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`,
   `gate-spec-align.sh`, `gate-spec-width-band.sh`, `gate-spec-draft-window.sh`, `gate-spec-reuse-window.sh`,
   `gate-hc-fusions.sh`, `gate-norm-rows-alias.sh`, `gate-grid-stall-pad.sh`,
   `gate-qwen4exp-quant-coherence.sh`, `gate-uma-experts.sh`, `gate-fusion-alias-copy.sh`.
8. `scripts/gate-rebase-release.sh` — the one-command pre-release verdict for a re-base (r236), then
   `scripts/check-gpu-faults.sh` on the journal + log.
9. Release: `scripts/fork-refresh.sh --build --gates --push` (personal fork only), the container images via
   the off-host worker, then the records — a `WORKLOG.md` entry with the new base/tip/tree and the gate
   numbers, `patches/README.md` per-block notes for what moved, `BASELINE.md` current baseline,
   `INVARIANTS.md` appended facts, `ENVIRONMENT.md` for any variable that appeared/vanished, `TODO.md` for
   the deferred decision items, and this campaign moved to `archive/work/`.

## §5  Resuming

**Cold start: read [`HANDOVER.md`](HANDOVER.md) first** — it is the state document (branches, rung shas, the
next rung's conflict plan, decision statuses, traps, the R18 ladder with this campaign's additions).  The
copy-paste brief for a fresh session is [`PROMPT.md`](PROMPT.md).

```bash
cd ~/llama-cpp-rdna-boosts && grep -n '^## R' wip/upstream-rebase-261113/RUNGS.md   # find the open rung
cd ~/llama.cpp && git log --oneline -3 && git status                               # confirm the branch state
```

A rung that was interrupted mid-conflict leaves `git status` dirty and `.git/CHERRY_PICK_HEAD` present —
finish or `git cherry-pick --abort` and redo it; do not leave a half-resolved rung for the next session.
A rung that is *blocked* by a regression is different: roll the branch back to the last green rung, park the
resolution as a patch in this directory, and record it.

**Every session ends by updating:** the rung's ledger row + resolution log in `RUNGS.md`, the affected
`DECISIONS.md` items, and (if the delivery is affected) a dated `WORKLOG.md` entry.
