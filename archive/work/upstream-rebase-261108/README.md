# Upstream re-base campaign — 16 blocks from `6050b467c` onto `a691b2345`

**Status: OPEN — rung 0 (setup) done 2026-11-08.  Rungs R01..R16 are the replay; R17 is the release
ladder.**  Nothing here is delivery: `patches/`, `release.json` and the fork are untouched until R17.
Target release is the next one after r235 (r236 unless an amendment intervenes) and the next fork point
is **`a691b2345`**.

This campaign is a **ladder climb**, not a single rebase: one rung per delivery block, each rung ending
in a resolved commit, a green build and a coherence run, so a session can stop after any rung and the next
one resumes with `git cherry-pick <next block>`.  The rungs are in [`RUNGS.md`](RUNGS.md), the
"upstream may subsume us" questions in [`DECISIONS.md`](DECISIONS.md), and the measured conflict map (the
evidence the ladder was laid out from) in [`CONFLICT-MAP.md`](CONFLICT-MAP.md) — regenerate it with
[`probe-conflicts.py`](probe-conflicts.py) (machine-readable output: `conflict-map.json`).

**Cold start:** read [`HANDOVER.md`](HANDOVER.md) first — it carries the state, the environment and the blocked
rung's debug plan.

**Scope: `patches/` only.**  The maintainer's ruling (2026-11-08): every open `wip/` campaign's patches are
**ignored** for this re-base — they are *not* applied, not rebased and not consulted while resolving a rung,
and a rung that seems to need one gets a `TODO.md`-shaped note instead.  They will each be adapted to the new
base in their own later session.  The same goes for `beta/`, `optional/` and `archive/work/` artifacts.  The
re-base's only inputs are the 16 block commits of `patches/` (and the reference records the rungs cite,
which are read, not applied).

## §0  Baseline facts (as measured 2026-11-08)

| item | value |
|---|---|
| old fork point | `6050b467c` — 2026-10-25 19:46 +0200 (`tests : adjust default threads for test-llama-context (#31157)`) |
| new fork point | **`a691b2345`** — 2026-11-08 11:50 +0200 (`webgpu : add COUNT_EQUAL support (#31578)`), the maintainer's snapshot of upstream `master` in `~/llama.cpp` |
| upstream drift | **318 commits**, **947 files**, +116932 / −45160 |
| delivery tip (old) | `69d36099b`, tree `ef5dcbcb0`, base tree `cb3251e17` (`release.json`, r235) |
| delivery shape | 16 blocks (00 + 01-15), 123 977 patch lines, 536 distinct files; 141 of them upstream also touched |
| probe result | replaying the 16 blocks onto `a691b2345` leaves **75 conflicted paths**; blocks **04, 06 and 14 apply clean**; worst are 00 (12), 05 (9), 15 (9) — see `CONFLICT-MAP.md` |
| working branch | `rebase-r236` in `~/llama.cpp`, created at `a691b2345`; **`master` there stays the pristine snapshot — never commit to it** |
| r235 reference binary | `~/llama.cpp/build-rocm/bin-r235` (a copy of the r235 `bin/`, version line `build 11903, commit 69d36099b`) |
| canonical build | `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `build-rocm/`, ccache 30 GiB) |
| models | `/llm/models/Qwen3.5/4B/Q8_0/…`, `/llm/models/Qwen3.6/35B-A3B/Q8_0/…` (`runme-rocm` carries the host-expert config) |

## §1  Method

**Replay one block commit at a time onto the new base.**  `git rebase --onto` would do the same merges but
leaves an un-committed rebase state that is fragile across sessions; a cherry-pick per rung leaves a
committed, buildable branch at every stop.

```bash
cd ~/llama.cpp
git status                                  # must be clean, on rebase-r236
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

**The text reference is NOT r235.**  Upstream moved numerics on purpose (`12ab59798` F32 src1 in the CPU
backend, `b74298c10` conv-state refactor, `162e2bb2e` bf16 fix, `b961…` rotation constants, the MoE layout
linearisation, …), so r235-identical greedy text is the **wrong** oracle for a rung.  A rung is judged by:
build + coherence + the block's own kill-switch A/B (a delivery feature that is default-on must still move
its own measurement) and, where the block owns a gate, "the gate reaches the code it is supposed to reach".
r235-identical text is only used as a *smoke* check when a block has no kill switch and touches no
numerics; a difference there is recorded, not "fixed".

### Traps that already exist here

- **Never commit to `master` in `~/llama.cpp`** — it is the pristine new-base snapshot (and a bare `git push`
  there targets upstream; see `AGENTS.md`).
- **One long-lived source path.**  The replay happens in `~/llama.cpp` because that is where `ccache` is
  warm (`scripts/scratch-tree.sh`'s measured rule).  Do not make a rebase worktree at a new path: the first
  build there is a full cold compile (measured 5m56s / 85 CPU-min).
- **Do not run the per-block gate or a container build beside a benchmark** (and never in parallel with
  another benchmark).
- **A rung is a cherry-pick, not a re-cut.**  Do not regenerate `patches/` mid-climb; `make-patches.sh` runs
  once, at R17, from the finished branch.
- **`-sm tensor` needs `HIP_VISIBLE_DEVICES` pinned** (the iGPU is otherwise enumerated and the load aborts
  with `invalid kernel file`).
- **A rung build needs a CLEAN configure** (`~/bin/build-llama-rocm clean`; ~1m20s wall on warm ccache).  A
  reused `CMakeCache.txt` carries the *tip's* options, and some of them are illegal at an earlier rung — the
  live example is `GGML_CUDA_FA_QUANTS`, which holds block 05's `iq4_nl-iq4_nl` instances and makes the
  configure fail for blocks 00-04 with `unknown type "iq4_nl"`.  This is the "r2 lesson" already recorded in
  `scripts/gate-per-block-build.sh`; the `iq4_nl` instance must be re-set by hand when R06 lands, because that
  variable is set by hand on this box, not by the delivery.
- **Build with `GGML_HIP_NO_VMM=OFF`** (the maintainer's decision, 2026-11-08: the delivery's host-expert tier
  needs VMM and upstream's HIP default is `ON`).  `~/bin/build-llama-rocm` now passes
  `-DGGML_HIP_NO_VMM=OFF`; the earlier `build-rocm-vmm` dir is redundant but kept.  `ENVIRONMENT.md` still calls
  the flag "an open investigation" — that row needs correcting at R17.
- **A rung's diff may shrink, and that is the point.**  R01 delivered 34 files / +986−340 from a block of 56
  files / +1366−909 because the difference was rename-only `-Wshadow` ballast (see `RUNGS.md` R01).  Judge a
  rung by its resolutions, not by its line count.
- **Report before deciding.**  Any rung that reaches a `DECISIONS.md` item stops there: present the
  measurement and the options to the maintainer and wait.  Never drop delivery code because "upstream's
  looked newer" (see `AGENTS.md` → report findings).

## §2  The ladder

16 replay rungs, grouped into sessions.  Per-rung conflict counts are the 2026-11-08 probe's (an upper
bound: the probe chained through marker-carrying trees).

| session | rungs | blocks | conflicts | notes |
|---|---|---|---|---|
| 0 | R00 | — | — | **DONE 2026-11-08** — campaign, branch, `bin-r235`, conflict map, decision register |
| 1 | R01 | 00 foundation | 12 | the biggest single cluster; everything later assumes it.  `D7` (Meta compute views) |
| 1 | R02 | 01 hybrid all-reduce | 1 | `D10` |
| 1 | R03 | 02 adaptive MTP | 4 | `D11` |
| 1 | R04 | 03 chunked GDN | 1 | **`D4` — upstream #31368 may subsume our rollback tracker** |
| 1 | R05 | 04 BF16 KV | **0** | `D6` (verify only) |
| 2 | R06 | 05 flash attention | 9 | **`D5`** — upstream #29510 restructures the FA entry points |
| 2 | R07 | 06 MMB prefill | **0** | — |
| 2 | R08 | 07 k-quant boosts | 1 | — |
| 3 | R09 | 08 fused kernels | 5 | **`D2`** — upstream #31188/#31288/#31296 fused MoE |
| 3 | R10 | 09 host-resident MoE | 7 | **`D1`** — upstream's own MoE cache grew ClassLRU + CPU share |
| 4 | R11 | 10 infrastructure | 8 | `D9` (graph key / async ring / views) |
| 4 | R12 | 11 qwen4exp | 7 | **`D3`** — upstream reworked QSA/indexer/sparse selection |
| 5 | R13 | 12 campaign remainder | 8 | `D8` (CPU wins), reference-counted checkpoints, DFlash |
| 5 | R14 | 13 MTP vocab subset | 3 | `D11` |
| 5 | R15 | 14 relaxed-tensor (**temporary**) | **0** | re-checked: upstream still denies those arches (`#31147` only adds laguna to the deny list), so the block is not subsumed |
| 6 | R16 | 15 streamed KV | 9 | `D12` + `D13` (memory instances after #31141/#31200) |
| 7 | R17 | — | — | release ladder: regenerate, validate, gates, `release.json`, fork refresh, records |

Rung mechanics, per-rung conflict file lists and the resume command: [`RUNGS.md`](RUNGS.md).

## §3  Decision points

Twelve upstream/delivery questions are registered in [`DECISIONS.md`](DECISIONS.md).  The five that can
**drop delivery code** are `D1` (MoE cache), `D3` (qwen4exp/indexer/sparse), `D4` (recurrent rollback),
`D5` (FA entry restructure) and `D7` (Meta compute views); every one of them stops for a maintainer
decision with a measurement in hand.  The rest are "coexist, verify the dispatch" (`D2`, `D6`, `D10`,
`D11`), "re-derive after a refactor" (`D9`, `D12`, `D13`) or "which of our wins survive" (`D8`).

## §4  End of the ladder (R17)

Not a rung in the climb — the release gate set, in the order the traps fire:

1. `scripts/make-patches.sh` from the rebased branch (16 files, block 00 keeps `0000`), then
   `scripts/make-release.sh --base a691b2345 --tip <tip> --tree <tree>`.
2. `scripts/validate-set.sh` — strict apply on a fresh tarball at `a691b2345`, tree equality, sha256s,
   Windows path/portability lints.
3. `scripts/gate-per-block-build.sh` — a clean build per block commit (all 16).
4. Coherence: dense (`Qwen3.5-4B`), the host-expert config (`Qwen3.6-35B-A3B` `runme-rocm`), and a hybrid
   recurrent model.
5. `scripts/gate-prefill-logits.sh --record` **from a clean unpatched `a691b2345` build** (the re-base's own
   base record, same ROCm line), then the patched build against it; plus the `--sm tensor --device 0,1,2`
   arm for block 01 and the `--kv q8_0` arm for the quantized-cache paths.
6. MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) + `llama-batched-bench -npl 1,4,8`.
7. Every block-specific gate whose code moved: `gate-host-pool.sh`, `gate-kv-residency.sh`,
   `gate-kv-window-split.sh`, `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`,
   `gate-spec-align.sh`, `gate-spec-width-band.sh`, `gate-spec-draft-window.sh`, `gate-spec-reuse-window.sh`,
   `gate-hc-fusions.sh`, `gate-norm-rows-alias.sh`, `gate-grid-stall-pad.sh`,
   `gate-qwen4exp-quant-coherence.sh`.
8. `scripts/check-gpu-faults.sh` on the journal + log.
9. Release: `scripts/fork-refresh.sh --build --gates --push` (personal fork only), the container images via
   the off-host worker, then the records — a `WORKLOG.md` entry with the new base/tip/tree and the gate
   numbers, `patches/README.md` per-block notes for what moved, `BASELINE.md` current baseline,
   `INVARIANTS.md` appended facts, `ENVIRONMENT.md` for any variable that appeared/vanished,
   `TODO.md` for the deferred decision items, and this campaign moved to `archive/work/`.
   **Known doc debts collected during the climb** (fix them at R17, they are listed per rung):
   * `ENVIRONMENT.md`: `LLAMA_RS_ROLLBACK_DEBUG` loses its RSROLL half (RSPREP only, R04); the r34
     `GGML_CUDA_FA_QUANTS` note if the instance list changes (R06);
   * `patches/README.md`: block 03's tracker/warn text (R04), block 00's MUL_MAT_ID shape (R01), the
     `-Wshadow` statement in the block-00 note (R01);
   * `INVARIANTS.md`: the recurrent rollback guard entry becomes "upstream's `rs_max` + our diagnostic" (R04).

## §5  Resuming

**Start with [`HANDOVER.md`](HANDOVER.md)** — it is the cold-start brief: exact state, the environment, the blocked
rung with its evidence and an ordered debug plan, the remaining rungs, the open decisions and the list of
"already paid for" facts.

```bash
cd ~/llama-cpp-rdna-boosts && sed -n '1,40p' wip/upstream-rebase-261108/RUNGS.md   # find the open rung
cd ~/llama.cpp && git log --oneline -3 && git status                               # confirm the branch state
```
A rung that was interrupted mid-conflict leaves `git status` dirty and `.git/CHERRY_PICK_HEAD` present —
finish or `git cherry-pick --abort` and redo it; do not leave a half-resolved rung for the next session.
A rung that is *blocked* by a regression is different: roll the branch back to the last green rung, park the
resolution as a patch in this directory, and record it (that is what R11 did — see `RUNGS.md` R11 and
`r11-resolution.patch`).
