# Prompt to paste into the fresh session

> Continue the r328 upstream re-base ladder in `~/llama-cpp-rdna-boosts/wip/upstream-rebase-261120/`.
>
> **First, read `COLD-START.md` in that directory top to bottom** — it is self-contained and it supersedes
> `HANDOVER.md` as the entry point. Then verify the state it claims (`cd ~/llama.cpp && git status`, `git log
> --oneline -1`: expect clean, `rebase-r328`, `dbe4fdd40`). Read `RUNGS.md` (last landed rung R11 plus R06–R10's
> open obligations) and, for the rung itself, `r12-wip/README.md`.
>
> **Job 1 — finish R12.** It is paused, not landed: its ten conflicted files are already resolved (see
> `r12-wip/r12-resolved-but-unbuildable.diff` and the rationale table in `r12-wip/README.md`), and the tree fails
> to build only in `common/speculative.cpp`, because `common_speculative_impl_draft_dflash` is two implementations
> of one class (upstream `#32016`'s `dflash_options`/`init(opt)` seam, 388 lines, vs block 12's inline ctor plus
> the DSpark/DFlash2/LiliCorr/ingest state and methods, 620 lines). Merge them the way `COLD-START.md` §6
> describes — upstream's seam, the delivery's features added as state that `init()` sets plus overrides — then
> build, run the coherence arm (`100 chars sha=1c5d32ac537d`) and the rung's own checks, and write the rung's log
> in `RUNGS.md`. Do not commit a tree that does not build or does not pass coherence; pausing again with the work
> captured is a better outcome than a green-looking commit.
>
> **Job 2 — settle the DFlash question by measurement, in the direction the maintainer set.** The maintainer's
> instruction: *"the best way to make a decision on which way to go with the dflash choice is to measure which is
> more performant. I know that we spent a lot of effort on optimizing DFlash2 within the delivery repo. I am
> reticent to drop its work for upstream's."* So: **preserve the delivery's work by default, and let numbers
> justify any deletion.** Read `COLD-START.md` §7 before measuring — I already checked the layers: the three graph
> builders are the *same code* (two byte-identical, the DFlash2 selector differing only by the delivery's
> all-gather, which is a correctness fix under `-sm tensor` where upstream's version aborts), so the real question
> is the *feature set* (ingest-tail ≈ +10 % prefill, F1 device features, probabilistic drafting, anchor-first
> DSpark) and the seam, not "which lattice is faster". Run the arms in §7's table (U / D / D-it / D-f1 / U-st) on
> the three targets listed there, report prefill and decode separately plus the acceptance counts, and record the
> numbers in `benchmarks/` and the choice in `RUNGS.md` R12.
>
> **Constraints that are not negotiable:** no `-lv` on a text arm (`-lv 4` only for a census); pin
> `HIP_VISIBLE_DEVICES`; one GPU workload at a time; build-completion by the `BUILD_RC=` sentinel, never `pgrep`;
> any abort gets the same arm re-run on `~/llama.cpp/build-rocm/bin-r327` before it is called a regression; never
> push from `~/llama.cpp` except the release-time `rdna-boosts` refresh to the personal fork; report findings
> before committing anything delivery-affecting.
>
> **Two things are already on the board and must not be lost** (details in `COLD-START.md` §8):
> 1. **R10.3 is a RELEASE BLOCKER** — the host-expert arena arm's decode fell 54.4 → 20.4 t/s between R09 and R10
>    (35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, `MOE_EXPERT_HEATMAP=load`), reproducibly, text unchanged,
>    and no switch recovers it. The maintainer: *"the decode regression must be fixed before we release; that can
>    happen later though."* Re-check it at R15; r328 does not ship with it open.
> 2. **R11.2** — `scripts/gate-hc-fusions.sh` is red (`mix_op`, `dsv4_pre_op`, `ops_no_post` left their r327
>    anchors; `comm_op` is exact). Re-run at R13/R14/R17; if the fused mixer still differs, it must not ship
>    default-on, and re-anchoring planted values is a maintainer decision, arm by arm.
>
> After R12, carry on down the ladder: R13, R14, R15, R16, R17 (block `0300`, which includes `TODO` #351 — the
> flint `-sm tensor` re-test that decides whether `0300` can be dropped), then R18's release checklist. Write each
> rung's log in `RUNGS.md` (resolutions table, exit criteria, findings, notes-for-later) and commit the delivery
> repo's docs per rung; `patches/`, `release.json`, `rdna-boosts-all.patch` and the fork stay untouched until R18.
