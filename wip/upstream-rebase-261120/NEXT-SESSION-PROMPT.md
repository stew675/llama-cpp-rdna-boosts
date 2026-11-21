# Prompt to paste into the fresh session

> Continue the r328 upstream re-base ladder in `~/llama-cpp-rdna-boosts/wip/upstream-rebase-261120/`.
>
> **First, read `COLD-START.md` in that directory top to bottom** — it is self-contained. Then verify the state it
> claims (`cd ~/llama.cpp && git status --short && git branch --show-current && git log --oneline -1`: expect
> clean, `rebase-r328`, `a9d8e1a96`, tree `f0f9db34b1c3ae8a93f834e5c2fad472113a1af9`). Read `RUNGS.md` §R12
> (what the last rung cost and what it left open), then the ledger rows for R13-R17 and `DECISIONS.md`'s live
> items (D4, D8, D9, D12, D13, **D17**).
>
> **R12 landed 2026-11-20** (`a9d8e1a96`): the paused speculator-class merge was finished by putting the
> delivery's DFlash features on upstream's `dflash_options`/`init(opt)` seam, two splice-orphaned server features
> were restored (`prompt_load_with_room` had no caller; `server_prompt_cache::load()` read three locals that had
> moved into `find()`), and the DFlash decision (**D17**) was settled by measurement in the maintainer's
> direction: **the delivery's whole feature set stays** (ingest tail +29.8 % pp at 26K on a 2-GPU 35B, +11.7 % on
> 3-GPU Flash-Next; F1 +6-7 % pp on the dense 27B and inert on both MoE heads; text and acceptance identical in
> every pair). Numbers in `benchmarks/2026-11-20-r328-r12-dflash-merge.md`. R12 ran six gates green
> (`gate-spec-reuse-window`, `gate-seqpar --record`, `gate-state-pipe`, `gate-ckpt-reuse`, `gate-park-idle-slots`,
> `gate-mtp-dev-rows`) plus coherence `1c5d32ac537d` and `MUL_MAT_ID` 1077/0 (== the base's own count).
>
> **Job 1 — R13** (`git cherry-pick 1014e6bf5`, 0 conflicts expected). Build, coherence, the block's own census /
> kill-switch arm, then the rung log in `RUNGS.md`.
>
> **Job 2 — the two carried obligations, in this order.** (a) **R11.2**: `scripts/gate-hc-fusions.sh` is still RED
> (`mix_op`, `dsv4_pre_op`, `ops_no_post` left their r327 anchors). Block 12's `hc`/`mmb` knob path is now in the
> tree, so re-run it and see whether the `mix_op` anchor is restored; if the fused **mixer** still differs it must
> not ship default-on, and re-anchoring planted values is a maintainer decision, arm by arm. (b) Then the R12-owed
> gates from `COLD-START.md` §8: `gate-spec-align.sh`, `gate-spec-ckpt-device.sh`, `gate-spec-width-band.sh`, and
> the `-sm tensor` arms of `gate-host-pool.sh` / `gate-prefill-logits.sh` for block 09's Meta split-state cache.
>
> **Non-negotiable constraints:** no `-lv` on a text arm — `-lv 4` only for a census, and note that `-lv 4` puts
> every census on **stderr** while the generated text stays on **stdout**, so one run can serve as both arms; pin
> `HIP_VISIBLE_DEVICES`; one GPU workload at a time; build-completion by the `BUILD_RC=` sentinel, never `pgrep`;
> any abort gets the same arm re-run on `~/llama.cpp/build-rocm/bin-r327` before it is called a regression, and on
> `build-base-novmm` when the question is "is this upstream's behaviour?"; never push from `~/llama.cpp` except the
> release-time `rdna-boosts` refresh to the personal fork; report findings before committing anything
> delivery-affecting.
>
> **Still on the board and not to be lost** (`COLD-START.md` §8 has the detail):
> 1. **R10.3 is a RELEASE BLOCKER** — the host-expert arena arm's decode fell 54.4 → 20.4 t/s between R09 and R10
>    (35B-A3B Q8_0, 1 GPU, `-ncmoe 20 -c 8192 -ub 1024`, `MOE_EXPERT_HEATMAP=load`), reproducibly, text unchanged,
>    no switch recovers it. Re-check at **R15**; r328 does not ship with it open.
> 2. **R11.2** — as above.
> 3. **R06.2 (routed half)** — the forced routed MMB arm still aborts on the pool LIFO assert where r327 reads
>    `86b7f9b5aa80`; owed at R15/R18 (an A/B override, never a shipped default).
>
> Then carry on: R14 (3 conflicts, the streamed KV window), R15 (**the R10.3 re-check**), R16 (server prompt
> admission), R17 (block `0300`, including `TODO` #351), then R18's release checklist (`RUNGS.md` R18: the
> `--all` per-block ladder, `TODO` #352, the R00 repair squash, `gate-prefill-logits.sh --record`, the full
> quant-coherence + QSA section E, `make-patches.sh`/`make-release.sh`/`validate-set.sh`, `fork-refresh.sh
> --build --gates --push`, the worker container build, and the `WORKLOG.md` release entry). Write each rung's log
> in `RUNGS.md` and commit the delivery repo's docs per rung; `patches/`, `release.json`,
> `rdna-boosts-all.patch` and the fork stay untouched until R18.
