# Prompt to paste into the fresh session — R18, the release

> Do the r328 release from the completed ladder in `~/llama-cpp-rdna-boosts/archive/work/upstream-rebase-261120/`.
>
> **First, read `COLD-START.md` top to bottom**, then verify its state: `cd ~/llama.cpp && git status --short &&
> git branch --show-current && git log --oneline -1` — expect clean, `rebase-r328`, `73a74454b`, tree
> `0c46248f7371a0eee84a980a1e16ad25a6fbd38f`, **18 commits** (the three rung repairs are squashed, `RUNGS.md`
> §R18.1).  All 18 blocks (R00-R17) are landed and each rung's log is in
> `RUNGS.md`; the release checklist is `RUNGS.md` → R18 and the pre-release obligation table is
> `COLD-START.md` §8.
>
> **Two things need the maintainer's decision before any release commit or tag:**
> 1. **R11.2 — `gate-hc-fusions.sh` is still RED at the completed tree**, reading exactly the R11 numbers
>    (`mix_op` `3f049f67781c`, `dsv4_pre_op`/`ops_no_post` `c8db61d5dcc8`, `dsv4_post_op`/`all_ops`
>    `6a102dbc62d6`, chain `a512ee68c4b3`; the combine op is exact and nothing faults).  Blocks 12-17 did not
>    restore the anchors, so the HC chain itself moved with the re-base — either the planted anchors are
>    re-anchored **arm by arm** (never wholesale), or the fused mixer does not ship default-on.
> 2. **Block 15's APU-side claims are unvalidated on this base** (`gate-uma-experts.sh` cannot fire on gfx1201 —
>    the store is inert there *by design*; `gate-prefill-server-slots.sh` needs the 122B >RAM model).  The
>    discrete side is green (`gate-host-pool.sh` 59/59).  Ship with that recorded as a documented
>    unvalidated state, or run them on the `halo` (gfx1151) box first.
>
> **R18's order of work** (`RUNGS.md` R18 is the contract).  **Already done 2026-11-21:** `TODO` #352 (the
> container `web` stage → `ui/apps/chat`), the squash of the three rung repairs (§R18.1), the per-block `--all`
> ladder (§R18.2), five block gates (§R18.3), the block-03 radiance attribution fixup (§R18.5) and the
> **R11.2 refutation + `gate-hc-fusions` fix** (§R18.4 — the gate is GREEN in both states; no delivery defect).  Left: 
> `scripts/gate-per-block-build.sh --base 919794500 --tip rebase-r328 --all` (running) + `scripts/check-gpu-faults.sh` → the block-owned gates the re-base
> touched, including the ones R12-R17 did not run (`gate-spec-align.sh`, `gate-spec-ckpt-device.sh`,
> `gate-spec-width-band.sh`, `gate-norm-rows-alias.sh`, and `gate-node-budget.sh` **if a phi-4 file is fetched** —
> it is not on this box) → `scripts/gate-prefill-logits.sh --record` on the **final** tree, then the gate itself
> (the R12 record was taken on an intermediate tree and is not valid) → `scripts/make-patches.sh` from a canonical
> fork rebuilt at `release.json.base` via `scripts/apply-all.sh`, then `scripts/make-release.sh`, then
> `scripts/validate-set.sh` → `scripts/fork-refresh.sh --build --gates --push` (the **personal fork only**), the
> tag, then let the worker watcher build and push the ROCm images (never on the primary box) →
> `WORKLOG.md`'s release entry, `patches/README.md`'s per-block notes, `INVARIANTS.md` for the facts the re-base
> changed, `TODO.md`/`ENVIRONMENT.md` for anything deferred.
>
> **Non-negotiable:** no `-lv` on a text arm (`-lv 4` only for a census, and it lands on **stderr** while the text
> stays on stdout, so one run can serve both); pin `HIP_VISIBLE_DEVICES`; one GPU workload at a time; build
> completion by the `BUILD_RC=` sentinel, never `pgrep`; any abort or divergence gets the same arm re-run on
> `~/llama.cpp/build-rocm/bin-r327` (and on `build-base-novmm` when the question is "is this upstream's
> behaviour?"); never push from `~/llama.cpp` except the release-time `rdna-boosts` refresh to the personal fork;
> report findings before committing anything delivery-affecting, and remember that `patches/`, `release.json`,
> `rdna-boosts-all.patch` and the fork are still untouched — R18 is the first epoch that may change them.
>
> **Numbers of record from the climb** (do not re-derive): coherence `100 chars sha=1c5d32ac537d` at every rung;
> arena decode at the finished tree 49.9 t/s median (R10.3 closed, r327 = 49.7); DFlash feature set kept (D17,
> ingest tail +29.8 % pp at 26K on a 2-GPU 35B); `MUL_MAT_ID` 1077/0 (the new base's own count); `gate-host-pool`
> 59/59; `gate-seqpar` PASS and its `--record` refreshed at R12 (re-record at R18); `gate-kv-residency`,
> `gate-spec-draft-window`, `gate-state-pipe`, `gate-ckpt-reuse`, `gate-park-idle-slots`, `gate-mtp-dev-rows`,
> `gate-spec-reuse-window`, `gate-prefill-fairness`, `gate-prefill-duty` all PASS at their rungs.
