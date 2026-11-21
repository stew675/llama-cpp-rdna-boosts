# r328 follow-ups — cold-start handover

> ## CLOSED — 2026-11-21 (r329 follow-up session).  Every row below is closed; this file is the record.
>
> | row | outcome |
> |---|---|
> | **F1** `TODO` #357 | **not a delivery defect.** Both halo gate runs certified `~/llama-scratch/build-rocm` (the gate's then-default, a 2026-11-12 build of `7efb30ae5`) instead of r328: its `apply_heat_seed_locked` predates r277's transport tail, so the census check read an absence. With `LLAMA_CLI=~/llama.cpp/build-r328/bin/llama-cli`: **49/49**, `120 seed line(s)` on the placement queue. Both gates now default to `~/llama.cpp/build-rocm` and print their binary's `--version` on line 1 (`INVARIANTS.md` r329). Log: `logs/halo-gate-uma-experts-r328.log`. |
> | **F2** `TODO` #354 | **closed.** `gate-prefill-server-slots.sh` on halo, r328 server, **`LME=85 EXTRA_ARGS='--lazy-mode auto'`: 3 checks / 0 failures** — A identical `ee7c607ad86e`, A 2.28 → 2.47 tok/s, and the in-place band really ran (159 calls × 75 tables, 3351 admissions, 1113 landing / 2238 reclaiming a non-routed resident). The old default `-lme 90` is what killed it: 94 GiB anon on top of a >RAM mapping → server OOM-killed + 7 desktop tasks (`logs/halo-gate-prefill-server-slots-r328-lme60.log`, `…-lme85.log`). Per-box prefill-logit bases recorded from `build 6d9d54238` on **halo** (gfx1151, 27B Q8_0) and **fingon** (gfx1100, 27B Q4_K_M), both `mean KLD 0.000000 / top-p 99.990 %` (`logs/halo-kld-record-r328.log`, `logs/fingon-kld-record-r328.log`). gfx1100's +7.2 %: that box runs the arm with `RECOVER=1.05` and records the number (its pair is memory-bound; the inert arm alone moves +12 %) — `MANIFESTS.md` gate 11. |
> | **F3** `TODO` #356 | **refuted, no block-02 change.** `-lv 5` shows the caps ARE equalised in the call — `aligned draft depth seq N: 3 -> 5 (width band 16, drafting 2, hard 5)` on every two-drafting call — and on a `-ncmoe` host-pool shape `llama_model_moe_cache_band()` answers 0, so the width band (16) is what applies. The 2 residual splits are **prefill/decode boundary** steps (a last prompt chunk cannot merge with the other slot's verify batch, and padding would advance the recurrent state). `gate-spec-align.sh`'s text corrected; `NCMOE_SPLIT_MAX=2` unchanged (`logs/f3-spec-align-diag-lv5.log`). |
> | **F4** `TODO` #352 | **closed on the worker's evidence.** `radiance` built and pushed all six r328 images (both ROCm lines × full/light/server) at 07:07; its log shows the `web` stage building `ui/apps/chat` (`vite build` + PWA assets) and `COPY --from=web … ui/apps/chat/dist`. No local image build was run (policy). |
> | **F5** R06.2 routed half | **closed.** r328 forced routed arm: exit 0 and the control hash `86b7f9b5aa80` (r327 identical), with the routed kernel proven reached (`MMB_GLU fused gate/up+swiglu`, 2 hits). The abort belonged to the intermediate ladder tree (blocks 06+09 without block 10's `pool_stack`) — `logs/f5-mmb-routed-ab.log`, `logs/f5-mmb-routed-census.log`. |
> | **F6** `gate-node-budget` | **closed.** phi-4 IQ4_XS fetched to `/llm/models/phi-4/IQ4_XS/` (the gate's `MODEL` default); four arms PASS, all `31 chars sha=a1b7eb2ee7a6` (`logs/f6-gate-node-budget-phi4.log`). |
>
> Delivery change in this session: **three gate scripts only** (`gate-uma-experts.sh`, `gate-prefill-server-slots.sh`,
> `gate-spec-align.sh` — binary anchoring + defaults, an `EXTRA_ARGS` passthrough, and F3's corrected text). No patch
> block changed, no block renumbered; r329 is unblocked.  Full record: `WORKLOG.md` 261121.007.

**Read this top to bottom; it is self-contained.**  It exists because `v17-919794500-r328` was released on
2026-11-21 **before** two validations the policy asks for (the MTP four-axis protocol and the gfx1151/gfx1100
arches).  Both have since been run and both pass — see §2 — and what remains is a short, named list.  **A new
release (`r329`) must not be cut until every row in §3 is closed or explicitly parked here with a reason.**

| item | value |
|---|---|
| release | **`v17-919794500-r328`** (tagged, published, CI green, fork refreshed, images built by the worker) |
| base / tip / tree | `919794500` / `6d9d54238` / `0c46248f7371a0eee84a980a1e16ad25a6fbd38f` |
| delivery repo | `~/llama-cpp-rdna-boosts`, `main` at **`ecfce7bd`** or later (pull first) |
| release record | `WORKLOG.md` 261121.005 (the release) and 261121.006 (the post-release validation) |
| re-base campaign | `archive/work/upstream-rebase-261120/` (`RUNGS.md` per rung, `DECISIONS.md`, `REMOTE-VALIDATION.md`) |
| evidence for this file | `archive/work/r328-followups/logs/` (29 logs: every gate below, both primary-box and remote) |
| ladder checkout | `~/llama.cpp`: branch `rebase-r328` = `73a74454b` (the pre-cut ladder; `rdna-boosts` is now the re-cut r328 branch), `backup-pre-squash-r328` = the pre-squash tip |
| remote boxes | `ssh halo` (gfx1151 APU, 123 GiB, 16 threads) · `ssh fingon` (gfx1100, 24.5 GiB, 16 threads) — both hold the r328 tree in `~/llama.cpp/build-r328` and the logs in `~/r328-arch/` |

---

## §1 What is actually finished (do not redo)

* The 18-block set re-based onto `919794500`: per-block `all` ladder **18/18**, coherence `1c5d32ac537d`,
  prefill-logit **KLD 0.000000 / same-top-p 100 %** against a base recorded from r327's own binaries,
  `validate-set.sh` **green**, fork re-cut tree+tip verified, tag + GitHub Release assets, worker images.
* Ten block-owned gates green on the final tree (list: `WORKLOG.md` 261121.005) plus `gate-spec-purity` and the
  four-axis MTP protocol (§2).
* Both other arches validated from a **fresh apply of the released set** (§2).

## §2 The two post-release validations — both **PASS**, records in `benchmarks/`

1. **Four-axis adaptive-MTP protocol** (`benchmarks/2026-11-21-r328-mtp-4axis.md`): 3-device `-sm tensor`, both
   cache types, `-n 3000`, delivery default `--spec-draft-n-max 7`.  `draft-mtp` is **1.97×-4.42×** plain,
   pos-1 acceptance **0.761-0.984**, and `--spec-type none` == `draft-mtp` **byte-identical on all four axes on
   both caches**.
2. **`gate-spec-purity.sh`** (the dedicated issue-#330 gate, run *after* the release to answer the "is purity
   broken?" question): **PASS** — manifest A (q8_0/prose) identical `c1f9d0b3f5b6` (41.3 → 99.1 t/s),
   manifest B (bf16/reasoning) identical `2fbc183348a8` (42.3 → 99.7), the four axes on both caches all
   identical, and the width probe's flat **and 256-cell-boundary** arms ran.  Log:
   `logs/spec-purity.log`.  **Nothing is broken here**; the property holds at the strictest shape.
3. **gfx1151 + gfx1100** (`benchmarks/2026-11-21-r328-arch-validation.md`): apply **18/18** with tree *and* tip
   matching `release.json` on both boxes, build OK, coherence identical to r327 on each
   (`da2e2d192e21` / `2f7f092f2383`), the full op matrix green on both — including `GATED_DELTA_NET` **75/75 on
   gfx1100**, the first real gfx11 run of `gated_delta_net_chunked_bf16_gfx11.cu` — block 15's
   `gate-uma-experts` **48/49** on the only box where it is not inert, and `gate-hc-fusions` **PASS** on gfx1100.

---

## §3 The open work — every row must be closed or parked before `r329`

Sorted by what blocks what.  "Owner" is the file/doc that must change.

### F1 · `TODO` #357 — gfx1151: the GTT sizing fill does not report its transport (P0 for `halo`)

* **Symptom**: `gate-uma-experts.sh` `48/49` on `halo`; the HARD check at line 456 fails:
  `FAIL ref: the GTT sizing fills did not report the placement queue (the batch path declined silently)`.
* **What the log actually shows** (`logs/halo-gate-uma-experts.log`): the ref arm's stderr carries
  `I apply_heat_seed_locked: layer=31 role=blk.31.ffn_down_exps.weight heat-seeded 56/56 slots from the
  persisted map (256 experts, ranked by the DECODE band)` and **stops there** — no `, X MiB in Y ms
  (Z MiB/s, placement queue)` and no `serial fills`.  Line length is 168 bytes in `err.txt`.
* **Why it is not automatically a transport bug**: `strings ~/llama.cpp/build-r328/bin/libggml-hip.so` on that
  box carries the full format `… ), %.1f MiB in %.0f ms (%.0f MiB/s, %s)` and exactly one `serial fills`, and
  `moe-expert-cache.cu` has exactly one `heat-seeded` print site (line ~5587, the tail of
  `apply_heat_seed_locked`, whose `batched` flag comes from `fill_slots_read_locked`).
* **Next step (cheap, in order)**: (a) reproduce with one minimal arm on `halo` and `cat -A` the captured
  stderr — is the tail really absent, or is the capture truncating?  (b) if absent, instrument: does
  `fill_slots_read_locked` return `false` (→ `serial fills`, and then the *expected* transport on RDNA3_5 UMA
  is the buffered arm, so the gate's expectation is wrong for this arch) or `true` (→ the print is losing its
  tail, i.e. a logger bug, which would also affect the gfx1201 census the day the store engages there);
  (c) fix either the emitter or the check's expectation and record which.
* **Where it lands**: a block amendment (block 15's `.cu`/log or `scripts/gate-uma-experts.sh`) — a *delivery*
  change, so **ask the maintainer before folding it in** (`AGENTS.md` → *Report findings before shipping*).
* **Does not affect the shipped gfx1201 behaviour** in any way: the store is inert there by design.

### F2 · `TODO` #354 remainder — `gate-prefill-server-slots.sh` on `halo`, and per-box KLD bases

* **The `>RAM` two-slot arm has never completed.**  On `halo` the `copy` arm ran (slot A **160 tok at
  3.76 tok/s**, interrupted by the prefill window, longest gap 511 ms; slot B **17 chars in 18.4 s**) and the
  `ip` arm **aborted the harness** — `gate-prefill-server-slots.py` raised `URLError` streaming slot B, i.e. the
  second arm's server did not answer in time (`logs/halo-gate-prefill-server-slots.log`).  That is an
  environment limit, not a verdict.
* **What to do**: raise the request timeout / lower the arm's `-n` and re-run on `halo` (`gate-prefill-server-slots.sh`
  with the 122B-A10B Q8_0, `Qwen3.5/122B-A10B/Q8_0`, 4 shards ≈ 200 GB > 123 GiB RAM).  If it still cannot
  complete, record it as a **documented unvalidated state** with these numbers — that is allowed — and say so in
  block 15's notes.
* **Per-box prefill-logit bases**: the gfx1201 base is refused by design on another ROCm line, so each box must
  record its own (`BASE_DIR` on disk, `MODEL=` explicit on `halo` — its APU pool makes the auto-resolve look for
  a 27B Q8_0 it does not have).  Cheap: `MODEL=… ./scripts/gate-prefill-logits.sh --record` on each box once.
* **gfx1100's `gate-spec-align` host-expert pair: +7.2 %** where the gate wants ≥ 10 % (`RECOVER`).  Everything
  else in that run passes (splits 44+1 → 2+1, texts identical, equal-length arm inert).  **Decision needed**:
  either pass `RECOVER=1.05` on a 24 GiB single-card arm and record why, or leave it reported.  Nothing to fix
  in the code on this evidence.

### F3 · `TODO` #356 — the host-expert aligned pair still splits 2 draft-length steps (block 02)

* **Symptom**: with `LLAMA_SPEC_ALIGN` on, the host-expert (arena) shape leaves **2** draft-length splits
  (`46+1 → 2+1` on gfx1201; `44+1 → 2+1` on gfx1100).  Bounded in the gate (`NCMOE_SPLIT_MAX=2`) so it reports
  rather than fails.
* **Mechanism (already established, do not re-derive)**: `spec_align_width_band()` deliberately stands down where
  an MoE expert-cache arena exists — "the arena's own band replaces it" — and that band is derived **per step**
  from how many rows the arena serves from VRAM, so the caps `align_draft_caps()` equalizes can be unequal at a
  step boundary.
* **The fix** belongs in **block 02** (`common/speculative.cpp`): equalize the band-derived caps inside the same
  call, not only the ones `spec_align_width_band()` produces.  Then re-measure: the gate should read `0+<=2`,
  and `NCMOE_SPLIT_MAX` can drop back to 0 (or the knob is retired).
* Gate to re-run afterwards: `gate-spec-align.sh` on gfx1201 (2- and 3-device) **and** on `fingon` (it has the
  host-expert model), plus the coherence/text arms.

### F4 · `TODO` #352 — the r328 container image smoke build

* The `web` stage now builds `ui/apps/chat` (`261121.001`); the **verification** is `scripts/build-containers.sh
  --rocm 10.1 --arch gfx1201 --targets server` (a local smoke, **no `--push`**, never the full two-line build on
  the primary box) and then confirm the image's `llama-server` serves `index.html`.  The release images
  themselves were built off-host by the worker watcher; check its log first
  (`ssh radiance 'tail -20 ~/.cache/rdna-containers/watch.log'`) — if the worker's images are good, the smoke is
  a formality and the row closes with that evidence.

### F5 · `R06.2`'s forced routed-MMB arm — an A/B, never a shipped default

* `GGML_CUDA_MMB_TYPES=q8_0 GGML_CUDA_MMB_ROUTED=1` on 35B-A3B still aborts on the pool LIFO assert where r327
  reads `86b7f9b5aa80`.  RDNA4 ships `routed=0`.  Owed as a **measurement**, not a blocker: run the arm on the
  r328 build and on `bin-r327`, and either explain the assert or file it.

### F6 · `gate-node-budget.sh` — needs phi-4, which is on **none** of the three boxes

* Either fetch a phi-4 IQ4_XS shape (the gate needs it: "the tight shape") and run it, or record it as a
  documented gap in `MANIFESTS.md` → gate 06's row with the reason.  Cheap to decide, easy to forget.

### F7 · Not work — do not "fix" these

* **`TODO` #351** (`0300` / `-sm relaxed-tensor`): **answered for r328 — it cannot be dropped** (flint still
  refuses `-sm tensor` at this base, and `relaxed-tensor` is byte-identical to `-sm layer`).  Re-test at the next
  re-base; nothing to do now.
* **`TODO` #353** (Inkling-2): waiting on upstream PR #32695; a *feature campaign* when it lands, not a fix.

---

## §4 The `r329` gate (policy, so it is not re-litigated in the next session)

> **No `r329` cut, tag or fork refresh until F1-F6 are each closed — or explicitly parked in this file with the
> maintainer's agreement and a reason.**  A green gate ladder is not permission to skip a row; a row parked
> without a reason is a bug in this file.

Order of work for the `r329` session, once the rows are decided:

1. Any **delivery** fixes (F1 if it is the emitter, F3's block-02 change) are folded into their **owning blocks**
   with one `git rebase -i` pass — the discipline that worked at R18 (`archive/work/upstream-rebase-261120/RUNGS.md`
   §R18.1/§R18.5): never a trailing commit, never in `0300`.
2. Re-run the gates the changed blocks own (`AGENTS.md` → *Pick the gate for what you changed*), then
   `gate-per-block-build.sh --all` (or `--changed-from <prev tip>` for a delta), then `check-gpu-faults.sh`.
3. `gate-prefill-logits.sh --record` on the **final** tree, then the gate; the four-axis MTP protocol; and
   `gate-spec-purity.sh` (a decode/verify change touches both).
4. `make-patches.sh` from a canonical fork rebuilt at `release.json.base`, `make-release.sh`, `validate-set.sh`
   — including the **optional artifacts re-cut against the new tree** (`optional/` must never be left stale:
   `validate-set.sh` check 8 fails on it).
5. `fork-refresh.sh --build --gates --push` (personal fork only), the tag, then **let the worker build the
   images** (`ssh radiance`, cron `*/10`) — never the full two-line build on the primary box.
6. `WORKLOG.md` entry, `patches/README.md` per-block notes, `INVARIANTS.md` for any fact that changed,
   `TODO.md` rows retired (a closed item is a WORKLOG entry, never a TODO row).

---

## §5 Rules that bit in this session (carry them forward)

* **Measure before you tag.**  The MTP protocol and the other two arches were owed *by policy* and ran after the
  release; both pass, but the order was wrong and that is why this file exists.
* **A red gate is not a delivery defect until the previous release's binaries say so.**  Two of this session's
  "blockers" (R11.2 and the spec-align host-expert check) reproduced **byte for byte** on r327: the anchors were
  stale and the gate had no anchor on its *reference* arm.  Always run the gate with
  `LLAMA_CLI=~/llama.cpp/build-rocm/bin-r327/llama-cli` first.
* **Poll a long job without blocking the session**: start it with `nohup setsid … &`, then poll with
  `for i in $(seq …); do grep -q <sentinel> && break; sleep 15; done` — a bare `sleep 900` in one tool call
  wasted two rounds this session.
* **One GPU workload at a time — and check it**: `pgrep -cx llama-cli` (exact name).  `pgrep -af` matches your
  own command line *and* prints the whole `-p` prompt, which is how two overlapping MTP passes happened.
* **`/tmp` is tmpfs** (46 GiB of RAM on the primary box): huge artifacts go to `~/.cache/rdna-boosts/gates`
  (`gate-prefill-logits.sh` now refuses a tmpfs base; `gate-host-pool.sh`'s KLD base moved there too).
* **`-lv` per arm**: a text/hash arm takes **no `-lv`**; a census arm takes `-lv 4`.  `-lv 3` does neither.
* **Small-card configs**: a 24.5 GiB card needs `-ncmoe` or a fitting model — a `-ngl 99` arm on `fingon` simply
  OOMs, and a gate run there needs `EXTRA_ARGS='-ngl 99 -fa on -ncmoe 40'` and `SOFT_EXPECT=""` (its anchors are
  per class/state/3-device, so a 1-GPU run has none).
* **`halo`'s model inventory is not the primary box's**: 27B Q8_0 *is* there, 35B-A3B UD-Q4_K_M is **not**
  (that one is on `fingon`), and the 122B-A10B Q8_0 is `halo`'s.  `REMOTE-VALIDATION.md` has the table.
* **Evidence travels**: every gate log for this release is in `archive/work/r328-followups/logs/` — quote from them instead
  of re-running a 20-minute gate to read a number.

## §6 Where things are

| what | where |
|---|---|
| this file | `archive/work/r328-followups/HANDOVER.md` |
| the next session's prompt | `archive/work/r328-followups/NEXT-SESSION-PROMPT.md` |
| every log behind §2/§3 | `archive/work/r328-followups/logs/` |
| the release + post-release records | `WORKLOG.md` 261121.005 / 261121.006 |
| the re-base's per-rung record | `archive/work/upstream-rebase-261120/RUNGS.md` (§R18.1 squash, §R18.3 gate sweep, §R18.4 R11.2 refutation, §R18.5 attribution fixup, §R18.6 tmpfs, §R18.7 spec-align, §R18.8 release) |
| the arch runbook + box inventory | `archive/work/upstream-rebase-261120/REMOTE-VALIDATION.md` |
| MTP + KLD protocols | `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md` |
| policy / engineering facts | `AGENTS.md`, `INVARIANTS.md` (see the r328 entries) |
