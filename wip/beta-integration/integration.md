# `beta-integration` — fold the 28 `beta/mmb-general` patches into the 16 delivery blocks

**Branch:** `beta-integration` (this repo; cut from `main` at `0699a3d`, 2026-09-25).
**Status:** **FOLD EXECUTED** (2026-09-25) — the 16 amended patches are regenerated on this branch and
validated at the tree level; build clean; see the Executed record below.  Remaining: the review/
docs sweep and the maintainer's promotion decision.
**Owner:** current session; this file is the running work-task list and the hand-off record.
**Not part of the delivery until the maintainer signs off** (the `AGENTS.md` promotion rule).

---

## 0. Executed record (2026-09-25)

**Result: `84e76d8a2` + the 16 amended `patches/*.patch` reproduces the beta tree exactly.**

* **Rebuild worktree:** `~/llama-integration`, branch `beta-integration`, tip
  `f373450de489dd0fafba5bd285e71844109cd0ec`, tree
  **`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** == the 28-patch beta reference tree.
* **Method:** fresh linear rebuild from `84e76d8a2`; each delivery block cherry-picked and the
  mapped beta patches cherry-picked into it (`-n`), conflicts resolved to the net final content.
* **Mapping actually used** (differs from the v1 plan in §4 — see the deviation note):
  * **block 06 (catch-all):** 0025 (host-buffer input layer), 0028 (CPU tiny split).
  * **block 08 (prefill/MMB):** 0001 (MMB core), 0003 (F32/tiny-M + width probe), 0006, 0007,
    0008 (per-arch defaults), 0009, 0010, 0012, 0014 (conv1d fusions), 0015 (norm rows).
  * **block 13:** the deferred half of 0001 (MMB stand-down of the swiglu->mmq fusion).
  * **block 14:** the deferred half of 0001 (MMB stand-down of the pair fusion) and 0027
    (MMVQ routed band).
  * **block 15 (campaign memory/attention):** 0002 (qsa3), 0003 (qwen4exp always-QSA flip), 0004
    (HC16 producers), 0005 (indexer top-k), 0008 (HC16/beneficial defaults), 0011, 0013, 0016-0024,
    0026 — **because block 15 modifies the same `qwen4exp.cpp`/`ggml-cuda.cu` regions the beta
    series was authored against**, so folding them earlier would require forward references.  This
    is the main deviation from the v1 plan (which aimed most of these at block 14).
* **Verification:**
  * `git diff <rebuilt tip> mmb-beta` **empty** (index tree == `24bb0f5acb…`).
  * Fresh worktree at `84e76d8a2` + strict `git am` of the 16 regenerated patches -> tree
    `24bb0f5acb…` (**STRICT-AM-TREE-OK**, 16/16).
  * `scripts/validate-set.sh` **PASS** (checksums + strict apply on a fresh codeload tarball +
    tree/count match).
  * `~/bin/build-llama-rocm-714` (gfx1201, `BUILD_DIR=build-rocm`) **EXIT 0**; `llama-cli`,
    `llama-bench`, `test-backend-ops` present.
* **Delivered on this branch:** regenerated `patches/` (16), `rdna-boosts-all.patch`, `release.json`
  (`tip f373450de…`, `tree 24bb0f5acb…`, `release v16-84e76d8a2-r8`).
* **Done since:** the top-level `README.md` now describes the folded state (intro, block table,
  0006/0008/0015 rows, the "The mmb campaign is in the delivery" section replacing the old beta
  addendum, Layout, Releases, Current state).
* **Not yet done:** the runtime gates (§6.3-6.6) were not re-run individually because the tree is
  byte-identical to the already-validated beta tree (same code -> same result); the **rest of the
  docs sweep** — `AGENTS.md`, `BASELINE.md`, `TODO.md`, `WORKLOG.md`, `patches/README.md` and
  `scripts/apply-beta.sh` still describe the beta as a separate opt-in layer (and `apply-beta.sh`
  would now conflict if run, since `release.json.tree` is the integrated tree); the maintainer's
  promotion decision.

> **Open design question for review:** the qwen4exp/HC/QSA/indexer group landed in **block 15**,
> not block 14, because block 15 owns the intervening code.  If the maintainer prefers them under
> block 14, the fold must instead *split* each patch's block-15-context hunks — more churn for a
> less clean intermediate history.  The current split (MMB/prefill -> 08, catch-all -> 06,
> qwen4exp campaign -> 15) is the dependency-clean one.

---

## 1. Objective

Fold every one of the **28 `beta/mmb-general/patches/*`** into the appropriate patch of the
**16-patch delivery set** (`patches/0000-…` … `patches/0015-…`), so that the delivery *becomes* the
beta: a fresh `84e76d8a2` + the amended 16 patches must reproduce the current beta tree
**`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** exactly, with each block patch a coherent
amendment of its own feature.

The beta set stays put as the historical/verification record; the deliverable is the amended
`patches/` + `release.json` + docs.  Nothing is pushed.

### Definition of done

1. A rebuilt fork branch carries **16 amended block commits** whose tip tree == `24bb0f5acb…`.
2. `scripts/make-patches.sh` regenerates `patches/` from it; **strict `git am`** of the 16 on a fresh
   `84e76d8a2` checkout reproduces `24bb0f5acb…` (`scripts/apply-all.sh` + `validate-set.sh`).
3. The regenerated set **builds clean** on gfx1201 (`~/bin/build-llama-rocm-714`, 0 errors).
4. The final tree passes the beta-window gates (see §6): width purity, op oracles, MTP acceptance,
   same-seed coherence; plus the delivery's own `FLASH_ATTN_EXT` / `GATED_DELTA_NET` / `MUL_MAT_ID`.
5. `release.json` (`tip`, `tree`, per-patch sha256), `rdna-boosts-all.patch`, `patches/README.md`,
   `README.md`, `WORKLOG.md`, `MANIFESTS.md` are updated; `beta/mmb-general/` is marked integrated.

**Explicitly out of scope:** re-tuning the campaign, changing defaults, editing the beta patch
bodies, deleting `beta/`.  The net tree must not change by one byte.

---

## 2. Immutable reference state (record before touching anything)

| name | ref | commit | tree |
|---|---|---|---|
| fork point | `84e76d8a2` | `84e76d8a2` (upstream master) | `5112eedbce0548ab9547d883e8aa54e993852e94` |
| delivery r7 | `~/llama.cpp` `rdna-boosts` | `bed2726cc` | `7726e514284ea7393bb9097ce305dc5b6dacdb11` |
| beta 28-patch tip | `~/llama.cpp` `mmb-beta` | `7c7be3439` | **`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`** |
| delivery patch files | this repo `patches/` | — | `release.json` `v16-84e76d8a2-r7` |

The 16 delivery commits on `rdna-boosts` are
`35193bc5d` (block 00) … `bed2726cc` (block 15); the 28 beta commits follow on `mmb-beta`
(`c65f556e5` (beta 1) … `7c7be3439` (beta 28)).

**Do not rewrite `rdna-boosts` or `mmb-beta`.**  Rebuild on a new branch / worktree.

---

## 3. Method — fresh linear rebuild, try-fold-else-relocate

The proven approach from `wip/closing-the-gap/consolidation.md` (§3), retargeted: instead of folding
closing patches into beta patches, fold beta patches into **delivery block commits**.

Work in a dedicated worktree so `~/llama.cpp`'s `mmb-beta` (the reference tree) is untouched:

```sh
cd ~/llama.cpp
git worktree add ~/llama-integration -b beta-integration 84e76d8a2
cd ~/llama-integration
```

For each delivery block `B` in order `00..15`:

```sh
git cherry-pick -n <block B commit>            # from rdna-boosts (original block change)
for each beta patch assigned to B, in a dependency-safe order:
    git cherry-pick -n <beta commit>           # from mmb-beta (the beta change)
    # resolve any conflict to the NET final content; the assigned block is the owner
git commit -C <block B commit>                 # one amended commit; reword the subject
```

At the end: `git rev-parse HEAD^{tree}` MUST equal `24bb0f5acb…` (the hard invariant).  Then:

```sh
git diff <rebuilt-tip> rdna-boosts               # empty? no — must equal the beta net
git diff <rebuilt-tip> mmb-beta                  # MUST be empty (byte-identical tree)
bash ~/llama-cpp-rdna-boosts/scripts/make-patches.sh ~/llama-integration 84e76d8a2 <rebuilt-tip>
bash ~/llama-cpp-rdna-boosts/scripts/make-release.sh --tip <rebuilt-tip> --tree 24bb0f5acb...
```

**Conflict guidance.**  A beta patch was authored on the *full* beta tree; replaying it earlier can
conflict.  Resolve to the **net combined content** (what the final tree needs), never to the
pre-beta content.  The tree equality check at every checkpoint is the safety net: after each block
commit, `git diff <state> <corresponding subset>` must contain only the not-yet-folded beta deltas.

**Relocation rule.**  If a beta patch cannot fold into its planned block because it depends on a
*textually later* block's code (a missing function/`struct` field/context), then either
(a) split the patch: the independent half folds into the planned block, the dependent half folds
into the block it depends on, or
(b) relocate the whole patch to the block it depends on (recording the deviation).
Do **not** invent forward references in an early block's patch: an early block that references a
symbol defined in a later block still *applies* (`git am` is textual), but it is bad review material
and can break a bisect.  Prefer (a)/(b).

**Catch-all: block 06.**  Per maintainer guidance (2026-09-25), block 06 — the original host-buffer
revert, whose functional delta upstream has since reverted — is now the **catch-all** block for
mixed fixes that do not neatly fit the feature blocks.  Use it for the independent backend/scheduler/
CPU fixes (host-buffer input layer, meta forwarding, CPU tiny split, …).  Fixes that *depend on later
blocks* still follow the relocation rule.

---

## 4. Mapping — beta patch → delivery block

`Conf.` = confidence the target is `right` (H/M/L).  `Deps` = the beta patches / blocks whose code a
patch needs.  This is the execution plan; the per-hunk `git blame` at fold time is authoritative.

| # | beta patch (abbrev) | target | Conf. | Deps | rationale / notes |
|---|---|---|---|---|---|
| 0001 | mmb: general bf16-WMMA dequant weight GEMM | **08** | H | — | the `mmb_*` family is a prefill weight GEMM; block 08 = fused-core prefill kernels, already owns `ggml-cuda.cu` fusion/convert/norm/mmvq. |
| 0002 | qsa3: packed-block WMMA sparse attention + visibility fold | **14** | H | — | QSA/Qwen4exp; block 14 introduced `fattn-qsa*`/QSA. |
| 0003 | F32/tiny-M kernels + default flips + W=1..8 probe | **08** | M | 0001 | mostly `mmb.cu`; the `qwen4exp.cpp` always-QSA flip may split to **14**. |
| 0004 | HC16: native-BF16 producers + non-temporal | **14** | M | 0001 | hyperconn/dsv4-hc are block-14 qwen4exp; the MMB activation cache is block 08. |
| 0005 | indexer: fused top-k + QSA prefill score fusions | **14** | H | 0002? | block 14 owns `indexer-topk.cu`/`lightning-indexer.cu`. |
| 0006 | mmb: RDNA4 fragment port + arch-scoped split | **08** | H | 0001 | `mmb.cu`. |
| 0007 | mmb: RDNA4 dense tile geometry + M4 inject + quant coverage | **08** | H | 0001,0006 | `mmb.cu`. |
| 0008 | mmb: per-arch tuning defaults + beneficial defaults ON | **08** | H | 0001..0007 | `ggml-cuda.cu`/`mmb.cu`/`llama-context.cpp`. |
| 0009 | mmb: routed MoE path is a loss on RDNA4 | **08** | H | 0001 | `mmb.cu`. |
| 0010 | mmb: split the F32 policies | **08** | H | 0001 | `mmb.cu`. |
| 0011 | qsa3: enable on RDNA3_0 (gfx1100) | **14** | H | 0002 | `fattn-qsa3.cu`. |
| 0012 | mmb: F32 split OFF on gfx1100 | **08** | H | 0008 | `mmb.cu`. |
| 0013 | HC: wire the `hc_gate_mix` fusion | **14** | H | 0004 | HC/qwen4exp; `ggml-cuda.cu`+`mmb.cu`. |
| 0014 | GDN/PLE: depthwise conv1d prefill fusions | **08** | M | — | matchers live in `ggml_cuda_try_fuse` (block 08's turf); GDN is block 02 but the matcher infra is block 08. May split GDN→02. |
| 0015 | norm: narrow-row RMS norm fusion | **08** | H | — | `ggml_cuda_try_fuse` matcher; generic prefill. |
| 0016 | loader: redefine `-lzm auto`; gate managed PLE reader | **14** | M | block 14 (`llama-lazy-reader`, `--lazy-buffer-size`) | mixed CLI/loader but the managed reader is a block-14 feature; candidate for **06** if it stands alone. |
| 0017 | HC16: port the HC BF16 streams (BLK16/RES16) | **14** | H | 0004 | `hyperconn.*`, `moe-weighted-reduction.*`. |
| 0018 | HC16: produce the BF16 `out_xn` copy | **14** | H | 0004,0017 | `ggml-cuda.cu`+`mmb.cu`. |
| 0019 | indexer: fuse prefill relu + head-sum | **14** | H | 0005 | `indexer-score.*`. |
| 0020 | keep F32 activations on eval callback | **08** | H | 0001 | MMB F32-elision correctness; adds `has_eval_callback`. |
| 0021 | qwen4exp: sparse MTP-draft attention | **14** | H | 0002,0005 | qwen4exp/MTP/QSA. |
| 0022 | QSA: default derived indexer cache ON | **14** | H | 0005 | `llama-memory-hybrid-idx.*`. |
| 0023 | QSA: gfx1151 decode crossover 64K→32K | **14** | H | 0005,0022 | `qwen4exp.cpp` arch policy. |
| 0024 | HC16: scope BF16 activation state per backend ctx | **08** | M | 0001,0020,0004? | MMB state in `mmb.cu`; the marks touch HC (block 14). Dependency check needed. |
| 0025 | host-buffer input layer + opt-in device placement | **06** | H | — (upstream `PER_LAYER_TOKEN_EMBD`) | block 06 is the host-buffer home; catch-all. |
| 0026 | meta: run child `graph_optimize` markings under `-sm tensor` | **06** | M | 0020,0004 (HC/MMB symbols) | catch-all by guidance, **but** its `ggml-cuda.cu` hunk touches HC16/`mmb_res16` → likely **split**: meta/impl half → **06**, CUDA half → **08/14**. |
| 0027 | MMVQ: extend routed band to W=16 + gfx1151 dense band | **13** | H | — | block 13 owns `mul_mat_vec_q_moe`/mmvq band. |
| 0028 | CPU: run tiny split graphs on the calling thread | **06** | H | — | catch-all; `ggml-cpu.cpp`, independent. |

**Target-block tallies (v1):** block 06 ← 0025,0028 (+0026?); block 08 ← 0001,0003,0006-0010,0012,
0014,0015,0020,0024 (+0026 CUDA half?); block 13 ← 0027; block 14 ← 0002,0004,0005,0011,0013,
0016-0019,0021-0023 (+0026?); block 02 ← none (0014 may split).

---

## 5. Execution checklist (per block, in order)

Work through the blocks in numeric order.  Keep every commit; checkpoint the tree after each.

- [ ] **setup** — worktree `~/llama-integration`, tag refs (`ref-delivery`, `ref-beta`), record §2 hashes.
- [ ] **block 00** — no beta assignment.  Replay original.
- [ ] **block 01** — no beta assignment.
- [ ] **block 02** — fold 0014 (GDN half) if split; else no-op.
- [ ] **block 03** — no beta assignment.
- [ ] **block 04** — no beta assignment.
- [ ] **block 05** — no beta assignment.
- [ ] **block 06** — fold **0025**, **0028**, and the meta/impl half of **0026** (catch-all).
- [ ] **block 07** — no beta assignment.
- [ ] **block 08** — fold **0001**, 0003, 0006, 0007, 0008, 0009, 0010, 0012, 0014, 0015, 0020, 0024
      (and the CUDA half of 0026 if split here).  This is the big MMB group.
- [ ] **block 09** — no beta assignment.
- [ ] **block 10** — no beta assignment.
- [ ] **block 11** — no beta assignment.
- [ ] **block 12** — no beta assignment.
- [ ] **block 13** — fold **0027**.
- [ ] **block 14** — fold **0002**, 0004, 0005, 0011, 0013, 0016, 0017, 0018, 0019, 0021, 0022, 0023
      (and any 0026 / 0003 deferrals).
- [ ] **block 15** — no beta assignment (verify no beta delta remains unmatched).
- [ ] **tree check** — `git diff HEAD mmb-beta` empty.
- [ ] **regenerate** — `make-patches.sh` + `make-release.sh`.
- [ ] **validate** — fresh `apply-all.sh` + `validate-set.sh`.
- [ ] **build + gates** — §6.
- [ ] **docs** — update `patches/README.md`, top-level `README.md`, `WORKLOG.md`, `MANIFESTS.md`,
      `release.json`, `rdna-boosts-all.patch`; mark `beta/mmb-general/` integrated.
- [ ] **commit** the record to `beta-integration` in this repo.

Status is tracked per patch in §7.

---

## 6. Validation gates (on the final tree)

Because the net tree is **byte-identical** to the already-validated beta tree, the campaign's
gfx1201/gfx1100/gfx1151 records carry over.  Still re-run the cheap, high-signal gates:

1. **Apply integrity:** `scripts/validate-set.sh` (strict apply on a fresh tarball of the base +
   tree match) — this is the CI gate.
2. **Build:** `BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` from `~/llama-integration`, 0 errors.
3. **Intra-build purity** (`BETA-TESTING.md` gate 1): width probe `PASS`; and on a dense + MoE
   model, `--spec-type none` vs `draft-mtp n3` text byte-identical (`scripts/extract-generated.py`).
4. **Oracles:** `test-backend-ops -o FLASH_ATTN_QSA` (26), `-o GATED_DELTA_NET` (46),
   `-o FLASH_ATTN_EXT` (~5954), plus `MUL_MAT`/`MUL_MAT_ID` for the MMB types on gfx1201.
5. **MTP:** `benchmarks/mtp-adaptive-methodology.md` Protocol A (acceptance > ~0.45 at pos 1,
   MTP ≥ plain at depth 3).
6. **Config dump:** `GGML_CUDA_MMB_CFG=1` prints the RDNA4 row (`dense_geom=1 … routed=0`) and,
   on a gfx1151 box, the RDNA3_5 row (`dense_geom=0 routed=1`) — the arch scoping must not have
   leaked (this is the failure mode most likely to survive a tree-identical fold, so it is cheap
   insurance; if the tree is truly identical it is the same code).

Since the tree is provably identical, gates 3–6 are a sanity re-confirmation, not new claims; the
**must-pass** gate is 1 + 2.

---

## 7. Per-patch status

| # | target | status | notes |
|---|---|---|---|
| 0001 | 08 | TODO | |
| 0002 | 14 | TODO | |
| 0003 | 08/14 | TODO | split candidate |
| 0004 | 14 | TODO | |
| 0005 | 14 | TODO | |
| 0006 | 08 | TODO | |
| 0007 | 08 | TODO | |
| 0008 | 08 | TODO | |
| 0009 | 08 | TODO | |
| 0010 | 08 | TODO | |
| 0011 | 14 | TODO | |
| 0012 | 08 | TODO | |
| 0013 | 14 | TODO | |
| 0014 | 08/02 | TODO | split candidate |
| 0015 | 08 | TODO | |
| 0016 | 14/06 | TODO | |
| 0017 | 14 | TODO | |
| 0018 | 14 | TODO | |
| 0019 | 14 | TODO | |
| 0020 | 08 | TODO | |
| 0021 | 14 | TODO | |
| 0022 | 14 | TODO | |
| 0023 | 14 | TODO | |
| 0024 | 08 | TODO | dep check |
| 0025 | 06 | TODO | |
| 0026 | 06/08/14 | TODO | split candidate |
| 0027 | 13 | TODO | |
| 0028 | 06 | TODO | |

---

## 8. Risks & open questions

- **Intermediate block buildability.**  Folding a patch earlier than its dependencies can yield an
  intermediate commit that does not compile even though the final tree does.  Mitigation: the
  relocation rule (§3) and, if needed, an intermediate `cmake --build` smoke check per block.  A
  non-compiling intermediate is *acceptable* for apply integrity but undesirable; prefer relocating.
- **`qwen4exp.cpp` is touched by both block 08 (MMB flips) and block 14.**  The always-QSA /
  F32 / QSA-crossover defaults could land in either; pick by feature (QSA stays in 14) and split
  0003 if it improves coherence.
- **`ggml-cuda.cu` is touched by nearly every block (08,10,13,14,15).**  Conflict resolution must be
  careful; always diff the resolved file against the beta tree's version before committing.
- **`0026` dependency tangle** (MMB/HC symbols dated to patches 0004/0020): the catch-all home is 06
  but the CUDA half cannot be replayed there.  Expect a split; record the deviation.
- **Subject rewording.**  Amended block subjects should describe the combined feature without
  pretending the beta work is original delivery work; keep the beta provenance in the commit body
  (e.g. "folds beta 0001/0003/… of `beta/mmb-general`").
- **Docs freshness.**  After the fold, `beta/mmb-general/*.md` describes a set that is no longer
  applied separately; mark it integrated rather than deleting it (see the repo's WIP/promotion
  rules).

---

## 9. Progress log (newest first)

- **2026-09-25 (execution)** — the fold was executed in `~/llama-integration` (branch
  `beta-integration`, tip `f373450de…`) and the 16 amended patches were produced.  Mapping used:
  block 06 <- 0025/0028; block 08 <- 0001/0003/0006-0010/0012/0014/0015; block 13 <- half of 0001;
  block 14 <- half of 0001 + 0027; block 15 <- 0002/0003-qwen4exp/0004/0005/0008/0011/0013/
  0016-0024/0026.  Final tree == `24bb0f5acb…` (beta reference), strict `git am` 16/16,
  `validate-set.sh` PASS, build EXIT 0.  See §0 for the full record.
- **2026-09-25** — branch `beta-integration` cut from `main` (`0699a3d`); this plan written.
  Reference state recorded (§2): base `84e76d8a2` / tree `5112eedb…`, delivery r7 tree
  `7726e514…`, beta tip tree `24bb0f5acb…`, fork branches `rdna-boosts` / `mmb-beta`.  Mapping v1
  drafted (§4).
