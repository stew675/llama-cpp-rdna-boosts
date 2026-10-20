# `wip/nwarps` — promotion handover (fold into a block, then release)

**Audience: the main development host** (which owns `patches/`, `release.json` and the release flow).
Everything here was validated on the RDNA3_0 box (gfx1100, RX 7900 XTX, ROCm 10.1.0); the gates that
must be re-run elsewhere are in §5.

Read [`README.md`](README.md) first for the finding and the measurements.  This file is only about
turning the campaign into a delivery block.

| artifact | sha256 | use |
|---|---|---|
| [`patches/per-shape-nwarps-rdna3-0-promoted.patch`](patches/per-shape-nwarps-rdna3-0-promoted.patch) | `55a7cd76…` | **the one to fold** — single commit, default-on, applies to the current `release.json` tree (`git am` verified on a fresh worktree) |
| [`patches/per-shape-nwarps-rdna3-0.patch`](patches/per-shape-nwarps-rdna3-0.patch) | `eb32d36c…` | the same change with the knob **default-off**.  For an A/B, a beta test, or a bisect of *this* feature.  Do **not** fold this one (it is inert unless the env var is set) |

The two patches differ **only** in the `getenv` default (`0` vs `8192`) and its comment.  The promoted
tree was verified on gfx1100: with the environment unset it produces exactly the measured
`GGML_CUDA_MMVQ_RDNA3_SMALL_M=8192` result (width-probe row-0 hash `9712377b5b34f8f3`, PASS, 35B
`tg128` @ d16384 **127.26 t/s**, gemma-26B **131.32 t/s**), and `GGML_CUDA_MMVQ_RDNA3_SMALL_M=0`
reproduces the delivery (row-0 hash `7e21f25829185919`, PASS) — so the kill-switch is intact.

## 1. What the patch contains

Three files, `ggml/src/ggml-cuda/`:

| file | block that **creates** it | blocks that also modify it |
|---|---|---|
| `mmvq-rdna.cuh` | **08** (fused kernels) | 08 only |
| `mmvq-rdna-params.cuh` | **08** (fused kernels) | 08 only |
| `mmvq.cu` | upstream | 07, **08**, 09, 10 |

So the change belongs in **block 08** (it edits the two policy headers that block creates, and the
ksplit dispatch + the three fusion ops — `ssm_gate_beta`, `ssm_conv_l2_gatebeta`, `shexp_down_gate` —
that block 08 owns).  Splitting it across blocks is not worth the churn: the two headers cannot go
anywhere but 08.

## 2. Folding it

```sh
# canonical tree, built the normal way
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <release.json.base>
bash <delivery>/scripts/apply-all.sh .            # 12 commits, branch rdna-boosts
git am <delivery>/wip/nwarps/patches/per-shape-nwarps-rdna3-0-promoted.patch   # 13th commit
```

Then fold the 13th commit into block 08.  Either is acceptable:

* **Preferred — amend into block 08** (`git rebase -i <block-08>^`, mark the 13th commit `fixup`).
  Blocks 09/10 replay on top; their `mmvq.cu` hunks are in different regions (they touch the fusion
  op bodies and the qwen4exp paths, this patch touches `calc_nwarps`, the ksplit template, the ksplit
  switch, the ncols dispatch and the three fusion-op *front ends*), so a conflict there is mechanical —
  keep both sides.  Rebuild the series and confirm the applied tree equals the pre-fold tree.
* **Acceptable — keep it as its own commit** on the chain (the delivered tree is what ships; block
  attribution is bookkeeping).  Precedent for multi-block attribution: PR #167 went into blocks 07/10.

Then regenerate, never hand-edit:

```sh
bash scripts/make-patches.sh          # re-cuts patches/ from the block commits
bash scripts/make-release.sh          # bumps release.json (fork point, tip/tree, per-artifact sha256)
bash scripts/validate-set.sh          # checksums + fresh-tarball strict apply (this is also CI's gate)
```

## 3. Doc edits that belong with the fold

### 3.1 `ENVIRONMENT.md` — §7.4 "Other kernel tuning", one row (ready to paste)

```
| `GGML_CUDA_MMVQ_RDNA3_SMALL_M` | `8192`, RDNA3_0 only | **kill-switch / tuning** (block 08) | row-count `M` band for the dense mmvq *weight* kernel's narrow (1-warp) block.  `Q4_0/Q4_1/Q5_0/Q5_1` always take it; `Q8_0` takes it for `2048 <= M <= this`; `Q6_K`/`IQ4_NL` never do.  `M` is the weight's row count, i.e. a tensor property, so all widths `W = 1..8` of a tensor take the same branch — and the three fusion kernels that reproduce a dense weight launch (`ggml_cuda_op_ssm_gate_beta`, `_ssm_conv_l2_gatebeta`, `_shexp_down_gate`) take the same value, which is the width-purity requirement (`GREEDY-PURITY.md` §19).  `0` restores the pre-2026-10-19 wide-block table (byte-identical to r91).  Measured on gfx1100: 35B-A3B MTP n3 **+10.7 %** (prose) / **+6.3 %** (code), gemma-26B `tg128` **+5.0 %**, 35B-A3B `tg128` @ d16384 **+1.7-2.0 %**, gemma-12B +0.4 %, 27B neutral; `W = 1..8` probe PASS on all four models and all eight KV types.  Record: `wip/nwarps/` |
```

Also add the variable to the "you do not need to set anything" mental model implicitly (it is a
kill-switch, not an opt-in).

### 3.2 `GREEDY-PURITY.md` — the invariant this campaign paid for

Add to the per-shape-launch-parameter section (§19 band uniformity / §25 pinned fusion anchors):

> **A shape-scoped launch parameter must be forwarded to every kernel that reproduces that launch.**
> `nwarps` / `rows_per_block` / VDR are chosen per weight shape, which is width-uniform *only* if
> every kernel that computes the same tensor uses the same value.  The fused single-token kernels
> (`ggml_cuda_op_ssm_gate_beta`, `ggml_cuda_op_ssm_conv_l2_gatebeta`, `ggml_cuda_op_shexp_down_gate`,
> the rms_norm→Q8_1 fold, the gate/bias folds) each *replace* a dense mmvq weight launch, so they must
> call the same selector (`calc_nwarps_weight(type, ncols, table, long_k, <row scope>)`).  A scope that
> is applied to the plain launch but not to the fused one runs `W = 1` at one warp count and `W = 2..8`
> at another — a decode-vs-verify impurity with exactly the symptom this rulebook exists to catch
> (measured: 35B-A3B 0.150, gemma-26B **3.35** on the `W = 1..8` probe).  The ksplit K loop
> (`kbx = tid/(qi/vdr) + m*blocks_per_iter`) is a fixed per-thread residue class, so a *correctly
> forwarded* per-shape value needs no further derivation.

### 3.3 `patches/README.md` — block 08 note

One line under the block-08 notes: the RDNA3_0 dense mmvq weight kernel now picks `nwarps` per weight
shape (`GGML_CUDA_MMVQ_RDNA3_SMALL_M`, default 8192, `=0` restores the table), and why the fusion
kernels must share the value (pointer to `GREEDY-PURITY.md` §19/§25).

### 3.4 `wip/nwarps/` after promotion

Per the WIP rule: move the whole directory to `archive/work/nwarps/`, record the promotion in
`WORKLOG.md`, and update `wip/CAMPAIGNS.md` (remove the row, no stub left behind).

## 4. Expected results on RDNA3_0 (acceptance evidence)

`test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512`, `HIP_VISIBLE_DEVICES=0`,
`KV=f16` unless noted — promoted build, environment unset:

| model | result |
|---|---|
| Qwen3.6-35B-A3B-UD-Q4_K_M | PASS `maxdiff 0`, row-0 `9712377b5b34f8f3` |
| gemma-4-26B-A4B QAT UD-Q4_K_XL | PASS `maxdiff 0` |
| gemma-4-12B Q8_0 | PASS `maxdiff 0` |
| Qwen3.8-27B-UD-Q4_K_M | PASS `maxdiff 0` |
| 27B × `f16 bf16 q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl` | PASS `maxdiff 0` |

Perf (`llama-bench -p 0 -n 128 -r 3`, one GPU): 35B-A3B `tg128` d16384 **125.1-125.6 → 127.2-127.4**
(+1.7-2.0 %), gemma-26B d16384 **125.1 → 131.3-131.5** (+5.0 %), gemma-12B d0 **54.05 → 54.28**
(+0.4 %), 27B d16384 40.15 → 40.15 (neutral).  MTP n3 (35B-A3B, `-n 400 --temp 0`): prose
**200.6 → 222.0 t/s**, code **235.5 → 250.2 t/s**; acceptance 0.653→0.750 / 0.801→0.851, so part of the
MTP gain is an acceptance shift and the *plain*-decode numbers are the kernel-time-only claim.
PPL (wikitext-2, 64 × 2048, `-fa on`) 5.7861 both ways.

## 5. Gates to run before the release

1. **gfx1201 (primary box) must be byte-identical to r91.**  The whole change is
   `table_id == MMVQ_PARAMETERS_RDNA3_0`-gated, so on RDNA4 the width probe, the coherence text and
   PPL must not move *at all*; the only expected delta is the HIP object size (+12.6 % on the gfx1100
   object; measure it on gfx1201 too).  A diff here means the scope leaked.
2. **gfx1151 (Strix Halo)**: build + smoke.  RDNA3_5 has its own table and is untouched; the release
   gates still need the usual `test-recurrent-state-depth` per-arm FAIL/PASS set and a decode number.
3. **gfx1100 (RDNA3_0)**: §4 above — the `W = 1..8` probe on all four models (all eight KV types on at
   least one), the adaptive-MTP four-axis gate (`benchmarks/mtp-adaptive-methodology.md`, `-n 3000`,
   reasoning pinned, R/P/C/K + the phase-switching prompt), the stock-relative
   `llama-batched-bench -npl 1,4,8` gate, PPL parity, and `scripts/gate-prefill-logits.sh`
   (expected no-op — mmvq never runs in prefill — but it is the release gate).
4. **Bloat record**: `mmvq.cu.o` 13.64 → 15.36 MiB, `mul_mat_vec_q_ksplit` symbols 2188 → 2640
   (+20.7 %; the ungated original was +100 %).  Add the gfx1201 numbers to the block note if the
   object-size trend matters there.
5. **Kill-switch sanity**: `GGML_CUDA_MMVQ_RDNA3_SMALL_M=0` must reproduce the r91 tree exactly on
   gfx1100 (hash `7e21f25829185919` on the 35B probe) — that is the bisect handle if a RDNA3_0 user
   reports a decode regression.

## 6. Release mechanics

Nothing campaign-specific: follow `AGENTS.md` "Pushing policy" and "Build the release container
images".  In short — tag, push the tag, refresh the personal fork's `rdna-boosts` branch from a clean
`apply-all.sh` at `release.json.base` (build it, run the coherence/MTP/prefill-logit gates, force-push
`--force-with-lease` to the **fork only**), and let the worker watcher on `radiance` publish the ROCm
7.14.1 / 10.1 images.  The block-08 note, `ENVIRONMENT.md` and `GREEDY-PURITY.md` edits go in the same
release.

## 7. Rollback

The feature is one `getenv` default.  If a box regresses: set `GGML_CUDA_MMVQ_RDNA3_SMALL_M=0` to
confirm the cause, then either flip the default back to `0` (a one-line block-08 amend, exactly the
`patches/per-shape-nwarps-rdna3-0.patch` state) or narrow the rule (e.g. drop the always-narrow
`Q4_0/Q4_1/Q5_0/Q5_1` arm, which rests on gemma-26B alone and reverses a 2026-08-28 *shallow* sweep —
the most likely place for a surprise on an untested RDNA3_0 model).  Do not re-add the `!has_fusion`
condition (§2 of `README.md`).
