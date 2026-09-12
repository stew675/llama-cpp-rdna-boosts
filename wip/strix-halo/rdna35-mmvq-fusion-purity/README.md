# RDNA3_5 single-token-only mmvq fusions are not decode/verify bit-identical

> **FOLDED 2026-09-12 into block 13** (canonical tip `13af95ac1`, tree
> `f4791066f4a582316b1ca95f51c96cd10b905ef7`).  This directory is the pre-fold record; the
> standalone `.patch` here is superseded by the regenerated `patches/0013`.  See
> `GREEDY-PURITY.md` §25 and the 2026-09-12 (2) `WORKLOG.md` entry.  What actually landed in block
> 13: the dense-GLU guards stay in `ggml-cuda.cu` (six matcher sites), but the weighted-down gate
> lives in `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok` (`mmvq.cu`, block 13) rather than the
> block-14 `try_fuse` matcher, so the whole fix is block-13-local.

Date: 2026-09-11.  Box: Strix Halo APU, Radeon 8060S (**gfx1151**), 1 device,
ROCm 7.14 (`/opt/rocm-7.14-gfx1151`).  Fork branch `qwen4exp-investigations`
(commit `ab4444eb2`, forked from the delivery + block-15 tip `24920605d`).
Delivery-pending patch: [`rdna35-single-token-mmvq-fusions.patch`](rdna35-single-token-mmvq-fusions.patch).

Found while investigating the reported qwen4exp MTP incoherence at
`--spec-draft-n-max > 7`.  The `n_max > 7` part is the separate `QSA_DECODE_BAND` /
FA-family band (see `GREEDY-PURITY.md` §11/§16); **this** item is the second,
independent width dependence: it breaks draft-vs-verify purity on gfx1151 even at
`n_max <= 7`, and it is already tracked in the issue25 investigation as the
"block-13 `n_q=1` short-K mmvq variance"
(`wip/strix-halo/issue25/GATE-2026-09-11-block00-rdna35.md` §5,
`wip/strix-halo/issue25/RECORD-2026-09-11-issue25.md` §5.1).

This document is the localisation, root-cause and fix; it was folded into **block 13** on
2026-09-12 (§6 preserves the placement rationale).

## 1. Symptom / mechanism

The fork's central MTP guarantee is that a 1-token decode and an `n_max+1`-token
speculative verify of the same layer compute **bit-identical** logits, so the
draft steps and the verify agree.  On gfx1151 two **single-token-only** `ggml_cuda_try_fuse`
fusions run at `W=1` but not at `W>=2`, and their fused kernels do **not**
reproduce the standalone `mul_mat_vec_q` / `mul_mat_vec_q_moe` arithmetic:

1. **`ggml_cuda_mul_mat_id_weighted_rdna3_5`** — the MoE down-tail fusion
   (`mmvq.cu`, matcher in `ggml-cuda.cu`).  RDNA3_5-only and single-token by its
   shape fingerprint (`ggml_nelements(y) == 640 * n_used`).  Introduced by
   blocks 13 (kernel) / 14 (try_fuse matcher).
2. **The dense gate+up+GLU mmvq fusion** — `mul_mat_vec_q<..., ncols=1, has_fusion=true>`
   in the `{MUL_MAT, MUL_MAT, GLU}` matchers.  `mmvq.cu` restricts fusion to
   `ncols_dst == 1` (`GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1")`),
   so it fires at `W=1` only.  The matchers and `ggml_cuda_should_fuse_mul_mat_vec_q`
   are **upstream at the fork point** (`9113cc188`); block 13's mmvq item-split
   rewrite changed the `n_q=1` path they route through.

The other `W=1`-only arms were ruled out and stay enabled (they are bit-identical
to their unfused reference): dual-output K/V (`Vcur`), SSM conv-input, the GDN
`ssm_gate_beta` fusion, the L2-norm pair, HC (`hc_mix`/`hc_combine` fire at every
width in the band), and the MoE gate+up+GLU (`MUL_MAT_ID`, which fires at every
width and is band-uniform after block 13's fixes).

## 2. Evidence (width probe, qwen4exp UD-IQ4_XS, `P=100`, `RS=0`, 1 GPU)

| config | `W=1` | `W=8` |
|---|---|---|
| default | `8abc6206d1e80709` | `453eaa618738273d` |
| `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1` | `8d036a7b8c8a5ce5` | `453eaa618738273d` |
| dense-GLU fusion disabled | `969940599c5426e9` | `453eaa618738273d` |
| **both disabled** | **`453eaa618738273d`** | **`453eaa618738273d`** |
| `GGML_CUDA_DISABLE_FUSION=1` | `5a7e4c21e86e34c2` | `5a7e4c21e86e34c2` |

Each fusion independently shifts `W=1`; only both together make `W=1 == W=8`
(and equal to the `W=8` standalone reference).  `GGML_CUDA_DISABLE_MMVQ_MAT`
(all dense `MUL_MAT` mmvq fusions) + `GGML_CUDA_DISABLE_WEIGHTED_DOWN` also
reconciles — confirming the whole gap is inside the mmvq fusion family.

## 3. Fix

`ggml/src/ggml-cuda/ggml-cuda.cu` only (from context: this tree has block 15 on
top; block 15 does not touch `ggml-cuda.cu`, and the file is byte-identical at
blocks 14 and 15):

- new `ggml_cuda_rdna3_5_single_token_fusions_disabled()` — true on RDNA3_5
  unless `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` (A/B opt-in);
- the `weighted_rdna3_5` matcher gets `&& !…disabled()`;
- the six gate+up+GLU matcher sites get
  `&& (ids != nullptr || !…disabled())`, so **`MUL_MAT_ID` (MoE) fusions are
  untouched** — only the dense GLU is skipped.

The pure dense fusions (K/V dual-output, etc.) stay enabled; `GGML_CUDA_DISABLE_FUSION=1`
still works as the broad control.

## 4. Validation (post-fix `W=1,2,4,8` one hash per `(model, KV)`)

| model | KV | pre-fix `W=1` / `W=8` | post-fix |
|---|---|---|---|
| qwen4exp UD-IQ4_XS | f16 | `8abc6206` / `453eaa61` | all `453eaa61` |
| qwen4exp UD-IQ4_XS | q8_0 | impure | all `113696b9` |
| MoE 35B-A3B Q4_K_M | f16 | `1717b756` / `18999a78` | all `18999a78` |
| 27B Q8_0 dense | f16 | `e165ef98` both | `e165ef98` both (unchanged) |

`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1` reproduces the pre-fix
impurity exactly (clean A/B).  A fusion trace confirms the `W=1`-only arms are
exactly `n=21 MUL_MAT_ID(ffn_moe_down)` and `n=3 MUL_MAT(ffn_gate)`.

## 5. Performance

`llama-bench` qwen4exp UD-IQ4_XS, 1 GPU, `-p 512 -n 128 -r 2`:

| | pp512 | tg128 |
|---|---|---|
| fix | 646.02 ± 0.25 | **25.53 ± 0.02** |
| opt-in (pre-fix) | 647.31 ± 3.85 | **25.77 ± 0.01** |

≈ **−0.9 % tg128**, prefill flat.  Per the §19 purity-first policy this is the
accepted trade; the follow-up that would restore it is to make the fused kernels
reproduce the standalone reduction (pin `nwarps`/`rps`/item-split for the
fused `ncols_dst==1` path) rather than to skip the fusion.

## 6. Placement — where this belongs (and why not block 00)

> **Folded into block 13** on 2026-09-12 (see the header note).  The recommendation below is the
> pre-fold analysis that was followed.

**Not block 00.**  Two reasons:

1. **Apply order / provenance.**  Block 00 is generated from the fork point
   (`9113cc188`) and currently touches only `ggml/src/ggml-cuda/fattn-common.cuh`
   and the Vulkan `flash_attn*.comp` shaders.  The `weighted_rdna3_5` matcher
   does **not exist** at the fork point (`git show 9113cc188:.../ggml-cuda.cu |
   grep weighted_rdna3_5` → 0); block 14 adds it.  A block-00 hunk guarding it
   has no context and cannot apply.  Only the dense-GLU half would physically
   fit, which would split one invariant across two blocks.
2. **Repo classification.**  The issue25 record already separates the two:
   block 00 owns the **FA** structural width fix (`parallel_blocks`/`Q->ne[1]`,
   since re-homed to `fattn-common.cuh`); the **mmvq** width variance is tracked
   as a block-13 regression ("block 13 re-breaks it for pure attention",
   "the block-13 `n_q=1` short-K mmvq variance").  The fix belongs with the block
   that owns the code, i.e. **fix the regression in the block that introduced it**.

**Recommended home: block 13** (the mmvq item-split / band-uniformity block, and
the block whose rewrite left this gap).  To make the *whole* fix a block-13
amendment, move the weighted-down gate from the try_fuse matcher (block 14) to
inside `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok` (`mmvq.cu`, block 13) — the
matcher calls `_ok`, so the effect is identical, and the helper then needs to be
visible in `mmvq.cu` (declare it in a shared header, or repeat the tiny
arch+env check).  Block 13 also already modifies `ggml-cuda.cu`, so the dense-GLU
half fits there.

Alternative single home: **block 14** (`qwen4exp support`), which owns the
weighted-down try_fuse matcher; it can guard both matcher families without the
helper move.  Do **not** re-home the weighted-down kernel back to block 13 just
for this — the matcher and kernel were split across 13/14 deliberately.

If the maintainer prefers to centralise the *policy* in block 00 instead, the
workable shape is: block 00 defines the helper + the invariant, block 08 guards
the dense GLU, block 14 guards the weighted-down.  That is more churn and was
not the issue25 precedent, so block 13 is the recommendation.

## 7. Fold-in checklist (when the gfx1201 work lands)

1. Apply `rdna35-single-token-mmvq-fusions.patch` to `main` (it applies to
   `ggml-cuda.cu` context that block 15 does not modify).
2. Amend **block 13** (preferred) — move the weighted-down gate into `_ok`
   (`mmvq.cu`) and keep the dense-GLU guards in `ggml-cuda.cu`; or amend
   **block 14** as-is.
3. Re-run `scripts/make-patches.sh` from a canonical fork at `9113cc188`;
   confirm strict 15/15 `git am` + 0 whitespace warnings.
4. Re-cut the block-15 beta patch on the new base.
5. Update `GREEDY-PURITY.md` (new section, next number after §20), the block-13
   notes in `patches/README.md`, and a dated `WORKLOG.md` entry.
6. Re-gate: gfx1151 probe `W=1..8` (qwen4exp f16/q8_0, MoE, dense),
   greedy `plain == n_max 3 == n_max 7`, MTP acceptance; gfx1201 must be
   byte-identical (the gate is `GGML_CUDA_CC_IS_RDNA3_5`-only).

## 8. Repro

```sh
# probe (wip/kv-quant-purity-followups/tools/logits-dump-kv.cpp), 4B/1-GPU style
clang++ -O2 -std=c++17 -I <tree>/include -I <tree>/ggml/include \
  logits-dump-kv.cpp -o /tmp/lw -L <tree>/build/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,<tree>/build/bin
HIP_VISIBLE_DEVICES=0 W=1 CTK=q8_0 CTV=q8_0 RS=0 CB=0 SPLIT=layer \
  /tmp/lw <qwen4exp.gguf> <text.txt> 100 512      # default: 453eaa61...
# GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1 -> 8abc6206 (the pre-fix W=1)
```

The gfx1151 width instrument from the issue25 work
(`wip/strix-halo/issue25/logits-width.cpp`) measures `max|W1-W3|` / `max|W3-W5|`
directly and is the natural gate to add this fix to.
