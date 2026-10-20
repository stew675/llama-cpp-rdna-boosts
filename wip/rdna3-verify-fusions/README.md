# rdna3-verify-fusions: the verify-band fusions on RDNA3_0 too (PR #192 + PR #193)

**Status: ACCEPTED as a promotion candidate (2026-10-20).**  Nothing here is applied to the delivery.
This directory is a self-contained handover for the next block-set release.  The fold recipe is
`PROMOTION.md`; the prompt for the main development box is `HANDOVER-PROMPT.md`.

Origin: two contributor PRs to this repository by `briansp2020` (Brian S. Park):

- **PR #192** (`wip/rdna3-verify-band-q8`): admits RDNA3_0 (gfx1100-1102) to the existing 2..8-token
  verify-band Q8_1 pre-quantize fusions (`residual ADD + rms_norm -> Q8_1`, `rms_norm -> Q8_1`,
  `gated unary -> Q8_1`), which were RDNA4-only.
- **PR #193** (`wip/rdna3-gdn-verify-fusions`): admits RDNA3_0 to the direct GDN conv for 2..255-token
  verify batches and to the batched run of f32 `GGML_OP_CPY` nodes, both of which were RDNA4-only.

The patches here are the submitted changes re-cut against the **current release tree**
`v12-097cf33d9-r93` (`release.json`: base `097cf33d9`, tree `260e6f49f41b9dbd18ce1b2d5214eba1917e8752`).
They apply cleanly with `git am -3`; the only difference from the submitted files is the base commit
metadata.  They are independent and can be folded separately.

## What the patches change

`0001` (PR #192), 3 files, +5 / -4:

- `mmvq.cu`: `ggml_cuda_should_fuse_mul_mat_vec_q(tensor, verify_band = true)` gains
  `GGML_CUDA_CC_IS_RDNA3_0(cc)` alongside `GGML_CUDA_CC_IS_RDNA4(cc)`.  The `verify_band` callers only
  pre-fill the mmvq Q8_1 activation cache; their kernels are row-generic and write the values
  `quantize_q8_1` writes.
- Comment updates in `ggml-cuda.cu` and `ggml-cuda-try-fuse.cu`.
- `GGML_CUDA_FUSE_Q8_1_VERIFY=0` still turns the band fusions off; RDNA3_5 stays out (not measured);
  RDNA4 is unchanged.

`0002` (PR #193), 3 files, +6 / -4:

- `gdn-conv.cu`: the direct GDN conv (same 4-tap order and silu; `gdn_concat_tail` still materializes
  the snapshot tail) takes the 2..255-token verify batches on RDNA3_0 as well.  The pre-existing
  single-device restriction is kept.
- `cpy-batch.cu` / `cpy-batch.cuh`: the run of up to 16 consecutive same-layout f32 `GGML_OP_CPY`
  nodes in one launch is admitted on RDNA3_0 as well.
- `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` / `GGML_CUDA_FUSE_CPY_BATCH=0` still turn them off; RDNA3_5 stays
  out; RDNA4 is unchanged.

## Independent validation (gfx1100, this box, 2026-10-20)

Box: RX 7900 XTX (gfx1100, 24 GiB), ROCm 10.1.0-gfx1100, single visible GPU
(`HIP_VISIBLE_DEVICES=0`).  Build: `~/bin/build-llama-rocm` (`-DGPU_TARGETS=gfx1100`).  Reference is
the r93 `build-rocm` tree (build `b11583-82c9e6495`); candidate is `build-pr192-193` (build
`b11585-cce5e5c11`).  Model: Qwen3.8-27B-UD-Q4_K_M, `-fa 1 -ctk q8_0 -ctv q8_0 -t 8`, unless noted.

**Bit-identity.**

- `test-logits-width-probe` (the delivery's W=1..8 matrix), `q8_0` KV:
  - 27B, prefix 256: row 0 `c37241bd00c1d561`, all eight hashes identical across the two builds,
    `width_purity=PASS (worst maxdiff 0)`.
  - 27B, prefix 3584: row 0 `86541410b7c618b0`, PASS, maxdiff 0 on both builds.
  - Qwen3-30B-A3B Q4_K_M (MoE control), prefix 256: row 0 `899608bf91600e1b`, PASS, maxdiff 0 on both.
  - Qwen3.6-35B-A3B Q4_K_M (GDN hybrid MoE), prefix 256: row 0 `423a5f353f0a0e45`, PASS, maxdiff 0 on
    both.
- Same-binary kill-switch A/B on 27B, prefix 256: default, `GGML_CUDA_FUSE_Q8_1_VERIFY=0`,
  `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0`, `GGML_CUDA_FUSE_CPY_BATCH=0`, and all three off all produce the
  same per-W hashes and row 0 `c37241bd00c1d561` (maxdiff 0).  The debug switches confirm the paths
  actually fire (`GDN_CONV_DIRECT C=10240 T=2 ...`, `cpy_batch: 2 copies ...`).
- Greedy text, 400 tokens, seed 42, temp 0, `--no-display-prompt --single-turn`, extracted with
  `scripts/extract-generated.py`: prose `698b60ce0b5d` and code `c5a4ef044b64` for **both** builds and
  for **both** `--spec-type none` and `--spec-type draft-mtp --spec-draft-n-max 3`.
- `test-backend-ops -o RMS_NORM,ADD,MUL_MAT,CPY,CONCAT,SSM_CONV`: **3130/3130 passed on ROCm0** (the
  dGPU).  The iGPU (`ROCm1`, gfx1036) aborts in `ggml_cpy_scalar_cuda` with "invalid kernel file",
  before and independent of this patch; run with `HIP_VISIBLE_DEVICES=0` or `-b ROCm0`.
- Prefill-logit gate: **mean KLD 0.000000, same-top-p 100.000 %** (PASS; identical to the r93 base).
- `scripts/check-gpu-faults.sh --since "3 hours ago"`: clean on the A/B and batched-bench logs.

**Performance, combined #192 + #193.**  Interleaved 4-round A/B, dense 27B, fixed `draft-mtp` n-max 3,
400 tokens, order flipped each round:

| arm | prose (t/s) | code (t/s) |
|---|---|---|
| r93 reference | 65.3 / 65.8 / 65.9 / 65.9 | 68.3 / 68.5 / 68.5 / 68.5 |
| r93 + both PRs | 68.6 / 68.5 / 68.6 / 68.7 | 71.6 / 71.6 / 71.4 / 71.5 |
| mean delta | **+4.4 %** | **+4.5 %** |

Plain (one-token) decode is flat, as expected (no verify band): prose 41.4 -> 41.3, code 42.0 -> 42.1.
The individual PR claims (+1.9 % for #192, +1.3 % for #193) are the author's per-patch numbers on
r91; the combined effect on r93 is about +4.4 % on this box.

**Stock-relative verify-width gate** (`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, dense 27B,
`q8_0` KV, 3 interleaved rounds): B=1 flat; B=4 and B=8 are **not worse**, and are about +1 to +3 %
after the first (cold) round.  The first reference round was visibly cold (its PP was also low), so
treat the B=4/B=8 deltas as "no regression, small gain", not as a headline.

## Limitations and open gates

- Measured on **gfx1100 only**, single GPU.  gfx1101/gfx1102 are inside the macro but were not tested
  by the author or here.  **gfx1151 (RDNA3_5) is deliberately excluded** and is unchanged by design.
- The GDN conv fusion keeps its pre-existing single-device restriction.  The **batched f32 CPY is not
  single-device restricted**, and it was not measured on a multi-device split here (this box has one
  dGPU).  A `-sm tensor` / `-sm layer` RDNA3_0 run with a GDN model and MTP is the remaining
  correctness check for that half.
- On a host with a visible iGPU plus a dGPU, `ggml_backend_cuda_get_device_count()` is 2 unless
  `HIP_VISIBLE_DEVICES` hides one, so the GDN conv fusion is disabled.  That is pre-existing behaviour,
  not introduced by this patch; the measurements above use `HIP_VISIBLE_DEVICES=0`.
- The full release gates on the folded tree (four-axis MTP at `-n 3000`, prefill-logit,
  `validate-set.sh`, gfx1201 scope check, container/fork refresh) are listed in `PROMOTION.md` and must
  be run on the main box.  This directory's note is the gfx1100 acceptance evidence, not a release
  sign-off.

## Files

- `0001-mmvq-the-verify-band-Q8_1-fusions-on-RDNA3_0-too.patch` (PR #192, re-cut on r93)
- `0002-cuda-the-GDN-conv-verify-path-and-the-batched-f32-CP.patch` (PR #193, re-cut on r93)
- `PROMOTION.md`: fold targets, environment/doc updates, release gate checklist
- `HANDOVER-PROMPT.md`: the prompt to give the main development box
