# gdn-conv-norm-deferred-gather: the conv + q/k norm fusion on a deferred conv-state gather

**Status: candidate (contributor patch), not part of the delivery.**  One patch, cut against the release tree
`v16-4f9dfa0e3-r289` (`release.json`: base `4f9dfa0e3`, tree `7181d396977e90270528d8478ee7b04894a3f39e`); it
applies with a strict `git am` on that tree.

| file | sha256 |
|---|---|
| `0001-gdn-conv-the-conv-q-k-norm-fusion-is-no-longer-refus.patch` | `5cba0d89d74364c5f68015ff5b0bffa69ba11c921fee272ca1b7352b56e3c0bf` |

## What it fixes

`gdn_conv_norm_match` folds the Gated DeltaNet q / k l2-norm pair (`RMS_NORM` + `SCALE`, run as
`rms_norm_scale_pair_f32<256>`) into the direct conv kernel.  Before folding it refuses any layout where a
`SCALE` output overlaps `x`, `state`, `w` or `conv_out`, because the kernel writes the normalized outputs while
it still reads those buffers.

On the speculative verify path (2..16 tokens with snapshots) the conv-state `GET_ROWS` is a **deferred gather**
(`ggml_cuda_gdn_conv_gather_deferrable`): `ggml_cuda_gdn_conv_direct` reads the cache row through `state_ids`
and never reads `m.state->data`.  The allocator then legitimately places the q `SCALE` output over that dead
`GET_ROWS` output, the overlap test fires on `m.state`, and the fusion is refused.  The two norms then run as a
separate kernel plus one launch in every Gated DeltaNet layer of every verify step.

Observed on a Radeon AI PRO R9700 with Qwen3.8-27B UD-Q4_K_M and DFlash2: every `T = 8` verify call was refused
on `q overlaps state`, every one of them with a deferred gather.

The patch passes the CUDA context to `gdn_conv_norm_match` (from `ggml_cuda_gdn_conv_try_fuse`) and skips the
overlap test against `m.state` when that state's root is in `ctx.gdn_deferred_gathers`.  The overlap tests against
`x`, `w` and `conv_out` are unchanged.  `gdn_conv_decode_ok` still calls the matcher without a context, so the
single-token decide is unchanged.

## Numerics

The fused kernel is already documented as bit-identical to `rms_norm_scale_pair_f32<256>` (same values in the
same lanes).  Checked on gfx1201:

- `test-backend-ops` `SSM_CONV`, `GATED_DELTA_NET`, `RMS_NORM`, `MUL_MAT`: no failure;
- greedy (temperature 0) generation with DFlash2 is byte-identical to the non-speculative run of the same build,
  and to the run without the patch.

`GGML_CUDA_FUSE_GDN_CONV_NORM=0` still disables the fusion.

## Measured

Radeon AI PRO R9700 (gfx1201, ROCm 10.1), Qwen3.8-27B UD-Q4_K_M, DFlash2 `--spec-draft-n-max 7`, 256 tokens,
temperature 1, two runs of 15 seeds each: the verify cycle goes from **43.49 to 42.74 / 42.86 ms** (−0.7 ms),
acceptance unchanged (0.694 in all four runs).  Measured on the delivery of commit `81e969e` (2026-11-03) with the
fusion toggled by `GGML_CUDA_FUSE_GDN_CONV_NORM`; the matcher is unchanged between that tree and r289.

Only gfx1201 was available here; the fix is not architecture-specific.
