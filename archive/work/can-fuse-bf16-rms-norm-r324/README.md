# can-fuse-bf16-rms-norm-r324: BF16 RMS_NORM + MUL (+ ADD) aborts the fusion check

**Status: CLOSED at r335 (PR #339).** The report is accurate for **r324-r327** and was **already fixed at
r328** by the re-base onto `919794500`, which refreshed block 10's moved copy of `ggml_cuda_can_fuse` from
the fork point (upstream `31953bcf7` relaxed these checks in `ggml-cuda.cu`, and the block's copy predated
it). Nothing is folded in, the `0001-*.patch` below is **not** applied, and it no longer applies to the
current release tree. See [`RESOLUTION.md`](RESOLUTION.md) for the evidence, `WORKLOG.md` 261121.014 and
`INVARIANTS.md` r335 for the rule this taught. Kept as the campaign's original brief (verbatim from PR #339
below the status header); it is not work in flight.

A crash fix for default-set code, offered for folding into block 10 if it is useful. It is not RDNA2-specific:
the check runs on every HIP/CUDA device.

**What happens.** `ggml_cuda_can_fuse` (block 10, `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`) asserts an F32 norm
before it looks at a `RMS_NORM + MUL (+ ADD)` chain:

```
GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
```

The fork point (`4f9dfa0e3`, `ggml-cuda.cu`) accepts an F32 or BF16 norm at the same place, and `norm.cu` in the
delivery has the BF16 fused kernels, so a BF16 `RMS_NORM` followed by `MUL` aborts instead of fusing. On r324:

```
$ test-backend-ops test -o RMS_NORM_MUL_ADD -b ROCm1
.../ggml-cuda-try-fuse.cu:1508: GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32) failed
```

The `RMS_NORM + SCALE` arm below it has the same two asserts.

**The change** (`ggml-cuda-try-fuse.cu`, +16 / -12).

- `RMS_NORM + MUL (+ ADD)`: the fork point's type checks, verbatim -- the norm is F32 or BF16 with an input of the
  same type, the weight is F32, the add operand is F32 or the norm's type.
- `RMS_NORM + SCALE`: returns `false` for a non-F32 norm instead of asserting.

An F32 chain takes the same decisions as before: with an F32 norm the new conditions reduce to the old ones.

**Tested.** r324 default tree (`d3babb65`) and r324 + this patch (`8d02f8d5`), gfx1030 (Radeon Pro V620),
ROCm 7.14, default HIP flags, `test-backend-ops test -b ROCm1`:

| `-o` | r324 | r324 + patch |
|---|---|---|
| `RMS_NORM` | 121/121 | 121/121 |
| `RMS_NORM_MUL_ADD` | abort (above) | 113/113 |
| `GATED_DELTA_NET` | not run | 55/55 |
| `FLASH_ATTN_EXT` | not run | 7381/7381 |

Only gfx1030 was available; no RDNA3/RDNA4 or CUDA run. No model run is claimed: the abort was found with
`test-backend-ops`, not with a GGUF.

Files: `0001-*.patch` (`git format-patch` against the r324 default tree; applies with a strict `git am`) and this
README.

(Claude-assisted.)
