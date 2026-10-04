# wip/rdna4-verify-fusions-lf — three bit-identical speculative-verify fusions (revision 2)

One `git am` patch made on `v16-a55e952b8-r10` (canonical tip `b86854900`, net tree `dab5186b…`); applied tree `38ebce2f…`. It replaces `lf-verify-fusions.patch` from PR #100 and addresses the review in `TODO.md` item 31.

| fusion | what it does | off switch |
|---|---|---|
| GLU -> Q8_1 | a verify-step GLU also writes the Q8_1 activation of the next mmvq matmul, which then skips its `quantize_q8_1` | `GGML_CUDA_FUSE_GLU_Q8_1=0` |
| GDN conv at 2..255 tokens | `gdn_conv_check` also accepts `T >= 2`, so the verify step's CONCAT is no longer a generic non-contiguous copy per layer | `GGML_CUDA_FUSE_GDN_CONV_VERIFY=0` |
| batched copy | consecutive same-layout f32 CPY nodes (the per-position conv-state snapshots) in one launch | `GGML_CUDA_FUSE_CPY_BATCH=0` |

All three are bit-identical, default-on and RDNA4-only. `GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1` prints every batched run.

`vf_sweep.cpp` is the geometry sweep used for the bit-identity check (how to build and run it is in its header). On r10 / gfx1201 it ran 16,130 cases. The fusions-on, fusions-off and stock r10 hashes were identical in every case.

*The patch, the sweep and this note were written by Claude (AI).*
