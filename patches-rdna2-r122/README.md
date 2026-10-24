# rdna-boosts RDNA2 (gfx1030) fixes: MMB per-device arch dispatch + q8_1 cache view-offset

**2 patches** that apply on top of the 14-block rdna-boosts delivery set
(`release.json`, `v14-097cf33d9-r126`, applied tree `6abcc1a9`). The set also
applies byte-identically on the `v14-097cf33d9-r120` tree (`0fd9827d`) — the
five touched files are byte-identical between the r120 and r126 bases, so the
same patches are valid on both (strict `git am` on both, asserted).

1. **MMB per-device arch dispatch** (`ggml/src/ggml-cuda/mmb.cu`). The MMB
   bf16-WMMA dequant-weight GEMM keyed every architecture decision
   (`mmb_enabled`, `mmb_cfg`, `mmb_dense_flag`, `mmb_wtype_mask`,
   `mmb_dense_tmask`, `mmb_cfg_dump_once`) on
   `ggml_cuda_info().devices[0].cc` — the PRIMARY's compute capability. MMB is
   designed OFF on RDNA2 (no WMMA), but in a mixed-arch multi-GPU setup
   (gfx1100 primary + gfx1030 secondary) the policy froze at the primary's:
   the gfx1030 secondary ran RDNA3-tuned WMMA kernels on RDNA2 hardware and
   produced byte-exact zero output-projection logits (PPL == vocab size,
   248320), or garbled tail-layer hidden GEMMs (PPL ~917) when the output
   projection sat on the healthy primary. All six sites now use the current
   device's cc (`ggml_cuda_get_device()`), the same idiom the mmq/mmvq
   dispatch already uses; `mmb_cfg()` takes per-arch magic statics and the
   wtype/dense tmask statics move inside the env-override branch so the arch
   policy is resolved per call.

2. **q8_1 cache view-offset mis-hit + source range guard**
   (`ggml-cuda/common.cuh`, `mmvq.cu`, `unary.cu`). The Q8_1 on-the-fly
   quantize cache is keyed by the ROOT tensor's identity, but the arena buffer
   is filled from the (possibly offset) VIEW pointer. An offset view stored
   its rows under the root's identity, so a later op with the same root +
   canonical row layout could hit an entry quantized from a different base
   pointer — and when the view's compute buffer was reallocated (graph replay
   / compute-buffer pool reuse) the quantize kernel read through a stale
   pointer: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on decode, silent
   stale reads on multi-GPU. The `src1_key` walks (mmvq x4, unary x2) now stop
   at the first pointer change, so offset-free views keep the shared root
   entry and offset views get their own arena entry; `q8_1_src_range_ok()`
   checks the source read range against the root buffer's allocation before
   every `quantize_row_q8_1_cuda` and fused gated-q8_1 launch, so a stale
   pointer aborts loudly with the layout logged instead of faulting.

Apply with `../scripts/apply-rdna2-r122.sh` **after** `../scripts/apply-all.sh`;
the result is committed on branch `rdna-boosts-rdna2-r122` and asserted against
`../release-rdna2-r122.json`.

Purity: block 14 is dispatch-only — on a single-arch system the current
device's cc equals `devices[0].cc`, so behavior is bit-for-bit what it was
(V620-primary PPL unchanged in the validation matrix). Block 15 is
decode-path only (q8_1 / MMVQ, batch <= 8); prefill is untouched. No new
environment variables; both sets read only the pre-existing `GGML_CUDA_MMB_*`
and `GGML_CUDA_Q8_1_CACHE_EXACT` knobs.

## Validation (ROCm 7.2.4, `GPU_TARGETS=gfx1030;gfx1100` dual build, Qwen3.8-27B Q6_K_XL, gutenberg corpus)

| config | before | after |
|--------|--------|-------|
| 2 GPU (XTX, V620) `-sm layer` PPL (200KB) | 248320 (uniform logits) | 2.9844 |
| 2 GPU (V620, XTX) `-sm layer` PPL (200KB) | 2.9979 | 2.9983 |
| 2 GPU (XTX, V620) `-ot output=ROCm0` PPL (200KB) | 917.54 | 2.9842 |
| 2 GPU (XTX, V620) / (V620, XTX) PPL (772KB) | - | 2.7546 / 2.7697 |
| 3 GPU (2x XTX, V620) PPL (200KB) | 248320 | 2.9871 |
| single V620 PPL (200KB) | 2.9959 | 2.9968 |
| single V620 PPL + `GGML_CUDA_Q8_1_CACHE_EXACT=1` | - | 2.9963 |
| single V620 decode bench `pp512/tg128` (llama-bench) | MEMORY_APERTURE_VIOLATION (decode) | clean, tg 18.5 t/s |
| single V620 `test-backend-ops -o MUL_MAT` | - | 2/2 backends passed |

Repro before the fix (the bug topology): `CUDA_VISIBLE_DEVICES=<xtx>,<v620>
llama-perplexity -m <model> -f <corpus> -c 8192 -b 8192 -ub 2048 -fa on -sm
layer` -> `Final estimate: PPL = 248320.0000`.

Note: a true 3-GPU tensor split (`-ts >1`) engages the internal-AR pipeline
that is UNVERIFIED on gfx1030 and is pathologically slow; the 3-GPU number
above uses the stock config (no `-ts`), matching the 14-block set's own
validation. That perf path is a separate, pre-existing concern.
