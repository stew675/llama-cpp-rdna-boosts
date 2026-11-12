# wip/fa-derived-mask-gqa — the RDNA band / tile GQA choices count the derived KQ mask at src[6]

Written by Claude (AI).  One `git am` file, one commit, made on `v16-a691b2345-r262` (tree `891d4acb`, rebuilt here from the
release's 16 block patches; the patches' sha256 and the tree match `release.json`).  It only changes which kernel variant an op
with the derived KQ mask takes; results are unchanged.

| file | touches | sha256 |
|---|---|---|
| `0001-FA-the-GQA-choices-of-the-RDNA-band-and-tile-launche.patch` | `ggml/src/ggml-cuda/fattn-common.cuh`, `fattn-rdna.cuh`, `fattn-tile.cuh` (+14 −2) | `a8ef0b47536dbb36f861e09f072bd6bd6c9618b294934b497163efb51988a21d` |

## What it fixes

Upstream #29510 (the `a691b2345` base, so r236 onward) gives `GGML_OP_FLASH_ATTN_EXT` a `kv_rows` operand at `src[5]`, and the
derived KQ mask (`ggml_flash_attn_ext_add_kq_derived`) moved to `src[6..8]`.  `ggml_cuda_flash_attn_ext_has_mask` (fattn.cu) counts
`src[6]`, but two launcher checks in block 05 still ask `src[5]`:

- `fattn-rdna.cuh`, `ggml_cuda_fattn_band_wmma_applies`: `gqa_opt = (mask != nullptr || dst->src[5] != nullptr) && …`
- `fattn-tile.cuh`, `launch_fattn_tile_switch_ncols2`: `use_gqa_opt = (mask != nullptr || KQV->src[5] != nullptr) && …`

An op that carries the derived mask has no mask tensor, so these checks now test `kv_rows` instead, and the op takes other kernel
variants than the same op with a mask tensor.  The derived mask is used for ubatches of 9+ tokens: n-gram verify batches of 9..16
rows and short prefill chunks.

## Change

- `ggml_cuda_fattn_derived_mask(dst)` in `fattn-common.cuh` returns `dst->src[6] != nullptr`; both checks use it.
- When `kv_rows` is present a mask tensor is required (ggml.c asserts it), so `mask != nullptr` already covers that case.
- Kill switch: `GGML_CUDA_FA_DERIVED_GQA=0` restores the `src[5]` check (the r236..r262 behaviour).

## Results

Validated on r241 (the same lines; they are unchanged from r241 to r262).  On r262 only `git apply --check` was run: no rebuild,
no rerun.  Linux, Ryzen 9 3900X, one R9700 (gfx1201), ROCm 10.0.0.  `ROCBLAS_USE_HIPBLASLT=0` (GREEDY-PURITY §44) in every arm.
Every arm is a freshly started server.

**Same output.**  Arms: stock r241; stock r241 with `LLAMA_KQ_MASK_DERIVED=0` (packed masks only); r241 + #292's three commits +
this commit (#292 has since landed in r244); and that with packed masks.  Compared: text and draft counts of every request, and
the top-5 logprobs after 2K / 8K / 30K prompts.

| config | requests | differing between arms |
|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL, DFlash2 draft 5, `-np 2 --kv-unified -c 131072 -ub 2048`, q8_0 K/V | 25: tool-calling ERP ×3, an 8K copy, 7 prompt lengths, two prompts in one request, prompt-cache continuation and round trips, 110K, two requests at once, logprobs ×3 | none |
| same + `--spec-type ngram-mod,draft-dflash` (verify batches of 9..16 rows take the derived mask) | 7 | none |

**Speed.**  The copy request (where the n-gram drafts make the most 9..16-row verify batches), r241 + #292 with and without this,
A B B A: −0.13 % tokens/s, +0.13 % ms per step.  That is within noise: this is a consistency fix, not a speed-up.

## Not validated

- A rebuild and rerun on r262 (only `git apply --check`).
- Several GPUs, Windows, RDNA3, NVIDIA (the tile launcher's GQA limit differs there).
- Models other than the dense 27B with a DFlash2 draft (no head-64 / head-128 models, no MoE).
