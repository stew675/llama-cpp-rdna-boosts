Thank you for fixing #332 in r321, and for the write-up. The partial-arena table made it clear where the staged answer actually shows up. Separately, we compared our decode against the fully unfused graph. That turned up two small, related rounding differences in the MoE weighted reduction, and an ordering of two folds that costs one launch per MoE layer. Sharing the patches and data in case they're useful. Nothing is urgent on our side, so please look whenever things are quiet.

**What it does**

- **0001** (`mmvq.cu`, `mmvq-moe.cu`, `common.cuh`):
  - **Fold precedence.** On Qwen3.5/3.6-style MoE decode, `ggml_cuda_moe_down_fold_try_fuse` folds the weighted MUL into the down (`x_scale`). After that, the larger down + weighted-reduction fold that `ggml_cuda_mwr_fold_mark` armed for the same `MUL_MAT_ID` can no longer run. The reduction's ADD chain then becomes a separate launch per MoE layer. The MUL-only fold now steps aside when the larger fold is armed for that node (`GGML_CUDA_MWR_FOLD_FIRST=0`: old precedence). It defaults on for HIP only; see the nvcc note under Limits.
  - **No fma in the fold's sum.** HIP clang contracts `sum += (x * scale) * w` into an fma, so the fold differs from the unfused MUL + ADD chain at rounding level. The sum now runs under `#pragma clang fp contract(off)` (`GGML_CUDA_MWR_FOLD_CONTRACT=1`: old sum).
- **0002** (`moe-weighted-reduction.cu`): the same `contract(off)` in the standalone `moe_weighted_reduction_f32_vec4` / `_f32` kernels (`GGML_CUDA_MWR_EXACT=0`: old kernels). The BF16 variants (BLK16 / DOWN16, default off) are untouched.

We tried `__fmul_rn` / `__fadd_rn` first. HIP lowers them to plain ops, which still get contracted, so only the pragma worked.

**Output.** KLD against the fully unfused graph (`GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1 GGML_CUDA_MWR_FOLD=0 GGML_CUDA_DISABLE_MWR=1 GGML_CUDA_DISABLE_FUSED_ADDMUL=1`). Setup: wikitext-2, 2 chunks, `-fa on`, `ROCBLAS_USE_HIPBLASLT=0`, 1 x R9700, r321. r317 gave the same values to the last digit.

| model | `-ub` | path exercised | stock | patched |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 512 | standalone reduction kernel | 0.008215, same top 96.58 % | 0.000000, 100 % |
| Qwen3.6-35B-A3B Q4_K_M | 1 | decode (stock: MUL-only fold + fused add) | 0.000000, 100 % | 0.000000, 100 % (now via the larger fold) |
| Qwen3-30B-A3B Q4_K_M | 512 | standalone reduction kernel | 0.003479, 98.53 % | 0.000000, 100 % |
| Qwen3-30B-A3B Q4_K_M | 1 | decode fold | 0.002054, 98.43 % | 0.000000, 100 % (PPL 8.6081 = unfused 8.6081) |

The second row shows the patch keeps an already-exact result while switching to the faster path.

Our production model is Flash-Next UD-IQ4_XS on 2 GPUs (`-sm tensor`, `-ub 4096`), running with our own patches on top of your release. There, 0002 moved KLD against the unfused graph from 0.011450 to 0.000000 (PPL 2.5558 -> 2.5504).

**Speed.** llama-bench `-p 512 -n 128 -r 3 -fa 1`, 1 x R9700, r321. Where a cell has two numbers, the runs alternated stock, patched, stock, patched.

| model | tg128 stock | tg128 patched |
|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 155.78 / 155.84 | 159.39 / 159.34 (+2.3 %) |
| Qwen3-30B-A3B Q4_K_M | 176.53 | 176.71 (flat, as expected: the fold already ran there) |
| gemma-4-26B-A4B-it MXFP4_MOE | 97.12 | 97.13 |

pp512 is in the log. The first run was noisy (about ±440 t/s on stock), and the remaining runs agree within noise, so we see no change.

**Base:** formatted against and measured on r321 (`v17-4f9dfa0e3-r321`, tree `add239134e`).

**In use:** our production server has run both patches since 2026-11-16 (Flash-Next UD-IQ4_XS, 2 x R9700), including the nightly stress sessions, and we have seen no faults.

**Limits / not tested:**
- **NVIDIA / nvcc: not built or run.**
  - nvcc ignores `#pragma clang fp contract(off)`, so both kernels compile there as before.
  - With that, 0001's precedence change would swap an exact path (MUL fold + ADD chain) for the contracted fold.
  - So `GGML_CUDA_MWR_FOLD_FIRST` defaults off outside `GGML_USE_HIP` (`=1` turns it on).
- RDNA3 / gfx1151 and Windows are not tested. The pragma needs HIP clang's default `-ffp-contract=fast-honor-pragmas`.
- Without fma, the standalone kernels could in principle be slightly slower. pp512 showed no change beyond noise; we did not time the kernel on its own.

Happy to run any further checks on these cards if they would help.

Files:
- `0001`-`0002` `*.patch`: `git format-patch` against r321; apply with `git am`.
- This README.
- `logs/ab-r321.log`: the KLD and llama-bench A/B above.
- `logs/ab-r317.log`: the same on r317 (decode +2.4 %).
- The log headers name each commit before its message was reworded. The trees are the same: r321 `5549ecd8f5`, r317 `e112e286c3`.

(Claude-assisted, as before.)

---

## How this was taken (r324, archived from `wip/mwr-exact-r321/` as submitted)

Both items shipped in **block 10** (`WORKLOG.md` 261119.008, `benchmarks/2026-11-19-mwr-exact.md`,
`GREEDY-PURITY.md` §102, `INVARIANTS.md` r324).  Four things were corrected first; nothing in this directory was
applied verbatim:

1. **One knob, not two.**  `GGML_CUDA_MWR_FOLD_CONTRACT` was removed and `GGML_CUDA_MWR_EXACT` drives both the
   fold's sum and the standalone kernels.  They replace each other at the same node, so `EXACT=0` alone would
   leave the fold exact and the kernel contracted - a state no release shipped, and a bisect would blame the
   fold.
2. **The stand-aside predicate was looser than the fold's.**  `ggml_cuda_mwr_fold_mark` armed from the graph
   pattern alone, while the fold also needs `warp_size == 32`, `ne[2] == ne[3] == 1`, a contiguous down output
   and `ne[0] <= 2*GGML_CUDA_MOE_RED_CNT-2`.  Any disagreement (CDNA's warp 64, a large `n_embd`, an
   `ne[3] != 1` graph) lost the MUL fold for nothing.  The conditions now live in
   `ggml_cuda_mwr_fold_serve_ok()` and are enforced at the ARMING site.
3. **A census was added** (`GGML_CUDA_MWR_DEBUG`: `MWR EXACT`, `MWR FOLD`, `MWR FOLD ASIDE`,
   `MWR FOLD DECLINE`), because the pre-existing line only fired when the *standalone* kernel ran, which on
   these shapes is never - so it read 0 in both arms.
4. **Two of the submission's own readings were corrected.**  Its `-ub 1` decode row read 0.000000 on the
   unpatched build and concluded "keeps an already-exact result"; that window cannot see the decode difference
   (the 256-token greedy text can, and the unpatched build's text is NOT the unfused chain's).  And
   "(flat, as expected: the fold already ran there)" for Qwen3-30B-A3B does not generalise: on
   Qwen3-Coder-30B-A3B (same architecture) r323 takes the MUL fold on all 238 sites and r324 is +2.2 %.

Its central claim reproduced exactly, and is worse than the submission stated: the r323 delivery's MoE decode
text was not the unfused chain's text on **three** architecture families, not just the reporter's Flash-Next
shape.  Record: `benchmarks/2026-11-19-mwr-exact.md`.
