# Session handoff 2026-09-01 (afternoon): qwen35moe 2-GPU "hang" root-caused + fixed in block 13

## TL;DR

The "qwen35moe Q5_K/Q6_K 2-GPU hang" was **never a hang** — it is a
**pathologically slow weight load** on multi-GPU tensor split for quants
whose block size is not a multiple of 4.  It is an **upstream llama.cpp
bug** (present in pristine 0eadefebd), and it is now **fixed in block 13**
of the rdna-boosts delivery set.

## The investigation (how we got there)

1. Reproduced: Q6_K/Q5_K/True-Q3_K_M 2-GPU `--split-mode tensor` appeared
   to hang (timeout-kill).  Q8_0/Q4_K_M/Q3_K_M(UD) worked.
2. gdb on the stuck process: main thread in `hsa_amd_memory_async_copy`
   spin; later `hipStreamSynchronize` in `set_tensor_2d` -> the load
   path, not compute.  User's observation (one GPU at 100%, hopping every
   few seconds) matches the split upload hopping between devices.
3. Added debug prints: the stall is on the **first Q6_K expert tensor**
   (`ffn_down_exps`), copy `size=210 n_copies=524288 stride 210/420`.
   Attention tensors (Q8_0, 272-byte rows) copy in ~13ms; Q6_K rows take
   ~2.3s each.  **The load is SLOW, not hung** — 90s only reached blk.19.
4. Standalone repro: `hipMemcpy2DAsync` H2D with width=210 (Q6_K) or
   110 (Q3_K) = **~2300ms**; width=212 or 420 (4-aligned) = **~0-10ms**;
   D2D with unaligned width = **~2ms**.  ~1000x H2D slowdown for
   non-4-aligned widths on this ROCm.
5. Discriminator: quant block size % 4 != 0 -> Q6_K (210), Q3_K (110)
   hang/slow; Q8_0 (32), Q4_K (144), Q5_K (180), IQ4_XS (36) fast.
6. Pristine upstream 0eadefebd + ROCm 7.14 (same toolchain as docs-era)
   built from scratch: **same slow load** -> upstream bug, NOT block 13,
   NOT block 12, NOT the 22-commit range (pristine a7cc83bba via
   q4exp-ref also shows it).  The 09-01 "hangs at EVERY point" bisect was
   WRONG: timeouts were shorter than the ~3min slow load.

## The fix (in block 13)

`ggml_backend_cuda_buffer_set_tensor_2d` (ggml-cuda.cu): when
`size % 4 != 0 && stride_data % 4 == 0`, stage through device memory:
aligned-width H2D into a temp, then unaligned-width D2D gather.  Both
legs are fast; output byte-identical (standalone memcmp 0 over 110MB).

- 18 insertions; no new files; uses cudaMalloc/cudaFree (load-time only,
  once per tensor, no pool needed in the buffer context).
- Committed into block 13 by amending the fork's block-13 commit
  (`a14257996` -> `834a8d3ff`) so the block stays one commit.

## Verification

- Standalone copy test: current path 2303ms vs staged fix 19.7ms,
  memcmp identical.
- Q6_K 2-GPU: load <15s (was ~3min), pp512 4455 / tg32 82 (fork build),
  4183/66 from the fresh-apply parity-default build.
- True-Q3_K_M 2-GPU (the other hanging quant): load <15s, pp512 4524 /
  tg32 83.
- Q8_0 2-GPU (aligned, was working): pp512 4991 / tg32 85.8, unchanged.
- docs-era build (~/prs/llama.cpp/build-rocm @ 554691a72) reproduces the
  08-31 docs numbers exactly (pp512 4526-4575 / tg32 82) -> the docs
  were real, the earlier sessions just waited out the slow load.
- 13-patch set regenerated (tip 834a8d3ff), fresh `apply-all.sh` at
  0eadefebd: clean git am, applied tree byte-identical to fork tip, zero
  whitespace warnings.
- test-backend-ops: still times out at FLASH_ATTN_EXT iq4_nl in BOTH
  pristine and fixed builds -> pre-existing, unrelated to this fix.

## State

- Fork `~/llama.cpp` rdna-boosts @ `834a8d3ff` (block 13 amended with
  the fix); working tree clean.
- Boosts repo @ `24b0a01` (regenerated 0013 + all.patch, README/TODO
  updated, make-patches.sh default tip -> 834a8d3ff).
- TODO.md: root-cause entry rewritten (FIXED + upstream note).
- Worktrees/temp files cleaned.

## Follow-ups (open)

- File the upstream llama.cpp issue: split-load 2D H2D copies with width
  not multiple of 4 (Q6_K/Q3_K quant blocks) are ~1000x slower on ROCm.
  Candidate to contribute upstream once local validation settles.
- test-backend-ops FLASH_ATTN_EXT iq4_nl timeout: pre-existing, worth a
  look eventually.
- The qwen35moe 2-GPU decode numbers are now measurable: with the load
  fixed, decode (tg32 ~66-82 t/s) and prefill (~4200-4500) are real.
  The earlier "62 t/s Q3_K/Q4_K 2-GPU" TODOs should be re-measured.
