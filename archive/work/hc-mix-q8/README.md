# `hc-mix-q8` — the Q8_0 hyper-connection mixer path (contributor PR #326, `briansp2020`)

**PROMOTED at r313 into official block 11.**  There is nothing to apply from this directory and no `wip/` entry
for it: the contributor submitted the work as a wip handover (`wip/hc-mix-q8-r304/`: README + three
`git format-patch` files + logs, kept verbatim under [`pr326/`](pr326/)), this repo assessed it, and the maintainer
decided to fold it into **block 11** — the block that owns `hc-mix.cu`, the qwen4exp hc graph code and the
`HC_MIX` half of the CPU reference — rather than cut a new block.  All surfaces it touches were already carried
by `0011`, so the fold has no cross-block dependency.

## What it does

The fused Q8_0 `GGML_OP_HC_MIX` (a qwen4exp GGUF whose hc weights are Q8_0 — every unsloth UD quant; the BF16
quants take `hc_mix_bf16_*` instead) ran ~8 launches per hc site where the BF16 mixer runs 3.  The patch set
brings it to 3-4:

1. **persistent `down_dots` counters** — per-stream slots zeroed once at allocation, the finishing block resets
   the counter it completes, so the per-call `cudaMemsetAsync` launch goes away;
2. **planar output** — `mixed [n_embd, T]` then `inject [hc, T]`, both contiguous, so the verify band needs no
   `ggml_cont` of the mixed head (the BF16 op already did this, `LLAMA_HC_MIX_PLANAR`);
3. **Q8_1 epilogue** — the collapse (or the fused up+collapse kernel) also writes the mixed rows' Q8_1 into the
   quantize cache with `quantize_q8_1`'s arithmetic including its `d * (integer sum)` term, so the matmuls that
   read `hc_mixed` skip their quantize;
4. **`hc_mix_combine_rms_gamma_q8`** — the `HC_COMBINE -> HC_MIX` pattern's combine + grouped RMSNorm + gamma +
   Q8_1 in ONE kernel for Q8_0 hc weights (the BF16 arm's `hc_mix_combine_rms_gamma`, `GGML_CUDA_FUSE_HC_COMBINE_MIX`,
   now covers both classes);
5. **up band weight preload** — each thread loads its weight ints and scales before the prologue barrier, with the
   compile-time `BPR`/`MAXI` (the dot is the same `vec_dot_q8_0_q8_1_impl` on the same values);
6. **`hc_mix_up_collapse_q8`** — the up rows of 8 reference blocks and the collapse of their columns in one
   launch, with the per-warp gates kept in shared memory and summed in the band kernel's order;
7. **F32 inject rows on the down grid** — as its first blocks (they read a whole `hc_dim` row, ~10x a down row,
   so at the tail of the grid they stretched it).

## What we verified (r313, gfx1201, 2 x R9700 `-sm tensor` unless stated)

* **Applies and folds cleanly**: the three patches apply to the r312 tree with `git am -3`; `hc-mix.cu` is
  byte-identical between r304 and r312, `common.cuh`/`ops.cpp` are untouched there, `qwen4exp.cpp` changed only
  in a comment and `ggml-cuda.cu` only elsewhere, so the fold into block 11 needed one conflict resolution
  (`planar_env`'s pre-lambda form) and nothing else.  The rewritten chain's tip tree equals the folded tree.
* **Text byte-identical to r312, in every arm we can build**: greedy 96 and 128 tokens; the MTP verify band
  (`nt = 4`, draft acceptance **0.68548, 85/124, mean len 3.02** on both builds); a 5246-token prompt + MTP
  (acceptance **0.41600, 52/125** on both); `-sm layer`; `-ub 4`; `-ub 1`; and **each of the seven switches
  alone** (`GGML_CUDA_HC_DONE_PERSIST=0`, `GGML_CUDA_HC_Q8_EPI=0`, `GGML_CUDA_HC_UP_PRELOAD=0`,
  `GGML_CUDA_HC_UP_FUSE=0`, `GGML_CUDA_HC_INJECT_DOWN=0`, and the two pre-existing knobs the two redundant
  `_Q8` sub-switches were folded into at promotion — `LLAMA_HC_MIX_PLANAR=0`,
  `GGML_CUDA_FUSE_HC_COMBINE_MIX=0`).
* **`scripts/gate-hc-fusions.sh` on the affected class** (`UD-Q4_K_XL`, Q8_0 hc + F32 inject, `-sm tensor`,
  `SOFT_EXPECT=""` because the built-in anchors are the BF16-hc GSQ class): **PASS**, and every one of the ten
  arms reads exactly the hash the r312 build reads in the same arm (`chain_chain`/`comb_op`/`cuda_fuse_off`
  431 `df42686290aa`; `mix_op`/`both_ops` 433 `8cd639da2fbf`; `defaults`/`all_ops` 461 `8b39a8568222`).  Its
  `both_ops == mix_op` comparison is the oracle for the new combine+norm Q8 kernel: it is byte-identical.
* **`test-backend-ops`**: `HC_MIX` 30/30, `HC_COMBINE` 20/20, `HC_COMBINE_MIX` **13/13** — the last now carries
  **Q8_0 arms** (`inject` F32 and Q8_0, `nt` 1/2/3/5/8, plus the head call), which we added because the case was
  BF16-only and the new kernel had no unit-level oracle.  Its teeth were checked by perturbing
  `hc_mix_combine_rms_gamma_q8` (`xn * 1.5`) and confirming exactly the Q8_0 arms fail.
* **Adaptive-MTP four axes on this model** (`-n 512`, `draft-mtp n3`): acceptance identical to the totals on every
  axis (K 0.86385 368/426, R 0.67126 341/508, C 0.50993 308/604, P 0.52872 313/592) with generation +6.7 / +7.9 /
  +8.9 / +8.0 % -- the drafter's verify logits are what decide accept/reject, so identical counts are the sharpest
  available statement that the mixer's numerics did not move.
* **`scripts/gate-prefill-logits.sh`**: mean KLD **0.000000**, same-top-p **100.000 %** against a base recorded
  from the r312 build (the recorded base was refreshed in the same session — it had been a 2026-11-16 build).
* **No GPU faults** in the kernel journal over every arm of the two gate runs.
* **Perf reproduced independently of the contributor** (`llama-cli`, greedy 128 tokens after a short prompt,
  `-c 8192 -ngl 99 -fa on -sm tensor --n-cpu-moe 42 -lme auto --load-mode mmap --lazy-mode auto -t 7`,
  median of 3, interleaved arms): decode **45.1 -> 48.3 t/s (+7.1 %)**; with all seven switches at `0` the
  same binary reads **45.1**, i.e. the win is the new code and the switch set is complete.  After a 5246-token
  prompt: **48.9 -> 51.3 (+4.9 %)**.  Prompt rates unchanged (36.3-36.9 t/s in every arm).  Final release
  build: **44.7/44.8 -> 48.2/48.3 (+7.7 %)**.  Protocol and raw rows: `benchmarks/2026-11-18-hc-mix-q8-decode.md`.
* **The BF16 path is untouched** at source level: every removed line in `hc-mix.cu` is in a Q8_0 kernel; the only
  BF16-touching edit is `ggml_cuda_op_hc_combine_mix`'s dispatch.

## What changed relative to the submission

* the two redundant sub-switches were folded away (one switch per feature, maintainer's call):
  `GGML_CUDA_FUSE_HC_COMBINE_MIX_Q8` -> the existing `GGML_CUDA_FUSE_HC_COMBINE_MIX`,
  `LLAMA_HC_MIX_PLANAR_Q8` -> the existing `LLAMA_HC_MIX_PLANAR`.  Both are text-inert and both were re-tested on
  the final build;
* the preload/fused `MAXI` instantiations are guarded by `maxi_up <= 3` (they are `MAXI` 3; `ceil(160 / 32*vdr)`
  is 5/3/2/1 for vdr 1/2/4/8, so 4 is unreachable today, but the `MAXI` 4 instantiation must not be the one a
  future VDR silently lands on);
* `gate_alloc` is only allocated on the two-launch path (the fused kernel keeps the gates in its block);
* the `hc_down_done` counter comment now states the real reason for its **synchronous** zeroing — every stream
  slice is zeroed before any kernel can read it, which an async memset on the allocating stream alone cannot
  promise — instead of the submitted "so it can never be captured";
* a Q8_0 `HC_COMBINE_MIX` test arm (see above).

## Limits / open (accepted)

* **A merged-layout GGUF (`hc_*_down_inject`, post-#31028) with Q8_0 hc weights is still untested.**  The op takes
  two row views of the merged tensor and its kernels read packed rows through `->data`, which a row view keeps,
  so the code path is structurally the same as the separate-tensor one — but no such GGUF exists on this box (the
  valid/invalid pairs we have are all separate-tensor), and the synthetic merged-HC fixture in
  `tests/test-llama-archs.cpp` is F32/`hc_lr 8`, so it never reaches the fused op.  Recorded as an accepted
  limitation in `TODO.md`; `LLAMA_FUSED_HC_MIX=0` is the reference for anyone who has such a file.
* Not tested by the contributor and not testable here: NVIDIA/CUDA, gfx1151, Windows, or a model other than
  Flash-Next.  The new kernels are 32-wide with `warp_reduce_*<32>`, like the rest of the file.
* A merged `hc_down_inject` **Q8_0 inject** would take `n_inject_q8` on the down grid — the same path the separate
  Q8_0-inject class takes (covered by `test-backend-ops`' `inject=2` arm), but again only through a synthetic
  shape, not a real file.

## PR #344 — the same op's band kernels, r339

The contributor's follow-up ([`pr344/`](pr344/), `wip/hc-mix-q8-r325/`) is **also folded into block 11** (r339):
the down dots serve every token of the decode/verify band from one block per weight row, and the up+collapse
computes each row's gate from the reference's two chunk partials directly instead of emulating eight reference
blocks.  Bit-identical, `HC_MIX` 33.1 -> 21.6 us/run at `nt 4`, default-on with `GGML_CUDA_HC_DOWN_MT=0` /
`GGML_CUDA_HC_UP_ROWS=0`.  Its five 32-bit shuffle masks did not compile against any ROCm line here and were
replaced with the delivery's `ggml_cuda_shfl*` wrappers in the fold — see [`pr344/FOLD-NOTES.md`](pr344/FOLD-NOTES.md).
Record: `benchmarks/2026-11-21-hc-mix-band-kernels.md`, `INVARIANTS.md` r339, `WORKLOG.md` 261121.018.
