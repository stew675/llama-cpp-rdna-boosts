# UPSTREAM-PR: make the CUDA fused MoE router bit-identical to the generic chain (and the argsort stable)

**Status:** prepared 2026-09-13; applies clean to master `790cf51aa` (`git apply --check`); validated on
gfx1201/ROCm 7.14 against the delivery (block 08, seventh amendment).  Not filed.

**Where it lives in the delivery:** inside `patches/0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch`
(the 2026-09-13 (seventh) amendment, TODO item 19).

## The bug

`ggml_cuda_try_fuse()` selects the fused MoE router (`ggml_cuda_op_topk_moe`, `topk-moe.cu`) by calling
`ggml_cuda_check_fusion_memory_ranges()`, which decides from **buffer-address overlap** whether the
fused outputs may alias the fusion's inputs.  The fused kernel was **not** bit-identical to the generic
`soft_max -> reshape -> argsort -> view -> get_rows -> [norm] -> [scale]` chain it elides, so *whether
the fusion fired changed the model output* — the same graph produced different greedy text depending on
where the allocator happened to place its tensors.  This is reachable upstream: any unrelated change
that re-addresses the graph (e.g. moving one op between the CPU and the GPU) can flip the coverage.

Three independent gaps:

1. **Softmax reduction order.**  The generic `soft_max_f32` launches one thread per column (a power of
   two `>= ncols`, capped at 1024) and reduces with `block_reduce`: a per-warp butterfly over each
   consecutive 32-column group, then a cross-warp butterfly over the per-warp results.  The fused
   kernel held column `l + i*32` in lane `l` and did a single flat 32-lane butterfly — a different
   floating-point association (36 % of random 512-value rows disagree, up to 2.4e-7 relative).
2. **Normalization.**  The generic chain is `sum_rows -> clamp -> div` (`weights[i] / sum`); the fused
   kernel accumulated the selected weights in the per-winner lanes and multiplied by `1/sum` — a
   different sum order *and* a reciprocal instead of a division.
3. **Top-k tie-break.**  The CUDA `argsort`'s bitonic network uses a strict comparator and is **not
   stable**, so for exact ties its top-k set/order is a function of the network, not of the index; the
   fused iterative argmax breaks ties by the smaller index, and the CUDA CUB `argsort` path
   (`SortPairsDescending`) is stable too — so the two CUDA argsort implementations already disagreed
   with each other.  Exact router ties are real (4 occurred in one 3.3k-prefill + 64-token run).

## The fix

`ggml/src/ggml-cuda/topk-moe.cu`: the softmax now does the generic per-“virtual warp”
`warp_reduce_sum(vals[i])` phase followed by the cross-warp phase (the `experts_per_thread == 1` case
keeps the single warp reduction the generic kernel uses for `ncols <= WARP_SIZE`); the norm sums the
selected weights in the generic `reduce_rows_f32` order (`warp_reduce_sum(lane j < n_expert_used ?
output_weights[0] : 0.f)`, lane `j` holding selection `j`'s weight) and **divides** by the clamped sum
like `ggml_div`.

`ggml/src/ggml-cuda/argsort.cu`: the bitonic comparator breaks ties by index (smaller index first for
`DESC`, larger index first for the second branch), matching the CUB path and the fused router.  The op
was already non-deterministic for ties; this makes it a total order.  The CPU `std::sort` comparator
leaves ties unspecified, so no cross-backend tie contract is broken, and `test-backend-ops`'
`ARGSORT` case initializes unique values by construction.

The delivery also adds a `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` A/B kill-switch in `ggml-cuda.cu`; the
upstream patch omits it (it is a fork-local testing aid).

## Validation

* **Fused == unfused** for the qwen4exp router: `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` and the normal
  build produce byte-identical greedy text for **all eight native KV types**
  (f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) on `-sm tensor` **and** on `-sm layer` (the split where
  the tie divergence reproduced: `6e2290d44875` vs `8bd14f326f2b` pre-fix).  A temporary forced-fusion
  probe (guard ignored) gives the same hash as both.
* The delivery's `plain == draft-mtp --spec-draft-n-max 3/7` purity invariant still holds for every
  native KV type; the pre-fix *unfused* reference now is the fused hash too (`iq4_nl` tensor
  `086df944f6af`; pre-fix fused `14a1a3f257f4`).
* `test-backend-ops test`: **18065/18065** (`ARGSORT`, `TOP_K`, `GET_ROWS` pass).
* 4B dense coherence unchanged (`1c5d32ac537d`); qwen4exp pp2048/pp8192/tg128 within run-to-run noise.

## What was NOT validated

* No NVIDIA/CUDA hardware was available — the change is backend-generic CUDA source and was only
  compiled and run as HIP/ROCm.  Compile and regression-test it for CUDA before filing.
* Other backends (Vulkan/Metal/SYCL/CPU) are untouched; the CPU argsort's tie order remains
  unspecified (it was already).
* Only the `n_experts`/`n_expert_used` geometries of the models at hand were exercised (512 experts,
  top-8, the `softmax` + `norm` router; the qwen4exp paths).  The `experts_per_thread == 1` and
  `n_expert_used > WARP_SIZE` fallback paths are reasoned about but not separately measured.
