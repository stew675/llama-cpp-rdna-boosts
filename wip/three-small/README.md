# wip/three-small — GLU in place, graph replay up to 16 tokens, one FA KV-group bitmap per graph

Written by Claude (AI).  Three `git am` files, one commit each, made on `v16-4f9dfa0e3-r295` (tip `58184e329`).  Each applies on
its own, and all three in order, on `v17-4f9dfa0e3-r319` (tree `9f53a853`, rebuilt here from the release's block patches on
`4f9dfa0e3`; line offsets only), so any of them can be taken alone.

| file | touches | kill switch |
|---|---|---|
| `0001-ggml-alloc-a-GLU-may-take-the-memory-of-one-of-its-i.patch` | `ggml/src/ggml-alloc.c` (+15) | `GGML_GLU_INPLACE=0` |
| `0002-CUDA-HIP-graph-replay-up-to-16-tokens-GGML_CUDA_GRAP.patch` | `ggml/src/ggml-cuda/ggml-cuda.cu` (+8 −1) | `GGML_CUDA_GRAPH_MAX_TOKENS=8` |
| `0003-CUDA-HIP-FA-KV-group-bitmap-built-once-per-graph-eva.patch` | `fattn-common.cuh`, `fattn.cu`, `fattn.cuh`, `ggml-cuda.cu` (+118 −15) | `GGML_CUDA_FA_KQ_BLOCKS_REUSE=0` |

## 1. A GLU may take the memory of one of its inputs

`ggml_op_can_inplace` gains `GGML_OP_GLU` (read once from `GGML_GLU_INPLACE`, on unless 0).  The allocator then gives a GLU the
memory of its gate or up when that input has the same layout and no other reader (the two-input form; the one-input form halves
the row and never qualifies).  The kernels that write a GLU read each element before writing it: CPU `ggml_vec_*glu_*` (no
`restrict`), CUDA / HIP `unary_gated_op_kernel`, `swiglu_oai_kernel`, `swiglu_clamp_kernel` and the Q8_1-writing variant.

## 2. Graph replay up to 16 tokens

`ggml_cuda_graph_is_multi_token` compares with `GGML_CUDA_GRAPH_MAX_TOKENS` (default 16) instead of `MMVQ_MAX_BATCH_SIZE`.  A
9..16-token verify (two slots verifying in one batch, an n-gram draft longer than 7) has the same shape every step, so capture
amortises as it does for 1..8.  The kernels are the same; only the graph path changes.  WORKLOG 260928 measured a small 4B graph
~2.5 % slower with verify graphs, which is why `GGML_CUDA_DISABLE_VERIFY_GRAPHS` exists; the same can hold for 9..16 tokens on a
small model, so a small model is in the numbers below and `GGML_CUDA_GRAPH_MAX_TOKENS=8` keeps the previous limit.

## 3. One FA KV-group bitmap per graph evaluation

`launch_fattn` asks `ggml_cuda_fattn_kq_blocks_reuse` (fattn.cu) for the bitmap buffer.  The first attention layer of a graph
evaluation builds the bitmap into one fixed buffer per CUDA context (64 KiB, allocated outside capture so a captured graph keeps
its address); a later layer with the same mask tensor, mask data pointer, stream and tile shape (an 8-value key) reuses it without
the memset and the `flash_attn_mask_to_KV_blocks` prepass.  `ggml_backend_cuda_graph_compute` starts a new evaluation
(`ggml_cuda_fattn_new_graph_eval`).  Anything else falls back to the pool buffer as before: another stream within the same
evaluation, a bitmap above 16384 words, a first build while a graph is being captured.  Models with two masks (sliding-window
and global layers) rebuild when the mask changes, as before.

## Results

One R9700 (gfx1201), Linux, ROCm 10.0.0.

**Applies.**  Each file alone, and all three in order, with `git apply --check` on `v17-4f9dfa0e3-r319` (tree `9f53a853`): line
offsets only.  Between r295 and r319 the files they touch changed only outside these spots: `fattn-common.cuh` in the band's gate,
its P constant and the decode/verify split (#330), none of which the bitmap depends on; `fattn.cu` in the 128-dim band dispatch;
`ggml-cuda.cu` in the MoE expert cache, hc, the GDN epilogue order and the host-expert tier.  `ggml-alloc.c` and the GLU kernels
did not change.

**Built and checked on r295** (`v16-4f9dfa0e3-r295` plus the three commits).  CPU backend, Qwen3.8-4B Q6_K, `-ub 512`, a
2000-token prompt and 24 greedy tokens: `GGML_GLU_INPLACE` on and off give the same text and the same top-5 log-probabilities at
every position; the CPU compute buffer goes 73.02 → 67.21 MiB.

**From our downstream tree** (these three changes in daily use there since early November, alongside other changes; indicative
only, not re-measured on stock): the GLU in place takes 136 MiB off Qwen3.8-27B's compute buffer at `-ub 2048`; graph replay up
to 16 tokens makes an n-gram copy −3.3 % per step on a 27B with the text unchanged; the shared bitmap saves about 0.1 ms per step
on a 27B with 16 attention layers, within run-to-run noise on its own.  Greedy text was checked identical with each switch off.

**Not tested on this patch set**: GPU bit-identity and timings on stock r295 / r319; a small model for item 2 (WORKLOG 260928
measured verify graphs ~2.5 % slower on a 4B, which may carry over to 9..16 tokens); Windows; RDNA3 / 3.5; NVIDIA; Vulkan / Metal
and other backends (item 1 changes the allocator rule they all share); several GPUs (`-sm tensor`).
