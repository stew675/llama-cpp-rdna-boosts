Thank you for the q8_1 quantize cache and the dual-output matmul fusion in r171. Most of this series builds directly on them: 0003 and 0004 only have the preceding kernel publish a Q8_1 input into that cache, and 0009 extends the dual-output idea to matrices of different heights.

Seven decode patches for single-GPU RDNA3, offered in case they're useful. They continue the numbering of #249 and apply on top of its 0001 / 0002. Each has an env switch that turns it off, and each gave identical PPL / KLD and byte-identical greedy replies in our tests (concurrent requests are covered under Gates). Most of the gain is on Qwen3-30B-A3B (+20 % tg128 on a 7900 XTX); Qwen3.6-35B-A3B gains +2.5 %.

**0003: FA combine writes the Q8_1 of its output** (`fattn-common.cuh`, `fattn-qsa.cu`; `GGML_CUDA_FA_COMBINE_Q8=0` off). `flash_attn_combine_results` also writes the Q8_1 blocks of its output into the q8_1 cache entry of the following quantized MUL_MAT (the attention output projection), so that matmul skips its `quantize_q8_1` launch. The math is `quantize_q8_1`'s. It is used only when the output is contiguous F32 with a row length that is a multiple of `MATRIX_ROW_PADDING`.

**0004: MoE gate/up + GLU writes the Q8_1 for the down projection** (`mmvq.cu`, `mmvq-moe.cu`, `common.cuh`; `GGML_CUDA_MOE_GLU_Q8=0` off). In the 1-token fused gate/up/GLU kernel, the last block to finish each (expert slot, 32-row group) quantizes that group. It finds out it is last through a self-resetting counter (a fence, then one atomic per block). The down MUL_MAT_ID finds the cache entry and skips its quantize.

**0005: the fused K norm+rope+SET_ROWS kernel also writes the V row** (`rope.cu`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_ROPE_SET_ROWS_V=0` off). This is the F32 / F16 cache path. When the next two nodes are the V row's VIEW + SET_ROWS into a cache of the same type and row length, the per-head blocks also copy the V row with `k_set_rows`' cast, and the V set_rows launch goes away.

**0006: MoE GEMV, 4 rows per block on RDNA3_0** (`mmvq-moe.cu`; `GGML_CUDA_MOE_RPB_WIDE=0` off). The gate is: K of 8+ blocks, `blocks_per_row_x % n_groups == 0`, and at least 32 blocks per WGP left. Each wave then streams twice the rows for the same y load. With K-blocks splitting evenly over the thread groups, every row keeps its K-blocks in the same group and order, so the result is bit-identical. Qwen3-30B gate/up (768 rows x 8 experts) goes from 24.1 to 19.2 us; Qwen3.6 (512 x 9) stays at 2 rows because 4 measured slower there.

**0007: MoE down + weighted reduction in one launch** (`mmvq.cu`, `mmvq-moe.cu`, `moe-weighted-reduction.cu`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_MWR_FOLD=0` off). For 1 token, `moe_weighted_reduction`'s try-fuse arms the down MUL_MAT_ID instead of launching. The last expert lane to finish a row block sums `(e * scale) * w` in the reduction's order, and the reduction node launches nothing. When armed, the kernel takes 8 rows per block so there are fewer counter atomics; this is bit-identical under 0006's rule, or when `n_groups <= 2`.

**0008: Q and K norm+rope in one launch** (`rope.cu`, `rope.cuh`, `common.cuh`, `ggml-cuda-try-fuse.cu`, `src/models/qwen3moe.cpp`; `GGML_CUDA_ROPE_QK_PAIR=0` and `LLAMA_QK_ADJACENT=0` off). `rms_norm_mul_rope_f32_qk_pair` runs a layer's Q chain (F32 out) and K chain (with its cache writes) in one grid, Q blocks first.
- **Why the model file changes:** deferring Q to the K node was not safe as the graph was ordered. The V matmul between the two chains reuses the Q projection's freed buffer. So `qwen3moe.cpp` now expands V and the raw K projection before the Q norm. The order becomes V, K, Q matmul, then the Q chain, then the K chain: the same nodes, with the two chains adjacent.
- **The try-fuse gate** still checks that nothing between the two chains writes the Q chain's inputs or output, or reads the Q chain. A deferred Q whose K node is taken by another fusion is launched on its own.
- **Only qwen3moe is reordered.** Qwen3.6 pairs without any model change; it is the 1.1 % in the table below. Other models keep the upstream order.
- Traced on Qwen3-30B: K 4.10 us + Q 3.46 us becomes 4.19 us.
- If you'd rather not touch model code, the `qwen3moe.cpp` hunk can be dropped. Qwen3.6 keeps its gain; Qwen3-30B then pairs almost never, and 0009 falls back to two-way there.

**0009: V, K and Q projections in one launch** (`mmvq.cu`, `mmvq.cuh`, `common.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_MMVQ_ROW_CONCAT=0` off). Two or three consecutive 1-token MUL_MATs over the same src1, with the same weight type and K, run as one ksplit launch. The largest weight goes first and the rows of the others are appended. Every row keeps its own unchanged dot product.
- **RDNA3_0 only.** There, the launch config depends on the row count only through `small_m` and the rows per block, and the gate requires both to agree for each weight and for the total. On RDNA4 `two_rows` and the decode row loop also depend on the row count, so it stays off there.
- **A separate template instance (`row_cat`).** Checking the segments inside every ksplit kernel cost Qwen3.6 2 % with the feature off, so the plain kernels are left unchanged.
- **It runs ahead of the dual-output fusion.** In Qwen3-30B's all-Q4_K layers it takes V + K + Q (3.4 + 7.0 us becomes 8.7 us); in the layers where V is Q6_K it takes K + Q.

**Setup:** RX 7900 XTX (gfx1100), 1 GPU, Qwen3-30B-A3B and Qwen3.6-35B-A3B, both Q4_K_M (unsloth). Stock r171 (tree `2f5870713e94`) vs r171 + #249's 0001 / 0002 + 0003-0009, both built with `-DAMDGPU_TARGETS=gfx1100`, your default env.

**Gates** (stock vs the full series):
- wikitext-2 perplexity / KLD (24 chunks, `-c 512`, against logits from a plain upstream HIP build): identical to the last digit. Qwen3-30B PPL 8.476910, KLD 0.006154; Qwen3.6 PPL 5.920938, KLD 0.009509.
- llama-server greedy, 4 chats x 3 turns: byte-identical replies on both models (1034 and 743 tokens).
- 2 concurrent requests decoding together: on Qwen3.6, the same output as stock. On Qwen3-30B, stock itself gave 3 different outputs over 7 runs; the series gave 3 over 4 runs, two of them stock outputs. The third combines, prompt by prompt, replies that stock also produced, so it looks like the same timing dependence.
- 0 GPU faults.

**Throughput**, `llama-bench -fa 1 -r 5`, t/s, two alternating rounds. The "#249" arm is the series build with every switch of 0003-0009 off (both of 0008's included):

| model | arm | tg128 | tg128 @ d4096 | pp512 |
|---|---|---|---|---|
| Qwen3-30B | stock r171 | 169.50 / 169.77 | 152.46 / 152.03 | 4975 / 5162 |
| | #249 | 168.54 / 168.37 | 152.09 / 151.76 | 5151 / 5158 |
| | + 0003-0009 | 203.27 / 203.04 | 178.89 / 178.68 | 5149 / 5160 |
| Qwen3.6 | stock r171 | 152.93 / 152.68 | 150.46 / 149.55 | 5023 / 5015 |
| | #249 | 155.51 / 155.44 | 152.66 / 152.35 | 5043 / 5023 |
| | + 0003-0009 | 159.29 / 159.58 | 155.96 / 156.13 | 5044 / 5022 |

So, over #249 and averaged over the two rounds, decode is about +20.6 % (tg128) / +17.7 % (@ d4096) on Qwen3-30B and +2.5 % / +2.3 % on Qwen3.6; prefill is within noise.

How much each patch carries, as the tg128 loss when only that patch is switched off from the full series (one run each, noise about 0.3 %). 0008's figure includes 0009 falling back to two-way, since 0009 relies on 0008's order for the three-way launch:

| | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | 0009 |
|---|---|---|---|---|---|---|---|
| Qwen3-30B | -3.1 % | -2.7 % | -2.9 % | -5.3 % | -1.2 % | -7.2 % | -4.2 % |
| Qwen3.6 | ~0 | -1.6 % | -0.4 % | -0.2 % | -0.4 % | -1.1 % | -0.2 % |

An independent review of the logic found two issues, both fixed in these patches. The fixes only change host gates that pass on this GPU, so the numbers above are from before them; the folds still fire (traced) and the greedy replies were re-checked after them (same on both models). 0004 and 0007 now arm only on wave32 devices, because their kernel side is wave32-only and a GFX9 build would have skipped the quantize or the reduction without writing them. 0007 also steps aside when the reduction output overlaps the expert ids, which later blocks of the same launch still read. Three more notes from it, flagging in case they're useful:
- 0004 writes only the Q8_1 blocks of the row, not the zero padding up to `MATRIX_ROW_PADDING` that `quantize_q8_1` writes. Every 1-token consumer today stops at K / qk, but a future consumer that walks the padded row would read stale blocks.
- 0008's deferred Q is launched at the K node by a matcher that runs after many others. If an earlier matcher ever claimed the K node's RMS_NORM, the K norm could run first, in memory the Q projection just freed. None does today, but moving the pair and flush check to the top of `ggml_cuda_try_fuse` would close it.
- None of the patches here, nor the existing dual-output fusion that 0009 builds on, is guarded against `-sm row` splits. We have not tested multi-GPU.

Not tested: RDNA4 (our R9700s are not in the machine this week), multi-GPU / `-sm tensor`, and MTP / speculative verify batches. 0006 and 0009 are gated to RDNA3_0. 0003, 0004, 0005, 0007 and 0008 are not arch-gated, so they would be new on RDNA4 too; we're happy to run the RDNA4 and 2-GPU checks when the cards are back, or anything else you'd like. No urgency on our side.

Files: `0003-*.patch` ... `0009-*.patch` (`git format-patch` on top of #249's 0002, apply with `git am`) and this README.

(Claude-assisted, as before.)
