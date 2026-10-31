Thank you for taking #252, #253 and #256 so quickly, and for the scope notes you recorded with them. This one is about decode with a quantized KV cache (`-ctk q8_0 -ctv q8_0`). With a quantized cache, `build_attn` rotates Q, K and V and the attention output with a Hadamard MUL_MAT (run as an FWHT). On Qwen3-30B-A3B that rotation turns off most of the 1-token attention fusions the release has for an F16 cache: the V / K / Q row concat, the Q / K norm+rope pair, and the combine kernel's Q8_1 output for the projection. Decode with q8_0 KV ran 726 launches per token, against 510 with F16. Patches 0001-0004 bring it back to 510. 0005 does the same for attention without Q / K norm (Llama-style): Llama-3.2-1B goes from 293 to 229 launches per token, the same count as with F16. Each patch has its own env switch and keeps the same arithmetic in the same order; in every gate below the greedy replies are byte-identical to stock.

**0001: QK rotate across the KV rotation** (`ggml-cuda.cu`; `GGML_CUDA_QK_ROTATE_HADAMARD=0` off). Your `graph_optimize` QK rotate bailed when Q's ROPE was followed by its rotation (RESHAPE, Hadamard MUL_MAT, RESHAPE), and V's rotation sat between the V and K projections. That stopped the row concat. The QK rotate now accepts Q's rotation after its ROPE. V's rotation nodes move behind the Q chain: the Hadamard MUL_MATs between the chains, the views they read, and everything downstream of them. The projections stay in front.
- 1-token graphs only, so prefill keeps the stock order.
- If any node left in front would read a deferred one, the rotate is skipped.

**0002: V's rotation folded into the K cache-write kernel** (`rope.cu`, `rope.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_FUSE_V_FWHT=0` off). After 0001, V's 64-wide Hadamard sits directly in front of the K chain whose fused kernel already writes the q8_0 K and V rows. That kernel now takes the unrotated V and rotates it itself, with the same `ggml_cuda_fwht_warp<64, 32>` and the same `ggml_cuda_quantize_q8_0_group` lane-to-element map. The standalone `fwht_cuda<64>` launch goes away.
- The rotated V must have no other reader (single use, not an output), and its input must not overlap the cache rows written.
- 32-wide warps only; elsewhere the old path runs.

**0003: the output rotation folded into the FA combine kernel** (`fattn-common.cuh`, `fattn.cu`, `fattn.cuh`, `fattn-qsa.cu`, `fwht.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_FA_COMBINE_FWHT=0` off). At a 1-token FLASH_ATTN_EXT whose output feeds the 64-wide output rotation and then an mmvq projection (the `ggml_cuda_fwht_q8_1_try_fuse` pattern), the dispatcher offers the rotation to the FA launch. If the combine kernel runs (`parallel_blocks > 1`), `flash_attn_combine_results<D, true>` stages each output row in shared memory, then applies the same FWHT, stores and Q8_1 rounding as `fwht_cuda<64, float, true>`. The rotation output is still written. The projection finds its Q8_1 input in the cache and skips its quantize.
- If the combine does not run, the rotation runs as before. The Q8_1 cache entry is created only on the path that fills it.
- The plain `flash_attn_combine_results<D>` instantiation is unchanged.

**0004: Q / K norm+rope pair for a quantized cache** (`rope.cu`, `rope.cuh`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_QK_PAIR_FWHT=0` off). As with your F16 pair, Q's norm+rope+FWHT is deferred to the V-rotation + K site seven nodes later (only views in between, the 0001 order). Both run as one `rms_norm_mul_rope_f32_qk_pair<has_ff, float, float, fwht_n, true>` launch. Each block runs the same body as the single launch, so results do not change.
- Q's rotation output is checked against everything the K half reads, and the K / V cache rows against everything the Q half reads.
- If a deferred Q is left without its K partner, the existing fallback now runs Q's norm+rope+FWHT as a single launch.

**0005: rope + KV rotation in one launch, for attention without Q / K norm** (`rope.cu`, `ggml-cuda-try-fuse.cu`; `GGML_CUDA_FUSE_ROPE_FWHT=0` off). A ROPE with no RMS_NORM / MUL in front, followed by the rotation (RESHAPE, Hadamard MUL_MAT), runs as one `rms_norm_mul_rope_f32<.., no_norm = true>` launch: the fused norm+rope kernel with the norm skipped and x read as is. For K it also takes the RESHAPE, VIEW and q8_0 SET_ROWS that follow, and the V row's VIEW and q8_0 SET_ROWS where allowed.
- NORMAL / NEOX rope with the full rotation only (`n_dims == ne0`, no offset); the rope arithmetic is rope_norm's / rope_neox's, the same as your norm+rope fusion uses.
- The same hazard checks as the norm form. Any token count, so prefill takes it too.
- On Qwen3-30B and Qwen3.6 it does not fire (their ROPE follows a norm).

**Gates** (stock vs the series, 7900 XTX; 0001-0004 on r183 and again on r184, 0005 on r184):
- llama-server greedy, 4 chats x 3 turns: byte-identical replies on Qwen3-30B-A3B, Qwen3.6-35B-A3B and Llama-3.2-1B with q8_0 KV, and on Qwen3-30B-A3B with F16 KV.
- wikitext-2 KLD, 24 chunks, `-c 512`, q8_0 KV, base = stock: KLD 0, PPL identical (Qwen3-30B 8.433900, Qwen3.6 5.910709, Llama-3.2-1B 14.985723; the last one exercises 0005 in prefill).
- Because prefill never takes the 1-token paths of 0001-0004, also a decode-path check: `llama-perplexity -b 1 -ub 1`, 8 chunks, q8_0 KV, Qwen3-30B. Every token is its own graph. Result: Mean PPL(Q) 8.763026 and max KLD 0.000054 for both stock and the series. Stock against its own base also gives 0.000054, so this is the stored-logit precision, not a difference. The same check on Llama-3.2-1B with 0005: 14.737627 and 0.000057 for both; and on Qwen3-30B at `-c 8192` (one chunk, so the combine runs with many blocks per row): 9.485929 and 0.000057 for both.
- 0 GPU faults. `test-backend-ops -b ROCm0` with the series: ROPE 450/450, FLASH_ATTN_EXT 6450/6450, SET_ROWS 159/159.
- Kernel census, Qwen3-30B-A3B q8_0 KV decode on the 7900 XTX, per token: 726 -> 510 launches. Per layer, Q / K / V become one row-concat GEMV, Q and K norm+rope one pair launch, and both `fwht_cuda` launches are gone.

**RX 7900 XTX (gfx1100), stock r184** (tree `f2141780f11a`; with 0001-0004 `e47f7b3ee7d4`, with 0001-0005 `9e04bb10e3bb`; `-DAMDGPU_TARGETS=gfx1100`, your default env; llama-bench `-fa 1 -p 0 -n 128 -d 0,4096 -ctk q8_0 -ctv q8_0 -r 5`, three alternating rounds after a warm-up, mean of the three). Each row adds one patch: the full build with the later switches off.

| Qwen3-30B-A3B Q4_K_M, q8_0 KV | tg128 | @4096 |
|---|---|---|
| stock r184 | 172.3 | 159.2 |
| 0001 | 177.2 (+2.8 %) | 163.9 |
| 0001-0002 | 179.8 | 166.6 |
| 0001-0003 | 185.9 | 170.5 |
| 0001-0004 | 194.0 (+12.6 %) | 177.2 (+11.3 %) |

Deeper, same flags with `-r 3`, two rounds: @16384 131.3 -> 143.9 (+9.5 %), @32768 111.2 -> 120.6 (+8.5 %). pp512 with q8_0 KV: 4738 stock vs 4729 (within noise).

**Radeon AI PRO R9700 (gfx1201)**, the same four patches on r183 (the only difference to r184 is #254's server code): `-fa 1 -p 0 -n 128 -d 0,4096 -ctk q8_0 -ctv q8_0 -r 5`, three alternating rounds.

| Qwen3-30B-A3B Q4_K_M, q8_0 KV | tg128 | @4096 |
|---|---|---|
| stock r183 | 148.5 | 136.8 |
| 0001-0004 | 164.5 (+10.8 %) | 151.4 (+10.7 %) |
| 0001-0004, all four switches 0 | 148.5 | 137.1 |

Greedy replies on the R9700 are byte-identical to stock (Qwen3-30B-A3B and Qwen3.6-35B-A3B, q8_0 KV). The census there goes from 737 to 518 launches per token: the V / K / Q row concat, the pair and the combine fold all fire on gfx1201. With 0005 (kvrot-r184), Llama-3.2-1B q8_0 KV on the R9700: tg128 394.3 stock -> 401.2 (0001-0004) -> 429.3 (+8.9 %), @4096 371.0 -> 376.9 -> 401.2 (+8.1 %); 314 -> 233 launches per token; greedy replies byte-identical.

**0005, Llama-3.2-1B UD-Q4_K_XL, q8_0 KV, 7900 XTX** (`-fa 1 -p 512 -n 128 -d 0,4096 -ctk q8_0 -ctv q8_0 -r 5`, three alternating rounds):

| | tg128 | @4096 |
|---|---|---|
| stock r184 | 469.8 | 432.3 |
| 0001-0004 | 478.0 | 439.2 |
| 0001-0005 | 502.8 (+7.0 %) | 459.7 (+6.3 %) |

Only 0003 fires on Llama without 0005. pp512 is within its run-to-run noise; F16 KV decode is unchanged (stock 523.9, 0001-0005 524.1).

Notes:
- Prefill: 0001-0004 are 1-token only; 0005 also runs in prefill (Llama KLD above), with pp512 within noise.
- F16 KV takes none of these paths. I did see Qwen3-30B F16 decode come out 0.8 % slower on builds of 0001-0004. The same build with all four switches off shows it too, and adding one unused `__global__` to `rope.cu` brings it back to stock. That points to code layout in the binary rather than to the patches. Flagging in case you see it; nothing needed from you.
- Qwen3.6-35B-A3B with q8_0 KV: none of the five fire. Its gated attention breaks the patterns, and it has only 10 attention layers.
- MTP: on the R9700, Qwen3.8-27B NVFP4-MTP with q8_0 KV gives byte-identical replies to stock with and without `--spec-type draft-mtp --spec-draft-n-max 3` (0001-0004; their verify batches keep the stock path, being wider than one token).
- Untested: multi-GPU and `-sm tensor`; 0005 with MTP (it would take a verify batch on a model without Q / K norm).

Files: `0001-*.patch` to `0005-*.patch` (`git format-patch` against the r184 tree, apply in order with `git am`) and this README. No rush on any of this; happy to run further tests on either card.

(Claude-assisted, as before.)
