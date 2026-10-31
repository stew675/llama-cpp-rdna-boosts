Thank you for folding #258 so quickly, and for the extra coverage: 3-GPU tensor/layer, MTP acceptance and gpt-oss for 0005. This picks up the scope gap you recorded as TODO #190, for q4_0, the type you mentioned for 16 GiB cards.

With `-ctk q4_0 -ctv q4_0` on r187, #258's 0001 (QK rotate / row concat) and 0003 (combine rotation) already fire. What stays per layer is the cache write: K and V each run their own q4_0 `k_set_rows_quant`, V's rotation runs as `fwht_cuda<64>`, and Q and K take two norm+rope launches instead of the pair. Qwen3-30B-A3B decode runs 702 launches per token with q4_0 KV, against 510 with q8_0. This patch brings it to 510. One patch on r187; setting `GGML_CUDA_KV_FOLD_Q4_0=0` turns it off. It applies on wave32 devices only (the group quantizer ballots over one 32-lane wave); wave64 keeps the separate launches.

**0001: fused K / V cache write for a q4_0 cache** (`set-rows.cuh`, `rope.cu`, `ggml-cuda-try-fuse.cu`).
- `ggml_cuda_quantize_q4_0_group` quantizes one block from 32 lanes (lane = element), next to the q8_0 one. It matches `quantize_f32_q4_0_block` bit for bit:
  - the scale comes from the element the sequential `amax < fabsf(v)` scan keeps: an `fmaxf` butterfly for the maximum, then a ballot for the lowest lane holding it, i.e. the first on a tie;
  - a NaN never wins, and an all-zero group keeps vmax = +0, so d = -0.0 as in the reference;
  - each element is rounded with the reference's two statements, and lane j packs element j with element j + 16.
- The kernels take the cache type as a template value: `kv` goes from `bool` to `int` (1 = q8_0, 2 = q4_0). The q8_0 instantiations compile as before. The host picks kv = 2 at the three dispatch sites (norm form, rope-only form, Q / K pair).
- The two matchers accept q4_0 through `ggml_cuda_kv_fold_type_ok` (q8_0, or q4_0 on wave32: the group quantizer ballots over one 32-lane wave). The plain V write is folded only when V's type equals K's, as the V-rotation matcher already required. (r187 refuses mixed K / V cache types for these models anyway, so this only guards the general case.)

**Gates** (stock r187 vs the patch; 7900 XTX unless noted):
- llama-server greedy, 4 chats x 3 turns: byte-identical replies on Qwen3-30B-A3B, Qwen3.6-35B-A3B and Llama-3.2-1B with q4_0 KV on both cards, and on Qwen3-30B-A3B and Qwen3.6 with q8_0 KV on the 7900 XTX.
- Decode-path check (`llama-perplexity -b 1 -ub 1`, 8 chunks, q4_0 KV): Qwen3-30B PPL 9.060447 and max KLD 0.000064 for both; Qwen3.6 6.189585 and 0.000059 for both. Stock against its own base gives the same maxima, so this is the stored-logit floor.
- Kernel census, Qwen3-30B-A3B q4_0 KV decode: 702 -> 510 launches per token. The q4_0 `k_set_rows_quant` and `fwht_cuda<64>` launches are gone, and the pair runs as `rms_norm_mul_rope_f32_qk_pair<.., 128, 2>`, 7.8 us vs 7.1 us for the q8_0 one.
- 0 GPU faults.

**RX 7900 XTX (gfx1100)**, llama-bench `-fa 1 -p 0 -n 128 -d 0,4096 -ctk q4_0 -ctv q4_0 -r 5`, three alternating rounds; the stock column uses the last two, because its first round ran cold:

| q4_0 KV | stock r187 | patch | patch, switch 0 |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M tg128 | 171.7 | **190.9 (+11.2 %)** | 171.5 |
| @4096 | 158.3 | **173.7 (+9.7 %)** | 158.1 |
| Llama-3.2-1B UD-Q4_K_XL tg128 | 473.6 | **497.6 (+5.1 %)** | 472.5 |
| @4096 | 438.0 | **457.4 (+4.4 %)** | 437.3 |
| Qwen3.6-35B-A3B Q4_K_M tg128 | 166.7 | 166.3 | 166.1 |

q8_0 KV is unchanged (Qwen3-30B 193.3 vs 193.5, Qwen3.6 164.2 vs 163.5, two rounds).

**Radeon AI PRO R9700 (gfx1201)**, the same patch and flags, three rotated rounds:

| q4_0 KV | stock r187 | patch | patch, switch 0 |
|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M tg128 | 149.5 | **163.6 (+9.4 %)** | 150.6 |
| @4096 | 137.3 | **150.6 (+9.7 %)** | 138.4 |
| Llama-3.2-1B UD-Q4_K_XL tg128 | 408.3 | **428.5 (+4.9 %)** | 408.1 |
| @4096 | 384.3 | **402.5 (+4.7 %)** | 384.1 |
| Qwen3.6-35B-A3B Q4_K_M tg128 | 133.5 | 133.8 | 134.2 |

Stock's first Qwen3-30B round ran cold; without it the gain is +9.8 % / +9.2 %. Census there: 713 -> 518 launches per token. q8_0 KV unchanged (Qwen3-30B 167.0 vs 166.8, Qwen3.6 137.4 vs 137.1). Greedy replies byte-identical on all three models with q4_0 KV.

Notes:
- Qwen3.6 is flat. Its fused K write already fires with q8_0 (since before #258), and with q4_0 it now replaces three launches with one, but the 256-wide rotation in one warp takes about as long as the launches it saves.
- An earlier cut picked the quantizer at run time. That slowed the q8_0 kernel on Qwen3.6 (12.8 -> 15.1 us), hence the template value.
- q5_0 has the same shape (vmax / -16 plus the high bits) and could follow the same way. iq4_nl's quantizer searches its scale, so it's a separate piece of work.

Files: `0001-*.patch` (`git format-patch` against r187, apply with `git am`; it also applies unchanged to r188, which doesn't touch these three files) and this README. No rush on any of this.

(Claude-assisted, as before.)
