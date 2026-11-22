Thank you for the fused hc mixer. It replays the unfused mmvq arithmetic bit for bit, and the header comment names the reference it emulates. That let us check these two changes against the reference by hash. They touch only the Q8_0 path of `ggml_cuda_op_hc_mix_q8`, which Flash-Next UD quants take (Q8_0 hc weights). In our tests the outputs were bit-identical to stock (hashes below).

**What it does** (`ggml/src/ggml-cuda/hc-mix.cu` only)

- **0001** `hc_mix_down_dots` ran one block per (row, token), so the 3.5 MB `w_down` was read once per token of the verify band. The new `hc_mix_down_dots_mt<NT>` serves every token from one block per row. Per token, the item order, the dp4a dot, the warp butterfly and the fixed-order sum of the 8 warp partials are the per-token kernel's. The F32 inject rows keep one block per (row, token). They are the longest blocks, and looping over tokens inside them would lengthen the kernel's tail. The v = silu(lo/hc) quantize uses one completion counter per q8_1 group instead of one per (token, group): each block writes lo for every token before it increments the counter, so the group's 32nd increment means every token's group is complete. `GGML_CUDA_HC_DOWN_MT=0`: the per-token grid.
- **0002** `hc_mix_up_collapse_q8` emulated eight reference blocks per fused block (16 KB LDS partials, a butterfly per (warp, token, row)). In the reference (rpb 16, vdr 2, lane = 4*g + p, items it = 8*warp + g + 64*m), a row's 10 items fall in exactly two 8-item chunks, each one warp's partial. Every other lane and warp holds only its initial +0.0 partial, which does not change the sum. So each gate is a fixed expression: the xor-16/8/4 tree over g, then xor 2 and xor 1 over p, for each of the two chunks, then their sum. The new `hc_mix_up_collapse_q8_rows<NT>` has one thread per (row, p): it preloads the row's 10 weight items, computes the 10 dots per token, each added to a 0.0f start value as the reference's accumulator does, does the g-levels in registers and the p-levels with two lane xors. Only the gates go through LDS (`gate_s[NT][4][32]`). The collapse and its Q8_1 epilogue are unchanged. `GGML_CUDA_HC_UP_ROWS=0`: the emulating kernel.

**Per op** (`test-backend-ops perf -o HC_MIX`, `n_embd=2560, hc=4, hc_lr=320, inject=1, wtype=q8_0`; r325, ROCm 10.1, one R9700), us/run:

| n_tokens | stock | patched | patched, both kill switches `=0` |
|---|---|---|---|
| 1 | 20.64 | 16.03 | 20.70 |
| 2 | 23.68 | 18.25 | 23.15 |
| 3 | 27.45 | 19.50 | 27.22 |
| 4 | 31.19 | 22.01 | 31.24 |
| 5 | 35.40 | 24.57 | 35.62 |
| 8 | 49.27 | 31.24 | 49.40 |

The `wtype=bf16` cases are unchanged (26.3 us at n_tokens 4). `test-backend-ops -o HC_MIX`: 30/30 pass on the patched build.

**Server, bit-identity.** r325, ROCm 10.1, 2 x R9700, `-sm tensor`, Flash-Next UD-IQ4_XS, your PR #304 pool settings (`MOE_HOST_POOL_RAM_PCT=0`, `LLAMA_KV_REGION_POOL=0`, headroom 1536, `GGML_CUDA_ALLREDUCE=ce`, `--load-mode dio`) with `--n-cpu-moe 48 -lme 36 -ub 4096 -c 65536` so the model loads in our 59 GB of RAM, MTP expQ4K head. Stock, patched, and patched with both switches `=0` gave the same greedy, 512-token prompt-set, 4.4k-prompt and 30k-token summary hashes; 0 faults. In this config the host experts are read from pageable (unpinned) memory and dominate the run (short-prompt prefill ~20 t/s), so the decode differences are diluted (512-token prompt set, stock / patched / both `=0`: 65.9 / 69.5 / 64.9 t/s); the per-op table is the cleaner comparison.

**On our tree** (same two changes; UD-IQ4_XS, 48 CPU-MoE layers on our mmap host experts, ROCm 10.2, one run each, kill switch as control): decode +1-2 % for 0001 (small, but consistent across the tests and confirmed by the kernel trace) and +3.5-8 % for 0002 on top of it (0002 alone, 512-token prompt set: 136.0 -> 147.3 t/s); every output hash identical. Kernel trace at 4 tokens: down dots 13.5 -> 9.9 us, up + collapse 18.5 -> 8.9 us.

**Base:** formatted against and measured on r325 (`v17-4f9dfa0e3-r325`). r326 and r327 do not touch `hc-mix.cu`.

**Limits / not tested:**
- Only the Q8_0 hc path; the BF16 path (the GSQ quants) is untouched. It emulates a different reference (mmvf), so it would need its own design; a similar restructure may help there, but we have not tried it.
- RDNA4 (gfx1201) only. 0002 assumes the vdr 2 item geometry (`static_assert`) and the existing `fuse` guard (hc 4, hc_lr 320, n_tokens <= 8).
- NVIDIA and Windows: not tested.

Nothing is urgent on our side. We are happy to run any further checks on these cards if they would help.

Files:
- `0001-*.patch`, `0002-*.patch`: `git format-patch` against r325; apply with `git am`.
- This README.
- `logs/ab-r325.log`: the server A/B and the `test-backend-ops` runs above.

(Claude-assisted, as before.)
