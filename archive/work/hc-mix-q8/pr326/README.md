Thanks for r304 and for landing #322 with the extra gate and tests, and for r303's row views of the merged `hc_down_inject` tensor, which the Q8_0 path below picks up unchanged. This is three patches for the hc mixer on GGUFs whose hc weights are Q8_0 (up/down) with an F32 or Q8_0 inject, which covers the unsloth UD quants of Flash-Next. They follow what your BF16 mixer already does: fewer launches per hc site, a planar output, and the combine + norm folded in. In our tests the output is byte-identical to stock; the BF16 path is untouched, and every piece has its own off switch. Offered in case it's useful; nothing is urgent on our side. We have run it in our own production server (UD-IQ4_XS) since 2026-11-13 (0002 and 0003 since 2026-11-14), through several nights of stress on that model.

**What it does** (`hc-mix.cu`, plus a few lines in `qwen4exp.cpp`, `common.cuh`, `ggml-cuda.cu`, `ops.cpp`; decode/verify band only, `nt <= HC_FUSED_MAX_TOKENS`)

On a UD file the mixer takes the Q8_0 op, which ran 8 launches per hc site where the BF16 path runs 3.

- **0001** (Q8_0 mixer path):
  - **Persistent `down_dots` counters:** per-stream, zeroed once, and the finishing block resets its own counter, so there is no memset per call (`GGML_CUDA_HC_DONE_PERSIST=0`).
  - **Planar output:** mixed then inject, as in your BF16 path, so the verify band needs no `ggml_cont`. Only for hc weights in a GPU buffer, because the CPU op keeps the packed layout (`LLAMA_HC_MIX_PLANAR_Q8=0`; `LLAMA_HC_MIX_PLANAR=0` turns off both layouts).
  - **Q8_1 epilogue:** the collapse kernel writes the mixed rows' Q8_1 into the quantize cache with `quantize_q8_1`'s arithmetic, including the `d * sum` term, so the next projection skips its quantize (`GGML_CUDA_HC_Q8_EPI=0`).
  - **Combine + norm + Q8_1 in one kernel:** `hc_mix_combine_rms_gamma_q8` (`GGML_CUDA_FUSE_HC_COMBINE_MIX_Q8=0`).
- **0002:** the up band kernel loads its weight ints and scales into registers before the prologue barrier. The dot is the same `vec_dot_q8_0_q8_1_impl` on the same values (`GGML_CUDA_HC_UP_PRELOAD=0`).
- **0003:** up + collapse run in one launch (`hc_mix_up_collapse_q8`), and the gate is summed in the band kernel's order. The F32 inject rows run as the first blocks of `hc_mix_down_dots` (`GGML_CUDA_HC_UP_FUSE=0`; `GGML_CUDA_HC_INJECT_DOWN=0` gives the inject its own launch).

On gfx1201 with UD-IQ4_XS (rocprofv3 traces), 0001 takes a hc site from 8 launches to 4, and 0003 to 3. 0002 + 0003 cut the up / collapse / down time of a 4-token site from 36.1 to 31.9 us. Prefill ubatches are above the band and take the unchanged chain.

**Output:** byte-identical. Stock r303 and r303 + these patches gave the same text on:
- the greedy reply;
- 512-token greedy decodes after fresh 4.2k- and 101k-token prompts;
- the 4.4k prompt set.

On our production tree, UD-IQ4_XS gave the same hashes with each piece on and with its switch at 0: greedy, decode, the 4.4k set and a depth test.

**Measurements** (2x Radeon AI PRO R9700, gfx1201, PCIe 5.0 x8 each, ROCm 10.1; stock r303 `db99a803d`, tree `f80b71fab`; patches on top; one run per arm).

Flash-Next UD-IQ3_XXS with your host-pool setup from #304: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 GGML_CUDA_ALLREDUCE=ce`, MTP head (`--spec-draft-n-max 3`).

| decode (t/s) | stock | patches | change |
|---|---|---|---|
| greedy, 400 tokens after a short prompt, 1st / 2nd request | 102.7 / 113.4 | 111.4 / 124.3 | +8.5 / +9.6 % |
| 512 tokens after a fresh 4.2k-token prompt, two requests | 110.1 / 115.8 | 118.0 / 123.5 | +7.2 / +6.6 % |
| 512 tokens after a fresh 101k-token prompt | 105.0 | 111.9 | +6.6 % |
| 4.4k prompt set, 8 x 128 tokens | 90.6 | 102.4 | +13 % |

The same A/B on r299 and r302 gave the same picture: +7 to +9 % greedy, +6 to +7 % on the 512-token decodes, identical hashes.

Prompt rates: on the 31.5k prompt the patched arm was faster (stock 322 / 338, patched 339 / 362), and on the 4.4k set it was slower (stock 2088, patched 1946). Across r299, r302 and r303 both have gone either way, by as much as stock alone moves between back-to-back requests in this pool setup. These patches don't touch the prefill path (ubatches above the band take the unchanged chain), so we read prompt rates as unchanged, not as a gain or a loss.

**Base and measured release:** the patches are formatted against r304 (`39824ba24`, tree `e4791d001`) and were measured on r303. `hc-mix.cu` and `qwen4exp.cpp` are byte-identical in r303 and r304, the hunks are the same on both, and a server run of the r304 build gives the same three hashes (`logs/hashes-r304.log`).

**Faults:** 0 GPU faults in the 11 UD-IQ3_XXS server runs behind these numbers (r299, r302, r303, r304).

**Limits / not tested:**
- **Merged `hc_down_inject` GGUF with Q8_0 hc weights.** The Q8_0 path takes r303's row views (`w_down_f` / `w_inject_f`, buffer probe on `w_down_buf`), and its kernels read packed rows through `->data`, which a row view keeps. We have no such GGUF to run, though. `LLAMA_FUSED_HC_MIX=0` is the reference if you have one.
- **UD-IQ4_XS** is our production file, but it doesn't fit in RAM with the host-pool settings above on our 60 GB box, so the table uses UD-IQ3_XXS.
- **Not tested:** NVIDIA / CUDA, RDNA3, gfx1151, Windows, or models other than Flash-Next. The new kernels use 32-wide blocks and `warp_reduce_*<32>`, so wave64 builds may need a look.

Env (all default on; `=0` turns that piece off): `GGML_CUDA_HC_DONE_PERSIST`, `LLAMA_HC_MIX_PLANAR_Q8`, `GGML_CUDA_HC_Q8_EPI`, `GGML_CUDA_FUSE_HC_COMBINE_MIX_Q8`, `GGML_CUDA_HC_UP_PRELOAD`, `GGML_CUDA_HC_UP_FUSE`, `GGML_CUDA_HC_INJECT_DOWN`.

Files:
- `0001`-`0003` `*.patch`: `git format-patch` against r304; apply with `git am`.
- This README.
- `logs/`: the r303 A/B, the r304 hash check, and the greedy, 4.2k-decode and 4.4k-set reply texts behind those hashes.

Happy to run any further checks on these cards if they would help.

(Claude-assisted, as before.)
