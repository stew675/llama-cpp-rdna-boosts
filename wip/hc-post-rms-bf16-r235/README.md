Thanks for the `bf16_src1_cache` in the cuBLAS BF16 path and the `dst16` side outputs in the hc kernels. They made this small, and r235 has been our production base since this morning. Below is a speedup for qwen4exp prefill under `-sm tensor`, offered in case it's useful. Nothing is urgent on our side; we already run it.

Under `-sm tensor`, each qwen4exp layer ends with `DSV4_HC_POST`, and the next layer normalizes that output with `RMS_NORM` + `MUL(gamma)`. That's two kernels and an extra read of the 10240-wide stream. The normalized stream then feeds the BF16 `hc_*_down` / `hc_*_inject` GEMMs. On RDNA4 those run on rocBLAS (MMB does not take BF16 dense weights there), so each call first converts the whole activation to BF16. For a 4096-token ubatch that is 252 MB read and written, about 460 us per call. One patch on stock r235 with two switches:

**0001: `DSV4_HC_POST -> RMS_NORM -> MUL` as one kernel, plus producer-written BF16 copies** (`norm.cu`, `norm.cuh`, `dsv4-hc.cu`, `common.cuh`, `ggml-cuda-try-fuse.cu`, `ggml-cuda.cu`)
- `dsv4_hc_post_rms_f32` computes the hc_post sum with the same statements as `dsv4_hc_post_f32` and stores it (the next residual). It then does the scalar-path reduction, scale and gamma product of `rms_norm_f32`, so the output is bit-identical to the unfused ops.
- The matcher keeps x / residual / post alive until the MUL. An overlap check falls back to the unfused kernels, and the fusion stands down under `GGML_CUDA_RMS_NORM_VEC=1`, whose float4 sum order differs. `GGML_CUDA_HC_POST_RMS=0` turns it off.
- `dsv4_hc_post_rms` (xn, for the hc GEMMs) and `dsv4_hc_pre` (the mixed stream, for the router / indexer / beta-alpha GEMMs) also write the RNE BF16 copy of their output in the same pass. They leave it in `bf16_src1_cache`, so `ggml_cuda_mul_mat_cublas_impl` finds it instead of converting. The F32 output is always written.
- The rounding is the same integer RNE as `__float2bfloat16` for every finite value, ±0, denormals and ±Inf; only NaN payloads could differ. The conditions match the lookup's (T >= 64, contiguous root tensor), so decode is untouched. `GGML_CUDA_BF16_PRODUCER=0` turns this part off, and `=2` is a debug mode that re-converts every cache hit and compares element by element (it skips hits inside a graph capture).

**Gates** (2x Radeon AI PRO R9700, gfx1201, ROCm 10.1; stock r235 `77b4f43a3`, tree `ef5dcbcb0ae7`):
- Qwen3.8-Flash-Next GSQ-RCO IQ3_S with your host-pool setup: `-sm tensor --n-cpu-moe 47 -lme 47 -ub 6144 -b 6144 --load-mode dio --lazy-mode auto --cache-ram 1024`, `MOE_HOST_POOL_RAM_PCT=0 LLAMA_KV_REGION_POOL=0 GGML_CUDA_SLAB_HEADROOM_MIB=1536 MOE_EXPERT_HEATMAP=load`, MTP head, `-c 262144`.
- Identical output in every arm (patch on, each switch off, both off): greedy, the 4.4k-token prompt set and a 101k-token prompt give the same hashes as stock. 0 GPU faults.
- `GGML_CUDA_BF16_PRODUCER=2` over the same prompts: 32,501 cache hits checked, 0 mismatching BF16 elements.
- 7 other models (Llama-3.2-1B, Qwen3-30B-A3B, Qwen3.6-35B-A3B, gemma-4-26B-A4B, Qwen3.8-27B, Qwen3.8-27B NVFP4-MTP, Qwen3.5-4B), llama-server multi-turn replies vs stock r235 (14 pairs): all identical, 0 faults; KLD 0 on all 6 KLD checks. `test-backend-ops -b ROCm0`: 21679/21680. The one failure, `MUL_MAT f16 m=16 n=2105344 k=32`, also fails on stock r235.

**Prefill, same settings** (t/s, second of two prefills per server; "a / b" = two servers of the same arm, arms alternated):

| | both off (= stock) | `HC_POST_RMS` only | patch |
|---|---|---|---|
| 31.5k-token prompt | 2143 / 2140 | 2223 (+3.8 %) | **2262 / 2263 (+5.6 %)** |
| 101k-token prompt | 2088 / 2088 | 2179 (+4.4 %) | **2209 / 2214 (+5.9 %)** |

Decode speed and greedy output are unchanged (105-106 t/s in every arm).

On our production build (r235 + our patches, `--host-experts mmap`, `-ub 4096`, same two cards), measured as each part was added: the fusion +6.2 % (31.5k-token prompt), the BF16 copies a further +2.0-2.5 %.

Notes:
- Untested: one GPU, layer split, RDNA3 / gfx1151. The fusion is matched only for the `DSV4_HC_POST -> RMS_NORM -> MUL` sequence that qwen4exp emits under `-sm tensor`. On gfx1151 your MMB HC16 marks keep their path; the new producer writes only when no MMB copy was requested.

Files: `0001-*.patch` (`git format-patch` against r235, apply with `git am`) and this README. No rush; happy to run anything else on these cards.

(Claude-assisted, as before.)
