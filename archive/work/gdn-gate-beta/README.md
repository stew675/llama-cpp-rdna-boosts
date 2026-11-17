Thanks for the 1-token GDN gate/beta fusion (`ggml_cuda_gate_beta_bf16_try_fuse`) and the ksplit mmvq body. These three patches extend both to the projection types many published Qwen3.5 / 3.6 / 3.8 GGUFs use. The output is the same as stock, with one switch per piece. Offered in case it's useful; nothing is urgent on our side. They are in our own production build (since 2026-11-16).

**What they do** (`mmvf.cu`, `mmvq.cu`, `mmvq-ksplit-body.inc`, `mmvq.cuh`, `ggml-cuda.cu`, `ggml-cuda-try-fuse.cu`, `common.cuh`; +319 / -15)

A GDN layer's `ssm_alpha` / `ssm_beta` projections feed `ADD(dt) -> SOFTPLUS -> MUL(ssm_a)` and `SIGMOID`. Today that chain is fused only when the projections are BF16; other types run each step as its own launch.

- **0001** F32 projections (unsloth's UD quants of Flash-Next store `ssm_alpha` / `ssm_beta` as F32): the existing gate/beta fusion with a `T = float` instantiation. Each row is the plain F32 `mul_mat_vec_f`, and the epilogue is `add_softplus_mul_bcast_f32`'s arithmetic. The host assert now accepts F32. Each type has its own switch: `GGML_CUDA_FUSE_GATE_BETA_F32=0` (this patch) and `GGML_CUDA_FUSE_GATE_BETA_BF16=0` (new, for the existing BF16 path).
- **0002** quantized projections (Q4_K / Q5_K / Q6_K / IQ4_XS / MXFP4; Qwen3.6-35B-A3B Q4_K_M, Qwen3.8-27B Q4_K_M and UD-Q4_K_S, ...). The ksplit body applies the chain in its epilogue (template flag `gdn_epi`, per row-concat segment):
  - Decode folds it into the existing alpha / beta / z row concat: 3 launches become 1 per GDN layer.
  - The verify band (2..8 tokens) runs alpha + beta as one row-concat launch with the epilogue: 5 launches become 2. The graph reorder puts z in front of the pair.
  - Every row keeps its own ksplit dot product, with the same `nwarps` / `long_k` / `small_m` as its unfused launch. There are at most 256 rows, so one row per block.
  - `GGML_CUDA_FUSE_GDN_EPI=0` turns it off, including the reorder. `GGML_CUDA_GDN_EPI_LOG=1` prints one line the first time each epilogue launch runs.
- **0003** makes 0002's memory check precise. The generic `ggml_cuda_check_fusion_memory_ranges` declined almost every match (116 of 120 on Qwen3.6-35B-A3B): the allocator places the ADD / SOFTPLUS / MUL outputs in place over the alpha matmul output, which the fused launch never materializes. The launch reads only the weights, the activation's Q8_1 copy, `dt` and `ssm_a`. So the check is now: `dt` / `ssm_a` are weights, not compute-buffer copies, and no tensor the launch writes overlaps them.

**Output:** the same as stock. Stock r298 vs r298 + the series, 1 GPU, `ROCBLAS_USE_HIPBLASLT=0`, full-logit KLD over wikitext (stock as the base) at `-ub 1` (decode), 4 and 8 (verify band):

| model (`ssm_alpha` type) | PPL at ub 1 (`-c 512`) / ub 4 (`-c 2048`) | mean KLD (ub 1/4/8) | same top token | max KLD |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M (Q4_K) | 6.1554 / 6.1514 | 0.000000 | 100 % | 0.000042-0.000054 |
| Qwen3.8-27B UD-Q4_K_S (MXFP4) | 6.2042 / 6.3330 | 0.000000 | 100 % | 0.000051-0.000052 |
| Qwen3.6-35B-A3B Q4_K_M (Q4_K) | 5.8921 / 5.7627 | 0.000000 | 100 % | 0.000045-0.000059 |
| control: Qwen3.6-35B-A3B, stock vs stock | | 0.000000 | 100 % | 0.000045-0.000059 |

On Qwen3.6-35B-A3B the maximum KLD is identical to the stock-vs-stock control; the 27B rows are of the same size. Server replies (`-c 65536`, a greedy reply and a 30k-token-depth reply) were byte-identical between stock and the series on all three models. For F32, Flash-Next UD-IQ3_XXS at `-ub 1` (512 tokens, 2 GPUs, your host-pool setup) gave PPL 1.8622 for both.

**Measurements** (2x Radeon AI PRO R9700, gfx1201, ROCm 10.1; stock r298 `99c4717c6`, tree `5f55d95a5`; the files these patches touch are byte-identical in r299, and the series applies there unchanged). Greedy decode t/s, 1 GPU, two requests each:

| | stock | series | |
|---|---|---|---|
| Qwen3.8-27B Q4_K_M, MTP n-max 3 | 64.7 / 65.0 | 65.4 / 65.7 | +1.1 % |
| Qwen3.8-27B UD-Q4_K_S (MXFP4), MTP n-max 3 | 71.0 / 70.8 | 71.7 / 71.4 | +0.9 % |
| Qwen3.6-35B-A3B Q4_K_M | 145.5 / 146.4 | 147.4 / 147.7 | +1.1 % |
| Qwen3.6-35B-A3B Q4_K_M, two concurrent requests (sum) | 201.2 | 206.2 | +2.5 % |

- **F32 (0001):** the separate softplus launch is gone, and the alpha/beta matvecs run as the fused launch. This is from a rocprofv3 kernel trace of the 512-token `-ub 1` run on Flash-Next UD-IQ3_XXS (36 GDN layers with F32 `ssm_alpha` / `ssm_beta`), in your host-pool setup on 2 GPUs: 1,895,362 kernel launches with stock, 1,821,342 with the series (-3.9 %). Decode wall time in that setup did not change measurably: greedy 101.8 / 114.2 vs 104.9 / 113.7 t/s, 512-token decode 110.8 / 114.1 vs 110.9 / 115.9 t/s. Expert streaming dominates there.
- **Faults:** 0 GPU faults in the 8 server runs and 2 traced runs.

**Limits / not tested:** NVIDIA / CUDA, RDNA3, gfx1151, Windows; projection types other than Q4_K, MXFP4 and F32 (Q5_K / Q6_K / IQ4_XS take the same code path, but we had no local GGUF with them).

Env: `GGML_CUDA_FUSE_GATE_BETA_F32=0`, `GGML_CUDA_FUSE_GATE_BETA_BF16=0`, `GGML_CUDA_FUSE_GDN_EPI=0` (each default on), `GGML_CUDA_GDN_EPI_LOG=1` (diagnostic).

Files: `0001-0003-*.patch` (`git format-patch` against r298; apply to r299 with `git am`), this README, and `logs/`. Happy to run anything else on these cards.

(Claude-assisted, as before.)
