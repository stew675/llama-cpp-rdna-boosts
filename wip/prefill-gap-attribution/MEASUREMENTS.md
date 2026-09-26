# MEASUREMENTS — frozen prefill baseline (2026-09-26)

Box: AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.14, `~/llama.cpp` @ `527d39401`, `build-rocm`.
Reproduce with `tools/attribution.sh`.

---

## 1. Whole-model prefill throughput (1 GPU, 27B, pp8192)

| model | pp8192 t/s | file |
|---|---:|---|
| Q8_0 | 1371.25 | `Qwen3.8-27B-Q8_0.gguf` |
| Q6_K | 975.19 | `Qwen3.8-27B-Q6_K.gguf` |
| Q4_K_XL | 1253.60 | `Qwen3.8-27B-UD-Q4_K_XL.gguf` |

## 2. Where prefill GPU time goes (kernel-trace attribution, 2 passes folded)

27B, 1 GPU, pp8192, `rocprofv3 --kernel-trace` (PMC counters return **0** on gfx1201; timestamps work):

| class | Q8_0 | Q6_K | Q4_K_XL |
|---|---:|---:|---:|
| **mmq** | **79.4 %** | **80.0 %** | **81.1 %** |
| attn | 5.5 % | 3.7 % | 4.8 % |
| other (glue) | 6.8 % | 11.1 % | 6.5 % |
| quant | 2.5 % | 1.3 % | 2.4 % |
| gdn | 2.5 % | 1.7 % | 2.2 % |
| norm | 2.3 % | 1.6 % | 2.1 % |
| rope | 0.6 % | 0.4 % | 0.5 % |
| glue / mmvq / copy | ~0.3 % | ~0.3 % | ~0.3 % |

Top kernels (Q8_0 pp8192): `mul_mat_q` 9109.5 ms (79.4 %), `flash_attn_ext_f16<256,256,32,2,…>` 628.5 ms
(5.5 %), `unary_gated_op_kernel<silu>` 390.7 ms (3.4 %), `quantize_mmq_q8_1` 287.0 ms (2.5 %).

### Long context (Q8_0, pp32768)

| class | pp8192 | pp32768 |
|---|---:|---:|
| mmq | 79.4 % | **69.8 %** |
| attn | 5.5 % | **17.1 %** |
| rest | ~15 % | ~13 % |
| pp32768 t/s | — | 1192.24 |

⇒ attention becomes material only at long context, and even there GEMM is ~70 %.

## 3. The precision confound

`/llm/models/Qwen3.8/27B/FP8/` is HF **safetensors** (`quant_method: fp8`, `fmt: e4m3`,
`activation_scheme: dynamic`), 29 GB, `Qwen3_5ForConditionalGeneration`. llama.cpp has **no FP8
weight type** (`grep F8_E4M3 ggml/include/ggml.h` → 0 hits) and loads GGUF, so it cannot load this
checkpoint. Any "llama.cpp vs vLLM" prefill ratio on this box is therefore engine **and** precision.

## 4. Reference numbers carried from the FP8 work (see `FP8-PRIOR-ART.md`)

| config (Qwen3.5-4B, 1× gfx1201) | pp512 t/s | tg64 t/s |
|---|---:|---:|
| Q8_0 GGUF | 6177 | 90.9 |
| fp8 (L9 single-copy embd, `origin/cllm`) | **7162-7170** | 88.3-88.6 |
| fp8 (L7 in-proj fp8) | 7184-7260 | 88.3-88.5 |
| fp8 aiter bf16-GEMM port earlier | 5546-5596 | 71.4 |

AITER fp8 blockscale GEMM on gfx1201, M=512, 4B shapes: **121-137 TFLOP/s** (raw WMMA ceiling ~370).
`mul_mat_fp8_wmma` (cllm) measured ~**77 TFLOP/s** effective (~92-98 hot-loop).
