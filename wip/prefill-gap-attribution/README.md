# wip/prefill-gap-attribution — where the single-request prefill gap actually is

**Status: ACTIVE exploration (2026-09-26). NOT delivery work.** Nothing here is in `patches/`;
`apply-all.sh` ignores it. Nothing reaches `main` without the maintainer.

**Question.** llama.cpp's single-request prefill on RDNA4 tops out at roughly **75 % of vLLM's**.
Is there a viable path forward, and does `mudler/vllm.cpp` supply it?

**Answer (measured 2026-09-26):** the gap is a **GEMM** problem (70-81 % of prefill GPU time is
`mul_mat_q`), so vllm.cpp's differentiators (paged KV, continuous batching, prefill-attention ports)
cannot close it, and vllm.cpp's own GEMM is behind llama.cpp's. **vllm.cpp is not the path.** The
path is the maintainer's own **`origin/cllm` FP8 E4M3 port**, which already measured **+13-17 %
prefill over Q8_0** on gfx1201, plus **AMD AITER's** gfx1201 fp8 GEMM tuning (121-137 TFLOP/s vs the
cllm kernel's 77-98) to close the rest.

| doc | what |
|---|---|
| `MEASUREMENTS.md` | the frozen attribution baseline (what prefill time is made of) |
| `RESULTS-2026-09-26.md` | the dated findings: attribution, the precision confound, the vllm.cpp verdict |
| `FP8-PRIOR-ART.md` | the inventory of `origin/cllm` + AITER — the actual path forward |
| `HANDOVER.md` | cold-start brief + next-session prompt |
| `tools/` | the rocprofv3 fold + repro scripts |

---

## 1. The measurement that decides it

`rocprofv3 --kernel-trace` (durations work on gfx1201; PMC counters return 0), 27B, 1 GPU, pp8192:

| class | Q8_0 | Q6_K | Q4_K_XL |
|---|---:|---:|---:|
| **`mul_mat_q` (GEMM)** | **79.4 %** | **80.0 %** | **81.1 %** |
| attention (`flash_attn_ext`) | 5.5 % | 3.7 % | 4.8 % |
| glue (silu/add/norm/rope/…) | ~9 % | ~11 % | ~9 % |
| `quantize_mmq_q8_1` | 2.5 % | 1.3 % | 2.4 % |
| GDN | 2.5 % | 1.7 % | 2.2 % |

At pp32768 (Q8_0) attention grows to 17.1 %, **GEMM is still 69.8 %**.

**Consequence:** even a *perfect* attention kernel buys ≤ 5 % at 8k / ≤ 17 % at 32k. The lever is the
GEMM. (This also definitively answers the earlier vllm.cpp question: their prefill-attention work
cannot move the number.)

## 2. The comparison is also confounded

The FP8 checkpoint at `/llm/models/Qwen3.8/27B/FP8/` is **HF safetensors** (`quant_method: fp8`,
`fmt: e4m3`, `activation_scheme: dynamic`, arch `Qwen3_5ForConditionalGeneration`), not GGUF.
llama.cpp today has **no FP8 weight type at all** (`ggml.h`: only a comment "in theory the library can
be extended to support FP8") and loads GGUF. So the 75 % figure compares **vLLM-FP8 vs
llama.cpp-int8/Q4_K** — engine *and* precision. That is not a pure engine gap.

## 3. The findings

* **vllm.cpp builds on gfx1201** (its CMake says the ROCm skeleton "has no build report from any
  machine" — we are likely first). But it registers **no ROCm FP8 GEMM** (only `kReshapeAndCacheFp8`
  and an fp8 decode GEMV), its own gfx1151 survey puts it *behind* llama.cpp, and its `#2109` says
  its keep-quant GEMM has no tensor-core arm and names *llama.cpp's RDNA3 WMMA* as the arm to port.
  → **Assessed; not viable.** Keep it only as a reference for vLLM/SGLang prefill algebra
  (`src/vt/rocm/rocm_paged_attn.hip`) and its benchmark discipline.
* **`origin/cllm` is the path.** It is a complete FP8 E4M3 port: `GGML_TYPE_F8_E4M3`, an
  **aiter-style RDNA4 WMMA GEMM** (`mul_mat_fp8_wmma`), runtime activation staging (`quantize_fp8`),
  direct safetensors loading, `--outtype fp8_e4m3` GGUF conversion, and a chunked-GDN rewrite.
  Measured on Qwen3.5-4B / gfx1201: **pp512 7184-7260 t/s vs Q8_0 6177 (+16-17 %)**, tg64 -1.5 %.
* **AITER is what vLLM actually uses on ROCm.** On gfx1201 the CK path is CDNA-only, so AITER
  dispatches to **Triton JIT** fp8 blockscale GEMM. Measured here at **121-137 TFLOP/s** on the 4B
  shapes (raw gfx1201 WMMA ceiling ~370 TFLOP/s → ~35 %, normal for a staged kernel), against the
  cllm kernel's 77-98 TFLOP/s. That difference is the remaining gap.

## 4. Plan

1. **Re-base `origin/cllm` onto the current base** and re-measure the 4B StewFP8 numbers to confirm
   the +13-17 % survives. (22 commits / 44 files / +6606 lines over merge-base `6ea215d17`,
   2026-08-05 → ~7 weeks of drift from `84e76d8a2`.)
2. **Convert the 27B FP8 safetensors to F8_E4M3 GGUF** with the cllm converter and measure 27B fp8
   prefill vs Q8_0/Q6_K on gfx1201 (the direct comparison the user actually wants).
3. **Lift AITER's gfx1201 fp8 GEMM tuning** (tile/GROUP_M/kpack, M-bound config selection) into the
   WMMA kernel to close the 77-98 → 121-137 TFLOP/s part.
4. If it holds up, promote through the normal route (`beta/` → env-gated A/B → delivery block).

## 5. Success criteria

| gate | threshold |
|---|---|
| 4B fp8 re-base | reproduces pp512 ≥ 7184 t/s (+16 % vs Q8_0) on gfx1201 |
| 27B fp8 prefill | beats the int8 MMQ pp8192 baseline (Q8_0 1371 t/s) |
| fp8 GEMM kernel | ≥ 110 TFLOP/s effective on the large 27B shapes (AITER 121-137) |
| purity | greedy text matches the fp8/vLLM oracle within the cllm PPL record (6.2250 vs Q8_0 6.2464) |
