# wip/fp8-support — native FP8 E4M3 for llama.cpp (RDNA4)

**Status: PREPARED, not started (2026-09-26).**  Branch `wip/fp8-support`.  **NOT delivery work** —
nothing here is in `patches/`, `apply-all.sh` ignores `wip/`, and the fork stays scratch until the
promotion path is followed.  This tree is the ready-to-start handoff for reviving the FP8 port.

**Sibling:** `../prefill-gap-attribution/` established *why* this is the lever — single-request
prefill is **70-81 % `mul_mat_q`**, so the vLLM gap is a GEMM problem.  This campaign is the GEMM
answer: **native FP8 E4M3 + the delivery's MMB/GEMM work + AITER's gfx1201 fp8 tuning.**

---

## 1. Why FP8

* The 27B FP8 checkpoint (`/llm/models/Qwen3.8/27B/FP8/`, HF safetensors, e4m3, dynamic activation)
  is what vLLM runs.  llama.cpp today has **no FP8 weight type at all** — so the current 75 %-of-vLLM
  comparison is engine **and** precision, not a pure engine gap.
* FP8 tensor cores need **no int8 dequant epilogue** — the epilogue is exactly what makes the current
  int8 `mul_mat_q` expensive (see `archive/work/per16-f16-mma/`, `archive/work/mmq-pipeline/`).
  gfx1201's fp8 WMMA peak is ~171-370 TFLOP/s-class; the int8 MMQ runs at ~41.5 T-MAC/s.
* The port already exists and **already wins**: `origin/cllm` / `~/cllm` measured **+16-17 % prefill
  over Q8_0** on a 4B / gfx1201 (see `MEASUREMENTS.md`).

## 2. What is in this tree

| path | what |
|---|---|
| `patches/` | the **23-commit series** (`git format-patch 6ea215d17..cllm`) — the cleanest re-base unit; `git am`-able |
| `cllm-fp8-full.diff` | the **net diff** (`6ea215d17..cllm`, 7605 lines / 412 KB) for a squash or overview |
| `reference/` | raw copies of the branch's campaign docs (LEVERS, PERF_HANDOVER, AITER_FINDINGS, HANDOVER, implementation-plan, handoff, GDN_DEBUG_HANDOVER, vllm-vs-llamacpp-performance) |
| `MEASUREMENTS.md` | the frozen FP8-vs-Q8_0 scoreboard + AITER reference |
| `PLAN.md` | the phased checklist — **follow this** |
| `ROCmFPX-ASSESSMENT.md` | assessment of `ciru-ai/ROCmFPX` — no fp8 kernels, but the RDNA4 MMQ-vs-hipBLAS constraint + the DualView / ActiveFPX prefill ideas |
| `HANDOVER.md` | cold-start brief + next-session prompt |

**Provenance.** Branch `cllm` of the fork `stew675/llama.cpp`; base `6ea215d17` (2026-08-05, the
branch's stale `master`); tip `cllm` = `7c17faffc`.  `~/cllm` is a **second clone** of that fork,
**1 commit ahead** of the pushed `origin/cllm` (`535d3bcb1`) — that commit (`7c17faffc`) is unpushed
and absent from `~/llama.cpp`, so **`~/cllm` is the source of truth**, not `origin/cllm`.

## 3. What the port contains

| piece | detail |
|---|---|
| type | `GGML_TYPE_F8_E4M3` (43), self-contained 128-block layout `block_f8_e4m3: [f32 scale][128 fp8]` |
| GEMM | `ggml/src/ggml-cuda/fp8.cu/.cuh` — `mul_mat_fp8_wmma`, an **aiter-style** RDNA4 WMMA kernel: CTA 128x64, 8 warps, `GROUP_M=4` pid swizzle, 16x16x128 tiles, 2 CTAs/CU, register-staged, repacked weights; plus `quantize_fp8` activation staging and a `dot4` GEMV for decode |
| loader | `src/llama-safetensors.{cpp,h}` — direct HF safetensors loading |
| convert | `convert_hf_to_gguf.py --outtype fp8_e4m3`, `conversion/base.py`/`qwen.py` |
| CPU | `ggml-quants.c` fp8 + `getrows.cu` fp8 GET_ROWS (bit-exact vs CPU) |
| GDN | chunked-GDN prefill rewrite (phase A/B) + ssm-conv fusion — **largely already landed as delivery block 02**, so expect overlap on re-base |

## 4. The hypothesis this campaign tests

> The delivery's MMB/GEMM gains (blocks 08/13/15) + RDNA4 WMMA configs, **plus** a native FP8 E4M3
> GEMM, **plus** AITER's gfx1201 fp8 tuning, together close the vLLM FP8 prefill gap.

The three additive pieces, in order of confidence:

1. **FP8 over int8** — removes the dequant epilogue; measured +16-17 % over Q8_0 on the 4B.
2. **The delivery's MMB/GEMM work** — the re-based port inherits the current `mmq`/`mmb`/FA/staging
   improvements the cllm branch predates (7 weeks).
3. **AITER-level fp8 GEMM efficiency** — the cllm kernel is at 77-98 TFLOP/s where AITER's Triton
   reaches 121-137; lifting AITER's gfx1201 tiles/GROUP_M/kpack is the remaining headroom.

**Design constraint (`ROCmFPX-ASSESSMENT.md`):** the win must come from a **native fp8 WMMA GEMM**, not
a dequant-to-bf16 + hipBLAS pipeline — upstream already measured that on RDNA4 MMQ beats
"dequantization + hipBLAS" (`ggml-cuda/mmq.cu:608`, PR #18537).

## 5. Non-goals

* Not an upstream contribution (FP8 is already a direction upstream is exploring; ours is RDNA4-tuned).
* Not a new quant format for the delivery's GGUF set — this is a *path*, gated on the measurements in
  `PLAN.md`.
