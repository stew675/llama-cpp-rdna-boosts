# ROCmFPX (`ciru-ai/ROCmFPX`) — assessment for the FP8 campaign

Investigated 2026-09-26 from `~/ROCmFPX` (clone of `github.com/ciru-ai/ROCmFPX`, branch `main`,
tip `112629f1e`, `git describe` = `main-b10499-112629f`, tip date 2026-08-17).

**Verdict: not an FP8 E4M3 source, but three useful things — one of which is already ours and one of
which is a warning for this campaign's plan.**

---

## 1. What it is

Ciru's AMD inference lab: a **maintained llama.cpp fork** forward-ported onto newer upstream lines
(`b10499`), carrying a family of **low-bit weight formats** plus a prefill-specialised runtime.
It is *not* an FP8 project.

| family | what |
|---|---|
| ROCmFP2 | 2.50-bpw S40 fixed codebook (`Q2_0_ROCMFPX`, type 108) |
| ROCmFP3 / ROCmFP6 | 3-bit / 6-bit, UE4M3-scale layouts (`Q3_0_ROCMFPX` 104, `Q6_0_ROCMFPX` 102) |
| ROCmFP4 | 4-bit + UE4M3 scales (`Q4_0_ROCMFP4` 100) |
| ROCmFP7 / **DualView** | packed signed **Q7** + FP16 group scales (`Q7_0_ROCMFPX` 107) |
| ROCmFP8 | `Q8_0_ROCMFPX` (103) — an **8-bit integer** layout with a UE4M3 *scale byte* |

**No IEEE E4M3 weight type.**  Every `e4m3`/`UE4M3` hit is a **scale byte** (67 files), not an e4m3
weight tensor.  So this tree cannot supply FP8 GEMM kernels, and the name collision is a trap worth
recording: "ROCmFP8" is 8-bit int, not fp8.

## 2. The RDNA4 finding — already in our base (and a warning)

`ggml/src/ggml-cuda/mmq.cu` carries:

```c
// For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
// https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
return true;
```

This is **upstream** (PR #18537) and our delivery base already has it at
`~/llama.cpp/ggml/src/ggml-cuda/mmq.cu:608` — so it is **not** a lift.  But it matters here:

> **Do not plan the FP8 path as "dequantize to bf16 and call hipBLAS".**  On RDNA4 that route was
> already measured slower than a native quant MMQ.  The FP8 win has to come from a **native
> fp8 WMMA GEMM**, which is what `mul_mat_fp8_wmma` already is.

## 3. DualView — the asymmetric-representation idea (gfx1151-gated)

`docs/DUALVIEW.md`: the stored weight is `Q7_0_ROCMFPX`; the **decode** path reads Q7 (small,
bandwidth-friendly), and the **prefill** path sign-extends the same bits into a **standard Q8 int8
WMMA "compute shadow"** with the *same* FP16 scales (`ggml/src/ggml-cuda/q7-q8-view.cuh`,
`ggml/rocmfpx/q7-q8-view.h`).  Rationale, in their words: *"Prompt processing is wide matrix work and
can exploit AMD INT8 dot/WMMA throughput; token generation repeatedly streams weights and benefits
from the smaller Q7 representation."*

* Measured (Ornith 35B, gfx1151): **+15.34 % prefill** vs AMD-safe `Q8_K_XL`
  (`docs/DUALVIEW-ORNITH-35B-RESEARCH.md`).
* **Hard-gated to `gfx1151`** — "the Q8 compute shadow is currently hard-gated to gfx1151"; other HIP
  targets are compile/fallback only.  Even the shadow Q8 kernel is RDNA3.5-specific
  (`mmq.cu` gates read `gfx1151`).

**Relevance:** the *principle* (a per-phase representation: compact for decode, wide int8/fp8 WMMA for
prefill) is exactly the shape of a good FP8 plan, and it is evidence that a **prefill-only wide-format
execute path pays**.  The *code* is not portable to gfx1201 as written.

## 4. ActiveFPX PromptForge — a Qwen3.8-27B prefill runtime (gfx1151-gated)

Branch `release/qwen3.8-activefpx-promptforge-v1`, +1548 lines over `main`, dominated by
**`ggml/src/ggml-cuda/promptforge.cu` (1249 lines)** and `docs/activefpx-promptforge-qwen38.md`.

It adds, for the exact model this repo tunes (Qwen3.8-27B):

* **PromptForge FFN routing** for a 2048-row prompt block, a 2044-row checkpoint block, a 1476-row tail;
* a **fused gate/up path**, **fused SwiGLU-to-down packing**, **accelerated down projection**;
* a **merged QKV/Z projection** for the recurrent Gated DeltaNet layers;
* request-level route telemetry + fail-closed shape checks;
* prepacked **companion "compute views"** (`.pfs` sidecars) loaded at startup.

Pinned: CK `fdf4bb7fc`, TheRock ROCm 7.15 dev, `AMDGPU_TARGETS=gfx1151`, `GGML_HIP_MMQ_MFMA=ON`,
`PROMPTFORGE_MODE=m2048_fused_tail1476`.  **Specialised to one published GGUF + sidecars and gated to
gfx1151** — startup fails closed elsewhere.  So it is not droppable into our tree either, but it is a
**direct comparison target** for our block-13 fused gate+up+GLU / shared-expert work on the same model:
what they fuse for prefill (gate/up, SwiGLU-to-down packing, QKV/Z merge, 2048-row routing) versus what
we already fuse.

## 5. What to take, concretely

| item | take? |
|---|---|
| FP8 E4M3 kernels | **No** — none exist here |
| RDNA4 "MMQ > dequant+hipBLAS" | Already in our base; **use it as a design constraint** for this campaign |
| DualView asymmetric representation | **Principle yes**, code no (gfx1151-gated) |
| PromptForge Qwen3.8-27B prefill fusions | **Compare ideas** against our block-13 fusions; not droppable |
| gfx1200/gfx1201 build + validation infra + low-bit MMQ/MMVQ tuning | Worth reading for tuning conventions; tuning targets gfx1151 |
| 8-bit integer `Q8_0_ROCMFPX` | Not a substitute for fp8 (different accuracy/throughput profile) |

## 6. Net effect on the FP8 plan

* Unchanged: the FP8 win must come from a **native RDNA4 fp8 WMMA GEMM** (`mul_mat_fp8_wmma`), not from
  a dequant-to-bf16 pipeline.  `PLAN.md` already reflects that; this assessment hardens it.
* Added: a **cross-check task** — compare our Qwen3.8-27B prefill fusions against ActiveFPX
  PromptForge's (gate/up, SwiGLU-to-down, QKV/Z) before writing new fusion work.
* Added: the **dual-representation** idea as a fallback shape if the 27B fp8 conversion disappoints —
  store a compact format, execute prefill in a wider form with shared scales (DualView's pattern).
* ROCmFPX's own numbers are gfx1151-first; do not assume they transfer to gfx1201.
