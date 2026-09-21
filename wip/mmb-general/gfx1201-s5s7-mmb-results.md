# gfx1201 port — S5/S6/S7: `mmb` (G1) on RDNA4 — measurement record (2026-09-21)

Session: S5 (shim) + S6 (gate/correctness) + S7 (re-tune / decision) of `gfx1201-porting.md` §6.5.
This is the raw evidence; the plan is the source of truth.  **Headline: the port works and is
gfx11-bit-identical, but `mmb` is NOT a blanket win on RDNA4 — so the single `GGML_CUDA_MMB` switch
was split into an arch-scoped weight-type policy plus a path policy, and RDNA4 now defaults to the
safe subset.**

## Setup

* Hardware: 3× Radeon AI PRO R9700 (gfx1201), ROCm `/opt/rocm-7.14.1-gfx102X`; 1 GPU for the
  21–29 GiB models, 2 GPU `-sm tensor` where stated, 3 GPU `-sm tensor` for Flash-Next.
* Build: `~/llama.cpp` branch `mmb-port-qsa3` (+ the S5–S7 commits), `-b/-ub 2048` where relevant.
* **Patch layout:** the work landed as **patch 6** of the delivered set (tree
  `580db5174574f10cc92fb1cefa72281a65c77b12`, `git am` 6/6) rather than folded into patch 1: patches
  1, 3 and 4 all touch `mmb.cu`, so no single theme owns it.  The set went 5 → 6 patches.
* `llama-bench … -n 0 -r 5`.  **The first prefill test of an invocation is cold-start-limited** —
  several early `r=3` readings here were 5–9 % low (see the qwen4exp r=3 vs r=5 note in §4), so only
  `r=5` and interleaved back-to-back rounds are quoted for decisions.

## 1. S5 — the shim, and gfx11 is bit-identical

`mmb.cu` now has the same arch-selected shim as `fattn-qsa3.cu` (7 sites: the three A/B fragment
loads, the `mmb_store_tile` epilogue, the `hc_gate_mix` epilogue, the `mmb_f32split` fragments and
epilogue, and the `mmb_wmma_bf16/f16` wrappers).

**Verification** (not just "the `#if` looks the same"): compile `mmb.cu` for **gfx1151**
(`--cuda-device-only -S`, offload arch swapped, the rest of the command taken from
`build-rocm/compile_commands.json`) before and after the port and diff the device assembly:

* **byte-identical except the `__hip_cuid_*` module-id symbol** (a hash of the source text; 18 diff
  lines, all of them that symbol);
* every `codeLenInByte` line identical, and the opcode histogram md5 identical
  (`6cfb7964202ab648c90c17bf5cc61ec4`).

**The trap worth recording:** the first attempt routed the gfx11 load through a
`__device__ __forceinline__` helper.  That left a **dead `lane >> 4` argument** at every call site
and perturbed the inliner/scheduler — the asm was still semantics-preserving, but *not*
bit-identical (30320 → 30304 B, register renumbering and `s_delay_alu` changes throughout).  Making
the gfx11 arm a **macro** so the preprocessed source at each call site is the original expression
restores bit-identity.  Same lesson applies to the RDNA4 shim: it is a function only because
`_gfx12` genuinely needs the `hi` operand.

## 2. S6 — gate and first correctness run

`mmb_enabled()` gained RDNA4 (so `GGML_CUDA_MMB=1` reaches it on gfx1200/gfx1201).

PPL parity on the fast model (`Qwen3.6-35B-A3B-UD-Q4_K_M`, `prose-rdna-boosts.txt`, `-c 2048`):

| MMB | PPL |
|---|---|
| off | 14.3981 ± 0.96706 |
| on  | 14.4087 ± 0.96890 |

+0.07 % — parity, and (importantly) MMB was demonstrably running (`LLAMA_MMB_CVT_LOG=1` shows the
activation-conversion stream).  A fragment-layout error in a bf16 WMMA GEMM is an exact-permutation
error: it scrambles the contraction (cf. the GDN notes' ~0.53×) and would move PPL by orders of
magnitude, not 0.07 %.

## 3. S7 — the measurement that changed the design

`r=5`, pp8192 / pp32768 unless noted.  "Δ" is MMB-on minus MMB-off.

| model | family | weight composition | Δ with **all** types + dense on | Δ with the **split** policy |
|---|---|---|---|---|
| Flash-Next IQ4_XS | qwen4exp (HC/QSA) | 100 % IQ4_XS | +2.7 / +2.3 … +4.8 / +2.7 % | **+3.1 / +3.4 %** (interleaved) |
| Flash-Next IQ3_XXS | qwen4exp (HC/QSA) | 100 % IQ3_XXS | **+11.8 / +7.3 %** | (kept) |
| 35B-A3B UD-Q3_K_M | qwen35moe | **88.5 % IQ** (IQ3_XXS 59, IQ4_XS 29.5) | +3.2 / +3.7 % | **+6.7 / +5.6 %** |
| 27B UD-IQ3_S | qwen35 **dense** | **67 % MMB-eligible IQ** (IQ3_S 32, IQ4_XS 18, IQ3_XXS 18) | −3.4 / −3.0 % | **0** (neutral) |
| 27B UD-Q4_K_XL | qwen35 **dense** | 23 % IQ4_XS, **75 % K-quant** | **−12.7 / −11.0 %** | **0** (neutral) |
| 27B Q8_0 | qwen35 dense | 100 % Q8_0 | −6.1 / −5.8 % | **0** (neutral) |
| 35B-A3B Q8_0 | qwen35moe | 100 % Q8_0 | −4.6 / −4.6 % | (type-excluded) |
| 35B-A3B Q4_K_M | qwen35moe | Q4_K/Q6_K | −6.1 / −4.5 % | **0** (neutral, ±0.5 %) |

What the matrix says, in order of how much it changed the design:

1. **"IQ weights win" is not the rule.**  The *dense* 27B UD-IQ3_S is 82 % IQ and still lost −3 %
   with the dense path on; the dense `UD-Q4_K_XL` lost −11 % with only 23 % IQ.  So the weight type
   is *a* factor, not *the* factor — the **calling path** is at least as strong.
2. **The generic quantized dense tile GEMM loses on RDNA4 for every type measured** (Q8_0 −6 %,
   K-quant −11 %, IQ-heavy −3 %).  Turning it off makes every dense model exactly neutral, i.e. MMB
   then takes nothing at all there.
3. **The routed MoE path and the qwen4exp HC paths win**, and they win *more* once the dense GEMM and
   the K-quant types stop dragging them down: the MoE IQ-heavy model went +3.2 % → **+6.7 %**.
4. **The F32 split (MoE router) also loses on RDNA4** — it was the residual −0.3…−1.3 % left on a
   K-quant model once the weights were already type-excluded (a type-excluded model has no quantized
   MMB GEMM left, so the only thing still running was the F32 router).
5. **The dense path is model-specific, not universally bad**: on qwen4exp the dense GEMM *helps*
   (+5.2 % vs +4.4 % with it off at pp32768).  So the safe default (dense off) trades ~1 % of the
   qwen4exp win for ~2–3 % on the dense qwen35 models; `GGML_CUDA_MMB_DENSE=1` restores it there.

## 4. The scope split (the actual S7 deliverable)

`GGML_CUDA_MMB` bundled four independent things: arch, weight type, kernel path, and model family.
The type list was also **duplicated five times** (`dense_will_take`, `routed_will_take`,
`supported_mm`, `supported_mmid`, `supported_glu`), arch-independent, and drifted (IQ3_XXS was
treated differently in different copies).  It is now one policy with two axes:

* **`mmb_wtype_mask()` / `mmb_wtype_ok(t)`** — one arch-scoped weight-type set replacing all five
  copies.  RDNA4 defaults to the **IQ family** (`IQ4_NL`, `IQ3_S`, `IQ4_XS`, `IQ3_XXS`); RDNA3_5 /
  RDNA3_0 keep the full set, so **gfx1151 behaviour is unchanged**.  Override for A/B or a future
  re-tune with **`GGML_CUDA_MMB_TYPES=<csv>`** (`iq3_s,iq4_xs`, …).
* **`mmb_dense_flag()`** — separates the generic *quantized dense tile GEMM* **and** the F32 split
  router (the shapes that lose) from the routed MoE path and the qwen4exp HC paths (tall-M,
  tiny-M inject, gate-mix — the shapes that win).  Default: **off on RDNA4, on elsewhere**;
  `GGML_CUDA_MMB_DENSE=0|1` overrides.
* `mmb_tall_shape()` keeps the qwen4exp HC tall-M tile (`IQ4_NL`, `M<=384`, `K>=4096`, `T>=2048`)
  out of the dense stand-down — without it the dense flag would silently kill the HC win.

The graph's MMQ-fusion stand-down goes through these same predicates (`ggml-cuda.cu`), so a type or
path this policy excludes costs **nothing** — it simply keeps the delivery's MMQ path.

## 5. Correctness of the shipped subset

| gate | result |
|---|---|
| 7-chunk PPL, MoE IQ-active (`UD-Q3_K_M`, routed IQ3_XXS/IQ4_XS on) | 12.7378 (off) vs **12.7221** (on) = −0.12 %, parity |
| 2-chunk PPL, K-quant (`Q4_K_M`, type-excluded) | 14.3981 (off) vs 14.3928 (on) = parity |
| `mmb.cu` gfx1151 device asm | byte-identical but for `__hip_cuid_*`; opcode histogram md5 equal |
| 2-chunk PPL, fast model (all types, pre-split build) | 14.3981 vs 14.4087 = +0.07 % |

## 6. Conclusions

1. **G1 `mmb` is PORTED to RDNA4** (the gfx12 bf16 fragment shim + the gate) and gfx11 remains
   bit-identical, so gfx1151 is untouched.
2. **It is not a blanket win**, which is exactly why the switch had to be split: with all types and
   the dense path on it is a −4…−13 % regression on three models; with the split policy it is
   **0 % (neutral) where it does not win** and **+3…+7 % where it does**.
3. **RDNA4 default = the safe subset**: IQ-family weights, routed/HC paths only.  A user who sets
   `GGML_CUDA_MMB=1` on gfx1201 can no longer lose on it.
4. **Opt-ins for the wins it forgoes**: `GGML_CUDA_MMB_DENSE=1` (qwen4exp, +1 %) and
   `GGML_CUDA_MMB_TYPES=…` (per-type A/B).
5. **A future re-tune (S7+)** should treat the axes separately: per-weight-type tile geometry, and a
   real gfx1201 geometry for the dense tile (the current one is gfx1151-tuned and is where every
   RDNA4 loss lives).  `GGML_CUDA_MMB_TILE` covers only the big/small split.
6. **gfx1100 keeps the gfx11 path** and the full type set (`GGML_CUDA_MMB_RDNA3=1`), unaffected.
