# gfx1100 port — S8/S9 record (2026-09-21)

Sessions S8 (G3b/c F32/tiny-M + G4 HC16) and S9 (the delivery re-examination + the gate matrix) of
`gfx1100-porting.md`.  **S8 is a documented no-op on gfx1100 by construction; S9 is partially done
— the FA head-cap re-examination and the verify-width gate are closed, the mmvq/VDR re-sweeps and
the block-13-vs-MMB-routed kernel-time A/B are the remaining work.**

## S8 — G3b/c + HC16

| item | gfx1100 status |
|---|---|
| **F32 split (MoE router)** | **off by policy** (patch `0008`, S6/S7).  Nothing left to tune — its knob is deliberately off on RDNA3_0. |
| **tiny-M F32** (`mmb_tiny_m_f32_kernel`, `M <= 8`) | the qwen4exp HC `*_inject` shape; **not reachable** on the available models (no qwen4exp fits 24 GB) → trust-RDNA3_5. |
| **HC16 bf16 producers** (`GGML_CUDA_MMB_HC16`) | **hard-gated to `GGML_CUDA_CC_IS_RDNA3_5`** in `ggml-cuda.cu` (both the `DSV4_HC_PRE` gate marking at ~line 6261 and the "wants a BF16 copy" activation marking at ~line 6306: `hc16 >= 1 && ggml_cuda_mmb_active() && GGML_CUDA_CC_IS_RDNA3_5(...)`).  **Setting it on gfx1100 is a no-op** — there is nothing to A/B here. |

So S8 has no measurable gfx1100 work: the F32 path is policy-off, and HC16/tiny-M are gfx1151-only or
qwen4exp-only.  If a future session wants HC16 on RDNA3_0 it is a code change (widen the `cc` gate),
not a tuning change; revisit only if the qwen4exp/MoE HC producer path is shown to matter on gfx1100.

## S9 — the delivery re-examination

### 9.1 §2.5 FA WMMA head cap (the gfx1100-specific one) — **the prior decision HOLDS, with a large margin**

The delivery (block 04, r5) caps WMMA at head 256 on `RDNA3_0`, so the gemma-4 head-512
full-attention layers take the **tile** kernel.  Re-tested on the current tree (with block 15's V3
tile-kernel derived mask and V4/V5 native KV), interleaved `r=5`,
`GGML_CUDA_FA_WMMA_MAX_HEAD=576` versus the default cap:

| model | point | cap 256 (tile, default) | cap 576 (WMMA) | Δ |
|---|---|---:|---:|---:|
| gemma-12B Q8_0 | pp8192 | 2157.09 / 2143.08 | 2033.51 / 2034.51 | **+6.1 % / +5.3 %** |
| gemma-12B Q8_0 | pp16384 | 1905.97 / 1903.06 | 1745.97 / 1746.01 | **+9.2 % / +9.0 %** |
| gemma-26B-A4B | pp8192 | 3294.96 / 3279.28 | 3118.04 / 3111.49 | **+5.7 % / +5.4 %** |
| gemma-26B-A4B | pp32768 | 2412.68 / 2409.13 | 2104.33 / 2102.57 | **+14.7 % / +14.6 %** |

**Decision: keep the cap at 256.**  WMMA at head 512 is 5-15 % slower on gfx1100 on the current tree;
the decision is not stale and the margin is larger than the 2026-09-18 measurement, consistent with
the r9 V3-on-tile improvement.

### 9.2 Rule-5 verify-width gate — **green**

`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` with a `q8_0` KV on a dense K-quant model
(`gemma-4-12B-QAT UD-Q4_K_XL`, 6.3 GB), delivery vs WIP (both MMB off):

| B | delivery T_TG s | WIP T_TG s |
|---|---:|---:|
| 1 | 0.572 | 0.437 (cold/warm-up; not comparable) |
| 4 | 0.529 | 0.534 |
| 8 | 0.775 | 0.775 |

B=4/B=8 are equal within noise → **no verify-width regression**.  (The 27B UD-Q4_K_M OOMs this test
on 24 GB at `-npl 4/8` because of its hybrid-SSM recurrent-state cache; the 6.3 GB gemma is the
practical dense-K-quant probe here.)

### 9.3 Still open from §2 of the plan

* **§2.3 the `mmvq` RDNA3_0 `nwarps` table** — needs a dedicated per-type decode + verify-width
  sweep (the table is the 2026-08-28 one; block 10/13 and the band-uniformity work postdate it).
* **§2.4 the `VDR_Q8_0_Q8_1_MMVQ_MOE`=4 choice** — re-measure at the verify widths with MMB on/off.
* **§2.6 block-13 fused MoE vs MMB routed** — a kernel-time A/B (the S7 `DENSE=0` isolate shows the
  routed MMB path wins ~+4 %, but the two are alternatives and the fused-MMQ path is the delivery
  baseline; `rocprofv3` attribution is the proper instrument).
* **§2.7 native-KV auto policy** — the width probe (S1/S6) and `FLASH_ATTN_EXT` (S1) are green on the
  current tree, but the per-KV-type mstep/deep-prefill numbers are gfx1201/gfx1151; a gfx1100-specific
  per-type check is still owed.
* **The full B1-B9 matrix with MMB on** is largely covered piecemeal (PPL S5, width purity + greedy +
  MTP S6/S7, decode unchanged S6); a single consolidated re-run on the frozen tree belongs to S10.

## Carry-forward

1. The gfx1100 headline is already banked: `mmb` is a large win (patch `0008` policy), `qsa3` is
   ported (patch `0007`), the arch-neutral groups are neutral/green.
2. The remaining gfx1100 work is the **decode-side re-examination** (§2.3/§2.4/§2.6) and the final
   gate matrix — a focused kernel-tuning session, not a port.
3. Then S10: freeze, regenerate the overlay patch set, and merge `wip-mmb-general-gfx1100` back into
   `wip-mmb-general`.
