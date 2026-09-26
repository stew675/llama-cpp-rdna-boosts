# gfx1201 port — S12: the routed MoE path is a loss on RDNA4 (2026-09-21)

Session S12 of `gfx1201-porting.md` §13.  **Headline: the routed `MUL_MAT_ID` path loses on *every*
model measured — it was masking most of the qwen4exp win — so RDNA4 now disables it by default
(`routed = 0`).  Landed default vs the delivery: qwen4exp +4.9/+5.2 % prefill (was +2.3/+1.8), the
IQ-heavy MoE neutral (was −1.4 %), the IQ3_S dense model +0.5 % unchanged, and the same-seed text is
byte-identical to the delivery everywhere.**

## 1. Why this became the top S12 item

S7 recorded a **+6.7 %** routed-MoE win for `35B-A3B UD-Q3_K_M`; S10 re-measured the **unmodified S7
binary** and got **−1.4 %** (`gfx1201-s10-dense-geometry.md` §6).  The kernel breakdown blamed the
routed path: mmb's routed kernels only **match** the delivery's block-13 `mul_mat_q_routed_compact` +
`mul_mat_q`, and standing that fused kernel down costs an extra `mm_ids_helper` launch.

So S12's job was not a threshold sweep but a **policy re-decision** — which needed a mechanism the WIP
did not have: `mmb_wtype_ok()` gated the routed path and the dense path *and* the HC tall-M path
together, so "routed on/off" could not be expressed.

## 2. The mechanism

The counterpart of `mmb_dense_flag()`:

```cpp
struct mmb_arch_cfg { ... int routed = 1; ... };          // per arch, S11's table
bool mmb_routed_flag() { ... getenv("GGML_CUDA_MMB_ROUTED") ... : mmb_cfg().routed; }
```

gating the three routed predicates — `ggml_cuda_mmb_supported_mmid`, `ggml_cuda_mmb_supported_glu`
and `ggml_cuda_mmb_routed_will_take`.  Because the graph's MMQ-fusion stand-down goes through
`routed_will_take`, a disabled routed path keeps the delivery's fused kernel and **costs nothing**.

`routed = 0` on RDNA4; `1` (the gfx1151 value) everywhere else, so gfx1151 is unchanged.

## 3. The measurement

End-to-end, interleaved back-to-back rounds, `llama-bench -n 0 -b 2048 -ub 2048 -r 3`:

| model | pp | MMB off | routed ON | routed OFF |
|---|---|---|---|---|
| 35B-A3B UD-Q3_K_M (qwen35moe, 1 GPU) | 8192 | 5930.86 | 5845.75 (**−1.44 %**) | 5927.93 (**−0.05 %**) |
| | 32768 | 4865.96 | 4798.21 (−1.39 %) | 4861.39 (−0.09 %) |
| Flash-Next IQ4_XS (qwen4exp, 3-GPU tensor) | 8192 | 2751.81 | 2814.32 (+2.3 %) | **2934.23 (+6.6 %)** |
| | 32768 | 2726.56 | 2775.99 (+1.8 %) | **2872.67 (+5.4 %)** |

Both rounds agreed (e.g. routed-OFF on Flash-Next 2872.31 / 2872.67 at pp32768, 0.01 %), and the
result reproduces on both the `MMB_TYPES=iq4_nl` workaround (which also disables MMB for those models)
and the new `MMB_ROUTED=0`: on the MoE, `−0.08 / −0.04 %` vs `−1.44 / −1.39 %`.

**A note on instruments.**  The obvious tool here — `rocprofv3` per-kernel time — is *not* usable for
the 3-GPU tensor case: under trace serialization the `ncclDevKernel_Generic_4` time moved by
**+9 % (3.874 -> 4.225 s)** between the two configs, which swamps the effect and swamped the total
(the profile said ON was +0.75 % *worse* while the bench said +3.2 % *better*).  The plan's §12.6 rule
("judge routed/GLU on kernel time, not end-to-end t/s") is right for a **single-GPU** routed shape;
for a tensor-split model the end-to-end interleaved A/B is the trustworthy instrument.  The kernel
attribution in S10 §6 is the 1-GPU MoE case and stands.

## 4. Why the routed path loses (the cause, from S10's 1-GPU kernel breakdown)

`35B-A3B UD-Q3_K_M`, pp8192, 1 GPU, OFF -> ON:

| | OFF | ON |
|---|---|---|
| `mul_mat_q<IQ3_XXS, 64>` | 0.650 s | 0.000 |
| `mul_mat_q_routed_compact<IQ4_XS, 64>` | 0.230 s | 0.000 |
| `mmb_routed_kernel<...,10>` (IQ3_XXS, 2 tiles) | -- | 0.649 s |
| `mmb_routed_kernel<...,8>` (IQ4_XS, 2 tiles) | -- | 0.226 s |
| `mm_ids_helper<8>` | 0.127 s | **0.192 s** |

The delivery's block-13 `mul_mat_q_routed_compact` plus its `mul_mat_q` fallback cost 0.880 s; mmb's
routed kernels cost 0.875 s — a **tie** — and the stand-down adds 0.065 s of `mm_ids_helper`.  So the
routed path can never win here: it replaces a *fused* expert kernel with a comparable
dequant-to-WMMA kernel but loses the fusion, and pays the routing-helper pass that the fused kernel
did not need.  (That the delivery's fused kernel is this good is itself the 2026-09-13/14 block-13
work; S7 was written before it, which is the likely origin of the stale +6.7 %.)

**Open follow-up (not needed for the decision):** if mmb's routed kernel consumed the same compact
routing descriptor the fused kernel uses, the `mm_ids_helper` pass might disappear — but that is a
rewrite, and the routed kernel is only at parity, so it would have to beat parity by more than the
helper costs.  It is not worth doing while the dense/HC paths are the real wins.

## 5. Purity

Same-seed greedy text, `--seed 42 --temp 0 -n 24`:

| model | delivery | landed (`routed=0`) | `MMB_ROUTED=1` |
|---|---|---|---|
| 35B-A3B UD-Q3_K_M | `461ca8cd0e88` | `461ca8cd0e88` | `461ca8cd0e88` |
| Flash-Next IQ4_XS | `d73f9238f6d6` | `d73f9238f6d6` | `d73f9238f6d6` |

The routed path changes neither the text nor the answer on these models — only the speed.  (It is not
a *bit-identity* claim: the routed arithmetic differs from MMQ's; the point is that the landed default
reproduces the delivery's output exactly, which is the gate the WIP has been holding.)

## 6. The landed RDNA4 default, in full

| workload | default vs delivery |
|---|---|
| qwen4exp (Flash-Next IQ4_XS, HC/QSA, 3-GPU tensor) | **+4.9 % pp8192 / +5.2 % pp32768** |
| IQ3_S-heavy dense (27B UD-IQ3_S) | +0.5 % (S10's dense tile) |
| IQ-heavy MoE (35B UD-Q3_K_M) | neutral |
| Q8_0 / Q4_K_M / non-IQ dense | neutral (type-excluded) |
| text | byte-identical to the delivery on every model tested |

That is strictly better than both the S7 default (qwen4exp +2.3, MoE −1.4) and the S10 default
(qwen4exp +2.3, MoE −1.4, dense +0.5): `routed = 0` is what turns MMB-on-RDNA4 from "a win on
qwen4exp and a small loss on MoE" into "a win everywhere it does anything, and nothing where it does
not".

`MMB_CFG` line now:
`MMB_CFG cc=0x1001201 dense_geom=1 ... iq3xxs_glu=0 routed=0`.

## 7. Patch layout

Landed as **patch 9** (tree `4a78af6349df3fd92e747d223ea3b4b32817f592`, `git am` **9/9** verified on a
fresh r12 worktree).  Patches 6-9 are a chain on `mmb.cu`; patch 9 also carries the `TODO(S12)` list
in `mmb_arch_defaults` for the fields that are now *inert* on RDNA4.

## 8. Handing on

* **With `routed = 0` on RDNA4, the routed/GLU threshold sweep is no longer worth running** — it
  tunes a path that is disabled.  `glu_thresh`/`routed_thresh`/`iq3xxs_glu`/`glu` stay as measured on
  gfx1151 and are inert on RDNA4.  If the routed path is ever revisited, the place to start is §4's
  open follow-up.
* The remaining S12-style per-arch fields (`tall_mode`, `tiny_m`/`tiny_tt`, `f32split_*`,
  `cache_max`) *are* worth measuring — they belong to the HC/F32 paths that now carry the qwen4exp
  win.  `cache_max` is the cheapest (the `mmb_cvt_f32_bf16` tax).
* **S13** (HC16 producers) and **S14** (the B1-B9 matrix, incl. MTP) are next; S14 is the bigger gap.
