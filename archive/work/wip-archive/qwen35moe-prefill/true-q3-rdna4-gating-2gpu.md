# True Q3_K_M + RDNA4 gating + 2-GPU scaling

## True Q3_K_M (bartowski quant, no hidden IQ quants)

Confirmed at runtime: gate/up experts are GGML_TYPE_Q3_K (id 11). The
"UD" Q3_K_M from before had IQ4_XS experts and was excluded from fusion.

J cap sweep (fused vs unfused 3074/4854/4272):

| cap | pp512 | pp2048 | pp16384 |
|-----|-------|--------|---------|
| 128 | 3125 (+1.7%) | 4961 (+2.2%) | 4351 (+1.9%) |
| 96  | 3323 | 5095 | 4466 |
| 64  | 3574 (+16.3%) | **5231 (+7.8%)** | **4572 (+7.0%)** |
| 48  | 3699 (+20%) | 5195 | - |
| unfused | 3074 | 4854 | 4272 |

Q3_K register profile (fused, fb=0): J=48 172/0, J=64 190/0, J=96 204/0,
J=128 254/0 - all clean. J=64 wins like Q8_0 (q3_K_q8_1_mma and
q8_0_q8_1_mma both prefer J=64; q8_1_q8_1_mma prefers 96).

Final: Q3_K cap 64. Bit-exact (536M, 0 diff). Decode 102.5 t/s unchanged.
test-backend-ops OK.

## RDNA4 gating (gfx120x)

The fusion dispatch AND the J caps are now gated on GGML_CUDA_CC_IS_RDNA4(cc):

1. try_fuse MMQ arm (ggml-cuda.cu): requires GGML_CUDA_CC_IS_RDNA4(cc) in
   addition to the should_use_mmq check. On other arches (e.g. Strix Halo
   gfx1151 = RDNA3.5) the fused kernel does not fire - falls back to the
   3-op path. Strix Halo work can enable it after its own tuning.
2. J_max_gate in mul_mat_q_switch_J: only applies caps on RDNA4; other
   arches keep J=128 default (no cap).

Why: the per-type J caps were measured on gfx1201 (vec_dot paths and
register allocation differ per arch; e.g. Q4_K's q8_1_q8_1_mma spills at
J=32..80 on RDNA4 but the profile may be completely different on RDNA3.5).

Final J_max_gate:
```cpp
const int J_max_gate = GGML_CUDA_CC_IS_RDNA4(cc)
    ? (type == GGML_TYPE_Q6_K ? 64 :
       type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 96 :
       type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K ? 64 : 128)
    : 128;
```

## 2-GPU tensor-split scaling (--split-mode tensor, fused)

Relevant for Qwen3.8-Flash-Next (Q4_K_XL = mixed Q4_K/Q5_K/Q6_K blocks,
needs multi-GPU on this system).

Q6_K:
| test | 1 GPU | 2 GPU | scaling |
|------|-------|-------|---------|
| pp512 ub512 | 3425 | 4538 | +32% |
| pp2048 ub512 | 3336 | 4338 | +30% |
| pp2048 ub2048 | 5191 | 6570 | +27% |
| pp16384 ub2048 | 4557 | 5976 | +31% |

Q4_K:
| test | 1 GPU | 2 GPU | scaling |
|------|-------|-------|---------|
| pp512 ub512 | 3765 | 4566 | +21% |
| pp2048 ub2048 | 5655 | 6774 | +20% |
| pp16384 ub2048 | 4949 | 6181 | +25% |

Q8_0 (2-GPU only, 35.19 GiB): pp512 5260, pp2048 7184, pp16384 6493.

Fusion under tensor split still pays: Q6_K +12%, Q4_K +4.7%, Q8_0 +8%.
Scaling is better for the larger quant (Q6_K +27-32% vs Q4_K +20-25%) -
the bigger model gets more from 2x compute; tensor-split overhead is
relatively smaller. MoE exchange cost is inside the split-buffer
machinery (not explicit graph reduce nodes).

## Status

All M4 types validated on RDNA4 (single + 2-GPU tensor split):
Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 - all bit-exact, all win with tuned caps.
Remaining: M5 qwen4exp cross-validation; Strix Halo (RDNA3.5) validation
to open the gating.
