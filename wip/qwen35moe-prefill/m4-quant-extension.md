# Milestone 4: quant extension (Q3_K, Q4_K, Q5_K, Q8_0)

## Result so far

The fused gate+up+GLU MMQ kernel (patch 0003) now covers Q3_K, Q4_K, Q5_K,
Q8_0 in addition to Q6_K. The critical finding: **the J<=64 cap from patch
0003 is WRONG for every type except Q6_K** - the register pressure is
per-type, not uniform.

## Per-type J cap (the key data)

Fused (has_gate=1, fb=0) VGPR/spill counts extracted from the .hip_fatbin
msgpack (ggml_type enum: Q3_K=9, Q4_K=12, Q5_K=13, Q6_K=14, Q8_0=6):

| type | vec_dot path | J=64 | J=96 | J=112 | J=128 | cap chosen |
|------|--------------|------|------|-------|-------|------------|
| Q6_K | q6_K_q8_1_mma | 205/0 | 215/0 | 240/0 | 255/0 (regression measured) | 64 |
| Q4_K | q8_1_q8_1_mma | **256/153** | 241/0 | 227/0 | 251/0 | 96 |
| Q5_K | q8_1_q8_1_mma | **256/115** | 245/0 | 253/0 | 241/0 | 96 |
| Q8_0 | q8_0_q8_1_mma | 163/0 | 222/0 | 251/0 | 237/0 | 128 |
| Q3_K | q3_K_q8_1_mma | 190/0 | 204/0 | 231/0 | 254/0 | 128 |

Key observations:
- Q4_K/Q5_K share the q8_1_q8_1_mma path: J=32..80 SPILL BADLY (256 VGPR +
  100+ spills) while J=96..128 are clean. The cap 96 wins because J=96
  balances register pressure vs tile efficiency.
- Q8_0 and Q3_K are clean everywhere; J=128 (natural pick) works.
- Q6_K is the only one where the J=128 config measurably regresses even
  without spills (occupancy drop at 255 VGPR) - hence cap 64.

The J-max is now type-conditional in `mul_mat_q_switch_J`:
```cpp
constexpr int J_max_gate =
    type == GGML_TYPE_Q6_K ? 64 :
    type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 96 : 128;
```

## Q4_K_M measurements (cap 96, fused vs GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1)

| test | fused | unfused | delta |
|------|-------|---------|-------|
| pp512 ub512 | 3791 | 3548 | +6.9% |
| pp2048 ub2048 | 5719 | 5405 | +5.8% |
| pp16384 ub2048 | 4949 | 4666 | +6.1% |
| pp32768 ub2048 | 4186 | 4016 | +4.2% |
| tg128 ub2048 | 97.3 | 97.3 | 0% |

Bit-exact: 536M logits, 0 diff. test-backend-ops MUL_MAT_ID OK.

## Status of the other quants

- Q5_K_M: model was still downloading during Q4 validation; register
  profile known (cap 96), benchmark pending.
- Q3_K_M: wired + compiles; register profile clean (cap 128); model
  download in progress; benchmark pending.
- Q8_0: wired + compiles; register profile clean (cap 128); requires
  2 GPUs + --split-mode tensor (see milestone4-handoff.md Q8_0 section).

## Files changed (delta over patch 0003)

- mmq.cu: `ggml_cuda_mul_mat_q_switch_type_gate` now has Q3_K/Q4_K/Q5_K/
  Q8_0/Q6_K cases.
- mmq.cuh: J_max_gate type-conditional (above).
- generate_cu_files.py: MMQ_GATE_TYPES extended.
- mmq-instance-q3_k.cu / q4_k / q5_k / q8_0: DECL_MMQ_CASE_GATE added.
