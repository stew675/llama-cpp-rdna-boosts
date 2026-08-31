# Milestone 4: quant extension - FINAL RESULTS

## Summary

The fused gate+up+GLU MMQ kernel now covers **Q4_K, Q5_K, Q8_0** (+ Q6_K
from patch 0003). Measured on Qwen3.6-35B-A3B:

| experts | pp512 ub512 fused vs unfused | pp2048 ub2048 | depth (16384) |
|---------|-------------------------------|---------------|----------------|
| **Q4_K** | **3803 vs 3539 (+7.5%)** | **5705 vs 5400 (+5.6%)** | 4949 vs 4666 (+6.1%) |
| **Q5_K** | **3536 vs 3284 (+7.7%)** | **5467 vs 5206 (+5.0%)** | 4817 vs 4566 (+5.5%) |
| **Q6_K** | +14-16% (prior) | +6.3% (prior) | +5.8% (prior) |
| Q8_0 | untested single-GPU (needs 2 GPUs, --split-mode tensor) | | |
| IQ4_XS | **-3.9% (REJECTED, excluded)** | neutral | neutral |

All enabled types: bit-exact (536M logits, 0 diff), decode unchanged
(mmvvq owns decode), test-backend-ops full suite OK.

## KEY finding 1: the J<=64 cap is per-type, not global

VGPR/spill data extracted from .hip_fatbin msgpack (fused has_gate kernels,
fb=0):

| type | vec_dot path | J=64 | J=96 | J=128 | cap |
|------|--------------|------|------|-------|-----|
| Q6_K | q6_K_q8_1_mma | 205/0 | 215/0 | 255/0 but regression | 64 |
| Q4_K | q8_1_q8_1_mma | 256/153 | 241/0 | 251/0 | **96** |
| Q5_K | q8_1_q8_1_mma | 256/115 | 245/0 | 241/0 | **96** |
| Q8_0 | q8_0_q8_1_mma | 163/0 | 222/0 | 237/0 | 128 |
| IQ4_XS | q8_0_16_q8_1_mma | 224/0 | 224/0 | 239/0 | (excluded) |

- Q4_K/Q5_K share q8_1_q8_1_mma: J=32..80 SPILL badly (256 VGPR + 100+
  spills), J=96..128 clean. J=96 measured best (not 128!).
- Q6_K is the only type where J=128 regresses without spills (occupancy
  drop at 255 VGPR).
- Q8_0/IQ4_XS clean everywhere.

J_max_gate in mul_mat_q_switch_J:
```cpp
constexpr int J_max_gate =
    type == GGML_TYPE_Q6_K ? 64 :
    type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 96 : 128;
```

## KEY finding 2: the "Q3_K_M" download has IQ4_XS experts

The Unsloth UD-Q3_K_M.gguf does NOT have Q3_K experts - the ffn_gate_exps/
ffn_up_exps tensors are **IQ4_XS** (type 23) and ffn_down_exps are also
IQ4_XS. This was confirmed at runtime: the fused dispatch saw gate type 23
(IQ4_XS), not Q3_K. The user confirmed HuggingFace's block breakdown shows
IQ quants in the UD quants.

IQ4_XS was wired, tested, bit-exact - but **measured net-negative**
(pp512 3390 vs 3529 = -3.9%; pp2048 neutral). The compact format makes
gate/up GEMMs small, so the doubled accumulator + J constraints outweigh
the shared sort/quantize/launch savings. **Excluded from the fused path**;
the Q3_K_M model falls back to the 3-op sequence (verified 0 fused ops).

IQ3_XXS/Q3_K were wired during debugging but no model uses them as
gate/up experts, so they were also removed from the switch + generator +
instances to keep the compiled surface minimal.

## Op-timing analysis (why Q4_K wins big, IQ4_XS doesn't)

Q4_K fused: FUSED gate+up+glu block = 82.7ms vs unfused 102.9ms
(-20.2ms, -19.6% on the MoE block; ~5% of total eval).
IQ4_XS fused: 107.9ms vs unfused 113.1ms (-5.2ms, -4.6%) - too small
to overcome the pp512 J-config penalty.

NOTE: GGML_CUDA_OP_TIMING=1 disables CUDA graphs; bench numbers (graphs
on) are the authoritative ones. The Q4 fused total via op-timing (361ms)
vs bench (358ms) agree within noise.

## Final config

- mmq.cu gate switch: Q4_K, Q5_K, Q8_0, Q6_K (has_gate=true cases)
- J_max_gate: Q6_K=64, Q4_K/Q5_K=96, Q8_0=128
- All other types fall back to the 3-op path via try_fuse
- GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1 still the opt-out

## Files (delta over patch 0003)

- mmq.cu: switch_type_gate cases for Q4_K/Q5_K/Q8_0 (Q6_K kept)
- mmq.cuh: type-conditional J_max_gate
- generate_cu_files.py: MMQ_GATE_TYPES = Q4_K, Q5_K, Q8_0, Q6_K
- mmq-instance-q4_k/q5_k/q8_0.cu: DECL_MMQ_CASE_GATE added
- (iq3_xxs/iq4_xs/q3_k instances: gate DECL added then removed - net no-op)

## Remaining

- Q8_0 2-GPU tensor-split validation (needs --split-mode tensor per
  handoff; model is 35 GiB > 32 GiB single GPU)
- M5: qwen4exp cross-validation
