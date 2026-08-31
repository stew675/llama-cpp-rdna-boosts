# Q8_0 2-GPU tensor-split validation (--split-mode tensor)

## Result: fused kernel works under tensor split, +7-8%

The Q8_0 gguf (35.19 GiB) requires 2 GPUs. Tested on gfx1201 with
`HIP_VISIBLE_DEVICES=0,1` + `--split-mode tensor`.

| test | fused | unfused | delta |
|------|-------|---------|-------|
| pp512 ub512 | 5260 | 4852 | **+8.4%** |
| pp2048 ub2048 | 7184 | 6630 | **+8.1%** |
| pp16384 ub2048 | 6493 | 6067 | **+7.0%** |
| tg128 ub2048 | 97.5 | 97.6 | 0% |

Bit-exact: 536M logits, 0 diff (fused vs unfused on the SAME 2-GPU setup).
CUDA graphs on/off identical for the fused path (7166/7147).

## J cap for Q8_0: 64 (measured sweep)

Q8_0 uses the q8_0_q8_1_mma vec_dot path (not q8_1_q8_1_mma like Q4_K/
Q5_K). Fused has_gate register profile (fb=0): J=48 -> 135 VGPR, J=64 ->
163, J=96 -> 222, J=112 -> 251, J=128 -> 237 (all 0 spills). Sweep:

| cap | pp512 | pp2048 |
|-----|-------|--------|
| 128 | 4664 (-3.9%) | 6811 (+2.3%) |
| 96  | 4942 (+1.9%) | 6976 (+4.8%) |
| 64  | **5262 (+8.4%)** | **7182 (+7.9%)** |
| 48  | 5300 (+9.2%) | 7162 (+7.6%) |

J=64 (163 VGPR) is the sweet spot; J=48 marginally better at pp512 but
worse at pp2048. Final: J_max_gate Q8_0 = 64.

## MoE under tensor split - observations

- The fusion fires normally under tensor split: the try_fuse MMQ gate
  (ggml_cuda_should_use_mmq on src0->type/ne) is device-local; each GPU
  runs the fused gate+up+GLU on its slice of the expert weights.
- Bit-exact across the split: the per-weight accumulation order is
  preserved per device, and the split partial sums combine identically
  to the unfused path (0 diff confirms).
- Exchange/reduce cost: the op-timing dump (2719 vs 2731 ops) shows ~12
  fewer nodes; the tensor-split combine happens inside the split buffer
  machinery, not as explicit graph reduce nodes. NOTE: op-timing totals
  are inflated 4-5x on 2-GPU (per-op instrumentation overhead doubles
  the op count), so ONLY the llama-bench numbers are authoritative.
- The fused kernel does NOT reduce the all-reduce exchange count
  materially (the split exchange is per-tensor-part, not per-op); the
  win comes from the same kernel-level factors as single GPU (shared
  sort/quantize/launch + J-config).

## Final J_max_gate (all types)

```cpp
constexpr int J_max_gate =
    type == GGML_TYPE_Q6_K ? 64 :
    type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? 96 :
    type == GGML_TYPE_Q8_0 ? 64 : 128;
```

## Notes

- logitdump2 --split-mode is an INT: NONE=0, LAYER=1, ROW=2, TENSOR=3.
  ROW (2) fails on HIP ("does not support split buffers"); TENSOR (3) works.
- llama-bench --split-mode tensor (string) works.
