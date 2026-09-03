# Milestone 4 handoff: extend the fused gate+up MMQ kernel to Q4_K, Q5_K, Q8_0

## Current state (what the fused kernel looks like today)

The fused gate+up+GLU MMQ kernel (patch 0003, committed `72605d1`, in the
working tree) is enabled for **Q6_K only**. Everything is templated on the
quant type, so extending is mostly a matter of switching more types on; the
per-type vec_dot/load_tiles/write_back machinery is already shared.

## Files touched by patch 0003 (the whole change)

1. `ggml/src/ggml-cuda/mmq.cuh`
   - `mul_mat_q_process_tile` + `mul_mat_q` gained template bool `has_gate`
     (default false) + runtime args `x_gate`, `glu_op`, `glu_limit`.
     The gate K-loop reuses `tile_x`/`tile_y`; GLU epilogue is elementwise
     per thread-owned sum slot before `write_back`.
   - `mmq_args` gained `x_gate`, `glu_op`, `glu_limit` (defaults; the two
     aggregate-init call sites in mmq.cu are unaffected).
   - `mul_mat_q_switch_J` gained `has_gate` template param (default false)
     and now **caps J at 64 when has_gate** (see register problem below).
   - `mul_mat_q_case` gained `has_gate`; new `DECL_MMQ_CASE_GATE` macro.
2. `ggml/src/ggml-cuda/mmq.cu`
   - `ggml_cuda_mul_mat_q` takes optional `const ggml_cuda_mm_fusion_args_host * fusion = nullptr`.
   - When `fusion && fusion->gate`: asserts gate shares src0's type/shape/
     strides, then routes to new `ggml_cuda_mul_mat_q_switch_type_gate`.
   - The gate switch handles **Q6_K only** today; other types abort.
3. `ggml/src/ggml-cuda/ggml-cuda.cu`
   - try_fuse `{op, op, GLU}` branch gained an MMQ arm (after the mmvq/mmvf
     checks): `op == GGML_OP_MUL_MAT_ID && ids && !disable_moe_mmq &&
     ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], src0->ne[2])`.
   - Env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` (static bool at the top
     of try_fuse, next to `disable_fusion`).
4. `template-instances/generate_cu_files.py`
   - `MMQ_GATE_TYPES = ["GGML_TYPE_Q6_K"]`; `SOURCE_MMQ_GATE`; the generator
     appends `DECL_MMQ_CASE_GATE(type)` to each type's mmq-instance file.
5. `template-instances/mmq-instance-q6_k.cu`
   - Now has `DECL_MMQ_CASE(GGML_TYPE_Q6_K);` + `DECL_MMQ_CASE_GATE(...)`.

## The register problem (MUST keep in mind for new types)

The first cut was SLOWER than the 3-op path (2797 vs 2978 t/s at ub=512)
because `sum` + `sum_gate` (doubled accumulators) + WMMA tile registers
pushed the **J=128 config to 256 VGPR + 14 spills**. The fix that works:
**cap J at 64 for has_gate** in `mul_mat_q_switch_J`:

```cpp
// The fused gate+up kernel keeps two accumulators live (sum and
// sum_gate); the WMMA path at J>=96 needs ~256 VGPR and spills. Cap the
// tile width so the register budget stays inside the config occupancy.
if (has_gate && J > 64) {
    continue;
}
```

Cap sweep (pp512 ub=512 / pp2048 ub=2048 t/s): 32->3620/5081, 48->3559/5169,
64->3458/5244, 96->3156/5074, uncapped->2797/(regression), unfused->
2998/4892. **cap 64 chosen.**

For Q4_K/Q5_K/Q8_0 the register pressure per accumulator is SMALLER than
Q6_K (fewer bits per weight, different sram layouts), so cap 64 is likely
conservative-but-fine. Verify with the VGPR count extraction below if a type
regresses at ub=2048.

## What to change for each new type

### Step 1: enable the type in the gate switch (mmq.cu)

```cpp
static void ggml_cuda_mul_mat_q_switch_type_gate(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, true>(ctx, args, stream);  // WRONG
            ...
    }
}
```

Actually the switch must call the case matching the type:
`mul_mat_q_case<GGML_TYPE_Q4_K, true>(ctx, args, stream);` etc. The existing
single-type version is the template.

### Step 2: instantiate the gate case (generator + instance files)

- `generate_cu_files.py`: `MMQ_GATE_TYPES = ["GGML_TYPE_Q4_K",
  "GGML_TYPE_Q5_K", "GGML_TYPE_Q8_0", "GGML_TYPE_Q6_K"]` (keep Q6_K).
- Regenerate (or manually edit) `mmq-instance-q4_k.cu`, `mmq-instance-q5_k.cu`,
  `mmq-instance-q8_0.cu` to add `DECL_MMQ_CASE_GATE(GGML_TYPE_Q4_K);` etc.
  The generator appends the gate line to each file; if hand-editing, mirror
  the q6_k file exactly.
- Verify the type has a `GGML_CUDA_MMQ_SRAM_LAYOUT_*` mapping and RDNA4
  config entries (they do: Q4_K/Q5_K -> Q8_1 layout, Q8_0 -> Q8_0 layout,
  all present in mmq-config-rdna4.cuh).

### Step 3: correctness per type

- Bit-exactness: the fused kernel reproduces the per-weight accumulation
  order (same K tiling per weight inside the shared loop), so each type
  should be bit-exact like Q6_K was. Verify with logitdump:
  `fused vs GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` -> max abs diff must be 0.
- GLU op: the epilogue switch (SWIGLU/GEGLU/SWIGLU_OAI/SWIGLU_CLAMP) is
  type-independent; qwen35moe uses SWIGLU.
- If a type's config has stream_k=true (check `ggml_cuda_mmq_get_config`):
  the fused kernel asserts `!has_gate` on the stream-k path in
  `launch_mul_mat_q`, so stream-k configs cannot fuse (they will fall back
  to the 3-op path via should_use_mmq -> no fusion). RDNA4 configs are all
  stream_k=false, so fine on gfx1201; note for other GPUs.

## Validation recipe (repeat for each type)

1. `cmake --build build-rocm --target llama-bench -j 16`
2. Bit-exactness: build/run /tmp/logitdump2 (see below), compare fused vs
   `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` dumps: 0 diff.
3. Bench: `./llama-bench -m $M -p 512 -n 0 -ub 512 -r 2` (expect +14-16%)
   and `-p 2048 -n 0 -ub 2048 -r 2` (expect +6%).
4. Decode must stay 98 t/s (mmvq owns it; fusion must not fire).
5. `test-backend-ops -o MUL_MAT,MUL_MAT_ID -b ROCm0` OK.

## VGPR/spill check (only if a type regresses)

The metadata is msgpack in the .hip_fatbin of the mmq-instance-<type>.cu.o.
The working python parser lives in the session history; the essential trick:
find `\x83\xaeamdhsa.kernels`, parse map16(0xde) records, read `name` +
`vgpr_count` + `vgpr_spill_count`. Look for has_gate=1 kernels with
spill > 0 or vgpr == 256. If Q8_0 regresses, try capping J at 48 or 32 for
that type only.

## The other two gate switches (be careful)

`ggml_cuda_mul_mat_q_switch_type_gate` is the ONLY gate switch. The dispatch
in try_fuse calls `ggml_cuda_mul_mat_q` with the fusion arg; that function
checks `fusion->gate` and routes to the gate switch. The mmvq decode path
(`ggml_cuda_mul_mat_vec_q` with fusion) is UNTOUCHED and must stay that way
(decode uses it; it handles n_tokens <= MMVQ_MAX_BATCH_SIZE).

## Follow-ups after M4

- M5: cross-validate on qwen4exp (same build_moe_ffn hook; run the same
  recipe with its gguf, expect same win).
- Optional: per-shape J rule (cap relative to ncols_max) to squeeze ub=512.
- Server `--ubatch-size 2048` deployment test (Phase-3 validation item from
  plan-decode, still pending).
- **Q8_0 on 2 GPUs with `--split-mode tensor`** (see below).

## Q8_0 tensor-split testing (2 GPUs, `--split-mode tensor`)

NOTE: tensor-mode splitting is `--split-mode tensor`, NOT `row`. Row-mode
splitting is being phased out in llama.cpp; all the tensor-split work moved
into `--split-mode tensor`, and that is the flag to use.

The Q8_0 gguf is ~35 GiB (37801097504 bytes), so it does not fit a single
32 GiB R9700 - it requires 2 GPUs with tensor splitting. This is not just a
capacity workaround: MoE is one of the areas that scales POORLY under tensor
split today (the expert GEMMs need cross-GPU partial-sum exchange per
layer), so Q8_0 testing doubles as an opportunity to tune MoE for
`--split-mode tensor` operation. This is a real research item, not a
validation checkbox.

Context that matters for this:

1. **The fused gate+up MMQ kernel must be verified under tensor split**: the
   dispatch currently gates on `ggml_cuda_should_use_mmq(src0->type, cc,
   src1->ne[2], src0->ne[2])` and nothing else. Under tensor split, src0's
   ne[2] (n_expert) is unchanged (experts are NOT split across devices - it
   is the K/n_ff dims that split), so the fusion should still fire, but the
   per-device src1 partial rows and the cross-GPU exchange for the GLU
   output need a correctness check (logitdump on 2 GPUs vs 1-GPU Q6_K
   reference is NOT directly comparable across quant types - compare fused
   vs GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1 on the SAME 2-GPU setup).
2. **Block 12 (hybrid HIP all-reduce, RDNA4-gated) is the relevant infra**:
   the boosts branch's multi-GPU reduction path is what the expert
   partial-sums flow through. The plan-decode notes flagged "3-GPU config
   (hybrid all-reduce, block 12)" as a Phase-3 validation item; Q8_0 2-GPU
   is the first real workload to exercise it for MoE prefill.
3. **What to measure**: pp t/s on 2 GPUs (fused vs unfused), the per-layer
   all-reduce cost in the op timing (look for the reduce nodes between the
   gate/up and down), and whether the fused kernel reduces the exchange
   count (it collapses gate+up into one launch, so at least one exchange
   boundary disappears).
4. **Environment**: `HIP_VISIBLE_DEVICES=0,1` (2 GPUs); the multi-GPU load
   path fails under rocprof tracing (known), so use GGML_CUDA_OP_TIMING for
   measurements. Do NOT set HSA_OVERRIDE_GFX_VERSION.

## Model paths / env for reference

- moe Q6_K: /llm/models/Qwen3.6/35B-A3B/Q6_K/Qwen3.6-35B-A3B-Q6_K.gguf
- moe Q4_K_M: /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  (downloading 2026-08-31; ~6.5 GiB of expected ~8 GiB at last check)
- moe Q5_K_M: /llm/models/Qwen3.6/35B-A3B/Q5_K_M/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
  (downloading 2026-08-31; ~2.7 GiB of expected ~4 GiB at last check)
- moe Q8_0: /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
  (~35 GiB, needs 2 GPUs with --split-mode tensor)
- dense: /llm/models/Qwen3.6/27B/Q6_K/Qwen3.6-27B-Q6_K.gguf
- logitdump2: /tmp/logitdump2 (built from /tmp/logitdump.cpp with
  mp.n_lazy_buf_size removed); needs LD_LIBRARY_PATH=~/llama.cpp/build-rocm/bin.
- Op timing: `GGML_CUDA_OP_TIMING=1 llama-bench -p N -n 0 -ub N -r 1 -v`
  (look for "FUSED MUL_MAT_ID ffn_moe_gate-N" rows; the gate+up pair shows
  as one op).
