# Implementation Plan: Direct HF Safetensors Loading + FP8_E4M3 Inference

Fork: `stew675/llama.cpp` @ `6ea215d17` (2026-08-05), branch `cllm`.
Target model: `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B, multimodal, FP8_E4M3,
official Qwen FP8 convention, validated byte-for-byte against the Qwen 3.6 27B FP8 release).

This is a personal-fork effort. No upstream submission. Code must still follow the
repo style rules in `AGENTS.md` (ASCII only, concise comments, blend with surrounding code).

---

## 0. Goals and design principles (confirmed with the user)

1. **Direct loading of vanilla HuggingFace weights.** `llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/`
   must work with no GGUF conversion step. Weights are preserved as-is ("no conversions"):
   FP8 tensors stay FP8 and are consumed by FP8 kernels; BF16 tensors stay BF16.
2. **Native FP8_E4M3 inference, hardware-gated.** `GGML_TYPE_FP8_E4M3` is consumed only
   by hardware with **native FP8 support**. Target: 3x `gfx1201` (RDNA4, native FP8 WMMA
   + v_dot4). CUDA sm_89/sm_90 paths are included (native FP8 there too) but untested.
3. **No software-emulated FP8, ever.** If the system has no FP8-capable device, loading
   FP8 weights is **rejected with a clear "no hardware support" error**. Rationale
   (user's position): FP8 has zero advantage over an integer GGUF (Q8_0 etc.) without
   native FP8 hardware - same memory footprint, worse accuracy, more per-element work.
   Users without FP8 hardware should use a normal integer GGUF. There are therefore
   **no CPU FP8 kernels** (no conversion work, no AVX paths).
4. **Multimodal via the existing mmproj.gguf mechanism.** Qwen3.5 vision tower is handled
   by the existing tools-layer mtmd + `--mmproj` flow (converter already registers
   `Qwen3_5ForConditionalGeneration` for mmproj export; `tools/mtmd/clip.cpp` already
   supports the deepstack vision encoder). The direct loader skips `model.visual.*`
   tensors (vision tower is BF16, unaffected by the FP8 gate).
5. **MTP/NextN works with FP8.** The MTP block's 7 projection weights
   (`mtp.layers.0.self_attn.{q,k,v,o}_proj`, `mtp.layers.0.mlp.{gate,up,down}_proj`)
   are FP8 and run through the same fp8 kernels and hardware gate as the main layers
   (`graph_mtp` in `qwen35.cpp` uses the same `ggml_mul_mat` path). Only `mtp.fc` and
   the MTP norms are BF16 (official release convention).

Per-token compute note (agreed with user): FP8 does not reduce FLOPs; it halves weight
memory vs BF16 (~5.3GB -> ~2.6GB) and roughly doubles matmul throughput **on native
FP8 hardware** (RDNA4: 389 TFLOPS FP8 vs 195 TFLOPS FP16). The linear-attention layers
of this architecture additionally do less work per token than full attention; that is
already implemented in `src/models/qwen35.cpp` and is independent of the FP8 work.

---

## 1. Current state (verified facts)

### What already exists
- `src/models/qwen35.cpp` (645 lines): full Qwen3.5 text support - hybrid linear
  attention (gated delta net), full-attention layers every 4th block, MTP/NextN,
  MRoPE (`rope_sections`), tied embeddings. No vision in this file (vision lives in mmproj).
- `conversion/qwen3vl.py:16` registers `Qwen3_5ForConditionalGeneration` for **mmproj**
  export. `tools/mtmd/clip.cpp` supports deepstack vision layers (qwen3vl-style).
- **`llama_model_init_from_user()`** (`include/llama.h:489`) - public API that builds a
  model from an in-memory `gguf_context * metadata` plus a `llama_model_set_tensor_data_t`
  callback. When the loader has no backing files it calls `set_tensor_data(t, ud)` for
  every tensor (`src/llama-model-loader.cpp:1407`). This is the hook for direct loading.
- `gguf_init_empty()` / `gguf_add_tensor()` / `gguf_set_val_*` / `gguf_set_arr_data()`
  (`ggml/include/gguf.h:82,140,163`) - enough to synthesize GGUF metadata in memory.
- `convert_hf_to_gguf.py --vocab-only` (`convert_hf_to_gguf.py:52`, `write_vocab()` at
  `conversion/base.py:1309`) - emits a tiny GGUF with only KV metadata (tokens, scores,
  merges, added tokens, pre-tokenizer id, special ids, chat template). The pre-tokenizer
  family detection (`get_vocab_base_pre`) and BPE handling stay in Python.
- `vendor/nlohmann/json.hpp` - JSON parsing for `config.json` / safetensors headers.
- `src/llama-mmap.h` `llama_mmap` - mmap helper reusable for safetensors files.
- Unified `ggml-cuda` sources compiled for HIP via `ggml/src/ggml-hip/CMakeLists.txt`;
  `vendors/hip.h:248` already defines `FP8_AVAILABLE` and `__nv_fp8_e4m3` on HIP.
- ROCm 7.14 at `/opt/rocm-7.14` (hipcc, runtime 7.15, `__hip_fp8_e4m3` OCP type).
  `rocm_agent_enumerator` sees 3x `gfx1201` + 1x `gfx1036`.
- `/home/stew675/rocm-libraries/` - ROCm math libraries **source** super-repo
  (hipblas, rocblas, rocwmma, composablekernel, ...). Required by `ggml-hip`
  (`find_package(hipblas REQUIRED); find_package(rocblas REQUIRED)`), not yet built.
- RDNA4 (gfx1201) native FP8 instructions, verified against AMD's ISA spec:
  `V_WMMA_F32_16X16X16_FP8_FP8` (and FP8/BF8 mixed forms), `V_DOT4_F32_FP8_FP8`.
  No `V_DOT2_F32_FP8` on RDNA4 (that is CDNA3). 389 TFLOPS FP8 dense (9070 XT).

### What does NOT exist (the work)
- No `GGML_TYPE_FP8_E4M3` in `ggml.h` (enum ends at `Q2_0 = 42`). Upstream PR #25336
  proposes a QK=32 variant; never merged. NVFP4's E4M3 is a 4-bit-weight *scale* type,
  unrelated.
- No safetensors reader anywhere in `src/` (only Python-side in `conversion/`).
- No FP8 compute kernels in ggml-cuda/ggml-hip (the only fp8 references are NVFP4's
  UE4M3 scale conversions and the hip.h type typedefs).
- `qwen35.cpp` has no vision; vision handled via mmproj (to be verified end-to-end).
- `ggml-hip` build requires hipblas/rocblas which are not installed (only source).

---

## 2. Design decisions

### D1. FP8 type layout: `GGML_TYPE_FP8_E4M3`, QK=128, f32 block scale
- `GGML_TYPE_FP8_E4M3 = 43` (append at end of enum; `GGML_TYPE_COUNT` -> 44).
- Block = `[128 x fp8 bytes][f32 scale]` = 132 bytes, 1.03125 B/elem.
- Layout maps 1:1 onto the Qwen/StewFP8 convention. Qwen stores
  `weight` [out, in] F8_E4M3 + `weight_scale_inv` [ceil(out/128), ceil(in/128)] BF16 with
  `w[m,k] = fp8(q[m,k]) * S[m/128, k/128]`. In GGML blocked layout the block
  `(m, cb)` carries scale `S[m/128, cb]`, duplicated across the 128 rows of a row-block
  (128x scale duplication = 3.1% size overhead, but keeps blocks self-contained and
  all standard ggml machinery working with zero kernel-indexing changes).
- Scale stored as **f32** (converted from BF16 at load, exact) so per-element dequant
  is bit-identical to the HF reference `fp8_to_f32(q) * bf16_scale`.
- Decode: standard OCP `e4m3fn` (1 sign, 4 exp bias 7, 3 mantissa; max 448; no NaN/Inf
  in this model - bytes 0x7F/0xFF unused).
- **No CPU kernels.** ggml core registers the type for serialization/validation only
  (name, blck_size, type_size, finite-check). No `vec_dot`/`to_float`/`from_float`
  type_traits. The CPU backend rejects any op touching fp8 tensors (see D2/D3).
- No activation-format decision needed: the GPU kernels consume fp8 weights with
  BF16/F16 activations (activations are never quantized on the fp8 path).

### D2. Hardware gate (backend level)
- `ggml_cpu_supports_op` returns false for any op with an F8_E4M3 operand, so the
  backend scheduler never routes fp8 work to CPU (defense in depth; avoids hitting
  NULL type_traits in the CPU compute path).
- The HIP backend declares support for the fp8 ops it implements (`mul_mat`, plus any
  `get_rows`/`cpy` needed for data movement).
- If the graph contains fp8 ops that no registered backend supports, ggml's scheduler
  fails with the standard "no suitable backend" error; llama.cpp surfaces it.

### D3. Hardware gate (load level, user-facing)
- At model load, after tensor placement is decided: every F8_E4M3 tensor must be
  placed on a buffer whose device supports fp8 ops; otherwise the load **aborts** with
  a clear error, e.g.:
  `FP8_E4M3 weights require a device with native FP8 support (RDNA4, Hopper/Ada, MI300+); this system has none. Use an integer GGUF (e.g. Q8_0) instead.`
- If no FP8-capable device exists at all, the same error fires before any tensor work.
- Consequence: `--n-gpu-layers` below the number of fp8 layers is an error, not a
  silent CPU fallback. Document this (fp8 layers must be fully offloaded).

### D4. Direct loader architecture: synthesize GGUF metadata + data callback
Reuse `llama_model_init_from_user` instead of writing a parallel loader:
1. Parse `config.json` (arch, hparams) + safetensors index/headers.
2. Synthesize an in-memory `gguf_context` (arch, all hparams, tensor list with GGUF
   names/dims/types, vocab keys copied from `tokenizer.gguf`).
3. mmap all safetensors files.
4. `set_tensor_data` callback: map GGUF tensor name -> HF tensor, apply transforms
   (below), fill `tensor->data`.
5. `llama_model_init_from_user(metadata, cb, ud, params)` - everything downstream
   (hparam checks, layer loading, buffer placement/offload, MTP flag, vocab) is unchanged.

Key simplification, verified: **no byte-level transpose is needed** for 2D weights.
HF stores `[out, in]` row-major; ggml reads the same bytes as `ne[0]=in`. Only the
shape is reinterpreted (this is exactly what `conversion/base.py` relies on).

Transforms to port from `conversion/qwen.py` to the loader (per tensor, at load time;
all are pure data movement / byte rearrangement, no float compute):
- `linear_attn.A_log` -> `-exp(A_log)` (stored as `ssm.a_noscan`). (Small [32] tensor;
  exp on host is fine - not an inference op.)
- `linear_attn.dt_bias` -> renamed to `dt_proj.bias` (no data change).
- `linear_attn.conv1d.weight` [8192,1,4] -> squeeze -> [8192,4] (ggml `{4, 8192}`).
- norms (`*_norm.weight`, not `linear_attn.norm.weight`) -> `+1`.
- V-head reorder when `linear_num_key_heads (16) != linear_num_value_heads (32)`:
  `in_proj_qkv` (V rows only), `in_proj_z` (rows), `in_proj_a/b` (rows),
  `A_log`/`dt_bias` (1D), `conv1d` (V channel portion), `out_proj` (columns).
  Port `_reorder_v_heads` / `_LinearAttentionVReorderBase` exactly.
- FP8 re-blocking: for row m, col-block cb: copy 128 fp8 bytes, append
  `f32(S[m/128][cb])` where `S` comes from the BF16 `weight_scale_inv` tensor.
  Contiguous per row (HF row-major), cheap and parallelizable.
- MTP: `mtp.layers.0.*` -> `blk.32.*` (n_layer_nextn=1); `mtp.fc.weight` -> nextn
  eh_proj `{2*n_embd, n_embd}`; `mtp.*_norm.weight` etc. per tensor map.

gguf context synthesis must reproduce the exact hparams keys the qwen35 loader reads
(`src/models/qwen35.cpp:7-35`): rms eps (1e-6), rope sections `[11,11,10,0]` (required),
`ssm.conv_kernel=4`, `ssm.state_size=128` (linear_key_head_dim), `ssm.group_count=16`
(linear_num_key_heads), `ssm.time_step_rank=32` (linear_num_value_heads),
`ssm.inner_size=4096` (linear_value_head_dim * linear_num_value_heads),
`nextn.predict_layers=1`, `full_attention_interval=4` (or recurrent-layers array),
plus standard QWEN2 keys (context_length 262144, embedding_length 2560, block_count 32,
head_count 16, head_count_kv 4, key/value_length 256, feed_forward_length 9216,
rope_dimension_count 64 = 256 * partial_rotary_factor 0.25, rope_freq_base 1e7,
vocab_size 248320, arch/name/file_type). Mirror `conversion/qwen.py` exactly.

### D5. Tokenizer: converter-assisted (user-approved)
- One-time per model dir:
  `python3 convert_hf_to_gguf.py /llm/models/Qwen3.5/4B/StewFP8 --vocab-only --outfile .../tokenizer.gguf`
- Loader opens `tokenizer.gguf` (auto-detect `<model-dir>/tokenizer.gguf`, override via
  `--vocab-file`), copies all `tokenizer.*` (+ relevant `general.*`) KV pairs into the
  synthesized context with `gguf_set_val_*` / `gguf_set_arr_data`.
  `llama_vocab_load` (`src/llama-vocab.cpp:2401+`) then works unchanged.
- Loader errors with the exact one-liner if the file is missing.
- No tokenizer.json parsing in C++. Pure-C++ parsing remains a possible future milestone.

### D6. Multimodal: existing mmproj flow (user-approved compromise)
- Text loader ignores `model.visual.*` (299 tensors in `outside.safetensors`; all BF16,
  no FP8 gate involvement).
- Generate `mmproj.gguf` with the existing converter (MmprojModel path for
  `Qwen3_5ForConditionalGeneration`). Verify clip/mtmd runs it (`mtmd-cli`, server
  `--mmproj`). No changes expected in `tools/mtmd`; verification is the work.

### D7. Scope
- GPU kernels: ROCm for gfx1201 (native FP8 WMMA + v_dot4), CUDA sm_89/sm_90
  (`mma.sync` e4m3) compile-guarded, untested.
- **No CPU FP8 kernels of any kind** (per user decision, D3).
- Fork-local; no compatibility obligation with upstream PR #25336 (noted as future option).

---

## 3. Milestones

### M0. Environment bring-up (ROCm build)

Environment facts (verified):
- Build script `/home/stew675/bin/build-llama-rocm`: uses system ROCm via `hipconfig -l`
  (`/usr/lib64/rocm/llvm/bin/clang`, ROCm 7.1.1, clang 20), `-DGGML_HIP=ON`,
  `-DGPU_TARGETS="gfx1200;gfx1201"`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_RCCL=1`,
  `GGML_CUDA_NO_PEER_COPY=1`, `GGML_RPC=1`, `CMAKE_HIP_FLAGS="-mllvm
  --amdgpu-unroll-threshold-local=600"`.
- hipblas/rocblas ARE installed as system packages (`/usr/lib64/cmake/hipblas`,
  `/usr/lib64/librocblas.so.5.1`), so `find_package(hipblas/rocblas REQUIRED)` in
  `ggml-hip` succeeds. Building them from `/home/stew675/rocm-libraries` source is
  NOT required (only needed if newer libs are wanted later).
- `/opt/rocm-7.14` (ROCm 7.15 runtime, clang 23) also present; not required for the
  FP8 work (see M2 spike) but available.

Tasks:
- [x] Baseline: the HIP build (`-DGPU_TARGETS="gfx1200;gfx1201"`) builds and all 3
      gfx1201 GPUs are usable with `--split-mode layer` and `row` (verified during
      M4 multi-GPU parity).
- [x] Baselines recorded: Q8_0 GGUF gen 91.7 t/s / prompt 348 t/s; BF16 dir
      gen 57-71 t/s; PPL (Pride & Prejudice, 512-ctx): bf16 8.60, Q8_0 8.61,
      fp8 8.72 (both paths).
- [x] `rocm_agent_enumerator` = 3x gfx1201 + 1x gfx1036, matches the build targets.

### M1. `GGML_TYPE_FP8_E4M3` core type (serialization/validation only, NO CPU kernels)

Status: **DONE** - all items below implemented and verified.
- [x] `ggml/include/ggml.h`: enum entry `GGML_TYPE_F8_E4M3 = 43`, `GGML_TYPE_COUNT = 44`.
- [x] `ggml/src/ggml-common.h`: `QK_F8_E4M3 128` + `block_f8_e4m3 { float d; uint8_t qs[128]; }`
      (132 B, static_assert'd).
- [x] `ggml/src/ggml.c`: `type_traits` entry - blck_size 128, type_size 132,
      is_quantized, name "f8_e4m3", **no vec_dot/to_float/from_float** (NULL);
      `ggml_quantize_chunk` has an explicit `GGML_ABORT` case for f8_e4m3
      (no software quantization; llama-quantize output is M5 and needs native HW).
      Note: no `gguf.cpp` change was needed - GGUF stores tensor types as the
      integer `ggml_type` enum, so the new type serializes automatically.
- [x] `ggml/src/ggml-quants.c`: `ggml_validate_row_data` case for f8_e4m3
      (rejects NaN bytes 0x7F/0xFF and non-finite scales).
- [x] **CPU rejection**: `ggml_backend_cpu_device_supports_op` returns false for any
      op with an F8_E4M3 operand (src or dst), after the passthrough-op block
      (`ggml/src/ggml-cpu/ggml-cpu.cpp`).
- [x] Tests: `test-quantize-fns` passes (f8_e4m3 auto-skips its quant checks, no
      from_float/to_float); `test-backend-ops` gets `GGML_TYPE_F8_E4M3` in `all_types`
      (mul_mat/get_rows/cpy cases; "not supported" on CPU, run on HIP once M2 lands;
      NOTE: M2 must provide data init for these tests - host reference quantize or
      test-side fp8 block builder, since `ggml_quantize_chunk` aborts).
- [x] Verified (standalone check program linked against the built libs): name/blck/size/
      quantized flags correct; 256x256 tensor = 67584 B; validate_row_data rejects
      NaN byte + NaN scale; CPU `ggml_backend_supports_op` = false for fp8 mul_mat,
      true for f16 mul_mat.
- [x] Python cross-check vs torch on **all 207 StewFP8 fp8 tensors (3.7B values)**: raw
      byte decode bit-exact with `torch.float8_e4m3fn` (incl. subnormals); dequant
      rule `w = fp8_to_f32(q) * scale_inv[m/128, k/128]` confirmed; scale range
      ~6e-5..1.5e-3 as documented in FORMAT.md.

### M2. GPU kernels (ROCm gfx1201 primary, CUDA guarded)

Spike result (already done, verified by compiling a test kernel):
- **Native FP8 WMMA + dot4 confirmed on BOTH toolchains** for gfx1201:
  - System ROCm 7.1.1 clang 20 (the build script's compiler): emits
    `v_wmma_f32_16x16x16_fp8_fp8` and `v_dot4_f32_fp8_fp8`.
  - `/opt/rocm-7.14` clang 23: same, plus `w64` variants and mixed fp8/bf8 forms.
- Builtins (wave32, what llama.cpp uses):
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` with A/B fragments as
  `v2i32` (8 fp8 per lane = 8 bytes) and accumulator as `v8f32` (8 floats per lane);
  `__builtin_amdgcn_dot4_f32_fp8_fp8(i32, i32, float)`.
- The `w64` variants need the `wavefrontsize64` feature; not needed (ggml-cuda uses
  wave32 on gfx12, same as the existing f16 `w32_gfx12` path).

Tasks:
- [x] fp8 tile-load + mmq-style mul_mat kernel (WMMA path): `v_wmma_f32_16x16x16_fp8_fp8` with the verified gfx12 fragment layout. **Fragment layout (empirically verified on gfx1201 + TileLang/CK docs): A lane l byte e = A[l%16][(l//16)*8+e]; B lane l byte e = B[(l//16)*8+e][l%16]; C slot s = C[(l//16)*8+s][l%16].** Staging is [n][k] (token-major, k-minor) so B fragments are contiguous 8-byte loads. Correctness: standalone test + 12-config stress suite PASS (max rel err <= 3.4e-3).
- [x] dot4 path (GEMV only): `mul_mat_fp8_gemv` uses `__builtin_amdgcn_dot4_f32_fp8_fp8`
      for n <= 16 tokens (avoids the wmma 16-token tile waste at batch 1). A full
      dot4 matmul for large n is not needed - WMMA covers it.
- [x] CUDA side: NOT implemented - documented instead (see section "CUDA sm_89/sm_90
      fp8 path"). The unguarded scalar fallback runs on CUDA (correct, slow); the
      sm_89 mma PTX fragment layout cannot be verified without NVIDIA hardware, so
      it is deliberately not shipped untested.
- [x] Backend registration: HIP backend `supports_op` for f8_e4m3 mul_mat (RDNA4 gate); host-side launcher `ggml_cuda_mul_mat_fp8` pre-quantizes activations (fp8 staging + scales).
- [x] `tests/test-backend-ops` builds; f8_e4m3 MUL_MAT shows "not supported [CPU]" (correct skip) and runs on the ROCm devices; no FAILs. (NOTE: re-run `cmake .` in the build dir so the GLOB picks up fp8.cu.)
- [x] Perf: 1x R9700 generation: FP8 safetensors 72.5 t/s vs Q8_0 GGUF 91.7 t/s vs BF16 safetensors 59-72 t/s. Prompt: FP8 143 t/s vs Q8_0 348 t/s. (Two fixes were needed: the host-side RDNA4 dispatch was compiled out by a device-only #if guard -> scalar fallback; and a dot4 GEMV path for n <= 16 avoids the wmma batch-1 waste.)

### M3. Direct safetensors loader (text model) + hardware gate
- [x] `src/llama-safetensors.h/.cpp` (new): safetensors file parse (u64 header len +
      JSON header; mmap via `llama_mmap`; dtype/shape/data_offsets), index.json
      `weight_map`, single-file fallback, config.json parse (nlohmann).
- [x] Arch dispatch: `architectures[0]` -> qwen35 only (rejects others with a clear
      message). `text_config` -> hparams.
- [x] Tensor name map HF -> GGUF for qwen35 (port from `conversion/qwen.py`;
      MTP names, skip `model.visual.*`).
- [x] gguf_context synthesis (all hparams keys + tensor list via `gguf_add_tensor`
      from a scratch ggml context).
- [x] `set_tensor_data` implementation: per-tensor transforms (D4 list), fp8 re-blocking,
      BF16/F32 passthrough, V-head reorder, tie_word_embeddings. Output types mirror
      conversion/base.py (1D + *_norm.weight + conv1d are F32 - required, the GPU mul
      op does not support BF16). Source dtype awareness (A_log/norm are F32 in the
      original model, BF16 in StewFP8).
- [x] **Hardware gate** (D3): `has_fp8_device()` probes registered devices with a dummy
      F8 mul_mat; fires for CPU-only builds and for `--n-gpu-layers` partial offload
      (< n_layer_all + 1) with clear messages.
- [x] Vocab merge from `tokenizer.gguf` (D5) with clear one-liner error if absent.
- [x] Integration: in `llama_model_load_from_file` (`src/llama.cpp:445`), auto-detect
      safetensors input and route to the new loader. All tools inherit it via
      `common/common.cpp:1250`.
- [x] `llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/ -p "..."` works on 1 and 3 GPUs
      (58-72 t/s generation); `--spec-type draft-mtp` works (MTP block offloaded and
      executing on GPU, 38 t/s).
- [x] Negative test: CPU-only build loading the fp8 dir -> exact "no hardware support"
      error, clean exit. `--n-gpu-layers 10` -> exact partial-offload error.
      NOTE: `--no-alloc` / `--vocab-only` skip tensor loading entirely (verified sane).
- [x] mmproj companion: `convert_hf_to_gguf.py <orig-dir> --mmproj` produces a
      298-tensor 675MB mmproj.gguf (NOTE: must use the standard-named original dir,
      `model.safetensors-*.safetensors`; the StewFP8 `layers-*.safetensors` layout is
      invisible to `get_model_part_names("model", ...)`).
- [x] `llama-cli -m <safetensors dir> --mmproj <mmproj.gguf> --image <img>` works:
      vision tower reads the actual image content (NYT "MEN WALK ON MOON" front page
      correctly identified). `llama-server` + mmproj + image via the chat API works too
      (content went to reasoning_content - normal Qwen3.5 thinking mode).
      CRITICAL FIX discovered: llama-server batches 4 sequences (ne12=4) and the fp8
      mul_mat launcher asserted ne12==1 - added a batch loop over ne12*ne13 in
      `ggml_cuda_mul_mat_fp8` and dropped the `b->ne[2]*b->ne[3] != 1` support check.

### M4. Validation & parity (GPU)
- [x] Greedy parity: llama.cpp-BF16-via-loader == HF-BF16 greedy, byte-exact over 20
      tokens; **llama.cpp-FP8 == llama.cpp-BF16 for all 20 tokens** after the encode
      fix below (before the fix it diverged at token 10 on a True/False near-tie).
      `parity_check.py` (torch): fp8-dequant weights == bf16 weights greedy-identical.
- [x] PPL sanity (Pride and Prejudice, 512-token windows, n_ctx=512, n_batch=2048):
      FP8 9.9907 vs BF16 9.8881 vs Q8_0 9.8881 - fp8 within 1% of baseline.
      **CRITICAL BUG FOUND+FIXED**: `fp8_e4m3_from_f32` had `if (E >= 15) return 0x7E`
      which saturated ALL values in the top exponent range (256-448, the top ~43% of
      each 128-block's dynamic range) to 448. Activations got systematically inflated
      -> logits shifted ~2.0, PPL 23.8 (2.4x). Fix: only saturate for E >= 16 (or the
      E=15/man3=7 NaN pattern). The self-consistent stress tests missed it because the
      host reference shared the same buggy encode. After the fix: logits track bf16
      within ~0.1-0.25, PPL 9.99, generation parity exact. Also fixed the encode copy
      in the test harnesses.
- [x] Multi-GPU: default layer split across 3x gfx1201 == single GPU output, identical
      (tested with 40-token greedy, temp 0).
- [x] Memory (1x R9700, default n_ctx=262144 from the model): fp8 model buffer
      4730 MiB + KV cache 8192 MiB + RS 50 MiB + compute 320+266 MiB = ~14.3GB peak
      on the GPU during generation (rocm-smi confirmed). BF16 model: ~17.8GB peak.
      The fp8 weight savings show in the model buffer (4.7GB vs 7+GB incl. the tied
      token_embd); the KV cache at full context dominates both.
- [x] Edge cases: single-file safetensors (no index) loads; non-128 fp8 k-dim rejected
      with a clear message ("must be a multiple of 128"); missing scale_inv rejected;
      E5M2 tensors rejected ("unsupported tensor dtype"); corrupt header wrapped with
      the file name; missing tokenizer.gguf gives the regenerate one-liner. All
      verified with synthetic model dirs.
- [x] `--vocab-file` override plumbed: `llama_model_params.vocab_file` +
      `common_params_model.vocab_file` + `--vocab-file FILE` (default
      `<model-dir>/tokenizer.gguf`).

### M5. (DONE 2026-08-06) GGUF-side FP8 preservation

`--outtype fp8_e4m3` writes FP8 weights as native `block_f8_e4m3` GGUF tensors
instead of dequantizing them. The safetensors loader (M3) remains for direct HF
dirs; M5 adds the GGUF round trip so fp8 models can also be distributed as a
single standard GGUF file.

- [x] `gguf-py/gguf/constants.py`: `GGMLQuantizationType.F8_E4M3 = 43` (matches
      ggml.h), `GGML_QUANT_SIZES[F8_E4M3] = (128, 132)`,
      `LlamaFileType.MOSTLY_F8_E4M3 = 42`.
- [x] `include/llama.h`: `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42`; `llama-model-loader.cpp`
      ftype name + type->ftype guess map.
- [x] `conversion/base.py`: `_generate_fp8_tensors()` reblocks `weight` +
      `weight_scale_inv` pairs into block_f8_e4m3 rows `[out, in/128 * 132]`
      (d = stored scale_inv value, fp8 bytes verbatim) and writes them directly
      (NVFP4-style, before dequant_model). Runs when
      `ftype == MOSTLY_F8_E4M3`. Non-fp8 tensors default to BF16. `transform_fp8_weight`
      hook for arch-specific reorders; E5M2 rejected; 2D-only; warns when no fp8
      tensors found. GUESSED heuristic detects float8_e4m3fn.
- [x] `conversion/qwen.py`: `transform_fp8_weight` V-head reorder with the scale
      grid in 128-row/col units (`row_perm[::128] // 128`).
- [x] `convert_hf_to_gguf.py`: `--outtype fp8_e4m3` choice (replaces the earlier
      `--preserve-fp8` flag per review feedback).
- [x] index fallback: dirs with nonstandard part naming (StewFP8 `layers-*.safetensors`)
      now load via any `*.safetensors.index.json` weight map.
- [x] D3 gate extended to the plain GGUF path in `llama_model_load` (fp8 tensors +
      no fp8 device -> same clear error as the safetensors path).
- [x] Loader: `general.file_type = 42` written for fp8 safetensors models.
- [x] Verified: 207 F8_E4M3 tensors in the GGUF, block bytes byte-identical to the
      source safetensors; tokens identical to the safetensors path and bf16;
      PPL 8.715 (fp8 GGUF) == 8.716 (fp8 safetensors) vs 8.602 (bf16);
      gen 73-75 t/s, MTP 101 t/s, llama-server works (batched fp8); CPU-only
      build -> the D3 error message.

## CUDA sm_89/sm_90 fp8 path (DONE 2026-08-06 - documented)
The RDNA4 WMMA kernels are AMD-builtin only; on CUDA the unguarded
`mul_mat_fp8_scalar` fallback runs (correct, slow). Implementing the sm_89
`mma.sync.m16n8k8.f32.e4m3.e4m3` PTX path without NVIDIA hardware to verify the
fragment layout is exactly the class of untestable code AGENTS.md forbids, so it
is documented as supported-but-scalar rather than shipped untested. The D3 gate
message ("RDNA4, or NVIDIA Ada/Hopper+") matches reality: CUDA Ada/Hopper can
load fp8 models, they just run the scalar kernel.

## 4. Files touched (summary)

Core type (no CPU compute):
- `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-quants.h/.c` (validate only),
  `ggml/src/gguf.cpp` (type id mapping), `ggml/src/ggml-cpu/ggml-cpu.c` (reject fp8 ops)

GPU kernels:
- `ggml/src/ggml-cuda/`: `mmq*.cuh`, `mma.cuh`, `vecdotq.cuh`, `common.cuh`,
  `ggml-cuda.cu` (dispatch + supports_op), template instances

Direct loader:
- `src/llama-safetensors.h/.cpp` (new), `src/llama.cpp` (integration + auto-detect +
  hardware gate), `src/llama-model-loader.cpp` (minor: verify flags work with external
  data source), `common/common.cpp` (help text, `--vocab-file`), CMakeLists for the new file

Tests:
- `tests/test-quantize-fns.cpp` (skip fp8), `tests/test-backend-ops.cpp` (fp8 on GPU)

Optional GGUF:
- `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py`, `conversion/base.py`

Docs:
- `docs/safetensors.md` (new): usage, tokenizer.gguf one-liner, mmproj flow, FP8
  hardware requirement + error text, format reference to `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md`

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RDNA4 FP8 WMMA not exposed by compiler | **RESOLVED (spike done)**: both system clang 20 (ROCm 7.1.1) and rocm-7.14 clang 23 emit `v_wmma_f32_16x16x16_fp8_fp8` / `v_dot4_f32_fp8_fp8` for gfx1201; builtin spellings and fragment layout verified |
| Activation quantization differs from HF reference (per-token fp8) | Accept near-parity; document; PPL check gates quality |
| hipblas/rocblas not installed (only source) | M0 builds them from `/home/stew675/rocm-libraries`; or install full ROCm |
| Tokenizer.gguf staleness / missing | Auto-detect + actionable error; regenerate per tokenizer change |
| V-head reorder or transform bugs cause silent wrong outputs | Unit-test each transform against the Python converter output; parity gate in M4 |
| qwen3.5 mmproj.gguf config mismatch (pos 2304 vs qwen3vl) | M3 verification item; clip.cpp reads dims from mmproj metadata, likely fine |
| 3x gfx1201 split-mode quirks | Baseline in M0; row-split parity check in M4 |
| fp8 tensors accidentally routed to CPU (NULL type_traits assert) | D2/D3 double gate: supports_op rejection + load-time placement check; negative tests in M3 |
| Scope creep (vision port, tokenizer.json C++ parser, CPU fp8, FP4, from_float, ...) | Explicitly out of scope (section 6) |

## 6. Out of scope (future)
- Vision tower inside `qwen35.cpp` (mmproj flow covers it).
- Pure-C++ tokenizer.json parser.
- Any software-emulated FP8 (CPU kernels, non-native GPU fallback) - rejected by design (D3).
- FP8_E4M3 QK=32 type for generic re-quantization (upstream #25336 alignment).
- E5M2 / FP8 activation quantization / FP4.
- `llama-quantize` f8_e4m3 output (from_float) unless explicitly requested.
- Repo streamlining (user's later project).
