# Handoff — FP8 E4M3 + Direct Safetensors Loading for llama.cpp

Last updated: 2026-08-05, end of session. Working tree state captured below.

## 1. Project overview

Extending the `stew675/llama.cpp` fork (branch `cllm`, commit `6ea215d17` + local changes) to:

1. **Run FP8_E4M3 weights natively** (`GGML_TYPE_F8_E4M3`, QK=128, f32 block scale), hardware-gated: **only** native-FP8 devices (RDNA4 gfx1201, CUDA sm_89+). NO CPU kernels, NO software emulation — by explicit user decision (FP8 has zero advantage without native HW; users without it should use an integer GGUF).
2. **Load vanilla HuggingFace safetensors directly** (no GGUF conversion), using the existing `llama_model_init_from_user()` API + synthesized in-memory gguf_context.
3. Target model: `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B, multimodal, FP8 E4M3, official Qwen 128x128-block convention). Format spec in `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md`.
4. Multimodal via existing mmproj.gguf + mtmd flow (converter already registers Qwen3_5ForConditionalGeneration for mmproj).
5. Full plan: `implementation-plan.md` in repo root (read it first; M1 marked DONE).

Personal fork, no upstream submission. Follow AGENTS.md style (ASCII only, concise comments).

## 2. Environment (all verified)

- **CPU**: AMD Ryzen 9 9950X3D (16 cores, Zen 5).
- **GPUs**: 3x AMD Radeon AI PRO R9700, **gfx1201** (RDNA4), wave32, 32GB VRAM each (total ~95GB), + 1x gfx1036 (RDNA2, no FP8). `rocm_agent_enumerator` confirms.
- **ROCm**: System ROCm 7.1.1 (Fedora 44): hipcc/clang at `/usr/lib64/rocm/llvm/bin`, hipblas/rocblas installed as packages (`/usr/lib64/cmake/hipblas`). Also `/opt/rocm-7.14` (ROCm 7.15, clang 23) — both work for FP8 WMMA. **The build script uses the system one.**
- **Build script**: `/home/stew675/bin/build-llama-rocm` — uses `hipconfig -l` (system ROCm), `-DGGML_HIP=ON -DGPU_TARGETS="gfx1200;gfx1201" -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=1 -DGGML_CUDA_NO_PEER_COPY=1 -DGGML_RPC=1`, clang/clang++.
- `/home/stew675/rocm-libraries/` — ROCm math libs source (hipblas/rocblas/rocwmma) — NOT needed to build (system packages suffice).
- Python: torch 2.12.0 (CPU build), safetensors available.

## 3. Completed milestones

### M0 (environment) — verified
- HIP build works; 3562 MUL_MAT tests pass on the 3 R9700 GPUs (baseline).
- Both clang 20 (system) and clang 23 (/opt/rocm-7.14) emit native FP8 instructions for gfx1201:
  `v_wmma_f32_16x16x16_fp8_fp8` and `v_dot4_f32_fp8_fp8`.
- Builtin spellings (wave32): `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` with A/B fragments `v2i32` (8 fp8/lane = 8 bytes), acc `v8f32`; `__builtin_amdgcn_dot4_f32_fp8_fp8(i32, i32, float)`. w64 variants need `wavefrontsize64` — unused (llama.cpp uses wave32).

### M1 (FP8 type registration) — DONE and COMMITTED (commit `d56412c8c`)
- `ggml/include/ggml.h`: `GGML_TYPE_F8_E4M3 = 43`, `GGML_TYPE_COUNT = 44`.
- `ggml/src/ggml-common.h`: `block_f8_e4m3 { float d; uint8_t qs[128]; }` (132B, static_assert).
- `ggml/src/ggml.c`: type_traits (name "f8_e4m3", blck 128, size 132, is_quantized, NO to_float/from_float/vec_dot) + `ggml_quantize_chunk` case calling `quantize_f8_e4m3`.
- `ggml/src/ggml-quants.c/.h`: fp8 e4m3fn encode (RNE, saturating to ±448) + decode helpers, `quantize_row_f8_e4m3_ref`, `dequantize_row_f8_e4m3`, `quantize_f8_e4m3`, `ggml_validate_row_data` case (rejects NaN bytes 0x7F/0xFF + non-finite scales).
- `ggml/src/ggml-cpu/ggml-cpu.cpp`: `ggml_backend_cpu_device_supports_op` returns false for any op with an F8_E4M3 operand.
- `tests/test-backend-ops.cpp`: F8_E4M3 added to `all_types`.
- Verified: quantize-fns passes; CPU rejects fp8 mul_mat; all 207 StewFP8 fp8 tensors (3.7B values) decode **bit-exact vs torch.float8_e4m3fn** (script pattern in session: numpy decode formula + `torch` cross-check; dequant rule `w = fp8_to_f32(q) * scale_inv[m/128,k/128]`).

## 4. M2 (GPU kernels) — IN PROGRESS, current state

### Files (uncommitted)
- `ggml/src/ggml-cuda/fp8.cuh` (new): device fp8 encode/decode, `quantize_fp8` kernel, `mul_mat_fp8_wmma` kernel (RDNA4), `mul_mat_fp8_scalar` fallback (CUDA).
- `ggml/src/ggml-cuda/fp8.cu` (new): host launcher `ggml_cuda_mul_mat_fp8` — pre-quantizes activations via pool alloc, launches wmma kernel via `ggml_cuda_kernel_launch`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch branch in `ggml_cuda_mul_mat` (after mmf check) + `supports_op` MUL_MAT case for F8_E4M3 (gated: `GGML_CUDA_CC_IS_RDNA4(cc) || (NVIDIA && cc >= GGML_CUDA_CC_ADA_LOVELACE)`, F32 src1, k%128==0, non-batched) + extern decl.
- `ggml/src/ggml-quants.c/.h` + `ggml.c` also have uncommitted M2 additions (host quantize/dequantize — needed for test data init and llama-quantize later).

### Design (follows llama.cpp launcher patterns — CRITICAL per user advice)
Read `ggml/src/ggml-cuda/mmq.cu` + `mmq.cuh` + `quantize.cu` for the pattern:
1. **Separate pre-quantization kernel** (`quantize_fp8`): F32 activations [k,n] → fp8 staging `[k][n_pad]` (k-major, token-minor, transposed!) + scales `[n][k/128]` f32. One CTA of 128 threads per (token, 128-col block), block-reduce amax → scale = amax/448, inverse for encode. **Verified byte-exact vs host quantization.**
2. **Hot wmma kernel** (`mul_mat_fp8_wmma`): weights = `block_f8_e4m3` rows [m][k/128]; activations from staging. 128 threads (4 wave32 warps), each warp = 16x16 output tile, grid (ceil(n/64), ceil(m/16)). Per 128-col block (cb): 8 wmma k-steps of 16 accumulate into acc_cb (unscaled), then scale-and-accumulate: `acc[s] += wd * ad * acc_cb[s]` where wd = weight block d (per m-row), ad = staging scale (per token). Uses `__launch_bounds__(128,1)` and `ggml_cuda_kernel_launch` (PDL-aware).
3. Launcher pads tokens to multiple of 64 (zeroed) so OOB wmma reads are safe; guards m-boundary via warp-uniform `row_valid`.

### THE gfx12 fp8 wmma fragment layout (DEFINITIVELY VERIFIED on gfx1201, 2026-08-05)
**The original probe-derived layout was WRONG.** The earlier probes used identity B, which is symmetric under transpose and CANNOT distinguish the B fragment's k-major vs n-major layout; and the probe4 matcher reported HOST-array byte positions instead of true fragment positions. The empirical probes on real data (single wmma with random fp8 data, checked 256/256 slots) plus TileLang/composablekernel documentation converge on:

- **A fragment** (weights): lane l byte e = `A[l%16][(l//16)*8 + e]` — lane l holds weight row (l%16); lanes 0-15 carry k=0..7 of each row, lanes 16-31 carry k=8..15.
- **B fragment** (activations): lane l byte e = `B[(l//16)*8 + e][l%16]` — lane l holds the 8 k-values of token (l%16), k-contiguous. Hence staging MUST be `[n][k]` (token-major, k-minor) so each lane's fragment is one contiguous 8-byte load.
- **C fragment** (output store): slot s of lane l = `C[(l//16)*8 + s][l%16]` — lane l owns column (l%16), 8 rows (l//16)*8..+7.

Row/col helpers in fp8.cuh: `fp8_wmma_row(l,s) = (l/16)*8 + s`, `fp8_wmma_col(l,s) = l%16`. Kernel locals: `row_lane = lane%16`, `k_half = (lane/16)*8`. A load: `wblk->qs + kk*16 + k_half` (8 contiguous bytes) of block `(m0+row_lane, cb)`. B load: `src1_q + (n0+row_lane)*k + cb*128 + kk*16 + k_half` (8 contiguous bytes; note the `cb*128` k-offset — a missing cb offset was a second bug).

### Bugs fixed this session (all in fp8.cuh)
1. **B fragment transposed layout**: staging was [k][n]; changed to [n][k] (token-major) + quantize write `y_q[token*k + col0 + tid]`.
2. **A fragment row mapping**: old code used `lane/2` rows / `(lane&1)*8` cols; true mapping is `lane%16` row, k half `(lane/16)*8`.
3. **C store/scale mapping**: old `fp8_wmma_row/col` (odd-lane formula) was wrong for 56/128 odd-lane slots; replaced by the true (l/16)*8+s, l%16.
4. **B fragment cb offset**: kk loop must add `cb*128` to the staging k index.
5. `mul_mat_fp8_scalar` (CUDA fallback) updated for the [n][k] staging (`src1_q[ni*k + cb*128 + c]`).

### Verification status
- Standalone HIP test `/tmp/fp8_correctness.cpp` (k=256 m=64 n=32 random data): **PASS** (0/2048 bad, max rel err ~7e-6).
- Stress test `/tmp/fp8_stress.cpp`: 12 configs, k=128..1024 / m=16..128 / n=16..256, all PASS (max rel err <= 3.4e-3, pure fp8 accumulation-order noise).
- Full HIP build + test-backend-ops rebuild OK. **NOTE: must re-run `cmake .` in the build dir to re-evaluate `file(GLOB "*.cu")` now that fp8.cu exists** (the dir was configured before fp8.cu was added). test-backend-ops: f8_e4m3 MUL_MAT shows "not supported [CPU]" (correct skip; CPU reference can't compute fp8) and runs on ROCm0/1/2; 0 FAILs.
- Probe/verification files in /tmp: `wmma_probe4.cpp` (definitive bijective-A map), `verify_layout.cpp` (real-data 256/256 slot check), `verify_chain.cpp` (full 8-kk chain), `fp8_stress.cpp`, `fp8_correctness.cpp`. Analysis: `convert_probe4.py` (host→true position conversion), `brute*.py`.

### Remaining M2 work
- Perf check (t/s vs f16 baseline, memory ~2.6GB) — needs a runnable model; blocked until M3 (safetensors loader) or a quick synthetic benchmark.
- Optional: CUDA sm_89/sm_90 path is compile-guarded + scalar fallback only (untested on NVIDIA).

## 5. Next steps (in order)

1. ~~Fix the C-fragment mapping~~ **DONE (2026-08-05)**: root cause was the gfx12 fp8 wmma fragment layout itself. `fp8_correctness` + stress suite PASS.
2. ~~Perf check~~ **DONE (2026-08-06)**: 1x R9700: FP8 safetensors 72 t/s generation vs Q8_0 GGUF 91.7 t/s vs BF16 57-71 t/s. Prompt: FP8 139-148 t/s.
3. ~~M3 - direct safetensors loader~~ **DONE (2026-08-06)** for the qwen35 text path + mmproj. See section 9. Remaining minor: mtmd-cli direct run (llama-cli + llama-server + API verified instead), non-128-rows fp8 (guard rejects non-128 k only - rows are fine), E5M2 (rejected).
4. ~~M4 - validation & parity~~ **DONE (2026-08-06)**: greedy parity exact (20 tokens), PPL 9.99 vs 9.89 bf16 (< 1%), row-split parity exact, memory measured (14.3GB peak 1-GPU full-ctx). See section 9 for the critical encode bug found via PPL.
5. **Remaining**: M5 (optional gguf-py fp8 preservation) - not started, low priority. CUDA sm_89/sm_90 mma path still compile-guarded/untested (scalar fallback).

## 6. Key references
- `implementation-plan.md` (repo root) — full plan, M1 marked done, M2 spike results documented.
- `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md` — byte-level format spec (authoritative).
- StewFP8 dir: 945 tensors, 207 fp8+scale_inv pairs, layers-0..31.safetensors + outside.safetensors (299: embeds/norms/vision) + mtp.safetensors (22). All dims multiples of 128.
- llama.cpp patterns: `ggml/src/ggml-cuda/mmq.cu` (launcher), `mmq.cuh`, `quantize.cu` (`quantize_mmq_q8_1_cuda`), `common.cuh` (`ggml_cuda_kernel_launch`, `ggml_cuda_pool_alloc`, arch macros `GGML_CUDA_CC_IS_RDNA4`), `vendors/hip.h` (RDNA4 define).
- Original probe files in /tmp: `wmma_probe*.cpp`, `fp8_correctness.cpp` (may be cleaned; recreate from session history if needed).

## 7. Gotchas
- HIP `__shfl_xor_sync` macro requires a 4th `width` argument (unlike CUDA).
- `__global__` kernel definitions must be visible to the HOST pass (for device-stub generation) — put RDNA4-only types/builtins inside `#if defined(GGML_USE_HIP) && defined(RDNA4)` WITHIN the kernel body, and give the `#else` branch a trivial `(void)` body. Forward declarations alone do NOT generate the stub (link error `__device_stub__`).
- `ext_vector_type` is clang-only (fine for HIP; nvcc CUDA would need different types — CUDA path is compile-verified only, scalar fallback kernel).
- ggml-cuda CMakeLists uses `file(GLOB "*.cu")` — new .cu files are picked up automatically, but only after re-running `cmake .` in the build dir (the GLOB is evaluated at configure time; a build dir configured before fp8.cu existed will silently miss it).
- `RDNA4` macro comes from `vendors/hip.h` when `__GFX12__` defined (device pass only).
- supports_op gets `dev_ctx->device` for the per-device cc — needed to exclude the gfx1036 from fp8.
- The `ggml_cuda_mul_mat` dispatch requires src1 F32 + dst F32 for custom kernels (else cublas → dequant → fails for fp8); supports_op gates this.

## 8. Session state at handoff
- Working tree: M1 changes committed (`d56412c8c`). Uncommitted: fp8.cuh, fp8.cu (new, kernel CORRECTED — fragment layout fixed), ggml-cuda.cu (dispatch+supports_op), ggml-quants.c/.h + ggml.c (host quantize additions).
- `/tmp/llama-hip-build` — configured HIP build (targets: ggml-hip, test-backend-ops); re-ran `cmake .` so fp8.cu is included; both targets build.
- The earlier `wmma_probe3` "hang" was a host-side infinite loop in its byte-assignment `while (used[b])` (all 256 bytes exhausted) — NOT a GPU wedge; the GPUs were never stuck. probe4 replaced probe3 as the definitive mapping probe.
- **Do NOT commit/push without explicit user approval.**

## 9. M3 — direct safetensors loader (DONE 2026-08-06, qwen35 text path)

### Files
- `src/llama-safetensors.h/.cpp` (new): path detection, config.json + index.json + per-file header parsing (nlohmann), mmap via llama_mmap, HF->GGUF tensor mapping with per-tensor transforms, gguf metadata synthesis, tensor data callback.
- `src/llama.cpp`: `llama_model_load_from_file` auto-detects safetensors (dir with config.json + *.safetensors, or a *.safetensors file), builds metadata, runs the D3 hardware gates, calls `llama_model_load_from_file_impl(metadata, set_tensor_data, &loader, ...)` with load_mode NONE.
- `src/llama-model-loader.h/.cpp`: 3 small fixes to the metadata-only (files.empty) path: (a) `get_weight` reports tensors present in the gguf metadata (dummy_weight) so the mtp_only checks work; (b) `create_tensor` returns nullptr for TENSOR_SKIP and for NOT_REQUIRED tensors missing from metadata (matches the file-backed semantics — output/nextn.embed_tokens fall back to duplicated/absent); (c) buft_for_tensor nullptr handled instead of asserted.
- `src/CMakeLists.txt`: added llama-safetensors.cpp + `../vendor` include dir.
- `ggml/src/ggml-cuda/fp8.cu/.cuh`, `ggml-cuda.cu`: see perf fixes in section 4.

### Transforms (all validated byte-exact vs an independent Python reference, 14/14)
- fp8 re-blocking: per output row r and 128-col block cb, write [f32 scale][128 fp8 bytes], scale = bf16(scale_inv[src_r/128][src_cb]) promoted to f32. Row/column V-head reorders applied in the same pass (permutation is block-preserving for head_dim 128).
- V-head reorder (16 K-heads != 32 V-heads, num_v_per_k=2): applied to in_proj_qkv rows, in_proj_z rows, in_proj_a/b rows, A_log/dt_bias elements, conv1d V channels, out_proj columns — and to the fp8 scale tensors in lockstep. Permutations verified against conversion/qwen.py `_reorder_v_heads`.
- `A_log` -> `-exp` (rounded through bf16 when the HF source is bf16; direct f32 for F32 sources), `dt_bias` -> `ssm_dt.bias`, conv1d squeeze [C,1,K]->[K,C], norms `+1` (bf16-rounded) except linear_attn.norm, MTP mapping mtp.* -> blk.32.* (fc->nextn.eh_proj, pre_fc_norm_embedding->enorm, pre_fc_norm_hidden->hnorm, norm->shared_head_norm).
- Output types mirror conversion/base.py: 1D tensors, `*_norm.weight` and ssm_conv1d are F32; 2D .weight are BF16 (or F8_E4M3). CRITICAL: the GPU binary ops (mul) do not support BF16 operands, so norms MUST be F32.
- Source dtype awareness: the ORIGINAL model stores A_log and linear_attn.norm as F32 (mamba_ssm_dtype float32); StewFP8 converted them to BF16. The loader reads the per-tensor dtype from the safetensors header.
- tie_word_embeddings: output.weight is absent; the loader fix makes the model fall back to the duplicated token_embd (same as the GGUF flow).
- Tokenizer: `convert_hf_to_gguf.py <dir> --vocab-only` -> `<dir>/tokenizer.gguf` (already generated for StewFP8 and SafeTensors); the loader copies all `tokenizer.*` KV pairs into the synthesized context; clear one-liner error if missing.

### D3 hardware gate
- `has_fp8_device()`: probes every registered device with a dummy F8 mul_mat op (ggml_backend_dev_supports_op). Fires when the model has fp8 tensors and no fp8-capable device: "FP8_E4M3 weights require a device with native FP8 support (RDNA4, or NVIDIA Ada/Hopper+); this system has none. Use an integer GGUF (e.g. Q8_0) instead."
- Partial offload: fp8 tensors live in every decoder block, so --n-gpu-layers must cover all of them (n_layer_all + 1 = 34): "FP8_E4M3 weights must be fully offloaded to the GPU - increase --n-gpu-layers to at least 34". Verified on CPU-only build and with -ngl 10.
- The natural buffer-selection error ("failed to find a compatible buffer type") is a further backstop.

### Perf fixes discovered while measuring (M2)
- **CRITICAL launcher bug**: the RDNA4 dispatch block in ggml_cuda_mul_mat_fp8 was wrapped in `#if defined(GGML_USE_HIP) && defined(RDNA4)` — RDNA4 is a DEVICE-pass macro, so the host build compiled the whole dispatch out and silently used the scalar fallback (3.2 t/s!). The dispatch is now an unguarded runtime `if (GGML_CUDA_CC_IS_RDNA4(cc))` (kernel bodies keep their #if guard for the device pass).
- **dot4 GEMV path** (`mul_mat_fp8_gemv`): for n <= 16 the wmma kernel wasted the 16-token tiles (batch 1 = 1/64 useful). One warp per output row, 32 lanes read the 128 weight bytes coalesced + dot4 with the shared activation block, warp-reduce, per-block scale. Verified correct (max rel err <= 2.2e-3) and it fixed the batch-1 efficiency.
- **GLU fusion gate**: `ggml_cuda_should_fuse_mul_mat_vec_q` now excludes F8_E4M3 (the fused-mmvq path has no F8 case and aborted at generation).
- Result: 3.2 -> 72.5 t/s generation (vs 91.7 Q8_0 baseline, 59-72 BF16).

### Verification status
- Loader transform validation: 14/14 byte-exact vs Python reference (fp8 re-blocks, V-reorders, +1, -exp, conv1d, F32 types, MTP, token_embd). Also validated the BF16 model's reorders.
- End-to-end: `llama-cli -m <StewFP8 dir>` generates coherent text on 1 GPU (72 t/s) and 3 GPUs (58 t/s); --spec-type draft-mtp works (38 t/s); BF16 SafeTensors dir loads and runs (59-72 t/s).
- Parity: llama.cpp-BF16-via-loader == HF-BF16 greedy, byte-exact over 20 tokens ("Paris.\nA. True\nB. False\nAnswer:\nA\n\n"); llama.cpp-FP8 matches through 9 tokens then flips a True/False near-tie; Q8_0 diverges at token 10. HF-side parity_check.py (BF16 vs FP8-dequant in torch): PASS, byte-identical.
- Negative tests: CPU-only build -> exact D3 error; -ngl 10 -> exact partial-offload error.

### Known gaps / next
- mmproj/vision companion not verified (text loader skips model.visual.*; need convert_hf_to_gguf.py --mmproj + mtmd-cli/server run).
- --vocab-file override not plumbed (loader auto-detects <dir>/tokenizer.gguf only).
- Edge cases untested: single-file no-index safetensors, non-128 dims, E5M2 rejection, corrupt headers.
- CUDA sm_89/sm_90 fp8 mma path still compile-guarded/untested (scalar fallback).
- ~1-ulp expf differences vs numpy for F32-source A_log (numerically irrelevant; the bf16-sourced StewFP8 path rounds through bf16 and matches byte-exact).

## 10. M3 completion + M4 results (DONE 2026-08-06)

### Critical bug found via PPL (the big one)
The fp8 kernel's activation encoder `fp8_e4m3_from_f32` had `if (E >= 15) return 0x7E`
- saturating EVERY value in the top exponent range (256..448, i.e. the top ~43% of
each 128-block's dynamic range) to 448. Activations were systematically inflated to
their block max, distorting the hidden states: logits shifted ~2.0, PPL = 23.8 (2.4x
vs bf16's 9.89), generation diverged at token 10. The self-consistent stress tests
missed it (host reference shared the same buggy encode). Fix in fp8.cuh: saturate only
for E >= 16, or the E=15/man3=7 NaN pattern (man3=7 clamped to 6 = 448). After the
fix: logits track bf16 within 0.1-0.25, **PPL 9.9907 vs 9.8881 (< 1%)**, generation
parity EXACT over 20 tokens. The test harnesses' host encode copies were fixed too.

### New features this session
- **Batched fp8 mul_mat** (needed by llama-server, which batches 4 sequences):
  `ggml_cuda_mul_mat_fp8` loops over ne12*ne13; the supports_op check dropped the
  `b->ne[2]*b->ne[3] != 1` restriction. Verified with a 4-batch correctness test
  (bad=1/16384, max_rel 3.9e-3 - the residual is the ULP-scale tie-flip noise).
- **llama-server works** with the fp8 dir + mmproj: text + image chat via the API
  (image content read correctly; output goes to reasoning_content = Qwen3.5 thinking).
- **--vocab-file override**: `llama_model_params.vocab_file` + common_params_model +
  arg.cpp (env LLAMA_ARG_VOCAB_FILE). Default remains <model-dir>/tokenizer.gguf.
- **mmproj verified**: `convert_hf_to_gguf.py --mmproj` on the ORIGINAL dir
  (`model.safetensors-*.safetensors` naming - the StewFP8 `layers-*` layout is
  invisible to the converter's get_model_part_names) -> 298 tensors, 675MB.
  llama-cli + llama-server + image work end-to-end.

### M4 results
- Greedy parity: llama.cpp fp8 == llama.cpp bf16 for all 20 tokens (after the fix).
- PPL (Pride and Prejudice, 512-token windows): FP8 9.9907, BF16 9.8881, Q8_0 9.8881.
- Row-split parity: default 3-GPU layer split == single GPU, identical output.
- Memory (1x R9700, n_ctx=262144 default): model 4730 MiB + KV 8192 MiB + RS 50 MiB +
  compute 320+266 MiB = ~14.3GB peak. BF16 same setup: ~17.8GB.
- Edge cases verified with synthetic dirs: single-file no-index, E5M2 rejection,
  non-128 k rejection, missing scale_inv, corrupt header, missing tokenizer one-liner.

### Known gaps / next
- mtmd-cli direct binary not run (llama-cli/server + API verified instead).
- CUDA sm_89/sm_90 fp8 mma path compile-guarded, untested (scalar fallback).
- M5 (gguf-py F8_E4M3 constants, --preserve-fp8) not started.


## 11. M5 - GGUF-side FP8 preservation (DONE 2026-08-06)

`--outtype fp8_e4m3` now produces a standard GGUF with native F8_E4M3 tensors
(block_f8_e4m3), so fp8 models work through the regular GGUF path (no safetensors
loader). Summary of the changes:

- gguf-py: `GGMLQuantizationType.F8_E4M3 = 43`, `GGML_QUANT_SIZES` entry
  `(128, 132)`, `LlamaFileType.MOSTLY_F8_E4M3 = 42`; `quantize()` passes the
  already-packed blocks through.
- llama.cpp: `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42` in llama.h (the GGUF metadata
  value; matching integer in gguf-py so the file type round-trips), ftype name +
  type->ftype guess mapping in llama-model-loader.cpp.
- converter: `_generate_fp8_tensors()` reblocks weight + weight_scale_inv into
  `[out, in/128 * 132]` uint8 rows (d = stored scale_inv as f32, 128 fp8 bytes
  verbatim) and writes them NVFP4-style before dequant_model. The qwen35 mixin's
  `transform_fp8_weight` applies the V-head reorder to weight + scale grid in
  128-row/col units. Non-fp8 tensors default to BF16. E5M2 rejected, 2D-only,
  warns when a non-fp8 model is given the fp8_e4m3 outtype. The GUESSED
  heuristic also detects float8_e4m3fn.
- D3 gate extended to the plain GGUF path (llama_model_load): fp8 tensors +
  no fp8-capable device -> the same clear error as the safetensors path.
- The converter's index discovery now falls back to any `*.safetensors.index.json`
  weight map, so nonstandard part naming (StewFP8 `layers-*.safetensors`) converts.
- The safetensors loader writes `general.file_type = 42` so `ftype: F8_E4M3`
  displays there too.

Verification (Qwen3.5-4B, 1x R9700):
- GGUF has 207 F8_E4M3 + 234 F32/BF16 tensors; block bytes byte-identical to the
  source safetensors (d == scale_inv, qs == weight bytes).
- 20-token greedy parity: fp8 GGUF == fp8 safetensors == bf16 (exact).
- PPL: 8.7152 (fp8 GGUF) vs 8.7164 (fp8 safetensors) vs 8.6024 (bf16) vs 8.6126
  (Q8_0) - the two fp8 paths agree to 4 digits, bf16 gap ~1.3%.
- Perf: gen 73-75 t/s, prompt 140 t/s; MTP drafting 101 t/s; llama-server
  (4-seq batched) serves the fp8 GGUF fine (output in reasoning_content, as
  always with Qwen3.5 thinking mode).
- CPU-only build: "FP8_E4M3 weights require a device with native FP8 support
  (RDNA4, or NVIDIA Ada/Hopper+); this system has none. Use an integer GGUF
  (e.g. Q8_0) instead."

## 12. CUDA sm_89/sm_90 fp8 path (DONE 2026-08-06 - documented)

Not implemented. On CUDA the unguarded `mul_mat_fp8_scalar` fallback runs
(correct, slow); the RDNA4 WMMA kernels are AMD-builtin only. An sm_89
mma.m16n8k8 PTX implementation would be unverifiable here (no NVIDIA HW on this
box, and the wmma fragment saga proved how easy these are to get wrong), so it
is documented rather than shipped untested. CUDA Ada/Hopper loads fp8 models
fine and runs them on the scalar kernel; the D3 gate message already reflects
that.


## 13. Prefill/generation performance investigation (DONE 2026-08-06)

### Where we stand vs the goals (1x R9700, Q8_0 GGUF baseline)

| metric | Q8_0 GGUF | fp8 (GGUF = safetensors) | fp8/Q8_0 |
|---|---|---|---|
| pp128 | 3801 t/s | 3257 t/s | 86% |
| pp512 | 5847 t/s | 4837 t/s | 83% |
| pp1024 | 5699 t/s | 4764 t/s | 84% |
| pp2048 | 5512 t/s | 4648 t/s | 84% |
| tg64 | 89.5 t/s | 70.7 t/s | 79% |

Goals were prefill >= +50% over Q8_0 and generation ~ +10%; both are NOT met
(prefill is at ~83-86% of Q8_0, generation at ~79%). The prefill DID improve
2.4x over the original naive kernel (2048 -> 4830 t/s pp512).

### What was measured/found
- Raw fp8 wmma 16x16x16 rate on gfx1201: ~370 TFLOP/s (44G wmma/s) with 4
  independent accumulator chains; a single serial chain saturates at ~182
  TFLOP/s (latency-limited).
- fp8 dot4 (`v_dot4_f32_fp8_fp8`): only 8.7 TFLOP/s - 40x below the wmma, so the
  mmq-style kernel that makes Q8_0 fast (dot4 over smem tiles) is NOT viable for
  fp8; wmma is the only fast path.
- The naive wmma kernel (direct global fragment loads) ran at ~6% of the raw rate
  (pp512 2085 t/s). Progressive fixes:
  1. smem staging of the weight tile + activation block with coalesced loads,
     conflict-free padded layout (sB rows 136 B): 2085 -> 4554 t/s.
  2. 2x2 wmma tiles per warp (each fragment feeds 4 wmma): ~4550 -> ~4950 t/s.
  3. Software-pipelined fragment reads (prefetch kk+1 fragments): small gain.
- Staging is ~23% of the time (a no-staging probe runs 6265 t/s); the wmma
  structure itself caps at ~6265 t/s (~54 TFLOP/s effective, 15% of the raw
  ceiling) - the per-wmma smem reads + issue overhead is the hard wall.
- Double-buffered smem staging regresses (smem 2x -> occupancy 1 CTA/CU).
- Register-file double buffering (prefetch next k-block into regs) also lost:
  bpre[2][8] (64 regs) + acc tiles push past 256 VGPRs -> spills. The best
  correct configuration is the synchronous smem-staged 2x2 kernel (~4830 t/s).
- A transient bug during the prefetch experiments (A staging only for block 0)
  passed the short-parity test but produced garbage PPL (2.1M) - caught by the
  PPL check; the wmma correctness tests only covered k <= 128 (1 block) and
  missed it. Lesson: fp8_correctness should test multi-block k too.

### Generation
- tg is memory-bound; the fp8 model loads BIGGER than the Q8_0 model (4.75 GiB
  vs 4.28 GiB) because the non-fp8 tensors (token_embd, norms, conv1d,
  in_proj_a/b, A_log) stay BF16 in both the safetensors dir and the
  `--outtype fp8_e4m3` GGUF. The Q8_0 GGUF quantizes everything. So fp8 moves
  more bytes per token at generation; the fp8 GEMV achieves ~337 GB/s vs the
  Q8_0 mmvq's ~381 GB/s.

### Remaining levers (not done)
- Quantize the non-fp8 tensors (e.g. Q8_0) when building the fp8 GGUF -> smaller
  model -> faster memory-bound generation (and prefill staging traffic).
- Deeper wmma work (register-buffered staging that stays under 256 VGPRs, or a
  fused quantize+matmul) - attempted, did not beat the synchronous version.
- Larger pp128 variance (short runs) makes small-prompt comparisons noisy.
