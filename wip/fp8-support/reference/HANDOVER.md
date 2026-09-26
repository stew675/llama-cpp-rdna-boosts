# HANDOVER — FP8 E4M3 + Direct Safetensors Loading for llama.cpp

**Author**: AI-assisted session (pi/Claude) working with the repo owner
**Date**: end of session 2026-08-06
**Repo**: `stew675/llama.cpp` (personal fork), branch `cllm`
**Status**: M1-M5 complete and committed; prefill kernel optimization in progress (uncommitted); performance goals from the user NOT fully met (details in section 8)

---

## 1. What this project is

Extending the `stew675/llama.cpp` fork (branch `cllm`) to:

1. **Run FP8_E4M3 weights natively** (`GGML_TYPE_F8_E4M3`, 128-block, f32 scale), hardware-gated to native-FP8 devices only (RDNA4 gfx1201, CUDA sm_89+). NO CPU kernels, NO software emulation. Users without native FP8 hardware get a clear error and should use an integer GGUF (e.g. Q8_0).
2. **Load vanilla HuggingFace safetensors directly** (no GGUF conversion) via the existing `llama_model_init_from_user()` API + a synthesized in-memory gguf_context.
3. **Write native-FP8 GGUFs** via `--outtype fp8_e4m3` (M5), so fp8 models also work through the standard GGUF path.
4. Target model: `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B, multimodal, FP8 E4M3, official Qwen 128x128-block convention). Format spec: `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md`.

**This is a personal fork; nothing goes upstream.** Follow AGENTS.md (ASCII only, concise comments, no AI-sounding text, contributor must understand every line).

The full original plan: `implementation-plan.md` (all items checked off as of 2026-08-06).

---

## 2. Current status dashboard (2026-08-06)

| Item | Status | Where |
|---|---|---|
| M1 fp8 type registration | DONE + committed `d56412c8c` | ggml.h, ggml-common.h, ggml.c, ggml-quants.c/h, ggml-cpu.cpp, test-backend-ops |
| M2 fp8 kernels (correctness + perf) | DONE + committed (kernel fix, GEMV, GLU gate) | ggml/src/ggml-cuda/fp8.cuh/.cu, ggml-cuda.cu |
| M2 prefill kernel optimization | **DONE + committed this session (aiter port, +12.5%)** | fp8.cuh/fp8.cu (PERF_HANDOVER.md) |
| M3 direct safetensors loader | DONE + committed `748d52e0f` | src/llama-safetensors.h/.cpp, llama.cpp, llama-model-loader.cpp/h, llama-model.cpp |
| M4 validation (PPL, parity, multi-GPU, memory) | DONE + committed `b678f8a4d` | — |
| M5 `--outtype fp8_e4m3` GGUF | DONE + committed `533abf443` | convert_hf_to_gguf.py, conversion/base.py, conversion/qwen.py, gguf-py, include/llama.h, llama-model-loader.cpp, llama-safetensors.cpp, llama.cpp |
| CUDA sm_89/sm_90 mma path | Documented as supported-but-scalar (deliberately NOT implemented) | handoff.md section 12 |
| User's perf goals (prefill +50%, gen +10% vs Q8_0) | **NOT MET** (prefill ~83-86% of Q8_0, gen ~79%) | section 8 |

**Git history (all pushed to `origin/cllm`):**
```
533abf443 convert : --outtype fp8_e4m3 writes native F8_E4M3 GGUFs (M5)
b678f8a4d fp8 : fix activation encoder saturation + M3/M4 completion
748d52e0f llama : direct safetensors loading (M3) + fp8 perf fixes
5f566c315 Phase 2 of FP8 project
d56412c8c ggml : add FP8 E4M3 type registration (M1)
6ea215d17 (upstream base)
```

**Uncommitted working-tree changes:** none (kernel + docs committed this session;
see PERF_HANDOVER.md for the perf tracking doc).

---

## 3. Environment (all verified)

- **CPU**: AMD Ryzen 9 9950X3D (16 cores, Zen 5).
- **GPUs**: 3x AMD Radeon AI PRO R9700, **gfx1201** (RDNA4), wave32, 32GB VRAM each (total ~95GB), + 1x gfx1036 (RDNA2, no FP8). `rocm_agent_enumerator` = `gfx1201 gfx1201 gfx1201 gfx1036`.
- **ROCm**: system ROCm 7.1.1 (Fedora 44); hipcc/clang at `/usr/lib64/rocm/llvm/bin`; hipblas/rocblas as system packages (`/usr/lib64/cmake/hipblas`). Also `/opt/rocm-7.14` (ROCm 7.15, clang 23) — both emit native fp8 wmma for gfx1201. Build uses the system one.
- **Build script**: `/home/stew675/bin/build-llama-rocm` — `-DGGML_HIP=ON -DGPU_TARGETS="gfx1200;gfx1201" -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=1 -DGGML_CUDA_NO_PEER_COPY=1 -DGGML_RPC=1`, clang/clang++.
- `/home/stew675/rocm-libraries/` — ROCm math libs source — NOT needed (system packages suffice).

### Build dirs & binaries
- **HIP full build**: `/tmp/llama-hip-full` — llama-cli, llama-server, llama-perplexity, llama-bench (rebuilt after every kernel change with `cmake --build . -j16 --target <tgt>`).
- **CPU-only build** (for D3 gate testing): `/tmp/llama-cpu-build` — `cmake --build . -j16 --target llama-cli`.
- Previous ggml-only dir: `/tmp/llama-hip-build` (mostly superseded).

### Model dirs
- `/llm/models/Qwen3.5/4B/StewFP8/` — **fp8 model** (source of truth for the loader). `layers-*.safetensors` naming (NONSTANDARD), `model.safetensors.index.json`, `mtp.safetensors`, `outside.safetensors`, `tokenizer.gguf`. 945 tensors: 207 F8_E4M3 + 234 BF16/F32 (+ vision). fp8 weights are 2D `[rows, cols]` with 2D `weight_scale_inv [rows/128, cols/128]` BF16 (per-128x128-block scale, 128 rows share one scale).
- `/home/llm/models/Qwen3.5/4B/SafeTensors/` — **bf16 reference** (2 shards, standard `model.safetensors-*.safetensors` naming; A_log + linear_attn.norm are F32 here, BF16 in StewFP8).
- `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` — Q8_0 perf baseline.
- `/tmp/stewfp8-preserved.gguf` — fp8 GGUF produced by `--outtype fp8_e4m3` (5.1GB, 441 tensors: 207 F8_E4M3 + 234 F32/BF16).

### Test artifacts (all in /tmp)
- `/tmp/fp8_correctness` (+ `.cpp`): standalone wmma correctness vs host fp8 (PASS, max rel err 6.6e-6). **NOTE: only tests k <= 128 (1 block) — see section 9 gap.**
- `/tmp/fp8_stress` (+ `.cpp`): 12-config stress (PASS, max rel err 3.4e-3).
- `/tmp/fp8_gemv_test`: GEMV path (PASS, 1.52e-4).
- `/tmp/fp8_batch_test`: batched (ne12=4) mul_mat (bad=1/16384, the residual is a ULP tie-flip from the division roundtrip).
- `/tmp/st_loader_test.cpp` + `/tmp/validate_loader.py`: loader transforms 14/14 byte-exact.
- `/tmp/parity_llama.cpp` + `/tmp/parity_llama`: C-API token comparison (prompt + N tokens). **Must be REBUILT against the current libllama whenever llama.h/struct layouts change** (`g++ -std=c++17 -O2 parity_llama.cpp -o parity_llama -I /home/stew675/cllm/common -I /home/stew675/cllm/include -I /home/stew675/cllm/src -I /home/stew675/cllm/ggml/include -L /tmp/llama-hip-full/bin -lllama -lllama-common -lggml -lggml-base -lggml-hip -lggml-cpu -lggml-rpc -lhipblas -lrocblas -pthread -Wl,-rpath,/tmp/llama-hip-full/bin`). A stale binary segfaults in `llama_hparams::n_head` (struct mismatch) — rebuild on any header change.
- `/tmp/wmma_rate.cpp`, `/tmp/wmma_rate2.cpp`: raw wmma throughput microbenchmarks (see section 8).
- `/tmp/dot4_rate.cpp`: fp8 dot4 throughput (8.7 TFLOP/s — the reason the mmq approach is a dead end).
- `/tmp/corpus_pride.txt`: PPL corpus (Pride and Prejudice).
- `/tmp/pp_bench.sh`, `/tmp/sample_vram.sh`, `/tmp/sample_verbose2.sh`: helper scripts.

---

## 4. Code map (what changed, file by file)

### Committed (M1)
- `ggml/include/ggml.h`: `GGML_TYPE_F8_E4M3 = 43`, `GGML_TYPE_COUNT = 44`.
- `ggml/src/ggml-common.h`: `QK_F8_E4M3 128`, `block_f8_e4m3 { float d; uint8_t qs[128]; }` (132B, static_assert).
- `ggml/src/ggml.c`: type_traits (name "f8_e4m3", blck 128, size 132, is_quantized, NO to_float/from_float/vec_dot); `ggml_quantize_chunk` case.
- `ggml/src/ggml-quants.c/.h`: fp8 e4m3fn encode (RNE, saturating to ±448) + decode, `quantize_row_f8_e4m3_ref`, `dequantize_row_f8_e4m3`, `quantize_f8_e4m3`, `ggml_validate_row_data` (rejects NaN bytes 0x7F/0xFF + non-finite scales).
- `ggml/src/ggml-cpu/ggml-cpu.cpp`: CPU rejects any op with an F8_E4M3 operand.
- `tests/test-backend-ops.cpp`: F8_E4M3 in `all_types`.

### Committed (M2 kernels)
- `ggml/src/ggml-cuda/fp8.cuh` (new): device fp8 encode/decode; `quantize_fp8` kernel (F32 [k,n] -> fp8 staging [n][k] token-major + scales [n][k/128]); `mul_mat_fp8_wmma` (RDNA4); `mul_mat_fp8_gemv` (dot4, n <= 16); `mul_mat_fp8_scalar` (CUDA fallback).
- `ggml/src/ggml-cuda/fp8.cu` (new): host launcher `ggml_cuda_mul_mat_fp8` — pools staging buffers, pads n to multiple of 64, quantizes, dispatches per-cc (RDNA4 -> wmma/gemv, else scalar).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: MUL_MAT dispatch branch + `supports_op` F8 case (gated: RDNA4 or NVIDIA Ada+; F32 src1; k%128==0; `a->ne[2]*a->ne[3]==1` — weights unbatched; src1 may be batched since the 2026-08-06 server fix); `ggml_cuda_should_fuse_mul_mat_vec_q` excludes F8_E4M3.

### Committed (M3 loader)
- `src/llama-safetensors.h/.cpp` (new): parses config.json + safetensors headers/index, mmaps via llama_mmap, builds a synthetic gguf_context (metadata + tensor list), `set_tensor_data` callback applies per-tensor transforms (fp8 re-block, V-head reorder, +1 norms, -exp A_log, conv1d squeeze, MTP mapping), `has_fp8_device()` probe (dummy fp8 mul_mat), `has_fp8_tensors()`, `n_layer_all()`, `set_vocab_path()`.
- `src/llama.cpp`: `llama_model_load_from_file` safetensors branch (D3 gate: fp8 device + full offload >= n_layer_all+1 = 34); **also D3 gate on the plain GGUF path** (`llama_model_load`: scan gguf tensor types for F8_E4M3 -> `llama_safetensors_loader::has_fp8_device()` else clear error).
- `src/llama-model-loader.cpp/.h`: empty-path `create_tensor` semantics (TENSOR_SKIP -> null, missing + TENSOR_NOT_REQUIRED -> null, buft null -> null); `get_weight` metadata fallback (mtp_only detection); ftype name "F8_E4M3" + type->ftype guess mapping; `general.file_type` read.
- `src/llama-model.cpp`: qwen35 tensor loading with fp8 handling.
- `src/llama-context.cpp/.h`: (minor, context-side fp8 handling).

### Committed (M4/M5)
- `include/llama.h`: `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42`; `llama_model_params.vocab_file` (--vocab-file).
- `common/common.h/.cpp`, `common/arg.cpp`: `--vocab-file`, `--spec-type draft-mtp` plumbing.
- `gguf-py/gguf/constants.py`: `GGMLQuantizationType.F8_E4M3 = 43`, `GGML_QUANT_SIZES[F8_E4M3] = (128, 132)`, `LlamaFileType.MOSTLY_F8_E4M3 = 42` (matches llama.h).
- `gguf-py/gguf/quants.py`: F8_E4M3 passthrough in `quantize()`.
- `conversion/base.py`: `_generate_fp8_tensors()` (reblocks weight + weight_scale_inv into block_f8_e4m3 rows, NVFP4-style, before dequant_model, when `ftype == MOSTLY_F8_E4M3`); `transform_fp8_weight` hook; E5M2 rejection; 2D-only; no-fp8-found warning; GUESSED heuristic detects float8_e4m3fn; **index discovery fallback** (any `*.safetensors.index.json` — fixes the `layers-*` naming blind spot).
- `conversion/qwen.py`: `transform_fp8_weight` V-head reorder with the scale grid in 128-row/col units (`row_perm[::128] // 128`).
- `convert_hf_to_gguf.py`: `--outtype fp8_e4m3` choice; `--fp8-as-q8`/`--preserve-fp8` mutually-exclusive validation (preserve flag removed in favor of the outtype).
- `src/llama-safetensors.cpp`: writes `general.file_type = 42` for fp8 models.

---

## 5. Architecture & design decisions (read before touching anything)

### The fp8 type
- Weights are `block_f8_e4m3`: `[f32 d][128 fp8 bytes]`, 132 bytes/block, k-blocks per row.
- **Scale semantics**: dequant = `fp8_to_f32(qs) * d`, where for the HF model `d` = the stored `weight_scale_inv` value (NOT inverted — the HF convention here is `weight * scale_inv`; verified against `dequant_simple` in conversion/base.py and the kernel math).
- fp8 e4m3fn encode: RNE, saturate to ±448, NaN patterns 0x7F/0xFF handled, 480 (E=15/man3=7) clamped to 448. **THE E>=15 SATURATION BUG** (fixed in `b678f8a4d`): original code saturated ALL E>=15 values (the 256-448 range) to 448 — the top ~43% of every block's dynamic range — inflating activations (PPL 23.8, logit shift ~2.0, divergence at token 10). Fix: saturate only E>=16 (or E=15/man3=7 -> 6). The host test harnesses share the encode; update them together.

### The M3 loader (safetensors direct load)
- `llama_model_load_from_file` detects a safetensors path (dir with `model.safetensors.index.json` or similar), builds an in-memory gguf_context (`gguf_init_empty` + keys + `gguf_add_tensor`), and calls `llama_model_load_from_file_impl` with `set_tensor_data` = `llama_safetensors_loader::set_tensor_data` (load_mode NONE). The callback fills tensor data from the mmaps with transforms.
- **Transforms** (dtype-aware): TF_NONE, TF_FP8 (re-block + perms), TF_NORM_P1 (+1), TF_A_LOG (-exp + v_perm_1), TF_REORDER (row/col perms). `st_mapping.src_f32` flag: F32-source tensors compute directly in F32; BF16 sources round through bf16 (byte-exact with the converter).
- **Hardware gate (D3)** at load: fp8 tensors present -> require `has_fp8_device()` (dummy F8 mul_mat probe) + `n_gpu_layers >= n_layer_all + 1` (34). The natural buffer-selection error is the backstop. The same device check now runs on the plain GGUF path.
- Output.weight intentionally NOT in metadata -> token_embd duplication (matches GGUF flow).
- Vision tensors (`model.visual.*`) skipped in mapping (multimodal handled by the existing mmproj.gguf + mtmd flow).

### The M5 GGUF path
- `--outtype fp8_e4m3` -> `_generate_fp8_tensors()` reblocks `weight` + `weight_scale_inv` pairs into `[out, in/128 * 132]` uint8 rows and writes them directly to the gguf_writer (NVFP4-style, BEFORE dequant_model). Non-fp8 tensors default to BF16. `d` = stored scale_inv as f32 LE; qs = the 128 fp8 bytes verbatim (byte-identical to the safetensors loader output — verified).
- The GGUF metadata `general.file_type = 42` round-trips (gguf-py == llama.h) and displays as `ftype: F8_E4M3`.
- qwen35 V-head reorder applies to fp8 weights too, moving rows/cols in units of 128 so the scale grid stays aligned.

### The M2 kernels (current state)
- **Launcher** (`ggml_cuda_mul_mat_fp8`): asserts src0 F8_E4M3, src1 F32 contiguous, weights unbatched; loops over `ne12*ne13` (batched src1, fixed 2026-08-06 for llama-server); pads n to a multiple of 64 (zeroed); quantizes activations once per batch into staging `[n][k]` (token-major/k-minor) + scales `[n][k/128]`; dispatches on runtime cc: RDNA4 -> wmma (n > 16) or dot4 GEMV (n <= 16); else scalar fallback (CUDA sm_89+).
- **wmma kernel** (current uncommitted version): 128 threads (4 wave32 warps), CTA = 32 weight rows x 128 tokens, each warp computes a 2x2 grid of 16x16 wmma tiles (32 rows x 32 tokens per warp). sA (32 x 132 B: f32 scale + 128 fp8 per row) + sB (128 tokens x 136 B padded for bank-conflict-free 8-byte fragment reads) staged in smem with coalesced loads once per 128-k block; per-kk: 4 fragment LDS reads feed 4 wmma; per-block: 8 kk steps accumulate into per-tile temps, then `acc[s] += wd * ad * t[s]` (wd from sA[..][0:4], ad from src1_s). Software-pipelined fragment loads (kk+1 prefetched).
- **GEMV kernel** (n <= 16, generation): one warp per output row; 32 lanes read 128 fp8 weight bytes coalesced (4 B each), `__builtin_amdgcn_dot4_f32_fp8_fp8` + shfl reduce; avoids the wmma 16-token tile waste at batch 1.

---

## 6. Hard-won technical findings (DO NOT rediscover these)

1. **The gfx12 fp8 wmma fragment layout** (empirically verified on gfx1201 + TileLang/composablekernel docs):
   - A: lane l byte e = `A[l%16][(l//16)*8 + e]` (weight row l%16)
   - B: lane l byte e = `B[(l//16)*8 + e][l%16]`
   - C: slot s = `C[(l//16)*8 + s][l%16]`
   - The earlier probe-derived layout was WRONG because identity-B is symmetric under transpose. Builtins (wave32): `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` with A/B as `v2i32` (8 fp8/lane) and acc as `v8f32`.
2. **Raw wmma throughput on gfx1201**: ~370 TFLOP/s (44G wmma/s) with 4 independent acc chains; a single serial chain saturates at ~182 TFLOP/s (latency-limited). fp16 wmma ~129-150 TFLOP/s.
3. **fp8 dot4 (`v_dot4_f32_fp8_fp8`) is only ~8.7 TFLOP/s** — 40x below wmma. The mmq-style kernel that makes Q8_0 fast (dot4 over smem tiles, ~51 TFLOP/s effective) is NOT viable for fp8. wmma is the only fast path.
4. **The host-side RDNA4 dispatch** must be an unguarded runtime `if (GGML_CUDA_CC_IS_RDNA4(cc))` — `#if defined(RDNA4)` is device-only and silently compiled out the launcher path (3.2 t/s scalar fallback bug).
5. **The fp8_e4m3_from_f32 E>=15 saturation bug** (section 5) — the single biggest correctness fix; PPL 23.8 -> 9.99, exact 20-token parity.
6. **Batched mul_mat** needed for llama-server (n_seq_max=4 -> 3D activations); the fp8 kernels only handled batch 1 -> GGML_ASSERT in ggml-backend.cpp:1283.
7. **The wmma kernel optimization journey** (pp512): naive global-fragment loads 2085 t/s -> smem staging 4554 -> 2x2 tiles/warp + pipelining ~4950 -> final ~4830 (synchronous staging is the best CORRECT config; see section 8 for the experiments that failed).
8. **Staging bandwidth/latency is ~23%** of the wmma kernel time (no-staging probe = 6265 t/s); the wmma structure itself caps at ~6265 t/s (15% of the raw 370 TFLOP/s ceiling) due to per-wmma smem-read + issue overhead. Register file (256 VGPRs) walls double-buffering: `bpre[2][8]` (64 regs) + acc tiles spill. An array-indexed `bpre[buf]` also defeats aliasing analysis.
9. **A transient prefetch bug** (A staging only for block 0) passed short-parity but produced PPL 2.1M — caught only by the PPL check. The standalone correctness tests only cover k <= 128 (single block). **Gap: fp8_correctness should test multi-block k.**
10. **Alignment matters in weird ways**: splitting sA into separate d/qs arrays (8-byte-aligned fragment reads) measured SLOWER than the unaligned single-array layout (misaligned uint4 global staging loads were likely the cause). The best config uses the unaligned single sA array with 4-byte staging loads.
11. **llama-cli in this fork runs a server-style flow** — use `--single-turn --no-conversation` to exit after one prompt; logs contain binary NULs (use `tr -d '\0'`); MTP via `--spec-type draft-mtp` (NOT `--mtp`); single GPU via `-dev ROCm0`.
12. **`parity_llama` must be rebuilt after any llama.h struct change** (stale binary -> segfault in `llama_hparams::n_head`).
13. **The converter's `get_model_part_names` requires the `model` prefix** — the StewFP8 `layers-*.safetensors` naming is invisible to it; the index-file fallback (any `*.safetensors.index.json`) fixes conversion of such dirs.
14. **Profiling**: `rocprofv3 -r -d <dir> -f csv -- <app>` for kernel traces (kernel name, duration, VGPR, LDS); `--pmc SQ_INSTS_VALU ...` for counters (only SQ_BUSY_CYCLES + SIMD_UTILIZATION collected reliably on gfx1201). `rocprofv3-avail list --pmc` lists counters.

---

## 7. Verification suite (how to prove it still works)

```bash
# correctness (rebuild against current libs when kernels change)
cd /tmp && g++ -std=c++17 -O2 fp8_correctness.cpp -o fp8_correctness \
  -I /home/stew675/cllm/ggml/include -I /home/stew675/cllm/ggml/src \
  -L /tmp/llama-hip-full/bin -lggml -lggml-base -lggml-hip -lhipblas -lrocblas \
  -pthread -Wl,-rpath,/tmp/llama-hip-full/bin && ./fp8_correctness   # PASS
./fp8_gemv_test   # GEMV PASS
./fp8_batch_test  # batched: bad=1/16384 (ULP tie-flip), max_rel 3.9e-3
python3 /tmp/validate_loader.py   # 14/14 byte-exact

# end-to-end parity (rebuild parity_llama on header changes)
/tmp/parity_llama /tmp/stewfp8-preserved.gguf "The capital of France is" 20
/tmp/parity_llama /llm/models/Qwen3.5/4B/StewFP8/ "The capital of France is" 20
# expected: 11751 13 198 32 13 2912 198 33 13 3439 198 15666 25 198 32 271 22365 314 279 2614

# PPL (the reliability check — catches silent kernel bugs)
cd /tmp/llama-hip-full && ./bin/llama-perplexity -m /tmp/stewfp8-preserved.gguf \
  -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"
# expected ~8.715 (fp8) vs 8.602 (bf16) vs 8.61 (Q8_0)

# perf
./bin/llama-bench -m <model> -p 512 -n 64 -r 2 -dev ROCm0

# D3 gate on CPU-only build
cd /tmp/llama-cpu-build && ./bin/llama-cli -m /tmp/stewfp8-preserved.gguf -p hi -n 2 \
  --single-turn --no-conversation 2>&1 | tr -d '\0' | grep "FP8_E4M3 weights require"
```

**Current measured numbers (1x R9700, `-dev ROCm0`):**

| metric | Q8_0 GGUF | fp8 GGUF | fp8 safetensors | bf16 safetensors |
|---|---|---|---|---|
| pp128 | 3801 | 3257 | ~3050-3760 (noisy) | — |
| pp512 | 5847 | 4837 | 4827 | 1086 |
| pp1024 | 5699 | 4764 | 4757 | — |
| pp2048 | 5512 | 4648 | 4634 | — |
| tg64 | 89.5 | 70.7 | 70.6 | 57.1 |

PPL: fp8 GGUF 8.7152 == fp8 safetensors 8.7164, bf16 8.6024, Q8_0 8.6126.
20-token greedy parity: fp8 == bf16 exactly (both paths).
Memory (1 GPU, n_ctx=262144): fp8 peak ~14.3GB (model 4730 MiB + KV 8192 + RS 50 + compute 320+266); bf16 ~17.8GB.

---

## 8. The performance investigation (why the goals aren't met, and what's left)

The user's goals: **prefill >= +50% over Q8_0, generation ~ +10%**. Current reality: prefill ~83-86% of Q8_0 (but 2.4x better than the original naive kernel), generation ~79%.

### Findings
- The fp8 wmma hardware is capable of ~370 TFLOP/s — 6x the Q8_0 mmq kernel's effective rate — so the +50% prefill target is *theoretically* reachable.
- The kernel structure caps at ~6265 t/s even with staging removed (15% of raw ceiling): per-wmma smem fragment reads + issue overhead. Staging costs another ~23%.
- Attempted and REJECTED (each measured worse than the synchronous 2x2 version ~4830 t/s):
  - Double-buffered smem staging (smem 2x -> 1 CTA/CU occupancy -> 3633)
  - 8-warp CTA (64 rows x 128 tokens): 4054
  - Register-file B prefetch (bpre[buf] aliasing + register pressure): 3978-3548
  - Explicit ping-pong register prefetch (separate variables): 4535
  - Aligned-A split arrays: ~3950
  - MT=4 tiles (register-limited, incomplete implementation)
- Generation is memory-bound; the fp8 model loads **bigger** than Q8_0 (4.75 vs 4.28 GiB) because the non-fp8 tensors (token_embd, norms, conv1d, in_proj_a/b, A_log) stay BF16 in both the safetensors dir and the `--outtype fp8_e4m3` GGUF. The Q8_0 GGUF quantizes everything. fp8 GEMV achieves ~337 GB/s vs Q8_0 mmvq ~381 GB/s.

### Remaining levers (not tried / not done)
1. **Quantize the non-fp8 tensors (e.g. Q8_0) in the fp8 GGUF** -> smaller model -> faster memory-bound generation AND less prefill staging traffic. This is the most promising single lever and is a converter change, not a kernel change. (The safetensors dir's BF16 tensors are fixed by the model format.)
2. Deeper wmma work: a register-buffered staging scheme that stays under 256 VGPRs; fused quantize+matmul to avoid the staging round-trip; larger k per staging round.
3. Test-suite gap: fp8_correctness only tests k <= 128 (single block) — add a multi-block k case (the transient prefetch bug would have been caught).
4. pp128 variance is high (short runs, ±200-400) — measure with -r 5+ before drawing conclusions at small n.
5. CUDA sm_89/sm_90 mma path: deliberately NOT implemented (cannot verify fragment layout without NVIDIA hardware; scalar fallback is correct but slow). The D3 gate already accepts Ada/Hopper with the scalar path.

---

## 9. Known gaps & open questions

- CUDA sm_89/sm_90: scalar fallback only (documented, accepted).
- mmproj/vision: verified via llama-cli/server + mmproj.gguf + image (reads actual image content). mtmd-cli binary itself not run directly.
- E5M2, non-128 dims, corrupt headers, missing tokenizer: rejected with clear messages (edge-case dirs verified).
- `--vocab-file` override: plumbed and tested (default `<model-dir>/tokenizer.gguf`).
- The fp8 GGUF's non-fp8 tensors are BF16 (see section 8 lever 1).
- MTP works via `--spec-type draft-mtp` (both safetensors and GGUF paths; ~101 t/s with drafting on the GGUF).
- The wmma kernel's `__launch_bounds__(128, 1)` — the `1` may be a hint ceiling; not investigated.

---

## 10. Session gotchas (from experience)

- **Build**: `cmake --build /tmp/llama-hip-full -j16 --target llama-bench` (or llama-cli/server/perplexity). GLOB picks up new .cu files only after re-running `cmake` configure.
- **CLI**: `--single-turn --no-conversation`; logs have NUL bytes — `tr -d '\0' | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g'` before grepping.
- **-lv N**: verbosity threshold (INFO is 3; `-lv 4` shows buffer sizes). `-lv 2` HIDES INFO.
- **Single GPU**: `-dev ROCm0`; rocm-smi GPU indices may NOT match llama.cpp device order (llama.cpp ROCm0 = rocm-smi GPU[1] on this box).
- **VRAM sampling**: generation finishes in seconds — use a long `-n` (e.g. 3000) or llama-server for stable samples.
- **Perf measurements**: use `llama-bench` with `-r 2+`; pp128 needs `-r 5` due to variance.
- **rocprofv3**: `-r` for kernel traces; `--pmc` for counters; output in `<dir>/soar/*.csv`.
- **The `--outtype fp8_e4m3` GGUF is the fast path for testing** — it equals the safetensors path byte-for-byte behaviorally but is a single file.
- **Rebuild parity_llama after any change to llama.h** (struct layout).

---

## 11. Commands cheat sheet

```bash
# build
cmake --build /tmp/llama-hip-full -j16 --target llama-bench llama-cli llama-server llama-perplexity
cmake --build /tmp/llama-cpu-build -j16 --target llama-cli

# convert fp8 GGUF (StewFP8 dir, nonstandard naming works now)
python3 convert_hf_to_gguf.py /llm/models/Qwen3.5/4B/StewFP8/ --outfile /tmp/stewfp8-preserved.gguf --outtype fp8_e4m3

# run
/tmp/llama-hip-full/bin/llama-cli -m /tmp/stewfp8-preserved.gguf -p "..." -n 20 \
  --single-turn --no-conversation -dev ROCm0
/tmp/llama-hip-full/bin/llama-bench -m <model> -p 512 -n 64 -r 2 -dev ROCm0

# profile
rocprofv3 -r -d /tmp/rocp -f csv -- /tmp/llama-hip-full/bin/llama-bench -m /tmp/stewfp8-preserved.gguf -p 512 -n 1 -r 1 -dev ROCm0
```

---

## 12. PENDING — user information for the next session

**The user (repo owner) has additional information to provide at the start of the next session.** This section is intentionally left as a placeholder. Likely topics based on this session's open threads: the performance goals (prefill +50%, generation ~10%), the fp8 model format (the BF16 non-fp8 tensors, whether they can be quantized), CUDA expectations, or new requirements.

**Before the next session, the assistant should:**
1. Ask the user for their information FIRST (do not start new work until it's provided).
2. Confirm the working-tree state (section 2) matches reality.
3. Decide with the user whether to commit the current uncommitted kernel work (fp8.cuh/fp8.cu/handoff.md) before starting anything new.

## 13. UPDATE (2026-08-06 afternoon) — aiter inspection done

> Superseded by section 14 (port completed this session).
- The uncommitted kernel work from section 2 was actually committed as `821d423ac` ("Performance investigations") — working tree is clean except HANDOVER.md. Baseline re-measured on build 821d423ac: pp512 4929.78 t/s, tg64 71.37 t/s (fp8 GGUF, -dev ROCm0).
- Deep inspection of `/home/stew675/aiter` (ROCm/aiter, main @ 22beb1caa) completed and written up in **AITER_FINDINGS.md** (read it before touching the kernel). Summary: aiter's `gemm_a8w8_blockscale` Triton kernel is the exact op we implement, tuned for gfx1201 by AMD (PR #3228). Measured on this box at M=512 on our shapes: 89-137 TFLOP/s vs our wmma ~55 TFLOP/s (2.2x gap; rocprof: wmma = 61% of pp512, delta-net = 17%, bf16 mmv = 12.5%). The liftable recipe is a kernel-structure port (128x64 CTA, 8 warps, 2-stage smem pipeline, grouped-M swizzle) — details + projections in AITER_FINDINGS.md section 5. Also found: (16,16) preshuffle is 5-8x SLOWER on gfx1201 (negative result, do not adopt); aiter has a gfx1201 gated-delta-rule kernel (second prefill lever); latent n_pad OOB read in fp8.cu for 17<=n<=64.
- Repro env: py3.12 venv `/tmp/aiter-venv` (ROCm torch 2.12.0+rocm7.1, triton 3.7.0), bench scripts `/tmp/bench_aiter_*.py`, jax-stub import trick (details in AITER_FINDINGS.md section 8).

## 14. UPDATE (2026-08-06 evening) — aiter GEMM port done, +12.5%

**PERF_HANDOVER.md is the dedicated next-session tracking doc for performance**
(scoreboard, ranked levers, kernel facts, failed experiments, repro). Summary:

- Kernel now: CTA 128 weight rows x 64 tokens, 8 warps (4m x 2n, 2x2 wmma
  tiles/warp), grouped-M pid swizzle (GROUP_SIZE_M=4), single-buffered smem
  (26.4 KB) so 2 CTAs fit per CU, register-staged k-block pipelining.

Measured (pp512, -r 3, same build/GPU): **5546-5596 t/s vs baseline 4929.8
(+12.5-13.5%)**. tg64 unchanged (71.0) — decode untouched. GEMM kernel rate
55 -> ~77 TFLOP/s (rocprof: wmma 113.9 -> 80.6 ms/run). All 1186 MUL_MAT
backend-ops tests pass; generation matches the BF16 reference model.

Key findings this session (details + numbers in AITER_FINDINGS.md section 10):
- The real lever was OCCUPANCY, not the double buffer: gfx1201 fits 3 CTAs/CU
  by registers but only 1 by LDS when the tile is double-buffered (61 KB);
  aiter's 24 KB single-buffered tile runs 2 CTAs/CU (16 warps) — that is where
  its 121 TFLOP/s comes from. Single buffer + register staging gets the same
  occupancy with loads still overlapped.
- Failed experiments: 16 warps/CTA (2x1 tiles) regressed; GROUP_M=8 noise.
- LLVM miscompile found: the fa[4][4] burst-fragment array produces v[0:1]
  operands for half the wmma chain under the single-buffer loop (same source
  correct in the double-buffer variant). Reverted to software-pipelined loads.
- The GGUF stores token_embd.weight as BF16 (type 30), so lm_head runs the
  fused BF16 mmv path at 2.05 ms/launch — keep it that way (fp8 lm_head would
  be ~5.4 ms). The "quantize non-fp8 tensors" lever is now minor.
- The n_pad OOB read from section 7 is gone (CTA_N=64 pad == CTA span).

Status: committed as one unit (kernel + docs AITER_FINDINGS.md / HANDOVER.md /
PERF_HANDOVER.md). Remaining path to +50% over Q8_0 (8770 t/s), in order:
(1) gated_delta_net_cuda linear-attention kernel = 17% of pp512 (aiter has a
    gfx1201 port in this repo) - PERF_HANDOVER.md L1;
(2) GEMM weight-layout repack at load for 16-B staging loads - L2;
(3) smaller: fused activation quantize (L4), non-fp8 tensors to Q8_0 (L5,
    token_embd must stay BF16 - see L5).
