# The FP8 E4M3 commit series (vendored for a clean re-base)

`git format-patch 6ea215d17..cllm` from the fork `stew675/llama.cpp`, branch `cllm`.
**Source of truth is `~/cllm`** (tip `7c17faffc`), not `origin/cllm` (`535d3bcb1`) — the last commit
is unpushed.

| | |
|---|---|
| base | `6ea215d17` (2026-08-05, the branch's stale `master`) |
| tip | `7c17faffc` |
| size | 23 commits, **45 files, +6653/-79** |
| net diff | `../cllm-fp8-full.diff` (same range, for a squash or an overview) |

## Apply

```bash
git am wip/fp8-support/patches/*.patch        # on a fresh tree at the current base
```

**Do not apply blindly.**  The series was written against `6ea215d17`; the current base is `84e76d8a2`
(+ the 16 delivery patches, tree `a3dc4bbb…`).  Expect conflicts, and **expect to drop commits whose
content already landed** — the chunked-GDN work (commits 9-14) became delivery **block 02**.

## The commits

| # | sha | subject | note |
|---:|---|---|---|
| 1 | `d56412c8c` | ggml : add FP8 E4M3 type registration (M1) | `GGML_TYPE_F8_E4M3 = 43` |
| 2 | `5f566c315` | Phase 2 of FP8 project | |
| 3 | `748d52e0f` | llama : direct safetensors loading (M3) + fp8 perf fixes | `llama-safetensors.cpp` (905 L) |
| 4 | `b678f8a4d` | fp8 : fix activation encoder saturation + M3/M4 completion | |
| 5 | `533abf443` | convert : `--outtype fp8_e4m3` writes native F8_E4M3 GGUFs (M5) | needed for the 27B |
| 6 | `821d423ac` | Performance investigations | baseline measurement commit |
| 7 | `26b7be281` | fp8 : aiter-style wmma port: 128x64 CTA, 8 warps, 2 CTAs/CU | **the GEMM kernel** |
| 8 | `734d0dc46` | fp8 : close the decode gap to Q8_0 (tg64 72 -> 89) | dot4 GEMV |
| 9 | `bb31b45d5` | gated_delta_net : add chunked prefill kernel (opt-in, slow) | **likely already in block 02** |
| 10 | `ef41a940a` | gated_delta_net : fix chunked phase A closure, enable by default | likely block 02 |
| 11 | `56e7cb0c9` | gated_delta_net : rewrite chunked phase B, 33.8ms -> 12.8ms | likely block 02 |
| 12 | `9c859e63b` | docs : split perf docs into LEVERS.md / PERF_HANDOVER.md | docs |
| 13 | `a921f8fb0` | gated_delta_net : rewrite chunked phase A, 293us -> 210us | likely block 02 |
| 14 | `c8757d649` | gated_delta_net : vectorize the phase B delta loads | likely block 02 |
| 15 | `d4e0b83fb` | docs : update LEVERS/PERF_HANDOVER | docs |
| 16 | `290e3f610` | docs : record wmma GEMM investigation, L1 spent | documents the L1 ceiling |
| 17 | `b148430ea` | ggml-cuda : repack fp8 weights for the wmma kernel | +~9 % |
| 18 | `5088acd51` | docs : update LEVERS/PERF_HANDOVER | docs |
| 19 | `fdbabb216` | ggml-cuda : fuse ssm conv input concat | pp512 6303 -> 6538 |
| 20 | `7550bfa48` | ggml-cuda : barrier-free warp quantize for fp8 staging | pp512 6630 -> 6818 |
| 21 | `ad620cafa` | convert : fp8 delta-net gate projections (in_proj_a/b) | pp512 6818 -> 7260 |
| 22 | `535d3bcb1` | ggml : fp8 get_rows + single-copy fp8 token_embd | -22 % file, same speed |
| 23 | `7c17faffc` | convert : fp8 output.weight for untied lm_head + fix qwen35 tensor split | **unpushed**, in `~/cllm` |

## Expected conflict surface

`ggml/src/ggml-cuda/gated_delta_net.cu`, `ssm-conv.cu`, `ggml-cuda.cu`,
`ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/ops.cpp`, `convert_hf_to_gguf.py` / `conversion/*`.
The `hunk`-heavy FP8 files (`fp8.cu/.cuh`, `llama-safetensors.*`) are new and should apply clean.
