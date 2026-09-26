# FP8 prior art in the house — `origin/cllm` + AITER

Inventory for reviving the FP8 prefill path. All of this predates the current delivery and is **not**
in `patches/`. Treat it as WIP prior art (WIP rule applies).

---

## 0. Where it lives now

| tree | path | state |
|---|---|---|
| FP8 llama.cpp branch | `origin/cllm` @ `~/llama.cpp` (`git@github.com:stew675/llama.cpp.git`) | tip `535d3bcb1` (2026-08-06); merge-base with `rdna-boosts` = `6ea215d17` (2026-08-05) |
| FP8 working tree | `~/cllm` | **same fork, second clone**, branch `cllm`, tip `7c17faffc` = `origin/cllm` **+ 1 unpushed commit**; **not built** (`build-xcframework.sh` only) |
| AITER inspection tree | `~/aiter` | `22beb1caa` |
| 4B fp8 model | `/llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf` (4.5 GB, L9 single-copy) + `-2copy` + `preserved` | present |
| 27B fp8 source | `/llm/models/Qwen3.8/27B/FP8/` (safetensors, e4m3, dynamic act) | present, **not yet converted to GGUF** |

### `~/cllm` and the `cllm` branch are the same thing

Both are the `cllm` branch of the fork `stew675/llama.cpp`.  `~/cllm` is an independent **clone** of
that fork (not a worktree of `~/llama.cpp`, and not a different project), checked out on `cllm`; both
clones fetch `origin/cllm` = `535d3bcb1`.  The only difference is one **local, unpushed** commit:

```
7c17faffc convert : fp8 output.weight for untied lm_head + fix qwen35 tensor split
  PERF_HANDOVER.md  +31 | conversion/base.py  +28/-10 | ggml/src/ggml-backend-meta.cpp  +8
```

It is **not** in `~/llama.cpp` (which carries only the remote-tracking ref), so a re-base driven from
`origin/cllm` would silently drop it.  **Use `~/cllm` as the source of truth**; push `7c17faffc`
before relying on the server ref.  `~/cllm`'s local `master` is stale at `6ea215d17`, which is exactly
the FP8 merge-base — so `cllm` sits on a 2026-08-05 master.

`~/cllm` is the fast path (working tree, all the docs, the extra commit). It just needs building and
re-basing.

## 1. Phase status (from `LEVERS.md` / `PERF_HANDOVER.md` on `origin/cllm`)

| phase | content | state |
|---|---|---|
| M1-M5 | `F8_E4M3` type registration, CUDA/HIP WMMA GEMM, safetensors load, fp8 GGUF convert, activation-encoder fixes | done |
| aiter port | `mul_mat_fp8_wmma` (CTA 128x64, 8 warps, GROUP_M=4, 16x16x128, 2 CTA/CU) | done, ~77 TFLOP/s |
| GDN | chunked GDN prefill rewrite (phase A/B), conv fusion | done — **this part became the delivery's chunked GDN** |
| L1 | fp8 WMMA GEMM further tuning | "mostly spent; revisit only with a new idea" → the AITER configs are that idea |
| L2/L5/L6/L7/L9 | weight repack, conv fusion, quantize rewrite, in-proj fp8, single-copy embd | done, +16-17 % cumulative |

## 2. Scoreboard (Qwen3.5-4B, 1× gfx1201)

| config | pp512 t/s | tg64 t/s |
|---|---:|---:|
| Q8_0 GGUF | 6177 | 90.9 |
| fp8, pre-GDN | 4930 | 71.0 |
| fp8 + aiter GEMM port | 5596 | 71.4 |
| fp8 + chunked GDN | 6068 | 89.9 |
| fp8 + L7 in-proj fp8 | 7184-7260 | 88.4 |
| fp8 + L9 single-copy (final) | 7162-7170 | 88.5 |
| **target** | 8770 (+50 % over Q8_0) | 100 (+10 %) |

## 3. What to read first (on `origin/cllm`)

```bash
git -C ~/llama.cpp show origin/cllm:AITER_FINDINGS.md     # AITER on gfx1201, the fp8 GEMM asset
git -C ~/llama.cpp show origin/cllm:LEVERS.md             # ranked remaining levers
git -C ~/llama.cpp show origin/cllm:PERF_HANDOVER.md      # history + operational rules
git -C ~/llama.cpp show origin/cllm:HANDOVER.md           # the project as a whole
```

## 4. The re-base (the main risk)

```bash
git -C ~/llama.cpp diff --stat 6ea215d17..origin/cllm     # 44 files, +6606/-79
```

44 files, dominated by `fp8.cu/.cuh` (706), `llama-safetensors.cpp` (905), `gated_delta_net.cu`
(651), `ssm-conv.cu` (204), `ggml-quants.c` (146), plus `convert_hf_to_gguf.py` / `conversion/*`.
Note **the GDN + ssm-conv changes are largely already in the delivery** (chunked GDN landed as
block 02), so the true FP8 delta for a re-base is smaller than the raw diffstat suggests — expect
conflicts in `gated_delta_net.cu`, `ssm-conv.cu`, `ggml-cuda.cu`, `ggml-quants.c`.

## 5. Convert the 27B (the direct test)

```bash
# on a cllm build
python convert_hf_to_gguf.py /llm/models/Qwen3.8/27B/FP8/ --outtype fp8_e4m3 --outfile /llm/models/Qwen3.8/27B/FP8/Qwen3.8-27B-FP8.gguf
```

Then `llama-bench -p 8192 -n 0` and compare against the int8 baselines in `MEASUREMENTS.md` §1.
The 27B checkpoint's dynamic per-token activation scheme is handled at runtime by `quantize_fp8`
(weights carry the block scales); the HF → `block_f8_e4m3` scale mapping is the thing to verify.

## 6. AITER configs to lift

`~/aiter/ops/triton/configs/gemm/gfx1201-GEMM-A8W8_BLOCKSCALE*.json` (30 files) — tile sizes,
`GROUP_SIZE_M`, `kpack`, and the `M_LEQ_x` selection. The generic file applies when no specialized
one matches; the 27B shapes will need the generic or a new entry. **Do not** adopt the preshuffled
`bpreshuffle` path: measured 5-8× slower on gfx1201; row-major wins.
