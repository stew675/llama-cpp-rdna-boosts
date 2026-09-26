# HANDOVER — prefill-gap attribution + FP8 path

**Branch:** `wip/prefill-gap-attribution`. **Fork scratch:** `~/llama.cpp` @ `527d39401` (clean).
**Read order:** this file → `README.md` → `RESULTS-2026-09-26.md` → `FP8-PRIOR-ART.md`.

## State at handover (2026-09-26)

* Attribution done: **prefill is 70-81 % `mul_mat_q`**, 4-17 % attention. The gap is GEMM.
* `mudler/vllm.cpp` **built on gfx1201** (first ROCm build per its own CMake) at `~/vllm.cpp/build-hip`
  — but it has no ROCm FP8 GEMM and is behind llama.cpp. Verdict: **not viable**; keep
  `src/vt/rocm/rocm_paged_attn.hip` as the prefill-attention reference.
* The precision confound: the 27B FP8 checkpoint is HF safetensors; llama.cpp has no FP8 type.
* **The path: `origin/cllm`** — a complete FP8 E4M3 port, measured +16-17 % prefill over Q8_0 on a 4B
  gfx1201. Working tree at `~/cllm` (same fork, second clone, 1 unpushed commit past `origin/cllm`, not built). `~/aiter` holds the AITER tree.

## Next actions (in order)

1. Build `~/cllm` for gfx1201 and reproduce the 4B numbers (`llama-bench -m stewfp8-ow.gguf -p 512`);
   confirm 7162-7260 t/s vs Q8_0 6177. Gate: the +16 % holds.
2. Re-base `origin/cllm` onto the current base (conflicts expected in `gated_delta_net.cu`,
   `ssm-conv.cu`, `ggml-cuda.cu`, `ggml-quants.c`; the GDN/ssm-conv parts largely already landed).
3. Convert `/llm/models/Qwen3.8/27B/FP8/` → F8_E4M3 GGUF and measure 27B fp8 pp8192 vs the int8
   baselines. **This is the number the maintainer actually wants.**
4. Lift AITER's gfx1201 fp8 GEMM configs into `mul_mat_fp8_wmma` (77-98 → target ≥110 TFLOP/s).

## Traps

* **PMC counters return 0 on gfx1201** — use `--kernel-trace` durations only.
* **llama-bench on an occupied GPU** silently CPU-offloads and looks like a regression — bench on a
  free `-dev ROCm#`.
* `llama-cli` in the cllm fork is a **server client**; keep benches and parity runs on the same GPU.
* The cllm box notes say `pp512` has ±60-170 t/s noise; trust rocprof kernel times (±3 %) for A/B.
* Do **not** use the AITER `bpreshuffle` path (5-8× slower on gfx1201).
* `~/llama.cpp` is shared: `~/llama-integration` is a second worktree on `beta-integration`. Keep the
  FP8 work in `~/cllm`.

## Next-session prompt

> Continue `wip/prefill-gap-attribution` in `~/llama-cpp-rdna-boosts`.  Read
> `README.md` / `RESULTS-2026-09-26.md` / `FP8-PRIOR-ART.md`.  The prefill gap is a GEMM problem
> (70-81 % `mul_mat_q`), vllm.cpp is assessed and not the path, and the real lead is the FP8 E4M3 port
> on `origin/cllm` (`~/cllm`) which measured +16-17 % prefill over Q8_0 on a 4B gfx1201.
> Start by building `~/cllm` and reproducing the 4B number, then convert the 27B FP8 safetensors to
> F8_E4M3 GGUF and measure it against the int8 baselines in `MEASUREMENTS.md`.  Keep everything in
> `~/cllm`; nothing reaches `patches/` without the promotion path.
