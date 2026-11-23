# wip/gdn-norm-gate-silu — the GDN output norm folds into the SILU gate too

Written by Claude (AI).  One `git am` file made on `v17-919794500-r352` (tip `e695c46ce`, tree `0d2f70e6c`); applied, the tree
is `473076375`.

| file | touches | switches |
|---|---|---|
| `0001-CUDA-HIP-the-GDN-output-norm-folds-into-the-SILU-gat.patch` | `ggml-cuda-try-fuse.cu`, `common.cuh`, `ggml-cuda.cu` (+34 −5) | `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0` (kill switch), `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1` (census) |

## What it does

The gate -> Q8_1 kernel that computes the GDN output `RMS_NORM x w` itself (`ggml_cuda_gdn_norm_gate_*`,
`unary_gated_q8_1_norm_kernel`) is already templated on the gate op and takes `op_silu`, but `ggml_cuda_gdn_norm_gate_match`
and the skip at the `RMS_NORM` accepted only a SIGMOID gate.  The qwen35 / qwen3next gate is SILU
(`build_norm_gated()`), so on those models the fusion never fired: the census below shows `fused sigmoid=0` on Qwen3.8-27B,
and each GDN layer ran its output `RMS_NORM` (+ `MUL`) as its own kernel on every decode step.

- The matcher and the skip accept SIGMOID or SILU; the kernel is the same, launched with `op_silu` for a SILU gate.  A SIGMOID
  gate takes exactly the path it took before.
- `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0` restores SIGMOID only.
- `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1` logs, when the context is freed,
  `GDN_NORM_GATE_CENSUS dev=<d> silu=on|off fused sigmoid=<n> silu=<n> declined silu=<n>`: the gates fused per op, and the SILU
  gates that matched but the switch declined, so the OFF arm is not vacuous.  Host-side decisions: a replayed graph is not
  counted.

## Measurements

One R9700 (gfx1201), Linux, ROCm 10.0.0.  Arms: **S** stock r352, **P** r352 + this commit, **F** the same binary with
`GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0`.  `ROCBLAS_USE_HIPBLASLT=0`, one fresh process per run.

**Bits.**  Qwen3.8-27B UD-Q4_K_XL, q8_0 K/V: four texts (Chinese prose, Chinese technical, English technical, English code),
an 8192-token prefix prefilled at `-b/-ub 2048`, then 400 teacher-forced positions at widths 1 and 6 (decode, and a verify
width), each position's full-vocabulary log-probabilities written out (397,312,000 bytes per text and width).  S, P and F
are **byte-identical** for all 4 texts x 2 widths.

**Census** (the same tool, one text, 27B): P `silu=on fused sigmoid=0 silu=432 declined silu=0`; F `silu=off fused sigmoid=0
silu=0 declined silu=432`.

**Text** (`llama-cli -p "The capital of France is" -n 128 --seed 42 --temp 0 --single-turn`, no `-lv`; the spinner, build and
timing lines left out of the hash): the same in S, P and F on Qwen3.8-27B UD-Q4_K_XL (`cc3a39f64a7d`) and Qwen3.6-35B-A3B
UD-Q4_K_M (`8ed33a8c8f75`).

**Speed** (`llama-bench -fa 1 -p 0 -n 128 -d 0,8192 -r 5`, order S P F F P S per model, t/s; the first run of each order, S,
is left out: it ran right after the model's first load and is low on both models):

| model | depth | S | P (two runs) | F (two runs) | P vs F |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL | 0 | 32.12 | 32.05, 32.16 | 32.05, 32.06 | +0.2 % |
| | 8192 | 31.18 | 31.39, 31.40 | 31.21, 31.18 | +0.6 % |
| Qwen3.6-35B-A3B UD-Q4_K_M | 0 | 118.48 | 117.81, 118.28 | 117.88, 117.91 | +0.1 % |
| | 8192 | 114.73 | 114.81, 115.24 | 114.11, 113.87 | +0.9 % |

So: small.  At depth 0 nothing beyond the run-to-run spread; at 8192 +0.6 % / +0.9 % in both runs of both models.  The
mechanism is the launch count: one `RMS_NORM` (+ `MUL`) kernel fewer per GDN layer and decode step (48 on the 27B).  It
could be worth more where a launch costs more (Windows / WDDM); we have no Windows measurement.

## Not tested

Windows; RDNA3 / 3.5; NVIDIA (the CUDA build was not compiled, only HIP); several GPUs (`-sm layer` / `tensor`); MTP or other
draft decoding (llama-bench and the teacher-forced tool above run no drafts; width 6 is a verify-sized batch, not a
draft run); qwen3next itself (only qwen35 / qwen35moe models); `test-backend-ops` (no case reaches this fusion).
