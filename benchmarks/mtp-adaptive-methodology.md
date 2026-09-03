# Adaptive MTP baseline methodology & validation records

Why this file exists: on 2026-09-02, a regression in the delivery's decode
kernels collapsed **adaptive MTP / draft-mtp on MoE models** (and cost ~33%
on dense MTP). It went unnoticed for weeks because no validation gate
exercised MTP end-to-end: the llama-benchy decode suites (see
`benchmarks/`) run plain (non-speculative) decode only, and MTP numbers
never made it into a dated baseline. This file is the MTP-specific gate:
protocols, canonical commands, expected numbers (dense + MoE), and the
acceptance/verification rules that would have caught the regression.

Status: 2026-09-02 — gate defined; dense + MoE baselines recorded (pre- and
post-fix). All runs on 3x R9700 (gfx1201), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`), fork build flags as in
`wip/qwen35moe-prefill/bench-config.md`; 1-GPU runs pin `HIP_VISIBLE_DEVICES=0`
(GOLDEN RULE 1: without the pin llama.cpp layer-splits and decode drops
~97 -> ~81 t/s on the MoE model).

## Why decode benches cannot see MTP regressions

MTP adds two decode shapes that plain decode never produces:
- **verify batches**: the target is decoded at ncols = n_draft+1 (2..13),
  i.e. multi-token MUL_MAT and multi-token MUL_MAT_ID (MoE) rows;
- **draft-context decode**: single-token steps through the masked/nextn
  path, re-captured graphs, and (for single-head MTP) sequential drafting.

llama-bench tg decodes 1 token/step regardless of `-b`, so it only covers
the ncols==1 path. Kernel or fusion bugs confined to ncols 2..13 (or to the
draft context) are invisible to llama-bench/benchy and to single-token
same-seed coherence. The 2026-09-02 regression is the worked example: the
block-08 rms_norm->mmvq Q8_1 quantize-cache fold corrupted multi-token
MUL_MAT_ID, so MoE verify logits diverged from single-token decode and MTP
draft acceptance collapsed to 0/1527 (draft-mtp ~53 t/s vs plain ~90, where
MTP should accelerate). Dense models and single-token MoE decode were
unaffected, so every existing gate passed.

## Protocols

### Protocol A — fast per-build gate (llama-cli, fixed seed)

Canonical (dense): `bench.sh` style, seed 42, temp 0, predict >= 512:

```sh
HIP_VISIBLE_DEVICES=0 GGML_CUDA_DISABLE_GRAPHS=0 <build>/bin/llama-cli \
  --model <model> --fit false --top-k 20 --threads 8 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --predict 1024 --load-mode mlock \
  --cache-ram 16384 --ctx-size 102400 --flash-attn auto --temperature 0.0 \
  --batch-size 1024 --ubatch-size 1024 --n-gpu-layers all \
  --cache-type-k bf16 --cache-type-v bf16 --ctx-checkpoints 64 \
  --reasoning-budget 65536 --reasoning-preserve --repeat-penalty 1.0 \
  --presence-penalty 1.5 --seed 42 --single-turn \
  --spec-type draft-mtp --prompt '<prose prompt>'
```

Run the same with `--spec-type none` and compare. Gate rules:
1. **Acceptance**: with `--log-verbosity 4`, the `draft acceptance` /
   `acc per pos` lines must show a healthy rate on prose (>= ~0.45 at pos 1
   for these models; the 2026-09-02 regression showed 0.000). A collapse to
   0.0 = verify/logit numerics divergence (draft-vs-verify mismatch), not a
   tuning issue.
2. **MTP must not lose to plain decode on the same build**: `draft-mtp`
   Generation t/s should be >= `none` Generation t/s on predictable content
   and within ~10% on generic prose at draft depth 3. (Do NOT test at
   `--spec-draft-n-max 12` fixed depth: fixed-depth over-drafting is
   expected to lose; the adaptive configs C3/C6 below are the meaningful
   high-depth tests.)
3. **Same-seed determinism vs the previous release** (dense): outputs must
   be byte-identical between the build under test and the known-good build.
   On MoE this is not required (fusion-ordering numerics drift is an
   accepted trade-off); sane output is the bar there.

### Protocol B — server harness (dense canonical, long-context workloads)

The full adaptive-MTP comparison harness lives at
`/home/stew675/stew675/adaptive-results-perf/` (`run_sweep.py`; configs C0
no-spec / C3 draft-mtp-adaptive n 3..12 / C6 adaptive+ngram; workloads
R=reasoning, K=verbatim recall, P=prose, C=code; server = 2-GPU tensor
split `HIP_VISIBLE_DEVICES=0,2`, Q8_0 27B dense, f16 KV, ctx 262144).
Run passes 1 and 2, repeat >= 2, and compare against the baselines below.
The dense expectations table was recorded 2026-08-23 (rdna-boosts era
`e0aa19e25`) and re-verified 2026-09-02 on the fixed 13-block build —
see the results table.

## Baselines

### Dense — Qwen3.8-27B Q8_0, 2-GPU tensor split (0,2), f16 KV

Predicted tokens/s (predicted_per_second). C0 = plain decode.

| config | workload | 2026-08-23 perf-era | 2026-09-02 fixed 13-block | base (upstream) C0 |
|---|---|---|---|---|
| C0 | R | - | 31.8 | 30.0 |
| C0 | K | - | 31.6 | 29.9 |
| C0 | P | - | 31.9 | 30.1 |
| C0 | C | - | 31.8 | 30.0 |
| C3 | R | 53.2 | 53.0 | - |
| C3 | K | 149.2 | 148.2 | - |
| C3 | P | 57.6 | 59.5 | - |
| C3 | C | 78.0 | 79.1 | - |
| C6 | R | 53.0 | 53.1 | - |
| C6 | K | 329.8 | 324.6 | - |
| C6 | P | 57.6 | 59.4 | - |
| C6 | C | 75.6 | 79.1 | - |

Dense verdict: MTP baseline fully restored (C3/C6 parity or better;
C0 plain decode also ahead of upstream base). C3 speedups vs C0 on the same
build: R 1.67x, K 4.7x, P 1.87x, C 2.5x — MTP is a strong accelerant.

### MoE — Qwen3.6-35B-A3B (qwen35moe, nextn MTP head)

1-GPU UD Q4_K_M (has the nextn head), seed 42, temp 0, draft depth 3
(default n_max). There was NO pre-2026-09-02 MoE MTP baseline — that is why
the regression was not caught (the MoE decode suites are MTP-free).

| build | plain (none) | draft-mtp | draft acceptance | verdict |
|---|---|---|---|---|
| upstream 9cffdcc80 | 75.0 | 113.4 | 0.49 | MTP accelerates +51% |
| shipped 13-block (pre-fix) | 88.6 | 53.2 | 0.000 (0/1527) | MTP collapses (bug) |
| fixed 13-block (2026-09-02) | 89.5 | 125.8 | 0.51 | MTP accelerates +41% |

MoE verdict: post-fix MTP acceptance (0.51) equals the fully-unfused
internally-consistent numerics (0.49) and upstream — the draft-vs-verify
numerics relationship is not depressing acceptance. Fix summary and the
mechanism in `patches/README.md` (block 13 notes, 2026-09-02).

### MoE multi-GPU — Qwen3.6-35B-A3B-UD Q8_0, tensor split (new baselines, 2026-09-02)

First-ever MoE 2- and 3-GPU rows (no prior baseline existed). Model has the
nextn/MTP head. Env conventions as `run_sweep.py`: `NCCL_PROXY_CPUSET=8..15`,
`NCCL_P2P_DISABLE=1`, `GGML_CUDA_DISABLE_GRAPHS=0`, tensor split. Fixed
13-block build (delivery tip `8f2838d1`). llama-bench: `-t 16 -r 2 -ub 2048
-p 512 -n 128`; Protocol A MTP: temp 0, seed 42, ctx 32768, bf16 KV,
predict 512.

| config | tg128 f16 | tg128 bf16 | pp512 f16 | pp512 bf16 | plain (proto A) | draft-mtp | draft acceptance |
|---|---|---|---|---|---|---|---|
| 2-GPU (0,2) | 97.63 | 97.50 | 4603 | 5009 | 91.2 | 125.1 | 0.542 |
| 3-GPU (0,1,2) | 103.17 | 102.91 | 4770 | 4764 | 96.6 | 124.6 | 0.529 |

MoE decode scales weakly across GPUs at batch 1 (expert compute per token;
tensor split cannot share experts) — 2-GPU ~= 1-GPU Q6_K (~98), 3-GPU ~103.
MTP accelerates at both splits (+37% 2-GPU, +29% 3-GPU) with acceptance
~0.53 — the hybrid-AR + MTP combination is healthy on multi-GPU.

### MoE single-token decode anchors (MTP-free, for reference)

Canonical commands in `wip/qwen35moe-prefill/bench-config.md`. Recorded
baseline (2026-09-02, fork tip) vs fixed 13-block (2026-09-02):

| test | baseline | fixed 13-block |
|---|---|---|
| Q6_K 1-GPU tg128 | 97.6 | 98.8 |
| Q8_0 2-GPU tensor tg128 | 95.6 | - |

## What to run before shipping a decode/fusion change

1. Protocol A on the dense Q4_K_XL-UD and the MoE Q4_K_M-UD (none +
   draft-mtp, seed 42) — acceptance > ~0.45 and MTP >= plain.
2. Protocol B (C0/C3/C6, passes 1+2) on the dense Q8_0 row.
3. If any fusion/try_fuse/multitoken-kernel code changed, also run MoE MTP
   at draft depth 3 with fusion ON vs `GGML_CUDA_DISABLE_FUSION=1` and
   compare acceptance (must match within noise; a 0-acceptance split = the
   fused multi-token path diverged again).
4. MoE multi-GPU MTP (2-GPU 0,2 and 3-GPU 0,1,2, Q8_0-UD): draft-mtp vs
   plain Protocol A rows — acceptance ~0.53 and MTP >= plain at both
   splits (baselines in the MoE multi-GPU table above).
