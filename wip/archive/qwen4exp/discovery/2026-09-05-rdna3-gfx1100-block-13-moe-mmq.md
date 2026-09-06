# RDNA3 (gfx1100 / RX 7900 XTX) validation of the block-13 fused MoE MMQ

Date: 2026-09-05
Machine: AMD Ryzen 9 7950X 16C, single AMD Radeon RX 7900 XTX (gfx1100,
Navi31, 24 GiB) + a HIP-visible gfx1036 Raphael iGPU (excluded with
`HIP_VISIBLE_DEVICES=0`; see the pinning note below).  ROCm
`/opt/rocm-7.14-gfx1100` (build toolchain AND runtime; do not mix in
system ROCm).
Build: ~/llama.cpp `rdna-boosts` (upstream master synced to `6a1a922d2`,
blocks 01-13 re-applied; block-13 commit `a8d0e5edc` after this fold),
ggml-hip Release, `GPU_TARGETS=gfx1100`, `GGML_HIP_GRAPHS=ON
GGML_HIP_MMQ_MFMA=ON GGML_HIP_NO_VMM=ON GGML_HIP_RCCL=1 GGML_NATIVE=1`
(ROCm 7.14 clang; no `CMAKE_HIP_FLAGS` override — matches this box's
`~/bin/build-llama-rocm-714` reference build).
Model: `/llm/models/Qwen3.6/35B-A3B/True-Q3_K_M/Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf`
(15.93 GiB, 35.51 B params, Q3_K - Medium; Q3_K is in the fused type
list Q3_K/Q4_K/Q5_K/Q8_0/Q6_K; fully VRAM-resident at `-ngl 99`).
Single discrete GPU => block 12's hybrid HIP all-reduce is N/A here
(no multi-GPU layer-split / all-reduce path; its internal AR is
RDNA4-gated anyway and never engages on gfx1100).  The dual-7900XTX
block-12 validation remains a separately-tracked parallel task.

> **Pinning note (why `HIP_VISIBLE_DEVICES=0`):** this box's rocminfo
> lists TWO HIP devices — the gfx1100 7900XTX and a gfx1036 Raphael
> iGPU (15 GiB carve-out).  Without the pin llama.cpp layer-splits
> across both (the golden rule from
> `wip/archive/qwen35moe-prefill/bench-config.md`).  All numbers below
> are pinned to the 7900XTX (`HIP_VISIBLE_DEVICES=0`).

## What was validated

Block 13's **fused MoE gate+up+GLU MMQ** (prefill; the {MUL_MAT_ID(gate),
MUL_MAT_ID(up), GLU} triple fused into one MMQ kernel) was RDNA4-only,
then RDNA4+RDNA3_5 after the 2026-09-05 Strix Halo session.  RDNA3_0
(gfx1100) was the remaining excluded arch ("until validated").  This
session ungated it for RDNA3_0 (gfx1100), validated numerics + perf,
probed whether the RDNA4-tuned J caps transfer (they do), and (after a
clean re-apply + full build at the `9cffdcc80` fork point) folded the
result into the block-13 commit / patch `0013`.

## Protocol

- llama-bench, `-ngl 99 -t 16 -r 8 -ub 2048`, prefill rows `-p 512
  -p 2048 -p 16384 -n 0` (pp512 always runs first — process-start
  warmup); decode rows `-r 4 -ub 2048 -p 512 -n 128` (tg128 from empty
  ctx).
- A/B toggle in the SAME build: `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`
  = 3-op fallback (the pre-ungate behavior on gfx1100), unset = fused.
- Coherence: llama-cli same-seed greedy (`-p "The capital of France
  is" -n 20 --temp 0 --seed 42 --single-turn --no-display-prompt`);
  outputs diffed with the spinner/timing/build-id lines stripped.
- Do NOT run parallel benches.  Warm the model page cache first.
- Phase-1 baseline = the shipped status quo on gfx1100: the fused arm
  is gated off there, so the shipped binary already runs the 3-op path
  (this also gives a pre/post-ungate fallback-purity check).

## Results (fused-on vs 3-op fallback; t/s, mean ± stdev)

Phase 1 baseline (shipped build, fusion OFF on gfx1100) and the
in-build Phase 2 A/B:

| test | fusion OFF (3-op) | fused ON (caps) | Δ |
|------|------------------:|----------------:|---:|
| pp512   | 3038.5 ± 37.5 (base) / 2991.2 ± 54.3 (in-build) | 3598.2 ± 48.5 | **~+20%** (noisy: single ubatch) |
| pp2048  | 4985.0 ± 14.5 (base) / 4938.7 ± 9.8 (in-build) | 5405.0 ± 9.2 | **+9.4%** |
| pp16384 | 4202.5 ± 13.2 (base) / 4161.7 ± 6.5 (in-build) | 4487.2 ± 11.5 | **+7.8%** |
| tg128   | 130.65 ± 1.15 (base) / 130.39 ± 1.19 (in-build) | 130.32 ± 1.18 | 0 (decode untouched) |

Coherence: 20-token same-seed greedy output **byte-identical** fused-on
vs off, and the ungated build's 3-op output is byte-identical to the
pre-ungate baseline binary's (the ungate does not perturb the fallback
path).  Fusion confirmed firing on gfx1100 (one-time stderr log during
the session: `fused MoE MMQ fired: src0 q3_K, cc 16781568` — cc =
0x1001100 encoding; log removed before landing).

The gains are in the same band as RDNA4 (qwen35moe pp16384 Q6_K +5.1%,
Q4_K_M +3.6%) and larger than Strix (+5.3%/+4.6%) on this card.

## J-cap transfer probe (port-tuning question)

With the fusion ON, the RDNA4-tuned `J_max_gate` caps (Q3_K/Q8_0=64,
Q4_K/Q5_K=96, Q6_K=64) were probed against alternatives on gfx1100
(Q3_K is the only fused type this model exercises):

| test | cap 64 (RDNA4 caps) | cap 96 (probe) | uncapped (J=128) |
|------|--------------------:|---------------:|-----------------:|
| pp512   | 3598-3600 | 3028.9 ± 52.6 | 2814.0 ± 51.9 |
| pp2048  | 5405.0 ± 9.2 | 5094.1 ± 18.9 | 4818.9 ± 15.1 |
| pp16384 | 4487.2 ± 11.5 | 4250.7 ± 43.6 | 4070.1 ± 26.9 |

Uncapping (J=128) regresses below even the 3-op fallback (pp2048 4939,
pp16384 4162) — register pressure with two live accumulators, same
class as Strix (1674 -> 1111).  The Q3_K@96 probe also loses to the cap
64.  **The RDNA4-tuned caps transfer to gfx1100 unchanged; no per-arch
port tuning is needed for the fused MMQ.**

## Clean re-application (the delivery fold)

The validated change (2 files, +17/-11: the try_fuse gate in
`ggml-cuda.cu` and the `J_max_gate` cap branch + comments in `mmq.cuh`)
was folded into the block-13 commit on the source-of-truth fork
(~/llama.cpp, block-13 commit `a8d0e5edc`; the ~/llama.cpp master is
synced past the `9cffdcc80` fork point, so the delivery set is
regenerated from a canonical fork rebuilt AT `9cffdcc80`).  Canonical
fork rebuilt with the pre-amendment delivery (`scripts/apply-all.sh`,
git am clean), block-13 amended there (`8c2ace510`), and the 13-patch
set regenerated with `scripts/make-patches.sh` (base `9cffdcc80`, blocks
tip `8c2ace510`): patches 0001-0012 changed only in patch headers (new
canonical-rebuild commit hashes; bodies byte-identical), 0013 carries
the fold.  `rdna-boosts-all.patch` regenerated
(`git diff 9cffdcc80..8c2ace510`; applies cleanly at `9cffdcc80`).
Clean-apply sim on this box, same ROCm/GPU: fresh worktree at
`9cffdcc80` + `apply-all.sh` of the regenerated set — git am clean with
zero whitespace warnings; applied tree byte-identical to the canonical
fork tip `8c2ace510`; full build clean (gfx1100, same flags as the
working tree); sim same-seed coherence identical (output token-identical
to the working-tree fused-on build); sim perf reproduces the
working-tree numbers (pp2048 5394.2 ± 18.5 / pp16384 4481.6 ± 12.1
fused vs the 3-op 4938.7 / 4161.7).

## Status / follow-ups

- Folded into patch `0013` (delivery regenerated 2026-09-05; see
  MANIFESTS.md dated record).  RDNA4 / RDNA3_5 behavior unchanged (the
  gate and caps for gfx120x/gfx115x are exactly the pre-2026-09-05
  code).
- Block 12 stays N/A on this single-GPU box (no all-reduce path;
  internal AR RDNA4-gated).  The dual-7900XTX block-12 validation
  remains a separately-tracked parallel task — allreduce code was not
  touched.
- The qwen4exp beta patch set (`beta/qwen4exp`) still needs its own
  Strix Halo validation — the next active task.
