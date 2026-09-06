# Strix Halo (RDNA3.5 / gfx1151) validation of the block-13 fused MoE MMQ

Date: 2026-09-05
Machine: Ryzen AI MAX+ 395 w/ Radeon 8060S (Strix Halo APU), 16C/123 GB,
single gfx1151 device (rocm-smi VRAM carve-out 2 GiB; model runs from
unified memory).  ROCm `/opt/rocm-7.14-gfx1151` (build toolchain AND
runtime; do not mix in system ROCm).
Build: master `8b4b3558f` + blocks 01-13 (`~/llama.cpp`, branch
`rdna-boosts`, block-13 commit `4c7e2267b`), ggml-hip Release,
`AMDGPU_TARGETS=gfx1151`, `CMAKE_HIP_FLAGS="-mllvm
--amdgpu-unroll-threshold-local=600"`, `GGML_HIP_GRAPHS=ON
GGML_HIP_MMQ_MFMA=ON GGML_HIP_NO_VMM=ON GGML_HIP_RCCL=1 GGML_NATIVE=1`.
Model: `/llm/models/Qwen3.6/35B-A3B/True-Q3_K_M/Qwen_Qwen3.6-35B-A3B-Q3_K_M.gguf`
(15.93 GiB, 35.51 B params, Q3_K - Medium; Q3_K is in the fused type
list Q3_K/Q4_K/Q5_K/Q8_0/Q6_K).

## What was validated

Block 13's **fused MoE gate+up+GLU MMQ** (prefill; the {MUL_MAT_ID(gate),
MUL_MAT_ID(up), GLU} triple fused into one MMQ kernel) was RDNA4-only:
the try_fuse arm in `ggml-cuda.cu` and the per-type `J_max_gate` tile
caps in `mmq.cuh` were gated on `GGML_CUDA_CC_IS_RDNA4(cc)` ("disabled
until validated on other arches").  This session ungated it for
RDNA3_5 (gfx1151), validated numerics + perf, probed whether the
RDNA4-tuned J caps transfer, and (after clean re-apply + build) folded
the result into the block-13 commit / patch `0013`.

Block 12's internal all-reduce is N/A here (single GPU).  Decode-side
work already carries RDNA3_5 parameter tables (block 10 `mmvq`), and
decode was re-measured to confirm the prefill-only change did not touch
it.

## Protocol

- llama-bench, `-ngl 99 -t 16 -r 8 -ub 2048`, prefill rows `-p 512
  -p 2048 -p 16384 -n 0`; decode rows `-n 128` (tg from empty ctx).
- A/B toggle in the SAME build: `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`
  = 3-op fallback (the pre-ungate behavior), unset = fused.
- Coherence: llama-cli same-seed greedy (`-p "The capital of France
  is" -n 20 --temp 0 --seed 42 --single-turn --no-display-prompt`);
  outputs diffed with the timing line stripped.
- Do NOT run parallel benches.  Warm the model page cache first.
- APU clock note: the first llama-bench test after process start runs
  cold (pp2048 read 1275±166 when it was the first test).  Always give
  a short pp512 warmup test first — pp2048 then reads 1674±3.

## Results (fused-on vs 3-op fallback; t/s, mean ± stdev)

| test | fusion OFF (3-op) | fused ON | Δ |
|------|------------------:|---------:|---:|
| pp512   | 963 ± 72 | 1094-1100 ± 97 | ~+14% (noisy: single ubatch) |
| pp2048  | 1590.7 ± 4.5 | 1673.4-1676.3 ± 3-13 | **+5.3-5.4%** |
| pp16384 | 1360-1361 ± 44-48 | 1422.9-1424.6 ± 47-51 | **+4.6%** |
| tg128   | 71.5 | 71.5 | 0 (decode untouched) |

Coherence: 20-token same-seed greedy output **byte-identical** fused-on
vs off (this held on both the working tree and the final clean-apply sim
build).  Fusion confirmed firing on gfx1151 (one-time stderr log during
the session: `src0 q3_K, cc 1001151`).

RDNA4 reference (for comparison; qwen35moe 1-GPU R9700, from the
block-13 notes): pp16384 Q6_K +5.1%, Q4_K_M +3.6% — the Strix gains are
in the same band.

## J-cap transfer probe (port-tuning question)

With the fusion ON, the RDNA4-tuned `J_max_gate` caps (Q3_K/Q8_0=64,
Q4_K/Q5_K=96, Q6_K=64) were removed for RDNA3_5 (J search uncapped to
128) to test whether Strix wants its own tile widths:

| test | capped (RDNA4 caps) | uncapped (J=128) |
|------|--------------------:|-----------------:|
| pp2048  | 1673.9 ± 3.3 | 1111 ± 172 |
| pp16384 | 1422.9 ± 47   | 1334 ± 53   |

Uncapping regresses (register pressure / occupancy at J=128 with two
live accumulators) — **the RDNA4 caps transfer; no per-arch port tuning
needed for the fused MMQ**.  The caps are applied on RDNA3_5 as-is.

## Clean re-application (the delivery fold)

The validated change (2 files, +13/-7) was folded into the block-13
commit and the 13-patch set regenerated from a canonical fork rebuilt at
`9cffdcc80` (block-13 tip `ace0a5d54`).  Clean-apply sim on the Strix
machine itself, same ROCm/GPU: `git am` of the regenerated 0001-0013 at
`9cffdcc80` clean with zero whitespace warnings; applied tree
byte-identical to the canonical fork tip; full build clean; sim
coherence identical; sim perf reproduces the working-tree numbers
(pp2048 1676.3 / pp16384 1424.6 fused vs 1590.7 / 1361.4 unfused).
`rdna-boosts-all.patch` regenerated and re-checked (applies cleanly at
`9cffdcc80`).

## Status / follow-ups

- Folded into patch `0013` (delivery regenerated 2026-09-05; see
  MANIFESTS.md dated record).  RDNA4 behavior unchanged (gate/caps for
  gfx120x are exactly the pre-2026-09-05 code).
- **Update (same day, 2026-09-05):** the RDNA3_0/gfx1100 leg below was
  completed on a single RX 7900 XTX — validated (fusion fires,
  coherence IDENTICAL, pp2048 +9.4% / pp16384 +7.8%, decode unchanged,
  RDNA4 J caps transfer) and folded into the same patch `0013`
  (canonical rebuild tip `8c2ace510`).  Record:
  `benchmarks/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
- RDNA3_0 (gfx1100, 7900XTX-class) is still excluded from the fused arm
  until validated there — that leg runs in parallel with a community
  member (dual-7900XTX box).
- The qwen4exp beta patch set (`beta/qwen4exp`, documented as "Requires
  ROCm gfx1201") still needs its own Strix Halo validation — the next
  active task (environment prep pending).
