# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.
Forward-looking: open items + current active experiments; closed work is a
one-line bullet (details live in AGENTS.md, patches/README.md, MANIFESTS.md,
`beta/qwen4exp/README.md`, `wip/` handovers, `benchmarks/`). Current
delivery = the 13-patch set against fork point `9cffdcc80` (blocks 01-13).
Reality pass: 2026-09-05.

## Current active

### Validate beta qwen4exp on Strix Halo (gfx1151) — env prep pending
- Block 13's Strix leg is DONE (2026-09-05): the fused MoE gate+up+GLU
  MMQ (RDNA4-gated fused arm) was ungated for RDNA3_5 / gfx1151,
  validated (pp2048 +5.3%, pp16384 +4.6%, coherence IDENTICAL, decode
  unchanged; RDNA4-tuned J caps transfer — an uncap probe regressed),
  and folded into patch `0013` (delivery regenerated at `9cffdcc80`,
  clean-apply sim + full build + coherence re-verified on the Strix
  box). Record:
  `benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
- Remaining here: validate `beta/qwen4exp` (the three patches,
  currently documented as "Requires ROCm gfx1201") on Strix Halo.
  Environment prep is pending (per the maintainer, 2026-09-05) — the
  beta base (master `8b4b3558f` + blocks 01-13 incl. the fold) is
  already built in `~/llama.cpp/build-rocm` on this machine.
- Standing gate before shipping any decode/fusion change:
  `benchmarks/mtp-adaptive-methodology.md` (MTP baseline + acceptance).
- Where: `beta/qwen4exp/README.md` (gates + carried-forward open items).

### Parallel (community member): dual 7900XTX (RDNA3, gfx1100)
- Block-12 (hybrid HIP all-reduce) validation on RDNA3 pairs is ongoing
  with a community member on their dual-7900XTX box: hybrid-dispatch
  matrix (internal vs nccl vs none) + bounded-spin path at depth-16384.
  The block-12 gate stays RDNA4-only until verified, then the arch check
  is removed. Volunteer test env: `GGML_CUDA_ALLREDUCE=internal`.
- Also the RDNA3.0/gfx1100 leg for block 13's fused MoE MMQ (excluded
  until validated there) — the same box can double for it.
- Where: `patches/0012` + the block-12 notes in `patches/README.md`.

## Open follow-ups

### Upstream monitor: ROCm unaligned-width split-load (Q6_K/Q3_K 2-GPU)
- Upstream bug: H2D 2D copies whose width is not a multiple of 4 (Q6_K
  quant block = 210 B, Q3_K = 110 B) are ~1000x slower on ROCm. Fixed
  locally in block 13 (`set_tensor_2d`: aligned-H2D + unaligned-D2D
  staging). No PR planned — upstream is busy with its own qwen4exp work;
  the follow-up is to watch whether they fix it themselves. If they do,
  it will surface as a rebase conflict and resolve naturally. Re-check at
  each re-base.
- "Done": upstream ships the fix (or the local fix gets upstreamed).

### MXFP4 (and NVFP4) fused gate+up+GLU MMQ — LAST block-13 item
- The fused MoE MMQ kernel (`ggml_cuda_mul_mat_q_switch_type_gate`) is
  instantiated for Q3_K/Q4_K/Q5_K/Q8_0/Q6_K only; MXFP4/NVFP4 (and
  Q4_0/Q4_1/Q5_0/IQ*/...) would abort if `ggml_cuda_should_use_mmq`
  admits them, so the try_fuse arm is gated on the instantiated type list.
- Why it matters: MXFP4 is the interesting type for future MoE models
  (deepseek-style native MXFP4 experts) and would let the gate be
  relaxed. Tracked from the block-13 notes in `patches/README.md`.
- "Done": add MXFP4 (+ maybe NVFP4/Q4_0-class) switch cases + instance
  files + generators, then bit-exact + bench validation per the 0004
  recipe.

## Parked

### LFRU host->GPU slow hot-weight migration
- Survivor of the expert-tiering experiment, which was DROPPED
  (2026-09-05): investigation showed most of its aims are already covered
  by current llama.cpp options. The one idea left: an LFRU-style very
  slow migration of hot weights from host to GPU (persistent GPU slot
  cache + CPU-computed cold tail). NOT active now; may become active soon.
- Design notes: `wip/qwen4exp/LRU_EXPERTS.md`, `PHASE0_ROUTING.md`,
  `HANDOVER-2026-09-04-tiering.md` (wip = experimental, not delivery).

## Closed (one-liners; details in the dated docs)

- 2026-09-01 Block 13 released as the 13th delivery patch (fork
  a14257996): fused MoE gate+up+GLU MMQ + mmvq item-split; qwen4exp MoE
  work split out of the delivery.
- 2026-09-02 Multi-token MUL_MAT_ID `x_scale_channel_dst` fusion (fork
  9db2fcbdc, folded into block 13): per-(expert, token) x_scale;
  test-backend-ops 16222/16222.
- 2026-09-01 ROCm unaligned-width split-load fix in block 13 (fork
  834a8d3ff): Q6_K/Q3_K 2-GPU load <15 s (the old ~3 min "hang" was this).
- 2026-09-02 Delivery re-based to upstream `9cffdcc80` (+42 upstream
  commits; 3 manual block re-base hunks; clean apply).
- 2026-09-02 Block 13 amended with the two MTP regression fixes: mmvq
  ksplit dispatch for verify batches (dense MTP 18.3 -> 27.5 t/s) and the
  rms_norm-fold gated to single-token MMID (MoE MTP acceptance 0 -> 0.51,
  119-129 t/s).
- 2026-09-04 Block 12 amended with the runtime NCCL-failure fallback
  (issue #13): on first NCCL runtime failure clear the sticky HIP errors,
  warn once, stop using NCCL, route AllReduce to internal/butterfly. No
  behavior change on healthy setups.
- 2026-09-03 qwen4exp WIP promoted to `beta/qwen4exp/` (QSA sparse FA is
  the default FA path); QSA decode regression fixed (V smem staging +
  top-k slicing + 64-cell slices; server flat ~46.5-48 t/s).
- 2026-09-04 beta/qwen4exp re-based onto master `8b4b3558f` + blocks
  01-13; patch 3 added (NextN/MTP draft head, `--spec-type draft-mtp`);
  layer-split crash fixed (head-grouped launches); AesSedai
  Qwen3.8-Flash-Next supported. qwen4exp gfx1201/RDNA4 work done.
- 2026-09-05 ITEM B (QSA sparse FA latency push) + the decode push:
  CLOSED at ~48 t/s @ ~95% GPU occupancy. Early probing suggested ~3x
  headroom; it never materialized (register pressure, CU occupancy, VRAM
  bandwidth saturation). ~20% speedup + near-100% utilization were won,
  but attention was not the dominant cost. Remaining decode levers are
  tracked in `beta/qwen4exp/README.md`.
- 2026-09-05 Block 13 fused MoE MMQ ungated for RDNA3_5 (Strix Halo /
  gfx1151) and folded into patch `0013`: validated on Ryzen AI MAX+ 395
  (Qwen3.6-35B-A3B True-Q3_K_M, ub 2048) — pp2048 +5.3% (1590 -> 1674),
  pp16384 +4.6% (1360 -> 1423), decode unchanged, coherence IDENTICAL;
  RDNA4 J caps transfer (uncap probe regressed 1674 -> 1111). Delivery
  regenerated at `9cffdcc80`; clean-apply sim + full build + coherence
  re-verified on the Strix box. Record:
  `benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
- 2026-09-05 Expert-tiering experiment dropped (see Parked).
- Older resolved items (block-12 fused-stage/pacing closure, ITEM A JIT,
  indexer head-sum revert, qwen35moe dense-GQA N/A, ...) are recorded in
  `archive/docs` + `archive/work`; not tracked here.

## Where the current lists live

- qwen4exp carried-forward open items: `beta/qwen4exp/README.md` ("Open
  items (carried forward from WIP)") — the authoritative list for the
  beta tree.
- Delivery verification contract + dated records: `MANIFESTS.md`,
  `patches/README.md`, AGENTS.md headers.
- Benchmarks + gates: `benchmarks/` (`mtp-adaptive-methodology.md` etc.).
