# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.
Forward-looking: open items + current active experiments; closed work is a
one-line bullet (details live in AGENTS.md, patches/README.md, MANIFESTS.md,
`beta/qwen4exp/README.md`, `wip/` handovers, `benchmarks/`). Current
delivery = the 13-patch set against fork point `9cffdcc80` (blocks 01-13).
Reality pass: 2026-09-10.

## Current active

### Strix Halo (gfx1151): prefill-gap follow-ons after WS4 (WS3 #2/#3)
- WS3 #2 (QSA dense-shortcut below the selection width) DONE + DEFAULT
  ON (2026-09-10, A commits `a2f2a6ceb` ggml fix + `250e48e97` flip;
  beta patches 6-7): the llama-bench multi-ubatch artifact was
  ROOT-CAUSED (llama-bench's sync-free decode pipeline x ggml-gallocr's
  single-layout alloc-fallback doing an unconditional full-device sync on
  every dense/sparse topology flip — drains the whole ~3 s GPU queue)
  and FIXED at the ggml level: `ggml_gallocr_reserve_n_probe()` lets the
  sched fallback sync only when a buffer must actually GROW (buffers are
  grow-only; a fitting reserve just re-points tensors, stream-ordered,
  no sync needed). Same-session r3: depth-0 ON >= OFF at every size
  (was -17/-29/-36% at pp4096/8192/16384; now +1.3-3.2%), tg128@0 25.25
  vs 24.23 (+4.2%), depth 12k/32k rows flat, zero fallback syncs in the
  steady state; OFF-path == known-good text, ON == SPARSE_FA=0 dense
  reference, multi-ubatch deterministic. `LLAMA_QSA_DENSE_SHORTCUT` =
  0 forces the pre-flip selection path (known-good numerics); unset/=1 =
  ON (B parity). Record:
  `benchmarks/2026-09-10-strix-halo-gfx1151-ws3-shortcut-fix.md`.
- WS3 #3 (routed-compact MoE mmq for the i-quants) DONE (2026-09-08, A
  commit `1da01fa67`, DEFAULT ON): port of B's
  mul_mat_q_routed_compact + per-expert J selection
  (mmq_rdna3_5_id_get_J, 16/48/64/128 by rows-per-expert, gfx1151-tuned)
  into A's mmq path; RDNA3.5-only gate (B parity; gfx1201 stays off until
  the delivery flow's gfx1201 box validates). Bit-exact by construction
  (same process_tile) + text-verified (7-tok and 4572-tok pp + 40 decode
  identical on/off/known-good); rocprof confirms compact fires on the IQ
  expert GEMMs (IQ3_S J64 x184, IQ4_NL J64 x86, IQ4_XS, Q8_0 J48 @
  pp2048). Same-session: compact adds +2.4-5.3% over plain-at-same-J on
  every depth-0 row, tg flat, pp@d12288 +1.8%; A-vs-B gap moved pp512
  2.05->1.61x, pp1024 2.04->1.78x, pp2048 2.21->2.01x, pp4096 1.68->1.53x,
  pp8192 1.37->1.26x, pp16384 1.14->1.06x. Env opt-out:
  GGML_CUDA_DISABLE_MMQ_ROUTED=1 (compact only; J selection stays). Record:
  `benchmarks/2026-09-08-strix-halo-gfx1151-ws3-routed-moe-mmq.md`.
- Remaining on the Strix prefill leg: (a) gfx1201/RDNA4 validation of
  the ggml fix (patch 6) + WS3 #3 gate (patch 5) via the delivery flow
  when a gfx1201 box is available (multi-GPU/pipeline-parallel not
  exercisable here; the no-sync ordering argument holds per-device but
  should be re-checked there); the ggml fix is also a candidate upstream
  PR at the maintainer's discretion (core-ggml, arch-agnostic); (b) the
  per-ubatch routing/reduction tail (B's weighted-expert-sum/concat
  graph fusions; A's tail unfused outside WS4 hyperconn coverage) — not
  yet requested; (c) re-derive the A-vs-B gap on the new default (B
  already defaults the shortcut ON, so depth-0 rows are now regime-)
  matched). tg@0 decode gap (24.2 vs 26.0) is the later generation
  phase. WS6 re-base NOT indicated.
### Validate beta qwen4exp on Strix Halo (gfx1151) — DONE for WS4; beta set runs on gfx1151
- Block 13's Strix leg is DONE (2026-09-05): the fused MoE gate+up+GLU
  MMQ (RDNA4-gated fused arm) was ungated for RDNA3_5 / gfx1151,
  validated (pp2048 +5.3%, pp16384 +4.6%, coherence IDENTICAL, decode
  unchanged; RDNA4-tuned J caps transfer — an uncap probe regressed),
  and folded into patch `0013` (delivery regenerated at `9cffdcc80`,
  clean-apply sim + full build + coherence re-verified on the Strix
  box). Record:
  `benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
- The beta/qwen4exp set now RUNS and is gated on Strix Halo: patches
  1-3 are the code base every Strix pp/decode number in this campaign
  was measured on (~/llama.cpp `qwen4exp` = beta base + block 13
  fold), and patch 4 (WS4) was validated + gated there 2026-09-06 (see
  the Closed bullet). The old "Requires ROCm gfx1201" README note is
  dropped (see `beta/qwen4exp/README.md`); remaining Strix work is the
  prefill-gap follow-on section above.
- Standing gate before shipping any decode/fusion change:
  `benchmarks/mtp-adaptive-methodology.md` (MTP baseline + acceptance).
- Where: `beta/qwen4exp/README.md` (gates + carried-forward open items).

### Parallel (community member): dual 7900XTX (RDNA3, gfx1100)
- Block-12 (hybrid HIP all-reduce) validation on RDNA3 pairs is ongoing
  with a community member on their dual-7900XTX box: hybrid-dispatch
  matrix (internal vs nccl vs none) + bounded-spin path at depth-16384.
  The block-12 gate stays RDNA4-only until verified, then the arch check
  is removed. Volunteer test env: `GGML_CUDA_ALLREDUCE=internal`.
- The block-13 RDNA3.0/gfx1100 leg is DONE (2026-09-05, single-GPU 7900
  XTX validation — see the Closed bullet); what remains here is block 12
  only. The single-GPU result means block 12 stays N/A on this box (no
  all-reduce path) — still tracked for the dual-7900XTX parallel task.
- Where: `patches/0012` + the block-12 notes in `patches/README.md`.

## Open follow-ups

### Strix: MoE topk-moe fusion is numerics-divergent for qwen4exp (quality-gate decision)
- FINDING (2026-09-06, record `benchmarks/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`):
  A's MoE routing full-512 argsort per token (94 x 0.264ms = 25ms/capture ~0.5% wall) is the
  launch ledger's biggest TIME item. The CUDA topk-moe fusion that would replace it with a
  partial top-10 (B's path, 96 x 0.022ms) is byte-identical in both trees but A's newer
  ggml_cuda_check_fusion_memory_ranges CORRECTLY refuses: the gallocr aliases the fused output
  (ffn_moe_weights_norm) into the dead ffn_moe_logits buffer -> multi-block read/write race at
  2048 rows. Pinning the logits (ggml_set_output) unlocks it (-26ms/capture, ~188MB held).
- NUMERICS FORK: the fused topk is NOT transparent for qwen4exp - top1 logit 18.424 (A unfused)
  -> 18.690 (A fused); B = 18.086. All three diverge. The kernel's internal softmax/top-k/weights
  differ from the plain ggml chain at 0.27 logit scale (behavioral, not ulp). Adoption needs a
  quality gate (CPU reference + PPL/KL) to establish which routing behavior is correct. NOT
  adopted; pin experiment reverted; tree clean at 376f02aa0.
- (b) CLOSED 2026-09-06 (fork f5ac11903, patch 21, record scale-unary-fusion): the +380
  unary-silu launch excess = A's missing scale->unary fusion (no upstream model has qwen4exp's
  hc gate silu(x/hc), so the try_fuse refactor dropped B's peek-ahead). ggml_cuda_op_scale_unary
  ported + 2-node window after the big hc windows; BIT-IDENTICAL (elementwise, in-place-safe,
  no mem gate); pp2048 +0.34%, pp512 +0.42%; kernels 7755->7565. Remaining: +38 scale_f32 +
  fusion-surface diffs (architecture).
- Open leads: (a) quality-gate the fused topk and adopt if it validates (~0.5% + B-alignment);
  (c) GDN +72 launches (2-kernel split) cosmetic post-NW16.

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

- 2026-09-06 WS4 Strix Halo (gfx1151) gates PASSED on the final
  qwen4exp build (branch tip 248e47704 = beta base + the WS4 commit):
  prefill hyperconn fusions DEFAULT ON vs `GGML_CUDA_DISABLE_HC_FUSION=1`
  (same build, clean warm-clock r3) — depth-0 pp +5.2-8.8% across
  pp512..16384 (pp2048 349.5 vs 321.2; pp16384 527.8 vs 490.5), depth
  12k/32k pp rows keep +4-6% (pp2048 @d12288 321.8 vs 308.6; @d32768
  311.4 vs 297.5), tg@depth flat (decode untouched: 22.04 vs 22.12
  @12k; 20.10 vs 20.01 @32k), memory stable −r3 through 32k, llama-cli
  same-seed text on == off. Routed: `beta/qwen4exp/ws4-hc-prefill-fusions.patch`
  (4th beta patch; clean `git apply` at the beta base → applied tree
  byte-identical to `248e47704`). Record:
  `benchmarks/2026-09-06-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
- 2026-09-06 Real determinism root cause FIXED (folded into the same
  qwen4exp commit/patch): the indexer top-k atomicAdd gather scrambled
  the QSA list ORDER run-to-run (>1 block/row) — replaced with an
  ascending-column count/scan/write (fresh-process harness + llama-cli
  now bit-identical). Also ported halo-box `aad5adb08` kv-cache
  stale-cell zeroing (cross-request cell reuse).
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
- 2026-09-05 Block 13 fused MoE MMQ ungated for RDNA3_0 (gfx1100 / RX
  7900 XTX) and folded into patch `0013` (canonical rebuild tip
  `8c2ace510`): validated on this single-GPU 7900XTX box (Qwen3.6-35B-A3B
  True-Q3_K_M, ub 2048, `HIP_VISIBLE_DEVICES=0`) — fusion fires,
  coherence IDENTICAL fused-on vs off, pp2048 +9.4% (5405 vs 4939),
  pp16384 +7.8% (4487 vs 4162), decode unchanged (tg128 130.3); RDNA4 J
  caps transfer (uncap probe regressed 5405 -> 4819, below the 3-op
  fallback; Q3_K@96 also lost to 64). Delivery regenerated at
  `9cffdcc80` (0001-0012 header-only churn); clean-apply sim + full
  build + coherence + perf re-verified on this box. Single GPU => block
  12 stays N/A here; the dual-7900XTX block-12 leg remains the parallel
  task. Record:
  `benchmarks/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
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
