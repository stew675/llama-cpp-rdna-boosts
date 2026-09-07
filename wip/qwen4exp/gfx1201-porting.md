# gfx1201 porting — implementation plan + live worklog (qwen4exp completion stretch)

Status: ACTIVE WORKLOG (live doc — append dated entries at the bottom; tick checkboxes).
Date created: 2026-09-06 (post-reboot session; sched-gate fix `c63f7f2a0`/`d6eb551` closed).
Scope: the final long stretch of the qwen4exp campaign — port/tune/validate the
gfx1151-gated (Halo-campaign) items on the gfx1201 box, close cross-arch coherence, then a
lighter RDNA3 (gfx1100) follow-up via an env-level opt-in ungating.  This file is the
authoritative live tracker for that work; `TODO.md` + `beta/qwen4exp/README.md` point here.

## Machine map (who runs what)

| box | IP | GPU | role in this stretch |
|---|---|---|---|
| soar | 192.168.50.100 | 3x R9700 gfx1201 (RDNA4), 34 GiB each | THE gfx1201 dev box (this plan's primary target) |
| halo | 192.168.50.101 | Strix Halo 8060S gfx1151 (RDNA3.5), 124 GiB unified | gfx1151 canonical reference (campaign box); re-run for cross-arch coherence only |
| radiance | 192.168.50.103 | Strix Halo gfx1151 | spare gfx1151 |
| fingon | 192.168.50.102 | single RX 7900 XTX gfx1100 (RDNA3.0), 24 GiB + 30 GiB RAM | the RDNA3 follow-up (light validation; small models only) |

Fork: `~/llama.cpp` branch `qwen4exp` = master `465e49b9c` + blocks 01-13 (`c261553a1`) +
consolidated beta + sched-gate fix → tip `c63f7f2a0`, then Phase-1 ports on top (1.1 →
`76411193a`).  Delivery repo (this repo): 13 patches +
`beta/qwen4exp/qwen4exp-support.patch`, delivery tip local `f7c7a35`.
Halo's gfx1151 builds of the same content: `/tmp/val-master/build` (pre-fix campaign build,
content == `f5ac11903`) and `~/llama-delivery/build-gated` (gated `c63f7f2a0`) — both
verified this session; see `beta/qwen4exp/HALO_HANDOFF.md`.

## North star / reference semantics (maintainer-confirmed 2026-09-06)

- The **gfx1151-generated output is the canonical numerics reference** for this fork; the
  gfx1201 port converges on it once complete (accumulation ordering may differ from base CPU
  llama.cpp — do NOT use CPU output as the coherence oracle).
- Coherence oracle = gfx1151 build of the same delivery state (byte-identity via same-seed
  llama-cli / the logitcmp family), NOT the pre-re-base builds (their qwen4exp text differs
  by the upstream GDN-normalization fix `5fdfa6282` — a numerics change that shipped in the
  `465e49b9c` re-base; see `beta/qwen4exp/README.md` 2026-09-06 record).

## Baseline captured this session (2026-09-06, soar, gfx1201, fixed build `c63f7f2a0`)

- IQ4_XS Qwen3.8-Flash-Next (`/models/Qwen3.8/Flash-Next/IQ4_XS/...-00001-of-00003.gguf`),
  clean rebooted box, UNPINNED, `HIP_VISIBLE_DEVICES=0,1,2`, hybrid AR default:
  - tensor split pp8192 ub2048: **2130-2156 t/s ×3 (no hang)**; decode tg128 **49.1 t/s**.
  - layer split pp8192 (QSA on, pre-reboot/this-session records): ~1650-1700 t/s.
  - old pre-re-base control build (~/llama-cpp-qwen4exp-old `d4c8e66ae`): pp8192 tensor
    ~445 t/s (not a comparable reference — different master/blocks/beta + pre-GDN-norm-fix
    numerics; kept only for hang A/B history).
- GAP vs the halo campaign ladder: no gfx1201 same-session full ladder (pp512..16384 depth-0
  + depth rows + QSA/HC-fusion toggles) exists for the consolidated delivery yet — that is
  Phase 2 below.  The pp8192-only numbers above are the "before" anchor.

---

## PHASE 0 — gfx1201 harness + pre-port baseline (first)

- [x] **0.1** Write the soar bench/coherence runner (reuse `/tmp/ab-run.sh` pattern; keep it
      under `wip/qwen4exp/gfx1201/` so it survives reboots): VRAM-plateau load detection,
      hang detection (one-GPU-100% + host-100%), pkill bracket pattern, logs to a dated dir.
      → `wip/qwen4exp/gfx1201/bench-run.sh` (watcher + timeout 900 + bracket pkill + dated
      log dir under `runs/`).
- [x] **0.2** Full pre-port baseline table on the fixed build — DONE 2026-09-06 (soar, 3x
      R9700 gfx1201, IQ4_XS Qwen3.8-Flash-Next, `-ngl 99 -t 15 -r 3 -b 2048 -ub 2048
      -fa on --load-mode none`, prompts descending in one process, UNPINNED, tensor split
      [maintainer's preferred mode]; BF16 KV primary [maintainer's daily config], Q8_0
      secondary, F16 for campaign continuity).  Depth-0 t/s:
      | row | f16 | bf16 | q8_0 |
      |---|---|---|---|
      | pp16384 | 2274.27 | 2211.17 | 2230.66 |
      | pp8192 | 2264.79 | 2225.29 | 2258.26 |
      | pp4096 | 2174.07 | 2197.99 | 2247.06 |
      | pp2048 | 2026.21 | 2149.14 | 2268.76 |
      | pp1024 | 1797.52 | 1894.74 | 1969.22 |
      | pp512 | 1434.21 | 1489.75 | 1527.39 |
      | tg128 | 50.07 | 50.04 | 48.75 |
      Depth spot (bf16, r1): pp2048@d12288 **1765.65**, tg128@d12288 **42.34**.
      Continuity vs pre-re-base gfx1201 record (q8_0: pp512 1538 / pp8192 2024 / tg128
      45.7): pp512 1527 ✓, tg128 48.75 ✓, **pp8192 2024 → 2258 (+11.5%)** — the
      model-level campaign gains (QSA shortcut default + fusions) now show on gfx1201.
      Note: prefill rises with prompt length (pp512 1434 → pp16384 2274) — QSA sparse +
      multi-ubatch pipelining.  Raw logs: `wip/qwen4exp/gfx1201/runs/p0-*.log`.
- [x] **0.3** Coherence fingerprint gfx1201 vs gfx1151 canonical — DONE 2026-09-06 (bf16 KV,
      same-seed llama-cli, both fixed build c63f7f2a0): **TEXTS DIVERGE** — same opening
      reasoning tokens, then mid-trace divergence; final differs (gfx1151 "Paris" vs gfx1201
      "Could complete sentence.").  The cross-arch convergence the port must deliver is NOT
      yet satisfied at the pre-port state — this is the Phase-3 baseline gap (per-op
      bisect of which arch-gated path first diverges once Phase-1 ports land).  Logs:
      `wip/qwen4exp/gfx1201/runs/coh-*` + halo `/tmp/halo-ab/coh-bf16-halo.txt`.
- [x] **0.4** Record the baseline — this worklog entry + raw logs under
      `wip/qwen4exp/gfx1201/runs/` (2026-09-06).

## PHASE 1 — Kernel/config ports (RDNA3_5-gated → RDNA4 enable + tune + validate)

Each item: currently **inert on gfx1201** (arch gate), i.e. gfx1201 runs the pre-campaign
RDNA4 paths.  Cross-arch lesson from the campaign (block-13 relaxations): **tuned J caps /
config rows do NOT transfer across arches — "uncapping regressed" on both gfx1151 and
gfx1100**; expect RDNA4 needs its own sweep, not a copy.  All items bit-exact-by-construction
where the kernel reuses the same tile math → coherence check is about perf-path correctness,
and A/B is same-session on/off (or vs the pre-port build).

- [x] **1.1 Routed-compact MoE MMQ** (beta patch 5 / `mul_mat_q_routed_compact` +
      `mmq_rdna3_5_id_get_J` per-expert J tables, `ggml/src/ggml-cuda/mmq.cuh:1677/1758/1767`,
      gate `GGML_CUDA_CC_IS_RDNA3_5`).
      DONE 2026-09-06 (fork `76411193a`): arch gate relaxed to RDNA3_5 || RDNA4
      (`mmq_routed_compact_arch_ok`); env-gated probe A/B first, probe stripped, final
      commit = the gate flip.  Results (soar, IQ4_XS, ub2048 tensor bf16, r3 bracket):
      prefill +4-8% (pp512 +8.2 / pp2048 +5.7 / pp8192 +5.0 / pp16384 +5.0), tg128 ~0;
      846 compact launches/pp2048-ub; same-seed text byte-identical compact vs plain at
      pp40 + pp~2000.  J sweep at 40-rpe: J48 == J64 (tie), J128 marginally behind, plain
      at J32 behind — the gfx1151 bands transfer for the reachable bands (16/48/64); the
      >64-rpe band is unreachable at ub2048 on this 512-expert model, J=128 kept.  Record:
      `wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-rdna4-routed-moe-mmq.md`.
- [x] **1.2 Quantize mmq-q8_1 chunk** (`ggml_cuda_quantize_mmq_q8_1_n_chunks`,
      `quantize.cuh:26-27`, hard gate `cc == RDNA3_5 + 1` → n_chunks=2; gfx1201 = 1).
      DONE 2026-09-06 (fork `90ea7e22e`): **flat on RDNA4 — keep n_chunks=1**.
      Env-gated probe (2x chunk on gfx1201), same-session r3 brackets: pp8192 2482.4 /
      2487.4 / 2462.0 (+0.2% over OFF mid), pp16384 2414.8 / 2408.9 / 2385.4 (ON sits on
      the drift line), pp2048 noisy but ON==OFF2.  gfx1201's block dispatcher does not
      share gfx1151's small-block dispatch bottleneck (where the 2x chunk measured ~1.5x
      per call), so halving the quantize block count does not pay.  Bonus code-quality
      fix folded into the same commit: `cc == GGML_CUDA_CC_RDNA3_5 + 1` (fragile — only
      equals gfx1151 because 0x1150+1 == 0x1151) → `GGML_CUDA_CC_IS_GFX1151(cc)` exact-cc
      macro in `common.cuh` (OFFSET_AMD + 0x1151, decoupled from the family base).
- [x] **1.3 split_j Q8_0 mma specialization + B-parity Q8_0 config rows** (`mmq-vec-dot.cuh`
      split_j constexpr; `mmq-config-rdna3-5.cuh` vs `mmq-config-rdna4.cuh`).
      DONE 2026-09-06: **keep RDNA4's native I=128 — do NOT port the I=64 split_j
      geometry**.  Scope probe (env J-log, then removed): dense Q8_0 J=128 is the dominant
      Q8_0 shape (2838/2862 launches at pp2048, 22704 at pp16384; MoE Q8_0 J=48 negligible)
      — the question was load-bearing.  A/B on the RDNA4 Q8_0 J=128 row (I 128 native vs
      64 split): pp16384 FLAT (2383.5 vs 2394/2379), pp8192 -0.5% (2449 vs 2459/2462),
      pp2048 -2..-3% (2607 vs 2658/2698; tight σ on both sides).  gfx1201's register file
      handles the I=128 profile without the gfx1151 spilling (232->136-vgpr win does NOT
      transfer) — the wider-row geometry wins on RDNA4.  Fallback J=128 row (rdna3-5
      nthreads128/I64 vs rdna4 nthreads256/I128) left as-is (fallback rarely fires here;
      needs nrows_x%128 != 0).
- [x] **1.4 mmq accumulator-overflow latent-defect audit on the RDNA4 table** (TODO 3d;
      the `I < nwarps*16` mma sum[] overflow): audit every row in `mmq-config-rdna4.cuh`
      for `I >= nwarps*16`; feed the upstream defect report (no perf value, correctness
      hygiene).
      DONE 2026-09-06: **rdna4.cuh is CLEAN** — no row violates `I < nwarps*16`
      (warp=32 → I < nthreads/2); every row sits exactly at the boundary.  Same for
      rdna2/rdna3 (243-260 rows each).  The ONLY violator across the arch tables is the
      rdna3-5 Q8_0 J=128 I=64 non-fallback row — deliberate + guarded by the split_j
      machinery in BOTH the accumulator (mmq-vec-dot.cuh) and write-back (mmq.cuh) arms.
      Upstream report content: state the invariant (config rows must keep
      I >= nwarps*16, warp-32 RDNA) + recommend a static_assert in the CASE macro;
      rdna3-5's single violator is handled in-tree.  CDNA tables need their own warp-64
      audit if included in the report scope.
- [x] **1.5 GDN gfx12 chunked kernel — post-consolidation re-validation**: gfx1201 uses the
      gfx12 kernel in `gated_delta_net.cu` (NOT the gfx11 first-gen-WMMA file the campaign
      retuned).  The 0002 chunked-prefix dispatch was validated on 3x R9700 **pre-re-base**;
      re-run the chunked-prefix A/B (`GGML_CUDA_GDN_CHUNKED=0` opt-out) on the current
      delivery + MTP/depth rows per `benchmarks/mtp-adaptive-methodology.md`.
      DONE 2026-09-06 (no code change): pp8192 **+8.1%** (2484 vs 2298), pp16384 **+8.0%**
      (2400 vs 2222) on the current delivery (3-GPU tensor, bf16, r3, bracketed); depth
      pp2048@d12288 **+6.6%** (2035 vs 1908, r1); tg@d12288 unchanged (42.7 both) — decode
      GDN is sequential regardless, so the MTP acceptance gate is unaffected by construction
      (and the chunked path is bit-exact: long-prefill same-seed text BYTE-IDENTICAL ON vs
      OFF).  MTP decode-side acceptance remains covered by Phase 2.3.
- [x] **1.6 Per-file RDNA3_5 config-row audit** (fattn / mmf / concat / mmvq / mmid / vecdotq
      RDNA3_5 references): classify each as (a) RDNA3_5-only row (leave; RDNA4 has its own
      pre-campaign rows) vs (b) a "generality" finding that should carry to RDNA4.
      DONE 2026-09-06 (classification; per-item relax/validate deferred to Phase 2):
      (a) leave (RDNA4 has its own rows): fattn.cu:666 `wmma_max_head` (RDNA4 576 already,
      its own cap); vecdotq.cuh:247 `VDR_Q8_0_Q8_1_MMVQ` (RDNA4 already VDR=4, measured);
      mmq.cu:587 IQ2_XS/IQ2_S mmq-vs-hipblas rule (RDNA3_5 always-mmq; others ne11<=128 =
      upstream; inert for IQ4 workloads); ggml-cuda.cu:4399 moe_mmq fused gate+up+GLU
      K-quant path (ALREADY RDNA4 — block 13, caps tuned on gfx1201).
      (b) carry candidates — RDNA3_5-gated TODAY on gfx1201, need a relax-probe + A/B
      (mostly Phase 2 scope): concat.cu `concat_transposed_tile_y` 16-vs-8 (memory-bound;
      likely small given 1.2's dispatcher finding); ggml-cuda.cu:3808 swiglu fused MMQ for
      IQ4_NL/Q8_0 gate+up (qwen4exp 640/2560 shape; the K-quant sibling 4399 is already
      RDNA4); mmid.cu:209 `mm_ids_helper_512_10` (our 512/10 shape runs the generic path on
      gfx1201); mmvq.cu:2622 weighted decode expert-sum
      (`mul_mat_id_iq4_nl/q8_0_weighted_rdna3_5`, 640/2560/512/10) — gfx1201 = UNFUSED
      decode weighted-sum today.
      → Phase-2 DEPENDENCY: the 2.1/2.3 toggles for MMID_512 / WEIGHTED_DOWN / the
      IQ4-NL-swglu + weighted-sum paths only measure the OFF-state on gfx1201 unless the
      gates are relaxed first (Phase 2.0 relax-probes, env-gated, before the 2.1 ladder).
      Arch-agnostic model fusions (QSA, hc hyperconn — opt-out-only) DO fire on RDNA4.

## PHASE 2 — Consolidated-beta model-level validation on gfx1201 (same-session ladder)

The beta content (QSA shortcut default, PLE host-gather, weighted-down, hc hyperconn
fusions, repeat-absorb, mmid/mwr ports, managed-ngrams, MTP draft head) was gfx1151-gated;
gfx1201 has only pre-re-base records (QSA decode fix etc.).  The sched-gate fix (this
session) made the ggml layer safe multi-GPU; now validate the model level end-to-end.

- [ ] **2.0** Phase-2 relax-probes (from the 1.6 audit; env-gated, adopt-only-if-wins):
      swiglu→mmq fusion (3808) = DONE 2026-09-06: does NOT transfer (slight loss, keep
      RDNA3_5-only, see the 2.0.1 entry below); weighted decode expert-sum (mmvq.cu:2622)
      = DONE: does NOT fire on the IQ4_XS subject (0 fusions at tg; its down tensors aren't
      the Q8_0/IQ4_NL routed pattern) — RDNA4 relax UNTESTED, needs a Q8_0-down decode
      model; mmid 512_10 (mmid.cu:209) = DONE: **WINS, LANDED** (fork `b298cbe7f`, +4.1%
      pp2048 / +2.6% pp512 tight-window); concat tile_y 16 = DONE: flat (+0.4/+0.2%,
      noise) — keep 8.  Phase 2.0 COMPLETE.

- [x] **2.1** Depth-0 ladder + tg (tensor AND layer split) on the fixed build, IQ4_XS + MTP
      draft (Q4_K_M mtp model), same-session toggles where they exist: `LLAMA_QSA_OFF=1`,
      `GGML_CUDA_DISABLE_HC_FUSION=1`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`,
      `GGML_CUDA_DISABLE_MMID_512=1`, `LLAMA_QSA_DENSE_SHORTCUT=0`.  Record ON-vs-OFF deltas
      (expect the gfx1151-validated fusion gains to reproduce — the kernels are model-level).
      DONE 2026-09-06 — toggle deltas (tensor, bf16; window-matched controls): default
      anchor pp512 1929 / pp2048 2973 / pp8192 2619 / pp16384 2528 / tg128 50.6.  QSA_OFF:
      prefill crossover >16K ctx (dense slightly ahead ≤16K, sparse +14.5% @32K, +48%
      @64K); sparse DECODE loses at depth (dense +20% @32K / +29% @64K, flat at d0) → new
      gfx1201 QSA-decode tuning package (OPEN, see the 2.1 entry + QSA-depth discovery
      record).  DENSE_SHORTCUT=0: -15.5% pp2048.  HC_FUSION=1: -15.8% pp2048 / -14.3%
      pp8192 (gfx1151 gains reproduce).  MMID_512=1: -4.1% (2.0.3).  WEIGHTED_DOWN=1:
      no-op control ✓.  Layer-split anchor: pp2048 2238 / pp16384 1710 / tg128 37.2.
      QSA depth interleaves double as partial 2.2 depth rows.
- [x] **2.2** Depth rows (12k/32k, r1) + memory stability −r3 through 32k.
      DONE 2026-09-07 — default tensor bf16 current-state ladder (1.1+mmid in):
      pp2048/tg128 @d8192 2246/43.8, @d12288 2198/43.0 (+24.5% pp vs Phase-0 pre-1.1 1765.7),
      @d32768 2038.3±34/39.38±0.37 **-r3 memory-stable** (no OOM/drift across reps;
      matches the 2.1 d32768 r1 1971/39.7 within window), @d65536 1702/35.1 (2.1 r1).
      Smooth degradation, no cliff: pp 2246→2198→2038→1702, tg 43.8→43.0→39.4→35.1.
- [x] **2.3** Decode leg: tg128/512, MTP acceptance per `benchmarks/mtp-adaptive-methodology.md`
      (acceptance must stay > ~0.45, MTP >= plain at depth 3), server smoke (user config,
      flat decode through 4k gen — the gfx1201 QSA decode fix pre-dates the re-base).
      DONE 2026-09-07 — tg512: 52.2±0.9 @d0 (r3) / 44.2 @d12288.  Protocol A (IQ4_XS main +
      IQ4_XS/mtp Q4_K_M draft, seed 42 temp 0, prose): plain 49.9 → draft-mtp 53.1 t/s
      (+6.4%), acceptance profile per-pos (0.606, 0.317, 0.154) — pos-1 0.606 > the 0.45
      bar, mean len 2.08, NO 0-collapse → draft-vs-verify numerics consistent (512-expert
      Flash-Next draws shorter drafts than the 128-expert A3B's 0.51/2.9, hence the modest
      +6.4%).  Server smoke (3-GPU tensor, bf16, hybrid default): 4096-token greedy gen at
      flat 48.8-49.0 t/s sustained (tg_3s 45.6 at the deepest point), graphs reused,
      0 faults/hangs → the pre-re-base QSA decode fix holds on the re-based build.
      Note: raw /completion prose without the chat wrapper stops at 1 token on this model
      (argmax = stop) — wrap `<|im_start|>…<|im_end|>\nassistant` for server gens.
- [x] **2.4** Multi-GPU-specific: hybrid-AR + RCCL toggles (`GGML_CUDA_ALLREDUCE=nccl` /
      default hybrid / issue-13 fallback), the tensor-split + MTP-verify path, ubatch-2048
      multi-chunk prefill determinism (the sched-gate's original failure mode — confirm the
      fixed build is deterministic run-to-run, same-seed, tensor split).
      DONE 2026-09-07 — 4000-token prompt (2+ ub2048 chunks) × -c 32768, seed 42 temp 0,
      3-GPU tensor bf16: hybrid 3/3 runs byte-IDENTICAL (sched-gate fix holds — no
      multi-chunk nondeterminism); RCCL 2/2 byte-identical (also internally deterministic);
      hybrid-vs-nccl deterministic-but-DIFFERENT mid-reasoning (AR op-order numerics drift —
      internal-pipeline vs RCCL-tree summation; same family as the cross-arch + fusion-order
      drift the MoE methodology accepts; the dense 27B identity gate (2026-09-04) does not
      transfer to the topk-10 MoE at long prefill).  Open with the maintainer: require MoE
      hybrid-vs-nccl byte-identity or accept as a numerics regime.  Issue-13 fallback:
      not triggerable on this healthy box (covered by the block-12 amendment validation on
      the 27B).  Tensor-split + MTP-verify path: exercised by the 2.3 Protocol A run.
      Side-finding: llama-cli with the DEFAULT small ctx (4096) + a near-ctx prompt aborts
      at `ggml-backend-meta.cpp:1758 GGML_ASSERT(bufs.back() != nullptr)` on 3-GPU tensor
      (meta-buffer alloc edge; -c >= prompt len avoids it; llama-bench unaffected).
- [ ] **2.5** Regression check for the gfx1201 fallback paths: with the RDNA3_5 kernels
      inert, confirm the mmq.cuh/mmq-config refactors did not perturb the RDNA4 id-MMQ path
      (this is implicitly covered by 2.1 vs the pre-port numbers; call it out explicitly).

## PHASE 3 — Cross-arch coherence (gfx1201 converges on the gfx1151 canonical)

- [ ] **3.1** Establish the cross-box logitcmp/fingerprint flow (soar ↔ halo): 838-token
      fingerprint family on both boxes, same delivery state.  Run before/after each Phase 1
      kernel port that could touch accumulation order.
- [ ] **3.2** End state check: gfx1201 same-seed output == gfx1151 output for the depth-0
      and depth rows used by the campaign ladder (byte-identical text; logitcmp at 9dp).

## PHASE 4 — RDNA3 (gfx1100) follow-up: env-level opt-in ungating (open design)

Context (maintainer 2026-09-06): the gfx1100 box (fingon) is a single 24 GiB GPU — the 87 GiB
IQ4_XS model cannot load there; validation is lighter and uses small models (Qwen3.6-35B-A3B
True-Q3_K_M ~16 GiB / Q8_0 35 GiB-class).  OPEN QUESTION: how to test qwen4exp support on
RDNA3.  Working plan: an **environment-level opt-in that un-gates the gfx1151 work for
gfx1100/RDNA3** so community members with enough RDNA3 capacity can opt in; otherwise RDNA3
falls back to llama.cpp's own (slower) qwen4exp support.  Design considerations below are
provisional — validate with the maintainer before implementing.

- [ ] **4.1** Env-gate design: a single opt-in knob (e.g. `GGML_CUDA_QWEN4EXP_RDNA3=1` or a
      generalized `..._UNLOCK_RDNA3_5_FOR_RDNA3=1`) evaluated at runtime where the campaign
      code tests `GGML_CUDA_CC_IS_RDNA3_5(cc)`; when set AND `cc` is RDNA3_0 (gfx1100/gfx1101),
      the RDNA3_5 arm fires.  Prefer a macro + runtime flag over compile-time so one binary
      serves both modes; per-kernel safety gates stay (see 4.2).
- [ ] **4.2** Safety audit of WHAT gets un-gated — not everything gfx1151-validated is safe to
      un-gate blind on gfx1100:
      - validated on gfx1100 already (safe to include): block-13 fused MoE gate+up+GLU MMQ +
        J caps (RDNA4 caps transfer — 2026-09-05 record), the decode-side mmvq tables
        (block 10 RDNA3_5 params; RDNA3_0 shares the family), QSA sparse-FA decode path.
      - needs gfx1100 checks before ungating: GDN gfx11 NW16 scan retune (~106K VGPRs/CU
        needed vs possibly 64K classic on gfx1100 → revert to NW8 constants there if it
        fails — deferred ledger item), split_j/config rows, quantize chunk, routed-compact
        (gfx1100 has no campaign data), hc hyperconn fusions + PLE/weighted-down (model-level;
        likely fine but only the block-13/QSA items were measured on gfx1100).
      - the two MTP regression fixes' gating (fold gated to single-token MMID etc.) must be
        re-checked under RDNA3 (MTP acceptance gate).
- [ ] **4.3** Short fingon campaign (light): build the delivery for gfx1100
      (`AMDGPU_TARGETS=gfx1100`, ROCm `/opt/rocm-7.14-gfx1100`); Qwen3.6-35B-A3B True-Q3_K_M
      ladder + coherence vs the gfx1151/gfx1201 canonical; then the env-opt-in A/B for the
      audited kernel set.
- [ ] **4.4** Docs + community story: how to opt in, what is/is not validated on RDNA3, the
      fallback statement (default = llama.cpp plain qwen4exp support).

## PHASE 5 — Carried-forward items that surface again at port time (not arch-gated)

Tracked from TODO.md open follow-ups (they are model-level; when/if adopted they need gfx1201
validation too):
- [ ] **5.1** MoE topk-moe fusion quality gate (~0.5% prefill; numerics fork 18.424/18.690/
      18.086 — CPU-ref + PPL/KL gate before adoption; gallocr aliasing needs the logits pinned).
- [ ] **5.2** ssm_alpha+ssm_beta single-walk fusion (~0.3-0.6%; blocked on graph order —
      stacked weights or qwen4exp graph restructure; design in the cijk record).  PARKED.
- [ ] **5.3** Launch-ledger remainder (+38 scale_f32 etc.) — root-cause-only value.  PARKED.
- [ ] **5.4** mmq latent defect → upstream report (ties to 1.4).

## Delivery/doc closeout (after each phase)

- [ ] Fold approved gfx1201 ports into the fork (qwen4exp branch) → regenerate
      `beta/qwen4exp/qwen4exp-support.patch` (or amend the owning block) via the canonical
      flow (`scripts/make-patches.sh` from a rebuilt fork) — never hand-edit patches.
- [ ] Add dated records: this worklog + `benchmarks/` + `beta/qwen4exp/README.md` +
      TODO.md checkoffs; update AGENTS.md headers only if delivery content changes.
- [ ] Re-verify clean-apply sim (fork-point worktree + apply-all + build) before any commit
      that touches the delivery.

---

## Worklog (append newest at the END; one entry per session/subject)

### 2026-09-06 — session start (post-reboot): sched-gate fix closed; baseline captured
- Root-cause recap on record (HANDOVER.md): the gfx1151 campaign's no-sync sched re-reserve
  probe raced cross-GPU at multi-ubatch prefill on multi-GPU.  Fix `c63f7f2a0`/delivery
  `d6eb551`: gate the no-sync path to single-device schedulers (count non-CPU device types;
  >1 → full sync).  Delivery commit already in place; build-rocm binaries contain it.
- gfx1201 (soar) clean-box tensor-split A/B: fixed build **3/3 no-hang** pp8192 ub2048 at
  2130-2156 t/s (pre-reboot flake did NOT reproduce → degraded-box artifact; no bisect, no
  gate change).  Old control build no-hang ~445 t/s (expected gap).  See TODO.md Closed.
- gfx1151 (halo) single-GPU gate validation: same-session parity vs the pre-fix campaign
  build (all rows within ±0.6%) + pure-gate pair (627506c1c vs c63f7f2a0) byte-identical
  text + perf parity.  Details: `beta/qwen4exp/HALO_HANDOFF.md` + README 2026-09-06 record.
- Coherence-semantics note: delivery qwen4exp text differs from pre-re-base builds — upstream
  GDN-norm fix `5fdfa6282` (in the re-base range); gfx1151 canonical is the oracle.
- Upstream-PR materials prepared (task 4): `beta/qwen4exp/UPSTREAM-PR-ggml-sched-probe.{md,
  patch}` — scheduler change isolated to core ggml (+70/−9, applies clean to `465e49b9c`).
- Baseline anchor (this file, above) captured.  Phase 0 TODO list open.
- IQ3_XXS/IQ4_XS shard-1 "truncation" resolved as non-issue (metadata-only 10.9 MB first
  shard; upstream sizes match byte-for-byte).

### 2026-09-06 (session cont.) — PHASE 0 COMPLETE: harness + pre-port baseline on gfx1201
- Harness: `wip/qwen4exp/gfx1201/bench-run.sh` (watcher + hang-safe timeout + dated logs).
- Baseline (fixed build `c63f7f2a0`, 3x R9700 gfx1201, IQ4_XS, tensor split, UNPINNED):
  depth-0 ladder at f16/bf16/q8_0 KV + depth spot — see Phase 0 table above.  Highlights:
  bf16 KV pp512 1489.75 / pp2048 2149.14 / pp8192 2225.29 / pp16384 2211.17, tg128 50.04;
  q8_0 pp8192 2258.26 (vs 2024 pre-re-base, +11.5%); pp2048@d12288 1765.65 (bf16).
- Coherence fingerprint (0.3): gfx1201 vs gfx1151 canonical at bf16 KV, same build — TEXT
  DIVERGES mid-reasoning (gfx1151 "Paris" vs gfx1201 "Could complete sentence.").  The
  cross-arch convergence goal is NOT met at the pre-port state → Phase 3 has a concrete
  starting gap to bisect per-op after Phase 1.  Config note: maintainer's preferred bench
  config = `-sm tensor` + BF16 KV (daily driver); Q8_0 secondary; f16 = campaign continuity
  only.

### 2026-09-06 (session cont.) — PHASE 1.1 DONE: routed-compact MoE MMQ enabled on RDNA4
- Method: env-gated probe (`GGML_CUDA_MMQ_ROUTED_RDNA4` + J override + fire-print) →
  same-session OFF/ON/OFF r3 ladder + J sweep at pp2048 + coherence pairs → probe stripped
  → fork commit `76411193a` (arch test = RDNA3_5 || RDNA4).
- Result: prefill +4-8% (bigger at short prompts), tg128 unchanged, same-seed text
  byte-identical compact vs plain (pp40 + pp~2000).  846 compact launches per pp2048
  ubatch (IQ experts J=64 @ 40 rpe; Q8_0/Q6_K rows J=48).
- J transfer: gfx1151 bands hold on RDNA4 — J48 == J64 at 40 rpe, J128 (I=256) marginally
  behind, plain-at-J32 clearly behind.  >64-rpe band not reachable at ub2048 on this
  model (keeps J=128).  No RDNA4-specific table needed for the reachable bands.
- Record: `wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-rdna4-routed-moe-mmq.md`.
- Note: the Phase-0 table numbers (18:56) ran cooler/colder-cache than the 19:30+ bracket
  (pp2048 2149 vs bracket OFF-mid 2452) — later same-session brackets are the A/B truth.

### 2026-09-06 (session cont.) — PHASE 1.2 DONE: quantize chunk flat on RDNA4 (keep 1); exact-SKU helper
- Env-gated 2x-chunk probe on gfx1201 → same-session r3 brackets at pp8192/pp16384/pp2048:
  FLAT (ON within ±1% of the OFF drift line at every row).  gfx1201's block dispatcher
  does not share gfx1151's small-block dispatch bottleneck (the 2x chunk was ~1.5x per
  call there) → RDNA4 keeps n_chunks=1; decision recorded in the code comment.
- Review fix folded in (maintainer): `cc == GGML_CUDA_CC_RDNA3_5 + 1` was assuming the
  family-base + 1 == gfx1151 (true only via AMD's contiguous gfx numbering) → new
  `GGML_CUDA_CC_IS_GFX1151(cc)` exact-cc macro in common.cuh
  (`cc == OFFSET_AMD + 0x1151`), no comment bloat.  Fork `90ea7e22e`.

### 2026-09-06 (session cont.) — PHASE 1.3 DONE: split_j Q8_0 geometry does NOT transfer to RDNA4
- Scope probe first (env J-log, removed after): dense Q8_0 mmq runs J=128 (2838/2862 Q8_0
  launches at pp2048; 22704 at pp16384) — MoE Q8_0 (J=48) negligible.  So the Q8_0 J=128
  config-row question is the real content of 1.3.
- A/B of the RDNA4 Q8_0 J=128 non-fallback row: I=128 native vs I=64 split_j (the
  gfx1151-winning geometry, vgpr 232->136 there).  gfx1201: pp16384 FLAT, pp8192 -0.5%,
  pp2048 -2..-3% (tight σ both sides; interleaved).  gfx1201 handles the I=128 profile
  without gfx1151's spilling → wider rows win → KEEP I=128.  Cross-arch lesson confirmed
  again: per-arch config rows do not transfer.
- Method note (maintainer): bench runs already pass --load-mode none (lm column = none;
  eager host read, no mmap lazy page-in); added explicit '-lzm off' guidance to
  bench-run.sh so lazy mode is never a variable in any run.

### 2026-09-06 (session cont.) — PHASE 1.4 DONE: rdna4 mmq table is clean (no sum[] overflow rows)
- Programmatic audit of every CASE row in rdna2/rdna3/rdna3-5/rdna4 (warp=32): invariant
  I >= nwarps*16  <=>  I >= nthreads/2.  rdna4: 260/260 rows at or above the boundary —
  no defect (the TODO-3d "likely applies too" suspicion is resolved negative).  rdna2/rdna3
  clean.  rdna3-5's single violator (Q8_0 J128 I64 nt256) = the deliberate split_j row,
  guarded in both arms.
- Upstream defect report (5.4): document the invariant + propose a CASE-macro static_assert;
  note CDNA needs a warp-64 variant if included.

### 2026-09-06 (session cont.) — PHASE 1.5 DONE: GDN gfx12 chunked re-validated on the current delivery
- A/B (bracketed r3, 3-GPU tensor, bf16): pp8192 +8.1%, pp16384 +8.0% chunked vs
  GGML_CUDA_GDN_CHUNKED=0 — matches the pre-re-base +7.5/+7.7% finding; the 0002
  chunked-prefix dispatch survived the re-base + consolidated beta + sched-gate intact.
- Depth leg: pp2048@d12288 +6.6%; tg@d12288 unchanged.  Bit-exactness: long-prefill
  same-seed text byte-identical ON vs OFF.  No code change; MTP acceptance gate
  unaffected by construction (decode sequential both ways) + covered in Phase 2.3.

### 2026-09-06 (session cont.) — PHASE 1.6 DONE: per-file RDNA3_5 row audit (classification)
- (a) leave: fattn wmma_max_head (RDNA4 576 own), vecdotq VDR (RDNA4 4 own), mmq.cu:587
  IQ2 rule (inert for IQ4), ggml-cuda.cu:4399 K-quant moe_mmq (already RDNA4, block 13).
- (b) carry candidates (RDNA3_5-gated today on gfx1201): concat tile_y 16, swiglu fused
  MMQ IQ4_NL/Q8_0 (3808), mm_ids_helper_512_10 (mmid.cu:209), weighted decode expert-sum
  (mmvq.cu:2622).  DEPENDENCY for Phase 2: those Phase-2 toggles measure only the OFF
  state on gfx1201 until env-gated relax-probes land (Phase 2.0).  Arch-agnostic fusions
  (QSA, hc) fire on RDNA4 already.
- PHASE 1 COMPLETE: 1.1 landed (fork `76411193a`, +4-8% prefill), 1.2 landed (refactor,
  `90ea7e22e`; chunk flat), 1.3/1.4/1.5/1.6 = decisions/validations/audit (no code).

### 2026-09-06 (session cont.) — PHASE 2.0.1 DONE: swiglu→mmq fusion does NOT transfer to RDNA4
- Probe (env-gated `GGML_CUDA_SWIGLU_MMQ_RDNA4`, ggml-cuda.cu:3808) on gfx1201: fires
  282x/pp2048 (Q8_0 weights, dense down-feed, 640/2560/512/10 qwen4exp shape); same-seed
  text BYTE-IDENTICAL fused vs unfused (bit-exact by construction, verified); but perf =
  consistent -1.1..-1.6% at every pp row (2824.7->2788.9 @pp2048, 2430->2392 @pp16384,
  sigma < 0.5% both sides), tg unchanged.  gfx1151's win (skip the GLU-intermediate
  round-trip) does not transfer — gfx1201's separate swiglu path wins.  Decision: keep the
  fusion RDNA3_5-only (reverted; tree clean at `90ea7e22e`).  Fused-vs-unfused
  bit-exactness means this does NOT affect the Phase-3 cross-arch convergence target.

### 2026-09-06 (session cont.) — PHASE 2.0 COMPLETE: relax-probes (mmid-512 WINS +4.1%)
- 2.0.2 weighted decode expert-sum (mmvq.cu:2622): env-gated relax + fire check → ZERO
  fusions at tg on the IQ4_XS model (down tensors are not the Q8_0/IQ4_NL routed pattern;
  only the dense PREFILL down-feed is Q8_0 — that is the swiglu site, 2.0.1).  RDNA4 relax
  remains UNTESTED; the fusion targets Q8_0/IQ4_NL-down decode models (Q3_K-A3B class —
  fingon/Phase 4 subject).  Reverted.
- 2.0.3 mm_ids_helper_512_10: env-gated probe showed +4.5-5.2% (fast window) → flip to
  unconditional RDNA3_5||RDNA4 → tight-window interleaved A/B confirms **+4.1% pp2048 /
  +2.6% pp512** vs generic (DISABLE_MMID_512 toggle); text byte-identical.  LANDED fork
  `b298cbe7f`.  (Surprise: a 'small helper' was worth 15x my estimate — the 512-block
  one-warp generic path is a real RDNA4 bottleneck too.)
- 2.0.4 concat_transposed tile_y 16 (RDNA4 probe): pp2048 +0.4% / pp512 +0.2% — flat;
  keep tile_y=8.  Reverted.
- Machine-drift note: box swings +-5-7% across 20-40 min windows this session (a rebuild
  + coherence runs shift it); interleaved brackets (toGGLE-based) are the only sound A/B.

### 2026-09-06 (session cont.) — PHASE 2.1 depth-aware QSA toggle finding (needs a gfx1201 tuning package)
- Depth-0 toggle A/B (tensor, bf16, r3): DENSE_SHORTCUT=0 regresses pp2048 -15.5% (the
  shortcut's top-k-selection-avoidance is worth it at short ctx); MMID_512 disable -4.1%
  (2.0.3); WEIGHTED_DOWN toggle = no-op (never fires on IQ4_XS); tg depth-0 ~flat for all.
- QSA_OFF depth ladder (interleaved r1, tensor bf16):
  | row | sparse (default) | dense (QSA_OFF) | note |
  | pp2048 ctx16k (depth-0) | ~2528-2973 | ~+6.6% pp16384 | dense slightly ahead |
  | pp2048 @ d32768 | 1971 | 1720 | SPARSE +14.5% |
  | pp2048 @ d65536 | 1702 | 1141 | SPARSE +48% |
  | tg128 @ d32768 | 39.7 | 47.5 | DENSE +19.6% |
  | tg128 @ d65536 | 35.1 | 45.3 | DENSE +29% |
  → QSA default-ON is CORRECT on gfx1201 (dense prefill craters at 32K+); the prefill
  crossover sits between ~16K and ~32K context.  BUT sparse DECODE loses at depth (dense
  +20..29% at 32-64K; flat at d0) — the gfx1151 QSA decode tuning does NOT transfer; the
  gfx1201 sparse decode path (indexer scoring + sparse-FA decode kernels per token) needs
  its own tuning package (maintainer: "we got good gains on gfx1151 by tuning it there").
  OPEN: new work item (gfx1201 QSA decode tuning or a depth-regime decision) — parked for
  a Phase-5-style package; record `wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-qsa-depth.md`.
- 2.1 toggle matrix otherwise: default-anchor ladder (tensor bf16 r3) pp512 1929 / pp1024
  2607 / pp2048 2973 / pp4096 2718 / pp8192 2619 / pp16384 2528 / tg128 50.6 (with 1.1 +
  mmid wins vs Phase-0: pp2048 +38%, pp16384 +14%).


## SESSION-END HANDOFF 2026-09-06 (for the next session — read the worklog from the top for full context)
State: fork `~/llama.cpp` qwen4exp = master `465e49b9c` + blocks 01-13 (`c261553a1`) +
consolidated beta + sched-gate (`c63f7f2a0`) + Phase-1/2 kernel ports: 1.1 routed-compact
MoE MMQ on RDNA4 (`76411193a`), exact-SKU gfx1151 macro + quantize-chunk flat
(`90ea7e22e`), 2.0.3 mmid_512_10 helper on RDNA4 (`b298cbe7f`, +4.1% pp2048). Clean
worktree. Delivery local tip `b28fb31` (unpushed; never push from `~/llama.cpp`).

Phase 0/1/2.0/2.1 DONE (details + numbers above); **next = Phase 2.2-2.4**: 2.2 depth rows
(32K/64K r1 already done via the QSA interleaves; add memory stability -r3 through 32K +
default depth ladder); 2.3 decode leg (tg128/512 at depth, MTP acceptance per
`benchmarks/mtp-adaptive-methodology.md` with `/models/Qwen3.8/Flash-Next/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`,
server smoke); 2.4 multi-GPU determinism (same-seed run-to-run, tensor split, ub2048 — the
sched-gate's original failure mode), AR toggles (hybrid default / `GGML_CUDA_ALLREDUCE=nccl` /
issue-13 fallback).

**OPEN work package (confirm direction with the maintainer before implementing):** gfx1201
QSA-decode tuning (sparse decode loses at depth on RDNA4: dense +20% @32K / +29% @64K;
flat at d0; gfx1151's tuning doesn't transfer — indexer store/scoring + sparse-FA decode
kernels) OR a depth-regime default decision. Record:
`wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-qsa-depth.md`.  Also untested: 2.0.2
weighted decode expert-sum RDNA4 relax (needs a Q8_0/IQ4_NL-down decode model, e.g. a
Q3_K-A3B-class subject).

Protocol reminder: model `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`;
`llama-bench -fa on -sm tensor -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -ctk bf16 -ctv bf16
--load-mode none`; env `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib HIP_VISIBLE_DEVICES=0,1,2
RCCL_BUFFSIZE=16777216`; UNPINNED; harness `wip/qwen4exp/gfx1201/bench-run.sh`; box drifts
+-5-7% across 20-40-min windows → toggle-interleaved brackets only. Coherence oracle =
gfx1151 (halo) same-build output, NOT CPU/pre-re-base builds (upstream GDN-norm fix
`5fdfa6282`). Discovery records:
`wip/archive/qwen4exp/discovery/2026-09-06-gfx1201-rdna4-routed-moe-mmq.md` +
`...-2026-09-06-gfx1201-qsa-depth.md`.  Tensor-vs-layer A/B (tensor wins 1.31-1.47×) in
`runs/p1-splitAB-*`.


### 2026-09-07 (cont.) — QSA-DECODE DEPTH ROOT-CAUSE FOUND (maintainer deep-dive)
- Fall-off (tg128, interleaved r1, tensor bf16): dense beats sparse decode from ~2K ctx
  (d2K +11%, d8K +13%, d32K +20%, d64K +29%); sparse falls -31% d0->64K vs dense -11%.
  Prefill inverts (sparse +14.5% @32K / +48% @64K — batched path amortizes).
- Isolation @d32768: sparse-FA == masked-dense-FA with the same top-k build (39.4 vs
  38.1) → the attention kernel is NOT the cost; QSA_OFF dense 47.5 → the ~9 t/s gap = the
  per-token top-k build + mask stack.  Slicing already helps the sparse kernel ~22%
  (QSA_SLICES=1 → 32.4 vs default 39.4).
- Op census: ~15 ggml ops/layer/token above the width (index_k mm → cpy_k → get_rows(all
  raw keys) → r-slice pooling → norm → rope → index_q → score mm → relu → sum → top-k →
  mask fill(-INF) → set_rows → add → FA) vs dense ~1-2.  Every token RE-POOLS the whole
  raw-only indexer cache; prefill amortizes, n_tps=1 decode does not.  Fixed per-layer
  launch stack dominates (d2K already -11% at the width boundary); n_kv-scaled component
  is secondary.  lightning-indexer (fused score kernel) is NVIDIA-WMMA-only.
- Fix directions (record: `2026-09-07-gfx1201-qsa-decode-rootcause.md`): A) fuse the
  decode indexer path into ~1 HIP kernel/layer (the real fix — restores flat fall-off);
  B) fold the per-layer [n_kv] mask into the sparse kernel; C) RDNA4 sparse-FA decode
  geometry (inherit the dense decode tuning); D) store-side fusion (minor).  A+B expected
  to recover most of the ~9 t/s @d32K.  GO/NO-GO with the maintainer before implementing A.


### 2026-09-07 (cont.) — QSA-DECODE FIX: probe landed + fusion design ready (fork `e6b7ae6f0`)
- Env-gated layer-skip probe (GGML_CUDA_QSA_DECODE_SKIP, sparse-layers-only counter, default
  off) @d32768 tg128 tensor bf16: all-sparse 39.7 / half-sparse 42.6 / quarter-sparse 44.1 /
  dense 47.5 → the indexer build cost is ~per-layer additive, ~7.8 t/s total @32K (NOT one
  dominating op).  Fusion ceiling = sparse decode ≈ dense at depth.  (The first probe run
  that skipped odd layers and measured ~47 was a measurement mistake — odd layers are the
  ratio=0 DENSE layers, so it had removed ALL sparse layers.)
- Mask side: verified the build_attn_qsa fill(-INF)+set_rows+add assembly is NOT in the
  executed graph on the default sparse path (only the masked-dense fallback consumes it) —
  already free; not a fix target.
- NEXT: the fused per-layer score op (pool+norm+mrope+score in one kernel, mrope math
  replicated byte-identically from the ggml rope kernel; keep ggml_indexer_top_k; env-gated
  probe → adopt; then cross-arch halo check for gating decisions).  Detailed design in
  `2026-09-07-gfx1201-qsa-decode-rootcause.md` §7.  Expected @32K: most of the ~7.8 t/s.


### 2026-09-07 (cont.) — QSA-DECODE FIX increment 1 LANDED: fused INDEXER_POOL (fork `fde1f2def`)
- New GGML_OP_INDEXER_POOL replaces the decode per-op chain get_rows + r-slice pool +
  scale + rms_norm (~10 kernels/layer) with ONE kernel, env-gated
  (`GGML_CUDA_QSA_INDEXER_POOL=1`, decode n_tokens==1 only, default OFF until the
  score-side fusion lands).  Parity contract met: bf16→f32 gather (ggml bits<<16),
  sequential member adds, 1/r scale, 256/1024-thread rms_norm reduction order —
  same-seed decode text BYTE-IDENTICAL ON vs OFF.
- gfx1201 (3x R9700, IQ4_XS, tensor bf16, interleaved r1): decode @d32768 39.7 → 42.1
  (+6.0%), @d65536 35.5 → 38.9 (+9.6%).  Dense still 47.5@32K → the remaining ~13
  kernels/layer (rope + q-side + score + relu/headsum + topk) = increment 2.
- Ground truth established: indexer cache mirrors `-ctk` type (BF16 here; idx_dim=128,
  r=4, n_stream=1); get_rows UPCASTS the bf16 keys to F32 (the per-op pooling is F32).
- Plumbing mirrored from INDEXER_TOPK (enum/name/symbol, builders, backend-choice
  skip, meta mirrored split, dispatch, RPC bump); CPU fallback deferred (op is
  decode-gated GPU-only until adoption).


### 2026-09-07 (cont.) — increment-2 scoping: score-side fusion needs its own design unit
- After INDEXER_POOL, the per-layer decode chain is ~13 kernels: rope(pooled, over
  n_blocks) + q mm/norm/rope + score mm + relu + head-sum(cont+3 adds) + bias + topk
  (store mm + cpy + pool + topk kept).  The scaled kernels (rope + score mm read/write
  the whole [128 x n_blocks] pooled tensor) are the 8K->32K fall-off driver.
- BLOCKER for fusing through the score mm: the F32 mul_mat dispatch is shape-dependent
  (mmvf/mmf/mmvq/mmq/cublas decision tree, ggml-cuda.cu:2007-2030) - a fused dot kernel
  must replicate the exact accumulation order of whichever fires, per-arch.  The
  elementwise tail (relu + hsum + bias) is parity-safe to fuse (~4-5 tiny kernels/layer,
  ~1 t/s) but small.  The pooled-key rope needs an mrope (sections) port for full fusion.
- Recommend next unit (fresh session): pick ONE of - (a) full INDEXER_SCORE op absorbing
  mm+relu+hsum+bias with the mmf/mmvq order replicated for the F32 small-batch case then
  byte-toggle; (b) precompute pooled+normed+ROTATED keys in the store path (incremental
  side cache) to kill the per-token rope+pool reads; (c) env-gated mrope-port rope fusion.
  Each needs its own parity toggle + halo A/B.
- Increment 1 state: fork `fde1f2def` (env-gated OFF), +6.0% @32K / +9.6% @64K decode,
  byte-identical; NOT yet folded into the beta patch (waits for the full fusion to
  adopt + strip the gate).


### 2026-09-07 — QSA-DECODE FIX increment 2 LANDED: fused INDEXER_SCORE (fork `e1e5a474b`)
- New GGML_OP_INDEXER_SCORE replaces the ENTIRE per-token decode chain get_rows + pool +
  scale + rms_norm + rope_multi + score mm + relu + head-sum + bias (~12 kernels/layer)
  with ONE kernel, env-gated (`GGML_CUDA_QSA_INDEXER_SCORE=1`, decode n_tokens==1,
  blk_bias, n_idx_h<=8, IMROPE).  Subsumes increment 1 (INDEXER_POOL stays in the tree,
  its graph call replaced when SCORE is on).  Parity contract met byte-identically:
  increment-1 pool/norm geometry, the IMROPE half-pair rope (rope.cu theta_scale =
  powf(freq_base,-2/n_dims), sections [11,11,10,0] interleave, n_rot=64 of idx_dim=128,
  cos/sin math verbatim) and the mmvf F32 vec-dot order (wave32: lane float2 partials,
  per-warp xor tree, w0+w1) - same-seed decode text BYTE-IDENTICAL ON vs OFF (first try).
- gfx1201 (3x R9700, IQ4_XS, tensor bf16, interleaved): @d32768 39.4 -> 42.4 (+7.6% vs
  per-op); @d65536 35.0 -> 38.9 (+11.1%).  In-window dense refs: 47.4 @32K / 48.0 @d0
  (dense IS ~flat: -1.2% over 0->32K; the earlier -11% d0->64K ladder was a pre-fusion
  window).  Fused-path marginal (skip probe, same window): ALL 42.4 / HALF 43.95 /
  QUARTER 44.8 -> the fused build costs ~3.1 t/s @32K (vs 7.8 t/s per-op: the two
  increments removed ~60% of the sparse build cost).  Residual vs dense ~4.8-5 t/s =
  the store mm + cpy_k + q-side (mm/norm/rope) + topk launches (~6 kernels/layer) +
  the sparse-FA path, NOT the pool->score stack (now 1 kernel).
- Model geometry captured (IDXDBG probe): idx_dim=128, n_idx_h=4, n_rot=64, rope_type 40
  (IMROPE), n_ctx_orig 262144, freq_base 1e7, freq_scale/attn_factor 1, ext 0,
  sections [11,11,10,0], rms eps 1e-6, ratio layers r=4 from layer 3.
- Next levers for the ~4.8 t/s residual (in order of value): (a) fold the q-side/store/
  topk remaining launches via a fused per-layer op or side-cache reuse (option b of the
  worklog increment-2 note: incremental pooled+normed+ROTATED cache at store time kills
  the re-pool read); (b) sparse-FA decode geometry vs dense; (c) accept at ~90% parity
  and fold.  Halo (gfx1151) A/B + adoption/fold still pending (maintainer call).


### 2026-09-07 (cont.) — the WASTE fix: design + scoping (next unit after INDEXER_SCORE)
- Halo regime data landed (`wip/archive/qwen4exp/discovery/2026-09-07-halo-gfx1151-dense-vs-qsa-regime.md`):
  dense beats QSA at EVERY tested depth on Strix too (d0 ~flat 25.8; d12K 24.8 vs 23.0;
  d32K 23.4 vs 20.9) and QSA falls ~2x faster (25.8->20.9 = -19% vs dense -9% over 0->32K).
  The per-token indexer waste is ARCH-UNIVERSAL and context-proportional - the fix is the
  same on both boxes; crossover depths differ (Strix sooner - its dense already falls -9%
  @32K vs gfx1201's ~-1%).  Re-measure this protocol against the fused build next.
- The remaining waste decomposes into THREE pieces (the fix stack):
  [1 DONE] kernel-count: INDEXER_POOL + INDEXER_SCORE (12-15 ops/layer -> 1); residual
  build ~3.1 t/s @32K on gfx1201 (skip probe).
  [2 gfx1201-only, CHEAPEST] the fused INDEXER ops are declared MIRRORED in
  ggml-backend-meta.cpp -> on 3-GPU tensor split EVERY GPU re-reads the full raw cache
  (8.4MB/layer @32K each = 25MB aggregate) while dense FA reads its KV shard only
  (~5.6MB/GPU).  Score rows are per-block deterministic -> the score op can be ROW-SPLIT
  across devices (each computes its 1/3 of blocks from its own cache mirror, identical
  values) + the meta backend's existing split->mirrored allgather feeds topk (the per-op
  mm did exactly this pre-fusion).  Expected: ~3x read cut on the indexer, worth ~1.5-3
  t/s @32K on the 3-GPU box.  Needs a meta-backend split-state study (how MUL_MAT's
  row-split + allgather is expressed for a custom op) - a gfx1201-only measurement step.
  NOTE: INDEXER_POOL mirrored may also have masked part of increment-1's gain.
  [3 the asymptotic fix, both arches] kill the re-pool entirely: maintain the
  pooled+normed+ROTATED block vectors incrementally.  A completed block's vector is
  invariant (pool over its r members -> rms_norm(W) -> rope at the block position), so it
  can be computed ONCE at store time and gathered at decode.  Removes the context-
  proportional read + pool/norm/rope compute from the decode path -> sparse decode
  becomes weight-bound + capped-FA reads (the 2051-cell cap) like dense, and the true
  per-arch crossovers become measurable.
- [3] design (spec for the next session):
  * Home: a per-layer buffer in llama_memory_hybrid_idx next to the raw indexer cache
    (mem_idx), sized n_blocks_max = ceil(kv_size/r) x idx_dim x ns F32; the raw cache stays
    untouched (still needed for prefill re-pool + the tail partial block + evictions).
  * Lifecycle: mirror mem_idx's slot/seq ops on block granularity (block b covers cells
    rb..rb+r-1); seq_rm/cp/keep/add/div + state io + clear must rewrite/expire block rows;
    the partial block at sequence boundaries is handled by the existing raw path (decode
    pools only the <=1 unfinished tail block on the fly from raw).
  * Fill: a store-side op/kernel, scheduled when a block completes (graph-side: after
    cpy_k when (pos%r) == r-1; prefill ubatch stores complete several blocks per step),
    replicating INDEXER_SCORE's pool/norm/rope arithmetic EXACTLY (the byte-toggle gate).
  * Read: INDEXER_SCORE v2 takes the derived cache view (+ raw only for the tail block)
    and runs dot+relu+headsum+bias (+ the tail-block pool) - no pool/norm/rope over
    completed blocks; same per-block values by construction (same arithmetic at store).
  * Parity: fused ON == per-op chain ON byte-toggle must still pass (values are
    deterministic per block -> stored == recomputed).
  * Payoff estimate: read halves vs raw (n_blocks*128*4B = half of n_kv*128*2B @ r=4) +
    no norm/rope compute; the gain grows with context (the waste is context-proportional).
  * Order of work: [2] first (cheap, gfx1201 measurement), then [3]; re-run the halo
    protocol + the gfx1201 depth ladder after each; halo A/B of the fused build is owed
    before any gating/adoption decision.


### 2026-09-07 (cont.) — [2] META-SPLIT STUDY: "cheap row-split flip" is NOT expressible; [3] spec tightened
- Studied the meta backend split machinery (ggml-backend-meta.cpp calculate_split_state: handle_generic/
  per_row/mul_mat/flash_attn_qsa) + dumped the LIVE decode graph split states (GGML_META_DEBUG=1,
  llama-cli -c 16384 -ctk bf16, fused ON; log /tmp/meta-debug4.log).  Ground truth:
  * The whole QSA indexer domain is MIRRORED by tensor allocation: cache_idx_k/v_lNN (raw indexer
    keys), INDEXER_SCORE, INDEXER_TOPK all MIRRORED; the indexer key store is a MIRRORED SET_ROWS;
    the indexer q/k proj + norm weights are statically replicated (MIRRORED, no srcs).
  * The DENSE KV (cache_k_lNN) is SPLIT (axis 2); FLASH_ATTN_QSA runs split on the split KV + the
    MIRRORED topk/mask (handle_flash_attn_qsa mixes them - the precedent for mixed ops).
  * Splits are a property of ALLOCATED TENSORS, propagated down-op; mirrored srcs FORCE a mirrored
    op (handle_per_row/generic return the src split; no op may create a split output from mirrored
    srcs).  => A "row-split INDEXER_SCORE" is NOT a meta-side declaration flip: the indexer cache
    itself must be physically SHARDED (like the dense KV) for the score to read only 1/3, plus the
    host-built blk_cells/blk_pos/bias inputs must carry matching row splits, plus the SET_ROWS store
    (today a mirrored broadcast) must become a sharded write (mirrored->split resplit scatter), plus
    the score shards need an allgather into the (global) topk.  That is llama-kv-cache allocation +
    memory-layer + meta + state-io surgery = DAYS, NOT the cheap flip the handoff assumed, and it is
    gfx1201-ONLY (halo is single-GPU: no mirror tax at all).
  * Expected [2] payoff re-estimate from the fusion data: the residual fused build is LAUNCH-bound,
    not read-bound (pool fusion with the SAME read gained +2.4; score fusion +0.3): @32K the read
    tax is maybe ~1 t/s, growing at 64K+ where the raw read doubles per device.  Dense FA reads its
    1/3 KV shard; mirrored INDEXER_SCORE reads the full raw cache on each device.
- RECOMMENDATION (deviating from the [2]-then-[3] order, on this finding): skip standalone [2]; do
  [3] (derived block-vector cache) FIRST, allocated MIRRORED like the raw cache.  [3] halves the
  read on EVERY device (derived 4.2MB @32K vs raw 8.4MB) AND removes the per-token pool/norm/rope
  compute AND is the only lever that helps halo (single GPU).  A later [2]-style split of the
  DERIVED cache (once [3] proves out) would additionally cut the mirror 3x - keep that as the
  follow-on, not the prerequisite.
- NEW FINDING for the halo A/B (owed): the fused INDEXER_SCORE/POOL builders assert K type
  F32/BF16 and the gather is bf16-shift-only (indexer-score.cu ld_val).  Halo runs -ctk f16 -> f16
  cells -> the fused op ABORTS there; QSA decode on halo has only ever run the per-op chain (f16
  handled by get_rows).  The halo fused A/B requires an f16 gather path in the fused kernels
  (f16->f32 upcast is exact, so byte-parity holds; small change).
- [3] spec tightened with the live memory-layer reading (llama-memory-hybrid-idx.cpp set_input_qsa):
  * Grouping is HOST-side, per step, and only FULL blocks (all r slots of one seq-set present,
    grp_slots==slots_full) are EVER pooled; incomplete cells are attended per-cell via the spare
    "dead block" (dead_bid, 1e9 bias + mask).  So the derived cache only ever needs FULL blocks -
    which is exactly what the score uses - and the dead/tail handling stays on the raw path.
  * Single-seq decode is append-only: full blocks are contiguous from 0 and stay full; block b is
    valid iff cells rb..rb+r-1 exist.  => derived validity = a host watermark n_derived (contiguous
    from 0), maintained on the append path; ANY seq mutation (rm/cp/keep/add/div/state io) drops the
    watermark to 0 (the decode op then pools raw for all blocks for one step, then the fill
    backfills as tokens append).  Fill = ONE graph op after the raw store that backfills full blocks
    in (n_derived, n_full] (steady decode: <=1 block/step, reads 4 cells + W + pos, writes 128 f32;
    the first decode after a long prefill backfills ~n_kv/r blocks = one raw-pooling pass, the same
    cost today's EVERY step pays - acceptable one-time).
  * Decode op v2: for b < min(n_derived,n_blocks) read the derived row (f32, pre-normed+
    pre-roped) and dot; else pool raw exactly as today (dead/spare + not-yet-full rows).  Values
    identical by construction (same arithmetic at fill, same pos/weights).  Byte-toggle = the gate.
  * Home: a per-layer F32 [idx_dim x n_blocks_max x n_stream] buffer in llama_memory_hybrid_idx,
    allotted mirrored like mem_idx's K; lifecycle = watermark only (no llama_kv_cache cell/seq
    bookkeeping needed since blocks are positional under the decode gate).
  * Expected payoff: raw read 8.4 -> derived 4.2MB/layer @32K (both arches; on gfx1201 each device
    reads its own mirror so the same 2x) + zero pool/norm/rope per token; the gain grows with
    context (context-proportional waste); enables the flat sparse decode + per-arch crossover
    measurements the maintainer wants.
- [3] DESIGN RESOLVED (2026-09-07, two open questions closed against the live code):
  * GRAPH CACHING: llama.cpp reuses ONE decode graph across consecutive steps (same shape), so a
    fill op can NOT be conditionally omitted per step - it must exist in every graph and act on
    per-step HOST data, exactly like the raw SET_ROWS store (whose per-step write index comes from
    a host leaf - cache_idx_k SET_ROWS srcs = {indexer_k_raw view, host leaf, cache}).  The fill
    op's target = a per-step host leaf "fill range" (from=n_derived, to=n_full_this_step; fills
    [from,to) -> steady decode fills 0-1 blocks; the first decode after a long prefill backfills
    up to n_kv/r blocks as a one-time pass).  Fill kernel reads the completing block's r raw cells
    via the SAME blk_cells row + W + blk_pos + eps + r and writes the normed+roped vector into the
    derived buffer - replicate the pool/norm/rope arithmetic byte-exactly (the INDEXER_SCORE
    pass-1..3 code is the template).
  * INVALIDATION = the watermark, no memset needed: derived rows are only trusted for
    b < min(watermark, n_blocks) (watermark passed to the score op as a host leaf, per stream);
    stale rows above it are never read.  ANY seq mutation (rm/cp/keep/add/div/state io, host
    side) drops the watermark(s) to 0 -> the next step re-pools raw for one step then rebuilds as
    tokens append.  The decode score op pools RAW for b >= watermark (dead/spare + not-yet-full
    rows, exactly as today) and reads the derived f32 row + dots for b < watermark.  The
    watermark lives in llama_memory_hybrid_idx host state per (layer via ratio group, stream);
    set_input_qsa (already host-side, O(n_kv), knows n_full per stream) is the natural place to
    update it + emit the host leaves.
  * BUFFER HOME: per-layer F32 [idx_dim x n_blocks_max x n_stream] tensors created + allocated
    exactly like llama_kv_cache's K (per-layer dev ctx, ggml_backend_alloc_ctx_tensors_from_buft,
    buffer_clear) - a small per-layer tensor array in llama_memory_hybrid_idx (NOT a second
    llama_kv_cache: cells are blocks not tokens, and the watermark model replaces cell/seq
    bookkeeping).  n_blocks_max = ceil(kv_size/r).
  * FILL TRIGGER: single-seq decode stores 1 token/step; n_full advances by (n_kv_used%r == 0 ? 1 : 0)
    host-side from the ubatch positions; the fill range leaf = (n_derived, n_full].  Prefill steps
    (n_tokens>1) store many tokens -> the same range logic bulk-fills (or the first decode
    backfills).  Decode-path only to start (env-gated like the rest); prefill stays raw.

### 2026-09-07 (cont.) — fused INDEXER ops now accept F16 caches (fork `c07e70e6f`)
- The fused INDEXER_POOL/SCORE gather was bf16-shift/F32-only and the ggml builders asserted
  F32/BF16, so with an f16 indexer cache (-ctk f16 = the STRIX-HALO config) the fused decode
  path ABORTED at graph build - QSA decode on halo has only ever run the per-op chain.  Added a
  3-way ktype (F32/BF16/F16) to the pool gather in both kernels + the builder/supported type
  checks.  Parity is safe by construction: ggml GET_ROWS always outputs F32 (the per-op chain's
  pooling arithmetic runs F32 regardless of the cache type - the type only affects the gather's
  half->f32 upcast, and f16->f32 is an exact bijection matching get_rows' native half load).
- gfx1201 3-GPU same-seed toggle under an f16 cache (default ctk): generated text BYTE-IDENTICAL
  ON vs OFF; fused decode +4.3% (41.5 -> 43.3 t/s) at ~4K context - validates the f16 path.
  Prior bf16 parity/benches stand (e1e5a474b).
- Halo (gfx1151) fused A/B now unblocked: fork range c63f7f2a0..c07e70e6f bundled and fetched
  into halo ~/llama-delivery.  NOTE the first build attempt compiled the WRONG source: the bundle
  fetch refspec failed (`refs/heads/qwen4exp` not in the bundle) and the script's `||` fallback
  never triggered (piped to tail), so cmake built the still-checked-out c63f7f2a0 (per-op).  Killed
  that A/B, fetched the bundle by hash (`git fetch <bundle> HEAD` -> FETCH_HEAD = c07e70e6f,
  verified), and rebuilt: branch qwen4exp-fused @ c07e70e6f, build-fused with the build-gated flags
  (gfx1151, Release).  A/B watcher armed on HALO_FUSED_BUILD2_DONE (log /tmp/halo-fused-cmake2.log,
  outer /tmp/halo-fused-build2-outer.log; stale logs cleared).  When done (~25 min build + ~40 min
  A/B): same-seed parity toggle + the dense-vs-QSA regime protocol against the FUSED build (halo
  anchors: dense 25.8/24.8/23.4 vs per-op QSA 25.8/23.0/20.9 at d0/12K/32K, f16) - expect the QSA
  rows to move up (fusion gain transfers; f16 load now supported) and the dense-vs-QSA gap to
  narrow.

### 2026-09-07 (cont.) — HALO FUSED A/B LANDED (gfx1151): fusion transfers, gap to dense -2.3% @32K
- Record: `wip/archive/qwen4exp/discovery/2026-09-07-halo-gfx1151-fused-AB.md`.  Fused build
  (c07e70e6f) on halo, f16 config, interleaved r2: parity BYTE-IDENTICAL ON vs OFF; regime
  d12K per-op 23.0 -> QFUSED 24.04 (+4.5%), d32K 20.9 -> 22.88 (+9.4%); dense 24.87/23.42 -> the
  dense-vs-QSA decode gap collapsed from -11.9% (per-op) to -2.3% @32K (-7.8% -> -3.4% @12K).
  QSA decode is now within a whisker of dense on Strix at 32K; crossover plausibly 64-96K there.
- [3] derived block-vector cache is now the clear next unit on both arches (removes the re-pool
  read + pool/norm/rope; spec above is fully resolved).  gfx1201 re-baseline with the f16-capable
  tip owed but expected flat (bf16 unchanged).

### 2026-09-07 (cont.) — [3] IMPLEMENTED + MEASURED: correct, provably active, but FLAT on gfx1201 <=64K
- Full [3] landed on the fork: `328ddfa4e` (derived cache) + `6703ad09f` (guard fix + capped fill).
  Same-binary parity holds across per-op == fused == fused+derived (byte-identical text).
- DEBUG-COUNTER LESSON (maintainer's question was right): the FIRST flat results were the fused
  baseline - pool_create early-returned on `dsv4_compress_ratios[0]==0` (layer 0 has no QSA) so no
  pool ever existed and the derived path silently fell back.  Fix: scan the filter_idx layers for
  any ratio>0.  The path is now PROVABLY active: GGML_CUDA_QSA_INDEXER_CACHE=2 (probe mode, kept
  env-gated) passes the pool to the score WITHOUT the fill -> output diverges (garbage rows read) =>
  the pool read + limit plumbing genuinely execute; cache=1 stays byte-identical to fused.
- gfx1201 3x R9700 interleaved (pool real, fill grid capped to 512 + grid-stride):
  d32768 fused 42.53/42.37 vs derived 42.39/42.43; d65536 38.82/38.83 vs 38.78/38.80: FLAT.
  CONCLUSION: the decode at <=64K on this box is LAUNCH/LATENCY-bound (the per-layer kernel chain:
  store + q-side + fill + score + topk + FA, x12 layers, mirrored x3) - intra-kernel work removal
  (the pool/norm/rope the derived cache eliminates) does not show even when provably executed.
  The score kernel was never the binding cost; the ~3.1 t/s fused residual is launch structure.
- REPRODUCIBILITY FLAG: the absolute same-seed per-op text drifted across build states of
  "identical" sources this session (312 chars from the guard-fix-era binary vs 238/240 from clean
  rebuilds of 328ddfa4e+c07e70e6f) - agrees for ~39 tokens then ulp-diverges.  Same-binary toggles
  are deterministic and the parity gates hold; cross-build coherence comparisons need a pinned
  binary.  Worth an investigation before any adoption claim relies on absolute text.
- Open: the read-halving thesis can only show where reads bind - Strix Halo (single GPU,
  bandwidth-limited) and 128K+ depths.  Halo transfer staged (bundle to ~/llama-delivery branch
  qwen4exp-fused at c07e70e6f); needs the 6703ad09f delta + a fused-vs-derived A/B there.
  Remaining structural lever on gfx1201: fewer kernels/layer (the launch-chain), not less work
  per kernel - the fused mega-op direction or accepting the fused build.

### 2026-09-07 (cont.) — [3] PARKED (maintainer); rocprof cost isolation: the sparse tax is the topk + chain, NOT the FA
- [3] (derived block-vector cache, fork 328ddfa4e + 6703ad09f) parked as a wash: provably active
  + byte-correct but flat at <=64K on gfx1201 (launch-bound decode).  Keep env-gated OFF; the
  guard fix + capped fill stay.  Record: `wip/archive/qwen4exp/discovery/2026-09-07-sparse-decode-cost-isolation-rocprof.md`.
- rocprof decode-token windows (1-token @ ~30K, fused vs dense, 3x R9700 bf16):
  * The block-sparse FA is ~8x CHEAPER per call at 30K (30us vs 242us) - the read-cap premise
    delivers; the FA is the sparse WIN, not the tax.
  * The tax = the topk (O(n_kv) radix-select over the whole 32K score vector per layer per token,
    the step block-sparse was supposed to avoid) + the per-layer small-kernel chain (indexer
    q-side/k-side ropes + norms + gathers + copies, ~1-2ms/token) + extra inter-device copies.
  * Model: dense ~ a*n_kv (tiled FA); sparse ~ b*n_kv (topk, b<<a) + c*2051 (FA) + fixed
    machinery; sparse wins past the measured crossover (>64K on gfx1201; halo sooner).
  * Remaining levers: (1) the topk's O(n_kv) select, (2) the per-layer kernel count.  FA and the
    pool->score stack are done.  Fused prefill is also ~10% slower than dense (machinery
    amortized over the ubatch: 1313 vs 1449 t/s pp on the 30K prompt).
- CORRECTION (maintainer challenge): the topk is ~0.5 ms/token (all stages), NOT 4-6 ms - that
  figure was wrong.  The deeper device-level finding: the fused decode does LESS GPU busy work
  than dense (16.1 vs 18.5 ms/device/token at 30K) but wastes ~4.5 ms/token more in inter-kernel
  gaps (~66% vs ~88% utilization; ~2000 vs ~1613 dependent small dispatches/token).  The deficit
  is the per-layer kernel CHAIN serialization, not any kernel's work - which is why the fusion
  increments (busy-time cuts) moved nothing.  Lever = fewer/fatter/less-serial kernels per layer.
  See the corrected discovery record.
- Next candidates: the per-layer mega-op (store+q-side+score+topk+FA toward one kernel); halo
  fused-vs-dense A/B owed (only fused-vs-dense matters now, [3] parked).

### 2026-09-07 (cont.) — MEGA-OP phase opens; CRITICAL: the old decode profiles were the PER-OP path, not fused
- ARTIFACT FOUND: llama-cli decode token #1 (the first generated token after prefill) runs the PER-OP
  indexer chain; tokens #2+ run the fused INDEXER_SCORE.  n=1 runs (all the 2026-09-07 rocprof
  profiles) therefore captured PER-OP decode - the 23.4ms/66-70% util/+375-kernel numbers described
  the per-op path, NOT fused.  Verified: rocprof shows indexer_score_kernel fires for -n 3 (72 kerns
  = 12 layers x 6 tok) but not -n 1; per-token clusters in n=8 runs: token#1 per-op, #2-8 fused.
  Root cause not yet chased (graph-reuse/params subtlety) - steady-state decode (llama-bench, real
  serving) IS fused, so llama-bench remains the measurement vehicle; llama-cli n must be >= 3.
- TRUE FUSED steady-state numbers (llama-bench tg64 d32768 profiles, /tmp/prof/d32, box @11:57):
    fused 33.88 t/s (29.5ms/token): 2629 disp/device/token, busy ~18.3ms, util ~62%
    dense 37.55 t/s (26.6ms/token): 2412 disp/device/token, busy ~16.7ms, util ~63%
    deficit ~2.9ms/token = +217 disp (x3.6us gap ~ +0.8ms) + ~+1.6ms busy + noise.
  Sparse machinery/device/token = 159 launches (indexer_score 13.1 + topk init 13.1/hist 52.7/
  select 52.9/count 13.3/scan 13.3/write 13.3) ~= 12 launches/QSA layer x ~13.1 layers.
  The whole topk pipeline ~= 63us busy + 12 x 3.6us gap per layer ~= 1.0-1.4ms/token combined.
- CUDA graphs: decode DOES replay graphs (ids reused every token); GGML_CUDA_DISABLE_GRAPHS=1 costs
  only ~2% (31.1 vs 31.7 t/s) -> the ~3.6us/kernel gap is GPU-side dispatch turnaround inside graph
  replay: a hard per-kernel floor.  Kernel-count reduction is the only lever on the gap.
- MEGA-OP increment 1 (in progress): INDEXER_TOPK launch cut 11 -> ~7/layer (fold radix_init into
  pass-1 select; 4x 8-bit radix passes -> 3x (12+10+10 bits, shifts 20/10/0, smem 16KB/4KB/4KB)).
  Then q-side norm+rope fold into INDEXER_SCORE (-2/layer); then the score+topk-pass1 fusion.
  Parity rule: the topk cell LIST must be identical (tie order = ascending column) - the qsa kernel
  consumes the cells; any list change breaks the same-seed toggle contract.
- First-decode-token anomaly note for llama-bench users: llama-bench decode is steady-state fused.

### 2026-09-07 (cont.) — mega-op increment 1 MEASURED: launch-count reduction is at/below the noise floor; busy reduction is the lever
- Fork commit d6164ad6a: INDEXER_TOPK radix_init folded into the pass-1 select (11 -> 10
  launches/op; 13 launches/token saved).  Two-build same-seed A/B (32 tokens): output
  byte-IDENTICAL (the topk is int-exact; text diff = only the t/s banner).  Kept as a
  harmless cleanup.
- MEASUREMENT LESSON (interleaved llama-bench d32768 tg64, staged old/new binaries in
  /tmp/bins-old /tmp/bins-new with the bench launcher + lib set):
  * 12+10+10-bit 3-pass radix (12-bit first pass): REGRESSED -2.8% (39.06 vs 40.21 t/s).
    The 16x histogram smem-clear + global write-out per block (4096 bins vs 256) costs
    more busy than the 2 saved launches save.
  * init-fold only (per-pass work unchanged, -1 launch): FLAT (40.27 vs 40.28 t/s) - the
    ~0.2% expected saving sits at the box noise floor (+/-0.3% between runs).
  => At this kernel scale, per-launch BUSY dominates launch COUNT (~4:1): removing a launch
     is invisible; adding work to remove launches loses.  The sparse-machinery lever is
     busy reduction, not launch shaving.
- NEXT LEVER (designed, not yet built): the decode topk does ~6 full cell scans/layer
  (4 radix passes + count + write = ~192K value evals over n_kv=32K cells, all re-gathered
  via cell_blk) where at decode additive == 0 (single query, causal mask all-open -> every
  cell of a block shares score[block(c)]) so the selection is EXACTLY a top-k over the 8K
  BLOCK scores (r=4 cells each).  A block-granular topk path (flag from the builder when
  n_tps==1 && blk_bias, cells uniform) scans n_blocks instead of n_kv: ~4x less topk busy.
  Estimated +1-1.5% (the topk decode busy ~0.6-0.8ms/token of the 2.9ms fused-vs-dense
  deficit; per-launch floors keep the launch/gap part intact).  Parity rule: identical cell
  list (ascending (block, cell) order; the qsa kernel consumes the cells).
- STILL OPEN: why llama-cli decode token #1 takes the per-op path (graphs rebuild every
  token - rising CUDA graph ids 2939+ per token - so it is NOT graph reuse; the branch
  condition is static).  Not chased to root cause; llama-bench + llama-server steady state
  are fused, which is what matters.  n>=3 for llama-cli profiling.
- d32 llama-bench traces are contaminated by prefill ubatches (flash_attn_ext_f16 x432 =
  16 prefill ubatches x ~9 attn layers; the per-call qsa ~1978us there = PREFILL qsa at
  ubatch 2048, not decode) - clean decode slicing needs the qsa/tile-kernel-count method
  (last 640 qsa/device = 64 tg tokens at ~10 qsa/token).

### 2026-09-07 (cont.) — CLEAN 32K PROFILE (kernel-count-sliced decode phase): THE SCORE KERNEL IS THE #1 COST
- Method: d32 llama-bench traces (tg64 @ d32768, box @11:57, fused 33.88 / dense 37.55 t/s).
  Decode phase = last ~73 tokens (64 measured + warmup), sliced from the trace end by the
  qsa/tile count (520 kernels = 73 tokens x ~7.1 FA calls/device/token).  Per-token results
  (per device): fused 2136 disp / 15.6ms busy / 29.5ms wall; dense 1920 / 14.0 / 26.6.
  BOTH ~53% GPU-utilized (the earlier 88%-dense claim was window-divisor error; llama-bench
  async decode idles ~47% on this 3-GPU box even for dense).
- Per-token selection+attend side (busy):
    indexer_score_kernel: 1.43ms  (136us/call x 10.5)   <- THE TARGET (was thought ~0.1ms)
    indexer_topk total:    0.57ms (hist 0.30 + select 0.19 + rest 0.08)
    flash_attn_qsa:        0.13ms (7 x 18us)
    fused side total:      2.13ms  vs  dense flash_attn_tile 0.66ms (7 x 94us)
    => +1.47ms/token, which CLOSES the measured total busy delta (+1.6ms).  The whole fused
    deficit is the score+topk machinery vs the FA swap; the score alone is 2.5x the topk.
- Score kernel scaling: 27us/call @4.3K (n_blocks~1075) -> 136us @32K (~8000 blocks): ~linear.
- Why the score is slow (kernel anatomy): 1 block per score-row (grid = n_blocks), 256 threads
  for a 128-dim job, ~5 syncthreads-serialized passes with thin active thread counts per pass
  (128 pool / 32 rope / 64 dot / 1 epilogue), and the rope pass runs 32 POWF per row for
  values that are per-model constants x a position scalar.  Throughput-bound on the per-row
  barrier-chain latency at maxed occupancy.
- Previous turn's direction (topk = the lever) was WRONG: launch-count cuts on the topk were
  flat because the topk (0.57ms) is not the dominant cost; the score (1.43ms) is.
- This reopens the parked [3] derived-cache: its premise (completed rows' pooled+normed+roped
  vectors are invariant; only ~1 new block/step changes) is exactly the right lever for the
  score's cost - yet [3] measured FLAT on the same 32K bench.  Contradiction (cutting 1.43ms
  should show ~3%) -> either [3]'s derived read path does not cut the score busy as designed,
  or the score busy is not fully on the critical path.  NEXT EXPERIMENT (in progress): re-run
  cache=1 vs cache=0 under rocprof at 32K and compare the score kernel's per-call busy.
- Fix directions if [3]-read-path is broken: (a) kill the per-row powf (precompute the
  freq^j table once per launch; the yarn corr is per-pair constant too) - the rope pass is
  the fat barrier stage; (b) verify the derived-row path actually skips passes 1-3 with the
  pool rows (it should: rows below LIM read 128 floats and jump to the dot).

### 2026-09-07 (cont.) — cache=1 experiment: score busy cut VERIFIED but wall FLAT -> the decode wall is CHAIN-LATENCY-bound, not busy-bound
- cache=1 (derived [3]) at d32768 under rocprof: indexer_fill fires 1/layer/decode-step
  (768/device = 10.5 x 73 tokens); decode score per-call DROPS 136 -> 58us (busy 1.43 ->
  0.61ms/token, -0.8ms).  Yet llama-bench wall is flat (the original [3] A/B 42.39 vs 42.53).
- INTERPRETATION: the fused decode wall is NOT busy-bound.  Both fused and dense idle ~47%
  of the wall (53% util, 3-GPU async decode with AR syncs).  The sparse chain adds ~14
  DEPENDENT kernels/layer to the critical path (score -> 11-stage topk -> qsa FA), each
  serialized at ~10-13us dispatch+latency -> ~150-200us of chain latency/layer x 10.5
  layers ~= the ~2ms deficit.  cache=1's busy cut just converts busy into idle (the chain
  structure is unchanged) -> wall flat.  This explains every prior result: launch cuts flat
  (1 off a 14-long chain), wider radix regressed (per-kernel latency up), score busy cut
  flat (chain unchanged).  Total busy (15.6 vs 14.0) rises by the sparse work, but the WALL
  is set by the critical-path chain structure, and busy fills whatever it allows.
- NEXT DECISIVE EXPERIMENT: measure the per-layer chain latency directly with the existing
  GGML_CUDA_QSA_DECODE_SKIP=N probe on the CURRENT fused build (skip the whole sparse
  chain on every Nth layer -> dense attend): if the wall drops ~150-200us per skipped
  layer, chain-latency is confirmed as the wall driver and the mega-op must FUSE the
  score+topk+FA chain into few kernels (the only structure that removes dependent edges).
  The true mega-op target is the ~14 dependent kernels/layer -> ~2-3, NOT busy or launches.

### 2026-09-07 (cont.) — DECODE_SKIP marginal CONFIRMS chain-latency-bound; the mega-op = shorten the per-layer dependent chain
- GGML_CUDA_QSA_DECODE_SKIP at d32768 (interleaved, box ~40.3 t/s baseline):
    skip0 (none):      40.33 t/s (24.8ms wall)
    skip1 (every 2nd): 41.65 t/s (24.0ms)   ~5.25 skipped layers -> ~0.15ms/layer freed
    skip2 (every 3rd): 41.70 t/s (24.0ms)
    skip3 (every 4th): 42.41 t/s (23.6ms)   ~2.6 skipped -> ~0.46ms/layer freed
  Removing the sparse chain from a fraction of the ~10.5 sparse layers frees wall at
  ~0.15-0.46ms/skipped layer (non-linear: freeing whole chains compounds).  Matches the
  ~150-200us/layer dependent-chain latency estimate (14 kernels: score 1 + topk 11 +
  qsa+combine 2, each ~10-13us serialized on the critical path).
- CONCLUSION (the whole session's arc): the fused decode deficit vs dense at 32K is the
  CRITICAL-PATH LENGTH of the per-layer sparse chain, in an idle-bound pipeline (53% util).
  Launch-count shaving (flat), score-busy cuts via [3] (flat), and wider radix (regressed)
  all follow: none shorten the dependent chain.  The mega-op must REPLACE the chain
  structure, not optimize its parts.
- MEGA-OP DESIGN (next build): replace the topk's 11-kernel exact-radix + deterministic
  gather (init, 4x(hist+select), count, scan, write = ~11 dependent steps ~= 120-145us)
  with a 2-round select over the full 32-bit key: hist1 (cells -> 2^16 GLOBAL bins,
  1 launch) + sel1 (single block scans 64K bins, 1) + hist2 (re-scan cells in the boundary
  bin's top-16 range -> low-16 bins) + sel2 + emit (filter exact prefix, deterministic
  ascending order) ~= 5 dependent kernels ~= 60us.  Chain 14 -> ~8 kernels/layer; est
  -0.5 to -0.9ms/token wall (+2-4%) IF the busy stays off the wall (skip experiment says
  the wall is chain-bound, so per-kernel busy may rise within limits - the 12-bit smem
  regression warns to keep histogram passes global-bucket cheap).
  Parity rule: 2x16-bit rounds resolve the full 32-bit key exactly; the emit must produce
  the same deterministic ascending-column cell list.
- Alternative/compounding: fuse the score's tail into hist1 (score blocks atomicAdd their
  r cells' worth into hist1's buckets) and/or the block-uniform mode (8K values not 32K)
  once the 2-round select is in.

<!-- keep the newest entry below this marker -->
