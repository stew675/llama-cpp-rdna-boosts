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

- [ ] **2.1** Depth-0 ladder + tg (tensor AND layer split) on the fixed build, IQ4_XS + MTP
      draft (Q4_K_M mtp model), same-session toggles where they exist: `LLAMA_QSA_OFF=1`,
      `GGML_CUDA_DISABLE_HC_FUSION=1`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`,
      `GGML_CUDA_DISABLE_MMID_512=1`, `LLAMA_QSA_DENSE_SHORTCUT=0`.  Record ON-vs-OFF deltas
      (expect the gfx1151-validated fusion gains to reproduce — the kernels are model-level).
- [ ] **2.2** Depth rows (12k/32k, r1) + memory stability −r3 through 32k.
- [ ] **2.3** Decode leg: tg128/512, MTP acceptance per `benchmarks/mtp-adaptive-methodology.md`
      (acceptance must stay > ~0.45, MTP >= plain at depth 3), server smoke (user config,
      flat decode through 4k gen — the gfx1201 QSA decode fix pre-dates the re-base).
- [ ] **2.4** Multi-GPU-specific: hybrid-AR + RCCL toggles (`GGML_CUDA_ALLREDUCE=nccl` /
      default hybrid / issue-13 fallback), the tensor-split + MTP-verify path, ubatch-2048
      multi-chunk prefill determinism (the sched-gate's original failure mode — confirm the
      fixed build is deterministic run-to-run, same-seed, tensor split).
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

<!-- keep the newest entry below this marker -->
