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
- [ ] **1.2 Quantize mmq-q8_1 chunk** (`ggml_cuda_quantize_mmq_q8_1_n_chunks`,
      `quantize.cuh:26-27`, hard gate `cc == RDNA3_5 + 1` → n_chunks=2; gfx1201 = 1).
      Plan: measure the gfx1201 launch-bound vs occupancy tradeoff for the 262144-row q8_1
      feed (the unchunked 512-float-slice launch); set the RDNA4 chunk count from data.
      Validation: same-session chunked vs unchunked on the affected rows + coherence.
- [ ] **1.3 split_j Q8_0 mma specialization + B-parity Q8_0 config rows** (`mmq-vec-dot.cuh`
      `#elif defined(RDNA3_5)`; `mmq-config-rdna3_5.cuh` vs untouched `mmq-config-rdna4.cuh`).
      Plan: benchmark RDNA4's native I=128/192-vgpr profile vs a split_j I=64 port; adopt only
      if it wins (the gfx1151 win came with a vgpr 232→136 change — RDNA4's profile differs);
      else document the decision in the worklog.
      Validation: same-session + logitcmp; note the campaign TRAP (the `!use_mmvq` exclusion
      is required or single-token decode diverges — re-check when porting the geometry).
- [ ] **1.4 mmq accumulator-overflow latent-defect audit on the RDNA4 table** (TODO 3d; the
      `I < nwarps*16` mma sum[] overflow — "likely applies [to rdna4.cuh] too" per the brief):
      audit every row in `mmq-config-rdna4.cuh` for `I >= nwarps*16`; feed the upstream defect
      report (no perf value, correctness hygiene).
- [ ] **1.5 GDN gfx12 chunked kernel — post-consolidation re-validation**: gfx1201 uses the
      gfx12 kernel in `gated_delta_net.cu` (NOT the gfx11 first-gen-WMMA file the campaign
      retuned).  The 0002 chunked-prefix dispatch was validated on 3x R9700 **pre-re-base**;
      re-run the chunked-prefix A/B (`GGML_CUDA_GDN_CHUNKED=0` opt-out) on the current
      delivery + MTP/depth rows per `benchmarks/mtp-adaptive-methodology.md`.
      (The gfx11 NW16 scan retune itself only needs gfx1100/1101 launch-fitness checks — see
      Phase 4; it does not reach gfx1201.)
- [ ] **1.6 Per-file RDNA3_5 config-row audit** (fattn / mmf / concat / mmvq / mmid / vecdotq
      RDNA3_5 references): classify each as (a) RDNA3_5-only row (leave; RDNA4 has its own
      pre-campaign rows) vs (b) a "generality" finding that should carry to RDNA4 (e.g. the
      flash (256,256,64) Q_in_reg register-pressure finding — check whether the RDNA4 flash
      row set shares the geometry).  List the carry candidates + validate each.

## PHASE 2 — Consolidated-beta model-level validation on gfx1201 (same-session ladder)

The beta content (QSA shortcut default, PLE host-gather, weighted-down, hc hyperconn
fusions, repeat-absorb, mmid/mwr ports, managed-ngrams, MTP draft head) was gfx1151-gated;
gfx1201 has only pre-re-base records (QSA decode fix etc.).  The sched-gate fix (this
session) made the ggml layer safe multi-GPU; now validate the model level end-to-end.

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

<!-- keep the newest entry below this marker -->
