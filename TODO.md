# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.
Forward-looking: open items + current active experiments; closed work is a
one-line bullet (details live in AGENTS.md, patches/README.md, MANIFESTS.md,
`beta/qwen4exp/README.md`, `wip/` handovers, `benchmarks/`). Current
delivery = the 15-patch set against fork point `9113cc188` (block 00 + blocks 01-14).
Block 15 is STAGED in `beta/block-15-campaign-wins/`, not promoted.

## Current active

### Issue #25 follow-up: GDN chunked prefill (plain vs spec divergence) - IMMEDIATE, GFX1201
- **OPEN (2026-09-11).**  On one build+prompt `--spec-type none` and MTP differ, and it is entirely
  the GDN chunked prefill: `GGML_CUDA_GDN_CHUNKED=0` makes `none == n-max 2 == n-max 4`
  byte-identical (p0: `9216c6d1` -> `bba7741d`).  Cause: `gated_delta_net.cu` branch 1 runs the
  plain multi-token prefill (`K=1`) chunked while spec prefill (`K=n_max+1`) runs sequential; and
  for prompts `> K+64` branch 2's prefix boundary `n_tokens-K` shifts with `n_max`.  Fork-only
  (block 02) - upstream `9113cc188` ships only the sequential kernel (`gated_delta_net.cu:180`
  `//TODO: Add chunked kernel`), so this is **not** a Block 00 item.  Latent in practice (a
  2.8k-token MTP run did not flip in 200 tokens) but a real state divergence; probe
  `P=256 RS=from_w` gives W3-W5 = 0.210405 (chunked on) vs 0.000000 (off).  Fix directions, repro
  and the validation gate: `wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FOLLOWUP.md`.
  Owner: the GFX1201 box, after the updated delivery `main` is pushed.

### Memory campaign -> Block 0015 (derived kq mask + FA scratch + QSA wins) - STAGED IN beta/ (not promoted)
- **DONE (2026-09-10): RDNA3_5 (gfx1151) validation pass + beta block-15 amendment.**  First
  single-device iGPU run (Strix Halo, ROCm 7.14).  Block-14 masked-V fixes +
  V3/V4/V5 exercised with a BF16 KV cache in both arm states; the block-14 fixes are clean on
  gfx1151 and V4/V5 do not reintroduce a masked-cell leak.  Two V3 regressions were found and
  fixed as a dated amendment to the **beta** block-15 patch (**beta patch tip `377f8e790`**; the
  delivery stays 14 patches, clean-apply strict 14/14, applied tree `6ce36849`): (1) the derived
  probe rejected `GGML_BACKEND_DEVICE_TYPE_IGPU`, silently disabling V3 on the iGPU;
  (2) `n_seq_max > 1` aborted in `ggml_flash_attn_ext_add_kq_derived` (`kq_mask_derivable()`
  now rejects `n_stream != 1`).  After the fix the reserves reproduce RDNA4 exactly (4B V3
  -799.20/-799.21, V5 bf16 968.86->256.86, V4 q8_0 1001.13->257.13; Flash-Next 3251.39/63.69
  indexer 318.76), 14 ROCm + 7 Vulkan gates PASS 16/16, probes clean, MTP identical.  V4 is
  *faster* on gfx1151 at depth (+2.6 % pp20480) and V5 costs 0.4-0.9 % vs RDNA4 0.2-2.4 %.
  Record: `wip/strix-halo/GATE-2026-09-10-block15-rdna35.md`.
- **Follow-up (RDNA3_5, low priority): V3 prefill cost is arch-dependent.**  gfx1151 measured
  -3.2 % at pp20480 (4B, q8_0) vs the RDNA4 4B reference -1.3 %, decode flat.  Still a large
  net win (-799 MiB compute + -799 MiB host) and on by default; if an iGPU tuning pass ever
  runs, the derived MMA kernel's `J`/occupancy on gfx1151 is the place to look.
- **Follow-up (block 13, RDNA3_5): the isolated fused-MoE delta is now ~0 on Strix Halo.**
  With the 2026-09-06 model-neutral folds in the tree, `GGML_CUDA_DISABLE_MOE_MMQ_FUSION` on vs
  off measured pp2048 +0.4 %/pp16384 +0.2 % (was +5.3 %/+4.6 % on the 2026-09-05 build);
  absolute prefill is ~10-13 % higher and the fusion still fires, so this is the folds
  capturing the same work, not a regression.  Re-check whether the gate+up+GLU arm still has a
  unique win before any future tuning.
- **Block 15 is STAGED in `beta/block-15-campaign-wins/`**, NOT in the delivery
  (`beta/block-15-campaign-wins/block-15-campaign-wins.patch`, beta patch tip `377f8e790`,
  including the 2026-09-10 V5 and RDNA3_5 amendments).  Six wins,
  each with an env A/B gate (V4 is opt-in): W1 QSA score-chain (`GGML_QSA_SCORE_MEM`), W2 derived QSA bias
  + visibility + the input-fill null guards (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), W3 keys-only
  indexer cache (`LLAMA_QSA_KEYS_ONLY`), W4 ggml-alloc unused-view release (no gate; `ab/w4-revert.patch`),
  V3 derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), V4 native q8_0 FA K/V
  (`GGML_CUDA_FA_KV_NATIVE`, default 0).
- **Measured** (ctx 204800 / q8_0 / ub 2048): qwen4exp compute 6690.40 -> **3251.39** MiB/GPU + host
  1262.70 -> **63.69** MiB, indexer KV 956.26 -> **318.76** MiB/GPU; dense models -799 MiB/GPU + -799 MiB
  host (V4 a further -744 (4B) / -632 (27B)); byte-identical same-seed output on all five models across
  every gate combination; MTP unchanged (27B 0.76744, qwen4exp 0.44262); ~1.3 % prefill / ~0.3 % decode
  (V4 ~1.7-1.9 % more, hence opt-in).  Combination-validated on the merged tree AND on the tree built from
  the beta patch on top of the 14-block tree (fresh worktree + build).  Records:
  `beta/block-15-campaign-wins/README.md`, `WORKLOG.md`.
- **Beta window open** (2026-09-10, ~4-5 days): tester material is `beta/block-15-campaign-wins/BETA-TESTING.md`.
  Promotion = declaring it stable; feedback that needs a change becomes a dated amendment to block 15.
- **DONE (2026-09-10, D12 closed): V5 native bf16 K/V, folded into the beta Block 15 patch as a dated amendment.**
  bf16 was the last KV type paying the F16 staging scratch in prefill; the MMA loader now converts each
  16-byte staged chunk in registers (bit-identical to the launcher's own conversion), behind the **same
  `GGML_CUDA_FA_KV_NATIVE` switch as V4 (default 0, opt-in)**.  With it enabled a bf16 cache costs
  exactly an f16 one: 4B ub 2048 968.86 -> **256.86** MiB/GPU (ub 1024 884.82 -> 128.82, ub 512
  842.80 -> 64.80), 27B 1072.86 -> **488.86**, gemma-4-E4B 1062.89 -> **404.89**, gemma-4-31B
  2068.89 -> **716.89**; qwen4exp unchanged (its FA path never staged bf16); TILE/verify unaffected.
  Opt-in because dropping the scratch costs 0.2-2.4 % prefill (growing with the prompt) and the
  maintainer's instruction for this item was explicitly "treat it similarly to V4, gated by the same
  environment variable".  Design + full measurements: `wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md`
  (section 9) and the V5 amendment section in `patches/README.md`.  Scope was fixed by D10 (no
  pure-bf16 rework).
- **Follow-up (would make V5 free): the loss is not the conversion — native bf16 staging measures within
  0.2 % of an f16 cache — it is the removed F16 scratch, which is a *dense, normalised* copy of the cache
  view (for a 4-KV-head model `nb[1]` is 4x the row size: the GQA heads are interleaved), while the
  native path re-reads that interleaved view on every staging pass.  Options: (a) restrict the arm to
  layouts where `nb[1] == ne[0]*2` (single-KV-head models — a 1-line predicate change, then free there),
  (b) make the native staging read densely, or (c) make the KV cache itself non-interleaved (a
  llama.cpp-wide change).  None is needed for the current opt-in delivery.
- **Cleanup candidate (block-15 wart, pre-existing): `src/llama-kv-cache.h:274` warns
  `-Wunused-private-field` for `v_enabled` on a full build** (the field *is* used, in
  `llama-kv-cache.cpp:232`; clang's per-TU analysis is what fires).  A `[[maybe_unused]]` one-liner
  silences it — left alone here to keep the V5 amendment scoped to the FA kernels.
- **Documented, NOT fixed (pre-existing): mixed K/V types fall off the GPU attention path.**  Any mixed
  pair (`bf16`+`q8_0`, `f16`+`q8_0`) gives `graph splits = 18`, a ~1.5 GiB host compute buffer and
  pp2048 7924 -> 640-1049 t/s on the 4B.  Same-type K/V is the practical choice; fixing it needs the FA
  kernels to accept a mixed `(type_K, type_V)` pair (bigger than V3/V4) - out of scope, see the plan's
  section 6.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`: documented only (D11).**  Pre-existing meta-splitter abort
  (2 KV heads < 3 devices); works on 1/2 GPUs and with `-sm layer`; maintainer's call: no fix.
- **Found, documented, NOT fixed** (pre-existing - reproduces on block 14): gemma-4-E4B-it on 3 GPUs with
  `-sm tensor` aborts in the meta splitter (`ggml-backend-meta.cpp:1177`) because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share); it works on 1 GPU, on 2 GPUs and on 3 GPUs with
  `-sm layer`.  Every other model is unaffected.  A future block (or an upstream report) should make the
  splitter tolerate a zero-extent device share.  See `patches/README.md`.
- **Fork/canonical state**: the working checkout's `rdna-boosts` is a local rebuild and must NOT be used
  for regeneration if it sits on a master newer than the fork point (it would export `f3f1a8f27`
  + `304665fe7` as patches 0001/0002).  The canonical 15-block chain used for the delivery ends at the
  block-14 commit `33ccf7e28` (rebuilt at `9113cc188`; block 02 amended 2026-09-11 with the
  `GGML_CUDA_GDN_ALIGN_BOUNDARY` boundary, **default ON / opt out with `=0`**); `make-patches.sh`
  default tip = `33ccf7e28`.
  The beta block-15 patch is applied manually on top of that tree.
- Superseded/still-useful artifacts: the work branch `wip/block15-campaign-wins` (`b26ae06f0`) and
  `wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch` remain as the pre-merge
  record; the per-win patches/plans under `wip/arch-independent-memory/` + `wip/qwen4exp/qsa-memory/` are
  the designs (V3-DERIVED-KQ-MASK-PLAN.md, V4-NATIVE-Q8-KV-PLAN.md, DERIVED-MASK-DESIGN.md).
- Upstream PR candidates: `upstream/README.md` - **the backlog is empty: all four are written up**
  (the allocator view-release probe, the sched probe, the keys-only indexer cache A1, and the `attn_k`
  null-mask guard A2), each with its own `.md` evidence verified on pristine master `9cf3bf256`.

### gfx1201 (RDNA4) port of the gfx1151-gated Halo campaign items — ACTIVE (final qwen4exp stretch)
- The sched-gate fix (fork `c63f7f2a0`, delivery `d6eb551`) is CLOSED on BOTH arches
  (2026-09-06): gfx1201 clean-box tensor-split A/B 3/3 no-hang 2130-2156 t/s (pre-reboot
  flake = degraded box; no bisect, no gate change); gfx1151 same-session parity vs the
  pre-fix campaign build + pure-gate byte-identity (627506c1c vs c63f7f2a0).
- The remaining gfx1151-gated campaign content (routed-compact MoE MMQ, quantize chunk,
  split_j/config rows, per-file RDNA3_5 rows) is INERT on gfx1201 today and needs RDNA4
  port + per-arch tuning + validation, plus the model-level beta ladder + cross-arch
  coherence, then a lighter RDNA3 (gfx1100) env-opt-in follow-up.
- LIVE WORKLOG / implementation plan: `wip/qwen4exp/gfx1201-porting.md` (phases 0-5 +
  dated entries).  Track items there; this TODO entry is the pointer.

### Strix Halo (gfx1151): prefill-gap follow-ons after WS4 (WS3 #2/#3)
- WS3 #2 (QSA dense-shortcut below the selection width) DONE + DEFAULT
  ON (2026-09-05, A commits `a2f2a6ceb` ggml fix + `250e48e97` flip;
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
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws3-shortcut-fix.md`.
- WS3 #3 (routed-compact MoE mmq for the i-quants) DONE (2026-09-05, A
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
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws3-routed-moe-mmq.md`.
- Remaining on the Strix prefill leg: (a) ~~gfx1201/RDNA4 validation of the ggml fix
  (patch 6)~~ DONE 2026-09-06 on gfx1201 (multi-GPU, the sched-gate fix restores the full
  sync there) AND gfx1151 (single-device parity + pure-gate byte-identity — see the
  gfx1201-porting worklog 2026-09-06 entry); the ggml fix is prepared as an upstream PR
  candidate (`beta/qwen4exp/UPSTREAM-PR-ggml-sched-probe.{md,patch}`) — filing is the
  maintainer's call after a clean-upstream build + coherence check; ~~WS3 #3 gate
  (patch 5) via the delivery flow~~ MOVED to `wip/qwen4exp/gfx1201-porting.md` Phase 1.1
  (RDNA4 enablement + own J sweep — caps do not transfer); (b) the
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
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
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

### Strix: REMAINING code-changing items (2026-09-06 consolidated list; all PREFILL - decode is closed)
- (a) MoE topk-moe fusion adoption (~0.5%, prefill): replaces the full-512 argsort
  (25ms/capture) with the fused partial top-10. NUMERICS FORK - fused topk moves top1 logit
  18.424 -> 18.690 (B 18.086); needs a CPU-reference + PPL/KL quality gate before any
  adoption; ~188MB arena cost if adopted. Detail in launch-overhead-topk record.
- (b) ssm_alpha+ssm_beta single-walk fusion (~0.3-0.6%, prefill): [2560x48]x2 MMs on the
  shared hc_mixed (36/48 recurrent layers); stacked-rocblas M96 measured 1.43x, NOT
  bit-identical (~3e-7, user-accepted "essentially correct"). Blocked by graph expansion
  order (alpha-MM@52/beta-MM@58 non-adjacent). Routes: load-time stacked weights (~0.3%) or
  qwen4exp graph restructure + custom single-walk kernel (~0.6%). Design in cijk-dense-gemm.
- (c) launch-ledger remainder (small-pp prefill): +38 scale_f32/eval + rms_norm<256,true>
  count diff + deep fusion-surface diffs after the scale-unary fix; likely sub-0.2%, partly
  architectural (A fuses MORE scale_unary-sigmoid than B; B has gated-silu A lacks). Root-
  cause-only value.
- (d) mmq accumulator-overflow latent defect (I < nwarps*16) - report upstream (correctness
  hygiene, no perf value).
- (e) gfx1100/gfx1201 deferred validation: the gdn NW16 retune (376f02aa0, patch 20/21) needs
  launch-fitness on gfx1100 (~106K VGPRs/CU vs possibly 64K classic -> revert to NW8
  constants if it fails); split_j/config + quantize-chunk + fattn row also re-check on other
  arches. Small code change ONLY if hardware testing fails.  MOVED to
  `wip/qwen4exp/gfx1201-porting.md` (Phase 1 = gfx1201 RDNA4 port/tune; Phase 4 = gfx1100
  env opt-in + fingon campaign).
- NOT worth pursuing: decode fq-inline-quantize port (wash-to-negative - A already launches
  fewer kernels/step and sits at wall parity); GDN +72 launches (cosmetic).


- FINDING (2026-09-06, record `wip/archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`):
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

### RDNA3 (gfx1100) qwen4exp opt-in ungating — design open (maintainer 2026-09-06)
- The gfx1100 box (fingon) is a single 24 GiB GPU — the 87 GiB IQ4_XS model cannot load
  there; validation is light, on small models (Qwen3.6-35B-A3B class).  Working plan: an
  env-level opt-in that un-gates the gfx1151 work for gfx1100/RDNA3 (community members with
  RDNA3 capacity opt in; default = llama.cpp's slower plain qwen4exp support).  Safety
  audit of what may be un-gated (validated on gfx1100: block-13 fused MoE MMQ + QSA decode;
  needs gfx1100 checks: GDN NW16 scan launch fitness ~106K VGPRs, split_j/config,
  quantize chunk, routed compact, hc hyperconn + PLE/weighted-down) + the short fingon
  campaign are Phase 4 of `wip/qwen4exp/gfx1201-porting.md`.

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

- 2026-09-06 sched-gate fix (fork `c63f7f2a0`, delivery `d6eb551`) validated on BOTH
  arches: gfx1201 clean-box tensor-split A/B 3/3 no-hang pp8192 ub2048 at 2130-2156 t/s
  (pre-reboot pass-once-then-flake = degraded-box artifact; no bisect, no gate change),
  decode tg128 49.1; gfx1151 same-session parity vs the pre-fix campaign build (±0.6%
  all rows, campaign anchors reproduced) + pure-gate pair (627506c1c ungated vs
  `c63f7f2a0` gated) same-seed text BYTE-IDENTICAL + perf parity — the gate is inert at
  1 async device.  Coherence note: delivery qwen4exp text vs pre-re-base builds differs by
  upstream GDN-norm fix `5fdfa6282` (in the `465e49b9c` re-base), NOT the gate.  Records:
  `beta/qwen4exp/README.md` (2026-09-06 bullet), `beta/qwen4exp/HALO_HANDOFF.md`,
  `wip/qwen4exp/gfx1201-porting.md`.
- 2026-09-06 IQ3_XXS / IQ4_XS shard-1 "truncation" = non-issue: shard 1 is a
  metadata-only 10.9 MB first shard (n_tensors=0); on-disk sizes match the HF tree
  byte-for-byte; no re-download.

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
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
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
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md`.
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
  `wip/archive/qwen4exp/discovery/2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`.
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
- Memory campaign (wins, V3/V4 plans, Block-0015 staging): `beta/block-15-campaign-wins/HANDOVER.md`
  + `README.md` + `BETA-TESTING.md`; upstream PR candidates: `upstream/README.md`.
