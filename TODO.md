# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.  **Forward-looking only**: this
file lists what is still open (active work, waiting items, accepted limitations, parked ideas) and
keeps closed work as a one-liner with a pointer to the dated record.  Details never live here — they
live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`, `GREEDY-PURITY.md`, `beta/*`,
`wip/*` and `benchmarks/`.

**Current state (2026-09-12):** the delivery is the 15-patch set against fork point `9113cc188`
(block 00 + blocks 01-14), canonical tip **`13af95ac1`** (tree `f4791066f4a582316b1ca95f51c96cd10b905ef7`),
`make-patches.sh` default tip = `13af95ac1`.  Block 15 (the attention-memory campaign) is **staged in
`beta/block-15-campaign-wins/`, not promoted** (12th re-cut: `13af95ac1` → `888a59ee0`).  F1/F2/F3 (the
KV-quant purity/parity campaign) are **all closed** — every KV cache type the delivery supports is
width-pure and takes the f16 attention path — and so is the gfx1151 within-band mmvq fusion variance
(block-13 amendment, 2026-09-12; see Closed).  The QSA *sparse* regime was re-measured on gfx1151
2026-09-12: default configs are pure (item 7 closed); one prompt-dependent **q8_0** forced-sparse
residual is tracked in item 4.

## Active

### 1. Block 15 promotion — **UNBLOCKED** (waiting on the beta window + the maintainer's go-ahead)
- **Live state:** the 12th re-cut is on the current base (`13af95ac1` → beta tip **`888a59ee0`**, tree
  **`476d2d1e95947de7cc8cd806c40efc0f01927cd3`**); it builds clean, applies strict `git am`, and
  revalidates (width probe `W = 1,4,8` one hash, same-seed greedy byte-identical delivery-vs-beta,
  `FLASH_ATTN_QSA` + `GATED_DELTA_NET` pass).  The dense-arm blocker and its fix are closed — see the
  Closed section; the cut is in `beta/block-15-campaign-wins/` (BETA-TESTING.md 12th-re-cut section).
- **Now gating the promotion:** only the ~4–5 day beta window + the maintainer's go-ahead.  Six wins, gates:
  W1 `GGML_QSA_SCORE_MEM`, W2 `GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`, W3 `LLAMA_QSA_KEYS_ONLY`,
  W4 (no gate; `ab/w4-revert.patch`), V3 `LLAMA_KQ_MASK_DERIVED`, V4+V5 `GGML_CUDA_FA_KV_NATIVE` (opt-in).
  Tester material: `BETA-TESTING.md` — its gate list now includes the perplexity oracle
  (`tools/qsa-ppl-oracle.sh`, which is what caught this) **and** the dense-arm text/random-text gates.
- **Post-fix gates (all against the delivery build, identical configs):** oracle sparse `6.5394` / dense
  `6.5377`; dense texts tensor f16 `2daa19579316`, tensor `iq4_nl` `3c46e47ab345`, layer f16
  `e656b50f2cc8`, layer f16 `-fa off` `b96459bf02ca` (all == the delivery); random text `19.0589` /
  `7.9682` (== the delivery); production arm untouched (sparse f16 `804de0576868`, q4_1 `886292b17a93`,
  `plain == n_max 3 == n_max 7`, MTP f16 bit-identical `0.56028`/`(0.681, 0.553, 0.447)`,
  `LLAMA_QSA_OFF=1` `6.5376`); KV reserves unchanged; backend suites OK.
- **Accepted caveat (do not re-report):** W2's derived per-block bias is not bit-exact for `iq4_nl` — its
  greedy text (`fcb2d47f94cf`) and MTP acceptance (`0.46203` / pos-1 `(0.717, 0.434, 0.226)`) differ from
  the delivery's while the sparse-arm PPL is identical (`6.5244`), and `GGML_QSA_DERIVED_*=0` restores the
  delivery's values exactly; the last ULP flips an indexer top-k boundary.  See `BETA-TESTING.md` §4d.
- **Optional follow-up (would make V5 free, not needed for the opt-in delivery):** the native-bf16 loss
  is not the conversion (native staging measures within 0.2 % of an f16 cache) but the removed F16
  scratch, which was a *dense, normalised* copy of the cache view (for a 4-KV-head model `nb[1]` is 4×
  the row size — the GQA heads are interleaved), while the native path re-reads that interleaved view on
  every staging pass.  Options: (a) restrict the arm to `nb[1] == ne[0]*2` layouts (1 line, then free
  there), (b) make the native staging read densely, (c) make the KV cache non-interleaved (llama.cpp-wide).
  Design: `wip/arch-independent-memory/BF16-NATIVE-KV-PLAN.md` §9.

### 3. qwen4exp `iq4_nl` prefill delta (~8–12 %, open — profiled to be host/launch-side)
- Measured on the reference `-sm tensor`: `iq4_nl` 2303.1/2421.0 t/s at pp8192 (sparse/dense) vs f16
  2615.5/2736.2 and `q4_0` ~2597 — and the gap grows with context (pp32768 1992.1 vs 2434.5).  Dense
  models are unaffected (27B within 0.7 %, 4B −2 %), and `q4_0` has the **identical 18-byte layout**.
- **Not** the new code: `rocprofv3` puts the QSA kernel's `iq4_nl` instantiation within 1.3 % of
  `q4_0`'s (same VGPR/LDS/occupancy), the dequant kernels at an identical 1.2 ms, the *executed graph*
  identical (1010 nodes, 0 diff), and the traced kernel *sum* lower for `iq4_nl` — while the wall clock
  is slower and host CPU is +95 ms/token in the forced-sparse-decode case (11.9 vs 41.9 t/s; *not* the
  production arm — the arch policy uses dense decode and still wins by 5 %).
- **Before re-measuring: this axis is noisy on this host.**  On 2026-09-12 the *same* qwen4exp config
  (pp2048, f16 KV, 3-GPU tensor) drifted 2042.6 -> 1933.7 -> 1906.1 -> 1822.0 -> 1730.8 t/s over one
  session (−15 %, box at 141 GiB buff/cache with swap full) while a 35B-A3B pp512 control reproduced
  to 0.2 %; use only same-session interleaved brackets, and note that `LLAMA_QSA_OFF=1` shifts pp2048
  by only +2.2 % / pp8192 +7.4 %, so the QSA machinery does not explain the drift.  See the 2026-09-12
  WORKLOG entry.
- Leads: the dense/sparse topology-flip sync the qwen4exp graph documents, and the per-type indexer op
  counts (`iq4_nl` runs *fewer* `k_argsort`/`soft_max` dispatches than `q4_0`).  Instruments:
  `rocprofv3 --kernel-trace` + the `[GD]` graph dump (`wip/kv-quant-purity-followups/tools/`), and
  `tools/qperf.sh` for the interleaved per-type table.  Analysis: `GREEDY-PURITY.md` §22.

### 4. QSA *sparse*-regime width purity (re-scoped 2026-09-12; default configs pure)
The two items previously recorded here were re-measured on gfx1151 (2026-09-12, after the block-13
RDNA3_5 mmvq-fusion fix) and **do not reproduce**: the fused indexer score is byte-identical to the
per-op chain (512-token forced-sparse A/B), and the "residual split" was the block-13 single-token mmvq
fusion (§25) — the pre-fix divergence reproduces only with
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`.  **Default gfx1151 configs are pure** (shallow dense:
every KV type incl. q8_0; deep sparse at ~74K: f16 `83e0ed0f0f80`, q8_0 `7205399d367d`), so the 64K
crossover stays.
- **Open (low severity, re-scoped again 2026-09-12 (6) to a *driver-level* divergence).**  A
  prompt-dependent **q8_0** dependence in the *forced*-sparse shallow regime
  (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, `/tmp/p5000.txt`: `plain a57bc13bbf2a` vs `n3 3124adfd2b94`).
  `LLAMA_QSA_SPARSE_FA=0` does not fix it; `LLAMA_QSA_OFF=1` does; `GGML_CUDA_DISABLE_FUSION=1` and
  `GGML_CUDA_GDN_CHUNKED=0` "reconcile" only by perturbing the trajectory (both move the plain stream
  early), so neither localises it.  **Deep dive (2026-09-12 (6)): it is not a width dependence** —
  teacher-forced replay at every verify width, with batch+rollback schedules and unrelated rolled-back
  tokens, is bit-pure (200 positions, `W=1..8`); the snapshot rollback restore is exact; `n_rs_seq`,
  `n_outputs_max`, CUDA-graph capture and the chunked-GDN boundary (call sequence *identical* between the
  runs) are all ruled out.  Sharp signature: pure at `--spec-draft-n-max 1`, and all `n_max 2/3/5/7`
  land on the same divergent text (first diff at char 458).  **Two sub-items:**
  * **(a) `embeddings_nextn` breaks logits-level plain==MTP on qwen4exp** (real defect, fixable on its
    own): the MTP driver enables the target's export (`common/speculative.cpp:1431`), which makes the
    last-layer output gather defer (`gather_now` in `src/models/qwen4exp.cpp`) so the last layer runs on
    the full ubatch, shifting the **prefill's last-position logits by a ULP** (`ad3acaa7…` vs
    `b624a79f…`).  Fix direction: keep the output path bit-identical (gather early for the logits, export
    the full rows) or accept and document.  It does not by itself flip the replayed tokens.
  * **(b) the char-458 token divergence itself** — driver-level; needs a faithful mini-MTP driver
    (target + draft contexts, `embeddings_nextn`, real proposals + driver rollback, per-step target-logit
    dump) to find the first step whose logits differ.  Everything cheaper is exhausted.
  Instruments: `wip/strix-halo/qsa-item4/{mstep,rbprobe}.cpp`; record
  `wip/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md` (the earlier disposition is
  `wip/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`); analysis `GREEDY-PURITY.md` §18.

### 5. Strix Halo (gfx1151) prefill-gap follow-ons (all prefill; decode is closed)
Consolidated list with the records under `wip/archive/qwen4exp/discovery/`:
- (a) **MoE topk fusion adoption (~0.5 % prefill)** — replaces the full-512 argsort with the fused
  partial top-10.  **Numerics fork**: top1 logit 18.424 (unfused) vs 18.690 (fused); needs a CPU
  reference + PPL/KL quality gate before adoption.  ~188 MB arena cost if adopted
  (`2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`).
- (b) **`ssm_alpha`+`ssm_beta` single-walk fusion (~0.3–0.6 % prefill)** — blocked by graph expansion
  order (the two MMs are non-adjacent); routes: load-time stacked weights or a qwen4exp graph
  restructure + custom kernel (`2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`).
- (c) **launch-ledger remainder** (small-pp): +38 `scale_f32`/eval, `rms_norm<256,true>` count diff,
  fusion-surface diffs — likely sub-0.2 %, root-cause-only value.
- (d) **mmq accumulator-overflow latent defect** (`I < nwarps*16`) — report upstream (correctness
  hygiene, no perf value).
- (g) **V3 prefill cost is arch-dependent (low priority)**: gfx1151 measured −3.2 % at pp20480 (4B, q8_0)
  vs the RDNA4 reference −1.3 %, decode flat.  Still a large net win (−799 MiB compute + −799 MiB host)
  and on by default; if an iGPU tuning pass ever runs, the derived MMA kernel's `J`/occupancy on gfx1151
  is the place to look.

### 6. Cross-arch / gfx1100 validation (the gfx1201 port + its Phase 2.5 probe are DONE — see Closed)
- **Still open, needs other hardware:**
  * gfx1100 (`fingon`, 24 GiB): the §4.2 remainder with *no* gfx1100 data yet — the GDN gfx11 NW16 scan
    retune (~106K VGPR/CU vs a possible 64K classic), `split_j`/config rows, the quantize chunk,
    routed-compact, the hc/PLE fusions, and the two block-13 MTP regression fixes under RDNA3
    (acceptance gate).
  * gfx1151 (`halo`): Phase 3's cross-arch fingerprint check (gfx1201 == gfx1151 numerics) — a
    verification goal, not a port; also item 4's §18 items and item 7's MTP crossover re-measure.
- **Tracker hygiene:** the plan's own open checkboxes are **stale** (Phase 1 is complete and the doc
  predates qwen4exp's promotion to block 14); read the banner at the top of
  `wip/qwen4exp/gfx1201-porting.md` before trusting them.

### 8. Dual 7900XTX (gfx1100, community): block-12 validation
- Hybrid HIP all-reduce on RDNA3 **pairs** is being validated by a community member on their dual-7900XTX
  box (hybrid-dispatch matrix internal/nccl/none + the bounded-spin path at depth-16384).  The block-12
  arch gate stays RDNA4-only until then.  Volunteer env: `GGML_CUDA_ALLREDUCE=internal`.
- The block-13 gfx1100 leg is DONE (single-GPU 7900 XTX, §Closed); what remains here is block 12, which
  is N/A on a single-GPU box.  Where: `patches/0012` + the block-12 notes in `patches/README.md`.

### 9. QSA knobs: a tensor-tuned prefill crossover + the fused-op probe (small, from F3)
- **`LLAMA_QSA_DENSE_PREFILL_UNTIL`-style gate**: the prefill crossover is not depth-configurable today
  (reference `-sm tensor`: dense wins pp8192 by ~4.7 %, parity at pp16384, sparse wins pp32768 by
  +14.5 %).  Tensor-tuned, per the maintainer's rule that the crossover policy follows the tensor split.
- **`LLM_FUSED_OP_FLASH_ATTN_QSA` probe** so `qsa_kv_native` stops duplicating the backend predicate —
  note the probe compares the *device* a fused node lands on, which does not by itself catch a
  meta-split inconsistency.

### 11. MXFP4 (and NVFP4) fused gate+up+GLU MMQ — the last block-13 item
- `ggml_cuda_mul_mat_q_switch_type_gate` is instantiated for Q3_K/Q4_K/Q5_K/Q8_0/Q6_K only, so the
  `try_fuse` arm is gated on that list (MXFP4/NVFP4 would abort if admitted).  MXFP4 is the interesting
  type for future native-MXFP4 MoE models.
- "Done" = add the switch cases + instance files + generator entries, then bit-exact + bench validation
  per the `0004` recipe.  Tracked from the block-13 notes in `patches/README.md`.

### 12. Upstream: file the staged PR candidates
- `upstream/README.md` — five are written up and evidence-verified on pristine master `9cf3bf256`:
  the ggml-alloc unused-view release, the sched probe, the keys-only indexer cache (A1), the `attn_k`
  null-mask guard (A2), plus `UPSTREAM-PR-fa-decode-verify-kernel-family.{md,patch}` (the F1 chooser fix,
  whose NVIDIA/Ada half the fork deliberately does not land — see the AGENTS.md scope policy).
- Filing is the maintainer's call.

### 13. Upstream monitor: ROCm unaligned-width split-load (Q6_K/Q3_K, 2-GPU)
- Upstream bug: H2D 2D copies whose width is not a multiple of 4 (Q6_K block = 210 B, Q3_K = 110 B) are
  ~1000× slower on ROCm.  Fixed locally in block 13 (`set_tensor_2d`: aligned H2D + unaligned D2D
  staging).  No PR planned (upstream is busy with its own qwen4exp work) — watch whether they fix it
  themselves; if so it surfaces as a re-base conflict and resolves naturally.  Re-check at each re-base.

### 14. Canonical-fork hygiene (do this before ANY regeneration)
- The working `~/llama.cpp` `rdna-boosts` branch is a local rebuild and must **not** be used for
  regeneration while it sits on a master newer than the fork point (it would export `f3f1a8f27` +
  `304665fe7` as patches 0001/0002).  Regenerate from a canonical fork rebuilt at `9113cc188` by
  `scripts/apply-all.sh` — currently tip `13af95ac1`, tree
  `f4791066f4a582316b1ca95f51c96cd10b905ef7`.  See `BASELINE.md`/`AGENTS.md`.
- Superseded-but-useful artifacts: the work branch `wip/block15-campaign-wins` (`b26ae06f0`) and
  `wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch` are the pre-merge
  record; the per-win plans under `wip/arch-independent-memory/` + `wip/qwen4exp/qsa-memory/` are the
  designs.

### 15. Enable `-Wshadow` for `src/` (would have caught the Block 15 dense-arm bug as a compile error)
The 2026-09-11 (11) blocker was a one-token shadowing bug (`ggml_tensor * kq_mask_top_k = ...` inside a
block that already had an outer declaration of the same name) that made a whole mask chain dead code —
silent because the code still compiles and the chain still gets built.  `-Wshadow` reports it directly.
Not currently enabled anywhere in the build.  **Audited 2026-09-12 (7)**: replaying the tree's own host
compile commands for the `llama` target (186 `src/` TUs) with `-Wshadow` gives **128 warnings in 27
files**, of which **46 are the risky `shadows a local variable` class** (the Block-15 class) and 82 are
benign `shadows a field` (mostly constructor params).  `src/models/qwen4exp.cpp` is clean — the delivery
does not carry the bug.  **Revised proposal**: enable `-Wshadow -Wno-shadow-field-in-constructor` for
`src/` (kills the constructor-param noise) and fix the ~46 local-variable sites (mechanical renames);
doing that touches ~20 upstream `src/` files, so it wants its own block/cleanup commit to avoid colliding
on every re-base.  Record: `wip/shadow-warnings/RECORD-2026-09-12-shadow-audit.md` (with the full
46-site list).  Reference: `GREEDY-PURITY.md` §23.3, `WORKLOG.md` 2026-09-11 (11).

### 16. Restore the block-13 RDNA3_5 single-token fusion perf (low priority, gfx1151)
- The block-13 RDNA3_5 mmvq purity amendment (Closed) skips the two single-token-only fusions at a cost of
  ≈ −0.9 % `tg128` on qwen4exp (25.53 vs 25.77 t/s; prefill flat).  **Re-scoped 2026-09-12 (7): the
  "pin `nwarps`/`rps`/item-split" plan does not apply** — the fused and unfused dense arms already share
  the same kernel template (`mul_mat_vec_q_ksplit<...,has_fusion,...>`), the same `calc_nwarps`,
  `rows_per_block` (= 1 on RDNA3_5) and launch dims, and the fused epilogue uses the same
  `ggml_cuda_op_silu_single` as the standalone GLU (`op_silu`).  Two live candidates: **(a) codegen** —
  `has_fusion=true` adds registers + a second `vec_dot` in the inner loop and may contract the `tmp` (up)
  FMAs differently; **(b) the Q8_1 cache** (`common.cuh:1611`, keyed on the src1 tensor/layout only, *not*
  the weight type, while `quantize_row_q8_1_cuda` takes `src0->type`) — fusing changes which call fills it.
  Next step: dump `tmp`/`tmp_gate` from the ksplit kernel under an env at `W=1` and compare the `up`
  values bit-for-bit (match → epilogue/cache; differ → codegen).  A/B:
  `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`.  See `GREEDY-PURITY.md` §25 and
  `wip/strix-halo/rdna35-mmvq-fusion-purity/README.md` §5/§7/§9.

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **Mixed K/V cache types fall off the GPU attention path.**  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`)
  gives `graph splits = 18`, a ~1.5 GiB host compute buffer and pp2048 7924 → 640–1049 t/s on the 4B.
  Maintainer policy (2026-09-11): **reject differing K/V types** — every mixed pair is 1.7–3.6× slower
  and never smaller; upstream already enforces same-K/V for DeepSeek V4 (#25871).  **Decided and
  implemented 2026-09-11 (12): hard-rejected at context creation** — `params.type_k != params.type_v`
  now fails `llama_init_from_model` with a message naming both types and telling the user to set
  `--cache-type-v` to match (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `--cache-type-k`/`-v` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter (`ggml-backend-meta.cpp:1177`)
  because its 2 KV heads are fewer than the 3 devices (one device gets a zero-extent share).  Works on
  1/2 GPUs and on 3 GPUs with `-sm layer`; maintainer's call: no fix.  Every other model is unaffected
  (a future block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).

## Parked (not planned now)

### LFRU host→GPU slow hot-weight migration
- Survivor of the expert-tiering experiment (dropped 2026-09-05 — most of its aims are already covered by
  current llama.cpp options).  The one idea left: an LFRU-style very slow migration of hot weights from
  host to GPU (persistent GPU slot cache + CPU-computed cold tail).  Design notes:
  `wip/qwen4exp/LRU_EXPERTS.md`, `PHASE0_ROUTING.md`, `HANDOVER-2026-09-04-tiering.md`.

## Closed (one-liners; details in the dated docs)

**Item 5(f) — the block-13 fused MoE gate+up+GLU arm still wins on Strix Halo (closed 2026-09-12).**
Re-measured on the current delivery tip (35B-A3B Q4_K_M, 1 GPU, `-p 2048`/`-p 16384`, interleaved
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` off/on ×3): fusion active **+0.6 %** at pp2048
(1711.9/1710.2 vs 1710.1/1701.4 t/s) and **+0.6 %** at pp16384 (1485.3/1485.8 vs 1476.4/1478.6), the
fusion fires, prefill absolute ~1710/1485 t/s.  So the arm is **kept** (a small but real Strix win).

**The gfx1151 dense-decode-at-every-depth policy (TODO item 7, closed 2026-09-12).**  The proposed
workaround (force gfx1151 decode dense at every depth, so the sparse regime becomes unreachable) was
motivated by the sparse regime's recorded width impurity.  Re-measured 2026-09-12: the two recorded
items were artifacts of the block-13 RDNA3_5 mmvq fusion (fixed the same day), and the sparse regime is
**pure** in the default configs (deep sparse ~74K: f16 `83e0ed0f0f80`, q8_0 `7205399d367d`), so gfx1151
**keeps the 64K crossover** (sparse wins deep decode).  A pure per-*perf* MTP-side crossover re-measure
is parked — no purity driver.  The one residual is the q8_0 forced-sparse item now tracked under item 4;
record `wip/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`, analysis `GREEDY-PURITY.md` §18.

**The gfx1151 within-band mmvq fusion variance (block 13, closed 2026-09-12 (2)).**  The 2026-09-11
block-13 band work made the *standalone* mmvq path `W = 1..8`-uniform, but on gfx1151 two
**single-token-only** fusions still ran at `W=1` only and their fused kernels do not reproduce the
standalone arithmetic, so a 1-token decode and an n-token verify of the same layer were not
bit-identical (the issue-25 "block-13 `n_q=1` short-K mmvq variance"): the dense gate+up+GLU mmvq fusion
(`mul_mat_vec_q<..., ncols=1, has_fusion=true>`) and the MoE weighted-down tail
`ggml_cuda_mul_mat_id_weighted_rdna3_5`.  Fixed by guarding the six `{op,op,GLU}` /
`{op,bias,op,bias,GLU}` matchers in `ggml_cuda_try_fuse` (keeping the band-uniform `MUL_MAT_ID`/MoE
fusions) and `ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`, both RDNA3_5-only unless
`GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`.  Post-fix `W = 1,2,4,8` one hash per config: qwen4exp
f16 `453eaa61`, q8_0 `113696b9`, MoE 35B-A3B `18999a78`; the 27B dense (`e165ef98`) was already pure and
is unchanged; cost ≈ −0.9 % `tg128` (the purity-first trade, follow-up = item 16).  Canonical tip
`13af95ac1`, tree `f4791066f4a582316b1ca95f51c96cd10b905ef7`; `GREEDY-PURITY.md` §25, `WORKLOG.md`
2026-09-12 (2), `wip/strix-halo/rdna35-mmvq-fusion-purity/README.md`.

**The fused shared-expert epilogue's band cost (TODO item 10, closed 2026-09-12).**  The
band-uniformity fix's `grid = (nrows, ncols)` launch shape (one block per `(output row, token)`,
down-weight row re-read per token, 7 of 8 warps idle on the 35B-A3B geometry) is replaced by a
`ncols_dst`-templated kernel with the token loop inside the k-block loop and `grid = (nrows)` — a
**bit-identical** restructure (old-vs-new `.so` A/B: every gate hash equal, incl. the MoE probe
`W = 1..8` `ac8825358d9adfda` and MTP `0.87179`) that repays the item-5 cost: `pl=8` 461.0 -> 475.4
t/s (+3.1 %), `pl=4` 299.1 -> 306.5 (+2.4 %), `pl=1` flat, and the fused default now beats the
unfused reference at every width.  Block-13 patch anyway; see the 2026-09-12 WORKLOG entry,
`patches/README.md`'s 2026-09-12 section and `GREEDY-PURITY.md` §24.

**The gfx1201 (RDNA4) port of the gfx1151-gated campaign items (2026-09-11 (12) note — mostly closed
2026-09-06/07).**  Every gated kernel was ported and is enabled by default on RDNA4: the
**routed-compact MoE MMQ** (`mmq_routed_compact_arch_ok() = RDNA3_5 || RDNA4`,
`2026-09-06-gfx1201-rdna4-routed-moe-mmq.md`: +4-8 % prefill, byte-identical, `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
to A/B), the quantize chunk (flat, kept), and the block-13 fused MoE gate+up+GLU MMQ is **ungated
outright** for RDNA3_5 *and* RDNA3_0 (2026-09-05).  **Phase 2.5 (the fallback-path probe) is DONE
2026-09-12:** the routed-compact path's "bit-identical" claim holds on both MoE models (qwen4exp IQ4_XS
text `804de0576868`, 35B-A3B Q4_K text `68c0a24ed8d4`, both identical with `GGML_CUDA_DISABLE_MMQ_ROUTED`
on/off; `W = 1..8` and MTP `0.87179` identical) and the perf reproduces (+4.0..+11.1 % / +5.1..+7.8 %
prefill, tg flat), with two wording corrections: the Q4_K model *does* take the routed path (480
`mul_mat_q_routed_compact` launches per pp512 — the real control is that the dispatch is prefill-only, 0
launches in a `tg` run), and `GGML_CUDA_DISABLE_MMQ_ROUTED=1` isolates only the compact *enumeration*
(the per-expert J selection stays active in both arms).  What remains is validation on other boxes — see
item 6.  The plan doc (`wip/qwen4exp/gfx1201-porting.md`) carries a status banner; its checkboxes are
stale.

**Issue #25's GDN plain-vs-MTP divergence (2026-09-11 (12)) — FIXED and re-verified.**  The `K`-dependent
chunked/sequential boundary in `gated_delta_net.cu` was removed by block 02's **K-independent whole-batch
chunked prefill** (Option B, 2026-09-11): both the plain (`K == 1`) and the MTP (`K == n_max + 1`) prefill
now make the *same* call, so the post-prefill state no longer depends on `n_max`; `GGML_CUDA_GDN_ALIGN_BOUNDARY`
and both K-dependent branches were deleted and `GGML_CUDA_GDN_CHUNKED=0` remains as the A/B switch and the
fully-snapshot-safe fallback.  **Re-verified 2026-09-11 (12) on the current tree** (27B Q8_0, 2-GPU
`-sm tensor -ts 1/1`, `p0long.txt`, 512 greedy tokens, `-c 8192 -ctk f16 -ctv f16 -fa auto`):
`--spec-type none == draft-mtp n_max 1 == 4 == 5` → all `299566b902bb` (2727 chars), byte-identical.
(With `GGML_CUDA_GDN_CHUNKED=0` the plain text differs → `60777872b890`, which is the expected
chunked-vs-sequential kernel difference, not a plain-vs-spec divergence.)  The stale status lines in
`wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-{FOLLOWUP,FIX}.md` (they still describe the opt-in
`GGML_CUDA_GDN_ALIGN_BOUNDARY` fix, a gate that no longer exists) are corrected there.

**Block 15 dense-arm blocker (2026-09-11 (11)) — FIXED, one line.**  `LLAMA_QSA_SPARSE_FA=0` gave PPL
`1.0558` for every KV type because the top-k mask chain's `ggml_tensor * kq_mask_top_k` shadowed the outer
declaration added by the V2/V3 refactor, so the attention got a null mask (a full causal leak).  Found via
the node dump (the map: the delivery consumed `attn_inp_kq_mask` 36×, the beta 0×) and a `[QDM]` log
(`kq_mask=1` … `outer_top_k=0`).  Ninth beta re-cut: base `6d3155faa` → tip `3712e2dc1`, tree
`e39f8c2b6f0593113b93c4e57c512bc7373a2250`, patch 3 811 lines; oracle sparse `6.5394` / dense `6.5377`,
dense texts and random-text PPL byte-identical to the delivery, production arm untouched.  Details:
`wip/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md`, `WORKLOG.md` 2026-09-11 (11),
`GREEDY-PURITY.md` §23, `beta/block-15-campaign-wins/BETA-TESTING.md` §4c.  Follow-ups filed: `-Wshadow`
(item 15) and the residual `iq4_nl` W2 sensitivity (accepted, above).

**KV-quant purity / parity campaign — ALL CLOSED (2026-09-11).**  Brief, evidence and tooling:
`wip/kv-quant-purity-followups/` (`README.md` + `tools/`); analysis: `GREEDY-PURITY.md` §14–§22.
- **F1** (`q8_0`/`q4_0` dense-band impurity) — FIXED as a block-08 amendment: the FA kernel-family
  chooser returned VEC for `n_q <= 2` with a quantized K/V and TILE above; the branch is deleted (the
  whole band is TILE), all four split configs `W=1..8` bit-identical, cost tg128 −0.5…−0.9 %.
- **F2** (qwen4exp width impurity) — all three causes fixed: the HC `nt == 1` gates (block-14 amendment),
  upstream's per-type **mmvq cap** in `mul_mat_vec_q_moe`'s `__launch_bounds__` (block-13 amendment,
  +14–26 % at the verify widths), and the QSA dense decode arm gated `n_tokens == 1` (`QSA_DECODE_BAND
  = 8`, block-14 amendment).
- **F3** (sub-`q8_0` KV parity) — both steps landed: `q4_1`/`q5_0`/`q5_1` (block-08 + block-14 amendments,
  2026-09-11 (8)) and `iq4_nl` (block-08 + block-14 amendments, 2026-09-11 (10); 4B pp512 2269.8 → 7931.8,
  tg32 48.5 → 95.0, `FLASH_ATTN_EXT` 5935/5935, `FLASH_ATTN_QSA` 22/22).  The QSA quantized-KV
  enablement also root-caused a **quality bug** every purity gate was blind to (the shared staging tile
  mixed two K/V heads at gqa 12; perplexity 7.33 → 6.53) and added the CPU oracle + `FLASH_ATTN_QSA`
  test.  Open remainders are items 3 and 9 above.
- **F2's superseded framing** ("multi-step / roll-back", "the fused sparse QSA path", "same cause as F1")
  was wrong on all three counts — the corrected record is `GREEDY-PURITY.md` §13–§16.

**Other closed work** (each with a dated record):
- 2026-09-10 block 00 added (FA small-batch KV-split width invariance, issue #25, + the Vulkan masked-V
  fixes); block 06 reduced to a host-buffer rationale marker on the re-base (upstream reverted #24233).
- 2026-09-11 block 02: the K-independent whole-batch chunked GDN prefill (`GGML_CUDA_GDN_ALIGN_BOUNDARY`
  and its K-dependent branches deleted, + rollback guard) — `patches/README.md`.
- 2026-09-06 sched-gate fix (`c63f7f2a0` / delivery `d6eb551`) validated on both arches; the pre-reboot
  "flake" was a degraded box.  Records: `beta/qwen4exp/README.md`, `wip/qwen4exp/gfx1201-porting.md`.
- 2026-09-06 WS4 Strix Halo hc-prefill-fusion gates PASSED (depth-0 pp +5.2–8.8 %, decode flat) —
  `wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws4-hc-fusion-gates.md`.
- 2026-09-06 real determinism root cause fixed (the indexer top-k `atomicAdd` gather scrambled the QSA
  list order run-to-run; replaced with an ascending count/scan/write) + the stale-cell zeroing port.
- 2026-09-05 WS3 #2 (QSA dense-shortcut artifact) root-caused to `ggml_gallocr_reserve_n_probe` /
  the dense↔sparse topology flip and fixed at the ggml level; WS3 #3 (routed-compact MoE MMQ) landed for
  gfx1151, default on.  Records: `...ws3-shortcut-fix.md`, `...ws3-routed-moe-mmq.md`.
- 2026-09-05 block 13's fused MoE MMQ ungated for RDNA3_5 (gfx1151: pp2048 +5.3 %, pp16384 +4.6 %) and
  for RDNA3_0 (gfx1100: pp2048 +9.4 %, pp16384 +7.8 %), both with coherence IDENTICAL and the RDNA4 J
  caps transferring.  Records: `...gfx1151-block-13-moe-mmq.md`, `...rdna3-gfx1100-block-13-moe-mmq.md`.
- 2026-09-05 the scale→unary fusion port (`ggml_cuda_op_scale_unary`, bit-identical, pp2048 +0.34 %).
- 2026-09-05 ITEM B (QSA sparse-FA latency push) closed at ~48 t/s / ~95 % GPU occupancy — the probed
  3× headroom never materialised (register pressure, CU occupancy, VRAM bandwidth).
- 2026-09-05 expert-tiering experiment dropped (see Parked); 2026-09-06 the IQ3_XXS/IQ4_XS shard-1
  "truncation" turned out to be a metadata-only first shard (non-issue).
- 2026-09-01…09-04: block 13 released (fused MoE gate+up+GLU MMQ + mmvq item-split); the multi-token
  MUL_MAT_ID `x_scale_channel_dst` fusion; the ROCm unaligned-width split-load fix; the two block-13 MTP
  regression fixes (mmvq ksplit dispatch for verify batches, the rms_norm fold gated to single-token
  MMID); the block-12 NCCL-failure fallback (issue #13); the qwen4exp WIP promotion to `beta/qwen4exp/`
  and its re-base onto `8b4b3558f` with the MTP draft head.
- Older resolved items (block-12 fused-stage/pacing closure, ITEM A JIT, the indexer head-sum revert,
  qwen35moe dense-GQA N/A, …) are recorded in `archive/docs` + `archive/work`; not tracked here.

## Where the current lists live

- Remaining gfx1151 work + the 2026-09-12 TODO audit: `wip/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- QSA sparse-regime width purity (items 4/7 disposition): `wip/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- Environment, instruments, reference hashes and the landing procedure for KV/FA work:
  `wip/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its items 1/5
  and F3 are **superseded** — see the Closed section here).
- qwen4exp carried-forward open items: `beta/qwen4exp/README.md` ("Open items (carried forward from WIP)").
- Delivery verification contract + dated records: `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- Purity/invariant analysis and the instrument rules: `GREEDY-PURITY.md`.
- Benchmarks + gates: `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- Memory campaign (wins, V3/V4 plans, Block 15 staging): `beta/block-15-campaign-wins/HANDOVER.md`,
  `README.md`, `BETA-TESTING.md`; upstream PR candidates: `upstream/README.md`.
