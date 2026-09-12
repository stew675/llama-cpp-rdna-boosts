# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.  **Forward-looking only**: this
file lists what is still open (active work, waiting items, accepted limitations, parked ideas) and
keeps closed work as a one-liner with a pointer to the dated record.  Details never live here — they
live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`, `GREEDY-PURITY.md`, `beta/*`,
`wip/*` and `benchmarks/`.

**Current state (2026-09-11 (10)):** the delivery is the 15-patch set against fork point `9113cc188`
(block 00 + blocks 01-14), canonical tip **`6d3155faa`** (tree `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`),
`make-patches.sh` default tip = `6d3155faa`.  Block 15 (the attention-memory campaign) is **staged in
`beta/block-15-campaign-wins/`, not promoted**.  F1/F2/F3 (the KV-quant purity/parity campaign) are
**all closed** — every KV cache type the delivery supports is width-pure and takes the f16 attention path.

## Active

### 1. Block 15 promotion — **UNBLOCKED** (the dense-arm defect was found and fixed 2026-09-11 (11))
- **The blocker is FIXED.**  It was a variable-shadowing bug in block 15's own `build_attn_qsa` dense path:
  the V2/V3 refactor added an outer `ggml_tensor * kq_mask_top_k = nullptr;` while the top-k mask chain
  inside `if (kq_mask != nullptr) { ... }` still declared its own `kq_mask_top_k`, so the chain was built
  but its result never reached the attention — `build_attn_mha` got `nullptr`, the chain's nodes were
  unreachable from the graph output, the packed mask lost its only consumer (so it was left unallocated
  and its input fill skipped) and the dense arm attended unmasked: a full causal leak, seen as PPL
  `1.0558` on every KV type where the delivery gives `6.49–6.55`.  One-line fix (drop the inner
  `ggml_tensor *`).  Ninth re-cut: base `6d3155faa` → beta tip **`3712e2dc1`**, tree
  **`e39f8c2b6f0593113b93c4e57c512bc7373a2250`**, patch **3 811 lines**; round-trips exactly, builds clean.
  Full evidence chain, the instruments and the new leak gate are in
  `wip/block15-dense-arm/HANDOVER-2026-09-11-block15-dense-arm.md` (now with an OUTCOME banner) and
  `WORKLOG.md` 2026-09-11 (11).
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

### 2. Issue #25: GDN chunked prefill makes `--spec-type none` differ from MTP (correctness, latent)
- On one build+prompt the plain and MTP streams differ and it is entirely the GDN chunked prefill:
  `GGML_CUDA_GDN_CHUNKED=0` makes `none == n-max 2 == n-max 4` byte-identical (`9216c6d1` → `bba7741d`).
  Cause: `gated_delta_net.cu` branch 1 runs the plain multi-token prefill (`K=1`) chunked while spec
  prefill (`K=n_max+1`) runs sequential; and for prompts `> K+64` branch 2's prefix boundary `n_tokens-K`
  shifts with `n_max`.  Fork-only (block 02 — upstream ships only the sequential kernel).
- Latent in practice (a 2.8k-token MTP run did not flip within 200 tokens) but a real state divergence;
  probe `P=256 RS=from_w` gives W3–W5 = `0.210405` (chunked on) vs `0.000000` (off).
- Repro, fix directions and the validation gate: `wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FOLLOWUP.md`.

### 3. qwen4exp `iq4_nl` prefill delta (~8–12 %, open — profiled to be host/launch-side)
- Measured on the reference `-sm tensor`: `iq4_nl` 2303.1/2421.0 t/s at pp8192 (sparse/dense) vs f16
  2615.5/2736.2 and `q4_0` ~2597 — and the gap grows with context (pp32768 1992.1 vs 2434.5).  Dense
  models are unaffected (27B within 0.7 %, 4B −2 %), and `q4_0` has the **identical 18-byte layout**.
- **Not** the new code: `rocprofv3` puts the QSA kernel's `iq4_nl` instantiation within 1.3 % of
  `q4_0`'s (same VGPR/LDS/occupancy), the dequant kernels at an identical 1.2 ms, the *executed graph*
  identical (1010 nodes, 0 diff), and the traced kernel *sum* lower for `iq4_nl` — while the wall clock
  is slower and host CPU is +95 ms/token in the forced-sparse-decode case (11.9 vs 41.9 t/s; *not* the
  production arm — the arch policy uses dense decode and still wins by 5 %).
- Leads: the dense/sparse topology-flip sync the qwen4exp graph documents, and the per-type indexer op
  counts (`iq4_nl` runs *fewer* `k_argsort`/`soft_max` dispatches than `q4_0`).  Instruments:
  `rocprofv3 --kernel-trace` + the `[GD]` graph dump (`wip/kv-quant-purity-followups/tools/`), and
  `tools/qperf.sh` for the interleaved per-type table.  Analysis: `GREEDY-PURITY.md` §22.

### 4. QSA *sparse*-regime width-dependences (2 items, open; gfx1151's default regime)
Both are in `GREEDY-PURITY.md` §18; gfx1201's default (dense decode) is unaffected.
- (a) `GGML_CUDA_QSA_INDEXER_SCORE`'s "byte-identical" claim is **measurably false** and the probe is
  itself `n_tokens == 1`-gated — unreachable on gfx1201's default, but the default path on gfx1151 above
  its 64K crossover.  Fix = make the kernel token-generic, or default the probe OFF.
- (b) A residual split survives even with one arm (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`: common prefix 706
  chars vs 100, then divergence) — a state/store width-dependence still unlocalised
  (`GGML_CUDA_QSA_INDEXER_CACHE=0` does not reconcile them).

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
- (e) **gfx1100/gfx1201 deferred validation** (the GDN NW16 retune needs launch-fitness on gfx1100,
  ~106K VGPRs/CU vs possibly 64K) — small code change only if hardware testing fails; moved to
  `wip/qwen4exp/gfx1201-porting.md` Phase 1/4.
- (f) **Re-check whether the block-13 gate+up+GLU arm still has a unique win on Strix Halo**: with the
  2026-09-06 model-neutral folds in the tree the isolated fused-MoE delta is now ~0 there
  (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` on vs off: pp2048 +0.4 %, pp16384 +0.2 %; absolute prefill ~10–13 %
  higher, the fusion still fires) — the folds are capturing the same work, not a regression.
- (g) **V3 prefill cost is arch-dependent (low priority)**: gfx1151 measured −3.2 % at pp20480 (4B, q8_0)
  vs the RDNA4 reference −1.3 %, decode flat.  Still a large net win (−799 MiB compute + −799 MiB host)
  and on by default; if an iGPU tuning pass ever runs, the derived MMA kernel's `J`/occupancy on gfx1151
  is the place to look.

### 6. gfx1201 (RDNA4) port of the gfx1151-gated campaign items — ACTIVE (pointer only)
- The remaining gfx1151-gated content (routed-compact MoE MMQ, quantize chunk, `split_j`/config rows,
  per-file RDNA3_5 rows) is **inert on gfx1201** and needs an RDNA4 port + per-arch tuning + validation,
  plus the model-level beta ladder + cross-arch coherence.  Phase 4 of the same plan covers the
  **gfx1100 (RDNA3) env-level opt-in** (the fingon box is a single 24 GiB GPU, so validation is light,
  on 35B-A3B-class models) with its safety audit.
- Live plan and dated entries: `wip/qwen4exp/gfx1201-porting.md`.  Track items there; this is the pointer.

### 7. Strix Halo / gfx1151 bundle (deprioritised 2026-09-11 — gfx1201 first)
- Re-measure the **MTP-side** QSA crossover on the Strix box: the published 64K "dense below, QSA above"
  table was measured for the W=1 decode regime, and the verify batch now takes the dense arm below the
  crossover.  Above 64K the sparse regime is at parity with dense per the controlled 2026-09-07 protocol
  **and** still impure for MTP (§18), so under the purity-first policy the likely answer is **dense
  decode at every depth on gfx1151 too** (a one-line `build_layer_attn` change) once §18 is fixed.
- Needs the Strix Halo box; blocked until then.  See `GREEDY-PURITY.md` §18–19.

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

### 10. Fused shared-expert kernel: column-block it (~2.4 % at the widest verify batches)
- The epilogue band-uniformity fix is `grid = (nrows, ncols)` — one block per `(row, token)` — so the
  down-weight row is re-read once per token (35B-A3B `llama-batched-bench`, `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1`
  as the A/B: pl=8 332.1 fused vs 341.5 unfused; pl=4 252.0 vs 254.2; pl=1 unchanged).
- Fix = the `mul_mat_vec_q` pattern: template the kernel on `ncols_dst` and keep the token loop *inside*
  the k-block loop (one weight read per row block, per-token accumulators), which stays bit-identical per
  token.  Not a purity issue.  Where: block 13.

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
  `scripts/apply-all.sh` — currently tip `6d3155faa`, tree
  `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`.  See `BASELINE.md`/`AGENTS.md`.
- Superseded-but-useful artifacts: the work branch `wip/block15-campaign-wins` (`b26ae06f0`) and
  `wip/arch-independent-memory/snapshots/fork-tree-W1-W2-V3-V4-2026-09-10.patch` are the pre-merge
  record; the per-win plans under `wip/arch-independent-memory/` + `wip/qwen4exp/qsa-memory/` are the
  designs.

### 15. Enable `-Wshadow` for `src/` (would have caught the Block 15 dense-arm bug as a compile error)
The 2026-09-11 (11) blocker was a one-token shadowing bug (`ggml_tensor * kq_mask_top_k = ...` inside a
block that already had an outer declaration of the same name) that made a whole mask chain dead code —
silent because the code still compiles and the chain still gets built.  `-Wshadow` reports it directly.
Not currently enabled anywhere in the build.  Proposal: add it to the fork's HIP/CUDA C++ flags (or at
least to CI) and clean up whatever pre-existing warnings appear; keep it scoped to `src/` first.
Reference: `GREEDY-PURITY.md` §23.3, `WORKLOG.md` 2026-09-11 (11).

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

- **Mixed K/V cache types fall off the GPU attention path.**  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`)
  gives `graph splits = 18`, a ~1.5 GiB host compute buffer and pp2048 7924 → 640–1049 t/s on the 4B.
  Maintainer policy (2026-09-11): **reject differing K/V types** — every mixed pair is 1.7–3.6× slower
  and never smaller; upstream already enforces same-K/V for DeepSeek V4 (#25871).  Open sub-decision
  only: hard error vs warning vs docs-only.
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
