# Strix Halo session brief — 2026-09-10 end state (handoff for the next session)

CONTINUE from this file + ~/make-strix-halo-faster.md + wip/strix-halo/notes-ws1-survey.md +
the dated records. The PRIMARY task (WS3 #2 llama-bench artifact → ggml-level fix → shortcut
default ON) is COMPLETE this session. Read `benchmarks/2026-09-10-strix-halo-gfx1151-ws3-shortcut-fix.md`
for the full evidence. Remaining items are in the ledger at the bottom.

## Commits / tree state (all clean, verified)

- ~/llama.cpp (qwen4exp) tip `1682d32a9`, clean, build current (= tip content):
  - `248e47704` (patch 4) WS4 fusions DEFAULT ON → `151798ed2` WS3#2 shortcut opt-in
    (superseded) → `a1121cf2d` (patch 5) WS3#3 routed-compact MoE MMQ DEFAULT ON → NEW:
  - `fcfb0a522` (patch 6): ggml-level fix — `ggml_gallocr_reserve_n_probe()` +
    sched alloc-fallback syncs only when a buffer must actually GROW (ggml-alloc.c/.h,
    ggml-backend.cpp; +57/-11). Numerics-inert.
  - `1682d32a9` (patch 7): `LLAMA_QSA_DENSE_SHORTCUT` DEFAULT ON (opt-OUT via =0; B parity).
    NOTE: the tree default is no longer the byte-identical "known-good" sparse-selection
    numerics below the 2051-cell width — the default is now the dense-below-width regime
    (= LLAMA_QSA_SPARSE_FA=0 text-identical; =0 env restores the old known-good default).
- ~/llama-cpp-rdna-boosts (delivery): patches 6-7 staged in beta/qwen4exp/
  (`ggml-sched-fallback-sync.patch`, `ws3-shortcut-default-on.patch`; clean-apply verified at
  a1121cf2d, byte-identical to 1682d32a9). TODO reality pass 2026-09-10.
- ~/strix-llama.cpp (B) `c7af5c6c2` untouched (~1% stable reference). No pushes anywhere.

## What is DONE (this session — the WS3 #2 artifact, root-cause → fix → flip)

1. Root cause (prior session) confirmed + FIXED at the ggml level (Option 2): the sched
   alloc-fallback did an unconditional full-device sync when the graph layout differed from
   gallocr's single stored layout. llama-bench pipelines async decodes (no sync between
   2048-token chunks), so each fallback drained the whole ~3 s GPU queue; the dense ubatch
   (7273 nodes, 16 CPU-assigned nodes) vs sparse ubatches (7773, 20 CPU nodes) alternation
   flipped the layout EVERY ubatch of EVERY rep → 8 × ~3 s syncs over 2 passes at pp8192.
2. Fix: buffers are grow-only, so a reserve that FITS only re-points tensors — safe without a
   sync (the graph's compute is ordered after the previous graph's on the backend streams; the
   layout-reuse path already does this every decode). Sync ONLY on real growth (free+realloc
   moves addresses an in-flight graph may still use). New `ggml_gallocr_reserve_n_probe()`
   computes+stores the layout without touching buffers and reports growth; `no_alloc` reserve
   no longer frees buffers. Also removed `backend_ids_changed` from the sync condition (its
   positional node-backend comparison fires on every dense/sparse flip due to the ~16-20
   CPU-assigned mask-view/indexer-chain nodes, but backend-id changes without growth are also
   just re-pointing). NOTE: the brief's original design-B premise ("same-size tensors hash to
   stable addresses across reserves") is FALSE in this code (the per-graph hash is reset every
   reserve; addresses come from the free-list by allocation sequence) — the correct argument is
   the ordering one above. Instrumented: ZERO fallback syncs in the steady state (was 8).
3. Gate evidence (same-session r3, ON vs =0): depth-0 ladder ON >= OFF at every size — pp16384
   574.1 vs 566.8 (+1.3%), pp8192 544.0 vs 535.0 (+1.7%), pp4096 487.9 vs 473.0 (+3.2%),
   pp2048 399.7 vs 382.7 (+4.4%), pp1024 418.6 vs 407.6 (+2.7%), pp512 407.7 vs 401.3 (+1.6%),
   tg128@0 25.25 vs 24.23 (+4.2%); artifact rows pp4096/8192/16384 were -17/-29/-36% pre-fix.
   Depth flat through 32k (pp2048@d12288 329.3 vs 330.4; tg 22.09 vs 22.12; @d32768 334.0 vs
   333.1; tg 20.11 vs 20.13).
4. Coherence/determinism: OFF-path 7-tok text == stored known-good (cli-restore.txt); ON 7-tok
   == LLAMA_QSA_SPARSE_FA=0 dense reference; ON p5000 (multi-ubatch, exercises the new no-sync
   re-pointing) run twice byte-identical; ON-vs-OFF divergence below the width = the documented
   pre-existing dense-vs-sparse kernel signature (matches the pre-fix p5000 on/off pattern).
5. DECISION (maintainer rule "default to fastest and coherent"): `LLAMA_QSA_DENSE_SHORTCUT`
   DEFAULT ON. =0 restores the selection path.

## NEXT SESSION (in order — all lower priority than done items; nothing urgent queued)

1. (MAINTAINER DECISIONS, 2026-09-10 — recorded so they do not resurface): patches 6-7 stay as a
   staged COLLECTION in beta/qwen4exp/ (likely to be SQUASHED together with earlier patches
   later; no per-patch routing ceremony needed now). The ggml fix (patch 6) is NOT an upstream
   candidate at this moment (maybe another day). gfx1201 / multi-GPU validation is DEFERRED
   until ALL Strix Halo work is done (less churn) — do not schedule it per-change.
2. ACTIVE CAMPAIGN (maintainer direction 2026-09-10): equal/surpass the community repo (B) for
   PREFILL and TG at EVERY data point (depths 0/12k/32k x pp512..16384 + tg128). First step:
   re-derive the A-vs-B gap matrix on the NEW default (A shortcut ON = dense-below-width like B,
   so depth-0 rows now compare regime-matched); then attack the largest remaining component per
   the WS1 attribution (A's per-ubatch elementwise/norm/cpy + MoE routing/reduction tail vs B's
   fused hc_* + weighted-expert-sum/concat graph ops; decode TG@0 vs B ~26 t/s).
3. (If requested) the MoE routing/reduction tail: B's ggml_cuda_op_weighted_expert_sum +
   ggml_cuda_mul_mat_id_weighted_rdna3_5 (IQ4_NL down-proj fused with the n_used=10 weighted
   sum) graph fusions into A's ggml-cuda.cu — the biggest remaining pp512-4096 gap component
   after WS3 #3. Verify the A graph pattern matches B's before porting.
4. WS6 re-base NOT indicated.

## Carried-forward open items (full ledger)

- beta/qwen4exp README "Open items" (gfx1201 3xR9700 box / delivery flow): "The answer"
  3-token K=2 multi-seq decode drift at step 3; mixed K/V cache types crash; FA-off +
  tensor-split unsupported; decode levers (batch decode M>1, decode-expert mmvq sweep, GDN
  state fold ~1.5-3%, body-op elementwise fusion); prefill thread (pp8192 ~2024); ML-Kernel/
  gpudh review items (3x R9700 env: GGML_CUDA_FA_WMMA_256=0, sparse FA default).
- Strix (this box): tg@0 decode gap (A 24.2-25.25 now vs B 26.0 — some closed by the
  shortcut tg win) + MTP = the later generation phase
  (benchmarks/mtp-adaptive-methodology.md is the standing decode/fusion gate when re-enabled).
- Delivery/upstream monitors: ROCm unaligned-width split-load (Q6_K/Q3_K 2-GPU; local
  block-13 fix; re-check at each re-base); MXFP4/NVFP4 fused gate+up+GLU MMQ = LAST block-13
  item (add switch cases + instance files + generators, bit-exact + bench per the 0004 recipe).
- Parallel (community member): dual-7900XTX (RDNA3) block-12 hybrid all-reduce validation on
  their box (block-12 gate stays RDNA4-only until verified; GGML_CUDA_ALLREDUCE=internal).
- Parked: LFRU host->GPU slow hot-weight migration (wip/qwen4exp/LRU_EXPERTS.md etc.).

## Hygiene + envs

No parallel benches; warm page cache (dd the 3 shards); long pp first (warm clock); verify
non-empty llama output in loops (spaced runs); depths 0/12k/32k only; ~116 GB VRAM / ~419 GB
disk; one server (port 8033); records benchmarks/YYYY-MM-DD-strix-halo-*.md; scratch in
wip/strix-halo/ + /tmp/gateA/. Env toggles: GGML_CUDA_DISABLE_HC_FUSION (WS4 off),
LLAMA_QSA_DENSE_SHORTCUT (=0 = pre-flip selection path / known-good; unset/=1 = dense shortcut
DEFAULT), GGML_CUDA_DISABLE_MMQ_ROUTED (WS3 #3 compact off; J stays), LLAMA_QSA_SPARSE_FA=0
(dense masked FA). Fix code locations: ggml_gallocr_reserve_n_probe (ggml-alloc.c ~970),
no_alloc-preserves-buffers in reserve_n_impl (~930), sched fallback (ggml-backend.cpp ~1644).
New raw evidence: /tmp/gateA/fix-{on,off,matrix,depth,d32768,default-spot}*.log + fix-coh-*.txt.
