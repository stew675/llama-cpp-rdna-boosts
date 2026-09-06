# Strix Halo session brief — 2026-09-09 end state (handoff for the next session)

CONTINUE from this file + ~/make-strix-halo-faster.md + wip/strix-halo/notes-ws1-survey.md +
the dated records listed below. Read all first. The PRIMARY next-session task is the WS3 #2
artifact FIX (maintainer chose Option 2 — the ggml-level fix, see "NEXT SESSION (in order)").
This brief also carries the full ledger of unfinished work so nothing is lost across compaction.

## Commits / tree state (all clean, verified)

- ~/llama.cpp (qwen4exp) tip `a1121cf2d`, clean, build current (binaries == committed; default
  llama-cli text re-verified == known-good after the instrumentation revert):
  - `248e47704` (ROUTED, beta patch 4): WS4 prefill hyperconn fusions DEFAULT ON (bit-exact),
    deterministic indexer-topk, kv stale-cell zeroing.
  - `151798ed2` (EXPERIMENTAL, NOT routed): WS3 #2 QSA dense-shortcut, OPT-IN default OFF
    (LLAMA_QSA_DENSE_SHORTCUT=1; =0/unset = known-good). The llama-bench artifact this enabled is
    now ROOT-CAUSED (see DONE 2) but UNFIXED — fixing it is the next session's task.
  - `a1121cf2d` (ROUTED, beta patch 5): WS3 #3 routed-compact MoE MMQ (RDNA3.5 gate, DEFAULT ON,
    GGML_CUDA_DISABLE_MMQ_ROUTED=1 opt-out). gfx1201 enablement still pending (delivery flow).
  - NOTE: instrumentation added for the root-cause hunt (LLAMA_UBATCH_TIMING / GGML_SCHED_TIMING
    prints in llama_context::decode, llama_context::process_ubatch, ggml_backend_sched_alloc_splits)
    was REVERTED; the tree is byte-clean at a1121cf2d. Re-add the same prints to re-measure.
- ~/llama-cpp-rdna-boosts (delivery) tip `e96da5c`, clean. beta/qwen4exp = FIVE patches
  (managed-ngrams, qwen4exp-support, mtp-draft-support, ws4-hc-prefill-fusions,
  ws3-routed-moe-mmq). TODO.md reality pass carried to 2026-09-09.
- ~/strix-llama.cpp (B, halo-box) `c7af5c6c2`, untouched. B is a ~1% stable bench reference.

## Records (this campaign, most recent first)

- `benchmarks/2026-09-08-strix-halo-gfx1151-ws3-shortcut-artifact-rootcause.md` — ROOT CAUSE of
  the WS3 #2 llama-bench artifact + same-session tradeoff numbers (the fix task's evidence base).
- `benchmarks/2026-09-08-strix-halo-gfx1151-ws3-routed-moe-mmq.md` — WS3 #3 gates + A-vs-B gap.
- `benchmarks/2026-09-06-strix-halo-gfx1151-post-fusion-gap.md` — post-WS4 gap + WS3 #2 section.
- `benchmarks/2026-09-06-strix-halo-gfx1151-ws4-hc-fusion-gates.md` — WS4 gates.
- `benchmarks/2026-09-05-strix-halo-gfx1151-block-13-moe-mmq.md` (+ the gfx1100 twin) — block 13.

## What is DONE

1. **WS3 #3 (routed-compact MoE MMQ) — DONE, DEFAULT ON, routed as patch 5** (09-08 record):
   A-vs-B gap moved pp16384 1.14->1.06x, pp8192 1.37->1.26x, pp4096 1.68->1.53x, pp2048
   2.21->2.01x, pp1024 2.04->1.78x, pp512 2.05->1.61x; bit-exact by construction + text-verified;
   depth gates clean; RDNA3.5-only gate (B parity) — RDNA4 stays OFF pending the gfx1201 box.
2. **WS3 #2 llama-bench artifact — ROOT CAUSE FOUND (this session), no code shipped** (root-cause
   record): see the mechanism summary below; the fix is the next session's PRIMARY task.

## Root-cause summary (the fix task's starting point)

- llama-bench test_prompt llama_decode's 2048-token chunks with NO llama_synchronize between
  (deep async GPU queue; llama_context::graph_compute is ASYNC). Real serving (common.cpp)
  syncs after every decode -> immune (validated: p5000 neutral, 259.9 vs 258.8).
- ggml_gallocr (ggml/src/ggml-alloc.c) keeps ONE stored layout (positional node_allocs[] sized by
  n_nodes + per-size hash of (buffer_id, addr, size_max)). When a graph's node count or per-node
  sizes differ from the stored layout (ggml_gallocr_needs_realloc, ~ggml-alloc.c:990), the sched's
  alloc path falls back: ggml_backend_sched_alloc_splits (~ggml/src/ggml-backend.cpp:1608) does an
  UNCONDITIONAL full ggml_backend_synchronize on every backend (drains the whole queued backlog)
  then ggml_gallocr_reserve_n + alloc_graph.
- With the shortcut ON, the dense ubatch (7273 nodes) alternates with sparse ubatches (7773
  nodes, sizes growing with n_kv) -> a fallback sync EVERY ubatch of EVERY rep. Measured
  (instrumented): pp8192 r1 -> 8 alloc-fallback syncs of 2.8-3.5 s over 2 passes (ON) vs 4 in
  pass 1 only (OFF). Result: ON never reaches the stable warm-layout state OFF reaches by rep 2
  -> every rep stays at pass-1 speed (-34% @pp8192; -17/-29/-36% @pp4096/8192/16384 in the
  09-06 record). OFF is only immune because its reps repeat pass-1's max layout exactly.
- Also explains: pass-1 of every config is slower than later reps (layout warm-up), for any model.

### Why the structure-stable shortcut variant is NOT the answer (measured, same-session)

Dense-masked kernel below the width WITH the top-k chain kept (= SPARSE_FA=0 structure) removes
the layout flip but forfeits the tg@0 win entirely (SF0 tg == OFF) and ~1/4 of the pp2048 win:
pp2048@0 r2: OFF 325.2, SF0 338.8 (+4.2%), true shortcut 343.6 (+5.7%); tg128@0: SF0 == OFF
23.55/23.54, true shortcut 24.53 (+4.2%). => The fix must keep the true shortcut's topology
change but stop the gallocr/sched serialization.

### Fix design space for the next session (weigh both; see the root-cause record for the detail)

- **A. Multi-layout gallocr cache** (the maintainer's Option 2): N layout slots (start with 2:
  dense + sparse), each a full reserved layout on its own lazily-allocated backend buffer set,
  tagged by a graph signature; alloc_graph picks the matching slot, else evicts LRU. Consecutive
  same-slot graphs serialize naturally on the GPU stream (in-order kernels) -> no host sync
  needed. Memory cost = N x per-layout buffer size (only the distinct layouts actually used).
  Watch: interplay with HIP-graph capture/replay (GGML_HIP_GRAPHS=1) and the sched's split-input
  tensors being re-pointed between slots.
- **B. Conditionalize the fallback sync** (smaller change, likely sufficient): the fallback's full
  sync exists because re-reserving can move tensor addresses while the previous (queued) graph is
  in flight. But the reserve is address-stable whenever (i) no backend buffer is REALLOCATED
  (buffers are grow-only: reserve_n_impl only reallocs when a chunk grows past its current size;
  shrinking keeps buffers) and (ii) the per-size hash table is not recreated (only recreated when
  n_nodes+n_leafs exceeds min_hash_size, i.e. on GROWTH). Same-size tensors hash to the same
  address across reserves. So: sync only when the new reservation would actually grow a buffer /
  recreate the hash set; otherwise skip it. ggml_gallocr_reserve_n_size (exists) can precompute
  the sizes to decide. MUST be validated empirically first: instrument address stability across
  the dense->sparse alternation in the steady state (no growth) before trusting the no-sync path.
- Either way the validation protocol below applies; B first (cheap), fall back to A if B proves
  unsound.

## NEXT SESSION (in order)

1. **PRIMARY — fix the WS3 #2 llama-bench artifact at the ggml level** (maintainer's Option 2):
   (a) re-add the env-gated instrumentation (LLAMA_UBATCH_TIMING in llama_context::decode +
   process_ubatch; GGML_SCHED_TIMING in ggml_backend_sched_alloc_splits) and reproduce pp8192
   r1/r2 shortcut ON vs OFF (expect ON ~339 vs OFF ~516 t/s + the 8-vs-4 sync pattern);
   (b) implement fix design B (conditional fallback sync) or A (multi-layout cache) per the
   analysis above, after empirically validating the address-stability precondition;
   (c) validate: artifact gone (ON pass-2 no longer serializes; pp8192 ON >= OFF and gains on the
   shallow rows preserved), OFF and unrelated paths unchanged (any model/arch on this box: at
   least a plain GEMM/attn llama-bench sanity + decode-at-depth + multi-ubatch pp rows + the
   mmvq decode path), ggml-level change = arch-agnostic C code but affects every backend's sync
   behavior -> be conservative; (d) re-run the WS3 #2 gate set with the shortcut ON vs OFF:
   depth-0 ladder (expect pp512-2048 +3-5%, tg@0 +4.2%, pp4096+@0 ~= OFF or better), depth
   12k/32k rows unchanged, llama-cli coherence text on == off == the LLAMA_QSA_SPARSE_FA=0 dense
   reference, memory ladder -r3 through 32k;
   (e) DECISION after (d): flip LLAMA_QSA_DENSE_SHORTCUT DEFAULT ON (B parity) or keep opt-in;
   if flipped, rerun the WS4-style gate set and route as beta patch 6 per AGENTS (never push from
   ~/llama.cpp) with a dated record; record everything regardless.
2. (If requested) the MoE routing/reduction tail: B's ggml_cuda_op_weighted_expert_sum +
   ggml_cuda_mul_mat_id_weighted_rdna3_5 (IQ4_NL down-proj fusion) graph fusions into A's
   ggml-cuda.cu (n_used=10 here; verify the A graph pattern matches B's before porting) — the
   biggest remaining pp512-4096 gap component after WS3 #3. NOT yet requested by the maintainer.
3. gfx1201/RDNA4 validation of patch 5 (ws3-routed-moe-mmq) via the delivery flow when a gfx1201
   box is available (required before patch 5 can claim RDNA4; the mmq.cuh compact kernel + J
   tables are RDNA3.5-tuned).
4. Re-derive the A-vs-B gap after anything lands; update records + TODO. WS6 re-base NOT
   indicated (rocBLAS parity; missing pieces are halo fusions, not upstream drift).

## Carried-forward open items (full ledger)

- Strix prefill (this box): WS3 #2 fix (above); tg@0 decode gap (A 24.2 vs B 26.0, and
  shortcut-on 24.53) = the later generation phase, NOT prefill; MTP work deferred to that phase
  (mtp-adaptive-methodology.md is the standing decode/fusion gate when re-enabled).
- beta/qwen4exp carried-forward items (authoritative list: beta/qwen4exp/README.md "Open items
  (carried forward from WIP)"; mostly the gfx1201 3xR9700 box / delivery flow): "The answer"
  3-token K=2 multi-seq decode drift at step 3; mixed K/V cache types (k=bf16/v=f16) crash at
  model init; FA-off + tensor-split unsupported (upstream Meta-backend constraint); decode
  levers not landed (server-level batch decode M>1, decode-expert mmvq config sweep, GDN state
  fold ~1.5-3%, body-op elementwise fusion); prefill thread (pp8192 ~2024, further tuning if
  resumed); ML-Kernel/gpudh review vs TP-V1, v_shifted probe, splitter->F32, shuffle-to-smem,
  ggml-backend-meta import gates, multi-device sync in meta_tp_test (3x R9700 gfx1201 box env:
  GGML_CUDA_FA_WMMA_256=0, sparse FA default).
- Delivery/upstream monitors: ROCm unaligned-width split-load (Q6_K/Q3_K 2-GPU; local block-13
  fix; re-check at each re-base); MXFP4/NVFP4 fused gate+up+GLU MMQ = LAST block-13 item
  (add switch cases + instance files + generators for MXFP4 (+NVFP4/Q4_0-class), then bit-exact +
  bench per the 0004 recipe).
- Parallel (community member): dual-7900XTX (RDNA3) block-12 hybrid all-reduce validation on
  their box (block-12 gate stays RDNA4-only until verified, then the arch check is removed; test
  env GGML_CUDA_ALLREDUCE=internal). Single-GPU 7900XTX here => block 12 N/A on this box.
- Parked: LFRU host->GPU slow hot-weight migration (wip/qwen4exp/LRU_EXPERTS.md,
  PHASE0_ROUTING.md, HANDOVER-2026-09-04-tiering.md) — not active.

## Hygiene (AGENTS.md + plan §9) + envs

No parallel/background benches. Back-to-back llama_decode/llama-cli runs of this 87 GiB model
intermittently produce EMPTY output in loops — standalone or spaced, verify non-empty before
diffing. Warm page cache (dd the 3 shards); first test after process start = long pp (cold GPU
clock); depths 0/12k/32k only; ~116 GB effective VRAM / ~419 GB disk; one server (port 8033);
records in benchmarks/YYYY-MM-DD-strix-halo-*.md; scratch in wip/strix-halo/ + /tmp/gateA/.
Env toggles: GGML_CUDA_DISABLE_HC_FUSION (WS4 off), LLAMA_QSA_DENSE_SHORTCUT (=1 on / =0|unset
off; WS3 #2), GGML_CUDA_DISABLE_MMQ_ROUTED (WS3 #3 compact off; J selection stays),
LLAMA_QSA_SPARSE_FA=0 (dense masked FA; the numerics twin of the shortcut below the width).
Instrumentation envs to re-add for the fix work: LLAMA_UBATCH_TIMING, GGML_SCHED_TIMING.
Raw files: /tmp/gateA/ (art-*.log = artifact timing runs, chk-*.md = pp2048 config A/B,
rc-*.md = WS3 #3 ladders, cli-*.txt = coherence references incl. cli-restore.txt == known-good).
