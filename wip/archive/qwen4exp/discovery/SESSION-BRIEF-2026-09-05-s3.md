# Strix Halo session brief s3 (2026-09-05) — end state / handoff for the next session

CONTINUE from this file + the plan (~/make-strix-halo-faster.md) + the
evidence trail (wip/strix-halo/notes-ws1-survey.md) + the two 2026-09-06
benchmark records (below). Read all of those first.

## Commits / tree state

- ~/llama.cpp (branch qwen4exp, tip `151798ed2`, parent `248e47704`):
  - `248e47704` (DONE, ROUTED): WS4 prefill hyperconn fusions DEFAULT ON
    (bit-exact), deterministic indexer-topk gather, kv-cache stale-cell
    zeroing. Packaged as the 4th beta patch
    `beta/qwen4exp/ws4-hc-prefill-fusions.patch`.
  - `151798ed2` (NEW this session, EXPERIMENTAL, NOT routed): WS3 #2 QSA
    dense-shortcut below the selection width — **OPT-IN, default OFF**
    (see the OPEN item). Working tree clean.
  - Build: ./build-rocm via ~/bin/build-llama-rocm-714 (canonical, no
    -mllvm). Binaries bake the pre-commit HEAD (248e47704) — content
    verified == committed tree.
- ~/llama-cpp-rdna-boosts (delivery), tip `dff3a9a` (this session's
  records + WS3 #2 writeup). beta/qwen4exp = FOUR patches now.
  wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws4-hc-fusion-gates.md and
  ...-post-fusion-gap.md carry this session's numbers. TODO.md updated
  (reality pass 2026-09-06).
- ~/strix-llama.cpp (B, community): `c7af5c6c2`, untouched, build current.
  B has NO sparse-QSA kernel; its dense-shortcut env LLAMA_QSA_DENSE_SHORTCUT
  defaults ON (A's does NOT — see the env note below).

## What is DONE (2026-09-05 session)

1. **WS4 gates PASSED on the final build** (record:
   wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-ws4-hc-fusion-gates.md):
   - depth-0 warm-clock r3, fusion default vs GGML_CUDA_DISABLE_HC_FUSION=1:
     +7.6/+7.4/+7.8/+8.8/+6.9/+5.2% at pp16384/8192/4096/2048/1024/512;
   - depth 12k/32k pp rows keep +4-6%; tg@depth flat (decode untouched);
   - memory stable -r3 through 32k; llama-cli same-seed text on == off.
2. **Post-fusion A-vs-B depth-0 gap re-derived** (record:
   wip/archive/qwen4exp/discovery/2026-09-05-strix-halo-gfx1151-post-fusion-gap.md), A
   (fusion ON) vs B (c7af5c6c2), same-session warm-clock r3:
   pp512 2.05x, 1024 2.04x, 2048 2.21x, 4096 1.68x, 8192 1.37x, 16384 1.14x.
   B reproduces its WS1 numbers (~1%); the fusion moved the long rows
   (WS1: 1.46x@8k -> 1.37x, 1.23x@16k -> 1.14x); shallow rows still ~2x,
   now dominated by the expert i-quant mmq (~20-25%) + the per-ubatch
   routing/reduction tail.
3. **WS3 #2 (QSA dense shortcut) IMPLEMENTED + characterized** (A commit
   `151798ed2`; full numbers + evidence in the post-fusion-gap record):
   - port of B's heuristic into build_layer_attn: while
     n_kv <= indexer_top_k + ratio - 1 (= 2048+4-1 = 2051) attend dense
     and store only this ubatch's raw indexer keys (store-only
     build_qsa_store_k + minimal llm_graph_input_qsa_k holding just
     k_idxs — B's exact split; a unified-input variant CRASHES: the full
     QSA input's unused cell_blk/bias tensors get no buffer in
     store-only graphs -> NULL-buffer assert in set_input_qsa).
   - numerics: below the width the shortcut is TEXT-IDENTICAL to A's
     existing LLAMA_QSA_SPARSE_FA=0 masked-dense path; the dense-vs-sparse
     kernel signature is the pre-existing env-selectable regime (7-token
     probe pp-last top0 15.708 dense vs 15.973 sparse).
   - perf (llama-bench, fusion ON): tg128@0 +4.2% (24.53 vs 23.54),
     pp512/1024/2048@0 +3-5%, pp2048 ub4096 single-chunk +4.5%, depth
     rows flat. llama-cli crossing prefill (p5000 + 40 gen) neutral:
     259.9 vs 258.8 t/s.
   - env semantics (DIFFER from B): LLAMA_QSA_DENSE_SHORTCUT=1 = on;
     =0 or UNSET = off (known-good byte-identical). B defaults ON; A
     defaults OFF pending the artifact + adjudication.

## OPEN items

1. **WS3 #2 llama-bench-only multi-ubatch artifact (root cause NOT
   found — highest-priority investigation):** with the shortcut on,
   llama-bench depth-0 pp4096/8192/16384 rows run SLOWER (367.3 vs 444.3
   @4k, 354.8 vs 499.1 @8k, 334.2 vs 525.0 @16k; also at r1, so not a
   rep-amortization effect; penalty grows with ubatch count). Yet:
   (a) rocprof kernel traces are identical-or-smaller with the shortcut
   (pp2048-on = pure dense FA, no indexer kernels, 7769 calls/6.47 s vs
   8591/6.90 s off); (b) the llama_decode/llama-cli path (incl. the same
   dense-then-sparse ubatch mixing at p5000) shows NO penalty. Suspect: a
   host-side graph-lifecycle cost in this fork's hybrid-memory machinery
   when one context mixes store-only ubatches (n_kv <= 2051) with scoring
   ubatches. Next probes: per-ubatch host timing inside llama_decode
   (not llama-bench), llama_memory_clear / llm_graph_input can_reuse
   lifecycle around the qsa_k_inp store-only input, graph-arena
   reallocation per ubatch. Decide: fix, or flip default ON with the
   artifact documented (llama-bench protocol comparability suffers while
   real serving is neutral-to-better), or keep opt-in. The maintainer
   must adjudicate with the evidence in the record.
2. **WS3 #3 (routed-compact MoE mmq for i-quants) — the main remaining
   shallow-row gap component** (WS1 attribution ~20-25% of pp2048; B's
   mul_mat_q_routed_compact "RDNA3.5 MoE, descriptor-compacted", J=64,
   mmq.cuh ~1389/1718). Verify B's exact arch gating before porting
   (B runs it on gfx1151; the maintainer's handoff text says "RDNA4-gated"
   — resolve which gate the A-side port should carry, then validate on
   gfx1151 + gfx1201 via the normal delivery flow).
3. Decode tg@0 still below B (A 23.5-24.5 vs B 26.0-26.9) even with the
   shortcut (+4.2%); B's other decode edges are NOT part of the prefill
   work — leave for the later generation phase unless asked.

## NEXT SESSION (in order)

1. WS3 #3: port B's routed-compact MoE mmq for IQ4_XS/IQ4_NL (and Q8_0?)
   into A — read B's mmq.cuh routed_compact (arch gate, descriptor
   compaction, J caps) vs A's plain mmq path; implement, validate
   bit-exact/coherence (fused-vs-unfused same-build), then pp A/B at
   depth 0 (expect the shallow-row gap to move). Re-check decode-at-depth
   does not regress.
2. WS3 #2 artifact: root-cause or adjudicate (see OPEN 1). If fixed and
   confirmed on the llama_decode path, flip the default ON and re-run the
   WS4-style gate set (coherence text on == off at the LLAMA_QSA_SPARSE_FA=0
   dense reference, depth gates, memory ladder) before routing.
3. Re-derive the A-vs-B gap after the above (expect ~2.0x -> sub-1.8x at
   pp512-2048 if both land); update the post-fusion-gap record.
4. Route any shipped change per AGENTS (qwen4exp tree -> beta/qwen4exp
   patch amendment; NEVER push from ~/llama.cpp); gfx1201 validation via
   the normal delivery flow; dated bench records; TODO.md update. WS6
   re-base still NOT indicated (rocBLAS parity; missing pieces are
   B-specific fusions, not upstream drift).

## Hygiene (all from AGENTS.md + plan §9)

No parallel/background benches. Back-to-back llama_decode/llama-cli runs
of this 87 GiB model intermittently produce EMPTY output in loops — run
standalone or spaced and verify non-empty before diffing. Warm page cache
(dd the 3 shards) before benching; first test after process start = long
pp (cold GPU clock; a llama-cli decode stint right before a bench depresses
the clock — warm with a short pp8192 run first). Depths 0/12k/32k only
(64k/128k deferred). ~116 GB effective VRAM / ~419 GB disk. One server
(port 8033). llama-bench env toggles that matter: GGML_CUDA_DISABLE_HC_FUSION
(WS4 off), LLAMA_QSA_DENSE_SHORTCUT (WS3 #2; =1 on / =0|unset off),
LLAMA_QSA_SPARSE_FA=0 (dense masked FA + full scoring; the numerics twin
of the shortcut below the width). Scratch/raw files: /tmp/gateA/.
