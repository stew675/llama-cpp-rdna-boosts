# Strix Halo session brief — 2026-09-08 end state (handoff for the next session)

CONTINUE from this file + the plan (~/make-strix-halo-faster.md) + the evidence
trail (wip/strix-halo/notes-ws1-survey.md) + the dated records (below). Read all
first.

## Commits / tree state

- ~/llama.cpp (qwen4exp) tip `a1121cf2d`, parent `151798ed2`, clean:
  - `248e47704` (ROUTED as beta patch 4): WS4 prefill hyperconn fusions DEFAULT
    ON (bit-exact) + deterministic indexer-topk + kv stale-cell zeroing.
  - `151798ed2` (EXPERIMENTAL, NOT routed): WS3 #2 QSA dense-shortcut, OPT-IN
    default OFF (env LLAMA_QSA_DENSE_SHORTCUT=1; see OPEN 1).
  - `a1121cf2d` (NEW, ROUTED as beta patch 5): WS3 #3 routed-compact MoE MMQ
    (see DONE 1). DEFAULT ON; env GGML_CUDA_DISABLE_MMQ_ROUTED=1 disables only
    the compact dispatch (per-expert J selection stays).
  - Build ./build-rocm current; binaries bake a1121cf2d-content == committed.
- ~/llama-cpp-rdna-boosts (delivery) tip `8edaa08`, clean. beta/qwen4exp = FIVE
  patches now (managed-ngrams, qwen4exp-support, mtp-draft-support,
  ws4-hc-prefill-fusions, ws3-routed-moe-mmq). README updated. TODO.md updated.
- ~/strix-llama.cpp (B) `c7af5c6c2` untouched. B is a ~1% stable bench reference.

## What is DONE (2026-09-08 session)

1. **WS3 #3 (routed-compact MoE MMQ for the i-quants) DONE, DEFAULT ON**
   (record: benchmarks/2026-09-08-strix-halo-gfx1151-ws3-routed-moe-mmq.md):
   - port of B's mul_mat_q_routed_compact (one descriptor per real (expert,
     J-tile) pair; build_mmq_routed_descriptors single block/thread per expert)
     + per-expert J selection (mmq_rdna3_5_id_get_J 16/48/64/128 by
     rows-per-expert, gfx1151-measured) into A's mmq path. A/B mmq-config
     tables were already identical — the whole i-quant expert gap was the
     dispatch structure.
   - GATE DECISION (open question resolved): RDNA3_5-only, exactly as B gates
     it. gfx1201/RDNA4 stays OFF (the kernel + J tables are RDNA3.5-tuned;
     enable only after the gfx1201 box in the delivery flow validates).
   - numerics: bit-exact by construction (same mul_mat_q_process_tile); llama-cli
     text identical {default, GGML_CUDA_DISABLE_MMQ_ROUTED=1, pre-change
     known-good} at 7-tok AND 4572-tok pp + 40 decode. rocprof pp2048: compact
     fires on the IQ expert GEMMs (IQ3_S J64 x184, IQ4_NL J64 x86, IQ4_XS J64,
     Q8_0 J48; 564 compact + 274 builder dispatches) — NOTE the model's expert
     down-proj mix is IQ3_S/IQ4_NL-heavy, IQ4_XS only ~4 layers; Q8_0 J128
     plain mul_mat_q calls are non-MoE/other GEMMs, unchanged.
   - same-session depth-0 r3: compact vs plain-at-same-J +2.4-5.3% on every row
     (pp512 401.2 vs 381.0; pp1024 407.8 vs 398.3; pp2048 382.2 vs 364.4;
     pp4096 473.7 vs 462.8; pp8192 535.0 vs 521.1; pp16384 567.6 vs 548.9);
     tg128@0 flat; pp2048@d12288 +1.8% (no depth regression; decode = mmvq
     untouched).
   - A-vs-B gap (same-session; B stable): pp512 2.05->1.61x, pp1024 2.04->1.78x,
     pp2048 2.21->2.01x, pp4096 1.68->1.53x, pp8192 1.37->1.26x, pp16384
     1.14->1.06x.
   - CAVEAT recorded: this box's absolutes drift +5-20% between sessions
     (clock/boost) — every A/B conclusion must be same-session; the 2026-09-06
     pre-change column in the record is context only.

## OPEN items

1. **WS3 #2 llama-bench-only multi-ubatch artifact** (unchanged, still open):
   dense-shortcut default OFF. pp4096+@0 llama-bench rows slower with the
   shortcut on (host-side; identical-or-smaller rocprof kernels; llama-cli/
   server crossing prefill neutral 259.9 vs 258.8). Root cause NOT found —
   needs a llama.cpp graph-lifecycle deep-dive (llama_memory_clear /
   llm_graph_input can_reuse / graph-arena reallocation around the store-only
   qsa_k_inp input mixing with scoring ubatches). Options: fix + flip default
   ON (+ rerun WS4-style gates), or adjudicate, or keep opt-in. Maintainer
   decision needed with the evidence in the post-fusion-gap record's WS3 #2
   section.
2. **Remaining prefill gap** (from the 2026-09-08 record): pp512-4096 still
   ~1.5-2x B, now dominated by (a) the per-ubatch elementwise/routing/
   reduction tail (B's weighted-expert-sum/concat graph fusions
   ggml_cuda_op_weighted_expert_sum + ggml_cuda_mul_mat_id_weighted_rdna3_5
   for the IQ4_NL down proj + concat_transposed; A's tail unfused outside WS4
   hyperconn coverage) and (b) shallow QSA-vs-dense (WS3 #2 opt-in) and (c)
   fused-Q8_0-up/gate residue. The routing/reduction tail workstream is NOT
   yet requested by the maintainer.
3. tg@0 decode gap (A 24.2 vs B 26.0) = later generation phase (not prefill).

## NEXT SESSION (in order, unless the maintainer re-prioritizes)

1. WS3 #2 artifact: root-cause or adjudicate (OPEN 1) — highest-value open
   technical item; needs llama.cpp host-side graph-lifecycle work, not GPU
   kernels.
2. (If requested) the MoE routing/reduction tail: port B's
   ggml_cuda_op_weighted_expert_sum (+ the ggml_cuda_mul_mat_id_weighted_
   rdna3_5 IQ4_NL down-proj fusion) as graph fusions in A's ggml-cuda.cu —
   verify fused-vs-unfused bit-exactness + pp A/B at depth 0; watch the
   qwen4exp build_moe_ffn graph pattern (A's shape may differ from B's;
   n_used=10 here).
3. Re-derive the A-vs-B gap after anything lands; update the records. Any
   shipped change -> beta/qwen4exp patch (currently five; ws3 #3 already
   packaged) or amendment per AGENTS; NEVER push from ~/llama.cpp. gfx1201
   validation via the delivery flow when a gfx1201 box is available
   (required before the ws3-routed patch claims RDNA4). WS6 re-base NOT
   indicated.
4. Env toggles recap: GGML_CUDA_DISABLE_HC_FUSION (WS4), LLAMA_QSA_DENSE_SHORTCUT
   (=1 on/=0|unset off, WS3 #2), GGML_CUDA_DISABLE_MMQ_ROUTED (WS3 #3 compact
   off; J selection stays), LLAMA_QSA_SPARSE_FA=0 (dense masked FA).

## Hygiene (AGENTS.md + plan §9)

No parallel/background benches. Back-to-back llama_decode/llama-cli runs of
this 87 GiB model intermittently produce EMPTY output in loops — standalone or
spaced, verify non-empty. Warm page cache (dd the 3 shards). First test after
process start = long pp (cold GPU clock). Depths 0/12k/32k only. ~116 GB
effective VRAM / ~419 GB disk. One server (port 8033). Scratch/raw: /tmp/gateA/
(rc-d0-on/off.md, rc-B-d0.md, rc-d12k-{on,off}.md = this session's raw ladders;
rc-prof/ = rocprof DB kept small, delete freely).
