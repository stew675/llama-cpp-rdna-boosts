# Strix Halo session brief — handoff after 2026-09-11 (compact me; run the NEXT SESSION prompt)

ACTIVE GOAL (maintainer): equal/surpass the community repo (B, halo-box `~/strix-llama.cpp`
c7af5c6c2, untouched ~1% reference) for PREFILL and TG at EVERY data point. CURRENT FOCUS:
close the remaining depth-0/mid prefill gap on the SHARED path (QSA is not the deficit - see
below), keep QSA on, verify the depth lead; decode followup afterwards.

## Tree state (all clean, nothing pushed)

- ~/llama.cpp (qwen4exp branch) tip `7a6a2e97b`. Chain: 248e47704 (WS4) -> 1da01fa67 (WS3#3
  routed mmq) -> a2f2a6ceb (ggml sched-fallback-sync fix) -> 250e48e97 (shortcut DEFAULT ON;
  the opt-in commit was SQUASHED into this on 2026-09-11 - no opt-in commit exists) ->
  b31940a5e (weighted-down decode fusion) -> 8b62ac25a (PLE host-gather prefill fix) ->
  3cb9168be (managed reader batched fetch) -> 2f8864cc8 (LLAMA_QSA_OFF gate knob) ->
  2bd516bab (transposed concat port) -> 7a6a2e97b (fused swiglu-input quantize port). Build
  current. Backup branch `fork-squash-backup-b004e9744` holds the pre-rewrite chain (old
  hashes a1121cf2d/fcfb0a522/1682d32a9/a18e24f97/32680d937/b004e9744 resolve there; old->new
  map in the beta README).
- ~/llama-cpp-rdna-boosts (delivery) tip `f594404`: TWELVE patches in beta/qwen4exp, staged as
  a collection (squash later). Verified: the full 12-patch series applies from-scratch at
  da67bcb88 and reproduces fork tip 7a6a2e97b byte-identically. ~/strix-llama.cpp untouched.
  NO pushes (AGENTS: never from ~/llama.cpp; gfx1201 validation deferred until ALL Strix work
  is done; ggml fix NOT an upstream candidate now).

## Prefill investigation — COMPLETE + root cause FIXED (records 2026-09-10/11)

1. The llama-bench shallow-row inflation was A's PLE n-gram table (28.8 GB IQ4_NL,
   input-layer = CPU-pinned) gathered by a single-threaded CPU get_rows faulting one 4 KB mmap
   page at a time (2.7 s per 2048-token ubatch). FIX `8b62ac25a` (patch 9, DEFAULT ON,
   LLAMA_QSA_PLE_HOSTGATHER=0 restores): host-table gather in set_input with one
   madvise(WILLNEED) page sweep; no CPU split. Depth-0 became: pp16384 A WINS 626 vs 600,
   pp8192 637, pp4096 646, pp2048 655, pp1024 643, pp512 591 (vs B 600/679/734/776/731/645);
   tg128 25.95 vs 26.01 (parity).
2. B ALSO runs the QSA architecture (lightning indexer + sparse FA + the same shortcut env) -
   A's depth/pp16384 lead = A's sparse implementation beats B's at large n_kv (B collapses
   776->600 as pp grows; A stays flat), NOT a missing B feature.
3. QSA attribution (LLAMA_QSA_OFF=1 gate = dense no-indexer; probe): depth-0 pp2048 A-on 590 ==
   A-off 589, B 754 -> the remaining gap is entirely the SHARED path (QSA free below the
   width; indexer store ~free).
4. Managed-ngrams reader (opt-in --lazy-buffer-size N only): added the SAME batched-fetch (A:
   coalesced posix_fadvise WILLNEED; B: parallel pread pool into a per-gather buffer + serial
   arena write-back). 10G test: parity with host-gather at every pp row (+/-0.5%). FOLDED into
   patch 1 (managed-ngrams) - no standalone artifact.

## Shared-path parity work (ws6, commits 2f8864cc8/2bd516bab/7a6a2e97b = patches 10-12)

Same-session depth-0 r3 A-vs-B progression at pp2048 (B 770.8 stable): 654.6 (post-host-gather)
-> 672.4 (concat) -> 675.0 (swiglu). pp4096: 645.8 -> 661.4 -> 667.0 (B 731.6). Both ports
text-identical, rocprof-verified firing (concat_transposed_src1_dim0 ~37/pass ~0.055 s;
quantize_mmq_q8_1_swiglu ~94/pass replacing the GLU gated-silu kernels). Gaps now: pp2048
0.876x, pp4096 0.91x, pp512/1024/8192 ~0.9-0.93x.

NEXT (in order):
1. RE-DERIVE the remaining kernel deltas with a FRESH same-session rocprof pair (A-now 7a6a2e97b
   vs B-now at pp2048) - the earlier per-kernel deltas were measured across separate sessions
   and pre-fix builds and are stale. rocprofv3 --kernel-trace -d /tmp/prof -o NAME -- <app>;
   summarize with /tmp/rocprof-sum.py.
2. Attack whatever is largest. Candidates from the stale profile: the Q8_0 gate/up mul_mat_q
   J128 (+13% on ~1182 calls / ~520 ms/pass, largest single kernel - CONFIRM first, the "feed"
   hypothesis needs a same-session check), A-only mm_ids_helper launches, host per-ubatch
   submit. Then re-measure the depth-0 ladder + depth-12k pp rows same-session A/B.
3. After parity: keep QSA on and verify the depth lead extends (pp16384@0 + 12k/32k rows);
   check for sparse-vs-dense crossover messiness at mid sizes now that the shared path no
   longer masks it.
4. DECODE followup (then): tg@0 gap is A ~25.0-25.3 vs B 25.96 - per-op A-vs-B decode
   kernel-mix profile (A decode = thousands of small mmvq launches/token; host-launch-bound).

## Ledger / open items

- Envs: LLAMA_QSA_DENSE_SHORTCUT (=0 pre-flip sparse path), GGML_CUDA_DISABLE_HC_FUSION,
  GGML_CUDA_DISABLE_MMQ_ROUTED, GGML_CUDA_DISABLE_WEIGHTED_DOWN (patch 8 opt-out),
  LLAMA_QSA_SPARSE_FA (=0 dense masked FA), LLAMA_QSA_PLE_HOSTGATHER (=0 old graph path),
  LLAMA_QSA_OFF (=1 dense no-indexer gate), LLAMA_LAZY_IO_THREADS (managed reader pool width).
- Beta README "Open items" (gfx1201 3xR9700): "The answer" 3-token K=2 multi-seq drift; mixed
  K/V cache crash; FA-off + tensor-split unsupported; decode levers; ML-Kernel/gpudh review.
- Delivery/upstream monitors: ROCm unaligned split-load; MXFP4 fused MoE MMQ; dual-7900XTX
  block-12; LFRU parked. MTP deferred (mtp-adaptive-methodology.md).
- Hygiene note: dated archive records (2026-09-06..09-10) reference pre-rewrite fork hashes -
  resolvable via fork-squash-backup-b004e9744 + the beta README map; do not "fix" them.

Hygiene/protocol: llama-bench -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16
--load-mode none, prompts DESCENDING in one process (long row first = warm clock; a short row
first costs ~10% cold-clock), same-session A/B only (B is the ~1% stable reference; A absolutes
drift between sessions/processes), B@depth at -r 1 only (B aborts under -r3 state-restore);
warm page cache (dd the 3 shards); no parallel benches; verify non-empty llama output in
loops; depths 0/12k/32k; one server port 8033; dated records benchmarks/YYYY-MM-DD-strix-halo-*;
scratch /tmp/gateA/ + /tmp/prof/. Delivery patches: each = git diff(parent, commit), verified
to apply in series from-scratch at da67bcb88 reproducing the fork tip (12/12 last verified).
