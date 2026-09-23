# Sparse MTP-draft attention (QSA) for qwen4exp — implemented 2026-09-22, **opt-in**

**Status:** implemented and measured; **default OFF** (opt-in `LLAMA_MTP_SPARSE=1`) because the
text-purity gate fails at the depth where the arm pays.  Fork `~/llama.cpp` branch `gap-closing-r13`
(r13 + `beta/mmb-general` + gap-closing `0001..0014`/`0016`/`0017`/`0018`/`0019`), commit
**`1bb1d794e`** (the draft) + **`94a1aa38e`** (the derived-indexer default fix, below).
Patches: [`patches/0020-mtp-sparse-draft.patch`](patches/0020-mtp-sparse-draft.patch) and
[`patches/0021-qsa-indexer-cache-default-on.patch`](patches/0021-qsa-indexer-cache-default-on.patch) -
the latter is a **delivery-level** default fix (all qwen4exp QSA decode above the crossover, not just
the draft).  This is the implementation session for
[`PLAN-mtp-sparse-draft.md`](PLAN-mtp-sparse-draft.md).

## Goal

The qwen4exp MTP draft (`src/models/qwen4exp.cpp::graph_mtp`) attended **dense** over the whole KV
cache while the trunk attends sparse QSA.  At ~150K the single dense draft layer was measured to cost
**2.1× all twelve sparse trunk layers** during prefill (14.3 s vs 6.9 s) and **5.6×** during decode.
Route the draft through the same QSA machinery so the draft's attention cost stops growing with depth.

## What was implemented (three coupled edits + two memory fixes)

1. **`load_arch_hparams` inherits the trunk compress ratio for the nextn layer.**  The converted GGUF
   leaves `dsv4_compress_ratios[nextn] = 0`, but the sidecar ships `blk.N.indexer.*`.  With the draft
   on, the trunk's last non-zero ratio is copied into the nextn layer (logged: `nextn layer 48
   compress ratio 0 -> 4` — the sidecar's nextn layer is **48**, not 47 as the plan guessed).
2. **`llama_model::create_memory` gives the qwen4exp MTP context a `llama_memory_hybrid_idx`**
   (`filter_attn = filter_idx = il >= n_layer()`, `filter_recr = false`) instead of a plain
   `llama_kv_cache`.  Gated on the same switch as the graph, so the memory type and the graph routing
   cannot disagree.
3. **`graph_mtp` takes its attention input from `build_inp_mem_hybrid()`** and, when the cache is
   deeper than the selection budget, builds `top_k` via `build_qsa_top_k` and attends through
   `build_attn_qsa`; otherwise it stores the indexer keys (`build_qsa_store_k`) and attends dense.
   The mask is passed as `qwen4exp_want_derived_vis(...) ? nullptr : inp_attn->get_kq_mask()` so the
   draft keeps the trunk's `-n_ubatch*n_ctx*2` mask-elision win on the prefill arm.
4. **`llama_memory_hybrid_idx` skips the empty recurrent child** (`hybrid_idx_no_recr`) in
   `prepare`/`seq_rm`/`state_write`/`state_read`/`state_drop`.  The MTP context's recurrent filter
   matches nothing, and `llama_memory_recurrent::seq_rm`'s partial-rollback path refuses
   (`return false`) for such a cache, which **aborted the server** on the first speculative cache trim
   (`common_context_seq_rm: failed to remove sequence 0 with p0=5261, p1=-1`).
5. **`llama_memory_recurrent::find_slot` stays quiet when no layer bound a state tensor** — the empty
   recurrent cache still needs its `n` bookkeeping (the graph input sizes read it), but the
   speculative verify positions spammed the non-consecutive-position warning (118 lines per 40K run).

## Gates (gfx1151, qwen4exp IQ4_NL + `Q4_K_XL/…Q4_K_M` sidecar, f16 KV)

| gate | result |
|---|---|
| **Context creation / memory** | `create_memory: MTP context uses a hybrid-idx memory (sparse draft attention)` + the indexer cache (`creating indexer KV cache`) with `LLAMA_MTP_SPARSE=1`; clean init, no abort.  With the feature off the line is absent and the draft keeps the plain KV cache. |
| **`plain == draft-mtp` greedy text** (`prompts/prose-rdna-boosts.txt`, seed 42, temp 0, `-c 8192`, `-n 200`) | **PASS**: plain == default (off) == `LLAMA_MTP_SPARSE=1` = **`3553e76d3a9e`** (578 chars).  Deterministic over 3 sparse repeats. |
| **Width probe** `test-logits-width-probe <IQ4_NL> prompts/prose-rdna-boosts.txt 1024 512` | **`width_purity=PASS (worst maxdiff 0)`**. |
| **Op oracles** | `FLASH_ATTN_QSA` **26/26** (2/2 backends), `GATED_DELTA_NET` **46/46**, 0 FAIL.  No kernel change — regression only. |
| **MTP acceptance** (`-c 12288 -n 3000`, prose, `--reasoning off`) | default (off) **0.85035** (1199/1410); `LLAMA_MTP_SPARSE=1` **0.85035** (1199/1410).  No regression at the gate depth (the sparse arm does not fire below 32K). |
| **Coherence** | coherent continuation on every arm. |

## Measured performance (`-b/-ub 2048`, `--ctx-checkpoints 0`)

Prefill, 150K-token wikitext prompt, `-n 1` (isolates the prompt):

| arm | pp150K |
|---|---:|
| dense draft (hybrid memory, no sparse arm) | 927.0 t/s |
| sparse from n_kv > 2051 (`LLAMA_MTP_SPARSE_MIN_KV=0`) | 697.3 t/s |
| sparse from n_kv > 32768 (default gate) | **990.8 t/s (+6.9 %)** |

Decode/verify (`--spec-draft-n-max 3`, whole-run `Generation:`).  The first block is the decode arm
A/B with the prefill arm held fixed (`LLAMA_MTP_SPARSE=1` both, `--ctx-checkpoints 0`); the second is
the earlier `LLAMA_MTP_SPARSE` on/off comparison (which also changes the memory type, hence the same
direction but a different magnitude):

| decode arm (`LLAMA_MTP_SPARSE_DECODE`) | 150K |
|---|---:|
| off (default) | 19.7 t/s |
| on | 16.0 t/s (−19 %) |

| depth | `LLAMA_MTP_SPARSE=1` (sparse prefill + sparse decode) | `=0` (plain KV + dense draft) |
|---:|---:|---:|
| 16K | 42.9 t/s | 49.3 t/s (−13 %) |
| 40K | 30.6 t/s | 36.1 t/s (−15 %) |
| 150K | 17.0 t/s | 20.2 t/s (−16 %) |

**Reading.**  The depth gate is essential: the indexer score + top-k is a **fixed per-query** cost,
while the dense attention it replaces is `O(n_q·n_kv)`, so at shallow ubatches (n_kv 2K–32K) sparse
is a *loss* (697 vs 927 at 150K if enabled from 2K), and only above ~32K does it pay.  The decode-arm
numbers below were taken with the derived indexer cache still OFF (the old default) and at short
generation, so they overstate the loss; with the cache ON (`patches/0021`) the sparse decode is at
**parity / a slight win** - see the root-cause section.

## Decode root cause (2026-09-22, follow-up): the incremental indexer was OFF by default

The "decode is slower" reading above is a **measurement + configuration artifact**, not the
selected-cell attention.  Kernel profile at 40K (`rocprofv3 --kernel-trace`, sparse decode arm):

| kernel | calls | total | per call |
|---|---:|---:|---:|
| draft sparse attention `flash_attn_qsa` | 76 | 3.76 ms | **0.05 ms** |
| draft dense attention `flash_attn_tile` (dense arm) | 38 extra | ~39 ms | **~1.03 ms** |
| draft fused indexer score `indexer_score_kernel` | 28 | 11.52 ms | 0.41 ms |
| draft top-k (`indexer_topk_*` extras) | ~200 | ~21 ms | — |

The sparse attention is **20× cheaper** than the dense one it replaces; the per-step indexer
(score + top-k) is what eats the saving, so at 40K the two arms are parity (clean interleaved A/B:
dense 30.5 t/s, sparse 30.1 t/s; the earlier −13/−15/−16 % table mixed the plain-KV vs hybrid-idx
memory type and used very short, noisy generations).

**Why the reference's sparse decode is faster: its incremental indexer is always on, ours was not.**
The graph side (`build_qsa_top_k`) has always defaulted its `idx_cache` to ON, but the memory layer set
`derived_enabled = env != nullptr && atoi(env)` - **default OFF** - so the derived block-vector pool was
never created and the fused decode score re-pooled/rotated/scored the whole raw indexer cache every
step.  (Both `AGENTS.md` and the Phase-2 audit describe the cache as default ON, so the code and the
docs disagreed.)  `patches/0021` flips the default ON; `GGML_CUDA_QSA_INDEXER_CACHE=0` restores the old
path.  Plain (non-MTP) decode, byte-identical text in every pair:

| depth | KV | pool OFF | pool ON |
|---:|---|---:|---:|
| 80K | f16 | 24.3 t/s | **26.5 (+9.1 %)** |
| 150K | f16 | 20.6 t/s | **23.6 (+14.6 %)** |
| 80K | bf16 | 24.2 t/s | **26.3 (+8.7 %)** |

The pool only engages for F32/BF16/F16 indexer keys; a quantized indexer cache (`-ctk q8_0`) keeps
the recompute path.  With the pool on, the sparse MTP draft decode is **parity / a slight win** at 80K
(dense 30.7/30.9, sparse 30.7/31.5 t/s; acceptance 0.792 vs 0.812) - i.e. with the reference's
incremental indexer the sparse draft is no longer the losing arm.

## The blocker — depth text purity

At **40K** (wikitext prompt, `--ctx-checkpoints 0`, `-n 200`), the arms are deterministic but the
sparse prefill changes the target's greedy text while the dense draft does not:

| arm | hash (40K) |
|---|---|
| plain | `687cec808661` |
| dense draft (`LLAMA_MTP_SPARSE=0`) | `687cec808661` |
| hybrid memory, sparse arm forced off (`LLAMA_MTP_SPARSE_MIN_KV=999999999`) | `687cec808661` |
| sparse draft | `c0a3bda5dff4` (differs from byte 613: "on **his** potter's wheel" → "on **a** potter's wheel") |

So the hybrid-idx memory and the indexer store are innocent; the **sparse attention itself** exposes a
target-side verify/rollback near-tie.  The target is *logit-width-pure* at the same depth
(`test-logits-width-probe` P=32768, `RS=0` and `RS=from_w`: `width_purity=PASS`, worst maxdiff 0, row-0
hash identical across W=1..8), so the divergence is in the iterative verify/rollback path, not a
single-batch width dependence.  Two more controls: the **dense draft at `--spec-draft-n-max 1`** also
diverges from plain at 40K (`b326d1fe8616`), i.e. a different draft-acceptance pattern trips the same
target issue; and at **150K plain, dense and sparse all differ** (even dense), so MTP-at-depth output
equivalence is already not reliable on this build.

Without `--ctx-checkpoints 0` the 40K MTP runs are *nondeterministic* (three dense runs gave three
hashes); the checkpoint save/restore is a separate confound to investigate.  **Use
`--ctx-checkpoints 0` for any depth purity test.**

## Why default OFF

The plan's gate is `plain == draft-mtp` byte-identical; it holds at the gate depth (5K) but not where
the prefill arm pays (40K/150K).  Per the plan ("if the text differs, that is a bug, not a
re-baseline") and the `AGENTS.md` default-on policy ("a kill-switch for a correctness risk may stay
default-off"), the feature is **opt-in**:

* `LLAMA_MTP_SPARSE=1` — enables the sparse draft (hybrid-idx memory + nextn ratio patch + QSA routing).
* `LLAMA_MTP_SPARSE_MIN_KV=<tokens>` — prefill depth gate, default **32768**.
* `LLAMA_MTP_SPARSE_DECODE=1` — opts the decode/verify band back in (default off), with
  `LLAMA_MTP_SPARSE_DECODE_UNTIL=<tokens>` as its depth gate (0 = whole band).

## Step-5 verification (derived block-vector cache / pool)

The MTP context's indexer cache is created (`creating indexer KV cache`), and its derived pool is
subject to the same `GGML_CUDA_QSA_INDEXER_CACHE` default as the trunk: on this campaign build the
env is unset, so `pool_create` logs `derived indexer cache pool skipped (f16 keys, derived cache
disabled)` and `build_qsa_top_k` takes its no-pool fused-score path.  That path is the trunk's default
too and is byte-identical to the pool path, so nothing misbehaves.  With
`GGML_CUDA_QSA_INDEXER_CACHE=1` the MTP layer (ratio 4) gets its own pool alongside the trunk's, and
the same check applies.

## Traps hit (beyond the plan)

* The plan's `llama_memory_hybrid_idx` call omitted **`cparams.n_rs_batch`** (our constructor takes 19
  args, the reference's snippet had 18) — the first build failed to compile.
* The empty recurrent child **aborts the server** on a partial `seq_rm`; the reference's
  `hybrid_idx_no_recr` helper is the fix and was not in the plan's change list.
* The empty recurrent child also **spams the non-consecutive-position warning** on every speculative
  verify batch (118 lines / 40K run), polluting the CLI output and the text extractor; suppressed in
  `find_slot`.
* The sidecar's nextn layer is **48** (`n_layer_all = 49`), not 47; the trunk-side GGUF is the 48-layer
  one.  The Step-1 fallback is still required (the sidecar's own `compress_ratios[48]` read 0).

## Files

`src/models/qwen4exp.cpp` (the switch, `qwen4exp_qsa_off`, `qwen4exp_mtp_sparse_arm`, the nextn ratio
fallback, the `graph_mtp` hybrid input + QSA routing), `src/llama-model.cpp` (`create_memory` MTP
hybrid-idx branch), `src/llama-memory-hybrid-idx.cpp` (`hybrid_idx_no_recr` + guards),
`src/llama-memory-recurrent.cpp` (`find_slot` quiet mode).
