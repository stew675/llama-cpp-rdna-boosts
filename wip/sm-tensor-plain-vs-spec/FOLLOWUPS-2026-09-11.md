# Follow-ups — free GDN prefill alignment (Option B) + the qwen35moe batch-width residual

Date: 2026-09-11.  Box: GFX1201, 3x R9700 (gfx1201, RDNA4), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`).  Companion to `HANDOVER-2026-09-11.md` (the
plain-vs-spec root-cause + the mmvq/block-13 and GDN-gate/block-02 fixes).

Status of the two items here: **both open, both deferred by the maintainer to
follow-up sessions.**  Nothing in this document is delivered; the delivery is
at canonical tip `eb26da812` (tree `b64f21644`), 15 blocks, block 02 with
`GGML_CUDA_GDN_ALIGN_BOUNDARY` default-ON and `KTAIL=16`.

---

## Part 1 — Option B: make the GDN prefill alignment free

### Where the cost comes from

`ggml_cuda_op_gated_delta_net_impl` (`ggml/src/ggml-cuda/gated_delta_net.cu`)
has three chunked-GDN entry points:

1. `K == 1 && n_tokens > 1` → `ggml_cuda_op_gated_delta_net_chunked` on the
   **whole** prompt (the plain path).  Fast.
2. `K > 1 && n_seqs == 1 && n_tokens > K + 64` → chunked on
   `n_tokens - K` + **sequential `K`** (the MTP long-prefill path, PR #9).
   The sequential tail exists to write the `K` rollback snapshots.
3. `align_boundary` (default ON) → chunked on `n_tokens - KTAIL` + sequential
   `KTAIL`, for **both** `K == 1` and `K > 1`.  The fixed `KTAIL` is what makes
   `K == 1` and `K > 1` bit-identical.

Because branches 1 and 2 pick **different** prefix boundaries, and the chunked
kernel is not bit-exact with the sequential one, a plain run (`K == 1`) and an
MTP run (`K == n_max+1`) end up with different post-prefill SSM state.  Branch 3
fixes that by making the boundary a constant — and that is why the plain path
pays a sequential tail it has no use for.

`KTAIL` is now 16, which cut the cost to ~0.3–0.8 % prefill (from ~1.5 % at 64).
The remaining tail is **pure overhead for the plain path**.

### The free design

Have branch 2 also chunk the **whole** prompt — i.e. the exact same kernel call
as branch 1 — and skip the sequential tail.  Then `K == 1` and `K > 1` are
bit-identical *by construction* (same call, same input), the plain path is
unchanged (zero cost), and the alignment branch 3 becomes unnecessary.  No
`n_max` bound, no `KTAIL` tuning, no opt-out needed.

The only casualty is the `K` rollback snapshots a long prefill currently writes.

### The invariant that must hold

> **A rollback never crosses a batch boundary.**  Precisely: for a batch of
> `n_tokens` tokens, every rollback that follows it removes at most
> `n_tokens - 1` tokens — so the state it restores was written by that same
> batch, and the snapshots a *prefill* leaves behind are never read.

If that holds, dropping the prefill tail is free and correct.

### Why it looks true (to be confirmed)

- The rollback depth is bounded by `n_rs_seq` (`= K - 1`): e.g.
  `llama-kv-cache-dsv4.cpp` guards with `rollback > 0 && rollback <= n_rs_seq`.
- The spec loop rolls back only **rejected draft tokens**, which are tokens
  decoded in the *current* verify batch, not in a previous batch.
- `split_equal(..., n_keep_tail)` keeps the last `n_rs_seq + 1` tokens of a
  sequence in the same ubatch — see the `[TAG_RECURRENT_ROLLBACK_SPLITS]`
  comment in `src/models/delta-net-base.cpp`, which says the snapshot logic
  "assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are
  inside the same ubatch".
- The GDN op itself already relies on the same bound: "the fused GDN op relies
  on the same bound — it writes only the last `min(n_seq_tokens, K)` snapshots".

### How to prove it (do all three)

1. **Static enumeration.**  Find every rollback call site (`llama_memory_seq_rm`,
   `llama_memory_recurrent`'s rollback, the spec loop's truncation) and show the
   removed range always starts at or after the current batch's first token.
   Check `split_equal` / `n_keep_tail` and the `n_rs_seq` clamp
   (`src/llama-context.cpp:105`).
2. **Dynamic assertion (the useful one).**  Tag each snapshot slot with the
   absolute token position it represents.  On every rollback, assert
   `target_pos >= last_batch_first_pos`.  Run the MTP gate suite plus
   forced-rollback cases.  Any hit invalidates the invariant *and* tells us
   exactly which flow crosses the boundary.
3. **Adversarial MTP tests.**  Drive the worst case: every draft rejected
   (maximum rollback), acceptance at position 0, `n_max` at its maximum, a
   prompt whose last tokens are prefill tokens, and interleaved
   prefill→verify→prefill (context switch / `n_keep_tail` split).

### If the invariant fails — the guard

Keep the free path for the common case, and make the degenerate case correct
without paying the tail up front:

- The memory module already knows `n_rs_seq`; give each snapshot slot a
  **validity tag** (the position it holds).
- A batch that chunked the whole prompt writes *only* the final state and marks
  the intermediate slots **invalid** (do not leave stale data).
- On a rollback, if the requested slot is valid, use it (free path).  If it is
  invalid — i.e. the rollback really does reach into the prefill — **fall back
  for that rollback only**: recompute the last `K` tokens sequentially (exactly
  what branch 2/3 does today) and write the slots.
- Net effect: correct for all flows, zero cost for every flow that never rolls
  into a prefill (which the evidence says is all of them), and the fallback is
  a rare, measurable event rather than a silent corruption.

### Implementation sketch

- In `gated_delta_net.cu`: relax branch 1's `K == 1` gate to allow `K > 1` for
  long batches, and delete branch 2 (or reduce it to the fallback above).
  Watch the `(cache == nullptr || K == 1)` condition — for `K > 1` with the
  fused state cache, the chunked call writes the state into a cache slot; the
  `K`-snapshot slot layout in the cache has to be handled explicitly (that is
  the other reason branch 2 is sequential-only today).
- Keep branch 3 (`align_boundary`) as the fallback / for the
  "rollback-into-prefill" recompute.
- Validate with: the per-process logits probe (`logits-dump-singlewidth.cpp`)
  for bit-identity, `benchmarks/mtp-adaptive-methodology.md` Protocol A
  (acceptance >= ~0.45, MTP >= plain), `test-backend-ops -o GATED_DELTA_NET`,
  and the hybrid-vs-NCCL coherence gate.

### Acceptance for Option B

- Default-config bit-identity everywhere the gate gives it today, **with the
  gate off** and `KTAIL` gone.
- Prefill back to `GGML_CUDA_GDN_ALIGN_BOUNDARY=0` numbers (~+0.3–0.8 % on the
  27B pp), i.e. the alignment cost goes to zero.
- No `n_max` bound; the KTAIL floor disappears.

---

## Part 2 — qwen35moe batch-width residual (separate, pre-existing)

### What it is

The `qwen35moe` model (Qwen3.6-35B-A3B Q4_K_M) still produces different
token-0 logits for a 1-token decode vs a 3-token batch, **after** the mmvq
(block 13) and GDN-boundary (block 02) fixes.  It is *not* the reported
symptom (that was the dense 27B under `-sm tensor`) and it is not the mmvq
issue.

### Evidence

Per-process token-0 logit hash (no `cb_eval` callback), 1 GPU, chunked GDN off:

| config | W=1 | W=3 |
|---|---|---|
| default | `6138fa2aed535b92` | `789feea67e58c1ed` |
| `GGML_CUDA_DISABLE_FUSION=1` | `661f9740781b0498` | `661f9740781b0498` |

So **`GGML_CUDA_DISABLE_FUSION=1` is the only switch that closes it**
(bit-identical).  All of these leave it divergent:

- `GGML_PAIR_OFF=1`, `GGML_PAIR_DENSE_OFF=1`
- `GGML_CUDA_DISABLE_WEIGHTED_DOWN=1`
- `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1`
- `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
- the mmvq `MUL_MAT_ID` dispatch — item-split vs `_moe` vs ksplit all identical
- `GGML_CUDA_FA=0` (still diverges)
- `GGML_CUDA_DISABLE_GRAPHS=1`, `LLAMA_GRAPH_REUSE_DISABLE=1`,
  `GGML_CUDA_NO_PEER_COPY=1`

Impact: **the MoE MTP gate is healthy** — acceptance 0.675 / 153.1 t/s
(recorded baseline 0.633).  So this is a latent numerics drift, not a
throughput problem; it is the last hole in the "decode == verify" invariant on
MoE.

### Instrumentation gotcha (important)

`llama_context_params.cb_eval` **changes MoE numerics**: with the callback set,
W=1 == W=3 (both `555cd19f4033a93f`); without it, W=1 != W=3.  So a
`cb_eval`-based per-node dump is **not** a valid instrument on MoE — it masks
the very bug.  Use the per-process logits hash
(`logits-dump-singlewidth.cpp`, no callback) to observe, and instrument
on the **backend** side instead.

A backend-side per-node dump was prototyped during the investigation: an
env-gated hash of `node` right after `ggml_cuda_compute_forward` in
`ggml_backend_cuda_graph_compute` (gated on the existence of
`/tmp/nodedump_on`, which the probe creates only around the decode batch).  It
showed **zero diverging compute nodes** — only state/cache writes (`CPY`,
`VIEW`, `SET_ROWS`) — which is exactly the callback artefact; with the
callback removed the divergence returns, so this needs reworking to run
without `cb_eval` (e.g. gate the dump on a phase file created by the probe and
dump **every** node, including fused ones).

### Where to look (ranked)

1. **`ggml_backend_cuda_graph_optimize` → `add_alloc_dep` for
   `ggml_match_moe_weighted_reduction`** (`ggml-cuda.cu`, gated by the same
   `disable_fusion`).  A *missing* alloc dependency is buffer aliasing, which
   would corrupt numerically in a batch-width-dependent way (different
   allocation patterns for 1 vs 3 tokens).  This is the single most suspicious
   match: it is the only MoE-specific thing behind the same `disable_fusion`
   flag that also gates some fusions.
2. **The `ggml_cuda_try_fuse` arms** behind `disable_fusion` that are not
   covered by `PAIR_OFF` / `DISABLE_WEIGHTED_DOWN` / `DISABLE_MOE_MMQ_FUSION`
   / `DISABLE_MMQ_ROUTED` — walk `ggml_cuda_try_fuse` and add a per-arm env
   gate temporarily, or bisect by compiling arms out.
3. **The `_moe`/weighted-reduction epilogue** (`x_scale_channel_dst`
   `channel_dst + token_idx*nchannels_dst`) under a fused kernel — batch-width
   dependent indexing.

### Acceptance for the MoE item

- Per-process token-0 logit hash: W=1 == W=3 == W=5 on qwen35moe, chunked GDN
  on and off, 1 GPU and 2/3-GPU tensor, **without** `DISABLE_FUSION`.
- The MoE MTP gate holds or improves (>= 0.633 acceptance, MTP >= plain).
- No perf regression from the fix.

---

## Tooling in this directory

- `logits-width-tensor.cpp` — the original `SPLIT=`-aware probe (3 contexts in
  one process).  **Note:** its W=1 vs W=3 numbers can reflect a
  first-context/multi-context artefact; prefer the single-width tool below.
- `logits-dump-singlewidth.cpp` — **the reliable one**: one decode width per
  process, prints the token-0 logit hash, no `cb_eval`.  `CB=0` disables the
  callback explicitly; `NGL`, `RS`, `FA`, `SPLIT` env as in the probe.
- `mmvq-dense-ksplit-block13.patch` — the block-13 dense-MMVQ alignment as
  originally applied (now in the delivery).
- `HANDOVER-2026-09-11.md` — the root-cause record for the fixes already
  delivered.

## Related records

- `../issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FIX.md` — the GDN gate
  (now default-ON, `KTAIL=16`).
- `patches/README.md` block-02 and block-13 notes.
- `benchmarks/mtp-adaptive-methodology.md` — the MTP gate protocol.
- `GREEDY-PURITY.md` §6 — the per-row bit-identity invariant this all serves.
