# Follow-ups — free GDN prefill alignment (Option B) + the qwen35moe batch-width residual

Date: 2026-09-11.  Box: GFX1201, 3x R9700 (gfx1201, RDNA4), ROCm 7.14
(`/opt/rocm-7.14-gfx1201`).  Companion to `HANDOVER-2026-09-11.md` (the
plain-vs-spec root-cause + the mmvq/block-13 and GDN-gate/block-02 fixes).

Status: **Part 1 is DONE and delivered** (2026-09-11, block-02 amendment, tip
`30d119ea9`) — see the closing note in Part 1.  **Parts 2 and 3 are open** and
deferred by the maintainer to follow-up sessions.  The delivery is
at canonical tip `30d119ea9` (tree `29714ad1f`), 15 blocks, block 02 with the
K-independent whole-batch chunked GDN prefill (no tail, no gate, rollback
guard).

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

### Acceptance for Option B — MET 2026-09-11 (delivered as the block-02 amendment)

- Default-config bit-identity **with the gate and `KTAIL` gone**: 27B 2-GPU
  tensor `none == n1 == n4 == n5` (`6e8ccd25`), 3-GPU tensor `none == n4`,
  1-GPU 4B `671d6096`; probe (`RS=from_w`, P=256) `W = 1/3/5/6` all
  `a4817ee6`, with `RS=6 W=6 == RS=0 W=1` proving the prefill is now
  K-independent.
- ~~Prefill back to `ALIGN=0` numbers~~ **better: parity with them at zero
  cost.**  27B Q8_0 1 GPU pp512/2048/4096 = 1385.3/1356.4/1328.2 vs
  1384.7/1355.0/1327.8 for the old K-dependent boundary, tg unchanged.  The
  previous -0.3..-0.8 % tail is gone entirely.
- **No `n_max` bound from this mechanism**; KTAIL gone.  (A *different*,
  pre-existing `n_max <= 5` purity cap remains — Part 3.)
- **Invariant proved as required:** static (the `n_max`/rollback bound is
  stated in `delta-net-base.cpp` itself) + dynamic (449 rollbacks over
  llama-cli `draft-mtp` n_max 1/4/8/16 plus 20 in llama-server
  `--cache-reuse`; *every* one preceded by a batch of `<= K` tokens, zero
  crossings) + adversarial (context shift, `n_cache_reuse`, adaptive depth).
  The guard option (B) was implemented: `llama_memory_recurrent::seq_rm` warns
  once if a rollback ever crosses a batch boundary.
- **The gate and both K-dependent branches were deleted** (~118 lines) rather
  than kept: they were unreachable with the gate ON, and `GGML_CUDA_GDN_CHUNKED=0`
  is a strictly better fallback (correct *and* bit-identical plain-vs-MTP)
  than `GGML_CUDA_GDN_ALIGN_BOUNDARY=0` was.

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

## Part 3 — the multi-token verify batch is not bit-identical to single-token decode

> **STATUS 2026-09-11: root-caused, and the fork-specific half is FIXED.**
> There are two independent causes.
> **Cause A (fork-specific) = block 12's size-based all-reduce dispatch** --
> `ggml_backend_cuda_comm_is_small()` routed reductions below 32768 elements (2
> devices) to the internal pipeline and above them to NCCL; the reduced tensors
> scale with the batch width (`ne = ne0 * n_tokens`, ne0 = 5120), so a 7-token
> verify batch crossed the limit and was reduced by a different algorithm than
> 1-token decode.  **Fixed** by raising the 2-device crossover to 131072 (block
> 12, tip `daf32f804`): probe `W = 1/6/7/8` are now all `a4817ee6` and text
> `none == n4 == n6 == n7` (`6e8ccd25`), with MTP +12% at `n_max 6`/`12` and
> pp/tg unchanged.  **Cause B (deliberate, still open) = the FA
> tile-vs-WMMA switch at `Q->ne[1] > 8`**, which caps the guarantee at the
> designed `n_max <= 7`; it is documented in the fork's own `fattn.cu` comment.
> Verified with the raw-logit probe across 1/2/3 GPUs and both split modes:
> `W = 1..8` bit-identical, `W = 9` divergent, in every one.  Note that the
> 3-GPU *text* gate at `n_max = 8` coincidentally matched while the logits had
> already diverged -- establish boundaries with the probe, not the text hash.
> The *upstream* comparison below is a separate mechanism again (the CPU
> backend's own batched-vs-single dispatch).  Everything after this box is the
> pre-fix record, kept for method.


### What it is

`--spec-type none == draft-mtp` is byte-identical only up to **`n_max = 5`**
(a 6-token verify batch).  From `n_max = 6` (7-token verify batch) onward the
verify batch's `MUL_MAT` columns are computed by a different kernel path than
single-token decode and the logits differ by ~1e-6, so greedy near-ties flip.

This is **pre-existing and independent of block 02**.  The delivered KTAIL=16
build and the current whole-batch-chunked build diverge in exactly the same
place:

| `--spec-draft-n-max` | none / 1 / 4 / 5 | 6 / 7 | 8 / 9 / 10 | 12 | 16 |
|---|---|---|---|---|---|
| delivered (KTAIL=16) | equal | `5037ef2e` | `e721b8b5` | `e721b8b5` | `5037ef2e` |
| whole-batch chunked | equal | `b6d86d62` | `ed922c76` | `5037ef2e` | `4f3ee41c` |

**This is the part nobody had validated:** every earlier "`n_max <= 15`" claim
was tested only to `n_max = 4`.  The claim is wrong and has been corrected in
`GREEDY-PURITY.md` §11 and `benchmarks/mtp-adaptive-methodology.md`.

### Evidence (bounded)

1. It is a pure **width** effect, not `K` and not the GDN.  At `RS=0` (no
   snapshots at all, `K = 1`) the probe still separates on width alone:

   ```
   RS=0   W = 1..6 -> a4817ee6      W = 7,8 -> e286b75c      W = 9 -> 24f302f6
   ```

   The probe hashes `llama_get_logits_ith(ctx, 0)` — batch item 0 — so this
   says *adding columns changes column 0's own result*.
2. The GDN is exonerated: at `RS=6` (K=7) `W=1` and `W=6` both give `a4817ee6`,
   identical to `RS=0 W=1`, i.e. the K=7 prefill is already K-independent.
3. **`W >= 9` is explained:** `MMVQ_MAX_BATCH_SIZE = 8` (`mmvq.cuh`) — past 8
   columns `ggml_cuda_mul_mat` leaves the vector kernels for MMQ, a different
   algorithm/K-accumulation order.
4. **`W = 7` is NOT yet explained** (inside MMVQ's range).
5. **Upstream is affected and is worse.**  Upstream master's own
   `calc_nwarps`/`calc_rows_per_block` switch on `ncols_dst`, and
   `ggml_cuda_mul_mat` uses MMVQ only for `ncols_dst <= MMVQ_MAX_BATCH_SIZE`.
   Measured on upstream master `9cf3bf256` (clean checkout, unmodified
   `mmvq.cu`), **CPU backend**, 4B Q8_0, `P = 256`:
   `W = 1 -> 9024dd2e...` but `W = 2..12 -> 3cd0eb0e...`.  So upstream diverges
   at the *first* width step.  (That build was CPU-only — upstream's ROCm
   boundary was not measured, so the fork's `n_max <= 5` is a fork result, not
   an upstream guarantee.)

### Where to look (ranked)

1. **The `ncols_dst`-dependent mmvq dispatch** — `calc_nwarps` (upstream
   groups 1-4 / 5-8 / default) and `calc_rows_per_block` (1 / 2-8 / default),
   plus the fork's `mul_mat_vec_q_switch_ncols_dst` arms added by block 13.
   The `W = 7` boundary must come from one of the template instantiations or an
   arm condition; bisect by forcing `MMVQ_PARAMETERS_GENERIC` / a fixed
   `nwarps` / `rows_per_block` and re-running the `RS=0` width probe (that
   probe is the whole instrument — it needs no spec decode).
   Note `ncols_dst` is a **template parameter**, so different widths are
   different code; the question is which instantiation changes the *K-order*
   of the dot product (`halve_iters`/`small_k`/rpb do).
2. **Non-MMVQ width-dependent selection on the verify path** — the FA kernel
   from `Q->ne[1]` (block 00 already handles the `parallel_blocks` half), the
   KV/mask path, and scheduler-level node splits that differ with width.
3. **Method:** extend the backend per-node dump to run without `cb_eval` (see
   the gotcha in Part 2 — on MoE the callback *changes* the numerics and masks
   the bug), then diff the *first* diverging node for `RS=0 W=6` vs `W=7`.
   This is dense 27B, so `cb_eval` is safe there; it is only MoE that needs the
   backend-side dump.

### Why it matters

Adaptive MTP's **recommended `n_max = 12` sits outside the pure range**, so on
this model speculative decoding *does* change the output at near-ties even
though acceptance and throughput are healthy.  Until this is fixed, the
`none == draft-mtp` equality gate must not be used above `n_max = 5`
(`benchmarks/mtp-adaptive-methodology.md` rule 4).

### Acceptance

- Probe `RS=0`: `W = 1..K` all equal for every width a verify batch can have
  (currently only `W <= 6`).
- Text: `none == draft-mtp` for at least `n_max = 12`, ideally 16, on 1/2/3-GPU
  and `-sm tensor`/`-sm layer`.
- No perf regression on pp or tg.
- Cross-check against Part 2: if the same dispatch fix closes the qwen35moe
  residual, say so — they may share a cause.

### Upstream

The mechanism is upstream (see evidence 5), so a fix is a strong
`upstream/UPSTREAM-PR-*.md` candidate once a minimal reproducer exists.  The
`RS=0` width probe is already a candidate test to attach to such a PR.

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
