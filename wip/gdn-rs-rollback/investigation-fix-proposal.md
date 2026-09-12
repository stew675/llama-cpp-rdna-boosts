# Fix proposal: ngram-mod long verify batches corrupt recurrent-state rollback

## TL;DR

Yes, `ngram-mod` is the trigger. It is not a bug in ngram-mod itself, but a bad
interaction between ngram-mod's long drafts and the RDNa chunked-GDN prefill
optimization.

The recurrent rollback path stores `n_rs_seq` "rollback snapshots" and assumes a
batch that is rolled back into is a speculative verify batch, i.e. at most
`n_rs_seq + 1` tokens. `need_n_rs_seq()` sizes the snapshot depth from
`speculative.draft.n_max` only (7 here, from `--spec-draft-n-max 7`). ngram-mod
drafts up to `--spec-ngram-mod-n-max` (64 here), so one verify batch decodes up
to 65 tokens in a single sequence. That is far past the chunked-GDN threshold
(`max(K, 16)`), so the whole-batch chunked kernel runs, which writes only the
newest state and no rollback snapshots. A partial acceptance of only a few
tokens then restores a snapshot plane that was never written this batch, and
the recurrent state silently rewinds.

Without ngram-mod the longest verify batch is `draft.n_max + 1 == 8 <= 16`, so
the sequential GDN kernel runs and writes valid snapshots. That is exactly why
removing ngram-mod made the warning and the corruption disappear.

The warning in `src/llama-memory-recurrent.cpp` is correct and was added to
catch this. It is not a false positive.

---

## 1. How the rollback is supposed to work

For recurrent / hybrid models the context is created with
`cparams.n_rs_seq > 0` (`common/common.cpp:1762`). The recurrent cache widens
every per-layer state tensor to `1 + n_rs_seq` groups
(`src/llama-memory-recurrent.cpp:101`). Group 0 is the current state, group `r`
is the state after `r` tokens are removed from the tail.

`seq_rm()` implements the rollback (`src/llama-memory-recurrent.cpp:193-220`):

```cpp
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    const llama_pos rollback = cell.pos - (p0 - 1);
    const bool pending = rs_idx[seq_id] != 0;
    if (!pending && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
        // guard/warning here
        set_rs_idx(seq_id, (uint32_t) rollback);
        cell.pos = p0 - 1;
        return true;
    }
    return false;
}
```

The next graph build reads the pending snapshot plane through `s_copy()`
(`src/llama-memory-recurrent.cpp:1332-1348`), which returns
`idx * mem->size + src0`, i.e. row `rs_idx[seq]` of the widest state tensor.

The producer side is the GDN op. Both the CPU kernel
(`ggml/src/ggml-cpu/ops.cpp:11138-11247`) and the CUDA **sequential** kernel
(`ggml/src/ggml-cuda/gated_delta_net.cu:146-158`) write per-token snapshots:

```cpp
// snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
const int target_slot = (int) n_tokens - 1 - t;
if (target_slot >= 0 && target_slot < K) {
    float * curr_state = state + target_slot * state_slot_stride;
    ...
}
```

This loop writes slots `0..K-1` for the **last** `min(n_tokens, K)` tokens of
the batch, whatever the batch length is. So a long batch is rollback-safe as
long as it goes through the sequential kernel. Mamba's `ssm_scan` CUDA kernel
does the same (`ggml/src/ggml-cuda/ssm-scan.cu:221-227`) and has no chunked
variant.

## 2. Where it breaks: the chunked-GDN prefill path

`ggml/src/ggml-cuda/gated_delta_net.cu:290-370` adds a whole-batch chunked
prefill path. The dispatch is:

```cpp
const int K = ggml_get_op_params_i32(dst, 0);
const bool keep_rs = K > 1;
...
if (!kda && n_tokens > 1 && (S_v == 16 || 32 || 64 || 128)) {
    const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
    if (env == nullptr || strcmp(env, "0") != 0) {
        const int64_t GDN_CHUNKED_MIN_TOKENS = K > 16 ? (int64_t) K : 16;
        ...
        if (n_seqs == 1) {
            if (n_tokens > GDN_CHUNKED_MIN_TOKENS) {
                if (launch_chunked(cache ? cache->data : nullptr, 0)) {
                    return;               // <-- no rollback snapshots written
                }
            }
        }
        ...
    }
}
```

When this path is taken the chunked kernel writes the final state into the
fused-cache slot 0 only (`gated_delta_net_chunked.cu:672-700`, and
`gated_delta_net.cu:373-379` sets `state_slot_stride = H*S_v*S_v`, i.e. one
slot). Slots `1..K-1` are not touched. The code comment states the
assumption explicitly (`gated_delta_net.cu:297-301`):

> A batch with more tokens than max(K, 16) cannot be a speculative verify
> batch (a verify batch decodes `n_tokens <= K = n_rs_seq + 1`) and is therefore
> never rolled back into, so the K rollback snapshots are useless to it.

That assumption is only true if every speculator's maximum draft is bounded by
`n_rs_seq`. It is **false** for ngram-mod.

## 3. Why ngram-mod breaks the assumption

`n_rs_seq` is derived here (`common/common.h:402-412`):

```cpp
uint32_t need_n_rs_seq() const {
    bool needs_rs_seq = has_mtp() || eagle3 || dflash || dspark;
    return needs_rs_seq ? draft.n_max : 0u;
}
```

Only `draft.n_max` matters. The ngram speculators are not considered. In this
config `draft.n_max = 7`, so `n_rs_seq = 7` and `K = 8`.

ngram-mod can return up to `params.n_max` tokens
(`common/speculative.cpp:2050-2090`, `--spec-ngram-mod-n-max 64`, floor
`--spec-ngram-mod-n-min 48`). The server then puts the sampled token plus all
draft tokens into one batch (`tools/server/server-context.cpp:525-535`), so the
verify batch is `1 + n_draft` tokens, up to 65, all in one sequence. This is
the "49 tokens" in the warning: an ngram-mod draft of 48 (`n_min`).

`common_speculative_n_max()` already knows this: it returns `max` over all
enabled implementations, so it returns 64 for this config
(`common/speculative.cpp:2439-2475`). It is used for output/batch sizing and
for `n_accepted_per_pos`, but **not** for `n_rs_seq`. That is the gap.

Because ngram-mod has higher priority than MTP in the impl list
(`common/speculative.cpp:2738-2743`: ngram types are added before draft-MTP),
ngram-mod is the drafter used during long-context recall, and MTP is only the
fallback when ngram-mod finds no match.

## 4. The server guard is internally inconsistent

There are two different `use_ckpt_tgt` decisions in the speculative server
path:

Before the decode, when the checkpoint is created
(`tools/server/server-context.cpp:3068-3081`):

```cpp
const bool use_ckpt_tgt =
    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));
```

After the decode, when acceptance is known
(`tools/server/server-context.cpp:3922-3930`):

```cpp
const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();
const bool use_ckpt_tgt =
    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));
```

For a 48-token ngram-mod draft that is almost fully accepted (`n_rollback` small):

* a checkpoint **is** created (48 > 7), so the safe path is available; but
* the accept path **does not** use it (1 <= 7), so it calls
  `slot.mem.seq_rm(...)` and takes the snapshot path instead.

The snapshot path is then read from a plane the chunked kernel never wrote.

## 5. Why the state does not recover

The restored plane contains the state from some older batch, i.e. a position
far behind the current context. The delta-net recurrent state is silently
rewound by many tokens. Subsequent tokens are then produced from a
wrong-but-finite state, so decoding "works" but is permanently off. Every later
partial rollback over that state is not necessarily enough to detect it, hence
the corruption the user observed.

---

## 6. Fix options

### Option A - server-side: use the checkpoint whenever the draft exceeded the snapshot depth (minimal, recommended stopgap)

In the accept path, make the post-decode condition match the pre-decode
condition:

```cpp
const bool use_ckpt_tgt =
    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
        (n_rollback > llama_n_rs_seq(ctx_tgt) || n_draft > llama_n_rs_seq(ctx_tgt)));
```

A checkpoint is already created whenever `draft.size() > n_rs_seq`, so the
restore is always available. This is correct for every speculator and does not
need to know the CUDA chunked threshold.

An even simpler variant, if you do not mind losing the snapshot fast path
entirely when long drafters are enabled: right after
`ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt)` at
`server-context.cpp:1238`, demote to `FULL` when
`common_speculative_n_max(&params_base.speculative) > (int32_t) llama_n_rs_seq(ctx_tgt)`.
Then all partial rollbacks use checkpoints.

Cost: on a small tail rejection (`n_rollback` 1..7) after a long draft, the
accepted prefix is restored and replayed through the target model instead of
using the snapshot. With ngram-mod this can happen often, so the perf hit is
real, but correctness is guaranteed.

### Option B - decouple the chunked threshold from `K` (recommended proper fix)

The snapshots do not need to cover the whole draft. The sequential kernel
already writes the last `K` per-token snapshots of **any** batch, and only
`n_rollback <= n_rs_seq` is ever served from snapshots; bigger rollbacks
already use checkpoints. So the only requirement is that any batch that can be
rolled back must run the sequential kernel. That means the chunked threshold
must exceed the longest possible verify batch, not just `K`.

Plumbing:

1. Add `uint32_t n_draft_max` to `llama_cparams` (max draft length any enabled
   speculator can produce). Fill it from the already existing
   `common_speculative_n_max(&params.speculative)` in
   `common_context_params_to_llama()` (`common/common.cpp:1762` area) and copy
   it in `llama-context.cpp` next to `cparams.n_rs_seq`.
2. Pass the bound to the GDN op. Either add an argument to
   `ggml_gated_delta_net()` (`ggml/src/ggml.c:6696`) or set a second op param
   from `build_recurrent_attn()` (`src/models/delta-net-base.cpp:576`).
3. In the CUDA dispatch (`gated_delta_net.cu:328`) use:

   ```cpp
   const int64_t n_rollback_batch = ggml_get_op_params_i32(dst, 1); // 0 = unknown
   const int64_t base  = K > 16 ? (int64_t) K : 16;
   const int64_t GDN_CHUNKED_MIN_TOKENS = std::max(base, n_rollback_batch);
   ```

   With `n_draft_max = 64` the bound is 65. Verify batches (<= 65 tokens) run
   the sequential kernel and write their `K` snapshots; long prefills still take
   the chunked path. `K` and the snapshot memory stay at `n_rs_seq + 1`.
4. Fix the guard in `seq_rm()` (`llama-memory-recurrent.cpp:206`) to compare
   against the same bound instead of `n_rs_seq + 1`, so it stays a real
   invariant check and does not warn on now-safe long batches. This needs
   `n_draft_max` (or the bound) plumbed into `llama_memory_recurrent`.

Cost: a verify batch in `(max(K,16), 65]` now runs the sequential GDN kernel
instead of the chunked one. The fork's chunked path deliberately avoids a
K-dependent boundary to keep prefill bit-identical with/without speculation;
this reintroduces that difference for batches in that narrow range. The existing
code already accepts the same kind of difference for `K > 16`
(`gated_delta_net.cu:310-313`), so this is consistent.

Net effect: small-tail rollbacks after a long ngram recall stay on the fast
snapshot path; no checkpoint replay, no extra snapshot memory.

### Option C - size `n_rs_seq` from the longest draft (simple but expensive)

Make `need_n_rs_seq()` return the max draft over all enabled speculators
instead of `draft.n_max`. Then `K = 65` and the existing invariant holds.
Memory cost on Qwen3.8-27B (qwen35, 48 GDN layers, `ssm_inner 6144`,
`state_size 128`, `n_rs_seq` rows on one cell):

* `n_embd_s = 128 * 6144 = 786432`
* `n_rs_seq = 7`:  `786432 * 8 * 4 * 48` = 1152 MiB ~= 1.13 GiB
* `n_rs_seq = 64`: `786432 * 65 * 4 * 48` = 9360 MiB ~= 9.14 GiB

About +8.0 GiB on the S tensor (plus ~0.3 GiB on the conv-state R tensor),
split over both GPUs. This
also forces the RS fast path even for MTP, so it is the most expensive option.

### Option D - disable the whole-batch chunked path when `K > 1`

Force the sequential kernel whenever snapshots are enabled. Correct and simple,
but it throws away the chunked prefill optimization for every long prefill
whenever speculation is on, which is the common case. Not recommended.

### Option E - make the chunked kernel snapshot-safe

Write the `K` newest per-token states from the chunked scan. This is the only
fix that keeps both the chunked kernel and the fast rollback. It needs the
chunked kernel to retain per-token states (or to run a sequential tail), and to
reintroduce a K-dependent boundary. The fork explicitly avoided this. High
complexity for the benefit already provided by Option B.

---

## 7. Recommendation

1. Apply **Option A** immediately (one line) to stop the corruption, and keep
   the `seq_rm` guard as-is: the guard warning is the correct detector and
   should no longer fire.
2. Follow up with **Option B** if the checkpoint replay cost on small tail
   rejections is measurable. Option B restores the fast snapshot rollback for
   long ngram-mod drafts without growing the snapshot planes.
3. Do **not** use Option C for `--spec-ngram-mod-n-max 64` unless the ~8.3 GiB
   is acceptable.

## 8. Reproduction

The bug needs three things together:

* a recurrent / hybrid model that uses the fused GDN path (e.g. qwen35),
* `n_rs_seq > 0` from a draft-model speculator (MTP here, so
  `--spec-draft-n-max 7`), and
* a speculator that can draft more than `n_rs_seq` tokens: ngram-mod with a
  high `--spec-ngram-mod-n-max` (64) and `n-min` (48).

Recipe:

1. Start llama-server as in the report (`draft-mtp-adaptive,ngram-mod`,
   `n-max 7` for MTP, `ngram-mod n-min 48 / n-max 64`).
2. Feed a long context, then make the model continue a passage it already has
   in context so that ngram-mod recalls a long contiguous run.
3. On a verify batch of more than 16 tokens, reject a few trailing tokens so
   `n_rollback` is small (1..7). The `seq_rm` warning fires and the recurrent
   state is rewound.

Confirming the diagnosis without a full run:

* set `GGML_CUDA_GDN_CHUNKED=0` (sequential kernel). The warning and the
  corruption should disappear at the cost of prefill speed. This is the
  workaround already printed in the warning text.
* or remove ngram-mod, or lower `--spec-ngram-mod-n-max` to at most
  `--spec-draft-n-max` (7). In both cases verify batches stay at or below the
  chunked threshold and the sequential kernel writes valid snapshots.

A targeted regression test can reuse `tests/test-recurrent-state-rollback.cpp`:
it currently decodes exactly `n_rs_seq + 1` tokens before rolling back. A case
that decodes more than `max(n_rs_seq + 1, 16)` tokens and then rolls back
`n_rollback <= n_rs_seq` would reproduce the stale-plane restore on CUDA (the
CPU kernel always writes snapshots, so the test only fails on the GPU backend).

## 9. Files touched by the recommendation

Option A:
* `tools/server/server-context.cpp` (accept-path `use_ckpt_tgt`)

Option B additionally:
* `src/llama-cparams.h`, `src/llama-context.cpp`, `common/common.cpp`
* `src/models/delta-net-base.cpp`
* `ggml/src/ggml.c`, `ggml/include/ggml.h` (`ggml_gated_delta_net` signature)
* `ggml/src/ggml-cuda/gated_delta_net.cu`
* `src/llama-memory-recurrent.{h,cpp}` (guard bound)
