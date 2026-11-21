# R12 — paused mid-rung (session 3, 2026-11-21)  ·  **since LANDED 2026-11-20 as `a9d8e1a96`**

> **Kept as evidence, no longer an entry point.**  The next session merged `common_speculative_impl_draft_dflash`
> onto upstream's `init(opt)` seam, fixed the two splice-orphaned server features, and landed the rung — the log is
> `../RUNGS.md` §R12 and the DFlash decision is `../DECISIONS.md` D17.  This file describes the paused state and
> stays accurate as a record of it.

**Status: NOT landed. The ladder is clean at `dbe4fdd40` (R11).**  `r12-resolved-but-unbuildable.diff` is the
whole work of the session's R12 attempt: `git cherry-pick dad1442b3`, all ten conflicted files resolved, and the
tree failing to build in exactly one place (`common/speculative.cpp`) — the speculator-class merge described
below.  Apply it over `dbe4fdd40` to resume, or just redo the cherry-pick (the resolutions are in RUNGS.md R12).

## What was already resolved (and is in the patch)

| path | resolution |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/include/ggml-rpc.h` | op count **108** = upstream's 100 + block 11's 6 + block 12's 2 (`ALL_GATHER`, `TOKEN_SCATTER`), recounted from the enum; `GGML_OP_SLEEP` / `ALL_GATHER` / `TOKEN_SCATTER` unioned in both string tables; rpc patch version 9 |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | the delivery's `q2_0_mul_add` kept (the live path), upstream's `q2_0_unpack_32/dot_32/acc_block_avx2` trio **not** carried: the merged `ggml_vec_dot_q2_0_q8_0` is block 12's body and the trio is referenced by nothing |
| `src/llama-kv-cache.cpp` | the delivery's **parked-aware** scan kept (block 12's `park_bound`/`used_max_p1_below`), which *supersedes* R09's adoption of upstream's `seq_rm(seq_id,p0,p1)` at this call site — that API has no notion of the parked region.  A future `seq_rm_below(..., lim)` would be the fast form |
| `src/models/dflash.cpp` | upstream's `get_lm_head` tail + the delivery's `build_dflash2_conv` and `build_lilicorr_scores` (the other two builders live in upstream's `dspark.cpp`/`dflash2.cpp` after the #32016 split) |
| `src/models/dflash2.cpp` | block 12's **all-gather of the vocab-split logits** ported into upstream's `build_dflash2_selector` — the *only* difference between the two versions of that function |
| `tools/server/server-task.{h,cpp}` | the delivery's duplicate `server_prompt`/`server_prompt_cache`/`server_prompt_data` structs **deleted**: upstream #32154 moved them to `server-slot-sched.{h,cpp}` |
| `tools/server/server-slot-sched.{h,cpp}` | the delivery's prompt-cache additions **ported into upstream's home**: `spare` buffers, `take()`/`recycle()` (+ `LLAMA_SERVER_CACHE_REUSE` / `HUGEPAGE` and the madvise path), `find()` split out of `load()` with lora filtering, `data.main.empty()` skip, and `recycle()` on the evict/restore paths |
| `tools/server/server-context.cpp` | the LCP-similarity selection block **dropped** (upstream #32154's `server_slot_sched` owns it); the checkpoint-restore path kept as upstream's, with block 12's `slot.n_tokens_ckpt_restored = cur.n_tokens` re-inserted so the ckpt-reuse gate still gets fed |
| `common/speculative.cpp` | members / logs / `is_mrope`+`strict_pos` / `opt.dev_features && !is_lilicorr` union applied — **but the splice is wrong** (see below) |

## The blocker

`common_speculative_impl_draft_dflash` is **two implementations of the same class**:

* upstream (`32bdc3a89`, #32016, the R02 / D5 refactor): a delegating ctor pair taking `dflash_options`, a
  `void init(const dflash_options & opt)` seam, members `{features_buf, smpls, backend_chains, n_embd_dec/enc/tgt,
  block_size, causal_attn, target_layer_ids(_n), use_dev_features}`, and no `is_dspark`/`is_mrope`/`strict_pos`/
  `ingest_*`/`lilicorr_*`/`dflash2_*` state at all;
* the delivery (block 12): one ctor doing everything inline, plus `is_dspark`, `sample_from_anchor`,
  `is_dflash2`, `is_mrope`, `strict_pos`, `selector_top_k`, `dflash2_rng/_seed`, `ingest_window`/`ingest_from`
  (the prompt-tail ingest), the `lilicorr_*` state, and methods (`begin`, `set_prompt_end`, the tail-filtered
  `draft`) that upstream's version does not have.

They are 388 and 620 lines and their methods interleave, so this is not a union: it is a merge of two live
implementations of the speculator seam, and a subtle error there silently corrupts drafting (it is the one place
where "the text still looks right" is not a check).  It needs a fresh session, the `benchmarks/mtp-adaptive-
methodology.md` four-axis protocol as its oracle, and the same treatment D5 got at R02: decide *where the delivery's
state and seams belong in upstream's `init(options)` shape* — most likely the delivery's per-feature state becomes
fields set by `init()` from the options, and its `begin`/`set_prompt_end`/`draft` additions are added as overrides
on upstream's base.

Nothing here is a decision for the maintainer; it is unfinished work.  The tree was reverted rather than committed
because a rung must build and pass its coherence arm, and this one did not.
