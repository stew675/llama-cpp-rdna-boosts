# Upstream-PR proposal — llama: `llm_graph_input_attn_k` tolerates an absent kq mask

Status: PREPARED 2026-09-10.  Verified against **unadulterated master** (`9cf3bf256`): the patch applies
clean (`git apply --check`), it compiles, and a same-seed run on the 4B control model is byte-identical
(no behaviour change for graphs that have a mask).  Not yet filed.  Patch file:
`UPSTREAM-PR-attn-k-null-mask-guard.patch` (1 file, `src/llama-graph.cpp`, +8/−2; also applies clean to
`9113cc188`).

**Value: consistency/hardening, not a live bug fix.**  On master today every `llm_graph_input_attn_k`
construction site assigns `self_kq_mask = build_attn_inp_kq_mask(...)`, which always returns a tensor,
so a null mask is unreachable — the guard is what the sibling class already does, and it is what any
future null-mask graph needs (see the fork note below).  File it if that consistency is wanted, or fold
it into the feature that first needs an absent mask.

## The inconsistency

`llm_graph_input_attn_kv::set_input` (the sibling, ~L470) already guards the mask fill:

```cpp
    // the mask is left unallocated when the graph only stores K/V without attending
    // (e.g. DFlash's KV-injection pass)
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }
```

`llm_graph_input_attn_k::set_input` (~L504) does not: it calls `set_input_kq_mask(self_kq_mask, ...)`
unconditionally.  If `self_kq_mask` is ever null (or present but unallocated — the allocator leaves an
input tensor with no consumer without a buffer), that call dereferences a null buffer:

* `llama_kv_cache::set_input_kq_mask` starts with `GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))`
  and then writes `dst->data` — a null-buffer abort, not a graceful skip.

The same class's `can_reuse_impl` also dereferences the mask through `can_reuse_kq_mask`, so a null mask
cannot even be *checked* for reuse as written.

## Change

1. Guard the fill exactly like the sibling class does.
2. Let `can_reuse_impl` accept a null mask (`self_kq_mask == nullptr || can_reuse_kq_mask(...)`) so a
   graph that legitimately carries no mask can be reused on its other properties.

No behaviour change for any graph that has a mask (the only kind master builds today).

## Validation

* master `9cf3bf256`, CPU build, Qwen3.5-4B-Q8_0: same-seed generated text
  (`-p "The capital of France is" -n 16 --seed 42 --temp 0`) is **byte-identical** with and without the
  patch.
* `git apply --check` clean on `9cf3bf256` and on `9113cc188`.

**Not validated here:** an actual null-mask graph (none exists upstream today, so the guarded branch is
not exercised).  The fork reaches it — its derived-kq-mask work (beta block 15, win V3) builds
graphs where the packed mask has no consumer, which is how the missing guard was found (the fork has
carried the guard since 2026-09-10 as part of that block).  If a null-mask feature lands upstream, this
guard is its prerequisite, and that feature's own tests are the real coverage.

## Notes for filing

* Review-sensitive point: this is defensive only — say so in the description rather than claiming a
  reproduction, because there is none on master.
* Alternative shape if a reviewer prefers: `GGML_ASSERT(self_kq_mask)` in both places (declare the
  invariant instead of tolerating its absence).  The fork needs the tolerant form, the sibling class
  already tolerates it, so tolerance is the smaller surprise.
* Base: applies clean to `9cf3bf256` (master at preparation time) and to `9113cc188`.
