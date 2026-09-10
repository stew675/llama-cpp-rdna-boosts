# Upstream-PR proposal — ggml alloc: release view sources whose views are never consumed

Status: PREPARED 2026-09-10.  Verified against **unadulterated master** (`9cf3bf256`): the patch
applies clean (`git apply --check`), the repro fails before / passes after, and the allocator tests
pass (see Validation).  Not yet filed.  Patch file: `UPSTREAM-PR-ggml-alloc-unused-view.patch`
(1 file, `ggml/src/ggml-alloc.c`, +35/−0; also applies clean to `9113cc188`).

## TL;DR

`ggml_gallocr_alloc_graph_impl` charges a view's contribution to its view source by **view-node
existence**, but gives it back only when that view node is *itself released*.  A view node with no
consumers is never released, so the source's `n_views` stays above zero forever: the source can then
neither be freed (`view_src_hn->n_views == 0`) nor reused in place (`p_hn->n_views == 0`), and the
buffer grows by the size of every such view source.

The idiom that triggers it is a `ggml_cpy` expanded into the graph purely for its side effect —
copying into a view of a preallocated tensor, a natural way to assemble one large tensor from chunks.
Measured with a 12-layer repro: **56.00 MiB arena with the idiom vs 16.00 MiB** for the identical
graph whose copies are consumed; with the fix both are 16.00 MiB.  A real-world instance: twelve
400 MB buffers never freed and a buffer reserve creeping 4450 → 6441 MiB in a twelve-layer prefill
graph.

## Root cause (`ggml/src/ggml-alloc.c`)

* counting pass (~L737): for every `ggml_impl_is_view(node) && node->op != GGML_OP_NONE` →
  `ggml_gallocr_hash_get(galloc, node->view_src)->n_views += 1`.
* free pass (~L805): the matching `view_src_hn->n_views -= 1` sits inside the block that runs when a
  **parent is consumed and released** (`if (p_hn->n_children == 0 && p_hn->n_views == 0)`) — so a view
  node that is nobody's `src` never gives its contribution back.
* `n_views` is a plain `int`; the count is not driven negative, it is merely permanently too high,
  which is enough to block both the free and the reuse.

## The repro

`../wip/arch-independent-memory/repro/ggml-alloc-unused-view.c` — self-contained, CPU backend, public
`ggml_gallocr_*` API, 12 layers x 4 MiB.  `argv[1]=1` builds the idiom (dangling side-effect copies),
`argv[1]=0` consumes the copies.  Build/run recipe in the file header.

| build | idiom (`1`) | control (`0`) |
|---|---|---|
| unadulterated master `9cf3bf256` | **56.00 MiB** | 16.00 MiB |
| master + this patch | **16.00 MiB** | 16.00 MiB |

With `GGML_ALLOCATOR_DEBUG` (and `-DGGML_ALLOCATOR_MAX_TENSORS=8192` for this graph) the free pass
shows it directly: 1 free before vs 13 after, and `view_src ...: 4 views` never returning to 0.

## Change

After the counting pass (child counts are final there), fold the contribution of every view node with
`n_children == 0` that is not a graph output: decrement its view source's `n_views`, and free the view
source when it reaches 0 children / 0 views and is allocated.  Graph outputs are exempt because they
are never freed and may alias the view source, which has to stay allocated for the application to read
it.

Soundness: a node with `n_children == 0` at that point has no consumers at all, so folding cannot
shorten the life of anything still needed; views that *are* used as a `src` (for example the
destination view of the cpy) keep their own contribution, so the view source still cannot be reused
before the last operation touching it has run.  (Folding is preferred over "release every childless
node in the free pass", which would decrement twice for a view whose contribution was already given
back through its consumer.)

## Validation

* master, allocator tests: `test-alloc` all cases PASSED; `test-batch-alloc` 198 assertions,
  0 failures, 0 exceptions, 0 skipped.
* hardware validation (3x R9700 gfx1201, ctx 204800, ub 2048, `-ctk/-ctv q8_0`, `-sm tensor`), patch
  carried in a full llama.cpp tree: same-seed generated text **byte-identical** to the unpatched tree
  on Qwen3.5-4B Q8_0, Qwen3.8-27B Q8_0 and Qwen3.8-Flash-Next IQ4_XS; compute/host reserves unchanged
  (1800.33/840.34, 1920.33/880.34, 3251.39/63.69 MiB); adaptive-MTP acceptance unchanged (0.61616);
  `test-backend-ops -o {VIEW,CONT,CPY,DUP,CONCAT}` OK on CPU and ROCm0.
* these models do not use the idiom, hence the unchanged numbers — the change is a latent-trap fix.

## Notes for filing

* Review-sensitive points: the output exemption, and that the fold runs exactly once before any
  allocation for the graph.
* Suggested extra test: a `tests/test-alloc.c` case in the idiom's shape asserting that the buffer
  size stays at one layer's worth (the repro is written to be adaptable).
* No behavior change for graphs that do not use the idiom.
* Base: applies clean to `9cf3bf256` (master at preparation time) and to `9113cc188`.
