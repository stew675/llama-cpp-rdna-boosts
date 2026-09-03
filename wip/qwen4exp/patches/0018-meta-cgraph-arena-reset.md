# Patch 0018: fix meta backend cgraph_main dangling pointer after arena reset

llama.cpp commit a61a7d4b9, on top of 236002e4a.
Apply: `git apply patches/0018-meta-cgraph-arena-reset.patch`

## The bug

llama-server (and the embedded-server llama-cli) crashed with SIGSEGV on the
FIRST completion request, at the same address in
`ggml_backend_meta_graph_compute` for every repro (no backtrace in the log =
the user's "prompt-time crash"). llama-bench and the bseq harnesses never hit
it; only callers with a scheduler eval callback (the server's progress
callback) did - the eval callback makes the scheduler feed the Meta backend a
stream of `ggml_graph_view` slices (uid 0), so the meta wrapper rebuilds its
subgraphs on EVERY eval.

Root cause: the per-device `cgraph_main` entries were re-allocated only for
`[0, n_subgraphs)` whenever the graph arena was recreated (max_nnodes or
n_subgraphs growth). If the graph that triggered the reset had FEWER subgraphs
than a historical max, entries `[n_subgraphs, max_subgraphs)` kept pointers
into the freed arena. A later graph whose subgraph count fell in that orphaned
range skipped the re-allocation (no growth vs max_subgraphs) and filled
through the dangling pointers. Confirmed by instrumentation: `dangling=1`
(one entry outside the live arena, zeroed struct, size=0).

## The fix

On every arena reset, re-allocate ALL `[0, max_subgraphs)` entries (the arena
was already sized for max_subgraphs per device, so this is within budget).

## Verification

- llama-server base config at ctx 16384 and ctx 102400: loads + listens +
  generates (39.4 / 39.0 t/s decode, was crash-on-first-request).
- User's full config (mlock, cache-ram, ctx-checkpoints, reasoning-budget,
  cpu pinning, NCCL_PROXY_CPUSET) at ctx 102400: works, 39.3 t/s.
- llama-bench pp512 1537.6 / tg128 45.70 unchanged (fill content identical).
- bseq_val K=2 coherence streams unchanged (198 42750 367..., 369 9542 11...).
- The user's earlier cores (04:43, 04:46, same crash) confirm this was the
  crash all along, at the base config too (not the extra flags).
