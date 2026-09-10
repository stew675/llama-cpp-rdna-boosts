# upstream/ — standalone upstream-PR candidates

This directory holds candidates to be **upstreamed to ggml-org/llama.cpp as pull requests**: clean,
self-contained changes that could stand on their own.  Two kinds:

* **already inside a delivery patch** (`patches/` 0001-0014 and `beta/qwen4exp/...`) — here as a
  reference and as the starting point for a PR branch;
* **campaign wins destined for the beta Block 0015** — here because they are upstream-applicable in
  their own right (if upstream takes one, Block 0015 must drop it on the next regeneration).

They are NOT additional deliverables: applying one of these on top of the full delivery
would double-apply the hunks.  Treat each as:

1. a reference for the exact upstreamable change (with its own README/notes), and
2. the starting point for a PR branch against upstream master (re-created from the
   file's current state in the fork, not blindly applied).

| file | what it is | where it lives in the delivery | status |
|---|---|---|---|---|
| `UPSTREAM-PR-ggml-sched-probe.patch` + `.md` | a debug probe for the ggml backend scheduler (per-op fallback / sync tracking), core-ggml only | inside `beta/qwen4exp/qwen4exp-support.patch` (the sched-fallback-sync hunk) | prepared, not filed; re-verified 2026-09-10 to still apply clean to current master (`9cf3bf256`) |
| `UPSTREAM-PR-ggml-alloc-unused-view.patch` + `.md` | ggml-alloc: release view sources whose views are never consumed (a real leak; repro included) | destined for the beta Block 0015 patch (see `../beta/block-15-campaign-wins/`) | prepared 2026-09-10, applies clean to `9cf3bf256`, repro + `test-alloc`/`test-batch-alloc` verified **on master**; not filed |

When filing, re-create the branch from upstream master and re-run the file's own Validation section —
do not blindly apply the copy, and never apply one of these on top of the full delivery (that would
double-apply the hunks).

If a hunk here is ever accepted upstream, it should be dropped from the delivery patch
set on the next regeneration (the delivery then carries only the fork-local remainder).
