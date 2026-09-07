# upstream/ — standalone upstream-PR candidates

This directory holds **copies** of selected patches that are already included in the
delivery patch sets (`patches/` 0001-0013 and `beta/qwen4exp/qwen4exp-support.patch`),
presented separately because they are clean, self-contained changes that could be
**upstreamed to ggml-org/llama.cpp as pull requests**.

They are NOT additional deliverables: applying one of these on top of the full delivery
would double-apply the hunks.  Treat each as:

1. a reference for the exact upstreamable change (with its own README/notes), and
2. the starting point for a PR branch against upstream master (re-created from the
   file's current state in the fork, not blindly applied).

| file | what it is | where it lives in the delivery |
|---|---|---|
| `UPSTREAM-PR-ggml-sched-probe.patch` + `.md` | a debug probe for the ggml backend scheduler (per-op fallback / sync tracking), core-ggml only | inside `beta/qwen4exp/qwen4exp-support.patch` (the sched-fallback-sync hunk) |

If a hunk here is ever accepted upstream, it should be dropped from the delivery patch
set on the next regeneration (the delivery then carries only the fork-local remainder).
