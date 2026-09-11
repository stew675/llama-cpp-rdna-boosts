# upstream/ — standalone upstream-PR candidates

This directory holds candidates to be **upstreamed to ggml-org/llama.cpp as pull requests**: clean,
self-contained changes that could stand on their own.  Two kinds:

* **already inside a delivery patch** (`patches/` 0001-0014 and `beta/qwen4exp/...`) — here as a
  reference and as the starting point for a PR branch;
* **campaign wins staged in the beta Block 15** (`beta/block-15-campaign-wins/`) — here because they are upstream-applicable in
  their own right (if upstream takes one, Block 0015 must drop it on the next regeneration).

They are NOT additional deliverables: applying one of these on top of the full delivery
would double-apply the hunks.  Treat each as:

1. a reference for the exact upstreamable change (with its own README/notes), and
2. the starting point for a PR branch against upstream master (re-created from the
   file's current state in the fork, not blindly applied).

| file | what it is | where it lives in the delivery | status |
|---|---|---|---|---|
| `UPSTREAM-PR-fa-kv-split-width.patch` + `.md` | ggml-cuda: make the flash-attention KV-split (`parallel_blocks`) heuristic query-width independent for small batches — decode and every speculative verify width then reduce identically, so greedy output no longer changes with the MTP draft length | **delivery block 00** (structural and architecture fixes); block 00 also carries the Vulkan masked-V fixes | prepared 2026-09-10; validated on gfx1201 against the delivery (issue #25: `--spec-draft-n-max 2` == `4` on 2/3-GPU, adaptive == both, plain decode byte-identical, acceptance gate holds); applies clean to `9113cc188`; not filed |
| `UPSTREAM-PR-ggml-sched-probe.patch` + `.md` | a debug probe for the ggml backend scheduler (per-op fallback / sync tracking), core-ggml only | inside `beta/qwen4exp/qwen4exp-support.patch` (the sched-fallback-sync hunk) | prepared, not filed; re-verified 2026-09-10 to still apply clean to current master (`9cf3bf256`) |
| `UPSTREAM-PR-ggml-alloc-unused-view.patch` + `.md` | ggml-alloc: release view sources whose views are never consumed (a real leak; repro included) | **staged in beta Block 15** (`../beta/block-15-campaign-wins/`, win W4); A/B via `../beta/block-15-campaign-wins/ab/w4-revert.patch` | prepared 2026-09-10, applies clean to `9cf3bf256`, repro + `test-alloc`/`test-batch-alloc` verified **on master**; upstream-drop check 2026-09-10: still absent upstream; not filed |
| `UPSTREAM-PR-kv-cache-keys-only.patch` + `.md` | llama: keys-only KV caches -- `llama_kv_cache` gains `v_enabled` (no V tensor, no V-side op); the qwen4exp indexer store passes `false` (its V is dead: the indexer scores keys) | **staged in beta Block 15** (`../beta/block-15-campaign-wins/`, win W3) | prepared 2026-09-10; verified on **master** (`9cf3bf256`, CPU): applies clean, compiles, indexer KV **72.00 → 24.00 MiB** at ctx 8192 (K 24 / V 48 → K 24 / no V), same-seed text byte-identical, `test-alloc`/`test-batch-alloc` 0 failures; not filed |
| `UPSTREAM-PR-attn-k-null-mask-guard.patch` + `.md` | llama: `llm_graph_input_attn_k` tolerates an absent kq mask (guard the fill like the sibling `attn_kv` class does; let `can_reuse_impl` accept a null mask) | **staged in beta Block 15** (`../beta/block-15-campaign-wins/`, part of win W2) | prepared 2026-09-10; verified on **master** (`9cf3bf256`, CPU): applies clean, compiles, same-seed text byte-identical; **hardening only** -- no reachable null-mask path on master today (every construction site builds a mask); not filed |

When filing, re-create the branch from upstream master and re-run the file's own Validation section —
do not blindly apply the copy, and never apply one of these on top of the full delivery (that would
double-apply the hunks).


When filing, re-create the branch from upstream master and re-run the file's own Validation section —
do not blindly apply the copy, and never apply one of these on top of the full delivery (that would
double-apply the hunks).

If a hunk here is ever accepted upstream, it should be dropped from the delivery patch
set on the next regeneration (the delivery then carries only the fork-local remainder).

**Backlog:** the FA KV-split width fix (above) is written up and validated against the delivery.  A
second candidate — the masked-V / freed-cell kernel fixes (HIP `fattn-tile.cuh`, HIP
`fattn-mma-f16.cuh`, Vulkan `flash_attn.comp`/`flash_attn_cm1.comp`, today folded into block 14) — is
already validated on gfx1151 + Vulkan but has **not** been re-cut against unadulterated master yet
(the HIP halves sit on the fork's native-bf16/WMMA FA path, so the upstream form needs a fresh port).
The four older candidates (the sched probe, the allocator view-release, the keys-only indexer cache,
and the `attn_k` null-mask guard) are written up above, each with its own `.md` evidence and an
explicit "what was not validated" section.  Re-run that section on the PR branch before filing; a hunk
accepted upstream is dropped from the delivery at the next regeneration.
