# HANDOFF: qwen4exp investigation session 2026-08-31

Status: MAJOR PROGRESS + ENVIRONMENT ROOT-CAUSE FOUND. Ready to compact.

## TL;DR

The "regression" was mostly ENVIRONMENTAL, not code:

1. **GPU runtime power-management cycling** (root cause of 604->364 swings): all
   3 R9700s were suspend/resuming every 1-5 minutes all day (527 SMU resume
   cycles since 06:00; `dmesg`/`journalctl` shows "SMU is resuming...").
   FIXED: `echo on | sudo tee /sys/bus/pci/devices/0000:{03,06,09}:00.0/power/control`
   (was "auto"). Zero cycles since. This explains why the SAME binary/command
   gave 604 t/s at 10:17 and 364 t/s at 11:33, and why the clean reference
   build (a7cc83bba+ngram) also dropped 685->317.
2. **/tmp is tmpfs (RAM-backed) and had 74 GB of stale logitdump .bin files**
   (my 16K logitdump runs wrote 16 GB bins; plus earlier session leftovers).
   Freed to 0.3 GB -> free RAM went 8 GB -> 81 GB, swap drained. The 103 GiB
   model + 28.8 GB ngram table + tmpfs bloat was starving the page cache.
3. **New bottleneck: HOST CPU** (user observed 90-95% CPU, ~60% GPU). This is
   qwen4exp's per-ubatch PLE predecessor-token lookup:
   `llama_kv_cache::get_prev_tokens()` (src/llama-kv-cache.cpp:1829) scans ALL
   used KV cells to resolve ngram predecessors - O(n_kv) per ubatch, 32x at
   16K prefill, and per-token in decode. **PR #27992 fixes exactly this**
   (kv-cache (seq,pos) cell index -> O(log n); 2.7x decode at 240k ctx on
   2xL40S, 1.27x at 16K). Related: #27941 (qwen4exp follow-up fixes incl.
   65535 gridDim.y abort at n_kv 262144), #27977.

## Current measured state (BEST, after env fix)

Build: ~/llama.cpp rdna-boosts @ 4fa92f0ae + block-11 REVERTED + ngram patches
0001-0007 + MoE fusion work, GGML_CUDA_FA_WMMA_256=0, tensor split, 3 GPUs,
ub=512:

| test | t/s | note |
|---|---|---|
| pp512 | 974.9 | was 346 before all fixes |
| pp2048 | 860.8 | |
| pp4096 | 797.6 | |
| pp8192 | 723.3 | |
| pp16384 | 654.9 | reference was 570-620 - BEATEN |
| tg128 | 39.5 | user target ~50 |

Reference build (master a7cc83bba + ngram patches, tensor split, ub 512,
clean env): pp16384 = 685 (was measured before env degradation; the ref
curves 343@512 -> 685@16384 look odd, likely first-test warmup/page-cache
effects). Both builds now agree the env is the dominant variable.

## What the fixes did (cumulative, pp512)

- baseline BAD (block11 on, no ngram, WMMA on): 346
- + block 11 revert (prefill CUDA graphs ON): 419  (+21%)
- + ngram patches: 460                          (+10%)
- + GGML_CUDA_FA_WMMA_256=0: 502                (+9%)
- + env cleanup (runtime PM + /tmp): 975        (+94% more!)

Fusion (try_fuse incl. MoE gate MMQ) is a WIN on qwen4exp (all-fusion-off
328 vs fusion-on 464 at 16K). GDN chunked neutral-positive. BF16 KV better
than F16 (334). Hybrid all-reduce better than internal (387).

## Environment fixes to keep (documented for the box)

1. Runtime PM: `echo on | sudo tee /sys/bus/pci/devices/0000:03:00.0/power/control`
   (and 06, 09). Needs root; may not survive reboot - re-apply after boot.
   Alternatively a udev rule. Verify: `journalctl -k | grep "SMU is resuming"`
   should be quiet during idle.
2. /tmp is tmpfs: keep logitdump output elsewhere (write bins to /home or
   delete after use). 16K-token logitdump bins are 16 GB each.
3. Swap was 100% full before cleanup; now drained.

## Remaining work / next steps

1. **CPU-bound prefill/decode**: implement or track PR #27992 (get_prev_tokens
   index). The host is at 90-95% while GPUs idle at 40% - this is now THE
   limiter. For 1000+ t/s prefill target (we are at 975 @512, 655 @16K) and
   ~50 t/s decode (39.5 now), fixing get_prev_tokens is the highest-value item.
   Note the PR is OPEN upstream, not merged - local implementation would be
   a fork of it (author disclosed AI use, states verify plumbing would be
   removed; test suite passed).
2. Optionally test the ngram-managed path (`--lazy-buffer-size 4G`) now that
   env is stable - earlier logitdump showed managed == mmap bit-identical and
   equal speed; re-verify tok/s is flat with the new env.
3. Qwen3.8-Flash-Next multi-GPU spread (Q4_K_XL = mixed Q4_K/Q5_K/Q6_K
   blocks) - the fused gate+up covers those types; re-validate on this model
   now that numbers are stable (pp512/pp16384 + bit-exact logitdump).
4. Strix Halo (RDNA3.5) validation to open the RDNA4 gate on the fused MMQ.
5. Consider `-lm none` (normal file loading, avoid mmap) for llama-server
   runs as user suggested - couples with managed ngram work.

## Tree state (dirty, uncommitted by design)

- ~/llama.cpp rdna-boosts @ 4fa92f0ae; 25 files changed +613/-207
- block 11 (skip graphs on multi-token prefill) REVERTED in working tree
  (backup: /tmp/ggml-cuda.cu.block11.bak)
- ngram patches 0001-0007 applied (from wip/managed-ngrams/patches/)
- MoE fusion work (mmq/mmvq) present
- Reference worktree: /home/stew675/q4exp-ref (a7cc83bba + ngram patches,
  build-ref/bin/llama-bench) - keep for A/B
- NOT committed to ~/llama.cpp (needs user approval; rdna-boosts branch
  commits are the norm per user preference)

## Key files/refs

- src/llama-kv-cache.cpp:1829 get_prev_tokens (THE CPU hot spot)
- src/models/qwen4exp.cpp llm_graph_input_ple::set_input (~990) calls it
- ggml/src/ggml-cuda/ggml-cuda.cu:3427 try_fuse (fusion arm ~4071)
- PR #27992 (get_prev_tokens index), #27941 (qwen4exp fixes), #27977
- wip/qwen35moe-prefill/plan-qwen4exp.md (tracking doc; update status)
- logs: /tmp/q4exp_final_state.log (best numbers), /tmp/q4exp_*.log
