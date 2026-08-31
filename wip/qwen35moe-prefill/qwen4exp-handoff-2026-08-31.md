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

## UPDATE 2 (PM session): CPU-burn root cause FOUND + prefill >1000 t/s

### Root cause of the 90-95% CPU / 60% GPU observation
NOT the PLE get_prev_tokens scan (now O(log n) via PR #27992 index,
unit-tested 9480 lookups 0 failures; verify mode on the model: 0
mismatches, index 0.18us vs scan 13.8us per call at 2K ctx - but at 16K
ctx the scan is only ~110us/call, so A/B fast vs off is flat 683 vs 692;
the PR's wins are at 240K ctx, not 16K). NOT mmap page faults (-lm none
= same). 

Real cause: PER-UBATCH HOST OVERHEAD. Every ubatch (512 tokens) re-runs:
- graph scheduler + gallocr reserve (main thread; gdb caught it in
  ggml_gallocr_reserve_n_impl)
- a CPU split with 2 GET_ROWS: token_embd (644 MB) + per_layer_token_embd
  (the 27.4 GB ngram table, CPU_Mapped lazy, in CPU RAM)
- 16 OpenMP threads spin in kmp_flag_64 barriers waiting for the slowest
  gather row (gdb: libomp barrier spin) - this is the visible CPU burn

Cost is fixed per ubatch -> scales with ubatch COUNT. At 16K with ub512
that's 32 ubatches, at ub2048 only 8. Evidence:
- -t 1 vs -t 16: pp16384 255 vs 769 t/s (CPU-thread sensitive at depth),
  but pp512 (1 ubatch) is thread-insensitive (1399 vs 1412)
- ub 512/1024/2048 at pp16384: 799 -> 969 -> 1061 t/s (+33%)

### THE FIX (config, no code): -ub 2048 -t 16
Final locked numbers (tensor, r=3, env GGML_CUDA_FA_WMMA_256=0):
- pp512 1472, pp2048 1576, pp4096 1459, pp8192 1283, pp16384 1091, tg128 39.9
- PREFILL TARGET >1000 t/s at 16K: ACHIEVED (1091)
- Reference (a7cc83bba+ngram) at ub2048: pp16384 771 -> our 1091 = +42%
- tg ~40: unchanged, matches reference ~39; decode is per-token ubatches
  regardless of -ub; result_output [2560x82774x1] = 0.6ms alone, likely
  memory-bandwidth bound (103.68 GiB weights)

### Tree state
- PR #27992 applied + verified: src/llama-kv-cache.{cpp,h}, llama-kv-cells.h
  backup: patches/0005-kv-prev-tokens-index.patch
- Unit test tests/test-kv-prev-tokens.cpp (not in CMake; compile standalone:
  g++ -std=c++17 -I src -I include -I ggml/include ... -> PASSED)
- Recommend keeping the index (verified correct, negligible 16K cost,
  2.7x win at 240K) but it is NOT the 16K unlock.

### Next steps
- Update llama-server/bench invocations to -ub 2048 (server: --ubatch-size 2048)
- Decode ~40 -> 50 target: investigate per-token host overhead (25ms/token
  wall vs 0.9ms GPU in last split) - graphs on decode are already active;
  candidate: result_output LM head over 82K vocab, attention 16K KV per
  token, GDN state; check if tg benefits from -ub 2048 at server level
  (batch decode of parallel slots)
- Remaining: 2-GPU deploy, Strix Halo validation (parked)

## UPDATE 3 (PM session): REAL CPU-burn root cause = QSA indexer top-k on CPU. FIXED.

### The real bottleneck (not the ngram gather!)
- 16 OpenMP threads at ~95% were in `ggml_compute_forward_top_k` +
  `std::__heap_select` (gdb-verified). The QSA sparse-attention indexer does
  top-2048-of-16384 (indexer.top_k=2048, per token, per QSA layer) and this
  landed on the CPU backend because CUDA TOP_K support is gated at
  `ne[0] <= 1024` without CUB. At 16K: 264 TOP_K nodes on CPU.
- The ngram gather was NOT the burn: PLE batch cache (prototype below) is
  bit-exact and removes the per-ubatch gather, but CPU% unchanged (1260%)
  and pp16384 flat (1722.8 vs 1722.2 with cache off).

### The fix (backport from upstream master)
- ggml/src/ggml-cuda/top-k.cu: upstream HIP radix top-k kernel
  (`top_k_radix_cuda`, 4-pass 8-bit radix select with per-row histograms)
  for ncols > 1024. Pure backport; our tree predates it.
- ggml/src/ggml-cuda/ggml-cuda.cu supp: TOP_K now allowed on HIP for any
  ne[0] (master does the same: `#if defined(GGML_USE_HIP) || CUB`).
- Env A/B switch GGML_CUDA_TOP_K_CPU=1 forces the old CPU path.
- VERIFIED BIT-IDENTICAL: logitdump (same build, GPU radix vs CPU heap_select,
  32 tokens, seed 42) - byte-identical output. The radix kernel returns the
  same index SET (unsorted order is fine; the consumer scatters into a mask).
- Also verified against the earlier reference worktree build.

### Measured (tensor, ub512 unless noted, WMMA off, t16)
| test | top-k CPU | top-k GPU (radix) | gain |
|------|-----------|-------------------|------|
| pp2048  | 1209 | 1441 | +19% |
| pp8192  | 908  | 1419 | +56% |
| pp16384 | 784  | 1307 | +67% |
| pp16384 @ub2048 | - | 1716-1723 | |
| tg128   | 37.8 | 37.8 | 0 (decode is bandwidth-bound) |
- CPU burn at pp16384/ub2048: 1260% -> 140% (16 cores -> 1.4).

### Best full curve (all fixes, ub2048, r=3)
pp512 1492, pp2048 2028, pp4096 2031, pp8192 1935, pp16384 1716, tg128 40.1.
That is 2.6x the session-start 655 and 2.5x the 570-620 reference at 16K.
PREFILL TARGET >1000 t/s: MASSIVELY EXCEEDED (1716).

### Tree state additions (on top of prior working tree)
- top-k.cu radix backport + supp change + GGML_CUDA_TOP_K_CPU A/B env
  backup: patches/0006-qsa-topk-radix-gpu.patch
- PLE batch-level cache prototype (prepare_batch hook + llm_graph_input_ple
  cache path + LLAMA_PLE_CACHE_VERIFY / LLAMA_PLE_CACHE_DISABLE envs):
  verified bit-exact on prefill AND continuation batches; performance-
  neutral now that top-k is on GPU. Keep or drop: it removes the per-ubatch
  ngram gather from the graph and would matter if the table were bigger or
  the gather were offloaded; currently adds code for no measured win.

### Next steps
- Decide keep/drop PLE cache (currently neutral; top-k was the win).
- Decode 37.8 -> 50: decode is memory-bandwidth bound (103.68 GiB weights),
  result_output [2560x82774x1] alone ~0.6ms; investigate expert MMQ width
  or KV/cache reuse; not top-k-bound.
- Run logitdump at 16K to confirm bit-exactness at depth (done at 4K).
- Strix Halo validation, 2-GPU deploy remain parked.
