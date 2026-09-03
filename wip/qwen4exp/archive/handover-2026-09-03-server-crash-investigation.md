# HANDOVER 2026-09-03: qwen4exp server crash + perf re-baseline (post-recreation)

Context: the qwen4exp branch was recreated on the new master (9cffdcc80 +
13 re-based blocks + 15 qwen4exp commits + managed-ngrams + rebase fixes) and
validated (single-seq decode byte-identical, K=2 coherence, arch tests all
green). The user reports **slowness + crashes in llama-server** after the
re-application. Investigation in progress; this doc = the checkpoint.

## State

- `~/llama.cpp` `qwen4exp` @ `236002e4a`, clean, NOT pushed. History:
  `9cffdcc80` (new master) + 13 blocks + 15 qwen4exp commits (cherry-picked
  1:1 from `~/qwen4exp.cpp` @ `9ed77c905`) + managed-ngrams commit + 2
  rebase-fix commits. Full record: boost repo
  `wip/qwen4exp/rebase-2026-09-02-9cffdcc80.md` (pushed @ `0166d36`).
- Canonical source repo `~/qwen4exp.cpp` kept as the local remote
  `qwen4exp-src` in ~/llama.cpp (fetchable for reference).
- Boost repo main @ `0166d36`.

## Rebase validation already done (all green, see the rebase doc)

- test-lazy-reader passed; test-llama-archs full matrix 607 OK / 0 fail
  (qwen4exp MoE: GPU NMSE 9.7e-14, CPU 0.00, Meta tensor-split 1.01e-13,
  ROUNDTRIP OK = the managed reader).
- Real model 3x R9700 tensor split: K=1 single-seq = `198 42750 367 367...`
  = byte-matches the pre-rebase reference; K=2 seq0==seq1; longer-prompt K=2
  == K=1 references; managed-ngrams `--lazy-buffer-size` run == mmap run
  byte-identical.
- Known follow-up: K=2 multi-seq with the short 3-token prompt "The answer"
  drifts from its K=1 single-seq stream at decode step 3 (self-consistent,
  seq0==seq1; suspect upstream 36b101543 block-position keying vs the
  multi-seq path). NOT a blocker; single-seq untouched.

## llama-server investigation (in progress)

### Reproductions that WORK (current tree)
- Base config at ctx 16384 and 65536: loads, listens, generates (38.8 t/s at
  ctx 16K, ~22 t/s at ctx 102400 with temperature 0 = 32 tokens/1.6s).
- ctx 102400 base config: loads + listens + answers prompts. GPUs end at
  ~31.1-31.4/32.6 GB each = ~96% VRAM (very tight).

### The user's exact command (see below) = different failure modes observed
- Sometimes the main process sticks at RSS ~2MB / Threads 1 BEFORE the
  "load_model: initializing" log line, while a worker thread (second PID)
  loads ~3.4GB RSS then stops growing - no further log output, no crash, no
  exit. GPU VRAM shows 33.27 GB used (ABOVE the 32.6 GB capacity) - stale
  allocations from a previous -9-killed server were NOT released (check the
  GPU state with rocm-smi after killing servers; may need to wait or the
  driver holds the memory).
- The user reports a prompt-time crash with NO backtrace (their env includes
  NCCL_PROXY_CPUSET, mlock, reasoning flags, ctx-checkpoints). Not yet
  reproduced exactly.

### The user's command (the target to make work)
```
HIP_VISIBLE_DEVICES=0,1,2 NCCL_PROXY_CPUSET=8..15 GGML_CUDA_DISABLE_GRAPHS=0 \
llama-server --model <UD-Q4_K_XL> --alias Qwen3.8-Flash-Next-Q4_K_XL \
--fit false --top-k 20 --port 8033 --threads 8 --parallel 1 --top-p 0.95 \
--min-p 0.001 --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --cpu-range 0-7 \
--predict 98304 --threads-http 4 --load-mode mlock --cache-ram 16384 \
--ctx-size 102400 --flash-attn auto --temperature 1.0 --batch-size 2048 \
--ubatch-size 2048 --n-gpu-layers all --no-kv-unified --cache-type-k bf16 \
--cache-type-v bf16 --ctx-checkpoints 64 --cache-idle-slots \
--reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 \
--repeat-penalty 1.0 --presence-penalty 0.0 --split-mode tensor
```

### Failure-mode hypotheses (not yet confirmed)
1. **VRAM pressure**: ctx 102400 base leaves ~2-4% VRAM free per GPU; the
   extra features (mlock, reasoning budget, ctx-checkpoints, ubatch 2048)
   push past the limit -> silent OOM-kill or a driver-side abort = the "no
   backtrace" crash. The 3x R9700 = 32.6 GB each; the UD-Q4_K_XL = 103 GB +
   the 102400 bf16 KV + the compute buffers are already at 96%.
2. **The model-load hang** (RSS stuck, no log): possibly the comm/NCCL init
   with NCCL_PROXY_CPUSET + --cpu-strict/--cpu-range pinning interacting with
   the boost block-12 hybrid all-reduce, or waiting on GPU memory held by a
   dead process. Verify with a clean GPU state + no NCCL_PROXY_CPUSET.
3. **ctx-checkpoints / reasoning-budget / cache-ram**: newer server features;
   any could interact badly with the hybrid model or the recreated branch.

### Next steps (the user's suggested order)
1. **Re-establish the llama-bench baseline first** (the decode campaign
   numbers must survive the rebase):
   - tg128 (3x R9700 tensor, ngl 99, ub 2048, -p 512 -n 128 -r 3, env
     GGML_CUDA_FA_WMMA_256=0 LLAMA_QSA_SPARSE_FA=1): expect ~45.6 t/s
     (the pre-rebase decode = 45.56-45.63; single-seq tokens already proven
     byte-identical, so the bench should match).
   - pp512 ~1500 t/s; optionally pp8192 (the prefill thread target).
   - If the bench = matches, the decode/perf work is intact and the server
     issues are config/feature-level, not the kernel numerics.
2. **Then debug the server setup** (bisect the user's flag set):
   - Start from the WORKING base (ctx 102400 + split tensor + bf16 KV +
     no-kv-unified) and add the user's flags one at a time: mlock,
     cache-ram, ctx-checkpoints, reasoning-budget/preserve, ubatch 2048,
     NCCL_PROXY_CPUSET/cpu-pinning.
   - Confirm the GPU state is clean between runs (rocm-smi VRAM ~0 after
     killing servers; wait for the driver to release).
   - When reproducing, use `setsid ... < /dev/null > log 2>&1 &` and monitor
     the PID's RSS + the log; the server often prints nothing while stuck,
     so watch /proc/<pid>/task counts + rocm-smi too.

## Environment / process notes for the next session
- Model: /models/Qwen3.8/Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf (4 shards, 103.68 GiB, IQ4_NL PLE ngram in shard).
- Machine: 3x R9700 gfx1201, 184 GB RAM, 9950X3D2 (16c/32t), ROCm
  /opt/rocm-7.14-gfx1201. Full rebuild ~5-10 min (build-rocm, -j 16).
- Stale-harness lesson: /tmp harness binaries (bseq_val etc.) must be
  REBUILT whenever llama_model_params changes (it grew with n_lazy_buf_size;
  stale binaries misread params and the model load appeared to crash with
  "n_layer_all=0 / vocab_only garbage"). Rebuilt 2026-09-03; keep them in
  sync.
- Harness binaries in /tmp (rebuilt against the current lib): bseq_val
  (coherence), bseq (throughput), logitdump_d (has --lazy-buffer-size now).
- K=1 single-seq decode gate: logitdump_d -ub 1 == the pre-rebase stream
  (byte-identical tokens verified via bseq_val: 198 42750 367 367...).
- Llama-server run/bench discipline: kill servers with
  `pkill -9 -f "llama-server --model"` (avoid plain -f llama-server which
  kills the invoking shell too), wait ~5 s, verify rocm-smi VRAM ~ 0 before
  the next load.

## Open threads (unchanged backlog)
- The "The answer" 3-token multi-seq drift (see above).
- The prefill thread (pp8192), ML-Kernel/gpudh review vs TP-V1, v_shifted
  probe, splitter->F32, meta import gates, multi-device sync in
  meta_tp_test, GDN state fold (decode ~1.5-3%).

## UPDATE 2026-09-03 (session): baseline re-established - the COLD-CACHE HUMP

STATE: llama.cpp qwen4exp @ 236002e4a, clean, build current (incremental
make only relinked the server/cli tools; core libs already at HEAD). No
stale llama processes; GPUs 0-2 clean (~60 MiB used). Manual DPM applied
(tune_r9700.sh current config = -85 mV / 265 W, was -70 mV / 250 W in the
decode-era notes). Machine idle (load ~1.6, firefox only).

THE HUMP (NEW LESSON - do not get caught again): on this model (104 GiB,
lazily mmapped, ~27 GiB/device in CPU-mapped host buffers), a fresh perf
run with a cold page cache pays DISK page-ins on the first evals, and the
hit is per-eval and DETERMINISTIC (tight error bars do NOT mean the number
is right). Measured on 2026-09-03 with ~28 GB of the file still on disk:
- pp512 = 351.33 +- 0.36 t/s, tg128 = 41.38 +- 1.76 t/s (both misleadingly
  tight) = the cold-cache artifact, NOT a regression.
- After warming the 4 shards into the page cache (dd read, ~19 s; added
  ~28 GB to Cached): pp512 = 1552.58 +- 7.53, pp2048 = 2002.93 +- 5.92,
  pp8192 = 2023.55 +- 1.01, tg128 = 45.73 +- 2.18 (one load, -r 3).
BASELINE VERDICT: tg128 45.73 ~= the 45.56-45.63 pre-rebase decode ref and
pp512 1552 ~= the ~1500 ref -> the decode/perf work survived the rebase.
pp8192 at 2024 = the thread-1 prefill target (~2000) is intact. (The
1248.54 pp512 recorded in decode-session-1 was the flag-on build era; the
noflag build measures ~1500-1550.)

GATE FOR ANY FUTURE PERF RUN: warm the model file first (dd/cat the 4
shards to /dev/null) or treat the first bench as warmup. This almost
certainly explains the "slowness" half of the original llama-server report:
a fresh server process on a cold cache degrades exactly like that.

NEXT: the llama-server work - the bench-vs-server disjoint. First
discriminator planned: llama-bench at -c 16384 / -c 102400 (bf16 KV to
match the server) vs the server's own numbers, to separate a context-length
tax in the model/memory path from a server-specific (threads/pinning/
mlock/NCCL/checkpoint) effect. The handover's own base-config numbers hint
at a ctx tax already: ~38.8 t/s at ctx 16K vs ~22 t/s at ctx 102400 on the
server; if llama-bench shows the same ctx dependence, the "disjoint" is the
KV-length-proportional decode cost, common to both tools.

## UPDATE 2026-09-03 (session 2): TWO ROOT-CAUSED BUGS FIXED - the crash and the garbage

llama.cpp qwen4exp now @ 1f5d431f7 (a61a7d4b9 + 1f5d431f7 on 236002e4a), not
pushed. Patches 0018 + 0019 + .md in wip/qwen4exp/patches/. Both fixes
verified end-to-end with the user's FULL config at ctx 102400 (mlock +
cache-ram + ctx-checkpoints 64 + reasoning-budget + cpu pinning +
NCCL_PROXY_CPUSET): loads, listens, REASONS COHERENTLY, 39.3 t/s decode.

BUG 1 (0018, ggml-backend-meta.cpp): the server crash - no-backtrace SIGSEGV
in ggml_backend_meta_graph_compute on the FIRST completion, base config too
(the user's 04:43/04:46 cores = the same crash, same address +0x654b2).
Trigger: the server sets a scheduler eval callback (progress) -> the meta
backend receives ggml_graph_view slices (uid 0) -> rebuilds subgraphs every
eval. On an arena reset the per-device cgraph_main entries were re-allocated
only for [0, n_subgraphs); a later graph with fewer subgraphs than a
historical max reused the dangling entries [n_subgraphs, max_subgraphs)
without re-allocating (instrumented: dangling=1, zeroed struct). Fix: allocate
ALL [0, max_subgraphs) on every arena reset (within the existing budget).

BUG 2 (0019, ggml-cuda/fattn-qsa.cu): bf16 KV + LLAMA_QSA_SPARSE_FA=1 =
GARBAGE decode (model emits "/" x n at any temperature, any prompt; raw and
templated). f16+sparse and bf16+dense are both coherent. The BF16 branch of
flash_attn_qsa never staged the per-cell mask M_smem (only the F16/Q8_0
gather did) but its score pass reads it - uninitialized smem corrupted every
attention score. The branch was dead code until true bf16 KV landed (bf16 was
previously silently downgraded to f16, so no gate ever exercised it). Fix:
stage M_smem in the BF16 gather too. NOTE: the user's env does not set
LLAMA_QSA_SPARSE_FA (default dense = coherent anyway); with 0019 both paths
are correct - sparse is the faster qwen4exp path, enable it.

THE BENCH-VS-SERVER DISJOINT (resolved): llama-bench tg128 measures an EMPTY
context at n_ctx = n_prompt + n_gen = 128 (n_prompt=0 for the tg test!), so
21.9 ms/token there was never representative of serving. Real decode:
server/bseq at any ctx capacity 128..102400 with a short real KV = 26.6 ms
(bseq) / 25.6 ms (server) - flat across capacity (bseq -c 128/640/16384/
102400 all 26.62 ms/step). The handover's "~22 t/s at ctx 102400" was a
cold-cache/first-touch artifact (see the hump lesson) - remeasured warm:
39.0 t/s at ctx 102400, ~39.4 at ctx 16K, matching bseq. The server's "extra
~2.6 GB/GPU" vs bseq at ctx 102400 = the larger ubatch-2048 prefill compute
buffer, harmless. No ctx-length decode tax exists.

GATE STATE: llama-bench pp512 1537.6-1537.9 / tg128 45.69-45.73 (unchanged);
bseq_val K=2 streams identical to the known-good references (198 42750 367...
369 9542 11...). f16-path numerics untouched by construction (both fixes are
in allocation-only / bf16-only code). VRAM at ctx 102400 ~30.9-31.1 GiB/GPU
(97-98%) - tight but works.

OPEN (noted, not chased): mixed K/V types (k=bf16/v=f16) crash at model init
with a meta split-state assert (ggml-backend-meta.cpp:537) - edge case, the
user config uses both-bf16; the "The answer" short-prompt multi-seq drift
(unchanged backlog); the prefill thread; ML-Kernel/gpudh review. Test tools
kept in /tmp: server_test.sh (base-config server smoke + timings),
srv_* probes, tdec.cpp (token-id decode helper). Cores 04:43/04:46/05:41/
05:47 all = bug 1; coredumpctl may still hold them.

## UPDATE 2026-09-03 (session 3): QSA sparse FA is now the default

llama.cpp qwen4exp @ 5cbef4c4d (patch 0020). Contract now: the QSA sparse
flash-attention path is the DEFAULT when flash attention is enabled;
LLAMA_QSA_SPARSE_FA=0 selects the dense masked FA path; -fa off uses the
manual (non-FA) attention path (the sparse branch is gated on
cparams.flash_attn). Env semantics flipped (was: unset = dense). The sparse op
now appears in every FA-on qwen4exp graph, so a CPU reference for
GGML_OP_FLASH_ATTN_QSA was added (ops.cpp) for -ngl 0 / CPU graph builds.
Verified: no-env llama-bench tg128 45.75 / pp512 1538.3 = the old env=1
numbers; bseq_val no-env streams == the known-good refs; env=0 dense still
coherent; -ngl 0 CPU decode gives the same stream as GPU; test-llama-archs
qwen4exp green (GPU 9.35e-14, CPU 0.00); user's full config at ctx 102400
39.4 t/s coherent with no env needed. NOTE: SPLIT_MODE_TENSOR + -fa off is a
pre-existing documented constraint (tensor split requires FA); FA-off works
with layer split.
