# HANDOVER + IMPLEMENTATION PLAN — qwen4exp (Qwen3.8-Flash-Next) QSA memory reduction

Date: 2026-09-10 · Session: VRAM investigation + keys-only indexer + ubatch study
Status: **WIP — nothing here is part of the delivery.** `wip/` items must not be folded into
`patches/` without the maintainer's explicit go-ahead and the block-14 amendment protocol.

---

## 0. Read-me-first / how to use this file

This is the single entry point for a fresh session. It contains:
- the environment + exact commands (section 1),
- verified findings that must NOT be re-derived (section 2),
- Deliverable A: the **keys-only QSA indexer cache** patch — already written, validated, and
  parked, awaiting packaging (section 3),
- Deliverable B: the **implementation plan** for the memory reductions (sections 4–6),
- the validation protocol the repo requires (section 7),
- packaging/rollout steps (section 8), open questions (section 9),
- appendices with measured numbers, artifact inventory, and a code-reference index.

Supporting artifacts (scripts, logs) are in this directory:
`wip/qwen4exp/qsa-memory/tools/` and `wip/qwen4exp/qsa-memory/logs/`.
The keys-only patch + its own validation record are in
`wip/qwen4exp/keys-only-indexer/`.

Repo rules that apply: read `AGENTS.md` (delivery repo) before touching anything; the fork at
`~/llama.cpp` is disposable/canonical (rdna-boosts @ `e2380eb67` this session) and must not be
pushed anywhere; all new work is env-gated and lives under `wip/` until promoted.

---

## 1. Environment & reproduction

**Hardware/model (this session):**
- 3× AMD Radeon AI PRO R9700 (gfx1201, 32624 MiB each; GPUs 0,1,2). GPU3 is a 4 GiB device, not used.
- Model: `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (split 3 parts; 87.24 GiB GGUF, IQ4_XS 4.25 bpw, 176.94 B params, arch `qwen4exp`).
- MTP draft (test only): `/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.79 GB).
- qwen35 control: `/home/stew675/Qwen3.5-4B-Q8_0.gguf` (4.16 GiB, 32 blocks, 1-in-4 dense hybrid).
- Host: Ryzen 9 9950X3D2, 184 GiB RAM. ROCm 7.14 at `/opt/rocm-7.14-gfx1201`.

**Builds / binaries:**
- Baseline (canonical rdna-boosts): `~/llama.cpp/build-rocm/bin/{llama-server,llama-cli,llama-bench}`.
- Keys-only build (this session): `/tmp/bin-keysonly/` — volatile; rebuild from the patch (section 3).
- Build command (incremental, from `~/llama.cpp`):
  ```bash
  export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
  export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH
  cmake --build build-rocm --target llama-server llama-cli llama-bench -j 16
  ```
- Runtime env for all runs:
  `HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`

**The user's production server command (verbatim intent):**
```bash
HIP_VISIBLE_DEVICES=0,1,2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_FA_WMMA_256=0 \
  ~/llama.cpp/build-rocm/bin/llama-server --model <IQ4_XS part1> --alias Qwen3.8-Flash-Next-IQ4_XS \
  --fit off --top-k 20 --port 8033 --threads 15 --parallel 1 --top-p 0.95 --min-p 0.001 \
  --verbosity 4 --host 0.0.0.0 --cpu-strict 1 --predict 98304 --cpu-range 1-15 --threads-http 4 \
  --load-mode mlock --cache-ram 16384 --ctx-size 204800 --flash-attn auto --temperature 0.2 \
  --batch-size 2048 --ubatch-size 2048 --n-gpu-layers all --no-kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 --ctx-checkpoints 64 --cache-idle-slots \
  --reasoning-budget 65536 --reasoning-preserve --checkpoint-min-step 4096 --lazy-mode off \
  --split-mode tensor --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

**Gotchas (do not rediscover the hard way):**
- `pkill -f "llama-server ..."` inside a bash-tool command **matches its own shell** and kills the
  call. Kill by PID captured at launch, or let `timeout` reap the server (all scripts here do this).
- Ports 8037–8039 are held by a `llama-slot-prox` supervisor. Use 809x for test servers.
- GPUs must be left idle between runs; never run benches concurrently.
- `--ubatch-size 2048` + f32 KV at ctx 204800 **does not fit** on 3×32 GiB (capacity abort, expected).
- `--ubatch-size 1536` (batch 2048) is a trap: ragged 1536+512 graphs make it slower *and* bigger
  than ub1024. Prefer ub that divides `--batch-size`.
- Decode/`tg` is ~ubatch-independent (50.7 t/s); only prefill (pp) cares.
- PLE tensor `per_layer_token_embd.weight` (27,465 MiB) is host/disk-resident (`lazy read enabled`);
  GPU weights are ≈62.6 GiB (model buffer 21,360.59 MiB × 3). The user's "~63 GB" is correct.

---

## 2. Verified findings (do not re-derive)

### 2.1 VRAM ledger at the user's exact config (ctx 204800, ub 2048, q8_0 KV, baseline build)

rocm-smi per-GPU used: 29.38 / 29.65 / 29.73 GiB; process total 88.58 GiB of 95.6 GiB.

| component | size | notes |
|---|---|---|
| weights | ~21.36 GiB/GPU (20,860 MiB) | `Meta() model buffer = 21360.59 MiB` per GPU; ~62.6 GiB total; PLE 27,465 MiB host-side |
| dense KV | 2550 MiB total | 12 layers (indices 3 mod 4), K+V q8_0 1275 each; ~850 MiB/GPU (4 layers each) |
| QSA indexer KV | 956.25 MiB total | K 318.75 + V 637.50 — **the V half is dead** (section 3) |
| recurrent state | 112.57 MiB | 1 cell, 48 layers, ctx-independent |
| **graph/compute buffer** | **6690.40 MiB/GPU** | per-GPU replica (proved by 2-GPU probe); `Meta() compute buffer` |
| AR pipeline + driver | ~0.05 GiB/GPU | 32 MiB copy-engine + 1 MiB kernel staging + ~60 MB driver |

Proof the compute buffer is per-GPU: the 2-GPU probe fit the weights (31,321.59 MiB/GPU) and then
aborted allocating `2652.07 MiB on device 0: cudaMalloc failed` (the compute buffer). The 1-GPU
probe tried a single 61,222 MiB model buffer → OOM. So weights are partitioned ~1/n per GPU,
graph/compute buffers are replicated per GPU.

### 2.2 Prefill speed model — why big ubatch is fast, and what it costs in VRAM

- Prefill ubatch graphs have a **constant node count ~7,923 regardless of ubatch** (log: `graph nodes
  = 7923 (with bs=2048/1536), 5146 (with bs=1)`). Executing a prefill graph costs a fixed
  ≈0.118 s ≈ 7,923 kernels × ~15 µs host dispatch. CUDA-graph replay is **deliberately disabled for
  multi-token graphs** in this fork (`ggml-cuda.cu` `ggml_backend_cuda_graph_compute`, comment:
  prefill shapes vary with KV length so capture never amortizes; measured pp512 +6.7% with graphs
  off). Decode (ne[1]==1) DOES replay.
- Fit from single-graph benches: per-token work w ≈ 0.2837 ms, fixed per graph F ≈ 0.1182 s
  (pp512: 1945.9 t/s; pp512@ub512: 1940.2 t/s — ubatch-independent for one graph; pp2048: 2928.6).
- Consequence: pp ≈ T / (work + 0.118 s × ceil(T/ub) per batch-2048 decode call).

| ubatch | pp20480 t/s (baseline, r3) | pp2048 t/s | compute buf/GPU @ctx204800 | pp512 |
|---|---|---|---|---|
| 2048 | 2509.6 ± 7.0 | 2928.6 | 6690 MiB | 1945.9 |
| 1536 | 2144.9 | 2414.5 | 5018 MiB | — |
| 1024 | 2207.7 | — | 3347 MiB | — |
| 512 | 1691.3 | 1859.3 | 1725 MiB | 1940.2 |

tg256 = 50.69 / 50.70 / 50.68 for ub 2048 / 1024 / 512 (no effect). ub1536 loses on both axes
(ragged graphs). The compute-buffer reserve decomposes roughly as ≈1.3 MiB/GPU per ubatch-token of
activation/MoE workspace (ctx-independent; 2652 MiB at ctx 8192/ub2048) + ≈2.0 MiB/GPU per
ubatch-token of ctx-proportional attention/QSA staging (≈4.0 GiB/GPU at ctx 204800).

### 2.3 KV geometry vs qwen35 (control: Qwen3.5-4B, same flags)

| | qwen4exp @204800 | qwen35-4B @204800 | qwen35-4B @8192 |
|---|---|---|---|
| dense KV | 2550 MiB, 12/48 layers, 2 KV heads ×256 | 3400 MiB, 8/32 layers, 4 KV heads ×256 | 136 MiB |
| indexer KV | 956 MiB (12 layers) | — | — |
| recurrent | 112.57 MiB | 50.25 MiB | 50.25 MiB |
| compute | 6690+1263 MiB | 1800+840 MiB | 288+72 MiB |
| per-token dense KV | 12.8 KB | 17.0 KB | — |

KV is confined to the 1-in-4 dense layers in both — no per-SSM-layer KV. The qwen4exp "2× qwen35
overhead" impression was the graph/compute buffer (bigger model → bigger per-op workspaces), not KV.

### 2.4 `tg` / decode is not affected by any of this

Decode graphs replay via CUDA graphs; ubatch/mask/score changes here target prefill only.

---

## 3. Deliverable A — keys-only QSA indexer cache — DONE, VALIDATED, PENDING PACKAGING

**What it is:** the qwen4exp QSA indexer store (`llama_memory_hybrid_idx::mem_idx`) is built as a
generic `llama_kv_cache`, which always allocates a V tensor (`has_v = !is_mla`, sized from the model
value_length = 256 dims). The graph only ever issues **K-side ops** against it (`cpy_k`/`get_k`/
`ggml_get_rows` in `build_qsa_store_k` / `build_qsa_top_k`); the sparse attention reads values from
the *dense* layer cache. So the 256-dim V store (272 B/token/layer at q8_0) is allocated but never
written or read — and it is replicated per GPU.

**Patch:** `wip/qwen4exp/keys-only-indexer/0001-keys-only-qsa-indexer-cache.patch` (62 lines, 3 files,
applies to rdna-boosts `e2380eb67`):
- `src/llama-kv-cache.{h,cpp}`: new ctor param `bool v_enabled = true`; `has_v = !is_mla && v_enabled`.
  Null-V caches are already fully supported in-tree (MLA precedent): `size_v_bytes`, stream copies,
  `state_read`/`state_write` all null-guard V. V-side ops (`get_v`/`cpy_v`/`type_v`) must not be
  issued against a keys-only cache.
- `src/llama-memory-hybrid-idx.cpp`: pass `v_enabled = false` for the indexer store.

**Results (fork build `e2380eb67` + patch):**
- indexer cache: `size = 318.75 MiB (204800 cells, 12 layers), K (q8_0): 318.75 MiB, V (q8_0): 0.00 MiB`
  (was 956.25 MiB).
- total VRAM @ctx204800/ub2048: **88.58 → 86.70 GiB** (−1.9 GiB; the V was triplicated per GPU).
- perf parity: pp20480 2497.4 vs 2509.6 (−0.5%, noise), tg256 50.65 vs 50.69.
- coherence: llama-cli same-seed (seed 42, temp 0) **byte-identical** at q8_0 and f16.
- Extended matrix (all clean): bf16 KV @ ctx204800; f32 @ ctx16384 (f32 @204800 is a capacity OOM,
  expected); `--parallel 2` (n_stream=2); `--kv-unified` + parallel 2; prompt-cache/context-checkpoint
  save+restore round-trips; MTP draft (`draft-mtp`, acceptance 0.741, 43/58, 20/20 calls);
  f16 A/B byte-identical. Details in `wip/qwen4exp/keys-only-indexer/README.md`.

**Not yet exercised:** 4096-token checkpoint spacing (the long-gen run hit `<|im_end|>` at 609
tokens), `--parallel > 2`, and non-text/mrope paths (server had no `--mmproj`).

**Packaging procedure (when the maintainer says go):**
1. Apply the patch to a canonical fork rebuilt at the fork point per `AGENTS.md`
   (fresh clone @ `9113cc188` + `scripts/apply-all.sh`, or the current `~/llama.cpp` worktree).
2. Fold into **block 14** (it owns `llama-memory-hybrid-idx.*`; `llama-kv-cache.*` is also
   block-14-amended) or keep as its own block if preferred — decide with the maintainer.
3. `scripts/make-patches.sh` regeneration; verify the white-space-clean `apply-all.sh` run and the
   clean-apply build.
4. Run the coherence gate + the extended matrix above on the regenerated fork.
5. Update `patches/README.md` (block-14 notes), `AGENTS.md`, `MANIFESTS.md`, and add a dated
   `WORKLOG.md` entry.

---

## 4. Deliverable B — L1: derive the QSA top-k additive (causal mask) in-kernel

### 4.1 Goal

Remove the materialized `[n_kv × n_tps]` KQ-mask tensor that is passed as the `additive` operand of
`ggml_indexer_top_k` during prefill, replacing it with compact position inputs and computing the
visibility in the kernel. Expected reclaim: the mask at ctx 204800 / ub 2048 is
`204800 × 2048 × 2 B` (F16, because `cparams.flash_attn` → `llama-graph.cpp:39,816` picks F16)
≈ **0.78 GiB per GPU ≈ 2.4 GiB box**; plus removal of the host-side `set_input_kq_mask` O(n_kv)
build. This is the *lowest-risk* item: for text-only, 1-D positions the tensor holds exactly
`{0.0f, -INFINITY}` (see `llama-kv-cache.cpp:1555 set_input_kq_mask_impl`, `mask_keep = 0.0f`,
`mask_drop = llama_cast<T>(-INFINITY)`), i.e. a pure position/sequence test.

### 4.2 Mechanism and exact code map

- `src/models/qwen4exp.cpp:964 build_qsa_top_k` — the QSA top-k graph:
  - `:990` decides `blk_bias` (true iff the mask is `[n_kv, n_tps, 1, n_stream]`, causal, no alibi).
  - `:1021` `qsa->bias` = F32 `[blk_bias ? n_blocks : n_kv, n_tps, n_stream]`.
  - `:1105` fused `ggml_indexer_score` (decode-only: gated on `n_tokens == 1` + float keys +
    `q->ne[1] <= 8`; kernel `ggml/src/ggml-cuda/indexer-score.cu`, out shape
    `[n_blocks, n_tps, n_stream]` post-bias). **Not usable for prefill** (see 5.4).
  - `:1141-1160` per-op prefill chain: `mul_mat` → `relu` → head-sum → `+ inp->bias`.
  - `:1174` `additive = blk_bias ? kq_mask : inp->bias`.
  - `:1176` `ggml_indexer_top_k(ctx0, score, inp->cell_blk, additive, width)`.
- `ggml/src/ggml.c:5662 ggml_indexer_top_k` — op; asserts `additive` is F16 or F32 and
  `cell_blk->ne[0] == additive->ne[0]`.
- `ggml/src/ggml-cuda/indexer-topk.cu` — GPU kernel:
  - `:26 indexer_topk_float_to_ordered` (total-order key), `:49 indexer_topk_value` =
    `score[...] + (float) additive[c + t*n_kv + s*n_kv*n_tps]`,
  - `:146-278` radix-select; rank-boundary ties are resolved **by cell order** (deterministic;
    the comment at `:139-146` records that earlier tie handling varied run-to-run).
- `ggml/src/ggml-cpu/ops.cpp:12550 ggml_compute_forward_indexer_topk` — **CPU reference exists**
  (useful for differential tests; `INDEXER_SCORE` has no CPU forward).
- `src/llama-memory-hybrid-idx.cpp:442 set_input_qsa` — builds `cell_blk`/`blk_cells`/`blk_pos`/
  `bias`; it already iterates `cells.pos_get(j)` and `cells.seq_has(...)`; `:469 dst_bias`,
  `:722 cur_blk_bias[b] ∈ {-INFINITY, 0.0f, 1e9f}`, `:744 cur_bias` (non-blk path).
- `src/llama-kv-cache.cpp:1555 set_input_kq_mask_impl` — the templated mask builder
  (`template<typename T, bool causal, bool swa, bool is_2d, bool alibi>`); the 2-D/mrope branch uses
  `pos, then ext.y, then ext.x` ordering (same total order the QSA pooling uses).
- `src/llama-batch.h:25 is_pos_2d()` — gate for the 1-D-only fast path.

### 4.3 Design (recommended)

Keep the existing op/ABI for the fallback path; add a **positional variant** (new op, or an
`additive == nullptr` + extra srcs form of the same op, or a bool op param):

- Instead of `additive` `[n_kv, n_tps, n_stream]` F16/F32, pass:
  - `cell_state` : I32 `[n_kv, n_stream]` — per (expanded cell column, stream): the cell's position,
    or a sentinel (`-1`) for cells that are not visible to that stream (not in its sequence). This
    reproduces `cells.seq_has(j, seq_id) ? pos : -inf`.
  - `q_pos` : I32 `[n_tps, n_stream]` — the query positions.
- Kernel value: `const float m = (cell_state[c,s] >= 0 && cell_state[c,s] <= q_pos[t,s]) ? 0.0f : -INFINITY;`
  then `return sc + m;` — **must keep the add**.
- Build both arrays in `llama_memory_hybrid_idx::set_input_qsa` (it already has the cells and the
  ubatch), only when the fast-path gate holds; otherwise upload the existing mask as today.
- Gate: `blk_bias && cparams.causal_attn && !hparams.use_alibi && !ubatch->is_pos_2d() && n_swa == 0`.
  All of these are already implied for `blk_bias`, except `is_pos_2d` (text vs image). Keep the
  tensor path for 2-D positions (mrope `ext` ordering) — the user's server never loads `--mmproj`,
  so text-only is the operative case, but the fallback must stay.
- Env gate for the experiment: e.g. `GGML_CUDA_QSA_TOPK_POS=0` to force the old path, default on
  once validated (follow the fork's convention of env-gates with =0 opt-out).

### 4.4 Exactness requirements (must hold for "bit-identical")

1. The derived f32 visibility values must equal the f32 the current kernel reads from the tensor:
   `0.0f` for visible, `llama_cast<T>(-INFINITY)` widened to f32 for invisible (both exact in
   F16/F32). Do not "optimize" the visible case to skip the add: `sc + 0.0f` also maps `-0.0f` →
   `+0.0f`, and the ordered-float key distinguishes `-0.0` from `+0.0`.
2. Sequence membership must be reproduced exactly (cells of another sequence are `-inf`).
3. The top-k's tie/order semantics (`float_to_ordered` key, rank-boundary ties by cell order) must
   be untouched — this change does not alter selection logic, so it should be provable by
   construction, but verify (section 7).
4. 2-D/mrope, alibi, SWA, non-causal → old tensor path.

### 4.5 Implementation steps

1. **L0 (measure first, 1–2 h):** attribute the 6690 MiB/GPU reserve. Options: `GGML_SCHED_DEBUG=1/2`
   (`ggml/src/ggml-backend.cpp:1891`), or a temporary env-gated dump of the largest tensors in the
   gallocr (`ggml/src/ggml-alloc.c`), run with ctx 204800/ub 2048. Record the per-tensor sizes
   (expected candidates: score `[n_blocks, 4, n_tps]` f32 ≈1.56 GiB; `kq_mask`
   `[n_kv, n_tps]` f16 ≈0.78 GiB; `bias` `[n_blocks, n_tps]` f32 ≈0.39 GiB; `kq_mask_all`
   fill copy ≈0.78 GiB). **Do not implement before this step** — the plan's numbers are estimates.
2. Add `cell_state`/`q_pos` inputs in `set_input_qsa` (+ graph-input struct fields in
   `llm_graph_input_qsa`, `src/models/qwen4exp.cpp:907-961`), gated.
3. Add the kernel path in `indexer-topk.cu` (and the CPU reference in `ggml-cpu/ops.cpp:12550` for
   parity testing if it is exercised).
4. Wire `build_qsa_top_k` to pass them instead of `kq_mask` when the gate holds.
5. Validate (section 7), measure the reserve delta, then consider L1b/L2.

### 4.6 Risks / fallbacks

- Host cost of building `cell_state` is O(n_kv) like today's mask build — no worse.
- If the mask tensor is needed elsewhere in the same graph (the attention mask), leave it alone;
  only the top-k additive operand is replaced.
- If the derived path can't be proven bit-identical, keep it env-gated off by default.

---

## 5. Deliverable C — L2: reduce the QSA prefill intermediates (conditional, higher risk)

### 5.1 Score tensor (the big prize)

The per-op prefill chain materializes `score` `[n_blocks, n_idx_h, n_tps, n_stream]` f32 (with
n_blocks = ceil(n_kv/r), n_idx_h = 4, n_tps = ubatch tokens) = **1.56 GiB/GPU** at ctx204800/ub2048,
then relu + head-sum to `[n_blocks, n_tps]`.

### 5.2 Option L2a — extend the fused score op to prefill (blocked)

`ggml_indexer_score` already outputs the post-bias per-block score `[n_blocks, n_tps, n_stream]` and
is byte-identical by design, but it is (a) gated to `n_tokens == 1` (`qwen4exp.cpp:1070`), (b) requires float indexer keys
(`idx_key_float`: F32/BF16/F16 — the user runs q8_0, which takes the per-op chain), and (c) the
kernel asserts `q->ne[1] <= 8` where `q->ne[1] = n_idx_h*n_tps` — so prefill cannot be enabled by a
gate flip; it needs a kernel redesign (per-token parallelism). Switching indexer keys to f16 costs
+0.28 GiB/GPU and only helps decode (already fused). **Deferred; do not treat as a quick win.**

### 5.3 Option L2b — KV-chunk the per-op score chain

Process `n_blocks` in slices, maintaining a running top-k. Numerics:
- Per-block scores are independent dot products → slice-wise values are bit-identical **provided the
  matmul does not pick a different reduction/split path for the changed shape** (verify at tensor
  level).
- The **selection** is the risk: rank-boundary ties (mass exact `0.0f` after relu, plus `-inf`) are
  resolved by cell order by the radix select. A running merge must preserve exactly that order
  (e.g. sequential merge in ascending cell order with the same `(value, cell)` order). A naive
  "top-k per slice, merge by value" will differ.
- This is the highest-effort/highest-risk item; do it only if L0 shows the score is still the top
  consumer after L1, and only with the bit-exact differential gate.

### 5.4 Option L1b — derive the per-block bias (`bias`, `[n_blocks × n_tps]` f32, ≈0.39 GiB/GPU)

`set_input_qsa` fills it with `{-INFINITY, 0.0f, 1e9f}` from block/sequence/tail bookkeeping
(`llama-memory-hybrid-idx.cpp:722`). It is index-derived, so it can in principle be derived in-kernel
from compact per-block/query inputs, but the semantics (tail sentinel `1e9f`, seq membership,
`bid_idx >= tail_start`) need care. Medium risk; do after L1.

### 5.5 Option L1c — avoid the full-size `kq_mask_all` fill copy

`build_attn_qsa` does `kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY)` (a full-size copy,
≈0.78 GiB/GPU f16) before carving the sparse mask. Investigate whether the fill can write only the
needed region / be folded. Low-risk if the graph allows it; check L0 output first.

---

## 6. Deliverable D (optional) — dispatch-overlap / node-count reduction

The 0.118 s/graph dispatch tax is the reason smaller ubatch is slower. Hiding it via overlapping
successive ubatch graphs needs double-buffered workspaces (which negates the memory saving) or
removing whatever serializes inside a graph (profile first: per-node launch vs multi-GPU AR sync).
CUDA-graph replay for prefill is closed (shape instability, measured in block 01). More fusion
reduces nodes and intermediates but is a large effort. Listed for completeness; not planned.

---

## 7. Validation protocol (repo bar: bit-identical where claimed)

1. **Build/apply check:** canonical fork at the fork point + all blocks; `scripts/apply-all.sh`
   whitespace-clean; incremental build of `llama-server`/`llama-cli`/`llama-bench`.
2. **Coherence gate (mandatory):** same-seed llama-cli comparison (seed 42, temp 0) against the
   baseline build; output must be IDENTICAL modulo the t/s status line (scripts:
   `tools/f16-ab.sh`; `/tmp/ab-*.txt` pattern).
3. **Bit-exact prefill differential (for L1/L2):** the same-seed text check is *not* sufficient —
   a changed top-k set can alter one KV row without changing the first visible tokens. Compare
   baseline vs modified on a long prompt at several depths:
   - selected `top_k` indices (add a temporary debug print / dump the op output),
   - resulting KV state (dump the cache tensors or `llama_state` blobs) bit-exactly,
   - logits at the end of prefill (`llama-cli` with a logits dump / `--perplexity` on a fixed corpus).
4. **Perf/VRAM protocol:** one server load per config, read the `sched_reserve` compute-buffer line
   + `rocm-smi --showpids` total; after any decode/fusion change also run
   `benchmarks/mtp-adaptive-methodology.md` (MTP acceptance > ~0.45, MTP ≥ plain at depth 3).
   Never run parallel benches; GPUs idle between runs.
5. **Regression set:** the keys-only matrix in `wip/qwen4exp/keys-only-indexer/README.md` (bf16/f32,
   parallel 2, kv-unified, checkpoints, MTP, f16 A/B) — reuse for any change touching the QSA
   memory/cache path.

---

## 8. Packaging / rollout

- **Keys-only (Deliverable A):** section 3 procedure. Keep the patch + README current; if the
  maintainer wants it now, fold into block 14 and regenerate. Otherwise it stays a `wip/` patch.
- **L1/L1b/L1c/L2:** develop on a scratch worktree/branch of `~/llama.cpp` (the fork is
  disposable), env-gated, with the differential harness. Only after validation + maintainer approval
  does it become a block-14 amendment (or its own block), following the same regeneration +
  documentation flow.
- Never push anything from `~/llama.cpp` (see `AGENTS.md` pushing policy).

---

## 9. Open questions for the maintainer

1. Priority/order: L1 (safe, ~0.8 GiB/GPU) → L1b/L1c (~0.4/0.8) → L2 (1.56, risky)? Or is the
   `--ubatch-size 1024` + keys-only combination (≈77 GiB @204800) "good enough" for now?
2. Should the keys-only patch be packaged into block 14 now, or kept as a `wip/` patch until the
   L1/L2 work lands (one regeneration instead of two)?
3. Indexer key dtype: keep q8_0 (per-op chain; smallest) or move the indexer K store to f16/bf16
   (enables the fused decode score + derived pool; +0.28 GiB/GPU)? Decode-fused is already the
   default with float keys; prefill does not benefit.
4. Is there a target VRAM budget (e.g. "full 262144 ctx must fit") that sets the required reclaim?

---

## Appendix A — measured numbers (this session)

VRAM / ubatch (baseline build unless noted):
```
ctx 204800 ub 2048 : compute 6690.40 MiB/GPU, box 88.58 GiB   pp20480 2509.6  tg256 50.69
ctx 204800 ub 1536 : compute 5018.45 MiB/GPU                    pp20480 2144.9  (ragged; worse than 1024)
ctx 204800 ub 1024 : compute 3346.50 MiB/GPU, box 78.78 GiB    pp20480 2207.7  tg256 50.70
ctx 204800 ub  512 : compute 1724.56 MiB/GPU, box 74.03 GiB    pp20480 1691.3  tg256 50.68
ctx   8192 ub 2048 : compute 2652.07 MiB/GPU                   (ctx-independent base)
keys-only @204800 ub2048: indexer 956.25→318.75 MiB, box 88.58→86.70 GiB; pp 2497.4 / tg 50.65
qwen35-4B @204800: dense KV 3400 MiB, rec 50.25, compute 1800+840, box 15.48 GiB
pp512 ub2048 1945.9 | pp512 ub512 1940.2 | pp2048 ub2048 2928.6 | pp2048 ub512 1859.3 | pp2048 ub1536 2414.5
fit: w ≈ 0.2837 ms/token, F ≈ 0.1182 s/graph
```

## Appendix B — artifact inventory (`wip/qwen4exp/qsa-memory/`)

```
tools/smoke.sh           generic server smoke (start, poll, curl, kill by PID)
tools/driver1.sh         keys-only functional matrix T1–T5 (ub1536/bf16/f32/parallel2/kv-unified)
tools/t6.sh              prompt-cache + checkpoint round-trip
tools/t7.sh              ctx-checkpoints + long generation
tools/t8.sh              MTP draft smoke
tools/f16-ab.sh          f16 coherence A/B (baseline vs keys-only)
tools/f32small.sh        f32 KV at capacity-safe ctx
tools/bench-parity.sh    llama-bench pp/tg parity (keys-only)
tools/ppcheck.sh         pp512/pp2048 × ub2048/512 (dispatch model validation)
tools/ub1536bench.sh     pp2048/pp20480 at ub1536
logs/                    the runs above + baseline/ctx8k/ub512/ub1024 server loads, 1-GPU/2-GPU
                         probes, qwen35 controls, benchmark tables, A/B text outputs
```
Volatile-but-useful: `/tmp/bin-keysonly/` (keys-only binaries; rebuild from the patch),
`/tmp/baseline-bin/` (launcher-only copy — incomplete, do not rely on it).

## Appendix C — code reference index

| location | what |
|---|---|
| `src/models/qwen4exp.cpp:881` | `build_qsa_store_k` (indexer K store only) |
| `src/models/qwen4exp.cpp:964` | `build_qsa_top_k` (blk_bias, bias, score, top_k) |
| `src/models/qwen4exp.cpp:1105` | fused `ggml_indexer_score` (gate at `:1070`: decode only, float keys, `n_idx_h<=8`) |
| `src/models/qwen4exp.cpp:1174,1176` | `additive`, `ggml_indexer_top_k` |
| `src/models/qwen4exp.cpp:1184` | `build_attn_qsa` (kq_mask_all fill, sparse FA) |
| `ggml/src/ggml.c:5662` | `ggml_indexer_top_k` op + asserts |
| `ggml/src/ggml-cuda/indexer-topk.cu:26,49,146` | ordered key, value add, radix select/tie order |
| `ggml/src/ggml-cuda/indexer-score.cu:16-22` | fused score op I/O (Q/B/dst shapes, `q->ne[1]<=8`) |
| `ggml/src/ggml-cpu/ops.cpp:12550` | CPU reference for INDEXER_TOPK |
| `src/llama-kv-cache.cpp:1555` | `set_input_kq_mask_impl` (mask keep/drop, 2-D mrope order) |
| `src/llama-memory-hybrid-idx.cpp:60` | indexer `llama_kv_cache` creation (keys-only patch site) |
| `src/llama-memory-hybrid-idx.cpp:442` | `set_input_qsa` (cell/blk pos, bias fill) |
| `src/llama-memory-hybrid-idx.cpp:722,744` | block bias vs per-cell bias values |
| `ggml/src/ggml-cuda/ggml-cuda.cu:5690+` | graph compute; prefill CUDA-graph skip (dispatch tax) |
| `ggml/src/ggml-backend.cpp:1891` | `GGML_SCHED_DEBUG` |
| `wip/qwen4exp/keys-only-indexer/` | Deliverable A patch + validation record |
