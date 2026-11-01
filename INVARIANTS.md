# INVARIANTS.md — the facts you must not re-derive

**What this is.**  The engineering knowledge base behind `AGENTS.md`: the hard-won facts, invariants and
"do not do that again" rules a change has to satisfy.  It was the *Critical facts (do not re-derive)*
section of `AGENTS.md` until 2026-11, which had grown to ~1300 lines and was drowning the policy layer;
the text below is that section, unchanged except for the two build facts noted at the end.

**Where the rest is.**  `AGENTS.md` = the policy layer (what is mandatory, what is in / out of scope, how a
release is cut).  `GREEDY-PURITY.md` = the purity rulebook (§ per finding).  `patches/README.md` = the
per-block reference.  `ENVIRONMENT.md` = every environment variable.  `MANIFESTS.md` = apply order and the
verification contract.  `WORKLOG.md` = the dated record of *when* and *why* each fact was learned.  This
file is the **current state** of those facts: correct one in place when it changes (with the `(rNN)` of the
change), and let `WORKLOG.md` carry the narrative.

> **Reading it.**  Every entry is one fact with the measurement behind it, tagged with the revision that
> established it (`rNN`, or a PR/issue number).  A fact that names an arm, a model or a device count is
> *specific to that arm* — quote the arm when you use the number.  If an entry and the code disagree, the
> code (and `WORKLOG.md`) wins: fix the entry.  Nothing here is normative policy; the policy is in
> `AGENTS.md`.

> **Old references.**  Older `WORKLOG.md` entries and `wip/` notes say "`AGENTS.md`'s <X> bullet" or
> "the `AGENTS.md` fact about <Y>" — those now live here.  `AGENTS.md` keeps the policy and a short
> "rules you must know before changing anything" list that points at this file.

---

- **`llama-cli` MUST be run with `--single-turn`** (plus `--no-display-prompt` for scripted output).
  Without it, the interactive chat loop blocks forever. Wrap potentially blocking commands in `timeout`.
- **Apply method:** all 16 blocks with **`git am`**; `scripts/apply-all.sh` is the tested path. Plain
  `git apply` of the concatenated series **silently drops hunks**. Every block is a normal commit.
- **Naming (V2.1):** the current set is **16 blocks, 00-15** (block 15 is the **streamed KV** — both Goals
  A and B of `TODO` #118; block 14 is the temporary `-sm relaxed-tensor` block, see `patches/README.md`;
  the release tag lives in `release.json` — read `.release` there, it is never restated here).  **15**
  carries **two opt-in placements of the same cache**, mutually exclusive by construction: the **window**
  (r178; `LLAMA_KV_RESIDENT_CELLS` — an opt-in fixed-VRAM resident window with the tail in pinned system
  RAM, one VA range, no kernel change) and the **attendance pool** (r190; `LLAMA_KV_REGION_POOL` — the
  authoritative copy in a pinned host mirror plus a compact device pool the sparse decode arm resolves its
  top-k through), plus the streaming arm policy that makes either take the sparse decode (r190).  Both are
  **default OFF** — fittability features, and on a config whose KV fits they cost the sparse-vs-dense gap.
  A run with both knobs unset keeps the plain device buffer and its OOM.  **They are placements of the KV, so
  each declines rather than fighting another placement (r197):** the pool declines on an active window and on
  `!offload`, and the window now declines on `!offload` too — with `-nkvo` the cache is pageable host RAM end to
  end, so there is nothing to window (before r197 it announced a VRAM window and then built a plain `CPU KV
  buffer`, and `is_streaming()` took qwen4exp's sparse arm off that plan).  `is_streaming()` therefore keys on
  `kv_region`, the placement actually taken, not on `kv_resident_cells`, the plan.  **15 also carries Goal C** (r193,
  `LLAMA_SPEC_DRAFT_WINDOW`): the MTP draft context's KV is a *window* instead of `-c` cells, **default ON**
  because it measures non-negative (adaptive = a quarter of the context clamped to [32768, 65536], `0` is the
  kill switch), and an embedded MTP head's slab draft reserve is derived with it (aux 3.50 -> 0.76 GiB, arena
  +2.8 GiB).  MTP only, embedded-head only; a separate MTP draft GGUF (qwen4exp/Flash-Next) keeps its full
  cache.  The ring is `find_slot`'s `n_swa` reclaim, so no allocator and no mask change — and `kq_mask_derivable`
  needs **no** exclusion, because its predicate compares each cell's *position*.
- **The MTP draft window is PER SEQUENCE, and the cache is sized per sequence when the KV is unified (r200,
  issue #264).**  Under `--kv-unified` the one draft cache holds every sequence, so its cell count is
  `(win + one batch) x n_seq_max` (capped at the context) — the r193-r199 `win + n_ubatch` shape was shared by
  all of them and the second sequence had no cell to place a batch in once the first had filled the window
  (`find_slot`'s SWA reclaim frees a cell by its OWNER's position, so an idle sequence's cells are not
  reclaimable by another), i.e. HTTP 500 `decode() failed: failed to process speculative batch` on every
  request of slot 1.  `llama_spec_draft_kv_cells()` is the one geometry function: `create_memory`, the slab's
  draft reserve estimate and `common`'s "is a window active" test all call it, so a reserve can never
  under-charge the allocation.  A per-stream cache (`--no-kv-unified`) is unchanged, `n_seq_max == 1` (every
  `llama-cli`/`llama-bench` run) measures exactly what it did, and the window is DECLINED rather than lowered
  when it cannot fit per sequence.  Gate: `scripts/gate-spec-draft-window.sh` (2 slots, both rings wrapping,
  windowed output must equal `LLAMA_SPEC_DRAFT_WINDOW=0` byte for byte).  Every `block NN`
  reference elsewhere in this file that is not one of the V2.1 numbers below is the **pre-V2.1 16-block
  numbering** and is kept as a historical handle.  The V2.1 map: **01** all-reduce, **02** adaptive MTP,
  **03** chunked GDN, **04** BF16 KV, **05** FA band/native/derived, **06** MMB, **07** k-quant,
  **08** fused kernels (fused MoE), **09** host-resident MoE (slab/cache), **10** slab + fusion dispatcher
  + async input ring + the rope/row-concat reorders + the 1-token router fold (infrastructure), **11**
  qwen4exp (model/QSA/HC/lazy reader), **12** campaign
  remainder, **13** MTP draft vocabulary subset (opt-in), **15** the streamed KV: window (Goal A) + attendance
  pool (Goal B), both opt-in.  V2.1 split r109's old block 10 in two: the
  **infrastructure stays 10**, the **qwen4exp architecture is 11** (QSA went with it by decision), so
  every pre-r109 "block 10" note about slab / dispatcher / async-input is still block 10 while a note
  about qwen4exp / QSA / HC / the lazy reader is now block 11.
- **A 1-token router fold is a width-1 fusion (r182, block 10).**  `ggml_cuda_router_topk_moe_ok` folds the
  MoE router `MUL_MAT` into the fused `topk_moe` launch only where the dispatcher would run
  `mul_mat_vec_f<float,float,1,256>` and the logits have **one row**, and the merged kernel reproduces that
  kernel's arithmetic exactly (same thread map, `float2` loads, `ggml_cuda_mad`, warp + cross-warp
  butterfly) with the last block -- threadfence + one atomic on the last `moe_red_counters()` slot -- running
  the unchanged selection.  The **prefill-logit gate at the default ubatch cannot see it** (the fold never
  fires with more than one token): judge it with `gate-prefill-logits.sh --ab "GGML_CUDA_ROUTER_TOPK=0" --ub 1`
  plus a `rocprofv3` census in the same build, and `test-backend-ops -o TOPK_MOE`.  The fold is **inert
  under any speculator** (a verify batch is wider than one token), so it is a plain-decode win
  (+1.2..+3.1 % on a 128/256-expert MoE, RDNA3/RDNA4), and it **steps aside on RDNA4 above 128 experts**
  (measured slower as one launch on both our box and the submitter's).  `GGML_CUDA_ROUTER_TOPK=0` is the
  kill switch; do not drop the logits store (the fold elides the *launch*, never the tensor, so another
  consumer of the logits keeps working).  `GREEDY-PURITY.md` §72,
  `archive/work/router-topk-fold/PROMOTED.md`.
- **Throughput is NOT a correctness signal.** Verify coherence (same-seed `llama-cli` diff, or
  `archive/work/tools/ar_kernel_unit.cpp`). Decode perf work is validated at **depth-16384**, not
  shallow `llama-bench`. **Never run parallel/background benches.**
- **A fresh-process comparison is only valid with `ROCBLAS_USE_HIPBLASLT=0`** (set it for that run, do not
  ship it).  On ROCm before the `rocm-libraries` fix (Linux 10.x included — 10.1.0's pin predates it),
  hipBLASLt picks its solution for an untuned FP32 GEMM size **per process start**, so two identical
  commands can produce different logits and a different gen t/s.  This is *not* a defect in this patch set
  (issue #67, external) but it **will** be misread as one: it was the entire residual of the 2026-10 MTP
  determinism hunt (3 logit classes in 18 fresh starts of one gemma4 MTP config; 10/10 with the env set),
  and it made every graph/capture arm look like a cause.  Attribute a per-start difference to the library
  before touching code; **never judge run-to-run determinism on a greedy text hash** — use a logit-level
  probe.  **Keep hipBLASLt enabled in the delivery** (disabling it costs up to -15 % decode; the switch is
  for reproducible runs and bisection only).  `GREEDY-PURITY.md` §44,
  `archive/work/fix-gemma-4/evidence/MEASUREMENTS-2026-10-21d.md`.
- **`--lazy-mode off` overrides a mapped load, and `-sm tensor` + `--load-mode none` uses the chunked async ring (r122, block 00).**
  An explicit `off` loads the `TENSOR_READ_LAZY` PLE/engram table into its own host buffer instead of
  serving it from the `--load-mode mmap` mapping (`auto` below 4 GiB stays mapped, so default loads are
  unchanged); `--load-mode mmap --lazy-mode off` is the "fast mapped bulk + resident PLE" combination.
  The stale `ggml_backend_dev_is_meta(dev) -> nullptr` loader bail is gone: the Meta async setter serves a
  mid-row chunk (`write_partial_row`) and the event proxy fans the 4-slot staging ring
  (`LLAMA_LOAD_STAGE_MB`, default 4 MiB) out to the simple devices, so `--load-mode none` under `-sm tensor`
  is ~2x faster (cold Flash-Next load 32.2 -> 13.6 s).  Both are **block 00**; do not reintroduce either
  workaround.  Same-seed greedy text is byte-identical across the load modes on Gemma4-E4B and Flash-Next.
- **A layer-input tap needs the HC-averaged state on qwen4exp (`build_hc_mean`, r121, TODO #94 closed).**
  `qwen4exp` runs `hc` parallel residual streams, so `res_hc` is `hc * n_embd` wide (10240 on Flash-Next)
  while `extract_layer_inp()` demands exactly `n_embd` per row: exporting the wide residual made any
  DFlash2/DSpark head abort at `GGML_ASSERT(row_floats == model.hparams.n_embd)` (llama-context.cpp:3437;
  the F1 device path asserts `t->ne[1] == n_tokens`).  The tap is `graph::build_hc_mean()` (mean over the
  streams, the same 8 lines as deepseek4/glm5-next and the head author's own `hidden_states.mean(dim=-2)`
  capture), **gated on `cparams.embeddings_layer_inp[il]`** so a graph that taps nothing (plain decode, MTP,
  QSA) gains no node — plain decode is byte-identical and the prefill logits are unchanged (KLD −0.000000).
  `target_layers` is `completed + 1` (the capture is a layer *output*), no index shift.  **Since the
  `3f03b1772` re-base the helper is upstream's `hc_stream_mean()`** (its own #30927 fix for the same
  problem, node-for-node identical); the delivery dropped its `graph::build_hc_mean()` for qwen4exp
  (deepseek4/glm5-next keep theirs).  Measured on
  Flash-Next: +40 % code / +91 % recall over plain decode at 3 GPU tensor + `-ncmoe 48`, six
  `plain == draft-dflash` text-identical arm pairs.  Index `n_layer` (the last layer's output) is **not**
  exported — it needs the pre-gather residual; no head taps the final layer.  Record:
  `archive/work/qwen4exp-dflash2/`.
- **An extracted layer input is written into F1's store from inside the graph, never kept as a graph output
  (r163, issue #240).**  `llm_graph_result::set_outputs()` emits a `GGML_OP_SET_ROWS` per extracted layer into
  the persistent device store (`cparams.lf_layer_inp_dev_store`), whose row indices are an `I32 [n_tokens]`
  graph **input** filled per ubatch — a baked view offset would be captured into a HIP graph and replay for
  the wrong ubatch.  The copy's **position in the node order is the memory win**: every arch graph records its
  layer inputs through `llm_graph_result::set_layer_inp()`, which at the next layer's recording expands that
  layer's nodes and then emits the pending copy, so the allocator can reuse the input after its last consumer.
  An arch that assigns `t_layer_inp[il]` directly is still correct but keeps the whole cost (5 x
  `n_embd x n_ubatch` f32 live to the end of the graph, 188.8 MiB at `-ub 2048` on a 27B).  The stores are
  created in `lf_layer_inp_dev_arm()` at the top of `decode()` (the context drops and re-reserves its compute
  buffers once, so the smaller layout is what the run settles on), and the wait on the draft's consumed event
  moves **before** that graph.  The host layer-input rows are reserved lazily by
  `layer_inp_host_reserve()` — F1 never read them and they were 200 MiB of pinned memory per context.  Do not
  re-add `ggml_set_output(t_layer_inp[il])` for the F1 case, do not replace the index input with a view
  offset, and do not drop the `cparams.lf_layer_inp_dev_store` term from `llm_graph_params::allow_reuse()`:
  `GGML_LF_DFLASH_DEV_GRAPH=0` is the A/B arm that restores the pre-r163 shape.  `GREEDY-PURITY.md` §66,
  `WORKLOG.md` 261028.005.
- **A host expert table's routing must be computed before the split that prunes its upload (r197, `TODO` #129
  closed).**  `ggml_backend_sched_copy_input` reads the routing ids to prune a host master's per-op upload, and
  it does that when the split's **inputs** are copied — before the split computes.  Block 09 searches for *any*
  `MUL_MAT_ID` consumer of the input inside the split (upstream only considers the split's *first* node; that
  generalisation is what lets one merged split hook the gate/up/down tables), while the split pass starts a new
  split for a host weight only once the split already has an input to reuse — so a config whose attention runs
  on the host (`-nkvo`) began its GPU split at the attention output with `n_inputs == 0`, swallowed the router
  chain and the routed op, and the readback read the previous split's memory.  That is a **delivery reachability,
  not upstream's bug** (upstream cannot get there and copies the whole table).  Two invariants: when the ids root
  (the `view_src` chain of `node->src[2]`) is a node of the split being copied, every routing-dependent shortcut
  (pruned upload, `moe_cache_gather`, `moe_cache_update`) must stand down and the general path copy the whole
  table — an out-of-range stale id aborted, an **in-range one silently selected the wrong experts**; and the
  split pass starts the split AT such a routed op (`sched_host_weight_layer`, `next_moe_layer` keeping one
  layer's routed ops together) so the pruning and the cache keep working.  Measured on one R9700, a 24-layer
  128-expert MoE at `-ncmoe 24 -nkvo`: abort -> 24.6 -> 82.7 t/s against 176 with the KV on the GPU; text
  byte-identical to the default arms on bailingmoe3/35B-A3B (± `draft-mtp`)/gpt-oss-20b/Flash-Next/dense.
  `patches/README.md` (block 09), `archive/work/kv-residency/C3-NKVO-HOST-EXPERTS.md`.
- **The decode band's routing readback is optional, and its slot map is the one thing that must never go stale
  (r203, PR #266, block 09).**  The eager hook pays a device synchronize per routed op to read `ids` back; since
  r203 a backend that implements the new iface member `moe_cache_remap` (appended **last**; NULL in meta/cpu/rpc,
  so `-sm tensor` keeps the eager path) is offered the split input first, and a kernel maps the device routing
  through a per-table expert -> slot map into `remap_dev` (`slots + e` = cold, read in place from the pinned
  master) while writing the routing into mapped pinned memory -- the residency policy then runs on it **one call
  late** and re-uploads the map only after a residency change.  Each expert is read from the same bytes, so the
  output is unchanged (byte-identical text, identical MTP acceptance, prefill KLD 0.000000); the win is **+8.4 %
  TG** on 35B-A3B `-ncmoe 40` (1 x R9700), +14.1 %/+9.2 % at 4/8 slots, inert under `-sm tensor`.  Two invariants:
  the map is rebuilt from **`slot_dirty`**, which every slot mutation must set (fill, eviction, stand-down, slab
  eviction, shrink, migration, re-arm) -- a stale map reads the wrong arena slot, and the `slots + e` cold
  encoding also depends on the slot COUNT, so a count change without a rebuild reads past the master -- and the
  lag is policy-only (a stale lag may admit the wrong expert, never change a read address).  Scope is pinned
  master, one unslice table, no identity table, band only, after `g_sized`.  `GGML_MOE_DEVREMAP=0` is the eager
  arm.  `MOE_HOST_POOL_COLD` (same fold) reads a **pool**-resident miss in place (cold id `n_res + pool slot`,
  `moe_cache_get_cold` returns the pool base): +4.8 % TG on the pool arm, and it is safe **only** because the
  eager path it rides synchronizes the backend before the hook -- a change that removes that sync must gate it.
  `ENVIRONMENT.md` §1.2 records that the r40 device-*policy* family stays removed and why this is not it;
  `GREEDY-PURITY.md` §82.
- **A cross-device copy must ask `ggml_cuda_slab_peer_copy_ok()` first (r121).**  The slab's
  peer-access grant is per-device and **best effort**: with three ~29 GiB slabs the driver refuses it for the
  second and third device (the peers have no free VRAM to build a 29 GiB peer mapping), and a
  `hipMemcpyPeerAsync` into an ungranted slab range **faults the GPU** (`HSA_STATUS_ERROR_MEMORY_FAULT` in
  `__amd_rocclr_copyBuffer`) or returns `invalid argument` — never anything catchable.  The query is
  process-wide for slab destinations: a copy between two fully granted devices still fails while a third
  slab's grants are incomplete.  When it says no, the copy stages through a grow-only pinned buffer
  (`ggml_cuda_peer_copy_staged`, D2H on the src stream, H2D on the dst stream).  Do not add a raw peer copy
  anywhere without the guard; `-sm layer` + `-ncmoe` warm-up aborted on Flash-Next before this (reproduced on
  the r120 base with no drafter).  No env knob.
- **DFlash2 under `-sm tensor` is supported (r106, TODO #78 closed):** the selector's `ggml_top_k` /
  `ggml_get_rows` over the draft logits -- which are the *target's* `output.weight` projection, placed
  vocabulary-split -- are served by a new general collective **`GGML_OP_ALL_GATHER`** (`ggml_all_gather()`)
  that `build_dflash2_selector()` emits when the draft is a >= 2-device tensor split.  On the Meta backend it
  is `MIRRORED`, its own subgraph boundary, and dropped from every per-device child graph; the collective
  copies one row (of every dim above the split axis) at a time because a slice is contiguous in its own copy
  but strided in the full tensor.  **Only a plain axis-0 split is implemented** (anything else aborts by
  design), and there is no env knob: `-sm layer` is the escape hatch.  Mirroring the *drafter* does not work
  (the offending tensor is the target's), and neither does demoting only the drafter to `-sm layer` (a
  layer-split context cannot read the target's Meta-buffer weights).  Two related fixes ship with it: the F1
  `extract_layer_inputs()` device copy cannot express a Meta buffer (host path instead; it aborted for **any**
  DFlash drafter with `dflash.target_layers`), and the **slab narrow-2** pin must yield from the arena when
  the draft's compute buffer exceeds the unmapped VA spare (a tensor-split drafter's is 1280 MiB against the
  estimator's 512 MiB tier; without it `-sm tensor` + draft + `-ncmoe` aborts in `graph_reserve`).  Rule:
  `patches/README.md` (block 12, "Facts that span blocks").
- **A quantized KV cache rotates Q/K/V, and the decode fusions for it are r187's five (block 10, PR #258).**  With
  `-ctk/-ctv q8_0` `build_attn` inserts a Hadamard rotation on Q/K (after their ROPE), on V (before the cache
  write) and on the attention output (before its projection), which had switched **every** decode fusion off on
  that arm: 726 kernel launches per token against 510 (Qwen3-30B-A3B).  `GGML_CUDA_QK_ROTATE_HADAMARD` (the
  `QK_ROTATE` pass takes Q's rotation, defers V's behind the Q chain — 1-token graphs only, and it verifies no
  node left in front reads a deferred one), `GGML_CUDA_FUSE_V_FWHT` (the fused K kernel rotates the V row with
  `ggml_cuda_fwht_warp<64, 32>` + the q8_0 group quantize map), `GGML_CUDA_FA_COMBINE_FWHT` (the combine kernel
  does the output rotation and publishes the projection's Q8_1 — **only** where the combine runs; otherwise the
  rotation runs as before and no cache entry is created), `GGML_CUDA_QK_PAIR_FWHT` (the pair, quantized form) and
  `GGML_CUDA_FUSE_ROPE_FWHT` (rope + rotation for models without Q/K norm — **the only one that runs in prefill
  too**, so it carries the prefill-logit obligation; it fires on gpt-oss on our models, not on qwen/gemma).
  All five key on the cache `SET_ROWS` being `GGML_TYPE_Q8_0`, and since **r199** also `GGML_TYPE_Q4_0` on a wave32
  device (`ggml_cuda_quantize_q4_0_group`, PR #262: +10.8 % tg128 on a `q4_0`-KV 30B-A3B, kill switch
  `GGML_CUDA_KV_FOLD_Q4_0=0`, bit-identical — a runtime quantizer select cost the `q8_0` kernel 12.8 -> 15.1 us, so
  the cache type stays the kernels' compile-time `kv` template value; `q5_0`/`iq4_nl`/`bf16` keep their launches,
  `TODO` #120), and
  they are worth **+13.4 % tg128 / +10.1 % @4096** on a 30B-A3B q8_0-KV arm (gfx1201) with **f16 KV unchanged** —
  so **a coherence or perf claim about them must name a `q8_0` (or `q4_0`) KV arm**, and multi-device arms are mandatory
  (the submitter tested one card).  `GREEDY-PURITY.md` §74, `benchmarks/2026-10-31-kv-rotation-decode.md`,
  `archive/work/kv-rotation-decode/`.  QSA's own combine passes the new `had_dst` as `nullptr` (block 11).
- **A Q8_1 publish only pays where its consumer's node still exists, and the quant decides that (r188, block 10, PR
  #259).**  The 1-token fused MoE gate/up+GLU kernel now also publishes the **shared-expert** lane's GLU output
  as Q8_1 (keyed on the shared GLU node, counters in the lane after the routed ones, nested inside
  `GGML_CUDA_MOE_GLU_Q8`'s arming so the host can never arm what the kernel does not do; kill switch
  `GGML_CUDA_MOE_SHARED_GLU_Q8=0`), so the **shared** down `MUL_MAT` skips its `quantize_q8_1`: one launch per
  layer, **+1.1 % tg128 / +0.8 % @4096** on gfx1201 and **+2.6 % / +2.3 %** on gfx1100, bit-identical (`--ub 1`
  A/B `0.000000 / 100.000 %`).  **It is inert wherever `ffn_down_shexp` is Q8_0** — block 08's `shexp_down_gate`
  epilogue already consumes the shared GLU + shared down there, which is every Unsloth **UD** quant we hold
  (measured: 839 launches/token before *and* after on `UD-Q4_K_M`, no perf delta); it fires on a conventional
  quant (LMStudio `Q4_K_M` with Q4_K/Q6_K shared tensors, `UD-IQ3_S` — and on gfx1100 `UD-IQ3_S` the upstream
  shared-pair matcher declines the lane entirely).  So judge a claim about it with a **launch census** and name
  the GGUF; a publish that extends an existing one owes `GREEDY-PURITY.md` §75's conditions.
  `benchmarks/2026-10-31-shared-glu-q8.md`, `archive/work/shared-glu-q8/`.
- **A DFlash draft reserve must measure its two graphs at their own widths (r162, issue #239).**  A
  DFlash/DFlash2 draft context alternates a KV-injection graph (target-feature rows, up to `n_ubatch`, on the
  main scheduler) and a noise-block decoder graph (at most `n_seq x block` tokens, on the alt scheduler).
  `sched_reserve()`'s generic token ubatch at `n_ubatch` measures the *decoder* shape, whose lm_head emits
  `[n_vocab, n_ubatch]` f32: 1.9 GiB at 2048 tokens and a 248k vocabulary, while no runtime token batch is
  wider than `n_seq x block`.  `graph_reserve(..., reserve_injection)` builds the measure ubatch with a
  non-null embd marker (a DFlash draft is `model.arch == LLM_ARCH_DFLASH && cparams.ctx_other != nullptr`),
  and `sched_reserve()` reserves the injection at `n_ubatch` and the decoder at
  `n_seq_max x max(dflash.block_size, n_outputs_max_per_seq)`.  The buffer keeps the larger of the two
  (issue #42: a shape left out grows the buffer mid-run), and the final layout stays at the injection graph
  because that is what the main scheduler runs.  Do not "simplify" this back to one reserve width: the
  decoder's lm_head is proportional to `n_ubatch` and the injection's feature input is not, so the two
  shapes must both be measured.  Measured: draft device compute 2444.41 -> 264.15 MiB (-2.18 GiB) at
  `-ub 2048`, host compute 40.18 -> 200.14 MiB, greedy text byte-identical (1/2 GPU, F1 on/off,
  `LLAMA_DRAFT_SCHED2=0`, a deterministic 1-slot server); `WORKLOG.md` 261028.004.
- **The draft context's second scheduler is selected per arch (`sched_alt_for`): `n_outputs` for MTP, the
  batch kind for DFlash/DFlash2 (r120, issue #208).**  MTP's two graphs differ by outputs; a DFlash2
  draft's do not (the lattice is read from `h_nextn`, so `output = !is_dflash2`), so selecting on
  `n_outputs` put the KV injection and the draft step on one scheduler *and* one `gf_res_prev` slot: 0 %
  graph reuse, a rebuild + re-allocate on every verify step (r65-r119).  DFLASH now sends the token batch
  (no `embd`, no `lf_dev_inject`) to the second scheduler and keeps the injection on the main one; every
  other arch keeps `n_outputs`, where the batch kind would *merge* MTP's ingest and step graphs.  Nothing
  else may pick a draft graph slot independently: `get_gf_res_prev()` shares the decision.  Reuse
  0 -> 97.8 %, decode +1.4 % (27B 2 GPU tensor) to +4.0 % (35B-A3B `-ncmoe 40`), text and acceptance
  identical.  Record: `archive/work/dflash2-draft-sched2/`.
- **The expert cache's routing trace is opt-in and is the policy instrument (r110, TODO #61 P1).**
  `MOE_EXPERT_TRACE=<path>` writes every `mul_mat_id`'s raw routing (`scripts/moe-trace-replay.py` scores
  policies against it offline: the shipped F-LRU, static top-f, per-layer quota, window/prefix
  pre-admission).  DECODE records come from the scheduler hook (the ids are already on the host);
  PREFILL records come from the device entry and pay one readback per op (prefill is never captured, so
  that is legal).  An IDENTITY (fully resident) table's decode routing is not traced -- it cannot miss.
  The trace is a *diagnostic* (never default-on) and its absolute hit rate is a model, not the cache:
  judge a policy by its delta.  `MOE_EXPERT_TRACE_MIB` (default 1024) caps the file.
- **Two host-tally gaps are closed and must stay closed (r110, TODO #64).**  (1) The host pool's warm
  order comes from the *device* prefill tally, folded in once at the first decode-band pass
  (`pool_rank_from_prefill_locked`): the eager `prefill_count` path never runs under the staged prefill,
  which left `pool_rerank_from_tally_locked` uncalled and warmed the pool in expert-id order.  (2) A
  fully-resident (identity) table's decode routing is tallied device-side, **sampled**
  (`MOE_EXPERT_HEATMAP_DECODE_SAMPLE`, default 64, 0 off) and weighted by the period.  The heat mirror
  is pre-allocated at sizing: the decode band is a captured graph and a `cudaMalloc` there aborts the
  capture (`operation failed due to a previous error during capture`).
- **The expert cache's *identity fast path* is an all-or-nothing cliff worth ~40 % of decode; r117 makes the
  cache prefer whole-table identity (block 09, TODO #93 closed).**  A table is identity only at **100 % of
  its own slots** ("the decode band reads the arena directly; no per-token routing readback"); one slot short
  and it takes the remap path and pays the readback exactly like a table with no arena.  Each table's arena is
  rounded up to the slab arena unit, so a budget "just enough" in aggregate landed one unit per table short
  and **every** table dropped off: measured (GSQ `-ncmoe 48 -sm tensor`, plain decode) 3 devices
  `MOE_EXPERT_CACHE_MIB=13800` → arena 40845.1 of 40925.0 MiB, 0/432 identity, **46.7 t/s** vs `15000` →
  432/432, **65.0 t/s**; the same cliff cost the 2-GPU arm 46.3 vs 66.4 t/s at `-c 204800`.  **r117:** when
  ≥ 96 % of the layers can hold their full slot count, those get full residency and the coldest ≤ 4 % are
  dropped to 0 slots (they run through the scheduler's host copy) — +27.6 % on the arm above, bit-identical
  text/acceptance, `MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the uniform placement, and the placement never
  exceeds `MOE_EXPERT_CACHE_MIB`.  The cap matters: a dropped layer is not free (1 of 48 = +27.6 %, 6 of 48 =
  −30 %), so an arena that is *percentages* short still pays the readback.  **Any 2-vs-3-device (or decode)
  comparison must still record the arena, the residency % and the count of `identity fast path ON` lines.**
- **Greedy text is a per-device-count reference on the MoE tensor split (r116).**  2 vs 3 GPUs on
  `-ncmoe` + `-sm tensor` produce different text (`PPL 10.8452` vs `10.8623`, 0.16 %, the tensor split's
  own rounding class, `GREEDY-PURITY.md` §45) — it is **not** the AR's BF16 wire (`GGML_CUDA_AR_BF16_THRESHOLD=0`
  does not reconcile it).  *Every* `-sm layer` arm is text-identical across device counts, and the recorded
  **dense** cross-count pairs were too — **but a dense tensor split is not guaranteed to be** (measured
  2026-10-24 on Qwen3.8-27B UD-Q4_K_XL: 3 GPU 850 chars vs 2 GPU 876, and r126 shows the same 897 vs 855;
  pre-existing, not a re-base defect).  So gate a build change **within** each device count (as r116's own
  gates did); use `-sm layer` when a cross-count text comparison is required; never read a 2-vs-3 text
  difference as a purity defect.
- **The backend-sampling compute buffers are reserved once per sequence, and again only for a WIDER chain
  (r169, contributor PR #245).**  `llama_context::set_sampler()` used to set `sched_need_reserve` on every set
  of an offloadable chain **and** on every removal, and `sched_reserve()` re-creates the scheduler (every
  compute buffer freed and re-allocated, at the same size) — so `llama-server --backend-sampling`, which sets a
  chain per request and clears it on release (and for a grammar request), paid that before each first token.
  The buffers now remember what they were reserved with (`sampling.reserved`: per sequence, the chain's
  `llama_sampler_backend_n_nodes()`) and a reserve is asked for only when a sequence gets a **wider** chain
  than that.  **The key is the node count, not the sequence id**: `graph_max_nodes()` is recomputed only by a
  reserve and it sizes the graph's node capacity (`llm_graph_result::max_nodes` -> `ggml_new_graph_custom`,
  which `ggml_build_forward` asserts on), the scheduler's arrays and the compute-metadata arena, while the
  allocator's on-demand growth only covers buffer *sizes* — a seq-id-only key misses the widening that
  `tests/test-backend-sampler.cpp`'s `test_set_sampler` performs (`dist` -> `top_k + dist` on one sequence: 1
  reserve with the seq-id key, 2 with the node count).  `LLAMA_BACKEND_SAMPLER_RESERVE_ONCE=0` restores the
  per-set reserve; the change is numerics-neutral (no kernel or dispatch touched), so every greedy-text arm
  and the prefill logits are byte-identical.  Measured median server prompt time 1963.9 -> 1895.2 ms per
  sampled request (27B, `-c 65536 --parallel 2 --kv-unified`).  `patches/README.md` (block 12),
  `upstream/backend-sampler-reserve-once/`, `WORKLOG.md` 261028.011.
- **The disk tier is bounded by this box's SSD at ~1.8 GB/s, so the cache's currency is `L3 MiB/token`, not
  `h`.**  1 MiB/token of reads that neither VRAM nor the pinned pool holds is ~0.55 ms/token of unavoidable
  stall (~1800 t/s of pure read budget at a 1 MiB/token traffic level, ~100 t/s at 18 MiB/token).  Every
  tiering claim in the records is therefore quoted in MiB/token beside `h` (`scripts/moe-trace-replay.py`
  reports both): r114's band-split heat prior took the long-prompt arm from 48.1 to 17.6 MiB/token, i.e.
  **~26 ms → ~9.5 ms of disk per token**, which is what a decode on that config (46 t/s = 22 ms/token) can
  and cannot afford.  Use the same conversion before proposing a prefetch/eviction policy: the pool-layout
  and topic-shift candidates were all declined because their whole prize was 0.3-0.5 MiB/token.
- **The heat prior is band-split (r114, block 12): the ARENA seed ranks by the decode-only counter, the host
  pool's layout by the combined one.**  `apply_heat_seed_locked` + the heat-ordered layer placement call
  `heat_rank_vec()` (the `heat_dec` vector, falling back to `heat` when a map has no decode mass); the pool's
  `pool_rank_experts_locked` / `pool_rerank_from_heat_locked` keep `heat`, because the pool stages whole
  tables *for a prefill* and there the prefill mass is signal, not noise.  Sidecar format **MHT2** (both
  arrays per table); a **MHT1** file loads with `dec = all` (pre-r114 behaviour), never an error.  Do not
  "simplify" this back to one counter: on a long prompt the prefill outweighs the decode 41x and the arena
  ends up seeded from a nearly uniform distribution (+1.2 to +2.5 pts of `h` and ~3x less disk traffic on
  the measured arms).  `ENVIRONMENT.md` (heat-map note), `patches/README.md` (block 12).
- **Thread sizing is not optional.** `/usr/local/bin/pin_gpu_irqs.sh` puts the GPU IRQs on the top
  `NUM_GPUS` cores (13,14,15 with 3 GPUs); the default `-t` puts the worker pool there and starves the
  GPU (Q4_K_M `-ncmoe 99` tg1024: `-t 16` 19.6 vs `-t 8` 38.9). Size `-t` (or `--cpu-mask` with `-t`
  *inside* the mask) to leave those cores free; one CCD (8) is within 1-2 % of best. r19's block-06
  cap only covers `-ncmoe` offloaded-MoE decode; override with `GGML_CPU_MOE_OFFLOAD_THREADS=N`.
- **The host link is PCIe 5.0, and it is not uniform: 2 cards at x8 and 1 at x4 (since the 2026-10 top-up).**
  Each R9700 sits behind its own PCIe switch (`Navi 10 XL Upstream Port`): the GPU side always negotiates
  **x16 @ 32 GT/s**, but the switch **uplink to the host** is what varies — read the *root port* behind the
  card, not the card: GPU 0 (`03:00.0`) -> `00:01.1` **x8**, GPU 1 (`06:00.0`) -> `00:01.2` **x4**, GPU 2
  (`09:00.0`) -> `00:01.3` **x8** (all `current_link_speed` = 32 GT/s).  That is ~32 GB/s raw for GPUs 0/2
  and ~16 GB/s for GPU 1.  Consequences for measurement, measured 2026-10-17 on 35B-A3B Q4_K_M q8_0
  `pp31500` when every card was x4: 1 GPU 3614 t/s, 2 GPU tensor **4482 (+24 %)**, 3 GPU tensor **4347
  (-3 % vs 2 GPU)**; the same +17 % / +1 % shape is in `archive/work/three-gpu-performance/` (1065 / 1246 /
  1258 t/s).  So **multi-GPU prefill on this box is link-bound and now heterogeneous** — a 2-GPU arm is not
  the same shape at `HIP_VISIBLE_DEVICES=0,1` (x8+x4) as at `0,2` (x8+x8), and a tensor-split prefill claim
  from a contributor host with x8/x16 everywhere can only be *under-measured* here, never refuted: report it
  as "not measurable on our link", not as "does not reproduce".  Decode claims do carry over (the per-token
  AR traffic is tiny): #175's decode gain reproduced at 2-GPU tensor while its prefill claim did not.
- **`-ncmoe`/`-cmoe` host experts are pinned (`ROCm_Host`) in the DEFAULT mode; `--load-mode mmap` alone
  does NOT change that.** The loader rewrites the `-ncmoe` CPU override for a host `MUL_MAT_ID` weight to
  the layer device's **pinned** host buffer, and the "avoid a host buffer when using mmap" downgrade is
  skipped for it. The pageable `--host-experts mmap` mode and the legacy `LLAMA_MMAP_HOST_EXPERTS` env
  were **removed in r33** (issue #116: a no-XNACK GPU -- gfx1201 reports `XNACK enabled: NO` -- cannot
  read a pageable host master in a kernel, and the pageable fill faults `hipMemcpy2DAsync` on ROCm 7.14).
  `--host-experts` (`llama_model_params.host_experts_mode`) selects **`pinned`** (default),
  **`pool`** (r35/r111) or `mmap` (logs "unsupported" and falls back to pinned).  Pinned is deliberate --
  the per-ubatch op-offload H2D upload and the expert-cache fill both read this master in place, and under
  `-sm tensor` a pageable master would land in the non-`is_host` `CPU_REPACK` (no device access, cache
  inert).  Cost of the default: the host expert set is non-swappable RAM (Q4_K_M 35B-A3B `-ncmoe 40`,
  cache off: `ROCm_Host` 18662 MiB; peak RSS ~41.7 GB).  `--lazy-mode` governs the mmap *reader*, not
  this. `ENVIRONMENT.md` §§1.2.1/4; `patches/README.md` (blocks 09/12).
- **A qwen4exp fused HC op must reproduce the model CHAIN, and the mix/combine arms are the oracle (r148, issue #228).**
  `build_hc_mix`/`build_hc_combine` each exist twice (upstream's ggml chain and `GGML_OP_HC_MIX` /
  `GGML_OP_HC_COMBINE`), selected **independently** by `LLAMA_FUSED_HC_MIX` / `LLAMA_FUSED_HC_COMBINE`, so
  there are four arms and all four must produce identical text.  The intermediate **inject scatter logit is
  the emitter's** (upstream's chain emits it *already* `1/hc`-scaled and the combine applies only
  `2*sigmoid`; v14 emitted it raw with the scale in the combine), a convention the `6050b467c` re-base
  inverted under the delivery's fused pair: through r147 two of the four arms were wrong (chain mixer +
  fused combine = `/hc` twice, which collapsed every hc-down quant that is not Q8_0/BF16 — Q5_K/IQ4_NL/Q4_XS —
  into greedy repetitions; fused mixer + chain combine = no scale at all, an **empty** generation on a
  BF16-hc model).  `test-backend-ops -o HC_MIX,HC_COMBINE` cannot see it (its CPU reference shared the
  convention) and greedy coherence cannot either (the wrong `w` is shared by every stream) — run
  `scripts/gate-hc-fusions.sh MODEL.gguf` for **both** hc-down weight classes (Q8_0/BF16 mixes fused, any
  other quant mixes on the chain; `--types MODEL.gguf` prints the class), before any change to an HC op, its
  chain or a matcher, on every release, and as a re-base checklist item.  `GREEDY-PURITY.md` §61.
- **A host-side call in a file the delivery ADDS must be Windows-portable (r147, issue #231).**  The
  set is built with HIP on Windows too (clang-cl + the MSVC STL), where `<unistd.h>`/`<fcntl.h>` do not
  exist: `close`, `pread`, `open`, `fsync`, `access`, `sysconf`, `getpid`, `posix_memalign`, `mmap` …
  are undeclared.  `ggml/src/ggml-cuda/moe-expert-cache.cu` is consciously portable — every host-pool
  file call goes through five helpers (`pool_open_file`, `pool_fd_align`, `pool_pread`,
  `pool_aligned_alloc`, `pool_close_file`; issue #130) whose Windows versions are no-ops, because the
  bounded pool never opens a file there and a miss is served from the pinned master.  r143's teardown
  added two direct `close()` calls and broke the reporter's Windows build; nothing on this box can see
  that (no Windows runner, no Linux compiler diagnostic), so `scripts/check-windows-portability.sh`
  lints for it and `validate-set.sh` runs it on the applied tree in tier-1 CI.  A new host C++/CUDA file
  owes the same discipline: guard the call with `#if !defined(_WIN32)` or add a helper beside the five
  (Windows arm empty, POSIX arm the real call).  The lint is textual — it cannot see a missing
  `#include`, an MSVC type difference or the `std::min<int64_t>` mangling clash (issue #217); those still
  need a Windows build log.
- **`--host-experts pool` (r111) has NO pinned master: the master IS the read-only model mapping, and
  every GPU read of a pooled host expert goes through pinned memory.**  The tensors live in the model
  mapping (host buft `ROCm_HostMap`, device-carrying, r111, **block 12**) and are never read by a device
  copy or a kernel; the L2 tier is the bounded pool (`MOE_HOST_POOL_MIB`/`FRAC`, default **80 %** of the
  host expert bytes since r112, *per tensor*, process-wide, capped at **`MOE_HOST_POOL_RAM_PCT` = 85 %** of
  `MemAvailable` since r114) and the L3 tier is the GGUF, read with O_DIRECT **only when the page cache
  cannot hold the set** (`MOE_HOST_POOL_DIO` is **`auto`** since r114: decided once at the first pool build
  from `MemAvailable` vs `planned pool + host expert set`; `0`/`1` still force either arm).  The
  mode is only for the case where the set does **not** fit in RAM: when it fits, `pinned` is faster on every
  axis.  80 % is the measured knee (90 % was slower and swapped).  The invariants:
  (1) a pooled table must never be served by a **pageable** `cudaMemcpyAsync` -- ROCm stages a pageable
  upload through a shared internal bounce and back-to-back copies tear (this is what made the first cut
  nondeterministic; `moe_cache_upload_pageable`, the staging bounce and the `stage_gather` host-gather
  shape all exist for it); (2) the cold-UVA serve and the device gather require a *pinned* master and
  stand down (`host_dev == nullptr`) -- a miss becomes a pool fill; (3) any new reader of a host expert
  master must go through `expert_host_src` (pool slot / per-table pinned bounce) or be added to
  `moe_cache_upload_pageable`; (4) the mode needs a **mapped** load (`--load-mode none/dio` keeps the
  pinned master and warns).  Measured: `RssShmem` 18.6 GiB -> 6.8 GiB on 35B-A3B `-ncmoe 40`, and a
  12 GiB or 8 GiB cgroup that OOM-kills the `pinned` arm runs this one at ~93 t/s.  Cost: one extra host
  memcpy per staged byte on the host-expert prefill path (single-ubatch `pp512` on 2-GPU tensor -34 %),
  which is why the staging copy itself must stay batched (`MOE_HOST_POOL_STAGE_BATCH`, r114).  Record:
  `archive/work/host-expert-master-drop/`.
- **The host pool is a LAYOUT, not a churning cache (r113, block 12; TODO #87's disk-I/O half).**  The
  pool's **content** is what decides the disk traffic — measured offline on a real trace: pool ranked by
  expert id = 3571 MiB of L3 reads over 451 decode tokens, ranked by the **persisted heat prior** as the
  arena's *complement* = 237 MiB, oracle = 237 MiB — while **every** eviction/hand-over policy lands within
  0-246 MiB of the others.  So: (1) `pool_rank_experts_locked` / `pool_rerank_from_heat_locked` lay the pool
  out as the arena's complement, in bulk, at sizing and at the first decode-band pass (the old flat
  prompt-tally re-rank is gone — a prefill routes every expert, so its tally cannot rank); (2) the **demand
  path never evicts and never copies the arena victim into the pool** (`pool_peek_locked` /
  `pool_fill_empty_locked`; the background worker and `MOE_HOST_POOL_BGFETCH` are gone) — do not reintroduce
  one: the literal swap cost **2x decode** for 0 MiB of L3; (3) the whole-table prefill staging reads the
  pool and fills empty slots (it used to bypass it and re-read the whole host set per ubatch); (4)
  `MOE_HOST_POOL_RAM_PCT` (default **85** since r114) caps the whole pinned pool at that % of `MemAvailable` — `FRAC`
  is a percentage of the model's host expert set, not of RAM, and an oversized pool fails each table's
  `cudaHostAlloc` *silently*.  Field config (70K prompt, 128-token decode, 2x R9700): pool hit 0.80 -> 0.92,
  evictions 4600 -> 0, prefill 39 -> 287 t/s (`DIO=1`) / 581 -> 733 t/s (`DIO=0`), decode 6.3 -> 23.1 /
  30.6 -> 53.7, against `pinned` 836 / 57.5.  The tool that decided it (`scripts/moe-trace-replay.py
  --pool-frac`, two-tier, reads a real `.moe-heatmap`): use it before proposing another pool policy.  Record:
  `archive/work/host-pool-layout/`.
- **POOL mode: the fill mode is automatic and the staging is one launch (r114, block 12, TODO #87).**
  `MOE_HOST_POOL_DIO` = `auto` (decided once at the first pool build from `MemAvailable` vs `planned pool +
  host expert set`: buffered when the set fits beside the pool, O_DIRECT when it does not; `0`/`1` force),
  and `moe_cache_stage_kernel` fills **both** tiers (arena slot / pool slot) for a pool-backed table, so a
  prefill staging is one launch instead of one `cudaMemcpyAsync` per expert.  Measured (2x R9700, 35B-A3B
  Q4_K_M `-ncmoe 40 --host-experts pool`, 70K prompt, `-ub 2048`): prefill **2103 t/s** vs **1564** with
  `MOE_HOST_POOL_STAGE_BATCH=0` (+34 %), decode 46.4; forced `DIO=1` 2088/**22.7**; `pinned` 2099/49.3.
  Text is byte-identical across every arm and `MOE_HOST_POOL_VERIFY=1` is clean.  **Long-session adaptation
  was measured and declined**: the oracle-layout ceiling is 7.45 MiB/token of L3 vs 8.05 for a
  previous-topic layout (~6 % per topic shift), and every churning policy costs several times that.
  Record: `archive/work/host-pool-swap-and-prefill/`.
- **MTP gates must be long and reasoning-pinned.** Use `-n 3000` (`-n 2000` floor), `--reasoning on`
  for R and `--reasoning off` for P/C/K. Short runs are a correctness smoke test only. Rule 0 in
  `benchmarks/mtp-adaptive-methodology.md`.
- **The MTP drafter is embedded in the model GGUF — check before claiming there is no drafter.**
  `--spec-type draft-mtp` runs the model's *own* MTP head, so every Qwen3.5/3.6/3.8 GGUF under `/llm/models`
  (and every other model whose GGUF carries an `mtp`/`nextn` tensor) can run an MTP gate with **no second
  model file**: `-md` is only for a *separate* draft model.  `scripts/fork-refresh.sh --gates` prints the
  canonical Protocol A command instead of running it because the arm needs a model and a device-count choice
  — that is a prompt to pick them, not a statement that a drafter is missing.  Run the arm on **both split
  modes** (`-sm layer` and `-sm tensor`; dense 27B UD-Q4_K_XL, `-c 102400`, bf16 KV, `-n 3000`).  **Quote the
  device count with the numbers** - the reference arm is **3 devices**: layer **51.7** and tensor **90.7**
  t/s with `--spec-type draft-mtp` (tensor plain 48.4), acceptance 0.62 / 0.60; the **2-device** arms (the
  `0,2` x8+x8 pair) are layer **54.6 vs 26.6** (acceptance 0.621) and tensor **81.4 vs 44.8**
  (acceptance 0.541).  A 2-device arm read against the 3-device reference looks like a ~10 % "regression"
  that is not there (`WORKLOG.md` 261028.002).  Also note that
  `llama-cli` needs `-v` to print its perf/acceptance summary at all.  Never report "MTP gate not run: no
  drafter available" without checking the model's tensors first (`llama_model_loader` logs the MTP head, or
  `gguf-dump --tensors | grep -i 'mtp\|nextn'`).
- **Prefill-logit purity is a release gate (issue #113).** Same-seed coherence and MTP are blind to a
  uniform prefill shift. Run `scripts/gate-prefill-logits.sh` before shipping any change to a prefill
  kernel (GDN/SSM chunked, FA, MMQ/MMVF/MMB, MoE prefill) or a default-on approximate path: it compares
  against a recorded known-good base and requires mean KLD <= 0.005 and same-top-p >= 98 %. The r29
  example: the default-on BF16 chunked GDN measured 0.032 / 93.6 % and was flipped off.  A
  **kernel-family switch** (a tile→WMMA head cap) uses `--family-switch` instead: it refuses a point
  outside the gate's calibrated domain (base PPL > 10; at a point the model finds hard even a benign
  same-kernel ubatch change measures 0.105 / 85 %) and always prints a same-kernel control, so the
  absolute limits stay the criterion without an insensitive point waving a bad change through.
  Protocol: `benchmarks/prefill-logit-methodology.md`, `GREEDY-PURITY.md` §46.
  **A base is per ROCm line (r192, TODO #126).**  The `.meta` records the ROCm line the base's binary
  linked, and a compare against a build on a different line is **refused** (`--allow-base-mismatch` shows
  the number anyway): the same tree reads 0.005657 / 96.794 % on 10.1.0 and 0.005688 / 96.765 % on 7.14.1
  at the 35B-A3B point, i.e. a cross-line "divergence" is the compiler.  When a gate failure appears at a
  point that used to pass, check `.meta`'s `build=`/`rocm=` and the `gate-prefill-logits.sh` header line
  before suspecting the change: a stale base read as a gate failure for six releases (#126) because the
  r161 re-record was written on another host.
- **The sequences that verify in one step must present the same row count (r153, issue #234).**
  `llama_memory_hybrid::init_batch()` splits a decode with `llama_batch_allocr::split_equal()`, which takes one
  token per sequence set per round and **stops at the shortest set**, so two slots verifying different draft
  lengths put the longer one's tail into a SECOND ubatch and the target runs a full extra pass for that step.
  A fixed `--spec-draft-n-max` never exposed it (every slot drafts `n_max`); the adaptive controller's
  **per-sequence** depth makes it the normal case on a multi-slot server (`--parallel >= 2 --kv-unified`).
  `LLAMA_SPEC_ALIGN` (default on) makes `draft-mtp-adaptive` give every sequence it drafts in one call the same
  cap: the deepest in the call, bounded by the smallest HARD cap (a response running out of tokens) and by the
  band.  The alignment is **inside that one drafter**; a generic cut of the finished drafts to the shortest was
  shipped in r153 and is a **measured loss** (the n-gram-first chain, -6 %, because it truncates the other
  drafter's good draft), so r154 removed it: a mismatch caused by CONTENT is not worth equalizing, only one a CAP
  caused.  Measured 27B UD-Q4_K_XL greedy `-n 400`, concurrent recall+reasoning pair: **108.1 ->
  140.6 t/s at 2 GPU (+30 %), 86.9 -> 133.7 at 3 GPU (+54 %)**, 35B-A3B `-ncmoe 40` 116.2 -> 149.0 (+28 %);
  the split decode steps read 58 before and 0 after.  **Padding the shorter sequence is dead** (a padded row
  advances the recurrent state in `mem_recr->prepare`), and **cutting the adaptive depth to the shallowest is the
  wrong direction** (adaptive depth is worth +13 % over a fixed one at equal lengths).  The band applies to the
  whole batch: the aligned depth is `band / n_drafting - 1` rows per sequence (two slots at depth 7 fill the
  RDNA4 band), never lowering a sequence below its own cap.  **A slot that cannot draft at all** (its last
  token, or a full context waiting for a shift) still splits that step, **deliberately**: that second ubatch
  carries the neighbour's genuine draft rows, and both removing it (suppress the step's drafts) and cutting the
  neighbour's draft measured neutral (`TODO.md` #111, analysed and declined — do not "fix" it; the gate counts
  those single-row splits separately and bounds them with `TAIL_MAX`).  Gate: `scripts/gate-spec-align.sh` (2 and
  3 GPUs, host experts, and a mixed-drafter arm that must stay inert).  Purity: a verify-width change moves
  greedy text by itself, so the oracle is the **single-sequence** byte-identity (`n_seq == 1` and every
  single-slot request) plus a same-width control, never a two-slot text diff (`GREEDY-PURITY.md` §63) — and a
  two-slot **aligned** pair is not even stable run to run (one unchanged binary, two text basins at 3 GPUs), so
  judge this code by the split count and the single-slot text.
  Record: `archive/work/spec-align-234/`.
- **A multi-slot draft's verify WIDTH is bounded by a kernel width, not only by the MoE arena band (r185, block 02).**
  `align_draft_caps()` lowers the aligned depth to `width / n_drafting - 1`; `width` was the expert-cache band
  alone, and r176 made `band` answer only for a model with **host-resident experts** — so on a band-0 config
  (all-in-VRAM, `-sm layer`, no routed expert) a 4-slot `draft-mtp-adaptive`/`draft-mtp` draft was **unbounded**
  (4 x `(1 + n_max 7)` = ~28-32 rows, past the MMVQ/MMVF -> MMQ and FA family switch) and the *default*
  recommended speculator ran **9.5-31.8 % slower** than the same arm bounded at 16 rows.  `width` is now the
  arena band where one exists and `LLAMA_SPEC_ALIGN_WIDTH_BAND` (default **16**, `0` = off) otherwise.  **16, not
  8:** a 16-row batch beats an 8-row one on 3 of 4 fixed-depth shapes (35B-A3B 3 GPU 432.7 vs 389.8 t/s, dense
  27B 3 GPU 240.1 vs 211.2; qwen4exp + a resident PLE table is the exception, 204.6 vs 220.8), so PR #169's
  "above ~8 rows each extra draft token costs more" is refuted — the *unbounded* width was the defect, and a
  shape/model whitelist would be wrong on the same model across quant, device count, slot count and speculator.
  4-slot gains: 35B-A3B 3 GPU +31.4 %, qwen4exp 2 GPU +19.0 %, dense 27B 3 GPU +14.2 %, fixed `n_max 7` +16.7 %;
  3 slots +16.9 %; **1 and 2 slots inert** (16 / 2 - 1 = 7 = the default `n_max`) with byte-identical 2-slot text.
  A host-expert config keeps its arena band and is unchanged.  Gate: `scripts/gate-spec-width-band.sh` (the only
  gate that reaches it: every other spec/coherence/MTP gate runs one or two sequences, where the bound is inert).
  `benchmarks/2026-10-31-multislot-verify-width.md`.
- **The draft context's KV cache type follows the main `-ctk` / `-ctv` (r125, block 02).**  Upstream gives
  the speculative draft its own type pair (`-ctkd` / `-ctvd`, default f16) and applies it unconditionally;
  since r125 the draft takes the main model's type unless `-ctkd` / `-ctvd` (or
  `LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K` / `_V`) pins it.  At `-c 131072` on a dense 27B that is 512 -> 144 MiB
  of draft KV.  No generated token can change (the draft only proposes; the target verifies) and greedy text
  is byte-identical between the two draft types, but **any VRAM/`--fit` arithmetic for an MTP or `-md` run
  must use the main KV type for the draft context too**, and a *small, fast* model under `-sm tensor` pays
  1.5-3 % decode for it (the quantized KV write costs a few host-side node evaluations per step; GPU kernel
  time is identical).  Record: `WORKLOG.md` 261024.005, issue #216.
- **Mixed K/V cache types are HARD-REJECTED** at context creation (`params.type_k != params.type_v`);
  every mixed pair measured 1.7-3.6x slower and never smaller. Pass matching `-ctk`/`-ctv`.
- **The MTP draft-vocabulary trim is opt-in and draft-only (block 13, r107).**  `--spec-draft-mtp-vocab N`
  copies the LM-head rows for ids `< N` (+ control/user ids) into the draft context, scores the draft over them
  and scatters into a full-vocab row of `-inf`; the target still verifies the full vocabulary, so the trim can
  only lose proposals.  `0` = off = today's code path.  **N is payload-specific** (98304 on Qwen3.8-27B Q8_0,
  64000 is the reporter's gfx1151 value and a *loss* on our 27B, gemma4's only lossless cut is 245760/262144 =
  a no-op), it costs `N/n_vocab` of the head in VRAM (510 MiB at 98304, per device under `-sm tensor`) and it
  is **not** in the `--fit` estimate.  Gate purity at **`n_max <= 7`** (`GREEDY-PURITY.md` §48: the trim moves
  acceptance, which moves the adaptive controller's depth, which moves the verify widths); the shipped
  recommendation is `--spec-draft-n-max 7 --spec-draft-mtp-vocab 98304` (leave `--spec-draft-n-start` at its
  default 3 — it is the *initial* depth, not a ceiling).  **Check the device count before enabling it:** the
  subset head is *mirrored* per device under `-sm tensor` while the untrimmed head is *split*, so it pays only
  when **`N * n_devices < n_vocab`** — on the 27B 98304 is +1.4 % at 2 GPU (share 124160) and **-2.2 % at
  3 GPU** (share 82773, the server shape), and no N pays at 3 GPU.  The `ggml` half is the
  `ggml-backend-meta` row-range read that makes a vocabulary-split LM head readable piecewise.
  Record: `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md`.
- **`--spec-draft-n-max` is capped at 15** (the recurrent snapshot bound); purity is promised only to
  **7** (the kernel-family switch at 8 rows). `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` escapes the clamp.
- **A draft length is a verify width *and* a memory-tier parameter (r123, TODO #59 closed).**  With host
  experts served by the expert-cache arena, a verify batch wider than the band (`moe_cache_max_tok()`: **16**
  on RDNA4/RDNA2, **8** on RDNA3/NVIDIA) leaves the arena and streams the routed experts from the pinned
  host master on *every* step — one extra row measured **159.2 -> 55.0 t/s** on 2x R9700 `-ncmoe 36`, and the
  README's `draft-mtp-adaptive,ngram-mod` pick 73.9 -> 168.3 t/s once the draft fits.  `common` clamps the
  **draftless** n-gram speculators (`ngram_mod.n_max` 64 / `n_min` 48 default, the ngram-map `size_m`) to
  `band - 1` whenever the arena is active — `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` opts out, and a
  `draft.n_max` above the band warns (not clamped: its 15 is a correctness bound).  A device-resident model
  or the CPU expert path reports band 0 and is **never** clamped (wide drafts are the win there: 144.7 ->
  378.7 t/s; a naked `--spec-ngram-mod-n-max` below `n_min` silently disables ngram-mod instead).  **That
  "0" is a property of the PREDICATE, not of `moe_cache_enabled()` (r176, TODO #117):** a band is answered
  only for a device that carries **host-resident experts** (`model->moe_host_expert_bytes`), because
  `moe_cache_enabled()` is true by default (AUTO) even with no registered table, and the old
  `model->devices` fallback made a **dense** model on a single GPU or any `-sm layer` run answer 16 — which
  n-gram-clamped a 64-token draft to 15 (the warning's routed-expert premise false) and bounded every
  multi-slot MTP/DFlash alignment to `band / n_drafting - 1` rows.  Measured: dense 27B 1 GPU
  `ngram-mod --spec-ngram-mod-n-max 64` on `prompts/recall.txt` **142.1 -> 186.8 t/s (+31 %), text
  byte-identical** (the r175 `LLAMA_SPEC_NGRAM_MOD_BAND_CLAMP=0` control reads 187.9, so the whole delta is
  the clamp).  Keep the device hook's `moe_cache_enabled()` gate (a slab-declined/arena-released host-expert
  config still reports 0) and do **not** test `moe_cache_has_tables()` there — tables register during the
  first compute, which is exactly the r129 bug below.  **The
  clamp runs AFTER `llama_init_from_model`, on the arena's real state (r129, TODO #96):** the model-load band
  only says the slab was *armed*, so clamping there clamped slab-declined (streaming, no-arena) configs.  It
  therefore no longer lowers `cparams.n_rs_batch` — that stays the **unclamped** safety bound (a partial
  rollback can land in any batch up to an unclamped draft; the value also floors the chunked-GDN threshold,
  so an ngram speculator's prefill *tail* ubatches take the sequential bit-identical GDN path) — and the
  slab's narrow verify reserve takes the band-limited width instead (`llama_context::verify_batch_width()`),
  so a live arena reserves exactly what it did before.  Rule:
  `GREEDY-PURITY.md` §50; records: `archive/work/ngram-mod-host-experts/EVIDENCE-2026-10-24.md`,
  `WORKLOG.md` 261029.008.
- **The mmvq band-uniform knobs are a purity requirement *and* a perf trap.** `nwarps` and the vec_dot
  VDR participate in the mmvq K-split order, so one value must hold across the whole band (`W=1..8`)
  and must be tuned at the **verify widths**. RDNA4 dense `nwarps=1`; dense VDR=2, MoE-expert VDR=4.
  **Dense Q6_K VDR=2 is upstream's default on every arch since #30517 (`2744e48d1`), and the delivery keeps
  it (r170):** block 07's own per-arch gate (`RDNA4`/`RDNA3_0` → 2, VDR=1 elsewhere) was silently
  re-scoped by that base bump, and the re-measurement on gfx1151 says keep it (B=1/B=4/d16384 within noise,
  batched B=8 +1.2 %, width purity `plain == draft-mtp` intact). Text moves against a pre-#30517 build —
  association order, not a defect (`GREEDY-PURITY.md` §68). **`-DGGML_HIP_Q6K_DENSE_VDR1`** is the
  build-time escape for a consumer with an output-identity gate (never a runtime knob: VDR is a `constexpr`
  the host K-split arithmetic also reads, so a runtime arm would duplicate the mmvq kernel family).
  The **pinned fusion ops (GDN/SSM, shared-expert, gate fusions) MUST keep plain `calc_nwarps`** — they
  are single-token reduction-order anchors. Run the stock-relative
  `llama-batched-bench -npl 1,4,8` gate (`mtp-adaptive-methodology.md` rule 5) before any decode/mmvq
  change. `GREEDY-PURITY.md` §§19/24/25.
- **The adaptive-MTP controller is the tuned bucketed one** (`v16-d1d3c3396-r2`): `delta = n_accepted -
  depth`, `climb_budget(d) = 20 + 6*(d-1)`, `drop_pressure(d) = max(60, 10*d)`, cold start
  `max(floor, cap-3)`, `--spec-draft-n-start N`. Judge any change on the four axes **and** the
  phase-switching `prompts/code-reasoning-mixed.txt`. Records:
  `benchmarks/2026-09-15-adaptive-mtp-tuning.md`, `archive/work/adaptive-mtp-ceiling-scaling/`.
- **FA instantiation discipline:** a FA kernel's KV *type* axis must be instantiated in the generated
  `template-instances/*.cu` files, never left implicitly in the dispatch TU (it gates the build).
  Check with `nm -C <obj> | grep -c <case>`: the dispatch TU shows `U`, the instance TUs `T`/`W`.
  Detail: `archive/work/build-time-regression/`.
- **Every kq-mask fill is guarded on `tensor && tensor->buffer` (r131, PR #222).**  Block 05's derived kq
  form (`build_attn_inp_kq_mask(..., allow_derived = true)`) leaves the packed mask with **no consumer** at
  one slot, and ggml-alloc gives an unconsumed input tensor no buffer, so an unconditional fill hits
  `llama_kv_cache::set_input_kq_mask`'s `GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer))` and the
  process aborts before the first token (`bailingmoe3` at `-np 1` was the live case).  The guard is the
  file's own idiom (`llm_graph_input_attn_kv` is the origin); three sites lacked it and were fixed in
  `llm_graph_input_mem_hybrid_k` (reachable) and `llm_graph_input_attn_k_dsa` (hardening, `allow_derived`
  false for its `mla`/`lid` masks).  A new fill site, a new `allow_derived = true` builder or a new hybrid
  input class must carry it; `can_reuse` needs nothing (`can_reuse_kq_mask()` reads `ne[]`, never
  `buffer`).  `LLAMA_KQ_MASK_DERIVED=0` is the work-around and the packed reference arm.
- **The packed kq mask is reserved only where a batch needing it is reachable, and "M-RoPE" alone is not
  that (r160, issue #238).**  `kq_mask_packed_reachable()` (block 05) used to answer "yes" for any
  `n_pos_per_embd() > 1` model, but Qwen3.x is IMROPE even for text, where every position is degenerate and
  `kq_mask_derivable()` accepts every batch: a text-only 27B at `-c 220160 -ub 2048` reserved **860 MiB of
  VRAM plus 860 MiB of pinned host memory** per context that was never allocated.  The M-RoPE term is now
  `cparams.kq_mask_pos_2d` (`llama_context_params.pos_2d`, set by `common` when `--mmproj` is given) or the
  `kq_mask_pos_2d_seen` latch that `decode()` sets on the first batch carrying a non-degenerate 2-D (embd)
  position, which forces one **re-reserve** with the packed mask before that batch is split.  Alibi and the
  `allow_derived == false` builders keep the packed reserve, and a multi-sequence batch does too **unless** the
  derived form can serve it (r172 - see the next bullet); the runtime mask path is unchanged (every derivable
  batch still takes the derived form).  So: a context that can receive
  image/audio chunks must be created with `pos_2d` (a caller that submits 2-D positions and skips it pays
  one re-reserve, which on a `-ncmoe` config can cost the expert-cache arena - declare it); do not
  reintroduce an `n_pos_per_embd()`-only reserve test, and do not remove the latch - it is the only thing
  that keeps an undeclared multimodal caller correct.  A/B arm: `LLAMA_KQ_MASK_PACKED_RESERVE=1` (the
  pre-r160 worst case).  `WORKLOG.md` 261028.001.
- **A second sequence in a batch is a position *stride*, not a new mask (r172, PR #247).**  On a `--parallel > 1
  --kv-unified` context sequence `k` of the ubatch gets its own range (`k * 2^23`, `KQ_DERIVED_SEQ_STRIDE`), so the one
  predicate `tok_lo <= cell_pos <= tok_hi` stays exact and **no kernel changes** - which is also why the batch-wide
  skip window stays conservative under it.  The form is taken only with one KV stream, one sequence per token, **no
  cell carrying two of the batch's sequences** (a prefix shared through `llama_memory_seq_cp`), every position below
  the stride, and **more than 32 tokens**: the multi-slot verify width keeps the packed mask, and so does every other
  refused shape - `kq_mask_derivable()` re-runs the whole predicate per ubatch, so correctness never depends on the
  reserve latch (`kq_mask_multiseq_seen` saves only the one growth the r160 latch pattern also prevents, and is the
  one refusal that depends on the *cache* rather than on the batch).  The shared-cell counter
  (`llama_kv_cells::get_n_multi_seq`) is a **fast path, never the semantics**: an under-count is a silently wrong
  mask, so the O(used cells) scan is the answer it licenses skipping, and a new way to put two sequences in one cell
  must bump it.  Measured -1.03 GiB device + 0.94 GiB pinned host at `-c 245760 -ub 2048 --parallel 2` (dense 27B),
  prefill neutral, and **zero** on a graph that also holds an `allow_derived == false` builder (qwen4exp/QSA, MLA -
  Flash-Next measures identical VRAM and RSS); a server that really shares prefixes pays one re-reserve and then
  keeps the packed mask.  `LLAMA_KQ_MASK_DERIVED_MULTISEQ=0` is the A/B arm (and the packed reserve); judge it with
  `gate-prefill-logits.sh --ctx 512 --ub 2048 --ab "LLAMA_KQ_MASK_DERIVED_MULTISEQ=0"` (0.000000 / 100.000 %) plus a
  cell-by-cell oracle run, **never** with a two-slot text diff.  `GREEDY-PURITY.md` §70, `WORKLOG.md` 261029.003.
- **A graph-reuse predicate must look through the memory-context wrappers, and a hybrid context has exactly one
  attention cache to look through (r164, PR #242).**  `llm_graph_input_kq_derived::can_reuse()` sees
  `params.mctx` (the wrapper), never the cache the input was built from, and the pre-r164 code rejected on that
  alone - which silently rebuilt the graph on **every** single-sequence batch of more than 8 tokens: a long
  n-gram draft's verification, or an MTP verify step at `--spec-draft-n-max >= 8` (measured: **0** graphs
  reused over 300 decode tokens at a fixed depth 15, i.e. r163 was 7.6 % slower at 3 GPU `-sm tensor` and 5.4 %
  at 2 GPU than its own `LLAMA_KQ_MASK_DERIVED=0`).  It now looks through
  `llama_memory_hybrid_context::get_attn()` - the cache `build_inp_mem_hybrid()` built the input from - and then
  re-runs the shape / `kq_mask_derivable()` checks unchanged; `LLAMA_KQ_MASK_DERIVED_REUSE=0` restores the
  rejection.  iSWA **must** keep rejecting: `build_inp_mem_hybrid_iswa()` registers two derived inputs (base
  and SWA), so a look-through would validate one against the other's cache - that needs the input to record
  which cache it belongs to first.  A new wrapper or a new `allow_derived = true` builder owes the same
  reasoning, and a change here is judged by the graph-reuse count (`llama-cli -v` prints `graphs reused`)
  against the packed arm, not by text alone.
- **A Q8_1 quantize-cache key walk stops at the first pointer change, and every fill validates its read
  range (r132, PR #218).**  The cache is keyed on the **root** tensor but the arena is filled from the
  caller's (possibly offset) **view** pointer, so an offset view that walked to the root stored its rows
  under the root's identity and a later op with the same canonical row layout could take them; a stale
  pointer after a compute-buffer reuse then faulted (`MEMORY_APERTURE_VIOLATION`) or read reused memory.
  Every walk is `while (key->view_src && src->data == key->data) key = key->view_src;` (offset-free views
  keep the shared entry — that is the common case and unchanged), and `q8_1_src_range_ok()` (block 08's
  `common.cuh`) aborts with the layout logged if a source's extent leaves its root buffer.  A new cache
  site or a new fused writer that fills the cache for a matmul carries both; there is no knob.  Not
  reachable on the maintainer's configs (0 offset-view keys across seven configs), so its validation is
  code review + the identity gates, not a repro.  `GREEDY-PURITY.md` §55.
- **A per-arch kernel-policy predicate keys on the device it runs on, never on `devices[0].cc` (r132).**
  MMB's `mmb_enabled`/`mmb_cfg`/`mmb_dense_flag`/`mmb_wtype_mask`/`mmb_dense_tmask` read
  `ggml_cuda_info().devices[ggml_cuda_get_device()].cc`; the primary-keyed form froze one arch's policy for
  every device, so a gfx1030 secondary behind a gfx1100 primary ran RDNA3-tuned MMB kernels and zeroed its
  output-projection logits (PPL == vocab).  A table that is per-arch is cached per architecture *class*
  (`mmb_arch_defaults()` discriminates only RDNA4/RDNA3_0, so the fallback table is cc-independent — add a
  branch if that changes).  A pass that consults such a predicate without launching selects its own
  context's device first: `ggml_backend_cuda_graph_optimize` does, because the Meta backend forwards its
  marking half per child and the current device is then still the previously computed one.  On a
  single-arch box the change is a no-op.  `GREEDY-PURITY.md` §56.
- **The host-resident-expert device gather's head pad must match the host path**
  (`min(expert_size, 512)`; **64 is IQ4_NL-only**), keyed on `(input_cpy buffer, expert_bytes)`. Run
  `scripts/gate-qwen4exp-quant-coherence.sh` before changing `moe_cache_gather_host` or the head zero.
  Needs no `MOE_EXPERT_CACHE_MIB`.
- **Never write `std::min`/`std::max` on `int64_t` in `moe-expert-cache.cu` (r124, issue #217).**  With HIP on
  Windows (clang + the MSVC STL) clang's force-included `cuda_wrappers/algorithm` and the MSVC STL
  `<utility>` both define an inline `std::min<long long>` under **one** mangled name, so a TU that
  instantiates both fails to codegen (*"definition with same mangled name"*) — that broke r123's Windows
  build.  Use the file's `moe_cache_min_i64`/`moe_cache_max_i64` (block 09); `std::min` on `int`/`size_t` is
  fine.  Other HIP TUs still contain the int64_t pattern and compile on the reporter's toolchain, so sweeping
  them is **not** sanctioned (TODO #95, accepted limitation — fix a named TU only from a Windows build log).
- **An over-budget config is reported, never silently repaired, and Windows cannot be fixed from here
  (r135, issue #219; the "~30x collapse" behind it explained and closed in r167, TODO #103).**  `--fit`
  prints the arithmetic at ERROR when it cannot meet its target and the user pinned `-c` / `-ngl`
  (`common/fit.cpp`), and `GGML_CUDA_MIN_FREE_MIB` is the only mechanism that *enforces* a device budget: an
  opt-in free-VRAM floor at `ggml_cuda_device_malloc` that fails an allocation like a driver OOM so the
  fail-soft chain still runs.  It is **skipped while the movable-boundary slab is armed** (the slab parks
  nearly all free VRAM by design), so no `-ncmoe` MoE config changes.  On Windows/WDDM an over-committing
  device allocation is not refused at all (it pages to shared memory and the GPU faults, `dxgkrnl`
  bugcheck); on Linux the same state aborts with `ROCm error: out of memory`.  **Do not re-derive a
  "silent over-commit" reading of the Linux side:** HIP *refuses* an allocation that does not fit and never
  demotes a resident working set (a 30 GiB resident set still reads at 505-513 GB/s with 1.7 GiB free), the
  at-the-limit cliff on this box is ~50 MiB wide and ends in a clean abort, and the only measurable
  degradation is **page-migration traffic for the *other* client's buffers** (0-48 % of prefill in the only
  arms that produced any, tracking GTT migration volume — a single huge holder BO is migrated whole, the same
  hold in 64 MiB BOs is not migrated at all and costs nothing).  `GGML_CUDA_LOW_FREE_MIB` (default **512**,
  block 10, r167) warns once when a run reaches that state and the low-water mark is tracked on every config;
  record: `archive/work/vram-limit-103/`.  Never read a Linux "it works at 0 MiB free" run as evidence that a
  16 GiB Windows card can run the config.
- **A compute buffer that cannot grow must never abort; the MoE arena is what yields (r20).**
  `GGML_COMPUTE_BUFFER_MARGIN_PCT` (default **10**, `0` disables) pads each COMPUTE buffer allocation so a
  runtime graph's slightly larger layout (+3.3 % measured) does not force a free-then-allocate-larger
  **contiguous** block -- that needs a block *bigger than the one just released*, so free VRAM elsewhere
  (including a smaller arena) does not substitute. The slack is an opt-in buffer-type capability
  (`get_compute_margin_pct`), set only by the RDNA/ROCm path and aggregated (min) by the meta buffer type,
  so no other backend's allocation sizes change. `ggml_cuda_device_malloc` is the single allocation choke
  point: on OOM it frees the largest arena table, then the whole arena, and retries; a failed ramp leaves
  the cache disabled rather than aborting. Keep the two together. Related invariant: a partially-failed
  cache **falls back wholesale** (the fusion guard and `moe_cache_take_over` must agree), and no arena may
  be freed while a kernel that carries its address in its launch parameters is in flight
  (`moe_cache_sync_devices_locked`). `GREEDY-PURITY.md`/`archive/work/moe-cache-autosize/ARENA-UB-TENSION.md`
  §13-§14.
- **The FA prefill staging arena's size hint is a CEILING, not a target (r156, issue #237).**  `launch_fattn`
  passes the F16 size of the K/V view **roots** (per-operand capped by `GGML_CUDA_FA_STAGE_MAX_MB`, 512 MiB
  default) so a growth step can jump towards the eventual size instead of climbing the 1.25x ladder -- on
  Windows/WDDM a freed allocation stays committed, so the ladder strands ~5x the final arena (issue #138).
  But the first FA call of a large-context prefill requests a couple of MiB (`-ub 512` on a 262144-token
  cache), so allocating the hint there commits the whole arena up front: **1 GiB per context**, and since
  r125 the MTP draft context stages too (its KV follows `-ctk`), which made a `q8_0` draft KV cost +544 MiB
  against the f16 default it exists to save.  `fattn_stage_try_get()` therefore allocates
  `min(hint, 4x the request that just arrived)` (never below the 16 MiB floor): a short prompt stays at
  16-72 MiB, a deep one still reaches the hint in <= 3 steps.  Do not "simplify" it back to
  `max(16 MiB, hint)`.  `GGML_CUDA_FA_STAGE_PREALLOC=0` is the ladder A/B arm; `GGML_CUDA_SLAB_STATS=1`
  prints one `FA_STAGE … request=/cached=/alloc=` line per growth.  A change here is judged with
  `scripts/gate-prefill-logits.sh --kv q8_0` (the f16 default never builds the arena).  `WORKLOG.md`
  261027.007.
- **The decode/verify band skips fully masked KV groups (r133, PR #223, block 05).**  `GGML_CUDA_FA_BAND_SKIP`
  (default on) lets the band consume the packed fully-masked-group bitmap the prefill path already builds, so
  the band keeps its tuned round-robin KV split (`block i` owns iterations `i, i+P, ...`) and only leaves out
  the iterations whose logits are all `-INF` for every query row of its tile: each block's accumulation order
  over the rest is unchanged, a block that owns no unmasked iteration writes the same neutral `-FLT_MAX/2`
  partial it already had to represent, and the result is bit-identical.  The gate is the band's own predicate,
  so it is RDNA4 + head 256 + GQA 5..12 + `n_q <= 8`; the tile-kernel decode that every other model uses keeps
  the `Q->ne[1] > 8` gate, and `GGML_CUDA_FA_MASK_SKIP=0` disables the whole mechanism (this is its band arm).
  This is the cost `LLAMA_SERVER_LAZY_IDLE_SLOTS` (r130) made common: with a ~105K conversation resident in a
  unified cache, an active 9.8K conversation decodes at **50.8 -> 41.7 ms/step (-18 %)**, token ids identical
  (with a DFlash2 draft at `n_max 5`: 18.6 -> 14.8).  With nothing to skip it is not a tax (the padded tail
  group is skipped too: -0.6 % at 9.8K, neutral at 105K single-slot) — which supersedes the 2026-09-26
  "prefill only, break-even" reading, measured on a 2-slot batched cell with a small foreign fraction.  The
  op-test oracle is the **decode-width packed-hole** case set (`mask_hole=1`, `nb` 1/2/3/8, f16/bf16/q8_0);
  mutating the kernel's group test to skip one visible group fails all 14 of those cases.  Do not widen the
  gate to the tile kernel without that family's own hole cases and a measurement.  `GREEDY-PURITY.md` §37.
- **Idle-slot parking is the sequential-reproducibility fix for #210's lazy idle slots (r136, PR #225, block 12).**
  `LLAMA_SERVER_PARK_IDLE_SLOTS` (on unless `0`, with the lazy idle slots and `--kv-unified`) parks idle slots at
  the top of a unified cache and moves the running slot to the low cells, so `n_kv` and the cell indices match the
  save-and-clear layout; new cells and `get_n_kv()` stop below the parked stack.  It requires one stream, cells no
  other cache shares, V not transposed and no sliding window; anything else keeps #210 and logs once.  A slot whose
  cells are not one run (shared through `seq_cp`, e.g. a cached prefix) falls back to save-and-clear, or stays
  resident with `--no-cache-idle-slots`.  It does **not** make concurrent decoding deterministic (that is §38).
  `LLAMA_KV_PARK_DEBUG=1` logs every move and refusal.  **A memory whose second cache mirrors the attention cells
  parks both or neither (r146, issue #227):** `llama_memory_hybrid_idx` (qwen4exp/glm5-next) moves the QSA indexer
  cache with the attention cache, refuses when the two no longer mirror (`attn_idx_in_step`), and stales the pool
  layout + drops the derived pool watermarks (the derived rows are indexed by the block ordinal of the cells in the
  decode window, which a park changes).  Never park a cache whose cells another cache indexes by cell number without
  moving that cache — r136 matched hybrid-idx through its `llama_memory_hybrid` base and left the indexer behind,
  which aborted the server on the model's own guard.  `GREEDY-PURITY.md` §57; gate: `scripts/gate-park-idle-slots.sh`.

- **A context checkpoint that a task just restored is not saved again (r155, PR #236, block 12,
  `LLAMA_SERVER_CKPT_REUSE`).**  A prompt that resumes from a cached prefix used to restore a checkpoint and then
  write the identical state back at the same position before its first batch (and again near the prompt end); each
  save allocates, zero-fills and faults in a host buffer of the recurrent state's size (150 MiB on 27B, 174 MiB with
  its DFlash2 draft).  A slot remembers the restored checkpoint's `n_tokens`; when a checkpoint lands at that same
  position nothing has been decoded since, so it is kept (moved to the back of the list, the ordering a re-save
  produces) and the buffers of dropped checkpoints are recycled (4 in total, largest kept).  No arithmetic and no KV
  cell moves: greedy text is byte-identical, and only the TTFT of a resuming request moves (25-100 ms here, 195-223
  ms on the submitter's DDR4 host, which is page-fault bound).  **Two invariants for anyone editing
  `create_checkpoint()`:** the keep is valid only because the restore re-established *everything* a save would
  capture, so a new per-checkpoint payload must be restored (not only saved) or it must be stashed before the keep
  check; and the guard is `n_tokens_ckpt_restored` alone, so a new task-entry path that skips the
  `SLOT_STATE_STARTED` reset must set it (it is also cleared in `server_slot::reset()`).  The keep path only fires
  on a **hybrid/recurrent** target, whose memory reports `pos_min == pos_max` (one recurrent cell carries the last
  position); on a dense or SWA memory a restore lands at `pos_max` = `n_tokens - 1`, one token short, so nothing is
  re-saved at the same position and only the buffer reuse applies.  Gate: `scripts/gate-ckpt-reuse.sh`.
- **The band's two query-width blocks must share one geometry (r134, issue #224, block 05).**  A band block
  splits each Q column's KV rows `np = nwarps*cols_per_warp/ncols` ways (`cols_per_warp` is 16 for the RDNA4
  256-dim WMMA tiles), so the 32-column block (widths 1..4, 8 warps) is `np = 4` and the 64-column block
  (widths 5..8) is `np = 4` only at 16 warps.  r126 deleted the 16-warp row because it was also the prefill
  instantiation; leaving widths 5..8 on prefill's 8-warp table row (`np = 2`) is a different reduction and
  greedy decode with a draft then differs from decode without one (dense head-256 GQA-6, 8K and 110K).  The
  64-column row is back **band-only**, via the compile-time `band_wide` flag + its own case/fetch-ahead
  translation units; prefill keeps the 256-thread table row.  A `static_assert` on the two band rows (`np`,
  `nbatch_fa/K2/V2`, `nstages_target`, `Q_in_reg`) makes a repeat a build error -- do not "simplify" it away.
  And `test-logits-width-probe` must run at **`P >= 4096`**: the wide block's gate is `K->ne[1] >= 4096` and
  `K->ne[1]` is the context's used length, so the historical `P=256` run reported PASS without reaching the
  block (that is how r126..r133 passed).  The ratio axis is measured at **6**, **8** and **12** since r166:
  ratio 8 is `Qwen3.6-35B-A3B` (16 q / 2 kv at `n_embd_head_k` 256), the one *exact* whole-group fold, and a
  `rocprofv3` census shows the band's 64-column instantiation dispatching at widths 5..8 there.  A **gemma4**
  model is *not* a GQA-8 arm: its SWA layers are head 256 at ratio 2 and its full-attention layers ratio 8 at
  head 512, so neither enters the predicate (`Q->ne[0] == 256 && 4 < gqa_ratio <= 12`).  `GREEDY-PURITY.md` §51,
  `benchmarks/2026-10-25-fa-band-width-purity.md`, `benchmarks/2026-10-28-fa-band-gqa8-arm.md`.
- **A re-arm is LAYER-UNIFORM or the layer is stood down, and an arena range that is returned must not be
  kept (r168, TODO #70).**  `moe_cache_rearm()` re-allocates what a slab boundary move evicted, to the slot
  count the sizing pass settled on.  `moe_cache_rearm_layer_locked()` converges a layer on ONE count and, if
  the retries run out, stands the **whole layer** down (every table at 0 slots) -- never a mixed-slot layer:
  the gate+up fused kernel seeds the gate lane from the **up** table's remap, and the sizing pass answers
  that state by disabling the cache, which a re-arm must not do (the graphs already reference the tables).  A
  uniformly smaller count is accepted (with a warning); a final sweep re-checks the touched layers.  Two
  invariants this depends on: **a stand-down must clear `t.arena`/`t.arena_reserved`** once it hands the
  range back -- `ggml_cuda_slab_arena_free` puts it on the slab's free list where the work pool or another
  table can take it, and `alloc_table_locked()` reads a non-null `t.arena` as *already allocated* (r168
  found `stand_down_table_locked` keeping it, so a re-arm of an allocation-OOM-dropped table claimed slots in
  a range it no longer owned); and **any path that frees a table's device buffer must bump
  `ggml_cuda_graph_mem_freed`** -- `moe_cache_evict_slab_range` hands ranges to the slab directly and
  bypasses `free_arena_backing`, so it bumps for itself.  `MOE_EXPERT_CACHE_VALIDATE=1` reports the state:
  `MIXED-LAYER` (a role stood down, the served tables agree -- legal, `moe_cache_shrink_step` does it on
  purpose) and `NONUNIFORM-LAYER` (two different non-zero counts, an inconsistency).  Do not "simplify" the
  re-arm back to accepting `min(achieved)`.
- **A starved wide-aux growth is decided ONCE per wide pass (r168, TODO #70).**  `ggml_cuda_slab_wide_aux_alloc()`
  runs on every FA op of a prefill; a refusal used to be returned per allocation, so a starved region became
  hundreds of decisions, each able to evict a band of arena tables -- or evict one and then refuse on a
  boundary move that raced the MTP draft context -- under a queued graph that still held the previous
  placement.  `wide_aux_denied_want` latches the pass (cleared by a `ring_set` pass transition, a boundary
  shrink, or a slab (re-)init); a request at least as large as the refused one returns immediately, a smaller
  one may still fit.  The two refusal classes are separate: a **VA-limit** refusal is structural and latched,
  a **snapshot-race** refusal (the boundary moved while the lock was dropped for the eviction) is retried but
  bounded to `GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX` (4) attempts per pass -- each costs another arena-band
  eviction.  One WARN per pass when a pass is finally refused; `GGML_CUDA_SLAB_STATS` prints the per-pass
  count as `aux_refused=`, and the per-growth `SLAB_WIDE_AUX` line is unchanged (it is the sizing tool).  The shape is **not reproducible**
  since r87 forces the prefill -> decode drop, so it is validated by code review plus a healthy-run no-op.
- **The expert cache's tables must not outlive the model that registered them (r143).**  `g_tables` is
  process-global and keyed by host-master pointers the MODEL owns, so `llama_model_free` must drop them
  before the masters go: `llama_model_moe_cache_forget()` -> the `moe_cache_forget` device iface entry ->
  `moe_cache_forget_all(device)` frees each dropped table's arena slice, remap staging, heat mirror, bounce
  ring and host pool, re-points the id maps, forgets the per-model preflight accounting (`g_preflight_device_bytes`,
  the unregisterable set, the latched floor) and un-sizes the cache.  Without it the deferred arena sizing —
  which fills **every** registered table — reads a freed master: a host memcpy **SIGSEGV in
  `fill_slot_from_host`**, measured on Flash-Next GSQ 2 GPU `-sm layer -ts 59,41 -ncmoe 24` (llama-bench
  frees and reloads the model between tests whose params differ!) and in the same shape under
  `--host-experts pool` (whose master is the model mapping, so a host-buffer-free hook cannot cover it).
  Un-sizing matters as much as dropping: a second model used to inherit `g_sized` and stayed uncached for
  its whole run.  A new iface field that appends to `ggml_backend_device_i` goes at the **end** (the
  initializers are positional); a hook that must run at teardown needs a matching call in
  `llama_model_free`/the model lifecycle, not a buffer-free hook.  Single-model runs are unchanged.
- **The expert cache's per-token traffic figure and the graph-replay census are the two instruments for
  "where does the time go" (r144); use them before proposing a memory-side optimisation.**  The exit
  report's `expert H2D traffic … MiB/token` divides a **decode-band op census** (`g_band_ops`) by a table
  count snapshotted **at sizing** (`g_band_n_tables`) — never by a live `g_tables` walk (`llama_model_free`
  drops the tables before the report runs, r143) and never by `g_takeover` (0 on the identity fast path).
  Quote the **2000-token** number: a short run's MiB/token is mostly warmup (25.4 vs 3.3 MiB/token for the
  same config at 200 vs 2000 tokens).  `GGML_CUDA_GRAPH_TRACE=1` additionally reports
  `graph_compute calls: direct/replay/capture/properties_changed` — a **captured graph that never replays
  still logs captures**, so read the census, and confirm with `GGML_CUDA_DISABLE_GRAPHS=1` measuring the
  same speed.  Measured 2026-10-26 (2x R9700 x8, Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`, 85 %
  residency): the MoE miss traffic is 3.3 MiB/token = 0.12 ms of a 28.4 ms token (2.0 ms at the 2 048
  MiB/device floor) — so `TODO #36` (CPU/GPU overlap for the misses) was **retired**, while the token is
  dispatch-bound: 297 `graph_compute` calls / 12 258 node dispatches per token at 1.1 us/node = 13.5 ms of
  host time, with 93 % of calls refused replay (`direct=56171 replay=3152 properties_changed=55321` over
  200 tokens).  Do **not** "fix" that by relaxing the property comparison alone: it enables recaptures
  (`hipGraphExecDestroy`+`Instantiate` per token on HIP) and measures **slower** (28.3 vs 35.2 t/s).
  Records `archive/work/moe-cpu-overlap/`, `WORKLOG.md` 261026.001.
- **A graph-cache key must name every dimension a subgraph varies in, and a replay's property compare is
  the only safety net (r149, TODO #107).**  The r144 census above was a **cache-key collision**: under a
  split mode every layer's subgraph begins on the *same* activation buffer, so `(first node data,
  n_tokens)` sent all of them to one entry and each layer was compared against the **previous layer's**
  properties (which is why the first differing byte was `src[]`, offset 152, and why the "genuine `data`
  move" appeared only after canonicalising it — both were the collision).  The key now carries the
  subgraph's **weight identity** + a structural signature (`GGML_CUDA_GRAPH_KEY_WEIGHT`, on; `=0` restores
  the old key): 2x R9700 x8 Flash-Next IQ4_NL `-sm tensor -ncmoe 48`, 2000-token decode **37.7 -> 42.8
  t/s (+13.5 %)**, `properties_changed` 452439 -> 28; +12.0 % at 3 devices, +1.4-2.0 % with a draft, flat on
  1 GPU / 2-3-device dense / `-sm layer`.  **Never** add a fingerprint that moves across a rebuild (an
  object address, an allocation pointer, a generation counter) — that is the r115 trap from the other
  direction — and **never** weaken `ggml_cuda_graph_update_required` (an identity-free compare measured
  +0.2-0.3 % and is not shipped; an executable whose baked arguments point at re-placed buffers is a
  silent wrong-answer bug).  **The guard runs on *every* `graph_compute`, not only after a rebuild** — the
  scheduler mints a fresh `cgraph->uid` per split per compute (upstream #21764) and the Meta rebuild mints
  fresh child uids, so its `uid` early-out cannot fire (verified: `GGML_CUDA_GRAPH_DEBUG=1` prints no
  "reused" lines on a decode).  That is why its per-node cost is a per-token host cost, and since r158 it
  compares the live tensors **in place** (no temp struct, stop at the first difference) over a store
  **packed to the sources a node has** (`prop_nodes` / `prop_src_off` / `prop_srcs`; measured 1.4 sources
  per node, 70,752 -> 29,840 B per graph entry, `memcmp` 1.41 -> 0.58 % of a decode window).  The compared
  information is unchanged (the node header verbatim, then each non-null source's `data`/`ne`/`nb`) — do
  not "simplify" it back to a fixed `GGML_MAX_SRC` temp, and do not turn the packed store into a hash: it
  is the only check that a replay's baked arguments still point where they were captured.  The fold also
  builds the graph key **once** per call (it was built 2-3×, and `ggml_backend_cuda_graph_optimize` built
  and discarded one before its `GGML_CUDA_GRAPH_OPT` early return) and reads the device arch **once** in
  `ggml_cuda_graph_check_compability` instead of a `cudaGetDevice()` per `MUL_MAT_ID` node.  It is
  hygiene, not a win: on this box the host runs ahead of the GPU, so every arm measures flat (+0.15 % /
  +0.75 % within noise, +0.7 % even pinned to 2 cores) — the value is CPU time, bus traffic and ~100 MB of
  RAM on a host-limited box, so do not quote a t/s figure for it.  A captured graph bakes **raw device
  pointers that are not tensor sources**
  (the expert cache's arena slot table, remap staging, routing staging and both heat mirrors), so freeing
  any of a table's device buffers must bump `ggml_cuda_graph_mem_freed` — put it in the helper that returns
  the memory, not in the caller.  **Known loss:** a **3-device `-sm tensor` qwen35 dense plain** decode is
  6-8 % slower now that replay works (27B Q8_0 `tg128` 41.7 -> 38.6), and the axis is the **comm layer**:
  `GGML_CUDA_ALLREDUCE=internal` (RCCL never initialised) is **42.2, i.e. +0.9 %**, while `hybrid` and `ce`
  (which inits hybrid first) both regress, and the per-call arm choice (`GGML_CUDA_AR_SMALL_BYTES`) is
  irrelevant.  That is the r34 "RCCL's presence costs the internal pipeline" effect resurfacing, not a
  property of dense models: Gemma4 31B Q8_0 (dense, 32.6 GB) is +0.8 % at 2 GPUs and flat at 3, 27B at 2
  GPUs is +0.5 %, and a draft hides it (`TODO #109`).  Records
  `archive/work/graph-replay-keys/` (the PR assessment, the corrected diagnosis, and r158's
  compare rework with the follow-up that offered the relaxed compare) plus
  `benchmarks/2026-10-26-graph-replay-keys.md` (the numbers) and
  `benchmarks/2026-10-27-graph-replay-props.md` (r158's).
  Read the census as: `direct ≈ 0`, `replay ≈ calls`, `properties_changed` in the tens, `capture` = the
  number of distinct keys (hundreds — one graph per layer per width, not recaptures).
- **On 2+ devices `-sm tensor` is not just "the tuned path", it is worth tens of percent at low residency
  (r144).**  2x R9700 on the x8 root ports, Flash-Next IQ4_NL `-ncmoe 48`, 2000-token decode: **35.5 t/s
  (`tensor`) vs 25.7 (`layer`) = +38 %** at the same expert hit rate (99.74 % vs 99.48 %), so the cost is
  the per-layer cross-device split and its split/event/handoff boundaries (~50 graph splits/token), not
  expert residency.  `layer` is the CLI default, so a multi-device `-ncmoe` config must pass
  `-sm tensor` explicitly; a *dense* cross-device-count text comparison still needs `-sm layer`
  (see "Greedy text is a per-device-count reference").  `COMMUNITY-CONFIG.md` (the split axis).
- **The movable-boundary slab is MoE-only and must stay that way (r37).** `ggml_cuda_slab_enabled()`
  returns `env_on && g_slab_armed`; `g_slab_armed` is set **only** by the MoE preflight
  (`ggml_backend_cuda_device_moe_cache_preflight`), which `llama_model_moe_cache_preflight` calls only for
  models with a non-empty `model->moe_host_expert_bytes` (`-ncmoe`/`-cmoe` > 0).  The slab exists to let the
  work region and the expert-cache arena coexist; with no arena it parks nearly all free VRAM in an unused
  region (measured ~18 GiB/GPU on a dense 27B).  The **compute chunk** (`GGML_COMPUTE_BUFFER_CHUNK_MIB`) is
  slab-motivated and is gated on the same condition.  Do **not** re-enable either for dense models; the
  dense regression is issues #118 (slab before the weights OOMs a 48 GiB card under `--mmproj`) and #120
  (the chunk cost a dense config +739 MiB and tipped it into shared memory).  The percentage margin
  (`GGML_COMPUTE_BUFFER_MARGIN_PCT`) is deliberately *not* gated (kept for dense).
- **An upload that writes a scheduler-managed buffer goes on the main stream (r62, block 10).**  The
  scheduler's `wait_before_overwrite()` (a wait on the previous split's event) guards the graph
  allocator's reuse of a split-input buffer and is enqueued on the **main** stream.  The compact
  splice-gather's 1-D H2D ran on `cuda_ctx->copy_stream()` and only ordered the *consumer*, so it could
  overwrite the buffer while the previous split still read it -- garbage/NaN for `-sm tensor` + `-ncmoe`
  on the below-gate path (35B-A3B, gemma4, Flash-Next GSQ).  A side stream is only safe with an explicit
  wait on the main stream recorded *before* this upload; the gather has no such need (it must wait for
  the reused-buffer read anyway).
- **The async graph-input ring (`GGML_CUDA_ASYNC_INPUT`) is opt-in and a run is not start-to-start
  deterministic when the consumer is a captured HIP graph** (r69; re-characterised 2026-10-19).  `-ub 8`
  and `-ub 1` self-KLD comparisons diverge with `n_seq = 1`; `-ub 512` and `GGML_CUDA_DISABLE_GRAPHS=1`
  are bit-identical.  **The cause is NOT the ring**: the uploads verify byte-correct at the graph's first
  read, HIP's wait-before-replay / event-after-replay ordering is correct, and consecutive graphs do not
  overlap — so the async path is the *trigger* (it removes the per-copy host block), not the defect.
  **FIXED in r90 (TODO #73 closed)**: the defect was `ggml_backend_cuda_set_tensor_async` /
  `_set_tensor_2d_async`, which issued a pinned-source `cudaMemcpyAsync` **straight from the caller's live
  host pointer** while the scheduler uses that path for a host-resident split input -- a CPU split's
  output that the next ubatch rewrites in place -- so with the host running ahead the copy could read the
  next ubatch's bytes (a rare single-graph corruption that propagated to the end of the chunk).  Both now
  snapshot through the pinned upload ring, unconditionally, and the ring's wait/done guard is
  unconditional; `GGML_CUDA_ASYNC_INPUT` is **default-on** (`=0` disables) and the batched uploads
  (`GGML_CUDA_ASYNC_INPUT_BATCH`, r89/PR #182) stay default-on inside it.  Self-KLD control 91.0-95.1 %
  -> 99.967-100.000 %; decode +9.3 % (35B-A3B 2 GPU tensor tg400) / +5.2 % (27B MTP n3 end-to-end).
  Evidence: `archive/work/async-input-graph-ordering/` (`RESOLUTION.md`, `FINDINGS-2026-10-19b.md`).
- **The target's speculative checkpoint lives in device memory when one GPU has room, and that checkpoint is NOT a
  byte image (r184, PR #254, block 12).**  On a recurrent/hybrid target the server saves the target's state before a verify
  whose draft is longer than `n_rs_seq` (`use_ckpt_tgt`) and restores it on a partial acceptance; since r184 it carries
  `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` when the target is on exactly one GPU, its state is offloaded and the GPU has the
  state's size + 512 MiB free (decided once per slot, logged, `LLAMA_SERVER_SPEC_CKPT_DEVICE=0` = host always; a
  multi-GPU split and `-nkvo` stay on the host path).  **The device arm stores only the metadata in `data_tgt`** - the
  payload is the context's per-seq staging buffer (`llama_context::mem_storage`), which the next save overwrites - so a
  save must be restored before the next save for that sequence (the spec path does: a draft round saves once and only
  restores that save, and the replay rounds save nothing), and one of these must never reach anything persistent.  A
  host-flagged save (a context checkpoint / the prompt cache) does not touch the stage and interleaves freely.  The win is
  (host save cost) x (checkpoints taken) and therefore host-CPU dependent: measured 1 GPU on this box 27B + DFlash2 rewrite
  +2.1 %, 35B-A3B `-ncmoe 40` +4.3 %, plain prose 0 % (the submitter's slower host measured +10.0 %/+15.8 %); +150 MiB
  VRAM per used slot on the 27B and nothing in the `--fit` estimate.  Gate: `scripts/gate-spec-ckpt-device.sh`.
  `benchmarks/2026-10-30-spec-ckpt-device.md`.
- **The batched state save and restore share one pinned `state_stage` per context (r96/r119, PRs #189/#209), the
  copy is pipelined (r151, PR #233), and its host side is streamed on two threads (r152, PR #235).**
  `LLAMA_STATE_ASYNC` covers saves, `LLAMA_STATE_ASYNC_READ` (default on) covers restores, and
  `LLAMA_STATE_ASYNC_MAX_MIB` bounds both.  Saves and restores on one context must not run concurrently.
  The restore is a multi-device win (2 GPU `-sm tensor`: 35B-A3B 49 -> 40 ms, Flash-Next 89 -> 75 ms);
  one GPU pays about 2 ms for the extra staging copy, so do not read the single-GPU number as a defect.
  `LLAMA_STATE_PIPE_GROUPS` (default **8**, `1` = the unpipelined copy) cuts the pieces into size-balanced
  groups so the host memcpy overlaps the device copies of another group: a restore packs a group and queues
  its async sets before packing the next (the groups own **disjoint** stage ranges - a group that overlapped
  the next one's pack with a live copy would be a torn state), and a save records one **per (backend, group)**
  event after that group's copies are queued and copies the group out when its event completes (a re-recorded
  event would move past the earlier one).  A backend with no events (the CPU backend) falls back to the single
  `llama_state_sync_all` path.  Measured -3.6 % (Flash-Next + MTP, 2 GPU tensor) and -2.9 % (35B-A3B, 1 GPU)
  on a 5246-token prompt-cache restore.  `scripts/gate-state-pipe.sh` is the gate, and it is the only one that
  reaches this code (the file-based slot save/restore uses `llama_io_write_file`/`read_file` and never touches
  the stage); a torn state is a wrong answer, not a crash.
  **r152's rule: a copy worker must never leave its pipe group.**  Each piece is copied with non-temporal (SSE2)
  stores over 64-byte-aligned runs, split over `LLAMA_STATE_COPY_THREADS` (default 2; `1` = one thread, still
  streamed; `0` = the plain pre-r152 `memcpy`, the A/B arm), and the split is *within* one group: a group's
  stage range may only be read after its event, so a worker that took the next group's pieces would read a
  range whose device copy is still in flight (the same failure the per-group events exist to prevent).  A
  streamed run ends with `_mm_sfence()` before its bytes reach the caller (save) or the device (restore);
  states below 16 MiB keep the plain copy and pieces below 64 KiB keep the memcpy.  The streaming store is
  worth ~36 % of the copy where the checkpoint does not fit in L3 (the submitter's 3900X + DDR4) and ~0 where
  it does (this box: 192 MiB of L3), while the threads are worth ~2x on both - so do not "simplify" it to one
  thread on the strength of the streaming-store half alone.  Measured 39.9 -> 33.4 ms (35B-A3B 1 GPU),
  44.2 -> 40.5 ms (2 GPU tensor) and 79.3 -> 74.9 ms (Flash-Next + MTP) of the server's save+load trace.
  `scripts/gate-state-pipe.sh` gains a **`copy0`** arm (`LLAMA_STATE_COPY_THREADS=0`) that is gated
  byte-for-byte like every other arm; `LLAMA_STATE_TIMING=1` is the per-op diagnostic.
  **The device-state class (`llama_io_write_device`/`read_device`, `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`) is a
  separate path and copies in bands through that same stage (r157, TODO #110).**  A piece must be handed to a
  meta backend as **the tensor plus an offset**, never as a fresh `ggml_view_1d` of it, and a band must be a
  run of **whole** pieces: a view's inherited split state makes the backend splice the piece row by row (4.5 ms
  vs 0.15 ms for 2 MiB), and the backend only accepts a window aligned to the tensor's own layout (a cut piece
  trips `size % row_stride == 0`).  Per-piece `ggml_backend_tensor_copy` between two meta buffers is ggml's
  pageable bounce and measured 58x slower than the host path; `LLAMA_STATE_STAGE_MIB` (default 64, `0` = old
  path) sets the band.  That class is reached by nothing but `tests/test-llama-context` (`cp_d`/`cp_d_s`), which
  is its oracle; `gate-state-pipe.sh` does not reach it.  `benchmarks/2026-10-27-state-device-path.md`.
- **The op-offload H2D staging ring (V2.0 block 09; old block 06, r12) has three invariants:** (1) prefill is never
  CUDA-graph captured, so a redirected split input that reaches a copy path aborts (tripwire asserts
  this); (2) the width gate is floored at **64 tokens**; (3) the arena lives **outside** the
  compute-graph reserve, so a failed allocation disables staging instead of aborting, and a partially
  staged ubatch is never allowed. Under `-sm tensor` the meta backend owns one ring per device and
  `MIRRORED` tensors must serve arbitrary byte ranges. The expert weights themselves are **split** per
  device, not mirrored (`GGML_META_SPLIT_COPY=1`; each cache table holds `expert_bytes = host_bytes/2`,
  measured 2026-10-07), so any `-sm tensor -ncmoe` vs `-sm layer` gap is the per-op upload/pruning
  machinery, not weight duplication. (`TODO.md` #44 was closed on that finding; see
  `archive/work/expert-cache-split/`.)
- **Cross-device split inputs (V2.0 block 09; old block 06, r12):** a device-to-device input copy runs on the SOURCE
  backend's stream, so `wait_before_overwrite` (which only orders the destination stream) is not
  enough; the block records a fresh event on the destination backend and makes the source wait on it
  before the copy. Without it, `-sm layer` on 2 GPUs with host experts (`--n-cpu-moe` >= 29) can
  overwrite an earlier split's output before its outbound copy reads it (garbage prefill);
  `GGML_SCHED_EVENTS=0` is the old workaround. Issue #103, contributor PR #104,
  `archive/work/2gpu-sched-fixes/VERIFICATION.md`.
- **The QSA op has a CPU oracle** (`test-backend-ops -o FLASH_ATTN_QSA`, 18/18 minimum). The `W=1..8`
  matrix is **blind to a width-uniform corruption**, and MTP acceptance is not a quality signal when
  draft and target share the defect; use the perplexity ratio vs the dense masked path
  (`LLAMA_QSA_SPARSE_FA=0`). A fused op with several heads sharing a staging buffer must keep the block
  homogeneous in every index the staging reads (QSA: the K/V head). `GREEDY-PURITY.md` §21.
- **Block 03 (GDN; old block 02):** the chunked prefill is K-independent with threshold `max(K>16?K:16, n_rs_batch)`,
  `n_rs_batch = common_speculative_n_max()+1`; `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel
  (correct, bit-identical, slow). The pure `none == draft-mtp` range is `n_max <= 7`.
  **Our chunked kernel is the only one the delivery dispatches, on every device class** (r127 gfx1151
  A/B): upstream's `gated_delta_net_mma` (in the 2026-10 window) *is* eligible on RDNA3.5 at
  `n_tokens >= 2048` (qwen35's `H_k = ssm.group_count = 16` satisfies its gate) but our branch returns
  first — measured gfx1151 pp2048 interleaved **3232 vs 3131 t/s (+3.2 %)** for ours, prefill-logit
  **KLD 0.000074 / 99.637 %**.  Upstream's MMA only runs behind our stand-down (`kda`, `S_v` outside
  {16,32,64,128}, or the `=0` kill switch): never add a second chunked kernel per device class.
  `GREEDY-PURITY.md` §27, `patches/README.md` (block 03), `WORKLOG.md` 2026-09-12.
- **The rollback bound is enforced by `n_rs_batch`, not observed — do not re-derive it.**  `common`
  sets `cparams.n_rs_batch = common_speculative_n_max() + 1` over **every** speculator type (draft
  simple/mtp/adaptive/eagle3/dflash/dspark *and* ngram simple/map/mod/cache, so ngram-mod's default
  n_max 64 gives 65), `llama_context` floors it at `n_rs_seq + 1`, and `delta-net-base` passes it as
  GDN op param 1.  The op's threshold is floored at it, so a **speculative verify batch can never take
  the chunked path** and always restores snapshots it wrote (r91; the campaign's "ngram is an exposure"
  claim was wrong).  The r91 sequential tail (`GGML_CUDA_GDN_TAIL`, block 10) then covers the remaining
  source of a partial rollback: a **non-speculator** rewind into a prefill batch (server cancel /
  `--cache-reuse`).  The per-sequence tracker is block 03's (`prepare()` folds in every ubatch;
  `LLAMA_RS_ROLLBACK_DEBUG=1` shows `RSPREP`/`RSROLL`).  Record + probe:
  `archive/work/rs-rollback-boundary/`.
- **A recurrent cell that `find_slot` moves keeps its rollback snapshot planes (r195, PR #261, block 03).**
  `build_rs` relocates the "extra states" (the cells inside `[head, head + n_rs)` that no sequence of the
  ubatch uses); it used to copy **plane 0 only**, so from 3 cells on a moved cell's planes `1..n_rs_seq`
  stayed at the old index and were overwritten by the cells that took their place, and that sequence's next
  rollback read another sequence's snapshot: silent wrong text on a multi-slot server, no log line.  It now
  also copies planes `1..n_rs_seq` through the new input `s_copy_extra`, gathered **before** the ubatch's
  snapshot writes (**node order is the only ordering guarantee**: two views of one tensor create no
  dependency edge), with the ids from `s_src()` because `s_copy()` *consumes* the sequence's pending
  rollback index.  Where `n_rs == n_seqs` (every single-sequence batch, and a 2-cell cache) the graph does
  not change, so every single-sequence gate (coherence, prefill-logit, the four-axis MTP protocol) is blind
  to it, as is text in general (the wrong read is still fluent).  The defect is **upstream's** (`6050b467c`
  and master `0ec4027a2` both carry it), so the same change is kept as an upstream candidate.  Gate:
  `test-recurrent-state-depth`'s **Phase C** (two-arm replay-logit comparison; 15/15 arms differ by
  2.94-9.64 logits on r194 and by 0 on r195, 1 GPU and 3 GPU `-sm tensor`).  Record:
  `archive/work/recurrent-moved-cell-snapshots/`, `GREEDY-PURITY.md` §79.
- **Test baselines are per box *and* per model: never compare to a number in this file.**  E.g.
  `test-recurrent-state-depth` on Qwen3.6-35B-A3B-UD-Q3_K_M: **130** failures on gfx1201, **153** on
  gfx1151; the gate is that the per-arm FAIL/PASS set is unchanged, not that the count is zero.  Run it
  (and any other GPU test) **alone** — two GPU jobs at once changes the arena/band layout and invents
  failures.
- **Block 01 (all-reduce; old block 12):** the `GGML_CUDA_AR_PROFILE` device-list init fix; on the first NCCL runtime failure the
  comm layer warns once, disables NCCL and re-routes to the internal pipeline/butterfly.
  `GGML_CUDA_ALLREDUCE=nccl` is **NOT bit-identical** under `-sm tensor` (the internal AR BF16-round-trips).
  Since r165 the Linux/HIP default is `auto`: internal pipeline for the decode/verify band, and the
  **large-tensor arm chosen once at the first large all-reduce** — RCCL's ring when the model offloads
  experts to host memory (`g_comm_host_experts`, armed by the slab block), the copy-engine arm otherwise
  (`GGML_CUDA_ALLREDUCE=hybrid` = the pre-r165 P2P chain, `ce` = the copy-engine half).
- **Never enable peer access on the delivery's own initiative (r165, TODO #109 closed).**
  `hipDeviceEnablePeerAccess` charges the **host submission path**, not the GPU: with the AR path held
  constant, 3x R9700 27B Q8_0 `-sm tensor tg128` is 42.25 with no peer access and 38.77 with it; a per-token
  synced loop is *equal* (41.1 vs 41.4), the decode kernel census is identical, and a 3-device probe sees
  0.0 % on bandwidth/ALU/per-dispatch cost.  It is **not reversible** — after `ncclCommInitAll` has built
  its P2P mappings, every `cudaDeviceDisablePeerAccess` returns `no error` and the cost stays (37.9), so a
  chain that does not want it must not enable it (`NCCL_P2P_DISABLE` before `ncclCommInitAll`).
  Consequences: an arm carrying peer access is an 8-9 % dense-decode cost whenever the host is the limiter
  and graphs replay (invisible to the direct node loop); a *kernel* microbenchmark cannot see it, so judge
  a change with a synced-vs-async pair; and the large-arm choice is a **prefill-numeric** change gated by
  `scripts/gate-prefill-logits.sh --sm tensor --device 0,1,2` (the gate cannot reach an all-reduce
  otherwise), never by greedy text.  `GREEDY-PURITY.md` §67,
  `benchmarks/2026-10-28-allreduce-peer-access.md`.
- **Block 08 MTP regression fixes (old block 13):** the dense K-split kernel for verify batches, and the
  rms_norm->mmvq Q8_1 fold gated to single-token MMID + plain `MUL_MAT`. **Purity ranks above raw
  non-MTP throughput.** `GREEDY-PURITY.md` §19.
- **MoE decode/verify is byte-identical by default:** the fused shared-expert band serves
  `1 <= nt <= 8` with `nwarps` pinned; `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` is the A/B kill-switch.
- **The RMS-norm reduction order is a numerics reference, not a knob (r145, issue #229).**
  `ggml_cuda_rms_norm_vec()` (`common.cuh`) is the single selector: **default OFF = the pre-#29720 strided
  scalar reduction**, `GGML_CUDA_RMS_NORM_VEC=1` = upstream #29720's float4 grouping (which the r127 re-base
  had made unconditional).  A reduction-order change moves every layer by ~1 ulp and greedy MTP draft
  acceptance by a payload-dependent amount (issue #229: -7..-9 % tg on the reporter's box; +0.7..+1.3 points
  here, but 0.611 vs 0.792 acceptance / -18 % t/s with the SSM decode fusions disabled), so float4 is worth
  **+1.0-2.7 % prefill and ~0 % decode** and is not a shipping default.  The strided reduction is also what
  every decode-band fused norm kernel is documented bit-identical to (the rms_norm->Q8_1 fold,
  `rms_norm_scale_pair_f32`, `rms_norm_mul_rope_f32`, gdn-conv's folded q/k norms, the GDN norm gate,
  `hc_combine_norm_f32`) - under the float4 default three of them were silently *approximate*
  (0.0005 mean KLD at `-ub 1` where the contract says 0).  A new fused norm kernel owes that probe
  (`scripts/gate-prefill-logits.sh --ab "<fusion env>" --ub 1`, must read 0.000000 / 100.000 %), and every
  kernel standing in for `rms_norm_f32` must consult the selector rather than hard-code a grouping.
  `GREEDY-PURITY.md` §60; `ENVIRONMENT.md` §7.5; `benchmarks/2026-10-26-rms-norm-float4-ab.md`.
- **The r25 tensor-split fold (PR #106 minus its patch 0002) has three invariants to keep together:** a
  retired **Q8_1 arena must outlive captured graphs** (`GGML_CUDA_Q8_1_ARENA_FREE_OLD=1` restores the
  `quantize_q8_1` page fault); a graph is **recaptured when the pool / FA-staging / H2D-ring memory it
  captured is freed** (`GGML_CUDA_GRAPH_MEM_GEN=0` disables); and the **meta split-state cache keeps 8
  versions with compact entries** (`GGML_META_SS_VERIFY=1` verifies hits).  **PR #106 patch 0002 (the
  data-pointer graph key) is deliberately NOT in the delivery**: on the r22+ movable-boundary slab the
  per-device allocations vary between otherwise-identical calls, so the extra key invalidates warm graphs
  and recaptures (measured -19 % `-sm tensor` MTP at `--spec-draft-p-min 0.5`, with identical text and
  acceptance).  Re-cut it against the slab before revisiting.  `TODO.md` #46, `WORKLOG.md` 2026-10-07 (r25).
- **A captured graph is only valid while the slab work region's layout is unchanged.**  The work region is a
  movable boundary: at the post-prefill drop the narrow re-reserve shrinks it and the bytes above the new
  boundary go back to the MoE arena.  A captured CUDA/HIP graph bakes the work-region addresses into its
  executable graph, so the shrink **must** bump `ggml_cuda_graph_mem_freed(device)` (block 10,
  `ggml_cuda_slab_work_alloc`) — otherwise a later replay reads released memory.  Issue #187: a 4-stream MTP
  decode replayed a graph whose KV `set_rows` source indices pointed into the released wide region and
  faulted with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`; `GGML_CUDA_DISABLE_GRAPHS=1` hid it.  Every
  shrink path taken at a drop still needs this; add it wherever a new one is introduced.
- **Under `-sm tensor` the HIP graph cache must be keyed on a *stable* identity — the Meta backend
  re-creates its per-device simple tensors on every rebuild (r115, TODO #51).**  `ggml_cuda_graph_get_key()`
  is `{ first node's `data` pointer, n_tokens }` and must stay that way (`GGML_CUDA_GRAPH_KEY_DATA=0`
  restores the old object-address key, A/B only).  With the object address the cache never warmed: 3954
  keys / 1206 captures / 484 ms of `hipGraphInstantiate` per 300-token decode at 3 devices vs 67 keys / 9 ms
  with `-sm layer` — the user-visible 200-450 ms 3-GPU stutter, and the reason 3 GPUs decoded slower than 2.
  Keyed on `data`: 144 keys / 15 ms and **+41 %** at 3 devices (+6 % at 2, unchanged at 1), identical text
  and acceptance.  Any new *fingerprint* added to the key (an object address, a generation counter, a
  per-allocation pointer) re-introduces the churn if it moves across a rebuild — the r25 PR #106-0002
  data-fingerprint key that was dropped for the slab is the same trap from the other direction.
  **Since the `3f03b1772` re-base the captured graph is a *vector* of per-chunk executables**
  (upstream's chunked CUDA graphs, #30804): `ggml_cuda_graph` has no `graph`/`instance` members, capture
  ends inside `ggml_cuda_graph_launch_chunk()`, and anything that used to run "just before
  `cudaStreamEndCapture`" (the WIP AR-stage hook) now runs before the **last** chunk's launch.  The HIP
  `hipGraphExecUpdate` leak workaround (`GGML_HIP_GRAPH_FORCE_UPDATE`) lives inside upstream's
  `ggml_cuda_graph_update_executable(cudaGraphExec_t &, cudaGraph_t)`, i.e. it re-instantiates one chunk
  per recapture.  3-GPU 27B MTP decode measured 84.2/84.8 t/s at rung 8 vs 79.8 before it (+5.5 %),
  text unchanged.
  Record: `archive/work/graph-key-churn/`.
- **There is NO periodic per-device decode stall left after r115 (TODO #89 closed, 2026-10-22).  Do not
  re-chase it.**  The ~300 ms that `CUDAGC_SLOW` still reports at 3 devices is a **one-time lazy HIP
  code-object load per (module, device)** (~55 ms of it is the `mul_mat_q` object, which holds mmq *and*
  mmvq), paid inside `common_init_from_params`'s existing 2-token warm-up for the decode families and
  inside the first real prefill for the prefill-only ones.  The count is identical at 300, 1000 and 1500
  generated tokens — it does not scale with tokens, only with device count, and `--no-warmup` moves the
  same cost onto the user's first step (which is what the warm-up exists to prevent).  A first use
  inflates the whole window (`RMS_NORM` 4.7 ms right after the FA load) because further enqueues block
  behind the load; `GGML_CUDA_DISABLE_GRAPHS=1` changes nothing.  `CUDAGC_SLOW`'s 5 ms threshold is per
  child-graph launch and is **not** device-count normalised, so on 1 GPU ordinary 4 000-node steps are
  reported — judge per node (~1-2 us) or by `FIRSTUSE`.  Also measured there: `hipFuncGetAttributes` in
  these runs is **librccl** (NCCL init), not ours.  Tooling (probe extension + an `LD_PRELOAD` HIP
  call-site interposer): `archive/work/decode-warmup-stall/`.
- **A fused kernel that replaces a ggml chain must call that chain's transcendental op VERBATIM, and a
  decode-band fusion owes its purity check at the VERIFY width, not only at `--ub 1` (r201, issue #265).**  Both
  SSM gate/beta fusions computed `logf(1.0f + expf(x))` while `unary.cu`'s `op_softplus` (the node they replace)
  computes `log1pf(expf(x))`: a 0.000562 KLD divergence at `-ub 4` that **moved the target's own greedy text**
  (4771 `sha=0ac05e4cb358` fused vs 4628 `sha=8631f9a3a623` chain) and its MTP acceptance, and it survived r183
  through r200 because the r183 gate's `--ub 1` arm is **vacuous** here -- at one token the SSM pre-scan fusion
  owns those nodes on qwen3.5/3.6/3.8, so the fusion under test never ran.  So for any change to a decode-band
  fusion: `gate-prefill-logits.sh --ab "<its kill switch>=0"` at **`--ub 1` AND `--ub 4`**, plus
  `test-logits-width-probe P=4096` on the arms, and extend the type-correct unit oracle rather than trusting a
  sibling (`test-fusion-gate-beta-multiseq` covered BF16-at-1024 while every delivered GDN model runs Q8_0 at
  K=5120).  **`GGML_CUDA_FUSE_GATE_BETA_VERIFY` gates only the VERIFY band** -- the `ne[1] == 1` decode arm is
  unconditional -- so pre-r201 the *default* build ran the wrong-softplus kernel at every width: width-uniform,
  0-diff against itself, and wrong (issue #263's gfx1100 acceptance delta; the gfx1100 default arm moved
  977/710 `sha=eca29f334d6a` -> 988/704 `sha=152af1d9feb7` with the fix, and its fuse-off arm now reads the same
  numbers -- `v16-6050b467c-r202`).  So judge a fusion against **`GGML_CUDA_DISABLE_FUSION=1` (the chain)**, never
  against the default, and treat a width-uniform PASS as no evidence of purity when the switch covers only part
  of the band.  That also retires the "RDNA3_0 kill-switch arm is not width-uniform" residual: it was this same
  softplus, and a fuse-off A/B on gfx1100 is a clean control again.  `GREEDY-PURITY.md` §81,
  `benchmarks/2026-11-01-ssm-softplus-265.md`, `benchmarks/2026-11-01-kill-switch-arm-130.md`.
- **The GDN decode folds are a 1-GPU win (r183, block 10, PR #256).**  The direct GDN conv now also writes the
  recurrent **snapshot tails** (`GGML_CUDA_DISABLE_GDN_CONV_SNAP=1` off) and reads the **conv-cache row itself**
  for `build_rs`'s deferred `GET_ROWS` (`GGML_CUDA_FUSE_GDN_CONV_GATHER=0` off), and the 1-token direct conv
  ships gated on `gdn_conv_norm_match() > 0` + `ssm_prescan`-first (`GGML_CUDA_FUSE_GDN_CONV_DECODE=0`), which is
  what lets the two folds fire at one token.  Worth **+5.9 % tg128 on gfx1100 / +3.3 % on gfx1201** (35B-A3B
  Q4_K_M), +1.7 % dense 27B, neutral on 4B Q8_0; decomposed on gfx1201 the enabler is +0.3 % and the folds are
  +2.2 % (snapshot) and +1.0 % (gather).  Two rules: the decode gate reads **data pointers**, so it lives in
  `ggml_cuda_gdn_conv_match_at_concat` / `_at_conv` and `graph_optimize` may only use the structural
  `ggml_cuda_gdn_conv_pattern_at_concat`, which must stay a **superset** (its alloc deps then cover every case);
  and a `CONCAT` over a deferred gather the direct conv does not take **aborts** by design.  All three ride the
  direct conv's single-device gate, so a 2+ device `-sm tensor` / `-sm layer` run (the recommended MoE server
  shape) is **inert** unless `GGML_CUDA_CONV_FUSION_MULTI=1` — do not quote them as multi-GPU wins.  **(r186
  removed that restriction:** the 3x R9700 `-sm tensor` divergence that motivated it no longer reproduces —
  bit-identical post-prefill logits (P=4096) and greedy text on 1/2/3 devices, `-sm tensor` and `-sm layer`,
  dense and MoE including the `-ncmoe` server shape, and the prefill-logit gate reads 0.000000 / 99.990 % —
  so the fusion and all four folds now run on every split, worth **+5.7 % tg128 / +1.1 % pp2048** on 3 GPU
  `-sm tensor` 27B IQ3_S and **+3.0 % tg128** on 35B-A3B Q4_K_M.  `GGML_CUDA_DISABLE_CONV_FUSION=1` is the
  A/B arm and `GGML_CUDA_CONV_FUSION_MULTI` is retired; the 3-device divergence is not root-caused, so
  re-gate with release gate 14's **multi-device** arm before touching the gate.  The
  shared-expert gate fold (`GGML_CUDA_FUSE_SHEXP_GATE_SIGMOID=0`) is the fourth patch: it needs a BF16/F32 gate
  weight, is inert on every quantized model we hold and is +0.6 % on Flash-Next.  Gate: the prefill-logit `--ab`
  at `--ub 1` (0.000000 / 100.000 %), text identity on a single device *and* a 2-GPU `-sm tensor -ncmoe` arm,
  and identical MTP draft/accept.  **Judge a change to these folds with `test-logits-width-probe` at
  `P=3584` and `P>=4096`, not with the `--ub 1` arm**: the folds change a multi-token step's launch order
  and nothing else, so text and the single-token KLD are blind to them — r183 shipped them with only those
  two arms, and the probe is the instrument that held PR #177 out for four releases (35B-A3B P=3584 was
  run-to-run unstable with clean text and clean KLD; that class, TODO #75, is now closed — r185 re-check
  reads `snap=1` and identical `W8`/row-0 hashes against the kill switch on 35B-A3B and dense 27B at
  P=256/3584/4096).  `scripts/validate-gpu.sh` step 4c and release gate 14 (`MANIFESTS.md`) run it; a run
  that never logs `snap=1` is a vacuous pass and fails.  `GREEDY-PURITY.md` §73,
  `archive/work/gdn-decode-folds/PROMOTED.md`.
- **A 2-D memcpy of a mirrored weight's view can be refused inside the slab (r183, block 10).**
  `ggml_cuda_cpy_as_memcpy_2d` (upstream #25057) keeps its `cudaMemcpy2DAsync` path only while the padded span
  (`height*pitch`) fits the tensor's own allocation.  A 2-D copy of a *view of a mirrored model weight* is
  rejected with `invalid argument` when the weight sits in the movable-boundary slab, which made
  `Flash-Next GSQ-IQ3_XXS -sm tensor -ncmoe 48` abort at load on r182 and earlier (`ggml_cuda_cpy`, device 1,
  qwen4exp's `blk.1.ple_conv1d.weight` column view; `-sm layer` and `GGML_CUDA_SLAB=0` both loaded).  The exact
  driver rule is unverified (shrinking the 2-D copy by one row still failed), so the guard is deliberately
  conservative and **stays**.  What it does *not* leave alone is the fallback: r183's "no other model reaches
  the path" was wrong for Flash-Next, which with a **quantized indexer key cache** (`-ctk q8_0`; the fused
  `INDEXER_SCORE`/`INDEXER_FILL` path is F32/BF16/F16 only) takes qwen4exp's per-op chain and makes one refused
  copy per layer per prefill chunk — and `cpy_scalar` costs three divisions and three modulos **per element**
  (1010 ms of a 105k prefill).  Since **r189** the refused F32/16-B-aligned case is copied by `cpy_2d_vec16`
  (one 16-byte unit per thread, row from the grid, bit-identical): 1010 -> 637 ms there, wall time unchanged on
  our arms (a **regression fix**, not a speed-up — the campaign's 5.20 -> 1.83 ms/token is its sparse/pool
  decode shape, which no delivery-only arm reproduced).  "The guard refused" is an **explicit flag**, never
  inferred from a partially-filled geometry: the `spitch < width || dpitch < width` rejection is a different
  case (overlapping rows need the flat mapping) and must not reach a row-per-row copy
  (`GREEDY-PURITY.md` §76).  Do not restore the unconditional fast path without reproducing the arm, and do not
  narrow the guard without the driver's actual rule.
  `archive/work/cpy-2d-memcpy-guard/README.md`, `archive/work/cpy-2d-vec16/README.md`.
- **A `-sm tensor` model load uploads tensors through a pinned staging buffer, one alloc/free per
  tensor** (`ggml_backend_cuda_buffer_set_tensor_2d`, 678 pairs = 2.2 s + 1.1 s of a ~10.7 s load on a 27B
  Q8_0 at 3 devices).  This is *load* time, not decode.  Since **r116** the staging buffer is grow-only
  per device (~5 allocations, 12.7 s -> 9.5 s, block 08; `GGML_CUDA_STAGE_PIN=0` is the kill switch).
- **The deepseek4/glm5-next `DSV4_HC_POST` arm is split-mode dependent (r97).**  `cparams.fused_dsv4_hc_post`
  is the fused **op** whenever a Meta (tensor-split) device is in play and the primitive
  `hc_combine_norm` chain on one device.  The chain is faster on one device (+0.9..+1.8 % prefill,
  gfx1151) but is **not correct across a tensor split**: `test-llama-archs -a deepseek4` on 3x gfx1201
  read NMSE 1.74e-01 through the chain vs 6.5e-06 through the op.  Do not make the chain unconditional
  again without fixing its split-state handling; `LLAMA_FUSED_DSV4_HC_POST` forces either arm.
- **A host buft with no device is never accounted as host experts (r97).**  `ggml_backend_cpu_buffer_type()`
  has `.device = NULL` upstream; the account map, the slab, the H2D ring and the expert cache all need a
  **pinned** master read in place by a device kernel, so a pageable placement (CPU-only, or
  `--override-tensor exps=CPU`) is left to the upstream `llama_moe_cache` engine.  Keying a null device
  aborted in `ggml_backend_dev_memory()`; keying the layer's device reserves a ring/arena the path cannot
  use.  `patches/README.md` (block 09).
- **Dense greedy purity:** `plain == draft-mtp` greedy text is kept for **f16/bf16/q8_0**; the coarse
  quants (q4_0/q4_1/q5_0/q5_1/iq4_nl) are relaxed to the **logits level** (argmax preserved, top-2
  margin >= 2.2). The FA chooser is TILE across the whole band. Re-run the 8-type x 5-length grid when
  a single-token-tuned kernel changes. `GREEDY-PURITY.md` §36.
- **The high-power pin regressed** (`~/bin/high-power`: tg -5-7 %, pp -15-18 % on RCCL/hybrid paths).
  Servers run **unpinned**, 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default.
- **There are TWO fused gate/up+GLU MMQ arms in one dispatch, and they are arch-disjoint (r138, block 08).**
  Upstream #29948's arm (`has_fusion`: gate rows interleaved into the weight tile by the loader, GLU in the
  MMA write-back, `I_dst = I/2`) is **NVIDIA-only by construction** — its predicate ends in
  `turing_mma_available()` and its write-back depends on the MMA accumulator layout — and the delivery's
  arm (`has_gate`: the matmul runs twice into two accumulators, GLU in registers, `J` capped per type) is
  the **RDNA** one.  Both live in `mul_mat_q_process_tile`/`mul_mat_q`/`mul_mat_q_case` as *separate*
  template parameters, with `static_assert(!(has_fusion && has_gate))` making "one fused path at a time" a
  build error, and a single J selector (`ggml_cuda_mmq_select_J`, which upstream's
  `ggml_cuda_mmq_get_J_best` now delegates to).  Do not "simplify" this to one arm: enabling the interleaved
  arm on RDNA is unvalidated and order-dependent, and dropping the delivery's arm loses the RDNA prefill win
  (pp2048 +5.3 % gfx1151 / +9.4 % gfx1100).  `ggml_cuda_should_fuse_mul_mat_q` is non-static and declared in
  `ggml-cuda-try-fuse.cuh` because block 10's moved dispatcher calls it.  Record:
  `archive/work/rebase-2026-10b/DECISIONS.md`.
- **A fusion that stands down for MMB must ask the DISPATCH predicate, and the BF16-only activation mark may not be
  preempted (r171, TODO #115).**  A gate+up pair is owned in this order: **MMB's fused GLU -> the mmq/mmvq triple ->
  `mul_mat_q_pair` (the fallback) -> the two plain muls**.  The pair stands down on
  `ggml_cuda_mmb_supported_mmid()` / `_supported_mm()` — the predicates `ggml_cuda_mul_mat()`/`_mul_mat_id()` dispatch
  on, a strict superset of the fused-GLU one — and never on a weight-type test: the type-only helpers
  (`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`) are **gone** because they stood the pair down for shapes MMB
  never claims (`T < mmb_min_t()`, `n_used > 64`, a `K` that is not a whole super-block count, ...) and, worse,
  disagreed with the graph optimizer's HC16 **BF16-only** marking, which skips a producer's F32 store when every
  consumer is a bf16 reader — a fusion that preempts such a reader then reads a buffer nobody wrote.  So the marking
  pass also refuses to elide an activation with a **routed** `MUL_MAT_ID` reader (it marks the bf16 *copy* instead).
  The two live in different passes (`graph_optimize` vs `graph_compute`) and each must hold on its own.  A matcher
  that probes an arbitrary node pair owes null-safe predicates (`supported_mm`/`_mmid` are, and the pair matcher
  checks both nodes' source slots): the first r171 cut segfaulted on gfx1151 loading Flash-Next at `-ub 256`.  Pair and
  triple are bit-identical (same dot order, same `J` tile) and time-neutral (gfx1151), and RDNA4 is inert
  (`mmb_cfg().routed = 0`).  `GGML_PAIR_ROUTED_OFF` is the routed half's A/B arm.  `GREEDY-PURITY.md` §69,
  `benchmarks/2026-10-29-mmb-pair-ownership.md`.
- **`ggml_cuda_mul_mat_q`'s `if (fusion)` assert is the ARM's precondition, not the function's: `ids` is legal
  on the routed arm (r161, issue #241).**  Upstream #29948's
  `GGML_ASSERT(gate && !ids && !x_bias && !gate_bias && !x_scale && !gate_scale)` is the *dense* interleaved
  arm's precondition, but the same entry point serves block 08's routed `has_gate` arm (`ids != nullptr`, no
  bias/scale, dispatches `ggml_cuda_mul_mat_q_switch_type_gate`), which is exactly what the MoE gate/up+GLU
  prefill fusion calls.  The assert arrived with the r138 re-base and made that path abort on every config
  that reached it (r138-r160; `lfm2moe` server prefill was the live case, `GGML_ASSERT(gate && !ids ...)` at
  `mmq.cu:468`).  Never re-add `!ids` there; never "fix" such an abort by guarding the dense matcher
  (`ggml_cuda_should_fuse_mul_mat_q` cannot match a `MUL_MAT_ID`) or by switching the fusion off (the env kill
  switch only hides it).  Reachability is arch- **and** model-dependent: on RDNA4 `mmb_cfg().routed = 0` lets
  `mul_mat_q_pair` consume the gate/up pair first (`GGML_PAIR_OFF=1` is the gfx1201 repro), while gfx1151
  reaches it from the server when MMB's fused GLU declines the shape - so a green daily arm does not prove it.
  The A/B arm is `GGML_CUDA_DISABLE_MOE_MMQ_FUSION=1` (fused == unfused: identical greedy text,
  prefill-logit KLD -0.000000 / 100.000 % top-p).  `GREEDY-PURITY.md` §65.
- **A combine+norm style fusion must read the tensor the model PINS, never an unpinned ancestor (r140, issue #226).**
  The repeat-anchored `hc_combine_norm` matcher starts its window at the `REPEAT`, so `scale1 -> sigmoid ->
  scale2` run standalone *before* it; a tensor read from before the window is only safe while the graph keeps
  it alive, i.e. only when it is a graph output.  r138's #31028 merge turned `inject` into a `SCALE` node and
  the matcher descended to `scale1->src[0]`, whose buffer the allocator then handed to the chain's own
  SIGMOID — a pre-#31028 (separate-tensor) GGUF read a sigmoid output as `inject` and its PPL went to 2690.
  The fix reads `inject = scale1` (the pinned tensor) with identity params (`s1 = 1`, `b1 = 0`) and requires
  `GGML_TENSOR_FLAG_OUTPUT`.  The **scale1-anchored A-chain** matcher is deliberately different: its window
  starts *at* `scale1`, so that node is elided and `scale1->src[0]` is the live read — do not unify the two.
  A new emitter that hands a `SCALE` to such a matcher must pin the tensor the matcher reads, or lose the
  fusion.  **r141 closes the two halves r140 left open:** the fused kernel now owns the same columns in the
  RMS sum as `rms_norm_f32`'s vec4 path (it was 0.0185 KLD / 95.6 % away from its own chain, now
  `0.000000 / 100.000 %`, and the fused arm's greedy text is byte-identical to
  `GGML_CUDA_DISABLE_HC_COMB=1`), and a **merged-tensor** (post-#31028) GGUF takes the fusion again because
  `build_hc_mix` materialises the inject slice with `ggml_cont` and the matcher accepts a pinned
  `view_src == nullptr` tensor instead of requiring a `SCALE` (a pin on a *view* does not keep the root
  alive — test that any new inject shape satisfies it).  `GREEDY-PURITY.md` §59; records `WORKLOG.md`
  261025.014 / 261025.015.
- **The qwen4exp HC mixer's two BF16 GEMMs share one converted src1, so the model must emit them
  adjacent (r142, issue #226 follow-up).**  `build_hc_mix`'s separate-tensor branch (pre-#31028 GGUF) runs
  `hc_down` and the 4-row `hc_inject` over the same F32 `xn`, and the CUDA BF16 GEMM path caches **one**
  converted src1 (`bf16_src1_cache`, block 10) — so the reuse only happens while nothing converts in
  between.  Upstream #31028's emission order puts the inject product *after* the gate product (whose src1
  is the silu output, which evicts the entry), which converted the whole `[hc*n_embd, n_tok]` tensor twice
  per mixer: measured on Flash-Next GSQ-IQ3_XXS (3 GPU `-sm tensor`, all-VRAM) as 4560 vs 2280
  `convert_unary_cont_vec4<float, bf16>` launches and **-3.4 % prefill**; the fix emits `lo` and the inject
  product back to back and expands **both** there (`ggml_build_forward_expand` order, not node creation
  order, is what lands a node in the graph).  Bit-identical (PPL, all 24 chunks of the reporter's command,
  6/6 greedy cells), `GGML_CUDA_BF16_SRC1_CACHE=0` is the A/B control.  The merged path is one GEMM and is
  untouched.  Record: `benchmarks/2026-10-25-qwen4exp-hc-inject-adjacency.md`,
- **qwen4exp's `hc_*_down`/`hc_*_inject` pairs are ONE tensor since r138 (upstream #31028).**
  `create_merged(LLM_TENSOR_HC_{ATTN,FFN}_DOWN_INJECT, …)` accepts the converter's merged weight and
  `build_hc_mix(…, w_down_inject, …)` does **one** GEMM for both products; the old separate tensors are
  still accepted (the loader falls back when the merged tensor is absent, which is why pre-r138 GGUFs keep
  loading).  A `build_hc_mix` caller must pass `layer.hc_*_down_inject` — a call that omits it compiles
  against the old 7-argument form and silently reads the wrong weight.
- **The set applies whitespace-clean** (`apply-all.sh`, no git whitespace warnings).

## Moved from the build notes (still facts, not build policy)

- **A re-base is a decode AND prefill change: measure throughput before the tag, not only correctness
  (r138).**  The per-block build, the prefill-logit KLD and the op tests prove the kernels are *faithful*;
  none of them measures speed, and a re-base can change launch geometry without changing logits (r138's
  hand-merged MMQ tiling, the qwen4exp one-GEMM HC path, rope).  Before tagging, run the stock-relative band
  gate (`llama-batched-bench -npp 512 -ntg 128 -npl 1,4,8`) **and** an rNN-1-vs-rNN prefill A/B
  (`llama-bench -p 2048,8192 -n 0`) on a build pair configured identically — including the same
  `GGML_CUDA_FA_QUANTS` set (see the next bullet; a differing set alone changes greedy text).  Keep the
  previous release's binaries (`build-<tag>/bin`) so the A/B does not need a rebuild.  Record in
  `benchmarks/` + a `WORKLOG.md` entry.  r138's A/B: `benchmarks/2026-10-25-r138-rebase-ab.md` (no
  regression; dense prefill +1.4-2.3 %).
- **A scratch build dir and `~/llama.cpp/build-rocm` can hold different `GGML_CUDA_FA_QUANTS` sets, and that
  changes greedy text (r138).**  The FA instantiation set is a CMake **cache STRING**, so it keeps whatever
  the tree's default was when that build dir was first configured: at r138 the two dirs held
  `q4_0;q4_1;q5_0;q5_1;q8_0;bf16;f16;iq4_nl` (canonical) vs `q4_0;q8_0;f16;bf16` (scratch) from the same
  source tree.  Different instances => a different FA kernel for a shape => different greedy text, with
  **identical prefill logits** (the gate read KLD 0.000531 / 98.863 % on both).  So: compare text only
  between builds of the **same** build dir, run the prefill-logit gate with the **canonical** build
  (`gate-prefill-logits.sh` defaults to `~/llama.cpp/build-rocm/bin/llama-perplexity` for exactly this
  reason), and never read a cross-build text difference as a numeric change without checking
  `grep GGML_CUDA_FA_QUANTS <build>/CMakeCache.txt` on both sides.

## The KV window on a tensor split (r204, `TODO` #124)

- **A Meta (tensor-split) device carries no backend reg, so every KV-region answer for it must come from
  `ggml_backend_meta_kv_region_*`.**  Those resolve each proc through a **child device's** reg and require
  *every* child to answer: a region buffer type built for only some children is a window that covers part of
  the cache and reports as if it covered all of it.  The same helpers are what `--fit` uses
  (`ggml_backend_meta_kv_region_is_kv_buft`), because the CUDA predicate is an identity on the *child's*
  `alloc_buffer` and cannot answer for a Meta buffer type.
- **The grow hook must never return 0 for a buffer it does not understand.**  `ggml_backend_cuda_kv_region_grow`
  returns 0 = "not a region buffer", which `llama_kv_cache::grow_regions` treats as success — for a Meta buffer
  that is a silent success on unmapped VA and the next batch **faults the GPU** (not an error).  A Meta buffer
  is fanned out by `ggml_backend_meta_kv_region_grow`, and a child buffer it cannot reach is a **refusal**
  (-1), never a 0.
- **A windowed region's initial commit must stay inside its own reservation and on the map grid.**  The commit
  rounds each tensor's window UP to a whole map unit; on a split each child is its own VA reservation, so the
  last tensor's inflated window ran past the end of a child reservation and was refused (`the KV window
  (4.00 MiB) could not be committed in VRAM on device 1`).  Clamp to the region end, in whole units — a
  shorter commit only loses at most the last partial unit, and nothing mapped is only ever a fresh host
  mapping, never a partial flip.
- **`ensure_span` may flip SEVERAL whole mappings in one request, but never a partial one.**  One batch can
  advance the spill boundary by more than one unit, and a region whose unit is floored to the driver's
  granularity has mappings smaller than the unit it was asked for (measured at `LLAMA_KV_REGION_UNIT_MIB=1` on a
  2 MiB granularity: the request covered two mappings and the old exact-match check refused it, aborting the
  batch).  A request that covers a mapping **in part** is still refused: a partial rewrite of a mapping loses
  bytes on this driver.
- **A 2-D device copy cannot serve a mixed host/VRAM region, and the batched state path is 2-D.**  A windowed
  region's spilled cells are pinned host memory mapped into its VA, so `hipMemcpy2DAsync` D2H fails with
  `invalid argument` on a range that begins in the host mapping (`cache_k_l64 (view)`, 1024 B x 4782 rows on a
  2-GPU split's checkpoint save).  The region-backed directions of `get/set_tensor_2d_async` must fall back to
  the region buffer's own copy (contiguous rows in one 1-D copy), **ordered behind the compute stream** like the
  pool mirror — a blocking default-stream copy does not wait for it, and a state saved mid-batch is torn.
  A tight block is only legal when **both** pitches equal the width: the caller's destination pitch is the
  split's full row, wider than one device's slice (r204 shipped that bug for one gate run and the state-pipe
  gate caught it).
- **A MIRRORED cache (dsv4 / HRM) declines the window on a split.**  Every device holds the whole cache, so one
  window costs `n_devices x W x cell_bytes` while the pinned-spill plan is built per logical cache — it would
  under-book the spill by the device count and the run-time ceiling would then refuse a mapping mid-run.
  `llama_model::kv_cache_split_mirrored()` is the one predicate (it mirrors the split-state callback's rule for
  `cache_k_*`/`cache_v_*`).
- **A split's window is sized against its per-device budget, not aggregate free VRAM.**  `--fit`'s Meta entry
  used to carry child 0's `-fitt` margin; it is now the **sum** over the children (otherwise
  `(n_devices - 1) x target` silently vanished from the budget).  The window is charged at its physical
  `W x cell_bytes` (aggregate, because that is what all devices commit together), which makes the aggregate
  check imply every device's: the loader shards the KV along the KV-head axis by the same `tensor_split` ratio
  it shards the weights by, and `budget = min_i(target_i / ratio_i)`.  This correction applies **only** where a
  window was asked for — a run without `LLAMA_KV_RESIDENT_CELLS` keeps the branch's old (conservative)
  arithmetic exactly, so the split's pre-existing `ntd x` KV over-count is unchanged outside the window arm.
