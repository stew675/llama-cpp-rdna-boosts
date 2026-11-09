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
  the cache quantizer stays the kernels' compile-time `kv` template value), and since **r205** also
  `GGML_TYPE_Q5_0` and `GGML_TYPE_IQ4_NL` on a wave32 device
  (`ggml_cuda_quantize_q5_0_group` — the q4_0 seed with the `-16` divisor and `qh` written whole from a ballot of
  the 32 lanes' 5th bits; `ggml_cuda_quantize_iq4_nl_group` — the q4_0 seed with `kvalues_iq4nl[0]`,
  `best_index_int8` per element, and the reference's 16-term **scale search** reproduced in its serial `j` order by
  broadcasting term `j` and adding it one at a time, because a butterfly rounds differently: +10.2 %/+24.8 % tg128
  on a `q5_0`/`iq4_nl`-KV 30B-A3B, kill switches `GGML_CUDA_KV_FOLD_Q5_0=0`/`GGML_CUDA_KV_FOLD_IQ4_NL=0`,
  bit-identical), and
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
  path never evicts a RESIDENT to hold a victim** (`pool_peek_locked` / `pool_fill_empty_locked`; the
  background worker and `MOE_HOST_POOL_BGFETCH` are gone) — do not reintroduce one: the literal swap measured
  here cost **2x decode** for 0 MiB of L3.  **CORRECTED IN PLACE (r224):** the r113 conclusion was "no
  write-back at all"; what it actually measured was a *swap into a resident's slot* on a **covering** pool
  where there was no L3 to win.  The demote now ships as `MOE_HOST_POOL_SINK=1`: the victim goes into a slot
  the pool **already plans as headroom** (`MOE_HOST_POOL_SINK_SPARE_PCT`, 4 % of a table's slots, below the
  budget — the pool does not grow), so it is one D2H and no eviction, and it declines (dropping the victim as
  before) whenever no headroom slot is free, the geometry is a strided `-tm tensor` slice, or the copy fails.
  Measured at `-lme 36` on a frozen real-world prior: +5.3 % alone, +7.8 % with the decode ranking; 6-8 % of
  demoted slots are used again against a ~3 % break-even on this box's NVMe.  Losing the duplicate-retirement
  win is why the headroom is a *fraction of slots* and not of experts.  (3) the whole-table prefill staging reads the
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
- **A non-quantized KV cache has no attention rotation, so the r187/r205 KV folds cannot fire on it (r205, `TODO`
  #120).**  `llama_kv_cache`'s `attn_rot_k`/`attn_rot_v` both require `ggml_is_quantized(type_k/type_v)`, so
  `-ctk/-ctv bf16` (or `f16`, `f32`) logs `n_rot_k = 0, n_rot_v = 0` and the graph carries no `attn_inp_k_rot`:
  the "bf16 has no fused write kernel" premise is wrong, there is no fold pattern to extend.  Measured on a `bf16`
  KV 30B-A3B: `k_set_rows<float,int64_t,nv_bfloat16>` x960 + `rms_norm_mul_rope_f32<256,false,float,0,0,false>`
  x960, i.e. zero folded launches.  The one exception is the forced indexer rotation
  (`LLM_ARCH_DEEPSEEK32/DEEPSEEK4/GLM_DSA/DOTS3NOTE` with `n_embd_head_k_full == indexer_head_size`) — an
  accepted limitation, no such model fits a 32 GiB card.  A bf16 write arm was implemented and removed rather
  than shipped unreachable; its `--ub 1` A/B read `-0.000001 / 100.000 %`, which the same-build **noise floor**
  arm reproduces, so that point cannot decide a bf16 arm at all.
- **`iq4_nl`'s scale search is order-sensitive: the fold must reproduce the reference's 16-term serial sum (r205).**
  `quantize_f32_iq4_nl_block` accumulates `sumqx`/`sumq2` over `j = 0..15` one term at a time and then sets
  `d = sumqx/sumq2`; the lane-parallel form computes term `j` on lane `j` and **broadcasts the terms in `j` order**,
  adding each one at a time (32 `__shfl_sync` per 32-element block).  A butterfly reduction is not the same float
  and would have flipped cache bytes; with the serial order the `--ub 1` A/B is `-0.000000 / 100.000 %`.  The
  `q5_0` group needs no such care (a per-element quantizer) but its `qh` word is only reachable as one ballot —
  bit `j` = lane `j`'s 5th bit — so it is wave32 only, like `q4_0`'s ballot.
- **The slab headroom default is `1152 + 128 x n_gpu` MiB and its cost is measured, not assumed (r206).**  It was
  `640 + 128 x n_gpu` (768 / 896 / 1024 for 1 / 2 / 3 GPUs) from r86 to r205; one GPU now leaves **1280** MiB
  outside the slab.  The 640 base was sized on the field GSQ / `-ub 4096` shape, whose widest outside transient is
  ~300 MiB per device, so block 15's `-c 262144` host-expert runme needed 1536 there (r194's `hipBLASLt out of
  memory` on the first prefill) — a shipped default that is one workspace short on a documented config is the
  wrong default.  **The bump is margin, not a demonstrated fix**: the r194 cliff is order-dependent and r206 did
  not reproduce it (5 246 / 25 763 / 51 475-token prefixes at `-c 262144`, at 768 and at 1280, every arm clean and
  text-identical).  What the bump COSTS, measured on the field shape (Flash-Next GSQ-IQ3_XXS, 1 GPU, `-ncmoe 48`,
  `-c 32768` q8_0, 5 246-token prompt, `-n 2000`, interleaved arms, `ROCBLAS_USE_HIPBLASLT=0`): 768 -> arena
  24 691.7 MiB / `h` 0.9603 / 47.95 t/s vs 1280 -> **24 576.4 MiB / 0.9602 / 47.98 t/s**, i.e. **-115 MiB of
  arena, -0.01 pt of expert hit rate, decode flat**, text byte-identical.  An explicit `GGML_CUDA_SLAB_HEADROOM_MIB`
  is still taken verbatim (no per-GPU addition), so the old value stays one env var away.

- **Device-visible host memory must be `RESIDENT`, not merely addressable — `hipHostRegister` does not provide
  residency, and without XNACK there is no fault-and-retry (closed 2026-11-02: `archive/work/host-experts-window-beta` / `archive/work/host-expert-pinned-window`,
  2026-11-02).**  A registered range installs page tables (UVA alias) but leaves the pages ordinary reclaimable
  page-cache pages (`Mlocked` stays 0 — that is exactly the property the window campaign advertised).  On a
  no-XNACK GPU a device read of a page that is **not present** is a fatal `Page not present` fault, so a
  registered-but-unlocked range is sound ONLY while nothing evicts it.  Measured on 1x R9700, Flash-Next IQ4_NL
  (93.16 GiB model, 98 GB RAM): `--host-experts window -ncmoe 44` (host set ~59 GB) faults the GPU —
  `mul_mat_vec_q_moe` → `Page not present` — while `pinned -ncmoe 44` runs (its pages are never evictable) and
  `-ncmoe 32` is clean.  So "one residency tier, registration instead of pinning" needs **locked or provably
  resident** pages; `MOE_EXPERT_WINDOW_MLOCK` was written as an optional extra and is in fact a REQUIREMENT
  (needs `LimitMEMLOCK=infinity`/`CAP_IPC_LOCK`; the hard limit is 8 MiB by default).
  **And the evidence that claimed otherwise was void**: `probe-cold-read` reads a HOT 8 MiB window, so its
  `MADV_PAGEOUT` arm and even 70 GiB of anonymous pressure never evict the range (re-verified 2026-11-02: all
  arms `DATA IDENTICAL`), and every real run before the gather memcpy'd the whole host expert set through the
  host on each ubatch — an accidental prefetch that re-faulted every page immediately before the device read it.
  A reclaim-safety test must actually evict the range it reads; the gather (which removed that prefetch) is what
  finally exposed this.
  **CORRECTED 2026-11-03 (`archive/work/host-expert-locked-window/`): `hipHostRegister` DOES take a hold.**  The
  paragraph above ("maps without pinning", "leaves the pages ordinary reclaimable page-cache pages") is wrong on
  ROCm 10.1 / kernel 7.2.15 / gfx1201, measured three ways: registering over **absent** pages faults them in
  (page cache `0/425` -> `425/425`, proven through a *fresh* mapping, never the registered one); a registered
  window's bytes survive `hipHostUnregister`, `munmap`, a `MAP_FIXED` remap, every registration flag
  (`ro`/`portable`/`mapped`) and **67 GiB of real anonymous pressure** with cgroup `file` unchanged (1 GiB
  scratch-file window: +1027.9 MiB at register, still +1037.0 MiB after unmap + pressure); and under a 4 GiB
  cgroup a 6144 MiB registration returns `ok`, pegs `memory.current` at the limit, leaves 14/16 sampled windows
  non-resident, and a non-resident window still reads.  So the reason a sliding window cannot be built is that
  the hold has **no release**, not that it is absent — and "registered-but-unheld" is a state that does not
  exist for a range that was ever resident at registration time.  The 2026-11-02 fault is not explained by this
  and did **not** reproduce for a partially-registered range; do not re-cite "maps without pinning", and do not
  build a sliding host window (successor: `wip/host-expert-eager-push/`).

  **And `Mlocked` is not a pinning probe**: a `hipHostMalloc`'d buffer is pinned by the driver yet reports
  `Mlocked: 0 kB` — measured 2026-11-02 with two 256 MiB buffers that stayed data-identical through 61.7 GiB of
  real anonymous pressure (`archive/work/host-expert-pinned-window/probe-host-malloc-mapped.cpp`).  So "`Mlocked`
  stayed 0" is not, on its own, evidence that pages can be reclaimed (nor that they cannot); `hipHostAlloc`/
  `hipHostMalloc` pin without mlocking, and `hipHostRegister` maps without pinning.  The same probe confirms the
  pinned path itself: a kernel (SM) read of a `hipHostMalloc`'d buffer is data-identical at **51.8 GiB/s** with
  `hipHostMallocDefault` and with `hipHostMallocMapped` alike, and `hipHostGetDevicePointer` returns the identity
  pointer on this platform (`unifiedAddressing` reports NO but `canMapHostMemory` yes), so the `Mapped` flag is
  not what buys anything here — the pinning is.

- **A 1-D copy-engine push out of the pageable model mapping does not tear here (wip, 2026-11-03;
  `archive/work/host-expert-locked-window/MEASUREMENTS.md` §5/§6).**  The delivery carries "a pageable
  `cudaMemcpyAsync` ... is staged through a shared internal bounce and tears under back-to-back use (measured:
  nondeterministic greedy text)" (`ggml-cuda.cu:6255`, prior campaign `PROTOTYPE.md:47`) and that note is what
  justifies the pool's `memcpy` into a pinned slot.  It has **no dated record, no ROCm line and no
  reproduction** in the repo, and it does not reproduce for a 1-D push on ROCm 10.1: 8-64 copies in flight
  (13-128 MiB) over 2-4 streams from one host thread, sources plain pageable / `mlock`ed / registered /
  `cudaMallocHost`, every copy verified against its own source — **0 torn in 20-100 iterations each**, including
  repeated pushes from the same page-cache regions (so one-shot semantics buy nothing).  The **2-D** path the
  note actually names is 4-5x slower (376 us per 1.66 MiB vs 66-93 us 1-D) but also clean here.  What the same
  probe measures is why the direction is worth pursuing: a CE push straight from the mapping costs **66-93 us
  hot / 1.32 ms cold** per 1.66 MiB against the pool's **137 us / ~3.5 ms**, i.e. the memcpy is the entire gap.
  The claim is NOT refuted for ROCm **7.14** (the delivery ships both lines) or for the **async-input ring**
  (a shared *destination*, a different failure); `wip/host-expert-eager-push/` P1 settles it in the real path
  before any default flips.
- **A pool-backed expert table must be served by the device gather in the prefill band; its whole-table staging
  is the slow path at every width (r207, block 12).**  For `--host-experts pool` the master is the read-only
  model mapping, so staging a table for a wide ubatch reads the pageable mapping plus the pool's host alias and
  degrades as coverage rises: on the 122B-A10B across three RDNA4 GPUs (`-ncmoe 48`, 5258-token prompt,
  `-ub 2048`, 2000-token greedy decode) pool FRAC 80 staged at 134.3 t/s and FRAC 95 at 12.9, against
  **1507.7-1631.3** with the gather and `pinned`'s 634-638; the same arm with `GGML_SCHED_DEVGATHER=0` collapses
  to 16.9.  Its prefill logits are **bit-identical** to the staging path (KLD -0.000000 / same-top-p 100.000 %),
  so this is a transport fact, not a numerics one.  The gate is a **band** one: `n_tokens > moe_cache_band`, because
  taking the decode band over with the gather costs an order of magnitude (1.2-4.1 t/s) — the decode band is
  served from the arena by the slot-remap read.  Do not widen the gather to the decode band, and do not make
  `GGML_SCHED_DEVGATHER=1` (force) the default: it costs a pinned master ~47 % prefill at two devices.
- **A memory ceiling on a cache must be applied to the DISTRIBUTION, not to whatever registers last (r207, block
  16).**  `MOE_HOST_POOL_RAM_PCT` shaved the registration-order tail to the one-slot floor: on the 122B at
  `RAM_PCT=70` that was 26 of 144 tables at one slot and **10.94 t/s**, while 10 % more ceiling decoded 19.52.
  Scaling every table's slot count by one process-wide factor gave **17.44** at the same pinned bytes (+73 % with
  `MOE_HOST_POOL_SIZING=1`: 18.92), and `RAM_PCT=85` went 19.52 -> 20.97.  A per-table floor is only defensible
  where the ceiling cannot cover one slot.  Same rule for a concentration fit: it must be fitted *to* the ceiling,
  because a uniform rescale of a fitted plan collapses it back to uniform.
- **At `-ncmoe 48` on a three-device split the expert set is fully VRAM-resident, and the DEFAULT mode still
  re-copies it out of the arena on every prefill ubatch (r207, `TODO` #133).**  The arena sizes to
  `67110.0 MiB of 67110.0` (256/256 slots/table, identity), yet the prefill stages **61 GiB for a 3049-token
  prompt** with `arena=61031.5 MiB host=510.5 MiB` — 99.2 % of it read back out of VRAM into the op's per-split
  weight buffer, once per table per ubatch, serialised against the compute (copy table -> compute -> copy next) at
  ~8 GB/s aggregate.  Consequences: `pinned` prefill at 3 GPUs (394-426 t/s) is *slower* than at 1 GPU (755.5)
  despite 3x the VRAM and 3x the compute, and `-sm layer` at 3 GPUs prefills at 944.3 for the same redundant bytes
  because it issues one staging call per table instead of one per table *per device*.  Host residency is
  irrelevant to this cost; a fully resident table should serve the prefill op in place.
- **A launch cap whose justification is an occupancy QUERY has to be re-checked when the query is replaced
  (r208, block 05).**  `rdna_cap_ncols2` capped the D=256 tile variants at `ncols2 <= 2` because HIP's RDNA2
  occupancy calculator (it assumes a quarter of the WGP register file) returned 0 blocks and the launcher
  asserted.  r76 replaced that query with a manual WGP computation, and the cap then guarded nothing: every
  D=256 shape the launcher can pick measures **>= 2 blocks** on the patched gfx1030 HSACO (4x8 = 235 VGPR,
  29 696 B LDS, 4 blocks), so the assert is unreachable.  A stale cap is not free: it was also withholding the
  GQA fold from D=256, which is most of the RDNA2 prefill deficit of issue #214.  When a "this aborts" cap and
  the mechanism it blamed are in different layers, prove the mechanism still exists before keeping the cap.
- **`i_sup - 1` in the RDNA2 tile tail is deliberately allowed to be -1 (r208, block 05).**  A V chunk that
  lies wholly past the KV end (the tail length is an exact multiple of `nbatch_V`) loads with `i_sup == 0`, so
  the clamped row index is `-1` and row `-1` of that chunk is `k_VKQ_max - 1`, the last VALID row; every weight
  in the chunk is zero, so the values are unused.  Do not "fix" it to `max(i_sup - 1, 0)`: row 0 of such a
  chunk IS `k_VKQ_max`, one row past the cache.  The comment in the loader says so for the same reason.
- **Two compiles of the same source do not produce byte-identical objects, so object or `.so` hashes are not
  a codegen oracle (r208).**  Comparing a release's `libggml-hip.so` before and after a change reports a
  difference even when nothing changed (the symbol/string-table order varies run to run).  What *is* stable is
  the code object: extract `.hip_fatbin`, unbundle it per arch, and compare each kernel's `.text` slice by
  symbol address/size, plus the amdhsa resource usage (`.vgpr_count`, `.vgpr_spill_count`,
  `.group_segment_fixed_size`).  That comparison is what proved r208's RDNA2-gated changes leave gfx1201
  bit-identical (88/88 and 106/106 kernels).
- **On Windows `hipMemGetInfo` is not a free-VRAM figure (r209, block 09).**  It counts only this process's
  allocations, so with any other process holding memory on the card it over-reports free VRAM; and WDDM does
  not refuse an allocation past that point, it pages the excess to system RAM and the GPU reads it over PCIe
  (issue #216).  The adapter's own figure is `D3DKMTQueryStatistics` over the non-aperture segments,
  `CommitLimit - BytesResident`, which counts every process; `cudaMemGetInfo` now returns the smaller of the
  two (`GGML_CUDA_WDDM_FREE=0` keeps HIP's).  Two consequences to keep straight: the reading is honest, but
  **an overcommit still does not fail**, so `TODO` #216's accepted limitation stands; and the two figures are
  not interchangeable — HIP's is this process's remaining budget, WDDM's is the adapter's remaining commit
  room, and the WDDM one *includes* memory other processes will keep.
- **Windows `llama_mmap::unmap_fragment` used to be a no-op, so every loader release freed nothing (r209,
  block 09).**  A view cannot be unmapped in part, so the Windows implementation now drops the range from the
  working set with `VirtualUnlock` on unlocked pages (it reports `ERROR_NOT_LOCKED` and the clean file pages
  move to the standby list; it still unlocks a locked page in the range, as `munmap` would).  The mapping and
  the data pointer stay valid, so unlike POSIX this is safe even for a range something still reads — do not
  "restore symmetry" by making the POSIX side a trim, because there the release is what bounds peak RSS.
  Measured on the reporter's box (Flash-Next, `--fit`): working set after load 67.4 -> 21.2 GiB.
- **Windows has no `MemAvailable`, and before r209 the delivery's two RAM ceilings and the pool's fill-mode
  decision therefore did not exist there (r209, blocks 12 and 15).**  `pool_available_ram_bytes()` returned
  `-1`, so `MOE_HOST_POOL_DIO=auto` always resolved to `O_DIRECT` ("no MemAvailable") and neither
  `MOE_HOST_POOL_RAM_PCT` nor `LLAMA_KV_SPILL_RAM_PCT` applied.  `GlobalMemoryStatusEx().ullAvailPhys` (free
  plus standby) is the counterpart and is what they now read; it is a *different* quantity from the WDDM
  device figure above, and one may not substitute for the other.
- **A Windows-only change can be folded into the delivery when it is provably inert on Linux, and the proof
  is preprocessing, not inspection (r209).**  Compile each changed translation unit with the delivery's own
  compile command (`build-rocm/compile_commands.json`) as `-E -P` on both trees and diff the token streams:
  identical apart from `__FILE__`/`__LINE__` strings is the statement "the Linux build compiles the same
  code"; object or `.so` hashes are not (see the r208 entry).  A header change is inert by construction when
  its only delta is inside `#if defined(_WIN32)`.  What this method cannot do is validate the Windows arms —
  those stay the reporter's numbers, and no gate on this box reaches them (there is no Windows toolchain,
  `TODO` #95).  Also: `ggml/src/ggml-cuda/CMakeLists.txt` globs `*.cu` **without** `CONFIGURE_DEPENDS`, so a
  newly added CUDA source needs a cmake re-run, not just a rebuild.
- **A pageable push costs nothing where a fill batch goes to ONE device, and that is the whole present
  constraint (r209, `wip/host-expert-eager-push` P3).**  Making the copy engine's source the read-only model
  mapping (no memcpy into a pinned slot) measured **55.1 -> 53.6 t/s** on 1 GPU and **55.6 -> 53.7 t/s** on a
  3-device `-sm layer` split — parity, byte-identical text — but **52.5 -> 26.1** on a 2-device and
  **20.6 -> 16.3** on a 3-device `-sm tensor` split, the configuration that issues the fills of *several
  devices inside one op*.  The marginal per-fill cost over the pool's memcpy+pinned+CE shape is 28 us / 205 us /
  302 us / 41 us for 1 dev / 2 dev tensor / 3 dev tensor / 3 dev layer, against a host that already spends
  ~1 core-second of CPU per token on the tensor-split host-expert path.  The driver's pageable staging
  ("staged through a shared internal bounce", `ggml-cuda.cu:6255`) is real and widens with device count
  (1.06x -> 1.54x of the pool's per-fill cost) but is ~20x too small to explain the tensor-split loss, so do
  not attribute it to the driver alone.  Record and candidate causes: `wip/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §2-§3.
- **The 1-D pageable tearing claim stays refuted, now multi-device (r209).**  Up to 65 260 consecutive direct
  pageable fills produced byte-identical greedy text on 1, 2 and 3 devices with no GPU fault, so the
  "nondeterministic greedy text" half of the `ggml-cuda.cu:6255` comment has no recorded reproduction here —
  the cost half does (previous entry).  Both ROCm lines still owe their own record; this one is 10.1 only.
- **A `-sm tensor` split never reaches `moe_cache_update_host`, so no per-op host boundary exists there
  (r209).**  With tensor split the whole graph is one Meta split, so a routed expert op's ids are computed
  inside that split and the issue-#129 guard stands the scheduler's host-readback branch down
  (`ggml-backend.cpp`, `host_expert_input && node_n_consumers == 1 && !ids_computed_here`); measured 3
  boundary events (the sizing calls, one per device) at 2 and 3 devices against 60 357 on `-sm layer`.  The
  expert arena is still driven (take-over + demand fills), so a policy hook placed there silently does nothing
  on a split — and a split is the only configuration where one host tensor has a device set > 1, so a
  cross-device release gate is currently unreachable in *any* configuration.
- **A diagnostic written to stderr from the fill path can corrupt a text-purity comparison (r209).**  Callers
  capture `2>&1`, so an unthrottled `fprintf(stderr, ...)` lands in the middle of llama-cli's streamed
  generation and an otherwise byte-identical arm hashes differently — measured twice: a `-lv 4` log (stats
  interleaved, which `scripts/extract-generated.py` documents) and a per-event `P1-PUSH` message re-fired
  after a budget reset.  Keep capture streams separate (`>out 2>err`: the text is stdout, every log line is
  stderr) and make such messages once-per-process.

## r210 — the host-expert tier is one pinned per-expert pool, and the pool mirrors the arena when the budget allows

Established 2026-11-04 (`wip/host-expert-eager-push/P5-RESULTS.md`, `P6-POOL-REGION.md`).  Each entry is a fact that
cost a session to learn; do not re-derive them.

1. **The pool IS the "pinned region addressed by expert index with a presence bitmap"** the design asked for: a
   pinned per-host-tensor allocation addressed by `expert -> slot`, with `slot_state` as the presence bitmap,
   `base_dev` as the UVA alias, heat-ranked contents and a native fill (a resident is a 1.3 µs D2D).  Nothing else
   had to be built to make it a tier — only permissions.
2. **A pool-resident expert can be cold-read in place at any device count.**  The historical restriction
   (`device_count == 1 && src_off == 0`) was a geometry accident: `base_dev` is already the cross-device D2D fill
   source, and a slice offset is folded into the cold base for a pinned master anyway.  Worth **+11.5 %** decode at
   `-lme 8` and **+10 %** for the not-fitting band (2 devices, byte-identical).  The host-side admission decision
   and the kernel-side cold description MUST share one predicate — if they disagree the host emits a cold id the
   kernel resolves from a different base, i.e. a silently wrong matmul, not a crash.
3. **"The pool is the arena's complement" is what forced page-cache re-staging.**  Two paths un-loaded an expert
   from the pool the moment VRAM also held it (the arena-seed retire, and the heat re-rank's wanted set).  At full
   coverage that emptied 44 % of the pool at start-up, and the first arena evictions of those experts then had to
   re-read from the model mapping: **20 116 synchronous 288 KiB `memcpy`s in the first 300 decode tokens**, i.e.
   11 % of decode — and invisible to the fill probe, which times only the `cudaMemcpyAsync` (1.2-1.4 µs), not the
   host memcpy that precedes it.
4. **The right layout rule is one rule: the complement first, then mirrored residents while capacity lasts.**  A
   budget that merely covers the complement already stops the re-read cycle; a budget that covers the set makes the
   pool a full mirror (`h=1.0000`, zero evictions, **no page-cache re-read at all**).  Measured at 2 devices:
   prefill **1318** vs a pinned master's 1105 (**+19 %**) and decode within 0.5-2 % (n=2000: 82.5 vs 82.9) **for the
   same 18 662 MiB of pinned RAM**.
5. **At equal RAM, a partial budget belongs to the pool, not to per-tensor pinning** — decoding interpolates
   linearly on the pinned fraction (0 % 37.4, 21 % 42.4, 43 % 50.1, 86 % 65.7, 100 % 71.9 t/s) while 8 064 MiB of
   pinned whole tensors measures 505 prefill / 50.1 decode against 713 / 57 for a 9 331 MiB per-expert pool.
6. **An eager back-fill of the arena's evicted victim does NOT pay** (built, measured, dropped): the victim is by
   construction the least-demanded resident, `ghost` cannot predicate it (reset to 0 on admission), and the copy
   competes with the demand path — page cache 52.8 vs 56.4, `O_DIRECT` disk 15.1 vs 15.9 with prefill collapsing
   24.1 → 16.3.  Same lesson as the L3 model: the pool's CONTENT matters far more than any hand-over policy.
7. **A worker thread must not be the first to touch the ROCm runtime** from a foreign thread: its lazily created
   `AsyncEventsLoop` aborts the process (`std::system_error: Operation not permitted`).  Keep HIP calls on the main
   thread; a host-only worker plus main-thread event polling is the shape that works.
8. **The `-ncmoe` tail bound is `hparams.n_layer()`, not `n_layer_all`** (the latter also counts the nextn/MTP
   head, which is never hosted) — an off-by-one there plans a pinned MTP head and misses trunk layer 0.

## r211 — the take-over guard is about REPRESENTABILITY; a covering pool is a complete cold source, and every block tree must build

1. **The eager take-over's completeness guard must not bound a covering pool by the arena.**
   `if (!table_cold_ok(t) && t.slots < n_used * n_tok) decline` is a statement about *representability* ("can
   every routed expert be expressed as a remap id"), not about arena capacity.  `table_cold_ok()` requires
   `host_dev != nullptr`, which is **null for every pool-backed (pageable-master) table**, so the guard silently
   bounded the *shipped default* tier by arena slots: at `-llama-batched-bench -npl 8` that is 8x8 = 64 ids
   against 56 slots, and EVERY routed op declined (`decline_all=1920`, `takeover=0`) and the run read the
   pageable mapping -- **tg 49.2 against 270.5** at `-npl 8` while `-npl 4` (32 <= 56) was untouched.  A pool
   whose capacity covers the tensor has the pinned master's own property (every expert is pool-resident, so
   `id = n_res_slots + slot` is a valid cold id for all of them), so `pool_covers_table()` -- the shared
   `pool_cold_geometry_ok()` **and** `slots >= n_experts`, plus an O(1) residency re-check -- exempts it.  Measured
   with the exemption: `-npl 8` **270.3 tg / 1829.5 pp** (best of every arm; pinned 276.5 / 1262.8, `-lme 64`
   full-coverage pool 271.4 / 1820.1).  A budget that does NOT cover the tensor keeps the bound (r211
   `-lme 8` still declines at `-npl 8`: 50.4 tg), because with `slots < n_experts` the resident set moves per
   token and a static per-shape bound cannot guarantee a complete remap.
2. **Do not reach for a `region`/`region_dev` rebind to express this, and never rebind `t.host`.**  The pool's
   cold encoding is already "a slot index into one pinned base", so the whole fix is one predicate; `t.host` is
   the key into `g_host_pools`/`g_host_src` and is shared by every per-device table of one tensor.
3. **A structural change to the host-expert tier is a decode-band change even when it looks like a prefill one.**
   The r210 handover attributed the `-npl 8` collapse to `moe_cache_stage_table`'s two-tier staging.  Measured
   false: for a covering table every expert is non-(-1) in that map, so staging is ONE kernel launch (the same
   shape as a pinned master's) and the pool's npl-8 prefill is **44 % faster** than pinned's; `GGML_SCHED_DEVGATHER`
   on/off changes nothing about the collapse (568 vs 635 pp, identical tg).  What looks like a 3x prefill collapse
   in that table is the **cold first-pass** shape (the arena is sized on the first decode-band pass, so the first
   prompt batch has none) -- the pre-r210 tree shows the same 128 vs 588 pp at `-npl 1`.
4. **A block tree that uses a later block's machinery is a broken delivery, and only gate 4 sees it.**  The r210
   fold left blocks 09/10/11 unbuildable: a spliced function body, `t.n_slices` read before block 10 grows it,
   `heat_prior_t` used before block 12 defines it, an iface entry pointing at a block-12 function, and a loader
   calling `plan_host_expert_placement`/`llama_host_experts_mode_apply` before they exist -- plus an out-of-line
   1-arg `host_expert_buft` that the header's inline forwarder already defines.  Rule: hoist the **definition**
   to the block that needs it (or restore the pre-fold body), and prefer the tip's content where the hoist would
   need a later block's *backend API* (an iface member shifts every backend's positional initializer).  The tip
   tree must be byte-identical afterwards -- check `git rev-parse HEAD^{tree}` at every step, not at the end.
5. **`MOE_HOST_LIMIT_FRAC=1` no longer means "force the pool".**  The `-lme` budget in the not-fitting band IS
   `auto_cap` = `MOE_HOST_LIMIT_FRAC` % of `MemAvailable` and it **overrides `MOE_HOST_POOL_MIB`**, so `=1` sizes
   an 812.9 MiB pool (11 slots/table) and every arm measured with it is a tiny-pool arm.  Use `-lme 64` (the whole
   set as a pool) and `-lme 8` (a genuine sub-complement budget) for those A/Bs.

## r212 — the host pool's fill mode mirrors `--load-mode` (the operator's caching decision), not a RAM guess

1. **`MOE_HOST_POOL_DIO=auto` mirrors `--load-mode`.**  A mapped load is a request for the page cache, so the pool fills through it
   (a `memcpy` out of the model mapping); O_DIRECT fills are an explicit `MOE_HOST_POOL_DIO=1`.  The pool only exists on a mapped
   load (its master *is* the mapping), so in practice auto = buffered.  The old rule ("the set does not fit beside the pool ->
   O_DIRECT") is preserved as a **note** on the decision line, not as the decision.
2. **The measured cost of getting this wrong is the POPULATION, not the steady state.**  On a 113 GB Flash-Next, 2 devices,
   `-ncmoe 48 -lme 60` (host expert set 71.7 GiB, pool 60 GiB, `per_layer_token_embd` 26.8 GiB host-resident): O_DIRECT fills read
   **97 692 × 1.0 MiB = 95.20 GiB at 2 785 MiB/s in 35.8 s** of drive time, all of it inside the load/first-prefill window
   (`TODO` #142), which reads as "prefill is reading every weight from disk".  Buffered: the same 95.20 GiB at **4 062 MiB/s in
   24.8 s**, and the pages stay reusable.  Nothing is evicted either way (`evictions=0`, `h=0.9073`).
3. **`-lme N` is a byte budget, so N below the host expert set is a SUB-COVER pool, and the model's expert count decides the
   coverage.**  On a 512-expert/layer model, `-lme 60` against a 71.7 GiB set is 428 slots of 512 = **84 %**, not "it fits"; the
   plan line prints both numbers, so read it before believing a pool is complete.  A pool *is* complete only when `slots >=
   n_experts` per table (`pool_covers_table()`, r211).
4. **A mapping-backed pool's master is the read-only model mapping, so `--load-mode dio`/`none` cannot have a pool** — the plan
   reports "NOT APPLIED: a pool needs a mapped load" and the whole-tensor pinned master is used instead.  Do not read that as the
   pool being disabled by a bug.

## r213 -- the attendance pool on a multi-device `-sm layer` split (per-device buffer types and counters)

1. **A per-device buffer type's NAME must be device-unique.**  `llama_kv_cache` keys its `buft -> ggml_context` map by
   `strcmp` on `ggml_backend_buft_name()` (deliberately, so the order is well defined), so two bufts that answer the same
   string are ONE entry and every tensor created for the second device lands in the first device's context and buffer.
   The pool's mirror buft returned a constant `ROCm_KV_Pool_Mirror` and did exactly that; the windowed KV's region buft
   reuses the plain CUDA getter (`ROCm<N>`) for this reason.  Any new per-device buft in this tree: make the name carry
   the device (and check `allocation` of a >1-device cache), or the collapse is silent until something reads the wrong
   device's memory.
2. **One counter block per DEVICE, never one per cache.**  A resolve node runs on its layer's device, so everything it
   reads/writes must be there: `kvpool_ctr` (epoch, claim cursor, miss list) was created once in the first pool layer's
   context, which put device-0 memory in every other device's `GGML_OP_KV_POOL` node.  `ggml_backend_cuda_device_supports_op`
   refuses an op whose CUDA-buffer source is on another device and the CPU backend has no `KV_POOL`, so *no* backend
   claims the node and the scheduler aborts at load -- `GGML_ASSERT(*cur_backend_id != -1)` in `ggml_backend_sched_split_graph`,
   inside the `--fit` probe's graph.  A cross-device src is a load-time assert, not a wrong answer.
3. **A per-device placement is invisible unless the report says so.**  The pool stats line now prints the number of
   counter blocks it summed; a 2-device arm reporting 1 means the blocks (or the bufts) collapsed, which is the defect
   above surfacing in a number instead of an abort.  `gate-kv-residency.sh` asserts it, and the enable line reports the
   device span.
4. **The pool is not the cause of every 2-device large-context fault.**  On the reported arm (`-c 204800`, 2 devices,
   `-ncmoe 48`, mmap) prefill faults `rms_norm_f32` on GPU 1 (`HSA_STATUS_ERROR_MEMORY_FAULT` + `gfxhub page fault`)
   right after `moe_cache_evict_slab_range` discards 5.4-5.8 GiB of arena tables, and it **reproduces with
   `LLAMA_KV_REGION_POOL=0`**; `-c 131072` (2 devices) and `-c 204800` (1 device) are clean.  Before blaming block 15 for a
   multi-device fault, run the same arm with the pool off.  (**Fixed in r214**, `TODO` #143: the fault was the
   slab's narrow-2 draft-compute region being handed out with its top unmapped -- see the r214 section below.)


## r214 -- the slab's narrow-2 region is MAPPED for its whole extent; the draft compute term is a measured floor

1. **A slab region handed to a graph must have physical behind EVERY byte of it.**  Reserved VA is not presence: the
   slab reserves `size` but maps `mapped`, and the gap (`va_spare`, deliberately left for `slab_extend`) is unmapped VA.
   `ggml_cuda_slab_narrow2_commit_locked()` mapped the narrow-2 (MTP draft compute) region **only when it lay entirely
   above `mapped`**; in the documented shortfall case, where the region straddles `mapped`
   (`narrow2_off < mapped < narrow2_off + nf2`), it took the "arena gave it up" branch and mapped **nothing**, so the
   top of the draft's compute buffer was absent VA and the first device access past `mapped` was an unrecoverable
   `HSA_STATUS_ERROR_MEMORY_FAULT` (no XNACK on RDNA4).  The commit now maps whichever part of
   `[off, off + nf2)` has no physical behind it and advances `s.mapped`.  The `min(mapped, ceiling)` clamp in
   `ggml_cuda_slab_arena_top()` is therefore **load-bearing**, not defensive (`mapped` ends up above `narrow2_off`).
2. **A page fault inside `rms_norm_f32` is not evidence of a host read.**  On the #143 arm the kernel is
   `rms_norm_f32<1024,1,false,false,false,false>` -- the plain instantiation, no fused `mul`/`add` -- over the MTP
   head's `mtp_h_state` `reshape_3d(inp->h, n_embd, hc, n_tokens)`, with **both** `src0` and `dst` in `ROCm1`
   (`host=0`).  The `per_layer_token_embd` host override (H1) and the block-12 arena evictions logged immediately
   before it (H2) were both innocent: the clean `-c 131072` arm evicts *more* (9.4 vs 5.7 GiB), and the fault address
   was inside the *draft's* compute region, `/proc/<pid>/maps` showing the slab's `---s /dev/dri` mapping ending
   exactly at the mapped boundary.  Attribute a fault by address-to-buffer (`-lv`/maps/slab line), never by the
   kernel name.
3. **The draft context's compute buffer grows with the draft's CONTEXT, in 256 MiB lumps -- a flat tier is wrong.**
   Measured per device (gfx1201, Flash-Next GSQ-RCO IQ3_S + shared `mtp-...-Q4_K_M`, `-ub 4096`, q8_0 KV):
   0.75 GiB at `-c 32768`/65536, 1.00 at 98304/131072, 1.25 at 163840/196608 (and 1.25 at 204800 on ONE device),
   1.50 at 204800/229376, 1.75 at 262144.  The old 512/768 MiB `compute` tier in `common_draft_reserve_estimate` was
   measured at `-c 16384`, i.e. 2-3.5x short at any real context, which is what pushed narrow-2 across `mapped`.
   The term is now the measured **floor** and stays a floor on purpose: over-estimating is not free either -- on a box
   with no VRAM left `slab_extend` cannot reclaim the surplus, so it stays unmapped, while a shortfall costs only a
   bounded, one-time arena yield that the pin now maps correctly.
4. **A shortfall is cheaper than a surplus on a full box.**  With the estimate corrected, the repro arm moved from
   the arena-yield branch to the clean spare branch and used *less* VRAM (32.82/33.11 vs 32.89/33.24 GB, 1.50 GiB
   pinned either way), because the yield path had mapped the region's tail on top of what the arena already held.
   Do not "safety-margin" this reserve upward without measuring: the margin comes straight out of the MoE arena.

## r215 -- the host pool's fill and backing (block 09/12 amendment)

**Measured facts.  Do not re-derive; do not "simplify" either mechanism back.**

1. **`--load-mode mmap` and `--load-mode none` read the SAME bytes.**  `/proc/PID/io read_bytes` is 54.6 GiB in
   both arms at 1 and at 2 devices, and the 48 GiB host-expert set lands in RAM in both.  A startup-time
   difference between them is a *mechanism* difference, never an amount-of-work difference -- measure the bytes
   before theorising.
2. **A mirror pool's fill order is content-free, so it must be read sequentially.**  When the budget covers the
   whole host tensor the pool is a MIRROR (`slots == n_experts`), its memory is one contiguous
   `n_experts * host_bytes` allocation, and the GGUF expert layout is expert-major at that same stride (slot `e`
   == file offset `file_offs + e*host_bytes`).  One sequential read of the tensor is 6.5 GB/s against 4.5 GB/s for
   one `host_bytes` `memcpy` per heat-ranked slot out of the demand-faulted mapping, and it avoids faulting a
   second copy of the bytes into a page cache that cannot be kept beside the pinned mirror (48 GiB: 10.8 s ->
   7.4 s).  A *partial* pool is a different thing -- there the prior decides the content, so the ranked path stays.
3. **`cudaHostAlloc(Portable|Mapped)` pins page by page, and pinning an EMPTY region is the expensive half.**  48 GiB
   = 5.4 s, and it allocates+pins a zero page for every 4 KiB page the fill then overwrites.  `mmap` +
   `MADV_HUGEPAGE` (2 MiB pages) + `cudaHostRegister` **after** the fill is ~1 ms plus page allocation the fill
   needed anyway.  Same UVA device alias, same in-place cold read, same cross-device D2D fill source.
4. **`cudaHostRegister` needs `Mapped`, not just `Portable`.**  A `Portable`-only registration returns success and
   then faults the GPU on the first in-place cold read ("Page not present or supervisor privilege") -- found at
   block 09, invisible in the log.  This stack has no XNACK, so an unregistered/unmapped range is not readable.
5. **The pin is tied to the VMA: `cudaHostUnregister` alone does NOT release it** (2.0 of 256 MiB re-readable after
   unregister, and it is not deferred -- `archive/work/host-expert-eager-push/` P4 §5).  The release is
   `unregister` + `munmap`; verified end to end (`AnonHugePages` 48.97 GiB -> 0 at teardown, `MemAvailable` back to
   baseline).  Anonymous THP is not a hugetlbfs carve-out, so nothing is retained.
6. **Slot recycling must never touch the pin, and does not.**  A pool slot is index bookkeeping inside the one
   per-tensor allocation, so churn pins nothing: with an 8 GiB partial pool and 35 559 slot fills, exactly
   144 `hostRegister` / 144 `hostUnregister` (one per host tensor).  Any future design that allocates per SLOT
   re-introduces pin/unpin churn and is wrong.
7. **A registered source under the expert cache is NOT automatically a prefill loss.**  The archived -22 % 1-device
   prefill regression (`archive/work/host-expert-eager-push/P4-ROOT-CAUSE.md` §5d) was a *pageable mapping*
   registered in place, whose fills became synchronous DMA; it does not transfer to a pinned mirror whose transport
   is unchanged.  Re-measured after this change: prefill 999.7 t/s (registered THP mirror) against 984.0 (the r214
   `cudaHostAlloc` pair), and the multi-slot band at B=8 is 121 against 118 t/s, not the r210 ~50 collapse.  Test
   both shapes anyway -- the mechanism is close enough that it must be re-measured, not assumed.
8. **Startup timing is sensitive to `MemAvailable` at the FIRST pool build.**  `MOE_HOST_POOL_RAM_PCT` (85) of
   `MemAvailable`, measured once, sizes the pool: a run started while the page cache still holds the model sizes a
   smaller pool (45 504 slots / 29.6 GiB against 73 728 / 47.9 GiB) and then takes the loader's pinned-master path
   (a 13.5 s load).  Drop the page cache (`POSIX_FADV_DONTNEED`) before any start-time comparison.

## r216 -- block 15 Goal B is a DECODE placement: its page-fill path is ~50 % of the prompt on a KV that fits

**Measured facts.  Do not re-derive; do not enable the pool on a prompt-heavy server.**

1. **The pool's page-fill path is per DECODE step, so at PROMPT time it is pure overhead.**  `LLAMA_KV_REGION_POOL=16384`
   against the plain device KV, one RX 7900 XTX, `-c 204800`, Qwen3.8-Flash-Next GSQ-RCO-IQ3_S (`-ncmoe 48`, q8_0
   KV): **pp32K 427.5 against 829.3 t/s**, pp8K 463 against 769, **pp113K 409.2 against 808.2**.  A prompt chunk
   writes every cell of its range, so the fill it pays is never reused; a decode step re-reads the same pages.
2. **The pool's decode gain is exactly the arena it frees, and it is small.**  Same run: +12 % decode (59.4 against
   52.8 t/s over 192 tokens) for 2.3 GiB of arena (12 084 against 9 742 MiB).  The KV here is ~13 KiB per token
   (`2.6 GiB` at `-c 204800`), so on a 24 GiB card the pool's fittability case does not arise; Goal A's window
   (`LLAMA_KV_RESIDENT_CELLS`) buys the same arena with no kernel change, free up to its own depth (530.1 against
   808.2 at pp113K once it spills past 64K).
3. **Prefill at this shape is not thread-, ubatch- or host-RAM-bound -- measure before tuning any of them.**
   `--threads` 7/12/14/16 = 769/771/769/769 t/s at pp8K; `-ub 4096` against 2048 = 771 against 769;
   `--cache-ram 6144 -> 0` = 768; `GGML_CUDA_SLAB_HEADROOM_MIB 1536 -> 768` = 776 (+1 %) for +843 MiB of arena.
   The expert arena is a decode feature: `MOE_EXPERT_CACHE_MIB=4096` = 789 pp / 38 tg, `=0` = 890 pp / 10.8 tg.
4. **A fixed `--limit-host-experts N` makes a config host-RAM-specific and refuses to start below it.**  `N` is
   refused when the pinned master would exceed 90 % of the memory available at load (`54` on a 59 GiB host:
   `more than 90 % of the available memory (59 GiB)`), and a later flag on the command line cannot rescue it
   because the refusal happens while the argument is parsed.  Use `auto` in anything shared.
5. **Pool on and pool off can differ by a near tie; the DEFAULT is the arm with the strong property.**  Same prompt,
   temp 0, 128 output tokens: pool / plain / window all produced `00f5b00b804d2ebf`.  A 32K prompt at 48 output
   tokens flipped one token between pool-on and pool-off (stable across two prompts).  The shipped default is the
   plain device KV -- block 15 unset, the arm `patches/README.md` calls byte-identical to a build without the block
   -- and `scripts/gate-kv-residency.sh` is the arbiter for the pool arm, not a sampled text hash.

## r217 -- the gfx1201 stall is a FIXED cost (it only pays on short grids); a parked slot bounds `seq_rm`; every `getenv` is read once

**Measured facts behind issue #273.  Do not re-derive; do not "optimise" the arms that are neutral.**

1. **The gfx1201 grid-size stall is a fixed ~10 us added to a short kernel, so padding it only pays below ~2048
   blocks.**  `archive/work/rdna4-grid-stall-pad/graphgap.hip` (2000 dependent trivial kernels in a captured HIP
   graph, 256 threads/block), extended to large grids: 256 blocks **12.63 us** against 257 **2.34**, 512 **13.63** /
   513 **2.71**, 1024 **11.62** / 1025 **3.32**, 1536 **6.42** / 1537 **4.21** -- then 2048 5.08 / 2049 4.83, 4096
   7.54 / 4097 7.52, 8192 12.71 / 8193 12.76, 16384 24.41 / 16385 24.49, 32768 46.06 / 32769 45.86.  Above ~2048
   blocks the kernel's own runtime hides the fixed cost and the pad changes nothing (it stays harmless and
   bit-identical).  A block-value multiple of 256 or a work-item multiple of 65536 both trigger it; a launch can
   hit the condition and be **neutral** -- check the grid size before claiming a gain (issue #273: the reporter's
   256-block grids pay, a 27B model's 16384-block grid does not).
2. **A parked idle slot sits at the TOP of a unified cache, so `used_max_p1()` is the cache size.**  `seq_rm()`
   must therefore scan to `used_max_p1_below(park_bound())` for a sequence that is **not** parked (a parked
   sequence holds only cells at or above the bound, and no other sequence has a cell there).  Measured on a
   131 072-cell unified cache with 4096 live cells and a 64-cell parked sequence: **42.127 us -> 1.358 us** per
   removal (31x); the parked cells, a full removal of the live sequence and the unpark are unchanged
   (`scripts/gate-park-idle-slots.sh` is the end-to-end arbiter).  The negative `seq_id` (any-sequence) branch
   must keep scanning the whole used range -- it has to remove parked cells too.
3. **A per-op `getenv()` is a measurable cost AND a correctness hazard, so every read is latched once.**  One call
   is **21.74 ns** here (a latched `static` 0.18 ns) and much worse on Windows (the CRT lock), but the harder
   reason is that the ROCm 7.14 `rocprofiler-register` calls `setenv()` during early init and can reallocate the
   process environment under a concurrent `getenv()` (ROCm issue #10196), which once made an env gate read as
   unset -- see the comment at the top of `src/models/qwen4exp.cpp`.  Audit with an `LD_PRELOAD` interposer that
   counts `getenv` per name on a real run: before r217 `GGML_CUDA_MMB_TYPES` (per matmul) was read 7 657-22 913
   times per run, `GGML_CUDA_SLAB_STATS` 7 681-15 552, `GGML_STAGE_TRACE` 144-240, `GGML_CUDA_FA_STAGE_CHUNK_DEBUG`
   125-431.  `getenv("X") ? atoi(getenv("X")) : D` (the doubled read) is gone from the tree; a read is
   `[] { const char * e = getenv("X"); return ...; }()` into a `static`, or a file-local accessor when several
   functions need it.  A `static` inside a **template** function is one copy per instantiation (that is the only
   multi-read left, e.g. `GGML_CUDA_DISABLE_MMQ_ROUTED` in `mmq.cuh`).
4. **Without media, the server slot's own token vector IS the prompt the drafters read**, so `slot.spec_prompt`
   must only be copied when `tokens.has_mtmd`; the copy is 2.8 us/step at 10.5K tokens and **28.8 us/step at
   110K**.  The drafters read it inside `common_speculative_draft()`, before the batch changes the prompt.

## r218 -- the drafter's KV must stay contiguous with the rows it ingests (issue #272)

**Measured facts.  Do not re-derive; a rejected draft batch is a HTTP 500 on the user's request, not a slow
path.**

1. **The ingest tail leaves the draft's cache ending below the first row a PARTIALLY reused prompt ingests.**
   `LLAMA_DFLASH_INGEST_TAIL` (block 12, r130) and `LLAMA_MTP_INGEST_TAIL` (block 02, r118) ingest only the
   last `n_swa` rows of the prompt.  A slot keeps a prefix from an earlier task, so when the new turn adds
   more than `n_swa` tokens the new prompt's window starts *past* the rows the draft kept: the draft's
   `seq_pos_max` is the reused prefix's end (P-1) while the first row the task injects is at
   `ingest_from = prompt_end - n_swa > P`.  The batch check requires `seq_pos_min == seq_pos_max + 1`.
2. **A strict (non-M-RoPE) draft is REJECTED; an M-RoPE draft is allowed to jump.**  `llama_batch_allocr::init()`
   has two branches: `n_pos_per_embd > 1` (M-RoPE: `X < Y` for tokens, `X <= Y` for embeddings) and the
   `Y = X + 1` rule otherwise.  A DFlash/DSpark draft carries `dflash.rope.dimension_sections` when it drafts
   for an M-RoPE target (`LLAMA_ROPE_TYPE_MROPE`, `n_pos_per_embd = 4`) and takes `NORM`/`NEOX`
   (`n_pos_per_embd = 1`) otherwise, so the same defect is a 500 for one draft class and a silently
   context-starved batch for the other.
3. **Reproduced on gfx1201, issue #272 (the reporter's shape).**  35B-A3B UD-Q4_K_M + the 35B-A3B DFlash2-Q8_0
   draft, `-np 2 --kv-unified -ctxcp 4 -fa on -c 49152`, three turns of `prompts/prose-rdna-boosts.txt`
   (+5246 tokens per turn, 2048-position window, `prompt_n` 5246/5251/5224, `cache_n` 0/5242/10516): the
   pre-fix build answers turn 2 with `HTTP 500 decode() failed: failed to process speculative batch` and
   `inconsistent sequence positions ... sequence 0 ... X = 5241 ... starting position of Y = 8445`.  The
   reporter's DSpark arm measured the same message at `X = 1995, Y = 2242` (prefix 1996 reused, window start
   2242).  `LLAMA_DFLASH_INGEST_TAIL=0` passes on the pre-fix build - the tail is the trigger.
4. **A metadata-only non-M-RoPE copy of an M-RoPE draft is a faithful stand-in, so the two draft classes can be
   compared on one model pair.**  Zeroing `dflash.rope.dimension_sections` (`[64,0,0,0] -> [0,0,0,0]`) switches
   only the batch-check rule: the derived draft measures the **same acceptance** as the original
   (0.42424 = 14 accepted / 33 generated, 0.20833 = 10/48, 0.35135 = 13/37, mean len 2.56/1.77/2.30) and the
   same text on every turn, i.e. an M-RoPE rotation with one section is numerically what the strict backbone
   does.  That is what makes `scripts/gate-spec-reuse-window.sh` able to hold one model pair for both rules.
5. **The stale prefix carries nothing, so dropping it is exact.**  Every row below `ingest_from` is masked for
   the prompt-end query (`GREEDY-PURITY.md` §53), and the injection batch does not contain them, so they can
   neither be attended nor re-ingested: the only exact repair is to drop them and start a fresh window at the
   first ingested row, exactly as the first-ever prompt of a session does.  After the repair the tail and
   no-tail arms produce byte-identical text on every turn of the reuse replay (6/6 requests across the two
   draft classes) - the drafter proposes, the target verifies.
6. **A draft found AHEAD of the target is rejected by BOTH branches**, so the overlapping tail is trimmed
   unconditionally (`pos_max + 1 > pos_first`): a checkpoint restore can put the draft past the position the
   task resumes from, and this call re-ingests those positions anyway.

## r219 -- the host pool's disk tier is an OPTIMISATION, and a failed fill is never resident (issue #274)

1. **A pool fill must never depend on a tier the platform may not have (r219, block 12).**  The bounded host
   pool's master IS the read-only model mapping, so `src_host` (the mapping) can always serve a fill; the pool's
   DISK TIER (`MOE_HOST_POOL_DIO=1`, `pread`/`O_DIRECT`) only decides whether the page cache is populated.  The
   file I/O is a compile-time stub on Windows (`pool_open_file` returns -1, issue #130), so treating it as
   required failed every fill there.  Measured on Windows (reporter, r209, `--load-mode mmap --n-cpu-moe 47`
   `--host-experts pool`, 38 GB free against a 40 GB host expert set): **7281 x** `access_locked: async fill
   failed ... : no error` and `fill_slot_from_host: no pinned host source ... (pool and bounce both
   unavailable)`, a 4996-token prompt at 20.4 t/s and a ONE-token generation.  r212 had already made
   `MOE_HOST_POOL_DIO=auto` mirror `--load-mode` (a mapped load fills buffered), which removes the default path
   into this -- but the mode is one env var away, and the failure mode was silent corruption, not a slow run.
   The fix: the disk tier is best-effort everywhere (`pool_fill_slot_locked`, `expert_host_src`'s bounce ring),
   a requested O_DIRECT is refused with one WARN on a platform without the file I/O, and a pool whose disk read
   fails falls back to the mapping instead of marking itself failed.  Reproduced on Linux with the new
   `MOE_HOST_POOL_DIO_FAULT` diagnostic; the pre-fix build then shows the issue's exact two messages and
   garbage text (`b7e2732392540006`), the fixed build is byte-identical to the shipped arm (`a99ccbd97ecb`).
2. **A failed fill must ROLL BACK: a slot is never resident because a copy was attempted.**  `access_locked()`
   registered `t.expert_slot[expert]` and returned the arena pointer BEFORE the copy, so a fill that could not
   be serviced left a resident slot holding whatever the arena had there.  That is a wrong answer, not a slow
   one (the reporter's one token, `h=0.4939`), and its only trace was `cudaGetErrorString(cudaGetLastError())`
   = `"no error"` -- `cudaGetLastError()` is empty for a host-side failure, which is why the message looked like
   a driver note.  The admission now rolls back, the table is marked `fill_failed` and stands down wholesale
   (`moe_cache_update_host`, `moe_cache_get_table`), so the scheduler serves the op from the host master, and
   the census reports `FAILED fills: N` plus `the slot is NOT resident` per table.  Measured with
   `MOE_HOST_POOL_FILL_FAULT`: 2790 rolled-back fills and **byte-identical** text to the shipped arm
   (`a99ccbd97ecb`); the first cut of the rollback alone was NOT enough -- it left the eager take-over emitting
   pool-cold ids for a table whose arena was empty, and that arm ran garbage (`b7e2732392540006`), which is why
   the stand-down is part of the fix and `gate-host-pool.sh` gates the text, not just the log.
3. **A one-off tier decision at INFO is invisible.**  `common_log_get_verbosity()` maps `GGML_LOG_LEVEL_INFO`
   to `LOG_LEVEL_TRACE` against a default threshold of `LOG_LEVEL_INFO`, so at the default verbosity an operator
   sees only WARN and above.  The reporter's log therefore contained every FAILURE and none of the decisions
   that produced them: no `host experts:` tier census, no `host pool enabled`, no per-table `host pool (...)`
   build line, no fill mode.  `report_host_expert_placement()` is now WARN (it is a one-off decision line, the
   same class as `MoE expert cache (auto): arena ...`) and it carries the pool-source count
   (`141/141 pool sources registered`), which is the first number a "the pool never appeared" report needs.
   Do not "quiet" a decision line back to INFO: at the default level it does not exist.

---

## r220 -- a backend-count gate is inert under `-sm tensor`; the copy event must be per copy (PR #268)

1. **`-sm tensor` is ONE backend, whatever the device count.**  `llama_prepare_model_devices()` wraps
   N devices in a single **Meta device** (`ggml_backend_meta_device`), and `llama_context` creates one
   backend per `model.devices` entry, so anything gated on "how many non-CPU backends" sees `1` under
   `-sm tensor --device 0,1,2` (measured, 3 x R9700).  A feature that pairs *backend schedulers* can
   therefore never reach a tensor split, and no amount of extra GPUs changes that: pairing devices inside
   the Meta device is a different feature.  The same gate is also inert under **plain `-sm layer`**,
   because the scheduler's own pipeline parallelism is enabled there
   (`pipeline_parallel = n_devices > 1 && all layers on GPU && -sm layer && offload_kqv &&
   !has_tensor_overrides() && moe_cache_size == 0`), and a two-lane prefill declines when it is on.  The
   reachable case is a layout where pipeline parallelism is *already* off -- `--override-tensor` -- which is
   exactly the layout the PR was measured in.  **`-sm row` is not a layout on this backend at all**: it fails
   at model load (`llama_model_load: device ROCm0 does not support split buffers`), so the r220 artifact's
   "`-sm row`, `-sm none` with 2+ GPU backends engage" line was an inference from the gate and is wrong for
   ROCm -- `-sm none` is one GPU backend, i.e. the `-sm tensor` case.  **`-sm layer` plus any `-ot`, and
   nothing else, is reachable here.**  **Measure a scheduler feature's REACHABILITY
   before its gain** (`optional/two-lane-prefill/README.md` carries the table).
2. **A queued HIP stream wait binds to the LATEST record, so a shared copy event re-binds consumers.**
   The cross-backend split-input copy used to record the backend's shared per-device `copy_event` and make
   the destination stream wait on it.  A copy still in flight re-records that event, and a wait that is
   already queued then resolves against the newer record: independent cross-backend copies serialize
   against each other.  @slavap hit it as ~0 % overlap in a two-lane prefill (PR #268) and it is reachable
   in this delivery wherever two copies overlap, which the pipelined layer split does.  r220 makes the
   ordering event **per copy** (`cpy_tensor_async_nowait` + a per-source-backend ring, re-recording a slot
   only after its previous record completed) and keeps `GGML_SCHED_COPY_EVENT=shared` as the A/B kill
   switch.  Bit-exact: same-seed greedy text identical on `-sm tensor`, `-sm layer` and `-sm layer -ncmoe`,
   and throughput-neutral (pp8192 3-GPU `-sm layer` 6311 vs 6317 t/s) -- the repair is a decoupling, not a
   speedup on a box where the copies were not actually contending.
3. **An overlap technique buys nothing on devices of equal throughput -- when both ubatches are ALREADY wide.**
   With identical cards every split of a wide ubatch is already the critical path, so a second *whole* ubatch
   in flight adds queueing and no work: measured +0.03 % (MoE 35B-A3B Q8_0) and nil (dense 27B Q6_K_XL) with
   the pair engaged (`pair decided: splits a=4 b=4, interleaved=1`) on 3 x R9700.  The submitter's +34 %/+13 %
   are a 7900 XTX (gfx1100) + V620 (gfx1030) pair, whose devices differ ~3x: there the overlap hides an
   imbalance.  A device-heterogeneous layout is not one of this delivery's validated configurations, which is
   why that feature is `optional/`, not a block.  **The r229 cut qualifies this rather than overturning it**
   (see *"no release, r229 tree"* below): pair two halves that were SPLIT out of one wide ubatch and balanced
   devices gain +27 %/+41 %, because one unsplit ubatch serializes the layer-split pipeline and the split hands
   the second lane work to do.  The pair *alone* is still worth nothing -- the split is what unlocks it.
4. **A `hip*` spelling in `ggml-cuda.cu` needs the `#ifdef GGML_USE_HIP` form.**  `ggml-cuda.cu` still
   builds under nvcc for CUDA, and `hipEventDefault` / `hipEventElapsedTime` have no `cuda*` alias in this
   toolchain (using `cudaEventElapsedTime` in a HIP build fails with `use of undeclared identifier`).  The
   file's precedent is `#ifdef GGML_USE_HIP ... #else cudaEventDefault ... #endif`; an unguarded `hip*`
   call breaks the CUDA build (PR #268 did, fixed in the optional artifact's rebase).  Related: do not
   switch every backend event to a **timing** event to serve one diagnostic -- create the timing event only
   on the debug path, or the cost lands on users who never set the knob.

## r221 -- the sync-free decode remap is worth what the SYNC it replaces costs, so it grows with the device count; a covering pool is a legitimate cold source

`TODO` #269's acceptance condition ("the remap must be tensor-split aware") is met by forwarding
`moe_cache_remap` from the Meta backend to each simple device, exactly as `moe_cache_update` / `take_over` /
`gather` already are.  The facts, all measured this session on 35B-A3B Q8_0 `-ncmoe 40`, byte-identical
greedy text throughout:

1. **The gain scales with the device count because the eager path synchronizes the WHOLE meta backend.**  The
   remap replaces `ggml_backend_synchronize(input_backend)` + the ids readback (a full meta synchronize, i.e.
   every child device) with a kernel plus a host-side policy one call late, so its value is that sync's cost:
   **+9.9 %** at 1 GPU (57.6 -> 63.3 t/s, `-sm tensor` == `-sm layer`), **+16.1 %** at 2 (64.0 -> 74.3) and
   **+28.4 %** at 3 (59.4 -> 76.3) at `MOE_EXPERT_CACHE_MIB=8000`.  The r203 single-device numbers (+8.4 %
   pinned, +4.8 % pool-cold) still reproduce on their own protocol (`llama-batched-bench -npp 64 -ntg 512`,
   Q4_K_M, `MIB=8000`: 61.2 -> 67.6 t/s).
2. **Nothing in the remap is device-shared, which is why the forwarding is a delegation and not a redesign.**
   The routing ids are global expert ids; each device maps them through its OWN expert -> slot map into its
   OWN `remap_dev`, and the cold half of the map is expressed with THAT device's slot count -- `t.slots`
   differs per device, so a single shared map would name different experts.  `check_only` must answer for all
   devices before anything commits (the scheduler's skip-copy decision is all-or-nothing); a decline in the
   committing pass falls through to the eager path, which resets `remap_n_used`, so a partial commit can only
   cost a synchronize.
3. **A host pool that COVERS its tensor is a cold source, so `dr_pool_ok` makes the cold id `slots + pool
   slot`.**  That is the default `-lme auto` layout on a mapped load ("a pinned per-EXPERT pool of the whole
   host expert set (a MIRROR)"), and it previously took the eager path for want of a cold encoding even
   though every expert was pool-resident: +9.9 % on that config.  It is expressible ONLY under coverage: a
   sub-complement pool declines every offer (measured `0 of 3840 served`), because a device-side map cannot
   express a miss whose expert is neither in the arena nor in the pool.  Do not "fix" that by pointing the
   unmapped expert at `slots + e` -- the kernel would resolve it against the pool base.
4. **A covering pool's expert -> slot map is WRITE-ONCE after sizing, and that is the freshness answer.**
   The demand path never evicts a resident (`pool_fill_empty_locked` only takes EMPTY slots), and the one
   re-rank (`pool_rerank_from_heat_locked`) frees nothing and fills nothing when `want` is every expert -- so
   r221 makes that early-return explicit, and `keep_all` in `expert_host_src` no longer retires a slot for a
   mirror.  Coverage is therefore a property of the BUDGET, not of the run, which is what lets a one-call-late
   map be safe without a synchronize (`TODO` #269's "layout epoch or freeze" -- this is the freeze, by
   construction).  Count LIVE slots (`slot_state == 1`): a slot retiring into the arena still has an index
   and cannot serve a cold read.
5. **`dr_pool_ok` is latched once, at sizing, because a captured graph bakes the take-over.**  A decision that
   could flip later would leave a replayed node reading a baked `remap` pointer while the host had taken the
   eager path -- a wrong read, not a slow one.  Gate any future extension of the remap's scope on the same
   question: "is this answer constant for the shape once `g_sized`?"
6. **Two independent kill switches compose, and the counter is the oracle.**  `MOE_HOST_POOL_COLD=0` and
   `MOE_HOST_POOL_COLD_MULTIDEV=0` both make `pool_cold_geometry_ok` false, so a covering pool's remap stands
   down with them; `GGML_MOE_DEVREMAP=0` is the remap's own.  The exit report's `device remap: N tables
   served, M of K offers declined` line is what tells "no gain" from "not engaged" -- read it before
   concluding anything about this path.
7. **A decode A/B on this box needs the ABBA protocol even for a 20-second arm, and a *prefill* rate that
   halves is the tell that the run is not comparable.**  This session's first pass at the remap read
   -2.5 % on `llama-cli -n 300` (three reps each, alternating, disk delta 0-29 MiB) and the same command
   later read +8.9 %; the discriminating symptom was prefill 1214 vs 2716 t/s at an *identical* arena size,
   i.e. a run in a degraded state that no counter in the decode rate shows.  Protocol that worked: 2+ ABBA
   cycles per arm pair, `MemAvailable` and used VRAM recorded per run, the model page-cache-resident (a 74 GB
   model against 91 GB of RAM re-reads itself every arm and is not measurable this way), and the
   `device remap:` census read before drawing a conclusion -- "no gain" and "not engaged" look identical
   without it.

## (no release, r221 tree) -- the mirror's prefill transport is the GATHER, it is serialised at the link's rate, and the alternative transports rank by STREAM, not by bytes

Measured 2026-11-05 on `soar` (1 x R9700, gfx1201, ROCm 10.1.0, 91 GiB host RAM, no cgroup), tree
`75059f1c4` (r221), model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, `-lme 47` + `MOE_HOST_POOL_RAM_PCT=0` (plan line:
"the whole host expert set (a MIRROR)"), 52 512-token prompt, greedy.  Record:
[`archive/work/host-expert-mirror/`](archive/work/host-expert-mirror/RESOLUTION.md) (the campaign closed here;
`WORKLOG.md` 261105.008).  No delivery change.

1. **On a mirror the prefill transport is the DEVICE GATHER, so no staging/miss-path number describes the
   shipped default.**  `sched_input_gatherable` mode 1 requires `moe_cache_pool_backed()` (the pageable-master
   `pool` tier) and a width above the cache band, so every pool-backed table above the band is gathered:
   measured `device gather: 1410 calls (1410 pool-backed), 0 routed experts copied ... (neither tier)` and the
   staging census **absent**; `GGML_MOE_CACHE_STAGE=0` changes nothing (1616.2 vs 1618.8 t/s).  Read the exit
   census before attributing any prefill cost to `moe_cache_upload_pageable` or `moe_cache_stage_table`.
2. **The gather moves the routed expert bytes at the link's ceiling, so it is not a bytes problem.**  Its kernel
   is **34.9 %** of the prefill's GPU time at `-ub 6144` (10.91 s of 31.22 s; **55.0 %** at `-ub 2048`), and it
   reads **329.19 GiB from the pool + 126.50 GiB from the arena** per 52k pass: 250.7 MB of pool reads per call
   in 7.52 ms = **33 GB/s**, against this stack's **28.6 GB/s** bulk-H2D calibration.  An in-place alias moves
   the same bytes at the same rate; the only thing it could buy is overlap.
3. **The gather is SERIALISED (kernel/wall 0.95), which is where the recoverable loss is.**  Sum-of-kernels
   ~= wall means it is a compute-stream node adding its full time to the prefill.  The transports rank by
   *stream and engine*, and the ranking inverts with the shape: 1 GPU `-ub 6144` gather 1627 / r114 single-launch
   staging kernel 1418 / per-expert copy-engine 1842; `-ub 2048` gather 1016-1021 / kernel 872-875 / CE 742-907;
   `-ub 16384` gather 1945 / kernel 1736 / CE 1788; 3 GPU `-sm tensor` gather 2359 / kernel 2075 / CE 2053.  The
   CE win is a **1-GPU width band (5120-7168)**; by 8192 its overlap stalls (kernel/wall **0.80** -- 20 % of the
   wall idle with no kernel running).
4. **A single-launch staging kernel that streams a whole 341 MiB table STARVES the compute kernels.**  At
   identical dispatch counts the same compute kernels run **3-41x** slower in the arm where
   `moe_cache_stage_kernel` (144 blocks x 256 threads) is concurrently resident: `mm_ids_helper_512_10` 0.051 ->
   2.081 s, `quantize_mmq_q8_1` 0.336 -> 1.657 s, `mul_mat_q_routed_compact<20>` 1.291 -> 4.438 s; total GPU time
   43.55 s vs 31.22 (gather) and 27.57 (many short copy-engine blits).  **r114's +34 % for that kernel was
   measured against a pageable master that paid a host memcpy into a pinned bounce per copy**; with the pool
   alias readable in place there is nothing to amortise, so the same kernel is now the *worst* option at every
   width and device count measured.  Do not re-derive "+34 % for one launch" from the r114 record.
5. **A pinned master's staging is one kernel launch on the COPY stream and genuinely overlaps** (kernel total
   33.80 s > wall 31.17 s) -- it is the "already overlapped" control, and the pool mirror's whole gap to it is
   ~5 % prefill for 46.84 GiB of duplicate RAM.  Its own engine preference is the opposite of the pool tier's
   (kernel 1820-1825 > per-expert CE 1545-1547, interleaved), confounded by 54 of 144 tables declining staging
   entirely (`ok=783 unusable=477`, `TODO` #271).
6. **`-ub` is a bytes-per-token lever on a mirror, and the arena's SIZED residency overstates what it serves.**
   Every prefill ubatch routes essentially the whole table, so bytes/token ∝ 1/ub while the ubatch count ∝ ub:
   `-ub 2048 -> 16384` is 1019 -> 1942 t/s (gather), at the cost of the arena (21.5 -> 16.4 GiB).  Enlarging the
   arena alone is nearly inert (`-c 65536`: pool 329 -> 312 GiB, +1.2 %) because only ~28 % of routed offers are
   served from it -- the slots fill on demand during the pass (`hits=951 fills=2019`), so size it *and* seed it.
7. **A block-ordered set of "reps" is not a measurement on this box** (this extends r221 fact 7).  Block-ordered
   arms drifted monotonically by **16 %** over one session (staging kernel 1241 -> 1434, copy-engine 1549 ->
   1844) while the interleaved gather control stayed at 1616-1629 (+-0.3 %), and the pinned tier swung 36 %
   across identical censuses.  Only interleaved ABAB within one cycle is trustworthy; `AnonHugePages` was 0 GiB
   in both pool alloc shapes and neither `MOE_HOST_POOL_ALLOC` nor `MOE_HOST_POOL_H2D` moved any arm.

## r222 -- a buffer type may not touch the driver while it is CONSTRUCTED, and a gate's verdict level is part of the gate (issue #275)

1. **A driver call made from a buffer-type CONSTRUCTOR runs in whatever context the calling thread is on, and
   that is not the probed device (r222, block 15, issue #275).**  `llama_kv_cache` init asks every child of a
   Meta (tensor-split) device for the region buffer type
   (`ggml_backend_meta_kv_region_buft(kv_dev, 1)`); the construction-time capability probe then executed its
   whole VMM round trip -- `cuMemAddressReserve` -> `cuMemCreate`/`Map`/`SetAccess` -> unmap ->
   `cuMemAddressFree`, plus the quarantine `cudaMalloc` -- in the LAST device's context.  Measured directly on
   Linux with a `cudaGetDevice()` print at the probe (`HIP_VISIBLE_DEVICES=0,2 -sm tensor`):
   `VMMPROBE device=0 current_context=1 cross=1`.  Linux tolerates it; on Windows/HCC the first kernel launch
   then dies with an access violation in `hipHccModuleLaunchKernel`, which is the reporter's crash and looks
   like a kernel defect rather than a bookkeeping one.  The construction-time answer must therefore come from
   facts already latched at init (`devices[].vmm` + granularity), and the real reserve/map validation belongs
   at ALLOCATION time, where the device is bound and a refusal still falls back to the plain device buffer.
   **The test of a fix here is the log**: a plain `-sm tensor` run must print no `VMM region` line at all.
2. **Bind the device in the MODULE, at every entry point that can reach the driver -- not at the call sites.**
   `alloc_buffer` bound it and the probe did not; `ggml_cuda_vmm_region_free` (reached from a buffer context's
   destructor) did not either, so it unmapped and released VA in a foreign context on every multi-device
   teardown.  A caller-by-caller audit does not survive the next caller; one `vmm_bind_device()` at the top of
   `create`/`free`/`map_span`/`ensure_span`/`clear`/`dump`/`available`/`selftest` and the granularity query
   does.  The numbers that prove nothing moved: same-seed text is identical to the pre-fix build on 1, 2 and 3
   GPUs (`1c5d32ac537d` in all three arms).
3. **A verdict that is logged at INFO does not exist, which makes a FAIL as quiet as a PASS (r222; this is r219
   fact 3 taken to a release gate).**  `GGML_CUDA_VMM_REGION_SELFTEST`'s PASS/FAIL line was `GGML_LOG_INFO`,
   and `common_log_get_verbosity()` maps INFO to TRACE -- so at every default neither verdict appeared, while
   `ENVIRONMENT.md` had said since r178 that the gate "must print PASS ... and must print FAIL with
   `LLAMA_KV_REGION_QUARANTINE=0`".  Two more layers hide a WARN in `llama-cli`: its default verbosity is
   error-only (`tools/cli/cli.cpp` sets `params.verbosity = LOG_LEVEL_ERROR`), and the `--fit` probe wraps the
   logger so anything below its own minimum is rewritten to DEBUG.  **A diagnostic that a gate asks for must
   emit its failure at ERROR** (and its pass at WARN, with the run's `-v` named in `ENVIRONMENT.md`).
   `scripts/gate-kv-residency.sh` now runs both arms (`wselftest` PASS, `wselftest0` with the quarantine off
   must FAIL), so the oracle is proven sensitive on every run.
4. **`release.json.tip` as a script default is a staleness trap after a block amendment (r222, tooling).**
   `scripts/make-patches.sh` defaults its tip from `release.json`, so re-cutting the set after amending a block
   -- without updating `release.json` first -- exported the PREVIOUS series, printed "Regenerated 16 patches"
   and exited 0 while the amendment was nowhere in the delivery.  It now refuses when the fork is checked out
   at a tip that is not `release.json.tip` unless the tip is passed explicitly (and the amended block's commit
   id is still reproducible: the canonical re-cut of the new series prints `Applied tip: 80bfb80be`, matching
   `release.json`).

## r223 -- a fill failure inside the call must decline that call; a pinned-allocation refusal is a LATCH, not a retry (issue #274 follow-up)

1. **The entry guard is not enough: a fill that fails DURING `moe_cache_update_host`'s pass 1 must decline THAT
   call (r223, block 12, issue #274 follow-up @briansp2020).**  `access_locked` rolls the failed admission back
   and sets `t.fill_failed`, but the hook's `fill_failed` guard sits at the TOP of the function, so the call that
   actually failed kept going: pass 2 found a cold id (pinned master) or a pool id (`pool_cold_slot_locked`; a
   pool HIT is rolled back as an arena admission but is still resident in the pool) for the failed expert, `all`
   stayed true, and the hook returned TRUE.  The scheduler then skipped `input_cpy` -- and `moe_cache_get_table`,
   which the consuming op calls a few lines later, sees the SAME flag and returns false, so the op read a buffer
   nobody filled.  That is the reported mid-run corruption (`This` followed by `!`), and it is invisible to a
   text gate whenever the stale `input_cpy` happens to hold the right experts (measured here: identical text on
   the pre-fix build, so the gate asserts the MECHANISM: `takeover=0` under a fill fault, 128/129 pre-fix).  The
   fix is one re-check after pass 1, plus `!t.fill_failed` in `devremap_ok_locked` so the one-call-late device
   remap follows the same wholesale-stand-down rule.
2. **`MOE_HOST_POOL_FILL_FAULT` needs a `late` shape to reach a mid-run failure (r223 diagnostic).**  A fault
   from the first call (`=1`) stands every table down before any take-over question can be asked, which is why
   the r219 arm did not find fact 1.  `=late` (each table's second call on) reproduces "fills that had already
   succeeded, then a failing call" deterministically on any box; a process-wide `=N` start does not, because the
   landing point moves with the box's arena/pool sizing (measured: two runs of the same command, different
   sha).  `=0` is off -- and note the r219 form treated ANY value as "on", so `FOO=0` used to mean "fail
   everything".
3. **A refused pinned allocation is a one-way latch on Windows, and retrying it manufactures a cascade (r223,
   block 12, issue #274 follow-up @DanoPTT).**  `hipHostMalloc`/`cudaHostAlloc` refusing a pinned range there
   surfaces as `hipErrorLaunchFailure` (719), which latches process-wide: every later HIP call -- the fill
   copies and even `hipHostFree` -- fails too (`ROCM-33542` / rocm-systems#14578, relaying ROCm/TheRock#9339).
   The shipment used to retry the per-table bounce `cudaHostAlloc` on every later fill, so one refusal became
   thousands of failures whose message named a different (synthesized) error.  The first refusal now logs once
   with the driver's error string and latches `g_pinned_off`; every later source resolution takes the pageable
   fallback or declines (slow, correct) instead of calling the driver again.
4. **The Windows PINNED ceiling is not `MemAvailable`, and a formula for it must NOT ship unmeasured
   (r223).**  WDDM budgets pinned host pages against the adapter's shared segment and the PAL maps them onto
   EVERY visible adapter without deduplicating, so the usable figure is about `(total RAM / 2) / visible
   adapters` -- measured by ROCm as ~13.4 GiB on a 27.6 GiB box with one adapter and ~6.5 GiB with two
   (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339).  A 47 GiB box therefore pins ~23 GiB while
   `MOE_HOST_POOL_RAM_PCT` (a percentage of MemAvailable) planned ~26 GiB, which is why the reporter's tail
   pools and then every bounce entry were refused.  **A cap derived from that model was implemented and then
   deliberately withdrawn**: it is a heuristic, it cannot be measured here (no Windows box, `TODO` #95), and it
   would cap a healthy box (an APU-only box, or a multi-GPU box where the adapter sum does not behave as the
   model says).  The shipped guards do not guess: the first refused `cudaHostAlloc` latches the pinned tier off
   with one WARN that names the platform cause (fact 3), and the exit census repeats the latch and its error.
   The real figure is the aperture segment's `CommitLimit` (`D3DKMTQueryStatistics`, the same call
   `wddm-mem.cu` already makes for the device free-memory query) -- measure it before adding a ceiling.
   **r233 corrects the premise: the ~9.3 GiB stop is the loaded HIP RUNTIME, not Windows or WDDM.**  Issue
   #278's box had `amdhip64_7.dll` from the Adrenalin driver in `System32`; with the ROCm/TheRock 10.1
   runtime (a different DLL file version) beside the same r229 binary the tier pinned **16 004 / 40 078
   MiB** with 141/141 host tensors and 0 failed fills, where the driver runtime stopped at ~9.3 GiB.  So
   a Windows report's pinned ceiling is not interpretable until the loaded runtime module is named --
   the delivery now logs it at startup (see the r233 entry at the end of this file).
5. **`Private Bytes` for a read-only file mapping is not commit charge, so a 40 GB figure there is a
   measurement artifact until proven (r223, issue #274 follow-up @DanoPTT).**  `CreateFileMapping(PAGE_READONLY)`
   + `MapViewOfFile(FILE_MAP_READ)` makes the GGUF page-cache-backed; the process's commit only grows for page
   tables and for anything that COPIES the range.  On a mapped `--host-experts pool` load the delivery copies
   nothing at load (`ggml_backend_host_buffer_from_ptr` over the mapping, `mark_mmap_used` only), so the only
   calls that touch the range are the pool/bounce fill `memcpy`s and the scheduler's fallback copy -- the exit
   census reports the bytes (`pool fills: N mapping memcpys, X GiB total`).  Attribute a large "private" figure
   with Task Manager's `Committed` / RAMMap's mapped-file page count before chasing it as a leak.

- **A prefill reads the host expert set ONE RUN AT A TIME, and the read SIZE is the ceiling, not the drive
  (r224, block 12; `wip/`→`archive/work/host-expert-split/W4-SPAN-READS.md`).**  Below the mirror the prefill
  band reads every expert that is in neither tier out of the GGUF, and it used to issue **one ~0.65 MiB
  `O_DIRECT` read per expert**: at `-lme 12` that is 198 723 reads / 128 GiB / **56.3 s of a 73 s run**, with
  the drive sitting at ~2.1 GiB/s against the 4.2 GiB/s it does on a 16 MiB read and 6.7 GB/s across four
  streams.  A read that small is latency-bound at depth 1, so the lever is the read **size**: the experts of
  one table are contiguous in the file (`src_offs + e*host_bytes`) and the missing ones form runs, so
  `stage_span_fill()` serves a run of >= 2 with ONE aligned read per 16-expert chunk (~10.9 MiB) into a pinned
  ring buffer, then one H2D per expert.  Measured prefill **+81 % at `-lme` 12, +57 % at 18, +14 % at 36**,
  generation unchanged, text byte-identical.  Three things a future reader must not undo: (a) the pinned ring
  is **grown on demand** (<= 32 entries/device) because the H2D copies are queued on the compute stream — a
  fixed 2- or 4-deep ring spent 5.0-5.6 s of a 16 s prefill WAITING on its own copies; (b) `SPAN_MIN = 2` is a
  code constant and is load-bearing (1 -> 362, 2 -> 379, 4 -> 331 t/s at `-lme` 36); (c) **zero byte
  amplification** — bridging short islands of resident experts reads 4.3x the bytes to save 1.5x on the rate,
  which loses at high pool coverage.
- **The heat prior is a MEASUREMENT INPUT, not scratch state (r224, block 12; found the hard way).**
  `<model>.moe-heatmap` is rewritten by every run (`MOE_EXPERT_HEATMAP=1`) and it decides the pool's **layout**,
  so a benchmark that lets it write is measuring a prior that is **learning the benchmark prompt**.  Measured
  at `-lme 36` on the same binary: a map drifted by the campaign's own 5 298-token prose runs gave 27.9 t/s,
  the frozen real-world map 26.2, no prior at all 23.0 — i.e. the campaign's original "+21.7 % headline" was
  **prior leakage**, and on a frozen map the honest decomposition is control 24.3 / ranking 24.6 (+1.2 %) /
  demote 25.6 (+5.3 %) / both 26.2 (+7.8 %).  The freeze mechanism is the file's own permission:
  `chmod 444` makes `heat_map_open_locked` load+seed and `heat_map_save_locked` refuse to write (it re-checks
  at every flush, so it can be frozen mid-run), and no cache directory is used instead.  **Quote the map's
  sha256 with any number that depends on it.**
- **A failed demote DECLINES; it is never published (r224, block 12; the issue-#274 rule extended).**  A demote
  copy that fails (`cudaMemcpyAsync` error, or the diagnostic `MOE_HOST_POOL_SINK_FAULT=1`) drops the victim
  exactly as the pre-r224 path did, marks the incoming fill's own request as needing a bounce slot, and never
  assigns the slot — a published victim whose bytes never arrived would serve stale bytes for the life of the
  run, which is the same defect class as a fill recorded as resident.  `scripts/gate-host-pool.sh` proves it
  on the gate's own pool config: `sink` 4 006-4 009 demoted with 0 copy-error; `sinkfail` **0 demoted** with
  4 069-4 081 `copy-error` declines and byte-identical text.
- **A writable heat map must never be truncated to a stub (r224, block 12; the bug that ate this campaign's
  map).**  `moe_cache_report`'s exit-time save walks `g_tables`, but the model teardown
  (`moe_cache_forget_all`) drops them first — so any run shorter than the 60 s heat debounce wrote a
  **header-only 32-byte sidecar**, a valid *empty* map that silently destroys a curated prior.  The tallies
  are now flushed in `forget_all`, before the tables go.  Verified: a 45 s run writes 601 376 B / 144 tables
  instead of 32 B.
- **Every ggml `INFO` counter line needs `-lv 4` (r224; a measurement invariant, not a code one).**
  `common_log_get_verbosity()` maps `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4) and the default threshold
  is 1, so at `-lv 3` — what every arm script used — `pool fills`, `prefill cache staging`,
  `coalesced prefill reads`, `per-expert storage reads` and `scheduler pageable-master fallback` are ALL
  invisible.  That is how this campaign once attributed a 2.3 GB/s ceiling to a function that never ran
  (`moe_cache_stage_table`; the reads were in `pool_prepopulate_locked` and the `moe_cache_gather_host`
  pre-pass).  Never quote a counter you have not seen printed.
- **A fusion that pre-dispatches a graph node ahead of its own launch owes an aliasing check, not just an
  alloc dep (r225, block 10, issue #276).**  The narrow-row norm fusion's sigmoid-gated form dispatches the
  gate `MUL_MAT` (`nodes[nm.pre]`) *before* the fused kernel, which then reads `x`, `w` and that gate output,
  and writes `dst`.  In **graph order** `x` is dead as soon as the `RMS_NORM` has consumed it, and the gate
  output has exactly `x`'s shape, so the allocator's best fit for it is `x`'s own block — the pre-dispatch
  then overwrites the activation the fused launch has not read.  Measured on Ling-3.0-tiny IQ4_NL
  (`-c 8192`, the 5256-token `prompts/prose-rdna-boosts.txt`): with the `x -> nm.dst` alloc dep skipped, all
  180 fused nodes take the alias and the 32-token greedy text collapses from **172 chars (`f476efaac086`,
  identical to the unfused chain) to 52**.  `ggml_backend_cuda_graph_optimize`'s `x`/`w`/`z -> nm.dst` deps
  are what make the shipped path correct, and they are **not** visible from `norm-gated.cu` — they were
  registered for a buffer-size trade, in another file, and a caller that drives `ggml_backend_graph_compute()`
  without a scheduler has none of them.  So the launch now verifies the byte ranges itself
  (`norm_rows_alias_conflict()`): a gate output or fused output overlapping `x`, `w` or each other **declines**
  the fusion (the unfused chain, byte-identical text), and the ONE allowed overlap is an exact dense alias of
  `x` with `dst` (same address, same layout, contiguous `x` — a read-then-write of the same address by the
  same thread).  Use `span = sum((ne[d]-1)*nb[d]) + type_size`, not `ggml_nbytes()`, which is the DENSE size
  and understates a padded view; `x` is a strided reshape often enough for that to matter.  Equivalence is
  proven two ways: `scripts/gate-norm-rows-alias.sh` (driver + real GGUF) and the fact that the fused launches
  are bit-identical to the chain at the driver's shape (digest `bcc36422f9a120ac` in both the fused and the
  `--no-fuse` arm).  The decision is observable with `GGML_CUDA_NORM_ROWS_DEBUG=1` (INFO, so `-lv 4`), which
  is also how the gate detects a **dropped dep**: the text stays correct either way because the guard falls
  back, so only `declined=0` distinguishes "the dep holds" from "the guard is doing the work".

## r226 -- a pad can be IMPOSSIBLE: one extra y row adds `grid.x` blocks, so the helper must refuse; and `GET_ROWS` returns F32 (issues #273 / #277)

**Measured facts behind issue #277.  Do not re-derive; do not claim a win for a pad that cannot fire.**

1. **A pad adds `grid.x` blocks wherever the caller pads an axis that is not the block multiplier.**  The four
   `get_rows` launches index `src1` with `blockIdx.x` (`i10 = blockIdx.x`), stride their row/column loop with
   `gridDim.y`/`gridDim.z` and clamp both to `UINT16_MAX`, so x cannot be padded and y is the only cheap axis --
   but one extra y row adds `grid.x` blocks, not one.  When `grid.x` is itself a multiple of 256 (a 256-multiple
   prefill ubatch; the MoE expert-cache remap's gather at `grid=(16384,1,1)`) the product stays a multiple of 256
   **and** the work-item multiple stays true, so no one-step pad on y or z can leave the stall.  The helper now
   REFUSES that pad (`pad_blocks` argument; `NO pad possible` under `GGML_CUDA_GRID_STALL_PAD_DEBUG=1`) instead
   of launching an empty block row that cannot help.  Every site that pads by exactly one block
   (`mul_mat_vec_f`, `mul_mat_vec_q_ksplit`, `mul_mat_vec_q_moe`, the FA stream-K fixup, the GDN q/k norm pair)
   passes the default `pad_blocks = 1` and is byte-for-byte unaffected.
2. **`GET_ROWS` always returns F32, so `get_rows_cuda_float`'s vec branch is reachable only from an F32 src0.**
   `ggml_get_rows` sets `type = GGML_TYPE_F32` unless the source is I32 (upstream's "TODO: implement non F32
   return"), and the `int4` copy sits inside `if constexpr (std::is_same<src0_t, dst_t>)`.  An f16/BF16 embedding
   lookup therefore takes the plain `k_get_rows_float` path, not `k_get_rows_float_vec`; do not read a
   `get_rows_float` launch in a trace as "the vec path did not fire".  Both are padded now, and each has an
   oracle shape (`F32 n=512, be1=256` -> `(1,1,256)` vec; `F32 n=1002, be1=64` -> `(1,4,64)` plain; `Q4_0 n=2048`
   and `IQ4_NL n=1024`, `be1=64` -> `(1,4,64)` for the two dequantizing paths).
3. **A missing pad site is a coverage gap, not a measured loss.**  The reporter's 27B shape launched this copy at
   768 blocks and measured **7.88 us at 768 blocks and 7.88 us at 769** -- no stall -- which is consistent with
   r217 fact 1 (a fixed ~10 us cost, hidden above ~2048 blocks) and with the r224/r225 arms (all gates and the
   whole test suite pass with `GGML_CUDA_GRID_STALL_PAD=0`).  So the r226 change is site coverage: `get_rows` was
   the one grid-stall-shaped launch family with no pad call from r46 to r225, and a shape that *does* stall there
   would have gone unnoticed.  Do not book it as a speedup.
4. **A pad site without an oracle shape cannot be verified.**  `test-backend-ops -o GET_ROWS` had **0 pad events
   in 220 cases** before the four shapes above were added, and no model run reaches them either (the 4B Q8_0,
   4B 3-device `-sm tensor` and 35B-A3B `-ncmoe 40` decode probes pad `mul_mat_vec_q_ksplit`,
   `mul_mat_vec_q_moe`, `mul_mat_vec_f` and the FA fixup, never `get_rows_*`), so the op oracle is the only
   non-vacuous arm for these sites.  `scripts/gate-grid-stall-pad.sh` is that arm: the padded launch must equal
   the **CPU** reference in both pad arms, each site must log its pad, the `MUL_MAT_ID` arm must reach the
   refusal, and the same-seed text must be identical with `GGML_CUDA_GRID_STALL_PAD=0`.

## r227 -- the host pool's disk tier exists on Windows, the census belongs to the MODEL TEARDOWN, and the Windows pinned ceiling is not the RAM formula (issue #274)

**Measured facts behind the #274 follow-up.  Do not re-derive; do not ship a Windows figure that was measured
on Linux.**

1. **A pool disk tier on Windows is `ReadFile`, not a stub, and the tight band needs it.**  From issue #130 to
   r226 `pool_open_file` returned -1 on `_WIN32`, so a Windows pool always filled by `memcpy` out of the
   read-only mapping.  The reporter measured what that costs when the expert set does not fit: ~250 MiB/s of
   disk reads, ~65 k pages-input/s and the GPU computing **~2 %** of a 4,180-token prefill, with the whole
   ~40 GiB set re-faulted per pass (`c`, default auto pool, 47 and 64 GiB).  A mapped view has no readahead to
   speak of, so the fix is a real direct tier: `CreateFileA` + positional `ReadFile` + `_aligned_malloc`, with
   `FILE_FLAG_NO_BUFFERING` for the DIO mode and `FILE_FLAG_OVERLAPPED` always.  It is the same five-helper
   abstraction the POSIX `pread`/`O_DIRECT` path uses, so the fill, the mirror prepopulate and the coalesced
   prefill span all reach it with no other change.  **Unmeasured here** (no Windows toolchain on this box): the
   value is the reporter's to confirm, and `MOE_HOST_POOL_DIO=0` is the escape hatch.
2. **The disk tier is best-effort on every platform, and a REFUSED pool allocation must say so.**  r219 already
   made a fill fall back to the mapping; a failed `cudaHostAlloc` for the pool itself was still silent, which
   is exactly the "the pool never appeared" shape the issue was opened for.  It now prints the real
   `cudaGetErrorString` and the size once, and on Windows it latches the pinned tier off for the run (the same
   policy the bounce ring's first refusal uses) instead of asking 141 more times for memory the driver will not
   give.  No cap is derived from any formula -- see 4.
3. **`moe_cache_report` cannot be the only printer on Windows.**  It is an `atexit` hook inside ggml-hip.dll, so
   on Windows it runs after `main` has returned, when llama-server's async log is already gone: the reporter's
   clean Ctrl+C printed `cleaning up before exit...`, the memory breakdown, `SLAB_STATS context-exit` and
   `model teardown: dropped 141 ...` and **not one** `MoE expert cache:` / `pool fills:` / `host pool (L2)` line.
   The census is now emitted from the **first `moe_cache_forget_all`**, under the same lock and before anything
   is un-sized, and the atexit hook is the backstop for a run that never tore down; a flag keeps it to exactly
   one report.  This also fixed a latent ordering bug: the teardown used to fold the heat delta into the
   persisted prior *before* the report, so the "this session" coverage line read `0.00 %` at exit.  The gate
   asserts the ordering (`host pool (L2)` before `model teardown: dropped`) and the single print.
4. **The Windows pinned ceiling is the adapter's SHARED (aperture) budget, and the `(RAM/2)/visible GPUs` model
   does not predict it.**  The reporter's own numbers: GPU shared usage tops out at **9.32-9.35 GiB** on a
   47 GiB box and again on the same box at 64 GiB (one R9700, one visible adapter, ~60 GiB of commit headroom
   left), with `-lme 4/8/auto` all landing on the same level and the first refusal arriving four minutes into a
   prefill for a 0.5 MiB bounce entry.  So MemAvailable is not the budget, the RAM formula is not either, and
   something already counts against it before the pool does.  What ships is measurement, not a guess: the
   pinned tier's realized bytes in the census, and on Windows the aperture segments' `CommitLimit` and
   `BytesResident` (all processes) from the same `D3DKMTQueryStatistics` call the WDDM free-memory path already
   uses.  Do not add a ceiling derived from the formula; it would cap a healthy box.
5. **Windows VRAM is charged to the process commit 1:1, and that is not the RAM the run is short of.**  The
   reporter traced ~40 GB of private commit to device allocations (`ROCm0 model buffer` +4.7 GiB, KV/compute
   +~2 GiB, the VMM slab +24.7 GiB at `SLAB_INIT`, the pinned pool +4.6-9.3 GiB) while the 70.6 GB of
   read-only GGUF views stayed out of the commit charge.  What actually binds a 47 GiB box is the **working
   set** (34-38 GB: resident mapping pages plus the pinned pool), which is why the small pinned pool (`l4`)
   was the better configuration there and why adding 16 GB of RAM moved prefill 16 -> 94 t/s with nothing in
   the delivery changing.
- **The demote's cost and the pool's fill cost are properties of the HOST LINK, not of the delivery
  (recorded 2026-11-06, before r225; `archive/work/host-expert-parity/`).**  r224's exchange rates — 9.6 µs per 679 762 B
  D2H, 0.0140 ms/MiB, ~2/3 hidden behind compute, ~3 % break-even reuse on this box's NVMe — were measured on a
  host whose link the same record's own naive-transport figure puts at **~26 GB/s** (26 µs per copy).  A host
  that negotiates more (one R9700 on PCIe 5.0 x16 = `32.0 GT/s` x 16) moves both the demote's marginal cost and
  the mirror-vs-partial gap, so a policy conclusion must not be carried across hosts without re-measuring the
  `MOE_HOST_POOL_SINK=0/1` delta.  Corollary: W1's instrument (`MOE_HOST_POOL_SINK_TAX`) was **removed at r224
  by design** — the shipped demote is the instrument now, and `SINK=0` is its control.
- **The coalesced prefill read is O_DIRECT-only (r224, block 12; recorded 2026-11-06).**  `stage_span_fill()`
  returns immediately unless `t.src_fd >= 0`, and that fd is opened only in O_DIRECT mode (`pool_open_file(t.src_path,
  g_pool_dio)`), so the `+81 %` at `-lme` 12 / `+57 %` at 18 / `+14 %` at 36 describe the
  **`MOE_HOST_POOL_DIO=1`** arm.  A buffered fill — the default whenever `--load-mode` is `mmap`, which r212
  made the policy — reads the mapping per expert and does not coalesce.  Quote the arm with the number, and
  never read a DIO-vs-buffered pair as a span A/B: the fill source changes with the mode, so that pair is two
  changes at once.
  **CORRECTED at r232 (issue #281): the span is no longer O_DIRECT-only.**  Both `span_geom` predicates dropped
  their `g_pool_dio` term and `stage_span_fill` opens the file with the mode's own flag, so a buffered fill
  coalesces too (a buffered `pread` reads the same page-cache pages as the mapping — the granularity is what
  changed, not the source).  It is not a small correction: on the 56 GiB rig at `-lme 12` (26 % coverage, the
  field shape) the buffered arm goes **513-545 -> 821-875 t/s** with 2.3-3.5x fewer storage reads, `-lme 36`
  (77 %) **985-1235 -> 1307-1446**, near-mirror **inert**, decode unchanged — and it is **+30 % over the
  `DIO=1` arm with the span**, whose decode is 2.4x worse (61 vs 26 t/s).  So the r224 percentages above are
  the *direct-tier* numbers and the buffered arm now has its own; a DIO-vs-buffered pair is still two changes
  at once, but it is no longer a coalescing pair.
- **The demote's value is BUDGET-shaped, and it is positive up to a near-mirror budget (measured 2026-11-06;
  `archive/work/host-expert-parity/RESULTS-2026-11-06.md`).**  One box, one model, `-n 64`, DIO, prose: `-lme` 36
  (76.9 % coverage) **+8 %** generation (24.0 → 26.0 t/s), `-lme` 44 (93.7 %) **+21 %** (28.0 → 33.9), mirror
  (100 %) **inert by construction** (0 demoted, 144 mirror tables).  This corrects `W1-SINK-TAX.md`'s inference
  that "the cost is constant while the benefit falls as the pool grows": the copy count is roughly constant, but
  a demoted victim at 94 % coverage is much more likely to be wanted again, and the pool's content (recency)
  replaces a prior layout.  A 94 % tier is −11 % of a mirror for 3.0 GiB less pinned RAM; 76.9 % is −32 % for
  11.1 GiB less.
- **The demote is a DECODE-CLASS notion: its headroom is capped by the coverage gap, a prefill fill ignores it,
  and a near-mirror pool disengages it (r228, block 12; found as `TODO` #281 on the r224 code).**  Three rules,
  each measured on a 59 GiB field box (56 GiB cgroup + ballast, one R9700), first as a defect then as the fix:
  1. `K = 4 %` of a table's slots (≈19 on a 500-slot table) exceeded `-lme auto`'s 12-expert gap, so the reserve
     held free the slots the complement needed: every prefill fill was refused and each ubatch re-read the
     missing experts through the bounce ring -- first-pass 50k prefill **1 837 t/s (`SINK=0`) against 867
     (`SINK=1`)**, decode flat, the gather's "neither tier" volume 10.8 k → 46 k experts.  `K` is now capped
     by the gap, and a PREFILL fill (`expert_host_src(..., prefill=true)`, `stage_span_fill`) ignores it.
  2. **A near-mirror pool disengages the exchange entirely** (coverage gap < 4 % of the set).  At 97.7 % the
     demote measured **no decode gain** (75.6 vs 76.2) and cost 43-53 % of the first prefill, because the
     decode-band churn moves the pool's content away from the complement the next prefill must read.  With the
     disengagement that cell is the `SINK=0` path to the counter (`sink near-mirror tables ... 144`, 0 demoted)
     and prefill returns to 1 907-1 938 t/s.
  3. Below that boundary the demote is a real win and the boundary is the **gap, not the budget**: `-lme 44`
     (93.7 %, gap 32) measured **+19-21 %** generation and +10-48 % prompt, `-lme 36` (76.9 %) +8-9 %, and a
     full mirror stays inert (0 demoted, 144 mirror tables).
  Gates on the fixed tree: coherence `sha=880c932888f8` identical on both `SINK` arms, `gate-host-pool.sh`
  **31/31** (the `sink`/`sinkfail` arms still prove a failed demote declines and never publishes).

- **A latched pinned tier stops the tier GROWING; it does not throw away what the run already owns (r229, block
  12; issue #278).**  `g_pinned_off` is set by the first refused `cudaHostAlloc` (on Windows a refusal also
  latches a process-wide HIP error, which is why the tier is not retried) and `bounce_acquire` has always drawn
  the right line: it reuses a drained ring ENTRY before its latch check.  `table_pool_locked` did not -- it
  checked the latch before looking for an existing pool, so ONE refusal (the reporter's box hits its adapter
  ceiling at ~9.3-9.5 GiB of shared usage during the load) made every pool the run had already built
  unreachable, and every expert copy after that point went through the pageable model mapping: a driver-staged
  copy, i.e. the exact cost the pool exists to remove, on the box where the pin budget is the scarce resource.
  Measured on the `pinlate` arm of `gate-host-pool.sh` (a latched tier forced after the pools and the arena
  exist): the lookup-first order serves **7 060 fills from pre-latch pools with 110 failed fills**, the
  guard-first order (the pre-r229 code) serves **0** from them with **1 587 failed fills** -- text identical in
  both, so only the counter can see it.  The same rule is what the census line `N expert fills were served from
  a host pool built BEFORE the refusal` reports.
- **An expert-cache arena it cannot fill must not be armed (r229, block 12; issue #278).**  `moe_cache_rearm`
  (the post-prefill, compute-buffer-drop re-arm) used to re-take the arena for every layer with a stood-down
  table.  When the pinned tier is latched off and a table has neither a pool nor a ring entry, there is no host
  source at all, so every one of those slots is armed, seeded (one doomed fill per slot, a `cudaDeviceSynchronize`
  per table) and then declined on use -- which is exactly what a STOOD-DOWN table does, at 5.5 GiB of VRAM and a
  sync per table.  The re-arm now tests the LAYER (`table_has_fill_source_locked`; layer-uniform, because a
  skipped table between armed ones is the mixed-slot layer that reads the wrong expert): a source-less layer is
  stood down wholesale and named once (`left N stood-down expert-cache tables in M layer(s) DISARMED ...`), and
  its `g_rearm_slots[layer]` is zeroed so the caller's second pass does not repeat it.  Measured on the
  reporter's config with the latch forced: the arena stays at **13 645 MiB instead of growing back to 19 225**,
  with the same declined-op behaviour and byte-identical text.
- **The eviction demote can never run beside a paired all-reduce, by construction (r229, campaign; `TODO` #276
  closed).**  `MOE_HOST_POOL_SINK`'s D2H declines when `t.n_slices != 0` (a tensor-split table has no single
  contiguous pool image, so the copy would be a strided 2-D one -- the class r113 measured as a 2x decode loss).
  A tensor split is the delivery's only layout with an intra-token all-reduce, so the two never coexist: 2 GPU
  35B-A3B `-sm tensor -ncmoe 40 -lme 12` gives `0 demoted / 14 480 geometry (structural)` with the RCCL banner in
  the log, and the same config under `-sm layer` gives **7 862** copies (4 771.4 MiB) with **no** all-reduce line.
  This is why PR #267's 0004/0006/0007 rationale (`hipMemcpyAsync` leaving later queued work unsubmitted, which
  stalls the *paired* peer past its spin timeout) has no reachable defect to fix here: do not re-open it without
  first extending the demote to a split table.  Record: `archive/work/host-expert-parity/RESULTS-2026-11-06-soar.md` §2.
- **The slab's arena eviction at a decode->prefill transition costs at most the evicted bytes, i.e. nothing you can
  measure (r229, campaign; `TODO` #278 closed).**  `moe_cache_evict_slab_range` hands the work pool the LOWEST
  arena chunks and `moe_cache_rearm` re-arms what fits afterwards, so a dropped table loses its slots and its
  experts must be re-filled at the next prefill.  It fires only when the work region outgrows the reservation it
  got at `ggml_cuda_slab_init_locked`: Flash-Next, one GPU, `-c 204800 -ub 6144` -> 3 events, **1 062 MiB**, one
  `re-armed 12 stood-down` (37-56 ms, <=0.2-1.2 % of the prompt), and three identical 4 096-token rows measure the
  post-transition row **+15.8 % over** its predecessor.  On the 3-GPU `-sm tensor` field shape the count is
  **zero**, and the cheap reproduction is **Q8_0** 35B-A3B (2 events, 380 MiB, `re-armed 3 stood-down` twice, the
  post-transition row +2.0 %) -- `Q4_K_M` cannot show it: its whole host set is 18 662 MiB (the arena clamps at
  `n_experts`, so no `MOE_EXPERT_CACHE_MIB` forces it larger) and its priming pass reserves the work region at the
  width the real prefills need.  So this is not the field prefill's residual -- that one is the near-mirror miss
  path.  Record: `archive/work/host-expert-parity/RESULTS-2026-11-06-soar.md` §1.

## (no release, r229 tree) -- the two-lane pair only pays when the SPLIT creates its partner; a declined pair must not still split

PR #279 (@slavap) re-cut `optional/two-lane-prefill` onto r229 and added the sub-ubatch split, which turns
r220's "nil on balanced devices" into a real win -- and a real regression that the same rule has to cover.

1. **A single wide prefill ubatch serializes the layer-split pipeline; splitting it and running the halves on
   two lanes is worth a quarter of the prefill.**  3 x R9700 gfx1201, the only reachable layout (`-sm layer
   -ts 1/1/1 -ot`, which turns pipeline parallelism off), `-ub 2048`, ABBA x4: MoE 35B-A3B Q8_0 pp2048 **6621
   -> 8436 t/s (+27.4 %)**, pp4096 6486 -> 8245 (+27.1 %), pp8192 with `-b 8192` (one decode call, 4 ubatches
   paired 2x2) 6240 -> 9251 (+48.3 %); dense 27B Q6_K_XL pp2048 1293 -> **1823 (+41.0 %)**, pp8192 1488 ->
   1745 (+17.3 %); tg nil.  The 2x2 says why it needs BOTH halves: **split alone** (lanes unset, `-ub 1152`,
   the same two ubatch boundaries run serially) is a **loss** (-15.3 % MoE, -3.6 % dense) and **pair alone**
   (`LLAMA_LANES_SPLIT_MIN=0`, one ubatch so nothing to pair) is nil (+1.2 % / -1.2 %).  That is r220 fact 3
   qualified, not contradicted: the pair added nothing when both partners were already wide.
2. **The pair itself is bit-exact; the split's divergence IS a ubatch-boundary change.**  lanes=2 vs lanes
   unset at the same ubatching measures **exactly** the divergence of a plain `-ub 1152` vs `-ub 2048` change on
   the same build (mean KLD `0.004088`, same-top-p `97.898 %`, identical to six digits -- the `GREEDY-PURITY.md`
   §46 benign-reordering scale), and the default path stays bit-identical to the delivery (KLD `0.000000`,
   same-top-p 100 %).  So "greedy text byte-identical between lanes" is true but it is the greedy path's luck:
   judge this artifact on the KLD equality, not on the text.
3. **A feature that declines a workload at one gate must decline it at EVERY gate.**  The split was gated on
   `prefill_lanes >= 2 && sched_lane && cb_eval == nullptr` while the pair also requires
   `!embeddings_nextn && !lf_layer_inp_dev`; a draft workload (MTP / DFlash) therefore declined the pair and was
   still split, paying a narrower ubatch for nothing -- 35B-A3B Q8_0 + DFlash2 pp2048 **3981.8 -> 3427.4 t/s
   (-13.9 %)** with diverging greedy text.  Mirroring the pair's fields in the split gate restores both (text
   sha `f61214fad249` == lanes unset).  Residual: arming the lanes on a draft context still costs ~4 % prefill /
   ~2 % decode (second compute-buffer set + one drain per ubatch) even though nothing pairs.
4. **The reachable layout costs ~1.4 GiB and is not free at the edges.**  A second compute-buffer set per armed
   context (441/430/430/65 MiB at `-ub 2048`); a **second context in one process is ~8 % slower** with the lanes
   armed (pp8192 7976 -> 7347, lanes unset flat 7513-7519 over three contexts); `llama-batched-bench -npp 2048
   -npl 1,4,8` gives **B=8 prefill -3.5 %** (8776 -> 8467) while B=1 gains +28.8 %.  Do not enable this on a
   many-slot server; it is a single-long-prompt design.

## r230 -- the PLE's residency is a *pin*, not a copy: `--lazy-mode mlock` keeps the lazy data path and mlocks the mapping (block 00)

1. **The three lazy arms differ in residency only, and the copy is the odd one out.**  `auto`/`on` read the
   rows from the mapping on demand (nothing resident); `off` reads the whole table into a private host buffer
   (resident, 27 GiB of anonymous memory, swappable, nothing reused across restarts); `mlock` reads the rows
   from the mapping on demand *and* pins that mapping in the page cache.  All three feed the same
   `host_gather` path, so same-seed text is identical (`7d898d3d3e58`, 125 chars, Flash-Next GSQ-IQ3_S 3 GPU
   `-sm tensor`) and `llama-bench` is at parity: pp2048 2570.4 / 2549.1 / 2574.8 and tg32 77.41 / 77.55 /
   76.27 for mlock / auto / off.  The `off` copy buys nothing but an allocation -- register the copy as an
   allocation, not as the residency mechanism.
2. **"Resident" must be measured with `VmLck`, not RSS.**  With `--lazy-mode mlock` on the whole table,
   `/proc/<pid>/status` reads `VmLck 28125136 kB` (26.82 GiB = the entire 27,465 MiB `per_layer_token_embd`,
   which is the whole of shard 2 of that GGUF) with `RssAnon` 1.3 GB, and `/proc/meminfo` `Mlocked` ==
   `Unevictable` == the same number.  RSS cannot tell a pinned page-cache table from a copied one.
3. **The pin costs the kernel's populate rate, which is the same order as the copy it replaces.**  `mlock`
   populates through `__mm_populate` (~4.2 GB/s here: 27.5 GiB in ~7 s cold), so `--load-mode mmap
   --lazy-mode mlock` loads in 32.5 s against `off`'s 25.7 s (a 3.9 GB/s `read_raw` + copy) and `auto`'s
   29.1 s (which reads nothing).  With the table already in page cache the pin is a page-table walk -- **0.19 s
   for 27 GiB** -- which is why a warm-cache restart (`--load-mode none --lazy-mode mlock`) is ready to serve
   in **8.26 s** (maintainer's box, 2026-11-07).
4. **A pin is per process; the page cache is not.**  The kernel releases the lock at exit (verified:
   `Mlocked`/`Unevictable` 28125136 kB -> 0 while `Cached` stays high) and the pages remain as ordinary
   reclaimable page cache.  That is what makes the mode useful: the next start re-pins cached pages instead
   of re-reading the table.
5. **`mlock` is refused, not emulated.**  Under a small `RLIMIT_MEMLOCK` (tested at `ulimit -l 2048`) the one
   `mlock` of the range fails, the loader warns and leaves the table on the evictable page-cache path (text
   unchanged).  There is no partial pin: `llama_mlock::grow_to` locks the whole range or nothing.

- **A non-mirror host tier's field prefill is dominated by the band-transition tail, and the tail is the pool
  RE-LAYOUT (measured 2026-11-07, `radix`; `archive/work/host-expert-parity/RESULTS-2026-11-07-radix.md`; r231).**  When
  the arena's slot count changes between the bands (179 -> 79 per table at `-ub 6144`, the prefill's work region
  taking the slab), `pool_rerank_from_heat_locked` re-derives the pool's "arena complement + mirrored residents"
  layout **after the last ubatch and inside the request**: 1 500-6 400 single-expert fills, all charged to the
  first token.  Measured tails (feature-attributed, not estimated -- the window between the last
  `progress = 1.00` row and `prompt eval time` is 100 % `pool_rerank_from_heat_locked`): **0.6 s on a mirror**
  (a covering pool returns early), **1.2-1.3 s with `MOE_HOST_POOL_DIO=1`**, **7.2 s at `-lme auto` buffered**,
  **13.5-18.6 s at `-lme 44` buffered** -- i.e. the tail, not the miss path, is what makes a buffered sub-mirror
  tier's first prefill look 2-3x worse than its processing rate.  Two consequences: (a) a per-fill cost is
  **0.45 ms warm (DIO) to 10.8 ms** on the same bytes, so never quote one rep -- the same arm moved 735 -> 1 191
  t/s between reps as page-cache state changed (`A0` 1 810 vs `A0b` 1 329); (b) the lever is the fill **source**
  (`MOE_HOST_POOL_DIO`, ~90 % of the tail) plus the fill **order** (r231: file order, ~30 % of it), not more
  bandwidth.  A microbenchmark of the fill shape (40 cold 0.65 MiB slices, `drop_caches`) is 0.74-1.61 ms for a
  mapping memcpy and 0.89-1.04 ms for a buffered `pread`, so the arm's 4.6-10.8 ms is the cgroup
  (`MemoryMax` + `MemorySwapMax=0`) reclaiming for every new page, not read latency.
- **On the field shape a low-coverage tier decodes at near-parity, and the demote is why (measured 2026-11-07,
  `radix`, 56 GiB rig, Flash-Next + MTP drafting; `archive/work/host-expert-parity/RESULTS-2026-11-07-radix.md`).**
  Decode (third 600-token pass, `-ub 6144`, one load per arm): **mirror 83.8 t/s; `auto` 97.9 % 75.6; `-lme 44`
  93.8 % 76.2; `-lme 36` 76.8 % 76.1** -- flat from 98 % down to 77 % coverage, 10.8 GiB of pinned RAM saved for
  ~9 %, and only the mirror is faster.  With `MOE_HOST_POOL_SINK=0` the same 77 % tier falls to **45.4 t/s**
  (-46 %): its decode turn reads **19.0 GiB** from storage against 2.2 GiB with the demote, whose census there is
  **50 793 demotions / 32.2 GiB / 67.6 % reuse** (the CLI cells measured 4.7-7.9 % reuse, because a spec-decode
  step verifies four tokens at once and re-uses exactly the experts a demote captures).  Prefill is *not* flat:
  mirror 1 895-1 939, `auto` 1 785-1 818 (`DIO`) / 1 431-1 810 (buffered), `-lme 44` 1 329-1 699 (`DIO`) / 735-1 191
  (buffered), `-lme 36` 1 037 (`DIO`) -- so **decode-parity at 77 %, not prefill-parity**, and a sub-complement
  budget should set `MOE_HOST_POOL_DIO=1` (`+43-81 %` prefill, decode neutral with the demote, 2.1x the bytes).

## (no release, r231 tree) -- the narrow-2 pin's eviction branch needs a SHORT estimate, and a `cudaMalloc` preference cannot avoid it (PR #267 0003a / `TODO` #279)

1. **The pin's VA spare is the draft's own reserve, minus the headroom, so the eviction branch is unreachable in
   the shipping geometry.**  `ggml_cuda_slab_narrow2_pin_locked()` straddles `mapped` only when the compute region
   exceeds `size - mapped`; at creation that spare is `reserve - headroom`, `reserve` is `headroom + draft-context
   estimate` (kv + compute + slack), and the MoE preflight **skips** `ggml_cuda_slab_extend_all()` at the priming
   sizing (`wip/first-prefill-latency`), so nothing has eaten it when the pin runs.  The estimate's `compute` term
   is a floor over the same measured quantity, so only a SHORT estimate reaches the branch — that is r106's
   deliberate fallback (the tensor-split drafter that used to abort in `graph_reserve`), never a routine path.
   Measured on four probes (Flash-Next + MTP 1 GPU, 35B-A3B + DFlash2 1 GPU and 3 GPU `-sm tensor`, and a `-ub`
   sweep to a 1.50 GiB region): all `unmapped VA spare`.  **Do not chase this branch as a performance item** —
   `archive/work/host-expert-parity/RESULTS-2026-11-07-narrow2.md`.
2. **When it is forced, its eviction is re-absorbed; the resident-set loss is not a loss.**  `GGML_CUDA_SLAB_RESERVE_MIB`
   sets the reserve verbatim, i.e. `spare = value - headroom`, which is the only knob that reaches the branch (and
   a >46k ubatch at a 64k context is the smallest natural shape that closes the gap).  Six interleaved arms: the
   pin evicts 314-450 MiB and the next sizing/`re-arm` pass returns it — the final arena MiB and the decode hit
   rate are identical to four decimals in both arms (`20 712.6 / 0.9471`, `21 418.6 / 0.9239`), at prefill +2.4 %
   and decode -0.8 % (noise).  The cost is one re-fill of the evicted bytes (~15-20 ms) — the same bound #278
   measured for the arena transition.
3. **A "prefer the cudaMalloc fallback" rule is a CHURN, not an avoidance** (PR #267 0003a).  Two mechanisms:
   (a) the fallback spends the free VRAM outside the slab, i.e. the very resource its own guard has to see, so it
   can only fire while the eviction is <= free - cushion (a fraction of the headroom) and can never prevent a
   *large* one — there is no free VRAM in that case, which is why the region did not fit; (b) `narrow2_floor`
   latches only on a COMMITTED pin, so the declined attempt is re-made on the next compute allocation and evicts
   anyway (measured: decline at `0.19.448` s, the same 314 MiB eviction at `0.19.465` s).  Latching the decision
   would give up the evicted bytes of cushion permanently for one ~20 ms re-fill — the wrong trade, since that
   cushion is what r206 raised to 1280 MiB/GPU to stop a hipBLASLt OOM.  Verdict: **do not implement**;
   `TODO` #279 closed 2026-11-07.
4. **`-lv` is TWO thresholds, and a census arm needs 4 while a text arm needs none.**  `common_log_get_verbosity()`
   maps `GGML_LOG_LEVEL_INFO` to `LOG_LEVEL_TRACE` (4), so every **ggml** census line (`MoE expert cache:`,
   `expert access:`, `takeover=`, `cold admission=`, the sink block's INFO) is `-lv 4`-only, while llama-cli's own
   `LLAMA_LOG_INFO` is `LOG_LEVEL_INFO` (3) and its per-slot statistics block is what pollutes the generated text.
   So `-lv 3` is the one setting that gives you the pollution WITHOUT the numbers: **census/mechanism arms take
   `-lv 4`, text-identity arms take no `-lv` at all** (and read their rates from the `[ Prompt: … | Generation: … ]`
   footer).  r231 blamed `-lv 4` for the pollution; the sharper rule is above.

## (no release, r231 tree) -- `MOE_EXPERT_CACHE_TOUCH` is a churn damper with no decode win, and the tier's cold read is why PR #267's registration is not needed

1. **`TOUCH=3` does damp churn, and the on-device outcome is a wash (measured 2026-11-07, `radix`).**  Flash-Next +
   MTP, `-c 204800 -ub 6144 -ncmoe 48`, `MOE_HOST_POOL_DIO=1`, `-lme 36` and `-lme 44` (the two budgets where the
   demote is live), 3 + 3 interleaved reps: demotions **-17.3 % / -18.3 %** (-17.2 % / -18.2 % of the bytes),
   sink uses **-27 %**, arena hit **-0.78 / -0.81 pt**, cold-served misses **+35 %** (3.27 -> 4.43 %, 3.39 ->
   4.58 %) at an unchanged fill count, and decode **+1.3 to +2.2 %** at 93.8 % coverage (inside the box's spread)
   with prefill flat.  **The default stays 2**; the replay's prediction ("touch-3 shrinks the sink's advantage by a
   third") was right about the mechanism and irrelevant to throughput.  `archive/work/host-expert-parity/RESULTS-2026-11-07-touch.md`.
2. **A pool-backed table already serves a rejected miss in place, so PR #267's 0002 registration has nothing to
   buy.**  With `admit=touch` live, `MOE_EXPERT_CACHE_COLD_UVA` (default on) served **3.27-4.58 %** of all reaches
   (15 147-21 206 misses) with **zero** declines on those arms, from the tier's own pinned-RAM alias.  A "the GPU
   cannot read the pageable master" premise was answered by the pool plus the per-table bounce ring; registering the
   expert ranges would add a second mapping for a path that is already cheap.
3. **The demote's break-even is not in doubt at these coverages.**  Sink reuse measured **13.6-15.8 %** at 77-94 %
   coverage, against W1's ~3 % break-even on NVMe (~0.2 % on slow storage) -- so the r231 decode-parity result
   holds, and `TOUCH=3`'s -18 % of copies changes the amount, not the verdict.

- **A MID-RUN pinned-tier latch perturbs the logits by ~1e-5, and that is enough to flip a near-tie token
  (measured 2026-11-07, `soar`, issue #281's gate run; `TODO` #282).**  `scripts/gate-host-pool.sh`'s `pinlate`
  arm (`MOE_HOST_POOL_PIN_FAULT=late`) serves a MIXTURE — some ops from the arena, some stood down — and
  reproduces the greedy text of neither end: 171 chars `b981e742aa13` against `ref`'s 182 / `a99ccbd97ecb`,
  deterministic across two builds and two runs, one token at ~31 of 32.  **The two ends are bit-exact**, which is
  what makes the mixture the whole story: `fail`/`late` (every table stood down, `get_ok=0`) match `ref`
  exactly, so the fallback computes the same logits and the divergence is not a wrong-expert read.  The
  delivery's own criterion on the same contrast reads **median KLD -0.000000, 99.9 % -0.000040, same-top-p
  100.000 %** over 2 x 512 tokens — two orders inside the 0.005 tolerance — and `MOE_HOST_POOL_VERIFY=1` finds
  **0** pool-read mismatches.  Ruled out by arm: the cold in-place read (`MOE_EXPERT_CACHE_COLD_UVA=0` still
  diverges), the decode remap (`GGML_MOE_DEVREMAP=0` still diverges), the staging transport
  (`MOE_HOST_POOL_STAGE_BATCH=0` still diverges), prewarm and the ranking.  **Narrowed to the demote**:
  `MOE_HOST_POOL_SINK=0` + latch reproduces `ref` exactly.  Consequence for gates: a byte-identical *text*
  assertion is the right instrument for a wholesale fault arm (bit-exact) and the wrong one for a mixed state —
  a ~1e-5 arithmetic difference flips near-ties, so that arm's evidence has to be the logit criterion.

## (no release, r232 tree) -- measuring a cold read here: eviction is FOLIO-granular, and an unverified arm will lie to you

Established while building [`utilities/storage-oracle/`](utilities/storage-oracle/) (WORKLOG `261107.007`);
the tool is the reproduction, and none of this is a delivery change.

1. **`POSIX_FADV_DONTNEED` is a folio operation, not a page operation.**  On XFS with large folios, asking for a
   4 KiB range that sits inside a larger cached folio **silently does nothing**: `fadvise64` returns 0, the folio
   stays in the page cache, and the next read is a cache hit in **~0.2 us** instead of a device read.  The first
   version of the probe reported exactly that as a p50 on a file with 39 GiB of hot page cache, and the number was
   plausible enough to believe.  Two rules follow, and the second one is the one that matters: evict with
   **folio-aligned windows** (2 MiB, escalating to 32/256 MiB), **verify with `mincore` on a bare mapping**, and
   count and *skip* (never time) a sample that could not be made cold.  A page that our own `mmap` has already
   faulted also pins its folio, so `madvise(map, MADV_DONTNEED)` has to come before the `fadvise`.  A cold-read
   benchmark on this filesystem without a `mincore` check is measuring the page cache.
2. **The corrected ordering at page granularity is `O_DIRECT` < the page-cache mechanisms, not parity.**  Verified
   cold, 4 KiB, QD1, random, on `/llm` (the Lexar): `O_DIRECT` **74 us**, buffered `pread` **150 us**, `mmap`
   fault **150 us**.  Both page-cache paths cost about **2x** `O_DIRECT` per page (a cold buffered read pulls a
   folio; a fault pays the fault machinery), which is the opposite of the "buffered is free" default assumption and
   is why `--load-mode dio` can win a *cold* load it looks like it should lose.  The unverified numbers from
   before this check were optimistic by up to 2x on those two arms.
3. **A `llama-cli` arm that failed to load reads ~10 MiB in under a second and looks like the fastest arm.**
   Measured: `--verify` reported a 0.86 s "load" that had read 11 MiB, where the arm had actually OOMed
   (`-sm layer -ngl 99` put 34.7 GiB on device 0).  Rule: an arm is only an arm if it exited 0 **and** printed a
   prompt-eval line; otherwise it is FAILED with the **first** ` E ` log line (the last one is usually the generic
   "server exited").  A load-time comparison also has to count **bytes the block device read** (`/sys/block/*/stat`
   sector field), because a mapped load defers its reading into the first forward pass.
4. **`--lazy-mode mlock` pins only `TENSOR_READ_LAZY` tables, and is INERT on a model without one** (the r230 mode,
   qualified).  Qwen3.5-4B has no such table: `mmap --lazy-mode mlock` **5.17 s** against `mmap --lazy-mode off`
   **5.18 s** cold, i.e. the flag did nothing, and only the ordinary page cache gave the 1.00 s restart.  Gemma4-E4B
   has one and logs `pinned 1 table(s), 2856 MiB`.  So residency must be confirmed from the loader's
   `pinned N table(s), X MiB` line (or `VmLck`, r230 §2), never inferred from the flag.  `--load-mode none` still
   creates a mapping for exactly those tables (the loader keeps one for them), which is why `none + mlock` is a
   valid composed arm -- and the form r230's fastest restart used.
5. **`--load-mode dio` does NOT read the same number of bytes as `mmap`/`none`.**  r215 §1 holds for the mapped
   arms, but O_DIRECT's alignment handling re-reads the unaligned prefix of a chunk: 4 921 MiB against 4 275 MiB
   (+15 %) for the same model load here.  Any A/B that charges `dio` only its throughput advantage is understating
   its cost -- read the device's byte counter per arm.


6. **The Lexar's 16 ms idle wake is APST, and it is removable -- but only by re-probing the controller.**  `nvme
   id-ctrl` shows APST enabled at `ITPT 100 ms -> PS3`; with APST off the same file measures **88 us** after a
   5 ms gap and **50 us** after 50 ms, against **2 533 us / 4 743 us** at the default, and the 250 ms step falls
   **16 313 -> 4 716 us**.  Nothing else moves: `O_DIRECT` bulk is 2.46 GB/s either way and every page-cache arm is
   unchanged, so this is a power-policy effect and not a throughput one.  Three shortcuts do NOT work, each tried:
   `nvme set-feature -f 0x0c -v 0` returns `EPERM` (the kernel driver owns the APST feature), `nvme reset` does not
   re-read `nvme_core.default_ps_max_latency_us`, and a user-space admin passthrough of Set Features is refused
   the same way.  What works is writing the parameter and **unbind/bind that PCI function** -- the parameter is
   read at controller *creation* -- which also scopes the change to one drive (the SN850X keeps its APST and its
   0.6 ms wake).  A ~5 ms firmware level survives any setting and is reached after roughly a second of idleness, so
   "APST off" means 50-90 us in use and ~5 ms cold; never "instant".
7. **Watts are NOT readable in band, so never quote one as a measurement.**  `id-ctrl`'s `mp` is a per-state
   *ceiling* -- this drive declares 10 W while TechPowerUp measured 7.0 W reading, 8.5 W writing and 1.454 W idle
   with ASPM disabled -- `IDLP`/`ACTP` are unpopulated on both drives here, NVMe SMART has no power field, the
   `nvme` hwmon has temperature channels only, and RAPL/`powercap` is the CPU socket (`package-0`, `core`) while
   the drive hangs off the 12 V rail.  Per-state watts need a shunt (Quarch/Monsoon, or an M.2 riser with an
   INA228) or a wall meter and a delta method; and any idle figure quoted without its APST/ASPM state is useless.

## (r233) -- the Windows pinned ceiling is the loaded HIP runtime, not Windows: name the module before believing a number

Established from issue #278's reporter (Windows 11, one R9700, 64 GiB), same r229 binary, same command, only the
HIP runtime swapped.  This is the missing half of the r223 fact 4 above and it changes what a Windows pinning
report means.

- **The ~9.3-9.4 GiB pinned stop belongs to the driver-shipped `amdhip64_7.dll` in `C:\Windows\System32`.**  That
  box had been running with the Adrenalin driver's HIP runtime (file version 10.0.3661.0) while the ROCm tree
  beside the binary supplied only rocBLAS/hipBLAS/hipBLASLt.  Putting the official TheRock ROCm 10.1.0 Windows
  runtime (`amdhip64_7.dll` 10.0.3684.0, `amd_comgr.dll`, `hiprtc0716.dll`, `rocm_kpack.dll`) next to the
  executable -- confirmed loaded from the binary's directory, not `System32` -- removed **both** symptoms:
  the #278 hang (2/2 runs before, 0/2 after; prefill 16.1 and 40.2 t/s, decode 9.4 and 11.8 t/s, MTP 272/378)
  and the ceiling (pinned tier **16 004 / 14 913 MiB** on the #278 arm, **40 078 MiB** on the disk-tier `c` arm
  with **141/141** tensors, a full mirror `h=1.0000`, 0 failed fills, 5k prefill **370.7 t/s** against 39.8).
  Text was byte-identical to the reference on the good runs and differed on the latched driver-runtime run.
- **What this does NOT say.**  The two spinning threads were sampled inside the HIP runtime's own yield-wait
  loop (`SwitchToThread` under `amdhip64_7.dll`, one started by the runtime, one under `ggml-hip.dll`'s
  `hipGetDevice`); the hang is a runtime defect on that box, not something the block set can fix.  It also does
  not license a WDDM/aperture formula: the driver runtime's ~9.3 GiB stop is still not derivable from RAM, the
  commit limit, or the adapter count.  Do not cap a healthy box from any of them.
- **What the delivery does about it (r233, block 12).**  `wddm-mem.cu` gains
  `ggml_hip_runtime_module_info()`: it resolves the loaded `amdhip64_7.dll` by name, asks for its file path,
  compares it case-insensitively with `GetSystemDirectoryA()`, and reads `hipRuntimeGetVersion()`; the MoE cache
  logs one INFO line (`HIP runtime 7.16.26385 from <path>`) and, when the module came from the system directory,
  one WARN that names the measured consequence and the remedy (put the TheRock DLLs beside the executable).  The
  new kill switch `MOE_HOST_POOL_PIN=0` closes the pinned tier from the start on any platform, so a box stuck on
  the driver runtime can still run the host-expert path (pageable mapping; correct, slower) without touching it.
  Both are inert on a box that does not hit either condition; `gate-host-pool.sh`'s `pin0` arm proves the switch
  builds no pool, says so in the census, and leaves the greedy text byte-identical to `ref`.


## r234 — the prefill staging decline, and the transport ranking is a band, not a stall (2026-11-07)

Facts from `TODO` #271/#272 (records: `archive/work/host-expert-transport/README.md` and `RESOLUTION.md`,
`WORKLOG.md` 261107.011).

- **A staging decline is a TRANSIENT stand-down, not a sizing bug.**  On a pinned master
  (`--host-experts pinned`, 47 962.5 MiB `ROCm_Host`) 50 of 144 tables decline
  (`prefill cache staging: ok=819 unusable=450`, 9 of 9 ubatches) and every one of them declines on
  `arena == nullptr && slots == 0`; the other three predicates (`host_dev == nullptr`, `!cold_safe`) never
  fire on an unfused table, and `host_pinned` implies `host_dev != nullptr` by construction.  The trigger is
  the prefill's own compute layout taking the slab chunks those arenas lived in
  (`moe_cache_evict_slab_range`, 7 536 MiB + 150 MiB here, mostly `--cache-ram`'s 32 context checkpoints);
  `moe_cache_rearm()` rebuilds them at the compute-buffer drop, before any decode, so the state is
  **prefill-only and designed**.  A 0-slot table is not a defect — do not "fix" it by relaxing the
  predicate.
- **The decline is FREE: the fallback is the same bytes on the same stream.**  The scheduler's own
  whole-table upload is one bulk async copy on the **copy stream** the staging ring uses, so the archived
  "per-tensor copies serialised on the compute stream" reading was a trace of a different arm.  Measured
  (1 GPU, pinned, `-ub 6144`, ABBA): **1826 t/s declined against 1824 t/s staged** (±0.1 %), both engines
  bulk.  Staging those tables through the pinned tier's one-launch kernel instead measures **-10 %**
  (1572/1595 against 1757/1765): the r114 kernel is a *worse* engine than a bulk DMA for a whole-table
  copy, on the pinned tier as well as the pool tier (`LEAD-pool-staging-gap.md` §2's CU monopolisation).
  So: a fallback is not a defect until it is measured against the alternative — and here both alternatives
  were equal or worse.
- **The transport band's upper edge does not exist (refutes `LEAD-pool-staging-gap.md` §1's "loses by
  8192").**  1 GPU, mirror pool, Flash-Next, ABBA × 4 runs per arm: the gather wins at `-ub 2048`
  (1066 vs 891), the per-expert copy-engine arm wins from 5120 up (5120 **+18 %**, 6144 +17 %, **8192
  +10 % — 4 of 4 runs**, 16384 a tie), and the gather is the only arm that is stable (±0.1 %).
- **A "stall" that reproduces as a two-level oscillation is the box, not the pipeline.**  Identical
  consecutive runs of ONE unchanged arm measured 1873 / 2177 / 1865 / 2180 t/s (period 2, amplitude 16 %;
  17-32 % on the pool copy-engine arm; 0 % on the gather).  Cause unidentified: not paging (`pswpin`
  +≤200 pages/run), not `MemAvailable`, not the arena/slot geometry (identical in every run), not THP
  accounting (`AnonHugePages` 0 GiB after every teardown).  Consequences: the archived `kernel/wall 0.80 at
  8192` was the low phase; `GGML_SCHED_STAGE_SLOTS` 8 vs 16 (the compile-time max) changes nothing, so ring
  depth was the wrong suspect.  **Protocol:** an interleaved `A B A B` reads as a ~6 % win for the even
  positions (measured: base 1757 at position 1 against base 1863 at position 2), so a transport A/B needs
  `A B B A` and must report the spread — never a single run, never a block-ordered set of reps.
- **The tier changes the ranking, so one default cannot serve both.**  On a pinned master at 1 GPU,
  `GGML_MOE_CACHE_STAGE=0` (no cache staging, one bulk H2D per table) measures **2023** against the shipped
  staging kernel's **1793** (+12.8 %) and forcing the gather measures 1493 (-16 %).  That is a *tuning* note
  today, not a default: it is one shape on one box (x8 link), it trades the arena's bytes for PCIe traffic,
  and the arm that would replace it (copy-engine staging) is bimodal.  A rule needs the 2048-5120 crossover
  measured first.
- **The prefill staging copy pays or loses depending on the TIER and the RESIDENCY, not the switch
  (measured 2026-11-07, Flash-Next GSQ-RCO IQ3_S, 52 512-token prompt, balanced ABBA, 4 runs/arm).**
  `GGML_MOE_CACHE_STAGE=0` (the plain whole-table upload) is 1 GPU / `-ub 6144`:
  * **Pageable master** (`--host-experts pool`): **5-10x slower** (103-281 t/s against 1052-1556 with
    `GGML_SCHED_DEVGATHER=0`).  The staging copy is the only fill that reads the arena (D2D); without it
    every byte goes through `moe_cache_upload_pageable`, i.e. a host `memcpy` into a pinned bounce plus a
    pinned H2D per expert.  **Never disable it on that tier** — this is the r83 case (+32-54 %) and the
    reason the default exists.
  * **Pinned master at 99.1 % residency, 2 devices `-sm layer`**: staging is **+4.0 %** (1961 against
    1885) and is the stable arm; with the arena covering the host set the staged copy moves ~0 bytes over
    PCIe (H2D 0.0 %) and the copy stream hides the rest.
  * **Pinned master at 43.4 % residency, ONE device**: staging is **-12.8 %** (1793 against 2023) — the
    one-launch fill kernel monopolises the CUs and there are too few arena bytes for the saving to pay.
  So "is the switch worth it" is a tier-and-residency question, and the 1-GPU pinned answer is an *engine*
  problem (a copy-engine or hybrid fill that keeps the arena source), not a reason to drop the feature.
  A `-ncmoe 24` arm is NOT a way to reach high residency on a 32 GiB card: the resident layers plus the
  auto arena over-commit and the context fails to initialise.
- **A run-to-run transport "phase" on this box is the SDMA copy-engine path, and its cost scales with the
  NUMBER of DMA operations (measured 2026-11-07, evening; record `archive/work/host-expert-transport/README.md`
  §6c).**  With identical work (arena 20 796.2 MiB / 43.4 % residency, staged bytes 113 363.1 + 148 086.9 MiB,
  the same evictions, the same 21 055 `copyBuffer` dispatches) consecutive runs alternate between a fast and a
  slow level, and the amplitude is a property of the transport's DMA op count, not of its byte count:

  | transport | DMA ops per table | low / high | amplitude |
  |---|---|---|---|
  | device gather (kernel read of the pool alias) | 0 | 1791 / 1793 / 1790 / 1788 | **±0.1 %** |
  | staging kernel (1 launch per table) | 1 launch | 1761 / 1876 | 6.5 % |
  | whole-table bulk upload (`GGML_MOE_CACHE_STAGE=0`) | 1 DMA | 1866 / 2180 | 17 % |
  | per-expert copy engine | 512 DMA | 1624 / 2147 | 32 % |

  **Ruled out, each by measurement**: PLE page cache (the oscillation is identical with `--lazy-mode auto`,
  `off` and `mlock`; on this box `auto` is also the *fastest* of the three because a 47 962.5 MiB host-expert
  tier leaves no room for a 26.8 GiB PLE pin), memory pressure (`MemAvailable` identical within a series),
  the GPU clock (sclk max ~3220 MHz in both), the ring depth (`GGML_SCHED_STAGE_SLOTS` 16 vs 8: +1 %), and the
  GPU slot (same on gpu0 x8 and gpu2).  **What it is**: a `rocprofv3` trace shows the low phase at 67 % GPU-busy
  against 85 %, with the copy engine *less* busy (0.16 s vs 0.36 s) in a longer wall — the copies stall instead of
  transferring and the GPU idles in many < 5 ms gaps — and `HSA_ENABLE_SDMA=0` removes the oscillation entirely
  (1635/1557/1541/1566 t/s: stable, at the low level).  So it is a ROCm/SDMA dispatch-latency effect, not the
  fill code, and it cannot be fixed here.
- **Consequence for transport decisions.**  (1) A transport A/B on this box must report both phases or include a
  gather control arm: the phase reverses the per-expert-copy-engine vs one-launch-kernel ranking
  (low: kernel +8.4 %; high: CE +14.4 %), so a mean over an unbalanced sample is a phase measurement.
  (2) A *default* should prefer the low-DMA-count engine — which is what ships (staging kernel on the pinned
  tier, gather on the pool tier).  (3) The whole-table bulk upload is the only arm that beat the shipped
  kernel in **both** phases (+6 % low, +16 % high, 1 GPU / 43 % residency), but it sends the arena-resident
  bytes over PCIe as well, and on a 2-device 99 %-residency shape the staged kernel wins by 1-7 % — so it is
  not a universal default either.
- **`MOE_HOST_POOL_STAGE_BATCH` stays, and its two arms are not interchangeable.**  The engine merge
  (per-expert copy engine for both tiers, kernel deleted) was built and measured: it wins the high phase and
  loses the low one, and the pinned tier's copy-engine fill would be a coin flip.  The switch therefore still
  describes a real choice: `1` = one launch (the low-variance default), `0` = one copy per expert (the A/B arm
  that wins the high phase and on 1 GPU at `-ub 2048`).

## (r235) — the scheduler's async host-input copy stops at the pinned ring's snapshot limit: above 1 MiB a non-weight host split input is copied synchronously

**What is true.**  `ggml_backend_sched_copy_input`'s `host_src && dev_dst` branch (block 09) hands the split
backend the **caller's live host pointer** via `set_tensor_async`.  The CUDA backend snapshots such an upload
through its pinned ring **only up to `GGML_BACKEND_ASYNC_SNAPSHOT_MAX` (1 MiB, formerly the CUDA-local
`GGML_CUDA_ASYNC_INPUT_MAX`); above it the `cudaMemcpyAsync` reads the source when the stream reaches the copy,
not when it is issued.  The buffer that bites is a CPU split's output written **in place** — the token embeddings
when `token_embd` is on the CPU (`CPU_Mapped`; `model.input_embed` / `node_1`, **41,943,040 B** per 2048-token
ubatch, **10,485,760 B** at `-ub 512`), rewritten by the next ubatch's CPU split.  Since r235 the scheduler gates
the async branch on the same constant and copies a non-weight host input above it **synchronously** — what
upstream ggml does for every host input — leaving the ≤1 MiB ring snapshot (r90) and a host **weight** (never
rewritten in place) on the async path.

**How it was measured here (r234 tree; the PR's own config).**  Instrumented `ggml_backend_sched_copy_input` on
`Qwen3.8-27B-UD-Q4_K_XL`, `-ngl 99`: `token_embd` in `CPU_Mapped` (682.03 MiB), the CPU split's output in a
`ROCm_Host` compute buffer of **40.96 MiB**, and `node_1` / `model.input_embed` uploaded async at exactly
10,485,760 B / 41,943,040 B per ubatch.  **No shipped CLI config corrupts prefill**: logits from the default arm
and from an `AMD_SERIALIZE_COPY=2` (wait-after-every-copy) arm are **byte-identical** (0 differing bytes of
1,013,178,324 at `-ub 512`; self-A/A also 0), and r234 vs r235 is 0.  So this is a **latent** hole here, not a live
defect: it needs something that queues large device-to-host work between prefill ubatches (the reporter's
layer-input exports — the delivery ships `llama_model_set_embeddings_layer_inp_preset` — and, in shipped opt-in
features, the KV streaming and the host-expert sink do that class of work).  It is the **third instance** of one
bug class this delivery has already fixed twice: r35 (issue #87, graph-input views) and r90 (block 10, the ≤1 MiB
ring snapshot, `archive/work/async-input-graph-ordering/`).

**Numbers of record (this box, 27B, `pp2048`, same binary, kill-switch A/B, 4 alternating starts/arm).**
1 GPU `-ub 2048` **−0.17 %**, 1 GPU `-ub 512` **−0.30 %**, 3 GPU `-sm layer` `-ub 2048` **+0.08 %** — all inside
the arm spread (old arm sd 12.7 t/s on a 1397 t/s mean).  The reporter's −0.4/−0.7 % (r220 fork, drift 1.2 %) is
compatible but not reproduced.  Prefill logits, greedy text unchanged; `gate-prefill-logits.sh`, the coherence
gate, `gate-host-pool.sh` and the per-block build ladder all green (r235).

**The rule.**  A host-resident tensor handed to `set_tensor_async` must be stable for the whole life of the copy
**or** snapshotted by the backend; a caller that cannot promise that (the scheduler, for a buffer another split
rewrites) must use the synchronous copy.  `GGML_SCHED_ASYNC_LARGE_HOST_INPUT=1` restores the asynchronous copy
(A/B and bisection only); the CUDA ring limit and the scheduler's gate are one `#define` so they cannot drift.
**Not changed here:** the async-input ring's lazy done event is per host thread (`origins` keyed by
`std::thread::id`), so a target and a draft context on one thread share one pending done event.  It is
conservative today (every upload flushes `done_pending`, and `has_done` then waits on the *latest* graph on that
origin), which is why it was left alone; it is noted so a future third context on one thread is checked, not
assumed.

## (no release, after r235) -- a fill chain is issued on ONE stream, and the link's first completion is what the pipeline waits on

Measured 2026-11-07 on a **PCIe5 x16** host, `wip/expert-transfer-engine/` (`h2d-probe.cpp` + the pipeline arms;
`WORKLOG.md` 261107.014).  No delivery change.

- **A fill chain that feeds one consumer must be issued on ONE stream, so completion order matches NEED order.**
  Extra copy streams do not add link bandwidth: eight 343 MiB H2D copies take **49.95 / 49.99 / 49.98 ms at 1 / 2 / 4
  streams**, while the **first** copy lands at **6.26 / 10.94 / 10.93 ms**.  A link-bound fill therefore *loses* the
  earliest-completion latency the 1-deep staging pipeline depends on (measured end-to-end: **-4.1 %** prefill on
  Flash-Next `-ncmoe 46`, bulk, 48.2 % residency, in every arm including the one-copy-per-table control), while a
  **dispatch-bound** fill gains strongly (**+33 %** fill bandwidth: 290 x 0.65 MiB copies 41.1 -> 54.8 GB/s;
  **+32 %** end-to-end on a 35B pool mirror at 43 % arena).  Concurrency is legitimate only across independent
  resources: copy engine vs compute engine, or one stream per device over its own link (`-sm tensor`'s
  `stage_input`).  `copy_stream()` is `stream(device, 1)` and that is load-bearing, not incidental.
- **`GGML_CUDA_GRAPH_OPT=1`'s branch streams collide with the staging copy stream's index.**  The fork/join mapping
  is `stream_mapping[n] = branch_idx + 1`, i.e. branch 0 lands on `stream(device, 1)` -- the same stream
  `copy_stream()` returns -- so enabling it queues the staging fills behind branch compute instead of beside it.
  Dormant (the option is default-off); recorded so it is fixed before anyone enables it (`TODO` #283).
- **A synchronous copy is not a benchmark for the async path, and one host allocation kind is a trap.**
  `hipMemcpy` out of a `hipHostMalloc` buffer measures **40.7 GB/s** against the **55.8 GB/s** the same buffer does
  asynchronously (registered mmap / write-combined / pageable are full speed both ways).  `stage_h2d_gbps()`
  calibrates the staging width gate with a *synchronous* copy; it happens to use `malloc`, i.e. the fast path, so
  the shipped gate is honest -- pointing that probe at a pinned buffer would move the gate by ~27 % with no warning.
- **A host-resident table costs three VRAM copies, and they are three different BUDGET POOLS.**  Arena (duplicates
  of the resident fraction; sized once from `free - reserve - aux`), the ring sub-region
  (`GGML_CUDA_SLAB_RING_MIB`, a *work* sub-region), and `input_cpy` (the consumer's real `src0`, inside the narrow
  work floor).  Measured: cutting the ring 6.00 -> 2.50 GiB leaves the arena **unchanged** (22269.8 MiB, 48.2 %)
  and costs **-11 %** prefill while returning **+2.7 %** decode -- so transport VRAM is not convertible into
  residency by shrinking it; that needs the arena *sizing policy*, and the ring's VRAM is buying prefill.
- **Representing every expert is almost free; backing them is not.**  70 656 experts (Flash-Next: 138 x 512) cost
  **276 KiB** as int32 residency ids and **~1.5 MiB** with per-expert {VA, host} offsets and the ids remap.  A
  unified store is therefore a VA reservation with VRAM backing for a resident subset (block 15's pattern at expert
  granularity) -- and it must stay **yieldable**: a raw `cudaMalloc` outside the slab is the pre-slab crash mode
  (issue #136), because a wide prefill's transient work needs that VRAM (`TODO` #285).
- **`--load-mode none` silently disables the `--host-experts pool`** (`a pool needs a mapped load`), so a
  `runme` built on it runs the bulk/pageable path its own comments do not describe.  Read
  `report_host_expert_placement` before attributing anything to the pool.
- **A re-base that renames a kernel's field owes a re-derivation of the VALUE, not a carry-over (r236, re-base
  R11).**  Two upstream renames in the `mmq` launch path changed the *meaning* of the value while keeping the
  arithmetic around it identical, and both were carried across with the old meaning:
  `mmq_args.ncols_max` (a column **count** the launcher divided by `J` itself) became `ntj` (the **J-tile
  count**, consumed directly as `grid.x`), so `ggml_cuda_mul_mat_q_pair` launched `grid.x = ncols` instead of
  `ncols/J` -- 512 instead of 4 on the 4B FFN -- and enumerated tiles past `src1_q8_1`/`dst`, which is a GPU
  **page fault** at a large context and nothing at all at a small one (the grid scales with the context, so
  `-c 2048` is in bounds); and `mul_mat_q_process_tile`'s `tile_y_max_j` (inclusive max index) became
  `tile_y_j_sup` (exclusive count), so the routed-compact kernel's `col_diff - jt*J - 1` dropped the **last
  column of every tile** (garbage text, no fault, no warning).  The discriminator for the first is a launch
  dump compared against the single-node path (`ntj` must equal `ceil(ncols/J)`); for the second it is the
  fused-vs-unfused text at a *prefill* width, which is the only width that kernel runs at.
- **The routed (`ids`) convention in `mmq_args` puts the expert in `tile_spec`, not in `grid.z` (r236).**  For
  `MUL_MAT_ID`, `nchannels_y` is **1**, `nchannels_x` carries `n_experts`, and `ntj = ceil(ne_get_rows/J) +
  n_experts`; the kernel indexes `tile_spec[jt]` whenever `ids_dst` is set, so a routed launch that passes
  `tile_spec = nullptr` is a null-deref on every path that is not the delivery's compact dispatch.  Every
  gate that asks "how many experts" on the ids path must read `nchannels_x` -- the delivery's compact-MoE gate
  read `nchannels_y` and was therefore **unreachable** from R09 until R11, which is how the off-by-one above
  stayed invisible for three rungs.
- **The fusion dispatcher exists once (`ggml_cuda_try_fuse.cu`), and a fallback copy of upstream's dispatcher
  is only safe if it is upstream's text verbatim (r236, closes `TODO` #290).**  The interim two-copy merge
  (`ggml_cuda_try_fuse_legacy()`) was ~490 lines larger than upstream's `ggml_cuda_try_fuse` because it had
  been resolved *merged* with stale duplicates of delivery arms (`ssm_conv_in`,
  `op_ssm_conv_l2_gatebeta`, `op_rms_norm_q8_1`, `add_softplus_mul`, `shexp_down_gate`, `l2_norm_pair`,
  `gate_beta_verify`); the inline SSM conv-input fold then ran on qwen35moe decode where the delivery's
  canonical arm declines, and it is not bit-identical to the unfused chain (a state-carrying path).  All
  upstream arms are now reachable from the one dispatcher (delivery arms first, upstream's DSV4 HC post folded
  in at upstream's position); the duplicates and the superseded `ggml_cuda_match_moe_weighted_reduction`
  matcher are deleted.
- **Fusion decisions on this tree are not deterministic run to run; the text is the oracle (r236).**  Identical
  runs of one binary differ by a few fusions (4B: `CONCAT skip=1` 334/336, `CONT skip=2` 40/37) because several
  arms are address-selected (`ggml_cuda_check_fusion_memory_ranges`).  A per-node fusion trace is therefore not
  a valid A/B oracle for a merge -- compare per-arm *multisets* within that variance, or better, the text.
- **A short decode run cannot compare all-reduce arms (r236).**  `Generation: t/s` over `-n 16` is dominated by
  one-off setup (block 01's hybrid pipeline stages 32 MB per GPU), which made hybrid read 58.8 t/s against
  rccl's 82.7 on the 3-GPU `-sm tensor` 35B-A3B; at `-n 64` the same arms read 100.2/105.6 (hybrid) vs
  90.1/90.0 (rccl).  Judge an AR arm at >= 64 decode steps, or from a `llama-bench` tg run.
- **The device pool is a STACK, so every pool buffer that outlives a statement must be released in reverse
  allocation order -- through the context's `pool_stack`, never in place (r236, re-base R12).**
  `ggml_cuda_pool_vmm::free()` asserts that the block being freed is the top of the pool
  (`ggml-cuda.cu:1348`), and the non-VMM pool silently corrupts its accounting on an out-of-order free.  The
  per-graph MMB/BF16 caches broke that four ways -- the yield freed the activation cache in vector order and
  then the slots in ascending index order, `mmb_cache_insert()` evicted the *oldest* entry while newer ones
  were live, and `mmb_slot_reserve()`/`bf16_src1_cache_put()` freed a grown/replaced block in place.  It was
  invisible on the delivery's old NO_VMM builds and fatal on ours (the slab needs VMM): **every qwen3.8 run
  aborted on the first compute step at `-ub 512`**, with three 17 MiB blocks live and the oldest freed first;
  `-ub 128` stayed under the two-block threshold, which is why every gate and every other model passed.  The
  fix is one per-context stack: register a block at allocation, `pool_stack_kill()` only MARKS it dead, and it
  is really freed when it reaches the top; `graph_compute`'s per-graph yield drains what is left, top first.
  Any *new* cache or scratch that keeps a `ggml_cuda_pool_alloc` across statements must join that stack --
  and a VMM build is the only build where the mistake is loud.
- **A feature input that was never registered with `res->add_input()` looks exactly like a bug in whatever
  consumes it, and every env bisect you try will "move" it (r236, re-base R13).**  `src/models/dflash.cpp`
  built the DFlash drafter's target-feature tensor (`build_inp_target_features()`) in two of its four
  injection paths and never called `res->add_input(std::move(inp))`, so `llm_graph_input_embd::set_input()`
  never ran: the host path uploaded nothing (all-zero features) and the device path consumed an
  **uninitialized** row-index tensor.  Both still *ran*: the target text stayed coherent, the draft produced
  tokens, and draft acceptance sat at 0.000-0.019 -- the signature of a silently dead drafter.  Because the
  uninitialized content changed with whatever else the allocator/slab left in that buffer, the symptom moved
  with unrelated switches (the delivery's prompt-ingest tail and `GGML_CUDA_SLAB=0` each "fixed" it), which is
  what made it look like a slab bug.  **Rule: for any graph input that reads as garbage, check that the
  building code registered it (`res->add_input`) before bisecting env switches** -- grep the `set_input`-ed
  tensors of that graph against its `add_input` calls.  Detection for the draft case: acceptance ~0 while the
  generated text is correct (`-lv 4`, the `draft acceptance =` line).
- **The device pool's LIFO also forbids a plain *local* allocation that is created before a `pool_stack`
  push in the same scope (r236, re-base R13).**  The `pool_stack` rule above fixes buffers that outlive a
  statement, but the mirror case is just as fatal: `ggml_cuda_hc_combine_norm_try_fuse()` took its base-out
  copy as a stack `ggml_cuda_pool_alloc<char>` and then called `set_bf16*()`, whose `bf16_src1_cache_put()`
  pushes a block of its own -- the local's destructor then freed below a live block and the VMM pool asserted
  (`ggml-cuda.cu:1360`).  It fired on **every `-sm layer` run of a qwen4exp model** (plain decode, no draft),
  i.e. a configuration the campaign had never run, and it was pre-existing at R12.  Rule: in a fusion helper,
  a pool block that must live until the kernel launch and is allocated before a cache `put()` joins the
  stack too; only allocations made *after* the last push may be plain locals.
- **Upstream's layer-input device buffer is allocated on `model.dev_output()`, and under `-sm tensor` that is
  the Meta device -- `ggml_get_rows` over a Meta buffer GPU-faults (r236, re-base R13).**  The DFlash
  drafter's target-feature gather (`ggml_get_rows(g.layer_inp_src, rows)`) faults in `k_get_rows_float_vec`
  on the first injection (3x gfx1201, `-sm tensor`, reproducible).  `layer_inp_dev_alloc()` therefore keeps
  the host-copies path when the output device is Meta -- the same refusal the delivery's F1 encoded
  ("layer inputs are in a Meta buffer").  A Meta buffer has `set_tensor_n`/`get_tensor_n`/`cpy_tensor_n` all
  NULL, so every batched state/device copy through one lands on the per-tensor pageable path; that is why the
  delivery's pinned `state_stage` path exists for those shapes (`gate-state-pipe.sh`, `-sm tensor`).
- **A prompt-cache checkpoint is shared, not owned: recycle its state buffers only when it is uniquely
  referenced (r236, re-base R13).**  Upstream's `server_prompt::checkpoints` is `shared_ptr`, shared with a
  `server_prompt_cache_state` entry; the delivery's r155 buffer-reuse fold *empties* a dropped checkpoint's
  `data_tgt`/`data_dft` into a spare pool, so recycling a checkpoint the cache still references leaves the
  cached copy state-less.  A later restore then matches it, `load_tgt()` is a no-op for it, the server
  believes the slot holds its position, and the KV remove for the rest of the prefix fails:
  `memory_seq_rm(0, 5242, -1)` -> `GGML_ABORT` in `common_memory::seq_rm`.  `ckpt_recycle()` now takes the
  `shared_ptr` and returns early unless `use_count() == 1`; the "keep the checkpoint a task just restored"
  fold moves the pointer to the back of the list instead of rewriting `id_task`/`pos` on the shared object.
  `gate-state-pipe.sh`'s fresh-reference request is what caught it (R12 runs the same gate clean).
- **A re-base's coherence oracles are per-build, not carried numbers (r236, re-base R13).**  The campaign's
  recorded hashes were inherited from the pre-re-base tree: dense 4B `1472ebdadd46`, 30B-A3B `013d09f165a5`
  and 3-GPU 35B-A3B `90decf558d96` do **not** reproduce on the a691b2345 base (the R12 tree gives
  `1de99b7e6ee6`, `4b4d56191368`, `86b7f9b5aa80` for the same commands, and R13 gives the same three);
  only Flash-Next GSQ `7a8734a9f1a6` reproduces.  A re-base moves numerics on purpose, so the oracle table
  must be re-recorded from a known-good build of the *current* base and the comparison made same-session
  (R12 vs R13), never against a number from the previous fork point.
- **A re-based call into an extended parameter list must be named, and a re-derived helper carries its node
  type (r236, re-base R16).**  Two defects of the same shape (compiles, prints nothing, fails at first use):
  (a) upstream inserted `is_indexer` **before** `v_enabled` in `llama_kv_cache`'s ctor tail, and the delivery's
  positional call in `llama-memory-hybrid.cpp` carried the old comment-labels, so `is_indexer = true` and
  `v_enabled = kv_pool` (false) — the attention cache allocated **no V tensor** and `llama_kv_cache::cpy_v`
  null-dereferenced at `+0x20` on the first graph build, i.e. every hybrid/GDN model segfaulted at load
  (`Qwen3.5-4B-Q8_0` died in the *fit probe*'s `sched_reserve`).  Four adjacent bare `bool`s: nothing warns.
  Name the arguments, and audit **every** call site of the symbol (the other sites were correct: `hybrid-idx`
  named it, `model.cpp` stops at `share`, `dsa`/`dsa-iswa`/`msa`/`dsv4` are upstream's).
  (b) `ggml_rope_get_truncate()` asserts `a->op == GGML_OP_ROPE || ROPE_BACK` since `#30206`; the delivery's
  fused `GGML_OP_INDEXER_SCORE` kernel (`indexer-score.cu`, added at R12) had been re-derived to call it on its
  own node, so **every sparse QSA decode arm** (`LLAMA_QSA_DENSE_DECODE_UNTIL=0`, and every attendance-pool arm)
  aborted after generating (`ggml.c:4648`), for four rungs, because every R12-R15 gate runs the *dense* arm.
  The value is `/*truncate*/ true`: the graph's own `ggml_rope_multi` indexer-key node leaves `no_trunc` 0 and
  nothing calls `ggml_rope_set_truncate` on it, and the pre-`#30206` 5-argument helper floor/ceil'd
  unconditionally.  Rule: a fused kernel that borrows another node's parameter block owes a check of what each
  borrowed helper *asserts on*, and each re-derived fused kernel owes a gate arm that **reaches** it
  (`GREEDY-PURITY.md` §85, `TODO` #297).
- **The r193 MTP draft window's reserve is derived in one place (r236, re-base R16).**  The reserve that makes
  Goal C pay (the slab holds `headroom + aux` back *before* the draft context exists) is derived by
  `common_draft_reserve_derive()`, which both `--fit` and the MoE preflight call; the `mtp_shared` branch
  (embedded MTP head + an active draft window ⇒ estimate from the **target's** GGUF with the weights zeroed,
  and pass `n_seq_max`/`kv_unified` into `llama_spec_draft_kv_cells()`) lives there and nowhere else.  When
  upstream absorbed the fit plumbing into `common_fit_apply()`, the branch was the casualty: the flat
  `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` came back.  Measured on 35B-A3B Q8_0, 1 GPU, `-ncmoe 48`,
  `-c 262144`, `--spec-type draft-mtp`: with the branch, `draft CONTEXT 1102 MiB` → arena 20 527.5 MiB
  (62.9 %); without it, the flat 3584 MiB → 17 977.5 MiB (55.1 %).  +2 550 MiB of arena.
- **A shared host/device memory pool must not be split: `-ncmoe` is refused on an integrated GPU (r236, R16
  cross-platform validation).**  On a UMA device (gfx1151) "host" and "device" are one physical pool, so a
  host-resident expert set buys no capacity, and the tier that stages a host master is already denied there
  (`moe-expert-cache.cu` refuses the arena on a shared pool).  The placement was not denied, so `-ncmoe N` put
  the last N layers' expert banks in a `ROCm_Host` buffer nothing could stage and the consumer read a page the
  GPU cannot address: Qwen3.8-Flash-Next GSQ-IQ3_XXS, `-ncmoe 48` (and `24`), `mul_mat_q<Q2_0,16>` at a host VA
  -> `HSA_STATUS_ERROR_MEMORY_FAULT`; with a drafter + KV live the same arm died in the VMM pool's free-order
  assert instead.  `-ncmoe 0` ran, and 35B-A3B `-ncmoe 48` ran (its expert GEMM never reads the master
  in-kernel), which is what made it look model-specific.  The pre-re-base r235 binary faults identically, so it
  is pre-existing.  `llama_model_loader::set_host_experts_umem()` (set once from `llama_model` when any device is
  `GGML_BACKEND_DEVICE_TYPE_IGPU`) now drops `n_cpu_moe_from_end`, answers false from
  `host_expert_selected()`, short-circuits `plan_host_expert_placement()` and skips a CPU expert
  `--override-tensor`, with one WARN.  Inert on a discrete GPU (gfx1201: `35B-A3B -ncmoe 20` still pins
  16 320 MiB of host experts).
- **No plain device-pool allocation may be created before a `pool_stack` push in the same scope (r236, re-base R16
  follow-up).**  `ggml_cuda_pool_vmm::free` asserts LIFO (and the non-VMM pool silently corrupts its accounting), so
  a block that is *registered* on the context's `pool_stack` and released after the whole graph must be taken
  **before** any plain temporary in that scope.  `ggml_cuda_mul_mat_id_mmb_glu` took the slot last, so on scope
  exit it freed its five temporaries below a live slot -- `pool free OUT OF ORDER: freeing 2176 bytes ...`, with
  `GGML_CUDA_POOL_TRACE=1` naming the allocator (gfx1151, Flash-Next + MTP drafter, every run, r235 and r236 alike).
  Fixed by hoisting the slot; the same rule was already paid for by `hc_combine_norm` at R13 and by the MMB/BF16
  pool blocks before R12.  Two consequences: (a) the violation is **layout/timing sensitive** -- the same source
  gave 5/5 aborts on one build and 0/4 on the next -- so "it passed once" is not evidence, and the pool's free-order
  assert always names sizes/offsets now (`GGML_CUDA_POOL_TRACE=1` adds the stacks); (b) the *class* is open: every
  `pool_stack_push` site still owes a check that no plain allocation precedes it in its scope (`TODO` #298).
- **The attendance pool is a distinct-memory design and is wrong on a shared host/device pool (r236, measured on
  gfx1151).**  `LLAMA_KV_REGION_POOL` keeps the authoritative KV in a pinned host *mirror* and fills a compact
  device pool from it by DMA, keyed by a page table with a MARK/CLAIM/MAP+FILL resolve.  On an integrated GPU the
  mirror and the weights are the same physical memory, and on gfx1151 the resolve is **wrong**: with the WHOLE
  cache resident (8192 pages) `LLAMA_KV_POOL_VERIFY=1` reports ~93 M byte mismatches (93 399 360; 95 173 888 on
  the forced-miss arm) and three runs of the same arm give three different texts, while the sparse reference is
  stable (`1e944569dbd6`).  So it is not a residency/thrash effect and not a text-noise effect -- the resolve
  itself is wrong, at least partly racily.  The **window** (Goal A, one VA range, no mirror) and the sparse decode
  arm are fine on the same box (`be904e86f248`, identical to gfx1201), which is what localises the defect to the
  mirror/pool tier rather than to the KV placement.  Until fixed, do not claim the pool for gfx1151 (`TODO` #300).
- **A single `rocprofv3 --kernel-trace` run cannot resolve a sub-1.5 % kernel delta — measure twice, and argue from
  the arm (r236, `TODO` #303).**  Two consecutive traces of the **same** binary gave dense-`mul_mat_q` totals of
  521.26 and 527.11 ms: a **±1.1 % run-to-run floor**, larger than the effect being chased.  Per-kernel numbers need
  **2+ interleaved rounds** (interleave *builds* within a round, not rounds within a build — the second round of every
  build was uniformly ~0.35 % slower, which interleaving cancels and sequential runs do not), and anything that has to
  be argued should be quoted as the interleaved `llama-bench -r 5` wall-clock ratio.  A "VGPR count changed" signal is
  not evidence of anything on its own: it tracks the *body*, so two builds with different bodies will always differ.
- **The gfx1201 ISA is the exact, noise-free arbiter for "same template instantiation, different behaviour" (r236,
  `TODO` #303).**  The whole device program is one `.hip_fatbin` section of `libggml-hip.so` holding one
  `__CLANG_OFFLOAD_BUNDLE__` per TU (`llvm-objcopy --dump-section .hip_fatbin`, then parse the bundle header — note
  the entry offsets are **bundle-relative**, not section-relative — then `llvm-objdump -d
  --disassemble-symbols`; scripted in `archive/work/upstream-rebase-261108/isa-kernel-diff.py`).  Comparing the normalised instruction sequence answers in seconds what profiling cannot:
  r236's dense MMQ came out **identical to unpatched `a691b2345`'s** (1436 instructions, 0 diff lines), which is what
  closed `#303` as an inherited cost rather than a lost delivery win — and it refuted two candidate fixes (a `min()`
  removal: 1436 → 1435 instructions; a type-gated NVFP4 prologue: byte-identical) before a single measurement was
  spent.  Check the ISA *before* benchmarking a codegen hypothesis; the same dump also gives the static instruction
  mix, which shows a "big" regression is usually four instructions and a scheduling change.
- **`llama-bench`'s prompt has no routing locality, so it CANNOT measure the expert arena or the host-expert
  transport (r236, `TODO` #296/#291).**  Its prompt is `std::rand() % n_vocab` (`tools/llama-bench/llama-bench.cpp`),
  i.e. uniformly random token ids: every ubatch routes over all 256 experts of every layer.  On that shape the
  arena is measurably **inert** — 35B-A3B Q8_0, 1 × R9700, `-ncmoe 48`, `-ub 512`, pp1450 = **414.1 / 414.5 /
  414.5 / 414.5** t/s at `MOE_EXPERT_CACHE_MIB` 0 / 4096 / 8192 / auto — while the same arm driven with the
  shipped prose prompt reads **401.6 → 583.6 → 676.9 → 1063.6**.  The delivery looks −15…−18 % behind unpatched
  upstream on the random shape and **+181 % ahead** on real text (1064 vs 379 t/s), and the arena's prefill is
  838 t/s there against 1442 on real text.  Any A/B of the pool/arena/host transport must be driven with real
  text (`llama-cli -f prompts/prose-rdna-boosts.txt -n 0`, now gate step 5b); `llama-bench` remains valid for
  **build-vs-build** regressions (both sides see the same degenerate prompt) and for kernels whose cost does not
  depend on routing.
- **`llama-batched-bench`'s npl=1 prefill row is a cold-first-pass artifact (r236; same trap as
  `archive/work/host-expert-eager-push/HANDOVER-R210.md` §3.1).**  Three identical npl=1 rows in one process read
  ours **0.824 / 0.207 / 0.197 s** (Flash-Next GSQ, 3 devices `-sm tensor`, 512-token prefill) and upstream
  **0.255 / 0.244 / 0.237 s**, so the first row understates the steady state by ~4× and must be discarded (a
  one-time ~0.6 s cost on the first prefill of a process is all that is real).  The **decode** column is sound.
- **`llama-batched-bench -c 16384` on Flash-Next at `-ncmoe 48` segfaults silently (`TODO` #270); it is a flag-set
  landmine, not a general unreliability.**  With `-c 32768` and no `-ncmoe` (or with `-c 8192`) the same binary
  prints complete tables and produces no assert, so the arm is usable once that shape is avoided.
- **A "numeric arm" is the anchor, not byte-identity (r236, `TODO` #292).**  The dsv4 `DSV4_HC_PRE/_POST` ops own
  the HC16 bf16 producer slots by design: vs their own ggml chain they read **0.013217 / 96.392 %** and
  **0.013198 / 96.186 %** mean KLD / same-top-p on the BF16-hc class (base PPL 2.8489), and they are
  **deterministic per class** (r235 and r236 produce the same `6a102dbc62d6`).  So they are gated by their own
  recorded text (`SOFT_EXPECT` in `scripts/gate-hc-fusions.sh`), never by chain identity — a stale "both arms
  agree on BF16-hc" line in that gate's header sent `#292` after a re-base drift that did not exist (r235's chain
  reads `d62340fbf566` too).
- **Delivery-vs-unpatched-upstream prefill-logit KLD grows with the surface the delivery replaces, and is
  insensitive to every kill switch it ships (r236, `TODO` #293).**  Measured against a base recorded from a
  detached `a691b2345` (wikitext 512 × 40): dense 27B Q8_0 (`qwen35`) **0.000697 / 98.784 %**, MoE 35B-A3B Q8_0
  **0.004980 / 97.127 %** (unchanged with MMB/pair/router-topk off, with `GGML_CUDA_DISABLE_MMQ=1`, and only
  0.004483 with `GGML_CUDA_GDN_CHUNKED=0`), qwen4exp Flash-Next **0.013426 / 96.431 %** (0.013955 with every HC
  op/fusion off, 0.013081 with the derived mask, QSA, GDN-bf16, rms-vec and all fusions off).  It is the
  accumulated arithmetic order of our own MoE/attention/HC implementations, **not** the qwen4exp model file — and
  delivery-vs-delivery is still exact (r235 == r236 text on the field config, `87 ac6445f74aa2`).
  `Qwen3.8-27B` is arch `qwen35`, so no `qwen4exp.cpp` change can affect it.
- **On qwen4exp the delivery's advantage grows with depth, and 160 K is the decisive arm (r236, `TODO` #293).**
  Ours vs unpatched `a691b2345`, Flash-Next GSQ-RCO-IQ3_XXS, 3 devices `-sm tensor`, device-resident (no
  `-ncmoe`): `pp512` **+16.6 / +28.8 / +173.4 %** at d 0 / 16 384 / **163 840**, `tg128` **+98.2 / +98.9 /
  +68.2 %**, real-text prefill **+19.1 / +88.4 %** at ~16 K / ~160 K, and aggregate concurrency **+41.6 / +67.2 /
  +53.7 %** at 1 / 2 / 4 slots.  A shallow single-stream arm is not a basis for judging the qwen4exp/indexer
  work.
- **`r235` cannot run `llama-perplexity` on Flash-Next (r236 fixed it).**  The shipped build aborts in
  `ggml-cuda.cu:1375` (`GGML_ASSERT(ptr == pool_addr + pool_used)`, the pool-order class `TODO` #298 fixed) under
  `-sm tensor` and core-dumps under `-sm layer`, so a delivery-vs-delivery KLD arm is not available on that model
  — compare text/throughput instead, or use r236-vs-upstream.
- **The release gate must not time the perf arms straight after the per-block build ladder, and it must not
  inherit `LOGDIR` (r236, found running R17).**  (1) The ladder leaves the GPU hot/busy: the dense-27B arm read
  **−3.44 % on round 1** against −1.68 / −1.59 after (verdict −2.24 %, spread 1.85) while the same A/B on a quiet
  box reads −1.31 / −1.19 / −1.32 (**−1.28 %, spread 0.13**), so `gate-rebase-release.sh` now runs one
  **discarded warm-up round** (`WARMUP=1`).  (2) `gate-prefill-logits.sh` uses the variable `LOGDIR` for its own
  log dir and `rm -rf`s it on success, so an *exported* `LOGDIR` makes step 3 delete the orchestrator's log
  directory mid-run and every later step fails on missing logs; the orchestrator now does `export -n LOGDIR`.
