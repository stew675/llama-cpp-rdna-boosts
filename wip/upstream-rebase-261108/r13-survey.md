# R13 (block 12, campaign remainder) — conflict survey, 2026-11-08

**State: RESOLVED at R13 (`2b09a50b0`, tree `87e2841f7`) — kept as the record of what the survey got right and
what it did not.  All three decisions are in `DECISIONS.md` (`D15`) and `RUNGS.md` → R13.  Two things the survey
did not predict: the state-pipe's competitor (`ggml_backend_tensor_get_n`) is CUDA/MUSA-only on the read side
(`ggml_cuda_get_tensor_n` returns false under `GGML_USE_HIP`), and the drafter's `0.000` acceptance that the
DFlash arms showed was neither a feature misalignment nor the slab -- it was a graph input that was built and
never registered (`res->add_input`), see `INVARIANTS.md`.**

**Original state: PARKED.**  The branch is back at the green R12 tip (`b1b0fa807`, tree `7baf3ba40`); nothing of
R13 is committed.  This file is the cold-start brief for the next session: it lists what conflicted,
which conflicts are mechanical, and the **three** places where upstream and the delivery built the same
thing (those are decisions, not merges).

Cherry-pick: `be4749a56` onto `b1b0fa807` → **8 files / 24 hunks** (the probe predicted 8 files):

| file | hunks | shape |
|---|---|---|
| `common/speculative.cpp` | 1 | **D15** — F1 `llama_lf_*` vs upstream's `embd_row` API |
| `ggml/src/ggml-backend.cpp` | 1 | mechanical — **both** event families (upstream `prefetch_events[b]`, delivery `copy_events[b][e]`) |
| `src/llama-context.cpp` | 13 | the state save/restore batching (**subsumption question #2**), the F1 inverse map / `lf_dev_rows`, `#31141`/`#31200` memory-instance drift |
| `src/llama-graph.cpp` | 2 | F1 hooks (`set_layer_inp`, `lf_emit_layer_inp_store`, `lf_store_armed`) vs upstream's `set_outputs`/`set_params` |
| `src/models/dflash.cpp` | 2 | **D15** — `build_inp_target_features` + `ubatch.embd_row` vs `lf_dflash_target_features` + `cparams.lf_dev_inject` |
| `src/models/gemma4.cpp` | 1 | F1 — `res->t_layer_inp[il] = inpL` (upstream) vs `res->set_layer_inp(il, inpL)` (delivery) |
| `tools/server/server-context.cpp` | 3 | the prompt-cache residency / idle-slot parking rework vs upstream's server refactors (`#30779`, `#30137`) |
| `tools/server/server-task.cpp` | 1 | `load(...)` (upstream) vs `find(...)` (delivery) — same prompt-cache rework |

## The three decisions

### 1. `D15` — F1 (`llama_lf_*`, the in-graph layer-input device store) vs upstream's `embd_row`

Upstream now has the same feature: `llama_set_embeddings_layer_inp_dev()`, `llama_supports_embd_row()`,
`llama_batch_ext_add_embd_row()` plus `#31506`'s device copies — i.e. the target exports its layer
inputs into a device buffer the drafter reads through the batch.  The delivery's F1 is its predecessor:
`cparams.lf_layer_inp_dev_store[]` + `llm_graph_result::set_layer_inp()` emitting a `set_rows` store per
in-epoch layer, `lf_row_offset`, `llama_lf_set_layer_inp_dev()`, `lf_dev_rows`/`lf_dev_inject`.

**Expected resolution: subsumption — keep upstream's, delete F1, carry ONE delivery finding across.**
The finding is a correctness guard, not plumbing: with host-resident MoE experts (`-ncmoe`) *and* more
than one sequence, the armed slab was measured to break the drafter's device reads (draft acceptance
near 0; `GGML_CUDA_SLAB=0` or one sequence fine), so the delivery forced the host path there:
`lf_dev = lf_on && (n_seq == 1 || !host_experts)`.  Re-express that on upstream's path in
`common/speculative.cpp` around `llama_set_embeddings_layer_inp_dev`, then re-measure whether it is
still needed on the re-based slab (R10/R11 changed it) — the guard is cheap, a silent 0-acceptance
draft is not.  Deleting F1 also removes `llama_lf_*` from `llama-ext.h` and the
`lf_layer_inp_dev_store`/`lf_row_offset`/`lf_dev_inject` cparams members — check each for other users.

### 2. The state save/restore batching — delivery `state-pipe` vs upstream's batched `llama_io_write_host`

Upstream's `llama_io_write_host` grew its own batching (`winfos`, `base_ptr`), while the delivery's
`llama_state_async_enabled` / `state_stage` / `state_backend` / `state_pipe_event` /
`llama_state_pipe_groups` machinery does group-aware pipelined copies with per-(backend, group)
events, plus `LLAMA_STATE_TIMING`.  `scripts/gate-state-pipe.sh` exists precisely to prove the
delivery's arms (batched / plain memcpy / pipelining off / save-only / synchronous) restore
byte-identical state — it is the only gate that reaches that code.

**Measure, don't assume:** if upstream's batching is the same mechanism, the delivery's is dead code and
should go (the gate then becomes inapplicable, like other vacuous arms); if it is not (grouping,
cross-backend pipelining, the async stage buffer), keep the delivery's and re-derive it on upstream's
`winfos`/`base_ptr` shape.  The gate is the oracle and it already enumerates the arms.

### 3. The server prompt cache — delivery residency/parking vs upstream's server rework

`server-task.cpp`'s `load(ctx_tgt, ctx_dft, id_slot)` (upstream) vs `find(prompt, tokens_new)`
(delivery) is an API-shape difference hiding a feature difference: the delivery's r130 "server
prompt-cache residency" + r146 "parking moves a mirroring indexer cache with the attention cache"
(`scripts/gate-park-idle-slots.sh`) vs upstream's reference-counted checkpoints (`#30137`) and slot
visitation order (`#30779`).  Resolve the cache-entry API to upstream's shape, then decide whether the
residency/parking behaviour rides on top of it (it should: it is a policy, not an API) and re-run
`gate-park-idle-slots.sh` + `gate-ckpt-reuse.sh`.

## Exit criteria (RUNGS.md, already amended)

build + coherence + `gate-park-idle-slots.sh` + `gate-ckpt-reuse.sh` + `gate-state-pipe.sh`, **plus the
DFlash2 draft attach smoke** (`llama-cli -m <Flash-Next> -md <DFlash2 drafter> --spec-draft-n-max 3`):
at R12 that aborts with `dflash requires ctx_other to be set`, and the block-12 DFlash work
(r106/r130/r162/r163/r218) is what makes it work.
