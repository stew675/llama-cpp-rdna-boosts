# HANDOVER-R13 — block 12 (campaign remainder), then R14 → R17

**Read order:** this file → [`RUNGS.md`](RUNGS.md) (R13's section + the ledger) → [`r13-survey.md`](r13-survey.md)
(the per-file conflict map) → [`DECISIONS.md`](DECISIONS.md) (`D8`, `D15`) → [`HANDOVER.md`](HANDOVER.md)
(§0.1 hard rules, §0.2 environment — they still apply verbatim) → [`../../AGENTS.md`](../../AGENTS.md).

**State as of 2026-11-09, end of the session that finished R13 (this file is now the cold-start brief for
R14-R17; the R13 brief below is kept as the record of how it was resolved).**

| item | value |
|---|---|
| fork checkout | `~/llama.cpp`, branch **`rebase-r236`**, **`2b09a50b0`** (block 12), tree `87e2841f7`, working tree clean; the R13 work tree is also on `r13-work` (4 wip commits) until the next rung makes it redundant |
| `master` in `~/llama.cpp` | the pristine snapshot at `a691b2345` — never commit to it |
| build | `~/bin/build-llama-rocm` (VMM on: `-DGGML_HIP_NO_VMM=OFF`, required by the slab). Green at R12 |
| ladder | **13 of 16 blocks done** (R01–R13 = blocks 00–12). Next: **R14** (block 13), R15 (block 14), R16 (block 15), R17 (release ladder) |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-r236`** (pushed); `main` still holds the pre-re-base delivery — the switch-over is R17 |
| R13 committed | `d84c5d74c`, re-cut as `2b09a50b0` with the series identity (deterministic committer date) so the tip is reproducible; `patches/`, `release.json` and the fork's `rdna-boosts` branch stay untouched until R17 |

## §1 How R13 was resolved (it was parked; this is the outcome)

`be4749a56` (block 12, 81 files, +5893/−995) cherry-picks into **8 files / 24 hunks**, and three of those
places are **not merges**: upstream and the delivery independently built the same thing, so each needs a
measured decision before any code is written.  Guessing them would either drop delivery code (forbidden) or
carry a duplicate mechanism forward (the `D1` mistake).

| # | decision | the two implementations | oracle |
|---|---|---|---|
| 1 | **`D15`** | delivery F1 (`llama_lf_*`, `cparams.lf_layer_inp_dev_store[]`, `llm_graph_result::set_layer_inp()` emitting per-layer `set_rows` stores) vs upstream's `llama_set_embeddings_layer_inp_dev()` / `llama_supports_embd_row()` / `llama_batch_ext_add_embd_row()` + `#31506` device copies | a DFlash2 draft arm (acceptance + text), see §5 |
| 2 | **state save/restore batching** | delivery `state-pipe` (`llama_state_async_enabled`, `state_stage`, `state_pipe_event`, `llama_state_pipe_groups`, the per-(backend, group) events) vs upstream's new batched `llama_io_write_host` (`winfos`, `base_ptr`) | `scripts/gate-state-pipe.sh` (5 arms, byte-identical restore) |
| 3 | **server prompt cache** | delivery prompt-cache residency + idle-slot parking (r130/r146) vs upstream's reference-counted checkpoints (`#30137`) and slot visitation order (`#30779`) | `scripts/gate-park-idle-slots.sh`, `scripts/gate-ckpt-reuse.sh` |

**Expected shape (do not treat as decided):**

1. `D15` → **subsumption, keep upstream's**, and carry exactly one thing from F1: the correctness guard that
   with **host-resident MoE experts (`-ncmoe`) and more than one sequence** the device path gave draft
   acceptance near 0 (the armed slab breaking the drafter's device reads; `GGML_CUDA_SLAB=0` or one sequence
   was fine), so F1 forced the host path there: `lf_dev = lf_on && (n_seq == 1 || !host_experts)`.
   Re-express that around `llama_set_embeddings_layer_inp_dev()` in `common/speculative.cpp`, then
   **re-measure** — R10/R11 changed the slab, so the guard may now be unnecessary (keep it if the draft arm
   shows any acceptance loss; a silently dead drafter is much worse than a host copy).  Deleting F1 removes
   `llama_lf_*` from `src/llama-ext.h`, `lf_layer_inp_dev_store`/`lf_row_offset`/`lf_dev_inject` from the
   cparams, `set_layer_inp`/`lf_emit_layer_inp_store`/`lf_store_armed`/`t_lf_rows` from `llama-graph.*`, the
   `lf_dflash_target_features` call in `src/models/dflash.cpp`, and `res->set_layer_inp()` in
   `src/models/gemma4.cpp` — grep each name for other users before deleting, and delete the users too.
2. State pipe → measure first.  If upstream's batching is the same mechanism, ours is dead code and goes
   (and `gate-state-pipe.sh` becomes a "cannot run" rung state, recorded not deleted); if it is not (grouping
   across backends, the async stage buffer, pipelining), keep ours and re-derive it on upstream's
   `winfos`/`base_ptr` shape.  The gate enumerates the arms, so run it as the first thing.
3. Prompt cache → resolve the **API** to upstream's shape (`load(ctx_tgt, ctx_dft, id_slot)` vs the delivery's
   `find(prompt, tokens_new)`) and keep the delivery's residency/parking **policy** on top of it: it is
   policy, not API.  Then the two gates decide.

## §2 The conflict map (8 files / 24 hunks)

| file | hunks | shape | direction |
|---|---|---|---|
| `ggml/src/ggml-backend.cpp` | 1 | mechanical: upstream frees `prefetch_events[b]`, the delivery frees `copy_events[b][e]` | **both** families freed at teardown (the same "both" as R10's scheduler work) |
| `src/llama-graph.cpp` | 2 | F1 hooks (`set_layer_inp`, `lf_emit_layer_inp_store`, `lf_store_armed`) vs upstream's `set_outputs(params)` / `set_params(params)` | follow decision 1 |
| `src/models/gemma4.cpp` | 1 | `res->t_layer_inp[il] = inpL` (upstream) vs `res->set_layer_inp(il, inpL)` (delivery) | follow decision 1 |
| `src/models/dflash.cpp` | 2 | `ubatch.embd_row` + `build_inp_target_features()` (upstream) vs `cparams.lf_dev_inject` + `lf_dflash_target_features()` (delivery) | follow decision 1 |
| `common/speculative.cpp` | 1 | `llama_set_embeddings_layer_inp_dev` + `llama_supports_embd_row` (upstream) vs `llama_lf_set_layer_inp_dev` (delivery) | decision 1, incl. the host-experts guard and the `batch_inject_tok` allocation (upstream's `llama_batch_ext_add_embd_row` may make the latter unnecessary) |
| `src/llama-context.cpp` | 13 | decision 2 (state io), the F1 inverse map / `lf_dev_rows`, `#31141`/`#31200` memory-instance drift | per hunk |
| `tools/server/server-context.cpp` | 3 | decision 3 | per hunk |
| `tools/server/server-task.cpp` | 1 | decision 3 (`load` vs `find`) | per hunk |

## §3 Resolution recipe (what worked at R11/R12)

1. `cd ~/llama.cpp && git status` → clean at `b1b0fa807`.
2. `git cherry-pick be4749a56`; expect the 8 conflicts above.
3. Resolve **decisions first** (they change which side wins in several files), then the mechanical hunks.
4. When a whole file is a fork of two implementations (as `src/models/qwen4exp.cpp` was at R12), do **not**
   hand-merge the fragments: `git checkout <block-sha> -- <file>` to take one side wholesale, then drive the
   base-API drift out with the compiler.  It is faster and gives a coherent implementation instead of a
   Frankenstein (this is exactly how R12 went wrong first and right second).
5. Base-API drift to expect on every rung now (all seen at R12, all mechanical):
   `#31733` — `model` is a member of `llm_graph_context`, so ctors/helpers lose the `const llama_model &`
   argument (and a derived class may not shadow `model`); `#31142` — output structs instead of
   `ggml_tensor **` out-params; `#31141`/`#31200` — memory instances take `no_alloc` and no `n_pad`;
   `#31073` — rotation constants are per-layer vectors (`self_k_rot[il]`), filled at graph-build time;
   `#29510`/`#31683` — `get_kq_mask()` is gone, use `self_attn_mask.raw`; `build_attn_mha` takes `n_kv_max`;
   `ggml_rope_yarn_corr_dims` takes the truncate flag.
6. Build often (`~/bin/build-llama-rocm`, incremental ~40 s).  `clean` only when CMake options moved.
7. Remove dead code in the same commit: after a subsumption the loser's helpers, declarations and env
   variables must go (grep each symbol across `ggml/`, `src/`, `common/`, `tools/`, `tests/` first).  R12 left
   `n_pad` dead and it was deleted, not carried.

## §4 Regression oracles — texts to compare against (intra-tree, this re-base)

A re-base moves numerics on purpose, so these are **regression detectors for this branch**, not a comparison
with r235.  Same command shape: `llama-cli -ngl 99 -c 8192 -f prompts/prose-rdna-boosts.txt -n 16 --seed 42
--temp 0 --no-display-prompt --single-turn`, no `-lv`, hash with `scripts/extract-generated.py`.

| arm | expected text |
|---|---|
| dense `Qwen3.5/4B/Q8_0` , 1 GPU | `1472ebdadd46` (87 t/s) |
| `Qwen3/30B-A3B/Q4_K_M`, 1 GPU | `013d09f165a5` (144 t/s) |
| `Qwen3.6/35B-A3B/Q8_0`, 3 GPU `-sm tensor -mg 0`, `GGML_CUDA_ALLREDUCE=rccl` | `90decf558d96` (80 t/s) |
| `Qwen3.8/Flash-Next/GSQ-IQ3_XXS`, 3 GPU `-sm tensor -mg 0` | `7a8734a9f1a6` (pp 1432 / tg 70) — **the DFlash2 draft arm's target** |
| `Qwen3.8/27B/Q4_K_M`, 1 GPU | `b21e94a5b6e1` (pp 1303 / tg 32) |

If a text moves, that is a finding to report *before* committing — do not "fix" it back.

## §5 R13 verification — DONE, results in `RUNGS.md` → R13

The exit criteria were met: build green, coherence unchanged against the R12 tree on the dense 4B,
30B-A3B, 35B-A3B 3-GPU tensor and Flash-Next, `gate-state-pipe.sh` / `gate-park-idle-slots.sh` /
`gate-ckpt-reuse.sh` green, and the DFlash2 drafter attaches **and drafts** (0.754 / 0.741 / 0.405
acceptance on the 27B / 35B-A3B / Flash-Next arms — it was 0.000 because of an unregistered graph
input, see `INVARIANTS.md`).  The commands below are kept as the reference for re-running them.

## §5.1 The R13 verification commands (as issued)

```bash
cd ~/llama.cpp && ~/bin/build-llama-rocm            # green, no `<<<<<<<` (git grep '<<<<<<<')
# 1. coherence: the three oracles in §4 (4B, 30B, 35B tensor) + Flash-Next
# 2. the three decision gates
~/llama-cpp-rdna-boosts/scripts/gate-state-pipe.sh
~/llama-cpp-rdna-boosts/scripts/gate-park-idle-slots.sh
~/llama-cpp-rdna-boosts/scripts/gate-ckpt-reuse.sh
# 3. the DFlash2 draft attach smoke — the one thing the gates do not cover
HIP_VISIBLE_DEVICES=0,1,2 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf \
  --spec-draft-n-max 3 -ngl 99 -c 8192 -sm tensor -mg 0 \
  -f ~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt -n 32 --seed 42 --temp 0 \
  --no-display-prompt --single-turn
```

At R12 that smoke **aborts** with `dflash requires ctx_other to be set` (`src/llama-context.cpp`): the drafter
cannot be attached at all before block 12's DFlash work (r106/r130/r162/r163/r218).  R13 is done only when it
attaches, generates, and reports a sane draft acceptance (`-lv 4` shows the draft acceptance line).

Then commit the rung on the fork (block 12's subject + a "re-based onto `a691b2345`" paragraph naming the
three decisions and their measurements), and update the delivery repo: `RUNGS.md` (rung log + ledger),
`DECISIONS.md` (`D8`/`D15` → DECIDED with numbers), `TODO.md`, `HANDOVER.md`, plus everything in §7.

## §6 After R13 — the rest of the ladder (all in `RUNGS.md` with per-rung detail)

| rung | block | expect |
|---|---|---|
| **R14** | 13 MTP draft vocabulary subset (`6ee8eb290`) | 3 conflicts, all graph/model-member refactors; exit: `--spec-draft-mtp-vocab N` runs and is inert when unset |
| **R15** | 14 `-sm relaxed-tensor` (`59a3e5206`) | **0 conflicts** (upstream's `#31147` only added laguna to the deny list, so the amax arches are still refused); exit: a flint/mellum/mistral4 MoE loads under the relaxed mode and matches `-sm layer` text |
| **R16** | 15 streamed KV (`69d36099b`) | 9 conflicts; `D12`/`D13` (quantized-KV state, `#31073`/`#31570` kv_rows, memory instances); gates `gate-kv-residency.sh` + `gate-kv-window-split.sh` + `gate-spec-draft-window.sh` |
| **R17** | — | the release ladder: `make-patches.sh` → `make-release.sh --base a691b2345` → `validate-set.sh` → `gate-per-block-build.sh` (all 16) → coherence + the host-expert and hybrid arms → `gate-prefill-logits.sh --record` **from a clean unpatched `a691b2345` build** then compare (+ `--sm tensor --device 0,1,2`, `--kv q8_0`) → MTP four-axis + `llama-batched-bench -npl 1,4,8` → every block gate → `check-gpu-faults.sh` → `fork-refresh.sh --build --gates --push` (personal fork only) → WORKLOG + headers + `release.json` |

## §7 Documentation owed at R17 (collect per rung — do not lose these)

* **`ENVIRONMENT.md`**: the new/changed variables this climb produced — `LLAMA_KV_N_PAD_MIN` /
  `_PREFILL` / `_PREFILL_FROM` (kept, `n_pad` itself is gone), block 12's variables, block 15's
  `LLAMA_KV_RESIDENT_CELLS` / `LLAMA_KV_REGION_POOL` (already there) and everything block 12 adds; the
  `GGML_HIP_NO_VMM` row must state the delivery's build requirement (R10); `LLAMA_RS_ROLLBACK_DEBUG` is now
  the RSPREP prepare-order instrument only (R04/R12); `GGML_CUDA_DISABLE_MMQ_ROUTED` semantics unchanged but
  the dispatch now keys on `nchannels_x` (R11).  If decision 1 deletes F1, its `GGML_LF_*` / `LLAMA_LF_*`
  variables go too.
* **`patches/README.md`**: block 00's MUL_MAT_ID shape + `-Wshadow` note (R01), block 03's tracker/warn text
  (R04), block 09's shape (R10), block 10's dispatcher split + the pair `ntj` fix (R11), block 11's qwen4exp
  (`D3`: the delivery's implementation stays, +31 % decode, byte-identical on Flash-Next 3-GPU) (R12),
  block 12's three subsumptions (R13).
* **`INVARIANTS.md`**: already appended this session (the rename traps `ncols_max`→`ntj` and
  `tile_y_max_j`→`tile_y_j_sup`, the ids convention `nchannels_y = 1` + `tile_spec`, the single dispatcher,
  fusion-decision nondeterminism, the short-run all-reduce trap, and the `pool_stack` LIFO rule).  Add
  whatever R13–R16 teach.
* **`GREEDY-PURITY.md`**: §84 landed this session (the renamed-field doctrine + "a fallback dispatcher copy
  must be upstream's verbatim").  R13–R16 add theirs; §-numbers are permanent.
* **`TODO.md`**: #290 resolved (lands with r236), #292 (the HC dsv4 soft-arm drift — quantify with the KLD
  gate at R17), #293 (the **Qwen3.8-27B assessment**: decide whether to keep carrying our qwen4exp model
  support; the discriminating measurement is **Q8_0 on 2 and 3 GPU `-sm tensor`**), #291 (the arena-vs-upstream
  cache investigation), #286 (Meta headroom re-measure), #287/#289 (upstream PRs owed).
* **`BASELINE.md`** current baseline, **`release.json`** (`--base a691b2345`, the `.tip` from
  `fork-refresh.sh`), **`WORKLOG.md`** entry with the new base/tip/tree + gate numbers, and this campaign
  moved to `archive/work/`.

## §8 Traps this climb has already paid for (do not re-derive)

1. `-Wshadow` rename sweeps are not carried (R01).  ~94 warnings, none errors; do not restore, do not add.
2. `FLASH_ATTN_EXT`'s `src[5]` is upstream's `kv_rows`; the derived kq mask lives in `src[6..8]` (R06).
3. The host-expert arena needs VMM; the build script passes `-DGGML_HIP_NO_VMM=OFF` (R10).  **A VMM build is
   the only build where pool misuse is loud** — see the `pool_stack` rule (R12) and use it for any new
   long-lived pool buffer.
4. `ggml_cuda_launch_mm_ids_helper` emits upstream's `tile_spec` *and* the delivery's optional
   `expert_bounds`; `tile_spec` may be `nullptr` (R01/R07/R09).
5. `mmq_args`' old `ncols_max` is now `ntj` (a TILE count) and `mul_mat_q_process_tile`'s `tile_y_max_j` is
   now `tile_y_j_sup` (an exclusive count) — re-derive, never carry (R11, `GREEDY-PURITY.md` §84).
6. The routed (`ids`) convention is `nchannels_y = 1`, experts in `nchannels_x`, expert per tile in
   `tile_spec`; a gate that asks "how many experts" must read `nchannels_x` (R11).
7. Fusion decisions are **not deterministic run to run** (address-selected arms); the text is the oracle, and
   a per-node fusion trace is not an A/B oracle (R11).
8. A short decode run cannot compare all-reduce arms — `-n 16` is dominated by setup; use `-n 64`+ or
   `llama-bench` (R11: hybrid 100.2/105.6 vs rccl 90.1/90.0 at `-n 64`).
9. The upstream range absorbed three delivery mechanisms — keep the invariant, drop the duplicate code (R03,
   R04, R06).  R12 added a fourth: upstream's `embd_row` is F1's successor (decision 1 above).
10. **Never trust a gate's model selection blindly**: `gate-qwen4exp-quant-coherence.sh` was testing the
    DFlash2 drafter instead of the model, and every arm failed for an unrelated reason (R12).  Read the first
    line of a gate's log before believing its verdict.
11. `-lv` per arm: **no `-lv` for a text/hash arm**, `-lv 4` for a census arm; `llama-cli`'s own `-lv 3` both
    pollutes the text and hides the ggml censuses.  Pin `HIP_VISIBLE_DEVICES` on every multi-device arm.
12. One long-lived source path (`~/llama.cpp`) — ccache keys on absolute paths.

## §9 Open register (see `DECISIONS.md` for the full text)

`D1` MoE cache (MEASURING, `TODO` #291) · `D3` qwen4exp (**DECIDED** at R12) · `D8` which CPU wins survive
upstream's rewrite (**OPEN — R13**) · `D9` graph cache/allocator (R11 raised nothing; residue R13/R16) ·
`D11` spec hidden state / conv snapshots / checkpoints (OPEN — R13/R14) · `D12` quantized-KV state (OPEN —
R16) · `D13` memory instances (mechanical, R13–R16) · `D15` F1 vs `embd_row` (**OPEN — R13, decision 1**).
Closed: `D2` (R09), `D4` (R04), `D5` (R06), `D6` (R05), `D10` (R02), `D14` (n/a).
