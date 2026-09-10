# Handover — securing the QSA visibility flip (−800 MiB/GPU, twice over)

**THIS IS THE TASK.** The flip that deletes the 800 MiB kq-mask tensor is *written*, *compiles*, and
its memory win is *measured* — but it aborts on a null-buffer assertion in the backend plumbing, so
it is **not applied**. The tree is at the validated state and patch
`patches/0003-prune-mask-flip-NOT-APPLIED.patch` contains the flip ready to apply. The remaining
work is a backend-robustness question, not a QSA question.

## 1. The prize (measured, not projected)

`tools/bufsize.sh <bin> 2048` (loads at ctx 204800 / ub 2048, reads
`Meta() compute buffer size` + `ROCm_Host compute buffer size`):

| state | compute / GPU | host | box delta |
|---|---|---|---|
| fork point (upstream) | 6690.40 MiB | 1262.70 MiB | — |
| + L2 (score chain) | 4450.40 MiB | 1262.70 MiB | −6.7 GiB |
| + steps 1–2 (bias) | 4051.39 MiB | 863.69 MiB | −1.2 GiB |
| **+ flip (mask gone)** | **3251.39 MiB** | **63.69 MiB** | **−3.2 GiB** |

Read the last row carefully: the mask costs **800 MiB in the per-GPU compute buffer *and* 800 MiB in
the shared host buffer** (both drop by exactly 799.21 MiB when it is gone). So the flip is worth
**3 × 800 MiB of VRAM (one per GPU) + 800 MiB of host RAM** — the single biggest remaining lever,
and it lands ub2048 at *less* memory than **pristine ub1024** (3347 MiB) while keeping ub2048 speed.
It also deletes the O(n_kv × n_tps) host mask build from every prefill ubatch (the mask is never
built at all), so prefill should get faster as well (unmeasured so far).

Reproduce (with the flip applied, see §3):
```bash
GGML_QSA_DERIVED_VIS=1 bash tools/bufsize.sh /tmp/bin-l1g 2048   # → 3251.39 / 63.69
GGML_QSA_DERIVED_VIS=0 bash tools/bufsize.sh /tmp/bin-l1g 2048   # → 4051.39 / 863.69 (reference)
```

## 2. State of the tree (nothing below is committed to the fork — by design)

- `~/llama.cpp` (rdna-boosts `e2380eb67`) has **9 files modified in the working tree**:
  `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`,
  `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/indexer-topk.cu`,
  `ggml/src/ggml-cuda/fattn-qsa.cu`, `src/llama-memory-hybrid-idx.{h,cpp}`,
  `src/models/qwen4exp.cpp`. **Never commit them on the branch** (`scripts/make-patches.sh` treats
  the fork tip as canonical); regenerate patches with the worktree recipe in §7 instead.
- Patch inventory in this directory:
  - `patches/0001-L2a-L2m-qsa-score-memory.patch` — the L2 lever (validated).
  - `patches/0002-derived-qsa-block-bias.patch` — **the current state** = L2 + steps 1–3 (derived
    per-block bias, derived visibility *state*, FA support with a null-mask tolerance). Applies
    cleanly on `0001`. The tree matches it.
  - `patches/0003-prune-mask-flip-NOT-APPLIED.patch` — **the flip** (33+/7−, 2 files:
    `src/models/qwen4exp.cpp`, `src/llama-graph.cpp`). Applies cleanly on top of `0002`.
- Binaries: `/tmp/bin-l1h` = the current validated state; `/tmp/bin-l2`, `/tmp/bin-pristine`,
  `/tmp/bin-keysonly`, `/tmp/bin-l0base`. The flip's aborting binary was `/tmp/bin-l1g` (volatile —
  rebuild after applying 0003).
- Gates: `GGML_QSA_DERIVED_BIAS` = 0/1 (2/3 = diagnostics) and `GGML_QSA_DERIVED_VIS` = 0/1.
  **A/B by running the same binary with `=1` vs `=0`** — that is the validation idiom used
  throughout (derived vs tensor/mask path in one process-binary pair).

## 3. Reproduce the failure in one run

```bash
cd ~/llama.cpp && git apply <R>/patches/0003-prune-mask-flip-NOT-APPLIED.patch
export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH
cmake --build build-rocm --target llama-cli llama-bench -j 16
rm -rf /tmp/bin-l1g && cp -r build-rocm/bin /tmp/bin-l1g
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_FA_WMMA_256=0
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4-XS-00001-of-00003.gguf
GGML_QSA_DERIVED_VIS=1 /tmp/bin-l1g/llama-cli -m $M -f /tmp/prompt3k.txt -n 24 --seed 42 --temp 0 \
  --single-turn --no-display-prompt -c 204800 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -fa auto \
  -ngl 99 -sm tensor -mg 0
```
(The real model path is `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
— double-check the exact name with `ls /models/Qwen3.8/Flash-Next/IQ4_XS/`.)

The reserve line prints **3251.39 MiB** first, so the failure is at *compute* time, not allocation:
`GGML_ASSERT(buffer) failed` inside `ggml_backend_buffer_get_usage` (the assert is the first
statement of that function; its caller at that moment is the scheduler's split-input copy loop,
`ggml_backend.cpp` ≈L1730).

## 4. What is known about the cause

- The mask really is gone: the host buffer drops to 63.69 MiB (= only the other inputs), so the
  flip's graph is correct and the win is real.
- A **tensor with a null buffer** is being probed. That is expected for a tensor that is *created
  but never consumed* (the gallocr never allocates it) — exactly what the mask is now. The backend
  plumbing does not tolerate that in a few places.
- Unguarded `tensor->buffer` probes found by audit (all in the meta/sched path):
  - `ggml_backend_buffer_get_usage(input->buffer)` — `ggml-backend.cpp` ≈L1730 (sched split-input
    copy; an *optimization* branch: `get_usage(...) == WEIGHTS && is_host(...)`, i.e. a null buffer
    means "not a weight" and the branch should simply be skipped);
  - `ggml_backend_buffer_get_type(tensor->buffer)` — `ggml-backend-meta.cpp:856`, inside
    `calculate_split_state`'s `get_usage(...) != COMPUTE && view_src == nullptr` branch;
  - `ggml_backend_buffer_is_host(tensor->buffer)` — `ggml-backend-meta.cpp:1249, 2138, 2275, 2314`.
- Making `ggml_backend_buffer_get_usage(nullptr)` return `GGML_BACKEND_BUFFER_USAGE_COMPUTE` (which
  is *semantically right*: weights are always allocated at load time, so an unallocated tensor is a
  compute tensor — and it is also harmless for every caller: it only ever *skips* weight-specific
  fast paths) moved the abort to the **next** probe (`get_type`). So this is a *chain*, not one fix:
  the useful diagnostic is to fix them one at a time and watch the failure move — each one you
  pass is a probe that was wrong to assume a buffer.
- Interesting hint: `ggml_backend_meta_get_split_state` dereferences `tensor->buffer` in its *first*
  two statements (`ggml_backend_meta_buffer_n_bufs(tensor->buffer)`,
  `tensor->buffer->context`) — that would *fault* (not assert) if the suspect tensor passed through
  it. Since we see an assert, the suspect tensor is **not** going through `get_split_state`, which
  narrows it: it is reached by `get_usage`/`get_type`/`is_host` from the scheduler or the meta
  buffer/assign path, not from the split-state computation.

## 5. Recommended plan, in order

1. **Name the tensor first — do not guess-fix.** Add a temporary print at the probe sites, e.g. at
   the `get_usage` caller in `ggml-backend.cpp` (the split-input loop) and at `meta:1249`:
   ```cpp
   if (input->buffer == nullptr) {
       fprintf(stderr, "NULL BUF: op=%s name=%s view_src=%p op_view_src=%d flags=%d\n",
               ggml_op_name(input->op), input->name, (void *) input->view_src,
               input->view_src ? (int) input->view_src->op : -1, (int) input->flags);
   }
   ```
   One run at ub 2048 with `GGML_QSA_DERIVED_VIS=1` tells you whether it is the (never-created)
   mask, a *view* of it, or something else entirely (e.g. a tensor the gallocr pruned that the
   backend still expects). `input->name` is set for llama.cpp graph tensors (`cb(...)` names them,
   e.g. `attn_inp_kq_mask`, `indexer_score`), so the name alone usually identifies the owner.
   Alternative: `backtrace()`/`backtrace_symbols_fd()` right before the assert, or run under
   `gdb -batch -ex run -ex bt` (llama.cpp's built-in crash handler also prints a backtrace — check
   the *whole* log, not just the first lines).
2. **Then either guard or allocate**, per the identity:
   - if it is the mask (or a view of it): guard the probes. A null buffer means "not allocated, not
     a weight" → `get_usage` → `COMPUTE`, `is_host` → `false`, `get_type` → skip the branch. This is
     the upstream-correct direction and makes the engine robust for *any* unread input, not just
     this one. Keep the guards narrow (`buffer == nullptr`) and comment the reasoning.
   - if it is a tensor that *should* exist (e.g. a view whose `view_src` got pruned): that is a
     deeper scheduler issue — prefer shaping the flip so the mask's *view chain* is not built
     either (in `build_attn_qsa` the whole `kq_mask_all`/`zeros`/`set_rows`/`view`/`add` chain is
     already skipped when `qsa_derive_vis` — but check whether *other* code creates views of
     `inp->get_kq_mask()`, e.g. the `kq_mask_swa`/MLA paths in `llama-graph.cpp`).
3. **Re-measure and validate** (the flip is only won when all of these hold):
   ```bash
   GGML_QSA_DERIVED_VIS=1 bash tools/bufsize.sh /tmp/bin-l1g 2048      # 3251.39 / 63.69
   # same-seed coherence, derived vs mask, must be byte-identical:
   for v in 0 1; do GGML_QSA_DERIVED_VIS=$v /tmp/bin-l1g/llama-cli ... -f /tmp/prompt3k.txt  -n 24 ... ; done
   for v in 0 1; do GGML_QSA_DERIVED_VIS=$v /tmp/bin-l1g/llama-cli ... -f /tmp/prompt40k.txt -n 24 ... ; done
   bash tools/mtp-ab.sh l1g                                            # MTP gate (≥0.45, MTP > plain)
   ```
   The text check is *stronger* here than usual: with the mask deleted, identical output **proves**
   the FA's derived path (there is no mask left to read) and that the top-k's derived visibility
   matches, at every depth the prompt covers.
4. **Then bank it**: apply 0003 permanently (fold into 0002 by regenerating), update
   `L1-step1-derived-block-bias-findings.md` §2c with the achieved numbers, run
   `tools/ub-sweep.sh` for ub 2048/1024/512 at the new reserve, and re-check the depth-16384
   decode protocol (`benchmarks/mtp-adaptive-methodology.md`) plus a `pp20480/tg256` parity bench.
5. **Then the next lever**: the score chain still holds ~700 MiB of concat peak (see
   `L2-score-chain-findings.md` §6 and step 2b). After the mask and the bias are gone, the
   remaining QSA-attributable memory is the `score` chain and the top-k's own peak.

## 6. Traps (learned the hard way)

- **Do not "create the mask anyway for its shape"**: `blk_bias` must come from
  `qwen4exp_want_derived_vis(cparams, hparams, n_tokens)` (the model-level predicate: the mask is
  always `[n_kv, n_tps, 1, n_stream]` for this model, always causal, never alibi), and the top-k
  call site must not call `inp->get_kq_mask()` on the derived path. 0003 does exactly this.
- `llm_graph_input_attn_kv::can_reuse` (`llama-graph.cpp` ≈L499) and
  `llm_graph_input_attn_k::can_reuse` (≈L521) dereference `self_kq_mask` → they need
  `self_kq_mask == nullptr ||` guards (0003 has both).
- The **decode** graph legitimately still creates the mask: with `n_tokens == 1`,
  `derived_vis` is false, so `inp->get_kq_mask()` is called and the mask is a harmless
  `[n_kv, 1, 1, n_stream]` = 400 KB. The `llm_graph_input_qsa::can_reuse` gate
  (`derived_vis == want_derived_vis`) forces a rebuild on the prefill↔decode transition, so a
  mask-less prefill graph and a mask-using decode graph never mix. Verify this holds in both
  directions (the MTP probe and a long `-n` run exercise decode; `tools/bufsize.sh` only loads).
- With `-fa 0` or `LLAMA_QSA_SPARSE_FA=0` (dense fallback) the mask *is* needed
  (`kq_mask_top_k`), so `build_attn_qsa` must keep the chain whenever `!qsa_sparse` — 0003's
  condition is `qsa_derive_vis = qsa_sparse && qsa_cell_vis != nullptr`; do not weaken it.
- `NULL` vs `nullptr`: `ggml.c` is C, `ops.cpp`/`qwen4exp.cpp` are C++. A previous attempt failed
  to build on exactly this.
- `patch 0003` deliberately does **not** include the `get_usage` null-tolerance described in §4 —
  apply that as a *diagnostic step* (it moves the failure to the next probe), not as the fix.
- The FA's null-mask tolerance (`maskh` guard + the two launch-argument lists in `fattn-qsa.cu`)
  is **already in 0002** — do not re-add it.

## 7. Regenerating a patch after edits (the worktree recipe)

```bash
cd ~/llama.cpp && R=/home/stew675/llama-cpp-rdna-boosts/wip/qwen4exp/qsa-memory
git worktree add -q --detach /tmp/base HEAD
cd /tmp/base && git apply $R/patches/0001-*.patch && git add -A \
  && git -c user.email=w@x -c user.name=w commit -q -m b1
# for a patch that sits on top of 0002, also: git apply $R/patches/0002-*.patch && commit -q -m b2
cd ~/llama.cpp && for f in $(git status --short | awk '{print $2}'); do \
  mkdir -p /tmp/base/$(dirname $f); cp "$f" "/tmp/base/$f"; done
cd /tmp/base && git diff > /tmp/new.patch && git diff --stat | tail -1
cd ~/llama.cpp && git worktree remove --force /tmp/base && git worktree prune
```
Then verify `git apply --check /tmp/new.patch` on a fresh worktree at the same base before saving it
into `patches/`. (This is how 0002 and 0003 were produced and verified.)

## 8. Related context

- `HANDOVER.md` §1 (environment, model paths, run commands) and §2 (verified findings: the VRAM
  ledger, the prefill speed model, the "everything is fast at depth 0" rule).
- `L1-step1-derived-block-bias-findings.md` — steps 1–3: the derived per-block bias (−400 MiB, §1–2),
  the derived visibility state (§2b), the FA support (§2c), and the **MTP-acceptance drift being a
  buffer-*layout* effect, not arithmetic** (§3 — expect the acceptance to move again when the mask
  disappears; judge by text identity plus the repo's ≥0.45 gate, not by expecting 62/96).
- `L2-score-chain-findings.md` — the 2240 MiB/GPU score-chain cut that is already in the tree, and
  the reason `ggml_cpy`-into-a-view must not be used (allocator `n_views` underflow leak).
- Prompts/scripts: `/tmp/prompt3k.txt`, `/tmp/prompt40k.txt`, `/tmp/ppl6k.txt`, `tools/bufsize.sh`,
  `tools/mtp-ab.sh`, `tools/ab-coherence.sh`, `tools/ub-sweep.sh`, `tools/peak-ledger.py`,
  `/tmp/ggml-alloc.instrumented.c` (for a peak-attribute rebuild if needed).
