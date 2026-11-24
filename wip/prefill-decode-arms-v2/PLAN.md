# PLAN — the step-by-step for `prefill-decode-arms` v2

Read [`README.md`](README.md) §0 first (the reading rules) and [`DESIGN.md`](DESIGN.md) once, end to end.  This
plan assumes both.  It is ordered so that **the tree builds and the delivery's own oracles pass after every
step**, and so that each step is independently reviewable.

| step | what lands | deletes | adds | primary oracle |
|---|---|---|---|---|
| 0 | branch + workspace + baseline anchors | — | — | the delivery's anchors reproduce |
| 1 | three backend concurrency fixes (independent) | — | per-thread host stream, VMM pool peer grants, unconditional `ggml_cuda_pool_peer_copy_ok` | two builds (VMM on/off), `check-shlib-symbols.sh`, text identity |
| 2 | `llama_context_params::ctx_memory` (the borrow) | `ctx_other_share_memory` (never landed in v2) | the borrow + I1/I2 asserts | inert: text + KLD identical, `gate-draft-ctx.sh` |
| 3 | one allocator, one lock | — | the reservation bitmap on the one `llama_kv_cache`; a plain `std::mutex` on the memory | inert single-context; then `gate-arms-n1-memory.sh` |
| 4 | Meta view slot per context | the role-derived slot | `n_view_slots = 8`, per-context base | a 3-context run prints 3 distinct bases; text identical |
| 5 | **`llama_memory_recurrent` contextified** | `mem->n`, `mem->rs_z` as graph-visible state; the `s_copy()` getter mutation | per-batch `placement` in the context | inert single-context (text + MTP four-axis); then `gate-arms-recurrent-shared.sh` |
| 6 | the server's two arms | the whole handover state machine | `ctx_dec` (borrow), the decode thread, slot ownership, the pairing predicate | `gate-arms-n1-memory.sh`, `gate-arms-multislot.sh` |
| 7 | `--spec-type` refused/off on the pair | — | one guard | the guard fires; the pair runs without a drafter |
| 7a | **the host-expert fault** (found by STEP 6's soak) | one narrow slab work base for every context; a process-wide routing flag | one pinned AUX region per extra context + a `thread_local` routing slot (`DESIGN.md` I9) | the host-tier soak PASSes (it faulted), the 4B soak/n1-memory/itl still PASS, A1-A6 unchanged |
| 8 | the gate set + the census | — | 4 gates under `gates/`, `--json` where useful | each gate green **and** non-vacuous |
| 9 | the shape/latency matrix + records | — | `WORKLOG.md`, `ENVIRONMENT.md`, `INVARIANTS.md` rows | the matrix reproduces text on every shape |
| 10 | re-admit the speculator on the pair | the STEP 7 guard | the draft/verify/accept on the decode arm | 3-way cold text + the four axes |
| 11 | the delivery regression sweep | — | — | `validate-set.sh` + the delivery ladder |
| 12 | the candidate patch + the promotion ask | — | `beta/prefill-decode-arms/` (replaces it whole) | the block-17 promotion is the maintainer's decision (never a gate outcome) |

---

## STEP 0 — workspace, branch, baseline

**Goal:** a clean, reproducible starting point, and the anchors that prove every later step is inert when the
option is off.

```bash
cd ~/llama-cpp-rdna-boosts
git switch -c prefill-decode-arms-v2          # from the commit that added this directory
```

Work tree (one long-lived path — ccache keys on it):

```bash
cd ~/llama.cpp && git fetch origin master
git checkout <release.json.base>              # 919794500 at r353
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .    # prints "Applied tip:" and the tree
git switch -c arms-v2                          # the feature branch, on top of the applied set
~/bin/build-llama-rocm
```

**Record the baseline before touching anything** (these become the regression set for every later step):

| # | anchor | how |
|---|---|---|
| A1 | same-seed greedy text, 4B Q8_0, 1 GPU, `-n 64` | `llama-cli … --seed 42 --temp 0 --no-display-prompt --single-turn`, hash with `scripts/extract-generated.py` |
| A2 | same-seed greedy text, 27B Q8_0, 3 GPU `-sm tensor -mg 0`, `-n 96` | same, with `HIP_VISIBLE_DEVICES=0,1,2` |
| A3 | same-seed greedy text, 35B-A3B `-ncmoe 40` (host tier), 1 GPU | `MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0` |
| A4 | prefill-logit base | `scripts/gate-prefill-logits.sh --record` on the *known-good pre-change* build, per ROCm line |
| A5 | MTP four-axis protocol (text + acceptance) | `benchmarks/mtp-adaptive-methodology.md` |
| A6 | single-context server, 2 slots `--kv-unified`, two concurrent greedy completions | the server text hash + the startup log saved verbatim |
| A7 | the delivery ladder's cheap arm | `scripts/validate-set.sh` |

`A6`'s saved startup log matters: STEP 2/3/5/6 each assert the default path prints **exactly** the same lines.

**Exit:** all seven reproduce; the baseline hashes/logs are committed to this directory as
`baseline/r353-anchors.txt` (hashes are records, not secrets).

---

## STEP 1 — the three backend prerequisites (landable on their own)

**Goal:** the concurrency bugs two contexts expose, fixed as *independent* correctness fixes.  They are worth
having even if the feature is abandoned; that is why they go first and alone.

**Read (narrow exception, `README.md` §0):** the relevant hunks of
`beta/prefill-decode-arms/prefill-decode-arms.patch`.  Re-derive each as a fix to the delivery tree; do not copy
the surrounding design.

1. **Per-thread host stream** (`ggml/src/ggml-cuda/ggml-cuda.cu`).  The synchronous host↔device paths
   (`set_tensor`, `get_tensor`, `memset`, the bounce/peer-stage copies, the KV-region row moves) issue on
   `cudaStreamPerThread`.  HIP maps that sentinel onto the **legacy default stream**, and graph capture claims the
   default stream, so a host copy issued by the other context is refused
   (`operation not permitted when stream is capturing`) and every later launch on the device fails.  Fix: one
   private non-blocking stream per (thread, device), cached; every caller already synchronises, so the stream
   choice carries no ordering.  The failure to create it must return `cudaStreamPerThread` **directly** and cache
   that decision (a recursive fallback is a stack overflow, not a fallback).
2. **VMM pool peer grants** (`ggml-cuda.cu`): grant the pool's own device first, then each peer **on its own**,
   keeping the mask of what was granted — the same best-effort rule the movable-boundary slab already follows.
   One refused peer descriptor in an all-descriptors call aborts the process (`HipVMM Failure: invalid argument`)
   exactly when the arena is full, i.e. at request time, i.e. only with a second context.
3. **`ggml_cuda_pool_peer_copy_ok()`** must be **defined unconditionally**: declared in `ggml-cuda-vmm.h` and
   called from `ggml-cuda-slab.cu` (backend-shared), it used to live inside `#if defined(GGML_USE_VMM)`.  Without
   VMM there is no pool, so no pool range can hold the destination and the answer is `true` — the answer the slab
   already gives for a destination outside every range.  (A shared library may carry an undefined symbol; every
   *executable* that links it then fails.  GNU ld.bfd refuses it, lld does not — so a green ladder on this box is
   not the evidence.)

**Do not:** take anything else from that patch in this step.  No meta changes, no memory changes.

**Oracle:** `scripts/check-shlib-symbols.sh` green on the default build **and** on `GGML_HIP_NO_VMM=ON`; both
builds run A1 and read the same text; `scripts/check-gpu-faults.sh` clean on a 3-GPU arm; A4 unchanged.

**Exit:** landed as its own commit, `WORKLOG.md` entry, `INVARIANTS.md` entries for the three facts (the default
stream is captured; a refused peer poisons an all-descriptors grant; a shlib's undefined symbol is a consumer
link break).

---

## STEP 2 — the borrow (`llama_context_params::ctx_memory`)

**Goal:** a context that *uses* another context's `llama_memory_i`.  Inert until the server asks for it.

**Files:** `include/llama.h`, `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-cparams.h`.

**Edits:**

1. `include/llama.h` — `llama_context_params`:
   ```c
   // Use `ctx_memory`'s llama_memory instead of creating a private one: the SAME cells, the same
   // recurrent state, the same indexer cache.  Requires the same model.  DEFAULT null: every existing
   // caller gets its own memory.  This is a MEMORY relationship only -- it does not change what
   // `ctx_other` means for any architecture.
   struct llama_context * ctx_memory;
   ```
   `ctx_other`'s comment block is reverted to its pre-feature text (I8).  `llama_context_default_params()`
   initialises the new field to `nullptr`.
2. `src/llama-cparams.h`: no new field is needed — the borrow is resolved in the constructor.
3. `src/llama-context.cpp`:
   * split the field: `llama_memory_ptr memory_own;` + `llama_memory_i * memory = nullptr;` (so all existing
     `memory->…` uses are untouched), plus `bool memory_borrowed = false;`.
   * constructor, in the `if (!hparams.vocab_only)` block: if `params.ctx_memory != nullptr`, require
     `&params.ctx_memory->get_model() == &model` (throw otherwise), require
     `params.ctx_memory->get_memory() != nullptr`, then `memory = params.ctx_memory->get_memory();
     memory_borrowed = true;` and log INFO with both `this` and the owner.  Otherwise behave exactly as today.
   * destructor: nothing to do for a borrowed memory (I2).  Add a debug-only assert that the owner is still
     alive is not possible portably — instead log at INFO from the borrowed context's destructor and let the
     server's teardown order be the contract (STEP 6).
4. `llama_get_memory()` returns `memory` (unchanged signature).

**Do not:** touch `create_memory()`'s `mem_other` (the MTP mirror path stays exactly as it is), and do not add a
per-module "shared view" concept of any kind.

**Oracle:** A1/A2/A6 unchanged; `gate-prefill-logits.sh` = 0.000000 / 100.000 %; `scripts/gate-draft-ctx.sh`
PASS (an external MTP head still keeps its own cache and does not move the target's text); the default run's
startup log is line-identical to A6's.  Then a 20-line throwaway: create two contexts of the same model, the
second with `ctx_memory` set, and assert `llama_get_memory(a) == llama_get_memory(b)`.

**Exit:** landed; `INVARIANTS.md` entry for I1/I2; the API documented in `include/llama.h` with the "same model"
and "the owner must outlive the borrower" contracts.

---

## STEP 3 — one allocator, one lock

**Goal:** make **one** `llama_memory_i` safe for **two** threads, without a second object anywhere.

**Files:** `src/llama-memory.h`, `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`, `src/llama-context.cpp`,
`src/llama-memory.cpp`, `src/llama-memory-hybrid*.cpp` (only where a lock must be taken).

**Edits:**

1. `llama_memory_i`: replace the shared `std::unique_ptr<std::mutex> mtx_apply` + `apply_mutex_share()` with a
   plain `mutable std::mutex mtx;` and `std::mutex & apply_mutex() const { return mtx; }`.  One object, one lock,
   no plumbing.  (Two uncontended lock operations per ubatch on the single-context path: measured noise, and the
   alternative is a second mechanism.)
2. `llama_kv_cache`: add
   ```cpp
   // I4: cells handed out by find_slot() are RESERVED until apply_ubatch() commits them, because
   // prepare() rolls the cell map back and a second thread's search would otherwise hand out the same
   // cells.  Only maintained once the cache is shared by a second context (single-context runs pay
   // nothing).
   bool shared = false;                       // set by llama_memory_i::notify_borrowed()
   std::vector<std::vector<uint8_t>> reserved; // [stream][cell]
   void reserve(const slot_info &, bool on);
   ```
   * `notify_borrowed()` (on `llama_memory_i`, called by the borrowing `llama_context`) sets `shared = true` and
     lets the composite memories forward it to their children (`llama_memory_hybrid(-idx)` / `kv_cache_iswa`).
   * `find_slot()`: skip a cell with `reserved[strm][idx]`, and at the end mark what it chose and advance
     `v_heads[strm] = idxs.back() + 1`.
   * `apply_ubatch(sinfo, ubatch, trial)`: clear the marks unless `trial` (the `prepare()` dry run).
   * `llama_kv_cache_context`'s destructor: release the marks of ubatches that were never applied.
   * `prepare()`'s failure path: release the marks of the ubatches it had placed.
   * `clear()`: reset the marks.
   * `seq_rm`/`seq_keep`/`seq_cp`/`seq_add`/`state_read_meta`: they already move `v_heads`; nothing else to do —
     **there is no second cursor to mirror** (that is the point).
3. Every `llama_memory_*` entry point and every `state_*` path in `llama-context.cpp` takes
   `memory->apply_mutex()` (the previous attempt's list is the right list; re-derive it from "who mutates the cell
   map" and check you have all of: `init_batch`, `apply`, `clear`, `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`,
   `seq_div`, `seq_park`, `state_write`, `state_read`, `state_seq_write`, `state_seq_read`, `opt_epoch_iter`).

**Do not:** add an env kill switch (V11).  The reservation bitmap is a correctness mechanism, not an option.

**Oracle:** A1/A2/A3/A6 unchanged and the startup log line-identical; `gate-prefill-logits.sh` 0.000000;
`gate-park-idle-slots.sh` PASS (it exercises `seq_park`); `gate-state-pipe.sh` PASS (it exercises the state
lock).  Then the first real two-context arm: two contexts on one memory, one thread per context, one decoding 1
token per step while the other runs a long prompt — text identical to the single-context server's, and the
boundary assert (`reserved == 0`) clean.

**Exit:** `gate-arms-n1-memory.sh` (STEP 8 writes it; a throwaway version here is fine) green, with the cell
census showing **no** duplicate placement and the reservation counts non-zero.

---

## STEP 4 — the Meta view slot keyed on the context (V3)

**Goal:** each `llama_context` gets its own pair of Meta view slots, for as many contexts as exist.

**Files:** `src/llama-context.cpp`, `src/llama-context.h`, `ggml/src/ggml-backend-meta.cpp`,
`ggml/include/ggml-backend.h`.

**Edits:**

1. `ggml-backend-meta.cpp`: `n_view_slots` 2 → **8**; `g_meta_view_slot` stays `thread_local`; the containers and
   the versioned split-state cache stay per slot (that partitioning is correct and must be kept — it is the fix
   for the released build's SIGSEGV); the assert becomes `0 <= slot < 8`.
2. `llama_context`: `const int meta_slot_base = <a process-wide atomic counter × 2, mod 8, assert no overflow>;`
   assigned **once**, at construction, and never derived from `cparams.ctx_other` again.
   `meta_bind_view_slot(alt)` → `set_view_slot(meta_slot_base + (alt ? 1 : 0))`, still called at the top of
   `sched_alt_select()` **and** `graph_reserve()` (a reserve can happen with no alt-state change), on the thread
   that builds the graph.
3. Document the contract: a `llama_context` is never inside two graphs at once, so a context's two slots are
   sufficient; the *slot identity* is what must be unique per context.

**Do not:** keep any role-derived mapping; do not make the slot a function of `ctx_other`, of `is_hybrid`, or of
"owner vs co-owner".

**Oracle:** a 3-context run (target + MTP draft + decode arm) on `-sm tensor` prints three distinct bases at load;
the text is identical to the 2-context arm; `GGML_META_DEBUG_SET=1` shows no `MISSING child` line.

---

## STEP 5 — contextify `llama_memory_recurrent` (the upstream TODO)

**Goal:** make one recurrent module safe to share by two contexts — the piece that makes the whole design
possible (`DESIGN.md` §7).

**Files:** `src/llama-memory-recurrent.h`, `src/llama-memory-recurrent.cpp`.

**Edits (behaviour-preserving — this is a *refactor*, not a redesign):**

1. Add
   ```cpp
   struct placement { uint32_t head = 0; uint32_t n = 0; int32_t rs_z = -1; };
   placement find_slot(const llama_ubatch & ubatch);   // was: bool, and it wrote mem->head/n/rs_z
   ```
   `head` keeps **exactly** today's module semantics as the *scan hint* (`seq_rm`/`seq_keep` pull it back;
   `clear()`/`state_read_meta()` reset it; `find_slot()` sets it to `min`).  It is simply no longer what the
   graph reads.  Keeping the value identical to today is deliberate: it is what makes the single-context path
   bit-identical (I7).
2. Delete the module members `n` and `rs_z` (they are only ever graph-visible).  `is_full` contexts keep
   returning `size` / `0` / `0`.
3. `llama_memory_recurrent_context`: add `placement slot;`; in `apply()` →
   `slot = mem->find_slot(ubatches[i_next]); mem->stage_rs_idx(slot, ubatches[i_next]);`
   `get_head()/get_n_rs()/get_rs_z()` read `slot`; `s_src(i)/s_copy(i)` read `slot.head`.
4. **Move the `rs_idx` reset out of `s_copy()`.**  Today `s_copy()` — a graph-time getter — writes
   `mem->rs_idx[seq] = 0`.  Compute the per-cell rollback index in `stage_rs_idx()` (called from `apply()`, under
   the memory lock) and let the getter *read* it.  This is the same class of bug as V5 and must not be carried
   over: a getter that mutates is a race the moment two graphs can be live.
5. `find_slot()`'s `used = std::count_if(...)` stays (module-wide truth, under the lock).

**Do not:** add a second cursor, a "placement override", or an `other`-style mirror to this module.  Do not change
`n_rs_seq`/`n_rs_batch` semantics or the rollback protocol; only *where the batch placement lives* changes.

**Oracle:**

* (a) **inertness:** A1/A3 unchanged; the MTP four-axis protocol identical on a GDN model (35B-A3B) and on qwen4exp;
  `gate-spec-purity.sh`, `gate-spec-ckpt-device.sh`, `gate-spec-gdn-split.sh` PASS (they all drive the recurrent
  rollback plane).
* (b) **the point:** two contexts on one memory, interleaved batches (one decoding 1 row/step while the other
  prefills a long prompt), text identical to the single-context server; the census (`LLAMA_RS_DEBUG_CELLS`-style,
  one line per context) prints both `(head, n, rs_z)` and they are **different at the same instant** — a
  non-extracted build would print one shared value, which is the non-vacuity check.

**Exit:** landed with the census; `INVARIANTS.md` entry (a graph reads only its memory *context*'s placement);
`gate-arms-recurrent-shared.sh` (STEP 8) green.

---

## STEP 6 — the server's two arms

**Goal:** the feature, with no handover.  This is the biggest step; land it in the sub-steps below, each one
building and each one's oracle passing before the next.

**Files:** `tools/server/server-context.cpp`, `common/common.cpp`, `common/common.h`, `common/arg.cpp`
(the option table), `tools/server/README.md` (generated — see STEP 12).

### 6a — the pairing predicate and the reserve (V2)

* `common/common.h`: `bool common_arms_pair_possible(const common_params &);` — **pure**, no I/O, encoding the
  refusal list once: `n_parallel >= 2`, no mmproj/`mctx`, no TTS/realtime, no embedding/rerank/decision, no LoRA,
  no `--sleep-idle`, no backend sampling, and no drafter whose target side is bound to one context (the
  layer-input families and a gemma4 assistant).  It returns the **reason** as a `const char *` so the server can
  print exactly why a shape cannot carry the pair.
* `common/common.cpp`: the second-context arena reserve is derived **only** when
  `common_arms_pair_possible()` is true.  The size stays a *measured* constant with its measurement printed at
  load (`the decode arm's graph reserve: N MiB, measured at …`), and it must not silently shrink the arena for a
  pair that will not be created.
* `common/arg.cpp`: `--prefill-decode-arms`, `--no-prefill-decode-arms`, `--prefill-decode-arms-mode
  off|on|auto` (default **off**), env forms.  `auto` falls back with a warning; `on` refuses to start.

### 6b — the handover state machine (delete)

Delete from the artifact's design, in the server, whatever STEP 6a-6g would otherwise have inherited: there is
**no** `slot.dec_state`, no take phase, no `take_state_back`, no reverse handover, no `arms_boundary_sync` beyond
the reservation assert.  If you find yourself writing any of them, stop and re-read `DESIGN.md` §1.

### 6c — create the decode arm

```cpp
// same model, same params; the memory is the prefill arm's (I1); no ctx_other (I8)
auto cparams = common_context_params_to_llama(params_base);
n_spec_max  = common_speculative_n_max(&params_base.speculative);   // 0 with no drafter
cparams.n_batch  = params_base.n_parallel * std::max(1, n_spec_max + 1);
cparams.n_ubatch = cparams.n_batch;
cparams.n_outputs_max         = std::max(cparams.n_outputs_max, cparams.n_batch);
cparams.n_outputs_max_per_seq = std::max(cparams.n_outputs_max_per_seq, std::max(1, n_spec_max + 1));
cparams.embeddings = false;
cparams.ctx_memory = ctx_tgt;                 // the borrow -- nothing else
ctx_dec = llama_init_from_model(model_tgt, cparams);
GGML_ASSERT(llama_get_memory(ctx_dec) == llama_get_memory(ctx_tgt));
```
Log one line with both memory pointers and both contexts' widths.  Order of teardown: stop the thread, free
`ctx_dec`, then the model (I2).

### 6d — the decode thread

* `slot_step(ctx, batch, slot, i)` — extract the sampling/emit/stop block out of `post_decode()` so both arms use
  one implementation (a round is not duplicated; the *response lifecycle* stays on the main thread).
* `decode_arm_loop()`: collect the slots it owns, build one batch (`handle_last_sampled_token` per slot), one
  `llama_process(ctx_dec, LLAMA_PROCESS_TYPE_DECODE, …)`, `llama_synchronize`, then `slot_step` per slot (or the
  speculator round, STEP 10).  On a stop: set the slot's arm back and wake the main loop; do **not** touch the
  memory (the sequence stays exactly where it is — that is what makes the next request's prefix reusable).
* The owned-slot batch may not exceed `llama_n_batch(ctx_dec)`; the speculator's width is clamped to `n_spec_max`.

### 6e — the slot ownership flag (state it as invariants)

* `slot.arm` under one mutex, with a condition variable for the handover.  Write the five invariants from
  `DESIGN.md` §8 into the block's header comment, each next to the assertion or census that checks it.
* `post_decode()`: when a slot's prompt row is sampled, hand it over — flag + notify, **no state movement**.
* `pre_decode()`, the batch builder, the `-ib` decode-active test, the speculator loops: skip a slot the arm does
  not own.  These guards are *correct but should be unreachable in practice* because a slot leaves the prefill arm
  the moment its prompt ends — say so in a comment (V9).
* The reverse direction: generation ends on the decode arm → the main thread does `print_timings`,
  `send_final_response`, `release`.  `release()` keeps the prompt, so prefix reuse is free and needs no take.

### 6f — the barrier (V6)

`arm_step_guard(main_arm)`: sets the in-step flag for **every** window in which this arm mutates the shared
memory or drives a decode-band context — `llama_process()` on either arm, the speculative accept round, the draft
call.  A request-boundary operation (the reservation assert, a checkpoint restore) waits for both flags clear.
State the deadlock argument in the comment (both waits under the same mutex, one direction).

### 6g — the speculator on the pair is OFF (STEP 7 owns the guard)

Keep `--spec-type` refused on the pair until STEP 10.  The decode arm's draft path must still be written (STEP 10
turns it on), but it is dead behind the guard, so its first version is reviewed, not exercised.

**Do-not list for STEP 6 (each one is the previous design):** a per-arm memory object; a transfer of the recurrent
state; a per-module "does it implement sharing" concept; a second allocator; merging the two arms' batches; the
decode arm owning a response; a handover that waits for a device copy.

**Oracle:** `gate-arms-n1-memory.sh` and `gate-arms-multislot.sh` (STEP 8) green; then the shape matrix of STEP 9.

---

## STEP 7 — `--spec-type` on the pair: refuse, with the reason

**Goal:** re-admit the drafter onto a foundation that is already sound, not with it.

**Files:** `tools/server/server-context.cpp` (the pairing predicate), `tools/server/README.md` (generated).

* `common_arms_pair_possible()` returns false with a message when `params.speculative.types` is non-empty and the
  mode is `on`; `auto` falls back; `off` refuses like any other unsupported combination.
* The message names the reason and the follow-up ("the decode arm's draft/verify round lands in STEP 10; until
  then run `--spec-type none` or single-context").

**Oracle:** the guard fires for `draft-mtp`, `draft-simple`, the n-gram family and the layer-input families (each
with its own message); the pair with `--spec-type none` is green.

---

## STEP 8 — the gates

Write these under `wip/prefill-decode-arms-v2/gates/` (they move into `scripts/` only at promotion).  Every gate
must be **non-vacuous**: it asserts that the mechanism it depends on actually happened.

| gate | arms | non-vacuity | claim |
|---|---|---|---|
| `gate-arms-default-inert.sh` | option off, 4 shapes | the startup log must be **byte-identical** to the pre-change build's | the default path is untouched (I7) |
| `gate-arms-n1-memory.sh` | 2 slots, one long prefill + one decode, `--kv-unified`; dense + hybrid; `layer` and `tensor` | prints `llama_get_memory(ctx_tgt) == llama_get_memory(ctx_dec)`; per handover INFO lines prove tokens 2..N ran on `ctx_dec`; cell census | text == single-context; no duplicate placement; `reserved == 0` at the boundary |
| `gate-arms-multislot.sh` | **4 slots**, 2 prefilling + 2 decoding, 20 rounds, fresh prompts, one cold prefill per request; dense, hybrid, host tier, qwen4exp; spec off | census: `n_dec_steps > 0`, every slot's handover count > 0, VRAM/RSS printed before/after | text per slot == single-context; no GPU fault (`check-gpu-faults.sh --since`); no OOM; boundary asserts clean; the server survives the whole run |
| `gate-arms-recurrent-shared.sh` | two contexts, one recurrent memory, interleaved batches (1 row/step vs a long prefill) on 35B-A3B and qwen4exp | the two contexts' `(head, n, rs_z)` census values must **differ at the same instant** | text identical to single-context; the MTP four-axis protocol identical |
| `gate-arms-itl.sh` | a decoder's ITL through a 2-5 s prefill (`-c 8192 -ub 4096 -np 2`) | the prefill's own step time is printed, and the decoder's worst gap must be *far* below it | the feature's whole point: p95 ITL bounded; a merged-step regression shows the old 300-880 ms stall |

Rules: one GPU job at a time; every server run under `setsid` + `timeout`, process group killed; every log checked
with `check-gpu-faults.sh --since <before the run>`; text arms take **no** `-lv`, census arms `-lv 4`.

**Written (2026-11-23):** all five exist and are green.  The two this step owed:

* `gates/gate-arms-default-inert.sh` — the shapes, their cli text anchors (with each row's **provenance**) and
  their startup shapes live in `baseline/inert-anchors.tsv` and `baseline/inert-shapes/`; the 4B startup shape's
  expectation IS `baseline/A6-startup-shape.txt` (the STEP 0 file, not a copy), and the gate also runs the same
  server with the pair ON to prove the comparison can see a change (4 added lines, same responses).  `--record`
  writes only a MISSING anchor, so an anchor can never be quietly re-recorded into agreement.
* `gates/gate-arms-recurrent-shared.sh` — `probe-recurrent-shared.cpp` promoted: §1 the placements must DIFFER,
  §2 concurrent == `--solo` byte for byte, §3 the four-axis `none` == adaptive **with a non-zero draft/accept
  count** (the non-vacuity line that caught an adaptive drafter refusing to initialise at `n_max 2` and the run
  silently continuing plain -- which was the HARNESS's invalid flag pair, not the feature; the gate now uses a
  valid ceiling and FAILs on 0 drafts).  Two models, both on a **3-device `-sm tensor -mg 0`** split: 35B-A3B and
  Flash-Next/qwen4exp + its external shared MTP head (`--axis-shape host` for the delivery's host-expert arm).

**Exit:** all five green, and each gate's non-vacuity line present in its output.  Record the numbers in this
directory (`RUNS.md`), not in `WORKLOG.md` (that gets the landing entry).  **Met 2026-11-23.**

---

## STEP 9 — the shape/latency matrix and the records

Run, on the maintainer's box, one arm at a time:

* **text identity** (pair ON vs pair OFF, one fresh process per arm, N = 3 repeats): 4B Q8_0 (1 GPU);
  phi-4 IQ4_XS (dense, 1 GPU); 27B Q8_0 (3 GPU `-sm tensor`); 35B-A3B Q8_0 (`-ncmoe 40`, host tier);
  Flash-Next IQ4_NL (3 GPU `-sm tensor`, 3 slots `--kv-unified`, 2 long prefills + 1 decoding);
  Gemma4 (iswa).  Record the hash per shape.
* **ITL and prefill cost** (`gate-arms-itl.sh` plus the prefill rate at `-ub` 512/2048/4096): the *previous*
  design's −21…−24 % is not a promise.  A different number is a result.
* **memory**: the reserve actually taken, the arena's free VRAM before/after the decode context, and the VRAM
  delta per extra slot.  This is the item to report to the maintainer **before** the patch is cut.
* records: `WORKLOG.md` entry per landing; `ENVIRONMENT.md` rows for every knob (`--prefill-decode-arms`,
  `--prefill-decode-arms-mode`, and nothing else unless a gate needs it); `INVARIANTS.md` entries for the facts
  (I1-I8 candidates, the measurement that set the reserve, the Meta slot identity).

**Done 2026-11-23** with `baseline/shape-matrix.sh` (+ `server-arm.sh --vram-probe`), 3 repeats per arm:
7/7 shapes text-identical (the seventh is the designed refusal for a drafter-bound shape), the cost is
+53..+154 MiB of VRAM (+149/154 per device on the tensor splits) plus one 1280 MiB slab aux region on a host
tier, and the trade is ITL −22.8 % at `-ub 2048` / −23.9 % at `-ub 4096` with prefill **+24.1 % / +19.4 %**
(the win grows with the chunk width; the prefill is faster because the decoder's row leaves the prefill's batch).
`ENVIRONMENT.md` §24 carries the rows (marked wip / NOT shipped).  Two corrections came out of it: **STEP 7f** (a
borrower's `cparams.type_k`/`type_v` were uninitialized -- Flash-Next + the pair aborted 2 of 3 runs) and
`gate-arms-itl.sh` no longer makes a byte claim in the one configuration where a byte claim is a lottery.

**The tensor-split rows (added after the maintainer asked for them):** Flash-Next 3 devices `-sm tensor`,
`-ub 4096`, **53.83 -> 22.89 ms/token (ITL −57.5 %)** with the pair's graph capture off, prefill +21.8 %; 27B 3
devices `-sm tensor`, `-ub 2048`, 64.45 -> 60.30 (−6.4 %, prefill +35.6 %).  Measuring them found **STEP 7g** (the
Meta split-state cache was per buffer, not per view slot -- two arms raced it) and left **one blocking decision**
(`HANDOVER-2026-11-24-campaign.md` §7.7): with hipBLASLt AND capture on, a tensor-split pair arm aborts
(`'operation would make the legacy stream depend on a capturing blocking stream'`), so the pair must pin one of
capture-off (fast) or hipBLASLt-off (1.8x slower) until a capture barrier exists.

---

## STEP 10 — re-admit the speculator on the pair  — **DONE 2026-11-24** (`f3e34859d`)

> Landed as `steps/STEP-10-0018-speculator-on-the-pair.patch` (+ 10a, `e7ae0b470`).  The scope below was met; the
> four things it did NOT anticipate are in `RUNS.md` §6 and were fixed in the same landing: the drafter's target
> rows must follow the arm that ran the batch (`common_batch::ctx`), an embedded-MTP draft context keeps no
> `ctx_other` and therefore had no pinned slab region (`llama_context_params::ctx_aux`), a compute-buffer
> allocation and a cross-stream event wait may not overlap the other arm's graph capture, and the slab's aux
> regions needed the floor's slot off-by-one fixed + growth-when-idle + a mapping for a region stacked under
> another.  Gate: `gates/gate-arms-spec.sh`.  Open follow-ups: `HANDOVER-2026-11-24-campaign.md` §7.2/§7.8/§7.9.

Only after STEPs 4-9 are green.

* The decode arm drives the draft/verify/accept round for the slots it owns; `common_batch` carries its `ctx` and
  the drafter reads the target's hidden rows from the context that **ran** the batch (register `ctx_dec` as a
  target context).
* Both arms' speculator entry points take the one speculator mutex (V4); the draft context's own
  `llama_process()` is decode-band, so it runs inside the barrier (rule 3 of `DESIGN.md` §8).
* The verify width is `n_parallel × (n_max + 1)`; a draft wider than `n_spec_max` is clamped, never asserted at
  run time.
* A slot carries **no** speculative state across a handover (`spec_draft`, `spec_i_batch`, `spec_ckpt`,
  `spec_is_replay` all describe a batch the other arm ran): reset them on the handover, in both directions.
* Leave refused, with a message: the layer-input drafters and a gemma4 assistant.

**Oracle:** a **three-way cold** text claim — `pair+draft-mtp` == `pair+none` == `single+draft-mtp` — one fresh
process per arm, one cold prefill per request, on 3 devices f16 / 3 devices q8_0 / 2 devices f16; the MTP
four-axis protocol on the pair and off; `LLAMA_SERVER_PARK_IDLE_SLOTS=0` pinned in every arm (idle-slot parking
changes cell placement and therefore can flip a near-tie — the *parking policy* must be the same on both sides of
a text comparison, always).

---

## STEP 11 — the delivery regression sweep

* `scripts/validate-set.sh` (checksums, a fresh-tarball strict apply, the file-set reconciliation).
* the delivery's own gates that the touched files owe (the memory layer, the state paths, the park path, the
  speculator, the Meta buffer): at minimum `gate-prefill-logits.sh`, `gate-state-pipe.sh`,
  `gate-park-idle-slots.sh`, `gate-kv-residency.sh`, `gate-host-pool.sh`, `gate-spec-purity.sh`,
  `gate-mtp-dev-rows.sh`, `gate-moe-promise.sh`, `gate-node-budget.sh`, `gate-graph-replay-width.sh`.
* `scripts/check-gen-docs.sh` (the pair adds options → the three generated documents must be regenerated),
  `scripts/check-windows-portability.sh` and `scripts/check-windows-paths.sh` (server code is compiled on Windows).
* one whole-set `all` build at the tip + `check-gpu-faults.sh` over every server log of the run.

---

## STEP 12 — the candidate and the ask

* Regenerate the patch **from a canonical chain**, never from the working branch:
  `scripts/make-patches.sh` on a tree built by `scripts/apply-all.sh` at `release.json.base`.
* Put it in **`beta/prefill-decode-arms/`**, REPLACING that artifact in its entirety (the maintainer's decision,
  2026-11-24): the old directory's design text, README and patch are deleted, not kept beside it, because this is a
  different design and must not inherit its readers' expectations.  `README.md` (what it is, the applied tree, the
  shapes it reaches and the ones it refuses, the measured numbers) and `BETA-TESTING.md` (the ladder a tester
  runs), and it stays **ONE self-contained patch** — on promotion it is **delivery block 17**, a modification to
  the lower-numbered blocks, cut against the release tree (`release.json`), never against the working branch.
* Report to the maintainer: the measured ITL/prefill trade, the memory cost, the shapes validated and not, and the
  open items.  **Promotion into `patches/` is a new block and is the maintainer's decision** — ask, do not cut.

---

## Failure handling

* **A step's oracle fails** → do not proceed and do not "fix it forward".  The step list is ordered so the failure
  is locally attributable: A1-A6 decide whether the *default path* moved; the pair's gates decide the feature.
  A default-path move is a bug in the step, never a new anchor to record.
* **The build breaks** → `scripts/gate-per-block-build.sh` exists for exactly one job: localising which *block*
  broke a build (`--changed-from`).  It is not a release gate and not a per-step requirement.
* **Text diverges with the pair on** → first check the **cells** (`LLAMA_KV_DEBUG_CELLS`): identical cells + a
  different text is a shape/kernel question (V8); different cells is a memory/locking question (I3/I4).
* **A GPU fault** → `rm -f gpucore.*.gpu`, kill the process group, and read the log with
  `check-gpu-faults.sh --since <before> --log <log>`; never grep for `page fault` by hand (the echoed prompt
  lines match).
* **Two arms starve each other** → measure before changing: the memory mutex is held for µs at a ubatch boundary;
  if a step takes longer than that, the lock is in the wrong place (or a compute is inside it — the previous
  design measured a 2.85 s starvation that way).

## Where the numbers go

`RUNS.md` in this directory (dated, per gate, with the command line and the shape) while the campaign is open;
`WORKLOG.md` gets the landing entries; `benchmarks/` gets anything that outlives the campaign; `INVARIANTS.md`
gets the facts.  **A gate's output is never the record of a claim** — the claim lives in the record with its
hash/rate and its shape.

## STEP 15 — never issue a 2-D device->host copy into host memory — **DONE 2026-11-24** (`67946d6f0`)

> Landed as `steps/STEP-15-0022-no-2d-d2h.patch` from a beta tester's third crash.  The work order
> (`HANDOVER.md`) suspected the Meta backend's split arithmetic; measurement refuted that (per-device chunks
> 331092 / 331092 / 331096 sum exactly to the meta row, and device 0's copy of the same arguments succeeded) and the
> driver log named the real cause: a 2-D D2H copy needs a peer grant on the destination's pinned allocation
> (`hsa_amd_agents_allow_access ... err 4096`), and this box refuses those grants.  The CUDA backend now serves every
> row-block read as 1-D copies (`ggml_cuda_rows_to_host`), device-staged when it may synchronize; both 2-D read entry
> points use it; the async read also selects its device (its write twin already did) and the staged arm can no longer
> gather a row larger than the staging chunk.  Records: `WORKLOG.md` 261124.007, `INVARIANTS.md` r384-wip, `RUNS.md`
> (2026-11-24, STEP 15).  Promotion obligation unchanged: cut into the delivery as a **block-09 amendment**
> (`TODO` #376).

## STEP 16 — the pair keeps the compute-buffer drop, fenced by the other arm's step boundary — **DONE 2026-11-24** (`56fdcbaff`)

> Landed as `steps/STEP-16-0023-pair-keeps-the-drop.patch` from the maintainer's report that uncontested decode is
> ~half speed with the pair on.  It was not the pair: disabling the drop keeps the slab's wide band armed, and
> ggml-cuda refuses to capture a graph while the band is armed, so every decode step ran direct (45.9 t/s against
> 83.2 on the single-context server, and 46.5 with only `LLAMA_DROP_COMPUTE_BUFFERS=0` on that same server).  The
> drop now runs, fenced by the OTHER arm's step window through `llama_context_set_drop_boundary()` — the capture
> barrier alone does not cover a replay, which is one `cudaGraphLaunch` whose GEMMs never take it.  Pair decode
> 88.4 t/s, capture restored (3140), arena back +5.6 GiB.  Records: `WORKLOG.md` 261124.009, `INVARIANTS.md`
> r385-wip, `RUNS.md` (2026-11-24 STEP 16).  Delivery-level: at promotion it is a second block-09 amendment
> (`TODO` #376).
