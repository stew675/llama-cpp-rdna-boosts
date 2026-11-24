# DESIGN — one memory, two contexts

This is the architecture of `prefill-decode-arms` **v2**.  It is self-contained: it is meant to be read with
`~/llama.cpp` open, and it deliberately does not cite the previous attempt's documents (`README.md` §0).

---

## 1. The target in one picture

```
                 main thread                                    decode thread
        ┌────────────────────────────────┐            ┌────────────────────────────────┐
        │ llama_context  ctx_tgt         │            │ llama_context  ctx_dec         │
        │   n_batch/n_ubatch: -b / -ub   │            │   n_batch = n_ubatch =         │
        │   sampler, logits, batch,      │            │       -np x (n_max+1)          │
        │   scheduler, compute buffers   │            │   (same as above, its own)     │
        └───────────────┬────────────────┘            └───────────────┬────────────────┘
                        │  OWNS                                       │  BORROWS
                        └──────────────►  llama_memory_i  ◄───────────┘
                                          (ONE object: cells, tensors,
                                           one search cursor, one
                                           allocator, one mutex)
```

* Both contexts are **of the same model** and run **the same memory module** — literally the same
  `llama_memory_i` pointer, so "shares the KV", "shares the recurrent state" and "shares the indexer cache" are
  all true by construction, for every architecture, including ones that do not exist today.
* Everything above the memory is **per context** and never shared: batch, ubatch, scheduler, graph result,
  compute buffers, logits, sampler, output rows.
* There is **no transfer of state between the arms, ever**.  A slot's sequence lives in the one memory; an arm is
  just the execution context that is currently evaluating batches for it.

---

## 2. The responsibility distinction (this is the project)

The previous attempt gave the decode arm its **own** `llama_memory_i` that *viewed* the prefill arm's cells.  That
reads like sharing and behaves like two memory models, and the difference is the whole defect register:

| responsibility | with two memory objects (previous) | with one memory object (v2) |
|---|---|---|
| the cell array | shared via `v_cells_impl` (`llama_kv_cache`), one per module | the object's own cells |
| the search cursor | two per-context cursors that must be kept in agreement → a shared `llama_kv_alloc::heads` **plus** the legacy `v_heads` kept as a "mirror" | the object's `v_heads`/`head` |
| cell reservation | an out-of-band bitmap per (stream, cell) + `alloc_head_pull/copy/set` at five sites + `alloc_sync()` on every module + an env kill switch | one bitmap, one place to commit it |
| the recurrent state | **two** `llama_memory_recurrent` modules; the state is *copied* between them at each boundary (batched D2D, serialised fallback, per-child Meta copies, an indexer-pool invalidation hook) | the object's cells and tensors — nothing to copy |
| the memory taxonomy | every module must implement a shared *view*; a module that does not silently leaves the decode arm on an empty cache (`implements no shared view` was a load-bearing log line) | every module works, because there is only one module |
| the lock | a `shared_ptr<mutex>` shared by hand (`apply_mutex_share`), threaded through `create_memory` | one mutex, a plain member |
| the relationship | overloaded onto `ctx_other` (which already means "MTP target", "gemma4 assistant view", "eagle3 token_embd donor") + a second bool to disambiguate | one field with one meaning: `llama_context_params::ctx_memory` |
| a slot's arm | a state machine with a handover: `dec_posted`, `dec_state`, the take phase, the reverse take, the per-request-boundary synchronisation | a flag: which thread is allowed to look at the slot |

**Rule of thumb for every review of this project:** if a change adds a place where the two arms must *agree* about
the memory, it is the previous design leaking back in.  The two arms may disagree about everything except the
memory object's own invariants, which its mutex protects.

---

## 3. What is shared, what is not (the boundary table)

| thing | shared? | why |
|---|---|---|
| `llama_memory_i` (cells, K/V tensors, recurrent tensors, indexer cache, meta buffers) | **YES — one object** | one sequence, one set of positions; a copy is a second truth |
| the memory's mutex | yes (it is the object's) | two threads call `init_batch()`/`apply()` on it |
| the model weights / backends | yes (pre-existing) | unchanged |
| `llama_context` itself | **NO** | not reentrant: one batch, one ubatch, one scheduler, one compute-buffer set |
| `n_batch` / `n_ubatch` | no | the prefill arm is width-optimised, the decode arm is row-count-optimised |
| sampler, logits, `output_reserve`, `n_outputs_max` | no | each arm samples from its own context |
| speculator (`common_speculative`), draft context, checkpoints, prompt cache | no ownership change; **serialised** (one DECODE-band context at a time) | see §8 |
| slot state (`server_slot`) | yes, with an **ownership flag** | tokens must be emitted promptly, so the decode thread touches the slot it owns |

---

## 4. Invariants (I1-I8) — every step of `PLAN.md` must keep all of these

**I1 — one memory object per model-context group.**  `llama_get_memory(ctx_dec) == llama_get_memory(ctx_tgt)`.
Asserted at construction; the gate prints both pointers and requires equality.

**I2 — a context that borrows a memory must not free it.**  The borrower's destructor releases *nothing*; the
owner's destructor runs last.  `llama_free()` order therefore matters and is asserted (INFO with both pointers,
`GGML_ASSERT` on the ordering in debug).

**I3 — the memory's bookkeeping mutates under one lock, and only the memory mutates it.**  `init_batch()`,
`apply()`, every `llama_memory_*` op and every `state_*` op take the memory's own mutex.  No per-context shadow
state may be consulted to decide a placement.

**I4 — a ubatch's cells are reserved before they are applied.**  `find_slot()` marks them, `apply()` clears the
marks, a batch context destroyed before applying them releases them.  This is the *only* mechanism that stops one
arm's `find_slot()` from handing out cells another arm has prepared but not yet applied, and it is required
exactly because `prepare()` rolls the cell map back.

**I5 — the graph is built only from per-context state.**  Anything a graph builder reads about *this batch*
(`head`, `n`, `rs_z`, the ubatch split, `n_kv`) must live in the batch's `llama_memory_context_i`, not in the
module.  Otherwise two arms' graph builds read each other's placement.  (`llama_kv_cache` already satisfies this;
`llama_memory_recurrent` does not — §7.)

**I6 — nothing is copied between the arms.**  No code may move a sequence's KV, recurrent rows or indexer keys
from one place to another as part of a request boundary.  A "take" is not a simplification of a copy, it *is* the
copy.

**I7 — the delivery default path is bit-identical.**  With the option off, the server allocates, reserves and logs
what it does today; every change that touches a shared code path is gated on the pair actually running, or is
provably inert (same-seed text + KLD 0.000000).

**I8 — `ctx_other` keeps exactly its three existing meanings.**  MTP/draft target, gemma4 assistant view, eagle3
token_embd donor.  The pair uses a **new** field and must not be visible to those code paths.

**I9 — two live compute buffers never share a slab work base, and "which context is allocating" is never a
process-wide flag.**  Everything *above* the memory is per context, and one of those things is the scheduler's
COMPUTE buffer — the slab's work region is process-wide, so it must hand each live view its own region.  It had
exactly one "narrow" (decode/verify) base for the target plus one pinned `narrow2` region for the MTP draft, so a
**third** context (the pair's decode arm) aliased the prefill arm's 1280 MiB view at offset 0 and the GPU faulted
in the K rope's `SET_ROWS` (`HANDOVER-2026-11-24-campaign.md` §2, `INVARIANTS.md` r364-wip).  Each extra context now gets its own
pinned region, selected by its **own identity** (`ctx_other` → the draft slot, `memory_borrowed` → the borrowing
slot) through a `thread_local`, because two arms reserve on two threads and a process-wide flag leaks.  This is
the same rule as I3/I5 one level up: no shared bookkeeping, no role-derived selection, one owner per resource.

---

## 5. Defect register V1-V11

Each entry is: **what it is**, **why it exists**, **what retires it**.  V1-V10 are properties of the *previous
design*; v2 retires them structurally rather than by fixing them one at a time.  V11 is a process rule.

### V1 — a state transfer that fails is installed as success

The forward handover logged its outcome and then marked the slot as "state is on the decode arm" **regardless**.
The transfer API answers `false` both for "nothing to move" (a dense model — the right answer) and for "the move
failed" (a hybrid — garbage from an empty recurrent state), so a caller must disambiguate by model class, and one
of the two callers did not.  *Retired by:* deleting the transfer (I6).  *Residual rule:* any future "move
something per slot" API must return a value that cannot be confused with a no-op.

### V2 — the second context's arena reserve is charged for a pair that is never created

The reserve was computed from `params` alone while the decision to create the second context consulted a list of
refusals (multimodal, TTS/realtime, embeddings/rerank/decision, LoRA, sleep-idle, backend sampling, five drafter
families).  A refused pair — or `auto` falling back — left the expert arena several GiB smaller for nothing.
*Retired by:* one **pure predicate** `common_arms_pair_possible(params)` that both the reserve derivation and the
server call, so they cannot disagree (`PLAN.md` STEP 6).

### V3 — the Meta view slot is keyed on a role, not on a context

The meta backend keeps, per (buffer, view slot), a rotating pair of simple-tensor containers and a versioned
split-state cache; a graph must be built *and* resolved under the same slot, and the selector is `thread_local`.
The slot was derived as `(ctx_other != nullptr ? 2 : 0) + alt` — a **role**.  In a pair with a drafter there are
**three** contexts and only two roles, so the decode context and the draft context share slot 2/3.  They happen
to be used sequentially on one thread today; the container is per buffer, not per thread, so that is luck.
*Retired by:* a per-context slot base from a process-wide counter, `n_view_slots = 8` (`PLAN.md` STEP 4).

### V4 — the speculator lock was one-sided

The decode arm took the speculator lock for its draft/verify/accept round; the prefill arm's draft path (the
single-context code) was not under it, although the design document claimed both were.  It was masked by the fact
that the pair hands every prompt-finished slot to the decode arm in the same step, so the prefill arm's
"drafting" set was always empty — an undocumented scheduling invariant.  *Retired by:* the lock on both paths
(`PLAN.md` STEP 6), and by the DECODE-band-context rule in §8.

### V5 — two recurrent modules, one copied state

A copy that is not byte-faithful is *fluent but wrong* text, so it is invisible to a text-length check and visible
only to a KLD point or a byte diff.  The previous design needed: a batched device copy, a serialised fallback, a
capability probe per buffer type, a per-child Meta copy written specifically for it, a state flag
(`RECR_ONLY`), a `seq_rm_recr()` special case, and an invalidation hook for qwen4exp's cell-derived indexer pool.
Each is a place to be wrong.  *Retired by:* one recurrent module (I1, I6) — and the upstream refactor that makes
one module safe to share (§7).

### V6 — the "neither arm may be in a step" barrier covered only the compute

The barrier set its in-step flag around `llama_process()` and nothing else, while the state take and the
speculative accept round — both of which mutate shared state — ran outside it.  Correctness rested on the main
thread happening to sequence those calls.  *Retired by:* the barrier covers **every** shared-memory mutation
(`PLAN.md` STEP 6), and by the fact that with one memory the only such mutations are `init_batch`/`apply`/the
memory ops, all of which also take the object's mutex (I3).

### V7 — the module taxonomy made "shares the KV" an assumption

A second memory object has to be *taught* to view the first, per module, and a module that was not taught left the
decode arm on an empty cache while the log said the KV was shared (the silent-garbage P0: a long prompt answered
as if only its tail were present).  *Retired by:* one module (I1).  This is the single most valuable property of
the design: **the set of memory modules that work is "all of them", including future ones.**

### V8 — the decode context is a different graph shape, so text identity is empirical

The decode context is narrow (`-np × (n_max+1)` rows) where the single-context server's decode row shares a
batch with prompt rows.  Every index- or shape-keyed decision in this fork (the tile-FA decode split, KV padding,
the MTP draft ring, band predicates) is therefore evaluated at a different shape, so "the pair reads the same text
as the single context" is a per-shape measurement, not a structural guarantee.  *Not retired by v2* — it is
inherent to two contexts, and it is *smaller* here (the decode arm is the only shape, uniformly).  *Mitigation:*
the text oracle runs a matrix of shapes, and a divergence is a kernel question, not a memory question.  Keep the
`LLAMA_KV_DEBUG_CELLS` cell census available for exactly this: a divergent text with identical cells is a
numerics/shape issue; different cells is a memory issue.

### V9 — one-thread-at-a-time rules resting on unstated scheduling invariants

The pair's safety depended on properties nobody wrote down: that the prefill arm never holds a generating slot,
that every posted handover is waited for before the next prompt, that a cancelled slot is only released after the
decode arm hands it back.  Each is a real property of today's control flow and none is enforced where a reader
would look.  *Retired by:* stating them in the server block's header comment as invariants with the assertion or
census that checks them (`PLAN.md` STEP 6), and by the slot ownership flag being the *only* way one arm learns
about the other.

### V10 — `ctx_other` overloaded by a fourth meaning

`ctx_other` already means three unrelated things.  Reusing it for "co-owns my memory" forced a disambiguating
bool, a cross-model refusal, and an "is this a draft head or a co-owner" branch in the constructor — all of it
irrelevant to the pair.  *Retired by:* `llama_context_params::ctx_memory` (I8).

### V11 — a kill switch with no gate

The shared-allocator A/B (`LLAMA_KV_ALLOC_SHARED`) had no arm that exercised it, and its "control" behaviour
(duplicate cell placement) is a silent wrong answer rather than a text difference.  *Retired by:* deleting it.
**Policy for v2:** a new env var must be (a) a kill switch for a *shipped* default, with a gate arm that flips it,
or (b) absent.  No knobs for the design that is being replaced.

---

## 6. What is deleted, and what takes its place

**Deleted (the previous artifact's contribution, ~1 500 lines):**

| region | lines (approx) |
|---|---|
| `struct llama_kv_alloc` + `alloc_release/head_pull/head_copy/head_set/alloc_sync` + `LLAMA_KV_ALLOC_SHARED` + the `prepare()` rollback bookkeeping | 200 |
| `ctx_other_share_memory` + `ctx_other_share` + `other_coowner` + `is_mirror()` + the `mem_other` threading through `create_memory` and six modules + the "implements no shared view" warning | 200 |
| `llama_memory_recurrent::seq_state_take()` + `can_take_batched()` + `llama_memory_seq_state_take()` + the `LLAMA_RS_*` diagnostics | 350 |
| the Meta buffer's per-child `cpy_tensor_n` + `copy_plan`/`copy_root_range` (written only for that take) | 350 |
| `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY`, `seq_rm_recr()`, `seq_state_take_note()`, the hybrid `state_write/read` guards | 80 |
| the server's handover state machine: `dec_state`, the take phase, `take_state_back`, the reverse handover, `arms_boundary_sync` | 200 |
| `apply_mutex_share()` / the shared `unique_ptr<mutex>` | 40 |

**Added:**

| region | why |
|---|---|
| `llama_context_params::ctx_memory` + the borrow in `llama_context` (§7 preceding) | the one relationship the pair needs |
| `common_arms_pair_possible(params)` (§5 V2) | one decision, two callers |
| a per-context Meta view slot base | V3 |
| the reservation bitmap, unconditional-per-shared-cache, no env | I4 |
| `llama_memory_recurrent` placement extraction | I5 (the upstream TODO) |
| the server's arm ownership, stated as invariants | V9 |

**Kept from the previous artifact (STEP 1 of `PLAN.md`, ported as standalone fixes):** the per-thread host stream
in `ggml-cuda.cu`, the VMM pool's per-descriptor peer grants + a pool range registry, and the unconditionally
defined `ggml_cuda_pool_peer_copy_ok()` (a `GGML_HIP_NO_VMM=ON` link break otherwise).  Two concurrent contexts
need all three, and each is defensible without the feature.

---

## 7. The missing piece: `llama_memory_recurrent` must be contextified

This is the upstream `TODO` in `src/llama-memory-recurrent.h`, and it is the reason a single shared memory object
was never the obvious answer: `llama_kv_cache` already keeps per-batch state in its
`llama_memory_context_i`, but `llama_memory_recurrent` does not.

**Today the module holds graph-visible per-batch state:**

| member | written by | read by (graph build / inputs) |
|---|---|---|
| `head` | `find_slot()` (`head = min`) | `context::get_head()`, `s_src(i)`, `s_copy(i)` — and it doubles as the **scan hint** the next `find_slot()` starts from |
| `n` | `find_slot()` (`n = max-min+1`) | `context::get_n_rs()` |
| `rs_z` | `find_slot()` | `context::get_rs_z()` |
| `cells[i].src0` | `find_slot()` | `s_src(i)` / `s_copy(i)` (per cell, so per sequence — safe) |
| `rs_idx[seq]` | `find_slot()`/`set_rs_idx()` **and `s_copy()`, which resets it inside a getter** | `s_copy()` |

With two contexts building graphs at the same time, the first four are read while the other arm writes them, and
the last is a *mutation inside a graph-time getter*.

**The refactor (behaviour-preserving, ~200 lines):**

1. `llama_memory_recurrent::find_slot()` returns a placement instead of writing the module:

   ```cpp
   struct placement { uint32_t head; uint32_t n; int32_t rs_z; };
   placement find_slot(const llama_ubatch & ubatch);   // returns {} / n == 0 on failure
   ```

   `head` keeps **exactly** today's module semantics as the **scan hint** (`seq_rm`/`seq_keep` pull it back,
   `clear()`/`state_read_meta()` reset it, `find_slot()` sets it to `min`); it is simply no longer what the graph
   reads.  Keeping the value identical is deliberate — it makes the single-context path bit-identical (I7).
2. `n` and `rs_z` become **context-only**; delete the module members (the `is_full` context keeps returning
   `size`/`0`/`0`).
3. `llama_memory_recurrent_context` stores `placement slot;` and:
   * `apply()`: `slot = mem->find_slot(ubatches[i_next]); mem->apply_rs_idx(slot /* + ubatch */);`
   * `get_head()/get_n_rs()/get_rs_z()` read `slot`;
   * `s_src(i)`/`s_copy(i)` read `slot.head` and take the rollback index from a **context-local** value computed
     in `apply()` (`rs_idx` reset moves out of the getter).
4. `set_rs_idx()`/`rs_idx` stay module state (it is a property of the sequence, not of a batch), but it is
   *written* only in `apply()` and in the state paths, i.e. under the memory lock.

**Oracle.** (a) Single context on a hybrid: same-seed text identical and the MTP four-axis protocol identical
(this proves the refactor is inert).  (b) Two contexts on one memory, interleaved batches: text identical to the
single context, with a census that prints both contexts' `(head, n, rs_z)` and shows them **different** at the
same moment — a build that had not extracted them would print one shared value, so the census is the
non-vacuity check.

**This is the step that makes the whole design possible.**  Do not skip it, and do not "work around it" by giving
the recurrent module a second cursor.

---

## 8. The server in v2

The control flow is the same shape as any two-thread server: the **main loop is the prefill arm**, a **decode
thread is the decode arm**, and a slot belongs to exactly one arm at a time.

```
request arrives ──► prefill arm: prompt ubatches on ctx_tgt (its own batch, its own sampler)
                        │  last prompt row evaluated + sampled ON ctx_tgt (the logits are there)
                        ▼
                    hand the slot to the decode arm  (flag + notify; NO state movement)
                        ▼
                    decode arm: one row per owned slot, one batch, one llama_process(ctx_dec),
                                sample, emit, stop conditions; the speculator round lives here too
                        ▼
                    generation ends → flag back to the prefill arm, which owns
                    print_timings / send_final_response / release / prompt reuse
```

**Invariants to write down in the block header (V9):**

1. A slot is in exactly one arm's `batch` at a time; the flag is the only cross-thread slot state and it is
   written only under the arms mutex.
2. The prefill arm never batches a slot it does not own, and the decode arm never touches a slot it does not own.
3. Only **one** context may be inside `llama_process()` in the *decode band* at a time (the MoE device remap
   table is process-wide).  The prefill arm's wide pass may overlap a decode step, which is the point of the
   feature; the **speculator's draft context is decode-band**, so drafting is serialised with the decode step --
   enforced by `arms_spec_mtx`, which the decode arm holds across its own `llama_process(ctx_dec)` as well.
   *Corrected 2026-11-23 (`HANDOVER-2026-11-24-campaign.md` §2):* the pair's host-expert GPU fault was **not** this invariant being
   violated — it was two contexts' COMPUTE buffers sharing one slab work base (I9).  The remap rule above still
   stands on its own (it is why a decode step and a prefill pass may overlap but two decode-band contexts may
   not), but a fault on a host-expert shape is not evidence of it; check `GGML_CUDA_WORK_ALLOC_DEBUG=1` first.
4. Any request-boundary memory operation that is not `init_batch`/`apply` (an allocation assert, a checkpoint
   restore) runs only when neither arm is inside `llama_process()`.
5. The response lifecycle (`print_timings`, `send_final_response`, `release`, the prompt cache) is **main-thread
   only**.  The decode arm may `process_token()`/stream (it must, for latency), and nothing else.

**The speculator** (STEP 10, landed 2026-11-24).  The decode arm drives draft/verify/accept for the slots it owns,
the prefill arm's ingest is bracketed by the same lock, and **both** arms' speculator entry points take the one
speculator mutex (V4).  Three facts the implementation settled, each measured:

* **the drafter's target-side reads follow the arm that RAN the batch** (`common_batch::ctx`): `ctx_dec` on the
  decode arm, `ctx_tgt` on the prefill arm, with the h_nextn extraction enabled on both.  Reading `ctx_tgt` from
  the decode arm would verify against the prefill arm's last pass -- garbage drafts, every one rejected, and the
  TEXT unchanged (which is why the gate asserts drafts were ACCEPTED, not just produced);
* the speculator lock is also the DECODE-BAND lock: the decode arm holds it across its own `llama_process(ctx_dec)`,
  so the draft context (decode-band, rule 3) can never run beside a decode step.  It is recursive because a round
  can end by printing the statistics line through `print_timings()`.  The prefill arm's WIDE pass does not take it;
* a slot's speculative state (`spec_draft`/`spec_i_batch`/`spec_ckpt`/`spec_is_replay`) names a batch the OTHER arm
  could not restore, so it is reset at every handover, with an assert that nothing is pending.  A speculating slot
  is handed over only after its accept round RESOLVED -- a replay round (a context that cannot roll back in place)
  keeps its draft and must stay where its checkpoint came from.

A drafter whose target side is bound to one context (the layer-input families: eagle3/dflash/dspark/dflash2/
lilicorr) or that shares the target's KV (a gemma4 assistant) stays **refused** with a message: with one shared
memory their assumption ("I am the only other user") is not just wrong, it is unreachable.  The family refusal is
a property of `common_params`; the gemma4 one needs the draft GGUF's `general.architecture`, which the predicate
reads (one metadata key) so that the arena reserve and the pair decision still cannot disagree (V2).  The
admitted set is `draft-simple`, `draft-mtp`, `draft-mtp-adaptive` and the model-free n-gram family.  `--spec-type` on the pair is **off by default and guarded separately**
(STEP 7) until the shared-memory ladder is green; that ordering is deliberate — the speculator is what made the
old design's concurrency unexplainable, and it must be re-admitted onto a foundation that is already sound.

**What the arms must NOT do:** merge their batches (that is the shared step the feature removes); move any state
between contexts; keep a per-arm shadow of the memory's bookkeeping; or let the decode arm own a slot's response
lifecycle.

---

## 9. What this design does not change

* the single-context server, when the option is off, in any observable way (I7);
* the delivery's blocks 0000-0016 and 0300 — the pair is additive code, and the only shared-file edits are the
  three backend fixes (which are correct independent of the pair) and the borrow plumbing;
* the KV placement policy when the pair is on: with one cursor and one allocator, a sequence's cells are a
  function of the request sequence, exactly as for one context;
* park/idle-slot behaviour, prompt cache, checkpoints, `-ib`/duty semantics (they remain inert on the pair, and
  say so at startup).

---

## 10. Risks and unknowns (retire each one explicitly)

| # | risk | how it is retired |
|---|---|---|
| R1 | a borrow makes a use-after-free possible if `llama_free(ctx_tgt)` precedes `ctx_dec` | I2 + a debug assert + the server's `destroy()` order (decode arm stopped and freed first) |
| R2 | two arms in `init_batch` deadlock or livelock on the memory mutex | the mutex is held only for bookkeeping (µs); the gate runs 4 slots × 20 rounds with a latency bound on the boundary |
| R3 | the reservation bitmap leaks a mark (a batch context destroyed in an unexpected order) | an assert at every request boundary (`reserved == 0`), and the counts are in the census |
| R4 | the recurrent extraction changes single-context numerics subtly | text + KLD oracles in the same step, before the pair uses it (STEP 5) |
| R5 | the decode arm's narrow graph picks different kernels than the single context's decode row (V8) | the shape matrix in the text oracle; a divergence is investigated as a kernel/shape question with the cell census |
| R6 | the arena reserve for the second context is still a hand-tuned constant | keep the constant small and *derived from a measurement*, I3-style: print the measured reserve at load, and fail loudly rather than silently shrinking the expert arena.  **Re-measure at STEP 10:** on a slab-armed shape the decode arm's *compute* buffer is 1280 MiB, pinned out of the arena (1.25 GiB rounded to a chunk), not the 157-235 MiB the 27B/3-GPU tensor arm measured — the constant covers the arena shrink of one narrow context and a drafter would make it two (`HANDOVER-2026-11-24-campaign.md` §7 item 2) |
| R8 | a third (or later) context aliases an extra context's slab region | I9: every extra context has its own pinned region, chosen by its own identity; `gates/gate-arms-multislot.sh` FAILS if the slab is armed and the decode arm got no region of its own |
| R7 | a "helpful" refactor re-merges the arms to save the prefill cost | `DESIGN.md` §1/§8 and the gate that measures ITL through a prefill: a merged step shows the 300-880 ms stall and fails |
