# HANDOVER — prefill-decode-arms, toward the server cut

**For a fresh session with no context.**  Read this file, then
[`RESULTS-2026-11-21.md`](RESULTS-2026-11-21.md), then [`README.md`](README.md) (the campaign's design and
landmines), and only then open code.  Everything here is **WIP**: nothing is in `patches/`, no delivery block has
been created or amended, no tag was cut, and `~/llama.cpp` carries the work on the branch `wip/prefill-overlap`
only.

**One paragraph of context.**  `llama-server` runs one `llama_context` on one thread and builds one batch per
step, so a prompt being processed and a reply being generated take turns inside the same graph and the decoder
inherits the whole prefill step as its inter-token latency.  The retired campaign
([`archive/work/prefill-transfer-overlap/`](../../archive/work/prefill-transfer-overlap/README.md)) established
that the fix is **two arms** — a prefill arm and a decode arm, each with its own context and thread, sharing one
KV, with a slot crossing between them only at a request boundary — and that **every host-side policy on top is
neutral or worse** (a per-layer yield hook, a rate gate and stream priority were all built and all lost).  This
campaign builds the product.  Pass 1 built the two enabling changes and the instrument; pass 3 built T0 (one
allocation owner); **pass 5 built the server cut and tier 1 passes**.

## 1. Where the work stands (2026-11-22, pass 5)

| task | state |
|---|---|
| **T1 — the state handover** | **DONE.** `enabling-0002-recurrent-handover.patch`: one sequence's recurrent state moves between the arms **device-to-device in 0.45-0.66 ms** (1.9-2.3 ms on a process's first handover) against **34.5-36.1 ms** for the serialised path (52.7 MiB).  Text-exact on a no-host-tier 4B in every mode/repetition and on the 35B host tier for a single handed-over request (3/3 warm reps) |
| **T2 — the server skeleton** | **DONE 2026-11-22 (pass 5).**  `llama-server --prefill-decode-arms` is the design, not an emulation: a prefill arm (main loop, `ctx_tgt`) + a decode arm (`ctx_dec`, its own thread, co-owning the prefill arm's KV), a slot crossing only at a request boundary.  **Tier 1 passes cold and over 3 warm request pairs** (both slots byte-identical, every allocation handshake clean).  `~/llama.cpp` `683deed67`; built in `T2-SERVER-CUT.md`'s code points + three small deviations; see [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) pass 5 |
| **T3 — the request-boundary handover + its oracle** | **DONE, both tiers (tier 2 2026-11-22, pass 6).**  Tier 1: the server reproduces the reference text exactly, cold and over 3 warm pairs.  Tier 2: `scripts/gate-prefill-logits.sh` on the WIP build reads **mean KLD 0.000000 / same-top-p 100 %** (the artifacts are bit-inert on the default path) -- with the honest limit that `llama-perplexity` has one context and so cannot reach the arms path (pass 6 §2) |
| **T6 — the REVERSE handover (prefix reuse)** | **DONE 2026-11-22 (pass 7), at the maintainer's direction.**  The decode arm no longer frees the slot's shared cells at the return: the main thread takes the recurrent state **back** to the prefill arm (`llama_memory_seq_state_take(mem_tgt, mem_dec, id)`, under the same `arms_quiet` bracket) and keeps the prompt, so a finished conversation stays resident and the next request reuses its prefix.  `~/llama.cpp` `16e942d2f` (1 file, +94/-29).  Measured: `gate-ckpt-reuse.sh` on the arms path goes **SKIP -> PASS** (3 restores, **33 ms** turns instead of 685 ms, i.e. single-context parity); the new `arms/gate-server-reuse.sh` proves a reused prefix **moves no token** (grow, and a fork below the prefix), and its `CONCURRENT=1` arm does it for two slots reusing at once (3.2x / 3.4x on the resumed turns).  Tier 1 unchanged (`459865c93d55cbaf` / `7dc20b0ce0bac255`, cold + 3 warm pairs) |
| **T4 — the opt-in gate** | Not started; its rules are now measured *in the server*: the 2-row decode graph asks **741.48 MiB** and does not fit at arena 5000 next to a grown prefill graph unless **`GGML_CUDA_SLAB_HEADROOM_MIB=1536`** (pass 5 §4) |
| **T5 — measurement + gates** | **The measurement is taken** (pass 5 §3): the smoothness target is met (~47-49 ms p50 through a 2.0-5.3 s prefill), the cost target was **accepted** (README §3b; -24 to -27 %, pass 5 §6 explains why).  **The gates that are reachable are RUN (pass 6, `RESULTS-2026-11-22.md` §3-§7)**: fairness + `--self-test`, duty, park, ckpt, `check-gpu-faults` on 32 logs.  Fairness and duty are red **as shipped** on a *relative* bound that the arms remove (and green under the gate's own absolute / inert-floor criterion); park and ckpt **SKIP by construction**.  Owed before promotion: an arms **arm** in the promise gate's `seq` shape + a server-side pool arm (neither gate reaches this path as shipped -- pass 6 §7b, where the promise gate's own criteria are MET by the oracle's exit census) |
| **T0 — one allocator for the pair** (new, 2026-11-21) | **DONE 2026-11-22 (pass 3).**  [`enabling-0003-shared-kv-allocator.patch`](enabling-0003-shared-kv-allocator.patch) (7 files, +386/-10, `~/llama.cpp` `386b7dd48`): ONE shared owner, **per-ubatch** reserve/commit/release, with `llama_memory_alloc_sync()` as the request-boundary handshake.  `LLAMA_KV_ALLOC_SHARED=0` is the A/B control.  Measured: the two-arm run is **bit-identical to the single-context reference in every rep on both shapes**, no harness control (`RESULTS-2026-11-22.md` pass 3).  **In the server it is called on both arms at every request boundary and asserted clean** (pass 5 §1/§2) |

## 2. What exists as code (do not re-derive any of it)

* **`enabling-0001-second-context-shared-kv.patch`** — `llama_context_params::ctx_other_share_memory` (one bool,
  default false), the generic same-model `ctx_other` gate, `llama_kv_cache::other_coowner` (a second context
  **owns** cells in the shared map; a mirror skips the mutable paths and reads garbage — `!!!!!!!!!!!!` was the
  first build), and the plain-hybrid wiring so `qwen35moe` works.  Text-verified.
* **`enabling-0002-recurrent-handover.patch`** (applies **on top of** 0001) —
  `llama_memory_recurrent::seq_state_take()` (one batched `ggml_backend_tensor_copy_n` over per-layer views),
  `llama_memory_seq_state_take()` (llama level, hybrid memories only), `llama_memory_i::seq_rm_recr()` /
  `llama_memory_seq_rm_recr()` (drop **only** the recurrent cells — `seq_rm()` would free the shared attention
  cells), and `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY` (the serialised control arm; the hybrid wrapper skips its
  attention half on both sides).
* **`arms/two_arms.cpp` + `arms/build-two-arms.sh`** — the instrument: one process, one model, a prefill arm
  (ctx A, `n_batch = n_ubatch = -ub`) and a decode arm (ctx B, `ctx_other = A`, `ctx_other_share_memory`,
  `n_batch = n_ubatch = ARMS_B_BATCH`), a request queue between them, the handover at the request boundary, and
  one token per active slot in one batch.  Modes `single` / `batch-pair` / `arms` / `handover` / `pf-alone` /
  `dec-alone`.  **This is the server shape without the server** — iterate the design here (minutes per arm)
  before touching `tools/server/server-context.cpp`.  Pass 2 added `ARMS_LOGITS_DIR=<dir>` (every sampled row as
  raw f32 + a manifest with each row's sha256/argmax/top-5) — the diff instrument for an arm-vs-arm numeric
  question.
* **The cell censuses** (pass 2, `~/llama.cpp` commit `c49584c60`): `LLAMA_KV_DEBUG_CELLS=1` prints one line per
  ubatch/sequence (cells as runs, `used`, the cursor the scan started from) and `LLAMA_RS_DEBUG_CELLS=1` prints the
  recurrent metadata (`head/n/min/max/rs_z/used` + per cell `pos/src/src0/tail/seq`, and the destination rows right
  after a take).  Both are diagnostic only, env-gated, and cost nothing when off.  `LLAMA_KV_CACHE_DEBUG=3` prints
  the whole cell map, which is too large to diff.
* **THE SERVER CUT (pass 5)** — `tools/server/server-context.cpp` on `wip/prefill-overlap` `683deed67`:
  `llama-server --prefill-decode-arms` (env `LLAMA_ARG_PREFILL_DECODE_ARMS=1`).  `ctx_dec` beside `ctx_tgt`
  (co-owner, decode-width batch), `batch_dec`, `slot.arm` + `m_arms` (one mutex for the transition),
  `hand_over_to_decode()` (post + wait), `decode_arm_loop()` (one row per active slot, one batch),
  `slot_step(ctx, batch, slot, row)` shared by both arms, `finish_returned_slots()`, `arm_release()`,
  `arms_boundary_sync()` (the T0 handshake, asserted clean at every request boundary).  Guards refuse a speculator,
  mtmd/TTS/realtime, embeddings/rerank/decision, LoRA, `--sleep-idle`, backend sampling; `-np 1` warns and stays
  single-context.  **Read `T2-SERVER-CUT.md`'s header and `RESULTS-2026-11-22.md` pass 5 before touching it** — the
  two defects it produced (a self-deadlock on `m_arms`, a `NEXT_RESPONSE` without an id) and the 741 MiB 2-row
  decode-graph OOM (`GGML_CUDA_SLAB_HEADROOM_MIB=1536`) are all recorded there.
* The WIP branch `wip/prefill-overlap` in `~/llama.cpp` (**7 commits**: shared KV, the handover, the two recorded
  losers — the per-gather yield point and stream priority — kept as the record, the censuses, T0, and the server
  cut).  **Rebased onto the delivery's r343** (`e6b3cd3e5`), tip **`9bb97b741`**; `~/llama.cpp` is **clean**; keep it
  that way.  The rebase is what puts the pair on r343's `GGML_CUDA_SLAB_HEADROOM_MIB` default (1536 MiB on one GPU),
  so the **2-row decode graph fits with no env knob** — pass 5's finding #1 is closed by the delivery, not by an
  override (`INVARIANTS.md` r343).
* **`arms/interactive.py`** (pass 5, new): the tool for the question the campaign is really about — *what a user
  sees*.  A streaming reply on slot 0, a long document pasted into slot 1 0.6 s later, both read concurrently; it
  reports the document's TTFT, the reply's gaps **inside** the prefill window, and a per-second table of tokens
  delivered, so a freeze or a burst is visible at a glance.  First run on r343 (arena 5000, `-c 16384 -ub 512`):
  single-context **4.46 s TTFT at 797 t/s, the reply drops to 23-36 tokens/s for 5 s with 300-390 ms stalls**; two
  arms **2.81 s at 1252 t/s, worst gap 95 ms** (`runs/interactive-2611/`) — and at the shape the pair *wants*
  (`-b 4096 -ub 4096`, which r343 makes fit) the contrast is sharper: single-context **878 ms freeze**, two arms
  **60 ms**, document **1131 ms at 3111 t/s**.
* **The ubatch is the knob, `-ib` is not** (pass 5 "The ubatch answer"): at `-ub 512 -> 2048 -> 4096` the pair's
  prefill goes **1056 -> 2455 -> 3246 t/s** (`-26.6 % -> -19.3 % -> -16.4 %` against one context, so close to the
  -14 % prediction) while the decoder's worst in-window gap *falls* (114 -> 66 -> 63 ms).  `decode_active` no longer
  counts a decode-arm slot (that leftover cost 8.1 % at `-ub 512`), so **`-ib` is inert on this path**: run the
  **`-ub`/`-b` 4096** and leave `-ib` unset.  **Correction (the 8192 measurement)**: that "cost falls with the
  ubatch" reading was a **one-step artifact** — at a ~8300-token prompt (several steps at every size) the cost is
  **flat at -21…-24 %** from 2048 up, and the ubatch buys *absolute* prefill rate (`lone` 2955 -> 3526 -> 3800 t/s,
  +29 %), not relative cost.  8192 adds +8 % prefill over 4096 but degrades the decoder's **tail** (p95 63.5 / max
  124.5 ms against 50.2 / 95.1: a 8192-token step is 2.2 s long), so 4096 is the sweet spot.
* The record: `RESULTS-2026-11-21.md` (pass 1: numbers, the three landmines, §3.4's reset experiment, §5's server
  code points, and a "Reproduce" section with the exact commands) and `RESULTS-2026-11-22.md` (pass 2: the residual
  resolved into two effects, the censuses, and the reproduce commands).  Delivery-repo commits `56443c1f`,
  `6e8acb26`, `aa520ebf`, `6592c874`, `a78dd73e`, then pass 2's.

## 3. The state of each open question (the crux of this handover)

### T0 — one allocator for the pair.  **DONE (pass 3).**  Read `RESULTS-2026-11-22.md` pass 3 §1 for the shape that
was built and §2 for the numbers; `~/llama.cpp` `386b7dd48`; the artifact is `enabling-0003-shared-kv-allocator.patch`.
Do not re-derive the scatter: with `LLAMA_KV_ALLOC_SHARED=0` it comes back in one warm rep, which is the control.

Two contexts co-own the KV cells (`v_cells` is shared through `v_cells_impl`) but each has its **own allocation
cursor** (`v_heads`, and its own `n`/`head`).  `llama_kv_cache::find_slot()` is `const` and only *searches* from
that cursor; `apply_ubatch()` advances it afterwards (`head = sinfo.idxs[s].back() + 1`).  Two consequences, both
measured:

1. **A correctness hazard**: between `find_slot()` and `apply_ubatch()` the cells are not reserved, so two
   allocating arms can be handed the *same* cells — fluent, wrong text, non-deterministically.  The measured arms
   did not visibly hit it, but it is unavoidable with two allocating contexts.
2. **The text carrier**: because the cursors drift, a slot lands in different cells on the two arms, and this
   fork's decode numerics depend on that (the tile-FA split is index-dependent).  **Proof**
   (`RESULTS-2026-11-21.md` §3.4): wipe the cell metadata and both cursors at every rep boundary
   (`ARMS_RESET_CACHE=1`, the harness control) and the first request becomes **byte-identical to the
   single-context reference in every rep, on both the host tier and at full residency**; without it, warm reps
   differ from the reference and from each other.

**The shape the maintainer should be asked to approve** (it is a `llama-kv-cache` change, so its own small
artifact — do **not** bury it in the server cut):

* the co-owner pair gets **one** allocation order: **the prefill arm reserves a slot's cell run at the handover
  and publishes it; the decode arm CONSUMES the reservation and never searches or advances the shared cursor**;
* the decode arm then does **no allocation at all** in its hot path — no lock, no `find_slot()` scan (a small
  perf win), and the double-allocation window does not exist;
* the machinery for "run this batch into these cells" already exists: `llama_kv_cache::state_read_sinfo(...,
  sinfos_in)`, the "mirrored slot layout" path the restore/draft code uses.  Reuse that shape rather than
  inventing one;
* **the take is NOT the setup.**  `seq_state_take()` moves the recurrent rows and touches **no** KV cell, so
  moving it to the prefill thread changes nothing here.  What needs one owner is the *allocation*; the decode arm
  must still apply the publish on its own thread, because a take mutates **its** memory module's metadata.
  Publish as a **posted message**, never as a lock the prefill thread waits on — measured: a direct mutex take
  starves the prefill arm for **2.85 s** behind the decoder's step loop (`RESULTS-2026-11-21.md` §2).

### T3 — the oracle.  Pass 2 measured it.

**Read [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) first — it replaces the "residual" story below.**

* **Full residency / no host tier, with `ROCBLAS_USE_HIPBLASLT=0` and ONE allocation state: the strong oracle
  exists.**  `arms` vs `batch-pair` (2 slots, an MoE model, warm reps) read **bit-identically** — 480/480 sampled
  logits rows, `max|Δ| = 0`, both requests, every rep.  Pass 2 got that with `ARMS_RESET_CACHE=1` standing in for
  T0's allocator; once T0 lands the control can be dropped.
* **Pass 3 changed this**: with T0 in, the hash oracle holds **on the product shape too** (arena 5000, both pins,
  3 reps, both requests) — pass 2's "§89 residency class" on the tier was the instrument's take race
  (`ARMS_NO_WAIT_TAKE=1` reproduces it deliberately).  **Tier 1 = the cross-configuration hash on `arms` vs
  `batch-pair`** (fixed shape pins + `ROCBLAS_USE_HIPBLASLT=0` + T0), **tier 2 = KLD + the residency census**,
  kept as insurance for shapes this campaign has not run (the maintainer asked for the greatest coverage).
* **`ROCBLAS_USE_HIPBLASLT=0` is mandatory on EVERY text arm, full residency included.**  The second context's
  first decode can move the hipBLASLt choice **mid-process** (stable afterwards); pass 1's full-residency oracle
  block omitted the pin and that omission manufactured the `b79e73e3fe9f` "residual".
* The earlier (pass 1) reading of that residual — "the second slot's recurrent tail arrives by a take" — is
  **refuted**: the divergent row is R1's own **prefill** last row (`req 1, step 0`), the take is not involved, and
  the recurrent cell index (arms put R1 in cell 0, the single context in cell 1) is not a carrier.

What remains true, and is measured: **the two allocation cursors are a text carrier** (the arms' KV layout is
byte-identical to the reference with one allocation state — 0 of 243 census lines — and scatters without it: 192
of 243, e.g. pos 512 placed in cell 0 while the prompt sits in 512–1023).  That is T0's justification.

### T2 — the server cut.  `RESULTS-2026-11-21.md` §5, seven steps.

`ctx_dec` beside `ctx_tgt` (created with `ctx_other = ctx_tgt`, `ctx_other_share_memory = true`,
`n_batch = n_ubatch = <decode width>`), a second `server_batch`, a `slot.arm` flag + per-slot mutex, the take in
`post_decode()` posted to the decode thread, the return through `llama_memory_seq_rm(ctx_dec)`, the decode step
sharing `post_decode`'s sample/stop/emit logic (factor it to take an explicit context + batch + slot), and guards
that refuse `spec` / mtmd / TTS / embeddings / rerank / decision / LoRA / sleep-wake and `-np 1`.  Main-thread
only: `slots[]`, `batch`, `ctx_tgt`, `spec`, `checkpoints`, the prompt cache.  Before any of it: T0.

### T4 — the opt-in gate.  Two measured rules, not the campaign's "~600 MiB".

* **The second arm's VRAM is a graph reserve per arm AND per active-slot width**: a 2-row decode graph on
  35B-A3B at `-ub 2048` asks for **741.47 MiB** on top of the 1-row one.  At arena 5000 / `-c 16384` /
  `-ub 2048` **the pair does not fit** on a 32 GiB card (the *prefill* graph's reserve fails); `-ub 512` does.
* The window needs a **residency-deficient host-expert tier** (zero at full residency), so the gate is the
  measured deficit, not a flag.

## 4. Landmines and open threads (each cost this session time)

1. **`llama_process()` is asynchronous.**  A prefill "wall" measured without `llama_synchronize()` is an *enqueue*
   time (a warm 512-token prompt reads 3 ms instead of 115 ms).  The instrument syncs inside `ingest_prompt()`;
   keep that.
2. **The take must be POSTED to the decode arm**, never taken under a lock by the prefill thread (2.85 s
   starvation, §3 T0).  Also: only one prompt in flight on the prefill arm, and it waits for the handover to
   complete before starting the next one (the take reads that arm's memory metadata).  **Pass 3: breaking that
   second rule is not subtle — it silently changes the next prompt's recurrent cell index and the reply's text**
   (it was a stale `handover_done` flag in the instrument, and it produced the value pass 1/2 chased as a
   "host-tier" effect).  `ARMS_NO_WAIT_TAKE=1` is the deliberate control arm; do not remove the wait.
3. **`n_parallel` / `n_ctx_seq`**: `n_ctx_seq = n_ctx / n_seq_max`, so raising `n_seq_max` for warm reps shrinks
   each sequence's context and a 4932-token prompt stops fitting.  Reuse the same seq ids across reps instead
   (that is what a server does with slot ids).
4. **A second context changes the graph shapes**, so `-ub`/`-c`/arena combinations that work single-context OOM
   in pairs.  The working two-arm shapes are in `RESULTS-2026-11-21.md` §2.1 (arena 5000, `-c 16384`, `-ub 512`)
   and §3.3 (full residency: no `-ncmoe`, no arena, `-c 8192`).
5. **`seq_state_take()` also sets the destination rows' `src0`** (not just `src`).  `src0` is what the graph
   reads (`llama_memory_recurrent_context::s_copy`) and `find_slot()` leaves it at `rs_z`, which is the cell
   itself only while nothing else names it as a source.  **No observable change on any arm measured so far**, and
   the corresponding change in `state_read_meta()` was **reverted** on purpose (that is the shipped
   prompt-cache/checkpoint path; nothing here can validate it).  Do not "fix" `state_read_meta` without an arm
   that shows the difference.
6. **The text moves with shape**: `LLAMA_KV_N_PAD_MIN=8192` changes even the *single-context* text
   (`b0281be7c5a6` → `649c8596467d`).  Any text comparison must pin the shape (pad, `-c`, `-ub`, cell layout).
7. **Purity pins** on a host-expert arm: `MOE_EXPERT_HEATMAP=load` **and** `ROCBLAS_USE_HIPBLASLT=0` (r310).  The
   verdict for a *prefill-path* change is `scripts/gate-prefill-logits.sh` (KLD), not a hash.  **Pass 2: the
   hipBLASLt pin is mandatory on a no-host-tier text arm too** — the second context's first decode can move the
   library's choice **mid-process** (stable afterwards), which is what produced pass 1's "residual".  A per-process
   text comparison is not a purity guarantee here.
8. **Two DECODE-band contexts on a partial arena corrupt each other** through the per-table, per-process
   `GGML_MOE_DEVREMAP` map (retired campaign, pass 10); a **prefill + decode** pair is safe.  This design must
   never run two decode contexts.

## 5. Mechanics (copy these; they are the ones that worked)

```bash
# the tree and the build (10.1.0 is the line every number in the results file is on)
cd ~/llama.cpp && git checkout wip/prefill-overlap      # 4 commits; must be clean afterwards
BUILD_DIR=build-2ctx JOBS=24 ~/bin/build-llama-rocm-101 # NEVER pass EXTRA_CMAKE_FLAGS with -DCMAKE_HIP_FLAGS

# the instrument (links that build)
cd ~/llama-cpp-rdna-boosts/wip/prefill-decode-arms/arms && ./build-two-arms.sh

# the campaign's 4932-token prompt (tmpfs is wiped on reboot)
head -200 ../../../prompts/kld-corpus-35b-in-domain.txt > /tmp/p4k.txt

export LD_LIBRARY_PATH=/opt/rocm-10.1.0-gfx120X/lib HIP_VISIBLE_DEVICES=0 TWO_CTX_PF_FILE=/tmp/p4k.txt
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf     # the campaign's model
M4=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf                    # the no-host-tier oracle model

# host-tier shape (the product numbers), warm reps
export MOE_EXPERT_CACHE_MIB=5000 MOE_EXPERT_HEATMAP=load ROCBLAS_USE_HIPBLASLT=0 \
       ARMS_SCENARIO="512:192@0,2048:48@0" ARMS_B_BATCH=4 ARMS_REPS=3
for m in single batch-pair pf-alone dec-alone arms; do
  ARMS_MODE=$m ./two_arms -m $M -ngl 99 -ncmoe 40 -lme auto -c 16384 -b 2048 -ub 512 -n 192 -t 8 2>&1 \
    | grep -E "^REP |^REQ |^TOTAL|^ITL"
done

# full residency (no host tier): the oracle shape, and the reset control
# NOTE (2026-11-22): ROCBLAS_USE_HIPBLASLT=0 is REQUIRED here too -- without it the arms' warm text diverges on
# the hipBLASLt path alone (RESULTS-2026-11-22.md §2).
unset MOE_EXPERT_CACHE_MIB MOE_EXPERT_HEATMAP
export ROCBLAS_USE_HIPBLASLT=0
ARMS_MODE=batch-pair ./two_arms -m $M -ngl 99 -c 8192 -b 2048 -ub 512 -n 192 -t 8
ARMS_MODE=arms ARMS_RESET_CACHE=1 ARMS_WAIT_ALL=1 ./two_arms -m $M -ngl 99 -c 8192 -b 2048 -ub 512 -n 192 -t 8

# the handover alone, 3 warm reps (single vs handover must read the same hashes)
ARMS_SCENARIO="4932:32@0" ARMS_MODE=single   ./two_arms -m $M -ngl 99 -ncmoe 40 -lme auto -c 24576 -b 2048 -ub 2048 -n 32 -t 8
ARMS_SCENARIO="4932:32@0" ARMS_MODE=handover ./two_arms -m $M -ngl 99 -ncmoe 40 -lme auto -c 24576 -b 2048 -ub 2048 -n 32 -t 8
# ARMS_HANDOVER=copy   = the serialised control arm of the take
```

Instrument knobs: `ARMS_MODE`, `ARMS_SCENARIO="<prompt_tokens>:<n_predict>@<admit_ms>,…"`, `ARMS_REPS`,
`ARMS_B_BATCH`, `ARMS_WAIT_ALL=1` (no overlap: the step loop holds until every handover is done),
`ARMS_HANDOVER=device|copy`, `ARMS_RESET_CACHE=1` (wipe the cells + both cursors at each rep boundary),
`ARMS_LOGITS_DIR=<dir>` (dump every sampled row as f32 + a sha256/argmax/top-5 manifest — the arm-vs-arm numeric
comparison).  Engine censuses: `LLAMA_KV_DEBUG_CELLS=1` (KV cells per ubatch, as runs) and
`LLAMA_RS_DEBUG_CELLS=1` (recurrent `head/n/rs_z` + per cell `pos/src/src0/tail`, and the destination rows of a
take).  `runs/diag-cells-2611/compare.py` diffs two arms' censuses and manifests; the pass-2 file's Reproduce
section has the exact arm commands.

**Never** run two arms in parallel (the 32 GiB card OOMs at load, and it contaminates every number).  `-lv` is
picked by the arm: a text/hash arm takes **no** `-lv`, a census arm takes **`-lv 4`** (`-lv 3` corrupts the text
*and* hides the ggml censuses).  `llama-cli` needs `--single-turn`; wrap anything that can block in `timeout`.

## 6. Repo rules that will bite you

* `wip/` is **not** the delivery.  Nothing goes into `patches/` without the maintainer's explicit approval, and
  a new block, a renumbering, or promoting an artifact is **always a question, never a gate outcome**
  (`AGENTS.md`, "Report findings before shipping").  A green gate is not approval.
* **Report findings before shipping**: surface review findings, open questions and nits to the maintainer
  *before* any delivery-affecting commit, and let them decide fix-now / defer / drop.
* `~/llama.cpp` must be left **clean** after every session (`git status --short` empty), never push anything from
  it except the release-time `rdna-boosts` refresh to the personal fork, and **never** push to upstream
  `llama.cpp`.  This repo (`llama-cpp-rdna-boosts`) is where the work is committed — commit the campaign record
  frequently so a cold session can pick it up.
* Record a pass as a dated section appended to `RESULTS-2026-11-21.md` (or a newer dated file), with the raw arm
  table, the caveats, and any **refuted** hypothesis marked in place so a later reader is not misled (pass 1's
  §3.3 → §3.4 is the pattern).  Add the `WORKLOG.md` entry (newest first, `YYMMDD.VVV`) for anything
  delivery-affecting, and update `wip/CAMPAIGNS.md`'s row.
* Gates when a change becomes delivery-bound: `scripts/gate-host-pool.sh`, `scripts/gate-moe-promise.sh` (if a
  pass boundary moves), `scripts/gate-prefill-logits.sh` (prefill path), `scripts/gate-prefill-fairness.sh` +
  `scripts/gate-prefill-duty.sh` (admission), `scripts/gate-park-idle-slots.sh`, `scripts/gate-ckpt-reuse.sh`,
  `scripts/check-gpu-faults.sh` on every arm (a two-arm race shows up as a fault, not a wrong token), and
  `scripts/validate-set.sh` when it becomes a patch.

## 7. Read in this order

1. this file;
2. `RESULTS-2026-11-22.md` — **pass 2**: the residual resolved (hipBLASLt at full residency, a §89 residency class
   on the tier), the cell censuses, and the two-tier oracle that follows from them;
3. `RESULTS-2026-11-21.md` — §3.1/§3.4 (the allocator + the reset proof), §2 (the instrument's numbers and the
   ordering lesson), §3.3 (the isolation matrix and the residual — **read its 2026-11-22 notes**), §5 (the server
   cut), "Reproduce";
4. `README.md` — the campaign's design, all the landmines, T0-T5, the gates and the open questions for the
   maintainer;
5. `archive/work/prefill-transfer-overlap/RESULTS-2026-11-20.md` passes 10-13 — the physics and the three
   policies that lost (rate gate, per-gather yield, stream priority); do not rebuild them;
6. `INVARIANTS.md` r305-r310 and `GREEDY-PURITY.md` §44/§89 — the purity traps this campaign keeps meeting.

## 7b. Where the numbers of record stand (end of 2026-11-22, r343)

**T2 is built and its oracle passes**; the campaign's question is answered on both weight tiers.  All numbers: 1 x
R9700 (gfx1201), 35B-A3B Q4_K_M, warm, sequential, `ROCBLAS_USE_HIPBLASLT=0` (and `MOE_EXPERT_HEATMAP=load` on the
host-tier arms).

| | host tier (arena 5000, `-ncmoe 40 -lme auto`) | full residency (no arena, no gathers) |
|---|---|---|
| prefill alone | 6321 -> **3800 t/s** depending on `-ub`/prompt | **6320.8 t/s** |
| **prefill cost of the overlap** | **-21…-24 %** (flat from `-ub 2048` up; `-ub 512` is -26.6 %) | **-24.2 %** |
| single-context server, same run | -34.7…-35 % (chunked) and a 300-880 ms stall | **-34.7 %** and a **327 ms** stall |
| two arms, decoder's worst in-window gap | 60-131 ms | **34.9 ms** |
| the pasted document (interactive) | 2.81 s / 1252 t/s at `-ub 512`; **1.13 s / 3111 t/s at 4096** | **737 ms / 4786 t/s** |

Read it as: **the arms are better than the single-context server on both axes on both tiers**, the prefill cost is
~-21…-24 % (accepted by the maintainer, `README.md` §3b), the decoder's worst case is 34-131 ms instead of 327-878,
and **`-ub`/`-b` 4096 is the sweet spot** (`-ib` inert; 8192 buys +8 % prefill for a worse tail).

**The gates that are reachable on this path are RUN (pass 6, `RESULTS-2026-11-22.md` §3-§7)** — and the short version
is that the delivery's own instruments confirm the campaign rather than contradict it: the single-context
fairness baseline's worst in-window ITL on the gate's own shape is **323 ms**, the arms' baseline is **29 ms**; the
single-context duty baseline delivers **16** decoder tokens through the prefill, the arms' baseline **73**.  Both gates'
remaining reds are *relative* bounds (`chunk/b`, `3x the baseline`) that a design which removes the shared step has no
lever to satisfy.  Park and ckpt **SKIP by construction** (a co-owner pair is not one stream; the arms path has no
cross-request prefix reuse).

**Prefix reuse is fixed (pass 7).**  The reverse handover takes the recurrent state back to the prefill arm at the
return, so a finished conversation stays resident: `gate-ckpt-reuse.sh` on the arms path is **SKIP -> PASS** (3
restores, a turn appending ~9 tokens costs **33 ms, not 685 ms** — single-context parity), and a reused prefix moves
**no token** (`arms/gate-server-reuse.sh`, also with two slots reusing concurrently).

Still open (the traps): the two gates that cannot reach this path as shipped (`gate-host-pool.sh` is llama-cli-only, and
`gate-moe-promise.sh`'s `seq` arm hard-codes `-np 1`) -- though the promise gate's criteria ARE met by the oracle's
exit census, pass 6 §7b: 0 repair lines, `broken=0 repaired=0 stale=3360` cold / `10338` warm over two interleaving
graphs; the T4 gate itself (its rules
are measured: `README.md` landmine 8 + §3b, plus the `-np`/`--kv-unified` interaction of pass 6 §5); what happens to
block 16's duty cycle (measured **inert** on this path, pass 6 §4, while its *fair-share* half stays live); and
promotion (a block needs the maintainer's explicit go-ahead plus the full ladder).

## 8. The one-line summary of what is left

**The server cut is built, both tiers of the oracle pass, prefix reuse is restored, and the smoothness target is met
at an accepted cost.**  What is left is not another mechanism: it is (a) an arms **arm** for the promise gate's `seq`
shape and a server-side pool arm (neither owed gate reaches this path as shipped), plus the *arms* that two existing gates need instead of a flag (fairness' ITL criterion, duty's inertness criterion —
pass 6 §7), (b) T4's opt-in gate — with the
measured VRAM rule (`741.48 MiB` for the 2-row decode graph, `GGML_CUDA_SLAB_HEADROOM_MIB` now a shipped default) and
the `-np`/`--kv-unified` interaction, and (c) the maintainer's decision on what this is worth:
**-24…-27 % prefill for an ITL p95 of 64-78 ms** against the shipped duty cycle's 519-568 ms at -30 %
(`RESULTS-2026-11-22.md` pass 5 §3/§6).  Nothing here is delivery-bound, and a block is a question, never a gate
outcome.
