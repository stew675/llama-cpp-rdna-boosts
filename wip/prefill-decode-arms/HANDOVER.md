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
campaign builds the product.  Pass 1 built the two enabling changes and the instrument; the server cut has not
started.

## 1. Where the work stands (2026-11-21, pass 1)

| task | state |
|---|---|
| **T1 — the state handover** | **DONE.** `enabling-0002-recurrent-handover.patch`: one sequence's recurrent state moves between the arms **device-to-device in 0.45–0.66 ms** (1.9–2.3 ms on a process's first handover) against **34.5–36.1 ms** for the serialised path (52.7 MiB).  Text-exact on a no-host-tier 4B in every mode/repetition and on the 35B host tier for a single handed-over request (3/3 warm reps) |
| **T2 — the server skeleton** | **NOT STARTED.**  Its design is now measured outside the server (the instrument below) and written out in code points: `RESULTS-2026-11-21.md` §5 |
| **T3 — the request-boundary handover + its oracle** | The handover is built and exercised in the instrument; **the oracle is now measured, not hypothesised** — see §3, T3, and [`RESULTS-2026-11-22.md`](RESULTS-2026-11-22.md) |
| **T4 — the opt-in gate** | Not started; its two rules are now measured (VRAM per slot width, the allocator) |
| **T5 — measurement + gates** | Not started (the instrument's numbers are the target T5 must beat) |
| **T0 — one allocator for the pair** (new, 2026-11-21) | **NOT STARTED, and it is the next code task.**  It is a *blocker* (a correctness hazard) and it is also the text carrier of the two-arm shape — `RESULTS-2026-11-21.md` §3.1 + §3.4, and `RESULTS-2026-11-22.md` §4 shows the scatter directly.  **Waiting on the maintainer's approval of the artifact and its shape (HANDOVER §8, Q1).** |

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
* The WIP branch `wip/prefill-overlap` in `~/llama.cpp` (**5 commits**: shared KV, the handover, then the two
  recorded losers — the per-gather yield point and stream priority — kept as the record, then the censuses).
  `~/llama.cpp` is **clean**; keep it that way.
* The record: `RESULTS-2026-11-21.md` (pass 1: numbers, the three landmines, §3.4's reset experiment, §5's server
  code points, and a "Reproduce" section with the exact commands) and `RESULTS-2026-11-22.md` (pass 2: the residual
  resolved into two effects, the censuses, and the reproduce commands).  Delivery-repo commits `56443c1f`,
  `6e8acb26`, `aa520ebf`, `6592c874`, `a78dd73e`, then pass 2's.

## 3. The state of each open question (the crux of this handover)

### T0 — one allocator for the pair.  NEXT, awaiting the maintainer's answer to Q1.  Read `RESULTS-2026-11-21.md`
### §3.1 and §3.4, then `RESULTS-2026-11-22.md` §4 (the scatter, measured).

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
* **On a partial host-expert arena a cross-configuration hash is still not an oracle** — but for the *residency
  class* reason (`GREEDY-PURITY.md` §89, the two runs' censuses differ: `h=0.8161` vs `h=0.8172`), not for the
  cursor reason pass 1 named.  There the verdict is `scripts/gate-prefill-logits.sh` (KLD) plus text compared
  **within one residency class**.
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
   complete before starting the next one (the take reads that arm's memory metadata).
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

## 8. The one-line summary of what is left

**Ask the maintainer two questions (the allocator artifact and its granularity; the T3 oracle, now with the pass-2
measurement in hand), then build the allocator (T0), then cut the server skeleton (T2, `RESULTS-2026-11-21.md` §5)
and measure it with the instrument's own numbers as the target: a decoder getting a token every ~50 ms through a
~2 s prefill for ≤ ~14 % of prefill.**
