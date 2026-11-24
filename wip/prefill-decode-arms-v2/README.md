# prefill-decode-arms **v2** — the two-arm server, rebuilt from the delivery tree

**Status: OPEN (2026-11-23).**  This directory is a **self-contained project**.  It is the only sanctioned
source for the two-arm server (`--prefill-decode-arms`).  Nothing here is in `patches/`, nothing is applied by
`scripts/apply-all.sh`, and nothing is covered by a release gate.  Promotion into the delivery is a **block
decision and belongs to the maintainer** — a green gate ladder is never permission.

| what | value |
|---|---|
| goal | a `llama-server` with a **prefill arm** and a **decode arm** on **one** `llama_memory_i`, so a reply keeps generating while another slot's prompt is being processed |
| work tree | `~/llama.cpp` (the fork checkout), re-cut from the delivery set at `release.json.base` |
| delivery tree | this repo, branch `main` at the commit that added this directory |
| branch | **`prefill-decode-arms-v2`** — created by the first session, never worked on directly in `main` |
| first step | [`PLAN.md`](PLAN.md) STEP 0 |

---

## 0. Read this before reading anything else

**The reading rules are part of the design, not a formality.**  A previous attempt at this feature (a "block 17",
r346-r352) accumulated a design of its own, a set of "hard-won rules", and a plan of follow-up work.  That
material is **wrong at the root** — not merely incomplete — and an agent that reads it tends to reproduce its
assumptions (a second context that *views* and *co-owns* the first context's cells; a device-to-device transfer
of the recurrent state between two memory modules; an allocator handshake bolted onto two objects).  Those
assumptions are exactly what this project deletes.  So:

* **In scope, and all you need:** this directory (`README.md`, `DESIGN.md`, `PLAN.md`),
  the code in `~/llama.cpp`, and the delivery records that describe the *shipping* system
  (`INVARIANTS.md`, `GREEDY-PURITY.md`, `MANIFESTS.md`, `ENVIRONMENT.md`, `scripts/README.md`,
  `benchmarks/mtp-adaptive-methodology.md`, `benchmarks/prefill-logit-methodology.md`).
* **Out of scope — do not read, do not cite, do not "check what the old one did":**
  `archive/work/prefill-decode-arms/`, `archive/work/prefill-decode-arms-followups/`,
  `archive/work/prefill-transfer-overlap/`, `archive/work/build-time-regression/`-style campaign narrative for
  this feature, and the design half of `beta/prefill-decode-arms/README.md` (its "what it contains",
  "hard-won rules", "what it does NOT cover").  `TODO.md` no longer carries any row for this feature, on purpose.
* **One narrow exception:** `beta/prefill-decode-arms/prefill-decode-arms.patch` may be opened for **STEP 1
  only** — three backend/concurrency fixes that v2 intends to port (per-thread host stream, VMM pool peer
  grants, the unconditionally-defined `ggml_cuda_pool_peer_copy_ok`).  Read those three hunks, re-review them as
  standalone fixes on the delivery tree, and close the file.  Do **not** use it as a template for anything in
  STEPs 2-9.  If a step in `PLAN.md` ever seems to need a passage from that patch outside STEP 1, that is a signal
  the step is being misread — stop and re-read `DESIGN.md`.

**If code and this document disagree, the code wins for *facts* and this document wins for *intent*.**  Correct
`DESIGN.md`/`PLAN.md` in the same change when a fact turns out different (that is how this project stays
self-contained).

---

## 1. What the feature is, and why it needs a second context at all

The single-context server multiplexes a prompt and a reply inside **one** step: one batch carrying both the
prompt's chunk and the decoding slots' next tokens, one graph, one compute.  A decoder's inter-token latency is
therefore bounded below by the prefill chunk's compute time (300-880 ms measured).  The two knobs that bound it
instead (`-ib` / `--interactive-batch-size`, `LLAMA_SERVER_PREFILL_DUTY`) both trade prefill throughput for it.

There is no way to give the decoder its own cadence inside one `llama_context`: a context is not reentrant, it
has one batch, one ubatch, one scheduler and one set of compute buffers.  So the decode must run on a
**second `llama_context`**.  That is the whole premise, and it is not up for re-litigation — if a change brings
the two arms back into one shared step it has deleted the feature.

What the second context **must** share is the KV: the reply's positions continue the prompt's, and the two
contexts cannot hold two copies of one sequence.  What it **must not** share is everything else (batch, ubatch,
scheduler, compute buffers, sampler logits, output buffers).  The design problem is exactly the boundary between
those two sentences, and `DESIGN.md` §2 is that boundary.

---

## 2. The one-sentence design change versus the previous attempt

> **One memory object, two contexts.  The prefill arm owns it; the decode arm borrows it.  Nothing is copied
> between them, ever.**

A second context that *views a copy* of the first context's memory is not a simplification, it is a second
memory model that must be kept in agreement with the first: two cell maps over one cell array, two search
cursors, two allocators, two recurrent modules whose state must be transferred byte-for-byte at every request
boundary.  Every defect register entry in `DESIGN.md` §5 comes from that duplication.

---

## 3. Deliverables

| # | deliverable | where |
|---|---|---|
| 1 | **backend prerequisites** (three independent concurrency fixes, valuable on their own) | `~/llama.cpp`, delivered as their own reviewed change first |
| 2 | **`ctx_memory`** — a context that *uses* another context's `llama_memory_i` (a borrow, in `llama_context_params`) | `include/llama.h`, `src/llama-context.cpp`, `src/llama-model.cpp` |
| 3 | **one allocator, one lock** — a per-ubatch reservation bitmap on the *one* `llama_kv_cache`, and a plain `std::mutex` on the memory | `src/llama-kv-cache.*`, `src/llama-memory.h`, `src/llama-context.cpp` |
| 4 | **`llama_memory_recurrent` contextified** — the upstream `TODO: extract all the state like head and n here`.  Per-batch placement moves out of the module into `llama_memory_recurrent_context` | `src/llama-memory-recurrent.*` |
| 5 | **the Meta view slot keyed on the context, not on a role** | `src/llama-context.cpp`, `ggml/src/ggml-backend-meta.cpp` |
| 6 | **the server's two arms** — prefill arm (main loop) + decode arm (own context, own thread), slot ownership, no state handover | `tools/server/server-context.cpp` |
| 7 | **new gates** for the shared-memory model (N:1 ownership, multi-slot soak, two-context recurrent placement) | `wip/prefill-decode-arms-v2/gates/` |
| 8 | a regenerated **beta candidate patch** for external testers (only after the ladder is green) | `beta/` (a *new* directory name; a block decision is still the maintainer's) |
| 9 | the records: `WORKLOG.md` entry per landing, `ENVIRONMENT.md` rows for any knob, `INVARIANTS.md` entries for facts learned | this repo |

**Definition of done** (all of them, on the maintainer's box, one GPU job at a time):

* the delivery default path is **bit-identical**: same-seed greedy text unchanged on 4 shapes, `gate-prefill-logits.sh`
  0.000000 / 100.000 %, and every gate the delivery already owes still green;
* the pair, ON, reproduces the single-context text **byte for byte** on: 4B (1 GPU), phi-4 (1 GPU, dense),
  27B (3 GPU `-sm tensor`), 35B-A3B (`-ncmoe 40`, host tier), qwen4exp/Flash-Next (3 GPU `-sm tensor`, 3 slots,
  `--kv-unified`), Gemma4 (iswa), **with a speculator ON and OFF**, N repeats, one cold prefill per request;
* **4 concurrent slots** (2 prefilling, 2 decoding) × 20 rounds: no GPU fault, no OOM, no text divergence,
  and a non-vacuous census (see `PLAN.md` STEP 8);
* the recurrent state is **never copied**: a source-level check that the deleted symbols do not exist
  (`llama_memory_seq_state_take`, `seq_state_take`, `LLAMA_STATE_SEQ_FLAGS_RECR_ONLY`) plus the two-context
  recurrent-placement gate;
* the decoder's ITL and the prefill cost are re-measured and recorded (the `-21…-24 %` prefill cost of the old
  design is **not** a promise; a different number is a result, not a regression).

---

## 4. Index

| file | what it is |
|---|---|
| `README.md` | this file: scope, reading rules, deliverables, definition of done, environment |
| [`DESIGN.md`](DESIGN.md) | the target architecture, the ownership table, the **defect register V1-V11**, what is deleted, the `llama_memory_recurrent` refactor design, the risks |
| [`PLAN.md`](PLAN.md) | the step-by-step: STEP 0-12, each with its files, edits, oracle and exit criteria; the gate list; the promotion path |
| [`HANDOVER.md`](HANDOVER.md) | **the LIVE handover: one task** -- the real fix for the 2-D state-save read of a split tensor (a beta tester's crash #3).  Start a session here |
| [`HANDOVER-2026-11-24-campaign.md`](HANDOVER-2026-11-24-campaign.md) | the **campaign record** (STEPs 0-14): design decisions, the traps, the per-step reasoning and anchors |
| [`RUNS.md`](RUNS.md) | the dated measurement record: one section per landing, with the command, the shape and the hash/rate |
| [`repro/`](repro/) | the live reproducers behind the three beta-tester crashes (`repro-mtp-prefill.sh` is the one the open defect needs) |
| [`steps/`](steps/) | one reviewable patch per landing (`STEP-<n>-<NNNN>-<name>.patch`, cut from the `arms-v2` tree) |
| [`baseline/`](baseline/) | the STEP 0 anchors (`r353-anchors.md`) and the arm harnesses: `mtp-axis.sh` (one four-axis arm, GPU-locked), `server-arm.sh` (a server arm + request driver), `norm-log.sh` (a log's *shape*, for the line-identical assertions) |

---

## 5. Environment (this box)

```bash
# work tree: one long-lived source path (ccache keys on the absolute path)
cd ~/llama.cpp && git fetch && bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .   # prints the tree

# build (the canonical tree: build-rocm, ROCm 10.1)
~/bin/build-llama-rocm
cmake --build build-rocm --target llama-cli llama-bench llama-server -j 16     # fast loop

# gates / checks that matter here
scripts/validate-set.sh                         # after any patch regeneration
scripts/gate-prefill-logits.sh                  # any prefill/kernel-touching step
scripts/check-shlib-symbols.sh                  # after any guard/#if change
scripts/check-gpu-faults.sh --since <date> --log <log>
```

Hygiene that has bitten this feature before (**mandatory**):

* **pin `HIP_VISIBLE_DEVICES`** on every multi-device arm (`0,2` for the x8 pair, `0,1,2` for all three R9700s) —
  the iGPU is otherwise enumerated and the load aborts with `invalid kernel file`;
* every server run under `setsid` + `timeout`, and kill the **process group** afterwards (after a GPU fault the
  server survives and keeps its port, so the next run fails to bind and looks like a different bug);
* `llama-cli` needs `--single-turn` (`--no-display-prompt` for scripted output); `-lv` is picked per arm — **no
  `-lv` for a text/hash arm**, `-lv 4` for a census arm, never `-lv 3`;
* pin the MoE residency prior (`MOE_EXPERT_HEATMAP=load`) and use `ROCBLAS_USE_HIPBLASLT=0` for any
  fresh-process text comparison;
* **never two GPU jobs at once** — and make it a mechanism, not a habit: `baseline/mtp-axis.sh` takes an
exclusive `flock` on `~/.cache/rdna-boosts/gpu.lock` and refuses after a bounded wait (two concurrent arms
already produced two different "same-seed" texts once, because a near-tie flipped under the interference);
* size `-t` so the GPU IRQ cores stay free;
* a large artifact (a KLD base) goes to `~/.cache/rdna-boosts/gates`, never `/tmp` (tmpfs here).

Models live under `/llm/models/`; resolve paths against that tree, not against an old document.

### Supervising a background job (mandatory — no unbounded waits)

**No background job runs without a bound and a failure detector.**  Every arm in this project is either
synchronous or explicitly supervised, and the supervisor fails fast rather than waiting:

* **every wait is bounded** — readiness 60 s, a request `--max-time`, a hard `timeout` around the whole
  arm (`timeout N …`, and `setsid` + process-group kill for a server); there is no "wait until it
  finishes" path anywhere;
* **a dead process fails the arm immediately** (its log tail is printed), and **a live process that stops
  producing fails it too** — a readiness poll that has not succeeded in its window is a failure, not a
  longer sleep;
* **wait by PID, never `wait` with no arguments**: a bare `wait` in a shell that also backgrounded a
  server waits for *the server*, so an arm that has already finished looks like a hang until the
  server's own timeout expires (this cost a 4-minute idle wait at STEP 0);
* **progress goes to stderr as each stage completes** (`healthy after N s`, `requests done in N s`), so a
  stall is visible in the caller's output instead of silent;
* **a timeout is a result, not a retry trigger**: an arm that times out is investigated (log, GPU fault
  check, `check-gpu-faults.sh --since`) before it is re-run, and never simply re-run with a bigger
  timeout.

---

## 6. Decision authority

* **Block creation, block numbering and promotion are the maintainer's** and are asked first, always
  (`AGENTS.md`).  This project ships a candidate; it does not create a block.
* **A gate that passes is not approval**; a green ladder plus a validated win is the *precondition* for asking.
* Any change that would make the delivery default path differ (a new default, a new reserve, a new env var)
  is reported to the maintainer **before** it is committed, not after.
