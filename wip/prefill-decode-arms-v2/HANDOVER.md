# HANDOVER — `prefill-decode-arms` v2, campaign state + the immediate task

**Read this file first, then work top to bottom.**  It is written for a session that has *no* memory of the
campaign: it says what exists, what is green, what is open, and what to do first.  Everything below was measured
on this box; nothing here is a plan that has not been run unless it says so.

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`prefill-decode-arms-v2`** (pushed; `main` untouched at `91a81505`) |
| work tree | `~/llama.cpp`, branch **`arms-v2`** (local only, **never pushed**), tip `5e668c768` |
| build | `~/bin/build-llama-rocm` → `build-rocm` (ROCm **10.2.0**); fast loop `cmake --build build-rocm --target llama-cli llama-server -j 16` |
| last landing | `604e21a8` (delivery) / `5e668c768` (`~/llama.cpp`) — STEP 6e |
| the one open defect | **the pair FAULTS a host-expert (`-ncmoe`) shape under load** — `-ncmoe > 0` is therefore *refused* by the predicate; making it safe is TASK 1 below |
| next action | TASK 1 §2.3, experiment 1 (`GGML_CUDA_QK_PAIR_FWHT=0`) |

---

## 0. Reading rules (inherited from `README.md` §0 — they are part of the design)

**In scope:** this directory (`README.md`, `DESIGN.md`, `PLAN.md`, `HANDOVER.md`, `RUNS.md`), the code in
`~/llama.cpp`, and the delivery records that describe the *shipping* system (`INVARIANTS.md`,
`GREEDY-PURITY.md`, `MANIFESTS.md`, `ENVIRONMENT.md`, `scripts/README.md`, the two methodology files in
`benchmarks/`).

**DO NOT READ, for any reason, "to see what the old one did":**
`archive/work/prefill-decode-arms/`, `archive/work/prefill-decode-arms-followups/` (including its
`DESIGN-REVIEW-2026-11-23.md`), `archive/work/prefill-transfer-overlap/`, and the design / "hard-won rules"
halves of `beta/prefill-decode-arms/README.md`.  That material is **wrong at the root** — a second context that
*views* and *co-owns* the first context's memory, with the recurrent state copied between two modules — and an
agent that reads it rebuilds those assumptions.  One narrow exception, already used and closed: the three
backend hunks of `beta/prefill-decode-arms/prefill-decode-arms.patch` were read for STEP 1 only.

If a step ever seems to need that material, you are misreading the step: re-read `DESIGN.md` §2 (the two memory
models, side by side) and §5 (the defect register V1-V11 — every entry there is a defect v2 deletes structurally).

---

## 1. Where the campaign stands

### 1.1 The design in one sentence

**ONE `llama_memory_i`, TWO `llama_context`s: the prefill arm OWNS the memory, the decode arm BORROWS it, and
nothing is copied between them.**  `llama_context_params::ctx_memory` is the only relationship the pair needs.

### 1.2 Landed (each is a separate, reviewable patch in `steps/`, each with its oracle run)

| landing | patch | what it is |
|---|---|---|
| STEP 0 | — | baseline anchors (`baseline/r353-anchors.md`), the harnesses, the supervision contract |
| STEP 1 | `STEP-1-0001-backend-concurrency.patch` | three independent backend fixes: a private per-(thread,device) host stream (`ggml_cuda_host_stream()`, incl. `batch-copy.cu` and the async-input ring), best-effort VMM pool peer grants + a range registry, and `ggml_cuda_pool_peer_copy_ok()` defined unconditionally |
| STEP 2 | `STEP-2-0002-ctx-memory-borrow.patch` | `llama_context_params::ctx_memory`: the borrow; `memory_own` + `memory` split; I1/I2 |
| STEP 3 | `STEP-3-0003-one-allocator-one-lock.patch` | the **reservation bitmap** (I4: a prepared batch's cells stay reserved until applied) + **one plain mutex** on the memory (I3) |
| STEP 4 | `STEP-4-0004-meta-view-slot-identity.patch` | the Meta backend view slot is a per-**context** identity (8 slots, a pair each), selector `thread_local` |
| STEP 5 | `STEP-5-0005-recurrent-placement.patch` | `llama_memory_recurrent` contextified: per-batch `placement` moves into the context; the `rs_idx` reset leaves the graph-time getter |
| STEP 6a/6c | `STEP-6a-0006-pair-predicate-and-context.patch` | `common_arms_pair_possible()` (pure, returns the reason) + `--prefill-decode-arms[-mode off\|on\|auto]` + the decode arm's context (a borrow, inert) |
| STEP 6d-1 | `STEP-6d-0007-slot-step-extraction.patch` | `slot_step()` extracted from `post_decode()` — ONE sample/emit/stop implementation for both arms |
| STEP 6d-2/6e/6f | `STEP-6d2-0008-decode-arm-routing.patch` | the decode thread, `slot.arm` ownership, the handover, the step barrier, the five invariants |
| STEP 6e | `STEP-6e-0009-reserve-and-census.patch` | the decode arm's arena reserve (256 MiB, measured), the reservation census on stderr, the `-ncmoe` refusal (TASK 1's subject) |

### 1.3 Green, with the number that proves it

| claim | evidence |
|---|---|
| default path inert (pair OFF) | A1 4B `be904e86f248`, A2 27B-3GPU-tensor `10a82b5559a9`, A3 35B-A3B-host `e9db5cb8db0b`, A4 prefill-logit **KLD 0.000000 / top-p 100.000 %**, A6 server `{7bd542418112, 6ce1f88dd5f7}` + a **13-line startup shape byte-identical** to `baseline/A6-startup-shape.txt` |
| pair ON == pair OFF, one request in flight | 4B: `7bd542418112`/`6ce1f88dd5f7`; 27B-3GPU-tensor: `f2753e13e419`/`c5f2fdc43642` |
| the decode arm really runs | census `the decode arm is generating: N step(s), M handover(s), widest batch W slot(s)` — e.g. 47/2/2 on the 4B, 4 handovers and **widest batch 4** in the soak |
| the pair's win (latency) | ITL through a 5246-token prefill, `-ub 2048`: **20.27 → 15.68 ms/token (−23 %)**, prefill **+25 %** (4935 → 6147 t/s), text identical.  **Flat at `-ub 512`** (13.25 vs 13.16) — the win needs the wide chunk |
| STEP 3's I4 bitmap | `gates/probe-kv-reservation.cpp`: borrowed A `[0..511]`, B `[512..1023]`, C `[1024..1535]`; the STEP-2 control build gives `[0..511]` twice |
| STEP 5's extraction | `gates/probe-recurrent-shared.cpp`: two **live** contexts `A head=0 n=1` vs `B head=1 n=1`; the STEP-4 control gives `head=1 n=1` for both |
| the gates | `gates/gate-arms-n1-memory.sh` **PASS**, `gates/gate-arms-itl.sh` **PASS**, `gates/gate-arms-multislot.sh` **PASS on a GPU-resident shape** (4B, 4 slots × 20 rounds, 80 requests, 0 errors) |

### 1.4 Open (this is what you are here for)

* **TASK 1 (blocking further validation): the pair faults a host-expert shape.** §2.
* Not yet done from STEP 6: nothing — but the host-expert half of the shape matrix is blocked by TASK 1.
* STEP 7 (speculator refused on the pair) is **already in place** inside `common_arms_pair_possible()`
  (`params.speculative.has_aux()` → refuse), so STEP 7 is done and its re-admission is STEP 10.
* STEP 8's gate set: three of five exist (§5).  Missing: `gate-arms-default-inert.sh` and
  `gate-arms-recurrent-shared.sh` (the latter's oracle exists as `gates/probe-recurrent-shared.cpp`).
* STEPs 9-12: see §3.

---

## 2. TASK 1 — make the host-tier expert remap context-safe

### 2.1 The evidence (do not re-derive it; re-run it if you doubt it)

`gates/gate-arms-multislot.sh` with the host-tier defaults, **np 4, 20 rounds, 80 requests, all of which
completed**, then:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address
  ... code: 0x29 [host: soar, GPU index: 0,
  kernel: void rms_norm_mul_rope_f32_qk_pair<false, float, __half, 0, 0>(rmr_args<float>, rmr_args<__half>)]
```

and the server aborted ~398 s in.  The same gate on the **4B, GPU-resident** shape passes (80 requests, 0
errors, widest decode batch 4), and the same 35B-A3B host-tier shape passes the whole delivery ladder with the
pair **OFF** (`gate-park-idle-slots.sh`, `gate-spec-*`, the four-axis protocol).  So the fault is
**pair × host-tier**.

Current mitigation (temporary, in `common/common.cpp`): `-ncmoe > 0` is refused with that reason, so nothing
regresses while you work.

### 2.2 Repro (exact)

```bash
cd ~/llama-cpp-rdna-boosts
# the fault needs the load, not a specific prompt: 4 slots, 20 rounds, 2 prefills + 2 decodes per round
HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0 timeout 2400 \
  wip/prefill-decode-arms-v2/gates/gate-arms-multislot.sh --keep        # host-tier defaults: 35B-A3B -ncmoe 40, np 4
# gates/gate-arms-multislot.sh currently REFUSES to start because of the predicate -- to reproduce, either
# remove the refusal for the experiment or run the server by hand with the same args:
HIP_VISIBLE_DEVICES=0 setsid ~/llama.cpp/build-rocm/bin/llama-server \
  -m /llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf \
  -np 4 -c 16384 --kv-unified -ngl 99 -t 16 --no-webui -lv 4 -ncmoe 40 --prefill-decode-arms
```

Then check with **`scripts/check-gpu-faults.sh --since "<before the run>" --log <log>`** — never grep for
`page fault` by hand.  Remove any `gpucore.*.gpu` (32 GB each) before moving on.

### 2.3 The experiment ladder (cheapest discriminator first — do them in this order)

Each rung is one soak run with an env var; stop at the first rung that stops the fault, because that names the
mechanism.

1. **`GGML_CUDA_QK_PAIR_FWHT=0`** — the faulting kernel is reached only through the **deferred Q/K "pair"
   fusion**.  That fusion is the one place in block 08 where state is carried **across graph nodes**
   (`cuda_ctx->qk_pend_graph` / `qk_pend_j`, `ggml-cuda-try-fuse.cu` ~3290-3400, `ggml-cuda-qk-pair` /
   `ggml_cuda_qk_pair_k_at`): the Q half is *deferred* to node `j` and both halves run in ONE launch.  With two
   contexts dispatching two graphs at the same time this deferral is the first thing to suspect, and the switch
   turns it off.  **If this rung is clean, the defect is the deferral, not the expert remap** — read
   `ggml_cuda_should_fuse_rms_norm_mul_rope_fwht()` and decide what makes a deferred fusion safe under two
   contexts (the honest answers: bind the pending state to the *graph object* AND assert it cannot outlive the
   dispatch, or refuse the deferral when a second context is live).
2. **`GGML_CUDA_MOE_HOST_OFFLOAD=0`** (and/or `MOE_EXPERT_CACHE_MIB=0`) — is the fault the host tier at all, or
   any MoE on the pair?  If `MOE_EXPERT_CACHE_MIB=0` is clean, the arena/remap path is implicated and the kernel
   name is a red herring.
3. **np=2 instead of 4, 1 round** — does it need two decoders, or one decoder plus a prefill?
4. **The same shape with the pair OFF but `-ncmoe 40` and np 4** — the control (expected clean; the park gate
   already covers a 3-slot host-tier shape).
5. **`LLAMA_KV_DEBUG_CELLS=1`** on a near-faulting arm — the cell census.  A fault with an identical cell map
   says "not placement"; a different cell map says the two arms disagreed about cells (I4/I5 territory).

### 2.4 The code map (where to look, in the order the evidence points)

| what | where |
|---|---|
| the faulting kernel + its arg struct | `ggml/src/ggml-cuda/rope.cu`: `struct rmr_args` (~1097), `rms_norm_mul_rope_f32_qk_pair` (~1113) |
| the **deferred** Q/K pair fusion (per-context state across nodes) | `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`: `ggml_cuda_qk_pair_k_at` (~1109), the two `qk_pend_graph` arm/match sites (~3290, ~3367, ~3397); `cuda_ctx->qk_pend_graph/j` live in `ggml_backend_cuda_context` |
| the host-expert cache: tables, residency, the **slot remap** | `ggml/src/ggml-cuda/moe-expert-cache.cu`: `g_mutex` (~323), `table_t::remap_dev` (~218), `moe_cache_remap_dev`, `moe_cache_get_table`, `moe_cache_in_place`, the takeover promise (`taken_cpy`) |
| the remap's host decision + the fused-launch placement | `ggml/src/ggml-cuda/ggml-cuda.cu` ~7788-7830 (`ggml_cuda_mul_mat_id`'s redirect), `ggml-cuda-moe-cache-hooks.cu`:293 (`ggml_backend_cuda_moe_cache_remap`), `ggml-cuda-try-fuse.cu`:2883 (`fuse_serve_routed`) |
| the KV cache the kernel writes (shared by BOTH arms) | `src/llama-kv-cache.cpp`: `grow_regions` (VA mapping — a no-op unless `LLAMA_KV_REGION`/a pool is on), `apply_ubatch`, `seq_park`/`park_move` |
| the pair's own concurrency contract | `tools/server/server-context.cpp`: the block header above `enum { ARM_PREFILL … }` — five invariants, the barrier, `arm_step_guard`, `arms_decode_loop` |

**The rule that decides the fix** (DESIGN §8 invariant 3): only one context may be inside `llama_process()` in
the **decode band** at a time, because the MoE remap is process-wide; the prefill arm's wide pass *may* overlap a
decode step — that overlap is the feature.  If the experiment ladder shows the overlap is simply unsound for a
host-expert shape, the two honest exits are:

1. **serialise the two arms' `llama_process` calls on such shapes** — correct, and it makes the pair inert on
   exactly the shapes it was measured on (do this only with the maintainer's agreement, and make it a *refusal*
   rather than a silent slowdown); or
2. **make the remap context-safe**: the decode arm must not read a table the prefill arm can remap under it —
   e.g. pin the tables a decode-band batch will read for the duration of the step (`moe_cache_promise`-style), or
   keep the decode arm on a private table set, or defer the remap out of the step.  This is the maintainer's
   chosen direction (2026-11-23).

### 2.5 Definition of done for TASK 1

* `gates/gate-arms-multislot.sh` **with the host-tier defaults** (35B-A3B Q8_0 `-ncmoe 40`, np 4, 20 rounds)
  **PASS**, with `check-gpu-faults.sh` clean and no `gpucore.*`;
* the `-ncmoe` refusal **removed** from `common_arms_pair_possible()`, and the predicate's comment updated;
* the 4B GPU-resident soak still PASS, `gates/gate-arms-n1-memory.sh` and `gates/gate-arms-itl.sh` still PASS,
  and A1-A4, A6 unchanged (the default path is not allowed to move);
* one `WORKLOG.md` entry + `INVARIANTS.md` facts + a `RUNS.md` section, and a patch in `steps/`;
* **report the finding and the fix to the maintainer before committing** (see §6).

---

## 3. The rest of the campaign, after TASK 1

Read `PLAN.md` for each step's own detail; this is the order and what is left.

* **STEP 7 — done.**  `--spec-type` is refused on the pair inside `common_arms_pair_possible()`; its re-admission
  is STEP 10.  Nothing to do unless you want the message wording reviewed.
* **STEP 8 — the gate set.**  Three exist (`gates/gate-arms-{n1-memory,itl,multislot}.sh`).  Still owed:
  * `gate-arms-default-inert.sh` — the pair OFF must print a **byte-identical startup shape** to
    `baseline/A6-startup-shape.txt` **and** read A1/A2/A3 identical, on 4 shapes.  (The shape anchor already
    exists; this gate is mostly plumbing + the full shape list.)
  * `gate-arms-recurrent-shared.sh` — promote `gates/probe-recurrent-shared.cpp` into a gate with its
    non-vacuity line (the two live contexts' placements **differ**), on **two** recurrent models (35B-A3B,
    Flash-Next/qwen4exp).
  * every gate must print its **non-vacuity** line; a gate that could pass on the single-context server is
    defective.
* **STEP 9 — the shape/latency matrix + the records.**  4B (1 GPU), phi-4 IQ4_XS (1 GPU, dense), 27B Q8_0
  (3 GPU `-sm tensor`), 35B-A3B Q8_0 (`-ncmoe 40`, host tier — **needs TASK 1**), Flash-Next IQ4_NL (3 GPU
  tensor, 3 slots `--kv-unified`), Gemma4 (iswa).  **Byte text claims at ONE request in flight only** — a
  concurrent arm's composition picks the kernels and even the unmodified server varies run to run
  (`INVARIANTS.md` r362-wip item 5).  Then `ENVIRONMENT.md` rows for `--prefill-decode-arms`,
  `--prefill-decode-arms-mode`, `LLAMA_ARG_PREFILL_DECODE_ARMS[_MODE]` and `LLAMA_RS_DEBUG_PLACEMENT`; the
  measured reserve; the VRAM delta per extra slot.
* **STEP 10 — re-admit the speculator on the pair.**  Three-way cold text claim
  (`pair+draft-mtp` == `pair+none` == `single+draft-mtp`), the MTP four axes, `LLAMA_SERVER_PARK_IDLE_SLOTS=0`
  pinned on every arm, the draft context's own `llama_process` inside the decode-band barrier, and the slot's
  speculative state reset on every handover.
* **STEP 11 — the delivery regression sweep:** `scripts/validate-set.sh`, plus `gate-prefill-logits.sh`,
  `gate-state-pipe.sh`, `gate-park-idle-slots.sh`, `gate-kv-residency.sh`, `gate-host-pool.sh`,
  `gate-spec-purity.sh`, `gate-mtp-dev-rows.sh`, `gate-moe-promise.sh`, `gate-node-budget.sh`,
  `gate-graph-replay-width.sh`, `scripts/check-gen-docs.sh` (the pair adds options → the three generated docs
  must be regenerated), `check-windows-portability.sh`, `check-windows-paths.sh`, and one whole-set `all` build.
* **STEP 12 — the candidate patch + the ask.**  Regenerate from a **canonical** chain
  (`scripts/apply-all.sh` at `release.json.base`), put it in `beta/<a NEW name>/` (never the old beta
  directory), report the measured trade/memory/shapes to the maintainer.  **Promotion into `patches/` is a new
  block and is the maintainer's decision.**

---

## 4. Environment and hygiene (every rule here cost this session real time)

* **`llama-cli` always `--single-turn`** (+ `--no-display-prompt` when scripted); bound anything that can block
  with `timeout`; every server under `setsid` and **kill the process group**.
* **`-lv` is per arm:** none for a text/hash arm, `-lv 4` for a census arm, **never `-lv 3`**.  Note that the
  *library's* INFO (`llama_kv_cache:`, `notify_borrowed:`, my census lines) needs `-lv 4` on the server too.
* **One signal, then wait, when you intend to read a teardown census.**  `timeout` forwards the signal it gets
  to its child, so an inner `timeout` around the server + a group TERM makes the server see a "second interrupt"
  and exit **without running a single destructor** — every census printed from a destructor then silently
  disappears.  `baseline/server-arm.sh` starts the server with no inner `timeout`, sends ONE TERM and waits for
  the exit.  Never SIGKILL a server whose teardown you are about to measure.
* **Never two GPU jobs at once**, and make it a mechanism: `baseline/mtp-axis.sh` takes an exclusive `flock` on
  `~/.cache/rdna-boosts/gpu.lock`.  Two concurrent arms already produced two different "same-seed" texts once.
* **Pin `HIP_VISIBLE_DEVICES`** (`0,2` / `0,1,2`); the 9950X3D iGPU (`gfx1036`) is otherwise enumerated and the
  load aborts with `invalid kernel file`.
* **`ROCBLAS_USE_HIPBLASLT=0` for every fresh-process text comparison**, and **never** put another ROCm install
  in `LD_LIBRARY_PATH`: `LD_LIBRARY_PATH` beats `RUNPATH`, so a 10.2 build then loads 10.1's runtime whose
  hipBLASLt 1.4 makes each process pick its own solution (`INVARIANTS.md` r359-wip).  `scripts/rocmlib.sh` is the
  resolver the delivery's gates now use.  **ROCm 10.1 is retired everywhere.**
* **Big artifacts never go to `/tmp`** (tmpfs here): prefill-logit bases live in `~/.cache/rdna-boosts/gates`.
* **`gpucore.*.gpu` files are 32 GB** — `rm -f gpucore.*.gpu` after any fault, before anything else.
* **Supervision contract (project README §5):** every wait bounded, a dead *or silent* process fails fast,
  progress on stderr, a timeout is a result and never a retry trigger.

---

## 5. Harness inventory (all under this directory)

```
baseline/  r353-anchors.md        the baseline anchors A1-A7 and how to re-run them
           A6-startup-shape.txt   the 13-line pair-OFF startup shape (the default-inert anchor)
           server-arm.sh          start a server arm, drive N requests, hashes + the log; ONE TERM then wait
           itl-arm.sh             the ITL-through-a-prefill measurement (slot A decodes while B prefills)
           mtp-axis.sh            one four-axis MTP arm (GPU flock; --spec none|draft-mtp|adaptive)
           norm-log.sh            a log's SHAPE (keeps both log line forms, masks addresses/decimals)
gates/     gate-arms-n1-memory.sh    both arms, ONE memory, byte text, decode-arm census, reservation census
           gate-arms-itl.sh         the latency claim (same text, ITL no worse) at a stated -ub
           gate-arms-multislot.sh   4 slots x 20 rounds soak (+ --model/--np/--extra for other shapes)
           probe-*.cpp              the five non-vacuity probes (host stream, ctx-memory, kv reservation,
                                    meta slots, recurrent placement) -- each compiles against the built tree:
   e.g.  /opt/rocm-10.2.0-gfx120X/lib/llvm/bin/clang++ -std=c++17 -O1 -pthread probes.cpp \
           -I$HOME/llama.cpp/include -I$HOME/llama.cpp/src -I$HOME/llama.cpp/ggml/include \
           -L$HOME/llama.cpp/build-rocm/bin -lllama -lggml -Wl,-rpath,$HOME/llama.cpp/build-rocm/bin
```

Every number of record is in `RUNS.md` (newest first) with its command and shape; the anchors are in
`baseline/r353-anchors.md`.

---

## 6. Git, records and the reporting rule

* **Delivery repo** (`~/llama-cpp-rdna-boosts`), branch `prefill-decode-arms-v2`: docs, gates, harnesses, patches.
  `git push -u origin prefill-decode-arms-v2` after each landing.  **`main` stays untouched.**
* **`~/llama.cpp`**, branch `arms-v2`: the code.  **Never push it** (a bare `push` in that checkout targets
  upstream `ggml-org/llama.cpp` and can get the account banned).
* One landing = one commit in each repo + a patch in `steps/` (`git diff > steps/STEP-<n>-<name>.patch` from
  `~/llama.cpp`).
* **Records, in the same landing:** a new dated `WORKLOG.md` entry at the **top** (`YYMMDD.VVV`, today's real
  date + the next free number), an appended `INVARIANTS.md` entry (facts, each tagged with the `rNNN-wip` of
  the landing), and a `RUNS.md` section (numbers with the command and the shape).
* **Report findings to the maintainer BEFORE committing a delivery-affecting change** — including this campaign's
  `wip/` landings, which is how every finding so far reached them.  **A new block, a renumbering, or a promotion
  into `patches/` is the maintainer's decision and is asked first; a green gate ladder is not permission.**
* If code and a document disagree: **the code wins for facts, the document wins for intent** — correct
  `DESIGN.md`/`PLAN.md`/this file in the same change.

---

## 7. Open questions for the maintainer (ask before deciding alone)

1. **TASK 1's exit:** if the experiment ladder shows the overlap is unsound for a host-expert shape and the
   remap cannot be made context-safe cheaply, is a *refusal* (the pair on GPU-resident shapes only) acceptable
   as the shipped state, with the host-tier case documented as a follow-up?
2. **The decode arm's arena reserve** is a 256 MiB constant measured on the 27B/3-GPU shape.  If TASK 1 changes
   the placement of the expert tables, re-measure it rather than keeping the constant.
3. **The Step-9 byte-claim rule** ("one request in flight") is this campaign's reading of DESIGN V8.  If the
   maintainer wants a byte claim *under* concurrency, the batch composition has to be pinned first — that is a
   separate instrument, not a gate tweak.
