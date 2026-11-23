# HANDOVER — `prefill-decode-arms` v2, campaign state + the immediate task

**Read this file first, then work top to bottom.**  It is written for a session that has *no* memory of the
campaign: it says what exists, what is green, what is open, and what to do first.  Everything below was measured
on this box; nothing here is a plan that has not been run unless it says so.

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`prefill-decode-arms-v2`** (pushed; `main` untouched at `91a81505`) |
| work tree | `~/llama.cpp`, branch **`arms-v2`** (local only, **never pushed**), tip `e460f0dea` (STEP 7a; the previous landing was `5e668c768`) |
| build | `~/bin/build-llama-rocm` → `build-rocm` (ROCm **10.2.0**); fast loop `cmake --build build-rocm --target llama-cli llama-server -j 16` |
| last landing | **STEP 7c** (retaken): `steps/STEP-7c-0012-aux-reserve-calibration.patch` (`WORKLOG.md` 261123.020) on top of **7d** `steps/STEP-7d-0011-pair-drop-disable.patch` (261123.019) and **7a** `steps/STEP-7a-0010-aux-narrow-regions.patch` (261123.017) |
| the defects it closes | (7a) the host-expert (`-ncmoe`) FAULT: **a slab work-base ALIAS** — two contexts' compute buffers were the same memory — fixed with one pinned aux region per extra context + thread-local routing; the `-ncmoe` refusal is gone.  (7d) the same soak's **INTERMITTENCY**: the prefill -> decode drop races the decode arm's graph capture; the pair now keeps both layouts reserved (§2.7) |
| the reserve loop | the aux compute region is charged to the slab RESERVE (an approximation, 1280 MiB), the slab logs the measured size + the reserve to set **once per slot**, and setting that value makes the region map from the unmapped VA spare with zero eviction (§2.8) |
| next action | **STEP 8** — the two missing gates (`gate-arms-default-inert.sh`, `gate-arms-recurrent-shared.sh`), each with its non-vacuity line (§3) |

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
| STEP 7a (= TASK 1) | `STEP-7a-0010-aux-narrow-regions.patch` | the **aux narrow region stack**: one pinned slab work region per extra context (draft = slot 0, a context that borrows another's memory = slot 1), a thread-local routing slot from the allocating context's identity, the iface rename, and the `-ncmoe` refusal **removed** |

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
| **the host-expert shape** (STEP 7a) | `gates/gate-arms-multislot.sh` on 35B-A3B Q8_0 `-ncmoe 40`, np 4, 20 rounds: **PASS**, 0 errors, 15 steps / 3 handovers / widest batch 3, `aux narrow-3 (borrowing-context compute) pinned at [26.31, 27.56] GiB`, journal clean, wall 28 s; the pair-OFF **control** on the same shape PASS (no aux region pinned) |

### 1.4 Open (this is what you are here for)

* **STEP 8 — the gate set.**  Three exist (`gates/gate-arms-{n1-memory,itl,multislot}.sh`; the multislot one
  now also takes `ARMS=` for the pair-OFF control and FAILS if a slab-armed shape did not give the decode arm
  its own pinned region).  Still owed: `gate-arms-default-inert.sh` and `gate-arms-recurrent-shared.sh` (§3).
* **STEP 9 — the shape/latency matrix** (the host-expert row is now unblocked) and the records
  (`ENVIRONMENT.md` rows; see the two doc gaps in §7).
* STEP 7 (speculator refused on the pair) is **in place** inside `common_arms_pair_possible()`
  (`params.speculative.has_aux()` → refuse); its re-admission is STEP 10.
* STEPs 10-12: see §3.
* **Both aux slots are covered** (draft slot 0 *and* borrowing slot 1), see §2.6 — the arm is an external MTP
  head on a MoE target with `-ncmoe`.
* **The host-tier soak is stable now** (5/5 with the pair's drop disabled, §2.7) — before 7d it failed about half
  the time, and 7a's three PASSes were luck (§2.5).

---

## 2. TASK 1 — CLOSED: the host-tier fault was a SLAB WORK-BASE ALIAS, not the MoE remap

**Status: fixed in STEP 7a, `-ncmoe > 0` is supported again.**  Read this section before touching the slab, the
pair, or any third context; the root cause generalises to every future extra context.

### 2.1 What was observed (and what it was blamed on)

`gates/gate-arms-multislot.sh` with the host-tier defaults (35B-A3B Q8_0 `-ncmoe 40`, np 4, 20 rounds):

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address
  ... kernel: void rms_norm_mul_rope_f32_qk_pair<false, float, __half, 0, 0>(rmr_args<float>, rmr_args<__half>)
```

and the server aborted.  The previous session's hypothesis was "the prefill arm remaps the MoE expert tables
from the decode arm's step" (the arena evictions in the log looked like the trigger).  **That was wrong.**

### 2.2 The root cause (`GGML_CUDA_WORK_ALLOC_DEBUG=1`)

```
0.07.583 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the PREFILL arm's compute view
0.07.586 W WORK_RELEASE dev=0 off=0.0 MiB need=910.8 MiB               <- its old (smaller) view, dropped
0.07.589 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the DECODE arm's view: SAME base 0
0.08.683 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB   <- the prefill arm re-takes base 0
0.08.686 GPU FAULT (the K rope's SET_ROWS kernel)
```

`ggml_cuda_slab_work_alloc()` has ONE "narrow" (decode/verify) base at offset 0 and assumed **one live narrow
view per device**.  Each `llama_context` keeps its own persistent compute buffer, so the pair's decode arm was
handed the prefill arm's memory: one arm's prefill wrote the other's activations, and the K rope then wrote KV
rows at the index tensor it found there.  The delivery fixed exactly this class once already — the MTP draft
context aliased at base 0 and got the `narrow2` region ("without this they both aliased at base 0").  The pair
is the **third** context; the region count was still one.

A second, independent defect: the routing flag was **process-wide** (`static bool g_slab_compute_narrow2`), so
the prefill arm's reserve inherited the decode arm's flag, was routed into the aux region, found a live view,
fell back to `cudaMalloc` and failed the graph reserve (1280 MiB OOM) 1.3 s into the soak.

### 2.3 The ladder that got there (each rung is one soak; keep it for the next defect)

| rung | env | result |
|---|---|---|
| 0 repro | — | FAULT at ~9 s, `rms_norm_mul_rope_f32_qk_pair<false,float,__half,0,0>` |
| 1 Q/K deferral off | `GGML_CUDA_ROPE_QK_PAIR=0 GGML_CUDA_QK_PAIR_FWHT=0` | **still faults**, in the *unfused* `rms_norm_mul_rope_f32<...>` — the deferral is not the mechanism.  **Note:** `GGML_CUDA_QK_PAIR_FWHT=0` alone could never have disabled the faulting instantiation (`fwht_n=0` = the plain pair, switch `GGML_CUDA_ROPE_QK_PAIR`) |
| 2 aux routing with a process-wide flag | temp `is_draft \| memory_borrowed` | server aborted at load (the flag leaked across threads) — a real finding, see §2.2 |
| 3 the fix | — | **PASS** |

Use `GGML_CUDA_WORK_ALLOC_DEBUG=1` early on any "two contexts, wrong numbers / GPU fault" report: it prints
every compute-buffer alloc/release with its base offset, which is what named this in one run.

### 2.4 The fix (`DESIGN.md` I9)

* the `narrow2` region becomes a **stack of N=2 pinned auxiliary regions** at the top of the reserved VA
  (slot 0 = an MTP draft context, slot 1 = a context that **borrows** another context's memory: the decode
  arm); the arena ceiling is the **lowest** pinned offset; the commit takes the pinned offset as an argument
  (the pin releases the lock to evict, so another context can pin in that window — the window is validated);
* the routing is a **`thread_local` slot derived from the allocating context's own identity**
  (`llama_context::slab_aux_slot()`: `cparams.ctx_other` → 1, `memory_borrowed` → 2).  Same pattern as STEP 4's
  Meta view slot, and it removes the process-wide flag from the MTP path too;
* `GGML_CUDA_SLAB_NARROW2=0` still bypasses the slab for such a context (the A/B arm);
* the `-ncmoe > 0` refusal is **removed** from `common_arms_pair_possible()`.

### 2.5 Definition of done — all met (numbers in `RUNS.md`, top section)

* host-tier soak **PASS** (0 errors, 15/3/3 census, aux region pinned, journal clean, wall 28 s vs 371-382 s
  for the faulting runs), plus the **pair-OFF control** on the same shape;
* the refusal removed; the 4B soak **PASS** (widest batch 4), `gate-arms-n1-memory.sh` and `gate-arms-itl.sh`
  **PASS** (-26 % ITL through a 5246-token prefill, +25 % prefill, text identical);
* A1/A2/A3 unchanged, **A4 KLD 0.000000 / 100.000 %**, A5 K axis `a87c4318b649`, A6 startup shape 13 lines
  identical, `check-shlib-symbols.sh` OK.

### 2.5a The DoD status, honestly (read this before quoting §2.5)

STEP 7a's DoD table in `RUNS.md` was measured at a moment when the persisted expert heat map made the soak pass;
**the same binary fails about half the time** once the prior drifts (`RUNS.md` STEP 7d).  So:

* 7a's **alias** fix stands (deterministic fault: 2/2 repro before, clean after) and is what keeps the host-tier
  shape from faulting;
* the soak's **green** line is only true with **STEP 7d** applied (pair keeps both compute layouts reserved:
  **5/5 PASS**, 0 capture errors);
* a host-expert gate must pin `MOE_EXPERT_HEATMAP=load` (the default writes the map; the gate now does).

### 2.7 The intermittency (STEP 7d) — the drop races the other arm's capture

The prefill -> decode **compute-buffer drop** is a process-wide mutation: it moves the movable-boundary slab's
work boundary and re-arms stood-down MoE arena tables.  Run from the prefill arm while the decode arm is inside
`llama_process()` capturing its graph, that fails the capture:

```
0.19.024 I compute buffer 1280.00 MiB from the slab work region (boundary 1536.00 MiB)
0.19.026 W moe_cache_rearm: re-armed 15 stood-down expert-cache tables after a compute-buffer drop
0.19.026 E ROCm error: operation failed due to a previous error during capture
0.19.058 E operation would make the legacy stream depend on a capturing blocking stream   <- consequence
```

`LLAMA_DROP_COMPUTE_BUFFERS=0` with the pair on: **3/3 clean**, which is the proof.  The fix is
`llama_context_disable_compute_buffer_drop()` called for **both** arms when the pair is created; the measured cost
is nil (arena 25537 MiB and hit-rate 1.0, identical to the drop-on arm, because the slab reclaims the wide view on
the next wide pass).  **This is exposure removal, not the general answer**: the general answer is a barrier that
keeps the other arm out of `llama_process()` around a drop, which would let the drop come back (DESIGN §8
invariants 3-4, `INVARIANTS.md` r366-wip).  Unmeasured for STEP 10: whether two more compute layouts (a drafter's)
make the always-reserved layout cost arena.

### 2.8 The reserve loop (STEP 7c) — approximation now, measured number for the next run

* the decode arm's compute region is charged to the **slab reserve** (not to the arena's headroom, which is
  `slab_extend`'s growth fuel — that variant aborts, `INVARIANTS.md` r365-wip);
* the figure is an **approximation** (`COMMON_ARMS_RESERVE_MIB = 1280`; measured 1280 MiB on 35B-A3B
  `-ncmoe 40` np 4, 512 MiB per aux context on Gemma4 26B-A4B + MTP);
* at the pin the slab logs, **once per slot**, the region's measured size and the reserve that would host every
  aux region (`headroom + sum(floors)`), and says whether this run had to evict arena tables;
* verified end to end: Gemma4 default 2496 MiB -> one eviction + "set 2560"; at 2560 -> both regions from the
  spare, zero eviction, same text;
* a defect this exposed in 7a's accounting is fixed with it (`mapped_arena`: the arena's mapped extent is not
  the slab's).

### 2.6 Both slots, and the arm that covers them

Which arm reaches which slot, so the next session does not re-derive it:

| shape | slot 0 (draft) | slot 1 (borrowing) |
|---|---|---|
| 35B-A3B Q8_0 `-ncmoe 40`, pair (the soak) | — | **pinned** (`aux narrow-3`) |
| **Gemma4 26B-A4B Q8_0 `-ncmoe 20` + `-md mtp-gemma-4-26B-A4B-it.gguf`** | **pinned** (`aux narrow-2`, 0.50 GiB) | **pinned** (`aux narrow-3`, 0.50 GiB) |
| 27B (dense, no slab) | n/a — no slab | n/a |
| 35B-A3B + `--spec-type draft-mtp-adaptive` (embedded head) | no second compute buffer at all | — |
| 35B-A3B + `--spec-type draft-dflash2` + `-md` | no second compute buffer; **and the text is not run-stable** (four runs, four hashes at two fixed lengths — the host tier's expert residency picks the kernel set, `GREEDY-PURITY.md` §89) | — |

The Gemma4 arm is the oracle: `none` == `draft-mtp` == `481 chars sha=a195d3d6953c`, stable over three cold
`none` runs, with both regions pinned in the one run.  Numbers in `RUNS.md`.

---

## 3. The rest of the campaign (TASK 1 is done)

Read `PLAN.md` for each step's own detail; this is the order and what is left.

* **STEP 7 — done.**  `--spec-type` is refused on the pair inside `common_arms_pair_possible()`; its re-admission
  is STEP 10.  Nothing to do unless you want the message wording reviewed.
* **STEP 7a — done** (= the old TASK 1): the host-tier fault, §2.
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
  (3 GPU `-sm tensor`), 35B-A3B Q8_0 (`-ncmoe 40`, host tier — **unblocked by STEP 7a**), Flash-Next IQ4_NL
  (3 GPU tensor, 3 slots `--kv-unified`), Gemma4 (iswa).  **Byte text claims at ONE request in flight only** — a
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

1. **`GGML_CUDA_SLAB_NARROW2` is not in `ENVIRONMENT.md`** — it is a delivery kill switch (default on) that
   predates this campaign, and STEP 7a now applies it to a borrowing context as well as a draft.  Add the row,
   or is the catalogue deliberately scoped to variables the delivery's own gates exercise?
2. **(ANSWERED 2026-11-23, STEP 7b: the "anonymous before the slab" placement was measured and ABORTS the
   host-tier soak -- the outside cushion is the slab's growth fuel, so it is not interchangeable with arena.
   The pinned region stands; `RUNS.md` has the three arms.)** **The two pinned aux regions cost real VRAM** (1.25 GiB on the 35B-A3B shape, taken from the arena).  With
   the pair and a drafter both live that is 2 × 1.25 GiB, and the delivery's `COMMON_ARMS_RESERVE_MIB` (256)
   accounts for only the measured *arena shrink* of one narrow context.  Should the pair's reserve grow to the
   measured pinned size at STEP 10, or is the eviction path (which handled it here) the accepted answer?
3. **The Step-9 byte-claim rule** ("one request in flight") is this campaign's reading of DESIGN V8.  If the
   maintainer wants a byte claim *under* concurrency, the batch composition has to be pinned first — that is a
   separate instrument, not a gate tweak.
4. **Where should the slab change live?**  STEP 7a touches a *delivery* file (`ggml-cuda-slab.cu`, blocks
   12/15's slab) from inside a wip campaign.  It is inert for the default path (A1-A6 unchanged, one region),
   but if the campaign is promoted it lands in whichever block owns the slab — that is a block decision.
