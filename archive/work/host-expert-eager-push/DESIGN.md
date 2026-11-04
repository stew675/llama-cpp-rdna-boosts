# DESIGN — `--host-experts push`: the host pushes, eagerly, and the GPU never pulls

Normative document for `wip/host-expert-eager-push/`.  It specifies the pipeline, the invariants and the open
questions; it is **not** a patch, and where it says "must" it means "the design is wrong if it does not".

## 0. The problem, stated exactly

An MoE expert that is not in VRAM must be in VRAM before the `MUL_MAT_ID` that routes to it runs.  On gfx1201
the GPU cannot fault a host page in and retry, and (measured, `archive/work/host-expert-locked-window/`) it
cannot be *given* a releasable host mapping either — `hipHostRegister` is a permanent hold.  So the bytes must
be **copied**, and the only questions are: from where, issued by whom, and how early.

```
   today (pool):   page cache --memcpy--> pinned slot --CE--> VRAM arena --> op
   proposed:       page cache ----------CE-----------> VRAM slot  --> op     (issued one op early)
```

The pipeline is identical in both; the proposal deletes the first hop.  Measured on the same box and blob:
137.3 µs (pool) vs **65.8-93.2 µs** (direct push) hot, ~3.5 ms vs **1.32 ms** cold.

## 1. The pipeline

```
policy (host)        the existing heat/LFRU admission + the eager lookahead (§3)
   |
   v
source               the read-only model mapping (page cache / GGUF on disk), or -- for a
                     table the pool still serves -- a pinned pool slot
   |
   v   CE copy, ordered on the device's stream, into a VRAM slot the policy owns
destination          an expert-cache arena slot (the existing two-tier map)
   |
   v   event: the slot becomes PUSHED
consumer             the MUL_MAT_ID that routes to it (the existing slot remap / DEVREMAP)
   |
   v   the slot returns to the policy on the next host boundary
recycle
```

There is no path in this design by which a kernel dereferences host memory.  That is the whole point.

## 2. Invariants

- **P1. Never device-pull.**  No GPU kernel reads a host address.  The in-place cold read stays off
  (`table_cold_ok()` keeps requiring a pinned master and the push mode never provides one), `host_dev` is not
  used for weights, and a code path that cannot express "push" must fall back to a copy, not to a read.
- **P2. Eager, not demand.**  The push for the op that will consume slot S is issued **before** that op runs,
  while the previous op computes.  The information already exists on the host: the routing ids are read back
  for the admission policy every decode step, and the prefill band has the whole ubatch's routing in staging
  order.  A demand-time push is the fallback, not the design.
- **P3. Slot ownership is explicit.**  A slot is `PUSHING` from the moment its copy is issued until its event
  completes; only a `PUSHED` slot may be consumed; the policy may not reuse a slot until the consuming op has
  completed.  This is the existing arena slot state machine (`slot_state`) plus one state — do not invent a
  second one.
- **P4. No whole-set hold, in any tier.**  The push source is the page cache (reclaimable) or the GGUF on disk.
  The only non-reclaimable bytes in the mode are the VRAM arena and (if the fallback is armed) a bounded pinned
  pool.  A registered range must never be held for the life of the process — that is the closed campaign's
  finding, and it is why this mode does not register anything.
- **P5. The fallback is counted, never silent.**  If a push cannot complete before its consumer (arena full,
  ordering not satisfied, source cold beyond the lookahead), that expert takes the pinned-pool copy — the
  measured-safe path — and the event is counted and logged at the same place the pool counts a fill.  The
  arithmetic result is identical either way (P7); only the cost differs.
- **P6. Split-portable by construction.**  One push queue per device, the decision on the host, no new
  per-simple-device backend iface member (PR #266's corner: `GGML_MOE_DEVREMAP` is NULL on the Meta backend and
  the feature silently vanishes under `-sm tensor`).  The `-sm tensor` arm is an acceptance gate for P1, not a
  later phase.
- **P7. Byte-identical.**  The push moves the same expert bytes into the same kernels; a mode difference that
  changes text is a bug in the mode, not a tolerance to accept.

## 3. Eagerness: what "one op early" means concretely

| band | where the routing is known | what the push does |
|---|---|---|
decode | the host already reads the ids back for the admission policy (LFRU/heat) | push the next step's admitted set while the current op computes; the lookahead is `n_used x n_layers x blob` bytes, bounded by free arena slots |
verify (≤ band) | same, one op ahead | same, at the band's width (`MOE_EXPERT_CACHE_MAX_TOK`) |
prefill (above the band) | the whole ubatch's routing, in staging order | the existing whole-table staging (`GGML_SCHED_STAGE`) already is an eager push — this campaign **changes its source** from the registered/pinned ring to the mapping, and must keep its measured 2103 t/s shape |

The decode lookahead is the campaign's engineering core: it must be bounded (a slot budget), ordered (an event
per push batch, not per expert), and cheap on the host (one syscall-free path per batch — this is where the
closed campaign's epoch/boundary lesson applies: one event per boundary, never a per-entry device query).

## 4. Source selection and the one open claim

`MOE_HOST_PUSH_SRC=pinned|mapping` (P1's switch).  `pinned` is the control and the fallback; `mapping` is the
candidate.  The claim to settle, with a dated record either way:

> `ggml-cuda.cu:6255`: a pageable `cudaMemcpyAsync` "is staged through a shared internal bounce and tears under
> back-to-back use (measured: nondeterministic greedy text)".

Measured 2026-11-03 (`archive/work/host-expert-locked-window/MEASUREMENTS.md` §6): **it does not reproduce** for
a 1-D push — 106-128 MiB in flight, 2-4 streams, pageable/locked/registered/pinned, 0 torn in 20-100
iterations each.  It cannot be cleared by a microbench, though, because the claim is bundled with the
loader/scheduler/async-input-ring in the real path, and the delivery ships two ROCm lines.  So P1 is a
**reproduce-or-refute gate**, not a benchmark: a back-to-back fill stress with same-seed text and
`check-gpu-faults.sh`, on both lines, 1 GPU and the split pair.  If it tears, the copy is load-bearing, this
campaign closes, and the pool keeps its memcpy with a record behind it.

## 5. Why `pinned` stays the mode of choice when the set fits

- It needs no staging of any kind: the master is `cudaMallocHost` and the fill is a direct H2D.
- It costs host RAM equal to the set, but that is the *only* non-reclaimable tier, and when it fits there is
  nothing to stream.
- Every push mode adds a lookup, an ordering step and a fallback path over it.

So `auto` must prefer `pinned` when `host_expert_bytes` fits `MemAvailable` with margin, and choose `push` only
when it cannot.  A push mode that runs on a box that could have been `pinned` is a regression in disguise.

## 6. Rejected alternatives (carry the reasons, do not re-derive them)

| alternative | why not |
|---|---|
device-pull (in-place read of the mapping) | `hipHostRegister` is an unreleasable, process-lifetime hold (`archive/work/host-expert-locked-window/RESOLUTION.md`); and an unheld read is a fatal `Page not present` |
mlock'd sliding window | `mlock` is releasable but gives no device page tables; registration gives page tables but holds forever — no primitive has both |
register-per-window | same hold, and the pages are not released on unregister/munmap either (measured) |
device-initiated request ("guided XNACK") | the host already has the routing one op earlier; a request adds a sync and no information, and the two implementations (in-kernel spin, pre-pass+relaunch) stall the device or break graph capture |
a second LRU over the push window | the closed campaign's v1 measured 118 hits in 11 776 promotes: the arena is the cache, this mode is only its *transport* |
a `memcpy` into a pinned staging ring "for safety" | that is `pool`.  If P1 says the direct push tears, the answer is `pool`, not a second staging copy that also tears |

## 7. The code this touches (when P2 starts)

| file | change |
|---|---|
`ggml/src/ggml-cuda/moe-expert-cache.cu` | source selection in `moe_cache_upload_pageable()`; the fill site; the PUSHING slot state; the boundary event for the push batch |
`ggml/src/ggml-cuda/ggml-cuda.cu` | the fill at `:6254` must take the mapping source when `MOE_HOST_PUSH_SRC=mapping`; keep the 2-D path on the pinned bounce (it is 4-5x slower and the loader's partial-row geometry is where the documented crash lives) |
`src/llama-model-loader.cpp` | `host_experts_mode`: `push` takes the `ROCm_HostMap` master like `pool` and never registers it |
`common/arg.cpp`, `include/llama.h` | `LLAMA_HOST_EXPERTS_MODE_PUSH` (+ `window` as an alias) |
`ggml/src/ggml-backend.cpp` | nothing: the scheduler already keeps a host-weight `MUL_MAT_ID` on its layer's device |

Non-negotiable at review: P1, P3, P4, P5, P6 — each is either a fatal fault, a leak, a silent corruption or a
`-sm tensor` dead end.
