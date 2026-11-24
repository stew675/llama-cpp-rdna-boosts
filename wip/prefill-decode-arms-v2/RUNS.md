## 2026-11-24 — STEP 16: the pair halved uncontested decode via a disabled drop (no graph capture) — FIXED (`WORKLOG.md` 261124.009)

Shape: the maintainer's `runme` (Flash-Next GSQ-IQ3_XXS, `-ncmoe 47`, attendance pool, `-sm tensor` on 2 devices,
`-np 2`, `draft-mtp` Q4_K_M, `-ub 4096`), their Three.js prompt, 1000 tokens, `GGML_CUDA_GRAPH_TRACE=1`.
No `ROCBLAS_USE_HIPBLASLT` override (it makes no difference here: 50.4 vs 45.9 t/s).

| arm | decode | captures | slab wide band | arena |
|---|---|---|---|---|
| `--prefill-decode-arms-mode off` | **83.2 t/s** | 2922 | not armed | 39 375 MiB |
| `off` + `LLAMA_DROP_COMPUTE_BUFFERS=0` (the pair's condition, no pair) | **46.5 t/s** | **none** | 5248 MiB armed | 33 938 MiB |
| `auto` (before STEP 16) | **45.9 t/s** | **none** | 5248 MiB armed | 31 551 MiB |
| `auto` (after STEP 16) | **88.4 t/s** | 3140 | released | 37 199 MiB |

Draft counts identical in every arm (1298 drafted / 565 accepted).  The `LLAMA_DROP_COMPUTE_BUFFERS=0` row is the
attribution: the same single-context server, only the drop withheld, halves.

Mechanism: the pair disabled the prefill→decode compute-buffer drop (STEP 7d/7f) to keep its process-wide mutation
away from the other arm's capture; with no drop the slab's wide band stays armed for the process's life,
`ggml_cuda_graph_compute` refuses to capture while it is armed, and the band-armed warmup branch defers
`capture_deferred` **silently** — so no decode graph is ever captured.  Fix: the drop runs again, fenced by the
OTHER arm's step boundary (`llama_context_set_drop_boundary()`), which is stronger than the capture barrier (that
one does not cover a replay).  The eviction's `ggml_cuda_graph_mem_freed()` bump is what already keeps captures
from baking a freed arena address (TODO #70).

Verification: `gates/gate-arms-state-split.sh` **PASS 11/11 twice** (concurrent prefill+decode — the race window),
`gate-arms-multislot.sh` **PASS** (20 rounds x 4 slots), `gate-arms-spec.sh` **PASS 36/36**,
`scripts/gate-prefill-logits.sh` **PASS**, whole-set `all` build clean, `check-windows-portability.sh` PASS.

Harness trap: do NOT wrap `gate-arms-state-split.sh` in `timeout` — the script becomes a process-group leader,
its `setsid` forks, `$!` names the short-lived parent, and the gate reports a false "the server aborted" while the
server is healthy (measured twice; unwrapped it passes).  The gate header now says so.

## 2026-11-24 — STEP 15: the split-tensor 2-D read was the PEER GRANT, not the split arithmetic — FIXED (`WORKLOG.md` 261124.007)

### The measurement that refuted the work order

Instrumented at the abort (`result_output`, 3-device `-sm tensor`, 5-token ubatch, `GGML_META_2D_DEBUG`):

| quantity | value |
|---|---|
| meta tensor | f32 `ne=[248320,5,1,1]`, `nb=[4,993280,4966400,4966400]`, `nbytes=4966400` |
| read | `offset=0 size=4966400` (the WHOLE tensor: the per-ubatch logits readback, `n_outputs=5`) |
| split state | `axis=0 n_segments=1 nr[0]=1 ne[]={82773,82773,82774}` |
| per-device | dev 0 `[82773,5] nb[1]=331092`, dev 1 `[82773,5] nb[1]=331092`, dev 2 `[82774,5] nb[1]=331096` |
| chunk arithmetic | `331092 + 331092 + 331096 = 993280` = the meta row = `nb[1]`; `GGML_ASSERT(offset_j == chunk_size_full)` holds |
| who failed | **device 1**, on the FIRST 5-row read; dev 0's identical copy (same pitches, same destination) SUCCEEDED |
| host destination | `0x...0d54` = the pinned output buffer (`buf_output`, reallocated 2 068 480 -> 5 621 760 B) + 331 092 — a `ROCm_Host` allocation registered by device 2 |

So the "4-byte discrepancy" is one extra ELEMENT on device 2, not a broken pitch, and the chunked path is exact.

### The mechanism (driver log, same run)

`AMD_LOG_LEVEL=3` on the aborting build, immediately before HIP returns the error:

```
rocdevice.cpp:2376: Allow p2p access failed - hsa_amd_agents_allow_access with err 4096
hip_memory.cpp:2593: hipMemcpy2DAsync: Returned hipErrorInvalidValue
hipMemcpy2DAsync (0x7ed55a450d54, 993280, 0x7f1b3fc96100, 331092, 331092, 5, hipMemcpyDeviceToHost, ...)
```

A 2-D D2H copy needs a peer grant of the DESTINATION's pinned allocation for the copying device's agent; this box
refuses those grants (it refuses `hipMemSetAccess` too — `hsa_amd_vmem_set_access ... status:4096` in the same log,
which is why `ggml_cuda_peer_copy_ok` exists).  The pinned output buffer is ONE device's allocation read from all
three; device 0 owns it and needed no grant.

### What was tried and did NOT help (recorded so it is not re-tried)

| probe | result |
|---|---|
| `ggml_cuda_set_device(ctx->device)` before the copy (its write twin does it) | still refused — kept anyway, the asymmetry was real |
| `hipDeviceSynchronize()` on the copying device first | still refused (nothing in flight was the blocker) |
| `GGML_CUDA_ASYNC_INPUT=0` (the async-input ring off) | still refused |
| the identical call twice: retry after the failure | **succeeds** — the refusal is state-dependent, so a retry must never be the fix |
| `n_copies` 1-D copies straight into the caller's pitch | **succeeds** on every device (this is the fix) |
| a standalone HIP probe of every argument combination (`repro/probe-hip-2d-copy.cpp`) | **all succeed** on an idle box: pitches, 1..6 copies, 1/4/16/64-aligned destinations, `Default`/`Portable`/`Mapped`/pageable hosts, a destination pinned by ANOTHER device, two copies in flight from different devices, bounce+scatter — so this class is only provable end-to-end |

### The fix

`ggml_cuda_rows_to_host(device, src, data, size, n_copies, stride_tensor, stride_data, stream, sync)` — the
KV-region `rows_to_host` generalised to any device — backs both 2-D read entry points
(`ggml_backend_cuda_get_tensor_2d_async`, `ggml_backend_cuda_buffer_get_tensor_2d`): a tight block or one copy is
ONE 1-D copy; a big synchronous block is device-gathered (2-D **device-to-device**) and read as one 1-D D2H per
chunk into the device's own pinned chunk, then scattered on the host; the async form / a small block is one 1-D
copy per row.  No 2-D D2H exists any more.  Also: the async read now selects its own device, and the staged arm
requires `size <= ggml_cuda_pin_chunk_bytes` (it could overflow the 64 MiB chunk for a bigger row).

### Verification

| check | result |
|---|---|
| `repro/repro-mtp-prefill.sh` `ROUNDS=3` (pair ON, the tester's shape) | **SURVIVED 3 rounds**; `http=200` + content in every round; `the decode arm is generating: 415 step(s), 2 handover(s)`; draft acceptance 0.611; fd trace 686 / max 688 / 11 |
| `repro/repro-mtp-prefill.sh MODE=off` (pre-fix control) | survived — and its ubatch census never contains `ubatch of 5 tokens`, so it never composes the failing read (see `gates/gate-arms-state-split.sh`) |
| `gates/gate-arms-state-split.sh` (new) | **PASS 11/11** — 2 rounds, best `prompt_n 77100`, both rounds `http=200`, 445 steps + 4 handovers, the 5-token verify batch present, pair-OFF control survives |
| `gates/gate-arms-cancel.sh` | **PASS** (39 s) |
| `gates/gate-arms-fd-budget.sh` | **PASS** (81 s) — 686 / 686 / 686, a second RCCL actually refused |
| `gates/gate-arms-default-inert.sh --no-cli` | **PASS** (4 shapes) — after item (b) below |
| `gates/gate-arms-multislot.sh` | **PASS** (32 s) — 20 rounds x 4 slots |
| `gates/gate-arms-spec.sh` | **PASS 36/36** (493 s) — one text across four arms on 3 devices f16/q8_0, drafts accepted on the decode arm (181 drafted / 157 accepted), the decode arm really ran (182 steps, 1 handover) |
| `scripts/gate-ckpt-reuse.sh` | **PASS** (13 s) — a kept restored checkpoint moves no token |
| `scripts/gate-state-pipe.sh` | **PASS** (115 s) — the batched / plain / pipelining-off / save-only / synchronous arms restore byte-identical text |
| `scripts/gate-meta-seg-drain.sh` | **PASS** (45 s) — the segmented async arm is live, the drain is wired, the fix is text-neutral |
| `scripts/gate-seqpar.sh` | **PASS** — arm 2 (OFF identity vs the base) **mean KLD -0.000000, same top p 100.000 %**; arm 3a/3b unchanged from the tip (0.010360 against its own 0.014351 control).  NOTE: this gate had NO Flash-Next prefill-logit base on this box (only the 27B one), so it could not pass on ANY build; the base was recorded the gate's documented way (`--record`) on the **STEP-14 tip**, then the fixed build was compared against it |
| `scripts/gate-spec-purity.sh` | **PASS** (506 s) — plain == draft-mtp on q8_0 and bf16, 3-device `-sm tensor`; width-probe maxdiff 0 |
| `scripts/gate-prefill-logits.sh` | **PASS** (24 s) — mean KLD 0.000000, same top p 100.000 % (27B base) |
| whole-set `all` build (`~/bin/build-llama-rocm`) | **clean** (rc 0, no errors) |
| `check-windows-portability.sh` / `check-windows-paths.sh` | **PASS** |
| `scripts/check-gpu-faults.sh --since <window>` | **PASS** in every gate above (each gate runs it) |

### The artifact's layout (261124.008)

`beta/prefill-decode-arms/` carries the fix as its **own patch file**, so it can be dropped whole at promotion
instead of re-cutting the block-17 patch:

| file | what | sha256 | size |
|---|---|---|---|
| `prefill-decode-arms.patch` | block 17 — unchanged from the tested STEP-14 artifact | `a2badad1…e366fe3c` | 42 files, +3073/−741 |
| `state-split-2d-read.patch` | STEP 15, cut **on top of** patch 1 (block 17 edits `ggml-cuda.cu` too) | `b3f0645a…c2a7336b` | 1 file, +85/−46 |

`apply-all.sh --include-beta` applies `beta/*/*.patch` in filename order (one commit each), so the order the fix
needs is the order the names give.  Verified: the tester path from `919794500` applies both and reports applied tree
**`29c9644350`** — the tree every result above was measured on — and a `git am` of both in filename order on a
checkout at `089f90067` reads the same tree.

### Findings (reported, not papered over)

1. **The work order's diagnosis was wrong** — see the table above; `INVARIANTS.md` r384-wip records the real one
   and r383-wip item 4 is corrected in place.
2. **`gate-arms-default-inert.sh`'s `27b-3gpu-tensor` anchor was stale since STEP 14**: STEP 14 added the
   `all-reduce: NOT bringing up a second RCCL comm layer` WARN (this shape creates an auxiliary band context), and
   the anchor was recorded at STEP 8.  Measured: the **STEP-14 tip** (fix stashed) fails that check **twice**, so
   261124.005's "PASS (4 shapes)" never held for that shape.  Re-recorded from the tip with the gate's own
   `--record` (2 header + 24 shape lines); a re-recorded anchor is a lock, not evidence.
3. **`--record` does not honour `--force`** in that gate (it only records a MISSING shape), so the stale anchor had
   to be removed by hand before re-recording.  The header promises otherwise.
4. **`gate-seqpar` had no Flash-Next prefill-logit base here** (only the 27B one), so its arm 2 could not pass on any
   build: recorded the base from the **STEP-14 tip** with the gate's documented `--record` (tip PASS, `13:55-14:00`),
   after which the fixed build reads **mean KLD -0.000000 / same top p 100.000 %** — prefill-logit neutral.  Fresh-box
   gap: `TODO` #377.
5. **Kernel GPU page faults exist at 12:20-12:24 only** — the pre-fix `GGML_ABORT` runs and the `-9` kills of those
   aborted servers.  `scripts/check-gpu-faults.sh --since 12:56` (and every gate's own window, and the run after
   12:24:30) is clean: nothing in the fixed build faulted.

## 2026-11-24 — STEP 14: `-sm tensor` on 3 devices was FD exhaustion, not memory — FIXED (`WORKLOG.md` 261124.005)

### The numbers (Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, pool, `-sm tensor -mg 0`, `-np 2`)

| arm | open fds | `/dev/shm/nccl-*` |
|---|---|---|
| 1 GPU, pair ON | 10 | — |
| 2 dev, pair ON | 568 | ~548 |
| 3 dev, pair OFF | 686 | 646 |
| 3 dev, pair ON (no drafter) | 1356 | 1292 |
| 3 dev, pair ON + MTP draft | **2026** | ~1938 |
| **3 dev, after the fix: base / pair+drafter / pair+drafter** | **686 / 686 / 686** | 646 |

One `ncclCommInitAll` = one `/dev/shm/nccl-*` segment per peer per channel = 646 fds on 3 devices.  The aborting
`select()` is in a prebuilt library, so `FD_SETSIZE` cannot be raised here; `ulimit -n` = 524288 is irrelevant.

### The fix and its verification

`llama_context` asks for the copy-engine chain for an AUXILIARY context (`ctx_memory` / `ctx_other` / `ctx_aux` —
the `slab_aux_slot()` identity) through a consume-once backend proc; ggml-cuda additionally refuses a second
`ncclCommInitAll` (`g_comm_rccl_layers`), so the invariant holds even if the request is not made.

| check | result |
|---|---|
| `gates/gate-arms-fd-budget.sh` (new) | **PASS** — base 686, pair off+drafter 686, pair on+drafter 686; a second RCCL actually refused (non-vacuity); below `FD_SETSIZE` |
| `gates/gate-arms-spec.sh` | **PASS 36/36** (pair on/off x spec on 3-dev f16 and q8_0: one text) |
| `scripts/gate-spec-purity.sh` (delivery) | **PASS** — same hashes as before (`4d87b7b221f7`, `eb89102cb7d8`, `a87c4318b649`), width probe maxdiff 0 |
| `scripts/gate-prefill-logits.sh` | **PASS** — KLD 0.000000, same-top-p 100.000 % |
| `gates/gate-arms-cancel.sh` | **PASS** |
| whole-set `all` build | clean |

### Still open (behind it)

The tester's recipe now reaches a 2-D state-save read failure on the same shape:
`2-D read of 'result_output' failed (invalid argument): 331092 bytes x 5 copies, tensor pitch 331092, host pitch
993280 (ROCm1 buffer)` — a row/vocab-split 2-D tensor, `host pitch = 3 x tensor pitch`, from
`llama_io_write_host` -> `ggml_backend_tensor_get_n` (N single-row reads collapsed into one `n_copies=N` 2-D read)
into the Meta backend's chunked path.  Pair OFF survives the same recipe and `GGML_CUDA_ALLREDUCE=ce` reproduces
it, so it is independent of the all-reduce chain and is a delivery-level path the pair's draft checkpoint reaches.

## 2026-11-24 — STEP 13: a beta tester's abort crash (FIXED) and the `-sm tensor` FD ceiling (diagnosed) — `WORKLOG.md` 261124.004

### 1. Crash: aborting one of two concurrent generations

Reproducer (`/tmp/repro-cancel.sh`, kept as `gates/gate-arms-cancel.sh`): 4B Q8_0, 1 GPU, `-np 2`, two streaming
generations, `kill -9` one client (the server enqueues a CANCEL), then check `/health`.

| build | pair | result |
|---|---|---|
| pre-fix | ON | **died on abort attempt 2 of 2 runs** — `GGML_ASSERT(t_start > 0)` (`server-common.h:463`) |
| pre-fix | OFF | survived 25/25 |
| fixed | ON | survived 25/25 |
| fixed | ON, tester's shape (Flash-Next `-ncmoe 48`, pool, `-sm tensor`) | survived 6/6 rounds, 0 errors |

Non-vacuity: `the decode arm is generating: 2158 step(s), 25 handover(s), widest batch 2 slot(s)` (from a round
that lets one request FINISH — an aborted request never sets `finish_pending`, so abort rounds alone are vacuous).

The new gate, both ways round (the reason it is a gate and not a test):

```
pre-fix build : FAIL — "the server DIED on abort round 2" / GGML_ASSERT(t_start > 0)
fixed build   : PASS — 12 abort rounds, census 2158/25/2, pair-OFF control passes, journal clean
```

### 2. Crash: long generation + a ~77K-token prefill, `-sm tensor`, 3 devices

Reproducer: Flash-Next GSQ-IQ3_XXS, `-ncmoe 48`, pool (`LLAMA_KV_REGION_POOL=102400`), `-c 204800`, `-np 2`,
`draft-mtp-adaptive` n_max 7 with the `-Q8_0` head, `-sm tensor -mg 0`, pair ON; a streaming 10000-word request,
then 5 s later a 340 KB / ~77K-token prompt on the other slot.

```
round 1, ~6 s after the prefill started:  *** bit out of range 0 - FD_SETSIZE on fd_set ***: terminated  -> Aborted
```

**fd census** (same shape, after load, before any request):

| shape | open fds | `/dev/shm/nccl-*` | verdict |
|---|---|---|---|
| 1 GPU, pair ON | 10 | — | fine |
| 3 dev `-sm tensor`, pair OFF | 686 | 646 | fine |
| 2 dev `-sm tensor`, pair ON | 568 | ~548 | fine (prefill processed: `http=200`, acceptance 0.72, arm 342 steps) |
| 3 dev `-sm tensor`, pair ON | 1356 | 1292 | **abort** |
| 3 dev `-sm tensor`, `GGML_CUDA_ALLREDUCE=ce`, pair OFF | **11** | 0 | fine |
| 3 dev `-sm tensor`, `GGML_CUDA_ALLREDUCE=ce`, pair ON | **11** | 0 | fine |

The 646 is RCCL's IPC rings: `auto` initialises RCCL eagerly (host-expert or not), and the pair's second
`llama_context` doubles it.  `FD_SETSIZE` is compile-time and the tripping `select()` is in a prebuilt library, so
it cannot be raised from here; `ulimit -n` is 524288 and irrelevant.  `ce` (no RCCL) has no fds at all and does
not grow with a second context.

**Not fixed yet.**  The maintainer's decision is a true fix.  Sharing one communicator is unsound (concurrent
arms cannot guarantee a matching collective order across ranks — see `INVARIANTS.md` r382-wip item 4), so the
candidates are a per-context all-reduce chain for the decode context, or a smaller RCCL descriptor footprint.

### 3. Regression set for STEP 13 (all green)

`gate-arms-cancel.sh` (new) PASS, `gate-arms-default-inert.sh --no-cli` PASS (4 shapes, pair-OFF shapes
byte-identical, the pair-ON sensitivity arm still reads the same responses), `gate-arms-multislot.sh` PASS
(`9 step(s), 3 handover(s), widest batch 3`, 0 errors), `gate-arms-spec.sh` **36/36 PASS**,
`scripts/gate-park-idle-slots.sh` PASS, `scripts/gate-ckpt-reuse.sh` PASS, whole-set `all` build clean.

## 2026-11-24 — STEP 11 (the delivery regression sweep) and STEP 12 (the candidate) — `WORKLOG.md` 261124.003

### STEP 11: 16/16 delivery gates green on `ffaee2c52`

Run order and verdicts (one GPU job at a time, under `flock ~/.cache/rdna-boosts/gpu.lock`; every server arm with
`HIP_VISIBLE_DEVICES` pinned):

| # | gate | verdict |
|---|---|---|
| 1 | `scripts/validate-set.sh` | PASS — 18 default + 2 optional, tree `27db9bfd4`, 0 unexplained file-set rows |
| 2 | `scripts/check-shlib-symbols.sh` | PASS — 16/16 libs |
| 3 | `scripts/check-gen-docs.sh` | **FAIL, expected** — 2 rows in `tools/server/README.md` (the candidate's obligation) |
| 4 | `check-windows-portability.sh` / `check-windows-paths.sh` | PASS — 525 files, 4660 paths |
| 5 | `gate-prefill-logits.sh` | PASS — KLD **0.000000**, same-top-p **100.000 %** |
| 6 | `gate-state-pipe.sh` | PASS — pipe 0.48x sync |
| 7 | `gate-park-idle-slots.sh` | PASS — A-D byte-identical, 6 parks / 2 unparks traced |
| 8 | `gate-kv-residency.sh` | PASS — window + pool + 2-device pool section; first run with `MODEL_POOL=.../GSQ-RCO-IQ3_S/...` (89.10 % hit rate, the model tree was missing), then **on its own defaults** once the tree was restored (`GSQ-IQ3_XXS` + its `-Q8_0` sidecar, PASS, 89.07 %) |
| 9 | `gate-host-pool.sh` | PASS **59/59**, every arm `83d058e8af9c` |
| 10 | `gate-mtp-dev-rows.sh` | PASS — `bbb051e2aeab` both arms, 87/119, `h_nextn` buffer only in the device arm |
| 11 | `gate-spec-purity.sh` | PASS — plain == draft-mtp on q8_0 + bf16 x 4 axes; width probe maxdiff 0 |
| 12 | `gate-draft-ctx.sh` | PASS 11 checks — `e7aae6034b4f`, own KV cache, acceptance 0.814 |
| 13 | `gate-moe-promise.sh` | PASS — `stale=1152` (non-vacuous), repair 238/238, text `c0e7e91d84c4` |
| 14 | `gate-node-budget.sh` | PASS — 4 arms `a1b7eb2ee7a6` |
| 15 | `gate-graph-replay-width.sh` | PASS — default untouched, kill switch, 21 prefill captures |
| 16 | `gate-ckpt-reuse.sh` | PASS — keep 3 / eager 3, `==` on all four requests |

Plus the whole-set `all` build at the tip and `scripts/check-gpu-faults.sh --since` over the whole sweep window:
journal clean.

**Two findings, neither a defect in this campaign** (`INVARIANTS.md` r380-wip):

1. **Two segfaults were STALE BUILD ARTIFACTS.**  `llama-perplexity` + `libllama-perplexity-impl.so` and
   `llama-gen-docs` were dated `11-23 15:17` beside a `libllama` from `11-24 08:13` (the campaign only rebuilt
   `llama-cli`/`llama-server`).  `~/bin/build-llama-rocm` fixed both; the `all` build now precedes the ladder.
2. **`gate-kv-residency.sh`'s default `MODEL_POOL`** (`.../GSQ-IQ3_XXS/...`) and its derived `-Q8_0` sidecar were
   absent from this box at the sweep's start, while `gate-draft-ctx.sh` already defaults to `GSQ-RCO-IQ3_S`.  Run
   first with `MODEL_POOL=.../GSQ-RCO-IQ3_S/...` (PASS), then **on its own defaults** after the maintainer restored
   the tree (`GSQ-IQ3_XXS` + `-Q8_0` sidecar; PASS, 89.07 %) -- an **environment gap, not gate debt**: its arms are
   self-referential, so nothing needs repointing and `TODO` #369 stays at three gates.

### STEP 12: the candidate — `beta/prefill-decode-arms/` replaced, one patch, round-tripped

Cut from a canonical chain, not from `arms-v2`:

```
git clone ~/llama.cpp ~/cand-chain     # clean checkout at 919794500
apply-all.sh .                         # -> tip 089f90067 (== release.json.tip), tree 27db9bfd4 (== .tree)
19 steps/ applied in order             # -> tree 14dfb1f64 == arms-v2 ffaee2c52^{tree}  EXACTLY
llama-gen-docs                         # the candidate's regeneration: 1 file, +2 rows
git diff 089f90067 857f18168           # 42 files, +2854/-715 ; single format-patch commit
```

| | |
|---|---|
| patch sha256 | `a4e207c96898b5c6a1075d55de340a4d2793a24f95e9cb0e34bbf87c4687585e` |
| size | 42 files, +2854 / -715 (317 302 bytes) |
| applied tree | **`84ffb1ef1c5fd56f2a02f44f9068bef839a95d98`** |
| round-trip | clone at `089f90067` + `git am` -> `84ffb1ef1c` (clean) |
| tester path | `apply-all.sh --include-beta .` from `919794500` -> reports the same applied tree |
| `check-gen-docs.sh` on the applied tree | **ok** (the regeneration is inside the patch) |

**A correction to `HANDOVER-2026-11-24-campaign.md` §12.1:** the cut base is the **release tree** `089f90067`, not `919794500`.  A
`beta/` candidate is applied by `apply-all.sh --include-beta` **after** the blocks, strictly, one commit each, so a
base-relative patch cannot apply; and a plain `git diff` is not `git am`-able at all ("Patch format detection
failed").  The candidate is therefore `format-patch` output over `release.json.tree -> campaign tree`.

# RUNS — dated measurement record for `prefill-decode-arms` v2

One section per landing, newest first.  A line here is the **record of a claim**: the command, the
shape, the hash/rate, and (for a comparison) the arm it is compared against.  A gate's own output is
not the record — it is the instrument.

Anchors (the inertness reference set) live in [`baseline/r353-anchors.md`](baseline/r353-anchors.md).

---

## 2026-11-24 — STEP 10: the speculator ON the pair, and the six memory defects it exposed (`WORKLOG.md` 261124.001)

`steps/STEP-10-0018-speculator-on-the-pair.patch` (`~/llama.cpp` `f3e34859d`).  The decode arm runs the WHOLE
speculative round for the slots it owns; one speculator lock covers both arms' paths; a slot's speculative state
is reset at every handover; `common_arms_pair_possible()` admits `draft-simple` / `draft-mtp` /
`draft-mtp-adaptive` / the n-gram family and keeps the layer-input families and a gemma4 assistant refused.

Re-admitting the drafter exposed six defects that only a SECOND context, a THIRD layout or a live drafter can
reach.  Four were mine to fix here (they are correctness); the two slab *sizing* ones are recorded as follow-ups
(`HANDOVER-2026-11-24-campaign.md` §7.2/§7.8).

### 1. The claim: one cold greedy TEXT across four arms, three shapes

`gates/gate-arms-spec.sh` (new; `--shapes 27b-3gpu-f16,27b-3gpu-q8,27b-2gpu-f16 --axes R,K`), 27B Q8_0 with its
embedded MTP head, `draft-mtp-adaptive` at the delivery ceiling 7, one fresh process per arm, one request in
flight, `ROCBLAS_USE_HIPBLASLT=0` on every arm, `LLAMA_SERVER_PARK_IDLE_SLOTS=0` (the harness pins it).
**30/30 checks PASS.**

| shape | single+none | single+mtp | pair+none | pair+mtp | drafts/acc (pair) |
|---|---|---|---|---|---|
| 3 dev f16 | `f2753e13e419` | `f2753e13e419` | `f2753e13e419` | `f2753e13e419` | 34 / 25 |
| 3 dev q8_0 | `f2753e13e419` | `f2753e13e419` | `f2753e13e419` | `f2753e13e419` | 32 / 26 |
| 2 dev f16 | `0f06ed90e017` | `0f06ed90e017` | `0f06ed90e017` | `0f06ed90e017` | 32 / 26 |

Non-vacuity per arm: the pair arms print `the decode arm is generating: 47 step(s), 1 handover(s)` (no drafter)
and `19 step(s), 1 handover(s)` (drafter -- a verify step emits more than one token, so the same 48-token
request needs fewer steps), and every drafter arm reports `#gen drafts > 0` AND `#acc drafts > 0`.  The 3-device
f16 text is the STEP 9 matrix's own row (`f2753e13e419`), so this shape's answer did not move.

### 2. The four-axis protocol, ON the pair (one request per arm, `-n 600`)

`gates/gate-arms-spec.sh --axes R,K,P,C` on 27b-3gpu-f16 (a server arm, because the pair is a server feature;
`baseline/server-arm.sh --nreq 1` is new for exactly this).

| axis | pair+none | pair+draft | drafts / accepted | decode-arm steps |
|---|---|---|---|---|
| R | `3a41e71cfaab` | `3a41e71cfaab` | 181 / 161 | 181 |
| K | `8fe00a125d38` | `8fe00a125d38` | 74 / 73 | 74 |
| P | `fb4519c618ef` | `fb4519c618ef` | 109 / 98 | 109 |
| C | `4d46d5ab509d` | `4d46d5ab509d` | 181 / 157 | 182 |

K is the STEP 0 A5 axis; its pair-off hash here is the SERVER's, not A5's cli value, so the two are not
comparable by value -- what this row claims is pair+none == pair+draft.

### 3. The default path did not move

`gates/gate-arms-default-inert.sh` PASS (4 shapes: cli text `be904e86f248` / `db5ab8e63e11` /
`10a82b5559a9` / `e9db5cb8db0b`, startup shapes byte-identical), `baseline/mtp-axis.sh --axis K --spec adaptive`
on 27B 3-dev: **`a87c4318b649`, 74 / 73 / 435 / 423, pos-1 0.986** -- the STEP 0 A5 / STEP 10a value exactly, so
the extracted draft helpers and the per-context target rows changed nothing on the single-context path.
`gates/gate-arms-n1-memory.sh` PASS.  `gates/gate-arms-multislot.sh` with **no** drafter: PASS, `widest batch 3`.

### 4. Concurrency: the host-expert shape with a drafter (the arm that found three of the defects)

35B-A3B Q8_0 `-ncmoe 40`, np 4, `--spec-type draft-mtp-adaptive --spec-draft-n-max 7`,
`MOE_EXPERT_HEATMAP=load`, `gates/gate-arms-multislot.sh` (3 devices, `-sm layer`), 20 rounds x 4 slots:
**PASS 2/2 at `ROUNDS=20`** and 7 of 8 at `ROUNDS=10` (the one failure was the gate's own overlap check --
`widest batch 1` for the whole soak, because a verify step emits up to 8 tokens and two decoders therefore rarely
share a batch at 16 tokens per request; no fault, no error).  Census: `6 step(s), 2 handover(s), widest batch 2`.

`baseline/itl-arm.sh` (slot A decodes 64 tokens while slot B prefills 5246), pair + drafter, 35B-A3B host tier,
5 of 5 runs: no fault, **ITL 1.26-1.28 ms/token** through the prefill, `20-28` decode-arm steps,
`2 handovers`, `widest batch 2` on three of them.

### 5. The cost, measured (single request each, 1 device, 35B-A3B `-ncmoe 40` np 4, prompt 5246, `-n 64`)

| | pair OFF + MTP | pair ON + MTP |
|---|---|---|
| text | `cb8fa5a1790a` | **`cb8fa5a1790a`** |
| arena | 17467.5 MiB (53.5 % residency) | 16192.5 MiB (49.6 %) |
| expert hit rate | 0.9173 | 0.8968 |
| prompt | 316-365 t/s | 364 t/s |
| generation | 52.94 t/s | 48.22 t/s |

i.e. the charged reserve is real: the draft's 3584 MiB term is the same in both arms and the pair's 1280 MiB
term comes out of the arena, for ~1.7 pt of hit rate and ~9 % of a single-request decode rate on this shape.
The 20-round soak and the ITL arm are the shapes where that trade pays (ITL 70 ms/token -> 1.3 ms/token through a
5 246-token prefill).

### 6. The six defects, each with the measurement that named it

| # | symptom | mechanism | fix |
|---|---|---|---|
| 1 | `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `rms_norm_mul_rope_f32_qk_pair`, 40/40 requests, 15 s into a 4-slot soak with a drafter | the DRAFT context of an embedded MTP head (`general.architecture = qwen35`) keeps no `ctx_other`, so `slab_aux_slot()` returned 0 and its 512 MiB compute buffer shared the target's base-0 narrow view -- safe only while the two are serialised on one thread, and the pair runs them concurrently | `llama_context_params::ctx_aux` (set by the spec init when the pair is live) routes it to the slab's draft region; `is_draft` follows the same test |
| 2 | `ROCm error: operation failed due to a previous error during capture`, abort in `hipStreamSynchronize` | the PREFILL arm re-reserved a compute buffer (3 devices x 768 MiB) while the DECODE arm was capturing its verify graph; the driver refuses an allocation during a capture and poisons it | `ggml_cuda_capture_excl_guard` -- an allocation takes the STEP 7h capture barrier exclusively (skipped on the recording thread), and the decode arm is reserved once at its widest batch at pair creation |
| 3 | `ROCm error: dependency created on uncaptured work in another stream` in `ggml_cuda_peer_copy_staged` (1 of 3 soaks) | a cross-stream `hipStreamWaitEvent` issued while another thread recorded a graph -- the same class as 2 | the event wait and the peer staging copy take the barrier SHARED (the recording thread skips it) |
| 4 | `HSA_STATUS_ERROR_MEMORY_FAULT` ("Page not present") in `ggml_cuda_ain_batch` at LOAD, with both aux regions present | the SECOND aux region is stacked BELOW the first, in the hole the first pin left "mapped" (`s.mapped` jumps over it) but never mapped -- a device write there faults | the commit maps the region when it lies entirely below the extent (and declines if the range really is mapped) |
| 5 | the decode arm's 1280 MiB verify buffer left the slab for a plain `cudaMalloc` while the arena had given up 1280 MiB for it | the aux floor call passed the 1-based slab slot where the CUDA side takes 0-based, so it pinned a PHANTOM region one slot up and the real one was always sized by the first allocation -- and an aux region could not grow | pass `slab_aux - 1`; and let a pin GROW when nothing is live at that slot (the decode arm's first reserve is 768 MiB, its verify-width one 1280) |
| 6 | (following the 5 fix) the slab's own recommendation named 2304 MiB for a shape whose two regions need 3328 | the census line totals the pinned floors, so #5's phantom pin and the undersized region made the number wrong too | no code change: with #5 the line reads `3328 MiB would host every auxiliary region (headroom 1536 + aux 1792)`, matching `COMMON_ARMS_RESERVE_MIB` + the draft term |

Every one of them was invisible on the single-context server and on every single-request arm: #1-#3 need two
contexts live at once, #4 needs two aux regions, #5/#6 need a drafter's verify width.  Reproductions (all with
`GGML_CUDA_WORK_ALLOC_DEBUG=1` where a layout question was open) are in the soak/ITL logs under
`~/.cache/rdna-boosts/arms-v2/step10/`.

### 8. STEP 10b -- the per-aux-context reserve count (the maintainer's decision, 2026-11-24)

`steps/STEP-10b-0019-per-aux-context-reserve.patch` (`~/llama.cpp` `ffaee2c52`), `common/common.cpp`.
`common_arms_reserve_derive()` replaces the single `COMMON_ARMS_RESERVE_MIB` term with `{n_aux, bytes}` =
`n_aux x COMMON_ARMS_RESERVE_MIB_PER_AUX`, and `n_aux` counts the auxiliary contexts whose compute region no other
term covers -- today the decode arm, because the drafter's region is `common_draft_reserve_derive()`'s `compute`
term (derived, or the flat fallback).  Counting both would withhold 2560 instead of 1280 MiB and spend ~1.3 GiB of
arena on nothing (6144 vs 4864 MiB of reserve on this shape).

Re-measured on 35B-A3B Q8_0 `-ncmoe 40` np 4 + `draft-mtp-adaptive` n_max 7, one request, `-n 64`:

| | value |
|---|---|
| log line | `the two-arm server reserves 1 auxiliary compute region(s) x 1280 MiB = 1280 MiB (the drafter's own region ... is in the draft reserve term below)` |
| total reserve | `4864 MiB in total (draft 3584 + two-arm auxiliary contexts 1280)` -- unchanged |
| the decode arm's region | grows `768 -> 1280 MiB` inside its first reserve (the growth path), from the unmapped VA spare, 0 evictions |
| the draft's region | `512 MiB` |
| the slab's recommendation | `3328 MiB would host every auxiliary region (headroom 1536 + aux 1792)` |
| text | `cb8fa5a1790a` -- unchanged |

Re-verified after it: `gate-arms-default-inert.sh --no-cli` PASS (the term is 0 with the pair off),
`gate-arms-spec.sh` 19/19 on 27b-3gpu-f16 + 27b-2gpu-f16 with the K axis, the A5 anchor `a87c4318b649` with
74/73/435/423, and `gate-arms-multislot.sh` with the drafter at its new enforced floor of 20 rounds (PASS,
`widest batch 2`).

### 9. What is NOT claimed

* no throughput claim: the numbers above are single runs at one request in flight, printed as a cost row;
* the two slab SIZING items are follow-ups, not fixed here (`HANDOVER-2026-11-24-campaign.md` §7.2/§7.8): `COMMON_ARMS_RESERVE_MIB`
  is still one constant for a decode arm whose layout now depends on the drafter's verify width, and a
  second/third aux context would need a per-context count (the slab's calibration line is the mechanism today);
* `LLAMA_MTP_DEV_ROWS=1` is inert on the pair (the device h-row path binds `ctx_dft`'s state source to ONE
  target context); it now says so in the log instead of silently reading the wrong arms' rows.

---

## 2026-11-23 — STEP 10a: the draft round extracted for both arms (`WORKLOG.md` 261123.027)

`steps/STEP-10a-0017-draft-round-extraction.patch` (`~/llama.cpp` `e7ae0b470`), `tools/server/server-context.cpp`
+161/-155.  `slot_accept_draft_round(slot, ctx, seq_rm_type)` now holds the verify/accept/rollback/emit round that
was inline in `post_decode`, with the context as a PARAMETER (the decode arm's verify batch runs on `ctx_dec`,
which borrows the same memory, so positions and rollback are the same object) and that context's `seq_rm_type` as
its capability.  The two separate `accept_special_token` lambdas became one member helper.  This is the same
shape as STEP 6d-1's `slot_step()` extraction: no behaviour change, and it is what lets the decode arm run the
round at all.

**Verified on the drafter path, pair OFF** (27B Q8_0, 3 devices `-sm tensor`, `draft-mtp-adaptive` n_max 7,
`-n 600`, `baseline/mtp-axis.sh`):

| axis | text | acceptance (drafts/gen tokens) | the STEP 0 A5 record |
|---|---|---|---|
| K | **`a87c4318b649`** (2174 chars) | **74 / 73 / 435 / 423**, pos-1 0.986 | `a87c4318b649`, **74 / 73 / 435 / 423** -- identical |
| R | `28c205358077` (3156 chars) | 187 / 145 / 715 / 411, pos-1 0.775 | (A5's R is `-n 3000`; not comparable by length) |

**Still owed by STEP 10** (the next landing): the decode arm's draft round (mirror the prefill arm's
`draft params -> common_speculative_draft -> handle_last_sampled_token -> process -> slot_accept_draft_round`),
the speculator mutex across the two arms, the slot's speculative state reset on BOTH handover directions, the
predicate change that admits `draft-mtp`/`-adaptive`/`draft-simple`/ngram while keeping the layer-input families
and a gemma4 assistant refused, and then the three-way cold text oracle on 3 devices f16 / 3 devices q8_0 /
2 devices f16.

---

## 2026-11-23 — STEP 9: the shape / latency / memory matrix — and the two defects it found (`WORKLOG.md` 261123.023)

`baseline/shape-matrix.sh` (new) + `server-arm.sh --vram-probe` (new).  Both landings it needed are
`WORKLOG.md` 261123.021 (STEP 7e) and this entry's **STEP 7f** — the matrix found 7f, and 7f is why the
Flash-Next row below is green.

### 1. Text identity, pair OFF vs ON, one request in flight, 3 fresh processes per arm

Command: `baseline/shape-matrix.sh` (7 shapes; `$OUT/shape-matrix.tsv` has the machine-readable table).

| shape | np | text | run-stable (off / on) | hashes |
|---|---|---|---|---|
| 4B Q8_0, 1 GPU | 2 | **IDENTICAL** | 3/3 / 3/3 | `7bd542418112`, `6ce1f88dd5f7` |
| phi-4 IQ4_XS, 1 GPU | 2 | **IDENTICAL** | 3/3 / 3/3 | `997e53774f78`, `66075569e702` |
| 27B Q8_0, 3 GPU `-sm tensor -mg 0` | 2 | **IDENTICAL** | 3/3 / 3/3 | `f2753e13e419`, `c5f2fdc43642` |
| 35B-A3B Q8_0, 1 GPU `-ncmoe 40` (host tier) | 4 | **IDENTICAL** | 3/3 / 3/3 | `50f4872a9ab8`, `f124fe95ee72` |
| Flash-Next GSQ IQ3_S, 3 GPU `-sm tensor -mg 0` (qwen4exp) | 3 | **IDENTICAL** | 3/3 / 3/3 | `4dbb571b8bdb`, `f1c288a48245` |
| Gemma4 26B-A4B QAT (iswa), 1 GPU | 2 | **IDENTICAL** | 3/3 / 3/3 | `c67b75fa272d`, `e2159222cb60` |
| Gemma4 26B-A4B + its shared MTP head, `-ncmoe 20` | 2 | **REFUSED-AS-DESIGNED** | 1/1 (pair OFF) | `5d4a679e2b8d`, `b23288ddab15` |

The refusal row is a designed answer, not a gap: the server prints
`cannot run the two-arm server: a speculator's draft/verify round is not admitted on the pair yet (--spec-type
none); the layer-input families and a gemma4 assistant are refused permanently ...` and refuses to start with
`--prefill-decode-arms`, while the pair-OFF arm on that shape runs normally (it is the draft-aux-slot shape,
`STEP 7c`).

**Defect found by this row (STEP 7f, `steps/STEP-7f-0014-borrower-kv-types.patch`).**  The first matrix run read
`flashnext/on` as **1 of 3 runs completing** (r2 and r3 aborted with
`GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT)`).  Under `gdb` the stack was
`ggml_new_tensor_4d <- qwen4exp_qsa_sparse <- qwen4exp_want_derived_vis <- build_layer_attn <- graph <-
graph_reserve <- resolve_fused_ops` — i.e. the DECODE arm's fused-op probe.  Instrumenting both boundaries gave
the mechanism in one run: the context's **`cparams.type_k` was already garbage at the borrow branch**
(`DBG borrow cparams.type_k=22095`, and the value differed per run: 21900, 22095, ...).  STEP 2 assigned
`cparams.type_k`/`type_v` only in the own-memory branch, and `llama_cparams` declares them with **no default
initializer**, so a borrower carried uninitialized stack.  Fix: a borrower mirrors the OWNER's types.  Verified
5/5 clean afterwards, and the q8_0-KV arm reads the same text on both arms (`-ctk q8_0 -ctv q8_0`:
`18de3d2a3286`, `4dbb571b8bdb`), which is the check that the inherited type is the *cache's*.

### 2. What the pair costs (whole-server VRAM while loaded, `--vram-probe`)

| shape | VRAM delta | RSS delta | reserve / aux region |
|---|---|---|---|
| 4B | **+78 MiB** (gpu0) | +25 MiB | — |
| phi-4 | +53 MiB | +23 MiB | — |
| 27B 3 GPU tensor | **+149 MiB per device** | +1808 MiB | — |
| 35B-A3B host tier np 4 | +60 MiB | +4231 MiB | **declares 1280 MiB**; the slab pins `aux narrow-3` 1280 MiB from its unmapped VA spare |
| Flash-Next 3 GPU tensor | +154 MiB per device | +1823 MiB | — |
| Gemma4 26B-A4B iswa | +140 MiB | +34 MiB | — |

### 3. The ITL / prefill trade (`baseline/itl-arm.sh`, 4B Q8_0, 1 GPU; slot A decodes 64 tokens while slot B
prefills `prompts/prose-rdna-boosts.txt` = 5246 tokens; **3 repeats per arm**)

| `-ub` | ITL OFF (ms/token) | ITL ON | Δ | prefill OFF | prefill ON | Δ |
|---|---|---|---|---|---|---|
| 512 | 14.36 / 14.34 / 14.31 | 13.35 / 13.36 / 13.36 | **−6.8 %** | 4726 / 4732 / 4738 | 5693 / 5669 / 5667 | **+20.0 %** |
| 2048 | 20.45 / 20.32 / 20.37 | 15.16 / 17.00 / 15.06 | **−22.8 %** | 4889 / 4925 / 4913 | 6138 / 5986 / 6145 | **+24.1 %** |
| 4096 | 20.79 / 21.14 / 20.57 | 16.00 / 15.72 / 15.82 | **−23.9 %** | 5096 / 5010 / 5169 | 6068 / 6088 / 6081 | **+19.4 %** |

The win grows with the chunk width (the decoder stops waiting for a WHOLE chunk), and the prefill is faster on
every width: the decoder's single row no longer joins the prefill's batch, so the prefill graph stays wide.

**A measurement, not a rate, is what a single run gives here.**  One `gate-arms-itl.sh` run in this session read
`-ub 2048` as 1028.8 ms (OFF) → 961.9 ms (ON), i.e. −6.5 %, and a `-ub 4096` OFF arm as 1319.2 ms.  The repeats
above put OFF at 1300-1310 ms and 1316-1353 ms.  The single run's baseline was the outlier; the pair's own number
is stable (964-1088 ms).  A claim from one run of this gate would have been wrong in BOTH directions.


### 3b. The same trade on a TENSOR SPLIT (the `-sm tensor` shapes), and what it took to measure it

`baseline/itl-arm.sh`, np 2, `-c 8192`, slot A decodes 64 tokens while slot B prefills
`prompts/prose-rdna-boosts.txt` (5246 tokens).  Two repeats minimum per arm; every arm's repeats are tight.

**Flash-Next GSQ-RCO IQ3_S, 3 devices `-sm tensor -mg 0`** -- the shape where the feature matters most (a large
MoE whose prefill chunks are ~2 s):

| config | OFF (ms/token) | ON | Δ ITL | prefill OFF -> ON |
|---|---|---|---|---|
| delivery default (hipBLASLt on, graph capture on) | 53.85 / 54.46 / 54.00 (ub 4096) | **ABORTS** 2 of 3 | -- | -- |
| `GGML_CUDA_GRAPH_MAX_TOKENS=0` (capture off, hipBLASLt on) | 53.73 / 53.78 / 53.99 | **23.14 / 22.64** | **−57.5 %** | 1794/1790/1784 -> 2188/2169 (**+21.8 %**) |
| `ROCBLAS_USE_HIPBLASLT=0` (Lt off, capture on) | 47.60 / 47.91 / 47.95 | 41.25 / 41.01 / 41.33 | −13.8 % | 2054/2052/2069 -> 2265/2517/2519 (+18.3 %) |
| ... at `-ub 2048` (Lt off, capture on) | 47.97 / 47.69 / 48.26 | 42.33 / 41.34 / 38.16 | −15.3 % | 1847 -> 2378 (+28.8 %) |

**27B Q8_0, 3 devices `-sm tensor -mg 0`, `-ub 2048`, `ROCBLAS_USE_HIPBLASLT=0`:**

| OFF | ON | Δ ITL | prefill |
|---|---|---|---|
| 64.29 / 64.61 ms/token | 60.01 / 60.58 | **−6.4 %** | 1396 -> 1893 t/s (**+35.6 %**) |

So the ITL win is **shape-dependent** and it is the chunk wait it removes: 4B −23 % (its own decode is 14 ms),
Flash-Next **−57 %** (a ~2 s chunk against a ~23 ms decode), 27B only −6.4 % (its own decode is 64 ms, so the
same chunk wait is a smaller fraction).  **The prefill is faster on every shape** (+18…+36 %).

**Two defects this measurement found, both fixed in this landing:**

1. **STEP 7g — the Meta backend's split-state cache was per BUFFER, not per view slot.**  SEGV inside
   `ggml_backend_meta_get_split_state` on the decode arm's thread, 2 of 3 runs (`steps/STEP-7g-*.patch`).
   STEP 4 partitioned the view *containers* per slot and assumed the cache went with them; it did not.  It is
   invisible with one request in flight -- every sequential gate passed on the same binary.
2. **The delivery default still aborts on this shape: hipBLASLt inside a graph capture with a second context.**
   After 7g the remaining failure is `Hip error: 'operation would make the legacy stream depend on a capturing
   blocking stream' (906)` from `hipblaslt.cpp` during `ggml_cuda_graph_evaluate_and_capture` on the decode arm --
   the same family as STEP 7d's drop-vs-capture defect (a process-wide/library state vs a second arm's capture).
   **Disabling either hipBLASLt or the capture removes it**; capture-off is the better escape by far (23 vs 41
   ms/token), i.e. the pair's capture is worth ~18 ms/token of decoder latency on this shape.  Which one the pair
   should pin is a maintainer decision (`HANDOVER-2026-11-24-campaign.md` §7.7) -- a crash is not an acceptable default either way.

### 3c. The tensor-split blocker is FIXED (STEP 7h): the capture/hipBLASLt barrier, and the ITL with everything on

`steps/STEP-7h-0016-capture-blas-barrier.patch` (`~/llama.cpp` `f4540afd0`).  Flash-Next GSQ IQ3_S, 3 devices
`-sm tensor -mg 0`, np 2, `-c 8192`, `-ub 4096`, **delivery defaults** (hipBLASLt ON, graph capture ON), slot A
decoding 64 tokens while slot B prefills 5246:

| arm | runs | decoder ITL | prefill |
|---|---|---|---|
| pair OFF | 3 | 53.91 / 54.08 / 53.99 ms/token | 1788 / 1780 / 1784 t/s |
| pair ON, barrier (GEMM side shared) | 3 | **22.78 / 23.63 / 23.51** | 2160 / 2156 / 2172 t/s |
| pair ON, barrier (GEMM side exclusive) | 5 | 23.52 / 23.08 / 23.16 / **45.33** / 23.32 | 2159 / 2195 / 2154 / 2116 / 2173 t/s |

**Δ ITL −56.9 % (53.99 -> 23.31), prefill +21.7 %** -- the best configuration measured on this shape, with
neither feature disabled.  (The GEMM side may be shared: it reads the same numbers, so two arms' GEMMs overlap as
before and only creation and capture are exclusive.  The single 45.3 ms run is a scheduling outlier -- the
decoder lost its slot in that pass; the other four agree to 0.4 ms.)

**What the abort actually was, and the three wrong answers it took to find it.**  The evidence chain, because the
next reader will start where I did:

* `gdb` on the failing run put the abort in the DECODE arm inside `ggml_cuda_graph_evaluate_and_capture` ->
  `ggml_cuda_compute_forward` -> `ggml_cuda_op_swiglu` -> `unary_gated_cuda` -- a plain **kernel launch**, whose
  sticky error had been set *elsewhere*.  The thread that dies is not the thread that failed.
* Hypothesis 1 ("the other arm's GEMM collides with my capture") -> **killed**: making every library GEMM
  exclusive against the capture did not move the failure rate (2 of 3 either way).
* Hypothesis 2 ("hipBLASLt is being RECORDED into the capture") -> **killed**: an instrumented build logged
  **zero** BLAS calls inside a capture on this shape.
* Hypothesis 3, the answer: the library's **lazy handle creation** (`hipblasCreate` initialises hipBLASLt, which
  synchronises the LEGACY stream) -- the first GEMM of every (context, stream, device), i.e. exactly the call a
  freshly started decode arm makes while the prefill arm is recording.  Guarding creation took the shape from
  **2 of 3 aborts to 7 of 7 clean** (5 exclusive-GEMM + 3 shared-GEMM runs above, minus the one overlap).

The shape needs all three of {two arms overlapping, graph capture, hipBLASLt}: one request in flight never
overlaps two computes (which is why every sequential gate passed), `GGML_CUDA_GRAPH_MAX_TOKENS=0` is clean, and
`ROCBLAS_USE_HIPBLASLT=0` is clean.  `GGML_CUDA_CAPTURE_BLAS_BARRIER=0` restores the old behaviour (A/B and
bisect; a pair arm then needs one of those two pins).

**27B Q8_0, 3 devices `-sm tensor -mg 0`, at delivery defaults (hipBLASLt on, capture on, barrier on)** -- the
number the earlier table could only give with hipBLASLt OFF:

| `-ub` | pair OFF | pair ON | Δ ITL | prefill |
|---|---|---|---|---|
| 2048 | 64.41 / 64.43 / 64.68 ms/token (3 runs) | 58.98 / 61.43 / 58.52 | **−7.5 %** | 1396 -> 1834 t/s (**+31.4 %**) |
| 4096 | 64.12 / 63.93 (2 runs) | 61.00 / 60.85 | −4.8 % | 1571 -> 1964 t/s (**+25.1 %**) |

So the 27B reads essentially the same as its hipBLASLt-off measurement (−6.4 % then, −7.5 % now): this model's own
decode is 64 ms/token, so the chunk wait the pair removes is a small fraction of it -- while its PREFILL gains a
quarter.  The pair's ITL win is not a property of the feature, it is a property of `(chunk wait) / (own decode
cost)`; quote it per shape or not at all.

**Inertness of the barrier:** Flash-Next text identity unchanged (`4dbb571b8bdb` / `f1c288a48245`, 2 repeats per
arm, both arms), `gate-arms-default-inert.sh --shapes 4b-1gpu` PASS (startup shape still byte-identical to the
recorded one, so the barrier adds no log line), `gate-arms-itl.sh -ub 2048` PASS.

### 4. A gate correction this section owes: `gate-arms-itl.sh`'s text check is not a byte claim

The ITL gate is the ONE arm whose point is two requests in flight, and that is the configuration in which a byte
claim is a lottery (`INVARIANTS.md` r362-wip item 5): at `-ub 2048` one run read `36216e77db27` (off) vs
`68584f8dcc40` (on), and **three repeats per arm then read `36216e77db27` 6/6**; at `-ub 4096` the *OFF* arm
itself read `3c54ba65350c` where the 512/2048 arms read `36216e77db27`.  The gate now prints both hashes as
INFO and keeps the byte claim where it can be made (`gate-arms-n1-memory.sh`, one request in flight); its HARD
checks are the census, the ITL bound and the fault scan.

---

## 2026-11-23 — STEP 8: the two owed gates (`WORKLOG.md` 261123.022)

`gates/gate-arms-default-inert.sh` (new, pair OFF on four shapes) and `gates/gate-arms-recurrent-shared.sh`
(new, the probe promoted to a gate on two recurrent models), plus `baseline/inert-anchors.tsv` and
`baseline/inert-shapes/` (the expectations they compare against) and `--no-lock` on `baseline/mtp-axis.sh`.

### default-inert (`gate-arms-default-inert.sh`, no `-lv` on the cli arms; servers at their default verbosity)

| shape | cli text (prose, seed 42, temp 0) | anchor provenance | startup shape |
|---|---|---|---|
| 4B Q8_0, 1 device | **251 chars `be904e86f248`** | **PRE-CHANGE** STEP 0 A1 (r353 tree) | **13 lines == `baseline/A6-startup-shape.txt`** (PRE-CHANGE) |
| phi-4 IQ4_XS, 1 device | 272 chars `db5ab8e63e11` | recorded STEP 8 (post-change lock) | 13 lines == `inert-shapes/phi4-1gpu.shape` |
| 27B Q8_0, 3 devices `-sm tensor -mg 0`, `-n 96` | **399 chars `10a82b5559a9`** | **PRE-CHANGE** STEP 0 A2 | 23 lines == `inert-shapes/27b-3gpu-tensor.shape` |
| 35B-A3B Q8_0, 1 device `-ncmoe 40`, `MOE_EXPERT_HEATMAP=load` | **262 chars `e9db5cb8db0b`** | **PRE-CHANGE** STEP 0 A3 | 20 lines == `inert-shapes/35b-a3b-host.shape` |

Non-vacuity (printed by the gate):
* the cli text is **run-stable** over 2 fresh processes on the sensitivity shape (`be904e86f248` twice);
* the SAME server with the pair ON prints a **different** shape and it is the recorded pair-ON shape
  (`inert-shapes/4b-1gpu.pair-on.shape`): the delta is exactly 4 lines -- the two `resolve_fused_ops` WARNs the
  second context resolves for itself, the `two-arm server: ... ONE memory ...` INFO, and
  `arms_start: decode arm started (own batch of 2 rows)`;
* the pair-ON responses are the pair-OFF responses as a **set** (`{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`),
  which is the A6 anchor -- the pair is a latency change, not an answer change.

Server shape lines are cut at `listening on` and normalised by `baseline/norm-log.sh`; the 4B shape's
expectation IS the STEP 0 file (no copy), the others carry a `# recorded` header the gate ignores.

### recurrent-shared (`gate-arms-recurrent-shared.sh --n 600`; BOTH models' four-axis arms on a **3-device
`-sm tensor -mg 0`** split -- the campaign's validation shape and the shape STEP 7e broke.  `AXIS_SHAPE=host`
switches to the delivery's host-expert arm for the external head)

| model | §1 placements (non-vacuity) | §2 concurrent vs `--solo` | §3 four-axis `none` == adaptive |
|---|---|---|---|
| 35B-A3B Q8_0 (embedded MTP, `-sm tensor`) | `A: head=0 n=1 rs_z=0 \| B: head=1 n=1 rs_z=1` | `arm_a=9eaff8b3b77dc104 arm_b=b17fc93ec1b442e4` **identical, round_ms=822** | R `8a8d0e7e2055` (182 drafts/153 acc), K `5ebd156073f6` (73/73), P `a7a7307e59bf` (168/146), C `3512551cd3c9` (123/115) -- **plain == drafter on all four** |
| Flash-Next GSQ-RCO IQ3_S + shared head (qwen4exp), **3 devices `-sm tensor -mg 0`, `-c 16384 -md mtp-...-shared-Q8_0`** | `A: head=0 n=1 rs_z=0 \| B: head=1 n=1 rs_z=1` | `arm_a=9eaff8b3b77dc104 arm_b=8d9f63064b455c95` **identical, round_ms=1183** | R `76ff3a7f59bb` (213/158, 104 t/s), K `a87c4318b649` (75/74, 167 t/s), P `b297abb3126d` (140/125, 133 t/s), C `82519c1d9ee1` (107/103, 153 t/s) -- **plain == drafter on all four** |

(The same Flash-Next four axes were also measured on the host-expert shape `1 device -sm layer -ncmoe 48 -c 8192`
-- R `f0d0bce6eb50` 219/163, K `a87c4318b649` 75/74, P `a738bc5e3b57` 150/131, C `904de4851b41` 106/102 -- so
both shapes are green; `--axis-shape host` is the arm to use when the 3-device split is out of memory.)

The gate refuses a model with no recurrent module (`PASS this model HAS a recurrent module`), so the
placement claim is never asserted against nothing.  `arm_a`'s digest is **model-insensitive** by construction
(it is the greedy continuation of a repeated-text prefill, and 27B/35B/Flash-Next all read
`9eaff8b3b77dc104`); `arm_b`'s is the discriminator, and the comparison that matters is concurrent vs `--solo`
**within one model**, which is what the gate does.

Two vacuous arms were caught by the gate's own non-vacuity lines, and BOTH were the harness's fault, not the
library's:
* the first version ran the Flash-Next adaptive arms at `--spec-draft-n-max 2` (copied from
  `scripts/gate-draft-ctx.sh`, whose `draft-mtp` arm is NOT adaptive).  The adaptive controller's
  `n_min_adaptive` default is **3** and it landed WITH that validation in delivery **block 02**
  (`git log -S n_min_adaptive`), so the pair is self-contradictory and has never been a working configuration:
  the delivery says so loudly (`E failed to initialize speculative decoding context: invalid adaptive draft
  range: n_min_adaptive=3, n_max=2`) and continues as a PLAIN run, 0 drafts.  Checked on a build of the delivery
  tip `089f90067` with the identical invocation: the same error, the same 0 drafts -- the campaign neither
  introduced nor changed it, and no delivery change is proposed: the flag pair is invalid, the feature is fine
  (the same gate's `--nmax 7` arms on the same shapes draft and accept, 73..213 drafts per axis).  The gate now
  FAILs on 0 drafts and names the initialisation error, so a harness bug of this shape cannot read as a pass.
* the first version held the exclusive GPU flock for its whole run AND called `mtp-axis.sh`, which takes the
  same lock: the box sat idle with `flock -w 900 9` waiting on the gate's own fd (5+ minutes).  Hence
  `mtp-axis.sh --no-lock` (documented as "the caller already holds the lock").

---

## 2026-11-23 — STEP 7e: STEP 4 broke the EXTERNAL DRAFT HEAD on a tensor split, and no gate was shaped to see it (`WORKLOG.md` 261123.021)

Found while choosing the recurrent gate's shapes, not by a gate: the campaign build aborts where a build of the
delivery tip runs.

| arm (Qwen3.8 Flash-Next GSQ-RCO IQ3_S + `mtp-...-shared-Q8_0`, 3 devices `-sm tensor -mg 0`, `-c 16384 --load-mode mmap --spec-type draft-mtp`, `-p "The capital of France is"`) | result |
|---|---|
| build of the delivery tip `089f90067` (a reference build in `~/llama.cpp/build-ref`) | **86 chars `be69e274b66d`** |
| campaign tip `16131c077` (7c) | **rc=134**: `ggml/src/ggml-backend.cpp:789: GGML_ASSERT(tensor) failed` in `ggml_backend_tensor_set_async`, after `sched_graph_realloc: nodes = 5538 (changed = 1), n_diff = 2` |
| the same command with `--load-mode none` instead of `mmap` | rc=134 (so the load mode is not the trigger) |
| campaign STEP 5 `25b73cfaf` | rc=134 |
| campaign STEP 3 `16c08ba70` | **rc=0** -- so the bisect lands on 4 |
| campaign STEP 4 `23552dea2` | rc=134 |
| STEP 7e fix (`3013e880a`) | **86 chars `be69e274b66d`** == the delivery's text |

Mechanism: STEP 4 bound the `thread_local` Meta view slot only in `sched_reserve()` and in `sched_alt_select()`,
and `sched_alt_select()` returns early when the requested state already is the current state -- so the ordinary
`llama_process()` path bound nothing and the slot stayed whatever the last context on that thread had bound.
`scripts/gate-draft-ctx.sh` (which owns the external head) did not see it because its arms are 1 device
`-sm layer`, where the Meta backend is not in the path; the Flash-Next `-sm tensor` arm it documents as
negative-tested at r348 is the shape that broke.

Re-verified after the fix (all green, in this order): the reproducer (`be69e274b66d`), `scripts/gate-draft-ctx.sh`
--quick (PASS 11 checks, `e7aae6034b4f`, acceptance 0.81429), the pair on 27B 3-device `-sm tensor`
(`gate-arms-n1-memory.sh`: identical text `8f396b237f0f`/`04116b9a99ad` both arms, 31 steps/1 handover,
0 of 8192 cells reserved), the 4-slot host-tier soak (`gate-arms-multislot.sh`: 20 rounds, 80 requests,
0 errors, 8 steps/3 handovers/widest 3, aux `narrow-3` pinned), `gate-arms-default-inert.sh` (4/4 shapes) and
`gate-arms-recurrent-shared.sh` (both models).

---

## 2026-11-23 — STEP 7c (retaken on top of 7d): the aux reserve is an approximation, and the slab now **tells the operator the number to set** (`WORKLOG.md` 261123.020)

7c was rejected before 7d because it changed table placement and thereby exposed the drop/capture race; with the
race gone it does what it was meant to, plus it exposed a residual defect in **7a's own** accounting.

### Part 1 — the reserve is charged up front (approximation), and calibrated by a one-time log

`COMMON_ARMS_RESERVE_MIB` 256 -> **1280** (measured: 1280 MiB on 35B-A3B `-ncmoe 40` np 4; 512 MiB per aux
context on Gemma4 26B-A4B + MTP).  The slab then maps the aux region from the **unmapped VA spare** instead of
evicting arena tables:

```
35B-A3B Q8_0 -ncmoe 40, np 4, before:  aux narrow-3 pinned at [26.31, 27.56] GiB (1.25 GiB, taken from the arena)
                                       1 x "arena alloc failed"
                              after:   SLAB_INIT ... reserve=2.75 GiB ... va_spare=1.25 GiB
                                       aux narrow-3 pinned at [26.31, 27.56] GiB (1.25 GiB, unmapped VA spare)
                                       0 x "arena alloc failed"
```

and the slab logs the calibration **once per slot**, with the exact numbers:

```
the aux narrow-3 region (1280 MiB) came from the unmapped VA spare -- no arena table was evicted.
The reserve in force is 2816 MiB; 2816 MiB would host every auxiliary region (headroom 1536 + aux 1280)
```

### Part 2 — the residual 7a defect this exposed: `mapped` conflated the arena with a sibling region

The Gemma4 arm (two aux contexts, 512 MiB each) still evicted on the second one and claimed **1024 MiB** with
`arena_before` == its own base.  Cause: `s.mapped` is the slab's VMM extent, and pinning region 0 ABOVE the arena
grows it -- so the second pin's "is the arena mapped over my window?" test read the *slab*, not the arena.  Fixed
with `s.mapped_arena` (the arena's own mapped extent: set at init, grown by `slab_extend`, never moved by an aux
pin); `ggml_cuda_slab_arena_top()` and the pin's eviction test now read it.

| arm | before the `mapped_arena` fix | after |
|---|---|---|
| Gemma4 26B-A4B + MTP, `GGML_CUDA_SLAB_RESERVE_MIB=2560` | slot 0 from spare, slot 1 "taken from the arena (1024 MiB evicted)" | **both from the unmapped VA spare**, zero eviction |

### Part 3 — the calibration loop, end to end (the point of the whole exercise)

On the Gemma4 shape the *default* reserve is the draft's own derived 2496 MiB, which does not cover both aux
contexts, so the run logs:

```
the aux narrow-3 region (512 MiB) had to TAKE arena tables (1024 MiB evicted) because the reserve in force
(2496 MiB) left only 960 MiB of unmapped VA spare.  Set GGML_CUDA_SLAB_RESERVE_MIB=2560 for a run that maps it
from the spare instead (headroom 1536 + aux 1024)
```

Setting **2560** on the next run gives both regions from the spare, zero eviction, and the same text
(`a195d3d6953c`).  That is the loop the maintainer asked for: approximate now, measure at startup, tell the
operator the exact value for next time.

### Verification (all on the final tree)

| arm | result |
|---|---|
| 35B-A3B `-ncmoe 40` np 4 soak, pair ON | **PASS 7/7** (5 runs on the retake, 2 after the `mapped_arena` fix), 0 `arena alloc failed` (7a: 1), region from the spare every run |
| Gemma4 26B-A4B `-ncmoe 20` + `-md` MTP head, reserve 2560 | both regions from the spare, 0 eviction, `none` == `draft-mtp` == `a195d3d6953c` |
| 4B soak (pair, no slab) | **PASS**, widest batch 4 |
| `gate-arms-n1-memory.sh` / `gate-arms-itl.sh` | **PASS**; text identical; **16.33 -> 15.09 ms/token** on the pair arm (1044.8 -> 965.5 ms / 64 tokens), prefill 5191 -> 6165 t/s |
| A1 / A3 | `be904e86f248` / `e9db5cb8db0b` unchanged (the pair's charge is inert with the pair off) |

---

## 2026-11-23 — STEP 7d: **the host-tier soak was INTERMITTENT, not green** — the prefill arm's compute-buffer drop races the decode arm's graph capture (`WORKLOG.md` 261123.019)

**A correction first.**  STEP 7a's host-tier PASSes were *lucky runs*.  The same 7a binary (`e460f0dea`, tree
clean) re-measured later gives:

| when | prior | result |
|---|---|---|
| 16:33-16:45 (STEP 7a session) | whatever the persisted heat map held | **PASS 3/3** |
| 17:0x, same binary | the map rewritten by those runs (default = load+**save**) | **FAIL 3/3** |
| 17:1x, same binary | `MOE_EXPERT_HEATMAP=load` (pinned) | **FAIL 1, PASS 1** |

So the host-tier shape fails ~half the time in the current state, and STEP 7a's aliasing fix is orthogonal to it
(that was deterministic: 2/2 repro before, clean after).

### The defect: the drop + arena re-arm vs the other arm's capture

The abort is always the same shape, and the line immediately before it is the re-arm:

```
0.19.024 I compute buffer 1280.00 MiB from the slab work region (boundary 1536.00 MiB)   <- prefill->decode drop
0.19.026 W moe_cache_rearm: re-armed 15 stood-down expert-cache tables after a compute-buffer drop (...)
0.19.026 E ROCm error: operation failed due to a previous error during capture
            in ggml_cuda_kernel_launch
0.19.058 E ROCm error: operation would make the legacy stream depend on a capturing blocking stream
            in ggml_cuda_async_input_batch_locked          <- consequence
```

`LLAMA_DROP_COMPUTE_BUFFERS=0` (pair ON, prior pinned) → **3/3 PASS, 0 capture errors**.  The drop is a
**process-wide memory mutation** (it moves the slab's work boundary and re-arms MoE arena tables) issued from the
prefill arm while the decode arm is inside `llama_process()` capturing its graph -- DESIGN §8 invariant 4's
subject, and invisible to a text/hash oracle.

### The fix (STEP 7d): the pair keeps both layouts reserved

`llama_context_disable_compute_buffer_drop()` (new, 32 lines over 4 files) + the server calls it for **both** arms
when the pair is created.  Neutral: only the pair calls it; every other run keeps the drop.

| arm | result |
|---|---|
| host-tier soak, pair ON, prior pinned | **PASS 5/5**, 0 capture errors, decode arm active (8 steps / 3 handovers / widest 3-4) |
| the same, cost of not dropping | arena **25537 MiB** and expert hit-rate **1.0** -- **identical** to the drop-on arm (the slab reclaims the wide view on the next wide pass) |
| A1 / A3 (neutrality) | `be904e86f248` / `e9db5cb8db0b` -- unchanged |
| 4B soak, pair ON | **PASS**, widest batch 4 |
| `gate-arms-n1-memory.sh` | **PASS**, text byte-identical, 0 of 8192 cells reserved |
| `gate-arms-itl.sh` | **PASS**, same text, **1311.4 -> 960.4 ms / 64 tokens (-27 %)** |

### Harness change this forced

`gates/gate-arms-multislot.sh` now pins `MOE_EXPERT_HEATMAP=load`.  The default **writes** the persisted map, so
two soaks in a row start from different residency priors -- which is what flipped this gate's own outcome between
the 7a session and this one (`GREEDY-PURITY.md` §89, `AGENTS.md` rule 5).

### Still owed after 7d

The *proper* fix for the underlying hazard is not "never drop": it is a barrier so a drop/re-arm can run while the
other arm is **not** inside `llama_process()` (DESIGN §8 invariants 3-4).  7d removes the exposure on the pair;
any future second-context design that wants the drop back needs that barrier.  Also unmeasured: whether the
drop's absence costs arena at STEP 10 (a drafter holds two more compute layouts).

---

## 2026-11-23 — STEP 7b: the aux compute buffer allocated **outside** the slab was measured and REJECTED (`WORKLOG.md` 261123.018)

The maintainer's question — "wouldn't it be better to allocate these out of anonymous memory *before* the slab
takes the VRAM?" — was implemented as a 3-file change (`aux_alloc` prefers `ggml_cuda_device_malloc` outside the
slab and falls back to the pinned region) and measured.  **It fails, and the reason is structural, so the code was
reverted and 7a (the pinned region) stands.**

| arms (35B-A3B Q8_0 `-ncmoe 40`, np 4, 20 rounds) | aux buffer placement | result |
|---|---|---|
| 7a (shipped) | pinned region, **arena** -1.25 GiB | **PASS**, `1` `arena alloc failed` line, 0 errors |
| 7b | ANONYMOUS, **outside cushion** -1.25 GiB | **ABORTS**: `compute buffer 1280.00 MiB ANONYMOUS (outside the slab)` at 7.6 s, then **42** `alloc_table_locked: arena alloc failed`, then `ROCm error: operation failed due to a previous error during capture` and `ggml_abort`; every request fails |
| control: `GGML_CUDA_SLAB_NARROW2=0` (**no 7b code**, same placement — the delivery's old "quick fix") | ANONYMOUS | **identical abort** (42 `arena alloc failed`) — so this is a property of the placement, not of the 7b code |

**Why.**  The VRAM outside the slab (`GGML_CUDA_SLAB_HEADROOM_MIB`, 1536 MiB here) is not spare capacity: it is
what `slab_extend` **maps the arena's physical from**.  Spending 1280 MiB of it on a compute buffer leaves ~256
MiB, so the arena cannot grow, tables cannot be allocated, the cache disables them — and with the pair's second
graph in flight the run dies.  The arena and the outside cushion are therefore **not interchangeable**: 1.25 GiB
costs the same VRAM either way, but only the arena route leaves the slab able to grow.

**What this retires.**  The old `GGML_CUDA_SLAB_NARROW2=0` arm ("the session's quick fix") is not a working
configuration on a slab-armed host-tier shape: it aborts, as the control shows.  And the deferred-owner-reserve
route (owner's first reserve postponed until the aux contexts exist) would put the aux buffer in the same wrong
budget, so it is not the way to get this either.

**So the answer to the maintainer's question is: the idea is right about the *source* but the cushion is the wrong
one.**  The aux buffer must come out of the arena (7a), and the only thing that could make "before the slab" work
is a bigger *reservation* (not headroom) — i.e. `GGML_CUDA_SLAB_RESERVE_MIB` raised by the aux buffers, which
shrinks the slab at creation and is therefore the same arena loss, one eviction earlier.

---

## 2026-11-23 — STEP 7a (= TASK 1): the host-tier fault is the SLAB WORK ALIAS, and each extra context gets its own pinned region (`WORKLOG.md` 261123.017)

12 files (+263/-227): `ggml-cuda-slab.cu` (the aux-region stack), the three iface/API renames, `llama-context.{h,cpp}`
(the per-context slot), and the `-ncmoe` refusal removed from `common/common.cpp`.

### The defect (root-caused, not the MoE remap)

`GGML_CUDA_WORK_ALLOC_DEBUG=1` on the faulting shape shows **two live compute buffers at the same slab work
base**:

```
0.07.583 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB ...            <- the PREFILL arm's view
0.07.586 W WORK_RELEASE dev=0 off=0.0 MiB need=910.8 MiB                            <- its old (smaller) view
0.07.589 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB ...            <- the DECODE arm's view, SAME base
0.08.683 W WORK_ALLOC dev=0 need=1280.0 MiB narrow=1 ret_off=0.0 MiB ...            <- the prefill arm re-takes base 0
0.08.686 HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION in rms_norm_mul_rope_f32_qk_pair / _f32 (the K rope's
         SET_ROWS: its I64 row-index tensor is one of the aliased activations)
```

Two contexts whose compute buffers are the SAME memory: one arm's prefill writes the other arm's
activations, and the K rope then writes KV rows at a garbage index.  The slab's work allocator has exactly one
"narrow" base (offset 0) and assumed a single live narrow view per device; the MTP draft had the same defect
and got the `narrow2` region for it (its own comment: "without this they both aliased at base 0"), so this is
that defect returning for a *third* context.  **The MoE expert remap is NOT involved** -- the arena evictions in
the log are the boundary moves of a second view, not the cause.

### The rungs that refuted the old hypothesis (cheap to re-run)

| rung | env | result |
|---|---|---|
| 0 repro | — | FAULT at ~9 s: `rms_norm_mul_rope_f32_qk_pair<false,float,__half,0,0>` (the **plain** pair) |
| 1 Q/K deferral off | `GGML_CUDA_ROPE_QK_PAIR=0 GGML_CUDA_QK_PAIR_FWHT=0` | **still FAULTS**, now in the *unfused* `rms_norm_mul_rope_f32<256,false,__half,0,0,...>` — the deferral is not the mechanism (and `GGML_CUDA_QK_PAIR_FWHT` alone would not have disabled this instantiation: `fwht_n=0`) |
| 2 aux routing, process-wide flag | temp `is_draft \| memory_borrowed` | server aborted at load: the **prefill** arm inherited the decode arm's flag, was routed into the aux region, found it live, fell back to `cudaMalloc` (1280 MiB OOM).  A second, real defect: the routing flag was process-wide while the arms reserve on two threads |
| 3 the fix | — | **PASS** (below) |

### The fix

* the `narrow2` region becomes a **stack of N=2 pinned aux regions** at the top of the reserved VA
  (slot 0 = a draft context, slot 1 = a context that BORROWS another context's memory), the arena ceiling is
  the lowest pinned offset, and the commit takes the pinned offset as an argument (it must not be recomputed:
  the pin releases the lock to evict, so another context may pin in that window);
* the routing is a **thread_local slot derived from the allocating context's own identity**
  (`slab_aux_slot()`: `ctx_other` -> 1, `memory_borrowed` -> 2), not a process-wide flag — the same pattern as
  STEP 4's Meta view slot;
* `GGML_CUDA_SLAB_NARROW2=0` still bypasses the slab for such a context (unchanged A/B arm).

### Green, with the number

| arm | shape | result |
|---|---|---|
| host-tier soak, pair ON | 35B-A3B Q8_0 `-ncmoe 40`, 1 GPU, np 4, 20 rounds, 80 requests | **PASS**: 0 errors, 15 steps / 3 handovers / widest batch 3, `aux narrow-3 (borrowing-context compute) pinned at [26.31, 27.56] GiB`, journal clean, **wall 28 s** (the faulting runs took 371-382 s) |
| the same, pair OFF (the control) | same shape | **PASS**, 80 requests, 0 errors, no aux region pinned (inert) |
| the same, pair OFF, `-ncmoe` REMOVED from the refusal | — | the refusal is gone; the predicate no longer names the shape |
| 4B soak, pair ON | 4B Q8_0, 4 slots, 20 rounds | **PASS**: 15 steps / 4 handovers / widest batch **4** (the slab is not armed here, so the aux path is a no-op) |
| `gate-arms-n1-memory.sh` | 4B, 2 slots, 1 request in flight | **PASS**: ONE memory, text byte-identical (`01c9cbe2e895` / `5e69982c7d94` both arms), 31 steps, 0 of 8192 cells reserved |
| `gate-arms-itl.sh` | 4B, 5246-token prefill beside 64 decoded tokens, `-ub 2048` | **PASS**: same text `36216e77db27`; **20.37 -> 15.04 ms/token (-26 %)**, prefill **4908 -> 6141 t/s (+25 %)** |
| A1 / A2 / A3 (inertness) | 4B / 27B-3GPU-tensor / 35B-A3B host | **unchanged**: `be904e86f248`, `10a82b5559a9`, `e9db5cb8db0b` |
| A4 prefill-logit | 27B Q8_0, default point | **KLD 0.000000 / same-top-p 100.000 %**, clean |
| A5, K axis, `--spec adaptive` | 27B-3GPU-tensor | **anchor reproduced**: `a87c4318b649`, 151.4 t/s, acc 0.98649/0.97241, pos-1 0.986 |
| A6 startup shape | 4B, 2 slots `--kv-unified` | **13 lines byte-identical** to `baseline/A6-startup-shape.txt`; responses `{7bd542418112, 6ce1f88dd5f7}` |
| `scripts/check-shlib-symbols.sh` | the `all` build | **OK** (16 libs, all symbols resolved) — the iface rename is consistent across cuda/cpu/rpc/meta |

### Two harness notes

* the multislot gate now takes `ARMS=` (default `--prefill-decode-arms`) so the **pair-OFF control** runs the
  same soak on the same shape, and it FAILS if the slab is armed and the decode arm did **not** get a pinned
  region (the non-vacuity check for this fix);
* "empty completion" in that gate counts `content` only: a reasoning model that emits 16 thinking tokens
  returns `reasoning_content` and an empty `content` (measured: pair ON 13-16 empty, pair OFF 20 empty, same
  shape, all on slot 0 = `reasoning.txt`).  Not a pair property, and not a failure.

### The DRAFT slot (0), closed the same day — an external MTP head on a slab-armed shape

The gap named below was closable immediately: a **separate MTP head GGUF** (`-md`, arch `gemma4-assistant`) on a
**MoE target with host experts** has a draft context with `ctx_other` *and* an armed slab, and it exercises
**both** aux slots in one run (the head's own context is the draft, and its target-side context borrows — the
`ctx_memory` field the pair uses):

```
shape: Gemma4 26B-A4B Q8_0 (MoE) + mtp-gemma-4-26B-A4B-it.gguf, -ncmoe 20, 1 GPU, f16 KV, -c 8192
  aux narrow-2 (draft compute)           pinned at [18.81, 19.31] GiB (0.50 GiB, unmapped VA spare)
  aux narrow-3 (borrowing-context compute) pinned at [18.31, 19.31] GiB (0.50 GiB, taken from the arena (tables evicted))
  -> the arena ceiling moves 17.94 -> 17.44 GiB; the two regions stack downward, in pin order
```

**Text (the four-axis purity rule, on a slab-armed shape):** `--spec-type none` and `--spec-type draft-mtp`
read **byte-identical** — `481 chars sha=a195d3d6953c` — and `none` reproduced `a195d3d6953c` on three
consecutive cold runs (`MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0`, `--temp 0 --seed 42`), so this
shape IS text-stable (unlike the DFlash2-on-35B-A3B arm below).  Wall: ~35 s per arm.

**Both slots are therefore green:** slot 1 (the pair's decode arm) by the host-tier soak, slot 0 (a draft
context) by this arm, and the two in one run by this arm as well.

### Not covered here (recorded as a gap)

*(Superseded by the subsection above: the draft slot WAS exercised the same day.)*  The **draft** slot (0) was
originally not exercised on a slab-armed shape: an embedded-MTP arm (`--spec-type
draft-mtp-adaptive`) allocates no second compute buffer, and a `--spec-type draft-dflash2` arm on the 35B-A3B
host tier ran (`none` == `draft-dflash2` at 1961/1839 chars, but the *text* is not run-stable there: two
builds x two arms gave two different hashes per arm at identical lengths — the host tier's expert residency
picks the kernel set, `GREEDY-PURITY.md` §89).  Slot 0's code path is identical to slot 1's (it ran), but its
*routing* (`ctx_other` -> slot 1) needs an external-head/assistant arm with `-ncmoe` to be called green.

---

## 2026-11-23 — STEP 6e: reserve, census, and the three gates (`WORKLOG.md` 261123.016)

2 files in the tree (+52/-3) plus three gate scripts.

| gate / arm | shape | result |
|---|---|---|
| `gates/gate-arms-n1-memory.sh` | 4B Q8_0, 2 slots `--kv-unified`, 1 request in flight | **PASS**: ONE memory line; text byte-identical to pair-off; `31 step(s), 1 handover(s)`; reservation census **0 of 8192** (HARD); journal clean |
| `gates/gate-arms-itl.sh` | 4B, 5246-token prefill on B while A decodes 64 tokens, `-ub 2048` | **PASS**: same text (`36216e77db27`); A 1028 -> 959 ms / 64 tokens with the pair on (the 6d-2 run: 20.27 -> 15.68 ms/token, -23 %) |
| `gates/gate-arms-multislot.sh` | 4B Q8_0, 4 slots, **20 rounds, 80 requests** | **PASS**: 0 errors, 0 aborts, 4 handovers, **widest batch 4** (two decoders in one decode batch) |
| … same gate, 35B-A3B Q8_0 `-ncmoe 40`, np 4 | host expert tier, 1 GPU | **GPU FAULT**: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `rms_norm_mul_rope_f32_qk_pair`, server aborted after 398 s (80 requests had completed) |
| reserve charge | 35B-A3B `-ncmoe 40` np=2 vs np=1 | `reserves 256 MiB` + `MoE arena reserve = 256 MiB in total (draft 0 + decode arm 256)` vs `reserves NOTHING` |
| the reserve's measurement | 27B Q8_0 3 GPU tensor np=2, VRAM before/after one tiny request | **+157 MiB** devices 0/1, **+235 MiB** device 2 → constant 256 |

### The finding (open)

The pair must NOT overlap on a shape where the prefill arm's step remaps process-wide device state.  A 4-slot
soak on the host-expert tier faults the GPU; `-ncmoe > 0` is now refused by `common_arms_pair_possible()` and
the gate reproduces it.  Two ways out, both a maintainer decision: a real decode-band mutual exclusion (which
makes the pair inert exactly on the shapes it was measured on), or making the host-tier expert remap
context-safe.

### Harness rule learned here (kept)

A census printed from a destructor disappears if the server is SIGKILLed -- and `timeout` inside the harness made
it worse (`timeout` forwards the TERM, so the server saw a "second interrupt" and skipped teardown entirely).
`server-arm.sh` now starts the server without an inner `timeout`, sends ONE signal, and waits for the process to
exit; the reservation census moved to `fprintf(stderr)`.

---

## 2026-11-23 — STEP 6d-2/6e/6f: the decode arm runs (`WORKLOG.md` 261123.015)

2 files, +331/-3 (`tools/server/server-context.cpp`, one refusal in `common/common.cpp`).

| arm | shape | result |
|---|---|---|
| text identity | 4B Q8_0 1 GPU, 2 slots `--kv-unified`, 2 concurrent, pair OFF vs ON | **identical** `7bd542418112` / `6ce1f88dd5f7` |
| census (mechanism) | same, pair ON | `the decode arm is generating: 47 step(s), 2 handover(s), widest batch 2 slot(s)` |
| text identity, tensor split | 27B Q8_0 3 GPU `-sm tensor -mg 0`, **1 request in flight** | **identical** `f2753e13e419` / `c5f2fdc43642` |
| concurrency on that shape | same, 2 at once | slot 0 identical; slot 1 differed once (`f783d2941fbe`, 178 ch vs 150).  Not the pair: the same slot varies run-to-run on the unmodified delivery under two concurrent requests (Flash-Next: `b2e646545bd9` once, `c60b5a919c9b` twice) — composition picks the kernels (V8) |
| **ITL through a prefill** (`baseline/itl-arm.sh`; A = 5-token prompt, 64 tokens; B = 5246-token prompt; `-ub 2048`) | pair OFF | A **20.27 ms/token**; B **4935 t/s** |
| | pair ON | A **15.68 ms/token (−23 %)**; B **6147 t/s (+25 %)**; text **identical** (`36216e77db27` / `e8279ccca426`) |
| same at `-ub 512` | OFF vs ON | 13.25 vs 13.16 ms/token — flat; the decoder never waits for a whole chunk at that width on this model.  A gate must report the ubatch |
| `gate-park-idle-slots.sh` | pair OFF | **PASS** (6 park + 2 unpark traced) |
| A1 | 4B 1 GPU | `be904e86f248` |
| GPU faults | whole window | clean |

### Two bugs the oracle caught

1. **The handover ran with `ctx_dec == nullptr`** → every generating slot left the prefill arm's batch with no
   arm to run it → the server spun at **94 % CPU** with no output.  Localised by bisecting the campaign commits
   (base + STEP 1-5 + 6a + 6d-1 all pass the repro; the working tree failed), then `gdb` on the live process.
2. **`post_decode` read a handed-over slot's `i_batch`** → samples the wrong row, or one with no logits
   (`GGML_ASSERT(logits != nullptr)`).  Both post_decode iterates now check the arm first.

### Still owed for STEP 6

The arena reserve for the decode arm (R6) measured and charged, the 4-slot × 20-round soak, and `gate-arms-itl.sh`
as a scripted gate (the numbers above are from `baseline/itl-arm.sh`, one arm at a time).

---

## 2026-11-23 — STEP 6d-1: `slot_step()` extracted from `post_decode()` (`WORKLOG.md` 261123.014)

1 file, +66/-42 (`tools/server/server-context.cpp`).  Inert by construction: the only caller is
`post_decode()`, with the same arguments it used to compute inline.

| arm | shape | result |
|---|---|---|
| pair OFF | 4B Q8_0, 2 slots `--kv-unified`, 2 concurrent | `7bd542418112` / `6ce1f88dd5f7`; startup shape **byte-identical** to the anchor |
| pair ON | same | `7bd542418112` / `6ce1f88dd5f7` |
| pair OFF | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, 2 slots | `f2753e13e419` / `c5f2fdc43642` (unchanged from 6a/6c) |
| `gate-park-idle-slots.sh` | Flash-Next host tier, 3 arms | **PASS** — 6 park + 2 unpark moves traced, journal clean |
| GPU faults | since the landing start | clean |

`slot_step(slot, ctx, tok_idx)` — `ctx` is the context that RAN the batch, which is what lets the decode arm
reuse it unchanged.  The speculative verify path keeps its own inline emit loop (n_accepted tokens with
per-position probs); STEP 10 owns that.

**Next landing (6d-2/6e/6f, deliberately not split further):** `decode_arm_loop()` + `slot.arm` ownership +
the handover in `post_decode()` + `arm_step_guard()`.  A loop with nothing routed to it has no oracle, and a
handover without the loop leaves a slot that stops generating, so they land together with the oracle that can
judge them: text identical to the single-context server, the cell census with `reserved == 0` at the request
boundary, and the ITL through a 2-5 s prefill (`gate-arms-itl.sh`).

---

## 2026-11-23 — STEP 6a/6c: the pairing predicate + the decode arm's context, INERT (`WORKLOG.md` 261123.013)

4 files, +160 (`common/common.{h,cpp}`, `common/arg.cpp`, `tools/server/server-context.cpp`).
Nothing is routed to `ctx_dec` in this landing, so the pair must change no token.

| arm | shape | result |
|---|---|---|
| pair OFF | 4B Q8_0, 2 slots `--kv-unified`, 2 concurrent greedy | `7bd542418112` / `6ce1f88dd5f7`; startup shape **byte-identical** to the (widened) anchor |
| pair ON | same | text **identical**; shape delta = **exactly** 2 `resolve_fused_ops` WARNs + the `two-arm server: ... ONE memory ...` INFO |
| pair ON | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, 2 slots | `f2753e13e419` / `c5f2fdc43642`, identical to the pair-off arm |
| `on`, `-np 1` | — | refuses to start: "it needs at least 2 slots (--parallel >= 2)" |
| `on`, `--spec-type draft-mtp` | 27B | refuses to start: the speculator rule (STEP 7/10) |
| `auto`, `-np 1` | — | warns, continues single-context |
| `--prefill-decode-arms-mode bogus` | — | throws at parse time with the three legal values |
| GPU faults | journal since the landing start | clean |

The pair line, verbatim:

```
two-arm server: prefill arm 0x…2280 with n_batch/n_ubatch 2048/512, decode arm 0x…3060 with 2/2;
ONE memory 0x…4270 (the decode arm borrows it, nothing is copied)
```

**Instrument fix in the same landing:** `baseline/norm-log.sh` had been dropping the library's single-tag log
lines (`llama_context:`, `llama_kv_cache:`, `resolve_fused_ops:`), so the startup-shape anchor could not see a
cache-geometry or memory-identity change.  It keeps both forms now (addresses -> `PTR`) and is cut at
`listening on`; the anchor was re-baselined from the pair-off arm of a build whose text and KLD still match
STEP 0.

**Not yet charged:** the arena reserve for the decode arm (R6).  It is unnecessary while nothing is routed to
`ctx_dec`; it becomes necessary with the routing (6d/6e/6f), where the measured cost is recorded.

---

## 2026-11-23 — tooling: the ROCm 10.1 shadow removed, the draft-ctx check fixed, containers on 10.2 (`WORKLOG.md` 261123.012)

Not a campaign step: the delivery's *instrumentation*.  Triggered by STEP 3's finding that
`gate-park-idle-slots.sh` failed without a `ROCBLAS_USE_HIPBLASLT=0` pin.

### The failure condition, measured (it was the toolchain, not the pin)

`LD_LIBRARY_PATH` is searched **before** `DT_RUNPATH`; 19 scripts defaulted it to the retired
`/opt/rocm-10.1.0-gfx120X/lib` while the build links 10.2.0:

```
with the old default : libamdhip64.so.7, librocblas.so.5, libhipblaslt.so.1 => /opt/rocm-10.1.0-.../lib
with the build's     : the same three                                      => /opt/rocm-10.2.0-.../lib
libhipblaslt: 10.1.0 -> .so.1.4 (9.2 MB)   |   10.2.0 -> .so.1.5 (3.4 MB)   # 1.4 = the pre-c50ea60130f container
```

The `park` arm (identical args, fresh process each run):

| library set | pin | same arm across runs | gate |
|---|---|---|---|
| 10.1.0 | none | `6a6e67…` · `6a6e67…` · `6a6e67/229547/2b4487/824b24` · `4f468c…` | FAIL 1-2 of every 2-4 runs |
| 10.1.0 | `ROCBLAS_USE_HIPBLASLT=0` | 3 runs x 3 arms identical | pass |
| 10.2.0 | none | 2 runs x 3 arms identical, `parkoff` stably different | **PASS** |

### After the fix (all re-run on the build's own 10.2 runtime, no pins)

| gate | result |
|---|---|
| `gate-park-idle-slots.sh` | **PASS**, no env pins at all; `parkoff` stably differs (the #210 dependence is reproducible) |
| `gate-state-pipe.sh` | **PASS** (pipe 19.8 ms vs sync 41.4 ms, 0.48x) |
| `gate-spec-ckpt-device.sh` | **PASS** (identical text/drafts/accepted; -148 MiB resident) |
| `gate-draft-ctx.sh` | **PASS 11/11** with the re-stated census arm (`e7aae6034b4f`, acceptance 0.81429) |

### Re-assessment of the Flash-Next concurrency question (STEP 9 input)

Flash-Next host-tier IQ4_NL, **2 concurrent slots**, 3 fresh processes, 10.2 + hipBLASLt **on**:

| run | slot 0 (reasoning) | slot 1 (recall) |
|---|---|---|
| 1 | `3ec8e6750cd6` (188 ch) | `b2e646545bd9` (251 ch) |
| 2 | `3ec8e6750cd6` | `c60b5a919c9b` (248 ch) |
| 3 | `3ec8e6750cd6` | `c60b5a919c9b` |

The toolchain cause is gone (slot 0 is stable, and the earlier 147-vs-188 observation was 10.1-era noise); a
**residual** effect is real: with two requests in flight the interleaving picks the batch composition and the
composition picks the kernels, so one slot's answer can vary run to run.  Conclusion for STEP 9: anchor the
**byte** claim at one request in flight; the concurrent arms carry the non-text claims (no fault/OOM, per-slot
== that slot solo with the others resident, cell census).

---

## 2026-11-23 — STEP 5: `llama_memory_recurrent` contextified (`WORKLOG.md` 261123.011)

2 files, +119/-48 (`src/llama-memory-recurrent.{h,cpp}`).

| arm | shape | result |
|---|---|---|
| `gates/probe-recurrent-shared.cpp` (new) §1 | 27B Q8_0 (GDN hybrid), 1 GPU; two batch contexts prepared against the ONE memory, both **kept alive** | **PASS**: `A: head=0 n=1 rs_z=0 | B: head=1 n=1 rs_z=1`; A's cell holds seq 0 only, B's seq 1 only |
| … the STEP 4 build (control) | same | **FAIL at that line**: `A: head=1 n=1 rs_z=1 | B: head=1 n=1 rs_z=1` (one module value for both) |
| `gates/probe-recurrent-shared.cpp` §2 | two threads, seq 0 prefill + seq 1 single-token steps on the borrower | `DIGEST arm_a=9eaff8b3b77dc104 arm_b=04c5d3a8e61c128e`, identical to `--solo` |
| A1 / A2 / A3 | 4B 1 GPU / 27B 3 GPU tensor / 35B-A3B host tier | `be904e86f248` / `10a82b5559a9` / `e9db5cb8db0b` |
| A4 | prefill-logit | 0.000000 / 100.000 % |
| A5 four-axis | 27B (recurrent), 3 GPU tensor, `-n 3000`, f16 KV | hashes `f27add1937e2` / `a87c4318b649` / `c494eb660f49` / `5f6c662c4e0f`, `none`==adaptive 4/4, t/s 41.7/79.1, 41.9/151.7, 41.3/111.1, 41.6/128.5, **acceptance counts bit-identical** to STEP 0 (R 1140/826/3574/1858, K 74/73/435/423, P 462/423/2409/1610, C 539/505/3559/2460) |
| `gate-spec-purity.sh` | 3 devices, q8_0 + bf16 | **PASS** (plain == draft-mtp both caches; width probe 0 maxdiff) |
| `gate-spec-gdn-split.sh` | 27B, 3 devices `-ts 1,2,2`, ngen 600 | **PASS** (one GDN prefill tail 16 across `n_rs_batch` 0/5/16/65; text `0a2b2aff2576` everywhere) |
| `gate-spec-ckpt-device.sh` | 27B host vs device checkpoints | **PASS** (identical text/drafts/accepted tokens; 18 created / 5 restored both arms) |
| A6 + startup shape | 2 slots `--kv-unified` | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`, shape `diff` empty |
| GPU faults | journal + every log | clean |

**A recurrent cell holds a sequence, not a token**: both batches occupy ONE cell (`n = 1`) even at 32 tokens,
which is why the oracle compares WHICH cell and whose sequence — an `n == n_tokens` expectation is wrong for
this module (`INVARIANTS.md` r358-wip item 3).

---

## 2026-11-23 — STEP 4: the Meta view slot is a per-context identity (`WORKLOG.md` 261123.010)

3 files, +74/-7 (`ggml/src/ggml-backend-meta.cpp`, `src/llama-context.{h,cpp}`).

| arm | shape | result |
|---|---|---|
| `gates/probe-meta-slots.cpp` (new) | 27B Q8_0, 3 devices `-sm tensor` (the Meta backend is really built), 3 contexts (`owner`, `borrower`, a third live context) | **PASS 11/11**: bases **0 / 2 / 4**, distinct, even, pairs inside the backend's 8; identical token (11751) with 2 and with 3 contexts live |
| `GGML_META_DEBUG_SET=1` | same model/shape, `-n 16` | **0 `MISSING child`** lines; text `26b1dc6dd4ad` |
| A1 / A2 | 4B 1 GPU / 27B 3 GPU tensor | `be904e86f248` / `10a82b5559a9` |
| A6 + startup shape | 2 slots `--kv-unified`, 2 concurrent | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`, shape `diff` empty |
| A4 | prefill-logit | 0.000000 / 100.000 % |

The per-context base line is DEBUG by design (an INFO line per context would add a line to every
single-context run's startup log, breaking the STEP 0 anchor for a diagnostic only a multi-context run
reads); the census rides on the STEP 2 borrow line: `Meta view slots 2/3 and 0/1`.

---

## 2026-11-23 — STEP 3: one allocator, one lock (`WORKLOG.md` 261123.009)

13 files, +314/-19 (`src/llama-memory.h`, `src/llama-kv-cache.{h,cpp}`, the 8 composite memory headers,
`src/llama-memory.cpp`, `src/llama-context.cpp`).

| arm | shape | result |
|---|---|---|
| `gates/probe-kv-reservation.cpp` | phi-4 IQ4_XS, CPU only (`n_gpu_layers=0`), **unified** cache so two sequences share one cell array, `--ntok 512` | **PASS**: unborrowed A/B `[0..511]` each (bitmap off by design); borrowed A `[0..511]`, B `[512..1023]`, C `[1024..1535]`; a dropped batch context releases (later prepare starts at 0); teardown census `0 of 8192` |
| … the STEP 2 build (control) | same probe, same shape | **FAIL at the I4 line**: `prepare A: [0..511]`, `prepare B: [0..511]`.  The other 10 checks pass, so the probe isolates the mechanism |
| `gates/probe-n1-threads.cpp` | 4B Q8_0; owner `n_batch 2048 / n_ubatch 512` (`n_ctx 8192`, `n_seq_max 4`, unified ⇒ 2048 cells), borrower `n_batch 8`; **two threads**: prefill 1900 tokens on seq 0 × 3 rounds while seq 1 decodes 32 single-token steps on the borrower | **PASS** and `DIGEST prefill=64a94dac76706dfd decode=897f0e3809ab747f` **identical to `--solo`**; wall 982 ms vs 1140 ms solo; teardown census `0 of 2048 reserved`.  Declared as the I3/no-corruption oracle: the pre-STEP-3 build reads the same digests |
| A1 / A2 / A3 | 4B 1 GPU / 27B 3 GPU tensor / 35B-A3B host tier | `be904e86f248` / `10a82b5559a9` / `e9db5cb8db0b` — unchanged |
| A6 | 2 slots `--kv-unified`, 2 concurrent greedy | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}`; **startup shape byte-identical** (`diff` empty) |
| A4 | `gate-prefill-logits.sh` | **0.000000 / 100.000 %** |
| `gate-state-pipe.sh` | ROCm 10.2, pinned | **PASS** — every arm byte-identical to the synchronous reference; pipe 18.7 ms vs sync 41.4 ms (0.45×) |
| `gate-park-idle-slots.sh` | Flash-Next host-tier IQ4_NL, `-ncmoe 48` | **PASS with `ROCBLAS_USE_HIPBLASLT=0`** (2 runs).  Failure without it, and the attribution grid, is in the WORKLOG entry §4 |

### The one thing this step's own record must not overstate

The reservation is proven by a **deterministic set-algebra probe**, not by the soak: the two-thread arm reads
the same digests on the STEP 2 build too.  That is not a weakness of the step, it is the reason the probe was
written -- a collision needs two batches that are prepared and not yet applied, which the public API does not
expose, so a timing-based arm would be luck.  The soak still earns its place as the I3 oracle (no deadlock, no
corruption, census clean) and it is the shape STEP 6's server will use.

---

## 2026-11-23 — STEP 2: the borrow (`llama_context_params::ctx_memory`) (`WORKLOG.md` 261123.008)

Files: `include/llama.h`, `src/llama-context.h`, `src/llama-context.cpp` (79 insertions, 6 deletions;
`src/llama-cparams.h` needs no field — the borrow is resolved in the constructor).

| arm | shape | result |
|---|---|---|
| `gates/probe-ctx-memory.cpp` | 4B Q8_0 (owner/borrower) + phi-4 IQ4_XS (the different-model refusal), 1 GPU | **PASS 11/11**: identity `0x55fe1f3ea260` both; negative control (an unborrowed context's memory differs); `t1=11751 t2=13` from the owner-only arm **and** from the arm whose last two rows were decoded on the borrower; both refusals; owner still decodes after the borrower is freed |
| A1 / A2 | 4B 1 GPU `-n 64` / 27B 3 GPU `-sm tensor -mg 0` `-n 96` | `be904e86f248` / `10a82b5559a9` — unchanged |
| A6 | 2 slots `--kv-unified`, 2 concurrent greedy | `{142 ch 7bd542418112, 207 ch 6ce1f88dd5f7}` — unchanged |
| A6 startup shape | `baseline/norm-log.sh … --drop-prefix 'slot (launch_slot_\|print_timing\|release\|select)'` vs `baseline/A6-startup-shape.txt` | **identical** (`diff` empty, 11 lines) |
| A4 | `gate-prefill-logits.sh` (no `--record`) | mean KLD **0.000000**, same-top-p **100.000 %**, journal clean |
| `gate-draft-ctx.sh --quick` | external MTP head (Flash-Next GSQ-RCO-IQ3_S + shared Q8_0 head), 1 device | text arms **5/5 PASS** — plain and 2 `draft-mtp` runs all `e7aae6034b4f`, acceptance 0.81429, no same-model refusal.  **census arm FAIL by construction**: it greps for `is a context of ANOTHER model (a draft head)`, which exists only in `beta/prefill-decode-arms/prefill-decode-arms.patch` — the r352 withdrawal deleted the generic `ctx_other` arm it belonged to (`AGENTS.md`: this gate is **beta-only since r352**).  Verified not mine: `grep -rn "ANOTHER model" src/*.cpp` finds nothing |
| build | `~/bin/build-llama-rocm` | clean |

### The two probe failures found on the way (both were MINE, and both are now facts)

1. **The first version of check 3 compared the wrong two tokens** (the argmax over the prompt's last row
   against the argmax after decoding one more token) — a probe bug, not a code bug; it now runs two
   independent (owner, borrower) pairs through the *same* three steps and compares like with like.
2. **The `ctx_other` guard did not fire.**  Written against `cparams.ctx_other`, which the constructor filters
   to `nullptr` for every architecture that does not need one — so it was dead on qwen35 (the probe's model)
   and alive only where `ctx_other` is meaningful.  Now on `params.ctx_other`.  `INVARIANTS.md` r355-wip item 3.

---

## 2026-11-23 — STEP 1: the three backend prerequisites (`WORKLOG.md` 261123.007)

**Tree:** same r353 tree, `~/llama.cpp` branch `arms-v2`.  Files: `ggml/src/ggml-cuda/ggml-cuda.cu`,
`batch-copy.cu`, `common.cuh`, `ggml-cuda-slab.cu`, `ggml-cuda-vmm.h`.  No delivery file touched.

| arm | command / shape | result |
|---|---|---|
| shlib symbols, default (`GGML_HIP_NO_VMM=OFF`) | `scripts/check-shlib-symbols.sh` | **OK**, 16/16 libraries resolve every symbol |
| shlib symbols, `GGML_HIP_NO_VMM=ON` | `BUILD_DIR=build-rocm-novmm EXTRA_CMAKE_FLAGS="-DGGML_HIP_NO_VMM=ON" ~/bin/build-llama-rocm` then `check-shlib-symbols.sh ~/llama.cpp/build-rocm-novmm` | **OK**, 16/16.  The `#else` definition is the one compiled: `strings … \| grep -c "peer access to a"` is **1** in the VMM build and **0** in the NO_VMM build |
| host-stream probe (non-vacuity for fix 1) | `gates/probe-host-stream.cpp` linked against `build-rocm/bin/libggml-hip.so` | **PASS**: devices 0/1/2 → `0x558a914d5c50` / `0x558a91dee140` / `0x558a91eb9c70`, repeat calls identical, `hipStreamPerThread` = `0x2`, legacy default = `(nil)`; a second **thread** gets `0x7f4b68000d90`, distinct |
| A1 | 4B Q8_0, 1 GPU, prose, `-n 64`, 2 runs × 2 builds | `be904e86f248` ×4 (anchor) |
| A2 | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, `-n 96` | `10a82b5559a9` (anchor) |
| A2 + `GGML_CUDA_P2P=1` | exercises the new split grant (P2P on ⇒ `use_peer_access`) | `10a82b5559a9`, **no** `peer access to a …` warning (all peers granted), kernel journal + log clean |
| A3 | 35B-A3B Q8_0, `-ncmoe 40` host tier, `MOE_EXPERT_HEATMAP=load` | `e9db5cb8db0b` (anchor) |
| A4 | prefill-logit, base recorded from the **pre-change** build (`gates-baseline/`), candidate = post-change build | mean KLD **0.000000**, same-top-p **100.000 %**; the two base files are **byte-identical** (`sha256 ce8bb26f…` both) |
| GPU faults | `check-gpu-faults.sh --since` over the 3-GPU and host-tier logs | clean |
| copy/tensor plumbing | `test-backend-ops -o CPY` on 3 ROCm backends; `test-backend-tensor` | 286/286 + 286/286 passed; all backends passed |

### Non-vacuity of the pool fix, and its one uncovered branch

`GGML_CUDA_P2P=1` takes the `use_peer_access` path, so the **split grant ran** (owner located, per-peer
`cuMemSetAccess`, mask registered) and every peer was granted — the copy behaviour is therefore unchanged,
which is what the identical text shows.  The **refusal** branch needs a peer grant the driver refuses, i.e. an
arena that has taken the VRAM *and* a second context whose pool grows at request time.  This box cannot
produce that without the pair: the branch stays *exercised but not fired* until STEP 6/8's pair gate on a full
host-expert arena.  Recorded so it is not mistaken for a validated path.

### Observation: the pre-change A4 base disappeared from `~/.cache/rdna-boosts/gates/`

The STEP 0 base (5.1 GiB, `sha256 ce8bb26f…`) was gone when A4 was re-run after the STEP 1 build: the
directory existed and was empty, and `~/.cache/rdna-boosts` + `gates/` both carry `ctime == mtime ==
10:20:39`, while `arms-v2/` and `gpu.lock` in the same parent were untouched.  Nothing in the delivery writes
that path (`grep -rn rdna-boosts` over `src/ ggml/src/ common/ tools/` finds no writer), no `tmpfiles.d` rule
names it, and the system journal has no entries for the window.  Cause unknown; the consequence was ~2
minutes of re-recording, and the workaround is the one used here: record the cross-build reference into a
**separate** directory (`gates-baseline/`), which is also the right protocol for a change whose oracle is a
KLD point.  A recurrence would be worth chasing; a one-off is not.

---

## 2026-11-23 — STEP 0: workspace, branch, baseline anchors (r353 tree)

**Tree:** delivery `v17-919794500-r353` (base `919794500`, tree `27db9bfd4`, tip `089f90067`, build
banner `b12615-089f90067`), ROCm **10.2.0** (`/opt/rocm-10.2.0-gfx120X`), 3 × gfx1201 R9700.
**Branches:** delivery `prefill-decode-arms-v2` (off `main`); `~/llama.cpp` `arms-v2` (local, never
pushed).

### The seven anchors

| # | shape | result |
|---|---|---|
| A1 | 4B Q8_0, 1 GPU, prose prompt, `-n 64`, greedy seed 42 | **251 chars `be904e86f248`** (prompt 7107.9 t/s, gen 101.3 t/s) |
| A2 | 27B Q8_0, 3 GPU `-sm tensor -mg 0`, `-n 96` | **399 chars `10a82b5559a9`** |
| A3 | 35B-A3B Q8_0, `-ncmoe 40` host tier, 1 GPU, `-n 64` | **262 chars `e9db5cb8db0b`** (prompt 1104.0 t/s, gen 52.3 t/s) |
| A4 | prefill-logit base, default point (27B Q8_0, `--device 0`, f16 KV, `-ub` default, 40×512) | base PPL **5.9325**, base sha256 `ce8bb26f…`, self-check **KLD 0.000000 / 100.000 %** |
| A5 | MTP four-axis, 27B Q8_0 3 GPU `-sm tensor -mg 0`, `-n 3000`, f16 KV, adaptive ceiling 7 | see the A5 table below |
| A6 | single-context server, 2 slots `--kv-unified`, 2 concurrent greedy completions | `{142 chars 7bd542418112, 207 chars 6ce1f88dd5f7}` + an 11-line startup shape |
| A7 | `scripts/validate-set.sh` | **OK** (18 default blocks, 2 optional, tree `27db9bfd4`, reconciliation 0 unexplained) |

### A5 — the four-axis table (this build's own baseline)

`baseline/mtp-axis.sh --axis <R|K|P|C> --spec none|adaptive`, one arm at a time, f16 KV,
`ROCBLAS_USE_HIPBLASLT=0`, `-n 3000`.

| axis | none t/s | adaptive t/s | speedup | acc drafts/gen | acc tok/gen | mean len | pos 1 | text (none == adaptive) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| R | 41.7 | 79.0 | 1.89× | 0.72456 | 0.51987 | 2.63 | 0.725 | `f27add1937e2` |
| K | 41.8 | 151.4 | 3.62× | 0.98649 | 0.97241 | 6.72 | 0.986 | `a87c4318b649` |
| P | 41.4 | 111.2 | 2.69× | 0.91558 | 0.66833 | 4.48 | 0.916 | `c494eb660f49` |
| C | 41.6 | 128.3 | 3.08× | 0.93692 | 0.69121 | 5.56 | 0.937 | `5f6c662c4e0f` |

Purity (`none` == drafter, byte for byte) holds on **4/4 axes**; pos-1 acceptance is 0.725-0.986 against
the ~0.45 floor; the drafter wins on every axis.  The **K hash matches the delivery's r328 record**
(`a87c4318b649`) and every `none` rate agrees with it to ~2 %, so this command reproduces the delivery
baseline's plain arms.  Its adaptive trajectory does **not** reproduce that record's (R mean-acc-len
2.63 here vs 3.40 there: different `n_min_adaptive`/`n_max` trajectory), so r328's `draft-mtp`
rates must not be quoted against this table, and vice versa.

### Harness defects found and fixed at STEP 0 (recorded because they will bite again)

1. **A bare `wait` in the server driver also waited on the server** (started as a background job in the
   same shell), so the arm "completed" only when the server's own `--timeout` expired — a 900 s
   unguarded wait behind a process that was already healthy and idle.  Fixed: background request jobs
   are waited on **by PID**, and every wait is bounded (readiness 60 s, per request `--max-time`, a hard
   server `--timeout`); a server that dies fails the arm immediately with its log tail.
2. **Two arms launched with `&` contaminated a text anchor**: the R axis's first `none`/`draft-mtp`
   pair ran concurrently and read *different* texts at the same seed.  Re-run sequentially they are
   identical.  Fixed (and enforced): `baseline/mtp-axis.sh` takes an exclusive `flock` on
   `~/.cache/rdna-boosts/gpu.lock` and refuses after a bounded wait, so "one GPU job at a time" is a
   mechanism, not a habit.

### Non-anchor observations

* `~/bin/build-llama-rocm` configures **ROCm 10.2.0** (`/opt/rocm-10.2.0-gfx120X`), not 10.1 as
  `AGENTS.md`'s "Build the fork" section states; the prefill-logit gate agrees (its base file carries
  `rocm-10.2.0-gfx120X`).  Reported to the maintainer rather than edited, since `AGENTS.md` is policy.
* The A6 server log's shutdown tail is `Received second interrupt, terminating immediately` plus the
  block-09/15 sink census lines: the harness kills the process group, so `baseline/norm-log.sh` cuts the
  shape at the first termination line.
