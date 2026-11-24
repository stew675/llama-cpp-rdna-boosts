# HANDOVER — the 2-D read of a SPLIT tensor (`result_output`, 3-device `-sm tensor`) — **FIXED 2026-11-24 (STEP 15)**

> **STATUS: DONE, and the work order's diagnosis was WRONG.**  The failing read is not the Meta backend's split
> arithmetic (it is exact: per-device chunks 331092 / 331092 / 331096 sum to the meta row 993280, and device 0's copy
> of the same arguments succeeded).  It is the **transport**: a `hipMemcpy2DAsync` D2H copy needs a peer grant on the
> destination's pinned allocation for the copying device's agent, and this box refuses those grants
> (`hsa_amd_agents_allow_access ... err 4096`, found with `AMD_LOG_LEVEL=3`).  The CUDA backend now issues **no** 2-D
> D2H into host memory (`ggml_cuda_rows_to_host`), landed as `steps/STEP-15-0022-no-2d-d2h.patch` on `67946d6f0`;
> gate `gates/gate-arms-state-split.sh`; records `WORKLOG.md` 261124.007, `INVARIANTS.md` r384-wip, `RUNS.md`
> (2026-11-24 STEP 15).  **STEP 16 (same date)** then fixed the pair's other measured regression: uncontested
> pair decode was running at HALF the single-context rate because the pair disabled the prefill->decode
> compute-buffer drop (keeping the slab's wide band armed, so no CUDA graph was ever captured); the drop now
> runs fenced by the other arm's step boundary, 45.9 -> 88.4 t/s (`WORKLOG.md` 261124.009, `INVARIANTS.md`
> r385-wip, `steps/STEP-16-0023-pair-keeps-the-drop.patch`).  Read §2 below as the *original* work order (its suspects were refuted in §3.2 terms — see
> the record); §5's promotion obligation (`TODO` #376) stands, and §9's open questions are answered in the record:
> the fix is delivery code in `ggml-cuda.cu` (a block-09 amendment, not block 13), the pair-OFF control provably
> never reaches the path (no multi-row verify batch), and `result_output` was never in a checkpoint — the read is the
> ordinary per-ubatch logits readback.

**Read this file first. It is one task, not a campaign.**  The campaign that produced the block-17 candidate is
finished and recorded: its reasoning and traps are in [`HANDOVER-2026-11-24-campaign.md`](HANDOVER-2026-11-24-campaign.md),
the numbers in [`RUNS.md`](RUNS.md), the dated entries in `WORKLOG.md` (261123.*, 261124.001-007), the engineering
facts in `INVARIANTS.md` (r353-wip … r384-wip).

| | |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`prefill-decode-arms-v2`** (pushed), `main` untouched at `91a81505` |
| release the candidate is cut against | `release.json`: base `919794500`, tip `089f90067`, tree `27db9bfd4ef7f41c7df60f704410de28863beac3`, 18 default blocks |
| work tree | `~/llama.cpp`, branch **`arms-v2`** (local only, NEVER pushed), tip **`67946d6f0`** (STEP 15), tree clean |
| build | `~/bin/build-llama-rocm` → `build-rocm` (ROCm **10.2.0**); fast loop `cmake --build build-rocm --target llama-server -j 16` |
| candidate artifact | `beta/prefill-decode-arms/prefill-decode-arms.patch`, sha256 **`f51a78b0…82d35bb`**, 42 files, +3147/−776, applied tree **`29c9644350`** (STEP 15 cut) |
| campaign `steps/` | 22 patches, `STEP-1-0001` … `STEP-15-0022-no-2d-d2h` (landed `67946d6f0`) |
| **the defect** | this file, §2 — **FIXED** (see the status banner; `WORKLOG.md` 261124.007, `INVARIANTS.md` r384-wip, gate `gates/gate-arms-state-split.sh`) |
| what is NOT the task | rewriting the two-arm design, re-litigating the AR chain, promotion into `patches/` (that is the maintainer's, and it comes *after* this fix) |

---

## 0. START HERE

**Goal: fix, properly, the out-of-slice 2-D read the pair reaches when it saves a state on a `-sm tensor` split.**
Short version:

1. Verify the state (§1) — 2 minutes.
2. Reproduce (§2.4) — one command, ~4 minutes: `repro/repro-mtp-prefill.sh`.
3. Get the facts we do not have (§3): a *symbolized* backtrace and the split state at the failure.
4. Fix the **Meta backend's chunked 2-D path** properly (§4) — not by disabling the fast path.
5. Verify (§5): the reproducer must survive N rounds, a **pair-OFF** arm must be shown to reach (or provably not
   reach) the path, and the campaign + delivery gates must stay green.
6. Land it as `STEP-15-0022`, re-cut the candidate, and **record the promotion-time obligation** (§6): this fix is a
   DELIVERY-level fix and must be folded into the delivery set **independently of the block-17 patch** when the
   candidate is promoted.

**Do not:** push `~/llama.cpp` anywhere (a bare `push` there targets upstream `ggml-org/llama.cpp`); touch
`patches/`, `release.json`, `MANIFESTS.md`'s gate contract or any release gate; create/renumber a delivery block;
"fix" a gate; or take the shortcut in §4.5 and call it done.

---

## 1. The state you must verify

```bash
cd ~/llama.cpp && git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD && git status --short   # arms-v2, 15fd7d832, clean
cd ~/llama-cpp-rdna-boosts && git rev-parse --abbrev-ref HEAD && git log --oneline -1                   # prefill-decode-arms-v2
head -1 WORKLOG.md                                                                                     # 261124.005
cmake --build ~/llama.cpp/build-rocm --target llama-server llama-cli -j 16                             # must be a no-op
ls ~/.cache/rdna-boosts/prompts/bigprefill-3x.txt || repro/make-bigprompt.sh                           # the 77K-token prompt
```

If `~/llama.cpp` is not at `15fd7d832` with a clean tree, STOP and read
[`HANDOVER-2026-11-24-campaign.md`](HANDOVER-2026-11-24-campaign.md) §1-§3 before touching it — every number in
`RUNS.md` is tied to that tree and its `build-rocm`.

**What is already green on this tip** (so you know what a regression looks like):
`gates/gate-arms-cancel.sh` PASS (STEP 13), `gates/gate-arms-fd-budget.sh` PASS (STEP 14), `gates/gate-arms-spec.sh`
36/36, `gates/gate-arms-default-inert.sh --no-cli` PASS (4 shapes), `gates/gate-arms-multislot.sh` PASS,
`scripts/gate-prefill-logits.sh` KLD 0.000000, `scripts/gate-spec-purity.sh` PASS, `scripts/gate-park-idle-slots.sh`
PASS, `scripts/gate-ckpt-reuse.sh` PASS, whole-set `all` build clean.

---

## 2. The defect

### 2.1 The symptom (a beta tester's crash #3, 2026-11-24)

Shape: Flash-Next `GSQ-IQ3_XXS`, `-ncmoe 48`, attendance pool, `-sm tensor -mg 0` on **3 devices**, `-np 2`,
external MTP head (`-md …-Q8_0.gguf --spec-type draft-mtp-adaptive --spec-draft-n-max 7`), pair ON.  One long
generation in flight; 5 s later a ~77K-token prefill on the other slot:

```
E ggml_backend_cuda_get_tensor_2d_async: device 1: 2-D read of 'result_output' failed (invalid argument):
    331092 bytes x 5 copies, tensor pitch 331092, host pitch 993280, offset 0 (ROCm1 buffer)
ggml-cuda.cu: ggml_backend_cuda_get_tensor_2d_async: 2-D read failed          [GGML_ABORT]
```

Note **host pitch = 3 × tensor pitch** (`993280 / 331092 ≈ 3.0`) — the tensor is split across the three devices and
the read is issued against ONE device's slice with the META tensor's host pitch.  `invalid argument` on
`cudaMemcpy2DAsync` means the copy leaves the source allocation (or a zero/mismatched pitch).

### 2.2 What is already ruled OUT

* **Not the all-reduce chain.**  It reproduces with `GGML_CUDA_ALLREDUCE=ce` (no RCCL at all), so STEP 14's change
  is not implicated, and the fd count is not either (STEP 14 took the shape from 2026 → 686 fds, and the abort
  above happens with 687 max).
* **Not a memory-layout/aliasing class.**  The copy is a plain `cudaMemcpy2DAsync`; the arguments are wrong, not
  the memory (the abort is `invalid argument`, not a `MEMORY_APERTURE_VIOLATION`).
* **Not reachable in the checks tried so far with the pair OFF:** a *prompt-cache save* on the same 3-device split
  (`repro/repro-statesave.sh MODE=off`, 3 completions) survives, and so does the same script with the drafter and
  the pair off.  So the pair's per-round **draft checkpoint** is what reaches it — but the *buggy code is
  delivery code* (§6).
* **Not the `FD_SETSIZE` abort** — that was a different crash, fixed in STEP 14 (`WORKLOG.md` 261124.005).

### 2.3 The call chain, as far as it is established

```
server_context_impl::arms_decode_loop()                     <- the ONLY frame a backtrace_symbols_fd() resolved
  -> slot_draft_setup / slot_draft_ckpt / slot_accept_draft_round / slot_step
     -> (the state writer) llama_io_write_host  (llama-context.cpp; the batched "pieces" writer)
        -> ggml_backend_tensor_get_n(...)                 (ggml-backend.cpp, the batched read)
           -> ggml_backend_tensor_get_2d(...) OR the async path
              -> the Meta backend's chunked 2-D branch, "simple tensors as chunks"
                 (ggml-backend-meta.cpp: the AXIS_0/1/2 case that calls ggml_backend_meta_get_2d_any(...))
                 -> ggml_backend_tensor_get_2d_async(simple_backend_j, ...)  -> CUDA -> cudaMemcpy2DAsync
```

`ggml_backend_meta_get_2d_any()` and the async variant were added by **block 09**
(`patches/0009-rdna-boosts-block-09-…patch`); **block 13** added one more call site (the axis-1 row-split MTP
vocab-subset copy).  Verify the current owner with `grep -n "get_2d_any\|simple tensors as chunks" patches/*.patch`.

### 2.4 Reproduce it (one command)

```bash
cd ~/llama-cpp-rdna-boosts/wip/prefill-decode-arms-v2
flock -w 1800 ~/.cache/rdna-boosts/gpu.lock timeout 2400 repro/repro-mtp-prefill.sh   # MODE=on DEV=0,1,2 CTX=204800
```

Expect: the server comes up (686 fds printed), the generation starts, the big request lands ~5 s later, and within
~6 s the server aborts with the `2-D read` error (or, if you have already changed something, survives — read the
script's output: it prints the big request's HTTP status so a *rejected* prefill cannot masquerade as a pass).

Preconditions that bit before: `Ctx` must leave the 77K prompt room in ONE slot (`-c 204800` at `-np 2`); a smaller
`-c` makes the run vacuous (the big request never reaches a slot).  `unset LD_LIBRARY_PATH` — the tester's `runme`
used to put ROCm 10.1 there, which beats RUNPATH (fixed in his script; never reintroduce it).

---

## 3. Step 1-2: get the facts we do NOT have

Two unknowns.  Both are cheap to settle and everything else depends on them.

### 3.1 A symbolized backtrace of the failing read

The only frame that resolved (`backtrace_symbols_fd` at the abort site) was `arms_decode_loop` — the build is
`Release`, so `libllama.so` has almost no symbols.  `gdb` itself died with "A fatal error internal to GDB" on this
abort, so do not plan around an interactive gdb session.  Options:

* build a **RelWithDebInfo** copy at the same source path (ccache stays warm enough) and repeat the reproducer, or
* keep the `backtrace_symbols_fd` at the abort site (it was added at the `GGML_ABORT` in
  `ggml_backend_cuda_get_tensor_2d_async`, and removed again — it is Linux-only, so it must not be committed) and
  add `-rdynamic`/`-g` to the *libllama* build, or
* cheapest: instrument `llama_io_write_host` (its constructor/destructor/`flush_async`) to log **which tensors, which
  offsets/sizes and which state flags** it is writing.  That answers "why does a state save touch `result_output` at
  all, and why FIVE rows?" — the two facts the fix has to be justified against.

### 3.2 The split state at the failure (this is where the bug is)

Log, in the chunked branch of `ggml_backend_meta_get_tensor_impl` (`ggml-backend-meta.cpp`, the `AXIS_0/1/2`
"simple tensors as chunks" case) and once per device:

```
tensor->name, tensor->ne[0..3], tensor->nb[0..3], ggml_nbytes(tensor), offset, size
split_state.axis, split_state.n_segments, split_state.nr[s], split_state.ne[s*n_bufs + j] (for every s, j)
per device j: simple_tensor->ne[0..3], simple_tensor->nb[0..3], simple_tensor->data, the buffer's size
chunk_size_full, chunk_size_j, i_start, i_stop, simple_offset
```

Then check the arithmetic by hand.  **The suspects, in order:**

1. `simple_offset = i_start * chunk_size_j` and `n_copies = i_stop - i_start` assume the device's rows are packed
   contiguously at the device's pitch — i.e. they **ignore `split_state.n_segments` / `nr[]`**, which is exactly
   what the two special-cased branches *above* in the same function walk explicitly ("a device's chunks (seg,
   repeat) are packed contiguously in its own slice, so repeat r of a segment does NOT start at `r * rows` of that
   slice").  If the tensor has more than one segment or more than one repeat, the offset and the copy count are
   both wrong — and the read walks off the end of that device's slice, which is precisely `invalid argument`.
2. `GGML_ASSERT(offset_j == chunk_size_full)` requires the device chunk sizes to sum EXACTLY to the meta chunk.
   `3 × 331092 = 993276 ≠ 993280`: four bytes of slack in the meta row.  If the device chunk sizes are all
   `331092` this assertion is what *should* fire (and does not, because the CUDA copy aborts first) — so either the
   split is uneven by 4 bytes (then the assert must be relaxed/corrected) or the sizes are `331092/331092/331096`.
   Settle that first: it tells you which branch of the fix you need.
3. The same question for the WRITE side: `ggml_backend_tensor_set_2d` / the meta `set_tensor_2d` path, and the
   batched `get_tensor_n`/`set_tensor_n` ifaces.  A read that is wrong because of segments is usually wrong on the
   write side too — check before you claim the class is fixed.

---

## 4. Step 3: the REAL fix

### 4.1 What "proper" means here

The failing read is a **legitimate** request: "give me rows [i_start, i_stop) of this meta tensor".  A split tensor
must answer it by computing, per device, the **byte range that device actually holds for those rows**, walking the
split state exactly as the setter and the axis-1 special case do (`n_segments`, `nr[s]`, `ne[s*n_bufs + j]`,
per-device offsets accumulated as the setter accumulates them).  Then issue one 2-D read per (device, segment,
repeat) with that device's own `nb`, or a 1-D read when the range is contiguous.

### 4.2 Where to implement it

`ggml/src/ggml-backend-meta.cpp`, the `ggml_backend_meta_get_tensor_impl()` chunked branch (and its write-side
twin).  The two specialized branches directly above it (axis-1/2-D row split, and the multi-segment walk) are the
model to follow: they already do the segment/repeat walk correctly for 1 copy.  The clean shape is to extend that
walk to a **row range** and let it emit one `ggml_backend_meta_get_2d_any()` per device/segment/repeat with the
device's own pitches — which also removes the `n_copies` forwarding that is wrong today.

### 4.3 Tests, before any gate

* A **unit-level** check is worth more than any end-to-end arm here: build a Meta buffer over two fake/adjoining
  buffers with (`n_segments`, `nr`) values of (1,1), (1,2), (2,1), (2,2) and a row-split and a non-row-split axis,
  then `set` a recognisable pattern through the meta tensor and `get` it back by row range (including ranges that
  start in the middle of a segment and cross a repeat), and compare byte for byte.  If that harness is too big,
  the delivery's `tests/test-backend-ops.cpp` already has meta/split coverage — extend it rather than inventing a
  new one (`test-backend-ops` runs without a model).
* The **cluster-free** rule: this bug must be reproducible and verifiable with a *single* process and a synthetic
  split (no RCCL, no 3 GPUs) if the harness above is used.  That is what makes the promotion-time folding honest
  (§6): the fix is provable on its own.

### 4.4 Verification ladder (after the fix)

1. `repro/repro-mtp-prefill.sh` — **N ≥ 3 rounds survive**, the big request returns `http=200` with real content,
   and the log has no `2-D read`, no `GGML_ASSERT`, no `FD_SETSIZE`.  Quote the fd trace (it prints one).
2. A **new gate** — `gates/gate-arms-state-split.sh` (name it as you see fit): the recipe above as a gate, with
   (a) the non-vacuity line (`the decode arm is generating: … step(s) … handover(s)`, and the big request's
   `prompt_n` from the timing line, so a rejected prefill cannot pass), (b) the **pair-OFF control** on the same
   shape (document whether it reaches the path: if it does not, say so in the gate header — a control that cannot
   reach the code is not evidence), and (c) the fault scan.
3. Campaign gates: `gate-arms-spec.sh` (36/36 — it is the text-identity instrument for 3-device splits),
   `gate-arms-cancel.sh`, `gate-arms-fd-budget.sh`, `gate-arms-default-inert.sh --no-cli`,
   `gate-arms-multislot.sh`.
4. Delivery gates that can see `ggml-backend-meta.cpp` / the state path: `scripts/gate-ckpt-reuse.sh`,
   `scripts/gate-state-pipe.sh`, `scripts/gate-meta-seg-drain.sh`, `scripts/gate-seqpar.sh`,
   `scripts/gate-spec-purity.sh`, `scripts/gate-prefill-logits.sh` — plus the whole-set `all` build
   (`~/bin/build-llama-rocm`) and `scripts/check-gpu-faults.sh` over the run window.
5. `scripts/check-windows-portability.sh` (you are inserting lines into a file the delivery ships).

### 4.5 The shortcut, and why it is not the answer

`ggml_backend_tensor_get_2d()` (ggml-backend.cpp) already falls back to a per-copy 1-D loop when a buffer has no
2-D iface.  Making it fall back whenever the tensor's split state is not trivially replicated would silence this
crash in a few lines.  **The maintainer explicitly rejected that as the fix**: it turns a wrong answer into a slow
right one *only* for the shapes that happen to hit the fallback, leaves the same arithmetic wrong for the
`get_tensor_n`/`set_*` batch paths, and hides the defect from the next shape.  If you take it as a *temporary*
measure while developing, it must not be what you commit.

---

## 5. Step 4: where the fix LANDS (read this before you commit)

**Now: in the beta candidate, as `STEP-15-0022`.**  It is one more `steps/` patch on top of `STEP-14-0021`, in the
candidate patch the tester uses, because the candidate is what is being tested.  Mechanics are the ones §7.2 of
[`HANDOVER-2026-11-24-campaign.md`](HANDOVER-2026-11-24-campaign.md) records (canonical replay → regenerate docs →
`format-patch` → round-trip), and they are repeated in §6.2 below because they are easy to get wrong.

**At promotion: the fix must be cut into the DELIVERY set on its own, independent of block 17.**  The maintainer's
instruction (2026-11-24) is explicit: this is a delivery-level defect (`ggml-backend-meta.cpp`'s 2-D path, reached
here through the pair's per-round checkpoint), so it must NOT be smuggled into the delivery as part of the two-arm
block.  When the time comes:

1. Cut the fix as its own change against the release tree — a delta on **block 09** (`ggml-backend-meta.cpp`'s
   chunked 2-D path + `ggml_backend_meta_get_2d_any`, which block 09 introduced) or on **block 13** if the defect
   turns out to be confined to the axis-1 row-split copy that block 13 added.  Determine the owning block by
   reading the patches, not by file mtime; if the fix spans both, it is two changes into two blocks.
2. Re-cut that block's patch in place (`scripts/make-patches.sh` from a canonical chain), then
   `scripts/check-rung-file-sets.py` for the rung and `scripts/validate-set.sh` — a block must not shrink.
3. The beta artifact then drops that fix — since 2026-11-24 it is **its own patch file**
   (`beta/prefill-decode-arms/state-split-2d-read.patch`), so "dropping it" is deleting that file, not re-cutting the
   block-17 patch; the candidate is still re-cut against the new `release.json` (the lower blocks moved), and the
   two-arm block keeps only what is *its*.
4. Record the split in `WORKLOG.md` and in the beta artifact's README, so a reader of either can see that the fix
   exists twice-transitionally and once permanently.

**Do not** do that folding now: it changes the delivery, and the block structure is the maintainer's decision
(`AGENTS.md`: "ALWAYS ask before creating a block" — and this is a block *amendment*, which is the same class of
decision).

---

## 6. Records and the mechanical bits

### 6.1 Records (every landing owes all of these)

* `WORKLOG.md` — a new dated entry at the TOP, `261124.VVV` (today's real date, next free number), with the
  symptom, the mechanism, the fix, the numbers and the verification list.
* `INVARIANTS.md` — an appended `rNNN-wip` entry: the *general* fact (a split tensor's row range is defined by the
  split state, never by `chunk_size_j × i`), tagged with the landing.
* `RUNS.md` — a dated section with the commands, the shape and the numbers (the fd trace, rounds survived, the
  gate verdicts).
* `beta/prefill-decode-arms/README.md` — its **"Open: a 2-D state-save read…"** section becomes *fixed*, with the
  gate name; keep a one-line pointer to the delivery-folding obligation (§5 above).
* The artifact's sha256 / applied tree / file count, in `beta/prefill-decode-arms/README.md`, `beta/README.md` and
  the `MANIFESTS.md` block-17 row (three places; they drifted once already).
* `steps/STEP-15-0022-<name>.patch` (`git diff HEAD~1` from the landing commit).

### 6.2 Re-cutting the candidate (exact, because it has been got wrong)

```bash
cd ~ && rm -rf cand-chain && git clone -q ~/llama.cpp cand-chain && cd cand-chain
git branch -D rdna-boosts -q 2>/dev/null; git checkout -q --detach 919794500
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .          # must print "Applied tip: 089f90067" and tree 27db9bfd4
S=~/llama-cpp-rdna-boosts/wip/prefill-decode-arms-v2/steps
for s in STEP-1-0001-backend-concurrency STEP-2-0002-ctx-memory-borrow STEP-3-0003-one-allocator-one-lock \
         STEP-4-0004-meta-view-slot-identity STEP-5-0005-recurrent-placement STEP-6a-0006-pair-predicate-and-context \
         STEP-6d-0007-slot-step-extraction STEP-6d2-0008-decode-arm-routing STEP-6e-0009-reserve-and-census \
         STEP-7a-0010-aux-narrow-regions STEP-7d-0011-pair-drop-disable STEP-7c-0012-aux-reserve-calibration \
         STEP-7e-0013-meta-slot-per-graph STEP-7f-0014-borrower-kv-types STEP-7g-0015-meta-split-state-per-slot \
         STEP-7h-0016-capture-blas-barrier STEP-10a-0017-draft-round-extraction STEP-10-0018-speculator-on-the-pair \
         STEP-10b-0019-per-aux-context-reserve STEP-13-0020-cancel-release-barrier \
         STEP-14-0021-one-rccl-per-process STEP-15-0022-<your-name>; do
    git apply --index "$S/$s.patch" && git commit -q -m "$s"      # plain `git diff` output, so NOT `git am`
done
git rev-parse HEAD^{tree}                                     # MUST equal ~/llama.cpp's tree at your landing
~/llama.cpp/build-rocm/bin/llama-gen-docs                     # the candidate owns the doc regeneration
git commit -q -am docs
NEW=$(git rev-parse HEAD)
git checkout -q -B cand-one 089f90067 && git diff --binary 089f90067 $NEW | git apply --index
git commit -q -F <the block-17 subject + a sentence on your fix>    # the SUBJECT must stay the block subject
git format-patch -1 --stdout > ~/llama-cpp-rdna-boosts/beta/prefill-decode-arms/prefill-decode-arms.patch
```

Then the round trip, which is the contract the tester relies on:

```bash
cd ~ && rm -rf cand-rt && git clone -q ~/cand-chain cand-rt && cd cand-rt
git checkout -q 089f90067 && git am ~/llama-cpp-rdna-boosts/beta/prefill-decode-arms/prefill-decode-arms.patch
git rev-parse HEAD^{tree}        # must equal the tree you cut from
```

And the documented tester path once: `bash scripts/apply-all.sh --include-beta .` from `919794500` must report the
same applied tree.

**Two traps, both already paid for:** a plain `git diff` is NOT `git am`-able ("Patch format detection failed") —
the artifact is `format-patch` output; and the squashed commit's SUBJECT must be the block-17 subject, or `git am`
names the block "docs" (it happened, and the artifact had to be re-cut).

---

## 7. Harness inventory (this directory)

```
repro/     README.md            the recipes and their traps
           make-bigprompt.sh    builds the ~77K-token prompt
           repro-mtp-prefill.sh THE reproducer for this defect (also reached the fd abort before STEP 14)
           fd-census.sh         an arm's fd count + its /dev/shm/nccl-* count
           repro-cancel.sh      crash #1 (now gate-arms-cancel.sh)
           repro-statesave.sh   a prompt-cache save, pair off (survives — the attribution evidence)
gates/     gate-arms-cancel.sh      STEP 13: aborting one of two generations (fails pre-fix, passes after)
           gate-arms-fd-budget.sh   STEP 14: an auxiliary context adds NO comm layer (three arms + non-vacuity)
           gate-arms-spec.sh        the four-arm text claim + the four axes, with a drafter
           gate-arms-default-inert.sh, gate-arms-multislot.sh, gate-arms-n1-memory.sh,
           gate-arms-itl.sh, gate-arms-recurrent-shared.sh
           probe-*.cpp              the five non-vacuity probes
steps/     STEP-1-0001 … STEP-14-0021      the review history (plain `git diff` output)
baseline/  server-arm.sh, shape-matrix.sh, itl-arm.sh, mtp-axis.sh, norm-log.sh, anchors
HANDOVER-2026-11-24-campaign.md   the campaign record (design, traps, per-step reasoning)
RUNS.md                            the numbers of record
```

Every campaign gate prints its own **non-vacuity** line; a gate that could pass on a single-context server is
defective.  A gate that takes the GPU flock itself (`gate-arms-cancel`, `gate-arms-fd-budget`,
`gate-arms-default-inert`, `gate-arms-spec`, `gate-arms-recurrent-shared`) deadlocks if you wrap it in `flock` —
check before wrapping (`grep -n "flock -w" gates/*.sh`).

## 8. Hygiene (each of these has cost real time)

* One GPU job at a time; `flock -w 1800 ~/.cache/rdna-boosts/gpu.lock`.
* `HIP_VISIBLE_DEVICES` pinned on every multi-device arm (the iGPU is otherwise enumerated → `invalid kernel file`).
* `MOE_EXPERT_HEATMAP=load` on every host-expert arm; `ROCBLAS_USE_HIPBLASLT=0` for every fresh-process text
  comparison; **never** put another ROCm install in `LD_LIBRARY_PATH` (it beats RUNPATH).
* `-lv` per arm: none for text/hash, `-lv 4` for censuses, never `-lv 3`.
* `setsid` + ONE `TERM` + wait; bound every wait with `timeout`; never `kill -9` a server whose teardown you want.
* Big artifacts (prefill-logit bases) never go to `/tmp` — it is tmpfs here.  `rm -f gpucore.*.gpu` after a fault.
* The build is RELEASE: if a backtrace is needed, say so in the same breath as the plan (§3.1).

## 9. Open questions for the maintainer

1. **Where the folded fix lands at promotion** — block 09 (the meta iface/scheduler) or block 13 (the axis-1
   row-split copy).  §5 says determine it from the patches; confirm before cutting, because it is a block
   amendment.
2. **A pair-OFF reproducer for this defect.**  None exists yet on this box.  If the fix is provable only through
   the pair, then the folded delivery change is validated by a *unit* test (the split-state walk) rather than an
   end-to-end arm — is that acceptable for the delivery, or does it need its own end-to-end shape?
3. **Whether `result_output` SHOULD be in a sequence checkpoint at all.**  `slot_draft_ckpt()` saves with
   `PARTIAL_ONLY` (+ maybe `ON_DEVICE`, see `spec_ckpt_save_flags_tgt()`), and a *partial* state should not carry
   the output rows — if it does, that is a second question (a state-save size/scope one) that is independent of the
   copy bug.  Settle it while instrumenting §3.1; if the output rows are there by design, say so in the record.
