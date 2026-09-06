# HALO HANDOFF — sched-gate single-GPU validation (gfx1151 / gfx1100)

Date: 2026-09-06 (post-reboot session).  Author: pi agent (soar session).
Audience: the maintainer, running the gfx1151/gfx1100 checks on the Strix Halo box(es).
Purpose: **task 3 of the session** — confirm the sched-gate fix (`c63f7f2a0`, delivery
`d6eb551`) preserves the gfx1151 campaign's single-GPU prefill perf + coherence vs the
pre-fix campaign build.  This is the whole point of the gate: single-GPU users must see
NO change from the fix; only multi-GPU (>1 async device) restores the full sync.

## TL;DR — what to do and what to expect

A/B the two binaries below **same-session** on the IQ4_XS Qwen3.8-Flash-Next model with the
campaign protocol.  **Expect parity** (|Δ| ≲ 3%, machine drift is ~2.5% same-session):
the gate is a *functional no-op on single-device schedulers* — both the pre-fix campaign
build and the gated build take the same no-sync fast path there.  If you see a real delta
or incoherence, stop and report (details below).

## Why parity is expected (read before running)

The fix (`c63f7f2a0`) changes one thing in `ggml/src/ggml-backend.cpp`
(`ggml_backend_sched_alloc_splits`):

```cpp
int n_async_devices = 0;
for (int i = 0; i < sched->n_backends; i++) {
    if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[i])) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        n_async_devices++;
    }
}
// ...
if (buffers_grown || n_async_devices > 1) { /* full cross-backend sync (upstream behavior) */ }
```

- Single GPU + CPU backend → `n_async_devices == 1` → **fast path kept** (sync only when a
  buffer actually grows) — byte-for-byte the same control flow as the pre-fix campaign build.
- ≥2 async devices (tensor/layer split) → full sync restored (the gfx1201 fix; validated on
  3x R9700 — see the soar-side record once written to `beta/qwen4exp/README.md`).
- `src/models/qwen4exp.cpp` changed comments only.

So on halo the two builds should behave **identically**.  The A/B is a regression gate on
the shipped binary, not a search for a delta.

## Halo artifacts (as left by the soar session)

| artifact | path on halo | identity |
|---|---|---|
| Pre-fix campaign build | `/tmp/val-master/build/bin/{llama-bench,llama-cli}` | content == fork tip `f5ac11903` (blocks 01-13 amended + qwen4exp-support), **UNGATED** probe (`ggml_backend_sched_alloc_splits`: `if (buffers_grown)` only, no `n_async_devices`) |
| FIXED (gated) build | `~/llama-delivery/build-gated/bin/{llama-bench,llama-cli}` | worktree `~/llama-delivery` at **`c63f7f2a0`** (master `465e49b9c` + 13 blocks + beta + gate).  Build launched 2026-09-06 ~18:00, was at 57% when last polled; check `/tmp/gated-halo-build.rc` for `DONE rc=0` |
| Reference B (optional 3rd anchor) | `~/strix-llama.cpp` (untouched) | community reference build |
| Delivery bundle | `/tmp/qwen4exp-c63f7f2a0.bundle` | re-fetch if the worktree needs re-adding |

Verify the fixed binary really has the gate before benching:
`grep -n "n_async_devices" ~/llama-delivery/ggml/src/ggml-backend.cpp` (expect 2 hits) —
the pre-fix `/tmp/val-master` tree has 0 hits.

Environment on halo: ROCm **`/opt/rocm-7.14-gfx1151`** (build AND runtime — do not mix in
7.12 or system ROCm); `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib`.  Build flags used for
the fixed build (mirrors `/tmp/val-master`): `-DGGML_HIP=ON -DGGML_CUDA_FA=ON
-DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_NATIVE=ON
-DGGML_LLAMAFILE=ON -DGGML_OPENMP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release`.
No RCCL (single GPU).  Rebuild after any change:
`cmake --build ~/llama-delivery/build-gated --target llama-bench llama-cli -j 16`.

## Model

`/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
(87.24 GiB, UD split — shard 1 is a metadata-only 10.9 MB file, **not** truncated; matches
the HF tree sizes byte-for-byte).  Fits the 124 GB unified memory.  Load ~15-25 s on a
quiet box; do not declare "hung" during/just after load (watch VRAM plateau).

## Protocol (the campaign's own, from the records)

No parallel benches.  Run **UNPINNED** (the high-power pin regresses perf on the R9700 box;
halo should likewise stay at its default power state).  Warm the page cache first
(`cat` the gguf shards to /dev/null once) or use the campaign's `--load-mode none` +
descending-prompts-in-one-process pattern.

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib
MODEL=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
PRE=/tmp/val-master/build/bin/llama-bench            # pre-fix campaign build
FIX=~/llama-delivery/build-gated/bin/llama-bench     # gated build

# depth-0 rows (same-session A/B, r3, interleave PRE/FIX/PRE/FIX... to cancel drift):
$FIX -m $MODEL -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none -p 512 -p 2048 -p 4096 -n 128
$PRE -m $MODEL -ngl 99 -t 15 -r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none -p 512 -p 2048 -p 4096 -n 128

# optional depth rows (shortcut inert at depth; still good to re-anchor):
$FIX -m $MODEL -ngl 99 -t 15 -r 1 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 -p 2048 -d 12288 -n 128
$PRE -m $MODEL -ngl 99 -t 15 -r 1 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 -p 2048 -d 12288 -n 128
```

Read the `n_ubatch` column (with `-ub 2048` llama-bench may print both the default-ub row
and the 2048 row; only the 2048 row is the campaign row).

## Expected numbers (pre-fix campaign build, same box, r3 — the anchor)

Depth-0 (from the delivery-revalidation record `wip/archive/qwen4exp/discovery/
2026-09-06-delivery-revalidation.md`, pre-fix A side):

| row | pre-fix A (t/s) |
|---|---:|
| pp512  @0 | ~647 |
| pp2048 @0 | ~759 |
| pp4096 @0 | ~724 |
| tg128  @0 | ~25.9 |
| pp2048 @d12288 | ~642 (r1) |
| tg128  @d12288 | ~23.1 (r1) |

Cross-session swing on this box/model is up to ~9% — **only same-session A/B is valid**.
The decision metric is `FIX vs PRE` same-session: expect |Δ| ≲ 3% at every row (both builds
take the identical single-device fast path).  For reference the brief's depth-0 ladder
(`wip/strix-halo/SESSION-BRIEF-2026-09-06.md`) showed A pp512 634 / pp2048 725 / pp4096 716
/ pp8192 693 / tg 25.9 — do NOT chase the older pre-split_j matrix numbers from
`2026-09-05-...-ws3-shortcut-fix.md` (pp2048@0 399 etc. predates the split_j Q8_0 re-block;
they are not the current state).

## Coherence (mandatory)

Same-seed llama-cli on BOTH builds; outputs must be byte-identical (the gate cannot change
single-GPU numerics — any text delta = problem):

```bash
$FIX  ... llama-cli -m $MODEL -ngl 99 -p "The capital of France is" -n 20 --temp 0 --seed 42 --single-turn --no-display-prompt </dev/null
$PRE  ... llama-cli -m $MODEL -ngl 99 -p "The capital of France is" -n 20 --temp 0 --seed 42 --single-turn --no-display-prompt </dev/null
# diff the generated text (strip the timing line).  For the qwen4exp model, if
# /tmp/gateA/lg-head-1.txt still exists on halo, the logitcmp (fixed 838-token prompt,
# FNV-40 + top5@9dp) is the stronger check — reuse the campaign's compare tooling.
```

llama-cli needs `--single-turn` AND `</dev/null`.  Check for stray processes
(`pkill llama-cli`/`llama-bench`) before runs.

## If you see a delta (shouldn't happen) — report, don't chase

The gate diff is confined to `ggml-backend.cpp` (+ a qwen4exp.cpp comment).  Possible
non-gate confounds if FIX≠PRE: build flag drift (compare `CMakeCache.txt`), ROCm library
mixing, or the ~9% cross-session swing (mitigate with tighter interleaving, r5).  If a
real single-GPU delta persists with identical flags, capture both `llama-bench` tables +
the coherence outputs and hand back to soar for a pure-gate isolation build
(`627506c1c` = c63f7f2a0's parent, ungated delivery tree, one-commit flip) rather than
changing the gate.

## Optional: exact gate isolation on the delivery tree

If you want the tightest possible gate-only pair on halo: in `~/llama-delivery` check out
`627506c1c` (ungated parent) into a second build dir and rebuild — incremental (only
`ggml-backend.cpp` + `qwen4exp.cpp` differ from `c63f7f2a0`), then A/B `627506c1c` vs
`c63f7f2a0`.  Expect **exact** parity on halo.  Do this only if the main A/B shows
something unexpected.

## gfx1100 (fingon, RX 7900 XTX) — optional second arch

Same protocol + expectations if you want the RDNA3_0 arch too.  ROCm
`/opt/rocm-7.14-gfx1100`; build with `-DAMDGPU_TARGETS=gfx1100`.  The campaign validated
block-13's fused-MoE-MMQ on gfx1100 (record `2026-09-05-rdna3-gfx1100-block-13-moe-mmq.md`);
the sched gate is arch-agnostic so gfx1151 parity is the substantive check.

## Cleanup / hygiene

- `~/llama-delivery` is a separate worktree (branch `qwen4exp-delivery` added to halo's
  `~/llama.cpp` object store); it does NOT touch the campaign tree `~/llama.cpp`
  (still at `f5ac11903`) or `~/strix-llama.cpp`.  Safe to delete after validation:
  `git -C ~/llama.cpp worktree remove ~/llama-delivery` + `git branch -D qwen4exp-delivery`.
- `/tmp/qwen4exp-c63f7f2a0.bundle` and `/tmp/gated-halo-*.log` can be removed at will.

## Report-back format (fill in)

```
date/time:            halo / fingon, same-session interleave r3
model:                IQ4_XS Qwen3.8-Flash-Next (or state which)
rows (FIX | PRE | Δ%): pp512 __ | __ | __   pp2048 __ | __ | __   pp4096 __ | __ | __   tg128 __ | __ | __
coherence:            byte-identical YES/NO (attach diff if NO)
notes:
```

Soar-side context: task 1 (gfx1201 tensor-split A/B on the clean rebooted box) is CLOSED —
fixed build 3/3 tensor-split pp8192 ub2048 passes at 2130-2156 t/s, no hang, decode tg128
49.1; the pre-reboot tensor flake did not reproduce (it was a degraded-box artifact).  The
old pre-re-base control build measured ~445 t/s pp8192 on the clean box — consistent with
the expected fixed≫old prefill gap (the gfx1151 campaign's whole point); the handover's
old-build 1792 pre-reboot figure is attributed to the degraded box state.  No bisect, no
gate change.
