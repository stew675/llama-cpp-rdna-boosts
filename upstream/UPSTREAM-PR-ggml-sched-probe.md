# Upstream-PR proposal — ggml sched: probe-based alloc-fallback sync (single-device fast path)

Status: PREPARED 2026-09-06, gated by single-GPU validation (task 3) — PASSED.  Not yet
filed.  Pre-submit step still required: clean build of the patch on upstream master
(465e49b9c) + a same-seed coherence run vs unpatched upstream on the target hardware.
Patch file: `UPSTREAM-PR-ggml-sched-probe.patch` (3 files, +70/−9, applies clean to
465e49b9c; verified with `git apply --check`).

## TL;DR

`ggml_backend_sched_alloc_splits`'s re-allocation fallback currently does an unconditional
full device synchronization whenever the stored graph layout does not fit a new graph.
With async pipelined callers (llama.cpp decodes multiple ubatches without synchronizing
between them) whose graphs alternate shapes (e.g. dense/sparse attention topologies), the
fallback fires on every ubatch of every repetition and drains the whole queued pipeline
each time — measured ~3 s stalls per fallback at long prompts, costing −17%..−36% at
pp4096..16384 (gfx1151, Qwen3.8-Flash-Next-style dense/sparse alternation).  This change
makes the fallback *probe* the layout first and synchronize only when a buffer must
actually grow — and, for safety across devices, whenever more than one async (non-CPU)
backend is present.  Single-GPU schedulers (a GPU, or a GPU + a CPU backend) keep the
no-sync fast path; multi-device schedulers keep the current upstream behavior exactly.

## Background / why the unconditional sync exists

The sched keeps one stored layout per buffer set.  Any graph whose node count or
per-position sizes differ from the stored layout sends `ggml_backend_sched_alloc_splits`
into its fallback, which (upstream) synchronizes every backend and fully re-reserves.
The sync exists because `ggml_gallocr_reserve_n` is destructive: when a buffer must grow
it frees + reallocates it, moving the addresses of tensors that an in-flight graph may
still be using.

The buffers are grow-only, so a re-reserve that *fits* the current allocation frees and
moves nothing — it only recomputes the tensor layout and re-points the *new* graph's
tensors to addresses in the existing buffers.  That is exactly what the layout-reuse path
already does on every decode with zero synchronization.  The safety argument is therefore
**ordering, not address stability**: all compute for a sched is ordered on the backend
stream(s) after the previous graph's compute, so re-pointing the new graph into
already-reserved (untouched) buffers is safe without a sync.  A sync is required only when
a buffer actually grows.

## Change

1. `ggml/include/ggml-alloc.h`, `ggml/src/ggml-alloc.c` — add `ggml_gallocr_reserve_n_probe`:
   computes and stores the graph layout without allocating or modifying the existing
   buffers, and returns whether any buffer would need to be grown.  Internally,
   `ggml_gallocr_reserve_n_impl`'s sizing (`no_alloc`) path no longer frees the buffers it
   was sizing (it did before, which made size-only calls destructive on a live gallocr)
   and reports growth through a new out-parameter.
2. `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_alloc_splits`'s fallback now probes
   first:
   - `buffers_grown || n_async_devices > 1` → synchronize all backends + `reserve_n`
     (alloc mode) — the upstream behavior, including the whole multi-device case.
   - otherwise → use the probed layout directly (`alloc_graph`), no sync.  Only tensor
     addresses change; nothing is freed or reallocated; the device stream ordering covers
     the compute.

   `n_async_devices` counts device *types* (`ggml_backend_dev_type != CPU`), not raw
   backend count, so a single GPU plus a CPU backend keeps the fast path.

   Also removed: `backend_ids_changed` no longer forces the sync on its own (a backend-id
   change with no buffer growth is also just re-pointing; per-device streams stay in
   order).

## Why the multi-device gate is required

The no-sync re-point is only sound when a single device stream orders all of the graph's
compute.  With more than one device (tensor or layer split) a re-reserve can re-point
tensor addresses while the *previous ubatch's* kernels are still in flight on *other*
devices — a cross-device race.  Reproduced on 3x R9700 (gfx1201) at multi-ubatch prefill:
one device pegged in an in-kernel spin, plus memory faults in `quantize_q8_1` /
`k_get_rows`; `pp2048` (single ubatch) was fine, ≥2 ubatches hung.  The gate restores the
full synchronization for `n_async_devices > 1`, i.e. multi-GPU keeps byte-for-byte the
upstream behavior.  (The campaign that introduced the probe validated single-GPU only;
multi-GPU validation was the follow-up that produced this gate.)

## Validation (numbers from the fork, hardware as noted)

Single-GPU gfx1151 (Strix Halo / RDNA3.5, Qwen3.8-Flash-Next UD IQ4_XS, `-ngl 99 -t 15
-r 3 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --load-mode none`, warm cache):
- The artifact rows recover: pre-change pp8192 339.4 → 544.0 (+60%), pp16384 ~334 → 574
  (+72%), pp4096 367.3 → 487.9 (+33%) (shortcut ON).
- Steady-state alloc-fallback syncs: 8 × ~3 s over 2 passes (pre) → ZERO (post); every
  dense/sparse alternation takes the ~1 ms probe path.
- Gated build vs pre-gate campaign build, same-session r3: parity within machine drift at
  every row (pp512 653.8/652.6, pp2048 777.7/781.9, pp4096 737.8/741.6, tg128
  25.80/25.95); depth d12288 pp2048 637.7/640.4, tg 23.05/23.20.
- Pure-gate isolation (identical tree ± gate): same-seed llama-cli text byte-identical;
  perf parity (pp512 657.95/655.99, pp2048 776.77/775.56, tg 25.78/25.79) — the gate is
  inert at one async device.
- Same-seed llama-cli outputs byte-identical vs the pre-change build (no numerics change:
  no op semantics or kernel schedule touched; only when/where the host re-points scratch
  while the queue drains).

Multi-GPU gfx1201 (3x R9700, tensor + layer split): full sync path active — no regression
vs upstream behavior; the multi-GPU hang is gone (fixed build 3/3 tensor-split pp8192
ub2048 runs at 2130-2156 t/s on the clean box; layer-split pp8192 ~1650-1700 t/s).

## Notes for filing

- The patch is the scheduler portion only of the fork's change; fork-specific hunks
  (references to custom op types `FLASH_ATTN_QSA` / `INDEXER_TOPK`) are excluded.  No
  changes to `ggml.c`/`ggml.h` or any kernel file are included.
- Scope is core-ggml / arch-agnostic.  The benefit shows up with any caller that pipelines
  async graphs with alternating shapes (multi-ubatch prefill with per-ubatch topology
  flips); llama.cpp's speculative/parallel decode paths are the natural consumers.
- Pre-submit TODO: configure+build the patch on a pristine upstream checkout
  (`465e49b9c`), run llama-bench pp512..16384 single-GPU + a multi-GPU tensor-split sanity
  run, and a same-seed coherence diff vs the unpatched build.  Reference records live in
  the delivery repo: `beta/qwen4exp/README.md` (2026-09-06 validation bullet),
  `beta/qwen4exp/HALO_HANDOFF.md` (halo protocol), and `wip/archive/qwen4exp/discovery/
  2026-09-05-strix-halo-gfx1151-ws3-shortcut-fix.md` (original artifact + fix analysis).
