# `archive/work/kv-residency/patches` — the campaign's code, self-contained

**Not delivery work.**  These patches belong to the `archive/work/kv-residency` campaign (`TODO` **#118**; the commit messages inside say
`#117`, which is the handle the campaign used before that number was allocated to another item).  They are not
in `patches/`, not in `release.json`, and no block in the delivery reads them.  Nothing here may be applied to
`~/llama.cpp`'s `rdna-boosts` branch, and folding any of it into the delivery is the promotion rule's job
(`AGENTS.md`, WIP and promotion rules), not a hand-edit.

They are stored here so a cold session can reproduce the exact tree the campaign was measured on, and so the
work is backed up with the record that explains it.

**Two series live here.**  `0001..0014` in *this* directory are **Goal A's** series (base r174, `5bdec04ab`);
it shipped as delivery **block 15** and is kept as the Goal A measurement record.  **Goal B's** series is
[`goal-b/`](goal-b/README.md) — 7 patches on the **r186** base (`8f9dad0be`), applied tree
`14e76aa2af4135b13cd3657895f04460d3d80774` == branch `wip-kv-residency-b` (`b98cb3e43`), verified 2026-10-31.
Patches 0006/0007 of that series (A12, A13) are **delivery fixes**, not campaign code, and are also staged as
`beta/0016-copy-2d-vec16/` / destined for blocks 10 and 11.

## How to reproduce the tree

The patches apply to the **r174** release tip, `5bdec04ab` (`v15-6050b467c-r174`, base `6050b467c`).
**r176 is now the delivery's current release** (`release.json`); these patches are *not* re-cut against it,
because the campaign is not promoted.  A promotion re-cuts the series from the then-current release's tip.

```bash
cd ~/llama.cpp
git checkout rdna-boosts && git checkout 5bdec04ab        # or: scripts/apply-all.sh after a fresh clone
git checkout -b wip-kv-residency
git am ~/llama-cpp-rdna-boosts/archive/work/kv-residency/patches/00*.patch     # 0001..0014, in order
~/bin/build-llama-rocm
```

If `5bdec04ab` is not available (an old checkout), re-cut it first from this repo:
`./scripts/fork-refresh.sh` (default `LLAMA_REPO=~/llama.cpp`, no `--push`).

## The series

| patch | what | state |
|---|---|---|
| `0001-ggml-cuda-grow-only-VMM-region-primitive.patch` | `ggml/src/ggml-cuda/ggml-cuda-vmm-region.{h,cu}`: one VA reservation, physical mapped per `unit` chunk, grown monotonically, clear-on-map, fail-soft.  The mapped extent is a **set of spans** (`grow_span`/`is_mapped`/`clear` over a per-unit bitmap), because a KV buffer lays its tensors out consecutively and a single prefix watermark would map ~the whole region even for a 4-token run.  Plus a one-shot `GGML_CUDA_VMM_REGION_SELFTEST=1` hook in `ggml_backend_cuda_init`.  Purely additive — no existing path changes. | **applied, building, selftest PASS**; gates green (see `../IMPLEMENTATION.md`) |
| `0002-ggml-cuda-KV-region-buffer-type.patch` | `ggml_backend_cuda_kv_region_buffer_type` (ggml-cuda.cu) + the `llama_kv_cache` selection and the `apply()` grow (src/llama-kv-cache.{h,cpp}).  Env-gated OFF (`LLAMA_KV_REGION=1`), plain device buffer type as the mandatory fallback. | **applied, building**, all gates green — but see the KNOWN LIMIT in `../IMPLEMENTATION.md`: on a slab-armed MoE config the arena keeps the VRAM the KV no longer takes up front, so a run that uses most of `-c` refuses where the plain arm completes |

| `0005-wip-kv-residency-A2-mixed-backing-windowed-KV-one-VA.patch` | **A2, the mixed-backing windowed KV**: one VA reservation, the KV's leading cells backed by pinned system RAM, the rest by VRAM; the mapping-table primitive, the split buffer type, the split point + loud warning.  See `../A2-MIXED-BACKING.md` | **applied, building, gates green** (the design of record) |
| `0006-…-the-SLIDING-window….patch` | the window follows the watermark (the newest W *used* cells stay resident); `ensure_span` flips a mapping's kind through a pinned scratch | applied, measured |
| `0007-…-quantise-the-slide-boundary-i….patch` | the slide boundary is quantised in BYTES to a whole map unit — without it a host chunk was created per tensor per token (**14.7 → 18.6 t/s**) | applied, measured |
| `0008-…-commit-the-resident-WINDOW-a….patch` | the resident window's VRAM is committed at buffer creation, **before `SLAB_INIT`** — without it a deep batch aborts (`growth refused`, `free=0.0 MiB`) | applied, measured |
| `0009-wip-kv-residency-A3-fit-sizing-reporting-of-the-wind.patch` | **A3**: `--fit` sizes + reports the window (`LLAMA_KV_RESIDENT_CELLS=auto`, `llama_kv_region_set_resident_cells`, the KV's bytes-per-cell from the region buffer type, the resident/spill split in `memory_breakdown()`); the pinned-RAM ceiling (`LLAMA_KV_SPILL_RAM_PCT`, creation-time clamp + run-time refusal); the four relaxed async-tensor buffer-type asserts the state gates caught; and the three slide bugs — unit-grid mappings + a refused straddle, kernel-copied staging (a `cudaMemcpy` into a host-location mapping does not land), no coalescing.  See `../A2-MIXED-BACKING.md` §3.7 | applied, building, gates green on a non-spilling arm; **the spilled-KV numeric difference of §3.7.6 was OPEN at this commit and is closed by `0011`** |
| `0010-wip-kv-residency-A3-diagnostic-LLAMA_KV_REGION_SPILL.patch` | `LLAMA_KV_REGION_SPILL_SKIP=<tensor indices>`: keeps the listed tensors of a windowed buffer fully resident.  **DIAGNOSTIC ONLY** (unset = shipped behaviour) — it exists to bisect the open finding (§3.7.6): a single 4 MiB tensor spill already moves the 27B's text, while skipping every tensor reproduces the plain hash exactly | applied, building; the bisect ladder is in `../A2-MIXED-BACKING.md` §3.7.6.  This commit also makes the VMM selftest's PASS line describe what it checks (a partial flip is refused) |
| `0011-wip-kv-residency-A4-close-the-spilled-KV-purity-bloc.patch` | **A4, the purity blocker closed**: a spilled KV is numerically transparent.  The flip was byte-exact all along (stage-out == an independent copy-engine read, restore == stage, exhaustively); the fault is in the primitive -- `cuMemUnmap`(device) + `cuMemCreate`(HOST)/`cuMemMap` at that VA **zeroes a different, live host-location mapping** in the same reservation.  `vmm_map` now holds the released physical for the duration of its create (`LLAMA_KV_REGION_QUARANTINE`, default on, `=0` = the A/B arm), on every map site, and a refused hold fails the mapping loudly.  Plus the instruments: byte-exhaustive restore verify, stage-out cross-check, per-host-mapping fingerprint (reports where and when a spill changes), region dump, canary, `LLAMA_KV_REGION_TRACE`.  See `../A2-MIXED-BACKING.md` §3.7.8 / §3.7.9 | **applied, building**, all gates green: prefill-logit **-0.000000 / 100.000 %** on a spilling arm (was 0.004973 / 98.761 %), 27B q8_0/f16 and 4B text back to the plain hash, OOM config still OOMs with the feature off |
| `0012-wip-kv-residency-A4-ii-the-host-mapping-hazard-as-a-.patch` | **the hazard as a `GGML_CUDA_VMM_REGION_SELFTEST` case** so it is gated instead of living in a `/tmp` probe: 8 tensors x 8.5 MiB, a 4 MiB window each, flipped to pinned host one at a time through the real path, every earlier window re-verified after each flip.  **Sensitive**: with `LLAMA_KV_REGION_QUARANTINE=0` it FAILS on *"hazard: every already-flipped window still holds its bytes"* from the 5th flip.  Selftest-only (no product path); needs `GGML_CUDA_VMM_REGION_SELFTEST=1`, so a gate must ask for it | applied, building, PASS with the hold / FAIL without |
| `0013-wip-kv-residency-A5-i-a-windowed-region-s-tensors-mu.patch` | **A5 (i): a windowed region's tensors must START on the driver's map granularity.**  A flip is a whole-mapping operation, so it must coincide exactly with a mapping -- and the window's boundary is quantised in whole map units, so that only holds while each tensor's *offset* is gran-aligned.  The plain CUDA alloc size pads quantized rows (`MATRIX_ROW_PADDING`), which is not: the qwen4exp QSA indexer pads **+512 B per tensor**, so the second tensor's first slide was refused and `grow_regions` **aborted** the process (measured on Flash-Next `-c 65536 W=1024`; the 27B/4B are unit-sized by luck).  The windowed buffer type now rounds each tensor's place up to `ggml_cuda_vmm_region_granularity()` -- <= gran-1 unused VA bytes per tensor, no mapped byte moved.  See `../A2-MIXED-BACKING.md` §3.7.11.2 | **applied, building**; Flash-Next spilling arm: 156 flips / 156 VERIFY ok, both caches spilling (576 + 48 MiB), text byte-identical, prefill-logit -0.000000 / 99.994 % |
| `0014-wip-kv-residency-A5-ii-a-region-backed-KV-declines-p.patch` | **A5 (ii): a region-backed KV DECLINES an idle-slot park** (`seq_park` returns false, warn once) so the server uses the save-and-clear fallback it already has for a slot that cannot be parked.  A park moves cells to `park_bound()` = the top of the CACHE, above the watermark the region has mapped, and the move is a copy-engine transfer -> a GPU **page fault** (`__amd_rocclr_copyBuffer`, measured with the window OFF too).  Growing the region instead was rejected for a windowed cache: the parked cold cells would take the resident window and the ACTIVE cells would fall into the spilled tail (a per-token cost against save-and-clear's one copy per transition).  `LLAMA_KV_REGION`-only, so the default path is untouched.  See `../A2-MIXED-BACKING.md` §3.7.11.3 | **applied, building**; park gate on a windowed arm: park == lazyoff on A-D, 0 faults (the gate's own "moved nothing" check still trips -- reported); no-region control PASSES unchanged |

| (A2-early) | **A2, the mixed-backing windowed KV**: one VA reservation, the KV's leading cells backed by pinned system RAM, the rest by VRAM; the mapping-table primitive, the split buffer type, the split point + loud warning.  Measured: the 27B Q8_0 `-c 262144` OOM config runs; `(used − resident) × cell_bytes / ~28 GB/s` matches to ~3 %; slab mode's arena gains 2.69 GiB and the 35B-A3B **Q8_0** expert residency goes 63.3 % → 77.0 %.  See `../A2-MIXED-BACKING.md` | **applied, building, gates green** (the design of record) |

| `0004-ggml-fa-A1-two-base-KV-read-SUPERSEDED.patch` | **A1, the two-base K/V read** — the op carries `k_tail`/`v_tail` + `n_tail`, and the tile and MMA-f16 kernels select the segment where a KV tile's base is formed (the cell index space stays global).  Gated byte-identical at several window widths on both families, MTP verify widths, and a negative control.  **SUPERSEDED 2026-10-29** by mixed backing (`../MIXED-BACKING.md`: one VA range backed partly by pinned host memory, measured at 28.4 GB/s) — kept as the record and the fallback, see `../A1-TWO-BASE-KV.md` | **applied, building, all gates green** (not the A2 path) |

| `0003-llama-model-MTP-draft-KV-window.patch` | **O3, the drafter ring**: the MTP draft cache can be a sliding window (`LLAMA_SPEC_DRAFT_WINDOW=<cells>`, default off) — one `create_memory` branch plus a helper, reusing `find_slot`'s reclaim and the cache's own `n_swa` for the mask.  **Implemented and measured; a poor product trade** (-6..-9 % decode, +8 % prefill, and the freed VRAM does not reach the arena) — see `../O3-DRAFTER-RING.md` §2.2 | **applied, building, gates green** |

`diffstat.txt` is the net diff of the series against `5bdec04ab`.

The A1 slice is also a **named branch**, so the fallback can be checked out without replaying any patch, and
**both campaign branches are pushed to the maintainer's personal fork** (`git@github.com:stew675/llama.cpp.git`,
the only remote `~/llama.cpp` has) — that is the backup:

| branch on the fork | tip | what |
|---|---|---|
| `wip-kv-residency` | `dc122f338` | A4 + A4 (ii) (the hazard hold, the instruments, the hazard as a selftest) -- **the design of record** |
| `wip-kv-residency-a1-two-base` | `e49e1cef4` | A1 alone (the two-base K/V read, superseded, kept as the fallback) |

`rdna-boosts` on the fork stays at the release tip and is **not** touched by this campaign.

As slice 2 (the KV buffer type) lands, it is appended as `0002-…`: re-cut with
`git format-patch 5bdec04ab..wip-kv-residency`, replacing the series rather than adding a patch on top of an
already-applied one.

## Verifying an applied tree

```bash
HIP_VISIBLE_DEVICES=0 GGML_CUDA_VMM_REGION_SELFTEST=1 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -c 4096 -t 8 -p hi -n 4 \
  --temp 0 --no-display-prompt --single-turn -v 2>&1 | grep vmm_region
# device 0: VMM region available (...)
# device 0: PASS (reserved 384 MiB in 6 x 64 MiB units, prefix + span growth, ...)
```

The KV region itself (needs `LLAMA_KV_REGION=1`):

```bash
HIP_VISIBLE_DEVICES=0 LLAMA_KV_REGION=1 LLAMA_KV_REGION_DEBUG=1 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -ncmoe 24 -c 65536 \
  -t 8 -p hi -n 4 --temp 0 --no-display-prompt --single-turn -v 2>&1 |
  grep -E "KV region reserved|kv_region_grow_cells|KV buffer"
# KV region reserved 1280.00 MiB (VA only, nothing mapped; 4 MiB map unit)
# ROCmKVRegion0 KV buffer size =  1280.00 MiB
```

and confirm the arm is inert by default: a run without the env var must print nothing from this file and
produce byte-identical same-seed text to an r174 build (`../IMPLEMENTATION.md`, "Gates for the slice").
