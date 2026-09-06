# HANDOVER — 2026-09-06 (pre-reboot) — gfx1201 multi-ubatch prefill regression: fixed; tensor-mode follow-up open

Session end state before a planned machine reboot.  The GPU box (`soar`, 3x AMD Radeon
AI PRO R9700 / gfx1201) absorbed ~10 GPU hangs during this session and should be
REBOOTED before further tensor-split A/B work (KFD/driver state degrades after hangs).

## What this session accomplished

1. **Delivery re-baseline to upstream master `465e49b9c`** + campaign date re-stamp
   (delivery commits `0610b75`, `c5da310`) — patches regenerated at the new fork point,
   all docs re-based, the gfx1151 campaign's week-advanced dates collapsed onto the real
   09-05/09-06 git dates.  See those commit messages.
2. **Fresh `qwen4exp` fork branch built** at `~/llama.cpp` (master + 13 blocks +
   consolidated beta patch).
3. **THE REGRESSION — root-caused and fixed** (see below).

## THE regression + fix (main event)

**Symptom** (reported by maintainer): llama-server crashed on the first request
(memory fault in `quantize_q8_1`); llama-bench at `-p 4096/8192 -ub 2048` on
Qwen3.8-Flash-Next-UD-IQ4_XS hung (one GPU pegged in-kernel spin) or faulted
(`quantize_q8_1`, `k_get_rows`).  pp2048 (single ubatch) was fine.

**Root cause**: the gfx1151 campaign's ggml scheduler change
(`ggml_gallocr_reserve_n_probe` in `ggml-backend.cpp`/`ggml-alloc.c`) replaced the
unconditional cross-backend sync in `ggml_backend_sched_alloc_splits` with a probe
that syncs only when a buffer must GROW.  A re-reserve fires on every ubatch chunk
(KV views advance) and re-points tensor addresses; with >1 device the previous
chunk's kernels can still be in flight on other GPUs when the re-point happens →
cross-GPU race → device spin / corruption.  The campaign validated single-GPU
(gfx1151/gfx1100 — Strix Halo is one iGPU); its own records say multi-GPU/gfx1201
validation was PENDING.  On single GPU one stream fully orders the compute, so the
no-sync path is sound there.

**Fix**: keep the probe/no-sync path for **single-device** schedulers (preserving the
validated single-GPU gain) and restore the full sync when >1 async-capable (non-CPU)
device is present.  The gate counts device types, not raw backend count (a GPU + CPU
backend keeps the fast path).  QSA scheduler hunks that are genuinely needed stay.

**Where it lives**:
- Fork branch fix commit: `~/llama.cpp` qwen4exp **`c63f7f2a0`** (parent 627506c1c).
- Delivery: `beta/qwen4exp/qwen4exp-support.patch` regenerated from the branch, delivery
  commit **`d6eb551`** (squashed single fix commit on top of `c5da310`).
- Rebuilt binaries: `~/llama.cpp/build-rocm/bin/{llama-bench,llama-server,llama-cli}`
  (contains the fix; validated).

**Validation (3x R9700, gfx1201, IQ4_XS model)**:
- Layer split, QSA default ON, pp8192 ub2048: **~1650-1700 t/s** (was: hang).
- Tensor split pp8192: **~2071 t/s** (best run — see the OPEN ITEM; flaky).
- QSA-off pp8192: 2857 t/s.  Decode tg64: 36.5 t/s.  pp512: 1127 t/s.
- Unfixed build at the same configs: hang (rc=124, one device pegged).
- Prior qwen4exp branch (pre-campaign scheduler, `~/llama-cpp-qwen4exp-old` build):
  pp8192 layer ~1116 t/s and tensor ~1792 t/s, both reliable — the A/B baseline.

## OPEN ITEMS (next session, after reboot)

1. **Tensor-split multi-ubatch flakiness — STILL OPEN.**  On the fixed build tensor
   pp8192 passed once (2071 t/s) then flaked ~5x (hang), even QSA-off / fusions-off /
   graphs-off / nccl-forced.  The PRIOR build passes tensor 2/2 on the same (degraded)
   box.  Because the box was degraded by ~10 GPU hangs, this MUST be re-A/B'd after a
   clean reboot before concluding.  If fixed-tensor still flakes on a clean box →
   bisect the 13-block fold amendments under tensor-split (mmvq/mmq/quantize base
   kernels; layer-mode testing never exercised tensor-split).  Repro command + watcher
   guidance below.
2. **Single-GPU validation of the gated fix** on the Strix Halo gfx1151 box (and a
   gfx1100 box if available): confirm the open gate preserves the campaign's prefill
   gain and coherence (no regression vs the pre-fix campaign build).  This is the
   whole point of the gate — the gfx1151 perf numbers must be re-measured with the
   gate in place.
3. **IQ3_XXS model download is INCOMPLETE**: shards 2-3 present
   (`/models/Qwen3.8/Flash-Next/IQ3_XXS/`), shard 1 truncated at ~10.9MB.  Re-run the
   `wget --continue` for shard 1 (it was downloading from
   unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ3_XXS) — smaller/faster-loading model for
   iteration.
4. **Upstream-PR candidate**: the campaign flagged the ggml sched change as core-ggml /
   upstream-PR material.  With the single-device gate it is defensible: propose the
   gated version upstream (single-device fast path + multi-device sync), after
   item 2's validation.
5. **Docs**: `beta/qwen4exp/README.md` validation-status section still carries the
   pre-reboot single-GPU narrative; add a dated record for the sched-gate fix + gfx1201
   findings once items 1-2 are closed.  The top-level delivery docs were re-based in
   `c5da310` (done).  GREEDY-PURITY/MANIFESTS unchanged by this work.
6. **Power scheduling**: per AGENTS.md run UNPINNED — the `~/bin/high-power` pin
   (dpm=high + runtime-PM) regressed tg -5-7% / pp -15-18% (session-7 finding).
   The maintainer mentioned power-scheduling issues around this reboot.
7. Keep `~/llama-cpp-qwen4exp-old` (worktree at d4c8e66ae, build `build-old` = prior
   branch, the A/B control).  `~/rdna-bisect` / `~/rdna-vanilla` are pre-existing
   worktrees (left alone).

## Repo state (authoritative SHAs)

```
~/llama.cpp (fork):
  master       465e49b9c   (upstream tip)
  rdna-boosts  c261553a1   (master + 13 blocks)
  qwen4exp     c63f7f2a0   (rdna-boosts + beta + the sched-gate fix)  <== HEAD
  qwen4exp-backup-20260906 d4c8e66ae (prior qwen4exp)
  rdna-boosts-backup-20260906 5b0e45129 (prior rdna-boosts)

~/llama-cpp-rdna-boosts (delivery, branch main):
  d6eb551  beta: gate the no-sync sched re-reserve to single-device schedulers
  c5da310  re-baseline to upstream master 465e49b9c + campaign date re-stamp
  0610b75  patches: restore format-patch mail headers on 0002/0004/0008/0013
  (origin/main still at 51fe8df — everything above is local, NOT pushed)
  patches/ = 13 blocks regenerated at 465e49b9c (canonical 45bf4d291..c261553a1)
  rdna-boosts-all.patch refreshed (45 files)
```

## Key commands

```bash
# repro (was hanging pre-fix):
./build-rocm/bin/llama-bench -fa 1 -m /models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf -ctk q8_0 -ctv q8_0 -ub 2048 -p 8192
# tensor split variant: add  -sm tensor
# QSA off (model fallback): LLAMA_QSA_OFF=1
# A/B control binary (prior branch): ~/llama-cpp-qwen4exp-old/build-old/bin/llama-bench

# rebuild after any branch change (from ~/llama.cpp, qwen4exp branch):
cmake --build build-rocm --target llama-bench llama-server llama-cli -j 16
# full clean rebuild: BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
```

## Methodology pitfalls learned this session (avoid repeating)

- Model load of the 87GB IQ4_XS is ~15-25 s on a quiet box but SLOW under concurrent
  disk I/O (downloads) — do not declare "hung" during/just after a load.  Use a
  stable-VRAM detection + generous post-load budget (watcher scripts were in /tmp and
  are LOST on reboot — re-create: poll `rocm-smi --showmeminfo vram`, treat "loaded"
  as VRAM plateau, budget ~90 s for the bench, kill with `pkill -9 -f "llama-ben[c]h"`
  — the bracket pattern avoids killing your own shell).
- llama-bench with `-ub N` may run BOTH the default ub and N (two result rows); pass
  `-ub 2048` (the default) explicitly and read the n_ubatch column.
- In layer-split, one-GPU-busy is NORMAL (layers run one GPU at a time) — only the
  *stuck* one-GPU-100% + host-at-100%-CPU pattern means a hang.
- After GPU hangs the box degrades; for clean A/Bs prefer a fresh reboot.

## The next-session prompt (paste after reboot)

Pasted separately in the session that requested this handover (also reproduced in the
chat); start the next session with it, referencing this file.
