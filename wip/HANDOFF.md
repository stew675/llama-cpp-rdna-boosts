# SESSION HANDOFF — decode-path tuning (2026-08-29, post-compaction)

## NEXT TASK (user request, 2026-08-29): review the repo + write AGENTS.md

The user wants a FRESH context to: (1) review the ENTIRE repo
(`~/llama-cpp-rdna-boosts/`), and (2) create an `AGENTS.md` at the repo root
describing the repo's purpose, layout, workflows, and how to work in it.

Key orientation for that task:
- This repo = the PATCH/DELIVERY repo for the `stew675/llama.cpp` fork's
  `rdna-boosts` branch (blocks 01-11 + the hybrid HIP all-reduce block 12).
- `patches/` = the delivery patch set (0001-0011 + 12-hybrid-allreduce-hip.patch)
  + README.md (apply instructions).  scripts/apply-all.sh = the verified apply
  flow (git am for 1-11, git apply for the 12th); scripts/make-patches.sh =
  regeneration.
- `wip/` = the exploration docs (HANDOFF.md = this file; 12-hybrid-allreduce-hip.md;
  tools/).  `archive/work/fused-stage-pacing/` = the closed experiments
  (fused-stage + pacing), preserved for a future ROCm re-evaluation.
- `benchmarks/` = the benchy methodology + v2-results; MANIFESTS.md/BASELINE.md
  = the block manifests + baselines; GREEDY-PURITY.md = bit-identical decode.
- The fork lives at `~/llama.cpp` (branch `rdna-boosts`, rebuilt clean as
  12 commits, tip `12d10267b`; historical fork on `old-rdna-boosts`,
  HEAD 155debcdc);
  builds: `build-rocm-hybrid` (the hybrid, current); the server config at
  `~/.llama-server-config.yaml` now points rocmstew1/2/3 at the hybrid build.
- Verified numbers (2026-08-29): 3-GPU hybrid depth-16384 38.71 t/s; tg512
  41.08; the clean-apply simulation passed (build + coherence + perf).

---


## Session 9.5 — clean patch-set apply: VERIFIED (2026-08-29)

Simulated the delivery patch set against a clean llama.cpp master
(`17252c769` = the fork point) in a fresh worktree:

1. `git am` blocks 1-11 (`git format-patch 17252c769..f6f8f6778`; plain
   `git apply` of the concatenated series SILENTLY DROPPED HUNKS — 30 files
   / 2483 lines vs the correct 35 files / 6094 lines — use `git am`).
2. The 12th patch (regenerated from the CURRENT fork tree as the delta vs
   f6f8f6778 — the previous wip patch's ggml-cuda.cu was STALE, missing
   block 10's q8_1 fusions, which broke the 4B decode with a silent
   generation-0 hang; the fresh 12th fixes it).
3. Full build: clean.
4. Coherence: 4B llama-cli same-seed output IDENTICAL to the fork build
   (76.7 t/s).
5. Performance holds: tg64 38.12 (fork 38.28), tg512 41.08 (fork 40.95-41.60).

**RDNA4-only gate added** (block 12): the internal/hybrid all-reduce refuses
to init unless every device is gfx1200/gfx1201 (falls back to RCCL with a
warning).  Community RDNA3 verification pending.

Delivery artifacts: `patches/0001-0011` (blocks) + `patches/12-hybrid-allreduce-hip.patch`
+ `patches/README.md` (apply instructions).  The stale wip/12-hybrid-allreduce-hip.patch
is superseded by patches/12-hybrid-allreduce-hip.patch.

WIP experiments (fused-stage, pacing) archived at `archive/work/fused-stage-pacing/`
(env-gated OFF by default; not part of the delivery).

---

## Session 9.6 — whitespace-clean regeneration: VERIFIED (2026-08-29)

`apply-all.sh` on a clean master printed git whitespace warnings on every
apply (8 trailing-whitespace lines — pure-whitespace blank lines in block
02's two bf16 GDN files — plus one blank-at-EOF line in block 12's
`allreduce.cuh`).  Cleaned at the SOURCE (no warning suppression): the
fork's `rdna-boosts` block commits were rebuilt in place (each commit's
diff re-applied with `git apply --whitespace=fix`, preserving messages and
author dates) and the whole set re-generated with `scripts/make-patches.sh`
(blocks tip `cc985ba9a`, block 12 committed as `12d10267b`).

Verification:
- Fresh `apply-all.sh` on a clean checkout at `17252c769`: **ZERO
  whitespace warnings** (exit 0, clean log).
- Applied tree byte-identical to the previous applied tree except the 8
  whitespace lines + 1 EOF blank (all inert; no string-literal or
  continuation content) — verified via tree-object diff.
- `rdna-boosts-all.patch` (root convenience file) regenerated from the
  clean range and re-checked (`git apply --check` on baseline): OK.  It
  previously MISSED the block-12 kernel file `allreduce-hip.cu`.
- Fork state: `~/llama.cpp` `rdna-boosts` = clean 12-commit branch (block
  12 committed, tip `12d10267b`); old fork history preserved on
  `old-rdna-boosts` (`155debcdc`).  Build/coherence re-run not needed:
  tree equivalence proven at the git level (whitespace-only).

---


## Where things stand

- **Block 12 (hybrid all-reduce): DELIVERED** as `patches/12-hybrid-allreduce-hip.patch` (clean hybrid + RDNA4-only gate, WITHOUT the fused/pacing WIP). Clean-apply simulation VERIFIED (build + coherence + tg64 38.12 / tg512 41.08). The fork's `rdna-boosts` is now the whitespace-clean rebuilt branch, block 12 committed, tip `12d10267b` (see session 9.6); the historical fork HEAD `155debcdc` (the gate commit) is preserved on `old-rdna-boosts`.
- **WIP experiments**: fused-stage + pacing archived at `archive/work/fused-stage-pacing/` (env-gated OFF by default; not part of the delivery).
- **Server config**: `~/.llama-server-config.yaml` rocmstew1/2/3 now point at the hybrid build (`~/llama.cpp/build-rocm-hybrid`). Recommended deployment: 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`), hybrid default, UNPINNED (the pin is a regression).

## Current server env (recommended, verified — superseded 2026-08-29 pm)

The pre-session-7 recommendation (pair `1,2`, pin on) was REVERSED: the
`~/bin/high-power` pin is a REGRESSION (tg -5-7%, pp -15-18% on RCCL/hybrid
paths), and the deployment moved to 3-GPU unpinned. Current config:

```
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_DISABLE_GRAPHS=0
```
(no pin, no `NCCL_P2P_DISABLE=1` — hybrid handles the split). Optionally
explicit: `GGML_CUDA_ALLREDUCE=hybrid`. Old values `nccl|internal|none` keep
old semantics.

## Verified numbers (27B Q8_0, depth 16384, 2 GPU)

| config | tg t/s |
|---|---|
| RCCL SHM (`NCCL_P2P_DISABLE=1`), pair 0,2 | 31.62 |
| RCCL P2P, pair 0,2 | 30.41 |
| hybrid, pair 0,2 | 31.15-31.47 |
| **hybrid, pair 1,2** | **32.34** (+2.8% vs 0,2) |

No-depth (llama-bench tg64, r2): 2-GPU (1,2) 33.2; **3-GPU (any order)
38.1** (+14.6% decode).  Single-GPU all cards identical: 20.67/20.68/20.69.
All PCIe 5.0 x4.

## Session 4 state (n>=2 generalization, 2026-08-29)

- **Chunked internal AR now supports n in [2,8]** (star gather; contiguous
  host_wire; CE path stays n==2).  New env knob carried: GGML_CUDA_AR_SLEEP.
- **Fixed a silent-corruption bug in phase 3** (whole-vector accumulation)
  that the n>=2 rewrite introduced — caught by llama-cli coherence check,
  isolated by `wip/tools/ar_kernel_unit.cpp` (unit test of the extracted
  kernel: n=2/3 x F32/BF16 wire, all PASS).  Coherence restored: 2-GPU
  internal == 3-GPU internal == RCCL output.  tg was NEVER a correctness
  signal for this bug.
- **3-GPU answer to the per-hop question**: NOT per-hop.  Per-call AR cost
  +5 us only (30 -> 35 us); the ~20-25 us wait is one/two devices arriving
  late (driver dispatch class), not chaining.  tg64 3-GPU 38.1.
- **NEW tuning target**: the no-spin AR kernel floor is ~16-18 us/call
  (phase-1/3 host staging + fences).  Decompose next.
- **Power-state verdict (session 5)**: real but partial.  Idle cards are
  runtime-suspended at sclk S/0MHz/mclk 96MHz/13W.  Pinning (dpm=high +
  runtime on) recovered ~7 us of the one-sided lag (25->18 us); cards run
  2350MHz/99% busy through decode.  COMPUTE profile + manual sclk lock:
  dead ends (crash / ignored).
- **Phase-1 decomposition**: entry->signal is SYMMETRIC (4.5 us all
  devices) => the remaining ~12 us one-sided wait is pre-kernel DISPATCH
  (dev0's AR kernel starts late), not power, not data path.  Decisive
  reboot test: `amdgpu.dpm=0` (S state persists even at high).  Stable
  metric: tg512 = 39.69 ± 0.41 (3-GPU, pinned).
- Server note: keep the pin (`~/bin/high-power` + runtime PM on) — ~7
  us/call ≈ 2% decode, costs ~30 W/card idle. **SUPERSEDED by session 7:
  the pin is a net REGRESSION (tg -5-7%, pp -15-18% on RCCL/hybrid); the
  server must run UNPINNED.**

## Key findings (do not re-derive)

1. Graph work is balanced (rocprof: every kernel replicated, totals 0.8% apart).
2. Signal round trip is fast (isolated: 1.4-4.8 us).
3. A stable ~15-20 us/call one-sided wait (dev1 waits dev0) costs ~6% decode.
4. **It is in the driver/runtime dispatch path**: refuted launch orders, pool
   depth, graphs, -ts, -mg; rocprof tracing makes it vanish.
5. RCCL cannot mix P2P+SHM in one process (maxP2pPeers=0 rejected, env global).

## Repro commands

```bash
# per-call AR profile (27B, any pair; prints at teardown)
HIP_VISIBLE_DEVICES=1,2 GGML_CUDA_AR_PROFILE=1 \
  build-rocm-hybrid/bin/llama-bench -m /llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -sm tensor -mg 0 -p 16 -n 64 -b 512 -r 2

# depth-16384 protocol (server must be running; see benchmarks/benchy-methodology.md)
~/ab-results/ab-16384.sh          # arms: hybrid / nccl / internal, saves to ~/ab-results/16384/

# isolated signal round trip
HIP_VISIBLE_DEVICES=1,2 ~/llama-cpp-rdna-boosts/wip/tools/ar_signal_bench (build first)

# per-device kernel work distribution
rocprofv3 -d /tmp/rocprof-out -r -- <cmd>   # then query the sqlite kernels table
```

## Item B — the rocprof mystery: RESOLVED (session 7/8, 2026-08-29)

Workflow fixed: rocprofv3 SIGABRTs on 7.1.10 in the default rocpd writer —
use `--kernel-trace --output-format csv` (works, exit 0).

Diagnostic (rocprof trace of 3-GPU decode, 2432 AR calls, aligned by index):

- dev0 (HIP 0 = bus 06 = rocprof Agent 2) is the late card: its AR kernel
  starts +12.7 us (p50) late vs the other two (+0.5 us).
- Decomposition: prev-kernel END +7.9 us late + AR-adjacent gap
  (prev-end -> AR-start) +4.8 us.
- The last pre-AR kernel is the SAME op (mul_mat_vec_q, 50.7-50.8 us) on
  all agents; the full kernel census is SYMMETRIC (equal types/counts);
  AR durations differ ONLY by the spin (dev0 13.3 us vs peers 27.0 us —
  the AR absorbs the lateness and ends aligned).
- Intra-step dispatch-gap sums: dev0 43.5 us vs 38.4/41.7 (p50).
- => the wait is per-device kernel-to-kernel DISPATCH-gap asymmetry at the
  KFD/CP level on bus 06 (~0.5 us/kernel x 9 + ~4.8 us AR-adjacent).  NOT
  op composition, NOT the signal path, NOT the AR kernel (phase-1/3
  symmetric).

Why rocprof "fixes" it: its per-dispatch tracing overhead re-paces the
host's enqueue stream (HANDOFF hypothesis 2, pacing, CONFIRMED) so the
per-step gap accumulation never builds.  It's perturbation, not a fix (its
overhead costs ~20% tg).

Replication verdict: the asymmetry is CP-level dispatch latency; it cannot
be fixed from the AR kernel.  The cross-stream AR variant cannot change CP
dispatch latency (and session-1 already showed the event handshake costs
more than it saves).  The only real fix would be fusing phase-1 staging
(shard->wire write + arrival token) into the END of each device's
subgraph graph, so peers stop spinning at dev0's graph end instead of
+12.7 us later — the HANDOFF's "split stage/consume" design, a deep
integration (graph construction, ggml-backend-meta) worth ~2-3% decode;
NOT attempted, documented as future work.

Dead end re-checked: GGML_CUDA_DISABLE_GRAPHS=1 is a WASH for tg (tg512
41.56 vs 41.60, tg64 41.26 vs 41.34) — no graphs change needed.  (An
apparent +18% for graphs-off in an r1 run with AR_PROFILE=1 was an
instrumentation artifact; re-measured clean.)

## Baselines on 7.1.10-200 (the new reality), UNPINNED (2026-08-29 pm)

Depth-16384 tg240 runs=5, Q8_0 bf16 KV, tensor split:

| config | tg | pp |
|---|---|---|
| 3-GPU hybrid | **38.71 ± 0.08** | 1896.7 |
| 3-GPU internal | 38.77 | 1539.7 |
| 3-GPU nccl | 36.24 | 1889.2 |
| 2-GPU (1,2) hybrid | **31.79 ± 0.02** | 1730.7 |
| 2-GPU (1,2) internal | 31.75 | 1446.6 |
| 2-GPU (1,2) nccl | 30.18 | 1731.7 |

tg512 (no-depth, 3-GPU, llama-bench r2): 40.95 ± 1.01.

3-GPU vs 2-GPU hybrid = +21.8%.  7.1.10-200 costs ~1-3% vs 7.1.8-200 when
UNPINNED (uniform across all 6 arms; masked when pinned => clock-management
mechanism).  Not worth fighting (pin costs more).  Server: 3-GPU,
HIP_VISIBLE_DEVICES=0,1,2, no pin, hybrid default.

## Session 7 results (2026-08-29 pm) — PIN IS THE REGRESSION, kernel exonerated

User's hunch validated: the `~/bin/high-power` pin (dpm=high + runtime-PM on)
REGRESSES the RCCL/hybrid paths by tg -5-7% and pp -15-18%.  Session-5's
"pin recovers ~7us of the AR wait" was an artifact / AR-only benefit that is
outweighed end-to-end (pin helps the AR spin slightly, hurts the rest more).
The kernel update 7.1.10-200 (installed today 13:27) is EXONERATED:
identical numbers to 7.1.8-200 in every corner.

A/B matrix, same depth-16384 tg240 protocol (Q8_0, bf16 KV, tensor split):

| config (3-GPU) | tg | pp | notes |
|---|---|---|---|
| hybrid, UNPINNED | **39.95 ± 0.01** | 1963.5 | record; +23.5% vs 2-GPU |
| internal, unpinned | 39.93 | 1588.0 | decode == hybrid (expected) |
| nccl, unpinned | 36.57 | 1960.6 | internal beats RCCL +9.2% |
| hybrid, pinned | 37.35 | 1668.2 | pin cost: -6.5% tg, -15% pp |
| v2 binary RCCL, unpinned | 36.50 | 1956-1961 | residual -4.3% tg vs 08-27's 38.12 |

2-GPU (1,2): hybrid unpinned **32.49** (reproduces the original 32.34; the
30.84 pinned was the pin), nccl 30.34, internal 32.50.

- Ladder-vs-standalone: no artifact (in-ladder cell@16384 = 36.49 vs
  standalone 36.53).
- RCCL tg residual -4.3% vs 08-27 while pp +11% better: unexplained, likely
  cmdline drift (iommu=off/pcie_aspm=off added since); minor — internal AR
  dominates anyway.  Links verified fine (x16@32GT/s, P2P 16.0 GB/s).
- CONCLUSION: server must run UNPINNED (remove high-power from the config;
  unpinned idle is also cheaper, 13-19W vs 44-52W).  Move to 3-GPU
  (HIP_VISIBLE_DEVICES=0,1,2): depth-16384 decode 32.5 -> 39.95 (+23.5%).
- Results: ~/ab-results/16384-{3gpu,2gpu12,v2repro}-unpinned/,
  16384-v2ladder/ (+-newkernel/ backups).

## Next steps (in priority order)

1. **3-GPU depth-16384 validation** (deployment win, no code): server with
   `HIP_VISIBLE_DEVICES=0,1,2` + benchy depth-16384.  No-depth already shows
   3-GPU 39.7 vs 2-GPU 33.2 (+19%).
2. **Subgraph-end diagnostic** (measure, no code): rocprof the per-step
   end-times of the last pre-AR kernel per device.  If dev0's subgraph
   consistently ends ~12 us later (non-split ops land on dev0 via
   tensor_config/rotation in llama-model.cpp:788-830 and ggml-backend-meta),
   the one-sided wait is a software-visible per-step stagger, not a driver
   wall.  Look for split-state MIRRORED/PARTIAL axis placement.
3. **rocprof-mimic experiments** (see below) — cross-stream AR variant
   (pre-queue AR kernels on their own streams + subgraph-done events) or
   host-side lockstep pacing (wait for peer's prev-1 AR event before
   enqueueing).  Both mimic rocprof's pacing; measure the stagger collapse
   vs the added overhead.
4. Decisive reboot test: `amdgpu.dpm=0` — **DONE, FAILED (session 7)**: on
   this kernel (7.1.10-200.fc44) + RDNA4 (gfx1201), amdgpu refuses to init
   ALL three R9700s (1002:7551) with dpm=0 — only the iGPU (1002:13c0,
   gfx1036) gets a drm/KFD node.  No drm card, no KFD node, no driver
   binding for the discrete cards.  Option C is CLOSED: dpm=0 is
   incompatible with RDNA4 SMU init.  Reverted via /etc/kernel/cmdline
   (removed `amdgpu.dpm=0`; kept iommu=off, processor.max_cstate=2,
   pcie_aspm=off, ppfeaturemask).  The `S`-state question stays open;
   dpm=high pin remains the practical answer.
5. Base-cost decomposition (phase-1 fence cost, write-combined host
   mapping, kernel blocks) — the ~16 us no-spin floor is ~2 ms/token at
   108 calls/token even with perfect sync.

Prize if the dispatch wait is removed: ~34 t/s at depth-16384 (from 32.34).

## Fused-stage experiment (session 8, 2026-08-29) — NEGATIVE, code REVERTED

Attempted the "fuse phase-1 staging into the subgraph tail" design
(handoff item 3, ~2-4% decode promised by removing the graph->AR dispatch
premium on the late card).  Built and tested FOUR designs; ALL net-negative
or broken.  Code fully REVERTED to the clean pre-fusion baseline (working
tree == the wip patch state; `git status` shows the same 4 files as always).

**MECHANISM PROVEN**: fusing the stage (shard->wire + arrival) into the
captured graph as its last node ELIMINATES the one-sided wait: the decode
spins collapsed from ~21/22 us (dev1/dev2) to ~2.4/2.4 us in the first
compute-kernel variant (v1).  The wait is real and the fusion concept is
sound.

**PLATFORM CONSTRAINTS FOUND (the killer)**:
1. `__threadfence_system()` inside a HIP graph replay costs ~50 us on RDNA4
   (measured: no-copy stage = 53-63 us; the same fence outside the graph is
   ~1-2 us).  The fence drains the graph's pending memory ops.  This alone
   makes any compute-kernel in-graph GTT write + fence net-negative.
2. The wire-before-arrival ordering requires the fence IN the kernel that
   wrote the wire (thread-scoped fences; a later kernel's fence does NOT
   order an earlier kernel's GTT writes — v3 split the copy/stage from the
   arrival and produced SILENT CORRUPTION, caught by the llama-cli coherence
   gate).
3. Cross-device SDMA D2H memcpy nodes (v4: convert->memcpy wire->memcpy
   arrival, the SDMA completion as the visibility barrier) hit capture /
   ordering / multi-block-spin issues (page faults, deadlock from blocks
   1-7 spinning on unwritten arrival slots, then corruption).  v4 also
   needed per-device control buffers (cross-device D2H source faulted).

**CAVEAT — measurements suspect**: the GPU wedge that forced the final
reboot (KFD hang, rocm-smi "map::at" exception) may have been BUILDING
during the fusion experiments (repeated aborts/deadlocks/faults).  The
v1-v4 tg numbers (26-34 t/s) and the 50 us fence measurement may be
contaminated by a degrading GPU state; the *structural* findings (fence
cost direction, ordering requirement) are sound but the exact magnitudes
need re-verification on a healthy machine if the thread is ever resumed.

**To resume the investigation (if ever)**: re-run the fusion A/B on a
healthy machine AFTER a fresh reboot; the cheapest next variant is the
v4-SDMA with the 1-block fused reduce (the multi-block spin deadlock is
fixed; the corruption + ordering remain unsolved).  Recommended gate:
coherence (llama-cli same-seed) BEFORE any tg measurement, and monitor
rocm-smi for the wedge (rocm-smi breaking = STOP).

**Session 8 state**: GPUs wedged at session end; reboot required.  After
reboot: verify `uname -r` = 7.1.10-200, pin = auto (unpinned — the
known-good config), then ONE coherence + tg512 sanity check (expect ~41.6
with the healthy baseline) before any further work.  All session-8 commits
preserved; the fusion code is NOT in the tree.

## Fused-stage + pacing: DEFINITIVE CONCLUSION (session 9, 2026-08-29)

The last two levers for the ~12.7us dispatch wait were implemented, tested on
the HEALTHY machine, and CLOSED:

1. **Fused stage (v2 compute-kernel) — COHERENT but net-negative.**
   - v4 SDMA variant is structurally dead: D2H hipMemcpyAsync nodes are NOT
     captured into HIP graphs (16 kernel nodes, 0 memcpy nodes); the memcpys
     execute once at hook time, copying uninitialized scratch.
   - v2 (convert + GTT wire + writer fence + arrival in ONE captured kernel,
     control-read at replay; 8-block reduce + all-block arrival slots):
     bit-exact wire (STAGE w0=3f96 == REDUCE w0=3f96 per token), llama-cli
     same-seed IDENTICAL.
   - Perf (27B tg64): FUSED=1 30.87 vs FUSED=0 34.45.  Mechanism: the
     one-sided wait PERSISTS (dev0 spin collapsed 3.5us; dev1/dev2 still
     20-22us — the subgraph-END asymmetry, not the AR dispatch); the in-graph
     GTT write + fence drains the graph's L2 and costs more than the phase-1
     it replaces; the validation's volatile reads add ~4us/call.
2. **Host-side lockstep pacing (GGML_CUDA_AR_PACE) — WASH.**
   - Gate: busy-poll peers' prev-slot end-of-AR event before enqueueing.
   - Decode AR 36.0 vs 36.2us/call; tg64 34.53 vs 34.45 — no change.
   - The gate is REDUNDANT with the AR's own barrier (the previous call is
     long done; the stagger is the subgraph-END difference).

CONCLUSION: the dispatch-gap asymmetry (dev0/bus-06 per-kernel dispatch
latency, Item B) is a platform-level CP/driver property.  It is NOT
reachable from the application: not from the AR kernel, not from the graph
tail (fusion), not from host-side pacing.  The rocprof "fix" is a
per-dispatch GPU-side scheduling perturbation that cannot be replicated
deliberately.  Decode stands at the measured baselines (3-GPU hybrid 38.71
at depth-16384, tg512 41.6).  This thread is CLOSED with high confidence.

State: GGML_CUDA_AR_FUSED (default 0) and GGML_CUDA_AR_PACE (default 0) are
both env-gated WIP artifacts, coherent, no default-behavior change.

## rocprof-mimic investigation (session 6 idea, from the user)

Question: rocprofv3 tracing makes the one-sided AR wait disappear (pair
0,2: 26 us without, ~balanced with).  Can we mimic its mechanism in the
custom all-reduce?

Mechanism candidates (unresolved):
1. keeps GPUs hot (power) — refuted as full explanation: power is only ~7 us.
2. paces dispatches (per-dispatch completion waits -> devices advance in
   lockstep -> no race window).
3. perturbs per-step subgraph END times (AR kernel runs behind the
   device's own subgraph on the same stream; if dev0's subgraph ends
   ~12 us later each step — balanced totals hide a consistent offset from
   non-split op placement — dev0's AR kernel starts late; rocprof's
   overhead re-aligns step boundaries).

Discriminate #3 first (pure measurement): rocprof per-step end-times of
last pre-AR kernel per device.

Mimic experiments:
- Cross-stream AR variant: enqueue all AR kernels on p->streams BEFORE the
  subgraphs finish, gated on subgraph-done events -> synchronized pickup.
  Note: tried once before and rejected for tg latency overhead — revisit
  with the stagger lens (event overhead vs the ~12 us it removes).
- Host-side lockstep pacing (cheapest): before enqueueing AR call N on
  dev0, wait for dev1's call N-1 AR completion event (event pool exists).
  If a ~1-2 us host sync collapses the ~12 us stagger, pacing is the
  mechanism.

## External cross-check (session 3)

u/nasone32 (r/ROCm, 2026-08-29): dual 7900xtx behind chipset x4 — RCCL
broken, ported internal AR to HIP (his fix = our block 12, validated),
s_sleep(1) poll (adopted, wash), Q8 wire for internal prefill (+27% PP on
bandwidth-starved x4 — only relevant for internal-mode large tensors, not
our baseline; gated option if ever needed).  Details in
`12-hybrid-allreduce-hip.md` §session 3.

## Files of record

- `wip/12-hybrid-allreduce-hip.md` — full exploration writeup + tuning roadmap
- `wip/12-hybrid-allreduce-hip.patch` — the llama.cpp delta (incl. profiler)
- `wip/tools/` — standalone benchs (ar_signal_bench, rccl_ar_bench, probes)
- `~/ab-results/16384/` — benchy JSONs (armA/B/C old build, armH* new build)
- `~/llama.cpp/build-rocm-hybrid/` — current build with the patch + profiler
