# SESSION HANDOFF — decode-path tuning (2026-08-29, pre-context-compaction)

## Where things stand

- **WIP block 12 (hybrid all-reduce)**: implemented, built (`build-rocm-hybrid`),
  validated, preserved as WIP in `wip/12-hybrid-allreduce-hip.{md,patch}`.
  NOT integrated into the apply chain. Commits: `dd66c07`, `153f975`.
- **Working tree**: `~/llama.cpp` has 3 changed files, uncommitted:
  `ggml/src/ggml-cuda/allreduce-hip.cu` (new: HIP port + env-gated profiler),
  `allreduce.cu` (guard), `ggml-cuda.cu` (hybrid dispatch + is_small).
- **User's server configs updated** to the recommended pairing
  (`HIP_VISIBLE_DEVICES=1,2` = cards 09+03; or 2,1). No `NCCL_P2P_DISABLE`.
  Default allreduce is now `hybrid` (Linux).

## Current server env (recommended, verified)

```
HIP_VISIBLE_DEVICES=1,2 NCCL_PROXY_CPUSET=8,9,10,11,12,13,14,15 GGML_CUDA_DISABLE_GRAPHS=0
```
(no `NCCL_P2P_DISABLE=1` — hybrid handles the split). Optionally explicit:
`GGML_CUDA_ALLREDUCE=hybrid`. Old values `nccl|internal|none` keep old semantics.

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
  us/call ≈ 2% decode, costs ~30 W/card idle.

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

## Next steps (in priority order)

1. **Driver-level probe**: from a rocprof kernel trace, measure submission-to-
   start latency per device (kernels.start vs host enqueue) to name the
   mechanism behind the ~20 us/call wait.
2. **Overlap fallback** if unfixable from userspace: split stage/consume pool
   so the next call's phase-1 starts during the peer's spin (or prefetch next
   layer's independent ops while spinning).
3. Leftover micro-tunings (kernel blocks, wire type, pipelining).  s_sleep
   poll already adopted (wash vs dummy spin; env GGML_CUDA_AR_SLEEP=0 to
   revert).  Q8 wire: NOT baseline (user policy: quality; no decode win, no
   RCCL-path win) — only if pursuing RCCL-less internal-mode prefill, gated
   GGML_CUDA_AR_WIRE=q8.

Prize if the dispatch wait is removed: ~34 t/s at depth-16384 (from 32.34).

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
