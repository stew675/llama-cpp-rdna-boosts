# Handover — gfx1151 masked-column KV sign leak: kernel fixes landed, remaining validation

Updated 2026-09-09 evening. Full session record (protocols, numbers, gotchas):
`~/llama-cpp-rdna-boosts/wip/strix-halo/kvzero/RECORD-2026-09-09.md`.
Patches: `wip/kv-sign-leak/0001-*.patch` (Vulkan), `0002-*.patch` (HIP TILE bf16),
`0003-*.patch` (HIP MMA_F16).

## Status: kernel-side stale-proofing is COMPLETE on ROCm and Vulkan

Three commits on `~/llama.cpp` branch `rdna-boosts` (fork tip `150108fcf` + these):

| commit | kernel | fix |
|---|---|---|
| `d37f107e0` | Vulkan `flash_attn_cm1.comp` + `flash_attn.comp` | never read V of fully masked columns |
| `e9777d5fe` | HIP `fattn-tile.cuh` packed native-BF16 PV | zero per-warp V registers of masked KV rows |
| `d514fd891` | HIP `fattn-mma-f16.cuh` WMMA prefill (f16+bf16) | zero staged V tile rows whose mask row is blocked for every block query column |

Validated on gfx1151/ROCm + RADV (Vulkan), all against pre-fix `.so` A/Bs:
- probe clean across f16/bf16 × GQA/MHA × hs 128/256 × nvalid sweep (incl. the
  non-monotonic leaky boundaries 1265/1275/1279);
- fully-live, zeroed-tail and orig-clean configs BIT-IDENTICAL to pre-fix;
  orig-leak configs now differ (leak removed);
- partial-column diag unchanged (f16 clean; bf16 ~2.8e-14 documented live-V
  artifact, out of freed-cell scope);
- 16/16 determinism gates PASS with `LLAMA_KV_ZERO_FREED=0` on f16 AND bf16 KV;
  zeroing ON ≡ OFF bit-identical over all 2064 gate cells on both;
- llama-bench pp512/pp2048/tg128 within +0.2% (noise).

**Mechanism note (surprising):** on HIP MMA the leak is NOT a plain row-level
"P=+0 × V≠0" effect. Triangular causal masks are EXACT in orig (even with
nonzero V in fully-dead rows); the leak fires under column-uniform masks and is
non-monotonic in geometry + K-tail content at some boundaries. The empirical
criterion (never feed a fully-dead row's V to the mma) removes every observed
signature and is behavior-neutral where orig was exact. Details in RECORD.

## Next session tasks (remaining before delivery proposal)

1. **Quantized KV cache types** — probe coverage on both backends:
   `q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl`. HIP converts them to f16 in the FA
   launcher (`need_f16_K/V` — verify); Vulkan reads natively
   (`USE_DECODE_K/V`). Host zeroing's "+0.0 invariant" only holds for these by
   layout luck. Extend `fattn-probe.cpp` to take a quant type (it currently
   does f16/bf16 only); run the same garbage-tail A/B. If leaks appear, the fix
   pattern is identical (they funnel into the same V staging paths).
2. **CPU coherence sweep** — repo gate: same-seed llama-cli GPU vs CPU for a
   short prompt; FLASH_ATTN_EXT vs CPU NMSE if feasible. Confirm the three
   commits don't move outputs on exact paths (they shouldn't: bit-identical A/B
   already shown vs pre-fix on live/zeroed content).
3. **Depth-16384 decode perf** methodology
   (`benchmarks/mtp-adaptive-methodology.md`) on the final builds — the
   pp-level llama-bench checks passed, but the repo's decode gate is at depth
   16384. Decode paths (VEC/TILE) are untouched except the bf16-TILE V-register
   fix; expect no movement; verify anyway.
4. **Delivery proposal** (only after 1-3): remove or re-gate `zero_freed` in
   `src/llama-kv-cache.cpp` (block 14 amendment in `~/llama-cpp-rdna-boosts`,
   regenerate the block-14 patch per AGENTS.md, re-run clean-apply + coherence).
   NOTE: the block-14 auto-gate keys off the device description containing
   "gfx1151", which never matches the Vulkan device string ("AMD Radeon 8060S
   Graphics (RADV STRIX_HALO)") — flag this in the proposal. Decide scope: the
   fixes are per-backend/per-type; zeroing removal needs all covered paths
   stale-proof (hence task 1).

## Repo state & gotchas (read before rebuilding)

- `~/llama.cpp` working tree CLEAN at `d514fd891`; build dirs
  `build-rocm` (HIP) and `build-vulkan` carry the committed .so's. Do not
  rebuild Vulkan from a git-reverted tree (mtime-based shader gen silently
  skips regeneration — `touch` the .comp files and md5-verify the .so; the
  real file is `build-vulkan/bin/libggml-vulkan.so.0.23.0`).
- Probe/dump/diag binaries: `wip/strix-halo/kvzero/fattn-probe-rocm`,
  `fattn-dump-rocm`, `fattn-diag-rocm` (dlopen a libggml-hip.so given as argv;
  runtime `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$HOME/llama.cpp/build-rocm/bin`).
- Reference pre-fix baselines kept: `/tmp/libggml-hip-orig2.so` (pre-MMA-fix =
  at e9777d5fe), `/tmp/libggml-hip-fixed.so` (= d514fd891). Vulkan orig:
  `/tmp/libggml-vulkan-orig.so`.
- Rebuild HIP fast: `cmake --build build-rocm --target ggml-hip -j 16`.
- Server gates: `run-gate.sh <bin-dir> <model> <label> [--cache-type-{k,v} ...]`
  with `LLAMA_KV_ZERO_FREED=0/1` exported; 16 requests × 129 tokens; per-cell
  top-8 logprobs at full JSON precision. Zeroing-ON vs OFF diff: compare the
  `runs/<label>/gate129/runNN.json` snapshots pairwise.
- No parallel benches; verify hardware determinism across fresh instances before
  trusting cross-run A/Bs.
