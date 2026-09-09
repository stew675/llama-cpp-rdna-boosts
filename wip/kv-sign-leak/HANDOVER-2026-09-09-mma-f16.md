# Handover — gfx1151 masked-column KV sign leak: kernel fixes + quantized coverage + coherence done

Updated 2026-09-09 evening. Full record:
`~/llama-cpp-rdna-boosts/wip/strix-halo/kvzero/RECORD-2026-09-09.md` (298 lines:
protocols, matrices, mechanism note, gotchas). Patches:
`wip/kv-sign-leak/0001-*.patch` (Vulkan), `0002-*.patch` (HIP TILE bf16),
`0003-*.patch` (HIP MMA_F16).

## State: the fix work is COMPLETE and validated on both backends

Three commits on `~/llama.cpp` `rdna-boosts` (fork tip `150108fcf`): `d37f107e0`
(Vulkan cm1/scalar), `e9777d5fe` (HIP TILE bf16), `d514fd891` (HIP MMA_F16).
All kernel-side stale-proofing of masked/freed KV cells. Working tree CLEAN.

**Coverage matrix** (see RECORD for detail): every KV type that reaches FA on
either backend is 16/16-gate PASS with `LLAMA_KV_ZERO_FREED=0`, and zeroing
ON == OFF bit-identical over 2064 cells/run where both were run:
- ROCm (F16/BF16/Q8_0/Q4_0 FA-supported; others supports_op=0 -> non-FA, safe):
  f16 PASS+ON==OFF, bf16 PASS+ON==OFF, q8_0 PASS+ON==OFF, q4_0 PASS+ON==OFF,
  q5_0 PASS zeroing-off.
- Vulkan (ALL quant types FA-supported): f16 PASS+ON==OFF, bf16/q8_0/q4_0/q5_0/
  q4_1/q5_1/iq4_nl PASS zeroing-off (q8_0/q4_0 also ON==OFF).

**Coherence**: test-backend-ops FLASH_ATTN_EXT vs CPU full matrix — ROCm0
4591/4591, Vulkan0 7822/7822 passed. CPU same-seed end-to-end: 51/64 greedy
tokens identical, divergence only at a near-tie (expected non-FA-vs-FA).
Depth-16384 llama-bench decode (fixed vs pre-fix .so): tg128 within 0.05% both
KV types; pp16384 within single-run drift.

## Next session task: the delivery proposal (the ONLY remaining item)

Goal: remove or re-gate the gfx1151-only host-side `zero_freed` workaround
(`llama_kv_cache::zero_rows`, auto-enable scans device descriptions for
"gfx1151", env `LLAMA_KV_ZERO_FREED` override) in the delivery repo
`~/llama-cpp-rdna-boosts` as a **block-14 amendment** (per AGENTS.md: regenerate
the block-14 patch from the `~/llama.cpp` fork commit, update `patches/README.md`
block-14 notes + WORKLOG + MANIFESTS headers; never edit the delivery directly).

Proposal sketch (decide before implementing):
1. The kernel fixes live only in `~/llama.cpp` (3 commits). The delivery repo
   ships `patches/` — decide whether this work should be (a) folded into the
   block-14 amendment as the zeroing's replacement, or (b) delivered separately.
   NOTE the AGENTS.md "what NOT to do": WIP stays out of patches until the
   maintainer says so — the user drives this decision.
2. Re-gate alternative: keep `zero_freed` code but disable the auto-enable
   (default OFF) since kernels no longer read freed cells — cheaper, reversible,
   preserves the belt-and-suspenders for any path not yet covered. Kernel
   evidence is per-backend/type (matrix above); Vulkan device-description gate
   never matched ("AMD Radeon 8060S Graphics (RADV STRIX_HALO)" has no
   "gfx1151") — that mismatch means Vulkan runs NEVER had host zeroing, yet the
   fixed shaders make it unnecessary anyway.
3. If removal: delete `zero_rows`/`zero_idxs` wiring + the ctor gate + the env
   override in `src/llama-kv-cache.{cpp,h}` of the fork commit; re-validate with
   the 16/16 gates (zeroing env should become a no-op or disappear) and the
   depth-2048/16384 methodology on the delivery box before regenerating.
4. Regenerate block 14 from the fork per `scripts/make-patches.sh` workflow,
   re-run the clean-apply simulation, coherence gate, build, then commit to the
   delivery repo with a dated WORKLOG entry. This step happens on the delivery
   machine (3-GPU R9700, RCCL/hybrid) or halo depending on where the user wants
   it validated — ask.

## Repo state & gotchas (read before rebuilding)

- `~/llama.cpp` clean at `d514fd891`; build dirs carry the committed .so's.
  Reference baselines: `/tmp/libggml-hip-orig2.so` (= e9777d5fe, pre-MMA-fix),
  `/tmp/libggml-hip-fixed.so` (d514fd891), `/tmp/libggml-vulkan-orig.so`.
  Do not rebuild Vulkan from a git-reverted tree (mtime shader-gen staleness);
  real file is `build-vulkan/bin/libggml-vulkan.so.0.23.0`.
- Probe/dump/diag binaries in `wip/strix-halo/kvzero/`; fattn-probe.cpp now
  accepts q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl but raw-tensor quantized FA **crashes**
  (in-place f16 conversion scratch is laid out for llama.cpp's buffer topology,
  not the probe arena) — see RECORD. Quantized coverage is done via llama-server
  gates (production path) — use run-gate.sh + the ON/OFF cell diff snippet.
- Runtime: `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$HOME/llama.cpp/build-rocm/bin`;
  Vulkan: `GGML_VK_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$HOME/llama.cpp/build-vulkan/bin`.
- Gate cleanup: run-gate.sh kills its server on exit; verify no strays with
  `ps aux | grep llama-server` between gates. llama-cli is interactive-hanging on
  this box — use llama-server /completion for end-to-end checks.
- Rebuild HIP fast: `cmake --build build-rocm --target ggml-hip -j 16`.
