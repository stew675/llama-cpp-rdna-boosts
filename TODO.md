# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.

**Forward-looking only.**  This file lists open work, items waiting on someone else, accepted limitations
and parked ideas.  **A closed or resolved item is not kept here** — it is one dated entry in
[`WORKLOG.md`](WORKLOG.md) plus the campaign record that entry points to (`archive/work/<campaign>/`).
Details never live here: they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `ENVIRONMENT.md`, `wip/*` and `benchmarks/`.

**Numbers are never reused.**  `#NN` is a permanent handle: an old `WORKLOG.md` entry that cites `TODO #74`
still means the same subject after that item is closed and removed from this file.  New items append.

**Current state:** the **16-block V2.1 set** (block 00 + blocks 01-15) at fork point **`6050b467c`**
(the 2026-10-25 re-base), release **`v16-6050b467c-r187`**.  `release.json` is the source of truth for the
tip/tree and hashes; the top of `WORKLOG.md` is the newest record.  **Block 14 is temporary** (item #104).

| # | subject | kind | status |
|---|---|---|---|
| **118** | windowed KV cache ("Strata-style"): a **fixed VRAM window** instead of `O(-c)`, so (A) a 27B Q8_0 or 35B-A3B Q6_K can run a large context on one 32 GiB card instead of OOM, and slab mode gets the freed VRAM for the decode arena; (B) a sparse attended set (QSA/DSA) streams cheaply; (C) the drafter ring (campaign `wip/kv-residency/`) | performance / memory | **OPEN — the ask is Goal A, order A -> B/C. `O2-PLAN.md` is the plan of record; `A2-MIXED-BACKING.md` is the state of the build.**  **Goal A's A2 (mixed backing: one VA range, the KV's leading cells in pinned system RAM) is BUILT AND MEASURED** on fork branch `wip-kv-residency`: the 27B Q8_0 `-c 262144` OOM config runs (8160 MiB RAM + 544 MiB VRAM), the spill cost matches `(used − resident)/28.8 GB/s` to ~3 % (18.7 / 13.6 / 10.8 t/s at 51 475 tokens), and at 35B-A3B **Q8_0** the arena gains 2.62 GiB with host-expert residency 63.3 % → 77.0 %.  **The SLIDING window is also in** (the newest W *used* cells stay resident): 27B Q8_0 `-c 262144` runs at 18.6 t/s at 51 475 tokens = the fully-resident reference, and at `-n 2000` the arena gain is +7.9 % / +10.1 % decode on 122B-A10B (residency 29.3 → 34.0 %, and 15.6 → 21.5 % at `-ncmoe 40`) with prefill +8.6 % / +13.1 % (Flash-Next).  **Remaining before it is a product:** ~~`--fit` sizing + reporting~~, ~~both state gates~~, ~~the `MemAvailable` gate~~ are DONE (A3).  **A4 (2026-10-30) CLOSED THE PURITY BLOCKER**: a spilled KV is now numerically transparent — the fault was the VMM primitive, not the KV (a `cuMemUnmap` of a device mapping makes the following `cuMemCreate`(HOST)/`cuMemMap` ZERO a live host-location mapping in the same reservation), fixed by a hold taken in `vmm_map` (`LLAMA_KV_REGION_QUARANTINE`, default on, `=0` = the A/B arm that reproduces the bug).  Measured on a SPILLING arm: prefill-logit KLD −0.000000 / 100.000 % (block was 0.004973 / 98.761 %), 27B and 4B text back to the plain hash, and the spill's cost is a straight line — ~61 GB/s effective, ≈3.6 ms/token per 256 MiB (no cliff; the window apparatus is free).  **GOAL A IS PROMOTED: block 15 in r178 (`v16-6050b467c-r178`).**  The window is **opt-in** (`LLAMA_KV_RESIDENT_CELLS=<cells>|auto`); with the knob unset the KV takes the plain device buffer and the run is byte-identical to a build without it, including its OOM.  Promoted as-is except for two shape changes: the region buffer is taken **only** when a window is active (the campaign's O1 fallback — fully-resident, lazily committed — is not shipped, because on a slab-armed MoE config it `GGML_ABORT`s where the plain arm completes), and the provider is on by default with `LLAMA_KV_REGION=0` as the kill switch.  A1 (the superseded two-base FA read) and Goal C (the drafter ring) are **not** in the block.  Records: `patches/0000`…`0015` block 15 + `wip/kv-residency/A2-MIXED-BACKING.md` §3.7.11.  **A5 (2026-10-30) did the §7.2 list and found two more bugs, both fixed:** the state gates and the MTP verify widths pass on arms that provably spill; the Flash-Next/QSA spilling arm is pure (text byte-identical, prefill-logit −0.000000 / 99.994 %, `FLASH_ATTN_QSA` 26/26) after `0013` (a windowed region's tensors were not placed on the driver's map granularity, so the second tensor's first slide was refused and the process aborted) and `0014` (an idle-slot park on a region-backed KV faulted the GPU, so such a cache now declines parking and the server uses its save-and-clear fallback, byte-identical to the parking gate's oracle).  The arena half is measured at the 55.7 GB/s link: **+2.2 to +2.9 GiB of arena (+3.9 to +6.7 points of residency)** whichever way the window is sized, free when `W` is at or above the run's depth (122B-A10B +3.8 % decode / +4.7 % prefill), 11-26 % lost when it is 35 k cells below it; `--fit`'s `auto` ordering (weights → arena reserve → `W` from the residue, capped at `-c`) is the settled policy and `W` stays a cell count.  **r181 (2026-10-30) fixed a crash the single-GPU box could not see**: default `--fit` aborted on EVERY
  multi-device `-sm tensor` run (the block-15 KV probes look up the backend reg of the Meta wrapper, which has
  none), and a decided `auto` "no window" was reported as "--fit did not run".  The window is now declined
  on a tensor split with a message (it is one VMM region per device) and `auto`'s decision reaches the cache
  as a decision; three `auto` wiring fixes came with it.  **Still open in the campaign: Goal B** (a sparse attended set streamed cheaply — plan + B0/B1 measured, 2026-10-30; B2 implemented, B-A designed (spec in wip/kv-residency/BA-POOL-DESIGN.md)) **and Goal C** (the drafter ring, implemented and measured but not promoted; three questions for the maintainer in `O3-DRAFTER-RING.md` §7), the parking gate's save-and-clear fallback is now graded (A6; the fix is verified and **awaits the maintainer's commit approval** — it used to count the region refusal warning as a park), and a window across a `-sm tensor` split is **designed, not built** (`wip/kv-residency/A6-TENSOR-SPLIT-WINDOW.md`: five changes plus four `common/fit.cpp` findings an implementation must honour; the arms need a 2/3-device host).  Records: `wip/kv-residency/A2-MIXED-BACKING.md`, cold start `wip/kv-residency/HANDOVER.md`.
| **116** | native FP8 E4M3 for RDNA4: a `GGML_TYPE_F8_E4M3` weight type plus an aiter-style `mul_mat_fp8_wmma` GEMM (campaign `wip/fp8-support/`) | performance / new type | **PARKED** (2026-09-26).  Re-based onto the delivery as 13 commits (`~/llama-fp8`, branch `fp8-rebase`), builds clean on gfx1201, fully RDNA4-gated — but **gate 2 fails**: on a 4B the fp8 arm is **0.69x Q8_0** where the pre-rebase `cllm` branch had **1.16x**, because the delivery's MMB/GEMM work lifted Q8_0 **+32 %** (8146 vs 6177 t/s pp512) and the fp8 WMMA path did not follow (5607 vs 7184-7260).  Kernel trace: `mul_mat_fp8_wmma` 71.9 %, **`fp8_repack_weights` 7.3 % (running per call instead of once — the likely bug)**, staging 3.3 %, FA 1.3 %.  Not delivery work: nothing here is in `patches/`.  Resume: A/B the GEMM against `~/cllm`, fix the repack cache, re-run the 4B gate, only then convert and measure the 27B.  Detail in §116 |

| **190** | the r187 quantized-KV decode fusions (`GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`, `GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT`) key on the K/V cache `SET_ROWS` being **`GGML_TYPE_Q8_0`**, because that is the quantizer the fused kernels reproduce; `-ctk/-ctv q4_0`, `q5_0`, `iq4_nl` and `bf16` keep their separate launches | parked idea (scope) | **OPEN — not started.**  The win is `q8_0`-only by construction; each further type needs its own fused write kernel (`k_set_rows_<type>_group`'s lane->element map and rounding reproduced in the K kernel / the V rotation) plus the matching `is_row_write` predicate.  No measurement yet; the `q8_0` arm is worth +13.4 % tg128 / +10.1 % @4096 on Qwen3-30B-A3B (gfx1201) and it is that type's own quantize traffic that pays.  Reach: any GGUF whose KV is quantized to `q4_0`-class (`-ctk q4_0 -ctv q4_0` is the common small-memory pairing on 16 GiB cards), so it is worth doing before the next re-base, not after.  Record: `WORKLOG.md` 261031.005, `benchmarks/2026-10-31-kv-rotation-decode.md` |

Accepted limitations that are deliberately **not** fixed are listed at the end — do not re-report them.

---

## 116. Native FP8 E4M3 for RDNA4 — PARKED behind a GEMM regression (campaign `wip/fp8-support/`)

**Status: parked (2026-09-26).  Not scheduled, not in `patches/`, and nothing may be applied from it without an
explicit go-ahead (`AGENTS.md`, WIP rules).  The campaign is `wip/fp8-support/` (`README.md`, `PLAN.md`,
`HANDOVER.md`, `RESULTS-2026-09-26-rebase-gate.md`).**

**Why it exists.**  Single-request prefill is **70-81 % `mul_mat_q`** (sibling campaign
`archive/work/prefill-gap-attribution/`), so the vLLM gap is a GEMM problem, and the int8 MMQ is
**epilogue-bound**: every per-16 quantization pays a scaled dequant epilogue in the MMA path (~57 TFLOPS
against Q8_0's 83 and F16's 107 — `archive/work/per16-f16-mma/`).  FP8 tensor cores need no int8 dequant epilogue, gfx1201's fp8 WMMA peak is
~171-370 TFLOP/s-class against the int8 MMQ's ~41.5 T-MAC/s, llama.cpp has **no FP8 weight type at all** (so
today's "75 % of vLLM" comparison is engine *and* precision), and the port already measured **+13-17 % prefill
over Q8_0** on a 4B / gfx1201 on the maintainer's `cllm` branch (last touched 2026-08-06, base `6ea215d17`).

**What exists.**  `patches/` = the original 23-commit `cllm` series (provenance; `~/cllm` tip `7c17faffc` is
one unpushed commit ahead of `origin/cllm` and is the source of truth), `patches-rebased/` = the delivered
re-base: **13 commits, 37 files, +5374/-12**, on the r9 base, with the 10 superseded commits (chunked GDN,
ssm-conv, perf docs) dropped because block 02 ships a newer and different GDN.  The port is
`GGML_TYPE_F8_E4M3` (43, self-contained `[f32 scale][128 fp8]` blocks), `ggml-cuda/fp8.cu` (`mul_mat_fp8_wmma`:
CTA 128x64, 8 warps, `GROUP_M=4` pid swizzle, 16x16x128 tiles, register-staged, repacked weights; plus
`quantize_fp8` activation staging and a `dot4` GEMV for decode), direct HF safetensors loading, a
`convert_hf_to_gguf.py --outtype fp8_e4m3` path, a CPU quant + `GET_ROWS` (bit-exact vs CPU), and RDNA4 gating
(compile-time `RDNA4` + runtime `CC_IS_RDNA4`, scalar fallback otherwise).

**Where it stopped (gate 2 fails).**  4B, `pp512`, gfx1201, same box: Q8_0 **8146 ± 221** (the `cllm`-era
figure was 6177, i.e. the delivery's MMB/GEMM work is +32 %) vs `F8_E4M3` **5607 ± 15** (the `cllm` figure was
7184-7260) — **0.69x** where it used to be **1.16x**.  The machine is not the problem, the fp8 path is: the
kernel runs ~65 TFLOP/s here against the ~77 the old notes recorded **on the same source**, and
`fp8_repack_weights` takes **7.3 %** because it appears to run **per call** instead of once (it is meant to be
a lazy one-time layout change cached on the context — check `fp8_backend_cuda_context::fp8_repack_buf`
invalidation on the newer base).  GDN/ssm is 3 % and attention 1.3 %, so this is GEMM + repack + staging, not
the dropped commits.

**Next actions** (`PLAN.md`, in order): (1) interleaved A/B of `mul_mat_fp8_wmma` on the re-based tree against
`~/cllm` at the same `pp512` to isolate codegen vs launch configuration; (2) fix/confirm the repack caching
(target its 7.3 % to ~0); (3) re-run the 4B gate (fp8 must beat Q8_0 again); (4) only then convert
`/llm/models/Qwen3.8/27B/FP8/` and measure `pp8192` against the int8 baselines (Q8_0 1371 / Q6_K 975 /
Q4_K_XL 1254 t/s), which is the number that decides the campaign; (5) lift AITER's gfx1201
`A8W8_BLOCKSCALE` tiles / `GROUP_SIZE_M` / `kpack` (target `mul_mat_fp8_wmma` ≥ 110 TFLOP/s effective; AITER's
Triton reaches 121-137 against our 77-98).

**Risks / constraints.**  The **conversion is the correctness cliff**: HF fp8 (per-tensor/per-channel weight
scales, dynamic per-token activation) → the 128-block `block_f8_e4m3` grid, and a wrong scale mapping produces
plausible-but-wrong text — so the oracle is PPL against the vLLM fp8 record (fp8 6.2250 vs Q8_0 6.2464) plus the
same-seed greedy gate, never "it runs" (there is **no CPU oracle** for an fp8 `MUL_MAT`).  FP8 is a different
numerics path, so it can never be bit-identical to int8 and the purity contract is a PPL/oracle comparison.  A
**non-RDNA4 policy** (reject vs dequantize rather than a silent scalar fallback) has to be settled before any
promotion.  "Dequantize to bf16 + hipBLAS" is a **trap** — upstream already measured that on RDNA4 MMQ beats
dequant+hipBLAS (`ggml-cuda/mmq.cu:608`, PR #18537) — so the win must come from the native WMMA GEMM.  And the
4B result may not scale (the 27B has different shapes, a bigger lm_head and more attention).  Promotion shape,
if it ever gets there: a new weight type + kernels + loader + converter is a **large block** and a maintainer
call.

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)

* **`cpy_2d_vec16` covers F32 only** (r189, `TODO` **#218**).  The vectorised fallback for the 2-D block copies
  r183's guard refuses is restricted to two **F32** tensors with 16-B-aligned width and pitches — the reachable
  case (qwen4exp's indexer key cache, which is F32 by construction).  Every other type and every unaligned
  geometry keeps `cpy_scalar`, which is bit-identical but element-per-thread.  Nothing in the delivery reaches a
  padded block copy of another type today (the census in `benchmarks/2026-10-31-cpy-2d-vec16.md` is F32-only), so
  this is a "extend it if a BF16/F16 copy shows up" note rather than work: the same 16-B unit is four whole floats
  but only **eight** halves, so a wider version needs its own alignment rule and its own oracle.  Re-check it when
  the KV campaign's sparse/pool arm is promoted (that arm is where the copy is made per *decode* step).


* **A verify step whose slot cannot draft at all still splits into two ubatches** (TODO #111, investigated and
  declined 2026-10-28).  A slot with `get_n_draft_max() <= 0` -- its last token of the response, or a full
  context waiting for a shift -- presents its sampled row alone while its neighbour presents `n_draft + 1`, so
  `split_equal()` puts the neighbour's draft rows in a second ubatch: `0+1` split decode steps per pair run on
  the 35B-A3B `-ncmoe 40` arm (`0+0` on the dense 27B arms, where the two responses do not overlap at a
  boundary).  Both alternatives were measured, one of them implemented and reverted: suppressing that step's
  drafts measures **-0.3 %** (below the floor, because the second ubatch carries the neighbour's *genuine*
  draft rows, so both arms are two passes), and cutting the neighbour's draft is the r153 cut, a **-6 %** loss.
  There is no correctness at stake (a verify width is a numerics axis, `GREEDY-PURITY.md` §63) and the candidate
  would add a cross-slot coupling, so nothing changed -- what did is the instrument:
  `scripts/gate-spec-align.sh` now counts these single-row splits separately and bounds them (`TAIL_MAX`,
  default 2) where it previously excluded them, so a change that *manufactures* them is still caught.
  Record: `benchmarks/2026-10-28-spec-align-111.md`.

* **Parking a memory whose second cache mirrors the attention cells is validated on qwen4exp only** (TODO
  #108, r146): `llama_memory_hybrid_idx` also serves glm5-next, whose indexer is the k-pool
  (`indexer_kpool > 0`) variant.  The park code, its mirror check and its invalidation are arch-agnostic
  and the k-pool layout is invalidated with everything else, but no glm5-next GGUF exists on this fleet, so
  only the qwen4exp arm is measured.  `llama_kv_cache_dsa` / `_iswa` (deepseek4, glm-dsa) report parking
  unsupported by design and keep #210 — they never half-park.

* **A dense `-sm tensor` split is not guaranteed to produce the same greedy text across device counts**
  (AGENTS "Greedy text is a per-device-count reference" was corrected 2026-10-24).  Qwen3.8-27B
  UD-Q4_K_XL, prose prompt, seed 42: 3 GPU 850 chars vs 2 GPU 876; the r126 build shows the same
  (897 vs 855), so it is the tensor split's rounding class, not a delivery defect.  Every `-sm layer`
  arm *is* text-identical across counts.  Gate a change within a device count.


- **A Windows VRAM overcommit is not catchable: it surfaces as a device loss, not an allocation failure
  (issue #216, accepted; nothing in the delivery can change it).**  On WDDM an allocation set larger than
  the card does not fail, it pages into system memory; the first big prefill (a full `-ub` compute buffer
  plus the FA staging) then loses the device, and the process's *next* rocBLAS touch aborts with
  `rocBLAS error: Could not initialize Tensile host: No devices found` (rocBLAS's `TensileHost` prints
  that when `hipGetDeviceCount()` returns non-success; it allocates nothing on that path, so the message
  is a symptom, not an OOM inside rocBLAS).  Measured budget for the reported config (dense 27B IQ3_S,
  `-c 131072 -ctk q4_0 -ctv q4_0 -b 4096 -ub 1024` + MTP n3, no mmproj): **14.8 GiB** of static device
  memory and a **16.0-16.2 GiB** first-prefill peak on a headless 32 GiB card, against a 16 GiB card that
  also drives the display.  Mitigations shipped/to use: the draft KV follows the main KV type (r125,
  block 02, -368 MiB at `-c 131072`), `-c`, `-ub`, a real `-ngl` offload (`-ngl` is a **layer count** and
  a NextN model has `n_layer + 1` layers, so `-ngl 85` on a 65-layer model is a no-op), and letting
  `--fit` size the context (it declines when `-ngl` or `-c` is set).  Do not re-investigate the rocBLAS
  message as a delivery defect without a Windows event log showing a TDR (`4101`/`4102`) *absent* at the
  abort.

- **`std::min<int64_t>` / `std::max<int64_t>` in other HIP TUs is a latent Windows-only build failure
  (TODO #95, accepted; no Windows toolchain here).**  With HIP on Windows (clang + the MSVC STL) clang's
  force-included `cuda_wrappers/algorithm` and the MSVC STL `<utility>` each define an inline
  `std::min<long long>` under **one** mangled name (the `enable_if` that makes clang's win overload
  resolution is not part of the MSVC mangling), so a translation unit that instantiates both fails to
  codegen with *"definition with same mangled name"* — r123's `moe-expert-cache.cu` was the first to
  trigger it (issue #217, fixed in r124 with the local ternary helpers; see `patches/README.md` block 09).
  The reporter's toolchain builds every other TU, but ~10 of them still carry the int64_t pattern
  (`argmax.cu`, `cpy-batch.cu`, `fattn.cu`, `mmvf.cu`, `fattn-common.cuh`, `gated_delta_net.cu`, `top-k.cu`,
  `ssm-scan.cu`, `ggml-cuda.cu`, `hc-mix.cu`): the collision only fires where the STL's own instantiation is
  emitted too, which is a header-instantiation accident, not a property of the call.  Not proactively
  rewritten because a non-AMD/Windows claim cannot be validated on this box (scope policy: non-AMD backends
  are out of validation scope); if a Windows build log names one of them, apply the same two-line helper
  fix there and bump the release.  Do not re-report as a new defect without a Windows build log.

- **The Meta split-state lookup-count headroom (TODO #80) — retired 2026-10-24, accepted.**  The measured
  defect (issue #184: the pre-warm walk allocating one heap node per visited tensor on every graph
  re-allocation) was fixed in r100 (block 09, `GGML_META_SS_SCRATCH`, default on).  What remained is the
  reporter's ~165k cache hits per re-allocation (30+ per node), and the only lever is the lookup **count**
  (the recursion/pruning structure of `calculate_split_state` and the pre-warm walk) — hashing the cached
  336-byte `ggml_tensor` copy to skip the `memcmp` costs about what the `memcmp` costs.  Not pursued because
  the frequency that makes it worth anything is his workload: it re-allocates **~230** graphs per run, while
  the delivery's own validated shapes re-allocate **3** in a whole 4-stream x 192-token Flash-Next IQ3_XXS
  + MTP window (`-sm tensor -ncmoe 47`, 63 graph reuses per stream; 4B Q8_0 2-GPU tensor: 12 in 4 x 128),
  so even at his own post-r100 16.0 ms per re-allocation the whole walk is at most ~1 % of a 4.2 s window
  and the lookup-count share of that is below this box's noise floor (r100's entire heap-churn removal
  measured +0.29 % on this shape).  Issue #184 is closed (2026-10-21) with the reporter confirming r100 in
  his production build (+1.6 % 4-stream decode, identical greedy output), and the walk's code is unchanged
  since r100.  A future change there still owes the r100 proof set (identical states under
  `GGML_META_SS_VERIFY=1`, identical greedy text and prefill KLD, same-binary A/B with
  `GGML_META_SS_SCRATCH=0`).  Records: `archive/work/issue-184-meta-r80r82/RESOLUTION.md` (addendum) +
  `probe-reallocs.py`, `WORKLOG.md` 261024.003.
- **`mmf32` covers the skinny F32 shapes only (r104, `TODO` #82 closed) — accepted.**  A full-F32 model's FFN
  GEMMs (`m` ≫ 256: dense F32/F16-weights-heavy prefill) still fall through to rocBLAS/hipBLASLt, so they keep
  that path's per-start solution lottery; the kernel's tile (32 output rows) is the wrong shape to be
  competitive there and no such model is validated.  Also untouched: BF16/f16 prefill (Flash-Next keeps both
  hipBLASLt's +51 % and its own lottery — a BF16 counterpart is its own job) and **RDNA3** (gfx1100), where the
  kernel stays opt-in (`GGML_CUDA_MMF32=1`) because it measured −1.3…−2.2 % at `pp2048` there; the lever would be
  a per-arch tile, not the gate.  RDNA3_5 was admitted and is default-on since r105 (`halo`: 12/12
  deterministic, +1.0 %/+1.6 % prefill, family-switch PASS).  Scope: `patches/README.md` (block 06), rule:
  `GREEDY-PURITY.md` §47.
- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.
  Upstream ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the
  block-13 rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480 (4B,
  q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute + −799 MiB
  host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.
- **Mixed K/V cache types are HARD-REJECTED.**  Every mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`) measured
  1.7-3.6× slower and never smaller; `params.type_k != params.type_v` fails `llama_init_from_model` with
  a message naming both types (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `-ctk`/`-ctv` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter because its 2 KV heads are fewer
  than the 3 devices (one device gets a zero-extent share: `[1, 1, 0]` over 2 heads).  The abort is the
  split-state **ratio-consistency** assert in `ggml_backend_meta_get_split_state`
  (`split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]`
  — `ggml-backend-meta.cpp`; 1582 at r101, 1487 at r99: the line number moves, the assert does not, and
  older records that cite `:1177` are one more tree back).  Works on 1/2 GPUs and on 3 GPUs with
  `-sm layer`; maintainer's call: no fix.  r101's split-state work deliberately does not change this
  (`GREEDY-PURITY.md` §45 accepts a slice that is *empty or the whole* extent, and one head of two is
  neither); re-verified on r99 and r101, identical abort.  Every other model is unaffected (a future
  block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Gemma4's full-attention (head_dim 512) layers stay outside the rope + FWHT + KV-write fusion**
  (documented 2026-10-21, deliberate).  The fused kernel instantiates `fwht_n` in `{64, 128, 256}`, and
  Gemma4 is the only arch with a distinct SWA/full head dim (`key_length_swa` 256 vs `key_length` 512),
  so its 1/6 full-attention layers run `rms_norm_mul_rope_f32<256,true,float,0,false>` + `fwht_cuda<512>`
  for Q and K plus two `set_rows` — 6 launches instead of 2.  Extending it needs 4 new kernel
  instantiations **and** a fix to the fused kernel's V-row write (its `ncols <= block_size` precondition
  would quantize half a 512-wide V row), which edits the shipping 256 fold, for ~0.1 % on gfx1201.
  Accepted as a gap; the analysis (what to change, why the arithmetic is bit-identical by construction)
  is in [`archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md`](archive/work/fix-gemma-4/ISWA-WRITE-FUSION.md) §5.6.
  (The iSWA builder order that made the 256-wide half of this fusion live is r101, `WORKLOG.md` 261021.003;
  `GREEDY-PURITY.md` §45 records the split-state rule that came with it.)
- **3-GPU plain (non-speculative) decode is a few percent slower than 2-GPU — accepted (this closes
  `TODO #51`; 2026-10-23, `archive/work/three-gpu-performance/RESOLUTION.md`).**  −2.0 % with a fully-resident
  arena (the 3-device internal all-reduce is 1.2 % of decode and no backend/size knob beats it), −10.5 % at
  91 % residency (the non-identity per-token routing readback, paid per device; the *marginal*-arena case is
  fixed in r117, an 8 %-short one is not), −10.2 % with
  `-sm layer` (no intra-token parallelism; a property of layer splitting, reproduced on dense 27B).
  Speculative decode does **not** lose: +14…+23 % at 3 devices when resident.  Production rule: 3 GPUs for
  speculative/MTP workloads, 2 GPUs for plain decode at partial residency, never `-sm layer` for decode.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).
- **`--spec-draft-mtp-vocab` two-band subset (closed 2026-10-22, measured, not shipped).**  A second band
  (keep `< N` **and** `>= n_vocab - M`) was meant to catch gemma4's space-piece band (229376..255999): it
  measured 0.610 acceptance at 37 % of the rows and was still 0.4 % *slower* on that payload.  With the
  width rule found in the 3-GPU arm (`N · n_devices < n_vocab`) a second band only makes the **mirrored**
  draft matmul wider, so it is a non-goal, not a parked idea.  Record:
  `archive/work/spec-draft-mtp-vocab/VALIDATION-2026-10-22.md` §5.

## Where the current lists live


- **Open campaigns and scoping notes:** [`wip/CAMPAIGNS.md`](wip/CAMPAIGNS.md) (the index for `wip/`).
- **Closed campaigns, with their evidence:** `archive/work/<campaign>/` (`RESOLUTION.md` where the campaign
  closed with a finding).
- **Remaining gfx1151 work + the 2026-09-12 audit:** `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- **QSA sparse-regime width purity (items 4/7 disposition):** `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- **Environment, instruments, reference hashes and the landing procedure for KV/FA work:**
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its
  items 1/5 and F3 are **superseded** — see `WORKLOG.md`).
- **Delivery verification contract + dated records:** `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- **Purity/invariant analysis and the instrument rules:** `GREEDY-PURITY.md`.
- **Benchmarks + gates:** `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- **Upstream PR candidates:** `upstream/README.md`.

---

## 118. Windowed / streamed KV cache — scope decision 2026-10-29 (campaign `wip/kv-residency/`)

**Number note (2026-10-29).**  The campaign's records cited "TODO #117" for three sessions, but `TODO.md`
never had a `#117` row — the reference was dangling.  Item **117** was then allocated to the separate MoE
band-predicate finding in a concurrent session and **closed in r176**, so the campaign takes the next free
handle, **#118**.
Old `wip/kv-residency/` text (and the patches' commit messages) still say #117; they mean this item.

**The ask was, and is, a Strata-style windowed KV cache: the KV's VRAM footprint is a *fixed window*
(host master + residency page table + in-graph resolve), so a large `-c` no longer costs VRAM that the
expert arena could use.**  That is the campaign's **O2** (`wip/kv-residency/README.md`, cost XL).

**O1 (`--kv-grow`: reserve the KV's VA, map physical lazily) is PARKED.**  It was proposed by the seeding
session as "the cheap win, do it first" (`README.md` line 43) and slice 2 (the KV region buffer type) was
built on that basis.  It is **not a step toward O2** — O2's provider is a *fully mapped fixed pool* plus a
host master and a page table, which lazy growth does not help — and it was measured to regress a config that
uses most of `-c` on a slab-armed MoE model (35B-A3B `-ncmoe 24 -c 262144` at 253k tokens: region off
completes, region on refuses at 41k cells because `SLAB_INIT` had already given the freed VRAM to the arena).
Detail and numbers: `wip/kv-residency/IMPLEMENTATION.md` ("Slice 2" / "KNOWN LIMIT").

**Why it is parked rather than refuted — the maintainer's framing (2026-10-29).**  O1 is a *temporal*
version of the window: the arena is bigger early and shrinks as the context grows, and once the context is
large it converges on fully-resident-KV speed.  Until the window fills it is **no faster than the windowed
design at all**.  The open question is only the crossover: past the window, what costs more — windowing
(misses through the link) or the shrinking arena (expert-cache misses and the identity cliff)?  There is
probably a brief window just after it fills where growing is a little faster, and then the arena's miss rate
steals it back.  **We do not know, and O2 has to be built to find out** — so O1 is waiting on a measurement
that only O2 can provide, not dismissed.

**CORRECTION 2026-10-29 (immediately after the above was written).**  The "O3 first" order was chosen from
this file's own option menu and produced a third session on a non-goal.  **The campaign is O2, and only O2**
(`wip/kv-residency/O2-PLAN.md`).  O1 and O3 are **closed detours**: both were built and measured, neither was
ever requested, and neither is a step toward the windowed cache.  **Do not restart either.**  Root cause, worth
remembering: O2's payoff shape (a small window beside a big expert set) needs Flash-Next plus 2-3 GPUs, and only
one 32 GiB card is attached — so each session quietly substituted something that *was* measurable.  The rule
from here: if the ask cannot be measured on the attached hardware, say so and build it for correctness with no
perf claim, or stop.

**The historical order line, kept for the record.**  **O3 (drafter ring) first, then O2.**
* **O3** is genuinely windowed residency (`--mtp-window` analogue: the draft KV is `O(window)`, not
  `O(-c)`), it is family-independent, it runs on the 27B dense that fits one card, and its cost is a clean
  acceptance number.  It is the smallest complete instance of the policy O2 needs — a rehearsal, not a
  detour.  Note its *standalone* VRAM win is modest (~256 MiB at `-c 65536` on the 27B: 1 draft layer).
* **O2** then reuses that policy on the K/V page table, on the eligible (bounded attended set) families.
  Blocked for *measurement* while only one 32 GiB card is attached: its target family is qwen4exp/QSA
  (Flash-Next, 75 GB).  Correctness work can proceed on a small model with a forced small window.

**PROGRESS 2026-10-30 (A3; fork `wip-kv-residency` tip `59b1bf1a8`, records in
`wip/kv-residency/A2-MIXED-BACKING.md` §3.7).**  Done: `--fit` **sizes and reports** the window
(`LLAMA_KV_RESIDENT_CELLS=auto`: bytes-per-cell from the region buffer type, `W` from the VRAM left after the
arena/slab reserve, the split + modelled cost printed at the calibrated link rate, the user's `-c` kept rather
than shrunk); the **pinned-RAM ceiling** (`LLAMA_KV_SPILL_RAM_PCT`, clamped at creation and refused at run time
instead of an OOM kill); `gate-ckpt-reuse.sh` + `gate-state-pipe.sh` **pass on a windowed arm** — and the
latter found a real ABORT (the batched state save asserted the plain CUDA buffer type, so a windowed KV killed
the server at the first prompt-cache save); MTP verify widths are byte-identical on a windowed arm (4B).
Three slide bugs were found and fixed (a coalesced mapping straddling the boundary meant the window spilled
**nothing**; `cudaMemcpy` into a host-location mapping silently leaves zeros, so the staging is a kernel copy;
a multi-piece mapping rewrite loses bytes).  The box was re-cabled and re-baselined (**link 28.8 → 55.7 GB/s**).

**The §3.7.6 purity blocker is CLOSED (A4, 2026-10-30)** — the fault was the VMM primitive (a `cuMemUnmap` of a
device mapping made the next `cuMemCreate`(HOST)/`cuMemMap` zero a *live* host-location mapping), fixed by the hold
`vmm_map` now takes (`LLAMA_KV_REGION_QUARANTINE`, default on).  Purity is claimed for **spilling** windowed arms
(prefill-logit −0.000000 / 100.000 %).  The narrative is `A2-MIXED-BACKING.md` §3.7.8.

**What is actually left (2026-10-30, end of A6):**

* **`auto` policy** — the maintainer's call, unchanged by r181: `auto` sizes the window to
  `free − margin − everything-but-the-KV` (the margin includes the arena/slab reserve, so on a `-ncmoe` config the
  arena wins the VRAM by design) and it only spills a pinned `-c`.  Deciding measurements: §3.7.11.4 A-C.  `W`
  stays a cell count.
* **Goal C** — the drafter ring is measured, not promoted (`O3-DRAFTER-RING.md` §7: default-on or not, one size or
  per-model, MTP-only or DFlash2).
* **A `-sm tensor` window** — designed only (`A6-TENSOR-SPLIT-WINDOW.md`); hardware-blocked here.
* **Goal B** — **the pool is built, gated and measured; the policy is dead; the indexer item is fixed.  What is
left is one stale measurement row and the three decisions below.**  Branch `wip-kv-residency-b` `b98cb3e43` (rebased
onto r186; `wip-kv-residency-b-r182` keeps the pre-rebase tree).  `LLAMA_KV_REGION_POOL=<cells>` makes the attention
KV a pinned host MIRROR (authoritative, so `seq_rm`/`seq_cp`/defrag/state/park are unchanged code) plus a compact
device pool of indexer-block pages with a page table and an in-graph resolve (`GGML_OP_KV_POOL`), and a second id
list for the QSA reader's K/V gather.  Acceptance at **5.3k / 103k / 180k** with `-n 512`, MTP depth 4, arms
interleaved: dense **72.09 / 59.10 / 48.73**, sparse 71.58 / 54.70 / 42.93, **pool 71.27 / 54.65 / 43.09** — i.e.
the pool lands ON the sparse ceiling at every depth and the remaining gap to a resident KV is the sparse arm's own
cost (7-12 % at depth), **not** placement.  A rocprofv3 census says the pool's own machinery is 0.10-0.11 ms/token
(with fills 0.03-0.05 ms/step), so the page policy (the design's step 3) is **measured dead**; and the sparse arm's
real cost was two bugs, both now fixed and gated on the campaign branch: A12 (`f8e47e392`, r183's 2-D copy fallback
made the indexer chain's per-step whole-cache F32 copy cost 5.20 ms/token — a **delivery regression**, block 10) and
A13 (`b98cb3e43`, the fused `INDEXER_SCORE`/`INDEXER_FILL` path skipped every quantized indexer key cache — now
q8_0 + the q4/q5 family, byte-identical to the chain on all eight key types, **plain decode 23.68 -> 17.73 ms/token
and 29.3 -> 36.3 t/s** at 103k; block 11).  Records: `wip/kv-residency/BA-POOL-DESIGN.md` §7-§10,
`COLD-START-PROMPT.md` (the next session's list).

**Open for Goal B, in order:** (1) re-measure the **180k** row on the A12+A13 build (the only stale number);
(2) the **delivery placement** of A12 (block 10) and A13 (block 11) — each owes a `WORKLOG` entry, a
`patches/README.md` note and the release gates, and both need a re-base onto the current release (the tree moved to
r188 while they were built); (3) the **promotion decision** for the pool itself (the maintainer's): a block in
`patches/`, `ENVIRONMENT.md` §, the pool's VRAM in `--fit`'s arithmetic, a **grow-on-demand mirror** (today the whole
`-c` is pinned at creation: 0.2 / 1.6 / 2.4 GiB at 16k / 131k / 196k), the release-flow gates, and a default-off
call (it forces the sparse arm, so it is a fittability feature like block 15, not a speed one).  Two accepted
limitations: k-quant indexer keys stay on the per-op chain (a 256-element superblock does not divide `idx_dim` =
128), and the pool declines a `-sm tensor` split (one page table per device would need a per-device pool and a
split-aware resolve) — the latter needs the maintainer's multi-GPU host.

**Nothing from `wip/kv-residency/` is in `patches/` or `release.json`, and nothing may be applied from it
without an explicit go-ahead.**  The slice-1/2 code is the record, not a candidate: sub-slice 2's *growth*
must not be promoted; its KV **buffer-type skeleton** (a custom buft carrying a host master + device pool +
page table, selected per device with a plain-buft fallback) is the container O2 needs and is worth reusing.

---
