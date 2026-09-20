# HANDOVER — general-purpose `mmb` (bf16/i8-WMMA dequant weight GEMM) + QSA/Q8_0 next steps

**Date:** 2026-09-19.  **Status:** ACTIVE WIP, not part of the delivery, **not** pushed to any fork.
This document is the self-contained entry point for the next session.  Read it top to bottom first,
then `README.md` (the running record) beside it.

> **One-line summary.** A general prefill weight-GEMM on the tensor cores (dequant-to-bf16 → WMMA)
> is implemented for **every** weight type the delivery's models use, validated PPL-parity, and
> measured at up to **+68 % pp2048 / +56 % pp8192** on the Flash-Next Q4_K_M.  **QSA v3 (packed-block
> WMMA sparse attention)** landed in sessions 2/3: the QSA attention went **2944 ms -> 728.6 ms
> (4.04x)**, and session 4 made it the **default** at every context length.  The next lever is the
> **indexer**, then a dedicated tiny-M F32 kernel, then the remaining `mmb_*` tuning.

**Current state (start here):**

| | |
|---|---|
| worktree | `~/llama-wip-mmb`, branch `wip-mmb-general`, tip **`3ff571bf9`** |
| base | `8a2567e1e` (the maintainer's applied delivery tree; **not** canonical r9) |
| backup | `wip/mmb-general/mmb-general.patch` + `patches/0001..0013` + `commits.txt`, in this repo, pushed to `origin/main` (`b78b96b`) |
| verify | `git apply --check mmb-general.patch` on a fresh `8a2567e1e` — clean (13 commits, 9 files, +2190/-11) |
| build | §3 | run | §4 |
| current numbers | the session-4 UPDATE below (the §5/§7 tables predate the default flips) |

---

## UPDATE — session 5d (2026-09-19): the W=1..8 probe — LAST GATE ITEM CLOSED

The `W = 1..8` logits matrix was the one §11 item never run, because no probe harness existed.  It
does now: **`tests/test-logits-width-probe.cpp`** (built with `cmake --build build-rocm --target
test-logits-width-probe`), adapted from `archive/work/strix-halo/issue25/logits-width.cpp`.

**Method.** Feed an identical prefix as ONE prefill batch, then decode a W-token batch
`[t_P .. t_{P+W-1}]`.  Row *j* of that batch sees exactly the same context for every W, so its logits
must not depend on W.  The probe runs W = 1..8 in separate contexts and checks (a) every row shared
with the W=8 batch matches it bit for bit, and (b) prints a per-row logits hash so two builds can be
diffed.  Env: `RS=0|from_w|<n>` (`n_rs_seq`), `FA`, `KV`.

**Result — gate PASSES.**

| prefill P | `MMB=0` row0 hash | `MMB=1` row0 hash | width purity (W=1..8) |
|---|---|---|---|
| **256** (below `MMB_MIN_T = 512`, so MMB is unreachable anywhere) | `6228d03bd2b501b4` | `6228d03bd2b501b4` — **identical** | PASS, maxdiff 0 |
| 1024 (MMB fires in prefill) | `ac4d5de3d40a2b1d` | `3703c13f03c4b25d` | PASS, maxdiff 0 |
| 2048 (MMB fires in prefill) | `1996b44e491de5c9` | `e3e4220fe83831da` | PASS, maxdiff 0 |

Read it in two halves, because the literal "MMB=1 == off" can only hold where MMB never runs:

* **Below the threshold** (`P = 256 < 512`) MMB is unreachable in *both* the prefill and the decode
  batch, and the two configs are **bit-identical across every row of every width**.  That is the
  "identical by construction" claim, demonstrated rather than asserted.
* **Above the threshold** the row-0 hashes differ — that is the **approved prefill re-baseline** (MMB
  replaces the MMQ reduction with a dequant-to-bf16 WMMA one), not a band defect.  What matters for
  purity is the second column: **`width_purity = PASS` with MMB on at every P** — no row of any width
  differs from the widest batch by even 1 ulp, so MMB introduces no width dependence.  The `T >= 512`
  gate is what keeps MMB out of the `W <= 8` band in the first place.

**Two probe bugs worth recording** (both cost a cycle and both are the same class as the `-md` trap):

1. `llama_batch_init(ubatch)` sizes the batch for the *micro*-batch; feeding a `P`-token prefill
   overruns it.  Worse, the first symptom was a **silent segmentation fault** (and a `GGML_ASSERT`
   abort for the `n_batch` half) rather than a clean error.
2. `llama_context_params.n_batch` is the max tokens per `llama_decode` call (the whole prefill batch)
   and `n_ubatch` the max per micro-batch — they are different knobs and both must scale with `P`.

Also fixed in the probe: `rows[W-1].data()` hashes the `std::vector` objects, not the floats — the
first run printed a plausible-looking `hash=` field that was hashing pointers.  The `width_purity`
column and the `row0_row1_hashes` line were always correct (they hash `.data()` of the inner vector).

---

## UPDATE — session 5c (2026-09-19): promotion gates actually run

Session 5's profile put the remaining big kernels (`mmb_routed_glu` 22.7 %, `mmb_dense` 21.1 %) out
of reach without a split-K / IU8 restructure, so this session ran the **promotion gates** instead —
the handover had listed all of them as never run.

The test models come from `test-llama-archs -o <dir>` (~200 tiny GGUF architectures), which is the
`test-generate-models` ctest fixture.  `build-rocm/bin` does not contain the test binaries by default.

| gate | result |
|---|---|
| `test-recurrent-state-rollback` qwen35-dense / nemotron_h-dense / deepseek4-moe | **PASS** (multi-seq split replay matched, max diff 0) |
| `test-recurrent-state-depth` (n_rs_seq 1..15 sweep) | **PASS** (`total failures = 0`) |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** |
| `test-backend-ops -o GATED_DELTA_NET` | **46/46** |
| `test-backend-ops -o FLASH_ATTN_EXT` | **OK** (ROCm0) |

All with `GGML_CUDA_MMB=1 GGML_CUDA_QSA3=1`.

**`plain == draft-mtp` greedy text on qwen4exp — PURE (byte-identical), gate PASSES.**

| config | plain | draft-mtp | verdict |
|---|---|---|---|
| `MMB=1 QSA3=1` | `bbd4bcb519e4` | `bbd4bcb519e4` | **identical** (1700 chars) |
| `MMB=0 QSA3=0` (features OFF) | `5120b28f2879` | `5120b28f2879` | **identical** (1720 chars) |

The hashes differ *between* the two configs and that is the approved **prefill re-baseline** (MMB and
QSA3 change prefill numerics by design) — but *within* each config the plain and `draft-mtp` arms are
byte-identical, which is the purity contract.  This also supplies two of the re-baseline hashes that
were previously missing from the record.

### METHODOLOGY TRAP — this result was reported WRONG first, and how

The first attempt concluded "`plain != draft-mtp`, pre-existing cause 3" from `320 chars vs 1942
chars`.  That was **entirely an artifact**:

* the **plain arm must not be given `-md`**.  Passing the draft model makes llama-cli initialise an
  MTP context even with `--spec-type none`, and it dies with `failed to initialize the context: this
  model is an MTP draft head without a trunk` -> `llama_server exited with code 1`, **exit status 1**;
* the failed run's stdout still contained the `Loading model... |\b-\b\\...` **spinner**, and the
  `grep -v '^\[' | tr -d '\b'` filter I used turned that spinner into exactly 320/318 "chars" — so I
  was hashing **a spinner against real text**;
* the 1942-char `draft-mtp` side was genuine; the 320-char side never generated a token.

Two rules out of it:

1. **Never pass `-md` to the plain arm** of a spec purity comparison.  Run the target model alone.
2. **Assert the arm actually generated something** before comparing — check `$?` and that the output
   is neither empty nor the spinner.  A comparison where one side is a load-failure artifact will
   always "diverge", and it looks exactly like a real near-tie flip.

### Other traps

1. The first purity matrix printed `LLAMA_QSA_OFF=1 ... PURE` where **both hashes were md5 of the
   empty string** — that run had failed to start.  A "pure" verdict where the two sides are empty is
   not a pass; always check the output is non-empty before comparing.  (`LLAMA_QSA_OFF=1` itself is
   genuinely unusable with `-md` here — `llama_server exited with code 1`, deterministic.)
2. `FLASH_ATTN_QSA` is now **22/22**, not the 18/18 in the older notes (cases were added);
   `GATED_DELTA_NET` is 46/46 as expected.

**Still not run:** the `W = 1..8` logits matrix with MMB on == off.  It needs a probe harness that does
not exist in the tree (there is no width-probe tool under `tests/` or `tools/`), and it is guaranteed
by construction anyway (`T >= 512` keeps MMB out of the whole `W <= 8` band).  Building that probe is
the remaining gate item.

---

## UPDATE — session 5b (2026-09-19): tiny-M F32 kernel — the hc `*_inject` GEMMs (+2.3-3.3 %)

Session-5 next-work #1 (the remaining F32 tiny-M) is **done**.

`hc_attn_inject` / `hc_ffn_inject` are M=4 (hc), K=10240, T=2048 — 1012 ms, 1.330 ms/launch, 5.7 % of
prefill.  Neither tile can serve them (M=4 gives no M parallelism: rocBLAS launches 64 blocks, the
128-row MMB tile pads 32x), and both read the 84 MB activation once yet sit at **63 GB/s** while
`rms_norm_f32` sustains **330 GB/s** — a *parallelism* wall, not bandwidth.  The two injects also
cannot be fused (different `xn`).

New `mmb_tiny_m_f32_kernel`: **one warp per token**, every lane accumulating a k-strided partial for
all M rows, so X is read once, coalesced, and reused across M in registers.  256 blocks, not 16-64.

| | per launch | GB/s |
|---|---:|---:|
| rocBLAS | 1.330 ms | 63 |
| MMB f32split | 1.825 ms | 46 |
| **tiny-M** | **0.500 ms** | **168** |

pp2048 933.0 → **964.1 (+3.3 %)**, pp4096 934.4 → **962.0 (+2.9 %)**, pp8192 934.6 → **955.9 (+2.3 %)**.
PPL c16384 bf16 3.3875 → 3.3851 (noise); decode identical (tg64 25.96; M=4 is a weight row count, so
the `T >= 512` gate still keeps the decode/verify band off this path by construction).

**Tuning (both negative, both recorded):** TT=1 beats TT=2/TT=4 (the W-traffic amortisation idea is
rejected — L2 serves the 164 KB W panels well enough); specialising `<8,TT>` → `<4,TT>` is +0.5 %.

**Trap:** the first version changed only the *launcher* and measured flat, because the new `M >= 128`
gate rejected the shape *before* the launcher ran — the kernel never executed.  The kernel trace
(symbol absent) caught it; a bench delta alone would have read as "the idea failed".

**Next-work order (revised):**

1. **`mmb_dense` / `mmb_routed_glu`** (52 % combined) — §9's Q8_0 IU8-WMMA and the routed-GLU geometry.
2. **`ssm_alpha/beta`** (207 ms, 0.359 vs a 0.082 floor, rocBLAS) — same ~60 GB/s parallel-wall shape
   as hc_inject had; the tiny-M kernel generalised to M=48 is the obvious next step.
3. **`dsv4_hc_pre` + `_post`** (8 %) — the HC prefill fusion (pwilkin's pair is ~0.93 s vs our 1.44 s).
4. **qsa3 attn + its PACK** (4.9 % + 3.0 %) — the pack is a pure copy; fusing it reclaims most of 3 %.
5. **The indexer** — 1 % at 8K, 3.2 % at 32K, growing.

---

## UPDATE — session 5 (2026-09-19): fresh profile reprioritises; F32 split by shape (+2-3 %)

**1. A fresh post-session-4 profile (MMB+QSA3 on, `-b/-ub 2048`) contradicts the session-4 ordering.**
% of total kernel time:

| family | pp8192 | pp32768 |
|---|---:|---:|
| `mmb_*` | 52.0 % | 48.2 % |
| **F32 rocBLAS** | **11.7 %** | **12.0 %** |
| `qsa3` attn/rows/merge | 4.9 % | 6.4 % |
| `rms_norm_f32` | 5.9 % | 5.5 % |
| `dsv4_hc_pre` + `_post` | 8.0 % | 7.4 % |
| PACK/copy (qsa3 pack) | 3.0 % | 3.5 % |
| indexer top-k family | 0.99 % | 3.2 % |

The **indexer is ~1 % at 8K**, not the #1 item - it was promoted on the pp2048 startup regime (0.5 %).
It does scale with `n_kv x n_tps` (3.2 % at 32K, output capped at 2051 cells) so it returns at depth,
but the **F32 path is 12 % at both** and was the bigger, better-scoped target.  Indexer deferred.

**2. F32 dense weights are now split by shape and default ON** (`GGML_CUDA_MMB_F32SPLIT=1`).
All-or-nothing was a wash because the paths disagree per shape:

| shape | rocBLAS | MMB f32split | winner |
|---|---:|---:|---|
| `ffn_gate_inp` M=512 K=2560 | 2.044 ms | **0.846 ms** | **MMB 2.4x** |
| `hc_attn/ffn_inject` M=4 K=10240 | **1.332 ms** | 1.825 ms | rocBLAS |
| `ssm_alpha/beta` M=48 K=2560 | **0.359 ms** | slower | rocBLAS |

Rule: MMB iff `M >= 128`.  pp2048 914.9 -> **933.9 (+2.1 %)**, pp4096 908.9 -> **933.0 (+2.6 %)**,
pp8192 902.0 -> **929.8 (+3.1 %)**; the old force-all mode 2 is worst at every length.  F32 GEMM
1947 -> ~1537 ms.  PPL c16384 bf16 3.3821 vs 3.3875 (bars ±0.027) = parity; decode untouched
(tg64 25.95 vs 26.00; the `T >= 512` gate is unchanged so `W=1..8` purity holds by construction).

**Trap, recorded so it is not relearned:** the first rule also took MMB when `K >= 4096` (expecting
the WMMA path to help the K=10240 hc inject pair).  It is wrong and costs 375 ms.  It looked right
only because bucketing launches by grid alone merged `hc_inject` + `ssm` into one 1.065 ms average.
**Split the bucket before believing a per-shape number.**

**Next-work order (revised, by measured cost):**

1. **F32 tiny-M**: the remaining ~1.5 s is `hc_inject` (M=4, 1.332 ms/launch, floor ~0.33) and
   `ssm` (M=48, 0.359 vs ~0.084 floor) - both rocBLAS-bound now.  A dedicated small-M kernel (or
   getting rocBLAS off its split-K) is the next ~8 % of prefill.
2. **`mmb_dense` / `mmb_routed_glu`** (52 % combined) - §9's Q8_0 IU8-WMMA and the routed-GLU
   geometry.
3. **`dsv4_hc_pre`+`_post`** (8 %) - the HC prefill fusion.
4. **qsa3 attn + its PACK** (4.9 % + 3.0 %) - the pack is a pure copy; fusing it into the kernel
   would reclaim most of its 3 %.
5. **The indexer** - 1 % at 8K, 3.2 % at 32K, growing; worth it once the above are done.

---

## UPDATE — session 4 (2026-09-19): two default flips + the F32 finding

**1. `LLAMA_QSA_DENSE_SHORTCUT` is now default OFF (always QSA)** — maintainer decision.  The
shortcut's rationale (below the selection width the top-k covers every cell, so sparse saves no
attention work) was **overtaken by qsa3**: in the fully-dense pp2048 regime `qsa3_attn_kernel` is
**137.8 ms vs 149.9 ms** for dense `flash_attn_ext_f16`, so qsa3 is already 8 % faster on the
attention even when nothing is skipped.  The path only still lost because indexer+top-k (20.9 ms)
cost more than the 12.1 ms saved.  Net end to end: +0.5 % pp512, -0.3 % pp1024, -1.1 % pp2048,
-1.1 % pp4096, -0.3 % pp8192 (≈ neutral, within run variance).  It removes the `n_kv == width`
numerics seam and makes qsa3 exercised at **every** context length - a `-c 2048` exercise used to be
silently dense, which is why the early "qsa3 is neutral" readings were vacuous.  Decode unaffected
(stays dense via `qsa_dense_decode_until`; verified tg64 25.80 -> 25.83).  PPL: c16384 bf16
3.3900 -> **3.3821**, q8_0 3.3879 -> 3.3861, c32768 4.3353 -> 4.3397 (noise).

**2. `GGML_CUDA_MMB_F32SPLIT` is now default 0.**  The F32 dense weights are all tiny-M (router
M=512, ssm alpha/beta M=48, hc `*_inject` M=4, shexp gate M=1) and both the MMB 128-row tile and
rocBLAS cost ~1.0 s at pp8192 (12 %).  rocBLAS measured faster: pp4096 896.0 -> **915.9**, pp8192
896.8 -> **902.9**.  A 16x256 small-M tile was tried and is **worse** (870/847 vs 895/885).

**Next-work order (revised):**

1. **The indexer** - the only thing keeping always-QSA from being a strict win.  pp2048:
   `indexer_topk_radix_histogram` 10.0 ms, `indexer_topk_deterministic_write` 4.7,
   `indexer_topk_count` 3.3, `indexer_topk_radix_select` 2.9 (n=96 each).
2. **A dedicated tiny-M F32 kernel** (or split-K): both current paths are ~10x off the memory-bound
   floor; A traffic = `(T/BN)*M*K*4`, B = `(M/BM)*T*K*4` (168 MB vs a 26 MB floor at M=512/T=2048).
3. The `mmb_*` kernels that still lead the profile (`mmb_routed_glu` 1873 ms real, `mmb_dense` 1758,
   `mmb_cvt_f32_bf16` 319) - §9/§10.
4. QSA v3 for gfx1200/gfx1100; then `qsa3_attn_kernel` itself (674.9 ms, ~93 % of qsa3).

---

## UPDATE — session 3 (2026-09-19): the qsa3 sort is done too

The `qsa3_rows_kernel` sort (session 2's remaining item, 441 ms of the 1152 ms qsa3 total) was
rewritten as a **bitmap counting sort**: `O(ns + nk/32)` instead of `O(ns^2)`, **order-exact** (PPL
bit-identical).  **Rows 441.0 -> 25.5 ms; qsa3 total 1151.9 -> 728.6 ms (4.04x vs the VEC kernel).**

Two facts from getting there, so session 4 does not repeat the detour:

1. **The index rows are NOT sets of 4-key blocks** (that was the natural guess from the source
   comments) - they are a handful of **long contiguous runs** (row0 = one run of 2051 keys, row2 =
   runs of 436/1611/4; 1-6 runs per row).  A block/group-based sort would have been *wrong*.  Always
   dump the real data before choosing the algorithm (`GGML_CUDA_QSA3_DUMP` was removed again; the
   procedure is in `README.md`).
2. A host-side `cudaMemcpy` of op data **must be ordered on `ctx.stream()`**, not the default stream,
   or it reads the tensor before the producing kernel has run.  The first dump was garbage for
   exactly this reason.

**Revised next-work order** (after session 3 - **SUPERSEDED by the session-4 UPDATE above**; kept
for history):

1. The `mmb_*` kernels lead the profile (`mmb_f32split` 2164 ms, `mmb_routed_glu` 2085+1626,
   `mmb_dense` 1904+1606, `mmb_cvt_f32_bf16` 648) - see §9/§10 for the Q8_0 IU8 and bf16-producer ideas.
2. QSA v3 for **gfx1200/gfx1100** (needs an RDNA4/RDNA3_0 WMMA variant; the wrapper is currently a
   deliberate no-op there, so only the VEC path runs).
3. `qsa3_attn_kernel` itself (674.9 ms, now 93 % of qsa3) - the remaining qsa3 cost.

---

## UPDATE — session 2 (2026-09-19): QSA v3 is DONE

**Read §8 as history; the work is landed.**  Everything below is still accurate for the MMB part.

Session 2 delivered the packed-block WMMA QSA prefill (§8) and extended it to every KV cache type:

* New file `ggml/src/ggml-cuda/fattn-qsa3.cu` (rows / merge / attn kernels, ported from the Strix
  Halo branch's `qsa-attn`).
* New optional op srcs `src[7]`/`src[8]` via `ggml_flash_attn_qsa_set_packed`; the pack itself is a
  pure graph composition (no new ggml op), built in `src/models/qwen4exp.cpp`.
* Gate: **`GGML_CUDA_QSA3=1`**, default OFF.  Prefill-only (`q->ne[1] >= 128`) and RDNA3_5, so the
  W = 1..8 decode/verify band still takes the VEC kernel and width purity holds by construction.
* gfx1201 + gfx1100 TU compiles verified (portable WMMA wrapper, no-op on RDNA4).

Measured (gfx1151, IQ4_XS, `-b/-ub 2048`): pp8192 **f16 819.4->861.7, bf16 791.2->854.5, q8_0
780.5->856.6**; PPL parity at c16384/c32768 on all three (e.g. bf16 3.3932 vs 3.3900); greedy text
coherent.  Kernel profile pp8192: **VEC 2944.4 ms -> qsa3 1151.9 ms** (attn 682.3 + rows 441.0 +
merge 28.6).

Three things session 3 must not re-derive - full detail in `README.md`:

1. **Do not remove the dense startup arm.**  `n_kv <= 2051` takes the dense path because the
   selection covers every cell there; forcing QSA there is measurably worse (pp2048 912.5 -> 903.2).
   Consequence: **qsa3 only engages above 2051 tokens**, so `-c 2048` tests silently prove nothing.
2. **The top-k rows are unsorted**, so the rank-sort in `qsa3_rows_kernel` is required.  Disabling it
   drops that kernel 441 -> 8 ms but blows the attn kernel up 682 -> 19593 ms and halves throughput.
3. **Cast on the natural contiguous cache view, never on a permuted one**, and route quantized types
   through F32 - the backend `dup` cannot permute a quantized tensor and only dequantizes to F32
   (otherwise: `ggml/src/ggml-cpu/ops.cpp:578` abort, once per QSA layer).

**Revised next-work order** (was: after session 2; superseded by the session-3 "UPDATE" above)

1. ~~`qsa3_rows_kernel` sort~~ **DONE in session 3** - bitmap counting sort, 441 -> 25.5 ms.  See the
   session-3 UPDATE at the top.  **Do not** reorder `idx` at the indexer: that tensor is shared with
   the VEC decode path, so it would be a decode numerics change.
2. The `mmb_*` kernels now lead the profile (`mmb_f32split` 2164 ms, `mmb_routed_glu` 2085+1626,
   `mmb_dense` 1904+1606, `mmb_cvt_f32_bf16` 648) - see §9/§10 for the Q8_0 IU8 and bf16-producer ideas.
3. QSA v3 for **gfx1200/gfx1100** (needs an RDNA4/RDNA3_0 WMMA variant; the wrapper is currently a
   deliberate no-op there, so only the VEC path runs).

---

## 0. TL;DR for the next session

1. Everything lives in the worktree `~/llama-wip-mmb`, branch **`wip-mmb-general`**, based on the
   maintainer's applied delivery tree at **`8a2567e1e`**.
2. The complete change is backed up here as
   **`wip/mmb-general/mmb-general.patch`** (combined, 3 files, +1451/−2) and
   **`wip/mmb-general/patches/0001..0009-*.patch`** (per-commit), plus `commits.txt`.
   Verified: `git apply --check` passes on a clean checkout of `8a2567e1e`.
3. Rebuild with the commands in §3; reproduce the numbers in §5.
4. **QSA v3 is DONE** (see the UPDATE section) — the remaining QSA work is the 441 ms sort in
   `qsa3_rows_kernel`.  The optional second lever is now **Q8_0 IU8-WMMA** (§9).

---

## 1. Why (context in one paragraph)

On Strix Halo (gfx1151) the delivery's prefill was ~2x behind the tuned RDNA3_5 stacks
(pwilkin `strix-halo` full env: 1349/1403 t/s on uniform IQ4_NL; halogen-flash 1246/1424 t/s on the
same llama.cpp GGUFs).  Root cause: our weight GEMMs run the **integer/vector MMQ** path while both
use **bf16 WMMA tensor cores** (roof ≈ 59 vs 29.7 TFLOPS on this part; prefill is compute-bound).
The parked port (`archive/work/wip-archive/iq4nl-prefill/`) proved the idea but was IQ4_NL-only and
was parked on the purity clash.  This session generalised it to the types the delivery's own models
actually use and made it purity-safe.  **GFX1201 note:** the kernels use the first-gen gfx11 WMMA
builtin, which gfx12 does not have — hence the RDNA3 gate and the portable wrapper (§6).

---

## 2. Where the code is / what changed

Two changesets in one WIP tree.

**MMB** (session 1):

| file | change |
|---|---|
| `ggml/src/ggml-cuda/mmb.cu` | new: the dequant row helpers, tile GEMM kernels, routed/GLU kernels, dispatch, predicates, shadow helpers |
| `ggml/src/ggml-cuda/mmb.cuh` | new: the exported API + `ggml_cuda_mmb_{dense,routed}_will_take` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MMB dispatch hooks in `ggml_cuda_mul_mat` / `_mul_mat_id` / `_glu`, and the **per-weight-type fusion stand-down** in the graph optimizer |

**QSA v3 / qsa3** (sessions 2-4):

| file | change |
|---|---|
| `ggml/src/ggml-cuda/fattn-qsa3.cu` | new (~540 lines): rows/merge/attn kernels, the bitmap sort, the support check + launcher |
| `ggml/src/ggml-cuda/fattn-qsa.cu` / `.cuh` | qsa3 declarations + a short "prefer qsa3 when supported" hook in the VEC launcher |
| `ggml/include/ggml.h` + `ggml/src/ggml.c` | `ggml_flash_attn_qsa_set_packed` (op `src[7]`/`src[8]`) |
| `src/models/qwen4exp.cpp` | the pack helpers (`qsa_pack_{keys,values}_graph`, `qsa3_f16_cast`), the `GGML_CUDA_QSA3` gate, and the **`LLAMA_QSA_DENSE_SHORTCUT` default flip** |

Digest: **9 files, +2190/-11 across 13 commits**.

Both new `.cu` files are picked up by the existing `file(GLOB … "*.cu")`, but the glob is evaluated at
**configure** time - adding `fattn-qsa3.cu` needs a one-off `cmake -S . -B build-rocm` re-run (see §3).

### WTYPE map (used throughout the file)

| WTYPE | type | bytes per k-step (64 values) | row block |
|---|---|---|---|
| 0 | IQ4_NL | 36 | 18 B / 32 |
| 1 | Q8_0 | 68 | 34 B / 32 |
| 2 | BF16 (direct, no dequant) | 128 | 2 B / value |
| 3 | Q4_K | 144 (16 hdr + 128) | 144 B / 256 |
| 4 | Q5_1 | 48 | 24 B / 32 |
| 5 | IQ3_S | 110 | 110 B / 256 |
| 6 | Q5_K | 176 | 176 B / 256 |
| 7 | Q6_K | 210 | 210 B / 256 |
| 8 | IQ4_XS | 136 | 136 B / 256 |
| 9 | Q3_K | 110 | 110 B / 256 |
| 10 | IQ3_XXS | 98 | 98 B / 256 |

### Key design decisions (do not undo)

* **Prefill-only.**  `ggml_cuda_mmb_supported_*` require `T >= GGML_CUDA_MMB_MIN_T` (default 512).
  The decode/spec-verify band is `W = 1..8`, so MMB is unreachable there — `W=1..8` bit-identity and
  `plain == draft-mtp` hold **by construction**, not by measurement.
* **Per-weight-type fusion stand-down.**  The graph optimizer's MoE pair and SWIGLU→mmq fusions
  stand down only when MMB will actually take *that weight type*
  (`ggml_cuda_mmb_dense_will_take` / `_routed_will_take`), never on the global gate.  This is the
  parked handover's resume-checklist item #5 and it is what stopped MMB-on from regressing models
  whose expert type it could not accelerate.
* **RDNA3_5 only by default.**  `mmb_enabled()` gates to `GGML_CUDA_CC_IS_RDNA3_5`; RDNA3_0
  (gfx1100/1101/1102) shares the gfx11 WMMA builtin but is untested, so it needs
  `GGML_CUDA_MMB_RDNA3=1`.  gfx12 is excluded (different builtin).
* **Portable WMMA wrappers.**  `mmb_wmma_bf16` / `mmb_wmma_f16` select the gfx11 builtin on
  non-`RDNA4` and are a **deliberate no-op** on `RDNA4`, so `mmb.cu` *compiles* for gfx1201 while
  never being launched there.  (Verified: `mmb.cu` and `fattn-qsa.cu` both compile for gfx1201 — §7.)
* **On-the-fly dequant, no shadow.**  A persistent bf16 shadow is impossible for a 120 GiB model's
  experts, so every non-trivial type is dequantized per k-step.  The legacy Q6_K shadow path still
  exists but is no longer needed for Q6_K (WTYPE 7 is on-the-fly).
* **IQ3_XXS fused GLU is default-off.**  Its routed path is a win, its fused GLU is a measured net
  loss (§5); `GGML_CUDA_MMB_IQ3XXS=1` re-enables the GLU arm for tuning.

---

## 3. Build / run

The build dir in the worktree is already configured; to rebuild:

```sh
cd ~/llama-wip-mmb
export ROCM_PATH=/opt/rocm-7.14-gfx1151
HIPCXX=$ROCM_PATH/lib/llvm/bin/clang HIP_PATH=$ROCM_PATH cmake -S . -B build-rocm \
  -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 -DHIP_PLATFORM=amd \
  -DGGML_HIP_GRAPHS=ON -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_PATH/lib" \
  -DCMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS= \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build-rocm -j 16 --target llama-bench llama-perplexity llama-cli llama-server
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
```

To recreate from scratch (the backup is authoritative):

```sh
git -C ~/llama.cpp worktree add --detach /tmp/mmb-rebuild 8a2567e1e
cd /tmp/mmb-rebuild
git apply /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/mmb-general.patch
# then the cmake configure+build above with -B build-rocm
```

**Caveat:** the WIP base is `8a2567e1e` (the maintainer's applied tree), **not** the current canonical
release tip (r9 = `76b10f1fb8391562e30364d6c307e6606100bc57`).  Before landing, the change must be
rebased onto a canonical fork rebuilt at the release base `ebbb18522` + `scripts/apply-all.sh`, then
the delivery `patches/` regenerated (see `AGENTS.md`).  None of the touched hunks are in code that
r9 changed (the r9 work was the block-15 tile kq-mask), so the rebase should be mechanical.

---

## 4. A/B and PPL commands used

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=<model>.gguf

# prefill A/B (MMB off vs on)
./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 2048 -ub 2048 \
    -p 2048,8192 -n 0 -d 0 -r 2
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 \
    -b 2048 -ub 2048 -p 2048,8192 -n 0 -d 0 -r 2

# PPL parity (the correctness gate for each new dequant)
F=~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt
./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
GGML_CUDA_MMB=1 ./build-rocm/bin/llama-perplexity -m "$M" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -f $F -c 2048 -b 2048 -ub 2048
```

Always **warm the model into page cache first** (`dd if=$M of=/dev/null bs=4M`); the box's page-cache
state swings prefill noticeably, and never run benches in parallel.

`GGML_CUDA_MMB=1` prints a few one-shot `MMB_GLU …` / `MMB_SHADOW …` lines — that is normal.

---

## 5. Measured results (gfx1151, ROCm 7.14, `-b/-ub 2048`, bf16 KV)

> **These were taken before the 2026-09-19 default flips** (`GGML_CUDA_MMB_F32SPLIT` 2->0 and
> `LLAMA_QSA_DENSE_SHORTCUT` ON->OFF).  The tables are still the right *method* and the MMB `off`
> column is unaffected, but the `on` column moves a little; re-measure rather than copy.

| model / test | MMB off | MMB on | PPL off → on |
|---|---:|---:|---|
| **Q4_K_M** pp2048 | 604.6 | **1015.2 (+68 %)** | — |
| **Q4_K_M** pp8192 | 568.9 | **888.1 (+56 %)** | 10.3328 → 10.2716 |
| IQ4_XS (Flash-Next) pp2048 | 723.3 | 838.0 (+15.9 %) | — |
| IQ4_XS pp8192 | 685.5 | 779.5 (+13.7 %) | 10.6938 → 10.6440 |
| Gemma4-26B-A4B Q8_0 pp2048 | 2058.5 | 2340.1 (+13.7 %) | — |
| Gemma4-26B-A4B Q8_0 pp8192 | 1723.5 | 1928.5 (+11.9 %) | parity (prose prompt is tokenizer-mismatched; both arms same ballpark) |
| Qwen3.6-35B-A3B UD-Q5_K_M pp2048 | 2045.3 | 2303.0 (+12.6 %) | — |
| Qwen3.6-35B-A3B UD-Q5_K_M pp8192 | 1917.8 | 2141.3 (+11.7 %) | 14.4349 → 14.3907 |
| Qwen3.6-35B-A3B Q6_K pp2048 | 2128.8 | 2143.2 (+0.7 %) | — |
| Qwen3.6-35B-A3B Q6_K pp8192 | 1974.6 | 2012.3 (+1.9 %) | 14.3682 → 14.3296 |
| Qwen3.6-35B-A3B UD-Q3_K_M pp2048 | 2091.9 | 2295.2 (+9.7 %) | — |
| Qwen3.6-35B-A3B UD-Q3_K_M pp8192 | 1941.6 | 2120.2 (+9.2 %) | 14.6517 → 14.6248 |

**Interpretation**

* The Q4_K_M jump is its **Q4_K MoE expert GLU + routed down** (WTYPE 3); IQ4_XS is IQ3_S gate/up +
  IQ4_NL down; Q5_K_M/Gemma4 are their expert types.
* **Q6_K is correct but barely moves** (+2 %): its MMQ path is already efficient on this workload.
  Keep it for coverage, not as a headline win.
* **IQ3_XXS finding:** routed is a win, fused GLU is a loss.  `UD-Q3_K_M` full-on 1921 vs routed-only
  2120 vs off 1942 t/s pp8192 → the fused GLU is default-off (`GGML_CUDA_MMB_IQ3XXS=1` re-enables).
* **Model composition trap:** file names mislead.  *Flash-Next UD-IQ4_XS* is IQ4_NL 52 % + IQ3_S 36 %
  + Q8_0 9.5 % (IQ4_XS type only 1 %).  *MiniMax-M3 UD-IQ4_XS* is IQ3_S 56 % + IQ4_XS 35 % → now
  fully covered.  *UD-Q3_K_M* is IQ3_XXS 47 % + IQ4_XS 33 %.

### Coverage matrix (final)

| type | dense | routed + GLU | note |
|---|:--:|:--:|---|
| IQ4_NL, Q8_0, Q4_K, Q5_1, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K | ✓ | ✓ | |
| IQ3_XXS | ✓ | routed only | fused GLU default-off (net loss) |
| F16 / BF16 | — | — | already tensor-core via `mmf` WMMA |
| F32 | ✓ (f32split) | — | f16 hi/lo WMMA |
| Q2_K, IQ2_*, IQ1_* | — | — | deliberately out of scope |
| IQ3_M, IQ3_XS, IQ4_S, IQ4_M | — | — | **do not exist in current ggml** |

Not yet covered (optional): the legacy **Q4_0 / Q4_1 / Q5_0** (32-value blocks; Q5_1 is already done,
so these are small additions).

---

## 6. Environment knobs

MMB (`mmb.cu` / `mmb.cuh`):

| env | default | meaning |
|---|---|---|
| `GGML_CUDA_MMB` | 0 | master gate (default **off**) |
| `GGML_CUDA_MMB_MIN_T` | 512 | prefill-only threshold |
| `GGML_CUDA_MMB_RDNA3` | 0 | allow RDNA3_0 (untested) |
| `GGML_CUDA_MMB_GLU` | 1 | fused gate/up+swiglu arm |
| `GGML_CUDA_MMB_IQ3XXS` | 0 | enable the (net-loss) fused GLU arm for IQ3_XXS |
| `GGML_CUDA_MMB_BF16W` | 1 | BF16 dense weights via MMB |
| `GGML_CUDA_MMB_F32SPLIT` | **1** | F32 dense via f16-hi/lo WMMA.  **Mode 1 = shape-aware and DEFAULT since session 5**: MMB only when `M >= 128` (the MoE router, 2.4x faster); rocBLAS keeps `hc_*_inject` (M=4) and `ssm_alpha/beta` (M=48).  `0` = all rocBLAS, `2` = all MMB (pre-session-5, worst).  Measured pp8192 902.0 (0) / **929.8 (1)** / 903.0 (2) |
| `GGML_CUDA_MMB_TINY_M` / `_TINY_TT` | 1 / 1 | the hc `*_inject` warp-per-token kernel.  `_TINY_M=0` forces rocBLAS (A/B); `_TINY_TT` = tokens per warp (**1 is measured best**; 2 and 4 are worse — L2 serves the W panels) |
| `GGML_CUDA_MMB_F32SPLIT_MIN_M` / `_MIN_K` | 128 / 0 | the mode-1 shape rule.  **Do not add a K condition**: taking MMB for long K (the K=10240 inject pair) measured 1.825 vs 1.332 ms and cost 375 ms |
| `GGML_CUDA_MMB_TALL` | 2 | the tall-M tile class |
| `GGML_CUDA_MMB_SHADOW` / `_SHADOW_MB` | 0 / 6144 | legacy bf16 shadow (Q6_K/IQ4_NL); not needed now |
| `GGML_CUDA_MMB_TILE` | -1 | **diagnostic**: force narrow(0)/wide(1) tile |
| `GGML_CUDA_MMB_LOG` | 0 | **diagnostic**: one-shot `MMB_DENSE` shape log |

QSA / qsa3 (`fattn-qsa3.cu`, `src/models/qwen4exp.cpp`):

| env | default | meaning |
|---|---|---|
| `GGML_CUDA_QSA3` | 0 | **the qsa3 gate** (the packed-block WMMA prefill path).  Session 4 put the *dense-startup* default in line with it, but qsa3 itself is still opt-in |
| `LLAMA_QSA_DENSE_SHORTCUT` | **0** | **FLIPPED ON->OFF on 2026-09-19 = always QSA** (maintainer decision).  `=1` restores the dense arm (the `LLAMA_QSA_SPARSE_FA=0` cross-check) |
| `LLAMA_QSA_DENSE_DECODE_UNTIL` | 65536 (gfx1151) | decode stays dense below this; independent of the above |
| `LLAMA_QSA_SPARSE_FA` | on | `=0` = the dense masked reference path |
| `GGML_CUDA_QSA3_DUMP` | (removed) | was a temporary idx-row dump; it is **gone from the tree** - re-add it if the sort/structure needs re-checking (the procedure is in `README.md`) |

---

## 7. Post-MMB profile + where the time now goes

> **This is the SESSION-1 profile** (MMB landed, no qsa3, `F32SPLIT`/shortcut at their old defaults).
> It is still the best *shape* inventory, but the ranking has moved - qsa3 took the QSA kernel from
> 2944 -> 728.6 ms and session 4 turned the F32 path off.  **Current numbers: the session-4 UPDATE**
> and the always-QSA section in `README.md`.

Q4_K_M pp8192, `GGML_CUDA_MMB=1`, rocprofv3 kernel trace (**total 17.75 s**, was ~28 s off):

| kernel | time | share |
|---|---:|---:|
| `flash_attn_qsa` | 2.94 s | **16.6 %** |
| `mmb_dense_kernel` (Q8_0 PLE + Q5_1) | 4.06 s | 23 % |
| `mmb_routed_glu` (Q4_K gate/up) | 2.46 s | 14 % |
| `mmb_routed` (Q4_K down) | 1.39 s | 7.8 % |
| `dsv4_hc_pre` + `_post` | 1.44 s | 8.1 % |
| `rms_norm_f32<1024>` | 0.65 s | 3.6 % |
| `gdn_bf16_scan` | 0.62 s | 3.5 % |
| `mmb_f32split` | 0.65 s | 3.7 % |
| `mmb_cvt_f32_bf16` | 0.65 s | 3.7 % |

**Both leading items were probed and are not tuning-reachable** (see `README.md` §investigation):

* **Q8_0 tiling is optimal** — forcing narrow/wide changes pp8192 by +0.6 % / −2.4 %
  (802 / 798 / 778 t/s) → the kernel is occupancy/LDS-bound.
* **QSA gather 8B→16B is neutral** (796 vs 798) → not load-issue-bound; it is
  compute/reduction/occupancy-bound.

---

## 8. QSA v3 — **DONE in session 2** (kept as the design record)

**Status: landed behind `GGML_CUDA_QSA3=1`; results and the remaining optimization are in the
"UPDATE" section above and in `README.md`.  The plan below is what was actually built - keep it for
the gfx1200/gfx1100 follow-up.**

**Goal (achieved):** replace, for prefill only, the VEC `flash_attn_qsa` score/PV with a packed-block
WMMA implementation (pwilkin's `qsa3`).  Result: 2944 ms -> 1152 ms on the kernel, +5-10 % prefill.

**Approved:** the maintainer accepts a prefill re-baseline (greedy prefill output changes) as long as
it is **consistent, deterministic and coherent**.  Keep the **VEC `flash_attn_qsa` for the W=1..8
band** (gate the new path `n_query >= 128`), so width purity and `plain == draft-mtp` are untouched
by construction.  Document the re-baseline with a new same-seed hash.

**Reference:** `~/pwilkin-llama-cpp` (branch `strix-halo`), `ggml/src/ggml-cuda/qsa.cu` (445 lines):
`qsa3_rows_kernel`, `qsa3_merge_kernel`, `qsa3_attn_kernel`, plus the graph's `qsa_pack_keys` /
`qsa_pack_values` tensors (his `src[6]`/`src[7]`).

**Concrete steps**

1. **Pack ops (graph side).**  Build contiguous f16 key/value block buffers once per graph from the
   cache + the selected indices.  In our tree that is `src/models/qwen4exp.cpp` +
   `GGML_OP_FLASH_ATTN_QSA` (see `build_qsa_top_k`, `build_qsa_store_k`) and
   `ggml/src/ggml-cuda/fattn-qsa.cu`.  Decide whether to add the buffers as extra op `src[]` or to
   build them inside the op's launcher (the latter avoids graph/allocator changes but re-packs per
   call).  The handover for the parked port (`archive/work/wip-archive/iq4nl-prefill/`) and
   `wip/prefill-arrangements/README.md` both scope this.
2. **Descriptor.**  Port `qsa3_rows_kernel` + `qsa3_merge_kernel`: merge `G = 4` consecutive queries'
   top-k lists into a sorted, deduplicated, block-aligned array of block ids + a 16-bit per-query
   membership mask + count (his `ublk`/`umask`/`ucount`).
3. **Attention.**  Port `qsa3_attn_kernel` (16×16×16 f16 WMMA over the packed blocks, membership
   mask folded into the score pass, online softmax as now).  Reuse the portable-WMMA pattern from
   `mmb.cu` so gfx1201 still compiles (or keep it gfx11-only with stubs, matching
   `gated_delta_net_chunked_bf16_gfx11.cu`).
4. **Gate + validate.**
   * prefill-only (`n_query >= 128`); VEC kernel unchanged in the band;
   * `test-backend-ops -o FLASH_ATTN_QSA` must stay green (add cases as needed);
   * the `W = 1..8` logits matrix on `LLAMA_QSA_SPARSE_FA` on/off/new must keep the band identical;
   * same-seed greedy text: new build vs old, once, to record the prefill re-baseline hash;
   * PPL vs the dense masked oracle (`LLAMA_QSA_SPARSE_FA=0`) within noise — **not** MTP acceptance,
     which is blind to a defect present in both draft and target (`GREEDY-PURITY.md` §21).

**Size estimate:** the scoping doc priced this at 6-10 days.  It is the single largest remaining item.

---

## 9. NEXT WORK #2 — Q8_0 IU8-WMMA dense path

**Goal:** replace the dequant-to-bf16 MMB Q8_0 dense path (4.06 s, 23 %) with an **int8 → int8 tensor
core** GEMM (`v_wmma_i32_16x16x16_iu8`), avoiding the LDS dequant staging that makes the current
kernel occupancy-bound.

**Facts already established**

* The big Q8_0 shapes: `attn_qkv M=10240 K=2560`, `attn_gate M=6144 K=2560`, `ssm_out M=2560 K=6144`,
  `hc_* M=320/10240`, `ffn_*_shexp M=640 K=2560`.
* The tile heuristic is optimal; the kernel is **occupancy/LDS-bound**, not tiling-bound.
* `v_wmma_i32_16x16x16_iu8` measured ~171-175 T-MAC/s on gfx1201 (see
  `wip/q8-prefill-tuning/README.md`); the same doc notes the mainline MMQ Q8_0 kernel reaches only
  31-34 % of the int8-WMMA ceiling and its k-loop has no double-buffering.

**Steps:** quantize the activations to int8 (Q8_1-style scales) — note this is a numerics change too,
so the same prefill-only gate applies — then a WMMA-iu8 tile kernel with int32 accumulators; A/B vs
the current bf16 Q8_0 MMB path on IQ4_XS/Q4_K_M.

---

## 10. Smaller remaining items

* **bf16-producer marking** — kill `mmb_cvt_f32_bf16` (0.65 s, 3.7 %): the parked port has the
  plumbing (`ggml_cuda_mmb_mark_bf16_only`, `ggml_cuda_mmb_cache_reserve`,
  `ggml_cuda_mmb_cache_lookup`) but nothing ever calls it.  Needs a graph-optimizer pass that marks a
  producer's output bf16-only when every consumer is MMB.  Prefill-only.
* **HC prefill fusion** — `dsv4_hc_pre`+`_post` are 8.1 %; pwilkin's `hc_combine_norm`/`hc_mix_reduce`
  are ~0.93 s vs our 1.44 s.  Block 14 already has the decode-band fused hc ops; the prefill arm is
  the gap.
* **Legacy Q4_0 / Q4_1 / Q5_0** — trivial next to what is done (32-value blocks).
* **Optional 7900 XTX test** (gfx1100, single card, small model) with `GGML_CUDA_MMB_RDNA3=1` — the
  tiling almost certainly needs an RDNA3_0 pass.  The maintainer wants gfx1151 exhausted first.

---

## 11. Purity / landing gates (unchanged from the parked handover)

Before this can be opt-in, let alone defaulted on:

1. `W = 1..8` logits matrix with `GGML_CUDA_MMB=1` == off (should be identical **by construction** —
   `T >= 512` — but must be run).
2. MTP acceptance gate (`benchmarks/mtp-adaptive-methodology.md`).
3. `test-recurrent-state-rollback`.
4. `test-backend-ops` suites (FLASH_ATTN_QSA / GATED_DELTA_NET / FLASH_ATTN_EXT).
5. Same-seed **prefill re-baseline** documented with a fresh hash.
6. gfx1100 / gfx1201 compile + consistency (gfx1201 TU compile already verified — §12).
7. Default stays **off**; the env kill-switch (`GGML_CUDA_MMB=0`) is retained.

The repo rulebook is `AGENTS.md` + `GREEDY-PURITY.md`; landing must regenerate `patches/` from a
canonical fork rebuilt at the release base (`scripts/make-patches.sh` / `make-release.sh`), never from
the drifted working tree.

---

## 12. Verification already done (cumulative; update this when the tree changes)

**Session 1 (MMB):**

* gfx1151 build clean after every commit; `llama-bench` / `llama-perplexity` / `llama-cli` /
  `llama-server` all build.
* PPL parity recorded for: Q4_K_M, IQ4_XS, UD-Q5_K_M, Q6_K, UD-Q3_K_M (table in §5).

**Sessions 2-4 (qsa3 + the default flips):**

* **gfx1151 always**; `fattn-qsa3.cu` also **TU-verified for gfx1201 and gfx1100** (extract the command
  from `build-rocm/compile_commands.json`, swap `--offload-arch=gfx1151`; `mmb.cu` and `fattn-qsa.cu`
  verified the same way in session 1).
* `fattn-qsa3.cu` is **order-exact**: the bitmap sort reproduces the rank sort's output, proven by
  PPL being **bit-identical** across the rewrite (c16384 3.3900, c32768 4.3353, q8_0 3.3879 pre-flip).
* Post-flip PPL: c16384 bf16 **3.3821**, q8_0 **3.3861**, c32768 bf16 **4.3397**.
* Decode unaffected by the always-QSA flip: tg64 shallow 25.80 -> 25.83.
* Greedy text coherent on a >2051-token prompt; the prefill re-baseline is expected and approved.
* Combined patch `git apply --check` clean on a fresh `8a2567e1e` worktree (re-verified at session 4).

**Still NOT run (blocking promotion, see §11):** nothing — the `W = 1..8` matrix is now done
(session 5d UPDATE at the top; probe `tests/test-logits-width-probe.cpp`, gate PASSES).  A same-seed
**prefill re-baseline hash** also exists now (`bbd4bcb519e4` with MMB+QSA3, `5120b28f2879` without).

---

## 13. Gotchas the next session should not re-learn

* `llama-cli` **must** be run with `--single-turn` (and usually `--no-display-progress`) or it blocks.
* Warm the page cache before every bench; never run benches in parallel.
* `llama-bench` `-b/-ub 16384` on the 94 GiB IQ4_XS can OOM at pp16384 (context creation fails); use
  `-ub 2048` (the server's config) for these models.
* The `-lm none -lzm on` flags are needed for the Q4_K_M/lazy models in `llama-bench`.
* `mmb_dq_row_*` for the K-/i-quants are loaded from the row base inside `store_lds` (the fields are
  non-contiguous), so their `load_regs` branch is intentionally a no-op — do not "optimise" it back
  into the register prefetch.
* A `case`/`if` added to the predicates must be added to **all** of:
  `supported_mm`, `supported_mmid`, `supported_glu`, `dense_will_take`, `routed_will_take`, **and**
  the three dispatch helpers — a missing one is either a silent fallback or an uninstantiated launch.
* `mmb.cu` uses `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (gfx11); gfx12 needs
  `…_gfx12` with different fragment types.  The wrapper is a no-op on RDNA4 on purpose.

---

## 14. Repo conventions

* This WIP lives under `wip/` in `llama-cpp-rdna-boosts` (the only push target:
  `github.com:stew675/llama-cpp-rdna-boosts`).  **Never** push anything out of `~/llama.cpp`.
* Commit records/doc updates to the delivery repo; keep the experimental code in the worktree and
  regenerate `mmb-general.patch` from it whenever the code changes.
* When the work is promoted, it becomes a block amendment (the parked handover suggests block 08 for
  the MMB module) and follows the promotion rule in `AGENTS.md`.
