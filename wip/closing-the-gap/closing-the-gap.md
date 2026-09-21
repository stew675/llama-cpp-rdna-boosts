# Closing the gap — `beta/mmb-general` vs pwilkin's `strix-halo` prefill

**Date:** 2026-09-20 (snapshot) · **updated:** 2026-09-21
**Box:** `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`), 123 GiB RAM / 124 GB unified VRAM
**Scope:** a 1:1 prefill comparison on **pwilkin's own uniform-IQ4_NL model** (not just our mixed UD-IQ4_XS), a kernel-level profile diff on the uniform model, and a gate-ablation of pwilkin's own stack on this box to price the still-missing families. This is an investigation record, not a delivery change.

> **This file is now a WIP planning record under `wip/closing-the-gap/` (moved here 2026-09-21).**
> Sections **0–11 below are the 2026-09-20 snapshot**, kept for its dated measurements. Read the
> **Update 2026-09-21** section first — two things it pinned have moved: our side became
> `beta/mmb-general` (12 patches), and pwilkin's branch moved `f5daaa3cf` → `b0f31f587`.

---

## Update 2026-09-21 — current state of play (read first)

This section supersedes the stale references in the body. The body's measurements remain valid as
**dated, gated-tree** evidence, but two references have moved and the plan needs five additions.

### A. Our side: `wip/mmb-general` was promoted to `beta/mmb-general`, 5 → 12 patches

The body compared a **5-patch, gfx1151-only WIP** at tip `90bf12997` (`~/llama-wip-mmb`). The current
reference is **`beta/mmb-general`** — **12 patches**, applied tree
**`bca69f23dd29acef2d8898c6fd492104e078eef1`**, verified `git am` **12/12** on top of the r12 delivery
(`~/llama.cpp` HEAD `72176ae8a`, tree `8a80535e…`).

* The new patches are the **gfx1201 (RDNA4) port** (0006–0010) and the **gfx1100 (RDNA3_0) deltas**
  (0011–0012). They are arch-scoped; on **gfx1151** the code is the core the body measured, so the body's
  gfx1151 numbers carry over except where §C says otherwise.
* **What the 12-patch beta still does NOT add** (grep of the applied tree, not inference):
  `gdn-conv.cu`, `ple-conv.cu`, `norm-gated.cu`, `idx-relu-sum.cu` and `hc-cn.cu` are still **absent**.
  `hc_gate_mix_kernel` **exists** in `mmb.cu` but has **no call site** — it is dead code behind
  `LLAMA_HC_GATEMIX` / `mmb_cfg().gatemix` (per-arch default `0`). So body §8.2 items **1–6 remain
  open**; item 1 is “present but unwired”, not “absent”, and the work is the *matcher/call site*, not
  porting the kernel.

### B. Pwilkin's side: `f5daaa3cf` → `b0f31f587`, 10 new commits

The body pinned `f5daaa3cf` (2026-09-12). The branch tip is **`b0f31f587`** (2026-09-16). The delta:

**Prefill-relevant**

| commit | what | why it matters here |
|---|---|---|
| `40a9f4d01` | *hip: extend MMB quants and fuse Flash-Next F32 PLE* | MMB quant coverage goes from a handful of types to **23** (adds Q4_0/Q4_1/Q5_0, Q2_K, the whole IQ1/IQ2 family, MXFP4, NVFP4) through a new `mmb-quant.cuh` generic dispatcher; the direct **PLE conv** now also takes **F32** weights (Flash-Next's PLE weights). Real gap: our beta covers **10** types and has **no** `ple-conv.cu` at all. |
| `40c0b9c38` | *qsa: drop the dense mask only where the qsa3 kernel will consume the op* | Correctness **and** a prefill win on his tree: the mask-forced workaround cost `pp16384` 945.68 → 1067.80 t/s; the fix decides maskless at the use site and `GGML_ASSERT`s the invariant in the dispatcher. The failure mode it fixed was decode non-determinism (10/10 → 1/10) that collapsed long sessions into repetitions. Our derived-visibility/maskless path should be audited against this. |

**Decode / MTP-relevant**

| commit | what | why it matters here |
|---|---|---|
| `d67d58836` | *hip: enable sparse QSA decode and incremental indexer state* | New **sparse selected-cell decode** kernels (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh` WMMA) that read selected F16 K/V cells directly, plus an **incremental indexer-key cache** (`src/qsa-prefix-state.h`, `llama-memory-hybrid-idx.*`). Measured on his tree: serial depth-40000 **25.85 → 28.82 t/s**, MTP 40680-token **31.17 → 35.57** (first) / **32.69 → 39.10** (repeat), for 104 MiB @65k / ~416 MiB @256k of cache. This is a **new axis** the body only gestured at (its item 11). Our delivery has a *different* QSA-sparse-FA decode path plus an incremental **derived-block-vector** cache (`GGML_CUDA_QSA_INDEXER_CACHE`, default on) — overlapping, not equivalent; needs a 1:1 audit. |
| `0f2950198` | *qwen4exp: skip unused HIP decode indexer work* | A temporary dense-decode bypass, **superseded** by `d67d58836`. Listed only so it is not mistaken for the current state. |

**Correctness / housekeeping**

| commit | what |
|---|---|
| `b0f31f587` | QSA block window sized by the highest stored position, not the occupied-cell count (fixed an M-RoPE image + MTP crash ~300 tokens after an image). |
| `14fff4f97` | Keep the `-1` selection sentinels out of the masked attention path. |
| `ac1ebb4e0` | **Compile in the tuned defaults and drop the env gating** — `mmb_enabled()`, `gdn_conv_enabled()`, `norm_gated_enabled()`, `norm_rows_enabled()`, `ple_conv_enabled()` now return `true`, and the `LLAMA_*` experiment switches are gone. |
| `0cfb81512` | Drop stale comment references to the removed gates. |
| `be905cf7d` | Recurrent cache: no warning for positions in a stateless cache. |
| `31b38632c` | server: a zero draft length means speculation off. |

### C. Impact on the plan

1. **The §6 ablation price-list can no longer be reproduced against current pwilkin HEAD.**
   `ac1ebb4e0` deleted the env switches Appendix D zeroed. Re-measure against `b0f31f587` as a *default*
   build, or bisect by reverting the compiled-in defaults; do not re-run the old env ablations.
2. **The prefill gap inventory (§8.2 items 1–6) is unchanged** — the beta set did not close any of them.
   Item 3 is now *larger*, because pwilkin also fuses the **F32 PLE** conv.
3. **Item 11 is promoted from a footnote to a first-class decode item** (sparse QSA decode + incremental
   indexer). It is the one newer pwilkin feature that is a measurable, self-contained optimisation
   rather than a refinement, and it is orthogonal to the prefill campaign — workable in parallel.
4. **MMB quant coverage:** our beta's 10 types vs his 23. Unlikely to move the uniform-IQ4_NL gap
   (both fire there), but a completeness/robustness gap for arbitrary GGUFs (Q4_0/Q4_1/Q5_0 and
   MXFP4/NVFP4 are common). Low-to-medium priority.
5. **Three cheap correctness items to port/audit** independent of perf: `40c0b9c38` (maskless only
   where qsa3 consumes it), `b0f31f587` (position-vs-cell block window), `14fff4f97` (sentinel
   handling). They prevent long-session corruption and are far cheaper than the perf items.
6. **MTP is not a missing optimisation in pwilkin's favour — it is a different axis.** He has upstream
   `draft-mtp` with a **fixed** `n_max` and only upstream's per-step `p_min`/`n_min` early stop; there
   is **no** cross-round adaptive controller in his tree. Our `draft-mtp-adaptive` controller is a
   depth-policy advantage that composes with his per-step decode gains. **Measured 2026-09-21** (see
   [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and §12): our plain decode is
   ahead (+2–6 %), the fixed-depth MTP **speedup is at parity** (ours `n3` 1.90/1.78/2.05 vs his
   1.91/1.79/2.02 on code/prose/recall), and our adaptive wins recall (2.40x) but over-drafts code and
   prose on qwen4exp — a tuning item, not a structural one.  The one real MTP gap is
   **`nextn_shared_target_tensors` support**: our build cannot load the shared MTP sidecar his IQ4_NL
   model ships (every draft position past the first fails an M-RoPE `X < Y` check), so we fell back to
   the `Q4_K_M` sidecar for the comparison.

### D. Body §6/§9 caveat

The kernel-level comparisons (`mmb_dense` +809 ms, HC, `rms`, `qsa3_attn` +195 ms) were made against
`f5daaa3cf`. Before trusting them again, re-profile `b0f31f587`: his tree gained the `mmb_quant`
dispatcher and dropped env gating, and the QSA decode/indexer changes add kernels to the trace.

### E. Where the current beta was built and validated

Applied and built on **gfx1151** on 2026-09-21 for the beta re-validation window
(`beta/mmb-general/BETA-TESTING.md`). Build: `~/bin/build-llama-rocm-714` from the `mmb-beta` branch of
`~/llama.cpp` (r12 + 12 patches, tree `bca69f23dd…`). The gfx1151 numbers in the body were measured on
the pre-beta WIP; the beta re-run is what confirms they still hold.

### F. Priority sequence (maintainer, 2026-09-21)

**Recall speed + correctness → decode speed + correctness → MTP tuning + correctness.**  The MTP
qualification is therefore **done to "is our MTP behind his?" depth only** and parked; its result and
the one real MTP gap are in [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md) and
§12 below.  The headline: our plain decode is ahead, the fixed-depth MTP **speedup** is at parity, and
the only MTP gap is **`nextn_shared_target_tensors` support** (we cannot load the shared MTP head
pwilkin's own IQ4_NL model ships).  The body's prefill items 1–8 are the "recall" phase.

---

## 0. TL;DR (2026-09-20 snapshot)

1. **On pwilkin's own model the WIP is no longer 2x behind.** It is **within ~4% at matched `-ub 2048`** and **~9% behind at pwilkin's best config (`-ub 16384`)**. The WIP took the uniform model from the delivery base's **734 t/s → 1149 t/s at pp8192/ub2048 (+57%)**; pwilkin gets 1194. Our earlier "2x behind" figure was the mixed model measured against *his fast path not firing there*.

2. **The remaining gap is NOT MMB and NOT the QSA attention kernel.** The WIP already **beats** pwilkin on `mmb_routed_glu` (−111 ms), the GDN recurrence (−250 ms), the F32 path (−239 ms vs his rocBLAS), the qsa3 sort (−86 ms) and the `mmb_cvt` bucket (−121 ms). The gap is concentrated in **three families pwilkin fuses and we do not**:
   * the **hyper-connection (HC) prefill fusions** — `hc_combine_norm` + `hc_gate_mix` — worth **−19.5%** on his stack when disabled;
   * the **depthwise conv1d** (`gdn_conv_direct`/`ple_conv`) — worth **−10.5%**;
   * the **gated RMS-norm** (`norm-gated`/`rms_rows`) and the **indexer relu-sum** — worth −2.9% / −1.3%.

3. **Our delivery's `hc_combine_norm` (in `hyperconn.cu`) is present but never fires** — verified 0 calls with and without the WIP's HC16 gate. This is the single highest-value fix on the table: pwilkin's equivalent fires 190× (554 ms) and he has an additional 408 ms `hc_gate_mix_kernel` we do not have at all. The missing gate-mix fusion also moves ~190 gate GEMMs *into* our `mmb_dense` (1266 launches vs his 956), which is most of the `mmb_dense` +809 ms delta.

4. **A separate, pre-existing delivery bug:** our tree (base r12 *and* the WIP) **cannot create a context at `n_batch == n_ubatch == n_ctx == 16384`** (`-b 16384 -ub 16384 -p 16384`), independently of MMB/HC16/QSA and of offload. pwilkin's tree runs the same config at **1399 t/s**. This caps the useful ubatch and is why we have no pp16384/ub16384 point.

5. **Priority:** (a) make/fix the HC `combine_norm` + gate-mix fusion; (b) port the depthwise conv1d; (c) the `-ub 16384` context bug; (d) `norm-gated` + `idx-relu-sum`; (e) tune `qsa3_attn` and the `mmb_dense` tall tile. (a)+(b) are ~30% of end-to-end prefill on pwilkin's own numbers, which is exactly the "1300+" delta.

---

## 1. Why this investigation

`wip/mmb-general/` generalized pwilkin's `mmb` weight GEMM, ported his `qsa3` attention and the bf16-producer machinery, and measured **+43–48%** over the delivery base on **our mixed UD-IQ4_XS** model. But pwilkin's headline `1300+` numbers were on **his uniform-IQ4_NL** checkpoint. The open question was: *what still stands between us and those numbers?*

The previous gap analysis (README "Attribution", `archive/work/wip-archive/iq4nl-prefill/HANDOVER-2026-09-12-…`) said the gap was the QSA kernel and the weight GEMM; both were since ported. This session re-measured everything 1:1 on the actual pwilkin GGUF, profiled both stacks, and priced the residual with pwilkin's own kill-switches.

---

## 2. Environment, builds, model

| | |
|---|---|
| WIP build | `~/llama-wip-mmb/build-rocm/bin/llama-bench`, tip **`90bf12997`** (38 commits / 5 thematic patches), base r12 applied tree `8a80535e…`, `LLAMA_QSA3_ENABLE=1` (compile-time) |
| delivery base | fresh worktree `/tmp/llama-r12-base` @ **`8568aaddb`** (block 15, r12 tree), built for this session (6 min with ccache) |
| pwilkin build | `~/pwilkin-llama-cpp/build-rocm/bin/llama-bench`, branch `strix-halo` @ **`f5daaa3cf`** |
| uniform model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93.16 GiB, 176.94 B params) |
| mixed model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (87.24 GiB) |
| pwilkin env | `archive/work/wip-archive/iq4nl-prefill/launcher-env.txt` (his `install.sh` "optimized" set, verbatim) |

Rules observed: **page cache warmed** (`cat` all shards to `/dev/null`) before every run; **no parallel benches**; `-p … -n 0 -r 2`; `rocprofv3 --output-format csv` (the ROCm 7.14 rocpd/SQLite writer aborts without it); `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib`; `HIP_VISIBLE_DEVICES=0`.

Pwilkin runs are `-dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct`; WIP runs are `-ngl 99 -fa 1`. Both `-ctk f16 -ctv f16`. WIP all-on = `GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1`.

> **Run-to-run variance is real: ~±2–3%** on this box (pwilkin pp8192/ub16384 measured 1320.9 and 1338.8 in the same session; WIP all-on measured 1191.7 and 1220.5). Treat single-digit differences as noise; the profile and the ablations are the reliable signals.

---

## 3. Throughput — the 1:1 comparison

### 3.1 Uniform IQ4_NL (pwilkin's checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | 733.6 | 737.1 |
| **base r12** | 16384 | — | 755.6 | **ctx failed** |
| **WIP all-on** | 2048 | 1181.5 | 1149.3 | 1129.2 |
| **WIP all-on** | 16384 | 1176.3 | 1220.5 | **ctx failed** |
| **pwilkin full env** | 2048 | 1233.2 | 1194.2 | 1187.0 |
| **pwilkin full env** | 16384 | 1233.0 | **1338.8** | **1399.3** |

* Base → WIP: **+56.7%** (pp8192/ub2048), **+61.5%** (pp8192/ub16384), **+53.2%** (pp16384/ub2048).
* WIP as % of pwilkin: **96.2%** (ub2048 pp8192), **91.2%** (ub16384 pp8192), **95.1%** (ub2048 pp16384). Pwilkin's best config (ub16384) is the one we cannot fully run.

### 3.2 Mixed UD-IQ4_XS (our checkpoint)

| build | ubatch | pp2048 | pp8192 | pp16384 |
|---|---:|---:|---:|---:|
| **base r12** | 2048 | — | ~707* | 710.5 |
| **base r12** | 16384 | — | 747.5 | — |
| **WIP all-on** | 2048 | 1137.9 | 1110.3 | 1082.2 |
| **WIP all-on** | 16384 | 1139.0 | 1192.3 | **ctx failed** |
| **pwilkin full env** | 2048 | 901.4 | 899.3 | 1046.9 |
| **pwilkin full env** | 16384 | 904.0 | 1070.1 | 1130.8 |

\* from `benchmarks/2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md` (same box).

On the mixed model the WIP is **+23–26% over pwilkin** at pp2048–8192/ub2048, because pwilkin's `mmb_supported_mmid`/`_glu` predicates reject anything but **IQ4_NL**, so the model's **IQ3_S gate/up experts (36% of bytes, ~2/3 of the MoE FLOPs)** and its Q8_0 dense tensors fall back to his MMQ path. Ours accelerates them. At pp16384/ub16384 pwilkin catches up (his 1130.8 vs our ub2048 1082.2).

**Conclusion:** the mixed-model comparison is *not* apples-to-apples in the direction the original "1.79x behind" implied. Each stack wins on the model its fast path was built for. The honest 1:1 is the uniform model, where the residual gap is ~4–9%.

---

## 4. The `-ub 16384` context-creation failure (delivery bug)

### 4.1 Symptom

```
llama_bench: error: failed to create context with model '…/Qwen3.8-Flash-Next-UD-IQ4_XS-…gguf'
```
`llama-bench` calls `llama_init_from_model` and gets `nullptr`. No underlying error is printed. It reproduces on **both** checkpoints, on the **base r12 build** as well as the WIP, and pwilkin's tree runs the same config fine (his 1399.3 on uniform).

### 4.2 Bisect matrix (uniform model, `-p 16384`)

| `n_batch` | `n_ubatch` | `n_ctx` | result |
|---:|---:|---:|---|
| 16384 | 16384 | 16384 | **ctx failed** |
| 16384 | 8192 | 16384 | 1192 t/s ✓ |
| 8192 | 16384 | 16384 | 1166 t/s ✓ (ubatch clamped to batch) |
| 16384 | 4096 | 16384 | 1162 t/s ✓ |
| 16384 | 2048 | 16384 | 1119 t/s ✓ |
| 16384 | 16384 | 8192 | 1221 t/s ✓ (`-p 8192`) |

**Trigger:** the triple **`n_batch == n_ubatch == n_ctx` = 16384** — i.e. a *full-batch, full-context prefill graph*.

### 4.3 Ruled out

* **Not OOM / not model VRAM:** still fails with `-ngl 50` (half the model unoffloaded). `rocm-smi` shows ~0.6 GiB VRAM used during the failing init (the model is mmap'd/GTT).
* **Not the WIP:** the **base r12** build fails identically → pre-existing delivery bug.
* **Not MMB/HC16:** fails with `GGML_CUDA_MMB=0 GGML_CUDA_MMB_HC16=0`.
* **Not QSA:** fails with `LLAMA_QSA_OFF=1`, `LLAMA_QSA_DENSE_SHORTCUT=1`, and `LLAMA_QSA_SPARSE_FA=0`.
* **`llama-cli` with the identical cparams works** (`n_ctx=16384 n_batch=16384 n_ubatch=16384`), because its init `graph_reserve` builds for `n_tokens = 64`; `llama-bench`'s init reserves the full-ubatch graph and dies there.

### 4.4 Hypothesis

The compute-graph reserve for a 16384-token × 16384-KV graph allocates one or more very large tensors (an `[n_kv, n_tokens]` mask/bias class tensor is 1 GiB as F32, 512 MiB as F16; the packed kq mask is `n_kv*n_tokens*2`), or hits an allocator/shape limit. The failure is silent, so the next step is a debug build that prints the reserve failure (or `gdb` on `llama_init_from_model`) — **not yet done**. Whatever it is, it is independent of the WIP and it costs us the `-ub 16384` regime where pwilkin is 9% ahead.

---

## 5. Kernel profile diff — uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=1

Both runs profiled with `rocprofv3 --kernel-trace`. WIP grand kernel sum **13591 ms**; pwilkin **11393 ms** (ratio 1.19). (Kernel-sum ratio > t/s ratio because the profiler captures the whole process; use the *family deltas*, not the absolute ratio.) Both traces confirmed the relevant fast paths were live: WIP `mmb_dense`/`mmb_routed_glu`/`mmb_routed` present, `qsa3_attn` present, `flash_attn_qsa` absent; pwilkin `mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `hc_gate_mix_kernel`, `gdn_conv_direct_kernel` all present.

### 5.1 Family table (Δ = WIP − pwilkin, ms)

| family | WIP ms | WIP % | pwilkin ms | PW % | **Δ(WIP−PW)** |
|---|---:|---:|---:|---:|---:|
| `mmb_dense` | 4008.6 | 29.5 | 3199.4 | 28.1 | **+809.2** |
| `rms_norm` (all) | 1059.9 | 7.8 | 505.3 | 4.4 | **+554.5** |
| MoE concat+reduction | 744.8 | 5.5 | 278.6 | 2.5 | **+466.1** |
| `ssm_conv_long_token` (conv) | 303.0 | 2.2 | 7.0 | 0.1 | **+296.0** |
| `qsa3_attn` | 813.8 | 6.0 | 618.4 | 5.4 | **+195.4** |
| elementwise | 706.9 | 5.2 | 555.2 | 4.9 | +151.8 |
| copy | 243.4 | 1.8 | 129.0 | 1.1 | +114.4 |
| indexer | 156.0 | 1.2 | 63.6 | 0.6 | +92.4 |
| mmq/mmvq | 120.5 | 0.9 | 55.1 | 0.5 | +65.4 |
| `mmb_tiny_m` (F32) | 54.7 | 0.4 | 0.0 | 0.0 | +54.7 |
| `mmb_routed` | 979.0 | 7.2 | 944.3 | 8.3 | +34.7 |
| `mmb_f32split` | 295.9 | 2.2 | 273.0 | 2.4 | +23.0 |
| HC (dsv4/hc_*) | 1103.8 | 8.1 | 962.0 | 8.4 | +141.8 |
| `mmb_routed_glu` | 1863.6 | 13.7 | 1974.7 | 17.3 | **−111.0** |
| `mmb_other`/`mmb_cvt` | 1.1 | 0.0 | 122.0 | 1.1 | **−120.8** |
| rocBLAS | 0.0 | 0.0 | 239.4 | 2.1 | **−239.4** |
| GDN | 938.3 | 6.9 | 1188.2 | 10.4 | **−250.0** |

(Watch the bucketing: pwilkin's `gdn_conv_direct_kernel` 250 ms landed in the GDN row, so the true conv comparison is our `ssm_conv_long_token_f32` 303 vs his `gdn_conv` 250 + `ple_conv` 7. And his GDN "1188" = `gated_delta_net_tiled` 936 + `gdn_conv_direct` 250; our pure recurrence is 938 — **parity**.)

### 5.2 The `mmb_dense` detail (raw instantiations)

| tile `WTYPE` | WIP ms / calls | pwilkin ms / calls | Δ |
|---|---:|---:|---:|
| `<128,256,64,64,0>` | 1627.3 / 168 | 1735.7 / 168 | −108 (we win) |
| `<128,128,32,64,0>` | 1060.0 / 594 | 785.6 / 498 | +274 / **+96 calls** |
| `<384,64,96,32,0>` (tall) | 1004.6 / **380** | 589.6 / **190** | +415 / **+190 calls** |
| misc | 316.7 | 228.5 | +88 |
| **total** | **4008.6 / 1266** | **3199.4 / 956** | **+809 / +310 calls** |

Our dense MMB launches **1266** GEMMs vs his **956** (+310), and the tall `384x64` tile runs **twice** as often (380 vs 190). A large part of this is structural, not tile tuning: pwilkin's **`hc_gate_mix_kernel`** (408 ms, 190 calls) fuses the HC gate GEMM + sigmoid + mix and *removes* ~190 dense GEMMs from his `mmb_dense`; we run those in `mmb_dense` and then do the mix separately in `dsv4_hc_pre/post`.

### 5.3 The RMS/HC detail

| | WIP | pwilkin |
|---|---|---|
| `rms_norm_f32<1024,true>` | **622.2 ms / 196 calls** | — |
| `rms_norm_f32<256,true>` | 288.8 / 168 | 0.3 / 24 |
| `rms_norm_f32<256,false>` | 148.8 / 144 | 166.4 / 144 |
| `rms_norm_f32<1024,false>` | — | 35.9 / 10 |
| **`rms_rows_f32<true>`** | — | **220.0 / 72** |
| **`rms_rows_f32<false>`** | — | **82.7 / 72** |
| `dsv4_hc_post_f32<false>` | **739.0 / 188** | — |
| `dsv4_hc_pre_f32<true,true,true>` | 364.7 / 190 | — |
| **`hc_combine_norm_f32_b256`** | **0** | **554.3 / 190** |
| **`hc_gate_mix_kernel<4>`** | **0** | **407.7 / 190** |

`rms_rows_f32` is pwilkin's fused **gated** RMS-norm (`LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS`); the 622 ms `rms_norm_f32<1024,true>` is our HC normalized stream. He folds the HC combine + norm into `hc_combine_norm_f32_b256`, and he has a whole `hc_gate_mix` kernel we have no analogue of.

### 5.4 The MoE detail

| | WIP | pwilkin |
|---|---|---|
| `concat_transposed_src1_dim0` | **375.3 / 74** | 0 |
| `moe_weighted_reduction_f32_vec4` | **369.5 / 96** | — |
| `moe_weighted_reduction_bf16_v4` | — | **213.3 / 94** |

Pwilkin's MoE epilogue reads **bf16** expert outputs (his `LLAMA_MMB_DOWN16` / `store_f32=0` routed-down) and avoids the `concat_transposed` materialisation entirely. We still materialise the concat and reduce in F32. (The WIP added non-temporal hints to these two kernels in session 14, but did not remove the concat or move to bf16 inputs.)

### 5.5 QSA

Same kernel name, same 24 calls, **813.8 vs 618.4 ms** — our `qsa3_attn_kernel` is 32% slower at identical work. Our `qsa3_rows`/`merge` are **faster** (64.6 vs 151.0). So the port's *sort/merge* is a win and the *attention body* is a regression, or his `qsa.cu` has an arch/tile difference the port did not carry.

---

## 6. What the missing pieces are worth — pwilkin's own ablations on this box

Run on the uniform model, `-b/-ub 16384`, `-p 8192`, r=2, source his `launcher-env.txt` and zero one family at a time. This is the cleanest "what is missing" price list, because it is the *same tree, same model, same box*.

| arm | pp8192 t/s | Δ vs full | % |
|---|---:|---:|---:|
| **FULL (baseline)** | 1320.9 | — | — |
| **NO `HC_*` (all 6)** | **1063.6** | **−257.3** | **−19.5%** |
| **NO `GDN_CONV`+`PLE_CONV`** | **1182.6** | **−138.3** | **−10.5%** |
| NO `NORM_GATED`+`NORM_ROWS` | 1282.1 | −38.8 | −2.9% |
| NO `IDX_RELU_SUM` | 1303.2 | −17.7 | −1.3% |
| NO `MMB_DOWN16` | 1321.8 | +0.9 | +0.1% (nil) |

The `HC_*` set zeroed is `HC_CN_SHAPE`, `HC_GATEMIX`, `HC_MIX_FUSE`, `HC_BLK16`, `HC_RES16`, `HC_PACK_DI`. The `GDN_CONV`/`PLE_CONV` ablation removes the *whole* direct-conv path (kernel + the concat/tail/reorder chain it replaces), so its 10.5% is more than the 257 ms of the two kernels themselves.

For cross-reference, the archived `iq4nl-prefill` Phase-1 ranking (pp16384, older build) measured: NO HC −289 (−21%), NO MMB −470 (−33%), NO QSA −696, NO CONV −62, NO NORM −30, NO IDX −10. The HC/NORM/IDX magnitudes reproduce; the fresh CONV number is larger because the `-ub 16384` single-shot prefill exposes the concat chain more.

---

## 7. WIP's own gate contributions on this model (for contrast)

Uniform IQ4_NL, `-b/-ub 16384`, pp8192, r=2:

| arm | pp8192 | Δ |
|---|---:|---:|
| WIP all-on | 1191.7 | — |
| WIP `HC16=0` | 1097.5 | HC16 bf16 producers **+8.6%** |
| WIP `MMB=0 HC16=0` | 856.2 | MMB **+28.2%** |
| WIP all-on `LLAMA_QSA_DENSE_SHORTCUT=1` | 1179.2 | always-QSA **+1.1%** |
| WIP all-on `LLAMA_QSA_OFF=1` | 1087.7 | QSA **+9.6%** |

So on the uniform model the WIP's MMB and bf16-producer work are doing exactly what they should. The gap is elsewhere.

---

## 8. Gap inventory (file + gate level, vs `~/pwilkin-llama-cpp @ f5daaa3cf`)

### 8.1 Ported / integrated (not the gap)

| pwilkin work | status |
|---|---|
| `mmb.cu` dequant→bf16 WMMA weight GEMM | ported **and generalized** to 9 weight types (`wip/mmb-general/patches/0001`) |
| `qsa.cu` qsa3 rows/merge/attn | ported as `fattn-qsa3.cu` (`patches/0002`) |
| bf16-producer marking (`mark_bf16_only`, `out_xn_bf16`) | ported (`patches/0004`) |
| F32 split / tiny-M | ported/ours (`patches/0003`) |
| non-temporal hints | **ours** (his tree has zero) |
| fused indexer top-k | **ours** (`patches/0005`; his tree uses `top_k_nary_search_cuda`) |
| `dsv4_hc_pre`/`hc_mix_reduce` | in delivery block 14 / WIP |

### 8.2 Missing or inactive

| # | pwilkin feature | his file / gate | our status | measured worth here |
|---|---|---|---|---|
| 1 | **HC gate-mix fusion** | `mmb.cu::hc_gate_mix_kernel`, `LLAMA_HC_GATEMIX` | **absent** | inside the −19.5% HC ablation |
| 2 | **HC combine+norm fusion (b256)** | `hc-cn.cu::hc_combine_norm_f32_b256` | delivery has `hyperconn.cu::hc_combine_norm_f32` (1024-thread) but it **never fires** (0 calls) | inside the −19.5% HC ablation |
| 3 | **depthwise conv1d, GDN + PLE** | `gdn-conv.cu`, `ple-conv.cu`; `LLAMA_GDN_CONV`/`LLAMA_PLE_CONV` | **absent** (we still build `concat`+transpose + `ssm_conv_long_token_f32`) | **−10.5%** |
| 4 | **gated RMS-norm** | `norm-gated.cu::rms_rows_f32`; `LLAMA_NORM_GATED`/`LLAMA_NORM_ROWS` | **absent** | −2.9% |
| 5 | **indexer relu-sum** | `idx-relu-sum.cu`; `LLAMA_IDX_RELU_SUM` | **absent** | −1.3% |
| 6 | **MoE bf16 epilogue / concat elimination** | `moe_weighted_reduction_bf16_v4` + `LLAMA_MMB_DOWN16` | F32 epilogue + concat still materialised | ~+466 ms kernel time |
| 7 | **`hc_combine_norm` b256 variant** | `hc-cn.cu` | only the 1024-block form exists | — |
| 8 | QSA graph-side options | `qwen4exp.cpp`: `QSA_WHOLE_ATTN`, `_BLOCK_SELECTION`, `_COMPACT_METADATA`, `_DIRECT_INDICES`, `_NO_DENSE_MASK`, `_QUERY_STRIP`, `_SCORE_BOUNDS`, `_SCORE_WMMA`, `_TOKEN_EMBD` | not ported; **partly redundant** with delivery block-14/15 derived-visibility / keys-only / fused indexer score (never audited 1:1) | small |
| 9 | HC knobs `HC_CN_SHAPE`/`HC_MIX_FUSE`/`HC_BLK16`/`HC_RES16`/`HC_PACK_DI` | HC variants | partial (we have the bf16 `xn` stream but not the variants) | inside −19.5% |
| 10 | depthwise conv2d | `conv2d-dw.cu` | absent | 0 on these models |
| 11 | MTP-side QSA | `LLAMA_MTP_QSA`, `_MTP_QSA_MIN_T`, `LLAMA_MTP_EH_FLATTEN` | absent | **decode/MTP, not prefill** |
| 12 | host/loader | `LLAMA_PLE_PREFETCH`, `LLAMA_LOAD_LOCALS`, `--lazy-mode on-direct` | our analogue | load-time only |

---

## 9. Root-cause notes and hypotheses

### 9.1 `hc_combine_norm` does not fire — highest-value item

* The delivery's `ggml_cuda_op_hc_combine_norm` lives in `ggml/src/ggml-cuda/hyperconn.cu`; the graph-optimizer match is at `ggml-cuda.cu:5514` and `:5677` (two sites) and is a long `ok_a…ok_f` shape/type/alias predicate.
* `ggml_cuda_hc_combine_norm_supported` would accept this model (`n_embd=2560 ≤ HC_CN_MAX_EMB=3072`, `warp_size=32`, `hc≤16`), so the **supported** gate is not the blocker.
* Empirically it is **0 calls** on the uniform model with **both** `GGML_CUDA_MMB_HC16=1` and `=0`, and the fallback is `dsv4_hc_pre_f32<true,true,true>` + `rms_norm_f32<1024,true>` + `dsv4_hc_post_f32<false>`.
* Therefore the failure is in the *pattern match* (`ok_*`, `ggml_can_fuse_subgraph_ext`, alias `overlap`), i.e. our `qwen4exp` graph no longer presents the shape the delivery's matcher expects, **or** the matcher was only ever validated in the beta and has been dormant since. pwilkin's equivalent fires 190× on the same model.
* **Action:** instrument the matcher (log which `ok_*` fails per layer), fix the pattern, or port pwilkin's `hc-cn.cu` + `hc_gate_mix_kernel` directly. Then add `LLAMA_HC_GATEMIX`-equivalent: fused gate GEMM + sigmoid + mix, which also removes ~190 `mmb_dense` launches and the `rms_norm_f32<1024>` pass.

### 9.2 Depthwise conv1d

Our path: `build_conv_state`/concat + `ggml_ssm_conv` → `ssm_conv_long_token_f32` (303 ms), plus the surrounding `concat_cont`/`cpy_scalar`/transpose traffic. Pwilkin's `gdn_conv_direct_kernel` reads `state`+`x` directly and writes the conv output (+ optional silu), 250 ms, and `ple_conv_kernel` 7 ms, with **no concat tensor**. Porting `gdn-conv.cu`/`ple-conv.cu` (and their graph-optimizer match hooks, `*_match_at_concat`/`*_match_at_conv`/`*_match_at_tap` + `*_write_tail`/`*_direct`) is self-contained and worth ~10.5% end-to-end on pwilkin's own measure.

### 9.3 `mmb_dense` +809 ms / +310 launches — mostly structural, not tile tuning

The WIP already closed the tile-tuning question (session 6: every tile/BN/VDR knob is a wash or worse; the kernel is at 54% of bf16 peak). The delta is that pwilkin **does fewer GEMMs**: `hc_gate_mix` absorbs ~190 gate GEMMs, and his tall tile runs 190× not 380×. Fixing item 1 should collapse most of this; the tall-tile 2x is worth a separate look (is the same A-panel dequantized/routed twice, or is our `mmb_tall` predicate applied to a tensor he handles with `<128,128>`?).

### 9.4 `qsa3_attn` +195 ms at identical launch counts

Our port is 32% slower on the body while our sort is faster. This is a kernel-shape/arch issue, not a graph issue. A/B the WIP `fattn-qsa3.cu` against pwilkin's `qsa.cu` on this exact model (the WIP's own qsa3 measurements were on the mixed model / gfx1201 for some arms). Possible causes: the pack layout (`qsa_pack_keys/values` graph vs his `src[6]/src[7]`), the `G=4`/`umask` handling, or the `ncols2`/`Q->ne[1]` selection.

### 9.5 The `-ub 16384` context bug

Pre-existing delivery (base r12 fails, WIP fails, pwilkin works), all WIP gates ruled out, not model VRAM. Blocks the pp16384/ub16384 point where pwilkin is strongest. Needs a debug print/gdb on the reserve, then a fix in the base graph/allocator. It also means our ub16384 numbers above are only valid up to pp8192.

---

## 10. Recommended next steps (prioritized)

| # | action | expected | effort |
|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) **or** port `hc-cn.cu`; add the `hc_gate_mix` fusion | large — the `HC_*` ablation is **−19.5%** | 2–4 days; pattern debug may be hours |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + their graph-optimizer matches | **−10.5%** (+ fewer concat/copy kernels) | 2–3 days |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation (unlocks `-ub 16384` and pp16384/ub16384) | access to pwilkin's best regime | 0.5–2 days |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9% / −1.3% | 1–2 days |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` (his `moe_weighted_reduction_bf16_v4`, `MMB_DOWN16`) | ~+466 ms kernel (~3–4%) | 1–2 days |
| 6 | Tune/port-align `qsa3_attn` body against `qsa.cu` | ~+195 ms (~1.5%) | 1–2 days |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 day |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 equivalents | small / likely redundant | 0.5 day |

Items 1+2 alone are ~30% of end-to-end prefill on pwilkin's own ablations — comfortably the difference between our 1221 and 1300+.

---

## 11. Caveats and data provenance

* **Variance.** Single runs on this box swing ±2–3%; the pwilkin baseline measured 1320.9 and 1338.8 in the same session. All family ablations share one session so their *relative* deltas are meaningful, but one or two points are within noise.
* **Kernel-sum ratio ≠ t/s ratio.** The profiles capture the whole process (including warm-up), so use the family deltas, not `13591/11393 = 1.19`.
* **Profiler caveat (`rocprofiler-register`, ROCm issue #10196).** Under `rocprofv3`, an env-gated path can read as *unset* (measured to flip `GGML_CUDA_QSA3` before it was made compile-time). I verified the fast paths were live from the kernel names in each trace (`mmb_*`, `qsa3_attn`, `hc_combine_norm_f32_b256`, `gdn_conv_direct_kernel` all present). The WIP's MMB/HC16 are still env-gated and could in principle flip; the family table is consistent with the un-profiled throughput, so it did not.
* **The `mmb_dense`/`rms`/HC kernels are *not* the same code in the two trees**, so their per-kernel times are not a pure A/B; the ablation (§6) is the authoritative price of the missing behaviour.
* **`-ub 16384` is required to reproduce pwilkin's 1339/1399**; our ub16384 numbers only exist up to pp8192 because of the context bug.
* **Not done:** gdb/debug of the context-creation failure; a 1:1 audit of the QSA graph-side flags; a from-scratch attempt to make `hc_combine_norm` fire; and any actual port work.

---

## Appendix A — raw throughput (t/s, `llama-bench -n 0 -r 2`)

```
Uniform IQ4_NL, base r12:
  ub2048  pp8192 733.62 ± ?      pp16384 737.06 ± 5.73
  ub16384 pp8192 755.59 ± 3.98   pp16384 FAIL
Uniform IQ4_NL, WIP all-on:
  ub2048  pp2048 1181.46 ± 3.31  pp8192 1149.32 ± 4.25  pp16384 1129.16 ± 1.19
  ub16384 pp2048 1176.28 ± 4.01  pp8192 1220.52 ± 1.53  pp16384 FAIL
Uniform IQ4_NL, pwilkin full env:
  ub2048  pp2048 1233.16 ± 39.79 pp8192 1194.24 ± 6.02  pp16384 1187.04 ± 0.49
  ub16384 pp2048 1233.03 ± 32.96 pp8192 1338.77 ± 37.03 pp16384 1399.34 ± 0.00

Mixed UD-IQ4_XS, base r12:
  ub16384 pp8192 747.48 ± 9.26   ; ub2048 pp16384 710.48 ± 7.75
Mixed UD-IQ4_XS, WIP all-on:
  ub2048  pp2048 1137.89 ± 1.59  pp8192 1110.32 ± 1.99  pp16384 1082.21 ± 0.35
  ub16384 pp2048 1139.03 ± 5.17  pp8192 1192.31 ± 1.11  pp16384 FAIL
Mixed UD-IQ4_XS, pwilkin full env:
  ub2048  pp2048 901.36 ± 96.74  pp8192 899.30 ± 61.23  pp16384 1046.86 ± 1.18
  ub16384 pp2048 904.04 ± 91.36  pp8192 1070.10 ± 29.01  pp16384 1130.80 ± 5.31
```

## Appendix B — pwilkin family ablations (uniform, `-b/-ub 16384`, pp8192, r=2)

```
FULL                     1320.88 ± 37.97
NO NORM_GATED+ROWS       1282.10 ± 31.20   -38.78  -2.94%
NO GDN_CONV+PLE_CONV     1182.58 ± 31.49   -138.30 -10.47%
NO IDX_RELU_SUM          1303.17 ± 41.83   -17.71  -1.34%
NO MMB_DOWN16            1321.76 ± 33.37   +0.88   +0.07%
NO HC_* (all 6)          1063.62 ± 30.46   -257.26 -19.48%
```

## Appendix C — WIP gate contributions (uniform, `-b/-ub 16384`, pp8192, r=2)

```
WIP all-on                1191.66 ± 4.29
WIP HC16=0                1097.46 ± 3.47
WIP MMB=0 HC16=0           856.21 ± 11.06
WIP all-on DENSE_SHORTCUT=1 1179.19 ± 6.23
WIP all-on QSA_OFF=1       1087.71 ± 25.75
```

## Appendix D — exact commands

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MM=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
W=/home/stew675/llama-wip-mmb/build-rocm/bin/llama-bench
BASE=/tmp/llama-r12-base/build-rocm/bin/llama-bench
PW=/home/stew675/pwilkin-llama-cpp/build-rocm/bin/llama-bench
ENV=/home/stew675/llama-cpp-rdna-boosts/archive/work/wip-archive/iq4nl-prefill/launcher-env.txt

# warm the page cache
for f in /llm/models/Qwen3.8/Flash-Next/IQ4_NL/*-0000*.gguf; do dd if=$f of=/dev/null bs=4M; done

# WIP all-on
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2

# pwilkin full env
( set -a; . $ENV; set +a; \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 0 -r 2 )

# pwilkin ablation (e.g. HC)
( set -a; . $ENV; set +a; \
  LLAMA_HC_CN_SHAPE=0 LLAMA_HC_GATEMIX=0 LLAMA_HC_MIX_FUSE=0 LLAMA_HC_BLK16=0 LLAMA_HC_RES16=0 LLAMA_HC_PACK_DI=0 \
  $PW -m "$MU" -dev ROCm0 -ngl 999 -fa on -lm none -lzm on-direct -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 8192 -n 0 -r 2 )

# profile (csv; the rocpd writer aborts on this ROCm)
rm -rf /tmp/prof && mkdir -p /tmp/prof
GGML_CUDA_MMB=1 GGML_CUDA_MMB_HC16=1 /opt/rocm-7.14-gfx1151/bin/rocprofv3 \
  --kernel-trace -f csv --output-format csv -d /tmp/prof -o k -- \
  $W -m "$MU" -ngl 99 -fa 1 -ctk f16 -ctv f16 -b 16384 -ub 16384 -p 8192 -n 0 -r 1

# delivery base (built this session)
cd ~/llama.cpp && git worktree add --detach /tmp/llama-r12-base 8568aaddb
# then configure/build as in wip/mmb-general/HANDOVER.md §3
```

---

## 12. MTP qualification (2026-09-21): adaptive (ours) vs fixed (pwilkin's)

This is the MTP half of the gap analysis, added because pwilkin's newer commits are decode/MTP-heavy
and it is easy to read his MTP t/s as a gap. It is **not** the same axis as our advantage, and the
qualification below is what the 2026-09-21 plan asks for before either side is claimed.

### 12.0 Result (measured 2026-09-21 — see [`2026-09-21-mtp-qualification.md`](2026-09-21-mtp-qualification.md))

Two findings, and one correction to the premise:

* **Our plain decode is ahead** of his on qwen4exp IQ4_NL (code 32.4 vs 31.1, prose 31.8 vs 29.9,
  recall 32.4 vs 31.9 t/s).  Absolute MTP t/s therefore flatters his stack; the fair metric is the
  **speedup over each tree's own plain decode**.
* **At fixed depth the MTP speedup is at parity** — ours `n3` **1.90x / 1.78x / 2.05x** vs his fixed
  **1.91x / 1.79x / 2.02x** (code / prose / recall).  He did **not** adopt our controller
  (`common/speculative-adaptive.h` is absent from his tree) and he is not ahead.
* **Our adaptive controller is mixed on qwen4exp** — the opposite of the 27B dense record.  It wins
  **recall** (2.34–2.40x) but over-drafts code and prose at `n_max 9..12` (code `adaptive 12`
  per-position acceptance falls 0.94 → 0.45 → 0.22 → 0.07); `adaptive 7` already beats `n3` on code
  (63.1 vs 61.4 t/s).  So the qwen4exp adaptive **ceiling is a tuning item**, not a structural gap.
* **The one real MTP gap is correctness/compat, not speed: `nextn_shared_target_tensors`.**  The sidecar
  pwilkin's IQ4_NL model ships is a *shared* MTP head; our build fails every draft position past the
  first on an M-RoPE `X < Y` check, so the head cannot be used at all.  The comparison above used the
  non-shared `Q4_K_M` sidecar, which both trees run clean.

### 12.1 Structural standing

| | pwilkin (`b0f31f587`) | ours (r12 + `beta/mmb-general`) |
|---|---|---|
| spec type | upstream **`draft-mtp` only** | `draft-mtp` **and** `draft-mtp-adaptive` |
| depth | **fixed** `--spec-draft-n-max` (default 3), capped at `n_mtp_layers` when chaining heads | adaptive controller picks the depth each round; `--spec-draft-n-start`, `n_min_adaptive`, clamp at 15 |
| cross-round feedback | none — only upstream's **within-round** `p_min`/`n_min` early stop | credit-bucket `common_speculative_adaptive` (delta = `n_accepted - depth`; full accept credits `max(1, n_accepted-1)`; surplus/deficit carried; `drop_pressure = max(60, 10*depth)`, `climb_budget = 20 + 6*(depth-1)`, cold start `cap-3`) |
| per-step cost | **new** sparse selected-cell decode (`qsa-decode.cuh` SIMT + `qsa-decode-wmma.cuh`) + incremental indexer key state (`d67d58836`): serial d40000 **25.85 → 28.82 t/s**, MTP 40680 **31.17 → 35.57** / **32.69 → 39.10** | our own QSA-sparse-FA decode + derived-block-vector cache; no dedicated selected-cell decode kernel for this model |

**The two optimise different things and compose.** His `d67d58836` lowers the cost of each verify/draft
step; our block-01 controller decides *how deep* to draft. Median accepted length is the quantity the
controller moves and his kernels do not.

Delivery evidence for the controller (all at `-n 3000`, `benchmarks/mtp-adaptive-methodology.md` rule 0):
+13 % prose, +28 % code, +61 % recall vs fixed `n3` (`benchmarks/2026-09-13-adaptive-mtp-4-axis-n12.md`),
and +72 % recall for `draft-mtp-adaptive` + `ngram-mod` (`benchmarks/2026-09-17-mtp-ngram-combo.md`).
At `-n 256` the same controller *lost* to fixed `n3` (-5 % code) — the length rule matters here as much
as anywhere.

### 12.2 Hypothesis and falsification

**Hypothesis:** on the same model and workload our adaptive depth beats our fixed `n3` (and his fixed
`n3`) by a margin larger than his per-step decode gains, because the depth policy is the term the
per-step kernels do not touch.

**Falsifiers:**
- if `draft-mtp-adaptive` ≤ `draft-mtp --spec-draft-n-max 3` on the four axes at `-n 3000` on
  qwen4exp, the controller does **not** transfer to this model (a real finding — it would need a
  model-specific investigation);
- if his *absolute* MTP t/s exceeds ours by more than his per-step kernel advantage explains
  (measured as our fixed-`n3` vs his reported fixed-`draft-mtp`), our depth policy is not the whole
  story and the decode kernels are the gap after all.

### 12.3 Protocol — single-build A/B (the portable claim)

On our beta build (`~/llama.cpp/build-rocm`, r12 + 12 patches), **pwilkin's own uniform IQ4_NL model**,
gfx1151, `-ctk f16 -ctv f16`, seed 42 / temp 0, `-n 3000` (reasoning pinned: `on` for R, `off` for
P/C/K), per `benchmarks/mtp-adaptive-methodology.md`. Four arms per axis:

```sh
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
BIN=~/llama.cpp/build-rocm/bin/llama-cli
for axis in P C R K; do for arm in none fixed3 adaptive adaptive12; do
  case $axis in R) REA=on;; *) REA=off;; esac
  case $arm in none) SPEC="--spec-type none";;
                fixed3) SPEC="--spec-type draft-mtp --spec-draft-n-max 3";;
                adaptive) SPEC="--spec-type draft-mtp-adaptive";;
                adaptive12) SPEC="--spec-type draft-mtp-adaptive --spec-draft-n-max 12";; esac
  # <prompt for $axis>, --log-verbosity 4 -> capture Generation t/s + acceptance + acc per pos
  $BIN -m "$MU" -ngl 99 -fa 1 --reasoning $REA $SPEC \
    --seed 42 --temp 0 --predict 3000 --single-turn --no-display-prompt \
    -p "$(cat prompts/<axis-prompt>.txt)" 2>&1 | tee /tmp/mtpq-${axis}-${arm}.log
done; done
```

Also run the **40680-token prompt** (his long case) for A1/A2/A3 only; record Generation t/s and mean
accepted length. `-n` is recorded with every number (rule 0).

### 12.4 Cross-build comparison — separate the axes

Pwilkin's 35.83 / 39.01 t/s include his per-step kernels, so our absolute numbers are expected to be
lower. Decompose, do not compare totals:

* **depth-policy delta** = `adaptive` − `fixed3` on *our* build (his kernels absent from both arms);
* **per-step-cost delta** = our `fixed3` vs his reported fixed-`draft-mtp` (same model, same depth) —
  this prices the sparse-decode + incremental-indexer gap in milliseconds per step;
* only the residual neither term explains is a genuine MTP gap.

### 12.5 Conclusion and the plan it implies

Measured, not predicted: our fixed-depth MTP is at parity with his, our plain decode is ahead, and our
adaptive controller is a clear win on recall and a tuning problem on code/prose for this model.  The
plan is therefore:

* **do not** treat his MTP as a speed gap;
* fold pwilkin's per-step decode path in as **item 9** (sparse selected-cell decode + incremental
  indexer) — that is the term his absolute numbers get for free;
* add **`nextn_shared_target_tensors` support** as a correctness/compat item (it gates his own model's
  MTP head);
* park the qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) until the MTP phase, per the maintainer's
  priority sequence.

### 12.6 What NOT to conclude

* **Do not** read his 39.10 t/s as "our adaptive MTP is 39 t/s behind" — he is measuring a fixed-depth
  stack plus his decode kernels on a different tree.
* **Do not** compare absolute MTP t/s without each build's own plain decode next to it.
* **Do not** compare at `-n 256`: our controller's warm-up transient inverts the ranking there.
* **Do not** use `none == draft-mtp` byte purity above `n_max 7` as the MTP gate; use acceptance and
  MTP-vs-plain throughput (rule 4).

---

## 13. Revised action plan — phased (2026-09-21)

**Maintainer's priority sequence (2026-09-21): recall speed + correctness → decode speed + correctness
→ MTP tuning + correctness.**  The 2026-09-20 §10 order was: (1) HC combine_norm/gate-mix, (2) depthwise
conv1d, (3) `-ub 16384` context bug, (4) norm-gated + idx-relu-sum, (5) MoE bf16 epilogue, (6) qsa3_attn
body, (7) tall tile, (8) QSA graph flags; items 1–9 survive, regrouped below.

### Phase 1 — recall (long-context prefill/attention) speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 1 | Make `hc_combine_norm` fire (debug the matcher) and **wire the existing `hc_gate_mix_kernel`** | large — `HC_*` ablation **−19.5 %** | 2–4 d | kernel already in beta; this is the call site + matcher, not a port |
| 2 | Port `gdn-conv.cu` + `ple-conv.cu` + matches (now incl. **F32 PLE**) | **−10.5 %** | 2–3 d | pwilkin's `40a9f4d01` made the PLE half F32-aware |
| 3 | Fix the `n_batch==n_ubatch==n_ctx` context creation | unlocks `-ub 16384` | 0.5–2 d | pre-existing delivery bug |
| 3.5 | **Port the three correctness fixes** (`40c0b9c38`, `b0f31f587`, `14fff4f97`) | prevents long-session corruption | 0.5–1 d | cheap; includes the QSA decode non-determinism fix |
| 4 | Port `norm-gated.cu` (`rms_rows`) + `idx-relu-sum.cu` | −2.9 % / −1.3 % | 1–2 d | |
| 5 | MoE: bf16 epilogue + drop `concat_transposed` | ~+466 ms kernel (~3–4 %) | 1–2 d | beta has `MMB_DOWN16` gated off; wire it + the bf16 reduction |
| 6 | Tune/port-align `qsa3_attn` body vs `qsa.cu` | ~+195 ms (~1.5 %) | 1–2 d | re-profile `b0f31f587` first |
| 7 | Investigate the tall `384x64` 2× launch count | unknown (part of +809) | 0.5–1 d | |
| 8 | Audit the 9 QSA graph-side flags vs block-14/15 | small / likely redundant | 0.5 d | |

Items 1+2 remain ~30 % of end-to-end prefill on pwilkin's own ablations.

### Phase 2 — decode speed + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 9 | **Port sparse QSA decode + incremental indexer state (`d67d58836`)** | **+11–20 % MTP/decode** | 2–4 d | this is the per-step term his absolute numbers get for free; audit vs our `GGML_CUDA_QSA_INDEXER_CACHE` (default on) first |
| 10 | MMB quant coverage: Q4_0/Q4_1/Q5_0/Q2_K/IQ1/IQ2/MXFP4/NVFP4 | completeness | 1–2 d | low priority for the delivery's models |

Our **plain decode is already ahead** of his (+2–6 % on qwen4exp, §12), so item 9 is a *hold/repay*
item, not a catch-up.

### Phase 3 — MTP tuning + correctness

| # | action | expected | effort | note |
|---|---|---|---|---|
| 12 | **`nextn_shared_target_tensors` support** | gates pwilkin's own IQ4_NL MTP head | 1–2 d | our build fails every draft position past the first (M-RoPE `X < Y`); see §12.0 |
| 11 | qwen4exp adaptive **ceiling sweep** (3/5/7/9/12) + a long-prompt run | recovers the recall win without over-drafting code/prose | 0.5–1 d | `adaptive 7` already beats `n3` on code; the 27B result does not transfer at `n_max 12` |

Item 11/12 are parked until Phase 1–2 land, per the maintainer's sequence.

