# HANDOVER — IQ4_NL prefill: port pwilkin's generic weight-GEMM fast path into rdna-boosts

**Date:** 2026-09-12
**Box:** Strix Halo (gfx1151), ROCm 7.14 at `/opt/rocm-7.14-gfx1151`, 124 GB unified
**Anchoring TODO item:** **NEW — propose item 18** (`qwen4exp weight-IQ4_NL prefill fast path`).
**Distinct from item 3** (`qwen4exp iq4_nl prefill delta`), which is the `iq4_nl` **KV-cache** axis and
stays exactly as recorded — see §1.
**Success criterion:** **> 1100 t/s** prefill at pp16384 on pwilkin's uniform-IQ4_NL model, up from the
current **787 t/s** (rdna-boosts) / **1409 t/s** (pwilkin's stack)
**Companion files:** `PROMPT.md` (the paste-ready session prompt), `launcher-env.txt` (pwilkin's full env),
`mmb-port.patch` (the ported module + hooks, applies to canonical tip `be0d23d57`)

> **STATUS: PARKED as WIP (2026-09-12).** The `mmb` port was built and measured (`GGML_CUDA_MMB=1`,
> default **0**): **+18.4 %** on the uniform-IQ4_NL model (787 → 934 t/s @pp16384), **+12.0 %** on our
> mixed UD-IQ4_XS model (755 → 846). It is **NOT in the delivery set** and must not be defaulted on.
> Parked deliberately — see §12 for the design-objective rationale and the resume checklist.

---

## 0. TL;DR

On the *same box and session*, on pwilkin's **uniform-IQ4_NL** checkpoint:

| build | pp8192 | pp16384 | tg128 |
|---|---|---|---|
| pwilkin `strix-halo` + **full stack env** | 1346.9 | **1408.8** | 30.36 |
| **rdna-boosts** (16-patch delivery) | 776.8 | **787.2** | **31.26** |

We are **1.79× behind on prefill and ahead on decode**. The gap is a single missing class of kernel:
a **quantized-weight → bf16 → WMMA GEMM** (pwilkin's `mmb`), plus the fusion gates around it. Our
generic MMQ path is at ~787; pwilkin's fallback (his `mmb` disabled arm) is ~828 — i.e. our MMQ ≈ his
fallback, and the fast path is the win. Porting the *generic* part of it is the plan.

Target after the port: **~1.4× → ~1100+ t/s**, which is the stated success bar.

---

## 1. This is a NEW item — do not fold it into item 3

Item 3 in `TODO.md` is **not** this work. It records a different axis:

1. **KV-cache type `iq4_nl` (TODO item 3 — unchanged, separate).**
   `iq4_nl` as a *K/V cache type* costs ~8–12 % prefill vs f16/`q4_0`/`q4_1` on the reference
   `-sm tensor` (`iq4_nl` 2303.1/2421.0 vs f16 2615.5/2736.2 at pp8192; pp32768 1992.1 vs 2434.5).
   Profiled to be **host/launch-side**, not the QSA kernel (`rocprofv3`: same kernel instantiations,
   identical executed graph, lower traced kernel sum, but +95 ms/token host CPU). Item 3 stays exactly
   as recorded; this handover does not touch it.
2. **Weight-quantization `IQ4_NL` (NEW item — propose #18; the subject of this handover).**
   The *weight tensor type* determines whether the engine can run a bf16-WMMA dequant GEMM at prefill.
   pwilkin's fast path is written for **IQ4_NL weights**; our checkpoint's gate/up experts are **IQ3_S**,
   so it cannot use that path. This is the 787 → 1100+ work.

**Action for the next session:** add the new item to `TODO.md` Active (proposed text below) and leave
item 3 alone. They are thematically adjacent (both say "IQ4_NL" and "prefill") but mechanically
unrelated.

Proposed `TODO.md` Active entry:

```
### 18. qwen4exp **weight**-IQ4_NL prefill fast path (>1100 t/s target)
- Our prefill on pwilkin's uniform-IQ4_NL model is **787 t/s** at pp16384 vs his stack's **1409**
  (1.79x); we are ahead on decode (31.26 vs 30.36).  The gap is a missing **quantized-weight -> bf16 ->
  WMMA** GEMM (his `mmb`), not the model and not the KV cache (that is item 3).  Success = **>1100**
  t/s on his model.  Port the *generic* parts only (weight GEMM first, fusions second); a follow-on
  IQ3_S expert variant is what would help *our* checkpoint.  Handover:
  `wip/iq4nl-prefill/HANDOVER-2026-09-12-iq4nl-weight-gemm-port.md`.
```

---

## 2. The full measured A/B (2026-09-12, this box)

`llama-bench`, `-ngl 99`/`999`, `-fa on`, `-ctk/-ctv f16`, `-b 16384 -ub 16384`, `-d 0`, `-r 2`, single
gfx1151 device.

**pwilkin's uniform-IQ4_NL** (93.16 GiB, 770 IQ4_NL tensors, card:
`huggingface.co/ilintar/qwen3.8-flash-next-gguf-strix-halo`):

| build | pp2048 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|
| pwilkin + full env (§4) | — | 1346.9 ± 36.3 | 1408.8 ± 0.2 | 30.36 |
| pwilkin, `LLAMA_MMB=1` only | — | 760.8 | 741.3 | — |
| pwilkin, defaults (no env) | — | 679.1 | 670.1 | 23.64 |
| **rdna-boosts bu16** | 755.4 | 776.8 | **787.2** | **31.26** |

**our mixed UD-IQ4_XS** (87.24 GiB: IQ4_NL 45.7 down-experts + PLE table, IQ3_S 31.6 gate/up, Q8_0 8.3,
Q6_K 0.5):

| build | pp512 | pp2048 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|---|
| pwilkin + full env | — | 698.3 | 791.5 | 722.6 | 23.53 |
| pwilkin defaults | — | — | 679.1 | 670.1 | 23.64 |
| rdna-boosts, `-ub 2048` | 642.4 | 772.8 | 729.6 | 732.4 | 25.53 |
| rdna-boosts, `-ub 16384` | — | 738.0 | 756.1 | 656.6 ±111 | 25.5 |

**Reading it:**
- On **IQ4_NL** pwilkin is 1.79× ahead on prefill; we are ahead on decode.
- On **our mixed model** the two are at parity (~730–790) because his fast path needs IQ4_NL for the
  weights it accelerates, and our **gate/up experts are IQ3_S**.
- **Ubatch is not a lever for us** (a wash from `-ub 2048` to 16384; 24576 needs the route-bounded
  `mmid` and only matters once the GEMM path is in).
- The page's "1,160 t/s" is the joint optimum for *(uniform-IQ4_NL) × (his kernels) × (tuned host)*.
  With the full env on his model we measured **1409** on this box.

---

## 3. What pwilkin's stack is, and what is actually general

Source: `github.com/pwilkin/llama.cpp`, branch **`strix-halo`** (we have it at
`/home/stew675/ll25/pwilkin`, tip `f5daaa3`, built with `/home/stew675/bin/build-llama-rocm-714`).

Key files (all CUDA/HIP backend, i.e. portable in shape, not model-specific):

| file | what it is |
|---|---|
| `ggml/src/ggml-cuda/mmb.cu` (`mmb.cuh`) | **The prize.** A matrix-matrix path that dequantizes quantized weights to **bf16 in LDS/shared memory** and runs the GEMM on the **bf16 WMMA** units. `mmb_dq_row36` stages two IQ4_NL blocks (36 B) → 64 bf16. Also a persistent per-tensor bf16 **shadow** map and the MoE (`mmid`) variant. Gated by `LLAMA_MMB` etc. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | The `ggml_backend_cuda_graph_optimize` hooks: `mmb_mark_bf16_only` (drop an f32 copy when every consumer reads bf16) and `mmb_shadow_prepare`. |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | The **tiled** GDN (state in registers across a `TOKEN_TILE`, `GDN_DPP_REDUCE` via `__builtin_amdgcn_permlanex16` + DPP). The single biggest win in his journey (2.37×) — but our chunked GDN already measures at parity with it, so **lower priority**. |
| `ggml/src/ggml-cuda/lightning-indexer.cu`, `norm-gated.cu` | Fused indexer scoring and gated rms-norm. |
| `src/llama-lazy-reader.{h,cpp}` | Direct `pread()` thread-pool reader (`--lazy-mode on-direct`). We already have an analogue (`src/llama-lazy-reader.cpp`, `--lazy-buffer-size`). |

**Why bf16 WMMA wins at prefill** (his own note): MMQ's integer path leaves the f16/bf16 WMMA units
idle; the f32 vector roof on this part is **29.7 TFLOPS vs 59 for bf16 WMMA**. At prefill batch sizes
the GEMMs are compute-bound, so the roof is everything.

---

## 4. The full stack env (verbatim from his `install.sh` launcher, §"optimized")

This is the key artefact I was missing on the first A/B. Without it his build is at ~670; with it,
1409. (Full text also in `launcher-env.txt` beside this file.)

```
LLAMA_MMB=1            LLAMA_MMB_MIN_T=512     LLAMA_MMB_BF16W=1   LLAMA_MMB_GLU=1
LLAMA_MMB_TALL=2       LLAMA_MMB_CACHE=4       LLAMA_MMB_F32SPLIT=2 LLAMA_MMB_HC16=2
LLAMA_MMB_SHADOW=2     LLAMA_MMB_DOWN16=1
LLAMA_HC_CN_SHAPE=1    LLAMA_HC_GATEMIX=1      LLAMA_HC_MIX_FUSE=1  LLAMA_HC_BLK16=1
LLAMA_HC_RES16=1       LLAMA_HC_PACK_DI=1
LLAMA_NORM_GATED=1     LLAMA_NORM_ROWS=1       LLAMA_IDX_RELU_SUM=1
LLAMA_PLE_CONV=1       LLAMA_GDN_CONV=1
LLAMA_QSA_SPARSE=1     LLAMA_QSA_WHOLE_ATTN=1  LLAMA_QSA_BLOCK_SELECTION=1
LLAMA_QSA_COMPACT_METADATA=1  LLAMA_QSA_DENSE_SHORTCUT=1  LLAMA_QSA_DIRECT_INDICES=1
LLAMA_QSA_PACK_KEYS=1  LLAMA_QSA_PACK_VALUES=1 LLAMA_QSA_QUERY_STRIP=512
LLAMA_QSA_SCORE_BOUNDS=1  LLAMA_QSA_NO_DENSE_MASK=1  LLAMA_QSA_FA_V3=1
LLAMA_QSA_FUSE_EXPAND=1  LLAMA_MTP_QSA=1  LLAMA_MTP_QSA_MIN_T=128
# plus the launcher's command line:
#   -m <model> -dev ROCm0 -ngl 999 -fa on -fit off --load-mode none --lazy-mode on-direct
#   -ctk f16 -ctv f16 -c 65536 -b 16384 -ub 16384 --parallel 1
# and (from the generic wrapper): GGML_HIP_ENABLE_UNIFIED_MEMORY=1, HSA_OVERRIDE_GFX_VERSION=11.5.1
```

Note `LLAMA_MMB_SHADOW=2` = **Q6_K shadows only** (`mmb_shadow_mode()==2` disables the IQ4_NL shadow);
the IQ4_NL speed therefore comes from the **runtime dequant GEMM** (`LLAMA_MMB=1`), not the persistent
shadow. Do not enable mode 1 expecting it to be the mechanism (I did on the first pass — the shadow cap
is 6 GiB and it shadows the wrong set).

His CMake flags differ from ours (`install.sh`): `-DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
-DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_RCCL=OFF -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF`. Our A/B
built **his branch with our script**, so the 1.79× is code, not flags — but a cheap Phase-0 check is to
see whether `GGML_HIP_MMQ_MFMA`/`NO_VMM` move *our* build at all.

---

## 5. Scope: what to port, what not to

**Port (generic, fits our framework):**
1. **`mmb`'s bf16-WMMA dequant GEMM for IQ4_NL (and Q6_K)** — the headline. Prefill-only
   (`T >= MMB_MIN_T`, his default 512), so it sits entirely *above* the decode/verify band and cannot
   touch purity. This is the item that should get us to >1100 on his model.
2. **The fusion gates that are generic and cheap to express as our existing fused-op mechanism:**
   `LLAMA_NORM_GATED`/`NORM_ROWS`, `LLAMA_IDX_RELU_SUM`, `LLAMA_PLE_CONV`/`GDN_CONV`, the `HC_*` set.
   Several overlap with what we already have (HC decode-band, fused MoE); check for duplicates first.
3. **The IQ3_S variant** of (1) — *not needed for the stated target*, but it is what our own checkpoint
   would benefit from (our gate/up experts are IQ3_S, 2/3 of MoE FLOPs; neither engine accelerates
   them). Flag it as the follow-on that makes the win real on our own model.

**Do NOT port:**
- His whole env-gate framework / flag names. Add our own gates in our idiom (env-gated, default-off
  until validated, bit-purity-checked) as *block amendments*, not a parallel config surface.
- His tiled GDN (we are already at parity), his lazy reader (we have one), the retained-PM4 runtime
  (separate project, `pwilkin/rocm-systems`).
- Anything `qwen4exp`-model-specific that duplicates block 14.

**Method that worked for him and should work for us:** rank the gates with the "disabled arm on the
final binary" technique (one build, one variable) before porting, so effort follows measured worth.

---

## 6. Repro (exact)

**Builds**
- rdna-boosts current delivery: `/home/stew675/ll25/bu16` (fresh worktree at `9113cc188` +
  `scripts/apply-all.sh`, strict 16/16, tree `c2e284c2acc032238ef85cb35d427c1598ed0949`; llama-bench
  reports build `be0d23d57 (10882)`). Rebuild: `cd /home/stew675/ll25/bu16 && bash ~/bin/build-llama-rocm-714`.
- pwilkin: `/home/stew675/ll25/pwilkin` (`strix-halo` `f5daaa3`), same build script.

**Models**
- pwilkin IQ4_NL: `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`
  (+ `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`). ~93 GiB, 9 shards, uniform IQ4_NL. Point llama-bench
  at shard `00001`.
- ours UD-IQ4_XS: `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`.

**Commands**
```sh
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf

# rdna-boosts baseline (expect ~787 @ pp16384)
/home/stew675/ll25/bu16/build-rocm/bin/llama-bench -m "$M" -ngl 99 -fa on -ctk f16 -ctv f16 \
  -b 16384 -ub 16384 -p 2048,8192,16384 -n 128 -d 0 -r 2

# pwilkin reference (expect ~1409 @ pp16384) — source launcher-env.txt first
( set -a; . "$(dirname "$0")/launcher-env.txt"; set +a; \
  /home/stew675/ll25/pwilkin/build-rocm/bin/llama-bench -m "$M" -dev ROCm0 -ngl 999 -fa on \
    -lm none -lzm on-direct -ctk f16 -ctv f16 -b 16384 -ub 16384 -p 2048,8192,16384 -n 128 -d 0 -r 2 )
```
`llama-bench` has **no `-fit`** flag (it is `-fitt`); `-lm none -lzm on-direct` are pwilkin's reader
flags and our fork does not accept them — for a matched-config row run his build with `-lzm off` too.
No parallel benches; the box drifts (§ the item-3 note in `TODO.md`).

---

## 7. Suggested work order

- **Phase 0 (cheap, ~1 h).** Re-run the §6 A/B to confirm 787 / 1409. Then two 5-minute checks:
  (a) rebuild our `bu16` with `-DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON` and see if it moves;
  (b) re-run his build with our script vs his CMake flags. This rules the build configuration in/out
  before any kernel work.
- **Phase 1 (ranking).** On his build+model, disable his gates one at a time (his code already has the
  disabled arms) and record the worth. Expected order from his journey: the GEMM path first (~1.4×),
  then the fusion set (1.03–1.19× each), the GDN far behind. Confirm here rather than assume.
- **Phase 2 (the port).** Implement an `mmb`-style **bf16-WMMA dequant GEMM for IQ4_NL/Q6_K** in our
  `ggml-cuda` MMQ dispatch, gated by a new env (default off), prefill-only (`T >= 512`). A/B on his
  model until **pp16384 > 1100**; verify same-seed coherence and the op suites.
- **Phase 3 (optional).** The IQ3_S expert variant — the version that helps *our* checkpoint.
- **Phase 4 (landing).** Fold into the owning block (MMQ dispatch → block 08 or a new amendment;
  weight-layout changes → block 10/13 as appropriate), regenerate `patches/` via
  `scripts/make-patches.sh`, update its default tip, regenerate `rdna-boosts-all.patch`, verify strict
  **16/16** `git am` at `9113cc188` (0 whitespace, applied tree == canonical), re-cut the beta block-15
  patch if the base moves, update the docs, push to the delivery repo's own origin.

---

## 8. Gates & purity (mandatory)

- **Our purity rulebook:** `GREEDY-PURITY.md` and `AGENTS.md` (read them). The new GEMM path must be
  **prefill-only and above the decode/verify band** — that is why the `T >= 512` threshold matters:
  `W = 1..8` must be bit-identical with the new path on and off, and the MTP acceptance gate
  (`benchmarks/mtp-adaptive-methodology.md`) must hold.
- Same-seed coherence on our models (27B Q8_0 and qwen4exp), byte-identical with the gate off.
- `test-backend-ops -o FLASH_ATTN_QSA` (22/22), `-o GATED_DELTA_NET` (46/46), `-o FLASH_ATTN_EXT`
  (5935/5935) all still pass.
- Cross-arch consistency: keep gfx1100/gfx1201 compiling and un-regressed (the AMD-WMMA selection in
  `mmq.cu` is arch-gated; extend it the same way).
- Follow the AGENTS.md scope policy (RDNA first; non-AMD backends consistent, not validated).

## 9. Landing rules (delivery)

`AGENTS.md` is authoritative. Summary: amend the owning block and regenerate from a **canonical fork
rebuilt at `9113cc188`** (never from the drifted `~/llama.cpp`), strict `git am`, applied tree ==
canonical, update `patches/README.md` + `WORKLOG.md` + the AGENTS/README/MANIFESTS/BASELINE headers,
**never push from `~/llama.cpp`**, push only to `github.com:stew675/llama-cpp-rdna-boosts`.

## 10. References

- pwilkin journey: `https://pwilkin.github.io/strix-halo/journey.html`
- pwilkin lab/installer: `https://pwilkin.github.io/strix-halo/`,
  `https://github.com/pwilkin/strix-halo` (`install.sh`, `install-flash-next.sh`)
- branch: `https://github.com/pwilkin/llama.cpp/tree/strix-halo`
- model: `https://huggingface.co/ilintar/qwen3.8-flash-next-gguf-strix-halo` (uniform IQ4_NL, 93.16 GiB;
  also `ilintar/qwen3.8-27b-gguf-strix-halo`)
- halogen-flash (the other co-designed stack, for contrast): `https://github.com/peonist-ai/halogen-flash-server`
- this repo: `TODO.md` item 3, `GREEDY-PURITY.md` §22, `archive/work/kv-quant-purity-followups/`

---

## 11. 2026-09-12 session results — the port was executed (result: +18.4 %, bar NOT met)

Build base: canonical 16-patch tip `be0d23d57` (bu16). Ported artifact:
`wip/iq4nl-prefill/mmb-port.patch` (938 insertions, 3 files: new `ggml/src/ggml-cuda/mmb.{cu,cuh}`
+ 6 hooks in `ggml-cuda.cu`).

### Phase 0 — A/B reproduced (interleaved, same session, gfx1151)

| build | pp2048 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|
| rdna-boosts bu16 b1/b2 | 761/757 | 784/778 | **794/791** | 31.39 |
| pwilkin full env b1/b2 | 1241/1236 | 1350/1349 | **1407/1401** | 30.35 |

Ratio 1404/793 = **1.77×** — matches §0. Build config **ruled out**: both CMakeCache already have
`GGML_HIP_MMQ_MFMA=ON` and `GGML_HIP_NO_VMM=ON` (also GRAPHS=ON, RCCL=1, FA=ON, FA_ALL_QUANTS=OFF) —
the 1.77× is code, not flags. Check (a)/(b) need no rebuild.

### Phase 1 — family ranking (his build, full env minus one family, pp16384)

| family disabled | pp16384 | Δ |
|---|---|---|
| — (full env) | 1404.8 | — |
| NO QSA | 708.6 | −696 (nonlinear: dense fallback) |
| NO MMB | 935.2 | **−470 (−33 %)** |
| NO HC | 1115.8 | **−289 (−21 %)** |
| NO CONV (PLE+GDN) | 1342.7 | −62 |
| NO NORM | 1374.3 | −30 |
| NO IDX_RELU_SUM | 1394.3 | −10 |
| defaults (no env) | 705.9 | −699 |
| MMB only | 765.6 | −639 |

Families interact (deltas sum to >the whole); `LLAMA_MMB=1` alone is misleadingly small because the
default attention is then slow.

### Kernel-level attribution (rocprofv3, pp8192, our build 19.88 s kernel time)

| component | ours | his | gap |
|---|---|---|---|
| IQ4_NL weight GEMM (`mmb_*` incl. f32split/cvt) | 8.53 | 6.40 | −2.13 |
| `quantize_mmq_q8_1` (removed by bf16 staging) | 1.33 | ~0.2 | −1.13 |
| **QSA attention** (`flash_attn_qsa` 2.78 → `qsa3_attn` 0.73) | 2.78 | 0.73 | **−2.05** |
| HC (`hc_combine_norm`+`hc_mix_reduce` → 2 fused) | 1.62 | 0.93 | −0.69 |
| rocBLAS f32 (`SB`) → `mmb_f32split` | 1.13 | 0.50 | −0.63 |
| `mmb_cvt_f32_bf16` (bf16-producer cache absent) | — | 0.12 | −0.53 |

### Phase 2 — the port, measured (our build, `GGML_CUDA_MMB`, interleaved)

| gate | pp2048 | pp8192 | pp16384 | tg128 |
|---|---|---|---|---|
| off b1/b2 | 756/755 | 767/771 | **785/788** | 31.34 |
| on b1/b2 | 967/984 | 935/938 | **934/935** | 31.46 |

**+18.4 % prefill at pp16384 (787 → 934), decode unmoved.** What works: the dense IQ4_NL/Q8_0
`mmb_dense_kernel` (both tile classes), the TALL (M ≤ 384) class, the fused MoE
`mmb_routed_glu_kernel` + `mmb_routed_kernel`, and `mmb_f32split` (which replaced the rocBLAS `SB`
GEMM). `quantize_mmq_q8_1` mostly disappears.

**Two implementation notes that were required:**
1. The qwen4exp MoE never reaches `ggml_cuda_mul_mat`/`_mul_mat_id` — our graph optimizer fuses the
   whole gate+up+GLU (`mul_mat_q_pair` / swiglu→mmq) and the down (`mul_mat_id_weighted_rdna3_5`).
   MMB therefore stands those **prefill** fusions down when `ggml_cuda_mmb_active()` (the pair and
   swiglu branches; the single-token weighted-down one must NOT be gated — that is a decode fusion
   and gating it would break W = 1..8 purity).
2. `ggml_cuda_launch_mm_ids_bounded` is not in our tree; the existing
   `ggml_cuda_launch_mm_ids_helper` (which already has the RDNA3_5 `mm_ids_helper_512_10` fast path)
   is used instead.

### Verdict — the remaining gap is NOT a weight GEMM

After the port our kernel time is 16.79 s vs his 11.18 s. The remaining 5.6 s is dominated by
`flash_attn_qsa` **2842 ms vs his `qsa3` 730 ms (−2.1 s = ~75 % of the 2.5 s still needed for
1100)**; then `mmb_cvt` 645 ms (his bf16-producer marking), HC 0.70 s, misc. Our QSA is already the
best arm we have — forcing `LLAMA_QSA_SPARSE_FA=0` (829) or `LLAMA_QSA_OFF=1` (851) is *slower* than
the default (937). **Reaching >1100 requires porting his QSA v3 sparse-attention kernel (block-14
work), not more weight-GEMM work.**

### Next steps
- [ ] QSA v3 (`qsa3_attn_kernel` + `qsa3_rows_kernel`, `LLAMA_QSA_SCORE_BOUNDS`/`PACK_KEYS`/`FA_V3`)
      → expected ~+117 t/s; plus the bf16-producer marking (~+32) and the HC fusions (~+42) → ~1130.
- [ ] Then land: the `mmb` module belongs in **block 08** (prefill kernels) as an amendment; regenerate
      from a canonical fork at `9113cc188`, strict `git am`, applied tree == canonical.
- [ ] Purity TODO before landing: with `GGML_CUDA_MMB=1` the MMB path is prefill-only (T ≥ 512) so
      W = 1..8 is unchanged by construction; still run the item-4 `mstep` W = 1..8 probe + the MTP
      acceptance gate on the MMB-on build.

---

## 12. Why this is parked — a design-objective clash (decision 2026-09-12)

Parked, not rejected. The blocker is not effort, it is that pwilkin's approach is **architecturally
opposed to the property this repo sells**.

**rdna-boosts' contract:** a given `main` build produces *one* answer. Decode (`W = 1`) and every
speculative verify width (`W = 2..8`) must be bit-identical, so greedy output does not depend on
`--spec-draft-n-max` or on which fast path fired. Anything that can change the arithmetic is therefore
either (a) confined strictly above the decode/verify band (prefill-only, `T >= 512`, as the `mmb` port
does), or (b) opt-in and default-off with an explicit accepted-risk note. That discipline is what
`GREEDY-PURITY.md` exists to enforce, and it is what the last several sessions (F1, F2, cause-2/3, the
QSA decode arm, the MoE band) were spent repairing.

**pwilkin's contract:** throughput. His stack achieves it by token-count-gated switches that *change
the numerics* — `LLAMA_MMB_MIN_T=512`, `LLAMA_MTP_QSA_MIN_T=128`, `LLAMA_QSA_DENSE_SHORTCUT`,
`LLAMA_QSA_QUERY_STRIP=512`, the `LLAMA_HC_*` single-token fusions, bf16-WMMA vs q8_1 accumulation. He
ships no determinism guarantee and has no W = 1..8 matrix; "the fast path turns on at N tokens" *is* the
optimisation. Every such boundary is a potential decode-vs-verify split of exactly the class we keep
fixing in our own tree.

He is **not** unsafe about recurrent state — his tiled GDN carries `keep_rs = K > 1` and writes
per-token rollback snapshots, and both trees share the upstream `test-recurrent-state-rollback`. The
clash is about *determinism*, not corruption.

**Consequence:** each win we take from his stack has to be re-homed behind band-uniform gating (as the
`mmb` port already is, `T >= 512) before it can be a delivery default — and that is a per-win
validation cost that scales with how many of his gates we adopt. Hence: park, keep the measurement,
and only resume with a *general-purpose* rationale.

**Resume checklist (before the port could be opt-in, let alone defaulted on):**
1. Correctness vs a reference — same-seed coherence on our models and/or perplexity vs the
   pre-port build. Never done.
2. W = 1..8 logits matrix **with `GGML_CUDA_MMB=1`** (must match the gate-off widths).
3. `benchmarks/mtp-adaptive-methodology.md` acceptance gate on the MMB-on build.
4. `test-recurrent-state-rollback` with the gate on.
5. **Narrow the fusion stand-down to the weight type.** `ggml_cuda_mmb_active()` is currently a
   *global* guard on the block-13 MoE prefill pair/swiglu fusions, so `GGML_CUDA_MMB=1` on a
   non-IQ4_NL MoE model disables our fusion without MMB being able to take over — a prefill
   regression. Must become per-weight-type.
6. Compile + consistency on gfx1100 / gfx1201 (only gfx1151 was built).
7. Measure the **dense Q8_0 / F32** path on the 27B/4B Q8_0 models — that is the only genuinely
   general-purpose candidate here (`mmb_supported_mmid`/`_glu` are IQ4_NL-only, so MoE wins are
   inherently IQ4_NL-expert-model wins).

**The bigger general-purpose lever is not a weight GEMM at all** — it is the QSA sparse-attention
kernel: `flash_attn_qsa` 2842 ms vs his `qsa3` 730 ms, worth ~2.1 s on **both** the uniform-IQ4_NL and
the mixed IQ4_XS model. That is the item to pick up first when this is resumed.
