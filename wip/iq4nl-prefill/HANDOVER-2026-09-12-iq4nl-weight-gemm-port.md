# HANDOVER — IQ4_NL prefill: port pwilkin's generic weight-GEMM fast path into rdna-boosts

**Date:** 2026-09-12
**Box:** Strix Halo (gfx1151), ROCm 7.14 at `/opt/rocm-7.14-gfx1151`, 124 GB unified
**Anchoring TODO item:** **NEW — propose item 18** (`qwen4exp weight-IQ4_NL prefill fast path`).
**Distinct from item 3** (`qwen4exp iq4_nl prefill delta`), which is the `iq4_nl` **KV-cache** axis and
stays exactly as recorded — see §1.
**Success criterion:** **> 1100 t/s** prefill at pp16384 on pwilkin's uniform-IQ4_NL model, up from the
current **787 t/s** (rdna-boosts) / **1409 t/s** (pwilkin's stack)
**Companion files:** `PROMPT.md` (the paste-ready session prompt), `launcher-env.txt` (pwilkin's full env)

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
