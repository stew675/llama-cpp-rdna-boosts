# gfx1201 (RDNA4) porting plan — the `mmb-general` WIP onto 3× R9700

**Status:** PLAN / active handover (opened 2026-09-21).  Not part of the delivery.  This file
supersedes the "gfx1201 is a no-op / new work, not a port" notes in `GROUPS.md` and `README.md`
(see §2 — those notes were written before the RDNA4 WMMA layout had an in-repo reference and
before the delivery's gfx1201 MMQ path was re-tuned).

**Session 1 log (2026-09-21):** the WIP was applied to `~/llama.cpp` as branch `rdna-boosts-mmb-port`
(`git am` **5/5**, clean) and **built green for gfx1201** with the delivery build script (`EXIT=0`,
100%, `llama-cli`/`llama-bench`/`llama-perplexity`/`llama-server` present) — the RDNA4 no-op wrappers
in `mmb.cu`/`fattn-qsa3.cu` do keep the multi-arch build compiling, as the WIP claimed.  The
baseline worktree `~/llama-base` is staged.  No porting code has been written yet.

**Audience:** whoever picks this up next — first on gfx1201, then on gfx1100.  Read this with
`GROUPS.md` (the 5-group triage) and `HANDOVER.md` (the gfx1151 development record).  This file is
the *porting* overlay; the group semantics stay as `GROUPS.md` describes.

---

## 0. TL;DR

* The prior "gfx1201 can't have `mmb`/`qsa3` because of the gfx11 WMMA builtin" decision is
  **stale in its conclusion**: the builtin genuinely differs, but the delivery already ships a
  **validated gfx12 bf16/f16 WMMA fragment layout** (`ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu`)
  and a portable abstraction (`mma.cuh`).  Porting MMB/qsa3 to gfx1201 is a **bounded mechanical
  fragment-layout transform**, not new algorithmic work.  See §2.
* But feasibility ≠ value.  The delivery's gfx1201 prefill is already re-tuned (blocks 08/10/13),
  unlike gfx1151 when MMB was conceived.  **Measure first, port second.**  The go/no-go for the
  WMMA groups is a baseline-vs-`mmb` A/B, not an article of faith.
* Three groups are **arch-neutral and were never actually run on gfx1201**: the indexer top-k
  (G5), the non-temporal hints (G4), and the always-QSA policy flip (G3a).  Do these first — they
  need no WMMA port and are the low-risk wins.
* Port order: **G5 → G4 → G3a → G2 (qsa3, RDNA4 WMMA) → G1 (mmb, RDNA4 WMMA) → G3b/c/G4-HC16 → gates/handover.**
* gfx1100 shares the gfx11 WMMA builtin, so it needs **none** of the gfx12 fragment work: it can
  test G1/G2 with `GGML_CUDA_MMB_RDNA3=1` as soon as the gfx1201 re-tune and gating land.  That is
  the explicit handoff in §11.

---

## 1. Scope, target hardware, working state

| | |
|---|---|
| Host | this box: **3× AMD Radeon AI PRO R9700 (gfx1201, RDNA4)**, Ryzen 9 9950X3D2, 184 GiB RAM |
| ROCm | `/opt/rocm-7.14.1-gfx102X` (the build script's `ROCM_714`; supports `--offload-arch=gfx1201`); `/opt/rocm-7.14-gfx1201` is the older parallel install |
| Delivery base | `~/llama.cpp` branch `rdna-boosts`, tip `c3ee45747` = the 16-block **r12** delivery (tree `8a80535e…`) |
| WIP base | the same r12 applied tree; the 5 WIP patches apply **5/5 clean** onto it |
| WIP working branch | `~/llama.cpp` branch **`rdna-boosts-mmb-port`** (created this session; do not push it) |
| Baseline worktree | `~/llama-base` (branch `rdna-boosts`, delivery only) — build the A/B baseline here |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (ccache; see §3) |
| Dense model | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| MoE/QSA/HC model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/` (94 GiB, 3 shards) + `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |
| Fast iteration model | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…` (21 GiB, `qwen35moe`, no HC/QSA — for `mmb_*` shapes) |
| Rule | all multi-GPU testing is **`-sm tensor`** (see the server invocation in `AGENTS.md` / the session brief) |

**WARNING:** `mmb-general.patch` / `patches/*.patch` under `wip/` must never be folded into the
delivery or applied to a *delivery* checkout.  This plan and the code on `rdna-boosts-mmb-port` are
WIP only.  Promotion is maintainer-gated (`HANDOVER.md` §E).

---

## 2. Re-examination of the prior gfx1201 decisions

### 2.1 The claim and why it is stale

`README.md` §"Why" (line 21) and `HANDOVER.md` §6/§13 say, and `GROUPS.md` repeats:

> the `mmb` kernels use the first-gen gfx11 WMMA builtin `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`.
> gfx12 needs `..._bf16_w32_gfx12` … So this is RDNA3-gated by construction; gfx1201/gfx1100 keep
> the existing MMQ/QSA path.  … if the RDNA4 `v_wmma_*` path is wanted, that is **new work**, not a port.

Every **premise** in that paragraph is true; the **conclusion** ("new work", "shelved") is what does
not survive re-examination, for two reasons:

1. **The RDNA4 WMMA layout is already probed, documented and shipped in this tree.**  The delivery's
   fused chunked GDN exists in two arch-specific files precisely because of this:
   * `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16_gfx11.cu` — the gfx11 (RDNA3/RDNA3.5) kernel,
     comment: *"16 bf16/lane fragments, layouts probed on gfx1100, 256/256 reference-matmul validated"*.
   * `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu` — the gfx12 (RDNA4) kernel, comment:
     *"libr4d's r4d_gdn_wmma.h, layouts probed on this hardware, not documented by AMD"*.
   * `ggml/src/ggml-cuda/mma.cuh` already selects `bf16x8_t`/`halfx8_t` + `..._gfx12` under `RDNA4`.
   So the exact lane mapping MMB/qsa3 need is **in-repo, validated, and reachable** — this is a port
   of a known layout, not an invention.

2. **The gfx1201 baseline changed after the WIP was written.**  MMB was motivated by gfx1151's weak
   MMQ path and its ~2× gap to the tuned RDNA3_5 stacks.  Since then the delivery added the
   gfx1201-tuned block-08 quantized-KV enablements, block-10 k-quant boosts, block-13 fused MoE
   gate+up+GLU MMQ and the mmvq band work.  The gfx1201 prefill is **not** the gfx1151 prefill, so
   the gfx1151 MMB win percentages do **not** transfer.  This is the single most important thing to
   measure before committing to G1/G2.

### 2.2 The actual difference: accumulator and operand fragments

Both WMMAs compute a 16×16×16 tile with fp32 accumulation; only the **lane distribution** differs.
The delivery's own GDN headers state it:

| | gfx11 (RDNA3 / RDNA3.5) | gfx12 (RDNA4) |
|---|---|---|
| A/B 16-bit fragment | **16 bf16/f16 per lane**, full K-row in one lane: `idx = lane%16`, `k = 0..15`.  Lanes 16-31 **mirror** 0-15. | **8 bf16/f16 per lane**, "two runs of four": `idx = lane%16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`.  Lane half selects the K-half. |
| A/B source read (K-contiguous) | 16 contiguous shorts at `row*pitch + kk` | two 8-byte groups 16 B apart: 4 shorts at `+4*(lane>>4)`, 4 shorts at `+4*(lane>>4)+8` |
| C/D f32 accumulator | `n = lane%16`, **`m = 2*e + (lane>>4)`** (interleaved rows) | `n = lane%16`, **`m = 8*(lane>>4) + e`** (contiguous halves) |
| builtin | `__builtin_amdgcn_wmma_f32_16x16x16_{bf16,f16}_w32` | `__builtin_amdgcn_wmma_f32_16x16x16_{bf16,f16}_w32_gfx12` |

`mma.cuh` (`tile<...>`/`mma()`) is the modern abstraction; MMB and qsa3 predate it and hand-roll
their fragments against the gfx11 layout.  That is the whole gap.

### 2.3 Concrete port surface (counted in the current WIP tree)

`ggml/src/ggml-cuda/mmb.cu` (1760 lines) bakes the gfx11 layout into 5 kernels:

| site | lines (current) | what changes |
|---|---|---|
| `mmb_wmma_bf16` / `mmb_wmma_f16` wrappers | 41-53 | add the `_gfx12` arm (currently a shaped no-op) |
| `mmb_store_tile` epilogue | 492-496 | `2*e + cn` → `8*cn + e` (cn = `lane>>4`) |
| `mmb_tile_gemm` (dense + routed + GLU core) | frag load 614, mma 624, epilogue 641 | `v16s`→8-wide, LDS read split, acc row map |
| `hc_gate_mix_kernel` | frag load 720, epilogue 736-742 | same |
| `mmb_tile_gemm_glu` | frag load 908, epilogue 942 | same |
| `mmb_f32split_kernel` | frag load 1088, epilogue 1111/1117 | same (f16 path) |
| `mmb_tiny_m_f32_kernel` | epilogue ~1111 region | same (f16 path) |

`ggml/src/ggml-cuda/fattn-qsa3.cu`: the `qsa3_wmma_f16` wrapper (41), every `v16s` fragment
(`qf`/`kf`/`vf`/`pf`), the accumulator map (`2*e+hi` at 505), and the PV `ptile`/shuffle exchange
(`pp`/`ph`, 452-463) which is layout-sensitive — the GDN gfx12 `gdn_fragT`/`gdn_store_acc8_*` are
the reference for that transform.

**Recommended shape:** do **not** fork `mmb.cu`/`fattn-qsa3.cu` into two files (the GDN split exists
because its two kernels diverged structurally).  Instead add a small arch-selected fragment/acc
shim at the top of each file (a `mmb_frag_t`, `mmb_ld_frag`, `mmb_mma_bf16/f16`, `mmb_acc_m(e,hi)`)
and keep one kernel body.  The two architectures share the dequant, the pipelining and the geometry;
only the lane bookkeeping differs.  (For qsa3 the shuffle block may force a small `#if RDNA4` arm.)

### 2.4 What still stands from the old assessment

* **gfx12 has no gfx11 builtin.**  True; the `_gfx12` suffix is required.  The `RDNA4` no-op wrapper
  is still correct *until* the port lands — keep it as the guard.
* **The tiling is gfx1151-tuned.**  `GROUPS.md` is right that BM/BN/VDR/`THRESH`/`load_regs` constants
  must be re-measured on a new arch, not transferred.  RDNA4 has a different LDS budget / WMMA
  throughput / occupancy than RDNA3.5.
* **gfx1201 is a *tuned-differently* sibling, not a copy.**  The delivery's own gfx1201 work
  (block-04 head cap, block-13 MMQ) shows arch-specific tuning is real.
* **qsa3 is prefill-only (`n_q >= 128`)** — the W=1..8 decode/verify band and width purity are
  untouched by the port by construction.  Keep that gate.
* **gfx1201 QSA prefill already favours QSA.**  `src/models/qwen4exp.cpp` records the Soar (3×R9700
  tensor) measurement: QSA wins prefill from ~8K monotonically to **+181 % @160K**, and dense wins
  decode at every depth 8K-160K.  So the arch policy is already correct; qsa3 only makes the
  already-chosen QSA path faster.

### 2.5 Re-classification

| group | old note | new classification | gate |
|---|---|---|---|
| G1 `mmb` | "RDNA4 no-op; new work" | **portable fragment port**; value must be measured | `GGML_CUDA_MMB=1` + arch gate |
| G2 `qsa3` | "needs RDNA4 variant; new work" | **portable fragment port** (f16) | compile-time `LLAMA_QSA3_ENABLE` + arch gate |
| G3a always-QSA flip | arch-neutral | **port directly** | env kill-switch only |
| G3b/c F32/tiny-M | rides G1 | **port with G1** | `GGML_CUDA_MMB*` |
| G4 non-temporal | portable | **port directly**, per-kernel A/B | none (code) |
| G4 HC16 producers | rides G1 | **port after G1**, measure | `GGML_CUDA_MMB_HC16=1` |
| G5 indexer top-k | generic | **port directly** | none (op-driven) |

---

## 3. Environment, build, run

### 3.1 Build

```sh
# WIP build (the port target)
cd ~/llama.cpp
BUILD_DIR=build-rocm JOBS=16 ~/bin/build-llama-rocm-714          # rm -rf + configure + build, ccache
# fast loop while iterating (no reconfigure):
cmake --build build-rocm --target llama-cli llama-bench llama-perplexity llama-server -j 16

# baseline build (delivery only) for A/B
cd ~/llama-base
BUILD_DIR=build-rocm JOBS=16 ~/bin/build-llama-rocm-714
```

Notes:
* The script hardcodes the ROCm tree to `/opt/rocm-7.14.1-gfx102X` and `-DGPU_TARGETS=gfx1201`, and
  enables ccache.  The rpath is `$ORIGIN:/opt/rocm-7.14.1-gfx102X/lib`, so no `LD_LIBRARY_PATH` is
  needed to run, but exporting it is harmless and matches `AGENTS.md`.
* Adding new `.cu` files (`fattn-qsa3.cu`, `indexer-topk.cu`, `mmb.cu`) needs a **configure** pass
  (the `file(GLOB … "*.cu")` is evaluated at configure time).  The full script does that; the fast
  `cmake --build` loop does not.
* ccache makes the A/B rebuild cheap (unchanged TUs are replayed).  Because the WIP touches the FA
  group (`fattn-*.cuh`) and `ggml-cuda.cu`, expect a partial recompile, not a pure cache hit.

### 3.2 Apply the WIP (already done on `rdna-boosts-mmb-port`)

```sh
cd ~/llama.cpp && git checkout rdna-boosts && git checkout -b rdna-boosts-mmb-port
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/patches/*.patch   # 5/5
```

To reset to a clean base: `git checkout rdna-boosts && git branch -D rdna-boosts-mmb-port` and redo.

### 3.3 Run

* **Always** `llama-cli --single-turn` (and `--no-display-prompt` for scripted output) or it blocks.
* Warm the page cache before benches; never run benches in parallel.
* Use `-b/-ub 2048` for the 94 GiB IQ4_XS (larger ubatch OOMs at pp16384).
* The 27B Q8_0 needs `-lm none -lzm on` on some subcommands.
* Server invocation for the target model is in the session brief; adapt `--ctx-size` / `--spec-*`
  as needed.  Keep `-sm tensor`.

---

## 4. Baseline gates to record before any port

Run every gate **twice**, on `~/llama-base/build-rocm` (delivery) and
`~/llama.cpp/build-rocm` (WIP, MMB off) — the second must be *identical* to the first on every
text/PPL/op gate (the arch-neutral groups are the only expected deltas, and only where enabled).

| # | gate | command sketch | pass |
|---|---|---|---|
| B1 | coherence, dense | `llama-cli -m 27B-Q8_0 -ngl 99 -sm tensor -p "The capital of France is" -n 20 --seed 42 --temp 0 --single-turn` | same-seed text identical build-to-build |
| B2 | coherence, qwen4exp | same on Flash-Next IQ4_XS (f16 or q8_0 KV), `--reasoning off` | same-seed text; record the delivery hash |
| B3 | prefill, dense | `llama-bench -m 27B-Q8_0 -ngl 99 -sm tensor -p 2048,8192,32768 -n 0 -b 2048 -ub 2048` | record t/s |
| B4 | decode, dense | `llama-bench … -p 0 -n 128` and a **depth-16384** run (benchy protocol) | record t/s |
| B5 | prefill MoE | `llama-bench -m Flash-Next … -p 2048,8192,32768,65536` | record t/s |
| B6 | op oracles | `test-backend-ops -o FLASH_ATTN_QSA` (18/18), `-o GATED_DELTA_NET` (46/46), `-o INDEXER_TOPK`, `-o FLASH_ATTN_EXT` | green |
| B7 | width purity | `test-logits-width-probe <model> <prompt> 1024 512` (built from `tests/test-logits-width-probe.cpp` by G3) | `PASS (worst maxdiff 0)` for the pure types |
| B8 | PPL | `llama-perplexity -m Flash-Next -f prompts/prose-rdna-boosts.txt -c 2048 -b 2048 -ub 2048`, and a long-context PPL | record |
| B9 | MTP | `benchmarks/mtp-adaptive-methodology.md` (acceptance > ~0.45 at pos 1; MTP ≥ plain at depth 3) | pass |

**Important:** B6's `FLASH_ATTN_QSA` on the WIP with the RDNA4 no-op will still run the VEC path and
must be green; `INDEXER_TOPK` is the G5 oracle and must be green before G5 is believed.

Record every number in a new dated section of `WORKLOG.md` at the end (not in this file).

---

## 5. Group inventory against gfx1201

| # | group | gfx1201 status today | port work | expected gfx1201 payoff | risk |
|---|---|---|---|---|---|
| G5 | indexer top-k op | **generic, always-on**, compiles+runs | none (apply) | qwen4exp long-context prefill: gfx1151 family 2.94 %→1.80 % of run | low |
| G4 | non-temporal hints (dsv4_hc, concat, moe-weighted-reduction, fused gated-unary) | **generic**, `GGML_CUDA_MMB_HC16`-independent code paths | none (A/B per kernel, loads only) | `dsv4_hc_pre` was −18.6 % on gfx1151; unknown on gfx1201 | low (a bad hint = regression, easy to drop) |
| G3a | always-QSA prefill flip (drop the dense shortcut) | policy, applies to gfx1201 | none | gfx1201 already prefers QSA prefill from ~8K; the flip mainly removes the `n_kv<=2051` seam — **but** without qsa3 the VEC kernel serves it, so it may be ~neutral | low |
| G2 | `qsa3` packed-block WMMA | **arch-gated RDNA3_5**, gfx12 no-op | RDNA4 f16 fragment port (~4 sites) | qwen4exp prefill above ~2051; gfx1151 was +8-12 %; gfx1201 likely bigger (R9700 WMMA) | medium |
| G1 | `mmb` bf16-WMMA dequant GEMM | **arch-gated RDNA3_5**, gfx12 no-op | RDNA4 bf16 fragment port (~7 sites) + re-tune | the big prize *if* baseline MMQ is weak; **unmeasured on gfx1201** | medium-high |
| G3b/c | F32 shape split + tiny-M kernel | rides G1 | with G1 | router/hc-inject shapes | low once G1 works |
| G4 | HC16 bf16 producers | rides G1 (`GGML_CUDA_MMB_HC16`, default 0) | after G1 | kills `mmb_cvt_f32_bf16`; +1-3 % on gfx1151 | low |

---

## 6. Port designs

### 6.1 G5 — indexer top-k (do first)

* **Files:** `ggml/src/ggml-cuda/indexer-topk.cu` (new), `src/llama-memory-hybrid-idx.cpp`
  (`blk_cells`), `src/models/qwen4exp.cpp`.
* **Gating:** none needed; the op is generic CUDA/HIP.
* **Correctness gate is special** (`GROUPS.md`): the indexer selection width is 2051, so a `-c 8192`
  prose prompt never leaves the dense shortcut.  Validate with a **≥16k-token same-seed A/B** plus
  `test-backend-ops -o INDEXER_TOPK` and a long-context PPL.
* **Also fold in the delivery item** `HANDOVER.md` §D: `GGML_OP_INDEXER_FILL` is missing from
  `GGML_OP_NAME` (one line).
* **Kill-switches present:** `LLAMA_INDEXER_NOBLOCK`, `LLAMA_INDEXER_NOGROUP`.

### 6.2 G4 — non-temporal

* **Files:** `common.cuh` (`ggml_cuda_nt_load`), `dsv4-hc.cu`, `concat.cu`,
  `moe-weighted-reduction.cu`, `unary.cu`.
* **Gating:** none; a per-kernel A/B.  `AGENTS.md` §Scope: non-temporal hints are a portable class.
* **Rule learned on gfx1151:** load hints only, per kernel; a non-temporal *store* evicts the next
  op's input.  Re-measure `dsv4_hc_pre`/`_post` and `concat_transposed` individually.
* The `dsv4_hc` code is HC/qwen4exp-only; `concat`/`moe-weighted-reduction`/gated-unary affect the
  MoE path broadly.

### 6.3 G3a — always-QSA flip

* One hunk in `src/models/qwen4exp.cpp`: `LLAMA_QSA_DENSE_SHORTCUT` default flips from on to off.
* On gfx1201 this is **policy**: measure the `n_kv <= 2051` band with the VEC QSA vs the dense
  shortcut (`-c 2048`, and a pp2048-at-depth shape per the 2026-09-07 record's caveat) before
  accepting.  If the VEC QSA loses that band on gfx1201, keep the flip **off on RDNA4** until G2
  lands, then re-measure (that is exactly the coupling the WIP comment describes).
* Do not conflate this with the `qsa_dense_decode_until`/`qsa_dense_prefill_until` arch policy —
  those already encode "gfx1201 = QSA prefill always / dense decode always".

### 6.4 G2 — `qsa3` on RDNA4 (f16 WMMA)

* **Files:** `fattn-qsa3.cu` (wrapper + fragments + acc map + PV shuffle), `fattn-qsa.cu`/`.cuh`,
  `ggml.h`/`ggml.c` (`ggml_flash_attn_qsa_set_packed`), `qwen4exp.cpp`.
* **Transform:** the §2.2 gfx12 mapping.  The probability tile (`ptile`) and the `__shfl_xor(x,16)`
  cross-lane reduction are the delicate part: the gfx12 accumulator row is `8*hi + e`, so the
  `ph[2*e]/ph[2*e+1]` packing that assumes interleaved rows must be re-derived.
  `gdn_fragT` / `gdn_store_acc8_b16` in `gated_delta_net_chunked_bf16.cu` is the worked example.
* **Gating:** extend `ggml_cuda_flash_attn_qsa3_supported()` from `RDNA3_5` to `RDNA3_5 || RDNA4`.
  Keep the `q->ne[1] >= 128` prefill gate and the `n_stream == 1` graph gate.
* **Validation:** `test-backend-ops -o FLASH_ATTN_QSA` 18/18 (the CPU oracle compares *the op*, which
  on gfx1201 now must exercise the WMMA kernel — add cases if the support predicate is what gates the
  test), long-context same-seed A/B, PPL vs the dense oracle (`LLAMA_QSA_SPARSE_FA=0`) within noise.
* **W=1..8 purity is preserved by construction** (prefill-only).  Re-run B7 anyway.

### 6.5 G1 — `mmb` on RDNA4 (bf16/f16 WMMA)

This is the largest item.  Split it into three sub-steps so a failure is isolated:

**6.5.1 The shim (no behaviour change on RD3).**  Add to `mmb.cu`:

```cpp
#if defined(RDNA4)
typedef short mmb_frag_t __attribute__((ext_vector_type(8)));   // 8 bf16/f16
static __device__ __forceinline__ mmb_frag_t mmb_ld_frag(const uint16_t * row, int kk, int lane) {
    const uint16_t * p = row + kk + 4 * (lane >> 4);
    const uint2 g0 = *(const uint2 *) p;             // k+0..3
    const uint2 g1 = *(const uint2 *) (p + 8);       // k+8..11
    return __builtin_bit_cast(mmb_frag_t, (uint2[2]){g0, g1});
}
static __device__ __forceinline__ int mmb_acc_m(int e, int hi) { return 8 * hi + e; }
static __device__ __forceinline__ v8f mmb_wmma_bf16(mmb_frag_t b, mmb_frag_t a, v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
// mmb_wmma_f16: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
#else
typedef short mmb_frag_t __attribute__((ext_vector_type(16)));
static ... mmb_ld_frag(...) { 16 contiguous shorts }
static ... int mmb_acc_m(int e, int hi) { return 2 * e + hi; }
...
#endif
```

Then replace the five kernels' fragment loads / WMMA calls / acc maps with the shim.  The gfx11
generated code must be **bit-identical** (verify by the width probe and same-seed text before/after
the shim alone).

**6.5.2 Enable RDNA4 in the gate.**  `mmb_enabled()` gains
`if (GGML_CUDA_CC_IS_RDNA4(cc)) return true;` (keep `GGML_CUDA_MMB_RDNA3` for RDNA3_0).  Add
`GGML_CUDA_MMB_GFX12=0` as a debug kill-switch if useful.

**6.5.3 Re-tune (the gfx1151 constants are not assumed).**  The WIP's own §6 conclusion: tile knobs
were a wash *on gfx1151*; that says nothing here.  Sweep at least `mmb_glu_thresh`/`mmb_routed_thresh`
(default 32), `mmb_tall`, `mmb_f32split_min_m`, and the `min_t` prefill threshold (default 512).
Confirm on the fast model first (`Q4_K_M` 35B-A3B), then the target models.

**Correctness gates for G1:** PPL parity (`10.6015` is the gfx1151 baseline **for the target model and
prompt — do not reuse it as a gfx1201 expected value**; it is a *relative* gate), same-seed greedy /
`plain == draft-mtp`, width purity by construction (`T >= 512`), `test-backend-ops`.

**Optimisation to consider on RDNA4:** `v_cvt_pk_bf16_f32` **exists** on gfx12 (the WIP comment says
it is gfx12-only).  `mmb_pack2`'s integer `v_perm` RNE pack could be replaced by the hardware pack
on RDNA4 — the WIP verified the integer form is RNE-correct, so this is a pure speed question, and
only if profiling shows the pack hot.

### 6.6 G3b/c + G4 HC16

Mechanical once G1's shim exists: the F32 tiny-M kernel and the F32 split use the f16 WMMA shim; the
HC16 producer pass is host-side graph marking and only meaningful with MMB active.  Keep
`GGML_CUDA_MMB_HC16` default 0 and measure it as a separate A/B.

---

## 7. Gating strategy (what is "gfx1201-specific")

1. **Device code** stays behind the existing `#if defined(RDNA4)` compile guards; the gfx12 builtins
   are only compiled for gfx12 targets.  Do not remove the no-op arm — it is the multi-arch build
   guard until the port is proven.
2. **Runtime arch gate:** extend `mmb_enabled()` and `ggml_cuda_flash_attn_qsa3_supported()` to
   accept RDNA4.  Keep the env master switch (`GGML_CUDA_MMB=1`) and the compile-time
   `LLAMA_QSA3_ENABLE` gate.  Do **not** default either on during porting.
3. **Per-arch tuning constants** (`mmb_*_thresh`, `mmb_tall`, `mmb_min_t`, `qsa_dense_*`) must be
   **selected by `ggml_cuda_info().devices[0].cc`**, not compiled in, so gfx1151/gfx1201/gfx1100 can
   hold different values.  The WIP already uses env overrides with measured defaults — promote the
   measured per-arch value into a small `switch`/conditional once each arch is measured, keeping the
   env override for A/B.
4. **Never** let a gfx1201-only change alter the gfx1151 path: guard it (shim `#if RDNA4`, arch
   conditional) and re-run the gfx1151 gates mentally (or note that gfx1151 re-validation is the
   maintainer's at promotion).
5. **gfx1100** (next machine) needs no gfx12 guard — it shares the gfx11 shim.  Its gate is
   `GGML_CUDA_MMB_RDNA3=1` and a re-tune.  Leave that env gate in place.
6. New `.cu` files must be added to the build (glob re-configure) and registered consistently in
   `supported_mm`/`supported_mmid`/`supported_glu`/`dense_will_take`/`routed_will_take` **and** the
   three dispatch helpers (the `HANDOVER.md` §13 trap).

---

## 8. Session work breakdown

Each session should: start from `git branch --show-current` = `rdna-boosts-mmb-port`, rebuild, run
the relevant gates, commit the record to `wip/mmb-general` (see §9), and leave this file's §10
checklist updated.

| session | goal | deliverables | exit gate |
|---|---|---|---|
| **S1 (this)** | plan + build + baseline | this file; WIP build compiles for gfx1201; baseline worktree built | build green; baseline gates B1-B3 recorded |
| **S2** | arch-neutral wins | G5 indexer + G4 non-temporal applied and A/B'd; `GGML_OP_NAME` fill fix | long-context same-seed A/B + `INDEXER_TOPK` op oracle; per-kernel NT A/B |
| **S3** | G3a policy | always-QSA measured on gfx1201 (VEC path); decide RDNA4 default | pp2048/p2048-at-depth + `-c 2048` coherence; document the decision |
| **S4** | G2 `qsa3` RDNA4 port | f16 fragment port; gate extended | `FLASH_ATTN_QSA` 18/18; long-context A/B; PPL vs dense oracle |
| **S5** | G1 shim | `mmb` fragment shim; gfx11 code bit-identical | width probe + same-seed text unchanged on gfx1151 path (compile-time only here) |
| **S6** | G1 on RDNA4 | gate enabled; first correctness run on the fast model | PPL parity + same-seed greedy on 35B-A3B |
| **S7** | G1 re-tune + target | sweep the knobs; run the 27B Q8_0 and Flash-Next gates | baseline-vs-MMB prefill A/B; decision: default on/off |
| **S8** | G3b/c + HC16 + gates | F32/tiny-M/HC16; B1-B9 full matrix | all gates green; gating documented |
| **S9** | handover to gfx1100 | update §10/§11; commit record | gfx1100 TODO list complete |

If G1 measures **no win** on gfx1201 (the baseline is already strong), stop at S4- and record it:
qsa3 + indexer + non-temporal may still be the gfx1201 delta, and G1 becomes a gfx1100-only item.

---

## 9. Per-session record convention

* Code goes in `~/llama.cpp` branch `rdna-boosts-mmb-port`.  Do **not** commit it to `rdna-boosts`.
* Regenerate the WIP backup after each code change and commit it to
  `llama-cpp-rdna-boosts` branch **`wip-mmb-general`** (never `main`):
  ```sh
  cd ~/llama.cpp && git format-patch --start-number 1 <r12-tip>..HEAD -o /tmp/mmb  # or the 5-patch layout
  # then copy into wip/mmb-general/ and commit the record on branch wip-mmb-general
  ```
  The current 5-thematic-patch backup stays valid until a code change; a new tree needs a re-cut
  (follow the `GROUPS.md` "consolidate 38→5" recipe and re-verify `git am` 5/5).
* Never push out of `~/llama.cpp` (`AGENTS.md` Pushing policy).
* New gate numbers go in `WORKLOG.md` (dated, newest first), not in this plan.

---

## 10. Live checklist

- [x] WIP applies 5/5 and builds for gfx1201 (S1 — verified 2026-09-21, EXIT=0)
- [ ] Baseline gates B1-B9 recorded for both builds (S1 — in progress)
- [ ] G5 indexer ported + validated (S2)
- [ ] G4 non-temporal A/B'd per kernel (S2)
- [ ] G3a always-QSA decided for RDNA4 (S3)
- [ ] G2 qsa3 f16 RDNA4 port + gate (S4)
- [ ] G1 mmb fragment shim, gfx11 bit-identical (S5)
- [ ] G1 RDNA4 correctness (fast model) (S6)
- [ ] G1 RDNA4 re-tune + baseline-vs-MMB decision (S7)
- [ ] G3b/c + HC16 + full gate matrix (S8)
- [ ] gfx1100 handover written (§11) (S9)

---

## 11. gfx1100 handover (next machine)

gfx1100 (RDNA3_0) solves a **different** half of the same problem:

* It **shares the gfx11 WMMA builtin**, so G1/G2 need **no gfx12 fragment work** — the existing
  `mmb.cu`/`fattn-qsa3.cu` kernels compile and run with the gfx11 shim.
* Its gate is already present: `GGML_CUDA_MMB_RDNA3=1` opens MMB on RDNA3_0 (untested by the WIP
  author).  qsa3's support predicate must be extended to `RDNA3_0` as well (same idea).
* The work is therefore: (a) verify G1/G2 correctness with the env gate on, (b) **re-tune** the
  gfx1151 tile/threshold constants (single 7900 XTX, smaller LDS, different WMMA rate), (c) measure
  against the gfx1100 baseline — the delivery's own block-04 work shows gfx1100 is a
  tuned-differently sibling.
* Do the gfx1201 arch-neutral groups (G5, G4) first on gfx1100 too — they are free.
* The gating in §7 must already support gfx1100 (env + per-arch constant selection) so gfx1100 does
  not have to re-architect anything.

---

## 12. Risks / open questions

1. **G1 value on gfx1201 is unknown.**  The delivery's gfx1201 MMQ is much better than gfx1151's was.
   If MMB is a wash, keep it gfx1100-only and ship qsa3+indexer+NT on gfx1201.
2. **The gfx12 bf16 `_gfx12` builtin operand type.**  MMB uses `short` vectors; GDN uses `__bf16`
   vectors.  Both should lower to `<8 x i16>`; confirm at first compile (S5).  If `short` is
   rejected, switch the shim to `__bf16` + a `bitcast`.
3. **The qsa3 PV shuffle** (§6.4) is the highest-risk single transform; budget S4 for it and keep the
   `FLASH_ATTN_QSA` oracle as the arbiter.
4. **Re-tuning is a time sink.**  The gfx1151 record shows tile knobs are mostly flat once the
   geometry is right; timebox the sweep and default to the gfx1151 constants until a shape is
   measured worse.
5. **`rocprofiler-register` env-gate flakiness** (`GROUPS.md` §5): never trust a gated path under
   `rocprofv3` without confirming the kernel name in the trace; `LLAMA_QSA3_ENABLE` was made
   compile-time for this reason.
6. **Don't judge routed/GLU on end-to-end t/s** (`GROUPS.md` §6): use `rocprofv3` kernel time and a
   fixed-token `llama-perplexity` for routed shapes (`llama-bench` random prefill is not comparable
   run-to-run).
