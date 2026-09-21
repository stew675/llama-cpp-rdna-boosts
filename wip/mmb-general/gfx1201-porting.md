# gfx1201 (RDNA4) porting plan — the `mmb-general` WIP onto 3× R9700

**Status:** PLAN / active handover (opened 2026-09-21).  Not part of the delivery.  This file
supersedes the "gfx1201 is a no-op / new work, not a port" notes in `GROUPS.md` and `README.md`
(see §2 — those notes were written before the RDNA4 WMMA layout had an in-repo reference and
before the delivery's gfx1201 MMQ path was re-tuned).

> **HANDING THIS FILE TO A NEW SESSION?  Read the S15 section of §13.**  **S1-S14 are done** and are
> history: the gfx1201 port is complete, every B1-B9 gate passes, and MTP has now been validated on
> gfx1201 for the first time.  The only work left is **S15** — freeze, regenerate the patch set, verify
> `git am` N/N, and hand gfx1100 the tree.  The qwen4exp prefill win went +3.1/+3.4 (S7) -> +4.9/+5.2
> (S12) -> +6.7/+6.5 (S13) on the *mmb* axis, and **+22 % whole-WIP vs the delivery at depth** (S14),
> by removing gates S7 had put in front of paths that actually win on RDNA4.

**Session 1-2 log (2026-09-21):** the WIP was applied to `~/llama.cpp` as branch `rdna-boosts-mmb-port`
(`git am` **5/5**, clean) and **built green for gfx1201** with the delivery build script (`EXIT=0`,
100%).  Baselines were recorded (B1/B2 same-seed hashes, B3-B5 prefill/decode, B6 op oracles, B7 width
purity) and the arch-neutral groups were A/B'd, including deep sweeps to **pp65536 and pp98304**.
Findings and raw numbers: **`gfx1201-s1s2-results.md`**.  In one line: **G5 (indexer) is a win that
grows with depth (+2 % @8k → +8.0 % @98k), G4 (non-temporal) is a small consistent win (+0.3-0.4 %
at depth), and the always-QSA flip (G3a) is a large regression** and is now gated off on non-gfx1151.
The delivered patch set was regenerated to fold in the G3a arch gate (patch 3); it applies clean 5/5
and the applied tree is `c0f8ea75ba` (byte-identical to the tested `bdf97a390` tip).  No WMMA porting
code has been written yet — that is S4+ below.

**S4 log (2026-09-21): G2 `qsa3` is PORTED to RDNA4 and validated.**  The gfx12 fragment shim landed
(one `#if` block, 7 sites) and the kernel now has its **first unit oracle on any arch** — the
`FLASH_ATTN_QSA` test never attached `src[7]/src[8]`, so it had only ever exercised the VEC kernel.
26/26 on gfx1201; **+7.6 / +11.5 / +10.4 % prefill** at pp4096/16384/32768 (same build,
`LLAMA_QSA3_ENABLE` 1 vs 0).  The G3a arch gate was re-tested with qsa3 active and stays.  Full data:
**`gfx1201-s4-qsa3-results.md`**; the port is folded into patch 2 (tree `e9aa886ac`, `git am` 5/5
verified).  The S1/S2 record also gained a missing long-context text gate: with qsa3 off the WIP is
byte-identical to the delivery, so G5/G4 are text-pure and the qsa3 delta is the approved
re-baseline.  Next: S5-S7 (G1 `mmb`).

**S5-S7 log (2026-09-21): G1 `mmb` is PORTED, but it is NOT a blanket win — the switch was split.**
The gfx12 bf16 fragment shim landed and the gfx11 device asm is verified **bit-identical** (only the
`__hip_cuid_*` source hash differs; opcode histogram byte-equal) — with the lesson that the gfx11
fragment load must expand via a **macro**, not a `__device__` function (the dead argument perturbed
the scheduler and broke bit-identity).  Correctness is PPL-parity (12.7378 vs 12.7221 over 7 chunks
on the MoE IQ path).  But the measurement is unambiguous: with all types and the dense path on, MMB
is a **−4…−13 % regression** on three models, while the wins are confined to the **routed MoE** and
**qwen4exp HC** shapes.  So `GGML_CUDA_MMB` (which bundled arch × weight-type × path × model) was
split into an arch-scoped **weight-type** policy (`mmb_wtype_ok`, one mask replacing five duplicated
hard-coded lists) plus a **path** policy (`mmb_dense_flag`), and RDNA4 now defaults to the safe
subset: IQ-family weights, routed/HC paths only, dense/router off.  Result: **neutral wherever it
would lose, +3…+7 % where it wins**.  Full matrix: **`gfx1201-s5s7-mmb-results.md`**.

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
| WIP base | the same r12 applied tree; the WIP patches apply **10/10 clean** onto it |
| WIP working branch | `~/llama.cpp` is currently on **`mmb-port-qsa3`** (tip `8c686d9ef`, tree `35fc853e6396cb0867e7e27c1e8e21093699db47`) = the **delivered 10-patch set**.  The set grew 5 -> 6 in S5-S7 (patch 6 = the `mmb` RDNA4 fragment port + the scope split), then 7 (S10 dense geometry + per-type dense policy), 8 (S11 per-arch tuning table), 9 (S12 routed policy) and 10 (S13 F32 policy split) — because patches 1, **3 and 4** all touch `mmb.cu`, so no single theme can own a later `mmb.cu` change without a full re-cut (see `gfx1201-s5s7-mmb-results.md` §4).  Older local branches `mmb-5-ported` / `rdna-boosts-mmb-port` (tree `c0f8ea75ba`) and the S2 isolation variants `v-no-g5` / `v-no-g4` also exist.  **Do not push any of them.** |
| Baseline worktree | `~/llama-base` (branch `rdna-boosts`, delivery only) — build the A/B baseline here |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (ccache; see §3) |
| Dense model | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| MoE/QSA/HC model | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/` (94 GiB, 3 shards) + `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |
| Fast iteration model | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…` (21 GiB, `qwen35moe`, no HC/QSA — for `mmb_*` shapes) |
| Rule | all multi-GPU testing is **`-sm tensor`** (see the server invocation in `AGENTS.md` / the session brief) |

**WARNING:** `mmb-general.patch` / `patches/*.patch` under `wip/` must never be folded into the
delivery or applied to a *delivery* checkout.  This plan and the local branches are WIP only.
Promotion is maintainer-gated (`HANDOVER.md` §E).

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
| G3a always-QSA flip | arch-neutral | **NOT portable — arch-gated** (always-QSA only on gfx1151) | arch gate in patch 3 + `LLAMA_QSA_DENSE_SHORTCUT` env |
| G3b/c F32/tiny-M | rides G1 | **port with G1** | `GGML_CUDA_MMB*` |
| G4 non-temporal | portable | **port directly**; gfx1201 verified win, kept | none (code) |
| G4 HC16 producers | rides G1 | **port after G1**, measure | `GGML_CUDA_MMB_HC16=1` |
| G5 indexer top-k | generic | **port directly**; gfx1201 verified win (deep) | none (op-driven) |

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

### 3.2 Apply the WIP

The **delivered** 6-patch set lives in `wip/mmb-general/patches/`.  It applies `git am` **6/6** onto a
fresh r12 tree and yields applied tree **`580db5174574f10cc92fb1cefa72281a65c77b12`** (verified):

```sh
# fresh build / iteration branch from the r12 delivery
cd ~/llama.cpp && git checkout rdna-boosts && git checkout -b mmb-port-work
git am /home/stew675/llama-cpp-rdna-boosts/wip/mmb-general/patches/*.patch   # 6/6
```

The 2026-09-21 session's already-applied branches are kept for convenience: `mmb-port-qsa3`
(`50b813085`, tree `580db5174`, the delivered set — **the current one**), `mmb-5-ported`
(`e7cd749bc`, tree `c0f8ea75ba`, the pre-S5-S7 5-patch state), `rdna-boosts-mmb-port`
(`bdf97a390`) and the S2 isolation variants `v-no-g5` / `v-no-g4`.  To reset: `git checkout
rdna-boosts && git branch -D <branch>` and redo.

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
| G2 | `qsa3` packed-block WMMA | **ported to RDNA4 2026-09-21** (was gfx11-only) | none left — re-measure/tune if wanted | qwen4exp prefill above ~2051; measured **+7.6..+11.5 %** on gfx1201 (`gfx1201-s4-qsa3-results.md`) | low (done) |
| G1 | `mmb` bf16-WMMA dequant GEMM | **ported to RDNA4 2026-09-21, but scoped** (IQ family + routed/HC paths only) | a real gfx1201 dense tile geometry, per-type (S7+) | qwen4exp HC + MoE routed: **+3…+7 %**; dense: neutral-by-default (was −4…−13 % unscoped) | measured (`gfx1201-s5s7-mmb-results.md`) |
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

### 6.4 G2 — `qsa3` on RDNA4 (f16 WMMA) — **DONE 2026-09-21**

**Status:** ported, oracle-covered and measured — **+7.6 / +11.5 / +10.4 % prefill** at
pp4096/16384/32768.  Folded into patch 2 (`c3724f627`, tree `e9aa886ac`); the gfx11 arm is a
compile-time `#if`, so gfx1151 is byte-identical.  Data: `gfx1201-s4-qsa3-results.md`.  The original
plan for the work follows.

* **Files:** `fattn-qsa3.cu` (wrapper + fragments + acc map + PV shuffle), `fattn-qsa.cu`/`.cuh`,
  `ggml.h`/`ggml.c` (`ggml_flash_attn_qsa_set_packed`), `qwen4exp.cpp`.
* **Transform:** the §2.2 gfx12 mapping.  The probability tile (`ptile`) and the `__shfl_xor(x,16)`
  cross-lane reduction are the delicate part: the gfx12 accumulator row is `8*hi + e`, so the
  `ph[2*e]/ph[2*e+1]` packing that assumes interleaved rows must be re-derived.
  `gdn_fragT` / `gdn_store_acc8_b16` in `gated_delta_net_chunked_bf16.cu` is the worked example.
* **Gating:** **DONE** — `ggml_cuda_flash_attn_qsa3_supported()` accepts `RDNA3_5 || RDNA4`.
  Keep the `q->ne[1] >= 128` prefill gate and the `n_stream == 1` graph gate.
* **Validation:** `test-backend-ops -o FLASH_ATTN_QSA` 18/18 (the CPU oracle compares *the op*, which
  on gfx1201 now must exercise the WMMA kernel — add cases if the support predicate is what gates the
  test), long-context same-seed A/B, PPL vs the dense oracle (`LLAMA_QSA_SPARSE_FA=0`) within noise.
* **W=1..8 purity is preserved by construction** (prefill-only).  Re-run B7 anyway.

### 6.5 G1 — `mmb` on RDNA4 (bf16/f16 WMMA) — **DONE 2026-09-21: ported, but opt-in by type/path**

**Status:** the port is done and gfx11-bit-identical, but RDNA4 defaults to a **restricted subset**
(IQ-family weights, routed/HC paths) because MMB is a *net regression* with everything on.  See
`gfx1201-s5s7-mmb-results.md` §4/§6 for the scope split and §7 for what a re-tune should attack.  The
original plan for the work follows.

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

**6.5.2 Enable RDNA4 in the gate — DONE, but *scoped*.**  `mmb_enabled()` gained RDNA4, and the
blanket type list was replaced by an arch-scoped `mmb_wtype_ok()` plus `mmb_dense_flag()` — see
`gfx1201-s5s7-mmb-results.md` §4.  RDNA4's default is the IQ family with the generic dense GEMM and
the F32 router off; `GGML_CUDA_MMB_TYPES` / `GGML_CUDA_MMB_DENSE` are the A/B overrides.

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
2. **Runtime arch gate:** extend `mmb_enabled()` to accept RDNA4 (`ggml_cuda_flash_attn_qsa3_supported()`
   already accepts `RDNA3_5 || RDNA4` since S4).  Keep the env master switch (`GGML_CUDA_MMB=1`) and the
   compile-time `LLAMA_QSA3_ENABLE` gate.  Do **not** default `mmb` on during porting.
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
| **S4** | G2 `qsa3` RDNA4 port | f16 fragment port; gate extended | `FLASH_ATTN_QSA` 26/26 (3 new packed cases + a VEC baseline); +7.6/11.5/10.4 % prefill; long-context text re-baseline documented |
| **S5** | G1 shim | `mmb` fragment shim; gfx11 code bit-identical | gfx1151 device asm **byte-identical** but for `__hip_cuid_*`; opcode histogram equal |
| **S6** | G1 on RDNA4 | gate enabled; first correctness run on the fast model | PPL parity 14.3981 vs 14.4087 (+0.07 %), MMB demonstrably running |
| **S7** | G1 re-tune + target | measured the matrix; **split the switch** (arch-scoped weight types + path policy) | neutral where it loses, +3…+7 % where it wins; PPL parity on the active subset |
| **S8** | G3b/c + HC16 + gates | F32/tiny-M/HC16; B1-B9 full matrix | all gates green; gating documented |
| **S9** | handover to gfx1100 | update §10/§11; commit record | gfx1100 TODO list complete |
| **S10** | **gfx1201 dense tile geometry** | **DONE** — 256x128 tile + a per-type dense policy (IQ3_S) | met: the dense tile beats MMQ for IQ3_S (-4.5 %, kernel-time + interleaved A/B, +0.5 % model) |
| **S11** | arch-scoped tuning constants | **DONE** — `mmb_arch_cfg`/`mmb_arch_defaults(cc)` + a config dump; gfx1151 byte-unchanged, RDNA4 preserved | met |
| **S12** | routed/GLU tuning + kernel time | **DONE** — the routed path is a loss on every model; per-arch `routed` policy, RDNA4 default **off**; qwen4exp +4.9/+5.2 % | met (better than planned: a policy reversal, not a sweep) |
| **S13** | G3b/c F32/tiny-M + G4 HC16 | **DONE** — F32 policies separated (the split tile was never running); it scales with depth; HC16 measured inert | met (better: +1.7 % more on qwen4exp) |
| **S14** | B1-B9 on the final tree | re-run B1-B7; B8 long-context PPL; **B9 MTP** | ✓ **DONE 2026-09-21 — every gate green, MTP validated on gfx1201** |
| **S15** | freeze + regenerate + hand off | update the docs, `git am` N/N, gfx1100 hand-off | ✓ **DONE 2026-09-21 — frozen at `35fc853e63…`, `git am` 10/10, gfx1100 handed the tree** |

> **S1-S15 are DONE — the gfx1201 port is complete and frozen.**  The S14/S15 briefs are kept below as
> the executed record; results in `gfx1201-s14-gates.md`, and the S14 brief's two bad reference numbers
> are corrected in §S14.3c.

**S1+S2 status (2026-09-21): DONE.**  B1/B2 identical to the delivery; B6 oracles 2/2; B7 width purity
PASS.  Deep sweep done (32k/64k/98k).  G5 and G4 verified as wins, the G3a gate is folded into the
patch set, and the 5-patch backup was regenerated (tree `c0f8ea75ba`, `git am` 5/5 verified).  Detail:
`gfx1201-s1s2-results.md`.  The delivered set is now suitable for both gfx1151 and gfx1201 for the
arch-neutral items; the WMMA port was S4-S7 and is also done (see below).

**S4 status (2026-09-21): DONE.**  `qsa3` runs on gfx1201 and is the second-biggest gfx1201 win
(+7.6..+11.5 % prefill over the VEC kernel, same build).  The kernel now has its first unit-oracle
coverage; the G3a gate is re-confirmed with qsa3 active; and the long-context text gate shows the WIP
is byte-identical to the delivery with qsa3 off (G5/G4 pure) with the qsa3 delta being the approved
re-baseline.  Detail: `gfx1201-s4-qsa3-results.md`.  G1 (`mmb`) was then done in S5-S7 (`mmb` is
ported but **scoped** — see `gfx1201-s5s7-mmb-results.md`).

**S1-S7 are all DONE.**  S10-S15 (the remainder) are also DONE and the set is frozen — see §13, and
in particular §11 for the gfx1100 hand-off.

If G1 measures **no win** on gfx1201 (the baseline is already strong), stop at S4- and record it:
qsa3 + indexer + non-temporal may still be the gfx1201 delta, and G1 becomes a gfx1100-only item.

---

## 9. Per-session record convention

* Code goes in `~/llama.cpp` (a disposable WIP branch; `mmb-port-qsa3` at the S5-S7 tip).  Do **not**
  commit it to `rdna-boosts`.
* Regenerate the WIP backup after each code change and commit it to
  `llama-cpp-rdna-boosts` branch **`wip-mmb-general`** (never `main`):
  ```sh
  cd ~/llama.cpp && git format-patch --start-number 1 <r12-tip>..HEAD -o /tmp/mmb   # the 6-patch layout
  # then copy into wip/mmb-general/ and commit the record on branch wip-mmb-general
  ```
  **`git am` 6/6, applied tree `580db5174574f10cc92fb1cefa72281a65c77b12`** (verified).  A code change
  lands as a **new patch** by default: fold it into a theme only when the files it touches are owned by
  exactly one patch.  Patches 1, **3 and 4** all touch `mmb.cu`, which is why the S5-S7 `mmb` work is
  patch 6 rather than a fold.
* Never push out of `~/llama.cpp` (`AGENTS.md` Pushing policy).
* New gate numbers go in `WORKLOG.md` (dated, newest first), not in this plan.

---

## 10. Live checklist

- [x] WIP applies (5/5 at S1; now **6/6**) and builds for gfx1201 (S1 — verified 2026-09-21, EXIT=0)
- [x] Baseline gates B1/B2/B3/B5/B6/B7 recorded; B4 decode identical (S1)
- [x] G5 indexer measured — win, scales with depth (S2; `gfx1201-s1s2-results.md` §3b)
- [x] G4 non-temporal A/B'd — small consistent win, kept (S2; interleaved rounds)
- [x] G3a always-QSA decided for RDNA4 — gated off (S2/S3; folded into patch 3)
- [x] Patch set regenerated + `git am` 5/5 verified (tree `c0f8ea75ba`)
- [x] G2 qsa3 f16 RDNA4 port + gate (S4 — DONE 2026-09-21, tree `e9aa886ac`; +7.6/11.5/10.4 % prefill;
      `FLASH_ATTN_QSA` 26/26 incl. 3 new packed cases)
- [x] G5/G4 long-context text purity (S4 — WIP with qsa3 off == delivery, byte-identical)
- [x] G1 mmb fragment shim, gfx11 bit-identical (S5 — DONE: gfx1151 asm byte-identical but for `__hip_cuid`)
- [x] G1 RDNA4 correctness (fast model) (S6 — DONE: PPL parity +0.07 %)
- [x] G1 RDNA4 re-tune + baseline-vs-MMB decision (S7 — DONE: **split the switch**; scoped default
      per `gfx1201-s5s7-mmb-results.md`; a per-type/gfx1201 dense tile geometry remains open)
- [ ] G3b/c + HC16 + full gate matrix — **now S13/S14, see §13**
- [ ] gfx1100 handover written — **now S15, see §13**
- [x] **gfx1201 dense tile geometry (S10)** — **DONE 2026-09-21**: the 256x128 tile makes the IQ3_S
      dense GEMM beat the delivery MMQ by 4.5 %; landed as per-arch geometry + a per-type dense
      policy (IQ3_S only) -> +0.5 % prefill on 27B UD-IQ3_S; `gfx1201-s10-dense-geometry.md`
- [x] arch-scoped `mmb_*` tuning constants (S11) — **DONE 2026-09-21**: `mmb_arch_cfg` +
      `mmb_arch_defaults(cc)`, every tunable now `env || arch default`, the dense geometry moved into
      the table, `GGML_CUDA_MMB_CFG=1` dumps the resolved config; gfx1151 kernel set byte-unchanged;
      `gfx1201-s11-arch-defaults.md`
- [x] routed/GLU tuning + kernel-time evidence (S12) — **DONE 2026-09-21**: the routed MoE path is a
      **loss on every model measured** (it was masking half the qwen4exp win), so RDNA4 now defaults
      to `routed = 0`.  Landed default: qwen4exp **+4.9 / +5.2 %** (was +2.3/+1.8), IQ-MoE **neutral**
      (was -1.4 %), IQ3_S dense +0.5 %, text byte-identical everywhere;
      `gfx1201-s12-routed-policy.md`.  The threshold sweep is now moot on RDNA4.
- [x] G3b/c F32/tiny-M + G4-HC16 (S13) — **DONE 2026-09-21**: the two F32 paths were entangled and the
      split *tile* was wrongly gated behind `mmb_dense_flag()`, so neither ran on RDNA4.  Separated,
      both win and the split tile **scales with depth** (-0.3 % pp8192 -> **+1.22 % pp98304**), taking
      qwen4exp to **+6.7 / +6.5 %** shallow and **+6.2 % at 64k/98k**.  HC16/BLK16/RES16/DOWN16 are
      **inert** (0.03 % spread where the conversion stream is live) -> stay default 0;
      `gfx1201-s13-f32-hc16.md`
- [x] **S14 — the B1-B9 gate matrix on the final tree** — **DONE 2026-09-21**: all green, no code changed.
      **MTP runs on gfx1201 for the first time** (dense 0.636 / MoE 0.724 / qwen4exp 0.644-0.701
      acceptance, MTP +56..77 % over plain, purity byte-identical); the whole-WIP qwen4exp win at depth
      is **+22 %**; `gfx1201-s14-gates.md`
- [x] **S15 — freeze, regenerate, verify `git am` N/N, hand gfx1100 the tree** — **DONE 2026-09-21**:
      no drift (`patches/*` and `mmb-general.patch` regenerate byte-for-byte), `git am` **10/10** onto a
      fresh `c3ee45747` worktree producing tree `35fc853e6396cb0867e7e27c1e8e21093699db47` == the tested
      tree; docs updated and gfx1100 handed `GROUPS.md`'s job section

**S1-S15 are complete.  There is no remaining gfx1201 work.**  A new session picking this up should
read §11 (the gfx1100 hand-off) and `gfx1201-s14-gates.md` §8 (the traps) — not §13's briefs, which are
now history.

**Patch layout (current): 10 patches, tree `35fc853e63…`** — 1 mmb, 2 qsa3, 3 F32/tiny-M +
width-probe, 4 HC16, 5 indexer, 6 the `mmb` RDNA4 port + scope split, 7 the S10 dense geometry +
per-type dense policy, 8 the S11 per-arch tuning table, 9 the S12 routed policy, 10 the S13 F32
policy split.  Patches 6-10 form a chain on `mmb.cu`.

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
  not have to re-architect anything.  **It now does**: S11's `mmb_arch_cfg` / `mmb_arch_defaults(cc)`
  table holds the per-arch constants, so a gfx1100 row is a table edit plus a
  `GGML_CUDA_CC_IS_RDNA3_0` arm — and `GGML_CUDA_MMB_CFG=1` dumps the resolved config so a run can be
  proven to have used the constants it claims.

**What gfx1201 hands over (S15, frozen 2026-09-21).**  The patch set is **10 patches**, verified
`git am` **10/10** onto r12 `c3ee45747`, producing tree
**`35fc853e6396cb0867e7e27c1e8e21093699db47`** — the exact tree S14 validated (B1-B9 all green, MTP
included).  `GROUPS.md`'s "gfx1100 job" section is the work item; `gfx1201-s14-gates.md` is the gate
record, and its §8 lists the traps a gfx1100 session will otherwise re-derive.  The three that matter
most there: **sweep the weight types separately** (on RDNA4 the type axis turned G1 from a loss into
three wins), **gate every tile geometry on a same-seed hash** (`BN = (8/(BM/WTM))*WTN` or the kernel
silently computes part of the output and *looks* fast), and **prefer depth for a verdict** (the
most compute-dense config is the one whose shallow numbers are most `sclk`-sensitive).

---

## 12. Risks / open questions

1. ~~**G1 value on gfx1201 is unknown.**~~  **ANSWERED (S7):** it is not a blanket win — see §13 and
   `gfx1201-s5s7-mmb-results.md`.  The remaining question is the dense tile geometry (§13 S10).
2. ~~**The gfx12 bf16 `_gfx12` builtin operand type.**~~  **ANSWERED (S5):** `short` vectors are
   rejected by the `_gfx12` builtins; the shim uses `__bf16` / `_Float16` vector typedefs and
   `__builtin_bit_cast` at the wrapper.  It compiles and is correct.
3. ~~**The qsa3 PV shuffle.**~~  **ANSWERED (S4):** done and oracle-covered (§6.4,
   `gfx1201-s4-qsa3-results.md`).  `FLASH_ATTN_QSA` is the arbiter — keep it 26/26.
4. **Re-tuning is a time sink.**  Still true.  Timebox every sweep; only `r=5` interleaved rounds
   decide anything (§13.0 protocol).
5. **`rocprofiler-register` env-gate flakiness** (`GROUPS.md` §5): never trust a gated path under
   `rocprofv3` without confirming the kernel name in the trace.  **The S5-S7 gates are affected:**
   `GGML_CUDA_MMB_TYPES`, `GGML_CUDA_MMB_DENSE` and the `mmb_*` tunables are lazy host `getenv`s.
   Prefer `llama-bench` A/B for `MMB=1` vs off, and use the trace only for *kernel-time attribution*.
6. **Don't judge routed/GLU on end-to-end t/s** (`GROUPS.md` §6): use `rocprofv3` kernel time and a
   fixed-token `llama-perplexity` for routed shapes.  **The S7 routed numbers are end-to-end** and
   should be re-backed with kernel time in S12 (§13).
7. **The S7 MoE win does not reproduce (S10 finding).**  `35B-A3B UD-Q3_K_M` measured **+6.7 %** in
   S7; the *unmodified S7 binary* now measures **-1.4 %** on this box, and the kernel breakdown says
   why: the routed MMB (0.875 s) only **matches** the delivery's `mul_mat_q_routed_compact` +
   `mul_mat_q<IQ3_XXS,64>` (0.880 s) and standing that fused kernel down costs an extra
   `mm_ids_helper` launch (+0.065 s).  So `GGML_CUDA_MMB=1` on RDNA4 is currently a clear win only
   for **qwen4exp (HC/QSA)** models (+3.2 / +2.2 %) and **IQ3_S-heavy dense** models (+0.5 %) -- **the
   routed MoE default must be re-decided in S12**, together with the `mm_ids_helper` overhead.
   Evidence: `gfx1201-s10-dense-geometry.md` §6.

---

## 13. The gfx1201 work record — S10-S15, all DONE (kept as the execution record)

**Read this section, not the session logs above.**  S1-S13 are done; this is everything that is
open on gfx1201 — **none: S14 and S15 are both DONE and the set is frozen** (tree
`35fc853e6396cb0867e7e27c1e8e21093699db47`, `git am` 10/10).  **S14 was executed and is green; its
results are in `gfx1201-s14-gates.md`.**  The S14 section below is kept as the executed record (its
two bad reference numbers are corrected in §S14.3c).  Nothing here blocks gfx1100 (it can start on
G5/G4 and its own RDNA3_0 work in parallel).

### 13.0 Status and the measurement protocol

| done | what | evidence |
|---|---|---|
| S1 | baselines B1-B7, build green | `gfx1201-s1s2-results.md` §1 |
| S2 | G5 indexer (+2.1→+8.0 % by depth), G4 non-temporal (+0.3-0.4 %) | `gfx1201-s1s2-results.md` §3b |
| S3 | G3a always-QSA gated off on RDNA4 | patch 3; re-confirmed with qsa3 in S4 §3.1 |
| S4 | G2 `qsa3` ported (+7.6/+11.5/+10.4 %), first oracle coverage | `gfx1201-s4-qsa3-results.md` |
| S5 | G1 `mmb` gfx12 shim; gfx1151 asm bit-identical | `gfx1201-s5s7-mmb-results.md` §1 |
| S6 | `mmb` correctness (PPL parity) | `gfx1201-s5s7-mmb-results.md` §2/§5 |
| S7 | `mmb` **scoped** (arch-scoped weight types + dense/path policy) | `gfx1201-s5s7-mmb-results.md` §4/§6 |
| S10 | **the RDNA4 dense tile geometry (256x128) + a per-type dense policy** — IQ3_S beats MMQ by 4.5 %, +0.5 % prefill on 27B UD-IQ3_S; and the **S7 MoE win does not reproduce** | `gfx1201-s10-dense-geometry.md` |
| S11 | **per-arch tuning defaults** — `mmb_arch_cfg` + `mmb_arch_defaults(cc)`, the dense geometry in the table, `GGML_CUDA_MMB_CFG=1` dump; gfx1151 kernel set byte-unchanged | `gfx1201-s11-arch-defaults.md` |
| S12 | **the routed MoE path is a loss on RDNA4** — per-arch `routed` policy, **default off**; qwen4exp +4.9/+5.2 % (was +2.3/+1.8), IQ-MoE neutral (was −1.4 %) | `gfx1201-s12-routed-policy.md` |
| S13 | **the F32 policies separated** — the split tile was wrongly gated behind `mmb_dense_flag()`, so it never ran on RDNA4; it **scales with depth** (−0.3 % @8k → +1.22 % @98k) → qwen4exp **+6.7/+6.5 %**; **HC16 is inert** (0.03 %) | `gfx1201-s13-f32-hc16.md` |
| S14 | **B1-B9 on the final tree — DONE 2026-09-21.  All gates green; MTP works on gfx1201 for the first time (dense 0.636 / MoE 0.724 / qwen4exp 0.644-0.701 acceptance, MTP +56..77 % over plain); the whole-WIP qwen4exp prefill win at depth is **+22 %**, not +6 %** | `gfx1201-s14-gates.md` |

**Protocol — apply to every measurement below.**

* `llama-bench … -n 0 -r 5`; **the first prefill test of an invocation is cold-start-limited** (up to
  −9 % — this fooled S7 twice).  Decide only on `r=5` and on **interleaved back-to-back rounds**
  (run A, run B, run A, run B in one warm session).  Never run two benches at once.
* **On the 3-GPU qwen4exp bench the variance is a clock ramp, not heat (S13):** temps stay at
  43 °C edge / 70–83 °C junction, and repeating one config gives *rising* throughput.  The MMB path
  varied **6.7 %** at pp8192 across a session against the delivery's 2.4 %, because MMB is the more
  compute-dense configuration and so is the more `sclk`-sensitive one.  Agreement by depth: pp8192
  ±3 %, pp32768 ±1.6 %, pp65536 ±0.1 %, pp98304 ±0.4 %.  **So a sustained (deep) run is the stable
  instrument and the shallow numbers are the noisy ones** — which is also where a long-context
  workload's time goes.  Prefer pp32768+ for a verdict; treat a shallow-only delta as ±2 %.
* Warm the page cache first (`cat <model> >/dev/null`) for the multi-shard models.
* For a *decision*, prefer `pp32768` over `pp8192`; `pp8192` is only trustworthy inside an
  interleaved pair.
* Routed/GLU shapes (MoE experts): attribute with `rocprofv3` kernel time (`-d /tmp/prof`; note that
  `-o` alone writes under `./<host>/<pid>/` in this ROCm build) **and** a fixed-token
  `llama-perplexity`, per §12.6.
* Record every number in a dated section of the results file for that session (not in this plan).

**Load-bearing traps (do not re-derive).**

1. **The gfx11 arm of any shim must be a macro, not a `__device__` function.**  A function leaves a
   dead argument that perturbs the scheduler and breaks codegen bit-identity.  Verify with the
   `mmb.cu`-for-gfx1151 asm diff recipe (`/tmp/asmgen.py`-style: take the compile command from
   `build-rocm/compile_commands.json`, swap `offload-arch`, add `-S --cuda-device-only`); the only
   acceptable diff is `__hip_cuid_*`.
2. **The MMQ-fusion stand-down is per-tensor**, driven by `ggml_cuda_mmb_supported_mm` /
   `_dense_will_take`.  So an excluded (type, path) costs *nothing* — it keeps the delivery's MMQ
   path.  Any new gate must be added to **all** of `supported_mm`, `supported_mmid`,
   `supported_glu`, `dense_will_take`, `routed_will_take` or the graph and the dispatch disagree.
3. **`MMB_BK = 64` is effectively fixed**: the `mmb_dq_row*` dequant helpers hard-code 64 values per
   row and `MMB_LDS_STRIDE = MMB_BK + 8`.  Treat changing `BK` as a rewrite, not a knob.
4. **No dispatch currently passes `DBUF`/`DBUF2 = true`** — the code comments claiming "DBUF is a
   −6.7 % win for the IQ3_S GLU" are stale.  `DBUF` is a live, untried lever (it needs `As` at 2x).
5. All `mmb_*` tunables are read once into function-local statics: changing an env var needs a new
   **process**, not just a new model load.

### S10 — the gfx1201 dense tile geometry  — **DONE 2026-09-21**

**Result:** `wip/mmb-general/gfx1201-s10-dense-geometry.md`.  The RDNA4 dense path lost for every
weight type because of the geometry, not the architecture:

* **Mechanism (rocprofv3, 27B UD-IQ3_S, pp8192, 1 GPU):** the gfx1151-tuned tile needs **55296 B of
  LDS** -> **1 block/CU** (8 warps, 2 per SIMD) against the delivery MMQ's **0 LDS / 3 blocks**;
  and with `BM=128` only **half** of the 256 threads dequantise the A panel (`A_ITEMS = 1`), which
  is serialised with the WMMA work.
* **A 256x128 tile** (WTM=64, WTN=64, TMxTN=4x4, same 55296 B) puts all 256 threads on the A dequant
  and makes the IQ3_S dense GEMM **1.856 s vs MMQ's 1.944 s = -4.5 %**.  IQ4_XS still loses (+10.9 %)
  and IQ3_XXS/IQ4_NL are break-even, so the dense path is a **per-TYPE** decision: RDNA4 enables it
  for **IQ3_S only** (`mmb_dense_tmask()`/`mmb_dense_type_ok()`, `GGML_CUDA_MMB_DENSE_TYPES=<csv>`).
* **27B UD-IQ3_S interleaved A/B, r=5:** **+0.52 % pp8192 / +0.46 % pp32768** (two rounds agreeing to
  0.02 %).  27B Q8_0 exactly neutral; Flash-Next IQ4_XS +3.2 / +2.2 % (the S7 qwen4exp win
  reproduces); **gfx1151 instruction-identical** (79/79 kernels, md5 `fc698705809822f4c821adb115367dc8`).
* **Validity rule, easy to get wrong:** `BN` must equal `(8/(BM/WTM))*WTN`, else the kernel silently
  computes only part of the output and *looks* fast (a 256x192/WTN48 arm measured a fake -39 %).
  Gate every geometry candidate on a same-seed text hash; the geometry itself is numerics-neutral.
* **Correction to S7 (§12 risk 7):** the `35B UD-Q3_K_M` "+6.7 % MoE" does **not** reproduce -- the
  unmodified S7 binary now measures **-1.4 %** (the routed MMB only matches the delivery's
  `mul_mat_q_routed_compact` and adds an `mm_ids_helper` launch).  The routed default needs a
  re-decision in S12.

The original brief follows (the runtime-selector recommendation was **not** taken: a real `mmb.cu`
rebuild is 14 s and the A/B reference `MMB=0` is a runtime switch in the same binary, so an
edit-and-rebuild sweep is simpler than a second instantiation set).

**Why:** every RDNA4 `mmb` loss lives in the *generic quantized dense tile GEMM*
(`gfx1201-s5s7-mmb-results.md` §3-§4): −3 % on an 82 %-IQ dense model, −6 % on Q8_0, −11 % on a
mixed-K model — for **every** weight type.  If that tile can be made to beat the delivery's MMQ path
on RDNA4, the dense path comes back and `mmb` becomes a win on dense models too, not just MoE/HC.

**Where:**
* geometry is chosen in `ggml_cuda_mmb_mul_mat` (`mmb.cu` ~1540-1630).  Today, with
  `big = (M >= 6144 && K >= 2560) || (shadow && …)`, overridable by `GGML_CUDA_MMB_TILE`:
  * dense big  `mmb_dense_kernel<128, 256, 64, 64, WTYPE>`
  * dense small `mmb_dense_kernel<128, 128, 32, 64, WTYPE>`
  * tall-M (HC, IQ4_NL) `mmb_dense_kernel<384, 64, 96, 32, 0>` / `<384, 32, 96, 16, 0>`
* routed big `mmb_routed_kernel<128, 128, 32, 64, WTYPE>` / small `<128, 32, 32, 16, WTYPE>`
* routed GLU big `mmb_routed_glu_kernel<64, 128, 32, 32, WTYPE>` / small `<64, 32, 16, 16, WTYPE>`
* the tile body is `mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, TAIL, DBUF, DBUF2>` (~line 546);
  `MMB_NT = 256` threads, `__launch_bounds__(MMB_NT, 2)`.

**Do this:**
1. Add a **runtime geometry selector** so the sweep needs no rebuild: extend the `MMB_TILE` idea into
   `GGML_CUDA_MMB_GEOM=<bm>x<bn>x<wtm>x<wtn>[,<big variant>]` choosing from a small set of
   pre-instantiated `mmb_dense_kernel<...>` arms under `#if defined(RDNA4)`.  Keep the counts small
   (compile time in this file is already the reason it is one TU).
2. Sweep, **for each dense weight type separately** (the type axis mattered in S7 — do not sweep them
   together): `BM` 128/256, `BN` 64/128/256, `WTM`/`WTN` 16/32/64/96, `DBUF` on/off, the big/small
   crossover (`MMB_TILE`, and the `M >= 6144 && K >= 2560` rule itself).
3. Check the **LDS budget** first, it is the likely binding constraint: the dense big tile needs
   `(BM + BN) * MMB_LDS_STRIDE * 2` bytes = `(128+256)*72*2` = **54 KiB**; confirm what the R9700's
   LDS per WGP actually allows at 2 blocks and whether `__launch_bounds__(256, 2)` is even met.  A
   larger `BN` may drop to 1 block and lose.
4. Attribute with `rocprofv3` kernel time against the MMQ kernel for the same op (this is the only
   way to see whether the tile is losing to MMQ or to something else), then back the winner with an
   interleaved `llama-bench` round on a dense model (`27B UD-IQ3_S` and `27B Q8_0` are the two best
   probes: 82 % IQ vs 100 % Q8_0, both 1 GPU).

**Exit gate:** a geometry where `mmb`'s dense tile ≥ the MMQ path for at least one weight type,
proven by an interleaved A/B **and** kernel time.  Then: purity check (PPL + same-seed greedy,
`GREEDY-PURITY.md` contract), and flip `mmb_dense_flag()`'s RDNA4 default **for that type only**
(the flag may need to become per-type rather than global — that is the natural shape of the fix).

### S11 — arch-scoped tuning constants (§7 point 3)  — **DONE 2026-09-21**

**Result:** `wip/mmb-general/gfx1201-s11-arch-defaults.md`.  Every `mmb_*` tunable is now
`env override || arch default` selected from the device cc by `mmb_arch_defaults(cc)` (one
`mmb_arch_cfg` table), and the S10 dense geometry moved out of the dispatch into `c.dense_geom`.
`GGML_CUDA_MMB_CFG=1` prints the resolved config once, which answers the §12.5 "a profiler cannot see
the env gates" problem.  RDNA4's row carries only the *measured* value (the geometry); every other
field keeps the gfx1151 value and is marked `TODO(S12)` — no invented tuning.  Verified: same-seed
hash unchanged, **gfx1151 kernel set byte-unchanged** (90 kernels, 0 differing), RDNA4 win preserved
(+0.45 / +0.46 %).  Landed as **patch 8** (`git am` 8/8).

**Why (original brief):** §7.3 requires per-arch values selected by `ggml_cuda_info().devices[0].cc`.  Only
`mmb_wtype_mask()` and `mmb_dense_flag()` do this today; every other tunable is env-only with a
gfx1151 default, so `mmb_min_t`, `glu_thresh`, `routed_thresh`, `tall_mode`, `tiny_m_*`,
`f32split_*`, `bf16w`, `hc16`, `down16`, `gatemix`, `iq3xxs_glu`, `shadow_*` cannot differ per arch
without the user setting env vars (and cannot be profiled safely, §12.5).

**Do this:** add `mmb_arch_defaults(cc)` (a small switch keyed on `GGML_CUDA_CC_IS_RDNA4` /
`_RDNA3_5` / `_RDNA3_0`) and route every tunable through it, keeping the existing env var as the
override.  Precedent in-tree: `mmb_wtype_mask()`, `mmb_dense_flag()`, and `qsa_arch_gfx()` for the
qwen4exp policy.

**Exit gate:** gfx1151 numbers **unchanged** (re-run one gfx1151-known workload if possible, otherwise
argue from the cc switch being a no-op there), and the RDNA4 defaults documented in one place.

### S12 — routed / GLU path: tuning and kernel-time evidence  — **DONE 2026-09-21**

**Result:** `wip/mmb-general/gfx1201-s12-routed-policy.md`.  This turned out not to be a threshold
sweep but a **policy re-decision**: the routed `MUL_MAT_ID` path loses on **every** model measured.
Added the counterpart of `mmb_dense_flag()` — a per-arch `routed` field in `mmb_arch_cfg` +
`mmb_routed_flag()` (`GGML_CUDA_MMB_ROUTED=0|1`) gating `supported_mmid`/`supported_glu`/
`routed_will_take` — and set **`routed = 0` on RDNA4**.  End-to-end interleaved (pp8192 / pp32768):
`35B UD-Q3_K_M` −1.44/−1.39 % -> **−0.05/−0.09 %**; `Flash-Next IQ4_XS` +2.3/+1.8 % ->
**+6.6/+5.4 %** — the routed path was *masking* most of the qwen4exp HC win.  Cause (S10 §6's 1-GPU
kernel breakdown): the delivery's block-13 `mul_mat_q_routed_compact` + `mul_mat_q` cost 0.880 s vs
mmb's 0.875 s, and the stand-down adds 0.065 s of `mm_ids_helper`.  Same-seed text is byte-identical
to the delivery with routed on or off.  Landed as **patch 9** (`git am` 9/9).  **The threshold sweep
is now moot on RDNA4** (it tunes a disabled path); the remaining per-arch fields that *are* worth
measuring (`tall_mode`, `tiny_m*`, `f32split_*`, `cache_max`) belong to the paths that now carry the
win.

**Why (original brief):** the routed MoE path is the RDNA4 `mmb` win (+6.7 % on `UD-Q3_K_M`), but its
tiling was never tuned here and the S7 comparison is end-to-end only (§12.6).

**Do this:**
1. Sweep `GGML_CUDA_MMB_ROUTED_THRESH` and `GGML_CUDA_MMB_GLU_THRESH` (both default 32) on
   `35B-A3B UD-Q3_K_M` (the +6.7 % case) and `Flash-Next IQ3_XXS`.  These pick the BN=128 vs BN=32
   expert tile class (`mmb_build_desc2`) and directly trade B-column padding against dequant volume.
2. **Re-try `DBUF` for the GLU** (trap 4): no dispatch passes it today, yet the comment claims −6.7 %
   for IQ3_S.  Wire it behind a knob and measure.
3. **Re-measure the IQ3_XXS fused-GLU arm** (`GGML_CUDA_MMB_IQ3XXS=1`, default off).  It was defaulted
   off on gfx1151/UD-Q3_K_M, but IQ3_XXS is one of the two RDNA4 winning types.
4. Back the numbers with `rocprofv3` kernel time for `mmb_routed_kernel` / `mmb_routed_glu_kernel`
   vs the MMQ/`mul_mat_id` kernel.

**Exit gate:** the routed win ≥ the S7 number with kernel-time evidence; the GLU/IQ3_XXS decisions
re-stated on gfx1201.

### S13 — G3b/c (F32/tiny-M) and G4-HC16 producers  — **DONE 2026-09-21**

**Result:** `wip/mmb-general/gfx1201-s13-f32-hc16.md`.  The two F32 paths shared one predicate (so
`F32SPLIT=0` also killed the tiny-M kernel) and the split **tile** was additionally gated behind
`mmb_dense_flag()` — which is off on RDNA4 — so neither had ever run there.  Separated:
`mmb_f32split_mode()` governs only the tile, `mmb_tiny_m_f32_ok()` only the tiny-M kernel.  Both are
wins, and the split tile **scales with depth** on Flash-Next IQ4_XS (interleaved r=3, vs the delivery):
**−0.3 % pp8192, +0.9 % pp32768, +1.00 % pp65536, +1.22 % pp98304** — a fraction of a percent at the
start and >1 % once the context is deep, which is where a long run's time goes.  tiny-M is the bigger
but flatter win (~+4.5 %, pp8192→32768).  Landed: **qwen4exp +6.7 / +6.5 % shallow, +6.2 % at 64k/98k**
(was +4.9/+5.2), 27B IQ3_S +0.5 %, MoE neutral, text byte-identical.  **HC16 / BLK16 / RES16 / DOWN16
are INERT** — 0.03 % spread on 27B UD-IQ3_S *where the conversion stream is live* (178 `MMB_CVT`
lines) and noise on qwen4exp — so they stay default 0 as an unused opt-in.  `TINY_TT` 2/4 and
`CACHE=32` are also noise.  Landed as **patch 10** (`git am` 10/10); gfx1151 byte-unchanged.

**Why (original brief):** both ride G1 and both are **unmeasured on gfx1201**.  S7 made two *policy* calls
on them (routed the F32 router off, kept the tiny-M HC inject on) without isolating either.

**Do this:**
* **F32 split (MoE router)** — currently off on RDNA4 via `mmb_dense_flag()`.  Sweep
  `GGML_CUDA_MMB_F32SPLIT_MIN_M` / `_MIN_K` to see whether a gfx1201 shape wins; the gfx1151 rule was
  `M >= 128` (the router) with `MIN_K = 0` disabled and the long-K hc-inject pair *worse* on MMB.
* **tiny-M warp-per-token kernel** (the qwen4exp HC `*_inject` pair, `M <= 8`) — kept on; measure
  `GGML_CUDA_MMB_TINY_TT` (default 1; TT>1 measured worse on gfx1151).
* **HC16 bf16 producers** — `GGML_CUDA_MMB_HC16=1` (default 0) plus `GGML_CUDA_MMB_DOWN16`,
  `LLAMA_HC_BLK16`, `LLAMA_HC_RES16`.  The point is to kill the `mmb_cvt_f32_bf16` conversions
  (+1-3 % on gfx1151).  It is host-side graph marking, so it needs MMB enabled to mean anything.
* **Purity for each**: PPL + same-seed greedy, and for HC16 also a check that the F32 tensor is
  still produced where a consumer needs it.

**Exit gate:** each knob A/B'd, purity-checked, and given an arch default in S11's table.

### S14 — the B1-B9 gate matrix on the final tree  — **DONE 2026-09-21**

**Result:** `wip/mmb-general/gfx1201-s14-gates.md`.  Every gate green, **no code changed**.  Highlights:
B1/B2 five same-seed hashes identical across delivery / WIP-MMB-off / WIP-MMB-on; B6 four oracles green
(`FLASH_ATTN_QSA` 26 cases incl. the 3 `qsa3=1` arms); B7 width-pure with MMB on **and** off; B8
WIP-MMB-off **bit-identical** to the delivery at 9.4293 and MMB-on +0.022 %; **B9 MTP green on all three
model families** (the first time MTP has run on gfx1201), with the dense and MoE results byte-identical
to the delivery including acceptance, and rule 5's verify-width gate within noise at B=1/4/8.
The whole-WIP qwen4exp prefill win at depth is **+22 %** (of which mmb is +6.0-6.5 % and the arch-neutral
groups + qsa3 are +14.4-16.3 %), and the delivery figures reproduce the S1/S2 record to 0.3 %.

**Two errors in the brief below, both from reading S12/S13's *intra-WIP* "ON vs OFF" as
*WIP vs delivery*:** the delivery Flash-Next reference row (pp32768/65536/98304) is **2372 / 2213 /
2073**, not 2728 / 2591 / 2465 (those are qsa3-on, mmb-off WIP numbers); and the "expected landed
+6.7/+6.5/+6.2 %" is the **mmb-only** delta, not the delivery-vs-WIP one.  Both are corrected in
§S14.3c.  Also: `FLASH_ATTN_EXT`'s case count is randomised run-to-run (two runs of the *same* binary
differed by 34 cases), so only "0 FAIL" is a gate — not the brief's 5951.

**Trap found:`-md <mtp-head>` with `--spec-type none` aborts.**  On 1 GPU it is a deliberate clean
error ("this model is an MTP draft head without a trunk; load it as a draft of its target model, not on
its own"); on a `-sm tensor` split it trips `GGML_ASSERT(!suffix_fallback.empty())` at
`llama-model.cpp:470`, and `-fit off` does not avoid it.  **Pre-existing and identical on the delivery**
(maintainer confirms upstream too) — not a WIP regression.  The harness rule: the `plain` arm must not
pass `-md`.  Separately, **Flash-Next has no built-in `nextn` head** (the 27B and both 35B-A3B models
do), so qwen4exp *must* be given `-md` while the others must not.

The original brief follows (kept as the record of what was executed).

#### S14.0 Where things stand

| | |
|---|---|
| Repo of record | `/home/stew675/llama-cpp-rdna-boosts`, branch **`wip-mmb-general`** (docs only, no WIP code) |
| WIP code | `~/llama.cpp`, branch **`mmb-port-qsa3`**, tip `8c686d9ef`, tree **`35fc853e6396cb0867e7e27c1e8e21093699db47`** |
| Patch set | `wip/mmb-general/patches/` = **10 patches**, `git am` **10/10** onto r12 (`c3ee45747`) verified |
| Delivery reference | `~/llama-base` — branch `rdna-boosts`, the r12 delivery, already built |
| Build | `cd ~/llama.cpp && cmake --build build-rocm --target llama-cli llama-bench llama-perplexity llama-batched-bench test-backend-ops -j 16` (ccache; a full `BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` is only needed if you add a `.cu` file) |
| Runtime env | `export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx102X/lib:${LD_LIBRARY_PATH:-}` |

**Why this is the remaining gap.**  B1-B7 were recorded at S1 as *delivery-vs-WIP* and B6 again at S4,
but never re-run as a matrix on the current (10-patch) tree; B8 is only partly done; and **B9 (MTP)
has never been run on gfx1201 at all**.  The WIP carries decode-affecting changes (the G4
non-temporal hints, the G5 indexer, the HC/fusion bands) and the adaptive-MTP controller's constants
are **gfx1151-tuned** (S12/S13 did not touch decode).  B9 is therefore the one gate that can still
falsify something.

**Everything measured so far this campaign** is in `gfx1201-s1s2-results.md` (B1-B7),
`gfx1201-s4-qsa3-results.md` (qsa3 + B6), `gfx1201-s5s7-mmb-results.md`, `gfx1201-s10-dense-geometry.md`,
`gfx1201-s11-arch-defaults.md`, `gfx1201-s12-routed-policy.md`, `gfx1201-s13-f32-hc16.md`.

#### S14.1 Protocol (read before measuring anything)

* **Always `llama-cli --single-turn`** (plus `--no-display-prompt`) or it drops into the chat loop and
  blocks forever.  Wrap in `timeout`.  Extract text with
  `python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py /dev/stdin` (a naive
  `sed`/`grep` slice does **not** reproduce the hashes).
* **Never run two benches at once.**
* **Prefer depth for a verdict.**  Measured this campaign: agreement between interleaved rounds is
  **±3 % at pp8192, ±1.6 % at pp32768, ±0.1 % at pp65536, ±0.4 % at pp98304**.  The shallow end is
  `sclk`-limited (the first prefill of an invocation is clock-ramped, and it is worst for the most
  compute-dense config, i.e. the MMB-on one).  **Treat a shallow-only delta as ±2 %.**
* Multi-GPU is always **`-sm tensor`** with `GGML_CUDA_ALLREDUCE=hybrid` (the default).  Do **not** use
  `GGML_CUDA_ALLREDUCE=nccl` as a reference for `-sm tensor`: the internal AR BF16-round-trips while
  NCCL reduces small tensors in FP32, so they differ by design.
* `GGML_CUDA_MMB_CFG=1` prints the resolved per-arch config once — use it to record what a run
  actually tested:
  ```
  MMB_CFG cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
          f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=0 down16=0 gatemix=0
          blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=0
  ```
* **All the `mmb_*` tunables are lazy host `getenv`s read once per process**, so changing one needs a
  new process, and `rocprofv3` traces cannot show them.  That is why `MMB_CFG` exists.
* **`rocprofv3` is only trustworthy on 1-GPU runs.**  Under trace serialization on a 3-GPU
  `-sm tensor` run the `ncclDevKernel_Generic_4` time moved **+9 %** between two configs and inverted
  the sign of the total (S12 §3).  Use it for kernel *attribution* on single-GPU models, and the
  interleaved end-to-end A/B for anything tensor-split.
* Record every number in a new dated results file (`gfx1201-s14-gates.md`), not in this plan, and add
  a `WORKLOG.md` entry.

#### S14.2 The models

| role | path | GPUs |
|---|---|---|
| dense, 100 % Q8_0 | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (29 GB) | 2-3, `-sm tensor` |
| dense, IQ3_S-heavy (the S10 dense win) | `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` (11.2 GiB) | 1 |
| **qwen4exp** (HC/QSA; the S12/S13 win) | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (87 GiB, 3 shards) | 3, `-sm tensor` |
| MoE `qwen35moe` | `/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (17 GB) | 1 |
| MoE, K-quant | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21 GiB) | 1 |
| **MTP draft head (for B9)** | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` — a **symlink** to `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (2.79 GB) | with Flash-Next |
| prompts | `/home/stew675/llama-cpp-rdna-boosts/prompts/` — each prompt's size/token count/**sha256** is in `prompts/README.md`; **never edit a shipped prompt in place** | |

#### S14.3 The gates, in order

**S14.3a — the config record (new, 1 min).**  Before anything else, capture the resolved policy so the
matrix below is attributable:
```sh
HIP_VISIBLE_DEVICES=0 GGML_CUDA_MMB=1 GGML_CUDA_MMB_CFG=1 ~/llama.cpp/build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf -ngl 99 -p hi -n 2 --single-turn 2>&1 >/dev/null | head -1
```
Expect exactly the line in §S14.1 (`routed=0`, `f32split=1`, `dense_geom=1`).  Anything else means the
tree or the env is not what you think.

**S14.3b — B1/B2, same-seed coherence.**  The WIP must be byte-identical to the delivery **with and
without** `GGML_CUDA_MMB=1`:
```sh
# B1 dense 27B.  NOTE: the 27B Q8_0 needs `-lm none -lzm on` -- without it llama-cli prints
# nothing and just exits (the loader tries the mmproj/vision path).  Verified 2026-09-21.
llama-cli -m <27B-Q8_0> -ngl 99 -lm none -lzm on -p "The capital of France is" -n 20 \
  --seed 42 --temp 0 --no-display-prompt --single-turn | extract-generated.py /dev/stdin
# B2 qwen4exp (needs all 3 GPUs; --reasoning off per §4)
HIP_VISIBLE_DEVICES=0,1,2 llama-cli -m <Flash-Next> -ngl 99 -sm tensor --reasoning off \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn | extract-generated.py /dev/stdin
```
Known-good values to compare against — **all of these were re-verified on 2026-09-21** and all three of
delivery / MMB-on / MMB-off agreed on each:

| model / command | reference | recorded |
|---|---|---|
| 27B Q8_0, `-n 20 -lm none -lzm on` | `110 chars sha=da2e2d192e21` | S1, re-verified |
| Flash-Next, `-n 20 --reasoning off` | `35 chars sha=359ff4337837` | S1, re-verified |
| **27B UD-IQ3_S**, `-n 24` | **`119 chars sha=42cdf36d0633`** | S10/S11/S13 |
| **Flash-Next**, `-n 24` (no `--reasoning`) | **`135 chars sha=d73f9238f6d6`** | S12/S13 |
| **35B UD-Q3_K_M**, `-n 24` | **`110 chars sha=461ca8cd0e88`** | S12/S13 |

The last three are the campaign's own working references and are the quickest regression check — run
them first (they are cheap and catch a broken build immediately).

**S14.3c — B3/B4/B5, throughput.**  `llama-bench -n 0 -b 2048 -ub 2048 -r 5` (deep: see the protocol).
Compare against the delivery (`~/llama-base/build-rocm`) **interleaved**, and against the campaign's
numbers in §2 of `gfx1201-s1s2-results.md` and in `GROUPS.md`.

**Measured delivery reference points** (3-GPU tensor, q8_0 KV, `-b/-ub 2048`) — use these, they are
verified: **Flash-Next pp32768 = 2372, pp65536 = 2213, pp98304 = 2073** (reproducing the S1/S2 record
of 2379 / 2216 / 2074 to 0.3 %); 27B Q8_0 pp8192 ≈ 2302, tg128 ≈ 36.7; 27B UD-IQ3_S pp8192 ≈ 929 /
pp32768 ≈ 852; 35B UD-Q3_K_M pp8192 ≈ 5903 / pp32768 ≈ 4830.

> **The `2728 / 2591 / 2465` figures that used to be here were wrong** — they are *qsa3-on, mmb-off
> WIP* numbers (S4 recorded qsa3 pp32768 as 2724), not the delivery.  If a Flash-Next delivery number
> looks like ~2700 at pp32768, the binary is not the delivery.

**Expected deltas**, all interleaved delivery-vs-WIP(MMB=1), measured:

| model | harness | measured |
|---|---|---|
| qwen4exp Flash-Next | **whole WIP** at pp32768/65536/98304 | **+21.9 / +22.5 / +22.7 %** |
| qwen4exp Flash-Next | **mmb only** (same binary, `GGML_CUDA_MMB` toggled) | **+6.5 / +6.2 / +6.0 %** |
| 27B UD-IQ3_S | pp8192 / pp32768 | +0.5 / +0.6 % |
| 27B UD-IQ3_S | tg128, tg128@d16384 | flat |
| 35B UD-Q3_K_M | pp8192 / pp32768 | +0.5 / +0.6 % |
| 27B Q8_0 | pp8192 / tg128 | +0.3 % / flat |

So the S13-recorded "+6.7/+6.5 % shallow and +6.2 % at 64k/98k" is the **mmb-only** delta, and the
whole-WIP headline is an order of magnitude bigger.  Keep the two comparisons distinct in every report:
**"ON vs OFF" in S10-S13 always meant MMB-on vs MMB-off within the WIP binary, never WIP vs delivery.**

**B4 decode, do not skip it:** `-p 0 -n 128`, plus a **depth-16384** run (benchy protocol) — decode
perf work must be validated at depth, and `tg` is *not* a correctness signal.

**S14.3d — B6, the op oracles.**  All four must be green:
```sh
~/llama.cpp/build-rocm/bin/test-backend-ops -o FLASH_ATTN_QSA    # now 26/26 (S4 added 3 packed qsa3 cases; it was 18/18)
~/llama.cpp/build-rocm/bin/test-backend-ops -o GATED_DELTA_NET   # 46/46
~/llama.cpp/build-rocm/bin/test-backend-ops -o INDEXER_TOPK      # the G5 oracle
~/llama.cpp/build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT   # 5951/5951 per the delivery record
```
`FLASH_ATTN_QSA` is the *only* oracle the qsa3 kernel has and it now exercises the WMMA path on RDNA4
— if it regresses, qsa3 is suspect.  `INDEXER_TOPK` is the G5 oracle.  A fragment-layout error is an
exact-permutation error and moves PPL by orders of magnitude, not percent.

**S14.3e — B7, width purity.**  `test-logits-width-probe` (built by patch 3 from
`tests/test-logits-width-probe.cpp`) **on 27B, prose prompt, P=1024** — run it with `MMB=1` **and** with
MMB off, on f16 KV first:
```sh
~/llama.cpp/build-rocm/bin/test-logits-width-probe <27B-model> <prose-prompt> 1024 512
```
Expect `width_purity=PASS (worst maxdiff 0)`.  **Verified 2026-09-21** on 27B UD-IQ3_S / prose / P=1024:
`width_purity=PASS (worst maxdiff 0)` (f16 KV, the default).  The *guaranteed*-pure types are
**f16, bf16, q5_0, q5_1, iq4_nl**; `q4_0`/`q4_1`/`q8_0` are the relaxed ones (a measured, data- and
arch-dependent near-tie edge — `GREEDY-PURITY.md` §36).  `MMB` itself is prefill-only (`mmb_min_t`
keeps the `n_tokens <= 8` band off it), so a width regression here would implicate the FA chooser or
the HC bands, not the GEMM.

**S14.3f — B8, perplexity.**  Two forms — the **quick 2-chunk** one is a smoke test, the **many-chunk**
one is the measurement (the CI is what matters; 7 chunks took it from ±0.98 to ±0.45):
```sh
# quick (2 chunks) -- verified 2026-09-21: 35B UD-Q3_K_M -> PPL = 14.6145 +/- 0.98382
llama-perplexity -m <35B-A3B UD-Q3_K_M> -f prompts/prose-rdna-boosts.txt -c 2048 -b 2048 -ub 2048
# the measurement: concatenate the prose prompt 3x (or 7x) into a temp file and re-run
cat prompts/prose-rdna-boosts.txt prompts/prose-rdna-boosts.txt prompts/prose-rdna-boosts.txt > /tmp/prose3x.txt
llama-perplexity -m <Flash-Next> -f /tmp/prose3x.txt -c 2048 -b 2048 -ub 2048
```
References: fast model **14.3981** (off) vs **14.4087** (on) = +0.07 % (S6, 2 chunks); MoE
`UD-Q3_K_M` **12.7378** vs **12.7221** over **7 chunks** = −0.12 % (S7).  Note the chunk count changes
the absolute value a lot (2-chunk 14.61 vs 7-chunk 12.74 on the same model) — **only compare like with
like**, and prefer the many-chunk form for anything you intend to call a win.

**MMB is not expected to be bit-identical to the delivery here** (it rounds the weights to BF16 before
the WMMA), so B8 is a *parity* gate: a fragment-layout bug shows up as an order-of-magnitude move, not
0.1 %.

**S14.3g — B9, MTP.  This is the one that has never run on gfx1201 and the one most likely to
find something.**

Read **`benchmarks/mtp-adaptive-methodology.md`** (in this repo) in full and follow it — it is the
delivery's standard decode gate and it encodes the rules that were learned the hard way:

* **Rule 0 — the run must be long enough and the reasoning mode must be pinned.**  Use **`-n 3000`**
  (`-n 2000` floor): the code axis at adaptive ceiling 12 read **−5 % vs fixed `n3` at `-n 256`** and
  **+28 % at `-n 3000`** — a short run measures the drafter's and the controller's transient, not the
  mode.  Pin **`--reasoning on` for the R (reasoning) axis and `--reasoning off` for P/C/K**; an
  unpinned P/C run on Qwen3.8 measures the thinking trace, not the content.
* **Protocol A correctness first:** acceptance must stay **> ~0.45 at pos 1**, and **MTP ≥ plain** at
the default depth 3.  Then the four-axis gate (prose/code/recall/reasoning) at `-n 3000`.
* **Rule 5 — the verify-width gate:** run `llama-batched-bench -npl 1,4,8` stock-relative.  Acceptance
  and `llama-bench tg128` both pass while a verify-width regression is present, so this is a separate
  check, not a duplicate.
* The adaptive controller is the **tuned credit-bucket** one (`climb_budget(d) = 20 + 6*(d-1)`,
  `drop_pressure(d) = max(60, 10*d)`, cold start at `max(floor, cap-3)`, `--spec-draft-n-start N`
  override).  The delivery's reference cells: **27B 0.76744**, **qwen4exp 0.44262** (the block-14
  baseline).  Numbers and the rejected variants: `benchmarks/2026-09-15-adaptive-mtp-tuning.md` and
  `archive/work/adaptive-mtp-ceiling-scaling/`.
* Draft head: the models split two ways and this must be checked before wiring a gate —
  **Qwen3.8-27B (Q8_0 and UD-IQ3_S) and both Qwen3.6-35B-A3B models carry a built-in `nextn` head**
  (`qwen35.nextn_predict_layers` / `qwen35moe.nextn_predict_layers`, 4 tensors), used automatically:
  **do not pass `-md`.**  **Qwen3.8-Flash-Next IQ4_XS has none** (`nextn tensors=0`), so qwen4exp
  **must** be given `-md mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` — that is the "separate-MTP-head" path
  the r12 block-06 amendment covered.  Verify with `GGUFReader` for `'nextn' in t.name`.
* **Trap: `-md <head>` with `--spec-type none` aborts.**  1 GPU gives a deliberate clean error ("this
  model is an MTP draft head without a trunk; load it as a draft of its target model, not on its
  own"); a `-sm tensor` split trips `GGML_ASSERT(!suffix_fallback.empty())` at `llama-model.cpp:470`
  in the meta-split graph builder, and `-fit off` does **not** avoid it.  Pre-existing and identical on
  the delivery (upstream too) — so **the `plain` arm must not pass `-md`.**
* `--spec-type draft-mtp` (and `draft-mtp-adaptive`); the default `--spec-draft-n-max` is 3, capped at
  15 (`> 7` prints the purity notice — the pure range is `n_max <= 7`).
* **What to look for on RDNA4 specifically:** the controller's constants are gfx1151-tuned, and the
  block-13 mmvq band-uniformity rules (`nwarps=1` on RDNA4 dense, per-`(type,K)` weight kernel, the
  pinned fusion ops keeping plain `calc_nwarps`) are exactly what the MTP purity contract depends on.
  If B9 fails, the first thing to check is whether a fusion the band depends on is being taken with a
  different `nwarps` on this arch — `GREEDY-PURITY.md` §19 and §25 are the map.

**S14.3h — B9b, the MTP purity check for free:** `plain == draft-mtp` same-seed greedy text (the WIP
must reproduce this for the pure KV types; `q4_0`/`q4_1` relax the *logits* level, not the text —
`GREEDY-PURITY.md` §36).  Cheapest version: the B1/B2 prompt with `--spec-type none` vs
`--spec-type draft-mtp --spec-draft-n-max 3`.

#### S14.4 Exit gate

Every gate green (or every miss root-caused and written down) with the numbers in a new
`gfx1201-s14-gates.md`, plus a `WORKLOG.md` entry.  Then **S15** (§13's S15 section): freeze the policy,
regenerate the patch set, verify `git am` N/N, update `GROUPS.md`/`README.md`/§10/§11 and hand gfx1100 a
clean state.

**If something does fail,** the two most likely places are (a) the B9 controller constants being
gfx1151-tuned, and (b) a fusion band whose `nwarps` differs on RDNA4 — both are decode-side, both are
documented, and neither has been looked at since S1.  Report the failure with the gate name and the
`MMB_CFG` line rather than "it was slower".

**Do not** re-open the S10-S13 conclusions with shallow-only measurements: three of S7's "loses on
RDNA4" verdicts turned out to be "was never enabled on RDNA4", and each was caught by overriding the
flag and measuring deep.  The pattern and the two rules are in §13's S13 section and in `GROUPS.md`.

### S15 — freeze the policy, regenerate, hand off  — **DONE 2026-09-21**

**Result:** S14 found nothing to fix, so no code changed and there was no patch 11.  The set was frozen
and verified:

* `git format-patch --start-number 1 c3ee45747..mmb-port-qsa3` reproduces `patches/*` **byte-for-byte**
  (no drift), and `git diff c3ee45747..mmb-port-qsa3` reproduces `mmb-general.patch` exactly.
* `commits.txt` matches `git log --format='%H %s' c3ee45747..HEAD`.
* **`git am` 10/10** on a fresh worktree at `c3ee45747`, and the applied tree is
  **`35fc853e6396cb0867e7e27c1e8e21093699db47`** — identical to the tested fork tree.
* Docs updated: this plan (§11, §13.0, the S14 corrections), `GROUPS.md` (the gfx1201 results,
  the frozen tree, the gfx1100 job with the three traps), `README.md` (session 34), `WORKLOG.md`.
* Handed to gfx1100: the tree above, the patch set in `patches/`, and `GROUPS.md`'s **gfx1100 job**
  section.  `~/llama.cpp`'s branch stays local and is never pushed (AGENTS.md Pushing policy).

The original brief follows.

**Prerequisite:** S14 green (or its misses root-caused) — it was green, so nothing needed landing.

* If S14 **found** something: land the fix as **patch 11** (the convention below), then re-run the
affected gates.
* Update §10/§11, `GROUPS.md` (the gfx1100 job and the group-1 row) and `README.md` with the S14 result
  and the final numbers.
* `cd ~/llama.cpp && git format-patch --start-number 1 c3ee45747..HEAD -o /tmp/mmbN` → copy into
  `wip/mmb-general/patches/`, refresh `commits.txt` (`git log --format='%H %s' c3ee45747..HEAD`) and
  `mmb-general.patch` (`git diff c3ee45747..HEAD`); then verify `git am` **N/N** on a **fresh r12
  worktree** and that the applied tree equals the fork tip tree:
  ```sh
  cd ~/llama.cpp && rm -rf /tmp/amtest && git worktree add --detach /tmp/amtest c3ee45747
  cd /tmp/amtest && git checkout -q -b amcheck && GIT_EDITOR=true git am /tmp/mmbN/*.patch
  git rev-parse HEAD^{tree}      # must equal: cd ~/llama.cpp && git rev-parse mmb-port-qsa3^{tree}
  ```
  (`git worktree remove --force /tmp/amtest` when done.)
* Commit + push `wip-mmb-general`.  **Never push out of `~/llama.cpp`** (`AGENTS.md` Pushing policy).
  Hand gfx1100 the tree and the `GROUPS.md` job.

**Patch layout:** the set is **10 patches, tree `35fc853e6396cb0867e7e27c1e8e21093699db47`**.  Patches
6-10 are a chain on `mmb.cu`, because patches 1/3/4 also touch it — so a new `mmb.cu` change cannot be
folded into one theme and lands as a new patch.  Continue that convention: **land new work as a new
patch unless the touched files are owned by exactly one existing patch.**

### 13.1 Definition of done for "the remaining gfx1201 work"

- [x] the dense tile geometry has a gfx1201 answer (win → density re-enabled per type; or a recorded,
      kernel-time-backed "MMQ wins on RDNA4 for dense" so the decision is closed) — **S10: both.
      IQ3_S wins (-4.5 %) and is enabled; IQ4_XS/IQ3_XXS/IQ4_NL stay off, kernel-time backed**
- [x] every `mmb_*` tunable has an arch-scoped default with gfx1151 unchanged — **S11**
- [x] the routed/GLU path is tuned and kernel-time-backed; the IQ3_XXS GLU arm re-decided — **S12:
      the routed path is a loss on every model measured → per-arch `routed`, RDNA4 default off (the
      threshold sweep is moot on a disabled path); the IQ3_XXS GLU arm stays default-off**
- [x] G3b/c and HC16 are measured, purity-checked and defaulted — **S13: both F32 paths win and stay
      on; HC16/BLK16/RES16/DOWN16 measured inert → stay 0 as an unused opt-in**
- [x] B1-B9 green on the final tree, **including MTP** — **S14: all green; MTP validated on gfx1201
      for the first time on all three model families, acceptance 0.636/0.724/0.701, purity
      byte-identical, verify-width gate within noise**
- [x] gfx1100 has a clean, documented handoff (and its own `RDNA3_0` predicate work listed) — **S15:
      tree `35fc853e63…`, `git am` 10/10, `GROUPS.md`'s gfx1100 job section**
