# gfx1100 — closing-the-gap validation & porting brief

**Audience:** the agent working on the **single RX 7900 XTX (gfx1100, RDNA3_0)** box, 24 GiB VRAM.
**Goal:** apply the full delivery + `beta/mmb-general` + `archive/work/closing-the-gap` stack, **validate
every arch-sensitive piece that fits in 24 GiB**, and **port / explicitly park** the rest.

**Companion file:** [`gfx1201-closing.md`](gfx1201-closing.md) (3× R9700, 184 GiB — the box that can
run qwen4exp).
**Hard constraint:** **qwen4exp (Qwen3.8-Flash-Next) does not fit in 24 GiB**, so every
qwen4exp-only patch can only be **compile-checked, op-oracle-checked and host-gate-checked** here.
That is expected and is what the companion gfx1201 session is for.  Do **not** mark them "broken";
mark them "not end-to-end testable on this box; arch gates verified; trust gfx1151/gfx1201".
**Source of truth for what each patch is:** [`closing-the-gap.md`](closing-the-gap.md),
[`README.md`](README.md) and the dated `2026-09-*` records in this directory.
**Prior gfx1100 work:** `beta/mmb-general/gfx1100-porting.md` + the `gfx1100-s*-results.md` records.
Read those first — this box already validated the `beta/mmb-general` set (S1-S10).
**Delivery policy:** `AGENTS.md` (default-on policy, purity rules, **never push the `~/llama.cpp`
fork**).  Push the delivery repo only if the maintainer asks.

---

## 0. The one-paragraph summary

The campaign is 25 patches (`archive/work/closing-the-gap/patches/0001..0014`, `0016..0026`) on top of the
**r13 delivery (16 blocks)** + the **12 `beta/mmb-general` patches**.  It was developed and tuned on
**gfx1151 (RDNA3_5)**.  gfx1100 **shares the gfx11 WMMA builtin** with gfx1151, so it needs none of
the gfx12 fragment work — but several things are **hard-gated to RDNA3_5** and the **MMB default now
flips ON** (`0002`), which is the biggest gfx1100 change in the campaign.  Your job:

1. prove the **default** gfx1100 build (MMB on) recovers the S5-S10 `+14 % / +11 % / +5.5 %` wins;
2. validate the **arch-neutral** additions (`0004`, `0006`, `0009`, `0017`, `0018`, `0019`, `0023`,
   `0025`) and the **MoE** ones (`0010`) end-to-end on the models that fit;
3. **compile-check + oracle-check + host-gate-check** the qwen4exp-only patches;
4. **port** the RDNA3_5-only kernels that the shared gfx11 WMMA makes plausible (`0016`
   `QSA_SCORE_WMMA`, `0003` `hc_gate_mix`) — these are the campaign's real gfx1100 port candidates.

> **Do not expect the gfx1151 hash values** (`3553e76d3a9e`, `8285d12d40ca`, `d140b40f0eee`, …).
> On gfx1100 the gates are **intra-build** (`plain == draft-mtp`, width purity, MTP acceptance, PPL
> parity) plus the gfx1100 baselines in the `gfx1100-s*-results.md` records and a
> pre-closing-vs-closing A/B on this box (§6.0).

---

## 1. The machine

| | |
|---|---|
| GPU | **1× AMD Radeon RX 7900 XTX (gfx1100, RDNA3_0), 24 GiB VRAM** |
| **Must mask the iGPU** | the box also exposes a **gfx1036 iGPU**; every GPU command must run with **`HIP_VISIBLE_DEVICES=0`** or multi-device tools abort |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (ccache) |
| Runtime | `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1100/lib:$LD_LIBRARY_PATH` (the build script's `ROCM_714=/opt/rocm-7.14-gfx1100`; the rpath also covers it) |
| Multi-GPU | none — single GPU, no `-sm tensor` |
| Models that fit | see §6.1 |

---

## 2. Apply the full stack

> **Apply-order trap (read this first).**  The delivery repo's **`main`** branch carries the **r13**
> delivery + the 12 `beta/mmb-general` patches.  The **`gap-closing`** branch carries the 25
> `archive/work/closing-the-gap` patches **but its `release.json` is stale at r13's predecessor (`r12`)**.
> Apply the delivery from **`main`**, then the closing patches from **`gap-closing`**.  Do not run
> `scripts/apply-all.sh` from a `gap-closing` checkout — it would apply the r12 delivery.

```sh
REPO=git@github.com:stew675/llama-cpp-rdna-boosts.git
WORK=$HOME/rdna-boosts
[ -d "$WORK" ] || git clone "$REPO" "$WORK"

# --- 1. delivery r13 (16 blocks) ---------------------------------------------------------
cd ~/llama.cpp
git fetch --all
git checkout ebbb18522                       # the fork point (release.json.base)
git checkout -b closing-gfx1100
bash "$WORK"/scripts/apply-all.sh .          # main's r13 release.json; 16/16 git am
git rev-parse HEAD^{tree}                    # expect bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12

# --- 2. beta/mmb-general (12) ------------------------------------------------------------
git am "$WORK"/beta/mmb-general/patches/*.patch
git rev-parse HEAD^{tree}                    # expect 79136a15cac1920c0dd334b4c119a9cb42f9143b

# --- 3. archive/work/closing-the-gap (0015 removed; it is r13 block 00) ----------------------------
git fetch origin gap-closing
mkdir -p /tmp/closing-patches
git archive origin/gap-closing archive/work/closing-the-gap/patches | tar -x -C /tmp --strip-components=3
for p in /tmp/closing-patches/0*.patch; do
  git am "$p"
done
git rev-parse HEAD^{tree}                    # expect 1f09fd97d916ca080f7f65cdc422a3d6c425baa7
```

Notes:

* **`0015` was removed** (the shared-NextN MTP fix is in the r13 block-00 base).
* **`0024` must be applied before `0025`** — `0025` reverts `0024`'s `src/llama-model.cpp` heuristic.
* The applied tree is `1f09fd97d916ca080f7f65cdc422a3d6c425baa7` (the gfx1100 RDNA3_0 arms are
  folded into `0016`/`0003`, so this is the arch-complete 25-patch set — no separate overlay).
  Record the actual `From <sha>`/tree.
* Keep a **second worktree at r13+beta** (step 2, before the closing patches) as the local
  pre-closing baseline for §6.0.  Build it once and keep it warm.
* `HIP_VISIBLE_DEVICES=0` on **every** model/bench command.

## 3. Build both trees

```sh
cd ~/llama.cpp
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
#   fast loop: cmake --build build-rocm --target llama-cli llama-bench llama-perplexity \
#     test-backend-ops test-logits-width-probe llama-batched-bench llama-imatrix -j 16
```
Build **both** trees.  The concern on gfx1100 is **compilation/instantiation** of the new kernels
(`gdn-conv.cu`, `ple-conv.cu`, `norm-gated.cu`, `indexer-score.cu`, `indexer-topk.cu`,
`lightning-indexer.cu`, `fattn-qsa3.cu`) for the gfx11 target — a template type axis that is not
instantiated for gfx1100 will compile into the dispatch TU and blow the build time (§5).

---

## 4. What is architecture-sensitive

"Arch-scoped" = explicit RDNA3_5/RDNA4/RDNA3_0 branch or a gfx1151-tuned constant.  On gfx1100 the
key facts are: **`MMB` is now default-ON with the RDNA3_0 row**, and **`HC16`/`gatemix` are inert**
(they are `GGML_CUDA_CC_IS_RDNA3_5`-gated at their call sites).

### 4a. Testable end-to-end on this box

| patch | what | gfx1100 expectation | what to do |
|---|---|---|---|
| `0002` default beneficial features ON | **MMB default ON** + RDNA3_0 arm ON | the biggest gfx1100 change | verify `MMB_CFG` shows the gfx1100 row (§6.5); recover the S5-S10 headline **with no env**; A/B `GGML_CUDA_MMB=0` |
| `0004` depthwise conv1d (GDN + PLE) | `gdn-conv.cu` + `ple-conv.cu` | arch-neutral; **GDN applies to 27B/35B**, PLE is qwen4exp | compile for gfx11; coherence + prefill A/B on 27B/35B |
| `0006` narrow-row RMS norm | `norm-gated.cu::rms_rows_f32` | arch-neutral | width purity + coherence + a small prefill delta (S4 was ~+0.3 % at `-ub 4096`) |
| `0009` `-lzm auto` + managed PLE reader | loader / `llama-model` | arch-neutral (RAM-based) | lazy-mode text identity on a shard model; `-lm none -lzm on` paths |
| `0010` MoE BF16 epilogue (`GGML_CUDA_MMB_DOWN16`) | `moe-weighted-reduction.cu` | **default OFF** | verify default-off no-op; if enabled on 35B-A3B, PPL/coherence + decode |
| `0017` MMB quant coverage Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 | `mmb.cu` weight-type mask | **default ON on non-RDNA4** → **this is gfx1100 work**, not gfx1201 | `MUL_MAT`/`MUL_MAT_ID` oracles for the five types; if a model of that type is on the box, a pp A/B |
| `0018` MMB quant coverage IQ2_S/IQ2_XS/IQ2_XXS | `mmb.cu` weight-type mask | same | oracles (`IQ2_S 14/14`, `IQ2_XS 4/15/75`-class) + a model if present |
| `0019` HC16 F32-elision eval-callback fix | scheduler + `mmb.cu` | **HC16 is RDNA3_5-gated → inert here**, but the fix is generic | `llama-imatrix` NanBeige BF16 clean; confirm `MMB_CFG hc16=1` does not dispatch HC16 on gfx1100 |
| `0023` MMB HC16 per-context state | `ggml-backend.cpp` + `mmb.cu` | inert here (HC16 gated), but the scheduler change is live | MTP determinism at depth (27B/35B); no width-purity change |
| `0025` host-buffer input layer | `ggml-cuda.cu` + `ggml-backend.cpp` | **discrete GPU → `prop.integrated = 0` → no-op** | verify the input layers stay on CPU and CPU/VRAM is unchanged; `GGML_FORCE_NO_INTEGRATED=1` identical |

### 4b. qwen4exp-only — **compile / oracle / host-gate only on this box**

These are the campaign's core qwen4exp prefill/attention wins.  You cannot run them end-to-end (no
model fits).  Do **not** mark them broken; verify what is verifiable:

| patch | what | gfx1100 check |
|---|---|---|
| `0001` hc_combine_norm matcher revival | qwen4exp HC | compiles; no gfx1100 dispatch (HC-only) |
| `0003` hc_gate_mix fusion | gate GEMM+sigmoid+mix | **not ported to RDNA3_0**; `gatemix=0` in the RDNA3_0 row → verify inert.  **Port candidate** (§7) |
| `0005` QSA block window by highest position | `llama-memory-hybrid-idx` | compiles; host-side only |
| `0007` QSA visibility fold | `fattn-qsa3.cu` | `FLASH_ATTN_QSA` 26/26 (the qsa3 arm runs on gfx1100; S10 verified) |
| `0008` M=4 HC inject out of the tall MMB tile | `mmb.cu` | compiles; HC shapes only |
| `0011` HC BF16 streams (`LLAMA_HC_BLK16`/`_RES16`) | `hyperconn.cu` + MMB | default OFF; compiles |
| `0012` `mmb_cvt` BF16 `out_xn` | `mmb.cu` | compiles; HC-combine only |
| `0013` prefill indexer relu+head-sum | `indexer-score.cu` | compiles; `LIGHTNING_INDEXER` oracle |
| `0014` QSA prefill scorer trim | `indexer-topk.cu` + CPU oracle | **`TOPK_QSA` 4/4** (the only end-to-end oracle for the qwen4exp top-k; S10 verified) |
| `0016` `QSA_SCORE_WMMA` fused indexer score | `lightning-indexer.cu` | **RDNA3_5-only predicate**; `LIGHTNING_INDEXER` oracle runs the generic fallback.  **Port candidate** (§7) |
| `0020` sparse MTP-draft attention | qwen4exp | compiles; memory/graph side |
| `0021` QSA derived indexer cache default ON | `llama-memory-hybrid-idx` | compiles; qwen4exp only |
| `0022` gfx1151 decode crossover 64K→32K | `qwen4exp.cpp` | **gfx1151-only** (`qsa_arch_gfx() == 0x1151`); on gfx1100 `1<<62` (dense-always) and qwen4exp cannot run → confirm inert / N/A |
| `0026` sparse MTP draft prefill default ON | `qwen4exp.cpp` + `llama-model.cpp` | compiles; qwen4exp only |

### 4c. The `beta/mmb-general` rows you are validating against

The 12 beta patches carry the RDNA3_0 row.  Confirm the resolved config (§6.5) and that the
RDNA4/RDNA3_5 rows did not leak.  Prior record: `gfx1100-s10-rebase-results.md` (13/13, tree
`cd306e6b60…`, S5-S7 reproduced).

* RDNA3_0 row: `dense_geom=0`, `f32split=0`, `routed=1`, `gatemix=0`.
* Closing `0002` flips the **master** `GGML_CUDA_MMB` default to ON; the per-arch row is the beta's.

---

## 5. Build-time instantiation check

Same discipline as `AGENTS.md`: a FA/KV *type* axis that is not instantiated for gfx11 compiles into
the dispatch TU and silently dominates the build.  After the build:

```sh
nm -C build-rocm/ggml/src/ggml-cuda/CMakeFiles/ggml-hip.dir/fattn-tile.cu.o | grep -c tile_case
# the dispatch TU must show 'U' for every KV type; template-instances/*.cu must show 'T'/'W'
```
A clean `-j16` build should be in the usual range; a single TU taking minutes means a type axis was
left implicit.

---

## 6. The test matrix

### 6.0 Establish the local baseline first

Build the **r13+beta** worktree and record, on this box (this is the pre-closing reference):

* same-seed greedy text for each §6.1 model, and
* `llama-bench` prefill/decode at `-b/-ub 2048` and `-b/-ub 4096` (MMB on **and** off), and
* the gfx1100 reference points from the prior records (below) as a smoke check.

Prior gfx1100 reference (S5-S10, `GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1`, `-r 5`, interleaved):

| model | point | MMB off | MMB on | Δ |
|---|---:|---:|---:|---:|
| 27B UD-Q4_K_M | pp8192 | 1021.96 | 1168.06 | **+14.3 %** |
| 27B UD-Q4_K_M | pp16384 | 981.17 | 1114.30 | **+13.7 %** |
| gemma-12B Q8_0 | pp8192 | 2153.97 | 2390.13 | **+11.0 %** |
| 35B-A3B (MoE) | pp8192 | 3669.58 | 3872.88 | **+5.5 %** |
| 35B-A3B | pp32768 | 2994.32 | 3136.41 | **+4.8 %** |
| gemma-26B-A4B | pp8192 | 3289.59 | 3289.37 | neutral |

S10 same-seed text (MMB on): 27B UD-Q4_K_M `140fe1b2d244` (225 ch), 35B-A3B Q3_K_M
`5a3bb565f0ad` (206 ch).  MTP smoke (S10): 27B acceptance **0.78070**, 35B **0.72917**.

### 6.1 Models that fit (24 GiB) and how to use them

| model | role |
|---|---|
| `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` | dense prefill/MMB headline; **built-in nextn → no `-md`** |
| `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` | dense, smaller (fits with more context); MMB |
| `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` | dense Q8_0; MMB dense tile; FA head-256 cap check |
| `/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | MoE prefill/MMB; **built-in nextn → no `-md`** |
| `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | MoE; the F32-router split isolate (`f32split=0`) |
| `/llm/models/NanBeige/Nanbeige4.2-3B-BF16.gguf` | `llama-imatrix` gate (`0019`/`0023`) |

(Paths are the ones the gfx1100 records used — `ls` to confirm the box layout; they may be under a
slightly different directory on this host.)

qwen4exp **cannot** be loaded here.  Run all commands with `HIP_VISIBLE_DEVICES=0`.

### 6.2 Same-seed coherence (per model)

```sh
HIP_VISIBLE_DEVICES=0 llama-cli -m "$M" -ngl 99 -fa auto -ctk f16 -ctv f16 -c 8192 -n 48 \
  --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
  -f prompts/prose-rdna-boosts.txt > /tmp/coh.log 2>&1
python3 <repo>/scripts/extract-generated.py /tmp/coh.log
```
Run at shallow and at `-c 32768`/`-c 65536` with the matching prompt (recipes in
`closing-the-gap.md`: `head -c 165000 wiki.train.raw > /tmp/p40k.txt`, etc.).

### 6.3 Intra-build purity (the real contract)

`--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3` must be **byte-identical** on the
dense 27B and the MoE 35B.  At depth always `--ctx-checkpoints 0`.  (qwen4exp is N/A here.)

### 6.4 Width probe

```sh
HIP_VISIBLE_DEVICES=0 test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512
# expect: width_purity=PASS (worst maxdiff 0)
```
Run it with **MMB on (default)** on 27B UD-Q4_K_M, 35B-A3B Q3_K_M, gemma-12B Q8_0, gemma-26B-A4B.
The campaign also ran P=32768, which needs the local `tests/test-logits-width-probe.cpp` extension.

### 6.5 MMB config dump + the default-on recovery

```sh
HIP_VISIBLE_DEVICES=0 GGML_CUDA_MMB_CFG=1 llama-bench -m <model> -ngl 99 -p 2048 -n 0
```
Expected gfx1100 row (after the closing patches; **no env needed — MMB is default ON**):

```
MMB_CFG cc=0x1001100 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=0(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=1 down16=0 gatemix=0
        blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=1
```
(`hc16=1` is the closing `0002` flip but is **inert** on gfx1100 — the call site is RDNA3_5-gated.
`f32split=0` is the RDNA3_0 row from beta `0012`; `dense_geom=0`, `routed=1` are RDNA3_0.)
Then interleaved A/B `GGML_CUDA_MMB=0` vs default and confirm the S5-S10 headline **with no env**.

### 6.6 Op oracles (fresh build, separate stdout/stderr)

```sh
for op in FLASH_ATTN_QSA GATED_DELTA_NET TOPK_QSA FLASH_ATTN_EXT LIGHTNING_INDEXER MUL_MAT MUL_MAT_ID; do
  HIP_VISIBLE_DEVICES=0 test-backend-ops -o $op > /tmp/orc-$op.out 2>/tmp/orc-$op.err
done
```
* **`TOPK_QSA`** is the op's name on this tree (some records call it `INDEXER_TOPK`).
* gfx1100 prior results: `FLASH_ATTN_QSA` **26/26**, `GATED_DELTA_NET` **46/46**, `TOPK_QSA` **4/4**,
  `FLASH_ATTN_EXT` **5953/5953**.
* `0017`/`0018` type oracle expectations: `MUL_MAT` 48/47/14/46/45 (Q4_0/Q4_1/Q5_0/MXFP4/NVFP4),
  `MUL_MAT_ID` 74/75/3/74/73; IQ2_S/IQ2_XS/IQ2_XXS `MUL_MAT` 14/14/46 and `MUL_MAT_ID` 4/15/75.
* `LIGHTNING_INDEXER` (the `0016` oracle) — on gfx1100 the RDNA3_5 WMMA predicate is false, so this
  exercises the **generic** path; record whether the gfx11 kernel *compiles* for gfx1100 (it is
  guarded by `AMD_WMMA_AVAILABLE && RDNA3`, so it should).
* **Do not count `FLASH_ATTN_EXT` from a merged `2>&1` log** (ANSI + stream interleaving).

### 6.7 MTP (27B dense, 35B MoE)

Protocol A (`benchmarks/mtp-adaptive-methodology.md`): seed 42, temp 0, `-n 3000`, acceptance
**> ~0.45 at pos 1**, MTP ≥ plain, `plain == draft-mtp` byte-identical **within the build**;
reasoning pinned per axis.  Report `draft acceptance` from `--log-verbosity 4`.
Dense 27B / MoE 35B have **built-in `nextn` heads — do not pass `-md`**; a draft head cannot be
loaded standalone (the `plain` arm must not pass it either).

### 6.8 PPL parity

27B UD-Q4_K_M and 35B-A3B Q3_K_M, MMB on vs off.  Prior gfx1100 parity (S10): 27B 10.0174 → 9.9258,
35B 14.8302 → 14.8248.  MMB is a BF16 weight rounding → **parity**, not bit-identity.

### 6.9 `llama-imatrix` (`0019`/`0023`)

NanBeige4.2-3B-BF16 `-c 512 -b 512 --chunks 4`: clean (no non-finite) and the imatrix file
byte-identical to the `GGML_CUDA_MMB_HC16=0` run (HC16 is inert on gfx1100, so this is really a
scheduler/split regression check).

---

## 7. gfx1100 port candidates (the campaign's real gfx1100 work)

gfx1100 and gfx1151 share the **gfx11 WMMA builtin**, so a kernel marked "RDNA3_5" is usually a
*call-site predicate*, not a hardware limit.  The two candidates:

| candidate | why plausible | what a port needs |
|---|---|---|
| **`0016` `QSA_SCORE_WMMA`** | the kernel is guarded `#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)` (gfx1100 is RDNA3) but `supports_indexer4()` is `GGML_CUDA_CC_IS_RDNA3_5` | extend the predicate to `RDNA3_0` (the qsa3 patch `0011` did exactly this for `fattn-qsa3.cu`); the `flash_attn_qsa3`/`lightning_indexer` gfx11 fragment is already there.  Then `LIGHTNING_INDEXER` must pass on gfx1100 and `qwen4exp` prefill must be A/B'd — but **qwen4exp does not fit here**, so this port must be finished on gfx1201/gfx1151, or the kernel unit-tested here and the end-to-end left to them |
| **`0003` `hc_gate_mix`** | call-site predicate is `GGML_CUDA_CC_IS_RDNA3_5`; the kernel is IQ4_NL + RDNA3_5 WMMA | same shape: extend the call-site predicate to `RDNA3_0`, then unit-test.  qwen4exp end-to-end is again not possible here |

If you make either port, **do not enable it by default on gfx1100** until a gfx1151/gfx1201 session
has A/B'd it end-to-end; land it as a patch on the WIP branch and record it in
`gfx1100-porting.md` + this file's return.

Everything else that is RDNA3_5-only (`HC16`, `gatemix` as shipped) stays inert — record that with
the predicate.

### 7.1 Port result (2026-09-23, folded into the 25-patch set 2026-09-23)

Both candidates are **ported and folded into the canonical `0016` and `0003` patches** (the
25-patch set is now arch-complete: gfx1100/gfx1151/gfx1201).  The fork commits are `eede54ce2`
(`0016`) and `8b7c5fad3` (`0003`); the separate gfx1100 overlay patch was removed.  Neither arm is
enabled by default on gfx1100:

* **`0016` `QSA_SCORE_WMMA`** — `supports_indexer4()` now uses `indexer4_arch_enabled(cc)`:
  RDNA3_5 stays the shipped arm; RDNA3_0 (gfx1100) is **opt-in** via
  `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1` (default off).  The kernel compiles for gfx11 and
  **`LIGHTNING_INDEXER` 225/225 passes on gfx1100 with the opt-in set** (the WMMA arm, not the
  generic vec fallback).  End-to-end (qwen4exp prefill) is **not testable on gfx1100** (no model
  fits in 24 GiB) — deferred to gfx1151/gfx1201.
* **`0003` `hc_gate_mix`** — the call-site predicate is now `GGML_CUDA_CC_IS_RDNA3(cc)`
  (gfx1100 + gfx1151).  `gatemix` **stays default OFF on RDNA3_0** (`mmb_arch_defaults`), so the
  arm is **opt-in via `LLAMA_HC_GATEMIX=1`**.  The kernel (`hc_gate_mix_kernel<4>`, IQ4_NL +
  gfx11 WMMA) compiles for gfx1100 (verified by the clean build), but there is **no op-level
  oracle** (it is a graph fusion) and qwen4exp does not fit, so the unit test is compile-only —
  end-to-end deferred to gfx1151/gfx1201.

Record the predicates that keep everything else inert: `HC16`/`blk16`/`res16` are
`GGML_CUDA_CC_IS_RDNA3_5`-gated at their call sites; `0022` is `qsa_arch_gfx() == 0x1151` only
(gfx1100 = dense-always `1<<62`).

---

## 7.2 Remaining gfx1100 port / transfer work (2026-09-23 handover)

**Framing.**  The point of the gfx1100 pass is not "make the gfx1151-gated stuff not break" — it is
"**a feature gated to gfx1151 because it wins there: does it also win on gfx1100, and if so, port
and enable it**".  Classify each arch-scoped item as (1) a wrong-arch predicate over a working
kernel, (2) a marking that does not run, or (3) a platform-specific win.  Then run the four gates:
predicate (`grep` + `MMB_CFG`), **correctness** (oracle + `plain == draft-mtp`; a graph fusion has no
oracle), **perf** (`-b/-ub 4096`, warm, interleaved), fold into the owning patch.

**The gfx1100 asymmetry (good news).**  gfx1100 **shares the gfx11 WMMA builtin** with gfx1151, so:

* there is **no gfx12 fragment port** to do — `mmb.cu`, `fattn-qsa3.cu` and `lightning-indexer.cu`
  already have gfx11 device code;
* gfx1100 is **single-GPU**, so the CUDA backend's `graph_optimize` **does run** — the HC16 /
  DOWN16 / blk16 / res16 markings are *measurable here*, unlike on the gfx1201 box's required
  `-sm tensor` mode where the meta backend bypasses them (that is the gfx1201 finding; it does not
  apply to a single-GPU gfx1100 box, only to a hypothetical 2× W7900 `-sm tensor` run).

**Reference results from the gfx1201 session (2026-09-23)** — the transfer lens in action:

| feature | gfx1201 result | implication for gfx1100 |
|---|---|---|
| `0016` `QSA_SCORE_WMMA` (4-head indexer WMMA) | **ported to RDNA4, +1.7/+3.2/+6.5/+12.3 %** qwen4exp prefill | gfx1100 shares the gfx11 WMMA — the equivalent port is the §7.1 opt-in; it should pay similarly if a box can run qwen4exp |
| `0010` DOWN16 + `0011` blk16/res16 | **no transfer** (flat under `-sm layer`; inert under `-sm tensor`) — APU/unified-memory bandwidth effect | single-GPU gfx1100 *can* run the markings; measure, do not assume either way |
| `0003` `hc_gate_mix` | RDNA4 epilogue needs the gfx12 acc map; not yet done | gfx1100's gfx11 kernel is the shipped one → the §7.1 opt-in is the end-to-end candidate |

### 7.2.1 Work item — MMB WMMA quant coverage (`0017`/`0018`) on gfx1100

`0017` adds Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 and `0018` adds IQ2_S/IQ2_XS/IQ2_XXS to the MMB weight-type
mask, and **both are default-ON on non-RDNA4** — so gfx1100 already runs them (that is the
`MUL_MAT`/`MUL_MAT_ID` oracle work and the `gemma-26B-A4B +11.4 %` row in §9.5).  This is *gfx1100's*
WMMA-quant work, not gfx1201's (RDNA4's mask is deliberately narrow).

Remaining: the same **per-(type, path, shape)** question the gfx1201 handover (`gfx1201-closed.md`
§12.1) lays out — the beta's RDNA4 numbers do **not** predict gfx1100 (the gfx11 MMQ path is much
weaker here, which is why MMB wins on gfx1100).  Concretely: for each of the eight types,
`llama-bench -p 8192,32768 -n 0 -b 4096 -ub 4096 -r 5` with `GGML_CUDA_MMB=0` vs default, dense
(`MUL_MAT`) and routed (`MUL_MAT_ID`, 35B-A3B), warm, interleaved; then the type oracle
(`MUL_MAT`/`MUL_MAT_ID`, expected counts in §6.6) and width purity.  Land any type that wins and is
not already covered.  `MXFP4`/`NVFP4`/IQ2 need a model of that type (quantize one).

### 7.2.2 Work item — finish the `0016`/`0003` ports end-to-end

§7.1 landed both as **opt-in** (`GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1`, `LLAMA_HC_GATEMIX=1`);
`LIGHTNING_INDEXER` 225/225 passes on gfx1100 with the WMMA arm, but **qwen4exp does not fit in
24 GiB**, so the end-to-end prefill A/B was never run.  Finish it on a box that can load qwen4exp:

* **A 2× W7900 (48 GiB each, gfx1100)** box is the natural target — qwen4exp has 2 KV heads so a
  pair is exactly right for `-sm tensor` (§9.9).  Run the gfx1201 `0016` A/B protocol there
  (`llama-bench -p 8192,32768,65536 -n 0 -b 4096 -ub 4096 -r 5`, WMMA vs fallback, warm) and, if it
  wins, flip the gfx1100 default (the gfx1201 session flipped RDNA4 to default-ON).
* Or finish it on the gfx1201/gfx1151 boxes with the gfx1100 kernel unmodified — but the win is
  hardware-specific, so the gfx1100 measurement should be made on gfx1100.

### 7.2.3 Work item — single-GPU lossy features (HC16 / blk16 / res16 / down16)

Because `graph_optimize` runs on this box, these engage here.  `0010` DOWN16 was already tested
(§9.8: 35B coherence identical, PPL 14.8248).  `0011` blk16/res16 and HC16 are **qwen4exp-only**, so
they are not end-to-end testable on a 24 GiB gfx1100 — same "no model fits" rule as the rest.  The
measurable gfx1100 action is: keep the `graph_optimize` marking path healthy (the `0019`/`0023`
scheduler+imatrix gates, §9.7) and, on a 2× W7900 box, re-check that the markings still run under
`-sm tensor` — they will **not**, per the gfx1201 meta-backend finding, so a W7900 qwen4exp run
would silently lose them unless Work item 3 of `gfx1201-closed.md` §12.3 is done.

### 7.2.4 Commands + traps

* `HIP_VISIBLE_DEVICES=0` on **every** command (the gfx1036 iGPU aborts multi-device tools).
* Warm the model in page cache before A/Bs; capture stdout/stderr to files (a piped bench can hang —
  §9.10); `pkill -9 -x llama-bench` between harnesses.
* Fold a port into its patch (`git commit --fixup` + `GIT_SEQUENCE_EDITOR=true git rebase
  --autosquash`), regenerate it, and re-verify the 26-patch apply tree.

### 7.2.5 Results (gfx1100, 2026-09-24) — §7.2.1 + §7.2.3 done

Built on the **new 26-patch set** (r13 + beta + `0001..0014`/`0016..0027`): fresh apply reproduces
tree **`803e6d908ade68b02a71af9d5d0cf605aec1382a`** (matches the gfx1201 §12.1 tree), 0 build errors.
The gfx1201 changes are **inert on single-GPU gfx1100** (`0004` single-device gate: fusion still
fires; `0016` RDNA4 arm: gfx11 path unchanged; `0017`/`0018` RDNA4 mask: gfx1100 mask unchanged;
`0027` meta pass: no meta backend) — all headline gates reproduce the §9 numbers.

#### §7.2.1 — MMB WMMA quant coverage (the gfx1100 asymmetry, confirmed)

`--pure` requantized models (`llama-quantize --allow-requantize --pure <Q8_0> out <TYPE> 16`),
`llama-bench -p 8192,32768 -n 0 -b 4096 -ub 4096 -r 3`, `GGML_CUDA_MMB=0` vs default:

| model | type | MMB off pp8192/pp32768 | MMB on | Δ | RDNA4 Δ (gfx1201 §12.1) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | Q4_0 | 6413.5 / 5077.9 | 7418.6 / 5701.3 | **+15.7 % / +12.3 %** | −2.9 % / −1.9 % |
| Qwen3.5-4B | Q4_1 | 6194.4 / 4940.6 | 7416.6 / 5687.1 | **+19.7 % / +15.1 %** | +5.1 % / +4.2 % |
| Qwen3.5-4B | Q5_0 | 6166.6 / 4925.4 | 7150.3 / 5527.6 | **+16.0 % / +12.2 %** | +10.4 % / +8.7 % |
| Qwen3.5-9B | Q4_0 | 3861.5 / 3302.4 | 4656.7 / 3910.3 | **+20.6 % / +18.4 %** | — |
| Qwen3.5-9B | Q4_1 | 3698.4 / 3208.8 | 4602.9 / 3874.6 | **+24.5 % / +20.7 %** | — |
| Qwen3.5-9B | Q5_0 | 3684.6 / 3203.9 | 4441.7 / 3758.4 | **+20.6 % / +17.3 %** | — |
| gemma-26B-A4B (Q4_0 **MoE**, routed) | Q4_0 | 4047.6 / 2746.3 | 4481.1 / 2951.9 | **+10.7 % / +7.5 %** | — |

**Every type wins on gfx1100**, including `Q4_0` — which *loses* on RDNA4 and is deliberately kept
out of RDNA4's dense mask.  The reason is the one §7.2 named: the **gfx11 MMQ path is much weaker**,
so MMB's WMMA wins for all three.  gfx1100 already enables all of them (routed mask
`IQ_FAMILY | K_AND_Q8 | Q4Q5 | IQ2`, dense mask `~0` — `mmb.cu:1817`/`1863`), so **no mask change is
needed**: the beta's non-RDNA4 policy was already right, and this confirms it.

* **Oracles** (`GGML_CUDA_MMB_MIN_T=1`, forces the MMB path at every shape): `MUL_MAT` **q4_0 48/48,
  q4_1 47/47, q5_0 14/14**; `MUL_MAT_ID` **q4_0 74/74, q4_1 75/75, q5_0 3/3** — the §6.6 counts.
* **Text gate** (the real contract): MMB off ↔ on is **byte-identical** (`ea43b94ecff1`, 96 ch) for all
  three types.
* **Width probe** at P=1024 shows the documented **coarse-quant relaxation** on the `--pure` models
  (4B Q4_0/Q4_1/Q5_0: MMB-off PASS/PASS/FAIL, MMB-on FAIL/FAIL/FAIL; worst 0.26–0.42).  MMB is gated
  `T >= 512`, the decode band `W=1..8` never takes it, and the text gate holds — same reading as the
  gfx1201 record (`GREEDY-PURITY.md` §36, the coarse-quant near-tie relaxation; *not* a kernel error,
  which the oracle + text + PPL all support).
* **Unreachable** (same tooling limits as gfx1201): `MXFP4` only via `MXFP4_MOE` (a MoE layout, no
  dense test model), no `NVFP4` path at all; `IQ2_*` — a matching 4B imatrix *was* generated
  (`--chunks 16` and `64`, both `gguf` and `--output-format dat`) and `--allow-requantize` used, but
  `llama-quantize` still refuses with `Missing importance matrix for tensor blk.32.attn_k.weight in a
  very low-bit quantization` — that tensor is not exposed as an imatrix-instrumented `MUL_MAT`, the
  same wall the gfx1201 record hit.  (Every *reachable* type wins, so IQ2 is expected to follow; it
  stays untested/unenabled rather than assumed.)

#### §7.2.3 — single-GPU lossy-marking path healthy

`graph_optimize` runs on this box, so the markings engage.  The `0019`/`0023` gate holds on the new
set: `llama-imatrix` Ornith-1.0-9B-BF16 `-c 512 -b 512 --chunks 4` is clean (PPL 7.2865, no
non-finite) and the imatrix file is **byte-identical** to `GGML_CUDA_MMB_HC16=0`.  `0010` DOWN16 was
already tested (§9.8).  `HC16`/`blk16`/`res16` stay qwen4exp-only (no model fits).

#### §7.2.2 — `0016`/`0003` end-to-end stays **deferred** (correctly)

Both remain **opt-in** on gfx1100 (`GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1`, `LLAMA_HC_GATEMIX=1`).
The gfx1201 session flipped **RDNA4** `0016` to default-ON on its own measurements; per §7.2.2 the
gfx1100 flip needs an **gfx1100** end-to-end A/B (a 2× W7900 box), which this 24 GiB box cannot do
(qwen4exp does not fit).  No flip made.

#### Standard gates re-verified on the 26-patch set

Oracles all green (`FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46, `TOPK_QSA` 4/4,
`FLASH_ATTN_EXT` 5953/5953, `LIGHTNING_INDEXER` 225/225, `MUL_MAT` 1297/1297, `MUL_MAT_ID`
913/913); 27B MMB **+18.7 % / +18.3 %** pp8192/16384 (off 1113.7/1064.0 → on 1321.5/1259.0), width
`PASS`, purity `e7ff203db696`; 35B purity `81d218ce9f5b`; PPL parity 27B 10.0174→9.9258, 35B
14.8302→14.8248 (all identical to the §9/§9.4/§9.5 numbers).

---

## 8. Per-patch verdict table (fill this in)

| patch | end-to-end on gfx1100? | verdict | evidence / why inert |
|---|---|---|---|
| `0001` | no (qwen4exp) | compile-only | `hc_combine_norm` matcher revival; no gfx1100 dispatch (HC-only); clean build |
| `0002` | **yes** | **PASS** | MMB default ON + RDNA3_0 arm; `MMB_CFG` row confirmed; S5-S10 wins recovered **with no env** (§9) |
| `0003` | no (qwen4exp) | **PORTED (opt-in)** | call-site predicate extended to RDNA3; `gatemix=0` stays default on RDNA3_0; kernel compiles for gfx11 (§7.1) |
| `0004` | **yes** | **PASS** | `gdn-conv.cu`/`ple-conv.cu` compiled for gfx11; 27B/35B coherence green; prefill win folded into the MMB A/B |
| `0005` | no (qwen4exp) | compile-only | host-side only (`llama-memory-hybrid-idx`); clean build |
| `0006` | **yes** | **PASS** | `norm-gated.cu::rms_rows_f32`; width purity PASS on all four models; coherence green |
| `0007` | no (qwen4exp) | **PASS (oracle)** | `FLASH_ATTN_QSA` **26/26** (the qsa3 arm runs on gfx1100; S10 verified) |
| `0008` | no (qwen4exp) | compile-only | M=4 HC inject out of the tall MMB tile; HC shapes only |
| `0009` | **yes** | **PASS** | lazy-mode text identity: `-lm auto/none × -lzm auto/on/off` all = `533172aeb7ab` |
| `0010` | **yes** (MoE) | **PASS** | default OFF (`down16=0`); enabled (`GGML_CUDA_MMB_DOWN16=1`): 35B coherence `81d218ce9f5b` identical, PPL 14.8248 |
| `0011` | no (qwen4exp) | compile-only | default OFF (`blk16=0`/`res16=0`); compiles |
| `0012` | no (qwen4exp) | compile-only | `mmb_cvt` BF16 `out_xn`; HC-combine only |
| `0013` | no (qwen4exp) | compile-only (oracle) | `LIGHTNING_INDEXER` generic path exercised |
| `0014` | no (qwen4exp) | **PASS (oracle)** | `TOPK_QSA` **4/4** (the qwen4exp top-k oracle; S10 verified) |
| `0016` | no (qwen4exp) | **PORTED (opt-in)** | `supports_indexer4` extended to RDNA3_0; `LIGHTNING_INDEXER` **225/225** with the WMMA kernel on gfx1100 (§7.1) |
| `0017` | **yes** | **PASS** | `MUL_MAT` **1297/1297**, `MUL_MAT_ID` **913/913**; gemma-26B-A4B +11.4 % (new Q4_0 coverage) |
| `0018` | **yes** | **PASS** | same oracles; IQ2 family covered |
| `0019` | **yes** | **PASS** | `llama-imatrix` clean (no non-finite), file byte-identical to `GGML_CUDA_MMB_HC16=0` |
| `0020` | no (qwen4exp) | compile-only | sparse MTP-draft attention; memory/graph side |
| `0021` | no (qwen4exp) | compile-only | QSA derived indexer cache; qwen4exp only |
| `0022` | no (qwen4exp) | inert | `qsa_arch_gfx() == 0x1151` only; on gfx1100 `1<<62` (dense-always) → N/A |
| `0023` | **yes** | **PASS** | MTP determinism + width purity unchanged; `hc16=1` inert (RDNA3_5-gated); scheduler change live; imatrix clean |
| `0024` | **yes** (no-op) | superseded | input-layer heuristic; superseded by `0025` (kept as A/B baseline) |
| `0025` | **yes** | **PASS (no-op)** | discrete GPU → `prop.integrated=0` → no-op; `GGML_FORCE_NO_INTEGRATED=1` identical (`533172aeb7ab`) |
| `0026` | no (qwen4exp) | compile-only | sparse MTP draft prefill default ON; qwen4exp only |

For each qwen4exp-only patch, state the **predicate** that makes it inert and the oracle/host-gate
you ran instead.

---

## 9. Session results (gfx1100, 2026-09-23)

**Box:** 1× RX 7900 XTX (gfx1100, 24 GiB), ROCm 7.14, `HIP_VISIBLE_DEVICES=0` on every command.
**Trees:** closing = `~/llama.cpp` `rdna-boosts` (r13 + beta + 25 closing patches, applied tree
`1f09fd97d916ca080f7f65cdc422a3d6c425baa7`, with the gfx1100 RDNA3_0 arms folded into `0016`/`0003`).
Baseline =
`~/llama-r13beta` (r13 + beta, applied tree `79136a15cac1920c0dd334b4c119a9cb42f9143b`).  Both built
with 0 errors.  Models available on this box: 27B UD-Q4_K_M, 35B-A3B Q3_K_M, gemma-12B Q8_0,
gemma-26B-A4B qat-UD-Q4_K_XL, Ornith-1.0-9B-BF16.  **Missing:** 27B IQ3_S (§6.1) and NanBeige BF16
(§6.9) — the imatrix gate used Ornith-1.0-9B-BF16 as the BF16 substitute.

### 9.0 §5 build-time instantiation check

`fattn-tile.cu.o` (dispatch TU) shows **96 `U`** tile_case externs; instance TUs define them.  No
type axis left implicit in the dispatch TU.  Clean `-j16` build green (closing + baseline).

### 9.1 §6.6 op oracles (closing tree, fresh build)

| op | result |
|---|---|
| `FLASH_ATTN_QSA` | **26/26** |
| `GATED_DELTA_NET` | **46/46** |
| `TOPK_QSA` | **4/4** |
| `FLASH_ATTN_EXT` | **5953/5953** |
| `LIGHTNING_INDEXER` | **225/225** (generic path; WMMA arm see §7.1) |
| `MUL_MAT` | **1297/1297** |
| `MUL_MAT_ID` | **913/913** |

### 9.2 §6.5 MMB config dump

```
MMB_CFG cc=0x1001100 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=0(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=1 down16=0 gatemix=0
        blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=1
```
Matches the expected gfx1100 row exactly (`hc16=1` is the `0002` flip but inert; `f32split=0`,
`dense_geom=0`, `routed=1`, `gatemix=0` are the RDNA3_0 row).  `GGML_CUDA_MMB=0` prints no row.

### 9.3 §6.2/§6.3/§6.4 correctness (closing tree, MMB default ON)

| gate | 27B UD-Q4_K_M | 35B-A3B Q3_K_M | gemma-12B Q8_0 | gemma-26B-A4B |
|---|---|---|---|---|
| coherence (`-n 48`, c=8192) | `533172aeb7ab` | `80aab0c0c53a` | `1b46f381feea` | `d21969fddd83` |
| purity plain == draft-mtp n3 (`-n 96`) | `e7ff203db696` == | `81d218ce9f5b` == | — | — |
| width probe (P=1024, ub=512) | PASS (0) | PASS (0) | PASS (0) | PASS (0) |

Depth purity (27B, `-c 40000 --ctx-checkpoints 0`, p40k prompt): plain == draft-mtp n3 =
`758fe22a91d9` (957 ch) byte-identical.

### 9.4 §6.8 PPL parity (MMB off → on, prose `-c 2048`)

| model | MMB off | MMB on | prior S10 |
|---|---|---|---|
| 27B UD-Q4_K_M | 10.0174 ± 0.62345 | 9.9258 ± 0.61417 | **identical** |
| 35B-A3B Q3_K_M | 14.8302 ± 1.00741 | 14.8248 ± 1.00481 | **identical** |

### 9.5 §6.0 A/B prefill (closing tree, `-b 4096 -ub 4096 -r 5`, MMB off vs on)

| model | point | MMB off | MMB on | Δ | S5-S10 prior Δ |
|---|---:|---:|---:|---:|---:|
| 27B UD-Q4_K_M | pp8192 | 1107.66 | 1321.32 | **+19.3 %** | +14.3 % |
| 27B UD-Q4_K_M | pp16384 | 1064.03 | 1257.94 | **+18.2 %** | +13.7 % |
| gemma-12B Q8_0 | pp8192 | 1900.35 | 2140.28 | **+12.6 %** | +11.0 % |
| 35B-A3B Q3_K_M | pp8192 | 5456.12 | 6406.97 | **+17.4 %** | +5.5 % |
| 35B-A3B Q3_K_M | pp32768 | 4281.18 | 4848.20 | **+13.2 %** | +4.8 % |
| gemma-26B-A4B | pp8192 | 3994.11 | 4451.20 | **+11.4 %** | neutral (new Q4_0 coverage) |

Decode parity: tg128 27B MMB off 40.01 → on 40.00 (untouched, `mmb_min_t=512`).
**Every S5-S10 win is recovered with no env (MMB default ON); the closing patches (GDN `0004`,
MMB quant coverage `0017`/`0018`) widen the wins beyond the beta-only figures.**

### 9.6 §6.7 MTP acceptance (`-n 3000`, draft-mtp n3, `-lv 4`)

| model | draft acceptance | prior S10 smoke |
|---|---|---|
| 27B UD-Q4_K_M | **0.82030** (1762/2148, mean len 3.46) | 0.78070 |
| 35B-A3B Q3_K_M | **0.75073** (1286/1713, mean len 3.25) | 0.72917 |

### 9.7 §6.9 `llama-imatrix` (Ornith-1.0-9B-BF16, `-c 512 -b 512 --chunks 4`)

Clean (no non-finite), PPL 7.2865 both arms; imatrix file **byte-identical**
(sha256 `e7c16342829553f66835b3d0747270cfc8a9d4adeca70246720a27e9fa132cdb`) for default vs
`GGML_CUDA_MMB_HC16=0` (HC16 is inert on gfx1100, so this is the scheduler/split regression check).

### 9.8 §4a patch-specific checks

* `0009` lazy mode: `-lm auto/none × -lzm auto/on/off` all byte-identical `533172aeb7ab`.
* `0010` MoE BF16 epilogue: default OFF (`down16=0`); enabled → 35B coherence identical
  (`81d218ce9f5b`), PPL 14.8248.
* `0025` host-buffer input layer: discrete GPU → `prop.integrated=0` → no-op;
  `GGML_FORCE_NO_INTEGRATED=1` identical `533172aeb7ab`.

### 9.9 qwen4exp gfx1100 static review (best-guess; end-to-end not runnable here)

A static pass over the qwen4exp-only path for a hypothetical **2× W7900 (48 GiB each, gfx1100)**
box that *can* load qwen4exp.  Verdict: **nothing correctness-untoward**; the arch-specific pieces
are performance tuning, and two things deserve an explicit flag.

**Arch policy (what a W7900 gets):** `qsa_arch_gfx()` reads the device gfx id, so on gfx1100 every
`qsa_arch_gfx() == 0x1151` branch is false and qwen4exp inherits the **gfx1201 policy**, not gfx1151:

| policy | gfx1151 (shipped) | gfx1100 (W7900) |
|---|---|---|
| `LLAMA_QSA_DENSE_SHORTCUT` | off (always-QSA) | **on** (first ~2051 tokens dense) |
| decode crossover (`0022`) | sparse ≥32K | **dense always** (`1<<62`) |
| prefill | QSA always | QSA always (same) |

**Flag 1 — qsa3 is ported but the policy wasn't re-measured.** `fattn-qsa3.cu` is gated
`RDNA3_0 || RDNA3_5 || RDNA4` (beta `0011`), and `FLASH_ATTN_QSA` 26/26 passes here — but the
`qwen4exp.cpp` arch-policy comment still says the dense-shortcut default was chosen because qsa3
wasn't ported, ending "**Revisit per arch once qsa3 is ported** — it is what made always-QSA viable
on gfx1151".  That revisit is now overdue: qsa3 *is* enabled on gfx1100, so always-QSA + sparse
decode (the gfx1151 winners) are plausibly wins here too but unmeasured.  **Recommend re-measuring
`LLAMA_QSA_DENSE_SHORTCUT=0` / `LLAMA_QSA_DENSE_DECODE_UNTIL=0` on gfx1100** — the only spot where
the gfx1100 default is a guess, not a measurement.

**RDNA3_5-gated → inert on gfx1100 (perf only, correct fallback):**

| item | gate | gfx1100 effect |
|---|---|---|
| HC16 (beta `0004`) | `GGML_CUDA_CC_IS_RDNA3_5` at both marking passes | inert → +4-5 % gfx1151 win lost |
| hc_gate_mix (`0003`) | now `RDNA3` (ported §7.1) but `gatemix=0` on RDNA3_0 | off; `LLAMA_HC_GATEMIX=1` opt-in |
| QSA_SCORE_WMMA (`0016`) | `RDNA3_5` shipped; §7.1 port opt-in | generic vec fallback; `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100=1` opt-in |
| HC BF16 streams (`0011` blk16/res16) | default off on all arch | opt-in only |
| MoE BF16 epilogue (`0010` down16) | default off on all arch | opt-in only |

**Arch-neutral (fire on gfx1100, oracle-backed):** hc_mix / hc_combine_norm (`0001`),
indexer top-k/score/fill (`0013`/`0014`/`0021`), mmb_cvt `out_xn` (`0012`), M=4 HC inject (`0008`),
sparse MTP draft (`0020`/`0026`), QSA block window (`0005`) — all shape/type predicates, no
gfx1151-specific gating.

**Multi-GPU (2× W7900):** `llm_arch_supports_sm_tensor(QWEN4EXP)` returns **true under
`GGML_USE_HIP`** (validated on 3× R9700).  qwen4exp has **2 KV heads**, so 2 GPUs is the natural
`-sm tensor` split (the "2 KV heads < 3 devices" meta-splitter abort only bites 3+ GPUs; a pair is
exactly right).  No single-device hardcoding in the QSA/indexer/hyperconn path — the `n_stream == 1`
gates are KV *sequence* streams (M-RoPE/multi-seq), not device count; `devices[0].cc` reads are
harmless on a homogeneous pair.

### 9.10 Cross-tree A/B (baseline `~/llama-r13beta` vs closing `~/llama.cpp`)

Same protocol (`-b 4096 -ub 4096 -r 5`); baseline MMB toggled with
`GGML_CUDA_MMB=1 GGML_CUDA_MMB_RDNA3=1` (beta default off), closing with `GGML_CUDA_MMB=0`
(closing default on).

| model | point | baseline MMB off | closing MMB off | baseline MMB on | closing MMB on |
|---|---|---:|---:|---:|---:|
| 27B UD-Q4_K_M | pp8192 | 1102.77 | 1107.66 | 1306.30 | 1321.32 |
| 27B UD-Q4_K_M | pp16384 | 1053.67 | 1064.03 | 1242.06 | 1257.94 |
| 35B-A3B Q3_K_M | pp8192 | 5332.20 | 5456.12 | 6249.50 | 6406.97 |

**No regression anywhere; the closing patches add +0.4…+1.3 % (27B) / +2.3…+2.5 % (35B) on top of
the beta baseline**, on top of the MMB default flip.  Coherence cross-check (27B, `-n 48`):
baseline MMB off == closing MMB off == **`3dc4df7edbf5` (215 ch)** byte-identical (the non-MMB
closing patches are bit-transparent); closing MMB on = `533172aeb7ab` (the approved MMB
re-baseline, a different GEMM contraction).

*Trap:* one baseline 35B run hung (no CPU/GPU activity) under the `2>/dev/null | grep | head`
pipeline; the same command with stdout/stderr captured to files completed in ~40 s.  Prefer
file-captured stdout/stderr for these benches so a stall is debuggable rather than silent.

---

## 10. Report template

For each gate: the exact command, the build/tree, the `MMB_CFG` line, the numbers (interleaved order
for A/Bs), and the extracted hash where a text gate applies.  For a port candidate, the patch, the
predicate change, the oracle result, and the explicit "end-to-end not testable on gfx1100" note.
Use `benchmarks/mtp-adaptive-methodology.md` rule 0 for MTP and `-b/-ub 4096` for A/Bs.

---

## 11. Traps

1. **Mask the iGPU: `HIP_VISIBLE_DEVICES=0` on every command** (the gfx1036 device aborts
   multi-device tools).
2. **`gap-closing`'s `release.json` is r12** — apply the delivery from `main` (§2).
3. **Apply `0024` then `0025`** (`0015` was removed; it is r13 block 00).
4. **gfx1151 hashes are not targets here.**  Use the gfx1100 S1-S10 records + a local
   pre-closing-vs-closing A/B.
5. **`MMB` is now default-ON** — an A/B must set `GGML_CUDA_MMB=0` explicitly (the old "unset = off"
   is gone).
6. **`FLASH_ATTN_EXT` cannot be counted from a merged `2>&1` log.**
7. **No qwen4exp** — do not try to load the 94 GiB model; verify its patches by compile/oracle/gate.
8. **Never benchmark in parallel**; interleave arms in one warm session.
9. **Do not push the `~/llama.cpp` fork.**  Push the delivery repo only on explicit request.
