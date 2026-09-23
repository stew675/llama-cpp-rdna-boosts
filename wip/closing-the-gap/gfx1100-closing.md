# gfx1100 — closing-the-gap validation & porting brief

**Audience:** the agent working on the **single RX 7900 XTX (gfx1100, RDNA3_0)** box, 24 GiB VRAM.
**Goal:** apply the full delivery + `beta/mmb-general` + `wip/closing-the-gap` stack, **validate
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

The campaign is 25 patches (`wip/closing-the-gap/patches/0001..0014`, `0016..0026`) on top of the
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
> `wip/closing-the-gap` patches **but its `release.json` is stale at r13's predecessor (`r12`)**.
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

# --- 3. wip/closing-the-gap (25; SKIP 0015 which r13 block 00 supersedes) -----------------
git fetch origin gap-closing
mkdir -p /tmp/closing-patches
git archive origin/gap-closing wip/closing-the-gap/patches | tar -x -C /tmp --strip-components=3
for p in /tmp/closing-patches/0*.patch; do
  case "$p" in *0015-*) echo "skipping $(basename "$p") (superseded by r13 block 00)"; continue;; esac
  git am "$p"
done
git rev-parse HEAD^{tree}                    # expect 2b15ecd26c97afb4dbe2f58566def2180949df82
```

Notes:

* **`0015` must be skipped** (the shared-NextN MTP fix is in the r13 block-00 base).
* **`0024` must be applied before `0025`** — `0025` reverts `0024`'s `src/llama-model.cpp` heuristic.
* The applied tree is `2b15ecd26c97afb4dbe2f58566def2180949df82`.  Record the actual `From <sha>`/tree.
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

---

## 8. Per-patch verdict table (fill this in)

| patch | end-to-end on gfx1100? | verdict | evidence / why inert |
|---|---|---|---|
| `0001` | no (qwen4exp) |  |  |
| `0002` | **yes** |  |  |
| … |  |  |  |
| `0026` | no (qwen4exp) |  |  |

For each qwen4exp-only patch, state the **predicate** that makes it inert and the oracle/host-gate
you ran instead.

---

## 9. Report template

For each gate: the exact command, the build/tree, the `MMB_CFG` line, the numbers (interleaved order
for A/Bs), and the extracted hash where a text gate applies.  For a port candidate, the patch, the
predicate change, the oracle result, and the explicit "end-to-end not testable on gfx1100" note.
Use `benchmarks/mtp-adaptive-methodology.md` rule 0 for MTP and `-b/-ub 4096` for A/Bs.

---

## 10. Traps

1. **Mask the iGPU: `HIP_VISIBLE_DEVICES=0` on every command** (the gfx1036 device aborts
   multi-device tools).
2. **`gap-closing`'s `release.json` is r12** — apply the delivery from `main` (§2).
3. **Skip `0015`; apply `0024` then `0025`.**
4. **gfx1151 hashes are not targets here.**  Use the gfx1100 S1-S10 records + a local
   pre-closing-vs-closing A/B.
5. **`MMB` is now default-ON** — an A/B must set `GGML_CUDA_MMB=0` explicitly (the old "unset = off"
   is gone).
6. **`FLASH_ATTN_EXT` cannot be counted from a merged `2>&1` log.**
7. **No qwen4exp** — do not try to load the 94 GiB model; verify its patches by compile/oracle/gate.
8. **Never benchmark in parallel**; interleave arms in one warm session.
9. **Do not push the `~/llama.cpp` fork.**  Push the delivery repo only on explicit request.
