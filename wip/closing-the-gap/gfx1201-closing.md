# gfx1201 — closing-the-gap validation & porting brief

**Audience:** the agent working on the **3× Radeon AI PRO R9700 (gfx1201, RDNA4)** box.
**Goal:** apply the full delivery + `beta/mmb-general` + `wip/closing-the-gap` stack, then
**validate every arch-sensitive piece on RDNA4** and **port** anything that is RDNA3_5-only or
gfx1151-tuned.  It can run (almost) every model the gfx1151 campaign used, so this box is the
primary cross-arch validation for the campaign.

**Companion file:** [`gfx1100-closing.md`](gfx1100-closing.md) (single RX 7900 XTX, 24 GiB — cannot
run qwen4exp; different job).
**Source of truth for what each patch is:** [`closing-the-gap.md`](closing-the-gap.md),
[`README.md`](README.md) and the dated `2026-09-*` records in this directory.
**Delivery policy:** `AGENTS.md` (default-on policy, purity rules, **never push the `~/llama.cpp`
fork**).  Push the delivery repo only if the maintainer asks.

---

## 0. The one-paragraph summary

The campaign is 25 patches (`wip/closing-the-gap/patches/0001..0014`, `0016..0026`) on top of the
**r13 delivery (16 blocks)** + the **12 `beta/mmb-general` patches**.  It was developed and tuned
on **gfx1151 (RDNA3_5)**, and a few pieces are explicitly **RDNA3_5-only or gfx1151-tuned**.  Your
job is to prove that on **gfx1201 (RDNA4)** the tree is *correct and not a regression*, and to
**port or explicitly document** every arch-scoped piece.  The gfx1201 win expectation is the
`mmb-general` port's own record (`beta/mmb-general/gfx1201-porting.md`, S14: **+22 % whole-WIP vs
the delivery at depth** on qwen4exp IQ4_XS); the closing campaign is expected to *add* to that, not
to lose it.

> **Do not expect the gfx1151 hash values** (`3553e76d3a9e`, `8285d12d40ca`, `d140b40f0eee`, …).
> Those are gfx1151-specific.  On gfx1201 the gates are **intra-build** (`plain == draft-mtp`,
> width purity, MTP acceptance, PPL parity, and the S14 reference hashes as a smoke), plus a
> pre-closing-vs-closing A/B on this box (§6.0).

---

## 1. The machine

| | |
|---|---|
| GPU | **3× AMD Radeon AI PRO R9700 (gfx1201, RDNA4)**, no iGPU masking needed |
| Host | Ryzen 9 9950X3D2, 184 GiB RAM |
| ROCm | build: `/opt/rocm-7.14.1-gfx102X` (the build script's `ROCM_714`, supports `--offload-arch=gfx1201`); `/opt/rocm-7.14-gfx1201` is the older parallel install |
| Build | `cd ~/llama.cpp && BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (ccache; the `EXTRA_CMAKE_FLAGS` override is required with CMake ≥ 4.3) |
| Runtime | `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib:$LD_LIBRARY_PATH` |
| Multi-GPU rule | **`-sm tensor` + `GGML_CUDA_ALLREDUCE=hybrid` (default)** for all qwen4exp / 3-GPU runs |

### Models on this box

| model | role |
|---|---|
| `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | dense; needs `-lm none -lzm on`; **built-in nextn → no `-md`** |
| `/llm/models/Qwen3.8/27B/IQ3_S/Qwen3.8-27B-UD-IQ3_S.gguf` | dense; MMB; built-in nextn |
| `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` | dense; rule-5 batched verify gate |
| `/llm/models/Qwen3.6/35B-A3B/Q3_K_M/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | MoE prefill; built-in nextn |
| `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | MoE; built-in nextn |
| `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` | dense Q8_0; FA head-512/tile policy |
| `/llm/models/Gemma4/26B-A4B-QAT/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | MoE; F32-router isolate |
| `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` + `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | **qwen4exp (the closing campaign's headline)**; 3-GPU `-sm tensor` |
| `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` + `.../Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | qwen4exp alternative (the gfx1151 campaign's exact model) |
| `/llm/models/NanBeige/Nanbeige4.2-3B-BF16.gguf` | `llama-imatrix` gate |
| `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf`, `/llm/models/Qwen3.5/9B/Q8_0/Qwen3.5-9B-Q8_0.gguf` | small dense smoke |

(Paths are the ones the gfx1201 records used — `ls` to confirm the box layout; they may be under a
slightly different directory on this host.)

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
git checkout -b closing-gfx1201
bash "$WORK"/scripts/apply-all.sh .          # main's r13 release.json; 16/16 git am
git rev-parse HEAD^{tree}                    # expect bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12

# --- 2. beta/mmb-general (12) ------------------------------------------------------------
git am "$WORK"/beta/mmb-general/patches/*.patch
git rev-parse HEAD^{tree}                    # expect 79136a15cac1920c0dd334b4c119a9cb42f9143b

# --- 3. wip/closing-the-gap (25; SKIP 0015 which r13 block 00 supersedes) -----------------
#    the patches live on the `gap-closing` branch, which is NOT checked out here:
git fetch origin gap-closing
mkdir -p /tmp/closing-patches
git archive origin/gap-closing wip/closing-the-gap/patches | tar -x -C /tmp --strip-components=3
#    -> /tmp/closing-patches/0001-*.patch ... /tmp/closing-patches/0026-*.patch
for p in /tmp/closing-patches/0*.patch; do
  case "$p" in *0015-*) echo "skipping $(basename "$p") (superseded by r13 block 00)"; continue;; esac
  git am "$p"
done
git rev-parse HEAD^{tree}                    # expect 2b15ecd26c97afb4dbe2f58566def2180949df82
```

Notes:

* **`0015` must be skipped** (`-gate-MTP-shared-KV-detection…`): the shared-NextN MTP fix is in the
  r13 block-00 base, so the WIP patch is superseded.
* **`0024` must be applied before `0025`** — `0025` reverts `0024`'s `src/llama-model.cpp` heuristic
  and takes the host-buffer path instead.  Do not drop `0024`.
* The applied tree at the end is `2b15ecd26c97afb4dbe2f58566def2180949df82`.  Record the actual
  `From <sha>`/tree in your report.
* Keep a **second worktree at r13+beta** (§2 step 2, before the closing patches).  It is the local
  pre-closing baseline for §6.0.  Build it once and keep it warm.

## 3. Build both trees

```sh
cd ~/llama.cpp
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
#   fast loop after edits:
# cmake --build build-rocm --target llama-cli llama-bench llama-perplexity \
#   test-backend-ops test-logits-width-probe llama-batched-bench llama-imatrix -j 16
```
Build **both** trees (campaign + r13+beta baseline).  gfx1201 shares the gfx12 fragment shim that
`beta/mmb-general` already ported, so all closing kernels should compile unchanged — the risk is
**instantiation**, not new porting.

---

## 4. What is architecture-sensitive

This is the inventory to work through.  "Arch-scoped" = contains an explicit RDNA3_5/RDNA4/RDNA3_0
branch or a gfx1151-tuned constant; "neutral" = no arch branch (verify only).

### 4a. Closing patches (`wip/closing-the-gap/patches/`)

| patch | what | arch status on gfx1201 | what to do |
|---|---|---|---|
| `0001` hc_combine_norm matcher revival | qwen4exp HC prefill fusion | **qwen4exp-only**, not arch-gated | verify the fusion fires on Flash-Next IQ4_XS and is a win; A/B `LLAMA_FUSED_DSV4_HC_POST=1` (forces the slower op) |
| `0002` default beneficial features ON | MMB default ON, RDNA3_0 arm ON, HC16 default 1 | **the main arch-sensitive flip** | verify `MMB_CFG` shows the gfx1201 row (§6.4); A/B `GGML_CUDA_MMB=0` |
| `0003` hc_gate_mix fusion | gate GEMM+sigmoid+mix | **RDNA3_5-only** (`GGML_CUDA_CC_IS_RDNA3_5` at the call site; `gatemix=0` on RDNA4) | verify it is **inert** on gfx1201 (`MMB_CFG gatemix=0`); a port is future work, not a blocker |
| `0004` depthwise conv1d (GDN + PLE) | `gdn-conv.cu` + `ple-conv.cu` | arch-neutral | must compile/instantiate on gfx12; coherence + oracles; qwen4exp (PLE) and 27B GDN |
| `0005` QSA block window by highest position | `llama-memory-hybrid-idx` | qwen4exp-only, neutral | qwen4exp long-context coherence (the image/2-D-position case on gfx1201 uses the same path) |
| `0006` narrow-row RMS norm | `norm-gated.cu::rms_rows_f32` | arch-neutral | width purity + coherence on a dense + MoE model |
| `0007` QSA visibility fold into `umask` | `fattn-qsa3.cu` | only used when qsa3 runs (all three arches) | `FLASH_ATTN_QSA` 26/26; qwen4exp prefill |
| `0008` M=4 HC inject out of the tall MMB tile | `mmb.cu` tall tile | MMB geometry — RDNA4 has its own dense tile | verify no regression on the RDNA4 dense path; qwen4exp HC shapes |
| `0009` `-lzm auto` semantics + managed PLE reader | loader / `llama-model` | arch-neutral (RAM-based) | 27B Q8_0 needs `-lm none -lzm on`; lazy-mode text identity |
| `0010` MoE BF16 epilogue (`GGML_CUDA_MMB_DOWN16`) | `moe-weighted-reduction.cu` | **default OFF**, MMB path | verify default-off is a no-op; if enabled, PPL/coherence on 35B-A3B |
| `0011` HC BF16 streams (`LLAMA_HC_BLK16`/`_RES16`) | `hyperconn.cu` + MMB | **default OFF**, qwen4exp HC | verify default-off no-op; if enabled, A/B on qwen4exp |
| `0012` `mmb_cvt` BF16 `out_xn` | `mmb.cu` | MMB path, neutral | verify it fires with MMB on and is not a regression |
| `0013` prefill indexer relu+head-sum fusion | `indexer-score.cu` | qwen4exp-only, neutral | `LIGHTNING_INDEXER` oracle + qwen4exp prefill |
| `0014` QSA prefill scorer trim | `indexer-topk.cu` + CPU oracle | qwen4exp-only, neutral | `TOPK_QSA` oracle (CPU oracle was added here) + qwen4exp prefill |
| `0016` `QSA_SCORE_WMMA` fused indexer score | `lightning-indexer.cu` | **RDNA3_5-only kernel** (`supports_indexer4` is `GGML_CUDA_CC_IS_RDNA3_5`) | verify the **generic fallback** is taken (no abort) and `LIGHTNING_INDEXER` passes; confirm `qwen4exp` prefill still wins.  A RDNA4 WMMA port is a candidate, not required |
| `0017` MMB quant coverage Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 | `mmb.cu` weight-type mask | **default OFF on RDNA4** (`CC_IS_RDNA4 ? IQ_FAMILY : …`) | verify the mask excludes them (`MMB_CFG`/dump) and there is no regression; testing them on RDNA4 is an **optional port** (the RDNA4 dense per-type policy enables only IQ3_S — see `gfx1201-s10-dense-geometry.md`) |
| `0018` MMB quant coverage IQ2_S/IQ2_XS/IQ2_XXS | `mmb.cu` weight-type mask | same as `0017` | same |
| `0019` HC16 F32-elision eval-callback fix | scheduler + `mmb.cu` | arch-neutral fix (HC16 itself is RDNA3_5-gated) | `llama-imatrix` clean + `in_sum2` byte-identical to `GGML_CUDA_MMB_HC16=0` |
| `0020` sparse MTP-draft attention | qwen4exp + `llama-memory-*` | qwen4exp-only; the **decode arm is opt-in** and the prefill arm became default-on in `0026` | qwen4exp MTP with the IQ4_XS sidecar: acceptance, `plain == draft-mtp`, depth sweep |
| `0021` QSA derived indexer cache default ON | `llama-memory-hybrid-idx` | qwen4exp-only, default ON | byte-identical A/B (`GGML_CUDA_QSA_INDEXER_CACHE=0`) + deep decode win |
| `0022` gfx1151 decode crossover 64K→32K | `qwen4exp.cpp` | **gfx1151-only** (`qsa_arch_gfx() == 0x1151 ? 32768 : 1<<62`) | verify **inert** on gfx1201 (dense-always decode); `LLAMA_QSA_DENSE_DECODE_UNTIL=0` forces sparse for an A/B |
| `0023` MMB HC16 per-context state | `ggml-backend.cpp` + `mmb.cu` | arch-neutral fix; HC16 RDNA3_5-gated | 128K MTP determinism; `llama-imatrix`; width purity |
| `0024` input layer on GPU (single device) | `llama-model.cpp` | **superseded by `0025`** | just apply it (it is reverted by `0025`); no separate test |
| `0025` host-buffer input layer | `ggml-cuda.cu` + `ggml-backend.cpp` | **APU-only effect** — on a discrete gfx1201 `prop.integrated = 0`, so `integrated=false` and the scheduler guard is not reached | verify **no-op**: input layers stay on CPU, no CPU/VRAM change; `GGML_FORCE_NO_INTEGRATED=1` must be identical |
| `0026` sparse MTP draft prefill default ON | `qwen4exp.cpp` + `llama-model.cpp` | qwen4exp-only | verify the draft context reports `MTP context uses a hybrid-idx memory`; `LLAMA_MTP_SPARSE=0` restores plain KV; deep prefill win + acceptance |

### 4b. Prerequisites that are already arch-scoped (do not re-port, but re-gate)

The 12 `beta/mmb-general` patches carry the RDNA4 rows you are validating against.  Confirm the
**resolved config** matches (`§6.4`), and that the gfx1100/gfx1151 rows did not leak into RDNA4.

* `mmb_arch_defaults(cc)` RDNA4 row: `dense_geom=1`, `routed=0`, `f32split=1`, `gatemix=0`.
* The closing `0002` flips the **master** `GGML_CUDA_MMB` default to ON; the per-arch row is the
  beta's and must be unchanged.

---

## 5. Build-time instantiation check (both arches do this)

The closing patches add kernels that, if a type/arm is not instantiated for the arch, compile into
the *dispatch TU* and silently slow the build (the `fattn-tile` lesson in `AGENTS.md`).  After the
build, check the FA/KV instance split the `AGENTS.md` way:

```sh
nm -C build-rocm/ggml/src/ggml-cuda/CMakeFiles/ggml-hip.dir/fattn-tile.cu.o | grep -c tile_case
# the dispatch TU must show 'U' (undefined) for every KV type; the generated
# template-instances/*.cu must show 'T'/'W'
```
Also confirm a clean `-j16` build time is in the usual range (a type axis left implicit in a
dispatch TU blows one TU up to minutes).

---

## 6. The test matrix

### 6.0 Establish the local baseline first

The campaign deliberately re-baselines prefill numerics (MMB on, qsa3, HC16).  Build the **r13+beta**
worktree and record, on this box:

* same-seed greedy text for each model (§6.1), and
* `llama-bench` prefill/decode at `-b/-ub 2048` and `-b/-ub 4096`, and
* the S14 reference hashes (below) — a smoke check that your box reproduces the campaign's RDNA4
  environment.

S14 gfx1201 reference hashes (from `beta/mmb-general/gfx1201-s14-gates.md`, **r12+beta** tree —
treat as smoke, not byte-gates for the r13+closing tree):

| model / command | hash |
|---|---|
| 27B Q8_0, `-n 20 -lm none -lzm on` | `da2e2d192e21` |
| Flash-Next, `-n 20 --reasoning off` | `359ff4337837` |
| 27B UD-IQ3_S, `-n 24` | `42cdf36d0633` |
| Flash-Next, `-n 24` | `d73f9238f6d6` |
| 35B UD-Q3_K_M, `-n 24` | `461ca8cd0e88` |

### 6.1 Same-seed coherence (25B; per model)

```sh
llama-cli -m "$M" -ngl 99 -fa auto -ctk f16 -ctv f16 -c 4096 -n 24 \
  --seed 42 --temp 0 --single-turn --no-display-prompt --reasoning off \
  -f prompts/prose-rdna-boosts.txt > /tmp/coh.log 2>&1
python3 scripts/extract-generated.py /tmp/coh.log
```
27B Q8_0 additionally needs `-lm none -lzm on` (otherwise the loader takes the vision path and the
CLI prints nothing).  Run at **two** depths: shallow (`-c 4096`) and `-c 32768`/`-c 65536` with the
matching prompt (`/tmp/p32k.txt`, `/tmp/p64k.txt`; recipes in `closing-the-gap.md`).

### 6.2 Intra-build purity (the real contract)

Same build, same state: `--spec-type none` vs `--spec-type draft-mtp --spec-draft-n-max 3` must be
**byte-identical** at 8K / 40K / 128K on every model.  This is the gate that catches a decode/verify
width impurity.  At depth **always** pass `--ctx-checkpoints 0`.

### 6.3 Width probe

```sh
test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512
# expect: width_purity=PASS (worst maxdiff 0)
```
The campaign also ran **P=32768**; that needs the local `tests/test-logits-width-probe.cpp` extension
(`n_ctx = P+NW+16`, token vector `1<<20`) that the gfx1151 session used — see `closing-the-gap.md`.

### 6.4 MMB config dump + MMB A/B

```sh
GGML_CUDA_MMB_CFG=1 llama-bench -m <model> -ngl 99 -p 2048 -n 0
```
Expected gfx1201 row (after the closing patches):

```
MMB_CFG cc=0x1001201 dense_geom=1 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
        f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=1 down16=0 gatemix=0
        blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=0
```
(`hc16=1` is the closing `0002` flip but is **inert** on RDNA4 — the call site is RDNA3_5-gated.
`gatemix=0` and `routed=0` are the RDNA4 policy rows.)  Then A/B `GGML_CUDA_MMB=0` vs default,
**interleaved**, and prefer `pp32768+` for the verdict.  The beta's S14 decomposition is the
reference: mmb-only **+6 %** at depth on Flash-Next IQ4_XS.

### 6.5 Op oracles (fresh build, separate stdout/stderr)

```sh
test-backend-ops -o FLASH_ATTN_QSA      > /tmp/orc-qsa.out 2>/tmp/orc-qsa.err
test-backend-ops -o GATED_DELTA_NET     > /tmp/orc-gdn.out 2>/tmp/orc-gdn.err
test-backend-ops -o TOPK_QSA            > /tmp/orc-topk.out 2>/tmp/orc-topk.err
test-backend-ops -o LIGHTNING_INDEXER   > /tmp/orc-li.out  2>/tmp/orc-li.err   # the 0016 oracle
test-backend-ops -o FLASH_ATTN_EXT      > /tmp/orc-fae.out 2>/tmp/orc-fae.err
```
* `TOPK_QSA` is the op's name (some records call it `INDEXER_TOPK`; on this tree it is `TOPK_QSA`).
* `FLASH_ATTN_EXT` is ~5953–5954 OK / **0 FAIL** — and **do not count it from a merged `2>&1` log**:
  the status is ANSI-wrapped and stderr interleaving orphans it.  Separate the streams and strip ANSI.

### 6.6 qwen4exp gates (this box's headline)

The campaign model is Flash-Next **IQ4_XS** (3 shards) + `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`, run
3-GPU `-sm tensor`.  Apply the x86/ROCm env and:

* **prefill**: `llama-bench -p 8192,32768,65536,98304 -n 0` on Flash-Next IQ4_XS, q8_0 KV, vs the
  r13+beta baseline — the S14 expectation is **+22 % at depth** for the beta alone; the closing
  patches should not lose it and `0012`/`0008` should add a little.
* **decode crossover** (`0022`): confirm gfx1201 stays **dense-always** (verify with
  `LLAMA_QSA_DENSE_DECODE_UNTIL=0` that sparse is *not* the default arm).
* **derived indexer** (`0021`): `GGML_CUDA_QSA_INDEXER_CACHE=0` must be byte-identical and slower at
  depth.
* **`QSA_SCORE_WMMA`** (`0016`): confirm the RDNA4 path is the generic fallback (no WMMA), that the
  `LIGHTNING_INDEXER` oracle is green, and that `LLAMA_QSA_SCORE_WMMA=0` is byte-identical.
* **sparse MTP** (`0020`/`0026`): `--spec-type draft-mtp --spec-draft-n-max 1/3`, `-n 3000`
  (`benchmarks/mtp-adaptive-methodology.md` Protocol A), acceptance **> ~0.45 at pos 1**, MTP ≥ plain,
  `plain == draft-mtp` byte-identical **within the build**.  The gfx1151 campaign gate depth is
  `-c 12288 -n 3000` prose `--reasoning off`; use the four-axis set (R/C/K/P, reasoning pinned) for
  the tuning verdict.
* **M-RoPE image case** (`0005`): if the vision projector is available, run an image + text prefill +
  MTP to exercise the "block window by highest position" fix (the gfx1151 repro was an image after
  12k tokens of text).

### 6.7 MTP (all model families)

* Dense 27B UD-IQ3_S and MoE 35B-A3B have **built-in `nextn` heads — do not pass `-md`**.
* qwen4exp has **no** built-in head — **must** pass the `mtp-*.gguf` sidecar.
* A draft head cannot be loaded standalone: the `plain` arm must **not** pass `-md` (it is a clean
  error, and a `-sm tensor` split trips a meta-split assert; pre-existing, not a regression).
* Report `draft acceptance` from `--log-verbosity 4` (or `-lv 4`).

### 6.8 PPL parity

`llama-perplexity` on 27B UD-IQ3_S, 35B UD-Q3_K_M and Flash-Next IQ4_XS (prose chunks), MMB on vs
off.  MMB is a BF16 weight rounding, so it is a **parity** gate (a layout error moves PPL by orders
of magnitude; +0.02–0.4 % is expected).

### 6.9 `llama-imatrix` (the `0019`/`0023` gate)

NanBeige4.2-3B-BF16 `-c 512 -b 512 --chunks 4`: clean (no non-finite), and the imatrix file
**byte-identical** to the `GGML_CUDA_MMB_HC16=0` run.  Note the absolute PPL is prompt-dependent —
compare the file, not the PPL, across arms.

---

## 7. Per-patch verdict table (fill this in)

| patch | fires on gfx1201? | verdict | evidence |
|---|---|---|---|
| `0001` |  |  |  |
| `0002` |  |  |  |
| … |  |  |  |
| `0026` |  |  |  |

For anything that does **not** fire (e.g. `0003`, `0016`, `0017`/`0018`, `0022`, `0025`), record
*why* (arch predicate / mask / APU-only) and whether a port is worth doing.

---

## 8. Known gfx1201 gaps / port candidates (from the campaign)

| item | status | note |
|---|---|---|
| `hc_gate_mix` (`0003`) | **not ported to RDNA4** | IQ4_NL + RDNA3_5 WMMA; `gatemix=0` on RDNA4.  Candidate port (the IQ4_NL WMMA is gfx11-shaped) |
| `QSA_SCORE_WMMA` (`0016`) | **RDNA3_5-only kernel**; generic fallback on RDNA4 | candidate port to RDNA4 WMMA; not a correctness blocker |
| MMB new quant types (`0017`/`0018`) | **masked off on RDNA4** | the RDNA4 dense per-type policy enables only IQ3_S; enabling Q4_0/… needs a per-type RDNA4 geometry + measurements (see `gfx1201-s10-dense-geometry.md`) |
| gfx1151 decode crossover (`0022`) | intentionally gfx1151-only | gfx1201 stays dense-always (measured flat ~8 % worse for sparse decode) |
| host-buffer input (`0025`) | **APU-only** | on a discrete GPU the flag stays false; the scheduler guard is not reached.  Nothing to port; just prove the no-op |

---

## 9. Report template

For each gate: the **exact command**, the **build/tree**, the **`MMB_CFG` line**, the **numbers**
(and the interleaving order for A/Bs), and the **extracted hash** where a text gate applies.  State
the gate name, not "it was slower".  If a patch is arch-inert, say so with the predicate that makes
it inert.  Use `benchmarks/mtp-adaptive-methodology.md` rule 0 (`-n 3000`, reasoning pinned) for any
MTP verdict, and the `-b/-ub 4096` protocol for A/Bs.

---

## 10. Traps

1. **`gap-closing`'s `release.json` is r12** — apply the delivery from `main` (§2).
2. **Skip `0015`; apply `0024` then `0025`.**
3. **gfx1151 hashes are not targets here.**  Use intra-build purity + the §6.0 baseline.
4. **`FLASH_ATTN_EXT` results cannot be counted from a merged `2>&1` log** (ANSI + stream
   interleaving).  Separate the streams.
5. **`-md` on a `plain` arm aborts** (draft head without a trunk).  Dense 27B / MoE 35B have
   built-in heads — do not pass `-md`; qwen4exp must.
6. **`--ctx-checkpoints 0` for anything at depth** — otherwise depth MTP is nondeterministic.
7. **Never benchmark in parallel**; interleave arms in one warm session.
8. **Do not push the `~/llama.cpp` fork.**  Push the delivery repo only on explicit request.
