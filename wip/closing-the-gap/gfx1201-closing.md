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

The campaign is 27 patches (`wip/closing-the-gap/patches/0001..0014`, `0016..0028`) on top of the
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
git rev-parse HEAD^{tree}                    # expect 1f09fd97d916ca080f7f65cdc422a3d6c425baa7
```

Notes:

* **`0015` must be skipped** (`-gate-MTP-shared-KV-detection…`): the shared-NextN MTP fix is in the
  r13 block-00 base, so the WIP patch is superseded.
* **`0024` must be applied before `0025`** — `0025` reverts `0024`'s `src/llama-model.cpp` heuristic
  and takes the host-buffer path instead.  Do not drop `0024`.
* The applied tree at the end is `1f09fd97d916ca080f7f65cdc422a3d6c425baa7` (the gfx1100 RDNA3_0
  arms are folded into `0016`/`0003`, so the 25-patch set is arch-complete — no separate overlay).
  Record the actual `From <sha>`/tree in your report.
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
| `0003` hc_gate_mix fusion | gate GEMM+sigmoid+mix | **ported to RDNA4 2026-09-24** (the call site now accepts RDNA4; `gatemix=1` on RDNA4) | fires on gfx1201; +5.3…5.8 % prefill; bit-identical text |
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
| `hc_gate_mix` (`0003`) | **PORTED to RDNA4 (2026-09-24, default ON)** | +5.3…5.8 % qwen4exp IQ4_NL prefill, bit-identical text.  gfx1100 stays opt-in pending an end-to-end A/B on a qwen4exp-capable box |
| `QSA_SCORE_WMMA` (`0016`) | **RDNA3_5-only kernel**; generic fallback on RDNA4 | candidate port to RDNA4 WMMA; not a correctness blocker |
| MMB new quant types (`0017`/`0018`) | **Q4_1 + Q5_0 ported (2026-09-23); Q4_0/MXFP4/NVFP4/IQ2 stay out** | RDNA4 dense now `IQ3_S + Q4_1 + Q5_0`; Q4_1/Q5_0 beat the delivery MMQ by +6..+14 % on 4B/9B/27B, Q4_0 loses.  MXFP4/NVFP4 cannot be produced by this box's quantizer; IQ2 needs an unmatched imatrix |
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

---

## 11. Session log

Newest first.  Each session appends its state, what it changed, and the next action.

### 2026-09-24 — session 8: the W=9 cliff is the MMVQ→MMQ family boundary, and it is fixed

**The correction.**  Session 7's J-selection hypothesis was wrong.  `rocprofv3 --kernel-trace` on
B=8 vs B=9 plus a `(type, J, ncols_dst, nchannels_y, ids)` dump proved the routed experts use
**J=16 in both arms** — the change is `MMVQ_MAX_BATCH_SIZE = 8`: at `n_tokens = 9` the routed
experts leave `mul_mat_vec_q_moe` for `mul_mat_q_routed_compact`/`mul_mat_q`, and the dense weights
leave `mul_mat_vec_q_ksplit` for `mul_mat_q`.  qwen4exp's weights have `nrows_x % 128 != 0`, so MMQ
selects `mul_mat_q_case`'s generic non-128-row **`fallback`** config (~3× the per-launch cost of the
ksplit kernel); 27B's rows are all ÷128, so MMQ is the *fast* config and it gains at the same
boundary — the jump and the dip are the same mechanism.

**The fix** (folded into the WIP set as
[`patches/0028`](patches/0028-gap-closing-WIP-extend-the-MMVQ-routed-expert-band-and-RDNA4-dense-fallback.patch),
commit `6b230ad59208`, 69 insertions): `MMVQ_MOE_MAX_BATCH_SIZE = 16` for the
routed-expert path (arch-independent AMD, whole supported verify range W ≤ 16), plus an RDNA4 rule
keeping ksplit MMVQ for `nrows % 128 != 0` dense shapes at 9..16.  Kill-switches
`GGML_CUDA_DISABLE_MMVQ_MOE_BAND=1` / `GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1`.

**Result (qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`, S_TG t/s, two interleaved
rounds):** B=8 unchanged (256.3/255.4 vs 255.7); **B=9 202.6 → 270.0/268.7**, B=10 226.1 →
289.0/288.2, B=11 245.0 → 302.0/301.7, B=12 264.7 → 324.5/324.0.  The B=8→9 dip is gone (B=9 is now
above B=8).  `n_max 8` MTP 94.3 → **102.7 t/s** at acceptance 0.6285; `n7` and `plain == n3`
(`3553e76d3a9e`) unchanged.  27B UD-Q4_K_XL keeps its pre-existing B=9 upward jump and is otherwise
unchanged; 35B-A3B Q3_K_M neutral.

**Not done:** the full `n7`/`n8`/adaptive matrix with the fix, per-type MoE-band tuning, and the
gfx1151/gfx1100 revalidation+port — the MoE band already fires on gfx1151 (arch-independent), the
dense band is RDNA4-gated.  Handover: [`gfx1151-closing.md`](gfx1151-closing.md).  Full evidence:
[`2026-09-24-qwen4exp-w9-verify-cliff.md`](2026-09-24-qwen4exp-w9-verify-cliff.md).

### 2026-09-24 — session 7: the qwen4exp W=9 verify cliff (the n7→n8 drop) root-caused (≈half)

**The observation (maintainer):** `draft-mtp n_max` 7→8 is an anomalously **sharp** drop
(t8/t7 ≈ 1.33–1.36 on every axis, constant) — the signature of a **path change**, not an
acceptance-curve effect.  Confirmed acceptance-free with `llama-batched-bench` (verify shape,
`OMP_WAIT_POLICY=PASSIVE`): qwen4exp `S_TG` B=8 **255.7** → B=9 **202.6**, recovering to 264.7 by
B=12; per-step time 32 ms at B=8 and a flat ~44 ms at B≥9 (a **fixed ~+12 ms/step once W≥9**).
**qwen4exp-specific**: 27B (`qwen35`) jumps *up* at B=9 (91.6→141.1) and 35B (`qwen35moe`) is flat
(433→430), so it is not the generic mmvq→mmq / FA switch.  Not acceptance (the spin pair was
byte-identical, and n8's mean accepted length is *higher*).

**Ruled out** (all with the verify shape): the QSA sparse top-k arm (the `N_KV<=2051` shortcut
covers it; `LLAMA_QSA_DENSE_PREFILL_UNTIL=1e9` is a no-op), the HC mixer (`LLAMA_FUSED_HC_MIX=0`
no-op — IQ4_NL's HC weights aren't Q8_0 so the fusion is off anyway), the FA tile→WMMA switch
(`GGML_CUDA_FA_WMMA_MAX_HEAD=0` leaves B=9 **bit-identical** at 1.422 s), the MoE fused GLU MMQ,
MMB, and fusions in general (`DISABLE_FUSION` shrinks both widths, ratio 0.848 vs 0.812).

**Found (≈half):** the routed-compact MoE MMQ dispatch (`mul_mat_q_routed_compact` /
`mmq_rdna3_5_id_use_compact(type,J)`) — `GGML_CUDA_DISABLE_MMQ_ROUTED=1` recovers B=9 202.5 →
**220.6 t/s** (step 1.422 → 1.306 s), B=8 unchanged.  The width-dependent
`mmq_rdna3_5_id_get_J(type, rows_per_expert)` tile choice is the prime suspect.  The remaining ~12 %
is another path (dense / routed-plain MMQ at `ncols=9`).

Full evidence, the ruled-out table and the next-session plan:
[`2026-09-24-qwen4exp-w9-verify-cliff.md`](2026-09-24-qwen4exp-w9-verify-cliff.md).

### 2026-09-24 — session 6: MTP revalidation on IQ4_NL + the MTP CPU-spin root cause

**Did:** the missing MTP acceptance revalidation (post-`0003`/`0017`/`0027`) on the newly downloaded
qwen4exp **IQ4_NL** model (3-GPU `-sm tensor`, q8_0 KV), a fixed-depth comparison across the four
axes + the phase-switch prompt, and — while measuring — the diagnosis of why MTP pins every CPU core.

**Acceptance revalidation (the gate that was missing):** `draft-mtp n3`, prose, `-c 16384 -n 3000`,
seed 42, reasoning off → **acceptance 0.83995** (1333/1587), mean len 3.52, acc per pos
(0.922, 0.841, 0.756), 78.4 t/s.  Healthy (gate: > ~0.45 at pos 1).  Session 2's IQ4_XS number was
0.81388; IQ4_NL is a different model, so the difference is expected.

**Depth comparison** (`-n 3000`; `--reasoning on` for R, off for C/P/K/X; `none` = plain decode).
Values are Generation t/s, parenthesised = draft acceptance:

| axis | prompt | none | fixed n7 | fixed n8 | adaptive cap 8 |
|---|---|---:|---:|---:|---:|
| R reasoning | `reasoning.txt` | 55.5 | 60.0 (0.363) | 45.5 (0.331) | 65.5 (0.594) |
| C code | `code-python.txt` | 56.2 | 104.1 (0.749) | 79.2 (0.744) | 87.4 (0.741) |
| P prose | `prose-rdna-boosts.txt` | 55.7 | 95.9 (0.690) | 78.7 (0.656) | 81.5 (0.717) |
| K recall | `recall.txt` | 53.2 | 127.8 (0.980) | 97.9 (0.957) | 97.3 (0.967) |
| X phase-switch | `code-reasoning-mixed.txt` | 55.9 | 73.2 (0.479) | 49.3 (0.380) | 72.5 (0.651) |

On this box **`n7` beat `n8` on every axis** and the adaptive controller (ceiling 8) was best or
near-best on R/X.

**Re-run with the spin removed (same protocol, `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0`):**

| axis | none | fixed n7 | fixed n8 | adaptive cap 8 |
|---|---:|---:|---:|---:|
| R reasoning | 55.4 | 75.4 | 57.1 | 83.8 |
| C code | 56.1 | 131.1 | 107.3 | 108.8 |
| P prose | 55.5 | 119.3 | 95.1 | 107.1 |
| K recall | 53.2 | 156.4 | 129.2 | 129.1 |
| X phase-switch | 55.7 | 92.2 | 62.9 | 95.4 |

Draft acceptance is **byte-identical** to the confounded run (the spin changes only timing) — e.g.
C n7 0.74918, P n7 0.69018, X n8 0.38008.  So the spin cost **~22–26 % of throughput per arm** (35 %
for the n3 prose pair) but **uniformly**, and it did **not** change the ranking: **`n7` still beats
`n8` on every axis on gfx1201**, and `n8`'s lower acceptance (C 0.744 vs 0.749, P 0.656 vs 0.690,
X 0.380 vs 0.479) shows the 8th draft token is mostly rejected — the extra verify width is not repaid.
So the session-6 table's *relative* conclusions were valid all along; only its absolute t/s were low.
The user's gfx1151 `n_max 8` observation does **not** transfer to gfx1201 and needs its own gate on
**gfx1151** (and gfx1100) before any conclusion.  Note also the purity boundary is on the **verify
width**: `n_max 7` = W=8 is the last pure depth, `n_max 8` = W=9 is the first that can diverge; above 7
use acceptance + MTP-vs-plain, never `plain == draft-mtp`.

**The CPU-spin root cause (the important find):** MTP decode pins **all 16 cores at 100 %**; plain
decode uses ~2.  `gdb` on the busy process:

```
ggml_graph_compute (OpenMP)
 <- ggml_backend_cpu_graph_compute
 <- ggml_backend_sched_graph_compute_async
 <- llama_context::graph_compute <- process_ubatch <- decode
 <- llama_decode
 <- common_speculative_impl_draft_mtp::draft
```

and `GGML_SCHED_DEBUG=1` shows **every** graph — including the 1-token draft graphs — as

```
## SPLIT #0: CPU  # 0 inputs
## SPLIT #1: Meta(ROCm0,ROCm1,ROCm2) # N inputs  [model.input_embed] [ple_embd] [mtp_tok_embd-48] ...
```

i.e. a small **CPU split** (the input / PLE `GET_ROWS`; on a discrete multi-GPU box those weights are
host-mapped) in front of every graph.  Each CPU graph compute forks the OpenMP pool, and the default
**active** wait policy makes the 15 idle workers spin in the barrier.  Plain decode does ~55 such
graphs/s (the pool settles between them → ~2 cores); MTP calls `llama_decode` **`n_max+1` times per
token** (target verify + each draft step), ~250–350 graphs/s, so the threads never sleep → all 16
cores pinned.  It is **not** the draft forward (offloaded 50/50) and not the sampler.

**Cost + mitigation** (same prompt/seed, `draft-mtp n3`, `-n 1500`):

| | CPU | acceptance | Generation |
|---|---:|---|---:|
| default | 15.8 cores | 0.83204 (1070/1286) | 78.9 t/s |
| `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` | 1.4 cores | 0.83204 (1070/1286) | **106.3 t/s** |

Identical acceptance/output ⇒ the same work; **~35 % of MTP throughput and 14 cores** go to the spin.
(`-n 2000` prose: 76.3 → 107.7 t/s.)  Do **not** ship this as a required env var — see §13: the
delivery must detect and pick the fast path itself.

**Related, smaller:** `W set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using
CPU` — the draft `top_k(10)` sampler also falls back to the CPU under `-sm tensor` (a per-draft-step
round-trip).  Separate cost; §13.

**Not done:** the `n7`/`n8` comparison needs re-running with the spin disabled; the gfx1151 gate is
the box where the `n_max 8` observation was made.

**Scope of the confound — NOT all MTP (session 6 follow-up).**  The spin needs the input/PLE embedding
to be **host-mapped**, so the scheduler emits a *non-empty* CPU split.  Same-config pairs measured the
same way:

| model | input table | active | passive | verdict |
|---|---|---:|---:|---|
| qwen4exp IQ4_NL (27 GiB PLE, host-mapped on 3× 32 GB) | host | 78.9 t/s / 15.8 cores | 106.3 / 1.4 | **−35 % confounded** |
| 27B UD-Q4_K_XL (`qwen35`, no PLE, `token_embd` in VRAM, 3-GPU tensor) | VRAM | 97.4 | 97.0 | clean |
| 4B Q4_1 (`qwen35`, 3-GPU tensor) | VRAM | 184.0 | 180.0 | clean |
| 4B Q4_1 (`qwen35`, 1 GPU) | VRAM | 195.2 | 193.9 | clean |

All pairs have byte-identical acceptance.  So the **delivery's dense/MoE MTP baselines (27B/35B, no
PLE) are not affected**, and **acceptance is unaffected everywhere**.  What *is* low is the **gfx1201
qwen4exp MTP throughput** — session 2's 57.5 t/s reading and any campaign qwen4exp t/s figure measured
on this discrete box are ~35 % below the true value and must be re-measured with the spin removed (or
the fixed build).  Reason it is qwen4exp-specific here: the 27 GiB `per_layer_token_embd` cannot fit
in the 96 GiB of VRAM alongside the 89 GiB of weights, so it is host-mapped; `qwen35`/`qwen35moe` have
no PLE and their `token_embd` fits, so their CPU split is empty.

**Practical stopgap for the harness/reporting:** any MTP t/s for a model with a host-mapped input/PLE
(qwen4exp on a discrete box) must be recorded with `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` (or the
fixed build) until §13 lands.  Do not carry forward the pre-fix numbers.

### 2026-09-24 — session 5: `0003` gate-mix ported to RDNA4 (item 2 done)

**Did:** work item 2 of §12.2, unblocked by the IQ4_NL Qwen3.8-Flash-Next download.  Widened the
`hc_gate_mix` call site to RDNA4 and flipped the RDNA4 `gatemix` policy to default ON.  The kernel
needed no change: its `mmb_frag_t`/`mmb_ld_frag`/`mmb_wmma_bf16`/`MMB_ACC_M` shim already has the
gfx12 arm, and there is no HC16 dependency — the `xn` BF16 cache is warmed by the `w_down` MMB GEMM
(not by the HC16 marks), which is why it fires on RDNA4 where HC16 is gated.

**Evidence (qwen4exp IQ4_NL, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`):** 285 `FIRED` (debug);
**byte-identical** greedy text default/on vs `LLAMA_HC_GATEMIX=0` (`471d102e7b7d`) and
`plain == draft-mtp`; width probe PASS; `MUL_MAT` 1297/1297; **+5.8 / +5.4 / +5.3 %** prefill at
pp8192 / 32768 / 65536 (warm, `-r 5`, 3 rounds).  Record:
[`2026-09-24-hc-gate-mix-rdna4.md`](2026-09-24-hc-gate-mix-rdna4.md).

**Set:** folded into `0003` (`c421c12ae`); 26/26 apply reproduces tree
**`ec54ad65f425c69b4dec4279efa68b0573e4afc8`**.

**Next:** nothing left in `gfx1201-closing.md` §12 except item 4 (`-ub 16384`); the gfx1100 §7.2.2
end-to-end `0003`/`0016` A/B needs a qwen4exp-capable box, and the `halo` HC16-under-`-sm tensor`
re-gate with `0027` is the gfx1151 follow-up.

### 2026-09-23 — session 4: the meta-backend `graph_optimize` gap closed (`0027`)

**Did:** work item 3 of §12.3 — make the child `graph_optimize` markings run under `-sm tensor`.
The meta backend owns the graph, so the scheduler never called the CUDA child's pass: the MMB
HC16/DOWN16/blk16/res16 marks were inert and only the MoE reduction's alloc dep was registered.
The child pass is now split into `marks_only`/`allocs_only` modes and forwarded twice by the meta
backend — alloc deps over the whole graph before allocation, markings per per-device subgraph after
the simple tensors exist (the marks must be keyed by the tensors the child compute sees).  The
res16 residual alloc dep moved into the alloc-deps half; the meta's own MoE loop stays as the
non-CUDA fallback.

**Evidence:** `MMB_OPT` **0 → 390** (4B q4_1) / **5418** (27B IQ3_S) under `-sm tensor`; **837**
MMB marks on qwen4exp IQ4_XS (was 0); default output unregressed (27B IQ3_S `-sm tensor`
`6073add19dac`, 4B `9f9f41270c70`, 35B `plain == draft-mtp` `5f6f93dd9b62`, MUL_MAT 1297/1297).
Record: [`2026-09-23-meta-graph-optimize-tensor-split.md`](2026-09-23-meta-graph-optimize-tensor-split.md).
This is **infrastructure, not a gfx1201 win** — HC16 is the gfx1151 feature and DOWN16/blk16/res16
were already measured neutral/negative here; but the mechanism is the prerequisite for HC16 under
`-sm tensor` on `halo`.

**Set:** new patch `0027` (`2af88bc06`); 26/26 apply reproduces tree
**`803e6d908ade68b02a71af9d5d0cf605aec1382a`**.

**Next:** §12.2 `hc_gate_mix` (needs an IQ4_NL qwen4exp model — not on this box); the gfx1100
brief `gfx1100-closing.md`; a gfx1151 `halo` re-gate of HC16 under `-sm tensor` with `0027`.

### 2026-09-23 — session 3: MMB quant coverage — Q4_1/Q5_0 ported to RDNA4 (item 1 done)

**Did:** work item 1 of §12.1.  Requantized Qwen3.5-4B/9B and Qwen3.8-27B to `--pure` Q4_0/Q4_1/Q5_0
and ran the interleaved pp8192/pp32768 A/B against the delivery MMQ (`GGML_CUDA_MMB=0`).  Result:
**Q4_1 wins** (+5.1/+4.2 % 4B, +8.0/+7.1 % 9B, +6.9/+6.1 % 27B) and **Q5_0 wins bigger** (+10.4/+8.7,
+14.4/+12.8, +14.1/+12.7); **Q4_0 loses** (−2.9/−1.9, −0.6/−0.3, −2.4/−1.6).  So RDNA4 now enables
`IQ_FAMILY | Q4_1 | Q5_0` (dense `IQ3_S | Q4_1 | Q5_0`); Q4_0 stays out.  `MXFP4`/`NVFP4` cannot be
produced by this box's quantizer and `IQ2_*` needs an unmatched imatrix — all three stay untested
and unenabled.

**Gates:** MUL_MAT oracle q4_1 47/47, q5_0 14/14 (MMB forced `MIN_T=1`); PPL parity 27B Q4_1
+0.19 %, Q5_0 −0.03 %; 27B greedy text **byte-identical MMB off↔on** for both (`plain ==
draft-mtp` holds for Q4_1; Q5_0's divergence is pre-existing with MMB off).  Width probe: MMB only
ever fires `T=512` (prefill), the `W=1..8` band never takes it; the logits-level probe divergence is
the documented coarse-quant relaxation (9B qwen35 fails with MMB off too; gemma4-12B fails both arms
at ~1e-6).  Full record: [`2026-09-23-mmb-q41-q50-rdna4.md`](2026-09-23-mmb-q41-q50-rdna4.md).

**Set:** folded into `0017` (new sha `5c3bb41c5`); `0018` regenerated (same mask line).  A fresh
`r13-beta-baseline` worktree applies the 26 patches **26/26** and reproduces tree
**`803e6d908ade68b02a71af9d5d0cf605aec1382a`** (was `b8700a6c2…`, before that `95f916a8e…`).

**Next:** §12.2 `hc_gate_mix` RDNA4 port (needs an IQ4_NL qwen4exp model — this box has only
IQ4_XS/Q4_K, whose HC gate is Q8_0, so it can only be code-checked here); §12.3 `graph_optimize`
markings under `-sm tensor` (infrastructure, low priority).

### 2026-09-23 — integration + first validation pass (gfx1201 / `soar`)

**Box:** 3× Radeon AI PRO R9700 (gfx1201), Ryzen 9 9950X3D2, 184 GiB, ROCm
`/opt/rocm-7.14.1-gfx102X` (build) + `/opt/rocm-7.14-gfx1201` (runtime).

**Integration — done; the r13+beta prerequisites carry the RDNA4 kernel ports.**  The brief's §2 order was followed exactly:
delivery r13 from `main` (16 blocks, applied tree `bb7b6d07b05ad8e23ab6e770172e7f597cfb3c12`), then
the 12 `beta/mmb-general` patches (`79136a15cac1920c0dd334b4c119a9cb42f9143b`), then the 25 closing
patches skipping `0015` (`1f09fd97d916ca080f7f65cdc422a3d6c425baa7`).  The closing set applied
**25/25 clean** on gfx1201 — the gfx12 shim is already in `beta/mmb-general`, so nothing needed
porting to make it build.  Baselines kept: `~/llama-baseline` (r13+beta, tree `79136a15…`),
campaign `~/llama.cpp` branch `closing-gfx1201`.

**Fix folded into the set (this session).**  `0004`'s GDN/PLE conv1d fusion is **not bit-identical
under `-sm tensor`** on 27B/35B — the direct kernel diverges from both the raw ops and upstream's
`SSM_CONV+SILU` fusion; on 1 GPU / `-sm layer` it is bit-identical.  The fusion is now gated to
single-device graphs (`GGML_CUDA_CONV_FUSION_MULTI=1` forces it back on).  Full record:
[`2026-09-23-gfx1201-conv-fusion-tensor-split.md`](2026-09-23-gfx1201-conv-fusion-tensor-split.md).
The `0004` patch was regenerated and the 25-patch set re-applied to tree
**`1be654fa71167e470ffcce70456bef7a23e6de25`** (25/25).  *(Root cause still open; see the record's
follow-up.)*

**Gates run so far:**

* `MMB_CFG` (default MMB on, 1 GPU) — exact expected gfx1201 row:
  `cc=0x1001201 dense_geom=1 … hc16=1 down16=0 gatemix=0 blk16=0 res16=0 glu=1 bf16w=1 iq3xxs_glu=0 routed=0`.
  `hc16=1` (the `0002` flip, inert on RDNA4), `gatemix=0`/`routed=0` (RDNA4 policy).  ✔
* Same-seed coherence (`-c 8192`, prose, `-n 24`, seed 42, temp 0, `--reasoning off`), campaign
  (post-fix) vs r13+beta baseline:

  | model | campaign | baseline |
  |---|---|---|
  | 4B Q8_0 (1 GPU) | `9f9f41270c70` (101) | `9f9f41270c70` |
  | 27B UD-IQ3_S (1 GPU) | `6073add19dac` (114) | `6073add19dac` |
  | 27B UD-IQ3_S (3 GPU tensor) | `6073add19dac` | `6073add19dac` |
  | 27B Q8_0 (3 GPU tensor, `-lm none -lzm on`) | `4d28938cbe05` (98) | `6073add19dac` (114) |
  | 35B-A3B UD-Q3_K_M (1 GPU) | `211639f7037d` (106) | `054353ad27ff` (105) |

  4B and 27B UD-IQ3_S are byte-identical to the baseline (a good first sign: the campaign's
  re-baselining is confined to where MMB/the new kernels actually fire).  27B Q8_0 and 35B differ
  — the expected prefill re-baseline (MMB is excluded for Q8_0, so this is the arch-neutral groups
  on the 3-GPU path / the MoE path).  **Do not treat the cross-tree hashes as gates**; the contract
  is intra-build purity (§6.2), not yet run.

* Conv-fusion A/B (27B/35B, tensor vs layer vs 1 GPU) — see the record above.  ✔

**Next actions (priority order):**

1. Finish the §6.1/§6.2 gates on the campaign tree: intra-build `plain == draft-mtp` at 8K/40K/128K
   (`--ctx-checkpoints 0`), and the per-model coherence at depth.  *(The Q8_0/35B cross-tree deltas
   above must be shown to be re-baselines, not impurities.)*
2. §6.3 width probe (P=1024 stock; extend for P=32768), §6.5 op oracles (`FLASH_ATTN_QSA` 26/26,
   `GATED_DELTA_NET`, `TOPK_QSA`, `LIGHTNING_INDEXER`, `FLASH_ATTN_EXT` from **separate** streams).
3. §6.6 qwen4exp headline: Flash-Next IQ4_XS + `-md mtp-…`, 3-GPU `-sm tensor`, pp8192…98304 vs the
   r13+beta baseline (S14 expectation +22 % at depth), then the `0021`/`0022`/`0016` A/Bs and the
   `0020`/`0026` sparse-MTP acceptance gate (`-n 3000`, reasoning pinned).
4. Per-patch verdict table (§7) and the port candidates' status (§8).  The known-inert ones
   (`0003`, `0016`, `0017`/`0018`, `0022`, `0025`) still need their "why inert" predicate recorded.
5. Re-check the conv-fusion root cause; if it is a real fixable bug, remove the multi-device gate.

#### 2026-09-23 — gate results

All on branch `closing-gfx1201` = delivery r13 + `beta/mmb-general` + the 25 closing patches
(post-fix tree `1be654fa71167e470ffcce70456bef7a23e6de25`), unless noted.  `MMB` default (on), 1 GPU
unless a split is stated.  Builds: campaign `~/llama.cpp/build-rocm`, baseline
`~/llama-baseline/build-rocm`.

**Build-time instantiation (§5) — PASS.**  `nm -C` on the FA dispatch TUs: `fattn-tile.cu.o` has **96 `tile_case` symbols, 0 defined / 96 `U`** (every KV type is instantiated in `template-instances/*.cu`, not the dispatch TU), `fattn-mma-f16.cu.o` defines **0**; the heaviest object is 33 KB and `ggml-hip` reached 50 % with no blown-up TU.  The closing kernels did **not** reintroduce the implicit-instantiation build trap.

**Op oracles (§6.5) — green.** `FLASH_ATTN_QSA` **26/26**, `GATED_DELTA_NET` **46/46**, `TOPK_QSA`
**4/4**, `LIGHTNING_INDEXER` **225/225**, `FLASH_ATTN_EXT` **5954 OK / 0 FAIL** (summary
5953/5953; counted from stdout-only, ANSI-stripped — §10 trap 4).

**Width purity (§6.3) — PASS.** 27B UD-IQ3_S P=1024 `width_purity=PASS (worst maxdiff 0)`, row-0
`3e870a40c63d3f2e` (MMB on) / `04f8b6a575db6e32` (MMB=0) — both reproduce the S14 gfx1201 record
exactly; 35B UD-Q3_K_M `d4d00b0661db280d`, PASS.

**Intra-build purity `plain == draft-mtp` (§6.2) — PURE everywhere tested** (bf16 KV unless noted,
prose prompt, seed 42 temp 0, `--ctx-checkpoints 0` at depth):

| model / arm | text |
|---|---|
| 27B UD-IQ3_S, 1 GPU, 8K | `857a25612912` |
| 35B UD-Q3_K_M, 1 GPU, 8K | `5f6f93dd9b62` |
| 27B UD-IQ3_S, 3 GPU `tensor`, 8K | `857a25612912` |
| 27B Q8_0, 3 GPU `tensor`, 8K (`-lm none -lzm on`) | `33ae8d598e7e` |
| 35B UD-Q3_K_M, 3 GPU `tensor`, 8K | `a6eba1d1350b` |
| 27B UD-IQ3_S, 3 GPU `tensor`, 40K | `b8b767d508a0` |
| qwen4exp IQ4_XS, 3 GPU `layer`, 8K, q8_0 KV | `79b90fcbcf98` |
| qwen4exp IQ4_XS, 3 GPU `tensor`, 40K, q8_0 KV | `db7d30353cd2` |
| qwen4exp IQ4_XS, 3 GPU `tensor`, 128K `n_max 1`, q8_0 KV | `a4017846f4cd` |

**qwen4exp MTP acceptance (§6.6/§6.7)** — 3 GPU `tensor`, q8_0 KV, `-c 16384 -n 3000`,
`draft-mtp n3`, prose, reasoning off: **acceptance 0.81388** (1360/1671, mean len 3.44), **57.5 t/s**
(HF-relevant threshold ~0.45 at pos 1 is comfortably met).  For reference the gfx1151 gate was
0.85035; the number is content-dependent, so this is healthy, not compared byte-for-byte.

**qwen4exp prefill A/B (§6.6)** — closing vs r13+beta, 3 GPU `tensor`, q8_0 KV, `-b/-ub 2048`,
`llama-bench -p …,98304 -n 0`, three interleaved rounds (only r=1 shown; the deep numbers are stable
to <0.5 %):

| depth | r13+beta | closing | Δ |
|---|---:|---:|---:|
| pp8192  | 2663.3 | 2645.2 | −0.7 % |
| pp32768 | 2685.2 | 2803.8 | **+4.4 %** |
| pp65536 | 2546.1 | 2625.7 | **+3.1 %** |
| pp98304 | 2408.0 | 2433.0 | **+1.0 %** |

The baseline *already* carries the `beta/mmb-general` +22 %-at-depth port, so this is what the
closing set adds on top.  The `0004` conv-fusion gate (this session) removes the PLE fusion from
this multi-GPU run; forcing it on (`GGML_CUDA_CONV_FUSION_MULTI=1`) should recover a little more
prefill at the cost of the purity bug — re-measure when the root cause is fixed.

**PPL parity (§6.8)** — `llama-perplexity -c 512 -b 512 --chunks 4`, MMB on vs off:
35B UD-Q3_K_M 5.2694 vs 5.2424 (**+0.51 %**, within CI ±0.40) and 27B UD-IQ3_S 5.8338 vs 5.8363
(**−0.04 %**).  Parity holds; a fragment-layout error would move this by orders of magnitude.

**Arch-scoped A/Bs (qwen4exp, 3 GPU `tensor`, 8K, `-n 200`):**

| knob | text | reading |
|---|---|---|
| default | `36019732357e` | — |
| `0021` `GGML_CUDA_QSA_INDEXER_CACHE=0` | `36019732357e` | **byte-identical** — derived indexer cache is a text no-op ✔ |
| `0022` `LLAMA_QSA_DENSE_DECODE_UNTIL=0` | `c99b4682b5d1` | text moves — proves gfx1201's default decode arm is *dense-always* ✔ |
| `0016` `LLAMA_QSA_SCORE_WMMA=0` | `5d920e5f715e` | **differs** — see correction below |

**Brief correction — `0016` on RDNA4.**  The brief (§6.6) expects `LLAMA_QSA_SCORE_WMMA=0` to be
byte-identical on gfx1201.  It is **not**: the default builds the fused lightning-indexer op on every
arch and the *generic* fallback still casts q/k to F16, which the patch itself documents as a
prefill re-baseline (`src/models/qwen4exp.cpp:1531-1542`).  The arch claim that *is* true: the
RDNA3_5 WMMA kernel is not taken (`supports_indexer4` = `GGML_CUDA_CC_IS_RDNA3_5`, plus the opt-in
`GGML_CUDA_LIGHTNING_INDEXER4_GFX1100` for gfx1100), the `LIGHTNING_INDEXER` oracle passes 225/225,
and width purity is untouched.  A RDNA4 WMMA port remains a candidate (§8).

#### 2026-09-23 — porting layers: what actually needed RDNA4 kernel work

The campaign runs on RDNA4 because **two layers** of RDNA4 work are present, and only one of them
is in the 25 closing patches:

1. **`beta/mmb-general` — the big WMMA ports (12 patches, a prerequisite).**  This is where the
   substantial new kernel work for RDNA4 lives, done in the beta's own gfx1201 sessions
   ([`gfx1201-porting.md`](../../beta/mmb-general/gfx1201-porting.md) S4–S13):
   * **qsa3** (packed-block sparse-attention WMMA): the gfx12 fragment shim + the first oracle the
     kernel ever had on any arch, `FLASH_ATTN_QSA` 26/26, **+7.6/+11.5/+10.4 % prefill** (S4);
   * **mmb** (general-purpose bf16-WMMA dequant GEMM): the gfx12 bf16 fragment shim, the gfx11 asm
     verified bit-identical, the arch × weight-type × path policy split (S5–S7), then RDNA4 dense
     tile geometry (S10), per-arch tuning defaults (S11), routed policy (S12), F32 policy split
     (S13).

   So "the campaign builds on RDNA4 with nothing ported in the closing set" is true **only because
   the closing set sits on top of a beta that *is* the RDNA4 port**.  The qsa3 kernel the closing
   `0007` touches is the beta's RDNA4-ported one; `FLASH_ATTN_QSA` 26/26 and the qwen4exp prefill A/B
   are its gfx1201 witness.

2. **The closing set — new kernels, and several are NOT active on RDNA4.**  These are the ones that
   would need a *new* RDNA4 port before they fire:

   | closing kernel | RDNA4 status |
   |---|---|
   | `0003` `hc_gate_mix` (`mmb.cu`) | **not ported** — IQ4_NL + RDNA3_5 WMMA; `gatemix=0`, call site RDNA3_5-gated |
   | `0013` prefill indexer relu-sum | **inert** — call site `GGML_CUDA_CC_IS_RDNA3_5` (`ggml-cuda.cu:4250`) |
   | `0016` `QSA_SCORE_WMMA` (`lightning-indexer.cu`) | **PORTED 2026-09-23** — gfx12 `I_MAJOR` A/B layout + RDNA4 enable, **default ON**, **+1.7…+12.3 % prefill** |
   | `0017`/`0018` MMB new quant types | **not enabled** — RDNA4 mask is `IQ_FAMILY` only; the dequant is arch-neutral but unreachable |
   | `0011` HC BF16 streams | default OFF; inert under `-sm tensor` (meta-backend `graph_optimize` gap); tested `-sm layer` — no gfx1201 win |
   | `0010` MoE BF16 epilogue | default OFF; same meta gap; tested `-sm layer` — no gfx1201 win |
   | `0023` MMB HC16 per-context | inert on RDNA4 (HC16 gated) |
   | `0001`/`0004`/`0006`/`0008`/`0012`/`0014`/`0019`/`0020`/`0021`/`0026` | arch-neutral and active (see the verdict table) |

   The new quant types are the sharpest case: **no porting is needed to make them *compile***, but
   *enabling* them on RDNA4 is the optional port the brief names — it needs an RDNA4 per-type
   geometry and measurements.  This box has **no** MXFP4/NVFP4/IQ2 model and the mask has no env
   override, so they could not be exercised even as a forced A/B; they are simply unreachable.

**Consequence for the summary:** "no port needed" meant *no fix was needed to make the 25 patches
apply and build*; it did **not** mean the campaign is port-free.  The qsa3/mmb RDNA4 kernel work is
in the beta prerequisite, and the closing set's own RDNA3_5-only kernels remain un-ported (see §8).

#### 2026-09-23 — do the gfx1151 lossy-prefill wins transfer? (`0010`/`0011`) — **No**

The maintainer's gfx1151 high-speed prefill config (`LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1
GGML_CUDA_MMB_DOWN16=1`) was tested on gfx1201.  Full record:
[`2026-09-23-gfx1201-lossy-prefill-transfer.md`](2026-09-23-gfx1201-lossy-prefill-transfer.md).

1. **Under `-sm tensor` the three markings never run.**  HC16/DOWN16/blk16/res16 are
   `ggml_backend_cuda_graph_optimize` markings; under tensor split the **meta backend** owns the
   graph and the CUDA child's `graph_optimize` is never called (`MMB_OPT=0` under `-sm tensor` vs
   `60` under `-sm layer`, same binary/model; see the meta backend's own comment).  So on this
   box's qwen4exp mode these features are **inert** — which is why the first tensor A/B was neutral
   and text-identical.
2. **Under `-sm layer` (markings run), the win is not there.**  IQ4_XS, f16 KV, `-b/-ub 4096`,
   `-r 3`: default pp8192/32768 = 3444/4963; `0011` 3427/4960; `0010` 3486/4947; all three
   3444/4952 — **flat/noisy**.  `0011` fires (same-seed text moves `551425b9758e` -> `07f7d16a144a`).
   The gfx1151 +4.9 % was an APU/unified-memory bandwidth effect.

**Portability finding for the set:** a `graph_optimize` marking (HC16/DOWN16/blk16/res16 or any
future one) is a *single-backend* feature; the tensor-split meta path bypasses it.  Making it apply
under `-sm tensor` needs the meta backend to run the CUDA markings on the per-device shard graphs
before allocation (marks are keyed by tensor pointer; residual marks add alloc deps).  That is the
concrete gfx1201 (multi-GPU) port — but per point 2 it is not worth it for these two features.

#### 2026-09-23 — `0016` `QSA_SCORE_WMMA` **ported to RDNA4 — a real gfx1201 win**

Following the lossy-prefill negative, the first *positive* transfer: `0016`'s 4-head/128-dim indexer
WMMA kernel was RDNA3_5-only.  It uses the `ggml_cuda_mma` `tile`/`mma()` abstraction (not
hand-rolled fragments), so the port was three small changes: select
`DATA_LAYOUT_I_MAJOR` A/B for RDNA4 (`MIRRORED` is gfx11-only), widen the compile gate to
`(RDNA3 || RDNA4)`, and enable RDNA4 in `indexer4_arch_enabled` (default ON,
`GGML_CUDA_LIGHTNING_INDEXER4_GFX1201=0` for the A/B).

* **Correctness:** `LIGHTNING_INDEXER` **225/225** (the 4-head cases now take the WMMA path vs the
  CPU oracle); width purity PASS; `plain == draft-mtp` byte-identical (`90069b3ed9c4`).
* **Prefill (qwen4exp IQ4_XS, 3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 4096`, `-r 5`, interleaved):**
  +1.7 % @pp8192, +3.2 % @pp16384, +6.5 % @pp32768, **+12.3 % @pp65536** vs the generic fallback.
  (The first cold run showed a spurious −21 % at pp8192 with ±121 t/s noise; the warm `-r 5` run
  is clean at every depth.)
* Folded into the `0016` commit; the 25-patch set re-applies to tree
  **`95f916a8e015efd68ee44bcea7620187ebc70019`**.  Full record:
  [`2026-09-23-qsa-score-wmma-rdna4.md`](2026-09-23-qsa-score-wmma-rdna4.md).

#### 2026-09-23 — per-patch verdict table (§7)

| patch | fires on gfx1201? | verdict | evidence |
|---|---|---|---|
| `0001` hc_combine_norm | yes (qwen4exp HC prefill) | not isolated this session; exercised by the qwen4exp gates | qwen4exp purity 8K/40K/128K |
| `0002` default flips | yes | MMB default on; RDNA4 row correct | `MMB_CFG`; MMB on/off PPL |
| `0003` hc_gate_mix | **yes — RDNA4 ported this session, default ON** | **+5.8/+5.4/+5.3 % prefill** pp8192/32768/65536 (qwen4exp IQ4_NL); bit-identical text; width probe PASS | [`2026-09-24-hc-gate-mix-rdna4.md`](2026-09-24-hc-gate-mix-rdna4.md) |
| `0004` conv1d fusions | yes | **fixed** — gated to single-device graphs | [`2026-09-23-gfx1201-conv-fusion-tensor-split.md`](2026-09-23-gfx1201-conv-fusion-tensor-split.md) |
| `0005` QSA block window | qwen4exp-only | exercised | 40K/128K purity |
| `0006` narrow-row RMS | arch-neutral | width purity + coherence | §6.3 |
| `0007` QSA visibility fold | qwen4exp-only | exercised | 40K/128K purity |
| `0008` M=4 HC inject | MMB geometry | exercised | qwen4exp prefill A/B |
| `0009` `lzm auto` | yes | text-identical (`-lm none -lzm on`) | 27B Q8_0 coherence |
| `0010` MoE BF16 epilogue | **`-sm layer` and `-sm tensor` (since `0027`)** | **transfer tested — no gfx1201 win** | [`2026-09-23-gfx1201-lossy-prefill-transfer.md`](2026-09-23-gfx1201-lossy-prefill-transfer.md), [`2026-09-23-meta-graph-optimize-tensor-split.md`](2026-09-23-meta-graph-optimize-tensor-split.md) |
| `0011` HC BF16 streams | **`-sm layer` and `-sm tensor` (since `0027`)** | **transfer tested — no gfx1201 win**; `0027` makes the markings run under `-sm tensor` too | same |
| `0012` mmb_cvt `out_xn` | yes (MMB on) | exercised | qwen4exp prefill A/B |
| `0013` indexer relu-sum | **no** | **inert on RDNA4** — call site is `GGML_CUDA_CC_IS_RDNA3_5(cc)` (`ggml-cuda.cu:4250`) | call-site gate; our tree already banks the reduction via the fused `GGML_CUDA_QSA_INDEXER_SCORE` |
| `0014` QSA scorer trim | qwen4exp-only | exercised | qwen4exp gates |
| `0016` `QSA_SCORE_WMMA` | yes, **RDNA4 WMMA (ported this session, default ON)** | **+1.7/+3.2/+6.5/+12.3 % prefill** @pp8192/16384/32768/65536; oracle 225/225; purity holds | [`2026-09-23-qsa-score-wmma-rdna4.md`](2026-09-23-qsa-score-wmma-rdna4.md) |
| `0017` MMB quant coverage | **Q4_1 + Q5_0 ported (default ON); Q4_0 stays out** | **done** — RDNA4 mask now `IQ_FAMILY | Q4_1 | Q5_0`, dense `IQ3_S | Q4_1 | Q5_0`; Q4_1 +6.9/+6.1 %, Q5_0 +14.1/+12.7 % (27B pp8192/pp32768), Q4_0 −2.4/−1.6 %; oracle 47/47 + 14/14, PPL parity, greedy text byte-identical off↔on | [`2026-09-23-mmb-q41-q50-rdna4.md`](2026-09-23-mmb-q41-q50-rdna4.md) |
| `0018` MMB IQ2 coverage | **not enabled** | not ported — `llama-quantize` requires an imatrix this box has no matching copy of; IQ2 dense is rare and RDNA4 routed is off anyway | same |
| `0019` HC16 eval-callback fix | **no** (HC16 is RDNA3_5-gated) | inert on RDNA4 | `MMB_CFG hc16=1` but call site gated |
| `0020` sparse MTP draft | qwen4exp | exercised | 40K/128K purity; acceptance 0.81388 |
| `0021` derived indexer cache | yes (default on) | byte-identical A/B | `=0` text identical |
| `0022` gfx1151 crossover | **no** — gfx1201 dense-always | inert | `LLAMA_QSA_DENSE_DECODE_UNTIL=0` moves text |
| `0023` MMB HC16 per-context | **no** on RDNA4 (HC16 gated); markings now run under `-sm tensor` (`0027`) | inert on RDNA4; gate still worth running | 40K/128K purity; [`2026-09-23-meta-graph-optimize-tensor-split.md`](2026-09-23-meta-graph-optimize-tensor-split.md) |
| `0024` input layer GPU | **no** | superseded by `0025` | — |
| `0025` host-buffer input | **no-op** (discrete GPU) | `prop.integrated = 0`; scheduler guard not reached | `GGML_FORCE_NO_INTEGRATED=1` matched default |
| `0026` sparse MTP default ON | qwen4exp | exercised | 40K/128K purity |

**Rule-5 batched verify-width gate (§6.7/S14.7f)** — 27B UD-Q4_K_XL, q8_0 KV, 1 GPU,
`llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8`, three interleaved rounds (`S_TG = TG*B/T_TG`):

| width | closing (r1/r2/r3) | r13+beta (r1/r2/r3) |
|---|---|---|
| B=1 | 28.67 / 28.67 / 28.70 | 28.70 / 28.70 / 28.74 |
| B=4 | 76.46 / 76.53 / 76.84 | 76.65 / 76.65 / 76.90 |
| B=8 | 90.36 / 90.46 / 90.90 | 90.81 / 90.82 / 90.87 |

Within noise at every width (≤0.5 %, both directions across rounds).  **PASS** — the check that
caught the 2026-09-12 mmvq regression.

**Still open / not run this session:** the qwen4exp four-axis MTP set (only the prose axis was
measured), the `llama-imatrix` `0019`/`0023` gate (the NanBeige model is absent on this box — note
HC16 is RDNA3_5-gated, so the gate is a no-op on RDNA4 anyway), and the `0001`/`0008` isolated
fusion A/Bs (they are exercised by the qwen4exp gates but not singled out).

---

## 12. Remaining RDNA4 porting work — handover for the next session

**Updated:** 2026-09-24 (session 6).  §11 holds the gate results; this section is the **open** work.
Items 1 (MMB quant coverage), 2 (`0003` gate-mix) and 3 (the meta `graph_optimize` gap) are now
DONE; 4 remains.  **§13 is the new main task for the next session** — remove the MTP CPU spin
without requiring `OMP_*`/`KMP_*` environment variables (auto-detect the degenerate scheduling and
pick the high-performance path).
The campaign's arch-scoped features are one of three shapes — classify before touching one:

1. **Wrong-arch predicate over a working kernel.**  The port is usually small (a layout / enable
   change) and the gfx1151 win transfers if the hardware bottleneck is the same.  *Worked example:
   `0016` `QSA_SCORE_WMMA` — ported this session, **+1.7/+3.2/+6.5/+12.3 %** qwen4exp prefill.*
2. **A marking that does not run at all.**  `graph_optimize`-based markings (HC16/DOWN16/
   blk16/res16) were inert under `-sm tensor` because the **meta backend** owns the graph and the
   CUDA child's `graph_optimize` never runs (`MMB_OPT=0`).  **FIXED (session 4, patch `0027`)** — the
   meta now forwards the child pass twice (alloc deps before allocation, marks per per-device
   subgraph after); the features themselves are still not gfx1201 wins.
3. **A platform-specific win.**  APU/unified-memory bandwidth (`0010`/`0011`) — measured, does not
   transfer; park it.

Checklist per candidate: (a) `grep` the predicate + `MMB_CFG` the policy row; (b) **correctness** —
op oracle (`test-backend-ops -o <OP>`, separate stdout/stderr; a graph fusion may have **no oracle**,
then the only gate is end-to-end text + `plain == draft-mtp`) and width purity; (c) **perf** —
`llama-bench -p 8192,32768,65536 -n 0 -b 4096 -ub 4096 -r 5 -sm tensor`, **warm page cache**,
interleaved vs the fallback, quote the deep point; (d) fold into the owning patch.

### 12.1 Work item 1 — MMB WMMA quant coverage on RDNA4 (`0017`/`0018`)  ← **DONE 2026-09-23**

**Result:** Q4_1 and Q5_0 ported (default ON).  RDNA4 `mmb_wtype_mask()` = `IQ_FAMILY | Q4_1 |
Q5_0`; `mmb_dense_tmask()` = `IQ3_S | Q4_1 | Q5_0`.  Q4_1 **+5–8 %**, Q5_0 **+9–14 %** pp8192/32768
on 4B/9B/27B (27B: +6.9/+6.1 and +14.1/+12.7); Q4_0 **−0.3…−2.9 %** and stays out.  Op oracle
47/47 + 14/14, PPL parity, greedy text byte-identical MMB off↔on.  MXFP4/NVFP4 unreachable (no
quantizer path), IQ2 needs an unmatched imatrix.  Full record:
[`2026-09-23-mmb-q41-q50-rdna4.md`](2026-09-23-mmb-q41-q50-rdna4.md).  Folded into `0017`;
26/26 apply reproduces tree `803e6d908ade68b02a71af9d5d0cf605aec1382a`.

The measurement campaign (below) is the method and context.

**What.**  `0017` adds Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 and `0018` adds IQ2_S/IQ2_XS/IQ2_XXS to the MMB
weight-type mask.  The dequant + WMMA (`mmb_tile_gemm`) is **already RDNA4-ported** by the beta, and
the new types **compile** for gfx1201 — but the RDNA4 masks exclude them:

* routed mask (`mmb.cu:1808`):  `RDNA4 ? IQ_FAMILY : (IQ_FAMILY | K_AND_Q8 | Q4Q5 | IQ2)`
* dense mask  (`mmb.cu:1849`):  `RDNA4 ? (1 << IQ3_S) : ~0`

So on RDNA4 the **only** MMB dense type is IQ3_S, and the routed set is IQ4_NL/IQ3_S/IQ4_XS/IQ3_XXS.
Enabling the new types is a **policy + measurement** task, not new kernel code.

**Why the default is narrow.**  The beta's S5-S7 (`gfx1201-s5s7-mmb-results.md`,
`gfx1201-s10-dense-geometry.md`) found MMB **dense** with the k-quants was a **−4…−13 % regression**
on RDNA4 (the delivery's RDNA4 MMQ path is already tuned) — that is why the dense mask is IQ3_S
only.  Do **not** flip the mask wholesale; measure **per (type, path, shape)**.

**How (a measurement campaign — budget a session).**

1. **Get models.**  This box has **no** Q4_0/Q4_1/Q5_0/MXFP4/NVFP4/IQ2 GGUF (only IQ4_XS/IQ3_XXS/
   Q4_K/Q8_0/Q6_K).  Quantize one: `llama-quantize <27B-Q8/BF16> <out> Q4_0` (and Q5_0; `MXFP4`/
   `NVFP4` need the newer quantizer).  A 4B/9B dense model is enough for a first pass; the 35B-A3B
   Q3_K_M is the routed-MoE shape.
2. **Baseline** = the delivery path (`GGML_CUDA_MMB=0`); **arm** = MMB with the type force-enabled.
   The set already carries the per-type env overrides `GGML_CUDA_MMB_TYPES` (whole wtype mask) and
   `GGML_CUDA_MMB_DENSE_TYPES` (dense tmask) from `0017`, so no rebuild is needed — e.g.
   `GGML_CUDA_MMB_TYPES=iq4_nl,iq3_s,iq4_xs,iq3_xxs,q4_1 GGML_CUDA_MMB_DENSE_TYPES=iq3_s,q4_1`.
   Land only the winners.
3. Per type: `llama-bench -p 8192,32768 -n 0 -b 4096 -ub 4096 -r 5`, warm, interleaved, dense
   (`MUL_MAT`) **and** routed (`MUL_MAT_ID`, MoE model); plus the `MUL_MAT`/`MUL_MAT_ID` oracle for
   the type (expected counts in `gfx1100-closing.md` §6.6) and `test-logits-width-probe`.
4. **Land** the winning types in the RDNA4 routed mask and/or the S10 dense per-type policy; leave
   the losers excluded.  `MXFP4`/`NVFP4` are the most likely RDNA4 dense candidates (weakest MMQ
   path); `Q4_0`/`Q4_1`/`Q5_0` the least (the beta's regression finding); IQ2 unknown.

### 12.2 Work item 2 — `0003` `hc_gate_mix` RDNA4 port  ← **DONE 2026-09-24**

**Result:** ported and default ON (folded into `0003`, `c421c12ae`).  The gate-mix fusion is
**bit-identical** to the unfused chain (`471d102e7b7d` default/on == off; `plain == draft-mtp`) and a
**+5.8 / +5.4 / +5.3 % prefill win** at pp8192 / 32768 / 65536 on qwen4exp **IQ4_NL** (3-GPU
`-sm tensor`, q8_0 KV, `-b/-ub 2048`, `-r 5`, warm, 3 rounds).  It fires 285 times; width probe
PASS; `MUL_MAT` 1297/1297.  The kernel itself needed **no** change (the `mmb_frag_t`/`mmb_wmma_bf16`/
`MMB_ACC_M` shim is already arch-aware) and there is **no HC16 dependency** (the `xn` BF16 cache is
warmed by the `w_down` MMB GEMM, not by the HC16 marks).  Full record:
[`2026-09-24-hc-gate-mix-rdna4.md`](2026-09-24-hc-gate-mix-rdna4.md); 26/26 apply reproduces tree
`ec54ad65f425c69b4dec4279efa68b0573e4afc8`.

**What.**  The fused HC gate GEMM+sigmoid+mix.  Call site `GGML_CUDA_CC_IS_RDNA3(cc)`
(`ggml-cuda.cu:5715`); `mmb_arch_defaults` sets `gatemix=0` on RDNA4 (`mmb.cu:1534`), and RDNA3_0 is
opt-in (`LLAMA_HC_GATEMIX=1`).

**The port is one map + two predicate edits.**  `hc_gate_mix_kernel<4>` already uses the
**arch-aware MMB shim** (`mmb_frag_t`, `mmb_ld_frag`, `mmb_wmma_bf16` — the latter has the `_gfx12`
arm), so the MMA is gfx12-ready.  **But its epilogue hand-rolls the gfx11 accumulator map**
(`token = 2*e + (lane>>4)` — see the comment in `hc_gate_mix_kernel`); RDNA4 needs
`8*(lane>>4) + e`.  That map, the call-site predicate (`RDNA3` → `RDNA3 || RDNA4`), and the policy
row are the whole change.

**Model requirement.**  The kernel requires `w->type == GGML_TYPE_IQ4_NL` (`mmb.cu:2223`).  This
box's qwen4exp is **UD-IQ4_XS**, and its HC gate weight (`blk.N.hc_attn_up.weight`, the `w_up` the
gate GEMM consumes) is **Q8_0**, not IQ4_NL — so the fusion is unreachable here even with the
predicate widened.  (`hc_attn_down.weight` is Q8_0 too; the IQ4_NL reference was the gfx1151
campaign's IQ4_NL model, which is not on this box.)  Confirm with:
`python3 -c "from gguf import GGUFReader; ..."` on shard 2, or `gguf-dump`.  Without an IQ4_NL
qwen4exp model the port can only be code-checked here; the end-to-end gate needs that model.

**Session 3 correction (2026-09-23) — the epilogue map is ALREADY arch-aware.**  The brief's
"epilogue hand-rolls the gfx11 accumulator map" is stale: `hc_gate_mix_kernel`'s epilogue uses
`MMB_ACC_M(e, cn)` (`mmb.cu:1000`), which is the arch-aware shim — on RDNA4 it is `8*cn + e`, on
gfx11 `2*e + cn` (the stale comment above it still says `2e + (lane>>4)`).  The kernel also uses
`mmb_frag_t` / `mmb_ld_frag` / `mmb_wmma_bf16`, all of which have the gfx12 arm, so the only edits
left are the **call-site predicate** (`GGML_CUDA_CC_IS_RDNA3(cc)` → `RDNA3 || RDNA4`) and the
**policy row** (`c.gatemix`).  Do NOT enable it default-on until an IQ4_NL model gates it.

**Gate.**  No op-level oracle (graph fusion) → correctness is `plain == draft-mtp` + coherence, and
the A/B is `LLAMA_HC_GATEMIX=1` vs `=0` (or the RDNA4 policy row).  Target is prefill.  The `0016`
result says a correct WMMA enablement here can be a several-percent win — worth doing.

### 12.3 Work item 3 — make the `graph_optimize` markings run under `-sm tensor`  ← **DONE 2026-09-23 (patch `0027`)**

**Result:** the meta backend now forwards the child `graph_optimize` in two modes
(`marks_only`/`allocs_only`): alloc deps over the whole graph before allocation, markings per
per-device subgraph after the simple tensors exist.  `MMB_OPT` goes **0 → 390 / 5418** under
`-sm tensor` (4B / 27B), 837 MMB marks land on qwen4exp (was 0), and the default output is
unchanged (27B IQ3_S `-sm tensor` `6073add19dac`, 4B `9f9f41270c70`, MUL_MAT 1297/1297).  All the
fused kernels' alloc deps are registered, not just the MoE reduction.  Full record:
[`2026-09-23-meta-graph-optimize-tensor-split.md`](2026-09-23-meta-graph-optimize-tensor-split.md)
— including the finding that the enabled features are still **not** gfx1201 wins (that is gfx1151
work) and that the `hc_combine_norm` fused window was rejected on the tested qwen4exp graph on both
splits.

**What.**  HC16 (`0023`), DOWN16 (`0010`), blk16/res16 (`0011`) are pre-allocation markings in
`ggml_backend_cuda_graph_optimize`, which **never runs** when the meta backend owns the graph
(§11).  To apply them under tensor split the meta backend must run the CUDA markings on the
per-device shard graphs **before allocation** — the marks are keyed by `const ggml_tensor *` and the
residual marks add gallocr alloc deps.  Precedent: `ggml_backend_meta_graph_optimize` already does
this for the `moe_weighted_reduction` alloc deps.

**Priority: low for `0010`/`0011`** (measured no transfer, §11), but it is the **only** way any
`graph_optimize`-based marking (including a future one that *is* bandwidth-bound here) works under
`-sm tensor`.  Infrastructure item, not a win.

### 12.4 Work item 4 — the remaining RDNA3_5-gated kernels (lower priority)

* **`0013` prefill indexer relu+head-sum** (`idx_relu_sum`, call-site `GGML_CUDA_CC_IS_RDNA3_5`):
  `0016`'s port already banks this reduction (the fused lightning-indexer computes
  `bias + sum_h relu(dot_h)`), so a separate enablement is likely redundant on RDNA4.  Verify by
  diffing `GGML_CUDA_IDX_RELU_SUM` on/off **after** the `0016` port — if the graph no longer
  contains that chain, there is nothing to port.
* **`0011` HC BF16 streams / `0023` HC16**: `0011` needs Work item 3 to run under `-sm tensor`;
  `0023`'s HC16 is RDNA3_5-gated and not bandwidth-bound on discrete RDNA4 → park unless a
  48 GB single-GPU RDNA4 box appears.

### 12.5 Harness + traps (copy these into the next session)

* **Warm the page cache** for the multi-shard 87 GiB model before any A/B — the first cold pp8192
  read ±121 t/s (a spurious −21 %) while the warm re-run was clean.
* **Kill leftover benches**: `pkill -9 -x llama-bench` — an orphan from a timed-out harness holds
  ~22 GiB/GPU and the next load dies with
  `ggml-backend-meta.cpp:1848 GGML_ASSERT(meta_buf_ctx->bufs[i])`.
* **Capture stdout/stderr to files and parse after** — never `| grep | head` a bench (it can hang).
* Use `pgrep -x llama-bench` (not `-f`; `-f` matches your own shell).
* Build loop: `cmake --build build-rocm --target llama-cli llama-bench test-backend-ops -j 16`
  (a `ggml-cuda` TU is ~2-4 min; the FA instances are the long pole, ccache covers the rest).
* **Fold a port** with `git commit --fixup=<commit>` then
  `GIT_SEQUENCE_EDITOR=true GIT_EDITOR=true git rebase --autosquash <commit>~1`; regenerate with
  `git format-patch -1 <sha> --stdout --no-numbered`; replace the patch file and re-verify a fresh
  worktree at `r13-beta-baseline` + the 26 patches reproduces the branch tip tree.
* The verified handover state at the end of session 5: branch `closing-gfx1201`, tip tree
  **`ec54ad65f425c69b4dec4279efa68b0573e4afc8`** (26/26 apply: r13 + beta + closing; `0003` carries
  the RDNA4 gate-mix port, `0017` the Q4_1/Q5_0 RDNA4 enablement, `0027` the meta `graph_optimize`
  forwarding).  Session 2's tree was `95f916a8e015efd68ee44bcea7620187ebc70019`.

---

## 13. Next session — automatic path selection: keep the MTP input off the CPU (no env vars)

**Goal (maintainer, 2026-09-24):** a user must be able to run a typical `llama-server` config with
**no** `OMP_*`/`KMP_*` environment variables and get the fast path.  The delivery has to **detect**
the degenerate scheduling and choose the high-performance path itself, default-on.

**Background:** §11 session 6 root-caused the MTP CPU burn — under `-sm tensor` the scheduler puts a
CPU split (the input / PLE `GET_ROWS`) at the front of every graph; MTP multiplies those graphs
~`n_max+1`× and the OpenMP active-wait pool spins all 16 cores, costing ~35 % of MTP throughput.
The env mitigation exists but must not be a user requirement.

### 13.1 The structural fix — get the input embeddings off the CPU

If there is no CPU split there is nothing to spin.  The input embedding is the only CPU graph, and
`0024` (single-device input-on-GPU) / `0025` (host-buffer input + scheduler guard) already cover the
**APU / 1-device** cases.  On a **discrete multi-GPU** box `n_devices() != 1` and
`prop.integrated == 0`, so neither applies and the host-mapped `token_embd` / `per_layer_token_embd` /
`mtp_tok_embd` `GET_ROWS` stays on the CPU.

Candidates, cheapest first:
1. **draft `mtp_tok_embd`** — one small table, a per-draft-step cost; put its `GET_ROWS` on a device.
2. **target `token_embd`** — one table.
3. **`per_layer_token_embd`** (~27 GiB) — the hard one: a `GET_ROWS` is **row-parallel**, so the table
   shards cleanly across the tensor-split devices (each device holds a row range; indices are routed
   to the owning shard, or every shard gathers with a masked add).  Investigate whether the meta
   backend's split machinery can carry the embedding table + `GET_ROWS`, or whether the existing
   `-sm tensor` `ncols2`/`GET_ROWS` split-state handlers already do (see
   `ggml_backend_meta_*` `handle_get_rows`).
4. Alternative to sharding the table: keep it host-resident but run the `GET_ROWS` **on the device**
   over a host-mapped pointer — this is the `0025` idea generalised past `integrated`.  Cheaper VRAM,
   but needs the device to read host memory (discrete GPUs can via HMM/`hipHostMalloc`-mapped, at a
   bandwidth cost; only the gathered rows are read).

The **structural** option is the right one if it validates; it removes the CPU graph rather than
hiding it.

### 13.2 The runtime fix — automatic spin mitigation (the detection half)

Even with a CPU split, the active-wait spin is pure waste.  Explore, in order:
* **Per-split thread count**, not per-process: the CPU split here is tiny.  Find where the scheduler /
  CPU backend picks the split's thread count and make **small CPU splits run single-thread**
  (`n_threads = 1` → no OpenMP fork → no barrier to spin on).  This is the most surgical and is
  automatic by construction.  Check `ggml_backend_cpu_graph_compute` / the threadpool wiring
  (`ggml_backend_cpu_set_n_threads`, `set_n_threads` list in `llama_context`).
* **Passive/low-spin pool**: detect the shape (GPU backend present + CPU split is input-only) and set
  the pool's wait policy / `KMP_BLOCKTIME` equivalent programmatically (check whether `kmp_set_blocktime(0)`
  or a `ggml_threadpool` pause/priority mode is reachable from the CPU backend; if not, whether the
  pool can be created with the right mode).
* A startup heuristic that logs one clear line when it engages, plus an env **kill-switch** (default-on
  per AGENTS.md, disable-only env).

Whatever is chosen: **automatic, default-on, arch-neutral, an env kill-switch for bisection only** —
not an env opt-in.  It must not regress plain decode, prefill, or CPU-only runs.

### 13.3 Also fix — draft sampler backend offload under `-sm tensor`

`llama_context::set_sampler` rejects the backend sampler outright when
`model.split_mode() == LLAMA_SPLIT_MODE_TENSOR` (`"backend sampling not supported with
SPLIT_MODE_TENSOR; using CPU"`), so the draft `top_k(10)` chain runs on the CPU every draft step.
Make the backend sampler tensor-split-aware (or keep it on a device) so no per-step CPU round-trip is
needed.  Small, but it composes with 13.1/13.2.

### 13.4 Acceptance criteria

* A plain `llama-server` with **no** `OMP_*`/`KMP_*` env keeps MTP decode at ~1–2 CPU cores on the
  3-GPU box and reproduces the passive-wait throughput (**~106 t/s**, qwen4exp IQ4_NL `draft-mtp n3`,
  prose) with **byte-identical output and identical acceptance** (0.83204).
* Plain decode and prefill unregressed; CPU-only builds unaffected; the `qwen35`/`qwen35moe` models
  (no PLE) stay on their current numbers (verified already clean, §11 session 6).
* **Re-baseline the gfx1201 qwen4exp MTP t/s** figures (they are ~35 % low, §11 session 6) and make
  the harness robust: the MTP gate should not depend on an env var being set.  Consider teaching the
  reporting harness to assert the CPU is quiet (e.g., a core-count sanity check) so a future
  degenerate-scheduling regression cannot silently poison the numbers again.
* Re-run the four-axis + mixed `n7`/`n8`/adaptive comparison **without** the spin confound (the session
  6 table is confounded), then extend it to gfx1151 (the `n_max 8` observation) and gfx1100.

### 13.5 Tools / harness carried over

* `/tmp/mon.py` — per-thread CPU sampler (`/proc/<pid>/task/*/stat` deltas); the instrument that made
  the spin visible (2 cores → 16 cores).
* `gdb -p <pid> -batch -ex "thread apply all bt"` — the stack that located the draft `llama_decode`.
* `GGML_SCHED_DEBUG=1` (with `-lv 5`) — per-graph split/backend assignment; showed the `CPU` split
  ahead of the `Meta` split and its `model.input_embed`/`ple_embd`/`mtp_tok_embd` inputs.
* `OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0` — the A/B that proved the spin was idle-wait, not work.
* Reuse the session-6 command shape: IQ4_NL 9-shard + `mtp-...-shared-Q8_0.gguf`, `-sm tensor`,
  q8_0 KV, `-b/-ub 2048`, `-c 16384`, `-n 1500..3000`, seed 42, temp 0.

### 13.6 Re-run the depth matrix (`n7`/`n8`/adaptive) — the `n_max 8` question

**Explicit follow-up** (maintainer, 2026-09-24): all `n7`/`n8` results must be re-measured on the
spin-free path, not just the one pair.  The **gfx1201 matrix is DONE** (session 6 re-run, §11): with
`OMP_WAIT_POLICY=PASSIVE`, **`n7` beats `n8` on every axis** (C 131.1 vs 107.3, P 119.3 vs 95.1,
K 156.4 vs 129.2, R 75.4 vs 57.1, X 92.2 vs 62.9), acceptance byte-identical, and the adaptive
ceiling-8 controller wins on R/X.  **Remaining:** run the same matrix (R/C/K/P/X × `none`/`n7`/`n8`/
adaptive) on **gfx1151** — the box where the `n_max 8` observation was actually made — and on
**gfx1100**, then, only if a box shows `n8 > n7` with healthy acceptance, investigate whether the
purity-boundary reasoning (first non-pure verify width) explains it rather than the acceptance curve.
Contract reminder: above `n_max 7` use acceptance + MTP-vs-plain throughput, **not**
`plain == draft-mtp`.

### 13.7 The qwen4exp W=9 verify cliff (`n_max` 7→8) — the sharp n8 drop

**Significant, open (session 7) — root-caused and FIXED (session 8, 2026-09-24).**  The n7→n8 MTP drop was not the acceptance curve: it is a sharp,
**qwen4exp-only** step-cost cliff once the verify width reaches 9.  **Session 8 corrects the cause**:
it is the `n_tokens = 8 → 9` **MMVQ→MMQ family boundary** (`MMVQ_MAX_BATCH_SIZE = 8`), not the
`mmq_rdna3_5_id_get_J` tile (J is 16 in both arms).  Routing experts *and* dense weights leave the
vector kernels at 9 columns; qwen4exp's weights have `nrows_x % 128 != 0`, so MMQ selects its generic
non-128-row **`fallback`** config (~3× the per-launch cost), while 27B's ÷128 rows make MMQ the fast
choice (hence the 27B *upward* jump at the same boundary).  Fix: `MMVQ_MOE_MAX_BATCH_SIZE = 16` for
routed experts (arch-independent AMD) + an RDNA4 dense rule that keeps ksplit MMVQ for
`nrows % 128 != 0` shapes at 9..16, each with a kill-switch (`GGML_CUDA_DISABLE_MMVQ_MOE_BAND`,
`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`; folded into the WIP set as
[`patches/0028`](patches/0028-gap-closing-WIP-extend-the-MMVQ-routed-expert-band-and-RDNA4-dense-fallback.patch)).
qwen4exp B=9
**202.6 → 270.0**, B=10..12 +5…+23 %, B≤8
unchanged, `n_max 8` MTP 94.3 → **102.7 t/s**, `plain == n3` byte-identical.  Full record:
[`2026-09-24-qwen4exp-w9-verify-cliff.md`](2026-09-24-qwen4exp-w9-verify-cliff.md).  **gfx1151
revalidation/port is the new brief [`gfx1151-closing.md`](gfx1151-closing.md)** — the MoE band
already fires there (arch-independent), the dense band is RDNA4-gated and needs a decision.  The
`n7`/`n8` matrix (§13.6) still stands and is now the gfx1151 job.
