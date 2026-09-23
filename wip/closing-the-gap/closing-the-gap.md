# Closing the gap — the `gap-closing` / MMB campaign (open work + handover)

**Status:** the **live handover and the open items** of the campaign.  Completed work lives in
[`closed-the-gap.md`](closed-the-gap.md) (dated session records, the 2026-09-20 snapshot body,
appendices, the MTP qualification).  This file is what a fresh session reads first.

**Box:** `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`),
123 GiB unified.
**Model:** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`
+ MTP sidecar `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.
**Fork:** `~/llama.cpp`, branch **`gap-closing-r13`**, tip **`d8334f929`**, tree **`fa185bbb…`**
= delivery r13 + the 12 `beta/mmb-general` patches + gap-closing `0001..0014`/`0016..0022`.
**Updated:** 2026-09-22 (session 10).

> **Rule 0: fix the HC16 bug below before promoting anything else.**  The campaign's default build
> (`GGML_CUDA_MMB_HC16=1` on RDNA3_5) produces wrong/nondeterministic output under any MTP at depth.
> Interim workaround for any gate: `GGML_CUDA_MMB_HC16=0`.

---

## The blocking bug — FIX THIS FIRST: MMB **HC16** F32-elision under MTP/speculation

### Symptom

With the campaign default (MMB on, HC16 on):

* **128K MTP is nondeterministic**: the same command produces a different greedy text every run
  (5 runs → 5 hashes), so `plain == draft-mtp` fails.
* **40K sparse-draft diverges** from plain deterministically (draft MTP with `LLAMA_MTP_SPARSE=1`).
* The **plain** path is always deterministic; at 120K/136K the MTP also happened to be stable, so it
  is intermittent and allocator/depth dependent — a **race**, not a monotonic depth effect.

### Reproduction

```sh
cd ~/llama.cpp
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MD=/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
head -c 528000 /llm/models/wikitext-2-raw/wiki.train.raw > /tmp/p128k.txt

run_once() {  # $1 = extra env assignments (string)
    env $1 timeout 3000 ./build-rocm/bin/llama-cli -m "$MU" -md "$MD" -ngl 99 -fa auto \
      -ctk f16 -ctv f16 -c 160000 -b 2048 -ub 2048 -n 100 --seed 42 --temp 0 \
      --single-turn --no-display-prompt --reasoning off --ctx-checkpoints 0 \
      --spec-type draft-mtp --spec-draft-n-max 1 -f /tmp/p128k.txt > /tmp/race.out 2>/dev/null
    python3 /home/stew675/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/race.out
}

# FAIL: run 3-5x, get a different hash each time
for i in 1 2 3 4 5; do run_once ""; done
# PASS: deterministic, and == the plain (--spec-type none) hash
for i in 1 2 3; do run_once "GGML_CUDA_MMB_HC16=0"; done
# PASS (whole MMB off; exonerates everything but HC16)
for i in 1 2 3; do run_once "GGML_CUDA_MMB=0"; done
# STILL FAILS (exonerates the BF16 weight copy)
for i in 1 2 3; do run_once "GGML_CUDA_MMB_BF16W=0"; done
```

40K equivalent (sparse-draft purity), `prompts`-free, `--ctx-checkpoints 0`, `-n 200`, prompt
`/tmp/p40k.txt` from the same `head -c 165000`:

| config | plain | default draft | sparse draft (`LLAMA_MTP_SPARSE=1`) |
|---|---|---|---|
| MMB default | `8285d12d40ca` | `8285d12d40ca` | **`c0a3bda5dff4` (wrong)** |
| `GGML_CUDA_MMB=0` | `a40528f24f2d` | `a40528f24f2d` | `a40528f24f2d` |
| `GGML_CUDA_MMB_HC16=0` | `8285d12d40ca` | `8285d12d40ca` | `8285d12d40ca` |

### Mechanism (what was established)

* HC16 marks a producer tensor **BF16-only** (`g_mmb_bf16_only`, file-scope in
  `ggml/src/ggml-cuda/mmb.cu:1435`) so the producer elides its F32 output, and every consumer re-reads
  a BF16 copy from the per-graph activation cache (`g_mmb_bf16_copy`/`_slot`, `ggml_cuda_mmb_cache_lookup`
  `mmb.cu:1865`).  The marking pass is in `ggml_backend_cuda_graph_optimize` (`ggml/src/ggml-cuda/ggml-cuda.cu`
  ~6640–6760, `elide_f32 = params == nullptr || !params->has_eval_callback`, line ~6661).
* The actual BF16 buffers/cache are cleared at the **start of every compute**
  (`ggml_cuda_mmb_begin_graph()` at `ggml-cuda.cu:6542`, impl `mmb.cu:1899`), but the **marks** are
  cleared only on the **next `graph_optimize`** — `ggml-cuda.cu:6648-6650`, keyed on
  `g_mmb_marks_first_split = cgraph->nodes[0]` (an allocator-reused pointer) plus
  `g_mmb_marks_after_compute` (set at `ggml-cuda.cu:6612`, end of each compute).
* The scheduler calls `graph_optimize` **per split** (`ggml/src/ggml-backend.cpp:1472`,
  `&split->graph`) for **all** splits, then computes all splits.  A mark set while optimizing one
  split therefore lives in a global set and is visible to the next split's compute — where the
  producer did not run (its F32 was elided in the earlier split) and the cache was re-cleared, so the
  consumer reads a never-written/stale F32.  The MTP loop's alternating target/draft decodes (and the
  sparse draft's extra nodes/splits) change the split boundaries and the allocator layout, which is
  why it is intermittent.
* This is the **same class** as the session-9 eval-callback bug, which `patches/0019` fixed by adding
  `has_eval_callback` to `ggml_backend_graph_optimize_params` (set at `ggml-backend.cpp:1465`) and
  standing the elision down (`ggml-cuda.cu:6661`).  The eval-callback fix is **not sufficient** for
  MTP.

### Fix direction (pick one, then gate)

1. **Per-split / per-compute mark scoping (root fix).**  The marks are only valid for the compute of
   the split that set them.  Make the mark set scoped to the current split graph (e.g. key on the
   split's `cgraph` pointer and clear on split change), or move the marking into `graph_compute` for
   the split being computed.  Confirm first whether `graph_optimize` under MTP is called with whole
   graphs or splits (the symptom says splits) and whether the scheduler reuses an optimized graph
   (skipping optimize on a reused split is the other leak path — then the marks are stale from a
   previous compute).
2. **Stand-down (minimal, mirrors `patches/0019`).**  Plumb a "speculation / MTP active" flag
   (or a generic "this graph may be split and reused across computes") into
   `ggml_backend_graph_optimize_params` and force `elide_f32 = false` for the MTP target/draft
   context.  Cheapest, loses the HC16 win only under MTP.
3. **Interim only:** `GGML_CUDA_MMB_HC16=0`.

### Gates after the fix

* The 128K reproduction above: **5 runs, one hash, == plain**.
* 40K: plain == default draft == sparse draft (`8285d12d40ca` with HC16 on).
* `llama-imatrix` on `Nanbeige4.2-3B-BF16` still clean (the `patches/0019` gate) and `in_sum2`
  byte-identical to `HC16=0`.
* Byte-identity of the plain path vs the pre-fix build at the depths the fix touches; width probe
  PASS; the usual oracles.

---

## Open items (priority order)

1. **Fix the HC16 F32-elision under MTP/speculation** (above).  This is the only true blocker.
2. **Promote the sparse MTP draft to default-on** once (1) is fixed: it is already pure with
   `HC16=0` (40K plain == sparse) and the decode is parity/slight-win with the pool on.  Flip
   `qwen4exp_mtp_sparse_enabled()` / the `mtp_sparse` default in `llama-model.cpp`; re-run the gates.
   Detail: [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md).
3. **gfx1100 / gfx1201 validation** of the session-8+10 additions: the new MMB quant types
   (Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 + the IQ2 family), `QSA_SCORE_WMMA`, the derived-indexer default
   (`patches/0021`) and the 32K decode crossover (`patches/0022`).  `beta/mmb-general/gfx1201-s14-gates.md`
   is the checklist (3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`); gfx1100 notes in
   `beta/mmb-general/gfx1100-porting.md`.  The dequant code is arch-neutral and gfx1201 keeps its
   per-type dense policy, so this is apply-and-gate, not a port.  Re-check the crossover claim on
   gfx1201 (it stays dense-always there) and the pool default on both.
4. **`-ub 16384`** — parked until the managed PLE reader's no-cache parallel-pread fast path is picked
   up (item 13 in the closed record).  Root cause in `closed-the-gap.md` (the full-vocab
   `result_output` reserve + the HC `block_out` pin + the resident PLE table).
5. **qwen4exp adaptive-MTP ceiling sweep** (3/5/7/9/12) — a tuning item, parked until 1–2 land.  The
   draft-mtp ceiling and the `--spec-draft-n-max` purity band are separate; see
   `benchmarks/mtp-adaptive-methodology.md`.

### Parked / do not restart without a reason

* `hc_combine_norm_f32_b256` (closed negative, not bit-identical), `concat_transposed` (already gone
  at `-ub 8192`), the reference's `d67d58836` indexer redesign (audited: already in our tree; only
  the MTP-draft attention was missing and is now `patches/0020`), the `nextn_shared_target_tensors`
  work (delivered in r13 block 00).

---

## Done (one-liners) — details in [`closed-the-gap.md`](closed-the-gap.md)

| item | patch(es) | result |
|---|---|---|
| sparse MTP draft (opt-in) | `0020` | pp150K +6.9 %, memory fixes the plan missed |
| QSA derived indexer default ON | `0021` | +9.1 % @80K / +14.6 % @150K decode, byte-identical |
| gfx1151 decode crossover 64K→32K | `0022` | 48K +1.8 %, 64K +4.6 % |
| HC16 eval-callback fix | `0019` | imatrix clean, `in_sum2` byte-identical |
| `QSA_SCORE_WMMA` | `0016` | pp32768 +1.0 %, 225/225 |
| MMB quant coverage (5 types + IQ2 family) | `0017`,`0018` | up to +25 % / +16.5 % |
| QSA scorer trim | `0014` | pp8192 +0.5 % |
| shared-NextN MTP | r13 block 00 | 0 draft errors |
| HC BF16 streams (opt-in) | `0011` | +4.9 %/+4.8 % |
| `mmb_cvt` `out_xn` | `0012` | +3.3 %/+3.2 % |
| prefill indexer relu-sum | `0013` | pp32768 +1.8 % |
| lazy-mode semantics | `0009` | text-identical |
| MoE BF16 epilogue (opt-in) | `0010` | 1479→846 ms |
| items 1–8, 13–16 (Phase 1) | `0003`–`0008` | all closed; see the index in `closed-the-gap.md` |

---

## Reproduce / gates (copy-paste)

```sh
# build (halo / gfx1151)
cd ~/llama.cpp
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
#   or the fast loop: cmake --build build-rocm --target llama-cli llama-bench -j 16
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MD=/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf

# --- purity gate (intra-build; ALWAYS use --ctx-checkpoints 0 at depth) ---
# plain vs default draft vs (opt-in) sparse draft must be byte-identical
for arm in "" "LLAMA_MTP_SPARSE=1"; do
  env $arm ./build-rocm/bin/llama-cli -m "$MU" -md "$MD" -ngl 99 -fa auto -ctk f16 -ctv f16 \
    -c 8192 -b 2048 -ub 2048 -n 200 --seed 42 --temp 0 --single-turn --no-display-prompt \
    --reasoning off --spec-type draft-mtp --spec-draft-n-max 3 \
    -f prompts/prose-rdna-boosts.txt 2>/dev/null > /tmp/purity.out
  python3 <repo>/scripts/extract-generated.py /tmp/purity.out
done

# --- width probe (P=1024 stock; extend tests/test-logits-width-probe.cpp for deeper P) ---
./build-rocm/bin/test-logits-width-probe "$MU" prompts/prose-rdna-boosts.txt 1024 512

# --- op oracles ---
./build-rocm/bin/test-backend-ops -o FLASH_ATTN_QSA      # 26/26
./build-rocm/bin/test-backend-ops -o GATED_DELTA_NET     # 46/46
./build-rocm/bin/test-backend-ops -o INDEXER_TOPK
./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT

# --- MTP acceptance (Gate 4, -n 3000, reasoning pinned) ---
# see benchmarks/mtp-adaptive-methodology.md; qwen4exp reference cell 0.44262
```

**Perf A/B protocol:** use **`-b/-ub 4096`** for A/Bs (the `-ub 8192` memory-pressure confound), and
**`--ctx-checkpoints 0`** for anything at depth (the checkpoint save/restore makes depth MTP runs
nondeterministic — see the closed record's methodology note).

**Purity gotcha (do not forget):** the MTP path is only output-equivalent to plain *within one
build and one env*; `GGML_CUDA_ALLREDUCE=nccl` is **not** a bit-identical reference under `-sm tensor`,
and the HC16 bug above makes depth MTP nondeterministic until fixed.

---

## References

* Records: [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md) (session 10),
  [`2026-09-22-phase2-sparse-qsa-audit.md`](2026-09-22-phase2-sparse-qsa-audit.md),
  [`PLAN-mtp-sparse-draft.md`](PLAN-mtp-sparse-draft.md),
  [`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md),
  and the rest of this directory's `2026-09-*` files.
* History: [`closed-the-gap.md`](closed-the-gap.md).
* Patches: [`patches/`](patches/) (`0001..0014`, `0016..0022`; `0015` superseded by r13 block 00).
* Delivery policy: `AGENTS.md` (default-on policy, purity rules, pushing policy — **never push the
  `~/llama.cpp` fork**).
