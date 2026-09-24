# Closing the gap — the `gap-closing` / MMB campaign (open work + handover)

**Status:** the **live handover and the open items** of the campaign.  Completed work lives in
[`closed-the-gap.md`](closed-the-gap.md) (dated session records, the 2026-09-20 snapshot body,
appendices, the MTP qualification).  This file is what a fresh session reads first.

**Box:** `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`),
123 GiB unified.
**Model:** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`
+ MTP sidecar `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`.
**Fork:** `~/llama.cpp`, branch **`gap-closing-hostbuf-integrated`**, tip **`df67fd133`**
= delivery r13 + the 12 `beta/mmb-general` patches + gap-closing `0001..0014`/`0016..0027`
(`0023` = the MMB HC16 per-context fix; `0024` = the input-layer GPU offload stopgap;
`0025` = the host-buffer input layer that **supersedes `0024`**; `0026` = the sparse MTP draft
**default ON**).  The `0024` tip `73a391aba` on branch `gap-closing-r13` is the pre-`0025` baseline
kept for A/B.
**Updated:** 2026-09-23 (session 12).

> **No open blockers.**  The HC16-under-MTP bug that gated the campaign is **fixed**
> ([`patches/0023`](patches/0023-mmb-hc16-per-context.patch),
> [`2026-09-23-mmb-hc16-mtp-per-context.md`](2026-09-23-mmb-hc16-mtp-per-context.md)); the MTP/plain
> purity gate is green at 8K/40K/128K and `llama-imatrix` is clean.  The input-embedding CPU burn is
> fixed by the reference's **zero-copy host-buffer path**
> ([`patches/0025`](patches/0025-host-buffer-input-layer.patch),
> [`2026-09-23-host-buffer-input-layer.md`](2026-09-23-host-buffer-input-layer.md)): `integrated =
> prop.integrated` on HIP plus a scheduler guard that keeps host-resident graph inputs off the compute
> backend.  The `GET_ROWS` runs on ROCm0, the ~28 GiB `per_layer_token_embd` stays in host RAM, and
> the 8K/40K/128K output is byte-identical to the pre-`0025` baseline.  **`0025` supersedes
> `0024`** (the single-device heuristic), which is kept on `gap-closing-r13` only as the A/B
> baseline.  See the **Host-buffer input layer** section below.  The **sparse MTP draft prefill is
> now default ON** ([`patches/0026`](patches/0026-mtp-sparse-default-on.patch),
> [`2026-09-23-mtp-sparse-default-on.md`](2026-09-23-mtp-sparse-default-on.md)): the HC16 fix made
> it pure, and the A/B is pp150K **+7.6 %** for −0.3 % at 16K / decode parity.  `LLAMA_MTP_SPARSE=0`
> is the opt-out; the decode/verify arm stays opt-in.

---

## The blocker — **FIXED 2026-09-23**: MMB **HC16** F32-elision under MTP/speculation

The campaign default (MMB on, HC16 on) produced nondeterministic greedy text under MTP at depth
(128K MTP `n_max 1`: five runs -> five hashes) and a 40K draft that diverged from plain; `HC16=0` was
deterministic and pure.  Root cause: HC16's BF16 activation cache/slots and its `bf16_only`/`bf16_copy`
marks were **file-scope, shared by every CUDA backend context**, so the MTP target and its draft head
(the two `llama_context`s) freed each other's activation buffers and leaked each other's marks.  A
second hole: the per-split step-3 marking pass treated "no consumer in this split" as "all consumers
are BF16 readers" and elided a producer whose consumer lived in the other split.

Fix (`patches/0023`): make the whole MMB BF16 state per backend context (an "active context" pointer
set at graph optimize/compute), give the marks an explicit per-context per-graph lifetime, and classify
a producer's consumers over the **whole** scheduled graph (new `full_graph` in
`ggml_backend_graph_optimize_params`) so a cross-split consumer forbids the elision.

Gates: **128K MTP 5/5 one hash `770e770ae7d6` == plain == HC16=0**; **40K plain == dense/sparse
`n1/n3/n5` == `8285d12d40ca`** deterministic; 8K pure (`3553e76d3a9e`); width probe PASS;
`llama-imatrix` NanBeige BF16 clean with the imatrix file byte-identical to HC16=0;
`FLASH_ATTN_QSA`/`GATED_DELTA_NET` 2/2.  40K `-n 200` `n_max 3`: plain 28.4, dense MTP 33.4 t/s with
HC16 on and off identical, so the win is kept.

Full detail, mechanism and the original reproduction recipe:
[`2026-09-23-mmb-hc16-mtp-per-context.md`](2026-09-23-mmb-hc16-mtp-per-context.md).

---

## Host-buffer input layer — **RESOLVED 2026-09-23**

The stopgap `patches/0024` is **superseded by [`patches/0025`](patches/0025-host-buffer-input-layer.patch)**
— see [`2026-09-23-host-buffer-input-layer.md`](2026-09-23-host-buffer-input-layer.md) for the full
record.  Summary:

* The crashing op is the **KV cache store** (`cpy_k`): `k_set_rows<float, long, __half>` = f32 source,
  **I64** indices, f16 destination.  The QSA mask store uses I32 indices (`ggml_indexer_top_k` returns
  `GGML_TYPE_I32`), so the handover's QSA-indexer guess was wrong.
* Root cause: with `integrated = true` the APU accepts the `ROCm_Host` buffer for ROCm0, so the
  scheduler elides the split-input copy and the compute backend reads a **host** graph input in place;
  the host's next-ubatch `set_inputs` then races the in-flight compute (**a view-reached input, e.g.
  the recurrent-state copy, is the one a naive `GGML_TENSOR_FLAG_INPUT` check misses**) and a torn I64
  index turns into an out-of-bounds `k_set_rows` store.  This is the documented #15034 class.
* Fix: report `prop.integrated` again on HIP and add a scheduler guard that forces the split-input
  copy for a host-resident graph input (view chains resolved) while `n_copies <= 1`.  Weights are
  unaffected, so the input embeddings stay zero-copy in `ROCm_Host` and their `GET_ROWS` runs on
  ROCm0 — `0024`'s `n_devices() == 1` heuristic and its ~28 GiB VRAM cost are gone.
* The reference's full input **ring** (`83e8382ba` + `1f2e34819` + hardening, ~500 lines) is the
  follow-up optimisation: it avoids the per-ubatch copy that this guard keeps.  `n_copies <= 1` in the
  guard makes the two compose.
* Multi-GPU: the flag follows `prop.integrated` **per device**, so the policy is per-device with no
  `n_devices()` branch; untested here (single-GPU box) — re-gate `-sm layer`/`-sm tensor` first.
* Acceptance met: CPU 818 % -> 122 %; 8K/40K/128K plain == dense/sparse MTP == the pre-`0025`
  baseline (`3553e76d3a9e` / `8285d12d40ca` / `d140b40f0eee`); width probe PASS at P=1024 and P=32768;
  QSA/GDN/INDEXER_TOPK oracles green; `per_layer_token_embd` (27.8 GiB) in the host buffer.

The old debug aids — `LLAMA_BUF_SEL_DEBUG=1`, `LLAMA_SCHED_BUF_DEBUG=1`, and the new
`GGML_FORCE_NO_INTEGRATED=1` A/B kill-switch — are documented in the session record.


---

## Open items (priority order)

1. **~~Resolve the input-layer placement fully (host-buffer path)~~ — DONE 2026-09-23**
   (`patches/0025`, superseding `0024`).  `integrated = prop.integrated` on HIP plus a scheduler
   guard that forces the split-input copy for a host-resident graph input (view chains resolved)
   while the input ring is off: zero-copy `ROCm_Host` input weights, `GET_ROWS` on ROCm0, ~28 GiB
   out of VRAM, byte-identical to the `0024` baseline at 8K/40K/128K.  Full record:
   [`2026-09-23-host-buffer-input-layer.md`](2026-09-23-host-buffer-input-layer.md).  The reference's
   input ring (`83e8382ba`/`1f2e34819`) remains a follow-up optimisation — the guard's
   `n_copies <= 1` condition composes with it.  Multi-GPU is stated but untested here.
2. **~~Promote the sparse MTP draft to default-on~~ — DONE 2026-09-23** (`patches/0026`).  The
   HC16 fix (`patches/0023`) made the sparse draft pure; the default is flipped with
   `LLAMA_MTP_SPARSE=0` as the opt-out.  A/B: pp150K **937.1 -> 1007.9 t/s (+7.6 %)**, pp16K −0.3 %,
   8K decode parity, 40K text byte-identical; acceptance 0.85035 unchanged; the `LLAMA_MTP_SPARSE_MIN_KV`
   depth gate (32768) and the opt-in decode arm (`LLAMA_MTP_SPARSE_DECODE=1`) are unchanged.  Full
   record: [`2026-09-23-mtp-sparse-default-on.md`](2026-09-23-mtp-sparse-default-on.md).
3. **gfx1100 / gfx1201 validation** of the session-8+10 additions: the new MMB quant types
   (Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 + the IQ2 family), `QSA_SCORE_WMMA`, the derived-indexer default
   (`patches/0021`) and the 32K decode crossover (`patches/0022`).  `beta/mmb-general/gfx1201-s14-gates.md`
   is the checklist (3-GPU `-sm tensor`, q8_0 KV, `-b/-ub 2048`); gfx1100 notes in
   `beta/mmb-general/gfx1100-porting.md`.  The dequant code is arch-neutral and gfx1201 keeps its
   per-type dense policy, so this is apply-and-gate, not a port.  Re-check the crossover claim on
   gfx1201 (it stays dense-always there) and the pool default on both.  **Hand-off briefs for the two
   target machines:** [`gfx1201-closing.md`](gfx1201-closing.md) and
   [`gfx1100-closing.md`](gfx1100-closing.md) — each has the full apply order (r13 + `beta/mmb-general`
   + `wip/closing-the-gap`), the per-patch arch-sensitive inventory, the gate commands, the expected
   `MMB_CFG` row and the port candidates.  Extend `beta/mmb-general/gfx1201-s14-gates.md` /
   `gfx1100-porting.md` with the session results.
   **gfx1201 status (2026-09-23): DONE.**  The full stack applies **25/25** on RDNA4 with no
   apply-time port (the RDNA4 **qsa3**/**mmb** WMMA kernel ports are in the `beta/mmb-general`
   prerequisite, not the closing set); oracles, width purity, `plain == draft-mtp` at 8K/40K/128K, MTP acceptance (0.81388),
   the rule-5 batched gate and PPL parity are all green; closing adds **+1.0…+4.4 %** prefill over
   r13+beta at depth.  One correctness fix was folded into `patches/0004` (the GDN/PLE conv1d
   fusion is not bit-identical under `-sm tensor`; it is now gated to single-device graphs).  See
   [`gfx1201-closing.md`](gfx1201-closing.md) §11 and
   [`2026-09-23-gfx1201-conv-fusion-tensor-split.md`](2026-09-23-gfx1201-conv-fusion-tensor-split.md).
   **`0016` `QSA_SCORE_WMMA` is now PORTED to RDNA4 (2026-09-23, default ON)** — the 4-head
   indexer WMMA kernel gives **+1.7…+12.3 % qwen4exp prefill** at pp8192…65536; oracle 225/225,
   purity holds.  See [`2026-09-23-qsa-score-wmma-rdna4.md`](2026-09-23-qsa-score-wmma-rdna4.md).
   The remaining RDNA4 port candidate is `0003` (gate-mix) — the generic fallback is in use and
   gated.  **gfx1100 §7.2 done (2026-09-24):** the new 26-patch set applies to tree
   `803e6d908a…` and all headline gates reproduce; **MMB quant coverage (`0017`/`0018`) is a win for
   *every* type on gfx1100** (Q4_0 **+15.7…+20.6 %**, Q4_1 **+19.7…+24.5 %**, Q5_0
   **+16.0…+20.6 %** dense; gemma-26B Q4_0 MoE **+10.7 %** routed) — `Q4_0` wins here but *loses* on
   RDNA4 (weak gfx11 MMQ), and gfx1100's mask already enables them, so **no change needed**; the
   `0019`/`0023` imatrix gate is clean+byte-identical.  The `0016`/`0003` gfx1100 arms stay opt-in
   (end-to-end needs a qwen4exp-capable box).  See [`gfx1100-closing.md`](gfx1100-closing.md) §7.2.5.
   **gfx1151 lossy-prefill transfer (`0010`/`0011`) — NEGATIVE 2026-09-23:** the maintainer's
   `LLAMA_HC_BLK16=1 LLAMA_HC_RES16=1 GGML_CUDA_MMB_DOWN16=1` config was tested on gfx1201 and shows
   **no measurable win** (flat at pp8192/32768 under `-sm layer`; under `-sm tensor` the markings do
   not run at all — the meta backend bypasses the CUDA child's `graph_optimize`).  Do not enable
   them on RDNA4.  See [`2026-09-23-gfx1201-lossy-prefill-transfer.md`](2026-09-23-gfx1201-lossy-prefill-transfer.md).
   The **meta-backend `graph_optimize` gap** was the reusable finding (any `graph_optimize`-based
   marking was inert under `-sm tensor`) and is now **FIXED by closing patch `0027`**: the meta
   backend forwards the child pass twice (alloc deps over the whole graph before allocation, marks
   per per-device subgraph after the simple tensors exist).  gfx1201 is verified unregressed; the
   gfx1151/HC16 re-gate on `halo` is the remaining step.  See
   [`2026-09-23-meta-graph-optimize-tensor-split.md`](2026-09-23-meta-graph-optimize-tensor-split.md).
4. **`-ub 16384`** — parked until the managed PLE reader's no-cache parallel-pread fast path is picked
   up (item 13 in the closed record).  Root cause in `closed-the-gap.md` (the full-vocab
   `result_output` reserve + the HC `block_out` pin + the resident PLE table).
5. **qwen4exp adaptive-MTP ceiling sweep** (3/5/7/9/12) — a tuning item, parked until the sparse-draft
   default is settled.  The draft-mtp ceiling and the `--spec-draft-n-max` purity band are separate;
   see `benchmarks/mtp-adaptive-methodology.md`.

### Parked / do not restart without a reason

* `hc_combine_norm_f32_b256` (closed negative, not bit-identical), `concat_transposed` (already gone
  at `-ub 8192`), the reference's `d67d58836` indexer redesign (audited: already in our tree; only
  the MTP-draft attention was missing and is now `patches/0020`), the `nextn_shared_target_tensors`
  work (delivered in r13 block 00).

---

## Done (one-liners) — details in [`closed-the-gap.md`](closed-the-gap.md)

| item | patch(es) | result |
|---|---|---|
| input embedding on the GPU (host-buffer path) | `0025` | restores `integrated=prop.integrated` + a host-input scheduler guard; GET_ROWS on ROCm0, ~28 GiB `per_layer_token_embd` in host RAM, CPU 818 % -> 122 %, 8K/40K/128K byte-identical |
| input embedding on the GPU (single device) — **superseded by `0025`** | `0024` | token_embd/mtp_tok_embd GET_ROWS -> ROCm0; MTP decode CPU 1090 % -> 155 %, byte-identical |
| MMB HC16 under MTP | `0023` | per-context activation state + whole-graph consumer scan; 128K MTP 5/5 one hash == plain == HC16=0 |
| sparse MTP draft prefill — **default ON** | `0020`,`0026` | pp150K +7.6 %, 16K −0.3 %, 8K parity, acceptance 0.85035 |
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

# --- depth prompts (the previous sessions' hash gates use these exact files) ---
head -c 165000 /llm/models/wikitext-2-raw/wiki.train.raw > /tmp/p40k.txt    # 40K gate
head -c 528000 /llm/models/wikitext-2-raw/wiki.train.raw > /tmp/p128k.txt   # 128K gate
# 40K: add -c 40000 --ctx-checkpoints 0 -f /tmp/p40k.txt ; plain==dense/sparse n1/n3 == 8285d12d40ca
# 128K: add -c 131072 --ctx-checkpoints 0 -f /tmp/p128k.txt ; plain==mtp n1 == d140b40f0eee

# --- host-buffer input layer A/B (patches/0025) ---
# CPU during sparse MTP decode: ~122 % default vs ~818 % with GGML_FORCE_NO_INTEGRATED=1
GGML_FORCE_NO_INTEGRATED=1 /usr/bin/time -v ./build-rocm/bin/llama-cli ... # 16384/n1500

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

* Records: [`2026-09-23-mtp-sparse-default-on.md`](2026-09-23-mtp-sparse-default-on.md) (session 12, sparse MTP default ON, `0026`),
  [`2026-09-23-host-buffer-input-layer.md`](2026-09-23-host-buffer-input-layer.md) (session 12, the host-buffer input layer, `0025`),
  [`2026-09-23-input-layer-gpu-single-device.md`](2026-09-23-input-layer-gpu-single-device.md) (session 11, the input-embedding CPU burn, `0024` — superseded),
  [`2026-09-23-mmb-hc16-mtp-per-context.md`](2026-09-23-mmb-hc16-mtp-per-context.md) (session 11, the HC16 fix),
  [`2026-09-22-mtp-sparse-draft.md`](2026-09-22-mtp-sparse-draft.md) (session 10),
  [`2026-09-22-phase2-sparse-qsa-audit.md`](2026-09-22-phase2-sparse-qsa-audit.md),
  [`PLAN-mtp-sparse-draft.md`](PLAN-mtp-sparse-draft.md),
  [`2026-09-22-mmb-eval-callback-f32.md`](2026-09-22-mmb-eval-callback-f32.md),
  and the rest of this directory's `2026-09-*` files.
* History: [`closed-the-gap.md`](closed-the-gap.md).
* Patches: [`patches/`](patches/) (`0001..0014`, `0016..0027`; `0015` superseded by r13 block 00; `0024` superseded by `0025`).
* Delivery policy: `AGENTS.md` (default-on policy, purity rules, pushing policy — **never push the
  `~/llama.cpp` fork**).
