# 2026-09-24 — `0003` `hc_gate_mix` ported to RDNA4: **+5.3…5.8 % qwen4exp prefill**

**Work item 2 of [`gfx1201-closed.md`](gfx1201-closed.md) §12.2**, unblocked by the IQ4_NL
Qwen3.8-Flash-Next download (`/llm/models/Qwen3.8/Flash-Next/IQ4_NL/`).  Box: 3× Radeon AI PRO
R9700 (gfx1201); campaign branch `closing-gfx1201`; 3-GPU `-sm tensor`.

## TL;DR

The fused HC gate GEMM + sigmoid + stream mix (`hc_gate_mix_kernel<4>`) now runs on RDNA4.  It is
**bit-identical** to the unfused chain and a **+5.8 / +5.4 / +5.3 % prefill win** at
pp8192 / 32768 / 65536 on qwen4exp IQ4_NL.  The port is the call-site predicate plus the RDNA4
policy row; the kernel itself needed no change.  Folded into closing patch **`0003`**
(`c421c12ae`); the 26-patch set re-applies and reproduces tree
**`ec54ad65f425c69b4dec4279efa68b0573e4afc8`**.

## Why the port is just the predicate + the policy

* The kernel already goes through the arch-aware MMB shim — `mmb_frag_t`, `mmb_ld_frag`,
  `mmb_wmma_bf16` and `MMB_ACC_M` all have the `#if defined(RDNA4)` arm (the `8*hi + e` accumulator
  map), so the handover's "epilogue hand-rolls the gfx11 map" concern was stale: the epilogue uses
  `MMB_ACC_M(e, cn)` and the comment above it is the gfx11 reading only.
* **No HC16 dependency.**  The kernel reads the `xn` operand out of the BF16 activation cache
  (`ggml_cuda_mmb_cache_lookup(xn)`).  On RDNA4 the HC16 *marks* are RDNA3_5-gated and absent, but the
  cache is still populated independently: `xn` is the `hc_norm` tensor, which is the activation
  (`src1`) of the `w_down` MMB dense GEMM, and `mmb_bf16_activation()` caches every activation it
  converts.  So the `w_down` GEMM warms the cache and the `w_up` gate-mix hits it.  Confirmed by the
  debug counter: **285 fires** on the IQ4_NL model's prefill.

## The change (folded into `0003`)

* `ggml-cuda.cu`: `GGML_CUDA_CC_IS_RDNA3(cc)` → `(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))`
  at the `hc_gate_mix` call site.
* `mmb.cu`: the RDNA4 `mmb_arch_defaults` row's `c.gatemix = 0` → `1` (default ON;
  `LLAMA_HC_GATEMIX=0` is the A/B / bisect opt-out).
* A gated `LLAMA_HC_GATEMIX_DEBUG=1` diagnostic was added at the fusion's guard points (it reports
  which guard rejected, and `FIRED xn=… w=… K M E T` on success) — this fusion has no op-level
  oracle, so the debug aid plus the text gate are the correctness evidence.

## Validation (gfx1201, qwen4exp IQ4_NL 9-shard + the shared-Q8_0 MTP sidecar, 3-GPU `-sm tensor`,
q8_0 KV, `-b/-ub 2048`)

| gate | command / metric | result |
|---|---|---|
| fires | `LLAMA_HC_GATEMIX_DEBUG=1`, `LLAMA_HC_GATEMIX=1`, pp2048 | **285 `FIRED`** (`xn=hc_norm-N`, `w=blk.N.hc_{attn,ffn}_up.weight`, E=2560, K=320, M=10240) |
| default == off | `-n 64` greedy, seed 42, prose | **byte-identical** `471d102e7b7d` (`=1`, default and `=0`) |
| MTP purity | `--spec-type none` vs `draft-mtp --spec-draft-n-max 3` (+ `-md`), `-n 64` | **byte-identical** `471d102e7b7d` |
| width probe | `test-logits-width-probe <IQ4_NL> prose 1024 512` | **PASS (worst maxdiff 0)** — the fusion is prefill-only (`T >= min_t = 512`), the `W=1..8` band never takes it |
| `MUL_MAT` oracle | `test-backend-ops -o MUL_MAT` | 1297/1297 |

**Perf A/B** (`llama-bench`, interleaved, 3 rounds **after** a warm-up round, `-r 5`; `=1` vs `=0`):

| depth | GATEMIX=1 (3 rounds) | GATEMIX=0 (3 rounds) | Δ |
|---|---:|---:|---:|
| pp8192  | 3346.9 / 3345.8 / 3340.2 | 3167.4 / 3158.7 / 3160.0 | **+5.8 %** |
| pp32768 | 3247.3 / 3263.8 / 3271.0 | 3082.6 / 3095.4 / 3098.3 | **+5.4 %** |
| pp65536 | 3127.0 / 3127.9 / 3128.5 | 2970.1 / 2971.4 / 2970.3 | **+5.3 %** |

(The first cold round measured pp8192 2684–3049 with 6 % spread — the model is a near-VRAM-limit
fit and the first large prefill is cold; the warm rounds above are stable to <0.3 %.)

## Notes / follow-ups

* The remaining `LLAMA_HC_GATEMIX` opt-in is gfx1100 (`RDNA3_0`): its own end-to-end A/B needs a
  qwen4exp-capable box (the 24 GiB gfx1100 cannot fit the model), per `gfx1100-closing.md` §7.2.2.
* `0017`/`0018` still need no RDNA4 change beyond the Q4_1/Q5_0 port
  ([`2026-09-23-mmb-q41-q50-rdna4.md`](2026-09-23-mmb-q41-q50-rdna4.md)); `MXFP4`/`NVFP4`/`IQ2`
  remain tooling-unreachable.
