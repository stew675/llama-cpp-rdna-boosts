# WIP: pwilkin's tiled Gated Delta Net — port scoping (TODO item 1)

**Status:** scoping complete 2026-09-13; a minimal RDNA4 prototype port was built and measured
(one 1-GPU gfx1201 box, Qwen3.8-27B Q6_K).  This tree is **not part of the delivery** — the WIP
rule in `AGENTS.md` applies.  Nothing here may be folded into `patches/` without the promotion
path (`beta/` staging + env-gated A/B + maintainer go-ahead).

Tracking branch: `tiled-delta-net` (this repo), path `wip/tiled-gdn/`.
Source under study: `pwilkin/llama.cpp` branch `strix-halo`, commit **`964c6f2f0`** ("ggml-cuda:
tiled gated delta-net for large prefill batches"), plus the more mature form on the related
`strix-halo-for-halobox` branch, commit **`8ab5a8373`** (by Gaetan Puleo) ("hip: DPP reductions and multi-column tiled GDN
kernel on RDNA3.5", which also carries a KDA tiled variant).

---

## TL;DR verdict

1. **Numerics: the tiled kernel is bit-exact.**  It is not a quality trade-off — pwilkin's journey
   page records the reduction as bit-identical and attributes the 0.21 % perplexity cost to the
   *chunked* GDN rewrite.  We confirmed it on gfx1201: tiled and the sequential kernel produce an
   **identical** PPL vector, while the delivery's chunked bf16 default carries a small
   near-lossless deviation (as documented).
2. **A faithful port is possible and portable** (DPP reduction + RDNA4 compile fine; 46/46
   backend tests at the tight gate), and it is roughly **1.7–1.9× the sequential kernel**.
3. **But it is not a performance win for this repo.**  On the maintainer's gfx1201 the delivery's
   chunked bf16 GDN is ~**5× faster** than the tiled kernel at the op level, and using tiled as
   the default costs ~**4 % end-to-end prefill** (pp2048/pp8192) versus chunked bf16.  The entire
   GDN prefill lever is only worth ~**9 %** end-to-end at pp2048, and tiled captures roughly half
   of it.
4. **So the honest conclusion is a quality play, not a speed play** — and this repo already made
   the bf16-compute trade deliberately.  The tiled kernel is worth landing only as (a) a
   **bit-exact opt-in / fallback** for the chunked path, (b) a **KDA prefill** path (chunked is
   non-KDA only), or (c) a **gfx1100/RDNA3 fallback** where the gfx11 NW16 chunked kernel may not
   fit.  As a default prefill kernel it loses to what we already ship.
5. **The journey's headline ~2.2× is not the GDN.**  It is the qwen4exp prefill stack.  The clean
   weight-dependent term is pwilkin's `mmb.cu` — an IQ4_NL-only dequant-to-BF16 WMMA GEMM (1.42× in
   his finished-stack ablation) with **no counterpart in our tree**; that is the "special weight
   set" signature.  The PLE reader and its prefetch are shared on both sides, so they are a weak
   explanation.  The step-11 2.37× is a *walk* number and does not appear in pwilkin's own
   finished-stack ablation — see [`05-where-the-speed-comes-from.md`](05-where-the-speed-comes-from.md).

### Answer to the session question

> *Is it possible to make his tiled GDN both generic AND perform similarly to his?*

- **Generic:** yes, with bounded work (it is already templated; the gaps are per-`S_v` tile
  configs, a KDA variant — which exists in the halobox lineage — and an arch/`n_seqs` retune).
- **Perform similarly to his:** only *relative to the sequential kernel he was comparing
  against*.  We reproduced ~1.7–1.9× vs sequential on gfx1201 (he saw ~2–3× on gfx1151).
  Against **this repo's existing chunked bf16 baseline it is ~5× slower** per op and ~4 % slower
  end-to-end, because his 2.37× headline was measured against **stock upstream sequential**,
  a bottleneck this repo already removed more aggressively.

---

## Key numbers (gfx1201, 1× R9700, Qwen3.8-27B Q6_K, H_v=48 / S_v=128 / n_seqs=1)

GDN op time, `test-backend-ops perf` (µs/run), strictly one process at a time:

| n_tokens | chunked bf16 (default) | sequential | chunked fp32 | tiled 16×4 | tiled 8×8 |
|---:|---:|---:|---:|---:|---:|
| 64   | **28.49** | 129.04 | 99.98 | 77.30 | 73.37 |
| 256  | **64.59** | 489.63 | 300.92 | 280.06 | 267.75 |
| 512  | **107.32** | 977.73 | 593.00 | 551.15 | 525.79 |
| 1024 | **230.36** | 1970.68 | 1184.18 | 1130.31 | 1077.84 |

Speed-up vs sequential: chunked bf16 **4.5–9.1×**, tiled **1.7–1.9×**, chunked fp32 1.3–1.7×.

End-to-end `llama-bench` prefill (t/s), same box/model:

| config | pp2048 | pp8192 (ub 8192) |
|---|---:|---:|
| sequential (exact reference) | 944.85 | 882.41 |
| **tiled 8×8 (exact)** | 989.06 | 923.24 |
| default **chunked bf16** | **1028.62** | **953.51** |

- chunked bf16 is **+8.9 %** (pp2048) / **+8.1 %** (pp8192) over the sequential kernel.
- tiled is **+4.6 %** over sequential, but **−4.0 %** (pp2048) / **−3.3 %** (pp8192) vs chunked bf16.

Wikitext-2 PPL (27B Q6_K, `-c 512`, 64 chunks):

| path | per-chunk vector | Final PPL |
|---|---|---|
| sequential (exact) | `4.3053, 6.0685, …` | **6.5078** |
| tiled 16×4 | **identical to sequential** | **6.5078** |
| tiled 8×8 | **identical to sequential** | **6.5078** |
| chunked bf16 (default) | differs from chunk 1 (`4.3101, …`) | 6.5092 (**+0.0215 %**) |

This confirms the numerics directly: **the tiled kernel is bit-neutral**, and the delivery's
chunked bf16 path carries a tiny near-lossless deviation (0.02 % here, consistent with its
documented model).

---

## Files

| file | what |
|---|---|
| [`01-kernel-analysis.md`](01-kernel-analysis.md) | what the tiled kernel does and why it is fast; the two published configs; the bit-exactness argument |
| [`02-port-assessment.md`](02-port-assessment.md) | what "generic" means per axis; integration options; effort and risk register |
| [`03-validation-gfx1201.md`](03-validation-gfx1201.md) | the prototype port, methodology, raw measurements, correctness/PPL evidence |
| [`04-handover.md`](04-handover.md) | session handover, state of the scratch fork, recommended next experiments |
| [`05-where-the-speed-comes-from.md`](05-where-the-speed-comes-from.md) | decomposition of the journey's ~2.2×: weight/loading path (`mmb`, `on-direct`), not the GDN |
| [`reference/`](reference/) | pwilkin's extracted commits, the journey page, and our prototype port patch |

---

## One-paragraph bottom line

pwilkin's tiled GDN is a well-engineered, **numerically exact** replacement for the stock
sequential GDN scan, and it ported to gfx1201 cleanly enough to validate that claim.  But its win
is defined against the sequential scan, which this repo already superseded with the chunked
bf16/WMMA kernel (block 02).  Porting it as a default would therefore *regress* prefill by ~4 %
while buying a ~0.02 % PPL improvement over a path that was already documented as
near-lossless.  The defensible work, if any, is narrow and quality-motivated: an opt-in exact
mode, a fallback when chunked is off, and — most interestingly — **KDA prefill**, which the
chunked path does not cover at all.
