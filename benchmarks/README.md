# Benchmark results: llama.cpp master vs rdna-boosts

The current, full benchmark suite. **v2 (2026-08-27)** is the complete
gamut; the **v1** record (curl + `/completion`) is preserved for history.

> **MTP gate (2026-09-02):** adaptive/draft MTP is NOT covered by these
> decode suites (llama-bench/benchy decode 1 token/step and never touch
> the speculative verify batch or draft context). MTP regressions slip
> through them (worked example: the 2026-09-02 block-13/block-08 MoE MTP
> collapse, 0/1527 draft acceptance). The MTP protocol, baselines and
> acceptance gate live in [mtp-adaptive-methodology.md](mtp-adaptive-methodology.md).
>
> **Freshness note:** these are the v1/v2 records of the 11-block era
> (builds: master `fe235f434` vs `build-rdna-boosts` @ `a265041b1`). The
> current delivery adds block 12 (hybrid all-reduce, RDNA4-gated) — its
> numbers live in `patches/README.md` (tg64 38.12 / tg512 41.08 on the
> clean-apply build; depth-16384 3-GPU hybrid 38.71 t/s unpinned) and the
> benchy depth-16384 protocol in `wip/HANDOFF.md`.

## 2026-09-20 — Qwen3.8-Flash-Next IQ4_XS prefill: delivery base vs the `mmb` WIP

**[2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md](2026-09-20-qwen4exp-iq4xs-prefill-wip-vs-base.md)** —
prefill (`-n 0`) on the 94 GiB qwen4exp model, gfx1151/ROCm 7.14, bf16 KV, `-b/-ub 2048`, `-r 3`, across
the **true delivery base** (`8a2567e1e`, unmodified — built in a separate worktree) and two WIP arms:
MMB/HC16 off and all gates on.  Headline: the WIP is **+32 % at pp2048 and +43–48 % from pp4096 to
pp32768** end to end, of which ~+14–16 % at 8K–32K is the WIP's always-on work (qsa3, always-QSA,
non-temporal, indexer) and ~+27–29 % is MMB/HC16.  Also records the base's pp4096 QSA dense-shortcut
cliff that the always-QSA flip removes.  Read it before quoting any prefill delta: the older "MMB off"
arm was the WIP tree with only the MMB gates off, **not** the delivery base.

## Adaptive MTP — four-workload records

**[2026-09-15-adaptive-mtp-tuning.md](2026-09-15-adaptive-mtp-tuning.md)** is the **current**
adaptive-MTP controller record: it replaced the mean-reverting table with the **credit bucket** and
tuned it (the block-01 amendment).  Read this one for the delivery's controller numbers.

**[2026-09-13-adaptive-mtp-4-axis-n12.md](2026-09-13-adaptive-mtp-4-axis-n12.md)** is the
authoritative four-workload measurement of the **pre-tuning table controller**:
`draft-mtp-adaptive` at its recommended **ceiling 12** (the
2026-09-13 clamp relaxation), across the R/P/C/K workloads, at a realistic **`-n 3000`** with
reasoning pinned per axis (`--reasoning off` for P/C/K).  Against fixed `n3` it is reasoning
-1%, prose +13%, code +28%, recall +61%; a 256-token run measured the warm-up and inverted the
code ranking.  It supersedes all earlier adaptive records -- both the `n_max <= 7` clamp and the
`-n 256` / default-reasoning protocol were artifacts.  **Beware (2026-09-17):** its prose acceptance
(0.50654) is the *table's*; the delivery credit bucket's is 0.60232, so cite this record as the table
arm, **not** as "the delivery".  The gate protocol (length, reasoning,
baselines) lives in [mtp-adaptive-methodology.md](mtp-adaptive-methodology.md).

## Block 12 — hybrid all-reduce env matrix (2026-08-30)

**[block12-hybrid-ar-matrix.md](block12-hybrid-ar-matrix.md)** — the
`GGML_CUDA_ALLREDUCE` (hybrid / nccl / internal / none) x
`NCCL_P2P_DISABLE` (0/1) matrix on 2x R9700 (gfx1201), ROCm 7.14, at
depth-16384: hybrid wins prefill (1620 t/s), internal/none unchanged by
P2P (as designed), RCCL legs lose ~6% with P2P disabled.

## 2026-09-13 — Qwen3.8-27B Unc Q8 ngram-mod 16/2/96

**[2026-09-13-qwen38-unc-ngram-mod.md](2026-09-13-qwen38-unc-ngram-mod.md)** — HIP 12-set, 2× R9700. Stock `--spec-type ngram-mod` with **n-min 2** drafts on dense Unc (farm 24/48/64 did not). Prefill ≈ nospec; MTP n-max 3 still wins unique-prose median. Not qwen4exp managed-ngrams.

## v2 — the current results (llama-benchy live-server suite)

**[v2-results.md](v2-results.md)** is the canonical results document:
24 throughput rows (16 ROCm + 8 Vulkan) × f16/bf16 KV, plus 18 PPL
corners.

Full matrix: build (master `fe235f434` vs rdna-boosts `a265041b1`) × KV
cache (f16 vs bf16) × 4 model/card sets (1-card Q6_K, 1-card Q4_K_XL,
2-card Q8_0, 3-card Q8_0), on both ROCm and Vulkan.

- **[v2-results.md](v2-results.md)** — the headline numbers, tables, and analysis.
- **All charts on one page**: [graphs/ALL-CHARTS.md](graphs/ALL-CHARTS.md)
  (16 charts, split by KV type: 3 series per chart, red=master,
  amber=boosts, blue=vulkan).
- **[benchy-methodology.md](benchy-methodology.md)** — protocol (llama-benchy
  v0.4.0, `--pp 2520 --tg 240 --depth 0 4096 8192 16384 32768 65536 131072
  --no-cache --runs 2`), harness validation, and why it matches v1.
- Harness: [`scripts/`](scripts/) (`benchy-run.sh`, `run-benchy-suite.sh`,
  `run-benchy-vulkan-suite.sh`, `run-all.sh`, `run-vulkan-all.sh`,
  `benchy-json-to-md.py`, `make-v2-graphs.py`).
- Raw data: [`results/benchy/`](results/benchy/) (per-row JSON + md) and
  [`results/ppl/`](results/ppl/).

### v2 headline

Master's ROCm BF16 KV silently degrades decode with depth (−2% shallow to
**−39 to −51% at 128K**); rdna-boosts fixes that and goes further — its
BF16 config is **faster than master-f16 at every depth and more accurate
(PPL) on every model**, and even its f16 config beats master-f16 at the
same KV type. Vulkan is a genuine native-BF16 reference with superb
single-card prefill, but its decode collapses on multi-card and its BF16
costs prefill at depth. No performance or accuracy reason to choose
either baseline approach.

## v1 — the historical record

**[v1-results.md](v1-results.md)** (2026-08-26) was the first benchmark
pass, using a curl + `/completion` protocol against a llama.cpp server.
It covered 4 configs (A/B/C/V) × 1/2/3 GPUs. The v2 suite supersedes it
(wider KV matrix, both backends, PPL corners), and v2's A/B/C/V corners
reproduce v1's baselines exactly — so the two are directly comparable.

## Environment

Machine: 3× Radeon R9700 (gfx1201, 34 GiB, ~640 MB/s each), ROCm 7.14,
9950X3D host. ROCm builds: `build-rocm` = master `fe235f434`,
`build-rdna-boosts` = `a265041b1` (all 10 blocks), `build-vulkan` = master
`fe235f434` (RADV/Mesa 26.1.7). Models: Q8_0 (29 GB), Q6_K (22.9 GB),
Q4_K_XL (17.6 GB); wikitext-2 for PPL.
