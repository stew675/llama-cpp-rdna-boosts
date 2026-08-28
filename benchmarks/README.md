# Benchmark results: llama.cpp master vs rdna-boosts

The current, full benchmark suite. **v2 (2026-08-27)** is the complete
gamut; the **v1** record (curl + `/completion`) is preserved for history.

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
