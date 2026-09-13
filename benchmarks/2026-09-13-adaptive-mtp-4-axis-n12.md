# Adaptive MTP across the four workloads, ceiling 12, `-n 3000` (2026-09-13)

The definitive four-axis adaptive-MTP record: `draft-mtp-adaptive` (block 001) at its recommended
ceiling **12**, across R/P/C/K, at a **realistic generation length**.

> **Protocol: `-n 3000`, reasoning pinned per axis (2026-09-13).**  Both matter, and both were wrong in
> the first cut of this record and in [2026-09-13-adaptive-mtp-4-axis.md](2026-09-13-adaptive-mtp-4-axis.md).
>
> * **Length.**  A short run measures the warm-up, not the mode.  The drafter needs context to predict,
>   and the controller needs hundreds of verify rounds to settle.  At `-n 256` the code axis at ceiling
>   12 read **-5%** vs fixed `n3` (mean accepted length 4.32, the controller still climbing); at
>   `-n 3000` it is **+28%** (mean length 7.02).  `-n 2000` is the floor; the gate uses `-n 3000`.
> * **Reasoning.**  Qwen3.8 emits a thinking trace for instruction-like prompts, so a run at the
>   template default measures thinking, not content (`-n 256` produced no Python at all).  Use
>   `--reasoning on` for R and `--reasoning off` for P/C/K.

The four axes are the workloads the mode is designed for: hundreds of lines of code (C), a
multi-thousand-word prose piece (P), a multi-thousand-character derivation (R), and a full verbatim
recall passage (K).

## Environment

* 1x R9700 (gfx1201), ROCm 7.14, one GPU used (`HIP_VISIBLE_DEVICES=0`)
* Build: the 2026-09-13 issue-#30 delivery (canonical tip `c45244c72`, tree `a5683e1b008e`)
* Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes), f16 K/V, `-fa auto -ngl 99`
* Drafter: the MTP head built into the model GGUF (`blk.64.nextn.*`, `nextn_predict_layers = 1`).
  **No `-md`**.
* `-n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048`

Prompts: `prompts/reasoning.txt` (R), `prose-rdna-boosts.txt` (P), `code-python.txt` (C),
`recall.txt` (K).  See `../prompts/README.md`.

## Commands

```sh
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
for axis in reasoning prose-rdna-boosts code-python recall; do
  case "$axis" in reasoning) REA=on;; *) REA=off;; esac
  for spec in "--spec-type none" \
              "--spec-type draft-mtp --spec-draft-n-max 3" \
              "--spec-type draft-mtp-adaptive --spec-draft-n-max 12"; do
    build/bin/llama-cli -m "$M" --reasoning $REA $spec -f prompts/$axis.txt \
      -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```

Acceptance and mean accepted length come from the `-lv 4` `draft acceptance = ... , mean len = ...`
line; generation t/s from the eval time line.

## Results (`-n 3000`)

| axis | plain t/s | stock `n3` t/s | stock `n3` acc | delivery `n3` t/s | delivery `n3` acc | `adaptive n12` t/s | `adaptive n12` acc | `adaptive n12` mean len |
|---|---|---|---|---|---|---|---|---|
| reasoning (R) | 28.9 | 47.3 | 0.59659 | 46.4 | 0.57781 | 46.0 | 0.57632 | 2.74 |
| prose (P) | 28.4 | 56.3 | 0.79948 | 55.9 | 0.79379 | **63.4** | 0.50654 | 5.10 |
| code (C) | 28.8 | 62.9 | 0.91079 | 63.7 | 0.91663 | **81.8** | 0.57863 | 7.02 |
| recall (K) | 28.9 | 68.0 | 0.99200 | 68.1 | 0.99200 | **109.6** | 0.96320 | 8.95 |

**Adaptive ceiling 12 vs fixed `n3`** (same build): R -1.0%, P **+13.3%**, C **+28.5%**, K **+61.0%**.
**Ceiling 12 vs ceiling 7**: R flat (46.0 both), P 50.2 -> **63.4** (+26%), C 60.4 -> **81.8** (+35%),
K 76.0 -> **109.6** (+44%).  Ceiling 12 is where the mode pays on every content axis; the old ceiling 7
left most of it on the table.

At fixed `n3` the delivery and stock are within ~2% of each other on every axis (R -1.8%, P -0.7%,
C +1.2%, K +0.2%) -- the delivery's win is concentrated in the deeper/adaptive drafts, which is what
block 001 is for.

## Observations

* **The controller tracks the workload across a real run.**  On reasoning it pins at the floor (mean
  length 2.74, essentially identical to fixed `n3`).  On prose, code and recall it climbs (mean length
  5.10 / 7.02 / 8.95) and wins 13-61%.
* **Lower acceptance, higher throughput.**  The adaptive acceptance is *below* fixed `n3` on every axis
  (e.g. code 0.579 vs 0.917) because it drafts deeper and rejects more -- but it accepts more tokens per
  target forward pass (mean length x acceptance), so it is faster.  Acceptance alone is not the metric.
* **The 256-token measurement inverted the ranking.**  Code at ceiling 12 was -5% at 256 tokens and
  +28% at 3000; prose at ceiling 12 was flat at 256 tokens and +13% at 3000.  At 3000 the old ceiling 7
  loses to fixed `n3` on prose and code (-10% / -5%) while ceiling 12 wins -- the two settings are not
  interchangeable and neither is visible in a short run.

## Text purity (`-n 3000`, no `-lv 4`)

| axis | `plain` | `n3` | `adaptive n12` |
|---|---|---|---|
| reasoning | `98d4e36a79fb` | `98d4e36a79fb` | `98d4e36a79fb` |
| prose | `27f3f7d3f80c` | `27f3f7d3f80c` | `7ec08bc22946` |
| code | `48241ec079f6` | `48241ec079f6` | `a2eceaad5743` |
| recall | `a87c4318b649` | `a87c4318b649` | `a87c4318b649` |

Fixed `n3` is byte-identical to plain on every axis.  Adaptive `n12` is byte-identical on reasoning and
recall, and diverges on prose and code -- the expected above-7 purity trade (a verify wider than 8 rows
switches kernel families), which only shows up once the run is long enough to hit a near-tie.  At
`-n 256` all four happened to match, which is exactly why purity must be checked at the gate length.
Depth 12 is inside the hard 15 bound; stock is **not** pure under the same protocol (its plain decode
disagrees with its own MTP), and the delivery's plain equals stock's MTP text.

## Reproducing

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
for axis in reasoning prose-rdna-boosts code-python recall; do
  case "$axis" in reasoning) REA=on;; *) REA=off;; esac
  for spec in "--spec-type none" \
              "--spec-type draft-mtp --spec-draft-n-max 3" \
              "--spec-type draft-mtp-adaptive --spec-draft-n-max 12"; do
    HIP_VISIBLE_DEVICES=0 build/bin/llama-cli -m "$M" --reasoning $REA $spec -f prompts/$axis.txt \
      -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```
