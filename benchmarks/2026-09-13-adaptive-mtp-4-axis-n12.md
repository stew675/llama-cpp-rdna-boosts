# Adaptive MTP across the four workloads, ceiling 12 (2026-09-13)

Measures the delivery's `draft-mtp-adaptive` mode (block 001) at **`--spec-draft-n-max 12`**, the ceiling
its original PR recommended, now that the old `n_max <= 7` clamp no longer caps it (the 2026-09-13
issue-#30 amendment; see `../WORKLOG.md` 2026-09-13 (latest)).  This supersedes the `adaptive n7`
column of [2026-09-13-adaptive-mtp-4-axis.md](2026-09-13-adaptive-mtp-4-axis.md), whose adaptive ceiling
was an artifact of the clamp.

Workloads are the adaptive-MTP gate's four axes: **R**easoning, **P**rose, **C**ode and verbatim recall
(**K**).  MTP acceptance is a function of *what* the model is generating, so a single-prompt
measurement is not representative and the adaptive controller is supposed to track the workload rather
than assume a fixed depth.

## Environment

* 1x R9700 (gfx1201), ROCm 7.14, one GPU used (`HIP_VISIBLE_DEVICES=0`)
* Build: the 2026-09-13 issue-#30 delivery (canonical tip `c45244c72`, tree `a5683e1b008e`)
* Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes), f16 K/V, `-fa auto -ngl 99`
* Drafter: the MTP head built into the model GGUF (`blk.64.nextn.*`, `nextn_predict_layers = 1`).
  **No `-md`** (a separate draft file is a different drafter and changes the numbers).
* `-n 256 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048`

Prompts (see `../prompts/README.md`):

| axis | file |
|---|---|
| R | `prompts/reasoning.txt` |
| P | `prompts/prose-rdna-boosts.txt` |
| C | `prompts/code-python.txt` |
| K | `prompts/recall.txt` |

## Commands

```sh
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
# plain
build/bin/llama-cli -m "$M" --spec-type none -f prompts/<axis>.txt <flags> -lv 4
# fixed depth 3
build/bin/llama-cli -m "$M" --spec-type draft-mtp --spec-draft-n-max 3 -f prompts/<axis>.txt <flags> -lv 4
# adaptive, ceiling 12
build/bin/llama-cli -m "$M" --spec-type draft-mtp-adaptive --spec-draft-n-max 12 -f prompts/<axis>.txt <flags> -lv 4
```

Acceptance and mean accepted length come from the `-lv 4` `draft acceptance = ... , mean len = ...`
line; generation t/s from the eval time line.  A second full matrix reproduced every cell within ~1%
(the recall adaptive cell read 80.5 and 81.1 t/s on the two runs), so treat the t/s column as +/-1%.

## Results - rdna-boosts at ceiling 12

| axis | plain t/s | `mtp n3` t/s | `mtp n3` acc | `mtp n3` mean len | `adaptive n12` t/s | `adaptive n12` acc | `adaptive n12` mean len |
|---|---|---|---|---|---|---|---|
| reasoning (R) | 29.1 | 58.6 | 0.79204 | 3.36 | 57.8 | 0.79204 | 3.36 |
| prose (P) | 28.5 | 43.8 | 0.54483 | 2.63 | 43.4 | 0.54483 | 2.63 |
| code (C) | 29.0 | 45.9 | 0.55789 | 2.66 | 45.5 | 0.55789 | 2.66 |
| recall (K) | 29.0 | 66.8 | 0.95939 | 3.86 | **81.1** | 0.93363 | **5.80** |

The stock-`n3` comparison and the stock-vs-delivery context are unchanged from the ceiling-7 record
(see it for the table); the delivery is ahead of stock on every axis at `mtp n3`, and the delivery's
`mtp n3` and `plain` cells here match that record to within noise.

## Observations

* **Acceptance is workload-dominated.** 0.934 on verbatim recall, 0.792 on reasoning, 0.558 on code,
  0.545 on prose.  Raw t/s is meaningless without the prompt.
* **The controller tracks the workload.**  On R/P/C it settles at the same effective depth as fixed
  `n3` (identical accepted/generated counts and mean length) and costs about 1% for the decision
  overhead.  On recall, where deeper drafts pay, it climbs: mean accepted length **5.80** vs 3.86 (fixed
  `n3`) and a **+21.4%** gain over fixed `n3` (81.1 vs 66.8 t/s).
* **The ceiling matters on recall, and 12 is where it pays.**  At the old ceiling 7 the same cell read
  71.03 t/s / mean len 5.43 / acc 0.94977; ceiling 12 is **+14.2%** over that, at a slightly lower
  acceptance (0.93363), i.e. the extra depth more than pays for itself on this workload.  This is the
  behaviour the mode was designed for and the reason the old clamp was hiding it.
* **Plain decode is workload-insensitive** (28.5 to 29.1 t/s everywhere), as expected without
  speculation.

## Text purity at ceiling 12

Although the bit-identical `plain` vs `draft-mtp` guarantee is only promised for `n_max <= 7` (a verify
wider than 8 rows switches FA and matmul kernel families), the observed runs at `-n 256` with **no
`-lv 4`** are byte-identical:

| axis | `plain` | `adaptive n12` |
|---|---|---|
| reasoning | `383323542388` | `383323542388` |
| prose | `509a9ebcc8e3` | `509a9ebcc8e3` |
| code | `03cef48f9c2b` | `03cef48f9c2b` |
| recall | `63f30098feea` | `63f30098feea` (`plain == n3 == n7 == adaptive n12`) |

Depth 12 is well inside the **hard** bound of 15 (the recurrent rollback snapshot set; the deterministic
`test-recurrent-state-depth` sweep is clean for `n_rs_seq` 1..15).  The purity above 7 is a consequence
of no greedy near-tie being hit on these axes, not a new guarantee: the CLI keeps a visible notice for
depths 8..15 for that reason (`../GREEDY-PURITY.md` §11/§32).

## Reproducing

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
for axis in reasoning prose-rdna-boosts code-python recall; do
  for spec in "--spec-type none" \
              "--spec-type draft-mtp --spec-draft-n-max 3" \
              "--spec-type draft-mtp-adaptive --spec-draft-n-max 12"; do
    HIP_VISIBLE_DEVICES=0 build/bin/llama-cli -m "$M" $spec -f prompts/$axis.txt \
      -n 256 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```
