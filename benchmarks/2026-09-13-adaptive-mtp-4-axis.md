# Adaptive MTP across the four workloads (2026-09-13)

Measures the delivery's MTP draft depth variants across the four workloads the adaptive-MTP gate uses:
**R**easoning, **P**rose, **C**ode and verbatim recall (**K**). It exists because MTP acceptance is a
function of *what the model is generating*, so a single-prompt measurement is not representative, and
because the adaptive controller (block 001) is supposed to track the workload rather than assume a
fixed depth.

## Environment

* 1x R9700 (gfx1201), ROCm 7.14 (`HIP 7.14.60850`), one GPU used (`HIP_VISIBLE_DEVICES=0`)
* Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes), f16 K/V, `-fa auto -ngl 99`
* Drafter: the MTP head built into the model GGUF (`blk.64.nextn.*`, `nextn_predict_layers = 1`).
  **No `-md`** (a separate draft file is a different drafter and changes the numbers).
* `-n 256 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048`

Prompts (see `../prompts/README.md` for sizes and hashes):

| axis | file |
|---|---|
| R | `prompts/reasoning.txt` |
| P | `prompts/prose-rdna-boosts.txt` |
| C | `prompts/code-python.txt` |
| K | `prompts/recall.txt` |

## Commands

```sh
# plain
build/bin/llama-cli -m "$MODEL" --spec-type none -f prompts/<axis>.txt <flags> -lv 4
# fixed depth 3
build/bin/llama-cli -m "$MODEL" --spec-type draft-mtp --spec-draft-n-max 3 -f prompts/<axis>.txt <flags> -lv 4
# adaptive, ceiling 7
build/bin/llama-cli -m "$MODEL" --spec-type draft-mtp-adaptive --spec-draft-n-max 7 -f prompts/<axis>.txt <flags> -lv 4
```

Acceptance and mean accepted length come from the `-lv 4` `draft acceptance = ... , mean len = ...`
line; generation t/s from the `eval time` line.

## Results - rdna-boosts (tip `f27dc6d80`, tree `bbbe005e`)

| axis | plain t/s | `mtp n3` t/s | `mtp n3` acc | `mtp n3` mean len | `adaptive n7` t/s | `adaptive n7` acc | `adaptive n7` mean len |
|---|---|---|---|---|---|---|---|
| reasoning (R) | 29.10 | 58.57 | 0.79204 | 3.36 | 57.75 | 0.79204 | 3.36 |
| prose (P) | 28.52 | 43.75 | 0.54483 | 2.63 | 43.29 | 0.54483 | 2.63 |
| code (C) | 28.94 | 45.82 | 0.55789 | 2.66 | 45.35 | 0.55789 | 2.66 |
| recall (K) | 28.97 | 66.65 | 0.95939 | 3.86 | **71.03** | 0.94977 | **5.43** |

## Results - stock `790cf51aa`

| axis | plain t/s | `mtp n3` t/s | `mtp n3` acc |
|---|---|---|---|
| reasoning (R) | 28.45 | 56.24 | 0.76623 |
| prose (P) | 27.87 | 39.29 | 0.45652 |
| code (C) | 28.31 | 43.52 | 0.51839 |
| recall (K) | 28.35 | 65.82 | 0.95455 |

(Stock has no `draft-mtp-adaptive`; that mode is block 001.)

## Observations

* **Acceptance is workload-dominated.** 0.955 to 0.959 on verbatim recall, 0.766 to 0.792 on reasoning,
  0.518 to 0.558 on code, 0.457 to 0.545 on prose. Raw t/s is therefore meaningless without the prompt.
* **The adaptive controller tracks the workload.** On the three lower-acceptance axes it settles at the
  same effective depth as fixed `n3` (identical accepted/generated counts and mean length), and costs
  about 1% for the decision overhead. On recall, where deeper drafts pay, it drafts deeper (mean
  accepted length 5.43 vs 3.86) and gains **+6.6%** over fixed `n3` (71.03 vs 66.65 t/s).
* **The delivery is ahead of stock on every axis** at `mtp n3`: reasoning +4.1%, prose +11.4%,
  code +5.3%, recall +1.3%, and the delivery's acceptance is higher on all four.
* **Plain decode is workload-insensitive** (27.9 to 29.1 t/s everywhere), as expected: without
  speculation there is no acceptance term.

## The reporter's configuration (`n_max 8 --spec-draft-p-min 0.55`)

Issue #30's measurement used `--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.55`.
On the prose prompt, `-n 128`, same environment:

| build | command | t/s | acceptance |
|---|---|---|---|
| stock `790cf51aa` | `n_max 7 --spec-draft-p-min 0.55` | 43.62 | 0.75000 |
| stock `790cf51aa` | `n_max 8 --spec-draft-p-min 0.55` | 45.71 | 0.77143 |
| rdna-boosts | `n_max 7 --spec-draft-p-min 0.55` | 44.62 | 0.76699 |
| rdna-boosts | `n_max 8 --spec-draft-p-min 0.55` (clamps to 7) | 44.53 | 0.76699 |
| rdna-boosts | `n_max 8 --spec-draft-p-min 0.55`, `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` | 47.28 | 0.71818 |

`--spec-draft-p-min` raises acceptance a lot on both arms (prose `n3` without it: 0.49020 stock /
0.61654 delivery; with it: ~0.77 on both), so it must be part of any comparison that uses it. The
`n_max` clamp is load-bearing: the delivery clamps to 7, so the reporter's exact command measures depth
7 on the patched arm; disable the clamp (`LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0`) to measure a true depth 8.

## Reproducing

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
for axis in reasoning prose-rdna-boosts code-python recall; do
  for spec in "--spec-type none" \
              "--spec-type draft-mtp --spec-draft-n-max 3" \
              "--spec-type draft-mtp-adaptive --spec-draft-n-max 7"; do
    HIP_VISIBLE_DEVICES=0 build/bin/llama-cli -m "$M" $spec -f prompts/$axis.txt \
      -n 256 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```
