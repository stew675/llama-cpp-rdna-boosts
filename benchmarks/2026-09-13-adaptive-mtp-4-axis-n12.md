# Adaptive MTP across the four workloads, ceiling 12 (2026-09-13)

Measures the delivery's `draft-mtp-adaptive` mode (block 001) at **`--spec-draft-n-max 12`**, the ceiling
its original PR recommended, now that the old `n_max <= 7` clamp no longer caps it (the 2026-09-13
issue-#30 amendment; see `../WORKLOG.md` 2026-09-13 (latest)).

> **Protocol correction (same day).**  The first cut of this record (and the earlier
> [2026-09-13-adaptive-mtp-4-axis.md](2026-09-13-adaptive-mtp-4-axis.md)) ran **all four axes with the
> model's default reasoning mode**.  Qwen3.8 emits a thinking trace for the prose and code prompts, so
> those two columns were measuring *thinking*, not prose/code output: at `-n 256` the code run never
> reached any Python.  The table below sets **`--reasoning off` for P/C/K** (the intended content) and
> **`--reasoning on` for R** (the reasoning axis).  It also changes R slightly, because the flag is part
> of the chat template.  Numbers in both older records are **not** comparable to these.

Workloads are the adaptive-MTP gate's four axes: **R**easoning, **P**rose, **C**ode and verbatim recall
(**K**).  Run all four: the adaptive controller's behaviour (and therefore the throughput) is a function
of the workload's acceptance rate.

## Environment

* 1x R9700 (gfx1201), ROCm 7.14, one GPU used (`HIP_VISIBLE_DEVICES=0`)
* Build: the 2026-09-13 issue-#30 delivery (canonical tip `c45244c72`, tree `a5683e1b008e`)
* Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes), f16 K/V, `-fa auto -ngl 99`
* Drafter: the MTP head built into the model GGUF (`blk.64.nextn.*`, `nextn_predict_layers = 1`).
  **No `-md`**.
* `-n 256 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048 -ub 2048`
* Two repetitions per cell; the table shows the mean (t/s within ~1%, acceptance discrete).

Prompts: `prompts/reasoning.txt` (R), `prose-rdna-boosts.txt` (P), `code-python.txt` (C),
`recall.txt` (K).  See `../prompts/README.md`.

## Commands

```sh
M=Qwen3.8-27B-UD-Q4_K_XL.gguf
# R keeps thinking; P/C/K must turn it off so the model generates the intended content
for axis in reasoning prose-rdna-boosts code-python recall; do
  case "$axis" in reasoning) REA=on;; *) REA=off;; esac
  for spec in "--spec-type none" \
              "--spec-type draft-mtp --spec-draft-n-max 3" \
              "--spec-type draft-mtp-adaptive --spec-draft-n-max 12"; do
    build/bin/llama-cli -m "$M" --reasoning $REA $spec -f prompts/$axis.txt \
      -n 256 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```

Acceptance and mean accepted length come from the `-lv 4` `draft acceptance = ... , mean len = ...`
line; generation t/s from the eval time line.

## Results - rdna-boosts at ceiling 12

| axis | plain t/s | `mtp n3` t/s | `mtp n3` acc | `mtp n3` mean len | `adaptive n12` t/s | `adaptive n12` acc | `adaptive n12` mean len |
|---|---|---|---|---|---|---|---|
| reasoning (R) | 29.1 | 58.0 | 0.79204 | 3.36 | 57.2 | 0.79204 | 3.36 |
| prose (P) | 28.5 | 52.8 | 0.72803 | 3.17 | 52.3 | 0.72803 | 3.17 |
| code (C) | 29.0 | **63.1** | **0.89372** | 3.68 | 59.7 | 0.72059 | 4.32 |
| recall (K) | 29.0 | 67.8 | 0.98446 | 3.92 | **91.6** | 0.98649 | **7.08** |

For reference, the same matrix at the old ceiling 7: recall 74.8 t/s / 0.98618 / 6.22; R/P are
identical to the n12 cells above (the controller stays at the floor, so the ceiling never matters);
code 57.9 t/s / 0.73684 / 4.32.

## Observations

* **Code is the predictable workload it was meant to be.**  With reasoning off, `mtp n3` acceptance is
  0.894 (fixed-depth) and the output is real Python.  The earlier 0.558 code acceptance was a thinking
  trace, not code.
* **The adaptive controller tracks R/P correctly and buys a lot on K.**  On R/P it settles at the
  floor (identical accepted/generated counts and mean length to fixed `n3`, ~1% decision overhead).
  On verbatim recall it climbs hard: mean accepted length **7.08** vs 3.92, and ceiling 12 gives
  **+35%** over fixed `n3` (91.6 vs 67.8 t/s) and **+22%** over the old ceiling 7 (74.8 t/s).
* **The controller over-drafts on code.**  Code acceptance at `n3` (0.894) is high enough that the
  controller climbs to mean length ~4.3, but the deeper verify accepts less (0.721) and the net is
  **-5.4%** vs fixed `n3` (59.7 vs 63.1 t/s).  Raising the ceiling from 7 to 12 recovers a little
  (57.9 -> 59.7) but does not reach fixed `n3`.  This is a block-001 controller-tuning observation, not
  a correctness issue, and it is why the gate runs all four axes rather than one.
* **Plain decode is workload-insensitive** (28.5 to 29.1 t/s), as expected without speculation.

## Text purity at ceiling 12

At `-n 256` with **no `-lv 4`**, the delivery is byte-identical across plain, fixed `n3`, fixed `n7`
and adaptive `n12` on **every axis**:

| axis | `plain` | `adaptive n12` |
|---|---|---|
| reasoning | `383323542388` | `383323542388` |
| prose | `ab94eb7db4d4` | `ab94eb7db4d4` |
| code | `355ce76d9c02` | `355ce76d9c02` |
| recall | `6562618b567c` | `6562618b567c` |

The bit-identical guarantee is only *promised* for `n_max <= 7` (a verify wider than 8 rows switches FA
and matmul kernel families), but no near-tie was hit on these runs, so depth 12 is pure here as well.
Depth 12 is well inside the hard 15 bound (the recurrent rollback snapshot set; the deterministic
`test-recurrent-state-depth` sweep is clean for `n_rs_seq` 1..15) — see `../GREEDY-PURITY.md` §11/§32.

Stock is **not** width-pure under the same protocol (prose, `-n 64`, reasoning off):
`stock plain 33ae8d598e7e` vs `stock n3 == stock n7 dc2b1cfd159f`; the delivery's plain/n3/n7 all give
`dc2b1cfd159f`.  The delivery's plain equals stock's MTP text, i.e. the delivery fixed the stock
plain-decode divergence.

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
      -n 256 --seed 42 --temp 0 --single-turn --no-display-prompt \
      -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
  done
done
```
