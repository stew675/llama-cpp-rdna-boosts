# MTP Quick Reference

Flags, depth policy, and copy-paste commands for MTP and Adaptive MTP. For the story and the evidence,
see **[MTP & Adaptive MTP](MTP-and-Adaptive-MTP)**.

> **Drafter:** use the model's **built-in MTP head** (`blk.<last>.nextn.*`, `nextn_predict_layers` in
> the GGUF). llama.cpp uses it automatically when no `-md` is given. **Do not** pass the standalone
> `mtp-*.gguf` — it is an older drafter and gives different (lower) acceptance.

---

## The two modes

```bash
# Static: same draft depth for the whole run (upstream behaviour)
--spec-type draft-mtp --spec-draft-n-max 3

# Adaptive: the depth follows measured acceptance (block 0001)
--spec-type draft-mtp-adaptive --spec-draft-n-max 12
```

---

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--spec-type draft-mtp` | — | Static MTP. |
| `--spec-type draft-mtp-adaptive` | — | Adaptive MTP (credit-bucket depth controller). |
| `--spec-type ...,ngram-mod` | — | Add the ngram speculator (composes with MTP; ngram takes priority). |
| `--spec-draft-n-max N` | `3` | Maximum draft depth. For adaptive mode this is the **ceiling**. Clamped at **15**. |
| `--spec-draft-n-min-adaptive N` | `3` | Adaptive floor. Must be ≥ 1. |
| `--spec-draft-n-start N` | `0` (→ `cap − 3`) | First verify round's depth, clamped to `[floor, cap]`. `0` keeps the default. |
| `--spec-ngram-mod-n-match K` | `24` | ngram-mod minimum match length. Use **45** for the recall combo. |
| `--spec-draft-p-min P` | — | Minimum draft probability; used by some server configs. |
| `-lv 4` | off | Print the `draft acceptance` / `acc per pos` lines. |

Environment equivalents exist for the server (e.g. `LLAMA_ARG_SPEC_DRAFT_N_START`).

### Depth policy

| `--spec-draft-n-max` | Verify width | Greedy guarantee |
|---|---|---|
| **≤ 7** | ≤ 8 | **Byte-identical**: `--spec-type none` == `draft-mtp`. The acceptance comparison runs exactly the decode arithmetic. |
| **8 … 15** | 9 … 16 | **Allowed with a visible notice.** Above 8 rows the FA chooser switches tile → WMMA and the matmuls switch MMVQ/MMVF → MMQ, so a greedy near-tie may flip. Valid, coherent output; a documented trade. |
| **> 15** | > 16 | **Clamped to 15.** The recurrent rollback snapshot bound (`n_max + 1 = K ≤ 16`) — correctness, not purity. |

`LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` keeps a larger value with its own notice. The default depth is **3**,
so nothing changes unless you opt in.

---

## Recommended starting points

| Workload | Command |
|---|---|
| **General / mixed** | `--spec-type draft-mtp-adaptive --spec-draft-n-max 12` |
| **Code-heavy** | same — the controller settles around 9–10 on its own |
| **Reasoning-heavy** | same — the controller parks at the floor |
| **Verbatim recall / long documents** | `--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45 --spec-draft-n-max 9 --spec-draft-n-start 9` |
| **Just want it to work** | `--spec-type draft-mtp-adaptive --spec-draft-n-max 7` (inside the pure band) |

There is no per-run workload switch to set: the controller measures acceptance and moves. Start with
`--spec-draft-n-max 12` and let it decide.

---

## Copy-paste commands

### Adaptive MTP, four-prompt gate

```bash
MODEL=/path/to/model.gguf
for axis in reasoning prose-rdna-boosts code-python recall; do
  case "$axis" in reasoning) REA=on;; *) REA=off;; esac
  build/bin/llama-cli -m "$MODEL" \
    --reasoning "$REA" \
    --spec-type draft-mtp-adaptive --spec-draft-n-max 12 \
    -f prompts/$axis.txt \
    -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
done
```

Acceptance and mean accepted length come from the `-lv 4` line; generation t/s from the eval-time line.

### Compare against static and plain

```bash
for spec in "--spec-type none" \
            "--spec-type draft-mtp --spec-draft-n-max 3" \
            "--spec-type draft-mtp-adaptive --spec-draft-n-max 12"; do
  build/bin/llama-cli -m "$MODEL" $spec -f prompts/code-python.txt \
    -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
done
```

### Pinned-depth oracle (find the optimum for a workload)

Pin the controller with an equal floor and ceiling; zero depth transitions.

```bash
for D in 3 5 7 8 9 10 11 12; do
  build/bin/llama-cli -m "$MODEL" \
    --spec-type draft-mtp-adaptive \
    --spec-draft-n-min-adaptive "$D" --spec-draft-n-max "$D" \
    -f prompts/code-python.txt \
    -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99
done
```

### Purity check (must be byte-identical)

```bash
for spec in "--spec-type none" \
            "--spec-type draft-mtp --spec-draft-n-max 3" \
            "--spec-type draft-mtp --spec-draft-n-max 7"; do
  build/bin/llama-cli -m "$MODEL" $spec -f prompts/code-python.txt \
    -n 64 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 > run.log 2>&1
  scripts/extract-generated.py run.log
done
```

Run **without** `-lv 4` (the verbose log interleaves statistics into the generated text), and use the
extractor helper — a naive slice does not reproduce the hashes.

### Verify-width check (catches wide-batch regressions)

```bash
build/bin/llama-batched-bench -m "$MODEL" \
  -npp 16 -ntg 32 -npl 1,4,8 -ctk q8_0 -ctv q8_0
```

Run interleaved against a stock build at the same fork point; B=4/B=8 must not regress.

---

## Reading a run

With `-lv 4`, look for:

```
draft acceptance = 0.81707, mean len = 3.62
```

- **Acceptance** is the fraction of proposed tokens accepted (often reported per position;
  position 1 is the one that matters most).
- **Mean accepted length** = tokens accepted per verify round. Throughput tracks
  `n / (1 + mean_len)` rounds, so **accepted tokens per round** is the real metric — not acceptance
  alone. Adaptive MTP deliberately runs at *lower* acceptance than a shallow fixed depth, because it
  drafts deeper.

At `TRC` verbosity the adaptive controller reports each depth transition with its bucket state.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| **Acceptance 0.0** | Verify/decode numerics diverged (a kernel or fusion bug), not a tuning issue. Check the build and the `GREEDY-PURITY.md` invariants. |
| **MTP slower than plain** | Over-drafting at a fixed deep depth, or a wide-verify regression. Use adaptive mode instead of a fixed `n_max ≥ 8`, and run the verify-width check. |
| **A `W`-level notice about depths 8–15** | Expected: above 8 verify rows the kernel families switch, so greedy near-ties may flip. Lower `--spec-draft-n-max` to 7 if you need the bit-identical contract. |
| **`--spec-draft-n-max` was clamped** | `> 15` hits the recurrent rollback bound. `LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0` overrides with a notice. |
| **Adaptive outperforms on one axis, loses on another** | Re-run the full four-axis gate *and* the phase-switching prompt (`prompts/code-reasoning-mixed.txt`) at `-n 3000`. A near-ratchet setting can win pure code and lose phase switching. |

---

## See also

- **[MTP & Adaptive MTP](MTP-and-Adaptive-MTP)** — the story, the numerics, and the results.
- [`benchmarks/mtp-adaptive-methodology.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/benchmarks/mtp-adaptive-methodology.md)
- [`GREEDY-PURITY.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/GREEDY-PURITY.md)
- [`prompts/README.md`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/prompts/README.md)
