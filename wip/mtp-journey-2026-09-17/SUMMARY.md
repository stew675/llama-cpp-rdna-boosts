# Adaptive MTP — PR #27210 to today: final report (2026-09-17)

This is the consolidated result of the day's investigation.  It started as an effort to document the
journey from the original controller PR (#27210, the **table**) through to the delivery's **credit
bucket**, and grew into a full multi-cell re-tune and a study of the **`ngram-mod` + adaptive-MTP
combo**.

**Bottom line:** the delivery's **credit-bucket** controller is the best controller design tested and
its **current constants are the best general default** — no block-01 change is recommended.  The
**`ngram-mod` + adaptive-MTP combo** is the best *configuration* for recall/mixed workloads and is
recommended as a per-context opt-in.  The only concrete defect found was a **stale documentation
pointer** that made the delivery look worse than it is.

---

## 1. Method — the anti-overfit corpus

The delivery had been tuned on **one prompt per axis** on **one cell** (2-GPU Q8_0).  That overfits in
two directions at once, and the whole investigation confirmed it.  The new gate is
`wip/mtp-journey-2026-09-17/corpus/`:

* **4 prompts per axis** (reasoning `r1`-`r4`, prose `p1`-`p4`, code `c1`-`c4`, recall `k1`-`k4`),
  hash-frozen in `corpus/MANIFEST.md`;
* **3 mixed phase-switch prompts** (`phase-switch.txt`, `ps2-phase.txt`, `ps3-phase.txt`);
* run on **4 cells**: dense Q4_K_XL 1 GPU, MoE 35B-A3B 1 GPU, Q8_0 2-GPU tensor, Q8_0 3-GPU tensor,
  plus 2-GPU `-sm layer`.

Judgement is always on the **per-axis geometric mean**, never a single prompt.

## 2. The controller: bucket vs every alternative

Measured on the same build with the old controllers re-added behind env switches (`GGML_ADAPTIVE_TABLE`,
`GGML_ADAPTIVE_MEAN`, `GGML_ADAPTIVE_TARGET`), so kernels/numerics are held constant.

**Bucket vs the PR #27210 table** (ratio, >1 = bucket wins):

| axis | dense Q4_K_XL 1 GPU | MoE 1 GPU |
|---|---:|---:|
| reasoning | 0.997 | 1.090 |
| prose | 0.949 | 0.943 |
| code | 0.930 | 1.089 |
| recall | 1.192 | 1.014 |
| **overall** | **1.012** | **1.032** |

On the Q8_0 2-GPU cell the bucket beats the table by **+14 % (code)** and **+22 % (prose)**.

**Bucket vs the alternatives** (full 16-prompt dense corpus, ratio vs table where available):

| controller | dense overall | worst axis | notes |
|---|---:|---:|---|
| **base bucket (delivery)** | 1.012 | 0.930 | the default |
| retuned bucket `ref_d` | 1.061 | 0.992 | wins dense, loses Q8_0 prose (§3) |
| sliding-mean (best) | 0.980 | 0.851 | 5-prompt set |
| target-acceptance-rate (best) | 1.048 | 0.945 | full 16-prompt set |

**The alternative designs fail for a structural reason, not a tuning one.**  Both are single-knob
controllers, and a single content-independent statistic cannot serve reasoning (which wants the floor)
and code (which wants depth) at once.  For the target-rate rule the inversion is explicit: reasoning's
pooled acceptance at its **floor** (0.587) is *higher* than code's at its **optimum** (0.536).  The
table sidesteps this because its per-depth thresholds approximate the *marginal* full-accept
probability; a proportional/marginal controller remains the one untried design.

## 3. Why the default is unchanged — no retune is Pareto-safe

The best retune found was `ref_d` (`GGML_MTP_DROP_FLOOR 250`, `DROP_SLOPE 40`, `CLIMB_BASE 10`,
`CLIMB_SLOPE 3`).  It is a **uniform** win on dense Q4_K_XL (all 16 prompts improve, +4.8 % overall)
and MoE-neutral — but it loses elsewhere, and the discriminator is the **weight quantization**, not the
GPU split:

| prose, `ref_d`/base | Q8_0 | Q4_K_XL |
|---|---:|---:|
| 1 GPU | **0.960** | 1.079 |
| 2 GPU tensor | 0.951 | — |
| 2 GPU layer (no AllReduce) | 0.950 | — |
| 3 GPU tensor | 0.914 | — |

Because `-sm layer` shows it too, it is **not** the AllReduce numerics.  A GPU-mode split was proposed
and tested: it does **not** protect Q8_0 (which is the server's quant), so it would regress the
maintainer's actual cell.  Every conservative variant (`climbonly`, `drop120`, `climb12`, `hi2`, `hi3`)
trades the same way.  **No tested constant set is a Pareto improvement**, so the base constants stay.

## 4. The configuration: `ngram-mod` + adaptive MTP

`ngram-mod` supplies the long verbatim-recall drafts that MTP can never match, freeing the MTP
controller from chasing the recall plateau.  Recommended per-context config:

```
--spec-type draft-mtp-adaptive,ngram-mod
--spec-ngram-mod-n-match 45
--spec-draft-n-max 9 --spec-draft-n-start 9
```

Measured on the 4×4 corpus, Q8_0 2-GPU, vs the plain adaptive default (`mtp12`):

| axis | combo / mtp12 |
|---|---:|
| reasoning | 0.981 |
| prose | 1.005 |
| code | 1.008 |
| **recall** | **1.675** |
| **overall** | **1.136** |

Per-prompt recall is `+70 / +78 / +71 / +52 %`.  On dense Q4_K_XL 1 GPU the same combo
(`c9 s9 nm45`) is code/prose-neutral (`0.987`-`1.014`) with the same recall win.  The ngram is provably
**silent off recall** (acceptance and mean len identical to MTP-alone on code and prose; recall mean len
jumps `~10 -> ~38`).

**Caveat found by the corpus:** the combo costs about **2 % on reasoning** (`r1` alone, the original
gate prompt, was `+1.0 %`; `r2`-`r4` are `-2.5` to `-3.3 %`).  That is a genuine cost of the
`n-max 9`/`n-start 9` change plus the occasional ngram hit on reasoning, and it should be documented
with the config.

## 5. Cap selection — corrected

The optimal MTP cap is **cell-dependent, and in the opposite direction to a cost intuition**:

| cell | cap 6 | cap 9 | cap 12 |
|---|---:|---:|---:|
| **dense Q4_K_XL 1 GPU** (R/P/C, vs cap 12) | **0.888** | **1.009** | 1.000 |
| **Q8_0 2-GPU tensor** (R/P/C, vs cap 9) | **1.008** | 1.000 | ~0.998 |

* **Single card → higher cap** (9, not 6): cap 6 costs **11-14 %** on code (`c1` alone `-26 %`).
* **Multiple cards → lower cap** (6-7 mildly beats 9, which beats 12 on prose/code).

**Mechanism:** a lower cap clips content whose natural accepted length exceeds it (`c1` accepts
`mean len 6.7`; cap 6 clips it to `5.63`), forcing more verify rounds.  On **1 GPU** the verify cost is
nearly flat in batch size (the weight read dominates), so those extra rounds are pure loss; on
**multi-GPU** the per-round cost falls with the batch, so shallow caps stay competitive.  This is a
**cost-structure** effect, not a drafter-quality effect (§6).

## 6. `n_match` selection

* **Use a value `>= 40`.**  `nm24` fires on incidental code repeats and drafts ~64 tokens accepting
  ~5 (`draft acceptance 0.078` on code), which loses code throughput.
* **Make it an integer multiple of the cap.**  Direct evidence: `nm42` at cap 12 (42 is not a multiple
  of 12) *crashes* the prose `p3` acceptance `0.623 -> 0.522` and costs ~5 % throughput, while `nm45`
  at cap 9 (`9×5`) and `nm42` at cap 6 (`6×7`) leave acceptance at or above the no-ngram value.
  The known-good pairings are `cap 9 / nm45`, `cap 8 / nm48`, `cap 6 / nm42`.

## 7. Drafter acceptance vs weight quantization

The built-in MTP head sees slightly drifted inputs on a Q4_K_XL model.  At a matched config
(`c9 s9 nm45`) the acceptance is **~0.9 pp lower** on Q4_K_XL than Q8_0 (lower in 6 of 8 prompts,
range `-2.6` to `+2.2` pp) — small but real, and a plausible partial cause of the pre-existing ~7 %
dense code gap.  It does **not** explain the cap effect: on `c1` the acceptance gap is `-0.5 pp` while
the cap-6 loss is `-26 %`, and `c4` accepts *more* on Q4_K_XL with no cap loss.  The MTP head in a
UD-Q4_K_XL GGUF is also kept at higher precision than the bulk weights, so it is not a BF16 head
reading Q4 weights.

## 8. Recommendations

1. **Do not change block 01.**  The base credit bucket is the best multi-cell default; the alternatives
   are dominated and the retune is quant-specific.  This is a *measured* conclusion, not a default by
   inertia.
2. **Ship the combo as a documented opt-in** (not a default — it changes the draft strategy):
   `--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45 --spec-draft-n-max 9
   --spec-draft-n-start 9`, with the ~2 % reasoning cost noted.
3. **Cap guidance:** on a single card keep the MTP cap at ~9; on multi-GPU 6-7 is mildly better.
4. **Fix the stale record pointer** — the one real defect: `benchmarks/README.md:23` and
   `benchmarks/mtp-adaptive-methodology.md:202` still call the 2026-09-13 four-axis record current, but
   it was measured with the pre-tuning **table**.  Point them at the 2026-09-15 tuning record.
5. **Keep the 3×prompt corpus** as the controller gate; add the multi-cell requirement to
   `benchmarks/mtp-adaptive-methodology.md`.

## 9. Open items

* The **GitHub wiki** publish waits for a later session (the wiki source is committed, but the site needs
  its first page created in the web UI; the controller wording there should match §8).
* The **marginal / full-accept-rate controller** is the one credit-worthy untried design (the
  statistically correct signal; the table approximates it).
* The stale-pointer fix is a delivery-doc change and needs the maintainer's go-ahead.

## 10. Artifacts

`wip/mtp-journey-2026-09-17/` (WIP, not part of the delivery):

| path | what |
|---|---|
| `README.md` | the full chronological narrative (findings 1-14) |
| `SUMMARY.md` | this report |
| `HANDOVER.md` | the session handoff / environment |
| `corpus/` | the 4×4 + 3 phase prompts, `MANIFEST.md` |
| `raw/*.tsv` | every result table (`ground*`, `target-valid*`, `ngram-q8t2`, `combo-q8t2`, `dense-combo`, `canonical`, …) |
| `raw/*.py` | the harnesses (`ground*`, `matrix`, `pinned`, `ab`, `prompt_ab*`, `tune1..tune18`, `ngram`, `combo`, `dense-combo`, `dense-iso`, `acc-quant`) |
| `raw/logs.tar.zst` | all 786 run logs (compressed from 33 MB to 2.8 MB); `logs-index.txt` lists them |
