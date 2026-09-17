# 2026-09-17 — `ngram-mod` + `draft-mtp-adaptive`: ordering, tuning, and the acceptance-feed question

**Question (issue-style).** Can `ngram-mod` supply the long-context recall win without costing the
`draft-mtp-adaptive` throughput on the other axes — and should the adaptive controller be fed another
speculator's acceptance (the old `bucketed-adaptive-mtp` experiment)?

**Answers.**

1. **Ordering is irrelevant:** `--spec-type a,b` and `--spec-type b,a` behave identically.  The code
   builds its implementation list from a **fixed priority list** in which the ngram family precedes the
   draft family, so `ngram-mod` always has precedence.  The docs state the rule: *"If a draft model is
   combined with a draftless decoding the draftless decoding has higher precedence."*
2. **The combo wins with the right tuning** — `--spec-ngram-mod-n-match 45` and MTP `n-max 9 / n-start 9`
   gives **R +1.2 %, P flat, C +0.6 %, K +72 %** against the MTP-only default, with no harm on any axis.
3. **Feeding ngram-mod's acceptances into the adaptive controller is a no-op** with the re-tuned
   controller (all deltas within noise).  The delivery's `common/speculative-adaptive.h` header comment
   still documents that feed, but the code deliberately does not do it — the comment is stale.

## Environment and commands

* 2x R9700 (`gfx1201`), ROCm 7.14, GPUs **1,2** (`-sm tensor -ts 1/1`), f16 K/V unless noted.
* Model `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf`; drafter = built-in MTP head, no `-md`.
* Delivery build `6b1e9ffd1` (release `v16-ebbb18522-r1`); `-n 3000 --seed 42 --temp 0 --single-turn
  --no-display-prompt -c 32768 -b 2048 -ub 2048 -fa auto -ngl 99 -lv 4`; `--reasoning on` for R and
  `off` for P/C/K.  Prompts: `reasoning.txt`, `prose-rdna-boosts.txt`, `code-python.txt`, `recall.txt`
  (hashes in `prompts/README.md`).
* Config shorthand: `cN sM nmK` = `--spec-draft-n-max N --spec-draft-n-start M --spec-ngram-mod-n-match K`
  on top of `--spec-type draft-mtp-adaptive,ngram-mod`.

## 1. Ordering is irrelevant (verified)

```
--spec-type draft-mtp-adaptive,ngram-mod   -> acceptance 0.52740, mean len 5.53
--spec-type ngram-mod,draft-mtp-adaptive   -> acceptance 0.52740, mean len 5.53
```

`common_speculative_init` (`common/speculative.cpp`) turns `params.types` into a **bitmask** and then
walks a hard-coded priority list (`NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE,
DRAFT_SIMPLE, EAGLE3, DRAFT_MTP, DRAFT_MTP_ADAPTIVE, ...`); the argument order only selects members.
`common_speculative_draft` then tries each impl in that order and stops at the first that produces a
draft, so `ngram-mod` always drafts before MTP when both are enabled.

## 2. The four-axis grid (f16 KV)

MTP-only baseline vs the combo.  `adaptive cap12` = the delivery's current recommendation
(`--spec-draft-n-max 12`, ngram-mod also present where noted).

| config | R | P | C | K |
|---|---:|---:|---:|---:|
| `mtp12` (MTP-only) | 60.5 | 80.5 | 94.5 | 136.0 |
| `mtp12` + ngram-mod (`nm24`) | 59.0 | 79.0 | 90.1 | 214.2 |
| `mtp12` + ngram-mod (`nm48`) | — | — | 93.9 | 225.1 |
| `c8 s8 nm48` | 60.0 | 80.2 | 96.1 | 229.4 |
| `c9 s9 nm48` | 61.3 | 80.5 | 95.9 | **207.6** |
| `c10 s9 nm48` | — | — | 95.4 | 224.3 |
| `c8 s8 nm45` | 60.1 | 80.0 | 95.7 | 237.9 |
| **`c9 s9 nm45`** | **61.2** | **80.3** | 95.1 | **234.3** |

`c9 s9 nm45` = `--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45
--spec-draft-n-max 9 --spec-draft-n-start 9`.  Vs `mtp12`: R **+1.2 %**, P −0.2 % (noise), C **+0.6 %**,
K **+72 %**.

Two findings worth naming:

* **The naive combo loses code.**  With `n_match 24` the ngram-mod hashes hit incidental short repeats
  in code and fire ~64-token drafts that accept ~5 tokens (`mean len` is `1 + accepted/verify_round`,
  and the `draft acceptance` for ngram-mod alone on code is 0.078).  `n_match 45` makes a hash hit
  require a much longer context match, so it reliably fires only on verbatim recall; on code it
  produces **zero** accepted drafts (the combo's code acceptance/mean-len equal MTP-alone's exactly).
* **`n_match 48` + `n-start 9` had a recall cliff (207.6).**  It is the **start**, not the cap: with
  `n_match 48`, `start 8` reads 227-229 t/s while `start 9/10` reads 207-209, reproducibly, and the
  generated-token counts differ only by 1.4 % (504 vs 511) — so it is a per-round effect, not a token
  count.  `n_match 45` produces longer, higher-acceptance recall drafts (`mean len` 38.3 vs 35.6,
  acceptance 0.978 vs 0.962) and is insensitive to the start.  Whatever the underlying interaction, the
  practical rule is: **use `n_match 45` if you intend to raise the MTP start above 8.**

## 3. The acceptance feed (`bucketed-adaptive-mtp`) is a no-op now

The old branch at `/home/stew675/stew675/llama-master` (`bucketed-adaptive-mtp`, commit `1207c1c24`)
fed another speculator's accepted count into the controller *when it met the current depth*:

```cpp
if (adaptive && (!is_other || n_accepted >= adaptive_ctrl[seq_id].n_cur)) { ... update ... }
```

The delivery's block 01 replaced that with `if (!is_other) adaptive_feedback(...)` — the feed is **not**
present.  Re-adding the gated feed and A/B-ing the two builds (identical otherwise) on the delivery's
re-tuned controller gave, on every case tested, results within noise / identical acceptance:

| case | no feed | gated feed |
|---|---:|---:|
| `mtp12`+ng code | 89.9 | 89.8 |
| `c8 s8 nm48` code | 95.0 | 94.8 |
| `mtp12`+ng recall | 214.7 | 214.6 |
| `c8 s8 nm48` recall | 229.5 | 231.0 |
| `mtp12`+ng recall, `n-start 3` | 211.4 | 210.4 |
| `mtp12`+ng `code-reasoning-mixed`, `n-start 3` | 63.6 | 63.5 |

Why: the old branch started at the **floor** and relied on the feed to climb; the delivery's re-tuned
controller starts at `cap − 3`, so the climb dependency is gone.  And even where the feed would climb
(strong ngram recall wins raise the depth), the depth is then wrong for the next reasoning step and has
to drop again — the climb and the drop cancel, which is exactly why the phase-switching probe is flat.

**Follow-up for the maintainer:** `common/speculative-adaptive.h`'s comment block still describes the
feed ("a strong run by another speculator ... climbs the depth one step per round while it lasts", and
"callers pass its accepted count"), but the code skips `is_other`.  The comment is stale and should be
corrected (a block-01 comment-only amendment) or the feed re-added deliberately — not left contradicting
the code.

## 4. bf16 KV does not raise acceptance on this cell

Same grid with `-ctk bf16 -ctv bf16` (the delivery's `GGML_CUDA_FA_KV_NATIVE` default stages bf16 to f16
for the native-capable sub-F16 quants; `=1` forces the native bf16 path — both were measured):

| axis | f16 `mtp12` | bf16 `mtp12` | bf16 `c9s9 nm45` |
|---|---:|---:|---:|
| reasoning | 60.5 (0.554) | 59.6 (0.547) | 59.1 (0.537) |
| prose | 80.5 (0.627) | 77.3 (0.617) | 77.2 (0.617) |
| code | 94.5 (0.661) | 93.0 (0.662) | 94.3 (0.668) |
| recall | 136.0 (0.960) | 135.0 (0.960) | **233.9** (0.968) |

Acceptance is the same (slightly lower on R) and bf16 costs ~2-4 % on R/P (the staging/decode cost);
native bf16 and staged bf16 produced identical acceptance (`0.53733/0.61736/0.66849/0.96813`) and
throughput within noise.  The combo's recall win is unchanged.  So on this model the "bf16 boosts
acceptance" effect does not appear; the recommendation stands with either KV type.

## Recommendation

Enable the combo per context (it is not a default — it changes the draft strategy):

```
--spec-type draft-mtp-adaptive,ngram-mod
--spec-ngram-mod-n-match 45
--spec-draft-n-max 9 --spec-draft-n-start 9
```

On the reference cell this is R 61.2 / P 80.3 / C 95.1 / K 234.3 t/s, i.e. the MTP-only default's
R/P/C throughput with **+72 %** on verbatim recall.  `c8 s8 nm45` is the alternative if the workload is
more code/recall-heavy (C 95.7, K 237.9, R 60.1).  Both are cell-specific measurements; re-run the
four-axis gate before adopting them on a different model or shape.
