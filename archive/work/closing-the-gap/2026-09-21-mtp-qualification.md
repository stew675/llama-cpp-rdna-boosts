# MTP qualification — ours (adaptive) vs the other solution's (fixed), qwen4exp IQ4_NL (2026-09-21)

**Status:** preliminary, one session, single box. Enough to answer "is our MTP behind its?", not a
tuned four-axis table. Runs the 2026-09-21 `archive/work/closing-the-gap` investigation.

## Environment

| | |
|---|---|
| box | `halo` — Strix Halo, Radeon 8060S (gfx1151, RDNA3_5), ROCm 7.14 (`/opt/rocm-7.14-gfx1151`) |
| ours | `~/llama.cpp` branch `mmb-beta` = r12 + 12 `beta/mmb-general` patches, tree `bca69f23dd…`, built 2026-09-21 (`90f081550`) |
| the other solution | `~/pwilkin-llama-cpp` @ `b0f31f587`, **rebuilt** 2026-09-21 (the `git pull` had not been rebuilt; the old binary was `f5daaa3cf`) |
| model | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (93 GiB, qwen4exp) |
| MTP head | `/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` — **see the trap below** |
| flags | `-ngl 99 -fa auto -ctk f16 -ctv f16 -c 32768 -b 2048 -ub 2048 -n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt`, `--reasoning off` (all three axes are content axes) |
| prompt | `prompts/{code-python,prose-rdna-boosts,recall}.txt` |

## Trap: the IQ4_NL model ships a **shared** MTP head our delivery cannot use

the other solution's IQ4_NL directory ships `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, which declares
`qwen4exp.nextn_shared_target_tensors = true`. On our build every draft step past the first fails:

```
init: the tokens of sequence 0 ... last position ... X = 544, starting position Y = 544
       for M-RoPE, it is required that the position satisfies: X < Y
E spec  draft: llama_decode[1] returned -1        (1601x over a -n 3000 run)
```

so `acc per pos = (0.82, 0.000, 0.000)` — the drafter is effectively one token wide and the adaptive
controller is inert. The other solution's build runs the same head clean (`0.970, 0.928, 0.878`), so this is **our
missing support for `nextn_shared_target_tensors`**, not a bad checkpoint. Confirmed by A/B on the same
target: the non-shared `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` gives **0 draft errors** on our build and
`acc per pos = (1.000, 0.882, 0.765)`; `--fit false` changes nothing in either direction.

All numbers below therefore use the **Q4_K_M** sidecar, which both trees run correctly. This is the one
MTP-side finding that is a genuine gap; it belongs with correctness, not tuning.

## Results — `-n 3000`, t/s (single run each)

| axis | ours **plain** | its **plain** | ours `draft-mtp n3` | ours `adaptive 9` | ours `adaptive 12` | its `draft-mtp` |
|---|---:|---:|---:|---:|---:|---:|
| code | **32.4** | 31.1 | 61.4 | 59.1 | 58.4 | 59.5 |
| prose | **31.8** | 29.9 | 56.5 | 51.9 | 49.8 | 53.6 |
| recall | **32.4** | 31.9 | 66.4 | 75.7 | **77.7** | 64.3 |

Also measured: ours `adaptive 7` on code = **63.1** t/s (the best code cell; `--spec-draft-n-max 9` is the
Strix Halo default the maintainer names, 7 was the extra point).

**MTP speedup over each build's own plain decode** (the metric that removes the base-decode difference):

| axis | ours `n3` | ours `adaptive 9` | ours `adaptive 12` | its fixed |
|---|---:|---:|---:|---:|
| code | 1.90x | 1.82x | 1.80x | 1.91x |
| prose | 1.78x | 1.63x | 1.57x | 1.79x |
| recall | 2.05x | 2.34x | **2.40x** | 2.02x |

## Conclusions

1. **Our plain (non-MTP) decode is ahead of its** on all three axes (+4 % code, +6 % prose, +2 % recall).
   Any absolute MTP comparison flatters its stack by that amount; the speedup table is the fair one.
2. **At fixed depth the MTP speedup is at parity** (ours `n3` 1.90/1.78/2.05 vs its 1.91/1.79/2.02).
   Its tree has no adaptive controller (`common/speculative-adaptive.h` is absent; fixed `n_max` plus
   upstream's within-round `p_min`/`n_min` stop only), so it did **not** adapt ours, and it is not ahead.
3. **The adaptive controller is mixed on qwen4exp** — the one divergence from the 27B dense record,
   where it won +13 % prose / +28 % code / +61 % recall. Here it wins big on **recall** (2.05x -> 2.40x)
   and *loses* on code and prose at `n_max 9..12`, because qwen4exp's acceptance profile lets it
   over-draft (code `adaptive 12` per-position acceptance falls 0.94 -> 0.45 -> 0.22 -> 0.07). This is a
   **tuning** item, not a structural one: `adaptive 7` already beats `n3` on code (63.1 vs 61.4).
4. **The only MTP-side gap is correctness/compat: `nextn_shared_target_tensors` support** (above). Until
   it lands, the MTP head the other solution's model ships cannot be used on our delivery at all.

Raw logs: `/tmp/mtpq-*.log` (this box; the `-lv 4` logs are large, the metric is the
`draft acceptance` / `acc per pos` line and the `[ Prompt ... | Generation ... ]` tail).
