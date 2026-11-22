# The MoE device gather's used-expert bitmap (r345, issue #350)

**Question.** `moe_cache_gather_kernel`'s grid is `n_experts * n_split` blocks and every block scans the whole
routing (`n_used * n_tok` ids) to answer one question: "is MY expert routed?".  r325 raised `n_split` 8 -> 32 for
the prefill band and measured +7.5 % at 84 % residency.  Issue #350 reports the opposite at **55 % residency** on
Windows: `n_split` 32 is ~11 % slower than 16, ~10 % slower than 8, and ~19 % slower than 4.  Why does a deep
split lose at low residency, and what should the default be?

**Answer.**  The copies shrink with `n_split`; the scans do not.  One pass builds a used-expert bitmap and each
block reads one word instead of scanning, which makes `n_split` proportional to the copy work only.  With it, a
deep split is the best arm at BOTH residency extremes and the low-residency regression disappears.

## Instrument

`moe_cache_gather_mask_kernel` (one pass, per-block shared-memory bitmap, one atomic per word) + a per-device
`uint32_t` bitmap, engaged when the scan volume it replaces (`n_experts * n_split * n_used * n_tok` ids) is at
least `1e8`.  `MOE_GATHER_MASK=0` restores the per-block scan.

Box: 1 x R9700 (gfx1201), ROCm 10.2, 16 cores, `~/.cache/rdna-boosts` on NVMe.  Pins on every arm:
`MOE_EXPERT_HEATMAP=load`, `ROCBLAS_USE_HIPBLASLT=0` (`INVARIANTS.md` r44/r284).  `-b 2048 -ub 2048`,
`--seed 42 --temp 0 -st`.  Every arm is a fresh process; the prefill rate is `llama-cli`'s own
`Prompt: N t/s` on an identical prompt.

* **low-residency shape (the reporter's)**: Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS** (512 experts/table),
  `-ngl all -ncmoe 47 -lme mirror --load-mode auto --lazy-mode on -fa on -ctk q8_0 -ctv q8_0 -c 32768 -t 8`,
  `-ot per_layer_token_embd.*=CPU`, `MOE_EXPERT_CACHE_MIB=21882` -> **arena 21881.8 MiB of 40012.5 MiB host
  experts = 54.7 % residency**, the reporter's own number.  Prompt: 17 624 tokens of wikitext prose.
* **high-residency shape (r325's)**: Qwen3.6-35B-A3B **Q8_0** (256 experts/table), `-ncmoe 40 -lme auto`,
  arena 26787.8 MiB = **84.0 % residency**, `-c 8192`.  Prompt: `prompts/prose-rdna-boosts.txt` (5246 tokens).

## 1. Low residency (54.7 %) — the reported regression, reproduced in direction

`ns=n_split`, rate in t/s (3 interleaved reps; the spread inside an arm was +-0.5 t/s):

| `n_split` | per-block scan (shipped until r345) | used-expert bitmap |
|---:|---:|---:|
| 1 | 1660.7 | -- |
| 4 | 1719.3 | 1690.2 |
| 8 | 1727.0 | 1706.6 |
| 16 | 1716.8 | 1728.0 |
| **32** | **1671.7** | **1741.0** |
| 64 | -- | 1738.9 |
| 128 | -- | 1732.0 |

So on this box the shipped default (`32`, scan) is **2.8 % below** `n_split` 4 — the reporter's ordering.  With
the bitmap the curve inverts and 32 is the best arm: **+4.1 % over the shipped default** and +1.9 % over the old
best (`4`).  The mechanism is the one the arithmetic predicts: the bitmap removes `n_experts * n_split - 1`
redundant scans (16384 blocks x 16 384 ids on this shape), which is exactly the term that grows with `n_split`.

Scope, stated so it is not over-read: the reporter measures a **~19 %** penalty where this box measures 2.8 %.
The bitmap removes the n_split-proportional *work*; if a platform's remaining penalty is per-block launch cost
rather than the scan, this changes nothing there and the fix for that class is a smaller grid, which needs their
data to size.  What this change does buy on every platform is that the shipped default is no longer the worst arm
at low residency.

## 2. High residency (84 %) — r325's win is kept, and the flat optimum moves up

| `n_split` | per-block scan | used-expert bitmap | delta |
|---:|---:|---:|---:|
| 4 | 3484.2 | -- | |
| 8 | 3645.2 | 3593.3 | -1.4 % |
| 16 | 3757.1 | 3669.5 | -2.3 % |
| **32** | **3722.3** | **3777.6** | **+1.5 %** |
| 64 | 3412.6 | 3817.8 | +11.9 % |
| 128 | -- | 3804.7 | |

The bitmap only pays above ~`1e8` scan ids: at 8/16 the extra pass is latency on the critical path and it buys
nothing (that is what the threshold is calibrated on, and why `n_split` 4-8 remains exactly the pre-r345 path).
At the shipped 32 it is a small win here and a large one at 55 % residency; at 64 it turns a 8 % regression into
a 12 % win, which is why the bitmap — not a different constant — is the fix.

## 3. The shipped default, on the released tree

`release.json` tree `0d2f70e6c` (the working chain's build `ac8515642`; the canonical re-cut is `e695c46ce`),
no env knobs, three interleaved pairs:

| shape | shipped (bitmap at 32) | `MOE_GATHER_MASK=0` (pre-r345) | delta |
|---|---:|---:|---:|
| low residency 54.7 % (IQ3_XXS, 17 624 tokens) | **1740.8 / 1730.4 / 1720.2** | 1669.4 / 1668.3 / 1674.5 | **+3.6 %** |
| the same shape, an earlier 1-pair set on the working chain build | **1748.5 / 1751.3** | 1676.5 | **+4.4 %** |
| high residency 84.0 % (35B-A3B Q8_0, 5246 tokens) | **3737.2 / 3787.8** | 3732.5 | +0.8 % |
| decode, low-residency shape, `-n 128` | 49.7 t/s | 49.5 t/s | noise |

Census (`-lv 4`, low-residency shape, one 17 624-token prefill): `device gather: 1380 calls (1380 pool-backed,
1242 by the used-expert bitmap)`; with `MOE_GATHER_MASK=0`, `0 by the used-expert bitmap`.  The 138 calls that
keep the scan are the narrow ones (the first decode-band-sized routings of the run) — the width gate working.
At the decode band the bitmap never engages: a decode arm's census reads `0 by the used-expert bitmap` out of
138 calls, and a 9..16-row verify is at 8 x 16 x 512 x 32 = 2.1e6 ids, five orders below the threshold, so no
captured graph changes shape either.

## 4. Purity

* Greedy text is **byte-identical** in every arm: `172 chars sha=de61b9b1bed7` (low-residency shape) across
  bitmap on/off and `n_split` 4/32; `206 chars sha=cf7f8b23f404` (high-residency shape) across bitmap on/off.
  The bitmap replaces an "is this expert routed" TEST; it never changes a byte of what is copied, from where, or
  in which order.
* `scripts/gate-prefill-logits.sh --ab "MOE_GATHER_MASK=0"` on the low-residency shape (`-ncmoe 47 -lme mirror
  -ub 2048`, q8_0 KV, `-c 512`, 40 chunks, pins above): mean KLD **-0.000000**, same-top-p **100.000 %**, and the
  GPU fault check clean — the exact-fallback A/B the mode exists for.
* `scripts/gate-host-pool.sh`: **58/58** checks, `takeover=0` under both fill faults, every arm's text as
  recorded, GPU faults clean.
* `scripts/gate-host-expert-text.sh`: **3/3** identical (`328 chars sha=7c5a512a8447`); the flipped-library arm
  reads its own text (`298 chars sha=55a50dee05e0`, reported, r310).
* MTP Protocol A (27B-UD-Q4_K_XL, `--fit off`, bf16 KV, `-n 3000`): reasoning **on** `draft-mtp` **54.9 t/s**
  against `none` **30.2** and reasoning **off** **72.2** against **30.3**, the same text inside each pair
  (`12331 chars sha=eace21fc3399` / `6789 chars sha=4172642d7dc2`) — the decode band is untouched, as the census
  says.
* `scripts/validate-set.sh` green; `check-rung-file-sets.py` accounts for every file/define/env/option
  (575 -> 576 envs: `MOE_GATHER_MASK`), 0 losses.

Record: `WORKLOG.md` 261122.005; `INVARIANTS.md` r345; `GREEDY-PURITY.md` §104; `ENVIRONMENT.md` §2;
`patches/README.md` block 09; issue #350.
