# Greedy text is a function of the cache's residency state (TODO #321, r284)

**Question.** `TODO` #321 recorded that an MTP draft over 2 devices was not reproducible (3 runs, 3 texts) and
attributed it to the draft path across a device split. This record corrects the attribution and names the real
input: the **persisted MoE expert heat prior**, i.e. which expert slots are resident.

**Answer in one line.** With the drafter model loaded, a 2-device `-sm layer` run's greedy text depends on the
contents of `<model>.moe-heatmap`; a fixed prior is deterministic (6/6 and 2/2+2/2 on separate occasions), a
different prior gives a different text, and `MOE_EXPERT_PREFILL_IN_PLACE=0` collapses the two to one text. The
nondeterminism is therefore **not** in the MTP driver, and it is not a race.

## The arm

Unless a row says otherwise: Flash-Next `GSQ-RCO-IQ3_XXS` + `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` as `-md`,
`HIP_VISIBLE_DEVICES=0,1` (2 × R9700, gfx1201), `-sm layer` (the default), `-ncmoe 48`,
`-ot 'per_layer_token_embd.*=CPU'`, `-lm none -lzm off`, `-c 32768 -ctk q8_0 -ctv q8_0`,
`-f prompts/kld-corpus-35b-in-domain.txt` (25 763 tokens), `-n 64 --seed 42 --temp 0
--no-display-prompt --single-turn`, `ROCBLAS_USE_HIPBLASLT=0`, hashed with `scripts/extract-generated.py`.

## What reproduces it, and what does not

| arm | runs | texts |
|---|---|---|
| 2 devices, `--spec-type draft-mtp` (the recorded arm) | 3 | `c5f360a75ec7` / `7ca535e1da91` / `7a3474cbe5f1` |
| 2 devices, `--spec-type none` **with** `-md` loaded (draft present, never used) | 4 + 5 | `7a3474cbe5f1` ×7, `f40ae789f219` ×1, `7ca535e1da91` ×1 → **unstable** |
| 2 devices, `--spec-type none`, **no `-md` at all** | 10 | `7a3474cbe5f1` ×10 → **stable** |
| 2 devices, draft, `GGML_CUDA_DISABLE_GRAPHS=1` | 3 | `c5f360a75ec7` / `7ca535e1da91` / `7a3474cbe5f1` → unstable (not graph replay) |
| 2 devices, draft, `MOE_EXPERT_CACHE_MIB=0` | 7 | `f40ae789f219` ×7 → stable (no cache, no residency) |
| 1 device, draft (r277 + r283 sessions) | 3 + 3 | stable ×6 |
| 2 devices, 35B-A3B Q8_0, `-ncmoe 0` **and** `-ncmoe 48`, long prompt | 6 + 6 | `75d0ffc67ca5` ×12 → stable (the tier is active and deterministic there) |
| 2 devices, draft, short prompt (`prompts/prose-rdna-boosts.txt`, `-c 8192`) | 6 | `7c5a512a8447` ×6 → stable |

So the trigger is **the drafter model being loaded** (not used) on a 2-device split with the arena in the
*partly resident* band: the draft's context reserve shrinks each device's arena, which is what puts the expert
tables below full residency (measured coverage 74–78 % with the drafter, 99.65 % without it).

## The differential: same build, same command, one file

`cp` a `.moe-heatmap` before the run, or delete it. Every other byte of the command line is identical.

| prior | runs | text |
|---|---|---|
| the file left by a live run | 2 + 2 | `7ca535e1da91` ×4 |
| absent (empty prior) | 2 + 2 | `7a3474cbe5f1` ×4 |

The second row pair was re-measured on a quiet box (the first sweep ran under self-inflicted memory pressure
from an unrelated `tmpfs` log file; the differential is unchanged). `MOE_EXPERT_HEATMAP=load` (frozen, never
written) gives `7ca535e1da91` ×6 — i.e. **the run-to-run variation of the recorded arms is entirely the prior
file that the previous run wrote**.

## Which knob carries it

| knob (prior SET vs prior absent) | texts | prior-dependent? |
|---|---|---|
| none (default) | `7ca535e1da91` / `7a3474cbe5f1` | yes |
| `GGML_MOE_DEVREMAP=0` | `7ca535e1da91` / `7a3474cbe5f1` | yes |
| `MOE_EXPERT_CACHE_IDENTITY_PREF=0` | `7ca535e1da91` / `7a3474cbe5f1` | yes |
| `GGML_CUDA_MMB=0` | `7ca535e1da91` / `8bd22760ce3c` | yes |
| **`MOE_EXPERT_PREFILL_IN_PLACE=0`** | `7a3474cbe5f1` / `7a3474cbe5f1` | **no** |

Two facts fall out of the `GGML_CUDA_MMB=0` row: the prior-SET arm's text does **not** change when the quantized
prefill GEMM is disabled, while the prior-absent arm's does. The two residencies genuinely run **different
kernel sets** — which is what `MOE_EXPERT_PREFILL_IN_PLACE` is documented to do: a table that is fully resident
is served in place and *its prefill fusions stand down* (`ENVIRONMENT.md` §`MOE_EXPERT_PREFILL_IN_PLACE`,
r258/r266).

## Where the difference lands, and how big it is

* The two priors agree for the first **517 characters** of the 64-token generation and diverge after it, so the
  first flipped token is deep inside the generation — the shape of a **near-tie** under a tiny numeric
  difference, not of a mis-read expert (a wrong expert of eight would move the output from the first token).
* **The magnitude is NOT measured.** `llama-perplexity --kl-divergence` is the instrument for it, and it cannot
  load a drafter (`-md` is a `llama-cli`/`llama-server` option), which is exactly the state that produces the
  partial residency. Run without the drafter, both priors give **KLD 0.000000 / same-top-p 100 %** at
  `-c 8192 --chunks 3 -ub 2048`, `-c 12288 --chunks 2 -ub 512` and (wikitext, `-c 32768 --chunks 1`)
  **99.65 % residency** — i.e. the probe measures the identity/full-residency regime, where the two paths agree.
  Whoever closes this owes that measurement (a prefill-logit arm that can load a drafter, or an `-md` in
  `llama-perplexity`).

## What shipped, and what it means for comparisons

* `scripts/gate-kv-residency.sh` now forces `MOE_EXPERT_HEATMAP=load` on **every** arm, next to
  `ROCBLAS_USE_HIPBLASLT=0`. Both are instrument hygiene for a fresh-process comparison: the two arms must
  start from the same residency, or the gate compares cache states instead of placements. The assertions are
  unchanged — the arms are still required to be **byte-identical** — and the multi-device section went from
  three different texts in three arms (r278: `7ca535e1da91` / `f260d1824ab3` / `b08d3dbcb1e9`) to
  `b08d3dbcb1e9` ×3 with the per-device counter blocks intact (r284, full log in the `WORKLOG.md` entry).
* The general rule, for every arm of every gate: **a text comparison across processes must pin the expert-cache
  state** (`MOE_EXPERT_HEATMAP=load`, or a read-only sidecar). This is the same class as `INVARIANTS.md` §44 and
  as `GREEDY-PURITY.md` §11 (a different kernel family reduces in a different order, so a near-tie can drift):
  the difference is that the *input* that picks the kernel here is a cache file rather than a width knob.
* The remaining question — whether the in-place arm and the staged arm *should* be bit-identical (they are
  documented as reading the same bytes) — is an accepted limitation, not an open item: `TODO.md` → accepted
  limitations, `GREEDY-PURITY.md` §89.
