# fast-topk-penalties — a penalty no longer stands the exact top-k fast path down

**Status: PROMOTED (r361, 2026-11-25) into a block-12 amendment.**  This directory is the record, not a
campaign: the code is in `patches/0012-*.patch`, the gate is `scripts/gate-fast-topk-penalties.sh`
(`MANIFESTS.md` gate 45), the knobs are `DIAGNOSTIC.md` §27, the rule is `GREEDY-PURITY.md` §110 and the
dated entry is `WORKLOG.md` 261125.006.

## The report (GitHub issue #366, `@mjohanning99`)

A presence penalty of `0.15` cost the reporter **5.7 %** of decode on a gfx1201 + gfx1100 pair
(`-sm tensor`, `llama-server`, Qwen3.8-27B Q4_K_M, q8_0 KV, `--spec-type draft-mtp-adaptive,ngram-mod`,
`--spec-draft-n-max 5`, `temp 0.8 top-p 0.95 top-k 20 min-p 0.03`, 9 samples per arm): 69.7 t/s at
`presence 0.15` against **73.7 t/s at `0`**, acceptance unchanged (57.5 % / 58.1 %).  A `presence 0` arm on
Qwen's thinking-mode set read 72.5 against the penalty set's 68.4, so it is the penalty and not the
temperature.  Their analysis was exactly right: `common_sampler_fast_topk_k()` returned 0 for any
non-trivial penalties sampler, so every penalised profile sampled from a **full-vocabulary** candidate array
— an array build plus a `token_count` hash lookup per candidate — for every sampled token, and MTP samples
several per step.  They also noted the fast path already handled a logit bias exactly, and asked whether a
penalty could be handled the same way.

## The reproduction on this box

3× R9700 (gfx1201), `-sm tensor -mg 0`, Qwen3.8-27B UD-Q4_K_M, `--spec-type draft-mtp`,
`--presence-penalty` the only difference, 300 tokens, 5 alternating rounds, same prompt, `--seed 7`:

| arm | decode (t/s) |
|---|---:|
| `GGML_LF_FAST_TOPK` default (fast path, penalised) | **78.26** |
| `GGML_LF_FAST_TOPK=0` (the pre-fix arm for a penalised profile) | 75.00 |

**+4.3 %** with byte-identical text.  `ab.sh` in this directory is the exactness A/B (7 profiles: presence,
repeat, frequency, all three, a penalty combined with a logit bias, no penalty, and a 4B model on one GPU);
the gate generalises it and adds the census/self-check/cap arms.

## What shipped

`common_sampler_fast_topk_k()` no longer stands down for a penalty.  The reasoning is the logit bias's,
generalised: whatever a sampler ahead of top-k does to a logit it does **per token**, so the fast path only
has to know which tokens can move and what the chain will make of them.

* **The affected set** is the logit-bias set ∪ the penalties sampler's last-n window — bounded (≤ 256
  distinct tokens; `penalty_last_n` is 64 by default) and observable.  The raw top `k + 1 + |affected|`
  still holds the top `k + 1` of the untouched tokens, so the candidate array needs nothing else.
* **The adjusted value** is computed from the token's RAW logit, with `llama_sampler_penalties_apply()`'s own
  arithmetic (`x <= 0 ? x * r : x / r`, then `x -= count * freq + (count > 0) * present`), applied to the
  BIASED value because `common_sampler_init` adds the logit-bias sampler before the user's sampler list.
* **The `k` survivors still carry their raw logits**, so the chain re-applies the bias and the penalty itself
  and the values are bit-identical to the full path.  Only the candidate array is smaller.
* **Two read-only accessors** (`llama_sampler_penalties_get_params()`, `llama_sampler_penalties_get_window()`)
  were added so the fast path reads the chain's own state instead of keeping a second copy of the map — a
  second copy is a state machine that drifts (the clamped `penalty_last_n`, the ring buffer's eviction,
  `llama_sampler_reset`).  The window is re-read once per sample, so an `accept` between samples is seen.
* **Everything else still stands the fast path down** (DRY, top-n-sigma, an unknown sampler, two penalties
  samplers in one chain).  A penalty that runs **after** top-k is deliberately not reproduced: it changes a
  set the fast path never saw.  A window past 256 distinct tokens falls back **per sample** and moves no
  token — measured on a 700-token run (472 fast / 228 fallbacks, text identical to the full-path arm).

## Why the instruments exist

The failure mode is a **slow text drift**, not a crash: a wrong candidate set or a wrong rounding moves the
sampled tokens by a hair and no text arm sees it if both sides fell back.  So the gate has three oracles and
none of them is readable alone:

1. **`GGML_LF_FAST_TOPK=0` is the text oracle** — for a penalised profile it IS what r360 shipped, so the
   fast arm must match it byte for byte (7 profiles, two shapes).
2. **`GGML_LF_FAST_TOPK_CENSUS=1` is the non-vacuity check** — `samples fast`, how many carried a penalty
   window, `full` fallbacks, the largest affected set, and the self-check's mismatch count.  A separate
   `-lv 4` run: a `-lv 4` stream is never hashed.
3. **`GGML_LF_FAST_TOPK_SELFCHECK=1`** scores the whole vocabulary the way the chain would, per sample, and
   compares the `k`; `=2` is its **control** — it swaps the fast path's two best-ranked tokens, so the check
   must fire on every sample and the text must move.  Without that control "the text is identical" could be
   an inert claim.

The self-check shares the fast path's value function, so it validates the **selection** (the raw-top
extension, the strictness rule) and the full-vocabulary text arm validates the **arithmetic**.

## Not covered, by design

* An **adaptive** speculator cannot be gated for text: its draft depth follows the measured rate, which
  differs between two arms, so the two arms are a different text by construction.  The gate pins
  `--spec-type draft-mtp` (the reporter's `draft-mtp-adaptive` measurement stands on its own).
* A penalty window past 256 distinct tokens still pays the full-vocabulary path (the cap is what keeps the
  per-sample work bounded).  Documented in `ENVIRONMENT.md` §27; it costs nothing until a run holds > 256
  distinct tokens in its window.
* Two penalties samplers in one chain stand the fast path down; nobody configures that.
* Non-AMD backends and Windows are unvalidated here — the change is host-side C++ in `common/` plus two
  presence-free read-only accessors, so the cross-backend surface is nil.
