# P1 — the pool's ranking vector (`MOE_HOST_POOL_RANK`), measured 2026-11-05

**Status: IMPLEMENTED + MEASURED as a wip arm on `~/llama.cpp` branch `p1-pool-rank-r223` (commit
`b2a8d2e76`).  NOT in `patches/`, NOT a release.  The shipped default is unchanged (`combined`), so delivery
behaviour is byte-identical until the gates below have run on a flip.**  Origin: the anomaly in
[`STEP1-REPLAY.md`](STEP1-REPLAY.md) §7.

## The change

`pool_rank_experts_locked` sorted by `t.heat`, the **combined** prefill+decode counter.  The arena already
seeds from the decode-only vector for exactly the band the pool exists to serve (TODO #88); the pool never
got the same treatment — measured worth 12–19 % of the decode L3 traffic in the offline model.

```
MOE_HOST_POOL_RANK=combined | decode | floor | invert        (default: combined = shipped behaviour)
MOE_HOST_POOL_RANK_FLOOR=N                                    (mode `floor`: the prefill-hot % kept ahead, 50)
```

* **`combined`** — `t.heat`, unchanged, the A/B reference.
* **`decode`** — `heat_rank_vec()` (the counter the arena seeds from).  The one the evidence supports.
* **`floor`** — keep the prefill-hot `N` % ahead (the pool also stages whole tables for prefill, which is
  why the combined vector was chosen originally and why that reasoning is not wrong), then rank the rest by
  decode.
* **`invert`** — diagnostic only: ascending heat, the inversion probe.

## Measurement

1 × R9700, Qwen3.8-Flash-Next GSQ-RCO IQ3_S, `-ncmoe 48 -lme 36` (a pool **below** the arena's complement,
which is the regime the offline model says this can matter in), `MOE_EXPERT_CACHE_MIB=4096` (a 4 GiB arena, so
the pool is where the misses go), `MOE_HOST_POOL_DIO=1` (a pool miss is a real disk read whatever the page
cache holds — the honest L3), `MOE_HOST_POOL_RAM_PCT=0`, `-c 8192 -ub 4096`, the shipped 5246-token prose
prompt, `-n 64`, `--temp 0 --seed 42`.  Every arm under `scripts/gpu-watch.sh --cmd`; three runs per arm, in
both orders.

| arm | generation t/s | prompt t/s | pool `h` |
|---|---|---|---|
| `combined` (shipped) | 22.5 / 22.6 / 22.6 | 307.8 / 310.2 / 311.0 | 0.2090 (14 445 / 69 108) |
| **`decode`** | **25.0 / 25.0 / 25.0** | **314.3 / 313.5 / 314.3** | **0.2205 (15 061 / 68 295)** |
| `floor` (50) | 24.4 | 313.5 | 0.2179 |
| `invert` (probe) | 16.7 | 313.9 | 0.1142 |

**+10.6 % generation, +1.1–2.1 % prompt, no extra copies, no extra memory.**  Order-independent (the reversed
pair reproduces 25.0 against 22.6 exactly), `read_bytes` is not quoted because it includes the ~85 GB model
mapping and is too noisy to compare at this scale.

**The inversion probe is the useful negative result**: ascending heat collapses to `h` 0.114 / 16.7 t/s, so
the vectors are *not* being read backwards, and the residual `h` (0.22 while the pool holds 77 % of the
experts) is not an inversion bug — it is the pool's whole-table prefill staging traffic and the arena's
miss distribution mixing in one counter, which is why `h` must not be read as "the fraction of misses the
pool answered" (see `STEP1-REPLAY.md` §7).

## What it owes before it can ship

1. **The prefill A/B on a proper config** — this run's prompt row is `-ub 4096` on one prompt; the pool's
   staging duty is exactly what the combined vector was protecting, so `floor` (and a floor sweep) must be
   compared on prefill-heavy shapes before any default flips.
2. **The four-axis MTP protocol** (`benchmarks/mtp-adaptive-methodology.md`) and
   `llama-batched-bench -npl 1,4,8`: the ranking is a placement prior, so the text must stay byte-identical
   and only the traffic may move.
3. **`scripts/gate-prefill-logits.sh`** — a placement change should be KLD 0, and a non-zero reading would
   mean the layout reached further than the cache.
4. **`ENVIRONMENT.md`** entries for both variables (class: `decode`/`floor` are opt-in until the default
   flips; `invert` is a diagnostic and must not ship).
5. **The default-on decision** is the maintainer's: the measured +10.6 % decode at a below-complement budget
   is exactly the shape `AGENTS.md`'s default-on policy describes (flip in the same change as the gate, keep
   the kill switch), but the prefill duty above is the argument for `floor` over `decode` as the flip value.
