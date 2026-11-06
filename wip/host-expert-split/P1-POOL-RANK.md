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

## The gate ladder, as far as it can honestly be run today

All arms: 1 × R9700, Flash-Next GSQ-RCO IQ3_S, `-ncmoe 48`, 4 GiB arena, `MOE_HOST_POOL_DIO=1`,
`MOE_HOST_POOL_RAM_PCT=0`, every run under `scripts/gpu-watch.sh --cmd`.

### Coherence — PASS, 9/9 byte-identical

`-p "The capital of France is" -n 24 --seed 42 --temp 0 --single-turn`, at **three pool budgets × three
rankings** (`-lme 36` below the complement, `-lme 44` at it, `-lme 47` the mirror):

```
lme36/44/47 × combined/decode/floor  ->  137 chars sha=dfc26e26f38c   (all nine)
```

A placement prior must not move a token, and this one does not.  (The first attempt looked like it failed —
nine different hashes at the *same* length — because `-lv 3` interleaves timing lines into the generated
text.  Clean logging, nine identical hashes.  Worth remembering: never hash an `-lv 3` log.)

### Prefill A/B — `decode`/`floor` are never worse, and the mirror is provably inert

5246-token prose prompt, `-ub 6144`, `-n 20`, DIO pool:

| `-lme` | pool covers | rank | prompt t/s | gen t/s | pool `h` |
|---|---|---|---|---|---|
| 36 | 76.9 % (below the complement) | combined | 309.5 | 18.2 | 0.1916 |
| | | **decode** | **313.1** | **20.6** | 0.1999 |
| | | floor | 315.3 | 20.2 | 0.1980 |
| 44 | ≈ the complement | combined | 719.5 | 22.8 | 0.6662 |
| | | **decode** | **727.4** | **23.9** | **0.7950** |
| | | floor | 727.2 | 23.6 | 0.7926 |
| 47 | **mirror** (all 46.84 GiB) | combined | 833.7 | 24.8 | 1.0000 |
| | | decode | 835.0 | 25.1 | 1.0000 |
| | | floor | 833.4 | 24.8 | 1.0000 |

**Answers to the two questions.**

* **A full complement, yes — that is where it is biggest.**  At `-lme 44` the pool's hit rate goes
  **0.666 -> 0.795 (+19 % relative)**, prompt +1.1 %, gen +4.8 %.  The mechanism is that the loader's
  "complement" is the *prior's* complement, and the prior is a bad predictor of what the arena will actually
  hold; the decode vector predicts it better, so a budget that is exactly the complement's size lands its
  slots on the experts that are actually cold.
* **A full mirror, no — provably inert.**  At `-lme 47` the pool holds every expert, so the ranking can only
  change slot *order*: the arena/pool counters are **identical to the digit** (`h=1.0000 (13606/13606)`, arena
  0.5028 / 15009 hits at the same shape) and prompt/gen are within run noise (833.7 / 835.0 / 833.4).  A
  mirror is also the fastest row by a wide margin (833 t/s prefill against 309 at a 77 % pool), which is the
  post-cache baseline to beat — and exactly why a partial pool has to earn its place.

**The coherence claim was measured without a pool (corrected 2026-11-06).**  The arm that produced
`sha=dfc26e26f38c` used `-p "The capital of France is" -n 24`, whose 9-token prompt never reaches the first
decode-band pass, so **no pool was ever built** and the hash proved only that the non-pool path is unchanged.
Re-run with the prose prompt, which does engage it (verified at `-lv 2` on the identical config), at default
verbosity (an `-lv 3` log cannot be hashed: its timing lines inflate the extraction from 126 to 1691 chars):

| `-lme 36`, pool active, prose, `-n 24` | `SINK=0` | `SINK=1` |
|---|---|---|
| `combined` | `880c932888f8` | `880c932888f8` |
| `decode` | `880c932888f8` | `880c932888f8` |

**4/4 byte-identical with the pool engaged**, so both the ranking change and the W2 demote are placement-only,
as they must be.  `880c932888f8` (126 chars) is the reference hash for a pool-active arm.

### `llama-batched-bench -npl 1,4,8` — a GATE, not a decision input (corrected 2026-11-06)

**The +20.8 % at 8 slots was a cold-start artifact and is RETRACTED.**  Three alternating repeats on the
unmodified binary give **+0.96 %** (npl 8: per-rep +4.06 / +0.03 / −1.20), −0.57 % at npl 1 and +1.29 % at
npl 4, with the pool hit rate **identical to three decimals** (`combined` 0.4665–0.4712, `decode`
0.4750–0.4654–0.4663) — at `-lme 44` the pool is 94 % of the set, i.e. nearly a mirror, where the ranking is
inert by construction (S3).  The original pair compared a cold first run with a warm second one; the same
first-run penalty is visible in today's `combined` r1 (31.00 t/s) against its r2/r3 (32.87/33.30).

Two things follow, and they are worth keeping:

1. **This utility must never be a tie-breaker for a default.**  `npp 512 -ntg 128 -npl 1,4,8` is a
   prefill-dominated synthetic (4 096 prefill tokens per 1 024 decode tokens at 8 slots, eight independent
   512-token sequences).  The campaign's workload is one user, a 5 298-token prompt, 512 generated tokens,
   one stream.  A sign flip between the two populations says the ranking is *workload-sensitive* (true and
   useful) and says nothing about which default a single user should get.
2. **On the concurrency path the ranking's sign inverts, so `decode` cannot be the default.**  At `-lme 36`
   (a budget where the prior actually decides placement) three alternating repeats give
   **−6.35 % / −11.41 % / +0.10 %** at npl 1 / 4 / 8, per-rep consistent and with no memory confound
   (MemAvailable ≈ 41.9 GiB in every arm), while the pool hit rate falls slightly (0.1738 → 0.1694).  The
   mechanism is the staging duty: the pool's content is *one* decision made early, and with a
   prefill-dominated mix `combined` heat is the better predictor of it.  This is a **gate observation on the
   multi-slot path**, which the delivery does ship — it is not evidence about the single-user default.

**What decides it is the single-user cell**, measured 3/3 in alternating order on this binary at `-lme 36`:

| `llama-cli`, prose, `-n 64` | gen t/s | prompt t/s | pool h |
|---|---|---|---|
| `combined` | 22.5 ×3 | 309–312 | 0.2090 (14445/69105) |
| `decode` | **24.9 ×3 (+10.7 %)** | 313–315 | 0.2205 (15061/68295) |

**Confound check (2026-11-06).**  Every arm above ran `-lm mmap -lzm auto`.  `auto` does *not* engage the
managed PLE reader here (`llama_lazy_managed_budget()` returns 0 without `LLAMA_LAZY_BUF_MB`, and its comment
says so), but it does leave the CPU-offloaded PLE tensor (`-ot 'per_layer_token_embd.*=CPU'`) on the classic
mmap-lazy path: page-fault loaded on demand, competing for the page cache that `MOE_HOST_POOL_DIO=1` makes the
pool bypass.  Re-run with `-lzm off` (PLE prefaulted, no on-demand loading):

| `-lme 36`, `-lzm off` | gen t/s | pool h |
|---|---|---|
| `combined` | 22.6 / 22.6 / 22.6 | 0.2090-0.2092 |
| `decode` | 24.5 / 24.8 / 25.0 (mean 24.77) | 0.2207 |

**+9.6 %**, so the effect survives the confound removal (the `-lzm auto` figure was +10.7 %); quote **+9.6 %**.
Process `read_bytes` is *not* comparable across lazy modes (78 GB under `auto` vs ~102 GB under `off` on the
same cells, because prefaulting PLE moves its reads into load time) — quote the pool's own DIO counters.

So: `decode` is a real **+9.6 %** on the single-user path and a real **−11 %** on the 4-slot batched path.
The only candidate that can serve both is **`floor`** (combined heat for the top `MOE_HOST_POOL_RANK_FLOOR`
experts, decode heat below), which is what the next arm must test — on both cells.

### Still owed (not run, and why)

* **The four-axis MTP protocol** (`benchmarks/mtp-adaptive-methodology.md`) — needs the MTP draft head and
  `-n 3000` per axis; it is the gate `MANIFESTS.md` demands for a decode-path change and it has **not** run.
* **`scripts/gate-prefill-logits.sh`** — there is no recorded base for this model (`~/.cache/rdna-boosts/gates/`
  has gpt-oss and 35B-A3B only), and recording one from this modified build would make the comparison
  vacuous.  It needs either the 35B-A3B's existing base (a different model, so it checks the change on a
  config with a known-good reference) or a base recorded from the delivery tip before the change.
* **A repeat of the 8-slot batched-bench row.**
