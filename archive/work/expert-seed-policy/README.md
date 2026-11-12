# The expert seed: lazy or eager, one policy for both tiers (`TODO` #318) — CLOSED, r270

**Outcome: measured, and the answer is `eager`.**  The sizing fill is not a fixed cost to defer — on both tiers it
is the *prefetch* the demand path would otherwise pay inline, at 8x (discrete pool) to 2.4x (APU) worse latency, for
the same or more bytes.  What the item was really pointing at was the fill's **transport**, which is fixed here.

* **The delivery:** `MOE_EXPERT_SEED` (`eager` default | `auto` | `lazy`), one knob covering the arena seed and the
  host-pool prewarm, plus `MOE_HOST_POOL_FILL_ORDER` extended to the prewarm (file order: -38 % of the seed,
  -20 % TTFT, text byte-identical).
* **The numbers, the arms and the configs:** [`benchmarks/2026-11-12-expert-seed-policy.md`](../../../benchmarks/2026-11-12-expert-seed-policy.md).
* **The raw rows:** `RESULTS.csv` (one row per arm; see `summarize.sh` for the column order).
* **The harness:** `arm.sh` (three runs per arm: TTFT wall clock, a no-`-lv` text/hash run with `/proc/io`
  sampling, and an `-lv 4` census run) and `summarize.sh`.

## Why this folder exists

`TODO` #318 was split out of #317 (r266) with the hypothesis that, once the prefill band is served IN PLACE out of
the GTT arena, the seed can be lazy — *"the prior only needs to RANK evictions"*.  It also said the two tiers must
share one policy.  They now share one knob and one question, and the answer is the same on both: the fill is a
placement/prefetch, not a choice.

The discrete pool tier is where the port fails hardest, and the reason is structural rather than a matter of tuning:
its prefill **stages the whole expert table** into the op's `input_cpy` and prefetches the complement whether the
pass routes it or not, so a prefill needs the host set resident every pass.  The APU tier keeps its in-place band,
so there the deferred seed really is cheaper at the front — but the follow-up decode then misses 2.4x as often and
reads 70 GiB more over a 250-token answer.

## Reproduce

```bash
# soar (gfx1201): the bounded pool, deterministic size, cold page cache before every run
cd archive/work/expert-seed-policy          # from the repo root
LABEL=pool32 MOE_HOST_POOL_RAM_PCT=0 ./arm.sh eager  MOE_EXPERT_SEED=eager -- -lme 32
LABEL=pool32 MOE_HOST_POOL_RAM_PCT=0 ./arm.sh auto   MOE_EXPERT_SEED=auto  -- -lme 32
./summarize.sh runs/pool32/eager pool32

# halo (gfx1151): the GTT arena
LABEL=apu MODEL=/llm/models/Qwen3.5/122B-A10B/Q5_K_XL/Qwen3.5-122B-A10B-UD-Q5_K_XL-00001-of-00003.gguf \
  NCMOE=none CTX=4096 UB=1024 NGEN=250 ./arm.sh eager MOE_EXPERT_SEED=eager MOE_EXPERT_HEATMAP=load -- -lme 70
```

Notes that cost a run each to learn: pin the pool budget (`-lme N` + `MOE_HOST_POOL_RAM_PCT=0`) or the AUTO sizing
reads `MemAvailable` at the first pool build and an arm that faulted the model into the page cache first sizes a
smaller pool; drop the page cache before every run or "cold" is not comparable; and `--ignore-eos` is required for a
`-n N` decode arm (this model otherwise answers in ~50 tokens).
