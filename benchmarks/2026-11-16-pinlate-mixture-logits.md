# The mid-run pinned-tier latch's MIXTURE: a ~1e-5 arithmetic path, and why the gate must use the logits (`TODO` #282, retired r299)

**Claim.**  `MOE_HOST_POOL_PIN_FAULT=late` (the Windows latched-pinned-refusal shape, issue #278) leaves a run in a
**mixture**: pools built before the refusal keep serving, and every op whose fill can no longer be sourced is stood
down and served by the scheduler.  That mixture takes a different fp32 association order.  It is **not** corruption:
the logits move by ~1e-5 (two orders inside the delivery's 0.005 tolerance) and a 32-token greedy run can flip one
near-tie token.  Therefore the arm's criterion is the **logit instrument**, never the generated text.

**Environment.**  `soar`, 1 x R9700 visible (`HIP_VISIBLE_DEVICES=0`, gfx1201, ROCm 10.1.0-gfx120X),
`Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS` shard 1 (`-ncmoe 47`, `MOE_HOST_LIMIT_FRAC=40`, `--load-mode mmap`,
`MOE_EXPERT_HEATMAP=load`, `--fit off -ngl all`, `-c 4096 -ub/-b 1024 -t 8`, `-fa on -ctk/-ctv q8_0`,
`-p "The capital of France is" -n 32 --temp 0 --seed 42`).  The KLD runs use the delivery's standard point:
`llama-perplexity -f /llm/models/wikitext-2-raw/wiki.test.raw -c 512 --chunks 2 --kl-divergence`.

## 1. The arm had silently stopped reproducing the field shape

`late` used to fire on the first bounce-ring grow after `g_sized`.  When sizing moved to **registration**
(r254/r292), `g_sized` became true *before* the pool tier is built, so the latch fired on the second pool:

| build | pools built before the latch | tier allocated | cache-served / declined ops | fills from pre-latch pools |
|---|---|---:|---|---:|
| r298 (as shipped) | **1 of 141** | 239 MiB | 2 / 5 067 | 73 |
| r298 + `g_routed_op_seen` | **141 of 141** | 31 245 MiB | 658 / 1 084 | 6 278 |
| r298 + fix, gate run (one rep) | 141 | 30 791 MiB | 660 / 1 082 | 6 342 (151 failed fills) |

So the "mixture" the gate was asserting about had degenerated to **two operations**.  The fix keys `late` on the
first **routed op** as well (`g_routed_op_seen`, set in `moe_cache_update_host` / `_gather_host` /
`_stage_table`): the pools are built at placement, the run goes live, and the first later grow is refused — the
field's sequence.

## 2. The divergence is real, deterministic on the affected builds, and ~1e-5

With the field-shape latch, the r231..r269 lineage reproduces a text divergence **deterministically**, and r270+
does not:

| build | `ref` | `pinlate` |
|---|---|---|
| r231 `a19aab55e` | 182 chars `a99ccbd97ecb` | 158 chars `c634f36735fa` |
| r254 `v16-a691b2345-r254` | 182 `a99ccbd97ecb` | 158 `c634f36735fa` (2/2) |
| r262 / r264 / r269 | 182 `a99ccbd97ecb` | 158 `c634f36735fa` (2/2 each) |
| **r270** `v17-a691b2345-r270` (first fixed) | 182 `a99ccbd97ecb` | 182 `a99ccbd97ecb` (2/2) |
| r273 / r298 + latch fix | 182 `a99ccbd97ecb` | 182 `a99ccbd97ecb` (5/5 at r298) |

The generated texts differ from the **fourth** token of the answer (the model closes its reasoning preamble one
token earlier); everything before it is identical.

**The magnitude, measured on r269 (which diverges) with the field-shape latch:**

```
Mean    KLD:   0.000001           Same top p: 100.000 ± 0.000 %
Maximum KLD:   0.000051
99.9%   KLD:   0.000050
99.0%   KLD:   0.000042
Mean PPL(Q) 3.694808  vs  Mean PPL(base) 3.694801
```

i.e. **~1e-5**, two orders of magnitude inside the 0.005 / 98 % limits — the same verdict as the original r231
measurement, now reproduced under the field-shape latch.  On the shipped tree (r298 + the latch fix) the same
comparison reads **mean KLD 0.000000 / max 0.000049 / same-top-p 100.000 %** with the text byte-identical in 5/5
runs: the flip is a near-tie, and *which* residue lands on it is luck, not correctness.

## 3. The flip is layout-dependent — so "the text matches" is not evidence about the arithmetic

On r269, with the same divergence, four arms isolate the carrier:

| r269 arm (all with `MOE_HOST_POOL_PIN_FAULT=late`) | text |
|---|---|
| default (`offset` prewarm order, sink on) | 158 `c634f36735fa` |
| `MOE_EXPERT_PREFILL_IN_PLACE=0` | 158 `c634f36735fa` (not the in-place read) |
| `MOE_HOST_POOL_SINK=0` | 175 `d3d1fe6b3db3` (a *third* text — the demote changes the layout too) |
| `MOE_HOST_POOL_FILL_ORDER=rank` (the pre-r231 prewarm order) | **182 `a99ccbd97ecb` (== ref)** |

The fill ORDER decides which experts land before a RAM-capped pool runs out, so it decides the residency set — and
with it the near-tie.  r270 (`TODO` #318) changed the prewarm's issue order, which is why the flip disappeared there
without any numerics change.  Nothing here is wrong bytes: `MOE_HOST_POOL_VERIFY=1` reports 0 mismatches in the
latched arm, and the WHOLESALE arms (`fail`, `late`: `get_ok=0`) stay byte-identical to `ref` on every build, i.e.
the fallback computes the same logits as the cache.

## 4. What the gate does now

`scripts/gate-host-pool.sh` gates the mixture on the delivery's own logit instrument:

* the `ref` config records the KLD base in the same run (no golden file to go stale), then the `pinlate` config is
  compared against it; the arm passes only inside **mean KLD <= 0.005 and same-top-p >= 98 %**;
* the KLD run must itself be a **mixture** (a rolled-back fill, a pre-latch pool still serving, the latch line), so a
  degenerate latch cannot pass it by being inert;
* the WHOLESALE arms remain byte-identical HARD checks, and still FAIL if they diverge;
* `pinlate`'s text is reported but is not a criterion (`GREEDY-PURITY.md` §95).

Measured at r299: `gate-host-pool.sh` **59/59** (56 + the three KLD checks), `pinlate` text byte-identical this run
(`a99ccbd97ecb`) and its logits at 0.000000 / 100.000 %.

**Record:** `WORKLOG.md` 261116.006 (r299); `INVARIANTS.md` r299; `GREEDY-PURITY.md` §95; `ENVIRONMENT.md`
(`MOE_HOST_POOL_PIN_FAULT`).
