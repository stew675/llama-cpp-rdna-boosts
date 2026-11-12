# The prefill band served IN PLACE from the GTT arena (`TODO` #317, r266)

Block-16 amendment.  The arena tier holds a table only partly; until this change its **prefill** band served every
ubatch by staging the used set into the scheduler's split-input copy (`moe_cache_prefill_stream`), which D2D-copies
every *resident* used expert out of the store on every pass.  The cost is `O(used)`, not `O(missing)`: measured
30 060.5 MiB of D2D against 1 578.6 MiB really read, for a **38-token** prompt on 122B-A10B Q5_K_XL `-lme 75`.

This change is the arena's slot-remap path at prefill width (the mechanism r258 gave the identity tier): admit the
used-and-missing experts into slots, publish the slot-remapped ids for the whole `n_used x n_tok` routing, and let
`MUL_MAT_ID` read the arena.  Per pass that is 0 copies + the miss fills.

Arm: `MOE_EXPERT_PREFILL_IN_PLACE` (existing kill switch; `=0` is the r265 copy path, byte-identical — verified).

## The placement ladder (the two decisions, as taken)

1. an **empty** slot — free room, nothing disturbed;
2. one of this table's **landing** slots — the coldest `slots/16` seeded slots (a *soft* reserve: seeded like any
   other, so a cold start pays no extra read and decode may use them; prefill may reclaim them);
3. a resident **this pass does not route**, coldest first by the **decode heat prior** (`heat_rank_vec`), then the
   LFRU `count`/`last`.

A slot this pass **routes** is never reclaimed, which is also the arm's hard bound: serving in place needs
`|used| <= slots`; a table that cannot is declined WHOLE (no partial admission) and keeps the scheduler's copy.
A landing occupant decode does reach is promoted at the hit (`slot_land` cleared) and is an ordinary resident from
then on; prefill admissions are recorded with `count=0, last=0` so the LFRU gives them up first.

Why rung 3 is needed at all: misses per pass scale as `|used| x (1 - resident/n_experts)`, so on a streaming arena a
landing reserve alone cannot cover a pass.  Measured (`-lme 90`, 200 of 256 slots): reserve-only declined 57 of 147
tables and was a net LOSS (-2 % prefill), while the ladder served all 147.

## Measured (halo: Ryzen AI MAX+ 395, gfx1151, ROCm 10.1.0, ONE GPU, `iommu=off`; heat prior frozen)

**Qwen3.5-122B-A10B Q8_0 — the model that does NOT fit (133 GiB of weights, 123 GiB RAM), `-lme 90`**
(200 of 256 slots per trunk table = 78 % resident).  Text identical in every arm (`58fba7e8b027` small arm,
`00a1029e9adf` wide arm):

| arm | prompt | copy path (r265) | in place | delta |
|---|---|---:|---:|---:|
| prompt eval | 34 tok, 3 paired runs (median) | 13.3 t/s | **13.6 t/s** | parity |
| prompt eval | 250 tok, 3 paired runs | 50.8 t/s | **54.0 t/s** | **+6.3 %** |
| generation | 250 tok + 64 decode | 10.9 t/s | **11.45 t/s** | **+5 %** |
| generation (`draft-mtp`, `-n 512`) | 250 tok, 2 paired runs | 10.55 t/s | **10.75 t/s** | +1.9 % |
| draft acceptance | same | 0.74630 (353/473) | **0.74630 (353/473)** | identical |
| prompt eval | 24 576 tok, `-ub 4096` | 366.6 t/s | 366.6 t/s | inert (see below) |
| prompt eval | 24 576 tok, `-ub 1024` | 151.7 t/s | **161.7 t/s** | +6.6 % |
| arena D2D (34-tok prompt) | one pass | 40 535 MiB | **0** | removed |
| read from storage | one pass | 5 231 MiB | 5 173 MiB | same set |

**Qwen3.5-122B-A10B Q5_K_XL (81 GiB expert set, fits RAM), `-lme 75`** (241 of 256 = 94 %):

| arm | prompt | copy path | in place | delta |
|---|---|---:|---:|---:|
| prompt eval | 34 tok, 3 paired runs (median) | 25.0 t/s | **32.9 t/s** | **+32 %** |
| prompt eval | 250 tok, 3 paired runs | 123.2 t/s | **158.3 t/s** | **+28.5 %** |
| generation | 250 tok + 64 decode | 30.1 t/s | 30.4 t/s | parity |
| prompt eval | 24 576 tok, `-ub 4096` | 580.3 t/s | **606.7 t/s** | +4.5 % |
| arena D2D (34-tok prompt) | one pass | 30 251 MiB | **0** | removed |

**The wide band is conditional, and the condition is structural.**  A 4 096-token ubatch routes all 256 experts of
a trunk table, so a table can only be served in place when the arena holds ≥ that many slots: at `-lme 90`
(200 slots) the Q8 arm declines 123 of 144 tables and is exactly inert (3 paired runs: 366.6 vs 366.5 / 366.8 vs
368.0 / 369.5 vs 366.6 t/s); at `-lme 75` (241 slots) the Q5 arm serves most calls (+4.5 %); at `-ub 1024` both
serve partially and gain.  Nothing regresses in the inert case — the arm declines before anything is admitted.

## Decode: does admitting on prefill cost the FOLLOW-UP decode?  (offline replay)

`scripts/moe-trace-replay.py --prefill-admit {off,free,landing,ladder}` (new model of this ladder, driving script
`archive/work/prefill-in-place-arena/admission-sim.sh`) over a real trace: Q8_0 `-lme 90`, 538-token `code-python`
prompt, 200 decode tokens, heat prior from the run's own frozen sidecar.  Decode misses over the 200-token
follow-up (11559 in the copy arm):

| prompt | reserve only (never evict) | **ladder (delivered)** | ladder, no reserve |
|---|---:|---:|---:|
| 200 tok | -333 (99 tables declined) | **-1101** | -1116 |
| 300 tok | -156 (120 declined) | **-954** | -945 |
| 400 tok | -27 (138 declined) | **-951** | -951 |
| 538 tok | -36 (138 declined) | **-1008** | -1008 |

Decode `h` 0.9515 -> 0.9534..0.9561, and on the FIRST 16 decode passes -72..-87 misses.  Admitting on prefill
does not disturb the continuation — it is a prefetch of it (a prompt's routing predicts its own continuation), and
the reclaim rung only ever gives up the coldest *non-routed* expert.  The replay's absolute `h` reads 1.2 % under
the run's logged 0.9632; judge the deltas.

**Two-slot server interference** (the case the replay cannot model — `scripts/gate-prefill-server-slots.sh`, new):
request A decodes while request B's 250-token PREFILL lands mid-decode, Q8_0 `-lme 90`, `-np 2`:

| | copy path | in place |
|---|---:|---:|
| A decode (whole request) | 3.78 t/s | **4.03 t/s** |
| A decode inside B's prefill window | 0.52 t/s | **0.58 t/s** |
| A longest inter-token gap | 623 ms | **414 ms** |
| A text | `3c2245a89587` | identical |
| B latency / text | 17.2 s | 15.6 s, identical |

## The defect the logits gate caught (recorded so it is not re-introduced)

The first cut of the ladder checked "never reclaim a slot this pass ROUTES" for rung 3 only.  Rung 2 (the landing
reserve) could therefore recycle a landing slot whose occupant this pass routed, mid-pass.
`gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0" --extra "-lme 16 --fit off"` measured **mean KLD 0.1157
/ same-top-p 87.3 %** (PPL 5.82 -> 6.30) on 35B-A3B while every greedy-text arm stayed byte-identical — the exact
class that gate exists for, and the reason a text hash is not a correctness signal.  With the rule applied to every
rung: **KLD -0.000000 / same-top-p 99.990 %**.

## Memory

The arena is byte-identical with and without the mechanism (`arena=78239.3 MiB` on the Q5 config and
`94248.0 MiB` on the Q8 config in BOTH arms), because the reserve is a *designation*, not a partition.  The real
cost is the remap staging, which grows from the decode-band `n_experts x 8` (8 KiB/table) to `n_used x -ub`
(128 KiB/table at `-ub 4096`): **+34.4 MiB** over 147 tables (host staging + device, ~9 MiB at `-ub 2048`), plus
`slot_land` at 1 byte per arena slot (**35 KiB**).  Per-call transients are ~1.5 KiB.

## What did NOT pay (measured, not adopted)

* **`preadv` / direct read straight into the arena slot** (P2's shape, no ring, no H2D): 601 vs 603 ms of hook
  time for the same bytes and the same text — a wash, and it needs a `cudaStreamSynchronize` per table for
  ordering.  The coalesced ring engine serves both prefill arms and is where the `O_DIRECT` offset round-out lives.
* **P2a (`O_DIRECT` span alignment) is FIXED** in that engine: a direct window is now read padded (`aoff`/`skip`/
  `blen`, `stage_span_fill`'s shape), so the `dio` arm no longer silently falls back to the per-expert source.
* **P4 (the eager seed) is untouched** and remains `TODO` #318: the 30 s / 78 GiB sizing fill is the small
  prompt's fixed cost and is independent of this arm.

## Gates

| gate | result |
|---|---|
| `gate-uma-experts.sh --quick --no-ram` (+ new `inplace` arm at `-lme 16`, new `pfcp` arm = r265 staging) | **36 checks, 0 failures**; `inplace`: 108 calls, 1603 admissions (829 landing / 774 reclaim), D2D 1194 vs 3358 MiB, read 971.7 of 11 446.2 MiB used, 9 declines named |
| `gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0" --extra "-lme 16 --fit off"` (35B-A3B Q4_K_M) | **PASS**: mean KLD -0.000000, same-top-p 99.990 %, journal clean |
| `gate-prefill-server-slots.sh` (NEW) | **PASS**: A identical + not slower, admission live (126 calls x 78 tables), no faults |
| MTP (`draft-mtp`, 250-tok prompt, `-n 512`, Q8 `-lme 90`) | acceptance **0.74630 identical**, generation +1.9 % |
| `check-gpu-faults.sh` on every arm's log | clean |
| `gate-fusion-alias-copy.sh` | **FAILS IDENTICALLY on the unmodified release build** (`~/llama.cpp/build-rocm`, r265): its repro and production arms do not reach the shape on gfx1151.  Pre-existing and box-specific, NOT caused by this change — reported, not skipped |
