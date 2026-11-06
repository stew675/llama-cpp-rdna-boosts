# HANDOVER — `host-expert-split`, state at 2026-11-06 (end of the W4 session)

**Read this first, then [`PLAN.md`](PLAN.md) for the standing plan, [`W4-SPAN-READS.md`](W4-SPAN-READS.md) for
the disk-I/O work this session finished, and [`W1-SINK-TAX.md`](W1-SINK-TAX.md) for the copy cost.**  This file
is the current state: what is decided, what is built, what is proven, what is owed, and the traps that cost
time so they are not paid for twice.

Nothing from this campaign is in `patches/`.  The delivery repo is untouched except for `wip/` docs and
`scripts/` tooling; `release.json` is unchanged.

---

## 1. The campaign in one paragraph

The r223 pool (`--host-experts` / `-lme`) holds a **pinned per-expert cache** of the host MoE set and serves a
miss from that tier instead of from the model mapping.  It had two weaknesses: its content was chosen by a heat
prior that is *worse than chance* for decode routing, and it had **no write-back**, so an arena eviction simply
discarded a recently-hot expert.  Both are now implemented, default-on and measured.  This session finished the
last open item — **prefill read granularity** (W4, +81 % prefill at `-lme 12`, +14 % at 36) — and then found
that the campaign's headline number **does not survive a frozen, real-world heat prior**: on one, the shipped
pair is **+7.8 % single-stream** (24.3 → 26.2 t/s at `-lme 36`), not +21.7 %, because the +21.7 % was measured
against a prior that had been learning from the benchmark prompts themselves (§4.2).

## 2. Decisions

| # | decision | basis |
|---|---|---|
| D1 | ~~Ranking ships ON~~ **REOPENED — recommendation is `combined` by default** | its `+11.1 %` was prior leakage; on a frozen real-world prior it is **+1.2 %** single-stream while it still costs the batched path (npl 4 prefill -11.5 %).  Maintainer's call |
| D2 | **Demote ships ON** (`MOE_HOST_POOL_SINK=1`, `0` = kill switch) | **+5.3 % alone, +7.8 % combined** on the frozen prior; never harmful in any cell measured.  Still the campaign's product |
| D3 | **`npl 4`/`npl 8` cost is an accepted limitation**, not a blocker | maintainer's earlier call; re-quantified in §4.3 |
| D4 | **Headroom `K` is a fraction of slots** (`_PCT=4`, floor 2, cap slots/8) | at `-lme 36` a table has ~393 slots, so the measured optimum K=16 is 4.1 % of *slots*; a flat 16 would be 6.25 % on a 256-expert table |
| D5 | **LFRU period stays 32** | plateau 32-64 with 32 at its top edge |
| D6 | **Never default a workaround for a bad prior**: the demote is the fix, the ranking is a hint | the demote measures, the ranking predicts |
| D7 | **`invert` mode must not ship** (diagnostic; remove before promotion) | it exists to prove the counter direction and was never a product option |
| D8 | ~~Prefill read granularity is the next target~~ **DONE — W4** | +81 % at `-lme 12`, +14 % at 36, gen unchanged, text identical |
| **D9** | **The heat prior is a MEASUREMENT INPUT: frozen, read-only, sha256 in the record** | a writable prior learns the benchmark prompts (§5) |

## 3. Code state

* Fork: `~/llama.cpp`, branch **`p1-pool-rank-r223`**.  Working tree carries this session's W4 + heat-save fix
  (uncommitted at the time of writing; commit before handing over).
* Build: `~/bin/build-llama-rocm` → `build-rocm` (ROCm 10.1, gfx1201).  The built binary is this branch.
* New in this session (on top of the r223 + W2 chain listed in the previous handover):
  | what |
  |---|
  | `stage_span_fill()` — the coalesced prefill read, wired into `moe_cache_gather_host` and `moe_cache_stage_table` (`W4-SPAN-READS.md`) |
  | `MOE_HOST_POOL_STAGE_SPAN` (kill switch, default on) + `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` (16); `SPAN_MIN=2` is a code constant |
  | pinned span ring (`span_buf_*`), grown on demand to 32 entries/device, leaked like the pool |
  | `g_stage_span_{reads,bytes,experts,fallback,read_us,wait_us}` + `g_bounce_read_{calls,us,bytes}` counters and two report lines |
  | **heat-save fix**: `moe_cache_forget_all` flushes the heat tallies *before* dropping the tables (the exit-time save walked an empty `g_tables` and wrote a header-only sidecar, truncating a curated map — §5.3) |

  Environment variables in this build (all read in `moe-expert-cache.cu`):

  | var | default | class |
  |---|---|---|
  | `MOE_HOST_POOL_RANK` | `decode` (**recommendation: `combined`**, D1) | kill switch |
  | `MOE_HOST_POOL_RANK_FLOOR` | 50 | tuning (only for `floor`) |
  | `MOE_HOST_POOL_SINK` | on | kill switch (`0`) |
  | `MOE_HOST_POOL_SINK_SPARE` | -1 (auto) | absolute override |
  | `MOE_HOST_POOL_SINK_SPARE_PCT` | 4 | tuning (% of a table's slots) |
  | `MOE_HOST_POOL_SINK_SPARE_MIN` | 2 | tuning |
  | `MOE_HOST_POOL_SINK_TAX` | 0 | **diagnostic — must not ship** |
  | `MOE_HOST_POOL_STAGE_SPAN` | on | kill switch (`0`) |
  | `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` | 16 | tuning |
  | `MOE_EXPERT_CACHE_PERIOD` | 32 (pre-existing) | tuning, D5 |

## 4. What is proven, and on what

All single-user numbers: `llama-cli`, `prompts/prose-rdna-boosts.txt` (5 298 tok), `--seed 42 --temp 0`,
`-ncmoe 48 -ngl 99 -ot 'per_layer_token_embd.*=CPU' -lm mmap -lzm off -fa on -c 8192`,
`MOE_HOST_POOL_DIO=1`, `MOE_EXPERT_CACHE_MIB=4096`, `HIP_VISIBLE_DEVICES=0`, one R9700, `gpu-watch.sh` around
every run.  Model: `Qwen3.8-Flash-Next/GSQ-RCO-IQ3_S` (144 tables × 512 experts, 46.84 GiB host set).

**The prior used for every number below is the frozen real-world map** restored from `radix`
(`/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/*.moe-heatmap`, 601 376 B, 144 tables, mode 444,
sha256 `6195152358e132a5…`).  A local backup lives at
`~/.cache/llama-moe-heatmap/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.canonical-261105.moe-heatmap`.

### 4.1 Prefill read granularity (W4) — the session's win

`-ub 6144`, `-n 64`, SPAN=0 vs SPAN=1, same binary:

| `-lme` | prefill SPAN=0 | SPAN=1 | Δ prefill | gen |
|---|---|---|---|---|
| 12 | 94.2 | **170.7** | **+81 %** | 9.9 / 9.9 |
| 18 | 115.6 | **181.6** | **+57 %** | 12.8 / 12.9 |
| 36 | 330.9 (3 reps ±0.9) | **375.5-378.8** (3 reps ±0.3) | **+13.5 … +14.6 %** | 26.2 / 26.2 |

Text `sha=880c932888f8` identical on both arms at `-lme 12`; counters, read rates and the mechanism are in
[`W4-SPAN-READS.md`](W4-SPAN-READS.md).

### 4.2 The headline, on the frozen prior (`arm-decay`, `-lme 36`, `-n 64`, 2 reps each)

| arm | gen t/s | vs control |
|---|---|---|
| `combined`, no demote (**= r223**) | 24.1 / 24.3 | — |
| `decode`, no demote (ranking only) | 24.6 | **+1.2 %** |
| `combined` + demote (demote only) | 25.6 | **+5.3 %** |
| `decode` + demote (**shipping**) | 26.1 / 26.2 | **+7.8 %** |

So **the demote carries the campaign** and the ranking is a rounding error on a well-conditioned prior.  The
previous record's `22.6 → 27.5 (+21.7 %)` does not reproduce, and the reason is measured (§5.1): the prior it
was measured against had been rewritten by that session's own short benchmark runs, so the pool had been
ranked *for the test prompt*.  The policy counters are unchanged across the two records (10 203 demoted,
6-8 % reuse, 326 no-free-slot declines), i.e. the mechanism is the same and only the *input* differs.

### 4.3 The batched gate on the frozen prior (`-lme 36`, `-lzm auto`, npl 1/4/8, 1 rep)

| arm | npl 1 prefill / TG | npl 4 prefill / TG | npl 8 total time |
|---|---|---|---|
| `decode` (shipping) | 20.25 / 36.73 | 1144.69 / 87.32 | 45.3 s |
| `combined` + `SINK=0` (r223) | 20.23 / 36.30 | 1292.88 / 87.07 | 35.9 s |

`npl 1` is unchanged, `npl 4` prefill is **-11.5 %**, `npl 8` total is **-21 %** (MemAvailable min 40 GiB, so
this is not the memory-collapse trap).  One rep each — enough to keep D1 open, not enough to size the cost.

### 4.4 Economics (unchanged)

`c_pcie` = 0.0140 ms/MiB for a 1-D D2H at `-lme 36` (9.6 µs per 679 762 B, ~2/3 hidden behind compute);
demote reuse 6.3-7.9 % against a ~3 % break-even on this box's NVMe.

## 5. Measurement findings that change how this campaign is measured

### 5.1 The heat prior is adaptive: a writable one learns the benchmark

`<model>.moe-heatmap` is rewritten at every run, and it decides the pool's layout.  If the benchmark prompt is
also the workload, the prior converges on **that prompt** and the measurement stops being a measurement of the
delivery.  Measured at `-lme 36`, same binary, `-n 64`:

| prior | gen | prefill |
|---|---|---|
| frozen real-world (canonical) | 26.2 | 375.5 |
| this session's drifted map (many 5 298-token prose runs) | 27.9 | 378.9 |
| no prior at all | 23.0 | 416 |

**Protocol from here on (D9):** every arm runs `MOE_EXPERT_HEATMAP=load` (the arms now default to it —
`HEATMAP=1` overrides when regenerating) **and** the sidecar itself is `chmod 444`, which the code treats as the
freeze signal (`heat_map_open_locked`: "read-only; loading and seeding from it, not updating it", and the save
path re-checks it, so it can even be frozen mid-run).  Verified end-to-end: with the map at 444 and
`MOE_EXPERT_HEATMAP=1` the run refuses to write and the sha256 is unchanged.  Quote the map's sha256 with any
number that depends on it.

### 5.2 Corrections to the previous record's attribution

* The prefill disk reads are **not** in `moe_cache_stage_table` (that path never runs on this config); they are
  `pool_prepopulate_locked` and the `moe_cache_gather_host` pre-pass (`W4-SPAN-READS.md` §1).
* Every ggml `INFO` counter line needs **`-lv 4`** (`INFO` maps to `LOG_LEVEL_TRACE`); the arms ran `-lv 3`, so
  all of them were invisible.  That is how the wrong function got blamed.  The arms now honour `LV=`.
* `arm-coherence.sh` reported **VACUOUS on every run while the pool was engaged** (this CLI logs nothing at its
  default threshold, and the arm grepped the pool `WARN` from a merged `2>&1` log).  Fixed by `-lv 3` plus a
  **stream split** (`> OUT 2> LOG`): the text stays byte-clean on stdout, the report lands in the log.

### 5.3 A real bug: a writable heat map was truncated to a stub

`moe_cache_report`'s exit-time save walks `g_tables`, but the model teardown (`moe_cache_forget_all`) drops them
first — so a run short enough that the 60 s debounce never fired wrote a **header-only 32-byte sidecar**, i.e. a
valid empty map that silently destroys a curated prior (it did exactly that to this campaign's map before the
freeze was adopted).  Fixed by flushing the tallies in `forget_all`, before the tables go; verified: a 45 s run
now writes 601 376 B / 144 tables instead of 32 B.  **This is a delivery-relevant fix (block 09) — fold it with
the campaign patch, do not leave it in `wip/`.**

## 6. What is NOT proven

* **One model, one prompt, one device, one box.**  Nothing here says how it behaves on another model, a
  different prompt population, or multi-device (`-sm tensor` declines the demote by design).
* **The previous session's `-lme` sweep** (the "W3 curve") was measured with a drifting prior and needs one
  re-run with the frozen one before it goes in a release table (only `-lme` 12/18/36 are re-measured).
* **The control curve** (`combined` + `SINK=0` across `-lme`) is still missing for the release table.
* **`moe_cache_stage_table`'s span path is unexercised** on this model: first proof is owed by the 35B-A3B
  prefill-logit gate (that config reaches the scheduler staging path).
* `allow_empty_fill` A/B, `floor` mode sweep, the demote under `-sm layer`, Windows portability: unchanged.

## 7. Gates owed before promotion (all mandatory per `MANIFESTS.md`)

| gate | state | note |
|---|---|---|
| same-seed greedy text | **PASS** `880c932888f8` | re-run per cell at promotion; arms fixed (§5.2) |
| `scripts/gate-prefill-logits.sh` | **BLOCKED** | needs a base recorded from the **delivery tip** build (recording from this branch is vacuous).  Also the first proof that the `stage_table` span path is right |
| MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) | **NOT RUN** | draft head: `/llm/models/Qwen3.8/Flash-Next/GSQ-RCO-IQ3_S/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` |
| `llama-batched-bench -npl 1,4,8` | run, gate only | §4.3; re-run at promotion with 3 reps now that D1 is open |
| `scripts/gate-per-block-build.sh` | **NOT RUN** | needs the change folded into blocks 09/12 first |
| `scripts/gate-host-pool.sh` | **NOT RUN / MUST EXTEND** | the demote adds a fill-like path; the "a failed demote must DECLINE, never take the op over" rule (#274) needs an arm |
| `scripts/check-gpu-faults.sh` | **clean** on every arm this session | |
| `invert` removal | **TODO** | D7 |
| heat-save fix | **code ready, ungated** | §5.3 — needs the per-block build + a text/coherence run at promotion |

## 8. Negative results — do not repeat these

1. **A concurrent 4-worker demand-fill pool changed nothing** and dispatched zero jobs (sub-mirror pool is
   saturated, so no slot was ever free).  Target the prefill stage path instead — done by W4.
2. **`madvise(MADV_WILLNEED)` hints** changed nothing (and with `MOE_HOST_POOL_DIO=1` the page cache is
   bypassed, so it cannot).
3. **LRU eviction on a saturated demote is harmful** (18.7 → 12.4 t/s).  Only duplicate reclamation is safe.
4. **Reserving headroom by shrinking `slots` reserves nothing** — it must be a `keep_free` argument.
5. **A periodic conditional swap** is 4-5× less copy-efficient than on-demand.
6. **The warmer is the wrong instrument** for a constrained box.
7. **A fixed 2- or 4-deep span ring stalls** (5.0-5.6 s of a 16 s prefill): the ring must grow to the backlog.
8. **Coalescing single-expert runs loses** (12 % at `-lme 36` before `SPAN_MIN=2` + the growing ring).
9. **Reading whole 16-expert windows regardless of residency loses** at high pool coverage (4.3× the bytes to
   save 1.5× on the rate) — zero byte amplification is the rule.

## 9. Methodology traps that cost time

1. **`llama-batched-bench` is a GATE, not a decider** (prefill-dominated; the ranking's sign inverts).
2. **`-lzm off` for single-user decision arms; `-lzm auto` for the batched gate** (the PLE over-subscription
   collapse reads exactly like a policy regression).  `arm-batched.sh` now defaults to `auto`.
3. **Never hash a merged `-lv 2/3` log**; hash the split stdout (`arm-coherence.sh` does it correctly now).
4. **The pool is built at the first decode-band pass** — a 9-token prompt never engages it.
5. **`gpu-watch.sh` word-splits `--cmd`**: put env prefixes on the wrapper invocation, not inside `--cmd`.
6. **Trap the model process or it orphans** (`arm-decay.sh` has the TERM trap).
7. **Check MemAvailable before trusting any `npl 8` number.**
8. **Counters, not hand-divided columns.**
9. **`-lv 4` for any counter you intend to quote** (§5.2).
10. **Pin the heat prior** (§5.1) — an adaptive prior turns a benchmark into a mirror of itself.
11. **A 20-token generation under-reports gen by 13-15 %** (41.6 vs 35.8 ms/token): warm-up + the pool's
    first ~20 decoded tokens.  Quote gen at `-n 64` (`NGEN=64`) or say which harness it came from.

## 10. Harnesses and where the records live

| what | where |
|---|---|
| arm scripts (decay / prefill / batched / coherence) | `archive/work/host-expert-split/tools/` + its `README.md` (`LV=`, `NGEN=`, `SINK=`, `HEATMAP=`) |
| flight recorder + fault scan | `scripts/gpu-watch.sh`, `scripts/check-gpu-faults.sh` |
| offline policy model | `scripts/moe-trace-replay.py` |
| the evidence | [`W4-SPAN-READS.md`](W4-SPAN-READS.md), `STEP1-REPLAY.md`, `W1-SINK-TAX.md`, `P1-POOL-RANK.md` |
| design + PR #267 review | `DESIGN.md` |
| the standing plan | `PLAN.md` |
| dated narrative | `WORKLOG.md` 261105.014-019, 261106.001-011 |
| campaign index | `wip/CAMPAIGNS.md` |

## 11. Suggested order for the next session

1. **Commit the branch work** (`W4` + the heat-save fix) and update `tools/README.md` if not already done.
2. **D1 decision** (maintainer): `MOE_HOST_POOL_RANK` default `combined` or `decode`.  Inputs: §4.2 (+1.2 % vs
   +5.3 % decomposition), §4.3 (batched cost), and the fact that a consumer's own prior will differ from any
   prior measured here.
3. **Finish the release table**: the `-lme` sweep with the frozen prior (shipping and `combined`+`SINK=0`).
4. **Gates**: prefill-logit base from the delivery tip (also proves the `stage_table` span path on 35B-A3B),
   MTP four-axis, `gate-host-pool.sh` demote arm, `invert` removal, per-block build.
5. **Promotion**: fold into blocks 09 (cache, incl. the heat-save fix) and 12 (loader/pool policy), regenerate
   `patches/`, `validate-set.sh`, release ladder, `fork-refresh.sh`.
6. **Docs at promotion**: `ENVIRONMENT.md` (the eight variables above, `SINK_TAX` excluded), `INVARIANTS.md`
   r113 "no write-back" corrected in place **plus** the new facts (heat prior is an input; span reads), a
   `TODO.md` entry for the accepted batched limitation, `patches/README.md` per-block notes.
