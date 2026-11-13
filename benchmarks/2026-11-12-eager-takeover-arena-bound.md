# The eager take-over's completeness bound is ARENA-triggered, not host-budget-triggered (`TODO` #141, retired r274)

**Verdict: `TODO` #141 retires as an accepted limitation, and its statement was wrong in a way that matters.**
The row said "a genuine sub-complement host-expert budget declines the eager take-over at wide decode".  The
bound is real and reproduces exactly (tg **46.1 against 102.8** for the cache being off, i.e. **0.45x**), but a
sub-complement host budget **alone never reaches it**: at the delivery's default arena the same `-lme 8` arm is
the **fastest** arm measured here (**298.7 / 384.0 tg**, `0` declines).  What crosses the bound is an
**explicitly capped `MOE_EXPERT_CACHE_MIB`** whose per-table slot count is below the band the take-over must
represent, `n_used * n_tok` — and the band's need scales with the widest batch served
(`MOE_EXPERT_CACHE_MAX_TOK` = 16 here, so 8 x 16 = 128 slots/table at the top of the band).

So the remaining cell is "a VRAM cap and a RAM cap at the same time", i.e. asking too much of too little, on a
box whose pragmatic answers are the cache **off** or fewer host-resident experts (the streaming KV window/region
is how the memory comes back).  Coding a cold-source redesign for it is not worth the delivery's time; the
census below is the record a future operator (or campaign) needs to recognise the cell in one line.

**Rig.**  `soar`, 2 x R9700 / gfx1201 (`HIP_VISIBLE_DEVICES=0,2`, the x8 pair), ROCm 10.1.0, the canonical
`~/bin/build-llama-rocm` build of the **r273** tree (fork tip `6245ba844`, unmodified checkout).
Model `Qwen3.6-35B-A3B-UD-Q4_K_M` unless stated (`-ncmoe 40`, host expert set **18 662.0 MiB** over 120 tables),
`-ngl 99 -t 12 -sm layer -fa on`, `-lme 8` = an **8192 MiB** per-expert host pool = **43.8 % coverage**
(`112/256` slots, the RESOLVED line) so the pool is a genuine complement of the arena.  `llama-batched-bench`,
`-npp 512`, `-npl 1,8` (or `1,16`), `-ntg` as noted.  Every number is one arm; no parallel runs.

**Reading the table** (worth stating: it is the trap this record is about being read wrongly).  The columns are
`PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s`, where **`B` is `-npl` (parallel slots)**
and **`TG` is `-ntg` (generated tokens *per slot*, not per cell)**: `tools/batched-bench/batched-bench.cpp` decodes
`tg` steps and adds one token per sequence in each (`for i < tg { for j < pl { batch.add(...) } }`), so one cell is
`TG x B` decoded tokens and `S_TG t/s` is the **aggregate** rate across slots.  `-ntg 128 -npl 8` is therefore 128
batched decode steps / 1024 tokens, not "8 tokens" — read a `B` column as the generation length and every arm here
looks like a warm-up.  Raw logs are not kept in this repo (the commands below + the censuses quoted above are the
record); the arms were run from `/tmp` on 2026-11-12.

## 1. The cliff reproduces, and it is the ARENA that opens it

`-npl 8`, `-ntg 1024` (a long run with a converged heat map) and `-npl 16`, `-ntg 256`:

| `MOE_EXPERT_CACHE_MIB` | slots/table | tg npl 8 | tg npl 16 | `decline_all` (npl 8 / 16) |
|---|---:|---:|---:|---:|
| unset (auto) | 256 (clamped at `n_experts`) | **298.7** | **384.0** | **0 / 0** |
| 5120 | 134 | – | **289.9** | – / **0** |
| 3584 | 94 | 182.7 | – | 117 / – |
| 2560 | 67 | **134.2** | **66.2** | 117 / 30 837 |
| 2048 | 53 | **46.1** | – | 122 997 / – |
| 0 (cache off) | – | **102.8** | **149.0** | – |

Every `-npl 8` cell is `-ntg 1024`; every `-npl 16` cell is `-ntg 256`.  The short-arm readings agree with them
(`-ntg 128`: auto **299.6**, 3584 **181.3**, 2560 **143.3**, 2048 **46.4**, off **102.8**), which is the point of §3.

The census is unambiguous, and it is the instrument — **the text is identical in every arm** (this path computes
the same answer, slowly; same-seed greedy, `llama-cli -n 20 --seed 42 --temp 0`, no `-lv`: `627b2ebd7792`, 89 chars
at `MIB=2048` (declining), `auto` and `0` alike), so a rate is the only thing that moves here:

```
MOE_EXPERT_CACHE_MIB=2048, -lme 8:  device remap: 0 tables served, 30726 of 30726 offers declined
                                    takeover=15369 decline_all=15477   slots/table(sum)=3339 (53)
MOE_EXPERT_CACHE_MIB=auto, -lme 8:  takeover=0 decline_all=0             slots/table(sum)=16128 (256)
```

The bound itself is one line in `moe_cache_update_host` (block 09), and it is unchanged at r273:

```c
if (!table_cold_ok(t) && !pool_covers_table(t) && t.slots < (int) (n_used * n_tok)) {
    g_decline_all++;
    return false;
}
```

`table_cold_ok` is false for a pageable master, `pool_covers_table` is false at 43.8 % coverage, so the test is
`t.slots < 8 * n_tok`: **53 < 64** declines at `-npl 8`, and **67 < 128** declines at `-npl 16` — which is why the
2 560 MiB arm fixes the first and not the second.  The default arena sizes each table from free VRAM and clamps at
`n_experts` (256 here), so on this model the bound can only ever be crossed by an explicit cap.

## 2. What the candidates in the row cost, now measured

* **(a) "require the arena to hold `n_used * band` slots — size it, do not decline it."**  It works, and the price
  is the operator's cap: 2 560 MiB/device (67 slots) takes `-npl 8` from 46.1 to **134.2** but leaves `-npl 16` at
  66.2; the full band needs **5 120 MiB/device** (134 slots) and reads **289.9** at `-npl 16`.  A floor on an
  explicit `MOE_EXPERT_CACHE_MIB` would silently override the one knob an operator uses to reserve VRAM for KV —
  the opposite of the r273 rule that an explicit budget is authoritative.
* **(b) "a neither-tier expert served from a ring inside the take-over."**  The memory math is the same order: the
  ring must hold a pass's distinct neither-tier experts (the `-lme 9` census of
  `2026-11-10-devgather-coverage.md` §5 counted 191 279 of them across one prefill), so it does not dodge (a) — it only changes who owns the bytes.
  It is also the prefill gather's problem as much as decode's, i.e. a campaign, not a fix.
* **(c) "accept the variance and let the graph re-capture."**  Refuted elsewhere: a graph-less batched decode was
  measured at **137 -> 52 t/s** (`ggml-cuda.cu`, the capture guard's own comment).  The per-shape constant is the
  reason the bound exists.

## 3. Why a longer generation does not change it (and the heat map is not the variable)

The take-over's success is **constant per shape by construction** — that is what makes it safe inside a captured
graph (`INVARIANTS.md` r221 fact 5) — so a run cannot warm up out of the band:

| arm | ntg 128 | ntg 1024 | `h` at 1024 | heat coverage |
|---|---:|---:|---:|---:|
| `MIB=2048`, `-lme 8`, `-npl 8` | 46.40 | **46.13** | 0.8359 | 82.34 % of the routed mass |
| `MIB=0`, `-lme 8`, `-npl 8` | 102.83 | **102.82** | – | – |
| `MIB=2560`, `-lme 8`, `-npl 8` | 143.31 | **134.17** | 0.8853 | 87.71 % |
| `MIB=3584`, `-lme 8`, `-npl 8` | 181.27 | **182.69** | – | – |
| `MIB=auto`, `-lme 8`, `-npl 8` | 299.58 | **298.71** | – | – |

An 8x-longer generation with the map fully converged (82-88 %) lands within 0.5 % on the arms that are not in the
band and within 6 % on the 2 560 arm: no arm moves toward or away from the cliff.  `llama-batched-bench` drives
uniformly random token ids, which is the *no locality* shape — the same convention r256's `#136`/`#141` cells were
taken on, and the reason this record is a **bound/representability** measurement, not a locality one.

## 4. Cross-check on the field model (512 experts): the bound does not bite, a different decline does

`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, 2 devices, `-ncmoe 36`, `-lme 20` (**20 480 MiB of a 36 887.5 MiB set =
55.5 % coverage**, `284/512`), **default arena**, `-npp 512 -ntg 128 -npl 1,8`:

| device | tables | slots/table | band need at `-npl 8` | arena |
|---|---:|---:|---:|---:|
| 0 | 39 | 483 | 64 | 11 845.3 MiB |
| 1 | 69 | 471 | 64 | 22 262.1 MiB |

Global census on that run: `takeover=27 753 decline_all=0` — the eager take-over served every routed op — while
`device remap: 0 tables served, 27 651 of 27 651 declined`, and the bench read **28.0 tg** at `-npl 1` / **106.0**
at `-npl 8`.

So a sub-complement budget on the delivery's *big* field model, with an arena it did not cap, is nowhere near this
bound (470-483 slots against 64) — but it still declines the **device remap** (r221's cold-id representability,
worth ~+9.9 % where it applies).  That is a separate, already-recorded limit: **do not read a `device remap: ...
declined` line as this item**, and do not read this item's bound as that one.

## 5. Why it retires rather than getting code (the maintainer's call, r274)

Reaching the cell takes **both** knobs at once: an `-lme N` well below the host set *and* a
`MOE_EXPERT_CACHE_MIB` below `n_used * MOE_EXPERT_CACHE_MAX_TOK` slots/table.  The default arena cannot land
there, the pinned tier (`MOE_HOST_TIER=tensor` / `--host-experts pinned`) is exempt by construction
(`table_cold_ok`), and the pragmatic answers are cheaper than any cold-source redesign:

* `MOE_EXPERT_CACHE_MIB=0` — the cache off is **2.2x faster** than the declining arm (102.8 / 149.0), which is the
  whole reason this was worth recording;
* or `-lme auto` (the mirror, covering pool) / `--host-experts pinned`;
* or recover the VRAM the cap was protecting with block 15's streaming KV (`--kv-window`, `--kv-attendance-pool`),
  which is the intended answer to "I need VRAM for the KV cache".

A `WARN` naming the cell was considered and is deliberately **not** shipped: it is another code change to a
release gate ladder for a configuration the delivery does not validate, and the census line (`decline_all`, plus
`takeover=`) already identifies it for anyone who looks.  The re-entry condition is in `TODO.md`'s accepted
limitations: a real operator report on a supported configuration, or a campaign that reduces the band's need
(a cold source for a neither-tier expert) for its own reasons.

## 6. Reproduce

```bash
cd ~/llama.cpp    # r273 tip 6245ba844
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
for mib in 2048 2560 5120 "" 0; do
  HIP_VISIBLE_DEVICES=0,2 MOE_EXPERT_CACHE_MIB=$mib \
  ./build-rocm/bin/llama-batched-bench -m $M -ngl 99 -ncmoe 40 -lme 8 -sm layer -fa on -t 12 \
      -npp 512 -ntg 1024 -npl 1,8 -lv 4 2>&1 | \
      grep -E "^\| *[0-9]|device remap:|takeover=|device [01]: tables="
done
```

`-lv 4` is required for `decline_all` / `device remap` (they are `LLAMA_LOG_INFO`); the rate itself comes from the
bench table, so no `-lv` is needed for a rate-only A/B (`AGENTS.md`, the `-lv` table).  The text check is the same
command set with `llama-cli -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn` and
`scripts/extract-generated.py` (no `-lv` at all); it must print the same hash in every `MOE_EXPERT_CACHE_MIB` arm.
