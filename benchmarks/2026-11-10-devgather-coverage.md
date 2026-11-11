# The pool prefill gather is a COVERAGE crossover, not a default that needs moving (`TODO` #136, retired r256)

**Verdict: `TODO` #136 retires.**  Its stated cell (35B-A3B UD-Q4_K_M, `-ncmoe 40`, 3 devices, `-sm tensor`,
`--host-experts pool` + `MOE_HOST_POOL_FRAC=50`) **cannot be run on the current tree at all** — `#135`'s r250
guard refuses the load (that shape gets a zero per-device share of `ffn_down_exps` axis 0) — and its command
line no longer selects a half-covered pool: since r210 `--host-experts pool` **is** `-lme auto`, which on a box
whose host set fits mirrors the whole set and does not consult `MOE_HOST_POOL_FRAC` at all (measured below:
every table 256/256 slots at FRAC 30, 50 **and** 95, with identical throughput).  Its 33 % delta does not
reproduce even with `GGML_TENSOR_SPLIT_ALLOW_DEGENERATE=1` (both arms 242-393 t/s, overlapping).

What is real, and what this record replaces it with: the gather's default is right in **every** band the
delivery validates and on every box where the mirror applies (**+8.7 % to +40 %, and +2-3x on the 3-device
tensor split**), and it costs **about -10 %** only in the **sub-complement band** (`-lme N` well below the host
set), where a census shows the *reason*: the uncovered experts are copied one at a time from the
bounce/pool-fill path (191 279 of them on one prefill) while the `=0` arm amortises the same bytes into a
whole-table copy.  So the default does not move, and the one engineering candidate left is a
**coverage-gated gather** (re-entry condition in `TODO.md`), not a default flip.

**Rig.**  `soar`, gfx1201 x3 (R9700), ROCm 10.1.0, `~/bin/build-llama-rocm`, build `b12221-1606bdaff`
= the r254 canonical tip, so every number here is the shipped delivery.  Model
`Qwen3.6-35B-A3B-Q4_K_M` (Q8_0 where stated), `-ncmoe 40`, `-ngl 99 -t 12`,
`prompts/prose-rdna-boosts.txt` (16 074 B, sha256 `fabdec65…`), `-n 1 --seed 42 --temp 0`, no `-lv`
(the rate comes from the `[ Prompt: … ]` footer).  `MOE_EXPERT_CACHE_MIB=2048` per device throughout (so the
arena is 2 GiB/device, i.e. 4 GiB on the pair — this matters for the coverage estimate).  Every cell is two
reps per arm in `A B B A` order, which is what the box's phase alternation requires (`ENVIRONMENT.md`,
"judge a transport with A B B A"); the two numbers are the two reps.

## 1. The #136 cell is refused, and its delta does not reproduce

Without the escape hatch the load fails:

```
llama_model_load: error loading model: -sm tensor: 120 of 120 host-resident expert tables get a ZERO
per-device share of their split axis (first: blk.0.ffn_down_exps.weight (axis 0: 512 columns over
3 devices, granularity 256) -> device 0 holds 0).  This split cannot work: …
```

With `GGML_TENSOR_SPLIT_ALLOW_DEGENERATE=1` (3 devices, `-sm tensor`, FRAC 50), the arm runs again and the
`DEGENERATE TENSOR SPLIT` line confirms the mirrored table:

| arm | r209-era record | r254, rep 1 | r254, rep 2 |
|---|---:|---:|---:|
| gather default | 282.2 / 282.7 | **303.5** | **362.6** |
| `GGML_SCHED_DEVGATHER=0` | 422.2 / 421.4 | **242.3** | **393.2** |

The two arms overlap and the ordering even inverts between reps, so the recorded -33 % does not reproduce;
what is left is the degenerate shape's own cost, in both arms, plus this box's phase spread.  Nothing here is
a supported configuration, so this is where #136's original measurement stops being a delivery question.

## 2. `MOE_HOST_POOL_FRAC` is inert in the mirror band (why the r209-era reading no longer holds)

`--host-experts pool` is now a deprecated alias for `-lme auto` (`common/arg.cpp`), and `auto` spends the
**whole** host expert set on the pool when the set fits `MOE_HOST_LIMIT_FRAC` (80 %) of `MemAvailable`
(~62 GiB here against an 18.6 GiB host set).  The pool is then a mirror, and the percentage knob is not read:

| arm (2 devices, `-sm tensor`, `--host-experts pool`) | prefill t/s | census |
|---|---:|---|
| `MOE_HOST_POOL_FRAC=30` | 1233.6 / 1228.1 | `120 × 256/256 slots` |
| `MOE_HOST_POOL_FRAC=50` | 1240.9 / 1233.7 | `30720 slots, 18600.0 MiB, h=1.0000` |
| `MOE_HOST_POOL_FRAC=95` | 1234.5 / 1238.2 | (same layout; not re-censused) |

Identical to 0.6 %, i.e. the knob #136 varied inside its own command line is a no-op in this band.  (It is
*not* inert in the not-fitting band, which is the band it was written for — see `ENVIRONMENT.md`.)

## 3. Where the gather's default stands, measured

`GGML_SCHED_DEVGATHER` unset (the shipped default: a pool-backed table's prefill op is gathered) against `=0`.
Every arm below is the mirror/full-coverage layout of §2 unless the budget column says otherwise.

| layout | model | budget | gather default | `=0` | delta |
|---|---|---|---:|---:|---:|
| 1 GPU | Q4_K_M | mirror | **857.4 / 858.0** | 612.1 / 611.3 | **+40.1 %** |
| 2 GPU `-sm tensor` (`0,2`) | Q4_K_M | mirror | **1240.9 / 1233.7** | 986.2 / 989.6 | **+25.4 %** |
| 2 GPU `-sm tensor`, `-ub 4096` | Q4_K_M | mirror | **2679.7 / 2681.5** | 2465.6 / 2466.2 | **+8.7 %** |
| 3 GPU `-sm layer` | Q4_K_M | mirror | **878.5 / 877.9** | 637.3 / 636.8 | **+37.8 %** |
| 3 GPU `-sm tensor` | **Q8_0** | mirror | **563.8 / 570.1** | 172.2 / 259.9 | **+2.2-2.3x** |
| 3 GPU `-sm layer` | Q4_K_M | `-lme 9` (≈70 % covered) | **665.0 / 663.2** | 571.7 / 570.6 | **+16.3 %** |

The wide-ubatch cell is the one r234 warned about ("from `-ub` 5120 up the copy-engine arm was 10-18 %
ahead"): at `-ub 4096` the default still wins, and the arm r234 measured against was `=0` **plus**
`MOE_HOST_POOL_STAGE_BATCH=0`, not `=0` alone.

## 4. The coverage axis: the crossover is between ~70 % and ~86 %

Same 2-device `-sm tensor` face, an **explicit** `-lme N` budget so the pool really is sub-complement
(estimated coverage = (pool GiB + arena 4 GiB) / 18.6 GiB host set):

| budget | ≈ covered | gather default | `=0` | delta |
|---|---|---:|---:|---:|
| `-lme 4` | 43 % | 456.1 / 457.3 | **507.7 / 507.7** | **-10.1 %** |
| `-lme 9` | 70 % | 696.1 / 696.7 | **762.6 / 764.9** | **-8.8 %** |
| `-lme 12` | 86 % | **902.0 / 913.7** | 827.7 / 831.9 | +9.0 % |
| `-lme 14` | 97 % | **1184.8 / 1186.6** | 951.4 / 951.8 | +24.5 % |
| `-lme 17` | 100 % | **1220.6 / 1222.7** | 975.8 / 979.8 | +25.1 % |
| mirror (`auto`) | 100 % | **1240.9 / 1233.7** | 986.2 / 989.6 | +25.4 % |

Two things to read off it.  (a) The loss is real but narrow: about -9 to -10 % in the sub-complement band,
against +8.7 to +2.3x everywhere the delivery validates.  (b) **That band is already 40-60 % below the
mirror** (456-762 t/s against 986-1241), so the transport is not the lever there — the tier's coverage is
(`#141` is the decode side of the same band).

## 5. The mechanism, from the census (`-lme 9`, 2 devices, `-sm tensor`, gather default, `-lv 4`)

```
pool fills: 0 storage reads 0 ms, 23970 mapping memcpys 1045 ms, 14.15 GiB total (0.6 MiB/fill, 14491 MiB/s effective)
device gather: 2574 calls (2574 pool-backed), 191279 routed experts copied into the destination from the
               bounce/pool-fill path (neither tier)
host pool (L2): 120 tensors, 15120 slots, 9154.7 MiB pinned; h=0.0348 (7127/204696), fills=23970 evictions=0
expert access: hits=14266 fills=854 colds=0 (hit=94.35% fill=5.65% cold=0.00%)
```

Against the same binary on the mirror (default, `-lv 4`):

```
device gather: 2574 calls (2574 pool-backed), 0 routed experts copied into the destination from the bounce/pool-fill path (neither tier)
host pool (L2): 120 tensors, 30720 slots, 18600.0 MiB pinned; h=1.0000 (13867/13867), fills=30720 evictions=0
```

So the gather is issued identically (2574 calls in both) and the whole difference is the **uncovered fill
path**: 191 279 experts go through the bounce/pool-fill route one at a time, where the `=0` arm gives the same
bytes to one whole-table copy.  That is the multi-device pageable/mapping-fill family the `push` campaign
recorded (`archive/work/host-expert-eager-push/P3-MULTIGPU-RESULTS.md` §3a: several devices' fills issued
inside one op), not a defect in the gather's transport choice.  It is also why the loss is
**tensor-split-only** (the layer split at the same `-lme 9` is +16.3 %, §3).

## 6. Purity (a transport switch must not change a token)

`-lme 9`, 2 devices, `-sm tensor`, `-n 24 --seed 42 --temp 0`, no `-lv`, hashed with
`scripts/extract-generated.py`:

| arm | text |
|---|---|
| gather default | `110 chars sha=461ca8cd0e88` |
| `GGML_SCHED_DEVGATHER=0` | `110 chars sha=461ca8cd0e88` |

Identical, as r207 recorded for the whole switch (prefill logits KLD -0.000000 / same-top-p 100.000 % there).

## 7. What is NOT measured, and the one candidate change

- **No not-fitting-band box.**  Every arm here is on the 91 GiB primary box, where `auto` mirrors; the
  r112 field case (30 GiB RAM, 31.9 GiB host set) is where the sub-complement band is a *default* rather than
  an experiment, and it was not available.  Its pool lands at ~81 % of the set by `MOE_HOST_POOL_RAM_PCT`, i.e.
  above the measured `-lme 12` (86 %) cell but below the crossover's steep part — so the direction is known and
  the exact cell is not.
- **The crossover between 70 % and 86 % is bracketed, not located.**  A gate would need the edge measured on
  each not-fitting box.
- **The candidate change**, if the band is ever worth recovering: gate `sched_input_gatherable` on the table
  being **fully covered** by arena + pool — the concept already exists as the eager take-over's covering-pool
  test (`INVARIANTS.md` r221).  It is a coverage predicate, not a width or a model list, and its fingerprint in
  a census is the line above (`gather … N routed experts … from the bounce/pool-fill path`): N > 0 means the
  cell is in the losing band.  Not implemented: it would need the not-fitting box, `gate-host-pool.sh`, the
  prefill-logit gate and `#273`'s transport question settled first.

## 8. The decode side of the same band (`TODO` #141), re-verified while the rig was warm

`#136` and `#141` are the two faces of one configuration — a **sub-complement host-expert budget** — in different
bands: `#136` was the prefill transport, `#141` is the eager take-over declining at wide decode.  Both were
measured in this session, so the `#141` row is re-verified rather than carried:

`llama-batched-bench -npp 512 -ntg 128 -npl 1,8`, 2 devices, 35B-A3B Q4_K_M `-ncmoe 40`,
`MOE_EXPERT_CACHE_MIB=2048`, `-sm layer`, `-fa on`, `-t 12`:

| arm | pp (npl 8) | tg (npl 8) | tg (npl 1) |
|---|---:|---:|---:|
| `-lme auto` (mirror, covering) | 1261.4 | **176.6** | 59.0 |
| **`-lme 8` (≈65 % covered, sub-complement)** | 825.0 | **48.0** | 42.9 |
| `MOE_EXPERT_CACHE_MIB=0` (cache off) | 745.5 | 109.6 | 37.1 |

The sub-complement budget is **below the cache being switched off** (48.0 against 109.6) and ~3.7x below a
covering pool — the r211 shape reproduces exactly.  Its absolutes do not, and are not comparable: the r211 record
reads 50.4 / 270.3 / 129.2 on a different rig (that table predates `-fa on` and this tree), which is why the
`TODO` row now carries both the shape and its own rig line.

One consequence for `#141`'s candidate list: the prefill's `191 279` neither-tier experts (§5) are the *same*
population the decode take-over declines, so candidate (b) there — a bounce ring legal as a cold source *inside*
the take-over — would serve both bands, and that is now a two-datum candidate rather than a decode-only one.
