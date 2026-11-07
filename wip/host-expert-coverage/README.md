# wip/host-expert-coverage - the tier must be a MIRROR, and the operator must be able to see it

**Status: OPEN - scoping note with a measured reproduction (2026-11-05).  Nothing implemented, nothing in
`patches/`, nothing applied to the fork.**

**What this is.**  While reproducing `briansp2020`'s 56 GB-container numbers
([PR #267 comment](https://github.com/stew675/llama-cpp-rdna-boosts/pull/267#issuecomment-6644149816)) on the
`bench-56gb` rig, three separate pieces of code were found to silently stop the host-expert pool from reaching
100 % coverage, and none of them says so at the default verbosity.  On a 59 GiB box that produces a tier that
pays the pinned cost without buying the property.  This campaign is about the **policy and the visibility**, not
about a new transport ([`host-expert-mirror/`](../../archive/work/host-expert-mirror/RESOLUTION.md), now archived) and not about the split (`host-expert-split/`).

## 1. What you inherit (do not re-derive)

* **The tier is a per-EXPERT pool at every budget.**  `-lme N` (block 09, r209-r212) is a byte budget spent on
  the pool; a whole-tensor pinned master is the escape hatch (`MOE_HOST_TIER=tensor`, `--host-experts pinned`).
  The reasoning and the anchors (86 % of the bytes ties on decode and the pool wins prefill; 100 % whole-tensor
  won decode before the pool-as-mirror shape existed) are in
  [`archive/work/host-expert-eager-push/P5-RESULTS.md`](../../archive/work/host-expert-eager-push/P5-RESULTS.md)
  §7 and in the loader's own comment (`src/llama-model-loader.cpp`, `plan_host_expert_placement`).
* **"Fits, so mirror" is `MOE_HOST_LIMIT_FRAC`.**  `ENVIRONMENT.md` has the full row: `auto` spends the whole set
  on the pool when it fits this percentage of `MemAvailable`, above it `auto` keeps the historical
  `MOE_HOST_POOL_FRAC` layout.  The r211 trap (in the not-fitting band this knob IS the budget and overrides
  `MOE_HOST_POOL_MIB`) is also there.  Read it before touching the arithmetic.
* **A miss is a bounce-ring read, and it is expensive.**  This campaign's numbers (below) put it at about
  125 MB/s on this model.  `INVARIANTS.md` already forbids reintroducing a demand-path eviction; that is the
  reason a partial pool is not a graceful degradation.
* **Start-time sizing is sensitive to `MemAvailable` at the FIRST pool build** (`INVARIANTS.md`, the r215 item 8:
  a warm page cache sizes a smaller pool).  Anything measured here must drop the page cache first, or the arm is
  not comparable.

## 2. Evidence (2026-11-05, `bench-56gb` rig, 56 GiB cgroup, MemAvailable 57.3 GiB)

Host expert set as the loader counts it: **47962.5 MiB = 46.84 GiB** (144 tensors x 512 experts).  So the budget
that buys a mirror is 47, and 52 (the value in the PR discussion) is not it.

| what happened | what the loader printed | coverage | cost |
|---|---|---|---|
| `-lme auto` (default) | `auto -> ... 46858.8 MiB of 47962.5 MiB: the arena's complement` | 97.7 % | first 50k prefill 1443 vs 1941 t/s |
| `MOE_HOST_POOL_RAM_PCT=50` | `does not fit the RAM ceiling ... scaled by 0.926` | 92.4 % | first 50k prefill 1061 vs 2080 t/s |
| `-lme 51`, default ceiling | same line, factor 0.936 | 93.6 % | decode -5 %, prefill -9 % |
| `-lme 52` | `more than 90 % of the available memory (57 GiB)` | refused | does not start |
| `-lme 47` + `MOE_HOST_POOL_RAM_PCT=0` | `-lme 47 -> ... the whole host expert set (a MIRROR)` | 100 % | decode 78.5, prefill 1802 t/s, 0 MiB of reads |

The three defects, in order of how much they cost:

1. **`auto` cannot mirror on a 59 GiB box.**  `MOE_HOST_LIMIT_FRAC=80` of MemAvailable (57 GiB) is 45.8 GiB
   against a 46.84 GiB set, so the default lands at 97.7 % - the worst of both worlds: it pays 45.8 GiB of pinned
   RAM *and* 2.3 % of its reads leave the tier, which is a 25 % first-pass prefill loss on this model.
2. **`MemAvailable` is not container-aware.**  `host_experts_available_bytes()`
   (`src/llama-model-loader.cpp`) and `pool_available_ram_bytes()`
   (`ggml/src/ggml-cuda/moe-expert-cache.cu`) both read `/proc/meminfo`, which inside a memory cgroup reports the
   HOST.  Measured: a 56 GiB cgroup on this 93 GiB host reported 89.8 GiB and sized from that number.  r209 fixed
   the same class of problem for Windows (there `MemAvailable` did not exist at all); a cgroup is the Linux
   counterpart and is still unhandled.
3. **The RAM ceiling silently overrides an explicit budget.**  `-lme N` is documented as the one knob, but
   `pool_clamp_slots_to_ram()` scales it by `MOE_HOST_POOL_RAM_PCT (85) % of MemAvailable` and reports that on an
   INFO line, filtered without `-lv 4`.  512/512 and 479/512 look identical to the operator, and the difference is
   a 2x prefill loss.  The ceiling's job is to stop an *unbudgeted* `FRAC` pool from eating RAM; it has no
   business trimming a budget the operator asked for.

Also worth fixing while here: the 90 % guard's message does not say what N would have covered and does not name
the right value, and `report_host_expert_placement()` (`src/llama-model-loader.cpp`) prints the resolved tier at
INFO, i.e. invisibly.

## 3. The work, in the order I would take it

1. **cgroup-aware available memory.**  One helper returning `min(MemAvailable, cgroup_headroom)`, where
   `cgroup_headroom` comes from this process's own cgroup (`memory.max`/`memory.high` minus `memory.current`,
   walking up while unlimited; v1 shapes fall back).  When a `memory.max` exists it is the binding figure, because
   pinned bytes cannot be reclaimed.  Both call sites use it; a non-cgroup consumer keeps today's behaviour.
   *Accept:* on the rig, the plan line reports 56 GiB, not 89.8 GiB, and `-lme auto` with a 56 GiB container and a
   46.84 GiB set picks a mirror.
2. **An explicit budget is authoritative.**  When `-lme N` covers the set, the ceiling must not shave it: either
   skip the ceiling for that plan, or clamp only the excess above 100 % coverage.  When a ceiling DOES bind, log
   it at WARN with the resulting coverage and the measured miss cost.
   *Accept:* `-lme 47` with the DEFAULT `RAM_PCT=85` still yields 512/512 on the rig; no configuration silently
   scales a mirror.
3. **One verdict, visible.**  One WARN line (and a `/props` field, so a client can read it) saying
   `MIRROR 512/512`, `PARTIAL 479/512 (93.6 %)` or `NONE`.  *Accept:* all three states are distinguishable
   WITHOUT `-lv 4`, and the model `runme` shows them.
4. **The guard explains itself.**  `common/arg.cpp`: name the set size, the coverage N would buy, and the value
   that would work.  *Accept:* `-lme 52` on the rig prints all three.
5. **Documentation.**  `ENVIRONMENT.md` rows for `MOE_HOST_LIMIT_FRAC` / `MOE_HOST_POOL_RAM_PCT` gain the
   `-lme`-versus-ceiling interaction and the `RAM_PCT=0` pairing; the model `runme` gains one sentence.
   *Accept:* the two rows no longer require the reader to infer it.

Nothing here changes a default; steps 1 to 3 make today's behaviour honest and visible, which is what the
`bench-56gb` matrix needs before any transport campaign can claim a result.

## 4. What would kill it, and what to watch

* If a cgroup-aware figure is used for the *ceiling* as well, a tight container shrinks the pool on boxes where
  the host has RAM to spare.  That is the correct answer (the container is the contract) but it will move
  numbers on the maintainer's primary box; gate it on "no cgroup limit -> byte-identical to today".
* `memory.max` is one of several shapes.  Do not derive the limit from a rule of thumb; read what exists and fall
  back to `MemAvailable`.
* The lazy PLE (`per_layer_token_embd`, 26.8 GiB on this model) must not be counted as resident anywhere in this
  arithmetic.  It is not, today; keep it that way.
* `-lme auto` deciding "mirror" from a *container* figure is a behaviour change on small containers.  That is the
  point, but it must be recorded in `WORKLOG.md` and in `ENVIRONMENT.md`, with the arm that shows it.

## 4b. r221 (2026-11-05) made MIRROR-vs-PARTIAL load-bearing for a default-on feature -- a motivating case for step 3

Nothing in this note changed, and r221 touches none of its three defects (no cgroup figure, no authoritative
explicit budget, no verdict line).  But it makes the coverage question cost something visible, in two ways:

* **`GGML_MOE_DEVREMAP` (default ON) now declines on a PARTIAL pool and serves a MIRROR one** (`dr_pool_ok`,
  latched from `pool_covers_table()` at sizing; block 09, r221).  So a tier that lands at 93.6 % instead of
  100 % no longer just pays a slower first prefill -- it also forfeits **+9.9 % decode** (1 GPU, 35B-A3B Q8_0
  `-ncmoe 40`, the default `-lme auto` mirror arm: 57.6 -> 63.3 t/s), silently, at default verbosity.  That is
  defect 3's "the operator cannot see the tier" with a decode number attached instead of a prefill one.
* **The same default config is a mirror or not depending on `MemAvailable` at load time, which I hit live this
  session.**  35B-A3B Q8_0 (32.6 GiB set, ~75 GiB available) planned `auto -> ... the whole host expert set (a
  MIRROR)`; the 122B UD-Q4_K_XL (67.1 GiB set, ~63 GiB cap) planned `auto -> ... the arena's complement` on the
  same box minutes later.  Neither line says MIRROR/PARTIAL/NONE, and the plan line is INFO on top of that.
  One precision for step 3: `report_host_expert_placement()` is already **WARN** (visible without `-lv 4`) and
  prints `pinned X MiB of Y, pageable Z MiB` -- so the work is to put the verdict and the coverage ratio on the
  line that is already visible, not to raise a level.
* Consequence for any arm that A/Bs a host-expert change on this box: **check the plan line before believing a
  decode delta**, because a PARTIAL tier turns a remap arm into a no-op and reads as "the feature did nothing".

`INVARIANTS.md` r221 fact 6 (`device remap: N tables served, M of K offers declined`) is the per-run tell; this
campaign's step 3 would replace reading it with one operator-facing line.

## 4c. r224 (2026-11-06) made the FILL MODE a policy and left its decision invisible -- a third instance of defect 3

**Nothing in §3 changed; this is a new measured instance, found while opening `archive/work/host-expert-parity/`.**  r224
folded `pool_decide_dio_locked` (block 12): the pool's fill mode is now a **policy** — it mirrors the operator's
`--load-mode` (a mapped load fills through the page cache, `--load-mode dio` gets O_DIRECT), and the old
MemAvailable heuristic survives only for a consumer that reports no load mode.  Two consequences for this
campaign's step 3:

* **The decision, and the warning attached to it, are `GGML_LOG_INFO`** — i.e. invisible at the default
  verbosity, exactly like the plan line §2's defect 3 describes.  The line that matters in the middle band is:
  `host pool fill mode: buffered (page cache as L3) -- mirroring --load-mode mmap …; NOTE MemAvailable does not
  hold the pool AND the host expert set, so the page cache cannot keep it -- MOE_HOST_POOL_DIO=1 is the
  memory-tight arm`.  On a 59 GiB box with a 46.84 GiB set and a ~46 GiB pool this is **the** operator-facing
  fact of the run, and it is filtered.  Step 3's one visible verdict should carry it: `MIRROR 512/512` /
  `PARTIAL 479/512 (93.6 %)` / `NONE` **plus** `fills: buffered (page cache as L3)` or `O_DIRECT`.
* **The middle band is unmeasured by the code's own admission.**  `pool_decide_dio_locked`'s comment says the
  case "the set fits, the set + a full pool does not" is unmeasured and points at the forced
  `MOE_HOST_POOL_DIO=0|1` escape hatch — and the `bench-56gb` rig **is** that band (46.84 GiB set against 57.3 GiB
  of `MemAvailable`, i.e. the set fits and the set + a 45.8 GiB mirror does not).  So this is not a corner case:
  it is the shipped shape of the box the tier exists for, and the arm that decides it is the rig's
  `DIO=auto` vs `DIO=1` pair.
* **The census distinguishes the two arms**, which is what makes the A/B legible: the exit line prints
  `pool fills: N storage reads … ms, M mapping memcpys … ms` separately (`-lv 4`).  Quote that line, not the
  process `read_bytes` — under `-lzm auto` the CPU-offloaded PLE moves its reads into load time and the process
  total is not comparable across lazy modes.

The measurement itself (the field matrix, the parity/crossover table) is `archive/work/host-expert-parity/` P0.2; this
note exists so step 3 does not ship a verdict line that omits the fill mode.

## 5. Records

| where | what |
|---|---|
| [`bench-56gb/RESULTS.md`](../bench-56gb/RESULTS.md) | the measured matrix these numbers come from |
| [`archive/work/host-expert-eager-push/`](../../archive/work/host-expert-eager-push/) | the tier's own campaign: `P5-RESULTS.md` §7 (the shape anchors), `P6-POOL-REGION.md` (the pool-as-region design), `P4-ROOT-CAUSE.md` (`MemAvailable`/fill-mode history) |
| `ENVIRONMENT.md` | `MOE_HOST_LIMIT_FRAC`, `MOE_HOST_POOL_FRAC`, `MOE_HOST_POOL_RAM_PCT`, `MOE_HOST_POOL_DIO`, `MOE_HOST_POOL_ALLOC` |
| `INVARIANTS.md` | the pool-is-a-layout rule, the r215 item 8 (first-build `MemAvailable`), the Windows `MemAvailable` note (r209) |
| `patches/README.md` block 09 | what shipped: the tier's knobs and the env-kill-switch contract |
