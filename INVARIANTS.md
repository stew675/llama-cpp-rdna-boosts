## r299 — the mid-run pinned-tier latch's MIXTURE is a ~1e-5 arithmetic path, and it is gated on the LOGITS (block 12, `TODO` #282 retired)

1. **`MOE_HOST_POOL_PIN_FAULT=late` must latch after the run is LIVE, not merely after the arena is sized.**  Sizing
   moved to registration (r254/r292), so `g_sized` is true before the pool tier is built; a `g_sized`-only latch fired
   on the second pool and left **1 of 141** pools built (239 MiB of a 31 GiB tier) with **2** cache-served ops out of
   5 069, i.e. the arm silently stopped reproducing issue #278's shape (the pools are built, the run is live, then a
   grow is refused).  Keyed on the first ROUTED op as well (`g_routed_op_seen`): 141 pools / 30 791 MiB built, 6 342
   fills served from pre-latch pools, 151 rolled-back fills, `get_ok` 660 / `decline_all` 1 082.
2. **The mixture's logit difference is real and ~1e-5; the generated TEXT is not its criterion.**  Measured on r269
   (which reproduces the divergence deterministically) under the field-shape latch:
   `llama-perplexity -c 512 --chunks 2 --kl-divergence` reads **mean KLD 0.000001, max 0.000051, 99.9 % 0.000050,
   same-top-p 100.000 %** (base PPL 3.6948), while the same pair's 32-token greedy text flips one token of the
   reasoning preamble (158 vs 182 chars).  That is an fp32 association difference between the set of cache-served and
   scheduler-served ops, not wrong bytes (`MOE_HOST_POOL_VERIFY=1` is clean; the WHOLESALE arms `fail`/`late` stay
   byte-identical to `ref`, so the fallback computes the same logits as the cache).  A byte-identical text assertion
   is therefore the right instrument for a WHOLESALE fault arm and the wrong one for a mixed state.
3. **The near-tie is LAYOUT-dependent, so "the text matches" is not evidence about the arithmetic.**  On r269 the same
   mixture reads 158 chars `c634f36735fa` with the default (offset-order) prewarm, 175 `d3d1fe6b3db3` with
   `MOE_HOST_POOL_SINK=0`, and **182 `a99ccbd97ecb` (== `ref`) with `MOE_HOST_POOL_FILL_ORDER=rank`**;
   `MOE_EXPERT_PREFILL_IN_PLACE=0` does not move it.  The prewarm's fill order decides which experts land before a
   RAM-capped pool runs out, hence the residency set, hence the residue on the near tie -- which is why r270's
   `TODO` #318 ordering change "fixed" the flip without any numerics change.  Never read a matching text as
   "identical arithmetic", and never read a differing text as a defect: use the KLD.
4. **A mixture gate must assert that the mixture is REAL.**  A near-tie flip needs the mixed state, so a degenerate
   latch passes for the wrong reason (r298 shipped 1 pool / 2 served ops).  The gate now requires, in the same run:
   the latch line, pools built before it, pre-latch fills > 0, rolled-back fills > 0, a byte-identical wholesale arm,
   and the mixture's KLD inside 0.005 / 98 %.  `pinlate`'s text is reported, never required.

## r298 — the in-place prefill read is a base-pointer swap; the residency-dependent text was a STALE ALIAS resolving a sibling ROLE's arena (blocks 09/10/15, `TODO` #335 + #336 retired)

1. **The in-place prefill read is NOT numerically different from the staged read — measure it before blaming it.**
   `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` on the arm that really serves in place
   (Flash-Next GSQ-RCO IQ3_XXS, 2 devices `-sm layer`, `-ncmoe 48`, `-lm none -lzm off`, `--ctx 2048 --chunks 7` on
   `prompts/kld-corpus-35b-in-domain.txt`, 143 identity tables, `get_ok` 1280) reads **mean KLD 0.000000 / same-top-p
   100.000 %**, base PPL 1.9143 — on the pre-fix build and on this one.  Per dispatch the two arms choose the **same
   kernel family for the same op and shape** (8022 identical `mul_mat_id` dispatches: `mmq T=512` prefill, `mmvq T<=3`
   decode, `mmq T=42` MTP prompt-ingest tail) and 276 of the hashed results are byte-identical.  `#336`'s owed
   magnitude is **zero**; a logit difference between those two arms can only be a *different pointer*.
2. **`moe_cache_get_table`'s alias guard MUST compare the semantic ROLE, not just the device and the layer.**  The
   consumer resolves a table from the scheduler's `input_cpy` ADDRESS (`g_alias_to_id`), and the graph allocator
   recycles those addresses; a layer's `gate`/`up`/`down` tables share device + layer, and a layer's routed ops are
   issued back to back, so a sibling role's alias is routine.  Worse, the two sides resolve by DIFFERENT keys: the
   take-over hooks use the HOST master (right role, so they decline for a non-identity table and the scheduler fills
   the copy) while the redirect uses the COPY POINTER (stale alias ⇒ it serves the neighbour's arena).  Measured
   (Flash-Next + MTP, `-c 32768`, prior absent): `blk.43.ffn_up_exps` at T=42 (`identity=0`, remap 10 wide) returned
   `nan` then `-2.1e7` because it was read out of `blk.43.ffn_down_exps`'s arena (`identity=1`, device 1, layer 43);
   both arms collapse to `835 chars 7a3474cbe5f1` with the role check.  It is one in-place compare per routed op and
   it rejects **once** in a 24 K-token prompt (the fallback is the existing `(layer, sem_role, device)` key).
3. **A cache-validating instrument cannot see a wrong POINTER.**  `MOE_EXPERT_CACHE_VALIDATE=3` byte-compares slots
   against the host master and read 0 `MISMATCH` over 157 sampled dispatches on the very arm producing NaN: the arena
   held the model's bytes, and the consumer was reading the wrong table's slots.  When an output is wrong but the
   bytes validate, instrument the *dispatch* (which table, which width, which kernel) before the data.
4. **`TODO` #335's workaround matrix is a stale-alias matrix, and `GGML_CUDA_DISABLE_GRAPHS=1` is the tell.**
   `DISABLE_GRAPHS=1` (a capture bakes the pointer it served into the launch parameters, and a replay repeats the
   wrong table for the graph's life), `GGML_CUDA_SLAB=0` / `MOE_EXPERT_CACHE_MIB=20000` / a smaller `-c` (no eviction
   ⇒ no stand-down/re-arm ⇒ no alias churn), `MOE_EXPERT_PREFILL_IN_PLACE=0` (the staged arms never run the redirect)
   and dropping the drafter (no T=42 MTP ingest batch, the width where the first wrong read appeared) each remove the
   symptom — so "removing the captures fixes it" does not prove a capture defect when captures also change which hook
   writes the alias.  `GGML_MOE_DEVREMAP=0` and `GGML_SCHED_DEVGATHER=0` do NOT move the pre-fix pair.
5. **A take-over's promise must be repairable, and it is.**  A hook returning true tells the scheduler "do not fill
   `input_cpy`", and the slab can take the table's arena BETWEEN that hook and the consumer's launch (an op's
   workspace allocation can move the work boundary from inside a graph) — the per-table guards cannot see it.
   `moe_cache_repair_copy` re-fills the buffer from the host master when the table still holds a promise for THAT
   buffer; `MOE_EXPERT_TAKEOVER_FAULT=1` breaks the promise on purpose and the arm's text is byte-identical to both
   no-fault arms (`595 chars c0e7e91d84c4`), so the path is exercised rather than assumed.  The census prints
   `take-over promises broken=N repaired=N`, and ordinary runs (including one that evicts mid-prompt) read **0 / 0**.

## r297 — the prefill MoE fusions cannot fire for a cache-managed table, so `TODO` #327's coupling never existed — and what does move the text (blocks 09 and 10, `TODO` #327 retired, #335 opened)

1. **A routed `MUL_MAT_ID` whose weight is a HOST buffer is its own single-node SPLIT, so `gate`, `up` and `GLU`
   never share a graph and the four prefill MoE fusions decline for SHAPE, not for the cache.** `ggml_cuda_try_fuse`
   runs per split graph; with `-ncmoe` the routed op's weight lives on the host backend, so the scheduler isolates
   it. Split-graph census on 35B-A3B Q3_K_M, `-ncmoe 48 -ub 2048`, 2 devices `-sm layer`
   (`GGML_CUDA_FUSE_TRACE=1`): **117 × 1 node, 117 × 2 nodes**, and the multi-node splits are the attention and the
   device-resident layers (26 / 72 / 74 / 99 nodes). The `-lv 4` census added here reads all four kinds at **zero** on
   every cache-managed arm (35B-A3B Q8_0 / Q4_K_M / Q3_K_M, Flash-Next GSQ-IQ3_XXS, `-sm layer` and `-sm tensor`,
   `-ncmoe 48`; `get_ok` 951-7510 in-place reads) and `pair=0/117` **only** on a fully device-resident 35B-A3B
   (no `-ncmoe`, where the cache has no tables at all). So the guard's above-band `moe_cache_table_serves()`
   stand-down was **dead code**: residency could not select the kernel set through these fusions, and r284's
   `GGML_CUDA_MMB=0` differential is MMB's **dense** GEMM (router + shared expert), not the routed GLU.
2. **The fusions carry the in-place contract anyway, three-state, and it is inert today.** `moe_cache_fusion_serve()`
   answers per lane with `moe_cache_get_table()` and returns NONE (use the graph's staged tensors) / IN_PLACE (use the
   arena copies + remapped ids) / MIXED (**stand down** — one lane's copy was never filled, so only the per-op chain,
   which resolves each table itself, is correct for that state). Taught sites: the MMB routed gate+up+GLU triple, both
   MMQ gate+up+GLU triples (RDNA and the NVIDIA dense arm), the MMQ shared-quantize pair
   (`ggml_cuda_mul_mat_q_pair` takes optional `src0`/`ids`) and the MMQ `swiglu→down` fold. The staged arms are
   byte-identical to r296 (35B-A3B Q3_K_M device-resident, `-ub 2048 -c 8192`, 16 tokens: `sha=86b7f9b5aa80` on both
   builds, `pair=0/117`). No environment variable is added; `MOE_EXPERT_PREFILL_IN_PLACE` stays the kill switch.
3. **What does move the text is the IN-PLACE PREFILL READ itself, in two shapes — neither of them a fusion.**
   (a) r284's coherent near-tie: Flash-Next + MTP sidecar, `-c 32768`, the prior absent → `f40ae789f219` with
   `MOE_EXPERT_PREFILL_IN_PLACE=1` against `7a3474cbe5f1` with `=0` (and `7a3474cbe5f1` for the prior-present arm
   either way); none of the four fusions fires in either arm, and `MOE_EXPERT_CACHE_VALIDATE=3` reads 0 slot
   mismatches. (b) **under VRAM pressure it is outright degenerate**: the same arm at `-c 16384` (`free-clamped=2`)
   produces `82 chars sha=a753000294c9` (`…3333333…`) against the staged arm's `328 chars 7c5a512a8447`, and
   `GGML_CUDA_DISABLE_GRAPHS=1` (coherent `04ca2590aebf`), `GGML_CUDA_SLAB=0` (`7c5a512a8447`) or a smaller arena
   (`MOE_EXPERT_CACHE_MIB=20000`, `04ca2590aebf`) each remove it. The carrier is therefore a **captured graph replaying
   an arena pointer whose table `moe_cache_evict_slab_range` stood down mid-run**: the eviction and its re-arm both
   land inside the prompt (`evicted 3946.0 MiB … 12 table(s) stood down`, then `re-armed 12`), the arena validates
   clean at every sampled dispatch (`resident=132 down=12`, 0 `MISMATCH`), and filling the copy the take-over skipped
   does not change it. That is `TODO` #335; until it is fixed, a text comparison on a pressure configuration must
   also pin `MOE_EXPERT_PREFILL_IN_PLACE=0`.
4. **The `-lv 4` fusion census is the instrument, not the log line.** `report_locked: prefill MoE fusions:
   <kind>=<in_place>/<staged> … mixed_lanes=N` (kinds: `mmb_glu`, `mmq_triple`, `pair`, `swiglu_down`) is what makes
   point 1 checkable in any future run: a residency state that reroutes a kind from one column to the other is
   exactly the violation, and a nonzero `mixed_lanes` names a broken lane-agreement assumption instead of a slow run.

## r293 — the MTP target h-rows path is a per-platform trade, not a win (block 02, GitHub issue #318)

1. **The device `h_nextn` path (upstream `#31927`) is NEUTRAL on gfx1201/Linux and NEGATIVE on
   Windows/HIP, so it is now opt-in (`LLAMA_MTP_DEV_ROWS=1`) and the default is the host readback.**
   Measured here on dense Qwen3.8-27B UD-Q5_K_XL, one R9700, `-sm layer`, `q8_0` KV,
   `--spec-type draft-mtp --spec-draft-n-max 3`, `--seed 42 --temp 0`, three interleaved reps per arm:
   at a **61 525-token** prompt **47.9 / 48.0 / 47.9 t/s** (device rows) against
   **47.9 / 47.9 / 48.0** (host rows) at `-n 128`, and at a 5 246-token prompt **49.2 vs 49.2** t/s at
   `-n 512`.  The reporter measured **-1.3 / -1.6 / -1.5 %** at depths 62 857 / 68 767 / 100 607 on
   Windows 11 + one R9700 (+0.25-0.30 ms per generated token, about 1 ms per verify+draft round), with
   byte-identical text and identical draft acceptance in both arms, and a kernel census showing the same
   graph shapes and the same GPU time.  So the cost is the **ordering handshake**, not the kernels: the
   device path replaces one async device->host readback the host path had already paid for with a
   host-blocking event wait plus two blocking row copies per round.
2. **The arm was never a delivery decision — it came in with the base.**  `r278`'s re-base brought
   `#31927` in, and the enable is upstream's own `if (!is_mem_shared && !chain_heads)` in
   `common_speculative_impl_draft_mtp`, so **r281 turned it on for every single-head MTP model** without
   anyone choosing it.  When a base brings a new default in, that default needs its own measurement: this
   one had none, and the first field report was a regression.
3. **Keeping the drafter's carry row on the device and ordering it with a stream wait is SLOWER here
   than the host round trip it removes.**  The rewrite (one 10 KB `ggml_backend_tensor_copy_async` into
   the reserved carry row per round, the drafter's first step reading that row too, and
   `ggml_backend_event_wait` instead of `ggml_backend_event_synchronize` in `decode()`) measured
   **47.7 / 47.7 / 47.7 t/s** against the host arm's **47.9 / 48.0 / 48.0**, i.e. **-0.5 %**; the
   stream-wait half alone was neutral (47.7 with either wait arm).  Two tiny copies + two drains +
   two event records cost more than two blocking copies of one row.  Preserved with its diff and numbers
   in `archive/work/mtp-dev-rows/` — do not re-derive it, and do not assume "remove the host sync" is a
   win on this platform.
4. **What the two arms may not differ in: the greedy text and the draft acceptance.**  They only move
   where the target's hidden rows are read from, so the text is byte-identical by construction (the target
   verifies every draft) — measured `542 chars sha=d56a982e90c2` with `79/141` accepted on the 27B and
   `sha=bbb051e2aeab` with `87/119` on the 4B — and an arm that changes the acceptance is being fed the
   wrong rows.  Both are gated by `scripts/gate-mtp-dev-rows.sh`, which also fails if a default run
   allocates the `h_nextn` buffer at all.
5. **A ROCm line is a codegen variable: a 10.0 build measures ~+0.6-0.8 % prefill here than 10.1, and
   the reporter measured ~2 % at depth on Windows.**  Same tree, same flags, one model (dense 27B
   UD-Q5_K_XL, 1 x R9700, `-sm layer`, q8_0 KV, `-ub 512`, `llama-bench -r 2`, arms interleaved 10.1 /
   10.0 / 10.0 / 10.1): `pp4096 @ d32768` **872.0 / 872.6 vs 878.5 / 878.6 t/s** (+0.7 / +0.7 %),
   `pp4096 @ d0` **1202.9 / 1199.9 vs 1207.7 / 1206.6** (+0.4 / +0.6 %), `tg64` flat at both depths
   (25.38/25.38 vs 25.45/25.44 at depth, 26.90/26.90 vs 26.84/26.84 at d0).  On Windows the same
   comparison is -2.0 to -2.7 % at `d32768` for a whole release's worth of builds, answers `-O2` (so it is
   not the clang-cl `-O0` issue), does not reproduce at `d0`, and about half of it is recovered by
   `-Xclang -target-feature -Xclang -real-true16` (device only) — i.e. it is device codegen, and the
   delivery's own native-q8_0 MMA path is more sensitive to it than clean upstream.  **Consequences:**
   never compare a prefill number across ROCm lines (the prefill-logit gate already refuses a base from
   another line, and this is why), and a toolchain bump is a re-baseline, not a flag flip.
## r294 — the load mode is not the tier: the host-expert master, the demand transport, and `-lme mirror`

1. **`--load-mode` decides STORAGE for the device weights and TRANSPORT for the pool, and nothing else.**  The host experts'
   master is the read-only model mapping whenever the tier has holes (`host_expert_master_mapped()` = a hole set exists and
   `llama_mmap::SUPPORTED`), for EVERY load mode.  A load mode that dropped the mapping materialised the whole set instead --
   measured: `-lme 30` under `--load-mode dio` pinned **48 284.6 MiB** of `ROCm_Host` with no pool at all, against
   **31 363.7 MiB** (`ROcm_HostMap` master + a 30 GiB pool) after the fix, `0 mapping memcpys` in both.  A budget you cannot
   honour is not a budget.
2. **The demand transport is `--load-mode`, alone.**  `MOE_HOST_POOL_DIO` and `MOE_HOST_POOL_TRANSPORT` are REMOVED (one WARN,
   then ignored): the load mode already reports itself through `moe_cache_set_host_fill_mode`, so a second knob was a second
   spelling of one choice.  The mirror is `dio` -> `O_DIRECT`, `buffered`/`none` -> a buffered `pread` (upstream's "no special
   loading mode" IS a buffered read), `mmap` -> the page cache, `auto` -> the adaptive ladder.  A rung stays pinnable for an A/B
   through `MOE_HOST_POOL_TR_START`.  **The one case the removed knob could express and the load mode cannot -- "page cache for
   the model, `O_DIRECT` for the experts" -- was not worth a variable: the benefit it assumed does not exist**, because
   `LLAMA_LOAD_RELEASE_MMAP=1` (default since r273) `munmap`s each device tensor and `posix_fadvise(DONTNEED)`s its page-cache
   copy immediately after upload (`src/llama-mmap.cpp`, `unmap_fragment`), so a warm restart gains nothing.  Measured, same
   config, two consecutive runs: `mmap` 9.2 / 9.2 s against `dio` 9.8 / 9.6 s -- and the second run is identical to the first,
   i.e. there is no warmth to harvest.
3. **The PLACEMENT transport is the one decision that must NOT follow the load mode**, and it keeps its own switch
   (`MOE_HOST_POOL_PLACEMENT_MODE=dio|buffered`, renamed from `MOE_HOST_POOL_BULK_DIO`).  A placement read is a copy into the
   pool's own storage, never a cache miss, so the page cache cannot keep the second copy.  Re-measured cold (drop_caches
   between arms, `-lme mirror` = a 47.9 GiB placement): **+1 675 MiB** of page cache on `dio` against **+31 796 MiB** on
   `buffered` for the same bytes -- r273's amplification, 19x, on one env var.
4. **`-lme mirror` is the only way to ask for a covering pool, and nothing scales it.**  `auto` resolves to the whole set when it
   fits `MOE_HOST_LIMIT_FRAC` of MemAvailable, which on a RAM-rich box *is* a mirror -- but that is inference, and r294 makes
   the request explicit: neither `MOE_HOST_LIMIT_FRAC` nor `MOE_HOST_POOL_RAM_PCT` may touch it, and it is not refused for
   exceeding MemAvailable.  Measured with `MOE_HOST_POOL_RAM_PCT=20 MOE_HOST_LIMIT_FRAC=5`: still 47 962.5 MiB pinned,
   `h=1.0000`, 512/512 slots.  An explicit mirror that does not fit is the operator's call.
5. **A census that reports the PLAN is not reporting the BACKING.**  `report_host_expert_placement` prints the plan's hole set as
   "pageable", which was a lie in a materialised load (48 GiB of pinned `ROCm_Host` labelled pageable) and is true again now.
   It names the realised backing whenever the platform cannot map.
6. **`parse_env()`'s logging never reaches the user.**  `ensure_env()` runs before the host installs its ggml log callback, so
   every `GGML_LOG_*` in `parse_env` is dropped -- that is why an invalid `MOE_HOST_POOL_TRANSPORT` value silently left the pool
   with NO transport decision (the WARN that would have said "ignoring" was invisible, and with the env var gone the failure
   mode is gone with it).  **Fixed at r295** (`TODO` #332 retired): the mechanism is even broader than this note assumed — the
   parse runs inside the `--fit` PROBE, whose log scope demotes sub-threshold levels to DEBUG — and the lines are now buffered
   and flushed late.  A new env diagnostic belongs in `parse_env` again; see r295 fact 1.
## r295 — `parse_env()`'s diagnostics are buffered, and the pool exchange's inertness is named

1. **`parse_env()` runs inside the `--fit` probe, and the probe's logger is deliberately quiet.**  `ensure_env()` is called at
   the first cache call, which on a default `--fit` run is the PROBE load — and the probe wraps the llama logger in
   `common_fit_log_scope`, which demotes every level below its own to `GGML_LOG_LEVEL_DEBUG`, which `common_log_add` drops at
   every default verbosity.  So *every* `GGML_LOG_WARN`/`INFO` `parse_env` issued was lost, not just on a pre-callback path:
   measured `MOE_EXPERT_SEED=bogus` printing `is not eager|auto|lazy; using eager` with `--fit off` and nothing at all with
   the default `--fit`.  The VALUES must still be parsed that early (the preflight reads them, and it can run before backend
   init — issue #136), so only the REPORTING is late: `env_logf()` buffers the lines and they are flushed at the first routed
   op, the first pool build, or the exit census, whichever comes first.  **A new env diagnostic belongs in `parse_env` again**
   — it is no longer a hole — and a diagnostic that must appear before any of those three points belongs in the pool build
   itself (the deprecation warnings live there).
2. **`demote_ok` is a pure table-geometry predicate, so on `-sm tensor` the exchange is disengaged before any attempt.**
   `demote_ok = (n_slices == 0 && host_bytes == expert_bytes)` (`pool_sink_active()` requires it), and a tensor split gives
   each device a SLICE of every expert.  Measured on this box, `-lm 99 --n-cpu-moe 47/48 --sm tensor`, `MOE_HOST_POOL_SINK=1`,
   the `-lv 4` `CACHE_GEOM` census (`GGML_CUDA_CACHE_GEOM_DEBUG=1`) as the geometry of record:
   `ffn_gate_exps`/`ffn_up_exps` `split_axis=1 host_pitch=0` (`expert_bytes` 253 440 / 168 960 against `host_bytes` 422 400),
   `ffn_down_exps` `split_axis=0 host_pitch=180` (184 320 / 276 480 against 460 800), `n_slices=0` everywhere — so **no table
   is expressible** and the census reads **0 demoted / 0 geometry declines / 0 retirements** with 141 of 141 pools
   non-expressible, on 2 AND 3 devices, on Flash-Next IQ3_XXS and IQ3_S.  Two consequences: the r294 ladder's
   `demoted=5399` reading is not reproducible on that same build with the arm's flags (recorded as such rather than
   explained away), and **a gate must not encode this model's block geometry as the assertion** -- it asserts a retirement
   is paid for by a landed demote, that `0 demoted` is EXPLAINED (a geometry decline, or the new
   `sink no-demote tables (...)` census line that names the guard), and full inertness only where the census says no pool
   can express a demote.
3. **The `11-26 %` spill figure is gone from the code as well as the docs.**  `common/arg.cpp`'s `--kv-window` help and
   `common/fit.cpp`'s placement hint now say the cost is linear in the depth past the window (and nothing at or above the run
   depth), matching `ENVIRONMENT.md` §17.0 since r292.  Text only: `--fit`'s per-token estimate is unchanged (it divides by
   the calibrated in-place read rate), and the r295 gates that print the line move no number and no token.

## r296 — the split-state walk cannot be stranded, the seqpar floor is the fused-HC band, and `mlock` composes with `-lme`

1. **A split-state cache entry is valid iff its key is present AND the tensor's raw bytes are identical; the cache
   VERSIONS are a performance mechanism, so pinning one version for the duration of the pre-warm walk costs nothing
   and is what makes the walk reliable.**  `ggml_backend_meta_get_split_state`'s non-recursive walk exists because
   `calculate_split_state` recurses every `src` (~500-1200 frames on qwen4exp, more than the 8 MiB stack).  Before
   r296 a nested lookup of a tensor whose bytes had changed **switched** the current version, which made every entry
   the walk had already filled invisible -- and when that switch happened on the root's own lookup, after the walk,
   the whole walk was stranded and the recursion descended the chain (the r291 stack overflow at a token-split ubatch
   below 128 tokens).  The fix is two-part and must not be "simplified" back: the tensor's own resolution runs
   **before** the walk, and while the walk runs a nested lookup **erases its own stale entry and computes into the
   current version** (`ss_pin`) instead of moving it.  Measured: recursion depth **1** on every arm (temporary probe;
   qwen4exp 2 devices `MIN=2 -ub 48 pp4096`, plus the multi-shape sweep `-p 24,48,64,80,112,160,256,512` in ONE
   process), and `GGML_META_SS_VERIFY=1` clean over the same sweeps.  Resolving a miss from *another* version's
   byte-identical entry is still NOT done (it is the one change the r291 note warned about); the pin makes it
   unnecessary.
2. **A token-split ubatch at or below the fused hyper-connection band is not a graph seqpar can serve, and that -- not
   a cache defect -- is the band's real floor.**  `GGML_OP_HC_MIX`/`HC_COMBINE` assert mirrored inputs
   (`ggml-backend-meta.cpp:1953` and following) and have no token-split form; they are selected for
   `nt <= HC_BF16_MAX_TOKENS` (16, BF16 mixer) or `nt <= HC_FUSED_MAX_TOKENS` (8, Q8_0).  Measured: 8 and 16 abort,
   17 and up run clean.  So the floor is **17** (one floor for both families), `LLAMA_SEQPAR_MIN` is clamped to it
   with one WARN, and the shipped default stays **512** (-1.0 % at 128, -0.8 % at 256, +6.4 % at 512, r291's sweep).
   r291 recorded this abort as "the token-scatter split-state assertion"; the line it named is the first HC_MIX
   assert, and the token-scatter handler was never reached (corrected here, with the measurement).
3. **`grow_to()` is a contiguous sweep from the mapping base, so any range that must stay unlocked needs per-range
   locking.**  `llama_mlock::grow_to(target)` locks `[base, target)`; with an `-lme` hole set the mapping-backed
   tensors are exactly the holes, so an mlock load mode pinned across the whole hole set (measured
   `-ncmoe 48 -lme 30 --load-mode mmap+mlock`: **43.5 GiB** VmLck) and silently overrode the budget.  r296 adds
   `llama_mlock::lock_range(offset, len)` and exempts the holes: **260 MiB** pinned after, against a 40 925 MiB hole
   set, with the pool's own `30 693.8 MiB` unchanged.  A hole's device reads are served by the pinned pool, which
   wires exactly what it serves.
4. **`--load-mode mlock`/`mmap+mlock` is a RESIDENCY request for the tensors the CPU reads, not a read transport and
   not a device-side decision.**  Measured on 4B Q8_0 (`--load-mode mmap+mlock`): **4 274 MiB** pinned at `-ngl 0`
   (CPU inference -- residency is the whole point) and **654 MiB** at `-ngl 99` (what still lives in the mapping).
   A fully offloaded model has little left for the pin to keep, which is why the flag can compose with a tier budget
   instead of outranking it.  And **a host-expert tier is pinned by necessity whatever `--load-mode` says** (no
   XNACK: a device read of a host-resident expert must come from a resident page -- the pinned master, or the bounded
   pool in front of a pageable hole).  Both statements are logged (INFO at the tier census; the hole exemption WARN
   plus the `mlock:` census), so an operator cannot read either flag's absence as "the host experts are reclaimable".
5. **A pooled KV's text is invariant across DEVICE COUNTS (and across its two spellings), but it is not the
   resident arm's text -- and the resident reference is the side that drifts.**  `gate-kv-residency.sh`'s
   multi-device section demanded `pmres == pmref`, and it had been red since at least r295 (measured identically
   on the r295 and the r296 build): pooled `f260d1824ab3` / 842 chars against resident `1e944569dbd6` /
   831 chars, diverging 539 characters into a coherent continuation, with `LLAMA_KV_POOL_VERIFY=1` reporting
   `bad 0` (every resolved page IS the authoritative mirror's bytes).  Every record that measured the identity
   agrees with itself and disagrees with today: r213 lists `f260d1824ab3` for all three of its 2-device arms,
   r284 (`GREEDY-PURITY.md` §89, after the heat-map pin) lists `b08d3dbcb1e9` for all three, and today the pooled
   arms read `f260d1824ab3` against the resident one's `1e944569dbd6` -- i.e. a re-baseline moved one side and
   the assertion was never re-cut.  It also conflated the placement with the device count (text is a
   per-device-count reference under `-sm layer`), and the reference arm alone carries
   `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (a forced-sparse decode the pool arms do not set).  The section now asserts `pmlegacy == pmres` and `pmres == pres` (2-device pool text == 1-device pool
   text: the multi-device placement's real signature) and REPORTS the resident number; do not restore the
   `pmres == pmref` form, and do not quote pool-on text as byte-identical to pool-off.

## r301 (2026-11-16, `WORKLOG.md` 261116.008)

1. **A server step's cost is `a·tokens + b`, and `b` is what decides every admission policy.**  On the
   Flash-Next field shape `b ≈ 196 ms`; on a dense 4B it is `≈ 2 ms`; a pooled/streamed KV adds a per-STEP term
   that does not scale with the chunk at all (measured **1 144 ms vs 878 ms** per step for the same prompt with
   and without `--kv-attendance-pool`, 1.30x).  Consequences, all measured: `CHUNK = n_ubatch` is a floor and not
   a sweet spot; the duty cycle's prefill stretch is `1 + (K-1)·decode_step/prefill_step` and therefore spans
   1.03x to 16x for one `K`; and on a shape whose `b` dominates, **fair-share chunking cannot shorten a short
   prompt's wait at all**, because its 3 narrow steps cost what the baseline's 3 wide ones did (5.17 s vs 5.21 s
   on the pooled arm).  Do not judge such a shape as a failed fix -- the gate reports it as no-room-to-improve.
2. **A prompt-progress chunk is not a token arrival, and counting it collapses TTFT.**  With `return_progress`
   enabled, every step's `prompt_progress` chunk also carries `tokens: [0]` and an empty `content`, so a harness
   that counts "any of content/reasoning/tokens" measures the *arrival* (0.002 s) instead of the first generated
   token (0.052 s on the same request).  Both admission gates skip a chunk that carries `prompt_progress`; a
   pre-existing "count any of the three" rule is only safe without it.
3. **The single-slot greedy-text oracle needs a fresh server per arm on a pooled or streamed KV.**  The cells a
   request is given follow the page residency the earlier requests left behind, and FA's summation depends on the
   cells' tile indices, so a near tie flips between two arms that did nothing wrong: `[P1]` read
   `643670b3ea353558` after the interaction arms in one arm and `79bd7d7fd684b399` in the other, and
   `643670b3ea353558` on a fresh server in both.  This is `GREEDY-PURITY.md` §52's mechanism reached through a
   cache's *placement*, not through a resize -- do not "document away" a text oracle, give it a clean server.
4. **The duty cycle is safe to default on because it is capped in TIME, not in steps.**  The cap is the previous
   prefill step's own measured wall time, so the stretch is bounded at ~2x on any shape (measured 2.13x dense,
   1.58x streamed host-expert, 2.13x MTP, 1.10x field) while the decode burst is `K-1` tokens at the decode-only
   step time (16 -> 137 dense, 46 -> 200 MTP, 9 -> 109 streamed).  A step-shape duty does **not** cost the expert
   cache: `h` 0.7358 -> 0.7345, evictions 28 291 -> 27 157 on a really streaming arena -- `TODO` #317's
   wide/narrow warning is about a slab **band move**, which this is not.
5. **A block that carries a second feature is a structure debt, and the split is tree-identical.**  Block 12's
   admission amendment became `0016` by replaying the chain and reverting exactly the amendment's hunks in block
   12 (`git cherry-pick` + `git checkout <pre-amendment> -- tools/server/server-context.cpp`): blocks 00-11 and
   13-15 byte-identical, and the applied tree after 0016 identical to the unsplit tip plus the new change.  The
   `From` ids change (the chain is re-cut), the tree is the contract -- compare trees, never ids.

## r302 (2026-11-17, `WORKLOG.md` 261117.001)

1. **A qwen4exp prefill-logit point needs `ROCBLAS_USE_HIPBLASLT=0` on BOTH arms, and the trap reads like a
   real regression.**  A base recorded from the r301 build (default env) against a candidate run with the pin
   reads **mean KLD 0.019670 / same-top-p 96.627 %** -- 4x the limit, the exact signature of the issue #113
   class.  The **pre-change** build reads the identical 0.019670 / 96.627 % with the same mismatch, and with the
   pin on both arms the same pair is **0.000000 / 100.000 %**.  So the carrier is hipBLASLt (which selects a
   different Tensile kernel: a per-block score matmul reorders), not the change -- §44's rule, reached at a new
   point.  Record the base with the pin, not just the candidate.
2. **A fast path guarded by a device-side layout check has an invisible fallback, so its gate must assert a
   DELTA.**  The indexer top-k's regular-layout path validates the cache layout per launch and silently falls
   back to the cell-level path; every text and every KLD arm stays green in both states because the two paths are
   bit-identical by construction.  What separates them is time (`+2.22 %` prefill at ~84k all-VRAM, `+0.16 %` on
   the streamed shape) and the census (`GGML_QSA_TOPK_REGULAR_LOG=1`), so the gate asserts both -- a gate built
   on text can only ever prove "not wrong", never "still taken".
3. **The check kernel must read back through mapped pinned host memory, not a D2H copy.**  The check kernel is
   part of a captured graph, and neither a `cudaMemcpyAsync` D2H nor a stream sync is legal inside a capture, so
   the verdict is mirrored by the kernel itself into mapped pinned host memory (`cudaHostAllocMapped` +
   `cudaHostGetDevicePointer`) and the host prints the PREVIOUS launch's value: a census, not a barrier.  Cost is
   zero when unset (the pool slot is used and the census pointer is `nullptr`), and a `GGML_QSA_TOPK_REGULAR=0`
   arm reports 0 lines, which is what makes the census sensitive.
4. **Pointing a graded kernel at mapped host memory corrupts the measurement, not just the text.**  Passing the
   census pointer as the kernels' `reg` argument would put a PCIe/cache-coherent read of `reg->ok` behind every
   thread of every pass (a few thousand per launch), which is invisible in the text and fatal to the timing arm.
   The graded path keeps the pool-allocated device struct; only the check kernel writes the pinned mirror.
5. **The box is period-2 bimodal on this shape, so only an interleaved A/B is a measurement.**  The same arm read
   **2545 t/s** at ~24k in one phase and **1905 t/s** at ~15k in another (both arms move together, ~25 %), while
   the on/off delta stayed +1.3 %, +3.5 % and +2.0 % across those phases.  The gate's floor (1.0 %) carries 2x
   margin on the slow phase; never quote an absolute prefill rate from one run as a baseline.

## r303 (2026-11-17, `WORKLOG.md` 261117.002)

1. **A post-#31028 merged qwen4exp GGUF is a SECOND graph shape, and it was never exercised until r303.**  The
   converter writes `hc_{attn,ffn}_down` and `hc_*_inject` as one `blk.N.hc_*_down_inject` tensor, so
   `create_merged()` leaves `w_down == nullptr` and `build_hc_mix` takes its merged branch.  Every local
   Flash-Next GGUF is separate-tensor (the r141 note above says the merged path "needs a post-#31028 GGUF this
   fleet still does not have"), so `build_hc_mix`'s fused-mixer gate dereferenced the null `w_down` for six
   releases: **SIGSEGV at `sched_reserve`** (`graph_reserve -> resolve_fused_ops`), on a target and on an MTP
   draft.  Reproduce with `scripts/make-merged-hc-gguf.py` (row-concat the pair; the tool's output is
   byte-identical to the session's hand-built fixture) and gate with `scripts/gate-hc-fusions.sh MODEL.gguf`
   plus the `test-llama-archs -a qwen4exp` "merged HC" arm.
2. **A row view of the merged tensor IS the fused mixer's weight, so the layout does not have to lose the band.**
   The kernel addresses each weight row packed at `ggml_row_size(ne[0])`, and `ne[0]` is unchanged, so
   `ggml_view_2d(down_inject, n_in, n_lo, pitch, 0)` / `(..., hc, pitch, n_lo*pitch)` are the two weights; the
   buffer probe must use the tensor that OWNS the rows (a view has no buffer at graph build).  Worth **+11.4 %
   tg128** on the merged layout and **+15.8 %** on the separate one (same invocation, `-sm tensor` 3 GPU
   all-VRAM); merged-vs-separate is **-2.3 % decode, +9.5-10.1 % prefill**.  Its prefill logits differ from the
   separate layout by **0.017579 KLD / 96.863 % top-p** (one 324-row GEMM instead of 320+4), while single-token
   decode is bit-identical -- never use `merged == separate` on a prefill oracle, and do not use the pair as a
   regression oracle in either direction.
3. **`TENSOR_SKIP_IF_VIRTUAL` means "skip on the no-file path", which is why a synthetic fixture cannot reach a
   merged layout unless it NAMES the merged tensor.**  The flag is not "this tensor does not exist": a
   metadata-only load creates no merged tensor at all, so `test-llama-archs` covered only the separate shape.
   The loader now creates one the metadata names (real loads have files and never take that branch), and the
   fixture writes it -- which is what makes the CI arm possible.  Negative control: r302's `qwen4exp.cpp` makes
   that arm SIGSEGV.
4. **The HC gate must pin `ROCBLAS_USE_HIPBLASLT=0` and `MOE_EXPERT_HEATMAP=load` itself, and its default arm
   must not be `env -`.**  Unpinned, its seven fresh processes split into 449/459/447 chars; with the library
   pinned they are 7/7 byte-identical (r302 fact 1, reached again).  And `env -` is GNU `env -i`: the
   `fused_fused` (default) arm ran with an EMPTY environment, so on a pinned multi-device run it enumerated the
   iGPU and died with `ROCm error: invalid kernel file` while every other arm passed -- a gate whose baseline
   arm cannot run looks like a real fusion failure.
5. **The IQ4_NL hc class no longer shows byte-identical combine arms, and that is PRE-EXISTING.**  On
   Flash-Next IQ4_NL (chain mixer; hc weights IQ4_NL in all 48 layers) the `LLAMA_FUSED_HC_COMBINE` arms read
   446 `a5d0ec7b809d` while the chain reads 459 `9f8819c4c2c5`, with **identical shas on the r302 build**, so
   the divergence predates r303 and gate 9's hard-arm invariant (r148, `GREEDY-PURITY.md`) does not currently
   hold on that class.  PPL is identical in both arms (2.1640), so it is ULP-class, not degenerate: it needs
   an owner (`TODO` #341) rather than a waiver, and the class's arms must not be quoted as a pass until then.
6. **An upstream UI asset bundle can break every fresh SOURCE configure -- and only that one -- so the
delivery no longer fetches one at all.**  On 2026-11-17 `ggml-org/llama-ui`'s `latest` (75 files) had no
`workbox[hash].js`, so `ui-assets.cmake:137` aborted a configure that had no pre-built asset tree:
`gate-per-block-build.sh` (it wipes the build dir per commit), a clean checkout, the hosted CI tiers.
Incremental builds never saw it (the stamp caches the resolved bundle).  **Corrected at r328:** the delivery
builds with `LLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF` everywhere it does not ship a UI (the per-block
gate, `scripts/scratch-tree.sh`, `~/bin/build-llama-rocm`, `UI=1` opts back in), so `ui-assets.cmake` fetches
nothing and runs no npm -- it emits `UI: embedded 0 assets` and a 1.8 KB `libllama-ui.a` (an embedded-UI
build of the same tree is 3.1 MB from a 13.5 MB generated `ui.cpp`).  The `scripts/ui-dist-cache.sh` mirror and
the `HF_UI_VERSION` pin are retired with it: the failure mode is gone by construction rather than cached
around.  **The container images remain the one place a real UI is built** -- `.devops/rdna-rocm.Dockerfile`
builds it in a `web` stage (npm) and copies the result in before the cmake step, which is
`ui-assets.cmake`'s Priority 1, so a container build neither downloads nor validates the upstream bundle.
That stage still targets the pre-r328 `tools/ui/` layout and moves to `ui/apps/chat` when the re-base lands
(`TODO` #352).

## r304 (2026-11-17, `WORKLOG.md` 261117.003)

1. **A fusion whose reads are a *staged copy* may ignore an overlap with the tensor it was staged from -- and
   must not ignore it otherwise.**  The GDN K-quant epilogue replaces the generic fusion-window check for its own
   window because that check declines ~29 of 30 layers, and the declines are real: the allocator places the
   gate `MUL` output **inside `attn_norm-N`**, the F32 layer input the projections read (measured on every GDN
   layer of Qwen3.6-35B-A3B; `GGML_CUDA_GDN_EPI_DBG`-style instrumentation, r304).  That is harmless only because
   `ggml_cuda_mul_mat_vec_q` quantizes the activation to a **Q8_1 copy in a separate launch** and the matvec reads
   the copy.  So the invariant is: the written set must be disjoint from `dt`/`ssm_a` (weights) **and, when the
   activation's type is already `Q8_1`, from the activation** (no copy exists then).  An F16/BF16/F32 activation
   with no staging copy in the launch would make the same allowance a corruption -- check the kernel's read set,
   not its argument list.
2. **A fusion that consumes a reshaped chain needs one test arm per weight type, and the arm must be
   negative-tested.**  The r202 test covered BF16 + Q8_0 and would have passed a `2*ne01` row-count mistake in
   the F32 branch (the beta rows would overwrite the alpha store region and the *text* hash can still match if
   the store order hides it).  Dropping the doubling now fails 11/11 F32 shapes at maxdiff 0.7-1.4, which is what
   makes the F32 arm a control rather than a formality.
3. **The ksplit row concat's segment boundaries are already protected for these shapes, by `launch_ksplit`'s own
   one-row fallback.**  Its `nrows_x % rpb != 0 || nrows_x < rpb * GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS` rule forces
   `rows_per_block = 1`, and GDN `ssm_alpha`/`ssm_beta` have 32-48 rows -- far below `512 * rpb` -- so no block
   can straddle a boundary and the epilogue's per-block segment id is exact.  Do not "simplify" that fallback
   away: it is what the per-segment epilogue relies on, and the same reasoning is why the decode path only needs
   even row counts on RDNA4.
4. **`-ub 1` is not reachable from `llama-cli` on these models.**  `-b 1 -ub 1` aborts in `common.cpp`'s
   `get_sub_batch` (`failed to add token 0 at index 1`), unrelated to any fusion, so a gate that sweeps ubatches
   must start at 2 and take the 1-token prefill point from `llama-perplexity --ub 1` (`gate-prefill-logits.sh`).
   A gate that captured `llama-cli`'s failure output into a variable while incrementing its own failure counter in
   a subshell reported PASS with a failed arm -- run the failure accounting in the parent shell.
5. **The F32 and BF16 gate/beta arms share one mmvf launch but not one row-count rule.**  `mul_mat_vec_f_cuda`
   takes `gb && gb->gb_x2 ? 2*ne01 : ne01` on the BF16 *and* the F32 branch; only the BF16 branch had it before
   r304.  Anything that extends this launch to another `T` must extend the row count with it.

## r305 (2026-11-17, `WORKLOG.md` 261117.004)

1. **A host-expert routed `MUL_MAT_ID` is a GPU op at every ubatch width; `GGML_OP_OFFLOAD_MIN_BATCH` is not
   its gate.**  When the expert bank is in the delivery's host tier (`ROCm_Host` pinned master /
   `ROCm_HostMap` pageable master), neither premise of the generic threshold holds: the GPU arm uploads no
   whole table (the cache serves the routed set from the arena/pool or streams only the non-resident experts),
   and the CPU arm reads the *same* bytes out of the pageable mapping through a page fault per 4 KiB.  Between
   the cache band (16 on RDNA4) and 32 tokens the CPU arm was chosen and is **31x** worse (2.4 vs 76.2 t/s
   prompt eval, Flash-Next IQ3_S, 2 devices, a 19-token prompt, cold page cache), **20 %** worse even with a
   fully pinned master (**on a 9950X3D**, i.e. the best CPU arm this box can build), and **7.2x** worse with
   the cache off (a whole-table upload is still the cheaper GPU arm).  A 436-token prompt is identical in both
   arms, so the rule closes exactly that window.  `GGML_MOE_HOST_OFFLOAD=0` restores the old decision; a plain
   CPU-buffer expert bank (`--override-tensor 'exps=CPU'`) keeps the threshold, and the APU GTT store is a
   device buffer type that never reaches the decision.  Numbers: `benchmarks/2026-11-17-issue-323-host-expert-offload.md`.
2. **The 17..31-token window is not a benchmark shape -- it is the *server's* shape.**  `llama-cli` with a long
   ubatch never shows it (a 436-token prompt is inert under the change); what reaches it is a chat server's
   short prompt chunk, a follow-up turn's new tokens, or one slot's prefill step while another slot decodes.
   So a host-expert gate that only ever drives `llama-cli` at `-ub 1024` cannot see this class of regression:
   judge it with a short prompt (`<= 31` tokens) at the default `GGML_OP_OFFLOAD_MIN_BATCH`, which is what the
   reporter's trace was.
3. **An expert-bank buffer that survives as a per-device host buffer reaches the offload hook; one the loader
   merges into a per-layer Meta buffer does not.**  `ggml_backend_buft_is_cuda()` is a function-pointer identity
   on the DEVICE buffer name, so `ROCm_Host`/`ROCm_HostMap` slices are owned by no GPU backend and pass 1's
   `1.off` path decides them by `offload_op` -- that is how `-sm layer` and Flash-Next + `-sm tensor` were
   exposed while `Qwen3.6-35B-A3B-Q4_K_M -sm tensor` was not (its host master is merged into
   `Meta(ROCm0,ROCm1)#...`, so buffer ownership already assigns it to the GPUs).  Do not conclude from one
   model's `-sm tensor` arm that the offload decision is unreachable.

## r306 (2026-11-17, `WORKLOG.md` 261117.005)

1. **The admission chunk's cost is the UBATCH width, not the step count.** A capped step is a
   `chunk`-wide ubatch instead of an `-ub`-wide one, and the ubatch is the unit of work
   (`archive/work/session-scheduling/README.md` §4.2.1): on Flash-Next IQ3_XXS with host-resident experts
   (1 GPU, `-ncmoe 47`, an 8401-token prompt) a 512-token cap reads **683.7 t/s** against **1885.8** with no
   cap at `-b/-ub 16384` (2.76x), while at `-b/-ub 512` the same cap is **free** (826.5 vs 828.8 t/s on a
   5601-token prompt -- the two-point fit gives `b ≈ 0`, so the r300 record's `b ≈ 196 ms` is that shape's
   *wide-ubatch* fixed cost, not a property of host-resident experts as such). So "a host-expert arm will show
   the chunk penalty" is only true at a wide `-ub`; a `-b/-ub 512` arm measures ~0 and is not evidence about
   the field shape.
2. **A per-step cap must not be applied to a step that carries no decode work.** It bounds a decoding slot's
   inter-token latency and a newcomer's admission wait and nothing else, so r300/r301's unconditional cap
   charged half of the field shape's uncontended prefill (**1454 vs 3036 t/s** for the same 65536 tokens) for
   a bound nothing was waiting on. The contended-only rule is safe because a wide step can never carry a decode
   token -- the decode token is added before the budget is computed -- so a decoder's first *contended* step is
   already capped; only an in-flight uncontended step can delay it. The test must be "some slot is in
   `SLOT_STATE_GENERATING`", not `batch.size() == 0`: slots that cannot batch together still serialise
   (`can_batch_with()`), so a waiting decoder's slot can be left out of the step's batch.
3. **The number of concurrent PROMPTS is irrelevant to the rule, and a small cap is worst there.** The window
   is shared max-min fairly either way, so N prompts with nothing decoding are still the uncontended case; four
   ~7k prompts on a 4B (`-np 4`) go 177 slot-steps / 4.89 s with a 512-token cap against **46 / 3.40 s** with
   the whole window (1.44x), even though the same shape's *lone*-prompt arm shows nothing.
4. **A greedy-text oracle is not available on a host-expert shape without pinning the residency prior.** Six
   identical-config runs on Flash-Next `-ncmoe 47` (same tree, same `-b/-ub`, same chunk, same prompt, greedy)
   produced four different texts (three distinct hashes and a fourth length), across both binaries. That is
   `GREEDY-PURITY.md` §89's mechanism (residency selects the MoE kernel set) rather than the change under test,
   and this GGUF ships no `.moe-heatmap` for `MOE_EXPERT_HEATMAP=load`, so a text assertion on that shape needs
   a fully resident arena, repeats with a stated spread, or `scripts/gate-prefill-logits.sh` instead.

## r307 (2026-11-17, `WORKLOG.md` 261117.006)

1. **A take-over promise is evidence about ONE pass, not about a buffer address.** `input_cpy` is recycled by the
   scheduler across graphs and across the decode and prefill layouts, and a promise is not always consumed: a
   **captured** decode graph replays its launches without dispatching, so `moe_cache_get_table` never runs and
   `t.taken_cpy` is never cleared, and a pass that stages the copy through the gather/stage paths does not
   re-promise. So an address that still matches at a later pass says nothing about whether *that* pass skipped a
   copy, and reconstructing "the copy was skipped" from it is how 138 whole-table refills (206-450 MiB each,
   ~27.8 GiB, ~10.4 s) landed on every 30-34 token prompt (issue #327). The lifetime is now enforced where a pass
   ends (`moe_cache_pass_end`, at the end of every `graph_compute`), not at the promise sites: a promise that
   outlives its pass is *unreachable*, so a new promise site cannot reintroduce the hole.
2. **Which passes reach the repair at all is decided by the ARENA's residency, not by the prompt's width.** An
   identity (fully resident) table takes the take-over path, whose consumer serves it, so a short prompt on a
   fitting arena never touches the repair; the regression needed a **partial** arena (295 of 512 slots/table on
   the issue's shape), where `pf_in_place_locked` declines and the width guard refuses the table. A gate for this
   class therefore has to assert `stale > 0` (the promise path really ran) as well as `broken == 0`, or a fitting
   configuration passes it vacuously.
3. **A captured graph's non-dispatch is the general source of un-consumed promises, and it is benign.** On the
   issue's shape 6486 promises were dropped at their pass end over 92 decode steps (~70 per step): the decode
   graph replays its baked redirect and never asks the cache anything, so no repair is owed. Do not read `stale`
   as a defect count.
4. **r305 (issue #323) made this reachable, and that is not a reason to revert it.** Before it, a 17..31-token
   ubatch ran its routed MoE on the **CPU** and the GPU cache's hooks were never reached at that width; the fix
   is worth 31x on the prompt (`INVARIANTS.md` r305 item 1), so the second-order cost had to be paid off in the
   promise's bookkeeping rather than by moving the op back. A width-independent rule must be re-checked against
   every path it newly makes reachable.
5. **A block may not name a symbol a LATER block defines.** r305's `#323` relaxation added
   `ggml_backend_buft_is_cuda_host_expert()` to block 09 and identified the pageable master by comparing against
   `ggml_backend_cuda_host_map_buffer_type_name` — a function **block 12** defines (it is the block that adds the
   `ROCm_HostMap` buffer type). The tip builds, so nothing caught it: the per-block ladder is run with
   `--changed-from <previous tip>`, and block 09's own delta is unchanged since the release that introduced it, so
   blocks **09, 10 and 11 were never rebuilt** and had been uncompilable since r305. The predicate is now the
   pinned tier's identity in block 09 and the pageable tier's arm in block 12, which is where the buffer type lives.
   Two lessons for the ladder: a `--changed-from` run is a *delta* gate and says nothing about the trees whose own
   delta did not move (run the full ladder after amending an early block, or after touching a block that defines a
   symbol another block uses), and a symbol one block introduces for a later block's feature is the shape to grep
   for when a mid-chain tree fails to build.

## r308 — the fused-op PROBE answers a question about a graph shape, not about the backend

Established by the `--kv-unified` kv_rows defect (gfx1151 field report; `WORKLOG.md` 261117.007, block 05, gate
`scripts/gate-fused-probe-splits.sh`).

1. **A fused-op probe verifies PLACEMENT, so it is only valid if its graph contains the node.**  The reserve/probe
   context carries `n_kv = kv->get_size()` — the whole **padded** cache extent (`llama-kv-cache.cpp`'s reserve
   constructor), never the used depth — and a model's graph shape may depend on that extent: qwen4exp's decode arm
   is dense only while `n_kv < qsa_dense_decode_until` (32768 on gfx1151, unbounded elsewhere).  At a large `-c`
   the kv_rows probe therefore built the SPARSE arm, contained no rows node, and — with the descriptor's
   `require_observed = false` — reported "enabled" having verified nothing.  A probe that does not require its op
   is a probe of the graph's *current shape*, not of the backend.  `require_observed` is what turns "cannot
   verify" into "disable".
2. **Upstream's `ggml_flash_attn_ext_rows` (kv_rows) has no AMD implementation, so on this delivery it can only
   ever be disabled.**  `ggml_cuda_get_best_fattn_kernel`'s kv_rows branch returns `MMA_F16` only for NVIDIA
   turing-MMA with a mask; every AMD arch gets `NONE`, i.e. the node is placed on the CPU backend.  Enabling it
   costs one CPU attention launch per joint layer per step plus a device→host round trip of every K/V view.  The
   arm is still worth probing (the delivery claims portability), but a PASS there is a defect signal on AMD.
   **CORRECTED at r358 (issue #363): the AMD MMA kernel does read the rows form** (MFMA/WMMA with the
   GQA-optimised layout), so on this delivery the arm now runs on the GPU and `enabled` is the *correct* verdict;
   the defect this entry describes is the CPU PLACEMENT, which item 3's `splits` oracle still catches.  See
   `r358` below.
3. **A CPU-placed attention node is silent, so judge it by `splits`, never by text.**  The text is byte-identical
   (the CPU rows FA computes the same thing), there is no fault, and on an APU the K/V is already host memory, so
   the only visible symptom is a 3x decode loss.  `sched_graph_realloc: … splits = N` (N > 1) at `-lv 4` is the
   oracle; the r307 shape read **25** splits (one per joint layer + 1).  Cost measured at 15.5K depth, one active
   slot: plain 32.5 → 11.0 t/s, MTP `draft-mtp n3` 62-65 → 18.7 t/s (a speculative step is the same dense arm
   — its verify band is `n_tokens <= max(8, n_rs_batch)` — and the drafter's own unified context is a second
   victim).
4. **Only qwen4exp on gfx1151 at `-c >= 32768` can reach this vacuity, and that is a coverage fact, not a pass.**
   gfx1100/gfx1201 have `qsa_dense_decode_until` unbounded, so their probe always contains the node and declines
   by placement; a non-QSA model has no arm policy at all.  Verified with the flag: 35B-A3B Q8_0 90.2 t/s and
   27B Q8_0 21.9 t/s (both embedded MTP, `splits = 1`), and the gfx1100/gfx1201 policy forced on gfx1151 with
   `LLAMA_QSA_DENSE_DECODE_UNTIL=1000000000` (60.9 t/s with the drafter).  An explicit
   `LLAMA_QSA_DENSE_DECODE_UNTIL` below the context reproduces the vacuity on ANY arch.
5. **`llm_fused_op_probe` (and its siblings in `llama-context.cpp`) are initialized POSITIONALLY; the
   `/*.field =*/` designators are comments, not C++20 designated initializers.**  Appending
   `/*.require_observed =*/ true` as the fourth value initializes `n_seqs`, leaves the flag false, compiles
   cleanly, and measures nothing — r308's first cut did exactly that and the `enabled` line did not move.  List
   every field through the one you want, in struct order (the file's `kq_derived` probe is the model), and check
   the compiled constant when a flag "has no effect" (`.data.rel.ro` of the object, or a one-line census print).
6. **`--kv-unified` is not itself a decode cost.**  With the flag, `--kv-unified -c 32768` reads the same 32.5 t/s
   as `--no-kv-unified`; the whole difference in the field report was the mis-probed feature.  Do not "fix" a
   unified-vs-per-stream gap by changing the cache — check the probe census first.

## r309 (2026-11-17, `WORKLOG.md` 261117.008)

1. **A gate arm must pin EVERY switch it does not name, because one of them has a split-dependent default.**
   `cparams.fused_dsv4_hc_post = meta_split_active` (`src/llama-context.cpp`), so a `-sm tensor` run turns the
   DSV4_HC_POST op ON by default — and `build_hc_mix`/`build_hc_combine` take their DSV4 branch only on the
   CHAIN path, so `LLAMA_FUSED_HC_MIX=1`/`=COMBINE=1` makes those ops UNREACHABLE.  `gate-hc-fusions.sh`'s old
   reference arm therefore was not the chain on a tensor split: the `LLAMA_FUSED_HC_COMBINE=0` arms ran
   DSV4_HC_POST while the `=1` arms ran GGML_OP_HC_COMBINE, and the gate reported that difference as a combine
   disagreement (IQ4_NL 446 `a5d0ec7b809d` vs 459 `9f8819c4c2c5`, at r303 and identically at r302 — the
   split-dependent default predates r303).  With all four switches pinned, the true chain and every op arm on
   that class read **450 `bc46095cf4c6`** at `-sm tensor`: the combine op and the CUDA matchers ARE exact.  The
   general rule: an A/B arm that leaves a *defaulted* switch alone is two arms, and a reference arm must be the
   all-switches-off state, not "whatever the model defaults to on this split".
2. **The DSV4 HC pre/post ops and the fused HC_MIX op are numeric arms by measurement, not aliases.**  Both
   replace the same intermediate the chain computes, at a different reduction order: the pre/post ops' HC16 bf16
   producer slots are a documented re-baseline (**0.013217 / 96.392 %** and **0.013198 / 96.186 %** vs the chain
   on the BF16-hc class, re-measured **0.015568 / 96.451 %** over 20 x 512 wikitext), and the fused mixer's
   inject half is the fused band dot (`hc_mix_down_dots`) where the chain's is `build_lora_mm(w_inject, xn)`,
   i.e. **mean KLD 0.000000 / same-top-p 100.000 %** — rounding level, and yet enough to flip a greedy near-tie
   (GSQ-IQ3_XXS `-sm tensor`: chain 449 `d62340fbf566`, mixer 449 `6a102dbc62d6`; under `-sm layer` the two
   coincide).  So a text hash is NOT a valid oracle for "the mixer fusion is inert", and an anchored arm per
   (class, split state) is the honest instrument.  The convention defect issue #228 was about (a /hc in the
   consumer) is a ~15-logit drift and still fails every arm.
3. **A generated text is a per-split AND per-class-state reference, so record anchors per state, not per class.**
   The `defaults` arm (no env at all) is the cheap way to know which state a run is in: it must equal `all_ops`
   (a Meta device is present, `fused_dsv4_hc_post` auto-ON) or `ops_no_post` (one device / `-sm layer`).  Made
   the gate print it and pick its anchor set from it, so the same script is anchored on `-sm tensor` and
   `-sm layer` without an operator-supplied hint — and a change to the auto rule becomes a FAIL instead of a
   silently different arm set.

## r310 (2026-11-18, `WORKLOG.md` 261118.001)

Established closing `TODO` #344 on the Flash-Next IQ3_XXS host-expert shape (1 GPU, `-ncmoe 47`; record
`benchmarks/2026-11-18-host-expert-prefill-oracle.md`).

1. **A host-expert shape is a text oracle once TWO states are pinned, and the heat map is not the only one.**
   With `MOE_EXPERT_HEATMAP=load` **and** `ROCBLAS_USE_HIPBLASLT=0` the shape reads one text 3/3 (and 3/3 with
   the prior off entirely), but with the library left at its default it reads a *different* one 4/4
   (`296/91ef8940d66e` off vs `302/fb8da5c09068` on).  So §44's per-start-library rule reaches a host-expert text
   arm too: pin the library **and** the prior, and report the repeats as a spread.  A text arm that pins only the
   map is comparing two states and will call it a change.
2. **The r306 spread does not reproduce in ANY state we can reconstruct, and the eliminations are the finding.**
   Twelve states, 42 runs: no spread *within* a state, and the only carrier that moves the text *between*
   states is the BLAS library.  Measured at
   `-c 16384 -b 2048 -ub 2048` (the r306 arm's 5-step shape; partial arena, `h = 0.4593`, 6 629 fills): a
   **mutating** heat map (default `=1`, 5/5), the **frozen** prior (`load`) and **no** prior (`=0`) all read the
   same text (the map is a placement prior and the pool covers the set, so §89 needs a *partial* arena — 5/5 at
   `h = 0.46` too); a **forced 8 GiB** arena and the 23.8 GiB auto arena read the same text (which also kills the
   "free VRAM varies, so the library picks another kernel" theory); and the **r305** build (`73295b281`,
   pre-r307, pre-r306's admission change) reads byte-for-byte what the r309 build reads, 6/6 each.  So the r306
   six-run spread was **not** the prior, the arena, the residency breadth, or r307's repair.  The GGUF now also
   ships `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS.moe-heatmap` (the r75 unified name), so the r306 note "this GGUF
   ships no heat map" no longer holds either.  What was *not* reconstructable is that session's prompt and box
   state, and a greedy near-tie flip (invisible to a hash, `GREEDY-PURITY.md` §46/§61) stays the residual
   explanation — which is why the coverage is now **mechanical** (`scripts/gate-host-expert-text.sh`): it repeats
   the pinned arm N times and fails on any spread, instead of a human noticing four texts in six runs, and the KLD
   point is the numerics verdict a near-tie cannot fool.
3. **The r306 admission rule's altered case is text-identical on this shape, measured across builds.**  The cap is
   unreachable on an r306+ binary (that is the fix), so the A/B is two binaries: r305 `73295b281` admits a
   5246-token prompt in 12 steps (`512 x 9, 122, 512, 4`) and r309 `87d3629df` in 5 (`2048, 2048, 634, 512, 4`) at
   `-b 2048 -ub 512`, and both read `98/65ba627f8eca`, 2/2 each.  The wide step really is a *step* change here,
   not a ubatch change: a lone prompt's ubatch boundaries are the consecutive 512 blocks either way.
4. **The point's identity includes its env, so the gate records it (`--pin`).**  `gate-prefill-logits.sh` now
   applies a `K=V,...` list to BOTH arms, folds it into the base file name and writes it to the meta as `pins=`,
   and refuses a compare whose pins differ.  Without it a base recorded under one library/heat-map state is
   found and compared under another — on this shape a text *and* kernel-set difference.
5. **The tier has a usable KLD instrument: self-KLD `0.000000` / `100.000 %` across two processes.**  The point is
   `--model .../GSQ-IQ3_XXS...gguf --device 0 --pin "MOE_EXPERT_HEATMAP=load,ROCBLAS_USE_HIPBLASLT=0" --extra
   "-ncmoe 47 --load-mode dio --lazy-mode mlockr -ot per_layer_token_embd.*=CPU"` (base PPL 2.8553, base sha256
   `effe46712734…`, ~35 s per pass, peak RSS 71.1 GiB — it needs the box to itself).  The r305 build against that
   base also reads `0.000000 / 100.000 %`, so the r305..r309 delta (including r307's repair on the prefill path)
   moves nothing on 40 x 512 chunks of host-expert prefill.  The 27B release point is dense and never reaches the
   pool/in-place transport at all, which is why this point is now the required arm for a change to that tier.

## r311 (2026-11-18, `WORKLOG.md` 261118.003) — the fused-op probes: a verdict that is not in the log is not a verdict

Established by `TODO` #345 (the r308 vacuity CLASS), block 05's descriptor set and `scripts/gate-fused-probe-splits.sh`'s
fixture ladder.

1. **`require_observed` is the wrong instrument for the probes that remain, and a bare `enabled` line is the defect.**
   Four descriptors carry `query_supports_op` and now say HOW they decided: `...(the probe graph contained the node
   and its placement was checked)` or `...(the probe graph built no such node; the acting device accepts the op,
   verified by ggml_backend_dev_supports_op)`.  The measured reachability audit (fixture ladder, `test-llama-archs -a
   <arch>`, gfx1201):

   | arch | HC pre | HC comb | HC post | top-k mask |
   |---|---|---|---|---|
   | `qwen4exp` | obs (fixture) / **query** (real GGUF) | **query** | **query** | **query** |
   | `deepseek4` | obs | obs | obs | obs |
   | `deepseek41` | obs | obs | obs | **query** |
   | `glm5-next` | obs | obs | obs | obs |
   | `kimi-k31`, `deepseek32`, `dots3note`, `glm-dsa`, `mistral-large4` | **query** | **query** | — (no tensor arm) | obs |

   `require_observed = true` on any of them would have DISABLED a working feature: `deepseek41` builds `TOP_K_MASK`
   at runtime (`deepseek41.cpp`'s compressed-attention path) but not in its 1-token probe graph, and qwen4exp builds
   `DSV4_HC_PRE/POST` only above `HC_FUSED_MAX_TOKENS`/`HC_BF16_MAX_TOKENS` (its single-token arm runs the fused
   `HC_MIX`/`HC_COMBINE` band).  For `DSV4_HC_POST` that would be a NUMERIC regression, not a perf one: the unfused
   chain reads NMSE 1.74e-01 under the Meta device (r309).
2. **The direct query is faithful exactly when the backend predicate reads the SOURCE TYPES, and no other probe can be
   built at all.**  `ggml-cuda.cu`'s `DSV4_HC_PRE/COMB/POST` cases are type checks on F32, and `TOP_K_MASK`'s is a
   mask-type check (F16/F32), so a minimal all-F32 (resp. F16-mask) op answers the real question.
   `ggml_cuda_lightning_indexer_generic_supported()` is **shape**-dependent (`q->ne[0] == 128`, a 64/32/4 head width),
   so no single probe shape answers it for both dsv4 (64 heads) and qwen4exp (4): that descriptor stays inert
   (`auto_flid = false`) and the models gate on their own geometry (`qwen4exp.cpp`'s
   `lightning_indexer_op_supported()` queries the real per-layer shape).  Every backend also has a generic fallback
   for the indexer op, so an unsupported device there is SLOW, not wrong.
3. **Under `-sm tensor` the query IS the split-safety condition.**  `model.dev_layer(il)` is the Meta device, whose
   `supports_op` is `all_of()` over the tensor-parallel children, so one query per distinct acting device answers for
   every layer and every device at once (the same argument as `qsa_op_supported()`).  Query each distinct device
   once, not once per layer.
4. **The gate's `splits` bound is a property of the LAYOUT, not of the probe.**  `ggml_backend_sched` counts one
   segment per backend boundary: 1 on a single device and on a tensor split (the Meta backend is ONE backend), and
   **N+1** on an N-device layer split — measured 4 at NGPU=3, because the output layer is placed back on the first
   device.  The r308 form (`splits > 1` fails) can only pass on a 1-GPU or tensor-split arm, and the r308 gate's
   `plain`/`mtp` arms are also box-dependent by construction: they leave `LLAMA_QSA_DENSE_DECODE_UNTIL` at the box
   default, which is the SPARSE policy on gfx1151 (`32768`) and the DENSE policy on gfx1201 (unbounded), so `plain`
   declines by OBSERVATION on halo and by PLACEMENT on the R9700 box.  Pin the expectation (`KVROWS_EXPECT`) or
   derive it; never encode one box's kernel policy in an assertion.
5. **`test-llama-archs -a <arch>` is the cheap per-probe oracle, and it has two known gaps.**  It builds each arch's
   graph from GGUF metadata (no model load, ~2.4 s/arch, real context → `resolve_fused_ops()` runs, plus a 64-token
   prefill and decode), which is how the whole reachability table above was measured on one box.  Gaps: **no fixture
   for `xing4-0` / `hy-v4`** (both build DSV4-HC / LIGHTNING_INDEXER nodes; no GGUF of either on this fleet), and a
   fixture's tensor **types are F32/F16**, so qwen4exp's `hc_mix`/`hc_combine` fused band does NOT fire there and its
   `HC_PRE` reads `obs` while the real GSQ-IQ3_XXS/IQ4_NL GGUF reads `query` — the real-model arm is the only
   instrument for that vacuity.  Its graph also legitimately reads `splits = 4..14` (tiny shapes fall back), so the
   splits assertion is a real-model-only check.

## r312 (2026-11-18, `WORKLOG.md` 261118.005) — the verify band's low columns are at the DRAM roof, so a "narrower tile" has nothing to win; the crossover is TYPE-shaped

Established by `TODO` #329 (the family's lowest columns), block 07's `mmvq-wmma.cu`, and a DRAM-cold one-`MUL_MAT`
sweep plus end-to-end `llama-batched-bench`.  `archive/work/rdna4-wmma-verify/LOW-BAND-2026-11-18.md` carries the
raw tables.

1. **At 2-4 verified columns the K-split `mmvq` kernel is 92-98 % of the DRAM roof, so NO kernel can beat it there.**
   One dense `MUL_MAT` (`m=88064`, `k=5120`; 253 MB Q4_K … 479 MB Q8_0, i.e. larger than the 64 MB L2 so it streams),
   one R9700 at its 644 GB/s spec: Q4_K reads **617.2 / 616.9 / 609.4 / 590.6 GB/s** at widths 1 / 2 / 3 / 4 (96 / 96 /
   95 / 92 %), Q5_K 624 / 620 / 613 / 595, Q6_K 626 / 623 / 619 / 615, IQ4_XS 621 / 618 / 612 / 605, Q8_0 630 / 627 /
   624 / 621.  The theoretical headroom in the whole low band is therefore 0-8 %, and the tile family lands 3-8 %
   behind it.  A "narrower tile or dot-product form for the low band" (the `TODO` #329 premise) cannot change that:
   the reference is the memory wall, not a kernel.  Ties to the same fact from the whole-model side: at 2 columns the
   27B UD-Q4_K_M step reads 34.6 ms against 30.2 ms at 1 column, i.e. 476 against 548 GB/s -- the 2nd column is
   nearly free for the K-split kernel because it is already bandwidth bound.
2. **The crossover to the WMMA tile family is type-shaped, because each type's K-split kernel leaves the roof at its
   own width.**  Same sweep, tile vs K-split (µs/op): Q4_K is **-1.5 % at 4 and +6.7 % at 5**, Q5_K -1.5 % at 4 and
   **+13.0 % at 5**, Q6_K **-7.3 % at 2 … -0.4 % at 7 and +5.5 % at 8**, IQ4_XS **-2.5 % at 2 … +0.3 % at 8**, Q8_0
   parity until **+4.4 % at 8**.  A single uniform floor is therefore wrong in both directions: a floor of 7 still
   paid Q6_K's -0.4 % and never reached the K-quants' 5- and 6-column wins.  `auto` is now that measured per-type
   floor (**5 / 5 / 8 / 8 / 8** for Q4_K / Q5_K / Q6_K / IQ4_XS / Q8_0) and an explicit `N` stays the uniform
   calibration arm.  End to end (27B UD-Q4_K_M, `-ntg 256`, TG t/s at widths 5 / 6 / 7 / 8): 1 GPU
   **131.70 / 147.28 / 158.74 / 169.68** off against **132.46 / 151.62 / 168.79 / 187.26** `auto` (+0.6 / **+2.9** /
   +6.3 / +10.4 %), 2-GPU `tensor` **204.82 / 231.33 / 250.56 / 271.51** against **204.33 / 234.11 / 260.12 /
   289.21** (-0.2 / +1.2 / +3.8 / +6.5 %); against the old uniform floor of 7 that is +2.9 / +1.5 % at width 6 and
   neutral (+/-0.3 %) elsewhere.
3. **Two adjacent ideas are measured and REJECTED; do not re-derive them.**  (a) Dropping the Q4_K/Q5_K min-term
   ("ones") MMA -- half of those types' tile MMA count -- is worth only **+2.2 % at 2 … +5.0 % at 8** for Q4_K,
   **+0.6..0.8 %** for Q5_K and **0.0 %** for Q6_K/IQ4_XS/Q8_0 (measured by zeroing the min term, which is wrong
   output, so timing only), i.e. at most +4-5 % on one type's already-winning widths for a new per-column `Σxq`
   precompute; it does not move the crossover (Q4_K would still be -1.1 % at 2 and -0.7 % at 3).  (b) Letting the
   wave count be a non-divisor of the super-block count (`k=3840` -> `nsb=15`, which today falls through to ONE
   32-thread wave) is **±1-2 % with mixed signs**: 547.3 -> 541.6 GB/s at 8 columns for Q4_K (worse) but 583.4 ->
   597.7 for Q8_0 (better), so the exact-divisor rule stays.
4. **The family is reached by ANY dense matvec in the band, not only by a draft's verify batch.**  A plain
   `llama-cli` prefill on a 4B Q8_0 (`-ub 512`, 12 tokens) dispatches the family **639 times of 2419 matvec ops** at a
   floor of 2, all before the first decode step, and its 40-token greedy text moves (`ed4ee91ec5b6` off against
   `1b72606b6cf9` on) -- which is why the option's warning is about the greedy contract, not about drafting, and why
   the release's purity evidence is the OFF arm and an explicit floor, never "the ON arm only runs at verify".
   Both are unchanged by r312: the 27B Q8_0 + DFlash2 `n-max 7` spec arm reads `56612340f420` off on the r311 and
   r312 builds, and an explicit `GGML_CUDA_WMMA_MMVQ=7` reads `b7d28482ca5f` on both.

## r313 (2026-11-18, `WORKLOG.md` 261118.007) — the Q8_0 hc mixer: a launch-count change, and what an "identical output" claim has to be tested against

**Changed.**  Official block 11 only: the fused Q8_0 `GGML_OP_HC_MIX` (every unsloth UD qwen4exp file — a
BF16-hc model takes `hc_mix_bf16_*` instead) now runs 3-4 launches per hc site instead of ~8, via persistent
`down_dots` counters, a planar output, a Q8_1 epilogue, `hc_mix_combine_rms_gamma_q8`, a weight-preloading up
band, an up+collapse fused launch and the F32 inject rows on the down grid (contributor PR #326, five switches,
`ENVIRONMENT.md` §7.7).  **+7.1 % decode** (45.1 -> 48.3 t/s) on `UD-Q4_K_XL` at 2 devices `-sm tensor`, +4.9 %
after a 5246-token prompt, prompt rate unchanged; the contributor's +7..10 % on a different UD file holds.

1. **"Byte-identical to stock" is not the invariant; byte-identical to the *chain* is.**  The submission's own
   arm set is stock-fused vs patched-fused, which is the right test for a *claim of no change* and it does hold
   (all ten `gate-hc-fusions.sh` arms read the r312 hash in the same arm, on a model neither side had).  But the
   delivery's contract for this op family is that each fused arm reproduces its own **unfused chain**
   (`ggml.h`, `gate-hc-fusions.sh`'s HARD arms), and on a fused-mixer class the MIXER half cannot meet it: its
   inject half is the fused band dot where the chain's is the lora GEMM (r309, `LLAMA_FUSED_HC_MIX` is a numeric
   arm).  What a change *inside* that family must preserve is the ARMS' own text, which is why this release was
   gated on the ten-arm table and not on a single greedy hash.  A patch that only ever compares fused-vs-fused
   would pass while breaking the combine op, and one that only checks a handful of tokens would pass while moving
   the verify band.
2. **A default-on Q8_0 fusion needs a Q8_0 unit arm, and the arm has to be proven to reach the kernel.**
   `test_hc_combine_mix` was BF16-only, so `hc_mix_combine_rms_gamma_q8` had no unit-level oracle at all; the
   Q8_0 arms added here are only worth anything once they are shown to *bite* — perturbing the kernel
   (`xn * 1.5`) must fail exactly the Q8_0 arms, and it does (and the same perturbation of the BF16 kernel fails
   exactly the BF16 arms).  A `1.001` perturbation does NOT bite: the case's `max_nmse_err` is 1e-3, so the
   falsification has to be larger than the tolerance (`INVARIANTS.md` r225/r238: a silent fusion makes an arm
   vacuous — this is the same trap one level down).
3. **A WIP handover can be promoted without a rebase, but not without a re-read.**  `hc-mix.cu` was byte-identical
   between the PR's base (r304) and r312, so the three patches applied with `git am -3`; the fold into block 11
   needed exactly one resolution, and it was a *later* block's refactor (`planar_env` became a lambda in block 14),
   not the PR's own code.  Verify the touched surfaces' ownership first: every file this PR edits was already
   carried by `0011`, which is what let it be a block amendment instead of a new block.
4. **Measured and rejected at review — do not re-derive.**  The two redundant `_Q8` sub-switches
   (`GGML_CUDA_FUSE_HC_COMBINE_MIX_Q8`, `LLAMA_HC_MIX_PLANAR_Q8`) are inert in both directions: the switches
   they duplicated already covered the Q8_0 class, and folding them changed no text on any arm (so their
   existence was a knob-count cost, not a mechanism).  Their per-piece A/B value is preserved by the five
   switches that remain.
5. **Untested, stated rather than implied.**  A merged-layout (`hc_*_down_inject`) GGUF with Q8_0 hc weights:
   no such file on this box, and the synthetic merged-HC fixture is F32/`hc_lr 8`, so it never reaches the fused
   op.  NVIDIA/CUDA, gfx1151 and Windows are likewise untested (the files' kernels are 32-wide with
   `warp_reduce_*<32>` like the rest of the file).  `TODO.md` carries the limitation; `LLAMA_FUSED_HC_MIX=0` is
   the reference for anyone holding such a file.

## r314 (2026-11-18, `WORKLOG.md` 261118.009) — the band for 128-dim odd-GQA heads, and why a test on the KV *cache size* is pure while a test on the used length is not

**Changed.**  Official block 05 only (contributor PR #329, @overdoingism): the RDNA4 decode/verify band also
takes 128-dim heads whose GQA ratio is not a multiple of 8 (5, 6, 7, 9..12), in its existing 128-dim
`ncols1 <= 4` instances and its 32-column block for every query width -- no new config rows, no new
instances.  Outside the band an odd ratio folds *nothing* (`ncols2` = the largest power of two dividing the
ratio, capped at 8), so a GQA 7 decode read each K/V head once per query head.  Two things changed from the
submission: the arm is **on where it pays** (`ggml_cuda_fattn_band_wmma_h128_applies()`), and the 2-D launch
is guarded on the compiled instance.

1. **`K->ne[1]` at the FA op is the layer's KV CACHE SIZE, not the used length.**  *(r315 correction: it is the
   PADDED USED LENGTH -- `min(cells.size(), max(256, GGML_PAD(used, 256)))`, measured 256 at d0 and 5376 -> 5632
   while generating; the cache-size values below are the graph-RESERVATION pass.  The rest of this item's
   reasoning, the attribute's usefulness and every number in it stand; the crossing cost is r315 item 3.)*  Measured on the fork's
   graph (a one-shot print in the band predicate, ricursive 65B): with a 51-token prompt the global layers
   read `K->ne[1]` = 8192 at `-c 8192` and 16384 at `-c 16384`, and the windowed layers read **1536** =
   `GGML_PAD(min(size_base, n_swa + n_ubatch), 256)` = `1024 + 512` (`llama-kv-cache-iswa.cpp`).  The value
   is fixed for the run.  This is the *same* attribute the existing wide-band gate already reads
   (`GGML_HIP_FA_BAND_WIDE` -> `K->ne[1] >= 4096`), so it is not a new kind of decision.
2. **... which is exactly what makes a kernel-family test on it width-pure, and a test on the used length a
   purity break.**  A verify batch moves the used length by up to `W-1` rows, and the padded form
   (`K->ne[1] % 256 == 0`) can then differ between a decode step and its own verify (used 8170 -> pad 8192
   vs used 8178 -> pad 8192 is stable, but 8190 -> 8192 vs 8198 -> 8448 is not), so a "decline below N cells"
   gate keyed on the used length would put a decode and its verify on different kernel families --
   `GREEDY-PURITY.md` invariant 1 and §36.  Anyone adding a "depth" gate to a band arm must use the cache
   size; there is no way to key on the used length without breaking the band.
3. **The band's per-layer cost does not amortize on a short cache, and the crossover is `cells x passes_saved
   ~ 20k`.**  Measured on gfx1201 (q8_0 K/V, decode width, band / tile-kernel time per op): ratio 5 (5 -> 1
   passes) 1.26 at 2048 cells / 0.94 at 4096 / 0.65 at 8192 / 0.51 at 16384; ratio 7 (7 -> 1) 1.05 / 0.72 /
   0.48 / 0.37; ratio 10 (5 -> 2) 1.70 at 2048 / 1.24 at 4096 / 0.96 at 8192 / 0.81 at 16384; ratio 12
   (3 -> 2) 1.83 / 1.45 / 1.14 / 0.95 at the same points.  Every row crosses at 16-25k, hence
   `K->ne[1] * ceil(ratio/ncols2_tile) - ceil(ratio/8) >= 20000` (`GGML_HIP_FA_BAND_H128_MIN_WORK`).
   Verify widths tilt the other way (band already wins at 1024-2048 cells), which is why the rule cannot be a
   flat cell count: it would enable ratio 12 at 8192 (a 1.14 loss) or deny ratio 7 at 4096 (0.72).
4. **A windowed/hybrid model is the case that matters, and it is a loss at EVERY depth, not at depth 0.**
   ricursive 65B (64 of 80 layers windowed at 1024), 3-GPU `-sm tensor`, interleaved `A B B A`,
   `ROCBLAS_USE_HIPBLASLT=0`, tg64: unconditional band **-5.8 % d0, -5.1 % d4096, -4.9 % d16384, -4.5 %
   d32768**; with the work gate **0.0 % / 0.0 % / -0.9 % / -0.5 %** (the residual is the 16 global layers,
   whose op-level win is only 0.77 at ratio 10 and does not survive a weight-bandwidth-bound 65B).  The
   submission documented this as "a loss below 4-8K cells, -5 % at depth 0"; the depth framing is wrong --
   the *layer's* cache is what decides, so a hybrid model pays it at any depth.  `-c 2048` (auto declines
   every layer) reads the r313 text byte-for-byte (`c1e00039b55c`).
5. **The 256-dim arm has the same short-cache property and was left alone.**  q8_0, decode width, band/tile:
   3.94 at 512 cells, 2.05 at 1024, 1.70 at 2048, 1.51 at 4096 for GQA 6; 2.94/2.18/1.94/1.97 for GQA 8.  It
   is a published default since r6 and moving it changes kernel selection for every GQA 5..12 256-dim model,
   so it keeps its behaviour: only the 128-dim arm is gated, and the follow-up is scoped in `TODO.md` #345
   (measured: -4.6 % at d0 and -1.5..-1.7 % at d4K..d32K on a weight-bound 3-GPU 27B decode, `benchmarks/2026-11-18-fa-band-h128.md` 5).
6. **The submission's own numbers reproduce, and its oracle has a hole this release closes.**  Apply on the
   r313 tree: clean `git am`, zero conflicts.  `test-backend-ops -o FLASH_ATTN_EXT` 7333/7333 in three arms
   (band, `H128=0` -> tile, `GGML_HIP_FA_BAND_WMMA=2`); width purity PASS, worst maxdiff 0, with f16, bf16 and
   q8_0 on ricursive 65B (bf16 is **not** in the PR's cases, and it is the band's third native K/V type), and
   PASS again in the mixed per-layer configuration the gate creates (windowed layers on tile, global layers on
   band, P=8192).  Op-level claims matched (ratio 7 kv 4096 nb 1: 69.9 -> 50.4 us against their 68.1 -> 49.6;
   ratio 5 kv 45056 nb 8: 1794 -> 378 against 1789.5 -> 356.9).  KLD `--family-switch` on ricursive 65B
   (`--ub 4`, band vs tile) **0.001013 / 99.412 %** against a same-kernel control of 0.002222 / 98.676 %,
   i.e. the band's own rounding is smaller than a plain ubatch change.  Coherence unchanged on 4B / 27B /
   35B-A3B (`8d89080db673` / `da2e2d192e21` / `ae39ec41bc1e`), four MTP axes acceptance-identical to r313 on
   the dense 27B (352/474, 349/485, 370/422, 321/564).  **The hole**: with the gate, the PR's short-cache
   cases take the tile kernel, so the band's oracle is the same suite run with `GGML_HIP_FA_BAND_WMMA_H128=1`
   (`MANIFESTS.md` gate 05); the new kv-32768 loop is what reaches the band in a default run.
7. **Not validated.**  a 128-dim *non-SWA* odd-ratio model with a real context (the submission's Nanbeige /
   Qwen2.5 case): no such file on this box, so the win is quoted from the contributor and corroborated only at
   the op level.  gfx1100/gfx1151 and Windows are untested for this arm.

## r315 (2026-11-18, `WORKLOG.md` 261118.010) — a band WORK gate has to key on the padded used length, and that costs one crossing; the 256-dim arm gets the gate, an adaptive `P` does not work

**Changed.**  Official block 05 only (`TODO` #345): `ggml_cuda_fattn_band_wmma_applies()` now gates the
**256-dim** head class (27B GQA 6, 35B-A3B GQA 8, qwen4exp GQA 12) with
`K->ne[1] * max(passes_tile - passes_band, 1) >= GGML_HIP_FA_BAND_H256_MIN_WORK` (default 20000), i.e. band
from 10000 cells for GQA 6 and from 20000 for GQA 8 and GQA 12.  Knobs: `GGML_HIP_FA_BAND_WMMA_H256`
(`auto` / `0` / `1`) and `GGML_HIP_FA_BAND_H256_MIN_WORK`.  The shared pass computation moved into
`ggml_cuda_fattn_band_passes_tile()` / `..._passes_band()` so the 128- and the 256-dim gates cannot disagree.

1. **`K->ne[1]` at the FA op is the PADDED USED LENGTH.**  This corrects r314 item 1: probe-printed on the 27B
   at `-c 8192`, a 51-token prompt reads 256 for every decode step, a 5246-token prompt reads 768/1280/…/3840
   during its ubatches then 8192 (the multi-device prefill pad) and 5376 → 5632 while generating.  The
   8192/16384 values r314 reported come from the graph-**reservation** context, whose constructor sets
   `n_kv = kv->get_size()`.  `llama_kv_cache::get_n_kv()` is
   `min(cells.size(), max(pad, GGML_PAD(used, pad)))` with `pad` 256 for a decode/verify batch.
2. **It is still the right attribute, and r314's measurements are not affected.**  It is exactly the number of
   KV iterations the kernel executes (`iter_k = ceil(ne11/nbatch_fa)`), which is what a per-launch cost has to
   be amortized over — and r314's own d0 row reads 0.0 %, which only a used-length key produces (a
   cache-size key would have banded those global layers at d0 and read -0.9 %).  Never key on `Q->ne[1]`.
3. **What it costs: one crossing per run, where a verify batch can straddle a 256-cell boundary.**
   `test-logits-width-probe` at P=9980 (cells 9984 → tile for W=1..4, 10240 → band for W=5..8) **FAILs with
   maxdiff 0.327**, while every constant-family arm passes with maxdiff 0.  This is inherent to any monotone
   function of the used extent (a shift or a quantization of the threshold only moves the crossing), it is
   already shipped for the 128-dim arm since r314, and the alternative — the cache size — reintroduces the
   shallow losses whenever `-c` is large.  `GGML_HIP_FA_BAND_WMMA_H256=0/1` pins the family.  Recorded in
   `GREEDY-PURITY.md` §96/§97.
4. **The 256-dim band is a verify-width kernel.**  Per op (head 256, 4 K/V heads, band ÷ tile, best `P`):
   GQA 6 decode width 2.64 (q8_0) / 1.96 (f16) at 512 cells, 2.13/1.59 at 1024, 1.30/1.89 at 2048, 1.26/1.48 at
   4096, 0.96/1.09 at 8192, 0.66/1.10 at 32768; verify width 8: 1.14/1.24 at 512, then 0.76/0.71, 0.39/0.47,
   0.31/0.40, 0.25/0.33, 0.18/0.25.  GQA 8 and 12 lose at the decode width just as hard (2.46 / 2.54 at 1024
   cells) even though GQA 8 folds whole outside the band, which is why the gate's `max(…, 1)` floor gates that
   ratio by cells alone instead of disabling it (its deep verify row measures 0.45 at 8192 cells).
5. **End to end, the gate is a win on every 256-dim arm this box validates, except one MTP arm.**  27B tg64
   44.20/44.84/44.38/43.48/41.91 → 46.13/45.65/45.18/43.55/41.90 t/s at d0/4K/8K/16K/32K (+4.4/+1.8/+1.8/0/0),
   35B-A3B 130.12/134.63 → 139.72/138.82 (+7.4/+3.1), Flash-Next 72.04/72.70 → 74.25/74.35 (+3.1/+2.3),
   35B-A3B `draft-mtp n3` 234.5 → 239.3 (+2.0), 27B `draft-mtp n3` at a 5.2K prompt 87.0 → 86.0 (**-1.1 %**, the
   band is worth ~1 % there); the deep arm keeps the band (27B `draft-mtp n7` at a 20.5K prompt 79.0 → 78.9).
   Every changed arm reproduces the previous build's all-tile arm **byte for byte** (35B-A3B
   `ae39ec41bc1e` → `265c381dbc11` = its `GGML_HIP_FA_BAND_WMMA=0` text; the four MTP axes R `7dca501eeb0e` /
   `759c1a4515c6`, P `fcc0bdeb636b`, C `eb89102cb7d8`, K `a87c4318b649`).
6. **An adaptive `P` does not work, on either half of its premise.**  At depth there are no idle blocks
   (32768 cells / 64 = 512 iterations against `P` = `nsm` = 32); at a short cache the idle blocks are free (a
   block owning no iteration writes a neutral partial and exits) and a smaller `P` serialises the KV
   iterations (512 cells: P8 52.6 µs, P16 31.1, P32 39.2, P64 56.6, tile 15.8 — `min(nsm, ceil(cells/64))`
   picks 8, the worst band value).  It is also not keyable (item 1).  `P` = 16 is the optimum in 30 of 36 rows
   and the quantized default `P = nsm` is 10-20 % off it — a separate tuning item, since it moves every banded
   model's numerics.
7. **Not validated.**  gfx1100/gfx1151 and Windows were not exercised for this arm (selection-only change on
   kernels those classes already use, with the same suite as its oracle); no non-SWA 256-dim model was
   available, and the class's members on this box are weight-bound below the gate, which is its purpose.

Record: `benchmarks/2026-11-18-fa-band-h256.md`, `MANIFESTS.md` gate 05, `patches/README.md` block 05,
`ENVIRONMENT.md` §8, `GREEDY-PURITY.md` §96/§97.

## r316 (2026-11-18, `WORKLOG.md` 261118.011) — the band's fixup period `P` is ONE constant (`nsm/2`) for every K/V type: the widths disagree, and the end-to-end arm decides

**Changed.**  Official block 05 only (`TODO` #346): `ggml_cuda_fattn_band_wmma_split()` returns
`max(2, nsm/2)` for every K/V type (`P` 16 on gfx1201, `nsm` 32).  The native-quantized types kept `nsm` from
r92 to r315; `ggml_cuda_fattn_band_wmma_two_byte()` and the launcher's `P` fallback are gone.
`GGML_HIP_FA_BAND_WMMA_SPLIT` still overrides both families for measurement.

1. **`P` is a width compromise, and the optimum differs by a factor of two between the widths.**  One R9700,
   `test-backend-ops perf`, band forced, `GGML_HIP_FA_BAND_WMMA_SPLIT` 8…96, q8_0, medians of 3, 256-dim GQA 6:
   at `n_q` 1 the minimum is **16** (kv 16384: 92.9 against 99.5 at 32 and 104.0 at 8; 32768: 151.2 / 163.9 /
   180.9; 102400: 425.0 / 447.5 / 558.2), at `n_q` 8 it is **8** and monotone in `P` (kv 32768: 269.5 / 279.7 /
   318.4 / 344.9 / 401.1 for 8/16/32/48/96).  The same holds for the 2-byte family (f16, `n_q` 8, kv 16384:
   144.9 / 172.5 / 201.2 / 254.7).  So a *single* shipping constant is necessarily suboptimal for one width,
   and `P` must not be keyed on `n_q` (it would split decode and verify differently).  `P` 8 was measured end
   to end and rejected (below).
2. **The quantized default `nsm` was 5-14 % off everywhere the band runs, and `nsm/2` is the better constant.**
   kv 16384-102400, `n_q` 1: 92.9 vs 99.5, 106.1 vs 112.4, 151.2 vs 163.9, 425.0 vs 447.5 µs; `n_q` 8:
   170.2 vs 179.7, 279.7 vs 318.4, 817.3 vs 840.2.  The only loss is the short cache (1024 cells, `n_q` 1:
   55.0 vs 42.6 -- reproducible to ±0.1 over three runs), which every band gate keeps off (10000 cells for the
   256-dim class, 5-9k for the 128-dim one), so it is recorded rather than avoided.
3. **End to end it pays only on the attention-bound arm, and there by ~4.5 %/step.**  3 GPU `-sm tensor`, q8_0
   K/V: plain 27B tg64 43.63/42.41 against 43.52/42.61 t/s at d16384/d32768 and Ripple-65B 26.07/25.42 against
   25.96/25.44 -- noise, because a 27B Q8_0 decode is at the DRAM roof (r315 item 5).  The 27B `draft-mtp n7`
   arm on the 20.5K prose prompt reads 88.9 against 75.8 t/s (**+17.1 %**) and 79.2 against 74.9 at ~10.5K
   (**+5.8 %**).  The decomposition matters: the 20.5K row's mean accepted length moved 3.26 -> 3.82, i.e. 30.3
   -> 26.6 steps for the same 128 tokens, so **~4.5 %/step is the kernel** and the rest is the acceptance
   re-roll any numeric-order change carries.  Never quote the token-rate delta of a greedy speculative arm as
   the kernel's win.
4. **`P` 8 wins the op-level verify rows and loses every end-to-end arm.**  27B tg64 d32768 40.86 t/s, the
   10.5K MTP arm 78.33, the 20.5K one 75.68 -- all worse than P16, and worse than the old P32 on the plain
   decode row.  A real MTP step's verify width is not the `n_q` 8 op case.
5. **The r133-era comment claim ("verify cost flat for P 48..96 at kv 20K..200K, degrading below 32") does not
   reproduce; it was a DECODE-width, long-KV reading.**  At verify widths the curve is monotone (f16 256-dim
   kv 32768, `n_q` 8: 289.7 / 312.3 / 337.3 / 385.2 for 8/16/32/64).  What is flat above 24 is the decode
   width at a long KV (q8_0, kv 102400, `n_q` 1: 467.2 / 447.5 / 457.2 / 467.4 / 491.5 for 24/32/48/64/96),
   and "degrades below 32" reproduces there (558.2 at 8).  Both comments now match the measurement.
6. **Both band gates stay correctly calibrated, and their decisions are unchanged in cell terms.**  With P16 the
   band/tile ratio is 0.85 at 16384 cells and 1.07 at 8192 for the 256-dim GQA 6 class (gate: band from 16384);
   0.88 at 8192 and 1.06 at 4096 for 128-dim ratio 10 (gate: 8192), 0.62 at 8192 (ratio 5), 0.60 at 4096
   (ratio 7), 1.06 at 8192 / 0.91 at 32768 (ratio 12, gate: 32768).  Every `TILE` decision is still a measured
   loss and every `BAND` decision still a win; P16 only improves the margins.
7. **A numeric-order constant is width-pure by construction, and that is what makes it cheap to gate.**  A
   constant `P` cannot straddle: `test-logits-width-probe` reads **PASS maxdiff 0 on 37 arms** (both head
   classes, q8_0/f16/bf16, 1 GPU + `SPLIT=tensor` + `SPLIT=layer`, P 3584/4096/10240).  Prefill KLD: canonical
   0.000000 / 100.000 %; `--family-switch` at a *banded* ubatch (`--kv q8_0 --ub 4`) 0.002733 / 99.243 %
   (27B) and 0.000823 / 98.981 % (Ripple-65B) against same-kernel `-ub` controls of 0.004182 / 98.822 % and
   0.004133 / 97.772 % -- i.e. **less** divergence than a benign ubatch reordering.  The arm has to be `-ub 4`:
   a default-ubatch prefill batch is not banded and the change is invisible to it.
8. **Text moves exactly where the band runs and nowhere else, and every moved arm reproduces a pinned
   alternative.**  f16 K/V coherence is byte-identical to r315 on all five models.  With q8_0 K/V at a 21K
   prompt the 27B moves `f7d157b1e07e` -> `774d1c325068`, which is **the r315 build's
   `GGML_HIP_FA_BAND_WMMA_SPLIT=16` arm** byte for byte, and `SPLIT=32` on the new build reproduces the old
   default -- the override still pins.  35B-A3B and Ripple-65B do not move at that depth (P16 == P32 there).
   The four MTP axes are byte-identical to r315 on bf16 K/V (acceptance 348/486, 357/460, 368/427, 372/375
   reproduced to the digit) and reproduce r315's own q8_0 arms under all three `P` values, because those axes
   sit below the band gate.
9. **Not validated.**  gfx1100 / gfx1151 and Windows were not exercised; `P` is a fraction of `nsm`, and the
   kernels are the ones both classes already ran.  A 128-dim GQA 5/7/12 *model* was not available (op level
   only).  **Open:** the 2-byte family's own optimum at verify widths is 8 (it is 12-16 % faster there) and it
   still ships 16 -- one constant cannot serve both widths, and P 8 loses every end-to-end arm measured here.
10. **`plain != draft-mtp` on a 3-device `-sm tensor` split is NOT the quantized-KV fold family, and it is not
   cache-type specific** (recorded 2026-11-18 in `WORKLOG.md` 261118.012, tracked as issue #330; the r316
   records first guessed the fold family and that guess is refuted).  Disabling all six fold switches
   (`GGML_CUDA_KV_FOLD_Q8_0`, `GGML_CUDA_QK_ROTATE_HADAMARD`, `GGML_CUDA_FUSE_V_FWHT`,
   `GGML_CUDA_FA_COMBINE_FWHT`, `GGML_CUDA_QK_PAIR_FWHT`, `GGML_CUDA_FUSE_ROPE_FWHT`) leaves both texts byte
   for byte unchanged, and a **bf16** cache shows the same divergence on the reasoning axis
   (`7dca501eeb0e` plain / `759c1a4515c6` draft, r315's own published pair).  It needs **3 devices on
   `-sm tensor`**: equal on 1 device, on a 2-device tensor split and on 3 devices `-sm layer`.
   `GGML_CUDA_ALLREDUCE=internal` changes both texts but does not remove the divergence.  `P` is not involved
   (identical at `GGML_HIP_FA_BAND_WMMA_SPLIT` 16 and 32 on the r315 build), and
   `test-logits-width-probe` reads PASS maxdiff 0 throughout because it drives its own batches rather than a
   speculator's verify.  Do not attribute a plain/draft text divergence to the KV type or to a fold without
   the kill-switch arm.

Record: `benchmarks/2026-11-18-band-fixup-period.md`, `MANIFESTS.md` gate 05, `patches/README.md` block 05,
`ENVIRONMENT.md` §8, `GREEDY-PURITY.md` §98, issue #330.

## r317 (2026-11-19, `WORKLOG.md` 261119.001) — every fused MoE launch resolves the expert cache, or the take-over's skipped `input_cpy` is read stale (GitHub issue #294 closed)

**Changed.**  Official blocks 09 and 10 only (`TODO` #327's fused-launch contract, completed).  The routed
Weight fusions in `ggml-cuda-try-fuse.cu` (the 5/7- and 11/13-node `mmvq` subgraphs, the per-lane scale/bias
pairs for `mmvf` and `mmvq`, the 3-node `mmvf` gate+up+GLU, the NVFP4 scale fold and both bias-only arms) and
the routed readers in `mmvq.cu` (the shared-expert pair and the MoE down fold) now resolve their lanes through
`moe_cache_fusion_serve()` instead of launching on the graph's tensors.  `MOE_CACHE_SERVE_MIXED` stands the
fusion down (the per-op chain resolves each table itself); the staged branch calls `moe_cache_repair_copy()` for
every lane, exactly as the per-op dispatch does.  The decode band got its own census
(`decode MoE fusions: mmvf_glu=…/… mmvq_glu=…/… (in_place/staged) mixed_lanes=…`, kinds 4/5).  No environment
variable and no default changed.

1. **A take-over promise and a fused launch do not meet.**  `ggml_backend_sched` asks
   `moe_cache_take_over()` before it fills a split input; a true answer skips the `input_cpy` copy and leaves
   the expert bytes only in the arena.  The per-op `MUL_MAT_ID` dispatch honours that by redirecting
   (`moe_cache_get_table`), but a fused launch never enters the dispatch, so every fused routed reader that
   was not redirecting read a buffer the scheduler deliberately never wrote.  The reporter's tiny GGUF shows the
   read side exactly: routed `up` all zero, one `gate` expert valid per device, first bad tensor
   `ffn_moe_down-1` after a finite `hc` input; their branch bitmask isolated `ggml_cuda_mul_mat_vec_f` (the
   BF16-expert gate/up path), and the same class was present at ten of the twelve routed fused sites.
2. **`MIXED` is not `NONE`, and only the per-op chain may consume it.**  The take-over is per TABLE, so one
   lane of a gate/up pair can be served in place while the other is staged.  A decline therefore re-fills the
   copy (`moe_cache_repair_copy`, a no-op when no promise is outstanding) and a `MIXED` answer stands the
   fusion down.  Before this, the report's `mmvq_glu` column could not even say whether the fused read went
   through the arena.
3. **A fully resident arena can read the stale copy and still be correct, which is why this box never saw the
   corruption.**  With every table identity, the staged copy is the WHOLE table and, once written, stays
   byte-correct for the life of the allocation, so the raw read lands on the same bytes (measured: base and
   patched are byte-identical on the 35B-A3B `-sm tensor` full-residency arm while its census reads
   `mmvq_glu=3/0`, 256/256 slots, `takeover=30`).  The failure needs a copy that was never written -- a new
   capture/allocation, which is the mixed-architecture tensor split the reporter runs and this box cannot build.
4. **The mechanism is exercised and text-neutral everywhere it can run here.**  Same-seed greedy text base vs
   patched, byte-identical: 4B dense 3-GPU `7972f0185e01`, 35B-A3B Q8_0 3-GPU `-ncmoe 34` (full-residency
   arena) `7399489b1685`, the same model 2-GPU `-sm layer` `7399489b1685`, Flash-Next IQ3_S 2-GPU `-sm tensor`
   `14f3577be465` and `-sm layer` `3e2d3910b121`, Qwen3-Coder 30B 2-GPU `-sm tensor` `bdf34a354857`.
   `gate-host-pool.sh` **59/59** (the take-over/repair gate), prefill-logit **0.000000 / 100.000 %**, MTP MoE
   protocol on 35B-A3B (`draft-mtp` 227.0 t/s against plain 118.2, acceptance 0.75110 = 1370/1824, identical
   both builds; fusion-off acceptance 0.76062 within noise), `llama-batched-bench -npl 1,4,8` flat, per-block
   ladder **18/18**.
5. **The reporter's patch is a correct subset, not the whole contract.**  It redirects five sites with
   `moe_cache_redirect_fused()` (v1) and falls back to the graph tensor on a decline; that is safe for the exact
   gate+up pair only because the fusion guard allows that pattern only when BOTH tables serve.  The per-lane
   scale/bias arms are misclassified by that guard as the down fold, so v1's fallback could read a lane nobody
   filled there -- which is why the shipped form uses the per-table `MIXED`-aware query and repairs instead.

Record: `WORKLOG.md` 261119.001, `patches/README.md` blocks 09/10, `MANIFESTS.md` gates 09/10, issue #294.

## r318 (2026-11-19, `WORKLOG.md` 261119.002) — a fusion that replaces a node's output has to count readers THROUGH VIEWS, and a LoRA is such a reader

**Changed.**  Official block 10 only (GitHub issue #331): `ggml_cuda_find_mul_q8_1_matmul()` counts with the
view-aware `ggml_cuda_count_uses()` and expects `1` (the matmul reads the MUL directly) or `2` (a no-op reshape
plus the matmul through it).  `GGML_CUDA_FUSE_TRACE=1` gains `FUSE_TRACE MUL_Q8_1 DECLINE ne=… readers=…
want=…`.  No kernel changed, no default moved, no new environment variable.

1. **The Q8_1 MUL fusion writes only the matmul's Q8_1 blocks; the F32 MUL buffer is never written.**  So its
   guard is a *reader* count, and the count must be the view-aware one: the matmul reads a `RESHAPE` of the MUL
   (mmvq wants a 2-D activation), so a direct-only count cannot see anything else that reaches the MUL through
   that reshape.  `ggml_cuda_count_uses()` (the helper the neighboring matchers already use) counts `src == t`
   *and* `src->view_src == t`; the LoRA-free graph therefore reads exactly 2 in the reshape form and 1 in the
   direct form, which is what makes the fix free for every existing arm.
2. **`build_lora_mm()` appends a `MUL_MAT` on the same activation, so every LoRA on a projection whose input is
   that MUL was silently dropped.**  The chain is `sigmoid(z) -> MUL(norm, ·) -> RESHAPE -> MUL_MAT(ssm_out)` in
   a GDN layer, and `silu(gate) * up -> MUL_MAT(ffn_down)` on a model that builds the GLU as UNARY+MUL; the
   adapter loads, `/lora-adapters` reports the scale, and the model behaves as if it were absent.  Measured on
   the reported shape (Qwen3.8-27B `UD-Q4_K_XL`, 1 GPU, seed 42, temp 0, 64 tokens, the published
   `abliteration-lora.gguf`): fused `f3e8501e7b38`, `GGML_CUDA_DISABLE_FUSION=1` `f7026ffb399a`, without the
   adapter `b7c2006e41dc` -- i.e. the fused LoRA arm was not the unfused one.  After the fix the fused LoRA arm
   reads `f7026ffb399a`, byte for byte with the unfused chain.
3. **A GLU node is NOT exposed to this bug, which is why only one of the reporter's three modules moved.**
   `ggml_cuda_glu_q8_1_mark()`'s launcher writes the F32 GLU *and* the Q8_1 blocks, and the prefill
   `GGML_CUDA_FUSE_SWIGLU_MMQ` arm requires `nodes[i+1]->src[1] == node` (a direct reader), so the graph's own
   use count catches the adapter's second reader and the fusion declines by itself.  The census agrees: the
   declines are all at the ssm_out activation (`ne=4096` on the 4B, `6144`/`12288`/`24576`/`258048` on the 27B)
   with `readers=2 want=1` (direct) and `readers=3 want=2` (through the reshape).
4. **The count is inert for a LoRA-free graph, and that is checkable rather than assumed.**  Five no-LoRA
   coherence arms are byte-identical to r317 (4B 1 GPU and 3 GPU `-sm tensor` `7e6229b260f6`, 27B 1 GPU and
   3 GPU `-sm layer` `1acb04bd9104`, 35B-A3B `-ncmoe 34` `461ca8cd0e88`), the same arms with
   `GGML_CUDA_FUSE_TRACE=1` log **zero** `MUL_Q8_1 DECLINE` lines, prefill KLD is `0.000000 / 100.000 %`,
   `gate-fusion-alias-copy.sh` keeps its anchor `3fab4da87774`, MTP acceptance on 35B-A3B 3 GPU `-sm tensor`
   is identical to five decimals (0.76374, 695/910, mean len 3.29) and `llama-batched-bench -npl 1,4,8` is
   flat.  A reader count that changed a LoRA-free decision would move one of those.
5. **The test adapter is synthetic on purpose.**  A published LoRA exists for exactly one model, so a gate
   using one could not run on the models this box validates: `scripts/make-test-lora.py` reads the model's own
   tensor shapes and writes rank-2 `lora_a`/`lora_b` for the requested projections from a fixed seed
   (`adapter.lora.alpha = 0`, i.e. scale = the adapter scale, like the published file).  Two traps are in that
   script and cost a load failure each: gguf-py writes a numpy shape **reversed** into the ggml dims (so
   `ne0 = in` is a `(rank, in)` array), and an adapter that carries `blk.<block_count-1>…` is refused with
   "does not exist in base model" because the appended nextn/MTP block is not addressable by name.

Record: `WORKLOG.md` 261119.002, `MANIFESTS.md` gate 1b + the block-10 row, `patches/README.md` block 10,
`ENVIRONMENT.md` `GGML_CUDA_FUSE_TRACE`, `scripts/gate-lora-fusion.sh`, `scripts/make-test-lora.py`, issue #331.

## r319 (2026-11-19, `WORKLOG.md` 261119.003) — a decode/verify KV split must be a run constant, not a function of the padded KV extent

**Changed.**  Official block 05 only (`ggml/src/ggml-cuda/fattn-common.cuh`) plus the width probe's new boundary
arm (block 10, `tests/test-logits-width-probe.cpp`, the block that last touched that file).  GitHub issue #330:
`--spec-type none` and `--spec-type draft-mtp` emitted different greedy text on a 3-device `-sm tensor` split.
No kernel arithmetic changed, no default moved; one environment variable added (`GGML_CUDA_FA_SPLIT_CONST`,
kill-switch, default on).

1. **`K->ne[1]` at the FA op is the PADDED USED extent, and a verify batch is `W-1` cells ahead of the decode of
   the same row.**  `llama_kv_cache::get_n_kv()` returns `min(cache, max(256, GGML_PAD(used, 256)))` and is called
   *after* `apply_ubatch()`, so a W-row batch at position `p` makes the extent `pad(p + W)` while the plain decode
   of that row makes it `pad(p + 1)`.  The two differ only in the W-1 cell window at each 256-cell multiple — which
   is exactly where issue #330 diverged (generated position 5630: the verify's extent 5888 against the plain's
   5632; the top-2 logits were bit-identical over the preceding 373 positions).
2. **The tile FA launcher's KV split `parallel_blocks` IS the summation order, so keying it on the extent breaks
   width purity.**  `gridDim.y = parallel_blocks`; block b owns KV iterations b, b+P, b+2P, … and
   `flash_attn_combine_results` sums the P partials in block order.  The efficiency search stops one below
   `ntiles_KV` whenever two candidate counts print the same integer efficiency (measured: `ntiles_KV` 176, P 175
   against 176; 525/576 and 528/576 both read 91 %), so P = 175 put iteration 175 in block 0 and added it before
   iterations 1..174.  Measured logit move: up to **0.232868** (q8_0) / **0.196008** (bf16) in the width probe's
   boundary arm, and a 0.04 near-tie flipped in the generated text.
3. **The fix is `min(P_const, ntiles_KV)` for `Q->ne[1] <= 8`, where `P_const` is the search's one-full-wave value
   (`nsm * max_blocks_per_sm`), and it is exact by construction.**  Deep KV: P fixed, so a longer extent only
   appends iterations that are fully masked for earlier rows (exact no-ops) and the combine order is fixed.
   Shallow KV: P = `ntiles_KV`, one iteration per block (ascending order), with the trailing blocks neutral
   (`M = -inf` -> `expf(-inf) = 0`, added as exactly 0).  The `ntiles_KV = P_const` crossing is exact because the
   round-robin split is one iteration per block there too.  The RDNA4 band kernel already pins its own P
   (`nsm/2`, r316) for the same reason; this is the tile path the band declines below its work gate.
4. **Only the 3-device `-sm tensor` LAYOUT's text moved, and it moved onto the ascending order.**  Post-fix
   manifest A plain == draft `c1f9d0b3f5b6` and B `2fbc183348a8` (3 GPU tensor, 27B Q8_0, `-n 512`); the old
   plain pair (`120c74935967` / `7dca501eeb0e`) is reproduced exactly by `GGML_CUDA_FA_SPLIT_CONST=0` on the same
   binary.  In A the plain arm moved onto the draft's text; in B both arms moved (the draft `759c1a4515c6` ->
   `2fbc183348a8`) because the old search keyed *every* band batch on its own padded extent.  4B 1 GPU
   `-sm layer` `be6407e421c8`, 27B 2 GPU tensor `b1c84ba23a2c` and 27B 3 GPU `-sm layer` `f59aa0be6536` are
   unchanged (the last two are issue #330's own published hashes).
5. **The class is not device-count specific; the LAYOUT is what exposes it.**  The width probe's boundary arm
   fails pre-fix on the 3-device tensor layout (`ndst_eff = 3`, `bpw = 576`) and passes pre-fix on the layer
   layout (its `ndst_eff` makes the old search pick `ntiles_KV`).  A gate for this class must therefore run a
   tensor split *and* a prefix that straddles a 256-cell multiple: the probe's flat arm at a 256-multiple prefix
   (256/4096/10240) puts every width in one padding block and cannot see it.
6. **The split was never the parallelism.**  tg128 on 3 GPU tensor is flat at every depth to 131k
   (d0 46.6, d8k 45.6, d32k 42.3, d65k 39.3, d98k 36.8, d131k 34.6 both arms) and `pp512` is identical: at deep KV
   the pinned value is the split the old search already returned whenever it did not trip the off-by-one.
7. **The gate is `scripts/gate-spec-purity.sh`.**  Manifest A (q8_0/prose) and B (bf16/reasoning) must read one
   text, with a non-vacuity check (draft Generation t/s >= 1.2x plain) and a `--control` arm that runs the pre-fix
   split and requires it to DIVERGE.  It also runs the probe's flat + boundary arms.  The four-axis MTP protocol
   was run on 1 GPU at `-n 3000` and on 3 GPU `-sm tensor` at `-n 1000` on BOTH cache types; plain == draft on
   every axis with healthy acceptance (0.53656/0.80437/0.86819/0.99200 and 0.57755/0.81776/0.86555/0.99200).

Record: `WORKLOG.md` 261119.003, `benchmarks/2026-11-19-fa-split-boundary-purity.md`, `GREEDY-PURITY.md` §99,
`ENVIRONMENT.md` `GGML_CUDA_FA_SPLIT_CONST`, `MANIFESTS.md` gate 36, `patches/README.md` blocks 05/10 (the probe),
issue #330.

## r320 (2026-11-19, `WORKLOG.md` 261119.004) — a fusion's overlap test must test the buffers the kernel READS, and a deferred gather makes the state node dead (PR #316)

1. **`m.state` is not always the state the conv reads.**  `gdn_conv_norm_match()` folds the GDN q/k l2 norm pair
   into the direct conv kernel, and it refuses a layout where a SCALE output overlaps `x`, `w`, the conv output
   or the conv state, because the kernel writes the normalized outputs while it still streams those buffers.
   With a **deferred conv-state gather** (`ggml_cuda_gdn_conv_gather_deferrable`) that last test is wrong:
   `ggml_cuda_gdn_conv_direct` reassigns `state` to `root->src[0]->data`, indexes it with the gather's row id
   (`state_ids`), and never dereferences `m.state->data`.  The matcher now takes the context and, exactly then,
   tests the **cache tensor** (`root->src[0]`) instead of the dead `GET_ROWS` node.
2. **The allocator does place the q output there, so the refusal was real and reachable.**  The q SCALE output is
   `128*nk*T` floats against the state's `3*C`, so the spans collide from `T = 6` (27B: 20 480 B per row against
   122 880 B).  A census in the matcher on r318 read `T=6,7,8 deferred=1 base=overlap_state` on
   `test-logits-width-probe` (`KV=q8_0`), `T=6,7,8` on `--spec-type draft-simple`, `T=6,9` on `ngram-cache`,
   `T=16` on `ngram-map-k` and `T=8/11/16` on `ngram-mod --spec-ngram-mod-n-max {7,15}` -- i.e. every
   non-store speculator verify arm at a width of 6 or more loses the norm fold in every GDN layer.
3. **It is inert in every arm that takes the rollback store.**  `--spec-type draft-dflash` (the submitter's own
   scenario), `draft-dflash,ngram-mod` and `draft-mtp-adaptive` n9 on 35B-A3B run `GDN_CONV_STORE ...` and never
   enter the matcher at all: that kernel already writes `q_out`/`k_out`.  So the PR's motivating observation
   ("every T=8 DFlash2 verify call refused") was true of its 2026-11-03 measurement tree and has been false since
   **r244** (2026-11-09, PR #292 / upstream #31660), six days before the r289 cut the PR was made against.
4. **Nothing changed the reach between r289 and r318.**  `gdn-conv.cu`, `gdn-conv.cuh`, `gdn-conv-store.cu` and
   `gdn-conv-store.cuh` are byte-identical between the two trees (materialised and diffed), and the only
   GDN-related delta in the dispatcher is the unrelated gate/beta epilogue fusion.  The r289 review's "no
   production arm reaches it" was too strong for a different reason: it widened `ngram-mod` with
   `--spec-draft-n-max`, but the ngram family's width knob is `--spec-ngram-mod-n-max`, and a dense model with no
   arena is never band-clamped, so `ngram-mod` does reach `T=8..16`.
5. **The win is real but tiny, and that is the decision.**  5 interleaved pairs, 1 GPU, 27B UD-Q4_K_M,
   `ngram-mod --spec-ngram-mod-n-max 15 --spec-ngram-mod-n-match 45`, `-n 2048`, greedy: 38 692.4 ->
   38 685.6 ms (**-0.018 %**, inside the ±0.03 % wall-clock resolution), against 39 262.4 ms with the fold
   forced off (+1.47 %, an upper bound that also moves the `T=1` decide).  The refusal is rare because the
   ngram draft is 10-15 tokens deep only occasionally; `ngram-map-k` agrees (67.1 vs 67.1 t/s against 66.0
   fold-off).  So it ships as a matcher correctness fix with a measured ~0, not as a decode win.
6. **A matcher-decision change moves no hash, so the probe arm needs a census to be non-vacuous.**  The width
   probe reads `width_purity=PASS` and byte-identical `W=1..8` hashes **both** before and after (dense 27B and
   MoE 35B-A3B, `P=3584` and `P=4096`, and against `GGML_CUDA_DISABLE_CONV_FUSION=1` and each fold kill
   switch).  The evidence that the arm is not vacuous is the matcher census above (`base=overlap_state` ->
   `folded`), which is the same instrument the r289 review needed.

Record: `WORKLOG.md` 261119.004, `GREEDY-PURITY.md` §100, `patches/README.md` block 10, `ENVIRONMENT.md`
`GGML_CUDA_FUSE_GDN_CONV_NORM`, PR #316.

## r321 (2026-11-19, `WORKLOG.md` 261119.005) — a fused MoE launch's STAGED answer is a PLACEMENT, not a stand-down, and it owns the broken-promise repair (issue #332)

1. **A single-lane caller is served exactly when its ONE lane is.**  `moe_cache_fusion_serve()` is asked per lane,
   and five call sites pass `gate == nullptr` (the decode `MUL_MAT_ID` + weighted-`MUL` down fold, the three
   single-lane routed bias/scale folds, the RDNA3_5 MMQ `swiglu -> down` fold).  Written as
   `gate_ok = gate == nullptr || lane_serves(gate)` that term is vacuously true, so a STAGED table answered
   `MOE_CACHE_SERVE_MIXED` (one lane "served", one not) and every gate-less fusion **stood itself down** for as
   long as its table was staged: residency selected the kernel set again, which is the impurity `TODO` #327 removed
   for the gated sites.  The answer must be `gate == nullptr ? up_ok : lane_serves(gate)`.
2. **The reach is "neither served in place nor served by the published slot-remap", not "staged".**  Measured on
   the r320 build: 35B-A3B UD-Q3_K_M, 1 GPU, `-ncmoe 24 -c 8192` reads `mmvq_glu=166/0 mixed_lanes=112`; the same
   with a 5 246-token prose prompt reads `121/0 mixed_lanes=80` (`pair=0/176` in both); with
   `MOE_EXPERT_TAKEOVER_FAULT=1` (every promise broken inside its own pass) it reads `0/0 mixed_lanes=200`.  A
   **small** arena (`MOE_EXPERT_CACHE_MIB=2048`, `-ncmoe 40`) and the reporter's Flash-Next IQ3_XXS 2-device
   `-sm tensor -ncmoe 47` shape both read `mixed_lanes=0`, because the eager tier serves every table it holds slots
   for — which is why the report could only find this by reading the code.
3. **The STAGED answer must be REPAIRED, and the serve query is the only place that can own it.**  A take-over
   promise can be broken by a mid-pass eviction between the hook and the launch, and a fused launch never passes
   `ggml_cuda_mul_mat_id`'s per-op repair.  `moe_cache_fusion_serve()`'s NONE answer therefore re-fills every
   staged lane it just read; the three site-local repairs the delivery had (`mmvq.cu` ×2, `fuse_serve_routed`)
   are removed and `gate-moe-promise.sh` still reads `broken=<n> repaired=<n>` with `repaired == broken` and text
   `c0e7e91d84c4`.  The MMQ shared-quantize pair, the MMB routed triple and the two MMQ triples had **no** repair
   at all, so a both-lanes-staged state with a broken promise would have read an unfilled buffer; that state was
   not observed to move text on this box (the 216 repairs of the fault arm happen on other paths), and it is now
   covered by construction rather than by each site remembering.
4. **The newly-fused launches are bit-identical to the per-op chain they replace.**  112/80/200 launches move from
   stand-down to fused with the text unchanged: `ba65a38cf871` (35B-A3B 1 GPU `-ncmoe 24`), `cd5e36218bd2` (5 246-
   token prose, with and without `MOE_EXPERT_TAKEOVER_FAULT=1`), `ecc56835cf49` (Flash-Next 2 x R9700
   `-sm tensor`), `a3a1a32d95c2` (35B-A3B `plain` == `draft-mtp`), `efdb7608a8c9` (35B-A3B 2-device `-sm tensor`,
   `plain` == `draft-mtp`), `7e6229b260f6` (dense 4B, 1 and 3 GPU).  MTP acceptance identical to the digit on all
   four Protocol A axes (0.67763 / 0.76233 / 0.86491 / 0.99467), prefill KLD 0.000000 / 100.000 %.
5. **A fix that touches a block-10-owned contract belongs in block 10, even when the line it edits is block 09's.**
   The first cut amended block 09 (which writes `moe_cache_fusion_serve()`) and `gate-per-block-build.sh` failed
   rung 1/3 with `error: use of undeclared identifier 'moe_cache_repair_copy'` — that function and every fused
   site are introduced by block 10.  The per-block ladder is the instrument that makes the ownership question
   decidable; the rule is "the block that owns the behaviour and the symbols it needs", not "the block that
   created the line".

Record: `WORKLOG.md` 261119.005, `GREEDY-PURITY.md` §101, `patches/README.md` block 10, `MANIFESTS.md` gate 28,
issue #332, `scripts/gate-moe-promise.sh`.

## r322 (2026-11-19, `WORKLOG.md` 261119.006) — the band's fixup period must FILL the machine, and a tensor split's per-device tile count decides how (issue #333)

1. **`gridDim = (tiles, P)`, so `P` alone does not set the block count.**  The RDNA4 band launches one block per
   (output tile, interleave phase); `tiles = ntiles_z_gqa * K->ne[2] * Q->ne[3]` and a **`-sm tensor` split cuts
   `K->ne[2]`** (each device owns a range of K/V heads), so adding devices *lowers* `tiles`: qwen4exp (2 K/V heads,
   GQA 12) is 4 tiles per device on one GPU but **2 on two**, and 27B (4 K/V heads) is 2/1/1 on three.  The
   32-column band row is an 8-warp block whose runtime `max_blocks_per_sm` is 3, and the measured saturation point
   is **2 blocks per SM**: at r316's flat `P` 16 a 2-tile device launched 32 blocks on 32 SMs (one per SM) and a
   1-tile device only 16 (half the SMs idle).  Symptom, per op (q8_0, `n_q` 1/4): 291.7 -> 215.0 us at 2 tiles
   and 278.2 -> 138.0 at 1 tile (kv 102400), 69.0 -> 65.8 at 2 tiles (kv 16384), and 5-26 % in every other row.
2. **The rule is a warps-in-flight target, and the tile count is a legal input.**  `P = max(nsm/2, ceil(nsm *
   blocks_per_sm / tiles))`, `blocks_per_sm = max(1, 16 / nwarps)`: 2 blocks/SM for the 8-warp 256-dim row and 4
   for the 4-warp 128-dim one, i.e. ~16 warps resident per SM.  The measured per-tile-count optima (1/2/4/8/16 ->
   `P` 64/32/16/8/4 on the 256-dim class, 2 -> 64 on the 128-dim one) are exactly that, and `nsm/2` is the floor,
   so a device with 4 or more tiles keeps r316's value and nothing can be lowered.  `tiles` depends on the K/V head
   count, the GQA ratio and `Q->ne[3]` only -- `ntiles_x` is 1 for every width 1..8 -- so this is still one value
   per shape and never keyed on `n_q` or the KV extent (`GREEDY-PURITY.md` §98.5).
3. **A per-device `P` is fine; a per-width one is not.**  The three devices of a 27B `-sm tensor` run now take
   `P` 32/64/64, i.e. three different summation orders inside one layer, and that is *not* an impurity: each
   device's attention output is its own heads and never crosses a device.  What would be an impurity is decode and
   its own verify choosing differently, and the width probe (`PASS`, worst maxdiff 0, q8_0 and bf16) plus
   `plain == draft-mtp` on both the 3-device 27B (`58439631ccc9`) and the 2-device qwen4exp (`35699210a30e`) arms
   are the proof that they do not.
4. **The fixup combine is proportional to `P`, which is why "more blocks" is not free.**  Per launch on a 27B
   trace: 8.8 us at `P` 16 -> 31.3 us at `P` 64, against an FA kernel that fell 150.4 -> 75.2 us.  The op table
   already prices it (it times the whole op), and the 1-tile rows' optimum drifts from `P` 32 (16-32k) to `P` 64
   (102k) as its share falls; one constant cannot follow that, and the saturation point is the compromise.
5. **A short greedy speculative arm cannot resolve a step rate.**  `-n 128` at a mean accepted length of ~2.8 is
   ~45 steps, so one step of arithmetic resolution is 2 %: the same 27B pair read -3.4 %/step at `-n 128` and
   **0.0 %** at `-n 512` (31.5 ms/step both, 188 against 191 steps for 512 tokens -- the token rate differs only
   by the acceptance re-roll).  Quote the long arm, and derive steps as `tokens - accepted` before comparing.

Record: `WORKLOG.md` 261119.006, `GREEDY-PURITY.md` §98, `patches/README.md` block 05, `MANIFESTS.md` gate 05/36,
`ENVIRONMENT.md` §8, issue #333, `benchmarks/2026-11-19-band-split-blocks-per-sm.md`.

## r323 (2026-11-19, `WORKLOG.md` 261119.007) — an FFN's GLU may take its GATE, and only its gate (contributor PR #334 item 1)

1. **The saving is the whole gate buffer, and the allocator's own preconditions are what make it safe to
   take.**  `GGML_OP_GLU` in `ggml_op_can_inplace()` reuses `ggml_gallocr_allocate_node`'s existing rule --
   same layout and exactly one reader with no views -- so nothing about the gate/up/GLU triple is special-cased:
   the two-input form's output has the gate's layout, and the one-input (split) form halves `ne[0]` and can
   never qualify.  Measured on Qwen3.8-27B UD-Q4_K_XL (1 GPU, `-ub 2048 -fa 1 -c 16384`): ROCm0 compute buffer
   **624.97 -> 475.37 MiB (-149.60, -23.9 %)**, and on the host-expert MoE shape (35B-A3B Q8_0, 2 GPUs
   `-sm tensor -ncmoe 40 --host-experts pool`) greedy decode **121.6/121.6/121.7 -> 130.5/130.2/130.1 t/s
   (+7.0 %)** over three interleaved pairs.  A dense 27B on 3 devices `-sm tensor` with `draft-mtp n7` is
   neutral (109.3 vs 109.2 t/s): the win is the MoE shape's, not a general decode gain.
2. **Only `src[0]` may be offered, and CANN is why.**  Every streaming GLU kernel reads both operands before
   writing the element at the same index (CPU `ggml_vec_*glu_*` with no `restrict`, CUDA/HIP
   `unary_gated_op_kernel` where the same layout makes `i == j0 == j1`, Metal `kernel_*glu`, Vulkan / SYCL /
   WebGPU / OpenCL one element per invocation), so `dst == src1` would be numerically safe on them -- but CANN's
   gated GLU is a **chain** (`aclnn Silu` into dst, then `aclnnInplaceMul(dst, src1)`), which with `dst == src1`
   squares a destroyed up operand and loses the gate.  `ggml_gallocr_allocate_node()` therefore stops its parent
   scan at `src[0]` for this op; `dst == src0` is the in-place `GGML_OP_UNARY` path every backend already runs,
   and the gate and the up are symmetric in an FFN, so the restriction costs nothing in practice.  The unit
   case (`tests/test-alloc.cpp::test_glu_inplace`) asserts the up is never taken and is negative-tested by
   offering both (`x[2]->data != x[1]->data` fails).
3. **It moves no token, so the gate is the buffer and the text, not an acceptance rate.**  Same-seed greedy
   text is byte-identical ON/OFF on a dense 4B, a dense 27B (1 GPU and 3-device `-sm tensor` + `draft-mtp`), the
   35B-A3B host-expert shape and the CPU backend, and those hashes equal the r322 build's
   (`b065f37593b3` / `707c7bec1465` / `5d485275db6d` / `54e07e5a163c` / `1acb04bd9104`); prefill logits are
   bit-identical too (mean KLD 0.000000, same-top-p 100.000 %) at 1 device **and** at 3 devices `-sm tensor`.
   The rule must therefore be judged on the compute buffer it saves (a size that stops moving means the FFN
   layout stopped giving the gate a single reader) plus the ON/OFF text pair -- `test-backend-ops` cannot see a
   memory-layout decision at all.
4. **It changes which fusion windows can overlap a dead scheduler copy, so the leaf-copy gate now pins
   `GGML_GLU_INPLACE=0` for its trace arms.**  With the rule on, the fused `MUL_MAT_ID(gate) MUL_MAT_ID(up) GLU`
   window's output takes the gate's memory, so it can no longer land on the host-expert scheduler copy the
   window still reads (issue #287) and `scripts/gate-fusion-alias-copy.sh` declines **0** classes on the
   default env, where r322 declined 1 (`GLU<-NONE/copy@268435456`).  The protection is still required wherever
   the rule does not apply (a second reader of the gate, the split form, the switch off), so the decline
   assertion moved to the pinned arm -- which still logs it -- and the default arm's count is printed, never
   asserted: a decline is the protection firing, so requiring zero would punish it.

Record: `WORKLOG.md` 261119.007, `benchmarks/2026-11-19-glu-inplace.md`, `patches/README.md` block 00,
`MANIFESTS.md` gate 00/37, `DIAGNOSTIC.md` §1.1b, `scripts/gate-glu-inplace.sh`, PR #334.

## r324 (2026-11-19, `WORKLOG.md` 261119.008) — the MoE weighted reduction is computed without fma contraction, and its fold outranks the MUL-only fold (contributor PR #335)

The expert weighted reduction has two implementations of one arithmetic — the standalone
`moe_weighted_reduction_f32_vec4`/`_f32` kernels and `mul_mat_vec_q_moe`'s `fusion.red_dst` fold — and a precedence
rule between that fold and `ggml_cuda_moe_down_fold_try_fuse`.  Five things were learned, in the order they bit.

1. **"The same expression" is not "the same arithmetic": the ROUNDING MODE is part of the claim.**  Both sites
   accumulate `sum + (x * scale) * w`, which HIP clang contracts into `v_fmac_f32`; no unfused graph does.  So the
   fold matched the standalone kernel exactly and **neither matched the `MUL`/`VIEW`/`ADD` chain**, and the r323
   delivery's MoE decode text was not the chain's text on qwen35moe, qwen3moe *and* gemma4 (`bd111389f051` vs the
   chain's `abed0bf40d87` on 35B-A3B Q4_K_M).  KLD against the fully unfused graph at `-ub 512` was
   **0.006613 / 96.627 %**, with the unfused base logits byte-identical on both builds (sha256 `0a4a69973ab8…`,
   base PPL 7.1177) — i.e. the reference was the same object and the divergence was entirely the fused paths.
2. **`__fmul_rn` / `__fadd_rn` do not work; `#pragma clang fp contract(off)` does — and only in the right place.**
   HIP lowers the intrinsics to plain ops and contracts those.  The pragma is legal only at file scope or **at the
   start of a compound statement** (so every site opens a block for it), it needs a toolchain default of
   `-ffp-contract=fast-honor-pragmas` (HIP clang's is; clang-CUDA's plain `fast` and nvcc both ignore it, so a
   non-AMD backend keeps the contracted arithmetic and changes nothing), and it survives `__forceinline__` template
   instantiation — verified in the emitted ISA (`v_fmac_f32` against `v_mul_f32` + `v_add_f32`), including through
   a helper.
3. **ONE knob for two implementations that replace each other.**  `GGML_CUDA_MWR_EXACT` drives both the fold's sum
   (`fusion.red_contract = !ggml_cuda_mwr_exact()`) and the kernels (a `mwr_acc<exact>` template).  Two independent
   switches are a trap: `EXACT=0` alone leaves the fold exact and the kernel contracted, a state no release
   shipped, and a bisect would blame the fold.  The pair must also be *proven* to revert:
   `GGML_CUDA_MWR_EXACT=0 GGML_CUDA_MWR_FOLD_FIRST=0` reproduces r323 bit for bit (`bd111389f051`), while
   `EXACT=0` alone is its own third text (`0e1e59e500e8`) — which is the non-vacuity check for the knob.
4. **A fold may only displace another where it can SERVE the node, so the predicate lives at the ARMING site.**
   The precedence change makes `ggml_cuda_moe_down_fold_try_fuse` stand aside for an armed node, worth a launch per
   MoE layer (`MWR FOLD` 1 of 83 armed sites on r323 → 163 of 163 here, with `MWR FOLD ASIDE` matching and
   `MWR FOLD DECLINE` 0).  But `ggml_cuda_mwr_fold_mark` armed from the graph pattern alone, and the fold itself
   also needs `warp_size == 32`, `ne[2] == ne[3] == 1`, a contiguous down output and
   `ne[0] <= 2*GGML_CUDA_MOE_RED_CNT-2` = 8190.  Any disagreement (CDNA's warp 64, a large `n_embd`, an
   `ne[3] != 1` graph) would have cost the MUL fold and bought nothing.  The conditions now live in one function,
   `ggml_cuda_mwr_fold_serve_ok()`, enforced where the node is **armed**, so "armed implies servable" holds by
   construction and the stand-aside site needs no copy of it.  `GGML_CUDA_MWR_FOLD_FIRST` defaults **off** outside
   `GGML_USE_HIP` because nvcc ignores the pragma and there the fold would be the less exact of the two.
5. **A short `-ub 1` KLD window cannot see this class; the greedy text and a census can.**  The submitter's own
   decode row read ≈0 on the unpatched build and concluded "already exact" — while the same-seed 256-token text
   differs from the chain's.  512 sequential token steps over 2 chunks do not accumulate what 256 greedy tokens
   do, so `-ub 1` KLD is the wrong instrument here and the census is not optional: `GGML_CUDA_MWR_DEBUG` now prints
   `MWR EXACT exact|contracted` once, plus `MWR FOLD` / `MWR FOLD ASIDE` / `MWR FOLD DECLINE` / `MWR FUSED`, which
   is what makes the precedence and the arithmetic observable rather than inferred.

Measured: KLD 0.006613 / 96.627 % → **0.000000 / 100.000 %**; the default text now EQUALS the fully unfused chain
on 35B-A3B Q4_K_M, Qwen3-Coder-30B-A3B and gemma-4-26B-A4B, dense 4B unchanged (`18d8b09cd3d3`); tg128 **+3.3 %**
on 35B-A3B Q4_K_M (121.66/121.71 → 125.90/125.33), **+2.2 %** on Qwen3-Coder-30B-A3B, flat on gemma4 (which already
took the larger fold), pp512 flat everywhere.

Two side findings, both fixed in the same change.  **The release prefill-logit base moves**: against an r323 base
the gate now reads 0.006513 / 96.392 % (canonical), 0.006874 / 96.627 % (`-ub 512`) and 0.006412 / 96.588 %
(3-device `-sm tensor`), so the base is re-recorded at r324 — the justification is item 1's first table, the new
build *is* the unfused chain and the old one was not.  **`gate-server-arms.sh` was not reproducible and had two
defects of its own**: it never pinned `MOE_EXPERT_HEATMAP`, so each run started from the prior the previous run
wrote back into the model's `.moe-heatmap`, and it fired both slots' requests simultaneously, so the admission
order was a race that selected between two states of both the text and the acceptance.  With
`MOE_EXPERT_HEATMAP=load` and a fixed-order stagger each build reproduces its own numbers exactly (verified twice
on r323 and on r324), and its anchors are re-recorded; the `35b10` arm's *second slot* still moves (a
fixed-time stagger lands at a variable point in slot 0's prefill), so that arm's text is now reported rather than
asserted — its acceptance checks, which are the collapse detector the arm exists for, stay hard (`TODO` #349).

Record: `WORKLOG.md` 261119.008, `benchmarks/2026-11-19-mwr-exact.md`, `patches/README.md` block 10,
`MANIFESTS.md` gate 37, `ENVIRONMENT.md` §8 (`GGML_CUDA_MWR_EXACT`, `GGML_CUDA_MWR_FOLD_FIRST`,
`GGML_CUDA_MWR_DEBUG`), `scripts/gate-mwr-exact.sh`, `GREEDY-PURITY.md` §102, PR #335.

## r325 (2026-11-20, `WORKLOG.md` 261120.002) — the MoE device gather's block count is a DECODE-band constant, and a prefill ubatch wants 4x more of them (block 09, `TODO` #350 retired)

1. **`moe_cache_gather_kernel`'s parallelism is `n_experts * n_split`, and `n_split = 8` was chosen for the decode
   band** ("~64 concurrent blocks for the usual 8 routed experts", the kernel's own comment).  A *prefill* ubatch
   routes essentially every expert of the table (8-of-256 per token over >= 512 tokens saturates the set), so the
   same constant leaves 8 blocks per expert, each carrying a 0.6-1 MiB copy, and the copy runs pipeline-limited
   instead of link-limited whenever the per-call volume is small — worst exactly at HIGH residency, where the
   non-resident set is small.  `n_split = 32` (r325, `MOE_GATHER_NSPLIT`): **+7.5 %** prefill on 1 x R9700,
   Q8_0 at 84 % residency (2307.9/2307.2 -> 2480.8/2480.8 t/s, interleaved reps), **+2.3 %** on Q4_K_M at an
   8000 MiB arena, **neutral (+0.4 %)** on a 3-device `-sm tensor` split.
2. **It is a pure launch-shape change: same bytes, same order per byte, different partition across blocks.**  The
   greedy text is byte-identical (`572 chars sha=40e29c2b46b6` on the Q8_0 arm) and
   `gate-prefill-logits.sh --ab "MOE_GATHER_NSPLIT=8"` reads mean KLD **-0.000000 / 99.990 % top-p** — an exact
   fallback, which is what the A/B mode exists to certify.  So a gather change does NOT owe a numerics apology;
   it owes the fill-path gates (`gate-host-pool.sh` 59/59 here) because it sits on the same transport.
3. **The optimum is flat, not fitted.**  The earlier sweep (2026-11-18) put Q8_0's peak at 16-32 (48 -> 2252 ms,
   128 -> 2342 ms on the old arm) and Q4_K_M flat across 8..32, so 32 is a safe default rather than a
   model-specific choice; `MOE_GATHER_NSPLIT` is the A/B knob and is clamped to 1..256.
4. **Scope, stated so it is not over-read: this is a 1-GPU host-expert-tier win.**  On a 3-device `-sm tensor`
   split each device gathers its own slice (a third of the volume) and the constant is already enough
   parallelism, so the arm reads 4125.8 -> 4141.4 t/s (noise).  A release that quotes "+7.5 % prefill" must say
   which shape it was measured on.

Record: `WORKLOG.md` 261120.002, `benchmarks/` — measured in the retired campaign
`archive/work/prefill-transfer-overlap/` (pass 9 + the r325 re-measurement), `patches/README.md` block 09,
`ENVIRONMENT.md` §2.

## r326 (2026-11-20, `WORKLOG.md` 261120.003) — the graph-replay width band is a TOKEN COUNT, not "a verify"; a moving width pays a capture per width, and a captured graph may no longer grow the FA staging arena (block 10, contributor PR #336)

1. **The band `ggml_cuda_graph_is_multi_token()` compares against was `MMVQ_MAX_BATCH_SIZE` (8) by accident of
   r20's argument, not by measurement.**  r20's rule was "only a true PRE-FILL (a varying ubatch) skips capture,
   because capture never amortises there" and it used 8 as the prefill/verify boundary.  That boundary is wrong
   for the one shape that matters more than any other on a server: **4 slots x `draft-mtp n3` under
   `LLAMA_SPEC_ALIGN` is a fixed 16-row verify every step**.  Measured r326 (gfx1201, ROCm 10.1.0, 27B
   UD-Q4_K_XL, 8 interleaved runs per arm): **178.1 -> 184.3 t/s (+3.5 %)**, with every run of one arm above
   every run of the other; 4B Q8_0 +1.8 %; `llama-batched-bench -npl 16` +1.2 % on both models with npl 1/4/8
   untouched (+-0.1 %).  `GGML_CUDA_GRAPH_MAX_TOKENS` (default 16) is that band.
2. **The delivery default is provably untouched, and that is a text claim, not a rate claim.**
   `--spec-draft-n-max 7` can produce at most an 8-row verify, so at the default the `GGML_CUDA_GRAPH_TRACE`
   census is **identical** at `=8` and `=16` (1 GPU `direct=53 replay=1713 capture=18 properties_changed=14`,
   3 GPU `-sm tensor` `3657/21819/813`) and the same-seed text is byte-identical; `ngram-mod` likewise.  A change
   to this predicate therefore owes a *census-identity* arm, not a throughput arm, for the default path.
3. **"It pays where the width repeats" is exact, and it is why a bare width threshold is only safe because the
   delivery default cannot reach it.**  A moving width captures once per new width: 3 devices `-sm tensor`,
   `--spec-draft-n-max 15`, captures **150-240 -> 1350-1430 (9x)**, `properties_changed` 279 -> 1053-1926.
   That shape's **rate is bimodal in BOTH arms** (9 runs span 46.5-57.6 t/s, acceptance 0.1395 vs 0.166-0.168),
   because above an 8-row verify the text is not run-to-run stable (§11) and the draft depth follows the
   acceptance the drifting text produces — so neither a win nor a regression can be attributed to the band
   there.  A "only capture when the width repeated" rule does NOT fix it (the depth saturates, so the width stays
   16 while the FA mask extent moves each step).  Recorded as an accepted limitation: it needs `n_max >= 8`,
   i.e. a user already outside the purity band, and `=8` restores r325 bit for bit.
4. **A token count is not a verify: 9..16-row PRE-FILL ubatches are captured too.**  A short prompt, a prefill
   tail, or a small `-ub` produces 9..16-row prefill graphs; `-ub 12` over a 5298-token prompt captures **21**
   graphs at `n_tokens=12` and reads **+1.2 %** prefill.  The r20 comment ("only a true pre-fill skips") is
   therefore no longer the whole story, and any future argument about prefill graph capture must be made on the
   *token count*, not on the word "prefill".
5. **A captured graph must never grow the FA staging arena — the old justification is retired.**  The arena's
   scratch is deliberately outside the node allocation, outside the pool budget and outside `--fit`, and
   `fattn-common.cuh` argued it was safe "because a multi-token graph is never CUDA-graph captured".  With the
   band at 16 that is false: above the 8-row band a native-capable K/V cache STAGES, so a 9..16-row staged FA op
   can be inside a capture (measured: the `-ub 12` arm logs the capture-time staged call once).  The rule is now
   `ggml_backend_cuda_context::capturing`: no growth from inside a capture (on CUDA `cudaMalloc`/`cudaFree`
   return `cudaErrorStreamCaptureUnsupported`; HIP's relaxed mode tolerates them, which would be a per-runtime
   difference), so that call reads the K/V cache natively — same arithmetic, same tokens, no scratch.  A growth
   *between* calls still frees the old arena only after `ggml_cuda_graph_mem_freed()`, so a capture that baked
   the old pointer is re-made before it is read.
6. **The gate's negative test is part of the evidence.**  `scripts/gate-graph-replay-width.sh` PASSes on the
   r326 build and **fails 3 of its 5 checks on the shipped r325 binary** (no 9..16 capture in the default arm,
   the kill switch changes nothing, no pre-fill capture) — a graph-decision change that cannot fail on the
   previous build is not testing the decision.

Record: `WORKLOG.md` 261120.003, `benchmarks/2026-11-20-graph-replay-16-tokens.md`,
`patches/README.md` block 10, `ENVIRONMENT.md` §7.4, `MANIFESTS.md` gate 38.

## r327 (2026-11-20, `WORKLOG.md` 261120.004) — one FA KV-group bitmap per graph evaluation: exact, worth nothing on this box, and OPT-IN (block 10, contributor PR #337)

1. **The packed bitmap is a function of the mask's CONTENTS and the producer's arguments, so one graph evaluation
   needs one build, not one per attention layer.**  With a unified KV cache every layer of a decode step classifies
   the same cells: on a dense 27B with 16 attention layers, 16 `cudaMemsetAsync` + 16 `flash_attn_mask_to_KV_blocks`
   launches per step become one, and the census reads `built=1 reused=15` in the shape that reaches it.  It is
   **bit-identical** by construction — the reused words are the same words, so the kernel sees nothing new — and
   measured so: the same greedy text as the previous build in both arms, `test-backend-ops -o FLASH_ATTN_EXT`
   7381/7381 with the switch ON.
2. **It is worth NOTHING on gfx1201, and that is the reason it ships off.**  2-slot `--kv-unified` server, 27B
   UD-Q4_K_XL, q8_0 K/V, a 25.7 k-token parked slot, 3 greedy decodes: **31.89 t/s on, 31.85 off, 31.87 on the
   release build** — the 15 launches it removes per step are free on Linux.  Its whole case is launch COST
   (Windows/WDDM, where this repo's own measurements put launch-bound moves at +16-19 %), i.e. a platform the
   delivery does not validate.  So: a first-class, always-present switch whose default is OFF, not a "beneficial
   feature turned off" — the promotion test is a measured win on a supported configuration, nothing else.
   A default-on candidate for this shape is the r20-era `MMVQ_MAX_BATCH_SIZE` class of mistake in reverse: an
   argument about launch count is not a measurement.
3. **Reach is narrower than "every attention layer".**  A single-sequence batch takes the *DERIVED* bitmap
   (`kq_block_skip_derived`, host-classified) and never reaches the packed form; the packed form needs a
   multi-sequence ubatch or the RDNA4 band at depth (its own work floor `K->ne[1]*saving >= 20000`).  So
   `test-backend-ops` (one attention node per graph) and `llama-bench` (`-p`/`-n`, one sequence) read `built=0`,
   and an arm built on them is vacuous about this feature by construction.  The reachable shapes are a server
   with a resident second slot, and a co-batched prefill (`built=10 reused=150` measured).
4. **A per-graph-evaluation cache keyed on the producer's ARGUMENTS is safe only while those arguments ARE the
   inputs.**  The key is `n_kv`, `n_tps`, `s31`, `s33`, `n_tiles`, `n_streams`, `n_mask_streams` and the template
   parameter `ncols1`, plus the mask tensor + its `data` pointer (`n_groups` follows from `n_kv` and
   `FATTN_KQ_STRIDE`).  A band tile/KV-split change that reaches the producer through those moves the key with it
   (r319/#330 and r322/#333 did, and needed no change here); a change to what the producer *computes from* them must
   extend the key.  This is the maintenance contract, and it is written at the declaration and in `patches/README.md`
   block 10 — a silent extension of what the producer reads is exactly how a shared cache goes wrong.
5. **A decline must be counted, and the OFF arm must not read all zeros.**  One shared 64 KiB buffer per context
   means four legitimate declines (another stream of the same evaluation, a bitmap over 16384 words, an allocation
   that would land inside a capture, a failed allocation) and a fifth (no evaluation started).  Each has its own
   census counter, and `calls` is incremented **before** the switch is read, so `reuse=off calls=16` proves the arm
   reached the code — the contributor's first form reported `built=0 reused=0 fallback=0` when off, i.e. an arm
   indistinguishable from one that never ran.  A failed 64 KiB allocation DECLINES rather than aborting: a cache may
   not turn a fit run into a crash.  A replayed graph repeats its captured decisions and is not counted, so a
   captured 16-layer step reports `built=1 reused=15`, not one per step.

Record: `WORKLOG.md` 261120.004, `patches/README.md` block 10, `ENVIRONMENT.md` §7.4, `MANIFESTS.md` gate 10.

## r328 — an anchor row is only as good as the arm it calls "the reference"

**An HC-class generated text is not portable across a numerics change, and a gate that does not anchor its
reference arm cannot tell a moved chain from a moved fused op.**  `gate-hc-fusions.sh` graded `chain_chain`
as "the true chain" and anchored only the five numeric arms; the two `fusedmixer` rows were recorded at r308.
The class's text then moved for *every* arm somewhere in **r290..r319** (`bin-r289` still reads the r308 row,
`bin-r320` already reads the current one), so the numeric arms' anchors went stale and the gate went red — on
the released r327 exactly as on the r328 tree, with byte-identical numbers (`local`: chain 444
`a512ee68c4b3`, pre 443 `c8db61d5dcc8`, post 449 `6a102dbc62d6`; `meta`: chain 449 `6a102dbc62d6`, pre 444
`a512ee68c4b3`).  With no anchor on the chain the only visible symptom was "three numeric arms left their
anchors", which reads exactly like "the re-base moved the HC chain" — and that is how R11.2 came to be
carried as a release blocker for a whole rung, and then to be blamed on the re-base.

Rules:

1. **Anchor the reference arm.**  A row of anchors is a *relative* statement; the reference is the absolute
   one, and it is the one that tells you the row is stale.  A gate that leaves its reference unanchored can
   only report "an arm moved", never "the world moved".
2. **Re-derive a row together.**  Re-anchoring one arm to the current text is how a real move gets erased;
   re-anchoring the whole row (chain included) is a dated statement of the shipped behaviour.
3. **Suspect the gate before the re-base.**  A red gate on a re-based tree needs the previous release's own
   binaries run on the same arm set *first* — `LLAMA_CLI=…/bin-r327/llama-cli ./scripts/gate-….sh`.  Two
   minutes of that refuted a carried blocker here.

Record: `RUNGS.md` R18.4, `archive/work/upstream-rebase-261120/COLD-START.md` §8, `WORKLOG.md` 261121.003.

## r329 — a census assertion cannot fail on a binary that predates it: anchor the binary too

**A gate that asserts a *census line* has no way to go red on a stale binary — it reads a run that never
emitted the print, and reports the absence as a mechanism defect.**  `gate-uma-experts.sh` and
`gate-prefill-server-slots.sh` both defaulted to `~/llama-scratch/build-rocm` (the `apu-expert-gtt` campaign
tree, from before that campaign became block 15), and the 2026-11-21 arch validation on `halo` launched them
with no `LLAMA_CLI`/`LLAMA_SERVER`.  So the r328 release certified a **2026-11-12** build of `7efb30ae5`:
`gate-uma-experts` reported 48/49 with `FAIL ref: the GTT sizing fills did not report the placement queue`,
because that build's `apply_heat_seed_locked` prints `… ranked by the %s band)` and the r277 transport tail
(`, X MiB in Y ms (Z MiB/s, queue|serial)`) did not exist yet — the `strings` check that "proved" the format was
present had been run against **the other binary** (`build-r328/bin/libggml-hip.so`), which is the r328 one.  With
the delivery binary the same gate reads **49/49** and 120 placement-queue seed lines, and the slots gate reads
3 checks / 0 failures.  Both gates now print `<binary>` + its own `--version` (build + commit) as their first
line, and default to `~/llama.cpp/build-rocm`.

Rules:

1. **A gate prints the binary it certifies.**  `<path>` and `--version` on line 1, always — a census arm cannot
   detect a stale anchor for you, and the arm's own numbers look plausible either way.
2. **`strings`/`nm` evidence is evidence about *that file*.**  A format string found in one build says nothing
   about the build that produced the log being explained; check the log's own `build NNNNN (sha)` line first
   (llama-cli/llama-server print one at `-lv 4`, and `--version` prints one cheaply).
3. **A default that points at a campaign tree outlives the campaign.**  When a campaign is promoted into a block,
   its gate's default path and its header comment are part of the promotion — this is the delivery-side twin of
   the `wip/` no-stub rule.

Two facts this session established alongside it:

* **`align_draft_caps()` already equalizes the band-derived caps inside the call.**  `TODO` #356's premise (the
  arena band replaces the width band, so the caps can differ at a step boundary) is wrong on a `-ncmoe` host-pool
  shape: `llama_model_moe_cache_band()` answers **0** there (a full-residency pool is not an arena), so
  `LLAMA_SPEC_ALIGN_WIDTH_BAND` (16) is the bound in force and every two-drafting call logs one width —
  `aligned draft depth seq N: 3 -> 5 (width band 16, drafting 2, hard 5)`.  The 2 surviving split steps are
  **prefill/decode boundary** steps (one slot's last prompt chunk cannot merge with the other's verify batch,
  and padding it would advance the recurrent state), which no cap policy can align.  `NCMOE_SPLIT_MAX=2` stays.
* **The `>RAM` arena's ceiling on a 123 GiB APU is the budget.**  `gate-prefill-server-slots.sh`'s default
  `-lme 90` on the 122B-A10B Q8_0 (~200 GB) puts a 90 GiB anonymous store on top of a >RAM mapping: measured
  94 GiB anon → the kernel OOM-killed `llama-server` and 7 desktop tasks.  At **`-lme 85`** with the default
  `--lazy-mode auto` both arms complete and the in-place band engages (159 calls × 75 tables, 3351 slot
  admissions, 1113 landing / 2238 reclaiming a non-routed resident), A's text identical `ee7c607ad86e` and A
  *faster* with the knob on (2.28 → 2.47 tok/s).

Record: `WORKLOG.md` 261121.007, `archive/work/r328-followups/` (§3 rows and `logs/`).

* **A generated file is a delivery artifact, and its check belongs next to the generator.**  `tools/cli/README.md`,
  `tools/completion/README.md` and `tools/server/README.md` are written by `llama-gen-docs` from `common/arg.cpp`,
  so every block that adds an option silently owes a regeneration.  The r328 set shipped them last regenerated at
  **block 02**: `llama-gen-docs` rewrites all three on the released tree (`34 insertions(+), 14 deletions(-)`), and
  a consumer reading the delivered CLI docs cannot see `--kv-window`, `--kv-attendance-pool`, `-lme`,
  `--wmma-verify`, `-ib`, `--spec-draft-mtp-vocab`, the `-ncmoe` LAST-N wording, `--load-mode buffered` or
  `relaxed-tensor`.  The obligation was written down (the re-base's R18 checklist) and skipped, and **no gate
  compares the committed docs against the generator** — the same shape as the stale-anchor rule above: a check
  that cannot fail is not a check.  Fixed by regenerating into block 16 (the last OFFICIAL block touching
  `common/arg.cpp`) and by `scripts/check-gen-docs.sh`, which runs the generator built from the tree under test and
  fails on a rewrite.  Rule: when a file is generated, the generator run is part of the block's change, and the
  drift check runs before a tag.

* **A re-base rung that lands fewer files than the patch it replaces has dropped something until proven otherwise, and the per-block build ladder cannot see it (r331, issue #345).**  Block 00's
  `ggml/src/ggml-alloc.c` half (+121/−2: `GGML_GLU_INPLACE` with the `n_src_inplace` gate-only rule, W4's
  unused-view release, and the `layout_fp` graph fingerprint that `ggml_gallocr_needs_realloc()` requires for
  issue #183) was dropped by the r328 re-base's R00 cherry-pick (`37 files → 34 files`) with no decision row and
  no fixup-ledger row, so r328 and r329 shipped without it while the block's own text, `ENVIRONMENT.md`, the
  per-block gate table and `tests/test-alloc.cpp`'s `test_glu_inplace` still described it — and **r329's
  `test-alloc` aborts at `test-alloc.cpp:704`**.  Two facts make it a class:
  (1) `scripts/gate-per-block-build.sh` proves each rung *compiles*, and a missing file's hunks inside other
  files compile fine, so an 18/18 `--all` ladder is green over it — only the block's own behavioural gate runs
  the rule, and neither release's gate table ran `gate-glu-inplace.sh` (the block-00 row of `MANIFESTS.md`'s
  per-block table names it for "every release that touches block 00", and a re-base touches all 18);
  (2) the deliverable is reproducible from the previous release: restoring the r325 chain's file gave a r331 tree
  whose `ggml-alloc.c` is **byte-identical at every one of the 18 rungs** (`git show <rung>:…` compared pairwise),
  which is what proves the restore reintroduced exactly the lost content and nothing else.  Rule: reconcile each
  rung's changed-file set against the patch it replaces (same subject, previous release), and treat a shrunk set
  as a defect until a decision row names the retirement — instrumented since r341 by
  `scripts/check-rung-file-sets.py` (gate 38, decision rows in `patches/FILE-SET-RETIREMENTS.tsv`; `TODO` #358
  retired).  Record: `WORKLOG.md` 261121.010, and 261121.022 for the check.

* **An undefined `GGML_CUDA_CC_*` name in `#if __CUDA_ARCH__ == …` reads as 0: the guard is silently false, the block compiles out with no warning, and only the target arch notices (r331).**
  Upstream renamed `GGML_CUDA_CC_DGX_SPARK` → `GGML_CUDA_CC_GB10` (both `1210`, `ggml/src/ggml-cuda/common.cuh`)
  between the r325 fork point `4f9dfa0e3` and the r328 base `919794500`.  Block 10's `mmvq-ksplit-body.inc`
  carried the old spelling through the re-base, so its early weight-load prefetch was compiled out on GB10
  (`mmvq-ksplit-body.inc:169`, issue #345's second finding) — **inert on every HIP target**, which is exactly why
  no gate here can see it, and why the check is a source audit instead: every `GGML_CUDA_CC_` token used in the
  tree must be `#define`d in `common.cuh`, and the grep must include `*.inc` (`.cu`/`.cuh`/`.h` alone misses a
  body file).  Audited clean at r331 apart from this one site, which now says `GGML_CUDA_CC_GB10`.  The same
  re-base pass also lost `ggml_cuda_should_fuse_rope_set_rows()`'s `GGML_CUDA_DISABLE_ROPE_SET_ROWS` kill switch
  (restored; a bisect knob only, `DIAGNOSTIC.md` §7.3).  Record: `WORKLOG.md` 261121.010.


* **Per-block independent buildability is NOT a delivery property (r332, maintainer's decision 2026-11-21).**  The
  set is the artifact: `apply-all.sh` applies every block and ONE `all` build of the applied tip proves it (6-10 min
  cold, ~10 s warm).  A block is a LAYER on its predecessors — it owns its functionality but has no standalone
  build/apply guarantee — and the r2 rule ("every block commit builds the `all` target on its own") was retired
  after it cost **35-55 min per re-base and weeks of wall time over the project's life**: the per-rung configure
  alone is ~30 s x 18, the FA template-instance set recompiles per rung when a widely included header moves, and
  on a 16-core box a re-base ladder measured 31 min even fully warm.  What the rule was credited with, it did not
  cover: a block that is uncompilable **in the applied set** fails the set build (the only thing a consumer
  compiles), and a file a re-base silently dropped is invisible to *any* build — that is the file-set
  reconciliation's job (`TODO` #358), which is what would have caught r328's `ggml-alloc.c` loss.  The walker
  (`scripts/gate-per-block-build.sh`) is kept as a bisection tool (`--changed-from <previous release tip>`) and is
  called by nothing in the release or per-push flows.  Cost of the decision: a regression that makes an intermediate
  tree uncompilable while the tip compiles is no longer detected by construction (the r305-r307 class); a bisect
  that needs it runs the walker on demand.  Record: `WORKLOG.md` 261121.011, `MANIFESTS.md` gate 5, `AGENTS.md`
  -> *Build the fork*.


* **An RDNA2-only change is INVISIBLE to this box's default build, and the narrow `RMS_NORM` is bit-identical by
  reduction order, not by luck (r334, contributor PR #338 → blocks 05/08/10).**  Three facts, each of which cost a
  wrong conclusion before it was written down:
  (1) **`GPU_TARGETS=gfx1201` does not compile the RDNA2 device pass at all.**  `RDNA2` is `#define`d in
  `ggml-cuda/vendors/hip.h` from `__gfx1030__`, so `#if defined(RDNA2)` code is excluded from every gfx1201 code
  object — a normal `all` build proves nothing about it (it will happily accept a `#error` inside such a block).  A
  change that lives in one therefore needs a second configure with `gfx1030` in the target list, and the
  gate itself is proved by putting a `#error` inside it and checking the two builds disagree (done here for both
  `norm.cu`'s narrow branch and `mmvq-moe.cu`'s rows-abreast loop).
  (2) **The one-warp narrow `RMS_NORM` reproduces the wide launch's bits because `block_reduce`'s multi-warp path
  is a 5-step warp butterfly with `+0` in the upper lanes, not a pairwise tree over the warp sums.**  Warp `w`'s
  sum lands in lane `w`, lanes `n_warps..31` hold `+0`, and the butterfly over 32 lanes therefore collapses to
  `(s0+s2)+(s1+s3)` — for 4 warps and for 8 warps (warps 4..7 are exactly `+0` at `ncols <= 128`, and adding
  `+0` is exact).  Any change to `block_reduce` or `warp_reduce_sum` breaks the identity; so does writing the
  square as `fmaf(x, x, 0.0f)` instead of `x*x`, because the compiler's contraction into the first butterfly add is
  part of the order.
  (3) **The RDNA2 FA tile caps were a workaround for `launch_fattn`'s `GGML_ASSERT(max_blocks_per_sm > 0)`, and
  that assert is only unreachable while the manual RDNA2 occupancy figure is non-zero.**  `launch_fattn` computes
  it as `4*waves_per_simd/waves_per_block` (integer division) and only replaces HIP's 0 when its own value is
  larger, so a kernel wide enough that `4*waves_per_simd < waves_per_block` still aborts.  The caps were removed
  on the contributor's full `FLASH_ATTN_EXT` suite; that suite, not the removal, is the fact.
  Record: `WORKLOG.md` 261121.013, `patches/README.md` blocks 05/08/10, `ENVIRONMENT.md` §7.4.


* **A block's MOVED COPY of an upstream function can drift from its own fork point, and no build or gate here
  sees it: the next re-base is what reconciles it (r335, contributor PR #339).**  Block 10 splits the fusion
  dispatcher out of `ggml-cuda.cu` into `ggml-cuda-try-fuse.cu`, so the delivery carries a *copy* of
  `ggml_cuda_can_fuse`.  Upstream `31953bcf7` ("ggml : add BF16 support to RMS_NORM (CPU, CUDA)", #32135)
  relaxed that matcher's `RMS_NORM` type checks **in the hub**, and the block's copy was taken before it, so on
  **r324-r327** a BF16 `RMS_NORM` + `MUL` graph hit `GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32)` -- an
  abort on **every** HIP/CUDA device -- while each of those releases' own base accepted an F32 or BF16 norm.  The
  block's *removal* hunk in `ggml-cuda.cu` kept applying across the re-cuts, so nothing flagged it, and every
  gate in the fleet is blind to it (no RDNA2/`gfx1030` device; the text arms use F32 norms; the code is
  reachable on any device only from a BF16-norm graph).  Two consequences worth keeping:
  (1) **a default-set defect can be introduced by an upstream commit between re-bases and removed by the next
  re-base with no patch at all**, so a repro against a stale tree is a report about the past -- re-run it on the
  current release before cutting a fix, and name the release revision;  (2) the check is cheap and should be part
  of a re-base's reconciliation alongside the file-set rule (r331): for each region a block *moved* rather than
  edited, diff the block's copy against `git show <base>:<upstream-file>` and treat a difference as a defect
  until a decision row explains it.  On r334 the `RMS_NORM + MUL (+ ADD)` branch and the `RMS_NORM + SCALE` arm
  are byte-identical (2944 bytes) to `919794500:ggml/src/ggml-cuda/ggml-cuda.cu`, `RMS_NORM + MUL + ROPE` is
  F32-only on both, and `test-backend-ops -o RMS_NORM_MUL_ADD` reads 124/124 with the BF16 and BF16-residual
  arms passing (gfx1201).  Record: `WORKLOG.md` 261121.014, `archive/work/can-fuse-bf16-rms-norm-r324/`.

* **The `ROCBLAS_USE_HIPBLASLT=0` pin is now nearly free on qwen4exp, because the BF16 dense GEMMs are MMB's
  while the pin is set — and a *version* gate for it would be the wrong switch (r336, contributor PR #340).**
  Facts, all gfx1201 / ROCm 10.1 / 2-GPU `-sm tensor` / Flash-Next GSQ IQ3_S (host experts `-ncmoe 47 -lme 47`):
  (1) **The gate is the user's pin, not a ROCm version.**  `mmb_hipblaslt_off()` reads `ROCBLAS_USE_HIPBLASLT`
  and the arch default is an RDNA4 field gated on it, so with hipBLASLt enabled the census is **0** BF16 dense
  launches and the default path is bit-identical (prefill-logit `-0.000000 / 100.000 %`, identical greedy text
  both on the 4B 3-GPU arm (`7e6229b260f6`) and on Flash-Next (`c75182eb1697`)).  With the pin, 80+ launches
  (the log caps at 80) and `GGML_CUDA_MMB_BF16_DENSE=0` returns to 0.  The pin is only *needed* below the
  `c50ea60130f` fix (hipBLASLt 1.5 = ROCm 10.2), so a version check would remove the win from a 10.2 user who
  pinned for another reason (e.g. a hipBLASLt OOM at a tight slab headroom, `runme`'s own documented
  workaround) and give a sub-10.2 user who did not pin nothing — there is no cost to recover while hipBLASLt is
  dispatching.  The honest limitation is the other direction: a box where hipBLASLt is *absent* (rocBLAS falls
  back with the variable unset) is not detected, and `GGML_CUDA_MMB_BF16_DENSE=1` is the escape hatch.
  (2) **The pin's prefill cost drops 13-23 % -> 1-4 %**: 1691 -> **2196 t/s (+29.9 %)** on a 5.2k-token prompt,
  which is above the hipBLASLt-enabled arm's 2027; decode flat (55.2/55.9 -> 55.0/56.1 t/s at `-n 128`).
  (3) **The change is provably prefill-only**: the W=1..8 width probe on the same (pinned) model reads
  `width_purity=PASS (worst maxdiff 0)` on both its flat and boundary arms and every per-width
  `row0/row1` hash is identical to the base build's.  That is the general fact — a BF16 dense path with a
  33-token floor cannot reach the decode band, and the probe is what demonstrates it rather than asserting it.
  (4) **Three of its four pieces are bit-identical by construction, and that is checkable**: `SMALL_TILE`
  (geometry only, the K sum order is unchanged), `PRODUCER_BF16` (the producer's RNE copy == the conversion it
  replaces) and `HC16=1` (the reverted slot write) each read **0.000000 / 100.000 %** on the pinned
  prefill-logit point; `TINY_BF16_XH` alone moves numerics (**0.014143 / 96.402 %** against the F32-read arm)
  because reading the activation's bf16 copy rounds it, which is the one piece that must be judged at the
  logit level.
  (5) **The pinned-mode family switch is smaller than the library gap it sits inside**, which is what makes it
  decidable rather than a regression: pinned-vs-r334-pinned **0.013781 / 96.427 %**, while r334's own
  default-vs-pinned pair is **0.015033 / 96.24 %** and r336's is **0.014772 / 96.41 %** — the pinned arm moves
  marginally *towards* the default.  A pinned base therefore needs one re-record on landing, never a
  cross-release hash identity claim (`GREEDY-PURITY.md` §44 addendum).
  (6) **The hc gate's BF16 side copy was dead on RDNA4 and it was 84 MB per call** (`2*T*10240` at `T 4096`):
  its readers only take it where a graph marks the output BF16-only and every marking pass is
  `GGML_CUDA_CC_IS_RDNA3_5`-gated, so turning `hc16` off on RDNA4 is output-neutral (`HC16=1` A/B
  0.000000 / 100.000 %) and worth **+0.4 %** prefill with hipBLASLt **on** (2594 -> 2604 t/s, 3 reps) — the
  only part of r336 a consumer gets by default.  A dead write is worth removing even when it is only 0.4 %.
  (7) **The same facts hold on ROCm 10.2, and the effect is larger there**: both builds were remade against
  `/opt/rocm-10.2.0-gfx120X` (the canonical build's new line), and 10.2-vs-10.2 gives the same coherence hash as
  10.1 (`7e6229b260f6`), prefill-logit **-0.000000 / 100.000 %**, the same census `0/80/0/0`, and pinned prefill
  **+33.3 %** (1553/1662/1648 -> 2146/2169/2164 t/s) against +29.9 % on 10.1.  And a *method* fact worth not
  re-deriving: **a fresh-start arm on the two shapes §44 names is now vacuous by construction**, because
  `mmf32` (this block, §47) claims the `m <= 256` F32 GDN `ssm_alpha`/`ssm_beta` and router shapes — measured
  10/10 and 6/6 one class on **both** ROCm lines with hipBLASLt enabled, which is "no spread", i.e. no evidence
  about the library rather than evidence about it (`TODO` #362 keeps the arm map and names the decisive
  candidate).
  Record: `WORKLOG.md` 261121.015, `benchmarks/2026-11-21-mmb-bf16-pinned.md`,
  `archive/work/mmb-bf16-pinned-r325/`, `ENVIRONMENT.md` §10.2, `patches/README.md` block 06.


* **A tier's own hole set is not the only way its master can be mapping-backed: an arm that is REFUSED a plan
  still has to declare its holes (r337, GitHub issue #347, block 15).**  Since r294 the host experts' master is
  the read-only model mapping whenever the tier has holes (`host_expert_master_mapped()`), and `--load-mode` has
  no say in it.  The APU GTT **arena** arm is the counter-example that rule did not cover: it records
  `host_experts_store_arena` and deliberately leaves `host_experts_plan_done` false (it has no pinned tier to
  balance, so `-lme`'s hole set is not its plan), so the predicate read "not mapped" and `llama-model.cpp`
  allocated a **pageable full copy of the whole expert set**, on every `--load-mode`, from r294 to r336.
  Numbers (gfx1151 `halo`, 35B-A3B Q4_K_M, `-lme 4`, 18 662.5 MiB expert set): one `ROCm_HostMap` buffer of
  **18 662.00 MiB** and peak `RssAnon` **23 649 MiB** before, per-range host-pointer buffers over the mapping
  (176/144 MiB) and **4 987 MiB** after -- text identical (`627b2ebd7792`) either way; the reporter's Flash-Next
  set (40 925.6 MiB, `-lme 28`) was a **46 800 MiB** copy on a 780M.  Rules:
  (1) **A predicate that answers "is this tier mapping-backed" must be satisfied by EVERY arm that keeps
  tensors in the mapping**, including one that has no plan; the arena now records its tables in
  `host_experts_holes` and the predicate accepts `host_experts_store_arena` as a hole source, so there is still
  exactly one decision.
  (2) **An `mlock` must not pin a tier's fill source** (TODO #333's rule) -- the arena's holes are now
  `host_expert_is_hole()`s too, so `--load-mode mlock|mmap+mlock` skips them and counts the bytes instead of
  sweeping the whole set.
  (3) **`use_mmap` is not a storage decision, and neither is a plan.**  r294's rule was "the load mode does not
  decide where the master lives"; its twin is that the master predicate has to be keyed on the TIER, and the
  arena tier's forced mapping was necessary but never sufficient (the arm's own comment claimed the copy was
  avoided; it was not).
  (4) **The tier's census must not name the other tier**: on the arena the pageable-master line said
  "pool-backed `--host-experts pool`" and a `host pool PLACEMENT reads` line was printed where the preflight
  says the pool does not apply.  Both now say the arena / stay silent.
  Gates: `gate-uma-experts.sh` **49/0** (12 arms one text, `pfverify` spans still equal to the mapping, the
  `>RAM` arm 85 861.7 MiB / 2 919 evictions), `gate-prefill-server-slots.sh` **3/3** (159 in-place calls at
  `LME=85`; also 120 at `LME=60 -ub 512`), prefill-logit **-0.000000 / 100.000 %** on the arena arm and
  **0.000000 / 99.990 %** on the default point (bases from the pre-fix build, same ROCm line), coherence 4B
  `7e6229b260f6`.  The discrete host pool is unaffected by construction (an `IGPU`-only flag).  The slots gate's
  other fact: at a tight budget its `ip` arm fails as "the scenario did not exercise the change" because the
  in-place admission's bound is `|used| <= slots` and a ~3548-token ubatch routes nearly every expert
  (`LME=60` default ubatch = 564 declines / 0 calls against 120 calls at `-ub 512`), so `LME=85` is the arm to
  run.  Record: `WORKLOG.md` 261121.016, `patches/README.md` block 15.

## r338 — an FFN split granularity is a reduction order; the delivery default stays at `lcm(blck_size,128)` and the even split is opt-in (block 09, contributor PR #342)

**`-sm tensor` splits an FFN tensor at `lcm(blck_size, 128)`, and at 2 devices that floors a 640-wide expert
set's share from 320 to 256 (256/384).**  Lowering the granularity to `lcm(blck_size, 64)` when the size divides
evenly gives 320/320 and measured **+7 % `draft-mtp` decode / +4 % prefill** on 2-device Flash-Next; it is
`-of tp-ffn-even` (`LLAMA_TP_FFN_EVEN=1`), **default OFF**.

Rules:

1. **The granularity is the grouping, so it is a reduction-order change and must be judged as one.**  On the
   2-device Flash-Next point it reads mean KLD **0.013963 / 96.467 %** against the recorded base — 2.8x the
   prefill-logit limit — but the same point's `-ub 4096` vs `-ub 3840` reordering control reads
   **0.015248 / 96.516 %** and moves base PPL +1.21 % against the change's +0.59 %.  A reordering, not an
   approximate path; still far above "rounding level", so it is opt-in (`GREEDY-PURITY.md` §104).
2. **The 64 floor is MMB's, not a preference.**  `mmb_supported_mm`/`_mmid` decline `K % 64 != 0`, so a lower
   floor would silently drop MMB on the down projection.  At the *affected* MoE shapes MMB is not on the down
   anyway (`routed=0`; the census shows only the F32 routers), so the floor currently protects dense shapes.
3. **The reach is arithmetic and must be enumerated, never assumed to be one model.**  On this box it fires for
   every Flash-Next quant and its MTP drafts at **2** devices, `gpt-oss-20B-MXFP4_MOE` at **3**, and
   gemma-4-26B-A4B / Inkling-2-Mini at 3-4.  It is **inert** for every 3-GPU shape the delivery validates
   (35B-A3B 512-wide, 27B 17408-wide, 4B/9B/122B, DeepSeek-V4, Gemma4 12B/31B/E4B) and for Flash-Next at 3 and
   4 devices (640 % 3 != 0; 160 % 64 != 0).
4. **The tag layer is the discoverable spelling of an env-only switch.**  `-of <tag>` sets the switch the
   feature already reads; the table in `common/common.cpp` is the single place a tag exists, and giving a
   feature a real option later means deleting its row.  A tag ENABLES only — there is no "off" form, because
   every tag's feature is off by default — and `LLAMA_ARG_OPTIONAL_FEATURES` is the arg's own env form.
5. **A default run must remain bit-identical, and that is the gate.**  After promotion the prefill-logit point
   reads **-0.000000 / 100.000 %** and the 2-device greedy text is unchanged (`c420dfe04c42`).  The enabled
   arm's text is a *separate* per-device-count reference (`cbdb63954bbb`), never a both-arms-equal claim.
6. **The generated docs are owed by the block that adds the option, and land in the last official block that
   touches `common/arg.cpp` (block 16), never in the temporary `0300`.**  `scripts/check-gen-docs.sh` is the
   check (r329).

Record: `WORKLOG.md` 261121.017; `benchmarks/2026-11-21-ffn-even-split.md`; `scripts/gate-ffn-even-split.sh`
(`MANIFESTS.md` gate 37); `GREEDY-PURITY.md` §104; `ENVIRONMENT.md` §4.1.

## r339 — the hc mixer's band kernels: a per-token grid is redundant work, and the emulated butterfly is a fixed expression (block 11, contributor PR #344)

**The Q8_0 `GGML_OP_HC_MIX` (`hc_mix_q8`) is a decode/verify-band op (`nt <= 8`), and two of its three kernels
were re-deriving per token what the band shares.**  `hc_mix_down_dots` ran one block per `(row, token)`, i.e. it
re-read the whole 3.5 MB `w_down` once per token; `hc_mix_up_collapse_q8` emulated eight reference blocks per
fused block, holding 16 KB of LDS partials and running a butterfly per `(warp, token, row)` for a sum that has
only two non-zero terms.  Restructured (`hc_mix_down_dots_mt<NT>`, `hc_mix_up_collapse_q8_rows<NT>`), the op
reads **33.1 -> 21.6 us/run at `nt 4`** and the output is **bit-identical**.  Both are default-on with the kill
switches `GGML_CUDA_HC_DOWN_MT=0` / `GGML_CUDA_HC_UP_ROWS=0`.

Rules:

1. **A band op may serve the band from one weight read; the reduction order is what must be preserved, not the
   grid.**  In the multi-token down kernel the per-token item order, the dp4a dot, the warp butterfly and the
   fixed-order sum of the 8 warp partials are the per-token kernel's, so `lo`/`inject`/`y_v` are unchanged.  The
   F32 inject rows deliberately keep one block per `(row, token)`: they read a whole `hc_dim` row (~10x a down
   row) and a token loop there would set the kernel's tail.
2. **A completion counter's SCOPE follows what it guards, not its previous allocation.**  The `v = silu(lo/hc)`
   quantize needs all 32 rows of a q8_1 group; with one block serving every token it can count per GROUP instead
   of per `(token, group)` -- because each block writes `lo` for every token before it increments, so the 32nd
   increment implies all NT groups are complete.  `y_v` stays bit-identical, and the per-stream persistent
   counters still reset in the block that completes the group.
3. **An emulated reference reduction is a fixed expression, and deriving it is cheaper than running it.**  In
   the reference (rpb 16, vdr 2, lane `4*g + p`, items `8*warp + g + 64*m`) a row's 10 items fall in **exactly
   two** 8-item chunks -- one warp's partial each; every other lane contributes its initial `+0.0`, which cannot
   change a sum.  So the gate is the xor-16/8/4 tree over `g` then the xor 2/1 tree over `p`, per chunk, then
   their sum, and only the `4x32` gates per token need LDS.  A static_assert pins the geometry (`vdr == 2`,
   4 lanes/item): a future dense VDR must revisit this kernel rather than silently mis-index it.
4. **`GGML_CUDA_HC_UP_FUSE` is subordinate to the rows kernel.**  With `HC_UP_ROWS` on (the default) the
   sub-block count is not read -- the rows kernel is one 512-thread shape -- so `HC_UP_FUSE=2/3` no longer
   changes the launch.  `HC_UP_FUSE=0` (the two-launch path) is unaffected.  A tuning knob that a newer default
   shadows must say so where the knob is documented, not just in the code.
5. **A 32-bit shuffle mask is not a portable mask.**  The submission's five `__shfl*_sync(0xFFFFFFFF, ...)`
   calls are a HARD build error against HIP's `amd_warp_sync_functions.h` (`sizeof(MaskT) == 8`) on ROCm 10.1,
   10.2 **and** 7.14 -- the delivery's `ggml_cuda_shfl()`/`ggml_cuda_shfl_xor()` wrappers (64-bit full mask,
   same lane moves as the DPP-XOR path) are the only sanctioned form.  "Measured on ROCm 10.1" in a submission
   is a claim about a build; check that the build exists.
6. **The instrument for a band-op change is the perf table plus the acceptance census, not an end-to-end t/s on
   a host-expert shape.**  This box has no fully-resident Q8_0-hc qwen4exp config (the only two are 106/130 GB),
   so `tg128` reads **+2.3 %** against the op's real win; the four-axis `draft-mtp` acceptance is identical to
   the digit (0.96652 / 0.36678 / 0.63463 / 0.78675 with equal totals and mean lengths) while generation moves
   +3.7 to +6.3 %.

Record: `WORKLOG.md` 261121.018; `benchmarks/2026-11-21-hc-mix-band-kernels.md`; `ENVIRONMENT.md` §7.7;
`archive/work/hc-mix-q8/pr344/` (the submission verbatim, with the mask fix marked).

## r340 — the fusion-alias gate's text oracle, and the hipBLASLt candidate container (tooling + toolchain, no patch change)

1. **A fused launch and its unfused chain are one kernel SET only while nothing else selects between them.**  The
   issue-#287 gate asserted `fused == GGML_CUDA_DISABLE_FUSION=1` text at `-ub 14` and held from r239 to r323; at
   the r328 base the **unfused chain** moved (194 chars `dfe59c9c5257`) while the fused arm kept the r239 anchor
   (193 `3fab4da87774`).  Cause, measured: the chain's per-op `MUL_MAT_ID` is placed and partitioned under block
   09's host-offload relaxation (issue #323, `GGML_MOE_HOST_OFFLOAD`, ON since r307): `GGML_SCHED_DEBUG=2 -lv 5`
   counts **958 splits** with it and **1443** without at that shape, and `GGML_MOE_HOST_OFFLOAD=0` reproduces the
   anchor in the chain 3/3.  `GGML_OP_OFFLOAD_MIN_BATCH=1` and `=999` (all-GPU / all-CPU per-op placement) both read
   the *chain* text, so it is the partitioning's kernel-set choice, not a CPU-vs-GPU kernel difference.  The
   divergence is rounding level -- mean KLD **0.000002** / same-top-p **100.000 %** (0.000008 with the gate's pins)
   -- one near-tie at the end of a 40-token run, so a text hash is the wrong instrument: the oracle is the KLD and
   the exact-bytes claim belongs to the fused arm's own anchor.  The same-kernel `-ub 8` reordering control at that
   shape reads **0.003746 / 93.701 %**, which is what makes 0.001 / 99.9 % a decidable limit.
2. **hipBLASLt 1.5 has no per-start solution choice to make on the shapes we reach, and the delivery barely
   reaches hipBLASLt at all.**  With `HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=1` (the `HIPBLASLT_LOG_MASK=160` of
   the 2026-10-21 record emits nothing on either line today): ROCm 10.1.0 (1.4.1) offers **18 candidates** per call
   for the F32 GDN `ssm` shape (`140215, 140219, 140218 …`, tag `Equal`, one `getAllSolutions sol-idx` line each),
   while ROCm 10.2.0 (1.5.0) offers **exactly one** (`getBestSolutions(): sol-idx = 138457`, tag `Others`, no list)
   in 320/320 calls over 8 starts -- so the container `c50ea60130f` stably sorts is already collapsed on 10.2 and no
   fresh start can pick differently there.  Meanwhile the delivery default (block 06's `mmf32` claiming the F32
   router/GDN shapes, block 10's GDN epilogue fusion removing the per-layer `ssm` GEMMs) dispatches **0** hipBLASLt
   GEMMs on a 35B-A3B prefill and 40 with `GGML_CUDA_MMF32=0`.  Consequence for doctrine: §44's pin is *not needed*
   on 10.2 for these shape classes, stays a valid control on <= 10.1, and can be retired only on a line whose
   candidate list has been measured (`scripts/check-hipblaslt-lottery.sh`); an arm that reads one logits class says
   nothing either way -- 68 starts on 10.1 and 52 on 10.2 across five arms read one class each, including a gemma4
   F32 router arm and two BF16 dense prefill arms.

Record: `WORKLOG.md` 261121.020 (the gate oracle) and 261121.021 (the lottery + the validated lines);
`GREEDY-PURITY.md` §44 addendum (r340); `BASELINE.md` (*Validated ROCm lines*); `MANIFESTS.md` gate 25.

## r341 — the re-base file-set reconciliation: a shrunk rung is caught by comparing ITEM SETS, not by a build, and the noisy leg is the symbol, not the line count

`TODO` #358 retired.  `scripts/check-rung-file-sets.py`, wired into `validate-set.sh` (check 9) and
`validate-gpu.sh` (step 1b); `MANIFESTS.md` gate 38; decision rows in `patches/FILE-SET-RETIREMENTS.tsv`.

1. **The instrument is the item set, and the comparison is against the PREVIOUS RELEASE, not against the fork.**
   Each block is reduced to five item kinds — the files it touches with the number of lines it *adds* to each,
   the `#define` names it adds, the env names it reads (`getenv("NAME")`), the long options it adds to
   `common/arg.cpp`, and the block itself — and compared with the same block of the release being replaced.
   Restoring from the tree is the other direction and does not work: the r331 restore started from "the old
   chain's file" and *proved* it byte-for-byte, which is only possible because the old patch still existed.  So
   the old side defaults to the newest `v*` tag before `HEAD` (`--old` overrides), and a **shallow clone skips
   loudly** — the hosted tier-1 checkout has no tags, tier 2 (a full clone, `radiance`) and the release flow run
   it for real, and a full clone with no resolvable tag is a FAIL.
2. **The file leg alone is not enough, and the r328 incident says why in three parts.**  `ggml/src/ggml-alloc.c`
   went 275 -> 155 added lines across the set (block 00's 120 gone, blocks 09/10/12 unchanged), so a
   "file touched by the old set and by no new block" test misses it: the file is still touched.  What fires is
   the **block-level** file drop combined with a set total that did not grow (SHRUNK), and independently
   `LOSS` on `GGML_GLU_INPLACE`, the env switch the lost half defines.  The same two legs catch the second r331
   finding (a lost `GGML_CUDA_DISABLE_ROPE_SET_ROWS` switch inside a file the block still edits) — a build, the
   set's op suites and every text arm are blind to it, because the switch is presence-only at the call site.
   The third (a renamed `GGML_CUDA_CC_DGX_SPARK`) is a *decision* row, not a loss, which is the point of rows.
3. **The line-count leg is calibrated, and 50 % is not a guess: 70 consecutive release transitions, two
   firings.**  r263..r339 (every adjacent tagged pair): `0 unexplained` on **68** of them; the two exceptions are
   r325->r328 (the real defect: 3 LOSS + 5 SHRUNK, i.e. the three r331 findings plus the block-12 -> block-16
   server move and the D1 `mmf.cu`/`mmf.cuh` retirement) and r287->r288 (the block re-org: 2 blocks renumbered,
   a documented decision).  A per-block-file drop of `--shrink-pct` (default 50) only counts when the *set's*
   total for that file did not grow, so a hunk that merely moved blocks is `MOVED`/INFO and never fails; the
   `mmid.cu 204 -> 100` re-cut lands exactly on the boundary and is a review item, which is what a re-base owes.
4. **A decision row is scoped to the transition that made it.**  A row names the `from-release` it was decided
   against and applies only while the check is comparing against that release, so it can never suppress a later
   release's finding by (block, kind, item) coincidence; rows from another transition are counted in the summary
   and printed, never silently ignored.  `patches/FILE-SET-RETIREMENTS.tsv` is therefore a **live ledger**, and
   the durable record of the decision is the `WORKLOG.md` entry.
5. **A chain is the DEFAULT blocks only.**  `--new-chain <repo> --new-base <sha>` reads the block chain out of a
   llama.cpp checkout (block number from each commit's `rdna-boosts: block NNNN:` subject, `block 00`..`block 16`
   and `block 0300` all normalising to the four-digit file prefix).  It exists for the ladder — after porting a
   rung, `--rung NNNN` reconciles that rung alone — and it deliberately drops the optional `01xx` artifacts from
   the comparison instead of reporting them as GONE by construction (`apply-all.sh --include-optional` is a
   consumer action and never part of a chain).  Reconciling the fork's own `rdna-boosts` branch against the
   release tag it was cut from reads **0 findings, every per-block added-line count equal** — an absolute oracle
   for "the branch and the patch set are the same set", which the tree hash already gives and this now gives
   per block.

Record: `WORKLOG.md` 261121.022; `MANIFESTS.md` gate 38; `scripts/README.md`; `patches/FILE-SET-RETIREMENTS.tsv`.

## r342 — a `#if defined(SYSTEM_MACRO)` guard silently compiles a feature out, and a NEED that MOVES to a new file is invisible to every per-file diff (issue #348)

**The defect.**  `server_prompt_cache::take()` moved to `tools/server/server-slot-sched.cpp` at the r328 re-base.
Its huge-page branch was guarded by `#if defined(__linux__) && defined(MADV_HUGEPAGE)`, and `#include <sys/mman.h>`
stayed behind in `server-task.cpp`, where the function used to live.  `MADV_HUGEPAGE` was therefore undefined in
the new file, the branch dropped out of every Linux build **with no diagnostic**, and `LLAMA_SERVER_CACHE_HUGEPAGE`
(default on, PR #248) did nothing from r328 to r339: a 64 MiB+ prompt-cache state buffer went back to 4 KiB zero
filling.  The only detector was `nm -D --undefined-only libllama-server-impl.so | grep madvise`, and only a
reporter ran it (issue #348).

**Why nothing caught it, which is the durable part.**

1. **A guard on a system macro is the trap, not the safety.**  `defined(MADV_HUGEPAGE)` reads as "is the header
   here?", but the answer is silently "no" when the include is missing, and a missing `#if` branch is not a build
   error.  **Rule: guard the OS (`#if defined(__linux__)`), never the macro a system header declares.**  Then a
   lost include is a compile error at the next build, which is the only signal that costs nothing.
2. **A need that MOVES is invisible to a per-file diff.**  `check-rung-file-sets.py` (gate 38) reconciles files,
   `#define`s, env vars and options: the include was still present -- in the file the code had LEFT.  Nothing
   about the *inventory* changed, so a per-block or per-release inventory check can never see this class.
3. **A build cannot see it** (the branch is not compiled) and **a text/logit gate cannot either** (the surviving
   path is correct, just slower).  It is a performance defect that only a mechanism census or an import survey
   exposes.

**The instrument.**  `scripts/check-system-headers.py` (check 10 of `validate-set.sh`): a table of
symbol -> declaring header plus the macro families (`MADV_*`, `MAP_*`, `POSIX_MADV_*`, `O_DIRECT`, ...), each
checked against the file's **include closure** (project headers followed transitively), plus the guard rule above.
Scope matches `check-windows-portability.sh`: a file the delivery adds is judged whole, an edited upstream file on
the lines it inserts (that is what keeps it CI-clean -- upstream has module-local `mmap()` and string literals full
of `O_DIRECT`).  Negative control: the r339 tree reports exactly the two `server-slot-sched.cpp` sites, the fixed
tree reports none.

**The same re-base dropped a second thing of the same shape (found by this session's scan, fixed at r342).**  Block
02's `throw` rejecting `draft-mtp` + `draft-mtp-adaptive` (one impl class, one `ctx_dft`, so two impls advance the
same draft context twice) vanished at r328 too: it was a statement inside a re-expressed hunk, i.e. not a file,
`#define`, env var or option, and a missing `throw` builds and runs.  **Rule: a behaviour that guards a
configuration must be TESTED, not grepped** -- `tests/test-spec-type-exclusion.cpp` asserts the pair throws, each
type alone does not, and the neighbouring DFlash-family rule still throws (no model needed: the rejection precedes
every context use).  A scan of the whole r327 -> r339 transition (delivery-owned identifiers, then delivery-owned
lines against the new base) found nothing else beyond that pair and the known r331 items.

**Numbers (r342, gfx1201, dense 4B Q8_0, `-c 32768 -np 2 --cache-ram 4096`, five ~11K-token prompts so the unified
KV evicts a ~0.9 GiB slot state).**  `madvise@GLIBC_2.2.5` present in `libllama-server-impl.so` (absent on r339);
`AnonHugePages` from `/proc/<pid>/smaps_rollup` **1 142 784 kB** at the default and **0 kB** with
`LLAMA_SERVER_CACHE_HUGEPAGE=0`; whole-slot saves **20.74 / 20.78 / 21.46 ms** against **54.23 / 52.78 / 49.65 ms**
(-61 %).  The r173 numbers (3.47 GiB alloc+zero 470 -> 125 ms; a 100K-token save 828.1 -> 497.0 ms) still stand.
A MoE host-expert shape cannot isolate this measurement: its `AnonHugePages` is ~48 GiB from the block 09/15 pool.

Record: `WORKLOG.md` 261122.001; `MANIFESTS.md` gate 10 / block 02 + block 12 rows; `ENVIRONMENT.md`;
`patches/README.md` (blocks 02 and 12).

## r343 — the slab headroom is sized for allocations made AFTER the slab extends, and a second context's graph is the big one

**Fact.**  `GGML_CUDA_SLAB_HEADROOM_MIB`'s default is `1408 + 128 x n_gpu` MiB (1536 / 1664 / 1792 on 1 / 2 / 3
GPUs), raised from `1152 + 128 x n_gpu` (1280 / 1408 / 1536) so that **one GPU now gets the value that used to be
the 3-GPU one**.  The headroom is the steady-state free VRAM the `slab_extend` reclaim leaves outside the slab, and
it is drawn on by *every allocation made after the extension* — including one that cannot be redirected and is not a
prefill transient at all.

**Why it moved, measured.**  A **second context's graph** comes up mid-run (a draft context;
`archive/work/prefill-decode-arms`' decode arm) and its reserve is a single large allocation.  The two-arm server's **2-row
decode graph asks for 741.48 MiB in ONE allocation**; at 1280 it failed with `ggml_cuda_device_malloc: 741.48 MiB
allocation failed ... the movable-boundary slab holds the VRAM; the arena cannot yield physical back to the driver`
→ `failed to reserve graph buffers` → `ret = -2`, on the *first 2-row step* (a 1-row decode step had run for 5.5 s
immediately before), while `GGML_CUDA_SLAB_HEADROOM_MIB=1536` ran the identical shape with zero OOMs.  So the r206
sizing (T = a prefill transient, ~300 MiB + one hipBLASLt workspace) is not the right T for this class: a late,
whole-graph allocation is larger, and it is the general case rather than one campaign's.

**Cost, measured.**  The slab leaves 256 MiB more outside, so its arena is 256 MiB smaller — always: `size(VA)` 28.12 → 27.88 GiB, and arena 27904 → 27648 MiB on a `-c 8192` shape.  What that costs downstream depends on what bounds the expert cache.  On the campaign's product shape (35B-A3B Q4_K_M, `-ncmoe 40 -lme auto`, `MOE_EXPERT_CACHE_MIB=5000`, `-c 16384 -b 2048 -ub 512`) it was **free**: the expert cache's arena 4957.1 MiB, `h` 0.7159, fills 17738 and evictions 9578 were **identical** at 1280 and 1536 and decode was flat (51.12 vs 51.36 t/s) — the withheld MiB came out of the slab's unclaimed reserve.  Where the arena is instead what bounds the cache, r206's ~1 pt of hit rate per GiB held back applies.  It also makes the slab **decline** on a card packed to the edge slightly more likely — the documented r206 trade, taken deliberately.  **No cold-shape text oracle exists for this default**: `-c 8192` greedy text read 3 distinct hashes in 3 cold runs *of the same arm* (and the text follows the arena placement, `GREEDY-PURITY.md` §89's class, whenever the residency changes), so any 2-arm text comparison on a cold short-prompt shape is vacuous — use a warm oracle.

**Rule.**  When sizing anything that must fit *after* the slab has extended, size for the widest allocation that can
appear then (a second/third context's graph reserve, per active-slot width), not for the widest prefill transient.  A
`ret = -2` from `llama_process` on the first step that changes a context's batch WIDTH is this class; raising
`GGML_CUDA_SLAB_HEADROOM_MIB` is the lever that does not cost arena on a budgeted shape.

Record: `WORKLOG.md` 261122.005; `ENVIRONMENT.md` §1.3; `archive/work/prefill-decode-arms/RESULTS-2026-11-22.md` pass 5 §4.

## r344 — on an APU GTT **arena**, the read transport is the page-cache decision, `dio` keeps the mapping, and the warning that should have named it never fired (block 15, issue #347 follow-up)

1. **`--load-mode` pins the ARENA's read transport like any host tier** (`src/llama-model-loader.cpp:1745-1779`,
   `moe-expert-cache.cu:2626-2706`): `mmap` / `mmap+mlock` / `mlock` => `POOL_TR_MAP`, **pinned, no monitor**.
   Every arena miss is then a `memcpy` out of the pageable model mapping, so the expert pages enter the page cache
   as **MAPPED** pages.  On a box whose page cache cannot hold the set (a cgroup `memory.max`, a >RAM model) that
   cache is the only elastic tier and it churns.  Measured (gfx1151, ROCm 10.2, GSQ-IQ3_XXS 40 925.6 MiB set,
   `-lme 24`, `MemoryMax=44G`, no swap, cold cache, 5 298-token prompt + 64 greedy tokens): `mmap` 216.7 t/s
   prefill / 9.7 t/s decode, 17.4 GiB mapped GGUF, 16.54 M refaults, 8 772 major faults; `buffered` 265.4 / 18.6,
   0.7 GiB, 4.04 M, 152; `auto` (forced map start) 255.4 / 18.0, 3.4 GiB, 5.53 M; `dio` 205.6 / 15.7, 0.9 GiB,
   **0** refaults.  `-lme 28`: `buffered` 314.6 / 21.5, `dio` 291.8 / 20.4.  The census names the fill cost:
   `direct slot fills ... 4 668 MiB/s` (buffered) against `per-expert mapping reads (bounce) ... 1 482 MiB/s`
   (map) and `per-expert storage reads (bounce) ... 1.93 GiB/s` (dio); the prefill span stream is 6.9 GiB/s
   buffered against 3.65 GiB/s O_DIRECT.
2. **`dio` does NOT drop the mapping on the ARENA arm** (the correction to the first version of this finding):
   `plan_gtt_expert_store()` force-maps the model as the arena's fill master (`use_mmap = true`, with a WARN),
   whatever the load mode, so `--lazy-mode on` still serves the PLE from the mapping and `anon` stays arena + ~0.2
   GiB.  The mapping is forced only in the **arena** arm; on the **placement** arm (budget covers the set) `dio`
   still means no mapping, and the lazy PLE would become resident.  Do not generalise the one to the other.
3. **The `buffered` vs `dio` order is the DRIVE's O_DIRECT latency, not a universal rule.**  On this box's Crucial
   T500 `buffered` wins at 24 GiB (+29 % prefill, +18 % gen) and 28 GiB (+7.8 %, +5.4 %); at a 36 GiB cap they tie
   on generation (15.7 both) and `buffered` is +7 % prefill.  `dio` removes every file-page refault, so its window
   is a tight page-cache budget or a controller whose O_DIRECT latency is low (the maintainer's SN850X finding);
   `buffered`'s window is a drive where the deep buffered reads and the direct slot fill are much faster.  All
   three transports read the same text (`96533be7cb80`), and the text is not a transport oracle.
4. **The pinned-transport memory-tight warning never covered the arena.**  It required `planned_pool > 0`, and the
   arena has no host pool, so the one shape it exists for ran silently.  It now also fires for
   `g_gtt_arena && transport == map`, names the budget and the per-device set, and points at `buffered`/`auto`.
   `gate-uma-experts.sh` grew `mapwarn` (must fire) and `bufferedwarn` (must not) and asserts each arm's named
   transport; 54/0 with every text arm still `4f29f1e4a4d6`.

Record: `WORKLOG.md` 261122.004; `ENVIRONMENT.md` §20; `patches/README.md` block 15; issue #347 follow-up.

## r345 (2026-11-22, `WORKLOG.md` 261122.005) — the MoE gather's `n_split` cost the ROUTING SCAN, not the copy shape: one used-expert bitmap makes a deep split the best arm at both residency extremes (block 09, issue #350)

1. **`moe_cache_gather_kernel`'s parallelism is `n_experts * n_split`, and its per-block USED TEST was
   `O(n_used * n_tok)` — the same routing scan duplicated up to `n_experts * n_split` times.**  The copies shrink
   with `n_split`; the scans do not, so the terms move opposite ways as residency falls.  Measured 1 x R9700,
   gfx1201, `-b 2048 -ub 2048`, the reporter's shape (Flash-Next IQ3_XXS, 512 experts/table, arena 21881.8 MiB of
   40012.5 MiB host experts = **54.7 % residency**, `-ncmoe 47 -lme mirror`, 17 624-token prompt): `n_split` 32
   reads **1671.7 t/s** against **1719.3** at 4 and **1727.0** at 8, i.e. the shipped r325 default was the worst
   arm — the ordering issue #350 reports (~19 % on their Windows box, 2.8 % here).  One bitmap pass
   (`moe_cache_gather_mask_kernel`, per-block shared-memory bitmap + one atomic per WORD; a global atomic per id
   measured 0.11 ms/call) makes 32 the BEST arm: **1741.0** (+4.1 % over the shipped default), and on the
   released tree with no knobs **+3.6 to +4.4 %** across two 3-pair sets (1669.4/1668.3/1674.5 ->
   1740.8/1730.4/1720.2 and 1676.5 -> 1748.5/1751.3).  At 84 % residency (35B-A3B Q8_0, 256 experts) 32 goes
   3722.3 -> **3777.6** (+1.5 %).
2. **The bitmap is a TEST replacement, not a copy change, and the text says so byte for byte.**  `172 chars
   sha=de61b9b1bed7` (low-residency shape) and `206 chars sha=cf7f8b23f404` (high-residency shape) are identical
   across bitmap on/off and `n_split` 4/32; `gate-prefill-logits.sh --ab "MOE_GATHER_MASK=0"` on the reporter's
   shape reads mean KLD **-0.000000 / 100.000 % top-p**.  So a gather change owes the fill-path gates
   (`gate-host-pool.sh` 58/58) and the exact-fallback KLD point, not a numerics apology.
3. **Gate the pass on the scan volume it replaces, not on the split alone.**  The condition is
   `n_experts * n_split * n_used * n_tok >= 1e8` ids, measured: 6.7e7 ids reads **-1.2 %** (512 experts at
   `n_split` 8) and **-2.3 %** (256 experts at 16) because the pass is then latency on the critical path, while
   1.34e8 reads **+0.7 %** (512 at 16), **+1.5 %** (256 at 32) and **+4.1 %** (512 at 32).  Two consequences:
   `n_split` 4-8 stay exactly the pre-r345 path (so the documented A/B arms keep their numbers), and the
   decode/verify band is five orders of magnitude below the threshold, so no decode op pays a launch — a 9..16
   row verify is 2.1e6 ids, which also means no captured graph changes shape.
4. **A constant tuned on one residency is not a constant; find out which term it is buying.**  The bitmap turns
   `n_split` 64 at 84 % residency from **3412.6 t/s (-8 % against 16)** into **3817.8 (+12 % over that same
   arm)**, which localises the r325 gain to the SCAN and says the answer is the mechanism, not a value.  Keep the
   constant at 32 (best at both extremes once the scan is gone) and keep `MOE_GATHER_MASK` as the kill switch.
5. **A lazily-allocated device buffer on a gather path must FAIL SOFT.**  `gather_mask_get` returns null on a
   failed `cudaMalloc` and the caller falls back to the scan, so a first use inside a graph capture is a slower
   call rather than a capture-time allocation.

Record: `WORKLOG.md` 261122.005, `benchmarks/2026-11-22-gather-bitmap.md`, `patches/README.md` block 09,
`ENVIRONMENT.md` §2 (`MOE_GATHER_MASK`, `MOE_GATHER_NSPLIT`), `GREEDY-PURITY.md` §104, issue #350.

## r346 (2026-11-22, `WORKLOG.md` 261122.027) — the two-arm server: a memory module that does not implement the shared view is SILENTLY blind, a second context's reserve must be DECLARED, and the pair's residency class is the arena's (block 17)

Promoted the `prefill-decode-arms` campaign as official **block 17** (`--prefill-decode-arms`,
`--prefill-decode-arms-mode off|on|auto`, **default OFF / opt-in only**).  The engineering facts:

1. **Every memory module must implement the shared view, or the pair is blind — and the failure is SILENT
   GARBAGE, not an error.**  `create_memory()` gave the decode arm a shared view of the plain KV and the plain
   hybrid memory; `llama_memory_hybrid_idx` (qwen4exp: attention KV + recurrent state + lightning indexer KV)
   and `llama_kv_cache_iswa` (gemma4, sliding window) did not, while the server still printed
   `KV shared, one allocation owner`.  The decode arm then ran on an **empty** attention cache and an empty
   indexer cache, so an **82 615-token** request answered as if only its tail were present ("the model cannot
   see the files") and gemma4 emitted `COCOCOCO…` on **1, 2 and 3 GPUs**.  Both are one bug class, found twice
   (the second time by the coverage matrix, not by any gate).  The engine's
   `W create_memory: ... implements no shared view -- this context has its OWN cache` line existed and **nothing
   read it**; it is load-bearing now, and `mem_other_used` makes it impossible for it to lie.  A co-owner also
   has to be **forwarded to both children** of a composite memory (an iswa pair has one allocation owner PER
   cache, so the request-boundary handshake must ask both) and an existing **mirror** path (gemma4's assistant)
   must keep passing `other_coowner = false`, because a mirror that touches the source's bookkeeping reads
   garbage.  After the fix: 30 shared layers in the log, `h-on` == `h-off`, content byte-identical to the
   single-context server.
2. **A second context's graph reserve must be DECLARED before the arena claims the space, and a PARTIAL
   declaration fails.**  The movable-boundary slab is created at load; the decode context's graph reserve arrives
   after it.  On a host-expert tier that made the arena refuse the second context (`the decode arm's step
   failed`, `SLAB_INIT ... aux=0`).  The reserve is now declared up front the same way an embedded MTP head's
   draft context declares its own, with **no KV term** (the pair shares one cache): a flat **3584 MiB** via
   `MOE_EXPERT_CACHE_AUX_RESERVE_MIB`, and a manually set `GGML_CUDA_SLAB_RESERVE_MIB` still wins.  Measured
   need for the *derived* form (a follow-up, `TODO` #366): 1792 MiB Flash-Next, 9.52 MiB on the 4B.
   *(Superseded in part by r350: the pair accepts a speculator now, so the derivation charges the decode arm's
   reserve PLUS the draft context's — two separate late contexts — instead of keeping the larger of the two.)*
3. **A BUDGETED arena puts both arms in one RESIDENCY CLASS; an `auto` arena does not.**  A text reference is
   per device count, per split **and per residency class** (§89): with `MOE_EXPERT_CACHE_MIB=13653` (40 GiB
   total) the expert cache's hit rate is **identical between arms** (`h` 0.9352 / 0.9352) and the pair's text
   equals the single context's; against an `auto` arena the two arms land in different classes and the
   comparison is not an oracle at all.  So a host-tier text claim must name the budget, and the maintainer's
   `-c 204800 --parallel 3` 3-GPU tensor shape passes there too.
4. **The indexer pool is laid out from CELLS and is PER CONTEXT, so a restore or a handover must stale it.**
   A checkpoint restore moves a sequence's attention cells behind the pair's derived indexer pool, so a
   `cache_prompt` repeat could not see the prefix it had just reused even after the shared view was fixed.  The
   handover therefore re-derives it; a memory whose second cache MIRRORS the attention cells must also never be
   told to park the attention cache alone (block 12's r146 rule, restated here because the pair reaches it).
5. **Default OFF is bit-identical, and the oracle for that is a REPRODUCTION of the previous release's
   record, not just an A/B.**  Flag off, both ROCm lines: prefill-logit KLD **0.000000 / 100.000 %** and the
   same `base_sha256` as the r343-era reference (`ce8bb26f…`), 4/4 shipped shapes byte-identical, and the MTP
   four-axis protocol reproduces the **r328 record's text hashes and acceptances exactly**
   (`218515b89b31`/`a87c4318b649`/`27d74f92ec5c`/`6fa43035b8d9`, acceptance `0.34306`/`0.96652`/`0.63593`/
   `0.73478`, q8_0 `3a70dd34be92`/`b29f3176fa31`/`45f5e6db9dc2`/`a87c4318b649`).  A block that touches
   `create_memory()`, the KV allocator and the memory modules the *draft* context also uses owes all of that —
   the MTP arm is a real risk, not a formality.
6. **The pair's own contract, so it is not re-litigated:** the prefill/decode overlap costs **-21…-24 %**
   prefill (flat from `-ub 2048` up; `-ub 4096` is the sweet spot) to hold a decoder at 47-49 ms p50 / 64-78 ms
   p95 through a 2-5 s prefill where the single context stalls 300-880 ms; a `-sm tensor` hybrid state boundary
   takes the **serialised** move (~74.6 ms, 112.57 MiB) because a meta-aware device copy is a follow-up
   (**superseded by r347**: it is a 1.4-1.9 ms device copy); a
   co-batched two-slot pair is **stress only, never a text oracle** (§63); `-ib` and `LLAMA_SERVER_PREFILL_DUTY`
   are **deprecated and inert** on this path (both bound a stall inside a SHARED step); and the pair refuses
   `--spec-type`, so speculative decoding with the arms is a separate campaign.

Record: `WORKLOG.md` 261122.027, `patches/README.md` block 17, `ENVIRONMENT.md` §23,
`archive/work/prefill-decode-arms/` (`RESULTS-2026-11-22.md` passes 2-18), `MANIFESTS.md` block-17 row.

## r347 (2026-11-22, `WORKLOG.md` 261122.036) — the two-arm server becomes SAFE to run: five structures it shares from two threads, a reserve it creates TWICE, and a state handover the Meta buffer can finally copy (block 17 amendment)

The block-17 amendment (same block, no new block, no renumbering).  `TODO` #373, #365 and #366 all shipped with it.
The engineering facts, in the order they were learned:

1. **The pair runs two `llama_process()` calls on structures that were never shared before, and every one of them
   had to be given a lock or an owner.**  Five, all reproduced before they were fixed
   (`archive/work/prefill-decode-arms-followups/repro-373/`): the **KV cell map** (`apply()` and `init_batch()`'s rollback
   snapshot — the qwen4exp `qwen4exp.cpp:1138` abort with the children ~1078 cells apart, and a libstdc++
   red-black-tree SIGSEGV inside `llama_kv_cache::prepare()`); the **Meta split-state cache** (per *buffer*, hence
   shared through the KV cache's buffer: the released build's SIGSEGV in `ggml-backend-meta.cpp`); the
   **async-input ring** (`cudaDeviceSynchronize()` on its wrap path); the **VMM pool's peer grant** (all peer
   descriptors in ONE `cuMemSetAccess` + `CU_CHECK`); and the **synchronous host paths** (`cudaStreamPerThread` is
   HIP's sentinel for the legacy default stream, which a peer context's graph capture claims).  The fixes:
   `llama_memory_i::apply_mutex()` (one lock per memory **GROUP**, shared through `create_memory()` — a take reads
   the PEER's map, so a lock there would have to span two groups, which is why `seq_state_take()` and
   `alloc_sync()` are deliberately unlocked and rely on the server's quiet window); the split-state cache per
   **view slot** (`[4][8]`) with the slot bound for every graph build/alloc (`meta_bind_view_slot()`); the ring
   waiting on each **origin's own** stream; the pool grant best effort with a per-pool mask and
   `ggml_cuda_peer_copy_ok()` answering for pools as well as slabs; and `ggml_cuda_host_stream()`, one private
   non-blocking stream per **(thread, device)**, for every synchronous host path.  **A lock per memory INSTANCE
   would not have been enough**: the pair shares one map, so the lock has to live with the group.
2. **A second context's graph reserve has to cover TWO late narrow views, because the pair creates the post-prefill
   re-reserve as well as the decode context's first one.**  The flat 3584 MiB (`r346`) covered one.  Measured on
   Flash-Next `-ncmoe 48 -c 131072`, arena 13.3 GiB, 3 slots, 71k + 23k + 800: **3584 fails, 4096 fails, 5120
   clean**, both views 1430.2 / 1430.6 MiB — hence `common_arms_dec_reserve()` = flat floor + a **width-keyed
   tier** (`0` at `-ub <= 1024`, 1536 MiB at `>= 2048`, +256 per doubling).  A *reserve* is capacity, not state:
   too small and the arena refuses a graph (a hard failure), too large and VRAM is wasted, and the width is what
   predicts it.
3. **The Meta buffer had no copy hook, so a `-sm tensor` hybrid paid a full host round trip at EVERY request
   boundary — and it is now a device copy.**  `seq_state_take()`'s 2-D row views are built **out of graph**, so no
   `simple_tensor` container knows them and the host paths cannot resolve them; that is why `cpy_tensor`/
   `cpy_tensor_n` were null and `can_take_batched()` refused.  The hook is per **child**: the split state of the
   view's ROOT (which IS registered) says which byte range of a row belongs to which child.
   **The non-obvious half: both sides of the copy are children of the SAME layout, so a child's covered rows are
   ONE contiguous range even when the split state is segmented/repeated** — the state of Qwen3.8-Flash-Next is one
   segment **repeated 3 or 5 times** (interleaved inside the row, identically on both sides), and walking that
   interleaving costs `nr[0]` copies per row per child: a planner that walked it measured **3.5 ms**, and one that
   copies the child's range measured **1.4-1.9 ms**.  Against **72-77 ms** serialised (112.57 MiB).
   `LLAMA_RS_VERIFY_TAKE=1` on the batched arm: **0 of 118 039 420 bytes differ**, cell metadata included; and the
   same turn driven once per arm, serially and greedily, gives **byte-identical** per-take destination dumps
   (`LLAMA_RS_VERIFY_TAKE_DUMP=`) and identical text.  A shape the planner cannot express is staged through the
   host via the root — **never partially copied**, because a half-moved recurrent state is fluent-but-wrong text
   (§89).  `cpy_tensor` (singular) stays null on purpose: `can_copy_tensor()` is satisfied by either hook, and
   adding it would silently reroute the KV-park and state-staging copies that take the host path today.
4. **A sequential arms oracle cannot see a request-boundary defect; that is a gate gap, not an oracle choice.**
   Every arms oracle the campaign had (the 16/16 coverage matrix, the soak, the reuse gate) drove requests **one at
   a time**, so nothing ever ran two arms' prefills at once — the shape the reported P0 lived in.  The arm that
   does is `scripts/gate-arms-soak.sh boundary` (gate 17b): two prefills in flight plus a decode, pad buckets
   shrunk to 256, `BB_ROUNDS >= 3`, and a fourth **sequential fresh** request compared against a single-context
   server.  **A text claim about the pair is only valid on a serial, greedy arm** (§63/§38): at the runme's
   temp 0.7 two arms differ in sampled tokens from kernel-shape noise alone, and their per-take dumps differ
   because the cell placement follows the interleaving.
5. **A small model over a 3-way split is NOT a shape anyone runs, and it faults on the released build too** — a
   4B on 3 devices `-sm tensor` with the pair dies in `llama_kv_cache::find_slot()` (a null deref at `+0x6a0`,
   `segfault at 25 ip … libllama.so[3561d0]`) with or without the fix set.  So the boundary arm must **not**
   default to the fast 1-GPU soak model; it defaults to the validated 35B-A3B hybrid split.  General form: a gate
   arm's *shape* has to come from a validated configuration, and "it crashes" is not a finding until the shape is
   one the repo supports.
6. **`TEMP` in the environment breaks the ROCm 10.1 runtime at LOAD, inside its LLVM-based JIT.**  A harness
   variable named `TEMP=0` (hipBLASLt's Tensile cache path) made every model load die with a stack-write fault in
   `libLLVM.so` (`segfault at <addr> ip … libLLVM[…815cd9] error 6`, i.e. a write at `sp`), reproducibly, and a
   64 MB stack made it *survive* — so it is a stack exhaustion in the JIT, not a bounds bug we can fix.  Recorded
   because it reads as a defect in the change under test: the harness variable is `ARM_TEMP` (`WORKLOG.md`
   261122.014 had the `hipStreamCreateWithFlags` face of the same landmine).

Record: `WORKLOG.md` 261122.036, `patches/README.md` block 17, `ENVIRONMENT.md` §23, `MANIFESTS.md` gates 17/17b,
`archive/work/prefill-decode-arms-followups/` (§W2/§W9 + `repro-373/`).

## r348 (2026-11-22, `WORKLOG.md` 261122.038) — a cross-model `ctx_other` is a DRAFT, never a memory co-owner: block 17's generic arm threw on every separate MTP head (block 17 amendment, issue #351)

The block-17 amendment (same block, no new block, no renumbering).  GitHub issue #351, reproduced and fixed on
this box with the reporter's own shape.

1. **`ctx_other` has TWO meanings and only one of them needs the same model.**  Block 17's generic arm lifts the
   gate so that a second context of the SAME model can view the first one's memory (that is the one place a decode
   can be submitted from while `llama_process()` is open).  A context of a DIFFERENT model is the other meaning:
   a DRAFT head.  The external shared-NextN MTP GGUF (`-md` / `--spec-draft-model`) loads as its **own
   `llama_model`**, borrows the target's `token_embd` / `lm_head`, and `common_speculative_init_result` sets
   `cparams.ctx_other = ctx_tgt` **unconditionally** (`common/speculative.cpp`) -- so the r346 same-model check in
   `llama_context` threw at the draft context's construction:
   `ctx_other must be a context of the same model`, then `failed to create MTP context`, and the server never
   reached readiness (also with `--prefill-decode-arms` off, because the check is in the constructor, not the
   pair's code).  The arch-gated arms above it (gemma4-assistant, eagle3, dflash, qwen4exp) are exactly the
   drafts that ASK for a cross-model `ctx_other`; a plain NextN MTP head is not one of them, so it fell through
   into the generic arm.  **The rule now: the generic arm adopts a same-model `ctx_other` only; a cross-model one
   is declined (its own cache, which is what the pre-r346 tree did) and `ctx_other_share_memory` with a
   cross-model `ctx_other` is refused, because another model's cells belong to ITS layer layout.**
   Deliberately NOT "adopt it anyway": adopting it flips every `cparams.ctx_other != nullptr` site in the fork
   (`sched_alt_for()`'s second scheduler, the slab's narrow-2 draft routing, `params_mem.kv_pool`) for a shape
   that has run without it since the block existed, i.e. an unvalidated behaviour change on hardware this repo
   does not validate (the reporter's 780M).  A future "an external head is a draft too" change is a measured
   change, not a regression fix.
2. **The draft really drafts, and it changes no token.**  Flash-Next GSQ-RCO IQ3_S + the Unsloth shared Q8_0 MTP
   head, `--spec-type draft-mtp --spec-draft-n-max 2`, `-ncmoe 48`, pins `MOE_EXPERT_HEATMAP=load` +
   `ROCBLAS_USE_HIPBLASLT=0`: `--spec-type none` and `draft-mtp` read the SAME greedy text, byte for byte, on
   1 device (`552 chars e7aae6034b4f`) and on 3 devices `-sm tensor` (`645 chars af9171d90336` -- a
   device-count-specific text, as any tensor split is), repeated; draft acceptance **0.77-0.85**, mean accepted
   length 2.5-2.7, and decode **34.0 -> 48.0 t/s** (1 GPU) / **40.0 -> 65.3 t/s** (3 GPU tensor).  The external
   head is inert to the target's text and worth +41..+63 % decode on this shape.
3. **A gate that can take an external head and defaults to an embedded one cannot see this class.**  `DRAFT=` /
   `-md` is optional in `gate-state-pipe.sh`, `gate-ckpt-reuse.sh`, `gate-park-idle-slots.sh` and
   `gate-spec-ckpt-device.sh`, so the r346 gate ladder drove them all with an embedded head and stayed green
   while every separate-MTP-head server was dead at load.  `scripts/gate-draft-ctx.sh` is the missing arm: the
   external head must load, its text must equal `--spec-type none`, the `-lv 4` census must show the constructor's
   DECLINE line and a non-zero acceptance, and no same-model refusal may appear.  Negative-tested on the r347
   build: **8 of 11 checks fail** (exit 1).

Record: `WORKLOG.md` 261122.038, `patches/README.md` block 17, `MANIFESTS.md` gates 17 (TIER A) / the block-17
row, `scripts/gate-draft-ctx.sh`, `GREEDY-PURITY.md` §106.

## r349 (2026-11-22, `WORKLOG.md` 261122.039) -- a diagnostic log line is COMPILED code on every platform: `ssize_t` is a type, so a portability lint that only knows calls and keyword macros cannot see it (block 17 amendment 2, PR #352)

1. **The Windows build break was in a `LLAMA_LOG_INFO` argument list, behind an opt-in diagnostic.**  Block 17's
   serialised state-move VERIFY prints a differing byte offset with `"first at %zd"` through `(ssize_t)`; MSVC/UCRT
   never declares `ssize_t`, so the Windows HIP toolchain stops at `src/CMakeFiles/llama.dir/llama-context.cpp.obj`
   with `error: use of undeclared identifier 'ssize_t'`.  Nothing on this box can compile that: every r348 gate ran
   on Linux/ROCm, where the line is fine.  Fixed as `(long long)` / `%lld` (the offset is never negative, and the
   `(size_t) -1` "no difference" sentinel still prints `-1` -- verified on the boundary arm with
   `LLAMA_RS_VERIFY_TAKE=1`).
2. **`check-windows-portability.sh` was blind to it by construction, not by an oversight in its list.**  It had a
   CALL rule (POSIX functions), a keyword-MACRO rule (`near`/`far`/`pascal`/`cdecl` from `<windows.h>`) and the
   separate include-closure check; `(ssize_t) first` is neither a call nor a macro, it is a *type name in a cast*.
   So the script now carries a third rule on the same Windows-compiled lines: **POSIX-only type names**
   (`ssize_t` today, the tuple is where a second one goes), with comments/literals already stripped (its own
   explanatory comment is not a hit) and `pool_ssize_t` untouched (no word boundary).  Negative control: on the
   **r348** tree it reports exactly one site (`src/llama-context.cpp:7284`); on r349 it is clean.
3. **A "verify-only" or "log-only" line is not exempt from the build.**  The general rule this records: a change
   review should not classify a hunk as "unreachable" because it needs an env var -- the compiler reads it on every
   platform the file is compiled for, and the only instrument for the Windows half is
   `scripts/check-windows-portability.sh` (run by `validate-set.sh`, checks 7/10) or a reporter's build log
   (`issue #231`, `#306`, `#310`, now `PR #352`).

Record: `WORKLOG.md` 261122.039, `scripts/check-windows-portability.sh`, `scripts/README.md`, `MANIFESTS.md`
(the portability-gate row), `AGENTS.md` scripts row, `patches/README.md` block 17.

## r350 (2026-11-22, `WORKLOG.md` 261122.040) — a speculator ON the two-arm server: the drafter reads the hidden rows of the context that RAN the batch, a verify batch is a co-batched shape, and the pair's text oracle also needs the SAME idle-slot parking policy (block 17, `TODO` #364)

1. **A drafter's target side is per-CONTEXT, not per-model.**  Block 17 refused `--spec-type` because the pair
   evaluates generation on `ctx_dec` while every drafter read `params.ctx_tgt` (the context the speculator was
   constructed with).  The rows a drafter consumes (`llama_get_embeddings_nextn`, `..._ith`, `..._rows`, the
   eagle3/dflash layer inputs) exist only in the context that just ran the batch, so `common_batch` now carries
   its `ctx` and `common_speculative_process()` reads the rows from it; a second target context is registered with
   `common_speculative_add_target_ctx()`, which is also what enables the nextn output there.  A batch from an
   unregistered context is REFUSED with a warning (drafting degrades, the text cannot) rather than read stale.
   The rule: **a hidden-row read is only valid on the context that produced the last batch** — a second target
   context is not an optimisation, it is what makes the read well defined.
2. **The verify width is a DECODE-ARM property, and it must be declared twice.**  `ctx_dec` was sized
   `n_batch = n_ubatch = n_parallel` (a 1-row-per-slot decode).  A speculative step is a verify batch of
   `n_parallel × (n_draft + 1)` rows (3 slots at `n_max 3` = 12 rows), so the context is now sized for that and
   its output limits are raised to the whole width (every verify row is an output).  The slab reserve must follow
   in the SAME change: `common_arms_dec_reserve()` is the decode arm's own graph reserve, so a speculator on the
   pair is charged the **sum** of that and the draft context's reserve — before r350 the two were disjoint and
   the code deliberately kept the larger, which under-charges by a whole draft context now that both exist.
3. **The draft context's own `llama_process()` is a second DECODE-band context.**  Block 17's rule ("never two
   decode-band contexts: the per-table `GGML_MOE_DEVREMAP` map is process-wide") was written for two arms; on the
   pair with a drafter there are three contexts, and the prefill arm's prompt-ingest runs the draft's heads on the
   main thread *while the decode arm steps*.  That is now bracketed by the same `arms_quiet` handshake a request
   boundary uses (the decode arm enters no step until the ingest ends).  Measured cost: once per prompt ubatch,
   bounded by one draft ingest.
4. **A speculative slot's state does not cross the request boundary.**  The draft, the draft candidates, the
   target checkpoint and `spec_is_replay` all describe the batch the OTHER arm ran, so the forward handover and
   the reverse take both clear them (`spec_slot_reset()`), and the prefill arm's `post_decode()` accept loop skips
   any slot the decode arm owns (it would otherwise read a `spec_i_batch` that thread is writing).
5. **The pair-vs-single text claim is only an oracle under the SAME placement policy.**  Measured 2026-11-22
   (27B Q8_0, `-ctk q8_0 -ctv q8_0`, 3 devices `-sm tensor`, `-np 3 --kv-unified`, three cold requests in one
   process): the pair read `38169dcc3325` on the second request and the single-context server `c3bf1df7aa1d`.
   The mechanism was **not** the pair and not the speculator: `LLAMA_SERVER_PARK_IDLE_SLOTS` is ON in the single
   server, so it parked the finished slot's cells and placed the next sequence at cell 0, while the **co-owner
   guard declines parking on the pair** (`idle slot parking: not supported by this KV cache`) and the sequence
   was appended at cell 5309.  The `LLAMA_KV_DEBUG_CELLS` census names it exactly (`seq=1 ... runs=0-528` against
   `runs=5309-5837`), and the SAME single context with `LLAMA_SERVER_PARK_IDLE_SLOTS=0` reads `38169dcc3325`.
   The rule: **a pair-vs-single text pair must pin the parking policy** (or compare first-request-per-process
   only) — `GREEDY-PURITY.md` §38's placement artifact, reached through block 16/17's parking guard.
6. **What the pair still refuses, and why (each is a per-context binding, not a missing feature).**  The
   layer-input drafters (eagle3/dflash/dspark/dflash2/lilicorr) read the target's per-layer inputs through a
   device buffer bound to ONE context and to a warm-up decode; a gemma4 assistant shares the target's KV instead
   of keeping its own, and that cache already has two co-owners.  Both are refused at start-up with a message.

Record: `WORKLOG.md` 261122.040, `scripts/gate-arms-spec.sh`, `MANIFESTS.md` gate 39, `patches/README.md`
block 17, `ENVIRONMENT.md` §23, `GREEDY-PURITY.md` §107.

## r351 (2026-11-23, `WORKLOG.md` 261123.001) — a shared library LINKS with a symbol nobody defines, and which linker refuses differs: a guarded DEFINITION with an unguarded CALLER is a consumer-side break no build here sees (block 17 amendment 3, issues #353/#354)

1. **A `.so` may carry undefined symbols, so "the library target built" is not a link proof.**  r347's crash-#373
   work added `ggml_cuda_pool_peer_copy_ok()` to `ggml-cuda-vmm.h`, DEFINED it inside
   `#if defined(GGML_USE_VMM)` in `ggml-cuda.cu`, and called it unconditionally from `ggml_cuda_peer_copy_ok()`
   in `ggml-cuda-slab.cu` (a backend-shared file, so the call is in every configuration).  `GGML_USE_VMM` is off
   for `GGML_HIP_NO_VMM=ON` — the HIP **default** — so from r347 to r350 a stock `-DGGML_HIP=ON` build produced a
   `libggml-hip.so` that linked (a shared library is allowed to carry undefined symbols) and then failed every
   executable linking it: `ld.bfd: libggml-hip.so.1.0.0: undefined reference to ggml_cuda_pool_peer_copy_ok`
   (issue #353, gfx1103 / ROCm 7.16 / GCC 15.2; `test-backend-ops`, `test-sleep`, ... all of them).  The same
   break existed on the NVIDIA `GGML_CUDA_NO_VMM=ON` path.  The fix is the reporter's: a no-VMM definition that
   answers "fine" (no pool, so no pool range can hold the destination — the answer the slab already gives for a
   destination outside every range), defined unconditionally instead of inside the VMM arm.
2. **WHICH LINKER refuses is a property of the box, and ours does not refuse.**  GNU ld.bfd checks a shared
   library's undefined symbols by default; lld (what ROCm's clang drives here) does **not**.  Measured on the
   pre-fix tree, both arms of one no-VMM build: `nm -DC libggml-hip.so` reads `U ggml_cuda_pool_peer_copy_ok`,
   `clang -fuse-ld=bfd` on a `int main()` against it FAILS with the reporter's exact message, and the same link
   with the default (lld) **succeeds** — and `cmake --build` of the whole set returned 0 because the library
   target was simply relinked (the executables were up to date).  So a green ladder on this box is not evidence
   that a consumer's link works, and the class needs the loader's own report: `scripts/check-shlib-symbols.sh`
   runs `ldd -r` (linker-independent) over every shipped library and fails on any `undefined symbol:`.  After
   the fix every library in both builds (default VMM and no-VMM) resolves every symbol; the same check names the
   symbol on the pre-fix tree (the negative control).
3. **A guard that is ASYMMETRIC is a link break; a guard that is symmetric is invisible to any build.**  The
   caller here was unconditional, which is why one configuration's build (and `ldd -r`) sees it.  A caller and a
   definition both behind the same `#if` compile out together and no build of either configuration can notice:
   that class still owes a build of the OTHER configuration whenever a guard, a header-declared API or a
   configuration predicate changes (`MANIFESTS.md` gate 40).
4. **A "never expected" fallback that returns the function it is in is a stack overflow, not a fallback.**  Both
   of `ggml_cuda_host_stream()`'s fallbacks (an out-of-range device id, a failed `cudaStreamCreateWithFlags`)
   returned `ggml_cuda_host_stream()` — the pre-r347 behaviour they meant to keep is `cudaStreamPerThread`, and
   the comment even said so (issue #354).  They now return the sentinel directly, and the failed-create arm
   CACHES it (`streams[device] = cudaStreamPerThread`) because a failed stream create is not transient: retrying
   it on every host copy costs more than the stream it replaces.
5. **Both r351 changes are unreachable in a default run, which is why the numbers are identical.**  The no-VMM
   definition is not compiled in the delivery's own build (`GGML_HIP_NO_VMM=OFF`), and the fallback streams are
   only reached where the old code recursed.  Measured: prefill-logit **mean KLD 0.000000 / same-top-p 100.000 %**
   against the recorded 10.2 base, same-seed greedy text unchanged (`111 chars sha=7e6229b260f6` before and
   after), and the no-VMM build reads the same text as the default build — so the fix can be judged on its own
   terms (link + symbol) rather than against a moving number.

Record: `WORKLOG.md` 261123.001, `scripts/check-shlib-symbols.sh`, `MANIFESTS.md` gate 40, `patches/README.md`
block 17.

## r352 (2026-11-23, `WORKLOG.md` 261123.002) — a BLOCK can be withdrawn without moving a byte, and a page fault that will not reproduce is a report, not a fix

1. **A withdrawal is verifiable when the artifact reproduces the old tree exactly.**  Block 17 left
   `patches/` at r352.  The check is not "the block still builds somewhere": it is that applying the beta
   artifact to the new default tree produces the **previous release tree byte for byte** (measured:
   default tip `e695c46ce` / tree `0d2f70e6c`; `beta/prefill-decode-arms/prefill-decode-arms.patch`
   applied alone → tree `8851b0d02`, i.e. r351's `.tree`).  That proves the withdrawal removed exactly the
   block and nothing else, so a default checkout is the r351 tree minus block 17 and no other r may have
   to be re-verified for behaviour.  The block 17 → 0300 **order** changed (0300 used to sit on top of
   block 17 and now sits directly on block 16); both orders give the same tree, so the two patches are
   textually disjoint, which is what makes the re-cut a `git am` and not a port.
2. **The optional artifacts had to be rebased, and one was not a clean apply.**  `0100` (two-lane
   prefill) conflicts in `src/llama-context.cpp` wherever block 17's crash-#373 bookkeeping sat
   (`memory->apply_mutex()`, `meta_bind_view_slot()`): the conflict is *context*, not content, so the
   resolution is to keep 0100's own change and **drop the block-17 lines** — the resolution is verified by
   the same tree relation (old `0100` tree vs new = exactly block 17's 3682/367/30, one line less than
   block 17's own 3683 because an unresolved block-17 blank line sat inside a hunk).  `0101` re-applied
   with `git am -3` and no conflict and reconciles exactly (3683/367/30).  An optional artifact is cut
   against the current default tree, so this rebasing is part of every release, not an exception.
3. **A withdrawn block is a `GONE` finding, and one decision row closes it.**  `check-rung-file-sets.py`
   reports the block as `GONE` (30 files) and nothing else: the files, defines, env vars and options it
   carried read as *lower totals* without individual `LOSS` rows, because a row is only emitted for an
   item that no block of the new set carries.  The row is
   `v18-919794500-r350	0017	block	0017	<reason>` in `patches/FILE-SET-RETIREMENTS.tsv` — scoped to the
   transition it was decided in, per the file's contract.
4. **A page fault that will not reproduce does not get a speculative fix.**  The r350 cache-off host-tier
   fault (`WORKLOG.md` 261122.040 §3) was re-tested at r352 on seven arms — `-sm layer` at 1 GPU and 3
   GPUs, `-sm tensor` with `-ncmoe 8` and `48`, the recorded 3-slot server command on **both** the r351
   build and the block-17-free build, and the same shape with the pinned tier forced off two ways
   (`MOE_HOST_POOL_PIN=0`, `MOE_HOST_POOL_PIN_FAULT=late`) — and every one is **clean** (467-469 t/s
   prefill on the server arm, 11.2-11.6 s per cold 5.2k-token POST).  What the failing state must be is
   narrow: the default host-expert tier is AUTO/POOL, so the expert master is the **pageable** read-only
   model mapping (`ROCm_HostMap`, 62 400 MiB here) and the only device-readable copies of it are the
   bounded **pinned per-expert pool** (62 256 MiB, 144/144 sources registered) and the per-table pinned
   bounce ring; `expert_host_src()` *declines* (returns false) rather than handing out a pageable pointer
   when both are unavailable, and forcing the pinned tier off does not fault.  So a reproduction needs a
   path that bypasses `expert_host_src()` or a mid-run pinned refusal that some caller does not treat as a
   decline — neither is identified, and no gate was added: a gate anchored on a fault that does not fire
   is vacuous (`MANIFESTS.md` requires non-vacuity).  The finding stays open with its control arms
   recorded (`TODO` #374), and the arms are re-runnable from its own entry.
5. **The withdrawal does not change the tag arithmetic.**  `n_blocks` is the official count and the
   `v<N>` prefix is derived from it (`scripts/make-release.sh`), so removing the 18th official block moves
   the release from `v18-` to `v17-` on its own; `n_default` follows (`18`), and `validate-set.sh` /
   CI read the counts from `release.json` rather than carrying them.

## r353 (2026-11-23, `WORKLOG.md` 261123.004) — a fold's matcher can be dead on the models the delivery validates, and a dead matcher moves no hash: the SILU GDN output-norm fold (block 10, contributor PR #357)

1. **`ggml_cuda_gdn_norm_gate_match` accepted a SIGMOID gate only, so the fold never fired on qwen35 or
   qwen35moe.**  r102 (PR #200) built the fold for qwen4exp's GDN output gate, which is `ggml_sigmoid`; the
   qwen35 / qwen35moe / qwen3next / clover-moe `build_norm_gated()` emits `ggml_silu`, and the census proves
   it: `fused sigmoid=0 silu=0` on r352 for both families, i.e. two launches per GDN layer per decode step
   (48 layers on the 27B) ran for nothing on the delivery's own models.  The kernel
   (`unary_gated_q8_1_norm_kernel<op>`) and `ggml_cuda_op_unary_mul_q8_1`'s dispatch already took `op_silu`,
   and `ggml_cuda_can_fuse({UNARY, MUL})` already admitted SILU, so the missing dimension was purely the
   matcher's op predicate (`GREEDY-PURITY.md` §108).
2. **The widening is bit-identical, and the instrument is the exact-fallback KLD at every width the fold
   reaches** (`GREEDY-PURITY.md` §60 rule 2): the in-kernel norm reduces like `rms_norm_f32`'s strided path, so
   `gate-prefill-logits.sh --ab "GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0"` reads **−0.000000 / 100.000 %** at
   `--ub 1` (the decode graph, where the fusion fires 96 times), `--ub 4` (48) and the canonical prefill point,
   and the canonical point read against a base recorded from the r352 build on the same ROCm line is
   **0.000000 / 100.000 %**.  Greedy text is unchanged everywhere measured: 27B UD-Q4_K_XL `2d0d685256cf`,
   35B-A3B UD-Q4_K_M `abed0bf40d87` (the r324 anchor), 3-GPU `-sm tensor` `9051286a12ae`, and a two-slot
   `--kv-unified` server's two concurrent greedy answers.
3. **A dead matcher is a MECHANISM defect, so the gate is a census, and the OFF arm needs its own counter.**
   `GGML_CUDA_FUSE_GDN_NORM_GATE_CENSUS=1` logs `fused sigmoid=<n> silu=<n> declined silu=<n>` from the context
   destructor; on r353 the 27B reads `96` (`-ub 1`) / `528` (`-ub 8`) fused-SILU with `declined=0`, and the
   `GGML_CUDA_FUSE_GDN_NORM_GATE_SILU=0` arm reads `fused silu=0 declined silu=528`.  Both counters are taken
   **at the gate node** (the earlier revision counted the decline at the `RMS_NORM`, which cost a full match per
   norm in an A/B arm) and the declined one is computed only while the census is on.  The two counts are NOT
   required to be equal: a two-slot unified server measured `fused=384` against `declined=672`, because the arms
   capture different numbers of graphs -- non-zero on both is the invariant.
4. **The multi-sequence guard is unchanged and is what keeps this out of the r108 class.**  `xn->ne[3] != 1`
   refuses a fold on a multi-sequence ubatch, so no arm of the fold reaches the shape `GREEDY-PURITY.md` §49
   covers; `tests/test-fusion-gate-beta-multiseq` still reads 77/77, and a two-slot unified server answers
   identically with the switch on and off.
5. **The win is small and it is at depth, not at depth 0**: tg128 **+0.5 %** (27B UD-Q4_K_XL, d8192) and
   **+0.9 %** (35B-A3B UD-Q4_K_M, d8192) over two interleaved rounds on one R9700 (ROCm 10.2), depth 0 within the
   run-to-run spread (+0.1 % / +0.5 %, the 35B-A3B's first run after load being the low outlier), and
   `llama-batched-bench -npl 1,4,8` +1.1 / +0.1 / +0.05 % with pp512 flat.  The mechanism is two launches per GDN
   layer per step, so the win scales with a launch's cost, not with the model.

## r354-wip (2026-11-23, `WORKLOG.md` 261123.007) — the synchronous host-copy stream is the stream a graph capture OWNS, a refused peer descriptor poisons an all-descriptors grant, and a `.so` links with a symbol nobody defines (`prefill-decode-arms` v2 STEP 1)

*No delivery revision:* this is `wip/prefill-decode-arms-v2/` STEP 1 and nothing in `patches/` changed
(`release.json` untouched).  `r354-wip` is the campaign's handle for the entry, not a revision number; the
delivery revision that eventually carries the change, if it is promoted, is the maintainer's to number.

1. **`cudaStreamPerThread` is the legacy DEFAULT stream on HIP, and a graph capture claims it.**  Its value
   is `(hipStream_t) 2` — the sentinel, not an identity — and HIP resolves it to the current device's legacy
   default stream at issue time.  Two contexts therefore cannot both use it: while one thread has a capture
   in flight, the other's host copy is refused with `operation not permitted when stream is capturing`, and
   the device is then poisoned for **every** later launch (`operation failed due to a previous error during
   capture`).  Measured on the 3-device `-sm tensor` pair from `llm_graph_input_ple::set_input`.  One context
   never reaches it, because only the capturing context issues host copies and the capture path does not
   upload — which is why this can sit in the delivery for years and appear only when a second context
   exists.  Every synchronous host↔device path must issue on a **private per-(thread, device)** stream
   (`ggml_cuda_host_stream()`), and the class includes more than the obvious setters: the pinned bounce and
   2-D staging paths, `cpy_tensor`/`cpy_tensor_n` and the peer-staged copies, the KV-region row moves, the
   async-input ring **and** `batch-copy.cu`'s multi-tensor `set_tensor_n`/`get_tensor_n`.
2. **A helper that replaces the sentinel must resolve the device the way the sentinel did, and must fail
   without recursing.**  Resolve against the **current** device (a caller that has not selected its device
   was already copying to the wrong one, so nothing regresses); on a failed `cudaStreamCreateWithFlags`,
   return `cudaStreamPerThread` **directly**, cache it, and never call back into the helper — a recursive
   "fallback" is a stack overflow, not a fallback (issues #354).  A host-side `cudaDeviceSynchronize()` is
   the same defect class in the other direction: it is illegal while another thread captures, and it
   barrier-waits work the caller does not depend on, so a wrap that only needs "every origin's copies have
   landed" must synchronise each **origin's own** stream.
3. **A refused peer descriptor poisons an all-descriptors `cuMemSetAccess` call.**  The movable-boundary slab
   learned it for its own mappings (r92): grant the owner alone, then each peer **on its own**, keep the mask
   of what was actually granted, and let the copy path stage the pairs the driver refused.  The VMM **pool**
   is the second user of the same machinery and did none of that — it granted every descriptor in one
   `CU_CHECK`ed call, so one refusal aborted the process (`HipVMM Failure: invalid argument`).  Unreachable
   with one context (its pools are created and grown during load and the first reserve, before the arena
   fills); reachable the moment a second context's pool grows at request time with the arena full.  The
   owner's descriptor must be **located**, not assumed first: the list is in device order, so for device 1 or
   2 the front is a peer — and granting "the front" with `CU_CHECK` is exactly the abort being removed.
4. **A shared library is allowed to carry undefined symbols, so a guarded DEFINITION with an unguarded CALLER
   breaks the consumer, not the build** (issue #353, `r351` records the delivery-side version of this).  The
   caller here (`ggml-cuda-slab.cu`) is backend-shared, the definition was inside `#if defined(GGML_USE_VMM)`,
   so `libggml-hip.so` linked and every executable that linked *it* failed.  Which linker refuses differs —
   GNU ld.bfd checks a shared object's undefined symbols, lld does not — so `scripts/check-shlib-symbols.sh`
   (`ldd -r`, linker-independent) on **both** configurations is the instrument, and it belongs in any change
   that touches a conditional-compilation guard.
5. **A text-only oracle is vacuous for this class.**  "The greedy text did not change" is equally consistent
   with the helper having returned the sentinel it was meant to replace, because the sentinel *is* the old
   behaviour.  The non-vacuity check is structural: link the built library and assert the returned handle is
   a real stream (`0x558a…`), not `hipStreamPerThread` (`0x2`), not the default (`0x0`), stable across calls
   and distinct per device and per thread — `wip/prefill-decode-arms-v2/gates/probe-host-stream.cpp`.

## r355-wip (2026-11-23, `WORKLOG.md` 261123.008) — a context can BORROW another context's `llama_memory`, and a guard written against `cparams.*` tests what the constructor DECIDED, not what the caller ASKED (`prefill-decode-arms` v2 STEP 2)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 2, nothing in `patches/` changed.

1. **"Shares the KV" must be structural, not per-module.**  `llama_context_params::ctx_memory` gives a
   context the owner's `llama_memory_i` object **itself** (`llama_get_memory(borrower) ==
   llama_get_memory(owner)`), so every memory module — `llama_kv_cache`, `llama_memory_recurrent`,
   `llama_memory_hybrid`, `llama_kv_cache_iswa`, the qwen4exp indexer pool, and anything added later —
   shares by construction.  The alternative (a second module taught to *view* the first's cells) makes
   sharing a property each module must opt into, and a module that is not taught reads an EMPTY cache
   while the log says it is shared — fluent wrong text, no crash, no gate.  One object, one truth.
2. **The borrower owns nothing: `llama_context` keeps `memory_own` (unique_ptr) and `memory` (raw pointer to
   either its own or the borrowed object).**  Every graph builder and every `llama_memory_*` call reads
   `memory`, so there is one code path and no per-context shadow of the memory's bookkeeping.  Freeing the
   borrower must not release the memory, so teardown order is part of the API contract: **the owner
   outlives the borrower** (verified: the owner still decodes after the borrower is freed; a borrower that
   freed the memory fails there, one that freed it twice aborts).
3. **A guard on `cparams.<x>` tests the constructor's DECISION, not the caller's REQUEST.**  `cparams.ctx_other`
   is zeroed by the constructor and re-adopted only for the four architectures that need one
   (gemma4-assistant, eagle3, dflash, qwen4exp), so a "ctx_memory and ctx_other are mutually exclusive" guard
   written against `cparams.ctx_other` was **dead on every other model** — it would have refused the
   contradiction exactly where `ctx_other` is meaningful and silently allowed it everywhere else.  Guards on a
   request belong on `params.*`.  Caught by the probe, not by any text gate (both arms read the same text).
4. **A text oracle cannot see a dead borrow either.**  Two contexts on one memory and two contexts on two
   memories both produce identical greedy text for a single request — the difference only appears when the
   second context reads the first's cells.  The instrument is: identity of the two `llama_get_memory()`
   pointers, a **negative control** (two unborrowed contexts must hold *different* memories, or "the pointers
   are equal" proves nothing about the borrow), and a decode whose rows are written by one context and read
   by the other (`gates/probe-ctx-memory.cpp`).

## r356-wip (2026-11-23, `WORKLOG.md` 261123.009) — a prepared batch's cells look EMPTY in the cell map, a shared library's undefined symbol is a link break, and a `const` searcher that allocates (`prefill-decode-arms` v2 STEP 3)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 3, nothing in `patches/` changed.

1. **`prepare()` ROLLS THE CELL MAP BACK, so "the cell map says free" does not mean free.**  Between a
   `prepare()` and the decode that commits it (`apply_ubatch(..., trial)` places each ubatch, then `v_cells`
   and `v_heads` are restored), the cells a batch will write are **empty** to every other search.  In a
   single-context run that is invisible: the caller applies before it prepares again.  With two contexts it is
   the whole defect -- both searches are handed the same cells and both batches write them, which is fluent
   wrong output under load, not a crash and not a text change.  **The only mechanism that closes it is a
   reservation bitmap that outlives the rollback** (one per stream/cell, enabled only when the cache is
   shared), with three release paths: a real `apply_ubatch`, `prepare()`'s failure path, and the batch
   context's DESTRUCTOR (a cancel/abort gets no cooperation, so the destructor is the only place that cannot
   be forgotten).  A handed-out cell that is already reserved must be an `GGML_ASSERT` -- a log line would let
   silent corruption through.
2. **The reservation is invisible to every text oracle, and a two-thread soak does not exercise it either.**
   Two arms on one memory read the *same* text with and without the bitmap (measured: identical digests on
   both builds), because the collision needs two batches that are **prepared and not yet applied**, a state
   the public API does not expose.  The decisive instrument is a deterministic set-algebra probe on the cache
   itself: `prepare(A)` then `prepare(B)` must be **disjoint** -- and on the build without the bitmap it is
   `[0..511]` twice, exactly.  Prefer that shape over timing/luck.
3. **`llama_memory_i`'s lock must be a MEMBER of the object, not a `shared_ptr` handed around.**  One object
   has one lock; an `apply_mutex()` accessor means no caller can forget to share it and no per-context shadow
   can exist.  The discipline is part of the invariant: bookkeeping only (µs at a ubatch boundary), released
   before the graph build and the compute -- a measurable hold means the lock is in the wrong place
   (the previous attempt measured a 2.85 s starvation that way).
4. **A composite memory must FORWARD `notify_borrowed()` to every module it holds** (hybrid attn+recurrent,
   the indexer, iSWA base+swa, dsa/dsv4/msa).  A module that never hears it keeps `n_borrowed == 0` and
   silently stops reserving -- the same "shares the KV" trap in a new place, caught only by the N:1 gate.
   `llama_memory_hybrid_idx` must call its PARENT's forwarder, not `llama_memory_i::notify_borrowed()`, or the
   base's two caches are skipped.
5. **`gate-park-idle-slots.sh` needs `ROCBLAS_USE_HIPBLASLT=0` on its own shape, and the delivery's lottery
   check does not cover that shape.**  Unpinned the gate fails `park != lazyoff` on 3 of 4 requests, with the
   random-divergence signature (`park == parkoff` on some requests, not others); the pin makes it pass twice on
   the same build, `MOE_EXPERT_HEATMAP=load` does not.  `scripts/check-hipblaslt-lottery.sh` reads
   `NO-CHOICE` because it measures the 27B Q8_0 *default* call sites, while the gate drives the Flash-Next
   host-tier IQ4_NL shape -- §44's "stability is config-dependent", made concrete.  Reported (the fix is a
   delivery-tooling change), and a fresh-process comparison on that shape needs **one request in flight**:
   even pinned, two concurrent 2-slot arms differed across runs (147 vs 188 chars) because the interleaving
   picks the batch composition and the batch composition picks the kernels.

## r357-wip (2026-11-23, `WORKLOG.md` 261123.010) — a view slot keyed on a ROLE breaks at three contexts, a `static` selector is not a thread's, and 2 slots cannot express 3 contexts (`prefill-decode-arms` v2 STEP 4)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 4, nothing in `patches/` changed.

1. **The Meta backend's view slot must be an IDENTITY (per context), never a ROLE.**  The backend keeps a
   rotating PAIR of view containers per (buffer, slot) and a graph must be built *and* resolved under the same
   slot.  Posting `alt ? 1 : 0` encodes "the second scheduler of `ctx_other`", and a pair plus a drafter is
   **three contexts and two roles** -- so the decode context and the draft context shared a pair.  It works
   today only because they are used sequentially on one thread; the container is per **buffer** (shared by
   every thread), so sequential use is luck, not a contract.
2. **A selector that picks state inside a BUFFER must be `thread_local`.**  `g_meta_view_slot` was a plain
   `static int`: with per-context slots and two threads building graphs, both threads write the same selector
   and one context resolves another's views.  The thread that builds a graph is the thread that resolves it.
3. **2 slots cannot express 3 contexts.**  `n_view_slots` is 8 (four pairs), a pair is created on its first
   use (an unused slot costs nothing), and both the setter and the container accessor assert the range.  A
   fifth concurrent context must WARN that it wraps -- sharing a slot silently is the defect returning.
4. **A reserve builds a graph too.**  Binding the slot only in the alt-scheduler switch leaves a reserve with
   no alt-state change under whatever slot the thread last posted, so it is bound at the top of
   `sched_reserve()` as well as in `sched_alt_select()`.
5. **The versioned split-state cache is not a view slot and needs no per-slot index.**  It survives graph
   alternation by VERSION (8, keyed on the tensor bytes the state was computed from), which is what upstream's
   SIGSEGV fix introduced; indexing it per slot would be a second mechanism for one job.
6. **A per-context diagnostic line at INFO changes every single-context startup log.**  The view-slot base is
   logged at DEBUG, and the pair's census goes on the BORROW line (which exists only when it matters), so the
   default path's startup shape stays byte-identical -- which is what lets that shape be an anchor at all.

## r358-wip (2026-11-23, `WORKLOG.md` 261123.011) -- a recurrent module's per-batch placement is MODULE state that a graph reads, and a getter that RESETS an index is a race with a second graph (`prefill-decode-arms` v2 STEP 5)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 5, nothing in `patches/` changed.

1. **`llama_memory_recurrent` kept this batch's graph-visible state on the MODULE** (`head`, `n`, `rs_z`), so
   two contexts building graphs read each other's placement.  It now returns a `placement` from
   `find_slot()` and the CONTEXT stores it.  `head` deliberately stays on the module: it is also the SCAN HINT
   (where the next search starts), it is genuinely module state, and keeping its **value** -- not just its
   meaning -- is what makes the single-context path bit-identical (A5's acceptance counts moved by zero).
2. **A graph-time getter that mutates is a race, not a style problem.**  `s_copy()` reset `rs_idx[seq]` as a
   side effect, so the value depended on HOW MANY TIMES the input had been set for one apply.  The reset moved
   to `apply()` (under the memory lock) and the per-cell rollback index is staged there; the getter only reads.
   The intended semantics is the staged one (the index belongs to the cell; the reset exists so the *next*
   apply starts fresh), and it is what a second live graph needs.
3. **A recurrent cell holds a whole SEQUENCE's state, not a token.**  A 32-token batch for one sequence
   occupies ONE cell (`n = 1`), and the 27B's recurrent store is `n_seq_max` cells wide (4).  Any oracle that
   expects `n == n_tokens` on this module is wrong; the distinguishing quantity is **which cell**, and whose
   sequence lives in it.
4. **The non-vacuity instrument for a placement extraction is two LIVE contexts read at one instant** -- prepare
   two batch contexts against the shared memory, apply both, keep both alive, read `(head, n, rs_z)` from each.
   Extracted: two different placements.  Not extracted: both read the module's one value (measured: the STEP 4
   build prints `head=1 n=1 rs_z=1` for BOTH).
5. **The rollback plane is the sensitive instrument for a recurrent change, and its acceptance COUNTS are the
   strictest form.**  The four-axis draft-acceptance counts (`#gen drafts`/`#acc drafts`/`#gen tokens`/
   `#acc tokens`) reproduced bit for bit through this refactor; a byte-identical text with different counts
   would have meant the rollback indices moved.

## r359-wip (2026-11-23, `WORKLOG.md` 261123.012) -- `LD_LIBRARY_PATH` beat the build's `RUNPATH`, so a 10.2 build ran on 10.1's hipBLASLt and every cross-process text comparison became a coin flip (tooling)

*No delivery revision:* `scripts/` + the workflows; nothing in `patches/` changed.

1. **`LD_LIBRARY_PATH` is searched BEFORE a binary's `DT_RUNPATH`.**  A gate that defaulted it to another ROCm
   install therefore ran the build against that install's runtime, silently -- every soname matches.  Measured
   2026-11-23: the delivery's gates defaulted to the retired `/opt/rocm-10.1.0-gfx120X/lib` while
   `~/bin/build-llama-rocm` links **10.2.0**, so `libamdhip64`/`librocblas`/`libhipblaslt` all came from 10.1.
   The rule is now a helper, `scripts/rocmlib.sh`: resolve the lib dir from the **build's own RUNPATH** (with
   `$ROCM_LIB` as the explicit override), never from a hardcoded tree.
2. **The symptom was per-PROCESS text non-reproducibility, and it looked exactly like a placement bug.**  With
   the 10.1 runtime the same gate arm (`park`, identical args) read three different texts across four fresh
   runs, so a gate that compares three arms across three processes failed whenever two of them drew different
   classes -- 1-2 failures in every 2-4 runs, with the pattern moving.  The tell is that the **same arm**
   disagreed with itself: a placement defect would make `park == parkoff` everywhere, consistently.
3. **That is the §44 mechanism, and it is a property of the LIBRARY, not of the pin.**  10.1.0 ships hipBLASLt
   **1.4** (`libhipblaslt.so.1.4`, 9.2 MB, the pre-`c50ea60130f` per-shape candidate container); 10.2.0 ships
   **1.5** (3.4 MB).  Under 1.4 each fresh process picks its own solution; `ROCBLAS_USE_HIPBLASLT=0` hid it by
   bypassing the library, and pointing `LD_LIBRARY_PATH` at the build's own tree removes it **with hipBLASLt
   enabled** (2 runs x 3 arms byte-identical, 3 runs pinned done previously).  A delivery gate must therefore
   never pin the library as a *substitute* for pinning the environment.
4. **A gate that compares text must state WHICH per-process variables are pinned.**  Two are documented and both
   matter on a host-tier MoE shape: the hipBLASLt runtime (§44) and the expert-cache residency prior
   (`MOE_EXPERT_HEATMAP=load`, §89 -- residency selects the MoE kernel set).  `gate-park-idle-slots.sh` now
   derives its library path from the build and needs **no** `ROCBLAS` pin; the resid­ency prior remains the
   caller's job where a gate's model has a writable `.moe-heatmap` sidecar.
5. **A residual, real concurrency effect survives the toolchain fix.**  On the Flash-Next host-tier shape with
   **two requests in flight**, one slot's greedy answer varied across three fresh 10.2 runs (`b2e646545bd9` once,
   `c60b5a919c9b` twice) while the other was stable throughout -- the interleaving picks the batch composition
   and the composition picks the kernels (DESIGN V8).  A **byte** text claim on such a shape belongs at one
   request in flight; concurrent arms carry the non-text claims.
6. **A gate's mechanism check must assert the CURRENT rule.**  `gate-draft-ctx.sh` grepped for block 17's
   `is a context of ANOTHER model (a draft head)` decline message, deleted at the r352 withdrawal -- and the
   delivery's rule for a cross-model head is adopt-as-a-draft with its **own** cache, so the absence of a
   decline line asserted nothing.  The faithful check is the positive fact and arch-agnostic: **>= 2 distinct
   KV-cache geometries at load** (`(N cells, L layers)`), since a draft that viewed or aliased the target's
   memory would print one.

## r360-wip (2026-11-23, `WORKLOG.md` 261123.013) — a policy flag is not a state flag, one predicate must serve the reserve AND the creation, and a log normalizer that drops a line form hides exactly what it should catch (`prefill-decode-arms` v2 STEP 6a/6c)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 6a/6c, nothing in `patches/` changed.

1. **`mmproj_use_gpu` is a POLICY flag (default ON) and says nothing about whether a projector is loaded.**  A
   pair-refusal predicate that tested it refused **every** shape for "multimodal input".  Only
   `params.mmproj.path` says a projector exists.  A predicate that gates a *capability* must test the state,
   never the policy that would apply if the state were present.
2. **The reserve derivation and the context creation must call ONE predicate.**  The previous attempt computed
   an arena reserve from `params` alone and then consulted a separate refusal list when creating the second
   context, so a refused pair still left the arena several GiB smaller (V2).  `common_arms_pair_possible()` is
   pure and returns the REASON, so "why not" is loggable from either caller and the two cannot drift.
3. **A landing that creates the second context but routes nothing to it is worth its own oracle.**  With the
   decode arm created and idle, the pair's text is byte-identical to the single-context arm on 4B/1-GPU and on
   27B/3-GPU `-sm tensor`, so the *routing* landing (6d/6e/6f) starts from a proven-inert base and any text
   change it introduces is attributable to the routing.
4. **A log normalizer that keeps only one line FORM silently drops the rest.**  `norm-log.sh` matched
   `<level> <component> <function>:` and therefore dropped every single-tag library line (`llama_context:`,
   `llama_kv_cache:`, `resolve_fused_ops:`) -- i.e. exactly the lines that carry a cache geometry, a memory
   identity or a fusion decision.  An anchor built on such a filter cannot see the class of change it exists
   for; keep every form and mask the volatile parts (addresses -> `PTR`, decimals -> `N`).
5. **`mmproj`, LoRA, backend sampling and `--sleep-idle` are per-context state machines, not shapes a second
   context can share.**  They belong on the pair's refusal list *by construction*, not because a test failed:
   each of them assumes it is the only user of its context's sampler, adapters or load/unload state.

## r361-wip (2026-11-23, `WORKLOG.md` 261123.014) — a round implemented twice is a round that drifts; the shared step must take the CONTEXT that ran the batch (`prefill-decode-arms` v2 STEP 6d-1)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 6d-1, nothing in `patches/` changed.

1. **The sampling/emit/stop block of a server round is the part two arms must share; the batch composition is
   the part that may differ.**  `post_decode()`'s inline block became `slot_step(slot, ctx, tok_idx)`, and its
   `ctx` parameter is the load-bearing one: it is the context that RAN the batch, so `common_sampler_sample()`
   reads the right output buffer and `common_token_to_piece()` the right vocab without a branch.  A duplicated
   emit block would drift on the first change to either arm.
2. **A step that may finish a slot needs a return value the two callers can each act on.**  One caller is an
   `iterate()` lambda (it must `return` from that slot), the other a loop body (it must stop stepping that
   slot and take the next); `bool slot_step(...)` (false = finished and released here) is the smallest shape
   that serves both without a callback.
3. **A speculative verify round is NOT the same step and must not be folded in.**  It emits `n_accepted` tokens
   with per-position probabilities taken from the verify batch's rows, so it keeps its own loop; sharing
   `slot_step()` there would mean passing a row list and a per-row prob mode -- a second implementation wearing
   the first one's name.
4. **An extraction is only worth landing if it is provably inert.**  Same-slot text identity (2 slots, pair OFF
   and ON), the same hashes on a second shape (27B, 3 GPU tensor), a byte-identical startup shape and the
   delivery's slot-lifecycle gate (park) all had to hold *before* the routing could be written on top.

## r362-wip (2026-11-23, `WORKLOG.md` 261123.015) — a handover with no arm to receive it spins the server at 94 % CPU, `i_batch` belongs to the arm that batched the slot, and the pair's win is a LATENCY win (`prefill-decode-arms` v2 STEP 6d-2/6e/6f)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 6d-2/6e/6f, nothing in `patches/` changed.

1. **A handover that runs when the receiving arm does not exist takes the slot out of the only batch that
   would run it.**  `post_decode` handed every freshly generating slot to `ctx_dec` even when the pair was OFF
   (`ctx_dec == nullptr`), so the slot left the prefill arm's batch, nothing decoded it, and the main loop spun
   at **94 % CPU** with no output — a hang, not a crash, and invisible to any text oracle.  Every "move a slot
   between owners" call must be gated on the owner existing, and the pair-OFF path must be byte-for-byte the
   old one.
2. **`server_slot::i_batch` belongs to the arm that BATCHED the slot, so the other arm must not read it.**  It
   is a row index into *that arm's* batch; read from the other arm it samples the wrong row of that context, or
   a row with no logits (`GGML_ASSERT(logits != nullptr)` in `common/sampler`).  The ownership flag is therefore
   not bookkeeping, it is a memory-safety property of the batching code: every iterate that touches a slot
   checks the arm first.
3. **Diagnosing a server hang: bisect the campaign commits, then `gdb` the live process.**  The repro (a
   5-token prompt on `-np 2 -c 8192 --kv-unified`) failed on the working tree and passed on every committed
   landings back to the base, which localised the change in minutes; `thread 1 bt` then named `pre_decode`'s
   lambda and 94 % CPU distinguished a spin from a deadlock.
4. **The pair's win is LATENCY, and it needs the wide chunk it exists for.**  On 4B/1 GPU with a 5246-token
   prefill: `-ub 2048` → the decoder's ITL **20.27 → 15.68 ms/token (−23 %)** and the prefill itself **+25 %**
   (4935 → 6147 t/s, the prefill no longer pays for the decoder's rows), with byte-identical output; at the
   delivery's default `-ub 512` the same measurement is **flat** (13.25 vs 13.16) because on this model the
   decoder never waits for a whole chunk.  A gate must report the ubatch with the number.
5. **A concurrent multi-slot arm cannot carry a byte text claim.**  On the 27B 3-GPU tensor shape the pair
   matches the single-context server **exactly** at one request in flight, and with two in flight one slot's
   answer varies run to run — on the unmodified delivery too (the interleaving picks the batch composition,
   the composition picks the kernels, V8).  Byte claims at one request in flight; concurrency carries the
   non-text claims.

## r363-wip (2026-11-23, `WORKLOG.md` 261123.016) -- the pair FAULTS a host-expert shape under load, a census in a destructor must go to stderr, and a refused pair must reserve NOTHING (`prefill-decode-arms` v2 STEP 6e)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 6e, nothing in `patches/` changed.

1. **(CORRECTED at r364-wip: the mechanism is a slab work-base alias, not the remap -- see r364-wip item 1 and
   `WORKLOG.md` 261123.017.  The refusal this item introduced has been removed.)**  **The two arms may NOT
   overlap when the prefill arm's step remaps process-wide device state.**  A 4-slot soak
   of the pair on 35B-A3B Q8_0 `-ncmoe 40` (host expert tier) faulted the GPU
   (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in `rms_norm_mul_rope_f32_qk_pair`) and aborted the server: the
   prefill arm's step remaps the MoE expert tables while the decode arm reads them.  The barrier deliberately
   does not serialise the two arms' `llama_process` calls (that overlap is the feature), so on a host-expert
   shape the overlap is UNSAFE as built.  Until it is root-caused, a host-expert shape (`-ncmoe > 0`) is
   REFUSED by `common_arms_pair_possible()`, and `gates/gate-arms-multislot.sh` reproduces the fault.
2. **A census printed from a destructor must go to STDERR.**  `~llama_kv_cache`'s reservation census at INFO was
   silently dropped at teardown (the log callback is already gone -- the delivery's own rule for destructor
   diagnostics).  A gate that asserts a teardown census will read "absent" and cannot tell that from a real
   failure.
3. **Never SIGKILL a server whose teardown you are about to measure -- and never wrap it in an inner `timeout`.**
   `timeout` forwards the signal it receives to its child, so a TERM to the process group reaches the server
   TWICE and its "second interrupt" path exits immediately without running a single destructor.  One signal,
   then wait for the process to exit; the outer `timeout` is the bound.
4. **A reserve must be charged only when the thing it reserves for will exist.**  `COMMON_ARMS_RESERVE_MIB`
   (256, measured 157-235 MiB/device) is charged only when the pair is possible: the np=1 arm prints "reserves
   NOTHING".  V2's defect was exactly the opposite, and it is now unrepresentable.
5. **A concurrent multi-request gate cannot carry a byte text claim, and its gate must say so.**  `gate-arms-
   multislot.sh` asserts survival, completion, the ownership census and the widest batch; the byte claims live in
   `gate-arms-n1-memory.sh` (one request in flight) and `gate-arms-itl.sh` (a fixed pair of requests).  A gate
   that quietly compared hashes across a concurrent arm would fail for reasons unrelated to its subject.

## r364-wip (2026-11-23, `WORKLOG.md` 261123.017) -- two live compute buffers must never share a slab work base, and a per-context routing flag must be per-thread (`prefill-decode-arms` v2 STEP 7a)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7a; nothing in `patches/` changed.  This corrects
r363-wip item 1, whose diagnosis was wrong.

1. **Two contexts' COMPUTE buffers must never be given the same slab work base.**  `ggml_cuda_slab_work_alloc()`
   has one "narrow" (decode/verify) base at offset 0 and assumed ONE live narrow view per device.  A second
   `llama_context` keeps its own persistent compute buffer, so the pair's decode arm landed on the prefill arm's
   1280 MiB view; one arm's prefill overwrote the other's activations and the K rope wrote KV rows at a garbage
   index.  Measured (35B-A3B Q8_0 `-ncmoe 40`, np 4, `GGML_CUDA_WORK_ALLOC_DEBUG=1`): `WORK_ALLOC ... narrow=1
   ret_off=0.0 MiB` twice, 3 ms before `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in
   `rms_norm_mul_rope_f32_qk_pair`.  The delivery already fixed this class once for the MTP draft (the
   `narrow2` top region; its comment: "without this they both aliased at base 0"); the rule is now:
   **one pinned auxiliary region per extra context, stacked downward from the top of the VA, with the arena
   ceiling at the lowest pinned offset** (`DESIGN.md` I9).
2. **A "which context is allocating" flag must be per-thread, not process-wide.**  `g_slab_compute_narrow2`
   was a global set around a reserve; with two arms reserving on two threads the prefill arm inherited the
   decode arm's flag, was routed into the aux region, found a live view, fell back to `cudaMalloc` and failed
   the graph reserve (1280 MiB OOM) 1.3 s into the soak.  It is now a `thread_local` slot derived from the
   allocating context's own identity (`ctx_other` -> draft, `memory_borrowed` -> borrowing context).  Same
   pattern as r355-wip's per-context Meta view slot, and it removes the same hazard from the MTP path.
3. **An aux region's pinned offset must be passed to its commit, not recomputed.**  The pin releases
   `g_slab_mutex` to evict arena tables (the eviction re-enters the slab); recomputing the offset after
   re-taking the lock can land on a different ceiling if another context pinned in that window, so the commit
   validates the passed window against the other slots and declines instead of overlapping.
4. **The fault was NOT the MoE expert remap.**  Rung 1 of the ladder (both Q/K deferral fusions off,
   `GGML_CUDA_ROPE_QK_PAIR=0 GGML_CUDA_QK_PAIR_FWHT=0`) still faulted -- in the *unfused*
   `rms_norm_mul_rope_f32<256,false,__half,0,0,false>` -- and the arena-eviction lines that suggested the remap
   are the boundary moves of the second view.  Note also that `GGML_CUDA_QK_PAIR_FWHT=0` alone cannot disable
   the `rms_norm_mul_rope_f32_qk_pair<...,0,0>` instantiation: `fwht_n=0` is the PLAIN pair, whose switch is
   `GGML_CUDA_ROPE_QK_PAIR`.
5. **A "grew past the reserve expectation" WARN at teardown is not a new finding.**  `~llama_context` prints
   `compute buffer size ... does not match expectation` on the 4B `--concurrent` shape; it is present on the
   pre-change build too (verified by stashing the change and re-running the A6 arm), so it must not be read as
   a regression of this step.

## r365-wip (2026-11-23, `WORKLOG.md` 261123.018) -- the VRAM outside the slab is the slab's *growth fuel*, not spare capacity, so an auxiliary compute buffer must come out of the arena (`prefill-decode-arms` v2 STEP 7b)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7b -- a measured NEGATIVE result; 7a's pinned region stands.

1. **`GGML_CUDA_SLAB_HEADROOM_MIB` is load-bearing, and it is not interchangeable with arena.**  It is what
   `slab_extend` maps the arena's physical FROM.  Spending it on anything fixed starves arena growth.  Measured
   (35B-A3B Q8_0 `-ncmoe 40`, np 4, pair ON, headroom 1536): an aux compute buffer of 1280 MiB taken outside the
   slab leaves ~256 MiB, the arena cannot grow, `alloc_table_locked: arena alloc failed` fires 42 times, the cache
   disables those tables, and the run aborts inside a graph capture (`ROCm error: operation failed due to a
   previous error during capture`).  The same 1280 MiB taken from the ARENA (a pinned aux region) passes with one
   transient arena failure.  So: an auxiliary context's compute buffer is charged to the arena, deliberately.
2. **`GGML_CUDA_SLAB_NARROW2=0` is not a working fallback on a slab-armed host-tier shape.**  It places the aux
   buffer outside the slab -- the same wrong budget as item 1 -- and the control arm aborts identically with no
   7b code in the tree.  It remains an A/B switch for the *draft* path on shapes where the cushion is not the
   binding constraint, not a "quick fix".
3. **A deferred owner reserve does not help.**  The slab is created inside `llama_context`'s constructor
   (`sched_reserve()`, `src/llama-context.cpp`), so "allocate the aux buffer before the slab exists" means
   postponing that reserve; the aux buffer would then be allocated from the same outside VRAM (the slab has not
   taken it yet only if the slab is also smaller), i.e. the same budget as item 1.  The only variant that keeps
   the arena intact is a bigger *reservation* (`GGML_CUDA_SLAB_RESERVE_MIB`), which shrinks the slab at creation --
   the same arena loss.

## r366-wip (2026-11-23, `WORKLOG.md` 261123.019) -- the prefill -> decode compute-buffer drop is a process-wide mutation and races a second context's graph capture; a host-tier soak must pin the expert-residency prior (`prefill-decode-arms` v2 STEP 7d)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7d.

1. **A compute-buffer drop must not run while another context is inside `llama_process()`.**  The drop releases
   the wide-prefill layout: it moves the movable-boundary slab's work boundary and re-arms stood-down MoE arena
   tables.  With a second context capturing its graph on another thread, the capture fails:
   `ROCm error: operation failed due to a previous error during capture` at the next kernel launch (and then
   `operation would make the legacy stream depend on a capturing blocking stream` in the async-input ring, which
   is a consequence, not the cause).  It is INTERMITTENT: measured on the two-arm server, 35B-A3B Q8_0
   `-ncmoe 40` np 4 -- PASS 3/3, then FAIL 3/3, then FAIL 1/PASS 1, and **PASS 3/3 with
   `LLAMA_DROP_COMPUTE_BUFFERS=0`**.  A text/hash oracle cannot see it: the run either aborts or is correct.
   Exposure removed for the pair by `llama_context_disable_compute_buffer_drop()` (both arms); the measured cost
   is nil (arena 25537 MiB and expert hit-rate 1.0 identical to the drop-on arm).  The proper answer for any
   future second-context design is a barrier around the drop, not a permanent disable.
2. **`MOE_EXPERT_HEATMAP` DEFAULTS TO WRITING the persisted map**, so two host-expert runs in a row start from
   different residency priors -- and residency selects the MoE kernel set (r89).  That flipped a gate's outcome
   between two sessions of the SAME binary (the map was rewritten at 17:06:20 by the session's own soaks).  A
   host-expert gate must pin `MOE_EXPERT_HEATMAP=load`.
3. **A "PASS 3/3" is not a stable-rate claim when the arm is intermittent.**  Re-measuring the same binary after
   the environment drifted (item 2) reversed the verdict; a soak gate's record should state the repeats and the
   prior it pinned, or the green line is a lottery ticket.

## r367-wip (2026-11-23, `WORKLOG.md` 261123.020) -- an auxiliary compute region must be charged to the slab RESERVE up front, and the slab's `mapped` is not the arena's extent (`prefill-decode-arms` v2 STEP 7c)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7c.

1. **Charge an auxiliary context's compute region to the slab reserve, not to the arena's headroom.**  With
   `reserve = headroom + aux` the region is mapped from the **unmapped VA spare** and no arena table is evicted
   at load (measured: 1 -> 0 `arena alloc failed` lines on 35B-A3B `-ncmoe 40` np 4, the region at
   `[26.31, 27.56] GiB` labelled `unmapped VA spare`).  Taking it from the headroom instead (STEP 7b) starves
   `slab_extend` and aborts (r365-wip); taking it from the arena (unreserved, STEP 7a) costs one eviction pass.
   The figure is shape-dependent (1280 MiB on that shape, 512 MiB per auxiliary context on Gemma4 26B-A4B + MTP),
   so the slab logs it once per slot with the reserve that would host it -- the operator sets that on the next run.
2. **`ggml_cuda_slab::mapped` is the slab's VMM extent, not the arena's.**  An aux region pinned ABOVE the arena
   grows `mapped`, so a second pin's eviction test ("is the arena mapped over my window?") reads the slab and
   evicts tables that are not there (measured: Gemma4, both regions 512 MiB, slot 1 reported `1024 MiB evicted`
   with the arena's mapped top exactly at slot 1's base).  Track the arena's extent separately (`mapped_arena`:
   seeded at init, grown by `slab_extend`, never moved by a pin) and read it in `arena_top()` and in the pin.

\n
## r368-wip (2026-11-23, `WORKLOG.md` 261123.021) — a `thread_local` slot that selects state inside a SHARED buffer must be bound on every path that builds or resolves a graph, not only when a state variable changes (`prefill-decode-arms` v2 STEP 7e)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7e (a defect in this campaign's STEP 4).

1. **A "the requested state is already the current state" early return is not a statement about a
   `thread_local` selection.**  STEP 4 made the Meta backend's view slot a per-context identity and bound it in
   `sched_reserve()` and after the early return in `sched_alt_select()`; the main scheduler's state is the
   default, so the ordinary `llama_process()` path returned before binding and the slot stayed whatever the
   **last context on that thread** had bound.  Two contexts whose graphs share one Meta buffer then build,
   allocate AND resolve under each other's view containers, and the view lookup yields a null tensor:
   `GGML_ASSERT(tensor)` in `ggml_backend_tensor_set_async` (measured: Flash-Next GSQ-RCO IQ3_S + the external
   shared MTP head, 3 devices `-sm tensor -mg 0`; the delivery build reads 86 chars `be69e274b66d`, the
   campaign's STEP 4 build aborts rc=134, and binding before the early return plus at the top of
   `graph_compute()` restores the delivery's text).  Bind where the graph is built/resolved, not where a flag
   changes -- the cost is a thread-local store.
2. **A gate's shape can be the reason a defect survives it.**  `scripts/gate-draft-ctx.sh` owns the external
   draft head and passed on the broken build, because its arms are 1 device `-sm layer` (no Meta backend in the
   path) while the arm it documents as negative-tested at r348 -- 3 devices `-sm tensor` -- is the one that
   broke.  A gate that validates a *feature* on the cheapest shape it can run does not validate the *splits*
   the feature is used on; the check belongs on a shape where the shared object is actually shared.

## r369-wip (2026-11-23, `WORKLOG.md` 261123.022) — a gate's non-vacuity line is what catches a silent fallback, and an anchor's PROVENANCE decides what it can prove (`prefill-decode-arms` v2 STEP 8)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 8.

1. **A "plain == drafter" text comparison is vacuous when the drafter never initialised, so a gate must assert
   the MECHANISM (`#gen drafts > 0` AND `#acc drafts > 0`) and name the initialisation error.**  Measured here on
   a HARNESS bug: the Flash-Next adaptive arms were first run at `--spec-draft-n-max 2` (copied from
   `scripts/gate-draft-ctx.sh`, whose `draft-mtp` arm is not adaptive).  The adaptive controller's
   `n_min_adaptive` default is **3** and landed WITH its validation in delivery **block 02**
   (`git log -S n_min_adaptive`), so max 2 is a self-contradictory pair that has never been a working
   configuration: the library says so (`E failed to initialize speculative decoding context: invalid adaptive
   draft range: n_min_adaptive=3, n_max=2`) and continues as a PLAIN run with 0 drafts -- on the campaign build
   and on a build of the delivery tip `089f90067` alike (checked with the identical invocation).  So this was
   the gate's bug, not the feature's: the SAME arms at `--nmax 7` draft and accept on both models (73..213
   drafts per axis).  **No delivery change is proposed** (the maintainer's decision, 2026-11-23): an invalid
   flag pair is rejected loudly enough, and `draft-mtp-adaptive` is a shipping feature this campaign must not
   constrain.  A harness that drives it must use a valid ceiling.
2. **An inertness anchor is only evidence if it predates the change, so the gate must print each anchor's
   provenance.**  A1/A2/A3 and the 4B server startup shape were recorded at STEP 0 on the r353 delivery tree;
   phi-4's row and the other three startup shapes were recorded at STEP 8 and are therefore a regression LOCK,
   not evidence of inertness.  Recording an anchor after the fact and calling the gate green would launder the
   change it is supposed to catch -- so `--record` refuses to overwrite an existing row and every row prints its
   provenance.
3. **A gate must not hold the GPU flock across a harness that takes the same flock.**  `flock` is not reentrant
   across processes: the gate's whole-run lock plus `mtp-axis.sh`'s own lock deadlocked silently (the box idle
   for 5+ minutes with `flock -w 900 9` waiting on the gate's own fd).  Hence `mtp-axis.sh --no-lock` for a
   caller that already holds it.

## r370-wip (2026-11-23, `WORKLOG.md` 261123.023) — a context that BORROWS a memory must inherit the memory's KV types: `llama_cparams::type_k`/`type_v` have no default initializer, so an unassigned one is stack garbage (`prefill-decode-arms` v2 STEP 7f)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7f (a defect in this campaign's STEP 2).

1. **`llama_cparams::type_k`/`type_v` are plain `ggml_type` fields with no default initializer, and they were
   assigned ONLY in the context's own-memory branch.**  A borrowing context therefore carried uninitialized
   stack, and the graph builders that pick between quantized-aware attention arms read it: qwen4exp's
   `qwen4exp_qsa_sparse()` hands `cparams.type_k` to `qsa_op_supported()`, which builds a probe tensor of that
   type, so the value reached `ggml_new_tensor_4d` and aborted at
   `GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT)` (measured: Flash-Next + the pair on 3 devices
   `-sm tensor`, 2 of 3 runs; the instrumented values differed per run -- 21900, 22095 -- which is why it was
   intermittent, and it is invisible on models whose builders never read the field, i.e. 27B and 35B-A3B).
   **Rule:** a field that selects a code path and has no default initializer must be assigned on EVERY branch of
   the constructor, or the branch that skips it trades a compile-time default for run-time garbage.  A borrower's
   KV types are a property of the memory it borrows: mirror the OWNER's, not the requester's tag (verified with
   `-ctk q8_0 -ctv q8_0`: both arms read the same text).
2. **A defect that only bites when a *second* consumer exists is invisible to every single-context gate, and a
   shape matrix is the instrument that finds it.**  STEP 8's five gates all passed on this build, including a
   3-device `-sm tensor` four-axis arm, because the field is only read by a builder the *borrower* reaches
   differently (a probe at `n_ubatch = 3` vs 512).  STEP 9's per-shape matrix read the same binary 3x per arm and
   caught it at 2/3.

## r371-wip (2026-11-23, `WORKLOG.md` 261123.023) — a byte claim inside the ITL gate is a lottery, and one run of a throughput harness is a measurement, not a rate (`prefill-decode-arms` v2 STEP 9)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 9.

1. **The one gate whose point is two requests in flight cannot also make a byte claim.**  `gate-arms-itl.sh`
   asserts a decoder's ITL through a 5k-token prefill -- two requests, deliberately.  In that configuration the
   concurrent arm's batch composition (and so the kernel set) depends on arrival timing, so the answers are
   timing-dependent even on the UNMODIFIED single-context server (`r362-wip` item 5).  Measured: one run read
   `36216e77db27` (off) vs `68584f8dcc40` (on); 3 repeats per arm then read `36216e77db27` **6/6**; and at
   `-ub 4096` the *OFF* arm itself read `3c54ba65350c` where the 512/2048 arms read `36216e77db27`.  A hard text
   check there reports a near-tie as a defect.  Keep the byte claim where it can be made (one request in flight)
   and print the hashes as INFO.
2. **A single run of a throughput harness is a measurement, not a rate -- and it can be wrong in both
   directions.**  One ITL run read `-ub 2048` as 1028.8 ms (off) -> 961.9 ms (on), i.e. −6.5 %; the 3-repeat set
   puts the same arm at 1300-1310 ms -> 964-1088 ms, i.e. −22.8 %.  The pair's own number was stable; the
   baseline was the outlier.  The measured trade (4B, 5246-token prefill against a 64-token decode): `-ub 512`
   ITL −6.8 % / prefill +20.0 %, `-ub 2048` −22.8 % / +24.1 %, `-ub 4096` −23.9 % / +19.4 % -- the win grows with
   the chunk width, and the prefill is FASTER because the decoder's row no longer joins the prefill's batch.

## r372-wip (2026-11-23, `WORKLOG.md` 261123.024) — a per-BUFFER cache inside a shared backend is shared state: the Meta split-state cache must be partitioned by view slot like the containers are (`prefill-decode-arms` v2 STEP 7g)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7g (a defect in this campaign's STEP 4).

1. **STEP 4 partitioned the Meta backend's view CONTAINERS per view slot and assumed the split-state cache went
   with them; the cache is a per-BUFFER member (`split_state_cache[8]` + one `_cur`/`_used`/`_tick`), and the
   buffer is the tensor-split backend's, shared by every context on those devices.**  `get_split_state()` WRITES
   it while walking a graph (it erases, recycles an LRU version with `clear()`, moves `_cur`), so two contexts
   computing at once resolved split states through one version pointer and one thread cleared a version the
   other held an iterator into: SEGV in `ggml_backend_meta_get_split_state`, measured on the decode arm's thread
   2 of 3 runs (Flash-Next, 3 devices `-sm tensor`, np 2, one slot decoding while another prefills).
   **Rule:** a cache that lives on a per-BUFFER object is shared state the moment two threads compute; partition
   it by the same selector the containers use (here the `thread_local` view slot), including its version
   bookkeeping.  Invisible with one request in flight -- every sequential gate passed on the same binary -- and
   invisible on one device, where no Meta backend is armed at all.
2. **The same measurement found a second, still-open blocker: hipBLASLt inside a graph capture with a second
   context.**  `Hip error: 'operation would make the legacy stream depend on a capturing blocking stream' (906)`
   from `hipblaslt.cpp` in `ggml_cuda_graph_evaluate_and_capture`, on the decode arm, 2 of 3 runs, *after* the
   cache fix.  Disabling either hipBLASLt (`ROCBLAS_USE_HIPBLASLT=0`) or the capture
   (`GGML_CUDA_GRAPH_MAX_TOKENS=0`) removes it, and they cost very differently on that shape: capture-off reads
   **22.9 ms/token** (ITL −57.5 % vs the single-context arm), hipBLASLt-off **41.2** (ITL −13.8 %).  This is the
   STEP 7d family (a process-wide/library hazard vs a second arm's capture), so the same reasoning applies: the
   exposure must be removed before promotion, and which escape the pair pins is a design decision, not a gate.

## r373-wip (2026-11-23, `WORKLOG.md` 261123.024) — the ITL win is the CHUNK WAIT it removes, so it is shape-dependent: −57 % on a shape with a 2 s chunk against a 23 ms decode, −6 % on one whose own decode is 64 ms (`prefill-decode-arms` v2 STEP 9)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 9.

Measured with 2-3 repeats per arm, slot A decoding 64 tokens while slot B prefills 5246 tokens
(`baseline/itl-arm.sh`): 4B Q8_0 1 GPU `-ub 2048` 20.38 -> 15.74 ms/token (**−22.8 %**, prefill +24.1 %);
Flash-Next 3 devices `-sm tensor` `-ub 4096` 53.83 -> 22.89 (**−57.5 %**, prefill +21.8 %); 27B Q8_0 3 devices
`-sm tensor` `-ub 2048` 64.45 -> 60.30 (**−6.4 %**, prefill +35.6 %).  The single-context arm's ITL is
`own decode cost + the chunk waits it eats`, and the pair removes the waits -- so the *relative* win is largest
where a chunk is long against a short decode, and it shrinks to ~6 % on a shape whose decode is already 64 ms
per token.  The PREFILL is faster on every shape (+18…+36 %): the decoder's row no longer joins the prefill's
batch.  A shape matrix that quotes one number for "the pair's speedup" is therefore wrong by construction.

**Confirmed at delivery defaults after STEP 7h's barrier** (hipBLASLt ON, capture ON): 27B 3 devices `-sm tensor`
`-ub 2048` 64.51 -> 59.64 ms/token (**−7.5 %**, prefill +31.4 %) and `-ub 4096` 64.03 -> 60.93 (−4.8 %, +25.1 %);
Flash-Next `-ub 4096` 53.99 -> 23.31 (**−56.9 %**, +21.7 %).  The barrier did not change the shape dependence --
it removed an abort (r374-wip) -- so the per-shape rule above stands, now on the shipping defaults.

## r374-wip (2026-11-23, `WORKLOG.md` 261123.025) — the library call that cannot run during a capture is its LAZY HANDLE CREATION, and the thread that aborts is not the thread that failed (`prefill-decode-arms` v2 STEP 7h)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 7h.

1. **A HIP library that touches the LEGACY (blocking) stream cannot be used while another thread records a graph
   -- and on this ROCm line the call that trips is `hipblasCreate`, not the GEMM.**  Creation initialises
   hipBLASLt, which synchronises the legacy stream, and it happens lazily on the FIRST GEMM of every
   (context, stream, device) -- i.e. exactly what a freshly started decode arm issues while the prefill arm
   records.  Guarding the GEMM dispatch alone left the abort at 2 of 3 runs; guarding creation took it to 7 of 7
   clean, and no GEMM is ever recorded INTO a capture on that shape (an instrumented run logged zero).  **Rule:**
   when a library is not capture-safe, guard its *initialisation* as well as its dispatch -- a lazy-create inside
   a hot call is the easiest one to miss, and it is the one a fresh second context makes first.
2. **The aborting thread is not the failing thread.**  The sticky HIP error surfaced on the capturing thread's
   next kernel launch (`ggml_cuda_op_swiglu` -> `unary_gated_cuda`), while the library error was set by the other
   arm.  Three hypotheses were tested and killed by measurement before the right one: exclusive GEMMs did not
   move the failure rate; an instrumented build showed zero BLAS calls inside any capture; and the stack alone
   pointed at a `swiglu` launch.  A crash whose stack names a kernel launch is evidence about the *sticky* error,
   not about the failing call -- instrument the boundary (who creates what, when) rather than reading the stack.
3. **The trigger is a conjunction, so every single-variable probe comes back "clean":** it needs two arms that
   overlap, a graph capture, AND hipBLASLt.  With one request in flight (every gate), or capture off, or
   `ROCBLAS_USE_HIPBLASLT=0`, the shape passes -- which is why five gates and a 7-shape matrix were green on the
   same binary.  A concurrency defect on a two-context design needs an arm with two computes in flight.

---

## r375-wip (2026-11-24, `WORKLOG.md` 261124.001) — a DRAFT context may keep no `ctx_other`, and then it shares the target's compute base (`prefill-decode-arms` v2 STEP 10)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 10.

1. **`cparams.ctx_other` is not "am I a draft context" — the constructor keeps it only for the archs that need a
   target VIEW** (`gemma4-assistant`, `eagle3`, `dflash`, `qwen4exp`; `llama-context.cpp`, "TODO: more generic").
   An **embedded MTP head of a qwen35 target** (the 27B/35B MTP shapes, i.e. the delivery's own MTP path) has its
   own `tok_embd`/`output`, so `ctx_other` is `nullptr` and `slab_aux_slot()` returned 0: the draft's compute
   buffer took the same slab narrow base as the target's.  On the single-context server that is SAFE only because
   the two contexts are serialised on one thread (both buffers are scratch, dead between steps); with two arms it
   is two live views of one base.
2. **Measured**: 35B-A3B Q8_0 `-ncmoe 40`, pair + `draft-mtp-adaptive` n_max 7, 4 slots, `GGML_CUDA_WORK_ALLOC_DEBUG=1`
   -- `GRAPH_RESERVE ctx=<draft> is_draft=0` and `WORK_ALLOC need=512 narrow=1 ret_off=0.0` (base 0) for the draft
   while the decode arm held its own region, and `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in
   `rms_norm_mul_rope_f32_qk_pair` 15 s into the soak (40/40 requests failed).  **Rule:** a context that runs
   CONCURRENTLY with its target needs its own compute scratch, and the flag must be explicit
   (`llama_context_params::ctx_aux`), because the arch-derived `ctx_other` answers a different question.
3. **The same class covers `draft-simple`** (its own model, no `ctx_other`): the spec init sets `ctx_aux` for every
   draft context it creates **when the pair is live** (`ctx_tgt_alt != nullptr`), and only then, so the
   single-context layout (and every A1-A6 anchor and rate) is untouched.

## r376-wip (2026-11-24, `WORKLOG.md` 261124.001) — an allocation and a cross-stream event wait are NOT capturable by another thread: the capture barrier must cover both (`prefill-decode-arms` v2 STEP 10)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 10, extending r374-wip's barrier.

1. **A device allocation may not overlap another thread's graph capture.**  The driver refuses it and POISONS the
   capture; every later launch on that device then fails with `operation failed due to a previous error during
   capture` -- so the aborting thread (`hipStreamSynchronize` inside `ggml_backend_cuda_buffer_set_tensor`) is not
   the failing one (r374-wip item 2 again).  Measured: 4 slots on the host tier + a drafter, the prefill arm
   re-reserved 3 x 768 MiB while the decode arm captured its verify graph -> 40/40 requests failed.
   **Rule:** `ggml_cuda_capture_excl_guard` (EXCLUSIVE, and skipped on the recording thread, whose own pool
   allocations are part of the capture) in `ggml_backend_cuda_buffer_type_alloc_buffer` -- the slab path maps VA and
   evicts arena tables, so it is an allocation too.
2. **A dependency on another stream's work is not capturable either**: `hipStreamWaitEvent` issued while a graph is
   being recorded fails with `dependency created on uncaptured work in another stream`.  Same fix, SHARED side:
   `ggml_backend_cuda_event_wait` and the peer staging copy (`ggml_cuda_peer_copy_staged`, which waits on its own
   `d3d`/`done` events).  Measured once in three soaks.
3. **Reserve a lazily-reserved second context UP FRONT at its widest batch** (`llama_graph_reserve` at pair
   creation): it removes both window classes on every shape (the widest layout is allocated before either arm has
   anything in flight) and it is what lets the slab's aux region be sized from the real layout (r377-wip).

## r377-wip (2026-11-24, `WORKLOG.md` 261124.001) — the slab's AUX regions: a slot off-by-one, a pin that could not grow, and a hole that was "mapped" but absent (`prefill-decode-arms` v2 STEP 10)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 10 (`ggml-cuda-slab.cu`, the file blocks 12/15 own).

1. **The floor call passed a 1-based slot where the CUDA side takes 0-based** (`slab_aux_slot()` returns 1/2, the
   alloc path does `g_slab_compute_aux - 1`, `llama-context.cpp` passed it un-subtracted).  So
   `slab_aux_floor` pinned a PHANTOM region one slot up -- visible as a `could not map ... aux narrow-3` warning on
   a shape with ONE aux context -- and the real region was always sized by the first ALLOCATION.  **Rule:** two
   call sites of one 1-based/0-based convention must be fixed together; a "no-op in practice" call is a defect.
2. **An aux region could not grow, and the first request is NOT always the widest layout** (the pin's own comment
   claimed it was).  Measured: the pair's decode arm reserves its prompt graph first (768 MiB) and its verify-width
   graph after (1280 MiB) -> the pin refused the second for the whole run and the buffer took a plain `cudaMalloc`
   *while the arena had given up 1280 MiB for it*.  Now: a pin grows when nothing is live at its slot, and the
   commit maps only the new part below the old base.
3. **A second aux region stacks BELOW the first, into the hole the first pin left.**  `s.mapped` jumps to the pin's
   END, so the VA between the arena's top and the pinned base counts as mapped and has no physical behind it -> a
   device write there is `HSA_STATUS_ERROR_MEMORY_FAULT` ("Page not present"), measured at LOAD in
   `ggml_cuda_ain_batch` with a 512 MiB draft region under a 1280 MiB borrowing region.  The commit now maps a
   region that lies entirely below the extent (and declines if the range really is mapped).  **Rule:** `mapped` is
   an extent, not a map -- anything placed inside it must be mapped explicitly.
4. **The census line is the operator's mechanism**: after these fixes the log reads `3328 MiB would host every
   auxiliary region (headroom 1536 + aux 1792)` on the 35B-A3B pair+MTP shape, which is the
   `GGML_CUDA_SLAB_RESERVE_MIB` to set (zero evictions at that value, both regions from the unmapped spare).

## r378-wip (2026-11-24, `WORKLOG.md` 261124.001) — re-admitting a drafter to two contexts: what the text oracle must assert (`prefill-decode-arms` v2 STEP 10)

*No delivery revision:* `wip/prefill-decode-arms-v2/` STEP 10.

1. **A text claim alone is VACUOUS for a drafter on the pair.**  If the drafter reads the wrong target context (the
   prefill arm's rows instead of the batch's), every draft is garbage, every one is rejected, and the text is
   unchanged -- the run looks perfect and the feature is silently dead.  The gate must assert drafts were **accepted**
   (`#gen drafts > 0` AND `#acc drafts > 0`) as well as the decode-arm census.  Measured non-vacuity on the pair:
   34/25, 32/26, 32/26 (48-token arms) and 74-181 drafts with 73-161 accepted (600-token axes).
2. **A verify step emits up to `n_max + 1` tokens, so a soak's "two decoders shared one decode batch" check needs
   more rounds to see the overlap**: at 16 tokens per request a decoder finishes in ~6 decode-arm steps, and one of
   eight 10-round soaks never saw `widest batch >= 2` (no fault, no error -- the CHECK failed).  20 rounds: 2/2.
3. **The pair's four-arm claim is `pair+draft == pair+none == single+draft == single+none`**, one fresh process per
   arm, one request in flight, `-lv 4` (the text comes over HTTP for a server arm, so the census cannot pollute it).
   Identical on 3 devices f16 (`f2753e13e419`), 3 devices q8_0 (`f2753e13e419`) and 2 devices f16 (`0f06ed90e017`).

## r379-wip (2026-11-24, `WORKLOG.md` 261124.002) — the two-arm reserve is a COUNT of auxiliary compute regions, one per auxiliary context (`prefill-decode-arms` v2, the maintainer's decision)

*No delivery revision:* `wip/prefill-decode-arms-v2/`.

1. **A reserve for "the pair" is the wrong shape: what the arena must hold back is ONE pinned compute region per
   auxiliary context.**  `common_arms_reserve_derive()` now returns `{n_aux, bytes}` and the log names the
   arithmetic; it replaced the flat `COMMON_ARMS_RESERVE_MIB` (1280 MiB for "the decode arm") so a third
   decode-band context is charged without editing a constant.
2. **Do NOT count a context twice.**  A speculative drafter's context needs a region too, but its charge is
   `common_draft_reserve_derive()`'s `compute` term (derived, or the flat `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT`
   fallback), so `n_aux` counts only the decode arm today.  Counting both would withhold 2560 instead of 1280 MiB
   of arena on the measured shape -- reserve 6144 vs 4864 MiB, with nothing using the difference.
3. **Measured, and the number to expect:** 35B-A3B Q8_0 `-ncmoe 40` np 4 + `draft-mtp-adaptive` n_max 7 (host
   expert tier, slab armed) -- regions: decode arm 1280 MiB at 32 verify rows, draft 512 MiB; arena reserve
   4864 MiB; arena 17467.5 -> 16192.5 MiB vs the pair off, hit rate 0.9173 -> 0.8968, generation 52.94 -> 48.22 t/s,
   text identical.  The slab's calibration line states the total to set (`3328 MiB would host every auxiliary
   region (headroom 1536 + aux 1792)`), and the region grows with the verify width (`n_parallel x (n_max+1)`), so
   a much wider verify batch is a re-measurement, not an assumption.

## r380-wip (2026-11-24, `WORKLOG.md` 261124.003) — a stale TOOL binary beside a fresh `libllama` segfaults, and the delivery sweep owes the whole-set build FIRST (`prefill-decode-arms` v2 STEP 11/12)

*No delivery revision:* `wip/prefill-decode-arms-v2/`, `beta/prefill-decode-arms/`.

1. **`llama-perplexity` (and `llama-gen-docs`) segfaulted with ESegv on start, and it was a STALE BUILD ARTIFACT --
   not a defect in the change under test.**  The tools are thin execs plus a `libllama-<tool>-impl.so` plugin linked
   against `libllama`; the campaign only ever rebuilt `llama-cli`/`llama-server`, so those two plugins were 8 hours
   older than the `libllama` that later landings changed (a struct layout moved), and they died before printing a
   line.  **Rule:** build the whole `all` profile (`~/bin/build-llama-rocm`) BEFORE the gate ladder, not after, and
   when a tool segfaults on start read `ls -la build-*/bin` before reading the diff.  Four `llama-*-cli` binaries
   from an older CMake config (`llama-{minicpmv,qwen2vl,llava,gemma3}-cli`, 2026-11-21) are still sitting there the
   current config does not build; no gate uses them.
2. **`check-gen-docs.sh` FAILing on a campaign tree is the EXPECTED answer when the campaign adds an option** -- the
   rows appear in `tools/server/README.md` and nothing else.  The delivery's rule (one regeneration, folded into the
   last block that touches `common/arg.cpp`) means the regeneration belongs to the CANDIDATE, and the check is green
   on the applied candidate tree.  A campaign tree is never the tree that check is green on; do not "fix" it early.
3. **A `beta/` candidate is `format-patch` output, cut against the RELEASE TREE, and neither is optional.**
   `apply-all.sh --include-beta` runs `git am <beta>/*/*.patch` AFTER the release blocks (strictly, one commit
   each), so (a) a base-relative patch cannot apply, and (b) a plain `git diff` is not `git am`-able at all
   ("Patch format detection failed").  Both were true of `HANDOVER.md` §12.1's `git diff <base>` instruction and the
   previous artifact (30 files, release-tree-relative) was the counter-example on disk.
4. **The candidate is `release.json.tree -> campaign tree` (`27db9bfd4 -> 84ffb1ef1c`, 42 files, +2854/-715) and the
   replay is the proof it is the campaign:** `apply-all.sh` at `919794500` reproduces `.tip` `089f90067` exactly (the
   deterministic-commit-ids contract), the 19 `steps/` replayed in chronological order reproduce the campaign tree
   `14dfb1f64` exactly, and one `git am` of the candidate on the release tree reproduces `84ffb1ef1c` exactly.  A
   tree that matched by *hand-fixing* the patch would not say that.
5. **A gate whose default model is ABSENT fails as a `no model at ...` usage error, and that reads exactly like a
   defect.**  `gate-kv-residency.sh`'s `MODEL_POOL` (`.../Flash-Next/GSQ-IQ3_XXS/...`) and the `-Q8_0` MTP sidecar
   its `DRAFT_POOL` derives from the same directory were both missing from this box at the sweep's start, while
   `gate-draft-ctx.sh` already defaulted to `GSQ-RCO-IQ3_S` -- so the sweep first ran the gate on `GSQ-RCO-IQ3_S`
   (PASS), then, with the model tree restored, on **its own defaults** (PASS, 89.07 % pool hit rate).  **This is an
   environment gap and NOT delivery gate debt** -- the name is the gate's own documented shape and its arms are
   self-referential, so there are no recorded anchors to re-record and nothing to repoint (`TODO` #369 covers the
   three gates whose anchors DO bind them to a model class, and it is not one of them).  The rule for the next
   sweep: check that every gate's default model exists BEFORE running the ladder, and read the failure as `check
   the box` first, `check the diff` second.

## r381-wip (2026-11-24, `WORKLOG.md` 261124.004) — a slot teardown must QUIESCE the arm that may be mid-round on it (`prefill-decode-arms` v2 STEP 13, from a beta tester)

*No delivery revision:* `wip/prefill-decode-arms-v2/`, `beta/prefill-decode-arms/`.

1. **A main-thread `release()` of a DECODE-owned slot is a data race on the slot, and `server_slot::release()` had
   no arm guard on the CANCEL path.**  Aborting one of two long generations (a client disconnect -> a
   `SERVER_TASK_TYPE_CANCEL`) released the slot the decode arm had ALREADY collected for its round: `reset()`
   zeroed `stats` (`t_start = 0`, `n_gen = 0`) and the task, and the arm's per-slot step -- which runs after
   `llama_process()` returns -- saw `n_gen == 0`, took the first-token branch, and aborted at
   `GGML_ASSERT(t_start > 0)` (`tools/server/server-common.h`).  **The rule:** taking a slot away from an arm is
   two steps, both under the arms mutex -- (a) clear the ownership so no LATER round claims it, (b) wait for the
   round that already holds it.  `release_slot()` is that; the claim (`server_slot::arm_busy`) must be taken in
   the SAME critical section that COLLECTS the round's slots (a separate claim leaves a window in between), and
   cleared on every exit path (an uncleared claim deadlocks the next release).
2. **Reproduce with `pair OFF` before blaming the feature, and the other way round:** the same recipe died 2 of 2
   with the pair ON and survived 25 of 25 with it OFF -- that pair of numbers is what classifies a crash as the
   arms path.  `gates/gate-arms-cancel.sh` is the regression instrument: it fails on the pre-fix build (round 2)
   and passes after, and it asserts non-vacuity (`the decode arm is generating: 2158 step(s), 25 handover(s),
   widest batch 2`) because an ABORTED request never sets `finish_pending` and so can never print the census line
   -- abort rounds alone are vacuous.

## r382-wip (2026-11-24, `WORKLOG.md` 261124.004) — the FD ceiling: RCCL's IPC rings cost ~646 descriptors per context, and `FD_SETSIZE` is not raisable for a prebuilt library (`prefill-decode-arms` v2, beta tester report)

*No delivery revision:* the measurement only; the fix is not landed.

1. **A `-sm tensor` split on 3 devices + the pair aborts with `*** bit out of range 0 - FD_SETSIZE on fd_set ***`,
   and it is descriptor exhaustion, not memory.**  Measured on Flash-Next `-ncmoe 48` + the attendance pool:
   **686 fds (646 `/dev/shm/nccl-*`) with the pair OFF, 1356 (1292) with it ON, 568 with 2 devices, 10 on one
   GPU.**  Every `llama_context` brings up its own RCCL comm layer (`GGML_CUDA_ALLREDUCE=auto` inits RCCL
   EAGERLY, host-expert shape or not), so a second context doubles the rings past `FD_SETSIZE` = 1024.  Note the
   DELIVERY alone already sits at 646/1024 on this shape.
2. **`FD_SETSIZE` cannot be raised for the offender.**  It is a compile-time constant of the TU that expands
   `__FD_SET` (glibc's inline check), and the `select()` in question is inside a PREBUILT library -- there is no
   `select()` in this tree at all (the vendored httplib 0.67 is thread-per-connection).  `ulimit -n` is unrelated
   (524288 here).
3. **`GGML_CUDA_ALLREDUCE=ce` removes the whole class: 11 fds with the pair OFF and 11 with it ON.**  The CE chain
   never brings RCCL up.  It is a supported selector value and its large-tensor arm measures FASTER than RCCL's on
   a 3-device config (pp2048 2227 vs 2187 t/s), but `ggml-cuda.cu`'s dual-arm default prefers the RCCL ring for
   host-resident experts, so it is a trade until the pair stops carrying a second RCCL.
4. **Do NOT "fix" this by sharing one communicator between the two contexts.**  The two arms compute concurrently
   (that is the feature) and a collective's ORDER must match on every rank; two unsynchronised arms cannot
   guarantee that, and any local lock orders them differently per rank.  The sound directions are a per-context
   all-reduce chain for the decode context (it only ever reduces decode-sized tensors) or a smaller RCCL
   descriptor footprint.

## r383-wip (2026-11-24, `WORKLOG.md` 261124.005) — ONE RCCL comm layer per process: an auxiliary context must never add another (`prefill-decode-arms` v2 STEP 14, from a beta tester)

*No delivery revision:* `wip/prefill-decode-arms-v2/`, `beta/prefill-decode-arms/`.

1. **A second `llama_context` on a `-sm tensor` split aborted the process with
   `*** bit out of range 0 - FD_SETSIZE on fd_set ***`, and it was the FD COUNT, not memory.**  RCCL's IPC rings cost
   one `/dev/shm/nccl-*` descriptor per peer per channel = **646 open fds per comm layer** on 3 devices; the delivery
   alone is at 63 % of `FD_SETSIZE` (1024) there.  Measured: 686 fds (pair off), 1356 (pair on), **2026** (target +
   MTP draft + decode arm), 568 (2 devices), 10 (1 GPU).
2. **`FD_SETSIZE` is not raisable for the offender.**  The range check is glibc's inline `__FD_SET`, expanded in the
   CALLER's TU, and the `select()` that trips is inside a PREBUILT library (RCCL/ROCr) -- there is no `select()` in
   this tree and the vendored httplib is thread-per-connection.  `ulimit -n` is unrelated (524288 here).
3. **The rule: an AUXILIARY context uses the copy-engine chain; exactly one RCCL comm layer exists per process.**
   "Auxiliary" is the identity `slab_aux_slot()` already defines (`ctx_memory` borrower, `ctx_other` draft view,
   `ctx_aux`); `llama_context` asks for CE for it (a capability-probed, consume-once backend proc) AND ggml-cuda
   refuses a second `ncclCommInitAll` outright, so the invariant does not depend on the request arriving.  Free
   where it applies: an auxiliary context only reduces decode-sized tensors, which every chain routes to the
   internal pipeline; the PREFILL context keeps the ring (pp2048 809 vs 604 t/s on CE, `ggml-cuda.cu`).  Measured
   after: 686 / 686 / 686 (base / pair+drafter / pair+drafter) = **0 extra fds**, text unchanged.
4. **Behind it (OPEN at the time, FIXED in STEP 15 / r384-wip):** a 2-D read with `n_copies > 1` of a row-split
   (vocab-split) 2-D tensor aborted with `invalid argument` (`331092 bytes x 5 copies, tensor pitch 331092, host
   pitch 993280`).  The first reading of it here — "the Meta backend's chunked path hands a per-device read a range
   past that device's slice" — was **WRONG**: the per-device arithmetic is exact for that shape.  The refusal is
   the peer grant a 2-D device->host copy needs on the destination's pinned allocation; see r384-wip.

## r384-wip (2026-11-24, `WORKLOG.md` 261124.007) — a 2-D D2H copy into a pinned buffer ANOTHER device owns is refused on this box: row blocks are read as 1-D copies (`prefill-decode-arms` v2 STEP 15)

*No delivery revision:* `wip/prefill-decode-arms-v2/`, `beta/prefill-decode-arms/`.

1. **The Meta backend's 2-D arithmetic was not the defect** (it was the campaign's first suspect).  Measured at the
   abort, `result_output` = f32 `ne=[248320,5,1,1]`, `nb[1]=993280`, `split_state.axis=0`, `n_segments=1`,
   `nr[0]=1`, `ne[] = {82773,82773,82774}` — per-device chunks 331092 / 331092 / 331096, which sum exactly to the
   meta row (993280) and to `ggml_nbytes` (4 966 400).  `simple_offset = i_start*chunk_size_j` and
   `n_copies = i_stop - i_start` are correct for this shape, and **device 0's copy of the same arguments
   succeeded**.  (A multi-segment / multi-repeat state never reaches that branch: the walk above it handles those.)
2. **The refusal is the peer grant on the DESTINATION.**  `AMD_LOG_LEVEL=3` on the aborting run names it
   immediately before HIP returns the error:
   `rocdevice.cpp: Allow p2p access failed - hsa_amd_agents_allow_access with err 4096` ->
   `hip_memory.cpp: hipMemcpy2DAsync: Returned hipErrorInvalidValue`.  A `hipMemcpy2DAsync` D2H asks the runtime to
   grant the COPYING device's agent access to the destination's pinned allocation.  llama.cpp's pinned output buffer
   is ONE device's `ROCm_Host` allocation (`llama_context::output_host_buft`) read from every device of a
   `-sm tensor` split, so every non-owner device needs a grant — and this box refuses these grants (it refuses
   `hipMemSetAccess` too, which is why `ggml_cuda_peer_copy_ok` / the staged peer copy exist).  Device 0 owns it and
   needs nothing, which is exactly the observed asymmetry.
3. **A 1-D copy of the same bytes is accepted from every device.**  Measured in the aborting process: `n_copies`
   1-D copies of the same block succeeded on device 1 straight after the 2-D form was refused.  A REPEAT of the
   identical 2-D call also succeeded, so the refusal is state-dependent — never use "the 2-D copy failed" as a
   probe, and never rely on a retry.
4. **The rule: the CUDA backend issues NO 2-D device->host copy into host memory** (the class is already recorded
   for the movable-boundary slab — r183/r253 — and for the mixed KV region — r204/#313).  `ggml_cuda_rows_to_host()`
   serves a tight block (both pitches = the width) or a single copy as ONE 1-D copy; a big synchronous block through
   the device-staged form (2-D device-to-**device** gather into the reading device's own staging buffer — a device
   buffer needs no grant — one 1-D D2H per chunk into that device's own pinned chunk, then a host scatter); and the
   ASYNC form (which must not synchronize) or a small block as one 1-D copy per row.  The write direction needs no
   change: every pitched upload already goes through the device's own pinned slot (`splice_pin_ring`,
   `ggml_cuda_stage_pin`) and never starts from a foreign pinned buffer.
5. **Reachability: it is the ORDINARY logits readback, not a state save.**  The caller is
   `process_ubatch` -> `ggml_backend_tensor_get_async(backend_res, t_logits, …)` for a ubatch with 5 output rows.
   That multi-row verify batch is composed by the two-arm decode arm's draft verification: the pair-OFF control on
   the identical shape never produces it (`not engaged: ubatch of 2/8/13 tokens` and no 5-token ubatch), which is
   why the pair-OFF reproducers "survived" the defect.  It is nevertheless DELIVERY code (`ggml-cuda.cu`), reachable
   by any multi-row read of a split tensor into a pinned host buffer — see `TODO` #376 for the promotion obligation.
6. **A standalone HIP probe cannot reproduce it** (`wip/prefill-decode-arms-v2/repro/probe-hip-2d-copy.cpp`: every
   argument combination, cross-device destinations, 1..6 copies, mapped/default flags, in-flight pairs — all
   succeed on an idle box).  The refusal depends on the machine's grant state, so the gate is end-to-end
   (`gates/gate-arms-state-split.sh`) and the driver log is the primary evidence.

## r385-wip (2026-11-24, `WORKLOG.md` 261124.009) — a disabled compute-buffer drop silently costs CUDA graph capture, and that is ~half the decode rate: fence the drop, do not disable it (`prefill-decode-arms` v2 STEP 16)

*No delivery revision:* `wip/prefill-decode-arms-v2/`, `beta/prefill-decode-arms/`.

1. **With the pair on, uncontested decode was half of the single-context rate and the pair was innocent.**
   Measured on the maintainer's `runme` shape (Flash-Next GSQ-IQ3_XXS, `-ncmoe 47`, pool, `-sm tensor` 2 devices,
   `-np 2`, `draft-mtp`, `-ub 4096`, 1000 tokens): pair off **83.2 t/s**, pair `auto` **45.9 t/s** — and the same
   single-context server with only `LLAMA_DROP_COMPUTE_BUFFERS=0` reads **46.5 t/s**.  Identical draft counts
   (1298/565) everywhere, so it is not speculation; it holds with experts on the GPU, so it is not the arena.
2. **The mechanism: no compute-buffer drop -> the slab's wide band stays armed -> ggml-cuda refuses to capture.**
   `ggml_cuda_graph_compute` will not start a capture while `ggml_cuda_slab_wide_available()`, and the band-armed
   warmup branch sets `capture_deferred` **without a log line** (it is silent even under
   `GGML_CUDA_GRAPH_TRACE=1`).  The pair had disabled the drop outright (STEP 7d/7f) to keep its process-wide
   mutation away from the other arm's capture, so it never captured a graph at all: 0 captures against 2922 in the
   single-context arm.
3. **The rule: fence the drop, never disable it.**  `llama_context_set_drop_boundary()` is called around the whole
   `drop_now` block and the two-arm server implements it with `arms_peer_boundary[]` — the drop waits until the
   OTHER arm is between steps and keeps it out for the mutation.  The capture barrier alone is NOT sufficient: it
   fences captures and library GEMM calls, while a REPLAY is one `cudaGraphLaunch` whose GEMMs are never dispatched
   and so never take it.  The calling arm is not waited for (the drop runs inside its step).
4. **Result:** 45.9 -> **88.4 t/s**, 3140 captures, arena 31 551 -> 37 199 MiB (the stood-down tables come back via
   the drop's own `moe_cache_rearm`), prefill 66 -> 116 t/s, single-context unchanged.  Consequence for any future
   "disable X to avoid a race" decision in this codebase: X usually also disables a *reclamation*, and the
   reclamation is where the rate lives.
5. **`ggml_cuda_graph_mem_freed()` is what makes the capture side safe** (TODO #70): an eviction bumps the graph
   memory generation, so a capture that baked a freed arena address is invalidated instead of replayed.  A
   *replay already dispatched* is the residual hazard, which is why the fence is the step boundary and not the
   barrier.

## r355 (2026-11-24, `WORKLOG.md` 261124.011) — a fusion matcher can pay its whole-graph reader counts TWICE per layer, and on a graphs-off harness that is a decode tax (block 10, contributor PR #361)

1. **The defect was the r353 fold's own, and it only showed on one configuration.**  `ggml_cuda_gdn_norm_gate_match()`
   ends with three whole-graph reader counts — `ggml_cuda_count_uses()` for the `RMS_NORM` and for its `MUL`, and the
   gate `MUL`'s inside `ggml_cuda_find_mul_q8_1_matmul()` — and after r353's SILU widening it ran on Qwen3.x, i.e. on
   the delivery's own GDN families.  It ran **twice per GDN layer** (the skip at the norm, the gate node), the gate
   then counted its `MUL` a third time: **seven graph walks per layer** against one for the unfused gate.  Measured on
   the 10.2 build, 27B tg128, `GGML_CUDA_GCDBG=1`: `try_fuse` **1 323.3 ms** over 258 `graph_compute` =
   **5.13 ms per call**.
2. **Where the cost is paid decides whether you can see it.**  A captured, replayed graph pays **nothing** (the node
   loop is not run), so with the default graphs on the fold costs a flat 0.00 t/s: 32.15 / 32.18 t/s and `pp512`
   1447/1450 in both arms.  With `GGML_CUDA_DISABLE_GRAPHS=1` it is **31.13 -> 31.77 t/s (+2.1 %)** once the walks are
   gone (`fuse` 752.4 ms, 2.91 ms per call).  That is the class issue #359 reported as −1.8 %: **the reporter's server
   harness sets `GGML_CUDA_DISABLE_GRAPHS=1`**, so a "host-only, bit-identical, invisible" change was a real regression
   on a documented configuration.  Rule: a host-time claim is per *serving mode* — replayed, direct and recaptured are
   three different products, and a graphs-off arm is the only one that prices the matcher.
3. **A matcher may not re-decide at a later node what an earlier node already decided, unless the invariant is stated
   and tripwired.**  The skip records the gate index (`gdn_norm_gate_at`) and the gate node trusts it.  That is sound
   only because the skip's own match proved the pending norm has **exactly one reader** (the gate's `MUL`), so no other
   node in the graph can produce that decision; the dispatcher's existing `i < gate_at` guard keeps every other fusion
   out of the window, so the graph cannot move underneath it.  Both conditions must be written down — the assert that
   encodes them (`direct || nodes[s+2]` is the `RESHAPE` over the gate `MUL`) is the tripwire, and it has to run
   **before** the pending state is cleared, or a broken invariant silently fuses the wrong matmul.
4. **A peek past a node is a read, and the reshape branch is the live one.**  The submitted form derived the fused
   matmul as `nodes[s + 2]->op == MUL_MAT ? nodes[s + 2] : nodes[s + 3]` — no bound on `s + 3`, assert after the read.
   On every shape measured the gate `MUL` is followed by a no-op `RESHAPE` then the matmul (**7 354 of 7 354**
   decisions took that branch, `direct=0`), so the unguarded read is the taken path.  Bound it.
5. **The decision was verified by re-running the OLD matcher, not by arguing.**  An instrumented arm called the
   pre-change `ggml_cuda_gdn_norm_gate_match()` at every gate node and compared `r`, the norm tensor and the matmul:
   **7 354 decisions, 0 mismatches** across qwen35 dense (1 GPU and 3-GPU `-sm tensor`, graphs on and off), qwen35moe
   and qwen4exp/Flash-Next.  For a "the same decision, cheaper" change this is cheaper than a KLD ladder and stronger
   than text alone, and it is how the SIGMOID arm (which no gate here can build at full size) gets covered.

## r356 — TQ1_0 / TQ2_0 on the GPU: the bit-identity to the CPU is a property of the ACCUMULATION, and the arm that was supposed to prove it was a placement (block 17, contributor PR #358)

1. **A "bit-identical to the CPU" kernel must reproduce the CPU's arithmetic AS COMPILED, and that is not the
   arithmetic in the CPU's source.**  `ggml_vec_dot_tq{1,2}_0_q8_K` writes
   `sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf)`; a Release clang/GCC build of
   `ggml-cpu` emits `vmulps` + `vaddps` and **no `vfmadd`** for it (disassembled on this build, `-march=native`).
   So the GPU's per-Q8_K-block accumulation must be an **unfused mul and add** (`TQ_FADD(TQ_FMUL(dot, y.d*x.d),
   acc)`).  The submitted patch used `fmaf`, and its claim ("`LLAMA_TEST_OP_NMSE=1` … every error exactly 0") did
   not reproduce: **25 of 61** `MUL_MAT` cases read ≈1e-14.  After the fix **305/305 `MUL_MAT` and 84/84
   `GET_ROWS` values read exactly 0**, on gfx1201 and on gfx1151 alike, and the fix costs no rate (TQ2_0 tg128
   181.5 → 180.7, TQ1_0 198.2 → 197.7 t/s).  `GET_ROWS` needed nothing: it dequantises, it does not reduce.
2. **The failure has a diagnostic SHAPE, and the shape is the whole test.**  With a fused accumulation every
   `k=256` case (one block, one addition onto a zero accumulator) is exactly 0 and **every** `k>256` case is not.
   A bit-identity arm therefore has to include a multi-block length — a single-block arm passes a fused kernel —
   and `scripts/gate-tq.sh` prints this `k=256`-vs-`k>256` split before it fails (`GREEDY-PURITY.md` §109).
3. **`GGML_CUDA_TQ=0` is a PLACEMENT, not a reference, and a text hash across the two placements is not an
   oracle.**  Moving the TQ matmuls to the CPU backend changes nothing about the TQ dot product — the op oracle
   reads exactly 0 in both `-sm layer`/`-sm tensor` splits and on both boxes — but it also moves that block's
   neighbouring ops onto the CPU, so the two arms are different configurations (the §38/§82 class).  Measured: on
   gfx1201 the default, `GGML_CUDA_TQ_FUSE=0` and `GGML_CUDA_TQ=0` arms read one text at 64 and at 200 tokens
   (all 539 characters shared); on **gfx1151** the first two agree at both lengths and the CPU arm shares a prefix
   and then settles into a different repetition basin on this degenerate 1.58-bit model (326 vs 538 characters) —
   with the NMSE still 0/305 **there**.  The gate's HARD text arm is `default == TQ_FUSE=0` (same placement) and
   the CPU arm is reported with its shared prefix.  A release that quoted the three-arm agreement as a purity
   property would have quoted gfx1201's luck.
4. **The block also removes an abort it was not written for.**  On r355 a TQ model on a 3-GPU `-sm tensor` split
   dies at `ggml-backend-meta.cpp:1460: GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]))`, because every TQ
   matmul was placed on the CPU and the Meta split states diverged; with the block the split runs and the census
   is live on each of the three devices (`small=442 mma=221 vec=262 vec_fused=195` per device on a 10.1 build;
   `mma=2210` on the 1-device wide-prefill arm).  The kill-switch configuration on a split is still that abort/a
   hang and is not a supported configuration, which is why the gate does not run it there.
5. **The int8-WMMA prefill path needs a contiguous weight and must DECLINE, not assert.**  It converts TQ1_0 to
   TQ2_0 with one kernel over the whole tensor, so a strided `TQ1_0` `src0` keeps the small-batch kernel (which
   takes every batch) rather than tripping the conversion's contiguity assert.  Every split the delivery
   validates hands it a contiguous slab (measured), so the decline is for the class, not a live path.
6. **The default path is untouched, measured three ways.**  Text: 4B `1c5d32ac537d` on 1 GPU and 3-GPU
   `-sm tensor`, 35B-A3B `461ca8cd0e88` on 3-GPU tensor, both identical to r355, and on gfx1151 the dense text is
   `1c5d32ac537d` on both builds too.  Prefill logits: KLD
   **0.000000 / 100.000 %** against a base recorded from the r355 build on this ROCm line.  Rate: pp512 at parity
   over five interleaved rounds (spread ±190 t/s on both builds), tg128 +0.08 %, `npl 1,4,8` within 0.7 %.  A
   fusor added to block 10's dispatcher and a node reorder in `graph_optimize` both run for EVERY graph, so this
   neutrality is the evidence that they are inert.
7. **An arch with no MMA kernel is not a smaller version of the same test.**  gfx1151 (RDNA3.5, and the box builds
   `GGML_HIP_NO_VMM` — gate 40's other configuration) takes the small-batch kernel for every batch and still reads
   **305/305 exactly 0**, so the two kernel families are separately exact against the same CPU oracle; its `mma`
   counter is 0 by construction and must not be read as a failure.  Rate there is 3.4x/4.0x over the CPU
   placement (633.8 vs 187.9 pp512, 84.3 vs 20.9 tg128) — the APU's bandwidth, not the kernel.
8. **The CUDA (nvcc) build of `tq.cu` has never been compiled** (`TODO` #380).  The clang-only pragma is
   `defined(__clang__)`-guarded and `TQ_FMUL`/`TQ_FADD` fall back to `__fmul_rn`/`__fadd_rn`, and
   `-use_fast_math` flushes subnormals — so the claim to test on an NVIDIA box is the **NMSE column**, not the
   text, which will be coherent either way.

## r357 — the server flattened what the renderer marks as INPUT, so a tool result's special-token strings became tokens (block 12 amendment, contributor PR #360)

1. **`parse_special = false` does not stop a USER_DEFINED token, so "turn it off at the call site" is not a fix.**
   `llama_vocab::impl::tokenizer_st_partition` skips only `CONTROL | UNKNOWN` when `parse_special` is false and
   **still pre-tokenizes USER_DEFINED ones** (Qwen's `<think>`, `<tool_call>`, `<tool_response>`) — that is deliberate
   (the HF regex-split behaviour).  A tool result's forgeable strings are therefore only closed by *preventing the
   match*, never by disabling parsing: the shipped fix cuts each protected string after its first character and
   tokenizes the pieces with `parse_special = false`.
2. **The renderer already says which span is data and the server throws it away.**  `common/jinja` marks every
   interpolated value `is_input` (`common/jinja/string.cpp`; propagated by `mark_input_based_on` through
   `lower`/`replace`/`format`), `common_chat.h` documents it as *"from user input, special tokens inside must not be
   parsed"*, and `common_chat_prompt` carries it as `parts` — **and no tokenizer call site reads it**:
   `oaicompat_chat_params_parse()` flattens with `prompt().str()` and `handle_completions_impl()` tokenizes that one
   string with `parse_special = true`.  That flattening is the defect; the mark scheme is the workaround (`TODO` #382).
3. **A hand-rolled piecewise tokenization must be verified, not trusted.**  The fix re-derives
   `llama_vocab::tokenize`'s BOS/EOS and per-fragment behaviour, and several pre-types change a fragment on the way in
   (`metaspace_prepend_scheme "first"` prepends U+2581 to the *first* fragment of each call, so every piece would get
   one; SOPRANO lower-cases and collapses whitespace; NEMOTRON_3_5_EMBD rewrites `[`+ZWNJ; MMBERT/GEMMA4 have their
   own metaspace rules; `<mask>` carries LSTRIP).  `common/chat.cpp`'s `common_chat_input_tokenize()` is the house
   pattern — tokenize, detokenize, compare, fall back — and the fix now does the same (`tool_text_tokenize_ok()`),
   dropping to the space fallback on a mismatch.  Measured: the tokens detokenize to the prompt byte-for-byte on
   Qwen3.8-27B, Gemma4-12B (GEMMA4 pre-type) and a Llama-3 vocab; a deliberate one-character change in the
   re-tokenization reads `mode=space (the tokenizer changed the text)` and still protects.
4. **A server-capability predicate is not a request predicate.**  Gating the space fallback on
   `opt.allow_image/audio/video` (the mtmd *capability*) made **every vision-capable server** mutate the tool data the
   model reads on a TEXT-ONLY request — measured on Qwen3.6-35B-A3B + mmproj: `22C and sunny.< /tool_response><
   |im_end|>`, five inserted spaces.  It must key on the request's own media (`out_files.empty()`), which also means
   the two mtmd routing sites (`handle_completions_impl` and `handle_count_tokens`) must accept the token array a
   protected prompt becomes — `handle_completions_impl` picked `process_mtmd_prompt()` on `mctx != nullptr` alone and
   required a string prompt; `/v1/chat/completions/input_tokens` then answered 373 tokens for the same request.
5. **The forgery's *text* is not the signal; the prompt's special-token COUNT is.**  The same probe reads like a
   normal answer on both arms (measured: with thinking disabled the OFF arm refused the injection with "I can't do
   that." while the ON arm answered the weather question — behaviour is a model property, not a mechanism).
   `LLAMA_SERVER_TOOL_TEXT_CENSUS=1` is the instrument: the clean conversation is 20, the protected probes are 20 (= the
   same conversation without the forged text) and the unprotected arm is 25 (Qwen3.8-27B, `BOS`+turn boundaries plus
   the forged `<|im_end|>`/`<|im_start|>`).  This is "throughput is not a correctness signal" in the security
   direction, and it is why gate 43 exists (`scripts/gate-tool-text.sh`; the pre-fix build fails arms 4 and 7).
6. **A protection marker a client can send must be stripped only while the protection is armed.**  Dropping
   U+FDD0/U+FDD1 from every message unconditionally made `LLAMA_SERVER_TOOL_TEXT=0` *not* the previous behaviour
   (stock keeps a 1654-byte prompt holding both noncharacters; the submission's OFF arm read 1652).  Arming the strip
   with the feature makes the kill-switch arm byte-identical to stock on every probe, which is the only thing an A/B
   can rest on.

## r358 (2026-11-24, `WORKLOG.md` 261124.012) — the kv_rows (unified-KV row list) form now RUNS ON THE GPU on AMD: enabling it needs three other fixes, and the AMD instance set dictates WHERE it may be offered (block 05, GitHub issue #363)

`ggml_cuda_get_best_fattn_kernel`'s kv_rows branch offered the MMA kernel to `GGML_CUDA_CC_IS_NVIDIA(cc) &&
turing_mma_available(cc)` only, so on every AMD arch the rows form returned `NONE` and the scheduler placed each
joint attention layer on the CPU backend — the r308 defect's *consequence*, not its cause (r308's cause was the
vacuous probe).  A `--kv-unified` AMD server therefore decoded with a CPU attention launch per joint layer per step.
`--kv-unified` is on by default whenever the slot count is auto, so this was the common server configuration.

1. **The AMD MMA instance set decides WHERE the rows form may be offered, not just whether the arch has MMA.**  Every
   AMD instance in `flash_attn_ext_f16` is compiled out unless the GQA-optimised layout is chosen
   (`AMD_WMMA_AVAILABLE`: `ncols1*ncols2 < 16 || ncols2 == 1` → `NO_DEVICE_CODE`; `AMD_MFMA_AVAILABLE`:
   `ncols1*ncols2 < 16`), and that layout is exactly what `gqa_opt_applies` selects in the ncols dispatcher.  Without
   the gate, an MHA graph (or any `gqa_ratio == 1` shape, which falls through to `switch_ncols1<..., 1>`) launches a
   kernel that returns without writing its output: no abort, no log, an unusable result.  This is the same shape of
   trap as r308's: a backend predicate that is *arch*-shaped while the real condition is *layout*-shaped.
2. **A compacted K/V row list breaks every place that assumes `mask->ne[0] == K->ne[1]` — and the delivery's own
   padding makes that assumption false, not just multi-slot occupancy.**  The rows form's mask has `n_kv_cols =
   kv_rows->ne[0]` columns while `K->ne[1]` is the padded extent of the cells in that stream; they coincide only by
   luck.  The batch-wide packed fully-masked-group bitmap asserted equality and sized the bitmap from `K->ne[1]`;
   with `LLAMA_KV_N_PAD_PREFILL` (default 8192 on a multi-device split once the used extent passes 4096) or a second
   resident slot's cells, a >8-row prefill ABORTS (`GGML_ASSERT(mask->ne[0] == K->ne[1]) failed`, measured on a
   2-slot `--kv-unified` server, 275 + 5246 tokens, gfx1201).  Both now follow `n_kv_cols`, which is a no-op without
   the rows form.  Note what the abort *hid*: had the assert been relaxed alone, the bitmap would have been built for
   the wrong group count and the kernel would have read past its word run.
3. **A native (non-staged) K/V read must index through the row list; "the launcher staged an F16 copy" is not the
   only way an operand reaches the kernel.**  The q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl and bf16 MMA loaders honoured
   `indices` only in their `use_sparse` arm, so at the `<= 8`-row widths (where the native read is chosen:
   `native_width`, and on RDNA3_5 at every width) a quantized or bf16 KV cache would have read rows `k_VKQ_0 + i`,
   i.e. whatever cells sit at those offsets — silently wrong text, no fault.  They take `use_indices` now, like the
   half2 loader.  Measured: q8_0 and bf16 caches read the SAME greedy text with `GGML_CUDA_FA_KV_NATIVE=0` (staged)
   and by default (native), and both equal the f16 arm — the staged and native paths stage identical F16 tiles.
4. **A tuned fast path that never had an index arm must DECLINE the rows form, not inherit it.**  The RDNA4 band's 2-D
   launch passes a null index list into `process_tile` (so its round-robin blocks read rows by position) and its q8_0
   fetch-ahead ring is row-addressed.  Both are now declined for `dst->src[5] != nullptr`, which costs the band's
   decode win on `--kv-unified` head-128/256 RDNA4 shapes and is still a strict improvement over the CPU fallback;
   the shape falls through to the ordinary stream-k MMA path, which is what upstream runs with the rows form on
   NVIDIA (the band is RDNA4-only, so nothing validated changes there).
5. **A box property can flip a gate's expectation; grade the invariant, not the verdict.**  `gate-fused-probe-splits.sh`
   asserted "no arm may log `Flash Attention over K/V rows enabled` on AMD" — which was the *symptom* of "no AMD
   kernel reads the rows form".  The invariant the r308 gate exists for is that the rows form must never be PLACED
   ON THE CPU and never add a backend segment, so the gate now grades (a) no `... is assigned to device CPU` line for
   the rows form, (b) `splits` within the LAYOUT's bound, (c) `sparse` declines by observation (arch-independent),
   and (d) `dense-always`/`mtp-rdna34` by placement OR `enabled` (`KVROWS_EXPECT=enable|obs|place` pins the box).
   Measured (gfx1201, 3 GPUs `-sm layer`, Flash-Next GSQ-IQ3_XXS, bf16 KV, `-c 32768 -np 2`, 5246-token prefill + 48
   greedy tokens): all six arms `splits = 4` (= the bound), `plain`/`dense-always`/`mtp`/`mtp-rdna34` `enabled`,
   `sparse` by observation, every arm's text **`f6f0a8fd998b`** = the sha r311 pinned for the same-regime arms, i.e.
   the GPU rows path is bit-identical to the previously-validated dense form on that shape.  Negative controls: a
   `... assigned to device CPU` line, a stripped audited-probe verdict and `splits = 25` each still FAIL.

## r359 (2026-11-25, `WORKLOG.md` 261125.002) — the `--kv-unified` K/V-rows form is a LOSS on RDNA at every shape this box can reach, and it silently costs the derived kq mask: it is opt-in (block 05, GitHub issue #365)

r358 made the rows form run on the GPU on AMD (issue #363).  The reporter then measured it as a 17 % decode loss at
90k on a single R9700 and asked whether it should be declined where the band applies.  The answer is stronger: it
loses everywhere measurable, so the delivery's default went back to the dense form and the rows form became
`LLAMA_KV_ROWS_DISABLE=0` (upstream's "do not disable" = the opt-in; unset/`1` = off).

1. **A rows graph cannot take the derived (materialised-mask-free) kq mask — enabling the rows form silently DISABLES
   the derived form, and the derived form is the cheaper of the two.**  The two are alternatives by construction: a
   rows graph builds `ggml_flash_attn_ext_rows` and no plain `ggml_flash_attn_ext` node, so the derived-mask probe's
   graph never contains its op and r308's `require_observed` disables it.  Measured on gfx1201, 27B Q8_0,
   `-kvu -np 2`, `-lv 4`: rows on -> `derived kq mask flash attention was not used in the probe graph, set to
   disabled`; rows off -> `derived kq mask flash attention enabled`.  A rows graph therefore MATERIALISES
   `n_kv_rows x n_tokens` f16 of mask per ubatch and host-fills it, where the dense form allocates no mask at all and
   the kernel derives causality from positions.  This is the mechanism behind the reporter's Flash-Next arm, where
   the `LLAMA_KV_ROWS_DISABLE=1` run lost ~7 GiB of expert arena: the switch changes the mask form, not just the
   attention kernel.  qwen4exp (Flash-Next) has neither the rows nor the derived probe reachable through its own
   probe graph, so it reads the dense form either way (`INVARIANTS.md` r308's item 3 shape) — do not assume a
   Flash-Next A/B is isolating the FA kernel.
2. **The rows form is the stream-k MMA kernel, and at decode width the MMA family loses to the band/tile family at
   EQUAL work — the compaction has to beat that, and it does not.**  gfx1201, ONE R9700, `--kv-unified -np 2`,
   `-ctk/-ctv q8_0`, 94428-token prompt, greedy (`--temp 0 --top-k 1`), same-seed text identical across every arm:
   * Qwen3.8-27B-UD-Q5_K_XL, `-c 131072`, the prompt into slot 0 (row list == the whole extent): rows on pp 833.6 /
     tg 14.74 vs rows off pp 847.1 / **tg 23.41** (+59 %).  Then a 5246-token prompt into slot 1 while slot 0 stays
     resident (row list 18x shorter, the shape the rows form exists for): 1219.3 / 25.96 vs 1223.2 / **27.23** (+5 %).
   * Flash-Next GSQ-RCO-IQ3_XXS (host experts, `-ncmoe 47 --host-experts pool`): 1998.9 / 14.80 vs 1994.9 /
     **18.20** (+23 %), and 1654.4 / 31.98 vs 1664.2 / **34.34** (+7 %) on the compacted shape.
   * Inside the MMA family alone (`GGML_HIP_FA_BAND_WMMA=0`, dense model): rows 14.74 vs dense 21.78 — the row list
     itself costs ~32 % at equal work (`use_indices` in the native q8_0 loader is a per-row gather where the dense
     arm reads tile rows contiguously).
   So a compaction of 18x does not pay for the kernel-family switch, let alone a compaction of 1x.  A user who wants
   the form can have it (`LLAMA_KV_ROWS_DISABLE=0`), and it is still correct there (r358's three fixes) — it is just
   not the delivery's default.
3. **The opt-out must be measured on the shape the REPORT is about, and the two forms are a family switch, so text is
   the wrong instrument for it.**  `-kvu` coherence (27B Q8_0, 20 greedy tokens, 1 GPU and 3 GPUs `-sm tensor`) reads
   `42 chars sha=bdebe52759ef` on the r359 default and the same on an r356 build (whose AMD rows form was declined by
   the probe) — i.e. the default is byte-identical to pre-r358.  Within the rows form, the r358 gates' text equality
   still holds; ACROSS the two forms a text difference is expected and is not evidence of a defect.
4. **A delivery default flip must be pinned by its gate, not left to the probe's phrasing.**  `gate-fused-probe-
   splits.sh` gained a `default` arm (no env at all) that must log `K/V rows attention: dense form`, must never log a
   rows-form claim, and must read the same greedy text as the `rows-off` control; every arm that exercises the form
   now asks for it explicitly.  Measured (gfx1201, 3 GPUs `-sm layer`, Flash-Next GSQ-IQ3_XXS, bf16 KV,
   `-c 32768 -np 2`, 5246-token prefill + 48 greedy tokens): 7 arms, `default` + `rows-off` identical, every graph
   `splits = 4`, no CPU placement, no GPU fault.
