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
6. **An upstream UI asset bundle can break every fresh SOURCE configure -- and only that one.**  On
   2026-11-17 `ggml-org/llama-ui`'s `latest` (75 files) has no `workbox[hash].js`, so `ui-assets.cmake:137`
   aborts a configure that has no `tools/ui/dist`: `gate-per-block-build.sh` (it wipes the build dir per
   commit), a clean checkout, the hosted CI tiers.  Incremental builds never see it (the stamp caches the
   resolved bundle) and **the container images are NOT affected**: `.devops/rdna-rocm.Dockerfile` builds the UI
   in its `web` stage and `COPY --from=web /app/tools/ui/dist tools/ui/dist` runs BEFORE the cmake step, which
   is `ui-assets.cmake`'s Priority 1 (pre-built assets win over npm and over the HF fetch, so a container build
   neither downloads that bundle nor validates it).  The workaround where it does bite is
   `HF_UI_VERSION=b12387` (the bundle `build-rocm` had stamped; it carries `workbox-b3c04f83.js`).

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
