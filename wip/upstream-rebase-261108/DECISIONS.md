# Decision register — where upstream and the delivery answer the same question

Each item is a **stop-and-ask** point: the rung that reaches it does not pick a winner, it measures and
presents.  `AGENTS.md` → "Report findings before shipping" and the WIP rules apply: nothing here changes
`patches/` until the maintainer decides, and a dropped delivery mechanism is recorded (a `WORKLOG.md` entry,
`TODO.md` item, or an `archive/work/` record) in the same change.

**Status vocabulary:** `OPEN` (not reached yet) · `MEASURING` (rung reached, numbers being taken) ·
`AWAITING` (numbers in hand, maintainer's call pending) · `DECIDED` (resolution + record).

**The delivery's standing principle** (from `AGENTS.md` and the maintainer, 2026-11-08): the set aims to be
*complementary* — a second preferred path, not a replacement — so most of these should resolve to "both
paths ship, ours stays the default where it is faster, upstream's stays reachable".  A drop only happens
where the two cannot coexist (one implementation of the same function, or upstream's is strictly better).

---

## D1 — The MoE cache: upstream's cache grew a policy and a CPU share; ours is a different tier

**Upstream** (same subsystem, `src/llama-moe-cache.cpp`): `#29887`/`#30112`/`#30637` gave it a VRAM cache of
hot host-resident experts with a 50/50 SLRU policy over multiple GPUs; this range adds `#31233` (**ClassLRU**,
an LRFU/MQ-like hybrid with a half-life) and `#30966` (**`--moe-cache-cpu-share F`** — compute a share of the
misses *on the CPU* from the host experts instead of uploading them, plus `scripts/moe-cache-tune.py`), and
`#31497` (do not apply the cache to the draft model).

**Delivery**: blocks 09/12 add the host-resident **arena/pool tier** on a separate path
(`ggml/src/ggml-cuda/moe-expert-cache.{cu,h}`, `ggml-cuda-slab.cu`, the scheduler copy hook, the Meta
per-device forwarding, `MOE_HOST_POOL_*`), and it explicitly keeps upstream's cache as a *second, dormant*
path — `src/llama-context.cpp` carries `// upstream MoE cache (dormant unless --moe-cache-mib)`.

**Relation:** complementary by design, but they share the scheduler copy callback and the context's
allocation of "the device cache", so the merge is the work.

**Questions:**
1. Keep coexistence as-is (upstream cache off by default, arena on)? — *expected: yes.*
2. Adopt upstream's **CPU-share** idea for the pool tier?  It is a genuinely new mechanism (our fill uploads;
   upstream computes), and it targets exactly the configs where our pool declines/uploads at low residency.
   Upstream's own doc says the share "often does not help" when the host experts are pinned — which is our
   mirror tier — so the honest test is the *unpinned* (`-lm mmap`) pool shape.
3. Adopt **ClassLRU** into the arena's admission?  Our admission is heat/RFU-based and validated; ClassLRU is
   reachable today only for the dormant cache, so this is a "port a policy" change, not a merge.

**Measurement (one session, 1 × R9700, Qwen3.6-35B-A3B Q8_0, `-cmoe`/`-ncmoe 40`-shaped placement):**
`prefill t/s`, `tg128`, the cache census lines (`-lv 4`), and text parity, across
(a) delivery arena+pool default, (b) upstream `--moe-cache-mib <same MiB as the arena>`, (c) upstream
`--moe-cache-mib + --moe-cache-cpu-share 0.5/0.95`, (d) the pool at `-lm mmap` with and without a
cpu-share-style arm.  Report the three questions together.

- status: **MEASURING (R10, 2026-11-08) — numbers in, decision pending.**  First measurement, 1 × R9700,
  `Qwen3.6-35B-A3B Q8_0 -ncmoe 20 -c 8192` (16.3 GiB host experts), VMM build, equal MiB, text byte-identical in
  every arm: delivery arena 4080 MiB **547.6 / 53.5** t/s (pp/tg), upstream `--moe-cache-mib 4096`
  **544.2 / 62.9**, streaming **470.9 / 49.0**.  Upstream's ClassLRU cache wins decode by 17.6 % on this shape.
  **Not yet the decision**: one GPU and `-sm layer` (not the field's 2-3 GPU `-sm tensor`/bf16-KV shape), the
  arms are either/or rather than the coexistence the merge preserves, and the arena's pool/prefetch/staging
  tier is not exercised by a `-ncmoe` config at this size.  A proper session must repeat this on the field
  shape, with the delivery's host-pool tier on, and with `--moe-cache-mib` + the arena both enabled (coexist).
  **Build prerequisite discovered here:** the arena requires VMM and upstream's HIP default is
  `GGML_HIP_NO_VMM=ON`, so all these arms ran on a separate `build-rocm-vmm` (`-DGGML_HIP_NO_VMM=OFF`).

- status: **DECIDED (2026-11-08, `RUNGS` `P13`) -- keep both, ours default-on, upstream's opt-in only.**
  Re-measured on real text (the `llama-bench` random-token prompt makes the arena look inert, `TODO` #296),
  1 x R9700, 35B-A3B Q8_0, two interleaved rounds, **text identical in every arm** (`504 a2cdef23588f`):
  `-ncmoe 20` -- delivery arena **1441.6 / 72.0** t/s (pp/tg, 69.9 % residency, 87.9 % decode hit rate),
  arena disabled 757.1 / 48.7, delivery arena + `--moe-cache-mib 4096` 808.4 / 63.9, upstream ClassLRU 4096
  782.3 / 53.3, upstream + `--moe-cache-cpu-share` 0.5 **/** 0.95 778.3 / 55.7 **/** 778.4 / 58.3, upstream
  no cache 747.5 / 42.5; `-ncmoe 40` -- ours **1394.2 / 69.2** against upstream's cache 455.8 / 32.6 and no
  cache 431.3 / 30.8.  So at equal MiB the arena is +84 % prefill / +35 % decode (+206 % / +112 % at
  `-ncmoe` 40); **coexistence is a LOSS** (44 % prefill, 11 % decode) because the ClassLRU cache takes the
  VRAM the arena sizes itself against -- so "keep both" means either/or, which is already the shipped state
  (`--moe-cache-mib` defaults to 0); and neither ClassLRU nor `cpu-share` is adopted (0.95 share reaches
  58.3 t/s against the arena's 72.0).  Upstream's cache stays shipped and dormant: it is the only device tier
  on a `GGML_HIP_NO_VMM=ON` platform, where the arena declines.  **`D1`'s original inversion (upstream
  +17.6 % decode) is stale and partly a `#296`-class artifact**: on today's tree ours measures 83.9 against
  upstream's 48.0 on the `llama-bench` arm (was 53.5 against 62.9), and the arena's *prefill* is
  regime-dependent (838 t/s on random tokens against 1442 on real text, same arm).
  `benchmarks/2026-11-08-moe-cache-arena-vs-classlru.md`.

## D2 — Fused MoE gate/up+GLU in MMQ (and top-k MoE)

**Upstream:** `#31188` fuses `MUL_MAT_ID` gate/up + GLU in MMQ, `#31288` adds a top-k MoE fusion for 384
experts, `#31296` fuses NVFP4 gate/up + GLU, `#31047` linearises the MoE data layout for MMQ/MMF, `#31005`
sanitises post-bias NaNs in MoE selection.
**Delivery:** block 08's fused gate+up+GLU on the RDNA path (block 08 also owns the per-shape `mmvq` `nwarps`
and the conv/RMS/top-k fusions).

**Relation:** expected **arch-disjoint** — the same precedent as `#29948` (the 2026-10-25 re-base's NVIDIA
fused FFN kernel, which duplicated block 08's path and both shipped on one dispatch).  No delivery code drop
expected.

**What to prove at the rung:** every arm's predicate (which arch/quant/cap selects it), that the RDNA arm
still fires after the merge (`-lv 4` + a shape that reaches it), and that no NVIDIA-only path becomes
reachable on gfx1201.  `#31047`'s layout change is the real risk to our `mmid`/`mmq` hunks — re-derive, do
not patch around it.

- status: **DECIDED (R09, 2026-11-08) — complementary, no delivery kernel dropped.**  Upstream's `has_fusion`
  (interleaved gate/up+GLU in MMQ, `#31188`/`#31288`/`#31296`) and the delivery's `has_gate` (RDNA fused gate)
  are two template arms on one `mul_mat_q_process_tile` with separate selection predicates, exactly the
  `#29948` pattern from the previous re-base.  The one structural need was `mmq_args` keeping per-expert
  `expert_bounds` next to upstream's `tile_spec` (the routed-compact dispatch keys on bounds; one compaction
  emits both).  Evidence: build + `MUL_MAT_ID` 931/931 + fused/chain byte-identical text.  Rung log: `RUNGS.md`
  R09.

## D3 — qwen4exp, the indexer and the sparse selection

**Upstream:** `#31443` (write the conv-state tail without an intermediate `cont`), `#31243` (**build the
k-pool sparse selection independently of the `kq_mask` shape**), `#31009` (glm5-next: QSA without `kq_mask`
+ DSA), `#29050` (MFMA lightning indexer for CDNA2), `#31803` (indexer shared-memory bank conflicts),
`#31289` (hexagon indexer).
**Delivery:** block 11 (indexer cache + residency, fused scorer with quantized indexer keys, HC fusions, the
lazy reader, QSA), block 15's sparse decode arm, block 05's derived kq mask.

**Relation:** `#31243`/`#31009` attack the same structural problem our **derived kq mask** and the pooled
sparse selection solved ("the mask shape must not decide what the sparse selection can build").  This is the
highest-probability *subsumption* in the whole re-base.

**Questions:** does upstream's selection make our derived mask redundant (drop), or does it only remove the
blocker (keep both, ours now the cheaper path)?  Does our fused scorer still own the quantized-key case that
upstream leaves on the dequant path?

**Measurement:** on the rebased tree, qwen4exp on the delivery's config: prefill `t/s`, `tg128`, the KLD
gate, and the indexer/sparse census lines, across (a) delivery fusions on, (b) `GGML_CUDA_DISABLE_FUSION=1`
(the chain), (c) the derived-mask kill switch, (d) the sparse arm off.  Judge the fused arm against the
chain, never against the default (`MANIFESTS.md` gate 3, `GREEDY-PURITY.md` §81).

- status: **DECIDED (R12, 2026-11-08) — keep the delivery's implementation, and it was measured,
  not assumed.**  Upstream turns out to carry its own qwen4exp (1514 lines, `#30965`/`#31243`/`#31443`/
  `#30927`/`#31086`) with plain-op HC mix/combine, a mask-based `build_qsa_sel` and a plain-op indexer;
  the delivery's (3197 lines) has fused HC ops, the mask-free store+top-k selection, the indexer
  ops and a keys-only indexer cache.  Same GGUFs, same build, `-c 8192`, 5256-token prompt:
  Flash-Next GSQ-IQ3_XXS 3 GPU `-sm tensor` — upstream **1399.7 / 53.5** t/s (pp/tg), delivery
  **1432.1 / 70.1**, **text byte-identical** (`7a8734a9f1a6`) → the delivery's wins by **+31 % decode**
  with no numeric difference at all; Qwen3.8-27B UD-Q4_K_M 1 GPU — upstream 1311.1 / 32.4, delivery
  1303.2 / 32.4 (neutral) and the greedy text diverges (`b905c9763a58` vs `b21e94a5b6e1`), which the
  R17 KLD gate must quantify (a chain difference, not a purity claim).  So: neither "subsumed" nor a
  preference — the delivery's implementation is the faster one exactly where this repo validates, and
  keeping it keeps r235's qwen4exp validation and both delivery gates meaningful.  Cost accepted: the
  3200-line model file stays ours while upstream keeps rewriting it.  Rung: `b1b0fa807`; log in
  `RUNGS.md` → R12.

  **Re-decided at r236 (2026-11-08, `RUNGS` `P11`) on depth and concurrency, not on a shallow single-stream arm —
  KEEP, and no merge session is warranted.**  Ours vs unpatched `a691b2345`, Flash-Next GSQ-RCO-IQ3_XXS, 3 devices
  `-sm tensor`, no `-ncmoe`, interleaved 3 rounds: prefill **+16.6 % / +28.8 % / +173.4 %** at d 0 / 16 384 /
  163 840; decode **+98.2 % / +98.9 % / +68.2 %** at the same depths; real-text prompt-eval **+19.1 %** at ~16 K
  and **+88.4 %** at ~160 K; aggregate concurrency **+41.6 % / +67.2 % / +53.7 %** at 1 / 2 / 4 slots.  There is no
  cell where upstream matches or beats us, so `#293`'s "keep only if it wins at depth" test passes at every depth,
  and the "mixed bag → merge" branch never opens.  The earlier 27B "text divergence" that motivated this item was
  an artefact: the 27B is `qwen35` (dense), so no `qwen4exp.cpp` swap can affect it, and its prefill-logit KLD
  against unpatched upstream is **0.000697 / 98.784 %** (rounding-level).  The qwen4exp-vs-upstream KLD is
  **0.013426 / 96.431 %** and is *not* the model file: it is the accumulated arithmetic order of our own MoE /
  attention / HC implementations (MoE alone costs 0.004980 on 35B-A3B), and it is insensitive to every switch we
  ship.  Dropping our model file would therefore leave the 0.0050 MoE term and lose the performance.
  `benchmarks/2026-11-08-qwen4exp-keep-or-drop.md`.

## D4 — The recurrent rollback bound: upstream enforces what we track and warn about

**Upstream:** `#31368` — the recurrent memory and the DSV4 cache record the deepest rollback the *last
ubatch* guarantees (`rs_max`: `n − 1` for a ubatch of `n ≤ n_rs_seq + 1` tokens, `0` for a longer one) and
`seq_rm` **refuses** anything past it.  New tests (`rollback_bound`) cover ubatches of 1, 2, `n_rs_seq + 1`
and `n_rs_seq + 2` tokens.
**Delivery:** block 03's **r91 fold** ("fold every ubatch into the per-sequence rollback tracker", issue
#131) tracks the per-sequence ubatch composition, and its guard **warns once** when a rollback crosses a
batch boundary (`LLAMA_RS_ROLLBACK_DEBUG=1` adds the per-ubatch instrumentation).  Record:
`archive/work/rs-rollback-boundary/`.

**Relation:** the same defect, two answers — ours diagnoses, upstream's refuses.  Upstream's is the stronger
contract; ours carries the diagnostic that produced the analysis.

**Question (needs a decision because it drops delivery code):** migrate to upstream's `rs_max` enforcement
and drop our duplicate bookkeeping?  Recommended shape: **take upstream's bound**, keep the
`LLAMA_RS_ROLLBACK_DEBUG` instrument if it is cheap on `update_rs_for_ubatches`, drop the warning path whose
job the refusal now does.  Verify with the rs-rollback-boundary scenario and `test-llama-context`'s rollback
cases, text identical.

- status: **DECIDED (R04, 2026-11-08) — upstream's `rs_max` bound taken, the tracker dropped** (the status line
  above was left OPEN by mistake; `RUNGS.md` R04 is the record).  R04: the block's duplicate bookkeeping went, the
  conv-state store upstream absorbed it, and `rs_max` is the bound; a pre-existing upstream defect was found and
  fixed on the way (`rs_max` not swapped on a reused slot) and filed as `TODO` #289.  The gate was
  `test-recurrent-state-depth` (3 phases, 0 failures) plus the GDN census, not the full rs-rollback scenario — if
  `#289`'s fix area is touched again, run `archive/work/rs-rollback-boundary/`'s scenario.

## D5 — The FA entry points after `flash_attn_ext_rows`

**Upstream:** `#29510` `llama : flash_attn_ext_rows - more efficient unified kv for multi-seq`, `#31550`
(its VGPR spill list), `#31073` (rotation constants stay on the GPU), `#31683` (kv shifting, per-model
`build_rope`).
**Delivery:** block 05 (the RDNA band kernels and their dispatch, native quantized K/V staging, the derived
kq mask), block 10's FA staging ring.

**Relation:** `#29510` restructures how the unified KV is addressed, which is exactly what our band kernels'
dispatch and the mask fill assume.  This is a **re-derivation**, not a merge — the rung's cost centre.

**Question:** does the new shape make any of our band kernels unreachable (same cap, same head size) on the
configs we validate (gfx1201, gfx1151, gfx1100)?  Any unreachable kernel is a delivery-code drop.

**Measurement:** after the re-derivation: `gate-prefill-logits.sh` (with `--family-switch` if a family cap
moves, `--kv q8_0` for the staging arm) + prefill `t/s` per band on the 4B and 35B-A3B models.

- status: **DECIDED (R06, 2026-11-08) — no delivery kernel dropped.**  The entry-point restructure is absorbed
  (both parameters on every kernel), the band/tile/prefill kernels keep their dispatch, and the derived mask had
  to move its ABI to `src[6..8]` because upstream's `kv_rows` took `src[5]` (merged as-is, the derived probe
  silently failed and every batch fell back to the packed mask).  The remaining coverage question — does the
  band family still fire at head 256/512 — is the R17 `--family-switch` gate's, not a code decision.
  Rung log: `RUNGS.md` R06.

## D6 — `ggml : fix bf16`

**Upstream:** `#31609` (`ggml/src/ggml.c` + `tools/export-lora`) and `#31276` (AVX512 BF16 clang-cl).
**Delivery:** block 04 (the bf16 KV cache type + native-BF16 FA).
**Relation:** expected independent (a type-traits/`_flush` fix upstream, a new cache type and FA arm here) —
verify, do not assume: the block's bf16 arm must still generate and the FA arm must still be selected.

- status: **CLOSED (R05, 2026-11-08) — independent, nothing changed on either side** (the status line above was
  left OPEN by mistake; `RUNGS.md` R05 is the record).  R05 applied with **zero conflicts**, the bf16 KV/FA arm
  generated, and the FA arm was still selected — upstream's `#31609` type-traits fix and the delivery's bf16 cache
  type are disjoint.

## D7 — Compute views in the Meta backend

**Upstream:** `#31447` — a buffer with few tensors can still get many views, so the compute headroom gets a
minimum of **256** views (`max(16 × n_tensors, 256)`).
**Delivery:** block 00/09 — the hybrid recurrent models create `~2 × (n_rs_seq + 1)` conv-state snapshot views
per recurrent layer, which overflowed the original 16× container; the delivery's answer is the
`GGML_COMPUTE_BUFFER_MARGIN_PCT` compute-margin path (`BASELINE.md` → provenance item 2).

**Relation:** the same failure class on a different axis (view count vs buffer bytes).  Upstream's floor may
make our margin unnecessary for the model that motivated it, and our margin may cover cases the floor does
not.

**Question:** does the hybrid recurrent allocation succeed with the margin at `0` on the new base?  If yes,
is the margin narrowed (not removed — it is a documented invariant, and `INVARIANTS.md` must be corrected in
the same change if it changes)?

**Measurement:** a hybrid recurrent model (qwen3.5/3.6/3.8 hybrid or deepseek41) with
`GGML_COMPUTE_BUFFER_MARGIN_PCT=0` vs default, watching for the Meta container overflow / a failed graph
alloc, plus coherence text.

- status: **OPEN** · rung R01 · record: —

## D8 — Which CPU wins survive the CPU-backend rewrite

**Upstream:** `#31854` (**F32 src1 in the CPU backend**: `vec_dot_type(F16)` is now F32, new
`ggml_vec_dot_f16_f32`, conv patches in F32, `supports_op` changes), `#31179` (concat as one row kernel),
`#27472`/`#30585` (SVE GEMM/GEMV kernels), `#31276` (AVX512 BF16 clang-cl).
**Delivery:** block 12's CPU files — `ggml/src/ggml-cpu/arch-fallback.h`, `arch/x86/quants.c`,
`ggml-cpu.c`, `ggml-cpu.cpp`, `repack.cpp`.

**Relation:** `#31854` changes the F16 `type_traits_cpu` entry our hunk sits next to, and it is a *numerics*
change on every CPU matmul.

**Question:** which of our CPU wins remain after the rewrite (the host-expert CPU paths and any `repack`
gains), and does anything become dead code?  Drop only what measurably no longer wins.

**Measurement:** CPU-side `pp`/`tg` on the host-expert config, `test-backend-ops` for the touched quant
types, and the delivery's `-lv 4` CPU-path census.

- status: **DECIDED (R17 pre-pass, 2026-11-08) — measured; two dead hunks DROPPED, three kept, and the win is not
  in those five files at all.**  Numbers and arms: `benchmarks/2026-11-08-host-expert-prefill-transport.md`,
  `WORKLOG.md` 261108.011, `RUNGS.md` `P9`.
  * `arch-fallback.h` — the delivery's only change was a **one-line `// quants.c` comment**.  Dropped; the file is
    byte-identical to upstream's again (blob `4e59d0ae0`).
  * `ggml-cpu.cpp` — the delivery marked the `getenv("GGML_CPU_MOE_OFFLOAD_THREADS")` result `static`, inside
    `ggml_backend_cpu_moe_offload_cap()`, which returns early from its own `static int cap` memoisation, so the
    `static` can never run twice.  **A literal no-op**; dropped.
  * `repack.cpp` — three `nullptr` entries for the new buffer-type fields.  No behavioural effect (an aggregate
    initialiser zero-fills), but it is exactly what silences `-Wmissing-field-initializers`: compiled without them
    (same TU flags, `-Wextra`) emits that warning on `repack.cpp:5254`.  **Kept** (the tree is warning-clean by
    standard).
  * `ggml-cpu.c` — `GGML_OP_ALL_GATHER` → `dup` + its `ggml_get_n_tasks` entry.  Not a perf win: it is the
    cross-backend consistency fill that lets a CPU split carry the delivery's all-gather node (`AGENTS.md`, scope
    policy).  **Kept**.
  * `arch/x86/quants.c` — the AVX2 Q2_0 dot, whose *purpose* is purity (bit-identical to the generic body).
    Functional: `test-backend-ops -b CPU -o MUL_MAT -p type_a=q2_0` **85/85**.  The price of purity is now
    measured: delivery **0.66-0.69 TFLOPS** against upstream `#30447`'s body **1.23-1.34 TFLOPS** on the same
    shapes (n=1: 177.74 µs vs 95.10 µs) — i.e. we carry a ~46 % slower dot than upstream on a type no delivery
    model uses.  **Kept**, recorded; a purity-preserving fast body is a possible future item, not a D8 blocker.
  * The one *measured CPU win* in block 12 is not one of those five files: the AVX2 top-k block prefilter in
    `common/sampling.cpp` — **+0.54 %** decode (103.93 vs 103.37 t/s, 3/3 rounds, 4B Q8_0, `--temp 0.8 --top-k 40`)
    with **byte-identical text** (`7f950c7d0f76`), which also proves it is a filter and not an approximation.
    **Kept.**
  * The "host-expert CPU path" arm D8 asked for was measured and it changed the question: on **real text** the
    delivery is **+181 %** prefill over unpatched `a691b2345` (1064 vs 379 t/s, 35B-A3B Q8_0, `-ncmoe 48`), while
    on `llama-bench`'s random-token prompt it is −18 % — an artifact of that prompt (the arena is inert there:
    0.09 % across a 22 GiB arena sweep).  The residual no-locality floor is `TODO` #273's, recorded there.

- record from R13 (superseded by the above, kept because it states the premise):
  *R13 (block 12, 2026-11-09): block 12's CPU hunks (`arch-fallback.h`, `arch/x86/quants.c`, `ggml-cpu.c`,
  `ggml-cpu.cpp`, `repack.cpp`) merged without conflict and `#31854`'s F32-src1 `type_traits_cpu` entry did not
  collide with ours, so **nothing was dropped** — the wins are still in the tree by construction.  The
  discriminating measurement (`pp`/`tg` on a host-expert config, `test-backend-ops` for the touched quant
  types, the `-lv 4` CPU-path census) was **not** taken; it is owed as an R17 release-ladder arm (see
  `TODO.md`).*


## D9 — Graph cache, views and the allocator

**Upstream:** `#30662` (prevent premature eviction of graphs about to be used), `#31584` (bound views by the
bytes their strides reach), `#31146` (relax scale to non-contiguous views), `#30881` (batched tensor copy
API), `#31447`, `#31141` (`no_alloc` out of `llama_hparams`), `#31200` (remove obsolete `n_pad`).
**Delivery:** block 00 (alloc view release, the compute margin), block 10 (the graph-cache key/replay guard,
the async input ring), block 15 (the memory instances).

**Relation:** no functional overlap, but these are the files where a *silent* behaviour change hides.  The
delivery's replay guard and alloc-deps are correctness-critical (`GREEDY-PURITY.md` §20-class).

**Measurement:** `gate-park-idle-slots.sh`, `gate-ckpt-reuse.sh`, `gate-state-pipe.sh`, and coherence on a
hybrid model; for `#31584`, a `test-backend-ops` pass (view-bounds regressions show as aborts, not as text).

- status: **OPEN** — no `D9` conflict was reached at R16 (the graph cache, the replay guard and the alloc-dep
  registration merged as both sides; `memory_breakdown()`'s re-derivation is accounting, not allocator behaviour).
  Residue: R13/R16 are done; the graph-cache arms (`gate-park-idle-slots.sh` / `gate-ckpt-reuse.sh` /
  `gate-state-pipe.sh` / a `test-backend-ops` pass for `#31584`'s view bounds) belong to R17's ladder.  record: —
  *R11 (block 10, 2026-11-08): no D9 conflict was reached — the graph-cache key, the replay guard and the
  alloc-dep registration merged as both sides, and nothing in the rung's three defects was a graph/allocator
  issue (they were launch-geometry and dispatcher-merge defects).  R11 did produce the dispatcher reconciliation
  (one dispatcher; `TODO` #290 closed) and `GREEDY-PURITY.md` §84.*

## D10 — All-reduce

**Upstream:** `#28967` (NCCL for multi-node tensor parallelism), `#25051` (vulkan tensor-parallel allreduce),
`#30956` (sycl).
**Delivery:** block 01 (the hybrid HIP all-reduce for small tensors + the per-size dispatch vs RCCL).

**Relation:** different backend and different layer — expected to coexist; verify the dispatch order, that
`GGML_CUDA_ALLREDUCE=hybrid` is still reachable, and that the RCCL path still wins where it should.

**Measurement:** `gate-prefill-logits.sh --sm tensor --device 0,1,2` (mandatory for an AR arm, r165) with
`--ab "GGML_CUDA_ALLREDUCE=hybrid"`, plus the `-lv 4` `all-reduce: … RCCL ring` census.

- status: **OPEN** · rung R02 · record: —

## D11 — Speculative decoding: hidden state, conv snapshots, checkpoints

**Upstream:** `#30141` (save and restore MTP hidden state), `#31660` (store the MTP conv snapshots of every
recurrent model in one `set_rows`), `#31350` (restore the spec checkpoint for drafts longer than `n_rs_seq`),
`#30779` (visit parent slots before child slots), `#31497` (no MoE cache on the draft model), `#31569` (fit:
skipped NextN layers).
**Delivery:** block 02 (adaptive draft depth, `spec_draft_align_cache()`), block 13 (draft vocab subset),
block 12's DFlash ingest tail + draft-reserve split (issues #239/#240/#272).

**Relation:** `#30141`/`#31660` change the hidden-state/conv-snapshot contract the adaptive controller and the
DFlash realign both ride on — this is the one place where an upstream fix can silently invalidate a delivery
invariant rather than conflict textually.

**Measurement:** MTP four-axis (`benchmarks/mtp-adaptive-methodology.md`) after the rung that merges
`#30141`/`#31660`; `llama-batched-bench -npl 1,4,8`.

- status: **OPEN** · rung R03 (residue R13/R14; R16 added block 15 Goal C) · record: `RUNGS.md` R16/R17
  *R16 (block 15 Goal C, the MTP draft KV window): the window is gated per sequence and `gate-spec-draft-window.sh`
  PASS — `LLAMA_SPEC_DRAFT_WINDOW=1024` sizes `1024 cells per sequence, cache 3072 cells ((win + 512) x 2)` and all
  six requests across both slots are byte-identical to `win 0` with identical acceptance.  The hidden-state /
  conv-snapshot contract `#30141`/`#31660` touches is `R03`'s business and is unchanged here; the **MTP four-axis
  protocol is still owed at R17**, which is the arm that can see a hidden-state defect.*

## D12 — Quantized-KV state, rotation and `kv_rows`

**Upstream:** `#30541` (on-device state copy sizes for quantized KV caches), `#31073` (rotation constants on
the GPU), `#31570` (never return 0 `kv_rows` for sequences without cells), `#30471` (opencl KV views on
state restore), `#31368` (rollbacks).
**Delivery:** block 15 (the streaming provider keeps one contiguous VA range; `set_rows`, the rope shift,
`state_write/read`, the kq mask and the FA kernels all address it), block 12 (batched state restore).

**Relation:** every one of these touches the exact API surface the streaming provider must preserve
("**no kernel or op change**" is block 15's core claim).  If upstream's state copy stops being a plain
buffer-to-buffer copy, the provider's region binding is affected.

**Measurement:** `gate-kv-residency.sh` (window + pool + the multi-device `-sm layer` section),
`gate-state-pipe.sh`, `gate-kv-window-split.sh`, and the `-ctk/-ctv q8_0` arms.

- status: **DECIDED (R16, 2026-11-08) — no re-derivation needed beyond the buffer type** · record: `RUNGS.md` R16
  *R16 (block 15): `#31073`'s rotation constants land as a per-cache, per-layer buffer (`attn_rot_bufs`), not a
  cell-indexed tensor, so they are outside the region the provider binds at — they are summed into
  `memory_breakdown()` and nothing else.  `#30541`'s on-device state copy keeps addressing the cache through the
  same buffers, so `state_write`/`state_read` keep going through the region buffer's own copy (block 15's r204
  path) and need no re-derivation.  `#31570` `kv_rows` is in the memory interface (`get_kv_rows_slices`), which
  merged automatically.  Proven by measurement, not by reading: the residency gate's spilling arms really spill
  (16 mapping flips, 64 MiB host-backed, `wref == wres == wspill` `be904e86f248`) and its `win2mtp` arm takes the
  region path for the batched state save (`gate-kv-window-split.sh`, acceptance `0.46154` both ways).  The one
  thing this rung did re-derive was **not** a state shape: the fused `GGML_OP_INDEXER_SCORE` kernel called
  `ggml_rope_get_truncate(dst)` after `#30206` gave that accessor an op-type assert (a latent R12 defect, every
  sparse QSA decode run aborted since then — `GREEDY-PURITY.md` §85, `TODO` #297).*

## D13 — Memory instances after `no_alloc` and `n_pad`

**Upstream:** `#31141` (`no_alloc` out of `llama_hparams`), `#31200` (remove obsolete `n_pad` from memory
instances), `#31142` (model output structs), `#31733` (`model` as a class member of `llm_graph_context`).
**Delivery:** block 11's `llama_memory_hybrid_idx`, block 15's region provider, every model file our blocks
touch.  Pure refactor churn with no decision attached — resolve mechanically, but **read `#31733` first**:
33 of block 12's 81 files are affected by it and resolving against the new member shape from the start
avoids redoing hunks.

- status: **DECIDED (R16, 2026-11-08) — resolved mechanically, plus two defects it exposed** · record: `RUNGS.md`
  R16
  *R16 (block 15): `hparams.no_alloc` → the ctor's `no_alloc` at all three block-15 sites; the pool's cell count
  rounds to whole PAGES now that `n_pad` is gone; `memory_breakdown()` is re-derived over `ctxs_bufs.bufs` +
  `ctxs_bufs.no_alloc` with upstream's `attn_rot_bufs` summed on top; the delivery-side `params_l` rename is
  dropped for upstream's `params`.  Two defects the mechanical merge could not see: (a) `src/llama-memory-hybrid.cpp`
  passed `llama_kv_cache`'s extended tail positionally, so the newly-inserted `is_indexer` shifted `v_enabled` into
  receiving `kv_pool` — no V tensor, `cpy_v` null-deref, **every hybrid/GDN model segfaulted at load**; (b)
  `ggml/src/ggml-cuda/indexer-score.cu` used a ROPE-only accessor on its fused node (see `D12`).  Both fixed, both
  recorded as `GREEDY-PURITY.md` §85 with the rule "a positional call into an extended parameter list must be
  named, and every call site of the symbol audited".*

## D15 — The delivery's F1 layer-input device store vs upstream's `embd_row` surface

**Upstream:** `llama_set_embeddings_layer_inp_dev()` / `llama_supports_embd_row()` /
`llama_batch_ext_add_embd_row()` plus `#31506`: the target writes its extracted layer inputs into one
persistent device buffer (`llm_graph_context::build_layer_inp_dev()`, one `SET_ROWS` per layer, indexed by the
batch's row ids) and the drafter reads rows of it (`ggml_get_rows`), with a per-batch event so the target does
not overwrite a row the draft is still reading.
**Delivery (F1, r163):** `llama_lf_*` + `cparams.lf_layer_inp_dev_store[]` + `llm_graph_result::set_layer_inp()`
emitting the per-layer `set_rows` store in-graph, plus `lf_row_offset` / `lf_dev_rows` / `lf_dev_inject` and a
token-batch injection whose token ids are device rows.

**Relation:** the same feature, independently built — upstream's is the successor.

**Decision (R13, 2026-11-09): upstream's, F1 deleted.**  Measured after the deletion, on the re-based tree:
* 27B UD-Q4_K_XL + DFlash2-Q4_K_M, 1 GPU: acceptance **0.754** (43/57, mean len 3.26), device and host
  transports bit-identical (same accepted/generated counts).
* 35B-A3B Q8_0 + DFlash2, 3 GPU `-sm tensor`: 0.741 (3.15); 3 GPU `-sm layer`: 0.639/0.629 with 2 slots.
* Flash-Next GSQ + DFlash2: 0.405 (2.21) on `-sm tensor`, 0.384 (2.14) on `-sm layer`.
* MTP control on the same Flash-Next arm: 0.493 (2.42).

**Two things had to be carried/added on upstream's path:**
1. **The Meta refusal.**  Upstream allocates the layer-input buffer on `model.dev_output()`, which under
   `-sm tensor` is the Meta device; the draft's `ggml_get_rows` over a Meta buffer GPU-faults
   (`k_get_rows_float_vec`, 3x gfx1201, first injection).  `layer_inp_dev_alloc()` keeps the host-copies path
   when the output device is Meta — the same refusal F1 encoded.  Harness note: `LLAMA_DFLASH_DEV` has no
   device value left to test there; the only lever is the layout.
2. **The host-experts guard, re-measured and dropped.**  F1 forced the host path when the target had
   host-resident MoE experts (`-ncmoe`) and `n_seq > 1` (r163: the armed slab was measured to break the
   draft's device reads there).  Re-measured on the re-based slab with two-slot servers (`-sm layer`, 3 GPU,
   guard bypassed vs guard active): Flash-Next GSQ `-ncmoe 24` **0.53793/0.51678 both ways** (identical
   accepted/generated and mean len, same wall time), `-ncmoe 10` **0.55245/0.47134 both ways**; 35B-A3B
   `-ncmoe 10` **0.63918/0.62887 both ways**.  The premise does not reproduce, so the guard is gone;
   `LLAMA_DFLASH_DEV=0` remains the lever that forces host copies anywhere, and the 100-128 MiB device buffer
   that the guard avoided has a graceful host fallback if it does not fit.

**Record:** `RUNGS.md` R13; `INVARIANTS.md` (the Meta-buffer, unregistered-input and per-build-oracle
entries); the rung commit's body.
