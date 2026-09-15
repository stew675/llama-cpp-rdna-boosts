# Issue #30 follow-up — widening the delivery across configuration scenarios

**Status:** ACTIVE exploration / development dossier (opened 2026-09-14).  **Not** delivery work — nothing
here is folded into `patches/` until it passes the promotion rule (a validated win, an env kill-switch,
the combination re-validated, then a delivery block with the maintainer's go-ahead).  Companion to the
historical tools already in `tools/` (the August/September issue-#30 harnesses).

Issue: <https://github.com/stew675/llama-cpp-rdna-boosts/issues/30>
Dossier anchor: the reporter's reconciliation comment
<https://github.com/stew675/llama-cpp-rdna-boosts/issues/30#issuecomment-5658394266> (2026-09-14).

---

## 0. Why this dossier exists

The issue was filed as *"15-patch delivery costs ~14 % MTP decode vs stock at the same fork point"*.  The
2026-09-14 comment **retracts the headline claim** (the early numbers were a stale snapshot, a depth
outside the purity band, and a mis-described workload) and replaces it with a clean, pinned,
independent reproduction that agrees with our own numbers.  What remains is **two findings plus a
disclosure that the reporter's build was never stock**:

1. The reporter's "stock" arm (arm **P**) is upstream master **plus a one-line local fix** raising the
   RDNA4 WMMA head-256 batch threshold (filed upstream as ggml-org/llama.cpp#28867).  Our delivery has
   its own answer to the same problem (a `Q->ne[1] > 8` purity guard), so the two are not the same arm.
2. On a **q8_0 KV cache the delivery is ~11 % slower than arm P at decode**, while arm P is
   KV-type-insensitive.  This is not a context-length effect (32k and 196k agree), and it is a
   *configuration* effect, so long-context/quantized-KV users — the majority of constrained-memory
   deployments — are exactly the users who do not get the delivery's gains.
3. **Adaptive MTP at ceiling 12 fails to load at a large context with a q8_0 cache**
   (`exiting due to model loading error`), so the delivery's single biggest decode win is unavailable
   in the reporter's actual deployment.
4. **Deep prefill inverts on q8_0**: pp4096 is +13.5 % for the delivery, but at 150k it is ~8 % *behind*
   arm P.  `llama-bench -p 4096` cannot see this (the same blind spot as `tg128`/`pp512`).

The delivery's own reference predicate has always been a **BF16 KV cache on a system with good VRAM
availability** — enough VRAM (discrete or the gfx1151 128 GiB unified-memory part) to hold a BF16 cache
at a deep context together with a high adaptive-MTP draft depth.  (The repo *carries* multi-GPU work —
the HIP-tuned hybrid all-reduce, the split-aware gates — but multi-GPU is not the target predicate; the
predicate is memory headroom.)  Block 03 is literally "BF16 KV cache and native BF16 flash-attn".  The
reporter is telling us what happens outside that predicate: quantized KV and tight VRAM.

**Framing rule for this dossier.**  Everything the reporter says is *evidence*, not a claim to adopt.
The delivery's gate list (GREEDY-PURITY invariants, the `benchmarks/mtp-adaptive-methodology.md` gate,
the width/text/oracle instruments) applies to every change made here.  A configuration win that breaks
band purity is not a win.

---

## 1. What the 2026-09-14 comment established (facts to carry forward)

### 1.1 Arm identities

| arm | what it is | why it matters |
|---|---|---|
| **A** | true stock `790cf51aa` | the only legitimate stock baseline for stock-vs-patched ratios |
| **B** | the delivery, pinned by `rdna-boosts-all.patch` sha256 + applied tree `a5683e1b008e` | our arm |
| **P** | upstream master `5f436dddb` + one line (the head-256 WMMA threshold) | **not stock**; the reporter's production arm |

The reporter built all three from one Dockerfile (same compiler/ROCm), so arm-to-arm differences are
source only.  His acceptance numbers are identical to ours to five decimals (`0.79948` / `0.79379`),
which proves the workload matches.

### 1.2 The reconciliation (accepted)

Under the delivery's own protocol (prose prompt, `--reasoning off`, `-n 3000`, `--seed 42 --temp 0`,
f16 KV, 1 GPU) the delivery matches or beats stock: level at `n3`, ahead at `n7` and at pp4096, and
byte-pure where stock is not.  The reporter's original −14/−19 % was:
`~11 %` q8_0-KV (config) `+ ~9 %` workload (256-token completion with reasoning at the template default
vs 3000-token prose with reasoning off), and **both factors cost the delivery and neither costs arm P**.

### 1.3 The new, substantive findings (the work)

* **F-q8 (config):** `q8_0` KV costs the delivery ~11 % decode vs its own f16 (42.15 vs 46.70); arm P is
  flat (52.18 vs 51.52).  Insensitive to context (32k ≈ 196k).  The reporter points at
  ggml-org/llama.cpp#27796 ("quantized KV cache decodes slower than f16 on RDNA4, worse the more
  unpacking the type needs", suspected `nthreads_KQ_q` in `fattn-vec.cuh`).
* **F-buf (load failure):** `--spec-type draft-mtp-adaptive --spec-draft-n-max 12` at
  `-ctk q8_0 -ctv q8_0 -c 196608` fails to load; f16/32768 loads.  The reporter's hypothesis:
  *"the ceiling-12 draft KV is sized without accounting for the target's KV footprint"*.  The maintainer's
  own note: the MTP algorithm allocates buffers for drafting, verification and rewinding, and we never
  examined how much of it is avoidable.
* **F-pp (deep prefill):** on q8_0, 27k +9 %, 64k +1 %, **150k −8.4 %** vs arm P.  (Note: arm P has the
  #28867 threshold fix, which is prefill-neutral, so this should reproduce against true stock A too — to
  be confirmed.)
* **F-wmma (upstream asset):** the reporter's one-line fix raises the gfx1201 WMMA head-256 dispatch
  threshold from 16 to 64 (matching the MFMA branch's own `> 64`), recovering ~20 % at the verify widths
  with zero prefill cost and bit-identical greedy output on his prompts.

---

## 2. Upstream assets to exploit

| id | state | what it is | how we use it |
|---|---|---|---|
| ggml-org/llama.cpp#28102 | **merged 2026-09-11**, in our base `790cf51aa` | gfx1201 FA tuning: head-256 admitted to the RDNA4 WMMA dispatch with `Q->ne[1]*gqa_ratio_eff > 16`, head>128 prefill configs | already in the delivery (re-based, block 04 keeps its own head-256 configs) |
| ggml-org/llama.cpp#28867 | open (issue) | the head-256 WMMA threshold is ~4x too low; **raise to 64**; ~20 % verify-width decode, zero prefill, bit-identical | Action E — but our delivery already has a `Q->ne[1] > 8` purity guard, so the correct fix is to *generalise the guard*, not blindly copy 64 |
| ggml-org/llama.cpp#27796 | open (issue) | quantized KV decodes slower than f16 on gfx1201, worse the more unpacking; deficit widens with depth; suspected `nthreads_KQ_q = 2` flat tuning | Action B — plus our own `V4` native q8_0 tile staging (block 15, opt-in) as a candidate |
| ggml-org/llama.cpp#28529 | closed | proposed `GGML_CUDA_FA_WMMA_256_OFF` escape hatch (removed by #28102) | our block 04 already ships `GGML_CUDA_FA_WMMA_256` / `GGML_CUDA_FA_WMMA_MAX_HEAD` |

---

## 3. The delivery's starting position (facts from the tree)

These are load-bearing, so record them here rather than re-deriving:

* **The decode/verify band is deliberately on the tile kernel.**  `ggml_cuda_get_best_fattn_kernel()`
  requires `Q->ne[1] > 8` before it will select WMMA, and block 08 (§14) deleted the VEC fallback so the
  whole `n_q <= 8` band is TILE.  A W=1 decode and a W<=8 verify therefore take the same family — this is
  the `n_max <= 7` purity guarantee, and Cause B in `GREEDY-PURITY.md` §11.  **Removing or widening that
  guard is a purity change, not a tuning change.**
* **The reporter's #28867 fix and our guard overlap but are not equal.**  His `> (ne[0] <= 128 ? 8 : 64)`
  keeps the head-256 verify band off WMMA by *raising the gqa-eff threshold*; ours keeps it off by
  *requiring `n_q > 8`*.  At `n_max 8` (`n_q = 9`) our guard lets the 9-token verify back onto WMMA,
  which is exactly the regime #28867 complains about; that depth is outside the purity band anyway
  (`GREEDY-PURITY.md` §11/§32), so the two fixes can be combined safely.
* **The delivery's quantized-KV decode is TILE with f16 staging**; upstream master takes VEC for
  `n_q <= 2` with a quantized K/V.  The block-08 deletion moved only `n_q = 1,2` (per §14) — which is
  precisely the q8_0/q4_0 decode width — so the VEC→TILE move is the prime suspect for F-q8.
* **V4 (native q8_0 K/V in the FA kernels) exists but is opt-in** (`GGML_CUDA_FA_KV_NATIVE=1`, default 0)
  because it costs ~1.7 % prefill; its own comment claims *"decode and verify are unaffected"*, so it is
  not automatically F-q8's fix — but it is the lever if the q8_0 cost turns out to be staging.
* **The MTP footprint is `n_rs_seq`/`n_rs_batch` + the draft context + the speculative batch.**  The
  recurrent snapshot set is widened to `(1 + n_rs_seq)` groups (`llama-memory-recurrent.h`) and
  `n_rs_batch = common_speculative_n_max() + 1`; the adaptive controller chooses `n_max` at runtime but
  the buffers are sized at load, which is the shape of a load-time OOM when the ceiling is 12 and the
  target KV already fills the card.

---

## 4. Action register

Each action has a hypothesis, a method, an acceptance gate, and a status.  Actions are ordered by
value-to-effort.  Nothing here is "land it if it's faster" — every perf change must clear §5.

### A. KV-type × context-depth scaling audit (the user's explicit ask)

**Question.**  Does the delivery's **BF16/f16** KV cache fall off with depth the way the reporter
reports for quantized caches, and how big is the delivery-vs-arm-P gap per type across depth?

**Hypothesis.**  The delivery's f16/bf16 slope matches stock and arm P; only the quantized types diverge
(F-q8), and the divergence is present at every depth, not growing.

**Result (2026-09-14, `MEASUREMENTS.md` §A):**  confirmed, with the comparison drawn the right way —
**delivery bf16 vs stock f16** (the delivery's intended cache type vs the stock one it replaces): the
slopes match (83.5 % vs 84.3 % at d65k) and the delivery is ahead at every depth, so there is no BF16
depth fall-off.  The delivery-specific regression is the **quantized** types: q8_0 retains 66.1 % vs
stock's 80.1 % and ends 15.6 % behind at d65k, q4_0 the same shape.  Prime suspect is block 08's F1
VEC→TILE move combined with the tile kernel's whole-cache f16 staging pass (grows with `n_kv`), i.e. the
same mechanism as F-q8 — so Action A folds into Action B.

**Method.**  `llama-bench -m Qwen3.8-27B-UD-Q4_K_XL -ngl 99 -p 0 -n 64 -r 2 -d 0,16384,32768,65536`
for `ctk=ctv ∈ {f16, bf16, q8_0, q4_0}`, on arms **A (stock 790)**, **B (delivery)** and, where useful,
**P** (stock + #28867).  Decode slope = `tg64(d)/tg64(0)`.  Record `pp512 @ depth` as the secondary axis.
One GPU (`HIP_VISIBLE_DEVICES=0`), ROCm 7.14.  The reporter's own depth claim came from #27796's
downstream model, so this is the first same-model, same-box measurement.

**Gate.**  No gate — this action produces the reference table every other action keys off.  It must
include the f16 control (a depth fall-off on f16 would be a *different*, higher-priority bug).

**Status: DONE** (`MEASUREMENTS.md` §A).  BF16/f16 are clean; the finding is handed to Action B.

### B. Quantized-KV decode/prefill on RDNA4 (F-q8 + F-pp) — **root-caused, fix prototyped**

**Answer.**  The penalty is neither the VEC→TILE reduction order nor `#27796`'s `nthreads_KQ_q`: it is
the **whole-cache F16 staging pass** the tile kernel needs for a quantized cache.  That pass is
proportional to `n_kv` and runs every decode step, so the cost grows with depth — exactly the reported
shape.  The block-15 **V4** native-staging arm removes it, and the fix is an activation-policy change
plus a new **q4_0** native arm (q8_0 only had one).

**Measured (1 GPU; `MEASUREMENTS.md` §B; experiment diff
`patches/2026-09-14-v4-default-plus-q4_0-native.diff`):**

| KV | staging d65k | native d65k | stock d65k | prefill cost |
|---|---|---|---|---|
| q8_0 | 18.92 (66.1 % ret.) | **23.29 (80.7 %)** | 22.43 (80.1 %) | −1.2 % |
| q4_0 | 19.72 (68.9 %) | **22.82 (79.2 %)** | 21.03 (75.9 %) | ~−1.3 % |

Bit-identity holds: q8_0 native == staging == `ab94eb7db4d4`; q4_0 native == staging == `edafcdc7f8df`;
`W=1..8` is one logits hash for every supported type (`f16 q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl`).

**Policy refinement (the maintainer's 2026-09-14 decision).**  `GGML_CUDA_FA_KV_NATIVE` is now a
three-state policy: **unset = auto** (native q8_0/q4_0 **on** — they are sub-F16 quants whose F16 staging
is the cost — and native bf16 **off**), `=1` forces all on, `=0` forces the pre-amendment F16-staging
path (the opt-in escape hatch).

**Audit of the other enabled quant types.**  `q4_1`/`q5_0`/`q5_1`/`iq4_nl` have **no native arm** and
still stage through F16, but they are **well supported**: their retention matches stock (q4_1 81.1 % vs
81.1 %, q5_0 78.8 % vs 77.9 %, q5_1 78.9 % vs 78.1 %) and the delivery is ahead in absolute terms at
d32k.  They sit ~12-16 % behind f16, so a native arm is a worthwhile follow-up, but the severe fall-off
was specific to the q8_0/q4_0 staging kernels.  `iq4_nl` **cannot be benched on a stock build** (no FA
enablement → host-only/CPU; the run was killed after >10 min).

**Remaining / follow-ups:**  (a) native arms for `q4_1`/`q5_0`/`q5_1`/`iq4_nl` (per-type dequant that
matches each `ggml_get_to_fp16_cuda` conversion bit-for-bit); (b) run
`test-backend-ops -o FLASH_ATTN_EXT` with the q4_0 cache before promotion; (c) the deep-prefill (Action
D) half of F-pp is not yet reproduced against true stock.

**Gate.**  `W=1..8` one hash on the affected type; `plain == draft-mtp` text on f16 *and* the quant;
MTP acceptance at pos 1 > ~0.45 and `draft-mtp >= plain` at `n_max 3`.  **Status: met for q8_0/q4_0** on
the experiment build; the delivery patch has **not** been cut yet (the experiment lives in the fork
working tree and in the dossier diff, pending a block-15 amendment decision).


### C. MTP / adaptive draft-buffer footprint at high context (F-buf)

**Question.**  What exactly is sized for the ceiling-12 draft, and how much of it can be reduced or
made lazy so adaptive MTP loads at a large context?

**Confirmed root cause (2026-09-14).**  Reproduced on 1 GPU; see `MEASUREMENTS.md` §C.  The failure is
not the draft KV.  The target context's **recurrent-state snapshot set** is the dominant buffer:
`llama_memory_recurrent` allocates `n_seq_max * (1 + n_rs_seq)` full GDN-state copies, and
`n_rs_seq = draft.n_max` (`common_params_speculative::need_n_rs_seq()`), so the adaptive ceiling of 12
means **13 planes x 4 server slots = 7781 MiB** (R 292.5 + S 7488 MiB) at any context length.  The
draft context then needs a further 260 MiB of compute buffers that no longer fit, and load aborts.
Per-plane cost is 598.5 MiB; ceiling 12 costs +2992 MiB over ceiling 7 and +5387 MiB over ceiling 3.
`--parallel 1` shrinks it to 1945 MiB, which is the workaround; the delivery fix has to keep depth 12.

**Method / levers to test.**  (Root cause confirmed; now the reductions.)
1. **`--parallel 1` control** — proves the `n_seq_max` factor and is the user workaround (RS 7781 -> 1945
   MiB).  Measure the exact headroom it buys at 196k.
2. **Memory-aware effective ceiling** — the fit already detects the shortfall
   (`common_params_fit_impl: cannot meet free memory target ... need to reduce by 1608 MiB`) but aborts
   when `-ngl 99` is set.  Lower the *effective* adaptive ceiling / `n_rs_seq` to what fits, warn, and
   let the controller respect the effective ceiling.  Lowest-risk, but it caps the depth on small cards
   — which is the very thing the maintainer wants to avoid, so it is a fallback, not the goal.
3. **Structural RS reduction** (the real goal): candidates, each needing the §C gate because rollback
   fidelity is a purity property:
   * *lazy per-cell snapshot planes* — allocate a cell's `(1+n_rs_seq)` planes only when that cell enters
     speculative decode, instead of `n_seq_max` sets up front;
   * *shared snapshot pool* — `n_rs_seq` planes reused by whichever sequence is currently verifying
     (only valid if two sequences never verify at once; needs a guard);
   * *snapshot precision* — f32 -> bf16 halves S (7488 -> 3744 MiB) but makes rollback inexact, so it
     must be shown not to change the post-rollback state that decode reads (unlikely; measure to close);
   * *recompute-on-rollback* — keep only the pre-batch state and re-run the (cheap, attention-free) GDN
     scan over the accepted prefix on a partial accept; trades a small recompute for the whole snapshot
     set.  The largest payoff and the largest change.
4. **Measure** the memory delta and re-run the MTP gate (depth-12 adaptive is already outside the purity
   band per §11/§32, so the acceptance/throughput gate is the one that applies; state it).

**Gate.**  Loads at the reporter's config; same-seed output unchanged where the arithmetic is unchanged;
the four-axis adaptive gate (`benchmarks/mtp-adaptive-methodology.md`) shows no throughput regression;
memory is *less*, not merely moved.  If the lever touches the rollback path, the
`test-recurrent-state-depth` sweep (`n_rs_seq` 1..15, every rollback) must stay green and the
`W=1..8` width probe must not move.

**Status:** the load failure is **FIXED by Action B's V4 policy** (2026-09-14): the ~744 MiB/GPU F16
staging scratch the old build allocated was exactly the missing margin, so the reporter's config now
loads at the default `n_slots = 4` and generates (34.76 t/s at 196k, ceiling 12); `KV_NATIVE=0`
reproduces the failure.  The structural RS reduction remains open for extra headroom on smaller cards
(lever 1 confirmed, levers 2-3 not implemented).

### D. Deep-prefill at depth (F-pp) — **root-caused; fix prototyped**

**Answer.**  Not q8_0 at all: the regression is KV-type-independent (delivery f16 609.5 vs stock 686.9 at
pp150k on 1 GPU) and lives in the common FA path.  The per-token fit shows the delivery's `a` is smaller
(low-depth wins) but the depth slope `b` is ~51 % steeper, i.e. a per-(query x KV)-cell attention cost.
Two FA differences from stock cause it: **(1)** the delivery's head-256 `ncols=64` WMMA config is a
Strix-Halo-tuned half-tile row used for *all* WMMA calls (prefill + wide verify), worth ~1.5x per cell on
RDNA4/RDNA3_0; **(2)** the delivery omits stock's AMD `switch_ncols2` block, so for gqa 6 it picks
`ncols2=8` (2 of 8 lanes wasted) where stock picks `ncols2=2`.

**Fix (experiment, `patches/2026-09-14-prefill-rdna-config-and-ncols2.diff`).**  (1) make the RDNA config
`cc`-aware — RDNA3_5 keeps the halo row, RDNA4/RDNA3_0 take upstream's config; (2) adopt the AMD
`switch_ncols2` block **split-aware**: a frontend hint (`ggml_set_fa_tensor_parallel`, set in
`llama_context` from `split_mode() == TENSOR && n_cuda_dev > 1`) selects the generic `ncols2=8` for tensor
split and stock's AMD `ncols2=2` for a whole card.  Result: 1 GPU f16 pp150k **703.4 (+2.4 % over
stock)**, pp64k 946.8 (+8.0 %), bf16 675.8 (−1.6 %); 3-GPU `-sm tensor` f16 **1218.6 (+9.6 % over
stock)**; the 4B q4_0 `W=1..8` band stays **pure**.  Details: `MEASUREMENTS.md` §D.

**Testing rule (the finding that mattered).**  `-sm tensor` **masks single-card regressions**: the halo
config was faster in tensor-only testing for exactly that reason.  
Screen a candidate with the `t = a + b*n` slope fit from **pp8192/16384/32768/49152**, and always measure
**1 GPU as well as `-sm tensor`**.

**Status: DONE (fix prototyped + validated; promotion pending).**

### E. Adopt the #28867 head-256 WMMA threshold (F-wmma) — **investigated; no delivery regression**

**Answer.**  The delivery does not have the reported regression.  Its `Q->ne[1] > 8` guard already puts
the whole purity band (`W <= 8`, the reporter's `n_max 3` repro range) on TILE — upstream master lacks
that guard, which is why upstream sees it.  For the uncovered range (`n_q = 9..N`) the delivery's tuned
block-04 head-256 WMMA configs are at **parity** with TILE: recall `n_max 8` (W=9) 115.10 vs 115.72 t/s
and `n_max 15` (W=16) 147.19 vs 147.80 t/s (TILE +0.4-0.5 %, within noise), with **bit-identical
acceptance**; `llama-batched-bench` npl 1/8/9/16/32 is neutral.  Full evidence: `MEASUREMENTS.md` §E.

**Recommendation.**  No delivery change to fix a regression — there is none.  Adopting the MFMA-style
threshold (`n_q*gqa_eff > 64` for head>128) is a **~0.4 % neutral** selection change and is *not* a
purity change (the band is bounded at `W=8` by the matmul family switch, independent of FA).  Include it
only if we want explicit upstream/MFMA alignment; otherwise leave the
`GGML_CUDA_FA_WMMA_MAX_HEAD` control in place for future A/B.

**Status: DONE** (no action).

### F. (Umbrella) Keep the configuration matrix honest

Every result in this dossier must state: arm identity (git/tree/patch sha), ROCm, GPU count/split KV
type, `-fa`, context, prompt hash, `-n`, `--reasoning`, `--seed`, sampler, and whether `-lv 4`.  The
reporter's reconciliation only worked because both sides pinned these.  New prompts are added as files
with hashes (`prompts/README.md`), never edited in place.

---

## 5. Protocol / gates (do not weaken)

* **Purity first** (`GREEDY-PURITY.md` §19): a fix that makes the verify batch compute the decode
  arithmetic may cost a few percent; land it and repay it later.
* **Use the instrument that can see the defect** (§§11, 14, 20, 21): the raw-logit width probe to
  establish a band, the text gate for a real divergence, an oracle for a fused op, and a *long enough*
  run (`-n 3000`, §33) for a regime claim.
* **Everything is fast at depth 0** — decode perf must be shown at depth (the reporter's q8_0 finding is
  flat in depth, but the #27796 claim is not; both need the sweep).
* **Never run parallel/background benches.**
* **The adaptive-MTP gate** (`benchmarks/mtp-adaptive-methodology.md`) runs before shipping any
  decode/fusion change; the four-axis form runs at `-n 3000` with per-axis `--reasoning`.

---

## 6. Status

| action | question | status | result |
|---|---|---|---|
| A | KV-type × depth scaling (f16/bf16 reference) | **DONE** | BF16 slopes match stock f16 and are ahead at depth; q8_0/q4_0 fall off faster than stock (`MEASUREMENTS.md` §A) |
| B | quantized-KV decode/prefill (F-q8) | **fix prototyped + validated** | V4 activation policy + q4_0 native arm; q8_0 d65k +23 %, q4_0 +16 %, bit-identical, band-pure |
| C | adaptive MTP buffer footprint (F-buf) | **load failure FIXED by V4; RS reduction open** | root cause: RS = `n_seq x (1+n_max)` f32 GDN planes + the 744 MiB F16 scratch; V4 removes the scratch -> loads at `n_slots=4`; `--parallel 1`/structural reduction for more headroom |
| D | deep-prefill at depth (F-pp) | **root-caused; fix prototyped** | not q8_0-specific: head-256 `ncols=64` WMMA config (halo row) + missing AMD `switch_ncols2`; fixed + split-aware -> single f16 +2.4 %, tensor +9.6 % vs stock, purity held |
| E | #28867 head-256 WMMA threshold (F-wmma) | **DONE — no action** | delivery has no regression: `n_q>8` guard + tuned head-256 configs; W=9/W=16 verify at parity with TILE, acceptance bit-identical |
| F | protocol discipline | continuous | — |

`MEASUREMENTS.md` holds the raw runs (commands, logs, tables); this file holds the conclusions and the
register.

**Fork / delivery state.**  The experiment for Action B lives as an **uncommitted working-tree change**
in `~/llama.cpp` (4 files under `ggml/src/ggml-cuda/`), saved as
`patches/2026-09-14-v4-default-plus-q4_0-native.diff`; `build-rocm` is built from it.  The delivery
`patches/` and the fork's committed `rdna-boosts` branch are **untouched**.  Promotion to a block-15
amendment (regenerate the set, re-run `apply-all.sh` + `validate-set.sh`) is a separate, explicit step
gated on the maintainer.

## 7. Repository hygiene (2026-09-15)

The dossier's `results/` had accumulated ~12 MiB of raw `test-backend-ops` logs (~27k lines / ~3 MiB
each) that dwarfed every other artifact in the delivery repo.  They were trimmed **in place** to their
claim-carrying content by `tools/trim-backend-ops-log.py` (kept: header, the per-K/V-type coverage
table, the failure groups with one verbatim sample each, the totals) — `results/` is 248 KiB now, and the
banner in each file records its original line/byte count, so the trim is auditable rather than invisible.
The content that mattered survived *and read better*: fix1 shows the loader-branch NaNs
(q4_1/q5_0/q5_1 at hsk 64/72, plus one `iq4_nl`), fix2/fix3 the q5 nibble `ERR`s, fix4 `5951/5951` with
zero failures — the three-bug progression at a glance.  `MEASUREMENTS.md` (§I) was annotated accordingly.

Not touched, deliberately: the two `2026-09-15-launchdump-fw{1,2}.txt` (74 KiB each — they *are* the
`W=1` vs `W>=2` evidence), the small text/hash logs, and everything under `tools/` and `patches/`.

`archive/work/wip-archive/qwen35moe-prefill/data/` got the same treatment via its own
`data/trim-op-timing-log.py` (11 MiB -> 60 KiB; the op-timing aggregates are what `report.md`'s tables
were built from, so every cited share is preserved).
