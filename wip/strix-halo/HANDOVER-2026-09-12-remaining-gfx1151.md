# HANDOVER — remaining gfx1151 (Strix Halo) work + TODO cleanup (2026-09-12)

Self-contained brief for the next session.  The context that produced it is at ~550K tokens and will be
compacted, so treat this file (plus the docs it points at) as the source of truth.  Box: Strix Halo APU,
**gfx1151**, 1 device, ROCm 7.14 at `/opt/rocm-7.14-gfx1151`, 123 GiB unified RAM.

**Two goals, in this order:**
1. **TODO.md cleanup (priority).**  `TODO.md` is stale and has completed work left as footnotes in the
   *Active* list.  Fix it first (see §6) so the working list is trustworthy before starting new work.
2. **Close out the remaining gfx1151 work** (§3–§5).

> **OUTCOME (2026-09-12, this session).**  Goal 1 is done — `TODO.md`'s Active list is now active-only
> (state header refreshed, closed items moved to the Closed section).  Of goal 2:
> **items 4 and 7 were re-measured and disposed** (record
> `wip/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`, `GREEDY-PURITY.md` §18): the two recorded
> sparse-regime items were artifacts of the block-13 mmvq fusion (fixed 2026-09-12), **default gfx1151
> configs are pure**, item 7 is **closed**, and item 4 is re-scoped to a prompt-dependent q8_0
> *forced*-sparse residual (open, unlocalised).  **Item 5(f) is closed** (the fused MoE gate+up+GLU
> still wins ~+0.6 % prefill on the current tip → keep).  **Still open:** item 4's q8_0 residual;
> item 5(a) (needs the logits pin + a PPL/KL gate), 5(b)/5(c)/5(d)/5(g) (small / upstream); item 6
> (gfx1151 Phase-3 fingerprint — needs the gfx1201 box); item 16 (fusion perf).
>
> **OUTCOME (2026-09-12 (6)/(7), follow-up session).**  Item 4 deep dive: the residual is **not** a
> width dependence — teacher-forced replay at every verify width, with rollback schedules and unrelated
> rolled-back tokens, is bit-pure over 200 positions; the snapshot rollback restore is exact; `n_rs_seq`,
> `n_outputs_max`, CUDA-graph capture and the chunked-GDN boundary (identical call sequence) are ruled
> out.  New concrete defect: the MTP target's `embeddings_nextn` (`common/speculative.cpp:1431`) defers
> qwen4exp's last-layer output gather and shifts the **prefill's last-position logits by a ULP** — a real
> logits-level violation of `plain == draft-mtp`.  Record
> `wip/strix-halo/RECORD-2026-09-12-qsa-item4-deep-dive.md`; instruments
> `wip/strix-halo/qsa-item4/{mstep,rbprobe}.cpp`.  **Item 16 re-scoped** (the "pin nwarps/rps/item-split"
> plan is a dead end — the arms are already launch-identical; candidates are codegen and the Q8_1 cache)
> and **item 15 audited** (128 `-Wshadow` warnings / 27 `src/` files, 46 in the risky class).  Records:
> `wip/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9,
> `wip/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`.

## 0. Session hygiene / policy

- **Pushing policy (AGENTS.md):** this delivery repo (`llama-cpp-rdna-boosts`) is the only thing pushed,
  to its own `origin` (`git@github.com:stew675/llama-cpp-rdna-boosts.git`).  Never push from `~/llama.cpp`
  (the fork is disposable; the only permitted fork target is the personal fork and only on explicit
  request).  Fetch first; use `--force-with-lease` (never bare `--force`) if a rewrite is ever needed.
- **Git identity:** the delivery repo's stale local `[user]` override was removed 2026-09-12; it now uses
  the global `Stew Forster <stew675@gmail.com>`.  Do **not** re-add a local identity.
- **Purity first** (`GREEDY-PURITY.md` §19): a correctness/width-purity fix may cost a few percent; land
  it, record the delta, file the optimisation follow-up.
- **Docs freshness:** current state lives in the headers (`patches/README.md`, `README.md`, the top of
  `MANIFESTS.md`/`BASELINE.md`/`TODO.md`, `AGENTS.md`); dated records are appended, never edited in place.
  Delivery-affecting changes get a dated `WORKLOG.md` entry.

## 1. Environment / how to build and probe

**Models (all local):**

| key | path |
|---|---|
| qwen4exp IQ4_XS | `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` |
| qwen4exp Q4_K_M (+MTP) | `/llm/models/Qwen3.8/Flash-Next/Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00005.gguf`, `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |
| MoE 35B-A3B Q4_K_M | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| dense 27B Q8_0 | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| gemma4 12B Q8_0 (SWA) | `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf` |

**Build (gfx1151):** `~/bin/build-llama-rocm-714` targets `/opt/rocm-7.14-gfx1151` + `gfx1151`.  Equivalent
one-shot (what this session used), from a checkout root:

```sh
export ROCM_PATH=/opt/rocm-7.14-gfx1151; export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
HIPCXX="$ROCM_PATH/lib/llvm/bin/clang" HIP_PATH="$ROCM_PATH" cmake -S . -B build-rocm \
  -DGGML_RPC=1 -DGGML_HIP=ON -DGGML_NATIVE=1 -DGGML_HIP_RCCL=1 -DHIP_PLATFORM=amd -DGGML_HIP_GRAPHS=ON \
  -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="\$ORIGIN:$ROCM_PATH/lib" \
  -DCMAKE_C_COMPILER="$ROCM_PATH/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++"
cmake --build build-rocm -j 16
```

**Width probe:** `wip/kv-quant-purity-followups/tools/logits-dump-kv.cpp`, compiled against the tree:

```sh
clang++ -O2 -std=c++17 -I include -I ggml/include <probe>.cpp -o /tmp/lw \
  -Lbuild-rocm/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm/bin
HIP_VISIBLE_DEVICES=0 W=1 CTK=q8_0 CTV=q8_0 RS=from_w CB=0 SPLIT=layer NGL=99 FA=auto \
  /tmp/lw <model.gguf> <text.txt> 100 512      # prints [L] W=.. logits0_hash=.. nv=..
```

The stock probe hardcodes `n_ctx=2048` and does a single-batch prefill (so `P <= ~512`).  For `P > n_batch`
it needs a chunked-prefill variant and a configurable `CTX` (this session's `/tmp/longtext.txt` +
chunked variant are **in /tmp and will be gone** — recreate them if a long-prefill probe is needed).  The
issue25 instrument `wip/strix-halo/issue25/logits-width.cpp` measures `max|W1-W3|` / `max|W3-W5|` directly
and is the natural width gate.

**Other instruments** (`wip/kv-quant-purity-followups/tools/`): `textgen.py` (greedy-text gate),
`sobench.sh` / `mtpab2.sh` (interleaved A/B + adaptive-MTP gate), `qperf.sh`, `qsa-ppl-oracle.sh`,
`qsa-arm-trace.patch`, `node-dump-instrumentation.patch`, `fa-kernel-chooser-trace.patch`.
**Never run benches in parallel**; the box drifts (see `TODO.md` item 3's noise note) — use same-session
interleaved brackets.

## 2. Current state (what landed 2026-09-12)

- **Delivery:** 15-patch set, block 00 + 01-14.  Canonical tip **`13af95ac1`**, net tree
  **`f4791066f4a582316b1ca95f51c96cd10b905ef7`**; `scripts/make-patches.sh` default tip `13af95ac1`.
  Delivery repo `main` = **`1b55b51`** (= `origin/main`, pushed): patches regen + docs + the beta re-cut.
- **Block 13 amended** with the RDNA3_5 single-token-only mmvq fusion skip (`GREEDY-PURITY.md` §25):
  the dense gate+up+GLU mmvq fusion (six matcher sites in `ggml_cuda.cu`) and the MoE weighted-down tail
  (`ggml_cuda_mul_mat_id_weighted_rdna3_5_ok`, `mmvq.cu`) are skipped on RDNA3_5 unless
  `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS=1`.  Cost ≈ −0.9 % `tg128` on qwen4exp; prefill flat.
- **Beta Block 15 re-cut (12th):** base `13af95ac1` → beta tip **`888a59ee0`**, tree
  **`476d2d1e95947de7cc8cd806c40efc0f01927cd3`**; the patch is byte-identical to the 11th re-cut except the
  `From` line.  Beta-only (NOT in `patches/`).
- **Worktrees** (from `~/llama.cpp`): `/home/stew675/ll25/reint` (`13af95ac1`, canonical amended chain,
  unbuilt), `/home/stew675/ll25/verify` (patch-applied delivery, **built** `build-rocm`), `/home/stew675/ll25/beta15`
  (delivery+block15 `888a59ee0`, **built** `build-rocm`).  `~/llama.cpp` is on the superseded
  `qwen4exp-investigations` (`ab4444eb2`, old-chain fix) — do not use it for regeneration.

**Reference hashes (gfx1151, probe `P=100`, f16 unless noted, `RS=from_w`):**
qwen4exp `W=1..8` f16 **`453eaa618738273d`**, q8_0 **`113696b9d3eff98c`**; MoE 35B-A3B `W=1==W=8`
**`18999a784b416966`**; 27B dense **`e165ef98a4efc016`** (unchanged by the fix); the opt-in reproduces the
pre-fix impurity (qwen4exp f16 `W=1` **`8abc6206d1e80709`** vs `W=8` `453eaa61`).  Text gates: qwen4exp
sparse `804de0576868`, MoE `68c0a24ed8d4`; beta PPL oracle sparse `6.5394` / dense `6.5377`.

## 3. Remaining gfx1151 work — QSA (items 4 and 7)

Both are gfx1151's **default regime** (dense decode below the 64K `qsa_dense_decode_until` crossover,
sparse above), and item 7 is the follow-on.  See `GREEDY-PURITY.md` §18–19.

**Item 4(a) — the fused indexer score is not byte-identical and is `n_tokens == 1`-gated.**
`GGML_CUDA_QSA_INDEXER_SCORE` (default ON, `src/models/qwen4exp.cpp` `idx_score_fused`) claims to replicate
the per-op F32 arithmetic byte-identically; measured false, and the gate is `n_tokens == 1`, so a W=1
decode and an n-token verify consume the indexer through different score paths.  Unreachable on gfx1201's
default (decode builds no scores), but the default path on gfx1151 above 64K.  Fix = make the kernel
token-generic, or default the probe OFF; add a width gate (W=1..8 one hash with the sparse arm forced,
`LLAMA_QSA_DENSE_DECODE_UNTIL=0`).

**Item 4(b) — a residual split survives with one arm.**  With `LLAMA_QSA_DENSE_DECODE_UNTIL=0` (both widths
on the sparse arm) `plain` and `n_max 3` agree for 706 chars then diverge; `GGML_CUDA_QSA_INDEXER_CACHE=0`
does not reconcile them.  Unlocalised state/store width-dependence.  Localise with the arm trace +
node dump, then fix.

**Item 7 — re-measure the MTP-side QSA crossover on this box, then set the gfx1151 policy.**
The published 64K "dense below / QSA above" table was measured for the W=1 decode regime; the verify batch
now takes the dense arm below the crossover (`QSA_DECODE_BAND = 8`), and above 64K the sparse regime is at
parity per the 2026-09-07 protocol **and** still impure (§18).  Under purity-first the likely answer is
**dense decode at every depth on gfx1151 too** (a one-line `build_layer_attn` change), once §18 is fixed.
Do this **on the Strix box** (it is available now) — the TODO entry still says "blocked until then".

## 4. Remaining gfx1151 work — prefill follow-ons (item 5)

All prefill; decode is closed.  Records under `wip/archive/qwen4exp/discovery/`.

- **(a) MoE topk fusion adoption (~0.5 % prefill).**  Replaces the full-512 argsort with the fused partial
  top-10.  **Blocked on a numerics quality gate**: fused top1 logit 18.690 vs unfused 18.424 → needs a
  CPU reference + PPL/KL gate (the `qsa-ppl-oracle.sh` pattern).  ~188 MB arena cost if adopted.
  Record: `2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`.
- **(b) `ssm_alpha` + `ssm_beta` single-walk fusion (~0.3–0.6 % prefill).**  Blocked by graph expansion
  order (the two MMs are non-adjacent).  Routes: load-time stacked weights, or a qwen4exp graph restructure
  + custom kernel.  Record: `2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`.
- **(c) launch-ledger remainder** (small-pp): +38 `scale_f32`/eval and an `rms_norm<256,true>` count diff —
  likely sub-0.2 %, root-cause-only value.
- **(d) `mmq` accumulator-overflow latent defect (`I < nwarps*16`)** — correctness hygiene; report upstream.
- **(f) Re-check whether the block-13 gate+up+GLU arm still has a unique Strix Halo win.**  With the
  2026-09-06 model-neutral folds the isolated fused-MoE delta is ~0 there (`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`
  on/off: pp2048 +0.4 %, pp16384 +0.2 %; absolute prefill ~10–13 % higher; the fusion still fires).  Decide
  keep vs. fold.
- **(g) V3 prefill cost is arch-dependent (low priority).**  gfx1151 −3.2 % at pp20480 (4B, q8_0) vs RDNA4
  −1.3 %; decode flat; large net memory win and on by default.  If an iGPU tuning pass runs, look at the
  derived MMA kernel's `J`/occupancy on gfx1151.

## 5. New follow-up from this session (record it, then decide)

- **Restore the ~0.9 % `tg128` the block-13 RDNA3_5 skip costs.**  The correct long-term fix is to make the
  fused `ncols_dst == 1` kernel reproduce the standalone mmvq reduction (pin `nwarps`/`rps`/item-split —
  the §17 pattern) instead of skipping the fusion; the same may be true for the weighted-down tail.  This
  would also retire the within-band gfx1151 width variance at the kernel level.  Low priority (purity is
  restored), but it is the honest closure of the issue-25 "block-13 `n_q=1` short-K mmvq variance".

## 6. TODO.md cleanup (do this FIRST — it is a maintainer priority)

`TODO.md`'s *Active* list currently mixes active work with finished-work footnotes, and its header is
stale.  The rule the maintainer wants enforced: **the Active list is only for active things; completed
work is a one-line entry in the Closed section.**

Specific fixes required:

1. **Header state line** still reads canonical tip `124abba9e` / `make-patches.sh` default `124abba9e`.
   Update to **tip `13af95ac1`**, tree **`f4791066f4a582316b1ca95f51c96cd10b905ef7`**, default `13af95ac1`.
2. **Add a Closed one-liner** for the block-13 RDNA3_5 single-token-only mmvq fusion skip (the issue-25
   "block-13 `n_q=1` short-K mmvq variance"): dense gate+up+GLU + weighted-down skipped on gfx1151, `W=1..8`
   one hash (`453eaa61` f16 / `113696b9` q8_0), cost −0.9 % tg128; pointer to `GREEDY-PURITY.md` §25 +
   WORKLOG 2026-09-12 (2).  Optionally the §5 perf follow-up above as a new *Active* gfx1151 item.
3. **Item 2** is a heading with no body ("demoted … FIXED; see the Closed section") — **delete the heading**;
   the GDN fix is already in Closed.
4. **Item 1 ("Block 15 promotion")** is genuinely active (beta window + go-ahead), but the fixed-blocker
   narrative belongs in Closed (it is already there) — trim the item to the live part.
5. **Item 6** keeps the "gfx1201 port is DONE" narrative in *Active* — move that to Closed and leave only
   the still-open gfx1100 / gfx1151 validation legs.
6. **Item 5** — confirm each sub-item is still open (some may have been folded); close any that are done.
7. Fix stale tip references wherever the cleanup touches (item 14 also names `124abba9e`).
8. After the audit, add a dated `WORKLOG.md` entry for the TODO cleanup (docs-only).

## 7. Suggested order for the next session

1. `TODO.md` cleanup (§6) — commit + push (docs-only).
2. Item 7 + Item 4 (QSA): the crossover re-measure and the two §18 items are all on this box and are the
   direct continuation of the gfx1151 MTP-purity thread; item 7's policy change depends on §18.
3. Item 5 prefill follow-ons — (a) needs the quality gate, (b)/(c) are smaller, (d) is an upstream report.
4. Item 6 gfx1151 Phase 3 cross-arch fingerprint check.
5. §5 kernel-level restoration (opportunistic).

## 8. Pointers

- Purity analysis + invariants: `GREEDY-PURITY.md` (§11 band, §18 sparse, §19 policy, §25 this session's fix).
- Delivery records: `patches/README.md`, `MANIFESTS.md`, `BASELINE.md`, `WORKLOG.md` (2026-09-12 (2)).
- Beta Block 15: `beta/block-15-campaign-wins/{README,HANDOVER,BETA-TESTING}.md` (12th re-cut).
- gfx1151 port plan (banner + stale checkboxes warning): `wip/qwen4exp/gfx1201-porting.md`.
- Issue-25 history: `wip/strix-halo/issue25/{GATE-2026-09-11-block00-rdna35,RECORD-2026-09-11-issue25}.md`.
- This session's fix record: `wip/strix-halo/rdna35-mmvq-fusion-purity/README.md`.
- Instruments: `wip/kv-quant-purity-followups/tools/` (see §1).
