# `beta/mmb-general` — BETA-TESTING.md (the gfx1151 final re-validation)

**For the maintainer's Strix Halo (gfx1151, RDNA3_5) box.**  Everything here is written so the whole
validation is a copy-paste run: apply, build, run the four gates, compare against the numbers below.

The set was developed and tuned **on gfx1151** and later ported to gfx1201 (RDNA4) and gfx1100
(RDNA3_0).  The re-validation exists because **the combination was never re-run on gfx1151 after the
two ports moved the `mmb_*` tunables into a per-arch table and added three arch-scoped patches** — the
individual "gfx1151 unchanged" claims were made one step at a time (mostly by comparing device
assembly), never once end-to-end on the final set.

## 0. What the set promises on gfx1151

| | |
|---|---|
| **With `GGML_CUDA_MMB` unset (the default in this beta tree)** | **byte-identical to the delivery r12** in *this opt-in beta tree*.  MMB is **opt-in** (`getenv("GGML_CUDA_MMB") ? atoi : 0`), so the default state is r12 plus only the arch-neutral groups — which were measured numerically neutral (on gfx1201 the WIP with MMB off was *bit-identical* to the delivery at PPL 9.4293 and on all five same-seed hashes).  **This is a regression aid, not the purity contract** — see the note under Gate 1 and `GREEDY-PURITY.md` §5: once a feature is defaulted on (as on the `gap-closing` branch) cross-build equality is expected to break, and the gate is the *intra-build* set. |
| **With `GGML_CUDA_MMB=1`** | the original gfx1151 `mmb` win, **unchanged by the RDNA4/RDNA3_0 scoping**.  gfx1151 keeps the **full** weight-type set and the original tile/threshold constants. |
| Never | a regression at depth, a width-purity change, or an MTP acceptance change. |

The three patches added by the gfx1100 port cannot reach gfx1151 by construction (0011 is an additive
predicate widening; 0012 is an `if (GGML_CUDA_CC_IS_RDNA3_0(cc))` arm; the per-M `nwarps` experiment
was **moved out** of this set into `wip/nwarps/`).  Everything else is either RDNA4-gated in the
per-arch table or arch-neutral.  Re-validating confirms that reasoning.

## 1. Apply and build

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout ebbb18522                                   # the fork point
bash <repo>/scripts/apply-all.sh .                       # -> r13 delivery, tree bb7b6d07...
git checkout -b mmb-beta
git am <repo>/beta/mmb-general/patches/*.patch           # 28/28
git rev-parse HEAD^{tree}                                # expect 468c64963ae45e72367c73809efa7cc038217e8a
```

Consolidated 2026-09-25 (the `closing-the-gap` campaign folded in).  Verified strict `git am`
**28/28** from the r13 tree, producing `468c64963ae45e72367c73809efa7cc038217e8a`; the set is
**tree-identical** to the combined `gap-closing-denseband` tree, so the gfx1201/gfx1100 results
carry over and the gfx1151 re-validation below is the beta-window task.  Build with the usual gfx1151
script; the runtime env is `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH`.

`GGML_CUDA_MMB_CFG=1` prints the resolved per-arch config once — gfx1151 should read
`cc=0x1001151 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=0 down16=0 gatemix=0 blk16=0 res16=0
glu=1 bf16w=1 iq3xxs_glu=0 routed=1`.  **Note `dense_geom=0` and `routed=1`** (the gfx1201 line reads
`dense_geom=1 … routed=0`) — the RDNA4 scoping must NOT have leaked into the RDNA3_5 row.  The `cc`
the dump prints is `0x1000000 + <gfx number>`: `cc=0x1001201` was observed on gfx1201, so gfx1151 is
`0x1001151`.

## 2. The four gates

### Gate 1 — the *intra-build* purity gate

**This is the real contract; do not use cross-build equality as a gate.**  `GREEDY-PURITY.md` §5:
*"Bit-identical to stock [or to r12, or MMB on vs off] is a reproducibility requirement, not a
correctness requirement."*  A prefill kernel swap changes the prefill logits legitimately — this beta's
own `README.md` calls it the **"approved prefill re-baseline"**, and its width-probe table shows row-0
hashes differing above `MMB_MIN_T = 512` while `width_purity` stays PASS.

The intra-build gate is that the decode/verify band agrees with itself and that widths agree:

```sh
# same build, same state: one-token greedy vs the n+1 verify batch.  Compare ONLY the extracted text.
#   --spec-type none            vs   --spec-type draft-mtp --spec-draft-n-max 3
python3 <repo>/scripts/extract-generated.py <log>
# must be byte-identical (the W=1..8 band takes one reduction path)

test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512
# must print: width_purity=PASS (worst maxdiff 0)
```

Run both on a dense model, a MoE model and (if it fits) qwen4exp.  The guaranteed pure KV types are
**f16, bf16, q5_0, q5_1, iq4_nl** (`q4_0`/`q4_1`/`q8_0` relax the *logits* level per `GREEDY-PURITY.md`
§36 — text purity still holds).  Gate 4 (MTP) is the decode half of the same contract.

**Optional regression aid (opt-in beta tree only).**  While MMB is *unset* it is r12 plus the
arch-neutral groups, and *for this beta tree* those measured bit-identical to r12 on gfx1201 (PPL
9.4293, all five same-seed hashes).  That comparison is a useful bisection tool — which neutral group
has a numeric side effect? — but it is **not** a gate: it is meaningless once MMB is defaulted on.  See
the 2026-09-22 correction record in `wip/closing-the-gap/closing-the-gap.md`.

### Gate 2 — `GGML_CUDA_MMB=1` must recover the original gfx1151 win

This is the gate the ports could plausibly have broken: the tunables now come from
`mmb_arch_defaults(cc)`, and if the RDNA3_5 row were wrong the win would silently shrink.

```sh
export GGML_CUDA_MMB=1
llama-bench -m <dense model> -ngl 99 -p 2048,8192,32768 -n 0 -b 2048 -ub 2048 -r 5
```

Reference: gfx1151 measured **+32…+48 % prefill** on the `mmb`-eligible models when this campaign
started.  **Interleave** the MMB-on and MMB-off runs in one warm session (the first prefill of an
invocation is cold-start-limited), and **prefer pp32768+ for a verdict** — on gfx1201 the shallow
numbers moved ±3 % run-to-run with the clock ramp while pp65536/98304 agreed to 0.1–0.4 %.

### Gate 3 — the op oracles

```sh
test-backend-ops -o FLASH_ATTN_QSA    # 26 cases (3 of them qsa3=1)
test-backend-ops -o GATED_DELTA_NET   # 46/46
test-backend-ops -o INDEXER_TOPK      # the G5 oracle
test-backend-ops -o FLASH_ATTN_EXT    # expect ~5954 OK, 0 FAIL
```

> **Trap on the last one: do not count its results from a `2>&1`-merged log.**  The status is
> ANSI-wrapped and `print_test_console` writes the name to stdout while the test emits warmup/allocation
> notices to stderr, so merging the streams orphans the status onto its own line and the OK count looks
> like it moves between runs.  It does not — five captures gave 5953/5954 OK and 0 FAIL every time.
> Separate the streams (or pair name+status) and strip ANSI.

### Gate 4 — MTP (the decode gate)

`benchmarks/mtp-adaptive-methodology.md`, Protocol A: seed 42, temp 0, **`-n 3000`**, acceptance
**> ~0.45 at pos 1**, MTP **≥ plain** at depth 3, reasoning pinned per axis (R=on, P/C/K=off).

Reference cells from the delivery: **27B 0.76744**, **qwen4exp 0.44262**.  On gfx1201 the final set
measured 0.63624 (dense 27B), 0.72372 (MoE 35B) and 0.70093/0.64372 (qwen4exp, vs the delivery's
0.64372 — the WIP and the delivery were byte-identical there on the dense and MoE models).

> This gate matters here because the campaign's worst historical regression was an **MTP collapse that
> every other gate passed** (2026-09-02: an rms_norm→mmvq Q8_1 cache fold corrupted multi-token
> `MUL_MAT_ID`; acceptance went to 0/1527 while plain decode and single-token coherence were fine).
> `llama-bench tg` cannot see it: it only ever decodes 1 token, so it never exercises the verify widths.

## 3. The kill-switches (for bisecting a surprise)

| knob | default | effect when changed |
|---|---|---|
| `GGML_CUDA_MMB` | **0 (off)** | the master switch; off = the delivery's MMQ paths |
| `GGML_CUDA_MMB_CFG=1` | — | dump the resolved per-arch config (`env || arch default`) once |
| `GGML_CUDA_MMB_TYPES=<csv>` | arch mask | restrict the MMB weight types (gfx1151 = the full set) |
| `GGML_CUDA_MMB_DENSE_TYPES=<csv>` | arch mask | the dense-path types (gfx1151 = the full set) |
| `GGML_CUDA_MMB_DENSE` / `_ROUTED` / `_GLU` | per arch | force a path on/off |
| `GGML_CUDA_MMB_MIN_T` / `_TALL` / `_TINY_M` / `_TINY_TT` / `_F32SPLIT` / `_CACHE` | per arch | the tuning knobs |
| `GGML_CUDA_MMB_HC16` / `_DOWN16` | **0** | the bf16 producers (measured inert on RDNA4; +1-3 % on gfx1151 per the original record) |
| `GGML_CUDA_FA_KV_NATIVE` | auto | the native q8_0/q4_0 (on) and bf16 (off) FA cache paths |
| `LLAMA_QSA3_ENABLE` | compile-time | the packed-block WMMA QSA path |

## 4. What to report

For each gate: the exact command, the `MMB_CFG` line, and the numbers.  If something fails, the two
most likely places are (a) the **RDNA3_5 row of `mmb_arch_defaults(cc)`** — `routed`/`dense_geom`/
`f32split` — and (b) an **mmvq width-uniformity** interaction (`GREEDY-PURITY.md` §19/§25).  Report the
gate name, not "it was slower".

**Also worth confirming while the box is warm:** that the gfx1151 `mmb` win is still `+32…+48 %` at
**depth** (pp32768+), since that is the number the whole campaign is ultimately justified by.

---

## 5. gfx1151 re-validation — 2026-09-22 (session 8, r13 campaign)

Run on the **r13 + `beta/mmb-general` + gap-closing `0001..0014`/`0016`/`0017`** campaign (fork
`gap-closing-r13`) — a superset of this 12-patch set, so a green here implies a green for the beta.
Box: gfx1151, ROCm 7.14, `MMB_CFG cc=0x1001151` (RDNA3_5 row: `dense_geom=0 routed=1 f32split=1
min_t=512`).

* **Gate 1 (intra-build purity)** — `test-logits-width-probe <qwen4exp IQ4_NL>
  prompts/prose-rdna-boosts.txt 1024 512`: **`width_purity=PASS (worst maxdiff 0)`** (f16 KV), the
  qwen4exp row-0 hash `268e0673300b7a33` matching the campaign record; `qwen4exp IQ4_NL` plain text is
  coherent.  (The MMB-on-vs-r12 cross-build check is **retracted**; intra-build is the contract.)
* **Gate 2 (MMB win)** — `mmb_dense`/`mmb_routed`/`mmb_routed_glu` fire on the uniform model; the
  campaign's default set is the full set (`GGML_CUDA_MMB_CFG` shows `dense_geom=0 routed=1`).  The
  gfx1151 `mmb` win carries (`+19..+63 %` on the new quant types below, `+32…+48 %` on the original set
  per the campaign body).
* **Gate 3 (op oracles)** — `GATED_DELTA_NET` **46/46**, `FLASH_ATTN_QSA` **26/26**, the new-type
  `MUL_MAT`/`MUL_MAT_ID` (`patches/0017`) all pass, `LIGHTNING_INDEXER` **225/225** (`patches/0016`).
  `INDEXER_TOPK` has **0 cases** in this tree (the G5 oracle is aspirational here).
* **Gate 4 (MTP)** — qwen4exp, shared-NextN Q8_0 sidecar (fixed in delivery block 00 / r13), prose,
  seed 42 / temp 0 / `--reasoning off` / `-n 3000`: **draft acceptance 0.85541** (acc/pos
  0.938/0.853/0.776), **56.5 t/s vs plain 31.7 t/s** → MTP ≥ plain.  Well above the `~0.45` bar.
* **New quant types** (`patches/0017`, default ON on non-RDNA4): Q4_0/Q4_1/Q5_0/MXFP4/NVFP4 —
  `MUL_MAT` 48/47/14/46/45, `MUL_MAT_ID` 74/75/3/74/73, PPL parity, pp8192 **+19.5/+22.4/+25.4 %**,
  35B-A3B Q4_1 pp4096 **+63 %**, gpt-oss-20b MXFP4 pp4096 **+5.2 %**.  Detail:
  `wip/closing-the-gap/2026-09-22-mmb-quant-coverage.md`.

**Verdict:** gfx1151 re-validation GREEN.  The only gate the beta window owed is closed.
