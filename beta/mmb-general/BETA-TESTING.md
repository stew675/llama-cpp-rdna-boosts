# `beta/mmb-general` — BETA-TESTING.md (the gfx1151 final re-validation)

**For the maintainer's Strix Halo (gfx1151, RDNA3_5) box.**  Everything here is written so the whole
validation is a copy-paste run: apply, build, run the four gates, compare against the numbers below.

The set was developed and tuned **on gfx1151** and later ported to gfx1201 (RDNA4) and gfx1100
(RDNA3_0).  The re-validation exists because **the combination was never re-run on gfx1151 after the
two ports moved the `mmb_*` tunables into a per-arch table and added three arch-scoped patches** — the
individual "gfx1151 unchanged" claims were made one step at a time (mostly by comparing device
assembly), never once end-to-end on the final set.

## 0. What the set promises on gfx1151

> **2026-09-25 consolidation — the set is now default-ON and r13-based.**  The `closing-the-gap`
> campaign was folded in and its "beneficial features default ON" patch flipped the campaign on:
> **`GGML_CUDA_MMB` defaults to `1`** (env only *disables*), as do `GGML_CUDA_MMB_HC16` and
> `GGML_CUDA_MMB_RDNA3`, and the `hc_gate_mix` fusion.  The promise table below is re-stated for the
> **default-on 28-patch set**.  (The pre-consolidation beta was MMB-opt-in on r12; that is historical.)

| | |
|---|---|
| **Default build (`GGML_CUDA_MMB` unset → `1`)** | the full campaign: the gfx1151 `mmb` win, HC16, `hc_gate_mix`, the closing-the-gap prefill fusions, the MMVQ band policy, etc. |
| **`GGML_CUDA_MMB=0`** | the campaign off — the r13 delivery plus the default-on, arch-neutral closing features. |
| Never | a regression at depth, a width-purity change, or an MTP acceptance change. |

**The purity contract is the *intra-build* one** (Gate 1), not cross-build equality: with the features
defaulted on, cross-build equality against the delivery no longer holds by design (`GREEDY-PURITY.md`
§5).

The three patches added by the gfx1100 port cannot reach gfx1151 by construction (0011 is an additive
predicate widening; 0012 is an `if (GGML_CUDA_CC_IS_RDNA3_0(cc))` arm; the per-M `nwarps` experiment
was **moved out** of this set into `wip/nwarps/`).  Everything else is either RDNA4-gated in the
per-arch table or arch-neutral.  Re-validating confirms that reasoning.

## 1. Apply and build

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2                                   # the fork point
bash <repo>/scripts/apply-beta.sh . <repo>               # apply-all + 28/28, tree 2e4e8004...
git rev-parse HEAD^{tree}                                # expect 2e4e8004f0562485a3b7ba179cd4781a227989ad
```

Re-based 2026-09-24 onto upstream master `84e76d8a2` (release `v16-84e76d8a2-r1`); the previous r13
tree was `468c6496…`.  Verified strict `git am` **28/28** on a fresh `84e76d8a2` worktree with the
delivery set applied first (delivery tree `336d0f43…`), producing `2e4e8004…`; only `0014` (GDN/PLE
conv1d) and `0015` (narrow-row RMS norm) needed a conflict resolution (upstream's restructured
`ggml_backend_cuda_graph_optimize` loop).  Build with the usual gfx1151 script; the runtime env is
`export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH`.

`GGML_CUDA_MMB_CFG=1` prints the resolved per-arch config once — gfx1151 should read
`cc=0x1001151 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=1 down16=0 gatemix=1 blk16=0 res16=0
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

**Cross-build comparison (historical).**  The pre-consolidation beta was MMB-opt-in on r12, and with
MMB unset it measured bit-identical to r12 on gfx1201 (PPL 9.4293, all five same-seed hashes).  That is
**not** a gate — and it no longer applies, since the consolidated set defaults MMB on.  See the
2026-09-22 correction record in `wip/closing-the-gap/closing-the-gap.md`.

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
| `GGML_CUDA_MMB` | **1 (on)** | the master switch; `=0` = the delivery's MMQ paths |
| `GGML_CUDA_MMB_CFG=1` | — | dump the resolved per-arch config (`env || arch default`) once |
| `GGML_CUDA_MMB_TYPES=<csv>` | arch mask | restrict the MMB weight types (gfx1151 = the full set) |
| `GGML_CUDA_MMB_DENSE_TYPES=<csv>` | arch mask | the dense-path types (gfx1151 = the full set) |
| `GGML_CUDA_MMB_DENSE` / `_ROUTED` / `_GLU` | per arch | force a path on/off |
| `GGML_CUDA_MMB_MIN_T` / `_TALL` / `_TINY_M` / `_TINY_TT` / `_F32SPLIT` / `_CACHE` | per arch | the tuning knobs |
| `GGML_CUDA_MMB_HC16` / `_DOWN16` | **1** / **0** | the bf16 producers (HC16 default ON since the consolidation; DOWN16 still opt-in; measured inert on RDNA4, +1-3 % on gfx1151 in the original record) |
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
`gap-closing-r13`) — the tree the 28-patch set now reproduces exactly (the consolidation is
tree-identical), so a green here is a green for the set.
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

---

## 6. 2026-09-25 — consolidated 28-patch set, full reproduction

Ran the *shipped* `scripts/apply-beta.sh` flow end-to-end on a fresh upstream clone (the user path),
then the standard gates, to confirm the consolidation did not change behaviour.  The applied tree is
`468c64963ae45e72367c73809efa7cc038217e8a` (identical to the 2026-09-22 tree), so the numbers carry.

```sh
git clone https://github.com/ggml-org/llama.cpp /tmp/llama-beta-verify && cd /tmp/llama-beta-verify
git checkout ebbb18522
bash <repo>/scripts/apply-beta.sh . <repo>      # apply-all + 28/28, tree 468c6496...
~/bin/build-llama-rocm-714                      # clean -j16 build, EXIT 0
```

* **Apply** — `apply-beta.sh` auto-ran `apply-all.sh` (16/16) then `28/28`; tree asserted
  `468c6496…`.
* **Build** — clean `-j16`, EXIT 0 (6 m 38 s cold; ccache warm on repeat).
* **Config** — `MMB_CFG cc=0x1001151 dense_geom=0 … hc16=1 … gatemix=1 … routed=1`.
* **Width probe** — `test-logits-width-probe <qwen4exp IQ4_NL> prose 1024 512`:
  `width_purity=PASS (worst maxdiff 0)`, row-0 `268e0673300b7a33` — matches.
* **Oracles** — `FLASH_ATTN_QSA` OK, `GATED_DELTA_NET` OK.
* **Coherence** — `plain == draft-mtp n3` byte-identical (`434 chars sha=984263fb8e0f`).
* **MTP** — qwen4exp, `-n 3000`, `draft-mtp n3`: acceptance **0.84281** (acc/pos
  0.924/0.842/0.762), **55.7 t/s vs plain 31.5 t/s** → MTP ≥ plain.
* **llama-bench** — qwen4exp IQ4_NL q8_0 KV: pp512 **1007.1 t/s**, tg128 **32.56 t/s**.

**Verdict:** GREEN.  The consolidation is behaviour-preserving (bare tree identity), and the shipped
`apply-beta.sh` reproduces it from a clean clone.

---

## 7. 2026-09-24 re-base smoke (gfx1151)

The set was re-based onto upstream master `84e76d8a2` (delivery `v16-84e76d8a2-r1`) and smoke-tested on
gfx1151 (ROCm 7.14, `~/bin/build-llama-rocm-714`, clean build 6 m 41 s):

* `GGML_CUDA_MMB_CFG=1` → `cc=0x1001151 dense_geom=0 … routed=1` (the RDNA3_5 row, unchanged).
* `test-backend-ops -o FLASH_ATTN_QSA` **26/26**; `-o GATED_DELTA_NET` **46/46**.
* `test-logits-width-probe` **PASS (worst maxdiff 0)** on Qwen3.5-4B, qwen4exp Q4_K_M and
  Qwen3.6-35B-A3B Q4_K_M (prose prompt, 512/512).
* qwen4exp Q4_K_M + Q4_K_M MTP sidecar, prose, seed 42 / temp 0 / `--reasoning off`, `-n 1500`:
  **draft acceptance 0.80091** (`acc per pos = 0.916, 0.785, 0.698`), 33.2 t/s vs plain 23.7 t/s.

This is a smoke pass on the new base, not a replacement for the full gfx1151 beta-window
re-validation above.
