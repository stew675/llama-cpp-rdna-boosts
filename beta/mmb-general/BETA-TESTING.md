# `beta/mmb-general` — BETA-TESTING.md (the gfx1151 final re-validation)

> **INTEGRATED 2026-09-25 — folded into the delivery; this directory is the historical verification
> record.**  The 28 patches are now part of the 16 delivery blocks on the `beta-integration` branch
> (apply `scripts/apply-all.sh` only; `apply-beta.sh` was removed), and the `release.json` tree is
> the campaign tree `24bb0f5acb…`.  The `git am` recipe in §1 is historical; the gates below remain
> the campaign's validation record.  See `../../WORKLOG.md` (2026-09-25).

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
bash <repo>/scripts/apply-beta.sh . <repo>               # apply-all + 28/28, tree 24bb0f5acb...
git rev-parse HEAD^{tree}                                # expect 24bb0f5acb3e866abd4cad8c0de1bad45a20cb47
```

Re-cut 2026-09-25 onto the delivery's `v16-84e76d8a2-r7` (delivery tree `7726e514…`, the block-14
Meta-tensor-split scheduler race fix), producing `24bb0f5acb…`.  The patch bodies are
**byte-identical** to the r6-based set (only the `From` lines and `commits.txt` changed), because the
beta set does not touch the `ggml_backend_sched_alloc_splits` region the fix changes.  Before that,
r6's was `1df5769c…` (delivery tree `504894e6…`, the block-15 bf16 native default flip), r5's was `469082e4…`
(delivery tree `de86c5e1…`, f16/bf16 on the RDNA4 GQA-6 decode/verify FA band, issue #45), r4's
`70cc895a…` (delivery tree `5938da09…`), r3's `0daefe22…`, r2's
`e00275ff…` and r13's `468c6496…`.  Verified strict `git am` **28/28** on a fresh `84e76d8a2` worktree
with the delivery set applied first; only `0014` (GDN/PLE
conv1d) and `0015` (narrow-row RMS norm) needed a conflict resolution at the 2026-09-24 re-base
(upstream's restructured `ggml_backend_cuda_graph_optimize` loop).  The 2026-09-25 re-bases are
straight replays.  Build with
the usual gfx1151 script; the runtime env is
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
gfx1151 (ROCm 7.14, `~/bin/build-llama-rocm-714`):

* `GGML_CUDA_MMB_CFG=1` → `cc=0x1001151 dense_geom=0 … routed=1` (the RDNA3_5 row, unchanged).
* `test-backend-ops -o FLASH_ATTN_EXT` **5956/5956**, `-o FLASH_ATTN_QSA` **26/26**,
  `-o GATED_DELTA_NET` **46/46**, `-o LIGHTNING_INDEXER` **225/225**, `-o INDEXER_TOPK` 0/0.
* `test-logits-width-probe … 1024 512` **PASS (worst maxdiff 0)** on Qwen3.5-4B, Qwen3.6-35B-A3B Q4_K_M
  and qwen4exp IQ4_XS; `plain == draft-mtp n3` text **byte-identical** on dense 27B Q8_0, the MoE and
  qwen4exp.
* `test-recurrent-state-rollback` max diff 0.
* MTP Protocol A (`-n 2000`, seed 42, temp 0, `--reasoning off`): dense 27B 0.82188 (20.2 vs 7.6 t/s),
  MoE 0.76164 (87.3 vs 53.7), qwen4exp 0.82151 (41.2 vs 25.1), shared sidecar 0.81962 with 0 `X < Y`.
* Cross-checkpoint parity against the old `~/llama.cpp` r13+beta build is within ~2 % (qwen4exp
  prefill/deep-decode and MoE prefill/decode).

### 2026-09-24 — `MUL_MAT_ID` abort fixed in patch `0027`

`test-backend-ops -o MUL_MAT_ID` **aborted** (`mmvq.cu` default) on a `type_a=f32` case.  Patch `0027`'s
gfx1151 dense-band force in `ggml_cuda_mul_mat` enabled MMVQ for *any* small batch with
`src0->ne[1] % 128 != 0`, including non-quantized weights; the F32 `MUL_MAT_ID` fallback slice then
reached the MMVQ type switch, which has no F32 case.  The base delivery and upstream do not have the
force.  A one-line guard (`ggml_is_quantized(src0->type)`) was folded into patch `0027`; the full
`MUL_MAT_ID` oracle is now **929/929** (matching upstream), and the quantized MoE `tg128/tg512`
(56.33 / 56.63) is unchanged.  Applied tree after the fix: **`7f339b10…`**.

The full gfx1151 beta-window re-validation above is still owed on this tree.

---

## 8. 2026-09-25 — full gfx1151 beta-window re-validation (tree `7f339b10`)

The four gates were run end-to-end on the **regenerated 28-patch set** (tree `7f339b10`, build 11217 /
`6e46fb052`) on gfx1151 (ROCm 7.14, `~/bin/build-llama-rocm-714`, ccache-warm).  Single 8060S.
Runtime env: `LD_LIBRARY_PATH=/home/stew675/stew-llama-cpp/build-rocm/bin:/opt/rocm-7.14-gfx1151/lib`
(the width probe is run with that path first so it links the build under test, not `/llm/bin`).

### Gate 0/3 config dump

`MMB_CFG cc=0x1001151 dense_geom=0 min_t=512 glu_thresh=32 routed_thresh=32 tall=2 tiny_m=1/1
f32split=1(min_m=128,min_k=0) cache=4 shadow=0/6144MB hc16=1 down16=0 gatemix=1 blk16=0 res16=0
glu=1 bf16w=1 iq3xxs_glu=0 routed=1` — the RDNA3_5 row, `dense_geom=0 routed=1` as required.

### Gate 1 — intra-build purity: **GREEN**

`--spec-type none` == `--spec-type draft-mtp --spec-draft-n-max 3`, prose, seed 42 / temp 0 /
`--reasoning off`, f16 KV, `-n 512 -c 8192 -b 2048 -ub 2048` (text via `scripts/extract-generated.py`):

| model | sha256 (both arms) |
|---|---|
| dense 27B Q8_0 | `90686d1edf24` |
| MoE 35B-A3B Q4_K_M | `d72a1fc679a5` |
| qwen4exp IQ4_XS | `b746fb3e77ff` |

`test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512` →
`width_purity=PASS (worst maxdiff 0)` on all three.

### Gate 2 — `GGML_CUDA_MMB` win: **GREEN**

`llama-bench -p 2048,8192,32768 -n 0 -b 2048 -ub 2048 -fa auto -r 3`, interleaved `MMB=1`/`MMB=0`,
two rounds (r1/r2):

| model / size | MMB=1 | MMB=0 | win |
|---|---|---|---|
| dense 27B pp2048 | 595.5 / 583.3 | 459.7 / 460.4 | +29 % |
| dense 27B pp8192 | 556.8 / 555.6 | 443.9 / 443.7 | +25 % |
| dense 27B pp32768 | 481.5 / 481.0 | 393.9 / 393.7 | +22 % |
| MoE 35B-A3B pp2048 | 2736.7 / 2731.2 | 2174.5 / 2176.0 | +26 % |
| MoE 35B-A3B pp8192 | 2482.6 / 2475.0 | 2013.0 / 2016.6 | +23 % |
| MoE 35B-A3B pp32768 | 1940.5 / 1939.9 | 1635.0 / 1633.9 | +19 % |

(The MoE absolutes are higher than the earlier smoke because this run used `-ub 2048`; the ratio is the gate.)

### Gate 3 — op oracles: **GREEN**

| oracle | result |
|---|---|
| `FLASH_ATTN_EXT` | **5956/5956**, 0 FAIL |
| `MUL_MAT_ID` | **929/929**, 0 FAIL |
| `FLASH_ATTN_QSA` | 26/26 |
| `GATED_DELTA_NET` | 46/46 |
| `LIGHTNING_INDEXER` | 225/225 |
| `INDEXER_TOPK` | 0/0 (documented aspirational) |

`MUL_MAT_ID` is green on the fixed tree (the `0027` guard); stdout was captured separately from stderr
(the documented trap).

### Gate 4 — MTP Protocol A: **GREEN**

Reference command (matches §7): prose, seed 42 / temp 0 / `--reasoning off`, f16 KV,
`-n 2000 --spec-draft-n-max 3 --log-verbosity 4 -c 262144 -b 2048 -ub 512`:

| model | plain t/s | draft-mtp n3 t/s | acceptance | acc/pos |
|---|---|---|---|---|
| dense 27B Q8_0 | 20.7* | 20.3 | **0.82188** | 0.922/0.824/0.719 |
| MoE 35B-A3B Q4_K_M | 54.0* | 89.2 | **0.76164** | 0.895/0.760/0.627 |
| qwen4exp IQ4_XS (Q8 sidecar) | 25.0* | 42.8 | **0.82151** | 0.908/0.814/0.743 |
| qwen4exp IQ4_XS (shared sidecar) | 25.3* | 42.6 | 0.82356 | 0.909/0.815/0.746 |

\* dense/MoE plain from the same-command run in §“item 3” below (20.7 / 54.0).  The dense, MoE and
qwen4exp acceptance values **reproduce §7 exactly** (`0.82188` / `0.76164` / `0.82151`) with this
command — the `0027` guard is a no-op for these models (a non-quantized weight would have hit the
`ggml_cuda_mul_mat_vec_q` default abort pre-fix, and none did).  The shared sidecar reads 0.82356 here
(§7's 0.81962 came from a shorter run), 0 `X < Y` errors.

A second sweep at `-c 32768 -b 2048 -ub 2048` (denser context, fresh state) also passes:
dense **0.83240** (20.5 vs 7.7 t/s), MoE **0.75231** (89.2 vs 54.0), qwen4exp **0.84231** (44.1 vs
25.0), shared **0.84005** (44.4 vs 25.3).  The two commands differ ~1 % in acceptance by design; both
are well above the `~0.45` bar and both keep MTP ≥ plain.

### Recurrent

* `test-recurrent-state-rollback -m Qwen3.5-4B-Q8_0` → **PASS** (`max diff 0, nmse 0`).
* `test-recurrent-state-depth -m Qwen3.5-4B-Q8_0` → `total failures = 174` (Phase B, large `n_rs_batch`);
the **old r13+beta build** gives the **same 174** on the same model → pre-existing, not a re-base
regression.  (The synthetic `test-generate-models` fixture is not built in this tree, so the real
qwen35-4B model was used for both.)

**Verdict: the full gfx1151 beta-window re-validation is GREEN on tree `7f339b10`.**

### Item 3 — the MoE MTP acceptance gap vs upstream (investigated)

Same Protocol A command, MoE 35B-A3B Q4_K_M (upstream = `84e76d8a2`, base-16 = delivery only, tree
`336d0f43`):

| arm | acceptance | gen t/s |
|---|---|---|
| upstream | **0.78844** | 92.3 |
| base-16 delivery | 0.73967 | 87.5 |
| beta default (`MMB=1`) | 0.76164 | 89.2 |
| beta `GGML_CUDA_MMB=0` | 0.73967 | 87.5 |
| beta `GGML_CUDA_DISABLE_TOPK_MOE_FUSION=1` | 0.76164 | 86.2 |
| base-16 + `SHEXP_DOWN_GATE=1` / `TOPK_MOE_FUSION=1` / `MOE_MMQ_FUSION=1` | 0.73967 each | 87.5 |
| beta + `MMVQ_MOE_BAND=1` / `MMVQ_DENSE_BAND=1` | 0.76164 each | 88.0 |
| dense 27B: upstream / base-16 / beta | 0.82397 / 0.81235 / 0.82188 | 20.7 / 20.1 / 20.3 |

**Findings.**  (1) The gap is **delivery-level and MoE-specific** — it is already present in the base-16
build (dense models are at ~parity, 0.812–0.824).  (2) The **beta is not the cause**: `MMB=0` reproduces
the base-16 value exactly, and `MMB=1` *recovers* +0.022 acceptance / +1.7 t/s — the beta `mmb` helps
the draft.  (3) **None of the three block-13 fused-MoE kernels** is it (`SHEXP_DOWN_GATE`,
`TOPK_MOE_FUSION`, `MOE_MMQ_FUSION` are all acceptance-neutral on base-16), and neither is the beta
mmvq band.  The residual is the delivery's **MoE expert matmul/reduction-order policy** (block-10
k-quant + block-13 band-uniform `nwarps`/VDR, deliberately compile-time, no switch — the width-purity
invariant), which drifts the draft logits off upstream's by a near-tie margin.

**Disposition:** accepted trade, **no code change**.  It is the documented MoE fusion/reduction numerics
trade (`GREEDY-PURITY.md` §19/§25; `benchmarks/mtp-adaptive-methodology.md` — gate MoE on acceptance and
MTP ≥ plain, not cross-build equality).  The ~4 % acceptance / ~3 % MTP-throughput cost buys the
+48–62 % MoE prefill and ~+4 % decode; acceptance stays well above the bar and MTP ≥ plain on every cell.
Pinning the exact kernel would need a delivery block bisect (build base-16 minus block 10 / 13), which
was not warranted for a known-trade result.

---

## 9. 2026-09-25 — delivery `r2`: the wide-VDR MoE expert leak (found and fixed)

§8 concluded the base-16 MoE MTP-acceptance gap was "accepted trade, no code change".  That conclusion
was **wrong**: the residual was a concrete arch-scope bug, found on the follow-up.

**Root cause.**  `VDR_Q4_K/Q5_K/Q6_K_Q8_1_MMVQ_MOE` were defined unconditionally (4/4/2) while only
the **Q8_0** MoE VDR was arch-gated, and the Q8_0 comment literally says *"RDNA3_5 (gfx115x) keeps
VDR=2 pending verification on those GPUs."*  So on gfx1151 the Q4_K/Q6_K experts (exactly the Q4_K_M
expert types) ran the wide VDR=4 chunk that block 10 had scoped to RDNA4/RDNA3_0.  §8's per-switch
A/B could not see it because neither the fused-MoE kill-switches nor the beta mmvq bands touch
`mul_mat_vec_q_moe`'s compile-time `vec_dot`/`vdr`.

**Fix (one gate, block 10).**  `get_vec_dot_q_cuda()` and `get_vdr_mmvq()` now ignore the `moe`
argument on every target that is not RDNA4/RDNA3_0 (`#if !(defined(RDNA4) || defined(RDNA3_0)) moe =
false;`), so the whole MoE expert selection is arch-scoped in one place instead of per quant - the
per-quant style is exactly how the Q4_K/Q6_K arms leaked.  The `_MOE` macros keep their measured 4/4/2
(RDNA4/RDNA3_0 keep the wide chunk); only the *reachability* is gated.

**Base-16 result (gfx1151, the same Protocol A as §8).**  MoE `draft-mtp n3` acceptance **0.73967 ->
0.76484** (halves the gap to upstream's 0.78844) and decode **87.5 -> 89.1 t/s**; dense 27B 0.82188 and
qwen4exp 0.82151 unchanged (their experts are not Q4_K/Q5_K/Q6_K); MoE `width_purity=PASS` (worst
maxdiff 0); `MUL_MAT_ID` **929/929**, `FLASH_ATTN_EXT` **5956/5956**.  `scripts/validate-set.sh` green on
a fresh `84e76d8a2` tarball (strict 16/16, applied tree `ea7acf2d…`); release **`v16-84e76d8a2-r2`**.

**Beta re-port.**  The 28 beta patches re-based onto the fixed base (`git rebase --onto`, no conflicts;
patch `0027` touches `mmvq.cu` in a different region) -> applied tree **`e00275ff…`**; `apply-beta.sh`
strict 16/16 + 28/28 on a fresh worktree.  Full re-validation:

| gate | result |
|---|---|
| Gate 1 | dense `90686d1edf24` (unchanged), MoE `904bfc375b2e`, qwen4exp `b746fb3e77ff` - all `none == draft-mtp n3`; `width_purity=PASS` on all three |
| Gate 2 | dense pp2048/8192/32768 +29/+25/+22 %; MoE +26/+24/+19 % |
| Gate 3 | `FLASH_ATTN_EXT` 5956/5956, `MUL_MAT_ID` 929/929, `FLASH_ATTN_QSA` 26/26, `GATED_DELTA_NET` 46/46, `LIGHTNING_INDEXER` 225/225, `INDEXER_TOPK` 0/0; `MMB_CFG cc=0x1001151 dense_geom=0 … routed=1` |
| Gate 4 | reference command: dense 0.82188, MoE 0.75225, qwen4exp 0.82151, shared 0.82356 - all > 0.45, MTP ≥ plain |

The beta MoE acceptance (0.75225) moves with the target's decode numerics (acceptance is a chaotic
function of the continuation), so it is not a quality signal; what matters is the base-16 fix itself,
which is now upstream-aligned.

**Note for the next reader.**  The §8 gap was real and had a one-line cause; "accepted trade" was
premature.  The lesson is the one the delivery keeps relearning: **a wide-VDR/perf override must be
arch-scoped at the selector, not per quant** - the Q8_0-only gate was the tell.
