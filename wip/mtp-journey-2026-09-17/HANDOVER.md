# HANDOVER — retuning the adaptive-MTP controller (2026-09-17)

**Status: investigation complete. Recommendation: do NOT change block 01.**  Nothing pushed. Read
§0 and §6 for the conclusions; §2-§5 are the evidence, §7 the remaining work.

---

## 0. TL;DR

The delivery's **credit-bucket** adaptive-MTP controller (`common/speculative-adaptive.h`, block 01)
was suspected of being overfit to the original 4-axis gate (one prompt per axis, 2-GPU Q8_0 cell).
It is not: the suspicion came from a **stale record pointer** (the 2026-09-13 four-axis record was
measured with the pre-tuning *table* controller; `benchmarks/README.md:23` and
`benchmarks/mtp-adaptive-methodology.md:202` still call it current).

Work done this session:

* Built a **4-prompts-per-axis corpus** (16 + a phase-switch guard) and a **sliding-mean** and a
  **target-acceptance-rate** controller in the scratch lab.
* Established the **multi-cell ground state** (dense 1 GPU, MoE 1 GPU, Q8_0 2 GPU tensor, Q8_0 3 GPU
  tensor, phase-switch).
* Tuned the bucket and both alternative controllers against the corpus.
* **Result: no constant set is a Pareto improvement across the cells.** The best retune (`ref_d`,
  `drop 250/40` + `climb 10/3`) wins uniformly on Q4_K_XL dense-1-GPU (+4.8 %, all 16 prompts) and is
  MoE-neutral, but **loses Q8_0 prose on every configuration** (1 GPU 0.960, 2 GPU tensor 0.936,
  2 GPU layer 0.950, 3 GPU tensor 0.914). Every conservative variant trades the same way.
* **A GPU-mode split was proposed and rejected.**  The prose response tracks the **weight
  quantization (Q8_0 vs Q4_K_XL)**, not the split and not the AllReduce (`-sm layer` shows it too);
  Q8_0 is the server's quant, so "ref_d on 1 GPU, base on 2+" does not protect it.  The one
  phase-switch objection was a single prompt's content (two new mixed prompts improve, geo 1.008).
* The **target-acceptance-rate controller is dominated** on the full corpus (dense 1.048 vs the
  retuned bucket's 1.061; MoE 0.972 vs 1.033), for a structural reason (§6.3).

**The delivery's base bucket is a good multi-cell compromise. No block-01 change is recommended.**

---

## 1. The question

`--spec-type draft-mtp-adaptive` picks the draft depth per workload. The delivery replaced the
upstream PR #27210 table controller with a credit bucket and tuned it on one prompt per axis on the
2-GPU Q8_0 cell (issue #35). Was that overfit? The old table is still selectable in the scratch build
(`GGML_ADAPTIVE_TABLE=1`), so every comparison is controller-vs-controller on the same kernels.

---

## 2. Environment & scratch builds

3x R9700 (`gfx1201`), ROCm 7.14 (`/opt/rocm-7.14-gfx1201`), 16 threads.
Export `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`.  1 GPU = `HIP_VISIBLE_DEVICES=0`;
2 GPU tensor = `1,2 -sm tensor -ts 1/1`; 3 GPU tensor = `0,1,2 -sm tensor`.

| label | path | commit | controller |
|---|---|---|---|
| stock | `/home/stew675/stock-9113/build-rocm/bin/llama-cli` | `9113cc188` | none (static) |
| pr | `/home/stew675/pr27210/build-rocm/bin/llama-cli` | `d236d41a2` | table (PR #27210 head) |
| deliv | `/home/stew675/llama.cpp/build-rocm-current/bin/llama-cli` | `31b179037` | credit bucket (delivery) |
| **deliv-ab** | `/home/stew675/deliv-ab/build-rocm/bin/llama-cli` | `31b179037` | **bucket + table + mean + target, env-switched** |

Rebuild after editing `deliv-ab/common/speculative-adaptive.h`:

```bash
/tmp/build-arm.sh /home/stew675/deliv-ab build-rocm      # ~6 min, llama-cli only
```

Models: dense `/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf` (1 GPU);
MoE `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (1 GPU);
Q8_0 `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (2 or 3 GPU tensor).

---

## 3. The prompt corpus

`wip/mtp-journey-2026-09-17/corpus/`, 4 per axis, hash-frozen (`MANIFEST.md`): reasoning `r1`-`r4`
(`--reasoning on`), prose `p1`-`p4` (`p1` = repo docs, `p2`-`p4` wikitext), code `c1`-`c4`,
recall `k1`-`k4`; plus `phase-switch.txt` (the code→reasoning guard).  Never edit in place.

---

## 4. Controller modes and env knobs (deliv-ab only)

| env var | default | effect |
|---|---|---|
| `GGML_ADAPTIVE_TABLE=1` | unset | upstream **table** controller |
| `GGML_ADAPTIVE_MEAN=1` | unset | **sliding-mean**: `n_cur = ceil(mean acc) + offset` |
| `GGML_ADAPTIVE_TARGET=1` | unset | **target-rate**: step toward `p*` |
| `GGML_MTP_DROP_FLOOR` / `_DROP_SLOPE` | 60 / 10 | bucket `drop_pressure = max(FLOOR, SLOPE*d)` |
| `GGML_MTP_CLIMB_BASE` / `_CLIMB_SLOPE` | 20 / 6 | bucket `climb_budget = BASE + SLOPE*(d-1)` |
| `GGML_MTP_COLD_START` | 3 | bucket/mean/target: start at `cap - N` |
| `GGML_MTP_CREDIT_MINUS` | 1 | bucket: full accept credits `max(1, n_acc - N)` |
| `GGML_MTP_MEAN_WINDOW` / `_OFFSET` / `_PREFILL` | 20 / 1 / cap | mean knobs (`cap`/`cold`/`floor`/`none`) |
| `GGML_MTP_TARGET_PCT` / `_WINDOW` | 75 / 16 | target set point and window |
| `GGML_MTP_TARGET_DEADBAND_PCT` / `_STRIDE` / `_PREFILL` | 3 / 1 / cap | target anti-hunting and prefill |

`--spec-draft-n-max 12` in all corpus runs; `n_cur` clamps to `[n-min-adaptive, n-max]`.

---

## 5. Results

### 5.1 Ground state, table vs the delivery's base bucket (1 GPU)

Per-axis geometric mean of bucket/table (raw: `raw/ground.tsv`, `raw/ground-moe.tsv`):

| axis | dense 27B | MoE 35B-A3B |
|---|---:|---:|
| reasoning | 0.997 | 1.090 |
| prose | 0.949 | 0.943 |
| code | 0.930 | 1.089 |
| recall | 1.192 | 1.014 |
| overall | 1.012 | 1.032 |

The base bucket beats the table on both model classes and massively on q8t2 (code +14 %, prose +22 %,
from `raw/canonical.tsv`).  Its one real weakness is **dense-1-GPU prose/code**.

### 5.2 Full-16 validation of the leading controllers (ratio vs table)

| controller | dense overall | dense worst axis | reason | prose | code | recall |
|---|---:|---:|---:|---:|---:|---:|
| base bucket | 1.012 | 0.930 | 0.997 | 0.949 | 0.930 | 1.192 |
| **`ref_d`** `drop 250/40`+`climb 10/3` | **1.061** | **0.992** | 1.030 | 1.024 | 0.992 | 1.211 |
| target `t50w32` | 1.048 | 0.945 | 0.945 | 1.004 | 1.013 | 1.257 |
| target `t55w64` | 1.042 | 0.958 | 0.958 | 0.976 | 1.004 | 1.258 |

MoE overall: `ref_d` 1.033, `t50w32` 0.972, `t55w64` 0.987.

### 5.3 The multi-cell Pareto check — `ref_d` vs the base bucket

| cell | prose | code | reasoning | recall |
|---|---:|---:|---:|---:|
| **Q4_K_XL** dense 1 GPU | **1.079** | 1.067 | 1.033 | 1.016 |
| Q4_K_M MoE 1 GPU | 0.981 | 0.991 | 1.000 | 1.033 |
| **Q8_0** 1 GPU | **0.960** | 1.009 | -- | -- |
| **Q8_0** 2 GPU tensor | **0.951** | 1.013 | 0.985 | 1.004 |
| **Q8_0** 2 GPU layer (no AR) | **0.950** | -- | -- | -- |
| **Q8_0** 3 GPU tensor | **0.914** | 1.017 | 1.008 | 1.020 |
| dense phase-switch guard (3 prompts) | 1.008 | | | |

`ref_d` improves **all 16** Q4_K_XL dense prompts (min +0.3 %) and all Q8_0 code/recall cells, but
**loses Q8_0 prose on every configuration tried** -- 1 GPU 0.960, 2 GPU tensor 0.936, 2 GPU layer
0.950, 3 GPU tensor 0.914.  The phase-switch number is the geometric mean of 3 mixed prompts (the
original guard alone was 0.969; two new ones are 1.039 and 1.016).  Conservative variants
(ratios vs base):

| candidate | phase-switch | q8t2 prose | q8t2 code | MoE prose | MoE code |
|---|---:|---:|---:|---:|---:|
| `climbonly` `climb 10/3` | 0.991 | 0.973 | 0.973 | **1.034** | **1.017** |
| `drop120` `drop 120/20` | 0.973 | 0.978 | 0.995 | 0.987 | 1.009 |
| `climb12` `climb 12/4` | 0.984 | 0.986 | 0.980 | **1.033** | **1.025** |
| `hi2` `drop 200/30` | 0.957 | 0.996 | 1.001 | 1.006 | 0.998 |
| `hi3` `drop 200/30`+`climb 10/3` | 0.982 | 0.960 | 1.014 | 1.013 | 0.966 |

### 5.4 The target-rate diagnostic (why it loses)

Pooled acceptance `p` at the operating depth, from the ground logs:

| prompt | table p | table depth | base bucket p | base depth |
|---|---:|---:|---:|---:|
| c4 code | 0.536 | 10.1 | 0.644 | 7.4 |
| c1 code | 0.581 | 11.9 | 0.663 | 10.1 |
| p1 prose | 0.510 | 10.9 | 0.602 | 8.5 |
| r1 reasoning | 0.587 | **4.9** | 0.568 | 5.3 |
| k1 recall | 0.937 | 9.4 | 0.960 | 11.6 |

### 5.5 Pinned-depth oracle and the credit sweeps

`c4` dense pinned 6/8/10/12 = 54.27 / **69.71** / 67.60 / 67.84 t/s; `p1` = 51.93 / 69.07 / 66.62 /
70.87.  The base bucket under-drafts `c4` (it sinks to 5-6).  Raising `drop_pressure` and lowering
the climb budget (`ref_d`) holds it deep and is the dense win — at the multi-GPU cost above.

---

## 6. Findings (do not re-derive)

1. The base bucket **beats the table** on dense (1.012), MoE (1.032) and q8t2 (code +14 %, prose
   +22 %).  Its "underperformance" was the **stale 2026-09-13 pointer**, not the constants.
2. The **dense code deficit (−7 %, worst −18 % on `c4`) is real** but is fixed by `ref_d`, which then
   costs the multi-GPU prose cells.
3. The **target / mean / rate-style controllers are all dominated**: they share one structural flaw —
   a single content-independent statistic cannot serve reasoning (floor) and code (deep) at once.
   For the target rule the inversion is explicit: reasoning's **floor** rate 0.587 > code's **optimum**
   rate 0.536.  The table escapes this only because its per-depth thresholds approximate the
   *marginal* full-accept probability.
4. **No tested constant set is a Pareto improvement.**  `ref_d`'s win/loss tracks the **weight
   quantization**, not the GPU configuration: it wins Q4_K_XL prose (1.079) and loses Q8_0 prose on
   1 GPU (0.960), 2 GPU tensor (0.936), 2 GPU layer (0.950) and 3 GPU tensor (0.914).  Since `-sm
   layer` removes the AllReduce, it is not the AR numerics either.  **A GPU-mode split would not
   protect Q8_0** -- and Q8_0 is the maintainer's server quant.  The only split the data supports is
   per-quant, which is unprincipled and breaks the one-controller property.  Keep the base.
5. MTP t/s is **content-sensitive** (runs stop at EOS at different token counts; the delivery's
   block-10 numerics change the greedy text): compare distributions and per-axis ratios, never one
   prompt.  The 5-prompt sweep's "win" for the target controller (1.038 vs 1.026) reversed on the
   full 16 — that is the overfit, reproduced live.

---

## 7. Remaining work

**See `SUMMARY.md` for the consolidated final report.**  The controller work closed as follows.

0. **Ship the `ngram-mod` + adaptive-MTP combo as a documented opt-in** (recall +67 %, overall +13.6 %
   on Q8_0 2-GPU; reasoning ~-2 %): `--spec-type draft-mtp-adaptive,ngram-mod --spec-ngram-mod-n-match 45
   --spec-draft-n-max 9 --spec-draft-n-start 9`.  **Cap guidance (corrected):** single card -> keep the
   cap ~9 (cap 6 costs 11-14 % on code); multi-GPU -> 6-7 is mildly better.  `n_match` should be >= 40
   and an integer multiple of the cap.

1. **No controller change, and no GPU-mode split.**  The GPU-mode split was proposed and tested: it
   does not work, because the loss is the weight quantization (Q8_0) and appears on 1 GPU too.  The
   only candidate that wins anywhere is `ref_d` (`DROP_FLOOR 250`, `DROP_SLOPE 40`, `CLIMB_BASE 10`,
   `CLIMB_SLOPE 3`) for Q4_K_XL-class quants; it was **not** landed, since it would regress the
   server's Q8_0 prose by 4-9 %.  If the maintainer wants it anyway, gate it on the quant, not the
   split -- but it is not recommended.
2. **Fix the stale pointers** (the actual defect): `benchmarks/README.md:23` and
   `benchmarks/mtp-adaptive-methodology.md:202` should stop calling the 2026-09-13 record current and
   point at `benchmarks/2026-09-15-adaptive-mtp-tuning.md` (or a new dated record).  Delivery doc
   change — needs the maintainer's go-ahead.
3. **Optionally** promote the 4x4 corpus to `prompts/` (hash-frozen) as the controller gate, and add
   the multi-cell (1/2/3 GPU) requirement to `benchmarks/mtp-adaptive-methodology.md`.
4. **Wiki** — see §10.
5. Do **not** push; the `deliv-ab` env knobs are a lab, not delivery code.

---

## 8. Gotchas

* **`llama-cli` always with `--single-turn`** (and `--no-display-prompt`); wrap in `timeout`.
  **Never run benches in parallel.**
* Canonical flags: `-n 3000 --seed 42 --temp 0 --single-turn --no-display-prompt -c 32768 -b 2048
  -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4`; `--reasoning on` for `r*`, off otherwise.
* Parse the generation rate from the `eval time =` line that is *not* `prompt eval time =`; acceptance
  from `draft acceptance = X (a / g), mean len = Y`.  `-lv 4` for acceptance; no `-lv 4` for
  `scripts/extract-generated.py`.
* Greedy is deterministic, so repeats reproduce — the spread in these tables is systematic
  (controller/content), not noise.
* Depth transitions log as `adaptive draft depth seq 0: N -> M (n_draft=.., n_accepted=.., n_bucket=..)`.

---

## 9. Artifacts

All under `wip/mtp-journey-2026-09-17/` (untracked, unpushed).  `README.md` = the full narrative;
this file = the snapshot.

| path | what |
|---|---|
| `corpus/` | the 4x4 corpus + `MANIFEST.md` (sha256) |
| `raw/ground.tsv`, `raw/ground-moe.tsv` | dense / MoE table-vs-base ground states |
| `raw/target-valid.tsv`, `raw/target-valid-moe.tsv` | dense / MoE full-16 for `ref_d`, `t50w32`, `t55w64` |
| `raw/canonical.tsv` | three-arm journey (stock/pr/deliv) incl. q8t2 |
| `raw/*.py` | harnesses: `ground*`, `matrix`, `extra`, `pinned`, `ab`, `prompt_ab*`, `tune1..tune15`, `analyze` |
| `raw/ground-logs/`, `raw/ground-moe-logs/`, `raw/ab-logs/`, `raw/prose-ab-logs/`, `raw/tune-logs/`, `raw/valid-logs/` | all run logs (`tune9_*` in `tune-logs`, the rest in `valid-logs`) |

Scratch worktrees (outside the repo): `/home/stew675/deliv-ab` (lab),
`/home/stew675/pr27210`, `/home/stew675/stock-9113`.

---

## 10. Pending non-controller items

* **Stale record pointers** — see §7.2 (the real bug behind this whole investigation).
* **Wiki**: local commit `856c73e` added `wiki/` (Home, MTP & Adaptive MTP, Quick Reference,
  `_Sidebar`).  Publishing is blocked until the GitHub wiki is initialised by creating one page in the
  web UI at <https://github.com/stew675/llama-cpp-rdna-boosts/wiki>; then
  `git clone git@github.com:stew675/llama-cpp-rdna-boosts.wiki.git` and copy the files in.  Update the
  controller wording there to match §0 (bucket is the compromise; the "underperformance" was the
  stale pointer).

## 11. Git state

* `git log -1` = `856c73e docs: add GitHub wiki source…` (local only, **not pushed**).
* `git status` = only `?? wip/mtp-journey-2026-09-17/`.
* Nothing pushed anywhere.
