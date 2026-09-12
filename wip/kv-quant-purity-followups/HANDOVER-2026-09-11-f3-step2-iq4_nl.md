# HANDOVER — F3 step 2: `iq4_nl` as a first-class KV cache type

**Point the next session at this file.**  It is self-contained for this task.  The shared
infrastructure (environment, models, probe, harnesses, landing procedure, trap list) is in
**`HANDOVER-2026-09-11-remaining-work.md`** (§2-§5, §12-§13)  and the *immediately preceding* briefs are
**`HANDOVER-2026-09-11-f3-kv-diagonals.md`** (§10 = F3 step 1, the shape this task repeats) and
**`HANDOVER-2026-09-11-qsa-quantized-kv.md`** (§9 = the fourth block-14 amendment, where the CPU
oracle and the perplexity gate came from — **read its §9 before starting**, this task must satisfy the
same gates).

## 1. Mission

Make `iq4_nl` a first-class KV cache type, exactly the way step 1 did for `q4_1`/`q5_0`/`q5_1`:

* **dense flash attention** (`GGML_OP_FLASH_ATTN_EXT`) must accept it — `iq4_nl` currently produces
  **no FA call at all** (the FA probe then disables FA for the whole context, so attention runs the
  slow non-FA path);
* **the fused sparse QSA kernel** (`GGML_OP_FLASH_ATTN_QSA`, qwen4exp) must read it natively, like the
  four types enabled on 2026-09-11 (9);
* every type list that decides either of those must end up consistent (that is the actual work — see
  §4), and the new paths must pass the same gates the four previous types passed, **plus** the two
  oracles that this repo now has (CPU reference + perplexity-against-dense).

`iq4_nl` is the **last** sub-`q8_0` type on the list: it is the smallest KV cache this repo can offer
(§2), and after this there is no further KV-type work in F3.

## 2. Why — the measured prize (measured 2026-09-11, canonical `a0cd6ce02`)

**Perf (the step-1 shape).**  4B, 1 GPU, `llama-bench -p 512 -n 32 -r 2 -fa 1 -ngl 99 -sm none`:

| `-ctk` | pp512 | tg32 | path |
|---|---|---|---|
| **`iq4_nl` (today)** | **2269.83** | **48.48** | non-FA (FA disabled for the context) |
| `q4_0` (step-1-enabled) | 7770.96 | 96.67 | TILE FA with f16 staging |
| f16 | 7810.71 | 99.45 | TILE FA native |

So the target is the same **~3.4x prefill / ~2x decode** that step 1 measured for `q4_1`
(2119.6/55.94 -> 7366.3/94.16) — `iq4_nl`'s pre-state numbers are nearly identical to `q4_1`'s were,
so expect to land at ~7700/95 t/s on the 4B.  **Measure your own pre-state first** (this table is the
"before"; a `-ctk iq4_nl` run today is *slow but works*, which is the tell that FA is off).

**Memory (the real prize — `iq4_nl` is the smallest cache type).**  4B, `c=32768`, ub 512,
`llama_kv_cache: ... KV buffer size`:

| type | KV buffer | vs f16 |
|---|---|---|
| **`iq4_nl`** | **288.00 MiB** | **-72 %** |
| `q4_0` | 288.00 MiB | -72 % |
| `q4_1` | 320.00 MiB | -69 % |
| `q5_0` | 352.00 MiB | -66 % |
| `q5_1` | 384.00 MiB | -63 % |
| `q8_0` | 544.00 MiB | -47 % |
| f16 / bf16 | 1024.00 MiB | — |

`iq4_nl` ties `q4_0` for the smallest cache and is the last type that can be turned on without any
further work (`iq4_xs`/k-quants need a different block layout and are **not** in scope).

**Correctness (the reason this brief is written the way it is).**  The QSA kernel that serves
qwen4exp's default attention path had **no oracle anywhere** until the 2026-09-11 (9) session; the
perplexity-vs-dense gate it added immediately found a ~12 % quality bug (a shared smem staging tile
mixing two K/V heads).  Enabling a *new* type on that kernel without running those oracles would
repeat exactly the mistake that cost that session.  Treat `test-backend-ops -o FLASH_ATTN_QSA` and
`qsa-ppl-oracle.sh` as mandatory gates, not as extras.

## 3. State you are starting from

* **Canonical fork**: `/tmp/canon-llama`, branch **`rdna-boosts`**, tip **`a0cd6ce02`**, net tree
  **`0966e66731a4c3da85ffd96525688865a89242cd`**, 15 blocks (00-14), clean, `build-base` built and
  instrument-free.  Block SHAs: 00 `1c7ab0e89`, 01 `aa4108b9d`, 02 `6e81ed5ed`, 03 `4dc962aa9`,
  04 `03d004517`, 05 `70f330aed`, 06 `d2fc2cb34`, 07 `110b5391d`, **08 `1a488fcf0`**, 09 `43cc5f850`,
  10 `f5236f3ee`, 11 `c71a657b0`, 12 `e008f9914`, 13 `1a88c92f5`, **14 `a0cd6ce02`**.
* **Delivery repo**: `~/llama-cpp-rdna-boosts`, `main` = **`7412410`** == `origin/main`, 15 patches
  `0000`-`0014`, `rdna-boosts-all.patch` = 21 750 lines, `make-patches.sh` default tip `a0cd6ce02`.
* **Block 15 beta (7th re-cut)**: base `a0cd6ce02` -> beta commit **`5a0734c9d`**, tree
  **`6b1155b68b1741d7e7c6e8f80b88ed90ce406bd6`**, patch **3 787 lines**, branch `blk15-f3` in the
  canonical fork (`/tmp/blk15q` worktree holds it, `build-beta` built there).  **Any block amendment
  invalidates it -> re-cut it** (§8), and note the beta carries a *second copy* of the QSA type gate
  (`qwen4exp_qsa_sparse()` in `src/models/qwen4exp.cpp`) that must be extended in lockstep (§4f).
* **Owner**: `ggml/src/ggml-cuda/fattn.cu` is **block 08**'s (the F3 step-1 enablement lives there);
  `ggml/src/ggml-cuda/fattn-qsa.{cu,cuh}`, `ggml/src/ggml-cpu/ops.cpp`, `src/models/qwen4exp.cpp` and
  `tests/test-backend-ops.cpp`'s QSA test are **block 14**'s; `src/llama-context.cpp`'s
  `llama_kv_type_has_native_fa()` sits in block 14.  So this task = **one block-08 amendment + one
  block-14 amendment** (or one rebase pass marking both `edit`).  `tests/test-backend-ops.cpp`'s FA
  matrix is block-04 territory (the WMMA/MTP band) — prefer loading the new FA case there or with the
  QSA test, whichever is closest to the code you touch.
* Nothing under `wip/` may be folded into `patches/`; the beta is re-cut, never merged.

## 4. What is VERIFIED about the mechanism (checked against `a0cd6ce02`; read before designing)

**a. The dense FA path *stages* every type `ggml_get_to_fp16_cuda` covers — and that already includes
`IQ4_NL`.**  `ggml/src/ggml-cuda/convert.cu:515` `ggml_get_to_fp16_cuda` lists `GGML_TYPE_IQ4_NL`
among its cases, so the TILE/MMA families (which stage K/V through it, `need_f16_K/V = 1`) can consume
an `iq4_nl` cache with **no new kernel** — exactly the step-1 finding for `q4_1`.  This is why the
dense half of this task is predicate/instance bookkeeping, not kernel work.

**b. `IQ4_NL` appears ZERO times in `ggml/src/ggml-cuda/fattn.cu`.**  So today the dense path rejects it
in three independent places, all of which must be updated:
* `ggml_cuda_fattn_kv_type_supported()` (`fattn.cu:488`, the `default: return false` clause) — add
  `case GGML_TYPE_IQ4_NL: return true;` with a comment, *and* update the "keep in sync" comment block
  above it;
* the `#else` branch of `ggml_cuda_flash_attn_ext_vec()` (`fattn.cu:453-463`, the default diagonal
  list) — add `FATTN_VEC_CASES_ALL_D(GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL)`;
* the `#ifdef GGML_CUDA_FA_ALL_QUANTS` branch above it is **not** touched (step 1 left the mixed
  `K != V` pairs to the flag, and `TYPES_KV` in the generator does not carry `iq4_nl`, so there are no
  cross-pair instances to keep consistent — see (c)).
* `src/llama-context.cpp:3681` `llama_kv_type_has_native_fa()` — add `case GGML_TYPE_IQ4_NL:` so
  `-sm tensor` stops rejecting the type with "not implemented" (the function's own comment says it must
  mirror `ggml_cuda_fattn_kv_type_supported()`).

**c. There is NO `fattn-vec-instance-iq4_nl-*.cu`.**  `ggml/src/ggml-cuda/template-instances/` holds
the **49-file cross product of the upstream `TYPES_KV`** (`F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16`),
and the backend CMakeLists compile the **7 diagonals** by default (plus all 49 under
`GGML_CUDA_FA_ALL_QUANTS`).  `iq4_nl` is not in `TYPES_KV`
(`template-instances/generate_cu_files.py:11`), so:
* create `template-instances/fattn-vec-instance-iq4_nl-iq4_nl.cu` **by hand** using the generator's
  exact template text (see any diagonal file, e.g. `fattn-vec-instance-q4_1-q4_1.cu`: a 3-line
  `DECL_FATTN_VEC_CASE(64/128/256, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL)` block with the
  "autogenerated ... do not edit manually" header);
* add it to the **default** branch of `ggml-{cuda,hip,musa}/CMakeLists.txt` (the same 6-line block
  step 1 extended);
* **do not run `generate_cu_files.py`**: it `os.remove`s every `*.cu` in the directory and rewrites
  the full cross product from `TYPES_KV` — running it would (i) delete the hand-made file and (ii)
  emit 15 new `iq4_nl` cross pairs nobody asked for.  If you ever *do* want the generator to be the
  source of truth for `iq4_nl`, extend `TYPES_KV` deliberately and re-run it for the whole directory
  as a separate, explicit change (and expect the patch to grow by 15 files).

**d. The in-repo dequant building blocks (for the QSA kernel, and for the vec kernel).**  The seven
`dequantize_V_*` helpers live in `ggml/src/ggml-cuda/fattn-common.cuh` (`typedef` at :375, f16 :378,
bf16 :396, q4_0 :409, q4_1 :448, q5_0 :488, q5_1 :538, q8_0 :588) and stop at `q8_0` — there is **no**
`dequantize_V_iq4_nl` anywhere in the tree.  You need one, templated `<T, ne>` with
`ne == 2 || ne == 4`, writing `half2` for `T = half` like its siblings, and wired into
`get_dequantize_V<type_V, T, ne>()` (`fattn-common.cuh:643`, whose final branch is
`static_assert(type_V == -1, "bad type")` — add the `IQ4_NL` arm before it).
* **The reference semantics** are `dequantize_row_iq4_nl` (`ggml/src/ggml-quants.c:2725`):
  `y[j] = d * kvalues_iq4nl[qs[j] & 0xf]` and `y[j + QK4_NL/2] = d * kvalues_iq4nl[qs[j] >> 4]` — i.e.
  the **same nibble layout as `q4_0`/`q5_0`** (nibble `j` carries elements `j` and `j + 16`; element
  `e` is the low nibble of `qs[e % 16]` for `e < 16` and the high nibble of `qs[e - 16]` otherwise).
  **`dequantize_V_q4_0`'s index arithmetic is the template to copy** (`iqs = i0 % (QK/2)`,
  `shift = (i0 % QK) / (QK/2)`) — do not invent a `2j`/`2j+1` mapping.
* **`iq4_nl` is a table lookup, not `q - 8`:** the 4-bit index selects a value from `kvalues_iq4nl`,
  which is `{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}` (int8, no
  offset).  So `d * kvalues_iq4nl[idx]` — **there is no `- 8` / `- 16` term** (q4_0/q5_0 have one,
  `iq4_nl` does not).
* **`kvalues_iq4nl` is device-visible**: `ggml/src/ggml-common.h:1120`
  (`GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)`, a `static const` table for CUDA), and CUDA code
  already uses it — `vecdotq.cuh:1591`, `mmq-load-tiles.cuh:1459`, `cpy-utils.cuh:175`.  For a
  vectorized lookup there is `get_int_from_table_16(const int & q4, const int8_t * table)`
  (`vecdotq.cuh:34`) returning an `int2` (even-index bytes in `.x`, odd in `.y`, with an AMD
  `__builtin_amdgcn_perm` fast path); for a first cut, scalar `kvalues_iq4nl[idx]` lookups converted to
  `half2` are enough and match the style of the existing helpers.

**e. The QSA kernel side is a 5-line change once the helper exists.**  In `fattn-qsa.cu` (all block 14):
* `constexpr bool kv_dequant_f16` (the alias list introduced by the (9) session) — add
  `type_KV == GGML_TYPE_IQ4_NL`;
* the staging arm already calls `get_dequantize_V<type_KV, half, 4>()` and dequantizes 4 elements per
  call, 8 calls per 32-element block — `iq4_nl`'s block is **also 32 elements**
  (`QK4_NL == 32`, `block_iq4_nl { ggml_half d; uint8_t qs[16]; }`), so no arithmetic changes;
* the dispatch switch in `ggml_cuda_flash_attn_qsa()` and
  `ggml_cuda_flash_attn_qsa_kv_type_supported()` / `ggml_cuda_flash_attn_qsa_supported()`;
* `src/models/qwen4exp.cpp:1307` `qsa_kv_native` (the graph gate — **must** stay in lockstep with the
  backend predicate: an unsupported QSA op is not split under `-sm tensor` and the meta splitter
  aborts on `attn_gated`);
* the CPU reference `ggml/src/ggml-cpu/ops.cpp`'s `read_kv` (the `ggml_compute_forward_flash_attn_qsa_f32`
  lambda, ~:9418) — add the `GGML_TYPE_IQ4_NL` case with the table lookup (this is the oracle the new
  `FLASH_ATTN_QSA` test compares against; **the test cannot pass without it**);
* and, in the beta only, the same conjunct inside `qwen4exp_qsa_sparse()` (§8).

**f. Two type lists exist for QSA and must both be extended** (`qsa_kv_native` in the delivery and
`qwen4exp_qsa_sparse()` in the beta) — the (9) session hit this as a merge conflict.  In the delivery
repo there is only one list; the beta's copy is why the re-cut needs a real merge.

**g. Today's pre-state (measured, so you can recognise the symptom):** `-ctk iq4_nl` *loads* and runs
on 1 GPU / `-sm layer` (`-sm tensor` is rejected by `llama_kv_type_has_native_fa()`), but generates at
**19.1 t/s** on qwen4exp because FA is disabled for the whole context — the exact step-1 symptom.

## 5. The plan

1. **Measure the pre-state** (4B pp512/tg32 = §2's table; qwen4exp/qwen4exp-q4_1 as the reference
   shape) so the win is a delta, not a claim.
2. **Dense FA** (block 08): `dequantize_V_iq4_nl` in `fattn-common.cuh` + the `get_dequantize_V` arm +
   `ggml_cuda_fattn_kv_type_supported()` + the diagonal vec case + the three CMakeLists + the new
   instance file.  Build `ggml-hip` and confirm the 4B flips to the FA path (pp512 ~7700, tg32 ~95).
3. **`llama_kv_type_has_native_fa()`** (block 14) — add `IQ4_NL`; then check `-sm tensor` + `iq4_nl`
   (this is where the step-1 session found the pre-existing `q4_0` meta-split abort; if `iq4_nl`
   aborts there, root-cause it the same way — `GGML_BACKEND_SPLIT_AXIS_UNKNOWN` on `attn_gated` — and
   fix the gate, do not paper over it).
4. **QSA/CPU oracle** (block 14): the kernel side of §4e, the CPU `read_kv` case, and add `iq4_nl` to
   the `test_flash_attn_qsa` case list in `tests/test-backend-ops.cpp` (the case list is a
   `for (ggml_type type_kv : {...})` loop — one entry; expect **18 -> 20** cases with the two shapes,
   and it must be **all green**).
5. **A dense-FA test case for `iq4_nl`**: the FA matrix has **none** today (`grep` finds 8
   `GGML_TYPE_IQ4_NL` mentions in `tests/test-backend-ops.cpp`, none of them a
   `test_flash_attn_ext(..., GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL, ...)` case).  Add at least one at a
   mainstream head size (128) plus one quantized control, so `-o FLASH_ATTN_EXT` covers the new load
   path against the CPU reference (the test count will grow — record the new total).
6. **`qsa_kv_native`** (block 14) + the lockstep comment; and the beta's twin (§8).
7. **Gates** (§6) — including both new oracles, the tensor-split reference perf, and the non-QSA
   regression sweeps (the dense enablement is model-agnostic, so 4B/27B/MoE/gemma all change once
   `iq4_nl` is an FA type: re-sweep them).
8. **Land** (§8).

Do **not** try to enable only one half: `iq4_nl` must be accepted by the dense FA path *and* (for
qwen4exp) by QSA, otherwise the type's attention path depends on the model and the "uniform with f16"
goal is lost.  If the QSA side proves slow (> ~2 h), land the dense half first as its own amendment
with a `qsa_kv_native` that still excludes `iq4_nl` — that is a *consistent* state (dense FA, dense
masked QSA) and it is still the bigger half of the win — then do QSA in a follow-up.

## 6. Gates (all mandatory; §4-§6 of the remaining-work handover has the commands)

**Correctness**

* `test-backend-ops -o FLASH_ATTN_QSA` — must be **all green** (18 + your new cases).  This is a real
  oracle: the (9) session's kernel bug scored NMSE ~1.0 here.
* `test-backend-ops -o FLASH_ATTN_EXT` — was **5599/5599**; it must stay green and grow by your new
  case(s).
* `test-backend-ops -o GATED_DELTA_NET` — 4/4.
* QSA width purity with the op **forced** at every width
  (`tools/qsa-width-qsaforced.sh`, i.e. `LLAMA_QSA_DENSE_SHORTCUT=0 LLAMA_QSA_DENSE_DECODE_UNTIL=0`):
  `W = 1..8` one hash per (split, type), `q4_0`/`q8_0`/f16 as controls.  Remember this is
  **necessary but not sufficient** — a width-uniform corruption is perfectly pure.
* Text purity: `plain == --spec-type draft-mtp --spec-draft-n-max 3 == 7` byte-identical on qwen4exp
  (`tools/qsa-text-gate.sh`), both splits, f16 + a quantized control + `iq4_nl`; `/tmp/prompt3k.txt` is
  2122 tokens so the *sparse* arm really runs.  Record the new hashes (the f16/q4_0 tensor-split
  values **must not move**: `804de0576868`, `886292b17a93`).
* MTP gate: pos-1 acceptance >= ~0.45 and MTP t/s >= plain at `n_max 3`.
* **Perplexity vs the dense masked oracle** (`tools/qsa-ppl-oracle.sh`): on the tensor split (the
  reference config) `LLAMA_QSA_SPARSE_FA=0` must match the sparse path within the error bars —
  f16 6.5267/6.5306 and q4_1 6.5787/6.5805 are the (9) values; `iq4_nl` must land in that band
  (expect slightly better than q4_0, or at worst a few 1e-3 worse).
* Non-QSA regression: 4B/27B/MoE-35B/gemma-4-E4B probe hashes with `iq4_nl` (and the f16 controls)
  reproduce; `-sm layer` and `-sm tensor` both load.

**Perf — tuned on the TENSOR split** (the maintainer's rule: tensor is the production config; the
layer split is informational only):

* 4B 1-GPU before/after for the FA flip (the §2 table).
* qwen4exp `-sm tensor` prefill table (`tools/qsa-tensor-perf.sh`): sparse vs dense at
  pp8192/16384/32768 per type — `iq4_nl` sparse should sit with `q4_1` (2404/2453/2384 t/s) rather than
  with dense (2493/2412/2080);
* qwen4exp `-sm tensor` decode at d0/8192/32768 (`tg128`) — the arch policy default (dense decode at
  every depth) must still win; this is a *confirmation*, not a re-tuning, unless your measurement says
  otherwise (then say so loudly — the (9) session re-confirmed it on the fixed kernel).
* KV reserves for `iq4_nl` on 1 GPU and 3 GPUs (the §2 table is the 4B one).

## 7. Traps

* **Everything in `HANDOVER-2026-09-11-qsa-quantized-kv.md` §7 still applies** (constant-cache-type
  isolation, prefill-heavy QSA, `i0` is an element index, the 3-blocks/CU occupancy, never enable only
  the sparse arm, default verbosity for text runs, `--log-verbosity 4` for acceptance, gemma +
  `-sm tensor` aborts in the meta splitter).
* **`kvalues_iq4nl` is a table, not an offset** — `d * kv[idx]`, no `- 8`/`- 16`.  Getting this wrong is
  a silent ~8 % bias on every attention score; the CPU oracle catches it immediately (that is the point
  of running the test).
* **The nibble indexing is the `q4_0` one** (`e % 16` nibble, `e >= 16` is the high nibble) — not
  `2j`/`2j+1`.
* **Do not run the CUDA instance generator** (§4c): it rewrites the whole `template-instances/`
  directory from `TYPES_KV`, which does not contain `iq4_nl`.
* **A type accepted by the predicate but missing from the vec dispatch/instance list is an
  instantiation `GGML_ABORT`** on whichever backend reaches the vec family (NVIDIA does at small
  `n_q`; AMD does not).  Keep the three lists (predicate / dispatch / CMake default) in step — the
  `fattn.cu` comment says so explicitly.
* **`-fa off` cannot be used as a control with a quantized V cache** (`SPLIT_MODE_TENSOR requires
  flash_attn`), and `-sm tensor` + gemma-4 aborts in the meta splitter (2 KV heads < 3 devices).
* **`test-backend-ops`' `-o` filter matches the *output op*** — `-o FLASH_ATTN_QSA` works out of the
  box for the new cases; a new *struct* is needed only if the op changes.
* The QSA op's `identity` mode comes from the env (`GGML_CUDA_QSA_IDENTITY`) and is **not** reachable
  from `test-backend-ops`; the test covers the `idx` path, which is the one the model uses.
* Instrumentation: revert every env-gated trace/print before landing (`build-base` must be
  instrument-free); `CB=0` on the probe; `--single-turn` on `llama-cli`; never run benches in parallel.

## 8. Landing

Standard flow (remaining-work §12), with the two-block wrinkle:

1. **Block 08** (`fattn.cu` + the three CMakeLists + the new instance file) and **block 14**
   (`fattn-qsa.cu`, `ggml-cpu/ops.cpp`, `qwen4exp.cpp`, `tests/test-backend-ops.cpp`).  Amend the
   **earlier** block first (08 before 14) or do both in one `rebase -i` pass marking both `edit`
   (`GIT_SEQUENCE_EDITOR` with `s/^pick <sha> /edit <sha> /`, no `$` anchor).
2. `git apply` leaves changes **unstaged** — the `git add -A` before `git commit --amend --no-edit` is
   load-bearing (a bare amend commits nothing and still rewrites the SHA).
3. `./scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`, bump the default tip in the
   script, refresh `rdna-boosts-all.patch` **by hand** (`git -C /tmp/canon-llama diff 9113cc188
   <new tip>`).
4. Clean-apply sim: `rm -rf /tmp/sim && git clone --no-local -q /tmp/canon-llama /tmp/sim && cd /tmp/sim
   && git checkout -q 9113cc188 && git branch -D rdna-boosts && bash
   ~/llama-cpp-rdna-boosts/scripts/apply-all.sh /tmp/sim` -> strict 15/15, **0 whitespace warnings**,
   applied tree == the canonical tree; build it and check the 4B coherence hash (`1c5d32ac537d` on both
   the canonical and the sim build).
5. **8th block-15 beta re-cut**: `git am -3` the beta patch on the new tip, resolve, squash, re-export
   with `git format-patch --stdout --start-number 15 -1 <sha>`, restore the `[PATCH 15/15]` subject if
   format-patch emits a bare `[PATCH]`, write the file to
   `beta/block-15-campaign-wins/block-15-campaign-wins.patch`, and **round-trip it** (apply the
   *exported* file on a fresh base and compare trees).  Expected conflict: `src/models/qwen4exp.cpp`
   (the beta's `qwen4exp_qsa_sparse()` needs the extended conjunct — the (9) session hit exactly this),
   and **the build will catch semantic gaps that the merge cannot** (block 15 adds `cell_vis`/`q_vis` to
   `ggml_flash_attn_qsa`, so the new QSA test case must pass `nullptr, nullptr` in the beta — this is
   *not* optional: the beta must compile).  Then verify the beta is a **no-op at the gate configs**
   against the delivery build (qwen4exp f16 plain `804de0576868`, q4_1 plain `886292b17a93`, 27B f16
   `n_max 3` acceptance `0.82716`).
6. Docs sweep (newest-first, never rewrite a dated record): a new `WORKLOG.md` entry; the block-08 and
   block-14 table rows + a block-08 section (the `iq4_nl` enablement) and a block-14 section (the
   QSA/CPU/test changes); `GREEDY-PURITY.md` §20's "a new KV type is a new kernel family" list gains
   `iq4_nl`; `AGENTS.md` (chain tip/tree + both amendment bullets); `TODO.md` (F3 step 2 DONE);
   `README.md`/`MANIFESTS.md`/`BASELINE.md` tips; the beta records; this file gets a §9 OUTCOME; the
   `wip/kv-quant-purity-followups/README.md` table row; and the `tools/` scripts if you add any.
7. Push **only** to the delivery repo's own `origin` (`main`), and only after §6 is green.

## 9. Definition of done

`iq4_nl` takes the same attention path as f16 on every model (dense FA, and QSA on qwen4exp); the 4B
prefill/decode flips from ~2270/48 to ~7700/95 t/s; the KV reserve stays the smallest of the set; the
QSA CPU oracle and `FLASH_ATTN_QSA`/`FLASH_ATTN_EXT` backend-op suites are green (with `iq4_nl` cases
added to both); `W=1..8` purity, text purity, MTP and the perplexity-vs-dense gate all hold on both
splits; the tensor-split perf table is recorded; the non-QSA sweeps reproduce; clean-apply is strict
15/15 with 0 whitespace warnings and the applied tree equal to the canonical one; block 15 is re-cut
(8th) and verified as a no-op at the gate configs; and the docs are swept.  Then **F3 is complete** —
the follow-ups that remain are the ones listed in `TODO.md` (the tensor-tuned `iq4_nl`/QSA prefill
crossover gate, the QSA fused-op probe, the gfx1151 bundle, and the block-15 promotion when its beta
window closes).
