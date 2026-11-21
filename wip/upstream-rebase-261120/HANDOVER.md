# HANDOVER — r328 re-base ladder, 6 of 18 rungs climbed, resume at **R06**

Written 2026-11-20 at the end of the first session.  Read [`README.md`](README.md) §1 (method + traps),
[`RUNGS.md`](RUNGS.md) (the per-rung ledger and every resolution so far) and [`DECISIONS.md`](DECISIONS.md)
(the decision log) — this file is the delta: what is done, what is owed, and how to not repeat the mistakes of
session 1.

## §0 State, exactly

| item | value |
|---|---|
| working branch | **`rebase-r328`** in `~/llama.cpp`, clean, **HEAD = `c3860af1e`** (tree `b52803164`) |
| rungs done | **R00, R01, R02, R03, R04, R05** + **two rung repairs** (R00's top-k gather, R05's `nb32` widening) |
| next rung | **R06** — block 06 (`f0564df7d`, MMB quantized prefill GEMM), 11 files, 2 conflicts (`ggml-cuda.cu`, `mmid.cu`) |
| delivery repo | `~/llama-cpp-rdna-boosts` at `beef56f6` — **`patches/`, `release.json`, `rdna-boosts-all.patch` and the fork are untouched** (that is R18) |
| target release | next after `v17-4f9dfa0e3-r327`, i.e. **r328**; new fork point **`919794500`** |
| reference binaries | `~/llama.cpp/build-rocm/bin-r327` (the shipping r327 `bin/`) |
| oracle builds | `~/llama.cpp/build-base` (pristine base, hangs — see §4) and `~/llama.cpp/build-base-novmm` (pristine base, `GGML_HIP_NO_VMM=ON`, **the one that runs**) |
| ladder build | `~/llama.cpp/build-rocm` — `GGML_HIP_NO_VMM=OFF`, `GGML_CUDA_FA_QUANTS` includes `iq4_nl-iq4_nl`, `LLAMA_USE_PREBUILT_UI=OFF` |

### The commit/tree table (oldest first) — compare **trees**, never SHAs, against a re-cut

```
919794500                    base
4cba0ad32  a3824ea0e  block 00: foundation
14a0414bb  6566ca3ac  block 01: hybrid HIP all-reduce
3ea8b0212  e12a34bad  R00 repair: top-k gather (block 10's kernel moved down)
3941bad4e  0799ebfad  block 02: adaptive MTP draft depth
e612388ba  971953b99  block 03: chunked GDN prefill
768d27561  0866d4229  block 04: BF16 KV + native BF16 FA
1c9bfa37c  13683655d  block 05: FA band + native quantized K/V
c3860af1e  b52803164  R05 repair: nb32 -> int64_t
```

## §1 What each rung cost, so R06..R17 can be planned

| rung | conflicts | what it actually took |
|---|---|---|
| R00 | 4 files / 7 regions | **D1** (take upstream `#32505`, drop the dup-ids port, retire the candidate) + re-port `mm_ids_helper_512_10` onto upstream's `ne_store` helper + keep the `#30542` revert.  Then **the R00 repair** (a VMM-pool top-k refusal, fixed by moving block 10's gather kernel down — **D15**) |
| R01 | 0 | one repair: HIP now static_asserts a 64-bit warp mask (`ggml_cuda_ce_xor` -> `ggml_cuda_shfl_xor`) |
| R02 | 5 files / 9 regions | all unions; **D5 resolved as MERGE** (upstream added sibling speculator classes, kept `impl_draft_mtp`), enum count 13 -> 14, docs **regenerated** with `llama-gen-docs` |
| R03 | 1 file / 2 regions | unions: upstream's `padded_q`/`permuted_k` layouts now also pass `/*n_rs_batch=*/1`; gate range `correlated ? -1e-4 : -0.5` |
| R04 | 2 files | **D3**: union of 7 rope arms + **dropped upstream's guard** that refused every non-BF16 rope into a BF16 cache (its own matrix lacked the arm ours supplies) |
| R05 | 7 files / **22 regions** | **D16** (maintainer): the delivery's FA band iteration model wins; upstream `#32407`'s rotation **not carried** — and `#32217`'s per-head mask + `#31617`'s typed masks still adopted.  Then the R05 repair: `nb32` -> `int64_t` |

## §2 Owed at specific later rungs (these are the ones that will be forgotten)

1. **R10 — drop block 10's `top_k_gather_rows` hunk.**  The R00 repair already put it in the tree; when block 10
   is cherry-picked, resolve that file **keep-local** and keep block 10's `ggml_cuda_cpy_as_memcpy_2d` guard
   (different site).  Duplicating the hunk is the failure mode.
2. **R17 — `TODO` #351**: does upstream now support `-sm tensor` for `flint`?  That is block `0300`'s drop
   condition.  3-GPU arm on `/llm/models/RecursiveAI/Flint/Q8_0/flint-Q8_0.gguf`; also the refusal arm (must
   still print the pointer and exit 1) and `-sm layer` vs `-sm relaxed-tensor` text identity
   (`103 chars sha=cc4fbc3e8367` at r278).  Dropping a block is a maintainer decision, never a gate outcome.
3. **R18 — squash `3ea8b0212` into block 00** (rung-repair pattern) and **`c3860af1e` into block 05**.
4. **R18 — re-measure `mm_ids_helper_512_10`** before quoting any number: the claim of record is
   **+4.5 % pp16384 / +5.2 % pp2048**; at this base it reads **+0.7 %** on a 5298-token prompt at 2 devices
   (`ON 1642.7/1638.7` vs `OFF 1635.4/1624.2`), text identical.  If it does not reproduce, the block-00 note and
   `patches/README.md`/`ENVIRONMENT.md` must say so.
5. **R18 — regenerate the generated docs and assert no drift.**  `llama-gen-docs` at R02 proved the committed
   tables at r327 are **stale**: block 00's `arg.cpp` adds the `--lazy-mode mlockr` help text and neither
   `tools/cli/README.md` nor `docs/speculative.md` lists `mlockr` (`arg.cpp` 2 hits, README 0).  Regenerate,
   then `git diff --exit-code` on the doc files.
6. **R18 — `TODO` #352**: move the container `web` stage from `tools/ui/` to `ui/apps/chat` (+ `ui/lib/ui`,
   `ui/lib/voice`) and the Priority-1 copy to `ui/apps/chat/dist`.  Only correct once the tree has the new layout.
7. **R18 — the FA gate for every latent risk this climb created**: `gate-prefill-logits.sh` (record a new base on
   the new build), `gate-spec-purity.sh`, `gate-spec-gdn-split.sh`, `gate-norm-rows-alias.sh` and the rest of the
   per-change table in `AGENTS.md`.  **D16** in particular means the FA band model is now the only iteration
   model in the tree, so the FA op matrix and the prefill-logit gate are its only guards.
8. **`TODO` #353** — Inkling-2 needs upstream [PR #32695](https://github.com/ggml-org/llama.cpp/pull/32695)
   (**unmerged** at 2026-11-20).  Nothing to do until it merges; then it is the local validation for the per-head
   FA mask and for the `nb32` widening.

## §3 The traps that actually bit in session 1 (all are avoidable)

- **Build-completion detection.**  Do **not** poll `pgrep`/`grep` for a build.  Session 1 lost two turns to a
  false "still building" (and one to a false "done").  Use a sentinel:
  `(cmake --build … > log 2>&1; echo "BUILD_RC=$?" >> log)` and poll the log for `BUILD_RC=`.
- **`GGML_CUDA_FA_QUANTS` is hand-set in `build-rocm/CMakeCache.txt`** and must contain `iq4_nl-iq4_nl` **only
  from R05 on** (block 05 adds `iq4_nl` to `FA_TYPES`).  Before R05 the configure dies with
  `unknown type "iq4_nl"`.  Set it with `cmake -B build-rocm -DGGML_CUDA_FA_QUANTS="…"`.
- **HIP's 64-bit warp mask** (`#31617`) — bit twice: R01's all-reduce and R05's band code.  Any raw
  `__shfl*_sync(0xFFFFFFFF, …)` must become `ggml_cuda_shfl*`.
- **`-fa on` is required** for any KV-cache-type arm (bf16/q8_0); with FA off the cache type is not used by the
  attention kernel and both arms collapse to the same path (a vacuous rung).
- **27B on one 32 GiB card needs `--fit false -c 8192 -fa auto`**; the default `--fit` tries a 16 GiB KV
  allocation and dies in `llama_init_from_model`.
- **Docs are generated**: never hand-merge `tools/*/README.md` or `docs/speculative.md` — union what you can,
  then run `./build-rocm/bin/llama-gen-docs` and check the diff.
- **`-sm tensor` needs `HIP_VISIBLE_DEVICES` pinned** (the i5503D iGPU is enumerated otherwise).
  `--device 0,1,2` is **not** a valid flag at this base (`invalid device: 0`); the pin is enough.
- **Never run two GPU workloads at once** (benchmarks, op tests, or a build + a benchmark).  An op test is not a
  benchmark, but it still owns the GPU.

## §4 Facts about this box's GPU stack that the ladder depends on

- **The pristine base hangs** on its first GPU compute with `GGML_HIP_NO_VMM=OFF` (the delivery's required
  config): `#30542`'s VMM pool retry loop spins inside libamdhip64.  `GGML_HIP_NO_VMM` is a **CMake option**, not
  an env var.  A base built with `GGML_HIP_NO_VMM=ON` runs fine — that is `build-base-novmm`, the oracle.
- The **`#30542` revert** (block 00) is what makes the pool work; it is carried a second time (**D2**) and its
  rationale is separate from block `0300` (see `DECISIONS.md` D2).
- **The VMM pool refuses a strided 2-D D2D copy into a VMM-backed destination** (`hipMemcpy2DAsync`,
  "invalid argument").  It is reachable with *no delivery code at all* (pristine base + the revert only), and the
  delivery's answer is the device-side gather (`top_k_gather_rows`, now in block 00).  The same class as issue
  #313's windowed-KV state copy.
- Sanity oracle for every rung: **`100 chars sha=1c5d32ac537d`** on
  `Qwen3.5-4B-Q8_0 -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt
  --single-turn`, `HIP_VISIBLE_DEVICES=0` (~242/91 t/s).  It has been identical at every rung so far.

## §5 Useful arms, with their measured baselines (do not re-derive)

| arm | command sketch | R05 reading |
|---|---|---|
| coherence (every rung) | 4B, as above | `1c5d32ac537d`, 242/91 t/s |
| 3-device tensor (block 01's arm) | `HIP_VISIBLE_DEVICES=0,1,2 … -sm tensor` | `1c5d32ac537d`; census `HIP AllReduce pipeline: 3 GPUs` |
| MTP adaptive (block 02) | 27B UD-Q4_K_XL, `--fit false -c 8192 -fa auto`, `--spec-type none` vs `draft-mtp-adaptive` | 29.3 -> **65.2 t/s**, text identical `55824e640aa0`, acceptance 0.95238 |
| chunked GDN (block 03) | 4B, 5298-token prompt, `GGML_CUDA_GDN_CHUNKED=1/0` | 6870.1 vs 6314.6 t/s (**+8.8 %**), text identical |
| bf16 KV + rope fusion (block 04) | 4B `-ctk bf16 -ctv bf16 -fa on`, `GGML_CUDA_DISABLE_FUSION=0/1` | 6786.5/88.8 vs 6417.0/76.2 (**+5.8 % pp, +16.5 % tg**) |
| native quantized K/V (block 05) | 4B `-ctk q8_0 -ctv q8_0 -fa on`, `GGML_CUDA_FA_KV_NATIVE=0/1` | **88.6 vs 83.9 t/s** decode (**+5.6 %**) |
| FA correctness (blocks 04/05 and anything touching `fattn-*`) | `test-backend-ops -o FLASH_ATTN_EXT` | **7428/7428, 0 FAIL**, ~4 min |
| MUL_MAT_ID (block 00) | `test-backend-ops -o MUL_MAT_ID` | 1078/1078, 16 `dup=1` |
| GATED_DELTA_NET (block 03) | `test-backend-ops -o GATED_DELTA_NET` | 76/76 |
| Flash-Next (host-expert field model, the known-red's repro) | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, `HIP_VISIBLE_DEVICES=0,2 -sm tensor -c 8192 -ub 1024 -b 1024`, the 5298-token prompt | 1692.5 t/s prefill after the R00 repair |

The 5298-token prompt is `prompts/prose-rdna-boosts.txt` (sha256 in `prompts/README.md`); hash results with
`scripts/extract-generated.py`, never a grep/sed slice.

## §6 The standing bias the maintainer gave (and its limit)

> *"In every single rebase we've ever done, there was never a time where upstream did something better than the
> delivery repo"* — so where upstream and the delivery are **alternatives to the same mechanism**, take the
> delivery's and record the cost (**D16** is the first case: upstream's FA rotation dropped, verified absent).

**The limit, learned the hard way at R05** (`DECISIONS.md` D16 addendum): that bias is about a shared
*mechanism*, not about a **correctness** change upstream made for its *own* new code.  `nb32` is the example:
keeping the delivery's `int32_t` looked right and was wrong, because the per-head term upstream added multiplies
it by a head index.  Ask of any contested hunk: *is this a preference between two ways to do our thing, or is it
the arithmetic/contract of upstream's new feature?*  Take the delivery's side for the first, upstream's for the
second.

## §7 How to resume (short form)

```bash
cd ~/llama.cpp && git status && git branch --show-current      # must be rebase-r328, clean
git log --oneline -1                                            # expect c3860af1e
git cherry-pick f0564df7d                                       # R06 = block 06
# resolve: ggml/src/ggml-cuda/ggml-cuda.cu, ggml/src/ggml-cuda/mmid.cu (2 regions total)
(cmake --build build-rocm -j 24 > /tmp/r06.log 2>&1; echo "BUILD_RC=$?" >> /tmp/r06.log)
# poll for BUILD_RC, then: coherence -> the block's own switch -> the rung's log in RUNGS.md -> commit
```

R06 starts **`D11`**: the AMD MMQ/MMVQ tuning A/B (upstream retuned `mmq-config-gcn.cuh` and the MMVQ `ne11`
gate in this range).  The maintainer's instruction is **measure both, pick the better of each knob** — the plan
is the table in `DECISIONS.md` D11, and the rule is "per knob, not per file"; a mixed result is expected.
