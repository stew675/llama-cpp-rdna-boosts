# archive/work/gdn-conv-store-multiseq (CLOSED, promoted in r249) — contributor PR #295

**Status: CLOSED 2026-11-10 — promoted into the delivery as a block-10 amendment in
`v16-a691b2345-r249`.**  Nothing here is delivery work and nothing is open.  The code is in
`patches/0010-*` (`ggml/src/ggml-cuda/gdn-conv-store.cu`/`.cuh`); the record of *shipping* it is `WORKLOG.md`
(newest entry), `INVARIANTS.md` r249, the `ENVIRONMENT.md` row for `GGML_CUDA_FUSE_GDN_CONV_STORE` and
`MANIFESTS.md` block 10.  This file is the campaign record: what the contributor offered, what we measured,
what changed on acceptance, and what was deliberately left out.

Contributor: **overdoingism** — <https://github.com/stew675/llama-cpp-rdna-boosts/pull/295>, one `git am` file
cut against **r244** (`wip/gdn-conv-store-multiseq/`), plus a README (`README-as-submitted.md`).  PR head at
acceptance: `ff9a04c5164837d23e21768135f987f0bfa66b78`.  The patch applies cleanly to the r248 tree as it
stands (`gdn-conv-store.cu`/`.cuh` were untouched between r244 and r248), and the submitted sha256 is
reproduced here exactly.

Artifacts kept here:

| file | what | sha256 |
|---|---|---|
| `0001-gdn-conv-store-the-one-kernel-GDN-conv-step-also-tak.patch` | as submitted (`gdn-conv-store.cu` / `.cuh`, +54 −35) | `eb963047cb7f842fd28575e85523328eafe527fe8df83e206b3fd75e42a8976b` |
| `r249-final-cut.diff` | the **delivery's** form (`git diff 12433634d fe0f62b7a -- <the two files>`), identical to the submitted patch | `39b101f4a133ccec858c0f9a72f1094eaa974cc79dcab042176da7931656d856` |
| `README-as-submitted.md` | the contributor's own README (claims, numbers, "not validated" list), unedited | — |

## What it does

r244's one-kernel GDN conv step (`GGML_CUDA_FUSE_GDN_CONV_STORE`) accepted **one** sequence; every other
sequence count fell back to the chain.  With `--kv-unified` and two or more slots verifying in the same
ubatch, the ubatch really does carry `S > 1`, so those graphs — most verify steps of a multi-slot
speculative server — ran `GET_ROWS`, `CONCAT`, the window view, `SET_ROWS`, `SSM_CONV`, `SILU` and the q/k
norm pair on every GDN layer.

The step now takes **1..8 sequences, one block row a sequence** (`grid = dim3(C/256, S)`).  `T`, `C` and `S`
come from the transposed view (its root is the in-projection's output, 2-D `[C, T*S]`, whose 3-D reshape view
is what `tr->view_src` names), the per-sequence strides are checked (`tr->nb[0] == C*4`,
`tr->nb[1] == 4`, `tr->nb[2] == T*C*4`), and the root must be contiguous with exactly `C*T*S` elements.  With
`S > 1` the state stays the `GET_ROWS` **copy** — the gather is deferred for one sequence only, because one
sequence's cache rows may be another's source (`ggml_cuda_gdn_store_gather_deferrable()` now refuses
`m.S != 1`).  `GGML_CUDA_CONV_DEBUG=1` prints the `GDN_CONV_STORE` line once **per sequence count** and it
carries `S=`.  Kill switches unchanged: `GGML_CUDA_FUSE_GDN_CONV_STORE=0`, or `GGML_CUDA_DISABLE_CONV_FUSION=1`
for the whole conv family.

## What we measured before accepting (gfx1201, ROCm 10.1.0, 1 x R9700 unless stated)

Every arm below is **byte-identical to r248** — the fingerprint is both generated texts *and* every per-token
logprob (so it is not a text-only comparison), driven as one `/v1/completions` request carrying N copies of
`prompts/prose-rdna-boosts.txt` (`sha256 fabdec65…`) so the sequences really do verify in lockstep.

| arm | shape | fingerprint |
|---|---|---|
| r248 (baseline) | `-np 2 --kv-unified`, DFlash2 n5, q8_0 KV, `-c 16384` | `ba595285c0e555e2` |
| **r249** (store on) | same | `ba595285c0e555e2` |
| r249 `GGML_CUDA_FUSE_GDN_CONV_STORE=0` (chain) | same | `ba595285c0e555e2` |
| r249 `GGML_CUDA_DISABLE_CONV_FUSION=1` (family chain) | same | `ba595285c0e555e2` |
| r249 repeat | same | `ba595285c0e555e2` |
| r249 / r248 | `-np 4`, 4 sequences, code prompt | `7ff4b91512e7ae57` (64 tok) / `c9a3b29675e4715d` (128 tok) |
| r249 / r248 | 35B-A3B (C=8192) `-np 2` | `89f2ea4785d5c0b2` |
| r249 / r248 | `GGML_CUDA_DISABLE_GRAPHS=1` | `74d9a48d81e22934` |
| r249 / r248 | `-sm tensor` 2 devices (C=5120) | `0edf500bb27cbdde` |
| r249 / r248 | `--spec-type draft-mtp` n3, `-np 2` | `d96d2182b1224e4b` |

The census proves the path fired and that the baseline declined it: r248 prints only
`GDN_CONV_STORE C=10240 T=2 …`, r249 prints that plus `… C=10240 T=4 S=2 Kw=4 col0=1 K=6`, and with
`-np 4` also `T=4 S=4` and `T=6 S=2`.

- **Step time** (two sequences decoding together, dense 27B UD-Q4_K_XL, DFlash2 n5, `predicted_ms /
  (predicted_n − accepted)`): **7011.9 / 6995.7 ms** vs r248's **7082.9 ms**, i.e. ~1.0-1.2 % faster; the
  submitter measured 1.6-1.8 % on their box.  Draft acceptance is identical to the digit (0.41646 / 0.64667).
- **VRAM** identical in every pair (2 slots `-np 2`: 19614 MiB total both builds; `-np 4`: 21137 MiB both).
- **Protocol A** (reduced: `-c 32768`, bf16 KV, `-n 1000`, seed 42, prose): dense text `c4cec4370cadee92`
  identical, acceptance **0.89545** (728/813) identical, plain 29.66 → 29.71 t/s and MTP 77.04 → 77.30 t/s;
  35B-A3B text `dc3a06849475dfda` identical, acceptance **0.92920** (735/791) identical, 101.26/200.19 →
  101.09/200.30.  MTP ≥ plain on both, as the protocol requires.
- **Prefill logits**: plain point **0.000000 / 100.000 %** against the r248 base (PPL 5.9325), and under
  `GGML_FORCE_N_RS_SEQ=3` (the r244 recipe, which is the only way to reach the store from `llama-perplexity`)
  **-0.000000 / 99.990 %** against a base re-recorded from r248, with the `GGML_CUDA_FUSE_GDN_CONV_STORE=0`,
  `GGML_CUDA_DISABLE_CONV_FUSION=1` and `--ub 8` arms all identical.
- **Width probe** `P=2048/3584/4096`, `RS=0` and `RS=3`, plus the `GGML_CUDA_DISABLE_CONV_FUSION=1` arm:
  `maxdiff 0`, `width_purity PASS`, and the r244 anchors exactly (`9d22f1cfc9d5511d`, `946b1435ffc6a107`,
  `b4b493442fa8e46d/f6d04c9117b6af98`).
- `test-backend-ops -o GATED_DELTA_NET` 55/55; `llama-batched-bench -npl 1,4,8` neutral (±0.5 % at B=4/8);
  no GPU faults (`scripts/check-gpu-faults.sh`, kernel journal + every log).

## What changed on acceptance

**Nothing in the code.**  This is the rare contributor patch that needed no adjustment: the arithmetic, the
stride checks, the deferral rule and the kill switches were all as claimed, and the numbers reproduced
(within 0.6 points on the step-time delta, on a different box).  The only delivery-side work was the fold into
block 10 and the documentation.

## Not taken / still open

- **A multi-sequence prefill ubatch** (more than 16 tokens a sequence) still declines in the direct conv
  (`st->ne[2] == 1`), and `ggml_cuda_ssm_conv_in_try_fuse` still does not take the store graph.  Unchanged by
  this patch, and not attempted.
- **3+ slots beyond 4**: 8 are accepted by the matcher; 4 was measured here, 2 by the submitter.
- **`GGML_CUDA_DISABLE_GRAPHS=1`** was untested by the submitter; it is now an arm above (identical).
- **Multi-GPU**: only a 2-device `-sm tensor` arm was run here (identical); `-sm layer` across 3 devices,
  `-ncmoe`/host-resident experts, Windows, RDNA3 and the other GDN architectures (qwen3next, qwen4exp) were
  not exercised.
- **`--ub 4`/`--ub 1` under `GGML_FORCE_N_RS_SEQ=3` aborts** in `llama_batch_allocr::split_equal` — on the r248
  build **too**, so it is pre-existing and not this patch's; it is filed as a `TODO` item, and the r244 recipe's
  R201 `--ub` arms are therefore run at `--ub 8`.
