# PR #322 — the GDN gate/beta epilogue for F32 and K-quant projections (folded, r304)

**Status: PROMOTED.** The series shipped in **r304** as an amendment to **block 10** (`WORKLOG.md` 261117.003,
`INVARIANTS.md` r304, `patches/README.md` r304). Nothing here is applied by `scripts/apply-all.sh`; this directory
is the submitted material as it arrived, plus the record of what changed on acceptance.

Submitted by `briansp2020` as PR #322 (`wip/gdn-gate-beta-r299`), cut against the r298 tree (`99c4717c6`,
tree `5f55d95a5`), applied here to `4f9dfa0e3`+r303 without fuzz.

Submitted files (sha256):

| file | sha256 |
|---|---|
| `0001-mmvf-GDN-gate-beta-fusion-also-for-F32-projections-G.patch` | `6fa161d58a9dc756004b4ab1cda43932da144e54b5203d011ec95948b4322a13` |
| `0002-mmvq-GDN-gate-beta-epilogue-for-non-Q8_0-ssm_alpha-s.patch` | `972b2052380cf4886b3b4bb8e1ef5b4eb8f7fe21d645f470f9fc871cf73009da` |
| `0003-mmvq-GDN-epilogue-precise-memory-check-the-window-ch.patch` | `00dec408e93b5be087cf688706a401fbed674969f852176c8d06a21e380892c4` |

## What was verified as submitted

Output bit-identical to the previous build on both new arms (prefill-logit KLD **-0.000001 / 100.000 %** on the
F32 arm, **0.000000 / 100.000 %** on the Q4_K arm, at `-ub 1`/`4`/`8`, each against a base recorded from the
previous build's binary), decode **+1.9 %** (F32 arm, `tg128` 114.84/114.92 -> 117.10/117.12) and **+1.0 %**
(Q4_K arm, 148.62/148.61 -> 150.21/150.08), no GPU faults, `test-fusion-gate-beta-multiseq` and
`gate-fusion-alias-copy.sh` unchanged. The submission's own figures (+0.9-1.1 % decode) are conservative.

## Three amendments made on acceptance

1. **The memory rule's already-`Q8_1` activation case** (`mmvq_gdn_epi_find`). 0003 replaces
   `ggml_cuda_check_fusion_memory_ranges` for this fusion's window because the allocator places the gate `MUL`
   output *inside* `attn_norm-N` — the F32 activation the projections read — on every GDN layer (instrumented and
   confirmed). That is harmless only because `ggml_cuda_mul_mat_vec_q` quantizes the activation to a Q8_1 copy in
   a **separate launch** and the matvec reads the copy; when the activation is already `Q8_1` there is no copy,
   the kernel reads it in place, and a written tensor overlapping it would corrupt rows still to be read. The rule
   now requires that disjointness too.
2. **The F32 arm could not be gated.** `GGML_CUDA_GDN_EPI_LOG` only existed in the mmvq half, so nothing could
   prove the mmvf F32 fusion fires; it now prints for both families and `scripts/gate-gdn-epi.sh` checks it
   (with `--self-test` against a pre-r304 binary as the negative control).
3. **No test and no gate shipped with it.** `tests/test-fusion-gate-beta-multiseq.cpp` gained F32 and Q4_K arms
   (77 shapes, bit-identical), and the F32 arm is **negative-tested**: reverting the F32 branch's
   `gb_x2 ? 2*ne01 : ne01` row count fails all 11 F32 shapes at maxdiff 0.7-1.4. `scripts/gate-gdn-epi.sh` is
   new. The submitted README also calls `GGML_CUDA_FUSE_GATE_BETA_BF16` "new" (it has shipped since r102) and
   names r298/r299 as the cut base (two releases stale) — documentation only, no code change.

## What was NOT changed

The matchers, the ksplit epilogue arithmetic (`__fadd_rn`/`__fmul_rn` + `op_softplus`/`op_sigmoid`), the concat
reorder's `GGML_CUDA_FUSE_GDN_EPI` branch and the per-type switches are as submitted. `logs/` is the submitter's
own evidence and was not re-run here.
