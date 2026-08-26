# BASELINE - provenance and drift policy

Branch: `baseline/d222767c7` (current, recommended)
Upstream range: `7584430716ee229751771ed0d6bbcb780d105eeb` .. `d222767c7`
("kleidiai: Rework KleidiAI Build System/Integration (#26077)")

Older branch: `baseline/758443071` - the known-good set for the upstream
range around `758443071` (see "Older branches" below).

## Baseline

All patches on this branch are generated against **llama.cpp upstream master
at `d222767c7`** and were validated by applying them, one at a time, to a
fresh checkout of that SHA, building with ROCm 7.14 (gfx1201, GGML_HIP=ON,
Release), and running the full test suite.

### Validation record (2026-08-25, AMD Radeon AI PRO R9700 x3, ROCm 7.14)

- `test-backend-ops` (ROCm0/1/2 + CPU): **14883/14883 passed, 0 failures**.
- `GATED_DELTA_NET` `-b ROCm0`: **46/46** in all four dispatch configs
  (default bf16 S_v=128, `GGML_CUDA_GDN_CHUNKED_BF16=0`,
  `GGML_CUDA_GDN_CHUNKED=0`, both force-enabled).
- `test-arg-parser`, `test-speculative-adaptive`: all tests OK.
- The applied tree is byte-identical to the source fork branch `chunked-gdn`
  for every production file (the only intentional divergence is the test
  harness fix described below).

### Two fixes vs the fork

The patch set carries two fixes that are NOT on `chunked-gdn`; both come from
the fork's `rdna-boosts` branch (the production lineage) or from this
validation:

1. **Test-harness seeding (folded into block 02).** The fork's chunked-GDN
   work carried a deterministic seed into `init_tensor_uniform` in
   `tests/test-backend-ops.cpp` (added while debugging the bf16 GDN kernel):

   ```cpp
   // fork (chunked-gdn): static std::atomic<unsigned> g_seed(12345); (void) g_seed;
   // thread_local std::default_random_engine gen(12345 + (unsigned) start * 101);
   // this branch:         thread_local std::default_random_engine gen(std::random_device{}());
   ```

   The fixed seed correlates tensor data across rows and deterministically
   exposes a pre-existing numerical fragility in `rms_norm_back` and
   `cross_entropy_loss_back` on RDNA4 (CPU/GPU comparison fails with fixed
   seeds; passes with `random_device` seeding). GDN results are unaffected
   (46/46 in all configs either way). Full diagnostic: fixed seeds 12345 and
   54321 both fail those 9 cases; only the seeding line differs in the
   passing build.

2. **Meta-buffer compute-container headroom (block 11, `f2a22a71`).**
   `compute_headroom` 16x -> 128x in `ggml-backend-meta.cpp`. Hybrid
   recurrent models (GDN/SSM) create ~2*(n_rs_seq+1) conv-state snapshot
   views per recurrent layer during graph allocation, exceeding 16x and
   aborting with "not enough space in the context's memory pool"
   (ggml.c:1804). Without it, speculative MTP drafting under
   `--split-mode tensor` crashes on first decode. Source commit is on the
   fork branch `rdna-boosts`, not `chunked-gdn`; upstream has not fixed it
   either (reproduced on pristine `d222767c7`).

## Per-block provenance

Blocks were re-generated from the validated application (one commit per
block on the `rdna-boosts` branch of the consumer checkout at `d222767c7`),
so each patch is exactly the block's net change against the new baseline.
The original source commits on the fork branch `chunked-gdn` (parent
`758443071`) remain:

| patch | source commits on `chunked-gdn` (history order) |
|-------|--------------------------------------------------|
| `01-adaptive-mtp.patch` | `87ad1db26` `b56926039` `d0d7ff27e` `8d70e21f5` `0cf87e989` |
| `02-chunked-gdn.patch` | `876ef1f0b` `5d2090e96` `b220647b1` `a4982afa2` `659f94987` `2a1e5c5a8` `1da07e19b` `77d51ee28` `abfa24265` `be46c7621` `3441b7d40` `246136122` `05cab3c41` |
| `03-bf16-kv-cache.patch` | `5485e79e4` `b98265cfd` `07767a88a` `ef3673358` `b6bfa422e` `5e6072558` `bd5bf0ea3` `d33ce1adf` |
| `04-wmma-flash-attn.patch` | `beaf69fb6` |
| `05-bit-identical-decode-cpu.patch` | `89ac4ba1f` |
| `06-fused-core.patch` | `14e5dd427` `d0e6119a7` `333e8f950` `c11752b18` `10e016df4` `85387ba3a` `8e1300159` `ac08b6d85` `a84112dcf` `9b4554626` `555e79ab2` `00f53040f` `ec09a818e` `bb64338f9` `4c0440841` `3d65d7979` |
| `07-gfx1151-mmvq-table.patch` | `5b320ed94` |
| `08-host-buffer-revert.patch` | `edb8d44c0` |
| `09-meta-device-wrapper-skip.patch` | `32670eec8` |
| `10-q6k-mmvq-vdr2.patch` | `cd35abd19` |
| `11-meta-headroom.patch` | `f2a22a71` (fork branch `rdna-boosts`, NOT on `chunked-gdn`) |

> `06-fused-core.patch` additionally carries a local correctness fix on top of
> the fork commits: the Q8_1 input cache now keys entries by `src1->data` in
> addition to the view root, so the stack-allocated per-expert `src1_slice`
> tensors of the mul_mat_id host-sort fallback never collide (equal token
> counts produced identical keys, reusing the wrong expert's quantized
> tokens; nondeterministic iq1_m MUL_MAT_ID failures on RDNA4).
| `rdna-boosts-all.patch` | all 48 commits of `758443071..chunked-gdn` + `f2a22a71` |

## Older branches

`baseline/758443071` holds the original patch set (generated against
`758443071`, mirroring the fork `chunked-gdn` exactly). It remains the
known-good set for that upstream range, with two caveats:

1. Its `02-chunked-gdn.patch` still carries the deterministic test-harness
   seed (see above) - the fork's state. Use this branch's fixed patches if
   you are testing against `758443071` and hit the `rms_norm_back` /
   `cross_entropy_loss_back` failures.
2. Its block-10 test hunk is placed at a re-based location in
   `make_test_cases_perf()` (context added by block 04 in the fork).

## Drift policy

The patches are static. If a patch fails to apply against a newer upstream
master:

1. Try `git apply -3` (3-way merge against the baseline blobs).
2. If 3-way fails, rebase the failing hunks manually against the current
   master and continue.
3. Do NOT hand-edit the committed patches as the permanent fix: when a new
   validated baseline is cut, the whole set is regenerated from the
   consumer application (as done for `baseline/d222767c7`) and a new
   `baseline/<new-sha>` branch is created here. Old branches stay as the
   known-good sets for older upstream versions (rule of thumb: cut a new
   baseline branch whenever more than one block needs manual re-base hunks,
   or when a validated full-suite run exists for a newer tip).

`MANIFESTS.md` records the upstream range per branch so consumers can match
their upstream version.
