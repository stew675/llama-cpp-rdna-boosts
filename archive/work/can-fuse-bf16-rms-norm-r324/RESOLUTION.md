# Resolution — CLOSED at r335 (already fixed at r328 by the re-base; no patch carried)

**PR #339 (`@jstamagal`) reported a real defect: it is accurate for r324-r327, and it was already fixed
before the PR was opened.**  The fix did not come from a patch — the r328 re-base onto `919794500`
refreshed block 10's moved copy of `ggml_cuda_can_fuse`, which had gone stale against its own fork point.
Nothing in the delivery changes, and the contributor's `0001-*.patch` is **not** applied (it no longer
applies to the current tree).  Record: `WORKLOG.md` 261121.014, `INVARIANTS.md` r335.

## 1. What was reported, and why it was right

`ggml_cuda_can_fuse` (block 10, `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`) asserted an F32 norm before it
looked at an `RMS_NORM + MUL (+ ADD)` chain:

```c
GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
```

and required **both** `mul` operands to be F32.  A BF16 `RMS_NORM` followed by `MUL` therefore aborted
instead of fusing, on every HIP/CUDA device, even though `norm.cu` has carried the BF16 fused kernels
(`norm.cu:1309`, `:1393`) and the fork point accepted an F32 **or** BF16 norm.  The `RMS_NORM + SCALE` arm
below it carried the same two asserts where the fork point returns `false`.

## 2. Where it came from: an upstream commit landed under a moved copy

Upstream `31953bcf7` — *"ggml : add BF16 support to RMS_NORM (CPU, CUDA)" (#32135)* — relaxed exactly
these checks **in `ggml-cuda.cu`**.  Block 10 splits the fusion dispatcher out of that hub into its own TU
(`ggml-cuda-try-fuse.cu`), so the delivery carries *a copy*, and that copy was taken from a pre-`31953bcf7`
base.  The block's own removal hunk in `ggml-cuda.cu` kept applying across the r320-r327 re-cuts, so the
re-bases did not notice that the new TU's body had fallen behind the hub's.

The delivered artefact proves it, tag by tag (`patches/0010-…patch`):

| release | `GGML_ASSERT(rms_norm` in block 10 | the same release's fork point |
|---|---|---|
| `v17-4f9dfa0e3-r320` … `-r327` | **4 occurrences** (both branches) | relaxed (F32 or BF16) |
| `v17-919794500-r328` … `-r334` | **0** | relaxed (F32 or BF16) |

`v17-4f9dfa0e3-r324`'s fork point `4f9dfa0e3` already had `31953bcf7`, so at r324 the delivery disagreed
with its own base — the abort the PR reports is real at that revision.

## 3. Why it is closed: r328 already carries the fork point's text verbatim

On `v17-919794500-r334` (current `main`, tree `0ba731781`, base `919794500`):

* `ggml-cuda-try-fuse.cu` has **no** `GGML_ASSERT(rms_norm…` at all; the only F32/BF16 asserts left are in
  `norm.cu:1309`/`:1393`, inside the fused kernels themselves.
* The `RMS_NORM + MUL (+ ADD)` branch **and** the `RMS_NORM + SCALE` arm were extracted from
  `919794500:ggml/src/ggml-cuda/ggml-cuda.cu` and from `HEAD:ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu`
  and compared: **2944 bytes each, identical** — i.e. both hunks the PR proposes are already present.
* `git apply --check` of the PR's `0001-*.patch` against the r334 tree **fails** on the assert context
  (`error: while searching for: … GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);`).
* `RMS_NORM + MUL + ROPE` stays F32-only (`ggml_cuda_should_fuse_rms_norm_mul_rope` declines BF16) — the
  same as the fork point, so there is no divergence to reconcile there either.

Hardware check, gfx1201 / ROCm 10.1, default flags, `HIP_VISIBLE_DEVICES=0`:

```
$ ./build-rocm/bin/test-backend-ops test -b ROCm0 -o RMS_NORM_MUL_ADD
  124/124 tests passed
$ ./build-rocm/bin/test-backend-ops test -b ROCm0 -o RMS_NORM
  121/121 tests passed
```

and the BF16 arms in `RMS_NORM_MUL_ADD` run, including the exact shapes the PR's conditions cover:
`type=bf16,weight_broadcast=1,type_w=f32` and `type=bf16,…,type_w=f32,type_add=bf16` (the BF16 residual).

**Not reproduced here:** the contributor's r324 abort and his r324 + patch `113/113`.  No gfx1030 device is
in the maintainer's fleet, and the r324 state is established from the source at that tag, not by a run.
His numbers are consistent with that source and are not disputed.

## 4. What was asked of the contributor (the correction)

1. Cut `wip/` (and `beta/`, `optional/`) artefacts against the **current** release tree
   (`release.json.tree`), not against the tree of the revision the bug was found on — a stale patch is
   neither applicable nor gateable.
2. Re-run the repro on that tree **before** offering the fix, and name the release revision (`v17-…-rNNN`)
   in the report, because a re-base can fix a default-set bug without any patch (`INVARIANTS.md` r335).
3. `TODO` #361 (the r334 RDNA2 items on his V620) is the more valuable re-run, and it covers this check on
   real RDNA2 hardware; it now names the BF16 arms explicitly.

## 5. Files

| file | what | sha256 |
|---|---|---|
| `README.md` | the contributor's brief, verbatim below the status header | `be2cd772bf09e90abc4438939fb3e737279019eb6a15f2c48335a5030d2d8efa` |
| `0001-rdna-boosts-BF16-RMS_NORM-MUL-ADD-no-longer-aborts-t.patch` | the contributor's `git format-patch` against the r324 tree, **not applied** | `3127c4cda3764bf2ca148e506fcd09cc0136a2dfa2fcda0149da659252d36361` |

Both files are byte-identical to the ones in PR #339's commit `7cb192e5` (`git show refs/pr/339:…`); the only
edit is the status header at the top of `README.md`.
