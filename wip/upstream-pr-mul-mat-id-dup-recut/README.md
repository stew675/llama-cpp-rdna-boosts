# `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` re-cut onto `a691b2345` — **WIP, does not pass**

`TODO` #287.  **Status: a partial re-cut, recorded so the next session starts where this one stopped.**  The
candidate patch in this directory **does not fix the bug** and must not be filed: it is here because the *work*
(half of it) is done and the remaining half is now precisely identified.

## What the item asked for

Re-cut `upstream/UPSTREAM-PR-mul-mat-id-dup.patch` against the new base, "confirm it applies to unadulterated
upstream and still removes the duplicate launch (a launch census before/after)".  The old patch was cut against
`097cf33d9`; upstream's `#31047` rewrote the same `MUL_MAT_ID` tile scheduling (the `tile_spec` grid), so the old
patch was already rejected by `git apply --check` on `a691b2345` (`mmq.cuh:995`).

## What was done (in `candidate-does-not-pass.patch`, sha256 `2e39b670…`)

| piece | state |
|---|---|
| `mmid.cu` — `mm_ids_helper` rewritten for **per-occurrence** compaction on upstream's `tile_spec` writer (no per-token staging buffer, `mm_ids_helper_store` and the shared-memory sizing gone) | **done** — ports cleanly onto upstream's helper, and the two-pass form is the delivery's (block 00), which upstream's seam writer already matches |
| `mmf.cu` — build the compact id map for **every** `MUL_MAT_ID`, not only `ncols_dst > 16` | **done** (the delivery's own delta is exactly this line) |
| `mmf.cuh` — select the compact kernel for **every** `MUL_MAT_ID` (`has_ids_data`, not `has_ids_data && ncols_dst > 16`) | **done** (likewise the delivery's own delta) |
| `tests/test-backend-ops.cpp` — re-enable the `[TAG_MUL_MAT_ID_DUPLICATE_ID]` cases that upstream `#if 0`s "until CUDA backend support is added" | **done** |
| the **MMQ** compact grid/tile handling for duplicates | **NOT DONE** — this is what still fails |

## What happens with the candidate

Built on a pristine `a691b2345` (`BUILD_DIR=build-rocm-dup`), `test-backend-ops -o MUL_MAT_ID -p 'dup=1'`:

* **`f16` cases FAIL numerically** at every width (`n = 9/16/33/64`, ERR ≈ 0.4-1.8 against a 5e-4 tolerance) —
  the compact `mul_mat_f_ids` path is selected but computes the wrong columns;
* **`q8_0` at `n = 9` page-faults** the GPU inside
  `mul_mat_q<(ggml_type)8, 16, false, (ggml_prec)30, false>(…, ggml_cuda_mm_id_tile_spec const *, …)`
  (`Memory access fault … Reason: Page not present or supervisor privilege`).

So the compaction is a necessary but not sufficient piece: upstream's **MMQ/MMF ids kernels** still cannot
consume a compact column list that is wider than the token count.

## The missing piece, and where the reference is

The delivery's own `ggml/src/ggml-cuda/mmq.cu` + `mmq.cuh` carry the equivalent (`+589` / `+481` lines against
`a691b2345`), but they are **entangled with the delivery's fusion work** (the gate+up pair, the routed-compact
dispatch, the seam writer) and cannot be lifted out as a patch.  The reference for the *ids* part is:

* `mmq.cu`'s compact launch: `ntj = (ne_get_rows + J - 1)/J + n_experts`, `nchannels_y = 1` on the ids path,
  `ncols_max` semantics, and the per-tile `tile_spec` read;
* `mmq.cuh`'s `mul_mat_q` ids branch (the column tile iterating over `tile_spec` seams rather than one tile per
  block, with the barrier between iterations);
* `mmf.cuh`'s `mul_mat_f_ids` (its store loop must `continue`, not `return`, and `C` must be re-zeroed per tile).

The delivery's absolute paths, at fork `2ff384571`: `ggml/src/ggml-cuda/{mmq.cu,mmq.cuh,mmf.cuh,mmid.cu}`.

## A delivery finding that came out of this (separate handle)

Running the **delivery's** tree (r236, fork `2ff384571`) with those `#if 0` dup cases temporarily enabled gives
**15/16**: every `q8_0`/`q4_0` case and the wide `f16`/`offsets` cases pass, and the single failure is
`MUL_MAT_ID(type_a=f16, n_mats=28, n_used=10, n=9, k=256, dup=1)` (ERR 0.374).  So the delivery's duplicate-id
handling survived the re-base, with one narrow-`f16` hole → recorded as its own `TODO` entry (`#304`), because it
is a delivery correctness gap and this PR job is not the place to fix it.

**The journal is now dirty for that window**: the `q8_0`/`f16` runs above page-faulted on 2026-11-08 at
**16:29-16:37** (`pasid` 3083/3137).  A gate that starts before that needs `check-gpu-faults.sh --since` past it
(`#294`).

## How to continue (next session)

```bash
git -C ~/llama.cpp worktree add /tmp/upstream-check a691b2345     # a pristine base to cut/check patches
cd /tmp/upstream-check && git apply /path/to/candidate-does-not-pass.patch
# build it in the MAIN worktree instead (ccache keys on the source path):
git -C ~/llama.cpp checkout a691b2345 && git -C ~/llama.cpp apply <patch>
BUILD_DIR=build-rocm-dup ~/bin/build-llama-rocm
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-rocm-dup/bin/test-backend-ops -o MUL_MAT_ID -p 'dup=1'
# ... and `git checkout -- . && git checkout rebase-r236` afterwards; `master` in ~/llama.cpp must stay pristine
```

Then iterate on `mul_mat_q`'s ids launch for the duplicate case (start from the q8_0 `n = 9` page fault — it is
the sharpest signal), and on `mul_mat_f_ids`'s store loop.  Re-run the **delivery's** 15/16 result as the
regression baseline: a re-cut that passes upstream's 16 cases should also keep the delivery green.

**Acceptance for filing**: `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` 16/16 on the patched pristine base, the
full `MUL_MAT_ID` suite green, `MUL_MAT` green, no GPU fault, and a launch census showing the duplicate columns
are covered by the launched tiles.  Delivery-neutral: it must not change the delivery tree.

**Not part of the delivery; nothing here is applied by `scripts/apply-all.sh`.**
