# CLOSED (r252, 2026-11-10) — the `MUL_MAT_ID` duplicate-id re-cut: the candidate is done, and it was **not** the kernel work

`TODO` #287, retired.  The deliverable is
[`upstream/mul-mat-id-dup/UPSTREAM-PR-mul-mat-id-dup.patch`](../../../upstream/mul-mat-id-dup/UPSTREAM-PR-mul-mat-id-dup.patch)
(sha256 `00b43499f7f55ce87c39395a43fa083c81ee083a26ec11cf706907f38dbf5ca3`, re-cut against
`a691b2345`), not anything in this directory.  `candidate-does-not-pass.patch` (sha256 `2e39b670…`) is kept
only as the record of the first attempt.

## The first attempt's diagnosis was wrong, and that is the finding

This campaign said the remaining work was "the MMQ/MMF **ids kernels** — the column tile iterating `tile_spec`
seams, the per-tile `tile_spec` read, and `mul_mat_f_ids`'s store loop using `continue` (not `return`) with `C`
re-zeroed per tile", because the delivery carries that code and the candidate still failed.

**It does not need writing.**  Upstream `#31047` (the PR that made the old patch unappliable in the first
place) *already* rewrote `MUL_MAT_ID` as a `tile_spec` stream, and its rewritten MMQ/MMF ids kernels already
handle an arbitrary compact column count:

* `mmq.cu` already launches `ntj = ceil(n_tokens*n_expert_used / J) + n_experts` blocks and passes `J` as the
  seam width to the helper; `mmq.cuh` already reads `tile_spec[jt]` per tile, reads `ids_dst` through shared
  memory and has a `tile_spec`-aware stream-k fixup.  Nothing in `mmq.*` was touched by the final patch.
* `mul_mat_f_ids` is **one block per `tile_spec` entry** (`blockIdx.y` is the tile), so its store loop's
  `return` is correct there — the `continue`/per-tile-`C` requirement is from the pre-`#31047` kernel.

The two real defects, both in the *helper* and the *dispatch*, were:

1. **`mm_ids_helper`'s offsets were computed before `nex_prev` was warp-reduced.**  The first attempt reduced
   `nex_prev` *after* pass 2 had already used it (and reduced `it_compact`, which is already warp-uniform), so
   every expert wrote its compact columns at a lane-local offset: `f16` numerically wrong at every width, and
   `q8_0` page-faulting after the MMQ tile read a garbage `tile_spec`/`ids_dst` — the fault was a *symptom*,
   exactly like the original `quantize_mmq_q8_1` fault was a symptom of the dedup.
2. **The compact-MMF column tile did not match the seam grid.**  `mmf.cu` builds `tile_spec` with a 16-column
   stride while `mul_mat_f_switch_cols_per_block` dispatched `cols_per_block = ncols_dst` and both `mmf` gates
   still required `ncols_dst > 16`.  That is the delivery's `#304` bug (closed in block 08) — a *base* defect
   that also breaks the non-dup narrow `MUL_MAT_ID`, not a duplicate-id defect.

## Validation (gfx1201 / ROCm 10.1.0, pristine `a691b2345` + the re-cut, 1 x R9700)

| gate | result |
|---|---|
| `test-backend-ops -o MUL_MAT_ID -p 'dup=1'` | **16/16** (this campaign's candidate: 4 numeric FAILs + a GPU page fault) |
| `test-backend-ops -o MUL_MAT_ID` | **947/947** |
| `test-backend-ops -o MUL_MAT` | **2517/2517** |
| `scripts/check-gpu-faults.sh --since … --log …` (journal + 3 run logs) | clean |
| launch census (temporary `tile_spec` read-back, reverted) | covered columns == `n_tokens*n_expert_used` on every dup shape; the widest expert range is 60 columns for 9 tokens / 475 for 64 (only possible with a repeated id); both the MMQ and the MMF launch reach it |
| delivery-neutral | the delivery tree `5fbe6aff5` is unchanged; re-verified **16/16** on that tree (its dup cases are already enabled in block 08) |

Census detail (compact columns expected/covered, tiles, widest expert range):

```
mmq/mmf  tokens= 9 n_used=10 experts=28   : 90  / 90  tiles=34 (23)  max_expert_range= 60 experts_over_tokens=1
mmq/mmf  tokens=16 n_used=10 experts=28   : 160 / 160 tiles=38 (29)  max_expert_range=116 experts_over_tokens=1
mmq/mmf  tokens=33 n_used=10 experts=28   : 330 / 330 tiles=49 (40)  max_expert_range=237 experts_over_tokens=1
mmq/mmf  tokens=64 n_used=10 experts=28   : 640 / 640 tiles=48 (42)  max_expert_range=475 experts_over_tokens=1
mmq/mmf  tokens=33 n_used=10 experts=1025 : 330 / 330 tiles=1046     max_expert_range=248 experts_over_tokens=1
mmq/mmf  tokens=33 n_used=10 experts=2050 : 330 / 330 tiles=2071     max_expert_range=237 experts_over_tokens=1
```

## Left behind

* The GPU journal window `2026-11-10 15:09:01` (the pre-fix reproduction's page fault, `pasid` 24738) is dirty,
  exactly as the 2026-11-08 window was; a gate that starts before **`2026-11-10 15:14:00`** needs
  `check-gpu-faults.sh --since` past it.
* Not filed upstream.  Filing the candidate is the maintainer's call and has never been made.
* No performance claim: this cut changes the same instructions as the previous one (which measured within
  noise) plus a narrower-tile **fix**.

**Not part of the delivery; nothing here is applied by `scripts/apply-all.sh`.**
