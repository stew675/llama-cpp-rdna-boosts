# archive/ — history and closed work

Everything that is NOT the here-and-now delivery lives here, out of the
top-level docs (per the 2026-08-29 restructure: top-level docs are lean and
current; history is archived, not deleted).

| path | what |
|------|------|
| `docs/validation-history.md` | all dated validation records moved out of MANIFESTS.md (block-02 segregation, issues #1/#2, the gfx1100/gfx1151/gfx1201 records, the block-11 perf profile, the RDNA3_0 wins, the retired block-structure churn, the old `block/*` tag lineage) |
| `docs/baseline-history.md` | the old checkpoint cuts, drift fixes, validation records, older `baseline/*` branches, and the original `chunked-gdn` source-commit provenance, moved out of BASELINE.md |
| `work/fused-stage-pacing/` | the CLOSED fused-stage + host-side-pacing experiment (sessions 8-9), preserved for re-evaluation after a future ROCm update |

If you are working on the CURRENT delivery, you do not need to read these —
they exist so the historical record is preserved without cluttering
`README.md` / `MANIFESTS.md` / `BASELINE.md`. The exception: when the block
structure or validation history is cited from the lean docs, the full story
is here.
