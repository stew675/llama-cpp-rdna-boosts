# archive/ — history and closed work

Everything that is NOT the here-and-now delivery lives here, out of the
top-level docs (per the 2026-08-29 restructure: top-level docs are lean and
current; history is archived, not deleted).

| path | what |
|------|------|
| `docs/validation-history.md` | all dated validation records moved out of MANIFESTS.md (block-02 segregation, issues #1/#2, the gfx1100/gfx1151/gfx1201 records, the block-11 perf profile, the RDNA3_0 wins, the retired block-structure churn, the old `block/*` tag lineage) |
| `docs/baseline-history.md` | the old checkpoint cuts, drift fixes, validation records, older `baseline/*` branches, and the original `chunked-gdn` source-commit provenance, moved out of BASELINE.md |
| `docs/HANDOVER-2026-09-06-pre-reboot.md` | the pre-reboot (2026-09-06) session handover that used to sit at the repo top level — gfx1201 multi-ubatch prefill regression root-cause + fix and the open tensor-mode follow-up (dev-session log, not delivery) |
| `work/fused-stage-pacing/` | the CLOSED fused-stage + host-side-pacing experiment (sessions 8-9), preserved for re-evaluation after a future ROCm update |
| `work/block-15-campaign-wins/` | the attention-memory campaign (W1-W4, V3-V5) — its patch was **promoted into the delivery as block 15 on 2026-09-12**; kept as the promotion/gate + beta-tester record |
| `work/{strix-halo,qwen4exp,kv-quant-purity-followups,arch-independent-memory,sm-tensor-plain-vs-spec,issue-25-mtp-batch-width,issue-30-mtp-decode-regression,items-6-10-wrapup,kv-sign-leak,gdn-rs-rollback,shadow-warnings,block15-dense-arm,tools}/` | the completed `wip/` trees, archived **2026-09-12** (each is the record for a closed item or campaign; the item-4 harness lives under `work/strix-halo/qsa-item4/`) |
| `work/wip-archive/` | the older `wip/archive/` group (hybrid-allreduce, managed-ngrams, qwen35moe-prefill, qwen4exp), preserved with its original grouping |

**2026-09-12 consolidation:** the completed `wip/` trees and the promoted block-15 beta record were
moved here; `wip/` now holds only the active `wip/iq4nl-prefill/` handoff.  Cross-references in the
repo were rewritten to the new paths (`wip/<x>` → `archive/work/<x>`, `beta/block-15-campaign-wins` →
`archive/work/block-15-campaign-wins`).  Some git-ignored run logs under `work/strix-halo/kvzero/runs/`
still carry the old absolute paths because they are verbatim outputs of the runs; they are not tracked.

If you are working on the CURRENT delivery, you do not need to read these —
they exist so the historical record is preserved without cluttering
`README.md` / `MANIFESTS.md` / `BASELINE.md`. The exception: when the block
structure or validation history is cited from the lean docs, the full story
is here.
