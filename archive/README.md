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
| `work/{beta-integration,bf16-native-prefill,build-time-regression,closing-the-gap,issue-30-mtp-decode-regression,issue-44-hc-combine-oracle,issue-45-band-port,kq-derived-tile,kq-mask-derived-ab,mtp-journey-2026-09-17,prefill-arrangements,q8-prefill-tuning,reasoning-aware-mtp,tiled-gdn,per16-f16-mma,mmq-pipeline}/` | the completed/abandoned `wip/` trees, archived **2026-09-26** (each is a closed campaign or a dormant scoping record; `nwarps/` is deliberately left in `wip/`) |
| `work/mmb-general/` | the **former top-level `beta/` record** of the `mmb`/`qsa3`/indexer campaign (the 28 patches are folded into the 16 delivery blocks; `apply-beta.sh` removed), moved here **2026-09-26** so `main`'s top level no longer carries a `beta/` directory |
| `work/wip-archive/` | the older `wip/archive/` group (hybrid-allreduce, managed-ngrams, qwen35moe-prefill, qwen4exp), preserved with its original grouping |

**2026-09-12 consolidation:** the completed `wip/` trees and the promoted block-15 beta record were
moved here; `wip/` now holds only the active `wip/iq4nl-prefill/` handoff.  Cross-references in the
repo were rewritten to the new paths (`wip/<x>` → `archive/work/<x>`, `archive/work/block-15-campaign-wins` →
`archive/work/block-15-campaign-wins`).  Some git-ignored run logs under `work/strix-halo/kvzero/runs/`
still carry the old absolute paths because they are verbatim outputs of the runs; they are not tracked.

**2026-09-26 consolidation:** every remaining campaign under `wip/` except `nwarps/` was closed and
moved here — `per16-f16-mma` and `mmq-pipeline` (both negative/parked kernel experiments),
`beta-integration`, `bf16-native-prefill` (closed negative), `build-time-regression` (fixed),
`closing-the-gap` (consolidated into `archive/work/mmb-general/`), `issue-30-mtp-decode-regression`,
`issue-44-hc-combine-oracle`, `issue-45-band-port`, `kq-derived-tile`, `kq-mask-derived-ab`,
`mtp-journey-2026-09-17`, `prefill-arrangements`, `q8-prefill-tuning`, `reasoning-aware-mtp` and
`tiled-gdn`.  `wip/` now holds **only `nwarps/`** — the per-M `nwarps` impurity, the one item
deliberately left open.  Cross-references in tracked docs were rewritten (`wip/<x>` →
`archive/work/<x>`, `wip/adaptive-mtp-ceiling-scaling/` → `archive/work/…`); verbatim profiler logs
and CSVs under `work/bf16-native-prefill/profiles/` and `work/mtp-journey-2026-09-17/` keep the old
absolute paths because they are unedited tool output.  `per16-f16-mma` and `mmq-pipeline` lived only
on wip branches; their content was materialised into `archive/work/` and the branches retired.

**2026-09-26 (later) — `beta/` archived too:** the top-level `beta/mmb-general/` record was moved to
`archive/work/mmb-general/` (the campaign has been folded into the delivery since `v16-84e76d8a2-r8`
and is no longer applied separately), so `main` no longer has a `beta/` directory; the redundant
`beta-integration` branch (fully contained in `main`) was retired.

If you are working on the CURRENT delivery, you do not need to read these —
they exist so the historical record is preserved without cluttering
`README.md` / `MANIFESTS.md` / `BASELINE.md`. The exception: when the block
structure or validation history is cited from the lean docs, the full story
is here.
