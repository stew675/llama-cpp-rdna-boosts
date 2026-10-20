# Handover prompt — fold `wip/nwarps` into a block and push a release

Paste the block below into the main development host's session (the box that owns `patches/`,
`release.json` and the release flow).  Everything it references is in this directory.

---

```
Task: fold the resolved wip/nwarps campaign into the delivery block set and ship it.

Read first, in this order:
  1. wip/nwarps/README.md      -- the finding: the RDNA3_0 per-shape mmvq nwarps impurity was the
                                  fusion exclusion, not a missing reduction-mapping derivation; purity
                                  and perf tables; the shapes each rule arm acts on.
  2. wip/nwarps/PROMOTION.md   -- the handover: artifacts + sha256, which block owns which file, the
                                  fold recipe, the ENVIRONMENT.md / GREEDY-PURITY.md / patches/README.md
                                  edits, the per-box gates, rollback.
  3. AGENTS.md "Pushing policy", "Default-on policy", "Editing the docs".

Deliverable to fold:
  wip/nwarps/patches/per-shape-nwarps-rdna3-0-promoted.patch   (sha256 55a7cd76..., default-ON)
  wip/nwarps/patches/per-shape-nwarps-rdna3-0.patch            (sha256 eb32d36c..., default-off: do
                                                                NOT fold this one, it is inert unless
                                                                GGML_CUDA_MMVQ_RDNA3_SMALL_M is set)

It changes exactly three files: ggml/src/ggml-cuda/{mmvq.cu,mmvq-rdna.cuh,mmvq-rdna-params.cuh}.
Those two headers are created by block 08, so the change belongs in block 08 (mmvq.cu is additionally
touched by blocks 07/09/10 -- replay conflicts are mechanical, keep both sides).  Default-on with
GGML_CUDA_MMVQ_RDNA3_SMALL_M=0 as the kill-switch, per the default-on policy (the promotion commit is
already the inversion, so no separate "enable" step is needed).

Steps:
  1. Clean clone at release.json.base -> scripts/apply-all.sh . -> git am the promoted patch (13th
     commit) -> fold into block 08 (rebase -i fixup; or keep it as its own commit if the replay is
     messy -- the delivered tree is what ships).
  2. Apply the doc edits from PROMOTION.md §3: the ENVIRONMENT.md §7.4 row (ready to paste),
     the GREEDY-PURITY.md §19/§25 invariant ("a shape-scoped launch parameter must be forwarded to
     every kernel that reproduces that launch"), and the block-08 note in patches/README.md.
  3. scripts/make-patches.sh, scripts/make-release.sh, scripts/validate-set.sh (never hand-edit
     patches/ or the hashes in release.json).
  4. Gates (PROMOTION.md §5): gfx1201 MUST be byte-identical to r91 (the scope is RDNA3_0-only -- a
     diff there means the scope leaked); gfx1151 build + the usual per-arm FAIL/PASS set; on gfx1100
     the W=1..8 probe on all four models (all eight KV types on at least one), the adaptive-MTP
     four-axis gate (-n 3000, reasoning pinned), llama-batched-bench -npl 1,4,8, PPL parity, and
     scripts/gate-prefill-logits.sh; plus GGML_CUDA_MMVQ_RDNA3_SMALL_M=0 reproducing the r91 hashes.
     Expected gfx1100 evidence is tabulated in PROMOTION.md §4.
  5. Release: WORKLOG entry (YYMMDD.VVV, newest first), move wip/nwarps/ -> archive/work/nwarps/ and
     drop its row from wip/CAMPAIGNS.md (no stub left behind), tag, push the tag, refresh the personal
     fork's rdna-boosts branch from a clean apply at release.json.base (--force-with-lease, fork only,
     never upstream), and let the radiance worker watcher publish the ROCm 7.14.1/10.1 images.

Report before shipping: any gate that is not green, any deviation from PROMOTION.md (especially a
scope leak on gfx1201/gfx1151), and any RDNA3_0 model whose numbers disagree with PROMOTION.md §4 --
wait for a decision rather than finishing the release and mentioning it afterwards.  Known caveats to
carry into the release note: the always-narrow Q4_0/Q4_1/Q5_0/Q5_1 arm rests on gemma-26B alone and
reverses a 2026-08-28 shallow sweep; part of the 35B MTP gain is an acceptance shift (0.653->0.750
prose, 0.801->0.851 code), the plain-decode numbers are the kernel-time-only claim; and the mmvq.cu.o
object grows 13.64 -> 15.36 MiB (+12.6 %) on gfx1100.
```
