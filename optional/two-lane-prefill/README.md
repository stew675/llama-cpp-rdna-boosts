# `two-lane-prefill` — `LLAMA_PREFILL_LANES`: consecutive-ubatch prefill overlap

**Optional artifact, not part of the delivery.  Optional block `0100`; PORTED onto `r369` (2026-11-27), verified on `r353`.**
r373 (18 official blocks `0000`-`0017` + temporary `0300`, default tree `95dbb18aa1e104547226407f6b29028e848366e3`, default tip `20ed444e9`):
RE-CUT, because r373 amends block 16 (`tools/server/server-context.cpp`) and a re-cut re-exports the artifact against the new default tree.  The artifact does not touch that file, so nothing was reconciled -- the diff body is the r372 one and only the
`From` id moved, and the sha256 is `f70aae35ee86457254a16ec17cc4216cbc96809c42b40f3d384a6f73f20923cf`; producing tree
`3a56f076702a8b414a36d12c38e1eea92ce262d3` alone and `3fa3b2a7c535de68bcc56429e2d869d82303324b` cumulative (both re-verified by
`scripts/validate-set.sh`; the `all`-target build of the cumulative tree is the r372 gate-47 run below -- this re-cut moves no artifact source line, so it was not re-built).  (r372, superseded: default tree `7c1728b37d8b914724e62156559eddf0301c062c`, default tip `5e844cbc6`):
applies strictly on the r369 port, **nothing to reconcile** -- r372 amends block 05 (`ggml/src/ggml-cuda/fattn*.{cu,cuh}`), none of this artifact's eight files, so its own diff and export are untouched and the sha256 is **unchanged** at
`ce21bccde08257b3f3e536141651ad63a8d22fd84ef52eeabb5d50f16c40e032`; producing tree
`bcdf4dfbc57ca5d1520a6b20ceb7d2983b83b3b3` alone and `d74527df00cb751fdf695ef3d7be38346c485f87` cumulative, and
`scripts/gate-optional-build.sh` (gate 47) **PASS** on the cumulative tree (`all` target, 212 s).  (r369, superseded: default tree `3518351ef19fef971683fdca1345ab5fcfa1c336`, default tip `2347ee8b3`:
applies strictly after a **PORT** -- it does NOT apply mechanically, see the r369 entry below), sha256
`4c318b23a1406f340b0eba49d88388ed4247f3358947d8293309582f1cc7a86d`, trees
`659b380ddcda232be74377adbf9981ffabcfe981` alone and
`83b34247b47c7f006717e792af2dccbaa91fedcd` cumulative.  (The r367 cut, superseded: default tree
`2ff670ba83e5160abf16d20355aeded52cb744f0`, sha256 `4093ef67...`, trees `9f9acf91...` / `c5ad54d9...`.)  The optional-applied tree is now also **built** by
`scripts/gate-optional-build.sh` (gate 47, added for issue #371) -- this artifact's own port verification was
never a compile.
The port verification and the numbers of record are the r353 run below, against the r353
default delivery tree `27db9bfd4ef7f41c7df60f704410de28863beac3` = official blocks `0000`-`0016` plus the
temporary `0300`; the r320 cut and its port verification are kept below; the r319 cut is at the bottom of the list, against the r319 default delivery tree
`9f53a85322eb8c0327c350fca89fe5af76a30806`; the r318 cut is at the bottom of the list, against the r318 default delivery tree
`d586e2a5fdfda303ea8363dfac20ca72784bf90a`; the r307 cut is at the bottom of the list, against the r307 default delivery tree
`f91aa81202fd32d060f9afb0f9c54824116758b9`; the r288 re-cut is at the bottom of the list, against the r288 default delivery tree
`79d18f41b9ab9b235d35fd3a5034858eddd94844` = official blocks `0000`-`0015` plus the
temporary `0300`; the r259, r254, r247, r246, r243, r241 and r229 cuts and their measurements are kept below
as the dated record).  The r288 re-cut is **mechanical**: the r285 patch applied STRICTLY on the r288 default
tree with nothing to reconcile, so the artifact's diff is byte-identical to the r285 cut and the r285 numbers
below stand (r285 -> r288 moved only the KV-window/spill, 2-D state-copy, `llama-bench` and `fit` code, outside
this artifact's own mechanisms).  The r288 block-number re-org is also what gave it the `0100` number; it was
an unnumbered `two-lane-prefill.patch` before.  The r285 rebase was a **port, not a replay**: the delivery
rewrote both files the artifact owns between r259 and r284 (D10's "do not upload an input that is not read"
now gates both copy passes; the copy helper is keyed per input slot; the decode loop grew the nextn/h_nextn
device path and the `has_next`-gated readback).  The artifact's own mechanisms are unchanged (arm gate,
sub-ubatch split, pair/hazard analysis, per-lane output extraction); `What changed in the rebase` item 16 says
exactly what had to be reconciled, and item 19 records the r288 re-cut.

From **PR #268, #279, #284, #296 then #301 by [@slavap](https://github.com/slavap)**, rebased onto the delivery,
reviewed and amended (see *What changed in the rebase* below).  The submitter's own description and measurements
are in the PRs; this file records what *this repo* verified, what it measures here, and what it cannot reach.

* `0100-two-lane-prefill.patch` — the artifact.  sha256
  `4c318b23a1406f340b0eba49d88388ed4247f3358947d8293309582f1cc7a86d` (the r369 PORT; every earlier cut is
  recorded below)
* **r369 PORT (current)** — the artifact vs r369's staging pipeline.  r369 rewrote the very function this
  artifact refactors (`ggml_backend_sched_compute_splits`: a per-split `compute_split(sched, split_id, st)` + a
  driver, against r369's per-graph pipeline prologue / per-split consume / epilogue), so the re-cut was a **port**,
  not a re-export: the pipeline's prologue and epilogue are now `pipe_begin()`/`pipe_end()`, the per-split pipeline
  consume lives inside `compute_split` (whose cursor state is this artifact's per-lane `compute_state`), the pair
  path runs each lane's prologue before its splits and its epilogue after them (both call sites), and the
  pipeline's stageable-split list moved into the scheduler (`pipe_host`/`pipe_n_host`) because a pair has one
  scheduler per lane.  **Two behaviour changes, both measured and both narrow**: the pipeline is DISABLED for a
  pair (its slots are per BACKEND, and a pair shares every backend between its lanes, so two lanes cannot both own
  slots 0..depth-1 -- one INFO line says so), and a **host-expert tier is refused by the arm gate**
  (`LLAMA_PREFILL_LANES ignored (... host_experts = 1)`) because the pair **LIVELOCKS** there: 3 x R9700,
  `-sm layer`, 35B-A3B Q8_0 `-ncmoe 40 -ub 2048`, lane 0 idles at 3 W while lanes 1/2 spin at 100 %/~90 W with no
  progress.  That arm was never reachable in this artifact's own measurements and its record measures staging
  there as a **14.6 % loss**, so nothing is given up.  Re-verified on this port (3 x R9700, gfx1201): default path
  inert -- `-sm tensor` 4B Q8_0 text `b065f37593b3` for lanes unset/1/2; the reachable layout engages and still
  wins -- `-sm layer -ts 1,1,1 -ot 'blk\.0\.attn_q\.weight=ROCm0' -b 2048 -ub 2048`, 35B-A3B Q8_0, 5256-token
  prefill **5761.2 -> 6811.9 t/s (+18.2 %)** with `two-lane prefill enabled (LLAMA_PREFILL_LANES)`, greedy text
  `e2dde701fc04` for lanes unset/2; the refused host-expert arm completes at `e2dde701fc04` too; and
  `scripts/gate-prefill-logits.sh` on the optional-applied build reads **mean KLD 0.000000 / same-top-p 100.000 %**
  (bit-inert on the default path).  `scripts/gate-optional-build.sh` (gate 47) builds the cumulative tree
  `83b34247b`, and `validate-set.sh` is green.
* **r366 re-cut verification (current)**: applies strictly at the r366 default tree `82afebd108d63b6220884d41080009561f0821e2` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `489810652`), sha256 `423ce70fafd17535cbd3646139062bb93b80f84bb859b6866e241aeb5658f361` -- the sha moved because the exported commit id did, not the artifact's own diff: r366 amends block 10 (`ggml/src/ggml-cuda/ggml-cuda-slab.cu`, the slab's wide band is armed by the prefill pass rather than by the carved ring hole), which no file here touches, so the cut was re-exported against the new tree and applied with nothing to reconcile; producing tree `1d79094920a15fc324c6d998d8606c36c0175134` alone, and with the other optional artifact on top, the cumulative tree is `f54c5d78e853f312b34f0ab6c5bcd4004e500667`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the r353 port verification below is still the numbers of record (the artifact's own code did not change).
* **r365 re-cut verification**: applies strictly at the r365 default tree `21efb08247edc142c649d2dcda771abb73fee810` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `7ad7f533c`), sha256 `6d9bc2a5b38d27e0fedba9ed2e147ba51ec260e2bf34831770f3487f77374b9b` -- the sha moved because the exported commit id did, not the artifact's own diff: r365 amends block 16 (`tools/server/server-{context,queue,slot-sched}.*`, the server's KV admission), which this artifact's server hunks now sit beside, so the cut was re-exported against the new tree and applied with nothing to reconcile; producing tree `3d7804c3db45487f9973a98bb5c52f193476d1bb` alone, and with the other optional artifact on top, the cumulative tree is `43ee504d461f63e663db7426e28273756bc00ee1`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the r353 port verification below is still the numbers of record (the artifact's own code did not change).
* **r360 re-cut verification (current)**: applies strictly at the r360 default tree `4674dd9f2e83894895d71b7e987a06d80fa70813` (default tip `3be8b9bd1` (the deterministic re-cut of the same tree); sha256 `33634cbcedf7c7ff5d19d6b70dd86517bb91f492a16542303ee7b44ffb7c9e57` -- the sha moved because the exported commit id did; r360 amends block 09, which owns `src/llama-context.cpp`, and this artifact edits that file too, so the cut was rebased and re-applied strictly), producing tree `831bc116a47559c4892b6f66f95fffbf869c76df`; with the other optional artifact on top, the cumulative tree is `8bd172df5d98e698a6868d645afa626cdfc6c19a`.
* **r355 re-cut verification (superseded by r360)**: applies strictly at the r355 default tree `4244d077effe09a44a4763dd0cabbbf8f37bdf05` (sha256 `1a7be1dd5e309ea4958a3fb4f616d3f541b16c0edfdb5e74b7eb3c0f9b82f020` -- **unchanged**, unlike a re-cut: r355 amends block 10 only, inside `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` (the GDN output-norm fold's matcher takes the skip's decision and its reader counts are one walk), none of this artifact's files, so the patch's own diff and export are untouched and it needed no rebase), producing tree `44a2de757fb4d9551cdce7bf92f9c22e16b6ad87`; with the other optional artifact on top, the cumulative tree is `cdeae8f29c7f419332c019d0d3d252aeee8aaa18`.  Only the trees moved, because a produced tree is `default tree + this patch` and the default tree moved.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r353 re-cut verification (superseded by r355)**: applies strictly at the r353 default tree `27db9bfd4ef7f41c7df60f704410de28863beac3` (sha256 `1a7be1dd5e309ea4958a3fb4f616d3f541b16c0edfdb5e74b7eb3c0f9b82f020` -- **unchanged**, unlike a re-cut: r353 amends block 10 only, inside `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` / `common.cuh` / `ggml-cuda.cu` (the GDN output-norm fold's SILU widening), none of this artifact's files, so the patch's own diff and export are untouched and it needed no rebase), producing tree `7974c67b096691a9669c1503b6e85c2b75bb46a0`; with the other optional artifact on top, the cumulative tree is `790e050d12ea05858ced2ebe50b46e38b3764f0c`.  Only the trees moved, because a produced tree is `default tree + this patch` and the default tree moved.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r351 re-cut verification (superseded by r353)**: applies strictly at the r351 default tree `8851b0d02b74060f462088617e8a7ff328c8c5ed` (sha256 `9d9793a294688d011e0a5ef41c145fa6f5dea3aaa933012da81cfd6c97559f20`), producing tree `7934579876e5103b9318accd6045bfdac70a0cc2`; with the other optional artifact on top, the cumulative tree is `91f52b417643df489dcba548894b5a394ad1dc8f`.  Mechanical re-cut (`git am`, no `-3` needed): r351 amends block 17 only (`ggml/src/ggml-cuda/ggml-cuda.cu` -- two wiring fixes for issues #353/#354, plus the block-0300 offset shift), and this artifact's only hunk in that file is far from both edits, so its own diff is unchanged and the sha256 moved only with the exported commit id and the re-anchored hunk offsets.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r350 re-cut verification (superseded by r351)**: applies strictly at the r350 default tree `83bdb702510693916340031ba2cb9e42eb5dfd04` (sha256 `f7c77bb967c2618a1b5cfbabcf653b2a5960a2dc21e369d460560dc8eea40dae`), producing tree `b8305aee59591155af6e294b7044081ce15d94e0`; with the other optional artifact on top, the cumulative tree is `8dff92e0062165631b6942b6ae7de63c67fe2f6b`.  Mechanical re-cut (`git am`, no `-3` needed): r350 amends block 17 only (`common/speculative.{h,cpp}`, `common/common.{h,cpp}`, `tools/server/server-context.cpp` -- the speculator on the two-arm server, `TODO` #364), none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r345 re-cut verification (current)**: applies strictly at the r345 default tree `0d2f70e6c44eb246a9c71f29e015d6ebe98f214d` (sha256 `8c33779f6fa0d48ebba24950124553f314a88db86097d44f5c9c79f41fcdbd3c`), producing tree `d794be3e769e520e16e7e6dfd95c1ab462228d69`; with the other optional artifact on top, the cumulative tree is `3cf56a5af604cc7ee67336b70248e439a17300d4`.  Mechanical re-cut (`git am`, no `-3` needed): r345 amends block 09 (`ggml/src/ggml-cuda/moe-expert-cache.cu`, the MoE device gather's used-expert bitmap, issue #350) plus the wording of block 12's `device gather:` census line, none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r344 re-cut verification (superseded by r345)**: applies strictly at the r344 default tree `39fcfe7b42035ffbdde2366926a2677cd4a89ea5` (sha256 `312147e4cef9b5b9bdc30ffbae6c290f54525694eaafa19f835b11d9713db239`), producing tree `2ac9015f7d4309d3638002fa6feba7a356413c34`; with the other optional artifact on top, the cumulative tree is `5f63cebbe95cc31aa056776bb92e9e163a17790c`.  Mechanical re-cut (`git am`): r344 amends block 15 only (`ggml/src/ggml-cuda/moe-expert-cache.cu`, the pinned-transport WARN), not this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r342 re-cut verification (superseded by r344)**: applies strictly at the r342 default tree `e4060a04c864a50c10c72431408cc898d7d275f3` (sha256 `0ce6a386a02365ff06d2fde1455a4ae5f8eb7d7779a10a2882d876898f849a52`), producing tree `c4122acb7b13576e486d8259e2e9c8346965ec04`; with the other optional artifact on top, the cumulative tree is `8b3b1301288f893ae2d1cbf84d02ecaa6a717c01`.  Mechanical re-cut (`git am`, no `-3` needed): r342 amends block 02 (`common/speculative.cpp` plus its new `tests/test-spec-type-exclusion.cpp`) and block 12 (`server-slot-sched.cpp` gains `<sys/mman.h>`, `server-task.cpp` loses it), none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  The consumer path `apply-all.sh --include-optional-all` reproduced that cumulative tree on a fresh `919794500` checkout, and `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r339 re-cut verification**: applies strictly at the r339 default tree `447ce143eb9d565fab78f81b780143d832af4baf` (sha256 `d292ba3bdbd6b6d83948b90140c384388c7d20653acaf6fdadab5f450c68519c`), producing tree `d06460d4c6172d6497f071e82172eec4bfa7d37d`; with the other optional artifact on top, the cumulative tree is `1c2af36770bde0b2b777eb0b71bc7ed90331c079`.  The re-cut is **mechanical**: r339 amends block 11 (`hc-mix.cu` only, the PR #344 band kernels), not this artifact's files, so its own diff is unchanged (the sha256 moved with the export's commit id).  `validate-set.sh` check 8 verifies the strict apply of all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r336 re-cut verification**: applies strictly at the r336 default tree `213764a7790eabdc896474d5cb7806ae15b488e8` (sha256 `805688a64e29d5494cdae88d5757f7e586c0d8625dea101bacd1624dcd6db147`), producing tree `13492d5b81efeacf0d5e83906880baa3f5a5f2bc`; with the other optional artifact on top, the cumulative tree is `2ccef3bc42c5167f87b03672d4ffb21cb9c4cc74`.  The re-cut is **mechanical**: r336 amends block 06 only (`mmb.cu` plus one comment in `ggml-cuda.cu`), neither of them this artifact's files, so its own diff is unchanged (the sha256 moved with the export's commit id and offsets).  `validate-set.sh` check 8 verifies the strict apply of all three trees on every run; the **r329-tree port verification below is still the numbers of record**.
* **r331 re-cut verification**: applies strictly at the r331 default tree `b0096a273caca67d0d17e07252eebb8b1cd8ad2c` (sha256 `8867b899f9f92e247139e9e78a46ab64c90801860a6dbfec374c5bd18f921687`), producing tree `071f0c22fc86ce8fe9e9451ff30faaedc1f6a3d7`; with the other optional artifact on top, the cumulative tree is `5d50ae4666aeba7688279d87135d2e77e92674be`.  The re-cut is **mechanical**: the patch applied cleanly (`git am`) on the r331 default tree, whose only change from r329 is block 00's restored `ggml-alloc.c`, block 10's GB10 constant and its `GGML_CUDA_DISABLE_ROPE_SET_ROWS` switch -- none of them this artifact's files -- so its own mechanisms and the numbers of record are unchanged.  `validate-set.sh` check 8 verifies the strict apply of all three trees on every run.
* **r329-tree port verification**: built and **measured** on the r329 tree (3 x R9700 gfx1201, ROCm
  10.1.0, `prompts/prose-rdna-boosts.txt`), not merely applied to it.
  * **Default path inert**: `-sm tensor` (3 GPUs) 4B Q8_0 greedy text is `1c5d32ac537d` for lanes unset, `1` and
    `2`, and the arm gate fires -- `LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0, draft = 0)`.
    `scripts/gate-prefill-logits.sh` on the optional-applied build reads **mean KLD 0.000000 / same-top-p
    100.000 %** against the r328-recorded base -- bit-inert on the default path.
  * **The reachable layout still engages and still wins**: `-sm layer -ts 1,1,1 -ot
    'blk\.0\.attn_q\.weight=ROCm0' -b 2048 -ub 2048`, 35B-A3B Q8_0: the `-lv 4` census shows `two-lane prefill
    enabled`, 3 x `pair decided: splits a=4 b=4, interleaved=1`, and the lane-2 compute buffers
    (`458.80 + 430.34 + 430.34` MiB device + `65.11` MiB host = **1384.6 MiB**).  Prefill
    (A B B A, `-n 0`): 5806.2 / 5793.9 t/s at one lane vs **6879.3 / 6883.2 t/s at two (+18.6 %)**; decode on the
    same shape is 72.7 / 73.4 / 72.7 t/s (nil); greedy text is `e2dde701fc04` for lanes unset / 1 / 2.
  * **The MTP target pairs again on the rebased tree** (the r246 feature, not re-run since r259): 27B
    `GSQ-RCO-IQ3_S-mtp` in the same layout, `--spec-draft-n-max 3` -- 3 pairs logged, both draft contexts refuse
    with `LLAMA_PREFILL_LANES ignored (n_gpu = 3, pipeline_parallel = 0, draft = 1)`, greedy text
    `a4a403d64c40` in **all four** arms (lanes 1/2 x `--spec-type none`/`draft-mtp`), prefill
    **1072.3 -> 1449.7 t/s (+35.2 %)** plain and **1040.0 -> 1364.1 t/s (+31.2 %)** on the MTP target; decode
    28.3/28.1 and 56.1/55.9 t/s (nil).
  * **The host-expert tier still runs and still loses at a wide ubatch** (the r259 abort fix holds): `-ncmoe 40`,
    `-b 2048 -ub 2048`, no assert, prefill 5230.4 -> 4465.7 t/s (**-14.6 %**), greedy text `49bf8044ec4b` for
    both lanes -- the artifact's own anchor for that arm.  `check-gpu-faults.sh` clean over every arm.
* **r329 re-cut verification**: applies strictly at the r329 default tree `63195b0ceb8dd9dea206a6a92b9bcd4a52e494e6` (sha256 `4d802e697473e8039d72c5c00440dba48075ef5b16a5cc2f0696ad837e3be341`), producing tree `7e9008463b1ce49e6fc6d32aabf24bdb312f6bac`; with the other optional artifact on top, the cumulative tree is `d8deaa13e8f5dd3ea0db0edb31050c33cdf1317c`.  The re-cut is **mechanical**: the patch applied cleanly on the r329 default tree (whose only change from r328 is block 10's width-probe TU, block 13's MTP-vocab TU and block 16's regenerated help docs -- none of them this artifact's files), so its own mechanisms are unchanged; the numbers of record are the r329-tree port verification above.  `validate-set.sh` check 8 verifies the strict apply of all three trees on every run.
* **r327 re-cut verification**: applies strictly at the r327 default tree `4792f98b82a4c6433bf11c629fdf01c21a143ce9` (sha256
  `1ffa60a9d2f8165ab1457bd85be50c51f7f8c190339a6c4b1d574e5f904b0945`), producing tree `410b6f1b16926d53a6a95f340bfb876cb2411a19`; with the other optional artifact on top, the cumulative tree is
  `8c1e74631c653862c3f2da5968042ec0cbba99f0`.  r327 amends **block 10** with one packed FA KV-group bitmap per graph evaluation
  (`GGML_CUDA_FA_KQ_BLOCKS_REUSE`, opt-in; `ggml-cuda.cu` gains two lines in the context destructor and
  at the head of `ggml_backend_cuda_graph_compute()`, PR #337).  This artifact's `ggml-cuda.cu` hunks sit
  in the backend's stream/lane plumbing, far from both, so nothing was reconciled and its own diff is
  byte-identical (the sha256 moved only with the export's commit id and offsets).  **Corrects a records
  gap:** the r324 entry below still names the r324 sha256, but r325 (block 09) and r326 (block 10) both
  re-cut this artifact -- the value quoted here is the current one.  `validate-set.sh` check 8 verifies
  all three trees on every run.
* **r324 re-cut verification**: applies strictly at the r324 default tree `d3babb65a68b83a2669933c902d5d981933d807e`
  (sha256 `c995ce750c286d0bce16c00921f871390747b8b9f718b8e272a8c61e7d1be551`), producing tree `9c1d723aab4cc50bbf4cfe6c99b842bae9201f63`; with the other optional artifact on top, the cumulative tree is
  `ab8eb507387c468712fb2142d7e2481a529a1516`.  r324 amends **block 10** (the MoE weighted reduction: no fma contraction, one knob, and the
  reduction fold ahead of the MUL-only down fold, PR #335), a file this artifact does not touch, so the
  artifact's diff is byte-identical to the r323 cut and every number recorded here stands.
* **r320 re-cut verification**: applies strictly at the r320 default tree
  `3334a4eb31124213805ae5a0a019f6aaf86d5d7b` (sha256
  `79cbf54eefcfeb699a863bed4054bef3f88a3a833a9eaed289680b27da76fa53`), producing tree
  `9d8474c19d8a3eca56334c0b99f7f61e5229fc12`; with the other optional artifact on top, the cumulative tree is
  `b61726912339d2f2e0a9abb646190f38690aec52`.  r320 amends **block 10** (`ggml/src/ggml-cuda/gdn-conv.cu`: the
  conv + q/k norm fold's overlap test follows a deferred conv-state gather, PR #316), a file this artifact owns
  no hunk of, so the re-cut moved only the exported commit id and the context offsets.
* **r319 re-cut verification**: applies strictly at the r319 default tree
  `9f53a85322eb8c0327c350fca89fe5af76a30806` (sha256
  `0eaff51b987b789bbef9960229505de6beefec82d6b95c8c379eb5868e25287a`), producing tree
  `cb07e2f41578163dbe5db56cc94da538f9a99110`.  r319 amends blocks 05 (the FA KV split) and 10 (the width probe),
  neither of which this artifact edits, so the re-cut moved only the exported commit id and the context offsets.
* **r318 re-cut verification**: applies strictly at the r318 default tree
  `d586e2a5fdfda303ea8363dfac20ca72784bf90a` (sha256
  `eb19211a0f3cbdf3bf6bb332d1d3fa389b64c17c8ff7ae1684e5cad6ad0cc911`), producing tree `ffb6c3239df6098f1f9315fc472273a062d3aa94`; with the other
  optional artifact on top, the cumulative tree is `820b327d2310a3d0f105d66de0e9767199bed7ff`.  r318 amends **block 10**
  (`ggml-cuda-try-fuse.cu`: the Q8_1 MUL fusion now counts readers through views, issue #331), a file this
  artifact owns no hunk of, so only the exported commit id moved and `validate-set.sh` check 8 verifies all
  three trees on every run.
* **r317 re-cut verification**: applies strictly at the r317 default tree
  `7d6b3f11062aa16922f6680af0c336973f847d1b`, producing tree `a7ce03f62b2a87d149bbdc82760005916a2c2934`; cumulative with the other
  optional artifact, numeric order, is `8ebf851d99874fc9992a212c775160ec16f4ee6f`.  Recorded here for the
  first time: r317 re-cut both artifacts (blocks 09/10 amendments) but wrote no bullet, so this file still
  claimed the r316 cut for one release.
* **r316 re-cut verification**: applies strictly at the r316 default tree
  `4119a6f862cff0f3b8840be51426e1a3028d8a71` (sha256
  `5ebab36f7f51f100eb7181498a2240325faeada169d1632ac87f61d4c565cd3a`), producing tree `d0cf0c723b1389dea7854f04ddd126a84d4f92cf`; with `0101` on top, the cumulative
  tree is `3447592e07c2a260e25b0c17e1e9ecb3bf5f6b02`.  Block 05's r316 amendment touches
  `fattn-rdna.cuh` / `fattn-common.cuh` only, which this artifact does not edit, so its **patch-id is unchanged**
  from the r315 cut and `validate-set.sh` check 8 verifies all three trees on every run.
* **r315 re-cut verification**: applies strictly at the r315 default tree
  `ff3c4d85732d9dda27a6a4914c728afb541d7ae6` and produces `6328a17fa5b71d5bb621ac74d6d906913935ae34`; cumulative with the other
  optional artifact, numeric order, is `7aec397b249a6f968e1cc425d63d89fbf8fad291`.  **The patch file moved**
  (sha256 `4ad739ce8b66f53ddee3fab6bbce990d816b5b48bd5e5fddb3af11a3406a47fc`): block 05's r315 amendment adds the 256-dim work gate (`ggml_cuda_fattn_band_wmma_h256_applies()` and the new rows of
  `fattn-rdna.cuh`), which is a file this artifact owns no hunk of, so the exported commit id and the hunk offsets
  moved while the produced diff is otherwise unchanged.  `validate-set.sh`
  check 8 verifies all three trees on every run.
* **r314 re-cut verification**: applies strictly at the r314 default tree
  `623e5a13f376e45f01fdecc6929c6dd5de0ab1d5` and produces `7704a44e6c7a629458fb5982a8a30b864b2dd117`; cumulative with the other
  optional artifact, numeric order, is `4ea663b57d2d150d7452657ec526c84241e4c1a9`.  The patch file moved
  (sha256 `4b61ecb6c5658dc067e1e3378e7c714fb66c34cb1c7f1ace4b55a6b43bcb3524`) **only in its `From <sha>` line** --
  block 05's r314 amendment touches the CUDA FA band, which this artifact owns no hunk of.  `validate-set.sh`
  check 8 verifies all three trees on every run.
* **r313 re-cut verification**: applies strictly at the r313 default tree
  `a9b88bab56ceb20a820044deda4e86504d47fe5b` and produces `b40c32caadee0ab2ed066f153d150bf901a08b33`; cumulative with the other
  optional artifact, numeric order, is `06013692f6ad9a28fc69ba7da7868b3b55420e80`.  **The patch file moved**
  (sha256 `4b61ecb6c5658dc067e1e3378e7c714fb66c34cb1c7f1ace4b55a6b43bcb3524`): block 11's r313 amendment (the Q8_0 hc mixer) rewrites `ggml/src/ggml-cuda/hc-mix.cu`,
  `common.cuh`, `ggml-cuda.cu`, `ggml/src/ggml-cpu/ops.cpp` and `src/models/qwen4exp.cpp`, which this artifact
  edits nothing of, so only the exported commit id and the hunk offsets moved -- the produced diff is otherwise
  unchanged.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r311 re-cut verification**: applies strictly at the r311 default tree
  `4a3db2c29d294987b95206fd663bc345114df455` and produces `48dbb770030653be116fb72f15fc64d58fa9b883`; cumulative with the other
  optional artifact, numeric order, is `5fa077f78da0c50265febcfaf6351484475b3973`.  **The patch file moved**
  (sha256 `00454bc676c9877792e9d5e951311f5db2906540ed89f518eb00f76b111ba92d`): block 05's r311 amendment rewrites `src/llama-context.cpp`, which this artifact edits
  nothing of, so only the exported commit id and the hunk offsets moved -- the produced diff is otherwise
  unchanged.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r307 re-cut verification**: applies strictly at the r307 default tree
  `f91aa81202fd32d060f9afb0f9c54824116758b9` and produces `7cafcacf5dea9677c318aba7713ba80802d144f9`; cumulative with the other
  optional artifact, numeric order, is `db5d26d45d5c8258b686151e3e42b63e139c301d`.  The patch file is
  **unchanged** (same sha256) -- r307's block-10 amendment touches the host-expert cache's take-over promises, which
  this artifact owns no hunk of, so only the produced tree moved with the default tree.  `validate-set.sh` check 8
  verifies all three trees on every run.
* **r306 re-cut verification**: applies strictly at the r306 default tree
  `f12b1a0c8ee6c24b77467f93550322e74762aef0` and produces `d1dbe7733171065b99f0f2adee189c1bfd0bc309`; cumulative with the other optional artifact, numeric order, is `445ace2475eb41c0ae543242efcee636513898ac`.
  The patch file moved (sha256 `00abb040e7debc4b6499f5262936c392019c93b2414d360cdb66fda7526eacb0`) because the exported commit id moved -- block 16's r306 amendment rewrites
  the commit this artifact is cut on top of -- and **only the `From <sha>` line changed**: the artifact owns no
  `tools/server/server-context.cpp` hunk, so the default tree's new admission code is the only difference in the
  produced tree.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r300 re-cut verification**: applies strictly at the r300 default tree
  `4baf1b67ae591e6661a1e3c5365457a747b688ab` and produces `49f3086f63ce7683705be315060b25a828152d61`; cumulative with the other optional artifact,
  numeric order, is `dc9a8815bd56be9a2e9cae48f7b19d26e7da922a`.  **The patch file is unchanged** (sha256 `d025be5f683d1594556803a0db0baaad7d8d17ca3c0496aac428cc55673a299f`,
  byte-identical to the r299 cut): the artifact does not touch `tools/server/server-context.cpp`, which is the
  only file block 12's r300 amendment changes, so nothing had to be reconciled -- only the produced tree moved
  with the base.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r299 re-cut verification**: applies strictly at the r299 default tree
  `3da54a4c10d7350eddc5c0d8eae384086a306b0f` and produces `1d389b8db32c28462aae2aca2f413daf31344022`; cumulative with
  the other optional artifact, numeric order, is `3eeb8f6f1dea6d7d550980afde5072049b5ac665`.  The default tree moved by
  block 12's r299 amendment (the `MOE_HOST_POOL_PIN_FAULT=late` latch waits for the first routed op), outside this
  artifact's mechanisms; `validate-set.sh` check 8 verifies all three trees on every run.
* **r295 re-cut verification**: applies strictly at the r295 default tree
  `bdfd4a182dc9bdf9a7b6a46a0f206622f37b5e34` and produces `87b399a4f61008d8af7443b444f244b0fde97a77`; cumulative with
  the other optional artifact, numeric order, is `af4c85cfedb7eea6a1eb3a71eb0448a40e03aea3`.  The default tree moved by
  blocks 12/14/15's r295 amendments (`parse_env` logging, the `--kv-window` help text, the pool exchange's census line),
  outside this artifact's mechanisms; `validate-set.sh` check 8 verifies all three trees on every run.
* **r293 re-cut verification**: applies strictly at the r293 default tree
  `92e77acfae0700b68439d69923fcb79d6fbe3a54` and produces `abe412bb86f98aa60224a8fdea41521b314d03f3`; cumulative with the other optional
  artifact, numeric order, is `795b1b0be4d56c75b6bb9707c46d207c0b5bb452`.  The patch file is unchanged (sha256 above) and the default tree
  moved by block 02's MTP h-rows default (r293, issue #318), outside this artifact's mechanisms.

* **r291 re-cut**: applies strictly at the r291 default tree `8423b2b5…` and produces
  `4b0b8da0154aac69ab61998865956deaec9ec9a8`; cumulative with `0101` is `310386642121ffd22eb74268a6446b31433aada0`.
  The default tree moved by block 12's r291 amendment (sequence parallelism), outside this artifact's mechanisms.
* applies at the r288 default tree `79d18f41…`, produces tree `1bfccf843b7da66ac2d464bd7b89c27d02a426be`
  (the r285 cut, sha256 `9ec906a277…`, produced `772c652409f2f5830edbce0fcee37077ed787889` on the r285 tree
  `3a6304107…`; the r284 tree `39ad82e0e…` produced `a9384ba7dce5b8d341de1ba63ad974af12ea2492`; the r259 cut
  produced `d1c3f0f9c3…` on `7995db44…`, the r254 cut `896446c422…` on `3f2cad2a…`, the r247 cut `17f51fb30c…`
  on `8f99d2dc6d…`).  The r288 tree is a different base, so a different produced tree is expected — the
  **diff** is unchanged and applied strictly.
* ```bash
  bash scripts/apply-all.sh --include-optional <clean-llama.cpp-at-release.json.base>
  # --include-optional steps through the artifacts in NUMERIC block order and asks [y/N] for each one
  # (0100 two-lane-prefill, then 0101 mixed-kv); --include-optional-all applies both, non-interactively
  ```
  (r288: it shares `optional/` with `mixed-kv` (`0101`), which moved out of the default delivery set in the
  same re-org.  The cumulative tree with both applied, in numeric order, is
  `d85868cdd23447b4d7fa0f9b9f5f7bb5fa7363cf`.)
* **r290 re-cut**: applies strictly at the r290 default tree
  `4f0770d7538b6e633ede9271bebdc5f83340a3c9` and produces
  `928c3c1c3442763f878d8b62025dd122fddc1238`; cumulative with `0101`, numeric order, is
  `00f2150fad1ad44c2d40c1456c708ea4c7103390`.  Only the base tree moved (block 07 gained the opt-in
  WMMA verify family); the artifact's own diff is unchanged.

## What it does

`LLAMA_PREFILL_LANES=2` (default `1`) keeps a **second backend scheduler** with its own compute buffers and, in
the decode loop, prepares two consecutive wide prefill ubatches on the two lanes and submits their splits
alternately (`ggml_backend_sched_graph_compute_async_pair`, `a0 b0 a1 b1 …`), so one device's work overlaps
another device's.  Lanes never move work: `-ts` / `-ot` placement decides which device owns what, the interleave
only changes submission order.

Since the r229 cut there are two halves, and **both are needed** (the numbers below are the proof):

1. **The sub-ubatch split.**  A batch of [`LLAMA_LANES_SPLIT_MIN` (default 1024; `0` disables), 2 x n_ubatch)
   tokens is split into two wide ubatches, the first taking `LLAMA_LANES_SPLIT_FRAC` of it (default 0.55,
   rounded up to 64 = the chunked-GDN chunk); the batch's output row lands in the second ubatch.  Before this
   cut only *whole* ubatches paired, so a prompt batch of fewer than two ubatches never paired at all.
2. **Output-aware pairing.**  The pair gate no longer requires both ubatches to be output-free: the outputs are
   extracted per lane after compute, on the scheduler of each lane (`extract_outputs`, with a sched-aware
   readback sync).  F1 layer-input readback stays single-lane.

Since the **r246 cut** an **MTP (nextn) target pairs too**: its prefill runs on 0-output nextn ubatches, and an
unmasked nextn row is per token, so the host offsets the lane split changes are the ubatch's own.  A **draft
context never arms** — `ctx_other` for Gemma4-assistant / EAGLE3 / DFlash / qwen4exp, and `ctx_type == MTP` for
an MTP draft, which the fork does *not* mark with `ctx_other` (it is the target's own model).

It degrades to the existing single-lane path when `< 2` non-CPU **backends**, pipeline parallelism, `no_alloc`,
or a draft context is present, and a workload that reads **F1** layer-input rows declines the pair **and also
declines the split** (see amendment 11).

A pair is admitted only when both lanes have the same split structure and the cache-hazard analysis finds no
cross-backend hazard (`GGML_SCHED_PAIR_HAZARD=0` keeps the name guard alone; `GGML_SCHED_PAIR_DEBUG` logs the
decision and, at `>= 3`, per-lane/per-backend GPU event spans).

## Which layouts it reaches — and which it cannot

This matters more than the knob does, so it is measured, not assumed (3× R9700, gfx1201):

| layout | result |
|---|---|
| `-sm tensor` (any device count) | **never engages.**  This tree's tensor split is ONE backend (a Meta device wrapping N GPUs) so the `n_gpu >= 2` gate sees 1: `sched_reserve: LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0, draft = 0)` |
| plain `-sm layer`, all layers on GPU | **never engages.**  The scheduler's own pipeline parallelism is enabled in exactly that configuration: `ignored (n_gpu = 3, pipeline_parallel = 1, draft = 0)` |
| `-sm layer` + any `--override-tensor` (`-ot`) | **engages** — a tensor override disables pipeline parallelism (`pair decided: splits a=4 b=4, interleaved=1`).  **This is the only reachable layout on ROCm.** |
| any layout, with a draft (`-md draft-dflash` / `-eagle3`, or `--spec-type draft-mtp`) | **never engages** — the draft context declines at the arm gate: `ignored (n_gpu = 3, pipeline_parallel = 0, draft = 1)`, and allocates no second buffer set |
| `-sm row` | **not usable at all on ROCm**: `llama_model_load: device ROCm0 does not support split buffers` (the r220 cut listed this as a reachable layout — it is an inference, not a measurement, and it is wrong for this backend) |
| `-sm none` | one GPU backend, i.e. the `-sm tensor` case |

| any layout, with a **host-expert tier** (`-ncmoe`, `-lme`, `--host-experts`) | **refused since the r369 port** — `ignored (... host_experts = 1)`.  The pair shares its backends between the lanes while the staging transport's slots are per BACKEND, and the pair **livelocks** on this tier (measured: lane 0 idle at 3 W, lanes 1/2 spinning at 100 %/~90 W, no progress).  This artifact's own record measures that arm as a **14.6 % loss**, so the single-lane path is what it always effectively was |

The **expert tier was orthogonal to reachability** before r369 (it only changed whether the pair was *worth it*);
since the port it is a hard refusal, for the reason in the table row above.

The submitter's **dense** reference layout (`-sm layer -ts 1,0`, no `-ot`) is refused here for the same reason,
so the +13.3 % PR #268 reported for it was not measured with this patch on this base.

## What it measures here

`-sm layer -ts 1/1/1 -ot 'blk\.0\.attn_q\.weight=ROCm0'` (a placement-neutral override: it only turns pipeline
parallelism off) and `-ub 2048 -b 2048` throughout; the `llama-bench` arms add `-t 16 -r 3`, the `llama-cli` arms
run the tool's default thread count (one arm per line, A B B A order).  Same binary, env-gated.

### The r285 cut — the port re-measured (3 x R9700 gfx1201, ROCm 10.1.0)

The same layout and the same arms as the r259 table below, on the r284/r285 delivery with the artifact applied
(`-sm layer -ts 1/1/1 -ot 'blk\.0\.attn_q\.weight=ROCm0'`, `-b 2048 -ub 2048`, the 5 298-token
`prompts/prose-rdna-boosts.txt`, `MOE_EXPERT_HEATMAP=load` pinned so a text comparison is meaningful -- §89):

| arm | 1 lane | 2 lanes | |
|---|---|---|---|
| 35B-A3B Q8_0, `-n 0` (prefill only) | 5426.6 t/s | **6815.6 t/s** | **+25.6 %**, 3 pairs logged (`GGML_SCHED_PAIR_DEBUG=1`) |
| the same, `-n 32` — decode | 71.2 t/s | 70.6 t/s | nil |
| the same, `-n 32` — greedy text | `e2dde701fc04` | `e2dde701fc04` | **identical** (the pair's per-lane output extraction is exact on this shape) |
| 35B-A3B `-ncmoe 40` (host-resident experts), `-n 8` | 5138.4 t/s | 4439.8 t/s | **-13.6 %** — the documented wide-ubatch loss on that tier; text identical (`49bf8044ec4b`) and the r259 abort stays fixed |

Default path (lanes unset) and the release anchors, this cut:

* `scripts/gate-prefill-logits.sh` (27B Q8_0, 40 chunks, 1 device): mean KLD **0.000697** / same-top-p
  **98.784 %** against the r279 base -- **the release's own reading**, so the artifact is inert on the default path.
* coherence, 4B Q8_0 `-p "The capital of France is" -n 20 --seed 42 --temp 0`: `1c5d32ac537d` on 1 GPU and on the
  3-GPU `-sm tensor` split, **identical to the release build** on the same box.
* `-sm tensor` with `LLAMA_PREFILL_LANES=2`: refused at the arm gate
  (`LLAMA_PREFILL_LANES ignored (n_gpu = 1, pipeline_parallel = 0, draft = 0)`), dense text unchanged.

### The r259 cut (PR #301) — re-measured here, and the host-expert tier

3× R9700 gfx1201, the same layout, same binary, env-gated, A B B A, `prompts/prose-rdna-boosts.txt` for the
`llama-cli` arms.  Two ubatch shapes: the delivery's `-b 2048 -ub 2048`, and the **tool defaults**
(`-ub 512 -b 2048`) — what a consumer who tunes nothing actually gets.

| arm | 1 lane | 2 lanes | |
|---|---|---|---|
| MoE 35B-A3B Q8_0, `llama-bench pp2048 -b 2048 -ub 2048` | 6544.9 / 6524.6 | **8070.2 / 8056.0** | **+23.4 %** |
| the same, default `-ub 512` (whole 512-token ubatches pair) | 4077.5 / 4051.0 | **6001.6 / 6033.7** | **+48.1 %** |
| the same, default `-ub 512`, `-ncmoe 40` (host-resident experts) | 3917.6 / 3930.8 | 4126.2 / 4157.1 | +5.6 % |
| the same, `-b 2048 -ub 2048`, `-ncmoe 40` | 6276.7 / 6276.7 | 5544.4 / 5530.4 | **−11.8 %** |
| 27B `GSQ-RCO-IQ3_S-mtp`, `--spec-type draft-mtp --spec-draft-n-max 3` — prefill | 954.4 / 963.5 | **1239.4 / 1238.1** | **+29.2 %** |
| the same — decode (`-n 32`) | 56.7 / 55.9 | 56.8 / 56.5 | nil |
| the same model, `--spec-type none` — prefill | 1061.6 / 1053.9 | **1295.2 / 1293.1** | **+22.3 %** |

**Text and the default path.**  Greedy `-n 32 --temp 0 --seed 42` text is `e2dde701fc04` (35B-A3B Q8_0) for
lanes=1 == lanes=2 == lanes unset, on `-sm tensor` and on the `-ot` layout, and with and without `-ncmoe 40` —
the delivery's own anchor for that model; the 27B arms read `a4a403d64c40` both ways and under both speculator
settings (the r246 anchor).  `scripts/gate-prefill-logits.sh` on the **default path** (lanes unset, 27B Q8_0)
reads mean KLD **0.000000 / same-top-p 100.000 %** against the base recorded from the r259 build, and
`check-gpu-faults.sh` is clean over every arm.

**The host-expert tier is where this knob loses.**  With `-ncmoe 40` the pair still engages (3 pairs logged,
`llama-cli -ub 2048 -b 2048`: prefill 5135.4 -> 4218.4 t/s, −17.9 %), but that tier is link-bound: the delivery's
H2D staging already overlaps the host→device traffic, a second compute-buffer set takes VRAM the expert-cache
arena wants, and the two lanes contend for the same link.  At the default `-ub 512` shape it still gains, because
there the second lane overlaps *layer* work instead of waiting on the link.  **On the host-expert tier leave
`LLAMA_PREFILL_LANES` unset** at a wide ubatch.

### The r246 cut (PR #296) — the MTP target pairs

3× R9700 gfx1201, `prompts/prose-rdna-boosts.txt` (5 298 tokens), `-ub 2048 -b 2048`, A B B A:

| arm | 1 lane | 2 lanes | |
|---|---|---|---|
| dense 27B `GSQ-RCO-IQ3_S-mtp`, `--spec-type draft-mtp --spec-draft-n-max 3` — prefill | 1047.7 / 1044.3 | **1361.1 / 1351.1** | **+29.9 %** |
| the same, decode (`-n 32`) | 63.4 | 63.2 | nil |
| the same model, `--spec-type none` — prefill | 1086.5 | **1436.3** | **+32.2 %** |
| MoE 35B-A3B Q8_0, `llama-bench pp2048` (`-t 16 -r 3`) | 6695.8 / 6751.0 | **8288.2 / 8322.9** | **+23.8 %** |
| MoE 35B-A3B Q8_0 + DFlash2 draft (`--spec-type draft-dflash`) — prefill | 5308 / 5346 / 5340 / 5300 | 5273 / 5287 / 5260 / 5280 | −0.9 % |
| `llama-batched-bench -npp 2048 -ntg 128 -npl 1,4,8` (PP/S) | 6214.6 / 8381.7 / 8822.9 | 7475.8 / 8258.2 / 8284.8 | +20.3 % / −1.5 % / **−6.1 %** |

`GGML_SCHED_PAIR_DEBUG=1`: `pair decided: splits a=4 b=4, interleaved=1`, once per batch — 5 298 tokens at
`-b 2048` is 3 batches, each split into 1152 + 896 (or 1152 + 50) and paired, so 3 pairs per prompt (a 15 k-token
prompt pairs several batches' worth).  The draft contexts log
`LLAMA_PREFILL_LANES ignored (n_gpu = 3, pipeline_parallel = 0, draft = 1)` (one WARN each — two draft
contexts) and allocate no lane-2 buffer set; the target logs `two-lane prefill enabled`.  An MTP draft has
`n_gpu = 2..3`, not 1: its backends are the model's whole device list, which is why the `ctx_type` predicate
is needed at all.

**The pair adds no approximation** (r246, 27B Q8_0, `-c 2048 --chunks 20 -ub 2048 -b 2048` + `-ot`,
`llama-perplexity --kl-divergence-base` recorded from the lanes-unset arm): lanes=2 reads mean KLD
**0.001394 / same-top-p 99.101 %**, and a plain `-ub 1152` arm (the same 1152 + 896 boundaries, run serially)
reads *exactly the same two numbers*.  So the divergence is the ubatch change, to every printed digit.

**The default path is bit-identical** to the r246 delivery build: `scripts/gate-prefill-logits.sh` mean KLD
**0.000697** / same-top-p **98.784 %** — the release's own reading — and `check-gpu-faults.sh` clean.

**Greedy text and MTP acceptance are unchanged**: `-n 32 --temp 0 --seed 42`, lanes=1 == lanes=2 ==
`--spec-type none`, sha `a4a403d64c40`; DFlash2 arm sha `e2dde701fc04` both ways; draft acceptance **0.62500
(20 / 32)** in both MTP arms (the submitter's arm shows 0.742 → 0.677 across the lanes — that arm leaves the
depth adaptive, so acceptance follows the speedup; a fixed `--spec-draft-n-max` is what makes the comparison
exact).

### The r243/r229 cuts (dated record): the gain needs both halves (MoE 35B-A3B Q8_0, pp2048)

| arm | t/s | vs lanes unset |
|---|---|---|
| lanes unset (one 2048-token ubatch) | 6621 | — |
| lanes=2, `LLAMA_LANES_SPLIT_MIN=0` (**pair alone**: one ubatch, nothing to pair) | 6703 | +1.2 % (nil) |
| lanes unset, `-ub 1152` (**split alone**: the same two ubatch boundaries, run serially) | 5608 | **−15.3 %** |
| lanes=2, split + pair | **8436** | **+27.4 %** |

So the r220 cut's measurement was right and irrelevant: pairing two *already wide* ubatches is worth nothing
here (this is why the r220 artifact measured +0.03 %), while pairing the two halves of **one** wide ubatch is
worth a quarter of the prefill — a single unsplit ubatch serializes the layer-split pipeline and the split
gives the second lane something to run.  The dense 27B Q6_K_XL shows the same 2×2: split alone −3.6 %
(1246 vs 1293), pair alone −1.2 % (1277, nil), split + pair **1823 (+41.0 %)**.

### Shape sweep (35B-A3B Q8_0)

| shape | 1 lane | 2 lanes | |
|---|---|---|---|
| pp512 (below the split minimum: nothing to split or pair) | 4179–4191 | 4149–4166 | neutral |
| pp2048 | 6621 | 8436 | **+27.4 %** |
| pp4096 | 6486 | 8245 | **+27.1 %** |
| pp8192, `-b 8192` (one decode call → 4 ubatches, paired 2×2) | 6240 | 9251 | **+48.3 %** |
| pp8192, `-b 2048` (4 decode calls, each one split batch) | 7530 | 7941 | +5.5 % |
| dense 27B pp2048 | 1293 | 1823 | **+41.0 %** |
| dense 27B pp8192 | 1488 | 1745 | +17.3 % |
| 27B `…-mtp.gguf` pp2048 (llama-cli prompt eval) | 844.7 | 1078.5 | +27.7 % |
| tg128 (MoE, dense, `…-mtp`) | 96.46 / 19.22 / 32.5 | 96.7 / 19.23 / 32.6 | nil |

The gain is largest when ubatches are actually paired inside one decode call; feeding a long prompt in `-b`
chunks (one split batch per call) leaves +5.5 %.

### Correctness

* **The default path is bit-identical** (r246): `scripts/gate-prefill-logits.sh` mean KLD **0.000697** /
  same-top-p **98.784 %** against the base recorded from the r246 delivery build — the release's own reading.
* **The pair is bit-exact** (r246): lanes=2 vs lanes unset at the same ubatching measures **exactly** the
  divergence of a plain `-ub 1152` vs `-ub 2048` change on the same build — mean KLD `0.001394`, same-top-p
  `99.101 %`, identical to every printed digit (the r229 point read `0.004088 / 97.898 %`; this is the §46
  "benign reordering" scale — a smaller ubatch, nothing more).  So the interleave adds no approximation; the
  split's divergence *is* the ubatch change, which is why greedy text staying identical is luck of the greedy
  path and not identity.
* Same-seed greedy text (`-n 32 --temp 0 --seed 42`, the 5 298-token `prompts/prose-rdna-boosts.txt`): lanes=2 ==
  lanes unset == `--spec-type none`, sha `a4a403d64c40`; a DFlash2 draft arm sha `e2dde701fc04` both ways.  MTP
  draft acceptance identical, `0.62500 (20 / 32)` in both arms (r229 record: `f61214fad249` on the MoE and
  `8c1c642fc7cd` on the dense 27B).
* Perplexity (`--ctx 2048 --chunks 8 --ub 2048`, every token an output row, i.e. **both** lanes carry outputs)
  restores the split's exact single-lane value — `llama-perplexity` output identical to the `-ub 1152` arm (r229).

### What it costs (all measured, all documented rather than tuned away)

* **The host-expert tier loses at a wide ubatch** (r259): `-ncmoe 40`, `llama-bench pp2048 -b 2048 -ub 2048`,
  6276.7 -> 5544.4 t/s (**−11.8 %**); `llama-cli` on the 5 256-token prompt, 5135.4 -> 4218.4 (**−17.9 %**).  It
  gains **+5.6 %** at the default `-ub 512`, so the shape decides.  Before the r259 cut it did not regress — it
  aborted (`What changed in the rebase`, item 13).

* **A second compute-buffer set per armed context**: ~1.4–1.5 GiB at `-ub 2048` (the r229 record: `ROCm0 441 +
  ROCm1 430 + ROCm2 430 + host 65` MiB, logged once as `compute buffer size = … (prefill lane 2)`).  A **draft**
  context no longer pays it (r246): the two MTP draft contexts on the 27B arm show only their own 134 MiB device /
  20 MiB host set, no `(prefill lane 2)` lines.
* **A second context in the same process runs ~8 % slower with the lanes armed** (r229 observation, *not*
  re-measured on this cut; pp8192: 7976 → 7347 on the 2nd context, while lanes unset is flat at 7513–7519 across
  three contexts).  Unexplained; the extra buffer set is the obvious suspect.
* **Multi-sequence prefill regresses**, and more so than at r243: `llama-batched-bench -npp 2048 -ntg 128 -npl
  1,4,8` → PP/S_TG 6214.6→7475.8 (+20.3 %), 8381.7→8258.2 (−1.5 %), **8822.9→8284.8 (−6.1 %)**; TG
  92.28→92.56, 242.2→239.1, 320.7→320.9 (r243 record: +28.8 / +0.9 / −3.5 %).  Server workloads with many
  concurrent long prompts therefore do not want this.
* **A draft workload's target still pays the arming cost** even though nothing pairs — `embeddings_layer_inp_dev`
  is a *decode-time* gate, not an arm-gate one.  Measured on this cut (35B-A3B Q8_0 + DFlash2, 5 298-token
  prompt): **−0.9 %** prefill, text identical (the r229 arm measured ~4 % prefill / ~2 % decode on a
  `llama-bench pp2048` arm; the difference is the shape, not a change in the mechanism).  An MTP workload no
  longer pays it at all (the draft is refused at the arm gate).

## What changed in the rebase

**r220 (the first artifact cut):**

1. **Dropped what the delivery now carries.**  PR #268 also diagnosed and fixed the shared per-device copy
   event (`cpy_tensor_async_nowait` + a per-copy event ring).  That part was **promoted separately** into block
   12 (`WORKLOG.md` 261105.006) because it is a real defect on its own — a queued HIP stream wait binds to the
   *latest* record, so a copy in flight re-binds consumers of copies still queued.  The artifact now uses
   `ggml_backend_sched_next_copy_event` / `cpy_tensor_async_nowait` from the delivery instead of defining them.
2. **CUDA portability.**  The PR spelled `hipEventDefault` / `hipEventElapsedTime` unconditionally;
   `ggml-cuda.cu` still builds under nvcc for CUDA and those have no hipify alias.  They now follow this file's
   own `#ifdef GGML_USE_HIP … #else cudaEventDefault … #endif` form.
3. **No cost on the default path.**  Events are timing-enabled only when `GGML_SCHED_PAIR_DEBUG>=3`.
4. **Docs corrected** to state the reachability table rather than implying `-sm layer -ts 1,1` works.

**r229 (the PR #279 re-cut):**

5. **The split is gated exactly like the pair** — `!cparams.embeddings_nextn && !cparams.lf_layer_inp_dev`, the
   pair gate's own conditions.  Without it a draft workload declined the pair but was still split, i.e. it paid
   a narrower ubatch and gained nothing: 35B-A3B Q8_0 + DFlash2 pp2048 measured **3981.8 → 3427.4 t/s
   (−13.9 %)** and its greedy text diverged.  With the gates agreeing, text is identical and the residual is the
   ~4 % arming cost above.
6. **Restored the delivery comment line** that the r220 cut dropped in `src/llama-context.h`
   (`// one scheduler each switch re-splits and re-allocates the graph. …`), which had left the MTP sched2 note
   reading "… every cycle; with / scheduler, so both stay allocated".  The artifact no longer deletes delivery
   text it does not mean to.

**r241 (the re-cut after the delivery moved under it):**

7. **`ggml_backend_sched_compute_splits` became a per-split function.**  The delivery now has
   `ggml_backend_sched_compute_split(sched, split_id, st)` plus the loop wrapper (which is exactly the structure
   the pair path needs), and since r229 the loop body also gained `sched_moe_register_tables()` before it and
   `ggml_backend_sched_prefetch_inputs()` between the two input-copy passes.  The artifact's cut carries both
   delivery calls, so a pool's host tables are still registered before a graph's first split and the prefetch
   still runs between the non-weight and the weight copies.
8. **`extract_layer_inputs()` lost its `host_also` argument** (the delivery's in-graph layer-input device store
   removed the post-graph host fill).  The artifact now adds only `sched_x` — the scheduler that owns the graph,
   which lane 1 must pass or its readback would query lane 0's scheduler — and keeps the r241 body and semantics
   verbatim.
9. **`graph_reserve()` takes `split_only` where it used to take `no_alloc`**, and `cparams.lf_layer_inp_dev`
   became `cparams.embeddings_layer_inp_dev`.  Both are renames at the two places the artifact arms the lanes.
10. **Removed: the `lf_wait_features_consumed()` ordering call** the r229 cut put at the end of
    `prepare_ubatch()`.  The function does not exist on r241 (the in-graph store made it unnecessary), and
    `prepare_ubatch` keeps r241's tail unchanged.

**Spot check at r241** (3 × R9700 gfx1201, 35B-A3B Q8_0, `-sm layer -ts 1,1,1` plus
`-ot 'blk\.0\.attn_q\.weight=ROCm0'`, `-b 2048 -ub 2048`, the 5 256-token
`prompts/prose-rdna-boosts.txt` with `-n 0`, two interleaved reps): prefill **5844.7 / 5827.9 → 6638.2 /
6614.8 t/s (+13.5 %)**.  Same-seed greedy text unchanged (`4ac3a523160f`, the delivery's own anchor for that
model), and under `-sm tensor` the lanes are refused and the text is the dense anchor (`1c5d32ac537d`).  The
number is smaller than the r229 table's +27 % because that table is `llama-bench pp2048`, one wide ubatch split
into two halves; this arm is `llama-cli` feeding a 5 256-token prompt in `-b` chunks, where only the chunks that
fit the split window pair.

**r243 (the re-cut after the r241 delivery moved again):** no behavioural delta and no conflict resolution —
the artifact was re-cut only so its index lines name the r243 blobs instead of r241's (a stale blob line is what
lets a 3-way apply land a hunk in the wrong place; `WORKLOG.md` 261109.008).  Its numbers are the `+13.5 %` spot
check above, carried over.  `release.json` was not touched.

**r244-r245 (PR #284 reviewed, NOT taken):** PR #284 re-cut the artifact onto r234 and, on top of the re-base,
removed `!cparams.embeddings_nextn` from the split and pair gates so an **MTP target** pairs.  It was turned
down with `CHANGES_REQUESTED` (`WORKLOG.md` 261109.011): the delta measures (+14.7 %, 946.7/942.0 →
1085.1/1080.6 t/s on 3 × R9700) but it would also *pair the MTP draft*, which the artifact's arm gate let arm
— an MTP draft is the target's own model with `ctx_type = MTP`, so `ctx_other` is nullptr, `n_gpu` is the
model's whole device list, and the draft was allocating a second scheduler it could never pair on.  The
requested change was the explicit predicate `ctx_other != nullptr || ctx_type == MTP` plus a re-cut on the
current tree with the gate delta and the layout/hardware stated.

**r246 (this re-cut, PR #296):**

11. **The MTP target pairs; the MTP draft is refused at the arm gate** (`ctx_type == MTP` added).  The split and
    pair gates keep only `!embeddings_layer_inp_dev`, so they agree with each other and with what actually
    pairs.  Measured on this cut: target prefill **+29.9 %** (27B MTP, 5 298-token prompt), MoE **+23.8 %**,
    dense **+32.2 %**; MTP greedy text and draft acceptance unchanged; the pair measures the plain-`-ub 1152`
    KLD exactly (all in *What it measures here*).
12. **The draft predicate is one boolean and it is logged.**  The submitted cut left the two conditions inline
    and the WARN at `ignored (n_gpu = %d, pipeline_parallel = %d)`, so a refused MTP draft printed a line whose
    two conditions both pass — the reader cannot see that the new predicate fired.  It reports
    `… draft = %d` again, and the `ctx_other`-vs-`ctx_type` explanation sits above the gate where the next
    reader will look (`ggml_backend_sched_graph_compute_async_pair`'s WARN is the other half of that story).

**r259 (this re-cut, PR #301 plus this repo's amendment):**

13. **FIXED: the pair aborted on a host-expert layout.**  `LLAMA_PREFILL_LANES=2 -sm layer -ot … -ncmoe 40`
    died at the first prefill on `GGML_ASSERT(sched_a->is_alloc && sched_b->is_alloc)` in
    `ggml_backend_sched_graph_compute_async_pair`.  `prepare_ubatch()` grew the `lane` parameter in this
    artifact, but the delivery's two lane-0 placement blocks inside it — the ring narrow→wide drop
    (`ring_carved && ring_arm && n_tokens_prev < 64`) and the `drop_now` wide→narrow drop/re-reserve — still
    dropped and re-reserved `sched` (lane 0) unconditionally, so the lane-1 prepare ran them *again* after lane
    0's ubatch had been prepared and handed the pair an unallocated lane 0.  Both blocks are now `lane == 0`
    (they manipulate lane 0's scheduler and the context's lane-0 size expectations; a lane-1 ubatch must not
    re-place lane 0).  Pre-existing since the r246 cut — every arm that cut measured keeps the experts
    device-resident, so the ring was never carved and the blocks never fired.  After the guard the arm runs and
    its text is the single-lane text (`e2dde701fc04`).
14. **Base drift r254 → r259**, no code delta: r258 (block 09, the identity expert table served IN PLACE:
    `ggml-backend-impl.h` gains `moe_cache_in_place`, `ggml-backend.cpp`'s staging decision and take-over call
    change, `ggml_backend_buffer_get_usage()` answers for a buffer-less table) and r259 (block 01, the `q8ce`
    wire) moved the files the artifact touches.  The artifact's +/− line set is identical across the r247, r254
    and r259 cuts (checked per file on sorted line multisets), so the arm gate, the MTP-target pairing, the draft
    refusal and the scheduler refactor are unchanged.  Note the r254 cut's own delta note attributed the
    `GGML_SCHED_STAGE_MODE` retirement to r254: it was **r250** (block 09, `TODO` #284) — the resolution was right,
    the attribution was not.
15. **Dropped a stray blank line** the artifact had been adding to `ggml/include/ggml-cuda.h` since the r220 cut
    (its only change to that file).  That file is untouched now.

**r285 (this re-cut): a PORT, because the delivery rewrote both files the artifact owns between r259 and r284.**

16. **What had to be reconciled, file by file.**  The artifact's own mechanisms (arm gate, split, pair, hazard
    analysis, per-lane extraction) needed no change; the code around them did.
    * `ggml/src/ggml-backend.cpp` — D10 ("do not upload an input that is not read") added
      `hv_tensor_requires_copy`, and it now gates **both** copy passes (non-weight and host-weight).  The
      artifact's two call sites keep that guard and pass the input **slot index** instead of the tensor, because
      the eager-copy events are per slot (`split->input_events[input_id]`, `ggml_backend_sched_copy_input(sched,
      split, input_id, st)`).  `hv_tensor_split_ids` is a new field beside the delivery's three, allocated and
      reset with them.  The pair/hazard machinery, `async_inputs`/`eager_copies` and the `eager_producer`
      allocation placement are the artifact's and are unchanged.
    * `src/llama-context.cpp` — the r259 file had the decode body inline; the artifact moved it into
      `extract_outputs()`.  The delivery's body had since grown the nextn/h_nextn **device** path
      (`write_h_nextn`, `#31927`), the `-ot` state-row ordering block, `extract_layer_inputs(..., sched_x)` and
      the `has_next`-gated readback sync.  The lambda is therefore built from **the delivery's current body**
      (`ubatch` -> `ub`, `sched.get()` -> `sched_x`) rather than from the artifact's r259 copy, so none of that
      delivery code is reverted; `ubatch_fail`, `count_outputs`, the lane branch, `n_ubatch_eff` and the pair
      entry points are the artifact's own and are unchanged.
    * `ggml/src/ggml-backend-impl.h` / `ggml/src/ggml-cuda/ggml-cuda.cu` — the one new iface member
      (`event_elapsed_time`, debug-only) appends **after** block 16's `gtt_store_*` members, because this iface
      is positionally initialized.
17. The three r259 items are carried unchanged: the host-expert-layout abort fix (both lane-0-only placement
    blocks), the MTP-target-pairs / draft-refused arm gate, and no stray blank line in
    `ggml/include/ggml-cuda.h`.
18. `optional/host-pool-fill-src` is **gone** (archived to `archive/work/`, r285): it was the other artifact in
    this directory, and its premise was promoted by r262's `MOE_HOST_POOL_TRANSPORT`.  `apply-all.sh
    --include-optional` therefore offers exactly one artifact now, and the combined-tree hash the r259 note
    records no longer applies.

**r288: a mechanical re-cut, and the `0100` number.**  No code delta.  The r285 patch applied strictly on the
r288 default tree (no conflict, no reconciliation), so the diff is byte-identical and item 16's reconciliation
still describes the artifact.  The re-cut exists because the r288 block-number re-org renamed the artifact to
optional block `0100` and re-based every optional artifact onto the new default tree; see `WORKLOG.md` (r288)
and `optional/README.md`.

## Known open points

* On the pair-failure path (`graph_compute_pair != SUCCESS`) lane 1's ubatch has already been applied to the
  memory module by `prepare_ubatch`, and the loop then processes it again in the single-lane path.  Error path
  only; a double `mctx->apply()` for the same positions is not obviously idempotent for every memory type.
* `include/llama.h` gains `llama_n_prefill_lanes()`, a public accessor for an internal feature.
* The ~8 % second-context penalty and the −6.1 % 8-sequence regression above are observed, not explained.
* The host-expert wide-ubatch loss (−11.8 % / −17.9 %, r259) is measured and explained as link-bound contention,
  not tuned away; it is documented rather than gated, because the layout that gains is the one the artifact is
  for.
* The delivery's own draft detection (`graph_reserve()`'s narrow-2 route, `process_ubatch()`'s `is_draft`, the
  narrow-2 floor re-pin in `sched_reserve()`) still tests `ctx_other` alone, so none of it fires for a
  qwen3-family MTP draft.  That is pre-existing delivery behaviour (`WORKLOG.md` 261109.011 item 3), not this
  artifact's; the artifact only had to stop letting the draft arm.

## Env knobs (see `ENVIRONMENT.md` for the class of each)

| variable | default | what |
|---|---|---|
| `LLAMA_PREFILL_LANES` | `1` | `2` enables the pair.  Anything else (or a layout the table above refuses, or a draft context) logs one WARN and stays single-lane |
| `LLAMA_LANES_SPLIT_MIN` | `1024` | minimum batch size (tokens) for the sub-ubatch split; `0` = whole-ubatch pairing only (the r220 behaviour) |
| `LLAMA_LANES_SPLIT_FRAC` | `0.55` | fraction the first sub-ubatch takes (clamped to [0.1, 0.9], rounded up to 64) |
| `GGML_SCHED_PAIR_DEBUG` | `0` | `1`: log each pair decision; `>= 2`: also run the hazard self-test; `>= 3`: per-lane/per-backend GPU event spans (creates timing events) |
| `GGML_SCHED_PAIR_HAZARD` | on | `0` keeps only the split-name guard, skipping the cache-hazard analysis (diagnostic) |

## Why this is not a block

Promotion needs a **validated win on one of the delivery's own configurations**, and this cut now has several
(gfx1201, +23.4 % MoE `pp2048`, +48.1 % at the default ubatch shape, +22.3 % dense prefill, **+29.2 % on an MTP
target**, all in the `-ot` layout) — but it still carries open regressions (a second context, −6.1 % at 8
sequences, −0.9 % on a layer-input draft's target, **−11.8 % on the host-expert tier at a wide ubatch**) and it
is **inert in both splits the delivery actually validates** (`-sm tensor` is one backend; plain `-sm layer` has
pipeline parallelism on).  Reaching it costs a second compute-buffer set and a hand-written `-ot` override, and
it reaches a *draft* workload only through its target.  So it stays an artifact: a consumer with an `-ot` layout
and device-resident experts gets a large prefill win for a knob and a documented set of costs; a consumer on the
host-expert tier should not set it.

It becomes a promotion candidate when it reaches `-sm tensor` (pairing inside the Meta device, or a tensor-split
hazard analysis), when plain `-sm layer` can use it (the pair needs to coexist with, or replace, the scheduler's
pipeline parallelism), or when the regressions above are gone.
