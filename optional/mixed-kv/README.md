# `mixed-kv` — optional block `0101`: a q8_0 K cache with a smaller V cache, read natively

**Optional artifact, NOT part of the delivery.**  It was promoted into the default set as **block 17** in
r287 (`v18-4f9dfa0e3-r287`) and moved OUT again by the **r288 block-number re-org**, which is what gives it
this `0101` optional number.  Nothing here is in `patches/`, `release.json.tree`, `release.json.tip`,
`rdna-boosts-all.patch` or the fork's `rdna-boosts` branch, and a default `scripts/apply-all.sh` checkout
never contains it.  Only `scripts/apply-all.sh --include-optional` (asks `[y/N]`) or
`--include-optional-all` applies it, on top of the verified default tree.

* `0101-mixed-kv.patch` — the artifact.  sha256
  `ba75c00cdd3c0e0e2efd49c52c3e384f2783686f797f4eddce6aa72544eccf4c` (the current cut; every earlier cut is
  recorded below)
* **r372 re-cut verification (current)**: applies strictly at the r372 default tree
  `7c1728b37d8b914724e62156559eddf0301c062c` (18 official blocks `0000`-`0017` + temporary `0300`, default tip
  `5e844cbc6`), sha256
  `ba75c00cdd3c0e0e2efd49c52c3e384f2783686f797f4eddce6aa72544eccf4c` -- **unchanged**, unlike a re-cut: r372
  amends block 05, which owns `ggml/src/ggml-cuda/fattn.cu` (one of this artifact's ten files), but the only
  hunk in that file is in `ggml_cuda_flash_attn_ext_get_alloc_size`, while r372's change is in
  `switch_ncols2`'s RDNA4 tensor-split rule, so the patch's own diff and export are untouched and it needed no
  rebase; producing tree `8bacc3b7c30c61cd3a5fae5dec4428a9bfd68a97` alone, and with the other optional artifact
  on top, the cumulative tree is `d74527df00cb751fdf695ef3d7be38346c485f87`.  `scripts/gate-optional-build.sh`
  (gate 47) **PASS** on the cumulative tree (`all` target, 212 s).  `validate-set.sh` check 8 re-verifies all
  three trees on every run; the r353 port verification below is still the numbers of record (the artifact's
  own code did not change).
* **r367 re-cut verification (superseded by r372)**: applies strictly at the r367 default tree `2ff670ba83e5160abf16d20355aeded52cb744f0`
  (18 official blocks `0000`-`0017` + temporary `0300`, default tip `83b254bea`), sha256
  `0783b165c67d497c1d8db7160b3557cec0f3b41c2fa4c27338804f0dbf0f70d3` -- **this cut also repairs the artifact's own
  content** (GitHub issue #371): r360's rebase had inserted `ggml_cuda_fattn_tile_type_V` between
  `ggml_cuda_fattn_kv_pair_native_host`'s `return` and its closing brace, leaving the outer function unclosed
  (`fattn-rdna.cuh` brace balance 97 open / 96 close) -- the patch applied, the tree re-verified, and the tree did
  not compile for seven releases.  The one missing `}` is restored (balance 97/97), so the produced tree goes
  `791c489d` -> `9357ca2924e80cbb0bf4593e3af7c9b1248dc553` alone, and with the other optional artifact on top the
  cumulative tree is `c5ad54d972c59c0717c8476647fd64d6621977d8`.  Verified on the optional-applied build:
  `scripts/gate-optional-build.sh` **PASS** (the `all` target now compiles; the pre-fix artifact at the same tree
  fails with the reporter's `function definition is not allowed here` at `fattn-rdna.cuh:369`), and
  `scripts/gate-mixed-kv.sh` **14/14 PASS** on gfx1100 (`FLASH_ATTN_EXT` 7493/7493 in the default, `KV_MIXED=0`
  and `KV_NATIVE=0` arms; width purity q8_0/q8_0 and q8_0/q4_0,q5_0,q5_1 `maxdiff 0`; four text arms; both
  refusals; journal clean).  `validate-set.sh` check 8 still re-derives all three trees on every run -- but a tree
  check is not a build, which is what gate 47 (`gate-optional-build.sh`) now covers.
* **r366 re-cut verification (superseded by r367)**: applies strictly at the r366 default tree `82afebd108d63b6220884d41080009561f0821e2` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `489810652`), sha256 `b9d9be9e9f2597ccd0ba6783f2faaf7e7e28219ac182852be5720d552ff960ff` -- the sha moved because the exported commit id did, not the artifact's own diff: r366 amends block 10 (`ggml/src/ggml-cuda/ggml-cuda-slab.cu`, the slab's wide band is armed by the prefill pass rather than by the carved ring hole), which no file here touches, so the cut was re-exported against the new tree and applied with nothing to reconcile; producing tree `791c489d6ddb7ab84177597cf177cce1e6c7820b` alone, and with the other optional artifact on top, the cumulative tree is `f54c5d78e853f312b34f0ab6c5bcd4004e500667`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the r353 port verification below is still the numbers of record (the artifact's own code did not change).
* **r365 re-cut verification**: applies strictly at the r365 default tree `21efb08247edc142c649d2dcda771abb73fee810` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `7ad7f533c`), sha256 `b651651b9ddda31f6518debe4a4d0dac480d741ea881c60830d4dd336f0d8b59` -- the sha moved because the exported commit id did, not the artifact's own diff: r365 amends block 16 (`tools/server/server-{context,queue,slot-sched}.*`, the server's KV admission), which this artifact's server hunks now sit beside, so the cut was re-exported against the new tree and applied with nothing to reconcile; producing tree `f31b444f11401efd821bd913428ec098300758b3` alone, and with the other optional artifact on top, the cumulative tree is `43ee504d461f63e663db7426e28273756bc00ee1`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the r353 port verification below is still the numbers of record (the artifact's own code did not change).
* **r360 re-cut verification (current)**: applies strictly at the r360 default tree `4674dd9f2e83894895d71b7e987a06d80fa70813` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `3be8b9bd1` (the deterministic re-cut of the same tree); sha256 `f861f9a36198a3c8bf46bdd2a15fac5b6a9ac0987833908e5a7dcb0d2f84594f` -- the sha moved because the exported commit id did, not the diff: r360 amends blocks 09 and 10, and block 10 owns `ggml/src/ggml-cuda/fattn-rdna.cuh`, where this artifact's `ggml_cuda_fattn_tile_type_V` hunk lands directly after r360's new `ggml_cuda_fattn_kv_pair_native_host` -- so the cut was REBASED (one textual conflict, resolved by keeping both functions, the artifact's unchanged)), producing tree `f455c3121bd4e1acd0a518c90eb947136cc5cd4c`; with the other optional artifact on top, the cumulative tree is `8bd172df5d98e698a6868d645afa626cdfc6c19a`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r358 re-cut verification (superseded by r360)**: applies strictly at the r358 default tree `84d50a416183a159c504fab508a84c0d5ddb859f` (18 official blocks `0000`-`0017` + temporary `0300`, default tip `dcee0fc34`; sha256 `10f4c66835586dd1449d527c3ae48f1848a590c4bf98276e4346e4b59788977d` -- the sha moved because the exported commit id did, not the diff: r358 amends block 05, which owns `ggml/src/ggml-cuda/{fattn.cu,fattn-common.cuh,fattn-mma-f16.cuh,fattn-rdna.cuh}` -- three of this artifact's ten files -- so the cut was re-verified rather than assumed, and it applied with nothing to reconcile: the `fattn-rdna.cuh` hunk lands on the band predicate's `kv_quant` term, while r358 adds an early `src[5] != nullptr` return at the top of that function, and the `fattn.cu` hunk is in `get_alloc_size`), producing tree `cca36b57f6e2f9bf699ee39127a9aa4c742eebd9`; with the other optional artifact on top, the cumulative tree is `b00ce41dd2d408a969cfc7ad1af42f9ce7eb1daf`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r356 re-cut verification (superseded by r358)**: applies strictly at the r356 default tree `2dbfffd60a5f7490a90498967d83ccbe8cb569e8` (18 official blocks `0000`-`0017` + temporary `0300`; sha256 `b7b496dc6b593ebdc0cf723e95a995c5285144daa23e80c3edafe3ed42f22062` -- the sha moved because the exported commit id did, not the diff: r356 adds block 17, which owns `ggml/src/ggml-cuda/{common.cuh,convert.cu,dequantize.cuh,getrows.cu,ggml-cuda.cu,ggml-cuda-try-fuse.cu,mmvq.cu,tq.cu,tq.cuh}`, none of this artifact's files), producing tree `669b40949e30cec77bf1c3613c9c4ef3d4091523`; with the other optional artifact on top, the cumulative tree is `2f7f97dc919539992107a4926c8771974e9494c2`.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r355 re-cut verification (superseded by r356)**: applies strictly at the r355 default tree `4244d077effe09a44a4763dd0cabbbf8f37bdf05` (sha256 `1f22b372d60b5c52776a4ccd83fe8100ba9564bbfa67125cbc8eb9c795172c8a` -- **unchanged**, unlike a re-cut: r355 amends block 10 only, inside `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` (the GDN output-norm fold's matcher takes the skip's decision and its reader counts are one walk), none of this artifact's files, so the patch's own diff and export are untouched and it needed no rebase), producing tree `1018f175b76cf4eca94d2654aed62ed72046149d`; with the other optional artifact on top, the cumulative tree is `cdeae8f29c7f419332c019d0d3d252aeee8aaa18`.  Only the trees moved, because a produced tree is `default tree + this patch` and the default tree moved.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r353 re-cut verification (superseded by r355)**: applies strictly at the r353 default tree `27db9bfd4ef7f41c7df60f704410de28863beac3` (sha256 `1f22b372d60b5c52776a4ccd83fe8100ba9564bbfa67125cbc8eb9c795172c8a` -- **unchanged**, unlike a re-cut: r353 amends block 10 only, inside `ggml/src/ggml-cuda/ggml-cuda-try-fuse.cu` / `common.cuh` / `ggml-cuda.cu` (the GDN output-norm fold's SILU widening), none of this artifact's files, so the patch's own diff and export are untouched and it needed no rebase), producing tree `c5cb865a4237729c1416670ddc5d7842dd3ec357`; with the other optional artifact on top, the cumulative tree is `790e050d12ea05858ced2ebe50b46e38b3764f0c`.  Only the trees moved, because a produced tree is `default tree + this patch` and the default tree moved.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r351 re-cut verification (superseded by r353)**: applies strictly at the r351 default tree `8851b0d02b74060f462088617e8a7ff328c8c5ed` (sha256 `d677b9253e07e8f02ea707d6c1c9eb06ec094d696fa37253fb9565be1334c824`), producing tree `e80a98cb4553c1accfe299c7a0ceee666537000e`; with the other optional artifact on top, the cumulative tree is `91f52b417643df489dcba548894b5a394ad1dc8f`.  Mechanical re-cut (`git am`, no `-3` needed): r351 amends block 17 only (`ggml/src/ggml-cuda/ggml-cuda.cu` -- two wiring fixes for issues #353/#354 -- plus the block-0300 offset shift), none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r350 re-cut verification (superseded by r351)**: applies strictly at the r350 default tree `83bdb702510693916340031ba2cb9e42eb5dfd04` (sha256 `6ccfd033fec99c1493a1a7c622b231d755a521fe8dec25f3c3f4cc8266dacb10`), producing tree `df1287b64c46e8c155a67d12e24a2c136e2cb749`; with the other optional artifact on top, the cumulative tree is `8dff92e0062165631b6942b6ae7de63c67fe2f6b`.  Mechanical re-cut (`git am`, no `-3` needed): r350 amends block 17 only (`common/speculative.{h,cpp}`, `common/common.{h,cpp}`, `tools/server/server-context.cpp` -- the speculator on the two-arm server, `TODO` #364), none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r345 re-cut verification (current)**: applies strictly at the r345 default tree `0d2f70e6c44eb246a9c71f29e015d6ebe98f214d` (sha256 `b91e2bf00ff1ea4f95a6dc8025d95f6a70d991aee2df5055becc1d00fa9365c5`), producing tree `13611034a625b0958addbbfb6b7fb346f188a617`; with the other optional artifact on top, the cumulative tree is `3cf56a5af604cc7ee67336b70248e439a17300d4`.  Mechanical re-cut (`git am`, no `-3` needed): r345 amends block 09 (`ggml/src/ggml-cuda/moe-expert-cache.cu`, the MoE device gather's used-expert bitmap, issue #350) plus the wording of block 12's `device gather:` census line, none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r344 re-cut verification (superseded by r345)**: applies strictly at the r344 default tree `39fcfe7b42035ffbdde2366926a2677cd4a89ea5` (sha256 `b1be5c5971c3ee51d02e2851b337bbf631b17e8a2735d01af09e6fcc156fd9e7`), producing tree `10ebc31c626f31f42b014987f501309eb0f9e829`; with the other optional artifact on top, the cumulative tree is `5f63cebbe95cc31aa056776bb92e9e163a17790c`.  Mechanical re-cut (`git am`): r344 amends block 15 only (`ggml/src/ggml-cuda/moe-expert-cache.cu`, the pinned-transport WARN), not this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r342 re-cut verification (superseded by r344)**: applies strictly at the r342 default tree `e4060a04c864a50c10c72431408cc898d7d275f3` (sha256 `1633424aa8022453858a77ff4427e45601823a4fbccbef06f992a2d1816bb2f5`), producing tree `abbac6fd66db0222e10cc9f1b76c5bc04d1e251b`; with the other optional artifact on top, the cumulative tree is `8b3b1301288f893ae2d1cbf84d02ecaa6a717c01`.  Mechanical re-cut (`git am`, no `-3` needed): r342 amends block 02 (`common/speculative.cpp` plus its new `tests/test-spec-type-exclusion.cpp`) and block 12 (`server-slot-sched.cpp` gains `<sys/mman.h>`, `server-task.cpp` loses it), none of them this artifact's files, so its own diff is unchanged and the sha256 moved only with the exported commit id.  The consumer path `apply-all.sh --include-optional-all` reproduced that cumulative tree on a fresh `919794500` checkout, and `validate-set.sh` check 8 re-verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r339 re-cut verification**: applies strictly at the r339 default tree `447ce143eb9d565fab78f81b780143d832af4baf` (sha256 `ba102b0d58ff446a5e3c4b915f26f6a16e5a3bd21bfde4826d0b0f937b31a962`), producing tree `868cedb45ed43af4a53f85409f9020b321f0fd50`; with the other optional artifact on top, the cumulative tree is `1c2af36770bde0b2b777eb0b71bc7ed90331c079`.  Mechanical re-cut: r339 amends block 11 (`hc-mix.cu` only, the PR #344 band kernels), not this artifact's files, so its own diff is unchanged.  `validate-set.sh` check 8 verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the artifact's own code did not change).
* **r336 re-cut verification**: applies strictly at the r336 default tree `213764a7790eabdc896474d5cb7806ae15b488e8` (sha256 `cbb2e258de2005aa1cf1ef4ef6a8a2420e1e233c24d216c528414fdbcc139b76`), producing tree `4c49d68b73d1be69fc72712a5a96ce94a65ff4e5`; with the other optional artifact on top, the cumulative tree is `2ccef3bc42c5167f87b03672d4ffb21cb9c4cc74`.  Mechanical re-cut: r336 amends block 06 only (`mmb.cu` plus one comment in `ggml-cuda.cu`), neither of them this artifact's files, so its own diff is unchanged and the hunk r329 re-anchored for the ext batch API still sits on the same context.  `validate-set.sh` check 8 verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the gate was not re-run on r336; the artifact's own code did not change).
* **r331 re-cut verification**: applies strictly at the r331 default tree `b0096a273caca67d0d17e07252eebb8b1cd8ad2c` (sha256 `3851a677325afd7f0b59ddbcb143407014f12b9a8e23e602d51b524056a9b3c2`), producing tree `b17097cb354d0da855b9f69424755ff4f8abbfb9`; with the other optional artifact on top, the cumulative tree is `5d50ae4666aeba7688279d87135d2e77e92674be`.  Mechanical re-cut (`git am`, no `-3` needed): the r331 default tree differs from r329 only in block 00's restored `ggml-alloc.c` and block 10's GB10 constant + `GGML_CUDA_DISABLE_ROPE_SET_ROWS` switch, none of them this artifact's files -- so the hunk r329 re-anchored for the ext batch API still sits on the same context.  `validate-set.sh` check 8 verifies all three trees on every run; the **r329-tree port verification below is still the numbers of record** (the gate was not re-run on r331; the artifact's own code did not change).
* **r329-tree port verification**: the artifact is not merely *applied* on r329, it is
  **exercised** there.  Built on one optional-applied tree (r329 chain tip `f8df8a484`, `build-opt`), its own gate
  `scripts/gate-mixed-kv.sh` reads **14/14 PASS**: `test-backend-ops -o FLASH_ATTN_EXT` **7453/7453** in the
  default, `GGML_CUDA_FA_KV_MIXED=0` and `GGML_CUDA_FA_KV_NATIVE=0` arms (the same case count as r328), width
  purity W=1..8 with `maxdiff 0` for q8_0/q8_0 and all three pairs at the 3-device `-sm tensor` split, a 32-token
  greedy decode per pair, **and both refusal arms** -- Ling-3.0-tiny (heads not 256-wide) and
  Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS (QSA), each still refused with the "models require the same K and V cache
  types" text.  The refusal arms are new coverage: no earlier run on this box exercised them (r328's "12/12"
  predates the r330 gate fix that defaults the two models, FAILs on a missing one and counts a skip).
  `check-gpu-faults.sh` clean.  This is also the first run of the artifact's `tests/test-logits-width-probe.cpp`
  hunk since r329 migrated that TU to the ext batch API -- the re-anchored hunk reproduces the width-purity verdict.
* **r329 re-cut verification**: applies strictly at the r329 default tree `63195b0ceb8dd9dea206a6a92b9bcd4a52e494e6` (sha256 `b4a5b11f4e09ef6fcf601bff06a21755a5106cc7ebcf75ce350d9c1b5fd74f63`), producing tree `7b9702682a3bb9f3cad22ee6887d10b9c1b6ef7e`; with the other optional artifact on top, the cumulative tree is `d8deaa13e8f5dd3ea0db0edb31050c33cdf1317c`.  **This one touches an amended file**: the artifact patches `tests/test-logits-width-probe.cpp` for its mixed-pair probe arms, and r329 migrated that TU to the ext batch API, so the hunk was re-anchored -- it still applies **strictly** (`git am`, not `-3`) and the artifact's own mechanisms are unchanged.  `validate-set.sh` check 8 verifies all three trees on every run; the gate is `scripts/gate-mixed-kv.sh` on the optional-applied build.
* **r327 re-cut verification**: applies strictly at the r327 default tree `4792f98b82a4c6433bf11c629fdf01c21a143ce9` (sha256
  `9dc88d433b8949fd0c5646e314a9276f997931a2c2bce60204efef9e960cd7c8`), producing tree `4a6da1ba3d5799e6983116fb22ff3be7869ef31e`; with the other optional artifact on top, the cumulative tree is
  `8c1e74631c653862c3f2da5968042ec0cbba99f0`.  r327 amends **block 10** with one packed FA KV-group bitmap per graph evaluation
  (`GGML_CUDA_FA_KQ_BLOCKS_REUSE`, opt-in; `common.cuh`, `fattn-common.cuh`, `fattn.cu`, `fattn.cuh`,
  `ggml-cuda.cu`, PR #337).  This artifact's `fattn.cu` hunk anchors at the end of the file, after the
  inserted block, so git moved it and nothing was reconciled: the artifact's own diff is byte-identical
  (its sha256 moved only because the export's commit id and offsets did).  **Corrects a records gap:**
  the r324 entry below still names the r324 sha256, but r325 (block 09) and r326 (block 10) both re-cut
  this artifact -- the value quoted here is the current one.  `validate-set.sh` check 8 verifies all
  three trees on every run.
* **r324 re-cut verification**: applies strictly at the r324 default tree `d3babb65a68b83a2669933c902d5d981933d807e`
  (sha256 `980b15e310adfbf46107eb7e57a3288e72ab7d2f89db3ebe603d6ff2e81d2e27`), producing tree `eef3589ce93c3c63f8c8bd4d59147d581d5444dc`; with the other optional artifact on top, the cumulative tree is
  `ab8eb507387c468712fb2142d7e2481a529a1516`.  r324 amends **block 10** (the MoE weighted reduction: no fma contraction, one knob, and the
  reduction fold ahead of the MUL-only down fold, PR #335), a file this artifact does not touch, so the
  artifact's diff is byte-identical to the r323 cut and every number recorded here stands.
* **r320 re-cut verification**: applies strictly at the r320 default tree
  `3334a4eb31124213805ae5a0a019f6aaf86d5d7b` (sha256
  `0b60c08e256bef1c81bb9e37e1e63c328f2f67c466f3009cf0837f6f06db8694`), producing tree
  `595d4bbf37c131f5a72a8f3d583abcff66ff1393`; with the other optional artifact on top, the cumulative tree is
  `b61726912339d2f2e0a9abb646190f38690aec52`.  r320 amends **block 10** (`ggml/src/ggml-cuda/gdn-conv.cu`: the
  conv + q/k norm fold's overlap test follows a deferred conv-state gather, PR #316), a file this artifact owns
  no hunk of, so the re-cut moved only the exported commit id and the context offsets.
  `validate-set.sh` check 8 verifies all three trees on every run.
* **r319 re-cut verification**: applies strictly at the r319 default tree
  `9f53a85322eb8c0327c350fca89fe5af76a30806` (sha256
  `3e0006670f55d625acc7539d690ba80543d8114ffdc440e29904fde5690f0079`), producing tree `9cb63478129bb63aa3c1529ee890610bceecb63d`; with the other
  optional artifact on top, the cumulative tree is `bccfacc44cf28ea3a86917c1f5bc6572eba70722`.  r319 amends **block 05**
  (`fattn-common.cuh`: the decode/verify KV split is a run constant, issue #330) and **block 10**
  (`tests/test-logits-width-probe.cpp`: its 256-cell-boundary arm), and this artifact's hunk of that probe file
  (`KVV=` / `kv_from_env("KVV", …)`) had to be reconciled onto the refactored probe: the resolved hunk is still the
  three-line `KVV` change and the artifact's own mechanisms are untouched.  `validate-set.sh` check 8 verifies all
  three trees on every run.
* **r318 re-cut verification**: applies strictly at the r318 default tree
  `d586e2a5fdfda303ea8363dfac20ca72784bf90a` (sha256
  `d347cb7248e62998d77be649eca6fc6b05c2a45f69b57b22a8c1529093606807`), producing tree `9a884993b61dbacebac025fa772ff1c1764088b7`; with the other
  optional artifact on top, the cumulative tree is `820b327d2310a3d0f105d66de0e9767199bed7ff`.  r318 amends **block 10**
  (`ggml-cuda-try-fuse.cu`: the Q8_1 MUL fusion now counts readers through views, issue #331), a file this
  artifact owns no hunk of, so only the exported commit id moved and `validate-set.sh` check 8 verifies all
  three trees on every run.
* **r317 re-cut verification**: applies strictly at the r317 default tree
  `7d6b3f11062aa16922f6680af0c336973f847d1b`, producing tree `818626942ed29d2d691ee1da8096e713a2d750c8`; cumulative with the other
  optional artifact, numeric order, is `8ebf851d99874fc9992a212c775160ec16f4ee6f`.  Recorded here for the
  first time: r317 re-cut both artifacts (blocks 09/10 amendments) but wrote no bullet, so this file still
  claimed the r316 cut for one release.
* **r316 re-cut verification**: applies strictly at the r316 default tree
  `4119a6f862cff0f3b8840be51426e1a3028d8a71` (sha256
  `caee1ff2b7a4799ba0f2de9259b71959053be943c695bf57c9c7a96b9fb3d152`), producing tree `003b6e09395c770fb5902e273ae9bad4f55fdf76`; with `0100` on top, the cumulative
  tree is `3447592e07c2a260e25b0c17e1e9ecb3bf5f6b02`.  Block 05's r316 amendment rewrites the *band split helper*
  in `fattn-rdna.cuh` (the tile-kernel `type_V` rows this artifact adds are untouched), so its **patch-id is
  unchanged** from the r315 cut and `validate-set.sh` check 8 verifies all three trees on every run.
* **r315 re-cut verification**: applies strictly at the r315 default tree
  `ff3c4d85732d9dda27a6a4914c728afb541d7ae6` and produces `dd400b9d74bf5d48cb8c96e2e2281e0cffe485e6`; cumulative with the other
  optional artifact, numeric order, is `7aec397b249a6f968e1cc425d63d89fbf8fad291`.  **The patch file moved**
  (sha256 `85bdd776ed5d2e77e74bc978563f7706376bb222cb29395934b2b6b94c74fe08`): block 05's r315 amendment adds the 256-dim work gate next to the `kv_quant` line this artifact edits, so the
  exported commit id, the blob ids and the hunk offsets moved while the produced diff is otherwise unchanged.  `validate-set.sh`
  check 8 verifies all three trees on every run.
* **r314 re-cut verification**: applies strictly at the r314 default tree
  `623e5a13f376e45f01fdecc6929c6dd5de0ab1d5` and produces `b3cd09f79fcd7e5a6390f9225071e26c62a9e565`; cumulative with the other
  optional artifact, numeric order, is `4ea663b57d2d150d7452657ec526c84241e4c1a9`.  **The patch file moved**
  (sha256 `03fee67ffcc3c8ea6b23de1a7b53c23b5279a124c84077fee187bcc043e882a5`): block 05's r314 amendment adds
  the 128-dim band predicate next to the `kv_quant` line this artifact edits, so the exported commit id, the
  blob ids and the hunk offsets moved while the produced diff is otherwise unchanged (24 changed header lines,
  three `index` and three `@@`).  `validate-set.sh` check 8 verifies all three trees on every run.
* **r313 re-cut verification**: applies strictly at the r313 default tree
  `a9b88bab56ceb20a820044deda4e86504d47fe5b` and produces `f46c3778ae0c9a87692844edea2dc88d61d3b4bb`; cumulative with the other
  optional artifact, numeric order, is `06013692f6ad9a28fc69ba7da7868b3b55420e80`.  **The patch file moved**
  (sha256 `03fee67ffcc3c8ea6b23de1a7b53c23b5279a124c84077fee187bcc043e882a5`): block 11's r313 amendment (the Q8_0 hc mixer) rewrites `ggml/src/ggml-cuda/hc-mix.cu`,
  `common.cuh`, `ggml-cuda.cu`, `ggml/src/ggml-cpu/ops.cpp` and `src/models/qwen4exp.cpp`, which this artifact
  edits nothing of, so only the exported commit id and the hunk offsets moved -- the produced diff is otherwise
  unchanged.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r311 re-cut verification**: applies strictly at the r311 default tree
  `4a3db2c29d294987b95206fd663bc345114df455` and produces `7ed22548c609149d3e306fbb384a7d5183ecf021`; cumulative with the other
  optional artifact, numeric order, is `5fa077f78da0c50265febcfaf6351484475b3973`.  **The patch file moved**
  (sha256 `7648a7e5772b373da0ce0ad4135634588ff32b7cfc952abda5f67ebd75cb03a6`): block 05's r311 amendment rewrites `src/llama-context.cpp`, which this artifact edits
  nothing of, so only the exported commit id and the hunk offsets moved -- the produced diff is otherwise
  unchanged.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r307 re-cut verification**: applies strictly at the r307 default tree
  `f91aa81202fd32d060f9afb0f9c54824116758b9` and produces `38edbd0eba73bd96303c5ce8824f4b0096658e01`; cumulative with the other
  optional artifact, numeric order, is `db5d26d45d5c8258b686151e3e42b63e139c301d`.  The patch file is
  **unchanged** (same sha256) -- r307's block-10 amendment touches the host-expert cache's take-over promises, which
  this artifact owns no hunk of, so only the produced tree moved with the default tree.  `validate-set.sh` check 8
  verifies all three trees on every run.
* **r306 re-cut verification**: applies strictly at the r306 default tree
  `f12b1a0c8ee6c24b77467f93550322e74762aef0` and produces `cef109647023afdb32ef4570edb0aa752cb0c09f`; cumulative with the other optional artifact, numeric order, is `445ace2475eb41c0ae543242efcee636513898ac`.
  The patch file moved (sha256 `b5b223baa8d8b8ed5f45aebd885702c783c0193f547d3db1ccfe31e256d32e34`) because the exported commit id moved -- block 16's r306 amendment rewrites
  the commit this artifact is cut on top of -- and **only the `From <sha>` line changed**.  `validate-set.sh`
  check 8 verifies all three trees on every run.
* **r300 re-cut verification**: applies strictly at the r300 default tree
  `4baf1b67ae591e6661a1e3c5365457a747b688ab` and produces `03ec82a1dd5a97d6ac6e266397897aa0c58debcb`; cumulative with the other optional artifact,
  numeric order, is `dc9a8815bd56be9a2e9cae48f7b19d26e7da922a`.  **The patch file is unchanged** (sha256 `b64c825acfe2e0736d18b75ffd5b047ae670aa6ea3e5ec282aee4eb90e22fff2`,
  byte-identical to the r299 cut): the artifact does not touch `tools/server/server-context.cpp`, which is the
  only file block 12's r300 amendment changes, so nothing had to be reconciled -- only the produced tree moved
  with the base.  `validate-set.sh` check 8 verifies all three trees on every run.
* **r299 re-cut verification**: applies strictly at the r299 default tree
  `3da54a4c10d7350eddc5c0d8eae384086a306b0f` and produces `9068f9c8aa15a50544d1ae20bb1ed5dce16400ca`; cumulative with
  the other optional artifact, numeric order, is `3eeb8f6f1dea6d7d550980afde5072049b5ac665`.  The default tree moved by
  block 12's r299 amendment, outside this artifact's mechanisms.
* **r295 re-cut verification**: applies strictly at the r295 default tree
  `bdfd4a182dc9bdf9a7b6a46a0f206622f37b5e34` and produces `5143475fc21ee964717497dd5f0eb4c4e8a48ac8`; cumulative with the
  other optional artifact, numeric order, is `af4c85cfedb7eea6a1eb3a71eb0448a40e03aea3`.  The default tree moved by blocks
  12/14/15's r295 amendments, outside this artifact's mechanisms.
* cut against the **r288 default delivery tree** `79d18f41b9ab9b235d35fd3a5034858eddd94844` (official blocks
  `0000`-`0015` plus the temporary `0300`; that is the r287 release tree with block 17 removed)
* produces tree `89b2555296146290f556f1cf017ed6ee124dfce6` when applied alone on that tree — byte-identical
  to the r287 release tree, which is the proof that moving the feature out of the default set did not change
  it (r288 invariant 2)
* cumulative with `0100` (`0100` first — numeric order) the pair produces
  `d85868cdd23447b4d7fa0f9b9f5f7bb5fa7363cf`; the two artifacts do not conflict
* **r290 re-cut**: applies strictly at the r290 default tree
  `4f0770d7538b6e633ede9271bebdc5f83340a3c9` and produces
  `097c51af763fc338ad0d33cf7f0a853be2aa4b17`; cumulative with `0100` is
  `00f2150fad1ad44c2d40c1456c708ea4c7103390`.  The produced tree is no longer the r287 release tree's
  bytes — expected, since the r290 default tree moved (block 07's opt-in WMMA verify family) while this
  artifact's own diff is unchanged.
* **r293 re-cut verification**: applies strictly at the r293 default tree
  `92e77acfae0700b68439d69923fcb79d6fbe3a54` and produces `fafeec707a0e6c128ee6026001088b85b595272c`; cumulative with the other optional
  artifact, numeric order, is `795b1b0be4d56c75b6bb9707c46d207c0b5bb452`.  The patch file is unchanged (sha256 above) and the default tree
  moved by block 02's MTP h-rows default (r293, issue #318), outside this artifact's mechanisms.

* **r291 re-cut**: applies strictly at the r291 default tree
  `8423b2b5eac33f97ad530229b8c35240aa64fa08` and produces
  `02f8e9d0b9c0077e795c71a2e32b2ae890b41ab1`; cumulative with `0100` (numeric order) is
  `310386642121ffd22eb74268a6446b31433aada0`.  The default tree moved by block 12's r291 amendment
  (sequence parallelism), not by anything this artifact owns.
* **its own gate is `scripts/gate-mixed-kv.sh`**, run on the **optional-applied build**; it is **not** a
  default release gate any more, and the delivery's own gates do not cover a mixed pair.  Since r330 the two
  refusal arms are defaulted (not opt-in) and counted, so the gate cannot report a complete pass with an arm
  it never ran.

## What it is

The delivery's context gate refuses every `type_k != type_v` pair by default, because a mixed pair misses the
native decode band and is staged through the whole-cache F16 scratch (1.7-3.6x slower, never smaller, when it
was first refused on 2026-09-11).  The artifact gives the tile kernel a second template parameter for V
(`type_V`, defaulting to `type_KV`) and instantiates it for **head 256 only**: K `q8_0` with V `q4_0`, `q5_0`
or `q5_1`.  The tile dispatch's V type comes from `ggml_cuda_fattn_tile_type_V()`, the same function the
F16-scratch allocator asks, so the instantiation, the launcher and the allocator cannot disagree — the
contract the three earlier drifts broke (the missing q4_0 arm, the mixed-K/V NaNs, the missing RDNA2 bf16
scratch).  `ggml_cuda_fattn_kv_mixed_native()` names the pairs, the RDNA4 band predicate accepts them, and
`llama_init_from_model` accepts them on a model whose attention layers are all 256-wide and that does not use
QSA (qwen4exp is refused; `GGML_OP_FLASH_ATTN_QSA` asserts `K->type == V->type`).  Every other head size and
pair keeps the same-type rule, and the error text names the one mixed form.

**Knob.**  `GGML_CUDA_FA_KV_MIXED=0` (kill-switch, default on): the mixed pair takes the staged F16 path (the
pre-promotion behaviour); the context gate still accepts the pair.  A same-type cache does not reach the new
code at all.  The three mixed instances are always compiled, so a build cannot accept a pair it did not
compile (the context gate cannot see the backend's build flags; making them conditional would need a backend
query).  `ENVIRONMENT.md` §8 is the catalogue entry.

## Which layouts it reaches — and which it cannot

It is a **cache-type** feature, so it is orthogonal to the device count: it works on one GPU, `-sm layer` and
`-sm tensor`, on gfx1201 / gfx1100 / gfx1151 (all three were gated).

| layout | result |
|---|---|
| Qwen3.5 / 3.6 / 3.8 hybrids (dense and MoE) whose attention layers are all 256-wide | **engages** — the pair is accepted and read natively |
| any model with a different head width (e.g. 128-wide attention) | **refused** by the context gate, same error text as before |
| a qwen4exp model (QSA sparse attention) | **refused** — its QSA op asserts `K->type == V->type` |
| any other mixed pair (e.g. K q6_0 with V q5_1) | **refused** |

## What it measures here (r287 gate ladder, 3 x R9700 gfx1201 unless noted)

* `test-backend-ops -o FLASH_ATTN_EXT` **7021/7021** on gfx1201, gfx1100 and gfx1151, in the default (native
  read), `GGML_CUDA_FA_KV_MIXED=0` and `GGML_CUDA_FA_KV_NATIVE=0` settings.
* W=1..8 width purity **maxdiff 0** for q8_0/q8_0 and all three pairs, on 1 GPU, `-sm layer` and `-sm tensor`.
* plain == draft-mtp at n_max 1/3/7 on all three arches.
* At `-c 131072`, 27B Q4_K_M, 3-GPU `-sm tensor`: KV **4352 MiB** (q8_0/q8_0) → **3712 / 3584 / 3328**
  (q5_1 / q5_0 / q4_0), the same compute buffers and `graph splits = 2`; decode at 32768 cells is 0.3-3.2 %
  below q8_0/q8_0, with mixed between the same-type neighbours (47.0 vs 51.6 / 46.0 t/s at 3 GPUs,
  q8_0/q5_1).  The staged kill-switch arm is 12 % behind the native read at 32768 cells and 25 % at 100000.
* Quality (27B, 3-GPU tensor, WikiText-2, 8 x 4096): mean KLD against an f16 base 0.0071 (q8_0/q8_0), 0.0079
  (q8_0/q5_1), 0.0082 (q8_0/q5_0), 0.0074 (q8_0/q4_0) — the mixed pair is within ~11 % of the same-type
  q8_0/q8_0 cost.

The trade is why it is an opt-in artifact: it buys a smaller KV cache (up to ~1 GiB at 131072 cells on 27B) at
a few percent of decode, which is worth it when the cache is what does not fit and is a small loss when it
does.  The default set stays at 16 official blocks plus the temporary `0300`.

## Not covered

Head sizes other than 256 (still refused), other mixed pairs (still refused), and quality at depth (the KLD
run is 4096-token chunks).  The artifact is not covered by the delivery's release gates; `scripts/gate-mixed-kv.sh`
on an optional-applied build is the gate.

## History

* PR #312 (@mjohanning99) — the submission, the review and the original measurements are in
  `../../archive/work/mixed-kv/` (revision 2 added the QSA refusal and the one-V-type-function change).
* r287 — promoted into the delivery as **block 17**, after the gate ladder.
* r288 — the block-number re-org moved it out of the default set as **optional block `0101`**.  The moved
  block's bytes are unchanged: applying this patch on the default tree reproduces the r287 release tree
  exactly (`89b25552…`).
