# HANDOVER — cold-start brief for the `4f9dfa0e3` re-base (`upstream-rebase-261113`)

**Written 2026-11-13 at the end of session 4.  Read this first, then `RUNGS.md`.**  It is a *state* document:
the method is in [`README.md`](README.md), the per-rung ledger and the R09 conflict plan are in
[`RUNGS.md`](RUNGS.md), the stop-and-ask register is in [`DECISIONS.md`](DECISIONS.md), and the measured
conflict map is in [`CONFLICT-MAP.md`](CONFLICT-MAP.md).

---

## §0  State at a glance

| item | value |
|---|---|
| campaign | `wip/upstream-rebase-261113/` — 17 blocks from `a691b2345` onto **`4f9dfa0e3`** (164 commits, 751 files, +40 612 / −24 226) |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-261113`** (never `main` until the climb is green) |
| fork checkout | `~/llama.cpp`, branch **`rebase-r278`**, clean, **HEAD `1d83b50e0`** (`master` = pristine `4f9dfa0e3`, never commit to it) |
| progress | **R00-R12 done = blocks 00-08 + the rung-repair + blocks 09-11**; next is **R13 = block 12 (campaign remainder)** |
| rung chain | `43a418c10` (00) `ab433a7dd` (01) `50a0b30f8` (02) `43891459e` (03) `4c6a5019a` (04) `22fcde21e` (05) `c1d8b7c04` (06) `626e398d2` (07) `a95c0497b` (08) **`9c94ce139` (R09r rung-repair, block 05's artifact)** `a6809f75b` (09) `13176810a` (10) `1d83b50e0` (11) |
| canonical build | `~/llama.cpp/build-rocm` — **R12 tree** (`1d83b50e0`), `GGML_HIP_NO_VMM=OFF`, `gfx1201`, ROCm 10.1.0 |
| old reference build | `~/llama.cpp/build-rocm/bin-r277` (the r277 release, `build 12222, commit 9d7c0502c`) |
| unpatched base build | `~/llama.cpp/build-rocm-base2` — `4f9dfa0e3`, **`GGML_HIP_NO_VMM=ON`** (the only unpatched configuration that *runs*; see §7) |
| coherence oracle (current) | 4B, 1 GPU, `-sm layer`, seed 42 -> **100 chars `sha=1c5d32ac537d`**, unchanged at every rung since R01 |
| delivery untouched | `patches/`, `release.json` and the fork are **not** changed until R18.  Nothing is pushed anywhere. |

A rung is **not** finished until its ledger row and resolution log in `RUNGS.md` are written; a session that
stops mid-rung must `git cherry-pick --abort` first (see §8).

---

## §1  Cold start (do this before touching anything)

```bash
cd ~/llama.cpp        && git status --short --branch && git log --oneline -3
cd ~/llama-cpp-rdna-boosts && git log --oneline -3 && sed -n '1,40p' wip/upstream-rebase-261113/RUNGS.md
```

Expect: `rebase-r278` clean at `a95c0497b`, no `CHERRY_PICK_HEAD`
(`ls ~/llama.cpp/.git/CHERRY_PICK_HEAD` must fail), delivery on `rebase-261113`.

Then read, in this order: **this file** -> [`RUNGS.md`](RUNGS.md) §R11 (the next rung) and §R10/§R09r ->
[`DECISIONS.md`](DECISIONS.md) `D6`/`D8`/`D10` (the R11 items) -> [`README.md`](README.md) §1 (rung mechanics)
and §3-§4 (decision policy + release ladder).
`BASE-BLOCKER.md` is history now (`D17` decided) but explains why block 00 carries a revert.

---

## §2  The rules of the climb (short form)

1. **One rung = one cherry-pick** of the block commit onto `rebase-r278`, resolved, committed, built and
   coherent.  Never `rebase --onto`, never regenerate `patches/` mid-climb (`make-patches.sh` runs once, at
   R18, from the finished branch).
2. **Scope is `patches/` only.**  Nothing from `wip/`, `beta/`, `optional/` or `archive/work/` is applied,
   consulted as a patch, or folded in.  A rung that seems to need one gets a `TODO.md`-shaped note.
3. **Report before dropping delivery code.**  A conflict that would delete a delivery mechanism stops the
   rung: measure, write the options into `DECISIONS.md`, and ask the maintainer.  *Nothing has been dropped to
   date* — `D2` and `D5` both resolved as "complementary", see §5.
4. **Never push.**  `~/llama.cpp` has upstream as a remote; a bare `git push` there is unacceptable.  This
   campaign does not push at all (R18 does the fork refresh, with the maintainer's go-ahead).
5. **Never commit to `master`** in `~/llama.cpp`.
6. Judge a rung by its resolutions + build + coherence + its own mechanism oracle, **not** by line count and
   **not** by r277-identical text (upstream moved numerics on purpose in this range).

---

## §3  Rung mechanics (exact commands)

```bash
cd ~/llama.cpp
git cherry-pick <block sha>                 # conflict -> resolve per plan
#   resolve; then:
git add -A                                  # ALWAYS -- see §8 trap 3
GIT_EDITOR=true git cherry-pick --continue
EXTRA_CMAKE_FLAGS="-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q4_1-q4_1;q5_0-q5_0;q5_1-q5_1;q8_0-q8_0;bf16-bf16;f16-f16;iq4_nl-iq4_nl;f8_e4m3-f8_e4m3" \
  timeout 3000 ~/bin/build-llama-rocm       # ~2-4 min incremental; the box's cache already holds this list
```

Coherence (the standard arm — **no `-lv`**):

```bash
HIP_VISIBLE_DEVICES=0 timeout -s KILL 180 ./build-rocm/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -sm layer -p "The capital of France is" -n 20 --seed 42 --temp 0 \
  --no-display-prompt --single-turn > /tmp/<rung>-coh.log 2>&1
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/<rung>-coh.log   # expect sha=1c5d32ac537d
```

Other arms that exist and are cheap (use the one the block owns):

| arm | command sketch | why |
|---|---|---|
| 3-GPU tensor | `HIP_VISIBLE_DEVICES=0,1,2 … -sm tensor -lv 4` | block 01's all-reduce; grep `ggml_cuda_ar_pipeline_init: initialized HIP AllReduce pipeline: 3 GPUs` |
| field MTP | `HIP_VISIBLE_DEVICES=0,2 … 35B-A3B Q8_0 -sm layer\|tensor -ctk bf16 -ctv bf16 --spec-type draft-mtp-adaptive --spec-draft-n-max 3 --spec-draft-n-min-adaptive 1 -lv 4` | blocks 02/13; grep `adaptive draft depth enabled`, `target h rows on the device`, `draft acceptance` |
| GDN prefill census | `HIP_VISIBLE_DEVICES=0 … 4B -b 2048 -ub 2048 --fit off -lv 4 -f prompts/prose-rdna-boosts.txt -n 8` | block 03; grep `whole-batch chunked GDN prefill` |
| MMB census | same + `GGML_CUDA_MMB_LOG=1` | block 06; grep `MMB_CFG` |
| op oracles | `./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT\|MUL_MAT\|MUL_MAT_ID` | blocks 05/07/00 (6996/0, 2681/0, 1040/0 at their rungs) |

A `-lv 4` run **pollutes the text stream** — a hash and a census are always two runs (`AGENTS.md`, the `-lv`
table).  Every gate that hashes text passes no `-lv` at all.

---

## §4  The ladder: done + remaining

| rung | block | status | sha | notes |
|---|---|---|---|---|
| R01 | 00 foundation | DONE | `43a418c10` | one conflict (`llama-mmap.cpp`, upstream-wins rename); **carries the `ac931b5bc` revert** (`D17` option 1) |
| R02 | 01 all-reduce | DONE | `ab433a7dd` | clean; 3-GPU pipeline proven |
| R03 | 02 adaptive MTP | DONE | `50a0b30f8` | `D7` measured; `D8` include lesson |
| R04 | 03 chunked GDN | DONE | `43891459e` | `D2` complementary; chunked census 6908 t/s |
| R05 | 04 BF16 KV | DONE | `4c6a5019a` | clean; `D3` audit passed |
| R06 | 05 flash attention | DONE | `22fcde21e` | `D5` FA half complementary; 6996/0 FA oracle; union instance matrix |
| R07 | 06 MMB prefill | DONE | `c1d8b7c04` | our blocks on upstream's MMVF/fusion predicates; MMB census 6915 t/s |
| R08 | 07 k-quant boosts | DONE | `626e398d2` | graph-reuse union; `MUL_MAT` 2681/0 (`D16`) |
| **R09** | **08 fused kernels** | **DONE** | `a95c0497b` | 7 conflicts resolved; **`D12` dual store kept**, `D6` partial; **found `D5-a`** (heads 40/72 have no mma instance — R06 gap, AWAITING) |
| R10 | 09 host-resident MoE | **DONE** | `a6809f75b` | 14 hunks, all "both sides"; `D1` answered (no port), arena live, pool text identity `a99ccbd97ecb`; its gate's `tsplit*` arms are **R13's** — see §5.5 |
| R11 | 10 infrastructure (slab, fusion dispatcher, async ring) | **DONE** | `13176810a` | 9 files / 20 hunks; the whole fusion dispatcher moved into `ggml-cuda-try-fuse.cu`; `gate-norm-rows-alias.sh` **PASS**; `gate-node-budget.sh` is **R13's** |
| R12 | 11 qwen4exp | **DONE** | `1d83b50e0` | 6 hunks — the cleanest rung; `D11` decided (op count **105**, RPC 7); `gate-hc-fusions.sh` **PASS** (`d62340fbf566`), quant-coherence A-D **PASS**; its section E (sparse QSA) is **R16's** (block 15 carries the `#30206` fix) |
| R13 | 12 campaign remainder | TODO | — | 4 conflicts; `D10`, `D11`, `D15`; **owes `gate-host-pool.sh` `tsplit`/`tsplitnosink`** (block 12 adds `moe_cache_pool_backed`) **and `gate-node-budget.sh`** (block 12 has `graph_max_nodes()` + the "over its node budget" refusal; at R11 arm B aborts, so R13 must show the margin, not just the pass) |
| R14 | 13 MTP vocab subset | TODO | — | **0 conflicts**; re-verify `D7` |
| R15 | 14 relaxed-tensor (TEMPORARY) | TODO | — | **0 conflicts**; check upstream has not started allowing those arches |
| R16 | 15 streamed KV | TODO | — | 5 conflicts; `D3`, `D19`; owes the KV residency gates; **also `gate-qwen4exp-quant-coherence.sh` section E** (the sparse QSA arm — block 15 replaces `ggml_rope_get_truncate(dst)` on the fused `INDEXER_SCORE` node, which upstream `#30206`'s assert now rejects) |
| R17 | 16 UMA GTT store | TODO | — | **0 conflicts**; verify `D17` (VMM region size) with `gate-uma-experts.sh --quick` |
| R18 | — | TODO | — | release ladder, `README.md` §4 + §10 below |

**Remaining block commits to cherry-pick** (from `release.json` order; a later block is judged against the
rebased earlier ones, so the order is not negotiable):

```
R11 05f298fea   R12 ea4963486   R13 c7628e974   R14 f0c7f3359   R15 6b4d9865e
R16 99d33117a   R17 9d7c0502c
```

Per-rung expected conflict paths and "what to expect" notes: `CONFLICT-MAP.md` (measured) and `RUNGS.md`
(per-rung tables).

---

## §5  What was decided so far (so it is not re-litigated)

### §5.1 `D17`/B1 — the base blocker (DECIDED: option 1)
Upstream `ac931b5bc` (`#30542`) hangs the HIP VMM pool on ROCm 10.1/gfx1201: the **unpatched** base never
returns from its first GPU compute with the delivery's required `GGML_HIP_NO_VMM=OFF`.  Block 00 therefore
carries a **full revert of that commit** (maintainer's call).  Drop the revert at the next re-base once
upstream fixes the pool.  Do not "try VMM on" to work around anything else — the arena/GTT store need VMM.

### §5.2 `D2` — chunked GDN (DECIDED at R04: complementary, nothing dropped)
Upstream `#32078` deleted **its own** non-fused GDN graph builders, the GDN fused-node tags and their
registry entries.  The delivery's substance is the `n_rs_batch` argument on `ggml_gated_delta_net()`, the
pre-batch snapshot copy and the `gated_delta_net_chunked*.cu` kernels — and since upstream now always routes
through that same fused op, the delivery's kernels are the surviving implementation.  Consequence for later
rungs: **`LLM_FUSED_OP_GDN_AR/CH` do not exist upstream any more**; do not re-add them, and their
`fused_gdn_ar/ch`/`auto_fgdn` cparams were removed at R06.

### §5.3 `D5` — flash attention (DECIDED at R06: complementary; few-row half deferred)
The FA band is a *selector* over the same kernel matrix, so upstream's FP8 vec FA, heads 40/72 and the sm70
case sit beside it.  Two things were learned that matter later:
* upstream added **`DKQp`/`DVp = GGML_PAD(head, 16)`** for head 72 — the delivery's inherited loops, Q-tile
  stride and `VKQ_C` sizing need it; if it ever looks "unused", it is not.
* the instance matrix is the delivery's 121 typed `...-dkqN-dvN.cu` files (the 16 head-agnostic files are
  deleted); new upstream head sizes must be declared in that typed set, and the oracle is
  `test-backend-ops -o FLASH_ATTN_EXT` (6996 cases, 0 failures at R06).
* the box's `GGML_CUDA_FA_QUANTS` cache now carries the union list (`…;iq4_nl-iq4_nl;f8_e4m3-f8_e4m3`), so
  the r277-era "hand-set iq4_nl" trap no longer bites.

### §5.5 R09r (rung-repair) + R10 (block 09) — both DONE; the `-sm tensor` fault is R13's

**`9c94ce139` (R09r) — block 05's FA instance set, found at R09 as `D5-a`, decided and applied.**  Upstream's
typed -> head-agnostic instance conversion folded its two new heads (40, 72) into itself, so R06's correct
"keep the delivery's typed layout" dropped exactly those two: 32 undefined
`ggml_cuda_flash_attn_ext_mma_f16_case<40|72,…>` symbols against an unconditional dispatch in `fattn.cu` (a
latent abort on AMD, not an NVIDIA-only artefact), plus the FP8 KV column the same conversion hid.  The repair
is **additive**: 32 new mma instance files + 14 FP8 KV instances + a **typed** `generate_cu_files.py` loop
(as merged it emits upstream's head-agnostic files and would delete the delivery's 126 on a re-run).  Verified:
undefined FA symbols **32 -> 0**, build green, `FLASH_ATTN_EXT` **6995/6995** (448 head-40, 1009 head-72),
coherence `sha=1c5d32ac537d`, generator idempotent on the tree, and `gate-norm-rows-alias.sh` now links and runs.
It is squashed into block 05 at R18.

**`a6809f75b` (R10) — block 09, DONE.**  14 conflict hunks across 8 files, every one "both sides" (table in
`RUNGS.md`); build green; coherence `sha=1c5d32ac537d`; the 1-device arena is live (`h=0.5205`, **fills=1991**,
`takeover=540`); the pool's `ref` / failed-fill / sink / no-disk-tier arms are all **182 chars
`sha=a99ccbd97ecb`**, the same hash as the 2-device `-sm layer` run.  `D1` was **answered, not deferred**:
upstream moved *its own* MoE cache to `llama-moe-cache.*`, the delivery's tier is a different implementation in
a different place, and the r236 either/or ruling stands.

**The `-sm tensor` arena fault was NOT a merge defect.**  R10 was parked on it (as `d4106d6f6`) and the
attribution to `D10`'s `hv_tensor_requires_copy` union was **wrong**.  The Meta backend at this rung has **no
`moe_cache_pool_backed` forwarding at all** (`grep -rn moe_cache_pool_backed` finds nothing) and **block 12 adds
it** — "forward the pool-backed query to the simple devices.  A tensor split gives every simple device the SAME
host master".  Without it a Meta-wrapped host expert weight is not reported as pool-backed, so the fused kernel
(`mul_mat_vec_q_moe`) reads the **pageable** host master -> `Page not present` on both GPUs, right after slab
init.  r277 passes that arm only because block 12 is in it.  So **`gate-host-pool.sh`'s `tsplit` +
`tsplitnosink` arms are R13's** (like `gate-norm-rows-alias.sh` is R11's) and they must go green there; at R10
the gate reaches 11 of its 13 arms and cannot print a verdict, so the substance was verified directly (the four
hashes above plus the 1-device census).  The parked tag `r10-blocked-d4106d6f6` remains in `~/llama.cpp` as
history; the patch file was deleted once re-applied (it no longer applied on the repair commit's parent).

### §5.4 R09 (block 08) — the last rung, DONE at `a95c0497b`
The full result, the six `norm.cu` conflict resolutions and the verification table are in
[`RUNGS.md`](RUNGS.md) `R09 result`.  The three facts a later rung needs:

* **`D12` merged as a dual store, not a collapse**: `if (store_f32) { rms_norm_store4(dstc, i, r); }` **and**
  `if (dst16c) { ... norm_f2bf ... }`, in *both* the 4-wide and the scalar path, with both `convert.cuh` and
  `mmb.cuh` included.  Nothing fills `dst16` until block 15, so the side channel is inert and its gate is R16's.
* **`mmq.cu` did not become a plain take-theirs**: upstream's new **no-gate fused-scale** arm
  (`ggml-cuda.cu`, nvfp4 `w_s`) calls `ggml_cuda_mul_mat_q()` with `gate == nullptr`, so the delivery's literal
  `GGML_ASSERT(gate && ...)` would abort it.  The narrowing is kept *inside `if (gate)`* instead.
* **`gate-norm-rows-alias.sh` is R11's, not R09's** — the r225 narrow-row fusion and its `norm-rows:` census
  are block 10's code and are absent at R09.  It is also blocked until **`D5-a`** is repaired, because its
  section A links a driver against `libggml-hip.so` and the merged matrix leaves 32 FA case symbols undefined.
  `D5-a` (heads 40/72 have no `fattn-mma-f16` instantiation; the generator disagrees with the tree) is an **R06**
  gap, is **AWAITING** the maintainer, and does not block the climb.

---

## §6  Decision register status (do not re-decide by accident)

| item | status |
|---|---|
| `D1` MoE cache relocation / CPU share | **ANSWERED at R10** — no port needed (upstream moved its *own* cache; the delivery's arena/slab/copy-callback tier is elsewhere and untouched).  The r236 either/or ruling stands: arena default-on, upstream cache dormant. |
| `D2` GDN pruning | **DECIDED** complementary (R04) |
| `D3` FP8/BF16 KV types | **CLOSED** (R05 audit + R06 FA oracle) |
| `D4` MMVF BF16 few-row | **DEFERRED to R18** (no rung conflict; a `llama-bench` arm) |
| `D5` FA + few-row MMQ | **DECIDED** FA half (R06); few-row half **deferred to R18**; **`D5-a` (heads 40/72 have no mma instance) found at R09 — AWAITING** |
| `D6` graph/stream state moved to `graph.cuh` | **PARTIAL** — R09: the move landed and no delivery hook had to follow it (block 08's hooks are `ggml_backend_cuda_context` members, which stayed in `common.cuh`); the cluster-dim key is verified at R11 |
| `D7` MTP device-buffer hidden state | **MEASURED/DECIDED** (R03): activates on `-sm layer`, declines on `-sm tensor` ("the device has no events" — the Meta device has no `event_new`) and falls back to the delivery's host path.  Upstream-gap note: give the Meta device events (an `upstream/` candidate).  Re-verify at R14. |
| `D8` the prune | **MECHANICAL/OPEN**: it also removes *includes* — a build break, not a conflict (§8 trap 4).  Blocks 10/11 will hit it. |
| `D9` mmap / Win-load / lazy-mode | **PARTIAL** — R01 resolved the only conflict and R10 took upstream's Windows rewrite wholesale (the delivery's `fname`/`path()` and `llama_mmap::impl` work survive); the substantive questions are R16/R17's |
| `D10` scheduler upload/prefetch/graph-result | **`MEASURING` — the risk is live at R10**: the `hv_tensor_requires_copy` union is the leading suspect for the `-sm tensor` arena fault; `gate-host-pool.sh`'s `tsplit` arm is the oracle |
| `D11` `ggml.c`/RPC/test refactors | **OPEN** — R12/R13 |
| `D12` BF16 RMS_NORM | **DECIDED at R09** — dual store kept (both destinations ship), upstream's typed `dstc` writer + the delivery's bf16 side channel; the rows gate is R11's, the side channel's own gate is R16's |
| `D13` HIP BLAS workspace | **CHECKED** (R02) |
| `D14` GDN/SSM kernels | **CHECKED** (R04) |
| `D15` external PLE | **merge done at R10** (`path_model` -> `paths_model` + the delivery's host-expert out-params); the PLE-residency question is R16's |
| `D16` MUL_MAT src1 precision | **CLOSED** (R08, 2681/0) |
| `D17` VMM pool | **DECIDED** (option 1) |
| `D18` SYCL device-side MoE | **CHECKED at R10** — no SYCL file in the conflict set, no shared op flag changed |
| `D19` plamo2 recurrent cache names | **OPEN (verify only)** — R16 |

---

## §7  Environment, models and gotchas

* **Build**: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `gfx1201`, `GGML_HIP_NO_VMM=OFF`,
  ccache ~30 GiB).  `clean` wipes `build-rocm` (only needed if the cache is inconsistent, ~6 min cold).
  Never pass `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=…"` (the script's include-order guard aborts).
* **`build-rocm-base2` is unpatched `4f9dfa0e3` with `GGML_HIP_NO_VMM=ON`** — the only unpatched base
  configuration that runs on this box (see `BASE-BLOCKER.md`).  It is the reference for "is this upstream or
  is this us?" questions, and R18 needs it for the prefill-logit base record (with the asymmetry noted).
* **Models**: `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` (the coherence oracle; also a hybrid
  GDN model), `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (the field host-expert/MTP
  config; `runme-rocm` beside it), `/llm/models/Qwen3.8/27B/…` (qwen4exp class), `prompts/prose-rdna-boosts.txt`
  (5 298 tokens, sha256 in `prompts/README.md`).
* **Pin `HIP_VISIBLE_DEVICES`** on every arm (the 9950X3D's iGPU is enumerated and the `-sm tensor` load
  aborts with `invalid kernel file`).  `0` = one R9700, `0,2` = the x8 pair (the field config), `0,1,2` = all
  three for the all-reduce arm.
* **`timeout -s KILL`** on anything that could hang: a spinning `llama-cli` ignores SIGTERM.
* The `llama-cli --version` string is baked **before** the commit, so a freshly committed rung still reports
  the previous sha.  Do not read it as "the build did not pick up the change".

---

## §8  Traps that have already bitten (in this campaign)

1. **Never commit to `master`**; the fork `master` is the pristine base snapshot.
2. **One long-lived source path** (`~/llama.cpp`): ccache keys on the absolute path; a new worktree is a cold
   build.
3. **`git add -A` after resolving.**  `git diff --diff-filter=U` reads the *index*, so a resolved-but-unstaged
   file still looks unmerged.  And for **UD** (deleted-by-them) instance files, `git rm --cached` leaves the
   file on disk — it must be deleted (or `git rm -f` before staging) or `git add -A` will resurrect it.
4. **A prune removes *includes*, and that is a build break, not a conflict** (`D8`): the block's include hunk
   attaches cleanly and the symbol then resolves nowhere.  After every prune rung, check the include surface
   of the delivery's symbols (`llama-ext.h` especially).
5. **Upstream refactored the APIs the delivery's blocks call.**  At R06 the block's call sites had to be
   re-expressed on: `graph_max_nodes()` (no `n_tokens` — `#32078` made it ubatch-independent),
   `sched_new({.max_nodes, .parallel, .op_offload})` (`#32009`), `graph_reserve(..., packed_kq_mask)` (already
   merged), and at R07 `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)` (`#29633`/`#31959` — takes the
   tensor now) and `should_fuse_mul_mat_vec_f(up, gate)` (the `gate` argument defaults to `nullptr`).
6. **`git grep '<<<<<<<'` matches binary GGUFs** — use `git grep -I`.
7. **A rung build may need a first-error pass** (R06: `DKQp`/`DVp` undefined because upstream *added* them and
   the delivery's code needed them).  Compile, read the errors, fix, rebuild — do not assume the merge is done
   because the text looks right.
8. **Do not leave a half-resolved rung.**  If the session is running out of room, `git cherry-pick --abort`,
   record the analysis in `RUNGS.md`, and hand over (that is what happened at R09).
9. **`test-recurrent-state-depth` is red on this box on the shipped r277 build too** (73-83 Phase-A
   mismatches depending on config) — pre-existing and config-sensitive, not a rung regression.  Run it with
   the 4B and `-ngl 99`; record the count rather than treating it as a gate.
10. **`GGML_CUDA_FA_QUANTS`** is a CMake cache variable: for blocks 00-04 an `iq4_nl` entry is a configure
    *error*; from R06 on it is valid (and the cache already carries it).  If a build fails in
    `ggml/cmake/common.cmake` with `unknown type`, that is the cause.

---

## §9  Observations parked for R18 (not blockers)

* **Prefill-logit base record asymmetry.**  Unpatched `4f9dfa0e3` cannot run with `VMM=OFF` (§7), so the base
  record comes from `build-rocm-base2` (unpatched, `VMM=ON`) and the candidate from the canonical `VMM=OFF`
  build.  The pool affects allocation only, but the asymmetry must be stated in the gate record (there is a
  note in `BASE-BLOCKER.md` §"Consequence for R18").
* **`gate-node-budget.sh` is owed** at R11/R18: upstream `#32078` made `graph_max_nodes()` ubatch-independent
  and the delivery's later blocks add graph nodes (the KV rotation, the async ring).  The delivery's patches
  still describe `graph_max_nodes(n_tokens)` in `patches/README.md` (line ~3494/4063) — fix at R18.
* **`D7`'s Meta-device event gap** is a genuine upstream hole worth an `upstream/` candidate (an
  `event_new` for `ggml-backend-meta`, or falling back to the output device's backend).
* **The r277 `test-recurrent-state-depth` mismatch** (§8 trap 9) is worth a `TODO` once R18 re-runs it on the
  released tree.
* `patches/README.md` block notes and `ENVIRONMENT.md` rows that will need updating after R18: the
  `LLAMA_KQ_MASK_*` / derived-mask block, the MMB census, the FA instance-matrix description
  (`iq4_nl` + `f8_e4m3`), the `graph_max_nodes` note.

---

## §10  R18 — the release ladder, with this campaign's additions

`README.md` §4 has the full list.  The additions discovered during the climb:

1. `scripts/make-patches.sh` from the finished branch (17 files, block 00 keeps `0000`) -> `make-release.sh`.
2. `scripts/validate-set.sh` — strict apply on a fresh tarball at `4f9dfa0e3`, tree equality, sha256s, Windows
   lints.
3. `scripts/gate-per-block-build.sh` — a clean build per block commit (17).
4. Coherence: dense 4B, the 35B-A3B field/host-expert config, a hybrid recurrent model, and the qwen4exp
   class.
5. `scripts/gate-prefill-logits.sh --record` from the **unpatched base** (`build-rocm-base2`, `VMM=ON`), then
   the candidate; plus the `--sm tensor --device 0,1,2` arm (block 01), a `--kv q8_0` arm, and an FP8 KV arm.
6. `scripts/gate-rebase-release.sh` (the one-command pre-release verdict) and `check-gpu-faults.sh`.
7. The block gates that moved: host-pool, kv-residency, kv-window-split, park-idle-slots, ckpt-reuse,
   state-pipe, spec-align, spec-width-band, spec-draft-window, spec-reuse-window, hc-fusions,
   norm-rows-alias, grid-stall-pad, qwen4exp-quant-coherence, uma-experts, fusion-alias-copy,
   **node-budget**, plus the FA and MUL_MAT op oracles.
8. MTP four-axis + `llama-batched-bench -npl 1,4,8`; the `D4`/`D5` few-row `llama-bench` arms.
9. **Records**: a `WORKLOG.md` entry (new base/tip/tree + gate numbers), `patches/README.md` per-block notes,
   `BASELINE.md` current baseline, `INVARIANTS.md` appended facts, `ENVIRONMENT.md` for every variable that
   appeared/vanished, `TODO.md` for the deferred decision items, and this campaign moved to
   `archive/work/upstream-rebase-261113/` (no `wip/` stub).
10. Fork refresh (`scripts/fork-refresh.sh --build --gates --push`, personal fork only) + the off-host
    container images — with the maintainer's go-ahead, never autonomously.

---

## §11  Session-close protocol

Before stopping, always:

1. Finish or **abort** the rung (no dirty tree, no `CHERRY_PICK_HEAD`).
2. Update `RUNGS.md`: the ledger row (status/sha), the rung's result + resolution log, and the pre-analysis
   of the *next* rung if it was reached and not done.
3. Update `DECISIONS.md` for any item the rung touched.
4. Update this file's §0 table (progress, HEAD sha, next rung) and, if a delivery-affecting fact changed,
   `WORKLOG.md` (dated entry, newest first; the 2026-11-13 campaign entry is `261113.004`).
5. Commit the delivery repo on `rebase-261113` (the fork branch is committed by the rung itself).
