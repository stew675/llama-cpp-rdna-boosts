# HANDOVER — cold-start brief for the `4f9dfa0e3` re-base (`upstream-rebase-261113`)

**Refreshed 2026-11-13 at the end of session 6 (R13 -> R15).**  Read this first, then `RUNGS.md`.  It is a
*state* document: the method is in [`README.md`](README.md), the per-rung ledger and resolution logs are in
[`RUNGS.md`](RUNGS.md), the stop-and-ask register in [`DECISIONS.md`](DECISIONS.md), the measured conflict map
in [`CONFLICT-MAP.md`](CONFLICT-MAP.md), and the paste-able brief for the next session in
[`PROMPT.md`](PROMPT.md).

---

## §0  State at a glance

| item | value |
|---|---|
| campaign | `wip/upstream-rebase-261113/` — 17 blocks from `a691b2345` onto **`4f9dfa0e3`** (164 commits, 751 files, +40 612 / −24 226) |
| delivery repo | `~/llama-cpp-rdna-boosts`, branch **`rebase-261113`** (never `main` until the climb is green) |
| fork checkout | `~/llama.cpp`, branch **`rebase-r278`**, clean, **HEAD `f040d394f`** (`master` = pristine `4f9dfa0e3`, never commit to it) |
| progress | **R00-R15 done = blocks 00-08 + one rung-repair + blocks 09-14**; next is **R16 = block 15 (the streamed KV, 5 expected conflicts)** |
| rung chain | `43a418c10` (00) `ab433a7dd` (01) `50a0b30f8` (02) `43891459e` (03) `4c6a5019a` (04) `22fcde21e` (05) `c1d8b7c04` (06) `626e398d2` (07) `a95c0497b` (08) **`9c94ce139` (R09r rung-repair, block 05's artifact)** `a6809f75b` (09) `13176810a` (10) `1d83b50e0` (11) `75d83f231` (12) `aca78625a` (13) **`f040d394f` (14)** |
| canonical build | `~/llama.cpp/build-rocm` — **R15 tree** (`f040d394f`, tree `0c8a1fd58`), `GGML_HIP_NO_VMM=OFF`, `gfx1201`, ROCm 10.1.0 |
| old reference build | `~/llama.cpp/build-rocm/bin-r277` (the r277 release, `build 12222, commit 9d7c0502c`) — **the whole 17-block delivery**, so it is the answer to "would block N alone do this?" |
| unpatched base build | `~/llama.cpp/build-rocm-base2` — `4f9dfa0e3`, **`GGML_HIP_NO_VMM=ON`** (the only unpatched configuration that *runs*; see §7) |
| coherence oracle (current) | 4B, 1 GPU, `-sm layer`, seed 42 -> **100 chars `sha=1c5d32ac537d`**, unchanged at every rung since R01 (re-verified on the R15 tree) |
| parked history | tags `r10-blocked-d4106d6f6` / `r10-reapplied` in `~/llama.cpp` (the R10 park, since re-applied as `a6809f75b`); no patch file is live — do not look for one |
| delivery untouched | `patches/`, `release.json` and the fork are **not** changed until R18.  Nothing is pushed anywhere. |

A rung is **not** finished until its ledger row and resolution log in `RUNGS.md` are written; a session that
stops mid-rung must `git cherry-pick --abort` first (see §8).

---

## §1  Cold start (do this before touching anything)

```bash
cd ~/llama.cpp            && git status --short --branch && git log --oneline -3
ls ~/llama.cpp/.git/CHERRY_PICK_HEAD        # must fail
cd ~/llama-cpp-rdna-boosts && git status --short --branch && git log --oneline -3
grep -n '^## R' wip/upstream-rebase-261113/RUNGS.md | head -30   # find the open rung
```

Expect: `rebase-r278` clean at `1d83b50e0`, no `CHERRY_PICK_HEAD`, delivery on `rebase-261113` clean.

Then read, in this order: **this file** -> [`RUNGS.md`](RUNGS.md) §R13 (the next rung) and §R12/§R11 (what the
last two rungs decided) -> [`DECISIONS.md`](DECISIONS.md) `D10`/`D11`/`D15` (the R13 items) ->
[`README.md`](README.md) §1 (rung mechanics) and §3-§4 (decision policy + release ladder).  `PROMPT.md` has the
same as a paste-able brief.  `BASE-BLOCKER.md` is history (`D17` decided) but explains why block 00 carries a
revert.

---

## §2  The rules of the climb (short form)

1. **One rung = one cherry-pick** of the block commit onto `rebase-r278`, resolved, committed, built and
   coherent.  Never `rebase --onto`, never regenerate `patches/` mid-climb (`make-patches.sh` runs once, at
   R18, from the finished branch).
2. **Scope is `patches/` only.**  Nothing from `wip/`, `beta/`, `optional/` or `archive/work/` is applied,
   consulted as a patch, or folded in.  A rung that seems to need one gets a `TODO.md`-shaped note.
3. **Report before dropping delivery code.**  A conflict that would delete a delivery mechanism stops the
   rung: measure, write the options into `DECISIONS.md`, and ask the maintainer.  *Nothing has been dropped to
   date* — `D1`, `D2`, `D5` and `D12` all resolved as "both sides", see §5.
4. **Never push.**  `~/llama.cpp` has upstream as a remote; a bare `git push` there is unacceptable.  This
   campaign does not push at all (R18 does the fork refresh, with the maintainer's go-ahead).
5. **Never commit to `master`** in `~/llama.cpp`.
6. Judge a rung by its resolutions + build + coherence + its own mechanism oracle, **not** by line count and
   **not** by r277-identical text (upstream moved numerics on purpose in this range).
7. **A gate arm can belong to a later block.**  Before concluding that a rung broke something, `grep -l` the
   *symbol or the message* in `patches/*.patch`: three items have already been cleared that way
   (`tsplit` -> R13, `node-budget` -> R13, sparse QSA -> R16).  Record the block and the symbol, never weaken
   the gate and never simplify it to the arm that passes (§8 trap 11).

---

## §3  Rung mechanics (exact commands)

```bash
cd ~/llama.cpp
git cherry-pick <block sha>                 # conflict -> resolve per plan
#   resolve; then:
git add -A                                  # ALWAYS -- see §8 trap 3
GIT_EDITOR=true git cherry-pick --continue
timeout 3000 ~/bin/build-llama-rocm         # ~20-30 s incremental warm; the FA quants cache is already set
#   first-error pass is NORMAL: fix, rebuild, then
git add -A && GIT_EDITOR=true git commit --amend --no-edit
```

**If the block adds or removes files under `ggml/src/ggml-cuda/template-instances/`** (or any bare
`file(GLOB …)` list), `touch ggml/src/ggml-cuda/CMakeLists.txt` **before** building: those globs are not
`CONFIGURE_DEPENDS`, so new files are otherwise silently not compiled (this bit at R09r).

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
| host-expert arena | `HIP_VISIBLE_DEVICES=0 … 35B-A3B Q8_0 -ngl 99 -sm layer -ncmoe 20 -c 8192 -lv 4` | block 09; grep `MoE expert cache:`, `expert access:`, `takeover=` |
| op oracles | `./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT\|MUL_MAT\|MUL_MAT_ID` | blocks 05/07/00 (6995-6996/0, 2681/0, 1056/0 at their rungs) |

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
| R06 | 05 flash attention | DONE | `22fcde21e` | `D5` FA half complementary; FA oracle 6996/0; union instance matrix |
| R07 | 06 MMB prefill | DONE | `c1d8b7c04` | our blocks on upstream's MMVF/fusion predicates; MMB census 6915 t/s |
| R08 | 07 k-quant boosts | DONE | `626e398d2` | graph-reuse union; `MUL_MAT` 2681/0 (`D16`) |
| R09 | 08 fused kernels | DONE | `a95c0497b` | 7 files / 14 hunks; **`D12` dual store kept**, `D6` partial then; **found `D5-a`** |
| **R09r** | **rung-repair, block 05's artifact** | **DONE** | `9c94ce139` | `D5-a`: 32 mma instance files (heads 40/72) + 14 FP8 KV instances + a **typed** `generate_cu_files.py`; undefined FA symbols 32 -> 0, FA oracle 6995/0, generator idempotent.  **Squashed into block 05 at R18** |
| R10 | 09 host-resident MoE | DONE | `a6809f75b` | 8 files / 14 hunks, all "both sides"; `D1` answered (no port needed), arena live (`fills=1991`), pool text identity `a99ccbd97ecb`; `tsplit*` arms are **R13's** |
| R11 | 10 infrastructure (slab, fusion dispatcher, async ring) | DONE | `13176810a` | 9 files / 20 hunks; the **whole fusion dispatcher moved into `ggml-cuda-try-fuse.cu`**; `D6`/`D10` closed; `gate-norm-rows-alias.sh` **PASS**; `node-budget` is **R13's** |
| R12 | 11 qwen4exp | DONE | `1d83b50e0` | 5 files / 6 hunks — the cleanest rung; `D11` decided (`GGML_OP_COUNT` **105**, RPC 7); `gate-hc-fusions.sh` **PASS** (`d62340fbf566`), quant-coherence A-D **PASS**; its section E (sparse QSA) is **R16's** |
| R13 | 12 campaign remainder (CPU wins, backend iface fills, DFlash, FA build grouping, server prompt cache) | **DONE** | `75d83f231` | 5 files / 8 hunks, **zero** first-error passes; `D11` op count **106** / RPC **8** / 24 messages paired; **FA generator = the typed set with NO head-40/72 skips** and idempotent; `gate-node-budget.sh` **PASS 4/4, margin 440 entries every arm**; `gate-host-pool.sh` **42/54 with `tsplit`+`tsplitnosink` CLEARED** — the 12 remaining checks are **R16's** (block 15 `host_expert_buft` -> `ROCm_HostMap`) and **R17's** (block 16 placement queue) |
| R14 | 13 MTP vocab subset | **DONE** | `aca78625a` | **0 conflicts, 0 first-error passes**; the block's own `test-mtp-draft-vocab` **PASS** (subset = -inf outside `N`, finite logits == full vocab, 0 mismatches, two live contexts independent); `--spec-draft-mtp-vocab 32768` drafts on 32 801/248 320 rows; **`D7` re-verified and it moved** — `-sm tensor` now takes the device row gather too (block 09's Meta `event_new`) |
| R15 | 14 relaxed-tensor (**TEMPORARY**) | **DONE** | `f040d394f` | **0 conflicts, 0 first-error passes**; `-sm tensor` refuses flint with the delivery's pointer and `-sm relaxed-tensor` == `-sm layer` byte-identically (`cc4fbc3e8367`; 4B tensor/relaxed/layer all `1c5d32ac537d`); **the drop condition is UNMET** — upstream's deny is verbatim at `4f9dfa0e3` and nothing in `a691b2345..4f9dfa0e3` touches `llm_arch_supports_sm_tensor`, so the block stays TEMPORARY (§5.8) |
| R16 | 15 streamed KV | TODO | — | 5 conflicts; `D3`, `D19`, `D15`'s loader question; owes the KV residency gates, `gate-qwen4exp-quant-coherence.sh` **section E** (the sparse QSA arm — block 15 replaces `ggml_rope_get_truncate(dst)` on the fused `INDEXER_SCORE` node, which upstream `#30206`'s assert rejects) **and `gate-host-pool.sh`'s `fail`/`late`/`pinlate` arms** (block 15 is what makes the host master pageable — see §5.7) |
| R17 | 16 UMA GTT store | TODO | — | **0 conflicts**; verify `D17` (VMM region size) with `gate-uma-experts.sh --quick`; owes **`gate-host-pool.sh`'s placement-transport checks** (`ref` `PLACEMENT reads: O_DIRECT` + placement queue, `qd1`, `bulkbuf` — block 16 ships `MOE_EXPERT_FILL_QD`, `MOE_HOST_POOL_BULK_DIO` and the read queue; §5.7) |
| R18 | — | TODO | — | release ladder, `README.md` §4 + §10 below |

**Remaining block commits to cherry-pick** (from `release.json` order; a later block is judged against the
rebased earlier ones, so the order is not negotiable):

```
R16 99d33117a   R17 9d7c0502c
```

Per-rung expected conflict paths and "what to expect" notes: `CONFLICT-MAP.md` (measured, generated from the
*old* base so treat its counts as an upper bound) and `RUNGS.md` (per-rung sections + resolution logs).

---

## §5  What was decided so far (so it is not re-litigated)

### §5.1 `D17`/B1 — the base blocker (DECIDED: option 1)

Upstream `ac931b5bc` (`#30542`) hangs the HIP VMM pool on ROCm 10.1/gfx1201: the **unpatched** base never
returns from its first GPU compute with the delivery's required `GGML_HIP_NO_VMM=OFF`.  Block 00 therefore
carries a **full revert of that commit** (maintainer's call).  Drop the revert at the next re-base once
upstream fixes the pool.  Do not "try VMM on" to work around anything else — the arena/GTT store need VMM.
Verified working at R10: the slab and the per-device arenas allocate on the VMM pool with the revert in place.

### §5.2 `D2` — chunked GDN (DECIDED at R04: complementary, nothing dropped)
Upstream `#32078` deleted **its own** non-fused GDN graph builders, the GDN fused-node tags and their
registry entries.  The delivery's substance is the `n_rs_batch` argument on `ggml_gated_delta_net()`, the
pre-batch snapshot copy and the `gated_delta_net_chunked*.cu` kernels — and since upstream now always routes
through that same fused op, the delivery's kernels are the surviving implementation.  Consequence for later
rungs: **`LLM_FUSED_OP_GDN_AR/CH` do not exist upstream any more**; do not re-add them, and their
`fused_gdn_ar/ch`/`auto_fgdn` cparams were removed at R06.

### §5.3 `D5` — flash attention (DECIDED at R06: complementary) + `D5-a` (DECIDED and APPLIED at R09r)
The FA band is a *selector* over the same kernel matrix, so upstream's FP8 vec FA, heads 40/72 and the sm70
case sit beside it.  Two things matter later:
* upstream added **`DKQp`/`DVp = GGML_PAD(head, 16)`** for head 72 — the delivery's inherited loops, Q-tile
  stride and `VKQ_C` sizing need it; if it ever looks "unused", it is not.
* the instance matrix is the delivery's **typed** `...-dkqN-dvN.cu` files, one declaration per file, one file
  per (head, ncols1, ncols2).  Upstream switched its own matrix to head-agnostic files **and folded its two new
  heads 40/72 into that switch**, which is what `D5-a` was: 32 undefined
  `ggml_cuda_flash_attn_ext_mma_f16_case<40|72,…>` symbols against an unconditional dispatch (a latent abort on
  AMD, not NV-only).  The repair `9c94ce139` added those 32 files, the 14 FP8 KV instances the same switch hid,
  and restored a **typed** `generate_cu_files.py` (as merged it would `os.remove` the delivery's 126 typed
  files on a re-run).  **Invariant to preserve:** running `generate_cu_files.py` in
  `ggml/src/ggml-cuda/template-instances/` must leave `git status` clean.  Oracle: `test-backend-ops -o
  FLASH_ATTN_EXT` (6995/0 on the R09r tree, 448 head-40 + 1009 head-72 cases exercised).
* `D5`'s **few-row half is still an R18 measurement**: at R11 the delivery's RDNA4/RDNA3_5 "dense band
  promotion" was placed *after* upstream's `ggml_cuda_should_use_mmq_few()` gate, so it only decides shapes
  few-row MMQ declines.  The delivery-vs-upstream ordering is a `llama-bench` arm at R18.

### §5.4 `D6`/`D10` — CLOSED at R11 (the delivery state follows the move; the scheduler is untouched)
`graph.cuh` owns `ggml_cuda_graph` / `ggml_cuda_stream_context` / `ggml_cuda_graph_instance`;
`ggml_backend_cuda_context` stayed in `common.cuh`.  Block 10's additions were transplanted into `graph.cuh` —
`capture_deferred` and the **packed** per-node replay guard (`src_properties` / `prop_nodes` / `prop_src_off` /
`prop_srcs`, 328 B/node against upstream's 1048), which *replaces* upstream's `node_properties`/`node_props` and
removes its last user.  Upstream's `ggml_cuda_graph_instance` + cluster snapshot is kept, so `#31284`'s
cluster-dim re-instantiation still fires in `ggml_cuda_graph_update_executable()` (upstream's signature, with
the delivery's HIP `hipGraphExecUpdate` leak workaround as its first body block).  The keyed cache
(`ggml_cuda_graph_key`) and its eviction sweep are unchanged, and the accessor is **out-of-line in
`ggml-cuda.cu`** (the body needs the complete struct — an inline body in `common.cuh` does not compile).
Block 10's async input ring landed as a context-owned host-pinned **splice ring** feeding
`ggml_backend_cuda_set_tensor_2d_async()`, i.e. *below* the scheduler, so upstream's upload reachability
(`#31419`, which the delivery's 1991 host-expert fills prove does not starve the staging), prefetch ordering
(`#32100`) and graph-result reuse (`#32089`) are untouched.

### §5.5 `D1` — the MoE cache (ANSWERED, no port needed) and the `-sm tensor` arena fault
Upstream `#32157` moved **its own** MoE cache from `llama-graph.cpp` to `src/llama-moe-cache.{cpp,h}`
(`build_lookup()` / `get_experts()` / `copy()`).  The delivery's tier is a **different implementation in a
different place** — the CUDA-side arena (`moe-expert-cache.{cu,h}`), the movable-boundary slab,
`llama_context::sched_copy_experts` and the device rearm/reserve hooks — and block 09 touches no file of
upstream's cache.  The r236 **either/or ruling stands**: arena default-on, upstream's cache shipped-but-dormant
behind `--moe-cache-mib`.  The r143 measurement in block 09's own commit message (35B-A3B Q8_0, `-ncmoe 20`,
`-c 8192`, 16.3 GiB host experts, 1 GPU, equal 4080 MiB budgets) is why: arena 547.6 prefill / 53.5 decode t/s
vs upstream cache 544.2 / **62.9** and streaming 470 / 49, text byte-identical.

**The `-sm tensor` arena fault was NOT a merge defect** (R10 was parked on it and the `D10` attribution was
wrong).  The Meta backend at R10/R11 has **no `moe_cache_pool_backed` forwarding at all**, and **block 12 adds
it** ("forward the pool-backed query to the simple devices.  A tensor split gives every simple device the SAME
host master").  Without it a Meta-wrapped host expert weight is not reported as pool-backed, so
`mul_mat_vec_q_moe` reads the **pageable** host master -> `Page not present` on both GPUs right after slab init.
r277 passes that arm only because block 12 is in it.  **`gate-host-pool.sh`'s `tsplit` + `tsplitnosink` are
therefore R13's** and must go green there; at R10 the gate reaches 11 of its 13 arms and cannot print a verdict,
so the substance was verified directly (four identical 182-char `sha=a99ccbd97ecb` arms + the 1-device census
`h=0.5205`, fills=1991, takeover=540).

### §5.6 R11/R12 — the two facts a later rung needs
* **Block 10 moves the whole fusion dispatcher** out of `ggml-cuda.cu` into `ggml-cuda-try-fuse.cu`.  R11
  accepted the move *after* diffing the arm set: everything the old dispatcher had is in the new TU except four
  names, and all four are still defined in their owning TUs (their dispatch was refactored into `_try_fuse()`
  arms).  **If you ever again accept a wholesale move, do that diff** — the moved copy is written against the
  old base and carries stale API usage (§8 trap 10).
* **`D11`'s op count is the merged value, never either side's literal one.**  `GGML_OP_COUNT` is **105** =
  R11's 99 + block 11's 6 (`FLASH_ATTN_QSA`, `INDEXER_TOPK`, `INDEXER_SCORE`, `INDEXER_FILL`, `HC_MIX`,
  `HC_COMBINE`); block 11 asserted 108 because *its* tree (old base + blocks 00-11) holds 3 more ops than the
  new base (upstream's `29dbbad2b` prune removed them).  `RPC_PROTO_PATCH_VERSION` stays **7** (the fork's table
  diverges from upstream's 0).  R13 answered that: block 12 adds one op (`GGML_OP_ALL_GATHER`), the count is
  **106**, `RPC_PROTO_PATCH_VERSION` **8**, and **all 24 `RPC_CMD` messages still have a server case and a
  client send site** (block 12 adds no message — its `ggml-rpc.cpp` hunk is three appended `buffer_type_i`
  fields set to `NULL`).

### §5.7 R13 — the pool gate's demand path is a LATER block's, not this rung's

R13 cleared the two arms it owed (`gate-host-pool.sh` `tsplit` + `tsplitnosink`: pool built on a 2-device
`-sm tensor` split, exchange INERT, both arms byte-identical at `171 chars sha=8c39335bbceb`) and
`gate-node-budget.sh` (4/4, margin 440 entries in every arm — see `RUNGS.md` §R13 for the table).  The gate is
**42/54** and the remaining 12 checks are **two later blocks' mechanisms**, measured and named:

* **`fail` / `late` / `pinlate` (8 checks) -> R16 (block 15).**  `llama_model_loader::host_expert_buft()`
  currently ends in `(void) want_mapping; return ggml_backend_dev_host_buffer_type(dev);`, i.e. the host master
  is always the **PINNED** `ROCm_Host` buffer; block 15 inserts `if (want_mapping && use_mmap && !pin_master)
  return ggml_backend_dev_moe_host_master_buft(dev);` (**`ROCm_HostMap`**, the pageable pool-backed master).
  Until then `bind_host_dev_locked()`'s `cudaHostGetDevicePointer()` **succeeds**, the table takes the in-place
  cold read, and `expert_host_src()` never reaches `pool_fill_fault()` or the bounce-grow latch — so the fault
  arms are **vacuous** (their text still equals `ref`, which is why R10's text-only check read green: **that
  part of the R10 record was over-optimistic**).  Evidence: our log has no
  `bind_host_dev_locked: the host expert master is pageable`, `tensor … overridden to ROCm_Host` and pool
  `h=0.0000 (0/0)`; `bin-r277` (block 15 in) prints the pageable line, `overridden to ROCm_HostMap`, pool
  `h=0.9559 (40837/42719)`, `FAILED fills` 142x and `takeover=0`.
* **`ref` placement transport, `qd1`, `bulkbuf` (4 checks) -> R17 (block 16).**  `MOE_EXPERT_FILL_QD`,
  `MOE_HOST_POOL_BULK_DIO` and the placement read queue (`host pool placement queue: N batch(es) … N reader(s)`)
  are all block 16's (`grep -l` -> `patches/0016-*`); this rung's pool prewarms serially out of the mapping.

**Do not weaken the gate and do not drop an arm**: both rungs must re-run it and expect 54/54.

### §5.8 R15 — block 14 (`-sm relaxed-tensor`) is still load-bearing, and it stays TEMPORARY

Block 14 is the delivery's local relaxation of upstream `#31081` (the MoE amax whole-row rescale cannot be
completed under a tensor split, so upstream denies `-sm tensor` for `flint`/`mellum`/`mistral4`).  R15's own
check, since "expect to drop it eventually":

* the deny is **verbatim** at the pristine base `4f9dfa0e3` and the three arches are still in
  `llm_arch_supports_sm_tensor()`'s false list;
* **no commit in `a691b2345..4f9dfa0e3` touches that function** (`git log -S'llm_arch_supports_sm_tensor'` on the
  range is empty) and the only range commit touching `ggml-backend-meta.cpp` is the `29dbbad2b` prune;
* on this tree `-sm tensor` still refuses flint with `… needs '-sm relaxed-tensor'`, `-sm relaxed-tensor` ==
  `-sm layer` byte-identically (`103 chars sha=cc4fbc3e8367`, 3 GPU), and a non-amax arch is byte-identical
  across all three modes (`1c5d32ac537d`).

So the drop condition (upstream lands a real fix) is **unmet**.  Dropping block 14 is a maintainer decision for a
future base, never a merge outcome; until then it is `patches/README.md`'s TEMPORARY item and its own gate is the
Flint/`-sm layer` text identity above.

---

## §6  Decision register status (do not re-decide by accident)

| item | status |
|---|---|
| `D1` MoE cache relocation / CPU share | **ANSWERED at R10** — no port needed; the r236 either/or ruling stands (arena default-on, upstream cache dormant) |
| `D2` GDN pruning | **DECIDED** complementary (R04) |
| `D3` FP8/BF16 KV types | **CLOSED** (R05 audit + R06 FA oracle + R09r FP8 KV instances) |
| `D4` MMVF BF16 few-row | **DEFERRED to R18** (no rung conflict; a `llama-bench` arm) |
| `D5` FA + few-row MMQ | **DECIDED** FA half (R06); **`D5-a` DECIDED and APPLIED** (`9c94ce139`); few-row **ordering** is an R18 `llama-bench` arm |
| `D6` graph/stream state moved to `graph.cuh` | **CLOSED at R11** — packed replay guard + cluster check kept, accessor out-of-line (see §5.4) |
| `D7` MTP device-buffer hidden state | **RE-VERIFIED at R14, and the R03 asymmetry is GONE**: block 09's `ggml_backend_meta_device_event_new()` (R10) gives the Meta device events, so `-sm tensor` takes the device row gather too (`target h rows on the device: yes` on both splits, 11/11 acceptance each).  R12's `build_inp_embd_state` still carries the host fallback.  The `upstream/` candidate (events for upstream's Meta device) is now purely upstream's gap.  R18 owes the four-axis measurement. |
| `D8` the prune | **MECHANICAL, and it has bitten three times** — it removes *includes* as well as declarations (`common/speculative.cpp` at R03, `diagmask.cuh` itself at R11).  See §8 trap 4 |
| `D9` mmap / Win-load / lazy-mode | **PARTIAL** — R01 resolved the only conflict, R10 took upstream's Windows rewrite wholesale (the delivery's `fname`/`path()` and `llama_mmap::impl` work survive); the substantive questions are R16/R17's |
| `D10` scheduler upload/prefetch/graph-result | **CLOSED at R13** — union validated by the 1991 host-expert fills (R11); the async ring sits below the scheduler; the `tsplit*` arms are **green** (R13: pool built on the split, exchange inert, `8c39335bbceb`) |
| `D11` `ggml.c`/RPC/test refactors | **CLOSED at R13** — op count **106**, RPC **8**, 24 messages all paired (§5.6); the merged value, never either side's literal |
| `D12` BF16 RMS_NORM | **DECIDED at R09** — dual store kept (both destinations ship); the rows gate passed at R11, the bf16 side channel's own gate is R16 |
| `D13` HIP BLAS workspace | **CHECKED** (R02) |
| `D14` GDN/SSM kernels | **CHECKED** (R04) |
| `D15` external PLE | **merge done at R10** (`path_model` -> `paths_model` + the delivery's host-expert out-params); the `madvise` loop is block 11's (R12), NOT block 12's; the PLE-residency question is R16's |
| `D16` MUL_MAT src1 precision | **CLOSED** (R08, 2681/0) |
| `D17` VMM pool | **DECIDED** (option 1); partially verified at R10 (slab/arena allocate with the revert in place), R17 verifies the GTT region |
| `D18` SYCL device-side MoE | **CHECKED at R10** — no SYCL file in the conflict set, no shared op flag changed |
| `D19` plamo2 recurrent cache names | **OPEN (verify only)** — R16 |

---

## §7  Environment, models and gotchas

* **Build**: `cd ~/llama.cpp && ~/bin/build-llama-rocm` (ROCm 10.1.0, `gfx1201`, `GGML_HIP_NO_VMM=OFF`,
  ccache ~30 GiB).  `clean` wipes `build-rocm` (only needed if the cache is inconsistent, ~6 min cold).
  Never pass `EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=…"` (the script's include-order guard aborts).  The
  `GGML_CUDA_FA_QUANTS` cache already carries the union list (incl. `iq4_nl` and `f8_e4m3`) — do not hand-set it.
* **The build dir tracks the branch**: after any rollback (`git reset --hard`) **rebuild before measuring**, or
  you are testing the discarded tree's binaries (this happened once at R10).
* **`build-rocm-base2` is unpatched `4f9dfa0e3` with `GGML_HIP_NO_VMM=ON`** — the only unpatched base
  configuration that runs on this box (see `BASE-BLOCKER.md`).  It is the reference for "is this upstream or is
  this us?" questions, and R18 needs it for the prefill-logit base record (with the asymmetry noted).
* **`build-rocm/bin-r277/llama-cli` is the full 17-block delivery.**  It is the fastest discriminator between
  "this block broke it" and "a later block fixes it" — and it is how the three deferred gate arms were each
  settled.
* **Models** (`/llm/models/`):
  * `Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` — the coherence oracle (also the hybrid GDN / SILU-gated norm model).
  * `Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` — the field host-expert/MTP config; `runme-rocm` beside it
    carries the reference server args (2-device `-sm tensor`, bf16 KV, `draft-mtp`).
  * `Qwen3.8/Flash-Next/{GSQ-IQ3_XXS,GSQ-RCO-IQ3_S,IQ3_XXS,IQ4_NL,Q4_K_M,Q4_K_XL}` — the qwen4exp class used by
    `gate-hc-fusions.sh` (it picks GSQ-IQ3_XXS) and `gate-qwen4exp-quant-coherence.sh`.  **There is no `IQ4_XS`
    directory**, so that gate's IQ4_XS arm SKIPs — that is expected, not a missing gate.
  * `Qwen3.8/27B/{BF16,FP8,IQ4_NL,Q5_K,Q6_K,Q8_0,…}` — the dense 27B (not the qwen4exp one).
  * `Ling/3.0-tiny/IQ4_NL/` — `gate-norm-rows-alias.sh`'s bailingmoe3 model (gated per-head norm).
  * `prompts/prose-rdna-boosts.txt` — 5 228-5 298 tokens, sha256 in `prompts/README.md`; above the ~2051-token
    QSA sparse threshold.
* **Pin `HIP_VISIBLE_DEVICES`** on every arm (the 9950X3D's iGPU is enumerated and the `-sm tensor` load
  aborts with `invalid kernel file`).  `0` = one R9700, `0,2` = the x8 pair (the field config), `0,1,2` = all
  three for the all-reduce arm.
* **`timeout -s KILL`** on anything that could hang: a spinning `llama-cli` ignores SIGTERM.
* The `llama-cli --version` string is baked **before** the commit, so a freshly committed rung still reports
  the previous sha.  Do not read it as "the build did not pick up the change".
* Gate runtimes (this box): `gate-hc-fusions.sh` ~6 arms × ~1 min; `gate-qwen4exp-quant-coherence.sh` ~10 min;
  `gate-host-pool.sh` 13 arms ~8 min; `gate-norm-rows-alias.sh` ~8 min; `gate-node-budget.sh` ~2 min.  Never
  run two gates (or a gate and a bench) in parallel.

---

## §8  Traps that have already bitten (in this campaign)

1. **Never commit to `master`**; the fork `master` is the pristine base snapshot.
2. **One long-lived source path** (`~/llama.cpp`): ccache keys on the absolute path; a new worktree is a cold
   build.
3. **`git add -A` after resolving.**  `git diff --diff-filter=U` reads the *index*, so a resolved-but-unstaged
   file still looks unmerged.  And for **UD** (deleted-by-them) files, `git rm --cached` leaves the file on
   disk — it must be deleted (or `git rm -f` before staging) or `git add -A` will resurrect it.
4. **A prune removes *includes*, and that is a build break, not a conflict** (`D8`): the block's include hunk
   attaches cleanly and the symbol then resolves nowhere.  Hit at R03 (`llama-ext.h` from
   `common/speculative.cpp`) and at R11 (`ggml-cuda/diagmask.cuh` was deleted by `29dbbad2b` itself, and block
   10's new TU still included it).  After every prune rung, check the include surface of the delivery's symbols
   — and remember a *new* file can carry a stale include.
5. **Upstream refactored the APIs the delivery's blocks call.**  Re-expressed so far: `graph_max_nodes()` (no
   `n_tokens`), `sched_new({.max_nodes, .parallel, .op_offload})` (`#32009`), `graph_reserve(...,
   packed_kq_mask)`, `ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)` (tensor form, `#29633`/`#31959` —
   six call sites needed it at R11), `should_fuse_mul_mat_vec_f(up, gate)`, `ggml_cuda_should_use_mmq_few()`,
   `quantize_row_q8_1_cuda` (11 args), `common_model_load(paths, mparams)`, `common_get_device_memory_data_impl(
   paths_model, …, host_expert_bytes, host_experts)`.
6. **`git grep '<<<<<<<'` matches binary GGUFs** — use `git grep -I`.  It also misses **stray end-markers**:
   check `git grep -In '^<<<<<<< \|^=======$\|^>>>>>>> '` — a resolution can leave a bare `>>>>>>>` with no
   opening marker (this happened twice, in `mmvf.cu` and `mmvq.cu`).
7. **A rung build needs first-error passes** — R06 one, R09 one, R10 three, R11 **six**, R12 one.  Compile, read
   the errors, fix, rebuild; never assume the merge is done because the text looks right.
8. **Do not leave a half-resolved rung.**  If the session runs out of room, `git cherry-pick --abort`, record the
   analysis in `RUNGS.md`, and hand over (R09's pre-analysis was written that way and made R09 fast).
9. **`test-recurrent-state-depth` is red on this box on the shipped r277 build too** (73-83 Phase-A mismatches
   depending on config) — pre-existing and config-sensitive, not a rung regression.  Run it with the 4B and
   `-ngl 99`; record the count rather than treating it as a gate.
10. **A wholesale *move* carries a stale copy of upstream-changed code.**  When a block extracts code into a new
    file (block 10's `ggml-cuda-try-fuse.cu`, `mmvq-ksplit-body.inc`), that copy was written against the *old*
    base: it re-adds pruned includes and uses pre-refactor signatures.  Accept the move only after (a) diffing
    the symbol/arm set against the outgoing file, and (b) grepping the moved region for APIs upstream changed in
    the range (`git log --oneline <old-base>..<new-base> -- <file>` is cheap and decisive).
11. **A gate arm can belong to a later block.**  Before calling a rung broken, `grep -l '<the symbol or the
    message>' patches/*.patch` and try `bin-r277`.  Cleared this way: `tsplit`/`tsplitnosink` -> R13
    (`moe_cache_pool_backed`), `gate-node-budget.sh` -> R13 (`graph_max_nodes` + the refusal), sparse QSA ->
    R16 (`#30206`'s `ggml_rope_get_truncate` assert + block 15's fix).  Record the block and the symbol; never
    weaken the gate.
12. **Unions can leave duplicated *continuation* lines.**  When both sides of a conflict end with the same
    half-finished construct (`for (…) {` then an emplace; `sched_new(...) {` before the shared body;
    `mctx_iswa`/`mctx_cur` declarations), concatenating them duplicates it and the error appears far away
    ("function definition is not allowed here").  Read the *seam* after every union, not just the markers.

---

## §9  Observations parked for R18 (not blockers)

* **Prefill-logit base record asymmetry.**  Unpatched `4f9dfa0e3` cannot run with `VMM=OFF` (§7), so the base
  record comes from `build-rocm-base2` (unpatched, `VMM=ON`) and the candidate from the canonical `VMM=OFF`
  build.  The pool affects allocation only, but the asymmetry must be stated in the gate record.
* **The gate arms cleared mid-climb must be re-run at R18 even though their rung passed them**:
  `gate-host-pool.sh` `tsplit`/`tsplitnosink` (cleared at R13), `gate-node-budget.sh` (**the margin is now on
  record**: 2064/2504 for the q8_0 arms and 1504/1944 for f16/no-rotation, i.e. 440 entries = the model's
  pre-rotation headroom, because `memory->graph_extra_entries()` models the rotation exactly), and
  `gate-qwen4exp-quant-coherence.sh` section E (R16).
* **Still deferred at R13, to be cleared at their own rungs and then re-run at R18:** `gate-host-pool.sh`'s
  `fail`/`late`/`pinlate` (8 checks -> **R16**, block 15's pageable `ROCm_HostMap` master) and its
  placement-transport checks (`ref` `PLACEMENT reads: O_DIRECT` + the read queue, `qd1`, `bulkbuf` — 4 checks
  -> **R17**, block 16).  §5.7 has the symbols.  Both rungs must reach **54/54**.
* **`GGML_OP_COUNT` 106 / `RPC_PROTO_PATCH_VERSION` 8** are fork-specific: revisit if upstream renumbers, and
  re-derive the count from the merged enum rather than trusting either side's literal.
* **`generate_cu_files.py` must stay idempotent** on the tree (a second run leaves `git status` clean).  If a
  later block regenerates the instance matrix, re-check this — it is what `R09r` exists for.
* **`D7`'s Meta-device event gap** is a genuine upstream hole worth an `upstream/` candidate (an `event_new` for
  `ggml-backend-meta`, or falling back to the output device's backend).
* **The r277 `test-recurrent-state-depth` mismatch** (§8 trap 9) is worth a `TODO` once R18 re-runs it.
* `patches/README.md` / `ENVIRONMENT.md` rows that will need updating after R18: the `LLAMA_KQ_MASK_*` /
  derived-mask block, the MMB census, the FA instance-matrix description (typed set, head 40/72, `f8_e4m3`), the
  `graph_max_nodes` note, the RPC patch version.

---

## §10  R18 — the release ladder, with this campaign's additions

`README.md` §4 has the full list.  The additions discovered during the climb:

1. `scripts/make-patches.sh` from the finished branch (17 files, block 00 keeps `0000`) -> `make-release.sh`.
   **The `R09r` repair commit is squashed into block 05 here** (the series must be exactly 17 block commits).
2. `scripts/validate-set.sh` — strict apply on a fresh tarball at `4f9dfa0e3`, tree equality, sha256s, Windows
   path/portability lints.
3. `scripts/gate-per-block-build.sh` — a clean build per block commit (17), plus the re-cut block 05.
4. Coherence: dense 4B, the 35B-A3B field/host-expert config, a hybrid recurrent model, and the qwen4exp class.
5. `scripts/gate-prefill-logits.sh --record` from the **unpatched base** (`build-rocm-base2`, `VMM=ON`), then the
   candidate; plus the `--sm tensor --device 0,1,2` arm (block 01), a `--kv q8_0` arm, and an FP8 KV arm.
6. `scripts/gate-rebase-release.sh` (the one-command pre-release verdict) and `check-gpu-faults.sh`.
7. Every block gate whose code moved, **including the three cleared mid-climb**: host-pool (all 13 arms),
   node-budget (**with the margin**), qwen4exp-quant-coherence (section E), kv-residency, kv-window-split,
   park-idle-slots, ckpt-reuse, state-pipe, spec-align, spec-width-band, spec-draft-window, spec-reuse-window,
   hc-fusions, norm-rows-alias, grid-stall-pad, uma-experts, fusion-alias-copy, plus the FA / MUL_MAT /
   MUL_MAT_ID op oracles.
8. MTP four-axis + `llama-batched-bench -npl 1,4,8`; the `D4`/`D5` few-row `llama-bench` arms (this is where
   `D5`'s promotion-vs-`mmq_few` ordering is finally measured).
9. **Records**: a `WORKLOG.md` entry (new base/tip/tree + gate numbers), `patches/README.md` per-block notes,
   `BASELINE.md` current baseline, `INVARIANTS.md` appended facts, `ENVIRONMENT.md` for every variable that
   appeared/vanished, `TODO.md` for the deferred decision items, and this campaign moved to
   `archive/work/upstream-rebase-261113/` (no `wip/` stub).
10. Fork refresh (`scripts/fork-refresh.sh --build --gates --push`, personal fork only) + the off-host
    container images — with the maintainer's go-ahead, never autonomously.

---

## §11  Session-close protocol

Before stopping, always:

1. Finish or **abort** the rung (no dirty tree, no `CHERRY_PICK_HEAD`; `git add -A` before `--continue`).
2. Update `RUNGS.md`: the ledger row (status/sha), the rung's result + resolution log, **the gate arms deferred
   to a later block (name the block and the symbol)**, and the pre-analysis of the *next* rung if it was
   reached and not done.
3. Update `DECISIONS.md` for every item the rung touched (status vocabulary: OPEN / MEASURING / AWAITING /
   DECIDED / CLOSED; an item must not stay `AWAITING` into R18).
4. Update this file's §0 table (progress, HEAD sha, next rung, build dir), and add a dated `WORKLOG.md` entry
   (newest first; the last ones are `261113.005`..`261113.009`).
5. Commit the delivery repo on `rebase-261113` (the fork branch is committed by the rung itself).
6. Leave `PROMPT.md` pointing at the *next* rung, and re-verify the coherence oracle one last time so the next
   session starts from a measured state.
