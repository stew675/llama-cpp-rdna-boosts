# HANDOVER — block 17: `TODO` #365 (the meta-aware device copy) and the delivery wrap-up

> **CONSUMED 2026-11-22 (r347).**  Everything in §3, §4 and §5 is done: the fix set is re-validated in both
> configurations, `TODO` #365 is implemented (the Meta buffer's per-child `cpy_tensor_n`: 1.4-1.9 ms against 72-77 ms,
> byte-identical state and text), and the whole thing is folded into **block 17** on a canonical chain with the
> patches, `release.json` and both optional artifacts regenerated and `validate-set.sh` green.  The owed gate arm is
> `scripts/gate-arms-soak.sh boundary` (`MANIFESTS.md` gate 17b).  **Read `WORKLOG.md` 261122.036 instead of this
> file** — it is the record, including the two shapes that must not be used (a 4B over 3 devices; `TEMP` in the
> environment) and what is still owed (the maintainer's go-ahead, the tag, the fork refresh).  §6's landmines all
> still stand, and §7's open questions were answered as recorded there.

**Written 2026-11-22 (end of the session that fixed `#373`/`#366`). Read this top to bottom before touching anything: it
exists so you do not have to re-derive the state from a long transcript.**

**Status: `#373` (the P0) and `#366` are FIXED and VALIDATED on a scratch branch — NOT shipped. The next two jobs are
`#365`, then the block-17 amendment + release.** The maintainer has already decided the block question: **block 17 is the
target** (a same-block amendment, not a new block). Nothing has been generated, amended or tagged.

---

## 0. TL;DR

1. **The fix for `#373`/`#366` is on `~/llama.cpp` branch `crash373-fix`** — 4 commits on top of `ccfd7042f` (a WIP
   commit with temporary instrumentation that **must not travel**), also exported as
   `repro-373/FIX-candidate-373.patch` (4 commits, 489 insertions, 10 files). Five defects + one reserve derivation; all
   validated on the maintainer's own shape in **both** configurations.
2. **`#365` is the last open item of the block-17 queue** and it is independent of those fixes: the recurrent-state
   handover on a `-sm tensor` hybrid still goes through the host (**112.57 MiB, ~74.6 ms per request boundary**, in
   TTFT) because the Meta buffer type exposes **no `cpy_tensor`/`cpy_tensor_n`**, so
   `llama_memory_recurrent::can_take_batched()` says no and `llama_memory_seq_state_take()` takes its serialised
   fallback. The target is **< 5 ms** with every tensor byte-identical.
3. **Then the wrap-up**: fold the whole thing into block 17, re-cut the patch set from a canonical chain, run the gates,
   record it, tag it, refresh the fork, let the worker push the images.
4. **The two gates to know about**: `scripts/gate-arms-soak.sh soak` passes 11/11; its **`pool` arm is VACUOUS** today
   (`TODO` #374 — pre-existing, fails with and without these changes, so do not "fix" it by relaxing the check).

---

## 1. Read these first, in this order

1. **This file.**
2. `WORKLOG.md` **261122.032 → .035** (newest first: .035 is the current state) — the diagnosis, the measurements, and
   what each fix is.
3. `wip/prefill-decode-arms-followups/README.md` **§W2** (`#365`'s design, "what exists / what is missing / acceptance"),
   **§W3** (`#366`, now DONE) and the **UPDATE 2026-11-22 (late)** block at the top of §W9's section.
4. `TODO.md` — **#373** (fixed, not shipped), **#374** (the vacuous gate arm), **#365** (next).
5. `patches/README.md` → **block 17**, and `INVARIANTS.md` **r346** (what the pair is, its six facts).
6. `AGENTS.md` — policy: gates, the pushing rules, **report findings before shipping**, and **the block structure is the
   maintainer's decision**.
7. `wip/slab-draft-aux-reserve/README.md` — read this **before** touching anything around
   `ggml_cuda_slab_set_aux_reserve()` (`#366`'s neighbour; the ordering hazard it records is still live).

---

## 2. Environment and repository state (verified at handover time)

| what | value |
|---|---|
| delivery repo | `~/llama-cpp-rdna-boosts`, branch `main`, tip **`b56c41b0`**, clean, **pushed** |
| release | `v18-919794500-r346`, `n_blocks 18`, `n_default 19`, `n_optional 2` (unchanged — this work amends block 17) |
| `release.json` | base `919794500`, **tip `8f88fc861`**, **tree `a6f877eabdf5b975ffbf2050a61f170c7e797566`** |
| fork checkout | `~/llama.cpp`, branch **`rdna-boosts`**, tip `8f88fc861`, **clean** (build-2ctx was rebuilt from it) |
| the FIX | `~/llama.cpp` branch **`crash373-fix`**: `00a568fba`, `7ea830922`, `eae8c79c0`, `431365ec5` on top of `ccfd7042f` (WIP, instrumentation — **do not** export it) |
| builds | `build-2ctx` (**ROCm 10.1**, the maintainer's `runme` binary; everything below was validated on it) and `build-rocm` (**10.2**, `~/bin/build-llama-rocm`, the delivery build — a release owes this line too) |
| model | `/llm/models/Qwen3.8/Flash-Next/STEW675-IQ4_NL/` (94 GB, 2 shards, embedded MTP4 head, `.moe-heatmap`), `runme` on port **8033** |
| harness | `wip/prefill-decode-arms-followups/repro-373/` (`armload.py`, `run.sh`, `README.md`) |
| evidence logs | `~/.cache/rdna-boosts/gates/crash373/` (`final-71k.log`, `A-hosttier*.log`, `d366*.log`, `R4096*.log`, `R5120.log`) |

**Get the fix and build it:**

```bash
cd ~/llama.cpp && git checkout crash373-fix
cmake --build build-2ctx --target llama-server llama-cli -j 16     # warm; ~1 min for the touched files
# the exported review artifact is already up to date; regenerate with:
git format-patch ccfd7042f..HEAD --stdout > ~/llama-cpp-rdna-boosts/wip/prefill-decode-arms-followups/repro-373/FIX-candidate-373.patch
```

**Put the checkout back before you finish** (`git checkout rdna-boosts` + rebuild `build-2ctx`) — the maintainer's
`runme` points at that binary, so leaving a fix build there without saying so is a trap for the next session.

---

## 3. What the fix set is (so you can review it, not re-derive it)

Five defects, one theme: **the pair runs two `llama_process()` calls on structures that were never shared before.** All
were found by driving the maintainer's shape, and each was reproduced before it was fixed. Details and numbers are in
`WORKLOG.md` 261122.032–.035; the short form:

| # | structure | symptom | fix | commit |
|---|---|---|---|---|
| 1 | **KV cell map** | `qwen4exp.cpp:1138` abort (children ~1078 cells apart) **and** a libstdc++ RB-tree SIGSEGV in `llama_kv_cache::prepare()`'s rollback snapshot | one **apply lock per memory GROUP** (`llama_memory_i::apply_mutex()`, shared through `create_memory()`), taken at `init_batch`/`apply`/`clear`/`state_*`/the `llama_memory_*` ops/park | `00a568fba` |
| 2 | **Meta split-state cache** | the *released* build's SIGSEGV in `ggml-backend-meta.cpp` — the cache was per **buffer**, so both arms corrupted one `unordered_map` | cache is per **VIEW SLOT** (`[4][8]`), and the slot is bound for **every** graph build/alloc (`meta_bind_view_slot()`) | `7ea830922` |
| 3 | **async-input ring** | `operation not permitted when stream is capturing` (its wrap path called `cudaDeviceSynchronize`) | wait on each **origin's own stream**; report "not staged" if refused | `7ea830922` |
| 4 | **VMM pool peer grant** | host tier: `HipVMM Failure: invalid argument` at ~2 s, pair-only | grant the owner alone then each peer alone, best effort; ranges in a registry and the predicate widened to **`ggml_cuda_peer_copy_ok()`** (slab **and** pool) so refused pairs stage | `eae8c79c0` |
| 5 | **synchronous host paths** | the same capture error from `set_tensor`/`get_tensor` (53 sites used the `cudaStreamPerThread` sentinel) | **`ggml_cuda_host_stream()`**: one private stream per (thread, device); the ring's origin shares it | `431365ec5` |
| — | **`#366`** | the pair's flat 3584 MiB slab reserve did not cover the prefill arm's **post-prefill re-reserve** | `common_arms_dec_reserve()` = flat floor + a **width-keyed tier** (0 at `-ub <= 1024`, 1536 MiB at `>= 2048`); env var still wins | `431365ec5` |

### Re-validate before you build anything on top (≈15 min, both configurations)

```bash
cd /home/stew675/.cache/rdna-boosts/gates/crash373
# 1. fully resident, 3 slots: 71k + 23k prompts + an 800-token generation, 2 rounds
LONG_TOK=71000 MED_TOK=23000 GEN_TOK=800 MED_DELAY=13 ROUNDS=2 ./run.sh reval-resident.log
# 2. host tier, the SAME workload (derived reserve, no knob -- must print aux=5.00 GiB)
MOE_EXPERT_CACHE_MIB=13653 EXTRA_ARGS="--n-cpu-moe 48 --ctx-size 131072" \
  LONG_TOK=71000 MED_TOK=23000 GEN_TOK=800 MED_DELAY=13 ROUNDS=2 ./run.sh reval-hosttier.log
grep -o 'aux=[0-9.]* GiB' reval-hosttier.log | head -1          # expect aux=5.00 GiB
# 3. the controls
EXTRA_ARGS="--no-prefill-decode-arms" LONG_TOK=20000 MED_TOK=6000 GEN_TOK=200 MED_DELAY=5 ROUNDS=1 ./run.sh reval-pairoff.log
cd ~/llama-cpp-rdna-boosts && LLAMA_SERVER=$HOME/llama.cpp/build-2ctx/bin/llama-server ./scripts/gate-arms-soak.sh soak --quick
```

**Pass criteria:** every request `ok`; `grep -c 'ROCm error'` = 0; `grep -c HipVMM` = 0; no `GGML_ASSERT`; the soak gate
**PASS 11/11** *including* "the pair's COLD answer equals the single-context server's" (that is the text oracle: if the
fix moved a number, this fails). The harness refuses to start if any `llama-server` is alive — kill the **process
group** of a leaked one (`run.sh` starts the server under `setsid` for exactly this).

---

## 4. NEXT TASK — `TODO` #365: a meta-aware device copy for the state handover

### The problem, exactly

At a request boundary the prefill arm hands one sequence's **recurrent state** to the decode arm.
`llama_memory_seq_state_take()` (`src/llama-context.cpp`, near the bottom) does:

```cpp
if (!force_serialised && dst_recr->can_take_batched(*src_recr)) {
    if (!dst_recr->seq_state_take(*src_recr, seq_id)) return false;   // device-to-device, 0.45-0.66 ms
    note_idx();  return true;
}
// SERIALISED FALLBACK: state_write() to a host blob, state_read() on the destination
//   -> 112.57 MiB through the host, ~74.6 ms per boundary, and it lands in TTFT
```

`can_take_batched()` (`src/llama-memory-recurrent.cpp:825`) requires, for every destination tensor whose buffer is not
both-hosts, `ggml_backend_buffer_can_copy_tensor(db)` — i.e. the buffer type must expose `cpy_tensor` or `cpy_tensor_n`.
The **Meta** buffer type has **both null** (`ggml/src/ggml-backend-meta.cpp`, the `ggml_backend_meta_buffer_type_i`
initialiser, `.cpy_tensor = nullptr, .cpy_tensor_n = NULL`), so on a `-sm tensor` hybrid the batched path is refused and
the fallback runs. `-sm layer` and 1 GPU are unaffected (the rows live on one device there).

The second half of the same paragraph in the code records a *second* reason it was never enabled: the Meta backend's host
path **cannot resolve a strided view of a split tensor** (measured as `GGML_ASSERT(buffer)` /
`SIGSEGV in ggml_backend_meta_get_tensor_impl`), because `llama_memory_recurrent` creates its row views **out of graph**
(`ggml_view_2d` + manually assigned `buffer`/`data`, see `view_of()` in `seq_state_take()`), so they are never
`init_tensor`ed and have **no `simple_tensor` registration** in the meta buffer's containers.

### What the state actually is

`seq_state_take()` drops the destination's cells for the seq, places them into one contiguous destination block, builds
**contiguous source runs** as 2-D row views (`[n_embd_r, n_rows]`, `nb[1]` stride) and issues **one**
`ggml_backend_tensor_copy_n(csrc, cdst, n)` for the whole set (up to 3 tensors per layer: `r_l`, `p_l`, `s_l`; 48 layers
on Flash-Next). So a Meta implementation only has to make **that one call** fast and correct; nothing about the rows
changes.

### The design to implement (recommended)

**Give the Meta buffer type `cpy_tensor_n` (and `cpy_tensor`), doing a per-child, split-aware row copy.** The shape is
already in the file twice — copy it, do not invent it:

* `ggml_backend_meta_kv_region_grow()` — a Meta-level hook that iterates the child buffers via
  `ggml_backend_meta_simple_dev_proc(child_dev, "<backend hook>")`.
* `ggml_backend_meta_get_tensor_impl()` / the `set_tensor_*` paths — they resolve a tensor's **split state**
  (`ggml_backend_meta_get_split_state()`: `axis`, `n_segments`, `ne[s*n_bufs + j]`, `nr[s]`) and slice the tensor's byte
  range **per child**. For a `-sm tensor` row tensor the split axis shards each row across children: child `j` gets
  `ne[seg][j]` contiguous bytes per row, at the tensor's own `nb[1]` stride. That is precisely the 2-D row copy the
  CUDA backend already implements (`ggml_backend_cuda_cpy_2d_rows` / the `cpy_2d` buft hooks) — so the per-child work can
  be delegated to the child backend's `cpy_tensor_n`/`cpy_tensor` (best) or issued as one 2-D copy per child.

**Preconditions to handle explicitly:**

1. **Resolve the tensor through `view_src`** when `tensor->buffer` is not a Meta buffer but the view chain root is
   (the `[TAG_META_UNALLOC_ALIAS]` rule already in the file) — and note `seq_state_take()`'s views carry `buffer`/`data`
   manually, so `tensor->buffer` **is** set, but they are **not registered** in any `simple_tensor` container: resolve
   the split from **geometry** (`get_split_state`), never from the container map.
2. Keep `can_take_batched()`'s contract: once `cpy_tensor_n` exists, `ggml_backend_buffer_can_copy_tensor()` returns true
   and the batched path is taken. Verify the *whole* set of tensors is covered before returning true, and prefer
   returning false (falling back) over a partially-copied state: **a half-moved state is fluent-but-wrong text**, the
   worst outcome this repo allows.
3. Do not change the *serialised* fallback: it is the A/B control (`LLAMA_RS_FORCE_SERIALISED=1`) and the oracle.

**Alternate/cheaper path (only if the row copy proves hard):** teach `ggml_backend_meta_get_tensor_impl` to serve a
strided view of a split tensor (a host readback), which makes the *existing* fallback correct rather than crashed. That
buys **nothing** on the timeline (it is still a host round trip) — it is `#365`'s correctness half, not its cost half.
Do not ship it as the answer.

### Acceptance (from the campaign §W2, and what to run)

* a `-sm tensor` hybrid boundary on the maintainer's shape moves the state in **well under 5 ms** (against **74.6 ms**),
  measured with the take's own timing (`took seq %d as a serialised state move (...)` is the fallback's line; add a
  temporary timing print for the batched path or use `LLAMA_RS_DEBUG_TAKE=1`, which dumps the per-layer tensor plan);
* **`LLAMA_RS_VERIFY_TAKE=1`** reports the destination identical to the serialised arm's result **byte for byte** (run the
  same turn twice: once with the new path, once with `LLAMA_RS_FORCE_SERIALISED=1`, and compare the VERIFY lines);
* **text identical** to the serialised arm, and to the single context on a class-matched shape;
* `scripts/gate-ckpt-reuse.sh` and the soak gate's text oracle still **PASS** (the reverse handover and the checkpoint
  restore use the same take);
* `scripts/validate-set.sh` stays green once the block is re-cut.

**Traps this item has already cost a session:**
`--kv-unified` + a tensor split means the recurrent rows are shared-cache-adjacent; the *destination* metadata decides
which rows matter and it is set by the same pass, so **never** "optimise" by copying only the rows the destination claims
before the pass; and the Meta path must not be reached via `st::simple_tensor` lookup (those views are unregistered).

---

## 5. Then: the DELIVERY wrap-up (block 17 amendment + release)

The maintainer's decision: **amend block 17** (no new block, no renumbering). Follow the documented flow, in this order —
`AGENTS.md` → *Regenerate the patches* is the contract:

1. **Fold the fix into block 17 on a CANONICAL chain** (never from the working branch): in a clean `~/llama.cpp`, check
   out `release.json.base` (`919794500`), run `scripts/apply-all.sh .`, then apply the fix diff and **amend it into the
   block-17 commit** (rebase/amend so the block-17 commit carries the change and no new commit appears). The set must
   still be 18 official blocks + `0300`; `n_blocks`/`n_default` do not change.
2. `scripts/make-patches.sh <fork> 919794500 <new blocks-tip>` → regenerated `patches/` + `rdna-boosts-all.patch`.
3. `scripts/make-release.sh` (refreshes `release.json`: the new `.tip`, `.tree`, sha256s) → `scripts/validate-set.sh`
   (checksums + fresh-tarball strict apply + Windows path/portability checks + CI).
4. `scripts/check-rung-file-sets.py --rung 0017 --new-chain ~/llama.cpp --new-base <sha>` **and** the whole set — this
   change must not lose a file/`#define`/env var/option block 17 carried. A deliberate retirement goes in
   `patches/FILE-SET-RETIREMENTS.tsv`.
5. **The owed gate arm** (part of the fix, not a nice-to-have): `repro-373/` is its skeleton. Turn it into
   `scripts/gate-arms-boundary.sh` (or a `boundary` mode of `gate-arms-soak.sh`): 3 slots, **two prefills in flight + a
   decode**, pad buckets shrunk (`LLAMA_KV_N_PAD_MIN=256 LLAMA_KV_N_PAD_PREFILL=0`, so a mismatch cannot hide in
   rounding), **N >= 3 repeats**, assert: no abort, every request completes, a fourth *sequential* request's text equals
   the single-context server's (a class-matched oracle, `GREEDY-PURITY.md` §105), and `check-gpu-faults.sh` clean. Add
   the row to `MANIFESTS.md` (block-17 verification) and to `scripts/README.md`.
6. **Records** (all of them, in the same change): a `WORKLOG.md` entry (`YYMMDD.VVV`, newest first — today's real date);
   `INVARIANTS.md` **appended** with the new `rNN` facts (the five defects + `#366` + `#365`'s result); `patches/README.md`
   block 17 (what the block now contains); `ENVIRONMENT.md` if (and only if) a new env var appears — **none of the
   current fixes adds one**, the block-17 knobs are already catalogued; `TODO.md` — close `#373`/`#365` by **deleting**
   the rows and the detail sections (a closed item is a `WORKLOG` entry, never a TODO entry), keep `#374` (the vacuous
   `pool` gate arm) open, and re-check `#364`/`#366` bookkeeping.
7. **Gates** (`AGENTS.md` → *Pick the gate for what you changed*, and `MANIFESTS.md` → *Release gates*): the whole-set
   build at the tip (`~/bin/build-llama-rocm`, the `all` profile) — *not* the per-block walker; `gate-arms-soak.sh
   soak`; `gate-ckpt-reuse.sh`; `gate-prefill-logits.sh` **per ROCm line** (`build-rocm` 10.2 too — every fix here is
   behaviour-neutral, so it must read bit-identical/KLD 0.000000); `check-gpu-faults.sh`; and the handover's own
   `#365` arms.
8. **Release mechanics**: tag `v18-919794500-rNNN` (next revision), then **`scripts/fork-refresh.sh --build --gates
   --push`** (this is mandatory on every release — a stale fork branch is what caused the 2026-10 "rdna-boosts is slow"
   incident), and **do not** build the container images on the primary box: the worker watcher
   (`scripts/container-watch.sh` on `radiance`) picks the tag up and pushes them (`CONTAINERS.md`).
9. **Pushing policy**: the only permitted targets are this repo and the personal fork (`git@github.com:stew675/llama.cpp`,
   branch `rdna-boosts`). **Never** upstream `ggml-org/llama.cpp`; never a bare `git push` in `~/llama.cpp`;
   `--force-with-lease` only.

---

## 6. Landmines (each has cost a session in this project)

* **Report findings before any delivery-affecting commit, and before any tag** (`AGENTS.md`). The block structure is the
  maintainer's; "the gates are green" is not permission.
* **`-lv` is picked by the arm**: text/hash arms take **no `-lv`**; census/mechanism arms take **`-lv 4`**; `-lv 3` is the
  trap (it pollutes text *and* hides the ggml censuses). `llama-server` hashes over HTTP, so `-lv 4` is safe there.
* **`HIP_VISIBLE_DEVICES` on every multi-device arm** (the iGPU is enumerated; an unpinned split aborts with `invalid
  kernel file`).
* **One arm at a time** on this box; size `-t` (the maintainer's `runme` uses 7); the model is **VRAM-marginal** (94 GB on
  ~96 GB) — a load can fail, retry.
* **Never `kill -9` a `timeout` wrapper** — it leaves the server alive and `rocm-smi` does not count VMM, so "VRAM
  drained" passes while a leaked server answers `/health` and becomes the arm you think you are driving. `run.sh`
  starts under `setsid` and kills the group; the gates refuse to start while `llama-server` exists.
* **No core dumps on this box** (`core_pattern = /bin/false`, `coredumpctl` storage off). For a SIGSEGV either run under
  `gdb -batch -ex run -ex 'thread apply all bt 15'`, or read `journalctl -k`'s `segfault at <off> ip ... in libX[<off>]`
  and map it with `nm -C --defined-only -n` on that `.so` (that is how the Meta and libstdc++ crashes were placed).
  `ggml_abort` already forks a `gdb -batch` for aborts, which is where the log's backtraces come from.
* **`/tmp` is tmpfs** on every box here — no large artifacts there; use `~/.cache/rdna-boosts/gates`.
* **Ports 8037-8039 are taken** by a `llama-slot-proxy`; the `runme` uses 8033.
* **`GGML_CUDA_GRAPHS` does not exist** — the knobs are `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_ASYNC_INPUT`,
  `GGML_CUDA_P2P`, `GGML_CUDA_SLAB*`, `MOE_EXPERT_CACHE_MIB`/`_AUX_RESERVE_MIB`. Getting the name wrong once cost an hour
  of "the bisect says it is not capture".
* **A text reference is per device count, per split AND per residency class** (`AGENTS.md` rule 6, `GREEDY-PURITY.md`
  §38/§89/§105). Compare within one shape.
* **`-ncmoe 48` needs a reserve**: the host tier is only clean with the derived reserve (`MOE_EXPERT_CACHE_MIB=13653`,
  `-c 131072`, arena 13.3 GiB on this box). `-c 204800` + `-ncmoe 48` sits on the VRAM cliff — expect OOM there and do
  not chase it as a defect.

---

## 7. Open questions for the maintainer

1. **The owed gate arm** — `scripts/gate-arms-boundary.sh` (its own file, its own row in `MANIFESTS.md`) or a `boundary`
   mode folded into `gate-arms-soak.sh`? (The soak gate already has the lifecycle/`setsid`/refusal discipline, so folding
   is less code; a separate file is clearer as "the thing that would have caught this".)
2. **`TODO` #374** (the vacuous `pool` arm): repair the census now, or keep it open and note in `MANIFESTS.md` that the
   pair's host-expert proof comes from the boundary arm instead? *(Do not relax the check.)*
3. **`#366`'s ordering hazard** (the slab's aux reserve is whatever the LAST preflight carried — see
   `wip/slab-draft-aux-reserve/`): in scope for this release or a follow-up? It is the same field this change now writes.
4. **The revision number** for the release (`r347`?) and whether the maintainer wants the host tier validated at
   `-c 204800` (marginal) or the `-c 131072` shape used here.

---

## 8. First 30 minutes, concretely

1. `cd ~/llama-cpp-rdna-boosts && git log --oneline -3 && git status -sb` → expect `b56c41b0` (or newer), clean, synced.
2. Read §1's list — at minimum `WORKLOG.md` 261122.032–.035 and campaign §W2.
3. `cd ~/llama.cpp && git log --oneline -1 && git branch --list 'crash373*' -v` → `rdna-boosts` @ `8f88fc861`, and
   `crash373-fix` carrying the 4 fix commits.
4. `git checkout crash373-fix && cmake --build build-2ctx --target llama-server -j 16` — then run §3's re-validation
   (resident + host tier + the soak gate). **If any of it fails, stop and report**: something changed under the fix.
5. Read the `#365` code: `llama_memory_seq_state_take()` (`src/llama-context.cpp`), `can_take_batched()` +
   `seq_state_take()` (`src/llama-memory-recurrent.cpp`, ~825 and ~863), the meta buft initialiser
   (`ggml/src/ggml-backend-meta.cpp`, `cpy_tensor = nullptr`), `ggml_backend_meta_get_split_state()`,
   `ggml_backend_meta_kv_region_grow()` and `ggml_backend_meta_get_tensor_impl()`.
6. Instrument the take's timing first (one print), confirm the **74.6 ms** baseline on the maintainer's shape, then
   implement `cpy_tensor_n` and re-measure. **Verify with `LLAMA_RS_FORCE_SERIALISED=1` vs the new path and
   `LLAMA_RS_VERIFY_TAKE=1` byte-for-byte before anything else.**
7. Record what you find in `WORKLOG.md` as you go (a dated entry, newest first) — and **ask before the block amendment**.
