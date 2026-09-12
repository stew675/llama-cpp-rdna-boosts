# HANDOVER — TODO item 1: the Block 15 dense-arm defect (`LLAMA_QSA_SPARSE_FA=0`)

# HANDOVER — TODO item 1: the Block 15 dense-arm defect (`LLAMA_QSA_SPARSE_FA=0`)

> **OUTCOME (2026-09-11 (11)): SOLVED — the task is done; this file is the record of how.**  The defect was
> a **variable-shadowing bug** in block 15's own `build_attn_qsa` dense path: the V2/V3 refactor added an
> outer `ggml_tensor * kq_mask_top_k = nullptr;` while the top-k mask chain *inside*
> `if (kq_mask != nullptr) { ... }` still declared its own `kq_mask_top_k`, so the chain was built but its
> result never reached the attention (`build_attn_mha` got the outer `nullptr`).  The chain's nodes were
> then unreachable from the graph output, the packed mask lost its only consumer (unallocated + unfilled)
> and the dense arm attended with no mask at all — the causal leak.  Fix: drop the inner
> `ggml_tensor *`.  Steps 5.1-5.3 below were followed; the decisive instrument was the node dump (the
> delivery consumed `attn_inp_kq_mask` 36× in its dense prefill, the beta **zero** times and emitted no
> `FILL`/`SET_ROWS` chain nodes), confirmed by a temporary `[QDM]` log printing `kq_mask=1` …
> `outer_top_k=0`.  Ninth re-cut landed: base `6d3155faa` → beta tip **`3712e2dc1`**, tree
> **`e39f8c2b6f0593113b93c4e57c512bc7373a2250`**, patch **3 811 lines**.  Post-fix gates (vs the delivery,
> identical configs): oracle sparse `6.5394` / dense `6.5377`; dense texts tensor f16 `2daa19579316`,
> tensor `iq4_nl` `3c46e47ab345`, layer f16 `e656b50f2cc8`, layer f16 `-fa off` `b96459bf02ca`;
> random text `19.0589` / `7.9682`; production arm untouched.  See `WORKLOG.md` 2026-09-11 (11) and
> `beta/block-15-campaign-wins/BETA-TESTING.md` §4c.  Two instruments worth reusing for any "does the model
> see the future?" question: **random text** with `llama-perplexity` (a leak scores ≈1 on noise) and the
> **node dump** (a tensor whose consumer is missing is *silently* dropped by the allocator).

**Read this file first; it is self-contained for this task.**  The shared environment, instruments,
reference hashes and landing procedure are in
`wip/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` §2–§5/§12/§13 (its *plan* is
spent — see the status banner there — but the environment/traps sections are live).  The beta record
is `beta/block-15-campaign-wins/README.md` + `BETA-TESTING.md` (§4c is this defect).

## 1. Mission

Block 15 (the attention-memory campaign, staged in `beta/block-15-campaign-wins/`) **cannot be
promoted until its dense masked arm works again**.  With `LLAMA_QSA_SPARSE_FA=0` the beta produces a
near-1 perplexity on qwen4exp for **every** KV type — the signature of a lost causal constraint — and
that arm is this repo's *documented quality oracle* (`tools/qsa-ppl-oracle.sh` compares the fused
sparse path against it on every QSA/FA change, so a broken oracle silently voids a whole class of
gates).

**Definition of done:** the dense arm's output matches the delivery's for every model/type/split in
the gate list below (or the difference is root-caused to a *documented, intended* change with the
maintainer's sign-off), the beta patch is amended + re-cut (9th), it still applies cleanly with
`git am -3`, round-trips and builds, the beta records and `TODO.md` item 1 are updated, and the
"Known and accepted differences" section of `BETA-TESTING.md` no longer needs the §4c warning —
either deleted (fixed) or rewritten to describe the intended behaviour.

## 2. The defect, exactly as measured (2026-09-11 (10))

Repro (one command, ~40 s):

```sh
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
BIN=/tmp/blk15x/build-beta/bin tools/qsa-ppl-oracle.sh tensor f16     # tools = wip/kv-quant-purity-followups/tools
```

| build | arm | qwen4exp PPL (8×4096) |
|---|---|---|
| delivery (`6d3155faa`, `build-base`) | sparse (default) | **6.5394 ± 0.132** |
| delivery | dense (`LLAMA_QSA_SPARSE_FA=0`) | **6.5377 ± 0.132** |
| beta (`d0f71b2e8`, `build-beta`) | sparse (default) | **6.5394 ± 0.132** ✓ identical |
| beta | dense | **1.0558 ± 0.003** ✗ |

The tiny std (±0.003 vs ±0.13) is itself diagnostic: every chunk is "equally (im)perfect" — the model
effectively sees the answer, i.e. the causal part of the attention mask is gone.

**A cheaper, sharper instrument than the PPL** (use this one first): the same comparison on a *text*
run.  `LLAMA_QSA_SPARSE_FA=0 -ctk f16 -ctv f16 -sm tensor`, the 3k-token prompt, greedy
(`--seed 42 --temp 0 --single-turn --no-display-prompt`, `-n 128`):

| build | arm | text hash |
|---|---|---|
| delivery | sparse | `804de0576868` |
| beta | sparse | `804de0576868` ✓ |
| delivery | **dense** | **`2daa19579316`** (720 chars) ← the reference to reproduce |
| beta | **dense** | **`d910d0b499ec`** (713 chars) ✗ |

## 3. What is already EXCLUDED (do not redo any of this)

* **Not the sparse arm / not the production path.**  The beta's sparse arm is *byte-identical* to the
  delivery on every gate: texts (`804de0576868` f16, `886292b17a93` q4_1, `acd18ad2d55c` iq4_nl), MTP
  acceptances (27B f16 `0.82716`, qwen4exp f16 `0.47009`, `iq4_nl` `0.52727`), `W=1..8` width purity
  (both splits × f16/`iq4_nl` × default/forced), `FLASH_ATTN_QSA` 22/22, `FLASH_ATTN_EXT`
  5940/5940, `LLAMA_QSA_OFF=1` PPL `6.5376`.
* **Not an environment gate.**  Individually and *all together*, `LLAMA_KQ_MASK_DERIVED=0`,
  `GGML_QSA_DERIVED_BIAS=0 GGML_QSA_DERIVED_VIS=0`, `GGML_QSA_SCORE_MEM=0`, `LLAMA_QSA_KEYS_ONLY=0`
  leave the dense PPL at `1.0558` (measured with the oracle).
* **Not W4.**  `git apply beta/block-15-campaign-wins/ab/w4-revert.patch` (35 lines out of
  `ggml/src/ggml-alloc.c`), rebuild, re-measure → **identical** (`1.0558`, text `d910d0b499ec`).
  (I ran this; the worktree was restored and rebuilt afterwards.)
* **Not the FA kernel side, and not V3's derived-mask kernel arm.**  Re-run the dense arm with
  **`-fa off`** (`-sm layer`, f16 KV — tensor mode requires FA): delivery `b96459bf02ca` vs beta
  `2f164757f253` — **still different ✗**.  With `-fa off` the manual attention path is used, so no FA
  kernel and no derived-mask machinery is involved ⇒ the defect is in the **graph / mask values**.
* **Not the perplexity harness.**  The same divergence appears in a plain `llama-cli` text run (§2).
* **Not the patch application / the re-cut.**  The beta's tree builds clean, round-trips exactly
  (8th re-cut: `git am -3` on a fresh `6d3155faa` ⇒ tree `39540b7f4`), and the defect reproduces on the
  **7th** re-cut (old tip `5a0734c9d` → same `1.0558`).

## 4. What the evidence points at

The dense arm is the **only** arm that builds the *top-k mask chain* in `build_attn_qsa`
(`src/models/qwen4exp.cpp`):

```cpp
const bool qsa_derive_vis = qsa_sparse && qsa_cell_vis != nullptr;
ggml_tensor * kq_mask = qsa_derive_vis ? nullptr : inp->get_kq_mask();
ggml_tensor * kq_mask_top_k = nullptr;
if (kq_mask != nullptr) {                                     // <- block 15's new guard
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(...);                          // [1, n_kv, n_tokens, n_stream]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, ...);
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(...), 0.0f);
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(...);
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);    // <- the CAUSAL term
}
```

With the **sparse** arm, `qsa_derive_vis` is true (V2's derived visibility is on by default) ⇒
`kq_mask == nullptr` ⇒ **the whole chain is skipped** (that is the −800 MiB win) ⇒ the sparse arm
never touches it ⇒ which is exactly why every existing beta gate passed.  With the **dense** arm
(`LLAMA_QSA_SPARSE_FA=0`) `qsa_sparse == false` ⇒ `qsa_derive_vis == false` ⇒ `kq_mask` is the packed
mask ⇒ the chain **is** built and consumed by `build_attn_mha(..., kq_mask_top_k, ...)`.

A PPL of ~1 with a ±0.003 spread is what you get if the **`ggml_add` causal contribution is lost**
(the top-k `set_rows` part zeroes the selected cells, and without the `add` the mask is *only* the
top-k structure — no causal boundary) ⇒ future tokens leak.  So the working assumption is:

> something in the dense arm makes `kq_mask` (or `kq_mask_all`, or the `set_rows`/`view_4d` result)
> read the wrong memory / the wrong strides — i.e. a **view-allocation / stride / aliasing** problem
> introduced by block 15's graph or allocator changes, in a chain that *only* this arm builds.

Candidates, in the order I would test them (all in block 15's 23-file diff):

| file(s) | why |
|---|---|
| `src/llama-graph.cpp` + `llama-graph.h` | the kq-mask input creation/filling and the new derived-mask plumbing (block 15 touches it 183/60 lines) |
| `src/llama-kv-cache.cpp` + `.h` | the cache/visibility side (189/20 lines), incl. the keys-only W3 change |
| `src/llama-memory-hybrid-idx.cpp` + `.h` | `set_input_qsa` — where `cell_vis`/`q_vis` come from (95/17) |
| `ggml/src/ggml.c` + `ggml/include/ggml.h` | V3's new derived-mask op + the `set_rows`/`view` APIs (99/31) |
| `src/llama-context.cpp` | the derived-mask graph plumbing (130) |
| `src/models/qwen4exp.cpp` | block 15's dense-arm refactor (335) — note both *my* delivery version and block 15 use `if (top_k)` at the call site, so `top_k` is never null inside `build_attn_qsa` |
| `ggml/src/ggml-backend-meta.cpp` | the split/axis bookkeeping (18) |
| `ggml/src/ggml-cpu/ops.cpp` | the derived-mask CPU reference (71) — not on the GPU path, low prior |

## 5. The plan

1. **Instrument before bisecting.**  Apply `wip/kv-quant-purity-followups/tools/node-dump-instrumentation.patch`
   (`git apply -3`; `ggml/src/ggml-cuda/ggml-cuda.cu`) in a **scratch worktree**, build the beta and the
   delivery, and run the **dense arm** on both with
   `touch /tmp/nodedump_on; GGML_CUDA_NODE_DUMP=1 <llama-cli ...>`: the dump prints
   `idx/op/ne/nb/h0..h3/dev/name` per executed node, so the **first tensor whose hash differs** tells
   you exactly which part of the chain is wrong (the mask tensors are F16 and CUDA-resident, so they
   are hashed; unnamed ones appear as `node_N`, so key the diff on `(name, node index)` and compare
   `ne`/`nb` too).  `GGML_CUDA_NODE_DUMP=2` additionally dumps each node's `src[]`.
   *Note the instrument syncs + reads back per node, so it is slow (a 128-token run takes minutes) —
   run it with `-n 2` and keep the prompt short.*
2. **Confirm the mechanism cheaply.**  If the mask is the culprit, a one-off `llama-perplexity` run
   with the dense arm and `-c 512` should still show a near-1 PPL (small graph, same chain); and a
   *hash of the mask* that is all-zeros (or missing the `-INFINITY` fill) confirms the causal leak.
   You can also print `kq_mask_top_k`'s shape/strides from the graph builder (a `cb()`-named view).
3. **Bisect in *coherent groups*** if step 1 is inconclusive.  The beta is a single squashed commit, so
   a per-file revert to the delivery is one command —
   `cd /tmp/blk15x && git checkout 6d3155faa -- <file>` — but **revert whole interfaces together**
   (e.g. `llama-graph.{cpp,h}`, `llama-memory-hybrid-idx.{cpp,h}`, `llama-kv-cache.{cpp,h}`,
   `ggml.{c,h}`): a lone `src/models/qwen4exp.cpp` revert will not compile against block 15's new
   signatures.  Each cycle = revert → `cmake --build build-beta --target llama-cli -j 16` (~1–3 min) →
   the dense-arm text run (~1.5 min).  Restore with `git checkout .` (the worktree is scratch; the
   canonical fork is untouched).
4. **Fix, then re-validate the whole beta** (§6), amend the beta patch in place
   (`git commit --amend`, or a new dated amendment commit squashed — see §7), re-cut (9th) and
   round-trip.
5. If the fix turns out to live in *shared* code (i.e. the delivery has the same latent bug behind a
   different configuration), then it is a **delivery block amendment** as well — check whether the
   delivery's own dense arm (`2daa19579316`) is affected (it is not, today) and land accordingly:
   block 08/14 (whichever owns the file) + a full regeneration.

## 6. Gates (all mandatory after any change)

**The defect itself**
* `tools/qsa-ppl-oracle.sh tensor f16` → dense column ≈ `6.53` (delivery `6.5377`), sparse unchanged.
* Dense-arm text on qwen4exp, tensor **and** layer, f16 **and** `iq4_nl` (+ `-fa off -sm layer` for
  the manual path) → must reproduce the *delivery's* values, not just be "sane".  Measure the
  delivery's value for each new config first (the delivery is the reference; there is no older
  reference for arm-specific texts).

**The beta must stay a no-op on the production path** (re-run against the delivery build)
* qwen4exp f16 plain `804de0576868`, q4_1 plain `886292b17a93`, `iq4_nl` plain `acd18ad2d55c`,
  `plain == n_max 3 == n_max 7`; MTP `n_max 3` f16 `0.47009` (pos-1 0.615), `iq4_nl` `0.52727`
  (pos-1 0.757), 27B f16 `0.82716`; `W=1..8` purity (both splits × f16/`iq4_nl` × default/QSA-forced);
  `LLAMA_QSA_OFF=1` PPL `6.5376`; `FLASH_ATTN_QSA` 22/22; `GATED_DELTA_NET` 46/46; `FLASH_ATTN_EXT`
  5940/5940.
* The win accounting still holds (the campaign's memory numbers): re-check the reserves
  (`ctx 204800 / ub 2048`) for qwen4exp + the 4B/27B if the fix touches the mask/derived path — the
  `-800 MiB` mask win is the *reason* the dense arm's chain is normally skipped, so a fix must not
  accidentally materialise the mask in the sparse arm.

**Beta patch mechanics**
* `git am -3` on a fresh worktree at the current delivery tip (`6d3155faa`): one expected
  `src/models/qwen4exp.cpp` conflict, then a clean build.
* Re-export with `git format-patch --stdout --start-number 15 -1 <sha>`, restore the
  `[PATCH 15/15]` subject by hand, round-trip (apply the *exported* file to a fresh base and compare
  trees), and record the new base/tip/tree in the beta files.

## 7. Landing (beta amendment, not a delivery block)

Block 15 is not in `patches/` — do **not** regenerate the delivery unless the fix changes shared code
(§5.5).  Flow:

```sh
# scratch worktree already exists: /tmp/blk15x (branch blk15-iq4, tree 39540b7f4) -- it is the
# build-verified 8th re-cut; keep it scrubbed and rebuild with:
#   BUILD_DIR=build-beta EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# edit -> build -> gates -> commit:
cd /tmp/blk15x && git commit -am "<what changed>"
# export (from the *current* delivery tip) and install into the repo:
git format-patch --stdout --start-number 15 -1 HEAD > /tmp/b15-new.patch
sed -i '4s/^Subject: \[PATCH\] /Subject: [PATCH 15\/15] /' /tmp/b15-new.patch   # format-patch drops the /15
cp /tmp/b15-new.patch ~/llama-cpp-rdna-boosts/beta/block-15-campaign-wins/block-15-campaign-wins.patch
# round-trip: fresh worktree at the delivery tip, git am -3 the *exported* file, compare trees
```

If the base has moved (a new delivery tip), first re-apply on the new tip (`git am -3` + the same
`qwen4exp.cpp` conflict resolution — block 15's `qwen4exp_qsa_sparse()` must keep the `iq4_nl`
conjunct, which the 8th re-cut already carries) and re-verify the no-op gates.

**Docs to update** (newest-first, never rewrite a dated record): a new `WORKLOG.md` entry; the beta
`README.md` (a 9th-re-cut bullet + the fix/root-cause), `BETA-TESTING.md` (§4c rewritten: either
"fixed by X" or "this arm intentionally differs because Y" + the gate list), `HANDOVER.md` (the
re-cut line), `TODO.md` item 1 (unblocked / narrowed), and — if the fix is a *campaign* win's
behaviour change — the `beta/block-15-campaign-wins/` inventory.  Then commit + push **only** to the
delivery repo's `origin`.

## 8. Traps (each cost time in the session that found this)

* **The env gates are not the answer here** (§3) — and a "no change" result from a gate needs a
  *positive control* that the knob reached the process (e.g. `GGML_QSA_DERIVED_*=0` changes the
  reserves by ~400/800 MiB; check `--log-verbosity 4 | grep 'KV buffer size'` or the graph dump).
* **This box can go through slow phases** (I hit 2–3× slower runs plus one "failed to load model"):
  `/tmp` is a 93 GiB tmpfs, the qwen4exp model is 93 GB and gets mmap'd, and the box swaps into a
  zram device.  Before believing *any* timing, run a **control you know** (e.g. the f16 sparse text
  `804de0576868` or a q4_0 `pp8192` ≈ 2597 t/s) in the same batch; and never run benches in parallel.
* `rocprofv3 --kernel-trace` **distorts relative times** (it serialises; I measured a 17 s kernel
  sum for a 3–4 s run) — use it for *structure* (which kernels exist, `ne`/VGPR/LDS), not for
  deltas.
* `LLAMA_QSA_SPARSE_FA=0` **and** `-fa off` is the only way to exercise the manual-attention dense
  path; it needs `-sm layer` (or 1 GPU), because `SPLIT_MODE_TENSOR` *requires* FA.
* `-fa off` cannot be used as a control with a **quantized** V cache (upstream #25871) — use f16.
* The `qsa-ppl-oracle.sh` `run()` passes a bare `-` as the third argument for the sparse column, so
  that run executes under `env -` (an *empty* environment; `LD_LIBRARY_PATH` therefore comes from the
  binary's RPATH).  It works, but do not add new env-dependent flags to that column without checking
  them in both columns.
* `tools/textgen.py` strips the ASCII banner and the `[ Prompt: … | Generation: … ]` footer; a
  `-sm layer` run on this box needs the same harness (my first attempt at a hand-rolled extractor
  produced `EXTRACT-FAILED` and a false difference).
* `--log-verbosity 4` interleaves log lines *into* the generation → text-purity runs use the default
  verbosity; the `draft acceptance` line needs 4 → run those measurements separately.
* The beta worktree is scratch (`/tmp/blk15x`, branch `blk15-iq4`) — the **canonical fork**
  (`/tmp/canon-llama`, `rdna-boosts`, tip `6d3155faa`, clean) must stay untouched except for a
  delivery amendment.

## 9. Reference points

> **Post-fix state (2026-09-11 (11)):** the beta is the ninth re-cut — `/tmp/blk15x` @ **`3712e2dc1`**
> (branch `blk15-iq4`), tree **`e39f8c2b6f0593113b93c4e57c512bc7373a2250`**, patch **3 811 lines**
> (`git am -3` on `6d3155faa`, same `qwen4exp_qsa_sparse()` `iq4_nl` conflict).  The values in the table
> below are the *pre-fix* eighth re-cut and are kept only to show what the broken build looked like.

* Delivery: `/tmp/canon-llama` @ `6d3155faa`, tree `0c3f0c2c2f4e7439d9489d45573a4021a8eee106`;
  build `build-base`; binaries `build-base/bin/{llama-cli,llama-perplexity,test-backend-ops}`.
* Beta: `/tmp/blk15x` @ `d0f71b2e8` (branch `blk15-iq4`), tree `39540b7f4fd8e8569dee64bfa3ee84bf1b20e75d`;
  build `build-beta`; patch `beta/block-15-campaign-wins/block-15-campaign-wins.patch` (3 787 lines).
* Deliverable build commands: `BUILD_DIR=build-beta EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  (the `EXTRA_CMAKE_FLAGS` override is required with CMake ≥ 4.3), or the fast loop
  `cmake --build build-beta --target llama-cli llama-perplexity -j 16`.
* Model: `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
  (**`/models/…`, not `/llm/models/…`**); prompts `/tmp/prompt3k.txt` (2122 tokens, text gates) and
  `/tmp/qa-text.txt` (~100k tokens, the PPL oracle).
* Beta files worth reading before anything else: `README.md` (inventory + the gate table + the
  "Open questions — ALL RESOLVED" section, which documents the intended
  `LLAMA_QSA_SPARSE_FA=0` interaction: *"with `LLAMA_QSA_SPARSE_FA=0` the packed mask must be kept
  (the gate already encodes this via the shared `qwen4exp_qsa_sparse()` predicate)"*) and
  `HANDOVER.md` (the merge/gate/cut steps + the `qwen4exp_qsa_sparse()` conflict resolution).
