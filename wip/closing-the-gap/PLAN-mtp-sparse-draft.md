# Implementation plan — sparse MTP-draft attention for qwen4exp (Phase-2 follow-up)

**Status:** PLAN, not yet implemented.  Written 2026-09-22 by the session that fixed `patches/0019`
and audited `d67d58836`.  **Hand this whole file to the implementing session.**

**Audience:** a fresh coding session with the repo context in `AGENTS.md`.  Read
`closing-the-gap.md` START HERE + `2026-09-22-phase2-sparse-qsa-audit.md` first for where this fits.

---

## 0. TL;DR

Our MTP draft (`src/models/qwen4exp.cpp::graph_mtp`) attends **dense** over the whole KV cache.  The
trunk attends **sparse** (QSA selected cells) above its crossover.  The reference
(`~/pwilkin-llama-cpp`, commits `ddaf5214b` + `d67d58836`) runs the draft **sparse** too.  Measured on
gfx1151 at ~150K context, the single dense draft layer costs **2.1× all twelve sparse trunk layers**
during prefill and **5.6×** during decode.  The port is three edits (hparams patch, MTP-context
memory, `graph_mtp` graph) plus gates.  Expected: recover most of the draft's dense attention time
(~8–10 % of MTP prefill, ~5 % of MTP decode at that depth), and remove a cost that grows with depth.

**Do not** attempt this as a one-line `graph_mtp` change: our MTP context currently gets a **plain KV
cache**, not the hybrid-idx memory the sparse path needs.  The memory construction must change too.

---

## 1. Measured justification (already done — do not redo)

gfx1151 (Strix Halo), qwen4exp IQ4_NL `-ctk/-ctv f16 -fa on`, MTP `--spec-type draft-mtp
--spec-draft-n-max 3`, `rocprofv3 --kernel-trace`, `-b/-ub 4096`:

| context | kernel | who | calls | total |
|---:|---|---|---:|---:|
| ~150K | `flash_attn_ext_f16` (dense) | **draft, 1 layer, prefill** | 35 | **14 332 ms** |
| ~150K | `qsa3_attn_kernel` (sparse) | trunk, 12 layers, prefill | 420 | 6 896 ms |
| ~150K | `flash_attn_tile` (dense) | **draft, 1 layer, decode** | 108 | 288 ms |
| ~150K | `flash_attn_qsa` (sparse) | trunk, 12 layers, decode | 672 | 51 ms |
| ~60K | `flash_attn_ext_f16` (dense) | draft prefill | 15 | 2 648 ms |
| ~60K | `qsa3_attn_kernel` (sparse) | trunk prefill | 180 | 2 635 ms |

The trunk's sparse cost per query is capped at `top_k + r - 1` (~2051 cells); the draft's dense cost is
`O(n_q · n_kv)`, so the ratio grows with depth (1× at 60K → 2.1× at 150K for prefill).  The 150K
generation number (16.8 t/s) is **not** OOM and **not** the draft — the decode window is dominated by
`mul_mat_vec_*` and the indexer; the claim here is the *attention share*, which is what this port
removes.

Model facts (verified with `gguf-py` on the IQ4_NL checkpoint):
`qwen4exp.block_count = 48`, `nextn_predict_layers` → `n_layer() = 47`, and
`qwen4exp.attention.compress_ratios[47] = 4` (the pattern is `0,0,0,4` from index 3).  blk.47 ships
`indexer.{q,k}_{proj,norm}` — i.e. the MTP block is a fully indexer-equipped full-attention layer, the
same shape as the trunk's sparse layers.

---

## 2. Environment / reproduce

```sh
# fork + campaign (as left by the previous session)
cd ~/llama.cpp && git branch --show-current      # gap-closing-r13 @ 575c4c091
~/bin/build-llama-rocm-714                       # full build (ccache; ~4 min warm)
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0

MU=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
MD=/llm/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf   # NOT the shared-* sidecar
```

**Use the `Q4_K_M` sidecar, not `IQ4_NL/mtp-…-shared-Q8_0.gguf`** — the shared-head sidecar needs the
r13 block-00 fix and behaves differently; the qualification numbers above all use `Q4_K_XL/…Q4_K_M`.

Reference tree: `~/pwilkin-llama-cpp` @ `b0f31f587`, built at `build-rocm`.  The two commits to study:

```sh
git -C ~/pwilkin-llama-cpp show --stat ddaf5214b   # sparse selected attention graph (MTP infra)
git -C ~/pwilkin-llama-cpp show ddaf5214b -- src/models/qwen4exp.cpp src/llama-model.cpp
git -C ~/pwilkin-llama-cpp show d67d58836 -- src/models/qwen4exp.cpp   # the decode arm
```

---

## 3. Current state

### 3.1 Our tree (what is missing)

* **`src/models/qwen4exp.cpp::graph_mtp`** builds the draft attention with plain dense
  `build_attn(inp_attn, …, Qcur, Kcur, Vcur, …, kq_scale, il)` at **line ~950**, with
  `il = hparams.n_layer() = 47`.  It never calls `build_qsa_store_k` / `build_qsa_top_k` /
  `build_attn_qsa`, so the draft layer's indexer cache is never written and no top-k is built.
* **`src/llama-model.cpp::llama_model::create_memory`** (the `default:` branch, ~line 2539):
  `mtp_on_hybrid_qwen` includes `LLM_ARCH_QWEN4EXP` unconditionally, so a qwen4exp MTP context gets a
  **plain `llama_kv_cache`** with `filter = il >= n_layer()` — **not** a `llama_memory_hybrid_idx`, and
  therefore no indexer cache at all for the draft.
* **`src/models/qwen4exp.cpp::load_arch_hparams`** reads
  `LLM_KV_ATTENTION_COMPRESS_RATIOS` into `hparams.dsv4_compress_ratios` with no nextn fallback.  (Our
  target GGUF happens to carry `[47] = 4`; the **MTP sidecar's own** metadata is the question — the
  reference found it 0 and patched it.)

### 3.2 The reference design

Three coupled changes, all in the reference tree:

**(a) `load_arch_hparams`: inherit the trunk ratio for the nextn layer** (reference `qwen4exp.cpp`
~line 85, added by `ddaf5214b`):

```cpp
ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

{   // The converted GGUF leaves the nextn/MTP layer's compress ratio at 0, but the MTP sidecar ships
    // blk.N.indexer.* and runs that layer with the same sparse attention as the trunk, so inherit it.
    if (hparams.n_layer_nextn > 0 && hparams.indexer_head_size > 0) {
        int32_t trunk_r = 0;
        for (uint32_t j = 0; j < hparams.n_layer(); ++j) {
            if (hparams.dsv4_compress_ratios[j] > 0) { trunk_r = hparams.dsv4_compress_ratios[j]; }
        }
        for (uint32_t j = hparams.n_layer(); j < hparams.n_layer_all && trunk_r > 0; ++j) {
            if (hparams.dsv4_compress_ratios[j] == 0) {
                hparams.dsv4_compress_ratios[j] = trunk_r;
                LLAMA_LOG_INFO("%s: nextn layer %u compress ratio 0 -> %d\n", __func__, j, trunk_r);
            }
        }
    }
}
```

**(b) `create_memory`: give the qwen4exp MTP context a hybrid-idx memory** (reference
`llama-model.cpp` ~line 2580, `ddaf5214b`):

```cpp
const bool mtp_on_hybrid_qwen =
    params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
    (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ||
     arch == LLM_ARCH_BAILINGMOE3 ||
     (arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size == 0));   // <-- the change

if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP && arch == LLM_ARCH_QWEN4EXP &&
        hparams.indexer_head_size > 0) {
    llama_memory_hybrid_idx::layer_filter_cb f_attn = [&](uint32_t il) { return il >= hparams.n_layer(); };
    llama_memory_hybrid_idx::layer_filter_cb f_recr = [&](uint32_t /*il*/) { return false; };
    llama_memory_hybrid_idx::layer_filter_cb f_idx  = [&](uint32_t il) { return il >= hparams.n_layer(); };
    LLAMA_LOG_INFO("%s: MTP context uses a hybrid-idx memory (sparse draft attention)\n", __func__);
    return new llama_memory_hybrid_idx(
        *this, params.type_k, params.type_v, !cparams.flash_attn,
        cparams.n_ctx_seq, /*n_pad*/ 1, hparams.n_swa, hparams.swa_type,
        GGML_TYPE_F32, GGML_TYPE_F32, std::max((uint32_t) 1, cparams.n_seq_max),
        cparams.n_seq_max, cparams.n_rs_seq, cparams.offload_kqv, cparams.kv_unified,
        std::move(f_attn), std::move(f_recr), std::move(f_idx));
}
```

**(c) `graph_mtp`: route the draft through the QSA path** (reference `qwen4exp.cpp` ~line 601,
`ddaf5214b` + `d67d58836`):

```cpp
// the MTP context is a hybrid-idx memory, so the draft head can use the same sparse attention
// path as the trunk (this is what Halogen does: one sparse attention call per target chunk)
llm_graph_input_attn_kv * inp_attn = nullptr;
const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;
if (hparams.indexer_head_size > 0) {
    auto * inp_hyb = build_inp_mem_hybrid();
    const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);
    if (m->get_idx() != nullptr) {
        mctx_hyb = m;
        inp_attn = inp_hyb->get_attn();
        // the MTP graph has no recurrent layers, so the hybrid input's recurrent tensors are never
        // used and the allocator skips them -- set_input would then hit a null buffer. Give them a
        // trivial use.
        auto * rs = inp_hyb->get_recr();
        for (ggml_tensor * t : { rs->s_copy, rs->s_copy_main, rs->s_copy_extra }) {
            if (t) { ggml_build_forward_expand(gf, ggml_scale(ctx0, ggml_cast(ctx0, t, GGML_TYPE_F32), 0.0f)); }
        }
    }
}
if (!inp_attn) { inp_attn = build_attn_inp_kv(); }
```

then, at the attention site (replacing our dense `build_attn`):

```cpp
if (mctx_hyb) {
    const int64_t r        = hparams.dsv4_compress_ratios[il];   // patched at load for the nextn layer
    const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
    const int64_t width    = (int64_t) hparams.indexer_top_k + r - 1;
    ggml_tensor * top_k = nullptr;
    bool sparse_decode = false;
#if defined(GGML_USE_HIP)
    sparse_decode = n_tokens <= 8 && mctx_hyb->get_n_stream() == 1 &&
        cparams.flash_attn && cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap;
#endif
    if (r > 0 && n_kv_idx > width && (n_tokens >= 128 || sparse_decode)) {
        top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, il);
    } else if (r > 0) {
        build_qsa_store_k(mctx_hyb, cur, il);
    }
    if (top_k) {
        cur = build_attn_qsa(inp_attn, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
} else {
    cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
}
```

Note `build_qsa_store_k` / `build_qsa_top_k` / `build_attn_qsa` and the `qsa_inps` / `qsa_k_inp`
members are **`protected`** in our `llama_model_qwen4exp::graph`, and `graph_mtp : public graph`, so
they are directly callable.  `build_qsa_store_k`/`build_qsa_top_k` populate `qsa_inps` on the
`graph_mtp` instance (fresh per graph), and `build_attn_qsa` finds them by ratio.

---

## 4. Exact change list for our tree

Work on `~/llama.cpp` branch `gap-closing-r13`, one commit, then export as `patches/0020` (see §8).

### Step 1 — `src/models/qwen4exp.cpp::load_arch_hparams` (~line 209)

After the `ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, …)` line, add the reference's nextn
fallback block **verbatim** (§3.2a).  This is required for the sidecar (its own metadata may have 0);
on our target GGUF it is a no-op (already 4) and must stay silent.  Add the log line so a test can see
which path fired.

### Step 2 — `src/llama-model.cpp::create_memory` (~line 2539)

1. Add `&& hparams.indexer_head_size == 0` to the `LLM_ARCH_QWEN4EXP` term of `mtp_on_hybrid_qwen`
   (§3.2b).  Note the reference wrote it as a nested `(arch == LLM_ARCH_QWEN4EXP && hparams.indexer_head_size == 0)`.
2. Add the dedicated MTP hybrid-idx branch **before** `mtp_on_hybrid_nemotron`, using `return new
   llama_memory_hybrid_idx(...)` exactly as §3.2b (our function is
   `llama_memory_i * llama_model::create_memory(...)`, so the early return is valid).
3. Keep the trunk path untouched: for `ctx_type != MTP`, qwen4exp still creates the trunk hybrid-idx
   memory with `filter_attn = il < n_layer()` etc. (the existing `needs_mem_idx` block, ~line 2587).

### Step 3 — `src/models/qwen4exp.cpp::graph_mtp` (~lines 826–960)

1. Replace the attention-input acquisition with the reference's hybrid-memory block (§3.2c).  Our
   current code is `auto * inp_attn = build_attn_inp_kv();` (around line 870).  The reference's
   `if (!inp_attn) inp_attn = build_attn_inp_kv();` fallback keeps the non-indexer path working
   (other qwen4exp GGUFs).
2. Replace the dense `build_attn(inp_attn, …)` at line ~950 with the gated QSA block (§3.2c).
   `build_attn_qsa` returns the same `[n_embd, n_tokens]` shape as `build_attn` and already handles
   the `self_k_rot`/`self_v_rot`, the K/V cache store, the qsa3 packing and the F32 precision — do not
   duplicate any of that.
3. Keep the rest of `graph_mtp` (HC mix/combine, FFN, head mixer, `t_h_nextn`) unchanged.

### Step 4 — policy / kill-switch / default-on

* Add a static env gate, e.g. `LLAMA_MTP_SPARSE=0` disables the sparse draft and restores the dense
  path (A/B + bisection).  **Default ON** per the `AGENTS.md` default-on policy once the gates are
  green; the env var only disables.
* The reference's gate (`n_tokens >= 128 || sparse_decode`) should be the **starting** policy, but
  this repo has measured arch crossovers for the *trunk* (`qsa_dense_decode_until = 65536` on gfx1151,
  always-dense on gfx1201).  The draft economics differ (1 layer, large prefill batches).  Keep the
  prefill arm unconditional (`n_tokens >= 128`) — that is the big measured win — and **measure the
  decode arm's crossover**: start with the reference's `n_tokens <= 8`, then A/B a decode depth gate
  if the decode numbers in §6 are not a clear win.  Make any decode-depth threshold an env override,
  not a compile-time constant.
* Do **not** gate the sparse draft on the trunk's `qsa_dense_decode_until` silently; if you reuse it,
  wire it explicitly and document it.

### Step 5 — derived block-vector cache coverage

`build_qsa_top_k` reads `mctx_hyb->get_pool(ctx0, il, n_blocks)`.  The pool is created by
`llama_memory_hybrid_idx::pool_create` over `il < n_layer_all && filter_idx(il) &&
dsv4_compress_ratios[il] > 0`.  With Step 2's `f_idx = il >= n_layer()` and Step 1's ratio patch, the
MTP layer (47) gets its own pool.  **Verify this at runtime** (the `MMB`/`QSA` debug logs, or add a
temporary print) — if the pool is absent, `build_qsa_top_k` will fall back or misbehave.  Also verify
the indexer cache itself (`mem_idx`) is non-null for the MTP context (`mctx_hyb->get_idx()`).

---

## 5. Correctness gates (must be green before any perf claim)

Run on gfx1151, qwen4exp IQ4_NL + the `Q4_K_XL/…Q4_K_M` sidecar:

1. **Context creation / memory**: the MTP context must log the new hybrid-idx line and create
   cleanly.  `llama-cli`/`llama-bench` must not abort at init.  Check `n_kv` agreement (the idx cache
   tracks the attn cache).
2. **`plain == draft-mtp` byte-identical greedy text** (the intra-build purity contract,
   `GREEDY-PURITY.md` §5/§6).  Sparse-draft changes the draft's *predictions*, which is allowed, but
   the **target verify** path must stay band-uniform and the accepted sequence must be the same.
   Use `prompts/prose-rdna-boosts.txt`, seed 42, temp 0, and the `extract-generated.py` hash.
   If the text differs, that is a **bug**, not a re-baseline: speculative decoding is output-
   equivalent when the target is unchanged.
3. **`test-logits-width-probe` PASS (worst maxdiff 0)** on qwen4exp (`P=512 ub=512` at minimum) **and**
   on a dense control model (e.g. NanBeige BF16) — the memory change touches shared code.
4. **MTP acceptance gate** (`benchmarks/mtp-adaptive-methodology.md`): acceptance > ~0.45 at pos 1
   and `draft-mtp >= plain` at the default depth, at `-n 3000`, prompt pinned
   (`--reasoning off` for prose/code, `--reasoning on` for reasoning).
5. **Op oracles** likely unaffected (no kernel change), but run
   `test-backend-ops -o FLASH_ATTN_QSA` (expect 26/26) and `GATED_DELTA_NET` if the memory change
   touches the recurrent path.
6. **Coherence**: same-seed `llama-cli` output is coherent on a long source-summary run and matches
   the pre-change text where the contract requires it.

A/B/reference commands: `closing-the-gap.md` "Reproduce (copy-paste)" and the session-8 gate suite in
`beta/mmb-general/BETA-TESTING.md`.

---

## 6. Performance protocol

Perf A/Bs use **`-b/-ub 4096`** (the `-ub 8192` memory-pressure confound; see
`2026-09-22-ubatch-8192-memory-confound.md`).  Measure:

1. **MTP prefill / priming at depth** — the big win.  Build ~150K-token prompts
   (`head -c 620000 /llm/models/wikitext-2-raw/wiki.train.raw > /tmp/p150k.txt`), run
   `--spec-type draft-mtp --spec-draft-n-max 3 -n 60`, and read `[ Prompt: … t/s | Generation: … t/s ]`.
   Compare `LLAMA_MTP_SPARSE=0` vs default.  Also profile once with
   `rocprofv3 --kernel-trace --output-format csv` and confirm the dense `flash_attn_ext_f16` /
   `flash_attn_tile` draft calls collapse to `qsa3_attn_kernel` / `flash_attn_qsa`.
2. **MTP decode at depth** — the acceptance-adjusted metric: `draft-mtp` t/s vs `plain` t/s, at
   ~40K (below the trunk 64K crossover) and ~150K (above).  If the decode arm is a loss at some depth,
   gate it.
3. **`--spec-type none` plain decode must be unchanged** — the target path is untouched; if it moves,
   the memory change leaked.
4. Record min free memory with every `-ub 8192` result (we are using 4096, so note it is clean).

Report: `pp`/`tg` at both depths, the MTP speedup, acceptance, the profile family split, and the
kernel-time delta.  Use `benchmarks/` conventions and a new dated record (see §8).

Record **negative or flat** results honestly; this is a hold/repay item.

---

## 7. Traps (read before coding)

* **The MTP context is currently a plain KV cache.**  If you only edit `graph_mtp` and call
  `build_inp_mem_hybrid()`, `mctx_hyb` is null / the cast is invalid.  Step 2 is mandatory.
* **`build_inp_mem_hybrid()` recurrent inputs**: the MTP graph has no recurrent layer, so the
  allocator skips the recurrent tensors and `set_input` hits a null buffer.  Keep the reference's
  trivial-use loop (`rs->s_copy`, `s_copy_main`, `s_copy_extra` — our `llm_graph_input_rs` has exactly
  those fields).
* **`dsv4_compress_ratios[47]`**: our target GGUF has 4, but the **sidecar** is a separate model
  loaded with its own hparams — do not assume.  Step 1 is the fix; verify with a log line.
* **Do not apply the trunk's arch-policy gates implicitly.**  The reference's draft gate is
  `n_kv > width` + the decode guard, not `qsa_dense_decode_until`.  Decide explicitly and document.
* **Do not apply the reference's `d67d58836` indexer-cache redesign.**  The audit
  (`2026-09-22-phase2-sparse-qsa-audit.md`) concluded our derived block-vector cache
  (`GGML_CUDA_QSA_INDEXER_CACHE`, `pool_layers`/`pool_wm`) already covers it; only the `graph_mtp`
  routing + the MTP hybrid-idx memory are missing.  Keep `qsa_keys`-style logic out.
* **`LLAMA_QSA_SPARSE_FA=0`** must still work: `qwen4exp_qsa_sparse()` is used by `build_layer_attn`
  but the reference's `graph_mtp` gate does not consult it.  Wire the kill-switch so a dense-draft
  fallback exists (Step 4) and so the env can force the old behaviour for bisection.
* **`-sm tensor` / gfx1201**: the MTP draft on a Meta/multi-GPU device must not hit the
  `ggml_flash_attn_qsa` op unsupported-split abort.  The trunk guards this via
  `qsa_op_supported` / `qsa_arch_gfx`; the draft needs its own guard (reuse `qwen4exp_qsa_sparse`'s
  backend query or the same `mctx_hyb->get_n_stream() == 1` check).  Validate on the available box;
  leave gfx1201 to the validation session.
* **The shared-NextN sidecar is a different path** (`nextn_shared_target_tensors`, r13 block 00).  Test
  with the non-shared `Q4_K_M` sidecar first; then re-check the shared one does not regress.
* **Purity is intra-build.**  `plain != draft-mtp` text after this change is a defect (see §5.2), not
  an approved re-baseline — unlike a prefill kernel swap, this is a *draft* change under a verifying
  target.

---

## 8. Deliverable conventions (from `AGENTS.md`)

* Land as **one fork commit** on `gap-closing-r13`, message prefixed `gap-closing WIP: ` (match
  `patches/0016`–`0019`).
* Export with `git -C ~/llama.cpp format-patch -1 --stdout > wip/closing-the-gap/patches/0020-<slug>.patch`.
* Write a dated record `wip/closing-the-gap/2026-09-2X-mtp-sparse-draft.md` with: symptom/goal, the
  design, the exact gates with results, the perf table, the profile delta, and the traps hit.
* Update `wip/closing-the-gap/closing-the-gap.md` START HERE + NEXT SESSION and `README.md` (record
  table + "Current state" + fork tip/tree).
* Add a `benchmarks/2026-09-2X-…md` perf record if a clean win is claimed.
* **Never push** `~/llama.cpp` (Pushing policy).  Commit only in this repo.
* Keep the default-on policy: default ON once green; env only disables.

---

## 9. Rollback

* The change is self-contained: revert the one fork commit.  The dense draft path is the `else`
  branches, so an env kill-switch (`LLAMA_MTP_SPARSE=0`) is a cheap in-run rollback before the
  revert.
* If Step 2 (memory) proves too invasive, the fallback is to keep the plain KV cache and **skip this
  item** — the trunk sparse path and the audit's conclusion are unaffected.  Record the blocker.

---

## 10. Open questions for the implementer to resolve empirically

1. **Does the sidecar's own `compress_ratios[47]` read 0?**  Add the Step-1 log and check.  (The
   target GGUF reads 4; the sidecar is the unknown.)
2. **Decode-arm crossover**: is `n_tokens <= 8` sparse a win at 40K (below the 64K trunk crossover)?
   The prefill arm is expected to be the clear win; the decode arm may need a depth gate.
3. **Prefill arm scope**: the reference runs `n_tokens >= 128` sparse in the draft.  Our trunk prefill
   is always sparse; confirm the draft prefill is a win at every measured depth (it should be, since
   dense is `O(n_q·n_kv)`).
4. **gfx1201 / `-sm tensor`**: the draft `ggml_flash_attn_qsa` op must not force a meta-split abort;
   validate or gate it there.
