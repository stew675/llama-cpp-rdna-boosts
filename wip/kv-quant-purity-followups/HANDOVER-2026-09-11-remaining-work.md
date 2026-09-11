# HANDOVER — remaining purity / perf work (written 2026-09-11, after the block-13 band fix)

**Hand this file to the next session first.** It is self-contained: environment, binary locations,
instruments, reference hashes, the code sites for the first two items, the landing procedure, and the
accumulated trap list. Read §1 (the plan) and §2–§5 (environment + instruments) before touching a GPU.

## 1. The plan (agreed order, 2026-09-11)

Do **item 1 and item 5 in the same session** (they are the same defect shape and share all validation), then
the rest in order:

| # | item | type | why now |
|---|---|---|---|
| **1** | **cause 3** — qwen4exp `plain != draft-mtp` text: the **QSA indexer** machinery | correctness | last obstacle to `plain == draft-mtp` for qwen4exp; small, pattern proven, kill-switch known |
| **5** | **the MoE "asterisk"** — the decode-only fused shared-expert down gate | correctness | last width-impurity in the MoE class; same fix pattern; +3.1 % decode win to preserve |
| 2 | **F3** — native FA path for sub-`q8_0` KV types (`iq4_nl` first) | perf (biggest) | 3.4× speedup at the same memory; `iq4_nl` would obsolete `q4_0` |
| 3 | **block-15 promotion** | memory (biggest) | −3.4 GiB/GPU + −1.2 GiB host on qwen4exp; validated; **time-gated by the beta window** (`BETA-TESTING.md`), so it cannot be next anyway |
| 4 | **upstream PRs** — today's mmvq band fix, then the staged FA kernel-family fix | leverage | today's fix is *general* (any arch with a per-type cap < the band), 26 lines, bug-class |
| 6+ | GDN chunked prefill (latent), gemma-4-E4B 3-GPU meta splitter, mixed-K/V policy, block-15 nits | tail | see `TODO.md` |

## 2. Environment and state

### 2.1 Hardware / toolchain

* 3× AMD Radeon AI PRO R9700 (gfx1201, RDNA4), 32 GiB each.
* ROCm at `/opt/rocm-7.14-gfx1201`; **always** `export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib`,
  `export PATH=/opt/rocm-7.14-gfx1201/bin:$PATH`.
* 16 cores. **Never run benches in parallel** — they contaminate each other.

### 2.2 Canonical fork (the source of truth for the delivery)

`/tmp/canon-llama`, branch **`rdna-boosts`**, tip **`bfaa83d8a`**, net tree
**`4e5f2952f016f1ac160c53261f7b01d346322534`**, 15 blocks (00–14), clean. Block SHAs:
00 `1c7ab0e89`, 01 `aa4108b9d`, 02 `6e81ed5ed`, 03 `4dc962aa9`, 04 `03d004517`, 05 `70f330aed`,
06 `d2fc2cb34`, 07 `110b5391d`, 08 `38cffdece`, 09 `3484c378f`, 10 `d60105926`, 11 `e07549b55`,
12 `f6198fbc9`, **13 `e3b189cee`** (the 2026-09-11 band fix), **14 `bfaa83d8a`**.

Scratch branches that also exist there and are **not** the canonical chain: `blk15-f2c2` (the block-15
re-cut commit `3f4e0747d`) and `blk15-recut` (held by the `/tmp/blk15` worktree). `rdna-boosts` is the
canonical chain; check `git branch --show-current` and `git rev-parse --short rdna-boosts` **before**
committing anything there (see the trap in §13).

Rebuild from scratch if `/tmp` is gone:

```sh
git clone https://github.com/ggml-org/llama.cpp /tmp/canon-llama && cd /tmp/canon-llama
git checkout 9113cc188
bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh .      # creates rdna-boosts: strict 15/15
BUILD_DIR=build-base EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
```

* The `EXTRA_CMAKE_FLAGS` override is **required** with CMake ≥ 4.3 (the script hardcodes a bare
  `-DCMAKE_HIP_FLAGS="-mllvm"`; CMake's HIP test injects `--cuda-host-only` right after it).
* Fast loop: `cd /tmp/canon-llama && BUILD_DIR=build-base cmake --build build-base --target
  llama-cli llama-bench ggml-hip -j 16` (a `ggml-hip`-only rebuild is ~2 min; a full build ~20 min).
* Binaries: `/tmp/canon-llama/build-base/bin/{llama-cli,llama-bench,llama-batched-bench,test-backend-ops}`.

### 2.3 Delivery repo

`~/llama-cpp-rdna-boosts`, `main` = **`8be338d`** == `origin/main` (pushed), 15 patches `0000`–`0014`,
`rdna-boosts-all.patch` (single net patch), `scripts/make-patches.sh` default tip `bfaa83d8a`.
Block 15 is **beta-only**: `beta/block-15-campaign-wins/block-15-campaign-wins.patch`, tip
**`3f4e0747d`**, tree **`d50b4e121`**, cut on base `bfaa83d8a` (re-cut 2026-09-11, metadata-only).

**Any block amendment invalidates the block-15 re-cut** — re-cut it before you finish (§12.6).

### 2.4 Volatile `/tmp` inventory (recreate what you need; the durable copies are in `tools/`)

| path | what |
|---|---|
| `/tmp/lw-f2` | the width probe (**rebuild**, §4.1) |
| `/tmp/so/so-fixed.so`, `/tmp/so/so-base.so` | the swappable `libggml-hip.so` pair used for the interleaved A/B (§4.4) |
| `/tmp/prompt3k.txt` | 13 000-byte prompt for the text / MTP runs (regenerate: any ~3.3k-token English text works, but the *hashes* below are tied to it) |
| `/tmp/p0long.txt` | only for the issue-25 probe; the *width* probe text lives in the repo (`wip/sm-tensor-plain-vs-spec/p0long.txt`, 532 tokens) |
| `/tmp/gd-w4.log`, `/tmp/gd-w5.log` | `[GD]` full-graph dumps at W=4/W=5 (the Task-1 evidence) |
| `/tmp/nd-w{4,5,6,7}.log` | `[ND]` node dumps |
| `/tmp/txt-*.log`, `/tmp/gdnoff-*.log`, `/tmp/qsaoff-*.log`, `/tmp/sparseoff-*.log` | the cause-3 text matrix of §6.2 |
| `/tmp/cap-w*.log`, `/tmp/fix-w*.log`, `/tmp/fixt-w*.log`, `/tmp/e3-w*.log`, `/tmp/tr-w*.log` | the cause-2 fix evidence |
| `/tmp/f2c2-fix.patch` | the landed block-13 band fix (26/11 lines) |
| `/tmp/blk15-sim` | a clean-apply sim of the beta patch |

## 3. Models

| key | path | notes |
|---|---|---|
| qwen4exp | `/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | **`/models/…`, not `/llm/models/…`**; 3 shards, shard 1 is metadata-only; 512 experts, 48 layers; **UD dynamic quant** ⇒ mixed expert types per layer (47 layers `IQ3_S` gate/up, layer 2 `IQ4_XS`, down `IQ4_NL`/`Q8_0`) — this is what made cause 2 visible; has GDN *and* QSA |
| qwen4exp MTP draft | `/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | `-md …` for `--spec-type draft-mtp` |
| MoE 35B-A3B | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 40 layers (30 GDN + 10 full-attn); 1 GPU (`HIP_VISIBLE_DEVICES=0`); Q4_K experts (cap 7) |
| 27B dense | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` | 3-GPU `-sm tensor`; MTP `nextn_predict_layers=1` |
| 4B dense | `/home/stew675/Qwen3.5-4B-Q8_0.gguf` | 1 GPU, no MTP head; fastest model, use for smoke tests |
| gemma-4-E4B | `/llm/models/Gemma4/E4B-IT/gemma-4-E4B-it-Q8_0.gguf` | SWA; 1 GPU (3-GPU `-sm tensor` aborts — known, §11) |
| gemma-4-31B | `/llm/models/Gemma4/31B-QAT/Q4_K_XL/gemma-4-31B-it-qat-Q4_K_XL.gguf` | SWA; 3 GPU |

`llama-cli` **must** be given `--single-turn` or it hangs in interactive mode. `-ts 1/1` is a
per-device *weight* list, not device ids.

## 4. Instruments (the framework)

All scripts live in `wip/kv-quant-purity-followups/tools/` (durable) — copy them to `/tmp` and chmod +x.

### 4.1 The width probe (the primary correctness instrument)

`tools/logits-dump-kv.cpp` — loads a model, prefills `P` tokens, then decodes a batch of `W` tokens and
prints the **hash of the logits of the first decode batch** (plus `nv`, the vocab size):

```sh
clang++ -O2 -std=c++17 -I /tmp/canon-llama/include -I /tmp/canon-llama/ggml/include \
  tools/logits-dump-kv.cpp -o /tmp/lw-f2 \
  -L/tmp/canon-llama/build-base/bin -lllama -lggml -lggml-base -Wl,-rpath,/tmp/canon-llama/build-base/bin

HIP_VISIBLE_DEVICES=0,1,2 W=8 NGL=99 SPLIT=layer RS=0 CB=0 \
  /tmp/lw-f2 <model.gguf> <text.txt> [P=256] [ubatch=512]     # prints: [L] W=8 logits0_hash=… nv=…
```

Env: `W` (decode batch width), `CTK`/`CTV` (KV types, default f16), `SPLIT=layer|tensor|row`,
`NGL`, `TS`, `FA`, **`CB=0` (mandatory** — anything else installs a per-node dump callback that
changes what you measure), `REPEAT=1` (batch = W copies of the same token — isolates batch *content*
from width), `RS` (`0` or `from_w` = `n_rs_seq = W-1`, i.e. the recurrent-state snapshot dimension).

**Limits:** the probe's context is hard-coded `n_ctx = 2048` and its token buffer is 8192, so
`P ≤ ~2040` and the text must tokenize to ≥ `P+8` (`p0long.txt` = 532 tokens; `prompt3k.txt` ≈ 3.3k).
It tests **one decode step** — that is its value (width purity at a fixed position) *and* its blind
spot (nothing that needs a roll-back or a second step).

### 4.2 The text harness (the user-visible acceptance instrument)

`tools/textgen.py` — extracts the *generated* text from a `llama-cli` log and prints
`<chars> chars  sha=<12 hex>`; it applies backspaces, takes everything after the first `> ` prompt echo
and before the `[ Prompt: … | Generation: … ]` footer, and writes `<log>.txt`:

```sh
python3 tools/textgen.py /tmp/txt-plain.log         # -> /tmp/txt-plain.log.txt + one summary line
```

The run it wraps (qwen4exp, greedy, deterministic):

```sh
M=/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
D=/models/Qwen3.8/Flash-Next/Q4_K_XL/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1,2 llama-cli -m $M [--spec-type none | --spec-type draft-mtp -md $D --spec-draft-n-max N] \
  -f /tmp/prompt3k.txt -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt \
  -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl all -sm tensor -mg 0 > log 2>&1
```

**Always run a control that must agree** before believing any divergence (e.g. plain twice, or
`plain` on the pre-fix vs post-fix build).

### 4.3 MTP acceptance / throughput

`tools/mtpab2.sh NMAX KV NGEN [REP]` — interleaved fixed/base MTP A/B, prints the `Generation: n t/s`
and the `draft acceptance` line. The **gate** config is `3 f16 96` (f16 KV, n=96, `n_max 3`) and must
reproduce acceptance `0.76744`; `7 f16 96` is the affected width (`W=8`). `tools/rv.sh mtp q4m` is the
older driver (pinned to q8_0 KV; its BIN must point at the build under test).

### 4.4 Perf A/B (swappable `.so`, never two build trees)

`tools/sobench.sh MODEL NPL NPP NTG [REP]` — copies `/tmp/so/so-{fixed,base}.so` over
`libggml-hip.so.0.23.0` in turn and runs `llama-batched-bench … --output-format jsonl`, printing
`speed_tg` per `pl`. JSONL fields are `pp`, `tg`, `pl`, `speed_pp`, `speed_tg` (**not** `n_pp`/`n_tg`).
`pl` = the number of parallel sequences = the decode batch width, so `pl=5..8` is exactly the
speculative-verify band. Use ≥ 2 reps and `RV_DEV`/`RV_SM` for 1-GPU models.

### 4.5 Code instrumentation (all env-gated, **all must be reverted before landing**)

* `tools/node-dump-instrumentation.patch` — applies to `ggml/src/ggml-cuda/ggml-cuda.cu` (use
  `git apply -3`); compiled in, gated at runtime by `GGML_CUDA_NODE_DUMP` **and** the file
  `/tmp/nodedump_on`; prints `[ND] idx=… (out|fdst) op=… ne=[…] h0..h3=… name` per executed node
  (`fdst` = the destination of a *fusion* — skipped nodes are never printed, which is the whole
  subtlety).
* **`[GD]` full-graph dump** (8 lines, used to settle "fusion vs graph-builder"; re-add at the top of
  `ggml_cuda_graph_evaluate_and_capture`):
  ```cpp
  if (access("/tmp/graphdump_on", F_OK) == 0) {
      static int gd_dumped = 0;
      if (gd_dumped < 8) { gd_dumped++;
          GGML_LOG_INFO("[GD] begin graph=%d n_nodes=%d\n", gd_dumped, cgraph->n_nodes);
          for (int k = 0; k < cgraph->n_nodes; ++k) { const ggml_tensor * n = cgraph->nodes[k];
              GGML_LOG_INFO("[GD] g=%d k=%d op=%s ne=[%lld,%lld,%lld,%lld] nb0=%lld %s\n", gd_dumped, k,
                  ggml_op_name(n->op), (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2],
                  (long long) n->ne[3], (long long) n->nb[0], n->name); }
          GGML_LOG_INFO("[GD] end graph=%d\n", gd_dumped); } }
  ```
* **Fusion/dispatch traces** (env `GGML_CUDA_FUSE_TRACE` / `GGML_CUDA_MMID_TRACE`) — print, per
  candidate, `ncols`, type, `cap`, `should_fuse_mul_mat_vec_{f,q}` and `should_use_mmq` inside the
  `{op, op, GLU}` arm of `ggml_cuda_try_fuse`, and the per-op branch data at the top of
  `ggml_cuda_mul_mat_id`. This is how the cause-2 mechanism was found.
* `tools/fa-kernel-chooser-trace.patch` — `GGML_CUDA_FA_TRACE=1`: the chosen FA family + launch plan.

### 4.6 Backend op gates

```sh
HIP_VISIBLE_DEVICES=0,1,2 /tmp/canon-llama/build-base/bin/test-backend-ops -o GATED_DELTA_NET   # 4/4 backends OK
HIP_VISIBLE_DEVICES=0,1,2 /tmp/canon-llama/build-base/bin/test-backend-ops -o FLASH_ATTN_EXT   # 4/4 backends OK
```

### 4.7 `tools/rv.sh` (the older driver, still useful)

`BIN=<build>/bin PROBE=/tmp/lw-f2 rv.sh {res|kv|coh|mtp|bench|width} …` — `coh` = same-seed coherence
text, `mtp` = the adaptive-MTP gate, `bench` = interleaved `llama-bench` A/B, `width` = the probe over a
width list. Its defaults point at the block-15 tree; **always override `BIN`/`PROBE`**.

## 5. Reference hashes (f16 KV, P=256, RS=0, CB=0, current canonical build)

**qwen4exp, the acceptance matrix (cause 2 fixed 2026-09-11):**

| split | `W = 1..8` |
|---|---|
| `-sm layer` | **all `3adeb313042a871b`** ( = the pre-fix `W=1` value) |
| `-sm tensor` | **all `dcf1ae667f730879`** ( = the pre-fix `W=1` value) |

Pre-fix (for regression triage): layer `1..4 3adeb313042a` \| `5 c999233926f0` \| `6,7 a8c532e12f9c` \|
`8 c56ebb61963a`; tensor `1..4 dcf1ae667f73` \| `5 2bfb89f59ec2` \| `6,7 e8b1253ea93e` \| `8 a7c5dfd26a56`.
HC off: layer `1..4 044715b66e72` \| `5 bdaa8fc57381` (+ the other values in `GREEDY-PURITY.md` §13).

**Other models:** 27B f16 1 GPU `W=1..8 4089b4d4`, `W=9 72af52db`; 2-GPU tensor `a4817ee6`/`b059daa6`;
3-GPU tensor `91434ea9`/`bc3faabd`; with q8_0 KV 3-GPU tensor `d4156dbeb225`. 4B 1 GPU = 2-GPU layer
`671d6096`, 2-GPU tensor `ef374ab3`, 3-GPU tensor `f4816fb0`.

**MoE 35B-A3B (the "asterisk"):** default `W=1` `ac8825358d9adfda`, `W=2/3/8` `bd138ad2326fbbf2`
(the fix removed the old `W=8` impurity); with `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` **all** widths
`bd138ad2326fbbf2` (pure); with `GGML_CUDA_DISABLE_FUSION=1` all `48bf23e1e9aa85f7`.

**Text (qwen4exp, `/tmp/prompt3k.txt`, 128 greedy tokens, f16 KV):** plain `3ee9daee5c07`;
`n_max 3` == `n_max 7` `8a50ea24e8d5` (pre-fix they disagreed: `8a50ea24e8d5` vs `e6918a7af1f9`);
plain == `n_max 3` byte-identical with `LLAMA_QSA_OFF=1` (`d4499ac8db72`); `GGML_CUDA_GDN_CHUNKED=0`
moves both (`dad4f4442580` / `9d29b773906f`); `LLAMA_QSA_SPARSE_FA=0` moves both and does not fix
(`25f300a81b9e` / `0d466b2dcf09`). 27B q8_0 KV 300 tokens: plain == `n_max 3` == `n_max 7`
`3537bc2b36be`; f16 control `f32aac948600`.

**Perf (interleaved, fixed/base, tg128 t/s):** qwen4exp 3-GPU tensor `b1 50.5/50.4`, `b2 85.4/85.6`,
`b4 134.1/132.9`, `b5 149.5/118.4`, `b6 162.5/130.6`, `b7 171.4/147.0`, `b8 178.0/155.4`;
35B-A3B 1 GPU `b1 98.3/98.1`, `b2 156.4/156.2`, `b4 254.2/254.1`, `b8 341.3/289.9`; 4B 1 GPU
unchanged (`b8 414.5/413.3`). MTP `n_max 3` f16/96: 80.0/80.1 t/s, acceptance `0.76744` both;
`n_max 7`: 41.8–42.5/36.1 t/s, acceptance `0.59375`/`0.55556`.

## 6. ITEM 1 — cause 3: qwen4exp `plain != draft-mtp` is the QSA **indexer**, not the FA kernel

### 6.1 Evidence (measured 2026-09-11, §5 hashes)

* `plain != draft-mtp` even with cause 2 fixed — and **the fix cannot be responsible**: at `n_max 3`
  (`W=4`) the cause-2 fix is a verified no-op (bit-identical logits, byte-identical text, byte-identical
  acceptance), and the plain text is identical pre/post fix (`3ee9daee5c07`).
* **`LLAMA_QSA_OFF=1` ⇒ byte-identical** (`d4499ac8db72` for both) — and the knob provably fired (the
  plain text moved `3ee9daee5c07` → `d4499ac8db72`). `LLAMA_QSA_OFF` "forces the dense no-indexer regime
  everywhere (no indexer store, scoring or sparse selection at any layer)".
* **`LLAMA_QSA_SPARSE_FA=0` does *not* fix it** (two different texts, both moved ⇒ the knob fired):
  the sparse-attention kernel (`fattn-qsa.cu`) is **exonerated**; the defect is in the **indexer
  store/score/top-k** machinery.
* The single-step width probe is **pure** on both splits and with `RS=from_w` ⇒ it is *not* a
  width-dispatch difference visible in one step. The divergence appears only after ~100 chars
  (~20 generated tokens) of the 3.3k-prompt run ⇒ **not** a prefill-state difference either.
* `GGML_CUDA_GDN_CHUNKED=0` moves both texts without making them agree ⇒ the known **Issue #25 GDN
  chunked-prefill** item is a *separate* contributor, and (with `LLAMA_QSA_OFF=1` and chunking **on**
  the texts already match) it is **not** currently breaking qwen4exp. Do not conflate the two.

### 6.2 First action: a two-knob bisection (≈ 8–12 min, 4 runs)

Both candidate paths are live in the default build and both are `n_tokens == 1`-gated (the same defect
class as cause 1), so each test is "does the W=1-only path move the W=1 run onto the verify run?":

```sh
# A) the fused indexer score (ON by default: `GGML_CUDA_QSA_INDEXER_SCORE` unset means 1)
GGML_CUDA_QSA_INDEXER_SCORE=0   <the §4.2 plain run and the n_max 3 run>
# B) the derived indexer cache (ON by default: `GGML_CUDA_QSA_INDEXER_CACHE` unset means 1)
GGML_CUDA_QSA_INDEXER_CACHE=0   <same two runs>
```

Whichever knob makes `plain == n_max 3` byte-identical names the site. Record all four hashes either
way, plus the *control* that the knob fired (the text must move vs its default).

### 6.3 The two candidate sites

**(a) fused indexer score — `src/models/qwen4exp.cpp:1094`** (live: `idx_score_fused` defaults to 1,
defined at `:1022`):

```cpp
if (idx_score_fused && idx_key_float && n_tokens == 1 && blk_bias && n_idx_h <= 8 &&
        rope_type == GGML_ROPE_TYPE_IMROPE) {
    // "FUSED PROBE (env-gated, GGML_CUDA_QSA_INDEXER_SCORE=1): ONE kernel replaces the per-token
    //  decode chain … The kernel replicates the per-op F32 arithmetic byte-identically
    //  (same gather addresses, add order, 256-thread rms_norm reduction, IMROPE half-pair rope,
    //  mmvf F32 vec-dot order) - see ggml-cuda/indexer-score.cu."
```

The comment *claims* byte-identity with the per-op chain — if that claim is wrong for some shape (a
different `n_idx_h`, a quantized indexer key, a different rms_norm reduction), that is exactly the F1 /
cause-1 / cause-2 pattern: a fused replacement that does not reproduce the unfused reduction order.
`ggml-cuda/indexer-score.cu` is the file to compare against the per-op chain.

**(b) the derived indexer cache — `GGML_CUDA_QSA_INDEXER_CACHE` (default 1)** — "a fill op first writes
the completed blocks' pooled+normed+ROTATED vectors" (see the comment at `:1100-1110`). A *cache* whose
content depends on the batch layout, or which is not rewound across a speculative roll-back, is the
classic multi-step divergence source; this is the more likely of the two given the ~20-token onset.

**(c) ruled out / less likely — `src/models/qwen4exp.cpp:1419`** (`qsa_dense_decode_until > 0 &&
n_tokens == 1 && n_kv < qsa_dense_decode_until`): the arm above it (`shortcut && n_kv <= width`) already
takes `build_qsa_store_k` for *both* widths while the context is below the selection width, so this gate
should be inert early — but check it once (a) and (b) are tested.

### 6.4 Fix shapes and gates

* Preferred fix: make the fused/cached indexer path take **one** arithmetic for the whole decode/verify
  band (`n_tokens = 1..8`), i.e. the cause-1 pattern (token on a block index + per-token strides, or
  simply gate the fused path to a band predicate and give the multi-token path the same fused form).
* If the fused form cannot be extended, the honest fallback is to disable it in the band (the
  `LLAMA_QSA_OFF`-style switch) — but **measure the cost**: §5's perf table must not regress, and the
  F1 acceptance rule applies ("a fix that restores purity by losing the fusion's win should FAIL the
  gate, not pass it").
* Acceptance: (i) with the fix, `plain == draft-mtp --spec-draft-n-max 3` **and** `--spec-draft-n-max 7`
  byte-identical, no env knob; (ii) qwen4exp `W = 1..8` still bit-identical on both splits (§5);
  (iii) the adaptive-MTP gate at `n_max 3` still `0.76744` and `n_max 7` not below `0.59375`;
  (iv) prefill/decode perf not regressed (interleaved, §4.4); (v) `GATED_DELTA_NET` +
  `FLASH_ATTN_EXT` 4/4; (vi) 27B/4B/gemma coherence unchanged.

## 7. ITEM 5 — the MoE "asterisk": the decode-only fused shared-expert down gate

### 7.1 Evidence

35B-A3B (1 GPU, f16 KV, P=256): `W=1 ac8825358d9adfda`, `W=2/3/8 bd138ad2326fbbf2`; with
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` every width is `bd138ad2326fbbf2`. So the residual is now
**strictly `W=1` vs `W>=2`** (the old `W=8` half was removed by the 2026-09-11 band fix).

### 7.2 The site: `ggml/src/ggml-cuda/ggml-cuda.cu:4700-4765`

A 6-node fusion `{MUL_MAT(down), MUL_MAT(gate), UNARY(sigmoid), MUL, ADD, ADD}` whose arm says:

```cpp
// Decode-only.  The fused gate reduction (shexp_gate_sigmoid) does not reproduce the order of the
// standalone mmvq/MUL_MAT it replaces, so a 1-token decode and an n-token verify batch of the same
// MoE layer are not bit-identical (the decode == verify invariant; the multi-token path runs the
// unfused chain).  Worth +3.1% decode on Qwen3.6-35B-A3B (tg128 101.6 vs 98.5 t/s), so it is on by
// default; set GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 for byte-identical MoE decode/verify results.
static const bool disable_shexp_down_gate = …;
if (!disable_shexp_down_gate && i + 5 < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
    …  const bool type_ok = … &&
        down_mm->src[1]->ne[1] == 1 && gate_mm->src[1]->ne[1] == 1; // decode only
```

* the **width gate is the last line**: `src1->ne[1] == 1` (for a dense `MUL_MAT` the batch dimension is
  `ne[1]`, so this is the same thing as `n_tokens == 1`);
* the kernels are in `ggml/src/ggml-cuda/mmvq.cu`: `shexp_gate_sigmoid` (`:2438`) computes **one scalar**
  dot with **one warp** (32 threads, `dst[0]`) and `shexp_down_gated_q8_0` (`:2458`) applies that scalar
  per row; the launch is `block_nums(1), block_dims(32)` (`:2537`). The design is single-token
  throughout.

### 7.3 Fix shapes

* **Extend the fusion to the band** (the cause-1 pattern): gate on `ne[1] <= MMVQ_MAX_BATCH_SIZE` and
  make the two kernels token-generic — the gate becomes a per-token vector (one warp per token, e.g.
  `block_dims(32, ncols)` or a loop), and `shexp_down_gated_q8_0` indexes the gate, `moe_out`,
  `ffn_residual` and `dst` per token column with explicit strides. Then every width takes the *fused*
  arithmetic ⇒ pure, and the band value moves to the decode value (`ac882535…`) — the F1 "move the
  cheap side and document it" pattern. Likely also a *perf win* at the verify widths (it was for the
  analogous MoE arm in cause 2).
* The alternative (make `shexp_gate_sigmoid`'s reduction reproduce the standalone mmvq order) is the
  "true fix" but harder; the comment implies the two orders genuinely differ.
* Do **not** simply flip the default to the kill-switch: that loses the +3.1 % decode.

### 7.4 Gates

(i) 35B-A3B `W = 1..8` all `bd138ad2326fbbf2` **or** all `ac8825358d9adfda` (whichever side the fix
moves — it must be *one* value at every width, and the moved side must be documented); (ii) the MoE
adaptive-MTP gate: acceptance `0.58378` not below, MTP ≥ plain (Protocol A in
`benchmarks/mtp-adaptive-methodology.md`); (iii) `llama-batched-bench` at `pl=1..8` not regressed
(`b1 98.3`, `b4 254.2`, `b8 341.3` are the current fixed values); (iv) qwen4exp + 27B + 4B unaffected;
(v) `GATED_DELTA_NET`/`FLASH_ATTN_EXT` 4/4; (vi) the MoE asterisk table in `patches/README.md` and
`AGENTS.md` is updated (the asterisk disappears if you make the band uniform).

## 8. ITEM 2 — F3: native FA paths for the sub-`q8_0` KV types

`q4_1/q5_0/q5_1/iq4_nl` are **pure** and 1800–2400 MiB (vs 3400 q8_0 / 6400 f16) but run
2197–2293 pp512 / 56–64 tg32 versus 7713–7838 / 95–99, because they are rejected by
`ggml_cuda_fattn_kv_type_supported()` (no native path ⇒ F16 staging scratch). `iq4_nl` is the
standout: same 1800 MiB as `q4_0`, pure, 3.4× slow — a native `iq4_nl` would obsolete `q4_0`.

* The mechanism already exists in **block 15** (the shared `FATTN_KV_NATIVE_{NONE,Q8_0,BF16}`
  per-operand staging type code, `GGML_CUDA_FA_KV_NATIVE`) — so this either rides on block-15's
  promotion or reimplements it in a delivery block.
* **First experiment is a build A/B** (`-DGGML_CUDA_FA_ALL_QUANTS=ON`), not a new kernel: measure the
  reserve deltas and the pp/tg speed for each type, then decide which types deserve a native path.
* Any new native path must be **width-invariant by construction** (F1 was exactly a native-path band
  split). Validate with §4.1 across `W = 1..8` on both splits, and re-check the `n_max <= 7` guarantee.

## 9. ITEM 3 — block-15 promotion (time-gated)

Block 15 (`beta/block-15-campaign-wins/`) is fully revalidated (the 2026-09-11 records in that dir) and
its patch is currently cut on base `bfaa83d8a` (tip `3f4e0747d`, tree `d50b4e121`). Promotion needs the
maintainer's go-ahead **and** the beta window to close (`BETA-TESTING.md`, ~4–5 days from 2026-09-10).
Procedure: `HANDOVER.md` §10.5 (move the patch to `patches/0015-…`, extend `scripts/apply-all.sh` and
`make-patches.sh` to a 16-block flow, sweep "15-block"/"15/15" → 16 in `AGENTS.md`, `MANIFESTS.md`,
`README.md`, `BASELINE.md`, `TODO.md`, `WORKLOG.md`, new WORKLOG entry). **Re-cut the beta patch after
any block amendment** (§12.6).

## 10. ITEM 4 — upstream PRs

* **Today's mmvq band fix** is the strongest candidate: the per-type cap splits the decode/verify band
  *and* `mul_mat_vec_q_moe`'s `__launch_bounds__` makes `ncols_dst > cap` unlaunchable, on any arch/tag
  whose cap is below `MMVQ_MAX_BATCH_SIZE` — i.e. batch-width-dependent arithmetic in a *purity*
  sense, plus 14–26 % measured on RDNA4. Cut as `upstream/UPSTREAM-PR-<slug>.md` + `.patch`, verified to
  apply to a **pristine** `9113cc188` (see `upstream/README.md` for the double-apply caution and the
  status table).
* The already-staged `upstream/UPSTREAM-PR-fa-decode-verify-kernel-family.{md,patch}` (the F1 fix) is
  still unfiled.
* `~/llama.cpp` must **never** be pushed; upstream PRs are filed from the `upstream/` artifacts.

## 11. ITEM 6+ — the tail

* **Issue #25 GDN chunked prefill** (plain vs spec): latent; it does *not* currently break qwen4exp
  (§6.1). Repro/fix directions: `wip/issue-25-mtp-batch-width/GDN-CHUNKED-PREFILL-FOLLOWUP.md`; the
  divergence is `gated_delta_net.cu` branch 1 (plain prefill `K=1` chunked vs spec prefill `K=n_max+1`
  sequential) and branch 2's `n_tokens-K` prefix boundary shifting with `n_max`. Needs a longer/deeper
  repro than 200 tokens.
* **gemma-4-E4B-it + 3-GPU `-sm tensor`**: aborts in the meta splitter (2 KV heads < 3 devices) —
  documented only; use 1/2 GPUs or `-sm layer`.
* **Mixed K/V cache types**: rejected by maintainer policy (1.7–3.6× slower, never smaller); the open
  sub-decision is hard error vs warning vs docs-only.
* **Block-15 nits**: the `src/llama-kv-cache.h:274` warning; V5's native-bf16 staging loss is not the
  conversion (measured within noise).

## 12. Landing procedure (any block amendment)

1. **Owner by provenance, not by guess**: `git log --diff-filter=A` / `git blame` the lines you change.
   Today: `mul_mat_vec_q_moe`'s invariant branch and the MoE fusion arms are **block 13** (commit
   `1e5580ee96` added the `has_ids` "decode == verify invariant" comment); `qwen4exp.cpp` graph gates
   are **block 14**; the shexp down gate is **block 13**. Prefer a tip-adjacent block unless a rebase is
   genuinely needed.
2. Work in `/tmp/canon-llama` with a **clean** tree (check `git branch --show-current` == `rdna-boosts`
   and `git status --porcelain` empty; see §13).
   **Items 1 and 5 land in *different* blocks (1 → block 14, 5 → block 13).  Amend the *earlier*
   block first (13 before 14), or do both in one `rebase -i` pass — amending the later block never
   disturbs the earlier one, but doing them in the opposite order costs two full rebases.**  If you
   touch block 13, block 14 replays on top; verify block 14's body is metadata-only afterwards.
3. Amend: `git diff -- <files> > /tmp/fix.patch && git checkout -- .` →
   `GIT_SEQUENCE_EDITOR="sed -i 's/^pick <sha>/edit <sha>/'" git rebase -i <prev-block-sha>` →
   `git apply /tmp/fix.patch && git add -A && git commit --amend --no-edit` → `git rebase --continue`.
   Replaying later blocks is expected; verify their bodies come out metadata-only
   (`git show <new> | diff - <old-replay>` ignoring the header).
4. `./scripts/make-patches.sh /tmp/canon-llama 9113cc188 <new tip>`; update the **default tip** in
   `scripts/make-patches.sh`; refresh `rdna-boosts-all.patch` by hand:
   `git -C /tmp/canon-llama diff 9113cc188 <new tip> > rdna-boosts-all.patch` (**the script does not
   write it**).
5. **Clean-apply sim**: `rm -rf /tmp/sim && git clone --no-local -q /tmp/canon-llama /tmp/sim &&
   cd /tmp/sim && git checkout -q 9113cc188 && git branch -D rdna-boosts &&
   bash ~/llama-cpp-rdna-boosts/scripts/apply-all.sh /tmp/sim` → expect **strict 15/15**, **0
   whitespace warnings**, and `git rev-parse HEAD^{tree}` == the new canonical tree.
6. **Re-cut block 15**: apply `beta/block-15-campaign-wins/block-15-campaign-wins.patch` to the new tip,
   squash to one commit, re-export (`git format-patch --stdout --start-number 15 -1 <sha>`), restore the
   `[PATCH 15/15]` subject if format-patch emits a bare `[PATCH]`, and verify the file differs from the
   previous one **only** in the `From <sha>` line (and the re-apply reproduces the recorded tree).
7. **Docs**: new dated `WORKLOG.md` entry at the top; `patches/README.md` (the block's table row + the
   block notes); `GREEDY-PURITY.md` (a new section, never rewrite dated ones); `AGENTS.md` (canonical
   tip/tree, block amendment, the purity critical fact); `TODO.md`; the `wip/` README; the beta records.
8. **Push only to `~/llama-cpp-rdna-boosts`'s own `origin`.** Nothing from `wip/` goes into `patches/`,
   and nothing is ever pushed from `~/llama.cpp`.

## 13. Traps (each has already cost a session)

**Measurement**

* `llama-cli`'s `/\|` spinner is `\b`-based and timing-dependent, and the banner embeds the build SHA:
  extract with §4.2 and **always run a control that must agree** (plain twice, or a pre/post pair).
* `--spec-type none` vs the default: not all runs are spec runs — check the invocation.
* Never run benches in parallel; do use ≥ 2 interleaved reps (the swappable `.so` is there for that).
* `-ts 1/1` is a per-device *weight* list, not device ids.
* `GGML_CUDA_ALLREDUCE=nccl` is **not** a bit-identical reference under `-sm tensor`.
* `-fa off` cannot be used as a control with a quantized V cache (upstream #25871 refuses it).
* Text-hash collisions between configs are common — the **raw-logit probe is the sensitive
  instrument**; text equality is evidence *for* purity, never against divergence.
* Any env knob test needs a **positive control** that proves the knob reached the process (otherwise
  "unchanged" is indistinguishable from "never applied").

**Instrumentation**

* Node dumps: only executed nodes are printed (`fdst` = a fusion destination, skipped nodes vanish);
  auto `node_N` names shift across widths (only `cb()`-named tensors are comparable); **shape equality
  is not sufficient** (cache/state tensors legitimately differ with `W`); sync before reading; gather
  with the real `nb[]` strides (`ggml_backend_tensor_get` ignores `nb[]`).
* A width pair that *already* agrees is the calibration set (W=6 vs W=7 was perfect for qwen4exp).
* `CB=0` on the probe is mandatory; `RS=0` and `RS=from_w` are different experiments.
* The probe's `n_ctx` is 2048 and its token buffer 8192 — a long-context experiment needs a source
  change, not a bigger `P`.

**Git / process**

* `git checkout -- <path>` restores from the **index**, so a *staged* instrumentation leftover comes
  back after "reverting". Check `git status --porcelain` (first column = staged) and use
  `git reset -q && git checkout -- .`; a `git apply -3` or an earlier `git stash pop` can stage things.
* **Never let a scratch commit land on `rdna-boosts`**: a `git checkout -b X || git checkout X` fallback
  once left the branch unchanged, so a block-15 commit went onto the canonical chain — recover with
  `git branch -f <tmp> <sha> && git checkout -B rdna-boosts <canonical-tip>` and re-verify
  `git rev-parse rdna-boosts^{tree}`.
* `git rebase -i` refuses to start with unstaged changes; `scripts/apply-all.sh` fails with
  "branch rdna-boosts already exists" in a clone that carries the branch (hence `git branch -D`).
* `pkill -f "llama-server.*8199"` self-matches the invoking shell.
* The block-15 patch file, `scripts/make-patches.sh`'s default tip and `rdna-boosts-all.patch` are all
  **derived** artifacts — regenerating the set without them leaves the repo inconsistent.

## 14. Out of scope / rules

* Everything under `wip/` is experimental and must not be applied to the fork or folded into `patches/`
  unless the user explicitly asks for that specific item.
* Never push from `~/llama.cpp` (the fork branch is disposable — delete and re-apply from the diff set).
  The only permitted non-delivery push target is the personal fork, and only on explicit request.
* Mixed K/V cache types are a rejected configuration.
* Do not present old dated records as current: the current state is in the header sections of
  `README.md` / `patches/README.md` and the top of `WORKLOG.md`.
