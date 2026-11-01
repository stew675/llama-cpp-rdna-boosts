# The MTP draft KV window is granted per sequence — issue #264 (r200)

**Question.** `LLAMA_SPEC_DRAFT_WINDOW` (block 15 Goal C, r193, default on) gives the MTP draft context a
sliding window instead of a `-c`-sized KV. Under `--kv-unified` that one cache holds every sequence, so what
is the cell count, and does a second sequence's batch still find room once the first has filled the window?

**Answer (r200; the bug shipped in r193-r199).** It did not: the r193-r199 shape (`win + n_ubatch` cells, shared by all sequences) left the
second slot nowhere to place a batch, and every request on it answered HTTP 500. The cache is now sized
`(win + one batch) x n_seq_max` (capped at the context) via `llama_spec_draft_kv_cells()`, the same function
the slab's draft reserve and `common`'s "is a window active" test call.

Recorded 2026-11-01, gfx1201 (one R9700, box `radix`), ROCm 10.1.0, `ROCBLAS_USE_HIPBLASLT=0` for every
fresh-process comparison, `-t 8`.  **Every arm is single-device** (this box has one R9700): the geometry below
depends on `n_seq_max` / `kv_unified` / the cell counts, not on the device count, and the 2-slot unified server
arm is the one that reaches it.

## The defect

`llama_kv_cache::find_slot` reclaims a cell that is more than `n_swa` behind its **owner's** newest position.
With a unified draft cache (one stream, N sequences) the cells of an *idle* sequence are therefore not
reclaimable by another one, and the only free cells left once sequence 0 has filled its window are the
`n_ubatch` + 256 headroom cells — which are exactly where sequence 0's own ring writes. So a second
sequence's first prefill batch finds no contiguous run:

```
W decode: failed to find a memory slot for batch of size 512
E spec      process: llama_process(ctx_dft) head=0 failed rc=1 (pos=35)
E srv  update_slots: decode() failed: failed to process speculative batch   -> HTTP 500
```

## Reproduced on the pre-fix build

Dense 27B `Qwen3.8-27B-UD-IQ4_NL` (qwen35, embedded MTP head, so the draft memory is the plain `llama_kv_cache`
branch), 1 GPU, `-c 40960 -fa on -np 2 --kv-unified -ub 512 -b 1024 --spec-type draft-mtp --spec-draft-n-max 3`,
`LLAMA_SPEC_DRAFT_WINDOW=8192` (draft cache 8 704 cells), slot 0 = a 10 498-token prompt, then slot 1 = 580
tokens:

| build | draft cache | slot 0 (10 498 tok) | slot 1 (580 tok) |
|---|---|---|---|
| r199 (pre-fix), window 8192 | 8 704 | OK | **HTTP 500** `failed to process speculative batch` |
| r199 (pre-fix), `LLAMA_SPEC_DRAFT_WINDOW=0` | 40 960 | OK | OK |
| r200 (fixed), window 8192 | 17 408 = `(8192+512)x2` | OK | OK |

The same comparison in the new gate (`WIN=1024`, `-c 8192 -np 2 --kv-unified`, 2 slots, 6 requests) fails
3 checks on the pre-fix build — `failed to find a memory slot for batch of size 953`, the second slot's HTTP
500, and `draft cache is 1536 cells, expected 3072` — and passes every check after it.

## The reporter's own shape (auto policy, 2 slots)

`-c 67584 -np 2 --kv-unified` with the auto window: `win = 32768` (the policy floor),
`per_seq = 33792`, so `32768 + 512 <= 33792` and the window is granted **per sequence**:
draft cache **66 560 = `(32768 + 512) x 2`** (the pre-fix shape was 33 280).  A 36 743-token prompt on slot 0
followed by a fresh prompt on slot 1: both answer.  At the reporter's `-c 262144` the same arithmetic gives
132 096 cells against the 262 144-cell full mirror — still a 130 048-cell saving.

## The draft reserve charges every stream

The same geometry function sizes the slab's draft reserve (`common_draft_reserve_estimate`), which is what the
slab holds back for a draft context created after it.  Under `--kv-unified` the returned cell count already
covers every sequence; **without** it, `create_memory` builds one `kv_size`-cell stream per sequence, so the
reserve must charge `n_seq_max x` the per-stream count — it charged one stream, i.e. it under-charged by
`(n_seq_max - 1) x (win + n_ubatch)` cells (the direction the reserve header calls dangerous).  Measured on
35B-A3B UD-Q3_K_M `-ncmoe 40 -np 2 -c 8192 --spec-type draft-mtp`, window 1024: **3 072 cells reported on
both arms** (2 x 1 536 per-stream, and `(1024 + 512) x 2` unified); the per-stream arm read 1 536 before.

## Where the window is declined

`per_seq = n_ctx_seq / n_seq_max` is the most any one sequence can own.  When `win + n_ubatch > per_seq` the
window is **dropped** (plain full-size draft cache, `n_swa` 0) rather than lowered to `per_seq - n_ubatch`: a
window that small saves one batch of cells per sequence and masks the drafter's newest cells to do it.
Measured: `-np 6 -c 8192` (per_seq 1 365 < 1 536) declines with no window log line and no reserve estimate
(the flat `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT` stays), and both slots run.

## Purity and behaviour

* **Text is the target's**, so the window can only cost acceptance: `plain == draft-mtp ==
  draft-mtp(W=1024) == draft-mtp(W=0)` — 876 chars, `sha=8c8fd395402f` (dense 27B UD-Q4_K_XL, 1 GPU,
  `-c 16384`, `prompts/prose-rdna-boosts.txt`, `-n 200 --seed 42 --temp 0`).
* **The new gate** (`scripts/gate-spec-draft-window.sh`, `WIN=1024`, `-c 8192 -np 2 --kv-unified`, 27B
  UD-IQ3_S, 1 GPU): all six steps (A slot 0 prose, B slot 1 mixed, C slot 0 mixed, D slot 1 prose, E/F repeats)
  are `win == win0` byte for byte, and draft acceptance is identical per prompt — 17/36 (0.472) on prose,
  23/23 (1.000) on the code+recall+reasoning mix.
* **Single sequence is unchanged**: `n_seq_max == 1` returns r198's exact expression, so every
  `llama-cli`/`llama-bench`/`-np 1` run measures the same draft cache (1 536 cells for a 1 024-cell window at
  `-c 8192`, 33 280 for `auto` at `-c 131072`).
* **Concurrent decode** with the window active (2 slots, 64 tokens each, after slot 0 has a resident window):
  both complete.
* **`--no-kv-unified`** is untouched (each stream already had its own window).

## Release-gate arms run for r200 (this fix)

| arm | result |
|---|---|
| `scripts/gate-spec-draft-window.sh` (default) | **PASS** — 3072 cells `(1024+512)x2`, 6/6 steps identical, acceptance identical |
| the same, `NPAR=1` | PASS — 1536 cells (the pre-fix number), confirming no single-sequence change |
| the same on the pre-fix build (negative control) | **FAIL** — HTTP 500 on slot 1, 1536 vs 3072 cells |
| reporter's shape (`WIN=auto`, `-c 67584`) | PASS — 66 560 cells, both slots |
| `-np 6 -c 8192` (window declined) | PASS — no window, both slots |
| concurrent 2-slot decode | PASS |
| text: `plain`/`draft-mtp`/`W=1024`/`W=0` | identical, `sha=8c8fd395402f` |
| `scripts/validate-set.sh` | strict 16/16, tree `c6c2f03de9ea2d13590b15ef77364665d8579466` |
| `scripts/gate-prefill-logits.sh` (build `2ed3997f5`) | **0.000000 / 100.000 %** |
| `scripts/gate-per-block-build.sh --fast --changed-from d5f241572` | 1/1 (block 15 only) |
| `scripts/fork-refresh.sh` (re-cut from `6050b467c`) | tree `c6c2f03d` + tip `2ed3997f5`, both == `release.json` |
| `scripts/check-gpu-faults.sh --since "4 hours ago"` | clean |

No kernel, no graph and no mask changed, so the prefill-logit gate is a formality on this amendment (it reads
`0.000000 / 100.000 %` on the shipped tree above and is run as part of the release flow).
