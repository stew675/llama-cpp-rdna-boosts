# Issue #25 on gfx1151 (Strix Halo) — raw investigation record (2026-09-11)

Reporter: 1337hero, dual R9700 (gfx1201), "Greedy output on dense qwen35 changes
with the MTP draft length (verify batch width) on the 14-block set".

**Verdict: reproducible — strongly — on a single Strix Halo (gfx1151).**
Not gfx1201-only.  Also: **not** caused by the 2026-09-08/09 adaptive-MTP
refresh, and **not** block 13/14.  The reporter's stated mechanism (ncols 3
vs ncols 5 verify logits differ) does **not** hold on gfx1151: verify batches
are bit-identical across widths.  What *is* broken on gfx1151 is the fork's
**decode (n_q=1) == verify (n_q>=2)** bit-identity, which the current block
chain does not deliver (the older `rdna-boosts-orig` chain did).

## 1. Environment

| item | value |
|---|---|
| machine | Strix Halo APU, Radeon 8060S (gfx1151), 1 device |
| ROCm | 7.14 (`/opt/rocm-7.14-gfx1151`) |
| model | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (exact reporter model) |
| model arch | `qwen35`, **hybrid**: 64 layers, `full_attention_interval=4` (48 GDN/SSM + 16 full-attn), n_head 24 / n_head_kv 4, `nextn_predict_layers=1` |
| config | f16 KV, `--flash-attn auto`, greedy (temp 0, top_k 1, seed 42), `-np 1`, ctx 16384 |
| driver | `repro.py` (llama-server + /v1/chat/completions, one warmup, `cache_prompt:false`, sha1 of `reasoning_content+content`) |

## 2. Reproduction — n-max 2 vs n-max 4 (b14 = 14-block delivery)

Same 5 prompts as the issue.  `none` = no speculation.

| prompt | no-spec | n-max 2 (ncols 3) | n-max 4 (ncols 5) |
|---|---|---|---|
| p0 hash map | `9216c6d1` | `a39e9459` | `4b4b99b0` |
| p1 bash backup | `5440dd9f` | `77306a04` | `6d6d8c11` |
| p2 TCP vs QUIC | `84cd0c72` | `2c476e43` | `3b1014ce` |
| p3 LRU Rust | `e591ded5` | `e3d05db2` | `c4e194e7` |
| p4 word problem | `8098106b` | `c7c88fa6` | `0697b28a` |

**All 15 outputs are distinct; n-max 2 != n-max 4 on 5/5 prompts** (reporter:
3/5).  Within-arm deterministic (`n2 p0` reproduced `a39e9459` twice, and again
with the warmup disabled — so it is not server/state warmup drift).  Divergence
is a late near-tie stylistic flip, e.g. b14 p0 at char 1393:
`...Need choose one.\n\nThe prompt: ...` (n2) vs `...Question: ...` (n4).

Also reproduced on the 4B hybrid, same day:
`/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf` — none=`ec6b39af`,
n2=`5b3f6be8`, n4=`e2a3280f` (all distinct).

## 3. Bisect — what it is NOT

Text hashes, prompts p0 / p2:

| build | n2 p0 | n4 p0 | n2 p2 | n4 p2 |
|---|---|---|---|---|
| upstream `9113cc188` (0 blocks) | `25085ea4` | `90b069b6` | `479cac73` | `0bf295ba` |
| mtpold `0994374fd` (PR #27210 pre-refresh) | `25085ea4` | `90b069b6` | `479cac73` | `0bf295ba` |
| mtpnew `d236d41a2` (PR #27210 post-refresh) | `25085ea4` | `90b069b6` | `479cac73` | `0bf295ba` |
| b12 (block 12) | `9aafdd27` | `9a536cb4` | `2c476e43` | `3b1014ce` |
| b13 (block 13) | `9aafdd27` | `9a536cb4` | `2c476e43` | `3b1014ce` |
| b14 (block 14, 14-block delivery) | `a39e9459` | `4b4b99b0` | `2c476e43` | `3b1014ce` |
| b15 (block 15 + V5 amendment) | `a39e9459` | `4b4b99b0` | `2c476e43` | `3b1014ce` |
| orig (`origin/rdna-boosts-orig`, 2026-08-16) | `a01114ae` | `de5bdc9d` | `b54afbc0` | `2619b3c1` |

Conclusions:

- **Present in stock upstream with zero fork blocks** -> not a rdna-boosts
  block regression.
- **Not the 2-day-old adaptive MTP change**: the pre-refresh (`0994374fd`) and
  post-refresh (`d236d41a2`) PR builds are **byte-identical**, and identical to
  upstream — the whole PR #27210 changeset (including the "fix rs snapshot
  copies" commit) is numerically neutral for fixed n-max 2/4.
- **Not blocks 13/14/15**: b12..b15 give identical output for p2; p0 only
  changes once at block 14 (one near-tie flip), and n2 != n4 throughout.
- The reporter's "12-block was self-consistent" is not reproduced: b12 diverges
  n2 vs n4 on both prompts (and so does the fork's old bit-identity chain).

## 4. Root cause — direct logits probe (`logits-width.cpp`)

The tool feeds one identical prefill batch, then decodes a batch of W tokens
`[t_P..t_{P+W-1}]`.  Row 0 of that batch sees the **same context** regardless of
W, so its logits must be identical for every W if the batched forward is
width-consistent.  `max|W1-W3|` = decode-vs-verify; `max|W3-W5|` = the
reporter's verify-width claim.  Row 0, P=256, f16 KV, FA auto:

| build | Qwen3.8-27B W1-W3 | W3-W5 | Gemma4-12B W1-W3 | W3-W5 |
|---|---|---|---|---|
| upstream `9113cc188` | 0.144547 | 0.144547 | 0.937368 | 1.086807 |
| b12 | 0.156145 | **0.000000** | **0.000000** | 0.000000 |
| b13 | 0.156145 | 0.000000 | 1.016227 | 0.000000 |
| b14 | 0.156145 | 0.000000 | 1.016227 | 0.000000 |
| b15 | 0.156145 | 0.000000 | 1.016227 | 0.000000 |
| orig (bit-identity chain) | **0.000000** | 0.000000 | **0.000000** | 0.000000 |

(Gemma4-12B = `/llm/models/Gemma4/12B/Q8_0/gemma-4-12b-it-Q8_0.gguf`, pure
attention control.  b14 FA-off: Qwen3.8 W1-W3 = 0.203913, Gemma4 1.079264 —
the decode/verify gap is **not** the flash-attention kernel.)

Findings:

1. **The reporter's mechanism is wrong on gfx1151.**  Verify batches at ncols 3
   and ncols 5 are **bit-identical** (`0.000000`, all rows 0..2) in every
   current fork build.  So the forward does not compute different logits for
   ncols 3 vs ncols 5.  The n2/n4 *text* split is a downstream consequence, not
   a verify-width logit difference.
2. **The real gfx1151 fault is decode (n_q=1) vs verify (n_q>=2).**  The target
   gives a different next-token distribution for the same context depending on
   whether it is fed 1 token or 3: 0.156 on Qwen3.8-27B, **1.016 on Gemma4**.
   This violates the fork's central "make decode and speculative verify
   batches bit-identical" guarantee.
3. **Block 13 breaks it for Gemma4 on gfx1151**: b12 = `0.000000`, b13 =
   `1.016227` (deterministic).  Block 13's mmvq item-split rewrite is therefore
   not bit-neutral for the n_q=1 endpoint on this arch — this contradicts the
   "output bit-identical to the 12-block build" note in AGENTS/GREEDY-PURITY
   for that case.
4. **The fork's `origin/rdna-boosts-orig` chain (2026-08-16) achieved
   W1 == W3 == W5 == 0 on both models.**  The current block chain (rebuilt at
   `9113cc188`) does not, for either the pure-attention or the hybrid model.
   That is a re-base / block-reorganisation gap in the delivery, not an
   upstream-only property.
5. **But W1==W3 alone does not make the text match**: `orig` has
   W1==W3==W5==0 and *still* splits n2 vs n4.  So the n-max-2 vs n-max-4 text
   split has a second, spec-loop-side cause (draft/verify interplay or
   recurrent-state snapshot handling) that is independent of the target
   forward.  The same is true of the `none` vs `n2` gap: MTP decode is not a
   faithful gate for the plain greedy stream even when the forward is
   width-consistent.

## 5. Interpretation

- The reporter's observation is real and worse on gfx1151 (5/5 vs 3/5).
- Their inference from GREEDY-PURITY §9 ("n-max 2 and n-max 4 would produce the
  same stream") does not follow: §9 is about block-13 vs block-12 *kernel*
  variance, and it does not claim decode == verify across n_q.
- Two distinct defects are visible on gfx1151 and both deserve their own issue:
  1. **decode != verify** target logits (0.16 hybrid / 1.02 pure-attention) in
     the current chain; the old `rdna-boosts-orig` chain had this at 0.0.
     Block 13 is a confirmed contributor for the pure-attention case.
  2. a **spec-loop** width dependence that survives even a bit-consistent
     forward (orig still splits n2 vs n4).
- Open: whether these also hold on gfx1201 (not available here).  The
  reporter's own 14-block gfx1201 n2/n4 split is consistent with (1) or (2).

## 6. Artifacts / how to rerun

- `wip/strix-halo/issue25/repro.py` — server harness (arms none/n1/n2/n4/n6/
  adaptive; env `LLAMA_BIN`, `LLAMA_MODEL`, `OUTROOT`, `FA`, `WARMUP`,
  `MAX_TOKENS`, `CTX`; `matrix` arg runs all arms x all prompts).
- `wip/strix-halo/issue25/logits-width.cpp` — the batch-width logits probe.
  Build against a tree's own libs, e.g.
  `$ROCM/lib/llvm/bin/clang++ -O2 -std=c++17 -I <tree>/include -I <tree>/ggml/include logits-width.cpp -o /tmp/lw -L <tree>/build/bin -lllama -lggml -lggml-base -Wl,-rpath,<tree>/build/bin`
- Runs under `wip/strix-halo/issue25/runs-*` (server logs, per-prompt text and
  `n_probs` dumps, `hashes.txt`).
- Scratch builds: `/home/stew675/ll25/{upstream,b12,b13,b14,mtpold,mtpnew,orig}`.
