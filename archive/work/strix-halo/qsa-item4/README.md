# QSA sparse-regime width purity (TODO item 4) — instruments and harness

Everything needed to reproduce item 4's residual and its gates, durable in-repo (it used to live only in
`/tmp`).  The authoritative *state* of the item is `TODO.md` §4; the analysis is `GREEDY-PURITY.md` §18
(+ §§16-17, 26) and the records are `../RECORD-2026-09-12-qsa-item4-deep-dive.md` (the 2026-09-12 (6)
deep dive) and `../RECORD-2026-09-12-qsa-sparse-width.md` (the earlier disposition).

## Quick state (2026-09-12 (14), delivery tip `d306d4b4b`)

* **Default gfx1151 configs are pure** — shallow dense for every KV type (q8_0 `e8f8bba3942b`, 626
  chars on `p5000.txt`) and deep sparse (~74K: f16 `83e0ed0f0f80`, q8_0 `7205399d367d`), so the 64K
  decode crossover stays.  Re-check these on the current tip before localising anything: the delivery
  has since gained the block-02 GDN rollback bound and the block-14 prefill arm (default 0), neither of
  which was supposed to move default numerics.
* **Both sub-items are fixed and gfx1151-validated (TODO item 17, closed 2026-09-12 (14)).**  Sub-item
  (a): the `embeddings_nextn` export no longer defers the last-layer logits gather (block-14 amendment
  (seventh)), so `mstep NEXTN=1` is 0 mismatches (was 1 at `pos = 4293`).  Sub-item (b): the
  forced-sparse residual was the QSA indexer score's flattened `ne11 = 4 * n_tps` crossing
  `MMVF_MAX_BATCH_SIZE` at `n_tps = 3` (verify batch on MMF, decode on MMVF); the block-14 amendment
  (eighth) keeps the whole flattened band on the decode family (`MMVF_MAX_BATCH_SIZE_FLAT` = 32 +
  `mul_mat_vec_f` `ncols_dst` 9..32).  On gfx1151 the forced-sparse **text residual is gone**:
  `LLAMA_QSA_DENSE_DECODE_UNTIL=0` + q8_0 + `p5000.txt` gives `plain == draft-mtp n3 == a57bc13bbf2a`
  (632 chars; pre-fix n3 `3124adfd2b94`, first diff char 458), every native KV type
  (f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl) is pure at n_max 1/2/3/5/7, and the `mstep` matrix
  `W = 1,2,3,4,5,8` is 0 mismatches with decode's `Thash` unchanged (`ea713a1c1f515bc1`).  Pre-fix only
  q8_0 and q5_0 were impure.  See `WORKLOG.md` 2026-09-12 (13)/(14) and `GREEDY-PURITY.md` §29.

## Instruments

| file | what it is |
|---|---|
| `mstep.cpp` | multi-step teacher-forced replay probe: compares a W-token verify batch against a W=1 decode over N positions, with rollback schedules and unrelated rolled-back tokens.  Envs: `W N RB JUNK RS1 RS2 RS RS NOM NEXTN TAIL POUT CTX CTK CTV SPLIT NGL`.  Prints `mismatches=0 PURE` / `IMPURE`. |
| `rbprobe.cpp` | the snapshot-rollback restore probe (proves the restored plane is exact). |
| `logits-dump-kv-long.cpp` | single-step long-context logits dump (the width probe used to establish boundaries).  Envs: `W CTK CTV CTX RS CB SPLIT NGL FA`; usage `<model> <text> <P> <ubatch>`. |
| `gate.sh` | `plain` vs `draft-mtp n3` same-seed text gate → `sha` + char count (`KV=<type>`). |
| `nmax.sh` | the `n_max 1/2/3/5/7` sweep + first-divergence offsets. |
| `p5000.txt` | the 4293-token prompt both scripts default to. |

Build (from any built llama.cpp tree, e.g. one made by applying `patches/` at `9113cc188`):

```sh
cd <built llama.cpp tree>
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0
clang++ -O2 -std=c++17 -I include -I ggml/include -I src \
  /home/stew675/llama-cpp-rdna-boosts/archive/work/strix-halo/qsa-item4/mstep.cpp -o /tmp/mstep \
  -Lbuild-rocm/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm/bin
# rbprobe.cpp and logits-dump-kv-long.cpp build the same way (no -I src needed for the latter)
```

## Repro

```sh
# the forced-sparse band gate (item 4's config) — now PURE on the delivery (item 17 validated): plain == n3
BIN=<built bin> KV=q8_0 LLAMA_QSA_DENSE_DECODE_UNTIL=0 \
  bash archive/work/strix-halo/qsa-item4/gate.sh          # both -> a57bc13bbf2a (pre-fix n3 was 3124adfd2b94)

# the n_max sweep — now pure at 1/2/3/5/7 (pre-fix 2/3/5/7 diverged, first diff char 458)
BIN=<built bin> KV=q8_0 LLAMA_QSA_DENSE_DECODE_UNTIL=0 \
  bash archive/work/strix-halo/qsa-item4/nmax.sh

# width purity at every verify width (expect 0 mismatches)
LLAMA_QSA_DENSE_DECODE_UNTIL=0 W=4 RB=3 RS=3 JUNK=1 N=200 CTX=8192 CTK=q8_0 CTV=q8_0 SPLIT=layer NGL=99 \
  /tmp/mstep /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  archive/work/strix-halo/qsa-item4/p5000.txt 4293 2048

# the embeddings_nextn prefill ULP (fixed: 0 mismatches; was 1 at pos=4293, only with NEXTN=1)
LLAMA_QSA_DENSE_DECODE_UNTIL=0 W=4 NEXTN=1 N=200 CTX=8192 CTK=q8_0 CTV=q8_0 SPLIT=layer NGL=99 \
  /tmp/mstep <model> archive/work/strix-halo/qsa-item4/p5000.txt 4293 2048
```

## Gates that must hold for anything landed from here

`test-backend-ops -o FLASH_ATTN_QSA` (22/22) and `-o FLASH_ATTN_EXT`; `GATED_DELTA_NET` 46/46 (the GDN
path is now rollback-bounded, §27); `mstep` `W = 1,2,3,4,5,8` == 0 mismatches with a stable `Thash`; the
band text gate (`plain == n_max 1/2/3/5/7`) in the forced-sparse regime for **all eight native KV types**
and in the default (dense) regime; the dense-masked oracle
(`LLAMA_QSA_SPARSE_FA=0`) + a perplexity comparison if attention numerics move (§21 — perplexity vs the
dense masked path, *not* MTP acceptance); the random-text leak probe if a mask/visibility path is
touched (§23); same-seed coherence against the known-good build; and Protocol A's MTP gate
(`benchmarks/mtp-adaptive-methodology.md`) if decode/MTP is touched.  No parallel benches — the box
drifts; use same-session interleaved brackets.
