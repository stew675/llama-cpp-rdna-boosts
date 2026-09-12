# 2026-09-12 — TODO item 9 resolved: the configurable QSA prefill arm + the device-query arm gate

Delivery change: **block-14 amendment (sixth)**, landed in `patches/0014`.  New canonical tip
**`890a9c5b1`**, tree **`0edf654cdea653b9969f866977a541ee4429f846`** (was `13af95ac1` /
`f4791066f4a582316b1ca95f51c96cd10b905ef7`).  Beta block-15 re-cut 14th on the new base: base
`890a9c5b1` -> beta tip **`86c7df1f5`**, tree **`66f0762a2ec19cbc34b1842d1b5984bb82ecec45`**, patch
3 819 lines.

The session's first pass got 9(a) **wrong** and is recorded here because the mistake is instructive:
it set the prefill default from a whole-prompt `llama-bench` A/B and from a parenthetical in
`patches/README.md`, against the documented policy.  The maintainer corrected it; the shipped default
is `0` (QSA prefill always) on every arch and split, and the replacement section below is what the
delivery actually contains.

## 9(a) The prefill arm is now configurable — and its default is the documented policy

Item 9's premise was right about the *mechanism*: prefill took the indexer top-k selection from the
`indexer_top_k + r - 1` (2051) shortcut width up, unconditionally, with no way to ask for the dense
attend at shallow depths, because only the *decode* axis was depth-configurable
(`qsa_dense_decode_until`).  That is now fixed: `qsa_dense_prefill_until` exists (env
`LLAMA_QSA_DENSE_PREFILL_UNTIL`, `K/M/G` suffixes, `0` disables the arm), and a prefill ubatch
(`n_tokens > QSA_DECODE_BAND`) whose `n_kv` is still below the threshold attends dense while storing
the indexer keys, exactly as the shortcut arm does, so the sparse path takes over seamlessly above it.

**The default is `0` = QSA prefill always, on every arch and split — the ARCH POLICY.**  The
documented decision is in two places:

* `beta/qwen4exp/README.md`: *"**ARCH POLICY (2026-09-07 crossover tables)**: decode uses the dense
  attend below a per-arch depth and QSA above; **prefill is always QSA**."*
* `wip/archive/qwen4exp/discovery/2026-09-07-qsa-dense-crossover-tables-soar-halo.md`: *"**Soar: QSA
  for prefill ALWAYS (wins from ~8K, monotonically to +181 % @160K)**; dense for decode ALWAYS"* and
  *"Halo: QSA for prefill always (already +14 % @16K, grows to +169 %)"*.

So there is **no** dense-prefill regime to default to on either arch, and the arm ships as an opt-in
A/B knob only.  The delivery's default behaviour is therefore **byte-identical to the pre-amendment
build** (verified below), and no recorded reference hash moves.

### Why the first pass's "crossover" was not usable evidence

The whole-prompt A/B below is the shape the 2026-09-07 record explicitly rejects: *"the old \"dense
wins prefill at 30K\" record is obsolete (predates the QSA prefill improvements; also a
**non-comparable whole-prompt llama-cli banner**)"*.  The 2026-09-07 tables are `pp2048` measured **at
depth**; a whole-prompt run at depth 0 measures a different thing (it also folds in the prompt-length
dependent prefill shape).  Recorded for completeness, **not** as a default-setting measurement:

`llama-bench -fa on -sm layer -ctk/-ctv f16 -b/-ub 2048`, whole-prompt pp, arm forced with
`LLAMA_QSA_DENSE_PREFILL_UNTIL=1000000000` vs `=0`, r2, interleaved in one session, gfx1151 IQ4_XS:

| pp | sparse (`=0`, the shipped default) | dense arm forced | winner |
|---|---|---|---|
| 2048 | 754.50 / 744.51 | 744.45 / 746.07 | wash (the 2051 shortcut already makes it dense) |
| 4096 | 740.79 / 734.97 | 754.61 / 754.62 | dense +2.4 % |
| 8192 | 737.79 / 732.83 | 747.33 / 746.16 | dense +1.6 % |
| 16384 | 735.09 / 732.36 | 712.36 / 713.70 | sparse +3.0 % |
| 32768 | 709.53 / 708.90 | 604.54 / 604.02 | sparse +17.4 % |

and the arm-*policy* form of it (the arm live at 8192 for comparison with `=0`, same session, r2):
pp4096 751.92/742.13 vs 724.03/723.22, pp8192 751.67/750.24 vs 731.79/730.27, pp16384 742.04/741.30
vs 731.67/731.24, pp32768 711.75/712.31 vs 707.68/707.36.  Anyone re-opening this should measure it in
the comparable shape (pp at depth) first.

## 9(b) The device query instead of the mirrored type list

`qsa_kv_native` mirrored `ggml_cuda_flash_attn_qsa_supported()`'s type list, kept in lockstep by a
comment — and a stale mirror is not a fallback but an abort in the meta splitter (the 2026-09-11 third
amendment).  The three options the item listed, decided on evidence:

* **(i) a `LLM_FUSED_OP_FLASH_ATTN_QSA` probe — structurally impossible.**  `resolve_fused_ops()`
  builds a *reserve* graph and looks for fused nodes of the op.  A QSA node only exists when
  `n_kv > indexer_top_k + r - 1` (2051), and at reserve time the cache is empty, so the probe graph
  contains no QSA node at all and the gate would answer "enabled" for every cache type.  Forcing it
  would need a > 2051-token probe graph (inflating the compute buffer) *and* a "probing" flag in the
  graph, for a device-mismatch heuristic that (unlike the query) cannot express "every
  tensor-parallel device must accept this op".
* **(iii) keep the list** — rejected: it is the duplication the item is about, and it silently
  accepts an unsupported head size (see below).
* **(ii) a corrected predicate — landed.**  `qsa_op_supported()` builds a minimal probe tensor
  (real head size `D` from `hparams.n_embd_head_k(il)`, the real K/V type, F32 Q, I32 idx, F16 mask;
  the predicate reads only the source types and `D`) and asks
  `ggml_backend_dev_supports_op(model.dev_layer(il), probe)`.  Under `-sm tensor`,
  `model.dev_layer(il)` is the **Meta** device, whose `supports_op()` is
  `std::all_of(sub_devices, supports_op)` — i.e. the query *is* the meta-split safety condition.
  Cost measured: **0.112 us/call** (~3.3 us per graph build at 30 calls).

Validation, `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib qsa-support-probe 128`:

```
kv type      device   legacy   verdict        head_dim 64/128/256 f16: device=yes
f32          no       no       agree          head_dim 80        f16: device=no   <- legacy said yes
f16          yes      yes      agree
bf16         yes      yes      agree          mismatches vs the legacy list: 0
q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl  yes yes agree
q6_K/q3_K/q4_K/iq4_xs  no no agree
```

plus a same-seed text A/B against the pre-amendment build: **byte-identical**.

## Gates (all on the amended delivery tip, gfx1151)

| gate | result |
|---|---|
| default behaviour vs the pre-amendment build | **byte-identical** — f16 `0fc4910d5824` (632 chars) and q8_0 `e8f8bba3942b` (626 chars, the recorded pre-amendment shallow q8_0 value), `plain == draft-mtp n_max 3 == n_max 7` in both |
| strict apply, fresh clone at `9113cc188` + `apply-all.sh` | **15/15**, 0 whitespace warnings, applied tree == canonical `0edf654cdea653b9969f866977a541ee4429f846` |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** on ROCm0 |
| predicate equivalence (`qsa-support-probe`) | 0 mismatches vs the old list; `D=80` now rejected |
| prefill arm A/B (`=0` vs `=1000000000`) | record of the knob only (table above); **not** the default |
| beta re-cut (14th) | builds clean, `FLASH_ATTN_QSA` 22/22, all four gate combos + `draft-mtp n_max 3` byte-identical (`0fc4910d5824`) = the delivery's value, patch round-trips |

## Repro

```sh
cd /home/stew675/ll25/verify            # or ll25/beta15 for the beta tree
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1151/lib:$LD_LIBRARY_PATH
M=/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf

# the predicate table
clang++ -O2 -std=c++17 -I include -I ggml/include \
  /home/stew675/llama-cpp-rdna-boosts/wip/strix-halo/qsa-item9/qsa-support-probe.cpp \
  -L$PWD/build-rocm/bin -lggml -lggml-base -Wl,-rpath,$PWD/build-rocm/bin -o /tmp/qsa-support-probe
/tmp/qsa-support-probe 128

# the opt-in knob (whole-prompt; see the caveat above before reading anything into it)
LLAMA_QSA_DENSE_PREFILL_UNTIL=0 ./build-rocm/bin/llama-bench -m $M -ngl 99 -fa on -sm layer \
  -ctk f16 -ctv f16 -b 2048 -ub 2048 -p 4096,8192,16384,32768 -r 2

# purity text (KV=<type>); the shipped default must reproduce the pre-amendment hashes
TAG=x KV=q8_0 bash /tmp/g1151-item4/gate.sh
```
