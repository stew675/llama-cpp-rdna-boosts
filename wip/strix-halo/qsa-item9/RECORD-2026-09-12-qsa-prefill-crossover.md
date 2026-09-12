# 2026-09-12 — TODO item 9 resolved: the QSA prefill crossover + the device-query arm gate

Delivery change: **block-14 amendment (sixth)**, landed in `patches/0014`.  New canonical tip
**`15e3bdcbd`**, tree **`86b6cce726b0f0f2f3935781ed782659529b38fe`** (was `13af95ac1` /
`f4791066f4a582316b1ca95f51c96cd10b905ef7`).  Beta block-15 re-cut 13th on the new base: base
`15e3bdcbd` -> beta tip **`3d9b578c5`**, tree **`b214b3d9d42e294fb351a58be7f05b10fe1d9a04`**, patch
3 808 lines.

Everything below was measured on the dev box (gfx1151 / Strix Halo, 1 GPU, ROCm 7.14,
`build-rocm`) unless the row says otherwise.  Instruments: `qsa-item9/qsa-support-probe.cpp`
(the predicate table), `/tmp/g1151-item4/gate.sh` (plain-vs-MTP text, `textgen.py`), the
item-4 `mstep` width probe, `llama-bench`, `llama-perplexity`, `test-backend-ops`.

## 9(a) The prefill half of the arch policy

Item 9's premise: the decode crossover is depth-configurable (`qsa_dense_decode_until`) but prefill
runs the indexer top-k selection from the 2051 shortcut width up, unconditionally — while the
recorded 2026-09-07 tables show `dense prefill` winning at short depths.  Implemented as
`qsa_dense_prefill_until` with per-(split, arch) defaults, plus `LLAMA_QSA_DENSE_PREFILL_UNTIL` for
A/B (`0` = the pre-amendment regime).  The arm is the *dense* one the shortcut and the decode gate
already use (`build_qsa_store_k` + the normal dense attention), so the indexer keys are still stored
and the sparse path takes over seamlessly above the threshold; it is disjoint from the decode arm by
`n_tokens > QSA_DECODE_BAND`, so a W=1 decode and a W-token verify still take one arm.

### The crossover (all-dense vs all-sparse, one session, r2)

`llama-bench -fa on -sm layer -ctk/-ctv f16 -b/-ub 2048 -p 2048,4096,8192,16384,32768` (whole-prompt
pp; the dense arm forced with `LLAMA_QSA_DENSE_PREFILL_UNTIL=1000000000`, the sparse one with `=0`):

| pp | sparse (t/s) | dense (t/s) | winner |
|---|---|---|---|
| 2048 | 754.50 / 744.51 | 744.45 / 746.07 | wash (the 2051 shortcut already makes it dense) |
| 4096 | 740.79 / 734.97 | 754.61 / 754.62 | **dense +2.4 %** |
| 8192 | 737.79 / 732.83 | 747.33 / 746.16 | **dense +1.6 %** |
| 16384 | 735.09 / 732.36 | 712.36 / 713.70 | **sparse +3.0 %** |
| 32768 | 709.53 / 708.90 | 604.54 / 604.02 | **sparse +17.4 %** |

Crossover ~8-16K on gfx1151 — materially the same place the recorded 3x R9700 tensor table puts it
(dense +4.9 % at pp8192, parity at pp16384, sparse +14.5 % at pp32768).  Note the recorded tables
compare `LLAMA_QSA_OFF=1` (no indexer store at all), so they *bound* this arm rather than equal it;
they were the reason the item existed and they are what the tensor default is set from.

### The policy itself (new default 8192 vs `=0`, same session, r2)

| pp | default `8192` | old regime | delta |
|---|---|---|---|
| 4096 | 751.92 / 742.13 | 724.03 / 723.22 | **+3.2 %** |
| 8192 | 751.67 / 750.24 | 731.79 / 730.27 | **+2.7 %** |
| 16384 | 742.04 / 741.30 | 731.67 / 731.24 | **+1.4 %** |
| 32768 | 711.75 / 712.31 | 707.68 / 707.36 | **+0.6 %** |

A strict win at every measured pp, including above the crossover: the arm is **self-limiting** (only
prefill chunks whose `n_kv` is still below the threshold take it, i.e. the shallow ones, and those
are exactly the chunks that measured faster dense), so it never repays the deep chunks' sparse
advantage.

### The quality side (the reason this is not just a speed knob)

The indexer top-k selection is a lossy approximation of attention; below the crossover the old
default used it anyway.  `llama-perplexity --chunks 8 -c 4096 -b 4096 -ub 512 -f /tmp/huge.txt`
(f16 KV, 1 GPU):

| arm | PPL |
|---|---|
| new default (full dense below 8192) | **23.2727 +/- 0.72715** |
| `LLAMA_QSA_OFF=1` (no-indexer full-dense reference) | **23.2727 +/- 0.72715** |
| old regime (selected sparse) | 24.7142 +/- 0.79150 |
| `LLAMA_QSA_DENSE_PREFILL_UNTIL=0 LLAMA_QSA_SPARSE_FA=0` (the selection computed densely) | 22.2844 +/- 0.71214 |

The new default is **bit-for-bit the no-indexer reference** at this depth: below 8192 the sparse
approximation is simply not used any more, which removes a **6.2 %** PPL penalty the old default
carried there.  Caveat on the last row: the documented oracle text (`/tmp/qa-text.txt`, PPL ~6.5) is
absent on this box, so the substitute text (PPL ~23, much harder) cannot be used for the
sparse-vs-its-own-dense-masked parity claim recorded on 2026-09-11 (6.5267 vs 6.5306); on the
substitute text the two computations of the *same* selected cell set read 24.71 vs 22.28.  That
sparse-path comparison is orthogonal to this change (the patch only adds an arm and replaces a
boolean predicate; the two PPL arms differ solely by `LLAMA_QSA_SPARSE_FA`) and is left as an
observation, not a claim.

## 9(b) The device query instead of the mirrored type list

`qsa_kv_native` mirrored `ggml_cuda_flash_attn_qsa_supported()`'s type list, kept in lockstep by a
comment - and a stale mirror is not a fallback but an abort in the meta splitter.  The three options
the item listed, decided on evidence:

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

plus a same-seed text A/B against the pre-amendment build: **byte-identical** (f16, 4293-token
prompt, `plain` and `draft-mtp n_max 3` both `0fc4910d5824` at 632 chars, i.e. with the new prefill
arm disabled via `=0` so only the predicate differs).

## Gates (all on the amended delivery tip, gfx1151)

| gate | result |
|---|---|
| strict apply, fresh clone at `9113cc188` + `apply-all.sh` | **15/15**, 0 whitespace warnings, applied tree == canonical `86b6cce726b0f0f2f3935781ed782659529b38fe` |
| `test-backend-ops -o FLASH_ATTN_QSA` | **22/22** on ROCm0 |
| `test-backend-ops -o FLASH_ATTN_EXT` | passed |
| text purity, q8_0 KV, `plain` vs `n_max 1/2/3/5/7` | all **685 chars `93deb49ca115`** (whole band pure) |
| text purity, f16 KV, new default | `plain == n3` = `d10a6c561b67` (652 chars) |
| the same with `LLAMA_QSA_DENSE_PREFILL_UNTIL=0` (old regime) | `plain == n3` = `0fc4910d5824` (632 chars) |
| width probe `mstep` W=4 (`RB=3 RS=3 JUNK=1 N=200`, 4293-token prompt, f16) | **0 mismatches**, both regimes |
| width probe W=8, new default vs old regime | **38 mismatches, identical position lists** (same positions, first at pos 4299) -> **pre-existing**, not from this change |
| perplexity, new default vs `LLAMA_QSA_OFF=1` | **identical** (23.2727), see above |
| MTP gate (Protocol A), q8_0, 4293-token prompt | `n_max 3` 28.3 t/s vs plain 24.5 (+16 %); pos-1 acceptance **0.667**, aggregate 0.417 |
| prefill perf, new default vs old regime | +3.2 % pp4096, +2.7 % pp8192, +1.4 % pp16384, +0.6 % pp32768 |

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

# the crossover / policy A/B
LLAMA_QSA_DENSE_PREFILL_UNTIL=0 ./build-rocm/bin/llama-bench -m $M -ngl 99 -fa on -sm layer \
  -ctk f16 -ctv f16 -b 2048 -ub 2048 -p 4096,8192,16384,32768 -r 2

# quality
./build-rocm/bin/llama-perplexity -m $M -f /tmp/huge.txt --chunks 8 -c 4096 -b 4096 -ub 512 \
  -ctk f16 -ctv f16 -fa auto -ngl all -sm layer -mg 0        # + LLAMA_QSA_OFF=1 for the reference

# purity text (KV=<type>)
TAG=x KV=q8_0 bash /tmp/g1151-item4/gate.sh
```
