# The in-place read was never the carrier: a stale expert-cache ALIAS resolved a sibling ROLE's arena (`TODO` #335 + #336, r298)

**Question.**  r297 left two items on the record, both written as defects of the *in-place prefill read*
(`MOE_EXPERT_PREFILL_IN_PLACE`): a degenerate output under VRAM pressure (`#335`, blamed on a captured graph
replaying an evicted arena pointer) and a coherent near-tie between the in-place and staged arms (`#336`, "the two
arms read the same bytes, so a logit difference between them is a bug", magnitude owed).  This record answers three
questions:

1. Is the in-place **read** numerically different from the staged read at all?
2. If not, what does move the text — and does one cause explain #335's whole workaround matrix?
3. What is #336's owed magnitude?

**Answers in one line each.**

1. **No.**  `scripts/gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` on a Flash-Next arm that really
   serves in place (143 identity tables, `get_ok` 1280 over two 2048-token chunks) reads **mean KLD 0.000000 /
   same-top-p 100.000 %** — before *and* after the fix.  Shown per dispatch, the two arms choose the **same kernel
   family for the same op with the same shape** (8022 identical `mul_mat_id` dispatches: `mmq T=512` for the prefill,
   `mmvq T<=3` for decode, `mmq T=42` for the MTP prompt-ingest tail) and produce **the same hashes** for 276 of them.
2. A **stale per-pointer alias** in the expert cache resolves a table by device + layer only, so a scheduler copy
   buffer reused for a *sibling role of the same layer* serves the neighbour's **arena** — `blk.43.ffn_up_exps` was
   read out of `blk.43.ffn_down_exps`'s bytes (a NaN, then `-2.1e7`), which is one near-tie flip on the whole-prompt
   arm and, deeper into a run with a captured graph replaying the wrong pointer, a degenerate continuation.  The fix
   is a role check on the alias; it collapses both of r297's pairs.
3. **0.000000 KLD, 100 % top-p** (the number `#336` asked for).

## 1. The measurement r297 could not make (and why it could not)

r297 recorded that `llama-perplexity` cannot load a drafter, and that the state producing the text difference only
existed *with* one, so both priors read KLD 0.000000 / top-p 100 % at 99.65 % residency and the magnitude was owed.
The premise was wrong: the difference reproduces on a **drafter-less** instrument too, because it never needed a
drafter — it needed the cache to serve a table from an arena:

| arm (Flash-Next GSQ-RCO IQ3_XXS, 2 devices `-sm layer`, `-ncmoe 48`, `--ab MOE_EXPERT_PREFILL_IN_PLACE=0`) | mean KLD | same-top-p |
|---|---|---|
| `--ctx 2048 --chunks 7` on `prompts/kld-corpus-35b-in-domain.txt`, base PPL 1.9143 | **0.000000** | **100.000 %** |
| the same on the pre-fix build (r297 + this session's first instrumentation) | 0.000000 | 100.000 % |

So the in-place read is a **base-pointer swap**: same bytes, same ids, same kernel, same arithmetic.  The claim in
`ENVIRONMENT.md` ("the two arms read the same bytes, so a logit difference between them is a bug") stands - the bug
was just not in the read.

## 2. What does move the text: the alias role gap

`ggml_backend_sched_copy_input` hands the cache a **copy tensor** (`input_cpy`) for each routed expert table, and the
cache resolves "which table is this?" from that tensor's *address* (`g_alias_to_id`, written by whichever hook last
took that buffer over).  The graph allocator recycles those addresses, so the lookup is validated: **device** and
**layer**.  A layer's `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` share both, and a layer's routed ops are
issued back to back — so the guard cannot see the one reuse that matters.

The consumer's redirect and the take-over hooks resolve the table by *different* keys, which is what turns the gap
into a wrong answer rather than a slow one:

* the hooks (`moe_cache_take_over`, `moe_cache_update_host`) resolve the **host master**, so they always find the
  right role.  `blk.43.ffn_up_exps` is **not** identity (`slots 512/512`, `identity=0`, its decode remap is 10 wide)
  ⇒ the take-over **declines** ⇒ the scheduler fills `input_cpy` with the right bytes;
* the consumer (`ggml_cuda_mul_mat_id` → `moe_cache_get_table`) resolves the **copy pointer** ⇒ the stale alias names
  `blk.43.ffn_down_exps` (`identity=1`, device 1, layer 43) ⇒ the redirect **serves down's arena in place** and `up`
  is computed from `down`'s weights.

Evidence, per `mul_mat_id` dispatch (a temporary per-op hash of the result; `MOE_EXPERT_OP_DUMP`):

```
OD 275 Q ROCm1#blk.42.ffn_down_exps T=42 h=058f7acd4b2d1946 v0=0.0558644     <- identical in both arms
OD 276 Q ROCm1#blk.43.ffn_gate_exps T=42 h=b48d93b5278925b8 v0=-0.415274     <- identical in both arms
OD 277 Q ROCm1#blk.43.ffn_up_exps   T=42 h=889e47de1b15f79a v0=nan          <- in place: NaN
OD 278 Q ROCm1#blk.43.ffn_down_exps T=42 h=dcfe2bd6920f70c1 v0=-2.12462e+07 <- and the next op is poisoned
```

and the state at that op (a name-keyed lookup, i.e. the op's own table): `layer=43 slots=512 nex=512 idn=0
remap_n=10 premap=0 band=16` — the served table was a *different* one.  With the fix, the guard's first refusal in
the same run names exactly it:

```
alias_find_checked: ignoring a stale MoE-cache alias
  (table device 1 layer 43 role blk.43.ffn_down_exps.weight, op device 1 layer 43); 1 refused so far
```

**One refusal in a 24 K-token prompt**, i.e. the check costs nothing (the fallback is the existing semantic key
`(layer, sem_role, device)`), and the two arms collapse:

| arm (Flash-Next + MTP sidecar, 2 devices `-sm layer`, `-c 32768`, `--seed 42 --temp 0`, 64 tokens) | pre-fix | r298 |
|---|---|---|
| `MOE_EXPERT_PREFILL_IN_PLACE=1` (default), prior absent | 837 chars `f40ae789f219` | **835 chars `7a3474cbe5f1`** |
| `MOE_EXPERT_PREFILL_IN_PLACE=0`, prior absent | 835 chars `7a3474cbe5f1` | 835 chars `7a3474cbe5f1` |
| either arm, prior present (`MOE_EXPERT_HEATMAP=load`) | 835 chars `7a3474cbe5f1` | 835 chars `7a3474cbe5f1` |
| either arm, no drafter | 835 chars `7a3474cbe5f1` | 835 chars `7a3474cbe5f1` |

`GGML_MOE_DEVREMAP=0` and `GGML_CUDA_DISABLE_GRAPHS=1` do **not** collapse the pre-fix pair (both still read
`f40ae789f219`), and `GGML_SCHED_DEVGATHER=0` does not either — the carrier is the alias, not the remap, the gather or
a capture.

### How this explains `#335`'s whole workaround matrix

r297's pressure arm (`-c 16384`, prose prompt, prior loaded) read `82 chars a753000294c9` (degenerate) in place
against `328 chars 7c5a512a8447` staged, and was coherent with each of: `GGML_CUDA_DISABLE_GRAPHS=1`
(`04ca2590aebf`), `GGML_CUDA_SLAB=0` (`7c5a512a8447`), `MOE_EXPERT_CACHE_MIB=20000` (`04ca2590aebf`), a smaller `-c`,
and dropping the drafter.  Every one of those is a **less-stale-alias** arm:

* `DISABLE_GRAPHS=1` — a capture bakes the *arena pointer* it served into the launch parameters; a replay repeats
  the wrong table for the rest of the graph's life, which is exactly `#335`'s "captured graph replays an arena
  pointer" signature (and why removing captures removes the symptom);
* `SLAB=0` / `MIB=20000` / a smaller `-c` — no eviction ⇒ no stand-down/re-arm ⇒ the layer's tables are allocated
  once, in one order, and the alias churn that produces the sibling resolution does not happen;
* `PREFILL_IN_PLACE=0` — the staged arms read the scheduler's copy (which the declining hook filled correctly), and
  the redirect never runs;
* no drafter — no `T=42` MTP prompt-ingest batch, which is the width where the first wrong read appeared.

And it explains why r297's arena validation was clean: `MOE_EXPERT_CACHE_VALIDATE=3` byte-compares each slot against
its host master and found **0 MISMATCH** over 157 sampled dispatches, because *the arena was right* — the consumer was
reading the wrong table's slots.  A validator that checks the bytes cannot see a wrong pointer.

**Not reproduced here:** `#335`'s exact degenerate text.  On this box the pressure arm is coherent pre-fix
(`328 chars 7c5a512a8447`), and its wrong-role refusal is at layer 30 (`table device 1 layer 30 role
blk.30.ffn_up_exps.weight`), not layers 43-47.  The *hazard* is present in that arm; its severity is one near-tie
(logits shift, one token flips) unless the wrong pointer is baked into a capture, which is the state `#335` was
recorded from and which this box did not enter.

## 3. The repair path (a second, unrelated hazard, closed as hardening)

While chasing the above, the take-over contract got an instrument and a repair: a hook that returns true tells the
scheduler "do not fill `input_cpy`", and the slab can still take the table's arena **between** that hook and the
consumer's launch (an op's workspace allocation can move the work boundary from inside a graph).  The consumer would
then read a copy nobody wrote.

* `moe_cache_repair_copy()` (called from the redirect's decline path) re-fills the skipped buffer from the host master
  when the table still holds a promise for **that** buffer, on the caller's compute stream.  It is inert otherwise.
* `MOE_EXPERT_TAKEOVER_FAULT=1` (diagnostic) makes the identity take-over break its own promise, so the path is
  exercised rather than assumed: the arm's text is **byte-identical** to both no-fault arms (`595 chars
  c0e7e91d84c4` at `-c 8192` on the kld corpus) while it re-fills 343.8-450.0 MiB per broken promise.
* The exit census now prints `take-over promises broken=N repaired=N`; ordinary runs (including the pressure arm,
  which evicts mid-prompt) read **0 / 0**, i.e. this hazard needs a boundary move inside one graph and the eval-time
  re-resolve normally catches the stand-down before the consumer runs.

## What shipped

Blocks **09** (the cache engine: `t.sem`, the role-aware `alias_find_checked` with its in-place comparer, the promise
field), **10** (`moe_cache_repair_copy`, the counters and the census line, the redirect's served/declined split and
its repair call) and **15** (the promise in the GTT prefill in-place path).  Two environment variables, both
diagnostic: `MOE_EXPERT_ALIAS_ROLE_CHECK` (default 1; `0` is the pre-r298 arm) and `MOE_EXPERT_TAKEOVER_FAULT`
(default off).

**Gates** (r298, gfx1201, ROCm 10.1.0):

* coherence: Qwen3.5-4B Q8_0, 20 tokens, 3 devices `-sm tensor` → `100 chars 1c5d32ac537d` (the documented reference);
* `gate-host-pool.sh` → **56/56**, nine arms byte-identical (`ref`/`dio`/`nodisk`/`fail`/`late`/`pinlate`/`pin0` =
  `a99ccbd97ecb`, `tsplit`/`tsplitnosink` = `8c39335bbceb`);
* `gate-prefill-logits.sh` canonical (27B wikitext, base re-recorded from this build) → 0.000000 / 100.000 %;
* `gate-prefill-logits.sh --ab "MOE_EXPERT_PREFILL_IN_PLACE=0"` Flash-Next host-expert arm → 0.000000 / 100.000 %;
* `gate-per-block-build.sh --changed-from 1347f895a --fast` → 5/5 (09, 10, 12, 15, 0300);
* `validate-set.sh` → GREEN (17 default + 2 optional, cumulative optional tree, Windows portability lint);
* throughput of the role check, 35B-A3B Q8_0 `-ncmoe 48` 2 devices `tg128`, interleaved: `=0` 62.61 / 64.26 t/s
  against the default 64.05 / 64.92 t/s (inside the run spread; the check is one in-place string compare).
