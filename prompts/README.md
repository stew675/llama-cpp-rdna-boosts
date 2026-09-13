# Test prompts (versioned)

Fixed, hash-stable prompts used by the delivery's decode/MTP/coherence gates.

Why this directory exists: throughput, acceptance and "plain == `draft-mtp`" results are only
comparable if the prompt is identical. Pasting a prompt into an issue, or letting it drift, makes the
numbers unreproducible. Every prompt here is a committed file with a recorded size, token count and
**sha256**; a reported result is only valid against the hash it names.

## Rules

* **Never edit a shipped prompt in place.** Its sha256 is a contract. If a prompt must change, add a
  new file (bump a `-v2` suffix if it is a revision of an existing one) and record the new hash.
* Reference prompts by path *and* hash, e.g.
  `prompts/prose-rdna-boosts.txt (sha256 fabdec65…, 16074 B)`.
* Prompts are plain UTF-8 text, one per file, no BOM, `\n` line endings.

## Prompts

| file | bytes | tokens¹ | sha256 | used for |
|---|---:|---:|---|---|
| `prose-rdna-boosts.txt` | 16074 | 5298 | `fabdec65f5859e5508cc863a6e5f976706d5a770bb53eb1b406dc5aee3667727` | the issue-#30 reproduction: dense 27B `plain`/`draft-mtp n3`/`n7` throughput + acceptance + text purity, and the same gate on qwen4exp. Long English prose (~5.3 k tokens) so the model has room to generate a multi-hundred-token greedy continuation. |
| `code-python.txt` | 2025 | 533 | `53da7f2387e36baf6300b40262f1e17550fc96f2fd32a9da08ee453bc35f9b65` | an optional code-generation workload. Used to show that acceptance and MTP throughput are **prompt-content dependent**: the same build moved stock `n3` 40.66 -> 46.02 t/s purely by swapping this file in for the prose prompt. Never compare raw t/s across different prompts. |

¹ Token count on the Qwen3.8-27B tokenizer with the standard llama.cpp BPE (a different model/tokenizer
will differ — the **byte size and sha256** are the stable identity, not the token count).

The file's content is the delivery repo's own documentation; it is deliberately stable once committed
and is *not* regenerated when the docs change. If a fresh prose prompt is wanted, add a new file.

**Acceptance and MTP throughput depend on what the model is generating.** Code and other highly
predictable output accept more than generic prose, so a stock-vs-patched MTP comparison is only
meaningful between runs that use the *same* prompt file. Two builds measured on two different prompts
are not comparable, no matter how identical the binaries are.

## Usage

```sh
MODEL=/path/to/model.gguf
PROMPT=prompts/prose-rdna-boosts.txt

# verify provenance first
sha256sum "$PROMPT"

# plain decode
build/bin/llama-cli -m "$MODEL" --spec-type none -f "$PROMPT" \
  -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt \
  -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4

# MTP verify (draft depth 3 / 7); acceptance is the `draft acceptance = ...` line.
# No -md: the drafter is the MTP head built into the model GGUF (blk.<n>.nextn.*,
# *.nextn_predict_layers).  A separate -md draft model is a different drafter and
# gives different acceptance.
build/bin/llama-cli -m "$MODEL" --spec-type draft-mtp \
  --spec-draft-n-max 3 -f "$PROMPT" \
  -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt \
  -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
```

> **Drafting head:** use the model's built-in MTP head.  Unsloth's Qwen GGUFs carry it at
> `blk.<block_count-1>.nextn.*` (e.g. `blk.64.nextn.eh_proj.weight` with `nextn_predict_layers = 1`),
> and llama.cpp uses it automatically when no `-md` is given.  Do **not** pass the old standalone
> `mtp-*.gguf`: it is a different (older) drafter, and it changes acceptance and throughput.

See `benchmarks/mtp-adaptive-methodology.md` for the full gate.

### Build-validity: text purity

Run plain / `n_max 3` / `n_max 7` with the same prompt, seed and greedy sampler, and hash the generated
text with `scripts/extract-generated.py`.  A correct build is byte-identical across the three (the
multi-token verify batch must compute the same thing as single-token decode); an impure build differs.
Run with `-n 64` and **without** `-lv 4` (the verbose log interleaves statistics lines into the
generated text).

```sh
for spec in "--spec-type none" \
            "--spec-type draft-mtp --spec-draft-n-max 3" \
            "--spec-type draft-mtp --spec-draft-n-max 7"; do
  build/bin/llama-cli -m "$MODEL" $spec -f "$PROMPT" \
    -n 64 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 > run.log 2>&1
  scripts/extract-generated.py run.log
done
```

The helper strips llama-cli's backspace (`\b`) stream corrections before slicing the generated span;
a naive `sed`/`grep` slice does **not** reproduce the hashes.

When reporting a result, quote the prompt path + hash, the model file(s) (the drafting head is in the
model; no separate draft file), the build (`git rev-parse HEAD` of the stock base and the applied tree
of the patched arm) and the full command lines.  The issue-#30 reproduction report shows the shape of
that provenance block.
