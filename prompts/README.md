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

¹ Token count on the Qwen3.8-27B tokenizer with the standard llama.cpp BPE (a different model/tokenizer
will differ — the **byte size and sha256** are the stable identity, not the token count).

The file's content is the delivery repo's own documentation; it is deliberately stable once committed
and is *not* regenerated when the docs change. If a fresh prose prompt is wanted, add a new file.

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

# MTP verify (draft depth 3 / 7); acceptance is the `draft acceptance = …` line
build/bin/llama-cli -m "$MODEL" --spec-type draft-mtp -md DRAFT.gguf \
  --spec-draft-n-max 3 -f "$PROMPT" \
  -n 128 --seed 42 --temp 0 --single-turn --no-display-prompt \
  -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 -lv 4
```

See `benchmarks/mtp-adaptive-methodology.md` for the full gate.

### Build-validity: text purity

Run plain / `n_max 3` / `n_max 7` with the same prompt, seed and greedy sampler, and hash the generated
text with `scripts/extract-generated.py`.  A correct build is byte-identical across the three (the
multi-token verify batch must compute the same thing as single-token decode); an impure build differs.
Run with `-n 64` and **without** `-lv 4` (the verbose log interleaves statistics lines into the
generated text).

```sh
for spec in "--spec-type none" \
            "--spec-type draft-mtp -md DRAFT.gguf --spec-draft-n-max 3" \
            "--spec-type draft-mtp -md DRAFT.gguf --spec-draft-n-max 7"; do
  build/bin/llama-cli -m "$MODEL" $spec -f "$PROMPT" \
    -n 64 --seed 42 --temp 0 --single-turn --no-display-prompt \
    -c 32768 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa auto -ngl 99 > run.log 2>&1
  scripts/extract-generated.py run.log
done
```

The helper strips llama-cli's backspace (`\b`) stream corrections before slicing the generated span;
a naive `sed`/`grep` slice does **not** reproduce the hashes.

When reporting a result, quote the prompt path + hash, the model/draft files, the build
(`git rev-parse HEAD` of the stock base and the applied tree of the patched arm) and the full command
lines — see the issue-#30 reproduction report for the shape of that provenance block.
