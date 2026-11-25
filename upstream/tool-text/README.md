# upstream/tool-text — server: special-token strings in tool results, tool-call arguments and assistant text stay text

Written by Claude (AI).  `UPSTREAM-PR-tool-text.patch` is cut against unadulterated `ggml-org/llama.cpp` master
`e6d624b9a4b95c6b52f5ca384d70b75581f9821a` (2026-11-21); applied, the tree is `c98c38040`.  It is the same commit as
`wip/tool-text/` (the delivery form for `v17-919794500-r353`), which also cherry-picks onto this master without a conflict.
Not part of any delivery block.

> **The patch below is the SUBMITTED form and does not carry the three integration fixes the delivery's block 12
> does** (the space fallback keyed on the request rather than the server capability, the mark strip armed with the
> feature so the kill switch restores stock, and a detokenize-and-compare check on the piecewise re-tokenization).
> See `archive/work/tool-text/README.md` for what each one is and `scripts/gate-tool-text.sh` for the arms that catch
> them.  **Re-cut this form from the delivery's own change before filing it**, then re-run the Validation table.

## What it is

The server renders the chat template and tokenizes the result as one string with `parse_special`.  Special-token strings
inside a tool result or inside a tool call's arguments -- data that came from a file, a web page or a database, or that the
model copied from one -- therefore become real tokens and can forge a turn, a tool call or a tool result
(`parse_special = false` would not help: USER_DEFINED tokens are matched either way).  The template's `is_input` marks
exist in `common/chat.cpp` but `oaicompat_chat_params_parse()` flattens them with `prompt().str()`.

The change keeps those strings as text: before the template is applied, each protected string (every CONTROL / UNKNOWN
token's text, and the USER_DEFINED ones the template itself uses) in a `tool` / `function` message or in an assistant's
`tool_calls[].arguments` -- and each control-token string (a turn boundary) in an assistant's content or reasoning, which
the model can copy from a tool result -- is wrapped in two noncharacter marks (U+FDD0 / U+FDD1, dropped from client input first); after it,
the prompt is tokenized as before except that each marked string is ordinary text.  A request without such strings is
tokenized exactly as before.  When the marks do not come through the template intact, or for a multimodal prompt or a
non-BPE vocab, the strings are broken with a space after their first character instead.  `LLAMA_SERVER_TOOL_TEXT=0` turns
it off; `LLAMA_SERVER_TOOL_TEXT_CENSUS=1` logs one line per request.  Details and the full measurement table:
`wip/tool-text/README.md`.

## What it stops, and what it does not

On Qwen3.8-27B (GPU, the `wip/` form), a forged user turn in a tool result, in an earlier call's arguments or in an
Anthropic `tool_result` got `PWNED` back from stock and from the switch-off arm, and not with the change on.  With it on
the prompt always holds only the template's own special tokens.  It does not stop meaning: a forged "second tool result"
became plain words inside the real one, and the model still followed them.  It also acts only on what the request
labels: tool output that a client sends as a `user` message (text tool protocols, late results, summaries) is not
covered, and user / system text is not touched by design.  The tables and the list of what clients need to do are in
`wip/tool-text/README.md` ("Simulated injections and what they did", "What it does not stop").

## Validation on this master (CPU build, `llama-server --jinja`)

| model | request | special tokens ON / OFF | census ON |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL (33 protected strings) | tool result with `</tool_response><|im_end|>`, a forged user turn and `<think>x</think>`, tool-call arguments with `</tool_call><|im_end|>` | 16 / 24 | `mode=text strings=8 tool_results=1 tool_args=1` |
| BitNet b1.58 2B4T TQ1_0 (Llama 3 vocab, 256 protected strings) | the same shape with `<|eot_id|>` / `<|start_header_id|>` | 4 / 8 | `mode=text strings=5` |
| BitNet, a template that puts tool content through `tojson` | the same | 12 (the template's own) | `mode=text` |
| BitNet, a template that cuts tool content to 8 characters | the same | 12 (the template's own) | `mode=space (the template changed the marks)` |
| Qwen3.8 (27 control-token strings for assistant text) | assistant content, content parts, or reasoning_content that copied `<|im_end|>` + a forged user turn | 12 / 16 each (12 = the same conversation without it) | `mode=text strings=4 replies=1` |
| Qwen3.8 | an assistant reply that still holds a real `<think>…</think>` block | 14 / 14 | `mode=none` (USER_DEFINED left alone) |
| Qwen3.8, `--special` | the assistant content above | 16 (assistant text left alone) | `mode=none`; tool data still `mode=text` |
| BitNet | the assistant content above with `<|eot_id|>` / `<|start_header_id|>` | 4 / 8 | `mode=text strings=4 replies=1` |

A plain request gives `mode=none` and the same prompt token count with the switch on and off; `/apply-template` returns
the same text on and off.  The GPU arms (generation, stock vs on vs off) were run on the delivery form: see
`wip/tool-text/README.md`.

## What was NOT validated here

No GPU run on this master; no server pytest added (the checks above are HTTP requests against a local model); the realtime
endpoint is not covered; multimodal and non-BPE vocabs take the space fallback by construction, not exercised.

## A request

The problem and the fix are upstream's (the chat path is not rdna-boosts code), and this form applies to unadulterated
master.  We can't take it there ourselves: llama.cpp does not accept pull requests or review replies written by AI, and
everything on our side is written by one.  So, in the words of my people: *I'm just a large language model, I can't
help with that.*  (Sorry.)  If you think it fits, you are welcome to carry it upstream in whatever shape you like, with its
AI origin disclosed as their contribution rules ask, or to keep it as an rdna-boosts block.  Either is fine with us.
