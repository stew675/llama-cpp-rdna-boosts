#!/usr/bin/env python3
"""Extract the generated text from a llama-cli log and print its hash.

Usage:
    scripts/extract-generated.py LOG [--text]

Prints `<N> chars sha=<first 12 hex of sha256>` (the form used in the delivery's
reported results), or the generated text itself with --text.

Why this exists: llama-cli's terminal streaming emits backspace (`\\b`) corrections,
so a naive `sed`/`grep` slice of the log does not reproduce the text the model
actually generated.  This strips the backspaces first, then takes the span between
the prompt echo (`> `) and the `[ Prompt: ... ]` footer, exactly as the delivery's
text-purity gate does.

Run llama-cli with `--single-turn --no-display-prompt` and, for this check, WITHOUT
`-lv 4` (the verbose log interleaves statistics lines into the generated text).
"""

import hashlib
import sys


def extract(path: str) -> str:
    s = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for ch in s:
        if ch == "\b":
            if out:
                out.pop()
        else:
            out.append(ch)
    lines = "".join(out).split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("> ")), None)
    if i is None:
        raise SystemExit("extract-generated: no prompt echo ('> ') found in " + path)
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("[ Prompt:")), len(lines))
    return "\n".join(lines[i + 1:j]).strip()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    text = extract(args[0])
    if "--text" in sys.argv:
        print(text)
    else:
        print("%d chars sha=%s" % (len(text), hashlib.sha256(text.encode()).hexdigest()[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
