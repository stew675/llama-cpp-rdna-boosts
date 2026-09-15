#!/usr/bin/env bash
# Fail if any tracked path cannot be checked out on Windows.
#
# Why this exists: git on Windows refuses to create a path containing a
# character the Win32 API forbids (`< > : " | ? *`, control characters), a
# component ending in a space or a dot, or a reserved device name (CON, NUL,
# COM1, ...).  The failure happens at `git checkout`, so a *single* such path
# makes the whole repository unclonable there -- reported as issue #36, a patch
# file whose name carried a `:` (`...qwen4exp:-PROBE-...`).  Nothing in this
# repo needs a colon in a filename, so the cheapest fix is to never let one in.
#
# Usage: ./check-windows-paths.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

python3 - <<'PY'
import re
import subprocess
import sys

tracked = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
paths = [p.decode("utf-8", "surrogateescape") for p in tracked.split(b"\x00") if p]

illegal   = re.compile(r'[<>:"|?*\x00-\x1f]')
reserved  = re.compile(r'(^|/)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|/|$)')

offenders = []
for p in paths:
    why = []
    if illegal.search(p):
        why.append("illegal character " + ", ".join(sorted(set(illegal.findall(p)))))
    if any(seg.endswith(" ") or seg.endswith(".") for seg in p.split("/")):
        why.append("component ends with a space or a dot")
    if reserved.search(p):
        why.append("reserved device name")
    if why:
        offenders.append((p, why))

longest = max(paths, key=len) if paths else ""

if offenders:
    print("ERROR: %d tracked path(s) are not checkable out on Windows:" % len(offenders), file=sys.stderr)
    for p, why in offenders:
        print("  %s\n      (%s)" % (p, "; ".join(why)), file=sys.stderr)
    print("\nRename the path (drop the character, keep the meaning).  git on Windows\n"
          "fails at checkout, so one bad path makes the whole repo unclonable there.\n"
          "See https://github.com/stew675/llama-cpp-rdna-boosts/issues/36", file=sys.stderr)
    sys.exit(1)

# informational: Windows' classic MAX_PATH is 260 including the clone root and
# a terminating NUL, so a long path is only a problem for a deep clone root
print("tracked paths are Windows-checkout-safe (%d paths; longest is %d chars)" % (len(paths), len(longest)))
PY
