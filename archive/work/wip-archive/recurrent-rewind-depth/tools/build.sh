#!/bin/bash
# Build a single test .cpp against the existing llama.cpp build tree, reusing
# the compile flags and link line of test-recurrent-state-rollback.
# usage: build.sh <src.cpp> <out> [build-dir]
set -e
REPO="${REPO:-$HOME/llama.cpp}"
BUILD_DIR="${3:-$REPO/build-rocm}"
SRC="$(readlink -f "$1")"
OUT="$2"
OBJ="$OUT.o"

CC_JSON="$BUILD_DIR/compile_commands.json"
LINK_TXT="$BUILD_DIR/tests/CMakeFiles/test-recurrent-state-rollback.dir/link.txt"

python3 - "$CC_JSON" "$SRC" "$OBJ" <<'PY'
import json, shlex, subprocess, os, sys
cc_json, src, obj = sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3]
cc = json.load(open(cc_json))
e = next(x for x in cc if x['file'].endswith('test-recurrent-state-rollback.cpp'))
toks, out = shlex.split(e['command']), []
i = 0
while i < len(toks):
    t = toks[i]
    if t == '-o': i += 2; continue
    if t == '-c': i += 1; continue
    if t == '-fsyntax-only': i += 1; continue
    if t in ('-MD', '-MT', '-MF'): i += 2; continue
    if os.path.realpath(t) == os.path.realpath(e['file']): i += 1; continue
    out.append(t); i += 1
out += ['-c', src, '-o', obj]
r = subprocess.run(out, cwd=e['directory'], capture_output=True, text=True)
if r.returncode:
    sys.stderr.write(r.stderr)
sys.exit(r.returncode)
PY

link=$(sed -e "s#\"CMakeFiles/test-recurrent-state-rollback.dir/test-recurrent-state-rollback.cpp.o\"#\"$OBJ\"#" \
           -e "s#-o ../bin/test-recurrent-state-rollback#-o $OUT#" "$LINK_TXT")
( cd "$BUILD_DIR/tests" && eval "$link" )
echo "built $OUT"
