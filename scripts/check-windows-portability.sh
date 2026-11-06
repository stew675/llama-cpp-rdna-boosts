#!/usr/bin/env bash
# Fail if the delivery adds a POSIX-only call outside a Windows-excluding preprocessor region.
#
# WHY THIS EXISTS (issue #231):
#   The delivery's host-side CUDA code is built with HIP *on Windows* too (clang-cl + the MSVC STL),
#   where <unistd.h>/<fcntl.h> do not exist and `close`, `pread`, `open`, `fsync`, ... are undeclared.
#   `ggml/src/ggml-cuda/moe-expert-cache.cu` is consciously portable: every file call goes through five
#   helpers (`pool_open_file`, `pool_fd_align`, `pool_pread`, `pool_aligned_alloc`, `pool_close_file`;
#   issue #130) with a POSIX branch (`open`/`pread`/`posix_memalign`) and a Windows branch
#   (`CreateFileA`/`ReadFile`/`_aligned_malloc`, and `pool_aligned_free` with it; the disk tier became real
#   there in r227).  The helpers are what keep a POSIX-only call out of the Windows build.
#   r143 then added two direct `close()` calls in `pool_free_locked` / `moe_cache_forget_all` teardown,
#   which is a hard compile error on Windows and invisible to every gate we can run here: there is no
#   Windows runner, so the only other detector is a reporter's build log (#231 was exactly that).
#   r131 and earlier were clean; the calls arrived with a feature, not with the portability contract.
#
# WHAT IT CHECKS:
#   For each scanned file, walk the preprocessor conditional stack and evaluate every condition with
#   `_WIN32`/`_MSC_VER` defined (unknown macros are three-valued, known non-Windows platform macros are
#   false, and a `#define NAME 0|1` that is only reached on Windows is learned into the table).  A line
#   is "compiled on Windows" unless an enclosing branch is provably not taken there.  In such a line, a
#   call to an unambiguous POSIX-only identifier is a Windows build error -- unless it is qualified
#   (`obj.close()`, `x->open()`, `Klass::close()`) or part of a longer name (`pool_close_file()`), which
#   is why this is a reviewable lint and not a compiler.  Comments and literals are stripped first, so a
#   documented `close(fd)` is not a finding.
#
#   SCOPE: by default only files the delivery **adds** (git `--diff-filter=A`) inside ggml/, src/,
#   common/ and tools/.  An upstream file the delivery merely edits is upstream's Windows CI's business,
#   and several of them use module-local names that collide with POSIX ones (hexagon's `mmap(bool)`,
#   vulkan's `Klass::close()`, zDNN's unguarded `sysconf` on a backend Windows never builds).  Use
#   `--modified-too` for a wider sweep; read its output, it is not CI-clean by construction.
#
# WHAT IT IS NOT: a Windows compiler. It cannot see a missing `#include`, a type that differs under
#   MSVC, or a `std::min<int64_t>` mangling clash (issue #217) -- those still need a build log.
#
# Usage:
#   ./check-windows-portability.sh [options]
#
# Options / environment:
#   --root DIR          ROOT        tree to scan (default: this repo's ../llama.cpp, the fork checkout)
#   --diff-from REV     DIFF_FROM   window for "added files" (`git diff --diff-filter=A REV..HEAD`).
#                                   Default: release.json's `.base` when that commit exists in ROOT,
#                                   else every tracked file.
#   --all                           every tracked source file (implies --all-paths)
#   --modified-too                  include files the delivery edits, not only the ones it adds
#   --path SPEC         PATHS       path prefix filter (repeatable; default: ggml/ src/ common/ tools/)
#   -h, --help
#
# Exit: 0 = clean, 1 = findings, 2 = usage/prerequisite error.
#
# The caller that matters is scripts/validate-set.sh (tier-1 CI, no compiler): it runs this on the
# freshly applied tree with `--diff-from HEAD~n_blocks`, so only files the delivery adds are judged.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$REPO_DIR/../llama.cpp}"
DIFF_FROM="${DIFF_FROM:-}"
SCAN_ALL=0
MODIFIED_TOO=0
declare -a PATHS=()

usage() { sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//; /^set -uo/d'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --root)         ROOT="$2"; shift 2 ;;
        --diff-from)    DIFF_FROM="$2"; shift 2 ;;
        --all)          SCAN_ALL=1; shift ;;
        --modified-too) MODIFIED_TOO=1; shift ;;
        --path)         PATHS+=("$2"); shift 2 ;;
        -h|--help)      usage 0 ;;
        *)              echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
command -v python3 >/dev/null || { echo "ERROR: python3 is required" >&2; exit 2; }
[ -d "$ROOT" ] || { echo "ERROR: --root $ROOT is not a directory" >&2; exit 2; }

if [ ${#PATHS[@]} -eq 0 ]; then
    PATHS=(ggml/ src/ common/ tools/)
fi

# Default the diff window to the delivery's fork point when the tree carries it (a fresh
# ../llama.cpp checkout does; the tarball tree validate-set.sh builds passes --diff-from).
WINDOW=""
if [ -z "$DIFF_FROM" ] && [ "$SCAN_ALL" = 0 ] && [ -f "$REPO_DIR/release.json" ] \
        && command -v jq >/dev/null 2>&1; then
    _base="$(jq -r '.base // empty' "$REPO_DIR/release.json" 2>/dev/null || true)"
    if [ -n "$_base" ] && git -C "$ROOT" cat-file -e "$_base^{commit}" 2>/dev/null; then
        DIFF_FROM="$_base"
    fi
fi

declare -a FILES=()
if [ -n "$DIFF_FROM" ]; then
    git -C "$ROOT" cat-file -e "$DIFF_FROM^{commit}" 2>/dev/null \
        || { echo "ERROR: --diff-from $DIFF_FROM not in $ROOT" >&2; exit 2; }
    if [ "$MODIFIED_TOO" = 1 ]; then
        FILTER="ACMRT"
    else
        FILTER="A"
    fi
    mapfile -t FILES < <(git -C "$ROOT" diff --name-only --diff-filter="$FILTER" "$DIFF_FROM"..HEAD 2>/dev/null)
    WINDOW="added since $DIFF_FROM"
    [ "$MODIFIED_TOO" = 1 ] && WINDOW="added/modified since $DIFF_FROM"
else
    mapfile -t FILES < <(git -C "$ROOT" ls-files)
    WINDOW="tracked"
fi

declare -a SCAN=()
for f in "${FILES[@]}"; do
    [ -n "$f" ] || continue
    [ -f "$ROOT/$f" ] || continue
    case "$f" in
        *.c|*.cc|*.cpp|*.cu|*.cuh|*.h|*.hpp|*.inl) ;;
        *) continue ;;
    esac
    for p in "${PATHS[@]}"; do
        case "$f" in
            "$p"*) SCAN+=("$f"); break ;;
        esac
    done
done

if [ ${#SCAN[@]} -eq 0 ]; then
    echo "==> Windows portability scan: no C/C++ file selected ($WINDOW) -- clean"
    exit 0
fi

echo "==> Windows portability scan: ${#SCAN[@]} $WINDOW C/C++ file(s) in $ROOT"

ROOT="$ROOT" python3 - "${SCAN[@]}" <<'PY'
import os
import re
import sys

root = os.environ["ROOT"]

# Unambiguous POSIX-only calls.  Deliberately NOT read/write/stat/rename/socket/usleep-as-a-name:
# those are also member functions or local helpers (`ofs.write`, `buf.read`), so a textual lint
# cannot decide them and a false failure in CI is worse than a missed one.
POSIX = (
    "open", "close", "pread", "pread64", "pwrite", "fsync", "fdatasync", "unlink",
    "fstat", "lstat", "opendir", "readdir", "closedir", "access", "mkdir",
    "posix_memalign", "posix_fadvise", "posix_fallocate", "fallocate", "copy_file_range",
    "ftruncate", "mkstemp", "symlink", "readlink", "sysconf", "getpid", "dlopen", "dlsym",
    "mmap", "munmap", "usleep",
)
CALL_RE = re.compile(r"(?<![.>:\w])(" + "|".join(POSIX) + r")\s*\(")

# Macros that are false under `_WIN32`; every other unknown macro stays three-valued.
# Platform macros that are false under `_WIN32`.  Keep this list to *platform* macros: a feature or
# compiler macro (`__HIP_PLATFORM_AMD__`, `__CUDACC__`) is true on a Windows HIP build, so treating it as
# non-Windows would silently stop checking that region (a false negative, the worse direction).
NON_WINDOWS_MACROS = {
    "__linux__", "__unix__", "__APPLE__", "__MACH__", "__FreeBSD__", "__NetBSD__", "__OpenBSD__",
    "__DragonFly__", "__HAIKU__", "__ANDROID__", "__gnu_linux__", "__linux", "linux",
    "__sun", "__sun__", "__hpux", "__sgi", "_AIX", "__AIX__", "__serenity__",
    "_POSIX_VERSION", "_POSIX_MAPPED_FILES", "_POSIX_C_SOURCE", "_GNU_SOURCE", "__USE_MISC",
}
WINDOWS_MACROS = {"_WIN32", "_WIN64", "_MSC_VER", "__MINGW32__", "__MINGW64__", "_WIN32_WINNT"}

T, F, U = True, False, None


def strip_comments_and_literals(text):
    """Blank out comments and string/char literals, preserving line structure."""
    out = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                out.append("  ")
                i += 2
                continue
            if c == '"':
                state = "str"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                state = "chr"
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
        elif state == "line":
            out.append("\n" if c == "\n" else " ")
            if c == "\n":
                state = "code"
            i += 1
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
        else:                                   # string / char literal
            quote = '"' if state == "str" else "'"
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == "\n":                        # unterminated: resynchronise
                state = "code"
                out.append("\n")
                i += 1
                continue
            if c == quote:
                state = "code"
            out.append(" ")
            i += 1
    return "".join(out)


class Parser:
    """Three-valued evaluator for a preprocessor condition, with `_WIN32` pinned."""

    def __init__(self, expr, macros=None, win=True):
        self.toks = self.tokenize(expr)
        self.pos = 0
        self.win = win
        self.macros = macros if macros is not None else {}

    @staticmethod
    def tokenize(expr):
        toks = []
        i, n = 0, len(expr)
        while i < n:
            c = expr[i]
            if c.isspace():
                i += 1
                continue
            if expr.startswith("defined", i) and not (
                    i + 7 < n and (expr[i + 7].isalnum() or expr[i + 7] == "_")):
                i += 7
                while i < n and expr[i].isspace():
                    i += 1
                if i < n and expr[i] == "(":
                    j = expr.find(")", i)
                    if j < 0:
                        j = n
                    toks.append(("defined", expr[i + 1:j].strip()))
                    i = j + 1
                else:
                    j = i
                    while j < n and (expr[j].isalnum() or expr[j] == "_"):
                        j += 1
                    toks.append(("defined", expr[i:j]))
                    i = j
                continue
            if expr.startswith("&&", i) or expr.startswith("||", i):
                toks.append(("op", expr[i:i + 2]))
                i += 2
                continue
            if c in "!()":
                toks.append(("op", c))
                i += 1
                continue
            if c.isdigit():
                j = i
                while j < n and expr[j].isalnum():
                    j += 1
                toks.append(("num", expr[i:j]))
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (expr[j].isalnum() or expr[j] == "_"):
                    j += 1
                toks.append(("id", expr[i:j]))
                i = j
                continue
            i += 1                               # unknown character: skip
        return toks

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.pos += 1
        return t

    def parse(self):
        return self.parse_or()

    def parse_or(self):
        v = self.parse_and()
        while self.peek() == ("op", "||"):
            self.take()
            r = self.parse_and()
            v = T if (v is T or r is T) else (F if (v is F and r is F) else U)
        return v

    def parse_and(self):
        v = self.parse_unary()
        while self.peek() == ("op", "&&"):
            self.take()
            r = self.parse_unary()
            v = F if (v is F or r is F) else (T if (v is T and r is T) else U)
        return v

    def parse_unary(self):
        if self.peek() == ("op", "!"):
            self.take()
            v = self.parse_unary()
            return U if v is U else (not v)
        return self.parse_atom()

    def parse_atom(self):
        kind, val = self.take()
        if kind == "op" and val == "(":
            v = self.parse_or()
            if self.peek() == ("op", ")"):
                self.take()
            return v
        if kind == "defined":
            if val in self.macros:
                return T
            if val in WINDOWS_MACROS:
                return T if self.win else F
            if val in NON_WINDOWS_MACROS:
                return F if self.win else U
            return U
        if kind == "num":
            try:
                return T if int(val.rstrip("uUlL"), 0) != 0 else F
            except ValueError:
                return U
        if kind == "id":
            if val in self.macros:
                return self.macros[val]
            if val in WINDOWS_MACROS:
                return T if self.win else F
            if val in NON_WINDOWS_MACROS:
                return F if self.win else U
            return U
        return U


def eval_on_windows(expr, macros):
    return Parser(expr, macros, win=True).parse()


def walk_directives(code, on_define=None):
    """Yield (lineno, word, rest, excluded_on_windows) for every preprocessor directive."""
    stack = []          # each: {"conds": [...], "excl": bool}
    for lineno, line in enumerate(code.split("\n"), 1):
        s = line.lstrip()
        if not s.startswith("#"):
            yield lineno, None, line, any(fr["excl"] for fr in stack)
            continue
        directive = s[1:].lstrip()
        m = re.match(r"[A-Za-z_]+", directive)
        word = m.group(0) if m else ""
        rest = directive[len(word):].strip()
        if word in ("if", "ifdef", "ifndef"):
            expr = ("defined(%s)" % rest) if word == "ifdef" else (
                   ("!defined(%s)" % rest) if word == "ifndef" else rest)
            stack.append({"conds": [expr], "excl": eval_on_windows(expr, on_define or {}) is F})
            continue
        if word == "elif" and stack:
            stack[-1]["conds"].append(rest)
            v = T
            for c in stack[-1]["conds"][:-1]:
                r = eval_on_windows(c, on_define or {})
                if r is F:
                    continue
                v = F if r is T else U
            last = eval_on_windows(rest, on_define or {})
            if last is F or v is F:
                stack[-1]["excl"] = True
            elif last is T and v is T:
                stack[-1]["excl"] = False
            else:
                stack[-1]["excl"] = False
            continue
        if word == "else" and stack:
            v = T
            for c in stack[-1]["conds"]:
                r = eval_on_windows(c, on_define or {})
                if r is F:
                    continue
                v = F if r is T else U
            stack[-1]["excl"] = (v is F)
            stack[-1]["conds"].append("1")
            continue
        if word == "endif" and stack:
            stack.pop()
            continue
        yield lineno, word, rest, any(fr["excl"] for fr in stack)


def learn_macros(paths, macros):
    """Fold in `#define NAME 0|1` that is provably reached on Windows (e.g. LLAMA_LAZY_READER_POSIX)."""
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                code = strip_comments_and_literals(f.read())
        except OSError:
            continue
        for _, word, rest, excluded in walk_directives(code, macros):
            if word != "define" or excluded:
                continue
            m = re.match(r"([A-Za-z_]\w*)\s+(.+)$", rest)
            if not m:
                continue
            name, body = m.group(1), m.group(2).strip()
            if body.endswith("\\"):
                continue
            v = eval_on_windows(body, macros)
            if v is U:
                continue
            if name in macros and macros[name] != v:
                macros[name] = U                   # conflicting definitions: unknown
            else:
                macros[name] = v
    return macros


# --- collect the local headers a file includes (one level), so a platform predicate defined there
#     (`#define LLAMA_LAZY_READER_POSIX 0` under `#if defined(_WIN32)`) is known when scanning the .cpp
def local_includes(rel):
    path = os.path.join(root, rel)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    out = []
    for m in re.finditer(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
        inc = m.group(1)
        for cand in (os.path.join(os.path.dirname(path), inc), os.path.join(root, inc)):
            if os.path.isfile(cand):
                out.append(cand)
                break
    return out


findings = []
scanned = 0
for rel in sys.argv[1:]:
    path = os.path.join(root, rel)
    macros = {}
    learn_macros([path] + local_includes(rel), macros)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError:
        continue
    scanned += 1
    code = strip_comments_and_literals(raw)
    raw_lines = raw.split("\n")
    for lineno, word, line, excluded in walk_directives(code, macros):
        if word is not None or excluded:
            continue
        for m in CALL_RE.finditer(line):
            findings.append((rel, lineno, m.group(1), raw_lines[lineno - 1].strip()))

if findings:
    names = ", ".join(sorted({f[2] for f in findings}))
    print("FAIL: %d unguarded POSIX call site(s) in %d scanned file(s) -- these do not compile with HIP"
          " on Windows (_WIN32):" % (len(findings), scanned))
    for rel, lineno, name, text in findings:
        print("  %s:%d: %s" % (rel, lineno, text))
    print("""
  `%s` is POSIX-only: <unistd.h>/<fcntl.h> are not included under _WIN32 and the identifier is
  undeclared in an MSVC/clang-cl build.  Wrap the call in `#if !defined(_WIN32)`, or route it through a
  portability helper (the host pool's pool_open_file/pool_fd_align/pool_pread/pool_aligned_alloc/
  pool_close_file are the model: empty on Windows, the real call elsewhere).
  No gate on this box can see this -- there is no Windows runner -- so the report in issue #231 (a
  reporter's build log) was the only detector.  Please do not ship it again.""" % names)
    sys.exit(1)

print("    OK: %d file(s) scanned, no unguarded POSIX call outside a Windows-excluding region" % scanned)
PY
