#!/usr/bin/env python3
"""
check-system-headers.py -- fail if a line the delivery inserts uses a symbol from a system header the file
does not (transitively) include, or guards a `#if` on a macro such a header declares.

WHY THIS EXISTS (issue #348, `INVARIANTS.md` r342)
    Since r328 `server_prompt_cache::take()` lives in `tools/server/server-slot-sched.cpp`.  Its
    huge-page branch is guarded by `#if defined(__linux__) && defined(MADV_HUGEPAGE)`, and when the
    function MOVED there the `#include <sys/mman.h>` stayed behind in `server-task.cpp`, where it had
    been added for the same function.  `MADV_HUGEPAGE` was therefore undefined in the new file, the
    branch dropped out of every Linux build with no diagnostic, and `LLAMA_SERVER_CACHE_HUGEPAGE`
    (default on, PR #248) did nothing for twelve releases: a 64 MiB+ prompt-cache state buffer went
    back to 4 KiB zero filling.  Only `nm -D --undefined-only libllama-server-impl.so | grep madvise`
    could see it -- and only a reporter ran that (issue #348, r339).

    Nothing in the flow could see it, and the two nearest instruments are blind by construction:
      * a BUILD is blind: the code compiles, and the branch it drops is on the default path;
      * `scripts/check-rung-file-sets.py` (gate 38) is blind: it reconciles FILES/defines/env/options
        per block, and the include was still present -- in the file the code had LEFT.  A need that
        MOVES to a new file is invisible to any per-file diff;
      * `scripts/check-windows-portability.sh` says so itself: "it cannot see a missing `#include`".
    The `defined(MACRO)` guard is what made it silent instead of a compile error, so both halves are
    checked: the guard (rule G) and the missing header (rule H).

WHAT IS CHECKED (on the lines the delivery inserts; a file the delivery ADDS is judged whole)
    G  a `#if`/`#elif`/`#ifdef`/`#ifndef` condition that tests a macro a system header declares
       (`MADV_*`, `MAP_*`, `POSIX_MADV_*`, `MSG_*`, `SOCK_*`, `SHUT_*`, `RTLD_*`, `SYS_*`, `SCHED_*`,
       `CLOCK_*`, `PTHREAD_COND_INITIALIZER`, `O_DIRECT`, `O_CLOEXEC`, `O_NOATIME`, ...).  Test the OS
       (`#if defined(__linux__)`), which turns a lost include into a compile error instead of a silently
       dropped feature.  This rule has no legitimate exception: to use `MADV_HUGEPAGE` you must have
       the header, so a guard on it is always "the header might be missing", which is exactly the trap.
    H  a call to a table symbol (`madvise`, `mmap`, `posix_fadvise`, `sendfile`, `getrandom`, `flock`,
       `dlopen`, `clock_gettime`, `sched_setaffinity`, `copy_file_range`, ...) or a use of a table
       macro (`MADV_*`, `MAP_ANON*`, `AIO_*`) in a file whose INCLUDE CLOSURE does not reach the
       declaring system header.  The closure follows `#include "..."` transitively through the tree
       (resolved by the including file's directory first, then by the tree index), so a header reached
       through `common.h` or `common.cuh` is fine -- that is how most of this tree legitimately gets
       its POSIX declarations.  C++ wrappers are equivalent (`<ctime>` -> `time.h`, `<cstdio>` ->
       `stdio.h`, ...).

SCOPE
    The files the delivery **adds** are judged whole; a file it merely **edits** is judged on the lines
    it inserts (`git diff -U0`, the same shape as `check-windows-portability.sh`).  That is what keeps
    the check CI-clean: upstream code has module-local names that collide with POSIX ones (hexagon's
    `mmap(bool)`, vulkan's `Klass::close()`) and upstream's own transitive includes are not ours to
    pin down.  Paths: `ggml/ src/ common/ tools/ tests/`.

WHAT IT IS NOT
    A compiler, and not IWYU: it knows a fixed table of symbol->header pairs (the ones the delivery
    actually calls) and a fixed family list, and it cannot see a type that a missing header changes.
    It cannot see a header that is included on the wrong side of an `#if` either -- rule H asks only
    whether the closure reaches the header, so `#include <sys/mman.h>` inside `#if defined(_WIN32)`
    would satisfy it.  A build on the target platform is still the only proof.

Usage:
    ./check-system-headers.py [options]

Options:
    --root DIR          tree to scan (default: this repo's ../llama.cpp, the fork checkout)
    --diff-from REV     window (`git diff -U0 --diff-filter=AMRT REV..HEAD`).  Default: release.json's
                        `.base` when that commit exists in ROOT, else every tracked file whole.
    --all               every tracked source file, judged whole (a review sweep; upstream code is not
                        clean by construction -- hexagon's module-local `mmap(bool)` is a finding)
    --path SPEC         path prefix filter (repeatable; default: ggml/ src/ common/ tools/ tests/)
    -v, --verbose       print every file scanned with the number of judged lines

Exit: 0 = clean (or nothing to scan), 1 = findings, 2 = usage/IO error.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# symbol -> the ONE system header that declares it (first entry is the preferred one)
SYMBOLS = {
    # sys/mman.h
    "madvise": "sys/mman.h",
    "posix_madvise": "sys/mman.h",
    "mmap": "sys/mman.h",
    "mmap64": "sys/mman.h",
    "munmap": "sys/mman.h",
    "msync": "sys/mman.h",
    "mprotect": "sys/mman.h",
    "mlock": "sys/mman.h",
    "munlock": "sys/mman.h",
    "mlock2": "sys/mman.h",
    "shm_open": "sys/mman.h",
    # fcntl.h
    "posix_fadvise": "fcntl.h",
    "posix_fallocate": "fcntl.h",
    "fallocate": "fcntl.h",
    # others, one header each
    "sendfile": "sys/sendfile.h",
    "getrandom": "sys/random.h",
    "flock": "sys/file.h",
    "dlopen": "dlfcn.h",
    "dlsym": "dlfcn.h",
    "dlclose": "dlfcn.h",
    "dlerror": "dlfcn.h",
    "clock_gettime": "time.h",
    "clock_nanosleep": "time.h",
    "nanosleep": "time.h",
    "sched_setaffinity": "sched.h",
    "sched_getaffinity": "sched.h",
    "pthread_setaffinity_np": "pthread.h",
    "numa_available": "numa.h",
    "numa_alloc_onnode": "numa.h",
    "getifaddrs": "ifaddrs.h",
    "freeifaddrs": "ifaddrs.h",
    "backtrace": "execinfo.h",
    "backtrace_symbols": "execinfo.h",
    "ioctl": "sys/ioctl.h",
    "copy_file_range": "unistd.h",
    "pread": "unistd.h",
    "pread64": "unistd.h",
    "pwrite": "unistd.h",
    "ftruncate": "unistd.h",
    "fsync": "unistd.h",
    "fdatasync": "unistd.h",
    "sysconf": "unistd.h",
    "getpid": "unistd.h",
    "gettid": "unistd.h",
    "syscall": "unistd.h",
    "posix_spawn": "spawn.h",
    "sem_timedwait": "semaphore.h",
    "sem_post": "semaphore.h",
}

# macro family prefix -> the header that declares it
FAMILIES = {
    "MADV_": "sys/mman.h",
    "POSIX_MADV_": "sys/mman.h",
    "MAP_": "sys/mman.h",
    "MSG_": "sys/socket.h",
    "SOCK_": "sys/socket.h",
    "SHUT_": "sys/socket.h",
    "IPPROTO_": "netinet/in.h",
    "RTLD_": "dlfcn.h",
    "SYS_": "sys/syscall.h",
    "SCHED_": "sched.h",
    "CLOCK_": "time.h",
    "O_DIRECT": "fcntl.h",
    "O_CLOEXEC": "fcntl.h",
    "O_NOATIME": "fcntl.h",
    "O_PATH": "fcntl.h",
    "O_TMPFILE": "fcntl.h",
    "PTHREAD_COND_INITIALIZER": "pthread.h",
    "PTHREAD_MUTEX_INITIALIZER": "pthread.h",
}

# keys of FAMILIES that are whole macro names, not prefixes
WHOLE = {"O_DIRECT", "O_CLOEXEC", "O_NOATIME", "O_PATH", "O_TMPFILE",
         "PTHREAD_COND_INITIALIZER", "PTHREAD_MUTEX_INITIALIZER"}

# C++ wrappers that provide the C header's declarations
EQUIV = {
    "time.h": {"ctime", "time.h"},
    "stdio.h": {"cstdio", "stdio.h"},
    "stdlib.h": {"cstdlib", "stdlib.h"},
    "string.h": {"cstring", "string.h"},
    "errno.h": {"cerrno", "errno.h"},
    "fcntl.h": {"fcntl.h"},
    "unistd.h": {"unistd.h"},
}

COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'', re.S)
RAWSTRING_RE = re.compile(r'R"([^()\\\s]{0,16})\((?:.|\n)*?\)\1"', re.S)


def strip_comments(text):
    """Remove comments AND string/char literals, keeping newlines so line numbers stay valid.

    Literals matter as much as comments here: `"... O_DIRECT fills unless ..."` and
    `"--host-experts pool without mmap (...)"` are log text, not calls, and a documented
    `close(fd)` must not be a finding (the same rule `check-windows-portability.sh` applies).
    """
    def blank(m):
        return "\n" * m.group(0).count("\n")
    text = RAWSTRING_RE.sub(blank, text)
    text = COMMENT_RE.sub(blank, text)
    return STRING_RE.sub(blank, text)


INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]', re.M)


def includes_of(text):
    """(quoted, name) pairs in order."""
    return [(m.group(1) == '"', m.group(2)) for m in INCLUDE_RE.finditer(text)]


class Tree:
    def __init__(self, root):
        self.root = root
        self.index = {}   # basename -> [relpath, ...]
        self.cache = {}

    def load(self, rel):
        if rel not in self.cache:
            try:
                with open(os.path.join(self.root, rel), "r", errors="replace") as fh:
                    self.cache[rel] = fh.read()
            except OSError:
                self.cache[rel] = None
        return self.cache[rel]

    def build_index(self, names):
        for rel in names:
            self.index.setdefault(os.path.basename(rel), []).append(rel)

    def resolve(self, quoted, name, from_rel):
        """Resolve an include to a tree-relative path (project header only), or None."""
        if not quoted:
            return None
        cand = os.path.normpath(os.path.join(os.path.dirname(from_rel), name))
        if self.load(cand) is not None:
            return cand
        hits = [p for p in self.index.get(os.path.basename(name), [])
                if p == name or p.endswith("/" + name)]
        if not hits:
            return None
        # prefer the shallowest, then the shortest path (deterministic)
        hits.sort(key=lambda p: (p.count("/"), len(p), p))
        return hits[0]

    def closure(self, rel):
        """Set of system header names (as written) reachable from rel, plus project headers visited."""
        seen = set()
        sys_headers = set()
        stack = [rel]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            text = self.load(cur)
            if text is None:
                continue
            for quoted, name in includes_of(text):
                if quoted:
                    target = self.resolve(True, name, cur)
                    if target is not None:
                        stack.append(target)
                        continue
                sys_headers.add(os.path.basename(name))
                sys_headers.add(name)
        return sys_headers, seen


def diff_window(root, diff_from, paths):
    """rel -> (whole_file: bool, added line numbers or None)."""
    cmd = ["git", "-C", root, "diff", "-U0", "--diff-filter=AMRT", f"{diff_from}..HEAD", "--"] + paths
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ERROR: git diff failed: {out.stderr.strip()}")
    status = subprocess.run(["git", "-C", root, "diff", "--name-status", "--diff-filter=AMRT",
                             f"{diff_from}..HEAD", "--"] + paths,
                            capture_output=True, text=True).stdout
    added_files = {l.split("\t")[-1] for l in status.splitlines() if l.startswith("A\t")}
    res = {}
    cur = None
    for line in out.stdout.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
            if cur in added_files:
                res[cur] = (True, None)
            else:
                res.setdefault(cur, (False, set()))
        elif line.startswith("+") and not line.startswith("+++"):
            pass   # content lines are ignored; hunk headers carry the numbers
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if m and not res[cur][0]:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                res[cur][1].update(range(start, start + count))
    return res


def scan_file(tree, rel, whole, wanted, findings, verbose):
    text = tree.load(rel)
    if text is None:
        return 0
    code = strip_comments(text)
    lines = code.splitlines()
    sys_headers, closure = tree.closure(rel)

    def judged(n):
        return whole or wanted is None or n in wanted

    count = 0
    for n, line in enumerate(lines, 1):
        if not judged(n):
            continue
        count += 1
        # ---- G: a guard on a macro that a system header declares
        if re.match(r"^[ \t]*#[ \t]*(if|elif|ifdef|ifndef)\b", line):
            macros = set(re.findall(r"\bdefined[ \t]*\([ \t]*([A-Za-z_][A-Za-z0-9_]*)", line))
            macros |= set(re.findall(r"\bdefined[ \t]+([A-Za-z_][A-Za-z0-9_]*)", line))
            m = re.match(r"^[ \t]*#[ \t]*(?:ifdef|ifndef)[ \t]+([A-Za-z_][A-Za-z0-9_]*)", line)
            if m:
                macros.add(m.group(1))
            for mac in macros:
                hdr = header_for_macro(mac)
                if hdr:
                    findings.append((rel, n,
                                     f"guard on `{mac}` (declared by <{hdr}>) -- test the OS instead, e.g. "
                                     f"`#if defined(__linux__)`: with a guard on the macro a missing "
                                     f"<{hdr}> silently compiles the branch out of every build (issue #348)"))
        # ---- H: a table symbol used without its header in the include closure
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
        for name in sorted(toks):
            hdr = SYMBOLS.get(name)
            if hdr and re.search(r"(?<![\w:.>])" + re.escape(name) + r"\s*\(", line):
                if not satisfied(hdr, sys_headers):
                    findings.append((rel, n, f"`{name}()` needs <{hdr}>, which the file's include "
                                             f"closure does not reach"))
        for name in sorted(toks):
            hdr = FAMILIES.get(name)
            if hdr and not satisfied(hdr, sys_headers):
                findings.append((rel, n, f"`{name}` needs <{hdr}>, which the file's include "
                                         f"closure does not reach"))
    if verbose:
        print(f"    {rel}: {count} line(s) judged, {len(closure)} project header(s) in closure")
    return count


def header_for_macro(mac):
    if mac in FAMILIES:
        return FAMILIES[mac]
    for pref, hdr in FAMILIES.items():
        if pref not in WHOLE and mac.startswith(pref) and len(mac) > len(pref):
            return hdr
    return None


def satisfied(hdr, sys_headers):
    return bool(EQUIV.get(hdr, {hdr}) & sys_headers)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", default=os.path.expanduser("~/llama.cpp"))
    ap.add_argument("--diff-from", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--path", action="append", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(os.path.join(root, ".git")) and not os.path.isfile(os.path.join(root, ".git")):
        raise SystemExit(f"ERROR: {root} is not a git checkout")

    paths = args.path or ["ggml/", "src/", "common/", "tools/", "tests/"]

    tracked = subprocess.run(["git", "-C", root, "ls-files", "--"] + paths,
                             capture_output=True, text=True).stdout.split()

    tree = Tree(root)
    tree.build_index(tracked)

    if args.all:
        window = {rel: (True, None) for rel in tracked}
    else:
        diff_from = args.diff_from
        if diff_from is None:
            diff_from = base_of_this_repo()
        if diff_from and subprocess.run(["git", "-C", root, "cat-file", "-e", diff_from],
                                        capture_output=True).returncode == 0:
            window = diff_window(root, diff_from, paths)
        else:
            print(f"note: no usable --diff-from ({diff_from}); judging every tracked file whole")
            window = {rel: (True, None) for rel in tracked}

    findings = []
    total = 0
    for rel in sorted(window):
        if not rel.endswith((".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".m", ".mm")):
            continue
        whole, wanted = window[rel]
        total += scan_file(tree, rel, whole, wanted, findings, args.verbose)

    n_added = sum(1 for w in window.values() if w[0])
    print(f"scanned {len(window)} delivered file(s) ({n_added} judged whole, "
          f"{len(window) - n_added} on inserted lines), {total} line(s) judged")

    if findings:
        for rel, n, msg in findings:
            print(f"FAIL {rel}:{n}: {msg}")
        print(f"\n{len(findings)} finding(s) -- see scripts/check-system-headers.py (issue #348).")
        return 1

    print("OK: every system header the delivery's own lines need is in the file's include closure, "
          "and no guard tests a system macro")
    return 0


def base_of_this_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "..", "release.json")) as fh:
            return json.load(fh).get("base")
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
