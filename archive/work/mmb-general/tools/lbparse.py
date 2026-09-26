#!/usr/bin/env python3
"""Parse llama-bench / test-backend-ops output.

Two modes:

  lbparse.py            # llama-bench table -> "pp8192=929.05  tg128=29.41  tg128@d16384=27.93"
  lbparse.py --ops NAME # test-backend-ops  -> "NAME: OK=5954 not_supported=11 FAIL=0"

Why this file exists rather than a grep: two traps cost real time during the mmb-general
campaign, and both are handled here.

  1. llama-bench names a depth-variant test "tg128 @ d16384", not "tg128" -- a naive
     regex on the test name silently drops it.

  2. test-backend-ops must NEVER be counted from a `2>&1`-merged log.  The status is
     ANSI-wrapped (`printf("\\033[1;32mOK\\033[0m\\n")`) and `print_test_console` writes
     the test name to stdout while the test itself emits CUDA-graph-warmup and
     allocation notices to stderr.  With `2>&1` into a *file*, stdout is block-buffered
     and stderr is not, so stderr lands between the name and its status and the status
     is orphaned onto its own line.  Counting `NAME(...): OK` then reports a different
     number every run (1947/1949/1951 were seen) and it looks like the test matrix is
     randomised -- it is not (it is deterministic: 8090 name lines every run).  This
     parser walks the log and attributes an orphaned status to the preceding name, so
     the totals are stable.  Run `--ops` on stdout-only or merged logs; both work.

Usage:
  llama-bench ... | lbparse.py
  test-backend-ops -o FLASH_ATTN_EXT 2>&1 | lbparse.py --ops FLASH_ATTN_EXT
"""
import re
import sys

ANSI = re.compile(r'\x1b\[[0-9;]*m')
BENCH_ROW = re.compile(r'^\s*\|.*\|\s*$')
# llama-bench test cell: pp512 / tg128 / tg128 @ d16384
BENCH_TEST = re.compile(r'((?:pp|tg|tpp)\d+)(?:\s*@\s*d(\d+))?$')
NUM = re.compile(r'([0-9.]+)')


def parse_bench(stream):
    out = []
    for line in stream:
        line = ANSI.sub('', line).strip()
        if not BENCH_ROW.match(line):
            continue
        cols = [c.strip() for c in line.strip('|').split('|')]
        if len(cols) < 2:
            continue
        test, val = cols[-2], cols[-1]
        m = BENCH_TEST.match(test)
        if not m:
            continue
        name = m.group(1) + (f"@d{m.group(2)}" if m.group(2) else "")
        v = NUM.match(val)
        if v:
            out.append(f"{name}={v.group(1)}")
    print("  ".join(out))


def parse_ops(stream, op):
    name_re = re.compile(r'^\s*(' + re.escape(op) + r'\(.*?\)):\s*(.*)$')
    ok = unsupported = failed = 0
    pending = None
    for raw in stream:
        line = ANSI.sub('', raw.rstrip('\n'))
        m = name_re.match(line)
        if m:
            # a name line: either carries its status, or the status follows (orphaned)
            if m.group(2).strip() == 'OK':
                ok += 1
                pending = None
            else:
                pending = m.group(1)
            continue
        s = line.strip()
        if pending is None:
            continue
        if s == 'OK':
            ok += 1
            pending = None
        elif s.startswith('not supported'):
            unsupported += 1
            pending = None
        elif s.startswith('FAIL') or s.startswith('Error'):
            failed += 1
            pending = None
    print(f"{op}: OK={ok} not_supported={unsupported} FAIL={failed}")
    # The gate is FAIL == 0.  The OK count is stable only with the pairing above.
    return failed


def main(argv):
    if len(argv) > 1 and argv[1] == '--ops':
        if len(argv) < 3:
            sys.exit("usage: lbparse.py --ops <OPNAME>   (e.g. FLASH_ATTN_EXT)")
        sys.exit(1 if parse_ops(sys.stdin, argv[2]) else 0)
    parse_bench(sys.stdin)


if __name__ == '__main__':
    main(sys.argv)
