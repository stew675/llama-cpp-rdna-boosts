#!/usr/bin/env python3
"""
check-rung-file-sets.py -- reconcile each block's change against the block it replaces (`TODO` #358, retired
at r341 as `MANIFESTS.md` gate 38; the engineering facts are `INVARIANTS.md` r331 (the incident) and r341
(this instrument)).

WHY THIS EXISTS
    The r328 re-base re-expressed block 00 from a new fork point and landed 34 of the old block's
    37 files.  Two were a decision (D1 retired mmf.cu/mmf.cuh), the third -- ggml/src/ggml-alloc.c
    (GGML_GLU_INPLACE, the unused-view release, the layout_fp fingerprint) -- was accounted for by no
    decision at all, so r328 and r329 shipped without it while the block's own text, ENVIRONMENT.md,
    the per-block gate table and tests/test-alloc.cpp still described it (issue #345, WORKLOG 261121.010).
    Nothing in the release flow could see it:
      * a BUILD cannot: the missing hunks live inside other files and compile fine;
      * `scripts/gate-per-block-build.sh` cannot (and is a bisection tool since r332, not a gate);
      * the set's own gates only fail if the block's OWN behavioural gate happens to run -- and the
        re-base's gate table did not run block 00's gate-glu-inplace.sh.
    The deliverable IS reproducible from the previous release, which is the instrument this script
    uses: reconcile the new set against the set it replaces, item by item, and treat anything that
    vanished as a defect until a decision row names the retirement (INVARIANTS.md, r331).

WHAT IS COMPARED (per block, old release -> new set)
    file    every path the block touches, with the number of lines it ADDS there.
    define  every `#define NAME` the block adds.
    env     every environment-variable name the block reads (`getenv("NAME")`, `std::getenv(...)`).
    option  every long option the block adds to `common/arg.cpp` (user-visible, `--help`-visible).
    block   the block itself (a block that disappears is a decision, not an accident).

    Findings, strongest first:
      LOSS    a file the old set added lines to is no longer touched by ANY block of the new set, or a
              define/env/option the old set added is gone from the whole new set.
      SHRUNK  a file whose added-line count in the new set (or in the same block) is at most
              `--shrink-pct` of the old count.  This is the "a hunk inside a file it still touches was
              dropped" class -- r331's lost GGML_CUDA_DISABLE_ROPE_SET_ROWS switch and its renamed
              GGML_CUDA_CC_DGX_SPARK -> GGML_CUDA_CC_GB10 token both read as a shrunk/short patch here.
      GONE    a block or an optional artifact that the old release had and the new set does not.
      MOVED   (informational) a file/symbol that merely changed which block carries it -- the set keeps
              the content, so this is not a defect; it is printed because block ownership is the
              maintainer's decision (AGENTS.md).
      NEW     (informational) a file/symbol/block the new set adds.

    A LOSS/SHRUNK/GONE finding FAILS the check (exit 1) unless a decision row in the allowlist
    (default `patches/FILE-SET-RETIREMENTS.tsv`) names it.  A row is scoped to the transition it was
    decided in: the check only applies rows whose `from` column equals the old release it is comparing
    against, so a row can never suppress a future transition's finding.  Rows that do not apply are
    counted, not silently ignored.

MODES (all four sides accept the same shapes)
    --old <git-ref|DIR>     the release being replaced.  Default: the newest `v*` tag reachable from
                            `HEAD^`, i.e. the last release strictly before the working state.  A DIR may
                            be a release root (patches/ + optional/) or a `patches/` directory.
    --new <git-ref|DIR>     default `patches/` (+ `optional/`) of the delivery repo.
    --new-chain <REPO> --new-base <SHA> [--new-tip <REF>]
                            instead of --new: take the new side from the block CHAIN in a llama.cpp
                            checkout -- the commits after <SHA>, with the block number read from each
                            commit's `rdna-boosts: block NNNN:` subject.  This is the mode the re-base
                            ladder wants: after porting a rung, check that rung against the block it
                            replaces before moving on:
                                scripts/check-rung-file-sets.py --rung 0000 \\
                                    --old v17-919794500-r339 --new-chain ~/llama.cpp --new-base 919794500
    --rung N[,N...]         restrict the comparison to these block numbers (the ladder use).

USAGE
    scripts/check-rung-file-sets.py                       # current patches/ vs the last release
    scripts/check-rung-file-sets.py --old v17-919794500-r328 --new v17-919794500-r331   # the r331 fix
    scripts/check-rung-file-sets.py --warn-only           # report, never exit 1 (mid-re-base triage)
    scripts/check-rung-file-sets.py -v                    # + the per-block inventory

    Exit: 0 = reconciled (or skipped with --allow-skip), 1 = unexplained findings, 2 = usage/IO error.

WHO CALLS IT
    * `scripts/validate-set.sh` check 9 (so it runs in CI and is part of "no patch set ships unverified").
      A clone without the previous release's tag objects (a shallow CI checkout) reports SKIP, loudly.
    * `scripts/validate-gpu.sh` step 1b (tier 2, a full clone, on every new `origin/main`).
    * the release flow, by hand, before `scripts/make-patches.sh` / `scripts/make-release.sh` -- and
      per rung during a re-base, which is what makes a lost file a five-minute fix instead of a
      two-release defect.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field


def die(msg: str) -> None:
    """A usage/IO error: print it and exit 2 (never confused with a finding's exit 1)."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)

# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------

RE_DIFF = re.compile(r'^diff --git ("?)a/(.*?)("?) ("?)b/(.*?)("?)$')
RE_DEFINE = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)')
RE_ENV = re.compile(r'\bgetenv\w*\s*\(\s*"([^"]+)"')
RE_OPTION = re.compile(r'"(--[A-Za-z0-9][A-Za-z0-9_-]*)"')
# The subject writes the block as `00`..`16` (official) or `0300` (temporary); the patch FILE prefix
# normalises both to four digits (`0000`..`0016`, `0300`) -- see scripts/make-patches.sh.
RE_SUBJECT = re.compile(r'^rdna-boosts: block (\d{2}|\d{4}):')
ARG_CPP = 'common/arg.cpp'

# Item kinds, in report order.  `block` is synthetic (it names the block itself).
KINDS = ('file', 'define', 'env', 'option', 'block')


@dataclass
class Patch:
    """One block's contribution: the files it touches (with added-line counts) and the symbols it adds."""
    files: dict[str, int] = field(default_factory=dict)
    defines: set[str] = field(default_factory=set)
    envs: set[str] = field(default_factory=set)
    options: set[str] = field(default_factory=set)
    subject: str = ''


def _unquote(path: str) -> str:
    """git quotes a path containing a space or a non-ASCII byte: `"b/a b.cu"`."""
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        return path[1:-1].encode('utf-8').decode('unicode_escape')
    return path


def parse_patch(text: str, subject: str = '') -> Patch:
    """Read one `git format-patch` / `git show --patch` body.

    The file set comes from `+++ b/<path>` (a rename contributes both sides through its `diff --git`
    line, so an old path that is only *removed* is still counted as "the block touched it").
    """
    p = Patch(subject=subject)
    cur: str | None = None
    for line in text.splitlines():
        if line.startswith('diff --git '):
            m = RE_DIFF.match(line)
            if m:
                for path in (_unquote(m.group(2)), _unquote(m.group(5))):
                    p.files.setdefault(path, 0)
                cur = _unquote(m.group(5))
            else:                                    # unparseable header: never attribute blindly
                cur = None
            continue
        if line.startswith('+++ '):
            path = line[4:].strip()
            if path == '/dev/null':
                cur = None
            else:
                if path.startswith('b/'):
                    path = path[2:]
                cur = _unquote(path)
                p.files.setdefault(cur, 0)
            continue
        if line.startswith('--- ') or not line.startswith('+') or line.startswith('+++'):
            continue
        if cur is None:
            continue
        p.files[cur] += 1
        body = line[1:]
        m = RE_DEFINE.match(body)
        if m:
            p.defines.add(m.group(1))
        for m in RE_ENV.finditer(body):
            p.envs.add(m.group(1))
        if cur == ARG_CPP:
            for m in RE_OPTION.finditer(body):
                p.options.add(m.group(1))
    return p


def parse_subject(text: str) -> str:
    for line in text.splitlines():
        if not line.strip():
            return ''
        return line
    return ''


# ---------------------------------------------------------------------------------------------
# loading the two sides
# ---------------------------------------------------------------------------------------------

Key = tuple[str, str]          # ('default'|'optional', '0000')


def _key_of(filename: str) -> str | None:
    m = re.match(r'^(\d{4})-.*\.patch$', os.path.basename(filename))
    return m.group(1) if m else None


def load_dir(root: str) -> dict[Key, Patch]:
    """A release root (`patches/` + `optional/*/`) or a bare `patches/` directory."""
    if os.path.isdir(os.path.join(root, 'patches')):
        patches_dir, optional_dir = os.path.join(root, 'patches'), os.path.join(root, 'optional')
    else:
        patches_dir, optional_dir = root, os.path.join(os.path.dirname(os.path.abspath(root)), 'optional')
    out: dict[Key, Patch] = {}
    if not os.path.isdir(patches_dir):
        die(f'check-rung-file-sets: no patches/ under {root}')
    for name in sorted(os.listdir(patches_dir)):
        if not name.endswith('.patch'):
            continue
        num = _key_of(name)
        if num is None:
            continue
        path = os.path.join(patches_dir, name)
        text = open(path, encoding='utf-8', errors='replace').read()
        out[('default', num)] = parse_patch(text, subject=name)
    if os.path.isdir(optional_dir):
        for sub in sorted(os.listdir(optional_dir)):
            d = os.path.join(optional_dir, sub)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                num = _key_of(name)
                if num is None or not name.endswith('.patch'):
                    continue
                text = open(os.path.join(d, name), encoding='utf-8', errors='replace').read()
                out[('optional', num)] = parse_patch(text, subject=f'{sub}/{name}')
    return out


def _git(repo: str, *args: str) -> str:
    proc = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'git {" ".join(args)} in {repo} failed: {proc.stderr.strip()}')
    return proc.stdout


def load_ref(repo: str, ref: str) -> dict[Key, Patch]:
    """Materialise a release's artifacts from a git ref in the delivery repo (no checkout)."""
    try:
        listing = _git(repo, 'ls-tree', '-r', '--name-only', ref, '--', 'patches', 'optional')
    except RuntimeError as exc:
        die(f'check-rung-file-sets: cannot read {ref} in {repo}: {exc}')
    out: dict[Key, Patch] = {}
    for path in listing.splitlines():
        if not path.endswith('.patch'):
            continue
        num = _key_of(path)
        if num is None:
            continue
        kind = 'optional' if path.startswith('optional/') else 'default'
        if kind == 'optional' and path.count('/') < 2:
            continue
        try:
            text = _git(repo, 'show', f'{ref}:{path}')
        except RuntimeError as exc:
            die(f'check-rung-file-sets: cannot read {ref}:{path}: {exc}')
        out[(kind, num)] = parse_patch(text, subject=path)
    return out


def load_chain(repo: str, base: str, tip: str | None) -> dict[Key, Patch]:
    """A fork checkout's applied block chain: '<base>..<tip>', block number from the subject."""
    rev = [f'{base}..{tip}'] if tip else [f'{base}..HEAD']
    commits = _git(repo, 'rev-list', '--reverse', *rev).split()
    if not commits:
        die(f'check-rung-file-sets: no block commits in {repo} {rev[0]}')
    out: dict[Key, Patch] = {}
    for sha in commits:
        text = _git(repo, 'show', '--format=%s%n%b', '--patch', sha)
        # The subject is the first non-empty line; the body follows.  Re-split at the first blank
        # line after it so the patch text is exactly the diff.
        lines = text.splitlines(keepends=True)
        idx = 0
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            die(f'check-rung-file-sets: empty commit {sha}')
        subject = lines[idx].strip()
        idx += 1
        while idx < len(lines) and lines[idx].strip():
            idx += 1
        m = RE_SUBJECT.match(subject)
        if not m:
            die(
                f'check-rung-file-sets: {sha[:9]} is not a block commit ({subject!r})\n'
                f'  the chain must be the applied set: rdna-boosts: block NNNN: ...')
        num = '%04d' % int(m.group(1))
        out[('default', num)] = parse_patch(''.join(lines[idx:]), subject=f'{sha[:9]} {subject}')
    return out


def resolve_old(repo: str, spec: str | None) -> tuple[str, dict[Key, Patch] | None]:
    """Return (identifier, patches) for the old side; patches is None when it cannot be resolved."""
    if spec is None:
        for rev in ('HEAD^', 'HEAD'):
            try:
                ref = _git(repo, 'describe', '--tags', '--abbrev=0', '--match', 'v*', rev).strip()
            except RuntimeError:
                continue
            if ref:
                return ref, load_ref(repo, ref)
        return '<unknown>', None
    if os.path.isdir(spec):
        return os.path.abspath(spec), load_dir(spec)
    try:
        _git(repo, 'rev-parse', '--verify', '--quiet', f'{spec}^{{commit}}')
    except RuntimeError:
        die(f'check-rung-file-sets: --old {spec} is neither a directory nor a ref in {repo}')
    return spec, load_ref(repo, spec)


def resolve_new(repo: str, spec: str | None, chain_repo: str | None, chain_base: str | None,
                chain_tip: str | None) -> tuple[str, dict[Key, Patch]]:
    if chain_repo or chain_base or chain_tip:
        if not (chain_repo and chain_base):
            die('check-rung-file-sets: --new-chain needs --new-base (and --new-chain <repo>)')
        return (f'chain {chain_base}..{chain_tip or "HEAD"} of {chain_repo}',
                load_chain(chain_repo, chain_base, chain_tip))
    if spec is None:
        spec = os.path.join(repo, 'patches')
    if os.path.isdir(spec):
        return os.path.abspath(spec), load_dir(spec)
    return spec, load_ref(repo, spec)


# ---------------------------------------------------------------------------------------------
# the allowlist (decision rows)
# ---------------------------------------------------------------------------------------------

@dataclass
class Row:
    frm: str
    block: str
    kind: str
    item: str
    reason: str
    line: int


def load_allowlist(path: str) -> list[Row]:
    rows: list[Row] = []
    if not path or not os.path.isfile(path):
        return rows
    for n, raw in enumerate(open(path, encoding='utf-8'), start=1):
        line = raw.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        cols = [c.strip() for c in line.split('\t')]
        if len(cols) < 5 or any(not c for c in cols[:4]) or not cols[4]:
            die(
                f'check-rung-file-sets: {path}:{n}: a row is <from>\\t<block>\\t<kind>\\t<item>\\t<reason>\n'
                f'  got: {line!r}')
        frm, block, kind, item, reason = cols[0], cols[1], cols[2], cols[3], '\t'.join(cols[4:])
        if kind not in KINDS:
            die(f'check-rung-file-sets: {path}:{n}: unknown kind {kind!r} (one of {", ".join(KINDS)})')
        rows.append(Row(frm, block, kind, item, reason, n))
    return rows


def allowlist_match(rows: list[Row], old_id: str, block: str, kind: str, item: str) -> Row | None:
    for row in rows:
        if row.frm not in ('*', old_id):
            continue
        if row.block not in ('*', block):
            continue
        if row.kind == kind and row.item == item:
            return row
    return None


# ---------------------------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------------------------

@dataclass
class Finding:
    cls: str                 # LOSS | SHRUNK | GONE | MOVED | NEW
    block: str
    kind: str
    item: str
    detail: str
    row: Row | None = None


def compare(old: dict[Key, Patch], new: dict[Key, Patch], shrink_pct: float,
            only: set[str] | None) -> list[Finding]:
    findings: list[Finding] = []
    shrink = shrink_pct / 100.0

    def wanted(num: str) -> bool:
        return only is None or num in only

    # ---- blocks ---------------------------------------------------------------------------
    for key in sorted(old):
        kind, num = key
        if not wanted(num):
            continue
        if key not in new:
            carried = sorted(old[key].files)
            findings.append(Finding(
                'GONE', num, 'block', num,
                f'{kind} artifact {num} is not in the new set (it carried {len(carried)} file(s))'))
    for key in sorted(new):
        kind, num = key
        if wanted(num) and key not in old:
            findings.append(Finding('NEW', num, 'block', num, f'new {kind} artifact ({new[key].subject})'))

    # ---- files ----------------------------------------------------------------------------
    # set-wide added-line totals, so "moved to another block" and "lost" are distinguishable
    tot_old: dict[str, int] = {}
    tot_new: dict[str, int] = {}
    for p in old.values():
        for f, n in p.files.items():
            tot_old[f] = tot_old.get(f, 0) + n
    for p in new.values():
        for f, n in p.files.items():
            tot_new[f] = tot_new.get(f, 0) + n

    # which new blocks (other than the matched one) carry a file, and with how many added lines
    carriers: dict[str, dict[str, int]] = {}
    for (kind, num), p in new.items():
        for f, n in p.files.items():
            if n > 0:
                carriers.setdefault(f, {})[num] = n

    for key in sorted(old):
        kind, num = key
        if key not in new or not wanted(num):
            continue
        o, n = old[key], new[key]
        for f, oa in sorted(o.files.items()):
            if oa <= 0:
                continue
            na = n.files.get(f, 0)
            if na >= oa * (1.0 - shrink):
                continue
            others = {b: c for b, c in carriers.get(f, {}).items() if b != num}
            set_new = tot_new.get(f, 0)
            set_old = tot_old.get(f, 0)
            where = (f'; now carried by {", ".join(f"{b} +{c}" for b, c in sorted(others.items()))}'
                     if others else '')
            # The verdict turns on the SET's total for the file, not on the block's: a block that sheds
            # a file another block absorbs has lost nothing.  Only a set total that did not grow makes
            # the block's own drop a question.
            if set_new == 0:
                findings.append(Finding(
                    'LOSS', num, 'file', f,
                    f'the old set added {set_old} line(s) here ({oa} by block {num}); no block of the new '
                    f'set touches it'))
            elif set_new >= set_old:
                findings.append(Finding(
                    'MOVED', num, 'file', f,
                    f'block {num} adds {na} line(s) here, was {oa}{where}; the set adds {set_new}, '
                    f'was {set_old}'))
            else:
                findings.append(Finding(
                    'SHRUNK', num, 'file', f,
                    f'block {num} adds {na} line(s) here, was {oa} ({100 * (na - oa) / oa:+.0f} %)'
                    f'{where}; the set adds {set_new}, was {set_old} ({100 * (set_new - set_old) / set_old:+.0f} %)'))

    # ---- symbols --------------------------------------------------------------------------
    # A symbol is set-wide by nature (a `#define` exists once in the tree), so the question is only
    # whether the SET still has it, and if so which block adds it.
    sym_attrs = {'define': 'defines', 'env': 'envs', 'option': 'options'}
    holders_new: dict[str, dict[str, set[str]]] = {k: {} for k in sym_attrs}
    for (_, num), p in new.items():
        for symkind, attr in sym_attrs.items():
            for s in getattr(p, attr):
                holders_new[symkind].setdefault(s, set()).add(num)

    for key in sorted(old):
        kind, num = key
        if key not in new or not wanted(num):
            continue
        o, n = old[key], new[key]
        for symkind, attr in sym_attrs.items():
            oset, nset = getattr(o, attr), getattr(n, attr)
            for s in sorted(oset - nset):
                holders = sorted(b for b in holders_new[symkind].get(s, ()) if b != num)
                if holders:
                    findings.append(Finding('MOVED', num, symkind, s,
                                            f'now added by block {", ".join(holders)}'))
                else:
                    findings.append(Finding('LOSS', num, symkind, s,
                                            'added by this block in the old set; no block of the new set '
                                            'mentions it'))
            for s in sorted(nset - oset):
                findings.append(Finding('NEW', num, symkind, s, 'added by this block (new in the set)'))
    return findings


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Reconcile every block of the delivery against the block it replaces (TODO #358).',
        epilog='See the header of this file for what is compared and why.  Exit 1 means: a file, '
               'symbol, option or block the old set carried has no counterpart in the new set, and no '
               'decision row in the allowlist names it.')
    ap.add_argument('--repo', default=repo, help='delivery repo (default: this script\'s parent)')
    ap.add_argument('--old', default=None,
                    help='the release being replaced: a git ref or a directory (default: the newest v* tag '
                         'before HEAD)')
    ap.add_argument('--new', default=None,
                    help='the new side: a git ref or a directory (default: <repo>/patches)')
    ap.add_argument('--new-chain', default=None, metavar='REPO',
                    help='take the new side from a llama.cpp checkout\'s applied block chain')
    ap.add_argument('--new-base', default=None, metavar='SHA', help='the chain\'s fork point (with --new-chain)')
    ap.add_argument('--new-tip', default=None, metavar='REF', help='the chain\'s tip (default: HEAD)')
    ap.add_argument('--rung', default=None, metavar='NNNN[,NNNN]',
                    help='compare only these block numbers (the re-base ladder use)')
    ap.add_argument('--shrink-pct', type=float, default=50.0, metavar='P',
                    help='a file whose added-line count falls by at least P%% is a SHRUNK finding '
                         '(default 50; 100 = only a drop to zero, 0 = any decrease)')
    ap.add_argument('--allow', default=None, metavar='FILE',
                    help='decision rows (default: <repo>/patches/FILE-SET-RETIREMENTS.tsv)')
    ap.add_argument('--no-allow', action='store_true', help='ignore the allowlist entirely')
    ap.add_argument('--warn-only', action='store_true', help='report findings but exit 0 (triage)')
    ap.add_argument('--allow-skip', action='store_true',
                    help='exit 0 when the old side cannot be resolved (a shallow CI clone)')
    ap.add_argument('--quiet', action='store_true', help='print findings only, no inventory')
    ap.add_argument('-v', '--verbose', action='store_true', help='print the per-block inventory')
    args = ap.parse_args(argv)

    if args.rung:
        only = {r.strip().zfill(4) for r in args.rung.split(',') if r.strip()}
        bad = [r for r in only if not re.fullmatch(r'\d{4}', r)]
        if bad:
            die(f'check-rung-file-sets: --rung wants NN[NN], got {bad}')
    else:
        only = None

    old_id, old = resolve_old(args.repo, args.old)
    if old is None:
        msg = (f'check-rung-file-sets: the previous release cannot be resolved in {args.repo} '
               f'(no v* tag and no --old): nothing to reconcile against')
        if args.allow_skip:
            print(f'SKIP: {msg} -- the release flow runs this with a full history')
            return 0
        print(f'FAIL: {msg}', file=sys.stderr)
        return 2
    new_id, new = resolve_new(args.repo, args.new, args.new_chain, args.new_base, args.new_tip)
    if not new:
        die(f'check-rung-file-sets: the new side ({new_id}) is empty')
    if args.new_chain:
        # An applied chain carries the DEFAULT blocks only -- the optional artifacts are never in it
        # (apply-all.sh --include-optional is a consumer action).  Comparing a chain against a release
        # would otherwise report every optional artifact as GONE by construction.
        if only and any('0100' <= n <= '0199' for n in only):
            die('check-rung-file-sets: --new-chain carries the default blocks only; '
                             'an optional (01xx) rung cannot be checked against a chain')
        old = {k: v for k, v in old.items() if k[0] == 'default'}

    allow_path = '' if args.no_allow else (args.allow or os.path.join(args.repo, 'patches',
                                                                     'FILE-SET-RETIREMENTS.tsv'))
    rows = load_allowlist(allow_path)
    findings = compare(old, new, args.shrink_pct, only)
    for f in findings:
        if f.cls in ('LOSS', 'SHRUNK', 'GONE'):
            f.row = allowlist_match(rows, old_id, f.block, f.kind, f.item)

    # ---- inventory ------------------------------------------------------------------------
    def totals(side):
        files = {k for p in side.values() for k in p.files}
        return (len(side), len(files),
                sum(len(p.defines) for p in side.values()),
                sum(len(p.envs) for p in side.values()),
                sum(len(p.options) for p in side.values()))

    tb, tf, td, te, to = totals(old)
    nb, nf, nd, ne, no = totals(new)
    print(f'check-rung-file-sets: old {old_id}')
    print(f'                      new {new_id}')
    if not args.quiet:
        print(f'    blocks {tb} -> {nb}   files {tf} -> {nf}   defines {td} -> {nd}   '
              f'envs {te} -> {ne}   options {to} -> {no}')
        if args.rung:
            print(f'    restricted to block(s) {", ".join(sorted(only))}')
        if args.verbose:
            for (kind, num) in sorted(set(old) | set(new)):
                if only is not None and num not in only:
                    continue
                o, n = old.get((kind, num)), new.get((kind, num))
                of = len(o.files) if o else 0
                nn = len(n.files) if n else 0
                adds_o = sum(o.files.values()) if o else 0
                adds_n = sum(n.files.values()) if n else 0
                mark = ' ' if (o and n) else ('-' if o else '+')
                print(f'      {mark} {kind:8s} {num}: files {of:3d} -> {nn:3d}   '
                      f'added lines {adds_o:6d} -> {adds_n:6d}')

    # ---- findings -------------------------------------------------------------------------
    order = {'LOSS': 0, 'GONE': 1, 'SHRUNK': 2, 'MOVED': 3, 'NEW': 4}
    findings.sort(key=lambda f: (order[f.cls], f.block, f.kind, f.item))
    shown = 0
    for f in findings:
        if f.cls in ('MOVED', 'NEW') and not args.verbose:
            continue
        shown += 1
        verdict = f.cls
        if f.row is not None:
            verdict = f'{f.cls}/ALLOWED'
        elif f.cls in ('MOVED', 'NEW'):
            verdict = f'{f.cls}/INFO'
        print(f'  {verdict:14s} block {f.block}  {f.kind:6s} {f.item}')
        print(f'                 {f.detail}')
        if f.row is not None:
            print(f'                 allowed by {os.path.basename(allow_path)}:{f.row.line} ({f.row.reason})')

    hard = [f for f in findings if f.cls in ('LOSS', 'GONE', 'SHRUNK')]
    failing = [f for f in hard if f.row is None]
    explained = [f for f in hard if f.row is not None]
    moved_n = sum(1 for f in findings if f.cls == 'MOVED')
    new_n = sum(1 for f in findings if f.cls == 'NEW')
    expired = 0 if args.no_allow else sum(1 for r in rows if r.frm not in ('*', old_id))

    def per_class(items):
        return ', '.join(f'{sum(1 for f in items if f.cls == c)} {c.lower()}'
                         for c in ('LOSS', 'GONE', 'SHRUNK'))

    print()
    print(f'==> {len(failing)} unexplained ({per_class(failing)})'
          + (f'; {len(explained)} explained by a decision row' if explained else '')
          + f'; {moved_n} moved, {new_n} new'
          + (f'; {expired} row(s) from another transition (not applied)' if expired else ''))
    if failing:
        print(f'==> FAIL: {len(failing)} unexplained finding(s) -- restore the content, or record the '
              f'retirement')
        if not args.no_allow:
            print(f'    in {allow_path} (format: <from-release>\\t<block>\\t<kind>\\t<item>\\t<reason>)')
        if not shown:
            print('    (run with -v to print the MOVED/NEW items)')
        return 0 if args.warn_only else 1
    print(f'==> OK: every file, define, env, option and block of {old_id} is accounted for in {new_id}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except BrokenPipeError:
        sys.exit(0)
