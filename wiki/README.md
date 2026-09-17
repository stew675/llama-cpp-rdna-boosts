# Wiki authoring notes

This directory holds the source for the
[project wiki](https://github.com/stew675/llama-cpp-rdna-boosts/wiki). The wiki lives in its own git
repository (`llama-cpp-rdna-boosts.wiki.git`); these files are the canonical, reviewable copy.

## Files

| File | Wiki page |
|---|---|
| `Home.md` | **Home** — the landing page (`Home` is the wiki's front page) |
| `MTP-and-Adaptive-MTP.md` | **MTP & Adaptive MTP** — the deep dive |
| `MTP-Quick-Reference.md` | **MTP Quick Reference** — flags and commands |
| `_Sidebar.md` | the wiki navigation sidebar |
| `README.md` | this file (do **not** copy it to the wiki) |

## Publishing

A GitHub wiki has to be initialised once from the web UI before its git remote exists. If
`git ls-remote git@github.com:stew675/llama-cpp-rdna-boosts.wiki.git` fails with "Repository not
found", open <https://github.com/stew675/llama-cpp-rdna-boosts/wiki> and create any first page.

After that:

```bash
git clone git@github.com:stew675/llama-cpp-rdna-boosts.wiki.git /tmp/rdna-wiki
cp Home.md MTP-and-Adaptive-MTP.md MTP-Quick-Reference.md _Sidebar.md /tmp/rdna-wiki/
cd /tmp/rdna-wiki
git add -A
git commit -m "Sync wiki from the delivery repo"
git push
```

Keep the wiki a mirror: edit these files in the repo, then re-run the copy. That keeps the content
under review and inside the normal change history.

## Style

- GitHub-flavored Markdown; tables and blockquotes carry most of the structure.
- Link between pages by page name: `[MTP & Adaptive MTP](MTP-and-Adaptive-MTP)`.
- Link to repo files with absolute URLs (`.../blob/main/...`) — the wiki is a separate repository, so
  relative paths do not resolve.
- Numbers must be traceable to a dated record in `benchmarks/` or `patches/README.md`. Do not invent,
  round-trip, or silently update a figure here; update the underlying record and cite it.
- Never present a measurement as a guarantee. The text/acceptance contract and the logits-level
  measurements are different things — say which one is being claimed.
