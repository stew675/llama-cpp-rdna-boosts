# utilities — measurement and reporting tools

Tools that measure or report on this stack. **Nothing here is part of the delivery set** in `patches/`,
nothing here is applied by `scripts/apply-all.sh`, and nothing here is covered by the release gates.
They are consumer- and maintainer-facing diagnostics: they read, they measure, they write a report, and
they change nothing.

| tool | what it answers |
|---|---|
| [`storage-oracle/`](storage-oracle/) | Given a model file, which `--load-mode`/`--lazy-mode` should you use on the device that file lives on — and which configurations should you avoid? Measures the file (band map, cold per-page latency for `O_DIRECT`/buffered/`mmap`, sequential throughput, idle-gap wake cost, cold reads under a stream, page-cache residency, RAM/`RLIMIT_MEMLOCK` budget), writes a markdown report with the measurement behind every verdict, and can verify the prediction against `llama-cli` head-to-head. |

Adding a tool here:

- it must be read-only (measure, never tune or apply), need no root, and leave the machine as it found it;
- it must print the numbers it decided from, and say `UNKNOWN` rather than guess when it cannot measure;
- it gets its own sub-directory with a `README.md`; a committed example output is welcome
  (`storage-oracle/EXAMPLE-REPORT.md`);
- if what it measures turns out to matter to the delivery, the *finding* goes to `INVARIANTS.md` and the
  dated record to `WORKLOG.md` — the tool stays here.
