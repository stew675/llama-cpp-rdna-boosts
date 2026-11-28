# utilities — measurement and reporting tools

Tools that measure or report on this stack. **Nothing here is part of the delivery set** in `patches/`,
nothing here is applied by `scripts/apply-all.sh`, and nothing here is covered by the release gates.
They are consumer- and maintainer-facing diagnostics: they read, they measure, they write a report, and
they change nothing.

| tool | what it answers |
|---|---|
| [`storage-oracle/`](storage-oracle/) | Given a model file, which `--load-mode`/`--lazy-mode` should you use on the device that file lives on — and which configurations should you avoid? Measures the file (band map, cold per-page latency for `O_DIRECT`/buffered/`mmap`, sequential throughput, idle-gap wake cost, cold reads under a stream, page-cache residency, RAM/`RLIMIT_MEMLOCK` budget), writes a markdown report with the measurement behind every verdict, and can verify the prediction against `llama-cli` head-to-head. |
| [`h2d-latency/`](h2d-latency/) | What the host->device path costs in **microseconds** at the sizes a model actually moves (4 KiB to 512 MiB, pageable and pinned), plus the 4-byte round-trip latency floor. The delivery's own calibration reports one number, bandwidth at 512 MiB; the staged fill issues ~11 000 DMAs of ~1.9 MB per 8192-token ubatch and the gather reads ~0.5 MB per expert, so the 64 KiB and 1 MiB rows are what separate two boxes with similar bandwidth (issue #368). One file, `hipcc`, no model, no root. |

Adding a tool here:

- it must be read-only (measure, never tune or apply), need no root, and leave the machine as it found it;
- it must print the numbers it decided from, and say `UNKNOWN` rather than guess when it cannot measure;
- it gets its own sub-directory with a `README.md`; a committed example output is welcome
  (`storage-oracle/EXAMPLE-REPORT.md`);
- if what it measures turns out to matter to the delivery, the *finding* goes to `INVARIANTS.md` and the
  dated record to `WORKLOG.md` — the tool stays here.
