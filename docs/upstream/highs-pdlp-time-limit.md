# Upstream: HiGHS PDLP stops early with a finite `time_limit`

**Status (2026-11-06): fixed upstream, not yet released. No new report is needed.**

- Verified on Windows only (Windows 11, x86-64, Python 3.13, highspy 1.15.1 wheel). Not
  checked on Linux or macOS.
- highspy 1.15.1 (2026-07-02) is still the latest release on PyPI, and the reproduction in
  [docs/repros/highs-pdlp-time-limit.md](../repros/highs-pdlp-time-limit.md) still shows
  the early stop with it.
- HiGHS pull request [#3450](https://github.com/ERGO-Code/HiGHS/pull/3450), "Fix cuPDLP-C
  time limit", merged into the `latest` branch on 2026-10-22, fixes the cause. On Windows,
  cuPDLP-C's `my_clock()` returned raw `clock()` ticks (milliseconds with MSVC) that were
  compared with `time_limit` in seconds, so the limit fired about 1000 times too early.
  Other platforms use `time(NULL)` in seconds, which is consistent with the problem being
  Windows-only. The same pull request also passes the remaining time, not the full
  `time_limit`, to cuPDLP-C.

## What to do when a release contains the fix

1. Bump `highspy` in `pyproject.toml` and `uv.lock` (its own change; it invalidates the
   published results until they are re-run).
2. Run `uv run python docs/repros/highs_pdlp_time_limit.py`: every run with a finite limit
   should end `Optimal` with 4960 iterations, as with no limit.
3. `tests/test_adapters.py::test_upstream_highs_pdlp_time_limit_stops_early` (slow) then
   fails. Remove it and the external time limit for `highs-pdlp` (DESIGN 12, item 14), and
   record the change in DESIGN section 12.

## Text for a comment on #3450 (optional)

Use only if the maintainers ask for independent confirmation; the pull request is already
merged.

> An independent, standalone reproduction (highspy and numpy only, in-memory random LP,
> no files) of the behaviour this fixes, on Windows 11 x86-64 with the highspy 1.15.1
> wheel and Python 3.13. With `solver = "pdlp"`, a 10 s `time_limit` ends with "Time
> limit reached" after 0.02-0.08 s; with no limit or a 1000 s limit the same LP is
> Optimal in about 0.2 s. HiPDLP, simplex and IPX keep the same limit. Script and output:
> https://github.com/h-a-forster/lpbench/blob/main/docs/repros/highs-pdlp-time-limit.md.
> Not checked on Linux or macOS.
