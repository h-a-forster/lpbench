# lpbench 0.1.0

First release (2026-10-30): a CPU benchmark harness for the open-source LP solvers on PyPI,
its validation, and the results of the sized-down quick study.

## What is in it

- Seven configurations with exactly pinned versions: HiGHS 1.15.1 (dual simplex, IPX,
  PDLP, HiPDLP), OR-Tools 9.15 (GLOP, PDLP) and SCIP 10 with SoPlex (PySCIPOpt 6.3.0).
- An independent verifier: every claimed optimum is checked on the original data
  (relative primal and dual residuals, duality gap, objective against Koch's exact Netlib
  optima or a three-solver MIPLIB consensus).
- A resumable runner with one solver thread per process, hard deadlines and atomic
  per-unit checkpoints; bootstrap intervals, performance profiles with confidence bands,
  Holm-corrected pairwise tests and ranking-sensitivity analyses in `lpbench report`.

## Quick study (`REPORT.md`, `docs/results/quick/`)

206 instances (92 Netlib, 114 MIPLIB 2017 relaxations with at most 1e5 nonzeros), 10 s
per solve, 2388 solves in 21.4 min on one shared Windows machine.

- `ortools-glop`, `highs-dsimplex` and `highs-ipx` cannot be ordered with confidence
  (SGM 0.251-0.260 CPU s; every pairwise ratio interval covers 1).
- At a requested 1e-6, 81 of 164 `highs-pdlp`, 91 of 164 `highs-hipdlp` and 49 of 170
  `ortools-pdlp` optimal claims fail lpbench's 1e-6 check.
- HiGHS PDLP in the highspy 1.15.1 wheel stops early with any finite time limit on Windows;
  lpbench enforces that limit externally. The cause is fixed upstream (HiGHS #3450) but
  not yet released.

## Fixes from the pre-release audit (2026-11-06)

None of these changes a published number; the quick report regenerates byte-identically
(DESIGN 12, item 24).

- The verifier no longer lets an overflowing product pass as "no violation".
- `lpbench reference` no longer blanks the committed MIPLIB values when instances are
  missing; it exits 5 and writes nothing.
- `--workers`, `--max-units` and `--bootstrap` must be at least 1 (`--workers -1` hung).
- `lpbench report` refuses a run without main-block units with a clear message, warns about
  blocks it does not analyse, and takes the option table from the unit records.
- Config files are validated strictly: unknown keys and values no code acts on are errors.
- `configs/full.toml` keeps its permutations in a `perm` block, which the report reads.
- The parser cross-check covers all 206 quick instances.
- `REPORT.md`: corrected the explanation of the trust-status ranking (`ortools-pdlp` claims
  optimality slightly more often than the HiGHS PDLP solvers, not less) and the description
  of `trento1`; explained the "numerical" column.

A second pass the same day added numbers to the tables and changed none:

- The pairwise table has a BCa interval as well; `summary.md` gives the elapsed time and the
  CPU model.
- An incomplete run scores a missing permutation unit as a failure at the time limit in the
  two-level bootstrap, not as a time of 0.
- A bad config ends with a message and exit 2 instead of a traceback, and
  `scripts/run_study.py` does not retry it.
- `REPORT.md` explains why most "reference only" failures sit just above the threshold.

A third pass corrected a headline; again no published number changed:

- The verification table has a "fail at 1e-5" column, derived from the stored errors. The
  README and `REPORT.md` no longer give "about half of PDLP's optimal answers fail" without
  its context: that rate has no tolerance headroom, and one decade looser it is 11%, 13% and
  6%.
- The pairwise table has a wall-time p-value. `REPORT.md` no longer reads the top three's
  CPU-time Wilcoxon tests, which rest on a 15.6 ms timer step.

## Limits

One machine, small instances and a short limit; nothing here describes first-order methods
at scale, multi-threaded or GPU solvers. The full-size study (`configs/full.toml`) has not
been run.
