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

## Limits

One machine, small instances and a short limit; nothing here describes first-order methods
at scale, multi-threaded or GPU solvers. The full-size study (`configs/full.toml`) has not
been run.
