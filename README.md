# lpbench

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A CPU benchmark of the open-source LP solvers you can `pip install`: HiGHS (dual simplex,
IPX, PDLP, HiPDLP), OR-Tools (GLOP, PDLP) and SCIP with SoPlex. It runs on Netlib and on LP
relaxations of the MIPLIB 2017 benchmark set, checks every reported optimum itself, and
puts intervals on every comparison.

Typical LP comparisons report solved counts and a shifted geometric mean from one run per
instance, and take each solver's "optimal" status on trust. lpbench instead:

- **verifies every claimed optimum** with its own code on the original problem data:
  relative primal and dual residuals and duality gap at one common tolerance, and the
  objective against the exact Netlib optima (Koch, 2004) or a three-solver consensus for
  the MIPLIB relaxations;
- **measures performance variability** across row and column permutations, comparing
  iteration-count (algorithmic) variability with time variability and with repeated
  identical runs (machine noise);
- **puts intervals on everything**: bootstrap intervals on shifted geometric means and
  their ratios, simultaneous confidence bands on performance profiles, rank probabilities,
  and Holm-corrected pairwise tests;
- **reports what changes the ranking**: trusting solver status, the tolerance, the shift,
  the time limit, the failure penalty, the instance suite, the permutation, and CPU versus
  wall time;
- **runs safely**: one solver thread per process, hard deadlines with kill-and-respawn,
  atomic per-unit checkpoints, resumable runs, and deterministic, byte-identical reports.

Scope: one CPU machine, one thread per solve, the pinned library versions. lpbench makes no
claims about GPU solvers, multi-threaded performance or commercial solvers.

## Results of the quick study

206 instances (92 Netlib, 114 MIPLIB relaxations with at most 1e5 nonzeros), 7 configs,
10 s per solve, 2388 solves in 21 minutes. Full write-up with limitations:
[REPORT.md](REPORT.md). Generated tables: [docs/results/quick/](docs/results/quick/summary.md).

| config | verified at 1e-6 (of 206) | SGM, CPU s [95% CI] |
|---|---|---|
| ortools-glop | 205 | 0.251 [0.177, 0.339] |
| highs-dsimplex | 200 | 0.253 [0.176, 0.348] |
| highs-ipx | 206 | 0.260 [0.199, 0.328] |
| scip-soplex (presolve off) | 198 | 0.406 [0.296, 0.534] |
| ortools-pdlp | 121 | 2.53 [2.06, 3.07] |
| highs-pdlp | 83 | 3.64 [2.99, 4.38] |
| highs-hipdlp | 73 | 3.96 [3.29, 4.76] |

- The three fastest configs cannot be ordered with confidence. Their pairwise SGM-ratio
  intervals all cover 1, and their order changes with the suite, the shift, the failure
  penalty and between runs.
- At a requested tolerance of 1e-6, about half the "optimal" answers of the HiGHS PDLP
  solvers and 29% of those from OR-Tools PDLP fail lpbench's 1e-6 check. Nearly all pass
  at 1e-4. One OR-Tools PDLP "optimal" answer (x = 0 on `trento1`) passes the relative
  residual check and is caught only by the reference objective.
- With a finite `time_limit`, HiGHS PDLP in the highspy 1.15.1 wheel stopped after a tiny
  fraction of the limit on Windows (standalone reproduction:
  [docs/repros/](docs/repros/highs-pdlp-time-limit.md)). lpbench enforces that config's
  limit externally.

These numbers hold for small instances and a short limit, on one shared Windows machine;
they say nothing about first-order methods at scale.

## Install

Not on PyPI. The study needs the configs and data manifests in the repository, so clone it
and let [uv](https://docs.astral.sh/uv/) create the environment with the exactly pinned
solver versions from `uv.lock`:

```sh
git clone https://github.com/h-a-forster/lpbench
cd lpbench
uv sync
uv run lpbench versions
```

Python 3.11+; Python 3.13 is the study environment. A clone is needed because the configs
and the run directories live next to the code.

## Quick start

```sh
uv run lpbench fetch --config quick      # Netlib + MIPLIB from netlib.org / miplib.zib.de, checksummed
uv run lpbench prepare --config quick    # canonical LPs and MPS files in cache/
uv run lpbench reference --config quick  # optional: recompute the MIPLIB consensus values
uv run python scripts/run_study.py --config pilot --out runs/pilot   # ~1 min, for the projection
uv run lpbench plan --config quick --pilot runs/pilot                # unit counts, projected wall time
uv run python scripts/run_study.py --config quick --out runs/quick --pilot runs/pilot   # resumable
uv run lpbench status runs/quick
uv run lpbench report runs/quick --publish docs/results/quick
```

Instances are downloaded into `cache/` (gitignored); this repository redistributes no
problem data. `lpbench reference --config quick` recomputes the MIPLIB consensus values
(committed in `src/lpbench/data/miplib_reference.csv`), and `configs/pilot.toml` is the
pilot used for the run-time projection. Without `--pilot`, `lpbench run` budgets every unit
at its hard deadline (332 min for quick), refuses to start beyond the 45 min budget and exits
with status 3; pass `--pilot runs/pilot` (or `--allow-long`). An interrupted run resumes where
it stopped: rerun the same command. `lpbench report` warns when a run is incomplete.

Example output:

```text
$ uv run lpbench status runs/quick
quick: 2388/2388 units recorded
  NUMERICAL      57
  OPTIMAL        2139
  TIME_LIMIT     192

$ uv run lpbench report runs/quick --publish docs/results/quick
report for quick: 2388 units -> docs/results/quick
primary ranking: ortools-glop < highs-dsimplex < highs-ipx < scip-soplex < ortools-pdlp < highs-pdlp < highs-hipdlp
```

The report writes `summary.md` (all tables), `results.json`, `units.csv` (one row per solve)
and `performance_profiles.svg`.

## How the harness is validated

- **Verifier.** Exact values on hand-built LPs with every row and bound type (minimisation
  and maximisation), randomly generated LPs with a known optimum built from the KKT
  conditions, perturbation tests, and an evaluation floor that flags results limited by
  floating-point accuracy.
- **Instance pipeline.** An independent expander for Netlib's compressed format: all 92
  files expand with zero checksum errors, and the nonzero counts match Koch's and the Netlib
  readme. HiGHS dual simplex reproduces Koch's exact optimum to 1e-10 relative on all 92.
  OR-Tools and SCIP read lpbench's MPS files exactly.
- **Adapters.** Every config solves the hand-built and generated LPs and passes the
  verifier. Infeasible and unbounded problems map to the right status, invalid options are
  reported as CONFIG_ERROR, and time limits are tested. Upstream behaviour that lpbench
  works around has a test that fails once it changes.
- **Runner.** Tests for kill-and-resume without duplicate records, hard deadlines, crashing
  workers, external time limits and config-change detection.
- **Statistics.** Hand-computed values for SGM, Holm and profiles, plus a coverage check: the
  95% bootstrap intervals cover a known SGM ratio at a rate consistent with 0.95 over 1000
  simulated studies.

```sh
uv sync --group dev
uv run pytest                 # fast suite (as in CI)
uv run pytest --runslow       # plus slow checks (needs the fetched instances)
uv run ruff check . && uv run ruff format --check .
```

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): scope, verification formulas, tolerance alignment,
  validation, statistics, study design and every deviation from the plan (section 12).
- [REPORT.md](REPORT.md): results of the quick study, with intervals and limitations.
- [docs/repros/](docs/repros/highs-pdlp-time-limit.md): minimal reproductions of library
  behaviour that lpbench works around.
- [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the solvers (HiGHS, OR-Tools, SCIP and
SoPlex), the instance collections (Netlib; MIPLIB 2017, Gleixner et al. 2021) and Koch
(2004) for the exact Netlib optima.

## Licence

MIT. See [LICENSE](LICENSE).
