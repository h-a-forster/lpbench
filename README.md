# lpbench

An honest CPU benchmark of the open-source LP solvers you can `pip install`: HiGHS (dual
simplex, IPX, PDLP, HiPDLP), OR-Tools (GLOP, PDLP) and SCIP with SoPlex, on Netlib and on
LP relaxations of the MIPLIB 2017 benchmark set.

**Status: pre-alpha. The design is written; the harness is not implemented and no results
exist yet.** See [docs/DESIGN.md](docs/DESIGN.md) for the specification and milestones and
[AGENTS.md](AGENTS.md) for the current status.

## What it does differently

Typical LP comparisons report solved counts and a shifted geometric mean of run times from
one run per instance, and take each solver's "optimal" status on trust. lpbench:

- **verifies every reported optimum** with its own code against the original problem data:
  primal and dual residuals and the duality gap at one common tolerance, and the objective
  against exact Netlib values (Koch, 2004) or a multi-solver consensus for MIPLIB relaxations;
- **measures performance variability** across row/column permutations and solver seeds,
  separating algorithmic variability (iteration counts) from machine noise;
- **puts intervals on everything**: bootstrap intervals on shifted geometric means and their
  ratios, confidence bands on performance profiles, rank probabilities, Holm-corrected
  pairwise tests;
- **reports what changes the ranking**: trusting solver status, tolerance, shift, time
  limit, failure penalty, instance suite, permutation, CPU versus wall time.

Scope: one CPU machine, one thread per solve, the pinned library versions. No claims about
GPU solvers, multi-threaded performance or commercial solvers.

## Planned usage

```sh
uv sync
uv run lpbench versions                      # works today
uv run lpbench fetch --config quick          # planned (M2): Netlib + MIPLIB, checksummed
uv run lpbench prepare --config quick        # planned (M2)
uv run python scripts/run_study.py --config quick --out runs/quick   # planned (M4), < 45 min
uv run lpbench report runs/quick --publish docs/results/quick        # planned (M7)
```

Instances are downloaded from netlib.org and miplib.zib.de into `cache/` (gitignored); no
problem data is redistributed in this repository.

## Development

```sh
uv sync
uv run pytest               # fast suite
uv run pytest --runslow     # adds slow tests
uv run ruff check .
uv run ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Licence: MIT.
