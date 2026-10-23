# AGENTS.md

Guide for agents and contributors continuing this project. The authoritative spec is
[docs/DESIGN.md](docs/DESIGN.md). Read it before changing any formula, tolerance, instance
rule or reporting rule. If you deviate from it, update DESIGN.md (section 12) in the same
change.

## What this is

An honest CPU benchmark of open-source LP solvers installable from PyPI: HiGHS 1.15.1 (dual
simplex, IPX, PDLP, HiPDLP), OR-Tools 9.15 (GLOP, PDLP) and SCIP 10 / SoPlex 8 via PySCIPOpt
6.3.0, on Netlib and size-capped MIPLIB 2017 benchmark LP relaxations. The contribution is
the methodology: independent verification of every reported optimum, performance
variability across permutations and seeds, bootstrap intervals on shifted geometric means
and performance profiles, and an analysis of what changes the ranking. No GPU claims.

## Conventions

- uv + hatchling, `src/lpbench`, Python 3.11+ (3.13 is the canonical study environment:
  `.python-version`). Typed code, ruff (line length 100), pytest.
- Solver distributions are pinned exactly in `pyproject.toml`; `uv.lock` is committed. A
  version bump is its own change and invalidates published results until re-run.
- CPU only, at most 3 worker processes, one solver thread each. No LLM or model API calls.
- Network access only in `lpbench fetch`, only to netlib.org and miplib.zib.de, with
  timeouts, a total download cap (2 GB) and recorded SHA-256 checksums.
- Never commit `cache/`, `runs/`, `data/` (root), downloaded instances or prepared MPS.
  Commit only names, sizes, checksums, reference values and small derived results under
  `docs/results/`. The instance licences are unstated (DESIGN 3.1, 3.2).
- Long computations checkpoint each finished unit to disk (atomic write), skip completed
  units on rerun, log progress to `runs/<study>/progress.log`, and run under the retry
  wrapper `scripts/run_study.py`. Each study run stays under ~45 minutes of wall time;
  the full-size command is documented in DESIGN section 8.
- Scratch files go in `tmp/` (gitignored). Scrub local paths from any log before committing.
- Statistical honesty: report what was run (instances, configs and options, epsilon, TL, K,
  seeds, versions, machine, wall time); intervals with point estimates; Holm within the
  families of DESIGN section 7. "Claimed optimal but failed verification" is a tolerance
  outcome unless the bug-claim protocol (DESIGN 6.4 item 5) is complete.
- Commits: end messages with the project's co-author trailer. Never push, create remotes or
  publish without the owner's go-ahead.

## Architecture map

| Path | Role | State |
|---|---|---|
| `src/lpbench/cli.py` | `versions` (done); `fetch`, `prepare`, `reference`, `plan`, `run`, `status`, `report` exit 2 until implemented | partial |
| `src/lpbench/versions.py` | library versions, SCIP/HiGHS lib versions, platform, git SHA | done |
| `src/lpbench/config.py` | StudyConfig, validation, config hash | M4 |
| `src/lpbench/lpmodel.py` | CanonicalLP, HiGHS extraction, permutations, MPS output | M1 |
| `src/lpbench/verify.py` | residuals, dual objective, gap, evaluation floor | M1 |
| `src/lpbench/netlib_emps.py` | netlib compressed format expander with checksums | M2 |
| `src/lpbench/instances.py`, `download.py` | suites, manifests, caps, checksummed download | M2 |
| `src/lpbench/solvers/` | adapters (highs, ortools, scip), registry with tolerance table | M3 |
| `src/lpbench/worker.py`, `runner.py` | persistent workers, hard deadlines, checkpoints | M4 |
| `src/lpbench/reference.py` | consensus references for MIPLIB relaxations | M5 |
| `src/lpbench/stats.py` | SGM, bootstrap, profiles with bands, ranks, Holm | M6 |
| `src/lpbench/report.py` | tables, figures, ranking sensitivity | M7 |
| `configs/quick.toml`, `configs/full.toml` | study configs (draft keys until M4) | draft |
| `tests/test_solver_env.py` | pinned wheels solve a tiny LP; dual sign convention | done |
| `tests/test_scaffold.py` | metadata, CLI, provenance hygiene | done |
| `docs/DESIGN.md` | spec | done (v0.0.1) |

## How to test

```sh
uv sync
uv run pytest                # fast suite
uv run pytest --runslow      # adds slow tests
uv run ruff check .
uv run ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs on `pull_request` and `workflow_dispatch` only:
ubuntu py3.11, ubuntu py3.13, windows py3.13, plus a lint job.

## How to run the study (once M2-M7 land)

```sh
uv run lpbench fetch --config quick
uv run lpbench prepare --config quick
uv run lpbench reference --config quick
uv run lpbench plan --config quick
uv run python scripts/run_study.py --config quick --out runs/quick   # target < 45 min, 3 workers
uv run lpbench report runs/quick --publish docs/results/quick
uv run python scripts/run_study.py --config full --out runs/full     # several hours
```

## Status and next steps

- **Done (M0):** scaffold (uv, hatchling, pinned solvers, lock, CI, hygiene files),
  `lpbench versions`, environment tests for all 7 configurations through the raw libraries,
  DESIGN.md, draft configs.
- **Findings from probing the wheels (2026-10-24), to re-check in M3:** HiGHS HiPO is not
  available in the highspy 1.15.1 wheel (built without BLAS/METIS); `solver="ipm"` maps to
  IPX; HiPDLP declared "Optimal" on a 2x2 LP at 1.05x its own relative primal tolerance;
  OR-Tools returns `INVALID_SOLVER_PARAMETERS` for bad option strings; SCIP needs presolve
  off for duals.
- **Next: M1** `lpmodel.py` + `verify.py` with the tests of DESIGN 6.1. Then M2 (netlib
  expander, MIPLIB manifest, fetch, prepare, Koch reference values transcribed from the
  open-access ZIB-Report 03-05 with readme cross-check).
- **Open questions:** whether the Koch (2004) table can be transcribed reliably (fallback:
  netlib readme values, documented); the share of first-order units that hit the 10 s time
  limit, which decides whether the quick study fits 45 minutes (M5 pilot, cut order in
  DESIGN 8).
