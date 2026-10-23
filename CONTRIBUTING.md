# Contributing

Bug reports, solver adapters, instance suites and documentation fixes are welcome. Read
[docs/DESIGN.md](docs/DESIGN.md) first: it fixes the verification formulas, the tolerance
alignment, the statistics and the reporting rules.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. From a clone:

```sh
uv sync
uv run lpbench versions
```

Python 3.13 is the canonical study environment (`.python-version`); 3.11 and 3.12 are
supported for development and CI.

## Tests and lint

```sh
uv run pytest              # fast suite, must stay under a few minutes
uv run pytest --runslow    # slow tests (time limits, all-instance checks, coverage checks)
uv run ruff check .
uv run ruff format --check .
```

CI runs the fast suite on Ubuntu (Python 3.11 and 3.13) and Windows (Python 3.13), plus
lint. Tests must run offline, write only under pytest's `tmp_path`, use fixed seeds and pass
on Windows (use `pathlib`). Tests that need downloaded instances skip when the cache is
absent. Mark tests that take more than a few seconds with `@pytest.mark.slow`. Statistical
assertions use an explicit exact interval, never a hand-picked tolerance.

## Code

- Python 3.11+, typed; ruff enforces style (line length 100). Write text files as UTF-8.
- Solver versions are pinned exactly. Bump one only in its own pull request, with the
  environment and adapter tests passing, and re-run the study before publishing results.
- Open an issue before adding a dependency.
- Never commit downloaded instances, prepared MPS files, caches or study runs; only small
  derived results under `docs/results/`.

## Solver adapters

An adapter (docs/DESIGN.md, sections 4 and 6.4) must, before any of its numbers are
reported: return verified solutions on the hand-built LPs with duals in the documented sign
convention; map infeasible, unbounded and time-limit statuses correctly; and turn invalid
options into a configuration error.

An apparent solver defect needs a minimal standalone reproduction that calls the library
directly on the original downloaded file, plus an independent residual check showing it is
not an adapter mistake (DESIGN 6.4, item 5). Report confirmed defects to the solver's
maintainers.

## Pull requests

- One change per pull request, with tests and docs.
- Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- Keep the "Status and next steps" section of [AGENTS.md](AGENTS.md) current.

Security issues: see [SECURITY.md](SECURITY.md). Do not open a public issue.
