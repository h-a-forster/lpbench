# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Project scaffold: uv project with exactly pinned solver versions (highspy 1.15.1,
  ortools 9.15.6755, pyscipopt 6.3.0) and `uv.lock`, CI (pull request and manual dispatch),
  hygiene files.
- `docs/DESIGN.md`: scope, verification formulas, validation strategy, statistics, study
  design and milestones.
- `lpbench versions`: library versions and provenance.
- Environment tests that solve a tiny LP with each solver configuration through the raw
  libraries and pin the dual sign convention.
- Draft study configs `configs/quick.toml` and `configs/full.toml`.
