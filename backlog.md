# Backlog

## In progress

## Waiting on me

- `polish-2026-10-30` is merged into `main` through a pull request, so CI runs for the first time, and `v0.1.0` is tagged.
- Private vulnerability reporting is enabled on GitHub, or `SECURITY.md` names another channel: it points at a Security tab form that is off.
- The GitHub repository description no longer says "seeds and permutations": solver seeds are not varied.
- A decision on the `Co-Authored-By` trailers in the 13 public `main` commits: leave them, or rewrite history with a force push.

## Decided

- 2026-10-30: nearest-neighbour run-time projection rejected; the tertile means stay (DESIGN 12, item 23).
- 2026-10-30: nothing to file upstream about the HiGHS PDLP time limit; HiGHS #3450 already fixes it.

## Proposed

- A decision, before the full study, on preregistering a componentwise primal pass rule (DESIGN 12, item 18): 29-58 verified first-order solves per config fail it at 1e-6.
- The full study (`configs/full.toml`) is run in chunks of under 45 min each, which also gives time-variability results on more than the 3-5 instances above 0.5 s per config.
- `lpbench report` analyses every block of the full config: `eps1e-8` enters no table today, and the `eps1e-4` tolerance table and analysis (c) assume Netlib only.
- The reference pass stops on CPU time or a longer limit, so that reference values no longer depend on machine load (a 60 s wall limit).
- The `highs-pdlp` external deadline counts solve time only: it also counts MPS reading and verification, which is harmless in the quick study but shortens the budget on large instances in the full study.
- highspy is bumped to the first release with HiGHS #3450, following `docs/upstream/highs-pdlp-time-limit.md` (bump, re-run the repro, drop the external limit and its upstream test).
