# MIP claims: mip-pilot

40 claim records of 40 expected units; time limit 60 s, seed 0, 3 workers; elapsed wall time 11.9 min.
Solver versions: highs 1.16.0, scip SCIP 10.0.

## Estimands

95% Wilson intervals; with zero events the interval is [0, 3.84/(n + 3.84)].

- E1: returned solution not exactly feasible (class P1-P3), per unit with a solution.
- E2: returned solution beyond the solver's own tolerances (P2-P3), per unit with a solution.
- E3: rounded solution has no exactly feasible completion, per unit with a solution.
- E4: dual claim refuted with severity above 1e-09, per unit with a dual claim.

| solver | estimand | events | resolved | unresolved | rate | Wilson 95% | unresolved as events |
|---|---|---|---|---|---|---|---|
| highs | E1 | 15 | 17 | 0 | 0.882 | [0.657, 0.967] | - |
| highs | E2 | 0 | 17 | 0 | 0.000 | [0.000, 0.184] | - |
| highs | E3 | 0 | 17 | 0 | 0.000 | [0.000, 0.184] | - |
| highs | E4 | 0 | 18 | 2 | 0.000 | [0.000, 0.176] | [0.028, 0.301] |
| scip | E1 | 7 | 18 | 0 | 0.389 | [0.203, 0.614] | - |
| scip | E2 | 0 | 18 | 0 | 0.000 | [0.000, 0.176] | - |
| scip | E3 | 1 | 18 | 0 | 0.056 | [0.010, 0.258] | - |
| scip | E4 | 0 | 18 | 2 | 0.000 | [0.000, 0.176] | [0.028, 0.301] |

## Outcomes per solver

Time limits are wall time; a single-threaded solve on a free core gets close to 1 CPU second per wall second.

- highs: statuses {'INFEASIBLE': 1, 'OPTIMAL': 3, 'TIME_LIMIT': 16}; primal classes {'P0': 2, 'P1': 15}; completions {'feasible': 17, 'none': 3}; claim fields missing none; objective misstated 0; median solve CPU seconds per wall second 0.17.
- scip: statuses {'INFEASIBLE': 1, 'OPTIMAL': 3, 'TIME_LIMIT': 16}; primal classes {'P0': 11, 'P1': 7}; completions {'feasible': 17, 'infeasible': 1, 'none': 2}; claim fields missing none; objective misstated 0; median solve CPU seconds per wall second 0.16.

## Refutations above 1e-9

None.

## Data rounding

- mip__mas74__scip__s0: primal_class

## Projected wall time of the full benchmark set

480 units (240 instances x 2 solvers) at 60 s on 3 workers: 2.6 h (units on instances outside the pilot run to the time limit).
