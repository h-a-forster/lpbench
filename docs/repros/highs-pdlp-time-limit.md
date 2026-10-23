# HiGHS PDLP stops early when `time_limit` is finite

**Status:** reproduced on one machine and one platform (Windows 11, x86-64, Python 3.13,
highspy 1.15.1 wheel). It has not been checked on Linux or macOS and has not been reported
upstream. lpbench works around it (DESIGN 12, item 14).

## Observation

With `solver = "pdlp"` (cuPDLP-C inside HiGHS), any finite `time_limit` makes the solve
stop with "Time limit reached" long before that much time has passed. The PDLP log shows
an elapsed time of about 11 s at the stopping iteration after 0.02 s of wall time. The
iterations completed grow with the limit, so the limit is applied, but on a clock running
roughly 500-1000 times too fast. With no limit, or a large one (1000 s), the same LP solves
to optimality in under 0.2 s. HiGHS's other solvers, including `hipdlp`, keep to the same
limit (a 10 s limit stops them after 10.0 s).

## Minimal reproduction

[`highs_pdlp_time_limit.py`](highs_pdlp_time_limit.py) needs only `highspy` and `numpy`.
It builds a random feasible, bounded LP in memory (no files, no lpbench code) and solves it
four times. Output on the machine above:

```
highspy 1.15.1
time_limit=   None: Optimal              wall   0.192 s  pdlp iterations 4960
time_limit=   10.0: Time limit reached   wall   0.019 s  pdlp iterations 240
time_limit=   60.0: Time limit reached   wall   0.070 s  pdlp iterations 1904
time_limit= 1000.0: Optimal              wall   0.194 s  pdlp iterations 4960
```

## Why this is not an lpbench adapter mistake

- The reproduction calls highspy directly on an in-memory LP; no lpbench code, MPS writer or
  option table is involved.
- The only option that changes between runs is `time_limit`. The lpbench adapter showed the
  same thing on Netlib AGG2 read from MPS: "Time limit reached" after 0.02 s with
  `time_limit = 10`, "Optimal" after 0.08 s without it.
- `tests/test_adapters.py::test_upstream_highs_pdlp_time_limit_stops_early` (slow) keeps
  checking the behaviour. It will fail once the behaviour changes, which is the cue to
  remove the workaround.

## Effect on lpbench and the workaround

Before the workaround, 26 of the 30 `highs-pdlp` units in the pilot ended as TIME_LIMIT
within 0.26 s. A benchmark run that way would have measured the bug, not the solver. Now
`highs-pdlp` gets no in-library time limit. The runner kills its worker at
`time_limit + 5 s` and records TIME_LIMIT. A solve that finishes on its own after the limit
is also scored TIME_LIMIT. Compared with the other configs, `highs-pdlp` can therefore run
up to 5 s past the limit on a unit it fails. That costs wall time only; it does not change
the scoring, because every failure is scored at the time limit.
