"""Exactly classified claims of MIP solvers on MIPLIB 2017 instances (DESIGN section 13).

A run records each solver's claims (status, solution, primal and dual bound, effective
tolerances); the checks read the instance's MPS decimals as rationals and classify every
claim exactly: the solution's feasibility class, an exact completion of its integer part,
and refutations of dual bounds by exactly feasible points.
"""
