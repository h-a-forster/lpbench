"""Minimal reproduction: HiGHS PDLP (cuPDLP-C) stops early when time_limit is finite.

Standalone (needs only highspy and numpy). Builds a random feasible, bounded LP, solves it
with solver="pdlp" with no time limit and with time_limit = 10, 60 and 1000 s, and prints
the model status, the wall time and the PDLP iteration count of each run.
"""

import time

import highspy
import numpy as np

rng = np.random.default_rng(0)
m, n = 300, 400
A = rng.random((m, n)) * (rng.random((m, n)) < 0.05)
x0 = rng.random(n)
rhs = A @ x0 + 1.0  # x0 strictly feasible for A x <= rhs, 0 <= x <= 2
cost = -rng.random(n)

lp = highspy.HighsLp()
lp.num_col_, lp.num_row_ = n, m
lp.col_cost_ = cost
lp.col_lower_, lp.col_upper_ = np.zeros(n), np.full(n, 2.0)
lp.row_lower_, lp.row_upper_ = np.full(m, -highspy.kHighsInf), rhs
csc = A.T  # column-wise storage from the dense matrix
starts, index, value = [0], [], []
for j in range(n):
    nz = np.nonzero(csc[j])[0]
    index.extend(nz.tolist())
    value.extend(csc[j, nz].tolist())
    starts.append(len(index))
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = starts, index, value

print("highspy", highspy.Highs().version())
for tl in (None, 10.0, 60.0, 1000.0):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.setOptionValue("solver", "pdlp")
    if tl is not None:
        h.setOptionValue("time_limit", tl)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    iters = h.getInfo().pdlp_iteration_count
    print(f"time_limit={tl!s:>7}: {status:<20} wall {dt:7.3f} s  pdlp iterations {iters}")
