"""Environment checks: the pinned solver wheels import and solve a tiny LP on this platform.

These call the libraries directly (no lpbench adapters exist yet) and pin the dual sign
convention of docs/DESIGN.md section 5.1 for each library on one problem:

    min -x - y  s.t.  x + 2y <= 4,  3x + y <= 6,  x, y >= 0

Optimum x = (1.6, 1.2), objective -2.8, row duals y = (-0.4, -0.2), reduced costs 0,
so that z = c - A^T y = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

X_OPT = np.array([1.6, 1.2])
Y_OPT = np.array([-0.4, -0.2])
OBJ_OPT = -2.8
C = np.array([-1.0, -1.0])
A = np.array([[1.0, 2.0], [3.0, 1.0]])
TOL = 1e-5  # first-order methods stop at about 1e-6 relative


@pytest.fixture(scope="module")
def mps_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import highspy

    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = 2
    lp.num_row_ = 2
    lp.col_cost_ = C.copy()
    lp.col_lower_ = np.zeros(2)
    lp.col_upper_ = np.array([inf, inf])
    lp.row_lower_ = np.array([-inf, -inf])
    lp.row_upper_ = np.array([4.0, 6.0])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array([0, 2, 4])
    lp.a_matrix_.index_ = np.array([0, 1, 0, 1])
    lp.a_matrix_.value_ = np.array([1.0, 3.0, 2.0, 1.0])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    path = tmp_path_factory.mktemp("lp") / "small.mps"
    h.writeModel(str(path))
    return path


def _check(x: np.ndarray, y: np.ndarray, obj: float) -> None:
    np.testing.assert_allclose(x, X_OPT, atol=TOL)
    np.testing.assert_allclose(y, Y_OPT, atol=TOL)
    np.testing.assert_allclose(C - A.T @ y, 0.0, atol=TOL)
    assert abs(obj - OBJ_OPT) <= TOL


@pytest.mark.parametrize("solver", ["simplex", "ipx", "pdlp", "hipdlp"])
def test_highs(mps_path: Path, solver: str) -> None:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    assert h.readModel(str(mps_path)) == highspy.HighsStatus.kOk
    assert h.setOptionValue("solver", solver) == highspy.HighsStatus.kOk
    assert h.setOptionValue("threads", 1) == highspy.HighsStatus.kOk
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    sol = h.getSolution()
    _check(
        np.asarray(sol.col_value), np.asarray(sol.row_dual), h.getInfo().objective_function_value
    )


def test_highs_rejects_unknown_solver_value(mps_path: Path) -> None:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    assert h.setOptionValue("solver", "no-such-solver") == highspy.HighsStatus.kError


@pytest.mark.parametrize(
    ("solver", "params"),
    [
        ("glop", "random_seed: 1"),
        (
            "pdlp",
            "termination_criteria { optimality_norm: OPTIMALITY_NORM_L_INF "
            "simple_optimality_criteria { eps_optimal_absolute: 1e-7 eps_optimal_relative: 1e-7 }"
            " } num_threads: 1",
        ),
    ],
)
def test_ortools(mps_path: Path, solver: str, params: str) -> None:
    from ortools.linear_solver.python import model_builder as mb

    model = mb.Model()
    assert model.import_from_mps_file(str(mps_path))
    s = mb.Solver(solver)
    assert s.solver_is_supported()
    s.set_solver_specific_parameters(params)
    s.set_time_limit_in_seconds(30)
    status = s.solve(model)
    assert status == mb.SolveStatus.OPTIMAL
    xs = model.get_variables()
    cs = model.get_linear_constraints()
    _check(s.values(xs).to_numpy(), s.dual_values(cs).to_numpy(), s.objective_value)


def test_scip_lp(mps_path: Path) -> None:
    from pyscipopt import SCIP_PARAMSETTING, Model

    m = Model()
    m.hideOutput()
    m.readProblem(str(mps_path))
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.setSeparating(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    m.setParam("lp/threads", 1)
    m.optimize()
    assert m.getStatus() == "optimal"
    x = np.array([m.getVal(v) for v in m.getVars()])
    y = np.array([m.getDualsolLinear(c) for c in m.getConss()])
    _check(x, y, m.getObjVal())


def test_scip_rejects_unknown_parameter() -> None:
    from pyscipopt import Model

    m = Model()
    m.hideOutput()
    with pytest.raises(KeyError):
        m.setParam("no/such/parameter", 1)
