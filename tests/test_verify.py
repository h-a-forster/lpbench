"""Verifier tests (DESIGN 6.1): exact optima, perturbations, thresholds, floor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lpbench.lpmodel import INF, make_lp
from lpbench.synthetic import all_types_lp, random_known_lp, with_free_row
from lpbench.verify import objective_agrees, objective_rel_error, scales, verify


@pytest.mark.parametrize("maximize", [False, True])
def test_all_types_optimum_exact(maximize: bool) -> None:
    k = all_types_lp(maximize)
    v = verify(k.lp, k.x, k.y)
    assert v is not None
    assert v.primal_residual == 0.0
    assert v.dual_residual == 0.0
    assert v.primal_objective == -5.0
    assert v.dual_objective == -5.0
    assert v.rel_gap == 0.0
    assert v.passes(1e-12)
    assert k.lp.source_objective(v.primal_objective) == (5.0 if maximize else -5.0)


def test_scales_of_all_types() -> None:
    lp = all_types_lp().lp
    # finite bounds: rows 3,1,1,1,1,3 ; columns 0,3,1,4,2,2 -> max 4
    assert scales(lp) == (5.0, 6.0)


def test_free_row_with_zero_dual_passes_and_nonzero_dual_fails() -> None:
    k = with_free_row(all_types_lp())
    v = verify(k.lp, k.x, k.y)
    assert v is not None and v.passes(1e-12)
    y = k.y.copy()
    y[-1] = 0.5  # a multiplier on a free row is dual infeasible
    v2 = verify(k.lp, k.x, y)
    assert v2 is not None
    assert v2.dual_residual >= 0.5
    assert not v2.passes(1e-3)


def test_primal_shift_raises_residual_by_delta() -> None:
    k = all_types_lp()
    delta = 2.0**-20
    x = k.x.copy()
    x[1] += delta  # x2 above its upper bound 3; rows 1 and 3 also violated by delta
    v = verify(k.lp, x, k.y)
    assert v is not None
    assert v.primal_residual == delta
    assert v.rel_primal == delta / 5.0
    # primal residual dominates: gap is delta / 11
    assert v.rel_gap == pytest.approx(delta / 11.0, rel=1e-12)
    assert v.max_rel == v.rel_primal
    # pass/fail switches exactly at the threshold
    assert v.passes(v.rel_primal)
    assert not v.passes(math.nextafter(v.rel_primal, 0.0))


def test_lower_bound_violation() -> None:
    k = all_types_lp()
    x = k.x.copy()
    x[0] = -0.25
    v = verify(k.lp, x, k.y)
    assert v is not None
    assert v.primal_residual == 0.25  # x1 >= 0 and row 2 (x1 + x3 >= 1) both by 0.25


def test_flipping_a_dual_sign_raises_dual_residual() -> None:
    k = all_types_lp()
    y = k.y.copy()
    y[1] = -2.0  # row 2 is ">=" (ru = +inf): a negative multiplier is infeasible
    v = verify(k.lp, k.x, y)
    assert v is not None
    assert v.dual_residual == 2.0
    assert v.rel_dual == 2.0 / 6.0
    assert not v.passes(0.1)


def test_reduced_cost_sign_violation() -> None:
    k = all_types_lp()
    y = k.y.copy()
    # z5 = c5 - (y3 - y4) must be 0 for the free column x5; shift y3 by 1 -> z5 = -1
    y[2] += 1.0
    v = verify(k.lp, k.x, y)
    assert v is not None
    assert v.dual_residual == 1.0


def test_scaling_y_moves_the_gap() -> None:
    k = all_types_lp()
    v1 = verify(k.lp, k.x, k.y * 1.001)
    assert v1 is not None
    assert v1.rel_gap > 0
    assert v1.dual_objective != pytest.approx(-5.0, abs=1e-6)


def test_missing_or_bad_dual_never_passes() -> None:
    k = all_types_lp()
    v = verify(k.lp, k.x, None)
    assert v is not None and not v.has_dual and not v.passes(1.0)
    assert v.primal_feasible(1e-12)
    y = k.y.copy()
    y[0] = np.nan
    v2 = verify(k.lp, k.x, y)
    assert v2 is not None and not v2.passes(1.0)


def test_no_primal_returns_none_and_nan_primal_fails() -> None:
    k = all_types_lp()
    assert verify(k.lp, None, k.y) is None
    x = k.x.copy()
    x[2] = np.inf
    v = verify(k.lp, x, k.y)
    assert v is not None and math.isinf(v.primal_residual) and not v.passes(1.0)


def test_wrong_shape_raises() -> None:
    k = all_types_lp()
    with pytest.raises(ValueError):
        verify(k.lp, k.x[:3], k.y)


@pytest.mark.parametrize("seed", range(20))
def test_random_known_optima_pass(seed: int) -> None:
    k = random_known_lp(m=8 + seed % 5, n=10 + seed % 7, seed=seed)
    v = verify(k.lp, k.x, k.y)
    assert v is not None
    assert v.primal_residual == 0.0
    assert v.dual_residual == 0.0
    assert v.primal_objective == k.objective
    assert v.dual_objective == pytest.approx(k.objective, abs=1e-9)
    assert v.passes(1e-12)
    assert not v.floor_limited(1e-8)


def test_floor_flags_badly_scaled_evaluation() -> None:
    # a row whose products are huge while the bounds are small: |A||x| ~ 1e20
    lp = make_lp("floor", [[1e10, -1e10]], c=[0, 0], rl=[0], ru=[0], l=[0, 0], u=[1e10, 1e10])
    x = np.array([1e10, 1e10])
    v = verify(lp, x, np.zeros(1))
    assert v is not None
    # floor = 2 * 2^-53 * 2e20 / (1 + 1e10) ~ 4.4e-6
    assert v.floor_primal == pytest.approx(2 * 2.0**-53 * 2e20 / (1 + 1e10), rel=1e-12)
    assert v.floor_limited(1e-6)
    assert not v.floor_limited(1e-3)


def test_objective_agreement() -> None:
    assert objective_agrees(-464.75314285714285, -464.753142857142857, 1e-12)
    assert objective_agrees(100.0, 100.0 + 1e-6 * 101 * 0.99, 1e-6)
    assert not objective_agrees(100.0, 100.0 + 1e-6 * 101 * 1.01, 1e-6)
    assert not objective_agrees(math.nan, 0.0, 1.0)
    assert objective_rel_error(1.0, 0.0) == 1.0
    assert math.isinf(objective_rel_error(math.inf, 0.0))


def test_dual_objective_uses_correct_bound_side() -> None:
    # min x  s.t. 2 <= x <= 5 as a column bound; z = 1 > 0 uses l = 2
    lp = make_lp("bnd", np.zeros((0, 1)), c=[1], rl=[], ru=[], l=[2], u=[5])
    v = verify(lp, np.array([2.0]), np.zeros(0))
    assert v is not None and v.dual_objective == 2.0 and v.passes(1e-15)
    # max x (min -x): z = -1 < 0 uses u = 5
    lp2 = make_lp("bnd2", np.zeros((0, 1)), c=[-1], rl=[], ru=[], l=[2], u=[5])
    v2 = verify(lp2, np.array([5.0]), np.zeros(0))
    assert v2 is not None and v2.dual_objective == -5.0 and v2.passes(1e-15)


def test_infinite_bound_side_counts_as_dual_infeasibility() -> None:
    # min x with x free has no finite optimum; any x with y = [] has z = 1 on a free column
    lp = make_lp("free", np.zeros((0, 1)), c=[1], rl=[], ru=[], l=[-INF], u=[INF])
    v = verify(lp, np.array([0.0]), np.zeros(0))
    assert v is not None and v.dual_residual == 1.0


def test_big_m_bound_hides_violations_from_the_scaled_check() -> None:
    """One rhs of 1e10 makes b_scale 1e10, so x = 0, y = 0 passes the (preregistered)
    infinity-norm check although it violates a row by 15 (MIPLIB trento1); only the
    componentwise diagnostic and the reference objective reveal it."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    lp = make_lp("bigm", A, [1.0, 1.0], [15.0, -INF], [INF, 1e10], [0.0, 0.0], [INF, INF])
    v = verify(lp, np.zeros(2), np.zeros(2))
    assert v is not None
    assert v.primal_residual == 15.0 and v.passes(1e-8)  # the weakness, recorded
    assert v.rel_primal_comp == pytest.approx(15.0 / 16.0)
    assert v.rel_primal_rows == v.rel_primal  # the big value sits in a row bound here
    assert not objective_agrees(v.primal_objective, 15.0, 1e-6)


def test_row_scale_ignores_variable_bounds() -> None:
    """PDLP's relative primal criterion scales by constraint bounds only; a big-M variable
    bound inflates b_scale but not the row-only scale (MIPLIB mas74: 1e12 vs 1.6e5)."""
    A = np.array([[1.0, 1.0]])
    lp = make_lp("bigu", A, [1.0, 1.0], [2.0], [INF], [0.0, 0.0], [1e12, INF])
    x = np.array([1.0, 1.0 - 1e-3])
    v = verify(lp, x, np.array([1.0]))
    assert v is not None
    assert v.b_scale == pytest.approx(1.0 + 1e12)
    assert v.rel_primal == pytest.approx(1e-3 / (1.0 + 1e12))
    assert v.rel_primal_rows == pytest.approx(1e-3 / 3.0)
    assert v.rel_primal_comp == pytest.approx(1e-3 / 3.0)
