"""Statistics against hand-computed values (DESIGN 6.5) and a bootstrap coverage check."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lpbench import stats as S


def test_sgm_hand_values() -> None:
    assert S.sgm(np.array([1.0, 2.0, 4.0]), shift=0.0) == pytest.approx(2.0)
    assert S.sgm(np.array([1.0, 2.0, 4.0]), shift=1.0) == pytest.approx(30 ** (1 / 3) - 1)
    # t_min = 0.001 applies before the shift
    assert S.sgm(np.array([0.0, 0.0]), shift=0.0) == pytest.approx(0.001)
    assert S.sgm(np.array([0.0, 0.0]), shift=1.0) == pytest.approx(0.001)
    T = np.array([[1.0, 4.0], [4.0, 1.0]])
    np.testing.assert_allclose(S.sgm(T, 0.0, axis=0), [2.0, 2.0])


def test_collapse_replicates() -> None:
    T3 = np.array([[[1.0, 3.0]]])  # one instance, one config, two replicates
    assert S.collapse_replicates(T3, 1.0)[0, 0] == pytest.approx(math.sqrt(2 * 4) - 1)


def test_penalize() -> None:
    t = np.array([1.0, 5.0, 12.0])
    ok = np.array([True, False, True])
    np.testing.assert_array_equal(S.penalize(t, ok, 10.0), [1.0, 10.0, 10.0])
    np.testing.assert_array_equal(S.penalize(t, ok, 10.0, par=2), [1.0, 20.0, 10.0])


def test_holm_textbook() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    np.testing.assert_allclose(S.holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(S.holm([0.5, 0.6]), [1.0, 1.0])
    assert S.holm([]).size == 0


def test_profile_step_function() -> None:
    T = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 5.0]])
    solved = np.array([[True, True], [True, True], [True, False]])
    taus = np.array([1.0, 1.5, 2.0, 100.0])
    rho = S.profile(T, solved, taus)
    np.testing.assert_allclose(rho[0], [2 / 3, 2 / 3, 1.0, 1.0])
    np.testing.assert_allclose(rho[1], [1 / 3, 1 / 3, 2 / 3, 2 / 3])
    r = S.perf_ratios(T, solved)
    assert math.isinf(r[2, 1]) and r[0, 1] == 2.0


def test_profile_all_failed_instance_counts_for_nobody() -> None:
    T = np.array([[1.0, 1.0], [3.0, 3.0]])
    solved = np.array([[True, True], [False, False]])
    rho = S.profile(T, solved, np.array([1.0, 1e4]))
    np.testing.assert_allclose(rho, [[0.5, 0.5], [0.5, 0.5]])


def test_profile_bands_contain_point_estimate() -> None:
    rng = np.random.default_rng(0)
    T = rng.lognormal(0, 1, size=(40, 3))
    solved = rng.random((40, 3)) > 0.1
    b = S.profile_bands(T, solved, S.default_taus(21), 400, rng)
    assert np.all(b.lo <= b.rho + 1e-12) and np.all(b.rho <= b.hi + 1e-12)
    assert np.all(b.sim_halfwidth >= np.max(np.maximum(b.rho - b.lo, b.hi - b.rho), axis=1) - 0.05)


def test_ratio_interval_basic() -> None:
    rng = np.random.default_rng(1)
    base = rng.lognormal(0, 1, 60)
    T = np.column_stack([2 * base, base])
    iv = S.ratio_interval(T, 0, 1, 0.0, 2000, rng)
    assert iv.point == pytest.approx(2.0)
    assert iv.lo == pytest.approx(2.0) and iv.hi == pytest.approx(2.0)  # no paired noise
    noisy = np.column_stack([base * rng.lognormal(0, 0.3, 60), base])
    iv2 = S.ratio_interval(noisy, 0, 1, 1.0, 2000, rng)
    assert iv2.lo < iv2.point < iv2.hi
    iv3 = S.ratio_interval(noisy, 0, 1, 1.0, 2000, rng, method="bca")
    assert iv3.lo < iv3.point < iv3.hi


def test_two_level_bootstrap_reduces_to_one_level_without_replicate_noise() -> None:
    rng = np.random.default_rng(2)
    base = rng.lognormal(0, 1, 50)
    T3 = np.stack([np.column_stack([1.5 * base, base])] * 3, axis=-1)
    iv = S.two_level_ratio_interval(T3, 0, 1, 0.0, 1000, rng)
    assert iv.point == pytest.approx(1.5) and iv.lo == pytest.approx(1.5)


def test_wilcoxon_and_ranks() -> None:
    rng = np.random.default_rng(3)
    base = rng.lognormal(0, 1, 40)
    med, p = S.wilcoxon_pair(base * 3, base, 0.0)
    assert med == pytest.approx(math.log(3)) and p < 1e-6
    assert S.wilcoxon_pair(base, base, 1.0) == (0.0, 1.0)
    T = np.column_stack([base * 3, base, base * 2])
    np.testing.assert_array_equal(S.ranks_by_sgm(T, 1.0), [3, 1, 2])
    rs = S.rank_stability(T, 1.0, 200, rng)
    assert rs.p_rank[1, 0] == 1.0 and rs.p_faster[1, 0] == 1.0
    np.testing.assert_allclose(rs.p_rank.sum(axis=1), 1.0)
    assert S.kendall_tau(np.array([1, 2, 3]), np.array([1, 2, 3])) == pytest.approx(1.0)
    assert S.kendall_tau(np.array([1, 2, 3]), np.array([3, 2, 1])) == pytest.approx(-1.0)


def test_variability_measures() -> None:
    assert S.geometric_sd(np.array([2.0, 2.0, 2.0])) == pytest.approx(1.0)
    assert S.geometric_sd(np.array([1.0, math.e])) == pytest.approx(math.exp(math.sqrt(0.5)))
    assert S.range_ratio(np.array([1.0, 4.0, 2.0])) == 4.0
    assert math.isnan(S.geometric_sd(np.array([1.0])))


def test_clopper_pearson() -> None:
    lo, hi = S.clopper_pearson(950, 1000, 0.999)
    assert lo < 0.95 < hi
    assert S.clopper_pearson(0, 10)[0] == 0.0 and S.clopper_pearson(10, 10)[1] == 1.0


@pytest.mark.slow
def test_bootstrap_coverage() -> None:
    """95% percentile intervals of a known SGM ratio cover it at a rate consistent with 0.95."""
    rng = np.random.default_rng(20261024)
    n, reps, B = 100, 1000, 1000
    true_ratio = math.exp(0.3)  # shift 0: SGM ratio = exp(mu_a - mu_b)
    covered = 0
    for _ in range(reps):
        inst = rng.normal(0, 1.5, n)
        ta = np.exp(inst + 0.3 + rng.normal(0, 0.5, n))
        tb = np.exp(inst + rng.normal(0, 0.5, n))
        iv = S.ratio_interval(np.column_stack([ta, tb]), 0, 1, 0.0, B, rng)
        covered += iv.lo <= true_ratio <= iv.hi
    lo, hi = S.clopper_pearson(covered, reps, 0.999)
    assert lo <= 0.95 <= hi, covered


def test_top_level_bootstrap_is_narrower_with_replicate_noise() -> None:
    """Resampling replicates as well adds within-instance variation a second time."""
    rng = np.random.default_rng(4)
    base = rng.lognormal(0, 1, 40)
    noise = rng.lognormal(0, 0.5, (40, 2, 4))
    T3 = np.stack([1.5 * base, base], axis=1)[:, :, None] * noise
    both = S.two_level_ratio_interval(T3, 0, 1, 0.0, 2000, np.random.default_rng(5))
    top = S.two_level_ratio_interval(
        T3, 0, 1, 0.0, 2000, np.random.default_rng(5), resample_replicates=False
    )
    assert both.point == top.point
    assert (top.hi / top.lo) < (both.hi / both.lo)
