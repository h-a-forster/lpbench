"""CanonicalLP, HiGHS extraction, permutations and the MPS writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from lpbench.lpmodel import (
    INF,
    CanonicalLP,
    load_npz,
    make_lp,
    make_permutation,
    permute,
    read_with_highs,
    save_npz,
    unpermute_dual,
    unpermute_primal,
    write_mps,
)
from lpbench.synthetic import all_types_lp, random_known_lp, with_free_row
from lpbench.verify import verify


def assert_same_lp(a: CanonicalLP, b: CanonicalLP, rtol: float = 0.0) -> None:
    assert a.A.shape == b.A.shape
    assert a.nnz == b.nnz
    diff = abs(a.A - b.A)
    scale = abs(a.A).max() if a.nnz else 1.0
    assert (diff.max() if diff.nnz else 0.0) <= rtol * scale
    for name in ("c", "rl", "ru", "l", "u"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=rtol, atol=0)


def test_make_lp_validates_shapes() -> None:
    with pytest.raises(ValueError):
        make_lp("bad", [[1, 2]], c=[1], rl=[0], ru=[1], l=[0, 0], u=[1, 1])
    with pytest.raises(ValueError):
        make_lp("bad", [[1]], c=[1], rl=[INF], ru=[INF], l=[0], u=[1])


def test_permutation_identity_and_reproducible() -> None:
    p0 = make_permutation(5, 7, 0, seed=1, name="x")
    assert p0.is_identity
    p1 = make_permutation(5, 7, 1, seed=1, name="x")
    p1b = make_permutation(5, 7, 1, seed=1, name="x")
    p2 = make_permutation(5, 7, 2, seed=1, name="x")
    q1 = make_permutation(5, 7, 1, seed=1, name="y")
    assert np.array_equal(p1.rows, p1b.rows) and np.array_equal(p1.cols, p1b.cols)
    assert not np.array_equal(p1.cols, p2.cols)
    assert not np.array_equal(p1.cols, q1.cols)
    assert sorted(p1.rows) == list(range(5)) and sorted(p1.cols) == list(range(7))
    with pytest.raises(ValueError):
        make_permutation(5, 7, -1, seed=1)


@pytest.mark.parametrize("seed", range(5))
def test_permute_unpermute_round_trip_exact(seed: int) -> None:
    k = random_known_lp(9, 12, seed=seed)
    perm = make_permutation(k.lp.m, k.lp.n, 1 + seed, seed=7, name=k.lp.name)
    plp = permute(k.lp, perm)
    # the optimum of the permuted problem is the permuted optimum
    xp = k.x[perm.cols]
    yp = k.y[perm.rows]
    vp = verify(plp, xp, yp)
    assert vp is not None and vp.passes(1e-12)
    assert np.array_equal(unpermute_primal(xp, perm), k.x)
    assert np.array_equal(unpermute_dual(yp, perm), k.y)
    # A x is permuted consistently
    np.testing.assert_array_equal(plp.A @ xp, (k.lp.A @ k.x)[perm.rows])
    assert plp.objective(xp) == k.lp.objective(k.x)


def test_npz_round_trip(tmp_path: Path) -> None:
    k = all_types_lp(maximize=True)
    save_npz(k.lp, tmp_path / "a.npz")
    back = load_npz(tmp_path / "a.npz")
    assert_same_lp(k.lp, back)
    assert back.c0 == k.lp.c0 and back.maximize_flipped and back.name == k.lp.name


@pytest.mark.parametrize("maximize", [False, True])
def test_mps_round_trip_through_highs(tmp_path: Path, maximize: bool) -> None:
    k = all_types_lp(maximize)
    path = tmp_path / "all.mps"
    write_mps(k.lp, path)
    back = read_with_highs(path)
    # c0 is not written; the written problem is always the canonical minimisation
    assert back.c0 == 0.0 and not back.maximize_flipped
    assert_same_lp(k.lp, back)


@pytest.mark.parametrize("seed", range(8))
def test_random_mps_round_trip_exact(tmp_path: Path, seed: int) -> None:
    k = random_known_lp(15, 20, seed=100 + seed)
    # non-integer data to exercise exact float printing
    rng = np.random.default_rng(seed)
    A = k.lp.A.copy()
    A.data = A.data * rng.uniform(0.1, 3.0, A.nnz) / 3.0
    lp = CanonicalLP(
        name="r",
        A=A,
        c=k.lp.c / 7.0,
        c0=0.0,
        rl=k.lp.rl / 3.0,
        ru=k.lp.ru / 3.0,
        l=k.lp.l,
        u=k.lp.u,
    )
    path = tmp_path / "r.mps"
    write_mps(lp, path)
    back = read_with_highs(path)
    # ranged rows are written as rl + (ru - rl), which may differ from ru by an ulp
    assert_same_lp(lp, back, rtol=1e-15)
    np.testing.assert_array_equal(back.c, lp.c)
    np.testing.assert_array_equal(back.rl, lp.rl)


def test_mps_rejects_free_rows(tmp_path: Path) -> None:
    k = with_free_row(all_types_lp())
    with pytest.raises(ValueError, match="free rows"):
        write_mps(k.lp, tmp_path / "f.mps")


def test_highs_extraction_drops_free_rows(tmp_path: Path) -> None:
    import highspy

    k = with_free_row(all_types_lp())
    lp = k.lp
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    A = lp.A.tocsc()
    model = highspy.HighsLp()
    model.num_col_ = lp.n
    model.num_row_ = lp.m
    model.col_cost_ = lp.c
    model.col_lower_ = np.where(np.isinf(lp.l), -highspy.kHighsInf, lp.l)
    model.col_upper_ = np.where(np.isinf(lp.u), highspy.kHighsInf, lp.u)
    model.row_lower_ = np.where(np.isinf(lp.rl), -highspy.kHighsInf, lp.rl)
    model.row_upper_ = np.where(np.isinf(lp.ru), highspy.kHighsInf, lp.ru)
    model.offset_ = 7.0
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = A.indptr
    model.a_matrix_.index_ = A.indices
    model.a_matrix_.value_ = A.data
    h.passModel(model)
    from lpbench.lpmodel import from_highs_lp

    back = from_highs_lp(h.getLp(), "x")
    assert back.m == lp.m - 1
    assert back.info["dropped_free_rows"] == 1
    assert back.c0 == 7.0


def test_empty_column_and_negative_upper_bound(tmp_path: Path) -> None:
    lp = make_lp(
        "edge",
        sp.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])),
        c=[1, 0, -1],
        rl=[-INF, 0],
        ru=[4, 0],
        l=[0, -3, -INF],
        u=[0.5, -1, -2],
    )
    path = tmp_path / "e.mps"
    write_mps(lp, path)
    back = read_with_highs(path)
    assert_same_lp(lp, back)
