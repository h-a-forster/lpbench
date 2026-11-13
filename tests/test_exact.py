"""Exact reference optima (DESIGN 6.3): HiGHS basis solved and certified exactly by certkkt."""

from __future__ import annotations

import csv
import math
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from lpbench.exact import (
    BOUNDED_REL_WIDTH,
    MAX_PIVOTS,
    ExactBasis,
    ExactResult,
    conic_dual,
    exact_optimum,
    run_exact,
    to_boxed,
    write_exact_csv,
)
from lpbench.instances import EXACT_REFERENCE, Instance, miplib_instances
from lpbench.lpmodel import INF, CanonicalLP, make_lp, save_npz
from lpbench.synthetic import all_types_lp, random_known_lp


@pytest.mark.parametrize("maximize", [False, True])
def test_every_row_and_bound_type_is_certified_optimal(maximize: bool) -> None:
    k = all_types_lp(maximize)
    r = exact_optimum(k.lp)
    assert (r.level, r.method) == ("optimal", "basis"), r
    assert r.objective == k.objective  # an integer optimum: the float is exact
    assert r.lower <= r.objective <= r.upper
    assert r.reference() == r.objective


@pytest.mark.parametrize("seed", range(4))
def test_random_known_lps_are_certified_optimal(seed: int) -> None:
    k = random_known_lp(12, 16, seed=1000 + seed)
    r = exact_optimum(k.lp)
    assert r.level == "optimal", r
    assert r.objective == pytest.approx(k.objective, rel=1e-12, abs=1e-12)


def _slack_basis(lp: CanonicalLP) -> tuple[list[int], list[int]]:
    """All rows basic; every column nonbasic at a finite bound (or free at zero)."""
    import highspy

    S = highspy.HighsBasisStatus
    cols = [
        int(S.kLower if np.isfinite(lo) else S.kUpper if np.isfinite(hi) else S.kZero)
        for lo, hi in zip(lp.l, lp.u, strict=True)
    ]
    return cols, [int(S.kBasic)] * lp.m


@pytest.mark.parametrize("seed", range(6))
def test_exact_pivots_reach_the_optimum_from_the_slack_basis(seed: int) -> None:
    """From the slack basis (primal and dual infeasible on these LPs) the cost-shifted dual
    simplex and then the primal simplex, in exact arithmetic, must reach an exactly optimal
    basis with the known optimum: the path a float basis that is optimal only within
    tolerance takes, with many more pivots."""
    from certkkt import certify

    k = random_known_lp(12, 16, seed=2000 + seed)
    boxed = to_boxed(k.lp)
    problem, origins = boxed.to_conic()
    eb = ExactBasis(boxed, *_slack_basis(k.lp))
    x, y, d = eb.optimise(time.perf_counter() + 60, MAX_PIVOTS)
    assert eb.pivots > 0
    z = conic_dual(origins, [-v for v in y], [-v for v in d[: boxed.n]])
    cert = certify(problem, x[: boxed.n], z)
    assert cert.level == "optimal"
    assert float(cert.pobj) == pytest.approx(k.objective, rel=1e-12, abs=1e-12)


def test_optimum_is_exact_for_the_floats_not_rounded() -> None:
    """min x0 + x1 s.t. 3 x0 + x1 >= 1, x0 + 3 x1 >= 1: optimum 1/2 at x = (1/4, 1/4).
    With 0.1 in place of 1 on the right-hand side the optimum is 0.05 of the *float* 0.1,
    which differs from 1/20; the exact objective rounds to the float nearest that."""
    A = np.array([[3.0, 1.0], [1.0, 3.0]])
    lp = make_lp("tenth", A, [1.0, 1.0], [0.1, 0.1], [INF, INF], [0.0, 0.0], [INF, INF])
    r = exact_optimum(lp)
    assert r.level == "optimal"
    exact = Fraction(0.1) / 2  # x0 = x1 = 0.1/4
    assert r.objective == float(exact)
    assert Fraction(r.lower) <= exact <= Fraction(r.upper)


def test_float_rounding_that_makes_the_lp_infeasible_is_proved() -> None:
    """x0 = 0.1, x1 = 0.2, x0 + x1 = 0.3 holds for the decimals but not for their doubles
    (the Netlib SCORPION/SIERRA situation): HiGHS reports Optimal within its tolerance,
    the exact pass returns a checked Farkas certificate instead of a reference."""
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = [0.1, 0.2, 0.3]
    lp = make_lp("pointthree", A, [1.0, 1.0], b, b, [-INF, -INF], [INF, INF])
    r = exact_optimum(lp)
    assert r.simplex_status == "Optimal"
    assert r.level == "infeasible" and r.reference() is None, r


def test_bounded_certificate_is_a_reference_only_when_narrow() -> None:
    def res(lo: float, hi: float) -> ExactResult:
        return ExactResult("bounded", "fixed", None, lo, hi, "Optimal", 1, 0.0)

    assert res(1.0, 1.0 + 1e-13).reference() == pytest.approx(1.0 + 5e-14)
    assert res(1.0, 1.0 + 10 * BOUNDED_REL_WIDTH * 2).reference() is None
    assert ExactResult("primal_feasible", "as_is", None, None, None, "", 0, 0.0).reference() is None


def _synthetic(root: Path, n: int) -> list[Instance]:
    out = []
    for s in range(n):
        k = random_known_lp(8, 10, seed=500 + s)
        inst = Instance("synthetic", f"r{s}", f"r{s}", "https://netlib.org/x", 8, 10, 1)
        save_npz(k.lp, inst.canonical_path(root))
        out.append(inst)
    return out


def test_run_exact_checkpoints_and_skips_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("LPBENCH_CACHE", str(root))
    insts = _synthetic(root, 2)
    missing = Instance("synthetic", "absent", "absent", "https://netlib.org/x", 1, 1, 1)
    lines: list[str] = []
    recs, absent = run_exact([*insts, missing], tmp_path / "run", workers=2, log=lines.append)
    assert absent == ["synthetic/absent"]
    assert {r["level"] for r in recs.values()} == {"optimal"} and len(recs) == 2
    stamp = {p: p.stat().st_mtime_ns for p in (tmp_path / "run" / "units").iterdir()}
    lines.clear()
    recs2, _ = run_exact(insts, tmp_path / "run", workers=2, log=lines.append)
    assert lines == ["exact: 2 done, 0 to do, 0 not prepared"]
    assert recs2 == recs
    assert stamp == {p: p.stat().st_mtime_ns for p in (tmp_path / "run" / "units").iterdir()}
    table = tmp_path / "exact.csv"
    write_exact_csv(recs, {"synthetic/r0": ("kkt", recs["synthetic/r0"]["objective"])}, table)
    with open(table, encoding="utf-8", newline="") as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    assert rows["r0"]["prior_rel_diff"] == "0.00e+00" and rows["r1"]["prior"] == ""
    assert float(rows["r1"]["reference"]) == recs["synthetic/r1"]["objective"]


# -- the committed table ---------------------------------------------------------------------


def _committed() -> list[dict[str, str]]:
    with open(EXACT_REFERENCE, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_committed_exact_values_agree_with_koch_on_netlib() -> None:
    """Koch (2004) computed the Netlib optima exactly from the decimal data; the exact
    optima of the float data lpbench solves must agree far below every tolerance used."""
    rows = [r for r in _committed() if r["suite"] == "netlib" and r["reference"]]
    assert rows
    for r in rows:
        assert r["prior"], r["name"]
        assert float(r["prior_rel_diff"]) <= 1e-9, (r["name"], r["prior_rel_diff"])


def test_miplib_instances_use_exact_references_where_certified() -> None:
    exact = {r["name"]: float(r["reference"]) for r in _committed()
             if r["suite"] == "miplib" and r["reference"]}  # fmt: skip
    insts = miplib_instances(100_000)
    assert exact
    for i in insts:
        if i.name in exact:
            assert (i.reference, i.reference_source) == (exact[i.name], "exact")
        else:
            assert i.reference_source in ("consensus", "")
    for r in _committed():
        if r["reference"] and r["level"] == "optimal":
            lo, hi = float(r["lower"]), float(r["upper"])
            assert lo <= float(r["reference"]) <= hi and math.isfinite(hi - lo)
