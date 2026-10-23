"""Adapter tests (DESIGN 6.4 items 1-4) and the parser cross-check (DESIGN 6.2)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lpbench.config import CONFIG_IDS
from lpbench.lpmodel import INF, CanonicalLP, load_npz, make_lp, write_mps
from lpbench.solvers.base import SolveRequest, SolverStatus
from lpbench.solvers.registry import FIRST_ORDER, make_adapter, option_table
from lpbench.synthetic import KnownLP, all_types_lp, infeasible_lp, random_known_lp, unbounded_lp
from lpbench.verify import objective_agrees, verify

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _solve(config: str, lp: CanonicalLP, tmp_path: Path, eps: float = 1e-6, tl: float = 30.0):
    path = tmp_path / f"{lp.name}.mps"
    if not path.exists():
        write_mps(lp, path)
    return make_adapter(config).solve(SolveRequest(config, path, eps, tl, seed=1))


@pytest.mark.parametrize("config", CONFIG_IDS)
@pytest.mark.parametrize("maximize", [False, True])
def test_all_types_lp(config: str, maximize: bool, tmp_path: Path) -> None:
    k = all_types_lp(maximize)
    res = _solve(config, k.lp, tmp_path)
    assert res.status == SolverStatus.OPTIMAL, res.raw_status
    v = verify(k.lp, res.x, res.y)
    assert v is not None and v.passes(1e-6), v
    assert objective_agrees(v.primal_objective, k.objective, 1e-6)
    assert k.lp.source_objective(v.primal_objective) == pytest.approx(
        5.0 if maximize else -5.0, abs=1e-5
    )
    # solver objective is reported for the MPS problem, i.e. without the constant c0
    assert res.objective is not None
    assert res.objective + k.lp.c0 == pytest.approx(k.objective, abs=1e-5)
    assert res.solve_wall_s >= 0 and res.solve_cpu_s >= 0


@pytest.mark.parametrize("config", CONFIG_IDS)
@pytest.mark.parametrize("seed", range(4))
def test_random_known_lps(config: str, seed: int, tmp_path: Path) -> None:
    k: KnownLP = random_known_lp(12, 16, seed=1000 + seed)
    res = _solve(config, k.lp, tmp_path)
    assert res.status == SolverStatus.OPTIMAL, res.raw_status
    v = verify(k.lp, res.x, res.y)
    assert v is not None
    # First-order methods stop on their own (scaled, relative) criteria at eps = 1e-6, which
    # can leave our residuals and objective error a little above 1e-6 (DESIGN 4.3).
    tol = 1e-4 if config in FIRST_ORDER else 1e-6
    assert objective_agrees(v.primal_objective, k.objective, tol)
    assert v.passes(tol), (config, v)


EXPECTED_INFEASIBLE = {c: {SolverStatus.INFEASIBLE} for c in CONFIG_IDS}
EXPECTED_INFEASIBLE["ortools-glop"] = {SolverStatus.INF_OR_UNBD}
EXPECTED_UNBOUNDED = {c: {SolverStatus.UNBOUNDED} for c in CONFIG_IDS}
EXPECTED_UNBOUNDED["ortools-glop"] = {SolverStatus.INF_OR_UNBD}
EXPECTED_UNBOUNDED["ortools-pdlp"] = {SolverStatus.UNBOUNDED, SolverStatus.NUMERICAL}


@pytest.mark.parametrize("config", CONFIG_IDS)
def test_infeasible_and_unbounded_status(config: str, tmp_path: Path) -> None:
    assert _solve(config, infeasible_lp(), tmp_path, tl=5).status in EXPECTED_INFEASIBLE[config]
    assert _solve(config, unbounded_lp(), tmp_path, tl=5).status in EXPECTED_UNBOUNDED[config]


def test_invalid_options_are_config_errors(tmp_path: Path) -> None:
    from lpbench.solvers.highs import HighsAdapter
    from lpbench.solvers.ortools import OrtoolsAdapter
    from lpbench.solvers.scip import ScipAdapter

    k = all_types_lp()
    path = tmp_path / "a.mps"
    write_mps(k.lp, path)
    req = SolveRequest("x", path, 1e-6, 10.0)
    assert HighsAdapter("highs-dsimplex").solve(req, {"no_such_option": 1}).status == (
        SolverStatus.CONFIG_ERROR
    )
    assert HighsAdapter("highs-ipx").solve(req, {"solver": "hipo"}).status in (
        SolverStatus.CONFIG_ERROR,
        SolverStatus.ERROR,
    )
    for c in ("ortools-glop", "ortools-pdlp"):
        assert OrtoolsAdapter(c).solve(req, "no_such_field: 1").status == SolverStatus.CONFIG_ERROR
    assert ScipAdapter().solve(req, {"no/such/param": 1}).status == SolverStatus.CONFIG_ERROR
    assert ScipAdapter().solve(req, {"limits/time": "abc"}).status == SolverStatus.CONFIG_ERROR


def test_scip_symmetry_handling_is_off(tmp_path: Path) -> None:
    """SCIP's default symmetry handling adds SST cuts even to a pure LP; with it on, a cyclic
    LP gets n extra constraints and the dual vector no longer matches the rows."""
    n = 6
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = A[i, (i + 1) % n] = 1.0
    lp = make_lp("cyc", A, -np.ones(n), [-INF] * n, [1.0] * n, [0.0] * n, [1.0] * n)
    path = tmp_path / "cyc.mps"
    write_mps(lp, path)
    ad = make_adapter("scip-soplex")
    assert ad.options(1e-6, 10.0, 1)["misc/usesymmetry"] == 0
    res = ad.solve(SolveRequest("scip-soplex", path, 1e-6, 10.0, seed=1))
    assert res.status == SolverStatus.OPTIMAL and res.y is not None and res.y.shape == (n,)
    assert verify(lp, res.x, res.y).passes(1e-8)
    on = ad.solve(SolveRequest("scip-soplex", path, 1e-6, 10.0, seed=1), {"misc/usesymmetry": 7})
    assert on.y is not None and on.y.shape == (2 * n,)  # the behaviour the option avoids


def test_option_table_alignment() -> None:
    """DESIGN 4.3: first-order configs get eps; others tightened only below their defaults."""
    t6 = option_table(1e-6, 10.0, 1)
    t8 = option_table(1e-8, 10.0, 1)
    for c in CONFIG_IDS:
        o = t6[c]
        assert o.get("threads", o.get("lp/threads", 1)) == 1
    assert t6["highs-pdlp"]["kkt_tolerance"] == 1e-6
    assert t6["highs-hipdlp"]["kkt_tolerance"] == 1e-6
    assert "eps_optimal_relative: 1e-06" in t6["ortools-pdlp"]["parameters"]
    assert "OPTIMALITY_NORM_L_INF" in t6["ortools-pdlp"]["parameters"]
    assert "num_threads: 1" in t6["ortools-pdlp"]["parameters"]
    assert "primal_feasibility_tolerance" not in t6["highs-dsimplex"]
    assert t8["highs-dsimplex"]["primal_feasibility_tolerance"] == 1e-8
    assert t8["scip-soplex"]["numerics/feastol"] == 1e-8
    assert "numerics/feastol" not in t6["scip-soplex"]
    # at 1e-4 and 1e-6 the non-first-order configs are identical (DESIGN 8)
    t4 = option_table(1e-4, 10.0, 1)
    for c in CONFIG_IDS:
        if c not in FIRST_ORDER:
            assert t4[c] == t6[c]


# ---------------------------------------------------------------------------------------------
# Parser cross-check: OR-Tools and SCIP read our MPS files as the canonical problem
# ---------------------------------------------------------------------------------------------


def _ortools_arrays(path: Path) -> tuple[np.ndarray, ...]:
    from ortools.linear_solver.python import model_builder as mb

    m = mb.Model()
    assert m.import_from_mps_file(str(path))
    p = m.export_to_proto()
    n, rows = len(p.variable), len(p.constraint)
    c = np.array([v.objective_coefficient for v in p.variable])
    lo = np.array([v.lower_bound for v in p.variable])
    up = np.array([v.upper_bound for v in p.variable])
    rl = np.array([r.lower_bound for r in p.constraint])
    ru = np.array([r.upper_bound for r in p.constraint])
    A = np.zeros((rows, n))
    for i, r in enumerate(p.constraint):
        for j, a in zip(r.var_index, r.coefficient, strict=True):
            A[i, j] += a
    return A, c, rl, ru, lo, up


def _scip_arrays(path: Path) -> tuple[np.ndarray, ...]:
    from pyscipopt import Model

    m = Model()
    m.hideOutput()
    m.readProblem(str(path))
    vs = m.getVars()
    col = {v.name: int(v.name[1:]) for v in vs}
    n = len(vs)
    conss = m.getConss()
    A = np.zeros((len(conss), n))
    rl = np.empty(len(conss))
    ru = np.empty(len(conss))
    for c in conss:
        i = int(c.name[1:])
        for name, a in m.getValsLinear(c).items():
            A[i, col[name]] += a
        rl[i], ru[i] = m.getLhs(c), m.getRhs(c)
    c_ = np.zeros(n)
    lo = np.zeros(n)
    up = np.zeros(n)
    for v in vs:
        j = col[v.name]
        c_[j], lo[j], up[j] = v.getObj(), v.getLbOriginal(), v.getUbOriginal()
    return A, c_, rl, ru, lo, up


def _clean(a: np.ndarray) -> np.ndarray:
    a = a.astype(float).copy()
    a[a >= 1e20] = INF
    a[a <= -1e20] = -INF
    return a


def _assert_reads_same(lp: CanonicalLP, path: Path, reader) -> None:  # type: ignore[no-untyped-def]
    A, c, rl, ru, lo, up = reader(path)
    np.testing.assert_allclose(A, lp.A.toarray(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(c, lp.c, rtol=1e-12, atol=0)
    for got, want in ((rl, lp.rl), (ru, lp.ru), (lo, lp.l), (up, lp.u)):
        np.testing.assert_allclose(_clean(got), want, rtol=1e-12, atol=0)


@pytest.mark.parametrize("reader", [_ortools_arrays, _scip_arrays], ids=["ortools", "scip"])
@pytest.mark.parametrize("seed", range(3))
def test_parser_cross_check_synthetic(reader, seed: int, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    k = random_known_lp(10, 14, seed=500 + seed)
    path = tmp_path / "r.mps"
    write_mps(k.lp, path)
    _assert_reads_same(k.lp, path, reader)
    k2 = all_types_lp(True)
    write_mps(k2.lp, path)
    _assert_reads_same(k2.lp, path, reader)


NETLIB_FIVE = ["AFIRO", "BOEING1", "PILOT4", "SCAGR7", "VTP.BASE"]


def _netlib_cached(name: str) -> tuple[CanonicalLP, Path]:
    npz = CACHE / "canonical" / "netlib" / f"{name}.npz"
    mps = CACHE / "mps" / "netlib" / f"{name}__p0.mps"
    if not (npz.exists() and mps.exists()):
        pytest.skip("netlib cache not prepared")
    return load_npz(npz), mps


@pytest.mark.slow
@pytest.mark.parametrize("reader", [_ortools_arrays, _scip_arrays], ids=["ortools", "scip"])
@pytest.mark.parametrize("name", NETLIB_FIVE)
def test_parser_cross_check_netlib(reader, name: str) -> None:  # type: ignore[no-untyped-def]
    lp, mps = _netlib_cached(name)
    _assert_reads_same(lp, mps, reader)


NETLIB_TEN = [
    "AFIRO",
    "ADLITTLE",
    "BLEND",
    "BOEING2",
    "E226",
    "KB2",
    "SC105",
    "SCAGR7",
    "SHARE2B",
    "STOCFOR1",
]


@pytest.mark.slow
@pytest.mark.parametrize("config", ["highs-dsimplex", "highs-ipx", "ortools-glop", "scip-soplex"])
def test_ten_netlib_verified_by_simplex_and_ipm(config: str) -> None:
    """M3 done-check: 10 netlib instances verified at 1e-6 and matching Koch's values."""
    from lpbench.instances import canonical_reference, netlib_instances

    insts = {i.name: i for i in netlib_instances()}
    adapter = make_adapter(config)
    for name in NETLIB_TEN:
        lp, mps = _netlib_cached(name)
        res = adapter.solve(SolveRequest(config, mps, 1e-6, 60.0, seed=1))
        assert res.status == SolverStatus.OPTIMAL, (name, res.raw_status)
        v = verify(lp, res.x, res.y)
        assert v is not None and v.passes(1e-6), (name, v)
        ref = canonical_reference(insts[name], lp.c0, lp.maximize_flipped)
        assert ref is not None and objective_agrees(v.primal_objective, ref, 1e-6), name


@pytest.mark.slow
def test_highs_simplex_matches_koch_on_all_netlib() -> None:
    """M2 done-check: every included netlib instance solved by highs-dsimplex matches Koch."""
    from lpbench.instances import canonical_reference, netlib_instances

    excl = ("QAP8", "QAP12", "QAP15", "STOCFOR3", "TRUSS", "STANDGUB")
    adapter = make_adapter("highs-dsimplex")
    for inst in netlib_instances(excl):
        lp, mps = _netlib_cached(inst.name)
        res = adapter.solve(SolveRequest("highs-dsimplex", mps, 1e-6, 120.0))
        v = verify(lp, res.x, res.y)
        ref = canonical_reference(inst, lp.c0, lp.maximize_flipped)
        assert res.status == SolverStatus.OPTIMAL and v is not None, inst.name
        assert objective_agrees(v.primal_objective, ref, 1e-6), inst.name  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Time limits (one per library, slow)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        "highs-dsimplex",
        "highs-pdlp",
        "highs-hipdlp",
        "ortools-pdlp",
        "scip-soplex",
        "ortools-glop",
        "highs-ipx",
    ],
)
def test_time_limit_is_respected(config: str) -> None:
    lp, mps = _netlib_cached("DFL001")
    tl = 0.5
    t0 = time.perf_counter()
    res = make_adapter(config).solve(SolveRequest(config, mps, 1e-8, tl, seed=1))
    elapsed = time.perf_counter() - t0
    assert res.status == SolverStatus.TIME_LIMIT, res.raw_status
    assert res.solve_wall_s < tl * 1.5 + 10.0
    assert elapsed < 60.0
