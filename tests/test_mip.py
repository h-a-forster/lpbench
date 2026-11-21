"""MIP claims (DESIGN 13): exact MPS reader, primal classes, exact completion, refutations
checked against brute-force optima, solver claim capture and the harness."""

from __future__ import annotations

import itertools
import math
import random
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from certkkt.rational import Q, to_q

from lpbench.mip.adapters import solve_highs, solve_scip
from lpbench.mip.check import classify, solution_vector
from lpbench.mip.claims import dual_claim, estimate, refutation, wilson, witnesses
from lpbench.mip.complete import complete, q_parse
from lpbench.mip.mps import MpsError, parse_mps, read_mps

ROOT = Path(__file__).resolve().parents[1]


def mps(
    rows: str, columns: str, rhs: str = "", ranges: str = "", bounds: str = "", head: str = ""
) -> str:
    text = f"NAME t\n{head}ROWS\n N obj\n{rows}COLUMNS\n{columns}"
    for name, body in (("RHS", rhs), ("RANGES", ranges), ("BOUNDS", bounds)):
        if body:
            text += f"{name}\n{body}"
    return text + "ENDATA\n"


INT = "    MARKER  'MARKER'  'INTORG'\n"
END = "    MARKER  'MARKER'  'INTEND'\n"

# --------------------------------------------------------------------------------------------
# 1. Reader
# --------------------------------------------------------------------------------------------


def test_reader_markers_bounds_ranges_and_exact_decimals() -> None:
    text = mps(
        rows=" E e1\n E e2\n L l1\n G g1\n N free\n",
        columns=INT
        + "    a obj 1 e1 1\n    b obj 1 e2 1\n    c obj 1 l1 1\n"
        + END
        + "    d obj 0.1 g1 1\n    e obj 1 free 3\n    f obj 1 l1 1\n    g obj 1 g1 2\n"
        + "    h obj 1 e1 1\n    i obj 1 e2 1\n    j obj 1 l1 1\n    k obj 1 g1 1\n",
        rhs="    rhs e1 4 e2 4\n    rhs l1 10 g1 1\n    rhs obj 2.5\n",
        ranges="    rng e1 3 e2 -3\n    rng l1 2\n",
        bounds=" UP bnd b 5\n UP bnd c -3\n BV bnd d\n LI bnd e 2\n UI bnd f 7\n"
        " MI bnd g\n PL bnd h\n FX bnd i 0.1\n FR bnd j\n LO bnd k -1e1\n",
    )
    m = parse_mps(text)
    names = m.col_names
    b = {n: (m.lb[j], m.ub[j]) for j, n in enumerate(names)}
    assert m.integer == [True, True, True, True, True, True, False, False, False, False, False]
    assert b["a"] == (0, 1)  # integer without bounds: binary
    assert b["b"] == (0, 5)
    assert b["c"] == (0, -3)  # negative UP keeps the lower bound (HiGHS, SCIP)
    assert b["d"] == (0, 1) and b["e"] == (2, None) and b["f"] == (0, 7)
    assert b["g"] == (None, None) and b["h"] == (0, None) and b["j"] == (None, None)
    assert b["i"] == (Q(1, 10), Q(1, 10)) and b["k"] == (-10, None)
    assert m.c[names.index("d")] == Q(1, 10)  # 0.1 is exactly 1/10
    assert m.c0 == Q(-5, 2)
    assert m.row_names == ["e1", "e2", "l1", "g1"]  # extra N rows are dropped
    assert list(zip(m.rl, m.ru, strict=True)) == [(4, 7), (1, 4), (8, 10), (1, None)]
    assert not m.maximize


@pytest.mark.parametrize(
    "head", ["OBJSENSE\n    MAX\n", "OBJSENSE MAXIMIZE\n", "OBJSENSE\n    MAXIMIZE\n"]
)
def test_reader_objsense(head: str) -> None:
    m = parse_mps(mps(" L c\n", "    x obj 1 c 1\n", "    rhs c 4\n", head=head))
    assert m.maximize and m.min_c() == [-1]


def test_reader_fixed_format_names_with_spaces() -> None:
    text = (
        "NAME          fixed\nROWS\n N  obj\n L  row one\nCOLUMNS\n"
        "    x 1       obj       1.5            row one   2\n"
        "RHS\n    rhs       row one   3\nENDATA\n"
    )
    m = parse_mps(text)
    assert m.row_names == ["row one"] and m.col_names == ["x 1"]
    assert m.rows[0] == {0: 2} and m.c == [Q(3, 2)] and m.ru == [3]


@pytest.mark.parametrize(
    "text",
    [
        mps(" L c\n", "    x obj 1 c 1\n", bounds=" SC bnd x 4\n"),
        mps(" L c\n", "    x obj 1 c 1\n").replace("ENDATA", "SOS\n S1 SOS s1 1\nENDATA"),
        mps(" L c\n", "    x obj 1 c 1\n").replace("ENDATA", "QUADOBJ\n    x x 1\nENDATA"),
    ],
)
def test_reader_rejects_sc_and_unknown_sections(text: str) -> None:
    with pytest.raises(MpsError):
        parse_mps(text)


def _pilot_names() -> list[str]:
    from lpbench.mip.study import load_mip_config

    return list(load_mip_config(ROOT / "configs" / "mip-pilot.toml").instances)


def test_mip_config_allows_at_most_three_workers() -> None:
    import tomllib

    from lpbench.config import ConfigError
    from lpbench.mip.study import parse_mip_config

    raw = tomllib.loads((ROOT / "configs" / "mip-pilot.toml").read_text(encoding="utf-8"))
    assert parse_mip_config(raw).workers == 3
    with pytest.raises(ConfigError):
        parse_mip_config({**raw, "workers": 4})


@pytest.mark.slow
@pytest.mark.parametrize("name", _pilot_names())
def test_reader_agrees_with_highs_on_pilot(name: str) -> None:
    """Every rational rounds to the float HiGHS holds; integrality and names are equal."""
    import highspy
    import numpy as np
    import scipy.sparse as sp

    from lpbench.instances import miplib_instances
    from lpbench.mip.study import ensure_mip_mps

    inst = {i.name: i for i in miplib_instances(10**12)}[name]
    if not inst.raw_path().exists():
        pytest.fail(f"{name} is not fetched: run lpbench fetch --config quick --suite miplib")
    path = ensure_mip_mps(inst)
    m = read_mps(path)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(path))
    lp = h.getLp()
    fa = m.float_arrays()

    def inf(a: Any) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        return np.where(a >= 1e20, np.inf, np.where(a <= -1e20, -np.inf, a))

    mat = lp.a_matrix_
    A = sp.csc_matrix((mat.value_, mat.index_, mat.start_), shape=(lp.num_row_, lp.num_col_))
    assert list(lp.col_names_) == m.col_names and list(lp.row_names_) == m.row_names
    assert fa["integer"].tolist() == [int(v) == 1 for v in lp.integrality_]
    assert np.array_equal(fa["c"], np.asarray(lp.col_cost_)) and fa["c0"] == lp.offset_
    for ours, theirs in (
        ("lb", lp.col_lower_),
        ("ub", lp.col_upper_),
        ("rl", lp.row_lower_),
        ("ru", lp.row_upper_),
    ):
        assert np.array_equal(fa[ours], inf(theirs)), ours
    assert (fa["A"].tocsc() != A).nnz == 0
    assert m.maximize == (lp.sense_ == highspy.ObjSense.kMaximize)


# --------------------------------------------------------------------------------------------
# 2. Checker
# --------------------------------------------------------------------------------------------

# x integer in [0, 2e6]; y continuous and free; r1: x + y <= 1; r2: z <= 1e6 for a second
# continuous z >= 0, so its row has |rhs| = 1e6.
CHECK_MPS = mps(
    rows=" L r1\n L r2\n",
    columns=INT + "    x obj 1 r1 1\n" + END + "    y obj 1 r1 1\n    z obj 1 r2 1\n",
    rhs="    rhs r1 1 r2 1e6\n",
    bounds=" UP bnd x 2e6\n FR bnd y\n",
)
HIGHS_TOL = {"mip_feasibility_tolerance": 1e-6}


def _class(x: float, y: float, z: float, claim: float | None = None) -> dict[str, Any]:
    m = parse_mps(CHECK_MPS)
    return classify(m, {"x": x, "y": y, "z": z}, "highs", HIGHS_TOL, claim)


@pytest.mark.parametrize(
    ("x", "y", "z", "cls"),
    [
        (0.0, 1.0, 0.0, "P0"),
        (0.0, 1.0 + 1e-12, 0.0, "P1"),
        (0.0, 1.0 + 1e-5, 0.0, "P2"),
        (0.0, 1.0 + 1e-3, 0.0, "P3"),
        (0.9999999, 0.0, 0.0, "P1"),  # integrality 1e-7 within HiGHS's 1e-6
        (1e6 + 5e-4, -1e6, 0.0, "P3"),  # integrality is absolute: 5e-4 > 1e-4 even at 1e6
        (0.0, 0.0, 1e6 + 5e-4, "P2"),  # rows are scaled: 5e-4 / 1e6 is within 1e-4
    ],
)
def test_primal_classes(x: float, y: float, z: float, cls: str) -> None:
    assert _class(x, y, z)["primal_class"] == cls


def test_scip_rows_are_relative() -> None:
    m = parse_mps(CHECK_MPS)
    out = classify(
        m, {"x": 0.0, "y": 0.0, "z": 1e6 + 5e-4}, "scip", {"numerics/feastol": 1e-6}, None
    )
    assert out["primal_class"] == "P1"  # 5e-4 / 1e6 within SCIP's relative 1e-6


def test_objective_misstatement() -> None:
    assert _class(1.0, 0.0, 0.5, claim=1.5)["objective_ok"] is True
    bad = _class(1.0, 0.0, 0.5, claim=1.6)
    assert bad["objective_ok"] is False and bad["objective_rel_diff"] > 0.03


def test_solution_must_cover_the_columns() -> None:
    with pytest.raises(ValueError):
        solution_vector(parse_mps(CHECK_MPS), {"x": 0.0, "y": 0.0})


# --------------------------------------------------------------------------------------------
# 3. Completion
# --------------------------------------------------------------------------------------------

# min x + y  s.t.  x + y >= 1.5,  x integer in [0, 3],  y in [0, 0.7]
MIXED = mps(
    rows=" G r\n",
    columns=INT + "    x obj 1 r 1\n" + END + "    y obj 1 r 1\n",
    rhs="    rhs r 1.5\n",
    bounds=" UP bnd x 3\n UP bnd y 0.7\n",
)


def test_completion_feasible_is_exact() -> None:
    m = parse_mps(MIXED)
    c = complete(m, [to_q(1.0000001), to_q(0.3)])
    assert c.outcome == "feasible" and c.z == Q(3, 2)  # x = 1 fixed, y = 1/2 exactly


def test_completion_certifies_the_decimal_data() -> None:
    # 1.1 is not a double: HiGHS solves the rounded data, the certificate is for 11/10
    m = parse_mps(MIXED.replace("rhs r 1.5", "rhs r 1.1"))
    c = complete(m, [to_q(1.0), to_q(0.1)])
    assert c.outcome == "feasible" and c.z == Q(11, 10)


def test_completion_infeasible_is_certified() -> None:
    m = parse_mps(MIXED)
    c = complete(m, [to_q(0.4), to_q(0.7)])  # x rounds to 0: y >= 1.5 > 0.7
    assert c.outcome == "infeasible" and c.level.startswith("farkas")


def test_completion_pure_integer_and_none() -> None:
    m = parse_mps(
        mps(" L r\n", INT + "    x obj -1 r 2\n" + END, "    rhs r 3\n", bounds=" UP bnd x 5\n")
    )
    assert complete(m, [to_q(1.2)]).z == -1
    assert complete(m, [to_q(1.6)]).outcome == "infeasible"  # 2 * 2 > 3
    assert complete(m, None).outcome == "none"


# --------------------------------------------------------------------------------------------
# 4. Refutation oracle by enumeration
# --------------------------------------------------------------------------------------------


def _random_problem(seed: int, mixed: bool) -> tuple[str, Fraction, list[float]]:
    """A random binary (or binary plus one continuous) problem, its exact optimum by
    enumeration and an optimal solution. Continuous coefficients are powers of two, so the
    optimum is a dyadic rational (exact as a double)."""
    rng = random.Random(seed)
    nb = rng.randint(4, 10)
    nrows = rng.randint(2, 4)
    cost = [rng.randint(-9, 9) for _ in range(nb)]
    A = [[rng.randint(-5, 5) for _ in range(nb)] for _ in range(nrows)]
    b = [rng.choice([1, 2, 4, -1, -2, -4]) for _ in range(nrows)] if mixed else [0] * nrows
    rhs = [rng.randint(0, 8) for _ in range(nrows)]
    cy, uy = rng.choice([-3, 3]), 6
    best: tuple[Fraction, list[float]] | None = None
    for xs in itertools.product((0, 1), repeat=nb):
        lo, hi = Fraction(0), Fraction(uy if mixed else 0)
        for i in range(nrows):  # sum a x + b y <= rhs
            slack = Fraction(rhs[i] - sum(a * x for a, x in zip(A[i], xs, strict=True)))
            if b[i] > 0:
                hi = min(hi, slack / b[i])
            elif b[i] < 0:
                lo = max(lo, slack / b[i])
            elif slack < 0:
                lo, hi = Fraction(1), Fraction(0)
        if lo > hi:
            continue
        y = lo if cy > 0 or not mixed else hi
        z = sum(c * x for c, x in zip(cost, xs, strict=True)) + (cy * y if mixed else 0)
        if best is None or z < best[0]:
            best = (Fraction(z), [float(x) for x in xs] + ([float(y)] if mixed else []))
    assert best is not None  # x = 0 (and y = 0 for the mixed case) is always feasible
    rows = "".join(f" L r{i}\n" for i in range(nrows))
    cols = INT
    for j in range(nb):
        cols += f"    x{j} obj {cost[j]}\n"
        cols += "".join(f"    x{j} r{i} {A[i][j]}\n" for i in range(nrows) if A[i][j])
    cols += END
    if mixed:
        cols += f"    y obj {cy}\n" + "".join(f"    y r{i} {b[i]}\n" for i in range(nrows))
    text = mps(
        rows,
        cols,
        "".join(f"    rhs r{i} {rhs[i]}\n" for i in range(nrows)),
        bounds="".join(f" UP bnd x{j} 1\n" for j in range(nb))
        + (f" UP bnd y {uy}\n" if mixed else ""),
    )
    return text, best[0], best[1]


SEEDS = [(s, s % 2 == 1) for s in range(12)]


@pytest.mark.parametrize(("seed", "mixed"), SEEDS)
def test_refutation_oracle(seed: int, mixed: bool) -> None:
    text, zstar, xopt = _random_problem(seed, mixed)
    m = parse_mps(text)
    comp = complete(m, [to_q(v) for v in xopt]).to_dict()
    assert comp["outcome"] == "feasible" and q_parse(comp["z"]) == Q(zstar)
    w = witnesses([{"instance": "p", "completion": comp}])["p"]
    exact = refutation(dual_claim({"dual_bound": float(zstar)}, False), w)
    above = refutation(dual_claim({"dual_bound": float(zstar + Fraction(1, 10**9))}, False), w)
    assert exact is not None and not any(exact["bins"].values())
    assert above is not None and above["bins"][">0"]


@pytest.mark.parametrize(("seed", "mixed"), SEEDS[:6])
def test_real_solver_claims_match_enumeration(seed: int, mixed: bool, tmp_path: Path) -> None:
    text, zstar, _ = _random_problem(seed, mixed)
    path = tmp_path / "p.mps"
    path.write_text(text, encoding="utf-8")
    m = parse_mps(text)
    for solve, solver, tol_key in (
        (solve_highs, "highs", "mip_feasibility_tolerance"),
        (solve_scip, "scip", "numerics/feastol"),
    ):
        claim = solve(path, 30.0, 0)
        assert claim.status == "OPTIMAL", solver
        assert abs(claim.primal_bound - float(zstar)) <= 1e-6 * max(1, abs(zstar))
        assert claim.dual_bound <= float(zstar) + 1e-6 * max(1, abs(zstar))
        assert tol_key in claim.tolerances
        chk = classify(m, claim.x, solver, claim.tolerances, claim.primal_bound)
        assert chk["primal_class"] in ("P0", "P1") and chk["objective_ok"]
        comp = complete(m, solution_vector(m, claim.x))
        assert comp.outcome == "feasible" and comp.z == Q(zstar)


def test_infeasibility_claim_is_refuted_by_any_witness() -> None:
    r = refutation(dual_claim({"status": "INFEASIBLE"}, False), Q(5))
    assert r is not None and all(r["bins"].values())


def test_maximisation_is_negated_on_read() -> None:
    assert dual_claim({"dual_bound": 7.0}, True) == -7


def test_wilson() -> None:
    lo, hi = wilson(0, 20)
    assert lo == 0 and hi == pytest.approx(3.8415 / (20 + 3.8415), rel=1e-4)
    lo, hi = wilson(5, 20)
    assert lo == pytest.approx(0.1119, abs=1e-4) and hi == pytest.approx(0.4687, abs=1e-4)
    assert wilson(0, 0) is None


def test_unresolved_units_are_counted() -> None:
    claims = [
        {"has_solution": True, "primal_class": "P0", "completion": "feasible"},
        {"has_solution": True, "primal_class": "P2", "completion": "unknown"},
        {"has_solution": False, "primal_class": None, "completion": "none"},
    ]
    e3 = estimate(claims, "E3")
    assert (e3["events"], e3["resolved"], e3["unresolved"]) == (0, 1, 1)
    e2 = estimate(claims, "E2")
    assert (e2["events"], e2["resolved"]) == (1, 2)


# --------------------------------------------------------------------------------------------
# 5. Adapters
# --------------------------------------------------------------------------------------------


def _market_split(tmp_path: Path) -> Path:
    """A market split problem: tiny, and too hard to solve in a fraction of a second."""
    rng = random.Random(3)
    nrows, n = 6, 60
    A = [[rng.randint(0, 99) for _ in range(n)] for _ in range(nrows)]
    rows = "".join(f" E r{i}\n" for i in range(nrows))
    cols = INT + "".join(f"    x{j} r{i} {A[i][j]}\n" for j in range(n) for i in range(nrows)) + END
    cols += "".join(f"    sp{i} obj 1 r{i} 1\n    sm{i} obj 1 r{i} -1\n" for i in range(nrows))
    rhs = "".join(f"    rhs r{i} {sum(A[i]) // 2}\n" for i in range(nrows))
    bounds = "".join(f" UP bnd x{j} 1\n" for j in range(n))
    path = tmp_path / "ms.mps"
    path.write_text(mps(rows, cols, rhs, bounds=bounds), encoding="utf-8")
    return path


@pytest.mark.parametrize("solve", [solve_highs, solve_scip])
def test_adapter_returns_every_claim_field(solve: Any, tmp_path: Path) -> None:
    path = tmp_path / "m.mps"
    path.write_text(MIXED, encoding="utf-8")
    c = solve(path, 10.0, 0)
    assert c.status == "OPTIMAL" and c.missing == []
    assert c.x is not None and set(c.x) == {"x", "y"}
    assert c.primal_bound == pytest.approx(1.5) and c.dual_bound == pytest.approx(1.5)
    assert c.gap is not None and c.gap <= 1e-4
    assert c.tolerances and all(isinstance(v, float | int) for v in c.tolerances.values())


@pytest.mark.parametrize("solve", [solve_highs, solve_scip])
def test_adapter_time_limit_gives_a_finite_dual_bound(solve: Any, tmp_path: Path) -> None:
    c = solve(_market_split(tmp_path), 0.5, 0)
    assert c.status == "TIME_LIMIT"
    assert c.dual_bound is not None and math.isfinite(c.dual_bound)
    assert c.x is None or c.primal_bound >= c.dual_bound
    assert c.missing == []  # a gap reported as infinite (dual bound 0) is not missing


# --------------------------------------------------------------------------------------------
# 6. Harness
# --------------------------------------------------------------------------------------------


def _harness(cache: Path, solvers: tuple[str, ...], tl: float = 5.0) -> tuple[Any, list[Any]]:
    from lpbench.instances import Instance
    from lpbench.mip.study import MipConfig, MipUnit, mip_mps_path

    inst = Instance("miplib", "mixed", "mixed.mps.gz", "https://miplib.zib.de/x", 1, 2, 2)
    p = mip_mps_path(inst, cache)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(MIXED, encoding="utf-8")
    raw = {"name": "t", "solvers": list(solvers), "time_limit_s": tl}
    cfg = MipConfig("t", 2, tl, 0, solvers, ("mixed",), 1.0, 2.0, raw=raw)
    return cfg, [MipUnit(inst, s, 0) for s in solvers]


def test_hung_solver_is_killed_and_resume_skips_finished(
    cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lpbench.mip.study import MipRunner
    from lpbench.runner import load_records

    monkeypatch.setenv("LPBENCH_TEST_SLEEP_S", "120")
    cfg, units = _harness(cache, ("fake-hang", "highs", "scip"), tl=0.5)
    out = tmp_path / "run"
    t0 = time.perf_counter()
    MipRunner(cfg, out, units, workers=2).run()  # type: ignore[arg-type]
    assert time.perf_counter() - t0 < 30
    recs = load_records(out)
    assert recs[units[0].unit_id]["status"] == "KILLED"
    assert recs[units[1].unit_id]["status"] == "OPTIMAL"
    assert recs[units[2].unit_id]["x"] == {"x": pytest.approx(1.0), "y": pytest.approx(0.5)}
    before = {p.name: p.read_text(encoding="utf-8") for p in (out / "units").glob("*.json")}
    s = MipRunner(cfg, out, units, workers=2).run()  # type: ignore[arg-type]
    assert (s.skipped, s.completed) == (3, 0)
    after = {p.name: p.read_text(encoding="utf-8") for p in (out / "units").glob("*.json")}
    assert after == before
