"""Exact reference optima from an optimal simplex basis, certified by certkkt (DESIGN 6.3).

HiGHS dual simplex (tolerances 1e-9) returns a basis. Nonbasic columns and rows are fixed at
the bound the basis names; the basic columns then solve a square system ``B x_B = r`` and
the row duals solve ``B' y = c_B``, both in exact rational arithmetic (certkkt's iterative
refinement with rational reconstruction, falling back to exact elimination). The pair is
mapped to certkkt's conic form and certified exactly. When the basis is primal and dual
feasible in exact arithmetic the certificate is ``optimal`` and its objective is the exact
optimum of the LP the solvers receive (every float of the canonical model, read as the
rational it denotes). Otherwise certkkt's repairs run on the float pair; a ``bounded``
certificate encloses the optimum in ``[dobj, pobj]``.

Nothing here trusts a solver: HiGHS only proposes the basis. certkkt's ``certify`` decides
``optimal`` and ``bounded``; ``infeasible`` comes from this module's exact Farkas check
(``ExactBasis._farkas``), which certkkt never sees.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lpbench.lpmodel import CanonicalLP

EXACT_TL_S = 120.0  # HiGHS time limit for the basis
EXACT_BUDGET_S = 600.0  # exact work per instance (solves, repairs) after the basis
SIMPLEX_TOL = 1e-9
#: A ``bounded`` certificate whose relative width is at most this serves as a reference
#: (its midpoint is then within 5e-13 relative of the optimum, far below every eps used).
BOUNDED_REL_WIDTH = 1e-12
LEVELS_USED = ("optimal", "bounded")
MAX_PIVOTS = 200  # exact pivots after the float basis; a handful is typical
EXACT_OPS = 10**12  # elimination work is bounded by the time budget, not by an op count


@dataclass(frozen=True)
class ExactResult:
    """Outcome for one instance; objective values in the canonical (minimisation) sense."""

    level: str  # certkkt level of the best certificate, or "error"/"no_basis"
    method: str  # "basis", or the certkkt repair that gave the best certificate
    objective: float | None  # nearest float to the exact optimum (level optimal)
    lower: float | None  # rigorous lower bound (dobj rounded down)
    upper: float | None  # rigorous upper bound (pobj rounded up)
    simplex_status: str
    basis_size: int
    seconds: float
    note: str = ""
    pivots: int = 0  # exact pivots from the float basis to the certified one

    def reference(self) -> float | None:
        """The value usable as a reference objective, or None."""
        if self.level == "optimal":
            return self.objective
        if self.level == "bounded" and self.lower is not None and self.upper is not None:
            mid = 0.5 * (self.lower + self.upper)
            if self.upper - self.lower <= BOUNDED_REL_WIDTH * (1.0 + abs(mid)):
                return mid
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_boxed(lp: CanonicalLP) -> Any:
    """certkkt's exact boxed form of ``lp``; every float becomes the rational it denotes."""
    from certkkt import BoxedQP, to_q
    from certkkt.sparse import QMatrix

    def bounds(arr: np.ndarray) -> list[Any]:
        return [None if not np.isfinite(v) else to_q(v) for v in arr.tolist()]

    n = lp.A.shape[1]
    return BoxedQP(
        P=QMatrix.zeros((n, n)),
        c=[to_q(v) for v in lp.c.tolist()],
        A=QMatrix.from_scipy(lp.A),
        rl=bounds(lp.rl),
        ru=bounds(lp.ru),
        lb=bounds(lp.l),
        ub=bounds(lp.u),
        c0=to_q(float(lp.c0)),
        name=lp.name,
    )


@dataclass
class _Simplex:
    status: str
    x: np.ndarray | None
    y: np.ndarray | None  # HiGHS row duals: c - A'y - d = 0
    d: np.ndarray | None
    col_basic: list[int] | None  # HiGHS basis status codes
    row_basic: list[int] | None


def _simplex(lp: CanonicalLP, time_limit_s: float) -> _Simplex:
    import highspy

    m, n = lp.A.shape
    csc = lp.A.tocsc()
    csc.sort_indices()
    inf = highspy.kHighsInf
    h = highspy.Highs()
    for key, val in (
        ("output_flag", False),
        ("threads", 1),
        ("solver", "simplex"),
        ("simplex_strategy", 1),
        ("primal_feasibility_tolerance", SIMPLEX_TOL),
        ("dual_feasibility_tolerance", SIMPLEX_TOL),
        ("time_limit", float(time_limit_s)),
    ):
        if h.setOptionValue(key, val) != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected option {key}={val!r}")
    hl = highspy.HighsLp()
    hl.num_col_, hl.num_row_ = n, m
    hl.col_cost_ = lp.c
    hl.offset_ = float(lp.c0)
    hl.col_lower_ = np.where(np.isfinite(lp.l), lp.l, -inf)
    hl.col_upper_ = np.where(np.isfinite(lp.u), lp.u, inf)
    hl.row_lower_ = np.where(np.isfinite(lp.rl), lp.rl, -inf)
    hl.row_upper_ = np.where(np.isfinite(lp.ru), lp.ru, inf)
    hl.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    hl.a_matrix_.start_ = csc.indptr
    hl.a_matrix_.index_ = csc.indices
    hl.a_matrix_.value_ = csc.data
    h.passModel(hl)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    sol, basis = h.getSolution(), h.getBasis()
    if not (sol.value_valid and sol.dual_valid):
        return _Simplex(status, None, None, None, None, None)
    out = _Simplex(
        status,
        np.asarray(sol.col_value, dtype=np.float64),
        np.asarray(sol.row_dual, dtype=np.float64),
        np.asarray(sol.col_dual, dtype=np.float64),
        None,
        None,
    )
    if basis.valid:
        out.col_basic = [int(s) for s in basis.col_status]
        out.row_basic = [int(s) for s in basis.row_status]
    return out


def _codes() -> dict[str, int]:
    import highspy

    S = highspy.HighsBasisStatus
    return {"lower": int(S.kLower), "basic": int(S.kBasic), "upper": int(S.kUpper),
            "zero": int(S.kZero)}  # fmt: skip


class Infeasible(ArithmeticError):
    """The exact data has no feasible point (proved by a checked Farkas certificate)."""


class ExactBasis:
    """Bounded-variable simplex in exact arithmetic, started from a HiGHS basis.

    Variables are the columns ``x`` (``j < n``) and the row activities ``r = A x``
    (``n + i``), so the constraint matrix is ``[A, -I]`` and every variable has bounds.
    ``basic[k]`` is the variable of basis position ``k``; a nonbasic variable sits at its
    lower bound ("lo"), its upper bound ("hi") or, if free, at zero ("zero"). Each step
    recomputes the basic solution and the duals exactly (no basis updates).

    The float basis is usually exactly optimal; otherwise it violates bounds or reduced-cost
    signs by about 1e-15 and a few exact pivots finish the job: the dual simplex restores
    primal feasibility (with the costs of dual-infeasible nonbasic variables shifted so that
    the start is dual feasible), then the primal simplex restores dual feasibility for the
    true costs. Both use the smallest-index rule, so neither cycles.
    """

    def __init__(self, boxed: Any, col_status: list[int], row_status: list[int]) -> None:
        from certkkt.rational import ZERO

        codes = _codes()
        self.m, self.n = boxed.m, boxed.n
        n = self.n
        self.lo = list(boxed.lb) + list(boxed.rl)
        self.hi = list(boxed.ub) + list(boxed.ru)
        self.cost = list(boxed.c) + [ZERO] * self.m
        self.cols: list[dict[int, Any]] = [dict() for _ in range(n)]
        for i, row in enumerate(boxed.A.row_dicts()):
            for j, v in row.items():
                self.cols[j][i] = v
        self.basic: list[int] = []
        self.at: dict[int, str] = {}
        for v, code in enumerate(list(col_status) + list(row_status)):
            if code == codes["basic"]:
                self.basic.append(v)
            elif code == codes["lower"] and self.lo[v] is not None:
                self.at[v] = "lo"
            elif code == codes["upper"] and self.hi[v] is not None:
                self.at[v] = "hi"
            elif code == codes["zero"] and self.lo[v] is None and self.hi[v] is None:
                self.at[v] = "zero"
            else:
                raise ValueError(f"variable {v}: nonbasic status {code} without that bound")
        if len(self.basic) != self.m:
            raise ValueError(f"basis has {len(self.basic)} basic variables for {self.m} rows")
        self.pivots = 0
        self.refine = True

    def col(self, v: int) -> dict[int, Any]:
        from certkkt.rational import ONE

        return self.cols[v] if v < self.n else {v - self.n: -ONE}

    def value(self, v: int) -> Any:
        from certkkt.rational import ZERO

        side = self.at[v]
        return self.lo[v] if side == "lo" else self.hi[v] if side == "hi" else ZERO

    def _solve(self, transpose: bool, rhs: list[Any], deadline: float) -> list[Any]:
        from certkkt.sparse import QMatrix, solve_any, solve_refined

        m = self.m
        rows: list[dict[int, Any]] = [dict() for _ in range(m)]
        for k, v in enumerate(self.basic):
            for i, a in self.col(v).items():
                if transpose:
                    rows[k][i] = a
                else:
                    rows[i][k] = a
        sol = None
        if self.refine:  # fast when the solution's denominators are modest
            sol = solve_refined(QMatrix.from_rows((m, m), rows), rhs, deadline=deadline)
            self.refine = sol is not None  # they are similar for every basis of an LP
        if sol is None:
            sol = solve_any(rows, rhs, m, budget=EXACT_OPS, deadline=deadline)
        if sol is None:
            raise ArithmeticError("basis matrix is singular")
        return sol

    def primal(self, deadline: float) -> list[Any]:
        """Values of all n + m variables for the current basis."""
        from certkkt.rational import ZERO

        rhs = [ZERO] * self.m
        for v in self.at:
            val = self.value(v)
            if val:
                for i, a in self.col(v).items():
                    rhs[i] -= a * val
        xb = self._solve(False, rhs, deadline)
        x = [ZERO] * (self.n + self.m)
        for v in self.at:
            x[v] = self.value(v)
        for k, v in enumerate(self.basic):
            x[v] = xb[k]
        return x

    def duals(self, cost: list[Any], deadline: float) -> tuple[list[Any], list[Any]]:
        """Row duals ``y`` (``B' y = c_B``) and reduced costs of all variables."""
        y = self._solve(True, [cost[v] for v in self.basic], deadline)
        d = []
        for v in range(self.n + self.m):
            s = cost[v]
            for i, a in self.col(v).items():
                if y[i]:
                    s -= a * y[i]
            d.append(s)
        return y, d

    def primal_infeasible(self, x: list[Any]) -> list[int]:
        return [
            v
            for v in self.basic
            if (self.lo[v] is not None and x[v] < self.lo[v])
            or (self.hi[v] is not None and x[v] > self.hi[v])
        ]

    def _fixed(self, v: int) -> bool:
        return self.lo[v] is not None and self.lo[v] == self.hi[v]

    def dual_infeasible(self, d: list[Any]) -> list[int]:
        out = []
        for v, side in self.at.items():
            if self._fixed(v):
                continue
            wrong = d[v] < 0 if side == "lo" else d[v] > 0 if side == "hi" else d[v] != 0
            if wrong:
                out.append(v)
        return sorted(out)

    def dual_step(self, x: list[Any], d: list[Any], deadline: float) -> None:
        """One dual simplex pivot (smallest-index leaving variable and entering tie-break)."""
        from certkkt.rational import ONE, ZERO

        p = min(self.primal_infeasible(x))
        k = self.basic.index(p)
        below = self.lo[p] is not None and x[p] < self.lo[p]
        e = [ZERO] * self.m
        e[k] = ONE
        rho = self._solve(True, e, deadline)
        best: tuple[Any, int] | None = None
        for v, side in self.at.items():
            if self._fixed(v):
                continue
            alpha = sum((a * rho[i] for i, a in self.col(v).items() if rho[i]), ZERO)
            if not alpha:
                continue
            # leaving below its lower bound: d' = d + t alpha; above its upper: d - t alpha
            slope = alpha if below else -alpha
            if side == "lo" and slope < 0:
                ratio = d[v] / -slope
            elif side == "hi" and slope > 0:
                ratio = -d[v] / slope
            elif side == "zero":
                ratio = ZERO
            else:
                continue
            if best is None or (ratio, v) < best:
                best = (ratio, v)
        if best is None:
            self._farkas(rho)
            raise ArithmeticError("dual ratio test found no entering variable")
        q = best[1]
        self.basic[k] = q
        del self.at[q]
        self.at[p] = "lo" if below else "hi"
        self.pivots += 1

    def _farkas(self, rho: list[Any]) -> None:
        """Raise Infeasible if ``rho`` proves the LP infeasible: every feasible point has
        ``g' v = 0`` for ``g = rho' [A, -I]``, so it is infeasible when ``g' v`` is
        strictly negative (or strictly positive) over the whole box of bounds."""
        from certkkt.rational import ZERO

        lo_sum: Any = ZERO
        hi_sum: Any = ZERO
        for v in range(self.n + self.m):
            g = sum((a * rho[i] for i, a in self.col(v).items() if rho[i]), ZERO)
            if not g:
                continue
            lo_b, hi_b = (self.lo[v], self.hi[v]) if g > 0 else (self.hi[v], self.lo[v])
            lo_sum = None if lo_sum is None or lo_b is None else lo_sum + g * lo_b
            hi_sum = None if hi_sum is None or hi_b is None else hi_sum + g * hi_b
        if (hi_sum is not None and hi_sum < 0) or (lo_sum is not None and lo_sum > 0):
            raise Infeasible("the LP is infeasible in exact arithmetic (Farkas certificate)")

    def primal_step(self, x: list[Any], d: list[Any], deadline: float) -> list[Any]:
        """One primal simplex pivot (smallest-index entering and leaving variable); returns
        the new values, updated exactly along the edge instead of solved again."""
        from certkkt.rational import ZERO

        q = self.dual_infeasible(d)[0]
        sigma = 1 if d[q] < 0 else -1  # increase x_q if its reduced cost is negative
        rhs = [ZERO] * self.m
        for i, a in self.col(q).items():
            rhs[i] = a
        w = self._solve(False, rhs, deadline)
        best: tuple[Any, int, int, str] | None = None  # (step, variable, position, bound)
        if self.lo[q] is not None and self.hi[q] is not None:
            best = (self.hi[q] - self.lo[q], q, -1, "hi" if sigma > 0 else "lo")
        for k, v in enumerate(self.basic):
            rate = -sigma * w[k]  # d x_v / d t
            if rate < 0 and self.lo[v] is not None:
                cand = ((x[v] - self.lo[v]) / -rate, v, k, "lo")
            elif rate > 0 and self.hi[v] is not None:
                cand = ((self.hi[v] - x[v]) / rate, v, k, "hi")
            else:
                continue
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is None:
            raise ArithmeticError("primal ratio test found no leaving variable (unbounded LP)")
        step, v, k, side = best
        x = list(x)
        x[q] += sigma * step
        for kk, vv in enumerate(self.basic):
            if w[kk]:
                x[vv] -= sigma * step * w[kk]
        if k < 0:
            self.at[q] = side  # bound flip
        else:
            self.basic[k] = q
            del self.at[q]
            self.at[v] = side
        if x[v] != (self.lo[v] if side == "lo" else self.hi[v]):
            raise AssertionError("the blocking variable does not land on its bound")
        self.pivots += 1
        return x

    def optimise(self, deadline: float, max_pivots: int) -> tuple[list[Any], list[Any], list[Any]]:
        """Pivot until the basis is exactly optimal; returns ``(x, y, d)`` for all n + m
        variables. Raises ArithmeticError (or certkkt's BudgetExceeded) if it cannot."""
        from certkkt.sparse import check_deadline

        x = self.primal(deadline)
        if self.primal_infeasible(x):
            _, d = self.duals(self.cost, deadline)
            shifted = list(self.cost)
            for v in self.dual_infeasible(d):
                shifted[v] -= d[v]  # its reduced cost becomes 0
            while self.primal_infeasible(x):
                if self.pivots >= max_pivots:
                    raise ArithmeticError(f"no primal feasible basis after {max_pivots} pivots")
                check_deadline(deadline)
                _, d = self.duals(shifted, deadline)
                self.dual_step(x, d, deadline)
                x = self.primal(deadline)
        while True:
            y, d = self.duals(self.cost, deadline)
            if not self.dual_infeasible(d):
                return x, y, d
            if self.pivots >= max_pivots:
                raise ArithmeticError(f"no optimal basis after {max_pivots} pivots")
            check_deadline(deadline)
            x = self.primal_step(x, d, deadline)


def conic_dual(origins: list[Any], y_row: list[Any], y_col: list[Any]) -> list[Any]:
    """Exact conic duals ``z`` from boxed multipliers (``c + A'y_row + y_col = 0``), the
    rational counterpart of ``certkkt.problem.boxed_duals_to_conic``."""
    from certkkt.rational import ZERO

    sides: dict[tuple[str, int], set[str]] = {}
    for o in origins:
        sides.setdefault((o.kind, o.index), set()).add(o.side)
    z: list[Any] = []
    for o in origins:
        v = y_row[o.index] if o.kind == "row" else y_col[o.index]
        two_sided = sides[(o.kind, o.index)] == {"lo", "up"}
        if o.side in ("eq", "up"):
            z.append(max(v, ZERO) if two_sided else v)
        else:
            z.append(max(-v, ZERO) if two_sided else -v)
    return z


def exact_optimum(
    lp: CanonicalLP, time_limit_s: float = EXACT_TL_S, budget_s: float = EXACT_BUDGET_S
) -> ExactResult:
    """Certify the optimum of ``lp`` exactly (see the module docstring)."""
    from certkkt import certify, repair
    from certkkt.problem import boxed_duals_to_conic
    from certkkt.rational import q_float_down, q_float_up
    from certkkt.sparse import BudgetExceeded

    t0 = time.perf_counter()
    boxed = to_boxed(lp)
    problem, origins = boxed.to_conic()
    sx = _simplex(lp, time_limit_s)
    deadline = time.perf_counter() + budget_s
    nbasic = sum(1 for s in sx.col_basic or [] if s == _codes()["basic"])
    certs = []
    notes = []
    pivots = 0
    if sx.col_basic is not None and sx.row_basic is not None:
        try:
            eb = ExactBasis(boxed, sx.col_basic, sx.row_basic)
            xa, y, d = eb.optimise(deadline, MAX_PIVOTS)
            pivots = eb.pivots
            z = conic_dual(origins, [-v for v in y], [-v for v in d[: boxed.n]])
            certs.append(certify(problem, xa[: boxed.n], z, method="basis"))
        except Infeasible as exc:
            return ExactResult("infeasible", "basis", None, None, None, sx.status, nbasic,
                               time.perf_counter() - t0, str(exc), eb.pivots)  # fmt: skip
        except (ValueError, ArithmeticError, BudgetExceeded) as exc:
            notes.append(f"basis: {type(exc).__name__}: {exc}")
    if (not certs or certs[0].level != "optimal") and sx.x is not None:
        assert sx.y is not None and sx.d is not None
        zf = boxed_duals_to_conic(origins, -sx.y, -sx.d)
        left = max(1.0, deadline - time.perf_counter())
        out = repair(problem, sx.x, zf, methods=("as_is", "active_set", "fixed"), time_budget=left)
        certs.extend(out.certificates)
        notes.extend(f"{k}: {v}" for k, v in out.notes.items())
    seconds = time.perf_counter() - t0
    if not certs:
        return ExactResult("no_basis", "", None, None, None, sx.status, nbasic, seconds,
                           "; ".join(notes))  # fmt: skip
    best = min(certs, key=lambda c: c.rank())
    level = best.level
    # int / int is correctly rounded, so this is the float nearest the exact optimum
    objective = None
    if level == "optimal":
        objective = int(best.pobj.numerator) / int(best.pobj.denominator)
    lower = q_float_down(best.dobj) if level in LEVELS_USED else None
    upper = q_float_up(best.pobj) if level in LEVELS_USED else None
    return ExactResult(
        level, best.method, objective, lower, upper, sx.status, nbasic, seconds,
        "; ".join(notes), pivots,
    )  # fmt: skip


# --------------------------------------------------------------------------------------------
# Checkpointed pass over a config's instances (one process per instance, hard deadline)
# --------------------------------------------------------------------------------------------

EXACT_FIELDS = ["suite", "name", "level", "method", "reference", "objective", "lower", "upper",
                "prior_source", "prior", "prior_rel_diff", "simplex_status", "basis_size",
                "note"]  # fmt: skip
HARD_SLACK_S = 120.0  # beyond the HiGHS limit and the exact budget: loading and certify


def _child(canonical: str, out: str, time_limit_s: float, budget_s: float) -> None:
    from pathlib import Path

    from lpbench.lpmodel import load_npz
    from lpbench.runner import atomic_write_json
    from lpbench.worker import pin_threads

    pin_threads()
    try:
        rec = exact_optimum(load_npz(Path(canonical)), time_limit_s, budget_s).to_dict()
    except Exception as exc:  # recorded, not raised: one bad instance must not stop the pass
        rec = {"level": "error", "note": f"{type(exc).__name__}: {exc}"[:500]}
    from importlib.metadata import version

    rec["versions"] = {d: version(d) for d in ("certkkt", "gmpy2", "highspy")}
    atomic_write_json(Path(out), rec)


def run_exact(
    instances: list[Any],
    out_dir: Any,
    *,
    workers: int = 3,
    time_limit_s: float = EXACT_TL_S,
    budget_s: float = EXACT_BUDGET_S,
    log: Any = print,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Certify every instance, one spawned process each, at most ``workers`` at a time.

    Each finished instance is written to ``out_dir/units/<suite>__<name>.json`` and skipped
    on a rerun. Returns the records by instance id and the ids that were not prepared.
    """
    import json
    import multiprocessing as mp
    from pathlib import Path

    from lpbench.runner import atomic_write_json

    units = Path(out_dir) / "units"
    units.mkdir(parents=True, exist_ok=True)
    hard = time_limit_s + budget_s + HARD_SLACK_S
    ctx = mp.get_context("spawn")
    missing = [i.id for i in instances if not i.canonical_path().exists()]
    todo = [
        i
        for i in instances
        if i.canonical_path().exists() and not (units / f"{i.suite}__{i.name}.json").exists()
    ]
    log(f"exact: {len(instances) - len(todo) - len(missing)} done, {len(todo)} to do, "
        f"{len(missing)} not prepared")  # fmt: skip
    running: dict[str, tuple[Any, float, Path]] = {}
    queue = list(todo)
    while queue or running:
        while queue and len(running) < workers:
            inst = queue.pop(0)
            path = units / f"{inst.suite}__{inst.name}.json"
            proc = ctx.Process(
                target=_child,
                args=(str(inst.canonical_path()), str(path), time_limit_s, budget_s),
                daemon=True,
            )
            proc.start()
            running[inst.id] = (proc, time.perf_counter(), path)
        time.sleep(0.2)
        for iid, (proc, started, path) in list(running.items()):
            elapsed = time.perf_counter() - started
            if proc.is_alive() and elapsed > hard:
                proc.kill()
                proc.join(10)
                atomic_write_json(path, {"level": "timeout", "seconds": elapsed,
                                         "note": f"killed after {elapsed:.0f} s"})  # fmt: skip
            elif proc.is_alive():
                continue
            proc.join(10)
            if not path.exists():
                atomic_write_json(path, {"level": "error", "seconds": elapsed,
                                         "note": f"process exit code {proc.exitcode}"})  # fmt: skip
            del running[iid]
            rec = json.loads(path.read_text(encoding="utf-8"))
            log(f"{iid}: {rec.get('level')} via {rec.get('method') or '-'} in {elapsed:.1f} s")
    recs = {}
    for i in instances:
        path = units / f"{i.suite}__{i.name}.json"
        if path.exists():
            recs[i.id] = json.loads(path.read_text(encoding="utf-8"))
    return recs, missing


def write_exact_csv(
    recs: dict[str, dict[str, Any]],
    priors: dict[str, tuple[str, float | None]],
    path: Any = None,
) -> None:
    """Merge records into the committed table (rows of other instances are kept).

    ``priors`` maps an instance id to the reference it had before (source, canonical
    value): Koch 2004 or the netlib readme for Netlib, the consensus for MIPLIB.
    """
    import csv
    from pathlib import Path

    from lpbench.instances import EXACT_REFERENCE

    path = Path(path or EXACT_REFERENCE)
    rows: dict[str, dict[str, str]] = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            rows = {f"{r['suite']}/{r['name']}": r for r in csv.DictReader(f)}

    def fmt(v: Any) -> str:
        return "" if v is None else repr(float(v)) if isinstance(v, float) else str(v)

    for iid, rec in recs.items():
        suite, name = iid.split("/", 1)
        res = (
            ExactResult(**{k: rec.get(k) for k in ExactResult.__dataclass_fields__})
            if (rec.get("level") not in ("error", "timeout"))
            else None
        )
        ref = res.reference() if res else None
        prior_src, prior = priors.get(iid, ("", None))
        diff = None
        if ref is not None and prior is not None:
            diff = f"{abs(ref - prior) / (1.0 + abs(prior)):.2e}"
        rows[iid] = {
            "suite": suite,
            "name": name,
            "level": str(rec.get("level")),
            "method": fmt(rec.get("method")),
            "reference": fmt(ref),
            "objective": fmt(rec.get("objective")),
            "lower": fmt(rec.get("lower")),
            "upper": fmt(rec.get("upper")),
            "prior_source": prior_src,
            "prior": fmt(prior),
            "prior_rel_diff": fmt(diff),
            "simplex_status": fmt(rec.get("simplex_status")),
            "basis_size": fmt(rec.get("basis_size")),
            "note": fmt(rec.get("note")),
        }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=EXACT_FIELDS, lineterminator="\n")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    tmp.replace(path)
