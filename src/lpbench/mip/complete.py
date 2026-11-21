"""Exact completion of a MIP solution (DESIGN 13.4).

The integer variables are rounded to the nearest integers and fixed; the remaining LP in
the continuous variables is solved and certified exactly (``lpbench.exact.certify_lp``):

* ``feasible``: an exactly feasible point exists; ``z_R`` is the exact objective of the
  best certified one (the LP optimum when certkkt certifies optimality);
* ``infeasible``: a rounded value lies outside its bounds, or a pure-integer point violates
  a row, or an exact Farkas certificate proves the LP infeasible;
* ``unknown``: neither within the budget;
* ``none``: there is no solution to complete.

Objective values are in the minimisation sense (a maximisation objective is negated).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from lpbench.mip.check import round_integer
from lpbench.mip.mps import MipModel

COMPLETION_BUDGET_S = 600.0
COMPLETION_SIMPLEX_TL_S = 60.0
COMPLETIONS = ("feasible", "infeasible", "unknown", "none")
_PRIMAL_FEASIBLE = ("optimal", "bounded", "primal_feasible")


@dataclass(frozen=True)
class Completion:
    outcome: str  # one of COMPLETIONS
    z: Any = None  # exact objective (minimisation sense) when feasible
    level: str = ""  # certkkt level, "integer" (no LP), or the reason for infeasible
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "z": q_str(self.z),
            "z_float": None if self.z is None else q_nearest(self.z),
            "level": self.level,
            "seconds": self.seconds,
            "note": self.note,
        }


def q_str(v: Any) -> str | None:
    """Exact text form of a rational ("p/q" or "p")."""
    if v is None:
        return None
    return f"{int(v.numerator)}/{int(v.denominator)}" if v.denominator != 1 else str(v.numerator)


def q_parse(text: str | None) -> Any:
    from certkkt.rational import Q

    if text is None:
        return None
    p, _, q = text.partition("/")
    return Q(int(p), int(q or 1))


def q_nearest(v: Any) -> float:
    return int(v.numerator) / int(v.denominator)


def complete(
    model: MipModel,
    x: list[Any] | None,
    budget_s: float = COMPLETION_BUDGET_S,
    float_model: MipModel | None = None,
) -> Completion:
    """Round, fix and complete ``x`` (rationals, model column order) on ``model``'s data.

    ``float_model`` is the float64 data HiGHS solves to propose a basis (default: ``model``
    rounded to doubles); the certificate is always for ``model``.
    """
    from certkkt.rational import ZERO

    t0 = time.perf_counter()
    if x is None:
        return Completion("none")
    fixed = list(x)
    for j, is_int in enumerate(model.integer):
        if not is_int:
            continue
        r = round_integer(x[j])
        lo, hi = model.lb[j], model.ub[j]
        if (lo is not None and r < lo) or (hi is not None and r > hi):
            return Completion(
                "infeasible",
                level="bounds",
                seconds=time.perf_counter() - t0,
                note=f"rounded {model.col_names[j]} = {r} is outside its bounds",
            )
        fixed[j] = r
    if model.is_pure_integer():
        for i, row in enumerate(model.rows):
            act = sum((a * fixed[j] for j, a in row.items()), ZERO)
            if (model.rl[i] is not None and act < model.rl[i]) or (
                model.ru[i] is not None and act > model.ru[i]
            ):
                return Completion(
                    "infeasible",
                    level="row",
                    seconds=time.perf_counter() - t0,
                    note=f"row {model.row_names[i]} is violated",
                )
        z = model.min_c0() + sum(
            (cj * xj for cj, xj in zip(model.min_c(), fixed, strict=True) if cj), ZERO
        )
        return Completion("feasible", z=z, level="integer", seconds=time.perf_counter() - t0)
    return _complete_lp(model, float_model or model.float_data(), fixed, budget_s, t0)


def _fixed_bounds(model: MipModel, fixed: list[Any]) -> tuple[list[Any], list[Any]]:
    lb = [fixed[j] if model.integer[j] else model.lb[j] for j in range(model.n)]
    ub = [fixed[j] if model.integer[j] else model.ub[j] for j in range(model.n)]
    return lb, ub


def _boxed(model: MipModel, fixed: list[Any]) -> Any:
    from certkkt import BoxedQP
    from certkkt.sparse import QMatrix

    lb, ub = _fixed_bounds(model, fixed)
    return BoxedQP(
        P=QMatrix.zeros((model.n, model.n)),
        c=model.min_c(),
        A=QMatrix.from_rows((model.m, model.n), [dict(r) for r in model.rows]),
        rl=list(model.rl),
        ru=list(model.ru),
        lb=lb,
        ub=ub,
        c0=model.min_c0(),
        name=model.name,
    )


def _float_lp(fm: MipModel, fixed: list[Any]) -> Any:
    from lpbench.lpmodel import CanonicalLP

    fa = fm.float_arrays()
    lb, ub = fa["lb"].copy(), fa["ub"].copy()
    for j in np.flatnonzero(fa["integer"]):
        lb[j] = ub[j] = float(fixed[j])
    sign = -1.0 if fm.maximize else 1.0
    return CanonicalLP(
        name=fm.name,
        A=fa["A"],
        c=sign * fa["c"],
        c0=sign * fa["c0"],
        rl=fa["rl"],
        ru=fa["ru"],
        l=lb,
        u=ub,
        maximize_flipped=fm.maximize,
    )


def _complete_lp(
    model: MipModel, fm: MipModel, fixed: list[Any], budget_s: float, t0: float
) -> Completion:
    from certkkt.sparse import BudgetExceeded

    from lpbench.exact import certify_lp

    boxed = _boxed(model, fixed)
    lp = _float_lp(fm, fixed)
    try:
        cd = certify_lp(lp, COMPLETION_SIMPLEX_TL_S, budget_s, boxed=boxed)
    except BudgetExceeded as exc:
        return Completion("unknown", seconds=time.perf_counter() - t0, note=str(exc)[:300])
    if cd.infeasible:
        return Completion(
            "infeasible", level="farkas_basis", seconds=time.perf_counter() - t0, note=cd.infeasible
        )
    feasible = [c for c in cd.certificates if c.level in _PRIMAL_FEASIBLE]
    if feasible:
        best = min(feasible, key=lambda c: c.pobj)
        return Completion(
            "feasible", z=best.pobj, level=best.level, seconds=time.perf_counter() - t0
        )
    if _farkas_ray(model, lp, boxed):
        return Completion(
            "infeasible",
            level="farkas_ray",
            seconds=time.perf_counter() - t0,
            note=f"HiGHS: {cd.simplex_status}",
        )
    note = f"HiGHS: {cd.simplex_status}; " + "; ".join(cd.notes)
    return Completion("unknown", seconds=time.perf_counter() - t0, note=note[:500])


def _farkas_ray(model: MipModel, lp: Any, boxed: Any) -> bool:
    """True if HiGHS's dual ray, read as exact rationals, proves the exact LP infeasible.

    For any multipliers y, every point with A x = r satisfies g'x - y'r = 0, g = A'y. The
    LP is infeasible when that expression is strictly positive (or strictly negative) over
    the whole box of column and row bounds.
    """
    import highspy
    from certkkt.rational import ZERO, to_q

    inf = highspy.kHighsInf
    h = highspy.Highs()
    for key, val in (
        ("output_flag", False),
        ("threads", 1),
        ("presolve", "off"),
        ("solver", "simplex"),
        ("time_limit", COMPLETION_SIMPLEX_TL_S),
    ):
        h.setOptionValue(key, val)
    csc = lp.A.tocsc()
    hl = highspy.HighsLp()
    hl.num_col_, hl.num_row_ = lp.n, lp.m
    hl.col_cost_ = lp.c
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
    _, has_ray, ray = h.getDualRay()
    if not has_ray:
        return False
    y = [to_q(float(v)) for v in ray]
    g = [ZERO] * model.n
    for i, row in enumerate(model.rows):
        if y[i]:
            for j, a in row.items():
                g[j] += a * y[i]
    terms = [(gj, boxed.lb[j], boxed.ub[j]) for j, gj in enumerate(g)]
    terms += [(-yi, model.rl[i], model.ru[i]) for i, yi in enumerate(y)]
    lo_sum: Any = ZERO
    hi_sum: Any = ZERO
    for coef, lo, hi in terms:
        if not coef:
            continue
        lo_b, hi_b = (lo, hi) if coef > 0 else (hi, lo)
        lo_sum = None if lo_sum is None or lo_b is None else lo_sum + coef * lo_b
        hi_sum = None if hi_sum is None or hi_b is None else hi_sum + coef * hi_b
    return (hi_sum is not None and hi_sum < 0) or (lo_sum is not None and lo_sum > 0)
