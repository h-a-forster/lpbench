"""Exact violation profile of a MIP solution and its primal class P0-P3 (DESIGN 13.3).

All arithmetic is rational: the solution's doubles are read as the rationals they denote
and the model's numbers come from ``lpbench.mip.mps``. A violation is measured against
three scalings, so one profile serves every tolerance model:

* ``abs``: the violation itself;
* ``rel`` (SCIP's relative difference): violation / max(1, |activity|, |side|);
* ``miplib`` (MIPLIB's solution checker): for rows violation / max(1, |side|, positive
  part of the activity, negative part of the activity), for bounds violation /
  max(1, |bound|, |x_j|).

Integrality is absolute for every model, MIPLIB's included: |x_j - round(x_j)| <= tol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lpbench.mip.mps import MipModel

#: MIPLIB 2017's class tolerance (its test scripts' linear and integrality tolerance).
MIPLIB_TOL = 1e-4
#: A claimed primal objective agrees with c'x + c0 when the difference is at most this
#: times max(1, |claim|).
OBJECTIVE_TOL = 1e-6
PRIMAL_CLASSES = ("P0", "P1", "P2", "P3")


@dataclass(frozen=True)
class ToleranceModel:
    """How a solver accepts a solution: a scaling and a tolerance per check."""

    row: tuple[str, float]  # ("abs" | "rel" | "miplib", tol)
    bound: tuple[str, float]
    integrality: float  # absolute

    def to_dict(self) -> dict[str, Any]:
        return {"row": list(self.row), "bound": list(self.bound), "integrality": self.integrality}


MIPLIB_MODEL = ToleranceModel(("miplib", MIPLIB_TOL), ("miplib", MIPLIB_TOL), MIPLIB_TOL)


def own_model(solver: str, tolerances: dict[str, Any]) -> ToleranceModel | None:
    """The solver's own acceptance rule, with the tolerance values it reported at run time.

    HiGHS checks a MIP solution's row and bound violations and its integrality against
    ``mip_feasibility_tolerance``, all absolute. SCIP compares activities with sides and
    values with bounds by relative difference against ``numerics/feastol`` and integrality
    absolutely against the same value. CBC (through Clp and CbcModel) uses the absolute
    ``primalTolerance`` for rows and bounds and ``integerTolerance`` for integrality.
    None when a tolerance the rule needs was not recorded.
    """
    try:
        if solver in ("highs", "fake"):
            t = float(tolerances["mip_feasibility_tolerance"])
            return ToleranceModel(("abs", t), ("abs", t), t)
        if solver == "scip":
            t = float(tolerances["numerics/feastol"])
            return ToleranceModel(("rel", t), ("rel", t), t)
        if solver == "cbc":
            p = float(tolerances["primalTolerance"])
            return ToleranceModel(("abs", p), ("abs", p), float(tolerances["integerTolerance"]))
    except (KeyError, TypeError, ValueError):
        return None
    raise ValueError(f"unknown solver {solver!r}")


@dataclass(frozen=True)
class Profile:
    """Exact maxima of the scaled violations (rationals) and counts of violated items."""

    row: dict[str, Any]  # scaling -> max scaled violation
    bound: dict[str, Any]
    integrality: Any
    n_rows_violated: int
    n_bounds_violated: int
    n_integrality_violated: int
    worst_row: str | None
    worst_bound: str | None
    worst_integer: str | None

    def exactly_feasible(self) -> bool:
        return not (self.n_rows_violated or self.n_bounds_violated or self.n_integrality_violated)

    def passes(self, tm: ToleranceModel) -> bool:
        from certkkt.rational import to_q

        return (
            self.row[tm.row[0]] <= to_q(tm.row[1])
            and self.bound[tm.bound[0]] <= to_q(tm.bound[1])
            and self.integrality <= to_q(tm.integrality)
        )

    def to_dict(self) -> dict[str, Any]:
        from certkkt.rational import q_float_up

        return {
            "row": {k: q_float_up(v) for k, v in self.row.items()},
            "bound": {k: q_float_up(v) for k, v in self.bound.items()},
            "integrality": q_float_up(self.integrality),
            "n_rows_violated": self.n_rows_violated,
            "n_bounds_violated": self.n_bounds_violated,
            "n_integrality_violated": self.n_integrality_violated,
            "worst_row": self.worst_row,
            "worst_bound": self.worst_bound,
            "worst_integer": self.worst_integer,
        }


def solution_vector(model: MipModel, x_by_name: dict[str, float]) -> list[Any]:
    """The solution as exact rationals in the model's column order."""
    from certkkt.rational import to_q

    missing = [c for c in model.col_names if c not in x_by_name]
    if missing:
        raise ValueError(f"solution misses {len(missing)} columns, e.g. {missing[0]!r}")
    extra = set(x_by_name) - set(model.col_names)
    if extra:
        raise ValueError(f"solution has {len(extra)} unknown columns, e.g. {sorted(extra)[0]!r}")
    return [to_q(float(x_by_name[c])) for c in model.col_names]


def _max_one(*vals: Any) -> Any:
    from certkkt.rational import ONE

    out = ONE
    for v in vals:
        if v > out:
            out = v
    return out


def profile(model: MipModel, x: list[Any]) -> Profile:
    """Exact violation profile of ``x`` (rationals, model column order)."""
    from certkkt.rational import ZERO

    row = {"abs": ZERO, "rel": ZERO, "miplib": ZERO}
    bound = {"abs": ZERO, "rel": ZERO, "miplib": ZERO}
    worst: dict[str, tuple[Any, str] | None] = {"row": None, "bound": None, "int": None}
    n_row = n_bound = n_int = 0
    for i, coefs in enumerate(model.rows):
        pos = neg = ZERO
        for j, a in coefs.items():
            t = a * x[j]
            if t > 0:
                pos += t
            else:
                neg -= t
        act = pos - neg
        lo, hi = model.rl[i], model.ru[i]
        if lo is not None and act < lo:
            viol, side = lo - act, lo
        elif hi is not None and act > hi:
            viol, side = act - hi, hi
        else:
            continue
        n_row += 1
        scaled = {
            "abs": viol,
            "rel": viol / _max_one(abs(act), abs(side)),
            "miplib": viol / _max_one(abs(side), pos, neg),
        }
        for k, v in scaled.items():
            row[k] = max(row[k], v)
        if worst["row"] is None or scaled["miplib"] > worst["row"][0]:
            worst["row"] = (scaled["miplib"], model.row_names[i])
    integ = ZERO
    for j, xj in enumerate(x):
        lo, hi = model.lb[j], model.ub[j]
        b = None
        if lo is not None and xj < lo:
            viol, b = lo - xj, lo
        elif hi is not None and xj > hi:
            viol, b = xj - hi, hi
        if b is not None:
            n_bound += 1
            s = viol / _max_one(abs(b), abs(xj))
            bound["abs"] = max(bound["abs"], viol)
            bound["rel"] = max(bound["rel"], s)
            bound["miplib"] = max(bound["miplib"], s)
            if worst["bound"] is None or s > worst["bound"][0]:
                worst["bound"] = (s, model.col_names[j])
        if model.integer[j]:
            frac = abs(xj - _round(xj))
            if frac:
                n_int += 1
                if frac > integ:
                    integ = frac
                    worst["int"] = (frac, model.col_names[j])
    return Profile(
        row=row,
        bound=bound,
        integrality=integ,
        n_rows_violated=n_row,
        n_bounds_violated=n_bound,
        n_integrality_violated=n_int,
        worst_row=worst["row"][1] if worst["row"] else None,
        worst_bound=worst["bound"][1] if worst["bound"] else None,
        worst_integer=worst["int"][1] if worst["int"] else None,
    )


def _round(v: Any) -> Any:
    """Nearest integer (halves rounded up), as a rational."""
    from certkkt.rational import Q

    f = v.numerator // v.denominator  # floor
    return Q(f + 1) if 2 * (v - f) >= 1 else Q(f)


def round_integer(v: Any) -> Any:
    return _round(v)


def primal_class(p: Profile, own: ToleranceModel | None) -> str | None:
    """P0 exactly feasible; P1 within the solver's own tolerances; P2 within MIPLIB's;
    P3 beyond. None when the solver's own rule is unknown and the point is not P0."""
    if p.exactly_feasible():
        return "P0"
    if own is None:
        return None
    if p.passes(own):
        return "P1"
    if p.passes(MIPLIB_MODEL):
        return "P2"
    return "P3"


def objective_value(model: MipModel, x: list[Any]) -> Any:
    """c'x + c0 in the source sense, exactly."""
    s = model.c0
    for cj, xj in zip(model.c, x, strict=True):
        if cj:
            s += cj * xj
    return s


def objective_check(model: MipModel, x: list[Any], claim: float | None) -> dict[str, Any]:
    """Compare the claimed primal objective with c'x + c0 (source sense)."""
    from certkkt.rational import ONE, q_float_up, to_q

    val = objective_value(model, x)
    out: dict[str, Any] = {"objective_exact_float": int(val.numerator) / int(val.denominator)}
    if claim is None:
        out.update(objective_rel_diff=None, objective_ok=None)
        return out
    cq = to_q(float(claim))
    rel = abs(val - cq) / max(ONE, abs(cq))
    out.update(objective_rel_diff=q_float_up(rel), objective_ok=bool(rel <= to_q(OBJECTIVE_TOL)))
    return out


def classify(
    model: MipModel,
    x_by_name: dict[str, float],
    solver: str,
    tolerances: dict[str, Any],
    primal_claim: float | None,
) -> dict[str, Any]:
    """Primal class, violation profile and objective check of one solution."""
    x = solution_vector(model, x_by_name)
    p = profile(model, x)
    own = own_model(solver, tolerances)
    return {
        "primal_class": primal_class(p, own),
        "violations": p.to_dict(),
        "own_tolerance_model": own.to_dict() if own else None,
        **objective_check(model, x, primal_claim),
    }
