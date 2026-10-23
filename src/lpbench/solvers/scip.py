"""SCIP 10 / SoPlex 8 adapter: the root LP of an all-continuous problem, presolve off.

SCIP returns duals (``getDualSolVal``) only with presolving off, so presolving,
heuristics, separation and propagation are switched off; this makes ``scip-soplex`` slower
than default SCIP and the report says so.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from lpbench.solvers.base import (
    ConfigurationError,
    SolveRequest,
    SolveResult,
    SolverStatus,
    name_index,
    reorder,
    timed,
)

_STATUS = {
    "optimal": SolverStatus.OPTIMAL,
    "infeasible": SolverStatus.INFEASIBLE,
    "unbounded": SolverStatus.UNBOUNDED,
    "inforunbd": SolverStatus.INF_OR_UNBD,
    "timelimit": SolverStatus.TIME_LIMIT,
}


def _row_duals(m: Any) -> np.ndarray:
    """Row duals in the convention z = c - A^T y, including singleton rows.

    SCIP keeps a linear constraint with a single variable as a bound in its LP, so
    ``getDualsolLinear`` returns 0 for it and the multiplier sits in the variable's reduced
    cost (``getDualSolVal`` attributes that reduced cost to the row even when the variable's
    own bound is the active one, which can give the wrong sign). For each singleton row we
    therefore split the reduced cost ourselves: it belongs to the variable's own bound if that
    bound is at least as tight as the row's implied bound on the active side, otherwise to the
    tightest singleton row on that side (``y_i = r_j / a_ij``).
    """
    big = 1e20
    conss = m.getConss()
    y = np.zeros(len(conss))
    by_var: dict[str, list[tuple[int, float, float, float]]] = {}
    for k, c in enumerate(conss):
        vals = m.getValsLinear(c)
        if len(vals) == 1:
            ((vname, a),) = vals.items()
            vname = vname[2:] if vname.startswith("t_") else vname
            lhs, rhs = m.getLhs(c), m.getRhs(c)
            # implied bounds on the variable from lhs <= a x <= rhs
            lo_s = lhs if lhs > -big else -np.inf
            hi_s = rhs if rhs < big else np.inf
            if a > 0:
                imp_lo, imp_hi = lo_s / a, hi_s / a
            else:
                imp_lo, imp_hi = hi_s / a, lo_s / a
            by_var.setdefault(vname, []).append((k, a, imp_lo, imp_hi))
        else:
            y[k] = m.getDualsolLinear(c)
    if by_var:
        for v in m.getVars():
            rows = by_var.get(v.name)
            if not rows:
                continue
            r = m.getVarRedcost(v)
            if r == 0.0:
                continue
            own_lo, own_hi = v.getLbOriginal(), v.getUbOriginal()
            own_lo = own_lo if own_lo > -big else -np.inf
            own_hi = own_hi if own_hi < big else np.inf
            if r > 0:  # a lower bound is active
                k, a, imp, _ = max(rows, key=lambda t: t[2])
                if not own_lo >= imp:
                    y[k] = r / a
            else:  # an upper bound is active
                k, a, _, imp = min(rows, key=lambda t: t[3])
                if not own_hi <= imp:
                    y[k] = r / a
    return y


class ScipAdapter:
    config = "scip-soplex"

    def __init__(self, config: str = "scip-soplex") -> None:
        if config != "scip-soplex":
            raise ValueError(config)

    def options(self, eps: float, time_limit_s: float, seed: int) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "presolving": "off",
            "heuristics": "off",
            "separating": "off",
            "propagating": "off",
            "lp/threads": 1,
            # symmetry handling adds SST cuts to a pure LP, which changes the feasible set
            "misc/usesymmetry": 0,
            "limits/time": float(time_limit_s),
            "randomization/randomseedshift": int(seed) % 2_147_483_647,
            "randomization/lpseed": int(seed) % 2_147_483_647,
        }
        if eps < 1e-6:
            opts["numerics/feastol"] = eps
        if eps < 1e-7:
            opts["numerics/dualfeastol"] = eps
        return opts

    def solve(self, req: SolveRequest, extra: dict[str, Any] | None = None) -> SolveResult:
        from pyscipopt import SCIP_PARAMSETTING, Model

        opts = self.options(req.eps, req.time_limit_s, req.seed)
        opts.update(extra or {})
        m = Model()
        m.hideOutput()
        t0 = time.perf_counter()
        try:
            m.readProblem(str(req.mps_path))
        except Exception as exc:  # pragma: no cover - depends on the file
            return SolveResult(SolverStatus.ERROR, "read error", options=opts, message=str(exc))
        read_s = time.perf_counter() - t0
        try:
            for key, value in opts.items():
                if key == "presolving":
                    m.setPresolve(SCIP_PARAMSETTING.OFF)
                elif key == "heuristics":
                    m.setHeuristics(SCIP_PARAMSETTING.OFF)
                elif key == "separating":
                    m.setSeparating(SCIP_PARAMSETTING.OFF)
                elif key == "propagating":
                    m.disablePropagation()
                else:
                    try:
                        m.setParam(key, value)
                    except (KeyError, LookupError, ValueError, TypeError) as exc:
                        raise ConfigurationError(f"SCIP rejected {key}={value!r}: {exc}") from exc
                    except Exception as exc:  # SCIP raises a plain Exception for bad values
                        raise ConfigurationError(f"SCIP rejected {key}={value!r}: {exc}") from exc
        except ConfigurationError as exc:
            return SolveResult(
                SolverStatus.CONFIG_ERROR, "option rejected", options=opts, message=str(exc)
            )
        with timed() as t:
            m.optimize()
        raw = str(m.getStatus())
        status = _STATUS.get(
            raw, SolverStatus.NUMERICAL if raw != "unknown" else SolverStatus.ERROR
        )
        x = y = None
        obj = None
        if m.getNSols() > 0:
            vs = m.getVars()
            sol = m.getBestSol()
            vals = np.fromiter((m.getSolVal(sol, v) for v in vs), dtype=np.float64, count=len(vs))
            x = reorder(vals, name_index([v.name for v in vs], "C"), len(vs))
            obj = float(m.getSolObjVal(sol))
            if status == SolverStatus.OPTIMAL:
                conss = m.getConss()
                duals = _row_duals(m)
                y = reorder(duals, name_index([c.name for c in conss], "R"), len(conss))
        return SolveResult(
            status=status,
            raw_status=raw,
            objective=obj,
            x=x,
            y=y,
            iterations={"lp": int(m.getNLPIterations())},
            solve_wall_s=t.wall,
            solve_cpu_s=t.cpu,
            solver_time_s=float(m.getSolvingTime()),
            read_wall_s=read_s,
            options=opts,
        )
