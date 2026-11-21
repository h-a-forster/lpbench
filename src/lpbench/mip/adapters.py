"""Claim capture for MIP solvers (DESIGN 13.2).

Each adapter reads the MPS file itself, solves with one thread, the given time limit and
seed and otherwise default options, and returns the solver's claims: status, the best
solution by column name in the original space, the primal and dual bounds and the gap as
reported (source sense), and the effective tolerances read back from the solver. A field
the solver does not provide is ``None`` and listed in ``missing``; nothing is derived.

Fake adapters for harness tests (never part of a study config): ``fake-hang`` sleeps for
``LPBENCH_TEST_SLEEP_S`` seconds (default 3600), ignoring the time limit.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lpbench.solvers.base import timed

SOLVERS = ("highs", "scip")
FAKES = ("fake-hang",)
CLAIM_FIELDS = ("x", "primal_bound", "dual_bound", "gap")


@dataclass
class MipClaim:
    status: str  # OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD, TIME_LIMIT, OTHER
    raw_status: str
    x: dict[str, float] | None = None
    primal_bound: float | None = None  # objective of the reported solution (source sense)
    dual_bound: float | None = None  # claimed bound on the optimum (source sense)
    gap: float | None = None  # as the solver reports it (None when infinite)
    gap_infinite: bool = False  # the solver reports an infinite gap (e.g. dual bound 0)
    tolerances: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    nodes: int | None = None
    solve_wall_s: float = math.nan
    solve_cpu_s: float = math.nan
    version: str = ""
    missing: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        def fin(v: float | None) -> float | None:
            return None if v is None or not math.isfinite(v) else float(v)

        return {
            "status": self.status,
            "raw_status": self.raw_status,
            "x": self.x,
            "primal_bound": fin(self.primal_bound),
            "dual_bound": fin(self.dual_bound),
            "gap": fin(self.gap),
            "gap_infinite": self.gap_infinite,
            "tolerances": self.tolerances,
            "options": self.options,
            "nodes": self.nodes,
            "solve_wall_s": self.solve_wall_s,
            "solve_cpu_s": self.solve_cpu_s,
            "solver_version": self.version,
            "missing": self.missing,
        }


def _missing(claim: MipClaim) -> MipClaim:
    """List the claim fields the solver's status calls for but it did not provide: a dual
    bound unless it claims infeasibility or unboundedness; with a solution, its objective
    and the gap (an infinite gap is a value); with an objective, the solution."""
    expected = set()
    if claim.status not in ("INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"):
        expected.add("dual_bound")
    if claim.x is not None:
        expected |= {"primal_bound"} | (set() if claim.gap_infinite else {"gap"})
    if claim.primal_bound is not None:
        expected.add("x")
    claim.missing = [f for f in CLAIM_FIELDS if f in expected and getattr(claim, f) is None]
    return claim


# --------------------------------------------------------------------------------------------
# HiGHS
# --------------------------------------------------------------------------------------------

HIGHS_TOLERANCES = (
    "mip_feasibility_tolerance",
    "primal_feasibility_tolerance",
    "dual_feasibility_tolerance",
    "mip_rel_gap",
    "mip_abs_gap",
    "small_matrix_value",
    "infinite_bound",
    "infinite_cost",
)


def solve_highs(mps: Path, time_limit_s: float, seed: int) -> MipClaim:
    import highspy

    h = highspy.Highs()
    options = {
        "output_flag": False,
        "threads": 1,
        "time_limit": float(time_limit_s),
        "random_seed": int(seed),
    }
    for k, v in options.items():
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected option {k}={v!r}")
    if h.readModel(str(mps)) == highspy.HighsStatus.kError:
        raise ValueError(f"HiGHS could not read {mps.name}")
    with timed() as t:
        h.run()
    tol: dict[str, Any] = {}
    for k in HIGHS_TOLERANCES:
        st, val = h.getOptionValue(k)
        if st == highspy.HighsStatus.kOk:
            tol[k] = val
    info = h.getInfo()
    M = highspy.HighsModelStatus
    status = {
        M.kOptimal: "OPTIMAL",
        M.kInfeasible: "INFEASIBLE",
        M.kUnbounded: "UNBOUNDED",
        M.kUnboundedOrInfeasible: "INF_OR_UNBD",
        M.kTimeLimit: "TIME_LIMIT",
    }.get(h.getModelStatus(), "OTHER")
    x = None
    primal = None
    if info.primal_solution_status == 2:  # kSolutionStatusFeasible: a solution exists
        lp = h.getLp()
        vals = h.getSolution().col_value
        x = {str(nm): float(v) for nm, v in zip(lp.col_names_, vals, strict=True)}
        primal = float(info.objective_function_value)
    dual = float(info.mip_dual_bound)
    gap = float(info.mip_gap)
    return _missing(
        MipClaim(
            status=status,
            raw_status=h.modelStatusToString(h.getModelStatus()),
            x=x,
            primal_bound=primal,
            dual_bound=dual if math.isfinite(dual) else None,
            gap=gap if math.isfinite(gap) else None,
            gap_infinite=math.isinf(gap),
            tolerances=tol,
            options=options,
            nodes=int(info.mip_node_count),
            solve_wall_s=t.wall,
            solve_cpu_s=t.cpu,
            version=h.version(),
        )
    )


# --------------------------------------------------------------------------------------------
# SCIP
# --------------------------------------------------------------------------------------------

SCIP_TOLERANCES = (
    "numerics/feastol",
    "numerics/epsilon",
    "numerics/sumepsilon",
    "numerics/dualfeastol",
    "numerics/lpfeastolfactor",
    "numerics/infinity",
    "limits/gap",
    "limits/absgap",
)


def solve_scip(mps: Path, time_limit_s: float, seed: int) -> MipClaim:
    import pyscipopt

    m = pyscipopt.Model()
    m.hideOutput()
    options = {
        "limits/time": float(time_limit_s),
        "lp/threads": 1,
        "parallel/maxnthreads": 1,
        "randomization/randomseedshift": int(seed),
    }
    for k, v in options.items():
        m.setParam(k, v)
    m.readProblem(str(mps))
    with timed() as t:
        m.optimize()
    tol = {k: m.getParam(k) for k in SCIP_TOLERANCES}
    raw = m.getStatus()
    status = {
        "optimal": "OPTIMAL",
        "infeasible": "INFEASIBLE",
        "unbounded": "UNBOUNDED",
        "inforunbd": "INF_OR_UNBD",
        "timelimit": "TIME_LIMIT",
    }.get(raw, "OTHER")
    x = None
    primal = None
    if m.getNSols() > 0:
        sol = m.getBestSol()
        x = {v.name: float(m.getSolVal(sol, v)) for v in m.getVars(transformed=False)}
        primal = float(m.getSolObjVal(sol, original=True))
    dual = float(m.getDualbound())
    gap = float(m.getGap())
    big = float(m.getParam("numerics/infinity"))
    return _missing(
        MipClaim(
            status=status,
            raw_status=raw,
            x=x,
            primal_bound=primal,
            dual_bound=dual if math.isfinite(dual) and abs(dual) < big else None,
            gap=gap if math.isfinite(gap) and gap < big else None,
            gap_infinite=not (math.isfinite(gap) and gap < big),
            tolerances=tol,
            options=options,
            nodes=int(m.getNTotalNodes()),
            solve_wall_s=t.wall,
            solve_cpu_s=t.cpu,
            version=f"SCIP {m.version()}",
        )
    )


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------


def solve(solver: str, mps: Path, time_limit_s: float, seed: int) -> MipClaim:
    if solver == "highs":
        return solve_highs(mps, time_limit_s, seed)
    if solver == "scip":
        return solve_scip(mps, time_limit_s, seed)
    if solver == "fake-hang":
        time.sleep(float(os.environ.get("LPBENCH_TEST_SLEEP_S", "3600")))
        return MipClaim(status="OTHER", raw_status="woke up")
    raise ValueError(f"unknown MIP solver {solver!r}")


def solver_family(solver: str) -> str:
    """The library a solver loads (one family per worker process, DESIGN 4.2)."""
    if solver in ("highs", *FAKES):
        return "highs"
    if solver == "scip":
        return "scip"
    raise ValueError(f"unknown MIP solver {solver!r}")


def execute(task: dict[str, Any]) -> dict[str, Any]:
    """Worker side of one MIP unit. Never raises: failures become ERROR records."""
    import traceback

    rec: dict[str, Any] = {"worker_pid": os.getpid()}
    try:
        claim = solve(
            task["solver"], Path(task["mps_path"]), float(task["time_limit_s"]), int(task["seed"])
        )
    except Exception as exc:
        rec.update(
            status="ERROR",
            raw_status="exception",
            message=f"{type(exc).__name__}: {exc}"[:2000],
            traceback=traceback.format_exc()[-4000:],
        )
        return rec
    rec.update(claim.to_record())
    return rec
