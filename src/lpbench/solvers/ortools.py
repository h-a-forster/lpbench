"""OR-Tools adapters: GLOP (simplex) and PDLP, through the MPSolver (pywraplp) interface.

The MPS file is read by OR-Tools' own MPS reader (``model_builder``) and loaded into an
MPSolver, which exposes iteration counts. Both steps happen before the timed region.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from lpbench.solvers.base import (
    SolveRequest,
    SolveResult,
    SolverStatus,
    time_limit_status,
    timed,
)

_SOLVER = {"ortools-glop": "GLOP", "ortools-pdlp": "PDLP"}


def pdlp_parameters(eps: float) -> str:
    return (
        "termination_criteria { optimality_norm: OPTIMALITY_NORM_L_INF "
        f"simple_optimality_criteria {{ eps_optimal_absolute: {eps!r} "
        f"eps_optimal_relative: {eps!r} }} }} num_threads: 1"
    )


class OrtoolsAdapter:
    def __init__(self, config: str) -> None:
        if config not in _SOLVER:
            raise ValueError(config)
        self.config = config

    def options(self, eps: float, time_limit_s: float, seed: int) -> dict[str, Any]:
        opts: dict[str, Any] = {"solver": _SOLVER[self.config], "time_limit_s": float(time_limit_s)}
        if self.config == "ortools-glop":
            # GLOP's feasibility tolerances default to 1e-8; tightened below that.
            params = f"random_seed: {int(seed) % 2_147_483_647}"
            if eps < 1e-8:
                params += (
                    f" primal_feasibility_tolerance: {eps!r} dual_feasibility_tolerance: {eps!r}"
                )
            opts["parameters"] = params
        else:
            opts["parameters"] = pdlp_parameters(eps)
        return opts

    def solve(self, req: SolveRequest, extra_params: str = "") -> SolveResult:
        from ortools.linear_solver import pywraplp
        from ortools.linear_solver.python import model_builder as mb

        opts = self.options(req.eps, req.time_limit_s, req.seed)
        params = (opts["parameters"] + " " + extra_params).strip()
        opts["parameters"] = params
        t0 = time.perf_counter()
        model = mb.Model()
        if not model.import_from_mps_file(str(req.mps_path)):
            return SolveResult(SolverStatus.ERROR, "read error", options=opts)
        solver = pywraplp.Solver.CreateSolver(_SOLVER[self.config])
        if solver is None:
            return SolveResult(SolverStatus.CONFIG_ERROR, "solver unavailable", options=opts)
        err = solver.LoadModelFromProto(model.export_to_proto())
        if err:
            return SolveResult(SolverStatus.ERROR, f"load error: {err}", options=opts)
        del model
        read_s = time.perf_counter() - t0
        if not solver.SetSolverSpecificParametersAsString(params):
            return SolveResult(
                SolverStatus.CONFIG_ERROR,
                "INVALID_SOLVER_PARAMETERS",
                options=opts,
                message=f"OR-Tools rejected parameters {params!r}",
            )
        solver.SetTimeLimit(int(round(req.time_limit_s * 1000)))
        with timed() as t:
            code = solver.Solve()
        names = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.MODEL_INVALID: "MODEL_INVALID",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        raw = names.get(code, str(code))
        if code == pywraplp.Solver.OPTIMAL:
            status = SolverStatus.OPTIMAL
        elif code == pywraplp.Solver.INFEASIBLE:
            # GLOP's presolve reports an unbounded LP as INFEASIBLE through MPSolver (it
            # cannot tell the two apart there), so a GLOP "infeasible" is ambiguous.
            status = (
                SolverStatus.INF_OR_UNBD
                if self.config == "ortools-glop"
                else SolverStatus.INFEASIBLE
            )
        elif code == pywraplp.Solver.UNBOUNDED:
            status = SolverStatus.UNBOUNDED
        elif code == pywraplp.Solver.MODEL_INVALID:
            status = SolverStatus.ERROR
        elif time_limit_status(t.wall, req.time_limit_s):
            status = SolverStatus.TIME_LIMIT
        else:
            status = SolverStatus.NUMERICAL
        x = y = None
        obj = None
        if code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            x = np.fromiter((v.solution_value() for v in solver.variables()), dtype=np.float64)
            y = np.fromiter((c.dual_value() for c in solver.constraints()), dtype=np.float64)
            obj = float(solver.Objective().Value())
        return SolveResult(
            status=status,
            raw_status=raw,
            objective=obj,
            x=x,
            y=y,
            iterations={"iterations": int(solver.iterations())},
            solve_wall_s=t.wall,
            solve_cpu_s=t.cpu,
            solver_time_s=float(solver.wall_time()) / 1000.0,
            read_wall_s=read_s,
            options=opts,
        )
