"""HiGHS adapters: dual simplex, IPX (with crossover), PDLP (cuPDLP-C port) and HiPDLP."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from lpbench.solvers.base import (
    EXTERNAL_TIME_LIMIT,
    ConfigurationError,
    SolveRequest,
    SolveResult,
    SolverStatus,
    name_index,
    reorder,
    timed,
)

_SOLVER = {
    "highs-dsimplex": "simplex",
    "highs-ipx": "ipx",
    "highs-pdlp": "pdlp",
    "highs-hipdlp": "hipdlp",
}

_DEFAULT_FEAS = 1e-7  # primal/dual_feasibility_tolerance default in HiGHS 1.15
_DEFAULT_IPM = 1e-8  # ipm_optimality_tolerance default


def _status_map() -> dict[Any, SolverStatus]:
    import highspy

    M = highspy.HighsModelStatus
    return {
        M.kOptimal: SolverStatus.OPTIMAL,
        M.kInfeasible: SolverStatus.INFEASIBLE,
        M.kUnbounded: SolverStatus.UNBOUNDED,
        M.kUnboundedOrInfeasible: SolverStatus.INF_OR_UNBD,
        M.kTimeLimit: SolverStatus.TIME_LIMIT,
        M.kIterationLimit: SolverStatus.ITER_LIMIT,
        M.kUnknown: SolverStatus.NUMERICAL,
    }


class HighsAdapter:
    def __init__(self, config: str) -> None:
        if config not in _SOLVER:
            raise ValueError(config)
        self.config = config

    def options(self, eps: float, time_limit_s: float, seed: int) -> dict[str, Any]:
        """Options set beyond the defaults (DESIGN 4.3, and section 12 for kkt_tolerance)."""
        opts: dict[str, Any] = {
            "solver": _SOLVER[self.config],
            "threads": 1,
            "random_seed": int(seed) % 2_147_483_647,
        }
        if self.config not in EXTERNAL_TIME_LIMIT:
            # highs-pdlp (cuPDLP-C) stops far too early with any finite time_limit in
            # highspy 1.15.1 on Windows; the runner enforces its limit (DESIGN 12, item 14)
            opts["time_limit"] = float(time_limit_s)
        if self.config == "highs-dsimplex":
            opts["simplex_strategy"] = 1  # dual simplex, serial
        if self.config in ("highs-dsimplex", "highs-ipx") and eps < _DEFAULT_FEAS:
            opts["primal_feasibility_tolerance"] = eps
            opts["dual_feasibility_tolerance"] = eps
        if self.config == "highs-ipx":
            opts["run_crossover"] = "on"
            if eps < _DEFAULT_IPM:
                opts["ipm_optimality_tolerance"] = eps
        if self.config in ("highs-pdlp", "highs-hipdlp"):
            # In highspy 1.15.1 the PDLP solvers terminate on the KKT tolerances;
            # pdlp_optimality_tolerance alone cannot loosen them (DESIGN 12).
            opts["kkt_tolerance"] = eps
        return opts

    def solve(self, req: SolveRequest, extra: dict[str, Any] | None = None) -> SolveResult:
        import highspy

        opts = self.options(req.eps, req.time_limit_s, req.seed)
        opts.update(extra or {})
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        t0 = time.perf_counter()
        if h.readModel(str(req.mps_path)) == highspy.HighsStatus.kError:
            return SolveResult(SolverStatus.ERROR, "read error", options=opts)
        read_s = time.perf_counter() - t0
        try:
            for key, value in opts.items():
                if h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                    raise ConfigurationError(f"HiGHS rejected option {key}={value!r}")
        except ConfigurationError as exc:
            return SolveResult(
                SolverStatus.CONFIG_ERROR, "option rejected", options=opts, message=str(exc)
            )
        with timed() as t:
            run_status = h.run()
        ms = h.getModelStatus()
        raw = h.modelStatusToString(ms)
        status = _status_map().get(ms, SolverStatus.ERROR)
        if run_status == highspy.HighsStatus.kError and status == SolverStatus.OPTIMAL:
            status = SolverStatus.ERROR
        if self.config in EXTERNAL_TIME_LIMIT and t.cpu > req.time_limit_s:
            status = SolverStatus.TIME_LIMIT  # finished, but after the limit
        info = h.getInfo()
        iters = {
            "simplex": int(info.simplex_iteration_count),
            "ipm": int(info.ipm_iteration_count),
            "crossover": int(info.crossover_iteration_count),
            "pdlp": int(info.pdlp_iteration_count),
        }
        iters = {k: max(v, 0) for k, v in iters.items() if v > 0}
        sol = h.getSolution()
        x = y = None
        lp = h.getLp()
        if sol.value_valid:
            x = reorder(
                np.asarray(sol.col_value, dtype=np.float64),
                name_index(list(lp.col_names_), "C"),
                int(lp.num_col_),
            )
        if sol.dual_valid:
            y = reorder(
                np.asarray(sol.row_dual, dtype=np.float64),
                name_index(list(lp.row_names_), "R"),
                int(lp.num_row_),
            )
        obj = float(info.objective_function_value) if sol.value_valid else None
        return SolveResult(
            status=status,
            raw_status=raw,
            objective=obj,
            x=x,
            y=y,
            iterations=iters,
            solve_wall_s=t.wall,
            solve_cpu_s=t.cpu,
            solver_time_s=float(h.getRunTime()),
            read_wall_s=read_s,
            options=opts,
        )
