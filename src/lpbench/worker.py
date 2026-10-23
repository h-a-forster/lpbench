"""Persistent worker process: pin threads, solve one unit at a time, verify, report.

A worker receives task dicts over a pipe and answers with record dicts (no vectors). It
never writes unit records itself; the parent is the single writer, so a killed or crashed
worker cannot leave a partial record behind.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Any

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pin_threads() -> None:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value


def _finite_or_none(v: float | None) -> float | None:
    if v is None:
        return None
    return float(v) if math.isfinite(v) else None


def _peak_rss_kb() -> int | None:
    try:
        import resource  # POSIX only
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss // 1024) if sys.platform == "darwin" else int(rss)


def execute(task: dict[str, Any]) -> dict[str, Any]:
    """Solve and verify one unit. Never raises: failures become ERROR records."""
    from pathlib import Path

    import numpy as np

    from lpbench.lpmodel import load_npz, make_permutation, unpermute_dual, unpermute_primal
    from lpbench.solvers.base import SolveRequest, SolverStatus
    from lpbench.solvers.registry import make_adapter
    from lpbench.verify import REPORT_EPS, objective_agrees, objective_rel_error, verify

    rec: dict[str, Any] = {"worker_pid": os.getpid()}
    try:
        req = SolveRequest(
            config=task["config"],
            mps_path=Path(task["mps_path"]),
            eps=float(task["eps"]),
            time_limit_s=float(task["time_limit_s"]),
            seed=int(task["seed"]),
        )
        res = make_adapter(task["config"]).solve(req)
    except Exception as exc:  # adapter bug or library exception
        rec.update(
            status=SolverStatus.ERROR.value,
            raw_status="exception",
            message=f"{type(exc).__name__}: {exc}"[:2000],
            traceback=traceback.format_exc()[-4000:],
        )
        return rec

    rec.update(
        status=res.status.value,
        raw_status=res.raw_status,
        objective_solver=_finite_or_none(res.objective),
        iterations=res.iterations,
        total_iterations=res.total_iterations,
        solve_wall_s=res.solve_wall_s,
        solve_cpu_s=res.solve_cpu_s,
        solver_time_s=res.solver_time_s,
        read_wall_s=res.read_wall_s,
        options=res.options,
        message=res.message,
        # process CPU clearly above wall time means more than one busy thread; the margin
        # covers the coarse process-time clock on Windows (about 15.6 ms)
        hidden_threads=bool(
            res.solve_wall_s > 0.25 and res.solve_cpu_s > 1.2 * res.solve_wall_s + 0.05
        ),
    )
    rss = _peak_rss_kb()
    if rss is not None:
        rec["worker_peak_rss_kb"] = rss

    try:
        lp = load_npz(Path(task["canonical_path"]))
        k = int(task["k"])
        x = res.x
        y = res.y
        if k and (x is not None or y is not None):
            perm = make_permutation(lp.m, lp.n, k, int(task["perm_seed"]), task["name"])
            x = unpermute_primal(x, perm) if x is not None else None
            y = unpermute_dual(y, perm) if y is not None else None
        if x is not None and x.shape != (lp.n,):
            raise ValueError(f"primal has {x.shape[0]} entries, problem has {lp.n} columns")
        if y is not None and y.shape != (lp.m,):
            y = None  # dual of the wrong size: treat as missing
        v = verify(lp, x, y)
        ref = task.get("reference")
        eps = float(task["eps"])
        verified: dict[str, bool] = {}
        floor: dict[str, bool] = {}
        if v is not None:
            rec["verification"] = {
                k_: (_finite_or_none(val) if isinstance(val, float) else val)
                for k_, val in v.to_dict().items()
            }
            rec["objective"] = _finite_or_none(v.primal_objective)
            rec["objective_source_sense"] = _finite_or_none(lp.source_objective(v.primal_objective))
            rec["max_rel"] = _finite_or_none(v.max_rel)
            for e in sorted(set(REPORT_EPS) | {eps}):
                key = f"{e:.0e}"
                ok = v.passes(e)
                if ref is not None:
                    ok = ok and objective_agrees(v.primal_objective, float(ref), e)
                verified[key] = bool(ok)
                floor[key] = bool(v.floor_limited(e))
        else:
            rec["verification"] = None
            rec["objective"] = None
            rec["max_rel"] = None
        rec["verified_at"] = verified
        rec["floor_limited_at"] = floor
        if ref is not None and v is not None:
            rec["ref_rel_error"] = _finite_or_none(objective_rel_error(v.primal_objective, ref))
        is_opt = res.status == SolverStatus.OPTIMAL
        rec["verified_solve"] = bool(is_opt and verified.get(f"{eps:.0e}", False))
        rec["claimed_unverified"] = bool(is_opt and not rec["verified_solve"])
        if not np.isfinite(res.solve_wall_s):
            rec["solve_wall_s"] = None
    except Exception as exc:
        rec.update(
            status=SolverStatus.ERROR.value,
            message=f"verification failed: {type(exc).__name__}: {exc}"[:2000],
            traceback=traceback.format_exc()[-4000:],
            verified_solve=False,
            claimed_unverified=False,
        )
    return rec


def warm_up() -> None:
    """Import the solver libraries once, so no unit's deadline pays for the imports."""
    import highspy  # noqa: F401
    import pyscipopt  # noqa: F401
    from ortools.linear_solver import pywraplp  # noqa: F401
    from ortools.linear_solver.python import model_builder  # noqa: F401

    import lpbench.solvers.highs  # noqa: F401
    import lpbench.solvers.ortools  # noqa: F401
    import lpbench.solvers.scip  # noqa: F401
    import lpbench.verify  # noqa: F401


def worker_main(conn: Any) -> None:  # pragma: no cover - runs in a child process
    """Entry point of a persistent worker (spawn start method)."""
    pin_threads()
    try:
        warm_up()
        conn.send({"ready": os.getpid()})
    except (EOFError, OSError, BrokenPipeError):
        return
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            return
        if task is None:
            return
        try:
            out = execute(task)
        except BaseException as exc:  # last resort; execute() should not raise
            out = {"status": "ERROR", "message": f"worker failure: {exc!r}"}
        out["unit_id"] = task["unit_id"]
        try:
            conn.send(out)
        except (EOFError, OSError, BrokenPipeError):
            return
