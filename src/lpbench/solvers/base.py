"""Common adapter types: request, result, status enum, timing helper."""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class SolverStatus(enum.StrEnum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INF_OR_UNBD = "INF_OR_UNBD"
    TIME_LIMIT = "TIME_LIMIT"
    ITER_LIMIT = "ITER_LIMIT"
    NUMERICAL = "NUMERICAL"  # solver stopped without a conclusion (unknown / imprecise)
    ERROR = "ERROR"  # exception, solver error, or worker died
    KILLED = "KILLED"  # exceeded the hard wall deadline and was terminated
    CONFIG_ERROR = "CONFIG_ERROR"  # an option was rejected: a harness error, never a failure


@dataclass(frozen=True)
class SolveRequest:
    config: str
    mps_path: Path
    eps: float
    time_limit_s: float
    seed: int = 0
    expected_shape: tuple[int, int] | None = None  # (m, n) of the canonical problem


@dataclass
class SolveResult:
    status: SolverStatus
    raw_status: str
    objective: float | None = None  # as reported by the solver (MPS problem: no constant)
    x: np.ndarray | None = None  # primal, in the order of the MPS file's columns
    y: np.ndarray | None = None  # row duals, convention z = c - A^T y
    iterations: dict[str, int] = field(default_factory=dict)
    solve_wall_s: float = math.nan
    solve_cpu_s: float = math.nan
    solver_time_s: float | None = None
    read_wall_s: float = math.nan
    options: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def total_iterations(self) -> int:
        return int(sum(self.iterations.values()))


class Adapter(Protocol):
    config: str

    def options(self, eps: float, time_limit_s: float, seed: int) -> dict[str, Any]: ...

    def solve(self, req: SolveRequest) -> SolveResult: ...


class ConfigurationError(RuntimeError):
    """An option was rejected by the library (reported as CONFIG_ERROR)."""


@dataclass
class Timer:
    wall: float = math.nan
    cpu: float = math.nan


@contextmanager
def timed() -> Iterator[Timer]:
    """Wall (perf_counter) and process CPU time (all threads) around the block."""
    t = Timer()
    w0, c0 = time.perf_counter(), time.process_time()
    try:
        yield t
    finally:
        t.wall = time.perf_counter() - w0
        t.cpu = time.process_time() - c0


def name_index(names: list[str], prefix: str) -> np.ndarray | None:
    """Positions encoded in names like ``R12``/``C7``; None if any name is not of that form."""
    out = np.empty(len(names), dtype=np.int64)
    for k, nm in enumerate(names):
        if not nm.startswith(prefix) or not nm[len(prefix) :].isdigit():
            return None
        out[k] = int(nm[len(prefix) :])
    return out


def reorder(values: np.ndarray, positions: np.ndarray | None, size: int) -> np.ndarray:
    """Place ``values[k]`` at ``positions[k]`` (identity if positions is None)."""
    if positions is None:
        return values
    if sorted(positions.tolist()) != list(range(size)):
        raise ValueError("solver returned names that do not cover the problem")
    out = np.empty(size, dtype=np.float64)
    out[positions] = values
    return out


def time_limit_status(elapsed: float, time_limit_s: float) -> bool:
    """Heuristic for libraries that do not report why they stopped."""
    return elapsed >= 0.98 * time_limit_s


#: Configs whose in-library time limit cannot be used (DESIGN 12, item 14). The adapter does
#: not pass the limit; the runner kills the worker at ``time_limit + EXTERNAL_TL_GRACE_S`` and
#: records TIME_LIMIT, and a solve that finishes after the limit is scored as TIME_LIMIT.
EXTERNAL_TIME_LIMIT = frozenset({"highs-pdlp"})
EXTERNAL_TL_GRACE_S = 5.0


AdapterFactory = Callable[[], Adapter]
