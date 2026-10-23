"""Fake adapters for harness tests (never part of a study config).

* ``test-ok``: solves with HiGHS dual simplex (a real result through a test id).
* ``test-sleep``: sleeps for ``LPBENCH_TEST_SLEEP_S`` seconds (default 3600), ignoring the
  time limit, to exercise the hard deadline.
* ``test-crash``: terminates the process with ``os._exit(3)``.
* ``test-raise``: raises an exception inside the adapter.
"""

from __future__ import annotations

import os
import time
from typing import Any

from lpbench.solvers.base import SolveRequest, SolveResult


class TestAdapter:
    __test__ = False  # not a pytest class

    def __init__(self, config: str) -> None:
        if config not in ("test-ok", "test-sleep", "test-crash", "test-raise"):
            raise ValueError(config)
        self.config = config

    def options(self, eps: float, time_limit_s: float, seed: int) -> dict[str, Any]:
        return {"fake": self.config, "time_limit_s": time_limit_s}

    def solve(self, req: SolveRequest) -> SolveResult:
        if self.config == "test-sleep":
            time.sleep(float(os.environ.get("LPBENCH_TEST_SLEEP_S", "3600")))
        if self.config == "test-crash":
            os._exit(3)
        if self.config == "test-raise":
            raise RuntimeError("deliberate failure")
        from lpbench.solvers.highs import HighsAdapter

        res = HighsAdapter("highs-dsimplex").solve(req)
        res.options = self.options(req.eps, req.time_limit_s, req.seed)
        return res
