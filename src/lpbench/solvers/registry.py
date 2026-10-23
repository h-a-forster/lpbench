"""Config id -> adapter, and the per-tolerance option table copied into reports."""

from __future__ import annotations

from typing import Any

from lpbench.config import CONFIG_IDS
from lpbench.solvers.base import Adapter


def make_adapter(config: str) -> Adapter:
    if config.startswith("highs-"):
        from lpbench.solvers.highs import HighsAdapter

        return HighsAdapter(config)
    if config.startswith("ortools-"):
        from lpbench.solvers.ortools import OrtoolsAdapter

        return OrtoolsAdapter(config)
    if config == "scip-soplex":
        from lpbench.solvers.scip import ScipAdapter

        return ScipAdapter()
    if config.startswith("test-"):
        from lpbench.solvers.testing import TestAdapter

        return TestAdapter(config)
    raise ValueError(f"unknown config {config!r}")


def option_table(eps: float, time_limit_s: float, seed: int) -> dict[str, dict[str, Any]]:
    """Exact options set for every study config at one epsilon (verbatim in the report)."""
    return {c: make_adapter(c).options(eps, time_limit_s, seed) for c in CONFIG_IDS}


FIRST_ORDER = ("highs-pdlp", "highs-hipdlp", "ortools-pdlp")
SIMPLEX = ("highs-dsimplex", "ortools-glop", "scip-soplex")
