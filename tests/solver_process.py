"""Run a test body in a process of its own: on Linux OR-Tools cannot load next to highspy."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_pool: ProcessPoolExecutor | None = None


def in_ortools_process(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a module-level ``fn`` in one shared child that loads no other solver family."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
    return _pool.submit(fn, *args).result()
