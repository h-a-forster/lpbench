"""Provenance: library versions, platform and git revision, recorded with every result."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from lpbench import __version__

#: Distributions whose versions determine a result (PyPI names).
TRACKED_DISTRIBUTIONS = ("highspy", "ortools", "pyscipopt", "numpy", "scipy")


def distribution_versions() -> dict[str, str | None]:
    """Installed version of each tracked distribution, or None if it is not installed."""
    out: dict[str, str | None] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def solver_library_versions() -> dict[str, str | None]:
    """Versions reported by the solver libraries themselves (not the Python wrappers)."""
    out: dict[str, str | None] = {"highs": None, "scip": None}
    try:
        import highspy

        out["highs"] = str(highspy.Highs().version())
    except Exception:  # pragma: no cover - depends on the installed wheel
        pass
    try:
        from pyscipopt import Model

        out["scip"] = str(Model().version())
    except Exception:  # pragma: no cover - depends on the installed wheel
        pass
    return out


def git_revision(root: Path | None = None) -> str | None:
    """Short git SHA of the working tree (with "-dirty" if modified), or None outside git."""
    cwd = root if root is not None else Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def platform_info() -> dict[str, str | int | None]:
    """Machine description for reports. Contains no user or host names."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
    }


def provenance() -> dict[str, object]:
    """Everything recorded alongside a run."""
    return {
        "lpbench": __version__,
        "distributions": distribution_versions(),
        "solver_libraries": solver_library_versions(),
        "git": git_revision(),
        "platform": platform_info(),
        "executable_basename": Path(sys.executable).name,
    }
