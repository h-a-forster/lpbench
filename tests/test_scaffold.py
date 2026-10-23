"""Scaffold checks: metadata consistency, CLI behaviour, provenance hygiene."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

import lpbench
from lpbench import cli
from lpbench.versions import TRACKED_DISTRIBUTIONS, provenance

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["version"] == lpbench.__version__


def test_solvers_pinned_exactly() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = meta["project"]["dependencies"]
    for name in ("highspy", "ortools", "pyscipopt"):
        (spec,) = [d for d in deps if d.lower().startswith(name)]
        assert "==" in spec, f"{name} must be pinned exactly, got {spec!r}"


def test_citation_version_matches() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {lpbench.__version__}" in text


def test_cli_versions_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["versions", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["lpbench"] == lpbench.__version__
    for name in TRACKED_DISTRIBUTIONS:
        assert info["distributions"][name], f"{name} not installed"


def test_cli_planned_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_cli_version_flag() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_provenance_has_no_local_paths() -> None:
    text = json.dumps(provenance())
    home = str(Path.home())
    assert home not in text
    assert str(Path(sys.executable).parent) not in text


@pytest.mark.parametrize("name", ["quick", "full"])
def test_study_configs_respect_standing_limits(name: str) -> None:
    cfg = tomllib.loads((ROOT / "configs" / f"{name}.toml").read_text(encoding="utf-8"))
    assert cfg["name"] == name
    assert 1 <= cfg["workers"] <= 3
    assert cfg["download_cap_bytes"] <= 2_000_000_000
    assert set(cfg["first_order"]) <= set(cfg["configs"])
    assert len(cfg["configs"]) == len(set(cfg["configs"])) == 7
