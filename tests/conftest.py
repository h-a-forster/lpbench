"""Shared pytest configuration: slow tests run only with --runslow."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty instance cache (``LPBENCH_CACHE``) under the test's temporary directory."""
    root = tmp_path / "cache"
    monkeypatch.setenv("LPBENCH_CACHE", str(root))
    return root


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", help="also run tests marked slow")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test (run with --runslow)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
