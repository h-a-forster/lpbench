"""Consensus reference objectives (DESIGN 6.3)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from lpbench.config import load_config
from lpbench.reference import (
    REFERENCE_CONFIGS,
    consensus,
    reference_config,
    write_reference_csv,
)


def rec(
    config: str, obj: float | None, max_rel: float | None = 1e-12, status: str = "OPTIMAL"
) -> dict:
    return {"config": config, "objective": obj, "max_rel": max_rel, "status": status}


def test_three_agree() -> None:
    c = consensus("p", [rec("a", 1.0), rec("b", 1.0 + 1e-10), rec("c", 1.0 - 1e-10)])
    assert c.reference == pytest.approx(1.0, abs=1e-15)
    assert c.agreed == ("a", "b", "c") and c.n_optimal == 3 and c.n_kkt_pass == 3
    assert c.spread_rel == pytest.approx(2e-10 / 2.0)


def test_majority_wins_and_outlier_is_excluded() -> None:
    c = consensus("p", [rec("a", 5.0), rec("b", 5.0), rec("c", 5.1)])
    assert c.reference == 5.0 and c.agreed == ("a", "b")


def test_kkt_failure_is_excluded() -> None:
    c = consensus("p", [rec("a", 5.0), rec("b", 5.1, max_rel=1e-6), rec("c", 5.0)])
    assert c.reference == 5.0 and c.n_kkt_pass == 2 and c.agreed == ("a", "c")
    c = consensus("p", [rec("a", 5.0), rec("b", 5.0, max_rel=None)])
    assert c.reference is None and c.note == "only one config passed the KKT check"


def test_no_consensus_cases() -> None:
    assert consensus("p", []).note == "no config passed the KKT check"
    c = consensus("p", [rec("a", 1.0), rec("b", 2.0)])
    assert c.reference is None and c.note == "kkt-passing solutions disagree"
    c = consensus("p", [rec("a", 1.0, status="TIME_LIMIT"), rec("b", 1.0, status="NUMERICAL")])
    assert c.reference is None and c.n_optimal == 0


def test_agreement_is_relative_to_magnitude() -> None:
    big = 1e9
    assert consensus("p", [rec("a", big), rec("b", big * (1 + 5e-9))]).reference is not None
    assert consensus("p", [rec("a", big), rec("b", big * (1 + 5e-8))]).reference is None


def test_reference_config() -> None:
    rc = reference_config(load_config("quick"))
    assert rc.name == "quick-reference" and tuple(rc.configs) == REFERENCE_CONFIGS
    assert rc.time_limit_s == 60.0 and [b.eps for b in rc.blocks] == [1e-9]
    assert [tuple(b.suites) for b in rc.blocks] == [("miplib",)]


def test_write_csv_merges_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "ref.csv"
    write_reference_csv([consensus("zz", [rec("a", 1.0), rec("b", 1.0)])], path)
    write_reference_csv([consensus("aa", [rec("a", 0.1)])], path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["name"] for r in rows] == ["aa", "zz"]
    assert rows[0]["reference"] == "" and rows[1]["reference"] == "1.0"
    assert rows[1]["agreed"] == "a;b"
