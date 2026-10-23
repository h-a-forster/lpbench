"""Manifests, reference values, download policy and the stratified subset (no network)."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest

from lpbench import download as dl
from lpbench.config import load_config
from lpbench.instances import (
    CHECKSUMS,
    MIPLIB_MANIFEST,
    NETLIB_REFERENCE,
    canonical_reference,
    miplib_instances,
    netlib_instances,
    nnz_tertile,
    stratified_subset,
    suite_instances,
)

EXCLUDE = ("QAP8", "QAP12", "QAP15", "STOCFOR3", "TRUSS", "STANDGUB")


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_netlib_suite_has_92_instances_with_koch_values() -> None:
    insts = netlib_instances(EXCLUDE)
    assert len(insts) == 92
    assert all(i.reference_source == "koch2004" for i in insts)
    names = {i.name for i in insts}
    assert {"AFIRO", "MAROS-R7", "DFL001", "PILOT.JA", "VTP.BASE"} <= names
    assert not names & set(EXCLUDE)
    afiro = next(i for i in insts if i.name == "AFIRO")
    assert afiro.reference == pytest.approx(-464.75314285714285714, rel=1e-15)
    assert afiro.url == "https://netlib.org/lp/data/afiro"


def test_koch_values_agree_with_readme_except_flagged() -> None:
    """DESIGN 3.1: Koch and readme agree to 1e-6 relative unless the readme flags the problem."""
    checked = 0
    for r in _rows(NETLIB_REFERENCE):
        if not (r["koch_value"] and r["readme_value"]):
            continue
        k, m = float(r["koch_value"]), float(r["readme_value"])
        rel = abs(k - m) / (1 + abs(k))
        assert (k < 0) == (m < 0), r["name"]
        if r["readme_flag"]:
            continue
        assert rel <= 1e-6, (r["name"], rel)
        checked += 1
    assert checked == 70  # 94 problems with both values, minus 24 the readme flags


def test_koch_value_spot_checks() -> None:
    """Values transcribed from Koch (2004), Tables 2 and 3, compared digit by digit."""
    vals = {r["name"]: r["koch_value"] for r in _rows(NETLIB_REFERENCE)}
    assert vals["AFIRO"] == "-0.46475314285714285714285714285714E3"
    assert vals["80BAU3B"] == "0.98722419240909025757911711738799E6"
    assert vals["SC50B"] == "-0.7E2"
    assert vals["PILOT87"] == "0.3017103473331105277216600201995E3"
    assert vals["TUFF"] == "0.29214776509361284488226309994302E0"
    assert vals["D2Q06C"] == "0.12278421081418945895739128812392E6"


def test_e226_reference_includes_objective_constant() -> None:
    e226 = next(i for i in netlib_instances(EXCLUDE) if i.name == "E226")
    assert canonical_reference(e226, 7.113, False) == pytest.approx(-18.751929066 + 7.113, 1e-9)


def test_miplib_manifest_counts() -> None:
    rows = _rows(MIPLIB_MANIFEST)
    assert len(rows) == 240
    assert len(miplib_instances(100_000)) == 114
    assert len(miplib_instances(1_000_000)) == 210
    for r in rows:
        assert int(r["nnz"]) > 0 and int(r["rows"]) >= 0 and int(r["cols"]) > 0
        assert set(r) == {
            "name",
            "rows",
            "cols",
            "nnz",
            "binaries",
            "integers",
            "continuous",
            "status",
        }


def test_checksum_manifest_covers_quick_suites() -> None:
    cfg = load_config("quick")
    sums = {f"{r['suite']}/{r['file']}": r for r in _rows(CHECKSUMS)}
    for suite in ("netlib", "miplib"):
        for inst in suite_instances(cfg, suite):
            r = sums[f"{inst.suite}/{inst.file}"]
            assert len(r["sha256"]) == 64 and int(r["bytes"]) > 0
    total = sum(int(r["bytes"]) for r in sums.values())
    assert total < cfg.download_cap_bytes


def test_stratified_subset_is_deterministic_and_balanced() -> None:
    insts = netlib_instances(EXCLUDE) + miplib_instances(100_000)
    a = stratified_subset(insts, 15, seed=1, salt="perm")
    b = stratified_subset(list(reversed(insts)), 15, seed=1, salt="perm")
    c = stratified_subset(insts, 15, seed=2, salt="perm")
    assert [i.id for i in a] == [i.id for i in b]
    assert [i.id for i in a] != [i.id for i in c]
    assert len(a) == 30
    for suite in ("netlib", "miplib"):
        tert = [nnz_tertile(i, insts) for i in a if i.suite == suite]
        assert sorted(tert) == [0] * 5 + [1] * 5 + [2] * 5


def test_subset_of_small_pool_returns_everything() -> None:
    insts = netlib_instances(EXCLUDE)[:4]
    assert len(stratified_subset(insts, 15, seed=1)) == 4


# ---------------------------------------------------------------------------------------------
# Download policy (fake network)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://netlib.org/lp/data/afiro",
        "https://example.com/afiro",
        "https://netlib.org.evil.com/afiro",
        "ftp://miplib.zib.de/x",
    ],
)
def test_disallowed_urls_are_refused(url: str) -> None:
    with pytest.raises(dl.DownloadError):
        dl.check_host(url)


def test_allowed_urls() -> None:
    dl.check_host("https://netlib.org/lp/data/afiro")
    dl.check_host("https://miplib.zib.de/WebData/instances/x.mps.gz")


def test_budget_cap() -> None:
    b = dl.Budget(cap_bytes=100)
    b.reserve(60, "a")
    with pytest.raises(dl.CapExceeded):
        b.reserve(41, "b")
    b.reserve(40, "c")
    assert b.used_bytes == 100


class _FakeResp(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()


def _fake_network(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    calls: list[str] = []

    def fake_open(url: str, method: str = "GET") -> _FakeResp:
        calls.append(method)
        return _FakeResp(payload)

    monkeypatch.setattr(dl, "_open", fake_open)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    return calls


def test_download_writes_file_and_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello netlib\n" * 100
    calls = _fake_network(monkeypatch, payload)
    budget = dl.Budget(cap_bytes=10_000)
    dest = tmp_path / "raw" / "afiro"
    digest, n, did = dl.download("https://netlib.org/lp/data/afiro", dest, budget)
    assert did and n == len(payload) and dest.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert budget.used_bytes == len(payload)
    assert calls == ["HEAD", "GET"]
    assert not (tmp_path / "raw" / "afiro.part").exists()
    # second call reuses the verified file without network access
    calls.clear()
    assert dl.download("https://netlib.org/lp/data/afiro", dest, budget, digest)[2] is False
    assert calls == []


def test_download_checksum_mismatch_is_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_network(monkeypatch, b"changed upstream")
    with pytest.raises(dl.ChecksumMismatch):
        dl.download("https://netlib.org/lp/data/afiro", tmp_path / "f", dl.Budget(10_000), "0" * 64)
    assert not (tmp_path / "f").exists()


def test_download_respects_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_network(monkeypatch, b"x" * 500)
    with pytest.raises(dl.CapExceeded):
        dl.download("https://netlib.org/lp/data/afiro", tmp_path / "f", dl.Budget(100))
    assert not (tmp_path / "f").exists()
