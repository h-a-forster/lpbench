"""Netlib compressed-format expander: token decoding, checksums, a synthetic problem.

No netlib data is committed (licence unstated), so the full-suite checks are slow tests that
run against a local cache populated by ``lpbench fetch``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from lpbench.lpmodel import read_with_highs
from lpbench.netlib_emps import TRTAB, NetlibFormatError, expand, format_literal, line_checksum

T = TRTAB


def _decode(chars: str) -> str:
    vals = iter(TRTAB.index(c) for c in chars[1:])
    return format_literal(TRTAB.index(chars[0]), lambda: next(vals)).strip()


def idx(x: int) -> str:
    """Encode a supersparse index."""
    if x <= 22:
        return T[x + 23]
    digits = []
    while x:
        digits.append(x % 46)
        x //= 46
    digits.reverse()
    if digits[0] >= 23:
        digits.insert(0, 0)
    return "".join(T[d] for d in digits[:-1]) + T[digits[-1] + 46]


def int_lit(x: int) -> str:
    neg = 23 if x < 0 else 0
    x = abs(x)
    if x <= 5:
        return T[46 + neg + 11 + 6 + x]
    digits = []
    while x:
        digits.append(x % 46)
        x //= 46
    digits.reverse()
    if digits[0] > 5:
        digits.insert(0, 0)
    return T[46 + neg + 11 + digits[0]] + "".join(T[d] for d in digits[1:-1]) + T[digits[-1] + 46]


@pytest.mark.parametrize(
    ("chars", "expected"),
    [
        (T[46 + 17 + 0], "0."),
        (T[46 + 17 + 4], "4."),
        (T[46 + 23 + 17 + 1], "-1."),
        # general floats: k = number of extra base-92 digits, exponent char = ex + 50
        (T[46 + 1] + T[50 - 3] + T[3] + T[25], ".301"),  # 301e-3
        (T[46 + 23 + 1] + T[50 - 2] + T[1] + T[14], "-1.06"),  # -106e-2
        (T[46 + 1] + T[50 - 3] + T[25] + T[64], "2.364"),  # 2364e-3
        (T[46 + 0] + T[50 + 1] + T[31], "310."),  # 31e1
        (T[46 + 0] + T[50 - 1] + T[4], ".4"),
        (T[46 + 0] + T[50 + 0] + T[0], "0."),
        (T[46 + 0] + T[50 - 13] + T[15], "1.5E-12"),  # 15e-13
    ],
)
def test_literal_decoding(chars: str, expected: str) -> None:
    got = _decode(chars)
    assert got == expected


def test_large_exponent_uses_e_notation() -> None:
    got = _decode(T[46 + 0] + T[50 + 12] + T[15])
    assert got == "1.5E13" and float(got) == 15e12


@pytest.mark.parametrize("x", [0, 1, 22, 23, 45, 46, 100, 2115, 2116, 97335, 123456])
def test_index_encoding_round_trip(x: int) -> None:
    from lpbench.netlib_emps import _Reader, _Tokens

    tok = _Tokens(_Reader(""))
    tok.line, tok.pos = idx(x), 0
    assert tok.index() == x
    assert tok.pos == len(tok.line)


@pytest.mark.parametrize("x", [0, 5, 6, 45, 46, 300, 99999, -1, -7, -4000])
def test_integer_literal_round_trip(x: int) -> None:
    assert float(_decode(int_lit(x))) == x


def test_checksum_function_is_position_sensitive() -> None:
    assert line_checksum("") == T[0]
    assert line_checksum("ab") != line_checksum("ba")
    assert line_checksum("NCOST") in TRTAB


def _tiny_file(corrupt: bool = False) -> str:
    """min x1 - x2  s.t.  x1 + 2 x2 <= 4,  0 <= x1 <= 3,  x2 >= 0 (number table holds 4)."""
    checked = [
        int_lit(4),  # number table (ns = 1)
        "NCOST",
        "LLIM1",
        idx(0) + "X1",
        idx(1) + int_lit(1) + idx(2) + int_lit(1) + idx(0) + "X2",
        idx(1) + int_lit(-1) + idx(2) + int_lit(2),
        idx(0) + "RHS",
        idx(2) + idx(1),  # row LIM1, number-table entry 1
        idx(0) + "BND",
        idx(1) + idx(1) + int_lit(3),  # UP, column 1, value 3
    ]
    sums = " " + "".join(line_checksum(s) for s in checked)
    if corrupt:
        checked[5] = idx(1) + int_lit(-1) + idx(2) + int_lit(3)
    head = [
        "",
        "NAME          TINY",
        "       2       2       2       4       1       1       0       0",
        "       1       1       1",
    ]
    return "\n".join(head + checked + [sums]) + "\n"


def test_tiny_problem_expands_and_reads(tmp_path: Path) -> None:
    res = expand(_tiny_file())
    assert res.name == "TINY"
    assert res.checked_lines == 10 and res.checksum_lines == 1
    assert "ROWS" in res.mps and res.mps.rstrip().endswith("ENDATA")
    assert " UP BND       X1                  3.\n" in res.mps
    path = tmp_path / "tiny.mps"
    path.write_text(res.mps, encoding="latin-1", newline="\n")
    lp = read_with_highs(path, fixed_format=True)
    assert (lp.m, lp.n, lp.nnz) == (1, 2, 2)
    np.testing.assert_array_equal(lp.c, [1.0, -1.0])
    np.testing.assert_array_equal(lp.A.toarray(), [[1.0, 2.0]])
    np.testing.assert_array_equal(lp.ru, [4.0])
    np.testing.assert_array_equal(lp.u, [3.0, np.inf])


def test_checksum_error_is_detected() -> None:
    with pytest.raises(NetlibFormatError, match="checksum"):
        expand(_tiny_file(corrupt=True))


def test_truncated_file_is_detected() -> None:
    text = _tiny_file()
    with pytest.raises(NetlibFormatError):
        expand("\n".join(text.split("\n")[:8]) + "\n")


def test_missing_name_line() -> None:
    with pytest.raises(NetlibFormatError, match="NAME"):
        expand("\n\n")


def test_mystery_line_is_checksummed_and_skipped() -> None:
    lines = _tiny_file().rstrip("\n").split("\n")
    # insert a mystery line after the row names; it must be included in the checksum
    lines.insert(7, ":hello")
    checked = lines[4:-1]
    lines[-1] = " " + "".join(line_checksum(s) for s in checked)
    res = expand("\n".join(lines) + "\n")
    assert res.checked_lines == 11


# ---------------------------------------------------------------------------------------------
# Full suite (requires `lpbench fetch --suite netlib` and `lpbench prepare --suite netlib`)
# ---------------------------------------------------------------------------------------------

CACHE = Path(__file__).resolve().parents[1] / "cache"
AFIRO_EXPANDED_SHA256 = "fd3562804ff19382a9cd8bcb22ec81bffd24a4143a2290783831d8c64516a24b"


def _netlib_raw() -> list[Path]:
    d = CACHE / "raw" / "netlib"
    return sorted(p for p in d.iterdir() if not p.name.endswith(".part")) if d.exists() else []


@pytest.mark.slow
def test_afiro_expansion_matches_recorded_checksum() -> None:
    raw = CACHE / "raw" / "netlib" / "afiro"
    if not raw.exists():
        pytest.skip("netlib cache not populated")
    res = expand(raw.read_text(encoding="latin-1"))
    assert hashlib.sha256(res.mps.encode("latin-1")).hexdigest() == AFIRO_EXPANDED_SHA256
    assert (res.stats.nrow, res.stats.ncol, res.stats.nz) == (28, 32, 88)


@pytest.mark.slow
def test_every_netlib_file_expands_with_zero_checksum_errors() -> None:
    import csv

    files = _netlib_raw()
    if not files:
        pytest.skip("netlib cache not populated")
    ref_path = Path(__file__).resolve().parents[1] / "src/lpbench/data/netlib_reference.csv"
    with open(ref_path, encoding="utf-8") as f:
        ref = {r["name"].lower(): r for r in csv.DictReader(f)}
    for p in files:
        res = expand(p.read_text(encoding="latin-1"))
        r = ref[p.name]
        # readme counts include the cost row
        assert res.stats.nz == int(r["nnz"]), p.name
        assert res.stats.ncol == int(r["cols"]), p.name
        assert res.checksum_lines >= 1
