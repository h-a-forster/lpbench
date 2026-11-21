"""Exact reader for MIP problems in MPS format (free format, then fixed format).

Every number is read as the rational its decimal literal denotes (``0.1`` is exactly 1/10),
so checks can run on the data as written. ``float_data`` gives the float64 data a solver
holds after reading the same file (each rational rounded to the nearest double).

Conventions follow HiGHS 1.16 and SCIP 10, which agree on all of them:

* an integer column (between ``'INTORG'`` and ``'INTEND'`` markers) without any BOUNDS
  entry has bounds [0, 1]; with any entry, the defaults are [0, inf) before it applies;
* ``UP`` with a negative value sets the upper bound only (the lower bound stays);
* ``RANGES`` on an ``E`` row gives [rhs, rhs + R] for R > 0 and [rhs + R, rhs] for R < 0;
* the RHS of the objective row is minus the objective constant;
* ``N`` rows after the first are dropped (they constrain nothing).

Semi-continuous bounds (``SC``), sections other than NAME, OBJSENSE, ROWS, COLUMNS, RHS,
RANGES, BOUNDS and ENDATA, and a second RHS, RANGES or BOUNDS set raise ``MpsError``.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SECTIONS = ("NAME", "OBJSENSE", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA")
_NO_VALUE_BOUNDS = ("FR", "MI", "PL", "BV")
_VALUE_BOUNDS = ("UP", "LO", "FX", "LI", "UI")


class MpsError(ValueError):
    """The file is not an MPS problem this reader accepts."""


@dataclass
class MipModel:
    """``min/max c'x + c0  s.t.  rl <= A x <= ru,  lb <= x <= ub,  x_j integer (j in I)``.

    Rationals throughout (certkkt's ``Q``); ``None`` is an infinite bound. ``c`` and ``c0``
    are in the source sense; ``min_c``/``min_c0`` give the minimisation form.
    """

    name: str
    col_names: list[str]
    row_names: list[str]
    integer: list[bool]
    c: list[Any]
    c0: Any
    maximize: bool
    rows: list[dict[int, Any]]  # row i: {column j: a_ij}, nonzeros only
    rl: list[Any]
    ru: list[Any]
    lb: list[Any]
    ub: list[Any]

    @property
    def m(self) -> int:
        return len(self.row_names)

    @property
    def n(self) -> int:
        return len(self.col_names)

    @property
    def nnz(self) -> int:
        return sum(len(r) for r in self.rows)

    def min_c(self) -> list[Any]:
        return [-v for v in self.c] if self.maximize else list(self.c)

    def min_c0(self) -> Any:
        return -self.c0 if self.maximize else self.c0

    def is_pure_integer(self) -> bool:
        return all(self.integer)

    def float_data(self) -> MipModel:
        """The same problem with every number rounded to the nearest double, as rationals."""
        from certkkt.rational import as_double

        def r(v: Any) -> Any:
            return None if v is None else as_double(v)

        return MipModel(
            name=self.name,
            col_names=self.col_names,
            row_names=self.row_names,
            integer=self.integer,
            c=[as_double(v) for v in self.c],
            c0=as_double(self.c0),
            maximize=self.maximize,
            rows=[{j: as_double(v) for j, v in row.items()} for row in self.rows],
            rl=[r(v) for v in self.rl],
            ru=[r(v) for v in self.ru],
            lb=[r(v) for v in self.lb],
            ub=[r(v) for v in self.ub],
        )

    def same_data(self, other: MipModel) -> bool:
        return (
            self.c == other.c
            and self.c0 == other.c0
            and self.rows == other.rows
            and self.rl == other.rl
            and self.ru == other.ru
            and self.lb == other.lb
            and self.ub == other.ub
        )

    def float_arrays(self) -> dict[str, Any]:
        """Float64 arrays (source sense, +-inf bounds) and a CSR matrix."""
        import scipy.sparse as sp

        def f(vals: list[Any], inf: float) -> np.ndarray:
            return np.array([inf if v is None else float(v) for v in vals], dtype=np.float64)

        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for row in self.rows:
            for j in sorted(row):
                indices.append(j)
                data.append(float(row[j]))
            indptr.append(len(indices))
        A = sp.csr_matrix(
            (np.array(data, dtype=np.float64), np.array(indices, dtype=np.int64), indptr),
            shape=(self.m, self.n),
        )
        return {
            "A": A,
            "c": np.array([float(v) for v in self.c], dtype=np.float64),
            "c0": float(self.c0),
            "rl": f(self.rl, -np.inf),
            "ru": f(self.ru, np.inf),
            "lb": f(self.lb, -np.inf),
            "ub": f(self.ub, np.inf),
            "integer": np.array(self.integer, dtype=bool),
        }


def read_mps(path: str | Path) -> MipModel:
    """Read an MPS file (optionally gzipped): free format first, then fixed format."""
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return parse_mps(raw.decode("latin-1"), name=p.name.split(".")[0])


def parse_mps(text: str, name: str = "") -> MipModel:
    try:
        return _Parser(text, name, fixed=False).parse()
    except MpsError as free_err:
        try:
            return _Parser(text, name, fixed=True).parse()
        except MpsError as fixed_err:
            raise MpsError(f"free format: {free_err}; fixed format: {fixed_err}") from None


def _fixed_fields(line: str) -> list[str]:
    """Fields of a fixed-format data line (columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61)."""
    spans = ((1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61))
    out = [line[a:b].strip() for a, b in spans]
    while out and not out[-1]:
        out.pop()
    if len(line.rstrip()) > 61:
        raise MpsError(f"text beyond column 61: {line!r}")
    return out


class _Parser:
    def __init__(self, text: str, name: str, fixed: bool) -> None:
        self.lines = text.splitlines()
        self.name = name
        self.fixed = fixed
        self.maximize = False
        self.obj_row: str | None = None
        self.row_index: dict[str, int] = {}
        self.row_names: list[str] = []
        self.row_type: list[str] = []
        self.free_rows: set[str] = set()
        self.col_index: dict[str, int] = {}
        self.col_names: list[str] = []
        self.integer: list[bool] = []
        self.c: list[Any] = []
        self.rows: list[dict[int, Any]] = []
        self.rhs: dict[int, Any] = {}
        self.ranges: dict[int, Any] = {}
        self.c0: Any = None
        self.bounds_given: set[int] = set()
        self.lb: list[Any] = []
        self.ub: list[Any] = []
        self.sets: dict[str, str] = {}
        self.in_int = False

    # -- helpers ------------------------------------------------------------------------------

    def _fields(self, line: str) -> list[str]:
        if not self.fixed:
            return line.split()
        f = _fixed_fields(line)
        return f[1:] if f and not f[0] else f  # field 1 is empty outside ROWS and BOUNDS

    @staticmethod
    def _num(text: str) -> Any:
        from certkkt.rational import to_q

        try:
            return to_q(text)
        except ValueError as exc:
            raise MpsError(str(exc)) from None

    def _set(self, section: str, set_name: str) -> None:
        prev = self.sets.setdefault(section, set_name)
        if prev != set_name:
            raise MpsError(f"a second {section} set ({set_name!r} after {prev!r})")

    def _row(self, name: str) -> int | None:
        """Row index, or None for the objective and dropped free rows."""
        if name == self.obj_row or name in self.free_rows:
            return None
        if name not in self.row_index:
            raise MpsError(f"unknown row {name!r}")
        return self.row_index[name]

    # -- sections -----------------------------------------------------------------------------

    def parse(self) -> MipModel:
        section = None
        pending_sense = False
        ended = False
        for raw in self.lines:
            if not raw.strip() or raw.startswith("*"):
                continue
            if ended:
                raise MpsError("text after ENDATA")
            if not raw[0].isspace():
                head = raw.split()
                section = head[0].upper()
                if section not in SECTIONS:
                    raise MpsError(f"unsupported section {head[0]!r}")
                if section == "NAME":
                    self.name = raw[4:].strip() if self.fixed else " ".join(head[1:])
                elif section == "OBJSENSE":
                    pending_sense = len(head) == 1
                    if not pending_sense:
                        self._sense(head[1])
                elif section == "ENDATA":
                    ended = True
                continue
            if section == "OBJSENSE" and pending_sense:
                self._sense(raw.strip())
                pending_sense = False
            elif section == "ROWS":
                self._rows_line(self._fields(raw))
            elif section == "COLUMNS":
                self._columns_line(raw)
            elif section == "RHS":
                self._rhs_line(self._fields(raw))
            elif section == "RANGES":
                self._ranges_line(self._fields(raw))
            elif section == "BOUNDS":
                self._bounds_line(self._fields(raw))
            else:
                raise MpsError(f"data outside a data section: {raw.strip()!r}")
        if not ended:
            raise MpsError("no ENDATA")
        if self.obj_row is None:
            raise MpsError("no objective (N) row")
        return self._model()

    def _sense(self, word: str) -> None:
        w = word.upper()
        if w in ("MAX", "MAXIMIZE", "MAXIMISE"):
            self.maximize = True
        elif w in ("MIN", "MINIMIZE", "MINIMISE"):
            self.maximize = False
        else:
            raise MpsError(f"unknown OBJSENSE {word!r}")

    def _rows_line(self, f: list[str]) -> None:
        if len(f) != 2:
            raise MpsError(f"bad ROWS line {f!r}")
        typ, name = f[0].upper(), f[1]
        if typ not in ("N", "E", "L", "G"):
            raise MpsError(f"unknown row type {f[0]!r}")
        if name in self.row_index or name == self.obj_row or name in self.free_rows:
            raise MpsError(f"duplicate row {name!r}")
        if typ == "N":
            if self.obj_row is None:
                self.obj_row = name
            else:
                self.free_rows.add(name)
            return
        self.row_index[name] = len(self.row_names)
        self.row_names.append(name)
        self.row_type.append(typ)
        self.rows.append({})

    def _column(self, name: str, create: bool, integer: bool = False) -> int:
        j = self.col_index.get(name)
        if j is None:
            if not create:
                raise MpsError(f"unknown column {name!r}")
            j = len(self.col_names)
            self.col_index[name] = j
            self.col_names.append(name)
            self.integer.append(integer)
            self.c.append(None)
        return j

    def _columns_line(self, raw: str) -> None:
        f = raw.split() if not self.fixed else None
        if f is not None and len(f) == 3 and f[1] == "'MARKER'":
            marker = f[2]
        elif self.fixed and raw[14:22].strip() == "'MARKER'":
            marker = raw[39:47].strip() or raw[24:36].strip()
        else:
            marker = None
        if marker is not None:
            if marker == "'INTORG'":
                self.in_int = True
            elif marker == "'INTEND'":
                self.in_int = False
            else:
                raise MpsError(f"unknown marker {marker!r}")
            return
        f = self._fields(raw)
        if len(f) not in (3, 5):
            raise MpsError(f"bad COLUMNS line {f!r}")
        j = self._column(f[0], create=True, integer=self.in_int)
        for row, val in ((f[1], f[2]), (f[3], f[4])) if len(f) == 5 else ((f[1], f[2]),):
            v = self._num(val)
            if row == self.obj_row:
                if self.c[j] is not None:
                    raise MpsError(f"duplicate objective entry for {f[0]!r}")
                self.c[j] = v
                continue
            i = self._row(row)
            if i is None:
                continue  # a dropped free row
            if j in self.rows[i]:
                raise MpsError(f"duplicate entry ({row!r}, {f[0]!r})")
            if v:
                self.rows[i][j] = v

    def _set_and_pairs(self, f: list[str], section: str) -> list[tuple[str, str]]:
        if len(f) % 2 == 1:
            self._set(section, f[0])
            f = f[1:]
        else:
            self._set(section, "")
        if len(f) not in (2, 4):
            raise MpsError(f"bad {section} line {f!r}")
        return [(f[k], f[k + 1]) for k in range(0, len(f), 2)]

    def _rhs_line(self, f: list[str]) -> None:
        for row, val in self._set_and_pairs(f, "RHS"):
            v = self._num(val)
            if row == self.obj_row:
                self.c0 = -v
                continue
            i = self._row(row)
            if i is not None:
                if i in self.rhs:
                    raise MpsError(f"duplicate RHS for {row!r}")
                self.rhs[i] = v

    def _ranges_line(self, f: list[str]) -> None:
        for row, val in self._set_and_pairs(f, "RANGES"):
            if row == self.obj_row:
                raise MpsError("RANGES on the objective row")
            i = self._row(row)
            if i is not None:
                if i in self.ranges:
                    raise MpsError(f"duplicate RANGES for {row!r}")
                self.ranges[i] = self._num(val)

    def _bounds_line(self, f: list[str]) -> None:
        if not f:
            raise MpsError("empty BOUNDS line")
        typ = f[0].upper()
        if typ == "SC":
            raise MpsError("semi-continuous bounds (SC) are not supported")
        if typ in _NO_VALUE_BOUNDS:
            # BV may carry a value (ignored): "BV [set] col [value]"
            if typ == "BV" and len(f) == 3 and f[1] in self.col_index and _is_number(f[2]):
                f = f[:2]
            if len(f) == 2:
                set_name, col, val = "", f[1], None
            elif len(f) == 3 or (typ == "BV" and len(f) == 4 and _is_number(f[3])):
                set_name, col, val = f[1], f[2], None
            else:
                raise MpsError(f"bad BOUNDS line {f!r}")
        elif typ in _VALUE_BOUNDS:
            if len(f) == 4:
                set_name, col, val = f[1], f[2], self._num(f[3])
            elif len(f) == 3:
                set_name, col, val = "", f[1], self._num(f[2])
            else:
                raise MpsError(f"bad BOUNDS line {f!r}")
        else:
            raise MpsError(f"unknown bound type {f[0]!r}")
        self._set("BOUNDS", set_name)
        j = self._column(col, create=False)
        if j not in self.bounds_given:
            self.bounds_given.add(j)
            self._ensure_bounds()
            self.lb[j], self.ub[j] = _zero(), None
        if typ == "UP":
            self.ub[j] = val
        elif typ == "LO":
            self.lb[j] = val
        elif typ == "FX":
            self.lb[j] = self.ub[j] = val
        elif typ == "FR":
            self.lb[j] = self.ub[j] = None
        elif typ == "MI":
            self.lb[j] = None
        elif typ == "PL":
            self.ub[j] = None
        elif typ == "BV":
            self.integer[j] = True
            self.lb[j], self.ub[j] = _zero(), _one()
        elif typ == "LI":
            self.integer[j] = True
            self.lb[j] = val
        elif typ == "UI":
            self.integer[j] = True
            self.ub[j] = val

    def _ensure_bounds(self) -> None:
        while len(self.lb) < len(self.col_names):
            self.lb.append(_zero())
            self.ub.append(None)

    # -- assembly -----------------------------------------------------------------------------

    def _model(self) -> MipModel:
        self._ensure_bounds()
        for j in range(len(self.col_names)):
            if self.integer[j] and j not in self.bounds_given:
                self.ub[j] = _one()
        rl: list[Any] = []
        ru: list[Any] = []
        for i, typ in enumerate(self.row_type):
            b = self.rhs.get(i, _zero())
            r = self.ranges.get(i)
            if typ == "E":
                lo, hi = b, b
                if r is not None:
                    lo, hi = (b, b + r) if r >= 0 else (b + r, b)
            elif typ == "L":
                lo, hi = (None if r is None else b - abs(r)), b
            else:
                lo, hi = b, (None if r is None else b + abs(r))
            rl.append(lo)
            ru.append(hi)
        return MipModel(
            name=self.name,
            col_names=self.col_names,
            row_names=self.row_names,
            integer=self.integer,
            c=[v if v is not None else _zero() for v in self.c],
            c0=self.c0 if self.c0 is not None else _zero(),
            maximize=self.maximize,
            rows=self.rows,
            rl=rl,
            ru=ru,
            lb=self.lb,
            ub=self.ub,
        )


def _zero() -> Any:
    from certkkt.rational import ZERO

    return ZERO


def _one() -> Any:
    from certkkt.rational import ONE

    return ONE


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
