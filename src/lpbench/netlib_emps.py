"""Expander for netlib's compressed LP format (Gay, 1985), with checksum verification.

This is an independent implementation written from the format's description and the
observable behaviour of netlib's ``emps`` program; no code from ``emps.c`` is included.

Format summary:

* Lines before the ``NAME`` line are ignored. The ``NAME`` line and the two statistics lines
  that follow it are not checksummed.
* The statistics give ``nrow ncol colmax nz nrhs rhsnz nran ranz`` and ``nbd bdnz ns``.
* Every later line contributes one checksum character; after every 71 such lines, and at the
  end of the problem, a line consisting of a blank followed by the checksum characters of
  the preceding lines must appear. Lines starting with ``:`` ("mystery lines") are
  checksummed and otherwise skipped.
* Characters are digits in base 92 (the printable ASCII characters except blank, ``:`` and
  ``\\``). Indices are "supersparse": a character with value < 23 starts a multi-character
  base-46 number terminated by a character with value >= 46; values 23..45 encode 0..22.
* A number token starting with a value >= 46 is a literal (integer or decimal mantissa and
  exponent); a value < 46 is a reference into the number table of ``ns`` entries.
* After the number table come ``nrow`` row lines (type character then name), then the
  COLUMNS, RHS, RANGES and BOUNDS sections as streams of (index, number) tokens; index 0
  means that the rest of the line holds a new column (or set) name.

The output is fixed-format MPS in the layout emps produces.
"""

from __future__ import annotations

from dataclasses import dataclass

TRTAB = (
    "!\"#$%&'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)
_INV = {ch: i for i, ch in enumerate(TRTAB)}
_BOUND_TYPES = ("UP", "LO", "FX", "FR", "MI", "PL")


class NetlibFormatError(ValueError):
    """The compressed file is malformed or fails a checksum."""


def tr(ch: str) -> int:
    """Value of one format character (92 for characters outside the table)."""
    return _INV.get(ch, 92)


def line_checksum(line: str) -> str:
    """Checksum character of one data line (without its newline)."""
    x = 0
    for ch in line:
        c = tr(ch)
        x = (x >> 1) + c + 16384 if x & 1 else (x >> 1) + c
    return TRTAB[x % 92]


@dataclass
class Stats:
    nrow: int
    ncol: int
    colmax: int
    nz: int
    nrhs: int
    rhsnz: int
    nran: int
    ranz: int
    nbd: int
    bdnz: int
    ns: int


@dataclass
class ExpandResult:
    name: str
    mps: str
    stats: Stats
    checked_lines: int
    checksum_lines: int


class _Reader:
    """Line source with checksum accounting."""

    def __init__(self, text: str) -> None:
        self._lines = text.split("\n")
        if self._lines and self._lines[-1] == "":
            self._lines.pop()
        self._pos = 0
        self.lineno = 0
        self.pending: list[str] = []  # checksum characters awaiting a checksum line
        self.checking = False
        self.checked_lines = 0
        self.checksum_lines = 0

    def _raw(self) -> str | None:
        if self._pos >= len(self._lines):
            return None
        line = self._lines[self._pos].rstrip("\r")
        self._pos += 1
        self.lineno += 1
        if len(line) > 75:
            raise NetlibFormatError(f"line {self.lineno} longer than 75 characters")
        return line

    def _verify_checksum_line(self) -> None:
        while True:
            got = self._raw()
            if got is None:
                raise NetlibFormatError("premature end of file (checksum line expected)")
            expected = " " + "".join(self.pending)
            if got == expected:
                self.pending = []
                self.checksum_lines += 1
                return
            if got.startswith(":") and len(self.pending) < 71:
                self.pending.append(line_checksum(got))  # mystery line inside the block
                continue
            raise NetlibFormatError(
                f"checksum error at line {self.lineno}: expected {expected!r}, got {got!r}"
            )

    def next(self, allow_eof: bool = False) -> str | None:
        while True:
            line = self._raw()
            if line is None:
                if allow_eof:
                    return None
                raise NetlibFormatError("premature end of file")
            if self.checking:
                self.pending.append(line_checksum(line))
                self.checked_lines += 1
                if len(self.pending) >= 71:
                    self._verify_checksum_line()
            if line.startswith(":"):
                continue
            return line

    def finish_block(self) -> None:
        if self.pending:
            self._verify_checksum_line()


class _Tokens:
    """Token stream over successive data lines."""

    def __init__(self, reader: _Reader) -> None:
        self.r = reader
        self.line = ""
        self.pos = 0

    def ensure(self) -> None:
        if self.pos >= len(self.line):
            nxt = self.r.next()
            assert nxt is not None
            self.line, self.pos = nxt, 0

    def _ch(self) -> int:
        if self.pos >= len(self.line):
            raise NetlibFormatError(f"token runs past end of line {self.r.lineno}")
        c = tr(self.line[self.pos])
        self.pos += 1
        return c

    def index(self) -> int:
        k = self._ch()
        if k >= 46:
            raise NetlibFormatError(f"bad index character at line {self.r.lineno}")
        if k >= 23:
            return k - 23
        x = k
        while True:
            k = self._ch()
            x = x * 46 + k
            if k >= 46:
                return x - 46

    def rest_of_line(self) -> str:
        s = self.line[self.pos :]
        self.pos = len(self.line)
        return s

    def number(self, table: list[str]) -> str:
        """Next number as emps prints it (right-aligned in 12 characters)."""
        if self.pos >= len(self.line):
            raise NetlibFormatError(f"number expected at line {self.r.lineno}")
        k = tr(self.line[self.pos])
        if k < 46:
            idx = self.index()
            if idx < 1 or idx > len(table):
                raise NetlibFormatError(f"number table index {idx} out of range")
            return table[idx - 1]
        self.pos += 1
        return format_literal(k, self._ch)


def format_literal(k: int, nextc: object) -> str:
    """Decode a literal number whose first character has value ``k`` (>= 46).

    ``nextc`` returns the value of the next character. The result follows emps's layout:
    integers end with ``.``, small exponents are written positionally, others with ``E``.
    """
    get = nextc  # type: ignore[assignment]
    k -= 46
    sign = ""
    if k >= 23:
        sign = "-"
        k -= 23
        nelim = 11
    else:
        nelim = 12
    if k >= 11:  # integer
        k -= 11
        if k >= 6:
            x = k - 6
        else:
            x = k
            while True:
                k = get()  # type: ignore[operator]
                x = x * 46 + k
                if k >= 46:
                    x -= 46
                    break
        body = f"{x}."
        out = sign + body
        return out.rjust(12)

    # general floating point: exponent character, then k+1 base-92 digits
    ex = get() - 50  # type: ignore[operator]
    x = get()  # type: ignore[operator]
    y = 0
    for _ in range(k):
        if x >= 100000000:
            y = x
            x = get()  # type: ignore[operator]
        else:
            x = x * 92 + get()  # type: ignore[operator]
    # digits, least significant first (mirrors the reference program's digit order)
    d: list[str] = []
    if y:
        while x > 1:
            d.append(str(x % 10))
            x //= 10
        while True:
            d.append(str(y % 10))
            if y < 10:
                break
            y //= 10
    elif x:
        while True:
            d.append(str(x % 10))
            if x < 10:
                break
            x //= 10
    else:
        d.append("0")
    s: list[str] = [sign] if sign else []
    nd = len(d) + ex
    use_e = False
    if ex > 0:
        if nd < nelim or ex < 3:
            s.extend(reversed(d))
            s.append("0" * ex)
            s.append(".")
        else:
            use_e = True
    elif nd >= 0:
        digits = list(reversed(d))
        s.extend(digits[:nd])
        s.append(".")
        s.extend(digits[nd:])
    elif ex > -nelim:
        s.append(".")
        s.append("0" * (-nd))
        s.extend(reversed(d))
    else:
        use_e = True
    if use_e:
        digits = list(reversed(d))  # most significant first
        ex += len(d) - 1
        pos = 0
        if ex == -10:
            ex = -9
        else:
            if 9 < ex <= len(d) + 8:
                while True:
                    s.append(digits[pos])
                    pos += 1
                    ex -= 1
                    if ex <= 9:
                        break
            s.append(digits[pos])
            pos += 1
        s.append(".")
        s.extend(digits[pos:])
        s.append("E")
        if ex < 0:
            s.append("-")
            ex = -ex
        if ex:
            s.append(str(ex))
    return "".join(s).rjust(12)


def _pad(s: str, width: int) -> str:
    """printf("%-W.Ws")."""
    return s[:width].ljust(width)


def expand(text: str) -> ExpandResult:
    """Expand the first (and only) problem of a compressed netlib file to MPS text."""
    r = _Reader(text)
    out: list[str] = []
    line = r.next()
    while line is not None and not line.startswith("NAME"):
        line = r.next(allow_eof=True)
    if line is None:
        raise NetlibFormatError("no NAME line")
    name_line = line
    rest = name_line[4:].strip()
    if not rest:
        raise NetlibFormatError("blank NAME line")
    name = rest.split()[0]
    out.append(name_line)

    s1 = r.next()
    s2 = r.next()
    assert s1 is not None and s2 is not None
    try:
        a = [int(t) for t in s1.split()]
        b = [int(t) for t in s2.split()]
        if len(a) < 8 or len(b) < 3:
            raise ValueError
    except ValueError as exc:
        raise NetlibFormatError(f"bad statistics lines: {s1!r} / {s2!r}") from exc
    st = Stats(*a[:8], *b[:3])
    r.checking = True

    tok = _Tokens(r)
    table: list[str] = []
    for _ in range(st.ns):
        tok.ensure()
        table.append(tok.number(table))

    names: list[str] = [""] * (st.nrow + st.ncol + 1)  # 1-based: rows, then columns
    for i in range(1, st.nrow + 1):
        row = r.next()
        assert row is not None
        if i == 1:
            out.append("ROWS")
        out.append(f" {row[0] if row else ' '}  {row[1:]}")
        names[i] = row[1:9]
    tok.line, tok.pos = "", 0

    col_counter = [st.nrow]

    def section(head: str, count: int, what: int) -> None:
        if count == 0:
            if what <= 2:
                out.append(head)
            return
        first = True
        k = 0
        curcol = ""
        rc1 = ""
        rown0 = ""
        remaining = count
        while remaining:
            remaining -= 1
            tok.ensure()
            if first:
                out.append(head)
                first = False
            while True:
                n = tok.index()
                if n != 0:
                    break
                if k:
                    out.append(f"    {_pad(curcol, 8)}  {_pad(rown0, 8)}  {rc1[:15]}")
                    k = 0
                curcol = tok.rest_of_line()[:8]
                if what == 1:
                    col_counter[0] += 1
                    if col_counter[0] >= len(names):
                        raise NetlibFormatError("more columns than the statistics line says")
                    names[col_counter[0]] = curcol
                nxt = r.next()
                assert nxt is not None
                tok.line, tok.pos = nxt, 0
            if what >= 4:
                if n >= 7:
                    raise NetlibFormatError(f"bad bound type index {n}")
                tok.ensure()
                j = st.nrow + tok.index()
                if j <= st.nrow or j >= len(names):
                    raise NetlibFormatError(f"bad column index in BOUNDS: {j - st.nrow}")
                colname = names[j]
                btype = _BOUND_TYPES[n - 1]
                if n >= 4:
                    out.append(f" {btype} {_pad(curcol, 8)}  {colname[:8]}")
                    continue
                tok.ensure()
                val = tok.number(table)
                out.append(f" {btype} {_pad(curcol, 8)}  {_pad(colname, 8)}  {val[:15]}")
                continue
            if n < 1 or n > st.nrow:
                raise NetlibFormatError(f"bad row index {n} in {head}")
            rname = names[n]
            tok.ensure()
            val = tok.number(table)
            if k == 0:
                rown0, rc1 = rname, val
                k = 1
                continue
            out.append(
                f"    {_pad(curcol, 8)}  {_pad(rown0, 8)}  {_pad(rc1, 15)}"
                f"{_pad(rname, 8)}  {val[:15]}"
            )
            k = 0
        if k:
            out.append(f"    {_pad(curcol, 8)}  {_pad(rown0, 8)}  {rc1[:15]}")

    section("COLUMNS", st.nz, 1)
    section("RHS", st.rhsnz, 2)
    section("RANGES", st.ranz, 3)
    section("BOUNDS", st.bdnz, 4)
    r.finish_block()
    r.checking = False
    out.append("ENDATA")
    nxt = r.next(allow_eof=True)
    while nxt is not None:
        if nxt.startswith("NAME"):
            raise NetlibFormatError("file contains more than one problem")
        nxt = r.next(allow_eof=True)
    if col_counter[0] != st.nrow + st.ncol:
        raise NetlibFormatError(
            f"read {col_counter[0] - st.nrow} columns, statistics say {st.ncol}"
        )
    return ExpandResult(
        name=name,
        mps="\n".join(out) + "\n",
        stats=st,
        checked_lines=r.checked_lines,
        checksum_lines=r.checksum_lines,
    )
