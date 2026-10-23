"""Canonical LP form, extraction from HiGHS, permutations and an exact MPS writer.

The canonical problem (docs/DESIGN.md, section 4.1) is

    minimise  c^T x + c0   subject to   rl <= A x <= ru,   l <= x <= u,

with infinite bounds allowed. A maximisation source is converted by negating ``c`` and ``c0``
(``maximize_flipped`` records it). Rows that are free on both sides constrain nothing and are
dropped on extraction (their dual is zero at every solution); the number dropped is reported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

INF = math.inf


@dataclass(frozen=True)
class CanonicalLP:
    """``min c^T x + c0  s.t.  rl <= A x <= ru,  l <= x <= u`` (float64, +-inf allowed)."""

    name: str
    A: sp.csr_matrix
    c: np.ndarray
    c0: float
    rl: np.ndarray
    ru: np.ndarray
    l: np.ndarray  # noqa: E741 - the design's notation
    u: np.ndarray
    maximize_flipped: bool = False
    info: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        m, n = self.A.shape
        for label, arr, size in (
            ("c", self.c, n),
            ("l", self.l, n),
            ("u", self.u, n),
            ("rl", self.rl, m),
            ("ru", self.ru, m),
        ):
            if arr.shape != (size,):
                raise ValueError(f"{self.name}: {label} has shape {arr.shape}, expected ({size},)")
        if np.isnan(self.c).any() or not np.isfinite(self.c).all():
            raise ValueError(f"{self.name}: non-finite objective coefficient")
        if not np.isfinite(self.A.data).all():
            raise ValueError(f"{self.name}: non-finite matrix entry")
        for lo, hi, label in ((self.rl, self.ru, "row"), (self.l, self.u, "column")):
            if np.isnan(lo).any() or np.isnan(hi).any():
                raise ValueError(f"{self.name}: NaN {label} bound")
            if (lo == INF).any() or (hi == -INF).any():
                raise ValueError(f"{self.name}: {label} bound infinite on the wrong side")

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.A.nnz)

    def objective(self, x: np.ndarray) -> float:
        """Canonical (minimisation) objective ``c^T x + c0``."""
        return float(self.c @ x + self.c0)

    def source_objective(self, canonical_value: float) -> float:
        """Convert a canonical objective value back to the source's sense (display only)."""
        return -canonical_value if self.maximize_flipped else canonical_value


def make_lp(
    name: str,
    A: sp.spmatrix | np.ndarray,
    c: Iterable[float],
    rl: Iterable[float],
    ru: Iterable[float],
    l: Iterable[float],  # noqa: E741
    u: Iterable[float],
    c0: float = 0.0,
    maximize: bool = False,
) -> CanonicalLP:
    """Build a CanonicalLP from dense or sparse data. ``maximize`` gives a max source."""
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    A_csr.eliminate_zeros()
    A_csr.sort_indices()
    c_arr = np.asarray(list(c), dtype=np.float64)
    if maximize:
        c_arr = -c_arr
        c0 = -c0
    return CanonicalLP(
        name=name,
        A=A_csr,
        c=c_arr,
        c0=float(c0),
        rl=np.asarray(list(rl), dtype=np.float64),
        ru=np.asarray(list(ru), dtype=np.float64),
        l=np.asarray(list(l), dtype=np.float64),
        u=np.asarray(list(u), dtype=np.float64),
        maximize_flipped=maximize,
    )


# --------------------------------------------------------------------------------------------
# HiGHS extraction
# --------------------------------------------------------------------------------------------


def _inf_clean(arr: np.ndarray, threshold: float = 1e20) -> np.ndarray:
    """Map HiGHS infinities (and anything at or beyond 1e20) to +-inf."""
    out = np.asarray(arr, dtype=np.float64).copy()
    out[out >= threshold] = INF
    out[out <= -threshold] = -INF
    return out


def from_highs_lp(lp: object, name: str) -> CanonicalLP:
    """Extract a CanonicalLP from a ``highspy.HighsLp``; integrality is dropped."""
    import highspy

    m = int(lp.num_row_)  # type: ignore[attr-defined]
    n = int(lp.num_col_)  # type: ignore[attr-defined]
    mat = lp.a_matrix_  # type: ignore[attr-defined]
    start = np.asarray(mat.start_, dtype=np.int64)
    index = np.asarray(mat.index_, dtype=np.int64)
    value = np.asarray(mat.value_, dtype=np.float64)
    if mat.format_ == highspy.MatrixFormat.kColwise:
        A = sp.csc_matrix((value, index, start), shape=(m, n)).tocsr()
    else:
        A = sp.csr_matrix((value, index, start), shape=(m, n))
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    c = np.asarray(lp.col_cost_, dtype=np.float64).copy()  # type: ignore[attr-defined]
    c0 = float(lp.offset_)  # type: ignore[attr-defined]
    maximize = lp.sense_ == highspy.ObjSense.kMaximize  # type: ignore[attr-defined]
    if maximize:
        c = -c
        c0 = -c0
    rl = _inf_clean(lp.row_lower_)  # type: ignore[attr-defined]
    ru = _inf_clean(lp.row_upper_)  # type: ignore[attr-defined]
    lo = _inf_clean(lp.col_lower_)  # type: ignore[attr-defined]
    up = _inf_clean(lp.col_upper_)  # type: ignore[attr-defined]
    integrality = list(lp.integrality_)  # type: ignore[attr-defined]
    n_int = sum(1 for t in integrality if int(t) != 0)
    free_rows = np.flatnonzero((rl == -INF) & (ru == INF))
    if free_rows.size:
        keep = np.setdiff1d(np.arange(m), free_rows)
        A = A[keep]
        rl = rl[keep]
        ru = ru[keep]
    return CanonicalLP(
        name=name,
        A=sp.csr_matrix(A),
        c=c,
        c0=c0,
        rl=rl,
        ru=ru,
        l=lo,
        u=up,
        maximize_flipped=bool(maximize),
        info={"dropped_free_rows": int(free_rows.size), "relaxed_integer_columns": n_int},
    )


def read_with_highs(path: Path, name: str | None = None, fixed_format: bool = False) -> CanonicalLP:
    """Read an LP/MPS file with HiGHS and return its CanonicalLP (LP relaxation)."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if fixed_format:
        h.setOptionValue("mps_parser_type_free", False)
    status = h.readModel(str(path))
    if status == highspy.HighsStatus.kError:
        raise ValueError(f"HiGHS could not read {Path(path).name}")
    return from_highs_lp(h.getLp(), name or Path(path).stem)


# --------------------------------------------------------------------------------------------
# Permutations
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Permutation:
    """Row and column orders: new row i is old row ``rows[i]``; new column j is ``cols[j]``."""

    rows: np.ndarray
    cols: np.ndarray

    @property
    def is_identity(self) -> bool:
        return bool(
            np.array_equal(self.rows, np.arange(self.rows.size))
            and np.array_equal(self.cols, np.arange(self.cols.size))
        )


def make_permutation(m: int, n: int, k: int, seed: int, name: str = "") -> Permutation:
    """Permutation index k of an m x n problem. k = 0 is the identity (original order).

    The draw depends only on (seed, instance name, k), so it is reproducible and does not
    depend on the order in which instances are prepared.
    """
    if k < 0:
        raise ValueError("permutation index must be >= 0")
    if k == 0:
        return Permutation(np.arange(m), np.arange(n))
    name_key = int.from_bytes(name.encode("utf-8")[:16].ljust(16, b"\0"), "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, name_key, k]))
    return Permutation(rng.permutation(m), rng.permutation(n))


def permute(lp: CanonicalLP, perm: Permutation) -> CanonicalLP:
    """The same problem with rows and columns reordered."""
    r, q = perm.rows, perm.cols
    if r.size != lp.m or q.size != lp.n:
        raise ValueError("permutation size does not match the problem")
    A = lp.A[r][:, q].tocsr()
    A.sort_indices()
    return CanonicalLP(
        name=lp.name,
        A=A,
        c=lp.c[q].copy(),
        c0=lp.c0,
        rl=lp.rl[r].copy(),
        ru=lp.ru[r].copy(),
        l=lp.l[q].copy(),
        u=lp.u[q].copy(),
        maximize_flipped=lp.maximize_flipped,
        info=dict(lp.info),
    )


def unpermute_primal(x_perm: np.ndarray, perm: Permutation) -> np.ndarray:
    """Map a primal vector of the permuted problem back to the original column order."""
    x = np.empty_like(x_perm)
    x[perm.cols] = x_perm
    return x


def unpermute_dual(y_perm: np.ndarray, perm: Permutation) -> np.ndarray:
    """Map a row-dual vector of the permuted problem back to the original row order."""
    y = np.empty_like(y_perm)
    y[perm.rows] = y_perm
    return y


# --------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------


def save_npz(lp: CanonicalLP, path: Path) -> None:
    """Store a CanonicalLP losslessly (atomic write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    A = lp.A.tocsr()
    np.savez_compressed(
        tmp,
        name=np.array(lp.name),
        data=A.data,
        indices=A.indices,
        indptr=A.indptr,
        shape=np.array(A.shape),
        c=lp.c,
        c0=np.array(lp.c0),
        rl=lp.rl,
        ru=lp.ru,
        l=lp.l,
        u=lp.u,
        maximize_flipped=np.array(lp.maximize_flipped),
    )
    tmp.replace(path)


def load_npz(path: Path) -> CanonicalLP:
    with np.load(path, allow_pickle=False) as z:
        A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return CanonicalLP(
            name=str(z["name"]),
            A=A,
            c=z["c"].copy(),
            c0=float(z["c0"]),
            rl=z["rl"].copy(),
            ru=z["ru"].copy(),
            l=z["l"].copy(),
            u=z["u"].copy(),
            maximize_flipped=bool(z["maximize_flipped"]),
        )


# --------------------------------------------------------------------------------------------
# MPS output
# --------------------------------------------------------------------------------------------


def _num(v: float) -> str:
    """Shortest decimal string that round-trips to the same float64."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


def row_name(i: int) -> str:
    return f"R{i}"


def col_name(j: int) -> str:
    return f"C{j}"


def write_mps(lp: CanonicalLP, path: Path) -> None:
    """Write the problem as free-format MPS with exact (round-trip) numbers.

    Always a minimisation problem. The objective constant ``c0`` is not written: solver
    objectives are compared after adding ``c0`` back, which avoids the reader-dependent sign
    of an objective-row RHS. Rows are named R0.., columns C0.. in canonical order, so
    adapters can map solutions by name. Free rows cannot be represented and are rejected.
    """
    rl, ru, lo, up = lp.rl, lp.ru, lp.l, lp.u
    if ((rl == -INF) & (ru == INF)).any():
        raise ValueError(f"{lp.name}: free rows cannot be written to MPS (drop them first)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"NAME {lp.name.replace(' ', '_') or 'LP'}", "ROWS", " N OBJ"]
    ranges: list[str] = []
    rhs: list[str] = []
    for i in range(lp.m):
        lo_i, hi_i = rl[i], ru[i]
        if lo_i == hi_i:
            lines.append(f" E {row_name(i)}")
            if lo_i != 0.0:
                rhs.append(f"    RHS {row_name(i)} {_num(lo_i)}")
        elif lo_i == -INF:
            lines.append(f" L {row_name(i)}")
            if hi_i != 0.0:
                rhs.append(f"    RHS {row_name(i)} {_num(hi_i)}")
        elif hi_i == INF:
            lines.append(f" G {row_name(i)}")
            if lo_i != 0.0:
                rhs.append(f"    RHS {row_name(i)} {_num(lo_i)}")
        else:  # ranged: G row at rl with range ru - rl
            lines.append(f" G {row_name(i)}")
            if lo_i != 0.0:
                rhs.append(f"    RHS {row_name(i)} {_num(lo_i)}")
            ranges.append(f"    RNG {row_name(i)} {_num(hi_i - lo_i)}")
    lines.append("COLUMNS")
    A = lp.A.tocsc()
    A.sort_indices()
    indptr, indices, data = A.indptr, A.indices, A.data
    for j in range(lp.n):
        cj = col_name(j)
        wrote = False
        if lp.c[j] != 0.0:
            lines.append(f"    {cj} OBJ {_num(lp.c[j])}")
            wrote = True
        for p in range(indptr[j], indptr[j + 1]):
            lines.append(f"    {cj} {row_name(int(indices[p]))} {_num(data[p])}")
            wrote = True
        if not wrote:
            lines.append(f"    {cj} OBJ 0")
    lines.append("RHS")
    lines.extend(rhs)
    if ranges:
        lines.append("RANGES")
        lines.extend(ranges)
    bounds: list[str] = []
    for j in range(lp.n):
        cj = col_name(j)
        lj, uj = lo[j], up[j]
        if lj == uj:
            bounds.append(f" FX BND {cj} {_num(lj)}")
        elif lj == -INF and uj == INF:
            bounds.append(f" FR BND {cj}")
        elif lj == -INF:
            bounds.append(f" MI BND {cj}")
            bounds.append(f" UP BND {cj} {_num(uj)}")
        else:
            if lj != 0.0:
                bounds.append(f" LO BND {cj} {_num(lj)}")
            if uj != INF:
                if uj < 0.0 and lj == 0.0:
                    bounds.append(f" LO BND {cj} 0")
                bounds.append(f" UP BND {cj} {_num(uj)}")
    if bounds:
        lines.append("BOUNDS")
        lines.extend(bounds)
    lines.append("ENDATA")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    tmp.replace(path)
