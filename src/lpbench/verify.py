"""Independent verification of a primal-dual pair against the canonical LP (DESIGN section 5).

Conventions: row duals ``y`` with ``z = c - A^T y`` (computed here, never taken from the
solver). At an optimum of the minimisation problem ``y_i > 0`` only if ``rl_i`` is finite,
``y_i < 0`` only if ``ru_i`` is finite, and likewise for ``z_j`` with ``l_j`` and ``u_j``.

All norms are infinity norms over finite entries. A pair passes at tolerance ``eps`` when

    r_P / b_scale <= eps,   r_D / c_scale <= eps,   |P - D| / (1 + |P| + |D|) <= eps.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from lpbench.lpmodel import CanonicalLP

#: Tolerances at which every unit is classified (DESIGN 5.2).
REPORT_EPS: tuple[float, ...] = (1e-4, 1e-6, 1e-8)
UNIT_ROUNDOFF = 2.0**-53


@dataclass(frozen=True)
class Verification:
    """Residuals, objectives and floors of one solution. Absolute and relative values."""

    primal_residual: float  # r_P
    dual_residual: float  # r_D
    primal_objective: float  # P = c^T x + c0 (canonical, minimisation)
    dual_objective: float  # D
    b_scale: float
    c_scale: float
    rel_primal: float  # r_P / b_scale
    rel_dual: float  # r_D / c_scale
    rel_gap: float  # |P - D| / (1 + |P| + |D|)
    floor_primal: float  # evaluation-error floor of the primal side, relative to b_scale
    floor_dual: float  # same for the dual side, relative to c_scale
    has_dual: bool

    @property
    def max_rel(self) -> float:
        """The smallest eps at which this pair passes (inf if no dual was available)."""
        return max(self.rel_primal, self.rel_dual, self.rel_gap)

    def passes(self, eps: float) -> bool:
        return (
            self.has_dual
            and self.rel_primal <= eps
            and self.rel_dual <= eps
            and self.rel_gap <= eps
        )

    def primal_feasible(self, eps: float) -> bool:
        return self.rel_primal <= eps

    def floor_limited(self, eps: float) -> bool:
        """True if floating-point evaluation error could exceed eps/10 (DESIGN 5.3)."""
        return max(self.floor_primal, self.floor_dual) > eps / 10.0

    def to_dict(self) -> dict[str, float | bool]:
        out: dict[str, float | bool] = {}
        for key, value in asdict(self).items():
            out[key] = value if isinstance(value, bool) else _json_float(value)
        return out


def _json_float(v: float) -> float:
    return float(v) if math.isfinite(v) else (math.inf if v > 0 else -math.inf)


def _finite_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def scales(lp: CanonicalLP) -> tuple[float, float]:
    """``b_scale = 1 + max finite |rl|,|ru|,|l|,|u|`` and ``c_scale = 1 + ||c||``."""
    parts = [lp.rl, lp.ru, lp.l, lp.u]
    bmax = 0.0
    for arr in parts:
        fin = np.abs(arr[np.isfinite(arr)])
        if fin.size:
            bmax = max(bmax, float(fin.max()))
    cmax = float(np.abs(lp.c).max()) if lp.c.size else 0.0
    return 1.0 + bmax, 1.0 + cmax


def primal_residual(lp: CanonicalLP, x: np.ndarray, ax: np.ndarray | None = None) -> float:
    """``r_P``: largest violation of a row or column bound (infinity norm)."""
    if ax is None:
        ax = lp.A @ x
    viol = 0.0
    for lo, val, hi in ((lp.rl, ax, lp.ru), (lp.l, x, lp.u)):
        fl = np.isfinite(lo)
        fh = np.isfinite(hi)
        if fl.any():
            viol = max(viol, _finite_max(np.maximum(lo[fl] - val[fl], 0.0)))
        if fh.any():
            viol = max(viol, _finite_max(np.maximum(val[fh] - hi[fh], 0.0)))
    return viol


def _dual_parts(mult: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Infeasibility and objective contribution of multipliers on ``lo <= . <= hi``."""
    pos = np.maximum(mult, 0.0)
    neg = np.maximum(-mult, 0.0)
    flo = np.isfinite(lo)
    fhi = np.isfinite(hi)
    infeas = max(_finite_max(pos[~flo]), _finite_max(neg[~fhi]))
    contrib = float(pos[flo] @ lo[flo]) - float(neg[fhi] @ hi[fhi])
    return infeas, contrib


def verify(lp: CanonicalLP, x: np.ndarray | None, y: np.ndarray | None) -> Verification | None:
    """Verify (x, y) against ``lp``. Returns None if there is no primal vector at all.

    A missing or non-finite dual gives ``has_dual = False`` and infinite dual quantities, so
    the pair never passes. A non-finite primal gives infinite residuals.
    """
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (lp.n,):
        raise ValueError(f"x has shape {x.shape}, expected ({lp.n},)")
    b_scale, c_scale = scales(lp)
    inf = math.inf
    absA = abs(lp.A)
    row_len = np.diff(lp.A.indptr).astype(np.float64)

    if not np.isfinite(x).all():
        r_p, p_obj, floor_p = inf, inf, inf
    else:
        ax = lp.A @ x
        r_p = primal_residual(lp, x, ax)
        p_obj = float(lp.c @ x) + lp.c0
        floor_p = _finite_max(row_len * UNIT_ROUNDOFF * (absA @ np.abs(x))) / b_scale

    has_dual = y is not None and np.shape(y) == (lp.m,) and bool(np.isfinite(y).all())
    if has_dual:
        y = np.asarray(y, dtype=np.float64)
        z = lp.c - lp.A.T @ y
        inf_y, d_y = _dual_parts(y, lp.rl, lp.ru)
        inf_z, d_z = _dual_parts(z, lp.l, lp.u)
        r_d = max(inf_y, inf_z)
        d_obj = lp.c0 + d_y + d_z
        col_len = np.diff(lp.A.tocsc().indptr).astype(np.float64)
        floor_d = (
            _finite_max((col_len + 1.0) * UNIT_ROUNDOFF * (np.abs(lp.c) + absA.T @ np.abs(y)))
            / c_scale
        )
    else:
        r_d, d_obj, floor_d = inf, -inf, 0.0

    if math.isfinite(p_obj) and math.isfinite(d_obj):
        rel_gap = abs(p_obj - d_obj) / (1.0 + abs(p_obj) + abs(d_obj))
    else:
        rel_gap = inf
    return Verification(
        primal_residual=r_p,
        dual_residual=r_d,
        primal_objective=p_obj,
        dual_objective=d_obj,
        b_scale=b_scale,
        c_scale=c_scale,
        rel_primal=r_p / b_scale,
        rel_dual=r_d / c_scale,
        rel_gap=rel_gap,
        floor_primal=floor_p,
        floor_dual=floor_d,
        has_dual=has_dual,
    )


def objective_agrees(p_obj: float, reference: float, eps: float) -> bool:
    """Reference agreement ``|P - P_ref| <= eps (1 + |P_ref|)`` (DESIGN 5.2)."""
    if not (math.isfinite(p_obj) and math.isfinite(reference)):
        return False
    return abs(p_obj - reference) <= eps * (1.0 + abs(reference))


def objective_rel_error(p_obj: float, reference: float) -> float:
    """``|P - P_ref| / (1 + |P_ref|)``."""
    if not (math.isfinite(p_obj) and math.isfinite(reference)):
        return math.inf
    return abs(p_obj - reference) / (1.0 + abs(reference))
