"""Hand-built and generated LPs with known primal-dual optima (for tests and adapter checks).

Generated problems are built backwards from the KKT conditions: choose x*, the active set,
y* and z* with the signs of DESIGN 5.1, then set the bounds so that x* is feasible and active
where intended, and ``c = A^T y* + z*``. Complementary slackness then makes (x*, y*) optimal.
All data are small integers, so every quantity is exact in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from lpbench.lpmodel import INF, CanonicalLP, make_lp


@dataclass(frozen=True)
class KnownLP:
    lp: CanonicalLP
    x: np.ndarray
    y: np.ndarray
    objective: float  # canonical (minimisation) objective of the optimum


def all_types_lp(maximize: bool = False) -> KnownLP:
    """Five columns (lower, upper, boxed, fixed, free) and four rows (<=, >=, =, ranged).

    Source problem (minimise, or maximise the negated objective when ``maximize``)::

        min 3x1 + x2 + x3 - 3x4 + 5x5 + 7
        s.t.  x1 + x2       <= 3
              x1 + x3       >= 1
              x2 + x5        = 1
              1 <= x3 - x5  <= 3
              x1 >= 0, x2 <= 3, 1 <= x3 <= 4, x4 = 2, x5 free

    Optimum x = (0, 3, 1, 2, -2), objective -5, y = (-1, 2, 3, -2), z = (2, -1, 1, -3, 0).
    """
    A = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, -1],
        ],
        dtype=float,
    )
    c = np.array([3.0, 1.0, 1.0, -3.0, 5.0])
    c0 = 7.0
    sign = -1.0 if maximize else 1.0
    lp = make_lp(
        "all_types_max" if maximize else "all_types",
        A,
        c=sign * c,
        rl=[-INF, 1, 1, 1],
        ru=[3, INF, 1, 3],
        l=[0, -INF, 1, 2, -INF],
        u=[INF, 3, 4, 2, INF],
        c0=sign * c0,
        maximize=maximize,
    )
    return KnownLP(
        lp=lp,
        x=np.array([0.0, 3.0, 1.0, 2.0, -2.0]),
        y=np.array([-1.0, 2.0, 3.0, -2.0]),
        objective=-5.0,
    )


def with_free_row(known: KnownLP) -> KnownLP:
    """Append the free row ``x1 + ... + x5`` (it constrains nothing; its dual is 0)."""
    lp = known.lp
    A = sp.vstack([lp.A, sp.csr_matrix(np.ones((1, lp.n)))]).tocsr()
    new = CanonicalLP(
        name=lp.name + "_freerow",
        A=A,
        c=lp.c,
        c0=lp.c0,
        rl=np.append(lp.rl, -INF),
        ru=np.append(lp.ru, INF),
        l=lp.l,
        u=lp.u,
        maximize_flipped=lp.maximize_flipped,
    )
    return KnownLP(new, known.x, np.append(known.y, 0.0), known.objective)


def infeasible_lp() -> CanonicalLP:
    """``x1 + x2 <= 1,  x1 + x2 >= 2,  x >= 0``."""
    return make_lp(
        "infeasible", [[1, 1], [1, 1]], c=[1, 1], rl=[-INF, 2], ru=[1, INF], l=[0, 0], u=[INF, INF]
    )


def unbounded_lp() -> CanonicalLP:
    """``min -x1  s.t.  x1 - x2 <= 1,  x >= 0`` (ray x1 = x2 -> inf)."""
    return make_lp("unbounded", [[1, -1]], c=[-1, 0], rl=[-INF], ru=[1], l=[0, 0], u=[INF, INF])


def random_known_lp(m: int, n: int, seed: int, density: float = 0.3) -> KnownLP:
    """A random feasible, bounded LP with a known optimum, built from the KKT conditions."""
    rng = np.random.default_rng(seed)
    A = np.zeros((m, n))
    mask = rng.random((m, n)) < density
    A[mask] = rng.integers(-5, 6, size=int(mask.sum()))
    # make sure every row and column has at least one nonzero
    for i in range(m):
        if not A[i].any():
            A[i, rng.integers(n)] = rng.choice([-1.0, 1.0, 2.0])
    for j in range(n):
        if not A[:, j].any():
            A[rng.integers(m), j] = rng.choice([-1.0, 1.0, 3.0])

    x = rng.integers(-5, 6, size=n).astype(float)
    lo = np.full(n, -INF)
    up = np.full(n, INF)
    z = np.zeros(n)
    for j in range(n):
        kind = rng.choice(["at_lower", "at_upper", "basic_boxed", "fixed", "free", "basic_lower"])
        if kind == "at_lower":
            lo[j] = x[j]
            up[j] = x[j] + rng.integers(1, 6) if rng.random() < 0.5 else INF
            z[j] = rng.integers(1, 4)
        elif kind == "at_upper":
            up[j] = x[j]
            lo[j] = x[j] - rng.integers(1, 6) if rng.random() < 0.5 else -INF
            z[j] = -rng.integers(1, 4)
        elif kind == "basic_boxed":
            lo[j] = x[j] - rng.integers(1, 4)
            up[j] = x[j] + rng.integers(1, 4)
        elif kind == "fixed":
            lo[j] = up[j] = x[j]
            z[j] = rng.integers(-3, 4)
        elif kind == "basic_lower":
            lo[j] = x[j] - rng.integers(1, 4)
        # free: no bounds, z = 0

    ax = A @ x
    rl = np.full(m, -INF)
    ru = np.full(m, INF)
    y = np.zeros(m)
    for i in range(m):
        kind = rng.choice(["le_active", "ge_active", "eq", "range_lo", "range_hi", "slack"])
        if kind == "le_active":
            ru[i] = ax[i]
            y[i] = -rng.integers(1, 4)
        elif kind == "ge_active":
            rl[i] = ax[i]
            y[i] = rng.integers(1, 4)
        elif kind == "eq":
            rl[i] = ru[i] = ax[i]
            y[i] = rng.integers(-3, 4)
        elif kind == "range_lo":
            rl[i] = ax[i]
            ru[i] = ax[i] + rng.integers(1, 6)
            y[i] = rng.integers(1, 4)
        elif kind == "range_hi":
            ru[i] = ax[i]
            rl[i] = ax[i] - rng.integers(1, 6)
            y[i] = -rng.integers(1, 4)
        else:  # inactive inequality
            if rng.random() < 0.5:
                ru[i] = ax[i] + rng.integers(1, 6)
            else:
                rl[i] = ax[i] - rng.integers(1, 6)
    c = A.T @ y + z
    c0 = float(rng.integers(-10, 11))
    lp = make_lp(f"random_{m}x{n}_s{seed}", A, c=c, rl=rl, ru=ru, l=lo, u=up, c0=c0)
    return KnownLP(lp=lp, x=x, y=y, objective=float(c @ x + c0))
