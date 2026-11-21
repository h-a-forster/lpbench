"""Witnesses, refutations of dual claims, the estimands E1-E4 and Wilson intervals
(DESIGN 13.5-13.6).

Everything here is in the minimisation sense: a maximisation instance's claims are negated
on read. The witness ``W`` of an instance is the least exact objective over all feasible
completions of all its units. A dual claim ``D`` (a claimed lower bound on the optimum; an
infeasibility claim is ``D = +inf``) is refuted when ``D > W``, with severity
``s = (D - W) / max(1, |W|)`` binned at the thresholds in ``SEVERITY_BINS``.

A refutation is evidence that a claim is wrong for the exact data. It is not called a bug
until the bug-claim protocol of DESIGN 6.4 item 5 is complete.
"""

from __future__ import annotations

import math
from typing import Any

from lpbench.mip.complete import q_parse

#: Severity thresholds: a refutation is in bin ``t`` when ``s > t``.
SEVERITY_BINS: tuple[float, ...] = (0.0, 1e-9, 1e-6, 1e-4)
BIN_LABELS = tuple(f">{t:g}" for t in SEVERITY_BINS)
#: The event of E4: a refutation above this severity.
E4_THRESHOLD = 1e-9
Z95 = 1.959963984540054  # standard normal quantile for 95%


def min_sense(value: float | None, maximize: bool) -> float | None:
    if value is None:
        return None
    return -value if maximize else value


def witnesses(checks: list[dict[str, Any]], key: str = "completion") -> dict[str, Any]:
    """Least exact feasible completion objective per instance (rationals, minimisation)."""
    out: dict[str, Any] = {}
    for c in checks:
        comp = c.get(key) or {}
        if comp.get("outcome") != "feasible":
            continue
        z = q_parse(comp["z"])
        inst = c["instance"]
        if inst not in out or z < out[inst]:
            out[inst] = z
    return out


def dual_claim(rec: dict[str, Any], maximize: bool) -> Any:
    """The claimed lower bound (minimisation sense) as a rational, +inf for an
    infeasibility claim, None when the solver made no dual claim."""
    from certkkt.rational import to_q

    if rec.get("status") == "INFEASIBLE":
        return math.inf
    d = min_sense(rec.get("dual_bound"), maximize)
    return None if d is None else to_q(float(d))


def refutation(d: Any, w: Any) -> dict[str, Any] | None:
    """Severity and bins of ``D`` against witness ``W``; None when either is missing."""
    from certkkt.rational import ONE, q_float_down, to_q

    if d is None or w is None:
        return None
    if d == math.inf:
        return {"severity": math.inf, "bins": dict.fromkeys(BIN_LABELS, True)}
    s = (d - w) / max(ONE, abs(w))
    return {
        "severity": q_float_down(s) if s > 0 else 0.0,
        "bins": {lab: bool(s > to_q(t)) for lab, t in zip(BIN_LABELS, SEVERITY_BINS, strict=True)},
    }


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for k events in n trials; zero events give [0, z^2/(n+z^2)]."""
    if n <= 0:
        return None
    if k == 0:
        return 0.0, z * z / (n + z * z)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


ESTIMANDS = {
    "E1": "returned solution not exactly feasible (class P1-P3), per unit with a solution",
    "E2": "returned solution beyond the solver's own tolerances (P2-P3), per unit with a solution",
    "E3": "rounded solution has no exactly feasible completion, per unit with a solution",
    "E4": f"dual claim refuted with severity above {E4_THRESHOLD:g}, per unit with a dual claim",
}


def _event(claim: dict[str, Any], e: str) -> bool | None:
    """True/False for the unit's event, None when unresolved; raises KeyError if the unit
    is outside the estimand's denominator."""
    if e in ("E1", "E2", "E3"):
        if not claim.get("has_solution"):
            raise KeyError(e)
        if e == "E1":
            pc = claim.get("primal_class")
            return None if pc is None else pc != "P0"
        if e == "E2":
            pc = claim.get("primal_class")
            return None if pc is None else pc in ("P2", "P3")
        comp = claim.get("completion")
        return None if comp == "unknown" else comp == "infeasible"
    if claim.get("dual_claim") is None:
        raise KeyError(e)
    bins = claim.get("refuted_bins")
    if bins is None:
        return None
    return bool(bins[f">{E4_THRESHOLD:g}"])


def estimate(claims: list[dict[str, Any]], e: str) -> dict[str, Any]:
    """Counts and a Wilson interval; unresolved units are counted and bound the rate."""
    k = n = u = 0
    for c in claims:
        try:
            ev = _event(c, e)
        except KeyError:
            continue
        if ev is None:
            u += 1
        else:
            n += 1
            k += int(ev)
    ci = wilson(k, n)
    worst = wilson(k + u, n + u)
    return {
        "events": k,
        "resolved": n,
        "unresolved": u,
        "rate": k / n if n else None,
        "wilson95": list(ci) if ci else None,
        "wilson95_unresolved_as_events": list(worst) if worst and u else None,
    }
