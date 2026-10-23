"""Consensus reference objectives for MIPLIB LP relaxations (DESIGN 6.3).

The three independent simplex implementations (``highs-dsimplex``, ``ortools-glop``,
``scip-soplex``) solve every relaxation in its original order at eps = 1e-9 (feasibility
tolerances tightened to 1e-9) with a 60 s limit, as a checkpointed run of the normal runner.
An instance gets a reference if at least two configs return OPTIMAL solutions that pass the
KKT check at 1e-8 and agree with each other to 1e-8 relative; the reference is the mean of
the largest agreeing group. The KKT certificate stays the primary oracle; this value is a
secondary check, and it is partly circular because it comes from benchmarked libraries.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lpbench.config import StudyConfig, parse_config
from lpbench.instances import MIPLIB_REFERENCE, Instance

REFERENCE_CONFIGS = ("highs-dsimplex", "ortools-glop", "scip-soplex")
REFERENCE_EPS = 1e-9
KKT_EPS = 1e-8
AGREE_EPS = 1e-8
REFERENCE_TL_S = 60.0
FIELDS = ["name", "reference", "agreed", "n_optimal", "n_kkt_pass", "spread_rel", "note"]


def reference_config(cfg: StudyConfig) -> StudyConfig:
    """The study config of the reference pass for ``cfg``'s MIPLIB suite."""
    raw = {
        "name": f"{cfg.name}-reference",
        "workers": cfg.workers,
        "time_limit_s": REFERENCE_TL_S,
        "hard_deadline_factor": cfg.hard_deadline_factor,
        "hard_deadline_extra_s": cfg.hard_deadline_extra_s,
        "download_cap_bytes": cfg.download_cap_bytes,
        "seed": cfg.seed,
        "configs": list(REFERENCE_CONFIGS),
        "first_order": [],
        "verify_eps": list(cfg.verify_eps),
        "suites": {"miplib": {"list": cfg.miplib_list, "max_nnz": cfg.miplib_max_nnz}},
        "blocks": [{"name": "ref", "suites": ["miplib"], "configs": "all", "eps": REFERENCE_EPS}],
        "max_projected_wall_min": 15.0,
    }
    return parse_config(raw)


@dataclass(frozen=True)
class Consensus:
    name: str
    reference: float | None
    agreed: tuple[str, ...]
    n_optimal: int
    n_kkt_pass: int
    spread_rel: float | None
    note: str = ""


def _agree(a: float, b: float) -> bool:
    return abs(a - b) <= AGREE_EPS * (1.0 + max(abs(a), abs(b)))


def consensus(name: str, records: list[dict[str, Any]]) -> Consensus:
    """Consensus value from the reference-pass records of one instance."""
    opt = [r for r in records if r.get("status") == "OPTIMAL"]
    good = []
    for r in opt:
        mr = r.get("max_rel")
        if mr is not None and mr <= KKT_EPS and r.get("objective") is not None:
            good.append(r)
    best: list[dict[str, Any]] = []
    for anchor in good:
        group = [r for r in good if _agree(r["objective"], anchor["objective"])]
        if len(group) > len(best):
            best = group
    if len(best) >= 2:
        vals = [float(r["objective"]) for r in best]
        ref = statistics.fmean(vals)
        spread = (max(vals) - min(vals)) / (1.0 + abs(ref))
        return Consensus(
            name, ref, tuple(sorted(r["config"] for r in best)), len(opt), len(good), spread
        )
    if len(good) >= 2:
        note = "kkt-passing solutions disagree"
    elif len(good) == 1:
        note = "only one config passed the KKT check"
    else:
        note = "no config passed the KKT check"
    return Consensus(name, None, (), len(opt), len(good), None, note)


def build_reference_table(
    instances: list[Instance], records: list[dict[str, Any]]
) -> list[Consensus]:
    by_inst: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_inst.setdefault(r["name"], []).append(r)
    return [consensus(i.name, by_inst.get(i.name, [])) for i in instances]


def write_reference_csv(rows: list[Consensus], path: Path = MIPLIB_REFERENCE) -> None:
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            existing = {r["name"]: r for r in csv.DictReader(f)}
    for c in rows:
        existing[c.name] = {
            "name": c.name,
            "reference": repr(c.reference) if c.reference is not None else "",
            "agreed": ";".join(c.agreed),
            "n_optimal": str(c.n_optimal),
            "n_kkt_pass": str(c.n_kkt_pass),
            "spread_rel": f"{c.spread_rel:.2e}" if c.spread_rel is not None else "",
            "note": c.note,
        }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for name in sorted(existing):
            w.writerow(existing[name])
    tmp.replace(path)
