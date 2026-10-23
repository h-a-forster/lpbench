"""``lpbench plan``: unit counts, worst-case wall bound and a pilot-based projection."""

from __future__ import annotations

import heapq
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from lpbench.config import StudyConfig
from lpbench.instances import Instance, nnz_tertile
from lpbench.runner import Unit, expected_cost

UNIT_OVERHEAD_S = 0.3  # read, permute, verify and IPC per unit when no pilot says otherwise


@dataclass
class Plan:
    units: int
    first_order_units: int
    by_block: dict[str, int]
    worst_case_wall_s: float
    projected_wall_s: float | None
    projection_basis: str
    detail: dict[str, Any] = field(default_factory=dict)


def lpt_makespan(durations: list[float], workers: int) -> float:
    """Makespan of longest-processing-time-first list scheduling on ``workers`` machines."""
    loads = [0.0] * max(1, workers)
    heapq.heapify(loads)
    for d in sorted(durations, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + d)
    return max(loads)


def _pilot_model(
    records: list[dict[str, Any]], instances: list[Instance]
) -> tuple[dict[tuple, float], dict[tuple, float], dict[str, float], float]:
    by_inst = {i.id: i for i in instances}
    exact: dict[tuple, float] = {}
    strata: dict[tuple, list[float]] = defaultdict(list)
    per_config: dict[str, list[float]] = defaultdict(list)
    overheads: list[float] = []
    for r in records:
        wall = r.get("unit_wall_s")
        if wall is None:
            continue
        exact[(r["instance"], r["config"], r["eps"])] = float(wall)
        inst = by_inst.get(r["instance"])
        tert = nnz_tertile(inst, instances) if inst else 1
        strata[(r["config"], r["suite"], tert)].append(float(wall))
        per_config[r["config"]].append(float(wall))
        if r.get("solve_wall_s") is not None:
            overheads.append(float(wall) - float(r["solve_wall_s"]))
    strata_mean = {k: statistics.fmean(v) for k, v in strata.items()}
    config_mean = {k: statistics.fmean(v) for k, v in per_config.items()}
    overhead = statistics.median(overheads) if overheads else UNIT_OVERHEAD_S
    return exact, strata_mean, config_mean, overhead


def make_plan(
    cfg: StudyConfig,
    units: list[Unit],
    instances: list[Instance],
    pilot: list[dict[str, Any]] | None = None,
) -> Plan:
    by_block: dict[str, int] = defaultdict(int)
    for u in units:
        by_block[u.block] += 1
    fo = sum(1 for u in units if u.config in cfg.first_order)
    worst = len(units) * cfg.hard_deadline_s / cfg.workers
    if not pilot:
        return Plan(len(units), fo, dict(by_block), worst, None, "none (no pilot records)")
    exact, strata, per_config, overhead = _pilot_model(pilot, instances)
    preds = []
    n_exact = n_strata = n_cfg = n_default = 0
    for u in units:
        key = (u.instance.id, u.config, u.eps)
        if key in exact:
            preds.append(exact[key])
            n_exact += 1
            continue
        sk = (u.config, u.instance.suite, nnz_tertile(u.instance, instances))
        if sk in strata:
            preds.append(strata[sk])
            n_strata += 1
        elif u.config in per_config:
            preds.append(per_config[u.config])
            n_cfg += 1
        else:
            preds.append(min(cfg.time_limit_s, 1e-5 * expected_cost(u)) + overhead)
            n_default += 1
    proj = lpt_makespan(preds, cfg.workers)
    return Plan(
        units=len(units),
        first_order_units=fo,
        by_block=dict(by_block),
        worst_case_wall_s=worst,
        projected_wall_s=proj,
        projection_basis=(
            f"pilot: {n_exact} units measured, {n_strata} from (config, suite, size tertile) "
            f"means, {n_cfg} from config means, {n_default} defaults"
        ),
        detail={"sum_unit_s": sum(preds), "median_overhead_s": overhead},
    )


def format_plan(cfg: StudyConfig, plan: Plan) -> str:
    lines = [
        f"study {cfg.name}: {plan.units} units ({plan.first_order_units} first-order), "
        f"{cfg.workers} workers, TL {cfg.time_limit_s:g} s, hard deadline "
        f"{cfg.hard_deadline_s:g} s",
    ]
    for b, n in plan.by_block.items():
        lines.append(f"  block {b:<10} {n:>6} units")
    lines.append(
        f"  worst case (every unit to its hard deadline): {plan.worst_case_wall_s / 60:.0f} min"
    )
    if plan.projected_wall_s is not None:
        lines.append(
            f"  projected wall time: {plan.projected_wall_s / 60:.1f} min ({plan.projection_basis})"
        )
    else:
        lines.append(f"  projected wall time: unknown ({plan.projection_basis})")
    return "\n".join(lines)
