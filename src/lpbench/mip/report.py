"""``lpbench mip report``: E1-E4 per solver, refutations, rounding flips, full-set projection."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from lpbench.instances import miplib_instances
from lpbench.mip.study import MipConfig, run_config
from lpbench.runner import atomic_write_json, load_records, status_counts


def load_claims(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "claims.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no claims.jsonl in {Path(run_dir).name}; run lpbench mip check")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def elapsed_s(run_dir: Path, recs: dict[str, dict[str, Any]]) -> float | None:
    """Wall time from the first start in the manifest to the last finished unit."""
    man = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    starts = [h["started"] for h in man.get("history", [])]
    ends = [r["finished_at"] for r in recs.values() if r.get("finished_at")]
    if not starts or not ends:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    first = dt.datetime.strptime(min(starts), fmt)
    return (dt.datetime.strptime(max(ends), fmt) - first).total_seconds()


def full_set_projection(
    cfg: MipConfig, recs: dict[str, dict[str, Any]], workers: int, time_limit_s: float
) -> dict[str, Any]:
    """Wall time of all MIPLIB 2017 benchmark instances with the same solvers.

    Pilot units keep their measured wall time (when the time limit is the pilot's); every
    other unit is assumed to run to the time limit plus the pilot's mean overhead beyond
    the solve (reading, IPC). LPT list scheduling on ``workers`` (``planning.lpt_makespan``).
    """
    from lpbench.planning import lpt_makespan

    overhead = [
        r["unit_wall_s"] - r["solve_wall_s"]
        for r in recs.values()
        if r.get("unit_wall_s") is not None and r.get("solve_wall_s") is not None
    ]
    extra = sum(overhead) / len(overhead) if overhead else 1.0
    measured = {(r["instance"], r["solver"]): r.get("unit_wall_s") for r in recs.values()}
    same_tl = time_limit_s == cfg.time_limit_s
    names = [i.name for i in miplib_instances(10**12)]
    durations = []
    for n in names:
        for s in cfg.solvers:
            m = measured.get((n, s))
            durations.append(m if (m is not None and same_tl) else time_limit_s + extra)
    return {
        "instances": len(names),
        "solvers": list(cfg.solvers),
        "units": len(durations),
        "time_limit_s": time_limit_s,
        "workers": workers,
        "projected_wall_s": lpt_makespan(durations, workers),
        "assumption": "units on instances outside the pilot run to the time limit",
    }


def report_run(
    run_dir: Path, out_dir: Path | None = None, full_time_limit_s: float | None = None
) -> Path:
    """Write ``summary.json`` and ``summary.md`` (default ``RUN_DIR/report``)."""
    from lpbench.mip.claims import BIN_LABELS, E4_THRESHOLD, ESTIMANDS, estimate

    run_dir = Path(run_dir)
    out = Path(out_dir or run_dir / "report")
    cfg = run_config(run_dir)
    recs = load_records(run_dir)
    claims = load_claims(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    def counts(values: list[Any]) -> dict[str, int]:
        return status_counts({"status": v} for v in values)

    per_solver: dict[str, Any] = {}
    for s in cfg.solvers:
        cs = [c for c in claims if c["solver"] == s]
        per_solver[s] = {
            "units": len(cs),
            "statuses": counts([c["status"] for c in cs]),
            "primal_classes": counts([c["primal_class"] for c in cs if c["has_solution"]]),
            "completions": counts([c["completion"] for c in cs]),
            "estimands": {e: estimate(cs, e) for e in ESTIMANDS},
            "missing_fields": counts([f for c in cs for f in c.get("missing") or []]),
            "objective_misstated": sum(1 for c in cs if c.get("objective_ok") is False),
            "median_cpu_per_wall": _median_cpu_per_wall(
                [r for r in recs.values() if r.get("solver") == s]
            ),
        }
    keys = (
        "unit_id",
        "solver",
        "instance",
        "status",
        "dual_claim",
        "witness",
        "refutation_severity",
        "refuted_bins",
    )
    flagged = [
        {k: c.get(k) for k in keys}
        for c in claims
        if c.get("refuted_bins") and c["refuted_bins"][f">{E4_THRESHOLD:g}"]
    ]
    summary = {
        "study": cfg.name,
        "config_hash": cfg.config_hash(),
        "units_expected": len(cfg.solvers) * len(cfg.instances),
        "units_with_claims": len(claims),
        "time_limit_s": cfg.time_limit_s,
        "seed": cfg.seed,
        "workers": max((h.get("workers", 0) for h in man.get("history", [])), default=None),
        "elapsed_wall_s": elapsed_s(run_dir, recs),
        "unit_wall_s_total": sum(r.get("unit_wall_s") or 0.0 for r in recs.values()),
        "solver_versions": sorted(
            {f"{r.get('solver')} {r.get('solver_version')}" for r in recs.values()}
        ),
        "provenance": man.get("provenance", {}),
        "estimand_definitions": ESTIMANDS,
        "severity_bins": list(BIN_LABELS),
        "per_solver": per_solver,
        "refutations_above_1e-9": flagged,
        "data_rounding_flips": [
            {"unit_id": c["unit_id"], "fields": c["data_rounding_flips"]}
            for c in claims
            if c.get("data_rounding_flips")
        ],
        "check_errors": [
            {"unit_id": c["unit_id"], "error": c["check_error"]}
            for c in claims
            if c.get("check_error")
        ],
        "full_set_projection": full_set_projection(
            cfg, recs, min(cfg.workers, 3), full_time_limit_s or cfg.time_limit_s
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_md(summary), encoding="utf-8", newline="\n")
    return out


def _median_cpu_per_wall(recs: list[dict[str, Any]]) -> float | None:
    """Median process CPU time per wall second of the solves that took at least 1 s: a
    single-threaded solver on a free core is close to 1 (time limits are wall time)."""
    import statistics

    ratios = [
        r["solve_cpu_s"] / r["solve_wall_s"]
        for r in recs
        if (r.get("solve_wall_s") or 0) >= 1.0 and r.get("solve_cpu_s") is not None
    ]
    return round(statistics.median(ratios), 2) if ratios else None


def _ci(ci: list[float] | None) -> str:
    return "-" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def summary_md(s: dict[str, Any]) -> str:
    wall = s["elapsed_wall_s"]
    lines = [
        f"# MIP claims: {s['study']}",
        "",
        f"{s['units_with_claims']} claim records of {s['units_expected']} expected units; "
        f"time limit {s['time_limit_s']:g} s, seed {s['seed']}, {s['workers']} workers; "
        + (f"elapsed wall time {wall / 60:.1f} min." if wall else "elapsed wall time unknown."),
        f"Solver versions: {', '.join(s['solver_versions'])}.",
        "",
        "## Estimands",
        "",
        "95% Wilson intervals; with zero events the interval is [0, 3.84/(n + 3.84)].",
        "",
    ]
    lines += [f"- {e}: {d}." for e, d in s["estimand_definitions"].items()]
    lines += [
        "",
        "| solver | estimand | events | resolved | unresolved | rate | Wilson 95% "
        "| unresolved as events |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for solver, ps in s["per_solver"].items():
        for e, est in ps["estimands"].items():
            rate = "-" if est["rate"] is None else f"{est['rate']:.3f}"
            lines.append(
                f"| {solver} | {e} | {est['events']} | {est['resolved']} | {est['unresolved']} "
                f"| {rate} | {_ci(est['wilson95'])} | {_ci(est['wilson95_unresolved_as_events'])} |"
            )
    lines += [
        "",
        "## Outcomes per solver",
        "",
        "Time limits are wall time; a single-threaded solve on a free core gets close to 1 CPU "
        "second per wall second.",
        "",
    ]
    for solver, ps in s["per_solver"].items():
        lines.append(
            f"- {solver}: statuses {ps['statuses']}; primal classes {ps['primal_classes']}; "
            f"completions {ps['completions']}; claim fields missing "
            f"{ps['missing_fields'] or 'none'}; objective misstated {ps['objective_misstated']}; "
            f"median solve CPU seconds per wall second {ps['median_cpu_per_wall']}."
        )
    lines += ["", "## Refutations above 1e-9", ""]
    if s["refutations_above_1e-9"]:
        lines += ["Recorded for follow-up; none is called a bug (DESIGN 6.4 item 5).", ""]
        lines += [
            f"- {r['unit_id']}: status {r['status']}, dual claim {r['dual_claim']}, "
            f"witness {r['witness']}, severity {r['refutation_severity']}"
            for r in s["refutations_above_1e-9"]
        ]
    else:
        lines.append("None.")
    lines += ["", "## Data rounding", ""]
    flips = s["data_rounding_flips"]
    if flips:
        lines += [f"- {f['unit_id']}: {', '.join(f['fields'])}" for f in flips]
    else:
        lines.append("No verdict differs between the exact data and the float64 data.")
    if s["check_errors"]:
        lines += ["", "## Check errors", ""]
        lines += [f"- {c['unit_id']}: {c['error']}" for c in s["check_errors"]]
    proj = s["full_set_projection"]
    lines += [
        "",
        "## Projected wall time of the full benchmark set",
        "",
        f"{proj['units']} units ({proj['instances']} instances x {len(proj['solvers'])} "
        f"solvers) at {proj['time_limit_s']:g} s on {proj['workers']} workers: "
        f"{proj['projected_wall_s'] / 3600:.1f} h ({proj['assumption']}).",
        "",
    ]
    return "\n".join(lines)
