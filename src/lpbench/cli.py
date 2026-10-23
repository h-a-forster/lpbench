"""Command-line interface. Subcommands not yet implemented exit with status 2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lpbench import __version__
from lpbench.versions import provenance

#: Planned subcommands and the milestone that implements each (docs/DESIGN.md, section 10).
PLANNED = {
    "reference": ("compute consensus reference objectives", "M5"),
    "report": ("build tables, figures and REPORT from a run directory", "M7"),
}


def _cmd_versions(args: argparse.Namespace) -> int:
    info = provenance()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0
    print(f"lpbench {info['lpbench']}")
    for name, version in info["distributions"].items():  # type: ignore[union-attr]
        print(f"  {name:<10} {version or 'not installed'}")
    for name, version in info["solver_libraries"].items():  # type: ignore[union-attr]
        print(f"  {name + ' (lib)':<10} {version or 'unavailable'}")
    print(f"  git        {info['git'] or 'unknown'}")
    return 0


def _suite_filter(cfg_name: str, suite: str | None):  # type: ignore[no-untyped-def]
    from lpbench.config import load_config
    from lpbench.instances import all_instances

    cfg = load_config(cfg_name)
    insts = all_instances(cfg)
    if suite:
        insts = [i for i in insts if i.suite == suite]
    return cfg, insts


def _cmd_fetch(args: argparse.Namespace) -> int:
    from lpbench.prepare import fetch

    cfg, insts = _suite_filter(args.config, args.suite)
    fetch(insts, cfg.download_cap_bytes, update_checksums=args.update_checksums)
    return 0


def _cmd_prepare(args: argparse.Namespace) -> int:
    from lpbench.prepare import prepare

    cfg, insts = _suite_filter(args.config, args.suite)
    prepare(cfg, insts)
    return 0


def _load_pilot(path: str | None) -> list[dict[str, object]]:
    from pathlib import Path

    from lpbench.runner import load_records

    if not path:
        return []
    return list(load_records(Path(path)).values())


def _plan_for(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from lpbench.planning import make_plan
    from lpbench.runner import build_units

    cfg, insts = _suite_filter(args.config, None)
    units = build_units(cfg, insts)
    plan = make_plan(cfg, units, insts, _load_pilot(args.pilot))
    return cfg, units, plan


def _cmd_plan(args: argparse.Namespace) -> int:
    from lpbench.planning import format_plan

    cfg, _, plan = _plan_for(args)
    print(format_plan(cfg, plan))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from lpbench.planning import format_plan
    from lpbench.runner import Runner, file_logger

    cfg, units, plan = _plan_for(args)
    out = Path(args.out or f"runs/{cfg.name}")
    log = file_logger(out / "progress.log")
    log(format_plan(cfg, plan))
    limit_s = cfg.max_projected_wall_min * 60
    bound = plan.projected_wall_s if plan.projected_wall_s is not None else plan.worst_case_wall_s
    if bound > limit_s and not args.allow_long and args.max_units is None:
        log(
            f"refusing to run: projected {bound / 60:.0f} min exceeds "
            f"{cfg.max_projected_wall_min:g} min (use --allow-long, or a pilot via --pilot)"
        )
        return 3
    runner = Runner(cfg, out, units, workers=args.workers, provenance=provenance(), log=log)
    summary = runner.run(max_units=args.max_units)
    errors = summary.statuses.get("CONFIG_ERROR", 0)
    return 4 if errors else 0


def _cmd_status(args: argparse.Namespace) -> int:
    from pathlib import Path

    from lpbench.runner import load_records, status_counts

    run_dir = Path(args.run_dir)
    recs = load_records(run_dir)
    manifest = run_dir / "manifest.json"
    total = None
    if manifest.exists():
        hist = json.loads(manifest.read_text(encoding="utf-8")).get("history", [])
        if hist:
            total = hist[-1].get("units_total")
    print(f"{run_dir.name}: {len(recs)}/{total if total is not None else '?'} units recorded")
    for st, n in status_counts(recs.values()).items():
        print(f"  {st:<14} {n}")
    log = run_dir / "progress.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            print(f"  last: {lines[-1]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lpbench",
        description="Honest CPU benchmark of open-source LP solvers (see docs/DESIGN.md).",
    )
    parser.add_argument("--version", action="version", version=f"lpbench {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_versions = sub.add_parser("versions", help="print library versions and provenance")
    p_versions.add_argument("--json", action="store_true", help="machine-readable output")
    p_versions.set_defaults(func=_cmd_versions)

    p_fetch = sub.add_parser("fetch", help="download instances (timeouts, size cap, checksums)")
    p_fetch.add_argument("--config", default="quick", help="config name or path")
    p_fetch.add_argument("--suite", choices=["netlib", "miplib"])
    p_fetch.add_argument(
        "--update-checksums", action="store_true", help="accept changed upstream files"
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_prep = sub.add_parser("prepare", help="canonicalise instances and write MPS files")
    p_prep.add_argument("--config", default="quick", help="config name or path")
    p_prep.add_argument("--suite", choices=["netlib", "miplib"])
    p_prep.set_defaults(func=_cmd_prepare)

    p_plan = sub.add_parser("plan", help="unit counts, worst-case bound, projected wall time")
    p_plan.add_argument("--config", default="quick", help="config name or path")
    p_plan.add_argument("--pilot", help="run directory of a pilot, for the projection")
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="run a study (checkpointed, resumable, hard deadlines)")
    p_run.add_argument("--config", default="quick", help="config name or path")
    p_run.add_argument("--out", help="run directory (default runs/<config name>)")
    p_run.add_argument("--workers", type=int, help="worker processes (at most the config's)")
    p_run.add_argument("--pilot", help="run directory of a pilot, for the projection")
    p_run.add_argument("--allow-long", action="store_true", help="run beyond the wall budget")
    p_run.add_argument("--max-units", type=int, help="stop after this many units")
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="progress of a run directory")
    p_status.add_argument("run_dir")
    p_status.set_defaults(func=_cmd_status)

    for name, (summary, milestone) in PLANNED.items():
        p = sub.add_parser(name, help=f"{summary} (planned, {milestone})")
        p.set_defaults(func=None, milestone=milestone)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.func is None:
        print(
            f"lpbench {args.command}: not implemented yet (milestone {args.milestone}).",
            file=sys.stderr,
        )
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
