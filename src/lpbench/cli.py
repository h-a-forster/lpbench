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
    "fetch": ("download instances with timeouts, size cap and checksums", "M2"),
    "prepare": ("canonicalise instances and write permuted MPS files", "M2"),
    "reference": ("compute consensus reference objectives", "M5"),
    "plan": ("print unit counts, worst-case bound and projected wall time", "M4"),
    "run": ("run a study with checkpoints and hard timeouts", "M4"),
    "status": ("show progress of a run directory", "M4"),
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
