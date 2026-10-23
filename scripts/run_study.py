"""Resume-on-failure wrapper around ``lpbench run`` (DESIGN 4.2).

Usage:
    uv run python scripts/run_study.py --config quick --out runs/quick [--pilot runs/pilot]
                                       [--attempts 3] [extra args passed to lpbench run]

Runs ``lpbench run`` up to ``--attempts`` times; every attempt resumes from the unit records
already on disk. Writes ``<out>/wrapper.log`` with local paths replaced by placeholders.
Exit status 0 when an attempt completes; otherwise the last attempt's status.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scrub(text: str) -> str:
    """Replace local paths (repo root, home, interpreter) with placeholders."""
    pairs = [
        (str(ROOT), "<repo>"),
        (str(ROOT).replace("\\", "/"), "<repo>"),
        (str(Path.home()), "<home>"),
        (str(Path.home()).replace("\\", "/"), "<home>"),
        (str(Path(sys.executable).parent), "<python>"),
    ]
    for old, new in sorted(pairs, key=lambda p: -len(p[0])):
        if old:
            text = text.replace(old, new)
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user and len(user) > 2:
        text = text.replace(user, "<user>")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="quick")
    ap.add_argument("--out")
    ap.add_argument("--attempts", type=int, default=3)
    args, extra = ap.parse_known_args()
    out = Path(args.out or f"runs/{args.config}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "wrapper.log"

    def log(msg: str) -> None:
        line = f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ} {scrub(msg)}"
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        print(line, flush=True)

    cmd = [
        sys.executable,
        "-m",
        "lpbench.cli",
        "run",
        "--config",
        args.config,
        "--out",
        str(out),
        *extra,
    ]
    rc = 1
    for attempt in range(1, args.attempts + 1):
        log(f"attempt {attempt}/{args.attempts}: {' '.join(cmd[1:])}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rc = proc.returncode
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        log(f"attempt {attempt} exited with {rc} after {time.perf_counter() - t0:.0f} s")
        for line in tail:
            log(f"  stderr: {line}")
        if rc in (0, 3, 4):  # done, refused by the wall budget, or config errors: no retry
            break
        time.sleep(5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
