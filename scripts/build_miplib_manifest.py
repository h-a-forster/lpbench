"""Build src/lpbench/data/miplib_benchmark.csv from the MIPLIB 2017 benchmark table.

Usage:
    uv run python scripts/build_miplib_manifest.py [--html FILE] [--list FILE]

Without arguments both pages are downloaded from miplib.zib.de (timeouts, size cap). The
HTML table is parsed only here, never at run time. Existing sha256/bytes columns of the
manifest are preserved for instances that remain in the list.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lpbench.download import Budget, download  # noqa: E402
from lpbench.instances import MIPLIB_MANIFEST, MIPLIB_SITE  # noqa: E402

FIELDS = ["name", "rows", "cols", "nnz", "binaries", "integers", "continuous", "status"]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.rows: list[list[str]] = []
        self.cur: list[str] = []
        self.text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and dict(attrs).get("id") == "miplibtable":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.cur = []
        elif self.in_table and tag in ("td", "th"):
            self.in_cell = True
            self.text = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self.in_table:
            self.in_table = False
        elif self.in_table and tag in ("td", "th"):
            self.in_cell = False
            self.cur.append(" ".join(self.text.split()))
        elif self.in_table and tag == "tr" and self.cur:
            self.rows.append(self.cur)

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.text += data


def parse_table(html: str) -> list[dict[str, str]]:
    p = _TableParser()
    p.feed(html)
    header, *body = p.rows
    idx = {h: i for i, h in enumerate(header)}
    out = []
    for r in body:
        out.append(
            {
                "name": r[idx["Instance"]],
                "rows": r[idx["Constraints"]],
                "cols": r[idx["Variables"]],
                "nnz": r[idx["Nonz."]],
                "binaries": r[idx["Binaries"]],
                "integers": r[idx["Integers"]],
                "continuous": r[idx["Continuous"]],
                "status": r[idx["Status"]],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path)
    ap.add_argument("--list", type=Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        budget = Budget(cap_bytes=20_000_000)
        html_path = args.html or Path(td) / "tag_benchmark.html"
        list_path = args.list or Path(td) / "benchmark-v2.test"
        if args.html is None:
            download(f"{MIPLIB_SITE}/tag_benchmark.html", html_path, budget)
        if args.list is None:
            download(f"{MIPLIB_SITE}/downloads/benchmark-v2.test", list_path, budget)
        rows = parse_table(html_path.read_text(encoding="utf-8", errors="replace"))
        names = [
            re.sub(r"\.mps\.gz$", "", ln.strip())
            for ln in list_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    by_name = {r["name"]: r for r in rows}
    missing = sorted(set(names) - set(by_name))
    if missing:
        print(f"instances in the list but not in the table: {missing}", file=sys.stderr)
        return 1
    if len(names) != 240:
        print(f"warning: benchmark list has {len(names)} instances, expected 240", file=sys.stderr)
    old: dict[str, dict[str, str]] = {}
    if MIPLIB_MANIFEST.exists():
        with open(MIPLIB_MANIFEST, encoding="utf-8") as f:
            old = {r["name"]: r for r in csv.DictReader(f)}
    MIPLIB_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MIPLIB_MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for n in sorted(names):
            w.writerow({k: by_name[n][k] for k in FIELDS})
    print(f"wrote {len(names)} instances ({len(old)} previously)")
    for cap in (100_000, 1_000_000):
        k = sum(1 for n in names if int(by_name[n]["nnz"]) <= cap)
        print(f"  nnz <= {cap:>9,}: {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
