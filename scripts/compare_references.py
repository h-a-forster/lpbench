"""How the exact references change the verdicts of an existing run (DESIGN 12, item 31).

Re-scores every MIPLIB unit record of a run directory with the reference it would get now
(``lpbench.instances.miplib_reference_values``: exact where certified, else consensus),
using the worker's rule: verified at eps iff the KKT measure passes and the objective agrees
with the reference to eps. Re-scoring with the record's own reference must reproduce its
stored verdicts; the script stops if it does not.

    uv run python scripts/compare_references.py runs/quick
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lpbench.config import REPORT_EPS
from lpbench.instances import miplib_reference_values
from lpbench.verify import objective_agrees


def verdict(rec: dict, ref: float | None, eps: float) -> bool:
    v = rec.get("verification") or {}
    max_rel = rec.get("max_rel")
    if not v.get("has_dual") or max_rel is None or max_rel > eps:
        return False
    return ref is None or (rec.get("objective") is not None and objective_agrees(
        float(rec["objective"]), ref, eps))  # fmt: skip


def main(run_dir: str) -> int:
    refs = miplib_reference_values()
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((Path(run_dir) / "units").glob("*__miplib__*.json"))]  # fmt: skip
    changed: dict[str, tuple[float, float]] = {}
    flips: list[str] = []
    n_checked = 0
    for rec in recs:
        old = rec.get("reference")
        new, source = refs.get(rec["name"], (None, ""))
        if old is not None and new is not None and old != new:
            changed[rec["name"]] = (old, new)
        for e in REPORT_EPS:
            key = f"{e:.0e}"
            stored = (rec.get("verified_at") or {}).get(key)
            if stored is None:
                continue
            if verdict(rec, old, e) != stored:
                print(f"cannot reproduce the stored verdict: {rec['unit_id']} at {key}")
                return 1
            n_checked += 1
            if verdict(rec, new, e) != stored:
                flips.append(f"{rec['unit_id']} at {key}: {stored} -> {not stored} ({source})")
    names = {r["name"] for r in recs}
    n_exact = sum(1 for k in names if refs.get(k, (None, ""))[1] == "exact")
    print(f"{len(recs)} MIPLIB unit records, {len(names)} instances, {n_exact} with an exact "
          f"reference; {n_checked} verdicts re-scored")  # fmt: skip
    print(f"reference values that change: {len(changed)}")
    for name, (old, new) in sorted(changed.items()):
        print(f"  {name}: {old!r} -> {new!r} (rel {abs(new - old) / (1 + abs(old)):.1e})")
    print(f"verdicts that flip: {len(flips)}")
    for f in flips:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/quick"))
