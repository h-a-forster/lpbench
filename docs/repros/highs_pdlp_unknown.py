"""HiGHS downgrades some PDLP "Optimal" results to "Unknown" after its own check.

Re-runs, with the options the unit recorded, every main-block PDLP unit of a run directory
that ended with HiGHS status "Unknown", and reads the HiGHS log of each. Needs the run
directory and the prepared MPS files of `cache/`; uses no lpbench code.

    uv run python docs/repros/highs_pdlp_unknown.py runs/quick

cuPDLP-C prints its iteration log to stdout whatever the options say, so the result goes
to stderr. Result on 2026-11-06 (highspy 1.15.1, Windows): 31 of 31 units in three of four
re-runs, 30 of 31 in the other (that unit was not identified; FIT2P under `highs-hipdlp`
needs 9.6 s of its 10 s limit).
"""

import json
import sys
import tempfile
from pathlib import Path

import highspy

MARK = 'Model status changed from "Optimal" to "Unknown"'


def downgraded(record: dict, log: Path) -> bool:
    suite, name = record["instance"].split("/")
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", str(log))
    h.readModel(f"cache/mps/{suite}/{name}__p0.mps")
    for key, value in record["options"].items():
        h.setOptionValue(key, value)
    h.run()
    h.setOptionValue("log_file", "")  # closes the file
    return h.modelStatusToString(h.getModelStatus()) == "Unknown" and MARK in log.read_text(
        errors="replace"
    )


def main(run_dir: str) -> None:
    paths = sorted(Path(run_dir, "units").glob("main__*pdlp*.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    unknown = [r for r in records if r["raw_status"] == "Unknown" and r["status"] == "NUMERICAL"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        hits = [downgraded(r, Path(tmp) / f"{i}.log") for i, r in enumerate(unknown)]
    others = [r["unit_id"] for r, hit in zip(unknown, hits, strict=True) if not hit]
    print(f"{len(unknown)} units, {sum(hits)} downgraded; others: {others}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/quick")
