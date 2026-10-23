"""``lpbench report`` on a synthetic run directory: exact values and byte-identical output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lpbench import report as R
from lpbench.cli import main

CONFIGS = ["highs-dsimplex", "highs-pdlp", "scip-soplex"]
TL = 10.0


def _rec(
    block: str,
    suite: str,
    i: int,
    cfg: str,
    k: int,
    t: float,
    ok: bool = True,
    status: str = "OPTIMAL",
    eps: float = 1e-6,
    repeat: int = 0,
) -> dict:
    inst = f"{suite}/p{i}"
    ver = {e: ok for e in ("1e-04", "1e-06", "1e-08")}
    return {
        "unit_id": f"{block}__{suite}__p{i}__p{k}__{cfg}__e{eps:.0e}__r{repeat}",
        "config_hash": "h",
        "block": block,
        "suite": suite,
        "instance": inst,
        "name": f"p{i}",
        "k": k,
        "config": cfg,
        "eps": eps,
        "repeat": repeat,
        "status": status,
        "solve_cpu_s": t,
        "solve_wall_s": t * 1.01,
        "unit_wall_s": t + 0.2,
        "total_iterations": 100 + k,
        "verified_at": ver,
        "floor_limited_at": {},
        "verified_solve": ok and status == "OPTIMAL",
        "claimed_unverified": status == "OPTIMAL" and not ok,
        "max_rel": 1e-9 if ok else 1e-3,
        "reference": 1.0,
        "reference_source": "test",
        "objective": 1.0,
        "ref_rel_error": 0.0,
        "nnz_published": 10 * i,
        "git": "abc",
        "verification": {"rel_primal": 1e-9, "rel_dual": 1e-9, "rel_gap": 1e-9, "has_dual": True},
    }


def make_run(root: Path) -> Path:
    rng = np.random.default_rng(0)
    recs = []
    for suite in ("netlib", "miplib"):
        for i in range(8):
            base = float(rng.lognormal(0, 1))
            for cfg, f in zip(CONFIGS, (1.0, 3.0, 1.5), strict=True):
                t = base * f
                fail = cfg == "highs-pdlp" and i == 0
                recs.append(
                    _rec(
                        "main",
                        suite,
                        i,
                        cfg,
                        0,
                        min(t, TL),
                        ok=not fail,
                        status="TIME_LIMIT" if fail else "OPTIMAL",
                    )
                )
                if i < 3:
                    for k in (1, 2):
                        recs.append(_rec("perm", suite, i, cfg, k, t * (1 + 0.1 * k)))
                if suite == "netlib" and cfg == "highs-pdlp":
                    recs.append(_rec("eps1e-4", suite, i, cfg, 0, t / 2, eps=1e-4))
            if i < 2:
                recs.append(_rec("noise", suite, i, CONFIGS[0], 0, base * 1.05, repeat=1))
    run = root / "run"
    (run / "units").mkdir(parents=True)
    for r in recs:
        (run / "units" / f"{r['unit_id']}.json").write_text(json.dumps(r), encoding="utf-8")
    cfg = {
        "name": "synthetic",
        "configs": CONFIGS,
        "first_order": ["highs-pdlp"],
        "time_limit_s": TL,
        "seed": 7,
        "workers": 3,
        "blocks": [{"name": "main", "eps": 1e-6}, {"name": "eps1e-4", "eps": 1e-4}],
        "stats": {"bootstrap_resamples": 300},
    }
    (run / "manifest.json").write_text(
        json.dumps({"study": "synthetic", "config_hash": "h", "config": cfg, "provenance": {}}),
        encoding="utf-8",
    )
    return run


def test_build_exact_values(tmp_path: Path) -> None:
    run = R.load_run(make_run(tmp_path))
    res = R.build(run)
    t = res["tables"]
    assert res["meta"]["units"] == len(run.records)
    assert t["outcomes"]["all"]["highs-pdlp"]["TIME_LIMIT"] == 2
    assert t["outcomes"]["all"]["highs-dsimplex"]["verified"] == 16
    assert t["primary_ranking"] == ["highs-dsimplex", "scip-soplex", "highs-pdlp"]
    # SGM of highs-dsimplex equals the hand-computed shifted geometric mean of its times
    main = [r for r in run.records if r["block"] == "main" and r["config"] == "highs-dsimplex"]
    times = np.array([r["solve_cpu_s"] for r in main])
    want = math.exp(np.mean(np.log(times + 1.0))) - 1.0
    assert t["sgm"]["all"]["highs-dsimplex"]["sgm"] == pytest.approx(want, rel=1e-5)
    pair = next(
        p for p in t["pairwise"]["all"] if p["a"] == "highs-dsimplex" and p["b"] == "scip-soplex"
    )
    assert pair["ratio"] < 1 and pair["p_holm"] < 0.05
    assert t["profile_points"]["highs-dsimplex"]["1"] == 1.0
    assert t["tolerance"]["highs-pdlp"]["verified_at_1e-4_run"] == 8
    assert t["variability"]["highs-dsimplex"]["instances"] == 6
    assert t["noise"]["highs-dsimplex"]["instances"] == 4
    sens = t["sensitivity"]
    assert sens["(a) trust solver status"]["tau"] == pytest.approx(1.0)
    assert "(e) PAR2" in sens and "(i) wall time instead of CPU time" in sens


def test_publish_is_byte_identical(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["report", str(run), "--publish", str(a)]) == 0
    R.publish(run, b)
    for name in ("results.json", "summary.md", "units.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    md = (a / "summary.md").read_text(encoding="utf-8")
    assert "## What changes the ranking" in md and "nan" not in md.lower()
    if (a / "performance_profiles.svg").exists():
        assert (a / "performance_profiles.svg").read_bytes() == (
            b / "performance_profiles.svg"
        ).read_bytes()
