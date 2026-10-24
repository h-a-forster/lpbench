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
    wall: float | None = None,
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
        "solve_wall_s": min(t * 1.01, TL) if wall is None else wall,
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
    manifest = {
        "study": "synthetic",
        "config_hash": "h",
        "config": cfg,
        "provenance": {},
        "history": [{"units_total": len(recs)}],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _write(run: Path, rec: dict) -> None:
    (run / "units" / f"{rec['unit_id']}.json").write_text(json.dumps(rec), encoding="utf-8")


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
    assert res["meta"]["completeness"]["complete"] is True
    # (c) is compared with the netlib-only primary ranking, not the all-instance one
    c = next(v for k, v in sens.items() if k.startswith("(c)"))
    assert c["baseline"] == "netlib only" and sens["(a) trust solver status"]["baseline"] == (
        "primary"
    )
    assert sens["(h) permutation k=0 (subset)"]["baseline"] == "primary"
    assert sens["(h) permutation k=1 (subset)"]["baseline"] == "subset, k=0"


def test_eps_sensitivity_baseline_is_netlib_only(tmp_path: Path) -> None:
    """If netlib alone ranks differently from all instances, the requested-epsilon analysis
    (netlib only) must be compared with the netlib ranking: identical data give tau = 1."""
    run_dir = make_run(tmp_path)
    for suite, t in (("netlib", 0.001), ("miplib", 9.0)):  # scip fastest on netlib only
        for i in range(8):
            p = run_dir / "units" / f"main__{suite}__p{i}__p0__scip-soplex__e1e-06__r0.json"
            r = json.loads(p.read_text(encoding="utf-8"))
            r["solve_cpu_s"] = r["solve_wall_s"] = t
            _write(run_dir, r)
    res = R.build(R.load_run(run_dir))
    sens = res["tables"]["sensitivity"]
    c = next(v for k, v in sens.items() if k.startswith("(c)"))
    g = sens["(g) suite netlib only"]
    assert c["order"] == g["order"] and c["tau"] == pytest.approx(1.0)
    assert g["tau"] < 1.0  # netlib alone differs from the primary ranking


def test_failure_breakdown_examples_and_late_answers(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    # KKT passes but the objective is wrong (trento1-like): must be the top example
    bad = _rec("main", "miplib", 1, "highs-pdlp", 0, 0.5, ok=False)
    bad["max_rel"] = 1e-9
    bad["ref_rel_error"] = 1.0
    bad["verification"] = dict(bad["verification"], primal_residual=15.0, rel_primal_comp=0.9)
    _write(run_dir, bad)
    # primal residual too large
    badp = _rec("main", "miplib", 2, "highs-pdlp", 0, 0.5, ok=False)
    badp["verification"] = dict(badp["verification"], rel_primal=1e-3)
    _write(run_dir, badp)
    # verified, but returned after the wall-time limit: scored as TIME_LIMIT
    late = _rec("main", "miplib", 3, "highs-pdlp", 0, 8.9, wall=10.85)
    _write(run_dir, late)
    res = R.build(R.load_run(run_dir))
    v = res["tables"]["verification"]["highs-pdlp"]
    assert v["examples"][0]["instance"] == "miplib/p1"
    assert v["examples"][0]["primal_residual"] == 15.0
    assert v["failed_criteria"] == {"P": 1, "ref": 1}
    out = res["tables"]["outcomes"]["all"]["highs-pdlp"]
    assert out["TIME_LIMIT"] == 3 and out["claimed_unverified"] == 2
    assert res["meta"]["optimal_after_limit"] == [late["unit_id"]]
    assert res["tables"]["solved_counts"]["highs-pdlp"] == 16 - 2 - 3


def test_short_times_are_left_out_of_gsd(tmp_path: Path) -> None:
    """Time GSDs use raw times and only instances whose every time is >= GSD_MIN_TIME_S."""
    run_dir = make_run(tmp_path)
    run = R.load_run(run_dir)
    res = R.build(run)
    var = res["tables"]["variability"]["highs-dsimplex"]
    perm = [r for r in run.records if r["block"] == "perm" and r["config"] == "highs-dsimplex"]
    main_t = {
        r["instance"]: r["solve_cpu_s"]
        for r in run.records
        if r["block"] == "main" and r["config"] == "highs-dsimplex"
    }
    long_ = {r["instance"] for r in perm}
    long_ = {i for i in long_ if main_t[i] >= R.GSD_MIN_TIME_S}
    assert var["instances_long"] == len(long_) < var["instances"]
    nz = res["tables"]["noise"]["highs-dsimplex"]
    assert nz["instances_long"] <= nz["instances"]


def test_incomplete_run_is_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = make_run(tmp_path)
    for p in sorted((run_dir / "units").glob("main__miplib__p*__highs-dsimplex__*.json"))[:3]:
        p.unlink()
    res = R.build(R.load_run(run_dir))
    comp = res["meta"]["completeness"]
    assert comp["complete"] is False and comp["missing"] == 3 and comp["main_missing_pairs"] == 3
    out = tmp_path / "pub"
    assert main(["report", str(run_dir), "--publish", str(out)]) == 0
    assert "WARNING: incomplete run" in capsys.readouterr().err
    assert "**Incomplete run:**" in (out / "summary.md").read_text(encoding="utf-8")


def test_publish_is_byte_identical(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["report", str(run), "--publish", str(a)]) == 0
    R.publish(run, b)
    for name in ("results.json", "summary.md", "units.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
        assert b"\r" not in (a / name).read_bytes(), name  # LF only, stable under git
    md = (a / "summary.md").read_text(encoding="utf-8")
    assert "## What changes the ranking" in md and "nan" not in md.lower()
    if (a / "performance_profiles.svg").exists():
        assert (a / "performance_profiles.svg").read_bytes() == (
            b / "performance_profiles.svg"
        ).read_bytes()
        assert b"\r" not in (a / "performance_profiles.svg").read_bytes()
