"""Runner, workers, checkpoints and the plan (M4 done-checks).

Kill-and-resume (no duplicates, no lost units), hard deadline with a sleeping fake adapter,
and a crash with a fake adapter that calls ``os._exit``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lpbench.config import ConfigError, load_config, parse_config
from lpbench.instances import Instance
from lpbench.lpmodel import save_npz
from lpbench.planning import lpt_makespan, make_plan
from lpbench.runner import Runner, Unit, build_units, load_records
from lpbench.synthetic import random_known_lp

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tl: float = 5.0, extra: float = 2.0, workers: int = 2) -> object:
    return parse_config(
        {
            "name": "test",
            "workers": workers,
            "time_limit_s": tl,
            "hard_deadline_factor": 1.0,
            "hard_deadline_extra_s": extra,
            "download_cap_bytes": 1000,
            "seed": 7,
            "configs": ["highs-dsimplex", "scip-soplex", "ortools-glop"],
            "first_order": [],
            "blocks": [{"name": "main", "suites": ["netlib"], "configs": "all", "eps": 1e-6}],
        }
    )


def _instances(root: Path, n: int = 4) -> list[Instance]:
    out = []
    for s in range(n):
        k = random_known_lp(10, 12, seed=900 + s)
        inst = Instance(
            suite="synthetic",
            name=f"rand{s}",
            file=f"rand{s}",
            url="https://netlib.org/x",
            rows=k.lp.m,
            cols=k.lp.n,
            nnz=k.lp.nnz,
            reference=k.objective,
            reference_source="kkt",
        )
        save_npz(k.lp, inst.canonical_path(root))
        out.append(inst)
    return out


@pytest.fixture()
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("LPBENCH_CACHE", str(root))
    return root


def _units(insts: list[Instance], configs: list[str], ks: tuple[int, ...] = (0,)) -> list[Unit]:
    return [Unit("main", i, k, c, 1e-6) for i in insts for k in ks for c in configs]


def test_runner_records_verified_units(cache: Path, tmp_path: Path) -> None:
    insts = _instances(cache)
    units = _units(insts, ["highs-dsimplex", "scip-soplex", "ortools-glop"], ks=(0, 1))
    out = tmp_path / "run"
    s = Runner(_cfg(), out, units, workers=2).run()
    assert s.completed == len(units) == 24
    recs = load_records(out)
    assert set(recs) == {u.unit_id for u in units}
    for r in recs.values():
        assert r["status"] == "OPTIMAL", r
        assert r["verified_solve"] is True and r["claimed_unverified"] is False
        assert r["verified_at"]["1e-06"] is True
        assert abs(r["ref_rel_error"]) < 1e-9
        assert r["solve_wall_s"] >= 0 and r["unit_wall_s"] >= r["solve_wall_s"]
    # permutation correctness: same objective for k = 0 and k = 1
    for i in insts:
        for c in ("highs-dsimplex", "scip-soplex"):
            o0 = recs[Unit("main", i, 0, c, 1e-6).unit_id]["objective"]
            o1 = recs[Unit("main", i, 1, c, 1e-6).unit_id]["objective"]
            assert o0 == pytest.approx(o1, rel=1e-9, abs=1e-9)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"] == _cfg().config_hash()  # type: ignore[attr-defined]
    log = out / "progress.log"
    assert not log.exists()  # the runner logs through the callback only
    # a rerun skips everything
    s2 = Runner(_cfg(), out, units).run()
    assert s2.completed == 0 and s2.skipped == len(units)


def test_hard_deadline_kills_sleeping_worker(
    cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LPBENCH_TEST_SLEEP_S", "120")
    insts = _instances(cache, 2)
    units = [Unit("main", insts[0], 0, "test-sleep", 1e-6)] + _units(insts, ["test-ok"])
    out = tmp_path / "run"
    t0 = time.perf_counter()
    Runner(_cfg(tl=0.5, extra=1.5), out, units, workers=1).run()
    elapsed = time.perf_counter() - t0
    recs = load_records(out)
    killed = recs[units[0].unit_id]
    assert killed["status"] == "KILLED"
    assert 2.0 <= killed["solve_wall_s"] < 10.0
    assert elapsed < 60
    for u in units[1:]:
        assert recs[u.unit_id]["status"] == "OPTIMAL"  # the respawned worker carries on


def test_crashing_worker_is_recorded_and_replaced(cache: Path, tmp_path: Path) -> None:
    insts = _instances(cache, 2)
    units = [
        Unit("main", insts[0], 0, "test-crash", 1e-6),
        Unit("main", insts[1], 0, "test-raise", 1e-6),
    ] + _units(insts, ["test-ok"])
    out = tmp_path / "run"
    Runner(_cfg(), out, units, workers=1).run()
    recs = load_records(out)
    crash = recs[units[0].unit_id]
    assert crash["status"] == "ERROR" and "code 3" in crash["message"]
    raised = recs[units[1].unit_id]
    assert raised["status"] == "ERROR" and "deliberate failure" in raised["message"]
    assert all(recs[u.unit_id]["status"] == "OPTIMAL" for u in units[2:])


_CHILD = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, {tests!r})
import test_runner as T
from lpbench.runner import Runner
insts = T._instances(Path({cache!r}), 6)
units = T._units(insts, ["test-ok", "highs-dsimplex", "scip-soplex", "ortools-glop"])
Runner(T._cfg(), Path({out!r}), units, workers=2).run()
"""


def test_kill_and_resume(cache: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    code = _CHILD.format(tests=str(ROOT / "tests"), cache=str(cache), out=str(out))
    env = dict(os.environ, LPBENCH_CACHE=str(cache), LPBENCH_TEST_SLEEP_S="0")
    proc = subprocess.Popen([sys.executable, "-c", code], env=env)
    units_dir = out / "units"
    deadline = time.time() + 120
    while time.time() < deadline:
        n = len(list(units_dir.glob("*.json"))) if units_dir.exists() else 0
        if n >= 3:
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=30)
    first = {p.name: p.read_text(encoding="utf-8") for p in units_dir.glob("*.json")}
    assert 3 <= len(first) < 24, len(first)
    # resume in-process: completes the rest, leaves finished records untouched
    insts = _instances(cache, 6)
    units = _units(insts, ["test-ok", "highs-dsimplex", "scip-soplex", "ortools-glop"])
    s = Runner(_cfg(), out, units, workers=2).run()
    files = sorted(p.name for p in units_dir.glob("*.json"))
    assert len(files) == len(set(files)) == 24  # no duplicates, nothing lost
    assert s.skipped + s.completed == 24
    assert s.skipped >= len(first)
    for name, text in first.items():
        assert (units_dir / name).read_text(encoding="utf-8") == text
    assert not list(units_dir.glob(".*.tmp"))


def test_config_change_invalidates_records(cache: Path, tmp_path: Path) -> None:
    insts = _instances(cache, 1)
    units = _units(insts, ["highs-dsimplex"])
    out = tmp_path / "run"
    Runner(_cfg(tl=5.0), out, units).run()
    s = Runner(_cfg(tl=6.0), out, units).run()
    assert s.completed == 1 and s.skipped == 0


# ---------------------------------------------------------------------------------------------
# Config validation and the quick study's units and plan
# ---------------------------------------------------------------------------------------------


def test_config_validation() -> None:
    good = {
        "name": "x",
        "workers": 3,
        "time_limit_s": 1,
        "download_cap_bytes": 10,
        "seed": 1,
        "configs": ["highs-dsimplex"],
        "blocks": [{"name": "b", "eps": 1e-6}],
    }
    parse_config(good)
    for patch in (
        {"workers": 4},
        {"download_cap_bytes": 3_000_000_000},
        {"configs": ["nope"]},
        {"time_limit_s": 0},
        {"blocks": [{"name": "b", "eps": 2.0}]},
        {"blocks": [{"name": "b", "eps": 1e-6}, {"name": "b", "eps": 1e-6}]},
    ):
        with pytest.raises(ConfigError):
            parse_config({**good, **patch})
    with pytest.raises(ConfigError):
        parse_config({k: v for k, v in good.items() if k != "seed"})


def test_quick_units_match_design_counts() -> None:
    cfg = load_config("quick")
    units = build_units(cfg)
    by_block: dict[str, int] = {}
    for u in units:
        by_block[u.block] = by_block.get(u.block, 0) + 1
    assert by_block["main"] == 206 * 7 == 1442
    assert by_block["eps1e-4"] == 92 * 3 == 276
    assert len({u.unit_id for u in units}) == len(units)
    assert cfg.hard_deadline_s == 25.0


def test_plan_projection_from_pilot() -> None:
    cfg = load_config("quick")
    from lpbench.instances import all_instances

    insts = all_instances(cfg)
    units = build_units(cfg, insts)
    plan0 = make_plan(cfg, units, insts, None)
    assert plan0.projected_wall_s is None
    assert plan0.worst_case_wall_s == pytest.approx(len(units) * 25.0 / 3)
    pilot = [
        {
            "instance": units[0].instance.id,
            "config": units[0].config,
            "eps": units[0].eps,
            "suite": units[0].instance.suite,
            "unit_wall_s": 2.0,
            "solve_wall_s": 1.8,
        },
    ]
    plan1 = make_plan(cfg, units, insts, pilot)
    assert plan1.projected_wall_s is not None and plan1.projected_wall_s > 0


def test_lpt_makespan() -> None:
    assert lpt_makespan([3, 3, 2, 2, 2], 2) == 7  # LPT, not the optimum 6
    assert lpt_makespan([5], 3) == 5
    assert lpt_makespan([], 3) == 0


def test_wrapper_scrubs_local_paths() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_study", ROOT / "scripts" / "run_study.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = f"error in {ROOT / 'src' / 'x.py'} and {Path.home() / 'y'} via {sys.executable}"
    out = mod.scrub(text)
    assert str(ROOT) not in out and str(Path.home()) not in out
    assert "<repo>" in out and "<home>" in out


def test_cli_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from lpbench import cli

    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "a.json").write_text(
        json.dumps({"unit_id": "a", "status": "OPTIMAL"}), encoding="utf-8"
    )
    assert cli.main(["status", str(tmp_path)]) == 0
    assert "OPTIMAL" in capsys.readouterr().out
