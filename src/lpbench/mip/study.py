"""MIP claim studies: config, units, run (lpbench's runner), check and report (DESIGN 13).

Layout of a run directory (in addition to the runner's ``manifest.json``, ``units/`` and
``progress.log``)::

    runs/<study>/check/<unit id>.json   exact checks of one unit (atomic, skipped on rerun)
    runs/<study>/claims.jsonl           one classified claim record per unit
    runs/<study>/report/summary.{md,json}
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lpbench.config import MAX_WORKERS, ConfigError
from lpbench.instances import Instance, cache_root, load_checksums, miplib_instances
from lpbench.mip.adapters import FAKES, SOLVERS
from lpbench.mip.complete import COMPLETION_BUDGET_S
from lpbench.runner import Runner, atomic_write_json, load_records

_KEYS = frozenset(
    {
        "name",
        "workers",
        "time_limit_s",
        "hard_deadline_factor",
        "hard_deadline_extra_s",
        "seed",
        "solvers",
        "instances",
        "completion_budget_s",
    }
)


# --------------------------------------------------------------------------------------------
# Config and units
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MipConfig:
    name: str
    workers: int
    time_limit_s: float
    seed: int
    solvers: tuple[str, ...]
    instances: tuple[str, ...]
    hard_deadline_factor: float = 1.5
    hard_deadline_extra_s: float = 10.0
    completion_budget_s: float = COMPLETION_BUDGET_S
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def hard_deadline_s(self) -> float:
        return self.time_limit_s * self.hard_deadline_factor + self.hard_deadline_extra_s

    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def parse_mip_config(raw: dict[str, Any]) -> MipConfig:
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ConfigError(f"unknown keys in the MIP config: {unknown}")
    try:
        solvers = tuple(str(s) for s in raw["solvers"])
        bad = sorted(set(solvers) - set(SOLVERS) - set(FAKES))
        if bad:
            raise ConfigError(f"unknown MIP solvers: {bad} (known: {list(SOLVERS)})")
        if not solvers or len(set(solvers)) != len(solvers):
            raise ConfigError("solvers must be a non-empty list without duplicates")
        names = tuple(str(n) for n in raw["instances"])
        if not names or len(set(names)) != len(names):
            raise ConfigError("instances must be a non-empty list without duplicates")
        known = {i.name for i in miplib_instances(10**12)}
        missing = [n for n in names if n not in known]
        if missing:
            raise ConfigError(f"not MIPLIB 2017 benchmark instances: {missing}")
        workers = int(raw["workers"])
        if not 1 <= workers <= MAX_WORKERS:
            raise ConfigError(f"workers must be between 1 and {MAX_WORKERS}")
        tl = float(raw["time_limit_s"])
        budget = float(raw.get("completion_budget_s", COMPLETION_BUDGET_S))
        if tl <= 0 or budget <= 0:
            raise ConfigError("time_limit_s and completion_budget_s must be positive")
        return MipConfig(
            name=str(raw["name"]),
            workers=workers,
            time_limit_s=tl,
            seed=int(raw["seed"]),
            solvers=solvers,
            instances=names,
            hard_deadline_factor=float(raw.get("hard_deadline_factor", 1.5)),
            hard_deadline_extra_s=float(raw.get("hard_deadline_extra_s", 10.0)),
            completion_budget_s=budget,
            raw=raw,
        )
    except KeyError as exc:
        raise ConfigError(f"missing MIP config key {exc}") from exc


def load_mip_config(path: str | Path) -> MipConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p.name}")
    with open(p, "rb") as f:
        return parse_mip_config(tomllib.load(f))


@dataclass(frozen=True)
class MipUnit:
    instance: Instance
    solver: str
    seed: int

    @property
    def config(self) -> str:
        """Runner config id; its prefix routes the unit to the MIP worker path."""
        return f"mip-{self.solver}"

    @property
    def unit_id(self) -> str:
        return f"mip__{self.instance.name}__{self.solver}__s{self.seed}"


def build_mip_units(cfg: MipConfig) -> list[MipUnit]:
    by_name = {i.name: i for i in miplib_instances(10**12)}
    return [MipUnit(by_name[n], s, cfg.seed) for n in cfg.instances for s in cfg.solvers]


def mip_mps_path(inst: Instance, root: Path | None = None) -> Path:
    return (root or cache_root()) / "mip" / f"{inst.name}.mps"


def ensure_mip_mps(inst: Instance, root: Path | None = None) -> Path:
    """The decompressed MPS file of a fetched instance, after checking the download's
    SHA-256 against the committed manifest (``lpbench fetch`` records it)."""
    from lpbench.download import sha256_file

    path = mip_mps_path(inst, root)
    if path.exists():
        return path
    raw = inst.raw_path(root)
    expected = load_checksums().get(f"{inst.suite}/{inst.file}")
    if expected is None:
        raise FileNotFoundError(f"no checksum recorded for {inst.file}; run lpbench fetch")
    if not raw.exists():
        raise FileNotFoundError(f"{inst.file} is not fetched; run lpbench fetch")
    if sha256_file(raw) != expected[1]:
        raise ValueError(f"{inst.file}: SHA-256 differs from the manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(gzip.decompress(raw.read_bytes()))
    os.replace(tmp, path)
    return path


class MipRunner(Runner):
    """lpbench's runner (persistent workers, hard deadlines, atomic records, resume) with
    MIP tasks and records."""

    def _task(self, u: Any) -> dict[str, Any]:
        mps = ensure_mip_mps(u.instance)
        return {
            "kind": "mip",
            "unit_id": u.unit_id,
            "config": u.config,
            "solver": u.solver,
            "mps_path": str(mps.resolve()),
            "time_limit_s": self.cfg.time_limit_s,
            "seed": u.seed,
            "name": u.instance.name,
        }

    def _base_record(self, u: Any) -> dict[str, Any]:
        return {
            "unit_id": u.unit_id,
            "instance": u.instance.name,
            "solver": u.solver,
            "config": u.config,
            "nnz_published": u.instance.nnz,
            "seed": u.seed,
            "time_limit_s": self.cfg.time_limit_s,
            "hard_deadline_s": self.cfg.hard_deadline_s,
            "config_hash": self.config_hash,
            "git": self.provenance.get("git"),
        }


# --------------------------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------------------------


def check_unit(rec: dict[str, Any], mps_path: str, budget_s: float) -> dict[str, Any]:
    """Exact checks of one unit record: primal class and completion on the exact data and
    on the float64 data (the latter only when rounding changed the data)."""
    from lpbench.mip.check import classify, solution_vector
    from lpbench.mip.complete import complete
    from lpbench.mip.mps import read_mps

    t0 = time.perf_counter()
    model = read_mps(mps_path)
    fmodel = model.float_data()
    rounded = not model.same_data(fmodel)
    out: dict[str, Any] = {
        "unit_id": rec["unit_id"],
        "unit_finished_at": rec.get("finished_at"),
        "instance": rec["instance"],
        "solver": rec["solver"],
        "maximize": model.maximize,
        "pure_integer": model.is_pure_integer(),
        "data_rounded": rounded,
        "has_solution": rec.get("x") is not None,
    }
    x_names = rec.get("x")
    if x_names is None:
        out.update(
            primal_class=None,
            primal_class_float=None,
            completion={"outcome": "none"},
            completion_float={"outcome": "none"},
        )
    else:
        cls = classify(
            model, x_names, rec["solver"], rec.get("tolerances") or {}, rec.get("primal_bound")
        )
        out.update(cls)
        x = solution_vector(model, x_names)
        comp = complete(model, x, budget_s, float_model=fmodel).to_dict()
        out["completion"] = comp
        if rounded:
            fcls = classify(
                fmodel, x_names, rec["solver"], rec.get("tolerances") or {}, rec.get("primal_bound")
            )
            out["primal_class_float"] = fcls["primal_class"]
            out["completion_float"] = complete(fmodel, x, budget_s, float_model=fmodel).to_dict()
        else:
            out["primal_class_float"] = out["primal_class"]
            out["completion_float"] = comp
    out["check_s"] = time.perf_counter() - t0
    return out


def _check_child(args: tuple[dict[str, Any], str, float, str]) -> str:
    rec, mps, budget, path = args
    try:
        res = check_unit(rec, mps, budget)
    except Exception as exc:  # recorded, never dropped
        res = {
            "unit_id": rec["unit_id"],
            "unit_finished_at": rec.get("finished_at"),
            "instance": rec["instance"],
            "solver": rec["solver"],
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }
    atomic_write_json(Path(path), res)
    return rec["unit_id"]


def run_checks(
    run_dir: Path,
    workers: int = 3,
    budget_s: float | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Check every unit record (``workers`` processes); skip units already checked."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cfg = run_config(run_dir)
    budget = budget_s if budget_s is not None else cfg.completion_budget_s
    recs = load_records(run_dir)
    by_name = {i.name: i for i in miplib_instances(10**12)}
    check_dir = run_dir / "check"
    todo = []
    for uid, rec in sorted(recs.items()):
        path = check_dir / f"{uid}.json"
        if path.exists():
            prev = json.loads(path.read_text(encoding="utf-8"))
            if prev.get("unit_finished_at") == rec.get("finished_at") and "error" not in prev:
                continue
        mps = str(ensure_mip_mps(by_name[rec["instance"]]))
        todo.append((rec, mps, budget, str(path)))
    log(f"check: {len(recs) - len(todo)} done, {len(todo)} to do")
    if todo:
        from lpbench.worker import pin_threads

        pin_threads()
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=min(workers, MAX_WORKERS), mp_context=mp.get_context("spawn")
        ) as ex:
            futs = [ex.submit(_check_child, t) for t in todo]
            for n, f in enumerate(as_completed(futs), 1):
                log(f"check: {n}/{len(todo)} {f.result()}")
    return [
        json.loads((check_dir / f"{uid}.json").read_text(encoding="utf-8")) for uid in sorted(recs)
    ]


def run_config(run_dir: Path) -> MipConfig:
    man = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    return parse_mip_config(man["config"])


def assemble_claims(
    recs: dict[str, dict[str, Any]], checks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One claim record per unit: checks, witnesses and refutations (minimisation sense)."""
    from lpbench.mip.claims import dual_claim, refutation, witnesses
    from lpbench.mip.complete import q_str

    w_exact = witnesses(checks, "completion")
    w_float = witnesses(checks, "completion_float")
    out = []
    for chk in checks:
        rec = recs[chk["unit_id"]]
        maximize = bool(chk.get("maximize"))
        d = dual_claim(rec, maximize) if "error" not in chk else None
        ref = refutation(d, w_exact.get(chk["instance"]))
        ref_f = refutation(d, w_float.get(chk["instance"]))
        comp = (chk.get("completion") or {}).get("outcome") if "error" not in chk else "unknown"
        comp_f = (chk.get("completion_float") or {}).get("outcome")
        flips = []
        if chk.get("primal_class") != chk.get("primal_class_float"):
            flips.append("primal_class")
        if comp != comp_f and "error" not in chk:
            flips.append("completion")
        if (ref or {}).get("bins") != (ref_f or {}).get("bins"):
            flips.append("refuted_bins")
        w = w_exact.get(chk["instance"])
        out.append(
            {
                "unit_id": chk["unit_id"],
                "solver": chk["solver"],
                "instance": chk["instance"],
                "status": rec.get("status"),
                "raw_status": rec.get("raw_status"),
                "maximize": maximize,
                "has_solution": rec.get("x") is not None,
                "primal_class": chk.get("primal_class"),
                "primal_class_float": chk.get("primal_class_float"),
                "violations": chk.get("violations"),
                "own_tolerance_model": chk.get("own_tolerance_model"),
                "objective_ok": chk.get("objective_ok"),
                "objective_rel_diff": chk.get("objective_rel_diff"),
                "completion": comp,
                "completion_detail": chk.get("completion"),
                "completion_float": comp_f,
                "witness": q_str(w),
                "witness_float": None if w is None else int(w.numerator) / int(w.denominator),
                "dual_claim": (None if d is None else "infeasible" if d == math.inf else q_str(d)),
                "refutation_severity": None if ref is None else ref["severity"],
                "refuted_bins": None if ref is None else ref["bins"],
                "refuted_bins_float": None if ref_f is None else ref_f["bins"],
                "data_rounding_flips": flips,
                "tolerances": rec.get("tolerances") or {},
                "missing": rec.get("missing", list(_ALL_FIELDS)),
                "primal_bound": rec.get("primal_bound"),
                "dual_bound": rec.get("dual_bound"),
                "gap": rec.get("gap"),
                "solve_wall_s": rec.get("solve_wall_s"),
                "check_error": chk.get("error"),
            }
        )
    return out


_ALL_FIELDS = ("x", "primal_bound", "dual_bound", "gap")


def write_claims(run_dir: Path, claims: list[dict[str, Any]]) -> Path:
    path = Path(run_dir) / "claims.jsonl"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for c in claims:
            f.write(json.dumps(c, sort_keys=True, allow_nan=False, default=_inf_json) + "\n")
    os.replace(tmp, path)
    return path


def _inf_json(o: Any) -> Any:
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def check_run(run_dir: Path, workers: int = 3, log: Callable[[str], None] = print) -> Path:
    from lpbench.runner import _sanitize

    run_dir = Path(run_dir)
    checks = run_checks(run_dir, workers=workers, log=log)
    claims = _sanitize(assemble_claims(load_records(run_dir), checks))
    path = write_claims(run_dir, claims)
    log(f"check: {len(claims)} claim records in {path.name}")
    return path
