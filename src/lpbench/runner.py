"""Units, scheduling, persistent workers with hard deadlines, checkpoints and progress logs.

Layout of a run directory::

    runs/<study>/manifest.json      config, config hash, provenance, history of starts
    runs/<study>/units/<id>.json    one record per finished unit (atomic write)
    runs/<study>/progress.log       one line per unit: done/total, rate, ETA
"""

from __future__ import annotations

import datetime as dt
import json
import math
import multiprocessing as mp
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from lpbench.config import StudyConfig
from lpbench.instances import Instance, canonical_reference, stratified_subset, suite_instances
from lpbench.solvers.base import EXTERNAL_TIME_LIMIT, EXTERNAL_TL_GRACE_S
from lpbench.worker import THREAD_ENV, pin_threads

FIRST_ORDER_COST = 25.0  # default relative cost of a first-order unit before a pilot exists


@dataclass(frozen=True)
class Unit:
    block: str
    instance: Instance
    k: int
    config: str
    eps: float
    repeat: int = 0

    @property
    def unit_id(self) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", self.instance.name)
        return (
            f"{self.block}__{self.instance.suite}__{name}__p{self.k}__{self.config}"
            f"__e{self.eps:.0e}__r{self.repeat}"
        )


def block_instances(
    cfg: StudyConfig,
    block_suites: tuple[str, ...],
    subset_per_suite: int,
    salt: str,
    pool: list[Instance] | None = None,
    ids: tuple[str, ...] = (),
) -> list[Instance]:
    if ids:
        block_suites = tuple(sorted({i.partition("/")[0] for i in ids}))
    if pool is None:
        pool = []
        for s in block_suites:
            pool.extend(suite_instances(cfg, s))
    else:
        pool = [i for i in pool if i.suite in block_suites]
    if ids:
        by_id = {i.id: i for i in pool}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"instances not in the study: {', '.join(missing)}")
        return [by_id[i] for i in ids]
    if subset_per_suite > 0:
        eligible = [i for i in pool if i.reference is not None]
        return stratified_subset(eligible, subset_per_suite, cfg.seed, salt)
    return pool


def build_units(cfg: StudyConfig, pool: list[Instance] | None = None) -> list[Unit]:
    """All units of a study in a deterministic order (DESIGN 8)."""
    units: list[Unit] = []
    seen: set[str] = set()
    for b in cfg.blocks:
        insts = block_instances(cfg, b.suites, b.subset_per_suite, b.subset_salt, pool, b.instances)
        for inst in insts:
            for k in b.perms:
                for c in b.configs:
                    for r in range(b.repeats):
                        u = Unit(b.name, inst, k, c, b.eps, r)
                        if u.unit_id in seen:
                            raise ValueError(f"duplicate unit {u.unit_id}")
                        seen.add(u.unit_id)
                        units.append(u)
    return units


def expected_cost(u: Unit, factors: dict[str, float] | None = None) -> float:
    f = (factors or {}).get(u.config)
    if f is None:
        f = FIRST_ORDER_COST if u.config.endswith("pdlp") else 1.0
    return f * max(u.instance.nnz, 1)


# --------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return o.name
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats by None (records are strict JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def is_harness_failure(rec: dict[str, Any]) -> bool:
    """True for a record that describes a harness-side failure, not a solver outcome."""
    return rec.get("status") == "ERROR" and str(rec.get("message") or "").startswith(
        "prepare failed"
    )


def load_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    units = Path(run_dir) / "units"
    if not units.exists():
        return out
    for p in sorted(units.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # a torn file cannot exist (atomic writes), but be defensive
        out[rec["unit_id"]] = rec
    return out


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


@dataclass
class _Slot:
    proc: Any
    conn: Any
    task: dict[str, Any] | None = None
    started: float = 0.0


@dataclass
class RunSummary:
    total: int
    skipped: int
    completed: int
    statuses: dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0
    not_run: int = 0  # units whose input could not be prepared (no record written)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Runner:
    def __init__(
        self,
        cfg: StudyConfig,
        run_dir: Path,
        units: list[Unit],
        workers: int | None = None,
        provenance: dict[str, Any] | None = None,
        cost_factors: dict[str, float] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.units = units
        self.workers = min(workers or cfg.workers, cfg.workers, 3)
        self.provenance = provenance or {}
        self.cost_factors = cost_factors
        self.log = log or (lambda s: None)
        self.config_hash = cfg.config_hash()
        self._lp_cache: dict[str, tuple[float, bool]] = {}

    # -- helpers ------------------------------------------------------------------------------

    def _unit_path(self, uid: str) -> Path:
        return self.run_dir / "units" / f"{uid}.json"

    def _done_ids(self) -> set[str]:
        """Units with a record for the current config hash.

        Harness-side failures are not results: a record of a unit whose input could not be
        prepared (written by versions before 0.1.1) does not count as done, so a rerun after
        fixing the data runs it.
        """
        done = set()
        for uid, rec in load_records(self.run_dir).items():
            if rec.get("config_hash") == self.config_hash and not is_harness_failure(rec):
                done.add(uid)
        return done

    def _reference(self, u: Unit) -> float | None:
        if u.instance.reference is None:
            return None
        key = u.instance.id
        if key not in self._lp_cache:
            from lpbench.lpmodel import load_npz

            lp = load_npz(u.instance.canonical_path())
            self._lp_cache[key] = (lp.c0, lp.maximize_flipped)
        c0, flipped = self._lp_cache[key]
        return canonical_reference(u.instance, c0, flipped)

    def _task(self, u: Unit) -> dict[str, Any]:
        from lpbench.prepare import ensure_mps

        mps = ensure_mps(u.instance, u.k, self.cfg.seed)
        return {
            "unit_id": u.unit_id,
            "config": u.config,
            "mps_path": str(mps.resolve()),
            "canonical_path": str(u.instance.canonical_path().resolve()),
            "eps": u.eps,
            "time_limit_s": self.cfg.time_limit_s,
            "seed": self.cfg.seed,
            "perm_seed": self.cfg.seed,
            "k": u.k,
            "name": u.instance.name,
            "reference": self._reference(u),
        }

    def _base_record(self, u: Unit) -> dict[str, Any]:
        return {
            "unit_id": u.unit_id,
            "block": u.block,
            "instance": u.instance.id,
            "suite": u.instance.suite,
            "name": u.instance.name,
            "nnz_published": u.instance.nnz,
            "k": u.k,
            "config": u.config,
            "eps": u.eps,
            "repeat": u.repeat,
            "seed": self.cfg.seed,
            "time_limit_s": self.cfg.time_limit_s,
            "hard_deadline_s": self.cfg.hard_deadline_s,
            "config_hash": self.config_hash,
            "git": self.provenance.get("git"),
            "reference_source": u.instance.reference_source or None,
        }

    def _write_manifest(self, total: int, todo: int) -> None:
        path = self.run_dir / "manifest.json"
        history: list[dict[str, Any]] = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
            except json.JSONDecodeError:
                history = []
        history.append(
            {
                "started": utc_now(),
                "units_total": total,
                "units_todo": todo,
                "workers": self.workers,
                "git": self.provenance.get("git"),
            }
        )
        atomic_write_json(
            path,
            _sanitize(
                {
                    "study": self.cfg.name,
                    "config": self.cfg.raw,
                    "config_hash": self.config_hash,
                    "provenance": self.provenance,
                    "history": history,
                }
            ),
        )

    def _spawn(self, ctx: Any) -> _Slot:
        from lpbench.worker import worker_main

        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=worker_main, args=(child,), daemon=True)
        proc.start()
        child.close()
        return _Slot(proc=proc, conn=parent)

    @staticmethod
    def _await_ready(slot: _Slot, timeout_s: float = 300.0) -> None:
        """Block until the worker has imported the solver libraries."""
        if not slot.conn.poll(timeout_s):
            raise RuntimeError("worker did not start within the startup timeout")
        msg = slot.conn.recv()
        if not (isinstance(msg, dict) and "ready" in msg):
            raise RuntimeError(f"unexpected worker greeting: {msg!r}")

    def _spawn_ready(self, ctx: Any) -> _Slot:
        slot = self._spawn(ctx)
        self._await_ready(slot)
        return slot

    # -- main loop ----------------------------------------------------------------------------

    def _deadline(self, slot: Any) -> float:
        """Seconds after dispatch at which the slot's worker is killed."""
        cfg_id = slot.task.get("config") if slot.task else None
        if cfg_id in EXTERNAL_TIME_LIMIT:
            return min(self.cfg.hard_deadline_s, self.cfg.time_limit_s + EXTERNAL_TL_GRACE_S)
        return self.cfg.hard_deadline_s

    def run(self, max_units: int | None = None) -> RunSummary:
        t_start = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for stale in (
            (self.run_dir / "units").glob(".*.tmp") if (self.run_dir / "units").exists() else []
        ):
            stale.unlink(missing_ok=True)
        done = self._done_ids()
        todo = [u for u in self.units if u.unit_id not in done]
        todo.sort(key=lambda u: -expected_cost(u, self.cost_factors))
        if max_units is not None:
            todo = todo[:max_units]
        total = len(self.units)
        self._write_manifest(total, len(todo))
        summary = RunSummary(total=total, skipped=len(done), completed=0)
        self.log(
            f"{utc_now()} start: {len(done)}/{total} done, {len(todo)} to run, "
            f"{self.workers} workers, config {self.config_hash}"
        )
        if not todo:
            return summary

        pin_threads()  # inherited by spawned workers
        for key, value in THREAD_ENV.items():
            os.environ[key] = value
        ctx = mp.get_context("spawn")
        slots = [self._spawn(ctx) for _ in range(min(self.workers, len(todo)))]
        for s in slots:
            self._await_ready(s)
        queue = list(todo)
        by_id = {u.unit_id: u for u in todo}
        n_done = len(done)
        t_rate0 = time.perf_counter()
        n_this_run = 0

        def finish(slot: _Slot, result: dict[str, Any]) -> None:
            nonlocal n_done, n_this_run
            assert slot.task is not None
            u = by_id[slot.task["unit_id"]]
            rec = self._base_record(u)
            rec.update(result)
            rec["unit_wall_s"] = time.perf_counter() - slot.started
            rec["reference"] = slot.task.get("reference")
            rec["finished_at"] = utc_now()
            atomic_write_json(self._unit_path(u.unit_id), _sanitize(rec))
            slot.task = None
            n_done += 1
            n_this_run += 1
            st = str(rec.get("status"))
            summary.statuses[st] = summary.statuses.get(st, 0) + 1
            summary.completed += 1
            elapsed = time.perf_counter() - t_rate0
            rate = n_this_run / elapsed * 60 if elapsed > 0 else 0.0
            remaining = total - n_done
            eta = remaining / (rate / 60) if rate > 0 else math.inf
            eta_s = f"{eta / 60:.1f} min" if math.isfinite(eta) else "?"
            self.log(
                f"{utc_now()} {n_done}/{total} {u.unit_id} {st} "
                f"{rec.get('solve_wall_s') or 0:.3f}s rate={rate:.1f}/min eta={eta_s}"
            )

        try:
            while queue or any(s.task is not None for s in slots):
                # hand out work
                for s in slots:
                    if s.task is None and queue:
                        u = queue.pop(0)
                        try:
                            task = self._task(u)
                        except Exception as exc:  # missing files etc.: not a unit result
                            summary.not_run += 1
                            self.log(
                                f"{utc_now()} {u.unit_id} NOT RUN (prepare failed: "
                                f"{type(exc).__name__}: {exc}); no record written"
                            )
                            continue
                        s.task, s.started = task, time.perf_counter()
                        s.conn.send(task)
                busy = [s for s in slots if s.task is not None]
                if not busy:
                    continue
                now = time.perf_counter()
                next_deadline = min(s.started + self._deadline(s) for s in busy)
                timeout = max(0.0, min(next_deadline - now, 1.0))
                ready = wait([s.conn for s in busy] + [s.proc.sentinel for s in busy], timeout)
                for i, s in enumerate(slots):
                    if s.task is None:
                        continue
                    if s.conn in ready:
                        try:
                            result = s.conn.recv()
                        except (EOFError, OSError):
                            result = None
                        if result is not None:
                            finish(s, result)
                            continue
                    if not s.proc.is_alive():
                        code = s.proc.exitcode
                        finish(
                            s,
                            {
                                "status": "ERROR",
                                "raw_status": "worker died",
                                "message": f"worker exited with code {code}",
                                "solve_wall_s": time.perf_counter() - s.started,
                            },
                        )
                        slots[i] = self._spawn_ready(ctx)
                        continue
                    limit = self._deadline(s)
                    if time.perf_counter() - s.started > limit:
                        s.proc.kill()
                        s.proc.join(timeout=10)
                        external = limit < self.cfg.hard_deadline_s
                        finish(
                            s,
                            {
                                "status": "TIME_LIMIT" if external else "KILLED",
                                "raw_status": "external time limit"
                                if external
                                else "hard deadline",
                                "message": f"killed after {limit:.1f} s",
                                "solve_wall_s": time.perf_counter() - s.started,
                            },
                        )
                        slots[i] = self._spawn_ready(ctx)
        finally:
            for s in slots:
                try:
                    if s.proc.is_alive():
                        s.conn.send(None)
                except (OSError, BrokenPipeError):
                    pass
            for s in slots:
                s.proc.join(timeout=5)
                if s.proc.is_alive():
                    s.proc.kill()
                    s.proc.join(timeout=5)
        summary.wall_s = time.perf_counter() - t_start
        self.log(
            f"{utc_now()} finished: {summary.completed} units this run in "
            f"{summary.wall_s:.0f} s; {n_done}/{total} done"
            + (f"; {summary.not_run} not run (prepare failed)" if summary.not_run else "")
        )
        return summary


def file_logger(path: Path, echo: bool = True) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(msg + "\n")
        if echo:
            print(msg, flush=True)

    return log


def status_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[str(r.get("status"))] = out.get(str(r.get("status")), 0) + 1
    return dict(sorted(out.items()))
