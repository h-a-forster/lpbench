"""Study configuration (configs/*.toml): parsing, validation and a stable hash."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_IDS = (
    "highs-dsimplex",
    "highs-ipx",
    "highs-pdlp",
    "highs-hipdlp",
    "ortools-glop",
    "ortools-pdlp",
    "scip-soplex",
)
SUITES = ("netlib", "miplib")
MAX_WORKERS = 3
MAX_DOWNLOAD_BYTES = 2_000_000_000

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Block:
    """One block of units: instances x configs x eps x permutations x repeats."""

    name: str
    configs: tuple[str, ...]
    eps: float
    perms: tuple[int, ...]
    suites: tuple[str, ...] = ()
    subset_per_suite: int = 0  # 0: all instances of the suites; >0: stratified subset
    subset_strata: str = "nnz_tertiles"
    subset_salt: str = ""  # distinguishes subsets of different blocks
    repeats: int = 1

    @property
    def is_subset(self) -> bool:
        return self.subset_per_suite > 0


@dataclass(frozen=True)
class StudyConfig:
    name: str
    workers: int
    time_limit_s: float
    hard_deadline_factor: float
    hard_deadline_extra_s: float
    download_cap_bytes: int
    seed: int
    configs: tuple[str, ...]
    first_order: tuple[str, ...]
    verify_eps: tuple[float, ...]
    netlib_exclude: tuple[str, ...]
    miplib_list: str
    miplib_max_nnz: int
    blocks: tuple[Block, ...]
    bootstrap_resamples: int
    shift_s: float
    shift_sensitivity_s: tuple[float, ...]
    penalty: str
    max_projected_wall_min: float = 45.0
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def hard_deadline_s(self) -> float:
        return self.time_limit_s * self.hard_deadline_factor + self.hard_deadline_extra_s

    def config_hash(self) -> str:
        """SHA-256 (first 16 hex digits) of the canonical JSON of the raw config."""
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def suites_used(self) -> tuple[str, ...]:
        out: list[str] = []
        for b in self.blocks:
            for s in b.suites or SUITES:
                if s not in out:
                    out.append(s)
        return tuple(out)


def _resolve_configs(value: Any, all_ids: tuple[str, ...], first_order: tuple[str, ...]) -> tuple:
    if value == "all":
        return all_ids
    if value == "first_order":
        return first_order
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        unknown = sorted(set(value) - set(all_ids))
        if unknown:
            raise ConfigError(f"unknown configs in block: {unknown}")
        return tuple(value)
    raise ConfigError(f"bad configs value {value!r}")


def parse_config(raw: dict[str, Any]) -> StudyConfig:
    try:
        configs = tuple(raw["configs"])
        unknown = sorted(set(configs) - set(CONFIG_IDS))
        if unknown:
            raise ConfigError(f"unknown config ids: {unknown}")
        if len(set(configs)) != len(configs):
            raise ConfigError("duplicate config ids")
        first_order = tuple(raw.get("first_order", []))
        if not set(first_order) <= set(configs):
            raise ConfigError("first_order must be a subset of configs")
        workers = int(raw["workers"])
        if not 1 <= workers <= MAX_WORKERS:
            raise ConfigError(f"workers must be between 1 and {MAX_WORKERS}")
        cap = int(raw["download_cap_bytes"])
        if not 0 < cap <= MAX_DOWNLOAD_BYTES:
            raise ConfigError(f"download_cap_bytes must be in (0, {MAX_DOWNLOAD_BYTES}]")
        tl = float(raw["time_limit_s"])
        if tl <= 0:
            raise ConfigError("time_limit_s must be positive")
        blocks: list[Block] = []
        for b in raw["blocks"]:
            subset = b.get("subset", {})
            suites = tuple(b.get("suites", SUITES if not subset else SUITES))
            for s in suites:
                if s not in SUITES:
                    raise ConfigError(f"unknown suite {s!r}")
            perms = tuple(int(k) for k in b.get("perms", [0]))
            if any(k < 0 for k in perms):
                raise ConfigError("permutation indices must be >= 0")
            eps = float(b["eps"])
            if not 0 < eps < 1:
                raise ConfigError("eps must be in (0, 1)")
            blocks.append(
                Block(
                    name=str(b["name"]),
                    configs=_resolve_configs(b.get("configs", "all"), configs, first_order),
                    eps=eps,
                    perms=perms,
                    suites=suites,
                    subset_per_suite=int(subset.get("per_suite", 0)),
                    subset_strata=str(subset.get("strata", "nnz_tertiles")),
                    subset_salt=str(subset.get("salt", b["name"])),
                    repeats=int(b.get("repeats", 1)),
                )
            )
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise ConfigError("duplicate block names")
        stats = raw.get("stats", {})
        netlib = raw.get("suites", {}).get("netlib", {})
        miplib = raw.get("suites", {}).get("miplib", {})
        cfg = StudyConfig(
            name=str(raw["name"]),
            workers=workers,
            time_limit_s=tl,
            hard_deadline_factor=float(raw.get("hard_deadline_factor", 1.5)),
            hard_deadline_extra_s=float(raw.get("hard_deadline_extra_s", 10.0)),
            download_cap_bytes=cap,
            seed=int(raw["seed"]),
            configs=configs,
            first_order=first_order,
            verify_eps=tuple(float(e) for e in raw.get("verify_eps", [1e-4, 1e-6, 1e-8])),
            netlib_exclude=tuple(s.upper() for s in netlib.get("exclude", [])),
            miplib_list=str(miplib.get("list", "benchmark-v2")),
            miplib_max_nnz=int(miplib.get("max_nnz", 100_000)),
            blocks=tuple(blocks),
            bootstrap_resamples=int(stats.get("bootstrap_resamples", 10_000)),
            shift_s=float(stats.get("shift_s", 1.0)),
            shift_sensitivity_s=tuple(float(s) for s in stats.get("shift_sensitivity_s", [])),
            penalty=str(stats.get("penalty", "par1")),
            max_projected_wall_min=float(raw.get("max_projected_wall_min", 45.0)),
            raw=raw,
        )
    except KeyError as exc:
        raise ConfigError(f"missing config key {exc}") from exc
    if cfg.penalty not in ("par1", "par2"):
        raise ConfigError("penalty must be par1 or par2")
    return cfg


def load_config(name_or_path: str | Path) -> StudyConfig:
    """Load ``configs/<name>.toml`` or an explicit path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = ROOT / "configs" / f"{name_or_path}.toml"
    if not p.exists():
        raise ConfigError(f"config not found: {p.name}")
    with open(p, "rb") as f:
        return parse_config(tomllib.load(f))
