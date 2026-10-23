"""Instance suites, committed manifests, cache layout and the stratified subset draw."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lpbench.config import StudyConfig

DATA = Path(__file__).resolve().parent / "data"
NETLIB_REFERENCE = DATA / "netlib_reference.csv"
MIPLIB_MANIFEST = DATA / "miplib_benchmark.csv"
MIPLIB_REFERENCE = DATA / "miplib_reference.csv"
CHECKSUMS = DATA / "checksums.csv"

NETLIB_SITE = "https://netlib.org/lp/data"
MIPLIB_SITE = "https://miplib.zib.de"


def cache_root() -> Path:
    """Cache directory (``LPBENCH_CACHE`` or ``./cache``)."""
    return Path(os.environ.get("LPBENCH_CACHE", "cache"))


@dataclass(frozen=True)
class Instance:
    suite: str
    name: str
    file: str  # file name at the source
    url: str
    rows: int  # as published (netlib: including the cost row)
    cols: int
    nnz: int  # as published (netlib: including the cost row)
    reference: float | None = None  # published optimum (source sense), if any
    reference_source: str = ""

    @property
    def id(self) -> str:
        return f"{self.suite}/{self.name}"

    def raw_path(self, root: Path | None = None) -> Path:
        return (root or cache_root()) / "raw" / self.suite / self.file

    def canonical_path(self, root: Path | None = None) -> Path:
        return (root or cache_root()) / "canonical" / self.suite / f"{self.name}.npz"

    def mps_path(self, k: int, root: Path | None = None) -> Path:
        return (root or cache_root()) / "mps" / self.suite / f"{self.name}__p{k}.mps"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def netlib_file_name(name: str) -> str:
    return name.lower()


def netlib_instances(exclude: tuple[str, ...] = ()) -> list[Instance]:
    """Netlib problems with a compressed file and a reference value, minus ``exclude``."""
    out = []
    excl = {e.upper() for e in exclude}
    for r in _read_csv(NETLIB_REFERENCE):
        name = r["name"]
        if name in excl or not r["readme_bytes"]:
            continue
        if r["koch_value"]:
            ref, src = float(r["koch_value"]), "koch2004"
        elif r["readme_value"]:
            ref, src = float(r["readme_value"]), "netlib_readme"
        else:
            ref, src = None, ""
        f = netlib_file_name(name)
        out.append(
            Instance(
                suite="netlib",
                name=name,
                file=f,
                url=f"{NETLIB_SITE}/{f}",
                rows=int(r["rows"]),
                cols=int(r["cols"]),
                nnz=int(r["nnz"]),
                reference=ref,
                reference_source=src,
            )
        )
    return out


def miplib_reference_values() -> dict[str, tuple[float, str]]:
    """Consensus LP-relaxation references (canonical sense) by instance name."""
    out = {}
    for r in _read_csv(MIPLIB_REFERENCE):
        if r.get("reference"):
            out[r["name"]] = (float(r["reference"]), r.get("agreed", ""))
    return out


def miplib_instances(max_nnz: int) -> list[Instance]:
    refs = miplib_reference_values()
    out = []
    for r in _read_csv(MIPLIB_MANIFEST):
        if int(r["nnz"]) > max_nnz:
            continue
        name = r["name"]
        ref = refs.get(name)
        out.append(
            Instance(
                suite="miplib",
                name=name,
                file=f"{name}.mps.gz",
                url=f"{MIPLIB_SITE}/WebData/instances/{name}.mps.gz",
                rows=int(r["rows"]),
                cols=int(r["cols"]),
                nnz=int(r["nnz"]),
                reference=ref[0] if ref else None,
                reference_source="consensus" if ref else "",
            )
        )
    return out


def canonical_reference(inst: Instance, c0: float, maximize_flipped: bool) -> float | None:
    """Reference objective in the canonical (minimisation, constant-included) sense.

    Netlib's published optima (readme and Koch 2004) exclude the objective-row RHS constant
    (only E226 has one: HiGHS reads it as c0 = 7.113), so ``c0`` is added. MIPLIB consensus
    values are computed and stored in the canonical sense already.
    """
    if inst.reference is None:
        return None
    if inst.suite == "netlib":
        value = -inst.reference if maximize_flipped else inst.reference
        return value + c0
    return inst.reference


def suite_instances(cfg: StudyConfig, suite: str) -> list[Instance]:
    if suite == "netlib":
        return netlib_instances(cfg.netlib_exclude)
    if suite == "miplib":
        return miplib_instances(cfg.miplib_max_nnz)
    raise ValueError(f"unknown suite {suite!r}")


def all_instances(cfg: StudyConfig) -> list[Instance]:
    out: list[Instance] = []
    for s in cfg.suites_used():
        out.extend(suite_instances(cfg, s))
    return out


# --------------------------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------------------------


def load_checksums() -> dict[str, tuple[int, str]]:
    """``suite/file -> (bytes, sha256)`` from the committed checksum manifest."""
    return {
        f"{r['suite']}/{r['file']}": (int(r["bytes"]), r["sha256"]) for r in _read_csv(CHECKSUMS)
    }


def save_checksums(entries: dict[str, tuple[int, str]]) -> None:
    rows = []
    for key in sorted(entries):
        suite, file = key.split("/", 1)
        nbytes, digest = entries[key]
        rows.append({"suite": suite, "file": file, "bytes": nbytes, "sha256": digest})
    tmp = CHECKSUMS.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "file", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CHECKSUMS)


# --------------------------------------------------------------------------------------------
# Stratified subset (DESIGN 8)
# --------------------------------------------------------------------------------------------


def _stable_int(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def stratified_subset(
    instances: list[Instance], per_suite: int, seed: int, salt: str = ""
) -> list[Instance]:
    """Seeded draw of ``per_suite`` instances per suite, spread evenly over nnz tertiles.

    Deterministic given (instances, per_suite, seed, salt); the order of ``instances`` does
    not matter. Instances should already exclude those without a usable reference.
    """
    out: list[Instance] = []
    suites = sorted({i.suite for i in instances})
    for suite in suites:
        pool = sorted((i for i in instances if i.suite == suite), key=lambda i: (i.nnz, i.name))
        if len(pool) <= per_suite:
            out.extend(pool)
            continue
        strata = np.array_split(np.arange(len(pool)), 3)
        quotas = [per_suite // 3] * 3
        for j in range(per_suite - sum(quotas)):
            quotas[j] += 1
        rng = np.random.default_rng(_stable_int(seed, suite, salt))
        for stratum, q in zip(strata, quotas, strict=True):
            pick = rng.choice(stratum, size=min(q, len(stratum)), replace=False)
            out.extend(pool[int(p)] for p in sorted(pick))
    return out


def nnz_tertile(instance: Instance, instances: list[Instance]) -> int:
    """0, 1 or 2: size stratum of an instance within its suite (by published nnz)."""
    pool = sorted(i.nnz for i in instances if i.suite == instance.suite)
    if not pool:
        return 0
    rank = sum(1 for v in pool if v < instance.nnz)
    return min(2, math.floor(3 * rank / len(pool)))
