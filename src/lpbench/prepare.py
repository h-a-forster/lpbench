"""``lpbench fetch`` and ``lpbench prepare``: download raw files, build canonical LPs and MPS."""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from lpbench.config import StudyConfig
from lpbench.download import Budget, download
from lpbench.instances import (
    Instance,
    all_instances,
    cache_root,
    load_checksums,
    save_checksums,
)
from lpbench.lpmodel import (
    CanonicalLP,
    load_npz,
    make_permutation,
    permute,
    read_with_highs,
    save_npz,
    write_mps,
)
from lpbench.netlib_emps import expand

Log = Callable[[str], None]


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class FetchSummary:
    files: int
    downloaded: int
    bytes_downloaded: int
    bytes_total: int
    new_checksums: int


def fetch(
    instances: Iterable[Instance],
    cap_bytes: int,
    update_checksums: bool = False,
    log: Log = _stderr,
) -> FetchSummary:
    """Download every instance's raw file (or reuse a verified cached copy)."""
    sums = load_checksums()
    budget = Budget(cap_bytes=cap_bytes)
    n = downloaded = total = new = 0
    insts = list(instances)
    for inst in insts:
        key = f"{inst.suite}/{inst.file}"
        expected = None if update_checksums else sums.get(key, (0, None))[1]
        digest, nbytes, did = download(inst.url, inst.raw_path(), budget, expected)
        n += 1
        total += nbytes
        if did:
            downloaded += 1
        if sums.get(key) != (nbytes, digest):
            sums[key] = (nbytes, digest)
            new += 1
            save_checksums(sums)  # record as we go
        if did:
            log(f"[fetch] {n}/{len(insts)} {inst.id} {nbytes} bytes")
    log(
        f"[fetch] {n} files, {downloaded} downloaded ({budget.used_bytes:,} bytes this run, "
        f"cap {cap_bytes:,}); {total:,} bytes on disk; {new} checksums recorded"
    )
    return FetchSummary(n, downloaded, budget.used_bytes, total, new)


def expanded_netlib_path(inst: Instance, root: Path | None = None) -> Path:
    return (root or cache_root()) / "expanded" / "netlib" / f"{inst.file}.mps"


def read_raw(inst: Instance, root: Path | None = None) -> tuple[CanonicalLP, dict[str, object]]:
    """Read a raw downloaded file into a CanonicalLP (the problem definition)."""
    raw = inst.raw_path(root)
    info: dict[str, object] = {}
    if inst.suite == "netlib":
        res = expand(raw.read_text(encoding="latin-1"))
        out = expanded_netlib_path(inst, root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.mps, encoding="latin-1", newline="\n")
        lp = read_with_highs(out, inst.name, fixed_format=True)
        info.update(
            checked_lines=res.checked_lines,
            checksum_lines=res.checksum_lines,
            stats_rows=res.stats.nrow,
            stats_cols=res.stats.ncol,
            stats_nz=res.stats.nz,
        )
    elif inst.suite == "miplib":
        tmp = (root or cache_root()) / "expanded" / "miplib" / f"{inst.name}.{os.getpid()}.mps"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            with gzip.open(raw, "rb") as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            lp = read_with_highs(tmp, inst.name)
        finally:
            tmp.unlink(missing_ok=True)
    else:  # pragma: no cover
        raise ValueError(inst.suite)
    info.update(lp.info)
    return lp, info


def ensure_canonical(inst: Instance, root: Path | None = None) -> CanonicalLP:
    path = inst.canonical_path(root)
    if path.exists():
        return load_npz(path)
    lp, _ = read_raw(inst, root)
    save_npz(lp, path)
    return lp


def ensure_mps(inst: Instance, k: int, seed: int, root: Path | None = None) -> Path:
    """Write the permuted MPS file for (instance, k) if it does not exist yet."""
    path = inst.mps_path(k, root)
    if path.exists():
        return path
    lp = ensure_canonical(inst, root)
    perm = make_permutation(lp.m, lp.n, k, seed, inst.name)
    write_mps(permute(lp, perm) if k else lp, path)
    return path


def prepare(
    cfg: StudyConfig,
    instances: Iterable[Instance] | None = None,
    perms: dict[str, set[int]] | None = None,
    log: Log = _stderr,
) -> list[dict[str, object]]:
    """Canonicalise every instance and write its identity-order MPS (plus requested perms).

    Writes ``cache/prepared.json`` with dimensions and reader notes, one entry per instance.
    """
    insts = list(instances) if instances is not None else all_instances(cfg)
    records = []
    t0 = time.perf_counter()
    for idx, inst in enumerate(insts, 1):
        path = inst.canonical_path()
        if path.exists():
            lp = load_npz(path)
            info: dict[str, object] = {}
        else:
            lp, info = read_raw(inst)
            save_npz(lp, path)
        for k in sorted({0} | (perms or {}).get(inst.id, set())):
            ensure_mps(inst, k, cfg.seed)
        rec = {
            "id": inst.id,
            "m": lp.m,
            "n": lp.n,
            "nnz": lp.nnz,
            "maximize": lp.maximize_flipped,
            "c0": lp.c0,
        }
        rec.update(info)
        records.append(rec)
        if info:
            log(f"[prepare] {idx}/{len(insts)} {inst.id}: {lp.m} x {lp.n}, nnz {lp.nnz}")
    prepared = cache_root() / "prepared.json"
    old: dict[str, dict[str, object]] = {}
    if prepared.exists():
        old = {r["id"]: r for r in json.loads(prepared.read_text(encoding="utf-8"))}
    for r in records:
        merged = dict(old.get(str(r["id"]), {}))
        merged.update(r)
        old[str(r["id"])] = merged
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(json.dumps(sorted(old.values(), key=lambda r: str(r["id"])), indent=1))
    log(f"[prepare] {len(insts)} instances in {time.perf_counter() - t0:.1f} s")
    return records
