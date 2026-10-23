"""``lpbench report``: tables, figures and ranking sensitivity from a run directory.

Everything here is computed from unit records and committed manifests only, with seeded
random number generators, sorted keys and no timestamps, so the output regenerates
byte-identically from the same run directory.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lpbench import stats as S
from lpbench.runner import load_records

PROFILE_FLOOR_S = 0.02  # timer granularity guard for performance-profile ratios (DESIGN 12)
FAIL_GROUPS = ("claimed_unverified", "TIME_LIMIT", "KILLED", "ERROR", "NUMERICAL", "other")


# --------------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------------


@dataclass
class RunData:
    run_dir: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    configs: list[str]
    time_limit: float
    seed: int

    def select(self, **kw: Any) -> list[dict[str, Any]]:
        out = []
        for r in self.records:
            if all(r.get(k) == v for k, v in kw.items()):
                out.append(r)
        return out


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    h = manifest["config_hash"]
    recs = [r for r in load_records(run_dir).values() if r.get("config_hash") == h]
    recs.sort(key=lambda r: r["unit_id"])
    cfg = manifest["config"]
    return RunData(
        run_dir, manifest, recs, list(cfg["configs"]), float(cfg["time_limit_s"]), int(cfg["seed"])
    )


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *salt]))


def time_of(r: dict[str, Any], key: str = "solve_cpu_s") -> float:
    v = r.get(key)
    return float(v) if v is not None else math.inf


def solved_verified(eps_key: str) -> Any:
    def f(r: dict[str, Any]) -> bool:
        return r.get("status") == "OPTIMAL" and bool(r.get("verified_at", {}).get(eps_key))

    return f


def solved_trust(r: dict[str, Any]) -> bool:
    return r.get("status") == "OPTIMAL"


def eps_key(eps: float) -> str:
    return f"{eps:.0e}"


def matrix(
    recs: list[dict[str, Any]],
    instances: list[str],
    configs: list[str],
    solved_fn: Any,
    time_limit: float,
    par: int = 1,
    time_key: str = "solve_cpu_s",
    tl_prime: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T penalised, raw times, solved) for one record per (instance, config)."""
    idx = {(r["instance"], r["config"]): r for r in recs}
    n, s = len(instances), len(configs)
    raw = np.full((n, s), np.inf)
    ok = np.zeros((n, s), dtype=bool)
    for i, inst in enumerate(instances):
        for j, c in enumerate(configs):
            r = idx.get((inst, c))
            if r is None:
                continue
            raw[i, j] = time_of(r, time_key)
            ok[i, j] = bool(solved_fn(r)) and math.isfinite(raw[i, j])
    tl = time_limit if tl_prime is None else tl_prime
    if tl_prime is not None:
        ok &= raw <= tl_prime
    T = S.penalize(np.where(np.isfinite(raw), raw, tl), ok, tl, par)
    return T, raw, ok


def failure_group(r: dict[str, Any], eps: str) -> str | None:
    if r.get("status") == "OPTIMAL":
        return None if r.get("verified_at", {}).get(eps) else "claimed_unverified"
    st = str(r.get("status"))
    if st in ("TIME_LIMIT", "KILLED", "ERROR", "NUMERICAL"):
        return st
    return "other"


# --------------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------------


def f3(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1 else f"{x:.2f}"
    return f"{x:.1e}"


def pval(p: float) -> str:
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def _kv(d: dict[str, Any]) -> str:
    return ", ".join(f"{k} {v}" for k, v in d.items())


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _round(obj: Any, nd: int = 6) -> Any:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return float(f"{obj:.{nd}g}")
    if isinstance(obj, dict):
        return {str(k): _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist(), nd)
    if isinstance(obj, (np.floating, np.integer)):
        return _round(obj.item(), nd)
    return obj


# --------------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------------


def outcome_table(recs: list[dict[str, Any]], configs: list[str], eps: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in configs:
        rs = [r for r in recs if r["config"] == c]
        cnt: Counter[str] = Counter()
        for r in rs:
            g = failure_group(r, eps)
            cnt[g or "verified"] += 1
        out[c] = {"units": len(rs), "verified": cnt["verified"], **{g: cnt[g] for g in FAIL_GROUPS}}
    return out


def sgm_table(
    T: np.ndarray, configs: list[str], shift: float, B: int, seed: int, salt: int
) -> dict[str, dict[str, float]]:
    out = {}
    for j, c in enumerate(configs):
        iv = S.sgm_interval(T, j, shift, B, _rng(seed, salt, j))
        out[c] = {"sgm": iv.point, "lo": iv.lo, "hi": iv.hi}
    return out


def pairwise(
    T: np.ndarray, configs: list[str], shift: float, B: int, seed: int, salt: int
) -> list[dict[str, Any]]:
    pairs = list(itertools.combinations(range(len(configs)), 2))
    m = len(pairs)
    res = []
    pvals = []
    for k, (a, b) in enumerate(pairs):
        med, p = S.wilcoxon_pair(T[:, a], T[:, b], shift)
        iv = S.ratio_interval(T, a, b, shift, B, _rng(seed, salt, k))
        ivb = S.ratio_interval(T, a, b, shift, B, _rng(seed, salt, k), level=1 - 0.05 / max(m, 1))
        pvals.append(p)
        res.append(
            {
                "a": configs[a],
                "b": configs[b],
                "ratio": iv.point,
                "lo": iv.lo,
                "hi": iv.hi,
                "lo_bonf": ivb.lo,
                "hi_bonf": ivb.hi,
                "median_log_diff": med,
                "p": p,
            }
        )
    for r, ph in zip(res, S.holm(pvals), strict=True):
        r["p_holm"] = float(ph)
    return res


def ranking(T: np.ndarray, configs: list[str], shift: float) -> dict[str, float]:
    return dict(zip(configs, S.ranks_by_sgm(T, shift).tolist(), strict=True))


def significant_pairs(pw: list[dict[str, Any]], alpha: float = 0.05) -> list[tuple[str, str]]:
    """Ordered pairs (faster, slower) with a Holm-significant difference."""
    out = []
    for r in pw:
        if r["p_holm"] < alpha:
            out.append((r["a"], r["b"]) if r["ratio"] < 1 else (r["b"], r["a"]))
    return out


def compare_rankings(
    primary: dict[str, float],
    other: dict[str, float],
    sig: list[tuple[str, str]],
    sgm_other: dict[str, float],
) -> dict[str, Any]:
    common = [c for c in primary if c in other]
    tau = S.kendall_tau(
        np.array([primary[c] for c in common]), np.array([other[c] for c in common])
    )
    flips = [
        f"{a} < {b}"
        for a, b in sig
        if a in sgm_other and b in sgm_other and sgm_other[a] > sgm_other[b]
    ]
    order = sorted(common, key=lambda c: (other[c], c))
    return {"tau": tau, "order": order, "flipped_significant_pairs": flips}


# --------------------------------------------------------------------------------------------
# Main report builder
# --------------------------------------------------------------------------------------------


def build(run: RunData, B: int | None = None, shift: float | None = None) -> dict[str, Any]:
    cfg = run.manifest["config"]
    stats_cfg = cfg.get("stats", {})
    B = int(B or stats_cfg.get("bootstrap_resamples", 10_000))
    shift = float(shift if shift is not None else stats_cfg.get("shift_s", 1.0))
    shifts = [float(s) for s in stats_cfg.get("shift_sensitivity_s", [0.1, 10.0])]
    first_order = list(cfg.get("first_order", []))
    configs = run.configs
    TL = run.time_limit
    seed = run.seed
    main_eps = next((float(b["eps"]) for b in cfg["blocks"] if b["name"] == "main"), 1e-6)
    ek = eps_key(main_eps)
    main = [r for r in run.records if r["block"] == "main" and r["k"] == 0]
    instances = sorted({r["instance"] for r in main})
    suites = sorted({r["suite"] for r in main})
    inst_suite = {r["instance"]: r["suite"] for r in main}
    inst_nnz = {r["instance"]: r.get("nnz_published", 0) for r in main}
    res: dict[str, Any] = {"meta": {}, "tables": {}}

    # -- what was run -------------------------------------------------------------------------
    blocks = Counter(r["block"] for r in run.records)
    status_all = Counter(r["status"] for r in run.records)
    total_cpu = sum(time_of(r) for r in run.records if math.isfinite(time_of(r)))
    total_wall_units = sum(float(r.get("unit_wall_s") or 0) for r in run.records)
    prov = run.manifest.get("provenance", {})
    res["meta"] = {
        "study": run.manifest.get("study"),
        "config_hash": run.manifest.get("config_hash"),
        "units": len(run.records),
        "units_by_block": dict(sorted(blocks.items())),
        "status_counts": dict(sorted(status_all.items())),
        "instances": {s: sum(1 for i in instances if inst_suite[i] == s) for s in suites},
        "configs": configs,
        "time_limit_s": TL,
        "hard_deadline_s": cfg.get("hard_deadline_factor", 1.5) * TL
        + cfg.get("hard_deadline_extra_s", 10.0),
        "workers": cfg.get("workers"),
        "seed": seed,
        "bootstrap_resamples": B,
        "shift_s": shift,
        "solve_cpu_total_s": total_cpu,
        "unit_wall_total_s": total_wall_units,
        "versions": prov.get("distributions"),
        "solver_libraries": prov.get("solver_libraries"),
        "platform": prov.get("platform"),
        "git": sorted({str(r.get("git")) for r in run.records}),
        "config_errors": status_all.get("CONFIG_ERROR", 0),
        "hidden_thread_flags": sum(1 for r in run.records if r.get("hidden_threads")),
    }

    # -- outcomes ------------------------------------------------------------------------------
    res["tables"]["outcomes"] = {"all": outcome_table(main, configs, ek)}
    for s in suites:
        res["tables"]["outcomes"][s] = outcome_table(
            [r for r in main if r["suite"] == s], configs, ek
        )

    # -- verification detail ------------------------------------------------------------------
    ver: dict[str, Any] = {}
    for c in configs:
        rs = [r for r in main if r["config"] == c]
        opt = [r for r in rs if r["status"] == "OPTIMAL"]
        d: dict[str, Any] = {"claimed_optimal": len(opt)}
        for e in ("1e-04", "1e-06", "1e-08"):
            d[f"fail_{e}"] = sum(1 for r in opt if not r.get("verified_at", {}).get(e))
            d[f"floor_limited_{e}"] = sum(1 for r in opt if r.get("floor_limited_at", {}).get(e))
        d["kkt_pass_ref_fail"] = sum(
            1
            for r in opt
            if r.get("max_rel") is not None
            and r["max_rel"] <= main_eps
            and r.get("reference") is not None
            and not r.get("verified_at", {}).get(ek)
        )
        d["no_dual"] = sum(
            1 for r in opt if r.get("verification") and not r["verification"].get("has_dual")
        )
        worst = sorted(
            (r for r in opt if not r.get("verified_at", {}).get(ek)),
            key=lambda r: -(r.get("max_rel") or math.inf),
        )
        d["examples"] = [
            {
                "instance": r["instance"],
                "max_rel": r.get("max_rel"),
                "rel_primal": (r.get("verification") or {}).get("rel_primal"),
                "rel_dual": (r.get("verification") or {}).get("rel_dual"),
                "rel_gap": (r.get("verification") or {}).get("rel_gap"),
                "ref_rel_error": r.get("ref_rel_error"),
            }
            for r in worst[:5]
        ]
        ver[c] = d
    res["tables"]["verification"] = ver
    # status disagreements (infeasible/unbounded claims on instances others solve)
    disagreements = []
    for inst in instances:
        rs = [r for r in main if r["instance"] == inst]
        claims = sorted(
            {r["config"] for r in rs if r["status"] in ("INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD")}
        )
        solved = sorted({r["config"] for r in rs if r.get("verified_solve")})
        if claims and solved:
            disagreements.append(
                {"instance": inst, "infeasible_or_unbounded_claims": claims, "verified_by": solved}
            )
    res["tables"]["status_disagreements"] = disagreements
    # instances dropped (all finishing configs agree infeasible/unbounded)
    dropped = []
    for inst in instances:
        rs = [r for r in main if r["instance"] == inst]
        fin = [
            r for r in rs if r["status"] in ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD")
        ]
        if fin and all(r["status"] != "OPTIMAL" for r in fin):
            dropped.append(inst)
    perf_insts = [i for i in instances if i not in dropped]
    res["meta"]["dropped_infeasible_or_unbounded"] = dropped

    # -- primary analysis ---------------------------------------------------------------------
    T, raw, ok = matrix(main, perf_insts, configs, solved_verified(ek), TL)
    primary_rank = ranking(T, configs, shift)
    primary_sgm = dict(zip(configs, np.atleast_1d(S.sgm(T, shift, axis=0)).tolist(), strict=True))
    res["tables"]["sgm"] = {"all": sgm_table(T, configs, shift, B, seed, 1)}
    pw_all = pairwise(T, configs, shift, B, seed, 2)
    res["tables"]["pairwise"] = {"all": pw_all}
    for si, s in enumerate(suites):
        sub = [i for i in perf_insts if inst_suite[i] == s]
        Ts, _, _ = matrix(main, sub, configs, solved_verified(ek), TL)
        res["tables"]["sgm"][s] = sgm_table(Ts, configs, shift, B, seed, 10 + si)
        res["tables"]["pairwise"][s] = pairwise(Ts, configs, shift, B, seed, 20 + si)
    rs_ = S.rank_stability(T, shift, B, _rng(seed, 3))
    res["tables"]["rank_probabilities"] = {c: rs_.p_rank[j].tolist() for j, c in enumerate(configs)}
    res["tables"]["p_faster"] = {
        a: {b: float(rs_.p_faster[i, j]) for j, b in enumerate(configs) if i != j}
        for i, a in enumerate(configs)
    }
    res["tables"]["primary_ranking"] = sorted(configs, key=lambda c: (primary_rank[c], c))
    res["tables"]["solved_counts"] = {c: int(ok[:, j].sum()) for j, c in enumerate(configs)}

    # -- performance profiles -----------------------------------------------------------------
    taus = S.default_taus(81)
    Tp = np.maximum(np.where(np.isfinite(raw), raw, TL), PROFILE_FLOOR_S)
    bands = S.profile_bands(Tp, ok, taus, B, _rng(seed, 4))
    res["profiles"] = {
        "taus": taus.tolist(),
        "rho": {c: bands.rho[j].tolist() for j, c in enumerate(configs)},
        "lo": {c: bands.lo[j].tolist() for j, c in enumerate(configs)},
        "hi": {c: bands.hi[j].tolist() for j, c in enumerate(configs)},
        "sim_halfwidth": dict(zip(configs, bands.sim_halfwidth.tolist(), strict=True)),
    }
    pick = {1.0: 0, 2.0: None, 10.0: None, 100.0: None}
    for t in pick:
        pick[t] = int(np.argmin(np.abs(np.log(taus) - math.log(t))))
    res["tables"]["profile_points"] = {
        c: {f"{t:g}": float(bands.rho[j, k]) for t, k in pick.items()}
        | {"sim_halfwidth": float(bands.sim_halfwidth[j])}
        for j, c in enumerate(configs)
    }

    # -- ranking sensitivity (a)-(j) ----------------------------------------------------------
    sig = significant_pairs(pw_all)
    sens: dict[str, Any] = {}

    def add(label: str, Tm: np.ndarray, cfgs: list[str] | None = None, sh: float = shift) -> None:
        cs = cfgs or configs
        rk = ranking(Tm, cs, sh)
        sg = dict(zip(cs, np.atleast_1d(S.sgm(Tm, sh, axis=0)).tolist(), strict=True))
        sens[label] = compare_rankings(primary_rank, rk, sig, sg) | {"n_instances": len(Tm)}

    Ta, _, _ = matrix(main, perf_insts, configs, solved_trust, TL)
    add("(a) trust solver status", Ta)
    for e in ("1e-04", "1e-08"):
        if e != ek:
            Tb, _, _ = matrix(main, perf_insts, configs, solved_verified(e), TL)
            add(f"(b) verify at {e}", Tb)
    eps4 = [r for r in run.records if r["block"] == "eps1e-4"]
    if eps4:
        net = [i for i in perf_insts if inst_suite[i] == "netlib"]
        fo_recs = {(r["instance"], r["config"]): r for r in eps4}
        mixed = [
            fo_recs.get((r["instance"], r["config"]), r) if r["config"] in first_order else r
            for r in main
            if r["instance"] in set(net)
        ]
        Tc, _, _ = matrix(mixed, net, configs, solved_verified("1e-04"), TL)
        add("(c) requested eps 1e-4 (netlib; first-order re-run, others re-verified)", Tc)
    for sh in shifts:
        add(f"(d) shift {sh:g} s", T, sh=sh)
    Te, _, _ = matrix(main, perf_insts, configs, solved_verified(ek), TL, par=2)
    add("(e) PAR2", Te)
    for tlp in (1.0, 3.0):
        if tlp < TL:
            Tf, _, _ = matrix(main, perf_insts, configs, solved_verified(ek), TL, tl_prime=tlp)
            add(f"(f) time limit {tlp:g} s (post hoc)", Tf)
    for s in suites:
        sub = [i for i in perf_insts if inst_suite[i] == s]
        Tg, _, _ = matrix(main, sub, configs, solved_verified(ek), TL)
        add(f"(g) suite {s} only", Tg)
    perm_recs = [r for r in run.records if r["block"] == "perm"]
    perm_insts = sorted({r["instance"] for r in perm_recs})
    if perm_insts:
        ks = sorted({r["k"] for r in perm_recs})
        for k in [0, *ks]:
            src = main if k == 0 else [r for r in perm_recs if r["k"] == k]
            Th, _, _ = matrix(src, perm_insts, configs, solved_verified(ek), TL)
            add(f"(h) permutation k={k} (subset)", Th)
    Ti, _, _ = matrix(main, perf_insts, configs, solved_verified(ek), TL, time_key="solve_wall_s")
    add("(i) wall time instead of CPU time", Ti)
    pool = sorted(perf_insts, key=lambda i: (inst_suite[i], inst_nnz[i], i))
    for t in range(3):
        sub = []
        for s in suites:
            si = [i for i in pool if inst_suite[i] == s]
            parts = np.array_split(np.arange(len(si)), 3)
            sub.extend(si[p] for p in parts[t])
        Tj, _, _ = matrix(main, sub, configs, solved_verified(ek), TL)
        add(f"(j) size tertile {t + 1} of 3 (per suite)", Tj)
    res["tables"]["sensitivity"] = sens
    res["tables"]["primary_sgm"] = primary_sgm

    # -- tolerance block ----------------------------------------------------------------------
    if eps4:
        net = sorted({r["instance"] for r in eps4})
        tol = {}
        for c in first_order:
            r4 = [r for r in eps4 if r["config"] == c]
            r6 = [r for r in main if r["config"] == c and r["instance"] in set(net)]
            T4, _, ok4 = matrix(r4, net, [c], solved_verified("1e-04"), TL)
            T6, _, ok6 = matrix(r6, net, [c], solved_verified(ek), TL)
            tol[c] = {
                "verified_at_1e-4_run": int(ok4.sum()),
                "verified_at_1e-6_run": int(ok6.sum()),
                "claimed_unverified_1e-4_run": sum(1 for r in r4 if r.get("claimed_unverified")),
                "sgm_1e-4_run": S.sgm(T4[:, 0], shift),
                "sgm_1e-6_run": S.sgm(T6[:, 0], shift),
                "n": len(net),
            }
        res["tables"]["tolerance"] = tol

    # -- variability --------------------------------------------------------------------------
    if perm_insts:
        allk = {(r["instance"], r["config"], r["k"]): r for r in perm_recs}
        for r in main:
            if r["instance"] in set(perm_insts):
                allk[(r["instance"], r["config"], 0)] = r
        ks_all = sorted({k for (_, _, k) in allk})
        var = {}
        T3 = np.zeros((len(perm_insts), len(configs), len(ks_all)))
        for j, c in enumerate(configs):
            gsd_t, gsd_it, rr, flips, n_pairs = [], [], [], 0, 0
            for i, inst in enumerate(perm_insts):
                rs = [allk.get((inst, c, k)) for k in ks_all]
                if any(r is None for r in rs):
                    continue
                okk = [solved_verified(ek)(r) for r in rs]  # type: ignore[arg-type]
                tt = [time_of(r) for r in rs]  # type: ignore[arg-type]
                for kk, (o, t) in enumerate(zip(okk, tt, strict=True)):
                    T3[i, j, kk] = min(t, TL) if o else TL
                if len(set(okk)) > 1:
                    flips += 1
                n_pairs += 1
                if all(okk):
                    gsd_t.append(S.geometric_sd(np.array(tt) + shift))
                    rr.append(S.range_ratio(np.array(tt) + shift))
                    its = [r.get("total_iterations") or 0 for r in rs]  # type: ignore[union-attr]
                    if min(its) > 0:
                        gsd_it.append(S.geometric_sd(np.array(its, dtype=float), 1.0))
            var[c] = {
                "instances": n_pairs,
                "status_flips": flips,
                "median_gsd_time": float(np.median(gsd_t)) if gsd_t else None,
                "p90_gsd_time": float(np.quantile(gsd_t, 0.9)) if gsd_t else None,
                "median_gsd_iterations": float(np.median(gsd_it)) if gsd_it else None,
                "p90_gsd_iterations": float(np.quantile(gsd_it, 0.9)) if gsd_it else None,
                "max_range_ratio_time": float(np.max(rr)) if rr else None,
            }
        res["tables"]["variability"] = var
        # one-level (k = 0 only) vs two-level intervals on the subset, versus highs-dsimplex
        base = configs.index("highs-dsimplex") if "highs-dsimplex" in configs else 0
        two = {}
        for j, c in enumerate(configs):
            if j == base:
                continue
            iv1 = S.ratio_interval(T3[:, :, 0], j, base, shift, B, _rng(seed, 5, j))
            iv2 = S.two_level_ratio_interval(T3, j, base, shift, B, _rng(seed, 6, j))
            two[c] = {"vs": configs[base], "one_level": iv1.as_tuple(), "two_level": iv2.as_tuple()}
        res["tables"]["two_level"] = two
        res["meta"]["perm_subset"] = {"instances": len(perm_insts), "ks": ks_all}

    # -- machine noise ------------------------------------------------------------------------
    noise = [r for r in run.records if r["block"] == "noise"]
    if noise:
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in noise:
            groups[(r["instance"], r["config"])].append(r)
        for r in main:
            if (r["instance"], r["config"]) in groups:
                groups[(r["instance"], r["config"])].append(r)
        nz = {}
        for c in sorted({c for (_, c) in groups}):
            gsd_c, gsd_w, same_it, n = [], [], 0, 0
            for (_inst, cc), rs in sorted(groups.items()):
                if cc != c or len(rs) < 2:
                    continue
                n += 1
                gsd_c.append(S.geometric_sd(np.array([time_of(r) for r in rs]) + shift))
                gsd_w.append(
                    S.geometric_sd(np.array([time_of(r, "solve_wall_s") for r in rs]) + shift)
                )
                its = {r.get("total_iterations") for r in rs}
                same_it += len(its) == 1
            nz[c] = {
                "instances": n,
                "median_gsd_cpu": float(np.median(gsd_c)) if gsd_c else None,
                "max_gsd_cpu": float(np.max(gsd_c)) if gsd_c else None,
                "median_gsd_wall": float(np.median(gsd_w)) if gsd_w else None,
                "identical_iterations": same_it,
            }
        res["tables"]["noise"] = nz

    # -- contention: wall vs cpu --------------------------------------------------------------
    ratios = [
        time_of(r, "solve_wall_s") / time_of(r)
        for r in run.records
        if r.get("solve_cpu_s") and r["solve_cpu_s"] > 0.5 and r.get("solve_wall_s")
    ]
    res["meta"]["wall_over_cpu"] = {
        "units": len(ratios),
        "median": float(np.median(ratios)) if ratios else None,
        "p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "max": float(np.max(ratios)) if ratios else None,
    }
    res["meta"]["reference_sources"] = dict(
        sorted(
            Counter(
                str(r.get("reference_source")) for r in main if r["config"] == configs[0]
            ).items()
        )
    )
    return _round(res)


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------


def to_markdown(res: dict[str, Any], option_table: dict[str, Any] | None = None) -> str:
    m = res["meta"]
    t = res["tables"]
    configs = m["configs"]
    out = io.StringIO()
    w = out.write
    w(f"# lpbench results: {m['study']}\n\n")
    w("Generated by `lpbench report` from the unit records; do not edit by hand.\n\n")
    w("## What was run\n\n")
    inst = ", ".join(f"{k} {v}" for k, v in m["instances"].items())
    w(
        f"- Instances (main block): {inst}; configs: {len(configs)}; units: {m['units']} "
        f"({', '.join(f'{k} {v}' for k, v in m['units_by_block'].items())}).\n"
    )
    w(
        f"- Time limit {f3(m['time_limit_s'])} s per unit (hard deadline "
        f"{f3(m['hard_deadline_s'])} s), {m['workers']} workers, one solver thread each, "
        f"seed {m['seed']}, bootstrap B = {m['bootstrap_resamples']}, shift {f3(m['shift_s'])} s, "
        "failures scored at the time limit (PAR1), times are process CPU seconds.\n"
    )
    w(
        f"- Unit statuses: {', '.join(f'{k} {v}' for k, v in m['status_counts'].items())}. "
        f"CONFIG_ERROR units: {m['config_errors']}. Hidden-thread flags: "
        f"{m['hidden_thread_flags']}.\n"
    )
    w(
        f"- Solver CPU time, all units: {m['solve_cpu_total_s'] / 60:.1f} min; summed unit wall "
        f"time: {m['unit_wall_total_s'] / 60:.1f} min.\n"
    )
    woc = m["wall_over_cpu"]
    if woc["units"]:
        w(
            f"- Contention check (units with > 0.5 s CPU): wall/CPU median {f3(woc['median'])}, "
            f"90th percentile {f3(woc['p90'])}, max {f3(woc['max'])} over {woc['units']} units.\n"
        )
    if m.get("versions"):
        w(
            f"- Versions: {_kv(m['versions'])}; solver "
            f"libraries: {_kv(m.get('solver_libraries') or {})}.\n"
        )
    if m.get("platform"):
        p = m["platform"]
        w(
            f"- Machine: {p.get('system')} {p.get('release')}, {p.get('machine')}, "
            f"{p.get('logical_cpus')} logical CPUs, Python {p.get('python')}; shared with other "
            "jobs during the run.\n"
        )
    w(
        f"- Reference sources (main block): "
        f"{', '.join(f'{k} {v}' for k, v in m['reference_sources'].items())}.\n"
    )
    if m.get("dropped_infeasible_or_unbounded"):
        w(
            f"- Dropped (all finishing configs report infeasible/unbounded): "
            f"{', '.join(m['dropped_infeasible_or_unbounded'])}.\n"
        )
    w("\n")

    w("## Outcomes (main block, original order)\n\n")
    hdr = [
        "config",
        "units",
        "verified",
        "claimed but unverified",
        "time limit",
        "killed",
        "error",
        "numerical",
        "other",
    ]
    for scope, tab in t["outcomes"].items():
        w(f"**{scope}**\n\n")
        rows = [
            [
                c,
                d["units"],
                d["verified"],
                d["claimed_unverified"],
                d["TIME_LIMIT"],
                d["KILLED"],
                d["ERROR"],
                d["NUMERICAL"],
                d["other"],
            ]
            for c, d in tab.items()
        ]
        w(md_table(hdr, rows) + "\n\n")

    w("## Verification of claimed optima (main block)\n\n")
    rows = []
    for c, d in t["verification"].items():
        rows.append(
            [
                c,
                d["claimed_optimal"],
                d["fail_1e-04"],
                d["fail_1e-06"],
                d["fail_1e-08"],
                d["kkt_pass_ref_fail"],
                d["no_dual"],
                d["floor_limited_1e-06"],
            ]
        )
    w(
        md_table(
            [
                "config",
                "claimed optimal",
                "fail at 1e-4",
                "fail at 1e-6",
                "fail at 1e-8",
                "KKT ok, objective off",
                "no dual",
                "floor-limited at 1e-6",
            ],
            rows,
        )
        + "\n\n"
    )
    w(
        "Failing at an epsilon means the solution misses the primal, dual or gap criterion "
        "of DESIGN 5.2 at that epsilon, or disagrees with the reference objective. These are "
        "tolerance-definition outcomes, not solver bugs (no bug-claim protocol was completed).\n\n"
    )
    ex_rows = []
    for c, d in t["verification"].items():
        for e in d["examples"][:3]:
            ex_rows.append(
                [
                    c,
                    e["instance"],
                    f3(e["rel_primal"]),
                    f3(e["rel_dual"]),
                    f3(e["rel_gap"]),
                    f3(e["ref_rel_error"]),
                ]
            )
    if ex_rows:
        w("Largest violations among claimed-but-unverified solves (up to 3 per config):\n\n")
        w(
            md_table(
                [
                    "config",
                    "instance",
                    "rel. primal",
                    "rel. dual",
                    "rel. gap",
                    "objective rel. error",
                ],
                ex_rows,
            )
            + "\n\n"
        )
    if t["status_disagreements"]:
        w("Status disagreements (infeasible/unbounded claims on instances verified by others):\n\n")
        w(
            md_table(
                ["instance", "claims", "verified by"],
                [
                    [
                        d["instance"],
                        ", ".join(d["infeasible_or_unbounded_claims"]),
                        ", ".join(d["verified_by"]),
                    ]
                    for d in t["status_disagreements"]
                ],
            )
            + "\n\n"
        )

    w("## Shifted geometric means (CPU s, 95% percentile bootstrap over instances)\n\n")
    scopes = list(t["sgm"])
    rows = []
    for c in configs:
        row = [c, t["solved_counts"][c]]
        for s in scopes:
            d = t["sgm"][s][c]
            row.append(f"{f3(d['sgm'])} [{f3(d['lo'])}, {f3(d['hi'])}]")
        rows.append(row)
    w(md_table(["config", "verified (all)"] + [f"SGM {s}" for s in scopes], rows) + "\n\n")
    w(f"Primary ranking (SGM, all instances): {' < '.join(t['primary_ranking'])}.\n\n")

    w("## Pairwise comparisons (all instances)\n\n")
    w(
        "Ratio = SGM_a / SGM_b with 95% percentile interval and Bonferroni-adjusted interval "
        "(21 pairs); p from the Wilcoxon signed-rank test on log(t + shift), Holm-adjusted "
        "over the 21 pairs of this family.\n\n"
    )
    rows = [
        [
            r["a"],
            r["b"],
            f3(r["ratio"]),
            f"[{f3(r['lo'])}, {f3(r['hi'])}]",
            f"[{f3(r['lo_bonf'])}, {f3(r['hi_bonf'])}]",
            pval(r["p"]),
            pval(r["p_holm"]),
        ]
        for r in t["pairwise"]["all"]
    ]
    w(md_table(["a", "b", "ratio", "95% CI", "Bonferroni CI", "p", "p (Holm)"], rows) + "\n\n")
    for s in scopes:
        if s == "all":
            continue
        sig = [r for r in t["pairwise"][s] if r["p_holm"] < 0.05]
        w(f"Suite {s}: {len(sig)} of {len(t['pairwise'][s])} pairs Holm-significant at 0.05.\n\n")

    w("## Rank probabilities (bootstrap over instances)\n\n")
    rows = [[c] + [f3(p) for p in t["rank_probabilities"][c]] for c in configs]
    w(md_table(["config"] + [f"P(rank {j + 1})" for j in range(len(configs))], rows) + "\n\n")

    w("## Performance profiles\n\n")
    w(
        "rho(tau) at selected tau with the simultaneous 95% band half-width (bootstrap of "
        f"sup |rho* - rho|). Times are floored at {PROFILE_FLOOR_S} s for the ratios. Profiles "
        "compare each config with the best one per instance; they do not rank two non-best "
        "configs, and adding or removing a config can change the others' curves (Gould and "
        "Scott 2016).\n\n"
    )
    rows = [[c] + [f3(v) for k, v in d.items()] for c, d in t["profile_points"].items()]
    keys = list(next(iter(t["profile_points"].values())).keys())
    w(
        md_table(
            ["config"] + [f"rho({k})" if k != "sim_halfwidth" else "band +-" for k in keys], rows
        )
        + "\n\n"
    )

    if "tolerance" in t:
        w("## Requested tolerance 1e-4 vs 1e-6 (first-order configs, netlib)\n\n")
        rows = [
            [
                c,
                d["n"],
                d["verified_at_1e-4_run"],
                d["claimed_unverified_1e-4_run"],
                d["verified_at_1e-6_run"],
                f3(d["sgm_1e-4_run"]),
                f3(d["sgm_1e-6_run"]),
            ]
            for c, d in t["tolerance"].items()
        ]
        w(
            md_table(
                [
                    "config",
                    "instances",
                    "verified at 1e-4 (1e-4 run)",
                    "claimed unverified (1e-4 run)",
                    "verified at 1e-6 (1e-6 run)",
                    "SGM 1e-4 run",
                    "SGM 1e-6 run",
                ],
                rows,
            )
            + "\n\n"
        )

    if "variability" in t:
        ps = m["perm_subset"]
        w(
            f"## Performance variability ({ps['instances']} instances, permutations "
            f"k = {', '.join(str(k) for k in ps['ks'])})\n\n"
        )
        rows = [
            [
                c,
                d["instances"],
                d["status_flips"],
                f3(d["median_gsd_time"]),
                f3(d["p90_gsd_time"]),
                f3(d["median_gsd_iterations"]),
                f3(d["p90_gsd_iterations"]),
                f3(d["max_range_ratio_time"]),
            ]
            for c, d in t["variability"].items()
        ]
        w(
            md_table(
                [
                    "config",
                    "instances",
                    "status flips",
                    "median GSD time",
                    "p90 GSD time",
                    "median GSD iterations",
                    "p90 GSD iterations",
                    "max range ratio",
                ],
                rows,
            )
            + "\n\n"
        )
        w(
            "GSD = geometric standard deviation over permutations of (t + shift) for times and "
            "of iteration counts (1 = identical), over instances verified in every permutation. "
            "Iteration counts do not depend on the machine, so their variability is "
            "algorithmic; time variability also contains machine noise.\n\n"
        )
        rows = [
            [
                c,
                d["vs"],
                f"{f3(d['one_level'][0])} [{f3(d['one_level'][1])}, {f3(d['one_level'][2])}]",
                f"{f3(d['two_level'][0])} [{f3(d['two_level'][1])}, {f3(d['two_level'][2])}]",
            ]
            for c, d in t["two_level"].items()
        ]
        w(
            "SGM ratio on the subset: one-level bootstrap on k = 0 only versus the two-level "
            "bootstrap over instances and permutations:\n\n"
        )
        w(md_table(["config", "vs", "k = 0 only", "two-level (all k)"], rows) + "\n\n")

    if "noise" in t:
        w("## Machine noise (identical units repeated)\n\n")
        rows = [
            [
                c,
                d["instances"],
                f3(d["median_gsd_cpu"]),
                f3(d["max_gsd_cpu"]),
                f3(d["median_gsd_wall"]),
                d["identical_iterations"],
            ]
            for c, d in t["noise"].items()
        ]
        w(
            md_table(
                [
                    "config",
                    "instances",
                    "median GSD CPU",
                    "max GSD CPU",
                    "median GSD wall",
                    "identical iteration counts",
                ],
                rows,
            )
            + "\n\n"
        )

    w("## What changes the ranking\n\n")
    w(
        f"Primary: {' < '.join(t['primary_ranking'])}. Kendall tau of each alternative ranking "
        "to the primary one, and Holm-significant primary orderings that flip.\n\n"
    )
    rows = [
        [
            k,
            d["n_instances"],
            f3(d["tau"]),
            " < ".join(d["order"]),
            "; ".join(d["flipped_significant_pairs"]) or "none",
        ]
        for k, d in t["sensitivity"].items()
    ]
    w(md_table(["change", "instances", "tau", "ranking", "flipped significant pairs"], rows))
    w("\n\n")
    if option_table:
        w("## Solver options at epsilon = 1e-6 (verbatim)\n\n")
        for c, o in option_table.items():
            w(f"- `{c}`: `{json.dumps(o, sort_keys=True)}`\n")
        w("\n")
    return out.getvalue()


# --------------------------------------------------------------------------------------------
# Figures and files
# --------------------------------------------------------------------------------------------


def write_figures(res: dict[str, Any], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # optional dependency
        return []
    plt.rcParams["svg.hashsalt"] = "lpbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = res["profiles"]
    taus = np.array(pr["taus"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in res["meta"]["configs"]:
        rho = np.array(pr["rho"][c])
        hw = pr["sim_halfwidth"][c]
        (line,) = ax.step(taus, rho, where="post", label=c, lw=1.4)
        ax.fill_between(
            taus,
            np.clip(rho - hw, 0, 1),
            np.clip(rho + hw, 0, 1),
            step="post",
            alpha=0.12,
            color=line.get_color(),
            lw=0,
        )
    ax.set_xscale("log")
    ax.set_xlim(1, taus[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("tau (time ratio to the best config on the instance)")
    ax.set_ylabel("share of instances")
    ax.set_title("Performance profiles, verified solves (simultaneous 95% bands)")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    name = "performance_profiles.svg"
    fig.savefig(out_dir / name, metadata={"Date": None})
    plt.close(fig)
    return [name]


def write_units_csv(run: RunData, path: Path) -> None:
    fields = [
        "unit_id",
        "block",
        "instance",
        "k",
        "config",
        "eps",
        "repeat",
        "status",
        "verified_solve",
        "claimed_unverified",
        "solve_cpu_s",
        "solve_wall_s",
        "total_iterations",
        "objective",
        "reference",
        "max_rel",
        "rel_primal",
        "rel_dual",
        "rel_gap",
        "ref_rel_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        for r in run.records:
            v = r.get("verification") or {}
            row = {k: r.get(k) for k in fields}
            for k in ("rel_primal", "rel_dual", "rel_gap"):
                row[k] = v.get(k)
            for k, val in list(row.items()):
                if isinstance(val, float):
                    row[k] = f"{val:.6g}"
            wr.writerow(row)


def publish(run_dir: Path, out_dir: Path, B: int | None = None) -> dict[str, Any]:
    from lpbench.solvers.registry import option_table

    run = load_run(run_dir)
    res = build(run, B=B)
    cfg = run.manifest["config"]
    opts = option_table(1e-6, float(cfg["time_limit_s"]), int(cfg["seed"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "summary.md").write_text(to_markdown(res, opts), encoding="utf-8", newline="\n")
    write_units_csv(run, out_dir / "units.csv")
    write_figures(res, out_dir)
    return res
