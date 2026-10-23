"""Statistics (DESIGN 7): shifted geometric means, bootstrap intervals, performance profiles
with confidence bands, pairwise tests with Holm correction, rank stability, variability.

Conventions: ``T`` is an (n instances x s configs) array of times in seconds in which every
non-verified outcome already carries the penalty value; ``solved`` is the boolean mask of
verified solves (used by performance profiles, where failures have ratio infinity). With
replicates, ``T3`` has shape (n, s, K).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

T_MIN = 0.001


# --------------------------------------------------------------------------------------------
# Shifted geometric mean
# --------------------------------------------------------------------------------------------


def shifted_log(t: np.ndarray, shift: float) -> np.ndarray:
    return np.log(np.maximum(np.asarray(t, dtype=float), T_MIN) + shift)


def sgm(t: np.ndarray, shift: float = 1.0, axis: int = 0) -> np.ndarray | float:
    """``exp(mean(log(max(t, t_min) + shift))) - shift`` along ``axis``."""
    out = np.exp(np.mean(shifted_log(t, shift), axis=axis)) - shift
    return float(out) if np.ndim(out) == 0 else out


def collapse_replicates(T3: np.ndarray, shift: float) -> np.ndarray:
    """Per (instance, config): geometric mean over replicates of (t + shift), minus shift."""
    return np.exp(np.mean(shifted_log(T3, shift), axis=-1)) - shift


def penalize(t: np.ndarray, solved: np.ndarray, time_limit: float, par: int = 1) -> np.ndarray:
    """PAR-k scoring: verified solves keep their time, everything else gets k * TL."""
    return np.where(solved, np.minimum(t, time_limit), par * time_limit)


# --------------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    level: float = 0.95

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.lo, self.hi)


def _percentile(samples: np.ndarray, point: float, level: float) -> Interval:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [a, 1.0 - a])
    return Interval(point, float(lo), float(hi), level)


def bootstrap_indices(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def sgm_bootstrap(
    T: np.ndarray, shift: float, B: int, rng: np.random.Generator, idx: np.ndarray | None = None
) -> np.ndarray:
    """(B, s) bootstrap distribution of per-config SGMs, resampling instances (rows)."""
    L = shifted_log(T, shift)
    if idx is None:
        idx = bootstrap_indices(T.shape[0], B, rng)
    return np.exp(L[idx].mean(axis=1)) - shift


def sgm_interval(
    T: np.ndarray, j: int, shift: float, B: int, rng: np.random.Generator, level: float = 0.95
) -> Interval:
    boot = sgm_bootstrap(T[:, [j]], shift, B, rng)[:, 0]
    return _percentile(boot, float(sgm(T[:, j], shift)), level)


def ratio_interval(
    T: np.ndarray,
    a: int,
    b: int,
    shift: float,
    B: int,
    rng: np.random.Generator,
    level: float = 0.95,
    method: str = "percentile",
) -> Interval:
    """Paired SGM ratio ``SGM_a / SGM_b`` with a percentile (or BCa) bootstrap interval."""
    L = shifted_log(T[:, [a, b]], shift)
    n = L.shape[0]

    def ratio(Ls: np.ndarray) -> np.ndarray:
        g = np.exp(Ls.mean(axis=-2)) - shift
        return g[..., 0] / g[..., 1]

    point = float(ratio(L))
    idx = bootstrap_indices(n, B, rng)
    boot = ratio(L[idx])
    if method == "percentile":
        return _percentile(boot, point, level)
    if method != "bca":
        raise ValueError(method)
    # BCa (Efron 1987): bias correction z0 and acceleration from the jackknife
    z0 = sps.norm.ppf(np.clip(np.mean(boot < point), 1e-12, 1 - 1e-12))
    jack = np.array([ratio(np.delete(L, i, axis=0)) for i in range(n)])
    d = jack.mean() - jack
    acc = float(np.sum(d**3) / (6.0 * np.sum(d**2) ** 1.5)) if np.any(d) else 0.0
    alpha = (1.0 - level) / 2.0
    qs = []
    for z in (sps.norm.ppf(alpha), sps.norm.ppf(1 - alpha)):
        qs.append(sps.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))
    lo, hi = np.quantile(boot, qs)
    return Interval(point, float(lo), float(hi), level)


def two_level_ratio_interval(
    T3: np.ndarray,
    a: int,
    b: int,
    shift: float,
    B: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> Interval:
    """Two-level bootstrap with replicates (Davison and Hinkley 1997, 3.8).

    Resample instances; within each sampled instance resample replicate indices k, shared by
    both configs to keep the pairing; collapse replicates, then take the SGM ratio.
    """
    n, _, K = T3.shape
    L = shifted_log(T3[:, [a, b], :], shift)  # (n, 2, K)

    def ratio_from(Lc: np.ndarray) -> np.ndarray:  # Lc: (..., n, 2) collapsed logs
        g = np.exp(Lc.mean(axis=-2)) - shift
        return g[..., 0] / g[..., 1]

    point = float(ratio_from(L.mean(axis=-1)))
    inst = rng.integers(0, n, size=(B, n))
    reps = rng.integers(0, K, size=(B, n, K))
    sel = L[inst]  # (B, n, 2, K)
    gathered = np.take_along_axis(sel, reps[:, :, None, :], axis=3)
    boot = ratio_from(gathered.mean(axis=-1))
    return _percentile(boot, point, level)


# --------------------------------------------------------------------------------------------
# Performance profiles (Dolan and More 2002) with bootstrap bands
# --------------------------------------------------------------------------------------------


def perf_ratios(T: np.ndarray, solved: np.ndarray) -> np.ndarray:
    """``r_{p,s} = t_{p,s} / min_s t_{p,s}`` over solved entries; inf for failures."""
    t = np.where(solved, np.maximum(T, T_MIN), np.inf)
    best = t.min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.isfinite(t), t / best, np.inf)
    return r


def profile(T: np.ndarray, solved: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``rho_s(tau)``: share of instances with ratio <= tau. Shape (s, len(taus))."""
    r = perf_ratios(T, solved)
    return (r[:, :, None] <= taus[None, None, :]).mean(axis=0)


def default_taus(n: int = 81) -> np.ndarray:
    return np.logspace(0, 4, n)


@dataclass(frozen=True)
class ProfileBands:
    taus: np.ndarray
    rho: np.ndarray  # (s, n_tau)
    lo: np.ndarray  # pointwise 95%
    hi: np.ndarray
    sim_halfwidth: np.ndarray  # (s,) simultaneous band: rho +- this


def profile_bands(
    T: np.ndarray,
    solved: np.ndarray,
    taus: np.ndarray,
    B: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> ProfileBands:
    """Pointwise percentile bands and a simultaneous band from ``sup_tau |rho* - rho|``."""
    rho = profile(T, solved, taus)
    r = perf_ratios(T, solved)
    n = T.shape[0]
    hits = (r[:, :, None] <= taus[None, None, :]).astype(np.float32)  # (n, s, tau)
    boots = np.empty((B, *rho.shape), dtype=np.float32)
    chunk = 500
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = rng.integers(0, n, size=(b, n))
        boots[start : start + b] = hits[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo = np.quantile(boots, a, axis=0)
    hi = np.quantile(boots, 1 - a, axis=0)
    sup = np.abs(boots - rho[None]).max(axis=2)  # (B, s)
    half = np.quantile(sup, level, axis=0)
    return ProfileBands(taus, rho, lo, hi, half)


# --------------------------------------------------------------------------------------------
# Tests and multiplicity
# --------------------------------------------------------------------------------------------


def holm(pvals: list[float] | np.ndarray) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_pair(ta: np.ndarray, tb: np.ndarray, shift: float) -> tuple[float, float]:
    """Wilcoxon signed-rank test on ``log(t_a + s) - log(t_b + s)``: (median diff, p)."""
    d = shifted_log(ta, shift) - shifted_log(tb, shift)
    nz = d[np.abs(d) > 1e-12]
    if nz.size == 0:
        return 0.0, 1.0
    res = sps.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(np.median(d)), float(res.pvalue)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# --------------------------------------------------------------------------------------------
# Ranks
# --------------------------------------------------------------------------------------------


def ranks_by_sgm(T: np.ndarray, shift: float) -> np.ndarray:
    """Rank (1 = fastest) of each config by SGM; ties share the mean rank."""
    return sps.rankdata(np.asarray(sgm(T, shift, axis=0)), method="average")


@dataclass(frozen=True)
class RankStability:
    p_rank: np.ndarray  # (s, s): P(rank of config i == j+1)
    p_faster: np.ndarray  # (s, s): P(SGM_i < SGM_j)


def rank_stability(T: np.ndarray, shift: float, B: int, rng: np.random.Generator) -> RankStability:
    boot = sgm_bootstrap(T, shift, B, rng)  # (B, s)
    s = T.shape[1]
    ranks = np.argsort(np.argsort(boot, axis=1, kind="stable"), axis=1)  # 0 = fastest
    p_rank = np.zeros((s, s))
    for j in range(s):
        p_rank[:, j] = (ranks == j).mean(axis=0)
    p_faster = (boot[:, :, None] < boot[:, None, :]).mean(axis=0)
    return RankStability(p_rank, p_faster)


def kendall_tau(r1: np.ndarray, r2: np.ndarray) -> float:
    if len(r1) < 2:
        return math.nan
    tau = sps.kendalltau(r1, r2).statistic
    return float(tau)


# --------------------------------------------------------------------------------------------
# Variability (Koch et al. 2011; Lodi and Tramontani 2013)
# --------------------------------------------------------------------------------------------


def geometric_sd(x: np.ndarray, floor: float = T_MIN) -> float:
    """``exp(sd(log x))`` (1 = no variation); x floored at ``floor``."""
    lx = np.log(np.maximum(np.asarray(x, dtype=float), floor))
    if lx.size < 2:
        return math.nan
    return float(np.exp(np.std(lx, ddof=1)))


def range_ratio(x: np.ndarray, floor: float = T_MIN) -> float:
    x = np.maximum(np.asarray(x, dtype=float), floor)
    return float(x.max() / x.min()) if x.size else math.nan
