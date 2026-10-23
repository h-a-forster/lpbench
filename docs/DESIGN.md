# lpbench design

Status: v0.0.1, design only. Nothing below is implemented yet except the scaffold (M0). Every
deviation made while implementing must be recorded in section 12 in the same change.

## 1. Question and contribution

Public LP solver comparisons usually report a solved count and a shifted geometric mean of
run times, taking each solver's "optimal" status on trust, from one run per instance. lpbench
asks what such a comparison looks like when it is done carefully on CPU with the open-source
solvers anyone can `pip install`:

1. **Verified solves.** Does every reported optimum satisfy primal feasibility, dual
   feasibility and a small duality gap at one common tolerance, checked by independent code
   against the original problem data, and does its objective match a reference value?
2. **Variability.** How much do run times, iteration counts and outcomes change when the same
   problem is presented with its rows and columns permuted, or with a different solver seed?
3. **Uncertainty.** What are the bootstrap intervals on shifted geometric means, ratios of
   them, and performance profiles, with instances as the sampling unit?
4. **Sensitivity.** Which analyst choices change the ranking: trusting solver status versus
   verification, the tolerance, the shift, the time limit, the failure penalty, the instance
   suite, or the permutation?

The methodology is the contribution. The headline deliverable is a reproducible report
(`REPORT.md` plus `docs/results/quick/`) with, for each solver configuration: verified-solve
counts, the number of claimed optima that fail verification, shifted geometric means with
intervals, performance profiles with confidence bands, variability measures, and a table of
which choices change the ranking. Everything is measured on one CPU machine, single-threaded.
lpbench makes no claim about GPU solvers, multi-threaded performance, or commercial solvers.

### 1.1 Scope decisions (what was cut and why)

| Item | Decision | Reason |
|---|---|---|
| GPU solvers (cuPDLP, cuOpt) | Out | No GPU on the study machine. The report must not extrapolate to GPUs. |
| Mittelmann "LPfeas" instances | Out of v0.1; optional suite later (S2) | Most instances need far more than the 45-minute budget on CPU; the redistribution terms are not stated. |
| MIPLIB 2017 "collection" (1,065 instances) | Out | The benchmark set (240) is curated and documented; a size cap keeps it tractable. |
| Kennington Netlib problems | Out | Distributed separately (ftp), larger; not needed for the headline. |
| HiGHS HiPO interior point | Out | The PyPI wheel of highspy 1.15.1 is built without BLAS/METIS; requesting `solver="hipo"` fails ("features are unavailable: amd, blas, metis, rcm"). Recorded as a finding about the wheel, not the solver. |
| Clarabel | Optional (S1), direct `clarabel` API only | It is a conic IPM; through cvxpy the timing would include cvxpy canonicalisation. Its tolerance semantics differ again. |
| Multi-threaded runs, memory profiling | Out | Single thread per solve is the only fair setting with 3 concurrent workers on a shared machine. |
| Presolve on/off, crossover on/off ablations | Out | Default settings, with the tolerance alignment of section 4.3, are the configurations. |
| Exact rational verification | Out | Floating-point residuals at stated tolerances (section 5); the evaluation floor is checked. |

## 2. Solvers and configurations

All installed from PyPI with wheels for Linux and Windows (checked for cp311 and cp313).
Versions are pinned exactly in `pyproject.toml`; `uv.lock` pins the rest.

| Config id | Library (PyPI, version, licence) | Algorithm | Seed control |
|---|---|---|---|
| `highs-dsimplex` | highspy 1.15.1 (MIT) | HiGHS dual revised simplex (`solver="simplex"`) | `random_seed` |
| `highs-ipx` | highspy 1.15.1 | IPX interior point with crossover (`solver="ipx"`, `run_crossover="on"`) | `random_seed` |
| `highs-pdlp` | highspy 1.15.1 | cuPDLP-C port running on CPU (`solver="pdlp"`) | none (deterministic) |
| `highs-hipdlp` | highspy 1.15.1 | HiPDLP, CPU (`solver="hipdlp"`; Halpern restarts, PID step) | none |
| `ortools-glop` | ortools 9.15.6755 (Apache-2.0) | GLOP primal/dual simplex (model_builder `Solver("glop")`) | `random_seed` |
| `ortools-pdlp` | ortools 9.15.6755 | PDLP (model_builder `Solver("pdlp")`) | none |
| `scip-soplex` | PySCIPOpt 6.3.0 (MIT); SCIP 10.0, SoPlex 8.0.2 bundled | SoPlex simplex via SCIP root LP, presolve/heuristics/separation/propagation off | `randomization/randomseedshift`, `randomization/lpseed` |

Notes recorded from probing the wheels on 2026-10-24 (re-check in M3):

- In highspy 1.15.1, `solver="ipm"` currently dispatches to IPX; we name `ipx` explicitly.
- `hipdlp` logs "performance may be disappointing" on CPU and, on a 2x2 LP, declared
  "Optimal" with a relative primal infeasibility of 1.05x its tolerance. This is exactly what
  the verifier exists to catch; it is a status-policy observation, not a bug claim.
- HiGHS `pdlp_optimality_tolerance` is applied to the scaled problem (HiGHS/GAMS docs), so
  unscaled residuals can exceed it. OR-Tools PDLP's default termination uses relative L2
  norms; we set `optimality_norm: OPTIMALITY_NORM_L_INF` to match our verifier's norm.
- SCIP returns duals (`getDualsolLinear`) and reduced costs (`getVarRedcost`) only with
  presolving off. This makes `scip-soplex` slower than default SCIP; the report must say so.
- Invalid options are detectable in every library (HiGHS returns `kError` from
  `setOptionValue`, OR-Tools returns `INVALID_SOLVER_PARAMETERS`, SCIP raises). Adapters must
  treat these as configuration errors, never as solver failures.
- Dual sign convention: on `min -x-y s.t. x+2y<=4, 3x+y<=6`, all three libraries return row
  duals (-0.4, -0.2) and zero reduced costs, consistent with `z = c - A^T y` (section 5.1).

## 3. Instances

### 3.1 Netlib LP (suite `netlib`)

- Source: `https://netlib.org/lp/data/<name>` (the old `www.netlib.org` URL redirects).
  Files are in Gay's compressed netlib format, which must be expanded to MPS (the reference
  expander is `emps.c`). The files carry per-line checksums.
- lpbench implements its own expander (`netlib_emps.py`) from the format as described by Gay
  (1985) and the behaviour of `emps`. `emps.c` carries no licence statement, so it is not
  copied or vendored. Validation: zero checksum errors on every file, and every instance's
  optimum reproduced (section 6.2).
- Included: the problems in `lp/data` that ship as compressed MPS, about 92 after removing
  QAP8/12/15, STOCFOR3 and TRUSS (generators or bundles, not MPS) and STANDGUB (GUB markers;
  identical to STANDATA once removed). DFL001 has no reliable published value (readme note).
- Reference objectives: the exact optimal values of Koch (2004), computed with rational
  arithmetic (open-access preprint ZIB-Report 03-05), transcribed to
  `src/lpbench/data/netlib_reference.csv` with source and page. Cross-check: each must agree
  with the netlib `readme` table to 1e-6 relative except problems the readme flags. If the
  transcription cannot be completed, the readme values are the fallback and the report says
  so. Values are facts; no problem data is committed.
- Licence: the netlib pages state no licence; problems were contributed for free
  distribution. We commit only names, sizes, checksums and reference values.
- Size: about 12 MB in total (MAROS-R7, 4.8 MB, is the largest).

### 3.2 MIPLIB 2017 benchmark-set LP relaxations (suite `miplib`)

- Source: `https://miplib.zib.de/WebData/instances/<name>.mps.gz`, instance list
  `https://miplib.zib.de/downloads/benchmark-v2.test` (240 instances). Sizes (rows, columns,
  nonzeros) from the benchmark table at `https://miplib.zib.de/tag_benchmark.html`.
- The LP relaxation drops integrality and keeps all bounds (binaries stay in [0, 1]).
- Size cap by nonzeros of A (from the table, before download): `quick` uses nnz <= 1e5
  (114 instances), `full` uses nnz <= 1e6 (210 instances). Counts measured from the table on
  2026-10-24.
- `scripts/build_miplib_manifest.py` produces `src/lpbench/data/miplib_benchmark.csv`
  (name, rows, cols, nnz, sha256 and bytes filled on first verified download). The HTML table
  is parsed once by that script, never at run time.
- Reference objectives: MIPLIB publishes MIP optima, not LP relaxation values. Section 6.3
  defines a consensus reference.
- Licence: the MIPLIB site states no licence for the 2017 instances (only "(c) ZIB"); the
  MIPLIB 2024 call publishes new instances under CC BY-SA 4.0. Because redistribution of 2017
  instances is unclear, lpbench commits no instance data, only names, sizes, checksums and
  derived results, and tells users to download from ZIB.

### 3.3 Download policy (`download.py`)

- Only the two hosts above. HTTPS, connect/read timeouts (10 s / 60 s), 3 attempts with
  backoff, streaming to `cache/raw/<suite>/<file>.part` then atomic rename.
- A pre-download HEAD request gives `Content-Length`; the running total of bytes for a run
  must stay under `download_cap_bytes` (default 2 GB). The quick study needs well under
  100 MB; `full` is projected at a few hundred MB (the whole 240-instance benchmark zip is
  317 MB).
- SHA-256 of every file is recorded on first download in the manifest; later downloads must
  match. A mismatch is a hard error (`lpbench fetch --update-checksums` to accept a changed
  upstream file deliberately).
- `cache/` is gitignored. `lpbench fetch --suite netlib|miplib --config quick` is the only
  network entry point; `run` never downloads.

## 4. Architecture

```
src/lpbench/
  cli.py            lpbench versions | fetch | prepare | reference | plan | run | status | report
  versions.py       library versions, SCIP/SoPlex versions, platform, CPU, git SHA        [M0]
  config.py         StudyConfig from configs/*.toml; validation; config hash
  lpmodel.py        CanonicalLP (CSR/CSC arrays, bounds), from HiGHS, permute/unpermute, MPS out
  netlib_emps.py    netlib compressed format -> fixed MPS, with checksum verification
  instances.py      suites, manifests (data/*.csv), size caps, stratification
  download.py       fetch with timeouts, caps, checksums
  verify.py         residuals, dual objective, gap, pass/fail per tolerance, evaluation floor
  solvers/
    base.py         SolveRequest, SolveResult, Adapter protocol, status enum
    highs.py        highs-dsimplex | highs-ipx | highs-pdlp | highs-hipdlp
    ortools.py      ortools-glop | ortools-pdlp
    scip.py         scip-soplex
    registry.py     config id -> adapter + per-tolerance option table
  worker.py         persistent worker process: env pinning, solve, measure, verify, return
  runner.py         units, scheduling, hard timeouts, checkpoints, provenance, progress log
  reference.py      consensus reference objectives for suites without published values
  stats.py          SGM, paired bootstrap, two-level bootstrap, profiles + bands, ranks, Holm
  report.py         tables (Markdown + JSON), figures (matplotlib optional), ranking sensitivity
  data/             netlib_reference.csv, miplib_benchmark.csv (committed, small)
scripts/
  run_study.py      retry wrapper: resume-on-failure loop around `lpbench run`, scrubbed log
  build_miplib_manifest.py
configs/            quick.toml, full.toml
```

Data flow: `fetch` downloads raw files. `prepare` reads each raw file with HiGHS once,
extracts a `CanonicalLP`, drops integrality, and writes, for each permutation index k, a
permuted MPS file `cache/mps/<suite>/<name>__p<k>.mps` plus `cache/canonical/<name>.npz`.
All solver configurations read the same MPS file for a given (instance, k). `run` executes
units (instance, k, config, tolerance) in persistent worker processes; each worker
un-permutes the returned solution and verifies it against the canonical LP before writing
the unit record. `report` reads only unit records and manifests.

### 4.1 Core types

```python
@dataclass(frozen=True)
class CanonicalLP:  # min c^T x + c0  s.t.  rl <= A x <= ru,  l <= x <= u
    name: str
    A: scipy.sparse.csr_matrix  # m x n, float64
    c: np.ndarray
    c0: float
    rl: np.ndarray
    ru: np.ndarray  # +-inf allowed, rl <= ru
    l: np.ndarray
    u: np.ndarray
    maximize_flipped: bool  # True if the source was a max problem (c negated)


@dataclass(frozen=True)
class SolveResult:
    status: SolverStatus  # OPTIMAL, INFEASIBLE, UNBOUNDED, TIME_LIMIT, ITER_LIMIT,
    # NUMERICAL, ERROR, KILLED, CONFIG_ERROR
    raw_status: str  # library's own status string
    objective: float | None  # as reported by the solver
    x: np.ndarray | None  # primal, in the permuted problem's variable order
    y: np.ndarray | None  # row duals, convention of section 5.1
    iterations: dict[str, int]  # simplex / ipm / pdlp / crossover counts as reported
    solve_wall_s: float  # perf_counter around the solve call only
    solve_cpu_s: float  # process CPU time around the solve call only
    solver_time_s: float | None  # library-reported time, if any
```

The unit record (JSON, one file per unit) holds the request, the `SolveResult` without the
vectors, the verification summary (section 5), the reference agreement, timings, and
provenance (versions, git SHA, config hash, platform, worker pid).

### 4.2 Execution model (`worker.py`, `runner.py`)

- At most `workers = 3` persistent processes (spawn context; the same code path on Windows
  and Linux). Before importing any solver the worker sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1. Every adapter sets its library's thread
  count to 1 (HiGHS `threads=1`, OR-Tools PDLP `num_threads: 1`, SCIP `lp/threads=1`).
  `solve_cpu_s > 1.2 * solve_wall_s` on any unit flags hidden threading.
- Model reading happens before the timed region; timing covers the solve call only.
- Each unit has a solver time limit `TL` (passed to the library) and a hard wall deadline
  `TL * 1.5 + 10 s` enforced by the parent. A worker past its deadline is terminated and
  respawned; the unit is recorded as `KILLED`. A worker that dies (segfault, OOM) yields
  `ERROR` with the exit code. Neither stops the run.
- Scheduling: units are ordered by descending expected cost (instance nnz times a per-config
  factor from the pilot) so that long units start first.
- Memory: the nnz caps keep each process well under 1 GB; the worker records peak RSS when the
  platform exposes it cheaply (`resource` on POSIX; skipped on Windows).
- Checkpointing: each finished unit is written to `runs/<study>/units/<unit_id>.json` via a
  temporary file and `os.replace`. A rerun skips unit ids whose record exists and whose config
  hash matches. `runs/<study>/progress.log` gets one line per unit (done/total, rate, ETA).
  `runs/<study>/manifest.json` stores the config, its hash, versions and a history of starts.
- `scripts/run_study.py` wraps `lpbench run` in a loop of up to 3 attempts that resumes after
  any failure, and writes `runs/<study>/wrapper.log` with local paths scrubbed.
- `lpbench plan --config quick` prints the unit count, the worst-case wall bound
  (sum of hard deadlines / workers) and, once a pilot exists, the projected wall time. `run`
  refuses a config whose projection exceeds 45 minutes unless `--allow-long` is given.

### 4.3 Tolerance alignment

A run requests accuracy epsilon (1e-6 by default). The rule: first-order configs have their
optimality tolerance set to epsilon (it defines the run). Simplex, IPM and SCIP configs keep
their defaults, except that a feasibility or optimality tolerance looser than epsilon is
tightened to epsilon. All configs get threads=1 and the time limit. The exact options per config and epsilon are listed in
`solvers/registry.py` and copied verbatim into the report.

| Config | Options at epsilon = 1e-6 (beyond threads, time limit) |
|---|---|
| highs-dsimplex | defaults (`primal/dual_feasibility_tolerance` 1e-7 are already tighter) |
| highs-ipx | defaults (`ipm_optimality_tolerance` 1e-8, crossover on) |
| highs-pdlp, highs-hipdlp | `pdlp_optimality_tolerance = 1e-6` (library default 1e-7) |
| ortools-glop | defaults (`primal/dual_feasibility_tolerance` 1e-8) |
| ortools-pdlp | `eps_optimal_absolute = eps_optimal_relative = 1e-6`, `optimality_norm: L_INF` |
| scip-soplex | defaults (`numerics/feastol` 1e-6, `dualfeastol` 1e-7); at 1e-8 both set to 1e-8 |

Solvers define tolerances differently (absolute or relative, scaled or unscaled, L2 or
L-infinity). A solver can therefore stop at its own epsilon and still fail ours. That is a
measured fact about "optimal" statuses, not evidence of a bug; the report labels it so.

## 5. Verification (`verify.py`)

### 5.1 Conventions

Canonical problem: minimise `c^T x + c0` subject to `rl <= A x <= ru`, `l <= x <= u`, with
infinite bounds allowed. A maximisation source is converted by negating c and c0 (and the
reported objective is converted back for display only).

Duals: `y` in R^m with the sign convention `z = c - A^T y`. At an optimum of a minimisation
problem, `y_i > 0` only if `rl_i` is finite (row at its lower bound), `y_i < 0` only if `ru_i`
is finite; likewise `z_j > 0` only if `l_j` is finite and `z_j < 0` only if `u_j` is finite.
Adapters map library output to this convention; adapter tests fix it on hand-built LPs with
every row and bound type, for minimisation and maximisation sources.

lpbench computes `z` itself from `y`; solver-reported reduced costs are only compared with it
(a diagnostic, not part of the pass rule).

### 5.2 Quantities

With `||.||` the infinity norm over finite entries and `[t]_+ = max(t, 0)`:

- Primal residual: `r_P = max( ||[rl - Ax]_+||, ||[Ax - ru]_+||, ||[l - x]_+||, ||[x - u]_+|| )`.
- Dual residual (infeasible parts of y and z): `r_D = max( ||y_i^+ : rl_i = -inf||,
  ||y_i^- : ru_i = +inf||, ||z_j^+ : l_j = -inf||, ||z_j^- : u_j = +inf|| )`, where
  `t^+ = [t]_+` and `t^- = [-t]_+`.
- Primal objective: `P = c^T x + c0`.
- Dual objective: `D = c0 + sum_i (y_i^+ rl_i - y_i^- ru_i) + sum_j (z_j^+ l_j - z_j^- u_j)`,
  summing only terms whose bound is finite (the infeasible parts are already in r_D).
- Scale terms: `b_scale = 1 + max(||rl||, ||ru||, ||l||, ||u||)` over finite entries,
  `c_scale = 1 + ||c||`.

A solution passes at tolerance epsilon if all three hold:

```
r_P <= eps * b_scale
r_D <= eps * c_scale
|P - D| <= eps * (1 + |P| + |D|)
```

This mirrors the relative termination criteria of PDLP (Applegate et al., 2021) in the
L-infinity norm. It is reported at epsilon in {1e-4, 1e-6, 1e-8} for every unit regardless
of the requested accuracy; the primary class equals the requested epsilon.

Reference agreement: `|P - P_ref| <= eps_obj * (1 + |P_ref|)` with `eps_obj = eps`.

**Verified solve** = solver status OPTIMAL and passes at epsilon and agrees with the
reference (when a reference exists). Status OPTIMAL but failing either check is a
**claimed-but-unverified** solve and counts as a failure in every performance metric.

### 5.3 Evaluation floor

Residuals are computed in float64 with scipy sparse products. The evaluation error of
`A x` is bounded by about `gamma_k * (|A| |x|)_i` with `gamma_k ~ k * 2^-53` for row length k.
`verify.py` reports `floor_P = max_i k_i * 2^-53 * (|A| |x|)_i / b_scale`; a unit whose floor
exceeds `eps / 10` is flagged `floor_limited` and listed separately, never silently passed or
failed. Same for the dual side with `|A^T| |y|`.

### 5.4 Status handling

INFEASIBLE / UNBOUNDED claims are not certified in v0.1 (no Farkas ray check). An instance is
dropped from performance metrics only if every configuration that finishes agrees it is
infeasible or unbounded; any disagreement is reported as a finding to investigate.

## 6. Validation strategy (how we know the harness is right)

### 6.1 Verifier

- Hand-built LPs with known primal-dual optima covering: <=, >=, =, ranged and free rows;
  lower, upper, boxed, fixed and free variables; c0 != 0; a maximisation source.
- Perturbation tests: shifting x by delta along a violated direction raises r_P by delta;
  flipping one dual sign raises r_D; scaling y moves the gap; pass/fail switches exactly at
  the thresholds.
- Property test: for random feasible bounded LPs solved by two different libraries, both
  pass at 1e-6 and agree on P to 1e-6 relative.

### 6.2 Instance pipeline

- `netlib_emps.py`: every file expands with zero checksum errors; the expanded MPS of AFIRO
  matches a recorded SHA-256 and hand-checked dimensions (no netlib data is committed); MPS round trip through HiGHS preserves
  dimensions and nonzeros given in the readme table (readme counts include the cost row).
- Every Netlib instance solved by `highs-dsimplex` at default tolerances must match the
  reference to 1e-6 relative; failures are investigated before any benchmark number is
  reported (the cause is either our expander, the MPS reading, or a flagged problem).
- Parser cross-check: for each prepared instance, the model OR-Tools reads from our MPS
  (exported to `MPModelProto`) matches the canonical LP's dimensions, bounds and matrix to
  1e-12 (slow test over all instances, fast test over 5).
- Permutation correctness: solving a permuted instance and un-permuting gives the same
  verification result and objective (to 1e-9 relative) for a simplex config on 10 instances.

### 6.3 Consensus reference for MIPLIB relaxations

`lpbench reference --config quick` runs, for the miplib suite, the three independent simplex implementations
(`highs-dsimplex`, `ortools-glop`, `scip-soplex`) on the identity ordering at epsilon = 1e-9
with a time limit of 60 s, as its own checkpointed run (same runner, budget under 15 minutes
wall for the quick suite; results committed as `src/lpbench/data/miplib_reference.csv`).
An instance gets a reference value if at least two of them return
OPTIMAL solutions that pass verification at 1e-8 and agree with each other to 1e-8 relative
(feasibility tolerances are tightened to 1e-9 for this pass);
the reference is their mean, and the record lists which configs agreed. Instances without a
consensus keep the KKT check only, and the report counts them. The KKT certificate is the
primary oracle: a primal-feasible x and dual-feasible (y, z) with a small gap bound the
optimum from both sides regardless of which solver produced them. The consensus value is a
secondary check, and its partial circularity (it uses benchmarked libraries) is stated.

### 6.4 Adapters

1. On every hand-built LP, each config returns a solution that passes at 1e-6 with the right
   objective, duals in the section 5.1 convention.
2. On small infeasible and unbounded LPs, statuses map correctly.
3. A time limit of 0.5 s on an instance that needs longer returns TIME_LIMIT within the hard
   deadline (one slow test per library).
4. Invalid options produce CONFIG_ERROR.
5. **Bug-claim protocol.** Before the report attributes a failure to a solver: a minimal
   standalone script that calls the library directly (no lpbench code) reproduces it on the
   original downloaded file; the same instance passes with another library through the same
   adapter path; and the residuals are recomputed by an independent short script. Only then
   is it called a solver issue, with versions, and it goes in `docs/repros/`.

### 6.5 Statistics

- `stats.py` functions are tested against hand-computed values (SGM with shift, profile
  step functions, Holm adjustment on a textbook example).
- Bootstrap coverage check (slow test): on synthetic paired log-normal run times with a known
  SGM ratio, the 95% percentile interval covers the truth in a share of 1,000 replications
  whose 99.9% Clopper-Pearson interval contains 0.95.

## 7. Statistics (`stats.py`)

Notation: instances p = 1..n, configs s, permutation replicates k = 0..K (k = 0 is the
original order). `t_{p,s,k}` is the solve time (CPU seconds; wall time as a sensitivity), set
to the penalty value for any non-verified outcome.

- **Shifted geometric mean** (Achterberg 2007; Mittelmann's benchmarks):
  `SGM_s = exp( (1/n) sum_p log(max(t_{p,s}, t_min) + sigma) ) - sigma`, with shift
  `sigma = 1 s` primary (our times are short), `sigma in {0.1, 10}` as sensitivity,
  `t_min = 0.001 s`. Penalty for failures: `TL` (PAR1) primary, `2 TL` (PAR2) sensitivity.
  With replicates, `t_{p,s}` is the geometric mean over k of `(t + sigma)`, minus sigma.
- **Solved count**: number of verified solves, with failures broken down into
  claimed-but-unverified, time limit, killed, error, other status.
- **Paired SGM ratio** `R_{A,B} = SGM_A / SGM_B` over the same instances; percentile bootstrap
  with B = 10,000 resamples of instances (BCa as a sensitivity). With replicates, a two-level
  bootstrap: resample instances, then resample replicate indices k within each instance,
  shared across configs to keep the pairing (Davison and Hinkley 1997, section 3.8).
- **Performance profiles** (Dolan and Moré 2002): `r_{p,s} = t_{p,s} / min_s t_{p,s}`
  (r = infinity for failures), `rho_s(tau) = |{p : r_{p,s} <= tau}| / n`, on a log grid of
  tau from 1 to 1e4. Confidence bands: bootstrap instances; pointwise 95% percentile bands
  and a simultaneous 95% band from the bootstrap distribution of
  `sup_tau |rho*_s(tau) - rho_s(tau)|`. Caveat stated in the report (Gould and Scott 2016):
  profiles compare each solver with the best; they do not rank two non-best solvers, and
  adding or removing a solver can change the curves of the others.
- **Pairwise tests**: Wilcoxon signed-rank test on `log(t_A + sigma) - log(t_B + sigma)` per
  instance; Holm (1979) correction over the 21 pairs within each family (suite x epsilon).
  Intervals are reported unadjusted and Bonferroni-adjusted.
- **Rank stability**: in each bootstrap resample rank configs by SGM; report
  `P(rank = j)` per config and `P(SGM_A < SGM_B)` per pair.
- **Variability** (Koch et al. 2011; Lodi and Tramontani 2013): per (instance, config) over
  k: geometric standard deviation of time and of iteration count, range ratio max/min, and
  status flips (verified in some k, not in others). Iteration counts are machine-independent,
  so their variability is purely algorithmic; time variability also contains machine noise.
  Machine noise is estimated separately by repeating k = 0 three times on a small subset.
- **What changes the ranking**: rankings by SGM under each single change from the primary
  analysis: (a) trust solver status instead of verification; (b) verify at 1e-4 or 1e-8
  instead of the requested epsilon; (c) requested epsilon 1e-4 (separate runs);
  (d) sigma in {0.1, 10}; (e) PAR2; (f) time limit TL' < TL (exact post hoc: a unit solved by
  t <= TL' is the same run); (g) suite netlib vs miplib; (h) each permutation k alone;
  (i) wall time instead of CPU time; (j) size strata. Report Kendall's tau to the primary
  ranking and whether any pairwise order with Holm-significant difference flips.

## 8. Study design

| | quick (this machine, < 45 min wall) | full (documented, not run by default) |
|---|---|---|
| Suites | netlib (~92) + miplib nnz <= 1e5 (114) | netlib + miplib nnz <= 1e6 (210) |
| Configs | 7 | 7 (+ clarabel if S1 lands) |
| Requested epsilon | 1e-6 all; 1e-4 for first-order configs on netlib | 1e-4, 1e-6, 1e-8 (first-order configs at all three; others at 1e-6 and 1e-8) |
| Permutations | k = 0 for all; k = 0..3 on a stratified subset of 30 instances | k = 0..4 for all |
| Machine-noise repeats | k = 0 x 3 on 10 instances, 2 configs | k = 0 x 3 on 30 instances, all configs |
| Time limit TL | 10 s (hard deadline 25 s) | 300 s (hard deadline 460 s) |
| Workers | 3 | 3 (more on a dedicated machine; record it) |
| Bootstrap B | 10,000 | 10,000 |

Under the alignment rule of section 4.3, the simplex, IPX and SCIP configurations use the
same options at epsilon = 1e-4 and 1e-6, so their 1e-4 runs would repeat the 1e-6 runs; only
the three first-order configs are run again at 1e-4 (the simplex/IPM records are re-verified
at 1e-4 instead, and the report says so).

Unit counts (quick): 206 x 7 at 1e-6 = 1,442; netlib at 1e-4, first-order only: 92 x 3 = 276;
permutations 30 x 7 x 3 = 630; noise repeats 10 x 2 x 2 = 40. Total 2,388 units, 1,164 of
them first-order. Most simplex and IPM units take well under a second; the budget is
dominated by first-order units that hit TL. Worst case (every first-order unit runs to TL =
10 s): 11,640 CPU-s, about 65 min on 3 workers, so the worst case does not fit and the plan
relies on the pilot. M5 runs a 10% stratified sample, and `lpbench plan` projects the wall
time from it. If the projection exceeds 40 minutes, cut in this order and record each cut in
the report: permutation subset 30 -> 20 instances; netlib 1e-4 runs dropped (re-verification
only); TL 10 -> 7 s. A short TL is unfavourable to first-order methods; analysis (f) shows
how results move with TL up to the limit used, and `full` uses TL = 300 s.

Commands:

```sh
uv run lpbench fetch --config quick
uv run lpbench prepare --config quick
uv run lpbench reference --config quick          # MIPLIB consensus values (cached)
uv run lpbench plan --config quick
uv run python scripts/run_study.py --config quick --out runs/quick
uv run lpbench report runs/quick --publish docs/results/quick
# full size (several hours; projection printed by `plan` after the quick run):
uv run python scripts/run_study.py --config full --out runs/full
```

The stratified permutation subset: 15 netlib + 15 miplib instances, chosen by a seeded draw
(seed in the config) within size tertiles, excluding instances on which no config verified
at k = 0 in the pilot. Recorded in the config, not chosen after seeing results.

## 9. Reporting rules

- State what was run: instance counts per suite after every exclusion, configs and options
  verbatim, epsilon, TL, K, seeds, library versions, machine (CPU model, cores, RAM, OS),
  worker count, total wall and CPU time, git SHA, config hash, and that the machine was
  shared with other jobs.
- Intervals with every point estimate; Holm-adjusted p-values for pairwise claims; the
  number of comparisons.
- Claimed-but-unverified results are described as a tolerance-definition outcome unless the
  bug-claim protocol (6.4 item 5) has been completed.
- Every number in REPORT.md is generated from run records by `lpbench report`; hand-written
  text cites the generated tables.
- Results apply to these versions, these instances, this machine, single thread. No GPU,
  multi-thread or commercial-solver claims.

## 10. Milestones and done-checks

Each milestone ends with `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`
passing, AGENTS.md "Status and next steps" updated, and a commit. Time estimates assume one
agent; total about 4.5 h plus the study run.

| M | Content | Done-check | Est. |
|---|---|---|---|
| M0 | Scaffold, pinned deps, CI, docs, `lpbench versions`, solver environment tests | uv sync/pytest/ruff pass; tiny LP solved by all 7 configs through the raw libraries | done |
| M1 | `lpmodel.py`, `verify.py` | 6.1 tests pass; permute/unpermute round trip exact; floor computed | 40 min |
| M2 | `netlib_emps.py`, `instances.py`, `download.py`, manifests, `fetch`, `prepare`, Koch reference CSV | all netlib files expand with zero checksum errors; HiGHS simplex matches the reference on every included netlib instance (exceptions documented); download total printed and under cap; checksums recorded | 50 min |
| M3 | `solvers/*` adapters and registry | 6.4 items 1-4 pass for all 7 configs; 10 netlib instances verified by each simplex/IPM config | 50 min |
| M4 | `config.py`, `worker.py`, `runner.py`, `run`, `status`, `plan`, `scripts/run_study.py` | kill-and-resume test (no duplicates, no lost units); hard-deadline test with a sleeping fake adapter; crash test with a fake adapter that calls `os._exit` | 45 min |
| M5 | `reference.py`; pilot on a 10% stratified sample | consensus coverage reported; `plan` projection < 45 min for quick (apply section 8 cuts if not) | 30 min + run |
| M6 | `stats.py` | 6.5 tests (coverage check marked slow) | 40 min |
| M7 | quick study, `report.py`, figures, `docs/results/quick/` | all units recorded; zero CONFIG_ERROR; report regenerates byte-identically from the run dir | 30 min + <= 45 min run |
| M8 | REPORT.md, README results, CHANGELOG 0.1.0, any repro in `docs/repros/` | every claim traceable to a generated table; bug-claim protocol completed for any solver issue | 20 min |
| S1 | Clarabel adapter (direct API) | 6.4 items 1-4 | optional |
| S2 | Mittelmann LP suite, size-capped, licence checked | manifest + checksums; within cap | optional |

## 11. References

- Applegate, D., Díaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., Schudy, W. (2021).
  Practical large-scale linear programming using primal-dual hybrid gradient. NeurIPS 34.
  arXiv:2106.04756. (PDLP; relative termination criteria; MIPLIB 2017 LP relaxations as a
  test set.)
- Applegate, D., et al. (2025). PDLP: A practical first-order method for large-scale linear
  programming. arXiv:2501.07018.
- Lu, H., Yang, J. (2023). cuPDLP.jl: A GPU implementation of restarted primal-dual hybrid
  gradient for linear programming in Julia. arXiv:2311.12180. Lu, H., Peng, Z., Yang, J.
  (2023). cuPDLP-C. arXiv:2312.14832. Lu, H., Yang, J. (2025). arXiv:2506.02174 (survey of
  GPU first-order LP). Cited as prior art only; lpbench runs nothing on GPU.
- Huangfu, Q., Hall, J. A. J. (2018). Parallelizing the dual revised simplex method.
  Mathematical Programming Computation 10(1), 119-142. (HiGHS simplex.)
- Schork, L., Gondzio, J. (2020). Implementation of an interior point method with basis
  preconditioning. Mathematical Programming Computation 12, 603-635. (IPX.)
- Gleixner, A., et al. (2021). MIPLIB 2017: data-driven compilation of the 6th mixed-integer
  programming library. Mathematical Programming Computation 13, 443-490.
- Koch, T., et al. (2011). MIPLIB 2010. Mathematical Programming Computation 3, 103-163.
  (Performance variability.)
- Lodi, A., Tramontani, A. (2013). Performance variability in mixed-integer programming.
  INFORMS TutORials in Operations Research, 1-12.
- Gay, D. M. (1985). Electronic mail distribution of linear programming test problems.
  Mathematical Programming Society COAL Newsletter 13, 10-12. (Netlib format.)
- Koch, T. (2004). The final NETLIB-LP results. Operations Research Letters 32(2), 138-142.
  doi:10.1016/S0167-6377(03)00094-4. Preprint ZIB-Report 03-05. (Exact Netlib optima.)
- Dolan, E. D., Moré, J. J. (2002). Benchmarking optimization software with performance
  profiles. Mathematical Programming 91(2), 201-213.
- Gould, N., Scott, J. (2016). A note on performance profiles for benchmarking software.
  ACM Transactions on Mathematical Software 43(2), article 15.
- Beiranvand, V., Hare, W., Lucet, Y. (2017). Best practices for comparing optimization
  algorithms. Optimization and Engineering 18(4), 815-848.
- Achterberg, T. (2007). Constraint Integer Programming. PhD thesis, TU Berlin. (Shifted
  geometric mean.)
- Mittelmann, H. D. Benchmarks for optimization software, https://plato.asu.edu/bench.html
  (SGM-based LP benchmarks; method notes).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian
  Journal of Statistics 6, 65-70.
- Davison, A. C., Hinkley, D. V. (1997). Bootstrap Methods and their Application. Cambridge.
- Efron, B. (1987). Better bootstrap confidence intervals. JASA 82, 171-185.

Prior-art code (none vendored): Mittelmann's benchmark scripts and logs; google-research
FirstOrderLp.jl (Apache-2.0; PDLP benchmark scripts and instance lists); perprof-py
(GPL-3.0, performance profiles; not used because of the licence, profiles are reimplemented);
benchopt (BSD-3; generic benchmark runner, no LP verification); the MIPLIB solution checker
(ZIB; MIP feasibility only).

## 12. Deviations

None yet.
