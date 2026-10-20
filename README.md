# duckdb_factor_sem

Confirmatory **and** Exploratory Factor Analysis in DuckDB SQL — the core of
JASP's *Factor* module, packaged as a standalone DuckDB C-API extension
(Rust + quack-rs).

Two estimators:

- **CFA** — confirmatory factor analysis, `X = Λ f + e` (f ~ N(0, I_q),
  e ~ N(0, diag(Ψ))) with a **pre-specified** loading pattern, fit by
  **maximum likelihood** (FIML) via **covariance EM with multi-start**.
  Returns loadings, uniquenesses, and the full fit-index suite
  (χ², df, AIC, BIC, RMSEA, CFI, TLI, log-likelihood).
- **EFA** — exploratory factor analysis on the **correlation** matrix,
  **no pre-specified structure** (all p×q loadings free). Factoring by
  **PCA**, **principal-axis (Thurstone)**, or **minres / OLS** (JASP's
  default); rotated by **varimax** (orthogonal) or **promax** (oblique,
  Kaiser-normalized — JASP's default oblique).
  Returns (rotated) loadings, factor correlations `Φ`, uniquenesses,
  communality, eigenvalues, proportion of variance, and SRMR.

This is the 6th (final) extension of the JASP-module → DuckDB series:
`read_jasp` · `bayes` · `meta` · `irt` · `spc` · **`factor_sem`**.

## Build

Linux build host (the `.so` → `.duckdb_extension` pipeline is WSL-only on
this machine; a Windows host yields a `.dll`). Offline cargo:

```bash
export CARGO_NET_OFFLINE=true
mkdir -p configure
echo "linux_amd64"  > configure/platform.txt
echo "v0.2.1"       > configure/extension_version.txt
DUCKDB_EXTENSION_NAME=factor_sem_ext \
DUCKDB_EXTENSION_MIN_DUCKDB_VERSION=v1.2.0 \
  cargo build --offline --release
python3 extension-ci-tools/scripts/append_extension_metadata.py \
  -l target/release/libfactor_sem_ext.so \
  -o target/release/factor_sem_ext.duckdb_extension \
  -n factor_sem_ext -dv v1.2.0 \
  -evf configure/extension_version.txt \
  -pf configure/platform.txt
```

Minimum DuckDB **v1.2.0**. Extension name `factor_sem_ext`.

## Functions

Both return a compact JSON `VARCHAR`. Invalid input (bad factor count,
non-numeric data, singular model, non-convergence) returns **NULL** — DuckDB
propagates it like any other NULL.

### `cfa_fit(data VARCHAR, factors VARCHAR) -> VARCHAR`

Fit on **raw data**. `data` is `'x1 x2 ... | y1 y2 ... | ...'` (rows
`|`-separated, values space-separated); `factors` is the number of factors
`q` as a string.

### `cfa_cov(cov VARCHAR, factors VARCHAR) -> VARCHAR`

Fit on a **covariance matrix** (faster, and the path the unit tests and e2e
verify against the oracle). `cov` is `'n p s00 s01 ... s(p-1)(p-1)'` — `n`
the effective sample size, then the upper triangle (row-major) of the
symmetric covariance `S`.

### `efa(data VARCHAR, factors VARCHAR, method VARCHAR, rotate VARCHAR) -> VARCHAR`

**Exploratory factor analysis** on raw data. `data` / `factors` as above;
`method` is `'pca'`, `'pa'` (principal-axis), or `'minres'` (minimum
residual / OLS — JASP's default); `rotate` is `'none'`, `'varimax'`
(orthogonal), or `'promax'` (oblique, Kaiser-normalized — JASP's default
oblique). The correlation matrix is computed from `data`; loadings are
estimated, optionally rotated, and sign-fixed so each factor column's
largest-|loading| is positive (deterministic).

### JSON output

CFA:
```json
{
  "p": 4, "q": 1, "n": 400, "df": 3,
  "lambda": [[-0.887], [0.600], [0.440], [0.420]],   // p x q loadings
  "psi": [0.447, 0.271, 0.244, 0.253],                // p uniquenesses
  "ll_model": -1779.56, "ll_saturated": -1779.18,
  "chi2": 0.767, "aic": -0.47, "bic": 0.36,
  "rmsea": 0.003, "cfi": 1.0, "tli": 1.0,
  "iterations": 44, "converged": true
}
```

EFA:
```json
{
  "meta": {"p": 6, "q": 2, "n": 40},
  "method": "pca", "rotate": "varimax",
  "eigenvalues": [2.75, 1.98, 0.55, ...],             // all p, descending
  "proportion_var": [0.33, 0.24, ...],                // eigenvalue / sum
  "lambda": [[...], ...],                             // p x q unrotated loadings
  "lambda_rot": [[...], ...],                         // p x q rotated loadings (sign-fixed)
  "phi": [[1,0],[0,1]],                               // q x q factor correlation (I for orthogonal)
  "uniquenesses": [0.45, ...],                        // p  (1 - communality)
  "communality": [0.55, ...],                         // p  (sum of squared loadings)
  "srml": 0.05,                                       // SRMR over upper triangle
  "rotate_iters": 1
}
```

CFA loadings are **identified up to sign** (the ML solution is invariant to
`f → −f`); comparisons use `|λ|`. EFA rotated loadings are sign-fixed by the
varimax sign rule, so they compare directly; unrotated ("none") loadings are
sign-arbitrary per factor column.

## Method

**CFA.** Maximum-likelihood factor analysis is **non-concave** — it has many
local maxima, so the optimizer's result depends on the starting point. We
address this the standard way: **multi-start** EM (9 starts — PCA-largest,
PCA-smallest, diagonal, and 6 deterministic pseudo-random loadings — keep the
highest-likelihood solution).

The M-step is the FIML covariance EM: with `Σ = ΛΛᵀ + Ψ`,
`Q = Σ⁻¹`, `B = ΛᵀQΛ + I − ΛᵀQ S Q Λ`… (full derivation in
`src/math.rs`). Verified against an independent **scipy L-BFGS-B** reference
(`test/oracle_factor.py`) and a pure-numpy covariance-EM cross-check.

**EFA.** On the sample **correlation** matrix `R`. `pca` takes the top-q
eigenvectors scaled by `√eigenvalue`; `pa` (principal-axis, Thurstone)
iterates the uniqueness diagonal to a fixed point (`R_r = R − diag(1−h)`).
`minres` (minimum residual / OLS, JASP's default) optimizes the uniqueness
diagonal `ψ` to minimize the **strictly-lower-triangular** squared OLS
residuals `Σ_{i>j}(Rᵢⱼ − LᵢLⱼᵀ)²`, by **two-loop L-BFGS-B** (box
`[0.005, 0.99]`) with the analytic gradient — the same solver R's
`optim(..., method="L-BFGS-B")` uses; verified to a 1e-9 objective match
against scipy L-BFGS-B plus an independent finite-difference gradient
check.
**Varimax** rotation (Kaiser/Guttman orthogonal criterion) is a fixed-point
iteration `Lᵀ = (L⊙(LᵀL)) (LᵀL)⁻¹` that increases `Σ_j (Σ_i lᵢⱼ²)²`,
converged when the change drops below `1e-8`, then sign-fixed.
**Promax** (Kaiser-normalized, oblique — JASP's default oblique) is
`kaiser(loadings, rotate="Promax")`: scale columns to unit norm, varimax the
scaled matrix, regress `W⊙|W|³` on `W` for the oblique rotation `U`, then
`Φ = U⁻¹(U⁻¹)ᵀ`.
All EFA methods/rotations are verified against an independent numpy oracle
(`test/oracle_efa.py`).

The cyclic **Jacobi** eigen-decomposition in `src/math.rs` was rewritten to
the correct Givens form (`A ← GᵀAG`, `V ← VG`, Numerical-Recipes angle
`t = sign(τ)/(|τ|+√(1+τ²))`, `τ = (a_jj−a_ii)/(2 a_ij)`) and re-verified
against `numpy.linalg.eigh` (eigenvalues and eigenvectors to ~1e-15). The
original had a wrong angle approximation that produced incorrect eigenvalues
(hidden in CFA, where eigenvalues only seed EM initialization and are masked
by multi-start; exposed once EFA started emitting eigenvalues directly).

## v0.2.1 scope

- **CFA** (unchanged): uncorrelated standard factors (Φ = I), arbitrary
  `q < p`, ML estimation, all fit indices. `q = 1` numerically exact against
  the scipy oracle; `q ≥ 2` best-effort (local-maxima sensitivity).
- **EFA** (extended in v0.2.1): correlation-basis, manual factor count,
  **PCA / principal-axis / minres (OLS)** factoring, **varimax (orthogonal)
  or promax (oblique)** or no rotation. Output: rotated loadings, factor
  correlations `Φ`, uniquenesses, communality, eigenvalues, proportion of
  variance, SRMR. `minres` + `promax` are JASP's defaults.

**Not yet** (roadmap → v0.2.2+):
- EFA: **quartimax / oblimin** (oblique) rotation, **WLS / ML** factoring,
  **parallel analysis / eigenvalue** factor-count rules, **polychoric /
  tetrachoric** input.
- **Full SEM** (path model, exogenous / endogenous predictors, correlated
  factors, covariates)
- **Measurement invariance / multiple-group CFA**
- **Mediation / moderated mediation** (indirect effects, bootstrap CIs)
- **PLS-SEM** (partial-least-squares path modeling)
- **GGM** (Gaussian graphical model — concentration matrix, MCMC posterior)
- Correlated factors (Φ ≠ I), missing-data FIML on raw data, bootstrapped
  parameter CIs.

## Tests

- **Unit** (`cargo test --lib`): 6 tests. `single_factor_recovers_loadings`
  and `over_identified_two_factor_matches_oracle` verify the CFA FIML EM
  against the scipy L-BFGS-B oracle. `efa_pca_varimax_matches_oracle` runs all
  8 EFA cases (pca/pa/minres × varimax/promax/none) and checks eigenvalues,
  rotated loadings (sign-normalized), uniquenesses, `Φ`, and SRMR against the
  numpy oracle (orthogonal cases `1e-3`; minres/oblique cases looser, since
  the OLS-uniquenesses surface is flat and the oblique rotation amplifies the
  1e-6 solver-solution difference).
- **e2e** (`bash test/run_factor.sh`): 50 self-checks against the built
  `.duckdb_extension` in the real DuckDB CLI (`-unsigned`). Part 1 fits the
  oracle covariance via `cfa_cov`; Part 2 smoke-tests `cfa_fit` on raw data;
  Part 3 runs `efa` on fixed datasets — pca+varimax (p=6 q=2), pa+none (p=4
  q=1), **minres+varimax (p=6 q=2)**, and **minres+promax (p=8 q=2, oblique:
  checks `Φ` ≠ I)** — checking eigenvalues / SRMR / rotated loadings / `Φ`
  against the numpy oracle. Also covers the NULL error paths (bad method /
  rotation).

Regenerate references: `python test/oracle_factor.py` → `test/factor_ref.json`,
`python test/oracle_efa.py` → `test/efa_ref.json`, then
`python test/gen_e2e_factor.py` → `test/test_factor.sql`.

## Files

- `src/math.rs` — pure Rust stats core: Jacobi eigen-decomp (Givens form),
  Cholesky logdet, Gauss-Jordan inverse, FIML covariance EM + multi-start,
  fit indices. No external deps beyond `libm`.
- `src/efa.rs` — EFA core: sample covariance → correlation, PCA / PA / minres
  (two-loop L-BFGS-B) loadings, Guttman varimax, Kaiser-normalized promax,
  factor correlation `Φ`, SRMR.
- `src/lib.rs` — DuckDB C-API glue (quack-rs): `cfa_fit`, `cfa_cov`, `efa`,
  JSON builders, unit tests.
- `test/oracle_factor.py` — scipy/numpy CFA oracle → `test/factor_ref.json`.
- `test/oracle_efa.py` — numpy EFA oracle (PCA/PA/minres + varimax/promax,
  minres cross-checked by analytic vs finite-difference L-BFGS-B) →
  `test/efa_ref.json`.
- `test/ref_{Promax,kaiser,smc,factor.minres}.R` — psych / JASP reference
  algorithms (provenance for the minres + promax ports).
- `test/gen_raw.py` — reproducible raw dataset for the CFA e2e smoke test.
- `test/gen_e2e_factor.py` — emits `test/test_factor.sql` (CFA + EFA).
- `test/run_factor.sh` — e2e runner (builds if needed, asserts `n_fail = 0`).
