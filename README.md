# duckdb_factor_sem

Confirmatory Factor Analysis (CFA) in DuckDB SQL — the core of JASP's *Factor*
module, packaged as a standalone DuckDB C-API extension (Rust + quack-rs).

Fit a factor model `X = Λ f + e` (f ~ N(0, I_q), e ~ N(0, diag(Ψ))) by
**maximum likelihood** (FIML formulation) via **covariance EM with
multi-start**, and get loadings, uniquenesses, and the full suite of fit
indices (χ², df, AIC, BIC, RMSEA, CFI, TLI, log-likelihood).

This is the 6th (final) extension of the JASP-module → DuckDB series:
`read_jasp` · `bayes` · `meta` · `irt` · `spc` · **`factor_sem`**.

## Build

Linux build host (the `.so` → `.duckdb_extension` pipeline is WSL-only on
this machine; a Windows host yields a `.dll`). Offline cargo:

```bash
export CARGO_NET_OFFLINE=true
mkdir -p configure
echo "linux_amd64"  > configure/platform.txt
echo "v0.1.0"       > configure/extension_version.txt
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

### JSON output

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

Loadings are **identified up to sign** (the ML solution is invariant to
`f → −f`); comparisons use `|λ|`.

## Method

Maximum-likelihood factor analysis is **non-concave** — it has many local
maxima, so the optimizer's result depends on the starting point. We address
this the standard way: **multi-start** EM (9 starts — PCA-largest,
PCA-smallest, diagonal, and 6 deterministic pseudo-random loadings — keep the
highest-likelihood solution).

The M-step is the FIML covariance EM: with `Σ = ΛΛᵀ + Ψ`,
`Q = Σ⁻¹`, `B = ΛᵀQΛ + I − ΛᵀQ S Q Λ`… (full derivation in
`src/math.rs`). Verified against an independent **scipy L-BFGS-B** reference
(`test/oracle_factor.py`) and a pure-numpy covariance-EM cross-check.

## v0.1.0 scope

Closed, verifiable slice: **CFA with uncorrelated standard factors (Φ = I)**,
arbitrary `q < p`, ML estimation, all fit indices. `q = 1` is verified
numerically exact against the scipy oracle; `q ≥ 2` is best-effort
(local-maxima sensitivity — the multi-start mitigates but does not
eliminate it; ML multi-factor is a known-hard global optimization).

**Not in v0.1.0** (roadmap):
- **EFA** (exploratory, promax/oblimin rotation, marker / geominic)
- **Full SEM** (path model, exogenous / endogenous predictors,
  correlated factors, covariates)
- **Measurement invariance / multiple-group CFA**
- **Mediation / moderated mediation** (indirect effects, bootstrap CIs)
- **PLS-SEM** (partial-least-squares path modeling)
- **GGM** (Gaussian graphical model — concentration matrix, MCMC posterior)
- Correlated factors (Φ ≠ I), missing-data FIML on raw data, bootstrapped
  parameter CIs.

## Tests

- **Unit** (`cargo test --lib`): 5 tests. `single_factor_recovers_loadings`
  and `over_identified_two_factor_matches_oracle` verify the FIML EM against
  the scipy L-BFGS-B oracle (LL / λ / χ² within tolerance).
- **e2e** (`bash test/run_factor.sh`): 27 self-checks against the built
  `.duckdb_extension` in the real DuckDB CLI (`-unsigned`). Part 1 fits the
  oracle covariance via `cfa_cov` and compares `|λ|`, Ψ, χ², CFI, TLI, RMSEA,
  LL to the scipy reference (verifies the binary + C-API glue + JSON + cov
  parse). Part 2 smoke-tests `cfa_fit` on raw data and the NULL error paths.

Regenerate e2e references if you change the model:
`python test/oracle_factor.py` (→ `test/factor_ref.json`) then
`python test/gen_e2e_sql.py` (→ `test/test_factor.sql`).

## Files

- `src/math.rs` — pure Rust stats core: Jacobi eigen-decomp, Cholesky
  logdet, Gauss-Jordan inverse, FIML covariance EM + multi-start, fit
  indices. No external deps beyond `libm`.
- `src/lib.rs` — DuckDB C-API glue (quack-rs): `cfa_fit`, `cfa_cov`, JSON
  builder, unit tests.
- `test/oracle_factor.py` — scipy/numpy oracle → `test/factor_ref.json`.
- `test/gen_raw.py` — reproducible raw dataset for the e2e smoke test.
- `test/gen_e2e_sql.py` — emits `test/test_factor.sql` from the reference.
- `test/run_factor.sh` — e2e runner (builds if needed, asserts `n_fail = 0`).
